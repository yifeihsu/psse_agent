#!/bin/bash
#SBATCH --job-name=dagger_e2b_prelim
#SBATCH --output=/scratch/yx3882/psse_agent/artifacts/logs/dagger_e2b_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/artifacts/logs/dagger_e2b_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|h100|rtx6000|l40s"
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu

# Fast, explicitly NON-RELEASE DAgger debugging pipeline.  It trains a short
# BC0 adapter and then starts a distinct continuation from that adapter on the
# preliminary mixed BC0+D1 view.  It never changes the production 31B study
# launcher, study manifest, source gates, or checkpoint receipts.

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-E2B-it}
readonly MODEL_REVISION="f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae"

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/psse_agent}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
DATASET_RECEIPT=${DATASET_RECEIPT:-data/preliminary_dagger/preliminary.dataset_receipt.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/yx3882/psse_agent/outputs/preliminary_dagger_e2b}
PIPELINE_STAGE=${PIPELINE_STAGE:-all}
EXPECTED_ACCELERATOR_CLASS=${EXPECTED_ACCELERATOR_CLASS:-auto}
ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD:-0}
REPORT_TO=${REPORT_TO:-none}
TRAIN_SEED=${TRAIN_SEED:-3407}

# Hard step/row ceilings keep this a preliminary run even when overridden.
BC0_MAX_STEPS=${BC0_MAX_STEPS:-64}
DAGGER_MAX_STEPS=${DAGGER_MAX_STEPS:-96}
BC0_MAX_TRAIN_ROWS=${BC0_MAX_TRAIN_ROWS:-256}
D1_MAX_VALID_ROWS=${D1_MAX_VALID_ROWS:-128}
DAGGER_MAX_TRAIN_ROWS=${DAGGER_MAX_TRAIN_ROWS:-512}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-}
LEARNING_RATE=${LEARNING_RATE:-0.00005}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
SAVE_STEPS=${SAVE_STEPS:-8}
EVAL_STEPS=${EVAL_STEPS:-8}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
PREEMPTION_EXIT_CODE=99

case "$PIPELINE_STAGE" in
    all|bc0|dagger) ;;
    *)
        echo "ERROR: PIPELINE_STAGE must be all, bc0, or dagger; got '$PIPELINE_STAGE'." >&2
        exit 2
        ;;
esac
case "$EXPECTED_ACCELERATOR_CLASS" in
    auto|h100|h200|rtx6000|l40s) ;;
    *)
        echo "ERROR: EXPECTED_ACCELERATOR_CLASS must be auto, h100, h200, rtx6000, or l40s." >&2
        exit 2
        ;;
esac
case "$ALLOW_DOWNLOAD" in
    0|1) ;;
    *)
        echo "ERROR: ALLOW_DOWNLOAD must be 0 or 1." >&2
        exit 2
        ;;
esac
case "$REPORT_TO" in
    none|wandb) ;;
    *)
        echo "ERROR: REPORT_TO must be none or wandb." >&2
        exit 2
        ;;
esac

# Offline compute nodes may load the exact pinned Hub snapshot by absolute
# path.  Keep the canonical repo id as the default and reject every other
# override before allocating optimizer state.
if [[ "$MODEL_NAME" != "unsloth/gemma-4-E2B-it" ]]; then
    if [[ "$MODEL_NAME" != /* \
        || "$(basename -- "$MODEL_NAME")" != "$MODEL_REVISION" \
        || ! -f "$MODEL_NAME/config.json" ]]; then
        echo "ERROR: MODEL_NAME must be the pinned repo id or its exact local snapshot." >&2
        exit 2
    fi
    MODEL_NAME=$(realpath -e -- "$MODEL_NAME")
fi
readonly MODEL_NAME

bounded_uint() {
    local name=$1
    local value=$2
    local minimum=$3
    local maximum=$4
    if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]] || (( 10#$value < minimum || 10#$value > maximum )); then
        echo "ERROR: $name must be a canonical integer in [$minimum, $maximum]; got '$value'." >&2
        exit 2
    fi
}
bounded_uint TRAIN_SEED "$TRAIN_SEED" 0 4294967295
bounded_uint BC0_MAX_STEPS "$BC0_MAX_STEPS" 1 64
bounded_uint DAGGER_MAX_STEPS "$DAGGER_MAX_STEPS" 1 128
bounded_uint BC0_MAX_TRAIN_ROWS "$BC0_MAX_TRAIN_ROWS" 1 1024
bounded_uint D1_MAX_VALID_ROWS "$D1_MAX_VALID_ROWS" 1 512
bounded_uint DAGGER_MAX_TRAIN_ROWS "$DAGGER_MAX_TRAIN_ROWS" 1 2048
bounded_uint MAX_SEQ_LENGTH "$MAX_SEQ_LENGTH" 1024 8192
bounded_uint LORA_R "$LORA_R" 1 64
bounded_uint LORA_ALPHA "$LORA_ALPHA" 1 128
bounded_uint SAVE_STEPS "$SAVE_STEPS" 1 64
bounded_uint EVAL_STEPS "$EVAL_STEPS" 1 64
bounded_uint SAVE_TOTAL_LIMIT "$SAVE_TOTAL_LIMIT" 1 8
bounded_uint DATALOADER_NUM_WORKERS "$DATALOADER_NUM_WORKERS" 0 16

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

cd "$REPO_ROOT"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: preliminary SFT interpreter not found: $PYTHON" >&2
    exit 2
fi

# This validator accepts only the dedicated non-release v1 contract, resolves
# every filename beside the receipt (never from its output_dir field), rehashes
# every JSONL, and independently checks physical-root separation.
"$PYTHON" -m psse_env.dagger.preliminary_receipt dataset \
    --receipt "$DATASET_RECEIPT"
DATASET_RECEIPT=$(realpath -e -- "$DATASET_RECEIPT")
DATASET_DIR=$(dirname -- "$DATASET_RECEIPT")
BC0_TRAIN_FILE=$DATASET_DIR/preliminary.bc0_train.jsonl
BC0_VALIDATION_FILE=$DATASET_DIR/preliminary.bc0_validation.jsonl
D1_TRAIN_FILE=$DATASET_DIR/preliminary.d1_train.jsonl
D1_VALIDATION_FILE=$DATASET_DIR/preliminary.d1_validation.jsonl
D1_TEST_FILE=$DATASET_DIR/preliminary.d1_test.jsonl
D1_EVAL_COMBINED_FILE=$DATASET_DIR/preliminary.d1_eval_combined.jsonl
MIXED_TRAIN_FILE=$DATASET_DIR/preliminary.mixed_train.jsonl

OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT")
DATASET_DIR=$(realpath -e -- "$DATASET_DIR")
if [[ "$OUTPUT_ROOT" == "$DATASET_DIR" || "$OUTPUT_ROOT" == "$DATASET_DIR"/* || "$DATASET_DIR" == "$OUTPUT_ROOT"/* ]]; then
    echo "ERROR: OUTPUT_ROOT and the immutable preliminary dataset directory must not overlap." >&2
    exit 2
fi
BC0_OUTPUT=$OUTPUT_ROOT/bc0
DAGGER_OUTPUT=$OUTPUT_ROOT/dagger
PINNED_INIT_ADAPTER=$OUTPUT_ROOT/pinned_bc0_init_adapter
mkdir -p "$BC0_OUTPUT" "$DAGGER_OUTPUT" artifacts/logs
mkdir -p /scratch/yx3882/.cache/huggingface /scratch/yx3882/.cache/torch

export HF_HOME=${HF_HOME:-/scratch/yx3882/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yx3882/.cache/torch}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
if [[ "$ALLOW_DOWNLOAD" == "0" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
fi

GPU_INVENTORY=$(nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader)
echo "===== Preliminary DAgger E2B (NON-RELEASE) ====="
echo "job:        ${SLURM_JOB_ID:-interactive}"
echo "host:       $(hostname)"
echo "pipeline:   $PIPELINE_STAGE"
echo "model:      $MODEL_NAME"
echo "revision:   $MODEL_REVISION"
echo "dataset:    $DATASET_RECEIPT"
echo "D1 train:   $D1_TRAIN_FILE (bound into mixed training view)"
echo "D1 val:     $D1_VALIDATION_FILE"
echo "D1 test:    $D1_TEST_FILE (held out; never optimizer-visible)"
echo "D1 eval:    $D1_EVAL_COMBINED_FILE (reserved for post-training evaluation)"
echo "output:     $OUTPUT_ROOT"
echo "GPU:        $GPU_INVENTORY"
echo "limits:     BC0=$BC0_MAX_STEPS steps, DAgger=$DAGGER_MAX_STEPS steps"

# H100/H200/RTX Pro decisions delegate to release_hardware.  The separate
# preliminary gate additionally accepts only exact NVIDIA L40S >=45,000 MiB;
# this does not widen any production hardware contract.
HARDWARE_ATTESTATION_JSON=$("$PYTHON" -m psse_env.sft.preliminary_hardware \
    --require-class "$EXPECTED_ACCELERATOR_CLASS")
ACTUAL_ACCELERATOR_CLASS=$("$PYTHON" -c \
    'import json,sys; print(json.load(sys.stdin)["accelerator_class"])' \
    <<<"$HARDWARE_ATTESTATION_JSON")
if [[ "$ACTUAL_ACCELERATOR_CLASS" == "l40s" ]]; then
    PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
else
    PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
fi
bounded_uint PER_DEVICE_TRAIN_BATCH_SIZE "$PER_DEVICE_TRAIN_BATCH_SIZE" 1 32
bounded_uint GRADIENT_ACCUMULATION_STEPS "$GRADIENT_ACCUMULATION_STEPS" 1 16
echo "Preliminary runtime hardware attestation: $HARDWARE_ATTESTATION_JSON"
echo "batch profile: class=$ACTUAL_ACCELERATOR_CLASS bs=$PER_DEVICE_TRAIN_BATCH_SIZE ga=$GRADIENT_ACCUMULATION_STEPS effective_batch=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"

COMMON_ARGS=(
    --model-name "$MODEL_NAME"
    --model-revision "$MODEL_REVISION"
    --require-pinned-model-revision
    --max-seq-length "$MAX_SEQ_LENGTH"
    --dataset-num-proc 1
    --load-in-16bit
    --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA"
    --lora-target-scope language_model
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$LEARNING_RATE"
    --warmup-steps 4
    --logging-steps 2
    --save-steps "$SAVE_STEPS"
    --eval-steps "$EVAL_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --seed "$TRAIN_SEED"
    --preserve-system-text
    --no-phase-gated-prompt
    --sanity-check-samples 0
    --include-tool-schemas
    --inject-empty-thought-channel
    --fail-on-prompt-truncation
    --repeat-first-tool-call 1
    --repeat-later-tool-call 1
    --repeat-final 1
    --report-to "$REPORT_TO"
)

TRAIN_PID=""
forward_signal() {
    local signal_name=$1
    if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[signal] forwarding $signal_name to trainer PID $TRAIN_PID"
        kill -s "$signal_name" "$TRAIN_PID" || true
    fi
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

run_trainer() {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    "$@" &
    TRAIN_PID=$!
    local status
    while true; do
        wait "$TRAIN_PID"
        status=$?
        # A trapped Slurm signal interrupts Bash's wait before the trainer has
        # finished its checkpoint callback. Keep waiting for the real RC.
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            continue
        fi
        break
    done
    TRAIN_PID=""
    return "$status"
}

handle_trainer_status() {
    local status=$1
    if [[ "$status" -eq 0 ]]; then
        return 0
    fi
    if [[ "$status" -eq "$PREEMPTION_EXIT_CODE" ]]; then
        echo "[requeue] trainer saved state after a preemption signal"
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            scontrol requeue "$SLURM_JOB_ID"
            exit 0
        fi
    fi
    exit "$status"
}

stage_complete() {
    local stage=$1
    local train_file=$2
    local validation_file=$3
    local output_dir=$4
    local receipt=$output_dir/preliminary_stage_receipt.json
    if [[ ! -e "$receipt" && ! -L "$receipt" ]]; then
        return 1
    fi
    if ! "$PYTHON" -m psse_env.dagger.preliminary_receipt stage-check \
        --stage "$stage" \
        --dataset-receipt "$DATASET_RECEIPT" \
        --train-file "$train_file" \
        --validation-file "$validation_file" \
        --output-dir "$output_dir" \
        --hardware-attestation "$output_dir/preliminary_hardware_attestation.json" \
        --stage-plan "$output_dir/preliminary_stage_plan.json"; then
        echo "ERROR: existing $stage stage receipt failed validation." >&2
        exit 2
    fi
}

publish_stage_receipt() {
    local stage=$1
    local train_file=$2
    local validation_file=$3
    local output_dir=$4
    "$PYTHON" -m psse_env.dagger.preliminary_receipt stage-write \
        --stage "$stage" \
        --dataset-receipt "$DATASET_RECEIPT" \
        --train-file "$train_file" \
        --validation-file "$validation_file" \
        --output-dir "$output_dir" \
        --hardware-attestation "$output_dir/preliminary_hardware_attestation.json" \
        --stage-plan "$output_dir/preliminary_stage_plan.json"
}

run_tool_generation_gate() {
    local output_dir=$1
    "$PYTHON" -m psse_env.dagger.preliminary_tool_gate \
        --stage-plan "$output_dir/preliminary_stage_plan.json"
}

ensure_stage_plan() {
    local stage=$1
    local train_file=$2
    local validation_file=$3
    local output_dir=$4
    local max_train_rows=$5
    local max_steps=$6
    local initial_adapter=${7:-}
    local -a plan_args=(
        stage-plan
        --stage "$stage"
        --dataset-receipt "$DATASET_RECEIPT"
        --train-file "$train_file"
        --validation-file "$validation_file"
        --output-dir "$output_dir"
        --repo-root "$REPO_ROOT"
        --training-seed "$TRAIN_SEED"
        --max-train-rows "$max_train_rows"
        --max-valid-rows "$D1_MAX_VALID_ROWS"
        --max-steps "$max_steps"
        --max-seq-length "$MAX_SEQ_LENGTH"
        --batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
        --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
        --learning-rate "$LEARNING_RATE"
        --lora-r "$LORA_R"
        --lora-alpha "$LORA_ALPHA"
        --save-steps "$SAVE_STEPS"
        --eval-steps "$EVAL_STEPS"
        --save-total-limit "$SAVE_TOTAL_LIMIT"
        --dataloader-workers "$DATALOADER_NUM_WORKERS"
        --report-to "$REPORT_TO"
    )
    if [[ -n "$initial_adapter" ]]; then
        plan_args+=(--initial-adapter "$initial_adapter")
    fi
    "$PYTHON" -m psse_env.dagger.preliminary_receipt "${plan_args[@]}"
}

validate_resume_checkpoint() {
    local output_dir=$1
    local checkpoint=$2
    "$PYTHON" -m psse_env.dagger.preliminary_receipt resume-check \
        --stage-plan "$output_dir/preliminary_stage_plan.json" \
        --checkpoint "$checkpoint"
}

attest_stage_hardware() {
    local output_dir=$1
    "$PYTHON" -m psse_env.sft.preliminary_hardware \
        --require-class "$EXPECTED_ACCELERATOR_CLASS" \
        --output "$output_dir/preliminary_hardware_attestation.json"
}

latest_checkpoint() {
    local output_dir=$1
    "$PYTHON" - "$output_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = []
for child in root.glob("checkpoint-*"):
    suffix = child.name.removeprefix("checkpoint-")
    if child.is_dir() and not child.is_symlink() and suffix.isdigit():
        candidates.append((int(suffix), child.resolve()))
if candidates:
    print(max(candidates)[1])
PY
}

run_bc0_stage() {
    ensure_stage_plan \
        bc0 "$BC0_TRAIN_FILE" "$D1_VALIDATION_FILE" "$BC0_OUTPUT" \
        "$BC0_MAX_TRAIN_ROWS" "$BC0_MAX_STEPS"
    if stage_complete bc0 "$BC0_TRAIN_FILE" "$D1_VALIDATION_FILE" "$BC0_OUTPUT"; then
        echo "BC0 stage receipt is valid; skipping completed BC0 stage."
        return 0
    fi
    attest_stage_hardware "$BC0_OUTPUT"
    local resume_checkpoint
    resume_checkpoint=$(latest_checkpoint "$BC0_OUTPUT")
    local -a resume_args=()
    if [[ -n "$resume_checkpoint" ]]; then
        validate_resume_checkpoint "$BC0_OUTPUT" "$resume_checkpoint"
        resume_args+=(--resume-from-checkpoint "$resume_checkpoint")
    fi
    if [[ -e "$BC0_OUTPUT/lora" || -L "$BC0_OUTPUT/lora" ]]; then
        if [[ -z "$resume_checkpoint" ]]; then
            echo "ERROR: BC0 has an unreceipted lora tree and no resumable Trainer checkpoint." >&2
            exit 2
        fi
    fi
    set +e
    run_trainer "$PYTHON" gpt_oss_power_sft_revised_v3.py \
        --train-file "$BC0_TRAIN_FILE" \
        --valid-file "$D1_VALIDATION_FILE" \
        --output-dir "$BC0_OUTPUT" \
        --max-train-rows "$BC0_MAX_TRAIN_ROWS" \
        --max-valid-rows "$D1_MAX_VALID_ROWS" \
        --max-steps "$BC0_MAX_STEPS" \
        --num-train-epochs 1 \
        "${resume_args[@]}" \
        --run-name "prelim-e2b-bc0-seed$TRAIN_SEED" \
        "${COMMON_ARGS[@]}"
    local status=$?
    set -e
    handle_trainer_status "$status"
    run_tool_generation_gate "$BC0_OUTPUT"
    publish_stage_receipt bc0 "$BC0_TRAIN_FILE" "$D1_VALIDATION_FILE" "$BC0_OUTPUT"
}

run_dagger_stage() {
    if [[ ! -e "$BC0_OUTPUT/preliminary_stage_receipt.json" && \
          ! -L "$BC0_OUTPUT/preliminary_stage_receipt.json" ]]; then
        echo "ERROR: DAgger continuation requires a local BC0 stage receipt." >&2
        exit 2
    fi
    stage_complete bc0 "$BC0_TRAIN_FILE" "$D1_VALIDATION_FILE" "$BC0_OUTPUT"
    local -a pinned_init_args=(
        --source-adapter "$BC0_OUTPUT/lora"
        --destination "$PINNED_INIT_ADAPTER"
    )
    if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
        pinned_init_args+=(--allow-download)
    fi
    "$PYTHON" -m psse_env.sft.preliminary_adapter "${pinned_init_args[@]}"
    ensure_stage_plan \
        dagger "$MIXED_TRAIN_FILE" "$D1_VALIDATION_FILE" "$DAGGER_OUTPUT" \
        "$DAGGER_MAX_TRAIN_ROWS" "$DAGGER_MAX_STEPS" "$PINNED_INIT_ADAPTER"
    if stage_complete dagger "$MIXED_TRAIN_FILE" "$D1_VALIDATION_FILE" "$DAGGER_OUTPUT"; then
        echo "DAgger continuation receipt is valid; skipping completed DAgger stage."
        return 0
    fi
    attest_stage_hardware "$DAGGER_OUTPUT"
    local resume_checkpoint
    resume_checkpoint=$(latest_checkpoint "$DAGGER_OUTPUT")
    local -a resume_args=()
    if [[ -n "$resume_checkpoint" ]]; then
        validate_resume_checkpoint "$DAGGER_OUTPUT" "$resume_checkpoint"
        resume_args+=(--resume-from-checkpoint "$resume_checkpoint")
    fi
    if [[ -e "$DAGGER_OUTPUT/lora" || -L "$DAGGER_OUTPUT/lora" ]]; then
        if [[ -z "$resume_checkpoint" ]]; then
            echo "ERROR: DAgger has an unreceipted lora tree and no resumable Trainer checkpoint." >&2
            exit 2
        fi
    fi
    set +e
    run_trainer "$PYTHON" gpt_oss_power_sft_revised_v3.py \
        --train-file "$MIXED_TRAIN_FILE" \
        --valid-file "$D1_VALIDATION_FILE" \
        --init-adapter "$PINNED_INIT_ADAPTER" \
        --output-dir "$DAGGER_OUTPUT" \
        --max-train-rows "$DAGGER_MAX_TRAIN_ROWS" \
        --max-valid-rows "$D1_MAX_VALID_ROWS" \
        --max-steps "$DAGGER_MAX_STEPS" \
        --num-train-epochs 1 \
        "${resume_args[@]}" \
        --run-name "prelim-e2b-dagger-seed$TRAIN_SEED" \
        "${COMMON_ARGS[@]}"
    local status=$?
    set -e
    handle_trainer_status "$status"
    run_tool_generation_gate "$DAGGER_OUTPUT"
    publish_stage_receipt dagger "$MIXED_TRAIN_FILE" "$D1_VALIDATION_FILE" "$DAGGER_OUTPUT"
}

case "$PIPELINE_STAGE" in
    bc0)
        run_bc0_stage
        ;;
    dagger)
        run_dagger_stage
        ;;
    all)
        run_bc0_stage
        run_dagger_stage
        ;;
esac

echo "Preliminary E2B pipeline stage '$PIPELINE_STAGE' completed."
echo "NON-RELEASE receipts:"
find "$OUTPUT_ROOT" -maxdepth 2 -name preliminary_stage_receipt.json -print
