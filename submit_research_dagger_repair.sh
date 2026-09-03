#!/usr/bin/env bash
#SBATCH --job-name=dagger-repair-e2b
#SBATCH --output=/scratch/yx3882/research_dagger_repair_20260824/repair_%j.out
#SBATCH --error=/scratch/yx3882/research_dagger_repair_20260824/repair_%j.err
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=rtx6000
#SBATCH --constraint=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"

# One research-only repair experiment after the first two DAgger continuations
# collapsed to measurement-context actions.  This launcher:
#
#   1. rebuilds the deterministic 512-row decision-balanced curriculum;
#   2. continues directly from BC0 for 32 optimizer steps;
#   3. greedily scores checkpoints 8/16/24/32 on the untouched 27-row trace
#      validation set;
#   4. selects by required-tool exact coverage, then exact action count; and
#   5. runs the 14-episode closed loop only if the selected checkpoint is exact
#      on all five routing/recovery tools, clears quality floors, and improves
#      over the BC0 greedy baseline.
#
# It never recollects DAgger data, retrains BC0/Base, runs the strict production
# pipeline, or submits additional seeds.  A failed preflight is a useful result:
# the job exits cleanly before closed-loop inference and preserves every report.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/research_dagger_src_20260823}
RUN_ROOT=${RUN_ROOT:-/scratch/yx3882/research_dagger_repair_20260824}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}

D0_FILE=${D0_FILE:-/scratch/yx3882/dagger_release_a5a7574_20260823/round0_aggregate_release/aggregate.train_view.jsonl}
NATURAL_ROOT=${NATURAL_ROOT:-/scratch/yx3882/prelim_e2b_baa7e35_attested_20260821}
NATURAL_TRAIN=${NATURAL_TRAIN:-$NATURAL_ROOT/preliminary.d1_train.jsonl}
NATURAL_VALIDATION=${NATURAL_VALIDATION:-$NATURAL_ROOT/preliminary.d1_validation.jsonl}
PROBE_DONOR=${PROBE_DONOR:-/scratch/yx3882/research_dagger_update_20260823/balanced_train.jsonl}
PROBE_AUDIT=${PROBE_AUDIT:-/scratch/yx3882/dagger_release_a5a7574_20260823/dagger1_a5a7574/recovery_probes.jsonl}

TRACE_VALIDATION=${TRACE_VALIDATION:-/scratch/yx3882/research_dagger_trace_20260823/trace_validation.jsonl}
HELDOUT_SUITE=${HELDOUT_SUITE:-/scratch/yx3882/research_dagger_trace_20260823/heldout_suite_14_root_closure.json}
BC0_ADAPTER=${BC0_ADAPTER:-/scratch/yx3882/prelim_e2b_fullnatural_2b32029_20260822/output/bc0/lora}
BC0_EVAL=${BC0_EVAL:-/scratch/yx3882/research_dagger_trace_20260823/bc0_eval_14.json}

REPAIR_TRAIN=$RUN_ROOT/repair_train_512.jsonl
REPAIR_REPORT=$RUN_ROOT/repair_train_512_report.json
TRAIN_OUTPUT=$RUN_ROOT/training_512_step32
NEW_ADAPTER=$TRAIN_OUTPUT/lora
RECIPE_CURRENT=$RUN_ROOT/repair_recipe.current.json
RECIPE_BINDING=$TRAIN_OUTPUT/repair_recipe.binding.json
TRAIN_COMPLETION=$TRAIN_OUTPUT/repair_training_completion.json
BASELINE_PREFLIGHT=$RUN_ROOT/preflight_bc0_v2.json
PREFLIGHT_DECISION=$RUN_ROOT/preflight_decision_v2.json
NEW_EVAL=$RUN_ROOT/dagger_repair_eval_14_v2.json
COMPARISON=$RUN_ROOT/comparison_repair_vs_bc0_v2.json
EVAL_BINDING=$NEW_EVAL.binding.json
COMPARISON_BINDING=$COMPARISON.binding.json

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-E2B-it}
MODEL_REVISION=${MODEL_REVISION:-f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae}
SEED=${SEED:-3407}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-32}
SAVE_EVAL_STEPS=${SAVE_EVAL_STEPS:-8}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-12}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WARMUP_STEPS=${WARMUP_STEPS:-4}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
PREEMPTION_EXIT_CODE=99

(( TRAIN_MAX_STEPS == 32 )) || {
    echo "TRAIN_MAX_STEPS must remain 32 because checkpoint selection is frozen"
    exit 2
}
(( SAVE_EVAL_STEPS == 8 )) || {
    echo "SAVE_EVAL_STEPS must remain 8 because checkpoint selection is frozen"
    exit 2
}
(( PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS == 16 )) || {
    echo "the repair recipe requires an effective train batch of 16"
    exit 2
}

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

mkdir -p "$RUN_ROOT" "$TRAIN_OUTPUT"
exec 9>"$RUN_ROOT/launcher.lock"
if ! flock -n 9; then
    echo "[duplicate] another repair launcher owns $RUN_ROOT; exiting without work"
    exit 0
fi

test -x "$PYTHON"
for required in \
    "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
    "$REPO_ROOT/eval_sft_agent_gemma_v4.py" \
    "$REPO_ROOT/scripts/research_dagger_demo.py" \
    "$REPO_ROOT/psse_env/dagger/dataset_builder.py" \
    "$REPO_ROOT/psse_env/dagger/evaluator.py" \
    "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py" \
    "$REPO_ROOT/psse_env/dagger/preliminary_tool_gate.py" \
    "$REPO_ROOT/psse_env/dagger/protocol_bridge.py" \
    "$REPO_ROOT/psse_env/dagger/release_factories.py" \
    "$D0_FILE" \
    "$NATURAL_TRAIN" \
    "$NATURAL_VALIDATION" \
    "$PROBE_DONOR" \
    "$PROBE_AUDIT" \
    "$TRACE_VALIDATION" \
    "$HELDOUT_SUITE" \
    "$BC0_EVAL" \
    "$BC0_ADAPTER/adapter_config.json" \
    "$BC0_ADAPTER/adapter_model.safetensors"; do
    test -s "$required"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"
export HF_HOME=${HF_HOME:-/scratch/yx3882/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yx3882/.cache/torch}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export RESEARCH_MAX_INPUT_TOKENS="$MAX_SEQ_LENGTH"
export RESEARCH_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"

cd "$REPO_ROOT"

required_suites=(
    recovery_measurement_parameter_sequential_handoff
    recovery_post_failure_no_candidate
    recovery_premature_commit
    recovery_premature_escalation
    recovery_rejected_candidate_rollback
    recovery_safe_continuation_after_partial_success
    recovery_unsupported_correction
)
suite_args=()
for suite in "${required_suites[@]}"; do
    suite_args+=(--required-suite "$suite")
done

file_sha256() {
    "$PYTHON" -c \
        'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); [h.update(block) for block in iter(lambda:f.read(1048576), b"")]; print(h.hexdigest())' \
        "$1"
}

tree_sha256() {
    "$PYTHON" -c \
        'import sys; from psse_env.dagger.release_factories import checkpoint_tree_sha256; print(checkpoint_tree_sha256(sys.argv[1]))' \
        "$1"
}

code_bundle_sha256() {
    "$PYTHON" - "$@" <<'PY'
import hashlib
from pathlib import Path
import sys

digest = hashlib.sha256(b"research-dagger-code-bundle-v1\0")
for raw in sys.argv[1:]:
    path = Path(raw).resolve(strict=True)
    digest.update(str(path).encode("utf-8") + b"\0")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

stable_args_sha256() {
    "$PYTHON" - "$@" <<'PY'
import hashlib
import json
import sys

payload = json.dumps(
    sys.argv[1:], sort_keys=False, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

write_kv_json_atomic() {
    local output=$1
    shift
    "$PYTHON" - "$output" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

output = Path(sys.argv[1])
payload = {}
for item in sys.argv[2:]:
    key, separator, value = item.partition("=")
    if not separator or not key or key in payload:
        raise ValueError(f"invalid or duplicate binding field: {item!r}")
    payload[key] = value
output.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

cache_binding_current() {
    local binding=$1
    local output=$2
    local contract=$3
    local request_sha=$4
    [[ -s "$binding" && -s "$output" ]] || return 1
    "$PYTHON" - "$binding" "$output" "$contract" "$request_sha" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

binding_path, output_path = map(Path, sys.argv[1:3])
contract, request_sha = sys.argv[3:5]
binding = json.loads(binding_path.read_text(encoding="utf-8"))
actual_output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
current = (
    binding.get("contract") == contract
    and binding.get("request_sha256") == request_sha
    and binding.get("output_sha256") == actual_output_sha
)
raise SystemExit(0 if current else 3)
PY
}

write_cache_binding() {
    local binding=$1
    local output=$2
    local contract=$3
    local request_sha=$4
    write_kv_json_atomic \
        "$binding" \
        "contract=$contract" \
        "request_sha256=$request_sha" \
        "output_path=$output" \
        "output_sha256=$(file_sha256 "$output")"
}

build_curriculum() {
    echo "[curriculum] deterministic 512-row decision-balanced repair view"
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" repair-curriculum \
        --d0 "$D0_FILE" \
        --natural "$NATURAL_TRAIN" \
        --natural "$NATURAL_VALIDATION" \
        --probe-donor "$PROBE_DONOR" \
        --probe-audit "$PROBE_AUDIT" \
        --protected "$TRACE_VALIDATION" \
        --protected "$HELDOUT_SUITE" \
        --output "$REPAIR_TRAIN" \
        --report-output "$REPAIR_REPORT" \
        --seed "$SEED"
}

latest_checkpoint() {
    local candidate suffix step
    local latest=""
    local latest_step=-1
    shopt -s nullglob
    for candidate in "$TRAIN_OUTPUT"/checkpoint-*; do
        [[ -d "$candidate" && -s "$candidate/trainer_state.json" ]] || continue
        suffix=${candidate##*/checkpoint-}
        [[ "$suffix" =~ ^[0-9]+$ ]] || continue
        step=$((10#$suffix))
        if (( step > latest_step )); then
            latest_step=$step
            latest=$candidate
        fi
    done
    shopt -u nullglob
    printf '%s' "$latest"
}

checkpoint_global_step() {
    "$PYTHON" -c \
        'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["global_step"]))' \
        "$1/trainer_state.json"
}

has_reusable_training_state() {
    if [[ -s "$NEW_ADAPTER/adapter_model.safetensors" || \
          -s "$NEW_ADAPTER/adapter_config.json" ]]; then
        return 0
    fi
    local checkpoint
    shopt -s nullglob
    for checkpoint in "$TRAIN_OUTPUT"/checkpoint-*; do
        if [[ -s "$checkpoint/trainer_state.json" || \
              -s "$checkpoint/adapter_model.safetensors" ]]; then
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

build_recipe_binding() {
    local bc0_revision training_code_sha policy_code_sha
    bc0_revision=$(tree_sha256 "$BC0_ADAPTER")
    training_code_sha=$(code_bundle_sha256 \
        "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        "$REPO_ROOT/psse_env/dagger/dataset_builder.py" \
        "$REPO_ROOT/psse_env/dagger/protocol_bridge.py")
    policy_code_sha=$(code_bundle_sha256 \
        "$REPO_ROOT/scripts/research_dagger_demo.py" \
        "$REPO_ROOT/eval_sft_agent_gemma_v4.py" \
        "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py" \
        "$REPO_ROOT/psse_env/dagger/preliminary_tool_gate.py" \
        "$REPO_ROOT/psse_env/dagger/protocol_bridge.py" \
        "$REPO_ROOT/psse_env/dagger/release_factories.py")
    write_kv_json_atomic \
        "$RECIPE_CURRENT" \
        "contract=research_dagger_repair_training_recipe_v1" \
        "curriculum_contract=research_dagger_repair_curriculum_v2" \
        "curriculum_path=$REPAIR_TRAIN" \
        "curriculum_sha256=$(file_sha256 "$REPAIR_TRAIN")" \
        "curriculum_report_sha256=$(file_sha256 "$REPAIR_REPORT")" \
        "bc0_adapter_path=$BC0_ADAPTER" \
        "bc0_adapter_tree_sha256=$bc0_revision" \
        "validation_sha256=$(file_sha256 "$TRACE_VALIDATION")" \
        "training_code_sha256=$training_code_sha" \
        "policy_code_sha256=$policy_code_sha" \
        "model_name=$MODEL_NAME" \
        "model_revision=$MODEL_REVISION" \
        "seed=$SEED" \
        "max_steps=$TRAIN_MAX_STEPS" \
        "save_eval_steps=$SAVE_EVAL_STEPS" \
        "max_seq_length=$MAX_SEQ_LENGTH" \
        "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE" \
        "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS" \
        "learning_rate=$LEARNING_RATE" \
        "warmup_steps=$WARMUP_STEPS" \
        "lora_r=$LORA_R" \
        "lora_alpha=$LORA_ALPHA" \
        "lora_target_scope=language_model" \
        "load_in_16bit=true" \
        "dataset_num_proc=1" \
        "per_device_eval_batch_size=1" \
        "dataloader_num_workers=4" \
        "num_train_epochs=1" \
        "save_total_limit=4" \
        "preserve_system_text=true" \
        "include_tool_schemas=true" \
        "phase_gated_prompt=false" \
        "inject_empty_thought_channel=false"
}

bind_training_recipe() {
    build_recipe_binding
    if [[ -s "$RECIPE_BINDING" ]] && cmp -s "$RECIPE_CURRENT" "$RECIPE_BINDING"; then
        echo "[recipe] current training binding matches reusable state"
        return 0
    fi
    if has_reusable_training_state; then
        echo "[recipe] refusing to resume or reuse training state under a different/missing recipe binding" >&2
        echo "existing: $RECIPE_BINDING" >&2
        echo "current:  $RECIPE_CURRENT" >&2
        return 2
    fi
    local temporary=$RECIPE_BINDING.tmp.$$
    cp "$RECIPE_CURRENT" "$temporary"
    mv -f "$temporary" "$RECIPE_BINDING"
    echo "[recipe] installed new immutable training binding"
}

training_completion_current() {
    local checkpoint=$1
    [[ -s "$TRAIN_COMPLETION" && \
       -s "$NEW_ADAPTER/adapter_model.safetensors" && \
       -s "$NEW_ADAPTER/adapter_config.json" ]] || return 1
    local adapter_revision recipe_sha trainer_state_sha step
    adapter_revision=$(tree_sha256 "$NEW_ADAPTER")
    recipe_sha=$(file_sha256 "$RECIPE_BINDING")
    trainer_state_sha=$(file_sha256 "$checkpoint/trainer_state.json")
    step=$(checkpoint_global_step "$checkpoint")
    "$PYTHON" - \
        "$TRAIN_COMPLETION" "$recipe_sha" "$adapter_revision" \
        "$checkpoint" "$trainer_state_sha" "$step" "$TRAIN_MAX_STEPS" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
current = (
    report.get("contract") == "research_dagger_repair_training_completion_v1"
    and report.get("recipe_sha256") == sys.argv[2]
    and report.get("adapter_tree_sha256") == sys.argv[3]
    and report.get("checkpoint_path") == sys.argv[4]
    and report.get("trainer_state_sha256") == sys.argv[5]
    and report.get("global_step") == sys.argv[6]
    and sys.argv[6] == sys.argv[7]
)
raise SystemExit(0 if current else 3)
PY
}

write_training_completion() {
    local checkpoint=$1
    local step
    step=$(checkpoint_global_step "$checkpoint")
    [[ "$step" == "$TRAIN_MAX_STEPS" ]]
    write_kv_json_atomic \
        "$TRAIN_COMPLETION" \
        "contract=research_dagger_repair_training_completion_v1" \
        "recipe_sha256=$(file_sha256 "$RECIPE_BINDING")" \
        "adapter_path=$NEW_ADAPTER" \
        "adapter_tree_sha256=$(tree_sha256 "$NEW_ADAPTER")" \
        "checkpoint_path=$checkpoint" \
        "trainer_state_sha256=$(file_sha256 "$checkpoint/trainer_state.json")" \
        "global_step=$step"
}

run_training() {
    local completed_checkpoint resume_checkpoint completed_step status
    completed_checkpoint=$(latest_checkpoint)
    if [[ -n "$completed_checkpoint" ]]; then
        completed_step=$(checkpoint_global_step "$completed_checkpoint")
        (( completed_step <= TRAIN_MAX_STEPS )) || {
            echo "[train] latest checkpoint exceeds the frozen 32-step recipe: $completed_checkpoint" >&2
            return 2
        }
    fi
    if [[ -n "$completed_checkpoint" ]] && training_completion_current "$completed_checkpoint"; then
        echo "[train] recipe-bound completed repair adapter exists; skipping: $NEW_ADAPTER"
        return 0
    fi
    if [[ -n "$completed_checkpoint" && \
          "$completed_step" == "$TRAIN_MAX_STEPS" && \
          -s "$NEW_ADAPTER/adapter_model.safetensors" && \
          -s "$NEW_ADAPTER/adapter_config.json" ]]; then
        write_training_completion "$completed_checkpoint"
        echo "[train] attested completed repair adapter; skipping: $NEW_ADAPTER"
        return 0
    fi

    resume_checkpoint=$(latest_checkpoint)
    local -a resume_args=()
    if [[ -n "$resume_checkpoint" ]]; then
        echo "[train] recipe-bound resume checkpoint: $resume_checkpoint"
        resume_args+=(--resume-from-checkpoint "$resume_checkpoint")
    else
        echo "[train] 32-step repair continuation directly from BC0"
    fi

    set +e
    "$PYTHON" "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        --train-file "$REPAIR_TRAIN" \
        --valid-file "$TRACE_VALIDATION" \
        --init-adapter "$BC0_ADAPTER" \
        --model-name "$MODEL_NAME" \
        --model-revision "$MODEL_REVISION" \
        --require-pinned-model-revision \
        --output-dir "$TRAIN_OUTPUT" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --keep-too-long-targets \
        --dataset-num-proc 1 \
        --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per-device-eval-batch-size 1 \
        --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
        --max-steps "$TRAIN_MAX_STEPS" \
        --num-train-epochs 1 \
        --learning-rate "$LEARNING_RATE" \
        --warmup-steps "$WARMUP_STEPS" \
        --logging-steps 2 \
        --eval-steps "$SAVE_EVAL_STEPS" \
        --save-steps "$SAVE_EVAL_STEPS" \
        --save-total-limit 4 \
        --load-in-16bit \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-target-scope language_model \
        --dataloader-num-workers 4 \
        --seed "$SEED" \
        --preserve-system-text \
        --include-tool-schemas \
        --no-phase-gated-prompt \
        --no-inject-empty-thought-channel \
        --repeat-first-tool-call 1 \
        --repeat-later-tool-call 1 \
        --repeat-final 1 \
        --sanity-check-samples 0 \
        --run-name research-e2b-dagger-repair-v2 \
        --report-to none \
        "${resume_args[@]}"
    status=$?
    set -e
    if (( status == PREEMPTION_EXIT_CODE )); then
        echo "[train] trainer checkpointed after a scheduler signal"
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            scontrol requeue "$SLURM_JOB_ID"
            exit 0
        fi
        return "$status"
    fi
    (( status == 0 )) || return "$status"

    completed_checkpoint=$(latest_checkpoint)
    [[ -n "$completed_checkpoint" ]]
    completed_step=$(checkpoint_global_step "$completed_checkpoint")
    [[ "$completed_step" == "$TRAIN_MAX_STEPS" ]]
    test -s "$NEW_ADAPTER/adapter_model.safetensors"
    test -s "$NEW_ADAPTER/adapter_config.json"
    write_training_completion "$completed_checkpoint"
}

preflight_report_current() {
    local report=$1
    local adapter_revision=$2
    local validation_sha=$3
    "$PYTHON" - "$report" "$adapter_revision" "$validation_sha" <<'PY'
from pathlib import Path
import sys

from scripts.research_dagger_demo import (
    _read_object,
    _validate_trace_preflight_report,
)

path = Path(sys.argv[1]).resolve(strict=True)
report = _read_object(path)
try:
    _validate_trace_preflight_report(report, path=path)
except (KeyError, TypeError, ValueError):
    raise SystemExit(3) from None
if report.get("adapter_tree_sha256") != sys.argv[2]:
    raise SystemExit(3)
if report.get("validation_file_sha256") != sys.argv[3]:
    raise SystemExit(3)
raise SystemExit(0)
PY
}

preflight_one() {
    local adapter=$1
    local output=$2
    local label=$3
    local binding=$output.binding.json
    local adapter_revision validation_sha policy_code_sha request_sha
    adapter_revision=$(tree_sha256 "$adapter")
    validation_sha=$(file_sha256 "$TRACE_VALIDATION")
    policy_code_sha=$(code_bundle_sha256 \
        "$REPO_ROOT/scripts/research_dagger_demo.py" \
        "$REPO_ROOT/eval_sft_agent_gemma_v4.py" \
        "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        "$REPO_ROOT/psse_env/dagger/dataset_builder.py" \
        "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py" \
        "$REPO_ROOT/psse_env/dagger/preliminary_tool_gate.py" \
        "$REPO_ROOT/psse_env/dagger/protocol_bridge.py" \
        "$REPO_ROOT/psse_env/dagger/release_factories.py")
    request_sha=$(stable_args_sha256 \
        "contract=research_dagger_preflight_cache_request_v1" \
        "adapter_path=$adapter" \
        "adapter_tree_sha256=$adapter_revision" \
        "validation_sha256=$validation_sha" \
        "max_input_tokens=$MAX_SEQ_LENGTH" \
        "max_new_tokens=$MAX_NEW_TOKENS" \
        "model_name=$MODEL_NAME" \
        "model_revision=$MODEL_REVISION" \
        "policy_code_sha256=$policy_code_sha")
    if cache_binding_current \
        "$binding" "$output" \
        research_dagger_preflight_cache_binding_v1 "$request_sha" && \
       preflight_report_current "$output" "$adapter_revision" "$validation_sha"; then
        echo "[preflight:$label] current bound report exists; skipping"
        return 0
    fi
    if [[ -e "$output" || -e "$binding" ]]; then
        echo "[preflight:$label] stale or incomplete cache will be replaced"
    fi
    echo "[preflight:$label] all 27 untouched validation actions"
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" trace-preflight \
        --validation "$TRACE_VALIDATION" \
        --adapter "$adapter" \
        --output "$output"
    preflight_report_current "$output" "$adapter_revision" "$validation_sha"
    write_cache_binding \
        "$binding" "$output" \
        research_dagger_preflight_cache_binding_v1 "$request_sha"
}

run_preflights_and_decide() {
    preflight_one "$BC0_ADAPTER" "$BASELINE_PREFLIGHT" bc0

    local step checkpoint report
    local -a candidate_args=()
    for step in 8 16 24 32; do
        checkpoint="$TRAIN_OUTPUT/checkpoint-$step"
        [[ -s "$checkpoint/adapter_model.safetensors" ]]
        [[ -s "$checkpoint/adapter_config.json" ]]
        report="$RUN_ROOT/preflight_checkpoint-$step-v2.json"
        preflight_one "$checkpoint" "$report" "checkpoint-$step"
        candidate_args+=(--candidate "$report")
    done

    local decision_status=0
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" preflight-decision \
        --baseline "$BASELINE_PREFLIGHT" \
        "${candidate_args[@]}" \
        --output "$PREFLIGHT_DECISION" \
        --required-tool wls_from_path \
        --required-tool rollback_state \
        --required-tool correct_parameters_from_path \
        --required-tool get_measurement_context \
        --required-tool get_topology_context \
        --minimum-exact 5 \
        --minimum-schema-rate 0.95 \
        --minimum-state-bound-rate 0.90 \
        --stop-on-fail || decision_status=$?
    if (( decision_status == 2 )); then
        echo "RESEARCH_DAGGER_REPAIR_STOPPED_BEFORE_CLOSED_LOOP"
        echo "decision: $PREFLIGHT_DECISION"
        touch "$RUN_ROOT/STOPPED_BEFORE_CLOSED_LOOP"
        exit 0
    fi
    (( decision_status == 0 )) || return "$decision_status"
    rm -f "$RUN_ROOT/STOPPED_BEFORE_CLOSED_LOOP"
}

evaluation_report_current() {
    local report=$1
    local adapter=$2
    local adapter_revision=$3
    local suite_sha=$4
    "$PYTHON" - "$report" "$adapter" "$adapter_revision" "$suite_sha" <<'PY'
import json
import os
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
provenance = report.get("provenance")
provenance = provenance if isinstance(provenance, dict) else {}
policy = provenance.get("policy_identity")
suite = provenance.get("input_suite")
policy = policy if isinstance(policy, dict) else {}
suite = suite if isinstance(suite, dict) else {}
current = (
    report.get("artifact_type") == "closed_loop_diagnostic_evaluation"
    and os.path.realpath(str(policy.get("model_id") or ""))
        == os.path.realpath(sys.argv[2])
    and policy.get("model_revision") == sys.argv[3]
    and suite.get("sha256") == sys.argv[4]
)
raise SystemExit(0 if current else 3)
PY
}

run_evaluation() {
    local selected_adapter selected_revision actual_revision heldout_sha
    local evaluation_code_sha decision_sha request_sha
    selected_adapter=$("$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["adapter_path"])' \
        "$PREFLIGHT_DECISION")
    selected_revision=$("$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["adapter_tree_sha256"])' \
        "$PREFLIGHT_DECISION")
    test -s "$selected_adapter/adapter_model.safetensors"
    actual_revision=$(tree_sha256 "$selected_adapter")
    [[ "$actual_revision" == "$selected_revision" ]] || {
        echo "[eval] selected adapter changed after preflight" >&2
        return 2
    }
    heldout_sha=$(file_sha256 "$HELDOUT_SUITE")
    decision_sha=$(file_sha256 "$PREFLIGHT_DECISION")
    evaluation_code_sha=$(code_bundle_sha256 \
        "$REPO_ROOT/eval_sft_agent_gemma_v4.py" \
        "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        "$REPO_ROOT/psse_env/dagger/dataset_builder.py" \
        "$REPO_ROOT/psse_env/dagger/evaluator.py" \
        "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py" \
        "$REPO_ROOT/psse_env/dagger/preliminary_tool_gate.py" \
        "$REPO_ROOT/psse_env/dagger/protocol_bridge.py" \
        "$REPO_ROOT/psse_env/dagger/release_factories.py")
    request_sha=$(stable_args_sha256 \
        "contract=research_dagger_repair_evaluation_request_v1" \
        "selected_adapter=$selected_adapter" \
        "selected_adapter_tree_sha256=$actual_revision" \
        "decision_sha256=$decision_sha" \
        "heldout_suite_sha256=$heldout_sha" \
        "evaluation_code_sha256=$evaluation_code_sha" \
        "max_input_tokens=$MAX_SEQ_LENGTH" \
        "max_new_tokens=$MAX_NEW_TOKENS" \
        "max_steps=$EVAL_MAX_STEPS" \
        "seed=20260823" \
        "required_suites=${required_suites[*]}")
    if cache_binding_current \
        "$EVAL_BINDING" "$NEW_EVAL" \
        research_dagger_repair_evaluation_cache_v1 "$request_sha" && \
       evaluation_report_current \
        "$NEW_EVAL" "$selected_adapter" "$actual_revision" "$heldout_sha"; then
        echo "[eval] current bound repair result exists; skipping: $NEW_EVAL"
        return 0
    fi
    if [[ -e "$NEW_EVAL" || -e "$EVAL_BINDING" ]]; then
        echo "[eval] stale or incomplete result cache will be replaced"
    fi
    echo "[eval] selected repair checkpoint on untouched 14-episode closure: $selected_adapter"
    "$PYTHON" -m psse_env.dagger.evaluator \
        --input "$HELDOUT_SUITE" \
        --output "$NEW_EVAL" \
        --env-factory psse_env.dagger.release_factories:production_environment_factory \
        --policy-factory psse_env.dagger.preliminary_e2b_eval:preliminary_e2b_policy_factory \
        --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
        --model-id "$selected_adapter" \
        --model-revision "$actual_revision" \
        --protocol canonical \
        --diagnostic-only \
        --allow-dirty-source \
        --seed 20260823 \
        --max-steps "$EVAL_MAX_STEPS" \
        "${suite_args[@]}" \
        --minimum-suites 7 \
        --minimum-episodes-per-suite 2 \
        --minimum-roots-per-suite 2
    evaluation_report_current \
        "$NEW_EVAL" "$selected_adapter" "$actual_revision" "$heldout_sha"
    write_cache_binding \
        "$EVAL_BINDING" "$NEW_EVAL" \
        research_dagger_repair_evaluation_cache_v1 "$request_sha"
}

run_summary() {
    local comparison_request_sha
    comparison_request_sha=$(stable_args_sha256 \
        "contract=research_dagger_repair_comparison_request_v1" \
        "bc0_eval_sha256=$(file_sha256 "$BC0_EVAL")" \
        "repair_eval_sha256=$(file_sha256 "$NEW_EVAL")" \
        "summary_code_sha256=$(file_sha256 "$REPO_ROOT/scripts/research_dagger_demo.py")")
    if cache_binding_current \
        "$COMPARISON_BINDING" "$COMPARISON" \
        research_dagger_repair_comparison_cache_v1 "$comparison_request_sha"; then
        echo "[summary] current bound comparison exists; skipping: $COMPARISON"
        return 0
    fi
    if [[ -e "$COMPARISON" || -e "$COMPARISON_BINDING" ]]; then
        echo "[summary] stale or incomplete comparison cache will be replaced"
    fi
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" summary \
        --bc0 "$BC0_EVAL" \
        --dagger "$NEW_EVAL" \
        --output "$COMPARISON"
    write_cache_binding \
        "$COMPARISON_BINDING" "$COMPARISON" \
        research_dagger_repair_comparison_cache_v1 "$comparison_request_sha"
}

build_curriculum
bind_training_recipe
run_training
run_preflights_and_decide
run_evaluation
run_summary

echo "RESEARCH_DAGGER_REPAIR_COMPLETE"
echo "curriculum: $REPAIR_TRAIN"
echo "view report: $REPAIR_REPORT"
echo "recipe:     $RECIPE_BINDING"
echo "decision:   $PREFLIGHT_DECISION"
echo "evaluation: $NEW_EVAL"
echo "comparison: $COMPARISON"
