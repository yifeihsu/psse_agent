#!/usr/bin/env bash
#SBATCH --job-name=dagger-research-update
#SBATCH --output=/scratch/yx3882/research_dagger_update_20260823/update_%j.out
#SBATCH --error=/scratch/yx3882/research_dagger_update_20260823/update_%j.err
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

# Research-only E2B DAgger continuation. This deliberately reuses the existing
# DAgger adapter and prebuilt data, trains one short continuation, and evaluates
# only that new adapter. It does not run BC0/Base, collect data, create receipts,
# enforce release gates, or perform promotion/multi-seed work.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/research_dagger_src_20260823}
RUN_ROOT=${RUN_ROOT:-/scratch/yx3882/research_dagger_update_20260823}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}

BALANCED_TRAIN_FILE=${BALANCED_TRAIN_FILE:-$RUN_ROOT/balanced_train.jsonl}
D1_VALIDATION_FILE=${D1_VALIDATION_FILE:-/scratch/yx3882/prelim_e2b_baa7e35_attested_20260821/preliminary.d1_validation.jsonl}
INIT_ADAPTER=${INIT_ADAPTER:-/scratch/yx3882/prelim_e2b_fullnatural_2b32029_20260822/output/dagger/lora}
EVAL_SUBSET=${EVAL_SUBSET:-/scratch/yx3882/research_dagger_demo_20260823/recovery_stress_small.json}
BC0_EVAL=${BC0_EVAL:-/scratch/yx3882/research_dagger_demo_20260823/bc0_eval.json}

TRAIN_OUTPUT=${TRAIN_OUTPUT:-$RUN_ROOT/training}
NEW_ADAPTER=$TRAIN_OUTPUT/lora
NEW_EVAL=${NEW_EVAL:-$RUN_ROOT/dagger_updated_eval.json}
COMPARISON=${COMPARISON:-$RUN_ROOT/comparison_updated_vs_bc0.json}

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-E2B-it}
MODEL_REVISION=${MODEL_REVISION:-f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae}
TRAIN_SEED=${TRAIN_SEED:-3407}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-64}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
EPISODE_MAX_STEPS=${EPISODE_MAX_STEPS:-12}

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

mkdir -p "$RUN_ROOT" "$TRAIN_OUTPUT"
test -x "$PYTHON"
test -s "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py"
test -s "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py"
test -s "$REPO_ROOT/scripts/research_dagger_demo.py"
test -s "$BALANCED_TRAIN_FILE"
test -s "$D1_VALIDATION_FILE"
test -s "$INIT_ADAPTER/adapter_config.json"
test -s "$INIT_ADAPTER/adapter_model.safetensors"
test -s "$EVAL_SUBSET"
test -s "$BC0_EVAL"

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

cd "$REPO_ROOT"

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

run_training() {
    if [[ -s "$NEW_ADAPTER/adapter_model.safetensors" && \
          -s "$NEW_ADAPTER/adapter_config.json" ]]; then
        echo "[train] completed adapter exists; skipping: $NEW_ADAPTER"
        return 0
    fi

    local resume_checkpoint
    resume_checkpoint=$(latest_checkpoint)
    local -a resume_args=()
    if [[ -n "$resume_checkpoint" ]]; then
        echo "[train] resuming latest checkpoint: $resume_checkpoint"
        # An explicit path is intentional: the trainer rejects the ambiguous
        # combination of --init-adapter and --resume-from-checkpoint=auto.
        resume_args+=(--resume-from-checkpoint "$resume_checkpoint")
    else
        echo "[train] starting short continuation from existing DAgger adapter: $INIT_ADAPTER"
    fi

    "$PYTHON" "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        --train-file "$BALANCED_TRAIN_FILE" \
        --valid-file "$D1_VALIDATION_FILE" \
        --init-adapter "$INIT_ADAPTER" \
        --model-name "$MODEL_NAME" \
        --model-revision "$MODEL_REVISION" \
        --require-pinned-model-revision \
        --output-dir "$TRAIN_OUTPUT" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --keep-too-long-targets \
        --dataset-num-proc 1 \
        --per-device-train-batch-size 2 \
        --per-device-eval-batch-size 1 \
        --gradient-accumulation-steps 8 \
        --max-steps "$TRAIN_MAX_STEPS" \
        --num-train-epochs 1 \
        --learning-rate 2e-5 \
        --warmup-steps 4 \
        --logging-steps 2 \
        --eval-steps 8 \
        --save-steps 8 \
        --save-total-limit 3 \
        --load-in-16bit \
        --lora-r 16 \
        --lora-alpha 16 \
        --lora-target-scope language_model \
        --dataloader-num-workers 4 \
        --seed "$TRAIN_SEED" \
        --preserve-system-text \
        --include-tool-schemas \
        --no-phase-gated-prompt \
        --no-inject-empty-thought-channel \
        --repeat-first-tool-call 1 \
        --repeat-later-tool-call 1 \
        --repeat-final 1 \
        --sanity-check-samples 0 \
        --run-name "research-e2b-dagger-update" \
        --report-to none \
        "${resume_args[@]}"
}

required_suites=(
    recovery_measurement_parameter_sequential_handoff
    recovery_post_failure_no_candidate
    recovery_premature_commit
    recovery_premature_escalation
    recovery_rejected_candidate_rollback
    recovery_safe_continuation_after_partial_success
    recovery_unsupported_correction
)

run_evaluation() {
    if [[ -s "$NEW_EVAL" ]]; then
        echo "[eval] completed output exists; skipping: $NEW_EVAL"
        return 0
    fi

    export RESEARCH_MAX_INPUT_TOKENS="$MAX_SEQ_LENGTH"
    export RESEARCH_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
    export RESEARCH_RELAXED_ADAPTER_IDENTITY=1

    local adapter_revision
    adapter_revision=$(sha256sum "$NEW_ADAPTER/adapter_model.safetensors" | awk '{print $1}')
    local -a suite_args=()
    local suite
    for suite in "${required_suites[@]}"; do
        suite_args+=(--required-suite "$suite")
    done

    echo "[eval] evaluating only the updated adapter on the existing 14-episode subset"
    "$PYTHON" -m psse_env.dagger.evaluator \
        --input "$EVAL_SUBSET" \
        --output "$NEW_EVAL" \
        --env-factory psse_env.dagger.release_factories:production_environment_factory \
        --policy-factory psse_env.dagger.preliminary_e2b_eval:preliminary_e2b_policy_factory \
        --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
        --model-id "$NEW_ADAPTER" \
        --model-revision "$adapter_revision" \
        --protocol canonical \
        --diagnostic-only \
        --allow-dirty-source \
        --seed 20260823 \
        --max-steps "$EPISODE_MAX_STEPS" \
        "${suite_args[@]}" \
        --minimum-suites 7 \
        --minimum-episodes-per-suite 2 \
        --minimum-roots-per-suite 1
}

run_summary() {
    if [[ -s "$COMPARISON" ]]; then
        echo "[summary] completed comparison exists; skipping: $COMPARISON"
        return 0
    fi
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" summary \
        --bc0 "$BC0_EVAL" \
        --dagger "$NEW_EVAL" \
        --output "$COMPARISON"
}

run_training
run_evaluation
run_summary

echo "RESEARCH_DAGGER_UPDATE_COMPLETE"
echo "adapter:    $NEW_ADAPTER"
echo "evaluation: $NEW_EVAL"
echo "comparison: $COMPARISON"
