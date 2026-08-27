#!/usr/bin/env bash
#SBATCH --job-name=dagger-trace-demo
#SBATCH --output=/scratch/yx3882/research_dagger_trace_20260823/trace_%j.out
#SBATCH --error=/scratch/yx3882/research_dagger_trace_20260823/trace_%j.err
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
#SBATCH --signal=B:USR1@120
#SBATCH --comment="preemption=yes;requeue=true"

# One research-only DAgger iteration:
#   1. make a 56-episode/14-episode physical-root-closure split;
#   2. roll out the current learner for four steps on the 16 training roots;
#   3. export observable expert actions on the states that learner visited;
#   4. run one short LoRA continuation from the current learner; and
#   5. greedily preflight every root-disjoint validation label; and
#   6. evaluate BC0 and the result on the same untouched four-root closure only
#      when at least one validation action is exactly correct.
#
# The stages are restartable and deliberately omit release receipts, immutable
# source hashes, production promotion gates, multi-seed runs, and base/BC0
# retraining. The completed BC0 adapter is reused for one matched evaluation.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/research_dagger_src_20260823}
RUN_ROOT=${RUN_ROOT:-/scratch/yx3882/research_dagger_trace_20260823}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}

FULL_SUITE=${FULL_SUITE:-/scratch/yx3882/dagger_release_a5a7574_20260823/dagger1_a5a7574/recovery_stress.json}
D0_FILE=${D0_FILE:-/scratch/yx3882/prelim_e2b_baa7e35_attested_20260821/preliminary.bc0_train.jsonl}
INIT_ADAPTER=${INIT_ADAPTER:-/scratch/yx3882/research_dagger_update_20260823/training/lora}
BC0_ADAPTER=${BC0_ADAPTER:-/scratch/yx3882/prelim_e2b_fullnatural_2b32029_20260822/output/bc0/lora}

COLLECTION_SUITE=$RUN_ROOT/collection_suite_56_root_disjoint.json
HELDOUT_SUITE=$RUN_ROOT/heldout_suite_14_root_closure.json
COLLECTION_EVAL=$RUN_ROOT/learner_collection_56.json
TRACE_TRAIN=$RUN_ROOT/trace_train.jsonl
TRACE_VALIDATION=$RUN_ROOT/trace_validation.jsonl
TRACE_REPORT=$RUN_ROOT/trace_view_report.json
TRAIN_OUTPUT=$RUN_ROOT/training
NEW_ADAPTER=$TRAIN_OUTPUT/lora
TRACE_PREFLIGHT=$RUN_ROOT/trace_validation_greedy.json
BC0_EVAL=$RUN_ROOT/bc0_eval_14.json
NEW_EVAL=$RUN_ROOT/dagger_trace_eval_14.json
COMPARISON=$RUN_ROOT/comparison_trace_vs_bc0.json

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-E2B-it}
MODEL_REVISION=${MODEL_REVISION:-f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae}
SEED=${SEED:-3407}
COLLECTION_MAX_STEPS=${COLLECTION_MAX_STEPS:-4}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-32}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-12}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

mkdir -p "$RUN_ROOT" "$TRAIN_OUTPUT"
test -x "$PYTHON"
test -s "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py"
test -s "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py"
test -s "$REPO_ROOT/scripts/research_dagger_demo.py"
test -s "$FULL_SUITE"
test -s "$D0_FILE"
test -s "$INIT_ADAPTER/adapter_config.json"
test -s "$INIT_ADAPTER/adapter_model.safetensors"
test -s "$BC0_ADAPTER/adapter_config.json"
test -s "$BC0_ADAPTER/adapter_model.safetensors"

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

make_collection_suite() {
    # This is the one scientific split check retained in the prototype. Rows
    # sharing one physical root stay together even when suites reuse that root.
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" root-split \
        --input "$FULL_SUITE" \
        --train-output "$COLLECTION_SUITE" \
        --heldout-output "$HELDOUT_SUITE" \
        --heldout-per-suite 2
}

run_collection() {
    if [[ -s "$COLLECTION_EVAL" ]]; then
        echo "[collect] completed learner rollout exists; skipping: $COLLECTION_EVAL"
        return 0
    fi

    local adapter_revision
    adapter_revision=$("$PYTHON" -c \
        'import sys; from psse_env.dagger.release_factories import checkpoint_tree_sha256; print(checkpoint_tree_sha256(sys.argv[1]))' \
        "$INIT_ADAPTER")
    echo "[collect] 56 episodes on 16 roots x at most $COLLECTION_MAX_STEPS learner steps"
    "$PYTHON" -m psse_env.dagger.evaluator \
        --input "$COLLECTION_SUITE" \
        --output "$COLLECTION_EVAL" \
        --env-factory psse_env.dagger.release_factories:production_environment_factory \
        --policy-factory psse_env.dagger.preliminary_e2b_eval:preliminary_e2b_policy_factory \
        --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
        --model-id "$INIT_ADAPTER" \
        --model-revision "$adapter_revision" \
        --protocol canonical \
        --diagnostic-only \
        --allow-dirty-source \
        --seed 20260823 \
        --max-steps "$COLLECTION_MAX_STEPS" \
        "${suite_args[@]}" \
        --minimum-suites 7 \
        --minimum-episodes-per-suite 8 \
        --minimum-roots-per-suite 8
}

make_trace_view() {
    if [[ -s "$TRACE_TRAIN" && -s "$TRACE_VALIDATION" && -s "$TRACE_REPORT" ]]; then
        echo "[export] completed trace view exists; skipping"
        return 0
    fi

    echo "[export] canonical SFT targets for learner-visited states"
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" trace-view \
        --artifact "$COLLECTION_EVAL" \
        --d0 "$D0_FILE" \
        --train-output "$TRACE_TRAIN" \
        --validation-output "$TRACE_VALIDATION" \
        --report-output "$TRACE_REPORT" \
        --protected-suite "$HELDOUT_SUITE" \
        --validation-roots-per-suite 1 \
        --max-rows-per-episode "$COLLECTION_MAX_STEPS" \
        --dagger-repeat 2 \
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

run_training() {
    local complete_marker=$TRAIN_OUTPUT/COMPLETE
    local completed_checkpoint
    completed_checkpoint=$(latest_checkpoint)
    if [[ -s "$NEW_ADAPTER/adapter_model.safetensors" && \
          -s "$NEW_ADAPTER/adapter_config.json" && \
          -n "$completed_checkpoint" ]] && \
       "$PYTHON" -c \
          'import json,sys; assert int(json.load(open(sys.argv[1], encoding="utf-8"))["global_step"]) == int(sys.argv[2])' \
          "$completed_checkpoint/trainer_state.json" "$TRAIN_MAX_STEPS"; then
        touch "$complete_marker"
        echo "[train] completed trace adapter exists; skipping: $NEW_ADAPTER"
        return 0
    fi

    local resume_checkpoint
    resume_checkpoint=$(latest_checkpoint)
    local -a resume_args=()
    if [[ -n "$resume_checkpoint" ]]; then
        echo "[train] resuming checkpoint: $resume_checkpoint"
        resume_args+=(--resume-from-checkpoint "$resume_checkpoint")
    else
        echo "[train] short continuation from: $INIT_ADAPTER"
    fi

    "$PYTHON" "$REPO_ROOT/gpt_oss_power_sft_revised_v3.py" \
        --train-file "$TRACE_TRAIN" \
        --valid-file "$TRACE_VALIDATION" \
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
        --seed "$SEED" \
        --preserve-system-text \
        --include-tool-schemas \
        --no-phase-gated-prompt \
        --no-inject-empty-thought-channel \
        --repeat-first-tool-call 1 \
        --repeat-later-tool-call 1 \
        --repeat-final 1 \
        --sanity-check-samples 0 \
        --run-name research-e2b-dagger-learner-trace \
        --report-to none \
        "${resume_args[@]}"

    completed_checkpoint=$(latest_checkpoint)
    [[ -n "$completed_checkpoint" ]]
    "$PYTHON" -c \
        'import json,sys; assert int(json.load(open(sys.argv[1], encoding="utf-8"))["global_step"]) == int(sys.argv[2])' \
        "$completed_checkpoint/trainer_state.json" "$TRAIN_MAX_STEPS"
    touch "$complete_marker"
}

run_trace_preflight() {
    local adapter_revision validation_sha current_status
    adapter_revision=$("$PYTHON" -c \
        'import sys; from psse_env.dagger.release_factories import checkpoint_tree_sha256; print(checkpoint_tree_sha256(sys.argv[1]))' \
        "$NEW_ADAPTER")
    validation_sha=$("$PYTHON" -c \
        'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
        "$TRACE_VALIDATION")

    if [[ -s "$TRACE_PREFLIGHT" ]]; then
        current_status=0
        "$PYTHON" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
current = (
    report.get("contract") == "research_dagger_trace_preflight_v1"
    and report.get("adapter_tree_sha256") == sys.argv[2]
    and report.get("validation_file_sha256") == sys.argv[3]
)
if not current:
    raise SystemExit(3)
if int(report.get("overall", {}).get("exact_target_match_count", 0)) == 0:
    raise SystemExit(2)
' "$TRACE_PREFLIGHT" "$adapter_revision" "$validation_sha" || current_status=$?
        if (( current_status == 0 )); then
            echo "[preflight] current nonzero-exact report exists; skipping: $TRACE_PREFLIGHT"
            return 0
        fi
        if (( current_status == 2 )); then
            echo "[preflight] current adapter has zero exact validation actions; closed-loop evaluation skipped"
            return 2
        fi
        echo "[preflight] stale or malformed report will be replaced: $TRACE_PREFLIGHT"
    fi

    echo "[preflight] greedy exact-action check on all trace-validation rows"
    "$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" trace-preflight \
        --validation "$TRACE_VALIDATION" \
        --adapter "$NEW_ADAPTER" \
        --output "$TRACE_PREFLIGHT" \
        --stop-on-zero-exact
}

run_model_evaluation() {
    local label=$1
    local adapter=$2
    local output=$3
    if [[ -s "$output" ]]; then
        echo "[$label] completed 14-episode result exists; skipping: $output"
        return 0
    fi

    local adapter_revision
    adapter_revision=$("$PYTHON" -c \
        'import sys; from psse_env.dagger.release_factories import checkpoint_tree_sha256; print(checkpoint_tree_sha256(sys.argv[1]))' \
        "$adapter")
    echo "[$label] adapter on untouched 14-episode/four-root closure"
    "$PYTHON" -m psse_env.dagger.evaluator \
        --input "$HELDOUT_SUITE" \
        --output "$output" \
        --env-factory psse_env.dagger.release_factories:production_environment_factory \
        --policy-factory psse_env.dagger.preliminary_e2b_eval:preliminary_e2b_policy_factory \
        --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
        --model-id "$adapter" \
        --model-revision "$adapter_revision" \
        --protocol canonical \
        --diagnostic-only \
        --allow-dirty-source \
        --seed 20260823 \
        --max-steps "$EVAL_MAX_STEPS" \
        "${suite_args[@]}" \
        --minimum-suites 7 \
        --minimum-episodes-per-suite 2 \
        --minimum-roots-per-suite 2
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

make_collection_suite
run_collection
make_trace_view
run_training
run_trace_preflight
run_model_evaluation trace "$NEW_ADAPTER" "$NEW_EVAL"
run_model_evaluation bc0 "$BC0_ADAPTER" "$BC0_EVAL"
run_summary

echo "RESEARCH_DAGGER_TRACE_UPDATE_COMPLETE"
echo "trace report: $TRACE_REPORT"
echo "adapter:      $NEW_ADAPTER"
echo "preflight:    $TRACE_PREFLIGHT"
echo "evaluation:   $NEW_EVAL"
echo "comparison:   $COMPARISON"
