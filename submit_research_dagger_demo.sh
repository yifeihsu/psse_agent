#!/usr/bin/env bash
#SBATCH --job-name=dagger-research-demo
#SBATCH --output=/scratch/yx3882/research_dagger_demo_20260823/logs/demo_%j.out
#SBATCH --error=/scratch/yx3882/research_dagger_demo_20260823/logs/demo_%j.err
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=rtx6000
#SBATCH --constraint=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"

# Research-only BC0 vs DAgger demonstration.  This script deliberately omits
# release receipts, source hashes, exact-commit gates, and production promotion
# checks.  It reuses completed adapters and evaluates both in one allocation.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/research_dagger_src_20260823}
RUN_ROOT=${RUN_ROOT:-/scratch/yx3882/research_dagger_demo_20260823}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
SUITE=${SUITE:-/scratch/yx3882/dagger_release_a5a7574_20260823/dagger1_a5a7574/recovery_stress.json}
BC0_ADAPTER=${BC0_ADAPTER:-/scratch/yx3882/prelim_e2b_fullnatural_2b32029_20260822/output/bc0/lora}
DAGGER_ADAPTER=${DAGGER_ADAPTER:-/scratch/yx3882/prelim_e2b_fullnatural_2b32029_20260822/output/dagger/lora}
PER_SUITE=${PER_SUITE:-2}
MAX_STEPS=${MAX_STEPS:-12}
RESEARCH_MAX_INPUT_TOKENS=${RESEARCH_MAX_INPUT_TOKENS:-16384}
RESEARCH_MAX_NEW_TOKENS=${RESEARCH_MAX_NEW_TOKENS:-256}

SUBSET=$RUN_ROOT/recovery_stress_small.json
BC0_OUTPUT=$RUN_ROOT/bc0_eval.json
DAGGER_OUTPUT=$RUN_ROOT/dagger_eval.json
SUMMARY=$RUN_ROOT/comparison.json

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

mkdir -p "$RUN_ROOT/logs"
test -x "$PYTHON"
test -s "$REPO_ROOT/psse_env/dagger/preliminary_e2b_eval.py"
test -s "$REPO_ROOT/scripts/research_dagger_demo.py"
test -s "$SUITE"
test -s "$BC0_ADAPTER/adapter_model.safetensors"
test -s "$DAGGER_ADAPTER/adapter_model.safetensors"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT"
export HF_HOME=${HF_HOME:-/scratch/yx3882/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yx3882/.cache/torch}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RESEARCH_MAX_INPUT_TOKENS
export RESEARCH_MAX_NEW_TOKENS
export RESEARCH_RELAXED_ADAPTER_IDENTITY=1

cd "$REPO_ROOT"

"$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" subset \
    --input "$SUITE" \
    --output "$SUBSET" \
    --per-suite "$PER_SUITE"

required_suites=(
    recovery_measurement_parameter_sequential_handoff
    recovery_post_failure_no_candidate
    recovery_premature_commit
    recovery_premature_escalation
    recovery_rejected_candidate_rollback
    recovery_safe_continuation_after_partial_success
    recovery_unsupported_correction
)

run_eval() {
    local label=$1
    local adapter=$2
    local output=$3
    if [[ -s "$output" ]]; then
        echo "[$label] completed output exists; skipping: $output"
        return 0
    fi
    local revision
    revision=$(sha256sum "$adapter/adapter_model.safetensors" | awk '{print $1}')
    local -a suite_args=()
    local suite
    for suite in "${required_suites[@]}"; do
        suite_args+=(--required-suite "$suite")
    done
    echo "[$label] evaluating adapter=$adapter max_input=$RESEARCH_MAX_INPUT_TOKENS max_new=$RESEARCH_MAX_NEW_TOKENS"
    "$PYTHON" -m psse_env.dagger.evaluator \
        --input "$SUBSET" \
        --output "$output" \
        --env-factory psse_env.dagger.release_factories:production_environment_factory \
        --policy-factory psse_env.dagger.preliminary_e2b_eval:preliminary_e2b_policy_factory \
        --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
        --model-id "$adapter" \
        --model-revision "$revision" \
        --protocol canonical \
        --diagnostic-only \
        --allow-dirty-source \
        --seed 20260823 \
        --max-steps "$MAX_STEPS" \
        "${suite_args[@]}" \
        --minimum-suites 7 \
        --minimum-episodes-per-suite "$PER_SUITE" \
        --minimum-roots-per-suite 1
}

run_eval bc0 "$BC0_ADAPTER" "$BC0_OUTPUT"
run_eval dagger "$DAGGER_ADAPTER" "$DAGGER_OUTPUT"

"$PYTHON" "$REPO_ROOT/scripts/research_dagger_demo.py" summary \
    --bc0 "$BC0_OUTPUT" \
    --dagger "$DAGGER_OUTPUT" \
    --output "$SUMMARY"

echo "RESEARCH_DAGGER_DEMO_COMPLETE"
