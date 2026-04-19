#!/bin/bash
#SBATCH --job-name=gemma4_strat_smoke
#SBATCH --output=/scratch/yx3882/psse_agent/logs/eval_strat_smoke_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/eval_strat_smoke_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --comment=preemption=no;requeue=false
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

REPO_ROOT=/scratch/yx3882/psse_agent
CACHE_ROOT=/scratch/yx3882/.cache
LOG_DIR=$REPO_ROOT/logs
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"

export HF_HOME=$CACHE_ROOT/huggingface
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

ADAPTER_PATH=${ADAPTER_PATH:-outputs/gemma4_power_agent/lora}
SOURCE_TEST_FILE=${SOURCE_TEST_FILE:-out_traces_balanced/sft_traces.test.jsonl}
MODEL_REVISION=${MODEL_REVISION:-d722512f8f1e4ef6629c1b24d16d65295c8c945e}
PER_FAMILY=${PER_FAMILY:-4}
SEED=${SEED:-13}
MAX_TURNS=${MAX_TURNS:-6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
INCLUDE_TOOL_SCHEMAS=${INCLUDE_TOOL_SCHEMAS:-1}
INJECT_EMPTY_THOUGHT_CHANNEL=${INJECT_EMPTY_THOUGHT_CHANNEL:-1}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-0}
LOAD_IN_16BIT=${LOAD_IN_16BIT:-1}
SMOKE_TEST_FILE=${SMOKE_TEST_FILE:-outputs/stratified_smoke_${SLURM_JOB_ID}.jsonl}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gemma4_power_agent/eval_stratified_smoke_${SLURM_JOB_ID}.jsonl}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$SMOKE_TEST_FILE")" "$(dirname "$OUTPUT_FILE")"

echo "===== Build stratified smoke set ====="
echo "source test file: $SOURCE_TEST_FILE"
echo "per family: $PER_FAMILY"
echo "seed: $SEED"
echo "smoke subset: $SMOKE_TEST_FILE"
"$PYTHON" make_stratified_smoke.py \
  --input "$SOURCE_TEST_FILE" \
  --output "$SMOKE_TEST_FILE" \
  --per-family "$PER_FAMILY" \
  --seed "$SEED" \
  --shuffle-output

echo
echo "===== Run Gemma v2 eval on stratified smoke set ====="
echo "adapter: $ADAPTER_PATH"
echo "model revision: ${MODEL_REVISION:-UNPINNED}"
echo "eval output: $OUTPUT_FILE"

ARGS=(
  eval_sft_agent_gemma_v2.py
  --adapter "$ADAPTER_PATH"
  --test-file "$SMOKE_TEST_FILE"
  --max-turns "$MAX_TURNS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --output "$OUTPUT_FILE"
  --verbose
)

if [[ -n "$MODEL_REVISION" ]]; then
  ARGS+=(--model-revision "$MODEL_REVISION")
fi

if [[ "$INCLUDE_TOOL_SCHEMAS" == "1" ]]; then
  ARGS+=(--include-tool-schemas)
else
  ARGS+=(--no-include-tool-schemas)
fi

if [[ "$INJECT_EMPTY_THOUGHT_CHANNEL" == "1" ]]; then
  ARGS+=(--inject-empty-thought-channel)
else
  ARGS+=(--no-inject-empty-thought-channel)
fi

if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  ARGS+=(--load-in-4bit)
else
  ARGS+=(--no-load-in-4bit)
fi

if [[ "$LOAD_IN_16BIT" == "1" ]]; then
  ARGS+=(--load-in-16bit)
else
  ARGS+=(--no-load-in-16bit)
fi

"$PYTHON" "${ARGS[@]}"
