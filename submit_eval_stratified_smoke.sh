#!/bin/bash
#SBATCH --job-name=gpt_oss_strat_smoke
#SBATCH --output=/scratch/yx3882/psse_agent/logs/eval_strat_smoke_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/eval_strat_smoke_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --comment=preemption=yes;requeue=true
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

ADAPTER_PATH=${ADAPTER_PATH:-harshith0214/psse-agent-gpt-oss-20b}
SOURCE_TEST_FILE=${SOURCE_TEST_FILE:-out_traces_balanced/sft_traces.test.jsonl}
PER_FAMILY=${PER_FAMILY:-4}
SEED=${SEED:-13}
MAX_TURNS=${MAX_TURNS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
SMOKE_TEST_FILE=${SMOKE_TEST_FILE:-outputs/stratified_smoke_${SLURM_JOB_ID}.jsonl}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gpt_oss_sft_power_agent/eval_stratified_smoke_${SLURM_JOB_ID}.jsonl}

cd "$REPO_ROOT"

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
echo "===== Run fixed eval on stratified smoke set ====="
echo "adapter: $ADAPTER_PATH"
echo "eval output: $OUTPUT_FILE"
"$PYTHON" eval_sft_agent_fixed.py \
  --adapter "$ADAPTER_PATH" \
  --test-file "$SMOKE_TEST_FILE" \
  --max-turns "$MAX_TURNS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --output "$OUTPUT_FILE" \
  --verbose
