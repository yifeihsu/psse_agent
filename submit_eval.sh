#!/bin/bash
#SBATCH --job-name=gpt_oss_eval
#SBATCH --output=/scratch/yx3882/psse_agent/logs/eval_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/eval_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

REPO_ROOT=/scratch/yx3882/psse_agent
LOG_DIR=$REPO_ROOT/logs
CACHE_ROOT=/scratch/yx3882/.cache

ADAPTER_PATH=${ADAPTER_PATH:-outputs/gpt_oss_sft_power_agent/lora}
TEST_FILE=${TEST_FILE:-out_traces_balanced/sft_traces.test.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_TURNS=${MAX_TURNS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12000}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gpt_oss_sft_power_agent/eval_${SLURM_JOB_ID}.jsonl}
VERBOSE=${VERBOSE:-1}

mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_ROOT/huggingface"
mkdir -p "$CACHE_ROOT/torch"

export HF_HOME=$CACHE_ROOT/huggingface
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$REPO_ROOT"

echo "===== Eval diagnostics ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "env python: $PYTHON"
echo "adapter: $ADAPTER_PATH"
echo "test file: $TEST_FILE"
echo "output: $OUTPUT_FILE"
$PYTHON -V
$PYTHON -m pip list | grep -E "unsloth|scipy|transformers|torch" || true
nvidia-smi
echo "============================"

ARGS=(
  eval_sft_agent_revised.py
  --adapter "$ADAPTER_PATH"
  --test-file "$TEST_FILE"
  --max-turns "$MAX_TURNS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --output "$OUTPUT_FILE"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$VERBOSE" == "1" ]]; then
  ARGS+=(--verbose)
fi

"$PYTHON" "${ARGS[@]}"
