#!/bin/bash
#SBATCH --job-name=gpt_oss_eval
#SBATCH --output=/scratch/yx3882/psse_agent/logs/eval_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/eval_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
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

ADAPTER_PATH=${ADAPTER_PATH:-outputs/gpt_oss_sft_power_agent_4k/lora}
TRACE_FILE=${TRACE_FILE:-tmp_show_trace.json}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}
INITIAL_MESSAGES=${INITIAL_MESSAGES:-2}
MAX_STEPS=${MAX_STEPS:-6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
REASONING_EFFORT=${REASONING_EFFORT:-low}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gpt_oss_sft_power_agent_4k/eval_${SLURM_JOB_ID}.json}
SHOW_REFERENCE=${SHOW_REFERENCE:-1}

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
echo "trace: $TRACE_FILE"
echo "output: $OUTPUT_FILE"
$PYTHON -V
$PYTHON -c "import sys; print('sys.executable =', sys.executable)"
$PYTHON -m pip show unsloth scipy || true
nvidia-smi
echo "============================"

ARGS=(
  interactive_agent_eval.py
  --adapter-path "$ADAPTER_PATH"
  --trace-file "$TRACE_FILE"
  --sample-index "$SAMPLE_INDEX"
  --initial-messages "$INITIAL_MESSAGES"
  --max-steps "$MAX_STEPS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --reasoning-effort "$REASONING_EFFORT"
  --output-file "$OUTPUT_FILE"
)

if [[ "$SHOW_REFERENCE" == "1" ]]; then
  ARGS+=(--show-reference)
fi

"$PYTHON" "${ARGS[@]}"
