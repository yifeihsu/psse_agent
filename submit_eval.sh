#!/bin/bash
#SBATCH --job-name=psse_eval
#SBATCH --output=/scratch/hk4488/psse_agent/logs/eval_%j.log
#SBATCH --error=/scratch/hk4488/psse_agent/logs/eval_%j.err
#SBATCH --chdir=/home/hk4488/psse_agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_627_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4488@nyu.edu
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ENV_PREFIX=/scratch/hk4488/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

ADAPTER_PATH=/scratch/hk4488/psse_agent/outputs/gpt_oss_sft_v2/lora
TEST_FILE=/home/hk4488/psse_agent/data/split_test.jsonl
OUTPUT_JSON=/scratch/hk4488/psse_agent/outputs/eval_results_v2.json
CKPT_FILE=/scratch/hk4488/psse_agent/outputs/eval_checkpoint_v2.jsonl

mkdir -p /scratch/hk4488/psse_agent/logs
mkdir -p /scratch/hk4488/psse_agent/outputs

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export HF_HOME=/scratch/hk4488/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/hk4488/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/hk4488/.cache/huggingface/datasets
export TORCH_HOME=/scratch/hk4488/.cache/torch
export WANDB_PROJECT="psse-agent-sft"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load WANDB key from file
if [ -f /home/hk4488/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /home/hk4488/.wandb_api_key)
fi
# WANDB_API_KEY loaded from ~/.wandb_api_key (see submit scripts)

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "===== Job diagnostics ====="
echo "Job ID  : $SLURM_JOB_ID"
echo "Host    : $(hostname)"
echo "Python  : $PYTHON"
$PYTHON -V
nvidia-smi
echo "==========================="

# ── Evaluate ───────────────────────────────────────────────────────────────
$PYTHON eval_agent.py \
    --adapter-path    "$ADAPTER_PATH" \
    --test-file       "$TEST_FILE"    \
    --output-json     "$OUTPUT_JSON"  \
    --checkpoint-file "$CKPT_FILE"    \
    --max-new-tokens  512             \
    --max-seq-length  16384           \
    --load-in-4bit                    \
    --wandb

echo "Evaluation complete. Results at $OUTPUT_JSON"
