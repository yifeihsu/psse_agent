#!/bin/bash
#SBATCH --job-name=psse_sft_v2
#SBATCH --output=/scratch/hk4488/psse_agent/logs/sft_v2_%j.log
#SBATCH --error=/scratch/hk4488/psse_agent/logs/sft_v2_%j.err
#SBATCH --chdir=/home/hk4488/psse_agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_627_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4488@nyu.edu
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ENV_PREFIX=/scratch/hk4488/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

LOG_DIR=/scratch/hk4488/psse_agent/logs
OUTPUT_DIR=/scratch/hk4488/psse_agent/outputs/gpt_oss_sft_v2

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p /scratch/hk4488/.cache/huggingface
mkdir -p /scratch/hk4488/.cache/torch

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
# WANDB_API_KEY loaded from ~/.wandb_api_key (see submit scripts)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load WANDB key from file
if [ -f /home/hk4488/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /home/hk4488/.wandb_api_key)
fi

# ── Resume from checkpoint if one exists ──────────────────────────────────
RESUME_ARG=""
# Find the most recent checkpoint directory (|| true prevents set -e from
# aborting when ls finds no matching files on a fresh run)
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume-from-checkpoint $LATEST_CKPT"
else
    echo "No checkpoint found — starting fresh."
fi

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "===== Job diagnostics ====="
echo "Job ID  : $SLURM_JOB_ID"
echo "Host    : $(hostname)"
$PYTHON -V
nvidia-smi
echo "==========================="

# ── Train ──────────────────────────────────────────────────────────────────
$PYTHON sft_gpt_oss.py \
    --output-dir        "$OUTPUT_DIR"   \
    --max-seq-length    16384           \
    --dataset-num-proc  1               \
    --load-in-4bit                      \
    --lora-r            64              \
    --lora-alpha        64              \
    --per-device-train-batch-size  4    \
    --gradient-accumulation-steps  4    \
    --learning-rate     1e-4            \
    --warmup-steps      50              \
    --num-train-epochs  3               \
    --logging-steps     5               \
    --save-steps        50              \
    --eval-steps        50              \
    --save-total-limit  3               \
    --report-to         wandb           \
    $RESUME_ARG
