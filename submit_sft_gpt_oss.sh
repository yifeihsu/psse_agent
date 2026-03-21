#!/bin/bash
#SBATCH --job-name=gpt_oss_sft
#SBATCH --output=/scratch/hk4488/psse_agent/logs/sft_%j.log
#SBATCH --error=/scratch/hk4488/psse_agent/logs/sft_%j.err
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
OUTPUT_DIR=/scratch/hk4488/psse_agent/outputs/gpt_oss_sft

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
export WANDB_API_KEY="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"

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "===== Job diagnostics ====="
echo "Job ID  : $SLURM_JOB_ID"
echo "Host    : $(hostname)"
echo "Python  : $PYTHON"
$PYTHON -V
nvidia-smi
echo "==========================="

# ── Train ──────────────────────────────────────────────────────────────────
$PYTHON sft_gpt_oss.py \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length 16384 \
    --dataset-num-proc 1 \
    --load-in-4bit \
    --lora-r 64 \
    --lora-alpha 64 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --warmup-steps 20 \
    --num-train-epochs 1 \
    --logging-steps 5 \
    --save-steps 100 \
    --eval-steps 100 \
    --report-to wandb
