#!/bin/bash
#SBATCH --job-name=gpt_oss_sft
#SBATCH --output=/scratch/yx3882/psse_agent/logs/sft_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/sft_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_627_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

LOG_DIR=/scratch/yx3882/psse_agent/logs
OUTPUT_DIR=/scratch/yx3882/psse_agent/outputs/gpt_oss_sft
TRAIN_FILE=${TRAIN_FILE:-out_traces_balanced/sft_traces.train.jsonl}
VALID_FILE=${VALID_FILE:-out_traces_balanced/sft_traces.valid.jsonl}
MODEL_NAME=${MODEL_NAME:-unsloth/gpt-oss-20b-unsloth-bnb-4bit}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-12288}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p /scratch/yx3882/.cache/huggingface
mkdir -p /scratch/yx3882/.cache/torch

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export HF_HOME=/scratch/yx3882/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/yx3882/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/yx3882/.cache/huggingface/datasets
export TORCH_HOME=/scratch/yx3882/.cache/torch
export WANDB_PROJECT=${WANDB_PROJECT:-psse-agent-sft}
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
fi

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "===== Job diagnostics ====="
echo "Job ID  : $SLURM_JOB_ID"
echo "Host    : $(hostname)"
echo "Python  : $PYTHON"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "WandB   : using WANDB_API_KEY from environment"
elif [[ -f "$HOME/.netrc" ]]; then
    echo "WandB   : using existing login from \$HOME/.netrc"
else
    echo "WandB   : no login detected; run 'wandb login' or export WANDB_API_KEY before sbatch"
fi
$PYTHON -V
nvidia-smi
echo "==========================="

# ── Train ──────────────────────────────────────────────────────────────────
$PYTHON gpt_oss_power_sft_revised.py \
    --train-file "$TRAIN_FILE" \
    --valid-file "$VALID_FILE" \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --dataset-num-proc 1 \
    --load-in-4bit \
    --include-tool-schemas \
    --lora-r 64 \
    --lora-alpha 64 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --warmup-steps 20 \
    --num-train-epochs "$NUM_TRAIN_EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --logging-steps 5 \
    --save-steps 100 \
    --eval-steps 100 \
    --report-to wandb \
    $EXTRA_TRAIN_ARGS
