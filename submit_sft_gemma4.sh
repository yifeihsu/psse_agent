#!/bin/bash
#SBATCH --job-name=gemma4_sft
#SBATCH --output=/scratch/yx3882/psse_agent/logs/sft_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/sft_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --account=torch_pr_627_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

LOG_DIR=/scratch/yx3882/psse_agent/logs
OUTPUT_DIR=/scratch/yx3882/psse_agent/outputs/gemma4_power_agent
TRAIN_FILE=${TRAIN_FILE:-out_traces_balanced/sft_traces.train.jsonl}
VALID_FILE=${VALID_FILE:-out_traces_balanced/sft_traces.valid.jsonl}
MODEL_NAME=${MODEL_NAME:-unsloth/Gemma-4-26B-A4B-it}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-4}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-auto}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
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
echo "Output  : $OUTPUT_DIR"
echo "Resume  : $RESUME_FROM_CHECKPOINT"
echo "Save/Eval steps: $SAVE_STEPS / $EVAL_STEPS"
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
    --load-in-16bit \
    --include-tool-schemas \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate 2e-4 \
    --warmup-steps 20 \
    --num-train-epochs "$NUM_TRAIN_EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --logging-steps 5 \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --resume-from-checkpoint "$RESUME_FROM_CHECKPOINT" \
    --report-to wandb \
    $EXTRA_TRAIN_ARGS
