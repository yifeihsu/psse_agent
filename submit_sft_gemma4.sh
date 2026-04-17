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
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --account=torch_pr_627_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

LOG_DIR=/scratch/yx3882/psse_agent/logs
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/yx3882/psse_agent/outputs/gemma4_power_agent}
TRAIN_FILE=${TRAIN_FILE:-out_traces_balanced/sft_traces.train.jsonl}
VALID_FILE=${VALID_FILE:-out_traces_balanced/sft_traces.valid.jsonl}
MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-26B-A4B-it}
MODEL_REVISION=${MODEL_REVISION:-}
ALLOW_UNPINNED_MODEL_REVISION=${ALLOW_UNPINNED_MODEL_REVISION:-0}
GPU_PROFILE=${GPU_PROFILE:-auto}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-}
EVAL_STEPS=${EVAL_STEPS:-}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-auto}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-}
LORA_R=${LORA_R:-}
LORA_ALPHA=${LORA_ALPHA:-}
LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-language_model}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-}
LEARNING_RATE=${LEARNING_RATE:-}
WARMUP_STEPS=${WARMUP_STEPS:-20}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SANITY_CHECK_SAMPLES=${SANITY_CHECK_SAMPLES:-3}
SANITY_CHECK_MAX_NEW_TOKENS=${SANITY_CHECK_MAX_NEW_TOKENS:-128}
SANITY_CHECK_FAIL_ON_MISS=${SANITY_CHECK_FAIL_ON_MISS:-0}
PHASE_GATED_PROMPT=${PHASE_GATED_PROMPT:-1}
INCLUDE_TOOL_SCHEMAS=${INCLUDE_TOOL_SCHEMAS:-1}
INJECT_EMPTY_THOUGHT_CHANNEL=${INJECT_EMPTY_THOUGHT_CHANNEL:-1}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
PREEMPTION_EXIT_CODE=99

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p /scratch/yx3882/.cache/huggingface
mkdir -p /scratch/yx3882/.cache/wandb
mkdir -p /scratch/yx3882/.config/wandb
mkdir -p /scratch/yx3882/.local/share/wandb
mkdir -p /scratch/yx3882/.cache/torch
mkdir -p /scratch/yx3882/psse_agent/wandb

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export HF_HOME=/scratch/yx3882/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/yx3882/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/yx3882/.cache/huggingface/datasets
export TORCH_HOME=/scratch/yx3882/.cache/torch
export XDG_CACHE_HOME=/scratch/yx3882/.cache
export XDG_CONFIG_HOME=/scratch/yx3882/.config
export WANDB_DIR=/scratch/yx3882/psse_agent/wandb
export WANDB_CACHE_DIR=/scratch/yx3882/.cache/wandb
export WANDB_CONFIG_DIR=/scratch/yx3882/.config/wandb
export WANDB_DATA_DIR=/scratch/yx3882/.local/share/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-psse-agent-sft}
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
fi

set_default_if_unset() {
    local var_name=$1
    local default_value=$2
    if [[ -z "${!var_name:-}" ]]; then
        printf -v "$var_name" '%s' "$default_value"
    fi
}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
GPU_PROFILE_SELECTED=$GPU_PROFILE
if [[ "$GPU_PROFILE" == "auto" ]]; then
    if [[ ( "$GPU_NAME" == *"A100"* || "$GPU_NAME" == *"H100"* || "$GPU_NAME" == *"H200"* ) && "${GPU_MEM_MB:-0}" -ge 70000 ]]; then
        GPU_PROFILE_SELECTED="highmem-accelerator"
        set_default_if_unset MAX_SEQ_LENGTH 6144
        set_default_if_unset PER_DEVICE_TRAIN_BATCH_SIZE 2
        set_default_if_unset GRADIENT_ACCUMULATION_STEPS 8
        set_default_if_unset LORA_R 32
        set_default_if_unset LORA_ALPHA 32
        set_default_if_unset SAVE_STEPS 25
        set_default_if_unset EVAL_STEPS 100
        set_default_if_unset SAVE_TOTAL_LIMIT 8
        set_default_if_unset DATALOADER_NUM_WORKERS 8
        set_default_if_unset LEARNING_RATE 1.5e-4
    else
        GPU_PROFILE_SELECTED="a100-safe"
        set_default_if_unset MAX_SEQ_LENGTH 4096
        set_default_if_unset PER_DEVICE_TRAIN_BATCH_SIZE 1
        set_default_if_unset GRADIENT_ACCUMULATION_STEPS 8
        set_default_if_unset LORA_R 16
        set_default_if_unset LORA_ALPHA 16
        set_default_if_unset SAVE_STEPS 25
        set_default_if_unset EVAL_STEPS 100
        set_default_if_unset SAVE_TOTAL_LIMIT 8
        set_default_if_unset DATALOADER_NUM_WORKERS 4
        set_default_if_unset LEARNING_RATE 2e-4
    fi
else
    set_default_if_unset MAX_SEQ_LENGTH 4096
    set_default_if_unset PER_DEVICE_TRAIN_BATCH_SIZE 1
    set_default_if_unset GRADIENT_ACCUMULATION_STEPS 8
    set_default_if_unset LORA_R 16
    set_default_if_unset LORA_ALPHA 16
    set_default_if_unset SAVE_STEPS 25
    set_default_if_unset EVAL_STEPS 100
    set_default_if_unset SAVE_TOTAL_LIMIT 8
    set_default_if_unset DATALOADER_NUM_WORKERS 4
    set_default_if_unset LEARNING_RATE 2e-4
fi

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "===== Job diagnostics ====="
echo "Job ID  : $SLURM_JOB_ID"
echo "Restart : ${SLURM_RESTART_COUNT:-0}"
echo "Host    : $(hostname)"
echo "Python  : $PYTHON"
echo "GPU     : $GPU_NAME (${GPU_MEM_MB:-unknown} MiB)"
echo "Profile : $GPU_PROFILE_SELECTED"
echo "Output  : $OUTPUT_DIR"
echo "Model   : $MODEL_NAME"
echo "Revision: ${MODEL_REVISION:-<unpinned>}"
echo "Resume  : $RESUME_FROM_CHECKPOINT"
echo "Save/Eval steps: $SAVE_STEPS / $EVAL_STEPS"
echo "Hyperparams: seq=$MAX_SEQ_LENGTH bs=$PER_DEVICE_TRAIN_BATCH_SIZE ga=$GRADIENT_ACCUMULATION_STEPS lora_r=$LORA_R lora_alpha=$LORA_ALPHA lora_scope=$LORA_TARGET_SCOPE lr=$LEARNING_RATE workers=$DATALOADER_NUM_WORKERS sanity=$SANITY_CHECK_SAMPLES phase_gated=$PHASE_GATED_PROMPT tool_schemas=$INCLUDE_TOOL_SCHEMAS empty_thought=$INJECT_EMPTY_THOUGHT_CHANNEL"
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

TRAIN_PID=""
forward_signal() {
    local sig=$1
    echo "[signal] Received $sig in batch launcher at $(date --iso-8601=seconds)"
    if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[signal] Forwarding $sig to trainer process $TRAIN_PID"
        kill -s "$sig" "$TRAIN_PID" || true
    fi
}

trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

# ── Train ──────────────────────────────────────────────────────────────────
if [[ -z "$MODEL_REVISION" && "$ALLOW_UNPINNED_MODEL_REVISION" != "1" ]]; then
    echo "ERROR: MODEL_REVISION is required by gpt_oss_power_sft_revised_v2.py to pin the Gemma 4 chat template." >&2
    echo "Set MODEL_REVISION=<hf commit/tag> before sbatch, or set ALLOW_UNPINNED_MODEL_REVISION=1 to opt into floating upstream behavior." >&2
    exit 2
fi

SANITY_FAIL_ARGS=()
if [[ "$SANITY_CHECK_FAIL_ON_MISS" != "0" ]]; then
    SANITY_FAIL_ARGS+=(--sanity-check-fail-on-miss)
fi
PHASE_GATED_ARGS=()
if [[ "$PHASE_GATED_PROMPT" == "0" ]]; then
    PHASE_GATED_ARGS+=(--no-phase-gated-prompt)
fi
MODEL_REVISION_ARGS=()
if [[ -n "$MODEL_REVISION" ]]; then
    MODEL_REVISION_ARGS+=(--model-revision "$MODEL_REVISION")
elif [[ "$ALLOW_UNPINNED_MODEL_REVISION" == "1" ]]; then
    MODEL_REVISION_ARGS+=(--allow-unpinned-model-revision)
fi
TOOL_SCHEMA_ARGS=()
if [[ "$INCLUDE_TOOL_SCHEMAS" == "0" ]]; then
    TOOL_SCHEMA_ARGS+=(--no-include-tool-schemas)
else
    TOOL_SCHEMA_ARGS+=(--include-tool-schemas)
fi
EMPTY_THOUGHT_ARGS=()
if [[ "$INJECT_EMPTY_THOUGHT_CHANNEL" == "0" ]]; then
    EMPTY_THOUGHT_ARGS+=(--no-inject-empty-thought-channel)
else
    EMPTY_THOUGHT_ARGS+=(--inject-empty-thought-channel)
fi

$PYTHON gpt_oss_power_sft_revised_v2.py \
    --train-file "$TRAIN_FILE" \
    --valid-file "$VALID_FILE" \
    --model-name "$MODEL_NAME" \
    "${MODEL_REVISION_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --dataset-num-proc 1 \
    --load-in-16bit \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-target-scope "$LORA_TARGET_SCOPE" \
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-steps "$WARMUP_STEPS" \
    --num-train-epochs "$NUM_TRAIN_EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --logging-steps "$LOGGING_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --resume-from-checkpoint "$RESUME_FROM_CHECKPOINT" \
    --sanity-check-samples "$SANITY_CHECK_SAMPLES" \
    --sanity-check-max-new-tokens "$SANITY_CHECK_MAX_NEW_TOKENS" \
    "${TOOL_SCHEMA_ARGS[@]}" \
    "${EMPTY_THOUGHT_ARGS[@]}" \
    "${SANITY_FAIL_ARGS[@]}" \
    "${PHASE_GATED_ARGS[@]}" \
    --report-to wandb \
    $EXTRA_TRAIN_ARGS &
TRAIN_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_EXIT=$?
set -e

if [[ "$TRAIN_EXIT" -eq "$PREEMPTION_EXIT_CODE" ]]; then
    echo "[requeue] Trainer exited with checkpoint/requeue code $PREEMPTION_EXIT_CODE"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scontrol requeue "$SLURM_JOB_ID" || true
    fi
    exit 0
fi

exit "$TRAIN_EXIT"
