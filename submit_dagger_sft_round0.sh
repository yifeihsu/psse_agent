#!/bin/bash
#SBATCH --job-name=dagger_sft_round0
#SBATCH --output=/scratch/yx3882/psse_agent/artifacts/logs/dagger_sft_%x_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/artifacts/logs/dagger_sft_%x_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu

# Staged launcher for the round-0 recovery-balanced DAgger aggregate
# (data/round0_aggregate_20260719, canonical protocol, 2273 chat rows).
# Submit STAGE=gate, one-batch, tiny-overfit, and round0 in that order on a
# high-memory GPU (--constraint=h200). STAGE=round0 trains the checkpoint
# whose held-out recovery evaluation gates any further stage; full
# production SFT remains refused here.
#
# The aggregate rows carry the complete canonical tool schema block, so
# prepared prompts run to ~5.3k tokens: MAX_LENGTH is 6144 (validated by the
# local gate run; 4096 would truncate ~35% of prompts).

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/psse_agent}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/dagger_gemma4_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
STAGE=${STAGE:-gate}
ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD:-0}

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-31B-it}
MODEL_REVISION=${MODEL_REVISION:-8a796db4df380b178065ed910849477ff0e99c87}
AGGREGATE_DIR=${AGGREGATE_DIR:-data/round0_aggregate_20260719}
TRAIN_FILE=${TRAIN_FILE:-$AGGREGATE_DIR/aggregate.train.jsonl}
VALIDATION_FILE=${VALIDATION_FILE:-$AGGREGATE_DIR/aggregate.validation.jsonl}
TEST_FILE=${TEST_FILE:-$AGGREGATE_DIR/aggregate.test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/yx3882/psse_agent/outputs/dagger_gemma4_31b_round0}
MAX_LENGTH=${MAX_LENGTH:-6144}
ROWS_MIN=${ROWS_MIN:-1024}
ROWS_MAX=${ROWS_MAX:-4096}
TINY_OVERFIT_STEPS=${TINY_OVERFIT_STEPS:-20}
TINY_OVERFIT_LR=${TINY_OVERFIT_LR:-0.001}
TRAIN_LR=${TRAIN_LR:-0.0001}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}

case "$STAGE" in
    gate|one-batch|tiny-overfit|round0)
        ;;
    full|production)
        echo "ERROR: full production SFT is blocked pending the held-out recovery evaluation of the round-0 checkpoint." >&2
        exit 2
        ;;
    *)
        echo "ERROR: STAGE must be gate, one-batch, tiny-overfit, or round0; got '$STAGE'." >&2
        exit 2
        ;;
esac

if [[ ! "$MODEL_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: MODEL_REVISION must be a pinned 40-character Hugging Face commit." >&2
    exit 2
fi

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

cd "$REPO_ROOT"
mkdir -p artifacts/logs "$OUTPUT_DIR"
mkdir -p /scratch/yx3882/.cache/huggingface /scratch/yx3882/.cache/torch

export HF_HOME=${HF_HOME:-/scratch/yx3882/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yx3882/.cache/torch}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found at $PYTHON; create it from psse_env/requirements-sft.txt first." >&2
    exit 2
fi
for path in "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required aggregate split not found: $path" >&2
        exit 2
    fi
done
if [[ -f "$AGGREGATE_DIR/SHA256SUMS" ]]; then
    (cd "$AGGREGATE_DIR" && sha256sum --check --quiet SHA256SUMS) || {
        echo "ERROR: aggregate split checksums do not match SHA256SUMS; re-ship the aggregate." >&2
        exit 2
    }
fi

echo "===== DAgger Gemma 4 round-0 stage ====="
echo "job:       ${SLURM_JOB_ID:-interactive}"
echo "host:      $(hostname)"
echo "stage:     $STAGE"
echo "python:    $PYTHON"
echo "model:     $MODEL_NAME"
echo "revision:  $MODEL_REVISION"
echo "train:     $TRAIN_FILE"
echo "output:    $OUTPUT_DIR"
echo "downloads: $ALLOW_DOWNLOAD"
nvidia-smi
"$PYTHON" -c 'import accelerate, bitsandbytes, datasets, peft, torch, transformers, trl; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "bf16": torch.cuda.is_bf16_supported(), "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__, "datasets": datasets.__version__, "accelerate": accelerate.__version__})'

COMMON_ARGS=(
    --model "$MODEL_NAME"
    --revision "$MODEL_REVISION"
    --train "$TRAIN_FILE"
    --validation "$VALIDATION_FILE"
    --max-length "$MAX_LENGTH"
    --pilot-min-rows "$ROWS_MIN"
    --pilot-max-rows "$ROWS_MAX"
)
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
    COMMON_ARGS+=(--allow-download)
elif [[ "$ALLOW_DOWNLOAD" != "0" ]]; then
    echo "ERROR: ALLOW_DOWNLOAD must be 0 or 1." >&2
    exit 2
fi

case "$STAGE" in
    gate)
        COMMAND=("$PYTHON" -m psse_env.sft gate "${COMMON_ARGS[@]}" --test "$TEST_FILE")
        ;;
    one-batch)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" --mode one-batch --load-in-4bit)
        ;;
    tiny-overfit)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" --mode tiny-overfit --tiny-overfit-steps "$TINY_OVERFIT_STEPS" --learning-rate "$TINY_OVERFIT_LR" --load-in-4bit)
        ;;
    round0)
        COMMAND=("$PYTHON" -m psse_env.sft train "${COMMON_ARGS[@]}" --output-dir "$OUTPUT_DIR" --batch-size 1 --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --learning-rate "$TRAIN_LR" --epochs "$TRAIN_EPOCHS" --smoke-steps 1 --load-in-4bit)
        ;;
esac

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
