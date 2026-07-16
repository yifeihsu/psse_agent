#!/bin/bash
#SBATCH --job-name=dagger_sft_pilot
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

# Staged launcher for the corrected 90-row DAgger pilot. This intentionally
# refuses a production/full stage: the recovery-balanced aggregate and its
# held-out evaluation do not exist yet.
#
# Before the first submission:
#   mkdir -p /scratch/yx3882/psse_agent/artifacts/logs
#
# Submit STAGE=gate, one-batch, tiny-overfit, and pilot in that order. Select a
# high-memory GPU at submission time, for example --constraint=h200.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/psse_agent}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/dagger_gemma4_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
STAGE=${STAGE:-gate}
ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD:-0}

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-31B-it}
MODEL_REVISION=${MODEL_REVISION:-8a796db4df380b178065ed910849477ff0e99c87}
TRAIN_FILE=${TRAIN_FILE:-psse_env/examples/sft_pilot/pilot.train.jsonl}
VALIDATION_FILE=${VALIDATION_FILE:-psse_env/examples/sft_pilot/pilot.validation.jsonl}
TEST_FILE=${TEST_FILE:-psse_env/examples/sft_pilot/pilot.test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/yx3882/psse_agent/outputs/dagger_gemma4_31b_pilot}
MAX_LENGTH=${MAX_LENGTH:-4096}
TINY_OVERFIT_STEPS=${TINY_OVERFIT_STEPS:-20}
TINY_OVERFIT_LR=${TINY_OVERFIT_LR:-0.001}
TRAIN_LR=${TRAIN_LR:-0.0001}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}

case "$STAGE" in
    gate|one-batch|tiny-overfit|pilot)
        ;;
    full|production)
        echo "ERROR: full production SFT is blocked pending a recovery-balanced aggregate and held-out recovery evaluation." >&2
        exit 2
        ;;
    *)
        echo "ERROR: STAGE must be gate, one-batch, tiny-overfit, or pilot; got '$STAGE'." >&2
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
        echo "ERROR: required pilot split not found: $path" >&2
        exit 2
    fi
done

echo "===== DAgger Gemma 4 pilot stage ====="
echo "job:       ${SLURM_JOB_ID:-interactive}"
echo "host:      $(hostname)"
echo "stage:     $STAGE"
echo "python:    $PYTHON"
echo "model:     $MODEL_NAME"
echo "revision:  $MODEL_REVISION"
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
    pilot)
        COMMAND=("$PYTHON" -m psse_env.sft train "${COMMON_ARGS[@]}" --output-dir "$OUTPUT_DIR" --batch-size 1 --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --learning-rate "$TRAIN_LR" --epochs "$TRAIN_EPOCHS" --smoke-steps 1 --load-in-4bit)
        ;;
esac

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
