#!/bin/bash
#SBATCH --job-name=bc0_sft_round0
#SBATCH --output=/scratch/yx3882/psse_agent/artifacts/logs/bc0_sft_%x_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/artifacts/logs/bc0_sft_%x_%j.err
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

# Staged launcher for a newly generated, provenance-bound, canonical round-0
# recovery-balanced observable-expert BC aggregate.  The historical
# data/round0_aggregate_20260719 directory is intentionally not the default:
# it predates physical-root fingerprints, explicit eligibility, and current
# registry/source provenance and must fail the release gate.
# Submit STAGE=gate, one-batch, tiny-overfit, and round0 in that order on a
# high-memory GPU (--constraint="h200|h100|rtx6000"; on a 48GB rtx6000 the
# 6144-token batches leave little headroom — if the one-batch stage OOMs
# there, resubmit that job with an h200/h100-only constraint rather than
# shrinking MAX_LENGTH). STAGE=round0 refuses to train until the observable
# expert and exact pinned base model both pass the content-pinned fixed-suite
# evaluation gate. After training, STAGE=checkpoint-gate validates one exact
# checkpoint artifact before promotion; full production SFT remains refused
# here.
#
# MAX_LENGTH=6144 is a conservative starting envelope.  The exact pinned
# processor gate for the newly generated release aggregate remains decisive.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/psse_agent}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/dagger_gemma4_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
STAGE=${STAGE:-gate}
ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD:-0}

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-31B-it}
MODEL_REVISION=${MODEL_REVISION:-8a796db4df380b178065ed910849477ff0e99c87}
AGGREGATE_DIR=${AGGREGATE_DIR:-data/round0_aggregate_release}
TRAIN_FILE=${TRAIN_FILE:-$AGGREGATE_DIR/aggregate.train_view.jsonl}
VALIDATION_FILE=${VALIDATION_FILE:-$AGGREGATE_DIR/aggregate.validation.jsonl}
TEST_FILE=${TEST_FILE:-$AGGREGATE_DIR/aggregate.test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/yx3882/psse_agent/outputs/bc0_gemma4_31b_round0}
MAX_LENGTH=${MAX_LENGTH:-6144}
ROWS_MIN=${ROWS_MIN:-1024}
ROWS_MAX=${ROWS_MAX:-4096}
TINY_OVERFIT_STEPS=${TINY_OVERFIT_STEPS:-20}
TINY_OVERFIT_LR=${TINY_OVERFIT_LR:-0.001}
TRAIN_LR=${TRAIN_LR:-0.0001}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
EVALUATION_SUITE=${EVALUATION_SUITE:-artifacts/evaluations/bc0_frozen_suite.json}
EVALUATION_POLICY=${EVALUATION_POLICY:-psse_env/dagger/bc0_evaluation_policy.json}
EXPERT_BASELINE_EVALUATION=${EXPERT_BASELINE_EVALUATION:-artifacts/evaluations/expert_baseline_evaluation.json}
BASE_GEMMA_EVALUATION=${BASE_GEMMA_EVALUATION:-artifacts/evaluations/base_gemma_evaluation.json}
EXPERT_POLICY_IDENTITY=${EXPERT_POLICY_IDENTITY:-bc0-observable-expert-v1}
BASELINE_EVALUATION_REPORT=${BASELINE_EVALUATION_REPORT:-$OUTPUT_DIR/baseline_evaluation_gate.json}
CHECKPOINT_EVALUATION=${CHECKPOINT_EVALUATION:-}
CHECKPOINT_MODEL_ID=${CHECKPOINT_MODEL_ID:-}
CHECKPOINT_MODEL_REVISION=${CHECKPOINT_MODEL_REVISION:-}
CHECKPOINT_GATE_REPORT=${CHECKPOINT_GATE_REPORT:-$OUTPUT_DIR/checkpoint_promotion_gate.json}

case "$STAGE" in
    gate|one-batch|tiny-overfit|round0|checkpoint-gate)
        ;;
    full|production)
        echo "ERROR: full production SFT is blocked pending the held-out recovery evaluation of the round-0 checkpoint." >&2
        exit 2
        ;;
    *)
        echo "ERROR: STAGE must be gate, one-batch, tiny-overfit, round0, or checkpoint-gate; got '$STAGE'." >&2
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
if [[ "$STAGE" != "checkpoint-gate" ]]; then
    for path in "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: required aggregate split not found: $path" >&2
            exit 2
        fi
    done
    if [[ ! -f "$AGGREGATE_DIR/aggregate.generation_provenance.json" ]]; then
        echo "ERROR: release generation provenance is missing from $AGGREGATE_DIR; regenerate round 0 from the clean commit before submitting SFT." >&2
        exit 2
    fi
    if [[ -f "$AGGREGATE_DIR/SHA256SUMS" ]]; then
        (cd "$AGGREGATE_DIR" && sha256sum --check --quiet SHA256SUMS) || {
            echo "ERROR: aggregate split checksums do not match SHA256SUMS; re-ship the aggregate." >&2
            exit 2
        }
    fi
fi
if [[ "$STAGE" == "round0" || "$STAGE" == "checkpoint-gate" ]]; then
    for path in "$EVALUATION_SUITE" "$EVALUATION_POLICY"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: required closed-loop evaluation input not found: $path" >&2
            exit 2
        fi
    done
fi
if [[ "$STAGE" == "round0" ]]; then
    for path in "$EXPERT_BASELINE_EVALUATION" "$BASE_GEMMA_EVALUATION"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: STAGE=round0 requires baseline evaluation artifact: $path" >&2
            exit 2
        fi
    done
    if [[ -z "$EXPERT_POLICY_IDENTITY" ]]; then
        echo "ERROR: STAGE=round0 requires EXPERT_POLICY_IDENTITY." >&2
        exit 2
    fi
fi
if [[ "$STAGE" == "checkpoint-gate" ]]; then
    if [[ -z "$CHECKPOINT_EVALUATION" || ! -f "$CHECKPOINT_EVALUATION" ]]; then
        echo "ERROR: STAGE=checkpoint-gate requires CHECKPOINT_EVALUATION." >&2
        exit 2
    fi
    if [[ -z "$CHECKPOINT_MODEL_ID" || ! "$CHECKPOINT_MODEL_REVISION" =~ ^([0-9a-fA-F]{40}|[0-9a-fA-F]{64})$ ]]; then
        echo "ERROR: checkpoint promotion requires CHECKPOINT_MODEL_ID and an immutable CHECKPOINT_MODEL_REVISION." >&2
        exit 2
    fi
fi

echo "===== BC0 Gemma 4 round-0 stage ====="
echo "job:       ${SLURM_JOB_ID:-interactive}"
echo "host:      $(hostname)"
echo "stage:     $STAGE"
echo "python:    $PYTHON"
echo "model:     $MODEL_NAME"
echo "revision:  $MODEL_REVISION"
echo "train:     $TRAIN_FILE"
echo "output:    $OUTPUT_DIR"
echo "downloads: $ALLOW_DOWNLOAD"
if [[ "$STAGE" == "round0" || "$STAGE" == "checkpoint-gate" ]]; then
    echo "eval suite: $EVALUATION_SUITE"
    echo "eval policy: $EVALUATION_POLICY"
fi
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
        COMMAND=("$PYTHON" -m psse_env.sft gate "${COMMON_ARGS[@]}" --test "$TEST_FILE" --report-output "$OUTPUT_DIR/gate_report.json")
        ;;
    one-batch)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" --mode one-batch --load-in-4bit)
        ;;
    tiny-overfit)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" --mode tiny-overfit --tiny-overfit-steps "$TINY_OVERFIT_STEPS" --learning-rate "$TINY_OVERFIT_LR" --load-in-4bit)
        ;;
    round0)
        COMMAND=("$PYTHON" -m psse_env.sft train "${COMMON_ARGS[@]}" --output-dir "$OUTPUT_DIR" --batch-size 1 --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --learning-rate "$TRAIN_LR" --epochs "$TRAIN_EPOCHS" --smoke-steps 1 --load-in-4bit --evaluation-suite "$EVALUATION_SUITE" --evaluation-policy "$EVALUATION_POLICY" --expert-baseline-evaluation "$EXPERT_BASELINE_EVALUATION" --base-baseline-evaluation "$BASE_GEMMA_EVALUATION" --expert-policy-identity "$EXPERT_POLICY_IDENTITY" --baseline-evaluation-report-output "$BASELINE_EVALUATION_REPORT")
        ;;
    checkpoint-gate)
        COMMAND=("$PYTHON" -m psse_env.dagger.validate_evaluation --role checkpoint-promotion --artifact "$CHECKPOINT_EVALUATION" --policy "$EVALUATION_POLICY" --expected-source-commit HEAD --expected-suite "$EVALUATION_SUITE" --expected-protocol canonical --expected-model-id "$CHECKPOINT_MODEL_ID" --expected-model-revision "$CHECKPOINT_MODEL_REVISION" --report-output "$CHECKPOINT_GATE_REPORT")
        ;;
esac

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
