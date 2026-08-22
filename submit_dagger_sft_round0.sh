#!/bin/bash
#SBATCH --job-name=bc0_sft_round0
#SBATCH --output=/scratch/yx3882/psse_agent/bc0_sft_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/bc0_sft_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|h100|rtx6000"
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu

# Staged launcher for a newly generated, provenance-bound, canonical round-0
# recovery-balanced observable-expert BC aggregate.  The historical
# data/round0_aggregate_20260719 directory is intentionally not the default:
# it predates physical-root fingerprints, explicit eligibility, and current
# registry/source provenance and must fail the release gate.
# Submit STAGE=gate, one-batch, targeted-tiny-overfit, tiny-overfit, and round0
# in that order on a
# pinned H200/H100/RTX Pro 6000 constraint above. RTX Pro 6000 is accepted only
# when runtime attestation confirms the Pro name and at least 90,000 MiB.
# STAGE=round0
# refuses to train until the observable
# expert passes the full content-pinned fixed-suite gate and the exact pinned
# base model supplies complete, reproducible identity/evaluation evidence. Base
# performance failures remain in the baseline report but do not block BC0
# training or mislabel the weak base as release-qualified. After training,
# STAGE=checkpoint-gate validates one exact checkpoint artifact and its paired
# per-root non-regression against the persisted base artifact before promotion;
# full production SFT remains refused here. STAGE=round1 is the bounded
# warm-start continuation: it requires one immutable round-0 LoRA tree identity,
# an explicit full or natural-only immutable view, and defaults to one epoch at
# 3e-5 without changing the cold round-0 defaults.
# For NYU's preemptible RTX Pro 6000 route, override the CPU request at submit
# time so CPU packing does not strand free GPUs:
#
#   sbatch --constraint=rtx6000 --cpus-per-task=4 --mem=128G \
#     --export="ALL,EXPECTED_ACCELERATOR_CLASS=rtx6000,..." \
#     submit_dagger_sft_round0.sh
#
# The runtime gate still requires an actual RTX Pro 6000 with >=90,000 MiB.
#
# MAX_LENGTH=6144 is a conservative starting envelope.  The exact pinned
# processor gate for the newly generated release aggregate remains decisive.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/psse_agent}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
STAGE=${STAGE:-gate}
ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD:-0}
REVIEWED_SOURCE_COMMIT=${REVIEWED_SOURCE_COMMIT:-}
ENABLE_WANDB=${ENABLE_WANDB:-0}
EXPECTED_ACCELERATOR_CLASS=${EXPECTED_ACCELERATOR_CLASS:-auto}

MODEL_NAME=${MODEL_NAME:-unsloth/gemma-4-31B-it}
MODEL_REVISION=${MODEL_REVISION:-8a796db4df380b178065ed910849477ff0e99c87}
PINNED_DEPENDENCY_LOCK_SHA256=58c1f4690803b0109d47ac81ae613e07d883a1cd5a14cbde0409f310d2dd4df5
AGGREGATE_DIR=${AGGREGATE_DIR:-data/round0_aggregate_release}
TRAIN_FILE_EXPLICIT=0
if [[ -n "${TRAIN_FILE+x}" ]]; then
    TRAIN_FILE_EXPLICIT=1
fi
TRAIN_FILE=${TRAIN_FILE:-}
VALIDATION_FILE=${VALIDATION_FILE:-$AGGREGATE_DIR/aggregate.validation.jsonl}
TEST_FILE=${TEST_FILE:-$AGGREGATE_DIR/aggregate.test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/yx3882/psse_agent/outputs/bc0_gemma4_31b_round0}
MAX_LENGTH=${MAX_LENGTH:-6144}
ROWS_MIN=${ROWS_MIN:-1024}
ROWS_MAX=${ROWS_MAX:-4096}
TINY_OVERFIT_STEPS=${TINY_OVERFIT_STEPS:-20}
TINY_OVERFIT_LR=${TINY_OVERFIT_LR:-0.0001}
TARGETED_TINY_OVERFIT_MIN_RELATIVE_LOSS_REDUCTION=${TARGETED_TINY_OVERFIT_MIN_RELATIVE_LOSS_REDUCTION:-0.20}
TARGETED_TINY_OVERFIT_REPORT=${TARGETED_TINY_OVERFIT_REPORT:-$OUTPUT_DIR/targeted_tiny_overfit_sweep.json}
TRAIN_LR=${TRAIN_LR:-0.0001}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}
ROUND1_LR=${ROUND1_LR:-0.00003}
ROUND1_EPOCHS=${ROUND1_EPOCHS:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TRAIN_SEED=${TRAIN_SEED:-}
STUDY_MANIFEST=${STUDY_MANIFEST:-psse_env/dagger/studies/dagger_multiseed_study_v1.json}
STUDY_VARIANT=${STUDY_VARIANT:-}
ROUND1_VIEW=${ROUND1_VIEW:-}
INITIAL_ADAPTER_PATH=${INITIAL_ADAPTER_PATH:-}
INITIAL_ADAPTER_REVISION=${INITIAL_ADAPTER_REVISION:-}
PARENT_CHECKPOINT_RECEIPT=${PARENT_CHECKPOINT_RECEIPT:-}
EVALUATION_SUITE=${EVALUATION_SUITE:-psse_env/dagger/suites/bc0_eval_suite_v1.json}
EVALUATION_POLICY=${EVALUATION_POLICY:-psse_env/dagger/bc0_evaluation_policy.json}
EXPERT_BASELINE_EVALUATION=${EXPERT_BASELINE_EVALUATION:-artifacts/evaluations/expert_baseline_evaluation.json}
BASE_GEMMA_EVALUATION=${BASE_GEMMA_EVALUATION:-artifacts/evaluations/base_gemma_evaluation.json}
EXPERT_POLICY_IDENTITY=${EXPERT_POLICY_IDENTITY:-bc0-observable-handoff-expert-v2}
BASELINE_EVALUATION_REPORT=${BASELINE_EVALUATION_REPORT:-$OUTPUT_DIR/baseline_evaluation_gate.json}
PROCESSOR_GATE_REPORT=${PROCESSOR_GATE_REPORT:-$OUTPUT_DIR/gate_report.json}
CHECKPOINT_EVALUATION=${CHECKPOINT_EVALUATION:-}
CHECKPOINT_MODEL_ID=${CHECKPOINT_MODEL_ID:-}
CHECKPOINT_MODEL_REVISION=${CHECKPOINT_MODEL_REVISION:-}
CHECKPOINT_GATE_REPORT=${CHECKPOINT_GATE_REPORT:-$OUTPUT_DIR/checkpoint_promotion_gate.json}

case "$STAGE" in
    gate|one-batch|targeted-tiny-overfit|tiny-overfit|round0|round1|checkpoint-gate)
        ;;
    full|production)
        echo "ERROR: full production SFT is blocked pending the held-out recovery evaluation of the round-0 checkpoint." >&2
        exit 2
        ;;
    *)
        echo "ERROR: STAGE must be gate, one-batch, targeted-tiny-overfit, tiny-overfit, round0, round1, or checkpoint-gate; got '$STAGE'." >&2
        exit 2
        ;;
esac
case "$ENABLE_WANDB" in
    0|1)
        ;;
    *)
        echo "ERROR: ENABLE_WANDB must be 0 or 1; got '$ENABLE_WANDB'." >&2
        exit 2
        ;;
esac
case "$EXPECTED_ACCELERATOR_CLASS" in
    auto|h100|h200|rtx6000)
        ;;
    *)
        echo "ERROR: EXPECTED_ACCELERATOR_CLASS must be auto, h100, h200, or rtx6000." >&2
        exit 2
        ;;
esac
TRAINING_STAGE=0
case "$STAGE" in
    round0)
        TRAINING_STAGE=1
        EXPECTED_STUDY_VARIANT=bc0
        ;;
    round1)
        TRAINING_STAGE=1
        ;;
esac
if [[ "$TRAINING_STAGE" == "1" && -z "$TRAIN_SEED" ]]; then
    echo "ERROR: STAGE=$STAGE requires an explicit TRAIN_SEED." >&2
    exit 2
fi
if [[ -n "$TRAIN_SEED" ]] && { [[ ! "$TRAIN_SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || (( 10#$TRAIN_SEED > 4294967295 )); }; then
    echo "ERROR: TRAIN_SEED must be a canonical integer between 0 and 4294967295; got '$TRAIN_SEED'." >&2
    exit 2
fi
if [[ "$STAGE" == "round0" ]]; then
    if [[ -z "$STUDY_VARIANT" ]]; then
        STUDY_VARIANT=bc0
    elif [[ "$STUDY_VARIANT" != "bc0" ]]; then
        echo "ERROR: STAGE=round0 requires STUDY_VARIANT=bc0; got '$STUDY_VARIANT'." >&2
        exit 2
    fi
elif [[ "$STAGE" == "round1" ]]; then
    case "$STUDY_VARIANT" in
        natural_dagger|natural_dagger_probes)
            ;;
        *)
            echo "ERROR: STAGE=round1 requires explicit STUDY_VARIANT=natural_dagger or natural_dagger_probes; got '$STUDY_VARIANT'." >&2
            exit 2
            ;;
    esac
fi
if [[ "$TRAINING_STAGE" == "1" ]]; then
    if [[ "$MODEL_NAME" != "unsloth/gemma-4-31B-it" || "$MODEL_REVISION" != "8a796db4df380b178065ed910849477ff0e99c87" ]]; then
        echo "ERROR: study training forbids MODEL_NAME or MODEL_REVISION drift." >&2
        exit 2
    fi
    if [[ "$MAX_LENGTH" != "6144" || "$GRADIENT_ACCUMULATION_STEPS" != "4" ]]; then
        echo "ERROR: study training requires MAX_LENGTH=6144 and GRADIENT_ACCUMULATION_STEPS=4." >&2
        exit 2
    fi
    if [[ "$STAGE" == "round0" && ( "$TRAIN_LR" != "0.0001" || "$TRAIN_EPOCHS" != "2" ) ]]; then
        echo "ERROR: BC0 protocol requires TRAIN_LR=0.0001 and TRAIN_EPOCHS=2." >&2
        exit 2
    fi
    if [[ "$STAGE" == "round1" && ( "$ROUND1_LR" != "0.00003" || "$ROUND1_EPOCHS" != "1" ) ]]; then
        echo "ERROR: Round-1 protocol requires ROUND1_LR=0.00003 and ROUND1_EPOCHS=1." >&2
        exit 2
    fi
fi

WANDB_ACTIVE=0
if [[ "$ENABLE_WANDB" == "1" && ( "$STAGE" == "round0" || "$STAGE" == "round1" ) ]]; then
    WANDB_ACTIVE=1
fi

if [[ -n "$INITIAL_ADAPTER_PATH" || -n "$INITIAL_ADAPTER_REVISION" ]]; then
    if [[ -z "$INITIAL_ADAPTER_PATH" || -z "$INITIAL_ADAPTER_REVISION" ]]; then
        echo "ERROR: INITIAL_ADAPTER_PATH and INITIAL_ADAPTER_REVISION must be supplied together." >&2
        exit 2
    fi
    if [[ "$INITIAL_ADAPTER_PATH" != /* ]]; then
        echo "ERROR: INITIAL_ADAPTER_PATH must be absolute." >&2
        exit 2
    fi
    if [[ ! "$INITIAL_ADAPTER_REVISION" =~ ^[0-9a-fA-F]{64}$ ]]; then
        echo "ERROR: INITIAL_ADAPTER_REVISION must be a 64-hex checkpoint tree SHA-256." >&2
        exit 2
    fi
fi
if [[ "$STAGE" == "round1" && -z "$INITIAL_ADAPTER_PATH" ]]; then
    echo "ERROR: STAGE=round1 requires INITIAL_ADAPTER_PATH and INITIAL_ADAPTER_REVISION." >&2
    exit 2
fi
if [[ "$STAGE" == "round1" ]]; then
    if [[ -z "$PARENT_CHECKPOINT_RECEIPT" ]]; then
        echo "ERROR: STAGE=round1 requires PARENT_CHECKPOINT_RECEIPT for the same-seed BC0 adapter." >&2
        exit 2
    fi
    if [[ "$PARENT_CHECKPOINT_RECEIPT" != /* || -L "$PARENT_CHECKPOINT_RECEIPT" || ! -f "$PARENT_CHECKPOINT_RECEIPT" ]]; then
        echo "ERROR: PARENT_CHECKPOINT_RECEIPT must be an absolute existing regular file, not a symlink." >&2
        exit 2
    fi
elif [[ "$TRAINING_STAGE" == "1" && -n "$PARENT_CHECKPOINT_RECEIPT" ]]; then
    echo "ERROR: only STAGE=round1 may bind PARENT_CHECKPOINT_RECEIPT." >&2
    exit 2
fi
if [[ ( "$STAGE" == "round0" || "$STAGE" == "checkpoint-gate" ) && -n "$INITIAL_ADAPTER_PATH" ]]; then
    echo "ERROR: initial adapter identity is valid only for the Round-1 data gate, warm-start smoke stages, or STAGE=round1." >&2
    exit 2
fi
ROUND1_SEED_COUPLING_REQUIRED=0
if [[ "$STAGE" == "round1" ]]; then
    ROUND1_SEED_COUPLING_REQUIRED=1
elif [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    case "$STAGE" in
        gate|one-batch|targeted-tiny-overfit|tiny-overfit)
            ROUND1_SEED_COUPLING_REQUIRED=1
            ;;
    esac
fi

if [[ "$ROUND1_SEED_COUPLING_REQUIRED" == "1" ]]; then
    case "$ROUND1_VIEW" in
        full)
            EXPECTED_ROUND1_TRAIN_FILE=$AGGREGATE_DIR/aggregate.train_view.jsonl
            ;;
        natural-only)
            EXPECTED_ROUND1_TRAIN_FILE=$AGGREGATE_DIR/aggregate.natural_only.train_view.jsonl
            ;;
        *)
            echo "ERROR: Round-1 warm-start stages require explicit ROUND1_VIEW=full or natural-only; got '$ROUND1_VIEW'." >&2
            exit 2
            ;;
    esac
    if [[ "$STUDY_VARIANT" == "natural_dagger" && "$ROUND1_VIEW" != "natural-only" ]]; then
        echo "ERROR: STUDY_VARIANT=natural_dagger requires ROUND1_VIEW=natural-only." >&2
        exit 2
    fi
    if [[ "$STUDY_VARIANT" == "natural_dagger_probes" && "$ROUND1_VIEW" != "full" ]]; then
        echo "ERROR: STUDY_VARIANT=natural_dagger_probes requires ROUND1_VIEW=full." >&2
        exit 2
    fi
    if [[ "$TRAIN_FILE_EXPLICIT" == "0" ]]; then
        TRAIN_FILE=$EXPECTED_ROUND1_TRAIN_FILE
    elif [[ "$(realpath -m -- "$TRAIN_FILE")" != "$(realpath -m -- "$EXPECTED_ROUND1_TRAIN_FILE")" ]]; then
        echo "ERROR: ROUND1_VIEW=$ROUND1_VIEW requires canonical TRAIN_FILE=$EXPECTED_ROUND1_TRAIN_FILE; got '$TRAIN_FILE'." >&2
        exit 2
    fi
else
    if [[ -n "$ROUND1_VIEW" ]]; then
        echo "ERROR: ROUND1_VIEW is valid only for a Round-1 warm-start stage." >&2
        exit 2
    fi
    TRAIN_FILE=${TRAIN_FILE:-$AGGREGATE_DIR/aggregate.train_view.jsonl}
fi

if [[ ! "$MODEL_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: MODEL_REVISION must be a pinned 40-character Hugging Face commit." >&2
    exit 2
fi
if [[ ! "$REVIEWED_SOURCE_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: REVIEWED_SOURCE_COMMIT must be the externally reviewed 40-hex freeze commit." >&2
    exit 2
fi
REVIEWED_SOURCE_COMMIT=${REVIEWED_SOURCE_COMMIT,,}

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

cd "$REPO_ROOT"
CURRENT_SOURCE_COMMIT=$(git rev-parse HEAD 2>/dev/null || true)
if [[ "$CURRENT_SOURCE_COMMIT" != "$REVIEWED_SOURCE_COMMIT" ]]; then
    echo "ERROR: checkout $CURRENT_SOURCE_COMMIT is not reviewed freeze commit $REVIEWED_SOURCE_COMMIT." >&2
    exit 2
fi
if [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    if [[ ! -d "$INITIAL_ADAPTER_PATH" ]]; then
        echo "ERROR: initial adapter directory does not exist: $INITIAL_ADAPTER_PATH" >&2
        exit 2
    fi
    INITIAL_ADAPTER_RESOLVED=$(realpath -e -- "$INITIAL_ADAPTER_PATH")
    OUTPUT_DIR_RESOLVED=$(realpath -m -- "$OUTPUT_DIR")
    if [[ "$OUTPUT_DIR_RESOLVED" == "$INITIAL_ADAPTER_RESOLVED" || "$OUTPUT_DIR_RESOLVED" == "$INITIAL_ADAPTER_RESOLVED"/* || "$INITIAL_ADAPTER_RESOLVED" == "$OUTPUT_DIR_RESOLVED"/* ]]; then
        echo "ERROR: OUTPUT_DIR and INITIAL_ADAPTER_PATH must not overlap." >&2
        exit 2
    fi
    if [[ -n "$PARENT_CHECKPOINT_RECEIPT" ]]; then
        PARENT_RECEIPT_RESOLVED=$(realpath -e -- "$PARENT_CHECKPOINT_RECEIPT")
        EXPECTED_PARENT_RECEIPT=$(realpath -m -- "$(dirname -- "$INITIAL_ADAPTER_RESOLVED")/checkpoint_receipt.json")
        if [[ "$PARENT_RECEIPT_RESOLVED" != "$EXPECTED_PARENT_RECEIPT" ]]; then
            echo "ERROR: PARENT_CHECKPOINT_RECEIPT must be the canonical sibling of INITIAL_ADAPTER_PATH." >&2
            exit 2
        fi
    fi
fi
mkdir -p artifacts/logs "$OUTPUT_DIR"
mkdir -p /scratch/yx3882/.cache/huggingface /scratch/yx3882/.cache/torch

export HF_HOME=${HF_HOME:-/scratch/yx3882/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yx3882/.cache/torch}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}

if [[ "$WANDB_ACTIVE" == "1" ]]; then
    WANDB_MODE=${WANDB_MODE:-online}
    case "$WANDB_MODE" in
        online|offline)
            ;;
        *)
            echo "ERROR: WANDB_MODE must be online or offline when W&B is enabled; got '$WANDB_MODE'." >&2
            exit 2
            ;;
    esac

    WANDB_SOURCE_SHORT=${REVIEWED_SOURCE_COMMIT:0:12}
    WANDB_SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}
    WANDB_SCRATCH_ROOT=${WANDB_SCRATCH_ROOT:-/scratch/${USER:-yx3882}/wandb}
    WANDB_DIR=${WANDB_DIR:-$WANDB_SCRATCH_ROOT/runs}
    WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$WANDB_SCRATCH_ROOT/cache}
    WANDB_DATA_DIR=${WANDB_DATA_DIR:-$WANDB_SCRATCH_ROOT/data}
    WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$WANDB_SCRATCH_ROOT/config}
    WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-$WANDB_SCRATCH_ROOT/artifacts}
    mkdir -p \
        "$WANDB_DIR" \
        "$WANDB_CACHE_DIR" \
        "$WANDB_DATA_DIR" \
        "$WANDB_CONFIG_DIR" \
        "$WANDB_ARTIFACT_DIR"

    export WANDB_PROJECT=${WANDB_PROJECT:-psse-agent-bc0}
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        export WANDB_ENTITY
    else
        unset WANDB_ENTITY
    fi
    if [[ "$STAGE" == "round1" ]]; then
        WANDB_ROUND_NAME=round1
        WANDB_ROUND_SHORT=r1
        WANDB_VARIANT_SUFFIX=-$STUDY_VARIANT-$ROUND1_VIEW
        WANDB_VARIANT_TAGS=,variant-$STUDY_VARIANT,round1-view-$ROUND1_VIEW
    else
        WANDB_ROUND_NAME=round0
        WANDB_ROUND_SHORT=r0
        WANDB_VARIANT_SUFFIX=-bc0
        WANDB_VARIANT_TAGS=,variant-bc0
    fi
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-bc0-$WANDB_ROUND_NAME$WANDB_VARIANT_SUFFIX-$WANDB_SOURCE_SHORT}
    export WANDB_TAGS=${WANDB_TAGS:-bc0,$WANDB_ROUND_NAME,gemma4-31b,source-$WANDB_SOURCE_SHORT,train-seed-$TRAIN_SEED$WANDB_VARIANT_TAGS}
    export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-bc0-$WANDB_ROUND_NAME-sft}
    WANDB_RUN_ID_DEFAULT=bc0-$WANDB_ROUND_SHORT$WANDB_VARIANT_SUFFIX-seed$TRAIN_SEED-$WANDB_SOURCE_SHORT-$WANDB_SLURM_JOB_ID
    export WANDB_RUN_ID=${WANDB_RUN_ID:-$WANDB_RUN_ID_DEFAULT}
    export WANDB_NAME=${WANDB_NAME:-$WANDB_RUN_ID}
    export WANDB_RESUME=allow
    export WANDB_LOG_MODEL=false
    export WANDB_WATCH=false
    export WANDB_MODE
    export WANDB_DIR WANDB_CACHE_DIR WANDB_DATA_DIR WANDB_CONFIG_DIR
    export WANDB_ARTIFACT_DIR
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found at $PYTHON; create it from psse_env/requirements-sft.txt first." >&2
    exit 2
fi
if [[ "$STAGE" == "round0" || "$STAGE" == "round1" ]]; then
    # Validate cheap, immutable prerequisite evidence before importing the
    # GPU stack or running native-runtime checks on an expensive allocation.
    "$PYTHON" - \
        "$PROCESSOR_GATE_REPORT" \
        "$MODEL_NAME" \
        "$MODEL_REVISION" \
        "$REVIEWED_SOURCE_COMMIT" \
        "$TRAIN_FILE" \
        "$VALIDATION_FILE" \
        "$TEST_FILE" \
        "$MAX_LENGTH" <<'PY'
import sys

from psse_env.sft.provenance import validate_release_gate_report

(
    report_path,
    model,
    revision,
    source_commit,
    train_file,
    validation_file,
    test_file,
    max_length,
) = sys.argv[1:]
result = validate_release_gate_report(
    report_path,
    model=model,
    revision=revision,
    source_commit=source_commit,
    datasets={
        "train": train_file,
        "validation": validation_file,
        "test": test_file,
    },
    max_length=int(max_length),
)
if not result["passed"]:
    raise SystemExit(
        "Training prerequisite processor/data gate is NO-GO:\n- "
        + "\n- ".join(result["failures"])
    )
print("Training prerequisite processor/data gate passed with AutoProcessor")
PY
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
    if [[ ! -f "$AGGREGATE_DIR/SHA256SUMS" ]]; then
        echo "ERROR: release aggregate checksum manifest is missing: $AGGREGATE_DIR/SHA256SUMS" >&2
        exit 2
    fi
    (cd "$AGGREGATE_DIR" && sha256sum --check --quiet SHA256SUMS) || {
        echo "ERROR: aggregate split checksums do not match SHA256SUMS; re-ship the aggregate." >&2
        exit 2
    }
fi
if [[ "$ROUND1_SEED_COUPLING_REQUIRED" == "1" ]]; then
    ROUND1_PROVENANCE="$AGGREGATE_DIR/aggregate.generation_provenance.json"
    ROUND1_PREFLIGHT="$AGGREGATE_DIR/aggregate.preflight.json"
    if [[ ! -f "$ROUND1_PREFLIGHT" ]]; then
        echo "ERROR: Round-1 warm-start stages require aggregate.preflight.json." >&2
        exit 2
    fi
    "$PYTHON" -m psse_env.sft.round1_source_gate \
        --provenance "$ROUND1_PROVENANCE" \
        --preflight "$ROUND1_PREFLIGHT" \
        --reviewed-source-commit "$REVIEWED_SOURCE_COMMIT" \
        --initial-adapter-revision "$INITIAL_ADAPTER_REVISION" \
        --round1-view "$ROUND1_VIEW"
fi
if [[ "$STAGE" == "round0" || "$STAGE" == "round1" || "$STAGE" == "checkpoint-gate" ]]; then
    for path in "$EVALUATION_SUITE" "$EVALUATION_POLICY"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: required closed-loop evaluation input not found: $path" >&2
            exit 2
        fi
    done
fi
if [[ "$STAGE" == "round0" || "$STAGE" == "round1" ]]; then
    for path in "$EXPERT_BASELINE_EVALUATION" "$BASE_GEMMA_EVALUATION" "$PROCESSOR_GATE_REPORT"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: STAGE=$STAGE requires prerequisite evidence: $path" >&2
            exit 2
        fi
    done
    if [[ -z "$EXPERT_POLICY_IDENTITY" ]]; then
        echo "ERROR: STAGE=$STAGE requires EXPERT_POLICY_IDENTITY." >&2
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
    if [[ ! -f "$BASE_GEMMA_EVALUATION" ]]; then
        echo "ERROR: checkpoint promotion requires the persisted base reference artifact: $BASE_GEMMA_EVALUATION" >&2
        exit 2
    fi
fi

echo "===== BC0 Gemma 4 staged SFT ====="
echo "job:       ${SLURM_JOB_ID:-interactive}"
echo "host:      $(hostname)"
echo "stage:     $STAGE"
echo "python:    $PYTHON"
echo "model:     $MODEL_NAME"
echo "revision:  $MODEL_REVISION"
echo "source:    $REVIEWED_SOURCE_COMMIT"
echo "train:     $TRAIN_FILE"
echo "output:    $OUTPUT_DIR"
echo "GPU class: $EXPECTED_ACCELERATOR_CLASS"
if [[ "$TRAINING_STAGE" == "1" ]]; then
    echo "train seed: $TRAIN_SEED"
    echo "study:      $STUDY_VARIANT ($STUDY_MANIFEST)"
fi
if [[ "$ROUND1_SEED_COUPLING_REQUIRED" == "1" ]]; then
    echo "R1 view:    $ROUND1_VIEW"
fi
echo "downloads: $ALLOW_DOWNLOAD"
if [[ "$WANDB_ACTIVE" == "1" ]]; then
    echo "wandb:     enabled ($WANDB_MODE; project=$WANDB_PROJECT; run=$WANDB_RUN_ID)"
elif [[ "$ENABLE_WANDB" == "1" ]]; then
    echo "wandb:     inactive for STAGE=$STAGE (monitoring starts at round0 or round1)"
else
    echo "wandb:     disabled"
fi
if [[ "$STAGE" == "gate" || "$STAGE" == "round0" || "$STAGE" == "round1" ]]; then
    echo "processor gate: $PROCESSOR_GATE_REPORT"
fi
if [[ "$STAGE" == "targeted-tiny-overfit" ]]; then
    echo "targeted smoke report: $TARGETED_TINY_OVERFIT_REPORT"
fi
if [[ "$STAGE" == "round0" || "$STAGE" == "round1" || "$STAGE" == "checkpoint-gate" ]]; then
    echo "eval suite: $EVALUATION_SUITE"
    echo "eval policy: $EVALUATION_POLICY"
fi
if [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    echo "initial adapter: $INITIAL_ADAPTER_PATH@$INITIAL_ADAPTER_REVISION"
    if [[ -n "$PARENT_CHECKPOINT_RECEIPT" ]]; then
        echo "parent receipt:  $PARENT_CHECKPOINT_RECEIPT"
    fi
fi
GPU_INVENTORY=$(nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader)
echo "GPU inventory: $GPU_INVENTORY"
"$PYTHON" -c 'import accelerate, bitsandbytes, datasets, peft, torch, transformers, trl; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "bf16": torch.cuda.is_bf16_supported(), "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__, "datasets": datasets.__version__, "accelerate": accelerate.__version__})'
# Fail-closed runtime identity: the same interpreter, Torch CUDA build, and
# GPU class the release evaluator attests.  A mismatched runtime here would
# produce training evidence the paired checkpoint gate cannot vouch for.
"$PYTHON" - "$EXPECTED_ACCELERATOR_CLASS" <<'PY'
import json
import sys

from psse_env.sft.release_hardware import validate_torch_release_accelerator

failures = []
if sys.version_info[:2] != (3, 12):
    failures.append(f"python: running {sys.version.split()[0]}, requires 3.12")
import torch

if torch.__version__ != "2.10.0+cu128":
    failures.append(f"torch: installed {torch.__version__}, requires 2.10.0+cu128")
try:
    required_class = None if sys.argv[1] == "auto" else sys.argv[1]
    accelerator = validate_torch_release_accelerator(
        torch,
        required_class=required_class,
    )
except RuntimeError as exc:
    failures.append(str(exc))
if failures:
    raise SystemExit(
        "SFT runtime does not match the release evaluation contract:\n- "
        + "\n- ".join(failures)
    )
print(
    "SFT runtime matches the release evaluation contract "
    "(py3.12, torch 2.10.0+cu128, H100/H200/RTX Pro 6000 >=90,000 MiB): "
    + json.dumps(accelerator, sort_keys=True)
)
PY
"$PYTHON" -m pip check
"$PYTHON" - "$REPO_ROOT/psse_env/requirements-sft.txt" "$PINNED_DEPENDENCY_LOCK_SHA256" <<'PY'
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from packaging.requirements import Requirement

failures = []
lock_path = Path(sys.argv[1])
expected_lock_sha256 = sys.argv[2]
actual_lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
if actual_lock_sha256 != expected_lock_sha256:
    failures.append(
        "dependency lock SHA-256: "
        f"expected {expected_lock_sha256}, got {actual_lock_sha256}"
    )
for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        failures.append(f"{requirement.name}: not installed")
        continue
    if requirement.specifier and installed not in requirement.specifier:
        failures.append(
            f"{requirement.name}: installed {installed}, requires {requirement.specifier}"
        )
try:
    import opendssdirect as dss

    opendss_banner = str(dss.Basic.Version()).strip()
except Exception as exc:
    opendss_banner = ""
    failures.append(
        "OpenDSSDirect native runtime: "
        f"{type(exc).__name__}: {exc}"
    )
else:
    for marker in (
        "DSS C-API Library version 0.14.5",
        "DSS-Python version: 0.15.7",
        "OpenDSSDirect.py version: 0.9.4",
    ):
        if marker not in opendss_banner:
            failures.append(f"OpenDSSDirect native runtime missing {marker!r}")
if failures:
    raise SystemExit("SFT environment does not match requirements-sft.txt:\n- " + "\n- ".join(failures))
print(
    "SFT environment matches psse_env/requirements-sft.txt; "
    "OpenDSS native runtime loaded"
)
PY
if [[ "$WANDB_ACTIVE" == "1" ]]; then
    "$PYTHON" - "$REPO_ROOT/psse_env/requirements-wandb.txt" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from packaging.requirements import Requirement

requirements = [
    Requirement(line.strip())
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(requirements) != 1 or requirements[0].name.lower() != "wandb":
    raise SystemExit(
        "requirements-wandb.txt must contain exactly one pinned wandb requirement"
    )
requirement = requirements[0]
try:
    installed = version(requirement.name)
except PackageNotFoundError as exc:
    raise SystemExit(
        "W&B monitoring requested but wandb is not installed; "
        "rerun setup_unsloth_env.sh with INSTALL_WANDB=1"
    ) from exc
if requirement.specifier and installed not in requirement.specifier:
    raise SystemExit(
        f"wandb: installed {installed}, requires {requirement.specifier}"
    )
try:
    import wandb  # noqa: F401
except Exception as exc:
    raise SystemExit(
        f"W&B monitoring requested but wandb cannot be imported: "
        f"{type(exc).__name__}: {exc}"
    ) from exc
print(f"W&B monitoring dependency verified: wandb {installed}")
PY
fi

COMMON_ARGS=(
    --model "$MODEL_NAME"
    --revision "$MODEL_REVISION"
    --train "$TRAIN_FILE"
    --validation "$VALIDATION_FILE"
    --max-length "$MAX_LENGTH"
    --require-auto-processor
)
if [[ "$WANDB_ACTIVE" == "1" ]]; then
    COMMON_ARGS+=(--report-to wandb --run-name "$WANDB_NAME")
fi
INITIAL_ADAPTER_ARGS=()
ROUND1_SOURCE_ARGS=()
ROUND1_GATE_SOURCE_ARGS=()
TRAIN_ARGS=()
TRAIN_PROTOCOL_ARGS=()
if [[ "$TRAINING_STAGE" == "1" ]]; then
    TRAIN_ARGS+=(
        --seed "$TRAIN_SEED"
        --study-variant "$STUDY_VARIANT"
        --study-manifest "$STUDY_MANIFEST"
    )
    TRAIN_PROTOCOL_ARGS+=(
        --batch-size 1
        --gradient-accumulation-steps 4
        --max-steps -1
        --optimizer adamw_torch
        --lr-scheduler-type linear
        --lora-rank 16
        --lora-alpha 16
        --lora-dropout 0.0
        --load-in-4bit
    )
fi
if [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    INITIAL_ADAPTER_ARGS+=(
        --initial-adapter-path "$INITIAL_ADAPTER_PATH"
        --initial-adapter-revision "$INITIAL_ADAPTER_REVISION"
    )
    if [[ -n "$PARENT_CHECKPOINT_RECEIPT" ]]; then
        INITIAL_ADAPTER_ARGS+=(
            --parent-checkpoint-receipt "$PARENT_CHECKPOINT_RECEIPT"
        )
    fi
    ROUND1_SOURCE_ARGS+=(
        --round1-provenance "$ROUND1_PROVENANCE"
        --round1-preflight "$ROUND1_PREFLIGHT"
        --reviewed-source-commit "$REVIEWED_SOURCE_COMMIT"
        --round1-view "$ROUND1_VIEW"
    )
    ROUND1_GATE_SOURCE_ARGS+=(
        "${ROUND1_SOURCE_ARGS[@]}"
        --initial-adapter-revision "$INITIAL_ADAPTER_REVISION"
    )
fi
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
    case "$STAGE" in
        one-batch|targeted-tiny-overfit|tiny-overfit|round0|round1|checkpoint-gate)
            echo "ERROR: ALLOW_DOWNLOAD=1 is forbidden for STAGE=$STAGE; the reviewed model must come from the verified local snapshot." >&2
            exit 2
            ;;
    esac
    COMMON_ARGS+=(--allow-download)
elif [[ "$ALLOW_DOWNLOAD" != "0" ]]; then
    echo "ERROR: ALLOW_DOWNLOAD must be 0 or 1." >&2
    exit 2
fi

case "$STAGE" in
    gate)
        # Training and smoke size the optimizer-visible train+validation rows.
        # The gate additionally audits test, so offset its total-row bounds by
        # the exact number of non-empty JSONL test records.  Without this
        # adjustment a corpus could pass gate and fail unchanged downstream.
        TEST_ROWS=$(awk 'NF { count += 1 } END { print count + 0 }' "$TEST_FILE")
        GATE_ROWS_MIN=$((ROWS_MIN + TEST_ROWS))
        GATE_ROWS_MAX=$((ROWS_MAX + TEST_ROWS))
        COMMAND=("$PYTHON" -m psse_env.sft gate "${COMMON_ARGS[@]}" "${ROUND1_GATE_SOURCE_ARGS[@]}" --test "$TEST_FILE" --pilot-min-rows "$GATE_ROWS_MIN" --pilot-max-rows "$GATE_ROWS_MAX" --report-output "$PROCESSOR_GATE_REPORT")
        ;;
    one-batch)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" "${INITIAL_ADAPTER_ARGS[@]}" "${ROUND1_SOURCE_ARGS[@]}" --pilot-min-rows "$ROWS_MIN" --pilot-max-rows "$ROWS_MAX" --mode one-batch --load-in-4bit)
        ;;
    targeted-tiny-overfit)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" "${INITIAL_ADAPTER_ARGS[@]}" "${ROUND1_SOURCE_ARGS[@]}" --pilot-min-rows "$ROWS_MIN" --pilot-max-rows "$ROWS_MAX" --mode tiny-overfit --tiny-overfit-steps "$TINY_OVERFIT_STEPS" --targeted-recovery-sweep --targeted-min-relative-loss-reduction "$TARGETED_TINY_OVERFIT_MIN_RELATIVE_LOSS_REDUCTION" --report-output "$TARGETED_TINY_OVERFIT_REPORT" --load-in-4bit)
        ;;
    tiny-overfit)
        COMMAND=("$PYTHON" -m psse_env.sft smoke "${COMMON_ARGS[@]}" "${INITIAL_ADAPTER_ARGS[@]}" "${ROUND1_SOURCE_ARGS[@]}" --pilot-min-rows "$ROWS_MIN" --pilot-max-rows "$ROWS_MAX" --mode tiny-overfit --tiny-overfit-steps "$TINY_OVERFIT_STEPS" --learning-rate "$TINY_OVERFIT_LR" --load-in-4bit)
        ;;
    round0)
        COMMAND=("$PYTHON" -m psse_env.sft train "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "${TRAIN_PROTOCOL_ARGS[@]}" --pilot-min-rows "$ROWS_MIN" --pilot-max-rows "$ROWS_MAX" --output-dir "$OUTPUT_DIR" --learning-rate "$TRAIN_LR" --epochs "$TRAIN_EPOCHS" --smoke-steps 1 --evaluation-suite "$EVALUATION_SUITE" --evaluation-policy "$EVALUATION_POLICY" --expert-baseline-evaluation "$EXPERT_BASELINE_EVALUATION" --base-baseline-evaluation "$BASE_GEMMA_EVALUATION" --expert-policy-identity "$EXPERT_POLICY_IDENTITY" --baseline-evaluation-report-output "$BASELINE_EVALUATION_REPORT")
        ;;
    round1)
        COMMAND=("$PYTHON" -m psse_env.sft train "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "${TRAIN_PROTOCOL_ARGS[@]}" "${INITIAL_ADAPTER_ARGS[@]}" "${ROUND1_SOURCE_ARGS[@]}" --pilot-min-rows "$ROWS_MIN" --pilot-max-rows "$ROWS_MAX" --output-dir "$OUTPUT_DIR" --learning-rate "$ROUND1_LR" --epochs "$ROUND1_EPOCHS" --smoke-steps 1 --evaluation-suite "$EVALUATION_SUITE" --evaluation-policy "$EVALUATION_POLICY" --expert-baseline-evaluation "$EXPERT_BASELINE_EVALUATION" --base-baseline-evaluation "$BASE_GEMMA_EVALUATION" --expert-policy-identity "$EXPERT_POLICY_IDENTITY" --baseline-evaluation-report-output "$BASELINE_EVALUATION_REPORT")
        ;;
    checkpoint-gate)
        COMMAND=("$PYTHON" -m psse_env.dagger.validate_evaluation --role checkpoint-promotion --artifact "$CHECKPOINT_EVALUATION" --policy "$EVALUATION_POLICY" --expected-source-commit "$REVIEWED_SOURCE_COMMIT" --expected-suite "$EVALUATION_SUITE" --expected-protocol canonical --expected-model-id "$CHECKPOINT_MODEL_ID" --expected-model-revision "$CHECKPOINT_MODEL_REVISION" --reference-artifact "$BASE_GEMMA_EVALUATION" --reference-model-id "$MODEL_NAME" --reference-model-revision "$MODEL_REVISION" --report-output "$CHECKPOINT_GATE_REPORT")
        ;;
esac

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
