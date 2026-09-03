#!/usr/bin/env bash
#SBATCH --job-name=gemma4-full-bc0
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --open-mode=append
#SBATCH --output=gemma4_full_bc0_%j.out
#SBATCH --error=gemma4_full_bc0_%j.err

# Research-only full 12B BC0. This trains a fresh adapter from the pinned base
# on every root-disjoint D0 training row, selects the lowest validation-loss
# checkpoint, reloads the saved adapter, and never collects DAgger data.

set -Eeuo pipefail

: "${RESEARCH_SOURCE_ROOT:?set RESEARCH_SOURCE_ROOT to the committed checkout}"
: "${RESEARCH_ENV:?set RESEARCH_ENV to the research conda environment}"
: "${RESEARCH_RUN_ROOT:?set RESEARCH_RUN_ROOT to the persistent run directory}"

D0_ROOT="${RESEARCH_D0_ROOT:-/scratch/yx3882/dagger_release_a5a7574_20260823/round0_aggregate_release}"
TRAIN_FILE="$D0_ROOT/aggregate.train_view.jsonl"
VALIDATION_FILE="$D0_ROOT/aggregate.validation.jsonl"
MINI_POSTFLIGHT="$RESEARCH_RUN_ROOT/bc0/mini/mini_bc0_postflight.json"
OUTPUT_DIR="$RESEARCH_RUN_ROOT/bc0/full"
RUN_IDENTITY="$OUTPUT_DIR/run_identity.json"

for required in \
  "$RESEARCH_SOURCE_ROOT/psse_env/requirements-sft-research.txt" \
  "$TRAIN_FILE" \
  "$VALIDATION_FILE" \
  "$MINI_POSTFLIGHT"
do
  if [[ ! -f "$required" ]]; then
    echo "required full-BC0 input is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.job.lock"
if ! flock -n 9; then
  echo "another full-BC0 job owns $OUTPUT_DIR" >&2
  exit 73
fi

module purge
module load anaconda3/2025.06
# shellcheck source=/dev/null
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$RESEARCH_ENV"
cd "$RESEARCH_SOURCE_ROOT"

SOURCE_COMMIT="$(git rev-parse HEAD)"
python - \
  "$MINI_POSTFLIGHT" \
  "$TRAIN_FILE" \
  "$VALIDATION_FILE" \
  "$RUN_IDENTITY" \
  "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def load_rows(path: str) -> tuple[list[dict], set[str]]:
    rows = [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    roots = set()
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("protocol") != "canonical":
            raise SystemExit(f"{path}[{index}] is not canonical protocol")
        root = str(
            row.get("physical_root_fingerprint")
            or metadata.get("physical_root_fingerprint")
            or ""
        ).strip()
        if not root:
            raise SystemExit(f"{path}[{index}] lacks a physical root")
        roots.add(root)
    return rows, roots


mini = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
train_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
validation_path = pathlib.Path(sys.argv[3]).resolve(strict=True)
identity_path = pathlib.Path(sys.argv[4])
source_commit = sys.argv[5]
train, train_roots = load_rows(str(train_path))
validation, validation_roots = load_rows(str(validation_path))
if mini.get("passed") is not True:
    raise SystemExit("the 12B mini-BC0 integration postflight has not passed")
if len(train) != 1280 or len(validation) != 304:
    raise SystemExit(
        f"unexpected full D0 split: train={len(train)}, validation={len(validation)}"
    )
if len(train_roots) != 182 or len(validation_roots) != 44:
    raise SystemExit(
        "unexpected full D0 root counts: "
        f"train={len(train_roots)}, validation={len(validation_roots)}"
    )
if train_roots & validation_roots:
    raise SystemExit("full-BC0 train and validation roots overlap")
train_ids = {str(row.get("example_id") or "") for row in train}
validation_ids = {str(row.get("example_id") or "") for row in validation}
if "" in train_ids or "" in validation_ids:
    raise SystemExit("full-BC0 rows require example_id")
if train_ids & validation_ids:
    raise SystemExit("full-BC0 train and validation example IDs overlap")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


train_sha256 = sha256(train_path)
validation_sha256 = sha256(validation_path)
expected_train_sha256 = (
    "28b733db96c6ce05dbdc8d43484bdbb14445e1105958a78f9a35024aa5b3844a"
)
expected_validation_sha256 = (
    "2ea1c79d8bc85faf40a8ea5edd2352688c141e658b98407fcfba04ba49a9e4ca"
)
if train_sha256 != expected_train_sha256:
    raise SystemExit("full-BC0 canonical training file SHA-256 mismatch")
if validation_sha256 != expected_validation_sha256:
    raise SystemExit("full-BC0 canonical validation file SHA-256 mismatch")

identity = {
    "contract": "research_gemma4_full_bc0_resume_identity_v1",
    "source_commit": source_commit,
    "model_id": "google/gemma-4-12B-it",
    "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "architecture": "gemma4_unified",
    "train": {
        "path": str(train_path),
        "sha256": train_sha256,
        "rows": len(train),
        "roots": len(train_roots),
    },
    "validation": {
        "path": str(validation_path),
        "sha256": validation_sha256,
        "rows": len(validation),
        "roots": len(validation_roots),
    },
    "training": {
        "max_length": 32768,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "epochs": 1.0,
        "save_steps": 64,
        "eval_steps": 64,
        "best_metric": "eval_loss",
        "seed": 20260720,
    },
    "lora": {"rank": 16, "alpha": 16, "dropout": 0.0},
}
if identity_path.is_file():
    recorded = json.loads(identity_path.read_text(encoding="utf-8"))
    if recorded != identity:
        raise SystemExit(
            "full-BC0 resume identity differs from the existing output directory"
        )
else:
    stale_names = {
        "lora",
        "research_run.json",
        "training_stage.json",
        "trainer_state.json",
        "full_bc0_postflight.json",
    }
    stale = [
        path.name
        for path in identity_path.parent.iterdir()
        if path.name in stale_names or path.name.startswith("checkpoint-")
    ]
    if stale:
        raise SystemExit(
            "full-BC0 output has training artifacts but no run identity: "
            + ", ".join(sorted(stale))
        )
    temporary = identity_path.with_name(
        f".{identity_path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, identity_path)
print(
    {
        "mini_bc0_passed": True,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_roots": len(train_roots),
        "validation_roots": len(validation_roots),
        "overlap": 0,
        "train_sha256": identity["train"]["sha256"],
        "validation_sha256": identity["validation"]["sha256"],
        "source_commit": source_commit,
    }
)
PY

python -m pip check

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RESEARCH_SOURCE_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${RESEARCH_HF_HOME:-/scratch/yx3882/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="${RESEARCH_TORCH_HOME:-/scratch/yx3882/.cache/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTORCH_ALLOC_CONF="${RESEARCH_PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python -m psse_env.sft research-cache \
  --model-choice 12b \
  --output "$OUTPUT_DIR/cache_preflight.json"

TELEMETRY="$OUTPUT_DIR/nvidia_smi_${SLURM_JOB_ID}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits \
  --loop=10 >"$TELEMETRY" 2>"$OUTPUT_DIR/nvidia_smi_${SLURM_JOB_ID}.err" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

resume_args=()
if [[ ! -d "$OUTPUT_DIR/lora" ]]; then
  checkpoint_path="$(python -m psse_env.sft.research_checkpoint \
    --output-dir "$OUTPUT_DIR" \
    --expected-base-model "google/gemma-4-12B-it")"
  if [[ -n "$checkpoint_path" ]]; then
    echo "resuming from newest complete checkpoint: $checkpoint_path"
    resume_args+=(--resume-from-checkpoint "$checkpoint_path")
  fi
fi

python -m psse_env.sft research-train \
  --model-choice 12b \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-length 32768 \
  --strict-prompt-length \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --epochs 1 \
  --logging-steps 1 \
  --save-steps 64 \
  --eval-steps 64 \
  --select-best-eval-loss \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0 \
  --smoke-steps 1 \
  --reload-canaries 1 \
  --seed 20260720 \
  "${resume_args[@]}"

python -m psse_env.sft.research_bc0_postflight \
  --output-dir "$OUTPUT_DIR" \
  --expected-train-rows 1280 \
  --expected-validation-rows 304 \
  --expected-eval-step 64 \
  --expected-eval-step 128 \
  --expected-eval-step 192 \
  --expected-eval-step 256 \
  --expected-eval-step 320 \
  --minimum-global-step 320 \
  --maximum-global-step 320
