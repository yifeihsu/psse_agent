#!/usr/bin/env bash
#SBATCH --job-name=gemma4-dagger-r1-fallback
#SBATCH --account=torch_pr_627_general
# NYU Torch spells the 96-GB RTX Pro 6000 feature ``rtx6000``.  A runtime
# attestation below rejects the older 48-GB RTX 6000 Ada card.
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
#SBATCH --output=gemma4_dagger_r1_fallback_%j.out
#SBATCH --error=gemma4_dagger_r1_fallback_%j.err

# One bounded, research-only Gemma-4-12B DAgger Round-1 fallback.  Do not submit
# this while the independently launched Round-1 jobs 16347744/16347745 (or an
# equivalent replacement) are active.  The script also enforces that rule at
# runtime and requires explicit fallback authorization.
#
# Frozen sequence:
#   1. attest checkpoint-192 (cb10...), the root-disjoint 27-row validation,
#      and the previously published 15-root D1 suite that must remain protected;
#   2. resumably collect five fresh roots (2 measurement+parameter,
#      2 multi_measurement, 1 parameter) with run_dagger_research.py;
#   3. build an exact 1:1 D0/D1 mixture;
#   4. continue checkpoint-192 for exactly 32 optimizer steps, saving and
#      evaluating at 8/16/24/32 and publishing the lowest-eval-loss adapter;
#   5. greedily preflight checkpoint-192 and the candidate on all 27 actions;
#   6. only if that comparison passes, evaluate both adapters on the exact same
#      newly generated 15 development roots (6/6/3).
#
# This launcher never calls sbatch or scontrol and never launches another seed.
# Slurm may requeue this same job after preemption; collection ledgers and the
# newest complete Trainer checkpoint make that restart safe.

set -Eeuo pipefail

: "${RESEARCH_SOURCE_ROOT:?set RESEARCH_SOURCE_ROOT to the exact committed checkout}"
: "${RESEARCH_ENV:?set RESEARCH_ENV to the research conda environment}"
: "${RESEARCH_RUN_ROOT:?set RESEARCH_RUN_ROOT to the persistent experiment root}"
: "${RESEARCH_DAGGER_R1_FALLBACK_AUTHORIZED:?set RESEARCH_DAGGER_R1_FALLBACK_AUTHORIZED=YES only after the active pipeline failed and the user authorized this fallback}"

if [[ "$RESEARCH_DAGGER_R1_FALLBACK_AUTHORIZED" != "YES" ]]; then
  echo "fallback authorization must be exactly YES" >&2
  exit 2
fi

D0_ROOT="${RESEARCH_D0_ROOT:-/scratch/yx3882/dagger_release_a5a7574_20260823/round0_aggregate_release}"
D0_RAW="$D0_ROOT/aggregate.raw.jsonl"
D0_TRAIN="$D0_ROOT/aggregate.train_view.jsonl"
VALIDATION="${RESEARCH_DAGGER_R1_VALIDATION:-/scratch/yx3882/research_dagger_trace_20260823/trace_validation.jsonl}"
PROTECTED_D1="${RESEARCH_DAGGER_R1_PROTECTED_D1:-/scratch/yx3882/research_gemma4_small_20260824_fe94580/evaluation/bc0_12b_replay_compare_v2/published_replay/d1_development_suite.json}"
WARM_START="${RESEARCH_DAGGER_R1_WARM_START:-/scratch/yx3882/research_gemma4_small_20260824_fe94580/bc0/full/checkpoint-192}"
OUTPUT_ROOT="${RESEARCH_DAGGER_R1_OUTPUT_ROOT:-$RESEARCH_RUN_ROOT/dagger/round1_12b_fallback}"
COLLECTION_DIR="$OUTPUT_ROOT/collection"
TRAIN_DIR="$OUTPUT_ROOT/training"
CANDIDATE="$TRAIN_DIR/lora"

RUN_IDENTITY="$OUTPUT_ROOT/run_identity.json"
COLLECTION_AUDIT="$OUTPUT_ROOT/collection_audit.json"
TRAIN_RECIPE="$TRAIN_DIR/training_recipe.json"
TRAIN_COMPLETION="$TRAIN_DIR/training_completion.json"
BASELINE_PREFLIGHT="$OUTPUT_ROOT/preflight_checkpoint-192.json"
CANDIDATE_PREFLIGHT="$OUTPUT_ROOT/preflight_candidate.json"
PREFLIGHT_DECISION="$OUTPUT_ROOT/preflight_decision.json"
EVALUATION_BINDING="$COLLECTION_DIR/evaluation/evaluation.binding.json"
STATUS_FILE="$OUTPUT_ROOT/stage_status.json"
RECEIPT_FILE="$OUTPUT_ROOT/stage_receipt.json"
HARDWARE_ATTESTATION="$OUTPUT_ROOT/hardware_attestation.json"
CACHE_PREFLIGHT="$OUTPUT_ROOT/cache_preflight.json"

MODEL_ID="google/gemma-4-12B-it"
MODEL_REVISION="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MODEL_ARCHITECTURE="gemma4_unified"
MODEL_PROMPT_PROFILE="native"
WARM_START_DIGEST="cb10b81d184409bde395eb6686cb5738ad25cfb2378039761f5e679399f44f2a"
D0_RAW_SHA256="2fa768ec2546c6952bb9699a684f24acb00e79eca4a316e00074ee1524cadcab"
D0_TRAIN_SHA256="28b733db96c6ce05dbdc8d43484bdbb14445e1105958a78f9a35024aa5b3844a"
VALIDATION_SHA256="f05b944f89fa03f61c11376bd05da513f21a7b747c66bfdf80ab11908290898e"
PROTECTED_D1_SHA256="dafe9925dedacc1aaeb861f7a6891a4bdc1ce2a3be4b13d7ba999275a505124d"

SEED=20260720
TRAIN_PLAN='{"measurement+parameter":2,"multi_measurement":2,"parameter":1}'
DEVELOPMENT_PLAN='{"measurement+parameter":6,"multi_measurement":6,"parameter":3}'
COLLECTION_BETA=0.25
COLLECTION_MAX_STEPS=4
D1_CAP=20
D1_SHARE=0.5
CANDIDATE_MULTIPLIER=6
TRAIN_MAX_STEPS=32
SAVE_EVAL_STEPS=8
EVAL_MAX_STEPS=24
MINIMUM_PREFLIGHT_EXACT=5

BLOCKING_JOB_IDS="${RESEARCH_DAGGER_R1_BLOCKING_JOB_IDS:-16347744,16347745}"
GLOBAL_LOCK="${RESEARCH_DAGGER_R1_GLOBAL_LOCK:-/scratch/yx3882/.locks/research_gemma4_dagger_round1.lock}"

CURRENT_STAGE=bootstrap
FINALIZED=0
STATUS_ENABLED=0
TELEMETRY_PID=""
SOURCE_COMMIT=""
PYTHON=python

write_status() {
  local stage=$1
  local state=$2
  local detail=${3:-}
  [[ "$STATUS_ENABLED" == 1 ]] || return 0
  "$PYTHON" - "$STATUS_FILE" "$stage" "$state" "$detail" \
    "${SLURM_JOB_ID:-manual}" "$SOURCE_COMMIT" <<'PY'
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile

path = Path(sys.argv[1])
payload = {
    "contract": "research_gemma4_dagger_round1_status_v1",
    "stage": sys.argv[2],
    "state": sys.argv[3],
    "detail": sys.argv[4],
    "slurm_job_id": sys.argv[5],
    "source_commit": sys.argv[6] or None,
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  if [[ "$FINALIZED" != 1 && "$STATUS_ENABLED" == 1 ]]; then
    write_status "$CURRENT_STAGE" failed "launcher_exit_code=$status" || true
  fi
  exit "$status"
}
trap cleanup EXIT

module purge
module load anaconda3/2025.06
# shellcheck source=/dev/null
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$RESEARCH_ENV"
PYTHON="${RESEARCH_PYTHON:-python}"

mkdir -p "$OUTPUT_ROOT" "$COLLECTION_DIR" "$TRAIN_DIR" "$(dirname "$GLOBAL_LOCK")"
STATUS_ENABLED=1
write_status bootstrap running "authorized fallback; checking queue and locks"

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue is required for the duplicate-job guard" >&2
  exit 2
fi
QUEUE_OWNER="${SLURM_JOB_USER:-${USER:-$(id -un)}}"
if ! QUEUE_SNAPSHOT=$(squeue -h -u "$QUEUE_OWNER" -o "%A|%T|%j|%o"); then
  echo "cannot inspect Slurm queue; refusing fallback launch" >&2
  exit 2
fi
QUEUE_CONFLICTS=$("$PYTHON" - \
  "${SLURM_JOB_ID:-}" "$BLOCKING_JOB_IDS" "$QUEUE_SNAPSHOT" <<'PY'
import re
import sys

self_id = sys.argv[1].strip()
blocked = {value.strip() for value in sys.argv[2].split(",") if value.strip()}
conflicts = []
for raw in sys.argv[3].splitlines():
    raw = raw.strip()
    if not raw:
        continue
    fields = raw.split("|", 3)
    if len(fields) != 4:
        raise SystemExit(f"cannot parse squeue row: {raw!r}")
    job_id, state, name, command = (field.strip() for field in fields)
    if job_id == self_id:
        continue
    text = f"{name} {command}".lower()
    equivalent = bool(
        re.search(r"(?:gemma.?4|d12b|(?:^|[^a-z0-9])12b(?:[^a-z0-9]|$))", text)
        and re.search(r"(?:dagger|round.?1|(?:^|[^a-z0-9])r1(?:[^a-z0-9]|$))", text)
    )
    if job_id in blocked or equivalent:
        conflicts.append(f"{job_id}|{state}|{name}|{command}")
print("\n".join(conflicts))
PY
)
if [[ -n "$QUEUE_CONFLICTS" ]]; then
  echo "equivalent or explicitly blocked Round-1 Slurm jobs are still active:" >&2
  printf '%s\n' "$QUEUE_CONFLICTS" >&2
  exit 73
fi

exec 8>"$GLOBAL_LOCK"
if ! flock -n 8; then
  echo "another Gemma-4 DAgger Round-1 launcher owns $GLOBAL_LOCK" >&2
  exit 73
fi
exec 9>"$OUTPUT_ROOT/.job.lock"
if ! flock -n 9; then
  echo "another launcher owns $OUTPUT_ROOT" >&2
  exit 73
fi

cd "$RESEARCH_SOURCE_ROOT"
for tracked in \
  submit_research_gemma4_dagger_round1.sh \
  scripts/run_dagger_research.py \
  psse_env/dagger/research_action_preflight.py
do
  git ls-files --error-unmatch "$tracked" >/dev/null || {
    echo "required launcher/runtime file is not tracked: $tracked" >&2
    exit 2
  }
done
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "research source checkout has tracked modifications" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi
SOURCE_COMMIT=$(git rev-parse HEAD)

for required in \
  "$D0_RAW" \
  "$D0_TRAIN" \
  "$VALIDATION" \
  "$PROTECTED_D1" \
  "$WARM_START/adapter_config.json" \
  "$WARM_START/adapter_model.safetensors" \
  "$RESEARCH_SOURCE_ROOT/psse_env/requirements-sft-research.txt"
do
  if [[ ! -s "$required" ]]; then
    echo "required Round-1 input is missing or empty: $required" >&2
    exit 2
  fi
done

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
export RESEARCH_MAX_INPUT_TOKENS=32768
export RESEARCH_MAX_NEW_TOKENS=256

file_sha256() {
  "$PYTHON" -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); [h.update(x) for x in iter(lambda:f.read(1048576), b"")]; print(h.hexdigest())' "$1"
}

tree_sha256() {
  "$PYTHON" -c 'import sys; from psse_env.dagger.release_factories import checkpoint_tree_sha256; print(checkpoint_tree_sha256(sys.argv[1]))' "$1"
}

stable_sha256() {
  "$PYTHON" - "$@" <<'PY'
import hashlib
import json
import sys
payload = json.dumps(sys.argv[1:], separators=(",", ":"), ensure_ascii=False).encode()
print(hashlib.sha256(payload).hexdigest())
PY
}

binding_current() {
  local binding=$1
  local request_sha=$2
  shift 2
  [[ -s "$binding" ]] || return 1
  "$PYTHON" - "$binding" "$request_sha" "$@" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

binding = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
paths = [Path(value).resolve(strict=True) for value in sys.argv[3:]]
recorded = binding.get("outputs")
current = (
    binding.get("contract") == "research_gemma4_dagger_round1_cache_binding_v1"
    and binding.get("request_sha256") == expected
    and isinstance(recorded, list)
    and len(recorded) == len(paths)
)
if current:
    for path, item in zip(paths, recorded):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if item != {"path": str(path), "sha256": digest}:
            current = False
            break
raise SystemExit(0 if current else 3)
PY
}

write_binding() {
  local binding=$1
  local request_sha=$2
  shift 2
  "$PYTHON" - "$binding" "$request_sha" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

destination = Path(sys.argv[1])
paths = [Path(value).resolve(strict=True) for value in sys.argv[3:]]
payload = {
    "contract": "research_gemma4_dagger_round1_cache_binding_v1",
    "request_sha256": sys.argv[2],
    "outputs": [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ],
}
destination.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(
    prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

CURRENT_STAGE=prerequisites
write_status "$CURRENT_STAGE" running "attesting environment, GPU, inputs, and immutable run identity"
"$PYTHON" -m pip check
"$PYTHON" -m psse_env.sft.release_hardware --require-class rtx6000 >"$HARDWARE_ATTESTATION.tmp"
mv -f "$HARDWARE_ATTESTATION.tmp" "$HARDWARE_ATTESTATION"
"$PYTHON" -m psse_env.sft research-cache --model-choice 12b --output "$CACHE_PREFLIGHT"

TELEMETRY="$OUTPUT_ROOT/nvidia_smi_${SLURM_JOB_ID:-manual}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits \
  --loop=10 >"$TELEMETRY" 2>"$OUTPUT_ROOT/nvidia_smi_${SLURM_JOB_ID:-manual}.err" &
TELEMETRY_PID=$!

"$PYTHON" - \
  "$D0_RAW" "$D0_TRAIN" "$VALIDATION" "$PROTECTED_D1" "$WARM_START" "$RUN_IDENTITY" \
  "$SOURCE_COMMIT" "$D0_RAW_SHA256" "$D0_TRAIN_SHA256" "$VALIDATION_SHA256" \
  "$PROTECTED_D1_SHA256" "$WARM_START_DIGEST" "$TRAIN_PLAN" "$DEVELOPMENT_PLAN" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from psse_env.dagger.research_action_preflight import (
    load_validation_rows,
    normalize_validation_row,
)
from psse_env.dagger.release_factories import checkpoint_tree_sha256


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def root(row: dict) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(row.get("physical_root_fingerprint") or metadata.get("physical_root_fingerprint") or "").strip()


d0_raw, d0_train, validation, protected_d1, warm, identity_path = map(Path, sys.argv[1:7])
source_commit = sys.argv[7]
expected_hashes = sys.argv[8:12]
expected_warm = sys.argv[12]
train_plan = json.loads(sys.argv[13])
development_plan = json.loads(sys.argv[14])
for path, expected in zip((d0_raw, d0_train, validation, protected_d1), expected_hashes):
    observed = sha(path.resolve(strict=True))
    if observed != expected:
        raise SystemExit(f"input SHA-256 mismatch for {path}: {observed}")
warm_digest = checkpoint_tree_sha256(warm.resolve(strict=True))
if warm_digest != expected_warm:
    raise SystemExit(f"checkpoint-192 tree digest mismatch: {warm_digest}")

raw_rows = rows(d0_raw)
train_rows = rows(d0_train)
validation_rows = load_validation_rows(validation)
protected_payload = json.loads(protected_d1.read_text(encoding="utf-8"))
if len(validation_rows) != 27:
    raise SystemExit(f"action validation must contain exactly 27 rows, got {len(validation_rows)}")
for row in validation_rows:
    normalize_validation_row(row)
raw_roots = {root(row) for row in raw_rows}
train_roots = {root(row) for row in train_rows}
validation_roots = {root(row) for row in validation_rows}


def nested_roots(value):
    found = set()
    def visit(node):
        if isinstance(node, dict):
            candidate = root(node)
            if candidate:
                found.add(candidate)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    return found


protected_roots = nested_roots(protected_payload)
if "" in raw_roots | train_roots | validation_roots | protected_roots:
    raise SystemExit("D0, validation, and protected D1 artifacts must retain physical roots")
if len(train_rows) != 1280 or len(train_roots) != 182:
    raise SystemExit(f"unexpected D0 train closure: rows={len(train_rows)}, roots={len(train_roots)}")
if not train_roots <= raw_roots:
    raise SystemExit("D0 training roots are absent from aggregate.raw.jsonl")
if validation_roots & raw_roots:
    raise SystemExit("27-row action validation overlaps D0 physical roots")
if len(protected_roots) != 15:
    raise SystemExit(f"published protected D1 suite must contain 15 roots, got {len(protected_roots)}")
protected_sets = {
    "d0": raw_roots,
    "action_validation": validation_roots,
    "published_d1": protected_roots,
}
for left, left_roots in protected_sets.items():
    for right, right_roots in protected_sets.items():
        if left < right and left_roots & right_roots:
            raise SystemExit(f"protected input roots overlap between {left} and {right}")

identity = {
    "contract": "research_gemma4_dagger_round1_identity_v1",
    "release_eligible": False,
    "source_commit": source_commit,
    "model": {
        "model_id": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "architecture": "gemma4_unified",
        "prompt_profile": "native",
    },
    "warm_start": {"path": str(warm.resolve()), "tree_sha256": warm_digest},
    "d0_raw": {"path": str(d0_raw.resolve()), "sha256": expected_hashes[0], "roots": len(raw_roots)},
    "d0_train": {"path": str(d0_train.resolve()), "sha256": expected_hashes[1], "rows": len(train_rows), "roots": len(train_roots)},
    "validation": {"path": str(validation.resolve()), "sha256": expected_hashes[2], "rows": 27, "roots": len(validation_roots)},
    "protected_d1": {"path": str(protected_d1.resolve()), "sha256": expected_hashes[3], "roots": len(protected_roots)},
    "collection": {
        "seed": 20260720,
        "beta": 0.25,
        "max_steps": 4,
        "d1_cap": 20,
        "d1_share": 0.5,
        "candidate_multiplier": 6,
        "train_plan": train_plan,
        "development_plan": development_plan,
    },
    "training": {
        "max_steps": 32,
        "save_steps": 8,
        "eval_steps": 8,
        "max_length": 32768,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-5,
        "seed": 20260720,
        "best_metric": "eval_loss",
    },
    "preflight": {"rows": 27, "minimum_exact": 5, "requires_exact_improvement": True},
    "paired_development_evaluation": {"roots": 15, "max_steps": 24},
}
if identity_path.is_file():
    recorded = json.loads(identity_path.read_text(encoding="utf-8"))
    if recorded != identity:
        raise SystemExit("existing output directory has a different Round-1 identity")
else:
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{identity_path.name}.", suffix=".tmp", dir=identity_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, identity_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
print(json.dumps({"passed": True, "warm_start_digest": warm_digest, "validation_rows": 27, "validation_roots": len(validation_roots), "protected_d1_roots": len(protected_roots)}))
PY

COLLECTION_ARGS=(
  --d0-raw "$D0_RAW"
  --d0-train "$D0_TRAIN"
  --protected-suite "$VALIDATION"
  --protected-suite "$PROTECTED_D1"
  --adapter-path "$WARM_START"
  --output-dir "$COLLECTION_DIR"
  --model-choice 12b
  --base-model "$MODEL_ID"
  --base-revision "$MODEL_REVISION"
  --architecture "$MODEL_ARCHITECTURE"
  --prompt-profile "$MODEL_PROMPT_PROFILE"
  --seed "$SEED"
  --beta "$COLLECTION_BETA"
  --max-steps "$COLLECTION_MAX_STEPS"
  --d1-cap "$D1_CAP"
  --d1-share "$D1_SHARE"
  --candidate-multiplier "$CANDIDATE_MULTIPLIER"
  --train-plan "$TRAIN_PLAN"
  --development-plan "$DEVELOPMENT_PLAN"
)

CURRENT_STAGE=collection
write_status "$CURRENT_STAGE" running "resumable fresh five-root collection and exact 1:1 replay mixture"
"$PYTHON" scripts/run_dagger_research.py "${COLLECTION_ARGS[@]}"

"$PYTHON" - \
  "$D0_RAW" "$VALIDATION" "$PROTECTED_D1" "$COLLECTION_DIR" "$COLLECTION_AUDIT" \
  "$TRAIN_PLAN" "$DEVELOPMENT_PLAN" <<'PY'
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def root(row: dict) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(row.get("physical_root_fingerprint") or metadata.get("physical_root_fingerprint") or "").strip()


def family(row: dict) -> str:
    return str(row.get("scenario_family") or row.get("grouping", {}).get("scenario_family") or "")


d0_raw, validation, protected_d1, collection, output = map(Path, sys.argv[1:6])
train_plan = json.loads(sys.argv[6])
dev_plan = json.loads(sys.argv[7])
training = json.loads((collection / "training_scenarios.json").read_text(encoding="utf-8"))
development = json.loads((collection / "development_scenarios.json").read_text(encoding="utf-8"))
mixture = load_jsonl(collection / "round1.train.jsonl")
mixture_report = json.loads((collection / "mixture_report.json").read_text(encoding="utf-8"))
metrics = json.loads((collection / "collection_metrics.json").read_text(encoding="utf-8"))
d0_roots = {root(row) for row in load_jsonl(d0_raw)}
validation_roots = {root(row) for row in load_jsonl(validation)}


def nested_roots(value):
    found = set()
    def visit(node):
        if isinstance(node, dict):
            candidate = root(node)
            if candidate:
                found.add(candidate)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    return found


protected_d1_roots = nested_roots(json.loads(protected_d1.read_text(encoding="utf-8")))
training_roots = {root(row) for row in training}
development_roots = {root(row) for row in development}
mixture_roots = {root(row) for row in mixture}
if len(training) != 5 or len(training_roots) != 5:
    raise SystemExit("collection does not contain exactly five unique training roots")
if len(development) != 15 or len(development_roots) != 15:
    raise SystemExit("generated development closure does not contain exactly 15 roots")
if Counter(map(family, training)) != Counter(train_plan):
    raise SystemExit("five-root training family allocation differs from the frozen plan")
if Counter(map(family, development)) != Counter(dev_plan):
    raise SystemExit("15-root development family allocation differs from the frozen plan")
sets = {"d0": d0_roots, "validation": validation_roots, "protected_d1": protected_d1_roots, "training": training_roots, "development": development_roots}
for left, left_roots in sets.items():
    for right, right_roots in sets.items():
        if left < right and left_roots & right_roots:
            raise SystemExit(f"physical-root overlap between {left} and {right}")
if len(protected_d1_roots) != 15:
    raise SystemExit("published protected D1 suite no longer contains 15 roots")
if mixture_roots & (validation_roots | protected_d1_roots | development_roots):
    raise SystemExit("Round-1 mixture overlaps validation, protected D1, or generated development roots")
d0_selected = int(mixture_report.get("d0_selected") or 0)
d1_selected = int(mixture_report.get("d1_selected") or 0)
if d0_selected <= 0 or d0_selected != d1_selected:
    raise SystemExit(f"mixture is not exact 1:1 D0/D1 replay: {d0_selected}:{d1_selected}")
if len(mixture) != d0_selected + d1_selected or mixture_report.get("actual_d1_share") != 0.5:
    raise SystemExit("mixture row count/share differs from exact 1:1 replay")
if metrics.get("episodes_completed") != 5 or metrics.get("episodes_total") != 5:
    raise SystemExit("five-root collection is incomplete")
payload = {
    "contract": "research_gemma4_dagger_round1_collection_audit_v1",
    "passed": True,
    "training_roots": sorted(training_roots),
    "development_roots": sorted(development_roots),
    "validation_roots": sorted(validation_roots),
    "protected_d1_roots": sorted(protected_d1_roots),
    "mixture_rows": len(mixture),
    "d0_selected": d0_selected,
    "d1_selected": d1_selected,
    "actual_d1_share": 0.5,
    "round1_train_sha256": hashlib.sha256((collection / "round1.train.jsonl").read_bytes()).hexdigest(),
}
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

ROUND1_TRAIN="$COLLECTION_DIR/round1.train.jsonl"
"$PYTHON" - \
  "$TRAIN_RECIPE" "$ROUND1_TRAIN" "$VALIDATION" "$WARM_START" \
  "$SOURCE_COMMIT" "$WARM_START_DIGEST" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


recipe_path, train, validation, warm = map(Path, sys.argv[1:5])
recipe = {
    "contract": "research_gemma4_dagger_round1_training_recipe_v1",
    "source_commit": sys.argv[5],
    "train_path": str(train.resolve(strict=True)),
    "train_sha256": sha(train),
    "validation_path": str(validation.resolve(strict=True)),
    "validation_sha256": sha(validation),
    "initial_adapter_path": str(warm.resolve(strict=True)),
    "initial_adapter_tree_sha256": sys.argv[6],
    "max_steps": 32,
    "save_steps": 8,
    "eval_steps": 8,
    "max_length": 32768,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-5,
    "seed": 20260720,
    "select_best_eval_loss": True,
}
if recipe_path.is_file():
    if json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
        raise SystemExit("existing Trainer state is bound to a different Round-1 recipe")
else:
    stale = [path.name for path in recipe_path.parent.glob("checkpoint-*")]
    if stale or (recipe_path.parent / "lora").exists():
        raise SystemExit("Trainer artifacts exist without a frozen training recipe")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{recipe_path.name}.", suffix=".tmp", dir=recipe_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(recipe, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, recipe_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
PY

training_current() {
  [[ -s "$TRAIN_COMPLETION" && -s "$CANDIDATE/adapter_config.json" && -s "$CANDIDATE/adapter_model.safetensors" ]] || return 1
  local candidate_digest
  candidate_digest=$(tree_sha256 "$CANDIDATE")
  "$PYTHON" - "$TRAIN_COMPLETION" "$TRAIN_RECIPE" "$CANDIDATE" "$candidate_digest" "$TRAIN_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

completion = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
recipe = Path(sys.argv[2]).resolve(strict=True)
candidate = Path(sys.argv[3]).resolve(strict=True)
train_dir = Path(sys.argv[5]).resolve(strict=True)
expected_steps = [8, 16, 24, 32]
current = (
    completion.get("contract") == "research_gemma4_dagger_round1_training_completion_v1"
    and completion.get("recipe_sha256") == hashlib.sha256(recipe.read_bytes()).hexdigest()
    and completion.get("candidate_path") == str(candidate)
    and completion.get("candidate_tree_sha256") == sys.argv[4]
    and completion.get("global_step") == 32
)
for step in expected_steps:
    path = train_dir / f"checkpoint-{step}"
    state = path / "trainer_state.json"
    if not state.is_file() or int(json.loads(state.read_text(encoding="utf-8")).get("global_step", -1)) != step:
        current = False
raise SystemExit(0 if current else 3)
PY
}

CURRENT_STAGE=training
write_status "$CURRENT_STAGE" running "32-step checkpoint-192 continuation; checkpoint/eval every 8"
if training_current; then
  echo "[training] recipe-bound completed candidate exists; skipping"
else
  resume_args=()
  if [[ ! -d "$CANDIDATE" ]]; then
    checkpoint_path=$("$PYTHON" -m psse_env.sft.research_checkpoint \
      --output-dir "$TRAIN_DIR" \
      --expected-base-model "$MODEL_ID")
    if [[ -n "$checkpoint_path" ]]; then
      echo "[training] resuming from newest complete checkpoint: $checkpoint_path"
      resume_args+=(--resume-from-checkpoint "$checkpoint_path")
    fi
  fi
  "$PYTHON" -m psse_env.sft research-train \
    --model-choice 12b \
    --train "$ROUND1_TRAIN" \
    --validation "$VALIDATION" \
    --initial-adapter "$WARM_START" \
    --output-dir "$TRAIN_DIR" \
    --max-length 32768 \
    --strict-prompt-length \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5 \
    --epochs 1 \
    --max-steps "$TRAIN_MAX_STEPS" \
    --logging-steps 1 \
    --save-steps "$SAVE_EVAL_STEPS" \
    --eval-steps "$SAVE_EVAL_STEPS" \
    --select-best-eval-loss \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --smoke-steps 1 \
    --reload-canaries 1 \
    --seed "$SEED" \
    --run-name research-gemma4-12b-dagger-round1-fallback \
    --report-to none \
    "${resume_args[@]}"
  candidate_digest=$(tree_sha256 "$CANDIDATE")
  "$PYTHON" - "$TRAIN_COMPLETION" "$TRAIN_RECIPE" "$CANDIDATE" "$candidate_digest" "$TRAIN_DIR" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

output, recipe, candidate = map(Path, sys.argv[1:4])
train_dir = Path(sys.argv[5])
report = json.loads((train_dir / "research_run.json").read_text(encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit("research-train report did not pass")
for step in (8, 16, 24, 32):
    checkpoint = train_dir / f"checkpoint-{step}"
    for name in ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / name).is_file():
            raise SystemExit(f"missing checkpoint artifact: {checkpoint / name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != step:
        raise SystemExit(f"checkpoint-{step} has the wrong global step")
payload = {
    "contract": "research_gemma4_dagger_round1_training_completion_v1",
    "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
    "candidate_path": str(candidate.resolve(strict=True)),
    "candidate_tree_sha256": sys.argv[4],
    "global_step": 32,
    "checkpoint_steps": [8, 16, 24, 32],
}
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
  training_current
fi

preflight_one() {
  local label=$1
  local adapter=$2
  local output=$3
  local binding="$output.binding.json"
  local adapter_digest request_sha
  adapter_digest=$(tree_sha256 "$adapter")
  request_sha=$(stable_sha256 \
    research_gemma4_dagger_round1_preflight_request_v1 \
    "$SOURCE_COMMIT" "$VALIDATION_SHA256" "$adapter_digest" \
    "$MODEL_ID" "$MODEL_REVISION" "$MODEL_ARCHITECTURE" "$MODEL_PROMPT_PROFILE")
  if binding_current "$binding" "$request_sha" "$output"; then
    echo "[preflight:$label] bound report exists; skipping"
    return 0
  fi
  "$PYTHON" -m psse_env.dagger.research_action_preflight run \
    --validation "$VALIDATION" \
    --adapter "$adapter" \
    --output "$output"
  "$PYTHON" - "$output" "$adapter" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("contract") != "research_gemma4_12b_action_preflight_v1":
    raise SystemExit("unexpected action-preflight contract")
if report.get("example_count") != 27 or len(report.get("example_ids") or []) != 27:
    raise SystemExit("action preflight did not score all 27 rows")
if Path(str(report.get("adapter_path") or "")).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("action preflight adapter path mismatch")
PY
  write_binding "$binding" "$request_sha" "$output"
}

CURRENT_STAGE=action_preflight
write_status "$CURRENT_STAGE" running "greedy checkpoint-192 and candidate action preflights on all 27 rows"
preflight_one baseline "$WARM_START" "$BASELINE_PREFLIGHT"
preflight_one candidate "$CANDIDATE" "$CANDIDATE_PREFLIGHT"

set +e
"$PYTHON" -m psse_env.dagger.research_action_preflight compare \
  --baseline "$BASELINE_PREFLIGHT" \
  --candidate "$CANDIDATE_PREFLIGHT" \
  --minimum-exact "$MINIMUM_PREFLIGHT_EXACT" \
  --output "$PREFLIGHT_DECISION"
PREFLIGHT_STATUS=$?
set -e
if (( PREFLIGHT_STATUS != 0 && PREFLIGHT_STATUS != 2 )); then
  exit "$PREFLIGHT_STATUS"
fi

write_receipt() {
  local outcome=$1
  "$PYTHON" - \
    "$RECEIPT_FILE" "$outcome" "$RUN_IDENTITY" "$COLLECTION_AUDIT" \
    "$TRAIN_COMPLETION" "$BASELINE_PREFLIGHT" "$CANDIDATE_PREFLIGHT" \
    "$PREFLIGHT_DECISION" "$COLLECTION_DIR/evaluation/comparison.json" \
    "$SOURCE_COMMIT" "${SLURM_JOB_ID:-manual}" "$TELEMETRY" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def artifact(value: str):
    path = Path(value)
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


output = Path(sys.argv[1])
decision = json.loads(Path(sys.argv[8]).read_text(encoding="utf-8"))
comparison = None
if Path(sys.argv[9]).is_file():
    comparison = json.loads(Path(sys.argv[9]).read_text(encoding="utf-8"))
payload = {
    "contract": "research_gemma4_dagger_round1_stage_receipt_v1",
    "outcome": sys.argv[2],
    "passed": sys.argv[2] == "complete",
    "release_eligible": False,
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[10],
    "slurm_job_id": sys.argv[11],
    "telemetry_path": str(Path(sys.argv[12]).resolve()),
    "artifacts": {
        "run_identity": artifact(sys.argv[3]),
        "collection_audit": artifact(sys.argv[4]),
        "training_completion": artifact(sys.argv[5]),
        "baseline_preflight": artifact(sys.argv[6]),
        "candidate_preflight": artifact(sys.argv[7]),
        "preflight_decision": artifact(sys.argv[8]),
        "paired_comparison": artifact(sys.argv[9]),
    },
    "preflight_decision": decision,
    "paired_evaluation": comparison,
}
output.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

if (( PREFLIGHT_STATUS == 2 )); then
  CURRENT_STAGE=gated_stop
  write_status "$CURRENT_STAGE" complete "candidate failed action gate; paired generated development evaluation was not run"
  write_receipt gated_stop
  FINALIZED=1
  echo "RESEARCH_GEMMA4_DAGGER_ROUND1_GATED_STOP"
  echo "decision: $PREFLIGHT_DECISION"
  echo "receipt:  $RECEIPT_FILE"
  exit 0
fi

CURRENT_STAGE=paired_development_evaluation
write_status "$CURRENT_STAGE" running "gate passed; paired baseline/candidate evaluation on 15 generated roots"
CANDIDATE_DIGEST=$(tree_sha256 "$CANDIDATE")
EVALUATION_REQUEST=$(stable_sha256 \
  research_gemma4_dagger_round1_paired_evaluation_request_v1 \
  "$SOURCE_COMMIT" "$WARM_START_DIGEST" "$CANDIDATE_DIGEST" \
  "$(file_sha256 "$COLLECTION_DIR/development_scenarios.json")" \
  "$(file_sha256 "$PREFLIGHT_DECISION")" "$EVAL_MAX_STEPS")
EVALUATION_OUTPUTS=(
  "$COLLECTION_DIR/evaluation/bc0_eval.json"
  "$COLLECTION_DIR/evaluation/r1_eval.json"
  "$COLLECTION_DIR/evaluation/comparison.json"
  "$COLLECTION_DIR/research_run_report.json"
)
if binding_current "$EVALUATION_BINDING" "$EVALUATION_REQUEST" "${EVALUATION_OUTPUTS[@]}"; then
  echo "[evaluation] bound paired 15-root result exists; skipping"
else
  "$PYTHON" scripts/run_dagger_research.py \
    "${COLLECTION_ARGS[@]}" \
    --eval-r1-adapter "$CANDIDATE" \
    --eval-max-steps "$EVAL_MAX_STEPS"
  "$PYTHON" - "$COLLECTION_DIR/evaluation/comparison.json" "$WARM_START" "$CANDIDATE" <<'PY'
import json
from pathlib import Path
import sys
comparison = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
roots = comparison.get("paired_physical_roots") or []
if len(roots) != 15 or len(set(roots)) != 15:
    raise SystemExit("paired evaluation did not use exactly 15 unique generated roots")
if Path(str(comparison.get("bc0_adapter") or "")).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("paired evaluation baseline adapter mismatch")
if Path(str(comparison.get("r1_adapter") or "")).resolve() != Path(sys.argv[3]).resolve():
    raise SystemExit("paired evaluation candidate adapter mismatch")
if comparison.get("max_steps") != 24:
    raise SystemExit("paired evaluation horizon mismatch")
PY
  write_binding "$EVALUATION_BINDING" "$EVALUATION_REQUEST" "${EVALUATION_OUTPUTS[@]}"
fi

CURRENT_STAGE=complete
write_status "$CURRENT_STAGE" complete "bounded Round-1 fallback completed"
write_receipt complete
FINALIZED=1

echo "RESEARCH_GEMMA4_DAGGER_ROUND1_COMPLETE"
echo "collection: $COLLECTION_DIR"
echo "candidate:  $CANDIDATE"
echo "decision:   $PREFLIGHT_DECISION"
echo "evaluation: $COLLECTION_DIR/evaluation/comparison.json"
echo "receipt:    $RECEIPT_FILE"
