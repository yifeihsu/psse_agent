#!/usr/bin/env bash
#SBATCH --job-name=fp-collect
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/collect-%j.out
# Full pipeline, DAgger collection for PIPELINE_ROUND (r1 or r2).
#
# The round's student rolls out on its training suite with the expert
# labelling every visited state (beta 0.25, 12-step episodes); the safe rows
# are mixed 1:1 with the round's pool.  Round 1 pools the expert aggregate;
# round 2 pools the aggregate plus round 1's labelled rows, so the mixture
# is D2 against D0 + D1.  The collector keeps a completed-root ledger, so a
# requeue resumes.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
ROUND=${PIPELINE_ROUND:?PIPELINE_ROUND (r1 or r2) must be exported by the submitter}
round_bindings "$ROUND"
echo "[collect:$ROUND] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$STUDENT/adapter_model.safetensors" ]] || { echo "student adapter missing: $STUDENT" >&2; exit 2; }
[[ -s "$TRAINING_SUITE" && -s "$DEVELOPMENT_SUITE" ]] || { echo "suite files missing" >&2; exit 2; }
mkdir -p "$ROUND_DIR"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-collect-$ROUND-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-collect-$ROUND-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
opendss_local_model
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

if [[ "$ROUND" == r2 && ! -s "$ROUND_DIR/pool/manifest.json" ]]; then
  # The round-2 pool: every D0 raw row plus every round-1 labelled row, and
  # the training views that go with them.  Raw rows are rebound by
  # example_id when the view is refreshed, so the ids must stay unique.
  mkdir -p "$ROUND_DIR/pool"
  "$PY" - "$D0_DIR/aggregate.raw.jsonl" "$R1_DIR/collection/d1.all.jsonl" \
    "$D0_DIR/aggregate.train_view.jsonl" "$R1_DIR/collection/d1.chat.jsonl" "$ROUND_DIR/pool" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
raw = rows(sys.argv[1]) + rows(sys.argv[2])
train = rows(sys.argv[3]) + rows(sys.argv[4])
ids = [str(row.get("example_id") or "") for row in raw]
if "" in ids or len(set(ids)) != len(ids):
    raise SystemExit("pool raw rows need unique example ids")
train_ids = {str(row.get("example_id") or "") for row in train}
if "" in train_ids or not train_ids <= set(ids):
    raise SystemExit("pool training rows must rebind to pool raw rows by example id")
pool = Path(sys.argv[5])
for name, data in (("raw.jsonl", raw), ("train.jsonl", train)):
    (pool / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in data), encoding="utf-8")
manifest = {
    "contract": "research_full_pipeline_round2_pool_v1",
    "raw_rows": len(raw),
    "train_rows": len(train),
    "d0_raw": sys.argv[1],
    "d1_all": sys.argv[2],
    "d0_train": sys.argv[3],
    "d1_chat": sys.argv[4],
    "raw_sha256": hashlib.sha256((pool / "raw.jsonl").read_bytes()).hexdigest(),
    "train_sha256": hashlib.sha256((pool / "train.jsonl").read_bytes()).hexdigest(),
}
(pool / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"raw_rows": len(raw), "train_rows": len(train)}))
PY
fi

cd "$SRC"
collection_args "$ROUND"
if [[ ! -s "$ROUND_DIR/collection.done" ]]; then
  "$PY" scripts/run_dagger_research.py "${COLLECTION_ARGS[@]}" > "$ROUND_DIR/collection_console.json"
  "$PY" - "$COLLECTION_DIR" "$ROUND_DIR/collection.done" "$ROUND" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

collection = Path(sys.argv[1])
report = json.loads((collection / "research_run_report.json").read_text(encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit("research run report did not pass")
metrics = report["collection_metrics"]
if metrics["episodes_completed"] != metrics["episodes_total"]:
    raise SystemExit("collection is incomplete")
training = json.loads((collection / "training_scenarios.json").read_text(encoding="utf-8"))
development = json.loads((collection / "development_scenarios.json").read_text(encoding="utf-8"))
mixture = collection / "round1.train.jsonl"
rows = sum(1 for line in mixture.read_text(encoding="utf-8").splitlines() if line.strip())
payload = {
    "contract": "research_full_pipeline_collection_done_v1",
    "round": sys.argv[3],
    "research_profile": report.get("research_profile"),
    "training_roots": len(training),
    "training_families": dict(sorted(Counter(row["grouping"]["scenario_family"] for row in training).items())),
    "development_roots": len(development),
    "development_families": dict(sorted(Counter(row["grouping"]["scenario_family"] for row in development).items())),
    "eligible_rows_by_scenario_family": metrics.get("eligible_rows_by_scenario_family"),
    "label_yield": metrics.get("label_yield"),
    "mixture_rows": rows,
    "mixture_sha256": hashlib.sha256(mixture.read_bytes()).hexdigest(),
    "mixture_report": report.get("mixture"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({k: payload[k] for k in ("round", "training_roots", "label_yield", "mixture_rows")}))
PY
fi
echo "fp-collect-$ROUND-complete"
