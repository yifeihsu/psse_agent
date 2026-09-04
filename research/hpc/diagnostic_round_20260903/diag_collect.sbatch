#!/usr/bin/env bash
#SBATCH --job-name=diagcol
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_diag_round_20260903/logs/collect-%j.out
# Diagnostic-family research round (2026-09-03), stage 1.
#
# Learner-in-the-loop collection on HIF, measurement+HIF, three-phase
# unbalance, and the balanced telemetry control (research preset
# "diagnostic": 12/6/12/6 training roots, 6/3/6/3 development roots), then
# the D0/D1 mixture.  Student and paired baseline: BC0 12B checkpoint-192.
# The research script keeps a completed-root ledger, so a requeue resumes.
set -euo pipefail
ROUND=/scratch/yx3882/research_diag_round_20260903
# shellcheck source=round.env
source "$ROUND/round.env"
round_environment
mkdir -p "$OUT" "$LOGS"
echo "[collect] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"

if [[ ! -s "$OUT/prerequisites.json" ]]; then
  bash "$ROUND/prerequisites.sh" --with-tests --output "$OUT/prerequisites.json"
fi
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-collect-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true

cd "$SRC"
collection_args
if [[ ! -s "$OUT/collection.done" ]]; then
  "$PY" scripts/run_dagger_research.py "${COLLECTION_ARGS[@]}" > "$OUT/collection_console.json"
  "$PY" - "$COLLECTION_DIR" "$OUT/collection.done" <<'PY'
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
    "contract": "research_diagnostic_round_collection_done_v1",
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
print(json.dumps(payload, indent=2, sort_keys=True))
PY
fi
echo diag-collect-complete
