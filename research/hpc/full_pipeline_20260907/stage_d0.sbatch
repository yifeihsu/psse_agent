#!/usr/bin/env bash
#SBATCH --job-name=fp-d0
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260907/logs/d0-%j.out
# Full pipeline, stage 0 (CPU only): the expert aggregate that trains BC0,
# then the DAgger suite drawn around it.
#
# The round-0 generator rolls the expert out on every D0 root (beta = 1),
# forces the recovery classes through counterfactual branches, audits every
# episode against truth, and splits the roots into train / validation / test.
# It is not resumable, so a requeue restarts it; the suite builder then draws
# the per-round training roots and the shared development roots excluding
# every D0 and protected root.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260907
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
mkdir -p "$OUT" "$LOGS"
echo "[d0] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} cpus ${SLURM_CPUS_PER_TASK:-?} $(date -u +%FT%TZ)"

if [[ ! -s "$OUT/prerequisites.json" ]]; then
  bash "$PIPE/prerequisites.sh" --with-tests --output "$OUT/prerequisites.json"
fi
opendss_local_model
stage_cleanup() { rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

cd "$SRC"
if [[ ! -s "$OUT/d0.done" ]]; then
  rm -rf "$D0_DIR"
  mkdir -p "$D0_DIR"
  printf '%s\n' "$D0_PLAN" > "$OUT/d0_plan.json"
  "$PY" -m psse_env.examples.generate_round0_aggregate \
    --research \
    --output-dir "$D0_DIR" \
    --plan "$OUT/d0_plan.json" \
    --seed "$SEED_D0" \
    --protocol canonical \
    --max-steps "$D0_MAX_STEPS" \
    --counterfactuals-per-scenario "$D0_COUNTERFACTUALS" \
    --hif-alpha-grid "$HIF_ALPHA_GRID" \
    --hif-r-grid "$HIF_R_GRID" \
    --hif-max-scans "$HIF_MAX_SCANS" \
    --hif-corpus "$HIF_CORPUS_TRAIN" \
    --hif-corpus "$HIF_CORPUS_VALID" \
    --imbalance-corpus "$IMBALANCE_CORPUS" \
    > "$OUT/d0_console.json"
  "$PY" - "$D0_DIR" "$OUT/d0.done" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

d0 = Path(sys.argv[1])
def rows(name):
    path = d0 / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def root(row):
    for container in (row, row.get("metadata") or {}):
        if isinstance(container, dict) and container.get("physical_root_fingerprint"):
            return str(container["physical_root_fingerprint"])
    return ""
raw = rows("aggregate.raw.jsonl")
train = rows("aggregate.train_view.jsonl")
validation = rows("aggregate.validation.jsonl")
test = rows("aggregate.test.jsonl")
if not raw or not train or not validation:
    raise SystemExit("the expert aggregate did not produce raw, train, and validation rows")
train_roots = {root(r) for r in train}
validation_roots = {root(r) for r in validation}
if train_roots & validation_roots:
    raise SystemExit("train and validation roots overlap")
provenance = json.loads((d0 / "aggregate.generation_provenance.json").read_text(encoding="utf-8"))
payload = {
    "contract": "research_full_pipeline_d0_done_v1",
    "raw_rows": len(raw),
    "raw_roots": len({root(r) for r in raw}),
    "raw_families": dict(sorted(Counter(r.get("scenario_family") for r in raw).items())),
    "train_rows": len(train),
    "train_roots": len(train_roots),
    "train_families": dict(sorted(Counter((r.get("metadata") or {}).get("scenario_family") or r.get("scenario_family") for r in train).items())),
    "validation_rows": len(validation),
    "validation_roots": len(validation_roots),
    "test_rows": len(test),
    "release_eligible": provenance.get("release_eligible"),
    "release_failures": provenance.get("release_failures"),
    "train_view_sha256": hashlib.sha256((d0 / "aggregate.train_view.jsonl").read_bytes()).hexdigest(),
    "raw_sha256": hashlib.sha256((d0 / "aggregate.raw.jsonl").read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({k: payload[k] for k in ("raw_rows", "raw_roots", "train_rows", "validation_rows", "test_rows", "release_eligible")}))
PY
fi

if [[ ! -s "$OUT/suite.done" ]]; then
  protected=(--protected-suite "$BC0_SUITE")
  if [[ -s "$TRACE_VALIDATION" ]]; then protected+=(--protected-suite "$TRACE_VALIDATION"); fi
  "$PY" "$PIPE/build_suite.py" \
    --source-root "$SRC" \
    --d0-raw "$D0_DIR/aggregate.raw.jsonl" \
    "${protected[@]}" \
    --round-train-plan "$ROUND_TRAIN_PLAN" \
    --development-plan "$DEVELOPMENT_PLAN" \
    --rounds "$DAGGER_ROUNDS" \
    --seed "$SEED_SUITE" \
    --candidate-multiplier "$CANDIDATE_MULTIPLIER" \
    --output-dir "$SUITE_DIR" \
    > "$OUT/suite_console.json"
  cp "$SUITE_DIR/manifest.json" "$OUT/suite.done"
fi
echo fp-d0-complete
