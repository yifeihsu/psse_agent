#!/usr/bin/env bash
#SBATCH --job-name=fp-xcpu
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/xcpu-%j.out
# Follow-ups of a finished cell (conventional-only cell, 2026-10-23), CPU part:
# the extra evaluation suites (build_eval_suites.py: the validation roots for
# the round selection and the hard-draw stress roots), the fresh expert
# aggregate the matched cloning control trains on (the aggregate builder with
# the D0 settings, SEED_CONTROL and CONTROL_D0_PLAN) and the control's mixture.
# Output-guarded; the aggregate restarts if requeued, like stage 0.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
EXTRA=$OUT/extra
mkdir -p "$EXTRA" "$LOGS"
echo "[xcpu] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$OUT/r2/round_summary.json" ]] || { echo "the cell's chain has not finished" >&2; exit 2; }
: "${EXTRA_VALIDATION_SOURCE:?}" "${EXTRA_HARD_SOURCE:?}" "${CONTROL_D0_PLAN:?}" "${SEED_CONTROL:?}"
cd "$SRC"

if [[ ! -s "$EXTRA/suites/report.json" ]]; then
  "$PY" research/hpc/full_pipeline_20260907/build_eval_suites.py --cell-dir "$PIPE" \
    --validation-source "$EXTRA_VALIDATION_SOURCE" --hard-source "$EXTRA_HARD_SOURCE" \
    --output-dir "$EXTRA/suites"
fi

if [[ ! -s "$EXTRA/control_d0.done" ]]; then
  opendss_local_model
  stage_cleanup() { rm -rf "$OPENDSS_LOCAL"; }
  trap stage_cleanup EXIT
  system_args=(--system "$SYSTEM" --admission-mode "$ADMISSION_MODE" --evidence-profile "$EVIDENCE_PROFILE" --hif-signature-mode "$HIF_SIGNATURE_MODE"
    --normalized-residual-threshold "$NORMALIZED_RESIDUAL_THRESHOLD" --progress-basis "$PROGRESS_BASIS")
  label_args=()
  if [[ "${UNVERIFIED_EXPERT_LABEL_POLICY:-abort}" == stop ]]; then label_args+=(--stop-episode-on-unverified-expert-label); fi
  waveform_corpus_files
  waveform_args=()
  for corpus in "${HIF_CORPUS_FILES[@]}"; do waveform_args+=(--hif-corpus "$corpus"); done
  if [[ -n "$IMBALANCE_CORPUS" ]]; then waveform_args+=(--imbalance-corpus "$IMBALANCE_CORPUS"); fi
  rm -rf "$EXTRA/control_d0"
  mkdir -p "$EXTRA/control_d0"
  printf '%s\n' "$CONTROL_D0_PLAN" > "$EXTRA/control_d0_plan.json"
  "$PY" -m psse_env.examples.generate_round0_aggregate \
    --research \
    --output-dir "$EXTRA/control_d0" \
    --plan "$EXTRA/control_d0_plan.json" \
    --seed "$SEED_CONTROL" \
    --protocol canonical \
    --max-steps "$D0_MAX_STEPS" \
    --counterfactuals-per-scenario "$D0_COUNTERFACTUALS" \
    --hif-alpha-grid "$HIF_ALPHA_GRID" \
    --hif-r-grid "$HIF_R_GRID" \
    --hif-max-scans "$HIF_MAX_SCANS" \
    --min-measurement-error-sigma "$MEASUREMENT_ERROR_MIN_SIGMA" \
    --topology-effects "$TOPOLOGY_EFFECTS" \
    "${system_args[@]}" \
    "${waveform_args[@]}" \
    "${label_args[@]}" \
    > "$EXTRA/control_d0_console.json"
  "$PY" - "$EXTRA/control_d0" "$EXTRA/control_d0.done" "$SEED_CONTROL" "$EVIDENCE_PROFILE" "$EXPERT_VARIANT" <<'PY'
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

d0 = Path(sys.argv[1])
rows = [json.loads(line) for line in (d0 / "aggregate.raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("the fresh expert aggregate produced no rows")
def root(row):
    return str(row.get("physical_root_fingerprint") or (row.get("metadata") or {}).get("physical_root_fingerprint") or "")
payload = {
    "contract": "conventional_only_control_aggregate_done_v1",
    "seed": int(sys.argv[3]),
    "evidence_profile": sys.argv[4],
    "expert_variant": sys.argv[5],
    "raw_rows": len(rows),
    "raw_roots": len({root(row) for row in rows}),
    "raw_families": dict(sorted(Counter(row.get("scenario_family") for row in rows).items())),
    "raw_sha256": hashlib.sha256((d0 / "aggregate.raw.jsonl").read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ("raw_rows", "raw_roots", "raw_families")}))
PY
fi

# The control's mixture: R1's mixture with every collected row replaced by a
# fresh expert row of its family (build_control_mixture.py).  Built here so a
# family shortfall fails on the CPU, before the GPU job starts.
if [[ ! -s "$EXTRA/control/mixture.report.json" ]]; then
  mkdir -p "$EXTRA/control"
  "$PY" research/hpc/full_pipeline_20260907/build_control_mixture.py --cell-dir "$PIPE" \
    --fresh-aggregate "$EXTRA/control_d0" \
    --exclude-suite "$EXTRA/suites/validation.json" --exclude-suite "$EXTRA/suites/hard.json" \
    --seed "$SEED_CONTROL" --evidence-profile "$EVIDENCE_PROFILE" --output "$EXTRA/control/mixture.jsonl"
fi
echo fp-xcpu-complete
