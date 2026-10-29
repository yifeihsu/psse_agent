#!/usr/bin/env bash
#SBATCH --job-name=fp-r2gate
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20261022_conventional/logs/r2gate-%j.out
# Round-2 rerun (2026-10-29, note section 9): run after r2c, before r2t.  Exit 0
# only if the cumulative mixture keeps all 437 round-1 rows (retained fraction
# 1.0) and its exact label-conflict audit finds no conflict; otherwise exit 3,
# so r2t (afterok on this job) never starts on that mixture.
set -euo pipefail
P=/scratch/yx3882/research_full_pipeline_20261022_conventional
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python
echo "[r2gate] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
"$PY" - "$P/out/r2/collection/mixture_report.json" "$P/out/r2/collection.done" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
done = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
conflicts = report.get("label_conflicts") or {}
checks = {
    "mixture contract research_dagger_cumulative_mixture_v2": report.get("contract") == "research_dagger_cumulative_mixture_v2",
    "prior_onpolicy_selected == 437": report.get("prior_onpolicy_selected") == 437,
    "prior_onpolicy_retained_fraction == 1.0": report.get("prior_onpolicy_retained_fraction") == 1.0,
    "label_conflicts audited (contract present)": conflicts.get("contract") == "research_mixture_exact_label_conflicts_v1",
    "label_conflicts conflicting_observations == 0": conflicts.get("conflicting_observations") == 0,
    "label_conflicts conflicting_rows == 0": conflicts.get("conflicting_rows") == 0,
    "collection.done round_aggregate == cumulative": done.get("round_aggregate") == "cumulative",
}
keys = ("d0_available", "d0_selected", "d1_available", "d1_selected", "prior_onpolicy_available",
        "prior_onpolicy_selected", "prior_onpolicy_retained_fraction", "prior_onpolicy_rows_by_round",
        "onpolicy_selected", "mixture_rows", "actual_onpolicy_share", "seed")
summary = {key: report.get(key) for key in keys}
summary["label_conflicts"] = {key: conflicts.get(key) for key in (
    "total_rows", "audited_rows", "unaudited_rows", "unique_observations", "conflicting_observations",
    "conflicting_rows", "conflict_rate", "conflicts")}
print(json.dumps({"checks": checks, "passed": all(checks.values()), "mixture": summary}, indent=1, sort_keys=True))
sys.exit(0 if all(checks.values()) else 3)
PY
echo "fp-r2gate-passed $(date -u +%FT%TZ)"
