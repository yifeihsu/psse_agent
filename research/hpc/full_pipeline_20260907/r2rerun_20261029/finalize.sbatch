#!/usr/bin/env bash
#SBATCH --job-name=fp-r2final
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20261022_conventional/logs/r2final-%j.out
# Round-2 rerun (2026-10-29, note section 9): after r2e and the new R2's eight
# evaluations, rebuild the pooled, budget and follow-up tables with the new R2,
# run r2rerun_tables.py, compare out/d0, out/bc0, out/r1 and out/extra with the
# pre-deploy snapshot, and archive the small records.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
P=/scratch/yx3882/research_full_pipeline_20261022_conventional
R=$P/r2rerun_20261029
source "$P/pipeline.env"
pipeline_environment >/dev/null
echo "[r2final] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
cd "$P/source"
T=research/hpc/full_pipeline_20260907
"$PY" $T/pooled_tables.py --cell-dir "$P" --output "$P/out/extra/pooled_tables.json" > "$R/pooled_tables.console.txt" 2>&1
"$PY" $T/budget_tables.py --cell-dir "$P" --output "$P/out/extra/budget_tables.json" > "$R/budget_tables.console.txt" 2>&1
"$PY" $T/followup_tables.py --cell-dir "$P" --r2-evaluation r2 --control-evaluation r2_control_pool_20261023 \
  --output "$P/out/extra/followup_tables.json" > "$R/followup_tables.console.txt" 2>&1
cd "$R"
"$PY" r2rerun_tables.py --cell-dir "$P" --traces 16 --output "$R/r2rerun_tables.json" > "$R/r2rerun_tables.console.txt" 2>&1
for d in d0 bc0 r1 extra; do
  (cd "$P/out" && find "$d" -printf '%p\t%y\t%s\t%T@\n' | sort) > "$R/snapshot_final_$d.tsv"
done
"$PY" - "$R" > "$R/snapshot_check.txt" <<'PY'
import re
import sys
from pathlib import Path

R = Path(sys.argv[1])
NEW = ("test2_40_r2", "test_budget12_r2", "test_budget16_r2", "test_budget20_r2", "test2_budget12_r2",
       "test2_budget20_r2", "validation_r2", "hard_r2")
expected = re.compile(r"^extra/(evaluations/(%s)(/|\.console\.json$|$)|specs/(%s)\.env$|"
                      r"(evaluations|specs)/[^/]*_pool_20261023|"
                      r"(pooled_tables|pooled_tables_episodes|budget_tables|followup_tables)(_pool_20261023)?\.(json|md|jsonl)$|"
                      r"evaluations$|specs$|extra$)" % ("|".join(NEW), "|".join(NEW)))
for d in ("d0", "bc0", "r1", "extra"):
    before = {line.split("\t")[0]: line for line in (R / f"snapshot_predeploy_{d}.tsv").read_text().splitlines()}
    after = {line.split("\t")[0]: line for line in (R / f"snapshot_final_{d}.tsv").read_text().splitlines()}
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    if d == "extra":
        bad = [p for p in removed + added + changed if not expected.match(p) and p != "extra"]
        print(f"{d}: removed {len(removed)}, added {len(added)}, changed {len(changed)}; outside the set-aside and new records: {len(bad)}")
        for p in bad[:20]:
            print("   unexpected", p)
    else:
        print(f"{d}: removed {len(removed)}, added {len(added)}, changed {len(changed)}" + ("" if removed or added or changed else " (identical)"))
        for p in (removed + added + changed)[:20]:
            print("   ", p)
PY
JOBS=21369734,21369831,21369832,21369834,21369835,21369837,21369843,21369846,21369847,21369848,21369850,21369851
{ sacct -D -j "$JOBS" -X -o JobID%10,JobName%10,State%12,Elapsed%10,NodeList%10,Start%20,End%20
  NODES=$(sacct -D -j "$JOBS" -X -n -o NodeList%20 | tr -s ' \n' ',' | sed 's/^,//;s/,$//')
  sinfo -N -n "$NODES" -o "%N %G" | sort -u; } > "$R/jobs.txt" 2>&1 || true
cd "$P"
files="r2rerun_20261029 out/pipeline_summary.json out/extra/pooled_tables.json out/extra/pooled_tables.md out/extra/pooled_tables_episodes.jsonl out/extra/budget_tables.json out/extra/budget_tables.md out/extra/followup_tables.json out/extra/followup_tables.md"
for f in collection.done training.done round_summary.json evaluation_console.json collection_console.json collection/mixture_report.json collection/collection_metrics.json collection/research_run_report.json collection/evaluation/comparison.json training/research_run.json; do
  [[ -e "out/r2/$f" ]] && files="$files out/r2/$f"
done
for n in test2_40_r2 test_budget12_r2 test_budget16_r2 test_budget20_r2 test2_budget12_r2 test2_budget20_r2 validation_r2 hard_r2; do
  files="$files out/extra/specs/$n.env out/extra/evaluations/$n.console.json"
  for f in summary.json policies.json episodes.jsonl comparison.json; do files="$files out/extra/evaluations/$n/$f"; done
done
files="$files $(ls logs/r2gate-*.out logs/r2final-*.out logs/gpu-train-r2-21369832-* logs/train-21369832.out logs/eval-21369834.out logs/collect-21369734.out 2>/dev/null | tr '\n' ' ')"
tar czf /scratch/yx3882/r2rerun_records_20261029.tgz --exclude='r2rerun_20261029/snapshot_*' $files
sha256sum /scratch/yx3882/r2rerun_records_20261029.tgz > "$R/archive.sha256"
cat "$R/archive.sha256"
echo "fp-r2final-complete $(date -u +%FT%TZ)"
