#!/usr/bin/env bash
#SBATCH --job-name=fp-r2wfinal
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20261022_conventional/logs/r2wfinal-%j.out
# Round 2 from R1 (2026-10-30, note section 10): after r2e and the R1-start R2's
# eight evaluations, rebuild the pooled, budget and follow-up tables with it,
# compare it with the BC0-start R2 (set aside under _bc0init_20261030), the
# recorded R2 (_pool_20261023), the expert, BC0 and R1 (r2rerun_tables.py),
# count the breaker-root context order (context_order_audit.py), check that the
# round-2 data r2e re-exported equal the copied ones, compare out/d0, out/bc0,
# out/r1 and out/extra with the listing taken before the set-aside, and archive
# the small records.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
P=/scratch/yx3882/research_full_pipeline_20261022_conventional
W=$P/r2warm_20261030
S=_bc0init_20261030
source "$P/pipeline.env"
pipeline_environment >/dev/null
echo "[r2wfinal] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
cd "$P/source"
T=research/hpc/full_pipeline_20260907
"$PY" $T/pooled_tables.py --cell-dir "$P" --output "$P/out/extra/pooled_tables.json" > "$W/pooled_tables.console.txt" 2>&1
"$PY" $T/budget_tables.py --cell-dir "$P" --output "$P/out/extra/budget_tables.json" > "$W/budget_tables.console.txt" 2>&1
"$PY" $T/followup_tables.py --cell-dir "$P" --r2-evaluation r2 --control-evaluation r2_control_pool_20261023 \
  --output "$P/out/extra/followup_tables.json" > "$W/followup_tables.console.txt" 2>&1
cd "$W"
"$PY" r2rerun_tables.py --cell-dir "$P" --bc0init-suffix "$S" --traces 16 --output "$W/r2warm_tables.json" \
  > "$W/r2warm_tables.console.txt" 2>&1
"$PY" context_order_audit.py --cell-dir "$P" --output "$W/context_order_audit.json" > "$W/context_order_audit.md"
# r2e re-runs the round's collection script before it evaluates, which re-exports
# the round-2 rows and the mixture from the ledger: they must equal the copies.
(cd "$P/out/r2/collection" && find . -path ./evaluation -prune -o -type f ! -name research_run_report.json -print0 \
  | sort -z | xargs -0 sha256sum) > "$W/collection_after_r2e.sha256"
if cmp -s "$W/collection_source.sha256" "$W/collection_after_r2e.sha256"; then
  echo "round-2 collection after r2e: identical to the BC0-start pass's ($(wc -l < "$W/collection_after_r2e.sha256") files)"
else
  echo "round-2 collection after r2e: DIFFERS from the BC0-start pass's"
  diff "$W/collection_source.sha256" "$W/collection_after_r2e.sha256" | head -20
fi > "$W/collection_check.txt"
for d in d0 bc0 r1 extra; do
  (cd "$P/out" && find "$d" -printf '%p\t%y\t%s\t%T@\n' | sort) > "$W/snapshot_final_$d.tsv"
done
"$PY" - "$W" "$S" > "$W/snapshot_check.txt" <<'PY'
import re
import sys
from pathlib import Path

W, S = Path(sys.argv[1]), sys.argv[2]
NEW = ("test2_40_r2", "test_budget12_r2", "test_budget16_r2", "test_budget20_r2", "test2_budget12_r2",
       "test2_budget20_r2", "validation_r2", "hard_r2")
expected = re.compile(r"^extra/(evaluations/(%s)(/|\.console\.json$|$)|specs/(%s)\.env$|"
                      r"(evaluations|specs)/[^/]*%s|"
                      r"(pooled_tables|pooled_tables_episodes|budget_tables|followup_tables)(%s)?\.(json|md|jsonl)$|"
                      r"evaluations$|specs$|extra$)" % ("|".join(NEW), "|".join(NEW), S, S))
for d in ("d0", "bc0", "r1", "extra"):
    before = {line.split("\t")[0]: line for line in (W / f"snapshot_preset_{d}.tsv").read_text().splitlines()}
    after = {line.split("\t")[0]: line for line in (W / f"snapshot_final_{d}.tsv").read_text().splitlines()}
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    if d == "extra":
        bad = [p for p in removed + added + changed if not expected.match(p)]
        print(f"{d}: removed {len(removed)}, added {len(added)}, changed {len(changed)}; outside the set-aside and new records: {len(bad)}")
        for p in bad[:20]:
            print("   unexpected", p)
    else:
        print(f"{d}: removed {len(removed)}, added {len(added)}, changed {len(changed)}" + ("" if removed or added or changed else " (identical)"))
        for p in (removed + added + changed)[:20]:
            print("   ", p)
PY
JOBS=21447607,21447703,21447704,21447836,21447838,21447839,21447841,21447842,21447843,21447844,21447845
{ sacct -D -j "$JOBS" -X -o JobID%10,JobName%12,State%12,Elapsed%10,NodeList%10,Start%20,End%20
  NODES=$(sacct -D -j "$JOBS" -X -n -o NodeList%20 | tr -s ' \n' ',' | sed 's/^,//;s/,$//')
  sinfo -N -n "$NODES" -o "%N %G" | sort -u; } > "$W/jobs.txt" 2>&1 || true
cd "$P"
files="r2warm_20261030 out/pipeline_summary.json out/extra/pooled_tables.json out/extra/pooled_tables.md out/extra/pooled_tables_episodes.jsonl out/extra/budget_tables.json out/extra/budget_tables.md out/extra/followup_tables.json out/extra/followup_tables.md"
for f in collection.done training.done round_summary.json evaluation_console.json collection_console.json collection/mixture_report.json collection/collection_metrics.json collection/research_run_report.json collection/evaluation/comparison.json training/research_run.json training/trainer_state.json; do
  [[ -e "out/r2/$f" ]] && files="$files out/r2/$f"
done
for n in test2_40_r2 test_budget12_r2 test_budget16_r2 test_budget20_r2 test2_budget12_r2 test2_budget20_r2 validation_r2 hard_r2; do
  files="$files out/extra/specs/$n.env out/extra/evaluations/$n.console.json"
  for f in summary.json policies.json episodes.jsonl comparison.json; do files="$files out/extra/evaluations/$n/$f"; done
done
files="$files $(ls logs/r2wgate-*.out logs/r2wfinal-*.out logs/gpu-train-r2-21447607-* logs/train-21447607.out logs/eval-21447704.out logs/gpu-eval-r2-21447704-* logs/xeval-2144783[6-9].out logs/xeval-2144784[1-5].out logs/gpu-xeval-*-2144783[6-9]-* logs/gpu-xeval-*-2144784[1-5]-* 2>/dev/null | tr '\n' ' ')"
# shellcheck disable=SC2086
tar czf /scratch/yx3882/r2warm_records_20261030.tgz --exclude='r2warm_20261030/snapshot_*' --exclude='r2warm_20261030/__pycache__' $files
sha256sum /scratch/yx3882/r2warm_records_20261030.tgz > "$W/archive.sha256"
cat "$W/archive.sha256"
echo "fp-r2wfinal-complete $(date -u +%FT%TZ)"
