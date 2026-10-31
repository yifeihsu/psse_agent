#!/usr/bin/env bash
#SBATCH --job-name=fp-basefinal
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20261022_conventional/logs/basefinal-%j.out
# The prompted, untuned base model (2026-10-31, note section 11): after its
# seven test evaluations, compare it with the expert, BC0, R1 and Table I's R2
# (base_tables.py), compare out/ with the listing taken before the deploy, list
# the jobs (JOBS, exported by the submitter) and archive the small records.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
P=/scratch/yx3882/research_full_pipeline_20261022_conventional
W=$P/base_prompted_20261031
L=base_prompted
: "${JOBS:?JOBS must list the evaluation job ids}"
source "$P/pipeline.env"
pipeline_environment >/dev/null
echo "[basefinal] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
cd "$P/source"
"$PY" research/hpc/full_pipeline_20260907/base_tables.py --cell-dir "$P" --label "$L" --traces 16 \
  --output "$W/base_tables.json" > "$W/base_tables.console.txt" 2>&1
for d in d0 bc0 r1 r2 r2_bc0init_20261030 r2_pool_20261023 extra; do
  (cd "$P/out" && find "$d" -printf '%p\t%y\t%s\t%T@\n' | sort) > "$W/snapshot_final_$d.tsv"
done
"$PY" - "$W" "$L" > "$W/snapshot_check.txt" <<'PY'
import re
import sys
from pathlib import Path

W, L = Path(sys.argv[1]), sys.argv[2]
NEW = [f"{n}_{L}" for n in ("test_40", "test2_40", "test_budget12", "test_budget16", "test_budget20",
                            "test2_budget12", "test2_budget20", "validation_dryrun")]
expected = re.compile(r"^extra/(evaluations/(%s)(/|\.console\.json$|$)|specs/(%s)\.env$|evaluations$|specs$|extra$)"
                      % ("|".join(NEW), "|".join(NEW)))
for d in ("d0", "bc0", "r1", "r2", "r2_bc0init_20261030", "r2_pool_20261023", "extra"):
    before = {line.split("\t")[0]: line for line in (W / f"snapshot_preset_{d}.tsv").read_text().splitlines()}
    after = {line.split("\t")[0]: line for line in (W / f"snapshot_final_{d}.tsv").read_text().splitlines()}
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    bad = [p for p in removed + added + changed if not (d == "extra" and expected.match(p))]
    print(f"{d}: removed {len(removed)}, added {len(added)}, changed {len(changed)}; outside the new records: {len(bad)}")
    for p in bad[:20]:
        print("   unexpected", p)
PY
{ sacct -D -j "$JOBS" -X -o JobID%10,JobName%12,State%12,Elapsed%10,NodeList%10,Start%20,End%20
  NODES=$(sacct -D -j "$JOBS" -X -n -o NodeList%20 | tr -s ' \n' ',' | sed 's/^,//;s/,$//')
  sinfo -N -n "$NODES" -o "%N %G" | sort -u; } > "$W/jobs.txt" 2>&1 || true
cd "$P"
files="base_prompted_20261031"
for n in test_40 test2_40 test_budget12 test_budget16 test_budget20 test2_budget12 test2_budget20 validation_dryrun; do
  n=${n}_$L
  for f in "out/extra/specs/$n.env" "out/extra/evaluations/$n.console.json"; do [[ -e "$f" ]] && files="$files $f"; done
  for f in summary.json policies.json episodes.jsonl comparison.json; do
    [[ -e "out/extra/evaluations/$n/$f" ]] && files="$files out/extra/evaluations/$n/$f"
  done
done
for id in ${JOBS//,/ }; do
  files="$files $(ls logs/xeval-$id.out logs/gpu-xeval-*-$id-* logs/gpu-util-xeval-*-$id.csv 2>/dev/null | tr '\n' ' ')"
done
files="$files $(ls logs/basefinal-*.out 2>/dev/null | tr '\n' ' ')"
# shellcheck disable=SC2086
tar czf /scratch/yx3882/base_prompted_records_20261031.tgz --exclude='base_prompted_20261031/snapshot_*' \
  --exclude='base_prompted_20261031/__pycache__' $files
sha256sum /scratch/yx3882/base_prompted_records_20261031.tgz > "$W/archive.sha256"
cat "$W/archive.sha256"
echo "fp-basefinal-complete $(date -u +%FT%TZ)"
