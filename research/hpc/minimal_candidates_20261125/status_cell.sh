#!/usr/bin/env bash
# Where the candidate read-out stands.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/minimal_candidates_20261125/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_candidates_20261125
OUT=$WORK/out
LOGS=$WORK/logs
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source $(git -C "$WORK/source" rev-parse --short=7 HEAD 2>/dev/null || echo none)"
echo "=== inputs"
for f in "$WORK"/data/candidate_inputs_dev_*.jsonl; do
  [ -f "$f" ] && echo "    $(basename "$f"): $(wc -l < "$f") states, sha256 $(sha256sum "$f" | cut -c1-16)"
done
echo "=== submitted"
tail -n 5 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E " mcs-" | sed 's/^/    /'
echo "=== jobs (sacct times are US Eastern)"
sacct -X -n -u "$USER" -S now-3days --format=JobID%12,JobName%14,State%12,Start,End,Elapsed,NodeList%8 2>/dev/null | grep -E " mcs-" | sed 's/^/    /'
for arm in base steps17; do
  written=$(ls "$OUT/$arm/states" 2>/dev/null | wc -l)
  summary="not yet"
  [ -s "$OUT/$arm/summary.json" ] && summary=present
  echo "=== arm $arm: $written states written; summary $summary"
  for f in $(ls -t "$LOGS"/mcs-"$arm"-*.out 2>/dev/null | head -n 1) $(ls -t "$LOGS"/candidates-"$arm"-*-shard*.out 2>/dev/null | head -n 2); do
    echo "--- $(basename "$f"): $(wc -l < "$f") lines, Traceback $(grep -c Traceback "$f")"
    grep -vE 'Warning|warnings.warn|^\s*$' "$f" | tail -n 3 | cut -c1-400 | sed 's/^/    /'
  done
  if [ -s "$OUT/$arm/summary.json" ]; then
    python3 - "$OUT/$arm/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
for name, cell in summary["by_split"].items():
    print(f"    {name}: {cell['states']} states, {cell['states_with_an_error_left']} with an error left, "
          f"{cell['every_object_still_wrong_can_be_named']} where every object still wrong can be named; "
          f"search stopped at its limit in {cell['search_stopped_at_its_limit']}")
    for size, found in cell["coverage"].items():
        one = found["an object still wrong is in the set"]
        print(f"      set of {size}: an object still wrong in {one['yes']} of {one['yes'] + one['no'] + one['undecided']}"
              f" ({one['undecided']} undecided)")
PY
  fi
done
echo status_done
