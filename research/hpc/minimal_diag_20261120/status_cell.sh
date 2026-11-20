#!/usr/bin/env bash
# Where the stopping-state diagnostic stands.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/minimal_diag_20261120/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_diag_20261120
OUT=$WORK/out
LOGS=$WORK/logs
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source $(git -C "$WORK/source" rev-parse --short=7 HEAD 2>/dev/null || echo none)"
echo "=== inputs"
for f in "$WORK"/data/scoring_inputs_*.jsonl; do
  [ -f "$f" ] && echo "    $(basename "$f"): $(wc -l < "$f") states, sha256 $(sha256sum "$f" | cut -c1-16)"
done
echo "=== submitted"
tail -n 5 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E "mdg-" | sed 's/^/    /'
echo "=== jobs (sacct times are US Eastern)"
sacct -X -n -u "$USER" --name=mdg-score -S now-3days --format=JobID%12,State%12,Start,End,Elapsed,NodeList%8 2>/dev/null | sed 's/^/    /'
written=$(ls "$OUT/scores/states" 2>/dev/null | wc -l)
summary="not yet"
[ -s "$OUT/scores/summary.json" ] && summary=present
echo "=== scores: $written states written; summary $summary"
for f in $(ls -t "$LOGS"/mdg-score-*.out 2>/dev/null | head -n 1) $(ls -t "$LOGS"/score-*-shard*.out 2>/dev/null | head -n 2); do
  echo "--- $(basename "$f"): $(wc -l < "$f") lines, Traceback $(grep -c Traceback "$f")"
  grep -vE 'Warning|warnings.warn|^\s*$' "$f" | tail -n 3 | cut -c1-400 | sed 's/^/    /'
done
if [ -s "$OUT/scores/summary.json" ]; then
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["by_split"], indent=1, sort_keys=True))' "$OUT/scores/summary.json"
fi
echo status_done
