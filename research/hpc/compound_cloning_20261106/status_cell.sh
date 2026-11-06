#!/usr/bin/env bash
# Queue, receipts and progress of the cloning cell:
#   wsl -- ssh torch bash -s < research/hpc/compound_cloning_20261106/status_cell.sh
set -uo pipefail
WORK=/scratch/yx3882/compound_cloning_20261106
source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"
export PATH=/opt/slurm/bin:$PATH
squeue -u "$USER" -o "%.12i %.28j %.8T %.10M %.24R"
echo ---
tail -5 "$WORK/submitted_jobs.txt" 2>/dev/null
echo ---
for receipt in smoke/gate_timing.json smoke/training.done clone/training.done; do
  if [[ -s "$OUT/$receipt" ]]; then echo "present: $receipt"; else echo "pending: $receipt"; fi
done
if [[ -d "$OUT/clone/training" ]]; then
  newest=$(find "$OUT/clone/training" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
  rows=$(grep -c . "$TRAIN_VIEW" 2>/dev/null)
  echo "newest checkpoint of the pass: step ${newest:-none} of $(( (${rows:-0} + 3) / 4 ))"
fi
echo ---
for dir in "$EVAL_OUT"/*/; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  if [[ -s "$dir/outcomes.json" ]]; then
    echo "$name: complete"
  else
    done_episodes=$(find "$dir/shards" -name 'episode-*.pkl' 2>/dev/null | wc -l)
    planned=$(grep -c '"standard_success:' "$dir"/shards/*/plan.json 2>/dev/null | cut -d: -f2 | head -1)
    echo "$name: $done_episodes of ${planned:-?} episodes"
  fi
done
echo ---
for log in "$LOGS"/cc-*.out; do
  [[ -f "$log" ]] || continue
  echo "== $log (last 2 lines) =="
  tail -2 "$log" | cut -c1-200
done
