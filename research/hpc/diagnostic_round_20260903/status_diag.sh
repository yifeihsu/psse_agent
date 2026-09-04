#!/usr/bin/env bash
# Queue state, log tails, and stage receipts for the diagnostic round.
set -uo pipefail
export PATH=/opt/slurm/bin:$PATH
ROUND=/scratch/yx3882/research_diag_round_20260903
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.6D %.24R"
echo ---
cat "$ROUND/submitted_jobs.txt" 2>/dev/null
echo ---
for f in "$ROUND"/logs/collect-*.out "$ROUND"/logs/train-*.out "$ROUND"/logs/eval-*.out; do
  [[ -f "$f" ]] || continue
  echo "== $f (last 6 lines) =="
  tail -6 "$f"
done
echo ---
for receipt in prerequisites.json collection.done training.done round_summary.json; do
  if [[ -s "$ROUND/out/$receipt" ]]; then echo "receipt present: $receipt"; else echo "receipt pending: $receipt"; fi
done
if [[ -s "$ROUND/out/collection/completed_roots.json" ]]; then
  /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python - "$ROUND/out/collection" <<'PY'
import json, sys
from pathlib import Path
collection = Path(sys.argv[1])
done = json.loads((collection / "completed_roots.json").read_text(encoding="utf-8"))
roots = done.get("completed_roots") if isinstance(done, dict) else done
total = None
if (collection / "training_scenarios.json").is_file():
    total = len(json.loads((collection / "training_scenarios.json").read_text(encoding="utf-8")))
print(f"collection progress: {len(roots)} of {total if total is not None else '?'} training roots")
PY
fi
