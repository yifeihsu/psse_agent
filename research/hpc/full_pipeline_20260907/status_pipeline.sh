#!/usr/bin/env bash
# Queue state, receipts, and progress for the full pipeline.
set -uo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260912
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.6D %.24R"
echo ---
tail -3 "$PIPE/submitted_jobs.txt" 2>/dev/null
echo ---
for receipt in out/prerequisites.json out/d0.done out/suite.done out/bc0.done \
  out/r1/collection.done out/r1/zeroshot_summary.json out/r1/training.done out/r1/round_summary.json \
  out/r2/collection.done out/r2/training.done out/r2/round_summary.json out/pipeline_summary.json; do
  if [[ -s "$PIPE/$receipt" ]]; then echo "receipt present: $receipt"; else echo "receipt pending: $receipt"; fi
done
echo ---
for round in r1 r2; do
  ledger=$PIPE/out/$round/collection/completed_roots.json
  suite=$PIPE/out/suite/${round}_training.json
  if [[ -s "$ledger" ]]; then
    "$PY" - "$ledger" "$suite" "$round" <<'PY'
import json, sys
from pathlib import Path
done = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
roots = done.get("completed_roots") if isinstance(done, dict) else done
total = len(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))) if Path(sys.argv[2]).is_file() else "?"
print(f"{sys.argv[3]} collection progress: {len(roots)} of {total} training roots")
PY
  fi
done
for f in "$PIPE"/logs/*.out; do
  [[ -f "$f" ]] || continue
  echo "== $f (last 4 lines) =="
  tail -4 "$f" | cut -c1-200
done
