#!/usr/bin/env bash
#SBATCH --job-name=diagtrain
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_diag_round_20260903/logs/train-%j.out
# Diagnostic-family research round (2026-09-03), stage 2.
#
# Rebuild the 1:1 D0/D1 mixture from a D0 pool without the stale voltage-only
# HIF families (they were taught to end every HIF episode in an operator
# handoff, which the diagnostic D1 rows contradict), then continue BC0
# checkpoint-192 on it for one pass with the proven 12B layout (micro-batch 1,
# four-step accumulation; the fused batch-4 layout produced a NaN adapter on
# 2026-09-01).  The final adapter is the candidate; no best-loss selection,
# because the only loss set is the classical-family trace validation and it
# would ignore the diagnostic rows.
set -euo pipefail
ROUND=/scratch/yx3882/research_diag_round_20260903
# shellcheck source=round.env
source "$ROUND/round.env"
round_environment
echo "[train] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$OUT/collection.done" ]] || { echo "collection stage has not completed" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-train-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true

cd "$SRC"
ROUND1_TRAIN=$COLLECTION_DIR/round1.train.jsonl
"$PY" - "$ROUND1_TRAIN" "$OUT/collection.done" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

mixture = Path(sys.argv[1])
done = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if hashlib.sha256(mixture.read_bytes()).hexdigest() != done["mixture_sha256"]:
    raise SystemExit("round1.train.jsonl changed after the collection stage")
print("collection mixture identity verified")
PY

FILTERED=$COLLECTION_DIR/round1.train.filtered.jsonl
FILTER_REPORT=$COLLECTION_DIR/mixture_filter_report.json
if [[ ! -s "$FILTERED" || ! -s "$FILTER_REPORT" ]]; then
  "$PY" "$ROUND/filter_mixture.py" \
    --d0 "$COLLECTION_DIR/d0.train.current.jsonl" \
    --d1 "$COLLECTION_DIR/d1.chat.jsonl" \
    --stale-families "$STALE_D0_FAMILIES" \
    --expected-dropped "$EXPECTED_STALE_D0_ROWS" \
    --d1-share "$D1_SHARE" \
    --d1-cap "$D1_CAP" \
    --seed "$SEED" \
    --output "$FILTERED" \
    --report "$FILTER_REPORT"
fi
read -r TRAIN_MAX_STEPS MIXTURE_ROWS < <("$PY" - "$FILTERED" "$FILTER_REPORT" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

mixture = Path(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if hashlib.sha256(mixture.read_bytes()).hexdigest() != report["output_sha256"]:
    raise SystemExit("round1.train.filtered.jsonl differs from its filter report")
rows = sum(1 for line in mixture.read_text(encoding="utf-8").splitlines() if line.strip())
if rows <= 0 or rows != int(report["mixture_rows"]):
    raise SystemExit("filtered mixture row count does not match its report")
# One pass at effective batch 4 (micro-batch 1 x 4 accumulation steps).
print(math.ceil(rows / 4), rows)
PY
)
echo "[train] filtered mixture rows $MIXTURE_ROWS -> $TRAIN_MAX_STEPS optimizer steps"

if [[ ! -s "$CANDIDATE/adapter_model.safetensors" || ! -s "$TRAIN_DIR/research_run.json" ]]; then
  resume_args=()
  if [[ -d "$TRAIN_DIR" ]]; then
    checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint \
      --output-dir "$TRAIN_DIR" --expected-base-model "$MODEL_ID" || true)
    if [[ -n "$checkpoint_path" ]]; then
      echo "[train] resuming from newest complete checkpoint: $checkpoint_path"
      resume_args+=(--resume-from-checkpoint "$checkpoint_path")
    fi
  fi
  "$PY" -m psse_env.sft research-train \
    --model-choice "$MODEL_CHOICE" \
    --train "$FILTERED" \
    --validation "$VALIDATION" \
    --initial-adapter "$WARM_START" \
    --output-dir "$TRAIN_DIR" \
    --max-length "$TRAIN_MAX_LENGTH" \
    --strict-prompt-length \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate "$TRAIN_LEARNING_RATE" \
    --epochs 1 \
    --max-steps "$TRAIN_MAX_STEPS" \
    --logging-steps 1 \
    --save-steps "$TRAIN_SAVE_EVAL_STEPS" \
    --eval-steps "$TRAIN_SAVE_EVAL_STEPS" \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --smoke-steps 1 \
    --reload-canaries 1 \
    --seed "$SEED" \
    --run-name "$TRAIN_RUN_NAME" \
    --report-to none \
    "${resume_args[@]}"
fi

"$PY" - "$TRAIN_DIR" "$CANDIDATE" "$OUT/training.done" "$TRAIN_MAX_STEPS" "$MIXTURE_ROWS" "$FILTERED" "$FILTER_REPORT" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from psse_env.dagger.release_factories import checkpoint_tree_sha256

train_dir, candidate, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
filtered, filter_report_path = Path(sys.argv[6]), Path(sys.argv[7])
report = json.loads((train_dir / "research_run.json").read_text(encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit("research-train report did not pass")
for name in ("adapter_config.json", "adapter_model.safetensors"):
    if not (candidate / name).is_file():
        raise SystemExit(f"candidate adapter is missing {name}")
filter_report = json.loads(filter_report_path.read_text(encoding="utf-8"))
payload = {
    "contract": "research_diagnostic_round_training_done_v2",
    "candidate_path": str(candidate.resolve()),
    "candidate_tree_sha256": checkpoint_tree_sha256(str(candidate)),
    "max_steps": int(sys.argv[4]),
    "mixture_rows": int(sys.argv[5]),
    "train_path": str(filtered.resolve()),
    "train_sha256": hashlib.sha256(filtered.read_bytes()).hexdigest(),
    "mixture_filter": {
        key: filter_report[key]
        for key in ("contract", "stale_families", "dropped_by_family", "dropped_total", "d0_rows_before", "d0_rows_after", "mixture_sources")
    },
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo diag-train-complete
