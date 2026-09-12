#!/usr/bin/env bash
#SBATCH --job-name=fp-bc0
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
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/bc0-%j.out
# Full pipeline, stage 1: BC0 from the base model on the expert aggregate.
#
# The full-BC0 recipe: 4-bit base plus rank-16 LoRA, micro-batch 1 with four
# accumulation steps (the fused batch-4 layout produced a NaN 12B adapter on
# 2026-09-01), lr 1e-4, one pass, the best validation-loss checkpoint kept
# as the adapter.  Validation is a fixed family-stratified subset of the
# aggregate's root-disjoint validation split so each evaluation pass stays
# short; the same subset monitors every later round.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
echo "[bc0] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$OUT/d0.done" && -s "$OUT/suite.done" ]] || { echo "stage 0 has not completed" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-bc0-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-bc0-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap stage_cleanup EXIT

reuse_previous_stage bc0 bc0.done
if [[ -s "$OUT/bc0.done" && -s "$BC0_DIR/lora/adapter_model.safetensors" ]]; then
  echo "[bc0] adapter present with receipt; nothing to train"
  echo fp-bc0-complete
  exit 0
fi
mkdir -p "$BC0_DIR"
VALIDATION_SUBSET=$BC0_DIR/validation.subset.jsonl
if [[ ! -s "$VALIDATION_SUBSET" ]]; then
  "$PY" - "$D0_DIR/aggregate.validation.jsonl" "$VALIDATION_SUBSET" "$VALIDATION_ROWS" "$SEED_D0" <<'PY'
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
target = int(sys.argv[3])
rng = random.Random(int(sys.argv[4]))
by_family = defaultdict(list)
for row in rows:
    metadata = row.get("metadata") or {}
    by_family[str(metadata.get("scenario_family") or row.get("scenario_family") or "")].append(row)
for bucket in by_family.values():
    rng.shuffle(bucket)
selected = []
while len(selected) < min(target, len(rows)):
    progressed = False
    for family in sorted(by_family):
        if by_family[family] and len(selected) < target:
            selected.append(by_family[family].pop())
            progressed = True
    if not progressed:
        break
selected.sort(key=lambda row: str(row.get("example_id") or ""))
Path(sys.argv[2]).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
print(f"validation subset: {len(selected)} of {len(rows)} rows")
PY
fi

cd "$SRC"
if [[ ! -s "$BC0_DIR/lora/adapter_model.safetensors" || ! -s "$BC0_DIR/research_run.json" ]]; then
  resume_args=()
  checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint \
    --output-dir "$BC0_DIR" --expected-base-model "$MODEL_ID" || true)
  if [[ -n "$checkpoint_path" ]]; then
    echo "[bc0] resuming from newest complete checkpoint: $checkpoint_path"
    resume_args+=(--resume-from-checkpoint "$checkpoint_path")
  fi
  "$PY" -m psse_env.sft research-train \
    --model-choice "$MODEL_CHOICE" \
    --train "$D0_DIR/aggregate.train_view.jsonl" \
    --validation "$VALIDATION_SUBSET" \
    --output-dir "$BC0_DIR" \
    --max-length "$TRAIN_MAX_LENGTH" \
    --strict-prompt-length \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate "$BC0_LEARNING_RATE" \
    --epochs "$BC0_EPOCHS" \
    --logging-steps 1 \
    --save-steps "$BC0_SAVE_EVAL_STEPS" \
    --eval-steps "$BC0_SAVE_EVAL_STEPS" \
    --select-best-eval-loss \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --smoke-steps 1 \
    --reload-canaries 1 \
    --seed "$SEED_D0" \
    --run-name "fp-bc0-$(date -u +%Y%m%d)" \
    --report-to none \
    "${resume_args[@]}"
fi
"$PY" - "$BC0_DIR" "$OUT/bc0.done" "$D0_DIR/aggregate.train_view.jsonl" "$VALIDATION_SUBSET" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from psse_env.dagger.release_factories import checkpoint_tree_sha256

bc0 = Path(sys.argv[1])
run = json.loads((bc0 / "research_run.json").read_text(encoding="utf-8"))
state_path = bc0 / "trainer_state.json"
state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
evals = [entry for entry in state.get("log_history", []) if "eval_loss" in entry]
best = min(evals, key=lambda entry: entry["eval_loss"]) if evals else None
def count(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
payload = {
    "contract": "research_full_pipeline_bc0_done_v1",
    "adapter_path": str(bc0 / "lora"),
    "adapter_tree_sha256": checkpoint_tree_sha256(bc0 / "lora"),
    "train_rows": count(sys.argv[3]),
    "validation_rows": count(sys.argv[4]),
    "global_step": state.get("global_step"),
    "best_model_checkpoint": state.get("best_model_checkpoint"),
    "best_eval": best,
    "research_run_passed": run.get("passed"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({k: payload[k] for k in ("train_rows", "validation_rows", "global_step", "best_eval")}))
PY
echo fp-bc0-complete
