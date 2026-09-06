#!/usr/bin/env bash
#SBATCH --job-name=fp-train
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
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260907/logs/train-%j.out
# Full pipeline, DAgger training for PIPELINE_ROUND (r1 or r2): continue the
# round's student for one pass over the 1:1 mixture (micro-batch 1 x 4
# accumulation, lr 3e-5) and keep the final adapter as the candidate.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260907
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
ROUND=${PIPELINE_ROUND:?PIPELINE_ROUND (r1 or r2) must be exported by the submitter}
round_bindings "$ROUND"
echo "[train:$ROUND] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$ROUND_DIR/collection.done" ]] || { echo "collection for $ROUND has not completed" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-train-$ROUND-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-train-$ROUND-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap stage_cleanup EXIT

cd "$SRC"
MIXTURE=$COLLECTION_DIR/round1.train.jsonl
VALIDATION_SUBSET=$BC0_DIR/validation.subset.jsonl
read -r TRAIN_MAX_STEPS MIXTURE_ROWS < <(
"$PY" - "$MIXTURE" "$ROUND_DIR/collection.done" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

mixture = Path(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if hashlib.sha256(mixture.read_bytes()).hexdigest() != report["mixture_sha256"]:
    raise SystemExit("the mixture differs from the collection receipt")
rows = sum(1 for line in mixture.read_text(encoding="utf-8").splitlines() if line.strip())
if rows <= 0 or rows != int(report["mixture_rows"]):
    raise SystemExit("mixture row count does not match its receipt")
print(math.ceil(rows / 4), rows)
PY
)
echo "[train:$ROUND] mixture rows $MIXTURE_ROWS -> $TRAIN_MAX_STEPS optimizer steps"

if [[ ! -s "$CANDIDATE/adapter_model.safetensors" || ! -s "$TRAIN_DIR/research_run.json" ]]; then
  resume_args=()
  if [[ -d "$TRAIN_DIR" ]]; then
    checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint \
      --output-dir "$TRAIN_DIR" --expected-base-model "$MODEL_ID" || true)
    if [[ -n "$checkpoint_path" ]]; then
      echo "[train:$ROUND] resuming from newest complete checkpoint: $checkpoint_path"
      resume_args+=(--resume-from-checkpoint "$checkpoint_path")
    fi
  fi
  "$PY" -m psse_env.sft research-train \
    --model-choice "$MODEL_CHOICE" \
    --train "$MIXTURE" \
    --validation "$VALIDATION_SUBSET" \
    --initial-adapter "$STUDENT" \
    --output-dir "$TRAIN_DIR" \
    --max-length "$TRAIN_MAX_LENGTH" \
    --strict-prompt-length \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate "$ROUND_LEARNING_RATE" \
    --epochs 1 \
    --max-steps "$TRAIN_MAX_STEPS" \
    --logging-steps 1 \
    --save-steps "$ROUND_SAVE_EVAL_STEPS" \
    --eval-steps "$ROUND_SAVE_EVAL_STEPS" \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --smoke-steps 1 \
    --reload-canaries 1 \
    --seed "$SEED_ROUND" \
    --run-name "fp-$ROUND-$(date -u +%Y%m%d)" \
    --report-to none \
    "${resume_args[@]}"
fi
"$PY" - "$TRAIN_DIR" "$ROUND_DIR/training.done" "$ROUND" "$MIXTURE_ROWS" "$TRAIN_MAX_STEPS" "$STUDENT" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path
from psse_env.dagger.release_factories import checkpoint_tree_sha256

train_dir = Path(sys.argv[1])
run = json.loads((train_dir / "research_run.json").read_text(encoding="utf-8"))
payload = {
    "contract": "research_full_pipeline_training_done_v1",
    "round": sys.argv[3],
    "student_path": sys.argv[6],
    "student_tree_sha256": checkpoint_tree_sha256(sys.argv[6]),
    "candidate_path": str(train_dir / "lora"),
    "candidate_tree_sha256": checkpoint_tree_sha256(train_dir / "lora"),
    "mixture_rows": int(sys.argv[4]),
    "max_steps": int(sys.argv[5]),
    "research_run_passed": run.get("passed"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({k: payload[k] for k in ("round", "mixture_rows", "max_steps", "research_run_passed")}))
PY
echo "fp-train-$ROUND-complete"
