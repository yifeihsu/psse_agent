#!/usr/bin/env bash
#SBATCH --job-name=fp-ctrain
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
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/ctrain-%j.out
# The matched cloning control (conventional-only cell, 2026-10-23): BC0
# continued on R1's mixture with every learner-visited row replaced by a fresh
# expert row of the same family (build_control_mixture.py), with R1's recipe:
# BC0 as the initial adapter, one pass of ceil(rows / 4) optimizer steps at
# ROUND_LEARNING_RATE, micro-batch 1 x 4 accumulation, SEED_ROUND, the same
# validation rows; the final adapter is the control.  Resumes from the newest
# complete checkpoint when requeued.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
EXTRA=$OUT/extra
CONTROL_DIR=$EXTRA/control
TRAIN_DIR=$CONTROL_DIR/training
MIXTURE=$CONTROL_DIR/mixture.jsonl
# build_control_mixture.py writes the report beside the mixture as mixture.report.json.
MIXTURE_REPORT=$CONTROL_DIR/mixture.report.json
VALIDATION_SUBSET=$BC0_DIR/validation.subset.jsonl
echo "[ctrain] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$MIXTURE" && -s "$MIXTURE_REPORT" ]] || { echo "the control mixture is missing (stage_extra_cpu.sbatch)" >&2; exit 2; }
[[ -s "$BC0_DIR/lora/adapter_model.safetensors" && -s "$VALIDATION_SUBSET" ]] || { echo "BC0 or its validation rows missing" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-ctrain-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-ctrain-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap stage_cleanup EXIT

cd "$SRC"
read -r TRAIN_MAX_STEPS MIXTURE_ROWS < <(
"$PY" - "$MIXTURE" "$MIXTURE_REPORT" "$R1_DIR/training.done" <<'PY'
import json
import math
import sys
from pathlib import Path

rows = sum(1 for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip())
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
r1 = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if rows != int(report["rows"]) or rows != int(r1["mixture_rows"]):
    raise SystemExit(f"control mixture rows {rows} differ from its report or from R1's {r1['mixture_rows']}")
print(math.ceil(rows / 4), rows)
PY
)
echo "[ctrain] mixture rows $MIXTURE_ROWS -> $TRAIN_MAX_STEPS optimizer steps"
if [[ ! -s "$TRAIN_DIR/lora/adapter_model.safetensors" || ! -s "$TRAIN_DIR/research_run.json" ]]; then
  resume_args=()
  if [[ -d "$TRAIN_DIR" ]]; then
    checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint \
      --output-dir "$TRAIN_DIR" --expected-base-model "$MODEL_ID" || true)
    if [[ -n "$checkpoint_path" ]]; then
      echo "[ctrain] resuming from newest complete checkpoint: $checkpoint_path"
      resume_args+=(--resume-from-checkpoint "$checkpoint_path")
    fi
  fi
  "$PY" -m psse_env.sft research-train \
    --model-choice "$MODEL_CHOICE" \
    --train "$MIXTURE" \
    --validation "$VALIDATION_SUBSET" \
    --initial-adapter "$BC0_DIR/lora" \
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
    --run-name "fp-control-$(date -u +%Y%m%d)" \
    --report-to none \
    "${resume_args[@]}"
fi
"$PY" - "$TRAIN_DIR" "$CONTROL_DIR/training.done" "$MIXTURE_ROWS" "$TRAIN_MAX_STEPS" "$BC0_DIR/lora" "$MIXTURE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from psse_env.dagger.release_factories import checkpoint_tree_sha256

train_dir = Path(sys.argv[1])
run = json.loads((train_dir / "research_run.json").read_text(encoding="utf-8"))
payload = {
    "contract": "conventional_only_control_training_done_v1",
    "initial_adapter": sys.argv[5],
    "initial_adapter_tree_sha256": checkpoint_tree_sha256(sys.argv[5]),
    "candidate_path": str(train_dir / "lora"),
    "candidate_tree_sha256": checkpoint_tree_sha256(train_dir / "lora"),
    "mixture": sys.argv[6],
    "mixture_sha256": hashlib.sha256(Path(sys.argv[6]).read_bytes()).hexdigest(),
    "mixture_rows": int(sys.argv[3]),
    "max_steps": int(sys.argv[4]),
    "research_run_passed": run.get("passed"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ("mixture_rows", "max_steps", "research_run_passed")}))
PY
echo fp-ctrain-complete
