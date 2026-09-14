#!/usr/bin/env bash
#SBATCH --job-name=fp-zs
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/zeroshot-%j.out
# Full pipeline, zero-shot evaluation: the round-1 student (a frozen adapter
# from an earlier run when FROZEN_STUDENT_ADAPTER is set) and the expert on
# the development suite, without a trained candidate.  Runs after the round-1
# collection, whose ledger the runner reuses, and writes the evaluation
# beside it under ZEROSHOT_EVAL_NAME so a later round-1 training/evaluation
# does not touch it.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
ROUND=r1
round_bindings "$ROUND"
echo "[zeroshot] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$ROUND_DIR/collection.done" ]] || { echo "round-1 collection has not completed" >&2; exit 2; }
[[ -s "$STUDENT/adapter_model.safetensors" ]] || { echo "student adapter missing: $STUDENT" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-zeroshot-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-zeroshot-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
opendss_local_model
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

cd "$SRC"
collection_args "$ROUND"
EVAL_DIR=$COLLECTION_DIR/$ZEROSHOT_EVAL_NAME
if [[ ! -s "$EVAL_DIR/expert_eval.json" || ! -s "$EVAL_DIR/comparison.json" ]]; then
  "$PY" scripts/run_dagger_research.py "${COLLECTION_ARGS[@]}" \
    --eval-student-only --eval-expert --eval-output-name "$ZEROSHOT_EVAL_NAME" \
    > "$ROUND_DIR/${ZEROSHOT_EVAL_NAME}_console.json"
fi
"$PY" - "$COLLECTION_DIR/round1.train.jsonl" "$ROUND_DIR/collection.done" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
mixture = Path(sys.argv[1])
done = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if hashlib.sha256(mixture.read_bytes()).hexdigest() != done["mixture_sha256"]:
    raise SystemExit("the zero-shot rerun changed the mixture; the collected mixture is no longer the recorded one")
print("mixture unchanged across the zero-shot evaluation")
PY
"$PY" "$PIPE/summarize.py" round --round-dir "$ROUND_DIR" --round "$ROUND" \
  --evaluation-name "$ZEROSHOT_EVAL_NAME" --output "$ROUND_DIR/${ZEROSHOT_EVAL_NAME}_summary.json"
echo "fp-zeroshot-complete"
