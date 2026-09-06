#!/usr/bin/env bash
#SBATCH --job-name=diageval
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
#SBATCH --output=/scratch/yx3882/research_diag_round_20260903/logs/eval-%j.out
# Diagnostic-family research round (2026-09-03), stage 3.
#
# Paired closed-loop evaluation of BC0 checkpoint-192 and the round candidate
# on the exact saved diagnostic development roots, then a per-family summary.
set -euo pipefail
ROUND=/scratch/yx3882/research_diag_round_20260903
# shellcheck source=round.env
source "$ROUND/round.env"
round_environment
echo "[eval] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$OUT/training.done" ]] || { echo "training stage has not completed" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-eval-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
# Sample GPU utilization once a minute for the run's own record (the cluster
# cancels jobs whose average utilization stays low); stopped with the stage.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-eval-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
trap 'kill "$GPU_SAMPLER_PID" 2>/dev/null || true' EXIT

cd "$SRC"
collection_args
if [[ ! -s "$COLLECTION_DIR/evaluation/comparison.json" ]]; then
  "$PY" scripts/run_dagger_research.py "${COLLECTION_ARGS[@]}" \
    --eval-r1-adapter "$CANDIDATE" > "$OUT/evaluation_console.json"
fi
"$PY" - "$COLLECTION_DIR/round1.train.jsonl" "$OUT/collection.done" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

mixture = Path(sys.argv[1])
done = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if hashlib.sha256(mixture.read_bytes()).hexdigest() != done["mixture_sha256"]:
    raise SystemExit("the evaluation rerun changed round1.train.jsonl; the trained mixture is no longer the recorded one")
print("mixture unchanged across the evaluation rerun")
PY
"$PY" "$ROUND/summarize.py" \
  --collection-dir "$COLLECTION_DIR" \
  --training-done "$OUT/training.done" \
  --prerequisites "$OUT/prerequisites.json" \
  --mixture-filter "$COLLECTION_DIR/mixture_filter_report.json" \
  --output "$OUT/round_summary.json"
echo diag-eval-complete
