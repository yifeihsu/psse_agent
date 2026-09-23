#!/usr/bin/env bash
#SBATCH --job-name=revised-bc0
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
#SBATCH --output=logs/revised-bc0-%j.out
set -euo pipefail
: "${REVISED_EVAL_ROOT:?Export the staged evaluation cell}"
source "$REVISED_EVAL_ROOT/source/research/hpc/revised_eval_20260922/common.env"
trap revised_eval_cleanup EXIT
revised_eval_local_model
[[ -s "$FROZEN_BC0_ADAPTER/adapter_model.safetensors" ]] || {
  printf 'Frozen BC0 adapter is unavailable: %s\n' "$FROZEN_BC0_ADAPTER" >&2
  exit 2
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "$SRC"
"$PY" scripts/evaluate_revised_expert.py --policy bc0 \
  --scenarios "$FROZEN_SCENARIOS" --output-dir "$BC0_EVAL_DIR" \
  --adapter-path "$FROZEN_BC0_ADAPTER" --baseline-eval "$BC0_BASELINE" --resume
