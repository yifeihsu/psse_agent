#!/usr/bin/env bash
#SBATCH --job-name=cc-row-gate
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/compound_cloning_20261106/logs/%x-%j.out
# The row gate over the full views, on a CPU node, before any GPU time: every row must render and
# tokenize, no prompt may be cut, and the result is kept for the training stage's restarts
# (train_runtime_tools.py, "The row gate").  Run again it is a cache hit.
set -euo pipefail
WORK=/scratch/yx3882/compound_cloning_20261106
# shellcheck source=cell.env
source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"
cell_environment
export CLONE_GATE_CACHE=$OUT/clone/row_gate_cache.pkl
mkdir -p "$OUT/clone/training"
cd "$SRC"
echo "[row gate] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
"$PY" "$CELL/smoke_inputs.py" gate --train "$TRAIN_VIEW" --validation "$VALIDATION_VIEW" --full-train "$TRAIN_VIEW" --tools runtime \
  --model-choice "$MODEL_CHOICE" --max-length "$TRAIN_MAX_LENGTH" --output-dir "$OUT/clone/training" | tee "$OUT/clone/row_gate.json"
echo "cc-row-gate-complete"
