#!/usr/bin/env bash
#SBATCH --job-name=orc-row-gate
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/clone_open_rules_20261112/logs/%x-%j.out
# The row gate over the full views, on a CPU node, before any GPU time: every row must render and tokenize with
# its run-time tool list, no prompt may be cut, and the result is kept for the training stage's restarts (the
# bounded cell's train_runtime_tools.py, "The row gate").  These rows have never been through the tokenizer: the
# local checks are the gate's message checks only.  Run again it is a cache hit.
set -euo pipefail
WORK=/scratch/yx3882/clone_open_rules_20261112
# shellcheck source=cell.env
source "$WORK/source/research/hpc/clone_open_rules_20261112/cell.env"
cell_environment
export CLONE_GATE_CACHE=$OUT/clone/row_gate_cache.pkl
mkdir -p "$OUT/clone/training"
cd "$SRC"
echo "[row gate] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} source $(git -C "$SRC" rev-parse HEAD) $(date -u +%FT%TZ)"
[[ -s "$TRAIN_VIEW" && -s "$VALIDATION_VIEW" ]] || { echo "rows missing under $DATA/rows: deploy them first" >&2; exit 2; }
"$PY" "$BOUNDED_CELL/smoke_inputs.py" gate --train "$TRAIN_VIEW" --validation "$VALIDATION_VIEW" --full-train "$TRAIN_VIEW" --tools runtime \
  --model-choice "$MODEL_CHOICE" --max-length "$TRAIN_MAX_LENGTH" --output-dir "$OUT/clone/training" | tee "$OUT/clone/row_gate.json"
echo "orc-row-gate-complete"
