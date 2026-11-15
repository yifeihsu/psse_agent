#!/usr/bin/env bash
#SBATCH --job-name=msf-row-gate
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/minimal_sft_20261115/logs/%x-%j.out
# The row gate over the full views, on a CPU node, before any GPU time: every row must render with its own six
# tools through the model's chat template and tokenize, its tool call must read back from the rendering, and no
# prompt may be cut.  The result is kept for the training stage's restarts (research/minimal_env/train_sft.py, "The
# row gate").  These rows have never been through the tokenizer: the checks made when they were written cover the
# schemas only.  Run again it is a cache hit.  The gate does not depend on where the pass starts, so one result
# serves both starts.
set -euo pipefail
WORK=/scratch/yx3882/minimal_sft_20261115
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_sft_20261115/cell.env"
cell_environment
export MINIMAL_GATE_CACHE=$OUT/row_gate_cache.pkl
mkdir -p "$OUT/row_gate"
cd "$SRC"
echo "[row gate] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} source $(git -C "$SRC" rev-parse HEAD) $(date -u +%FT%TZ)"
[[ -s "$TRAIN_VIEW" && -s "$VALIDATION_VIEW" ]] || { echo "rows missing under $DATA/rows: deploy them first" >&2; exit 2; }
"$PY" "$TOOLS/row_gate.py" gate --train "$TRAIN_VIEW" --validation "$VALIDATION_VIEW" \
  --model-choice "$MODEL_CHOICE" --max-length "$TRAIN_MAX_LENGTH" --output-dir "$OUT/row_gate" | tee "$OUT/row_gate.json"
echo "msf-row-gate-complete"
