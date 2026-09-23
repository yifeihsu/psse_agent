#!/usr/bin/env bash
#SBATCH --job-name=revised-preflight
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/revised-preflight-%j.out
set -euo pipefail
: "${REVISED_EVAL_ROOT:?Export the staged evaluation cell}"
source "$REVISED_EVAL_ROOT/source/research/hpc/revised_eval_20260922/common.env"
trap revised_eval_cleanup EXIT
revised_eval_local_model
cd "$SRC"
"$PY" "$TEMPLATE_DIR/preflight.py" --source "$SRC" --runtime-python "$PY" \
  --test-python "$TEST_PY" --receipt "$OUT/preflight.json"
