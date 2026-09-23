#!/usr/bin/env bash
#SBATCH --job-name=revised-expert
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/revised-expert-%j.out
set -euo pipefail
: "${REVISED_EVAL_ROOT:?Export the staged evaluation cell}"
source "$REVISED_EVAL_ROOT/source/research/hpc/revised_eval_20260922/common.env"
trap revised_eval_cleanup EXIT
revised_eval_local_model
cd "$SRC"
"$PY" scripts/evaluate_revised_expert.py --policy expert \
  --evidence-profile "${REVISED_EVIDENCE_PROFILE:-scada_only}" \
  --scenarios "$FROZEN_SCENARIOS" --output-dir "$EXPERT_EVAL_DIR" \
  --baseline-eval "$EXPERT_BASELINE" --resume
