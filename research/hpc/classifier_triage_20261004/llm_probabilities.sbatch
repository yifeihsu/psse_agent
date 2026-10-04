#!/usr/bin/env bash
#SBATCH --job-name=triage-prob
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/classifier_triage_20261004/logs/%x-%j.out
# The triage fine-tune read as a scored classifier (docs/classifier_triage_plan_20261004.md):
# the probability of each first action on the calibration rows, the test rows
# and the probe sample, from the adapter llm_triage.sbatch trained.
#
#   sbatch --export=ALL,VARIANT=prompt_top10_signed --job-name=triage-prob-top10s llm_probabilities.sbatch
#
# No training.  Scoring resumes from the rows already written.
set -euo pipefail
WORK=/scratch/yx3882/classifier_triage_20261004
VARIANT=${VARIANT:?set VARIANT to a prompt variant directory under data/}
SRC=$WORK/source
DATA=$WORK/data/$VARIANT
RUN=$WORK/out/$VARIANT
LOGS=$WORK/logs
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/scratch/yx3882/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
export RESEARCH_MAX_INPUT_TOKENS=32768 RESEARCH_MAX_NEW_TOKENS=256 PSSE_LOCAL_DIAGNOSTIC_BUILD=1

echo "[triage] probabilities $VARIANT job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[triage] source $(git -C "$SRC" rev-parse HEAD)"
[[ -s "$RUN/lora/adapter_model.safetensors" ]] || { echo "no adapter under $RUN/lora" >&2; exit 2; }
[[ -s "$DATA/score.jsonl" && -s "$DATA/score_labels.json" ]] || { echo "dataset for $VARIANT is incomplete under $DATA" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-prob-$VARIANT-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
trap 'kill "$GPU_SAMPLER_PID" 2>/dev/null || true' EXIT

cd "$SRC"
"$PY" -m research.classifier_triage.llm_score \
  --adapter "$RUN/lora" \
  --score "$DATA/score.jsonl" \
  --output "$RUN/probabilities.json" \
  --probe-per-cell 100 \
  --probabilities
echo triage-probabilities-complete
