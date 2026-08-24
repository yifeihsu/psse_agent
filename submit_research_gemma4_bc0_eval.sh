#!/usr/bin/env bash
#SBATCH --job-name=gemma4-bc0-eval
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --open-mode=append
#SBATCH --output=gemma4_bc0_eval_%j.out
#SBATCH --error=gemma4_bc0_eval_%j.err

# Research-only closed-loop evaluation of the completed Gemma-4-12B BC0
# adapter. Select exactly one phase with RESEARCH_EVAL_PHASE=d0 or d1.

set -Eeuo pipefail

: "${RESEARCH_SOURCE_ROOT:?set RESEARCH_SOURCE_ROOT to the research checkout}"
: "${RESEARCH_ENV:?set RESEARCH_ENV to the research conda environment}"
: "${RESEARCH_RUN_ROOT:?set RESEARCH_RUN_ROOT to the persistent run directory}"
: "${RESEARCH_EVAL_PHASE:?set RESEARCH_EVAL_PHASE to d0 or d1}"

case "$RESEARCH_EVAL_PHASE" in
  d0|d1) ;;
  *)
    echo "RESEARCH_EVAL_PHASE must be d0 or d1" >&2
    exit 2
    ;;
esac

D0_ROOT="${RESEARCH_D0_ROOT:-/scratch/yx3882/dagger_release_a5a7574_20260823/round0_aggregate_release}"
D0_TRAIN="$D0_ROOT/aggregate.train_view.jsonl"
FULL_BC0="$RESEARCH_RUN_ROOT/bc0/full"
POSTFLIGHT="$FULL_BC0/full_bc0_postflight.json"
ADAPTER_DIR="$FULL_BC0/lora"
OUTPUT_DIR="$RESEARCH_RUN_ROOT/evaluation/bc0_12b"
D0_REPORT="$OUTPUT_DIR/research_bc0_d0_eval.json"

for required in \
  "$D0_TRAIN" \
  "$POSTFLIGHT" \
  "$ADAPTER_DIR/adapter_config.json" \
  "$ADAPTER_DIR/adapter_model.safetensors"
do
  if [[ ! -f "$required" ]]; then
    echo "required BC0 evaluation input is missing: $required" >&2
    exit 2
  fi
done
if [[ "$RESEARCH_EVAL_PHASE" == "d1" && ! -f "$D0_REPORT" ]]; then
  echo "D1 requires the completed D0 report: $D0_REPORT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.${RESEARCH_EVAL_PHASE}.lock"
if ! flock -n 9; then
  echo "another ${RESEARCH_EVAL_PHASE} BC0 evaluation owns the phase lock" >&2
  exit 73
fi

module purge
module load anaconda3/2025.06
# shellcheck source=/dev/null
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$RESEARCH_ENV"
cd "$RESEARCH_SOURCE_ROOT"

python - "$POSTFLIGHT" <<'PY'
import json
import pathlib
import sys

postflight = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if postflight.get("passed") is not True:
    raise SystemExit("full BC0 postflight has not passed")
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RESEARCH_SOURCE_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${RESEARCH_HF_HOME:-/scratch/yx3882/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="${RESEARCH_TORCH_HOME:-/scratch/yx3882/.cache/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RESEARCH_MAX_INPUT_TOKENS=32768
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTORCH_ALLOC_CONF="${RESEARCH_PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python -m psse_env.sft research-cache \
  --model-choice 12b \
  --output "$OUTPUT_DIR/cache_preflight_${RESEARCH_EVAL_PHASE}.json"

TELEMETRY="$OUTPUT_DIR/nvidia_smi_${RESEARCH_EVAL_PHASE}_${SLURM_JOB_ID}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits \
  --loop=10 >"$TELEMETRY" 2>"$OUTPUT_DIR/nvidia_smi_${RESEARCH_EVAL_PHASE}_${SLURM_JOB_ID}.err" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

eval_args=(
  --phase "$RESEARCH_EVAL_PHASE"
  --adapter-path "$ADAPTER_DIR"
  --d0-train "$D0_TRAIN"
  --output-dir "$OUTPUT_DIR"
  --max-steps 24
)
if [[ "$RESEARCH_EVAL_PHASE" == "d1" ]]; then
  eval_args+=(--d0-report "$D0_REPORT")
fi

python -m psse_env.sft research-bc0-eval "${eval_args[@]}"
