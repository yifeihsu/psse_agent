#!/usr/bin/env bash
#SBATCH --job-name=gemma4-bc0-replay-compare
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
#SBATCH --output=gemma4_bc0_replay_compare_%j.out
#SBATCH --error=gemma4_bc0_replay_compare_%j.err

# Research-only, single-GPU replay of the published Gemma-4-12B BC0 adapter
# followed by a resumable comparison of the saved training checkpoints on the
# exact protected D1 development suite produced by the replay.  This launcher
# never trains or modifies adapter weights.

set -Eeuo pipefail

: "${RESEARCH_SOURCE_ROOT:?set RESEARCH_SOURCE_ROOT to the research checkout}"
: "${RESEARCH_ENV:?set RESEARCH_ENV to the research conda environment}"
: "${RESEARCH_RUN_ROOT:?set RESEARCH_RUN_ROOT to the persistent run directory}"

D0_ROOT="${RESEARCH_D0_ROOT:-/scratch/yx3882/dagger_release_a5a7574_20260823/round0_aggregate_release}"
D0_TRAIN="$D0_ROOT/aggregate.train_view.jsonl"
FULL_BC0="$RESEARCH_RUN_ROOT/bc0/full"
POSTFLIGHT="$FULL_BC0/full_bc0_postflight.json"
PUBLISHED_ADAPTER="$FULL_BC0/lora"
OUTPUT_ROOT="${RESEARCH_COMPARISON_OUTPUT_ROOT:-$RESEARCH_RUN_ROOT/evaluation/bc0_12b_replay_compare_v2}"
REPLAY_DIR="$OUTPUT_ROOT/published_replay"
COMPARISON_DIR="$OUTPUT_ROOT/checkpoint_comparison_d1"
CHECKPOINT_STEPS="${RESEARCH_CHECKPOINT_STEPS:-64 128 192 256 320}"

for required in \
  "$D0_TRAIN" \
  "$POSTFLIGHT" \
  "$PUBLISHED_ADAPTER/adapter_config.json" \
  "$PUBLISHED_ADAPTER/adapter_model.safetensors"
do
  if [[ ! -f "$required" ]]; then
    echo "required replay input is missing: $required" >&2
    exit 2
  fi
done

checkpoint_args=()
for step in $CHECKPOINT_STEPS; do
  checkpoint="$FULL_BC0/checkpoint-$step"
  for required in \
    "$checkpoint/adapter_config.json" \
    "$checkpoint/adapter_model.safetensors"
  do
    if [[ ! -f "$required" ]]; then
      echo "required checkpoint input is missing: $required" >&2
      exit 2
    fi
  done
  checkpoint_args+=(--adapter "checkpoint-$step=$checkpoint")
done

mkdir -p "$REPLAY_DIR" "$COMPARISON_DIR"
exec 9>"$OUTPUT_ROOT/.replay_compare.lock"
if ! flock -n 9; then
  echo "another BC0 replay/checkpoint comparison owns $OUTPUT_ROOT" >&2
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
  --output "$OUTPUT_ROOT/cache_preflight.json"

TELEMETRY="$OUTPUT_ROOT/nvidia_smi_${SLURM_JOB_ID}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits \
  --loop=10 >"$TELEMETRY" 2>"$OUTPUT_ROOT/nvidia_smi_${SLURM_JOB_ID}.err" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

report_ready() {
  python - "$1" "$2" "$3" <<'PY'
import json
import pathlib
import sys

from psse_env.sft.research_bc0_eval import (
    RESEARCH_BC0_EVAL_CONTRACT,
    adapter_content_fingerprint,
)

report_path = pathlib.Path(sys.argv[1])
phase = sys.argv[2]
adapter_path = pathlib.Path(sys.argv[3])
if not report_path.is_file():
    raise SystemExit(1)
report = json.loads(report_path.read_text(encoding="utf-8"))
expected = adapter_content_fingerprint(adapter_path)["content_sha256"]
ready = (
    report.get("contract") == RESEARCH_BC0_EVAL_CONTRACT
    and report.get("phase") == phase
    and report.get("evaluation_completed") is True
    and report.get("readiness_gate", {}).get("passed") is True
    and report.get("adapter", {}).get("content_sha256") == expected
    and isinstance(report.get("closed_loop_outcomes"), dict)
)
raise SystemExit(0 if ready else 1)
PY
}

D0_REPORT="$REPLAY_DIR/research_bc0_d0_eval.json"
if ! report_ready "$D0_REPORT" d0 "$PUBLISHED_ADAPTER"; then
  python -m psse_env.sft research-bc0-eval \
    --phase d0 \
    --adapter-path "$PUBLISHED_ADAPTER" \
    --d0-train "$D0_TRAIN" \
    --output-dir "$REPLAY_DIR" \
    --max-steps 24
fi

D1_REPORT="$REPLAY_DIR/research_bc0_d1_eval.json"
if ! report_ready "$D1_REPORT" d1 "$PUBLISHED_ADAPTER"; then
  python -m psse_env.sft research-bc0-eval \
    --phase d1 \
    --adapter-path "$PUBLISHED_ADAPTER" \
    --d0-train "$D0_TRAIN" \
    --d0-report "$D0_REPORT" \
    --output-dir "$REPLAY_DIR" \
    --max-steps 24
fi

python -m psse_env.sft.research_bc0_checkpoint_compare \
  --suite-json "$REPLAY_DIR/d1_development_suite.json" \
  --suite-name standard_success \
  "${checkpoint_args[@]}" \
  --output-dir "$COMPARISON_DIR" \
  --max-steps 24
