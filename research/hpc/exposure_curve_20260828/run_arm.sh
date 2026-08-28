#!/bin/bash
set -Eeuo pipefail

: "${CELL_CONFIG:?set CELL_CONFIG}"
: "${CELL_CONFIG_SHA256:?set CELL_CONFIG_SHA256}"
: "${CELL_PYTHON:?set CELL_PYTHON}"
: "${CELL_ARM:?set CELL_ARM to A, B, C, D, E, or F}"
: "${CELL_EXPECTED_GPU_FEATURE:?set by submit.sh}"
: "${CELL_EXPECTED_GPU_FAMILY:?set by submit.sh}"
[[ "$CELL_ARM" =~ ^[ABCDEF]$ ]] || exit 2
[[ "$CELL_EXPECTED_GPU_FEATURE" == 'a100|h100|h200|rtx6000' ]] || exit 2
[[ "$CELL_EXPECTED_GPU_FAMILY" == 'A100|H100|H200|RTX6000' ]] || exit 2

SUBMITTED_SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
CONFIG_SOURCE_ROOT=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_root"])
PY
)
EXPECTED_SCRIPT_DIR=$(readlink -f "$CONFIG_SOURCE_ROOT/research/hpc/exposure_curve_20260828")
export PYTHONPATH="$CONFIG_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
EXPECTED_SCRIPT="$EXPECTED_SCRIPT_DIR/run_arm.sh"
if [[ "$SUBMITTED_SCRIPT" != "$EXPECTED_SCRIPT" ]]; then
  cmp -s "$SUBMITTED_SCRIPT" "$EXPECTED_SCRIPT" \
    || { echo "Slurm script copy differs from the configured source" >&2; exit 2; }
fi
SCRIPT_DIR=$EXPECTED_SCRIPT_DIR
mapfile -t CONFIG < <("$CELL_PYTHON" "$SCRIPT_DIR/build.py" config-values \
  --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256")
CELL_ROOT=${CONFIG[0]}; SOURCE_ROOT=${CONFIG[1]}
[[ "$CELL_PYTHON" == "${CONFIG[2]}" ]] || exit 2
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${CONFIG[3]}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$SOURCE_ROOT"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" environment --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" verify-model-cache --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM"
mapfile -t ARM < <("$CELL_PYTHON" "$SCRIPT_DIR/build.py" arm-values \
  --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM")
MODEL_ID=${ARM[0]}; MODEL_REVISION=${ARM[1]}; MAX_STEPS=${ARM[2]}
TRAIN=${ARM[3]}; VALIDATION=${ARM[4]}; LANE=${ARM[7]}
RESTART_CONTRACT=${ARM[11]}; PASS_NORMALIZED_TOTAL_ROWS=${ARM[12]}
MILESTONE_SPEC=${ARM[13]}

JOB_ID=${SLURM_JOB_ID:?run_arm.sh must run under Slurm}
RESTART=${SLURM_RESTART_COUNT:-0}
[[ "$JOB_ID" =~ ^[0-9]+$ && "$RESTART" =~ ^[0-9]+$ ]] || exit 2

read_scheduler_constraint() {
  local raw
  local -a values=()
  if ! raw=$(sacct -X -j "$JOB_ID" --format=Constraints -n -P 2>/dev/null \
    | awk 'NF {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}'); then
    return 1
  fi
  [[ -n "$raw" ]] || return 1
  mapfile -t values <<<"$raw"
  [[ ${#values[@]} -eq 1 && -n "${values[0]}" && "${values[0]}" != "(null)" ]] \
    || return 1
  printf '%s' "${values[0]}"
}
SCHEDULER_GPU_CONSTRAINT=""
for _ in 1 2 3; do
  if SCHEDULER_GPU_CONSTRAINT=$(read_scheduler_constraint); then break; fi
  sleep 2
done
[[ "$SCHEDULER_GPU_CONSTRAINT" == "$CELL_EXPECTED_GPU_FEATURE" ]] \
  || { echo "Slurm constraint attestation failed" >&2; exit 2; }
export CELL_SCHEDULER_GPU_CONSTRAINT=$SCHEDULER_GPU_CONSTRAINT
export CELL_SCHEDULER_GPU_CONSTRAINT_SOURCE=sacct

JOB_ROOT="$CELL_ROOT/runs/$CELL_ARM/job-$JOB_ID"
if [[ -f "$JOB_ROOT/completed.json" ]]; then
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" verify-arm --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID"
  exit 0
fi
ATTEMPT="$JOB_ROOT/attempt-r$RESTART"
mkdir -p "$JOB_ROOT"
mkdir "$ATTEMPT"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$ATTEMPT/gpu_inventory.csv"
: "${CUDA_VISIBLE_DEVICES:?Slurm must bind exactly one visible GPU}"
ALLOCATED_GPU="$ATTEMPT/allocated_gpu.json"
ALLOCATED_GPU_TMP="$ATTEMPT/.allocated_gpu.json.tmp"
"$CELL_PYTHON" - <<'PY' > "$ALLOCATED_GPU_TMP"
import json, os, torch
count = torch.cuda.device_count()
if count != 1:
    raise RuntimeError(f"expected exactly one CUDA-visible device, got {count}")
properties = torch.cuda.get_device_properties(0)
json.dump({
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
    "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
    "slurm_job_constraint": os.environ["CELL_SCHEDULER_GPU_CONSTRAINT"],
    "slurm_job_constraint_source": os.environ["CELL_SCHEDULER_GPU_CONSTRAINT_SOURCE"],
    "torch_cuda_device_count": count,
    "torch_device_index": 0,
    "name": properties.name,
    "total_memory_bytes": properties.total_memory,
    "compute_capability": [properties.major, properties.minor],
}, fp=__import__("sys").stdout, sort_keys=True, allow_nan=False)
print()
PY
mv "$ALLOCATED_GPU_TMP" "$ALLOCATED_GPU"
GPU_NAME=$("$CELL_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["name"])' \
  "$ALLOCATED_GPU")
if grep -Eqi 'A100' <<<"$GPU_NAME"; then OBSERVED_GPU_FAMILY=A100
elif grep -Eqi 'H100' <<<"$GPU_NAME"; then OBSERVED_GPU_FAMILY=H100
elif grep -Eqi 'H200' <<<"$GPU_NAME"; then OBSERVED_GPU_FAMILY=H200
elif grep -Eqi 'RTX([[:space:]]+PRO)?[[:space:]]*6000' <<<"$GPU_NAME"; then
  OBSERVED_GPU_FAMILY=RTX6000
else
  echo "observed GPU is outside the authorized union: $GPU_NAME" >&2
  exit 2
fi
export CELL_OBSERVED_GPU_FAMILY=$OBSERVED_GPU_FAMILY

if [[ -f "$JOB_ROOT/training.completed.json" ]]; then
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" verify-training --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID"
else
  RESUME=$("$CELL_PYTHON" "$SCRIPT_DIR/build.py" find-resume --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" \
    --job-id "$JOB_ID" --restart "$RESTART")
  RESUME_ARGS=()
  if [[ -n "$RESUME" ]]; then RESUME_ARGS=(--resume-from-checkpoint "$RESUME"); fi
  MILESTONES="$JOB_ROOT/milestones"
  MILESTONE_ARGS=()
  IFS=',' read -ra PAIRS <<<"$MILESTONE_SPEC"
  [[ ${#PAIRS[@]} -eq 4 ]] || exit 2
  for pair in "${PAIRS[@]}"; do MILESTONE_ARGS+=(--milestone-step "$pair"); done
  ADAPTER="$ATTEMPT/adapter"
  "$CELL_PYTHON" -m research.train --train "$TRAIN" --validation "$VALIDATION" \
    --output-dir "$ADAPTER" --model-id "$MODEL_ID" --revision "$MODEL_REVISION" \
    --max-length 8192 --epochs 2 --batch-size 1 --gradient-accumulation-steps 4 \
    --learning-rate 0.0001 --seed 20260823 --max-steps "$MAX_STEPS" \
    --pass-normalized-total-rows "$PASS_NORMALIZED_TOTAL_ROWS" \
    --restart-save-steps 25 --restart-save-total-limit 2 \
    --restart-contract "$RESTART_CONTRACT" \
    --milestone-output-dir "$MILESTONES" "${MILESTONE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" --report-to none --run-name "exposure-curve-$CELL_ARM" \
    > "$ATTEMPT/training_console.json"
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" gate-training --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID" \
    --summary "$ADAPTER/research_training_summary.json" \
    --output "$ATTEMPT/training_gate.json"
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" publish-training --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID" \
    --gate "$ATTEMPT/training_gate.json"
fi

SCENARIOS=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["inputs"]["evaluation_scenarios"]["path"])
PY
)
for MILESTONE in p075 p100 p150 p200; do
  COMPLETED="$JOB_ROOT/evaluations/$MILESTONE/completed.json"
  if [[ -f "$COMPLETED" ]]; then
    "$CELL_PYTHON" "$SCRIPT_DIR/build.py" verify-evaluation --config "$CELL_CONFIG" \
      --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" \
      --job-id "$JOB_ID" --milestone "$MILESTONE"
    continue
  fi
  EVAL_ATTEMPT="$ATTEMPT/evaluation-$MILESTONE"
  mkdir "$EVAL_ATTEMPT"
  EVALUATION="$EVAL_ATTEMPT/evaluation.schema-v2.json"
  "$CELL_PYTHON" -m research.evaluate --scenarios "$SCENARIOS" \
    --adapter "$JOB_ROOT/milestones/$MILESTONE" \
    --label "exposure-curve-$CELL_ARM-$MILESTONE" --output "$EVALUATION" \
    --max-steps 24 --model-id "$MODEL_ID" --revision "$MODEL_REVISION" \
    > "$EVAL_ATTEMPT/evaluation_console.json"
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" gate-evaluation --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID" \
    --milestone "$MILESTONE" --evaluation "$EVALUATION" \
    --output "$EVAL_ATTEMPT/evaluation_gate.json"
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" publish-evaluation --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID" \
    --milestone "$MILESTONE" --gate "$EVAL_ATTEMPT/evaluation_gate.json"
done
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" publish-arm --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$JOB_ID"
