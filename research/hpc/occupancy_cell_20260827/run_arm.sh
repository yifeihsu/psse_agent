#!/bin/bash
set -Eeuo pipefail

: "${CELL_CONFIG:?set CELL_CONFIG}"
: "${CELL_CONFIG_SHA256:?set CELL_CONFIG_SHA256}"
: "${CELL_PYTHON:?set CELL_PYTHON}"
: "${CELL_ARM:?set CELL_ARM to A, B, C, or D}"
: "${CELL_EXPECTED_GPU_FEATURE:?set by submit.sh}"
: "${CELL_EXPECTED_GPU_FAMILY:?set by submit.sh}"
[[ "$CELL_ARM" =~ ^[ABCD]$ ]] || exit 2
GPU_FAMILY_PATTERN='^(A100|H100|H200|RTX6000)(\|(A100|H100|H200|RTX6000))*$'
[[ "$CELL_EXPECTED_GPU_FAMILY" =~ $GPU_FAMILY_PATTERN ]] || exit 2

SUBMITTED_SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
CONFIG_SOURCE_ROOT=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_root"])
PY
)
EXPECTED_SCRIPT_DIR=$(readlink -f "$CONFIG_SOURCE_ROOT/research/hpc/occupancy_cell_20260827")
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
MODEL_ID=${ARM[0]}; MODEL_REVISION=${ARM[1]}; UPDATES=${ARM[2]}
TRAIN=${ARM[3]}; VALIDATION=${ARM[4]}; LANE=${ARM[7]}

JOB_ID=${SLURM_JOB_ID:?run_arm.sh must run under Slurm}
RESTART=${SLURM_RESTART_COUNT:-0}
[[ "$JOB_ID" =~ ^[0-9]+$ && "$RESTART" =~ ^[0-9]+$ ]] || exit 2

# Torch does not currently export SLURM_JOB_CONSTRAINTS into batch jobs.  Read
# the job-level Constraints field back from Slurm accounting unconditionally,
# so an inherited SLURM_* value can never be mistaken for scheduler evidence.
# Capture the pipeline before splitting it: process substitution would hide a
# failing sacct exit status even under pipefail.
read_scheduler_constraint() {
  local raw
  local -a values=()
  if ! raw=$(
    sacct -X -j "$JOB_ID" --format=Constraints -n -P 2>/dev/null \
      | awk 'NF {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}'
  ); then
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
  if SCHEDULER_GPU_CONSTRAINT=$(read_scheduler_constraint); then
    break
  fi
  sleep 2
done
SCHEDULER_GPU_CONSTRAINT_SOURCE=sacct
[[ -n "$SCHEDULER_GPU_CONSTRAINT" ]] \
  || { echo "Slurm did not attest the applied job constraint" >&2; exit 2; }
[[ "$SCHEDULER_GPU_CONSTRAINT" == "$CELL_EXPECTED_GPU_FEATURE" ]] \
  || { echo "Slurm constraint drift" >&2; exit 2; }
export CELL_SCHEDULER_GPU_CONSTRAINT=$SCHEDULER_GPU_CONSTRAINT
export CELL_SCHEDULER_GPU_CONSTRAINT_SOURCE=$SCHEDULER_GPU_CONSTRAINT_SOURCE

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
"$CELL_PYTHON" - <<'PY' > "$ALLOCATED_GPU"
import json
import os

import torch

count = torch.cuda.device_count()
if count != 1:
    raise RuntimeError(f"expected exactly one CUDA-visible device, got {count}")
properties = torch.cuda.get_device_properties(0)
json.dump(
    {
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_constraint": os.environ["CELL_SCHEDULER_GPU_CONSTRAINT"],
        "slurm_job_constraint_source": os.environ[
            "CELL_SCHEDULER_GPU_CONSTRAINT_SOURCE"
        ],
        "torch_cuda_device_count": count,
        "torch_device_index": 0,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
    },
    fp=__import__("sys").stdout,
    sort_keys=True,
    allow_nan=False,
)
print()
PY
GPU_NAME=$("$CELL_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["name"])' \
  "$ALLOCATED_GPU")
OBSERVED_GPU_FAMILY=""
IFS='|' read -ra ALLOWED_GPU_FAMILIES <<<"$CELL_EXPECTED_GPU_FAMILY"
for family in "${ALLOWED_GPU_FAMILIES[@]}"; do
  case "$family" in
    A100|H100|H200) name_pattern=$family ;;
    RTX6000) name_pattern='RTX([[:space:]]+PRO)?[[:space:]]*6000' ;;
    *) exit 2 ;;
  esac
  if grep -Eqi -- "$name_pattern" <<<"$GPU_NAME"; then
    OBSERVED_GPU_FAMILY=$family
    break
  fi
done
[[ -n "$OBSERVED_GPU_FAMILY" ]] \
  || { echo "observed GPU does not match $CELL_EXPECTED_GPU_FAMILY" >&2; exit 2; }
export CELL_OBSERVED_GPU_FAMILY=$OBSERVED_GPU_FAMILY

ADAPTER="$ATTEMPT/adapter"
"$CELL_PYTHON" -m research.train --train "$TRAIN" --validation "$VALIDATION" \
  --output-dir "$ADAPTER" --model-id "$MODEL_ID" --revision "$MODEL_REVISION" \
  --max-length 8192 --epochs 2 --batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 0.0001 --seed 20260823 --max-steps "$UPDATES" \
  --report-to none --run-name "occupancy-cell-$CELL_ARM" \
  > "$ATTEMPT/training_console.json"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" gate-training --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" \
  --summary "$ADAPTER/research_training_summary.json" --output "$ATTEMPT/training_gate.json"

EVALUATION="$ATTEMPT/evaluation.schema-v2.json"
SCENARIOS=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["inputs"]["evaluation_scenarios"]["path"])
PY
)
"$CELL_PYTHON" -m research.evaluate --scenarios "$SCENARIOS" --adapter "$ADAPTER" \
  --label "occupancy-cell-$CELL_ARM" --output "$EVALUATION" --max-steps 24 \
  --model-id "$MODEL_ID" --revision "$MODEL_REVISION" > "$ATTEMPT/evaluation_console.json"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" gate-evaluation --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" \
  --training-gate "$ATTEMPT/training_gate.json" --evaluation "$EVALUATION" \
  --output "$ATTEMPT/evaluation_gate.json"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" publish-arm --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" \
  --job-id "$JOB_ID" --attempt "$ATTEMPT"
