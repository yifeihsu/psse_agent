#!/bin/bash
set -Eeuo pipefail

: "${CELL_CONFIG:?set CELL_CONFIG}"
: "${CELL_CONFIG_SHA256:?set CELL_CONFIG_SHA256}"
: "${CELL_PYTHON:?set CELL_PYTHON}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_SOURCE_ROOT=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_root"])
PY
)
EXPECTED_SCRIPT_DIR=$(readlink -f "$CONFIG_SOURCE_ROOT/research/hpc/occupancy_cell_20260827")
[[ "$(readlink -f "$SCRIPT_DIR")" == "$EXPECTED_SCRIPT_DIR" ]] \
  || { echo "launcher is outside the configured source tree" >&2; exit 2; }
mapfile -t CONFIG < <("$CELL_PYTHON" "$SCRIPT_DIR/build.py" config-values \
  --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256")
CELL_ROOT=${CONFIG[0]}
SOURCE_ROOT=${CONFIG[1]}
[[ "$CELL_PYTHON" == "${CONFIG[2]}" ]] || { echo "configured Python drift" >&2; exit 2; }
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$SOURCE_ROOT"
"$CELL_PYTHON" -m pip check
"$CELL_PYTHON" -m pytest research -q
"$CELL_PYTHON" -m compileall -q research

if [[ -f "$CELL_ROOT/build/completed.json" ]]; then
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" verify-build \
    --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256"
  exit 0
fi
JOB_ID=${SLURM_JOB_ID:?build.sh must run under Slurm}
RESTART=${SLURM_RESTART_COUNT:-0}
[[ "$JOB_ID" =~ ^[0-9]+$ && "$RESTART" =~ ^[0-9]+$ ]] || exit 2
ATTEMPT="$CELL_ROOT/build/job-$JOB_ID/attempt-r$RESTART"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" build --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --attempt "$ATTEMPT"
