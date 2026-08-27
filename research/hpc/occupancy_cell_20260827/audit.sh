#!/bin/bash
set -Eeuo pipefail

: "${CELL_CONFIG:?set CELL_CONFIG}"
: "${CELL_CONFIG_SHA256:?set CELL_CONFIG_SHA256}"
: "${CELL_PYTHON:?set CELL_PYTHON}"
: "${CELL_ARM:?set CELL_ARM}"
: "${CELL_PARENT_JOB_ID:?set CELL_PARENT_JOB_ID}"
[[ "$CELL_ARM" =~ ^[ABCD]$ && "$CELL_PARENT_JOB_ID" =~ ^[0-9]+$ ]] || exit 2

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
CELL_ROOT=${CONFIG[0]}; SOURCE_ROOT=${CONFIG[1]}
[[ "$CELL_PYTHON" == "${CONFIG[2]}" ]] || exit 2
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SOURCE_ROOT"
POINTER_JSON=$("$CELL_PYTHON" "$SCRIPT_DIR/build.py" verify-arm --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" --job-id "$CELL_PARENT_JOB_ID")
EVALUATION=$("$CELL_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["evaluation"])' <<<"$POINTER_JSON")
SCENARIOS=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["inputs"]["evaluation_scenarios"]["path"])
PY
)
JOB_ID=${SLURM_JOB_ID:?audit.sh must run under Slurm}
RESTART=${SLURM_RESTART_COUNT:-0}
[[ "$JOB_ID" =~ ^[0-9]+$ && "$RESTART" =~ ^[0-9]+$ ]] || exit 2
ATTEMPT="$CELL_ROOT/audits/$CELL_ARM/job-$JOB_ID/attempt-r$RESTART"
mkdir -p "$(dirname "$ATTEMPT")"
mkdir "$ATTEMPT"
FULL="$ATTEMPT/physical_audit.full.json"
SUMMARY="$ATTEMPT/physical_audit.summary.json"
"$CELL_PYTHON" -m research.physical_outcome_audit --scenarios "$SCENARIOS" \
  --evaluation "occupancy-cell-$CELL_ARM=$EVALUATION" --output "$FULL" \
  --summary-output "$SUMMARY" --max-steps 24 > "$ATTEMPT/audit_console.json"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" gate-audit --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --arm "$CELL_ARM" \
  --parent-job "$CELL_PARENT_JOB_ID" --full "$FULL" --summary "$SUMMARY" \
  --output "$ATTEMPT/audit_gate.json"
