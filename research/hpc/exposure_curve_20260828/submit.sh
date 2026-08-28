#!/bin/bash
set -Eeuo pipefail

: "${CELL_CONFIG:?set CELL_CONFIG to the immutable JSON configuration}"
: "${CELL_PYTHON:?set CELL_PYTHON to the configured Python executable}"
: "${CELL_SELECTED_ARMS:?set CELL_SELECTED_ARMS, normally A,B,C,D,E,F}"
GPU_FEATURE_UNION='a100|h100|h200|rtx6000'
GPU_FAMILY_UNION='A100|H100|H200|RTX6000'
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CELL_CONFIG=$(readlink -f "$CELL_CONFIG")
case "$CELL_CONFIG" in
  *','*|*$'\n'*|*$'\r'*|*$'\t'*) echo "CELL_CONFIG is not export-safe" >&2; exit 2 ;;
esac
CONFIG_SOURCE_ROOT=$("$CELL_PYTHON" - "$CELL_CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_root"])
PY
)
EXPECTED_SCRIPT_DIR=$(readlink -f "$CONFIG_SOURCE_ROOT/research/hpc/exposure_curve_20260828")
export PYTHONPATH="$CONFIG_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(readlink -f "$SCRIPT_DIR")" == "$EXPECTED_SCRIPT_DIR" ]] \
  || { echo "submit.sh is outside the configured source tree" >&2; exit 2; }
CELL_CONFIG_SHA256=$("$CELL_PYTHON" "$SCRIPT_DIR/build.py" sha256 --path "$CELL_CONFIG")
export CELL_CONFIG CELL_CONFIG_SHA256 CELL_PYTHON
mapfile -t CONFIG < <("$CELL_PYTHON" "$SCRIPT_DIR/build.py" config-values \
  --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256")
CELL_ROOT=${CONFIG[0]}; SOURCE_ROOT=${CONFIG[1]}
[[ "$CELL_PYTHON" == "${CONFIG[2]}" ]] || exit 2

IFS=',' read -ra SELECTED_ARMS <<<"$CELL_SELECTED_ARMS"
[[ ${#SELECTED_ARMS[@]} -ge 1 ]] || exit 2
declare -A SEEN_ARMS=()
for arm in "${SELECTED_ARMS[@]}"; do
  [[ "$arm" =~ ^[ABCDEF]$ ]] || { echo "invalid selected arm: $arm" >&2; exit 2; }
  [[ -z "${SEEN_ARMS[$arm]:-}" ]] || { echo "duplicate selected arm: $arm" >&2; exit 2; }
  SEEN_ARMS[$arm]=1
done
mkdir -p "$CELL_ROOT/logs"
GPU_RECEIPT="e2b=$GPU_FEATURE_UNION,12b=$GPU_FEATURE_UNION;families=e2b=$GPU_FAMILY_UNION,12b=$GPU_FAMILY_UNION"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-begin --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --gpu-constraints "$GPU_RECEIPT" \
  --selected-arms "$CELL_SELECTED_ARMS"
SUBMIT_FINISHED=0
SUBMITTED_JOBS=()
submission_cleanup() {
  status=$?
  trap - EXIT
  if [[ $SUBMIT_FINISHED != 1 ]]; then
    cancelled=""
    for id in "${SUBMITTED_JOBS[@]}"; do
      scancel "$id" || true
      cancelled+="${cancelled:+,}$id"
    done
    "$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-fail \
      --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256" \
      --detail "submit.sh exit $status; cancellation requested for [$cancelled]" || true
  fi
  exit "$status"
}
trap submission_cleanup EXIT

job_id() {
  local raw=$1
  local id=${raw%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || return 2
  printf '%s' "$id"
}
common=(--parsable --account=torch_pr_627_general --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=96G --requeue --comment="preemption=yes;requeue=true" \
  --open-mode=append --chdir="$SOURCE_ROOT")
case "${PATH:-}" in
  *','*|*$'\n'*|*$'\r'*|*$'\t'*) echo "PATH is not export-safe" >&2; exit 2 ;;
esac
export_args="CELL_CONFIG=$CELL_CONFIG,CELL_CONFIG_SHA256=$CELL_CONFIG_SHA256,CELL_PYTHON=$CELL_PYTHON,PATH=${PATH:-/usr/bin:/bin}"

BUILD=$(job_id "$(sbatch "${common[@]}" --hold --job-name=curve-build --time=02:00:00 \
  --output="$CELL_ROOT/logs/build-%j.out" --error="$CELL_ROOT/logs/build-%j.err" \
  --export="$export_args" "$SCRIPT_DIR/build.sh")")
SUBMITTED_JOBS+=("$BUILD")
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-job --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --role build --job-id "$BUILD"

declare -A ARM_JOBS
for arm in "${SELECTED_ARMS[@]}"; do
  case "$arm" in
    A|B|C) walltime=24:00:00 ;;
    D|E|F) walltime=36:00:00 ;;
    *) exit 2 ;;
  esac
  ARM_JOBS[$arm]=$(job_id "$(sbatch "${common[@]}" --job-name="curve-$arm" \
    --dependency="afterok:$BUILD" --time="$walltime" --gres=gpu:1 \
    --constraint="$GPU_FEATURE_UNION" \
    --output="$CELL_ROOT/logs/arm-$arm-%j.out" --error="$CELL_ROOT/logs/arm-$arm-%j.err" \
    --export="$export_args,CELL_ARM=$arm,CELL_EXPECTED_GPU_FEATURE=$GPU_FEATURE_UNION,CELL_EXPECTED_GPU_FAMILY=$GPU_FAMILY_UNION" \
    "$SCRIPT_DIR/run_arm.sh")")
  SUBMITTED_JOBS+=("${ARM_JOBS[$arm]}")
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-job --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --role arm --arm "$arm" \
    --job-id "${ARM_JOBS[$arm]}"
done

for arm in "${SELECTED_ARMS[@]}"; do
  audit=$(job_id "$(sbatch "${common[@]}" --job-name="curve-audit-$arm" \
    --dependency="afterok:${ARM_JOBS[$arm]}" --time=08:00:00 \
    --output="$CELL_ROOT/logs/audit-$arm-%j.out" --error="$CELL_ROOT/logs/audit-$arm-%j.err" \
    --export="$export_args,CELL_ARM=$arm,CELL_PARENT_JOB_ID=${ARM_JOBS[$arm]}" \
    "$SCRIPT_DIR/audit.sh")")
  SUBMITTED_JOBS+=("$audit")
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-job --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --role audit --arm "$arm" --job-id "$audit"
done
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-finish --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256"
scontrol release "$BUILD"
SUBMIT_FINISHED=1
trap - EXIT
echo "Submitted six-arm exposure curve; receipt: $CELL_ROOT/submission.json"
