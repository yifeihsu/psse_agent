#!/bin/bash
set -Eeuo pipefail

: "${CELL_CONFIG:?set CELL_CONFIG to the immutable JSON configuration}"
: "${CELL_PYTHON:?set CELL_PYTHON to the configured Python executable}"
: "${CELL_GPU_CONSTRAINT:?set CELL_GPU_CONSTRAINT explicitly; see README.md}"
: "${CELL_GPU_FAMILY:?set CELL_GPU_FAMILY explicitly; see README.md}"
: "${CELL_SELECTED_ARMS:?set CELL_SELECTED_ARMS explicitly, for example E}"
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
EXPECTED_SCRIPT_DIR=$(readlink -f "$CONFIG_SOURCE_ROOT/research/hpc/occupancy_cell_20260827")
[[ "$(readlink -f "$SCRIPT_DIR")" == "$EXPECTED_SCRIPT_DIR" ]] \
  || { echo "submit.sh is outside the configured source tree" >&2; exit 2; }
CELL_CONFIG_SHA256=$("$CELL_PYTHON" "$SCRIPT_DIR/build.py" sha256 --path "$CELL_CONFIG")
export CELL_CONFIG CELL_CONFIG_SHA256 CELL_PYTHON
mapfile -t CONFIG < <("$CELL_PYTHON" "$SCRIPT_DIR/build.py" config-values \
  --config "$CELL_CONFIG" --expected-config-sha "$CELL_CONFIG_SHA256")
CELL_ROOT=${CONFIG[0]}; SOURCE_ROOT=${CONFIG[1]}
[[ "$CELL_PYTHON" == "${CONFIG[2]}" ]] || { echo "CELL_PYTHON differs from config" >&2; exit 2; }

E2B_FEATURE=""; B12_FEATURE=""
FEATURE_PATTERN='^[A-Za-z0-9_.|&-]+$'
IFS=',' read -ra PAIRS <<<"$CELL_GPU_CONSTRAINT"
[[ ${#PAIRS[@]} -eq 2 ]] || { echo "expected e2b=FEATURE,12b=FEATURE" >&2; exit 2; }
for pair in "${PAIRS[@]}"; do
  key=${pair%%=*}; value=${pair#*=}
  [[ "$value" =~ $FEATURE_PATTERN ]] || { echo "invalid GPU feature" >&2; exit 2; }
  case "$key" in e2b) E2B_FEATURE=$value ;; 12b) B12_FEATURE=$value ;; *) exit 2 ;; esac
done
[[ -n "$E2B_FEATURE" && -n "$B12_FEATURE" ]] || { echo "both GPU features are required" >&2; exit 2; }

E2B_FAMILY=""; B12_FAMILY=""
FAMILY_PATTERN='^(A100|H100|H200|RTX6000)(\|(A100|H100|H200|RTX6000))*$'
IFS=',' read -ra PAIRS <<<"$CELL_GPU_FAMILY"
[[ ${#PAIRS[@]} -eq 2 ]] || { echo "expected e2b=FAMILY,12b=FAMILY" >&2; exit 2; }
for pair in "${PAIRS[@]}"; do
  key=${pair%%=*}; value=${pair#*=}
  [[ "$value" =~ $FAMILY_PATTERN ]] || { echo "invalid GPU family set" >&2; exit 2; }
  case "$key" in e2b) E2B_FAMILY=$value ;; 12b) B12_FAMILY=$value ;; *) exit 2 ;; esac
done
[[ -n "$E2B_FAMILY" && -n "$B12_FAMILY" ]] || { echo "both GPU families are required" >&2; exit 2; }

IFS=',' read -ra SELECTED_ARMS <<<"$CELL_SELECTED_ARMS"
[[ ${#SELECTED_ARMS[@]} -ge 1 ]] || { echo "at least one selected arm is required" >&2; exit 2; }
declare -A SEEN_ARMS=()
for arm in "${SELECTED_ARMS[@]}"; do
  [[ "$arm" =~ ^[ABCDE]$ ]] || { echo "invalid selected arm: $arm" >&2; exit 2; }
  [[ -z "${SEEN_ARMS[$arm]:-}" ]] || { echo "duplicate selected arm: $arm" >&2; exit 2; }
  SEEN_ARMS[$arm]=1
done
mkdir -p "$CELL_ROOT/logs"
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-begin --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" \
  --gpu-constraints "$CELL_GPU_CONSTRAINT;families=$CELL_GPU_FAMILY" \
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
common=(--parsable --account=torch_pr_627_general --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=96G --requeue --comment="preemption=yes;requeue=true" --open-mode=append \
  --chdir="$SOURCE_ROOT")
case "${PATH:-}" in
  *','*|*$'\n'*|*$'\r'*|*$'\t'*) echo "PATH is not export-safe" >&2; exit 2 ;;
esac
export_args="CELL_CONFIG=$CELL_CONFIG,CELL_CONFIG_SHA256=$CELL_CONFIG_SHA256,CELL_PYTHON=$CELL_PYTHON,PATH=${PATH:-/usr/bin:/bin}"

BUILD=$(job_id "$(sbatch "${common[@]}" --job-name=occ-cell-build --time=02:00:00 \
  --output="$CELL_ROOT/logs/build-%j.out" --error="$CELL_ROOT/logs/build-%j.err" \
  --export="$export_args" "$SCRIPT_DIR/build.sh")")
SUBMITTED_JOBS+=("$BUILD")
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-job --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256" --role build --job-id "$BUILD"

declare -A ARM_JOBS
for arm in "${SELECTED_ARMS[@]}"; do
  case "$arm" in
    A|C) feature=$E2B_FEATURE; family=$E2B_FAMILY; walltime=12:00:00 ;;
    B|D) feature=$B12_FEATURE; family=$B12_FAMILY; walltime=12:00:00 ;;
    E) feature=$B12_FEATURE; family=$B12_FAMILY; walltime=18:00:00 ;;
    *) exit 2 ;;
  esac
  ARM_JOBS[$arm]=$(job_id "$(sbatch "${common[@]}" --job-name="occ-cell-$arm" \
    --dependency="afterok:$BUILD" --time="$walltime" --gres=gpu:1 --constraint="$feature" \
    --output="$CELL_ROOT/logs/arm-$arm-%j.out" --error="$CELL_ROOT/logs/arm-$arm-%j.err" \
    --export="$export_args,CELL_ARM=$arm,CELL_EXPECTED_GPU_FEATURE=$feature,CELL_EXPECTED_GPU_FAMILY=$family" \
    "$SCRIPT_DIR/run_arm.sh")")
  SUBMITTED_JOBS+=("${ARM_JOBS[$arm]}")
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-job --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --role arm --arm "$arm" --job-id "${ARM_JOBS[$arm]}"
done

for arm in "${SELECTED_ARMS[@]}"; do
  audit=$(job_id "$(sbatch "${common[@]}" --job-name="occ-audit-$arm" \
    --dependency="afterok:${ARM_JOBS[$arm]}" --time=02:00:00 \
    --output="$CELL_ROOT/logs/audit-$arm-%j.out" --error="$CELL_ROOT/logs/audit-$arm-%j.err" \
    --export="$export_args,CELL_ARM=$arm,CELL_PARENT_JOB_ID=${ARM_JOBS[$arm]}" \
    "$SCRIPT_DIR/audit.sh")")
  SUBMITTED_JOBS+=("$audit")
  "$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-job --config "$CELL_CONFIG" \
    --expected-config-sha "$CELL_CONFIG_SHA256" --role audit --arm "$arm" --job-id "$audit"
done
"$CELL_PYTHON" "$SCRIPT_DIR/build.py" submission-finish --config "$CELL_CONFIG" \
  --expected-config-sha "$CELL_CONFIG_SHA256"
SUBMIT_FINISHED=1
trap - EXIT
echo "Submitted occupancy screening cell; receipt: $CELL_ROOT/submission.json"
