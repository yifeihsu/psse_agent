#!/usr/bin/env bash
# Submit the cloning cell's jobs (2026-11-06).  Run on torch, or send it through ssh stdin:
#
#   wsl -- ssh torch bash -s -- WHAT [SUITE ...] < research/hpc/compound_cloning_20261106/submit_cell.sh
#
# WHAT is one of
#   smoke       the training smoke, and an evaluation smoke of the start adapter on a slice of each suite
#   smoke-eval  the evaluation smoke alone (needs no rows)
#   train       the cloning pass as a chain of 12-hour jobs
#   before      the start adapter on each suite; waits for nothing
#   after       the cloned adapter on each suite, behind the chain's last link unless its receipt exists
#   pass        train, then after
#
# SUITE names lines of $DATA/eval_suites.env (default: all of them; every named file must be there).
# Every job is output-guarded, so resubmitting redoes only what is missing.  Links of a chain follow
# each other with afterany (a link that reaches its time limit must not end the chain; a link with
# nothing left exits at once); the dependents of a chain use afterok on its last link.  The number
# of links comes from the row count (train) and from the list (evaluation); TRAIN_LINKS and
# EVAL_LINKS override them.  ALLOW_DUPLICATE=1 lifts the refusal of a name already queued.
set -euo pipefail
WORK=/scratch/yx3882/compound_cloning_20261106
# shellcheck source=cell.env
source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"
export PATH=/opt/slurm/bin:$PATH
WHAT=${1:?usage: submit_cell.sh smoke|smoke-eval|train|before|after|pass [SUITE ...]}
shift
SUITES=("$@")

declare -A SUITE_FILE=() SUITE_SEED=() SUITE_LINKS=()
LISTED=()
eval_suite() {
  SUITE_FILE[$1]=$2
  SUITE_SEED[$1]=$3
  SUITE_LINKS[$1]=$4
  LISTED+=("$1")
}
# shellcheck source=eval_suites.env
source "$DATA/eval_suites.env"
if (( ${#SUITES[@]} == 0 )); then SUITES=("${LISTED[@]}"); fi

job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }

refuse_queued() {
  [[ -z "${ALLOW_DUPLICATE:-}" ]] || return 0
  local queued
  queued=$(squeue -h -u "$USER" -n "$1" -o %i || true)
  if [[ -n "$queued" ]]; then
    echo "jobs named $1 are already queued or running ($queued); ALLOW_DUPLICATE=1 lifts this" >&2
    exit 2
  fi
}

SUBMITTED=()
LAST_JOB=
# submit_chain LINKS NAME AFTER sbatch-arguments...: sets LAST_JOB to the last link.
submit_chain() {
  local links=$1 name=$2 previous=$3 link
  shift 3
  refuse_queued "$name"
  for (( link = 1; link <= links; link++ )); do
    local dependency=()
    if [[ -n "$previous" ]]; then
      if (( link == 1 )); then dependency=(--dependency="afterok:$previous"); else dependency=(--dependency="afterany:$previous"); fi
    fi
    previous=$(job_id "$(sbatch --parsable "${dependency[@]}" -J "$name" "$@")")
  done
  LAST_JOB=$previous
  SUBMITTED+=("$name=$LAST_JOB($links)")
}

SPEC=
# write_spec NAME SUITE SEED POLICY [SLICE_FROM]
write_spec() {
  mkdir -p "$OUT/specs"
  SPEC=$OUT/specs/$1.env
  printf 'EVAL_NAME=%s\nEVAL_SUITE=%s\nEVAL_SEED=%s\nEVAL_POLICIES="%s"\nEVAL_SLICE_FROM=%s\n' "$1" "$2" "$3" "$4" "${5:-}" > "$SPEC"
}

check_suites() {
  local name problems=0
  for name in "${SUITES[@]}"; do
    if [[ -z "${SUITE_FILE[$name]:-}" ]]; then
      echo "unknown suite $name; the list has: ${LISTED[*]}" >&2
      problems=1
    elif [[ ! -s "${SUITE_FILE[$name]}" ]]; then
      echo "suite $name has no file ${SUITE_FILE[$name]}: upload it (deploy_from_windows.sh, UPLOAD_SUITES) or leave it out by naming the others" >&2
      problems=1
    fi
  done
  (( problems == 0 )) || exit 2
}

check_rows() { [[ -s "$TRAIN_VIEW" && -s "$VALIDATION_VIEW" ]] || { echo "rows missing under $DATA/rows: deploy them first" >&2; exit 2; }; }

# One optimizer step per four rows; 40 s per step covers the 34 s measured on an A100 and the
# validation passes, 11 h of each 12 h link are counted, plus one spare link.
train_links() {
  local rows steps
  rows=$(grep -c . "$TRAIN_VIEW")
  steps=$(( (rows + 3) / 4 ))
  echo "train view: $rows rows, $steps optimizer steps" >&2
  echo $(( (steps * 40 + 39599) / 39600 + 1 ))
}

submit_train() {
  check_rows
  mkdir -p "$OUT/clone"
  submit_chain "${TRAIN_LINKS:-$(train_links)}" cc-train "" --export=ALL,CLONE_MODE=full "$CELL/stage_clone_train.sbatch"
  printf '%s\n' "$LAST_JOB" > "$OUT/clone/last_link.txt"
}

# submit_arm ARM ADAPTER AFTER
submit_arm() {
  local arm=$1 adapter=$2 after=$3 name eval_name
  for name in "${SUITES[@]}"; do
    eval_name=${name}_$arm
    write_spec "$eval_name" "${SUITE_FILE[$name]}" "${SUITE_SEED[$name]}" "$arm=$adapter"
    submit_chain "${EVAL_LINKS:-${SUITE_LINKS[$name]}}" "cc-$eval_name" "$after" --export=ALL,EVAL_SPEC="$SPEC" "$CELL/stage_eval.sbatch"
  done
}

submit_after() {
  local after=""
  if [[ ! -s "$OUT/clone/training.done" ]]; then
    [[ -s "$OUT/clone/last_link.txt" ]] || { echo "no training chain submitted and no receipt: submit train first" >&2; exit 2; }
    after=$(<"$OUT/clone/last_link.txt")
  fi
  submit_arm after "$CLONED_ADAPTER" "$after"
}

submit_eval_smoke() {
  local name eval_name
  for name in "${SUITES[@]}"; do
    eval_name=smoke_$name
    write_spec "$eval_name" "$OUT/smoke/suites/$name.json" "${SUITE_SEED[$name]}" "before=$START_ADAPTER" "${SUITE_FILE[$name]}"
    submit_chain 1 "cc-$eval_name" "" --time=02:00:00 --export=ALL,EVAL_SPEC="$SPEC" "$CELL/stage_eval.sbatch"
  done
}

case "$WHAT" in
  smoke|smoke-eval|before|after|pass) check_suites ;;
esac
case "$WHAT" in
  smoke|smoke-eval|before) [[ -s "$START_ADAPTER/adapter_model.safetensors" ]] || { echo "start adapter missing: deploy first" >&2; exit 2; } ;;
esac
mkdir -p "$OUT/smoke/suites" "$LOGS"
case "$WHAT" in
  smoke)
    check_rows
    submit_chain 1 cc-smoke-train "" --time=02:00:00 --export=ALL,CLONE_MODE=smoke "$CELL/stage_clone_train.sbatch"
    submit_eval_smoke
    ;;
  smoke-eval) submit_eval_smoke ;;
  train) submit_train ;;
  before) submit_arm before "$START_ADAPTER" "" ;;
  after) submit_after ;;
  pass) submit_train; submit_after ;;
  *) echo "unknown request: $WHAT" >&2; exit 2 ;;
esac
echo "$WHAT ${SUBMITTED[*]} submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"
squeue -u "$USER" -o "%.12i %.28j %.8T %.10M %.24R"
