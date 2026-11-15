#!/usr/bin/env bash
# Submit the jobs of the minimal environment's supervised pass.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- smoke [START]              < research/hpc/minimal_sft_20261115/submit_cell.sh
#   wsl -- ssh torch bash -s -- pass [START] [LINKS]       < ...   the row gate on a CPU node, then the pass (links from the rows)
#   wsl -- ssh torch bash -s -- eval [START] [LINKS] [AFTER_JOB] < ...   the pass's adapter on the 300 development cases (default 2 links)
#   wsl -- ssh torch bash -s -- all [START]                < ...   the row gate, the pass, its evaluation
#
# START is where the pass starts, base or r1 (cell.env; default base).  The two starts keep their output apart, so
# both can be submitted.
#
# smoke submits the training smoke (two optimizer steps on about 200 rows, the longest first, after timing the row
# gate on them) and the evaluation smoke (R1 on eight development cases in the minimal environment); they are
# independent.  Why chains: Slurm requeues a preempted job, not one that reaches its time limit.  Each link follows
# the one before it (afterany), resumes from what is on disk, and exits at once when nothing is left; a chain's
# dependents wait for its last link to end well (afterok).  Nothing here pins a card type or a partition.
# ALLOW_DUPLICATE=1 lifts the refusal of a job name already queued.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_sft_20261115
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_sft_20261115/cell.env"
COMMAND=${1:?smoke, pass, eval or all}
SFT_START=${2:-$SFT_START_DEFAULT}
resolve_start
[[ -d "$CELL" ]] || { echo "no source at $CELL: deploy first" >&2; exit 2; }
mkdir -p "$LOGS" "$OUT/pass_$START_NAME"

positive() { [[ "$1" =~ ^[0-9]+$ && "$1" -ge 1 ]] || { echo "$2 must be a positive integer, got $1" >&2; exit 2; }; }
not_queued() {
  [[ -n "${ALLOW_DUPLICATE:-}" ]] && return 0
  local queued
  queued=$(squeue -h -u "$USER" -n "$1" -o %i || true)
  [[ -z "$queued" ]] || { echo "jobs named $1 are already queued or running ($queued); ALLOW_DUPLICATE=1 lifts this" >&2; exit 2; }
}
record() { echo "$1 source=$(git -C "$SRC" rev-parse --short=7 HEAD) submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"; }
check_rows() { [[ -s "$TRAIN_VIEW" && -s "$VALIDATION_VIEW" ]] || { echo "rows missing under $DATA/rows: deploy them first" >&2; exit 2; }; }

# chain NAME LINKS SCRIPT FIRST_DEPENDENCY EXPORTS: prints the links' ids, the last one last.
chain() {
  local name=$1 links=$2 script=$3 first=$4 exports=$5 previous= link ids=()
  for (( link = 1; link <= links; link++ )); do
    local dependency=()
    if [[ -n "$previous" ]]; then dependency=(--dependency="afterany:$previous"); elif [[ -n "$first" ]]; then dependency=(--dependency="$first"); fi
    previous=$(sbatch --parsable "${dependency[@]}" --job-name="$name" --export="$exports" "$CELL/$script")
    previous=${previous%%;*}
    [[ "$previous" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id for $name: $previous" >&2; exit 2; }
    ids+=("$previous")
  done
  echo "${ids[*]}"
}

# One optimizer step per four rows.  The open-rules pass measured 25 to 27 s per step on an RTX PRO 6000 for rows
# of about 5,800 tokens; these rows are about 9,900 with their tools (a sibling tokenizer's count), so 60 s per
# step is counted with the validation passes, 11 h of each 12-hour link, plus one spare link.
train_links() {
  local rows steps
  rows=$(grep -c . "$TRAIN_VIEW")
  steps=$(( (rows + 3) / 4 ))
  echo "train view: $rows rows, $steps optimizer steps" >&2
  echo $(( (steps * 60 + 39599) / 39600 + 1 ))
}

PASS_LAST=
submit_pass() {
  check_rows
  local links=${1:-}
  [[ -n "$links" ]] || links=$(train_links)
  positive "$links" LINKS
  not_queued "msf-train-$START_NAME"
  local gate= first= ids
  if [[ -s "$OUT/row_gate.json" ]]; then
    echo "the row gate has passed ($OUT/row_gate.json); the pass starts without it"
  else
    # The row gate first, on a CPU node: the pass starts only if every row passes, and every start of a link then
    # loads the gate's result.  A gate another start's pass has queued is the one to wait for.
    gate=$(squeue -h -u "$USER" -n msf-row-gate -o %i | head -n 1 || true)
    if [[ -z "$gate" ]]; then
      gate=$(chain msf-row-gate 1 stage_row_gate.sbatch "" ALL)
      record "msf-row-gate job=$gate"
    fi
    first="afterok:$gate"
  fi
  ids=$(chain "msf-train-$START_NAME" "$links" stage_train.sbatch "$first" "ALL,SFT_MODE=full,SFT_START=$START_NAME")
  record "msf-train-$START_NAME links=$ids${gate:+ after=$gate}"
  PASS_LAST=${ids##* }
  printf '%s\n' "$PASS_LAST" > "$OUT/pass_$START_NAME/last_link.txt"
}

submit_eval() {
  local links=${1:-2} after=${2:-}
  positive "$links" LINKS
  not_queued "msf-eval-dev300-$START_NAME"
  if [[ -z "$after" && ! -s "$OUT/pass_$START_NAME/training.done" ]]; then
    [[ -s "$OUT/pass_$START_NAME/last_link.txt" ]] || { echo "no training receipt and no pass submitted for $START_NAME: name the pass's last job as AFTER_JOB, or submit pass first" >&2; exit 2; }
    after=$(<"$OUT/pass_$START_NAME/last_link.txt")
  fi
  local ids
  ids=$(chain "msf-eval-dev300-$START_NAME" "$links" stage_eval.sbatch "${after:+afterok:$after}" "ALL,EVAL_NAME=dev300,SFT_START=$START_NAME")
  record "msf-eval-dev300-$START_NAME links=$ids${after:+ after=$after}"
}

case "$COMMAND" in
  smoke)
    check_rows
    not_queued "msf-smoke-$START_NAME"
    not_queued msf-eval-smoke
    id=$(chain "msf-smoke-$START_NAME" 1 stage_train.sbatch "" "ALL,SFT_MODE=smoke,SFT_START=$START_NAME")
    record "msf-smoke-$START_NAME job=$id"
    if [[ -s "$OUT/eval_smoke/evaluation.json" ]]; then
      echo "the evaluation smoke is done ($OUT/eval_smoke/evaluation.json); not submitted again"
    else
      id=$(sbatch --parsable --time=03:00:00 --job-name=msf-eval-smoke --export=ALL,EVAL_NAME=smoke "$CELL/stage_eval.sbatch")
      record "msf-eval-smoke job=${id%%;*}"
    fi
    ;;
  pass) submit_pass "${3:-}" ;;
  eval) submit_eval "${3:-2}" "${4:-}" ;;
  all)
    submit_pass ""
    submit_eval 2 "$PASS_LAST"
    ;;
  *) echo "unknown command $COMMAND (smoke, pass, eval or all)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
