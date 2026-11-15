#!/usr/bin/env bash
# Submit the complete-evidence clone's jobs.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- smoke                 < research/hpc/clone_complete_evidence_20261115/submit_cell.sh
#   wsl -- ssh torch bash -s -- smoke_train           < ...   the training smoke alone, again (it resumes what the earlier one left)
#   wsl -- ssh torch bash -s -- smoke_baseline        < ...   the baseline smoke alone, again (an earlier one is moved aside)
#   wsl -- ssh torch bash -s -- pass [LINKS]          < ...   the row gate on a CPU node, then the pass (links from the rows)
#   wsl -- ssh torch bash -s -- clone [LINKS] [AFTER_JOB] < ...   the cloned adapter's baseline (default 2 links)
#   wsl -- ssh torch bash -s -- all                   < ...   the row gate, the pass, the clone's baseline
#
# smoke submits the training smoke (two optimizer steps on about 200 rows, the longest first, after timing the row
# gate on them) and the baseline smoke (S0 on eight development cases in this environment); they are independent.
# Why chains: Slurm requeues a preempted job, not one that reaches its time limit.  Each link follows the one before
# it (afterany), resumes from what is on disk, and exits at once when nothing is left; a chain's dependents wait for
# its last link to end well (afterok).  Nothing here pins a card type or a partition.  ALLOW_DUPLICATE=1 lifts the
# refusal of a job name already queued.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/clone_complete_evidence_20261115
# shellcheck source=cell.env
source "$WORK/source/research/hpc/clone_complete_evidence_20261115/cell.env"
COMMAND=${1:?smoke, smoke_train, smoke_baseline, pass, clone or all}
[[ -d "$CELL" ]] || { echo "no source at $CELL: deploy first" >&2; exit 2; }
mkdir -p "$LOGS" "$OUT/clone"

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

# One optimizer step per four rows.  The bounded pass measured 34 s per step on an A100 for rows of about 6,000
# tokens; these rows are about half as long again (a median near 9,100 by a sibling tokenizer), so 60 s per step is
# counted with the validation passes, 11 h of each 12-hour link, plus one spare link (as the bounded cell counts them).
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
  not_queued cec-row-gate
  not_queued cec-train
  local gate ids
  # The row gate first, on a CPU node: the pass starts only if every row passes, and every start of a link then
  # loads the gate's result.
  gate=$(chain cec-row-gate 1 stage_row_gate.sbatch "" ALL)
  record "cec-row-gate job=$gate"
  ids=$(chain cec-train "$links" stage_clone_train.sbatch "afterok:$gate" ALL,CLONE_MODE=full)
  record "cec-train links=$ids after=$gate"
  PASS_LAST=${ids##* }
  printf '%s\n' "$PASS_LAST" > "$OUT/clone/last_link.txt"
}

submit_clone() {
  local links=${1:-2} after=${2:-}
  positive "$links" LINKS
  not_queued cec-baseline-clone_complete
  if [[ -z "$after" && ! -s "$OUT/clone/training.done" ]]; then
    [[ -s "$OUT/clone/last_link.txt" ]] || { echo "no training receipt and no pass submitted: name the pass's last job as AFTER_JOB, or submit pass first" >&2; exit 2; }
    after=$(<"$OUT/clone/last_link.txt")
  fi
  local ids
  ids=$(chain cec-baseline-clone_complete "$links" stage_baseline.sbatch "${after:+afterok:$after}" ALL,BASELINE_NAME=clone_complete)
  record "cec-baseline-clone_complete links=$ids${after:+ after=$after}"
}

case "$COMMAND" in
  smoke)
    check_rows
    not_queued cec-smoke
    not_queued cec-baseline-smoke_complete
    id=$(chain cec-smoke 1 stage_clone_train.sbatch "" ALL,CLONE_MODE=smoke)
    record "cec-smoke job=$id"
    id=$(sbatch --parsable --time=03:00:00 --job-name=cec-baseline-smoke_complete --export=ALL,BASELINE_NAME=smoke_complete "$CELL/stage_baseline.sbatch")
    record "cec-baseline-smoke_complete job=${id%%;*}"
    ;;
  smoke_train)
    # The training smoke again: a finished one (a receipt) is renamed with the time, never removed; an unfinished one
    # keeps its rows and its row gate result, and the stage goes on from them.
    check_rows
    not_queued cec-smoke
    if [[ -s "$OUT/smoke/training.done" ]]; then
      aside="$OUT/smoke.before_$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$OUT/smoke" "$aside"
      echo "moved the earlier training smoke to $aside"
    fi
    id=$(chain cec-smoke 1 stage_clone_train.sbatch "" ALL,CLONE_MODE=smoke)
    record "cec-smoke job=$id"
    ;;
  smoke_baseline)
    # The baseline smoke again, on the same eight cases (data/suites keeps the slice): a finished one has left its
    # manifest past the baseline, so its output is renamed with the time, never removed.
    not_queued cec-baseline-smoke_complete
    if [[ -e "$OUT/baseline_smoke_complete" ]]; then
      aside="$OUT/baseline_smoke_complete.before_$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$OUT/baseline_smoke_complete" "$aside"
      echo "moved the earlier baseline smoke to $aside"
    fi
    id=$(sbatch --parsable --time=03:00:00 --job-name=cec-baseline-smoke_complete --export=ALL,BASELINE_NAME=smoke_complete "$CELL/stage_baseline.sbatch")
    record "cec-baseline-smoke_complete job=${id%%;*}"
    ;;
  pass) submit_pass "${2:-}" ;;
  clone) submit_clone "${2:-2}" "${3:-}" ;;
  all)
    submit_pass ""
    submit_clone 2 "$PASS_LAST"
    ;;
  *) echo "unknown command $COMMAND (smoke, smoke_train, smoke_baseline, pass, clone or all)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
