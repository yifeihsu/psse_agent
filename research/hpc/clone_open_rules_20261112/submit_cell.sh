#!/usr/bin/env bash
# Submit the open-rules clone's jobs.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- smoke                 < research/hpc/clone_open_rules_20261112/submit_cell.sh
#   wsl -- ssh torch bash -s -- smoke_baseline        < ...   the baseline smoke alone, again (an earlier one is moved aside)
#   wsl -- ssh torch bash -s -- s0 [LINKS]            < ...   S0's baseline in the open environment (default 2 links)
#   wsl -- ssh torch bash -s -- pass [LINKS]          < ...   the row gate on a CPU node, then the pass (links from the rows)
#   wsl -- ssh torch bash -s -- clone [LINKS] [AFTER_JOB] < ...   the cloned adapter's baseline (default 2 links)
#   wsl -- ssh torch bash -s -- all                   < ...   S0's baseline; and the row gate, the pass, the clone's baseline
#   wsl -- ssh torch bash -s -- arm [LINKS] [TARGET]  < ...   the RL arm from the cloned adapter in the open environment
#                                                            (default 6 links, target 2 iterations)
#
# smoke submits the training smoke (two optimizer steps on about 200 rows, the longest first, after timing the row
# gate on them) and the baseline smoke (S0 on eight development cases in the open environment); they are independent.
# Why chains: Slurm requeues a preempted job, not one that reaches its time limit.  Each link follows the one before
# it (afterany), resumes from what is on disk, and exits at once when nothing is left; a chain's dependents wait for
# its last link to end well (afterok).  Nothing here pins a card type or a partition.  ALLOW_DUPLICATE=1 lifts the
# refusal of a job name already queued.
#
# arm continues the run the clone's baseline began (configs/baseline_clone_open.json: arm A's config in the open
# environment from the cloned adapter), so it needs that baseline closed.  TARGET is the manifest iteration at which
# the run is complete, the series' stopping rule in practice: 2 for the screen, a third iteration a new submission
# made only to resolve a concrete uncertain signal, more than three only with ALLOW_LONG=1.  Six links cover two
# iterations of about 14 GPU-hours each with their preemptions; spare links exit at once.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/clone_open_rules_20261112
# shellcheck source=cell.env
source "$WORK/source/research/hpc/clone_open_rules_20261112/cell.env"
COMMAND=${1:?smoke, smoke_baseline, s0, pass, clone, all or arm}
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

# One optimizer step per four rows; 40 s per step covers the 34 s the bounded pass measured on an A100 and the
# validation passes, 11 h of each 12-hour link are counted, plus one spare link (as the bounded cell counts them).
train_links() {
  local rows steps
  rows=$(grep -c . "$TRAIN_VIEW")
  steps=$(( (rows + 3) / 4 ))
  echo "train view: $rows rows, $steps optimizer steps" >&2
  echo $(( (steps * 40 + 39599) / 39600 + 1 ))
}

submit_s0() {
  local links=${1:-2}
  positive "$links" LINKS
  not_queued orc-baseline-s0_open
  local ids
  ids=$(chain orc-baseline-s0_open "$links" stage_baseline.sbatch "" ALL,BASELINE_NAME=s0_open)
  record "orc-baseline-s0_open links=$ids"
}

PASS_LAST=
submit_pass() {
  check_rows
  local links=${1:-}
  [[ -n "$links" ]] || links=$(train_links)
  positive "$links" LINKS
  not_queued orc-row-gate
  not_queued orc-train
  local gate ids
  # The row gate first, on a CPU node: the pass starts only if every row passes, and every start of a link then
  # loads the gate's result.
  gate=$(chain orc-row-gate 1 stage_row_gate.sbatch "" ALL)
  record "orc-row-gate job=$gate"
  ids=$(chain orc-train "$links" stage_clone_train.sbatch "afterok:$gate" ALL,CLONE_MODE=full)
  record "orc-train links=$ids after=$gate"
  PASS_LAST=${ids##* }
  printf '%s\n' "$PASS_LAST" > "$OUT/clone/last_link.txt"
}

submit_clone() {
  local links=${1:-2} after=${2:-}
  positive "$links" LINKS
  not_queued orc-baseline-clone_open
  if [[ -z "$after" && ! -s "$OUT/clone/training.done" ]]; then
    [[ -s "$OUT/clone/last_link.txt" ]] || { echo "no training receipt and no pass submitted: name the pass's last job as AFTER_JOB, or submit pass first" >&2; exit 2; }
    after=$(<"$OUT/clone/last_link.txt")
  fi
  local ids
  ids=$(chain orc-baseline-clone_open "$links" stage_baseline.sbatch "${after:+afterok:$after}" ALL,BASELINE_NAME=clone_open)
  record "orc-baseline-clone_open links=$ids${after:+ after=$after}"
}

ARM_CONFIG=baseline_clone_open.json
submit_arm() {
  local links=${1:-6} target=${2:-2}
  positive "$links" LINKS
  positive "$target" TARGET
  if (( target > 3 )) && [[ -z "${ALLOW_LONG:-}" ]]; then
    echo "the series' stopping rule ends an arm after three iterations without a paired gain; ALLOW_LONG=1 lifts this for a promoted arm" >&2
    exit 2
  fi
  not_queued orc-arm
  not_queued orc-baseline-clone_open
  local run=$OUT/${ARM_CONFIG%.json}
  [[ -s "$CONFIG_DIR/$ARM_CONFIG" ]] || { echo "no config $CONFIG_DIR/$ARM_CONFIG: deploy it first" >&2; exit 2; }
  [[ -s "$OUT/clone/training.done" ]] || { echo "no training receipt $OUT/clone/training.done: the pass has not finished" >&2; exit 2; }
  [[ -s "$run/baseline/evaluation.json" ]] || { echo "the clone's baseline has not closed ($run/baseline/evaluation.json): submit clone first" >&2; exit 2; }
  # The update must label the choice on a rejected candidate, which the commit rule's lifting gives the policy.
  grep -q "def rejected_choice" "$SRC/psse_env/rl/train.py" \
    || { echo "the source at $SRC does not label the choice on rejected candidates: deploy the arm's commit first" >&2; exit 2; }
  [[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]] || { echo "the source at $SRC has uncommitted changes" >&2; exit 2; }
  local ids
  ids=$(chain orc-arm "$links" stage_arm.sbatch "" "ALL,ARM_CONFIG=$ARM_CONFIG,ITERATIONS=$target,TARGET_ITERATIONS=$target")
  record "orc-arm config=$ARM_CONFIG links=$ids target_iterations=$target"
}

case "$COMMAND" in
  smoke)
    check_rows
    not_queued orc-smoke
    not_queued orc-baseline-smoke_open
    id=$(chain orc-smoke 1 stage_clone_train.sbatch "" ALL,CLONE_MODE=smoke)
    record "orc-smoke job=$id"
    id=$(sbatch --parsable --time=03:00:00 --job-name=orc-baseline-smoke_open --export=ALL,BASELINE_NAME=smoke_open "$CELL/stage_baseline.sbatch")
    record "orc-baseline-smoke_open job=${id%%;*}"
    ;;
  smoke_baseline)
    # The baseline smoke again, on the same eight cases (data/suites keeps the slice): a finished one has left its
    # manifest past the baseline, so its output is renamed with the time, never removed.
    not_queued orc-baseline-smoke_open
    if [[ -e "$OUT/baseline_smoke_open" ]]; then
      aside="$OUT/baseline_smoke_open.before_$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$OUT/baseline_smoke_open" "$aside"
      echo "moved the earlier baseline smoke to $aside"
    fi
    id=$(sbatch --parsable --time=03:00:00 --job-name=orc-baseline-smoke_open --export=ALL,BASELINE_NAME=smoke_open "$CELL/stage_baseline.sbatch")
    record "orc-baseline-smoke_open job=${id%%;*}"
    ;;
  s0) submit_s0 "${2:-2}" ;;
  pass) submit_pass "${2:-}" ;;
  clone) submit_clone "${2:-2}" "${3:-}" ;;
  all)
    submit_s0 2
    submit_pass ""
    submit_clone 2 "$PASS_LAST"
    ;;
  arm) submit_arm "${2:-6}" "${3:-2}" ;;
  *) echo "unknown command $COMMAND (smoke, smoke_baseline, s0, pass, clone, all or arm)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
