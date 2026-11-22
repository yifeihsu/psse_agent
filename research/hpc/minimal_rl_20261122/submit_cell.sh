#!/usr/bin/env bash
# Submit the jobs of the RL plan of 2026-11-22.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- steps-smoke                  < research/hpc/minimal_rl_20261122/submit_cell.sh
#   wsl -- ssh torch bash -s -- steps ITERATIONS [LINKS]     < ...
#   wsl -- ssh torch bash -s -- probe-smoke                  < ...
#   wsl -- ssh torch bash -s -- probe dev|train [PARTS]      < ...
#   wsl -- ssh torch bash -s -- forks-smoke                  < ...
#   wsl -- ssh torch bash -s -- forks ITERATIONS [LINKS]     < ...
#
# steps and forks submit a chain that takes the run to ITERATIONS iterations in all, from wherever its manifest
# stands: a run at iteration 3 goes on to 9 with "steps 9".  Why a chain: Slurm requeues a preempted job, not one
# that reaches its time limit.  Each link follows the one before it (afterany), goes on from the manifest, and
# exits at once when the target is reached.  The two runs have their own folders and may run at the same time.
#
# steps-smoke is two small iterations of stage A, forks-smoke one small iteration of stage B, probe-smoke the
# probe on four development cases: each one job with a four-hour limit.  probe forks the unrepaired development
# cases (dev) or a sample of the unrepaired training cases (train) in PARTS jobs at once (2 for dev and 4 for train
# unless named) and a closing job that writes the summary.
#
# Nothing here pins a card type or a partition.  ALLOW_DUPLICATE=1 lifts the refusal of a job name already queued.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_rl_20261122
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_rl_20261122/cell.env"
COMMAND=${1:?steps-smoke, steps, probe-smoke, probe, forks-smoke or forks}
[[ -d "$CELL" ]] || { echo "no source at $CELL: deploy first" >&2; exit 2; }
mkdir -p "$LOGS" "$OUT"

positive() { [[ "$1" =~ ^[0-9]+$ && "$1" -ge 1 ]] || { echo "$2 must be a positive integer, got $1" >&2; exit 2; }; }
not_queued() {
  [[ -n "${ALLOW_DUPLICATE:-}" ]] && return 0
  local queued
  queued=$(squeue -h -u "$USER" -n "$1" -o %i || true)
  [[ -z "$queued" ]] || { echo "jobs named $1 are already queued or running ($queued); ALLOW_DUPLICATE=1 lifts this" >&2; exit 2; }
}
record() { echo "$1 source=$(git -C "$SRC" rev-parse --short=7 HEAD) submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"; }
# What every job of this cell reads in place.
inputs_ready() {
  [[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
  [[ -s "$TRAIN_PASS/evaluation.json" ]] || { echo "no training pass $TRAIN_PASS/evaluation.json of the supervised model" >&2; exit 2; }
  [[ -s "$DEV_PASS/evaluation.json" ]] || { echo "no evaluation $DEV_PASS/evaluation.json of the supervised model" >&2; exit 2; }
}

# chain NAME LINKS EXPORTS: prints the links' ids, the last one last.
chain() {
  local name=$1 links=$2 exports=$3 previous= link ids=()
  for (( link = 1; link <= links; link++ )); do
    local dependency=()
    [[ -n "$previous" ]] && dependency=(--dependency="afterany:$previous")
    previous=$(sbatch --parsable "${dependency[@]}" --job-name="$name" --export="$exports" "$CELL/stage_rl.sbatch")
    previous=${previous%%;*}
    [[ "$previous" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id for $name: $previous" >&2; exit 2; }
    ids+=("$previous")
  done
  echo "${ids[*]}"
}

# Links for the iterations still to run: HOURS an iteration (an estimate from the focused run's measured rates,
# README.md; nothing of these settings is measured), 11 hours of each 12-hour link, plus one spare.
run_links() { echo $(( ($1 * $2 + 10) / 11 + 1 )); }
STEPS_HOURS=7
FORKS_HOURS=11

# run RUN ITERATIONS [LINKS]: the chain of a run, from wherever its manifest stands.
run() {
  local name=$1 target=$2 links=${3:-} hours=$4 at=0
  positive "$target" ITERATIONS
  inputs_ready
  [[ -s "$OUT/$name/manifest.json" ]] && at=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["iteration"]))' "$OUT/$name/manifest.json")
  (( target > at )) || { echo "the run $name is at iteration $at already: name a target above it" >&2; exit 2; }
  [[ -n "$links" ]] || links=$(run_links $(( target - at )) "$hours")
  positive "$links" LINKS
  not_queued "mrp-$name"
  local ids
  ids=$(chain "mrp-$name" "$links" "ALL,RL_RUN=$name,RL_TARGET=$target")
  record "mrp-$name from=$at target=$target links=$ids"
}

case "$COMMAND" in
  steps-smoke)
    not_queued mrp-steps-smoke
    inputs_ready
    id=$(sbatch --parsable --time=04:00:00 --job-name=mrp-steps-smoke --export=ALL,RL_RUN=steps_smoke,RL_TARGET=2 "$CELL/stage_rl.sbatch")
    record "mrp-steps-smoke job=${id%%;*}"
    ;;
  steps)
    run steps "${2:?how many iterations the run should reach in all}" "${3:-}" "$STEPS_HOURS"
    ;;
  forks-smoke)
    not_queued mrp-forks-smoke
    inputs_ready
    id=$(sbatch --parsable --time=04:00:00 --job-name=mrp-forks-smoke --export=ALL,RL_RUN=forks_smoke,RL_TARGET=1 "$CELL/stage_rl.sbatch")
    record "mrp-forks-smoke job=${id%%;*}"
    ;;
  forks)
    run forks "${2:?how many iterations the run should reach in all}" "${3:-}" "$FORKS_HOURS"
    ;;
  probe-smoke)
    not_queued mrp-probe-smoke
    inputs_ready
    [[ ! -s "$OUT/probe_smoke/summary.json" ]] || { echo "the probe's smoke is done already: $OUT/probe_smoke/summary.json" >&2; exit 2; }
    id=$(sbatch --parsable --time=04:00:00 --job-name=mrp-probe-smoke --export=ALL,PROBE=smoke,PROBE_PART=close "$CELL/stage_fork_probe.sbatch")
    record "mrp-probe-smoke job=${id%%;*}"
    ;;
  probe)
    which=${2:?dev or train}
    case "$which" in
      dev) parts=${3:-2} ;;
      train) parts=${3:-4} ;;
      *) echo "probe takes dev or train, got $which" >&2; exit 2 ;;
    esac
    positive "$parts" PARTS
    inputs_ready
    [[ ! -s "$OUT/probe_$which/summary.json" ]] || { echo "the probe is done already: $OUT/probe_$which/summary.json" >&2; exit 2; }
    not_queued "mrp-probe-$which"
    not_queued "mrp-probe-$which-close"
    ids=()
    for (( part = 0; part < parts; part++ )); do
      id=$(sbatch --parsable --job-name="mrp-probe-$which" --export="ALL,PROBE=$which,PROBE_PART=$part/$parts" "$CELL/stage_fork_probe.sbatch")
      id=${id%%;*}
      [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id for part $part: $id" >&2; exit 2; }
      ids+=("$id")
    done
    after=$(IFS=:; echo "${ids[*]}")
    close=$(sbatch --parsable --dependency="afterany:$after" --job-name="mrp-probe-$which-close" --export="ALL,PROBE=$which,PROBE_PART=close" "$CELL/stage_fork_probe.sbatch")
    record "mrp-probe-$which parts=${ids[*]} close=${close%%;*}"
    ;;
  *) echo "unknown command $COMMAND (steps-smoke, steps, probe-smoke, probe, forks-smoke or forks)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
