#!/usr/bin/env bash
# Submit the jobs of the minimal environment's RL run.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- smoke                  < research/hpc/minimal_rl_20261116/submit_cell.sh
#   wsl -- ssh torch bash -s -- run ITERATIONS [LINKS] < ...
#
# smoke submits one job: one iteration on six training cases and eight development cases, from the cell's copy of
# the supervised pass's adapter or of its newest complete checkpoint (the pass may still be running).
#
# run submits a chain that takes the run to ITERATIONS iterations in all, from wherever its manifest stands: a run
# at iteration 3 goes on to 5 with "run 5".  It refuses to start before the supervised pass has its receipt.  Why a
# chain: Slurm requeues a preempted job, not one that reaches its time limit.  Each link follows the one before it
# (afterany), goes on from the manifest, and exits at once when the target is reached.  Nothing here pins a card
# type or a partition.  ALLOW_DUPLICATE=1 lifts the refusal of a job name already queued.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_rl_20261116
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_rl_20261116/cell.env"
COMMAND=${1:?smoke or run}
[[ -d "$CELL" ]] || { echo "no source at $CELL: deploy first" >&2; exit 2; }
mkdir -p "$LOGS" "$OUT" "$DATA"

positive() { [[ "$1" =~ ^[0-9]+$ && "$1" -ge 1 ]] || { echo "$2 must be a positive integer, got $1" >&2; exit 2; }; }
not_queued() {
  [[ -n "${ALLOW_DUPLICATE:-}" ]] && return 0
  local queued
  queued=$(squeue -h -u "$USER" -n "$1" -o %i || true)
  [[ -z "$queued" ]] || { echo "jobs named $1 are already queued or running ($queued); ALLOW_DUPLICATE=1 lifts this" >&2; exit 2; }
}
record() { echo "$1 source=$(git -C "$SRC" rev-parse --short=7 HEAD) submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"; }

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

# Links for the iterations still to run.  An iteration of 64 cases by 4 samples is counted at 14 hours (the upper
# end of the estimate in README.md; nothing of it is measured yet), 11 hours of each 12-hour link, plus one spare.
run_links() { echo $(( ($1 * 14 + 10) / 11 + 1 )); }

case "$COMMAND" in
  smoke)
    not_queued mrl-smoke
    [[ -s "$SMOKE_ADAPTER/adapter_model.safetensors" || -s "$PASS/training.done" || -n "$(newest_complete_checkpoint)" ]] \
      || { echo "the supervised pass has neither an adapter nor a complete checkpoint under $PASS" >&2; exit 2; }
    id=$(sbatch --parsable --time=04:00:00 --job-name=mrl-smoke --export=ALL,RL_RUN=smoke,RL_TARGET=1 "$CELL/stage_rl.sbatch")
    record "mrl-smoke job=${id%%;*}"
    ;;
  run)
    target=${2:?how many iterations the run should reach in all}
    positive "$target" ITERATIONS
    [[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
    at=0
    [[ -s "$OUT/ieee14/manifest.json" ]] && at=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["iteration"]))' "$OUT/ieee14/manifest.json")
    (( target > at )) || { echo "the run is at iteration $at already: name a target above it" >&2; exit 2; }
    links=${3:-$(run_links $(( target - at )))}
    positive "$links" LINKS
    not_queued mrl-run
    ids=$(chain mrl-run "$links" "ALL,RL_RUN=ieee14,RL_TARGET=$target")
    record "mrl-run from=$at target=$target links=$ids"
    ;;
  *) echo "unknown command $COMMAND (smoke or run)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
