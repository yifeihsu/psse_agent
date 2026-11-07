#!/usr/bin/env bash
# Submit an RL arm as a chain of linked 12-hour jobs (2026-11-07).  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- CONFIG LINKS TARGET_ITERATIONS [JOB_NAME] < research/hpc/compound_rl_20261107/submit_arm.sh
#   e.g.  ... -- rl_ieee14.json 7 6 crl-ieee14
#
# Why a chain: one arm iteration is about 13 GPU-hours on an RTX PRO 6000 (the smoke of 2026-11-07: about
# 7.3 s per update row, about 3,450 rows per iteration, plus the rollouts and the 300-case evaluation), and
# the baseline phase about 3 hours once, so the 12-hour limit runs out inside the first job.  Slurm requeues
# a preempted job, not one that reaches its time limit; a link that follows the one before it (afterany)
# resumes at the manifest's phase, the rollouts and evaluations per root and the update at its checkpoint.
# Every link runs rl.sbatch with TARGET_ITERATIONS, so links after the target exit at once.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/compound_rl_20261107
CONFIG=${1:?config file name under $WORK/configs}
LINKS=${2:?number of 12-hour links}
TARGET=${3:?target iteration count (the manifest iteration at which the run is complete)}
NAME=${4:-crl-${CONFIG%.json}}
[[ -s "$WORK/configs/$CONFIG" ]] || { echo "no config $WORK/configs/$CONFIG" >&2; exit 2; }
[[ "$LINKS" =~ ^[0-9]+$ && "$LINKS" -ge 1 ]] || { echo "LINKS must be a positive integer" >&2; exit 2; }
[[ "$TARGET" =~ ^[0-9]+$ && "$TARGET" -ge 1 ]] || { echo "TARGET_ITERATIONS must be a positive integer" >&2; exit 2; }
if [[ -z "${ALLOW_DUPLICATE:-}" ]]; then
  queued=$(squeue -h -u "$USER" -n "$NAME" -o %i || true)
  [[ -z "$queued" ]] || { echo "jobs named $NAME are already queued or running ($queued); ALLOW_DUPLICATE=1 lifts this" >&2; exit 2; }
fi
mkdir -p "$WORK/logs"
previous=
ids=()
for (( link = 1; link <= LINKS; link++ )); do
  dependency=()
  [[ -z "$previous" ]] || dependency=(--dependency="afterany:$previous")
  previous=$(sbatch --parsable "${dependency[@]}" --job-name="$NAME" \
    --export=ALL,CONFIG="$CONFIG",ITERATIONS="$TARGET",TARGET_ITERATIONS="$TARGET" \
    "$WORK/source/research/hpc/compound_rl_20261107/rl.sbatch")
  previous=${previous%%;*}
  ids+=("$previous")
done
echo "$NAME config=$CONFIG links=${ids[*]} target_iterations=$TARGET source=$(git -C "$WORK/source" rev-parse --short=7 HEAD) submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"
squeue -u "$USER" -o "%.12i %.24j %.9T %.10M %.26E %.20R"
