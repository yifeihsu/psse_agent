#!/usr/bin/env bash
# Submit one run of the RL improvement series as a chain of linked 12-hour jobs.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- CONFIG LINKS TARGET_ITERATIONS [JOB_NAME] < research/hpc/rl_improvement_20261110/submit_arm.sh
#   e.g.  ... -- arm_a_bounded.json 6 2 rli-a        (six links for the two-iteration screen: spare links exit at once)
#         ... -- calibration_1step.json 2 1 rli-cal1 (an update-only run: prepare it first with calibration.py)
#
# Why a chain: Slurm requeues a preempted job, not one that reaches its time limit, and an iteration of an arm
# takes more than one 12-hour job (the first arm: about 13 GPU-hours per iteration on an RTX PRO 6000, plus about
# 3 hours once for the baseline; the complete history and the continuation phase add to that).  Each link runs
# rl.sbatch with TARGET_ITERATIONS, follows the one before it (afterany) and resumes at the manifest's phase.
#
# TARGET_ITERATIONS is the series' stopping rule in practice: 2 for the screen.  A third iteration is a new
# submission with target 3, made only to resolve a concrete uncertain signal; after three completed iterations
# without a paired gain over the arm's own baseline the arm is not extended (series.json, protocol.stop_rule).
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/rl_improvement_20261110
CONFIG=${1:?config file name under $WORK/configs}
LINKS=${2:?number of 12-hour links}
TARGET=${3:?target iteration count (the manifest iteration at which the run is complete)}
NAME=${4:-rli-${CONFIG%.json}}
[[ -s "$WORK/configs/$CONFIG" ]] || { echo "no config $WORK/configs/$CONFIG" >&2; exit 2; }
[[ "$LINKS" =~ ^[0-9]+$ && "$LINKS" -ge 1 ]] || { echo "LINKS must be a positive integer" >&2; exit 2; }
[[ "$TARGET" =~ ^[0-9]+$ && "$TARGET" -ge 1 ]] || { echo "TARGET_ITERATIONS must be a positive integer" >&2; exit 2; }
if (( TARGET > 3 )) && [[ -z "${ALLOW_LONG:-}" ]]; then
  echo "the series' stopping rule ends an arm after three iterations without a paired gain; ALLOW_LONG=1 lifts this for a promoted arm" >&2
  exit 2
fi
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
    "$WORK/source/research/hpc/rl_improvement_20261110/rl.sbatch")
  previous=${previous%%;*}
  ids+=("$previous")
done
echo "$NAME config=$CONFIG links=${ids[*]} target_iterations=$TARGET source=$(git -C "$WORK/source" rev-parse --short=7 HEAD) submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"
squeue -u "$USER" -o "%.12i %.24j %.9T %.10M %.26E %.20R"
