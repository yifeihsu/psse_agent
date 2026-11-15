#!/usr/bin/env bash
#SBATCH --job-name=orc-arm
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/clone_open_rules_20261112/logs/%x-%j.out
# The RL arm in the open environment (2026-11-14): the driver's iterations from the cloned adapter with the
# controller's five rules lifted, as arm A of the RL series ran them with the rules in place
# (research/hpc/rl_improvement_20261110/rl.sbatch).  The run is the one the clone's baseline began:
# configs/baseline_clone_open.json is arm A's config in the open environment from the cloned adapter, its run folder
# holds the baseline (the clone on the 300 development cases) and its manifest stands at the first rollout.  So the
# arm's evaluations pair with that baseline, and nothing is measured twice.
#
#   sbatch --export=ALL,ITERATIONS=2,TARGET_ITERATIONS=2 stage_arm.sbatch        (submit_cell.sh arm does this, as a chain)
#
# ITERATIONS is how many iterations this job attempts; TARGET_ITERATIONS ends the job as soon as the manifest's
# iteration reaches it (the series' two-iteration screen; a third is a new submission).  ARM_CONFIG names another
# config under $CONFIG_DIR.  Each phase is its own process and the manifest says where the run stands, so a
# preemption loses at most the root group or the checkpoint interval in flight; an update that fails an integrity
# check leaves its marker and the run where it is.
#
# What differs from the series' script: this cell's source and work directory; the baseline is never run here (the
# clone's baseline is a job of its own, stage_baseline.sbatch, and this refuses to start before it has closed); a
# config that lifts no rule is refused.
set -euo pipefail
WORK=/scratch/yx3882/clone_open_rules_20261112
# shellcheck source=cell.env
source "$WORK/source/research/hpc/clone_open_rules_20261112/cell.env"
cell_environment
ARM_CONFIG=${ARM_CONFIG:-baseline_clone_open.json}
ITERATIONS=${ITERATIONS:-2}
TARGET_ITERATIONS=${TARGET_ITERATIONS:-}
CFG=$CONFIG_DIR/$ARM_CONFIG
NAME=${ARM_CONFIG%.json}
[[ -s "$CFG" ]] || { echo "no config $CFG: deploy it first" >&2; exit 2; }
mkdir -p "$LOGS"
cd "$SRC"
config_field() { "$PY" -c 'import json,sys; value=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if value is None else value)' "$CFG" "$1"; }
# The driver sets the policy's input limit from the config in every phase; the export gives the same limit to
# anything else this job runs.
RESEARCH_MAX_INPUT_TOKENS=$(config_field max_input_tokens)
export RESEARCH_MAX_INPUT_TOKENS=${RESEARCH_MAX_INPUT_TOKENS:-$TRAIN_MAX_LENGTH}
ADAPTER=$(config_field initial_adapter)
RUN=$(config_field output_dir)
OPEN_RULES=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_options"].get("open_rules", ""))' "$CFG")
echo "[arm:$NAME] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} iterations $ITERATIONS target ${TARGET_ITERATIONS:-none} $(date -u +%FT%TZ)"
echo "[arm:$NAME] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens observation $(config_field observation_mode)"
echo "[arm:$NAME] adapter $ADAPTER run $RUN open rules ${OPEN_RULES:-none}"
[[ -n "$OPEN_RULES" ]] || { echo "the config lifts no rule: this cell's arm runs in the open environment" >&2; exit 2; }
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "the adapter is not there: $ADAPTER" >&2; exit 2; }
# The update labels the choice on a rejected candidate where the commit rule is lifted; a source without that
# reading would train the arm on accepted candidates alone.
grep -q "def rejected_choice" "$SRC/psse_env/rl/train.py" \
  || { echo "the source at $SRC does not label the choice on rejected candidates: deploy the arm's commit first" >&2; exit 2; }
if another_link_running; then
  echo "[arm:$NAME] another link of $SLURM_JOB_NAME is running; leaving it the output directory" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-arm-$NAME-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
# A record of what this job runs on; a difference from what the series planned is reported, not refused.
"$PY" -m research.hpc.rl_improvement_20261110.provenance --config "$CFG" > "$LOGS/provenance-arm-$NAME-${SLURM_JOB_ID:-manual}.json" \
  || echo "[arm:$NAME] the provenance record could not be written; the job goes on" >&2

status_field() { "$PY" -m psse_env.rl.train --config "$CFG" --status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }
reached_target() { [[ -n "$TARGET_ITERATIONS" ]] && (( $(status_field iteration) >= TARGET_ITERATIONS )); }
stopped_by_integrity() {
  local marker
  marker=$(printf '%s/iteration_%03d/integrity_failure.json' "$RUN" "$(status_field iteration)")
  [[ -s "$marker" ]] && { echo "[arm:$NAME] stopped: the update of iteration $(status_field iteration) failed an integrity check ($marker)"; cat "$marker"; echo; }
}
"$PY" -m psse_env.rl.train --config "$CFG" --status
# The baseline belongs to stage_baseline.sbatch: the arm starts from a closed one, so every evaluation pairs with it.
if [[ "$(status_field phase)" == baseline || ! -s "$RUN/baseline/evaluation.json" ]]; then
  echo "the baseline of $ARM_CONFIG has not closed ($RUN/baseline/evaluation.json): run it first (submit_cell.sh clone)" >&2
  exit 2
fi
if reached_target; then echo "[arm:$NAME] the manifest is at iteration $(status_field iteration), the target $TARGET_ITERATIONS: nothing to do"; echo "orc-arm-complete"; exit 0; fi
if stopped_by_integrity; then echo "orc-arm-stopped"; exit 0; fi

# As rl.sbatch: on a card of 70 GB or more two processes share it for the rollout, continuation and evaluation
# phases, each taking every second root, and the unsharded call then finishes what is left and closes the phase.
# The update is one process; it checkpoints itself, so a preemption resumes it mid-step.
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= 70000 )) && RL_WORKERS=2
fi
echo "[arm:$NAME] $RL_WORKERS worker(s) for the rollout, continuation and evaluation phases"
for (( i = 0; i < ITERATIONS; i++ )); do
  if reached_target; then echo "[arm:$NAME] target of $TARGET_ITERATIONS iterations reached $(date -u +%FT%TZ)"; break; fi
  # The driver refuses a phase the manifest is not at, so the loop runs the phase it is at and skips the others;
  # the continuation runs only in a run with decision credit (the manifest never names it otherwise).
  for phase in rollout continuation update evaluate; do
    [[ "$(status_field phase)" == "$phase" ]] || continue
    echo "[arm:$NAME] iteration loop $i phase $phase $(date -u +%FT%TZ)"
    if [[ "$phase" != update && "$RL_WORKERS" -gt 1 ]]; then
      pids=()
      for (( s = 0; s < RL_WORKERS; s++ )); do
        "$PY" -m psse_env.rl.train --config "$CFG" --phase "$phase" --shard "$s/$RL_WORKERS" \
          >> "$LOGS/arm-$phase-$NAME-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || { echo "$phase shard failed (pid $pid)" >&2; exit 1; }; done
    fi
    if ! "$PY" -m psse_env.rl.train --config "$CFG" --phase "$phase"; then
      # An update that shows a stop condition leaves its marker and the run where it is.
      if stopped_by_integrity; then echo "orc-arm-stopped"; exit 0; fi
      echo "$phase failed" >&2; exit 1
    fi
  done
done
"$PY" -m psse_env.rl.train --config "$CFG" --status
echo "orc-arm-complete"
