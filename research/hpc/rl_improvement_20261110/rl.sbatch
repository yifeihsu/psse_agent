#!/usr/bin/env bash
#SBATCH --job-name=rli
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
#SBATCH --output=/scratch/yx3882/rl_improvement_20261110/logs/%x-%j.out
# The RL improvement series of 2026-11-10 (docs/rl_improvement_workplan_20261110.md): one arm, smoke or
# calibration run of research/hpc/rl_improvement_20261110/series.json.  The driver is psse_env.rl.train,
# as in the cell of 2026-11-07 (research/hpc/compound_rl_20261107/rl.sbatch): each phase is its own
# process, the manifest under the run says where it stands, and a preemption loses at most the root
# group, the branch or the checkpoint interval in flight.
#
#   sbatch --export=ALL,CONFIG=smoke_complete.json,ITERATIONS=1,TARGET_ITERATIONS=1 --job-name=rli-smoke rl.sbatch
#   sbatch --export=ALL,CONFIG=arm_a_bounded.json,ITERATIONS=2,TARGET_ITERATIONS=2 --job-name=rli-a rl.sbatch
#   sbatch --export=ALL,CONFIG=arm_b_complete.json,TASK=capacity --job-name=rli-capacity rl.sbatch
#
# CONFIG names a config under $WORK/configs.  ITERATIONS is how many iterations this job attempts;
# TARGET_ITERATIONS ends the job as soon as the manifest's iteration reaches it, which is how the
# workplan's two-iteration screen is held: a third iteration is a new submission, not a default.
#
# What this script adds to the earlier one:
#   * the continuation phase of an arm with decision credit (sharded like the rollout);
#   * the prompt limit is the config's (max_input_tokens), exported for every tool the job runs;
#   * a provenance record before the first phase (source commit, config and input digests);
#   * a run whose update failed an integrity check stops here: the marker beside its report says why;
#   * TASK=capacity measures what prompt length the card holds instead of training.
set -euo pipefail
WORK=/scratch/yx3882/rl_improvement_20261110
CONFIG=${CONFIG:?set CONFIG to a config file name under $WORK/configs}
ITERATIONS=${ITERATIONS:-2}
TARGET_ITERATIONS=${TARGET_ITERATIONS:-}
TASK=${TASK:-train}
SRC=$WORK/source
LOGS=$WORK/logs
CELL=research.hpc.rl_improvement_20261110
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python
CFG="$WORK/configs/$CONFIG"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/scratch/yx3882/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
export RESEARCH_MAX_NEW_TOKENS=256 PSSE_LOCAL_DIAGNOSTIC_BUILD=1
export PSSE_HIF_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
# The config names the teacher variant and every environment option; this variable only has to
# name a known variant for the modules that read it at import time.
export PSSE_EXPERT_VARIANT=baseline_compare
unset PSSE_MULTI_SCAN_RCOND

mkdir -p "$LOGS"
[[ -s "$CFG" ]] || { echo "no config $CFG" >&2; exit 2; }
cd "$SRC"
config_field() { "$PY" -c 'import json,sys; value=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if value is None else value)' "$CFG" "$1"; }
# The driver sets the policy's input limit from the config in every phase; the export gives the
# same limit to anything else this job runs (32,768, the earlier cells' value, for a config without one).
RESEARCH_MAX_INPUT_TOKENS=$(config_field max_input_tokens)
export RESEARCH_MAX_INPUT_TOKENS=${RESEARCH_MAX_INPUT_TOKENS:-32768}
OUT=$(config_field output_dir)

echo "[rl] config $CONFIG task $TASK iterations $ITERATIONS target ${TARGET_ITERATIONS:-none} job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[rl] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens observation $(config_field observation_mode)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-${CONFIG%.json}-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
# A record of what this job runs on; a difference from what the series planned is reported, not refused.
"$PY" -m "$CELL.provenance" --config "$CFG" > "$LOGS/provenance-${CONFIG%.json}-${SLURM_JOB_ID:-manual}.json" \
  || echo "[rl] the provenance record could not be written; the job goes on" >&2

if [[ "$TASK" == capacity ]]; then
  # What prompt length this card holds for a generation and for an update row (research.rl.prompt_capacity).
  card=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_')
  "$PY" -m research.rl.prompt_capacity --config "$CFG" --lengths ${CAPACITY_LENGTHS:-8192 16384 24576 32768 40960 49152} \
    --output "$WORK/out/prompt_capacity/${card}-${SLURM_JOB_ID:-manual}"
  echo "rli-job-complete"
  exit 0
fi
[[ "$TASK" == train ]] || { echo "unknown TASK $TASK (train or capacity)" >&2; exit 2; }

# One rollout, continuation or evaluation process leaves the GPU idle while the environment steps on
# the CPU; on a card of 70 GB or more two processes share it, each taking every second root, and the
# unsharded call of the phase then finishes what is left and closes the phase.  Under the complete
# history a long prompt needs more than the bounded ones did: a worker gives its cached memory back
# after one, and a prompt that still finds no room waits, tries again and is then refused and counted
# (never cut).  The update is one process; it checkpoints itself, so a preemption resumes it mid-step.
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= 70000 )) && RL_WORKERS=2
fi
echo "[rl] $RL_WORKERS worker(s) for the rollout, continuation and evaluation phases"
status_field() { "$PY" -m psse_env.rl.train --config "$CFG" --status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }
reached_target() { [[ -n "$TARGET_ITERATIONS" ]] && (( $(status_field iteration) >= TARGET_ITERATIONS )); }
stopped_by_integrity() {
  local marker
  marker=$(printf '%s/iteration_%03d/integrity_failure.json' "$OUT" "$(status_field iteration)")
  [[ -s "$marker" ]] && { echo "[rl] stopped: the update of iteration $(status_field iteration) failed an integrity check ($marker)"; cat "$marker"; echo; }
}
"$PY" -m psse_env.rl.train --config "$CFG" --status
if reached_target; then echo "[rl] the manifest is at iteration $(status_field iteration), the target $TARGET_ITERATIONS: nothing to do"; echo "rli-job-complete"; exit 0; fi
if stopped_by_integrity; then echo "rli-job-stopped"; exit 0; fi
for (( i = 0; i < ITERATIONS; i++ )); do
  if reached_target; then echo "[rl] target of $TARGET_ITERATIONS iterations reached $(date -u +%FT%TZ)"; break; fi
  # The driver refuses a phase the manifest is not at, so the loop runs the phase it is at and skips
  # the others; the baseline runs once before the first rollout, the continuation only in a run with
  # decision credit (the manifest never names it otherwise).
  for phase in baseline rollout continuation update evaluate; do
    [[ "$(status_field phase)" == "$phase" ]] || continue
    echo "[rl] iteration loop $i phase $phase $(date -u +%FT%TZ)"
    if [[ "$phase" != update && "$RL_WORKERS" -gt 1 ]]; then
      pids=()
      for (( s = 0; s < RL_WORKERS; s++ )); do
        "$PY" -m psse_env.rl.train --config "$CFG" --phase "$phase" --shard "$s/$RL_WORKERS" \
          >> "$LOGS/$phase-${CONFIG%.json}-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || { echo "$phase shard failed (pid $pid)" >&2; exit 1; }; done
    fi
    if ! "$PY" -m psse_env.rl.train --config "$CFG" --phase "$phase"; then
      # An update that shows a stop condition leaves its marker and the run where it is.
      if stopped_by_integrity; then echo "rli-job-stopped"; exit 0; fi
      echo "$phase failed" >&2; exit 1
    fi
  done
done
"$PY" -m psse_env.rl.train --config "$CFG" --status
echo "rli-job-complete"
