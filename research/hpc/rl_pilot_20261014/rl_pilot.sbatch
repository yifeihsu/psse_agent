#!/usr/bin/env bash
#SBATCH --job-name=rl-pilot
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
#SBATCH --output=/scratch/yx3882/rl_pilot_20261014/logs/%x-%j.out
# RL pilot (docs/rl_experiments_20261014.md): on-policy LoRA updates of the
# frozen round-2 student from its own sampled rollouts, scored by the offline
# audit.  Each phase is its own process so a preemption loses at most the root
# group in flight; the manifest under $RUN says where the run stands.
#
#   sbatch --export=ALL,CONFIG=rl_pilot.json --job-name=rl-pilot rl_pilot.sbatch
#   sbatch --export=ALL,CONFIG=rl_pilot.json,ITERATIONS=4 rl_pilot.sbatch
#
# CONFIG names a config under $WORK/configs (the driver reads it; paths inside
# are cluster paths), ITERATIONS how many iterations this job attempts before
# exiting (default 2; resubmit to continue), PHASES_PER_JOB=0 runs until the
# time limit.  The same job script restarts after a requeue and continues.
set -euo pipefail
WORK=/scratch/yx3882/rl_pilot_20261014
CONFIG=${CONFIG:?set CONFIG to a config file name under $WORK/configs}
ITERATIONS=${ITERATIONS:-2}
SRC=$WORK/source
LOGS=$WORK/logs
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/scratch/yx3882/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
export RESEARCH_MAX_INPUT_TOKENS=32768 RESEARCH_MAX_NEW_TOKENS=256 PSSE_LOCAL_DIAGNOSTIC_BUILD=1
export PSSE_HIF_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$LOGS"
echo "[rl] config $CONFIG iterations $ITERATIONS job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[rl] source $(git -C "$SRC" rev-parse HEAD)"
[[ -s "$WORK/configs/$CONFIG" ]] || { echo "no config $WORK/configs/$CONFIG" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-${CONFIG%.json}-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
# The OpenDSS model in node-local memory, as the pipeline stages do.
OPENDSS_LOCAL=/dev/shm/psse_opendss_${SLURM_JOB_ID:-manual}
rm -rf "$OPENDSS_LOCAL" && cp -r "$SRC/IEEE_14_OpenDSS" "$OPENDSS_LOCAL"
export PSSE_OPENDSS_MODEL_DIR="$OPENDSS_LOCAL"
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; rm -rf "$OPENDSS_LOCAL"; }
trap cleanup EXIT

cd "$SRC"
# One rollout or evaluation process leaves the GPU idle while the environment
# steps on the CPU; on an 80 GB card two processes share it (each holds a 4-bit
# copy), each taking every second root, and the unsharded call of the phase
# then finishes any leftover root and closes the phase.  The update is one
# process; it checkpoints itself, so a preemption resumes it mid-step.
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= 70000 )) && RL_WORKERS=2
fi
echo "[rl] $RL_WORKERS worker(s) for the rollout and evaluation phases"
"$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --status
for (( i = 0; i < ITERATIONS; i++ )); do
  # Each phase is one process; the driver refuses a phase the manifest is not at,
  # so after a requeue the loop resumes at the recorded phase.
  for phase in rollout update evaluate; do
    current=$("$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
    [[ "$current" == "$phase" ]] || continue
    echo "[rl] iteration loop $i phase $phase $(date -u +%FT%TZ)"
    if [[ "$phase" != update && "$RL_WORKERS" -gt 1 ]]; then
      pids=()
      for (( s = 0; s < RL_WORKERS; s++ )); do
        "$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --phase "$phase" --shard "$s/$RL_WORKERS" \
          >> "$LOGS/$phase-${CONFIG%.json}-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || { echo "$phase shard failed (pid $pid)" >&2; exit 1; }; done
    fi
    "$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --phase "$phase"
  done
done
"$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --status
echo "rl-pilot-job-complete"
