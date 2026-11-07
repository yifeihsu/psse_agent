#!/usr/bin/env bash
#SBATCH --job-name=crl
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
#SBATCH --output=/scratch/yx3882/compound_rl_20261107/logs/%x-%j.out
# RL on the compound suites (docs/rl_beyond_teacher_plan_20261105.md, stage 3): on-policy LoRA
# updates of the cloned student from its own sampled rollouts under scada_only with the lifecycle
# choice and the detailed evidence on, scored by the offline audit, the lifecycle term from the
# audit's labels.  The driver is psse_env.rl.train, unchanged in shape from the October pilot
# (research/hpc/rl_pilot_20261014/rl_pilot.sbatch): each phase is its own process so a preemption
# loses at most the root group in flight, and the manifest under the run says where it stands.
#
#   sbatch --export=ALL,CONFIG=rl_smoke.json,ITERATIONS=1 --job-name=crl-smoke rl.sbatch
#   sbatch --export=ALL,CONFIG=rl_ieee14.json,ITERATIONS=2 --job-name=crl-ieee14 rl.sbatch
#
# CONFIG names a config under $WORK/configs (cluster paths inside), ITERATIONS how many iterations
# this job attempts before exiting (default 2; resubmit to continue; a requeue restarts the loop and
# continues at the manifest's phase).  TARGET_ITERATIONS, when set, ends the job as soon as the
# manifest's iteration reaches it, so the later links of a chain (submit_arm.sh) exit at once.  A
# config with baseline_evaluation starts with the baseline phase, the starting adapter's greedy
# evaluation under this driver's bf16 loader (the cloning cell's arms ran on a 4-bit base, so they
# are not this run's baseline); it is sharded like the evaluation.
set -euo pipefail
WORK=/scratch/yx3882/compound_rl_20261107
CONFIG=${CONFIG:?set CONFIG to a config file name under $WORK/configs}
ITERATIONS=${ITERATIONS:-2}
TARGET_ITERATIONS=${TARGET_ITERATIONS:-}
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
# The config names the teacher variant and every environment option; this variable only has to
# name a known variant for the modules that read it at import time.
export PSSE_EXPERT_VARIANT=baseline_compare
# IEEE 14: the multi-scan corrector's default conditioning guard (the cloning cell's eval sets it
# per system; the compound suites here are IEEE 14).
unset PSSE_MULTI_SCAN_RCOND

mkdir -p "$LOGS"
echo "[rl] config $CONFIG iterations $ITERATIONS job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[rl] source $(git -C "$SRC" rev-parse HEAD)"
[[ -s "$WORK/configs/$CONFIG" ]] || { echo "no config $WORK/configs/$CONFIG" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-${CONFIG%.json}-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
# scada_only refuses every OpenDSS-backed tool, so no model copy to node-local memory.

cd "$SRC"
# One rollout or evaluation process leaves the GPU idle while the environment steps on the CPU; on
# an 80 GB card two processes share it (each holds a bf16 copy of about 24 GB), each taking every
# second root, and the unsharded call of the phase then finishes any leftover root and closes the
# phase.  The update is one process; it checkpoints itself, so a preemption resumes it mid-step.
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= 70000 )) && RL_WORKERS=2
fi
echo "[rl] $RL_WORKERS worker(s) for the rollout and evaluation phases"
"$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --status
manifest_iteration() {
  "$PY" -m psse_env.rl.train --config "$WORK/configs/$CONFIG" --status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["iteration"])'
}
reached_target() {
  [[ -n "$TARGET_ITERATIONS" ]] && (( $(manifest_iteration) >= TARGET_ITERATIONS ))
}
if reached_target; then echo "[rl] the manifest is at iteration $(manifest_iteration), the target $TARGET_ITERATIONS: nothing to do"; echo "crl-job-complete"; exit 0; fi
for (( i = 0; i < ITERATIONS; i++ )); do
  if reached_target; then echo "[rl] target of $TARGET_ITERATIONS iterations reached $(date -u +%FT%TZ)"; break; fi
  # The driver refuses a phase the manifest is not at, so the loop runs the phase it is at and
  # skips the others; the baseline phase runs once, before the first rollout.
  for phase in baseline rollout update evaluate; do
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
echo "crl-job-complete"
