#!/usr/bin/env bash
#SBATCH --job-name=mrl
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
#SBATCH --output=/scratch/yx3882/minimal_rl_20261116/logs/%x-%j.out
# RL in the minimal environment (research/minimal_env/rl_train.py).  RL_RUN names the run and its config:
#
#   ieee14  the run: it starts from the supervised pass's adapter and refuses to start without that pass's receipt.
#   smoke   one iteration on six training cases and eight development cases, from the cell's own copy of the pass's
#           adapter or, while the pass is running, of its newest complete checkpoint.  Read for what only the model
#           can show (README.md), not for a score.
#
# RL_TARGET is the iteration count at which the job stops.  The job runs whatever phase the run's manifest is at,
# one process per phase, until the manifest reaches the target or the time runs out; a later link or a preemption
# requeue goes on from the manifest.  Rollouts and evaluations keep one file per case and the update checkpoints
# itself, so a restart loses at most the case or the checkpoint interval in flight.
set -euo pipefail
WORK=/scratch/yx3882/minimal_rl_20261116
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_rl_20261116/cell.env"
cell_environment
: "${RL_RUN:?RL_RUN must be ieee14 or smoke}"
: "${RL_TARGET:?RL_TARGET must be the iteration count at which the job stops}"
case "$RL_RUN" in
  ieee14|smoke) ;;
  *) echo "RL_RUN must be ieee14 or smoke, got $RL_RUN" >&2; exit 2 ;;
esac
[[ "$RL_TARGET" =~ ^[0-9]+$ && "$RL_TARGET" -ge 1 ]] || { echo "RL_TARGET must be a positive integer, got $RL_TARGET" >&2; exit 2; }
CFG=$CELL/rl_$RL_RUN.json
RUN_DIR=$OUT/$RL_RUN
DRIVER=("$PY" -m research.minimal_env.rl_train --config "$CFG")
mkdir -p "$LOGS" "$OUT" "$DATA"
cd "$SRC"
echo "[rl:$RL_RUN] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[rl:$RL_RUN] source $(git -C "$SRC" rev-parse HEAD) target $RL_TARGET iteration(s) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
[[ -s "$CFG" ]] || { echo "no config $CFG" >&2; exit 2; }

if [[ "$RL_RUN" == ieee14 ]]; then
  [[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
elif [[ ! -s "$SMOKE_ADAPTER/adapter_model.safetensors" ]]; then
  # The smoke's adapter, copied once: the two files an adapter folder of the pass holds, so that the processor
  # comes from the base model as it will in the run (a checkpoint also carries the tokenizer's files).
  if [[ -s "$PASS/training.done" ]]; then origin=$PASS_ADAPTER; else origin=$(newest_complete_checkpoint); fi
  [[ -n "$origin" && -s "$origin/adapter_model.safetensors" && -s "$origin/adapter_config.json" ]] \
    || { echo "the supervised pass has neither an adapter nor a complete checkpoint under $PASS" >&2; exit 2; }
  rm -rf "$SMOKE_ADAPTER.part"
  mkdir -p "$SMOKE_ADAPTER.part"
  cp "$origin/adapter_config.json" "$origin/adapter_model.safetensors" "$SMOKE_ADAPTER.part/"
  echo "copied from $origin at $(date -u +%FT%TZ)" > "$SMOKE_ADAPTER.part/SOURCE.txt"
  mv "$SMOKE_ADAPTER.part" "$SMOKE_ADAPTER"
fi
[[ "$RL_RUN" == smoke ]] && echo "[rl:smoke] adapter: $(cat "$SMOKE_ADAPTER/SOURCE.txt")"

if another_link_running; then
  echo "[rl:$RL_RUN] another link of $SLURM_JOB_NAME is running; leaving it the run directory" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-rl-$RL_RUN-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

status_field() { "${DRIVER[@]}" --status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }
# An update that shows a stop condition leaves a marker beside its report and the run where it is.
stopped_by_integrity() {
  local marker
  marker=$(printf '%s/iteration_%03d/integrity_failure.json' "$RUN_DIR" "$(status_field iteration)")
  [[ -s "$marker" ]] && { echo "[rl:$RL_RUN] stopped: the update of iteration $(status_field iteration) failed an integrity check ($marker)"; cat "$marker"; echo; }
}
if (( $(status_field iteration) >= RL_TARGET )); then
  echo "[rl:$RL_RUN] the run is at iteration $(status_field iteration), the target $RL_TARGET: nothing to do"
  echo "mrl-$RL_RUN-complete"
  exit 0
fi
if stopped_by_integrity; then echo "mrl-$RL_RUN-stopped"; exit 0; fi
# The inputs, before any GPU time: the suites load, the first iteration's cases draw by class, the adapter is there.
"${DRIVER[@]}" --check

# A rollout or evaluation process leaves the card idle while the numerical tools run on the CPUs; on a large card
# two processes share it, each taking every second case, and the call without a shard then finishes what is left
# and closes the phase.  The update is one process.
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= RL_TWO_WORKERS_FROM_MIB )) && RL_WORKERS=2
fi
echo "[rl:$RL_RUN] $RL_WORKERS worker(s) for the rollout and the evaluations"
CPUS=${SLURM_CPUS_PER_TASK:-8}
while (( $(status_field iteration) < RL_TARGET )); do
  phase=$(status_field phase)
  echo "[rl:$RL_RUN] iteration $(status_field iteration) phase $phase $(date -u +%FT%TZ)"
  if [[ "$phase" != update && "$RL_WORKERS" -gt 1 ]]; then
    export OMP_NUM_THREADS=$(( CPUS / RL_WORKERS ))
    pids=()
    for (( s = 0; s < RL_WORKERS; s++ )); do
      "${DRIVER[@]}" --phase "$phase" --shard "$s/$RL_WORKERS" >> "$LOGS/$phase-$RL_RUN-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || { echo "$phase shard failed (pid $pid)" >&2; exit 1; }; done
  fi
  export OMP_NUM_THREADS=$CPUS
  if ! "${DRIVER[@]}" --phase "$phase"; then
    if stopped_by_integrity; then echo "mrl-$RL_RUN-stopped"; exit 0; fi
    echo "$phase failed" >&2
    exit 1
  fi
done
"${DRIVER[@]}" --status
echo "[rl:$RL_RUN] target of $RL_TARGET iteration(s) reached $(date -u +%FT%TZ)"
echo "mrl-$RL_RUN-complete"
