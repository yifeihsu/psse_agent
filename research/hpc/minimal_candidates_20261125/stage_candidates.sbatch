#!/usr/bin/env bash
#SBATCH --job-name=mcs-score
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/minimal_candidates_20261125/logs/%x-%j.out
# One arm of the candidate read-out (research/minimal_env/candidate_sets.py score): at every stopping state of the
# arm's unrepaired development episodes, the exact probability its model gives the most probable corrections it
# could write there.  Teacher-forced, a few dozen forward passes a state, nothing generated, nothing trained.
#
# ARM names the arm (cell.env: base, steps17); submit_cell.sh sets it.  On a card of CS_TWO_WORKERS_FROM_MIB or
# more two processes share it, each with half of the states; then the call without a shard scores whatever is left
# and writes summary.json.  One file per state, so a preempted job loses the state it was on and its requeued
# start skips what is there.
set -euo pipefail
WORK=/scratch/yx3882/minimal_candidates_20261125
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_candidates_20261125/cell.env"
cell_environment
ARM=${ARM:?the arm to score (see ARMS in cell.env); submit_cell.sh sets it}
ADAPTER=$(arm_adapter "$ARM")
INPUTS=$(arm_inputs "$ARM")
SCORES=$(arm_scores "$ARM")
mkdir -p "$LOGS" "$OUT"
cd "$SRC"
echo "[candidates] arm $ARM job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[candidates] source $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
[[ -s "$ADAPTER/adapter_config.json" ]] || { echo "no adapter at $ADAPTER" >&2; exit 2; }
[[ -s "$INPUTS" ]] || { echo "no inputs at $INPUTS: deploy uploads them" >&2; exit 2; }
if [[ -s "$SCORES/summary.json" ]]; then
  echo "[candidates] $SCORES/summary.json present; nothing to do"
  echo "mcs-score-complete"
  exit 0
fi
SCORE=("$PY" -m research.minimal_env.candidate_sets score --adapter "$ADAPTER" --inputs "$INPUTS"
       --output-dir "$SCORES" --model-choice "$MODEL_CHOICE" --max-actions "$CS_MAX_ACTIONS" --table-versions "$CS_TABLE_VERSIONS"
       --max-input-tokens "$CS_MAX_INPUT_TOKENS" --max-length "$CS_MAX_LENGTH" --temperature 1.0
       --top "$CS_TOP" --max-passes "$CS_MAX_PASSES")
echo "[candidates] adapter $ADAPTER inputs $INPUTS output $SCORES top $CS_TOP passes at most $CS_MAX_PASSES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-$ARM-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
CPUS=${SLURM_CPUS_PER_TASK:-8}
if [[ -z "${CS_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
  CS_WORKERS=1
  (( ${gpu_mib:-0} >= CS_TWO_WORKERS_FROM_MIB )) && CS_WORKERS=2
fi
echo "[candidates] $CS_WORKERS process(es) on the card"
if (( CS_WORKERS > 1 )); then
  export OMP_NUM_THREADS=$(( CPUS / CS_WORKERS ))
  pids=()
  for (( s = 0; s < CS_WORKERS; s++ )); do
    "${SCORE[@]}" --shard "$s/$CS_WORKERS" >> "$LOGS/candidates-$ARM-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "a shard failed (pid $pid)" >&2; exit 1; }; done
fi
export OMP_NUM_THREADS=$CPUS
"${SCORE[@]}"
echo "[candidates] summary written $(date -u +%FT%TZ)"
echo "mcs-score-complete"
