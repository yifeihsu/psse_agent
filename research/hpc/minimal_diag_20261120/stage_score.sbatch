#!/usr/bin/env bash
#SBATCH --job-name=mdg-score
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/minimal_diag_20261120/logs/%x-%j.out
# The supervised model's probability of named actions at the states where its unrepaired episodes stopped
# (research/minimal_env/score_actions.py): teacher-forced, one forward pass a state and action, nothing generated.
#
# On a card of DG_TWO_WORKERS_FROM_MIB or more two processes share it, each with half of the states; then the call
# without a shard scores whatever is left and writes summary.json.  One file per state, so a preempted job loses
# the state it was on and its requeued start skips what is there.
set -euo pipefail
WORK=/scratch/yx3882/minimal_diag_20261120
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_diag_20261120/cell.env"
cell_environment
mkdir -p "$LOGS" "$OUT"
cd "$SRC"
echo "[score] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[score] source $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
[[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
for inputs in "$INPUTS_DEV" "$INPUTS_TRAIN"; do
  [[ -s "$inputs" ]] || { echo "no scoring inputs at $inputs: deploy uploads them" >&2; exit 2; }
done
if [[ -s "$SCORES/summary.json" ]]; then
  echo "[score] $SCORES/summary.json present; nothing to do"
  echo "mdg-score-complete"
  exit 0
fi
SCORE=("$PY" -m research.minimal_env.score_actions --adapter "$PASS_ADAPTER" --inputs "$INPUTS_DEV" --inputs "$INPUTS_TRAIN"
       --output-dir "$SCORES" --model-choice "$MODEL_CHOICE" --max-actions "$DG_MAX_ACTIONS" --table-versions "$DG_TABLE_VERSIONS"
       --max-input-tokens "$DG_MAX_INPUT_TOKENS" --max-length "$DG_MAX_LENGTH" --temperature 1.0)
echo "[score] adapter $PASS_ADAPTER inputs $INPUTS_DEV $INPUTS_TRAIN output $SCORES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-score-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
CPUS=${SLURM_CPUS_PER_TASK:-8}
if [[ -z "${DG_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
  DG_WORKERS=1
  (( ${gpu_mib:-0} >= DG_TWO_WORKERS_FROM_MIB )) && DG_WORKERS=2
fi
echo "[score] $DG_WORKERS process(es) on the card"
if (( DG_WORKERS > 1 )); then
  export OMP_NUM_THREADS=$(( CPUS / DG_WORKERS ))
  pids=()
  for (( s = 0; s < DG_WORKERS; s++ )); do
    "${SCORE[@]}" --shard "$s/$DG_WORKERS" >> "$LOGS/score-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "a shard failed (pid $pid)" >&2; exit 1; }; done
fi
export OMP_NUM_THREADS=$CPUS
"${SCORE[@]}"
echo "[score] summary written $(date -u +%FT%TZ)"
echo "mdg-score-complete"
