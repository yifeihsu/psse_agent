#!/usr/bin/env bash
#SBATCH --job-name=mrl-train-pass
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
# The supervised model, greedy, on the training cases a truth-built sequence repairs
# (research/minimal_env/evaluate.py with --cases): which of them it repairs is what the focused run draws by.
#
#   PASS_PART=I/N   this job's part of the cases, I from 0 to N-1.  The cases are dealt to 2N shards and a part is
#                   two of them: on a large card both run at once, on a smaller one the second follows the first.
#   PASS_PART=close run whatever is still missing and write evaluation.json.  Submitted to follow every part.
#
# One file per case, so a preempted job loses the case it was on and a later start skips what is there.  Nothing
# here is a run of the RL driver; it reads the supervised pass's adapter in place and writes under this cell's out.
set -euo pipefail
WORK=/scratch/yx3882/minimal_rl_20261116
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_rl_20261116/cell.env"
cell_environment
: "${PASS_PART:?PASS_PART must be I/N (the part of the cases this job runs) or close}"
mkdir -p "$LOGS" "$OUT" "$DATA"
cd "$SRC"
echo "[train-pass:$PASS_PART] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[train-pass:$PASS_PART] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
[[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
[[ -s "$TRUTH_SEQUENCES" ]] || { echo "no file of truth-built sequences at $TRUTH_SEQUENCES" >&2; exit 2; }
EVALUATE=("$PY" -m research.minimal_env.evaluate --suite "$TRAIN_SUITE" --adapter "$PASS_ADAPTER" --output-dir "$TRAIN_PASS"
          --cases "$TRUTH_SEQUENCES" --model-choice "$MODEL_CHOICE" --max-actions "$RL_MAX_ACTIONS" --table-versions "$RL_TABLE_VERSIONS")
echo "[train-pass:$PASS_PART] adapter $PASS_ADAPTER suite $TRAIN_SUITE cases $TRUTH_SEQUENCES output $TRAIN_PASS"
if [[ -s "$TRAIN_PASS/evaluation.json" ]]; then
  echo "[train-pass:$PASS_PART] evaluation.json present; nothing to do"
  echo "mrl-train-pass-complete"
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-train-pass-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
CPUS=${SLURM_CPUS_PER_TASK:-8}

if [[ "$PASS_PART" == close ]]; then
  export OMP_NUM_THREADS=$CPUS
  "${EVALUATE[@]}"
  echo "[train-pass:close] evaluation written $(date -u +%FT%TZ)"
  echo "mrl-train-pass-complete"
  exit 0
fi

[[ "$PASS_PART" =~ ^([0-9]+)/([0-9]+)$ ]] || { echo "PASS_PART must be I/N or close, got $PASS_PART" >&2; exit 2; }
part=${BASH_REMATCH[1]}
parts=${BASH_REMATCH[2]}
(( parts >= 1 && part < parts )) || { echo "PASS_PART $PASS_PART names no part: I runs from 0 to N-1" >&2; exit 2; }
shards=$(( 2 * parts ))
mine=("$(( 2 * part ))/$shards" "$(( 2 * part + 1 ))/$shards")
if [[ -z "${PASS_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
  PASS_WORKERS=1
  (( ${gpu_mib:-0} >= RL_TWO_WORKERS_FROM_MIB )) && PASS_WORKERS=2
fi
echo "[train-pass:$PASS_PART] shards ${mine[*]}, $PASS_WORKERS at a time"
if (( PASS_WORKERS > 1 )); then
  export OMP_NUM_THREADS=$(( CPUS / 2 ))
  pids=()
  for shard in "${mine[@]}"; do
    "${EVALUATE[@]}" --shard "$shard" >> "$LOGS/train-pass-${SLURM_JOB_ID:-manual}-shard${shard%/*}.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "a shard of part $PASS_PART failed (pid $pid)" >&2; exit 1; }; done
else
  export OMP_NUM_THREADS=$CPUS
  for shard in "${mine[@]}"; do
    "${EVALUATE[@]}" --shard "$shard" >> "$LOGS/train-pass-${SLURM_JOB_ID:-manual}-shard${shard%/*}.out" 2>&1
  done
fi
echo "[train-pass:$PASS_PART] its shards are done $(date -u +%FT%TZ)"
echo "mrl-train-pass-part-done"
