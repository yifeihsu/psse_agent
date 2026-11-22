#!/usr/bin/env bash
#SBATCH --job-name=mrp-probe
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
#SBATCH --output=/scratch/yx3882/minimal_rl_20261122/logs/%x-%j.out
# The fork probe (research/minimal_env/fork_probe.py): the supervised model's greedy episode on cases it leaves
# unrepaired, forked at every state over every other tool, with no update.  It counts the cases one forced step
# repairs: an upper bound on what one round of stage B can teach, read before a run is spent.
#
#   PROBE=dev     the unrepaired development cases a truth-built sequence repairs (49).  For this reading only:
#                 a development fork is never a training row.
#   PROBE=train   a seeded sample of PROBE_TRAIN_SAMPLE of the unrepaired training cases (502).
#   PROBE=smoke   four development cases, with one sampled call a tool: what only the model can show (README.md).
#
#   PROBE_PART=I/N   this job's part of the cases, I from 0 to N-1.  The cases are dealt to 2N shards and a part is
#                    two of them: on a large card both run at once, on a smaller one the second follows the first.
#   PROBE_PART=close fork whatever is still missing and write summary.json.  Submitted to follow every part.
#
# One file per case, so a preempted job loses the case it was on and a later start skips what is there.  Nothing
# here is a run of the RL driver; it reads the supervised pass's adapter and its two evaluations in place.
set -euo pipefail
WORK=/scratch/yx3882/minimal_rl_20261122
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_rl_20261122/cell.env"
cell_environment
: "${PROBE:?PROBE must be dev, train or smoke}"
: "${PROBE_PART:?PROBE_PART must be I/N (the part of the cases this job forks) or close}"
mkdir -p "$LOGS" "$OUT"
cd "$SRC"
echo "[probe:$PROBE:$PROBE_PART] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[probe:$PROBE:$PROBE_PART] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
[[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
[[ -s "$PASS_ADAPTER/adapter_model.safetensors" ]] || { echo "no adapter at $PASS_ADAPTER" >&2; exit 2; }
PROBE_OUT=$OUT/probe_$PROBE
COMMON=("$PY" -m research.minimal_env.fork_probe --adapter "$PASS_ADAPTER" --output-dir "$PROBE_OUT" --model-choice "$MODEL_CHOICE"
        --max-actions "$RL_MAX_ACTIONS" --table-versions "$RL_TABLE_VERSIONS" --seed "$PROBE_SEED")
case "$PROBE" in
  dev)
    [[ -s "$DEV_PASS/evaluation.json" ]] || { echo "no evaluation $DEV_PASS/evaluation.json of the supervised model" >&2; exit 2; }
    PROBE_COMMAND=("${COMMON[@]}" --suite "$DEV_SUITE" --unrepaired-in "$DEV_PASS" --reference "$DEV_REFERENCE") ;;
  train)
    [[ -s "$TRAIN_PASS/evaluation.json" ]] || { echo "no training pass $TRAIN_PASS/evaluation.json of the supervised model" >&2; exit 2; }
    [[ -s "$TRUTH_SEQUENCES" ]] || { echo "no file of truth-built sequences at $TRUTH_SEQUENCES" >&2; exit 2; }
    PROBE_COMMAND=("${COMMON[@]}" --suite "$TRAIN_SUITE" --cases "$TRUTH_SEQUENCES" --unrepaired-in "$TRAIN_PASS" --sample "$PROBE_TRAIN_SAMPLE") ;;
  smoke)
    [[ -s "$DEV_PASS/evaluation.json" ]] || { echo "no evaluation $DEV_PASS/evaluation.json of the supervised model" >&2; exit 2; }
    PROBE_COMMAND=("${COMMON[@]}" --suite "$DEV_SUITE" --unrepaired-in "$DEV_PASS" --reference "$DEV_REFERENCE"
                   --sample "$PROBE_SMOKE_SAMPLE" --argument-samples 1) ;;
  *) echo "PROBE must be dev, train or smoke, got $PROBE" >&2; exit 2 ;;
esac
echo "[probe:$PROBE:$PROBE_PART] adapter $PASS_ADAPTER output $PROBE_OUT"
if [[ -s "$PROBE_OUT/summary.json" ]]; then
  echo "[probe:$PROBE:$PROBE_PART] summary.json present; nothing to do"
  echo "mrp-probe-$PROBE-complete"
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-probe-$PROBE-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
CPUS=${SLURM_CPUS_PER_TASK:-8}

if [[ "$PROBE_PART" == close ]]; then
  export OMP_NUM_THREADS=$CPUS
  "${PROBE_COMMAND[@]}"
  echo "[probe:$PROBE:close] summary written $(date -u +%FT%TZ)"
  echo "mrp-probe-$PROBE-complete"
  exit 0
fi

[[ "$PROBE_PART" =~ ^([0-9]+)/([0-9]+)$ ]] || { echo "PROBE_PART must be I/N or close, got $PROBE_PART" >&2; exit 2; }
part=${BASH_REMATCH[1]}
parts=${BASH_REMATCH[2]}
(( parts >= 1 && part < parts )) || { echo "PROBE_PART $PROBE_PART names no part: I runs from 0 to N-1" >&2; exit 2; }
shards=$(( 2 * parts ))
mine=("$(( 2 * part ))/$shards" "$(( 2 * part + 1 ))/$shards")
if [[ -z "${PROBE_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
  PROBE_WORKERS=1
  (( ${gpu_mib:-0} >= RL_TWO_WORKERS_FROM_MIB )) && PROBE_WORKERS=2
fi
echo "[probe:$PROBE:$PROBE_PART] shards ${mine[*]}, $PROBE_WORKERS at a time"
if (( PROBE_WORKERS > 1 )); then
  export OMP_NUM_THREADS=$(( CPUS / 2 ))
  pids=()
  for shard in "${mine[@]}"; do
    "${PROBE_COMMAND[@]}" --shard "$shard" >> "$LOGS/probe-$PROBE-${SLURM_JOB_ID:-manual}-shard${shard%/*}.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "a shard of part $PROBE_PART failed (pid $pid)" >&2; exit 1; }; done
else
  export OMP_NUM_THREADS=$CPUS
  for shard in "${mine[@]}"; do
    "${PROBE_COMMAND[@]}" --shard "$shard" >> "$LOGS/probe-$PROBE-${SLURM_JOB_ID:-manual}-shard${shard%/*}.out" 2>&1
  done
fi
echo "[probe:$PROBE:$PROBE_PART] its shards are done $(date -u +%FT%TZ)"
echo "mrp-probe-$PROBE-part-done"
