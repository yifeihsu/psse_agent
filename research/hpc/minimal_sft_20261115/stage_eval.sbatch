#!/usr/bin/env bash
#SBATCH --job-name=msf-eval
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
#SBATCH --output=/scratch/yx3882/minimal_sft_20261115/logs/%x-%j.out
# An adapter evaluated in the minimal environment (research/minimal_env/evaluate.py): greedy, bf16, 16 actions, the
# prompt the rows were built with.  EVAL_NAME is one of
#
#   dev300  the pass's adapter (SFT_START names the pass: r1 or base) on the 300 development cases.  Refuses to
#           start without that pass's receipt.
#   smoke   R1 on eight of the cases (the slice is cut here from the development subset).  R1 was trained on another
#           prompt and other tools, so what it solves says nothing: the smoke is read for prompt length, memory,
#           refusals and the files the evaluation writes.
#
# Resumable: each case's episode is its own file, so a preemption requeue or a later link skips what is done, and
# the job ends at once when evaluation.json is there.
set -euo pipefail
WORK=/scratch/yx3882/minimal_sft_20261115
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_sft_20261115/cell.env"
cell_environment
resolve_start
: "${EVAL_NAME:?EVAL_NAME must be dev300 or smoke}"
case "$EVAL_NAME" in
  dev300)
    ADAPTER=$OUT/pass_$START_NAME/training/lora
    SUITE=$DEVELOPMENT_SUITE
    EVAL_DIR=$OUT/eval_dev300_$START_NAME
    [[ -s "$OUT/pass_$START_NAME/training.done" ]] || { echo "no training receipt $OUT/pass_$START_NAME/training.done: the pass has not finished" >&2; exit 2; }
    ;;
  smoke)
    ADAPTER=$R1_ADAPTER
    SUITE=$SMOKE_SUITE
    EVAL_DIR=$OUT/eval_smoke
    ;;
  *) echo "EVAL_NAME must be dev300 or smoke, got $EVAL_NAME" >&2; exit 2 ;;
esac
mkdir -p "$LOGS" "$EVAL_DIR"
cd "$SRC"
echo "[eval:$EVAL_NAME] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[eval:$EVAL_NAME] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
echo "[eval:$EVAL_NAME] adapter $ADAPTER suite $SUITE output $EVAL_DIR"
if [[ -s "$EVAL_DIR/evaluation.json" ]]; then
  echo "[eval:$EVAL_NAME] evaluation.json present; nothing to do"
  echo "msf-eval-$EVAL_NAME-complete"
  exit 0
fi
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "the adapter is not there: $ADAPTER" >&2; exit 2; }
if [[ "$EVAL_NAME" == smoke && ! -s "$SUITE" ]]; then
  mkdir -p "$(dirname "$SUITE")"
  "$PY" "$TOOLS/row_gate.py" suite --suite "$DEVELOPMENT_SUITE" --roots "$SMOKE_EVAL_ROOTS" --output "$SUITE"
fi
[[ -s "$SUITE" ]] || { echo "suite missing: $SUITE" >&2; exit 2; }
if another_link_running; then
  echo "[eval:$EVAL_NAME] another link of $SLURM_JOB_NAME is running; leaving it the output directory" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-eval-$EVAL_NAME-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

EVALUATE=("$PY" -m research.minimal_env.evaluate --suite "$SUITE" --adapter "$ADAPTER" --output-dir "$EVAL_DIR"
          --model-choice "$MODEL_CHOICE" --max-actions "$EVAL_MAX_ACTIONS" --table-versions "$EVAL_TABLE_VERSIONS"
          --reference "$REFERENCE")
# On a card of 70 GB or more two processes share it, each taking every second case; the unsharded call then
# finishes what is left and writes the summary.  The numerical tools run on the CPUs between generations, so two
# processes also keep the card busier.
if [[ -z "${EVAL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
  EVAL_WORKERS=1
  (( ${gpu_mib:-0} >= 70000 )) && EVAL_WORKERS=2
fi
echo "[eval:$EVAL_NAME] $EVAL_WORKERS worker(s) $(date -u +%FT%TZ)"
if [[ "$EVAL_WORKERS" -gt 1 ]]; then
  export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / EVAL_WORKERS ))
  pids=()
  for (( s = 0; s < EVAL_WORKERS; s++ )); do
    "${EVALUATE[@]}" --shard "$s/$EVAL_WORKERS" >> "$LOGS/eval-$EVAL_NAME-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "evaluation shard failed (pid $pid)" >&2; exit 1; }; done
fi
"${EVALUATE[@]}"
echo "msf-eval-$EVAL_NAME-complete"
