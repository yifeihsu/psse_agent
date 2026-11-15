#!/usr/bin/env bash
#SBATCH --job-name=cec-baseline
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
#SBATCH --output=/scratch/yx3882/clone_complete_evidence_20261115/logs/%x-%j.out
# One baseline of the RL driver with the controller's five rules lifted and the estimator's output shown whole:
# the driver's baseline phase from configs/baseline_$BASELINE_NAME.json (baseline_config.py), as arm A of the RL
# series ran it for S0 (research/hpc/rl_improvement_20261110/rl.sbatch).  BASELINE_NAME is one of
#
#   clone_complete  this cell's cloned adapter on the 300 development cases (refuses to start without the pass's receipt);
#   smoke_complete  S0 on eight of the cases, for the smoke (the slice is cut here from the development subset).  S0
#                   was trained on another prompt: the smoke is read for prompt length, memory and refusals.
#
# Greedy, bf16, 40 actions, bounded history, the series' prompt limit.  Only the baseline runs: the job ends when the
# manifest has left that phase, and a later link finds nothing to do.  Resumable: the evaluation is stored per case,
# so a preemption requeue or a later link skips what is done.
set -euo pipefail
WORK=/scratch/yx3882/clone_complete_evidence_20261115
# shellcheck source=cell.env
source "$WORK/source/research/hpc/clone_complete_evidence_20261115/cell.env"
cell_environment
: "${BASELINE_NAME:?BASELINE_NAME must be clone_complete or smoke_complete}"
case "$BASELINE_NAME" in
  clone_complete|smoke_complete) ;;
  *) echo "BASELINE_NAME must be clone_complete or smoke_complete, got $BASELINE_NAME" >&2; exit 2 ;;
esac
CFG=$CONFIG_DIR/baseline_$BASELINE_NAME.json
[[ -s "$CFG" ]] || { echo "no config $CFG: deploy it first" >&2; exit 2; }
mkdir -p "$LOGS"
cd "$SRC"
config_field() { "$PY" -c 'import json,sys; value=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if value is None else value)' "$CFG" "$1"; }
environment_option() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_options"].get(sys.argv[2], ""))' "$CFG" "$1"; }
# The driver sets the policy's input limit from the config in every phase; the export gives the same limit to
# anything else this job runs.
RESEARCH_MAX_INPUT_TOKENS=$(config_field max_input_tokens)
export RESEARCH_MAX_INPUT_TOKENS=${RESEARCH_MAX_INPUT_TOKENS:-$TRAIN_MAX_LENGTH}
ADAPTER=$(config_field initial_adapter)
SUITE=$(config_field development_suite)
OPEN_RULES=$(environment_option open_rules)
COMPLETE_EVIDENCE=$(environment_option complete_evidence)
echo "[baseline:$BASELINE_NAME] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[baseline:$BASELINE_NAME] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens observation $(config_field observation_mode)"
echo "[baseline:$BASELINE_NAME] adapter $ADAPTER suite $SUITE open rules ${OPEN_RULES:-none} complete evidence ${COMPLETE_EVIDENCE:-off}"
[[ -n "$OPEN_RULES" ]] || { echo "the config lifts no rule: this cell measures the open environment" >&2; exit 2; }
[[ "$COMPLETE_EVIDENCE" == 1 ]] || { echo "the config does not show the evidence whole (environment_options.complete_evidence): this cell's clone reads no other prompt" >&2; exit 2; }
# The source must know the option: an older source would ignore it and hand the clone the prompt it was not trained on.
grep -q COMPLETE_EVIDENCE_CONTRACT "$SRC/psse_env/actions.py" \
  || { echo "the source at $SRC has no complete_evidence option: deploy this cell's commit first" >&2; exit 2; }
if [[ "$BASELINE_NAME" == clone_complete ]]; then
  [[ -s "$OUT/clone/training.done" ]] || { echo "no training receipt $OUT/clone/training.done: the pass has not finished" >&2; exit 2; }
fi
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "the adapter is not there: $ADAPTER" >&2; exit 2; }
if [[ "$BASELINE_NAME" == smoke_complete && ! -s "$SUITE" ]]; then
  mkdir -p "$(dirname "$SUITE")"
  "$PY" "$BOUNDED_CELL/smoke_inputs.py" suite --suite "$DEVELOPMENT_SUITE" --roots "$SMOKE_EVAL_ROOTS" --output "$SUITE"
fi
[[ -s "$SUITE" ]] || { echo "suite missing: $SUITE" >&2; exit 2; }
if another_link_running; then
  echo "[baseline:$BASELINE_NAME] another link of $SLURM_JOB_NAME is running; leaving it the output directory" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-baseline-$BASELINE_NAME-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
# A record of what this job runs on; a difference from what the series planned is reported, not refused.
"$PY" -m research.hpc.rl_improvement_20261110.provenance --config "$CFG" > "$LOGS/provenance-baseline-$BASELINE_NAME-${SLURM_JOB_ID:-manual}.json" \
  || echo "[baseline:$BASELINE_NAME] the provenance record could not be written; the job goes on" >&2

status_field() { "$PY" -m psse_env.rl.train --config "$CFG" --status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }
"$PY" -m psse_env.rl.train --config "$CFG" --status
if [[ "$(status_field phase)" != baseline ]]; then
  echo "[baseline:$BASELINE_NAME] the manifest is past the baseline (phase $(status_field phase)): nothing to do"
  echo "cec-baseline-$BASELINE_NAME-complete"
  exit 0
fi
# As rl.sbatch, two processes share a large card, each taking every second case, and the unsharded call then
# finishes what is left and closes the phase.  The bar is 90 GB here, not the series' 70: with these prompts two
# processes peaked at 69.2 GB on eight cases (the smoke, job 23274634), which an 80 GB card does not hold with
# room for a longer prompt.  An 80 GB card runs one process.
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= 90000 )) && RL_WORKERS=2
fi
echo "[baseline:$BASELINE_NAME] $RL_WORKERS worker(s) $(date -u +%FT%TZ)"
if [[ "$RL_WORKERS" -gt 1 ]]; then
  pids=()
  for (( s = 0; s < RL_WORKERS; s++ )); do
    "$PY" -m psse_env.rl.train --config "$CFG" --phase baseline --shard "$s/$RL_WORKERS" \
      >> "$LOGS/baseline-$BASELINE_NAME-${SLURM_JOB_ID:-manual}-shard$s.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "baseline shard failed (pid $pid)" >&2; exit 1; }; done
fi
"$PY" -m psse_env.rl.train --config "$CFG" --phase baseline
"$PY" -m psse_env.rl.train --config "$CFG" --status
echo "cec-baseline-$BASELINE_NAME-complete"
