#!/usr/bin/env bash
#SBATCH --job-name=rli-probe
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
# Stage 2 of the RL improvement series (docs/rl_improvement_workplan_20261110.md): the fork probe.
# From states the cloned learner reaches on its own, both lifecycle choices are forced and continued
# by the learner, under the bounded and under the complete history (research.rl.fork_probe).
#
#   sbatch fork_probe.sbatch                                   # the whole probe; a resubmission continues it
#   sbatch --export=ALL,CONTINUATIONS=4 fork_probe.sbatch      # the predeclared expansion (768 suffix episodes at most)
#   sbatch --export=ALL,PANEL_ADAPTER=$WORK/out/calibration_1step/iteration_000/adapter,PANEL_LABEL=cal1 fork_probe.sbatch
#       # stage 0's decision panel: the same forks, the policy's choices before (S0) and after an update
#
# Every step keeps what is done: rollouts by root, branches by file, the fork sample once it is frozen
# (forks.json is never rewritten: delete it by hand to draw another sample), views and decisions by file.
# The rule expert's own episodes on the probe's roots run on the CPUs beside the learner's rollouts;
# they say which roots it fails, which is what the screening gate counts.
set -euo pipefail
WORK=/scratch/yx3882/rl_improvement_20261110
SRC=$WORK/source
LOGS=$WORK/logs
PROBE=${PROBE:-$WORK/out/fork_probe}
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python
LEARNER_CONFIG=${LEARNER_CONFIG:-arm_a_bounded.json}
EXPERT_CONFIG=${EXPERT_CONFIG:-probe_expert.json}
CONTINUATIONS=${CONTINUATIONS:-2}
SAMPLES=${SAMPLES:-2}
PANEL_ADAPTER=${PANEL_ADAPTER:-}
PANEL_LABEL=${PANEL_LABEL:-}
CFG="$WORK/configs/$LEARNER_CONFIG"
ECFG="$WORK/configs/$EXPERT_CONFIG"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/scratch/yx3882/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
export RESEARCH_MAX_NEW_TOKENS=256 PSSE_LOCAL_DIAGNOSTIC_BUILD=1 PSSE_HIF_WORKERS=4
export PSSE_EXPERT_VARIANT=baseline_compare
unset PSSE_MULTI_SCAN_RCOND

mkdir -p "$LOGS" "$PROBE"
for file in "$CFG" "$ECFG" "$WORK/configs/reserved_roots.json" "$WORK/configs/probe_training_roots.json"; do
  [[ -s "$file" ]] || { echo "missing $file" >&2; exit 2; }
done
cd "$SRC"
config_field() { "$PY" -c 'import json,sys; value=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if value is None else value)' "$CFG" "$1"; }
RESEARCH_MAX_INPUT_TOKENS=$(config_field max_input_tokens)
export RESEARCH_MAX_INPUT_TOKENS=${RESEARCH_MAX_INPUT_TOKENS:-32768}
TRAIN=$(config_field training_suite)
echo "[probe] learner $LEARNER_CONFIG expert $EXPERT_CONFIG continuations $CONTINUATIONS job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} $(date -u +%FT%TZ)"
echo "[probe] source $(git -C "$SRC" rev-parse HEAD) prompt limit $RESEARCH_MAX_INPUT_TOKENS tokens"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-fork_probe-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT
if [[ -z "${RL_WORKERS:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  RL_WORKERS=1
  (( ${gpu_mib:-0} >= 70000 )) && RL_WORKERS=2
fi
probe() { "$PY" -m research.rl.fork_probe "$@"; }
log() { echo "$LOGS/probe-$1-${SLURM_JOB_ID:-manual}.out"; }

if [[ -n "$PANEL_ADAPTER" ]]; then
  # The fixed decision panel of stage 0: the frozen forks, scored by another adapter, beside the start adapter's scores.
  [[ -n "$PANEL_LABEL" ]] || { echo "PANEL_ADAPTER needs PANEL_LABEL" >&2; exit 2; }
  for file in forks.json views.json decisions.json; do
    [[ -s "$PROBE/$file" ]] || { echo "the probe has no $file yet: run the probe first" >&2; exit 2; }
  done
  probe decide --manifest "$PROBE/forks.json" --views "$PROBE/views.json" --config "$CFG" --adapter "$PANEL_ADAPTER" \
    --output "$PROBE/decisions_$PANEL_LABEL.json"
  probe panel --manifest "$PROBE/forks.json" --before "$PROBE/decisions.json" --after "$PROBE/decisions_$PANEL_LABEL.json" \
    --output "$PROBE/panel_$PANEL_LABEL" --title "Decision panel: the start adapter and $PANEL_LABEL"
  echo "rli-probe-complete"
  exit 0
fi

# 1. The rule expert on the probe's roots (CPU, beside the learner's rollouts).
( probe rollout --config "$ECFG" --suite "$TRAIN" --roots-file "$WORK/configs/probe_training_roots.json" \
    --store "$PROBE/expert_training" --samples 1 >> "$(log expert)" 2>&1
  probe rollout --config "$ECFG" --suite "$TRAIN" --roots-file "$WORK/configs/reserved_roots.json" \
    --store "$PROBE/expert_validation" --samples 1 >> "$(log expert)" 2>&1 ) &
EXPERT_PID=$!

# 2. The learner's own episodes on both pools: the states the forks are taken from.
echo "[probe] learner rollouts $(date -u +%FT%TZ)"
rollout_pool() {
  probe rollout --config "$CFG" --suite "$TRAIN" --roots-file "$WORK/configs/$1" --store "$PROBE/$2" --samples "$SAMPLES"
}
if (( RL_WORKERS > 1 )); then
  rollout_pool probe_training_roots.json training_store >> "$(log rollout-training)" 2>&1 &
  first=$!
  rollout_pool reserved_roots.json validation_store >> "$(log rollout-validation)" 2>&1 &
  second=$!
  wait "$first" || { echo "the training pool's rollouts failed" >&2; exit 1; }
  wait "$second" || { echo "the validation pool's rollouts failed" >&2; exit 1; }
else
  rollout_pool probe_training_roots.json training_store
  rollout_pool reserved_roots.json validation_store
fi
wait "$EXPERT_PID" || { echo "the rule expert's rollouts failed (see $(log expert))" >&2; exit 1; }

# 3. Which roots the rule expert fails, then the frozen sample: one fork per root, by kind, shortfalls recorded.
probe reference --store "$PROBE/expert_training" --store "$PROBE/expert_validation" --output "$PROBE/expert_reference.json"
if [[ -s "$PROBE/forks.json" ]]; then
  echo "[probe] the fork sample is frozen ($PROBE/forks.json); it is not drawn again"
else
  probe select --store "$PROBE/training_store" --validation-store "$PROBE/validation_store" --suite "$TRAIN" \
    --expert-reference "$PROBE/expert_reference.json" --output "$PROBE/forks.json"
fi

# 4. What the learner is shown at each fork in each mode (a replay on the CPU), then its choice there: 48 states x 2 views.
[[ -s "$PROBE/views.json" ]] || probe views --manifest "$PROBE/forks.json" --suite "$TRAIN" --config "$CFG" --output "$PROBE/views.json"
[[ -s "$PROBE/decisions.json" ]] || probe decide --manifest "$PROBE/forks.json" --views "$PROBE/views.json" --config "$CFG" \
  --output "$PROBE/decisions.json"

# 5. Both choices, both modes, CONTINUATIONS learner continuations each.
echo "[probe] branches $(date -u +%FT%TZ)"
if (( RL_WORKERS > 1 )); then
  pids=()
  for (( s = 0; s < RL_WORKERS; s++ )); do
    probe branch --manifest "$PROBE/forks.json" --suite "$TRAIN" --config "$CFG" --output-dir "$PROBE/branches" \
      --continuations "$CONTINUATIONS" --shard "$s/$RL_WORKERS" >> "$(log branch-shard$s)" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "a branch shard failed (pid $pid)" >&2; exit 1; }; done
fi
probe branch --manifest "$PROBE/forks.json" --suite "$TRAIN" --config "$CFG" --output-dir "$PROBE/branches" --continuations "$CONTINUATIONS"

# 6. The paired decision and continuation report, with the screening gate.
probe report --manifest "$PROBE/forks.json" --branches "$PROBE/branches" --decisions "$PROBE/decisions.json" --output "$PROBE/report"
echo "rli-probe-complete"
