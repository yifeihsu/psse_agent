#!/usr/bin/env bash
#SBATCH --job-name=rl-parity
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/rl_pilot_20261014/logs/%x-%j.out
# Sampling-versus-training parity of stored rollouts (psse_env.rl.parity):
# every completion token of an iteration's training rows scored under the
# sampler's record, generate's cached decode path, the full forward of the
# sampling loader and the full forward of the training loader, on one card.
#
#   sbatch --export=ALL,RUNS="rl_hard_screen_bf16_smoke.json:smoke_bf16/iteration_000 rl_hard_screen_fix_smoke.json:smoke_fix/iteration_000" parity.sbatch
#
# RUNS lists config:iteration pairs (configs under $WORK/configs, iterations
# under $WORK/out); each writes $WORK/out/<iteration>/parity.json and is
# skipped when that file already exists, so a requeue continues.
set -euo pipefail
WORK=/scratch/yx3882/rl_pilot_20261014
RUNS=${RUNS:?set RUNS to config:iteration pairs}
SRC=$WORK/source
LOGS=$WORK/logs
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/scratch/yx3882/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
export RESEARCH_MAX_INPUT_TOKENS=32768 RESEARCH_MAX_NEW_TOKENS=256 PSSE_LOCAL_DIAGNOSTIC_BUILD=1

mkdir -p "$LOGS"
echo "[parity] runs $RUNS job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[parity] source $(git -C "$SRC" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$SRC"
for pair in $RUNS; do
  config=${pair%%:*}; iteration=${pair##*:}
  out="$WORK/out/$iteration/parity.json"
  if [[ -s "$out" ]]; then echo "[parity] $iteration already done"; continue; fi
  [[ -s "$WORK/configs/$config" ]] || { echo "no config $WORK/configs/$config" >&2; exit 2; }
  echo "[parity] $config $iteration $(date -u +%FT%TZ)"
  "$PY" -m psse_env.rl.parity --config "$WORK/configs/$config" --iteration "$WORK/out/$iteration" --output "$out" ${PARITY_ARGS:-}
  echo "[parity] wrote $out $(date -u +%FT%TZ)"
done
echo "[parity] done $(date -u +%FT%TZ)"
