#!/usr/bin/env bash
#SBATCH --job-name=rl-parity-probe
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/rl_pilot_20261014/logs/%x-%j.out
# The parity probe (psse_env.rl.parity_probe): the first completion token of
# a parity run's worst rows under variants that differ in one thing each.
#
#   sbatch --export=ALL,RUNS="rl_hard_screen_bf16_smoke.json:smoke_bf16/iteration_000" parity_probe.sbatch
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
echo "[probe] runs $RUNS job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[probe] source $(git -C "$SRC" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$SRC"
for pair in $RUNS; do
  config=${pair%%:*}; iteration=${pair##*:}
  parity="$WORK/out/$iteration/parity.json"
  out="$WORK/out/$iteration/parity_probe.json"
  if [[ -s "$out" ]]; then echo "[probe] $iteration already done"; continue; fi
  [[ -s "$parity" ]] || { echo "no parity.json for $iteration" >&2; exit 2; }
  echo "[probe] $config $iteration $(date -u +%FT%TZ)"
  "$PY" -m psse_env.rl.parity_probe --config "$WORK/configs/$config" --iteration "$WORK/out/$iteration" --parity "$parity" --output "$out" ${PROBE_ARGS:-}
  echo "[probe] wrote $out $(date -u +%FT%TZ)"
done
echo "[probe] done $(date -u +%FT%TZ)"
