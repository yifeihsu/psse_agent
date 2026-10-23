#!/usr/bin/env bash
#SBATCH --job-name=fp-xeval
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
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/xeval-%j.out
# One or two fixed adapters (and the teacher) on a saved suite, without a
# collection (evaluate_suite.py; the conventional-only cell's follow-ups,
# 2026-10-23).  EVAL_SPEC names a file setting EVAL_NAME, EVAL_SUITE,
# EVAL_POLICIES ("label=adapter" words) and EVAL_EXPERT (0 or 1); the results
# go to $OUT/extra/evaluations/$EVAL_NAME.  Worker processes share the GPU as
# in stage_eval.sbatch, and a requeued job resumes from their per-episode
# checkpoints.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
: "${EVAL_SPEC:?EVAL_SPEC must name the evaluation spec file}"
# shellcheck source=/dev/null
source "$EVAL_SPEC"
: "${EVAL_NAME:?}" "${EVAL_SUITE:?}" "${EVAL_POLICIES:?}"
EVAL_OUT=$OUT/extra/evaluations
echo "[xeval:$EVAL_NAME] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$EVAL_SUITE" ]] || { echo "suite missing: $EVAL_SUITE" >&2; exit 2; }
mkdir -p "$EVAL_OUT"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-xeval-$EVAL_NAME-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-xeval-$EVAL_NAME-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
opendss_local_model
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

cd "$SRC"
if [[ -z "${EVAL_WORKERS_PER_POLICY:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  EVAL_WORKERS_PER_POLICY=1
  (( ${gpu_mib:-0} >= 70000 )) && EVAL_WORKERS_PER_POLICY=2
fi
echo "[xeval:$EVAL_NAME] $EVAL_WORKERS_PER_POLICY worker(s) per policy"
if [[ ! -s "$EVAL_OUT/$EVAL_NAME/summary.json" ]]; then
  policy_args=()
  for policy in $EVAL_POLICIES; do
    adapter=${policy#*=}
    [[ -s "$adapter/adapter_model.safetensors" ]] || { echo "adapter missing: $adapter" >&2; exit 2; }
    policy_args+=(--policy "$policy")
  done
  expert_args=()
  [[ "${EVAL_EXPERT:-0}" == 1 ]] && expert_args=(--expert)
  "$PY" research/hpc/full_pipeline_20260907/evaluate_suite.py --cell-dir "$PIPE" --suite "$EVAL_SUITE" \
    --output-dir "$EVAL_OUT" --name "$EVAL_NAME" "${policy_args[@]}" "${expert_args[@]}" \
    --seed "$SEED_ROUND" --max-steps "$EVAL_MAX_STEPS" --workers-per-policy "$EVAL_WORKERS_PER_POLICY" \
    > "$EVAL_OUT/$EVAL_NAME.console.json"
fi
echo "fp-xeval-$EVAL_NAME-complete"
