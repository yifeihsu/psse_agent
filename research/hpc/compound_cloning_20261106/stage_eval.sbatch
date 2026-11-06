#!/usr/bin/env bash
#SBATCH --job-name=cc-eval
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
#SBATCH --output=/scratch/yx3882/compound_cloning_20261106/logs/%x-%j.out
# One adapter on one saved suite, greedy decoding, in the environment the rows were made in:
# research/hpc/full_pipeline_20260907/stage_eval_suite.sbatch for this cell.  EVAL_SPEC names a file
# setting EVAL_NAME, EVAL_SUITE, EVAL_SEED, EVAL_POLICIES ("label=adapter") and, for a smoke,
# EVAL_SLICE_FROM (the suite the slice EVAL_SUITE is cut from).  submit_cell.sh writes them.
#
# evaluate_suite.py is unchanged: env_report.py reads the suite's system from its roots and writes
# the report evaluate_suite.py reads under $ENV_DIR/SYSTEM, whose path goes in as --cell-dir, and
# this script sets the conditioning guard for that system.  Worker processes share the GPU and
# checkpoint every episode, so a requeued job resumes where it stopped, and so does the next link
# of a chain (--requeue does not restart a job that reaches its time limit).  No OpenDSS copy to
# node-local memory as the pipeline stages make: scada_only refuses every OpenDSS-backed tool.
set -euo pipefail
WORK=/scratch/yx3882/compound_cloning_20261106
# shellcheck source=cell.env
source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"
cell_environment
: "${EVAL_SPEC:?EVAL_SPEC must name the evaluation spec file}"
EVAL_SLICE_FROM=
# shellcheck source=/dev/null
source "$EVAL_SPEC"
: "${EVAL_NAME:?}" "${EVAL_SUITE:?}" "${EVAL_POLICIES:?}" "${EVAL_SEED:?}"
RESULT=$EVAL_OUT/$EVAL_NAME
echo "[eval:$EVAL_NAME] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
if [[ -s "$RESULT/summary.json" && -s "$RESULT/outcomes.json" ]]; then
  echo "[eval:$EVAL_NAME] results present; nothing to do"
  echo "cc-eval-$EVAL_NAME-complete"
  exit 0
fi
if another_link_running; then
  echo "[eval:$EVAL_NAME] another link of $SLURM_JOB_NAME is running; leaving it the output directory" >&2
  exit 3
fi
cd "$SRC"
if [[ -n "$EVAL_SLICE_FROM" && ! -s "$EVAL_SUITE" ]]; then
  "$PY" "$CELL/smoke_inputs.py" suite --suite "$EVAL_SLICE_FROM" --roots "$SMOKE_EVAL_ROOTS" --output "$EVAL_SUITE"
fi
[[ -s "$EVAL_SUITE" ]] || { echo "suite missing: $EVAL_SUITE" >&2; exit 2; }
mkdir -p "$EVAL_OUT" "$ENV_DIR"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-eval-$EVAL_NAME-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-eval-$EVAL_NAME-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap stage_cleanup EXIT

# The report for the suite's system (refuses an environment that differs from the rows'), then the
# conditioning guard of that system, which the spawned workers inherit.
read -r EVAL_SYSTEM EVAL_RCOND < <("$PY" "$CELL/env_report.py" --suite "$EVAL_SUITE" --env-root "$ENV_DIR" | tail -1)
if [[ "$EVAL_RCOND" == default ]]; then unset PSSE_MULTI_SCAN_RCOND; else export PSSE_MULTI_SCAN_RCOND=$EVAL_RCOND; fi
echo "[eval:$EVAL_NAME] system $EVAL_SYSTEM, multi-scan conditioning guard ${PSSE_MULTI_SCAN_RCOND:-dense_svd}"
# One policy per job here, so the four copies an 80 GB card holds (~15 GB each) all roll it out.
if [[ -z "${EVAL_WORKERS_PER_POLICY:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1' | tr -dc '0-9')
  EVAL_WORKERS_PER_POLICY=2
  if (( ${gpu_mib:-0} >= 70000 )); then EVAL_WORKERS_PER_POLICY=4; fi
fi
echo "[eval:$EVAL_NAME] $EVAL_WORKERS_PER_POLICY worker(s) per policy"
if [[ ! -s "$RESULT/summary.json" ]]; then
  policy_args=()
  for policy in $EVAL_POLICIES; do
    adapter=${policy#*=}
    [[ -s "$adapter/adapter_model.safetensors" ]] || { echo "adapter missing: $adapter" >&2; exit 2; }
    policy_args+=(--policy "$policy")
  done
  "$PY" research/hpc/full_pipeline_20260907/evaluate_suite.py --cell-dir "$ENV_DIR/$EVAL_SYSTEM" --suite "$EVAL_SUITE" \
    --output-dir "$EVAL_OUT" --name "$EVAL_NAME" "${policy_args[@]}" \
    --seed "$EVAL_SEED" --max-steps "$EVAL_MAX_STEPS" --workers-per-policy "$EVAL_WORKERS_PER_POLICY" \
    > "$EVAL_OUT/$EVAL_NAME.console.json"
fi
# Per-episode records with the evaluator's decline_accounting and the spec's measures.
"$PY" "$CELL/episode_outcomes.py" --suite "$EVAL_SUITE" --evaluation-dir "$RESULT"
echo "cc-eval-$EVAL_NAME-complete"
