#!/usr/bin/env bash
#SBATCH --job-name=fp-eval
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
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/eval-%j.out
# Full pipeline, paired evaluation for PIPELINE_ROUND (r1 or r2): the round's
# student against its candidate on the shared development suite, then the
# round summary; after round 2 the pipeline summary joins BC0, R1, and R2.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
ROUND=${PIPELINE_ROUND:?PIPELINE_ROUND (r1 or r2) must be exported by the submitter}
round_bindings "$ROUND"
echo "[eval:$ROUND] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
[[ -s "$ROUND_DIR/training.done" ]] || { echo "training for $ROUND has not completed" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-eval-$ROUND-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-eval-$ROUND-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
opendss_local_model
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

cd "$SRC"
collection_args "$ROUND"
# Round 1 also rolls the teacher out on the development roots: its score is
# the ceiling every student is read against, per family and per stratum.
expert_args=()
[[ "$ROUND" == r1 ]] && expert_args=(--eval-expert)
# One rollout process leaves the GPU idle while the environment steps on the
# CPU (48-60% utilization, under the cluster's 60% H100/H200 cancellation
# line), so every policy is rolled out at once by worker processes sharing the
# GPU: two per policy when the card holds four ~15 GB copies, else one.  The
# workers checkpoint every episode, so a requeued job resumes where it stopped.
if [[ -z "${EVAL_WORKERS_PER_POLICY:-}" ]]; then
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  EVAL_WORKERS_PER_POLICY=1
  (( ${gpu_mib:-0} >= 70000 )) && EVAL_WORKERS_PER_POLICY=2
fi
echo "[eval:$ROUND] $EVAL_WORKERS_PER_POLICY worker(s) per policy"
if [[ ! -s "$COLLECTION_DIR/evaluation/comparison.json" ]]; then
  "$PY" scripts/run_dagger_research.py "${COLLECTION_ARGS[@]}" \
    --eval-r1-adapter "$CANDIDATE" "${expert_args[@]}" \
    --eval-workers-per-policy "$EVAL_WORKERS_PER_POLICY" > "$ROUND_DIR/evaluation_console.json"
fi
"$PY" "$PIPE/summarize.py" round --round-dir "$ROUND_DIR" --round "$ROUND" --output "$ROUND_DIR/round_summary.json"
if [[ "$ROUND" == r2 ]]; then
  "$PY" "$PIPE/summarize.py" pipeline --out-dir "$OUT" --output "$OUT/pipeline_summary.json"
fi
echo "fp-eval-$ROUND-complete"
