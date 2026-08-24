#!/usr/bin/env bash
#SBATCH --job-name=gemma4-research-smoke
#SBATCH --account=torch_pr_627_general
# A live 6.5k-token E4B optimizer step peaked at 45.0/46.1 GB and then needed
# another 6.4 GB, so L40S is outside this full-length smoke's measured envelope.
# Default to NYU's 96-GB RTX Pro 6000 preemptible route; callers can override
# this directive with --constraint=h100 or --constraint=h200 when appropriate.
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# NYU admits the generic RTX partition only when the job explicitly opts into
# preemption. Slurm requeues the same job, and the smoke safely reuses a saved
# adapter or restarts its bounded 20-step training stage when none was saved.
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --open-mode=append
#SBATCH --output=gemma4_research_smoke_%j.out
#SBATCH --error=gemma4_research_smoke_%j.err

set -Eeuo pipefail

MODEL_CHOICE="${1:?usage: sbatch submit_research_gemma4_smoke.sh e4b|12b}"
case "$MODEL_CHOICE" in
  e4b|12b) ;;
  *) echo "model choice must be e4b or 12b" >&2; exit 2 ;;
esac

: "${RESEARCH_SOURCE_ROOT:?set RESEARCH_SOURCE_ROOT to the exact committed checkout}"
: "${RESEARCH_ENV:?set RESEARCH_ENV to the separate research conda environment}"
: "${RESEARCH_RUN_ROOT:?set RESEARCH_RUN_ROOT to the persistent run directory}"
: "${RESEARCH_CLOSED_LOOP_SUITE:?set RESEARCH_CLOSED_LOOP_SUITE to the small suite JSON}"

VIEW_ROOT="$RESEARCH_RUN_ROOT/views"
PROCESSOR_AUDIT="$RESEARCH_RUN_ROOT/processor_audit/processor_audit.json"
TRAIN_FILE="$VIEW_ROOT/smoke.train16.jsonl"
VALIDATION_FILE="$VIEW_ROOT/smoke.validation8.jsonl"
PROBE_FILE="$VIEW_ROOT/smoke.probes10.jsonl"
OUTPUT_DIR="$RESEARCH_RUN_ROOT/smoke/$MODEL_CHOICE"
for required in \
  "$RESEARCH_SOURCE_ROOT/psse_env/requirements-sft-research.txt" \
  "$PROCESSOR_AUDIT" \
  "$TRAIN_FILE" \
  "$VALIDATION_FILE" \
  "$PROBE_FILE" \
  "$RESEARCH_CLOSED_LOOP_SUITE"
do
  if [[ ! -f "$required" ]]; then
    echo "required input is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.job.lock"
if ! flock -n 9; then
  echo "another $MODEL_CHOICE smoke owns $OUTPUT_DIR" >&2
  exit 73
fi

module purge
module load anaconda3/2025.06
# shellcheck source=/dev/null
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$RESEARCH_ENV"
cd "$RESEARCH_SOURCE_ROOT"

python - "$PROCESSOR_AUDIT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
models = report.get("models", {})
if report.get("passed") is not True or any(
    models.get(name, {}).get("passed") is not True for name in ("e4b", "12b")
):
    raise SystemExit(f"both live processor audits must pass before GPU work: {path}")
print({"processor_audit": str(path), "models": sorted(models), "passed": True})
PY

python -m pip check
python - <<'PY'
import importlib.metadata
import pathlib
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"research lock requires Python 3.12, got {sys.version}")
lock = pathlib.Path("psse_env/requirements-sft-research.txt")
mismatches = []
for raw in lock.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if "==" not in line:
        raise SystemExit(f"unpinned research requirement: {line}")
    name, expected = line.split("==", 1)
    observed = importlib.metadata.version(name)
    if observed != expected:
        mismatches.append(f"{name}: expected {expected}, got {observed}")
if mismatches:
    raise SystemExit("research dependency mismatch: " + "; ".join(mismatches))

import torch

if not torch.cuda.is_available() or not torch.version.cuda:
    raise SystemExit("research smoke requires a CUDA-enabled PyTorch build")
print(
    {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
)
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RESEARCH_SOURCE_ROOT"
export TOKENIZERS_PARALLELISM=false
# Do not inherit a login-shell HF_HOME that points somewhere other than the
# CPU-verified snapshot cache.  A caller may override the research cache
# explicitly, but generic ambient Hugging Face variables are not authoritative.
export HF_HOME="${RESEARCH_HF_HOME:-/scratch/yx3882/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="${RESEARCH_TORCH_HOME:-/scratch/yx3882/.cache/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export RESEARCH_MAX_INPUT_TOKENS=16384
export RESEARCH_MAX_NEW_TOKENS=256
export PYTORCH_ALLOC_CONF="${RESEARCH_PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python -m psse_env.sft research-cache \
  --model-choice "$MODEL_CHOICE" \
  --output "$OUTPUT_DIR/cache_preflight.json"

TELEMETRY="$OUTPUT_DIR/nvidia_smi_${SLURM_JOB_ID}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits \
  --loop=10 >"$TELEMETRY" 2>"$OUTPUT_DIR/nvidia_smi_${SLURM_JOB_ID}.err" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -m psse_env.sft research-smoke \
  --model-choice "$MODEL_CHOICE" \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --probe-file "$PROBE_FILE" \
  --closed-loop-suite "$RESEARCH_CLOSED_LOOP_SUITE" \
  --output-dir "$OUTPUT_DIR" \
  --max-length 16384 \
  --overfit-steps 20 \
  --learning-rate 1e-4 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --closed-loop-scenarios 3 \
  --closed-loop-max-steps 8 \
  --seed 20260720
