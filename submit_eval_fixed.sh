#!/bin/bash
#SBATCH --job-name=gpt_oss_eval_fixed
#SBATCH --output=/scratch/yx3882/psse_agent/logs/eval_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/eval_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

REPO_ROOT=/scratch/yx3882/psse_agent
LOG_DIR=$REPO_ROOT/logs
CACHE_ROOT=/scratch/yx3882/.cache

ADAPTER_PATH=${ADAPTER_PATH:-harshith0214/psse-agent-gpt-oss-20b}
TEST_FILE=${TEST_FILE:-out_traces_balanced/sft_traces.test.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_TURNS=${MAX_TURNS:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gpt_oss_sft_power_agent/eval_${SLURM_JOB_ID}.jsonl}
VERBOSE=${VERBOSE:-1}
CONTINUE_ON_TOOL_ERROR=${CONTINUE_ON_TOOL_ERROR:-0}
SMOKE=${SMOKE:-0}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-20}
EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS:-}

mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_ROOT/huggingface"
mkdir -p "$CACHE_ROOT/torch"

export HF_HOME=$CACHE_ROOT/huggingface
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

cd "$REPO_ROOT"

if [[ "$SMOKE" == "1" && -z "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES="$SMOKE_SAMPLES"
fi

echo "===== Eval diagnostics ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "env python: $PYTHON"
echo "adapter: $ADAPTER_PATH"
echo "test file: $TEST_FILE"
echo "output: $OUTPUT_FILE"
echo "smoke mode: $SMOKE"
echo "max samples: ${MAX_SAMPLES:-ALL}"
echo "max turns: $MAX_TURNS"
echo "max new tokens: $MAX_NEW_TOKENS"
echo "max seq length: $MAX_SEQ_LENGTH"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
$PYTHON -V
$PYTHON -m pip list | grep -E "unsloth|scipy|transformers|torch" || true
nvidia-smi
echo "============================"

echo "===== Python preflight ====="
"$PYTHON" - <<'PY'
import importlib
mods = ["scipy", "torch", "transformers"]
failed = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK ({getattr(mod, '__version__', 'unknown')})")
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"{name}: FAIL ({exc})")

try:
    from mcp_server.matpower_server import wls_from_path  # noqa: F401
    print("mcp_server.matpower_server: OK")
except Exception as exc:
    failed.append(("mcp_server.matpower_server", str(exc)))
    print(f"mcp_server.matpower_server: FAIL ({exc})")

if failed:
    raise SystemExit("Preflight failed: " + ", ".join(f"{name} -> {err}" for name, err in failed))
PY
echo "==========================="

ARGS=(
  eval_sft_agent_fixed.py
  --adapter "$ADAPTER_PATH"
  --test-file "$TEST_FILE"
  --max-turns "$MAX_TURNS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --output "$OUTPUT_FILE"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$VERBOSE" == "1" ]]; then
  ARGS+=(--verbose)
fi

if [[ "$CONTINUE_ON_TOOL_ERROR" == "1" ]]; then
  ARGS+=(--continue-on-tool-error)
fi

if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_EVAL_ARGS)
  ARGS+=("${EXTRA_ARR[@]}")
fi

"$PYTHON" "${ARGS[@]}"
