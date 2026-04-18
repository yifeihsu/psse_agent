#!/bin/bash
#SBATCH --job-name=gemma4_eval
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

# GPU selection is intentionally left configurable at submission time.
# Examples:
#   sbatch --constraint=a100 submit_eval.sh
#   sbatch --constraint=l40s submit_eval.sh
#   sbatch --gres=gpu:rtx_pro_6000:1 submit_eval.sh
# Use your cluster's exact GPU labels from `sinfo -o "%P %G %f"`.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

REPO_ROOT=/scratch/yx3882/psse_agent
LOG_DIR=$REPO_ROOT/logs
CACHE_ROOT=/scratch/yx3882/.cache

ADAPTER_PATH=${ADAPTER_PATH:-outputs/gemma4_power_agent/lora}
TEST_FILE=${TEST_FILE:-out_traces_balanced/sft_traces.test.jsonl}
MODEL_REVISION=${MODEL_REVISION:-d722512f8f1e4ef6629c1b24d16d65295c8c945e}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_TURNS=${MAX_TURNS:-6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gemma4_power_agent/eval_${SLURM_JOB_ID}.jsonl}
VERBOSE=${VERBOSE:-1}
CONTINUE_ON_TOOL_ERROR=${CONTINUE_ON_TOOL_ERROR:-0}
SMOKE=${SMOKE:-0}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-20}
INCLUDE_TOOL_SCHEMAS=${INCLUDE_TOOL_SCHEMAS:-1}
INJECT_EMPTY_THOUGHT_CHANNEL=${INJECT_EMPTY_THOUGHT_CHANNEL:-1}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-0}
LOAD_IN_16BIT=${LOAD_IN_16BIT:-1}
CONCURRENT_CONVERSATIONS=${CONCURRENT_CONVERSATIONS:-4}
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
mkdir -p "$(dirname "$OUTPUT_FILE")"

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
echo "model revision: ${MODEL_REVISION:-UNPINNED}"
echo "smoke mode: $SMOKE"
echo "max samples: ${MAX_SAMPLES:-ALL}"
echo "max turns: $MAX_TURNS"
echo "max new tokens: $MAX_NEW_TOKENS"
echo "max seq length: $MAX_SEQ_LENGTH"
echo "include tool schemas: $INCLUDE_TOOL_SCHEMAS"
echo "inject empty thought: $INJECT_EMPTY_THOUGHT_CHANNEL"
echo "load_in_4bit/load_in_16bit: $LOAD_IN_4BIT / $LOAD_IN_16BIT"
echo "concurrent conversations: $CONCURRENT_CONVERSATIONS"
echo "slurm gres: ${SLURM_JOB_GRES:-unknown}"
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
  eval_sft_agent_gemma_v2.py
  --adapter "$ADAPTER_PATH"
  --test-file "$TEST_FILE"
  --max-turns "$MAX_TURNS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --concurrent-conversations "$CONCURRENT_CONVERSATIONS"
  --output "$OUTPUT_FILE"
)

if [[ -n "$MODEL_REVISION" ]]; then
  ARGS+=(--model-revision "$MODEL_REVISION")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$VERBOSE" == "1" ]]; then
  ARGS+=(--verbose)
fi

if [[ "$CONTINUE_ON_TOOL_ERROR" == "1" ]]; then
  ARGS+=(--continue-on-tool-error)
fi

if [[ "$INCLUDE_TOOL_SCHEMAS" == "1" ]]; then
  ARGS+=(--include-tool-schemas)
else
  ARGS+=(--no-include-tool-schemas)
fi

if [[ "$INJECT_EMPTY_THOUGHT_CHANNEL" == "1" ]]; then
  ARGS+=(--inject-empty-thought-channel)
else
  ARGS+=(--no-inject-empty-thought-channel)
fi

if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  ARGS+=(--load-in-4bit)
else
  ARGS+=(--no-load-in-4bit)
fi

if [[ "$LOAD_IN_16BIT" == "1" ]]; then
  ARGS+=(--load-in-16bit)
else
  ARGS+=(--no-load-in-16bit)
fi

if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_EVAL_ARGS)
  ARGS+=("${EXTRA_ARR[@]}")
fi

"$PYTHON" "${ARGS[@]}"
