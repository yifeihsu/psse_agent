#!/bin/bash
#SBATCH --job-name=gemma4_eval
#SBATCH --output=/scratch/yx3882/psse_agent/artifacts/logs/eval_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/artifacts/logs/eval_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --mail-user=yx3882@nyu.edu

# GPU selection is intentionally left configurable at submission time.
# Examples:
#   sbatch --constraint=a100 submit_eval_v3.sh
#   sbatch --constraint=l40s submit_eval_v3.sh
#   sbatch --gres=gpu:rtx_pro_6000:1 submit_eval_v3.sh
# Use your cluster's exact GPU labels from `sinfo -o "%P %G %f"`.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

REPO_ROOT=/scratch/yx3882/psse_agent
LOG_DIR=$REPO_ROOT/artifacts/logs
CACHE_ROOT=/scratch/yx3882/.cache

GPU_PROFILE=${GPU_PROFILE:-auto}
ADAPTER_PATH=${ADAPTER_PATH:-outputs/gemma4_power_agent/lora}
TEST_FILE=${TEST_FILE:-artifacts/traces/out_traces_balanced/sft_traces.test.jsonl}
EVAL_SCRIPT=${EVAL_SCRIPT:-eval_sft_agent_gemma_v4.py}
MODEL_REVISION=${MODEL_REVISION:-d722512f8f1e4ef6629c1b24d16d65295c8c945e}
TOOL_SCOPE=${TOOL_SCOPE:-scada_harmonic}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_TURNS=${MAX_TURNS:-6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/gemma4_power_agent/eval_${SLURM_JOB_ID}.jsonl}
VERBOSE=${VERBOSE:-0}
CONTINUE_ON_TOOL_ERROR=${CONTINUE_ON_TOOL_ERROR:-0}
SMOKE=${SMOKE:-0}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-20}
INCLUDE_TOOL_SCHEMAS=${INCLUDE_TOOL_SCHEMAS:-1}
INJECT_EMPTY_THOUGHT_CHANNEL=${INJECT_EMPTY_THOUGHT_CHANNEL:-1}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-}
LOAD_IN_16BIT=${LOAD_IN_16BIT:-}
CONCURRENT_CONVERSATIONS=${CONCURRENT_CONVERSATIONS:-}
GC_COLLECT_EVERY_N_TURNS=${GC_COLLECT_EVERY_N_TURNS:-0}
EMPTY_CUDA_CACHE_EVERY_N_TURNS=${EMPTY_CUDA_CACHE_EVERY_N_TURNS:-0}
PREFER_BASE_TOKENIZER=${PREFER_BASE_TOKENIZER:-1}
# Keep eval prompts distribution-compatible with SFT by default. These eval-only
# mutations are opt-in diagnostics because they can change first-turn routing.
FILTER_UNAVAILABLE_HELPER_TOOLS=${FILTER_UNAVAILABLE_HELPER_TOOLS:-0}
INJECT_RUNTIME_HELPER_NOTE=${INJECT_RUNTIME_HELPER_NOTE:-0}
CONTINUE_ON_MISSING_CONTEXT_TOOL=${CONTINUE_ON_MISSING_CONTEXT_TOOL:-1}
ROLLING_BATCH_SCHEDULER=${ROLLING_BATCH_SCHEDULER:-1}
RESUME_OUTPUT=${RESUME_OUTPUT:-1}
TRUNCATE_PARTIAL_OUTPUT=${TRUNCATE_PARTIAL_OUTPUT:-1}
ALLOW_SLOW_PANDAPOWER=${ALLOW_SLOW_PANDAPOWER:-0}
CPU_THREADS=${CPU_THREADS:-${SLURM_CPUS_PER_TASK:-16}}
EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS:-}

mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_ROOT/huggingface"
mkdir -p "$CACHE_ROOT/torch"

set_default_if_unset() {
  local var_name=$1
  local default_value=$2
  if [[ -z "${!var_name:-}" ]]; then
    printf -v "$var_name" '%s' "$default_value"
  fi
}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
GPU_PROFILE_SELECTED=$GPU_PROFILE
if [[ "$GPU_PROFILE" == "auto" ]]; then
  if [[ ( "$GPU_NAME" == *"H200"* || "$GPU_NAME" == *"H100"* || "$GPU_NAME" == *"A100"* ) && "${GPU_MEM_MB:-0}" -ge 120000 ]]; then
    GPU_PROFILE_SELECTED="ultrahighmem-accelerator"
    set_default_if_unset MAX_SEQ_LENGTH 6144
    set_default_if_unset LOAD_IN_4BIT 0
    set_default_if_unset LOAD_IN_16BIT 1
    set_default_if_unset CONCURRENT_CONVERSATIONS 8
  elif [[ "$GPU_NAME" == *"L40S"* && "${GPU_MEM_MB:-0}" -ge 43000 ]]; then
    GPU_PROFILE_SELECTED="l40s"
    set_default_if_unset MAX_SEQ_LENGTH 4096
    set_default_if_unset LOAD_IN_4BIT 0
    set_default_if_unset LOAD_IN_16BIT 1
    set_default_if_unset CONCURRENT_CONVERSATIONS 2
  elif [[ ( "$GPU_NAME" == *"H100"* || "$GPU_NAME" == *"A100"* || "$GPU_NAME" == *"RTX PRO 6000"* ) && "${GPU_MEM_MB:-0}" -ge 70000 ]]; then
    GPU_PROFILE_SELECTED="highmem-accelerator"
    set_default_if_unset MAX_SEQ_LENGTH 6144
    set_default_if_unset LOAD_IN_4BIT 0
    set_default_if_unset LOAD_IN_16BIT 1
    set_default_if_unset CONCURRENT_CONVERSATIONS 4
  else
    GPU_PROFILE_SELECTED="portable"
    set_default_if_unset MAX_SEQ_LENGTH 4096
    set_default_if_unset LOAD_IN_4BIT 1
    set_default_if_unset LOAD_IN_16BIT 0
    set_default_if_unset CONCURRENT_CONVERSATIONS 4
  fi
else
  case "$GPU_PROFILE" in
    ultrahighmem-accelerator)
      set_default_if_unset MAX_SEQ_LENGTH 6144
      set_default_if_unset LOAD_IN_4BIT 0
      set_default_if_unset LOAD_IN_16BIT 1
      set_default_if_unset CONCURRENT_CONVERSATIONS 8
      ;;
    highmem-accelerator)
      set_default_if_unset MAX_SEQ_LENGTH 6144
      set_default_if_unset LOAD_IN_4BIT 0
      set_default_if_unset LOAD_IN_16BIT 1
      set_default_if_unset CONCURRENT_CONVERSATIONS 4
      ;;
    l40s)
      set_default_if_unset MAX_SEQ_LENGTH 4096
      set_default_if_unset LOAD_IN_4BIT 0
      set_default_if_unset LOAD_IN_16BIT 1
      set_default_if_unset CONCURRENT_CONVERSATIONS 2
      ;;
    portable)
      set_default_if_unset MAX_SEQ_LENGTH 4096
      set_default_if_unset LOAD_IN_4BIT 1
      set_default_if_unset LOAD_IN_16BIT 0
      set_default_if_unset CONCURRENT_CONVERSATIONS 4
      ;;
    *)
      set_default_if_unset MAX_SEQ_LENGTH 4096
      set_default_if_unset LOAD_IN_4BIT 0
      set_default_if_unset LOAD_IN_16BIT 1
      set_default_if_unset CONCURRENT_CONVERSATIONS 4
      ;;
  esac
fi

if [[ "$LOAD_IN_16BIT" == "1" && "$LOAD_IN_4BIT" != "1" && "${GPU_MEM_MB:-0}" -lt 52000 ]]; then
  echo "WARNING: pure 16-bit loading on a sub-52 GiB GPU can trigger CPU offload for Gemma 4 26B-A4B."
  echo "         The L40S profile defaults to 16-bit because the 4-bit fallback path was less reliable."
  echo "         If you still observe offload or slowdowns, lower CONCURRENT_CONVERSATIONS or override the load mode."
fi

export HF_HOME=$CACHE_ROOT/huggingface
export HF_HUB_CACHE=$CACHE_ROOT/huggingface/hub
export HF_ASSETS_CACHE=$CACHE_ROOT/huggingface/assets
export HF_XET_CACHE=$CACHE_ROOT/huggingface/xet
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
unset HF_HUB_ENABLE_HF_TRANSFER || true
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$CPU_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$CPU_THREADS}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_FILE")"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: eval script not found at $EVAL_SCRIPT"
  exit 1
fi

if [[ "$SMOKE" == "1" && -z "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES="$SMOKE_SAMPLES"
fi

echo "===== Eval diagnostics ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "env python: $PYTHON"
echo "gpu: $GPU_NAME (${GPU_MEM_MB:-unknown} MiB)"
echo "profile: $GPU_PROFILE_SELECTED"
echo "adapter: $ADAPTER_PATH"
echo "test file: $TEST_FILE"
echo "eval script: $EVAL_SCRIPT"
echo "output: $OUTPUT_FILE"
echo "model revision: ${MODEL_REVISION:-UNPINNED}"
echo "tool scope: $TOOL_SCOPE"
echo "smoke mode: $SMOKE"
echo "max samples: ${MAX_SAMPLES:-ALL}"
echo "max turns: $MAX_TURNS"
echo "max new tokens: $MAX_NEW_TOKENS"
echo "max seq length: $MAX_SEQ_LENGTH"
echo "include tool schemas: $INCLUDE_TOOL_SCHEMAS"
echo "inject empty thought: $INJECT_EMPTY_THOUGHT_CHANNEL"
echo "prefer base tokenizer: $PREFER_BASE_TOKENIZER"
echo "filter unavailable helper tools: $FILTER_UNAVAILABLE_HELPER_TOOLS"
echo "inject runtime helper note: $INJECT_RUNTIME_HELPER_NOTE"
echo "continue on missing context tool: $CONTINUE_ON_MISSING_CONTEXT_TOOL"
echo "rolling batch scheduler: $ROLLING_BATCH_SCHEDULER"
echo "resume output: $RESUME_OUTPUT"
echo "truncate partial output: $TRUNCATE_PARTIAL_OUTPUT"
echo "load_in_4bit/load_in_16bit: $LOAD_IN_4BIT / $LOAD_IN_16BIT"
echo "concurrent conversations: $CONCURRENT_CONVERSATIONS"
echo "gc collect every N turns: $GC_COLLECT_EVERY_N_TURNS"
echo "empty cuda cache every N turns: $EMPTY_CUDA_CACHE_EVERY_N_TURNS"
echo "cpu threads: $CPU_THREADS"
echo "allow slow pandapower: $ALLOW_SLOW_PANDAPOWER"
echo "slurm gres: ${SLURM_JOB_GRES:-unknown}"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
$PYTHON -V
$PYTHON -m pip list | grep -E "unsloth|scipy|transformers|torch|numba|pandapower" || true
nvidia-smi
echo "============================"

echo "===== Python preflight ====="
"$PYTHON" - <<'PY'
import importlib
import os

mods = ["scipy", "torch", "transformers", "numba", "pandapower"]
failed = []
allow_slow_pandapower = os.environ.get("ALLOW_SLOW_PANDAPOWER", "0") == "1"

for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK ({getattr(mod, '__version__', 'unknown')})")
    except Exception as exc:
        print(f"{name}: FAIL ({exc})")
        if name in {"numba", "pandapower"} and allow_slow_pandapower:
            print(f"{name}: allowed to fail because ALLOW_SLOW_PANDAPOWER=1")
        else:
            failed.append((name, str(exc)))

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
  "$EVAL_SCRIPT"
  --adapter "$ADAPTER_PATH"
  --test-file "$TEST_FILE"
  --tool-scope "$TOOL_SCOPE"
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

if [[ "$PREFER_BASE_TOKENIZER" == "1" ]]; then
  ARGS+=(--prefer-base-tokenizer)
else
  ARGS+=(--prefer-adapter-tokenizer)
fi

if [[ "$FILTER_UNAVAILABLE_HELPER_TOOLS" == "1" ]]; then
  ARGS+=(--filter-unavailable-helper-tools)
else
  ARGS+=(--no-filter-unavailable-helper-tools)
fi

if [[ "$INJECT_RUNTIME_HELPER_NOTE" == "1" ]]; then
  ARGS+=(--inject-runtime-helper-note)
else
  ARGS+=(--no-inject-runtime-helper-note)
fi

if [[ "$CONTINUE_ON_MISSING_CONTEXT_TOOL" == "1" ]]; then
  ARGS+=(--continue-on-missing-context-tool)
else
  ARGS+=(--no-continue-on-missing-context-tool)
fi

if [[ "$ROLLING_BATCH_SCHEDULER" == "1" ]]; then
  ARGS+=(--rolling-batch-scheduler)
else
  ARGS+=(--no-rolling-batch-scheduler)
fi

if [[ "$RESUME_OUTPUT" == "1" ]]; then
  ARGS+=(--resume-output)
fi

if [[ "$TRUNCATE_PARTIAL_OUTPUT" == "1" ]]; then
  ARGS+=(--truncate-partial-output)
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

ARGS+=(--gc-collect-every-n-turns "$GC_COLLECT_EVERY_N_TURNS")
ARGS+=(--empty-cuda-cache-every-n-turns "$EMPTY_CUDA_CACHE_EVERY_N_TURNS")

if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_EVAL_ARGS)
  ARGS+=("${EXTRA_ARR[@]}")
fi

"$PYTHON" "${ARGS[@]}"
