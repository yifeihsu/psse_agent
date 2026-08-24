#!/usr/bin/env bash
#SBATCH --job-name=gemma4-mini-bc0
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --open-mode=append
#SBATCH --output=gemma4_mini_bc0_%j.out
#SBATCH --error=gemma4_mini_bc0_%j.err

# Research-only 12B mini-BC0: 128 D0 train rows, 32 root-disjoint validation
# rows, one epoch, periodic evaluation/checkpoints, final adapter reload, and
# one generation canary. This launcher never collects DAgger data.

set -Eeuo pipefail

: "${RESEARCH_SOURCE_ROOT:?set RESEARCH_SOURCE_ROOT to the committed checkout}"
: "${RESEARCH_ENV:?set RESEARCH_ENV to the separate research conda environment}"
: "${RESEARCH_RUN_ROOT:?set RESEARCH_RUN_ROOT to the persistent run directory}"

VIEW_ROOT="$RESEARCH_RUN_ROOT/views"
TRAIN_FILE="$VIEW_ROOT/mini.train128.jsonl"
VALIDATION_FILE="$VIEW_ROOT/mini.validation32.jsonl"
E4B_SMOKE="$RESEARCH_RUN_ROOT/smoke/e4b/research_smoke.json"
NATIVE_SMOKE="$RESEARCH_RUN_ROOT/smoke/12b/research_smoke.json"
OUTPUT_DIR="$RESEARCH_RUN_ROOT/bc0/mini"

for required in \
  "$RESEARCH_SOURCE_ROOT/psse_env/requirements-sft-research.txt" \
  "$TRAIN_FILE" \
  "$VALIDATION_FILE" \
  "$E4B_SMOKE" \
  "$NATIVE_SMOKE"
do
  if [[ ! -f "$required" ]]; then
    echo "required mini-BC0 input is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.job.lock"
if ! flock -n 9; then
  echo "another mini-BC0 job owns $OUTPUT_DIR" >&2
  exit 73
fi

module purge
module load anaconda3/2025.06
# shellcheck source=/dev/null
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$RESEARCH_ENV"
cd "$RESEARCH_SOURCE_ROOT"

python - "$E4B_SMOKE" "$NATIVE_SMOKE" "$TRAIN_FILE" "$VALIDATION_FILE" <<'PY'
import json
import pathlib
import sys

e4b = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
native = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))


def load_rows(path: str, expected_rows: int) -> tuple[list[dict], set[str]]:
    rows = []
    for line_number, line in enumerate(
        pathlib.Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    if len(rows) != expected_rows:
        raise SystemExit(f"{path} has {len(rows)} rows; expected {expected_rows}")
    roots = set()
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("protocol") != "canonical":
            raise SystemExit(f"{path}[{index}] is not canonical protocol")
        root = str(
            row.get("physical_root_fingerprint")
            or metadata.get("physical_root_fingerprint")
            or ""
        ).strip()
        if not root:
            raise SystemExit(f"{path}[{index}] lacks a physical root")
        roots.add(root)
    return rows, roots


if e4b.get("passed") is not True:
    raise SystemExit("E4B integration smoke has not passed")
if native.get("passed") is not True:
    raise SystemExit("12B native integration smoke has not passed")
probes = native.get("probes", {})
if (
    probes.get("passed") is not True
    or int(probes.get("schema_valid_single_calls") or 0) < 9
    or int(probes.get("maximum_token_hits") or 0) != 0
    or int(probes.get("repetition_loops") or 0) != 0
):
    raise SystemExit("12B native generation probe gate is incomplete")
train_rows, train_roots = load_rows(sys.argv[3], 128)
validation_rows, validation_roots = load_rows(sys.argv[4], 32)
overlap = train_roots & validation_roots
if overlap:
    raise SystemExit(
        "mini-BC0 train/validation roots overlap: " + ", ".join(sorted(overlap)[:8])
    )
print(
    {
        "e4b_smoke": True,
        "native_smoke": True,
        "native_schema_valid": probes.get("schema_valid_single_calls"),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_roots": len(train_roots),
        "validation_roots": len(validation_roots),
    }
)
PY

python -m pip check

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RESEARCH_SOURCE_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${RESEARCH_HF_HOME:-/scratch/yx3882/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="${RESEARCH_TORCH_HOME:-/scratch/yx3882/.cache/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTORCH_ALLOC_CONF="${RESEARCH_PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python -m psse_env.sft research-cache \
  --model-choice 12b \
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

resume_args=()
if [[ ! -d "$OUTPUT_DIR/lora" ]]; then
  checkpoint_path="$(python -m psse_env.sft.research_checkpoint \
    --output-dir "$OUTPUT_DIR" \
    --expected-base-model "google/gemma-4-12B-it")"
  if [[ -n "$checkpoint_path" ]]; then
    echo "resuming from newest complete checkpoint: $checkpoint_path"
    resume_args+=(--resume-from-checkpoint "$checkpoint_path")
  fi
fi

python -m psse_env.sft research-train \
  --model-choice 12b \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-length 16384 \
  --strict-prompt-length \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --epochs 1 \
  --logging-steps 1 \
  --save-steps 8 \
  --eval-steps 8 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0 \
  --smoke-steps 1 \
  --reload-canaries 1 \
  --seed 20260720 \
  "${resume_args[@]}"

python - "$OUTPUT_DIR" <<'PY'
import json
import math
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = json.loads((root / "research_run.json").read_text(encoding="utf-8"))
state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
adapter = json.loads((root / "lora" / "adapter_config.json").read_text(encoding="utf-8"))
stage = report.get("preserved_training_stage")
stage = stage if isinstance(stage, dict) else report
model_selection = report.get("model_selection", {})
training_metrics = stage.get("training_metrics", {})
adapter_delta = stage.get("adapter_delta", {})
reload = report.get("reload", {})
splits = report.get("data", {}).get("splits", {})
eval_records = [
    {"step": int(row["step"]), "eval_loss": float(row["eval_loss"])}
    for row in state.get("log_history", [])
    if isinstance(row, dict)
    and isinstance(row.get("step"), (int, float))
    and not isinstance(row.get("step"), bool)
    and float(row["step"]).is_integer()
    and isinstance(row.get("eval_loss"), (int, float))
    and not isinstance(row.get("eval_loss"), bool)
    and math.isfinite(float(row["eval_loss"]))
]
checkpoints = sorted(path.name for path in root.glob("checkpoint-*") if path.is_dir())
expected_eval_steps = {8, 16, 24, 32}
observed_eval_steps = {row["step"] for row in eval_records}
global_step = state.get("global_step")
train_loss = training_metrics.get("train_loss")
finite_train_loss = bool(
    isinstance(train_loss, (int, float))
    and not isinstance(train_loss, bool)
    and math.isfinite(float(train_loss))
)
adapter_weights = [
    path
    for name in ("adapter_model.safetensors", "adapter_model.bin")
    if (path := root / "lora" / name).is_file()
    and not path.is_symlink()
    and path.stat().st_size > 0
]
adapter_files_valid = bool(
    str(adapter.get("peft_type") or "").upper() == "LORA"
    and (root / "lora" / "adapter_config.json").is_file()
    and not (root / "lora" / "adapter_config.json").is_symlink()
    and adapter_weights
)
model_contract_passed = bool(
    model_selection.get("model_id") == "google/gemma-4-12B-it"
    and model_selection.get("revision")
    == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    and model_selection.get("architecture") == "gemma4_unified"
    and adapter.get("base_model_name_or_path") == "google/gemma-4-12B-it"
)
split_contract_passed = bool(
    isinstance(splits, dict)
    and splits.get("train_rows") == 128
    and splits.get("validation_rows") == 32
    and int(splits.get("train_roots") or 0) > 0
    and int(splits.get("validation_roots") or 0) > 0
    and splits.get("overlap") == []
)
postflight = {
    "contract": "research_gemma4_mini_bc0_postflight_v1",
    "passed": bool(
        report.get("passed") is True
        and report.get("completion_errors", []) == []
        and model_contract_passed
        and split_contract_passed
        and finite_train_loss
        and int(adapter_delta.get("changed_tensors") or 0) > 0
        and global_step == 32
        and expected_eval_steps.issubset(observed_eval_steps)
        and "checkpoint-32" in checkpoints
        and adapter_files_valid
        and reload.get("fresh_base_reconstructed") is True
        and reload.get("adapter_reloaded") is True
        and reload.get("canaries_requested") == 1
        and reload.get("canaries_selected") == 1
        and reload.get("canaries_passed") == 1
        and reload.get("generation_canary_pass") is True
    ),
    "research_run_passed": report.get("passed") is True,
    "completion_errors": report.get("completion_errors", []),
    "model_contract_passed": model_contract_passed,
    "model_selection": model_selection,
    "split_contract_passed": split_contract_passed,
    "splits": splits,
    "finite_train_loss": finite_train_loss,
    "train_loss": train_loss,
    "changed_adapter_tensors": adapter_delta.get("changed_tensors"),
    "global_step": global_step,
    "expected_eval_steps": sorted(expected_eval_steps),
    "finite_evaluations": eval_records,
    "checkpoints": checkpoints,
    "adapter_files_valid": adapter_files_valid,
    "adapter_weight_files": [path.name for path in adapter_weights],
    "adapter_base_model": adapter.get("base_model_name_or_path"),
    "fresh_base_reconstructed": reload.get("fresh_base_reconstructed") is True,
    "adapter_reloaded": reload.get("adapter_reloaded") is True,
    "canaries_requested": reload.get("canaries_requested"),
    "canaries_selected": reload.get("canaries_selected"),
    "canaries_passed": reload.get("canaries_passed"),
    "generation_canary_pass": reload.get("generation_canary_pass") is True,
}
temporary = root / f".mini_bc0_postflight.json.tmp-{os.getpid()}"
temporary.write_text(json.dumps(postflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, root / "mini_bc0_postflight.json")
print(json.dumps(postflight, indent=2, sort_keys=True))
if postflight["passed"] is not True:
    raise SystemExit("mini-BC0 postflight failed")
PY
