#!/usr/bin/env bash
#SBATCH --job-name=cc-train
#SBATCH --account=torch_pr_627_general
#SBATCH --constraint=a100|h100|h200|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/compound_cloning_20261106/logs/%x-%j.out
# The cloning pass (spec: cloning_pass_training_spec_20261106.md): one supervised pass over the
# selected rows, continuing R1 of the conventional cell, final adapter kept.  The training command
# is the one of research/hpc/full_pipeline_20260907/stage_train.sbatch with three differences: it
# reads the rows of this cell instead of a round's collection receipt and mixture, it saves and
# validates at separate intervals, and by default it renders each row with its run-time tool list
# (CLONE_TOOLS, train_runtime_tools.py).  max-steps is ceil(rows / 4), read from the row file here.
#
# CLONE_MODE=smoke runs the same path on about 200 rows for two optimizer steps, saving at each and
# validating at the second (the rows are cut by smoke_inputs.py), and first times the row gate.
#
# Output-guarded and resumable.  A preemption requeue restarts this script, which resumes from the
# newest complete checkpoint.  --requeue does not restart a job that reaches its time limit, so a
# pass longer than 12 h runs as a chain of these jobs (submit_cell.sh); each link does the same.
set -euo pipefail
WORK=/scratch/yx3882/compound_cloning_20261106
# shellcheck source=cell.env
source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"
cell_environment
MODE=${CLONE_MODE:-full}
case "$MODE" in
  full)
    STAGE_DIR=$OUT/clone
    TRAIN_ROWS=$TRAIN_VIEW
    VALIDATION_ROWS=$VALIDATION_VIEW
    SAVE_STEPS=$CLONE_SAVE_STEPS
    EVAL_STEPS=$CLONE_EVAL_STEPS
    ;;
  smoke)
    STAGE_DIR=$OUT/smoke
    TRAIN_ROWS=$STAGE_DIR/rows/train.jsonl
    VALIDATION_ROWS=$STAGE_DIR/rows/validation.jsonl
    SAVE_STEPS=1
    EVAL_STEPS=2
    ;;
  *) echo "CLONE_MODE must be full or smoke, got $MODE" >&2; exit 2 ;;
esac
case "$CLONE_TOOLS" in
  runtime) TRAIN_COMMAND=("$PY" "$CELL/train_runtime_tools.py") ;;
  registry) TRAIN_COMMAND=("$PY" -m psse_env.sft research-train) ;;
  *) echo "CLONE_TOOLS must be runtime or registry, got $CLONE_TOOLS" >&2; exit 2 ;;
esac
TRAIN_DIR=$STAGE_DIR/training
RECEIPT=$STAGE_DIR/training.done
# A restart loads the row gate's result instead of spending its slot on it (train_runtime_tools.py).
export CLONE_GATE_CACHE=$STAGE_DIR/row_gate_cache.pkl
echo "[clone:$MODE] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
if [[ -s "$RECEIPT" ]]; then
  echo "[clone:$MODE] receipt present; nothing to train"
  echo "cc-train-$MODE-complete"
  exit 0
fi
if another_link_running; then
  echo "[clone:$MODE] another link of $SLURM_JOB_NAME is running; leaving it the output directory" >&2
  exit 3
fi
[[ -s "$TRAIN_VIEW" && -s "$VALIDATION_VIEW" ]] || { echo "rows missing under $DATA/rows: deploy them first" >&2; exit 2; }
[[ -s "$START_ADAPTER/adapter_model.safetensors" ]] || { echo "start adapter missing: $START_ADAPTER" >&2; exit 2; }
mkdir -p "$TRAIN_DIR"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader \
  > "$LOGS/gpu-train-$MODE-${SLURM_JOB_ID:-manual}-r${SLURM_RESTART_COUNT:-0}.csv" || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-train-$MODE-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
stage_cleanup() { kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap stage_cleanup EXIT

cd "$SRC"
if [[ "$MODE" == smoke && ! ( -s "$TRAIN_ROWS" && -s "$VALIDATION_ROWS" ) ]]; then
  "$PY" "$CELL/smoke_inputs.py" rows --train "$TRAIN_VIEW" --validation "$VALIDATION_VIEW" --out-dir "$STAGE_DIR/rows" \
    --train-rows "$SMOKE_TRAIN_ROWS" --validation-rows "$SMOKE_VALIDATION_ROWS" --seed "$CLONE_SEED"
fi
# Every row must declare the evidence profile of this cell (as stage_bc0.sbatch checks); the step
# count is read from the same file the trainer reads, never from a note or a receipt.
read -r TRAIN_MAX_STEPS TRAIN_ROW_COUNT < <(
"$PY" - "$TRAIN_ROWS" "$VALIDATION_ROWS" <<'PY' | tail -1
import math
import sys
from pathlib import Path
from scripts.run_dagger_research import load_jsonl, validate_training_evidence_profile
counts = []
for path in sys.argv[1:3]:
    rows = load_jsonl(Path(path))
    validate_training_evidence_profile(rows, evidence_profile="scada_only", source=path)
    counts.append(len(rows))
if not all(counts):
    raise SystemExit("empty row file")
print(math.ceil(counts[0] / 4), counts[0])
PY
)
if [[ "$MODE" == smoke ]]; then TRAIN_MAX_STEPS=2; fi
echo "[clone:$MODE] $TRAIN_ROW_COUNT rows -> $TRAIN_MAX_STEPS optimizer steps; save every $SAVE_STEPS, validate every $EVAL_STEPS"
if [[ "$MODE" == smoke && ! -s "$STAGE_DIR/gate_timing.json" ]]; then
  "$PY" "$CELL/smoke_inputs.py" gate --train "$TRAIN_ROWS" --validation "$VALIDATION_ROWS" --full-train "$TRAIN_VIEW" --tools "$CLONE_TOOLS" \
    --model-choice "$MODEL_CHOICE" --max-length "$TRAIN_MAX_LENGTH" --output-dir "$TRAIN_DIR" | tee "$STAGE_DIR/gate_timing.json"
fi

if [[ ! -s "$TRAIN_DIR/lora/adapter_model.safetensors" || ! -s "$TRAIN_DIR/research_run.json" ]]; then
  resume_args=()
  checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint \
    --output-dir "$TRAIN_DIR" --expected-base-model "$MODEL_ID" || true)
  if [[ -n "$checkpoint_path" ]]; then
    echo "[clone:$MODE] resuming from newest complete checkpoint: $checkpoint_path"
    resume_args+=(--resume-from-checkpoint "$checkpoint_path")
  fi
  if [[ "$MODE" == full ]]; then prune_checkpoints "$TRAIN_DIR" "$checkpoint_path"; fi
  "${TRAIN_COMMAND[@]}" \
    --model-choice "$MODEL_CHOICE" \
    --train "$TRAIN_ROWS" \
    --validation "$VALIDATION_ROWS" \
    --initial-adapter "$START_ADAPTER" \
    --output-dir "$TRAIN_DIR" \
    --max-length "$TRAIN_MAX_LENGTH" \
    --strict-prompt-length \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate "$CLONE_LEARNING_RATE" \
    --epochs 1 \
    --max-steps "$TRAIN_MAX_STEPS" \
    --logging-steps 1 \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --smoke-steps 1 \
    --reload-canaries 1 \
    --seed "$CLONE_SEED" \
    --run-name "$CLONE_RUN_NAME-$MODE" \
    --report-to none \
    "${resume_args[@]}"
fi
if [[ "$MODE" == smoke ]]; then
  # The resume path of the long job: what it would pick from what this run saved.
  checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint --output-dir "$TRAIN_DIR" --expected-base-model "$MODEL_ID")
  [[ -n "$checkpoint_path" ]] || { echo "no complete checkpoint after the smoke" >&2; exit 2; }
  echo "[clone:smoke] a resume would start from $checkpoint_path"
else
  prune_checkpoints "$TRAIN_DIR"
fi
"$PY" - "$TRAIN_DIR" "$RECEIPT" "$MODE" "$TRAIN_ROW_COUNT" "$TRAIN_MAX_STEPS" "$START_ADAPTER" "$ROWS_MANIFEST" "$SAVE_STEPS" "$EVAL_STEPS" "$CLONE_TOOLS" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path
from psse_env.dagger.release_factories import checkpoint_tree_sha256

train_dir = Path(sys.argv[1])
run = json.loads((train_dir / "research_run.json").read_text(encoding="utf-8"))
# A later link must not turn a failed reload or canary into a receipt.
if run.get("passed") is not True:
    raise SystemExit(f"{train_dir / 'research_run.json'} did not pass: see its reload and completion_errors")
state_path = train_dir / "trainer_state.json"
state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
manifest_path = Path(sys.argv[7])
payload = {
    "contract": "compound_cloning_training_done_v1",
    "mode": sys.argv[3],
    "train_rows": int(sys.argv[4]),
    "max_steps": int(sys.argv[5]),
    "save_steps": int(sys.argv[8]),
    "eval_steps": int(sys.argv[9]),
    "rendered_tools": sys.argv[10],
    "global_step": state.get("global_step"),
    "start_adapter_path": sys.argv[6],
    "start_adapter_tree_sha256": checkpoint_tree_sha256(sys.argv[6]),
    "adapter_path": str(train_dir / "lora"),
    "adapter_tree_sha256": checkpoint_tree_sha256(train_dir / "lora"),
    "rows_manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None,
    "validation_losses": [{"step": entry.get("step"), "eval_loss": entry["eval_loss"]}
                           for entry in state.get("log_history", []) if "eval_loss" in entry],
    "reload": run.get("reload"),
    "research_run_passed": run.get("passed"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({key: payload[key] for key in ("mode", "train_rows", "max_steps", "global_step", "research_run_passed")}))
PY
echo "cc-train-$MODE-complete"
