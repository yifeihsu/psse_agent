#!/usr/bin/env bash
#SBATCH --job-name=triage-llm
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
#SBATCH --output=/scratch/yx3882/classifier_triage_20261004/logs/%x-%j.out
# LLM leg of the IEEE 14 triage benchmark (docs/classifier_triage_plan_20261004.md):
# a classification fine-tune on truth-derived first actions, then the adapter's
# first action on the test rows and a probe sample, and the probability of
# each first action on the calibration rows, the test rows and the same sample.
#
#   sbatch --export=ALL,VARIANT=prompt_top10_signed --job-name=triage-top10s llm_triage.sbatch
#   sbatch --export=ALL,VARIANT=prompt_top10_signed,EPOCHS=3,RUN_TAG=_e3 --job-name=triage-top10s-e3 llm_triage.sbatch
#
# The recipe is the pipeline's BC0 one (4-bit base, rank-16 LoRA, micro-batch 1
# with four accumulation steps, lr 1e-4, one pass, best validation loss kept);
# EPOCHS changes the number of passes and RUN_TAG the output directory
# (out/<variant><tag>).  Every phase resumes: training from its newest complete
# checkpoint, scoring from the rows already written.
set -euo pipefail
WORK=/scratch/yx3882/classifier_triage_20261004
VARIANT=${VARIANT:?set VARIANT to a prompt variant directory under data/}
SRC=$WORK/source
DATA=$WORK/data/$VARIANT
EPOCHS=${EPOCHS:-1}
RUN_TAG=${RUN_TAG:-}
RUN=$WORK/out/$VARIANT$RUN_TAG
LOGS=$WORK/logs
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python
MODEL_CHOICE=12b
MODEL_ID=google/gemma-4-12B-it
SEED=20261004

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/scratch/yx3882/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
export RESEARCH_MAX_INPUT_TOKENS=32768 RESEARCH_MAX_NEW_TOKENS=256 PSSE_LOCAL_DIAGNOSTIC_BUILD=1

echo "[triage] variant $VARIANT epochs $EPOCHS run $RUN job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
echo "[triage] source $(git -C "$SRC" rev-parse HEAD)"
[[ -s "$DATA/train.jsonl" && -s "$DATA/validation.jsonl" && -s "$DATA/score.jsonl" && -s "$DATA/score_labels.json" ]] \
  || { echo "dataset for $VARIANT is incomplete under $DATA" >&2; exit 2; }
mkdir -p "$RUN" "$LOGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  >> "$LOGS/gpu-util-$VARIANT$RUN_TAG-${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &
GPU_SAMPLER_PID=$!
trap 'kill "$GPU_SAMPLER_PID" 2>/dev/null || true' EXIT

cd "$SRC"
if [[ ! -s "$RUN/lora/adapter_model.safetensors" || ! -s "$RUN/research_run.json" ]]; then
  resume_args=()
  checkpoint_path=$("$PY" -m psse_env.sft.research_checkpoint \
    --output-dir "$RUN" --expected-base-model "$MODEL_ID" || true)
  if [[ -n "$checkpoint_path" ]]; then
    echo "[triage] resuming training from $checkpoint_path"
    resume_args+=(--resume-from-checkpoint "$checkpoint_path")
  fi
  "$PY" -m psse_env.sft research-train \
    --model-choice "$MODEL_CHOICE" \
    --train "$DATA/train.jsonl" \
    --validation "$DATA/validation.jsonl" \
    --output-dir "$RUN" \
    --max-length 32768 \
    --strict-prompt-length \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --epochs "$EPOCHS" \
    --logging-steps 1 \
    --save-steps 128 \
    --eval-steps 128 \
    --select-best-eval-loss \
    --lora-rank 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --smoke-steps 1 \
    --reload-canaries 1 \
    --seed "$SEED" \
    --run-name "triage-$VARIANT$RUN_TAG" \
    --report-to none \
    "${resume_args[@]}"
fi
[[ -s "$RUN/lora/adapter_model.safetensors" ]] || { echo "training left no adapter" >&2; exit 3; }
echo "[triage] adapter ready $(date -u +%FT%TZ)"

"$PY" -m research.classifier_triage.llm_score \
  --adapter "$RUN/lora" \
  --score "$DATA/score.jsonl" \
  --output "$RUN/scores.json" \
  --probe-per-cell 100
"$PY" -m research.classifier_triage.llm_score \
  --adapter "$RUN/lora" \
  --score "$DATA/score.jsonl" \
  --output "$RUN/probabilities.json" \
  --probe-per-cell 100 \
  --probabilities
"$PY" - "$RUN" "$VARIANT" "${SLURM_JOB_ID:-manual}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
run = Path(sys.argv[1])
rows = json.loads((run / "scores.json").read_text(encoding="utf-8"))["rows"]
receipt = {"contract": "classifier_triage_llm_done_v1", "variant": sys.argv[2], "slurm_job_id": sys.argv[3],
           "rows": len(rows), "decisions": dict(Counter(item["decision"] for item in rows.values())),
           "probability_rows": len(json.loads((run / "probabilities.json").read_text(encoding="utf-8"))["rows"])}
(run / "done.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
print("[triage] done", json.dumps(receipt))
PY
echo triage-llm-complete
