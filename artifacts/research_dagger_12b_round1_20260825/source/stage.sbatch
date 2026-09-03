#!/usr/bin/env bash
#SBATCH --job-name=dagger12b
#SBATCH --output=/scratch/yx3882/research_dagger_12b_20260824/logs/%x_%j.out
#SBATCH --error=/scratch/yx3882/research_dagger_12b_20260824/logs/%x_%j.err
#SBATCH --account=torch_pr_627_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --requeue

# One stage of the 12B research pipeline, selected by $PIPELINE_STAGE.
# Stages are output-guarded so a requeue resumes instead of redoing work.

set -euo pipefail

RUN_ROOT=/scratch/yx3882/research_dagger_12b_20260824
REPO_ROOT=$RUN_ROOT/src
PYTHON=/scratch/yx3882/envs/dagger12b_overlay/bin/python
MODEL_ID="unsloth/gemma-4-12B-it"
MODEL_REV="55cdba0740a9765956f49501f689a66b098feda3"
SEED=20260823
DATA=$RUN_ROOT/data
OUT=$RUN_ROOT/out
mkdir -p "$OUT"
cd "$REPO_ROOT"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=/scratch/yx3882/.cache/huggingface
export WANDB_MODE=offline
export WANDB_DIR=$RUN_ROOT/wandb

model=(--model-id "$MODEL_ID" --revision "$MODEL_REV")

case "$PIPELINE_STAGE" in
  bc0)
    [ -f "$OUT/bc0/research_training_summary.json" ] && exit 0
    $PYTHON -m research.train \
      --train "$DATA/d0_full/aggregate.train_view.jsonl" \
      --validation "$DATA/d0_full/aggregate.validation.jsonl" \
      --output-dir "$OUT/bc0" "${model[@]}" \
      --max-length 8192 --epochs 2 --seed $SEED
    ;;
  collect_shard)
    out_shard=$OUT/round1_rows.shard${SHARD_INDEX}.jsonl
    [ -s "$out_shard" ] && exit 0
    $PYTHON -m research.collect \
      --scenarios "$DATA/shards/scenarios.shard${SHARD_INDEX}.json" \
      --output "$out_shard" \
      --adapter "$OUT/bc0" \
      --beta 0.3 --max-steps 24 --seed $((SEED + SHARD_INDEX)) \
      --round0-dir "$DATA/d0_full" \
      --development-holdout "$DATA/d1_run/development_holdout.json" \
      "${model[@]}" > "$OUT/collect_shard${SHARD_INDEX}.json"
    ;;
  round1)
    if [ ! -f "$OUT/round1/research_training_summary.json" ]; then
      cat "$OUT"/round1_rows.shard*.jsonl > "$OUT/round1_rows.jsonl"
      $PYTHON - <<PYEOF
import json, sys
sys.path.insert(0, "$REPO_ROOT")
from research.run_dagger import build_aggregate
report = build_aggregate(
    round0_train="$DATA/d0_full/aggregate.train_view.jsonl",
    round0_validation="$DATA/d0_full/aggregate.validation.jsonl",
    round1_rows="$OUT/round1_rows.jsonl",
    output_dir="$OUT/aggregate",
    seed=$SEED,
)
json.dump(report, open("$OUT/aggregate_report.json", "w"), indent=2)
PYEOF
      $PYTHON -m research.train \
        --train "$OUT/aggregate/aggregate.train.jsonl" \
        --validation "$OUT/aggregate/aggregate.validation.jsonl" \
        --output-dir "$OUT/round1" "${model[@]}" \
        --max-length 8192 --epochs 2 --seed $SEED
    fi
    ;;
  eval_bc0)
    [ -f "$OUT/eval_bc0.json" ] && exit 0
    $PYTHON -m research.evaluate \
      --scenarios "$DATA/eval_scenarios.json" \
      --adapter "$OUT/bc0" --label bc0-12b \
      --output "$OUT/eval_bc0.json" "${model[@]}"
    ;;
  eval_round1)
    [ -f "$OUT/eval_round1.json" ] && exit 0
    $PYTHON -m research.evaluate \
      --scenarios "$DATA/eval_scenarios.json" \
      --adapter "$OUT/round1" --label round1-12b \
      --output "$OUT/eval_round1.json" "${model[@]}"
    ;;
  *)
    echo "unknown PIPELINE_STAGE=$PIPELINE_STAGE" >&2; exit 2;;
esac
echo "stage $PIPELINE_STAGE done"
