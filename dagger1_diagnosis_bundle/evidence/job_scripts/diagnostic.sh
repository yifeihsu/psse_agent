#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HOME="$HFROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"
test "$(git rev-parse HEAD)" = "$COMMIT" && test -z "$(git status --porcelain)"
"$PY" -m pip check
"$PY" -m psse_env.sft.release_hardware
nvidia-smi
"$PY" scripts/collect_dagger1_recovery.py \
  --input "$D1_DIR/scenarios.json" \
  --scenario-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --scenario-generator-report "$D1_DIR/scenario_generator_report.json" \
  --development-holdout "$D1_DIR/development_holdout.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --development-holdout-generator-report "$D1_DIR/development_holdout.generator.json" \
  --d0-aggregate-dir "$D0_DIR" \
  --output "$D1_DIR/diagnostic_beta0.jsonl" \
  --all-output "$D1_DIR/diagnostic_beta0.all.jsonl" \
  --model-id "$LEARNER" --model-revision "$REV" \
  --collection-pass diagnostic --beta 0.0
test -s "$D1_DIR/diagnostic_beta0.jsonl" && test -s "$D1_DIR/diagnostic_beta0.all.jsonl" && test -s "$D1_DIR/diagnostic_beta0.jsonl.manifest.json"
test -z "$(git status --porcelain)"
