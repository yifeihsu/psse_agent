#!/usr/bin/env bash
#
# Canonical DAgger-1 collection wrapper.
#
# The collector's non-zero exit is *data*, not a reason to abort: a fail-closed
# NO-GO and a completed analysis run both exit 1 by contract.  The round-2 job
# scripts ran the collector under `set -euo pipefail`, so the shell aborted at
# the collector line and every post-run publication assertion was skipped --
# the safety checks that were supposed to guard the run never executed.
#
# This wrapper captures the collector's status, classifies it, and then always
# runs the assertions appropriate to the classification.
#
# Required environment:
#   PY        python interpreter
#   REPO      repository root (must be clean and at COMMIT)
#   COMMIT    expected source commit
#   D0_DIR    D0 aggregate directory
#   D1_DIR    DAgger-1 working directory
#   LEARNER   model id
#   REV       model revision
#   HFROOT    HF_HOME
# Optional:
#   MODE      "strict" (default) or "analysis"
#   BETA      default 0.25
set -euo pipefail

MODE="${MODE:-strict}"
BETA="${BETA:-0.25}"

case "$MODE" in
  strict)   PREFIX="training_beta025";  EXTRA=(--require-recommended-target) ;;
  analysis) PREFIX="analysis_beta025";  EXTRA=(--require-recommended-target
                                               --analysis-only-complete-schedule) ;;
  *) echo "MODE must be 'strict' or 'analysis', got '$MODE'" >&2; exit 2 ;;
esac

OUTPUT="$D1_DIR/$PREFIX.jsonl"
ALL_OUTPUT="$D1_DIR/$PREFIX.all.jsonl"
MANIFEST="$D1_DIR/$PREFIX.jsonl.manifest.json"
FAILED_DIR="$D1_DIR/$PREFIX.failed-collection"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HOME="$HFROOT" \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain)"
"$PY" -m pip check
"$PY" -m psse_env.sft.release_hardware
nvidia-smi

# ---- collector: capture status instead of aborting on it --------------------
set +e
"$PY" scripts/collect_dagger1_recovery.py \
  --input "$D1_DIR/scenarios.json" \
  --scenario-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --scenario-generator-report "$D1_DIR/scenario_generator_report.json" \
  --development-holdout "$D1_DIR/development_holdout.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --development-holdout-generator-report "$D1_DIR/development_holdout.generator.json" \
  --d0-aggregate-dir "$D0_DIR" \
  --output "$OUTPUT" \
  --all-output "$ALL_OUTPUT" \
  --model-id "$LEARNER" --model-revision "$REV" \
  --collection-pass training --beta "$BETA" \
  --failed-collection-dir "$FAILED_DIR" \
  "${EXTRA[@]}"
COLLECT_RC=$?

"$PY" scripts/classify_collection_result.py \
  --exit-code "$COLLECT_RC" \
  --production-output "$OUTPUT" \
  --production-manifest "$MANIFEST" \
  --failed-collection-dir "$FAILED_DIR"
RESULT_RC=$?
set -e

# ---- assertions: always run, branched on the classification ----------------
echo "collector_exit=$COLLECT_RC classification_exit=$RESULT_RC mode=$MODE"
case "$RESULT_RC" in
  0)  # STRICT_GO -- production outputs must exist and no failure bundle.
      test -s "$OUTPUT"
      test -s "$ALL_OUTPUT"
      test -s "$MANIFEST"
      test ! -e "$FAILED_DIR"
      ;;
  20|30)  # STRICT_NO_GO or ANALYSIS_COMPLETE -- nothing may be published.
      test -d "$FAILED_DIR"
      test -s "$FAILED_DIR/failure_evidence.json"
      test ! -e "$OUTPUT"
      test ! -e "$MANIFEST"
      ;;
  *)  # INFRASTRUCTURE_FAILURE -- publication must still not have happened.
      test ! -e "$MANIFEST"
      ;;
esac
test -z "$(git status --porcelain)"

exit "$RESULT_RC"
