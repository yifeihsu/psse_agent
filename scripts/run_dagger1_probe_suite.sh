#!/usr/bin/env bash
#
# Build the observable recovery-probe auxiliary suite.
#
# This is a separate source from the natural DAgger-1 corpus and publishes to
# its own JSONL and manifest.  It is deliberately NOT part of the strict
# collection wrapper: a probe suite can be rebuilt without re-running collection,
# and a probe shortfall must never be confused with a natural coverage failure.
#
# Exit codes mirror the collection wrapper so orchestration can branch without
# parsing stdout:
#   0   suite meets both probe floors
#   20  suite is short of a floor, or a disjointness check failed
#   1   infrastructure failure
#
# Required environment:
#   PY, REPO, COMMIT, D1_DIR, D0_DIR, HFROOT
set -euo pipefail

OUTPUT="$D1_DIR/recovery_probes.jsonl"
MANIFEST="$D1_DIR/recovery_probes.manifest.json"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HOME="$HFROOT" \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain)"

# The generator drives the environment and the observable expert only.  No
# learner model is loaded, so this stage needs no GPU.
set +e
"$PY" scripts/build_recovery_probe_suite.py \
  --scenarios "$D1_DIR/scenarios.json" \
  --scenario-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --scenario-generator-report "$D1_DIR/scenario_generator_report.json" \
  --output "$OUTPUT" \
  --manifest "$MANIFEST" \
  --reviewed-source-commit "$COMMIT" \
  --forbidden-suite "${FORBIDDEN_SUITE:-psse_env/dagger/suites/bc0_eval_suite_v1.json}" \
  --evaluation-policy "${EVALUATION_POLICY:-psse_env/dagger/bc0_evaluation_policy.json}" \
  --development-holdout "$D1_DIR/development_holdout.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --development-holdout-generator-report "$D1_DIR/development_holdout.generator.json" \
  --d0-aggregate-dir "$D0_DIR"
PROBE_RC=$?
set -e

echo "probe_suite_exit=$PROBE_RC"
case "$PROBE_RC" in
  0)  # Both floors met: rows and manifest must exist and agree.
      test -s "$OUTPUT"
      test -s "$MANIFEST"
      "$PY" - "$OUTPUT" "$MANIFEST" <<'PY'
import hashlib, json, sys
rows_path, manifest_path = sys.argv[1], sys.argv[2]
manifest = json.loads(open(manifest_path, encoding="utf-8").read())
declared = manifest["probe_rows"]
digest = hashlib.sha256(open(rows_path, "rb").read()).hexdigest()
assert digest == declared["sha256"], "probe rows do not match the manifest digest"
rows = [json.loads(line) for line in open(rows_path, encoding="utf-8") if line.strip()]
assert len(rows) == declared["row_count"], "probe row count disagrees with manifest"
# A probe row may train, but may never satisfy a natural on-policy floor.
assert all(r["production_label_eligible"] is False for r in rows)
assert all(r["state_origin"] == "observable_recovery_probe" for r in rows)
assert manifest["natural_on_policy_support_eligible"] is False
print(f"probe suite verified: {len(rows)} rows, {manifest['distinct_physical_roots']} roots")
PY
      ;;
  20) # Short of a floor: the artifacts stay for inspection, nothing is claimed.
      echo "probe suite did not meet its floors; see $MANIFEST"
      ;;
  *)  # Infrastructure failure: nothing may be left claiming to be a suite.
      test ! -e "$MANIFEST"
      ;;
esac
test -z "$(git status --porcelain)"

exit "$PROBE_RC"
