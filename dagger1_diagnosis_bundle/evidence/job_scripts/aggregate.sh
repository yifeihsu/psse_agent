#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd "$REPO"
test "$(git rev-parse HEAD)" = "$COMMIT" && test -z "$(git status --porcelain)" && test ! -e "$ROUND1"
"$PY" scripts/build_dagger1_training_aggregate.py \
  --d0-aggregate-dir "$D0_DIR" \
  --d1 "$D1_DIR/training_beta025.jsonl" \
  --d1-manifest "$D1_DIR/training_beta025.jsonl.manifest.json" \
  --output-dir "$ROUND1" --d1-share 0.25
(cd "$ROUND1" && sha256sum --check --strict SHA256SUMS)
"$PY" - <<'PYCODE'
import json, os
from pathlib import Path
root = Path(os.environ["ROUND1"])
expected = {"aggregate.raw.jsonl", "aggregate.d0.raw.jsonl", "aggregate.d1.raw.jsonl", "aggregate.train_view.raw.jsonl", "aggregate.train_view.jsonl", "aggregate.validation.jsonl", "aggregate.test.jsonl", "aggregate.generation_provenance.json", "aggregate.preflight.json", "SHA256SUMS"}
actual = {p.name for p in root.iterdir()}
if actual != expected:
    raise RuntimeError(f"round1 file-set mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
prov = json.loads((root / "aggregate.generation_provenance.json").read_text())
pre = json.loads((root / "aggregate.preflight.json").read_text())
if prov.get("release_eligible") is not True or prov.get("release_failures") != [] or pre.get("release_eligible") is not True:
    raise RuntimeError("round1 aggregate is not release eligible")
source = prov.get("source_state")
if not isinstance(source, dict) or source.get("source_commit") != os.environ["COMMIT"]:
    raise RuntimeError("round1 source commit binding mismatch")
if prov.get("generation_provenance_id") != pre.get("generation_provenance_id"):
    raise RuntimeError("round1 provenance ID mismatch")
print(json.dumps({"passed": True, "source_commit": os.environ["COMMIT"], "generation_provenance_id": prov["generation_provenance_id"], "split_rows": pre.get("split_rows"), "training_view": pre.get("training_view")}, sort_keys=True))
PYCODE
test -z "$(git status --porcelain)"
