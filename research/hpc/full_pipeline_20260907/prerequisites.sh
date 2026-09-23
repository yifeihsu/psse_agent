#!/usr/bin/env bash
# Verify the deployed source, corpora, model snapshot, and environment pins
# before any stage runs; optionally run the CPU test subset.
#
#   prerequisites.sh --output PATH [--with-tests]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=pipeline.env
source "$HERE/pipeline.env"
pipeline_environment
OUTPUT=""
WITH_TESTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --with-tests) WITH_TESTS=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
expected_commit=$(tr -d '[:space:]' < "$EXPECTED_SOURCE_COMMIT_FILE")
actual_commit=$(git -C "$SRC" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "source commit $actual_commit differs from deployed $expected_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]; then
  echo "source tree has tracked modifications" >&2
  git -C "$SRC" status --short --untracked-files=no >&2
  exit 2
fi
for path in "$HIF_CORPUS_TRAIN" "$HIF_CORPUS_VALID" "$HIF_CORPUS_TRAIN_EXTRA" "$HIF_CORPUS_VALID_EXTRA" "$IMBALANCE_CORPUS" \
  "$SRC/data/measurements_5class_merged.jsonl" "$BC0_SUITE"; do
  [[ -s "$path" ]] || { echo "missing or empty input: $path" >&2; exit 2; }
done
# Under wls_gated_diagnostics the HIF corpora must declare the PMU phasor
# precision of the study (pipeline.env PMU_PHASOR_SIGMA) in their meta.json;
# an empty PMU_PHASOR_SIGMA disables the check for an explicit ablation.
if [[ "$EVIDENCE_PROFILE" == wls_gated_diagnostics && -n "${PMU_PHASOR_SIGMA:-}" ]]; then
  "$PY" - "$PMU_PHASOR_SIGMA" "$HIF_CORPUS_TRAIN" "$HIF_CORPUS_VALID" "$HIF_CORPUS_TRAIN_EXTRA" "$HIF_CORPUS_VALID_EXTRA" <<'PY'
import json
import math
import sys
from pathlib import Path
expected = float(sys.argv[1])
problems = []
for samples in sys.argv[2:]:
    meta_path = Path(samples).with_name("meta.json")
    if not meta_path.is_file():
        problems.append(f"{meta_path}: missing")
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("three_phase_sigma", "branch_current_sigma_pu"):
        declared = meta.get(key)
        if not isinstance(declared, (int, float)) or not math.isclose(float(declared), expected, rel_tol=1e-9, abs_tol=0.0):
            problems.append(f"{meta_path}: {key}={declared!r}, expected {expected!r}")
if problems:
    sys.exit("HIF corpora do not declare the PMU phasor sigma of the wls_gated_diagnostics study "
             "(regenerate with --three-phase-noise-pu/--branch-current-noise-pu or set PMU_PHASOR_SIGMA= to ablate):\n  "
             + "\n  ".join(problems))
print(f"HIF corpora declare PMU phasor sigma {expected!r} per component")
PY
fi
[[ -s "$TRACE_VALIDATION" ]] || echo "note: trace validation set absent; only the BC0 suite is protected"
SNAPSHOT="$HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
if [[ ! -s "$SNAPSHOT/config.json" || ! -e "$SNAPSHOT/model.safetensors" ]]; then
  echo "offline model snapshot missing: $SNAPSHOT" >&2
  exit 2
fi
cd "$SRC"
for plan in "$D0_PLAN" "$ROUND_TRAIN_PLAN" "$DEVELOPMENT_PLAN"; do
  "$PY" -c 'import json,sys; p=json.loads(sys.argv[1]); assert p and all(int(v)>0 for v in p.values())' "$plan"
done
"$PY" - "$SRC/psse_env/requirements-sft-research.txt" <<'PY'
import sys
from importlib import metadata
from packaging.requirements import Requirement
mismatches = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        mismatches.append(f"{requirement.name}: missing")
        continue
    if not requirement.specifier.contains(installed, prereleases=True):
        mismatches.append(f"{requirement.name}: {installed} is not {requirement.specifier}")
if mismatches:
    sys.exit("research environment pins violated: " + "; ".join(mismatches))
print("environment pins satisfied")
PY
if [[ "$WITH_TESTS" == 1 ]]; then
  [[ -x "$TEST_PY" ]] || { echo "test interpreter missing: $TEST_PY" >&2; exit 2; }
  "$TEST_PY" -m pytest -q -p no:cacheprovider \
    psse_env/dagger/test_research_dagger_minimal.py \
    psse_env/providers/test_scenario_generator.py \
    test_hif_multiscan_estimator.py \
    research/test_hpc_full_pipeline.py
fi
"$PY" - "$OUTPUT" "$actual_commit" "$WITH_TESTS" "$SRC" "$SNAPSHOT" \
  "$HIF_CORPUS_TRAIN" "$HIF_CORPUS_VALID" "$IMBALANCE_CORPUS" "$BC0_SUITE" \
  "$HIF_CORPUS_TRAIN_EXTRA" "$HIF_CORPUS_VALID_EXTRA" "$EVIDENCE_PROFILE" "$HIF_SIGNATURE_MODE" <<'PY'
import datetime
import hashlib
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
output = Path(sys.argv[1])
def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "contract": "research_full_pipeline_prerequisites_v1",
    "evidence_profile": sys.argv[12],
    "hif_signature_mode": sys.argv[13],
    "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "model_snapshot": sys.argv[5],
    "inputs": {
        "hif_corpus_train": {"path": sys.argv[6], "sha256": sha(sys.argv[6])},
        "hif_corpus_validation": {"path": sys.argv[7], "sha256": sha(sys.argv[7])},
        "hif_corpus_train_extra": {"path": sys.argv[10], "sha256": sha(sys.argv[10])},
        "hif_corpus_validation_extra": {"path": sys.argv[11], "sha256": sha(sys.argv[11])},
        "imbalance_corpus": {"path": sys.argv[8], "sha256": sha(sys.argv[8])},
        "bc0_suite": {"path": sys.argv[9], "sha256": sha(sys.argv[9])},
        "tabular_corpus": {"path": sys.argv[4] + "/data/measurements_5class_merged.jsonl", "sha256": sha(sys.argv[4] + "/data/measurements_5class_merged.jsonl")},
    },
    "environment": {
        name: metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "numpy", "scipy", "OpenDSSDirect.py")
    },
    "tests_run": sys.argv[3] == "1",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(f"prerequisites receipt written: {output}")
PY
