#!/usr/bin/env bash
# Verify every input of the diagnostic round before any GPU work.
#
#   prerequisites.sh --output PATH [--with-tests]
#
# Checks the deployed source commit and cleanliness, the pinned input digests,
# the warm-start adapter, the offline model snapshot, and the research
# environment pins; optionally runs the CPU test files that cover this round's
# code path; then writes one JSON receipt with every digest it observed.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=round.env
source "$HERE/round.env"
round_environment

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

check_sha() {
  local path=$1 expected=$2 actual
  [[ -s "$path" ]] || { echo "missing or empty input: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for $path: $actual != $expected" >&2
    exit 2
  fi
}
check_sha "$D0_RAW" "$D0_RAW_SHA256"
check_sha "$D0_TRAIN" "$D0_TRAIN_SHA256"
check_sha "$VALIDATION" "$VALIDATION_SHA256"
check_sha "$PROTECTED_D1" "$PROTECTED_D1_SHA256"
for name in adapter_config.json adapter_model.safetensors; do
  [[ -s "$WARM_START/$name" ]] || { echo "warm start is missing $name" >&2; exit 2; }
done
[[ -s "$BC0_SUITE" ]] || { echo "frozen BC0 suite missing: $BC0_SUITE" >&2; exit 2; }
SNAPSHOT="$HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
if [[ ! -s "$SNAPSHOT/config.json" || ! -e "$SNAPSHOT/model.safetensors" ]]; then
  echo "offline model snapshot missing: $SNAPSHOT" >&2
  exit 2
fi
for corpus in \
  artifacts/measurements/hif_multiscan_currents_train_85x10_20260903/samples.jsonl \
  artifacts/measurements/hif_multiscan_currents_17x10_20260903/samples.jsonl \
  artifacts/measurements/out_measurements_imbalance_currents_20260903/samples.jsonl \
  data/measurements_5class_merged.jsonl
do
  [[ -s "$SRC/$corpus" ]] || { echo "deployed source lacks $corpus" >&2; exit 2; }
done

cd "$SRC"
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
    psse_env/dagger/test_error_injectors.py \
    psse_env/dagger/test_research_dagger_minimal.py \
    psse_env/providers/test_scenario_generator.py \
    test_branch_current_analysis.py \
    research/test_hpc_diagnostic_round.py
fi

WARM_DIGEST=$("$PY" -c 'import sys; from psse_env.dagger.release_factories import checkpoint_tree_sha256; print(checkpoint_tree_sha256(sys.argv[1]))' "$WARM_START")
"$PY" - "$OUTPUT" "$actual_commit" "$WARM_DIGEST" "$WITH_TESTS" "$SRC" "$WARM_START" "$SNAPSHOT" \
  "$D0_RAW" "$D0_TRAIN" "$VALIDATION" "$PROTECTED_D1" "$BC0_SUITE" <<'PY'
import datetime
import hashlib
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

output = Path(sys.argv[1])
source = Path(sys.argv[5])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


corpora = {
    name: sha(source / name)
    for name in (
        "artifacts/measurements/hif_multiscan_currents_train_85x10_20260903/samples.jsonl",
        "artifacts/measurements/hif_multiscan_currents_17x10_20260903/samples.jsonl",
        "artifacts/measurements/out_measurements_imbalance_currents_20260903/samples.jsonl",
    )
}
payload = {
    "contract": "research_diagnostic_round_prerequisites_v1",
    "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "warm_start_path": sys.argv[6],
    "warm_start_tree_sha256": sys.argv[3],
    "model_snapshot": sys.argv[7],
    "inputs": {
        "d0_raw": {"path": sys.argv[8], "sha256": sha(Path(sys.argv[8]))},
        "d0_train": {"path": sys.argv[9], "sha256": sha(Path(sys.argv[9]))},
        "validation": {"path": sys.argv[10], "sha256": sha(Path(sys.argv[10]))},
        "protected_d1": {"path": sys.argv[11], "sha256": sha(Path(sys.argv[11]))},
        "bc0_suite": {"path": sys.argv[12], "sha256": sha(Path(sys.argv[12]))},
    },
    "source_corpora_sha256": corpora,
    "environment": {
        name: metadata.version(name)
        for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "numpy", "scipy", "OpenDSSDirect.py")
    },
    "tests_run": sys.argv[4] == "1",
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
