#!/bin/bash
#SBATCH --job-name=bc0_release_eval
#SBATCH --output=/scratch/yx3882/psse_agent/bc0_release_eval_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/bc0_release_eval_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|h100"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu

# Step 5 (persist the pinned base baseline):
#   sbatch --export=ALL,REVIEWED_SOURCE_COMMIT=<freeze-commit>,EVALUATION_MODE=base \
#     submit_dagger_release_eval.sh
# Step 6 (evaluate and compare one exact local LoRA adapter):
#   sbatch --export=ALL,REVIEWED_SOURCE_COMMIT=<freeze-commit>,EVALUATION_MODE=checkpoint,CHECKPOINT_PATH=/absolute/output/lora \
#     submit_dagger_release_eval.sh
# CHECKPOINT_REVISION may name the expected 64-hex tree digest. When omitted it
# is computed here; the factory independently copies and verifies those bytes
# again before PEFT loads them. No mode permits a Hub download or dirty source.

set -euo pipefail
umask 077

REPO_ROOT=${REPO_ROOT:-/scratch/yx3882/psse_agent}
ENV_PREFIX=${ENV_PREFIX:-/scratch/yx3882/.conda/envs/unsloth_sft}
PYTHON=${PYTHON:-$ENV_PREFIX/bin/python}
EVALUATION_MODE=${EVALUATION_MODE:-base}

EVALUATION_SUITE=${EVALUATION_SUITE:-psse_env/dagger/suites/bc0_eval_suite_v1.json}
EVALUATION_POLICY=${EVALUATION_POLICY:-psse_env/dagger/bc0_evaluation_policy.json}
BASE_EVALUATION_ARTIFACT=artifacts/evaluations/base_gemma_evaluation.json
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-}
REVIEWED_SOURCE_COMMIT=${REVIEWED_SOURCE_COMMIT:-}

BASE_MODEL_ID=unsloth/gemma-4-31B-it
BASE_MODEL_REVISION=8a796db4df380b178065ed910849477ff0e99c87
ENV_FACTORY=psse_env.dagger.release_factories:production_environment_factory
POLICY_FACTORY=psse_env.dagger.release_factories:gemma_release_policy_factory
CASE_LOADER=psse_env.dagger.release_factories:deterministic_case_loader

case "$EVALUATION_MODE" in
    base|checkpoint) ;;
    *)
        echo "ERROR: EVALUATION_MODE must be base or checkpoint; got '$EVALUATION_MODE'." >&2
        exit 2
        ;;
esac
if [[ "$EVALUATION_MODE" == "base" && ( -n "$CHECKPOINT_PATH" || -n "$CHECKPOINT_REVISION" ) ]]; then
    echo "ERROR: checkpoint identity variables are invalid in base mode." >&2
    exit 2
fi
if [[ "$EVALUATION_MODE" == "checkpoint" ]]; then
    if [[ -z "$CHECKPOINT_PATH" || "$CHECKPOINT_PATH" != /* ]]; then
        echo "ERROR: checkpoint mode requires an absolute CHECKPOINT_PATH." >&2
        exit 2
    fi
    if [[ -n "$CHECKPOINT_REVISION" && ! "$CHECKPOINT_REVISION" =~ ^[0-9a-fA-F]{64}$ ]]; then
        echo "ERROR: CHECKPOINT_REVISION must be an exact 64-hex tree digest." >&2
        exit 2
    fi
fi
if [[ ! "$REVIEWED_SOURCE_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: REVIEWED_SOURCE_COMMIT must be the externally reviewed 40-hex freeze commit." >&2
    exit 2
fi
REVIEWED_SOURCE_COMMIT=${REVIEWED_SOURCE_COMMIT,,}

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

cd "$REPO_ROOT"
unset PYTHONHOME PYTHONPATH
export HF_HOME=${HF_HOME:-/scratch/yx3882/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yx3882/.cache/torch}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}
BC0_ADAPTER_TMPDIR=${BC0_ADAPTER_TMPDIR:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}
mkdir -p "$BC0_ADAPTER_TMPDIR"
export TMPDIR=$BC0_ADAPTER_TMPDIR

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: reviewed Python environment not found at $PYTHON." >&2
    exit 2
fi
for path in "$EVALUATION_SUITE" "$EVALUATION_POLICY" psse_env/requirements-sft.txt; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required release input not found: $path" >&2
        exit 2
    fi
done

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ! "$GPU_NAMES" =~ (H100|H200) ]]; then
    echo "ERROR: release evaluation requires an allocated H100 or H200; got '$GPU_NAMES'." >&2
    exit 2
fi

# Audit the reviewed Python 3.12 environment without contacting an index.
"$PYTHON" -m pip check
"$PYTHON" - psse_env/requirements-sft.txt <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from packaging.requirements import Requirement
import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, found {sys.version.split()[0]}")
failures = []
for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        failures.append(f"{requirement.name}: not installed")
        continue
    if requirement.specifier and installed not in requirement.specifier:
        failures.append(
            f"{requirement.name}: installed {installed}, requires {requirement.specifier}"
        )
try:
    import opendssdirect as dss

    opendss_banner = str(dss.Basic.Version()).strip()
except Exception as exc:
    opendss_banner = ""
    failures.append(
        "OpenDSSDirect native runtime: "
        f"{type(exc).__name__}: {exc}"
    )
else:
    for marker in (
        "DSS C-API Library version 0.14.5",
        "DSS-Python version: 0.15.7",
        "OpenDSSDirect.py version: 0.9.4",
    ):
        if marker not in opendss_banner:
            failures.append(f"OpenDSSDirect native runtime missing {marker!r}")
if torch.__version__ != "2.10.0+cu128":
    failures.append(f"torch: installed {torch.__version__}, requires 2.10.0+cu128")
if failures:
    raise SystemExit("reviewed release environment mismatch:\n- " + "\n- ".join(failures))
print(
    f"reviewed release environment: Python {sys.version.split()[0]}, "
    f"torch {torch.__version__}, OpenDSS native runtime loaded"
)
PY

# Fail before model load unless the selected policy is the packaged policy,
# the suite has its approved bytes, every requested factory is approved, and
# the source is one clean tracked commit. The validator receives that exact
# commit instead of resolving a potentially changed HEAD after evaluation.
SOURCE_COMMIT=$("$PYTHON" - "$EVALUATION_SUITE" "$EVALUATION_POLICY" \
    "$ENV_FACTORY" "$POLICY_FACTORY" "$CASE_LOADER" <<'PY'
from pathlib import Path
import re
import sys

from psse_env.dagger.evaluation_gate import DEFAULT_POLICY_PATH, load_evaluation_policy
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256

suite_path, policy_path = map(Path, sys.argv[1:3])
selected = load_evaluation_policy(policy_path)
packaged = load_evaluation_policy(DEFAULT_POLICY_PATH)
if stable_json_sha256(selected) != stable_json_sha256(packaged):
    raise SystemExit("selected evaluation policy does not match the packaged policy")
suite_policy = selected["suite_policy"]
if suite_policy.get("status") != "pinned":
    raise SystemExit("packaged evaluation suite policy is not pinned")
if file_sha256(suite_path) != suite_policy.get("approved_suite_sha256"):
    raise SystemExit("evaluation suite bytes do not match the packaged policy")
factory_hash = file_sha256("psse_env/dagger/release_factories.py")
for role, spec in zip(("environment", "model_policy", "case_loader"), sys.argv[3:]):
    identity = {"import_spec": spec, "source_sha256": factory_hash}
    if identity not in selected["approved_factories"].get(role, []):
        raise SystemExit(f"{role} factory is not approved by the packaged policy")
source = git_source_state(Path.cwd())
commit = str(source.get("source_commit") or "").lower()
if source.get("release_eligible_source") is not True or not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise SystemExit("release evaluation requires one clean tracked source commit")
print(commit)
PY
)
if [[ "$SOURCE_COMMIT" != "$REVIEWED_SOURCE_COMMIT" ]]; then
    echo "ERROR: clean checkout $SOURCE_COMMIT is not reviewed freeze commit $REVIEWED_SOURCE_COMMIT." >&2
    exit 2
fi

if [[ "$EVALUATION_MODE" == "base" ]]; then
    MODEL_ID=$BASE_MODEL_ID
    MODEL_REVISION=$BASE_MODEL_REVISION
    ROLE=base-baseline
    EVALUATION_ARTIFACT=$BASE_EVALUATION_ARTIFACT
else
    # checkpoint_tree_sha256 rejects symlinks, hard links, non-regular entries,
    # and files that mutate while read. The release factory repeats this check
    # while copying the exact adapter bytes to its private read-only snapshot.
    CHECKPOINT_IDENTITY=$("$PYTHON" - "$CHECKPOINT_PATH" <<'PY'
import shutil
import sys
import tempfile

from psse_env.dagger.release_factories import inspect_release_checkpoint

inspection = inspect_release_checkpoint(sys.argv[1])
temporary_root = tempfile.gettempdir()
free_bytes = shutil.disk_usage(temporary_root).free
required_bytes = int(inspection["total_bytes"]) + 1024**3
if free_bytes < required_bytes:
    raise SystemExit(
        "insufficient temporary space for private adapter copy: "
        f"need at least {required_bytes} bytes, found {free_bytes} in {temporary_root}"
    )
print(inspection["path"])
print(inspection["tree_sha256"])
PY
    )
    mapfile -t CHECKPOINT_IDENTITY_LINES <<<"$CHECKPOINT_IDENTITY"
    MODEL_ID=${CHECKPOINT_IDENTITY_LINES[0]:-}
    COMPUTED_REVISION=${CHECKPOINT_IDENTITY_LINES[1]:-}
    if [[ ! "$COMPUTED_REVISION" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: checkpoint digest computation returned no valid identity." >&2
        exit 2
    fi
    if [[ -n "$CHECKPOINT_REVISION" && "${CHECKPOINT_REVISION,,}" != "$COMPUTED_REVISION" ]]; then
        echo "ERROR: checkpoint tree digest mismatch: expected ${CHECKPOINT_REVISION,,}, computed $COMPUTED_REVISION." >&2
        exit 2
    fi
    MODEL_REVISION=$COMPUTED_REVISION
    ROLE=checkpoint-promotion
    EVALUATION_ARTIFACT=artifacts/evaluations/checkpoint_${MODEL_REVISION}.json
    if [[ ! -f "$BASE_EVALUATION_ARTIFACT" ]]; then
        echo "ERROR: checkpoint promotion requires base artifact: $BASE_EVALUATION_ARTIFACT" >&2
        exit 2
    fi
fi
GATE_REPORT=${EVALUATION_ARTIFACT}.gate.json

PATH_AUDIT=(
    "$PYTHON" - "$REPO_ROOT" "$EVALUATION_MODE" "$EVALUATION_ARTIFACT"
    "$GATE_REPORT" "$EVALUATION_SUITE" "$EVALUATION_POLICY"
    psse_env/requirements-sft.txt psse_env/dagger/release_factories.py
    "$BASE_EVALUATION_ARTIFACT" "$CHECKPOINT_PATH"
)
"${PATH_AUDIT[@]}" <<'PY'
import sys

from psse_env.dagger.release_launcher import validate_release_evaluation_paths

(
    repo_root,
    mode,
    artifact,
    report,
    suite,
    policy,
    requirements,
    factories,
    reference,
    checkpoint,
) = sys.argv[1:]
validated = validate_release_evaluation_paths(
    repo_root=repo_root,
    mode=mode,
    artifact=artifact,
    report=report,
    protected_inputs=(suite, policy, requirements, factories),
    reference_artifact=reference if mode == "checkpoint" else None,
    checkpoint_path=checkpoint if mode == "checkpoint" else None,
)
print(f"release path audit: {validated}")
PY
mkdir -p artifacts/evaluations

EVALUATE=(
    "$PYTHON" -m psse_env.dagger.evaluate_release
    --input "$EVALUATION_SUITE"
    --output "$EVALUATION_ARTIFACT"
    --env-factory "$ENV_FACTORY"
    --policy-factory "$POLICY_FACTORY"
    --case-loader "$CASE_LOADER"
    --model-id "$MODEL_ID"
    --model-revision "$MODEL_REVISION"
    --protocol canonical
    --seed 20260719
    --max-steps 24
    --required-suite standard_success
    --required-suite forced_error_recovery
    --required-suite partial_success_retention
    --required-suite invalid_action_recovery
    --required-suite efficiency
    --minimum-suites 5
    --minimum-episodes-per-suite 1
    --minimum-roots-per-suite 20
)
GATE=(
    "$PYTHON" -m psse_env.dagger.validate_evaluation
    --role "$ROLE"
    --artifact "$EVALUATION_ARTIFACT"
    --policy "$EVALUATION_POLICY"
    --expected-source-commit "$SOURCE_COMMIT"
    --expected-suite "$EVALUATION_SUITE"
    --expected-protocol canonical
    --expected-model-id "$MODEL_ID"
    --expected-model-revision "$MODEL_REVISION"
    --report-output "$GATE_REPORT"
)
if [[ "$EVALUATION_MODE" == "checkpoint" ]]; then
    GATE+=(
        --reference-artifact "$BASE_EVALUATION_ARTIFACT"
        --reference-model-id "$BASE_MODEL_ID"
        --reference-model-revision "$BASE_MODEL_REVISION"
    )
fi

echo "===== BC0 closed-loop release evaluation ====="
echo "job:       ${SLURM_JOB_ID:-interactive}"
echo "host:      $(hostname)"
echo "gpu:       $GPU_NAMES"
echo "mode:      $EVALUATION_MODE"
echo "source:    $SOURCE_COMMIT"
echo "model:     $MODEL_ID@$MODEL_REVISION"
echo "artifact:  $EVALUATION_ARTIFACT"
echo "gate:      $GATE_REPORT"
echo "downloads: disabled"
printf 'evaluate command:'
printf ' %q' "${EVALUATE[@]}"
printf '\n'
"${EVALUATE[@]}"
printf 'gate command:'
printf ' %q' "${GATE[@]}"
printf '\n'
"${GATE[@]}"
