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
#SBATCH --constraint="h200|h100|rtx6000"
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yx3882@nyu.edu

# Step 4 (persist the observable-expert baseline; CPU-sufficient, submit with
# for example --gres=gpu:0 --constraint= to skip the GPU allocation):
#   sbatch --export=ALL,REVIEWED_SOURCE_COMMIT=<freeze-commit>,EVALUATION_MODE=expert \
#     submit_dagger_release_eval.sh
# Step 5 (persist the pinned base baseline; add
# EVALUATION_SCOPE=development_holdout and the three DEVELOPMENT_HOLDOUT_*
# inputs to materialize the diagnostic study scope instead):
#   sbatch --constraint=rtx6000 --cpus-per-task=4 --mem=128G \
#     --export=ALL,EXPECTED_ACCELERATOR_CLASS=rtx6000,REVIEWED_SOURCE_COMMIT=<freeze-commit>,EVALUATION_MODE=base \
#     submit_dagger_release_eval.sh
# Step 6 (evaluate and compare one exact local LoRA adapter and its receipt):
#   sbatch --export=ALL,REVIEWED_SOURCE_COMMIT=<freeze-commit>,EVALUATION_MODE=checkpoint,STUDY_VARIANT=bc0,TRAIN_SEED=3407,CHECKPOINT_PATH=/absolute/output/lora,CHECKPOINT_RECEIPT=/absolute/output/checkpoint_receipt.json \
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
EVALUATION_SCOPE=${EVALUATION_SCOPE:-frozen_suite}
EXPECTED_ACCELERATOR_CLASS=${EXPECTED_ACCELERATOR_CLASS:-auto}

FROZEN_EVALUATION_SUITE=${EVALUATION_SUITE:-psse_env/dagger/suites/bc0_eval_suite_v1.json}
EVALUATION_POLICY=${EVALUATION_POLICY:-psse_env/dagger/bc0_evaluation_policy.json}
STUDY_MANIFEST=${STUDY_MANIFEST:-psse_env/dagger/studies/dagger_multiseed_study_v1.json}
STUDY_VARIANT=${STUDY_VARIANT:-}
TRAIN_SEED=${TRAIN_SEED:-}
CHECKPOINT_RECEIPT=${CHECKPOINT_RECEIPT:-}
DEVELOPMENT_HOLDOUT=${DEVELOPMENT_HOLDOUT:-}
DEVELOPMENT_HOLDOUT_MANIFEST=${DEVELOPMENT_HOLDOUT_MANIFEST:-}
DEVELOPMENT_HOLDOUT_GENERATOR_REPORT=${DEVELOPMENT_HOLDOUT_GENERATOR_REPORT:-}
BASE_EVALUATION_ARTIFACT=artifacts/evaluations/base_gemma_evaluation.json
BASE_DEVELOPMENT_ARTIFACT=artifacts/evaluations/development_base_gemma_evaluation.json
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-}
REVIEWED_SOURCE_COMMIT=${REVIEWED_SOURCE_COMMIT:-}

BASE_MODEL_ID=unsloth/gemma-4-31B-it
BASE_MODEL_REVISION=8a796db4df380b178065ed910849477ff0e99c87
EXPERT_POLICY_IDENTITY=bc0-observable-handoff-expert-v2
EXPERT_EVALUATION_ARTIFACT=artifacts/evaluations/expert_baseline_evaluation.json
ENV_FACTORY=psse_env.dagger.release_factories:production_environment_factory
CASE_LOADER=psse_env.dagger.release_factories:deterministic_case_loader
if [[ "$EVALUATION_MODE" == "expert" ]]; then
    POLICY_FACTORY=psse_env.dagger.release_factories:observable_expert_policy_factory
    POLICY_FACTORY_ROLE=expert_policy
else
    POLICY_FACTORY=psse_env.dagger.release_factories:gemma_release_policy_factory
    POLICY_FACTORY_ROLE=model_policy
fi

case "$EVALUATION_MODE" in
    expert|base|checkpoint) ;;
    *)
        echo "ERROR: EVALUATION_MODE must be expert, base, or checkpoint; got '$EVALUATION_MODE'." >&2
        exit 2
        ;;
esac
case "$EVALUATION_SCOPE" in
    frozen_suite|development_holdout) ;;
    *)
        echo "ERROR: EVALUATION_SCOPE must be frozen_suite or development_holdout; got '$EVALUATION_SCOPE'." >&2
        exit 2
        ;;
esac
case "$EXPECTED_ACCELERATOR_CLASS" in
    auto|h100|h200|rtx6000) ;;
    *)
        echo "ERROR: EXPECTED_ACCELERATOR_CLASS must be auto, h100, h200, or rtx6000." >&2
        exit 2
        ;;
esac
if [[ -z "$STUDY_MANIFEST" ]]; then
    echo "ERROR: model study evaluation requires STUDY_MANIFEST." >&2
    exit 2
fi
if [[ "$EVALUATION_MODE" == "expert" && "$EVALUATION_SCOPE" != "frozen_suite" ]]; then
    echo "ERROR: the non-model expert baseline has no development study role." >&2
    exit 2
fi
if [[ "$EVALUATION_MODE" != "checkpoint" && ( -n "$CHECKPOINT_PATH" || -n "$CHECKPOINT_REVISION" ) ]]; then
    echo "ERROR: checkpoint identity variables are only valid in checkpoint mode." >&2
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
if [[ "$EVALUATION_MODE" == "expert" ]]; then
    if [[ -n "$STUDY_VARIANT" || -n "$TRAIN_SEED" || -n "$CHECKPOINT_RECEIPT" ]]; then
        echo "ERROR: expert evaluation cannot claim model-study checkpoint fields." >&2
        exit 2
    fi
elif [[ "$EVALUATION_MODE" == "base" ]]; then
    if [[ -n "$STUDY_VARIANT" && "$STUDY_VARIANT" != "base" ]]; then
        echo "ERROR: base evaluation STUDY_VARIANT must be base." >&2
        exit 2
    fi
    if [[ -n "$TRAIN_SEED" || -n "$CHECKPOINT_RECEIPT" ]]; then
        echo "ERROR: base study evaluation requires canonical null seed and receipt." >&2
        exit 2
    fi
    STUDY_VARIANT=base
else
    case "$STUDY_VARIANT" in
        bc0|natural_dagger|natural_dagger_probes) ;;
        *)
            echo "ERROR: checkpoint mode requires a trained STUDY_VARIANT." >&2
            exit 2
            ;;
    esac
    case "$TRAIN_SEED" in
        3407|3408|3409) ;;
        *)
            echo "ERROR: checkpoint study TRAIN_SEED must be 3407, 3408, or 3409." >&2
            exit 2
            ;;
    esac
    if [[ -z "$CHECKPOINT_RECEIPT" || "$CHECKPOINT_RECEIPT" != /* ]]; then
        echo "ERROR: checkpoint study evaluation requires an absolute CHECKPOINT_RECEIPT." >&2
        exit 2
    fi
fi
if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
    if [[ -z "$DEVELOPMENT_HOLDOUT" || -z "$DEVELOPMENT_HOLDOUT_MANIFEST" || -z "$DEVELOPMENT_HOLDOUT_GENERATOR_REPORT" ]]; then
        echo "ERROR: development scope requires DEVELOPMENT_HOLDOUT, DEVELOPMENT_HOLDOUT_MANIFEST, and DEVELOPMENT_HOLDOUT_GENERATOR_REPORT." >&2
        exit 2
    fi
    EVALUATION_SUITE=$DEVELOPMENT_HOLDOUT
else
    if [[ -n "$DEVELOPMENT_HOLDOUT" || -n "$DEVELOPMENT_HOLDOUT_MANIFEST" || -n "$DEVELOPMENT_HOLDOUT_GENERATOR_REPORT" ]]; then
        echo "ERROR: development holdout inputs require EVALUATION_SCOPE=development_holdout." >&2
        exit 2
    fi
    EVALUATION_SUITE=$FROZEN_EVALUATION_SUITE
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
REQUIRED_INPUTS=(
    "$EVALUATION_SUITE"
    "$EVALUATION_POLICY"
    psse_env/requirements-sft.txt
)
if [[ "$EVALUATION_MODE" != "expert" ]]; then
    REQUIRED_INPUTS+=("$STUDY_MANIFEST")
fi
if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
    REQUIRED_INPUTS+=(
        "$DEVELOPMENT_HOLDOUT_MANIFEST"
        "$DEVELOPMENT_HOLDOUT_GENERATOR_REPORT"
    )
fi
if [[ "$EVALUATION_MODE" == "checkpoint" ]]; then
    REQUIRED_INPUTS+=("$CHECKPOINT_RECEIPT")
fi
for path in "${REQUIRED_INPUTS[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required release input not found: $path" >&2
        exit 2
    fi
done

# The observable expert runs WLS on CPU; only model-backed modes need the
# exact release GPU class. The `rtx6000` Slurm class is admitted only when the
# runtime check confirms RTX Pro 6000 with at least 90,000 MiB.
GPU_INVENTORY=$(nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null || echo "none")
echo "GPU inventory: $GPU_INVENTORY"
if [[ "$EVALUATION_MODE" == "expert" ]]; then
    :
else
    HARDWARE_ARGS=()
    if [[ "$EXPECTED_ACCELERATOR_CLASS" != "auto" ]]; then
        HARDWARE_ARGS+=(--require-class "$EXPECTED_ACCELERATOR_CLASS")
    fi
    "$PYTHON" -m psse_env.sft.release_hardware "${HARDWARE_ARGS[@]}"
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
    "$EVALUATION_SCOPE" "$POLICY_FACTORY_ROLE" "$ENV_FACTORY" "$POLICY_FACTORY" "$CASE_LOADER" <<'PY'
from pathlib import Path
import re
import sys

from psse_env.dagger.evaluation_gate import DEFAULT_POLICY_PATH, load_evaluation_policy
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256

suite_path, policy_path = map(Path, sys.argv[1:3])
evaluation_scope = sys.argv[3]
policy_factory_role = sys.argv[4]
selected = load_evaluation_policy(policy_path)
packaged = load_evaluation_policy(DEFAULT_POLICY_PATH)
if stable_json_sha256(selected) != stable_json_sha256(packaged):
    raise SystemExit("selected evaluation policy does not match the packaged policy")
suite_policy = selected["suite_policy"]
if suite_policy.get("status") != "pinned":
    raise SystemExit("packaged evaluation suite policy is not pinned")
if (
    evaluation_scope == "frozen_suite"
    and file_sha256(suite_path) != suite_policy.get("approved_suite_sha256")
):
    raise SystemExit("evaluation suite bytes do not match the packaged policy")
factory_hash = file_sha256("psse_env/dagger/release_factories.py")
for role, spec in zip(("environment", policy_factory_role, "case_loader"), sys.argv[5:]):
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

if [[ "$EVALUATION_MODE" == "expert" ]]; then
    MODEL_ID=""
    MODEL_REVISION=""
    ROLE=expert-baseline
    EVALUATION_ARTIFACT=$EXPERT_EVALUATION_ARTIFACT
elif [[ "$EVALUATION_MODE" == "base" ]]; then
    MODEL_ID=$BASE_MODEL_ID
    MODEL_REVISION=$BASE_MODEL_REVISION
    ROLE=base-baseline
    if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
        EVALUATION_ARTIFACT=$BASE_DEVELOPMENT_ARTIFACT
    else
        EVALUATION_ARTIFACT=$BASE_EVALUATION_ARTIFACT
    fi
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
    if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
        EVALUATION_ARTIFACT=artifacts/evaluations/development_${STUDY_VARIANT}_seed${TRAIN_SEED}_${MODEL_REVISION}.json
        BASE_REFERENCE_ARTIFACT=$BASE_DEVELOPMENT_ARTIFACT
    else
        EVALUATION_ARTIFACT=artifacts/evaluations/checkpoint_${MODEL_REVISION}.json
        BASE_REFERENCE_ARTIFACT=$BASE_EVALUATION_ARTIFACT
    fi
    if [[ ! -f "$BASE_REFERENCE_ARTIFACT" ]]; then
        echo "ERROR: checkpoint evaluation requires matching base artifact: $BASE_REFERENCE_ARTIFACT" >&2
        exit 2
    fi
fi
GATE_REPORT=${EVALUATION_ARTIFACT}.gate.json
if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
    GATE_REPORT=${EVALUATION_ARTIFACT}.study.json
fi
REFERENCE_ARTIFACT=${BASE_REFERENCE_ARTIFACT:-$BASE_EVALUATION_ARTIFACT}

PATH_AUDIT=(
    "$PYTHON" - "$REPO_ROOT" "$EVALUATION_MODE" "$EVALUATION_ARTIFACT"
    "$GATE_REPORT" "$EVALUATION_SUITE" "$EVALUATION_POLICY"
    psse_env/requirements-sft.txt psse_env/dagger/release_factories.py
    "$STUDY_MANIFEST" "$DEVELOPMENT_HOLDOUT_MANIFEST"
    "$DEVELOPMENT_HOLDOUT_GENERATOR_REPORT" "$CHECKPOINT_RECEIPT"
    "$REFERENCE_ARTIFACT" "$CHECKPOINT_PATH"
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
    study_manifest,
    development_manifest,
    development_generator_report,
    checkpoint_receipt,
    reference,
    checkpoint,
) = sys.argv[1:]
protected = [suite, policy, requirements, factories]
protected.extend(
    value
    for value in (
        study_manifest,
        development_manifest,
        development_generator_report,
        checkpoint_receipt,
    )
    if value
)
validated = validate_release_evaluation_paths(
    repo_root=repo_root,
    mode=mode,
    artifact=artifact,
    report=report,
    protected_inputs=protected,
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
    --protocol canonical
)
if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
    EVALUATE+=(
        --diagnostic-only
        --seed 20260721
        --max-steps 24
        --required-suite dagger1_development
        --minimum-suites 1
        --minimum-episodes-per-suite 1
        --minimum-roots-per-suite 30
        --development-holdout-manifest "$DEVELOPMENT_HOLDOUT_MANIFEST"
        --development-holdout-generator-report "$DEVELOPMENT_HOLDOUT_GENERATOR_REPORT"
    )
else
    EVALUATE+=(
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
fi
GATE=(
    "$PYTHON" -m psse_env.dagger.validate_evaluation
    --role "$ROLE"
    --artifact "$EVALUATION_ARTIFACT"
    --policy "$EVALUATION_POLICY"
    --expected-source-commit "$SOURCE_COMMIT"
    --expected-suite "$EVALUATION_SUITE"
    --expected-protocol canonical
    --report-output "$GATE_REPORT"
)
if [[ "$EVALUATION_MODE" == "expert" ]]; then
    EVALUATE+=(--policy-identity "$EXPERT_POLICY_IDENTITY")
    GATE+=(--expected-policy-identity "$EXPERT_POLICY_IDENTITY")
else
    EVALUATE+=(
        --model-id "$MODEL_ID"
        --model-revision "$MODEL_REVISION"
        --study-manifest "$STUDY_MANIFEST"
        --study-variant "$STUDY_VARIANT"
        --reviewed-source-commit "$SOURCE_COMMIT"
    )
    GATE+=(--expected-model-id "$MODEL_ID" --expected-model-revision "$MODEL_REVISION")
fi
if [[ "$EVALUATION_MODE" == "checkpoint" ]]; then
    EVALUATE+=(
        --training-seed "$TRAIN_SEED"
        --checkpoint-receipt "$CHECKPOINT_RECEIPT"
    )
    GATE+=(
        --reference-artifact "$REFERENCE_ARTIFACT"
        --reference-model-id "$BASE_MODEL_ID"
        --reference-model-revision "$BASE_MODEL_REVISION"
    )
fi

echo "===== BC0 closed-loop release evaluation ====="
echo "job:       ${SLURM_JOB_ID:-interactive}"
echo "host:      $(hostname)"
echo "gpu:       $GPU_INVENTORY"
echo "mode:      $EVALUATION_MODE"
echo "scope:     $EVALUATION_SCOPE"
echo "GPU class: $EXPECTED_ACCELERATOR_CLASS"
echo "source:    $SOURCE_COMMIT"
if [[ "$EVALUATION_MODE" == "expert" ]]; then
    echo "policy:    $EXPERT_POLICY_IDENTITY"
else
    echo "model:     $MODEL_ID@$MODEL_REVISION"
    echo "variant:   $STUDY_VARIANT"
    echo "train seed:${TRAIN_SEED:-null}"
fi
echo "artifact:  $EVALUATION_ARTIFACT"
echo "gate:      $GATE_REPORT"
echo "downloads: disabled"
printf 'evaluate command:'
printf ' %q' "${EVALUATE[@]}"
printf '\n'
"${EVALUATE[@]}"
if [[ "$EVALUATION_SCOPE" == "development_holdout" ]]; then
    STUDY_SEED=${TRAIN_SEED:-null}
    "$PYTHON" - "$EVALUATION_ARTIFACT" "$STUDY_MANIFEST" \
        "$STUDY_VARIANT" "$STUDY_SEED" "$SOURCE_COMMIT" "$GATE_REPORT" <<'PY'
import json
import os
from pathlib import Path
import sys

from psse_env.dagger.study_manifest import load_study_manifest
from psse_env.dagger.study_metrics import extract_artifact_metrics
from psse_env.sft.provenance import stable_json_sha256

artifact_path = Path(sys.argv[1])
manifest = load_study_manifest(sys.argv[2])
variant = sys.argv[3]
seed = None if sys.argv[4] == "null" else int(sys.argv[4])
metrics = extract_artifact_metrics(
    artifact_path,
    variant_id=variant,
    study_seed=seed,
    evaluation_scope="development_holdout",
    study_manifest=manifest,
    expected_source_commit=sys.argv[5],
)
report = {
    "contract": "dagger_study_evaluation_ingestion_report_v1",
    "artifact_role": "development_evaluation",
    "variant_id": variant,
    "training_seed": seed,
    "study_manifest_sha256": manifest["manifest_sha256"],
    "artifact_content_sha256": metrics["artifact_content_sha256"],
    "scope_binding": metrics["scope_binding"],
    "input_suite_sha256": metrics["input_suite_sha256"],
    "evaluator_seed": metrics["evaluator_seed"],
    "max_steps": metrics["max_steps"],
    "physical_root_set_sha256": metrics["physical_root_set_sha256"],
    "metric_contract": metrics["metric_contract"],
    "metrics": metrics["metrics"],
}
report["content_sha256"] = stable_json_sha256(report)
serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
report_path = Path(sys.argv[6])
descriptor = os.open(
    report_path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
    0o600,
)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    handle.write(serialized)
    handle.flush()
    os.fsync(handle.fileno())
if os.name != "nt":
    directory_descriptor = os.open(
        report_path.parent,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
print(json.dumps({"study_ingestion_report": str(report_path), **report}, sort_keys=True))
PY
else
    printf 'gate command:'
    printf ' %q' "${GATE[@]}"
    printf '\n'
    "${GATE[@]}"
fi
