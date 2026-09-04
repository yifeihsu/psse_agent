"""Fail-closed contract for the preregistered four-variant DAgger study.

The checked-in JSON is content-addressed here.  Its source binding deliberately
requires an externally reviewed clean Git commit instead of embedding a commit
hash in the same commit (which would be self-referential).  Run/checkpoint
artifacts must materialize that exact commit and bind this manifest's digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


STUDY_MANIFEST_CONTRACT = "dagger_multiseed_four_variant_study_v1"
STUDY_ID = "dagger_multi_error_comparison_v1"
DEFAULT_STUDY_MANIFEST = (
    Path(__file__).resolve().parent
    / "studies"
    / "dagger_multiseed_study_v1.json"
)
# Updated only through explicit protocol review.  This pins the raw LF-normalized
# bytes; .gitattributes preserves that representation on Windows and Linux.
EXPECTED_STUDY_MANIFEST_SHA256 = (
    "206ee008477dac92f53744b3107231e15e8d7de2064d3989c7e9687daa7853b3"
)
EXPECTED_STUDY_MANIFEST_CONTENT_SHA256 = (
    "e588f6f98c3d1423d0756c7baa41b4cf0cf13336407f02b593c2622c4ac18191"
)
EXPECTED_COMPARISON_POLICY_SHA256 = (
    "9763dc426de33e328a06cd5abfb4f5788a05ef91fac6cb4113e30680f8c2c550"
)
EXPECTED_OBJECTIVE_THRESHOLDS_SHA256 = (
    "1d5b32c6cde50b1c554878f9b3b20853d3c35e50c698de1c80edb409c4af2189"
)
EXPECTED_STABILITY_SCOPE_POLICY_SHA256 = (
    "f713c5763157f0dd88751fd605f6cb63c71d2fb5abd83a6be91a661d31ce9a00"
)

PINNED_BASE_MODEL_ID = "unsloth/gemma-4-31B-it"
PINNED_BASE_MODEL_REVISION = "8a796db4df380b178065ed910849477ff0e99c87"
TRAINING_PROTOCOL_CONTRACT = "dagger_study_training_protocol_v1"
TRAINING_RNG_CONTRACT = "dagger_training_rng_attestation_v1"
TRAINING_DEPENDENCY_LOCK_PATH = "psse_env/requirements-sft.txt"
TRAINING_DEPENDENCY_LOCK_SHA256 = (
    "58c1f4690803b0109d47ac81ae613e07d883a1cd5a14cbde0409f310d2dd4df5"
)
TRAINING_RNG_ENGINES = (
    "python_random",
    "numpy_random",
    "torch_cpu",
    "torch_cuda_all",
)
# Research-mode binding: these pins follow the current BC0 freeze (2026-09-03
# re-freeze under the 1.2 dominance contract) rather than the originally
# preregistered instrument.  Re-pin them together with bc0_evaluation_policy.json
# whenever the suite is rebuilt.
PINNED_SUITE_PATH = "psse_env/dagger/suites/bc0_eval_suite_v1.json"
PINNED_SUITE_SHA256 = (
    "613bba87413071782786fa18089624f0f5d431c98d783a9a24203bd8c76c029a"
)
PINNED_POLICY_PATH = "psse_env/dagger/bc0_evaluation_policy.json"
PINNED_POLICY_SHA256 = (
    "18ac8330f17f58bb20757237c8b2f50befb20784097224d1a94dd54101ab3852"
)
DEVELOPMENT_EVALUATION_PROTOCOL_CONTRACT = (
    "dagger_development_evaluation_protocol_v1"
)
EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256 = (
    "3aa33a88fa22f91eb8f0a7a5622cf30d5574af2c57f7ac9bda8614b30f8bb645"
)
_CANONICAL_DEVELOPMENT_EVALUATION_CONTRACT = {
    "contract": DEVELOPMENT_EVALUATION_PROTOCOL_CONTRACT,
    "evaluation_protocol": "diagnostic_model_selection_only",
    "diagnostic_only": True,
    "input_suite_name": "dagger1_development",
    "evaluator_seed": 20260721,
    "max_steps": 24,
    "required_suites": ["dagger1_development"],
    "minimum_suites": 1,
    "minimum_episodes_per_suite": 1,
    "minimum_roots_per_suite": 30,
    "exact_physical_roots": 30,
    "protocol": "canonical",
    "release_qualification_allowed": False,
}
RECOVERY_STRESS_EVALUATION_PROTOCOL_CONTRACT = (
    "dagger_recovery_stress_evaluation_protocol_v1"
)
EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256 = (
    "4a65b9950ef273d5ca5b4c1fc80e0b4831880bc53b327940688eb7fe3cfb9a19"
)
_RECOVERY_STRESS_SUITE_NAMES = (
    "recovery_measurement_parameter_sequential_handoff",
    "recovery_post_failure_no_candidate",
    "recovery_premature_commit",
    "recovery_premature_escalation",
    "recovery_rejected_candidate_rollback",
    "recovery_safe_continuation_after_partial_success",
    "recovery_unsupported_correction",
)
_CANONICAL_RECOVERY_STRESS_EVALUATION_CONTRACT = {
    "contract": RECOVERY_STRESS_EVALUATION_PROTOCOL_CONTRACT,
    "evaluation_protocol": "preregistered_recovery_stress_test",
    "diagnostic_only": False,
    "input_suite_names": list(_RECOVERY_STRESS_SUITE_NAMES),
    "evaluator_seed": 20260723,
    "max_steps": 24,
    "required_suites": list(_RECOVERY_STRESS_SUITE_NAMES),
    "minimum_suites": 7,
    "minimum_episodes_per_suite": 10,
    "minimum_roots_per_suite": 10,
    "exact_episode_count": 70,
    "exact_physical_roots": 20,
    "development_parent_subset_required": True,
    "zero_training_probe_frozen_overlap_required": True,
    "protocol": "canonical",
    "release_qualification_allowed": True,
}

REQUIRED_VARIANT_IDS = (
    "base",
    "bc0",
    "natural_dagger",
    "natural_dagger_probes",
)
TRAINED_VARIANT_IDS = REQUIRED_VARIANT_IDS[1:]
EXPECTED_TRAINING_SOURCES = {
    "base": (),
    "bc0": ("d0_bc0",),
    "natural_dagger": ("d0_bc0", "natural_dagger1"),
    "natural_dagger_probes": (
        "d0_bc0",
        "natural_dagger1",
        "observable_recovery_probe",
    ),
}
PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT = (
    "round1_production_d1_quarantine_binding_v1"
)
PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME = (
    "d1_offline_teacher_target_quarantine_summary"
)
PRODUCTION_D1_QUARANTINE_APPLICABLE_VARIANTS = (
    "natural_dagger",
    "natural_dagger_probes",
)
_PRODUCTION_D1_QUARANTINE_BINDING_FIELDS = frozenset(
    {
        "contract",
        "status",
        "reason",
        "generation_provenance_id",
        "generation_descriptor",
        "audit_report_name",
        "audit_report_sha256",
        "candidate_rows",
        "quarantined_rows",
        "summary",
    }
)
MAX_TRAINING_SEED = (2**32) - 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _training_configuration(*, learning_rate: float, epochs: float) -> dict[str, Any]:
    """Return the reviewed material settings shared by one study variant."""

    return {
        "model": {
            "model_id": PINNED_BASE_MODEL_ID,
            "model_revision": PINNED_BASE_MODEL_REVISION,
            "loader": "AutoModelForImageTextToText",
            "local_files_only": True,
            "trust_remote_code": False,
            "device_map": "auto",
        },
        "processor": {
            "model_id": PINNED_BASE_MODEL_ID,
            "model_revision": PINNED_BASE_MODEL_REVISION,
            "loader": "AutoProcessor",
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "max_length": 8192,
        "allow_prompt_truncation": False,
        "lora": {
            "rank": 16,
            "alpha": 16,
            "dropout": 0.0,
            "target_module_suffixes": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "use_double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "precision": {"bf16": True, "fp16": False},
        "trainer": {
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "max_steps": -1,
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "linear",
            "packing": False,
            "completion_only_loss": False,
            "remove_unused_columns": False,
            "skip_prepare_dataset": True,
        },
    }


EXPECTED_TRAINING_PROTOCOL_CONFIGURATIONS = {
    "bc0": _training_configuration(learning_rate=0.0001, epochs=2.0),
    "natural_dagger": _training_configuration(learning_rate=0.00003, epochs=1.0),
    "natural_dagger_probes": _training_configuration(
        learning_rate=0.00003,
        epochs=1.0,
    ),
}


class StudyManifestError(ValueError):
    """Raised when study preregistration or a bound file fails closed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_development_evaluation_contract() -> dict[str, Any]:
    """Return the sole preregistered diagnostic evaluator configuration."""

    contract = json.loads(
        json.dumps(
            _CANONICAL_DEVELOPMENT_EVALUATION_CONTRACT,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if _content_sha256(contract) != EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256:
        raise StudyManifestError(
            "internal development evaluator contract digest is inconsistent"
        )
    return contract


def canonical_recovery_stress_evaluation_contract() -> dict[str, Any]:
    """Return the sole preregistered recovery-stress configuration."""

    contract = json.loads(
        json.dumps(
            _CANONICAL_RECOVERY_STRESS_EVALUATION_CONTRACT,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if (
        _content_sha256(contract)
        != EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256
    ):
        raise StudyManifestError(
            "internal recovery-stress evaluator contract digest is inconsistent"
        )
    return contract


def _source_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields derived by :func:`load_study_manifest`."""

    return {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_sha256", "validation"}
    }


def study_manifest_sha256(path: str | Path = DEFAULT_STUDY_MANIFEST) -> str:
    """Hash the immutable study-manifest bytes."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise StudyManifestError(f"study manifest is missing: {manifest_path}")
    return _file_sha256(manifest_path)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyManifestError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StudyManifestError(f"{field} must be an array")
    return value


def _require_exact_sequence(value: Any, expected: Sequence[Any], *, field: str) -> None:
    observed = tuple(_sequence(value, field=field))
    if observed != tuple(expected):
        raise StudyManifestError(
            f"{field} must be exactly {list(expected)!r}; got {list(observed)!r}"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_absolute_portable_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def build_training_protocol_binding(
    manifest: Mapping[str, Any],
    *,
    variant_id: str,
) -> dict[str, Any]:
    """Materialize the exact protocol a trained checkpoint must receipt-bind."""

    if variant_id not in TRAINED_VARIANT_IDS:
        raise StudyManifestError(
            f"training protocol is not applicable to variant {variant_id!r}"
        )
    policy = _mapping(
        manifest.get("training_protocol_policy"),
        field="training_protocol_policy",
    )
    dependency_lock = _mapping(
        policy.get("dependency_lock"),
        field="training_protocol_policy.dependency_lock",
    )
    protocols = _mapping(
        policy.get("variant_protocols"),
        field="training_protocol_policy.variant_protocols",
    )
    configuration = _mapping(
        protocols.get(variant_id),
        field=f"training_protocol_policy.variant_protocols.{variant_id}",
    )
    # Round-trip through strict JSON to return a detached, JSON-native value.
    return json.loads(
        json.dumps(
            {
                "contract": policy.get("contract"),
                "variant_id": variant_id,
                "dependency_lock": dict(dependency_lock),
                "configuration": dict(configuration),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def canonical_training_rng_attestation(
    *,
    variant_id: str,
    training_seed: int,
) -> dict[str, Any]:
    """Return the only accepted RNG receipt shape for cold or warm training."""

    if variant_id not in TRAINED_VARIANT_IDS:
        raise StudyManifestError(
            f"training RNG attestation is not applicable to {variant_id!r}"
        )
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or not 0 <= training_seed <= MAX_TRAINING_SEED
    ):
        raise StudyManifestError("training RNG seed is outside the uint32 contract")
    applied = {
        "status": "applied",
        "seed": training_seed,
        "engines": list(TRAINING_RNG_ENGINES),
    }
    if variant_id == "bc0":
        cold_adapter = dict(applied)
    else:
        cold_adapter = {
            "status": "not_applicable",
            "seed": None,
            "engines": [],
            "reason": "warm_start_loads_existing_bc0_adapter",
        }
    return {
        "contract": TRAINING_RNG_CONTRACT,
        "training_seed": training_seed,
        "pre_model_construction": dict(applied),
        "cold_adapter_attachment": cold_adapter,
    }


def canonical_production_d1_quarantine_binding(variant_id: str) -> dict[str, Any]:
    """Return the sole canonical N/A binding for variants without Round-1 D1."""

    if variant_id not in {"base", "bc0"}:
        raise StudyManifestError(
            "only base and BC0 have a not-applicable production-D1 binding"
        )
    return {
        "contract": PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT,
        "status": "not_applicable",
        "reason": "variant_does_not_consume_round1_production_d1",
        "generation_provenance_id": None,
        "generation_descriptor": None,
        "audit_report_name": None,
        "audit_report_sha256": None,
        "candidate_rows": None,
        "quarantined_rows": None,
        "summary": None,
    }


def validate_production_d1_quarantine_binding(
    value: Any,
    *,
    variant_id: str,
    expected_generation_provenance_id: str | None,
) -> dict[str, Any]:
    """Validate exact variant semantics and recompute the bound audit digest."""

    binding = dict(
        _mapping(value, field="checkpoint.production_d1_quarantine_binding")
    )
    if set(binding) != _PRODUCTION_D1_QUARANTINE_BINDING_FIELDS:
        raise StudyManifestError(
            "checkpoint production-D1 quarantine binding fields are not exact"
        )
    if binding.get("contract") != PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT:
        raise StudyManifestError(
            "checkpoint production-D1 quarantine binding contract is invalid"
        )
    if variant_id in {"base", "bc0"}:
        expected = canonical_production_d1_quarantine_binding(variant_id)
        if binding != expected:
            raise StudyManifestError(
                f"{variant_id} production-D1 quarantine binding must be canonical "
                "not-applicable/null"
            )
        return binding
    if variant_id not in PRODUCTION_D1_QUARANTINE_APPLICABLE_VARIANTS:
        raise StudyManifestError(
            f"unknown production-D1 quarantine binding variant: {variant_id!r}"
        )
    if binding.get("status") != "applicable" or binding.get("reason") is not None:
        raise StudyManifestError(
            f"{variant_id} production-D1 quarantine binding must be applicable"
        )
    provenance_id = str(binding.get("generation_provenance_id") or "")
    if (
        _SHA256_RE.fullmatch(provenance_id) is None
        or provenance_id != expected_generation_provenance_id
    ):
        raise StudyManifestError(
            "checkpoint production-D1 quarantine provenance differs from the "
            "training view"
        )
    descriptor = _mapping(
        binding.get("generation_descriptor"),
        field="checkpoint production-D1 generation_descriptor",
    )
    if (
        not descriptor
        or _content_sha256(descriptor) != provenance_id
        or descriptor.get("builder_contract")
        != "deterministic_d0_d1_probe_balanced_union_v2"
    ):
        raise StudyManifestError(
            "checkpoint production-D1 generation descriptor does not hash to "
            "the training-view provenance"
        )
    if (
        binding.get("audit_report_name")
        != PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME
    ):
        raise StudyManifestError(
            "checkpoint production-D1 quarantine audit report name is invalid"
        )
    summary = binding.get("summary")
    try:
        # Keep the collection contract authoritative rather than maintaining a
        # second, weaker interpretation in the study layer.  The import stays
        # local because the aggregate builder imports training helpers.
        from psse_env.dagger.build_dagger1_aggregate import (
            validate_offline_teacher_target_quarantine_summary,
        )

        validated_summary = validate_offline_teacher_target_quarantine_summary(
            summary
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise StudyManifestError(
            f"checkpoint production-D1 quarantine summary is invalid: {exc}"
        ) from exc
    if binding.get("candidate_rows") != validated_summary["candidate_rows"]:
        raise StudyManifestError(
            "checkpoint production-D1 quarantine candidate count differs from "
            "the summary"
        )
    if binding.get("quarantined_rows") != validated_summary["quarantined_rows"]:
        raise StudyManifestError(
            "checkpoint production-D1 quarantine count differs from the summary"
        )
    report_sha256 = str(binding.get("audit_report_sha256") or "")
    if (
        _SHA256_RE.fullmatch(report_sha256) is None
        or report_sha256 != _content_sha256(validated_summary)
    ):
        raise StudyManifestError(
            "checkpoint production-D1 quarantine report hash does not match the "
            "summary"
        )
    descriptor_audit_hashes = _mapping(
        descriptor.get("audit_report_sha256"),
        field="checkpoint production-D1 descriptor audit_report_sha256",
    )
    if descriptor_audit_hashes.get(
        PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME
    ) != report_sha256:
        raise StudyManifestError(
            "checkpoint production-D1 quarantine report hash is not the hash "
            "authenticated by the generation descriptor"
        )
    return binding


def build_production_d1_quarantine_binding(
    *,
    variant_id: str,
    generation_provenance_id: str | None = None,
    generation_descriptor: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    audit_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Construct and validate one canonical receipt/report binding."""

    if variant_id in {"base", "bc0"}:
        if any(
            value is not None
            for value in (
                generation_provenance_id,
                generation_descriptor,
                summary,
                audit_report_sha256,
            )
        ):
            raise StudyManifestError(
                f"{variant_id} production-D1 quarantine binding cannot carry evidence"
            )
        return canonical_production_d1_quarantine_binding(variant_id)
    if not isinstance(summary, Mapping):
        raise StudyManifestError(
            f"{variant_id} production-D1 quarantine binding requires a summary"
        )
    materialized_summary = dict(summary)
    binding = {
        "contract": PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT,
        "status": "applicable",
        "reason": None,
        "generation_provenance_id": generation_provenance_id,
        "generation_descriptor": (
            dict(generation_descriptor)
            if isinstance(generation_descriptor, Mapping)
            else generation_descriptor
        ),
        "audit_report_name": PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME,
        "audit_report_sha256": audit_report_sha256,
        "candidate_rows": materialized_summary.get("candidate_rows"),
        "quarantined_rows": materialized_summary.get("quarantined_rows"),
        "summary": materialized_summary,
    }
    return validate_production_d1_quarantine_binding(
        binding,
        variant_id=variant_id,
        expected_generation_provenance_id=generation_provenance_id,
    )


def validate_study_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    """Validate the complete semantic contract and its bound suite/policy."""

    if manifest.get("schema_version") != 1:
        raise StudyManifestError("study manifest schema_version must be 1")
    if manifest.get("contract") != STUDY_MANIFEST_CONTRACT:
        raise StudyManifestError(
            f"study manifest contract must be {STUDY_MANIFEST_CONTRACT!r}"
        )
    if manifest.get("study_id") != STUDY_ID:
        raise StudyManifestError(f"study_id must be {STUDY_ID!r}")
    if manifest.get("status") != "preregistered":
        raise StudyManifestError("study manifest status must be 'preregistered'")

    seeds = list(_sequence(manifest.get("training_seeds"), field="training_seeds"))
    if len(seeds) < 3:
        raise StudyManifestError("training_seeds must contain at least three seeds")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > MAX_TRAINING_SEED
        for seed in seeds
    ):
        raise StudyManifestError(
            "training_seeds must contain only integers from 0 through 4294967295"
        )
    if len(set(seeds)) != len(seeds):
        raise StudyManifestError("training_seeds must be distinct")

    variants = list(_sequence(manifest.get("variants"), field="variants"))
    if len(variants) != 4 or any(not isinstance(item, Mapping) for item in variants):
        raise StudyManifestError("variants must contain exactly four objects")
    variant_ids = tuple(item.get("variant_id") for item in variants)
    if variant_ids != REQUIRED_VARIANT_IDS:
        raise StudyManifestError(
            "variants must be exactly, in preregistered order: "
            + ", ".join(REQUIRED_VARIANT_IDS)
        )
    for variant in variants:
        variant_id = str(variant["variant_id"])
        _require_exact_sequence(
            variant.get("training_sources"),
            EXPECTED_TRAINING_SOURCES[variant_id],
            field=f"variants[{variant_id}].training_sources",
        )
        expected_seed_policy = (
            "not_applicable"
            if variant_id == "base"
            else "all_preregistered_training_seeds"
        )
        if variant.get("training_seed_policy") != expected_seed_policy:
            raise StudyManifestError(
                f"variants[{variant_id}].training_seed_policy must be "
                f"{expected_seed_policy!r}"
            )
        expected_initialization = (
            "pinned_base_model"
            if variant_id in {"base", "bc0"}
            else "bc0_same_seed_checkpoint"
        )
        if variant.get("initialization") != expected_initialization:
            raise StudyManifestError(
                f"variants[{variant_id}].initialization must be "
                f"{expected_initialization!r}"
            )
        expected_roles = (
            (
                "development_evaluation",
                "evaluation",
                "recovery_stress_evaluation",
            )
            if variant_id == "base"
            else (
                "checkpoint",
                "development_evaluation",
                "evaluation",
                "recovery_stress_evaluation",
            )
        )
        _require_exact_sequence(
            variant.get("required_artifact_roles"),
            expected_roles,
            field=f"variants[{variant_id}].required_artifact_roles",
        )

    replication = _mapping(
        manifest.get("replication_policy"), field="replication_policy"
    )
    _require_exact_sequence(
        replication.get("trained_variants"),
        TRAINED_VARIANT_IDS,
        field="replication_policy.trained_variants",
    )
    for required_true in (
        "require_every_training_seed_for_every_trained_variant",
        "pair_comparisons_by_training_seed",
        "base_variant_is_untrained_reference",
    ):
        if replication.get(required_true) is not True:
            raise StudyManifestError(
                f"replication_policy.{required_true} must be true"
            )

    bindings = _mapping(manifest.get("bindings"), field="bindings")
    if set(bindings) != {
        "source",
        "base_model",
        "evaluation",
        "development_evaluation",
        "recovery_stress_evaluation",
    }:
        raise StudyManifestError("study bindings fields are not exact")
    source = _mapping(bindings.get("source"), field="bindings.source")
    if source.get("contract") != "externally_reviewed_clean_git_commit_v1":
        raise StudyManifestError("bindings.source has an unapproved contract")
    if source.get("commit_format") != "lowercase_40_hex":
        raise StudyManifestError("bindings.source.commit_format must bind 40-hex")
    if source.get("dirty_source_allowed") is not False:
        raise StudyManifestError("the study must forbid dirty source")
    if source.get("must_match_across_all_checkpoints_and_evaluations") is not True:
        raise StudyManifestError("all study artifacts must bind one source commit")

    base_model = _mapping(bindings.get("base_model"), field="bindings.base_model")
    if base_model.get("model_id") != PINNED_BASE_MODEL_ID:
        raise StudyManifestError("bindings.base_model.model_id is not the pinned model")
    if base_model.get("model_revision") != PINNED_BASE_MODEL_REVISION:
        raise StudyManifestError(
            "bindings.base_model.model_revision is not the pinned revision"
        )
    if base_model.get("must_match_across_all_variants") is not True:
        raise StudyManifestError("every variant must use the same pinned base model")

    training_protocol = _mapping(
        manifest.get("training_protocol_policy"),
        field="training_protocol_policy",
    )
    if set(training_protocol) != {
        "contract",
        "dependency_lock",
        "rng",
        "variant_protocols",
    }:
        raise StudyManifestError("training_protocol_policy fields are not exact")
    if training_protocol.get("contract") != TRAINING_PROTOCOL_CONTRACT:
        raise StudyManifestError("training protocol contract is not approved")
    dependency_lock = _mapping(
        training_protocol.get("dependency_lock"),
        field="training_protocol_policy.dependency_lock",
    )
    if dict(dependency_lock) != {
        "path": TRAINING_DEPENDENCY_LOCK_PATH,
        "sha256": TRAINING_DEPENDENCY_LOCK_SHA256,
    }:
        raise StudyManifestError("training dependency lock is not exactly pinned")
    rng_policy = _mapping(
        training_protocol.get("rng"),
        field="training_protocol_policy.rng",
    )
    if dict(rng_policy) != {
        "contract": TRAINING_RNG_CONTRACT,
        "engines": list(TRAINING_RNG_ENGINES),
        "seed_before_model_construction": True,
        "reset_before_cold_adapter_attachment": True,
        "warm_adapter_reset_policy": "not_applicable_existing_adapter",
    }:
        raise StudyManifestError("training RNG policy is not exactly pinned")
    variant_protocols = _mapping(
        training_protocol.get("variant_protocols"),
        field="training_protocol_policy.variant_protocols",
    )
    if set(variant_protocols) != set(TRAINED_VARIANT_IDS):
        raise StudyManifestError(
            "training protocol must define exactly every trained variant"
        )
    for variant_id, expected_configuration in (
        EXPECTED_TRAINING_PROTOCOL_CONFIGURATIONS.items()
    ):
        if variant_protocols.get(variant_id) != expected_configuration:
            raise StudyManifestError(
                f"training protocol for {variant_id} differs from preregistration"
            )

    evaluation = _mapping(bindings.get("evaluation"), field="bindings.evaluation")
    if evaluation.get("suite_path") != PINNED_SUITE_PATH:
        raise StudyManifestError("bindings.evaluation.suite_path is not pinned")
    if evaluation.get("policy_path") != PINNED_POLICY_PATH:
        raise StudyManifestError("bindings.evaluation.policy_path is not pinned")
    for hash_field in ("suite_sha256", "policy_sha256"):
        if _SHA256_RE.fullmatch(str(evaluation.get(hash_field) or "")) is None:
            raise StudyManifestError(f"bindings.evaluation.{hash_field} must be 64-hex")
    if evaluation.get("suite_sha256") != PINNED_SUITE_SHA256:
        raise StudyManifestError("bindings.evaluation.suite_sha256 is not pinned")
    if evaluation.get("policy_sha256") != PINNED_POLICY_SHA256:
        raise StudyManifestError("bindings.evaluation.policy_sha256 is not pinned")
    if evaluation.get("policy_id") != "bc0_closed_loop_hard_gate_v3":
        raise StudyManifestError("bindings.evaluation.policy_id is not approved")
    if evaluation.get("evaluator_seed") != 20260719:
        raise StudyManifestError("bindings.evaluation.evaluator_seed is not pinned")
    if evaluation.get("max_steps") != 24:
        raise StudyManifestError("bindings.evaluation.max_steps must be 24")
    if evaluation.get("same_physical_roots_and_protocol_for_every_variant") is not True:
        raise StudyManifestError("every variant must use identical evaluation roots")

    development_evaluation = _mapping(
        bindings.get("development_evaluation"),
        field="bindings.development_evaluation",
    )
    canonical_development = canonical_development_evaluation_contract()
    if (
        set(development_evaluation) != set(canonical_development)
        or _content_sha256(development_evaluation)
        != EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
        or dict(development_evaluation) != canonical_development
    ):
        raise StudyManifestError(
            "bindings.development_evaluation differs from the exact "
            "preregistered evaluator contract"
        )
    recovery_stress_evaluation = _mapping(
        bindings.get("recovery_stress_evaluation"),
        field="bindings.recovery_stress_evaluation",
    )
    canonical_recovery_stress = (
        canonical_recovery_stress_evaluation_contract()
    )
    if (
        set(recovery_stress_evaluation) != set(canonical_recovery_stress)
        or _content_sha256(recovery_stress_evaluation)
        != EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256
        or dict(recovery_stress_evaluation) != canonical_recovery_stress
    ):
        raise StudyManifestError(
            "bindings.recovery_stress_evaluation differs from the exact "
            "preregistered evaluator contract"
        )

    if verify_bound_files:
        root = Path(repo_root) if repo_root is not None else _repo_root()
        suite_path = root / str(evaluation.get("suite_path") or "")
        policy_path = root / str(evaluation.get("policy_path") or "")
        dependency_lock_path = root / TRAINING_DEPENDENCY_LOCK_PATH
        for label, path, expected_hash in (
            ("suite", suite_path, evaluation["suite_sha256"]),
            ("policy", policy_path, evaluation["policy_sha256"]),
            (
                "training dependency lock",
                dependency_lock_path,
                TRAINING_DEPENDENCY_LOCK_SHA256,
            ),
        ):
            if not path.is_file():
                raise StudyManifestError(f"bound evaluation {label} is missing: {path}")
            actual_hash = _file_sha256(path)
            if actual_hash != expected_hash:
                raise StudyManifestError(
                    f"bound evaluation {label} hash mismatch: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("policy_id") != evaluation["policy_id"]:
            raise StudyManifestError("bound policy_id differs from the study manifest")
        suite_policy = _mapping(policy.get("suite_policy"), field="suite_policy")
        if suite_policy.get("approved_suite_sha256") != evaluation["suite_sha256"]:
            raise StudyManifestError("bound policy does not approve the study suite")
        if suite_policy.get("evaluator_seed") != evaluation["evaluator_seed"]:
            raise StudyManifestError("bound policy evaluator seed differs from the study")
        if suite_policy.get("max_steps") != evaluation["max_steps"]:
            raise StudyManifestError("bound policy step budget differs from the study")

    artifact_policy = _mapping(
        manifest.get("artifact_binding_policy"), field="artifact_binding_policy"
    )
    checkpoint_fields = set(
        _sequence(
            artifact_policy.get("checkpoint_required_fields"),
            field="artifact_binding_policy.checkpoint_required_fields",
        )
    )
    evaluation_fields = set(
        _sequence(
            artifact_policy.get("evaluation_required_fields"),
            field="artifact_binding_policy.evaluation_required_fields",
        )
    )
    development_evaluation_fields = set(
        _sequence(
            artifact_policy.get("development_evaluation_required_fields"),
            field=(
                "artifact_binding_policy."
                "development_evaluation_required_fields"
            ),
        )
    )
    recovery_stress_evaluation_fields = set(
        _sequence(
            artifact_policy.get(
                "recovery_stress_evaluation_required_fields"
            ),
            field=(
                "artifact_binding_policy."
                "recovery_stress_evaluation_required_fields"
            ),
        )
    )
    if not {
        "artifact_schema_version",
        "artifact_role",
        "variant_id",
        "study_manifest_sha256",
        "reviewed_source_commit",
        "training_seed",
        "training_view_provenance_id",
        "training_protocol",
        "training_configuration",
        "training_rng_attestation",
        "parent_checkpoint_receipt_id",
        "production_d1_quarantine_binding",
        "base_snapshot_attestation_sha256",
        "adapter_path",
        "adapter_tree_sha256",
        "runtime_accelerator_attestation",
        "checkpoint_receipt_id",
    }.issubset(checkpoint_fields):
        raise StudyManifestError("checkpoint artifact binding fields are incomplete")
    if not {
        "artifact_role",
        "variant_id",
        "study_manifest_sha256",
        "reviewed_source_commit",
        "model_id",
        "model_revision",
        "checkpoint_receipt_id",
        "checkpoint_adapter_tree_sha256",
        "training_seed",
        "frozen_suite_sha256",
        "evaluation_policy_sha256",
    }.issubset(evaluation_fields):
        raise StudyManifestError("evaluation artifact binding fields are incomplete")
    if not {
        "artifact_role",
        "variant_id",
        "study_manifest_sha256",
        "reviewed_source_commit",
        "model_id",
        "model_revision",
        "checkpoint_receipt_id",
        "checkpoint_adapter_tree_sha256",
        "training_seed",
        "development_holdout_sha256",
        "development_holdout_provenance_id",
        "development_holdout_root_set_sha256",
        "development_holdout_physical_roots",
        "development_evaluation_contract_sha256",
        "evaluation_protocol",
    }.issubset(development_evaluation_fields):
        raise StudyManifestError(
            "development evaluation artifact binding fields are incomplete"
        )
    if not {
        "artifact_role",
        "variant_id",
        "study_manifest_sha256",
        "reviewed_source_commit",
        "model_id",
        "model_revision",
        "checkpoint_receipt_id",
        "checkpoint_adapter_tree_sha256",
        "training_seed",
        "recovery_stress_suite_sha256",
        "recovery_stress_manifest_sha256",
        "recovery_stress_provenance_id",
        "recovery_stress_root_set_sha256",
        "recovery_stress_physical_roots",
        "recovery_stress_episode_count",
        "recovery_stress_development_parent_sha256",
        "recovery_stress_evaluation_contract_sha256",
        "evaluation_protocol",
    }.issubset(recovery_stress_evaluation_fields):
        raise StudyManifestError(
            "recovery-stress evaluation artifact binding fields are incomplete"
        )
    if artifact_policy.get("base_evaluation_training_seed_must_be_null") is not True:
        raise StudyManifestError("base evaluation must not claim a training seed")
    if artifact_policy.get("base_evaluation_checkpoint_binding_must_be_null") is not True:
        raise StudyManifestError("base evaluation must not claim a checkpoint binding")

    stability = _mapping(
        manifest.get("stability_scope_policy"), field="stability_scope_policy"
    )
    if _content_sha256(stability) != EXPECTED_STABILITY_SCOPE_POLICY_SHA256:
        raise StudyManifestError("stability_scope_policy differs from preregistration")
    _require_exact_sequence(
        stability.get("required_scopes"),
        ("development_holdout", "frozen_suite"),
        field="stability_scope_policy.required_scopes",
    )
    development_scope = _mapping(
        stability.get("development_holdout"),
        field="stability_scope_policy.development_holdout",
    )
    if development_scope != {
        "artifact_role": "development_evaluation",
        "evaluation_protocol": "diagnostic_model_selection_only",
        "exact_physical_roots": 30,
        "content_sha256_required": True,
        "provenance_id_required": True,
        "root_set_sha256_required": True,
        "same_bound_holdout_across_all_variants_and_seeds": True,
        "release_qualification_allowed": False,
    }:
        raise StudyManifestError("development holdout stability scope is not pinned")
    frozen_scope = _mapping(
        stability.get("frozen_suite"),
        field="stability_scope_policy.frozen_suite",
    )
    if frozen_scope != {
        "artifact_role": "evaluation",
        "suite_sha256": PINNED_SUITE_SHA256,
        "release_qualification_allowed": True,
    }:
        raise StudyManifestError("frozen-suite stability scope is not pinned")
    for required_true in (
        "require_both_scopes_for_stability_decision",
        "frozen_suite_cannot_substitute_for_development_holdout",
    ):
        if stability.get(required_true) is not True:
            raise StudyManifestError(f"stability_scope_policy.{required_true} must be true")
    if stability.get("missing_or_mismatched_scope_policy") != "fail_closed":
        raise StudyManifestError("missing or mismatched stability scope must fail closed")

    comparison = _mapping(
        manifest.get("comparison_policy"), field="comparison_policy"
    )
    if _content_sha256(comparison) != EXPECTED_COMPARISON_POLICY_SHA256:
        raise StudyManifestError("comparison_policy differs from preregistration")
    if comparison.get("primary_metric") != "multi_error_episode_recovery_rate":
        raise StudyManifestError("the primary metric is not preregistered")
    _require_exact_sequence(
        comparison.get("primary_contrast"),
        ("natural_dagger_probes", "bc0"),
        field="comparison_policy.primary_contrast",
    )
    _require_exact_sequence(
        comparison.get("required_ablation_contrast"),
        ("natural_dagger_probes", "natural_dagger"),
        field="comparison_policy.required_ablation_contrast",
    )
    probe_ablation = _mapping(
        comparison.get("probe_ablation_policy"),
        field="comparison_policy.probe_ablation_policy",
    )
    if probe_ablation != {
        "pairing": "same_training_seed",
        "targeted_metrics": [
            "post_failure_no_candidate_action_accuracy",
            "unsupported_correction_recovery_action_accuracy",
        ],
        "targeted_improvement_operator": ">",
        "minimum_targeted_absolute_improvement_each": 0.0,
        "unrelated_metric_scope": (
            "explicit_preregistered_normalized_outcome_rate_registry"
        ),
        "unrelated_metric_scale": "unit_interval",
        "unrelated_metric_direction": "higher_is_better",
        "unrelated_metric_registry": [
            "multi_error_episode_recovery_rate",
            "diagnostic.multi_error_exact_fault_set_identification_rate",
            "diagnostic.measurement_parameter_family_macro_f1",
            "diagnostic.multi_error_correct_error_cardinality_rate",
            "diagnostic.mixed_measurement_parameter_sequential_resolution_rate",
            "physical.physically_valid_recovery_among_resolved_rate",
            "physical.healthy_measurement_preservation_rate",
            "physical.healthy_branch_parameter_preservation_rate",
            "physical.final_residual_chi_square_acceptance_rate",
            "physical.post_commit_powerflow_topology_feasibility_rate",
            "recovery_action_accuracy.premature_commit_recovery",
            "recovery_action_accuracy.premature_escalation_recovery",
            "recovery_action_accuracy.rejected_candidate_rollback",
            "recovery_action_accuracy.safe_continuation_after_partial_success",
            "recovery_action_accuracy.measurement_parameter_sequential_handoff",
        ],
        "excluded_numeric_scopes": [
            "evidence_counts_numerators_denominators_and_support",
            "safety_counts_governed_by_zero_tolerance_rules",
            "action_quality_efficiency_metrics_governed_by_objective_thresholds",
            "hashes_identifiers_and_threshold_constants",
        ],
        "maximum_unrelated_absolute_degradation": 0.02,
        "unsupported_or_empty_metric_policy": "fail_closed",
    }:
        raise StudyManifestError("probe ablation policy differs from preregistration")
    if comparison.get("minimum_complete_variant_recovery_rate") != 0.9:
        raise StudyManifestError("complete-variant recovery target must be 0.9")
    if comparison.get("minimum_mean_primary_improvement") != 0.1:
        raise StudyManifestError("primary improvement target must be 0.1")
    if comparison.get("minimum_training_seed_count") != 3:
        raise StudyManifestError("minimum_training_seed_count must be 3")
    if len(seeds) < comparison["minimum_training_seed_count"]:
        raise StudyManifestError("training seed count is below the comparison floor")
    _require_exact_sequence(
        comparison.get("required_stability_scopes"),
        ("development_holdout", "frozen_suite"),
        field="comparison_policy.required_stability_scopes",
    )
    if comparison.get("missing_stability_scope_policy") != "fail_closed":
        raise StudyManifestError("missing stability scope evidence must fail closed")
    if comparison.get("require_every_seed_non_degrading_vs_bc0") is not True:
        raise StudyManifestError("every seed must be non-degrading versus BC0")
    if comparison.get("require_bootstrap_95ci_excludes_zero") is not True:
        raise StudyManifestError("primary confidence interval must exclude zero")
    if comparison.get("maximum_complete_variant_seed_spread") != 0.05:
        raise StudyManifestError("complete-variant seed spread must be at most 0.05")
    if comparison.get("maximum_single_error_degradation") != 0.02:
        raise StudyManifestError("single-error degradation must be at most 0.02")
    if comparison.get("family_metric") != "multi_error_episode_recovery_rate":
        raise StudyManifestError("family metric is not preregistered")
    if comparison.get("family_scope") != "all_preregistered_error_families":
        raise StudyManifestError("family scope is not preregistered")
    if comparison.get("family_reference_variant") != "bc0":
        raise StudyManifestError("family reference variant must be BC0")
    if comparison.get("family_pairing") != "same_training_seed":
        raise StudyManifestError("family comparisons must pair the same training seed")
    if comparison.get("unsupported_or_empty_family_policy") != "fail_closed":
        raise StudyManifestError("unsupported or empty families must fail closed")
    if comparison.get("require_no_material_family_regression") is not True:
        raise StudyManifestError("material family regressions must be rejected")
    if comparison.get("maximum_material_family_regression") != 0.02:
        raise StudyManifestError("material family regression tolerance must be 0.02")
    for count_field in (
        "maximum_false_commit_count",
        "maximum_false_finalization_count",
        "maximum_false_rollback_count",
        "maximum_teacher_targets_quarantined_in_production_d1",
        "maximum_finalize_with_unresolved_private_fault_count",
        "maximum_physically_unsafe_commit_count",
        "maximum_truth_safe_accepted_candidate_rollback_count",
        "maximum_hidden_truth_leakage_count",
        "maximum_healthy_component_corruption_episodes",
        "maximum_healthy_target_corruption_episodes",
        "maximum_unknown_healthy_preservation_episodes",
        "maximum_evaluator_error_episodes",
    ):
        if comparison.get(count_field) != 0:
            raise StudyManifestError(f"comparison_policy.{count_field} must be zero")

    objectives = _mapping(
        manifest.get("objective_thresholds"), field="objective_thresholds"
    )
    if _content_sha256(objectives) != EXPECTED_OBJECTIVE_THRESHOLDS_SHA256:
        raise StudyManifestError("objective_thresholds differ from preregistration")
    if objectives.get("evidence_policy") != "required_fail_closed_if_unavailable":
        raise StudyManifestError("unavailable objective evidence must fail closed")

    return {
        "passed": True,
        "contract": STUDY_MANIFEST_CONTRACT,
        "study_id": STUDY_ID,
        "training_seeds": seeds,
        "variant_ids": list(variant_ids),
        "suite_sha256": evaluation["suite_sha256"],
        "policy_sha256": evaluation["policy_sha256"],
        "development_evaluation_contract_sha256": (
            EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
        ),
        "recovery_stress_evaluation_contract_sha256": (
            EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256
        ),
    }


def load_study_manifest(
    path: str | Path = DEFAULT_STUDY_MANIFEST,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load only the byte-pinned preregistration and validate all bindings."""

    manifest_path = Path(path)
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise StudyManifestError(f"study manifest cannot be read: {exc}") from exc
    actual_hash = hashlib.sha256(raw_manifest).hexdigest()
    if actual_hash != EXPECTED_STUDY_MANIFEST_SHA256:
        raise StudyManifestError(
            "study manifest digest mismatch: "
            f"expected {EXPECTED_STUDY_MANIFEST_SHA256}, got {actual_hash}"
        )
    try:
        payload = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StudyManifestError(f"study manifest is not valid UTF-8 JSON: {exc}") from exc
    manifest = _mapping(payload, field="study manifest")
    result = validate_study_manifest(
        manifest,
        repo_root=repo_root,
        verify_bound_files=True,
    )
    return {**dict(manifest), "manifest_sha256": actual_hash, "validation": result}


def validate_study_artifact_binding(
    manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    variant_id: str,
    artifact_role: str,
    expected_source_commit: str,
    expected_training_seed: int | None = None,
) -> dict[str, Any]:
    """Validate one future checkpoint/evaluation against the preregistration.

    ``expected_source_commit`` is intentionally external: the reviewed commit
    is frozen at execution time, then passed to every artifact validator so a
    comparison cannot mix implementations.
    """

    validate_study_manifest(manifest, verify_bound_files=False)
    if (
        _content_sha256(_source_manifest_payload(manifest))
        != EXPECTED_STUDY_MANIFEST_CONTENT_SHA256
    ):
        raise StudyManifestError(
            "artifact validation requires the exact immutable study manifest"
        )
    if variant_id not in REQUIRED_VARIANT_IDS:
        raise StudyManifestError(f"unknown study variant: {variant_id!r}")
    if artifact_role not in {
        "checkpoint",
        "development_evaluation",
        "evaluation",
        "recovery_stress_evaluation",
    }:
        raise StudyManifestError(
            "artifact_role must be checkpoint, development_evaluation, "
            "evaluation, or recovery_stress_evaluation"
        )
    variants = {
        str(item["variant_id"]): item
        for item in _sequence(manifest["variants"], field="variants")
    }
    allowed_roles = tuple(
        _sequence(
            variants[variant_id].get("required_artifact_roles"),
            field=f"variants[{variant_id}].required_artifact_roles",
        )
    )
    if artifact_role not in allowed_roles:
        raise StudyManifestError(
            f"variant {variant_id!r} does not permit {artifact_role!r} artifacts"
        )
    if _COMMIT_RE.fullmatch(expected_source_commit) is None:
        raise StudyManifestError("expected_source_commit must be lowercase 40-hex")

    binding_policy = _mapping(
        manifest["artifact_binding_policy"], field="artifact_binding_policy"
    )
    required_fields = _sequence(
        binding_policy[f"{artifact_role}_required_fields"],
        field=f"artifact_binding_policy.{artifact_role}_required_fields",
    )
    missing = [field for field in required_fields if field not in artifact]
    if missing:
        raise StudyManifestError(
            f"{artifact_role} artifact is missing required fields: {', '.join(missing)}"
        )
    if artifact.get("variant_id") != variant_id:
        raise StudyManifestError("artifact variant_id differs from the requested variant")
    if artifact.get("artifact_role") != artifact_role:
        raise StudyManifestError("artifact role differs from the requested role")
    if artifact.get("study_manifest_sha256") != EXPECTED_STUDY_MANIFEST_SHA256:
        raise StudyManifestError("artifact does not bind the immutable study manifest")
    if artifact.get("reviewed_source_commit") != expected_source_commit:
        raise StudyManifestError("artifact source commit differs from the reviewed commit")

    seeds = set(_sequence(manifest["training_seeds"], field="training_seeds"))
    observed_seed = artifact.get("training_seed")
    if variant_id == "base":
        if observed_seed is not None:
            raise StudyManifestError("base evaluation training_seed must be null")
        if expected_training_seed is not None:
            raise StudyManifestError("base variant cannot have an expected training seed")
    else:
        if (
            isinstance(observed_seed, bool)
            or not isinstance(observed_seed, int)
            or observed_seed not in seeds
        ):
            raise StudyManifestError(
                "trained artifact seed is not one of the preregistered training seeds"
            )
        if expected_training_seed is not None and observed_seed != expected_training_seed:
            raise StudyManifestError("artifact training seed differs from the paired seed")

    bindings = _mapping(manifest["bindings"], field="bindings")
    base_model = _mapping(bindings["base_model"], field="bindings.base_model")
    evaluation = _mapping(bindings["evaluation"], field="bindings.evaluation")
    if artifact_role == "checkpoint":
        receipt_payload = dict(artifact)
        receipt_id = receipt_payload.pop("checkpoint_receipt_id", None)
        if receipt_id != _content_sha256(receipt_payload):
            raise StudyManifestError("checkpoint_receipt_id does not match its payload")
        if artifact.get("artifact_schema_version") != 1:
            raise StudyManifestError(
                "checkpoint artifact_schema_version must be exactly 1"
            )
        if artifact.get("base_model_id") != base_model["model_id"]:
            raise StudyManifestError("checkpoint base_model_id differs from the study")
        if artifact.get("base_model_revision") != base_model["model_revision"]:
            raise StudyManifestError("checkpoint base_model_revision differs from the study")
        expected_training_protocol = build_training_protocol_binding(
            manifest,
            variant_id=variant_id,
        )
        if artifact.get("training_protocol") != expected_training_protocol:
            raise StudyManifestError(
                f"{variant_id} checkpoint training protocol differs from the study"
            )
        if artifact.get("training_configuration") != expected_training_protocol[
            "configuration"
        ]:
            raise StudyManifestError(
                f"{variant_id} checkpoint training configuration is not exact"
            )
        expected_rng = canonical_training_rng_attestation(
            variant_id=variant_id,
            training_seed=int(observed_seed),
        )
        if artifact.get("training_rng_attestation") != expected_rng:
            raise StudyManifestError(
                f"{variant_id} checkpoint training RNG attestation is not exact"
            )
        parent_receipt_id = artifact.get("parent_checkpoint_receipt_id")
        if variant_id == "bc0":
            if parent_receipt_id is not None:
                raise StudyManifestError(
                    "BC0 parent_checkpoint_receipt_id must be null"
                )
        elif _SHA256_RE.fullmatch(str(parent_receipt_id or "")) is None:
            raise StudyManifestError(
                f"{variant_id} parent_checkpoint_receipt_id must be lowercase 64-hex"
            )
        if _SHA256_RE.fullmatch(
            str(artifact.get("base_snapshot_attestation_sha256") or "")
        ) is None:
            raise StudyManifestError(
                "checkpoint base_snapshot_attestation_sha256 must be lowercase 64-hex"
            )
        if _SHA256_RE.fullmatch(
            str(artifact.get("training_view_provenance_id") or "")
        ) is None:
            raise StudyManifestError(
                "checkpoint training_view_provenance_id must be lowercase 64-hex"
            )
        validate_production_d1_quarantine_binding(
            artifact.get("production_d1_quarantine_binding"),
            variant_id=variant_id,
            expected_generation_provenance_id=str(
                artifact["training_view_provenance_id"]
            ),
        )
        _require_exact_sequence(
            artifact.get("training_sources"),
            EXPECTED_TRAINING_SOURCES[variant_id],
            field="checkpoint.training_sources",
        )
        expected_round1_view = {
            "bc0": None,
            "natural_dagger": "natural-only",
            "natural_dagger_probes": "full",
        }[variant_id]
        if artifact.get("round1_view") != expected_round1_view:
            raise StudyManifestError(
                f"{variant_id} checkpoint round1_view must be "
                f"{expected_round1_view!r}"
            )
        dataset_hashes = _mapping(
            artifact.get("training_dataset_sha256"),
            field="checkpoint.training_dataset_sha256",
        )
        if set(dataset_hashes) != {"train", "validation"}:
            raise StudyManifestError(
                "checkpoint training dataset hashes must bind exactly train "
                "and validation"
            )
        for split, digest in dataset_hashes.items():
            if _SHA256_RE.fullmatch(str(digest or "")) is None:
                raise StudyManifestError(
                    f"checkpoint {split} dataset SHA-256 must be lowercase 64-hex"
                )
        adapter_path = str(artifact.get("adapter_path") or "")
        if not _is_absolute_portable_path(adapter_path):
            raise StudyManifestError("checkpoint adapter_path must be absolute")
        if _SHA256_RE.fullmatch(
            str(artifact.get("adapter_tree_sha256") or "")
        ) is None:
            raise StudyManifestError("checkpoint adapter_tree_sha256 must be 64-hex")
        accelerator = _mapping(
            artifact.get("runtime_accelerator_attestation"),
            field="checkpoint.runtime_accelerator_attestation",
        )
        devices = list(
            _sequence(
                accelerator.get("devices"),
                field="checkpoint.runtime_accelerator_attestation.devices",
            )
        )
        if accelerator.get("device_count") != 1 or len(devices) != 1:
            raise StudyManifestError("checkpoint must attest exactly one accelerator")
        if accelerator.get("bf16_supported") is not True:
            raise StudyManifestError("checkpoint accelerator must support bf16")
        cuda_version = accelerator.get("torch_cuda_version")
        if not isinstance(cuda_version, str) or not cuda_version.strip():
            raise StudyManifestError(
                "checkpoint accelerator CUDA runtime version is missing"
            )
        if accelerator.get("required_accelerator_class") is not None:
            raise StudyManifestError(
                "portable study training must not require one accelerator class"
            )
        if accelerator.get("required_accelerator_class_matched") is not True:
            raise StudyManifestError("checkpoint accelerator attestation did not match")
        device = _mapping(devices[0], field="checkpoint accelerator device")
        claimed_accelerator_class = device.get("accelerator_class")
        if claimed_accelerator_class not in {"h100", "h200", "rtx6000"}:
            raise StudyManifestError("checkpoint accelerator class is not approved")
        if not isinstance(device.get("name"), str) or not device["name"].strip():
            raise StudyManifestError("checkpoint accelerator name is missing")
        if not isinstance(device.get("total_memory_bytes"), int) or device[
            "total_memory_bytes"
        ] <= 0:
            raise StudyManifestError("checkpoint accelerator memory is invalid")
        capability = device.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(component, bool)
                or not isinstance(component, int)
                or component < 0
                for component in capability
            )
        ):
            raise StudyManifestError(
                "checkpoint accelerator compute capability is invalid"
            )
        from psse_env.sft.release_hardware import (
            RTX_PRO_6000_MIN_MEMORY_MIB,
            normalize_accelerator_class,
        )

        observed_accelerator_class = normalize_accelerator_class(
            device["name"], device["total_memory_bytes"]
        )
        if observed_accelerator_class != claimed_accelerator_class:
            if claimed_accelerator_class == "rtx6000":
                normalized_name = " ".join(str(device["name"]).upper().split())
                if "RTX PRO 6000" in normalized_name:
                    raise StudyManifestError(
                        "RTX Pro 6000 checkpoint memory is below "
                        f"{RTX_PRO_6000_MIN_MEMORY_MIB} MiB"
                    )
                raise StudyManifestError(
                    "rtx6000 checkpoint is not an NVIDIA RTX Pro 6000 with "
                    "approved memory"
                )
            raise StudyManifestError(
                "checkpoint accelerator class does not match its name and "
                "approved memory floor"
            )
        parent_revision = str(artifact.get("parent_model_revision") or "")
        expected_parent_pattern = _COMMIT_RE if variant_id == "bc0" else _SHA256_RE
        if expected_parent_pattern.fullmatch(parent_revision) is None:
            expected_width = 40 if variant_id == "bc0" else 64
            raise StudyManifestError(
                f"{variant_id} parent_model_revision must be lowercase {expected_width}-hex"
            )
        if variant_id == "bc0" and parent_revision != base_model["model_revision"]:
            raise StudyManifestError("BC0 checkpoint must initialize from the pinned base")
    else:
        model_id = artifact.get("model_id")
        model_revision = str(artifact.get("model_revision") or "")
        if not isinstance(model_id, str) or not model_id.strip():
            raise StudyManifestError("evaluation model_id must be non-empty")
        if variant_id == "base":
            if model_id != base_model["model_id"] or model_revision != base_model["model_revision"]:
                raise StudyManifestError("base evaluation does not use the pinned base model")
            if artifact.get("checkpoint_receipt_id") is not None or artifact.get(
                "checkpoint_adapter_tree_sha256"
            ) is not None:
                raise StudyManifestError(
                    "base evaluation checkpoint binding must be null"
                )
        elif _SHA256_RE.fullmatch(model_revision) is None:
            raise StudyManifestError(
                "adapted evaluation model_revision must be a lowercase 64-hex tree hash"
            )
        else:
            if _SHA256_RE.fullmatch(
                str(artifact.get("checkpoint_receipt_id") or "")
            ) is None:
                raise StudyManifestError(
                    "adapted evaluation checkpoint_receipt_id must be 64-hex"
                )
            checkpoint_tree = str(
                artifact.get("checkpoint_adapter_tree_sha256") or ""
            )
            if _SHA256_RE.fullmatch(checkpoint_tree) is None:
                raise StudyManifestError(
                    "adapted evaluation checkpoint tree binding must be 64-hex"
                )
            if model_revision != checkpoint_tree:
                raise StudyManifestError(
                    "adapted evaluation model revision differs from checkpoint tree"
                )
        if artifact_role == "evaluation":
            if artifact.get("frozen_suite_sha256") != evaluation["suite_sha256"]:
                raise StudyManifestError(
                    "evaluation artifact uses a different frozen suite"
                )
            if artifact.get("evaluation_policy_sha256") != evaluation["policy_sha256"]:
                raise StudyManifestError("evaluation artifact uses a different policy")
        elif artifact_role == "development_evaluation":
            for hash_field in (
                "development_holdout_sha256",
                "development_holdout_provenance_id",
                "development_holdout_root_set_sha256",
            ):
                if _SHA256_RE.fullmatch(str(artifact.get(hash_field) or "")) is None:
                    raise StudyManifestError(
                        f"development evaluation {hash_field} must be lowercase 64-hex"
                    )
            development_contract = canonical_development_evaluation_contract()
            if (
                artifact.get("development_evaluation_contract_sha256")
                != EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
            ):
                raise StudyManifestError(
                    "development evaluation does not bind the exact "
                    "preregistered evaluator contract"
                )
            if artifact.get("development_holdout_physical_roots") != (
                development_contract["exact_physical_roots"]
            ):
                raise StudyManifestError(
                    "development evaluation must bind exactly 30 physical roots"
                )
            if artifact.get("evaluation_protocol") != development_contract[
                "evaluation_protocol"
            ]:
                raise StudyManifestError(
                    "development evaluation protocol is not model-selection-only"
                )
        else:
            for hash_field in (
                "recovery_stress_suite_sha256",
                "recovery_stress_manifest_sha256",
                "recovery_stress_provenance_id",
                "recovery_stress_root_set_sha256",
                "recovery_stress_development_parent_sha256",
            ):
                if (
                    _SHA256_RE.fullmatch(
                        str(artifact.get(hash_field) or "")
                    )
                    is None
                ):
                    raise StudyManifestError(
                        f"recovery-stress evaluation {hash_field} must be "
                        "lowercase 64-hex"
                    )
            recovery_contract = (
                canonical_recovery_stress_evaluation_contract()
            )
            if (
                artifact.get(
                    "recovery_stress_evaluation_contract_sha256"
                )
                != EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256
            ):
                raise StudyManifestError(
                    "recovery-stress evaluation does not bind the exact "
                    "preregistered evaluator contract"
                )
            if artifact.get("recovery_stress_physical_roots") != (
                recovery_contract["exact_physical_roots"]
            ):
                raise StudyManifestError(
                    "recovery-stress evaluation must bind exactly 20 physical roots"
                )
            if artifact.get("recovery_stress_episode_count") != (
                recovery_contract["exact_episode_count"]
            ):
                raise StudyManifestError(
                    "recovery-stress evaluation must bind exactly 70 episodes"
                )
            if artifact.get("evaluation_protocol") != recovery_contract[
                "evaluation_protocol"
            ]:
                raise StudyManifestError(
                    "recovery-stress evaluation protocol is not preregistered"
                )

    return {
        "passed": True,
        "variant_id": variant_id,
        "artifact_role": artifact_role,
        "reviewed_source_commit": expected_source_commit,
        "training_seed": observed_seed,
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
    }


__all__ = [
    "DEFAULT_STUDY_MANIFEST",
    "DEVELOPMENT_EVALUATION_PROTOCOL_CONTRACT",
    "EXPECTED_COMPARISON_POLICY_SHA256",
    "EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256",
    "EXPECTED_OBJECTIVE_THRESHOLDS_SHA256",
    "EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256",
    "EXPECTED_STABILITY_SCOPE_POLICY_SHA256",
    "EXPECTED_STUDY_MANIFEST_SHA256",
    "EXPECTED_STUDY_MANIFEST_CONTENT_SHA256",
    "EXPECTED_TRAINING_PROTOCOL_CONFIGURATIONS",
    "EXPECTED_TRAINING_SOURCES",
    "PRODUCTION_D1_QUARANTINE_APPLICABLE_VARIANTS",
    "PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME",
    "PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT",
    "RECOVERY_STRESS_EVALUATION_PROTOCOL_CONTRACT",
    "REQUIRED_VARIANT_IDS",
    "STUDY_ID",
    "STUDY_MANIFEST_CONTRACT",
    "TRAINED_VARIANT_IDS",
    "TRAINING_DEPENDENCY_LOCK_PATH",
    "TRAINING_DEPENDENCY_LOCK_SHA256",
    "TRAINING_PROTOCOL_CONTRACT",
    "TRAINING_RNG_CONTRACT",
    "TRAINING_RNG_ENGINES",
    "StudyManifestError",
    "build_production_d1_quarantine_binding",
    "build_training_protocol_binding",
    "canonical_development_evaluation_contract",
    "canonical_recovery_stress_evaluation_contract",
    "canonical_production_d1_quarantine_binding",
    "canonical_training_rng_attestation",
    "load_study_manifest",
    "study_manifest_sha256",
    "validate_study_artifact_binding",
    "validate_study_manifest",
    "validate_production_d1_quarantine_binding",
]
