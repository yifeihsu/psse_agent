"""Fail-closed tests for the immutable four-variant study contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psse_env.dagger.study_manifest import (
    DEFAULT_STUDY_MANIFEST,
    EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256,
    EXPECTED_STUDY_MANIFEST_SHA256,
    REQUIRED_VARIANT_IDS,
    StudyManifestError,
    build_production_d1_quarantine_binding,
    build_training_protocol_binding,
    canonical_development_evaluation_contract,
    canonical_production_d1_quarantine_binding,
    canonical_training_rng_attestation,
    load_study_manifest,
    study_manifest_sha256,
    validate_study_artifact_binding,
    validate_study_manifest,
)
from psse_env.sft.provenance import stable_json_sha256


SOURCE_COMMIT = "b" * 40
TREE_REVISION = "c" * 64
ACCELERATOR = {
    "device_count": 1,
    "bf16_supported": True,
    "torch_cuda_version": "12.8",
    "required_accelerator_class": None,
    "required_accelerator_class_matched": True,
    "devices": [
        {
            "index": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "total_memory_bytes": 96 * 1024**3,
            "compute_capability": [10, 0],
            "accelerator_class": "rtx6000",
        }
    ],
}
QUARANTINE_SUMMARY = {
    "contract": "dagger1_offline_teacher_target_quarantine_summary_v1",
    "candidate_definition": {},
    "total_rows": 25,
    "candidate_rows": 25,
    "non_candidate_rows": 0,
    "passed_rows": 25,
    "quarantined_rows": 0,
    "invalid_or_missing_audit_rows": 0,
    "quarantined_by_action_class": {},
    "quarantined_by_reason_code": {},
    "quarantined_example_ids": [],
    "zero_truth_audit_quarantine": True,
    "passed": True,
}
QUARANTINE_DESCRIPTOR = {
    "builder_contract": "deterministic_d0_d1_probe_balanced_union_v2",
    "audit_report_sha256": {
        "d1_offline_teacher_target_quarantine_summary": stable_json_sha256(
            QUARANTINE_SUMMARY
        )
    },
}
QUARANTINE_PROVENANCE = stable_json_sha256(QUARANTINE_DESCRIPTOR)


def _quarantine_binding(
    variant_id: str,
    provenance_id: str = QUARANTINE_PROVENANCE,
    descriptor: dict = QUARANTINE_DESCRIPTOR,
) -> dict:
    if variant_id in {"base", "bc0"}:
        return canonical_production_d1_quarantine_binding(variant_id)
    return build_production_d1_quarantine_binding(
        variant_id=variant_id,
        generation_provenance_id=provenance_id,
        generation_descriptor=descriptor,
        summary=QUARANTINE_SUMMARY,
        audit_report_sha256=stable_json_sha256(QUARANTINE_SUMMARY),
    )


def _manifest() -> dict:
    return json.loads(DEFAULT_STUDY_MANIFEST.read_text(encoding="utf-8"))


def _training_integrity_fields(
    manifest: dict,
    *,
    variant_id: str,
    seed: int,
) -> dict:
    protocol = build_training_protocol_binding(manifest, variant_id=variant_id)
    return {
        "training_protocol": protocol,
        "training_configuration": protocol["configuration"],
        "training_rng_attestation": canonical_training_rng_attestation(
            variant_id=variant_id,
            training_seed=seed,
        ),
        "parent_checkpoint_receipt_id": (
            None if variant_id == "bc0" else "e" * 64
        ),
    }


def _checkpoint_artifact(
    manifest: dict,
    *,
    variant_id: str,
    seed: int,
) -> dict:
    checkpoint = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": variant_id,
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "base_model_id": manifest["bindings"]["base_model"]["model_id"],
        "base_model_revision": manifest["bindings"]["base_model"][
            "model_revision"
        ],
        "base_snapshot_attestation_sha256": "9" * 64,
        "training_seed": seed,
        "training_view_provenance_id": "d" * 64,
        "training_sources": ["d0_bc0"],
        "round1_view": None,
        "production_d1_quarantine_binding": _quarantine_binding("bc0"),
        "training_dataset_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "parent_model_revision": manifest["bindings"]["base_model"][
            "model_revision"
        ],
        "adapter_path": f"/scratch/checkpoints/{variant_id}-seed{seed}/lora",
        "adapter_tree_sha256": "f" * 64,
        "runtime_accelerator_attestation": json.loads(json.dumps(ACCELERATOR)),
    }
    checkpoint.update(
        _training_integrity_fields(
            manifest,
            variant_id=variant_id,
            seed=seed,
        )
    )
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(checkpoint)
    return checkpoint


def test_default_manifest_is_content_addressed_and_fully_bound() -> None:
    loaded = load_study_manifest()

    assert study_manifest_sha256() == EXPECTED_STUDY_MANIFEST_SHA256
    assert loaded["manifest_sha256"] == EXPECTED_STUDY_MANIFEST_SHA256
    assert loaded["validation"]["passed"] is True
    assert tuple(loaded["validation"]["variant_ids"]) == REQUIRED_VARIANT_IDS
    assert loaded["validation"]["training_seeds"] == [3407, 3408, 3409]


def test_comparison_and_every_objective_threshold_are_explicit() -> None:
    manifest = _manifest()
    comparison = manifest["comparison_policy"]
    assert comparison["minimum_training_seed_count"] == 3
    assert comparison["maximum_complete_variant_seed_spread"] == 0.05
    assert comparison["maximum_single_error_degradation"] == 0.02
    assert comparison["maximum_material_family_regression"] == 0.02
    assert comparison["unsupported_or_empty_family_policy"] == "fail_closed"
    assert comparison["probe_ablation_policy"] == {
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
    }
    for field in (
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
        assert comparison[field] == 0

    objectives = manifest["objective_thresholds"]
    assert objectives["evidence_policy"] == "required_fail_closed_if_unavailable"
    assert set(objectives) == {
        "evidence_policy",
        "diagnostic",
        "physical",
        "recovery_action_accuracy",
        "action_quality_efficiency",
    }
    assert objectives["diagnostic"][
        "multi_error_exact_fault_set_identification_rate"
    ] == {"operator": ">=", "value": 0.9}
    assert objectives["physical"]["healthy_measurement_preservation_rate"] == {
        "operator": "==",
        "value": 1.0,
    }
    assert objectives["recovery_action_accuracy"][
        "unsupported_correction_recovery"
    ] == 0.9
    assert objectives["action_quality_efficiency"][
        "horizon_without_disposition_rate"
    ] == {"operator": "<", "value": 0.02}
    stability = manifest["stability_scope_policy"]
    assert stability["required_scopes"] == ["development_holdout", "frozen_suite"]
    assert stability["development_holdout"]["exact_physical_roots"] == 30
    assert stability["development_holdout"]["release_qualification_allowed"] is False
    assert stability["frozen_suite_cannot_substitute_for_development_holdout"] is True
    assert stability["missing_or_mismatched_scope_policy"] == "fail_closed"


def test_byte_change_fails_before_semantic_validation(tmp_path: Path) -> None:
    tampered = tmp_path / "study.json"
    tampered.write_text(
        DEFAULT_STUDY_MANIFEST.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(StudyManifestError, match="digest mismatch"):
        load_study_manifest(tampered)


@pytest.mark.parametrize(
    "seeds",
    (
        [3407, 3408],
        [3407, 3407, 3409],
        [3407, 3408, -1],
        [3407, 3408, True],
        [3407, 3408, 2**32],
    ),
)
def test_training_seed_set_fails_closed(seeds: list[object]) -> None:
    manifest = _manifest()
    manifest["training_seeds"] = seeds

    with pytest.raises(StudyManifestError, match="training_seeds"):
        validate_study_manifest(manifest, verify_bound_files=False)


def test_variant_set_and_source_semantics_are_exact() -> None:
    missing = _manifest()
    missing["variants"].pop()
    with pytest.raises(StudyManifestError, match="exactly four"):
        validate_study_manifest(missing, verify_bound_files=False)

    renamed = _manifest()
    renamed["variants"][3]["variant_id"] = "probe_only"
    with pytest.raises(StudyManifestError, match="preregistered order"):
        validate_study_manifest(renamed, verify_bound_files=False)

    contaminated = _manifest()
    contaminated["variants"][2]["training_sources"].append(
        "observable_recovery_probe"
    )
    with pytest.raises(StudyManifestError, match="training_sources"):
        validate_study_manifest(contaminated, verify_bound_files=False)

    model_drift = _manifest()
    model_drift["bindings"]["base_model"]["model_revision"] = "e" * 40
    with pytest.raises(StudyManifestError, match="pinned revision"):
        validate_study_manifest(model_drift, verify_bound_files=False)


def test_bound_suite_or_policy_change_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest()
    repo_root = Path(__file__).resolve().parents[2]
    suite_relative = manifest["bindings"]["evaluation"]["suite_path"]
    policy_relative = manifest["bindings"]["evaluation"]["policy_path"]
    suite = tmp_path / suite_relative
    policy = tmp_path / policy_relative
    suite.parent.mkdir(parents=True)
    policy.parent.mkdir(parents=True, exist_ok=True)
    suite.write_bytes((repo_root / suite_relative).read_bytes() + b"\n")
    policy.write_bytes((repo_root / policy_relative).read_bytes())

    with pytest.raises(StudyManifestError, match="suite hash mismatch"):
        validate_study_manifest(manifest, repo_root=tmp_path)


def test_training_protocol_and_dependency_lock_are_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest()
    drifted = json.loads(json.dumps(manifest))
    drifted["training_protocol_policy"]["variant_protocols"]["bc0"][
        "trainer"
    ]["epochs"] = 1.0
    with pytest.raises(StudyManifestError, match="bc0 differs"):
        validate_study_manifest(drifted, verify_bound_files=False)

    repo_root = Path(__file__).resolve().parents[2]
    for relative in (
        manifest["bindings"]["evaluation"]["suite_path"],
        manifest["bindings"]["evaluation"]["policy_path"],
        manifest["training_protocol_policy"]["dependency_lock"]["path"],
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo_root / relative).read_bytes())
    lock_path = (
        tmp_path
        / manifest["training_protocol_policy"]["dependency_lock"]["path"]
    )
    lock_path.write_bytes(lock_path.read_bytes() + b"\n")

    with pytest.raises(StudyManifestError, match="training dependency lock hash mismatch"):
        validate_study_manifest(manifest, repo_root=tmp_path)


def test_development_evaluator_contract_is_exact_and_digest_pinned() -> None:
    manifest = _manifest()
    contract = manifest["bindings"]["development_evaluation"]
    assert contract == canonical_development_evaluation_contract()
    assert stable_json_sha256(contract) == (
        EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
    )

    mutations = (
        ("evaluator_seed", 20260722),
        ("max_steps", 25),
        ("required_suites", ["dagger1_development", "forged"]),
        ("minimum_roots_per_suite", 29),
        ("protocol", "controller"),
    )
    for field, value in mutations:
        drifted = json.loads(json.dumps(manifest))
        drifted["bindings"]["development_evaluation"][field] = value
        with pytest.raises(StudyManifestError, match="preregistered evaluator"):
            validate_study_manifest(drifted, verify_bound_files=False)

    extra = json.loads(json.dumps(manifest))
    extra["bindings"]["development_evaluation"]["common_hash_allowed"] = True
    with pytest.raises(StudyManifestError, match="preregistered evaluator"):
        validate_study_manifest(extra, verify_bound_files=False)


def test_checkpoint_artifact_binding_enforces_source_seed_and_view() -> None:
    manifest = _manifest()
    checkpoint = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": "bc0",
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "base_model_id": manifest["bindings"]["base_model"]["model_id"],
        "base_model_revision": manifest["bindings"]["base_model"]["model_revision"],
        "base_snapshot_attestation_sha256": "9" * 64,
        "training_seed": 3408,
        "training_view_provenance_id": "d" * 64,
        "training_sources": ["d0_bc0"],
        "round1_view": None,
        "production_d1_quarantine_binding": _quarantine_binding("bc0"),
        "training_dataset_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "parent_model_revision": manifest["bindings"]["base_model"]["model_revision"],
        "adapter_path": "/scratch/checkpoints/bc0-seed3408/lora",
        "adapter_tree_sha256": "f" * 64,
        "runtime_accelerator_attestation": ACCELERATOR,
    }
    checkpoint.update(
        _training_integrity_fields(manifest, variant_id="bc0", seed=3408)
    )
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(checkpoint)

    result = validate_study_artifact_binding(
        manifest,
        checkpoint,
        variant_id="bc0",
        artifact_role="checkpoint",
        expected_source_commit=SOURCE_COMMIT,
        expected_training_seed=3408,
    )
    assert result["passed"] is True

    drifted_protocol = json.loads(json.dumps(checkpoint["training_protocol"]))
    drifted_protocol["configuration"]["trainer"]["learning_rate"] = 0.0002
    drifted_configuration = json.loads(
        json.dumps(checkpoint["training_configuration"])
    )
    drifted_configuration["lora"]["rank"] = 8
    drifted_rng = json.loads(json.dumps(checkpoint["training_rng_attestation"]))
    drifted_rng["pre_model_construction"]["engines"].remove("torch_cuda_all")
    for field, bad_value, message in (
        ("artifact_schema_version", 999, "artifact_schema_version"),
        ("reviewed_source_commit", "e" * 40, "source commit"),
        (
            "base_snapshot_attestation_sha256",
            "not-a-hash",
            "base_snapshot_attestation_sha256",
        ),
        ("training_seed", 99, "preregistered"),
        ("training_view_provenance_id", "not-a-hash", "provenance_id"),
        ("training_sources", ["d0_bc0", "natural_dagger1"], "training_sources"),
        ("round1_view", "full", "round1_view"),
        ("training_protocol", drifted_protocol, "training protocol"),
        (
            "training_configuration",
            drifted_configuration,
            "training configuration",
        ),
        ("training_rng_attestation", drifted_rng, "RNG attestation"),
        ("parent_checkpoint_receipt_id", "e" * 64, "must be null"),
        (
            "training_dataset_sha256",
            {"train": "1" * 64},
            "exactly train and validation",
        ),
    ):
        bad = {**checkpoint, field: bad_value}
        bad.pop("checkpoint_receipt_id")
        bad["checkpoint_receipt_id"] = stable_json_sha256(bad)
        with pytest.raises(StudyManifestError, match=message):
            validate_study_artifact_binding(
                manifest,
                bad,
                variant_id="bc0",
                artifact_role="checkpoint",
                expected_source_commit=SOURCE_COMMIT,
            )

    semantically_valid_but_unreviewed = _manifest()
    semantically_valid_but_unreviewed["training_seeds"] = [3407, 3408, 3410]
    with pytest.raises(StudyManifestError, match="exact immutable"):
        validate_study_artifact_binding(
            semantically_valid_but_unreviewed,
            checkpoint,
            variant_id="bc0",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
        )


def test_checkpoint_rejects_non_pro_name_claiming_rtx6000_class() -> None:
    manifest = _manifest()
    checkpoint = _checkpoint_artifact(manifest, variant_id="bc0", seed=3407)
    checkpoint["runtime_accelerator_attestation"] = json.loads(
        json.dumps(ACCELERATOR)
    )
    checkpoint["runtime_accelerator_attestation"]["devices"][0]["name"] = (
        "NVIDIA RTX 6000 Ada Generation"
    )
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_receipt_id"
        }
    )
    with pytest.raises(StudyManifestError, match="not an NVIDIA RTX Pro 6000"):
        validate_study_artifact_binding(
            manifest,
            checkpoint,
            variant_id="bc0",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3407,
        )


def test_checkpoint_rejects_low_memory_pro_claiming_rtx6000_class() -> None:
    manifest = _manifest()
    checkpoint = _checkpoint_artifact(manifest, variant_id="bc0", seed=3407)
    checkpoint["runtime_accelerator_attestation"]["devices"][0][
        "total_memory_bytes"
    ] = 48 * 1024**3
    checkpoint.pop("checkpoint_receipt_id")
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(checkpoint)

    with pytest.raises(StudyManifestError, match="below 90000 MiB"):
        validate_study_artifact_binding(
            manifest,
            checkpoint,
            variant_id="bc0",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3407,
        )


@pytest.mark.parametrize(
    ("name", "memory_bytes", "claimed_class"),
    [
        (
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            96 * 1024**3,
            "h200",
        ),
        ("NVIDIA H200", 1, "h200"),
        ("NVIDIA H200", 141 * 1024**3, "h100"),
    ],
)
def test_checkpoint_rejects_forged_hopper_class_identity(
    name: str,
    memory_bytes: int,
    claimed_class: str,
) -> None:
    manifest = _manifest()
    checkpoint = _checkpoint_artifact(manifest, variant_id="bc0", seed=3407)
    device = checkpoint["runtime_accelerator_attestation"]["devices"][0]
    device.update(
        {
            "name": name,
            "total_memory_bytes": memory_bytes,
            "accelerator_class": claimed_class,
        }
    )
    checkpoint.pop("checkpoint_receipt_id")
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(checkpoint)

    with pytest.raises(StudyManifestError, match="does not match its name"):
        validate_study_artifact_binding(
            manifest,
            checkpoint,
            variant_id="bc0",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3407,
        )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("torch_cuda_version", "CUDA runtime version"),
        ("compute_capability", "compute capability"),
    ],
)
def test_checkpoint_accelerator_binding_requires_runtime_identity(
    field: str,
    match: str,
) -> None:
    manifest = _manifest()
    checkpoint = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": "bc0",
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "base_model_id": manifest["bindings"]["base_model"]["model_id"],
        "base_model_revision": manifest["bindings"]["base_model"][
            "model_revision"
        ],
        "base_snapshot_attestation_sha256": "9" * 64,
        "training_seed": 3408,
        "training_view_provenance_id": "d" * 64,
        "training_sources": ["d0_bc0"],
        "round1_view": None,
        "production_d1_quarantine_binding": _quarantine_binding("bc0"),
        "training_dataset_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "parent_model_revision": manifest["bindings"]["base_model"][
            "model_revision"
        ],
        "adapter_path": "/scratch/checkpoints/bc0-seed3408/lora",
        "adapter_tree_sha256": "f" * 64,
        "runtime_accelerator_attestation": json.loads(json.dumps(ACCELERATOR)),
    }
    checkpoint.update(
        _training_integrity_fields(manifest, variant_id="bc0", seed=3408)
    )
    accelerator = checkpoint["runtime_accelerator_attestation"]
    if field == "compute_capability":
        accelerator["devices"][0].pop(field)
    else:
        accelerator.pop(field)
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(checkpoint)

    with pytest.raises(StudyManifestError, match=match):
        validate_study_artifact_binding(
            manifest,
            checkpoint,
            variant_id="bc0",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )


@pytest.mark.parametrize(
    ("variant_id", "training_sources", "round1_view"),
    [
        (
            "natural_dagger",
            ["d0_bc0", "natural_dagger1"],
            "natural-only",
        ),
        (
            "natural_dagger_probes",
            ["d0_bc0", "natural_dagger1", "observable_recovery_probe"],
            "full",
        ),
    ],
)
def test_round1_checkpoint_binding_enforces_exact_variant_view(
    variant_id: str,
    training_sources: list[str],
    round1_view: str,
) -> None:
    manifest = _manifest()
    checkpoint = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": variant_id,
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "base_model_id": manifest["bindings"]["base_model"]["model_id"],
        "base_model_revision": manifest["bindings"]["base_model"][
            "model_revision"
        ],
        "base_snapshot_attestation_sha256": "9" * 64,
        "training_seed": 3408,
        "training_view_provenance_id": QUARANTINE_PROVENANCE,
        "training_sources": training_sources,
        "round1_view": round1_view,
        "production_d1_quarantine_binding": _quarantine_binding(variant_id),
        "training_dataset_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "parent_model_revision": "e" * 64,
        "adapter_path": f"/scratch/checkpoints/{variant_id}-seed3408/lora",
        "adapter_tree_sha256": "f" * 64,
        "runtime_accelerator_attestation": json.loads(json.dumps(ACCELERATOR)),
    }
    checkpoint.update(
        _training_integrity_fields(
            manifest,
            variant_id=variant_id,
            seed=3408,
        )
    )
    checkpoint["checkpoint_receipt_id"] = stable_json_sha256(checkpoint)
    assert validate_study_artifact_binding(
        manifest,
        checkpoint,
        variant_id=variant_id,
        artifact_role="checkpoint",
        expected_source_commit=SOURCE_COMMIT,
        expected_training_seed=3408,
    )["passed"] is True

    for bad_parent in (None, "not-a-hash"):
        bad_parent_checkpoint = {
            **checkpoint,
            "parent_checkpoint_receipt_id": bad_parent,
        }
        bad_parent_checkpoint.pop("checkpoint_receipt_id")
        bad_parent_checkpoint["checkpoint_receipt_id"] = stable_json_sha256(
            bad_parent_checkpoint
        )
        with pytest.raises(
            StudyManifestError,
            match="parent_checkpoint_receipt_id",
        ):
            validate_study_artifact_binding(
                manifest,
                bad_parent_checkpoint,
                variant_id=variant_id,
                artifact_role="checkpoint",
                expected_source_commit=SOURCE_COMMIT,
                expected_training_seed=3408,
            )

    bad = {
        **checkpoint,
        "round1_view": "full" if round1_view != "full" else "natural-only",
    }
    bad.pop("checkpoint_receipt_id")
    bad["checkpoint_receipt_id"] = stable_json_sha256(bad)
    with pytest.raises(StudyManifestError, match="round1_view"):
        validate_study_artifact_binding(
            manifest,
            bad,
            variant_id=variant_id,
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )


def test_checkpoint_quarantine_binding_rejects_missing_forged_or_mismatched_evidence() -> None:
    manifest = _manifest()
    checkpoint = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": "natural_dagger",
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "base_model_id": manifest["bindings"]["base_model"]["model_id"],
        "base_model_revision": manifest["bindings"]["base_model"][
            "model_revision"
        ],
        "base_snapshot_attestation_sha256": "9" * 64,
        "training_seed": 3408,
        "training_view_provenance_id": QUARANTINE_PROVENANCE,
        "training_sources": ["d0_bc0", "natural_dagger1"],
        "round1_view": "natural-only",
        "production_d1_quarantine_binding": _quarantine_binding(
            "natural_dagger"
        ),
        "training_dataset_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "parent_model_revision": "e" * 64,
        "adapter_path": "/scratch/checkpoints/natural-seed3408/lora",
        "adapter_tree_sha256": "f" * 64,
        "runtime_accelerator_attestation": json.loads(json.dumps(ACCELERATOR)),
    }
    checkpoint.update(
        _training_integrity_fields(
            manifest,
            variant_id="natural_dagger",
            seed=3408,
        )
    )

    def rehash(value: dict) -> dict:
        value.pop("checkpoint_receipt_id", None)
        value["checkpoint_receipt_id"] = stable_json_sha256(value)
        return value

    assert validate_study_artifact_binding(
        manifest,
        rehash(checkpoint),
        variant_id="natural_dagger",
        artifact_role="checkpoint",
        expected_source_commit=SOURCE_COMMIT,
        expected_training_seed=3408,
    )["passed"]

    missing = json.loads(json.dumps(checkpoint))
    missing.pop("production_d1_quarantine_binding")
    with pytest.raises(StudyManifestError, match="missing required fields"):
        validate_study_artifact_binding(
            manifest,
            rehash(missing),
            variant_id="natural_dagger",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )

    cases = (
        ("candidate count", "candidate_rows", 24),
        ("quarantine count", "quarantined_rows", 1),
        ("report hash", "audit_report_sha256", "0" * 64),
        ("provenance", "generation_provenance_id", "a" * 64),
    )
    for message, field, forged_value in cases:
        forged = json.loads(json.dumps(checkpoint))
        forged["production_d1_quarantine_binding"][field] = forged_value
        with pytest.raises(StudyManifestError, match=message):
            validate_study_artifact_binding(
                manifest,
                rehash(forged),
                variant_id="natural_dagger",
                artifact_role="checkpoint",
                expected_source_commit=SOURCE_COMMIT,
                expected_training_seed=3408,
            )

    self_consistent_forgery = json.loads(json.dumps(checkpoint))
    forged_summary = {
        **QUARANTINE_SUMMARY,
        "total_rows": 26,
        "candidate_rows": 26,
        "passed_rows": 26,
    }
    forged_binding = self_consistent_forgery[
        "production_d1_quarantine_binding"
    ]
    forged_binding["summary"] = forged_summary
    forged_binding["candidate_rows"] = 26
    forged_binding["audit_report_sha256"] = stable_json_sha256(forged_summary)
    with pytest.raises(StudyManifestError, match="authenticated by the generation"):
        validate_study_artifact_binding(
            manifest,
            rehash(self_consistent_forgery),
            variant_id="natural_dagger",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )

    bc0 = _checkpoint_artifact(manifest, variant_id="bc0", seed=3408)
    bc0["production_d1_quarantine_binding"] = _quarantine_binding(
        "natural_dagger"
    )
    with pytest.raises(StudyManifestError, match="canonical not-applicable"):
        validate_study_artifact_binding(
            manifest,
            rehash(bc0),
            variant_id="bc0",
            artifact_role="checkpoint",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )


def test_evaluation_binding_distinguishes_base_and_adapted_models() -> None:
    manifest = _manifest()
    evaluation = manifest["bindings"]["evaluation"]
    common = {
        "artifact_role": "evaluation",
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "frozen_suite_sha256": evaluation["suite_sha256"],
        "evaluation_policy_sha256": evaluation["policy_sha256"],
    }
    base = {
        **common,
        "variant_id": "base",
        "model_id": manifest["bindings"]["base_model"]["model_id"],
        "model_revision": manifest["bindings"]["base_model"]["model_revision"],
        "checkpoint_receipt_id": None,
        "checkpoint_adapter_tree_sha256": None,
        "training_seed": None,
    }
    adapted = {
        **common,
        "variant_id": "natural_dagger_probes",
        "model_id": "/scratch/checkpoints/full-seed3409",
        "model_revision": TREE_REVISION,
        "checkpoint_receipt_id": "2" * 64,
        "checkpoint_adapter_tree_sha256": TREE_REVISION,
        "training_seed": 3409,
    }

    assert validate_study_artifact_binding(
        manifest,
        base,
        variant_id="base",
        artifact_role="evaluation",
        expected_source_commit=SOURCE_COMMIT,
    )["passed"]
    assert validate_study_artifact_binding(
        manifest,
        adapted,
        variant_id="natural_dagger_probes",
        artifact_role="evaluation",
        expected_source_commit=SOURCE_COMMIT,
        expected_training_seed=3409,
    )["passed"]

    with pytest.raises(StudyManifestError, match="training_seed must be null"):
        validate_study_artifact_binding(
            manifest,
            {**base, "training_seed": 3407},
            variant_id="base",
            artifact_role="evaluation",
            expected_source_commit=SOURCE_COMMIT,
        )


def test_development_evaluation_binds_exact_30_root_holdout_separately() -> None:
    manifest = _manifest()
    development = {
        "artifact_role": "development_evaluation",
        "variant_id": "natural_dagger",
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "model_id": "/scratch/checkpoints/natural-seed3408",
        "model_revision": TREE_REVISION,
        "checkpoint_receipt_id": "2" * 64,
        "checkpoint_adapter_tree_sha256": TREE_REVISION,
        "training_seed": 3408,
        "development_holdout_sha256": "d" * 64,
        "development_holdout_provenance_id": "e" * 64,
        "development_holdout_root_set_sha256": "f" * 64,
        "development_holdout_physical_roots": 30,
        "development_evaluation_contract_sha256": (
            EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
        ),
        "evaluation_protocol": "diagnostic_model_selection_only",
    }

    assert validate_study_artifact_binding(
        manifest,
        development,
        variant_id="natural_dagger",
        artifact_role="development_evaluation",
        expected_source_commit=SOURCE_COMMIT,
        expected_training_seed=3408,
    )["passed"]

    with pytest.raises(StudyManifestError, match="exactly 30"):
        validate_study_artifact_binding(
            manifest,
            {**development, "development_holdout_physical_roots": 29},
            variant_id="natural_dagger",
            artifact_role="development_evaluation",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )
    with pytest.raises(StudyManifestError, match="model-selection-only"):
        validate_study_artifact_binding(
            manifest,
            {**development, "evaluation_protocol": "canonical"},
            variant_id="natural_dagger",
            artifact_role="development_evaluation",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )
    with pytest.raises(StudyManifestError, match="exact preregistered"):
        validate_study_artifact_binding(
            manifest,
            {**development, "development_evaluation_contract_sha256": "1" * 64},
            variant_id="natural_dagger",
            artifact_role="development_evaluation",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )
    with pytest.raises(StudyManifestError, match="differs from checkpoint tree"):
        validate_study_artifact_binding(
            manifest,
            {**development, "checkpoint_adapter_tree_sha256": "3" * 64},
            variant_id="natural_dagger",
            artifact_role="development_evaluation",
            expected_source_commit=SOURCE_COMMIT,
            expected_training_seed=3408,
        )
