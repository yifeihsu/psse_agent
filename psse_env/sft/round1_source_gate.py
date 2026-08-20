"""Executable Round-1 aggregate, learner-seed, and provenance gate.

The Slurm launcher delegates its release-critical D0/D1/probe source checks to
this module so the same validation can be imported and exercised directly in
unit tests.  The gate validates only immutable artifacts; it does not mutate
the aggregate or adapter tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.dagger.build_dagger1_aggregate import (
    generation_id_independent_rows,
    validate_offline_teacher_target_quarantine_summary,
    validate_round1_learner_seed,
)
from psse_env.dagger.collect_dagger1 import DAGGER1_SCENARIO_BUILDER_CONTRACT
from psse_env.dagger.dataset_builder import examples_to_chat_sft, load_jsonl
from psse_env.dagger.replay_buffer import audit_dagger1_training_support
from psse_env.dagger.round1_view_policy import (
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_view_policy_digest,
)
from psse_env.dagger.three_source_view import (
    THREE_SOURCE_VIEW_CONTRACT,
    build_dagger1_three_source_view,
)
from psse_env.sft.gates import GateError
from psse_env.sft.provenance import (
    ROUND1_AGGREGATE_BUILDER_CONTRACT,
    file_sha256,
    stable_json_sha256,
)


ROUND1_IMMUTABLE_VIEW_NAMES = (
    "aggregate.raw.jsonl",
    "aggregate.d0.raw.jsonl",
    "aggregate.d1.raw.jsonl",
    "aggregate.probe.raw.jsonl",
    "aggregate.train_view.raw.jsonl",
    "aggregate.train_view.jsonl",
    "aggregate.validation.jsonl",
    "aggregate.test.jsonl",
)
ROUND1_CANONICAL_PROVENANCE_NAME = "aggregate.generation_provenance.json"
ROUND1_CANONICAL_PREFLIGHT_NAME = "aggregate.preflight.json"
ROUND1_CANONICAL_TRAIN_NAME = "aggregate.train_view.jsonl"
ROUND1_CANONICAL_VALIDATION_NAME = "aggregate.validation.jsonl"
ROUND1_CANONICAL_TEST_NAME = "aggregate.test.jsonl"
_REQUIRED_D1_REPORTS = (
    "offline_teacher_target_quarantine_summary",
    "recovery_label_audit",
    "target_aware_state_class_audit",
    "independent_root_support",
    "deterministic_collection_selection_binding",
    "three_source_training_support",
)
_REQUIRED_SEMANTIC_LEAF_REPORTS = (
    "natural_teacher_realizability",
    "training_view_teacher_realizability",
    "natural_approximate_teacher_realizability",
    "training_view_approximate_teacher_realizability",
)
_REQUIRED_SEMANTIC_STRATIFIED_REPORTS = (
    "approximate_teacher_realizability_by_scenario_family",
    "approximate_teacher_realizability_by_state_class",
    "approximate_teacher_realizability_by_recovery_stratum",
)
_D1_DEVELOPMENT_HOLDOUT_BINDING_KEYS = frozenset(
    {
        "holdout_sha256",
        "manifest_sha256",
        "generator_report_sha256",
        "physical_root_count",
        "root_set_sha256",
    }
)


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(
            f"{label} is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise GateError(f"{label} must contain a JSON object")
    return dict(decoded)


def round1_source_binding_required(*dataset_paths: str | Path) -> bool:
    """Return whether any consumed dataset sits beside a Round-1 aggregate.

    This detection happens before row eligibility is trusted.  Consequently,
    flipping probe flags or relabeling rows cannot suppress the source gate for
    a canonical Round-1 aggregate.  An unreadable sibling provenance file is
    treated as requiring the gate so corruption fails closed downstream.
    """

    for value in dataset_paths:
        path = Path(value).resolve()
        provenance_path = path.parent / ROUND1_CANONICAL_PROVENANCE_NAME
        if not provenance_path.is_file():
            continue
        try:
            provenance = _load_json_mapping(
                provenance_path,
                label="Round-1 sibling generation provenance",
            )
        except GateError:
            return True
        descriptor = _mapping(provenance.get("generation_descriptor"))
        if descriptor.get("builder_contract") == ROUND1_AGGREGATE_BUILDER_CONTRACT:
            return True
    return False


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _audit_report_hashes(
    *,
    recomputed_d1: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> dict[str, str]:
    reports: dict[str, Mapping[str, Any]] = {
        "d1_offline_teacher_target_quarantine_summary": _mapping(
            recomputed_d1.get("offline_teacher_target_quarantine_summary")
        ),
        "d1_recovery_label_audit": _mapping(
            recomputed_d1.get("recovery_label_audit")
        ),
        "d1_target_aware_state_class_audit": _mapping(
            recomputed_d1.get("target_aware_state_class_audit")
        ),
        "d1_independent_root_support": _mapping(
            recomputed_d1.get("independent_root_support")
        ),
        "d1_deterministic_collection_selection_binding": _mapping(
            recomputed_d1.get("deterministic_collection_selection_binding")
        ),
        "d1_three_source_training_support": _mapping(
            recomputed_d1.get("three_source_training_support")
        ),
        "union_realizability": semantic,
    }
    for key in (
        *_REQUIRED_SEMANTIC_LEAF_REPORTS,
        *_REQUIRED_SEMANTIC_STRATIFIED_REPORTS,
    ):
        report = semantic.get(key)
        if isinstance(report, Mapping):
            reports[f"union_{key}"] = report
    return {
        name: stable_json_sha256(report)
        for name, report in sorted(reports.items())
    }


def validate_round1_source_mix_gate(
    provenance_path: str | Path,
    preflight_path: str | Path,
    *,
    reviewed_source_commit: str,
    initial_adapter_revision: str,
    train_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one immutable Round-1 aggregate and its exact learner seed.

    ``GateError`` contains all independently detectable failures so an HPC job
    exits before loading the model or importing the GPU stack.
    """

    provenance_file = Path(provenance_path).resolve()
    preflight_file = Path(preflight_path).resolve()
    aggregate_dir = provenance_file.parent
    if (
        provenance_file != aggregate_dir / ROUND1_CANONICAL_PROVENANCE_NAME
        or preflight_file != aggregate_dir / ROUND1_CANONICAL_PREFLIGHT_NAME
    ):
        raise GateError(
            "Round-1 source gate requires canonical sibling provenance and "
            "preflight paths."
        )
    expected_data_paths = {
        "train": aggregate_dir / ROUND1_CANONICAL_TRAIN_NAME,
        "validation": aggregate_dir / ROUND1_CANONICAL_VALIDATION_NAME,
        "test": aggregate_dir / ROUND1_CANONICAL_TEST_NAME,
    }
    supplied_data_paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    mismatched_data_paths = [
        role
        for role, value in supplied_data_paths.items()
        if value is not None and Path(value).resolve() != expected_data_paths[role]
    ]
    if mismatched_data_paths:
        raise GateError(
            "Round-1 source gate is not coupled to the canonical dataset "
            "path(s): " + ", ".join(sorted(mismatched_data_paths))
        )
    provenance = _load_json_mapping(
        provenance_file,
        label="Round-1 generation provenance",
    )
    preflight = _load_json_mapping(
        preflight_file,
        label="Round-1 aggregate preflight",
    )
    descriptor = _mapping(provenance.get("generation_descriptor"))
    training_view = _mapping(preflight.get("training_view"))
    allocation = _mapping(training_view.get("placed"))
    d1_manifest = _mapping(preflight.get("d1_collection_manifest"))
    recomputed_d1 = _mapping(preflight.get("recomputed_d1_audits"))
    semantic = _mapping(preflight.get("semantic_realizability"))
    preflight_audit_hashes = _mapping(preflight.get("audit_report_sha256"))
    descriptor_audit_hashes = _mapping(descriptor.get("audit_report_sha256"))
    failures: list[str] = []

    if provenance.get("release_eligible") is not True:
        failures.append("round-1 generation provenance is not release eligible")
    if descriptor.get("builder_contract") != ROUND1_AGGREGATE_BUILDER_CONTRACT:
        failures.append("aggregate is not the reviewed D0/D1/probe build")

    provenance_id = provenance.get("generation_provenance_id")
    descriptor_id = stable_json_sha256(descriptor)
    if provenance_id != descriptor_id:
        failures.append(
            "round-1 generation_provenance_id does not hash its "
            "generation_descriptor"
        )
    if preflight.get("generation_provenance_id") != provenance_id:
        failures.append("round-1 preflight and provenance IDs differ")

    source = _mapping(descriptor.get("source_state"))
    if source.get("source_commit") != reviewed_source_commit:
        failures.append("round-1 aggregate source commit differs from reviewed source")
    if source.get("release_eligible_source") is not True:
        failures.append("round-1 aggregate source is not release eligible")
    if training_view.get("contract") != THREE_SOURCE_VIEW_CONTRACT:
        failures.append("preflight lacks the reviewed three-source view contract")
    if descriptor.get("training_view_report_sha256") != stable_json_sha256(
        training_view
    ):
        failures.append("preflight source-mix report is not bound by provenance")

    computed_audit_hashes = _audit_report_hashes(
        recomputed_d1=recomputed_d1,
        semantic=semantic,
    )
    if (
        not descriptor_audit_hashes
        or descriptor_audit_hashes != preflight_audit_hashes
        or descriptor_audit_hashes != computed_audit_hashes
    ):
        failures.append("round-1 semantic audit reports are not provenance bound")

    missing_d1_reports = [
        name
        for name in _REQUIRED_D1_REPORTS
        if not isinstance(recomputed_d1.get(name), Mapping)
    ]
    required_semantic_reports = (
        *_REQUIRED_SEMANTIC_LEAF_REPORTS,
        *_REQUIRED_SEMANTIC_STRATIFIED_REPORTS,
    )
    missing_semantic_reports = [
        name
        for name in required_semantic_reports
        if not isinstance(semantic.get(name), Mapping) or not semantic.get(name)
    ]
    if missing_d1_reports:
        failures.append(
            "round-1 preflight lacks required recomputed D1 audits: "
            + ", ".join(missing_d1_reports)
        )
    if missing_semantic_reports:
        failures.append(
            "round-1 preflight lacks required semantic reports: "
            + ", ".join(missing_semantic_reports)
        )

    try:
        validate_offline_teacher_target_quarantine_summary(
            recomputed_d1.get("offline_teacher_target_quarantine_summary")
        )
    except ValueError as exc:
        failures.append(str(exc))

    d1_audits_pass = all(
        isinstance(recomputed_d1.get(name), Mapping)
        and recomputed_d1[name].get("passed") is True
        for name in _REQUIRED_D1_REPORTS
    )
    semantic_leaves_pass = all(
        isinstance(semantic.get(name), Mapping)
        and semantic[name].get("passed") is True
        for name in _REQUIRED_SEMANTIC_LEAF_REPORTS
    )
    semantic_strata_pass = all(
        isinstance(semantic.get(name), Mapping)
        and bool(semantic[name])
        and all(
            isinstance(stratum_report, Mapping)
            and stratum_report.get("release_gate_passed") is True
            for stratum_report in semantic[name].values()
        )
        for name in _REQUIRED_SEMANTIC_STRATIFIED_REPORTS
    )
    if (
        not d1_audits_pass
        or not semantic_leaves_pass
        or not semantic_strata_pass
        or semantic.get("passed") is not True
    ):
        failures.append("round-1 recomputed semantic audits are not all passing")

    expected_allocation = ROUND1_THREE_SOURCE_VIEW_POLICY["allocation"]
    if allocation != expected_allocation:
        failures.append(
            "Round-1 source allocation is not the exact frozen 1317/525/38 view"
        )
    if (
        descriptor.get("round1_view_policy_digest")
        != round1_view_policy_digest()
        or descriptor.get("round1_view_policy")
        != ROUND1_THREE_SOURCE_VIEW_POLICY
        or training_view.get("policy_digest") != round1_view_policy_digest()
    ):
        failures.append("Round-1 view policy or digest is not provenance bound")
    d1_recovery_rows = int(allocation.get("natural_d1_rows") or 0)
    probe_rows_count = int(
        allocation.get("observable_recovery_probe_rows") or 0
    )

    d1_source = _mapping(d1_manifest.get("source_state"))
    if (
        d1_manifest.get("training_eligible") is not True
        or d1_manifest.get("release_evidence_eligible") is not False
        or d1_source.get("source_commit") != reviewed_source_commit
        or d1_source.get("release_eligible_source") is not True
    ):
        failures.append(
            "D1 collection manifest is not approved for current-source training"
        )
    if (
        d1_manifest.get("scenario_builder_contract")
        != DAGGER1_SCENARIO_BUILDER_CONTRACT
        or not d1_manifest.get("scenario_manifest_sha256")
    ):
        failures.append("D1 collection lacks the reviewed fresh-root scenario binding")

    inputs = _mapping(descriptor.get("input_artifacts"))
    d0_manifest_sha256 = inputs.get("d0_manifest_sha256")
    if not _is_sha256(d0_manifest_sha256):
        failures.append(
            "round-1 provenance lacks the bound D0 aggregate manifest hash"
        )
    elif d1_manifest.get("d0_manifest_sha256") != d0_manifest_sha256:
        failures.append(
            "round-1 D0 aggregate manifest binding differs from the D1 "
            "collection manifest"
        )
    descriptor_holdout_value = inputs.get("d1_development_holdout")
    preflight_holdout_value = preflight.get("d1_development_holdout")
    descriptor_holdout = _mapping(descriptor_holdout_value)
    preflight_holdout = _mapping(preflight_holdout_value)
    if (
        not isinstance(descriptor_holdout_value, Mapping)
        or set(descriptor_holdout) != _D1_DEVELOPMENT_HOLDOUT_BINDING_KEYS
    ):
        failures.append(
            "round-1 provenance development-holdout binding has an invalid shape"
        )
    if (
        not isinstance(preflight_holdout_value, Mapping)
        or set(preflight_holdout) != _D1_DEVELOPMENT_HOLDOUT_BINDING_KEYS
    ):
        failures.append(
            "round-1 preflight development-holdout binding has an invalid shape"
        )

    descriptor_root_count = descriptor_holdout.get("physical_root_count")
    descriptor_hashes_valid = all(
        _is_sha256(descriptor_holdout.get(key))
        for key in (
            "holdout_sha256",
            "manifest_sha256",
            "generator_report_sha256",
            "root_set_sha256",
        )
    )
    descriptor_count_valid = (
        isinstance(descriptor_root_count, int)
        and not isinstance(descriptor_root_count, bool)
        and descriptor_root_count > 0
    )
    if not descriptor_hashes_valid or not descriptor_count_valid:
        failures.append(
            "round-1 provenance development-holdout hashes or root count are invalid"
        )
    if descriptor_holdout != preflight_holdout:
        failures.append(
            "round-1 provenance and preflight development-holdout bindings differ"
        )

    manifest_root_count = d1_manifest.get("development_holdout_root_count")
    manifest_physical_root_count = d1_manifest.get(
        "development_physical_root_count"
    )
    expected_holdout = {
        "holdout_sha256": d1_manifest.get("development_holdout_sha256"),
        "manifest_sha256": d1_manifest.get(
            "development_holdout_manifest_sha256"
        ),
        "generator_report_sha256": d1_manifest.get(
            "development_holdout_generator_report_sha256"
        ),
        "physical_root_count": manifest_root_count,
        "root_set_sha256": d1_manifest.get(
            "development_holdout_root_set_sha256"
        ),
    }
    manifest_holdout_valid = (
        _is_sha256(expected_holdout["holdout_sha256"])
        and _is_sha256(expected_holdout["manifest_sha256"])
        and _is_sha256(expected_holdout["generator_report_sha256"])
        and _is_sha256(expected_holdout["root_set_sha256"])
        and isinstance(manifest_root_count, int)
        and not isinstance(manifest_root_count, bool)
        and manifest_root_count > 0
        and isinstance(manifest_physical_root_count, int)
        and manifest_physical_root_count == manifest_root_count
        and not isinstance(manifest_physical_root_count, bool)
    )
    if not manifest_holdout_valid:
        failures.append(
            "D1 collection manifest development-holdout binding is invalid"
        )
    if descriptor_holdout != expected_holdout:
        failures.append(
            "round-1 development-holdout binding differs from the D1 collection manifest"
        )

    d1_manifest_sha256 = inputs.get("d1_manifest_sha256")
    if (
        not inputs.get("d1_rows_sha256")
        or not d1_manifest_sha256
        or inputs.get("d1_manifest_content_sha256")
        != stable_json_sha256(d1_manifest)
    ):
        failures.append("round-1 provenance does not bind D1 rows and manifest")
    else:
        try:
            validate_round1_learner_seed(
                d1_manifest,
                collection_manifest_sha256=str(d1_manifest_sha256),
                aggregate_learner_seed=_mapping(descriptor.get("learner_seed")),
                initial_adapter_revision=initial_adapter_revision,
            )
        except ValueError as exc:
            failures.append(str(exc))

    dataset_hashes = _mapping(provenance.get("dataset_hashes"))
    for name in ROUND1_IMMUTABLE_VIEW_NAMES:
        path = aggregate_dir / name
        recorded_hash = dataset_hashes.get(name)
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            failures.append(f"round-1 provenance lacks a valid hash for {name}")
            continue
        if not path.is_file():
            failures.append(f"round-1 immutable source view is missing: {name}")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != recorded_hash.lower():
            failures.append(
                f"round-1 immutable source view hash mismatch for {name}: "
                f"recorded {recorded_hash}, computed {actual_hash}"
            )

    # Hashes establish byte identity, but a caller could forge a view and
    # update an unbound top-level hash.  Rebuild the frozen placement from the
    # three immutable source ledgers and require byte-equivalent parsed rows.
    try:
        d0_rows = load_jsonl(aggregate_dir / "aggregate.d0.raw.jsonl")
        d1_rows = load_jsonl(aggregate_dir / "aggregate.d1.raw.jsonl")
        probe_rows = load_jsonl(aggregate_dir / "aggregate.probe.raw.jsonl")
        materialized_raw = load_jsonl(
            aggregate_dir / "aggregate.train_view.raw.jsonl"
        )
        materialized_chat = load_jsonl(
            aggregate_dir / "aggregate.train_view.jsonl"
        )
        materialized_union = load_jsonl(aggregate_dir / "aggregate.raw.jsonl")
        materialized_validation = load_jsonl(
            aggregate_dir / ROUND1_CANONICAL_VALIDATION_NAME
        )
        materialized_test = load_jsonl(
            aggregate_dir / ROUND1_CANONICAL_TEST_NAME
        )
        expected_source_hashes = _mapping(
            inputs.get("immutable_source_view_content_sha256")
        )
        actual_source_hashes = {
            "d0_bc0": stable_json_sha256(d0_rows),
            "natural_dagger1": stable_json_sha256(d1_rows),
            "observable_recovery_probe": stable_json_sha256(probe_rows),
        }
        if expected_source_hashes != actual_source_hashes:
            failures.append(
                "Round-1 immutable source ledgers differ from descriptor-bound content"
            )
        rebuilt_raw, rebuilt_report = build_dagger1_three_source_view(
            d0_rows=d0_rows,
            natural_d1_rows=d1_rows,
            probe_rows=probe_rows,
            policy=ROUND1_THREE_SOURCE_VIEW_POLICY,
        )
        if rebuilt_report != training_view or rebuilt_report.get("passed") is not True:
            failures.append(
                "Round-1 three-source placement report does not recompute exactly"
            )
        for row in rebuilt_raw:
            row["generation_provenance_id"] = provenance_id
        if rebuilt_raw != materialized_raw:
            failures.append(
                "Round-1 train_view.raw is not the exact deterministic "
                "three-source reconstruction"
            )
        rebuilt_chat = examples_to_chat_sft(
            rebuilt_raw,
            protocol="canonical",
            allow_ineligible_auxiliary=True,
        )
        if rebuilt_chat != materialized_chat:
            failures.append(
                "Round-1 chat training view is not the exact canonical export"
            )
        if materialized_union != [*d0_rows, *d1_rows, *probe_rows]:
            failures.append(
                "Round-1 aggregate.raw is not the exact three-source ledger union"
            )
        expected_holdout_hashes = _mapping(
            inputs.get("immutable_holdout_content_sha256")
        )
        actual_holdout_hashes = {
            "validation": stable_json_sha256(
                generation_id_independent_rows(materialized_validation)
            ),
            "test": stable_json_sha256(
                generation_id_independent_rows(materialized_test)
            ),
        }
        holdout_id_failures = [
            f"{split_name}[{index}]"
            for split_name, split_rows in (
                ("validation", materialized_validation),
                ("test", materialized_test),
            )
            for index, row in enumerate(split_rows)
            if row.get("generation_provenance_id") != provenance_id
            or (
                isinstance(row.get("metadata"), Mapping)
                and row["metadata"].get("generation_provenance_id")
                != provenance_id
            )
        ]
        if (
            expected_holdout_hashes != actual_holdout_hashes
            or holdout_id_failures
        ):
            failures.append(
                "Round-1 validation/test rows are not the exact "
                "descriptor-bound holdouts"
            )
        recomputed_support = audit_dagger1_training_support(d1_rows, probe_rows)
        preflight_support = _mapping(
            preflight.get("three_source_training_support")
        )
        if (
            recomputed_support.get("passed") is not True
            or recomputed_support != preflight_support
            or recomputed_support
            != _mapping(recomputed_d1.get("three_source_training_support"))
            or descriptor.get("training_support_report_sha256")
            != stable_json_sha256(recomputed_support)
        ):
            failures.append(
                "Round-1 natural/probe/combined support does not recompute exactly"
            )

        probe_binding = _mapping(preflight.get("probe_binding"))
        provenance_probe_binding = _mapping(provenance.get("probe_binding"))
        probe_id = inputs.get("probe_generation_provenance_id")
        if (
            probe_binding.get("passed") is not True
            or probe_binding != provenance_probe_binding
            or probe_binding.get("generation_provenance_id") != probe_id
            or inputs.get("probe_binding_report_sha256")
            != stable_json_sha256(probe_binding)
            or probe_binding.get("view_policy_digest")
            != round1_view_policy_digest()
            or probe_binding.get("probe_rows") != len(probe_rows)
        ):
            failures.append("Round-1 probe manifest/validator binding is invalid")
        malformed_probe = [
            str(row.get("example_id") or index)
            for index, row in enumerate(probe_rows)
            if not (
                row.get("generation_provenance_id") == probe_id
                and row.get("dataset_mode") == "production"
                and row.get("dataset_source") == "observable_recovery_probe"
                and row.get("replay_source") == "observable_recovery_probe"
                and row.get("state_origin") == "observable_recovery_probe"
                and row.get("state_visited_by") == "observable_recovery_probe"
                and row.get("collector_contract")
                == "dagger1_observable_recovery_probe_v1"
                and row.get("collection_role") == "auxiliary_training"
                and row.get("auxiliary_training_eligible") is True
                and row.get("production_label_eligible") is False
                and row.get("natural_on_policy_support_eligible") is False
                and row.get("training_decision_evidence_verified") is True
                and row.get("recovery_stratum")
                in {
                    "post_failure_no_candidate",
                    "unsupported_correction_recovery",
                }
            )
        ]
        if malformed_probe:
            failures.append(
                "Round-1 probe source contains malformed auxiliary rows: "
                + ", ".join(malformed_probe[:5])
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failures.append(
            "Round-1 exact three-source reconstruction failed: "
            f"{type(exc).__name__}: {exc}"
        )

    if failures:
        raise GateError(
            "Round-1 source-mix gate is NO-GO:\n- " + "\n- ".join(failures)
        )
    return {
        "passed": True,
        "generation_provenance_id": provenance_id,
        "d1_recovery_rows": d1_recovery_rows,
        "probe_rows": probe_rows_count,
        "source_allocation": dict(expected_allocation),
        "round1_view_policy_digest": round1_view_policy_digest(),
        "aggregate_dir": str(aggregate_dir),
        "canonical_dataset_paths": {
            role: str(path) for role, path in expected_data_paths.items()
        },
        "canonical_dataset_content_sha256": {
            "train": stable_json_sha256(materialized_chat),
            "validation": stable_json_sha256(materialized_validation),
            "test": stable_json_sha256(materialized_test),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a provenance-bound Round-1 D0/D1/probe aggregate"
    )
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--reviewed-source-commit", required=True)
    parser.add_argument("--initial-adapter-revision", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = validate_round1_source_mix_gate(
            args.provenance,
            args.preflight,
            reviewed_source_commit=args.reviewed_source_commit,
            initial_adapter_revision=args.initial_adapter_revision,
        )
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "Round-1 D0/D1/probe source gate passed: "
        f"allocation={report['source_allocation']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())
