from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import psse_env.dagger.dataset_builder as dataset_builder_module
import psse_env.dagger.dagger1_semantic_audit as semantic_audit_module
import psse_env.dagger.collect_dagger1 as collect_dagger1_module
import psse_env.dagger.offline_teacher_target_audit as offline_audit_module
import psse_env.dagger.replay_buffer as replay_buffer_module
import psse_env.dagger.rollout_collector as rollout_collector_module
import psse_env.dagger.three_source_view as three_source_view_module
import psse_env.dagger.natural_only_view as natural_only_view_module
import psse_env.dagger.build_recovery_probe_suite as recovery_probe_builder_module
from psse_env.dagger.build_recovery_probe_suite import (
    validate_recovery_probe_suite_binding,
)
from psse_env.dagger.dataset_builder import examples_to_chat_sft, write_jsonl
from psse_env.dagger.dagger1_semantic_audit import (
    audit_dagger1_union_realizability,
)
from psse_env.dagger.offline_teacher_target_audit import (
    validate_offline_teacher_target_audit_metadata,
)
from psse_env.dagger.collect_dagger1 import (
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_ENV_FACTORY_SPEC,
    DEFAULT_EVALUATION_POLICY,
    DEFAULT_FORBIDDEN_SUITE,
    DEFAULT_POLICY_FACTORY_SPEC,
    frozen_physical_roots,
    validate_development_holdout_binding,
    validate_export_rows_truth_free,
)
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.dagger.replay_buffer import (
    DAGGER1_ROUND1_SOURCE_CAPACITY_CONTRACT,
    audit_dagger1_training_support,
    dagger1_round1_source_capacity_report,
)
from psse_env.dagger.round1_view_policy import (
    ROUND1_NATURAL_ONLY_VIEW_POLICY,
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_natural_only_view_policy_digest,
    round1_view_policy_digest,
)
from psse_env.dagger.natural_only_view import (
    NATURAL_ONLY_VIEW_BUILD_CONTRACT,
    build_round1_natural_only_view,
)
from psse_env.dagger.three_source_view import (
    FINAL_VIEW_SUPPORT_CONTRACT,
    THREE_SOURCE_VIEW_CONTRACT,
    audit_dagger1_final_view_support,
    build_dagger1_three_source_view,
)
from psse_env.dagger.rollout_collector import (
    OFFLINE_TEACHER_TARGET_QUARANTINE_SUMMARY_CONTRACT,
    audit_dagger1_recovery_labels,
    audit_target_aware_state_classes,
    summarize_dagger1_offline_teacher_target_quarantine,
)
from psse_env.sft.provenance import (
    AGGREGATE_MANIFEST_FILENAME,
    ROUND1_AGGREGATE_BUILDER_CONTRACT,
    file_sha256,
    git_source_state,
    stable_json_sha256,
    tool_schema_hashes,
    validate_aggregate_manifest_binding,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle


_ADAPTER_TREE_REVISION = re.compile(r"[0-9a-f]{64}")
DAGGER1_AGGREGATE_SELECTION_BINDING_CONTRACT = (
    "dagger1_aggregate_full_ledger_selection_binding_v1"
)

_ROUND1_OUTPUT_FILENAMES = (
    "aggregate.raw.jsonl",
    "aggregate.d0.raw.jsonl",
    "aggregate.d1.raw.jsonl",
    "aggregate.probe.raw.jsonl",
    "aggregate.train_view.raw.jsonl",
    "aggregate.train_view.jsonl",
    "aggregate.natural_only.train_view.raw.jsonl",
    "aggregate.natural_only.train_view.jsonl",
    "aggregate.validation.jsonl",
    "aggregate.test.jsonl",
    "aggregate.generation_provenance.json",
    "aggregate.preflight.json",
    "SHA256SUMS",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def validate_dagger1_execution_pipeline_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the approved overlap/barrier contract at release ingestion."""

    expected = collect_dagger1_module.dagger1_execution_pipeline_contract(
        overlap_policy_audit=True
    )
    observed = manifest.get("execution_pipeline_contract")
    if (
        not isinstance(observed, Mapping)
        or stable_json_sha256(observed) != stable_json_sha256(expected)
    ):
        raise ValueError(
            "D1 release-strict collection lacks the approved policy/audit "
            "execution-pipeline contract"
        )
    return copy.deepcopy(expected)


def _preflight_round1_output_directory(output_dir: Path) -> None:
    """Require an absent or genuinely empty final publication directory."""

    if output_dir.is_symlink() or (
        output_dir.exists() and not output_dir.is_dir()
    ):
        raise FileExistsError(
            "Round-1 output destination already exists and is not an empty "
            f"directory: {output_dir}"
        )
    if output_dir.is_dir():
        occupied = sorted(output_dir.iterdir(), key=lambda path: path.name)
        if occupied:
            raise FileExistsError(
                "Round-1 output files already exist: "
                + ", ".join(str(path) for path in occupied)
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)


def _write_text_artifact(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _verify_recorded_hash(
    *, provenance: Mapping[str, Any], path: Path, label: str
) -> None:
    hashes = provenance.get("dataset_hashes")
    hashes = hashes if isinstance(hashes, Mapping) else {}
    expected = hashes.get(path.name)
    if not expected or expected != file_sha256(path):
        raise ValueError(f"{label} hash does not match its D0 provenance")


def _load_jsonl_snapshot(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[list[dict[str, Any]], str]:
    """Read, hash, and parse one immutable JSONL input from the same bytes."""

    payload = Path(path).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != str(expected_sha256).strip().lower():
        raise ValueError(f"{label} changed before its validated snapshot")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{number} is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label}:{number} must be a JSON object")
        rows.append(dict(value))
    return rows, observed


def _validate_round1_source_capacity_binding(
    collection_manifest: Mapping[str, Any],
    recomputed_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the strict-collection capacity claim to match immutable rows."""

    recomputed = copy.deepcopy(dict(recomputed_report))
    if (
        recomputed.get("contract")
        != DAGGER1_ROUND1_SOURCE_CAPACITY_CONTRACT
        or recomputed.get("passed") is not True
    ):
        raise ValueError(
            "Recomputed Round-1 source capacity lacks the approved contract "
            "or did not pass"
        )
    recorded_value = collection_manifest.get("round1_replay_capacity")
    if not isinstance(recorded_value, Mapping):
        raise ValueError(
            "D1 collection manifest lacks round1_replay_capacity"
        )
    recorded = copy.deepcopy(dict(recorded_value))
    if recorded.get("contract") != DAGGER1_ROUND1_SOURCE_CAPACITY_CONTRACT:
        raise ValueError(
            "D1 collection manifest round1_replay_capacity uses a legacy or "
            "unapproved contract"
        )
    if recorded.get("passed") is not True:
        raise ValueError(
            "D1 collection manifest round1_replay_capacity did not pass"
        )
    if (
        recorded != recomputed
        or stable_json_sha256(recorded) != stable_json_sha256(recomputed)
    ):
        raise ValueError(
            "D1 collection manifest round1_replay_capacity differs from the "
            "immutable D0/natural-D1 source-capacity audit"
        )
    return recomputed


def validate_offline_teacher_target_quarantine_summary(
    value: Any,
    *,
    expected_total_rows: int | None = None,
    expected_candidate_rows: int | None = None,
) -> dict[str, Any]:
    """Validate the zero-quarantine collection claim and its row arithmetic."""

    if not isinstance(value, Mapping):
        raise ValueError("D1 manifest lacks an offline truth-audit summary")
    summary = dict(value)
    if (
        summary.get("contract")
        != OFFLINE_TEACHER_TARGET_QUARANTINE_SUMMARY_CONTRACT
    ):
        raise ValueError("D1 offline truth-audit summary contract mismatch")
    count_names = (
        "total_rows",
        "candidate_rows",
        "non_candidate_rows",
        "passed_rows",
        "quarantined_rows",
        "invalid_or_missing_audit_rows",
    )
    counts: dict[str, int] = {}
    for name in count_names:
        raw = summary.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(
                f"D1 offline truth-audit summary {name} must be nonnegative integer"
            )
        counts[name] = raw
    if (
        counts["candidate_rows"] + counts["non_candidate_rows"]
        != counts["total_rows"]
        or counts["passed_rows"] + counts["quarantined_rows"]
        != counts["candidate_rows"]
        or counts["invalid_or_missing_audit_rows"]
        > counts["quarantined_rows"]
    ):
        raise ValueError("D1 offline truth-audit summary counts are inconsistent")
    if expected_total_rows is not None and counts["total_rows"] != int(
        expected_total_rows
    ):
        raise ValueError("D1 offline truth-audit summary total row count mismatch")
    if expected_candidate_rows is not None and counts["candidate_rows"] != int(
        expected_candidate_rows
    ):
        raise ValueError("D1 offline truth-audit summary candidate count mismatch")
    empty_quarantine_metadata = bool(
        summary.get("quarantined_by_action_class") == {}
        and summary.get("quarantined_by_reason_code") == {}
        and summary.get("quarantined_example_ids") == []
    )
    if not (
        summary.get("passed") is True
        and summary.get("zero_truth_audit_quarantine") is True
        and counts["quarantined_rows"] == 0
        and counts["invalid_or_missing_audit_rows"] == 0
        and counts["candidate_rows"] == counts["passed_rows"]
        and empty_quarantine_metadata
    ):
        raise ValueError("D1 manifest does not prove zero truth-audit quarantine")
    return summary


def validate_dagger1_collection_selection_binding(
    selected_rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    collection_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and bind the published D1 subset to the complete ledger."""

    production_target = collection_manifest.get(
        "production_row_target_contract"
    )
    if not isinstance(production_target, Mapping):
        raise ValueError("D1 manifest lacks the production row-target contract")
    expected_target = collect_dagger1_module.dagger1_production_row_target_contract(
        target_min_rows=collect_dagger1_module.DEFAULT_TARGET_MIN_ROWS,
        target_max_rows=collect_dagger1_module.DEFAULT_TARGET_MAX_ROWS,
    )
    if dict(production_target) != expected_target:
        raise ValueError(
            "D1 collection does not bind the reviewed production row bounds "
            f"{collect_dagger1_module.DEFAULT_TARGET_MIN_ROWS}.."
            f"{collect_dagger1_module.DEFAULT_TARGET_MAX_ROWS}"
        )

    recorded_selection = collection_manifest.get(
        "deterministic_collection_selection"
    )
    if not isinstance(recorded_selection, Mapping):
        raise ValueError("D1 manifest lacks deterministic selection evidence")
    recorded_selection = dict(recorded_selection)
    if (
        recorded_selection.get("target_min_rows")
        != collect_dagger1_module.DEFAULT_TARGET_MIN_ROWS
        or recorded_selection.get("target_max_rows")
        != collect_dagger1_module.DEFAULT_TARGET_MAX_ROWS
    ):
        raise ValueError(
            "D1 deterministic selection does not use the reviewed production "
            "row bounds"
        )

    safe_candidates = [
        dict(row)
        for row in all_rows
        if row.get("production_label_eligible") is True
    ]
    recomputed_selected, recomputed_selection = (
        collect_dagger1_module.select_dagger1_collection_rows(
            safe_candidates,
            target_min_rows=collect_dagger1_module.DEFAULT_TARGET_MIN_ROWS,
            target_max_rows=collect_dagger1_module.DEFAULT_TARGET_MAX_ROWS,
        )
    )
    if recomputed_selection != recorded_selection:
        raise ValueError(
            "D1 deterministic selection report differs from the complete ledger"
        )
    if recomputed_selection.get("passed") is not True:
        raise ValueError("D1 deterministic production selection did not pass")
    materialized_selected = [dict(row) for row in selected_rows]
    if recomputed_selected != materialized_selected:
        raise ValueError(
            "D1 rows are not the exact deterministic selected subset of "
            "all-output"
        )

    candidate_count = len(safe_candidates)
    selected_count = len(materialized_selected)
    candidate_count_fields = (
        "candidate_recovery_rows",
        "candidate_recovery_row_count",
        "production_eligible_recovery_rows",
    )
    selected_count_fields = (
        "output_rows",
        "selected_recovery_row_count",
    )
    if any(
        collection_manifest.get(name) != candidate_count
        for name in candidate_count_fields
    ):
        raise ValueError(
            "D1 safe-candidate row counts do not match the collection manifest"
        )
    if any(
        collection_manifest.get(name) != selected_count
        for name in selected_count_fields
    ):
        raise ValueError(
            "D1 selected row counts do not match the collection manifest"
        )

    selected_ids = {
        str(row.get("example_id") or "") for row in materialized_selected
    }
    annotation_mismatches: list[str] = []
    for row in all_rows:
        example_id = str(row.get("example_id") or "")
        is_safe = row.get("production_label_eligible") is True
        is_selected = is_safe and example_id in selected_ids
        expected_disposition = (
            "selected_for_round1_training"
            if is_selected
            else (
                "safe_candidate_not_selected"
                if is_safe
                else "not_safe_candidate"
            )
        )
        raw_labels = row.get("labels")
        label_annotation_mismatch = bool(
            isinstance(raw_labels, Mapping)
            and (
                raw_labels.get("collection_training_eligible")
                is not is_selected
                or raw_labels.get("collection_disposition")
                != expected_disposition
            )
        )
        if (
            row.get("collection_training_eligible") is not is_selected
            or row.get("collection_disposition") != expected_disposition
            or label_annotation_mismatch
        ):
            annotation_mismatches.append(example_id)
    if annotation_mismatches:
        raise ValueError(
            "D1 all-output selection annotations are inconsistent: "
            + ", ".join(annotation_mismatches[:10])
        )

    return {
        "contract": DAGGER1_AGGREGATE_SELECTION_BINDING_CONTRACT,
        "production_row_target_contract": expected_target,
        "candidate_rows": candidate_count,
        "selected_rows": selected_count,
        "unselected_safe_candidate_rows": candidate_count - selected_count,
        "candidate_example_id_set_sha256": recomputed_selection[
            "candidate_example_id_set_sha256"
        ],
        "selected_example_id_sequence_sha256": recomputed_selection[
            "selected_example_id_sequence_sha256"
        ],
        "selection_report_sha256": stable_json_sha256(recomputed_selection),
        "exact_selected_row_content_match": True,
        "complete_ledger_annotation_match": True,
        "passed": True,
    }


def _bind_generation_id(row: Mapping[str, Any], provenance_id: str) -> dict[str, Any]:
    bound = dict(row)
    bound["generation_provenance_id"] = provenance_id
    metadata = bound.get("metadata")
    if isinstance(metadata, Mapping):
        bound["metadata"] = {**metadata, "generation_provenance_id": provenance_id}
    return bound


def generation_id_independent_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return canonical row content with aggregate generation IDs removed.

    Round-1 holdout rows inherit a D0 generation identity and are rebound to
    the new aggregate identity after the Round-1 descriptor is hashed.  This
    projection lets that otherwise-circular content remain descriptor-bound:
    the source gate verifies the projected content and the exact rebound ID.
    """

    projected: list[dict[str, Any]] = []
    for row in rows:
        item = copy.deepcopy(dict(row))
        item.pop("generation_provenance_id", None)
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            metadata = dict(metadata)
            metadata.pop("generation_provenance_id", None)
            item["metadata"] = metadata
        projected.append(item)
    return projected


def _recompute_final_view_support(
    *,
    raw_view: Sequence[Mapping[str, Any]],
    source_probe_rows: Sequence[Mapping[str, Any]],
    training_view_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind final support to the natural/probe rows actually placed."""

    reported = training_view_report.get("final_view_support")
    if not isinstance(reported, Mapping):
        raise RuntimeError(
            "three-source training-view report is missing the canonical "
            "final_view_support report"
        )
    if reported.get("contract") != FINAL_VIEW_SUPPORT_CONTRACT:
        raise RuntimeError(
            "three-source final_view_support report has the wrong contract"
        )

    placed_natural_d1 = [
        row
        for row in raw_view
        if row.get("replay_source") == "natural_dagger1"
    ]
    placed_probes = [
        row
        for row in raw_view
        if row.get("replay_source") == "observable_recovery_probe"
    ]
    recomputed = audit_dagger1_final_view_support(
        natural_rows=placed_natural_d1,
        probe_rows=placed_probes,
        source_probe_rows=source_probe_rows,
        policy=ROUND1_THREE_SOURCE_VIEW_POLICY,
    )
    if stable_json_sha256(reported) != stable_json_sha256(recomputed):
        raise RuntimeError(
            "three-source final_view_support report does not match the "
            "actual placed natural/probe rows"
        )
    if recomputed.get("passed") is not True:
        raise RuntimeError(
            "D0/D1/probe final placed-view support gate failed: "
            f"{recomputed}"
        )
    return copy.deepcopy(dict(recomputed))


def _source_hash_for_import_spec(spec: str) -> str:
    module_name, separator, attribute_path = spec.partition(":")
    if not separator:
        raise ValueError(f"invalid import spec {spec!r}")
    value: Any = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    source = inspect.getsourcefile(value)
    if source is None:
        raise ValueError(f"no source file for {spec}")
    return file_sha256(source)


def validate_round1_learner_seed(
    d1_manifest: Mapping[str, Any],
    *,
    collection_manifest_sha256: str,
    aggregate_learner_seed: Mapping[str, Any] | None = None,
    initial_adapter_revision: str | None = None,
) -> dict[str, str]:
    """Bind collection, aggregate, and warm-start to one adapter tree."""

    manifest_hash = str(collection_manifest_sha256).strip().lower()
    if _ADAPTER_TREE_REVISION.fullmatch(manifest_hash) is None:
        raise ValueError("D1 collection manifest SHA-256 must be exact 64-hex")

    learner_seed = d1_manifest.get("learner_seed")
    learner_seed = learner_seed if isinstance(learner_seed, Mapping) else {}
    adapter_revision = str(
        learner_seed.get("adapter_tree_sha256") or ""
    ).strip().lower()
    collection_revision = str(
        learner_seed.get("collection_model_revision") or ""
    ).strip().lower()
    top_level_revision = str(
        d1_manifest.get("model_revision") or ""
    ).strip().lower()
    collection_model_id = str(
        learner_seed.get("collection_model_id") or ""
    ).strip()
    top_level_model_id = str(d1_manifest.get("model_id") or "").strip()

    if (
        learner_seed.get("role") != "learner_seed_only"
        or _ADAPTER_TREE_REVISION.fullmatch(adapter_revision) is None
        or collection_revision != adapter_revision
        or top_level_revision != adapter_revision
    ):
        raise ValueError(
            "D1 collection learner_seed does not bind one exact 64-hex "
            "adapter tree revision"
        )
    if (
        not collection_model_id
        or not (
            Path(collection_model_id).is_absolute()
            or PurePosixPath(collection_model_id).is_absolute()
        )
        or top_level_model_id != collection_model_id
    ):
        raise ValueError(
            "D1 collection learner_seed does not bind its absolute local "
            "adapter model ID"
        )

    binding = {
        "role": "learner_seed_only",
        "collection_model_id": collection_model_id,
        "adapter_tree_sha256": adapter_revision,
        "collection_model_revision": collection_revision,
        "collection_manifest_sha256": manifest_hash,
    }

    if aggregate_learner_seed is not None:
        if not isinstance(aggregate_learner_seed, Mapping):
            raise ValueError("Round-1 aggregate learner_seed must be an object")
        observed = {
            key: str(aggregate_learner_seed.get(key) or "").strip().lower()
            if key != "collection_model_id"
            else str(aggregate_learner_seed.get(key) or "").strip()
            for key in binding
        }
        if observed != binding:
            raise ValueError(
                "Round-1 aggregate learner_seed differs from its D1 "
                "collection manifest"
            )

    if initial_adapter_revision is not None:
        initial_revision = str(initial_adapter_revision).strip().lower()
        if (
            _ADAPTER_TREE_REVISION.fullmatch(initial_revision) is None
            or initial_revision != adapter_revision
        ):
            raise ValueError(
                "INITIAL_ADAPTER_REVISION differs from the D1 learner_seed "
                "adapter tree revision"
            )
    return binding


def build_round1_aggregate(
    *,
    d0_aggregate_dir: Path,
    d1_path: Path,
    d1_manifest_path: Path,
    probe_path: Path,
    probe_manifest_path: Path,
    reviewed_source_commit: str,
    output_dir: Path,
    seed: int = 20260719,
    size: int | None = None,
    d1_share: float = 0.25,
    minimum_d1_share: float = 0.20,
    maximum_d1_share: float = 0.30,
    max_duplicate_count: int = 2,
    max_rows_per_root: int = 8,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError("Round-1 aggregate requires a clean committed source tree")
    reviewed = str(reviewed_source_commit).strip().lower()
    if not reviewed or reviewed != source_state.get("source_commit"):
        raise RuntimeError(
            "Round-1 aggregate reviewed source commit differs from clean HEAD"
        )
    legacy_defaults = {
        "seed": seed == 20260719,
        "size": size is None,
        "d1_share": d1_share == 0.25,
        "minimum_d1_share": minimum_d1_share == 0.20,
        "maximum_d1_share": maximum_d1_share == 0.30,
        "max_duplicate_count": max_duplicate_count == 2,
        "max_rows_per_root": max_rows_per_root == 8,
    }
    if not all(legacy_defaults.values()):
        changed = sorted(name for name, valid in legacy_defaults.items() if not valid)
        raise ValueError(
            "Round-1 release view allocation is frozen; unsupported legacy "
            "override(s): " + ", ".join(changed)
        )
    _preflight_round1_output_directory(output_dir)

    d0_raw_path = d0_aggregate_dir / "aggregate.raw.jsonl"
    validation_path = d0_aggregate_dir / "aggregate.validation.jsonl"
    test_path = d0_aggregate_dir / "aggregate.test.jsonl"
    d0_provenance_path = d0_aggregate_dir / "aggregate.generation_provenance.json"
    d0_manifest_path = d0_aggregate_dir / AGGREGATE_MANIFEST_FILENAME
    for path in (
        d0_raw_path,
        validation_path,
        test_path,
        d0_provenance_path,
        d0_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    d0_provenance = _load_mapping(d0_provenance_path)
    if d0_provenance.get("release_eligible") is not True:
        raise ValueError("D0 aggregate generation provenance is not release eligible")
    d0_descriptor = d0_provenance.get("generation_descriptor")
    if not isinstance(d0_descriptor, Mapping):
        raise ValueError("D0 aggregate lacks a generation descriptor")
    d0_descriptor = dict(d0_descriptor)
    d0_source = d0_descriptor.get("source_state")
    d0_source = d0_source if isinstance(d0_source, Mapping) else {}
    if d0_source.get("release_eligible_source") is not True:
        raise ValueError("D0 aggregate source state is not release eligible")
    if d0_source.get("source_commit") != source_state.get("source_commit"):
        raise ValueError("D0 aggregate source commit does not match current source")
    d0_provenance_id = stable_json_sha256(d0_descriptor)
    if d0_provenance.get("generation_provenance_id") != d0_provenance_id:
        raise ValueError(
            "D0 aggregate generation provenance ID does not match its "
            "generation descriptor"
        )
    for path, label in (
        (d0_raw_path, "D0 raw aggregate"),
        (validation_path, "D0 validation split"),
        (test_path, "D0 test split"),
    ):
        _verify_recorded_hash(provenance=d0_provenance, path=path, label=label)
    d0_manifest_binding = validate_aggregate_manifest_binding(
        d0_provenance,
        aggregate_dir=d0_aggregate_dir,
    )
    if d0_manifest_binding["passed"] is not True:
        raise ValueError(
            "D0 aggregate manifest does not match its provenance: "
            + "; ".join(d0_manifest_binding["failures"])
        )

    d1_manifest = _load_mapping(d1_manifest_path)
    d1_manifest_sha256 = file_sha256(d1_manifest_path)
    if d1_manifest.get("d0_manifest_sha256") != file_sha256(d0_manifest_path):
        raise ValueError(
            "D1 collection manifest is bound to a different D0 aggregate "
            "manifest"
        )
    if d1_manifest.get("release_evidence_eligible") is not False:
        raise ValueError("D1 manifest must explicitly reject promotion-evidence use")
    if d1_manifest.get("training_eligible") is not True:
        raise ValueError("D1 collection manifest is not training eligible")
    expected_execution_pipeline = validate_dagger1_execution_pipeline_contract(
        d1_manifest
    )
    if d1_manifest.get("output_sha256") != file_sha256(d1_path):
        raise ValueError("D1 rows do not match the collection manifest hash")
    learner_seed_binding = validate_round1_learner_seed(
        d1_manifest,
        collection_manifest_sha256=d1_manifest_sha256,
    )
    if (
        d1_manifest.get("scenario_builder_contract")
        != DAGGER1_SCENARIO_BUILDER_CONTRACT
    ):
        raise ValueError("D1 collection lacks the reviewed scenario-builder binding")
    scenario_manifest_value = str(
        d1_manifest.get("scenario_manifest") or ""
    ).strip()
    if not scenario_manifest_value:
        raise ValueError("D1 collection does not identify its scenario manifest")
    scenario_manifest_path = Path(scenario_manifest_value)
    if not scenario_manifest_path.is_absolute():
        scenario_manifest_path = repo_root / scenario_manifest_path
    if (
        not scenario_manifest_path.is_file()
        or d1_manifest.get("scenario_manifest_sha256")
        != file_sha256(scenario_manifest_path)
    ):
        raise ValueError("D1 scenario-builder manifest hash is unavailable or changed")
    scenario_input_value = str(d1_manifest.get("input") or "").strip()
    scenario_input_path = Path(scenario_input_value)
    if not scenario_input_path.is_absolute():
        scenario_input_path = repo_root / scenario_input_path
    if (
        not scenario_input_value
        or not scenario_input_path.is_file()
        or d1_manifest.get("input_sha256") != file_sha256(scenario_input_path)
    ):
        raise ValueError("D1 scenario input hash is unavailable or changed")
    scenario_report_value = str(
        d1_manifest.get("scenario_generator_report") or ""
    ).strip()
    scenario_report_path = Path(scenario_report_value)
    if not scenario_report_path.is_absolute():
        scenario_report_path = repo_root / scenario_report_path
    if (
        not scenario_report_value
        or not scenario_report_path.is_file()
        or d1_manifest.get("scenario_generator_report_sha256")
        != file_sha256(scenario_report_path)
    ):
        raise ValueError(
            "D1 scenario generator-report hash is unavailable or changed"
        )
    d1_source = d1_manifest.get("source_state")
    d1_source = d1_source if isinstance(d1_source, Mapping) else {}
    if (
        d1_source.get("release_eligible_source") is not True
        or d1_source.get("source_commit") != source_state.get("source_commit")
    ):
        raise ValueError("D1 collection source does not match current clean source")

    development_holdout_value = str(
        d1_manifest.get("development_holdout") or ""
    ).strip()
    development_manifest_value = str(
        d1_manifest.get("development_holdout_manifest") or ""
    ).strip()
    development_generator_report_value = str(
        d1_manifest.get("development_holdout_generator_report") or ""
    ).strip()
    development_holdout_path = Path(development_holdout_value)
    development_manifest_path = Path(development_manifest_value)
    development_generator_report_path = Path(
        development_generator_report_value
    )
    if (
        not development_holdout_value
        or not development_manifest_value
        or not development_generator_report_value
        or not development_holdout_path.is_absolute()
        or not development_manifest_path.is_absolute()
        or not development_generator_report_path.is_absolute()
    ):
        raise ValueError(
            "D1 training manifest must bind absolute development holdout, "
            "manifest, and generator-report paths"
        )
    if not development_holdout_path.is_file():
        raise FileNotFoundError(development_holdout_path)
    if not development_manifest_path.is_file():
        raise FileNotFoundError(development_manifest_path)
    if not development_generator_report_path.is_file():
        raise FileNotFoundError(development_generator_report_path)
    development_holdout_sha256 = file_sha256(development_holdout_path)
    development_manifest_sha256 = file_sha256(development_manifest_path)
    development_generator_report_sha256 = file_sha256(
        development_generator_report_path
    )
    if (
        d1_manifest.get("development_holdout_sha256")
        != development_holdout_sha256
        or d1_manifest.get("development_holdout_manifest_sha256")
        != development_manifest_sha256
        or d1_manifest.get("development_holdout_generator_report_sha256")
        != development_generator_report_sha256
    ):
        raise ValueError(
            "D1 development holdout bytes, manifest bytes, or "
            "generator-report bytes do not match the collection manifest"
        )
    development_manifest = _load_mapping(development_manifest_path)
    if (
        development_manifest.get("generator_report_sha256")
        != development_generator_report_sha256
    ):
        raise ValueError(
            "D1 development generator report does not match the development "
            "manifest"
        )
    development_roots = validate_development_holdout_binding(
        development_holdout_path,
        development_manifest_path,
        generator_report_path=development_generator_report_path,
        source_state=source_state,
        scenario_input_path=scenario_input_path,
        scenario_manifest_path=scenario_manifest_path,
        d0_raw_path=d0_raw_path,
        d0_provenance_path=d0_provenance_path,
        d0_manifest_path=d0_manifest_path,
        forbidden_suite_path=DEFAULT_FORBIDDEN_SUITE,
        evaluation_policy_path=DEFAULT_EVALUATION_POLICY,
        require_model_selection_eligible=True,
    )
    development_root_set_sha256 = stable_json_sha256(
        sorted(development_roots)
    )
    if (
        d1_manifest.get("development_holdout_root_count")
        != len(development_roots)
        or d1_manifest.get("development_physical_root_count")
        != len(development_roots)
        or d1_manifest.get("development_holdout_root_set_sha256")
        != development_root_set_sha256
    ):
        raise ValueError(
            "D1 development holdout root binding differs from collection"
        )
    development_holdout_binding = {
        "holdout_sha256": development_holdout_sha256,
        "manifest_sha256": development_manifest_sha256,
        "generator_report_sha256": development_generator_report_sha256,
        "physical_root_count": len(development_roots),
        "root_set_sha256": development_root_set_sha256,
    }
    identities = d1_manifest.get("factory_identities")
    identities = identities if isinstance(identities, Mapping) else {}
    expected_factory_bindings = {
        "environment": (
            DEFAULT_ENV_FACTORY_SPEC,
            _source_hash_for_import_spec(DEFAULT_ENV_FACTORY_SPEC),
        ),
        "learner_policy": (
            DEFAULT_POLICY_FACTORY_SPEC,
            _source_hash_for_import_spec(DEFAULT_POLICY_FACTORY_SPEC),
        ),
    }
    for role, (spec, source_hash) in expected_factory_bindings.items():
        binding = identities.get(role)
        binding = binding if isinstance(binding, Mapping) else {}
        if (
            binding.get("import_spec") != spec
            or binding.get("source_sha256") != source_hash
        ):
            raise ValueError(f"D1 {role} factory identity does not match source")
    expert_binding = identities.get("expert_oracle")
    expert_binding = expert_binding if isinstance(expert_binding, Mapping) else {}
    expert_source = inspect.getsourcefile(ExpertPolicyOracle)
    if (
        expert_source is None
        or expert_binding.get("source_sha256") != file_sha256(expert_source)
    ):
        raise ValueError("D1 expert oracle identity does not match source")
    release_contract = d1_manifest.get("release_environment_contract")
    release_contract = (
        release_contract if isinstance(release_contract, Mapping) else {}
    )
    if (
        release_contract.get("parameter_ranking_dominance_threshold")
        != BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        or release_contract.get("production_dataset_mode") is not True
        or release_contract.get("max_steps") != 24
    ):
        raise ValueError("D1 release environment contract is not approved")
    evaluation_policy = _load_mapping(DEFAULT_EVALUATION_POLICY)
    suite_policy = evaluation_policy.get("suite_policy")
    suite_policy = suite_policy if isinstance(suite_policy, Mapping) else {}
    current_suite_hash = file_sha256(DEFAULT_FORBIDDEN_SUITE)
    if (
        suite_policy.get("status") != "pinned"
        or suite_policy.get("approved_suite_sha256") != current_suite_hash
        or d1_manifest.get("forbidden_suite_sha256") != current_suite_hash
        or d1_manifest.get("evaluation_policy_sha256")
        != file_sha256(DEFAULT_EVALUATION_POLICY)
    ):
        raise ValueError("D1 frozen-suite holdout binding is not pinned/current")

    d0_dataset_hashes = d0_provenance.get("dataset_hashes")
    d0_dataset_hashes = (
        d0_dataset_hashes if isinstance(d0_dataset_hashes, Mapping) else {}
    )
    raw_d0, d0_raw_snapshot_sha256 = _load_jsonl_snapshot(
        d0_raw_path,
        expected_sha256=str(d0_dataset_hashes.get(d0_raw_path.name) or ""),
        label="D0 raw aggregate",
    )
    d0_train = [
        row
        for row in raw_d0
        if row.get("dataset_split") == "train"
        and row.get("production_label_eligible") is True
    ]
    d1, d1_snapshot_sha256 = _load_jsonl_snapshot(
        d1_path,
        expected_sha256=str(d1_manifest.get("output_sha256") or ""),
        label="D1 selected rows",
    )
    round1_source_capacity = _validate_round1_source_capacity_binding(
        d1_manifest,
        dagger1_round1_source_capacity_report(
            d0_train,
            d1,
            policy=ROUND1_THREE_SOURCE_VIEW_POLICY,
        ),
    )
    round1_source_capacity_sha256 = stable_json_sha256(
        round1_source_capacity
    )
    # Treat the collection manifest as an integrity binding, not as authority
    # for safety properties that are cheap to recompute.  A forged manifest
    # must not be able to reintroduce oracle truth or a frozen evaluation root
    # at the final D0+D1 ingestion boundary.
    validate_export_rows_truth_free(d1)
    for index, row in enumerate(d1):
        try:
            row["offline_teacher_target_audit"] = (
                validate_offline_teacher_target_audit_metadata(
                    row.get("offline_teacher_target_audit"),
                    require_passed=True,
                )
            )
        except ValueError as exc:
            identifier = row.get("example_id") or f"row_{index}"
            raise ValueError(
                "Immutable D1 input row has invalid offline teacher-target "
                f"audit: {identifier}"
            ) from exc
    if d1_manifest.get("collection_pass") != "training":
        raise ValueError("D1 aggregate input must come from the training pass")
    all_output_value = str(d1_manifest.get("all_output") or "").strip()
    all_output_path = Path(all_output_value)
    if not all_output_value or not all_output_path.is_absolute():
        raise ValueError(
            "D1 training manifest must bind an absolute --all-output path"
        )
    if not all_output_path.is_file():
        raise FileNotFoundError(all_output_path)
    if d1_manifest.get("all_output_sha256") != file_sha256(all_output_path):
        raise ValueError("D1 all-output ledger hash does not match its manifest")
    all_d1_rows, all_output_snapshot_sha256 = _load_jsonl_snapshot(
        all_output_path,
        expected_sha256=str(d1_manifest.get("all_output_sha256") or ""),
        label="D1 all-output ledger",
    )
    all_output_row_count = d1_manifest.get("all_output_row_count")
    if (
        isinstance(all_output_row_count, bool)
        or not isinstance(all_output_row_count, int)
        or all_output_row_count != len(all_d1_rows)
        or d1_manifest.get("visited_rows") != len(all_d1_rows)
    ):
        raise ValueError("D1 all-output ledger row count does not match manifest")
    validate_export_rows_truth_free(all_d1_rows)
    for index, row in enumerate(all_d1_rows):
        try:
            row["offline_teacher_target_audit"] = (
                validate_offline_teacher_target_audit_metadata(
                    row.get("offline_teacher_target_audit")
                )
            )
        except ValueError as exc:
            identifier = row.get("example_id") or f"all_row_{index}"
            raise ValueError(
                "D1 all-output row has invalid offline teacher-target "
                f"audit: {identifier}"
            ) from exc
    safe_candidates_from_all_output = [
        row
        for row in all_d1_rows
        if row.get("production_label_eligible") is True
    ]
    recorded_quarantine_summary = (
        validate_offline_teacher_target_quarantine_summary(
            d1_manifest.get("offline_teacher_target_quarantine_summary"),
            expected_total_rows=len(all_d1_rows),
            expected_candidate_rows=len(safe_candidates_from_all_output),
        )
    )
    recomputed_quarantine_summary = (
        summarize_dagger1_offline_teacher_target_quarantine(all_d1_rows)
    )
    if recomputed_quarantine_summary != recorded_quarantine_summary:
        raise ValueError(
            "D1 offline truth-audit summary differs from the all-output ledger"
        )
    d1_selection_binding = validate_dagger1_collection_selection_binding(
        d1,
        all_d1_rows,
        d1_manifest,
    )
    all_d0_roots = {
        str(row.get("physical_root_fingerprint"))
        for row in raw_d0
        if row.get("physical_root_fingerprint")
    }
    d1_roots = {
        str(row.get("physical_root_fingerprint"))
        for row in d1
        if row.get("physical_root_fingerprint")
    }
    all_output_d1_roots = {
        str(row.get("physical_root_fingerprint"))
        for row in all_d1_rows
        if row.get("physical_root_fingerprint")
    }
    observed_d1_roots = d1_roots | all_output_d1_roots
    leaked = sorted(all_d0_roots & observed_d1_roots)
    if leaked:
        raise ValueError(
            "D1 roots overlap the D0 aggregate: " + ", ".join(leaked)
        )
    frozen_roots = frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE)
    frozen_leaked = sorted(frozen_roots & observed_d1_roots)
    if frozen_leaked:
        raise ValueError(
            "D1 roots overlap the frozen evaluation suite: "
            + ", ".join(frozen_leaked)
        )
    development_protected_leaked = sorted(
        development_roots & (all_d0_roots | set(frozen_roots))
    )
    if development_protected_leaked:
        raise ValueError(
            "Development holdout roots overlap D0/frozen roots: "
            + ", ".join(development_protected_leaked)
        )
    scenario_rows = collect_dagger1_module._load_json_or_jsonl(
        scenario_input_path
    )
    collect_dagger1_module.validate_training_scenarios(
        scenario_rows,
        forbidden_roots=frozenset(
            all_d0_roots | set(frozen_roots) | set(development_roots)
        ),
    )
    development_leaked = sorted(development_roots & observed_d1_roots)
    if development_leaked:
        raise ValueError(
            "D1 roots overlap the development holdout: "
            + ", ".join(development_leaked)
        )

    probe_path = Path(probe_path)
    probe_manifest_path = Path(probe_manifest_path)
    if not probe_path.is_file():
        raise FileNotFoundError(probe_path)
    if not probe_manifest_path.is_file():
        raise FileNotFoundError(probe_manifest_path)
    probe_binding = validate_recovery_probe_suite_binding(
        rows_path=probe_path,
        manifest_path=probe_manifest_path,
        scenarios_path=scenario_input_path,
        scenario_manifest_path=scenario_manifest_path,
        scenario_generator_report_path=scenario_report_path,
        development_holdout=development_holdout_path,
        development_holdout_manifest=development_manifest_path,
        development_holdout_generator_report=development_generator_report_path,
        d0_aggregate_dir=d0_aggregate_dir,
        forbidden_suite=DEFAULT_FORBIDDEN_SUITE,
        evaluation_policy=DEFAULT_EVALUATION_POLICY,
        reviewed_source_commit=reviewed,
    )
    probes, probe_snapshot_sha256 = _load_jsonl_snapshot(
        probe_path,
        expected_sha256=str(probe_binding.get("rows_sha256") or ""),
        label="recovery probe rows",
    )
    probe_roots = {
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in probes
        if str(row.get("physical_root_fingerprint") or "").strip()
    }
    actual_natural_probe_overlap = sorted(d1_roots & probe_roots)

    # Source identity is explicit in the materialized immutable views.  These
    # markers are recomputed here rather than trusted from D0/D1 inputs.
    d0_train = [
        {**dict(row), "replay_source": "d0_bc0"} for row in d0_train
    ]
    d1 = [
        {**dict(row), "replay_source": "natural_dagger1"} for row in d1
    ]

    # Recompute semantic eligibility from immutable D1 rows.  The manifest
    # binds bytes and collection identity, but it is never authority for
    # properties that the final builder can independently verify.
    d1_recovery_audit = audit_dagger1_recovery_labels(d1)
    d1_class_audit = audit_target_aware_state_classes(d1)
    training_support = audit_dagger1_training_support(d1, probes)
    d1_root_support = training_support["natural_on_policy_support"]
    recomputed_d1_audits = {
        "round1_source_capacity": round1_source_capacity,
        "offline_teacher_target_quarantine_summary": (
            recomputed_quarantine_summary
        ),
        "recovery_label_audit": d1_recovery_audit,
        "target_aware_state_class_audit": d1_class_audit,
        "independent_root_support": d1_root_support,
        "three_source_training_support": training_support,
        "deterministic_collection_selection_binding": d1_selection_binding,
    }
    failed_d1_audits = [
        name
        for name, report in recomputed_d1_audits.items()
        if report.get("passed") is not True
    ]
    if failed_d1_audits:
        raise RuntimeError(
            "D1 final-ingestion semantic audits failed: "
            + ", ".join(failed_d1_audits)
        )

    raw_view, training_view_report = build_dagger1_three_source_view(
        d0_rows=d0_train,
        natural_d1_rows=d1,
        probe_rows=probes,
        policy=ROUND1_THREE_SOURCE_VIEW_POLICY,
    )
    if training_view_report.get("passed") is not True:
        raise RuntimeError(
            "D0/D1/probe three-source training-view gate failed: "
            f"{training_view_report}"
        )
    final_view_support = _recompute_final_view_support(
        raw_view=raw_view,
        source_probe_rows=probes,
        training_view_report=training_view_report,
    )
    natural_only_raw_view, natural_only_view_report = (
        build_round1_natural_only_view(raw_view)
    )
    if natural_only_view_report.get("passed") is not True:
        raise RuntimeError(
            "Natural-D1-only ordered projection failed: "
            f"{natural_only_view_report}"
        )

    natural_rows = [*d0_train, *d1, *probes]
    semantic_realizability = audit_dagger1_union_realizability(
        natural_rows,
        raw_view,
    )
    if semantic_realizability.get("passed") is not True:
        raise RuntimeError(
            "D0/D1/probe semantic realizability failed: "
            + "; ".join(semantic_realizability.get("failures") or [])
        )

    audit_reports: dict[str, Mapping[str, Any]] = {
        "round1_source_capacity": round1_source_capacity,
        "d1_offline_teacher_target_quarantine_summary": (
            recomputed_quarantine_summary
        ),
        "d1_recovery_label_audit": d1_recovery_audit,
        "d1_target_aware_state_class_audit": d1_class_audit,
        "d1_independent_root_support": d1_root_support,
        "d1_deterministic_collection_selection_binding": (
            d1_selection_binding
        ),
        "d1_three_source_training_support": training_support,
        "final_view_support": final_view_support,
        "natural_only_view": natural_only_view_report,
        "union_realizability": semantic_realizability,
    }
    for key in (
        "natural_teacher_realizability",
        "training_view_teacher_realizability",
        "natural_approximate_teacher_realizability",
        "training_view_approximate_teacher_realizability",
        "approximate_teacher_realizability_by_scenario_family",
        "approximate_teacher_realizability_by_state_class",
        "approximate_teacher_realizability_by_recovery_stratum",
    ):
        report = semantic_realizability.get(key)
        if isinstance(report, Mapping):
            audit_reports[f"union_{key}"] = report
    audit_report_sha256 = {
        name: stable_json_sha256(report)
        for name, report in sorted(audit_reports.items())
    }

    validation_rows, validation_snapshot_sha256 = _load_jsonl_snapshot(
        validation_path,
        expected_sha256=str(
            d0_dataset_hashes.get(validation_path.name) or ""
        ),
        label="D0 validation split",
    )
    test_rows, test_snapshot_sha256 = _load_jsonl_snapshot(
        test_path,
        expected_sha256=str(d0_dataset_hashes.get(test_path.name) or ""),
        label="D0 test split",
    )
    tentative_train = examples_to_chat_sft(
        raw_view,
        protocol="canonical",
        allow_ineligible_auxiliary=True,
    )
    tentative_natural_only_train = examples_to_chat_sft(
        natural_only_raw_view,
        protocol="canonical",
    )
    schema_hashes = tool_schema_hashes(
        [
            *tentative_train,
            *tentative_natural_only_train,
            *validation_rows,
            *test_rows,
        ]
    )
    if len(schema_hashes) != 1:
        raise ValueError(f"D0/D1/probe tool schema mismatch: {schema_hashes}")

    source_files = (
        Path(__file__),
        Path(collect_dagger1_module.__file__),
        Path(dataset_builder_module.__file__),
        Path(semantic_audit_module.__file__),
        Path(offline_audit_module.__file__),
        Path(replay_buffer_module.__file__),
        Path(rollout_collector_module.__file__),
        Path(three_source_view_module.__file__),
        Path(natural_only_view_module.__file__),
        Path(recovery_probe_builder_module.__file__),
    )
    generation_descriptor = {
        "generation_provenance_version": 1,
        "builder_contract": ROUND1_AGGREGATE_BUILDER_CONTRACT,
        "source_state": source_state,
        "protocol": "canonical",
        "schema_registry_hash": schema_hashes[0],
        "generator_hashes": {
            path.resolve().relative_to(repo_root).as_posix(): file_sha256(path)
            for path in source_files
        },
        "input_artifacts": {
            "d0_generation_provenance_id": d0_provenance.get(
                "generation_provenance_id"
            ),
            "d0_generation_provenance_sha256": file_sha256(d0_provenance_path),
            "d0_manifest_sha256": file_sha256(d0_manifest_path),
            "d0_raw_sha256": d0_raw_snapshot_sha256,
            "d0_validation_sha256": validation_snapshot_sha256,
            "d0_test_sha256": test_snapshot_sha256,
            "d1_rows_sha256": d1_snapshot_sha256,
            "d1_all_output_sha256": all_output_snapshot_sha256,
            "d1_all_output_row_count": len(all_d1_rows),
            "d1_safe_candidate_row_count": d1_selection_binding[
                "candidate_rows"
            ],
            "d1_selected_row_count": d1_selection_binding["selected_rows"],
            "d1_unselected_safe_candidate_row_count": d1_selection_binding[
                "unselected_safe_candidate_rows"
            ],
            "d1_selection_binding_sha256": stable_json_sha256(
                d1_selection_binding
            ),
            "d1_manifest_sha256": d1_manifest_sha256,
            "d1_manifest_content_sha256": stable_json_sha256(d1_manifest),
            "d1_execution_pipeline_contract": expected_execution_pipeline,
            "d1_execution_pipeline_contract_sha256": stable_json_sha256(
                expected_execution_pipeline
            ),
            "d1_development_holdout": development_holdout_binding,
            "probe_rows_sha256": probe_snapshot_sha256,
            "probe_manifest_sha256": probe_binding["manifest_sha256"],
            "probe_generation_provenance_id": probe_binding[
                "generation_provenance_id"
            ],
            "probe_binding_report_sha256": stable_json_sha256(probe_binding),
            "immutable_source_view_content_sha256": {
                "d0_bc0": stable_json_sha256(d0_train),
                "natural_dagger1": stable_json_sha256(d1),
                "observable_recovery_probe": stable_json_sha256(probes),
            },
            "immutable_holdout_content_sha256": {
                "validation": stable_json_sha256(
                    generation_id_independent_rows(validation_rows)
                ),
                "test": stable_json_sha256(
                    generation_id_independent_rows(test_rows)
                ),
            },
            "immutable_derived_view_content_sha256": {
                "natural_only_raw": stable_json_sha256(
                    generation_id_independent_rows(natural_only_raw_view)
                ),
                "natural_only_chat": stable_json_sha256(
                    generation_id_independent_rows(
                        tentative_natural_only_train
                    )
                ),
            },
        },
        "learner_seed": learner_seed_binding,
        "training_view_contract": THREE_SOURCE_VIEW_CONTRACT,
        "round1_view_policy_digest": round1_view_policy_digest(),
        "round1_view_policy": ROUND1_THREE_SOURCE_VIEW_POLICY,
        "natural_only_view_contract": NATURAL_ONLY_VIEW_BUILD_CONTRACT,
        "natural_only_view_policy_digest": (
            round1_natural_only_view_policy_digest()
        ),
        "natural_only_view_policy": ROUND1_NATURAL_ONLY_VIEW_POLICY,
        "natural_only_view_report_sha256": stable_json_sha256(
            natural_only_view_report
        ),
        "training_view_report_sha256": stable_json_sha256(training_view_report),
        "training_support_report_sha256": stable_json_sha256(training_support),
        "final_view_support_report_sha256": stable_json_sha256(
            final_view_support
        ),
        "round1_source_capacity_report_sha256": (
            round1_source_capacity_sha256
        ),
        "audit_report_sha256": audit_report_sha256,
        "generation_config": {
            "policy_digest": round1_view_policy_digest(),
            "allocation": dict(ROUND1_THREE_SOURCE_VIEW_POLICY["allocation"]),
            "global_caps": dict(ROUND1_THREE_SOURCE_VIEW_POLICY["global_caps"]),
            "natural_only_policy_digest": (
                round1_natural_only_view_policy_digest()
            ),
            "natural_only_allocation": dict(
                ROUND1_NATURAL_ONLY_VIEW_POLICY["allocation"]
            ),
        },
    }
    provenance_id = stable_json_sha256(generation_descriptor)
    for row in raw_view:
        row["generation_provenance_id"] = provenance_id
    for row in natural_only_raw_view:
        row["generation_provenance_id"] = provenance_id
    train_rows = examples_to_chat_sft(
        raw_view,
        protocol="canonical",
        allow_ineligible_auxiliary=True,
    )
    natural_only_train_rows = examples_to_chat_sft(
        natural_only_raw_view,
        protocol="canonical",
    )
    validation_rows = [
        _bind_generation_id(row, provenance_id) for row in validation_rows
    ]
    test_rows = [_bind_generation_id(row, provenance_id) for row in test_rows]

    release_checks = {
        "current_clean_source": source_state.get("release_eligible_source") is True,
        "d0_release_eligible_source": d0_source.get("release_eligible_source")
        is True,
        "d0_current_source": d0_source.get("source_commit")
        == source_state.get("source_commit"),
        "d0_generation_provenance_id_verified": (
            d0_provenance.get("generation_provenance_id") == d0_provenance_id
        ),
        "d1_current_source": d1_source.get("source_commit")
        == source_state.get("source_commit"),
        "d1_development_holdout_bound": not bool(
            development_roots & observed_d1_roots
        ),
        "d1_training_eligible": d1_manifest.get("training_eligible") is True,
        "d1_execution_pipeline_approved": (
            d1_manifest.get("execution_pipeline_contract")
            == expected_execution_pipeline
        ),
        "round1_source_capacity": (
            round1_source_capacity.get("contract")
            == DAGGER1_ROUND1_SOURCE_CAPACITY_CONTRACT
            and round1_source_capacity.get("passed") is True
            and stable_json_sha256(d1_manifest.get("round1_replay_capacity"))
            == round1_source_capacity_sha256
        ),
        "d1_zero_truth_audit_quarantine": (
            recomputed_quarantine_summary.get("passed") is True
        ),
        "d1_deterministic_selection_recomputed": (
            d1_selection_binding.get("passed") is True
        ),
        "training_view_release_ready": training_view_report.get("passed") is True,
        "natural_only_view_release_ready": (
            natural_only_view_report.get("passed") is True
            and natural_only_view_report.get("reselection_performed") is False
            and natural_only_view_report.get("retained_allocation")
            == ROUND1_NATURAL_ONLY_VIEW_POLICY["allocation"]
        ),
        "source_mix_passed": training_view_report.get("placed")
        == ROUND1_THREE_SOURCE_VIEW_POLICY["allocation"],
        "round1_view_policy_bound": training_view_report.get("policy_digest")
        == round1_view_policy_digest(),
        "probe_binding_verified": probe_binding.get("passed") is True,
        "three_source_training_support": training_support.get("passed") is True,
        "final_view_support": final_view_support.get("passed") is True,
        "d1_recovery_labels_recomputed": d1_recovery_audit.get("passed") is True,
        "d1_state_classes_recomputed": d1_class_audit.get("passed") is True,
        "d1_independent_root_support": d1_root_support.get("passed") is True,
        "union_semantic_realizability": semantic_realizability.get("passed")
        is True,
    }
    release_eligible = all(release_checks.values())
    if not release_eligible:
        raise RuntimeError(f"Round-1 aggregate release checks failed: {release_checks}")
    preflight = {
        "release_eligible": release_eligible,
        "release_checks": release_checks,
        "generation_provenance_id": provenance_id,
        "training_view": training_view_report,
        "natural_only_view": natural_only_view_report,
        "natural_only_view_policy": ROUND1_NATURAL_ONLY_VIEW_POLICY,
        "round1_view_policy": ROUND1_THREE_SOURCE_VIEW_POLICY,
        "probe_binding": probe_binding,
        "actual_natural_probe_root_overlap": actual_natural_probe_overlap,
        "three_source_training_support": training_support,
        "final_view_support": final_view_support,
        "round1_source_capacity": round1_source_capacity,
        "round1_source_capacity_report_sha256": (
            round1_source_capacity_sha256
        ),
        "recomputed_d1_audits": recomputed_d1_audits,
        "d1_collection_selection_binding": d1_selection_binding,
        "semantic_realizability": semantic_realizability,
        "audit_report_sha256": audit_report_sha256,
        "split_rows": {
            "train_view": len(train_rows),
            "natural_only_train_view": len(natural_only_train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "d1_collection_manifest": d1_manifest,
        "d1_development_holdout": development_holdout_binding,
    }

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        staged_paths = {
            name: staging_dir / name for name in _ROUND1_OUTPUT_FILENAMES
        }
        output_paths = {
            name: staged_paths[name]
            for name in _ROUND1_OUTPUT_FILENAMES
            if name.endswith(".jsonl")
        }
        write_jsonl(output_paths["aggregate.raw.jsonl"], natural_rows)
        write_jsonl(output_paths["aggregate.d0.raw.jsonl"], d0_train)
        write_jsonl(output_paths["aggregate.d1.raw.jsonl"], d1)
        write_jsonl(output_paths["aggregate.probe.raw.jsonl"], probes)
        write_jsonl(output_paths["aggregate.train_view.raw.jsonl"], raw_view)
        write_jsonl(output_paths["aggregate.train_view.jsonl"], train_rows)
        write_jsonl(
            output_paths["aggregate.natural_only.train_view.raw.jsonl"],
            natural_only_raw_view,
        )
        write_jsonl(
            output_paths["aggregate.natural_only.train_view.jsonl"],
            natural_only_train_rows,
        )
        write_jsonl(output_paths["aggregate.validation.jsonl"], validation_rows)
        write_jsonl(output_paths["aggregate.test.jsonl"], test_rows)

        dataset_hashes = {
            name: file_sha256(path)
            for name, path in sorted(output_paths.items())
        }
        provenance = {
            **generation_descriptor,
            "generation_descriptor": generation_descriptor,
            "generation_provenance_id": provenance_id,
            "dataset_hashes": dataset_hashes,
            "d1_collection_selection_binding": d1_selection_binding,
            "probe_binding": probe_binding,
            "three_source_training_support": training_support,
            "final_view_support": final_view_support,
            "natural_only_view": natural_only_view_report,
            "round1_source_capacity": round1_source_capacity,
            "round1_source_capacity_report_sha256": (
                round1_source_capacity_sha256
            ),
            "release_checks": release_checks,
            "release_eligible": release_eligible,
            "release_failures": [],
        }
        provenance_path = staged_paths["aggregate.generation_provenance.json"]
        _write_text_artifact(
            provenance_path,
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        preflight_path = staged_paths["aggregate.preflight.json"]
        _write_text_artifact(
            preflight_path,
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        )
        checksum_paths = [*output_paths.values(), provenance_path, preflight_path]
        _write_text_artifact(
            staged_paths["SHA256SUMS"],
            "".join(
                f"{file_sha256(path)}  {path.name}\n"
                for path in sorted(checksum_paths)
            ),
        )

        # Recheck immediately before the one visible publication operation.
        # If a concurrent writer populated the destination, os.replace refuses
        # to replace that non-empty directory and the staged bundle is removed.
        _preflight_round1_output_directory(output_dir)
        if output_dir.is_dir():
            # Windows cannot atomically replace even an empty directory.  The
            # preflight above proves this exact destination is empty; remove
            # only that leaf so the same one-step publication works on both
            # platforms.  A racing writer makes os.replace fail closed.
            output_dir.rmdir()
        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the provenance-valid frozen D0/natural-D1/probe Round-1 "
            "SFT aggregate"
        )
    )
    parser.add_argument("--d0-aggregate-dir", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_round1_aggregate(
        d0_aggregate_dir=args.d0_aggregate_dir,
        d1_path=args.d1,
        d1_manifest_path=args.d1_manifest,
        probe_path=args.probe,
        probe_manifest_path=args.probe_manifest,
        reviewed_source_commit=args.reviewed_source_commit,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
