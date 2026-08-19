from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.dagger.build_dagger1_development_holdout import (
    APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT,
    DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
    DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
    DAGGER1_DEVELOPMENT_SPLIT,
    DAGGER1_DEVELOPMENT_SUITE_NAME,
    DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
    REQUIRED_POST_EVALUATION_RECOVERY_STRATA,
)
from psse_env.dagger.dataset_builder import write_jsonl
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
    inspect_release_checkpoint,
)
from psse_env.dagger.replay_buffer import (
    DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA,
    DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS,
    DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS,
    DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS,
    audit_dagger1_independent_root_support,
    dagger1_replay_capacity_report,
    dagger1_targeted_state_cells,
)
from psse_env.dagger.rollout_collector import (
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    RECOMMENDED_DAGGER1_RECOVERY_STRATA,
    DaggerRolloutCollector,
    audit_dagger1_recovery_labels,
    audit_target_aware_state_classes,
    summarize_dagger1_offline_teacher_target_quarantine,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.providers.matpower import PARAMETER_RANKING_CONTRACT
from psse_env.sft.provenance import (
    AGGREGATE_MANIFEST_FILENAME,
    file_sha256,
    git_source_state,
    stable_json_sha256,
    validate_aggregate_manifest_binding,
)


DEFAULT_FORBIDDEN_SUITE = (
    Path(__file__).resolve().parent / "suites" / "bc0_eval_suite_v1.json"
)
DEFAULT_EVALUATION_POLICY = (
    Path(__file__).resolve().parent / "bc0_evaluation_policy.json"
)
DEFAULT_ENV_FACTORY_SPEC = (
    "psse_env.dagger.release_factories:production_environment_factory"
)
DEFAULT_POLICY_FACTORY_SPEC = (
    "psse_env.dagger.release_factories:gemma_release_policy_factory"
)
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_ADAPTER_TREE_REVISION = re.compile(r"[0-9a-fA-F]{64}")
DEFAULT_TARGET_MIN_ROWS = 300
DEFAULT_TARGET_MAX_ROWS = 600
DAGGER1_SCENARIO_BUILDER_CONTRACT = (
    "fresh_train_partition_dagger1_scenarios_v4"
)
DAGGER1_COLLECTION_SCHEDULE_CONTRACT = (
    "dagger1_predeclared_collection_schedule_v2"
)
DAGGER1_COLLECTION_SELECTION_CONTRACT = (
    "dagger1_floor_preserving_natural_row_selection_v1"
)
DAGGER1_PRODUCTION_ROW_TARGET_CONTRACT = (
    "dagger1_reviewed_production_row_target_v1"
)
DAGGER1_ROLLOUT_MATRIX_CONTRACT = "dagger1_rollout_disposition_matrix_v1"
DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY = {
    "measurement+parameter": 2,
    "multi_measurement": 3,
    "parameter": 1,
}
DAGGER1_RESERVE_FAMILY_PRIORITY = (
    "multi_measurement",
    "measurement+parameter",
    "parameter",
)
DAGGER1_PRIMARY_PLAN = {
    "measurement+parameter": 48,
    "multi_measurement": 48,
    "parameter": 24,
}
DAGGER1_BASE_RESERVE_PLAN = {
    "measurement+parameter": 48,
    "multi_measurement": 31,
    "parameter": 0,
}
DAGGER1_TOPUP_RESERVE_PLAN = {
    # These are new physical roots, not additional replicas of an existing
    # root.  They are an explicit deterministic margin for independent-root
    # recovery gates after the 02db912c9037 collection exhausted its reserve.
    "measurement+parameter": 12,
    "multi_measurement": 0,
    "parameter": 0,
}
DAGGER1_RESERVE_PLAN = {
    family: DAGGER1_BASE_RESERVE_PLAN[family]
    + DAGGER1_TOPUP_RESERVE_PLAN[family]
    for family in DAGGER1_PRIMARY_PLAN
}
DAGGER1_PREDECESSOR_SOURCE_COMMIT = (
    "02db912c9037f4ca22ba8c3e86299148091c13e9"
)
DAGGER1_PREDECESSOR_TRAINING_ROOT_SET_SHA256 = (
    "9eb1f70c6d957cb1ba886b207780b09807d476c4a82f57ad5f873a04594d3c70"
)
DAGGER1_TOPUP_SUBCOHORT = "fresh_root_topup"
_DAGGER1_REVIEWED_COMBINED_RESERVE_PLAN = {
    "measurement+parameter": 60,
    "multi_measurement": 31,
    "parameter": 0,
}
if DAGGER1_RESERVE_PLAN != _DAGGER1_REVIEWED_COMBINED_RESERVE_PLAN:
    raise RuntimeError("DAgger-1 base plus top-up reserve plan drifted")
DAGGER1_TRAINING_POOL_PLAN = {
    family: DAGGER1_PRIMARY_PLAN[family] + DAGGER1_RESERVE_PLAN[family]
    for family in DAGGER1_PRIMARY_PLAN
}
DAGGER1_CANDIDATE_REQUEST_PLAN = {
    "measurement+parameter": 108,
    "multi_measurement": 176,
    "parameter": 48,
}
DAGGER1_PRIMARY_MULTI_MEASUREMENT_CARDINALITY_QUOTA = {
    "2": 16,
    "3": 6,
    "4": 10,
    "5": 16,
}
DAGGER1_RESERVE_MULTI_MEASUREMENT_CARDINALITY_INVENTORY = {
    "3": 12,
    "4": 5,
    "5": 14,
}
DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_INVENTORY = {
    "2": 3,
    "3": 3,
    "4": 3,
    "5": 3,
}
DAGGER1_DEVELOPMENT_RESERVED_COUNT_BY_FAMILY = {
    "measurement+parameter": 0,
    "multi_measurement": 12,
    "parameter": 0,
}
DAGGER1_FRESH_CANDIDATE_COUNT_BY_FAMILY = {
    "measurement+parameter": 108,
    "multi_measurement": 91,
    "parameter": 35,
}
DAGGER1_FRESH_CANDIDATE_CARDINALITY_INVENTORY = {
    "measurement+parameter": {"2": 108},
    "multi_measurement": {"2": 19, "3": 21, "4": 18, "5": 33},
    "parameter": {"1": 35},
}
DAGGER1_UNUSED_FRESH_CANDIDATE_COUNT_BY_FAMILY = {
    "measurement+parameter": 0,
    "multi_measurement": 0,
    "parameter": 11,
}
DAGGER1_RAW_CANDIDATE_COUNT = 271
DAGGER1_FRESH_CANDIDATE_COUNT = 234
DAGGER1_DEVELOPMENT_CANDIDATE_REQUEST_PLAN = {
    "measurement+parameter": 48,
    "multi_measurement": 176,
    "parameter": 24,
}
DAGGER1_DEVELOPMENT_FRESH_CANDIDATE_COUNT_BY_FAMILY = {
    "measurement+parameter": 48,
    "multi_measurement": 12,
    "parameter": 9,
}
DAGGER1_DEVELOPMENT_FRESH_CANDIDATE_CARDINALITY_INVENTORY = {
    "measurement+parameter": {"2": 48},
    "multi_measurement": {"2": 3, "3": 3, "4": 3, "5": 3},
    "parameter": {"1": 9},
}
DAGGER1_DEVELOPMENT_RAW_CANDIDATE_COUNT = 187
DAGGER1_SCENARIO_SEED = 20260720
DAGGER1_DEVELOPMENT_SEED = 20260721
DAGGER1_DEVELOPMENT_SOURCE_BINDINGS = frozenset(
    {
        "psse_env/dagger/build_dagger1_development_holdout.py",
        "psse_env/dagger/suite_builder.py",
        "psse_env/providers/scenario_generator.py",
    }
)
FAILED_COLLECTION_ARTIFACT_TYPE = (
    "dagger1_failed_strict_collection_diagnostic_bundle"
)
# A complete-schedule analysis bundle and a strict NO-GO bundle are both
# training-ineligible, but they answer different questions: the strict bundle
# records where production collection stopped, the analysis bundle records what
# the full predeclared schedule can generate.  Giving them the same top-level
# identity made an analysis run indistinguishable from a genuine strict failure
# once its stdout was gone.
ANALYSIS_COMPLETE_ARTIFACT_TYPE = "dagger1_complete_schedule_analysis_bundle"
FAILED_COLLECTION_CANDIDATE_ROWS = (
    "diagnostic.candidate_recovery_rows.jsonl"
)
FAILED_COLLECTION_ALL_ROWS = "diagnostic.all_visited_rows.jsonl"
FAILED_COLLECTION_EVIDENCE = "failure_evidence.json"
FAILED_COLLECTION_CHECKSUMS = "SHA256SUMS"
_TRUTH_KEYS = frozenset(
    {
        "clean_case",
        "clean_measurements",
        "clean_parameter_values",
        "hidden_truth",
        "oracle_action_hints",
        "suggested_actions",
        "true_measurement_errors",
        "true_parameter_errors",
        "true_topology_errors",
    }
)


def _forbidden_collection_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}"
            if (
                key == "audit"
                or key in _TRUTH_KEYS
                or key.startswith("true_")
                or key.startswith("clean_")
                or key.startswith("remaining_true_")
            ):
                paths.append(path)
                continue
            paths.extend(_forbidden_collection_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_forbidden_collection_paths(item, f"{prefix}[{index}]"))
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fsynced_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_fsynced_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    default=str,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _failed_collection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = copy.deepcopy([dict(row) for row in rows])
    for row in result:
        row["collection_training_eligible"] = False
        row["collection_disposition"] = (
            "failed_strict_gate_diagnostic_only"
        )
        labels = row.get("labels")
        if isinstance(labels, dict):
            labels["collection_training_eligible"] = False
            labels["collection_disposition"] = (
                "failed_strict_gate_diagnostic_only"
            )
    return result


def _round1_publication_contract(
    training_eligible: object,
) -> dict[str, bool]:
    """Publish Round-1 outputs only for an exact training-eligibility GO."""

    eligible = training_eligible is True
    return {
        "strict_gate_passed": eligible,
        "round1_aggregate_eligible": eligible,
        "production_outputs_published": eligible,
    }


def failed_strict_collection_gate_names(
    *,
    collection_gate: Mapping[str, Any],
    targeted_coverage: Mapping[str, Any],
    independent_root_support: Mapping[str, Any],
    truth_audit_quarantine: Mapping[str, Any],
    selection_report: Mapping[str, Any] | None = None,
    round1_replay_capacity: Mapping[str, Any] | None = None,
    rollout_matrix: Mapping[str, Any] | None = None,
) -> list[str]:
    reports = {
        "independent_root_support": independent_root_support,
        "offline_teacher_target_quarantine_summary": (
            truth_audit_quarantine
        ),
        "recommended_collection_gate": collection_gate,
        "targeted_state_coverage": targeted_coverage,
    }
    if selection_report is not None:
        reports["deterministic_collection_selection"] = selection_report
    if round1_replay_capacity is not None:
        reports["round1_replay_capacity"] = round1_replay_capacity
    if rollout_matrix is not None:
        reports["rollout_disposition_matrix"] = rollout_matrix
    return sorted(
        name for name, report in reports.items() if report.get("passed") is not True
    )


def write_failed_collection_evidence_bundle(
    failure_dir: Path,
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish an immutable, diagnostic-only strict-gate failure.

    The bundle deliberately does not implement the production D1 manifest
    interface.  In particular, its evidence file is not named ``*.manifest``
    and it never exposes top-level ``output_sha256``/``all_output`` fields.
    """

    destination = Path(failure_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "DAgger-1 refuses to overwrite failed-collection evidence: "
            f"{destination}"
        )
    failed_gate_names = evidence.get("failed_gate_names")
    if (
        not isinstance(failed_gate_names, list)
        or not failed_gate_names
        or not all(isinstance(name, str) and name for name in failed_gate_names)
    ):
        raise ValueError(
            "failed-collection evidence must name at least one failed gate"
        )

    diagnostic_candidates = _failed_collection_rows(candidate_rows)
    diagnostic_all_rows = _failed_collection_rows(all_rows)
    validate_export_rows_truth_free(diagnostic_candidates)
    validate_export_rows_truth_free(diagnostic_all_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
    )
    try:
        candidate_path = staging / FAILED_COLLECTION_CANDIDATE_ROWS
        all_rows_path = staging / FAILED_COLLECTION_ALL_ROWS
        evidence_path = staging / FAILED_COLLECTION_EVIDENCE
        checksums_path = staging / FAILED_COLLECTION_CHECKSUMS
        _write_fsynced_jsonl(candidate_path, diagnostic_candidates)
        _write_fsynced_jsonl(all_rows_path, diagnostic_all_rows)

        # The stopping report is the authoritative in-band record of which mode
        # produced this bundle; promote it so the top-level identity is
        # unambiguous without reading nested structures or stdout.
        analysis_only = bool(
            (evidence.get("collection_stopping_report") or {}).get("analysis_only")
        )
        manifest = copy.deepcopy(dict(evidence))
        manifest.update(
            {
                "artifact_schema_version": 1,
                "artifact_type": (
                    ANALYSIS_COMPLETE_ARTIFACT_TYPE
                    if analysis_only
                    else FAILED_COLLECTION_ARTIFACT_TYPE
                ),
                "collection_outcome": (
                    "analysis_only_complete_schedule_exhausted"
                    if analysis_only
                    else "strict_gate_failed"
                ),
                "analysis_only": analysis_only,
                "diagnostic_only": True,
                "training_eligible": False,
                "release_evidence_eligible": False,
                **_round1_publication_contract(False),
                "strict_gate_requested": True,
                "strict_gate_evaluated": True,
                "expected_exit_code": 1,
                "diagnostic_artifacts": {
                    "candidate_recovery_rows": {
                        "relative_path": candidate_path.name,
                        "row_count": len(diagnostic_candidates),
                        "sha256": _file_sha256(candidate_path),
                    },
                    "all_visited_rows": {
                        "relative_path": all_rows_path.name,
                        "row_count": len(diagnostic_all_rows),
                        "sha256": _file_sha256(all_rows_path),
                    },
                },
            }
        )
        forbidden_manifest_fields = {"output_sha256", "all_output"}
        overlap = sorted(forbidden_manifest_fields & set(manifest))
        if overlap:
            raise ValueError(
                "failed-collection evidence may not expose production manifest "
                "fields: " + ", ".join(overlap)
            )
        _write_fsynced_text(
            evidence_path,
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        checksum_paths = (candidate_path, all_rows_path, evidence_path)
        _write_fsynced_text(
            checksums_path,
            "".join(
                f"{_file_sha256(path)}  {path.name}\n"
                for path in sorted(checksum_paths, key=lambda item: item.name)
            ),
        )
        _fsync_directory(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                "DAgger-1 refuses to overwrite failed-collection evidence: "
                f"{destination}"
            )
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return copy.deepcopy(manifest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        payload: Any = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(text)
    if not isinstance(payload, list) or not payload:
        raise ValueError("DAgger-1 input must be a non-empty JSON/JSONL list")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("every DAgger-1 scenario must be a JSON object")
    return [dict(row) for row in payload]


def frozen_physical_roots(path: Path) -> frozenset[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("forbidden evaluation suite must be a suite mapping")
    roots: set[str] = set()
    for rows in payload.values():
        if not isinstance(rows, list):
            raise ValueError("forbidden evaluation suite values must be lists")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("forbidden evaluation suite rows must be objects")
            grouping = row.get("grouping")
            if not isinstance(grouping, Mapping):
                raise ValueError("forbidden suite row is missing grouping")
            root = str(grouping.get("physical_root_fingerprint") or "").strip()
            if not root:
                raise ValueError("forbidden suite row is missing a physical root")
            roots.add(root)
    if not roots:
        raise ValueError("forbidden evaluation suite has no physical roots")
    return frozenset(roots)


def source_commit_drift(
    manifest_source: Any, source_state: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Report a manifest/source commit divergence, or ``None`` when aligned."""
    if not isinstance(manifest_source, Mapping):
        return None
    recorded = manifest_source.get("source_commit")
    running = source_state.get("source_commit")
    if recorded == running:
        return None
    return {"manifest_source_commit": recorded, "running_source_commit": running}


def validate_d0_provenance_binding(
    provenance: Mapping[str, Any],
    *,
    raw_path: Path,
    source_state: Mapping[str, Any],
    analysis_only: bool = False,
) -> None:
    """Require a clean, content-addressed D0 prerequisite for collection.

    ``analysis_only`` relaxes the manifest/source commit equality alone, for the
    non-publishing complete-schedule analysis mode: the predeclared schedule is
    bound to the commit that generated it, so any source change would otherwise
    make the existing schedule unrunnable.  The clean-tree requirement on the
    recorded manifest is unchanged, and the drift is recorded in the run report.
    """

    descriptor = provenance.get("generation_descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else None
    d0_source = descriptor.get("source_state") if descriptor is not None else None
    dataset_hashes = provenance.get("dataset_hashes")
    manifest_binding = validate_aggregate_manifest_binding(
        provenance,
        aggregate_dir=raw_path.parent,
    )
    checks = {
        "release_eligible": provenance.get("release_eligible") is True,
        "generation_descriptor": descriptor is not None,
        "generation_provenance_id": descriptor is not None
        and provenance.get("generation_provenance_id")
        == stable_json_sha256(descriptor),
        "release_eligible_source": isinstance(d0_source, Mapping)
        and d0_source.get("release_eligible_source") is True,
        "source_commit": isinstance(d0_source, Mapping)
        and (
            analysis_only
            or d0_source.get("source_commit") == source_state.get("source_commit")
        ),
        "raw_sha256": isinstance(dataset_hashes, Mapping)
        and dataset_hashes.get(raw_path.name) == _file_sha256(raw_path),
        "aggregate_manifest_sha256": manifest_binding["passed"] is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "D0 aggregate is not clean/content-addressed for current source: "
            + ", ".join(failed)
        )


def validate_development_holdout_binding(
    holdout_path: Path,
    manifest_path: Path,
    *,
    generator_report_path: Path,
    source_state: Mapping[str, Any],
    scenario_input_path: Path,
    scenario_manifest_path: Path,
    d0_raw_path: Path,
    d0_provenance_path: Path,
    d0_manifest_path: Path,
    forbidden_suite_path: Path,
    evaluation_policy_path: Path,
    require_model_selection_eligible: bool,
    analysis_only: bool = False,
) -> frozenset[str]:
    """Validate and return the independently reserved development roots."""

    if not (
        holdout_path.is_file()
        and manifest_path.is_file()
        and generator_report_path.is_file()
    ):
        raise FileNotFoundError(
            "DAgger-1 development holdout, manifest, and generator report "
            "must all exist"
        )
    payload = json.loads(holdout_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generator_report = json.loads(
        generator_report_path.read_text(encoding="utf-8")
    )
    scenario_manifest = json.loads(
        scenario_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("development holdout and manifest must be JSON objects")
    if not isinstance(generator_report, Mapping):
        raise ValueError("development holdout generator report must be an object")
    validate_training_source_report(generator_report)
    if not isinstance(scenario_manifest, Mapping):
        raise ValueError("D1 training scenario manifest must be a JSON object")
    if set(payload) != {DAGGER1_DEVELOPMENT_SUITE_NAME}:
        raise ValueError("development holdout has an unexpected suite mapping")
    rows = payload.get(DAGGER1_DEVELOPMENT_SUITE_NAME)
    if not isinstance(rows, list) or not rows:
        raise ValueError("development holdout suite must be a non-empty list")

    roots: set[str] = set()
    family_counts: Counter[str] = Counter()
    multi_measurement_roots: set[str] = set()
    multi_measurement_cardinality: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("scenario_schema_version") != 1:
            raise ValueError(f"development holdout row {index} is not schema-v1")
        execution = row.get("execution")
        audit = row.get("audit")
        grouping = row.get("grouping")
        if not all(
            isinstance(value, Mapping)
            for value in (execution, audit, grouping)
        ):
            raise ValueError(
                f"development holdout row {index} has a malformed envelope"
            )
        if grouping.get("split") != DAGGER1_DEVELOPMENT_SPLIT:
            raise ValueError(
                f"development holdout row {index} has the wrong split"
            )
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        family = str(grouping.get("scenario_family") or "").strip()
        if not root or not family:
            raise ValueError(
                f"development holdout row {index} lacks root/family metadata"
            )
        if root in roots:
            raise ValueError("development holdout physical roots are not unique")
        roots.add(root)
        family_counts[family] += 1
        if family == "multi_measurement":
            multi_measurement_roots.add(root)
            multi_measurement_cardinality[
                str(grouping.get("error_cardinality"))
            ] += 1

    manifest_source = manifest.get("source_state")
    source_bindings = manifest.get("source_bindings")
    repo_root = Path(__file__).resolve().parents[2]
    source_bindings_current = (
        isinstance(source_bindings, Mapping)
        and set(source_bindings) == DAGGER1_DEVELOPMENT_SOURCE_BINDINGS
    )
    if source_bindings_current:
        for relative_value, expected_sha256 in source_bindings.items():
            relative = Path(str(relative_value))
            candidate = (repo_root / relative).resolve()
            try:
                candidate.relative_to(repo_root.resolve())
            except ValueError:
                source_bindings_current = False
                break
            if (
                relative.is_absolute()
                or not candidate.is_file()
                or expected_sha256 != _file_sha256(candidate)
            ):
                source_bindings_current = False
                break
    d0_rows = _load_json_or_jsonl(d0_raw_path)
    d0_roots = {
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in d0_rows
        if str(row.get("physical_root_fingerprint") or "").strip()
    }
    frozen_roots = set(frozen_physical_roots(forbidden_suite_path))
    training_rows = _load_json_or_jsonl(scenario_input_path)
    training_roots = {
        str(
            _scenario_grouping(row).get("physical_root_fingerprint") or ""
        ).strip()
        for row in training_rows
        if str(
            _scenario_grouping(row).get("physical_root_fingerprint") or ""
        ).strip()
    }
    root_sets = {
        "d0": d0_roots,
        "frozen": frozen_roots,
        "d1_training": training_roots,
        "development": roots,
    }
    declared_root_counts = manifest.get("root_counts")
    declared_root_hashes = manifest.get("root_set_sha256")
    pairwise_overlap = manifest.get("pairwise_input_overlap")
    expected_pairwise_overlap = {
        "d0_frozen": sorted(d0_roots & frozen_roots),
        "d0_d1_training": sorted(d0_roots & training_roots),
        "frozen_d1_training": sorted(frozen_roots & training_roots),
    }
    plan = manifest.get("plan")
    selected_counts = manifest.get("selected_count_by_family")
    root_set_hashes = declared_root_hashes
    overlaps = manifest.get("development_protected_overlap")
    normalized_plan = (
        {str(key): value for key, value in plan.items()}
        if isinstance(plan, Mapping)
        else {}
    )
    approved_plan = (
        normalized_plan
        == dict(sorted(DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items()))
        and len(roots) == APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT
    )
    declared_model_selection_eligible = manifest.get(
        "diagnostic_closed_loop_model_selection_eligible"
    )
    training_reserved_roots, training_reservation_valid = (
        _development_reservation_from_manifest(scenario_manifest)
    )
    expected_reserved_multi_roots = training_reserved_roots.get(
        "multi_measurement", []
    )
    declared_training_reservation = manifest.get(
        "training_development_reserved_roots_by_family"
    )
    reservation_boundary_overlap = manifest.get(
        "training_development_reserved_boundary_overlap"
    )
    fresh_candidate_inventory = manifest.get("fresh_candidate_inventory")
    fresh_candidate_count = (
        sum(
            int(item.get("physical_root_count") or 0)
            for item in fresh_candidate_inventory.values()
            if isinstance(item, Mapping)
        )
        if isinstance(fresh_candidate_inventory, Mapping)
        else -1
    )
    filtered_protected_count = manifest.get("filtered_protected_root_count")
    filtered_parameter_scan_count = manifest.get(
        "filtered_multi_measurement_with_parameter_scans_root_count"
    )
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "scenario_schema_version": manifest.get("scenario_schema_version") == 1,
        "artifact_type": manifest.get("artifact_type")
        == "dagger1_development_holdout_suite",
        "builder_contract": manifest.get("builder_contract")
        == DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
        "suite_name": manifest.get("suite_name")
        == DAGGER1_DEVELOPMENT_SUITE_NAME,
        "suite_format": manifest.get("suite_format")
        == "evaluation_suite_mapping_v1",
        "split": manifest.get("split") == DAGGER1_DEVELOPMENT_SPLIT,
        "source_partition": manifest.get("source_partition") == "train",
        "parameter_threshold": manifest.get(
            "parameter_ranking_dominance_threshold"
        )
        == DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
        "seed": manifest.get("seed") == DAGGER1_DEVELOPMENT_SEED,
        "source_commit": isinstance(manifest_source, Mapping)
        and manifest_source.get("release_eligible_source") is True
        and (
            analysis_only
            or manifest_source.get("source_commit")
            == source_state.get("source_commit")
        ),
        "source_bindings": source_bindings_current,
        "output_sha256": manifest.get("output_sha256")
        == _file_sha256(holdout_path),
        "generator_report_sha256": manifest.get("generator_report_sha256")
        == _file_sha256(generator_report_path),
        "scenario_count": manifest.get("scenario_count") == len(rows),
        "physical_root_count": manifest.get("physical_root_count") == len(roots),
        "family_counts": isinstance(selected_counts, Mapping)
        and dict(selected_counts) == dict(sorted(family_counts.items()))
        and normalized_plan == dict(sorted(family_counts.items())),
        "root_set_sha256": isinstance(root_set_hashes, Mapping)
        and root_set_hashes.get("development")
        == stable_json_sha256(sorted(roots)),
        "root_counts": isinstance(declared_root_counts, Mapping)
        and dict(declared_root_counts)
        == {name: len(values) for name, values in root_sets.items()},
        "all_root_set_sha256": isinstance(declared_root_hashes, Mapping)
        and dict(declared_root_hashes)
        == {
            name: stable_json_sha256(sorted(values))
            for name, values in root_sets.items()
        },
        "pairwise_input_overlap": isinstance(pairwise_overlap, Mapping)
        and dict(pairwise_overlap) == expected_pairwise_overlap
        and not any(expected_pairwise_overlap.values()),
        "candidate_multiplier": manifest.get("candidate_multiplier") == 4,
        "candidate_request_plan": (
            _exact_string_int_mapping(
                manifest.get("candidate_request_plan"),
                DAGGER1_DEVELOPMENT_CANDIDATE_REQUEST_PLAN,
            )
            and _exact_string_int_mapping(
                manifest.get("candidate_plan"),
                DAGGER1_DEVELOPMENT_CANDIDATE_REQUEST_PLAN,
            )
        ),
        "candidate_count": manifest.get("candidate_count")
        == DAGGER1_DEVELOPMENT_RAW_CANDIDATE_COUNT,
        "candidate_count_arithmetic": (
            isinstance(filtered_protected_count, int)
            and not isinstance(filtered_protected_count, bool)
            and filtered_protected_count >= 0
            and isinstance(filtered_parameter_scan_count, int)
            and not isinstance(filtered_parameter_scan_count, bool)
            and filtered_parameter_scan_count >= 0
            and manifest.get("candidate_count")
            == fresh_candidate_count
            + filtered_protected_count
            + filtered_parameter_scan_count
        ),
        "fresh_candidate_inventory": _candidate_inventory_matches(
            manifest.get("fresh_candidate_inventory"),
            expected_counts=(
                DAGGER1_DEVELOPMENT_FRESH_CANDIDATE_COUNT_BY_FAMILY
            ),
            expected_cardinality=(
                DAGGER1_DEVELOPMENT_FRESH_CANDIDATE_CARDINALITY_INVENTORY
            ),
        ),
        "training_reservation": training_reservation_valid,
        "training_reservation_copy": (
            isinstance(declared_training_reservation, Mapping)
            and dict(declared_training_reservation)
            == training_reserved_roots
        ),
        "selected_multi_measurement_cardinality": (
            _exact_string_int_mapping(
                manifest.get(
                    "selected_multi_measurement_cardinality_inventory"
                ),
                DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_INVENTORY,
            )
            and dict(multi_measurement_cardinality)
            == DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_INVENTORY
        ),
        "selected_multi_measurement_reservation": (
            manifest.get(
                "selected_multi_measurement_matches_training_reservation"
            )
            is True
            and sorted(multi_measurement_roots)
            == expected_reserved_multi_roots
            and manifest.get(
                "training_development_reserved_multi_measurement_root_set_sha256"
            )
            == stable_json_sha256(expected_reserved_multi_roots)
            and manifest.get("selected_multi_measurement_root_set_sha256")
            == stable_json_sha256(sorted(multi_measurement_roots))
        ),
        "training_reservation_boundary_overlap": (
            isinstance(reservation_boundary_overlap, Mapping)
            and set(reservation_boundary_overlap)
            == {"d0", "frozen", "d1_training"}
            and all(value == [] for value in reservation_boundary_overlap.values())
        ),
        "intended_use": manifest.get("intended_use")
        == "dagger1_closed_loop_development_model_selection_only",
        "required_recovery_strata": manifest.get(
            "required_post_evaluation_recovery_strata"
        )
        == list(REQUIRED_POST_EVALUATION_RECOVERY_STRATA),
        "recovery_coverage_contract": manifest.get(
            "recovery_strata_coverage_requires_closed_loop_evaluation"
        )
        is True,
        "recovery_qualification_status": manifest.get(
            "recovery_strata_qualification_status"
        )
        == "pending_teacher_opportunity_trace_instrumentation",
        "training_eligible": manifest.get("training_eligible") is False,
        "training_collection_eligible": manifest.get(
            "training_collection_eligible"
        )
        is False,
        "release_evidence_eligible": manifest.get("release_evidence_eligible")
        is False,
        "promotion_evidence_eligible": manifest.get(
            "promotion_evidence_eligible"
        )
        is False,
        "model_selection_eligibility_consistent": (
            declared_model_selection_eligible is approved_plan
        ),
        "model_selection_eligible": (
            not require_model_selection_eligible
            or declared_model_selection_eligible is True
        ),
        "recovery_model_selection_eligible": manifest.get(
            "recovery_stratum_qualified_model_selection_eligible"
        )
        is False,
        "protected_overlap": isinstance(overlaps, Mapping)
        and set(overlaps) == {"d0", "frozen", "d1_training"}
        and all(value == [] for value in overlaps.values()),
        "d1_training_scenarios_sha256": manifest.get(
            "d1_training_scenarios_sha256"
        )
        == _file_sha256(scenario_input_path),
        "d1_training_manifest_sha256": manifest.get(
            "d1_training_manifest_sha256"
        )
        == _file_sha256(scenario_manifest_path),
        "d0_raw_sha256": manifest.get("d0_raw_sha256")
        == _file_sha256(d0_raw_path),
        "d0_provenance_sha256": manifest.get(
            "d0_generation_provenance_sha256"
        )
        == _file_sha256(d0_provenance_path),
        "d0_manifest_sha256": manifest.get("d0_manifest_sha256")
        == _file_sha256(d0_manifest_path),
        "frozen_suite_sha256": manifest.get("frozen_suite_sha256")
        == _file_sha256(forbidden_suite_path),
        "evaluation_policy_sha256": manifest.get("evaluation_policy_sha256")
        == _file_sha256(evaluation_policy_path),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "development holdout binding failed: " + ", ".join(failed)
        )
    return frozenset(roots)


def validate_training_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    forbidden_roots: frozenset[str],
) -> None:
    if not forbidden_roots:
        raise ValueError("forbidden physical-root holdout must be non-empty")
    for index, scenario in enumerate(scenarios):
        is_envelope = all(
            key in scenario for key in ("execution", "audit", "grouping")
        )
        if is_envelope:
            if scenario.get("scenario_schema_version") != 1:
                raise ValueError(
                    "DAgger-1 scenario_schema_version must equal 1"
                )
            execution = scenario.get("execution")
            audit = scenario.get("audit")
            grouping = scenario.get("grouping")
            if not all(
                isinstance(value, Mapping)
                for value in (execution, audit, grouping)
            ):
                raise ValueError("DAgger-1 execution/audit/grouping envelope is malformed")
            unexpected = sorted(
                set(scenario)
                - {"scenario_schema_version", "execution", "audit", "grouping"}
            )
            if unexpected:
                raise ValueError(
                    "DAgger-1 envelope has unexpected top-level keys: "
                    + ", ".join(unexpected)
                )
            leaked = sorted(
                _forbidden_collection_paths(execution)
                + _forbidden_collection_paths(grouping)
            )
            if leaked:
                raise ValueError(
                    "DAgger-1 execution/grouping is not truth-free: "
                    + ", ".join(leaked)
                )
            unexpected_audit = sorted(
                set(audit)
                - {"truth", "evaluation_intervention", "release_audit"}
            )
            truth = audit.get("truth")
            if unexpected_audit or not isinstance(truth, Mapping):
                raise ValueError(
                    "DAgger-1 audit must contain only private truth plus "
                    "optional quarantined evaluation metadata"
                )
            for key in ("evaluation_intervention", "release_audit"):
                if key in audit and not isinstance(audit.get(key), Mapping):
                    raise ValueError(
                        f"DAgger-1 private audit field {key!r} must be a mapping"
                    )
            invalid_truth_keys = sorted(
                str(key)
                for key in truth
                if not (
                    str(key) == "truth_complete"
                    or str(key).startswith("true_")
                    or str(key).startswith("clean_")
                    or str(key).startswith("remaining_true_")
                )
            )
            if invalid_truth_keys or truth.get("truth_complete") is not True:
                raise ValueError(
                    "DAgger-1 private audit truth contract is invalid: "
                    + ", ".join(invalid_truth_keys)
                )
            scenario_id = str(execution.get("scenario_id") or index)
            scenario_metadata = grouping
        else:
            if "audit" in scenario:
                raise ValueError(
                    "flat DAgger-1 scenarios may not contain audit; use the "
                    "versioned execution/audit/grouping envelope ($.audit)"
                )
            scenario_id = str(
                scenario.get("scenario_id") or scenario.get("id") or index
            )
            scenario_metadata = scenario
            forbidden_paths = sorted(_forbidden_collection_paths(scenario))
            if forbidden_paths:
                raise ValueError(
                    f"DAgger-1 scenario {scenario_id!r} is not truth-free: "
                    + ", ".join(forbidden_paths)
                )
        split = str(
            scenario_metadata.get("dataset_split")
            or scenario_metadata.get("split")
            or ""
        ).strip()
        if split not in {"train", "dagger_train"}:
            raise ValueError(
                f"DAgger-1 scenario {scenario_id!r} must declare train or "
                f"dagger_train split, got {split or 'missing'}"
            )
        root = str(
            scenario_metadata.get("physical_root_fingerprint") or ""
        ).strip()
        if not root:
            raise ValueError(
                f"DAgger-1 scenario {scenario_id!r} lacks a physical root"
            )
        if root in forbidden_roots:
            raise ValueError(
                f"DAgger-1 scenario {scenario_id!r} overlaps a protected "
                f"D0/evaluation root {root}"
            )


def validate_training_source_report(report: Mapping[str, Any]) -> None:
    """Require evidence that fresh roots came from the train source partition."""
    source_partition = report.get("source_partition")
    if not isinstance(source_partition, Mapping):
        raise ValueError("scenario-generator report lacks source_partition")
    if (
        source_partition.get("enabled") is not True
        or source_partition.get("selected") != "train"
    ):
        raise ValueError(
            "DAgger-1 scenarios must come from "
            "Round0ScenarioGenerator(source_partition='train')"
        )
    admission = report.get("parameter_ranking_admission")
    if not isinstance(admission, Mapping) or (
        admission.get("contract") != PARAMETER_RANKING_CONTRACT
        or admission.get("enforced") is not True
        or admission.get("threshold")
        != BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
    ):
        raise ValueError(
            "DAgger-1 scenario report lacks the reviewed parameter-ranking "
            "admission threshold"
        )


def _exact_string_int_mapping(value: Any, expected: Mapping[str, int]) -> bool:
    if not isinstance(value, Mapping):
        return False
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in value.values()
    ):
        return False
    return {str(key): item for key, item in value.items()} == dict(expected)


def _candidate_inventory_matches(
    value: Any,
    *,
    expected_counts: Mapping[str, int],
    expected_cardinality: Mapping[str, Mapping[str, int]],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(expected_counts):
        return False
    for family, expected_count in expected_counts.items():
        item = value.get(family)
        if not isinstance(item, Mapping) or set(item) != {
            "physical_root_count",
            "error_cardinality",
            "physical_root_set_sha256",
        }:
            return False
        root_hash = item.get("physical_root_set_sha256")
        if not (
            item.get("physical_root_count") == expected_count
            and _exact_string_int_mapping(
                item.get("error_cardinality"),
                expected_cardinality[family],
            )
            and isinstance(root_hash, str)
            and _ADAPTER_TREE_REVISION.fullmatch(root_hash) is not None
        ):
            return False
    return True


def _development_reservation_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, list[str]], bool]:
    raw_roots = manifest.get("development_reserved_roots_by_family")
    raw_hashes = manifest.get(
        "development_reserved_root_set_sha256_by_family"
    )
    raw_counts = manifest.get("withheld_for_development_count_by_family")
    families = set(DAGGER1_DEVELOPMENT_RESERVED_COUNT_BY_FAMILY)
    if not (
        isinstance(raw_roots, Mapping)
        and isinstance(raw_hashes, Mapping)
        and isinstance(raw_counts, Mapping)
        and set(raw_roots) == families
        and set(raw_hashes) == families
        and set(raw_counts) == families
    ):
        return {}, False
    normalized: dict[str, list[str]] = {}
    for family in sorted(families):
        roots = raw_roots.get(family)
        if not (
            isinstance(roots, Sequence)
            and not isinstance(roots, (str, bytes))
        ):
            return {}, False
        materialized = [str(root).strip() for root in roots]
        if (
            materialized != sorted(materialized)
            or any(not root for root in materialized)
            or len(set(materialized)) != len(materialized)
            or raw_counts.get(family)
            != DAGGER1_DEVELOPMENT_RESERVED_COUNT_BY_FAMILY[family]
            or len(materialized)
            != DAGGER1_DEVELOPMENT_RESERVED_COUNT_BY_FAMILY[family]
            or raw_hashes.get(family) != stable_json_sha256(materialized)
        ):
            return {}, False
        normalized[family] = materialized
    all_roots = [root for roots in normalized.values() for root in roots]
    return normalized, len(all_roots) == len(set(all_roots))


def _topup_reservation_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, list[str]], bool]:
    raw_roots = manifest.get("topup_reserve_roots_by_family")
    raw_hashes = manifest.get("topup_reserve_root_set_sha256_by_family")
    raw_counts = manifest.get("topup_reserve_count_by_family")
    families = set(DAGGER1_TOPUP_RESERVE_PLAN)
    if not (
        isinstance(raw_roots, Mapping)
        and isinstance(raw_hashes, Mapping)
        and isinstance(raw_counts, Mapping)
        and set(raw_roots) == families
        and set(raw_hashes) == families
        and set(raw_counts) == families
    ):
        return {}, False
    normalized: dict[str, list[str]] = {}
    for family in sorted(families):
        roots = raw_roots.get(family)
        if not (
            isinstance(roots, Sequence)
            and not isinstance(roots, (str, bytes))
        ):
            return {}, False
        materialized = [str(root).strip() for root in roots]
        if (
            materialized != sorted(materialized)
            or any(not root for root in materialized)
            or len(set(materialized)) != len(materialized)
            or raw_counts.get(family) != DAGGER1_TOPUP_RESERVE_PLAN[family]
            or len(materialized) != DAGGER1_TOPUP_RESERVE_PLAN[family]
            or raw_hashes.get(family) != stable_json_sha256(materialized)
        ):
            return {}, False
        normalized[family] = materialized
    all_roots = [root for roots in normalized.values() for root in roots]
    valid = bool(
        len(all_roots) == len(set(all_roots))
        and manifest.get("topup_reserve_physical_root_set_sha256")
        == stable_json_sha256(sorted(all_roots))
    )
    return normalized, valid


def validate_scenario_builder_manifest(
    manifest: Mapping[str, Any],
    *,
    scenarios: Sequence[Mapping[str, Any]],
    input_path: Path,
    generator_report_path: Path,
    source_state: Mapping[str, Any],
    d0_raw_path: Path,
    d0_provenance_path: Path,
    d0_manifest_path: Path,
    forbidden_suite_path: Path,
    evaluation_policy_path: Path,
    analysis_only: bool = False,
) -> None:
    """Bind the collected scenarios to the reviewed fresh-root builder."""

    manifest_source = manifest.get("source_state")
    manifest_source = (
        manifest_source if isinstance(manifest_source, Mapping) else {}
    )
    primary_counts = manifest.get("primary_count_by_family")
    reserve_counts = manifest.get("reserve_count_by_family")
    selected_counts = manifest.get("selected_count_by_family")
    schedule = manifest.get("collection_schedule")
    reserved_roots_by_family, reservation_valid = (
        _development_reservation_from_manifest(manifest)
    )
    topup_roots_by_family, topup_binding_valid = (
        _topup_reservation_from_manifest(manifest)
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            primary_counts,
            reserve_counts,
            selected_counts,
            schedule,
        )
    ):
        raise ValueError(
            "scenario-builder manifest lacks the exact v4 collection schedule"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (
            *primary_counts.values(),
            *reserve_counts.values(),
            *selected_counts.values(),
        )
    ):
        raise ValueError("scenario-builder family counts are invalid")
    try:
        normalized_primary = {
            str(key): int(value) for key, value in primary_counts.items()
        }
        normalized_reserve = {
            str(key): int(value) for key, value in reserve_counts.items()
        }
        normalized_selected = {
            str(key): int(value) for key, value in selected_counts.items()
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scenario-builder family counts are invalid") from exc
    expected_selected = {
        family: normalized_primary.get(family, 0)
        + normalized_reserve.get(family, 0)
        for family in sorted(
            set(normalized_primary) | set(normalized_reserve)
        )
    }
    actual_counts: Counter[str] = Counter()
    actual_primary_counts: Counter[str] = Counter()
    actual_reserve_counts: Counter[str] = Counter()
    actual_primary_multi_cardinality: Counter[str] = Counter()
    actual_reserve_multi_cardinality: Counter[str] = Counter()
    actual_topup_counts: Counter[str] = Counter()
    actual_topup_roots_by_family: dict[str, set[str]] = {
        family: set() for family in DAGGER1_TOPUP_RESERVE_PLAN
    }
    actual_roots: set[str] = set()
    actual_primary_roots: set[str] = set()
    actual_reserve_roots: set[str] = set()
    subcohort_valid = True
    for scenario in scenarios:
        grouping = _scenario_grouping(scenario)
        family = str(grouping.get("scenario_family") or "").strip()
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        cohort = str(grouping.get("collection_cohort") or "").strip()
        subcohort = str(grouping.get("collection_subcohort") or "").strip()
        if family:
            actual_counts[family] += 1
            if cohort == "primary":
                subcohort_valid = subcohort_valid and subcohort == "primary"
                actual_primary_counts[family] += 1
                actual_primary_roots.add(root)
                if family == "multi_measurement":
                    actual_primary_multi_cardinality[
                        str(grouping.get("error_cardinality"))
                    ] += 1
            elif cohort == "reserve":
                actual_reserve_counts[family] += 1
                actual_reserve_roots.add(root)
                if family == "multi_measurement":
                    actual_reserve_multi_cardinality[
                        str(grouping.get("error_cardinality"))
                    ] += 1
                if subcohort == DAGGER1_TOPUP_SUBCOHORT:
                    actual_topup_counts[family] += 1
                    actual_topup_roots_by_family.setdefault(family, set()).add(
                        root
                    )
                else:
                    subcohort_valid = (
                        subcohort_valid and subcohort == "base_reserve"
                    )
            else:
                subcohort_valid = False
        if root:
            actual_roots.add(root)
    # Recompute all per-scenario ordering/priority invariants, not only the
    # manifest summary.  Diagnostic uses the primary prefix; training consumes
    # the same immutable schedule plus its finite reserve/repeat suffix.
    dagger1_rollout_batches(scenarios, collection_pass="training")
    normalized_schedule_replicas = schedule.get(
        "maximum_rollout_replicas_by_family"
    )
    d0_roots = {
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in _load_json_or_jsonl(d0_raw_path)
        if str(row.get("physical_root_fingerprint") or "").strip()
    }
    frozen_roots = frozen_physical_roots(forbidden_suite_path)
    fresh_inventory = manifest.get("fresh_candidate_inventory")
    fresh_inventory_count = (
        sum(
            int(item.get("physical_root_count") or 0)
            for item in fresh_inventory.values()
            if isinstance(item, Mapping)
        )
        if isinstance(fresh_inventory, Mapping)
        else -1
    )
    filtered_protected_count = manifest.get("filtered_protected_root_count")
    filtered_parameter_scan_count = manifest.get(
        "filtered_multi_measurement_with_parameter_scans_root_count"
    )
    candidate_count = manifest.get("candidate_count")
    fresh_candidate_count = manifest.get("fresh_candidate_count")
    actual_topup_roots = {
        root
        for roots in actual_topup_roots_by_family.values()
        for root in roots
    }
    actual_base_training_roots = actual_roots - actual_topup_roots
    declared_development_roots = {
        root
        for roots in reserved_roots_by_family.values()
        for root in roots
    }
    declared_topup_roots = {
        root
        for roots in topup_roots_by_family.values()
        for root in roots
    }
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "builder_contract": (
            manifest.get("builder_contract")
            == DAGGER1_SCENARIO_BUILDER_CONTRACT
        ),
        "release_evidence_eligible": (
            manifest.get("release_evidence_eligible") is False
        ),
        "input_sha256": manifest.get("output_sha256") == _file_sha256(input_path),
        "generator_report_sha256": (
            manifest.get("generator_report_sha256")
            == _file_sha256(generator_report_path)
        ),
        "source_commit": (
            manifest_source.get("release_eligible_source") is True
            and (
                analysis_only
                or manifest_source.get("source_commit")
                == source_state.get("source_commit")
            )
        ),
        "source_partition": manifest.get("source_partition") == "train",
        "parameter_threshold": (
            manifest.get("parameter_ranking_dominance_threshold")
            == BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ),
        "seed": manifest.get("seed") == DAGGER1_SCENARIO_SEED,
        "d0_root_count": manifest.get("d0_root_count") == len(d0_roots),
        "frozen_root_count": manifest.get("frozen_root_count")
        == len(frozen_roots),
        "primary_plan": (
            _exact_string_int_mapping(
                manifest.get("plan"), DAGGER1_PRIMARY_PLAN
            )
            and _exact_string_int_mapping(
                manifest.get("primary_plan"), DAGGER1_PRIMARY_PLAN
            )
        ),
        "primary_count_contract": normalized_primary
        == DAGGER1_PRIMARY_PLAN,
        "reserve_count_contract": normalized_reserve
        == DAGGER1_RESERVE_PLAN,
        "base_reserve_plan": _exact_string_int_mapping(
            manifest.get("base_reserve_plan"), DAGGER1_BASE_RESERVE_PLAN
        ),
        "topup_reserve_plan": _exact_string_int_mapping(
            manifest.get("topup_reserve_plan"), DAGGER1_TOPUP_RESERVE_PLAN
        ),
        "topup_reserve_binding": topup_binding_valid,
        "topup_reserve_subcohort": (
            subcohort_valid
            and dict(actual_topup_counts)
            == {
                family: count
                for family, count in DAGGER1_TOPUP_RESERVE_PLAN.items()
                if count
            }
            and actual_topup_roots == declared_topup_roots
            and all(
                actual_topup_roots_by_family.get(family, set())
                == set(topup_roots_by_family.get(family, []))
                for family in DAGGER1_TOPUP_RESERVE_PLAN
            )
        ),
        "topup_predecessor_binding": (
            manifest.get("predecessor_source_commit")
            == DAGGER1_PREDECESSOR_SOURCE_COMMIT
            and manifest.get("predecessor_training_root_count")
            == len(actual_base_training_roots)
            == sum(DAGGER1_PRIMARY_PLAN.values())
            + sum(DAGGER1_BASE_RESERVE_PLAN.values())
            and manifest.get("predecessor_training_root_set_sha256")
            == DAGGER1_PREDECESSOR_TRAINING_ROOT_SET_SHA256
            == stable_json_sha256(sorted(actual_base_training_roots))
            and manifest.get("topup_predecessor_overlap") == []
            and not (actual_topup_roots & actual_base_training_roots)
        ),
        "topup_development_reservation_disjoint": (
            manifest.get("topup_development_reserved_overlap") == []
            and not (actual_topup_roots & declared_development_roots)
        ),
        "topup_protected_roots_disjoint": not (
            actual_topup_roots & (d0_roots | set(frozen_roots))
        ),
        "selected_count_contract": normalized_selected
        == DAGGER1_TRAINING_POOL_PLAN,
        "training_pool_plan": _exact_string_int_mapping(
            manifest.get("training_pool_plan"),
            DAGGER1_TRAINING_POOL_PLAN,
        ),
        "candidate_multiplier": manifest.get("candidate_multiplier") == 2,
        "candidate_request_plan": (
            _exact_string_int_mapping(
                manifest.get("candidate_request_plan"),
                DAGGER1_CANDIDATE_REQUEST_PLAN,
            )
            and _exact_string_int_mapping(
                manifest.get("candidate_plan"),
                DAGGER1_CANDIDATE_REQUEST_PLAN,
            )
        ),
        "candidate_counts": (
            candidate_count == DAGGER1_RAW_CANDIDATE_COUNT
            and fresh_candidate_count == DAGGER1_FRESH_CANDIDATE_COUNT
            and fresh_inventory_count == fresh_candidate_count
            and isinstance(filtered_protected_count, int)
            and not isinstance(filtered_protected_count, bool)
            and filtered_protected_count >= 0
            and isinstance(filtered_parameter_scan_count, int)
            and not isinstance(filtered_parameter_scan_count, bool)
            and filtered_parameter_scan_count >= 0
            and candidate_count
            == fresh_candidate_count
            + filtered_protected_count
            + filtered_parameter_scan_count
        ),
        "fresh_candidate_inventory": _candidate_inventory_matches(
            manifest.get("fresh_candidate_inventory"),
            expected_counts=DAGGER1_FRESH_CANDIDATE_COUNT_BY_FAMILY,
            expected_cardinality=(
                DAGGER1_FRESH_CANDIDATE_CARDINALITY_INVENTORY
            ),
        ),
        "unused_fresh_candidate_counts": _exact_string_int_mapping(
            manifest.get("unused_fresh_candidate_count_by_family"),
            DAGGER1_UNUSED_FRESH_CANDIDATE_COUNT_BY_FAMILY,
        )
        and all(
            DAGGER1_FRESH_CANDIDATE_COUNT_BY_FAMILY[family]
            == DAGGER1_TRAINING_POOL_PLAN[family]
            + DAGGER1_DEVELOPMENT_RESERVED_COUNT_BY_FAMILY[family]
            + DAGGER1_UNUSED_FRESH_CANDIDATE_COUNT_BY_FAMILY[family]
            for family in DAGGER1_FRESH_CANDIDATE_COUNT_BY_FAMILY
        ),
        "primary_multi_measurement_quota": (
            _exact_string_int_mapping(
                manifest.get(
                    "primary_multi_measurement_cardinality_quota"
                ),
                DAGGER1_PRIMARY_MULTI_MEASUREMENT_CARDINALITY_QUOTA,
            )
            and _exact_string_int_mapping(
                manifest.get(
                    "primary_multi_measurement_cardinality_count"
                ),
                DAGGER1_PRIMARY_MULTI_MEASUREMENT_CARDINALITY_QUOTA,
            )
            and dict(actual_primary_multi_cardinality)
            == DAGGER1_PRIMARY_MULTI_MEASUREMENT_CARDINALITY_QUOTA
        ),
        "reserve_multi_measurement_inventory": (
            _exact_string_int_mapping(
                manifest.get(
                    "reserve_multi_measurement_cardinality_inventory"
                ),
                DAGGER1_RESERVE_MULTI_MEASUREMENT_CARDINALITY_INVENTORY,
            )
            and dict(actual_reserve_multi_cardinality)
            == DAGGER1_RESERVE_MULTI_MEASUREMENT_CARDINALITY_INVENTORY
        ),
        "development_reservation": reservation_valid,
        "development_reservation_disjoint": reservation_valid
        and not (
            set(
                root
                for family_roots in reserved_roots_by_family.values()
                for root in family_roots
            )
            & actual_roots
        ),
        "development_reservation_cardinality": (
            _exact_string_int_mapping(
                manifest.get(
                    "withheld_for_development_multi_measurement_cardinality_inventory"
                ),
                DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_INVENTORY,
            )
        ),
        "exact_family_counts": normalized_selected == expected_selected,
        "primary_family_counts": dict(actual_primary_counts)
        == {key: value for key, value in normalized_primary.items() if value},
        "reserve_family_counts": dict(actual_reserve_counts)
        == {key: value for key, value in normalized_reserve.items() if value},
        "collection_schedule_contract": (
            schedule.get("contract") == DAGGER1_COLLECTION_SCHEDULE_CONTRACT
        ),
        "collection_schedule_fields_exact": set(schedule)
        == {
            "contract",
            "cohort_order",
            "reserve_family_priority",
            "priority_field",
            "order_field",
            "subcohort_field",
            "reserve_subcohort_order",
            "maximum_rollout_replicas_by_family",
        },
        "collection_schedule_cohorts": schedule.get("cohort_order")
        == ["primary", "reserve"],
        "collection_schedule_family_priority": schedule.get(
            "reserve_family_priority"
        )
        == list(DAGGER1_RESERVE_FAMILY_PRIORITY),
        "collection_schedule_fields": (
            schedule.get("priority_field") == "grouping.collection_priority"
            and schedule.get("order_field") == "grouping.collection_order"
            and schedule.get("subcohort_field")
            == "grouping.collection_subcohort"
            and schedule.get("reserve_subcohort_order")
            == ["base_reserve", DAGGER1_TOPUP_SUBCOHORT]
        ),
        "collection_schedule_replicas": isinstance(
            normalized_schedule_replicas, Mapping
        )
        and dict(normalized_schedule_replicas)
        == DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY,
        "scenario_count": manifest.get("scenario_count")
        == sum(expected_selected.values()),
        "physical_root_count": (
            manifest.get("physical_root_count") == sum(expected_selected.values())
        ),
        "actual_family_counts": dict(actual_counts) == expected_selected,
        "actual_scenario_count": len(scenarios)
        == sum(expected_selected.values()),
        "actual_unique_roots": len(actual_roots) == len(scenarios),
        "primary_root_set_sha256": manifest.get(
            "primary_physical_root_set_sha256"
        )
        == stable_json_sha256(sorted(actual_primary_roots)),
        "reserve_root_set_sha256": manifest.get(
            "reserve_physical_root_set_sha256"
        )
        == stable_json_sha256(sorted(actual_reserve_roots)),
        "training_root_set_sha256": manifest.get(
            "training_physical_root_set_sha256"
        )
        == stable_json_sha256(sorted(actual_roots)),
        "protected_root_overlap": manifest.get("protected_root_overlap") == [],
        "d0_raw_sha256": manifest.get("d0_raw_sha256") == _file_sha256(d0_raw_path),
        "d0_provenance_sha256": (
            manifest.get("d0_generation_provenance_sha256")
            == _file_sha256(d0_provenance_path)
        ),
        "d0_manifest_sha256": manifest.get("d0_manifest_sha256")
        == _file_sha256(d0_manifest_path),
        "forbidden_suite_sha256": (
            manifest.get("frozen_suite_sha256")
            == _file_sha256(forbidden_suite_path)
        ),
        "evaluation_policy_sha256": (
            manifest.get("evaluation_policy_sha256")
            == _file_sha256(evaluation_policy_path)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "scenario-builder manifest binding failed: " + ", ".join(failed)
        )


def validate_collection_output_paths(
    *,
    output: Path,
    all_output: Path | None,
    failed_collection_dir: Path | None = None,
    protected_paths: Sequence[Path],
) -> Path:
    """Fail before collection if an output can overwrite evidence or inputs."""

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    outputs = [output, manifest_path]
    if all_output is not None:
        outputs.append(all_output)
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("DAgger-1 output paths must be mutually distinct")
    protected = {path.resolve() for path in protected_paths}
    collisions = sorted(
        str(path)
        for path, resolved in zip(outputs, resolved_outputs)
        if resolved in protected
    )
    if collisions:
        raise ValueError(
            "DAgger-1 outputs alias protected input/evidence paths: "
            + ", ".join(collisions)
        )
    if failed_collection_dir is not None:
        failure_path = Path(failed_collection_dir)
        resolved_failure = failure_path.resolve()
        related_paths = [*resolved_outputs, *protected]
        if any(
            resolved_failure == path
            or resolved_failure in path.parents
            or path in resolved_failure.parents
            for path in related_paths
        ):
            raise ValueError(
                "DAgger-1 failed-collection directory must be separate from "
                "production outputs and protected evidence"
            )
    else:
        failure_path = None
    existing = sorted(
        str(path)
        for path in [*outputs, *([failure_path] if failure_path is not None else [])]
        if path.exists() or path.is_symlink()
    )
    if existing:
        raise FileExistsError(
            "DAgger-1 refuses to overwrite existing outputs: "
            + ", ".join(existing)
        )
    return manifest_path


def validate_export_rows_truth_free(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject any dynamic oracle-truth leak before writing either D1 output."""

    violations: list[str] = []
    for index, row in enumerate(rows):
        violations.extend(
            f"$[{index}]{path[1:]}"
            for path in _forbidden_collection_paths(row)
        )
    if violations:
        raise RuntimeError(
            "DAgger-1 export rows contain private oracle truth: "
            + ", ".join(sorted(violations)[:20])
        )


def _load_callable(spec: str, *, field: str) -> Callable[..., Any]:
    module_name, separator, attribute_path = str(spec).strip().partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"{field} must use MODULE:ATTRIBUTE syntax")
    value: Any = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{field} must resolve to a callable")
    return value


def _call_factory(
    factory: Callable[..., Any],
    *,
    seed: int,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> Any:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "seed" in parameters or accepts_kwargs:
        kwargs["seed"] = seed
    if "rng" in parameters or accepts_kwargs:
        kwargs["rng"] = random.Random(seed)
    if model_id is not None and ("model_id" in parameters or accepts_kwargs):
        kwargs["model_id"] = model_id
    if model_revision is not None and (
        "model_revision" in parameters or accepts_kwargs
    ):
        kwargs["model_revision"] = model_revision
    return factory(**kwargs)


def _callable_source_binding(value: Callable[..., Any], spec: str) -> dict[str, str]:
    source = inspect.getsourcefile(value)
    if source is None:
        raise ValueError(f"factory {spec} has no inspectable source file")
    path = Path(source).resolve()
    return {"import_spec": spec, "source_sha256": file_sha256(path)}


def validate_training_learner_seed(
    *, model_id: str, model_revision: str
) -> dict[str, Any]:
    """Validate and name the exact local adapter used to collect training D1.

    Production DAgger-1 collection is a continuation from a concrete learner,
    not generic inference from a compatible model identifier.  Resolve the
    local adapter before any rollout and bind its canonical tree digest into
    the collection manifest.
    """

    revision = str(model_revision).strip().lower()
    if _ADAPTER_TREE_REVISION.fullmatch(revision) is None:
        raise ValueError(
            "DAgger-1 training collection requires --model-revision to be "
            "an exact 64-hex adapter tree SHA-256"
        )

    requested = Path(str(model_id).strip()).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            "DAgger-1 training collection requires --model-id to be an "
            "absolute local adapter directory"
        )
    try:
        adapter_path = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"DAgger-1 learner adapter directory does not exist: {requested}"
        ) from exc
    if not adapter_path.is_dir():
        raise ValueError(
            f"DAgger-1 learner adapter is not a directory: {adapter_path}"
        )

    inspection = inspect_release_checkpoint(adapter_path)
    actual_revision = str(inspection.get("tree_sha256") or "").lower()
    if actual_revision != revision:
        raise ValueError(
            "DAgger-1 learner adapter tree digest mismatch: "
            f"expected {revision}, computed {actual_revision or '<missing>'}"
        )
    inspected_path = Path(str(inspection.get("path") or ""))
    if not inspected_path.is_absolute() or inspected_path != adapter_path:
        raise ValueError(
            "DAgger-1 learner adapter inspection did not preserve its "
            "resolved absolute path"
        )

    return {
        "role": "learner_seed_only",
        "collection_model_id": str(adapter_path),
        "collection_model_revision": revision,
        "adapter_tree_sha256": actual_revision,
        "adapter_file_count": int(inspection.get("file_count") or 0),
        "adapter_total_bytes": int(inspection.get("total_bytes") or 0),
    }


def validate_collection_pass(*, collection_pass: str, beta: float) -> dict[str, Any]:
    """Validate the review-prescribed diagnostic or mixed-policy beta."""
    pass_name = str(collection_pass).strip()
    beta_value = float(beta)
    if pass_name == "diagnostic":
        passed = beta_value == 0.0
        expected = {"minimum": 0.0, "maximum": 0.0}
    elif pass_name == "training":
        passed = 0.25 <= beta_value <= 0.5
        expected = {"minimum": 0.25, "maximum": 0.5}
    else:
        raise ValueError("collection_pass must be diagnostic or training")
    report = {
        "collection_pass": pass_name,
        "observed_beta": beta_value,
        "expected_beta": expected,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            f"{pass_name} DAgger-1 pass requires beta in "
            f"[{expected['minimum']}, {expected['maximum']}], got {beta_value}"
        )
    return report


def _scenario_grouping(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    grouping = scenario.get("grouping")
    return grouping if isinstance(grouping, Mapping) else scenario


def dagger1_rollout_seed(
    *, seed: int, physical_root_fingerprint: str, replica: int
) -> int:
    """Return a root-local mixture seed that is invariant to batch boundaries."""

    digest = stable_json_sha256(
        {
            "contract": "dagger1_root_replica_beta_rng_v1",
            "seed": int(seed),
            "physical_root_fingerprint": str(physical_root_fingerprint),
            "replica": int(replica),
        }
    )
    return int(digest[:16], 16)


def dagger1_rollout_batches(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    collection_pass: str,
) -> list[dict[str, Any]]:
    """Build the finite reviewed primary/reserve/repeat episode schedule."""

    if collection_pass not in {"diagnostic", "training"}:
        raise ValueError("collection_pass must be diagnostic or training")
    materialized = [dict(scenario) for scenario in scenarios]
    decorated: list[tuple[int, str, int, str, dict[str, Any]]] = []
    observed_orders: set[int] = set()
    for index, scenario in enumerate(materialized):
        grouping = _scenario_grouping(scenario)
        cohort = str(grouping.get("collection_cohort") or "").strip()
        family = str(grouping.get("scenario_family") or "").strip()
        raw_priority = grouping.get("collection_priority")
        raw_order = grouping.get("collection_order")
        if cohort not in {"primary", "reserve"}:
            raise ValueError(
                f"DAgger-1 scenario {index} has invalid collection_cohort"
            )
        if family not in DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY:
            raise ValueError(
                f"DAgger-1 scenario {index} has unsupported schedule family"
            )
        if (
            isinstance(raw_priority, bool)
            or not isinstance(raw_priority, int)
            or raw_priority < 0
        ):
            raise ValueError(
                f"DAgger-1 scenario {index} has invalid collection_priority"
            )
        if (
            isinstance(raw_order, bool)
            or not isinstance(raw_order, int)
            or raw_order < 0
            or raw_order in observed_orders
        ):
            raise ValueError(
                f"DAgger-1 scenario {index} has invalid/duplicate collection_order"
            )
        expected_priority = (
            0
            if cohort == "primary"
            else DAGGER1_RESERVE_FAMILY_PRIORITY.index(family) + 1
        )
        if raw_priority != expected_priority:
            raise ValueError(
                f"DAgger-1 scenario {index} collection priority is inconsistent"
            )
        observed_orders.add(raw_order)
        decorated.append((raw_order, cohort, raw_priority, family, scenario))
    if observed_orders != set(range(len(materialized))):
        raise ValueError("DAgger-1 collection_order must be contiguous from zero")
    decorated.sort(key=lambda item: item[0])

    def batch(
        batch_id: str,
        *,
        phase: str,
        replica: int,
        selected: Sequence[tuple[int, str, int, str, dict[str, Any]]],
    ) -> dict[str, Any] | None:
        if not selected:
            return None
        return {
            "batch_id": batch_id,
            "phase": phase,
            "replica": int(replica),
            "scenario_orders": [item[0] for item in selected],
            "scenarios": [item[4] for item in selected],
        }

    primary = [item for item in decorated if item[1] == "primary"]
    if not primary:
        raise ValueError("DAgger-1 schedule has no primary scenarios")
    batches: list[dict[str, Any]] = []
    primary_batch = batch(
        "primary-r0", phase="primary", replica=0, selected=primary
    )
    assert primary_batch is not None
    batches.append(primary_batch)
    if collection_pass == "diagnostic":
        return batches

    for priority, family in enumerate(DAGGER1_RESERVE_FAMILY_PRIORITY, start=1):
        reserve = [
            item
            for item in decorated
            if item[1] == "reserve"
            and item[2] == priority
            and item[3] == family
        ]
        reserve_batch = batch(
            f"reserve-{priority}-{family}-r0",
            phase="reserve",
            replica=0,
            selected=reserve,
        )
        if reserve_batch is not None:
            batches.append(reserve_batch)

    maximum_replica = max(DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY.values())
    for replica in range(1, maximum_replica):
        for family in DAGGER1_RESERVE_FAMILY_PRIORITY:
            if DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY[family] <= replica:
                continue
            repeated = [item for item in decorated if item[3] == family]
            repeat_batch = batch(
                f"repeat-{family}-r{replica}",
                phase="repeat",
                replica=replica,
                selected=repeated,
            )
            if repeat_batch is not None:
                batches.append(repeat_batch)
    return batches


def _decorate_rollout_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_id: str,
    collection_order: int,
    replica: int,
    rollout_seed: int,
) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for row in rows:
        item = copy.deepcopy(dict(row))
        original_id = str(item.get("example_id") or "").strip()
        if not original_id:
            raise ValueError("DAgger-1 rollout row lacks an example_id")
        item["base_example_id"] = original_id
        item["example_id"] = (
            f"{original_id}__order{int(collection_order)}"
            f"__replica{int(replica)}"
        )
        item["collection_batch_id"] = str(batch_id)
        item["collection_order"] = int(collection_order)
        item["collection_rollout_replica"] = int(replica)
        item["collection_rollout_seed"] = int(rollout_seed)
        decorated.append(item)
    return decorated


def _rollout_episode_disposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_id: str,
    collection_order: int,
    replica: int,
    max_steps: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "batch_id": batch_id,
            "collection_order": int(collection_order),
            "replica": int(replica),
            "disposition": "missing_episode_rows",
            "environment_terminal": False,
            "passed": False,
        }
    steps = [int(row.get("step", -1)) for row in rows]
    contiguous = steps == list(range(len(rows)))
    final = rows[-1]
    terminal_outcome = final.get("terminal_outcome")
    if terminal_outcome in {"resolved", "operator_escalation"}:
        disposition = str(terminal_outcome)
        environment_terminal = True
    elif contiguous and len(rows) == int(max_steps):
        disposition = "horizon_truncated"
        environment_terminal = False
    else:
        disposition = "unknown_incomplete"
        environment_terminal = False
    root = str(final.get("physical_root_fingerprint") or "").strip()
    return {
        "batch_id": str(batch_id),
        "collection_order": int(collection_order),
        "replica": int(replica),
        "scenario_id": str(final.get("scenario_id") or ""),
        "physical_root_fingerprint": root,
        "scenario_family": str(final.get("scenario_family") or "unknown"),
        "row_count": len(rows),
        "steps": steps,
        "steps_contiguous_from_zero": contiguous,
        "terminal_outcome": terminal_outcome,
        "disposition": disposition,
        "environment_terminal": environment_terminal,
        "passed": bool(
            root
            and contiguous
            and disposition
            in {"resolved", "operator_escalation", "horizon_truncated"}
        ),
    }


def dagger1_rollout_disposition_matrix(
    episode_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    seen_episode_keys: set[tuple[str, int]] = set()
    duplicate_episode_keys: list[str] = []
    for report in episode_reports:
        root = str(report.get("physical_root_fingerprint") or "")
        replica = int(report.get("replica") or 0)
        episode_key = (root, replica)
        if episode_key in seen_episode_keys:
            duplicate_episode_keys.append(f"{root}:replica{replica}")
        seen_episode_keys.add(episode_key)
        family = str(report.get("scenario_family") or "unknown")
        entry = by_family.setdefault(
            family,
            {
                "episodes": 0,
                "distinct_physical_roots": set(),
                "disposition_counts": Counter(),
                "physical_roots_by_disposition": {},
            },
        )
        disposition = str(report.get("disposition") or "unknown")
        entry["episodes"] += 1
        if root:
            entry["distinct_physical_roots"].add(root)
            entry["physical_roots_by_disposition"].setdefault(
                disposition, set()
            ).add(root)
        entry["disposition_counts"][disposition] += 1
    normalized: dict[str, dict[str, Any]] = {}
    for family, entry in sorted(by_family.items()):
        normalized[family] = {
            "episodes": int(entry["episodes"]),
            "distinct_physical_roots": len(entry["distinct_physical_roots"]),
            "physical_root_fingerprints": sorted(
                entry["distinct_physical_roots"]
            ),
            "disposition_counts": dict(
                sorted(entry["disposition_counts"].items())
            ),
            "physical_roots_by_disposition": {
                disposition: sorted(roots)
                for disposition, roots in sorted(
                    entry["physical_roots_by_disposition"].items()
                )
            },
        }
    malformed = [
        dict(report)
        for report in episode_reports
        if report.get("passed") is not True
    ]
    disposition_counts = Counter(
        str(report.get("disposition") or "unknown")
        for report in episode_reports
    )
    environment_terminal_episodes = sum(
        report.get("environment_terminal") is True
        for report in episode_reports
    )
    return {
        "contract": DAGGER1_ROLLOUT_MATRIX_CONTRACT,
        "episodes": len(episode_reports),
        "environment_terminal_episodes": environment_terminal_episodes,
        "all_environment_terminal": (
            environment_terminal_episodes == len(episode_reports)
        ),
        "horizon_truncated_episodes": sum(
            report.get("disposition") == "horizon_truncated"
            for report in episode_reports
        ),
        "duplicate_episode_keys": sorted(duplicate_episode_keys),
        "malformed_or_unknown_episodes": malformed,
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "by_family": normalized,
        # Training may legitimately retain horizon-truncated learner states.
        # Passing means every executed episode has one explicit disposition;
        # it deliberately does not claim that every environment terminated.
        "workflow_disposition_complete": (
            not duplicate_episode_keys and not malformed
        ),
        "passed": not duplicate_episode_keys and not malformed,
    }


def _selection_row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    example_id = str(row.get("example_id") or "").strip()
    return (
        stable_json_sha256(
            {
                "contract": DAGGER1_COLLECTION_SELECTION_CONTRACT,
                "example_id": example_id,
                "physical_root_fingerprint": row.get(
                    "physical_root_fingerprint"
                ),
                "recovery_stratum": row.get("recovery_stratum"),
                "targeted_state_cells": sorted(dagger1_targeted_state_cells(row)),
            }
        ),
        example_id,
    )


def _selection_groups(row: Mapping[str, Any]) -> frozenset[str]:
    groups = {
        f"targeted_state_cell:{cell}"
        for cell in dagger1_targeted_state_cells(row)
        if cell in DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS
    }
    stratum = str(row.get("recovery_stratum") or "").strip()
    if stratum in DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS:
        groups.add(f"recovery_stratum:{stratum}")
    return frozenset(groups)


def select_dagger1_collection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_min_rows: int = DEFAULT_TARGET_MIN_ROWS,
    target_max_rows: int = DEFAULT_TARGET_MAX_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a stable, no-replacement natural subset while retaining floors."""

    minimum = int(target_min_rows)
    maximum = int(target_max_rows)
    if minimum < 1 or maximum < minimum:
        raise ValueError("target row range must satisfy 1 <= minimum <= maximum")
    # Selection is read-only.  Keep the potentially large policy observations
    # shared while scoring, then deep-copy only the bounded published subset.
    materialized = [dict(row) for row in rows]
    example_ids = [str(row.get("example_id") or "").strip() for row in materialized]
    missing_example_ids = sum(not value for value in example_ids)
    duplicate_example_ids = sorted(
        example_id
        for example_id, count in Counter(example_ids).items()
        if example_id and count > 1
    )
    roots = [
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in materialized
    ]
    missing_physical_roots = sum(not value for value in roots)
    order = sorted(
        range(len(materialized)),
        key=lambda index: _selection_row_key(materialized[index]),
    )
    groups_by_index = {index: _selection_groups(materialized[index]) for index in order}
    root_by_index = {index: roots[index] for index in order}
    required_floors = {
        **{
            f"targeted_state_cell:{name}": int(floor)
            for name, floor in sorted(
                DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS.items()
            )
        },
        **{
            f"recovery_stratum:{name}": int(floor)
            for name, floor in sorted(
                DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS.items()
            )
        },
    }
    # Reservation and gating use different floor sets on purpose.  The
    # incidence-dependent strata are reserved so every attainable natural root
    # survives selection, but they are not gated: the complete 477-episode
    # schedule produced three roots each against a floor of ten, so an absolute
    # natural floor fails a learner for making fewer mistakes.  Their competence
    # guarantee moves to the probe and combined floors.  Every other floor stays
    # binding here.
    gated_floors = {
        group: floor
        for group, floor in required_floors.items()
        if group.removeprefix("recovery_stratum:")
        not in DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA
        or not group.startswith("recovery_stratum:")
    }
    report_only_groups = {
        group for group in required_floors if group not in gated_floors
    }
    candidate_roots_by_group = {
        group: {
            root_by_index[index]
            for index in order
            if root_by_index[index] and group in groups_by_index[index]
        }
        for group in required_floors
    }
    selected: set[int] = set()
    selected_roots_by_group: dict[str, set[str]] = {
        group: set() for group in required_floors
    }
    selected_rows_by_root: Counter[str] = Counter()
    # A group whose candidate pool is smaller than its release floor is
    # intrinsically infeasible for this collection.  Reserve only what the
    # candidates can actually support, so an infeasible group cannot abandon
    # the reservation pass for every group behind it.  The release gate still
    # fails through ``candidate_root_group_shortfalls`` below: this preserves
    # scarce evidence, it does not admit an under-supported corpus.
    attainable_floors = {
        group: min(int(floor), len(candidate_roots_by_group[group]))
        for group, floor in required_floors.items()
    }
    exhausted: set[str] = set()

    while True:
        unmet = [
            group
            for group, floor in attainable_floors.items()
            if group not in exhausted
            and len(selected_roots_by_group[group]) < floor
        ]
        if not unmet:
            break
        focus = min(
            unmet,
            key=lambda group: (
                len(candidate_roots_by_group[group]) - required_floors[group],
                group,
            ),
        )
        choices = [
            index
            for index in order
            if index not in selected
            and focus in groups_by_index[index]
            and root_by_index[index]
            and root_by_index[index] not in selected_roots_by_group[focus]
        ]
        if not choices:
            # This group cannot contribute another distinct root.  Retire it
            # and continue with the remaining groups rather than ending the
            # reservation pass, which previously discarded attainable support
            # for every group ordered behind an infeasible one.
            exhausted.add(focus)
            continue

        def choice_key(index: int) -> tuple[Any, ...]:
            newly_supported = sum(
                group in groups_by_index[index]
                and root_by_index[index] not in selected_roots_by_group[group]
                for group in unmet
            )
            return (
                -newly_supported,
                selected_rows_by_root[root_by_index[index]],
                _selection_row_key(materialized[index]),
            )

        chosen = min(choices, key=choice_key)
        selected.add(chosen)
        selected_rows_by_root[root_by_index[chosen]] += 1
        for group in groups_by_index[chosen]:
            if group in selected_roots_by_group:
                selected_roots_by_group[group].add(root_by_index[chosen])

    reserved_row_count = len(selected)
    target_size = min(maximum, len(materialized))
    while len(selected) < target_size:
        choices = [index for index in order if index not in selected]
        if not choices:
            break
        chosen = min(
            choices,
            key=lambda index: (
                selected_rows_by_root[root_by_index[index]],
                _selection_row_key(materialized[index]),
            ),
        )
        selected.add(chosen)
        selected_rows_by_root[root_by_index[chosen]] += 1
    # Reservation-stage support is not the published support: the fill stage
    # adds rows without touching ``selected_roots_by_group``.  Recount over the
    # final selection so the reported per-group support agrees with
    # ``selected_independent_root_support``, and so the loss check below is
    # evaluated against what is actually published rather than what was
    # reserved.
    final_roots_by_group: dict[str, set[str]] = {
        group: set() for group in required_floors
    }
    for index in selected:
        root = root_by_index[index]
        if not root:
            continue
        for group in groups_by_index[index]:
            if group in final_roots_by_group:
                final_roots_by_group[group].add(root)
    selected_rows = [
        copy.deepcopy(materialized[index])
        for index in sorted(
            selected,
            key=lambda index: _selection_row_key(materialized[index]),
        )
    ]
    selected_support = audit_dagger1_independent_root_support(
        selected_rows,
        recovery_stratum_minimum_distinct_roots=(
            DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS
        ),
    )
    selected_targeted_coverage = targeted_state_coverage(selected_rows)
    candidate_shortfalls = {
        group: {
            "distinct_physical_roots": len(candidate_roots_by_group[group]),
            "minimum_distinct_physical_roots": floor,
            "root_shortfall": max(
                floor - len(candidate_roots_by_group[group]), 0
            ),
        }
        for group, floor in sorted(gated_floors.items())
        if len(candidate_roots_by_group[group]) < floor
    }
    # Natural incidence for the ungated pair is recorded rather than gated, so a
    # reader never has to infer it from the absence of a shortfall entry.
    natural_incidence_report_only = {
        group: {
            "candidate_distinct_physical_roots": len(
                candidate_roots_by_group[group]
            ),
            "selected_distinct_physical_roots": len(
                final_roots_by_group.get(group, set())
            ),
            "reference_floor": int(required_floors[group]),
            "gated": False,
        }
        for group in sorted(report_only_groups)
    }
    # Selection must never lose root support the candidate pool could sustain.
    # This is separate from ``candidate_root_group_shortfalls``: that reports
    # what the collection could not generate, this reports what selection threw
    # away.
    attainable_root_loss = {
        group: {
            "attainable_root_target": attainable_floors[group],
            "candidate_distinct_physical_roots": len(candidate_roots_by_group[group]),
            "selected_distinct_physical_roots": len(final_roots_by_group[group]),
            "root_loss": attainable_floors[group] - len(final_roots_by_group[group]),
        }
        for group in sorted(required_floors)
        if len(final_roots_by_group[group]) < attainable_floors[group]
    }
    row_target_passed = minimum <= len(selected_rows) <= maximum
    passed = bool(
        row_target_passed
        and not missing_example_ids
        and not duplicate_example_ids
        and not missing_physical_roots
        and not candidate_shortfalls
        and not attainable_root_loss
        and selected_support.get("passed") is True
        and selected_targeted_coverage.get("passed") is True
    )
    report = {
        "contract": DAGGER1_COLLECTION_SELECTION_CONTRACT,
        "candidate_rows": len(materialized),
        "candidate_example_id_set_sha256": stable_json_sha256(
            sorted(example_ids)
        ),
        "candidate_distinct_physical_roots": len(set(root for root in roots if root)),
        "target_min_rows": minimum,
        "target_max_rows": maximum,
        "selected_rows": len(selected_rows),
        "selected_example_id_sequence_sha256": stable_json_sha256(
            [str(row.get("example_id") or "") for row in selected_rows]
        ),
        "selected_distinct_physical_roots": len(
            {root for root in selected_rows_by_root if root}
        ),
        "discarded_eligible_rows": len(materialized) - len(selected_rows),
        "selection_applied": len(materialized) > maximum,
        "missing_example_id_rows": missing_example_ids,
        "duplicate_example_ids": duplicate_example_ids,
        "missing_physical_root_rows": missing_physical_roots,
        "required_root_group_floors": required_floors,
        "gated_root_group_floors": dict(sorted(gated_floors.items())),
        "natural_incidence_report_only": natural_incidence_report_only,
        "candidate_distinct_roots_by_group": {
            group: len(candidate_roots_by_group[group])
            for group in sorted(required_floors)
        },
        "attainable_root_targets": dict(sorted(attainable_floors.items())),
        "selected_distinct_roots_by_group": {
            group: len(final_roots_by_group[group])
            for group in sorted(required_floors)
        },
        "reserved_distinct_roots_by_group": {
            group: len(selected_roots_by_group[group])
            for group in sorted(required_floors)
        },
        "selected_attainable_root_loss": attainable_root_loss,
        "minimum_rows_needed_for_floor_reservation": reserved_row_count,
        "candidate_root_group_shortfalls": candidate_shortfalls,
        "selected_independent_root_support": selected_support,
        "selected_targeted_state_coverage": selected_targeted_coverage,
        "row_target_passed": row_target_passed,
        "passed": passed,
    }
    return selected_rows, report


def dagger1_production_row_target_contract(
    *, target_min_rows: int, target_max_rows: int
) -> dict[str, Any]:
    """Bind exploratory CLI bounds to the reviewed production bounds."""

    configured_minimum = int(target_min_rows)
    configured_maximum = int(target_max_rows)
    passed = bool(
        configured_minimum == DEFAULT_TARGET_MIN_ROWS
        and configured_maximum == DEFAULT_TARGET_MAX_ROWS
    )
    return {
        "contract": DAGGER1_PRODUCTION_ROW_TARGET_CONTRACT,
        "required_target_min_rows": DEFAULT_TARGET_MIN_ROWS,
        "required_target_max_rows": DEFAULT_TARGET_MAX_ROWS,
        "configured_target_min_rows": configured_minimum,
        "configured_target_max_rows": configured_maximum,
        "exploratory_override": not passed,
        "passed": passed,
    }


def recommended_collection_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_min_rows: int = DEFAULT_TARGET_MIN_ROWS,
    target_max_rows: int = DEFAULT_TARGET_MAX_ROWS,
) -> dict[str, Any]:
    """Report the recommended first-round row and recovery-strata coverage."""
    minimum = int(target_min_rows)
    maximum = int(target_max_rows)
    if minimum < 1 or maximum < minimum:
        raise ValueError("target row range must satisfy 1 <= minimum <= maximum")
    observed = len(rows)
    strata = Counter(str(row.get("recovery_stratum") or "unclassified") for row in rows)
    missing = sorted(RECOMMENDED_DAGGER1_RECOVERY_STRATA - set(strata))
    row_count_passed = minimum <= observed <= maximum
    strata_passed = not missing
    return {
        "recommended_row_target": {
            "minimum": minimum,
            "maximum": maximum,
            "observed": observed,
            "passed": row_count_passed,
        },
        "recommended_recovery_strata": sorted(
            RECOMMENDED_DAGGER1_RECOVERY_STRATA
        ),
        "observed_recovery_strata": dict(sorted(strata.items())),
        "missing_recommended_recovery_strata": missing,
        "recovery_strata_passed": strata_passed,
        "passed": row_count_passed and strata_passed,
    }


def targeted_state_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit independent-root support for the review's first D1 state cells."""
    cells: dict[str, set[str]] = {
        name: set()
        for name in DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS
    }
    rows_missing_physical_root: Counter[str] = Counter()
    for row in rows:
        root = str(row.get("physical_root_fingerprint") or "").strip()
        for cell in dagger1_targeted_state_cells(row):
            if root:
                cells.setdefault(cell, set()).add(root)
            else:
                rows_missing_physical_root[cell] += 1
    counts = {name: len(roots) for name, roots in sorted(cells.items())}
    shortfalls = {
        name: {
            "distinct_physical_roots": counts.get(name, 0),
            "minimum_distinct_physical_roots": minimum,
            "rows_missing_physical_root": rows_missing_physical_root[name],
        }
        for name, minimum in sorted(
            DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS.items()
        )
        if counts.get(name, 0) < minimum or rows_missing_physical_root[name]
    }
    return {
        "contract": "dagger1_targeted_state_independent_root_support_v2",
        "minimum_distinct_physical_roots_by_cell": dict(
            sorted(DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS.items())
        ),
        "distinct_physical_roots_by_cell": counts,
        "rows_missing_physical_root_by_cell": dict(
            sorted(rows_missing_physical_root.items())
        ),
        "shortfalls": shortfalls,
        "missing_cells": sorted(shortfalls),
        "passed": not shortfalls,
    }


def evaluate_dagger1_collection_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    d0_training_rows: Sequence[Mapping[str, Any]],
    target_min_rows: int = DEFAULT_TARGET_MIN_ROWS,
    target_max_rows: int = DEFAULT_TARGET_MAX_ROWS,
    rollout_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute every strict gate at a whole-episode batch boundary."""

    all_rows = [dict(row) for row in rows]
    candidate_rows = [
        row for row in all_rows if row.get("production_label_eligible") is True
    ]
    selected_rows, selection_report = select_dagger1_collection_rows(
        candidate_rows,
        target_min_rows=target_min_rows,
        target_max_rows=target_max_rows,
    )
    collection_gate = recommended_collection_gate(
        selected_rows,
        target_min_rows=target_min_rows,
        target_max_rows=target_max_rows,
    )
    targeted_coverage = targeted_state_coverage(selected_rows)
    # Natural floors only.  The incidence-dependent pair is carried by the probe
    # and combined floors at aggregate ingestion; gating it here would fail the
    # strict collection for a learner that errs less often.
    independent_root_support = audit_dagger1_independent_root_support(
        selected_rows,
        recovery_stratum_minimum_distinct_roots=(
            DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS
        ),
    )
    truth_audit_quarantine = (
        summarize_dagger1_offline_teacher_target_quarantine(all_rows)
    )
    if selected_rows:
        replay_capacity = dagger1_replay_capacity_report(
            d0_training_rows,
            selected_rows,
        )
    else:
        replay_capacity = {
            "schema_version": 1,
            "contract": "dagger1_duplicate_and_root_limited_capacity_v1",
            "applicable": False,
            "reason": "no_selected_d1_recovery_rows",
            "passed": False,
        }
    matrix = dict(rollout_matrix or {"passed": True})
    passed = bool(
        selection_report.get("passed") is True
        and collection_gate.get("passed") is True
        and targeted_coverage.get("passed") is True
        and independent_root_support.get("passed") is True
        and truth_audit_quarantine.get("passed") is True
        and replay_capacity.get("passed") is True
        and matrix.get("passed") is True
    )
    failed_gate_names = failed_strict_collection_gate_names(
        collection_gate=collection_gate,
        targeted_coverage=targeted_coverage,
        independent_root_support=independent_root_support,
        truth_audit_quarantine=truth_audit_quarantine,
        selection_report=selection_report,
        round1_replay_capacity=replay_capacity,
        rollout_matrix=matrix,
    )
    return {
        "candidate_rows": len(candidate_rows),
        "selected_rows": selected_rows,
        "deterministic_collection_selection": selection_report,
        "recommended_collection_gate": collection_gate,
        "targeted_state_coverage": targeted_coverage,
        "independent_root_support": independent_root_support,
        "offline_teacher_target_quarantine_summary": (
            truth_audit_quarantine
        ),
        "round1_replay_capacity": replay_capacity,
        "rollout_disposition_matrix": matrix,
        "failed_gate_names": failed_gate_names,
        "passed": passed,
    }


def collect_dagger1_rollout_schedule(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    collection_pass: str,
    seed: int,
    max_steps: int,
    collect_episode: Callable[
        [Mapping[str, Any], int, int, str, int],
        Sequence[Mapping[str, Any]],
    ],
    checkpoint: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]], Mapping[str, Any]
    ]
    | None = None,
    analysis_only_complete_schedule: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Execute the finite schedule until a whole-batch terminal decision.

    ``analysis_only_complete_schedule`` runs every predeclared batch to
    exhaustion instead of stopping at the first quarantine or the first passing
    checkpoint.  It exists to answer coverage questions that an early-stopped
    run cannot: the DAgger-1 round-2 collection stopped after 2 of 6 batches at
    151 of 477 episodes, so its root-support and replay-capacity shortfalls were
    measured under censorship and could not be attributed to the schedule.

    The mode never produces production data.  Quarantines are still recorded and
    still mark the run failed; the report carries ``analysis_only`` and
    ``training_eligible: False`` so aggregate ingestion refuses the outputs.
    """

    batches = dagger1_rollout_batches(
        scenarios, collection_pass=collection_pass
    )
    rows: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    executed_batch_ids: list[str] = []
    gate_snapshots: list[dict[str, Any]] = []
    last_checkpoint: dict[str, Any] | None = None
    stopped_after_batch: str | None = None
    terminal_failure: dict[str, Any] | None = None
    for batch in batches:
        batch_id = str(batch["batch_id"])
        replica = int(batch["replica"])
        scenarios_in_batch = list(batch["scenarios"])
        orders = list(batch["scenario_orders"])
        for scenario, collection_order in zip(
            scenarios_in_batch, orders, strict=True
        ):
            grouping = _scenario_grouping(scenario)
            root = str(
                grouping.get("physical_root_fingerprint") or ""
            ).strip()
            if not root:
                raise ValueError("scheduled DAgger-1 scenario lacks a physical root")
            rollout_seed = dagger1_rollout_seed(
                seed=seed,
                physical_root_fingerprint=root,
                replica=replica,
            )
            episode_rows = _decorate_rollout_rows(
                collect_episode(
                    scenario,
                    replica,
                    rollout_seed,
                    batch_id,
                    int(collection_order),
                ),
                batch_id=batch_id,
                collection_order=int(collection_order),
                replica=replica,
                rollout_seed=rollout_seed,
            )
            rows.extend(episode_rows)
            episode_reports.append(
                _rollout_episode_disposition(
                    episode_rows,
                    batch_id=batch_id,
                    collection_order=int(collection_order),
                    replica=replica,
                    max_steps=max_steps,
                )
            )
        executed_batch_ids.append(batch_id)
        matrix = dagger1_rollout_disposition_matrix(episode_reports)
        if checkpoint is not None:
            last_checkpoint = dict(checkpoint(rows, matrix))
            truth_quarantine = last_checkpoint.get(
                "offline_teacher_target_quarantine_summary"
            )
            quarantined_rows = (
                int(truth_quarantine.get("quarantined_rows") or 0)
                if isinstance(truth_quarantine, Mapping)
                else 0
            )
            gate_snapshots.append(
                {
                    "batch_id": batch_id,
                    "visited_rows": len(rows),
                    "candidate_rows": int(
                        last_checkpoint.get("candidate_rows") or 0
                    ),
                    "selected_rows": len(
                        last_checkpoint.get("selected_rows") or []
                    ),
                    "failed_gate_names": list(
                        last_checkpoint.get("failed_gate_names") or []
                    ),
                    "offline_teacher_target_quarantined_rows": (
                        quarantined_rows
                    ),
                    "passed": last_checkpoint.get("passed") is True,
                }
            )
            if (
                collection_pass == "training"
                and quarantined_rows > 0
            ):
                # Record the first quarantine even in analysis mode: the run is
                # still a failure, it simply keeps executing so the remaining
                # batches' coverage contribution can be measured.
                if terminal_failure is None:
                    terminal_failure = {
                        "gate": "offline_teacher_target_quarantine_summary",
                        "reason": (
                            "strict_zero_quarantine_gate_is_cumulative_and_"
                            "irreversible"
                        ),
                        "quarantined_rows": quarantined_rows,
                    }
                    if analysis_only_complete_schedule:
                        # Only meaningful when execution continues past the
                        # stop; the production payload stays byte-identical.
                        terminal_failure["first_quarantined_batch"] = batch_id
                elif analysis_only_complete_schedule:
                    terminal_failure["quarantined_rows"] = quarantined_rows
                if not analysis_only_complete_schedule:
                    stopped_after_batch = batch_id
                    break
            if (
                collection_pass == "training"
                and last_checkpoint.get("passed") is True
                and not analysis_only_complete_schedule
            ):
                stopped_after_batch = batch_id
                break

    matrix = dagger1_rollout_disposition_matrix(episode_reports)
    planned_batch_ids = [str(batch["batch_id"]) for batch in batches]
    if analysis_only_complete_schedule:
        stopping_reason = "analysis_only_complete_schedule_exhausted"
    elif collection_pass == "diagnostic":
        stopping_reason = "diagnostic_primary_complete"
    elif terminal_failure is not None:
        stopping_reason = "irreversible_truth_audit_quarantine"
    elif stopped_after_batch is not None:
        stopping_reason = "strict_collection_gate_passed"
    else:
        stopping_reason = "reserve_exhausted"
    report = {
        "contract": DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
        "collection_pass": collection_pass,
        "root_local_rng_contract": "dagger1_root_replica_beta_rng_v1",
        "maximum_rollout_replicas_by_family": dict(
            sorted(DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY.items())
        ),
        "planned_batch_ids": planned_batch_ids,
        "executed_batch_ids": executed_batch_ids,
        "unexecuted_batch_ids": [
            batch_id
            for batch_id in planned_batch_ids
            if batch_id not in set(executed_batch_ids)
        ],
        "planned_episode_count": sum(
            len(batch["scenarios"]) for batch in batches
        ),
        "executed_episode_count": len(episode_reports),
        "gate_snapshots": gate_snapshots,
        "stopped_after_batch": stopped_after_batch,
        "stopping_reason": stopping_reason,
        "terminal_failure": terminal_failure,
        "workflow_terminal": True,
        "analysis_only": bool(analysis_only_complete_schedule),
        "training_eligible": not analysis_only_complete_schedule,
        "passed": (
            not analysis_only_complete_schedule
            and (
                collection_pass == "diagnostic"
                or stopping_reason == "strict_collection_gate_passed"
            )
        ),
    }
    return rows, matrix, report, last_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect production-safe DAgger-1 recovery labels on non-evaluation "
            "training roots."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--d0-aggregate-dir",
        type=Path,
        required=True,
        help="Current-commit D0 aggregate whose complete physical-root set is forbidden",
    )
    parser.add_argument(
        "--scenario-generator-report",
        type=Path,
        required=True,
        help=(
            "Round0ScenarioGenerator report proving source_partition='train'"
        ),
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        required=True,
        help="Fresh-root builder manifest binding input and generator report",
    )
    parser.add_argument(
        "--development-holdout",
        type=Path,
        help=(
            "Independent DAgger-1 development suite; required for the "
            "training pass and optional for diagnostics"
        ),
    )
    parser.add_argument(
        "--development-holdout-manifest",
        type=Path,
        help=(
            "Manifest byte-binding --development-holdout to this training "
            "scenario manifest"
        ),
    )
    parser.add_argument(
        "--development-holdout-generator-report",
        type=Path,
        help=(
            "Round0ScenarioGenerator report byte-bound by the development "
            "holdout manifest; required with the holdout inputs"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Training-eligible recovery rows for the training pass, or "
            "explicitly training-ineligible recovery rows for diagnostic"
        ),
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        help=(
            "Audit JSONL containing eligible and ineligible visited states; "
            "required for the training pass and optional for diagnostics"
        ),
    )
    parser.add_argument(
        "--forbidden-suite",
        type=Path,
        default=DEFAULT_FORBIDDEN_SUITE,
        help="Frozen evaluation suite whose physical roots are forbidden",
    )
    parser.add_argument(
        "--env-factory",
        default=DEFAULT_ENV_FACTORY_SPEC,
    )
    parser.add_argument(
        "--policy-factory",
        default=DEFAULT_POLICY_FACTORY_SPEC,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--collection-pass",
        choices=("diagnostic", "training"),
        default="diagnostic",
        help="Diagnostic requires beta=0; training requires beta in [0.25, 0.5]",
    )
    parser.add_argument("--target-min-rows", type=int, default=DEFAULT_TARGET_MIN_ROWS)
    parser.add_argument("--target-max-rows", type=int, default=DEFAULT_TARGET_MAX_ROWS)
    parser.add_argument(
        "--require-recommended-target",
        action="store_true",
        help="Fail unless eligible rows meet the recommended row and stratum target",
    )
    parser.add_argument(
        "--failed-collection-dir",
        type=Path,
        help=(
            "Atomic diagnostic-only evidence directory; required with "
            "--require-recommended-target and forbidden otherwise"
        ),
    )
    parser.add_argument(
        "--analysis-only-complete-schedule",
        action="store_true",
        help=(
            "Run every predeclared batch to exhaustion instead of stopping at "
            "the first quarantine, to measure what the complete schedule would "
            "supply. Never publishes production outputs; all results are "
            "training-ineligible and rejected by aggregate ingestion."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(list(argv) if argv is not None else None)
    production_row_target = dagger1_production_row_target_contract(
        target_min_rows=args.target_min_rows,
        target_max_rows=args.target_max_rows,
    )

    if args.analysis_only_complete_schedule:
        if args.collection_pass != "training":
            parser.error(
                "--analysis-only-complete-schedule is valid only for the "
                "training pass"
            )
        if args.failed_collection_dir is None:
            parser.error(
                "--analysis-only-complete-schedule requires "
                "--failed-collection-dir; the mode never publishes production "
                "outputs"
            )
        # Force the failure-evidence path so production outputs can never be
        # published from an analysis run, whatever the gates report.
        args.require_recommended_target = True
    if args.collection_pass == "diagnostic" and args.require_recommended_target:
        parser.error(
            "--require-recommended-target is valid only for the training pass"
        )
    if args.require_recommended_target and args.failed_collection_dir is None:
        parser.error(
            "--require-recommended-target requires --failed-collection-dir"
        )
    if args.failed_collection_dir is not None and not args.require_recommended_target:
        parser.error(
            "--failed-collection-dir is valid only with "
            "--require-recommended-target"
        )
    if (
        args.require_recommended_target
        and production_row_target["passed"] is not True
    ):
        parser.error(
            "strict production collection requires the reviewed row bounds "
            f"{DEFAULT_TARGET_MIN_ROWS}..{DEFAULT_TARGET_MAX_ROWS}; custom "
            "bounds are exploratory-only"
        )
    if args.collection_pass == "training" and args.all_output is None:
        parser.error(
            "training collection requires --all-output so truth-audit "
            "quarantines remain independently recomputable"
        )
    development_inputs = (
        args.development_holdout,
        args.development_holdout_manifest,
        args.development_holdout_generator_report,
    )
    if any(value is not None for value in development_inputs) and not all(
        value is not None for value in development_inputs
    ):
        parser.error(
            "--development-holdout, --development-holdout-manifest, and "
            "--development-holdout-generator-report must be provided together"
        )
    if args.collection_pass == "training" and args.development_holdout is None:
        parser.error(
            "training collection requires --development-holdout and "
            "--development-holdout-manifest and "
            "--development-holdout-generator-report"
        )
    learner_seed: dict[str, Any] | None = None
    if args.collection_pass == "training":
        try:
            learner_seed = validate_training_learner_seed(
                model_id=args.model_id,
                model_revision=args.model_revision,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        args.model_id = learner_seed["collection_model_id"]
        args.model_revision = learner_seed["collection_model_revision"]
    elif _IMMUTABLE_REVISION.fullmatch(args.model_revision) is None:
        parser.error("--model-revision must be a 40- or 64-character hex digest")
    if args.iteration < 1 or not 0.0 <= args.beta < 1.0:
        parser.error("DAgger-1 requires --iteration >= 1 and 0 <= --beta < 1")
    if args.max_steps != 24:
        parser.error("production DAgger-1 collection requires --max-steps 24")
    try:
        beta_contract = validate_collection_pass(
            collection_pass=args.collection_pass,
            beta=args.beta,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.env_factory != DEFAULT_ENV_FACTORY_SPEC:
        parser.error("DAgger-1 requires the reviewed production environment factory")
    if args.policy_factory != DEFAULT_POLICY_FACTORY_SPEC:
        parser.error("DAgger-1 requires the reviewed Gemma release policy factory")
    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError("DAgger-1 collection requires a clean committed source tree")
    scenarios = _load_json_or_jsonl(args.input)
    source_report = json.loads(
        args.scenario_generator_report.read_text(encoding="utf-8")
    )
    if not isinstance(source_report, Mapping):
        parser.error("--scenario-generator-report must contain a JSON object")
    validate_training_source_report(source_report)
    scenario_manifest = json.loads(
        args.scenario_manifest.read_text(encoding="utf-8")
    )
    if not isinstance(scenario_manifest, Mapping):
        parser.error("--scenario-manifest must contain a JSON object")
    frozen_roots = frozen_physical_roots(args.forbidden_suite)
    suite_sha256 = _file_sha256(args.forbidden_suite)
    policy_payload = json.loads(DEFAULT_EVALUATION_POLICY.read_text(encoding="utf-8"))
    suite_policy = (
        policy_payload.get("suite_policy")
        if isinstance(policy_payload, Mapping)
        else None
    )
    if not isinstance(suite_policy, Mapping) or (
        suite_policy.get("status") != "pinned"
        or suite_policy.get("approved_suite_sha256") != suite_sha256
    ):
        raise RuntimeError(
            "DAgger-1 forbidden suite does not match the pinned evaluation policy"
        )
    d0_raw_path = args.d0_aggregate_dir / "aggregate.raw.jsonl"
    d0_provenance_path = (
        args.d0_aggregate_dir / "aggregate.generation_provenance.json"
    )
    d0_manifest_path = args.d0_aggregate_dir / AGGREGATE_MANIFEST_FILENAME
    protected_paths = [
        args.input,
        args.scenario_generator_report,
        args.scenario_manifest,
        args.forbidden_suite,
        DEFAULT_EVALUATION_POLICY,
        d0_raw_path,
        d0_provenance_path,
        d0_manifest_path,
    ]
    if args.development_holdout is not None:
        protected_paths.extend(
            [
                args.development_holdout,
                args.development_holdout_manifest,
                args.development_holdout_generator_report,
            ]
        )
    output_manifest_path = validate_collection_output_paths(
        output=args.output,
        all_output=args.all_output,
        failed_collection_dir=args.failed_collection_dir,
        protected_paths=protected_paths,
    )
    if not (
        d0_raw_path.is_file()
        and d0_provenance_path.is_file()
        and d0_manifest_path.is_file()
    ):
        raise FileNotFoundError(
            "--d0-aggregate-dir must contain aggregate.raw.jsonl, "
            "aggregate.generation_provenance.json, and "
            "aggregate.manifest.json"
        )
    d0_rows = _load_json_or_jsonl(d0_raw_path)
    d0_roots = frozenset(
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in d0_rows
        if str(row.get("physical_root_fingerprint") or "").strip()
    )
    if not d0_roots:
        raise RuntimeError("D0 aggregate has no physical roots")
    d0_provenance = json.loads(d0_provenance_path.read_text(encoding="utf-8"))
    if not isinstance(d0_provenance, Mapping):
        raise RuntimeError("D0 aggregate provenance must be a JSON object")
    analysis_only = bool(args.analysis_only_complete_schedule)
    validate_d0_provenance_binding(
        d0_provenance,
        raw_path=d0_raw_path,
        source_state=source_state,
        analysis_only=analysis_only,
    )
    validate_scenario_builder_manifest(
        scenario_manifest,
        scenarios=scenarios,
        input_path=args.input,
        generator_report_path=args.scenario_generator_report,
        source_state=source_state,
        d0_raw_path=d0_raw_path,
        d0_provenance_path=d0_provenance_path,
        d0_manifest_path=d0_manifest_path,
        forbidden_suite_path=args.forbidden_suite,
        evaluation_policy_path=DEFAULT_EVALUATION_POLICY,
        analysis_only=analysis_only,
    )
    scenario_source_drift = source_commit_drift(
        scenario_manifest.get("source_state"), source_state
    )
    if scenario_source_drift is not None:
        print(
            "analysis-only source commit drift: "
            + json.dumps(scenario_source_drift, sort_keys=True)
        )
    development_roots: frozenset[str] = frozenset()
    if args.development_holdout is not None:
        development_roots = validate_development_holdout_binding(
            args.development_holdout,
            args.development_holdout_manifest,
            generator_report_path=(
                args.development_holdout_generator_report
            ),
            source_state=source_state,
            scenario_input_path=args.input,
            scenario_manifest_path=args.scenario_manifest,
            d0_raw_path=d0_raw_path,
            d0_provenance_path=d0_provenance_path,
            d0_manifest_path=d0_manifest_path,
            forbidden_suite_path=args.forbidden_suite,
            evaluation_policy_path=DEFAULT_EVALUATION_POLICY,
            require_model_selection_eligible=(
                args.collection_pass == "training"
            ),
            analysis_only=analysis_only,
        )
    if development_roots & (frozen_roots | d0_roots):
        raise RuntimeError(
            "DAgger-1 development roots overlap D0/frozen evaluation roots"
        )
    forbidden_roots = frozen_roots | d0_roots | development_roots
    validate_training_scenarios(scenarios, forbidden_roots=forbidden_roots)

    env_factory = _load_callable(args.env_factory, field="env_factory")
    policy_factory = _load_callable(args.policy_factory, field="policy_factory")
    env_factory_binding = _callable_source_binding(env_factory, args.env_factory)
    policy_factory_binding = _callable_source_binding(
        policy_factory, args.policy_factory
    )
    expert_source = inspect.getsourcefile(ExpertPolicyOracle)
    if expert_source is None:
        raise RuntimeError("ExpertPolicyOracle source is not inspectable")
    env = _call_factory(env_factory, seed=args.seed)
    if getattr(env, "production_dataset_mode", None) is not True:
        raise RuntimeError("DAgger-1 environment is not in production dataset mode")
    policy = _call_factory(
        policy_factory,
        seed=args.seed,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    expert = ExpertPolicyOracle(
        process_oracle=env.process_oracle,
        candidate_oracle=env.candidate_quality_oracle,
    )
    d0_training_rows = [
        row
        for row in d0_rows
        if row.get("dataset_split") == "train"
        and row.get("production_label_eligible") is True
    ]

    def collect_episode(
        scenario: Mapping[str, Any],
        replica: int,
        rollout_seed: int,
        batch_id: str,
        collection_order: int,
    ) -> Sequence[Mapping[str, Any]]:
        del replica, batch_id, collection_order
        return DaggerRolloutCollector(
            env=env,
            policy=policy,
            expert_oracle=expert,
            rng=random.Random(rollout_seed),
            supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            forbidden_physical_roots=forbidden_roots,
        ).collect_iteration(
            scenarios=[scenario],
            iteration=args.iteration,
            beta=args.beta,
            max_steps=args.max_steps,
            collection_role=args.collection_pass,
        )

    def checkpoint(
        visited_rows: Sequence[Mapping[str, Any]],
        rollout_matrix: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return evaluate_dagger1_collection_checkpoint(
            visited_rows,
            d0_training_rows=d0_training_rows,
            target_min_rows=args.target_min_rows,
            target_max_rows=args.target_max_rows,
            rollout_matrix=rollout_matrix,
        )

    rows, rollout_matrix, stopping_report, collection_checkpoint = (
        collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass=args.collection_pass,
            seed=args.seed,
            max_steps=args.max_steps,
            collect_episode=collect_episode,
            checkpoint=(
                checkpoint if args.collection_pass == "training" else None
            ),
            analysis_only_complete_schedule=(
                args.analysis_only_complete_schedule
            ),
        )
    )
    validate_export_rows_truth_free(rows)
    class_audit = audit_target_aware_state_classes(rows)
    if not class_audit["passed"]:
        raise RuntimeError(f"DAgger-1 class audit failed: {class_audit}")
    recovery_audit = audit_dagger1_recovery_labels(rows)
    if not recovery_audit["passed"]:
        raise RuntimeError(
            f"DAgger-1 recovery-label audit failed: {recovery_audit}"
        )
    training_eligible = [
        row for row in rows if row.get("production_label_eligible") is True
    ]
    if args.collection_pass == "training":
        if collection_checkpoint is None:
            raise RuntimeError("training collection produced no gate checkpoint")
        output_rows = list(collection_checkpoint["selected_rows"])
        selection_report = dict(
            collection_checkpoint["deterministic_collection_selection"]
        )
        collection_gate = dict(
            collection_checkpoint["recommended_collection_gate"]
        )
        targeted_coverage = dict(
            collection_checkpoint["targeted_state_coverage"]
        )
        independent_root_support = dict(
            collection_checkpoint["independent_root_support"]
        )
        truth_audit_quarantine = dict(
            collection_checkpoint[
                "offline_teacher_target_quarantine_summary"
            ]
        )
        replay_capacity = dict(
            collection_checkpoint["round1_replay_capacity"]
        )
        failed_gate_names = list(
            collection_checkpoint["failed_gate_names"]
        )
        if args.require_recommended_target and (
            collection_checkpoint.get("passed") is not True
            or stopping_report.get("stopping_reason") != (
                "strict_collection_gate_passed"
            )
        ):
            failure_evidence = {
                "failed_gate_names": failed_gate_names,
                "intended_production_outputs": {
                    "output": str(args.output.resolve()),
                    "all_output": (
                        str(args.all_output.resolve())
                        if args.all_output is not None
                        else None
                    ),
                    "manifest": str(output_manifest_path.resolve()),
                },
                "collector_contract": DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
                "recovery_label_contract": (
                    "observable_rank_one_learner_state_v1"
                ),
                "input": str(args.input.resolve()),
                "input_sha256": _file_sha256(args.input),
                "scenario_generator_report": str(
                    args.scenario_generator_report.resolve()
                ),
                "scenario_generator_report_sha256": _file_sha256(
                    args.scenario_generator_report
                ),
                "scenario_manifest": str(args.scenario_manifest.resolve()),
                "scenario_manifest_sha256": _file_sha256(
                    args.scenario_manifest
                ),
                "development_holdout": str(
                    args.development_holdout.resolve()
                ),
                "development_holdout_sha256": _file_sha256(
                    args.development_holdout
                ),
                "development_holdout_manifest": str(
                    args.development_holdout_manifest.resolve()
                ),
                "development_holdout_manifest_sha256": _file_sha256(
                    args.development_holdout_manifest
                ),
                "development_holdout_generator_report": str(
                    args.development_holdout_generator_report.resolve()
                ),
                "development_holdout_generator_report_sha256": _file_sha256(
                    args.development_holdout_generator_report
                ),
                "development_holdout_root_count": len(development_roots),
                "development_holdout_root_set_sha256": stable_json_sha256(
                    sorted(development_roots)
                ),
                "scenario_builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
                "source_partition": "train",
                "source_state": source_state,
                "factory_identities": {
                    "environment": env_factory_binding,
                    "learner_policy": policy_factory_binding,
                    "expert_oracle": {
                        "class": (
                            "psse_env.oracle.expert_policy:ExpertPolicyOracle"
                        ),
                        "source_sha256": file_sha256(expert_source),
                    },
                },
                "release_environment_contract": {
                    "parameter_ranking_dominance_threshold": (
                        BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
                    ),
                    "production_dataset_mode": True,
                    "max_steps": args.max_steps,
                },
                "forbidden_suite": str(args.forbidden_suite.resolve()),
                "forbidden_suite_sha256": _file_sha256(
                    args.forbidden_suite
                ),
                "evaluation_policy": str(
                    DEFAULT_EVALUATION_POLICY.resolve()
                ),
                "evaluation_policy_sha256": _file_sha256(
                    DEFAULT_EVALUATION_POLICY
                ),
                "forbidden_physical_root_count": len(forbidden_roots),
                "frozen_evaluation_root_count": len(frozen_roots),
                "d0_physical_root_count": len(d0_roots),
                "development_physical_root_count": len(development_roots),
                "d0_aggregate_dir": str(args.d0_aggregate_dir.resolve()),
                "d0_raw_sha256": _file_sha256(d0_raw_path),
                "d0_generation_provenance_sha256": _file_sha256(
                    d0_provenance_path
                ),
                "d0_manifest_sha256": _file_sha256(d0_manifest_path),
                "model_id": args.model_id,
                "model_revision": args.model_revision.lower(),
                "learner_seed": learner_seed,
                "iteration": args.iteration,
                "beta": args.beta,
                "beta_contract": beta_contract,
                "collection_pass": args.collection_pass,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "visited_rows": len(rows),
                "candidate_recovery_row_count": len(training_eligible),
                "selected_recovery_row_count": len(output_rows),
                "production_eligible_recovery_rows": len(training_eligible),
                "production_row_target_contract": production_row_target,
                "eligible_recovery_strata": dict(
                    sorted(
                        Counter(
                            str(row.get("recovery_stratum"))
                            for row in training_eligible
                        ).items()
                    )
                ),
                "eligible_physical_root_count": len(
                    {
                        str(row.get("physical_root_fingerprint"))
                        for row in training_eligible
                        if row.get("physical_root_fingerprint")
                    }
                ),
                "recommended_collection_gate": collection_gate,
                "deterministic_collection_selection": selection_report,
                "targeted_state_coverage": targeted_coverage,
                "independent_root_support": independent_root_support,
                "round1_replay_capacity": replay_capacity,
                "rollout_disposition_matrix": rollout_matrix,
                "collection_stopping_report": stopping_report,
                "class_audit": class_audit,
                "recovery_label_audit": recovery_audit,
                "offline_teacher_target_quarantine_summary": (
                    truth_audit_quarantine
                ),
            }
            assert args.failed_collection_dir is not None
            write_failed_collection_evidence_bundle(
                args.failed_collection_dir,
                candidate_rows=training_eligible,
                all_rows=rows,
                evidence=failure_evidence,
            )
            print(
                json.dumps(
                    {
                        "artifact_type": FAILED_COLLECTION_ARTIFACT_TYPE,
                        "failed_collection_dir": str(
                            args.failed_collection_dir.resolve()
                        ),
                        "failed_gate_names": failed_gate_names,
                        "production_outputs_published": False,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 1
    else:
        if training_eligible:
            raise RuntimeError(
                "diagnostic beta=0 collection unexpectedly produced "
                "training-eligible rows"
            )
        output_rows = [
            row
            for row in rows
            if row.get("state_origin") == "learner_policy"
            and row.get("recovery_stratum") is not None
        ]
        if not output_rows:
            raise RuntimeError(
                "DAgger-1 diagnostic reached no learner recovery states"
            )
        collection_gate = {
            "applicable": False,
            "reason": "diagnostic_beta_zero_output_is_training_ineligible",
            "passed": False,
        }
        selection_report = {
            "applicable": False,
            "reason": "diagnostic_beta_zero_output_is_training_ineligible",
            "passed": False,
        }
        replay_capacity = {
            "applicable": False,
            "reason": "diagnostic_beta_zero_output_is_training_ineligible",
            "passed": False,
        }
        truth_audit_quarantine = (
            summarize_dagger1_offline_teacher_target_quarantine(rows)
        )
        targeted_coverage = targeted_state_coverage([])
        independent_root_support = audit_dagger1_independent_root_support([])

    artifact_training_eligible = bool(
        args.collection_pass == "training"
        and beta_contract["passed"]
        and class_audit["passed"]
        and recovery_audit["passed"]
        and truth_audit_quarantine["passed"]
        and selection_report["passed"]
        and production_row_target["passed"]
        and collection_gate["passed"]
        and targeted_coverage["passed"]
        and independent_root_support["passed"]
        and replay_capacity["passed"]
        and rollout_matrix["passed"]
        and stopping_report["passed"]
    )
    selected_example_ids = {
        str(row.get("example_id") or "") for row in output_rows
    }
    for row in rows:
        example_id = str(row.get("example_id") or "")
        is_selected = bool(
            args.collection_pass == "training"
            and artifact_training_eligible
            and row.get("production_label_eligible") is True
            and example_id in selected_example_ids
        )
        if args.collection_pass == "training":
            if is_selected:
                disposition = "selected_for_round1_training"
            elif row.get("production_label_eligible") is True:
                disposition = "safe_candidate_not_selected"
            else:
                disposition = "not_safe_candidate"
        else:
            disposition = "diagnostic_training_ineligible"
        row["collection_training_eligible"] = is_selected
        row["collection_disposition"] = disposition
        labels = row.get("labels")
        if isinstance(labels, dict):
            labels["collection_training_eligible"] = is_selected
            labels["collection_disposition"] = disposition

    # The selected output is a deep copy made by the deterministic selector.
    # Apply the same selection annotation as the complete ledger so final
    # ingestion can require exact row-content equality, not merely matching IDs.
    for row in output_rows:
        row["collection_training_eligible"] = artifact_training_eligible
        row["collection_disposition"] = (
            "selected_for_round1_training"
            if args.collection_pass == "training"
            else "diagnostic_training_ineligible"
        )
        labels = row.get("labels")
        if isinstance(labels, dict):
            labels["collection_training_eligible"] = artifact_training_eligible
            labels["collection_disposition"] = row["collection_disposition"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    all_output_path: str | None = None
    all_output_sha256: str | None = None
    all_output_row_count: int | None = None
    if args.all_output is not None:
        args.all_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.all_output, rows)
        all_output_path = str(args.all_output.resolve())
        all_output_sha256 = _file_sha256(args.all_output)
        all_output_row_count = len(rows)
    manifest = {
        "schema_version": 1,
        "release_evidence_eligible": False,
        "training_eligible": artifact_training_eligible,
        **_round1_publication_contract(artifact_training_eligible),
        "output_sha256": _file_sha256(args.output),
        "all_output": all_output_path,
        "all_output_sha256": all_output_sha256,
        "all_output_row_count": all_output_row_count,
        "collector_contract": DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
        "recovery_label_contract": "observable_rank_one_learner_state_v1",
        "input": str(args.input),
        "input_sha256": _file_sha256(args.input),
        "scenario_generator_report": str(args.scenario_generator_report),
        "scenario_generator_report_sha256": _file_sha256(
            args.scenario_generator_report
        ),
        "scenario_manifest": str(args.scenario_manifest),
        "scenario_manifest_sha256": _file_sha256(args.scenario_manifest),
        "development_holdout": (
            str(args.development_holdout.resolve())
            if args.development_holdout is not None
            else None
        ),
        "development_holdout_sha256": (
            _file_sha256(args.development_holdout)
            if args.development_holdout is not None
            else None
        ),
        "development_holdout_manifest": (
            str(args.development_holdout_manifest.resolve())
            if args.development_holdout_manifest is not None
            else None
        ),
        "development_holdout_manifest_sha256": (
            _file_sha256(args.development_holdout_manifest)
            if args.development_holdout_manifest is not None
            else None
        ),
        "development_holdout_generator_report": (
            str(args.development_holdout_generator_report.resolve())
            if args.development_holdout_generator_report is not None
            else None
        ),
        "development_holdout_generator_report_sha256": (
            _file_sha256(args.development_holdout_generator_report)
            if args.development_holdout_generator_report is not None
            else None
        ),
        "development_holdout_root_count": len(development_roots),
        "development_holdout_root_set_sha256": (
            stable_json_sha256(sorted(development_roots))
            if development_roots
            else None
        ),
        "scenario_builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
        "source_partition": "train",
        "source_state": source_state,
        "factory_identities": {
            "environment": env_factory_binding,
            "learner_policy": policy_factory_binding,
            "expert_oracle": {
                "class": "psse_env.oracle.expert_policy:ExpertPolicyOracle",
                "source_sha256": file_sha256(expert_source),
            },
        },
        "release_environment_contract": {
            "parameter_ranking_dominance_threshold": (
                BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
            ),
            "production_dataset_mode": True,
            "max_steps": args.max_steps,
        },
        "forbidden_suite": str(args.forbidden_suite),
        "forbidden_suite_sha256": _file_sha256(args.forbidden_suite),
        "evaluation_policy": str(DEFAULT_EVALUATION_POLICY),
        "evaluation_policy_sha256": _file_sha256(DEFAULT_EVALUATION_POLICY),
        "forbidden_physical_root_count": len(forbidden_roots),
        "frozen_evaluation_root_count": len(frozen_roots),
        "d0_physical_root_count": len(d0_roots),
        "development_physical_root_count": len(development_roots),
        "d0_aggregate_dir": str(args.d0_aggregate_dir),
        "d0_raw_sha256": _file_sha256(d0_raw_path),
        "d0_generation_provenance_sha256": _file_sha256(d0_provenance_path),
        "d0_manifest_sha256": _file_sha256(d0_manifest_path),
        "model_id": args.model_id,
        "model_revision": args.model_revision.lower(),
        "learner_seed": learner_seed,
        "iteration": args.iteration,
        "beta": args.beta,
        "beta_contract": beta_contract,
        "collection_pass": args.collection_pass,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "visited_rows": len(rows),
        "candidate_recovery_rows": len(training_eligible),
        "candidate_recovery_row_count": len(training_eligible),
        "output_rows": len(output_rows),
        "selected_recovery_row_count": len(output_rows),
        "production_eligible_recovery_rows": len(training_eligible),
        "diagnostic_training_ineligible_recovery_rows": (
            len(output_rows) if args.collection_pass == "diagnostic" else 0
        ),
        "eligible_recovery_strata": dict(
            sorted(
                Counter(
                    str(row.get("recovery_stratum"))
                    for row in training_eligible
                ).items()
            )
        ),
        "eligible_physical_root_count": len(
            {
                str(row.get("physical_root_fingerprint"))
                for row in training_eligible
                if row.get("physical_root_fingerprint")
            }
        ),
        "recommended_collection_gate": collection_gate,
        "deterministic_collection_selection": selection_report,
        "production_row_target_contract": production_row_target,
        "targeted_state_coverage": targeted_coverage,
        "independent_root_support": independent_root_support,
        "round1_replay_capacity": replay_capacity,
        "rollout_disposition_matrix": rollout_matrix,
        "collection_stopping_report": stopping_report,
        "class_audit": class_audit,
        "recovery_label_audit": recovery_audit,
        "offline_teacher_target_quarantine_summary": truth_audit_quarantine,
    }
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
