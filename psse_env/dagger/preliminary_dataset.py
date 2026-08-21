"""Build a deterministic, explicitly non-release preliminary DAgger corpus.

This module is intentionally separate from the production DAgger aggregate
builder.  It admits only a narrowly characterized strict-failure diagnostic
bundle, preserves whole physical roots for preliminary validation/test, and
marks reused D1 rows as auxiliary/non-production before canonical chat export.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.dagger.collect_dagger1 import (
    DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
    DAGGER1_COLLECTION_SELECTION_CONTRACT,
    select_dagger1_collection_rows,
    validate_export_rows_truth_free,
)
from psse_env.dagger.dataset_builder import examples_to_chat_sft
from psse_env.dagger.replay_buffer import (
    DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA,
    DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS,
    DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS,
    dagger1_targeted_state_cells,
)
from psse_env.dagger.rollout_collector import (
    audit_target_aware_state_classes,
    summarize_dagger1_offline_teacher_target_quarantine,
)
from psse_env.sft.provenance import file_sha256, stable_json_sha256


PRELIMINARY_DATASET_RECEIPT_CONTRACT = "preliminary_dagger_dataset_receipt_v1"
PRELIMINARY_ARTIFACT_TYPE = "preliminary_dagger_nonrelease_dataset"
PRELIMINARY_BUILDER_SOURCE_CONTRACT = (
    "preliminary_dagger_builder_source_attestation_v1"
)
STRICT_FAILURE_ARTIFACT_TYPE = "dagger1_failed_strict_collection_diagnostic_bundle"
STRICT_FAILURE_SCHEMA_VERSION = 1
EXPECTED_FAILED_GATE = "round1_replay_capacity"
DEFAULT_EVALUATION_ROOTS = 30
DEFAULT_VALIDATION_ROOTS = 15
DEFAULT_D1_TRAINING_ROWS = 525
DEFAULT_SELECTION_SEED = 20260821

OUTPUT_FILENAMES = {
    "bc0_train": "preliminary.bc0_train.jsonl",
    "bc0_validation": "preliminary.bc0_validation.jsonl",
    "d1_train": "preliminary.d1_train.jsonl",
    "d1_validation": "preliminary.d1_validation.jsonl",
    "d1_test": "preliminary.d1_test.jsonl",
    "d1_eval_combined": "preliminary.d1_eval_combined.jsonl",
    "mixed_train": "preliminary.mixed_train.jsonl",
}
RECEIPT_FILENAME = "preliminary.dataset_receipt.json"

_HEX64 = frozenset("0123456789abcdef")
_HEX40 = _HEX64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX64
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value) <= _HEX40
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain one JSON object")
    return dict(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{path.name}:{line_number} must contain one JSON object"
                )
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def _root(row: Mapping[str, Any]) -> str:
    return str(row.get("physical_root_fingerprint") or "").strip()


def _roots(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    result = {_root(row) for row in rows}
    _require("" not in result, "every dataset row must bind a physical root")
    return result


def _root_set_sha256(roots: Iterable[str]) -> str:
    return stable_json_sha256(sorted(set(roots)))


def _row_multiset_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    return stable_json_sha256(
        sorted(stable_json_sha256(dict(row)) for row in rows)
    )


def _counter_by_field(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(field) or "<missing>") for row in rows).items()
        )
    )


def _root_counter_by_field(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    roots_by_value: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        roots_by_value[str(row.get(field) or "<missing>")].add(_root(row))
    return {
        value: len(roots_by_value[value]) for value in sorted(roots_by_value)
    }


def _distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roots = _roots(rows)
    return {
        "row_count": len(rows),
        "physical_root_count": len(roots),
        "physical_root_set_sha256": _root_set_sha256(roots),
        "physical_roots": sorted(roots),
        "rows_by_recovery_stratum": _counter_by_field(rows, "recovery_stratum"),
        "roots_by_recovery_stratum": _root_counter_by_field(
            rows, "recovery_stratum"
        ),
        "rows_by_scenario_family": _counter_by_field(rows, "scenario_family"),
        "roots_by_scenario_family": _root_counter_by_field(
            rows, "scenario_family"
        ),
        "rows_by_source_tier": _counter_by_field(rows, "source_tier"),
        "roots_by_source_tier": _root_counter_by_field(rows, "source_tier"),
    }


def _builder_source_attestation(repo_root: Path) -> dict[str, Any]:
    """Bind the transformation to tracked bytes at the current Git commit.

    Unrelated working-tree changes are permitted because this fast path can be
    prepared while longer release work remains in progress.  The two files
    that define and invoke this transformation must, however, be tracked and
    byte-identical to ``HEAD``.
    """

    root = repo_root.resolve(strict=True)
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        top_level.returncode == 0
        and Path(top_level.stdout.strip()).resolve(strict=True) == root,
        "preliminary builder repo_root must be the current Git worktree root",
    )
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = head_result.stdout.strip().lower()
    _require(
        head_result.returncode == 0 and _is_commit(head),
        "preliminary builder must run from a Git worktree with a concrete HEAD",
    )

    tracked: dict[str, dict[str, Any]] = {}
    for relative in (
        "psse_env/dagger/preliminary_dataset.py",
        "scripts/build_preliminary_dagger_dataset.py",
    ):
        path = root / Path(relative)
        _require(path.is_file() and not path.is_symlink(), f"missing builder source: {relative}")
        index = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        _require(
            index.returncode == 0,
            f"preliminary builder source is not tracked at HEAD: {relative}",
        )
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        working_bytes = path.read_bytes()
        _require(
            committed.returncode == 0 and committed.stdout == working_bytes,
            f"preliminary builder source differs from HEAD: {relative}",
        )
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        blob_oid = blob.stdout.strip().lower()
        _require(
            blob.returncode == 0
            and len(blob_oid) in {40, 64}
            and set(blob_oid) <= _HEX64,
            f"cannot resolve committed builder blob: {relative}",
        )
        tracked[relative] = {
            "git_blob_oid": blob_oid,
            "sha256": file_sha256(path),
            "size_bytes": len(working_bytes),
        }
    return {
        "contract": PRELIMINARY_BUILDER_SOURCE_CONTRACT,
        "source_commit": head,
        "tracked_files_match_head": True,
        "tracked_files": tracked,
    }


def validate_strict_failure_bundle(
    evidence: Mapping[str, Any],
    *,
    candidate_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit only the completed schedule whose sole failure was replay capacity."""

    checks: dict[str, bool] = {}

    def check(name: str, condition: bool, message: str) -> None:
        checks[name] = bool(condition)
        _require(condition, message)

    check(
        "artifact_contract",
        evidence.get("artifact_type") == STRICT_FAILURE_ARTIFACT_TYPE
        and evidence.get("artifact_schema_version")
        == STRICT_FAILURE_SCHEMA_VERSION,
        "failure evidence is not the reviewed strict-failure diagnostic contract",
    )
    source_state = evidence.get("source_state")
    source_state = source_state if isinstance(source_state, Mapping) else {}
    check(
        "clean_source_commit",
        _is_commit(source_state.get("source_commit"))
        and source_state.get("release_eligible_source") is True
        and source_state.get("source_worktree_dirty") is False,
        "failure evidence is not bound to a clean source commit",
    )
    check(
        "strict_failure_disposition",
        evidence.get("collection_outcome") == "strict_gate_failed"
        and evidence.get("collection_pass") == "training"
        and evidence.get("strict_gate_requested") is True
        and evidence.get("strict_gate_evaluated") is True
        and evidence.get("strict_gate_passed") is False
        and evidence.get("analysis_only") is False
        and evidence.get("diagnostic_only") is True
        and evidence.get("production_outputs_published") is False
        and evidence.get("release_evidence_eligible") is False
        and evidence.get("training_eligible") is False
        and evidence.get("round1_aggregate_eligible") is False,
        "failure evidence does not preserve the expected fail-closed disposition",
    )
    check(
        "sole_failed_gate",
        evidence.get("failed_gate_names") == [EXPECTED_FAILED_GATE]
        and isinstance(evidence.get(EXPECTED_FAILED_GATE), Mapping)
        and evidence[EXPECTED_FAILED_GATE].get("passed") is False,
        "round1_replay_capacity must be the sole failed gate",
    )

    stopping = evidence.get("collection_stopping_report")
    stopping = stopping if isinstance(stopping, Mapping) else {}
    planned_batches = stopping.get("planned_batch_ids")
    executed_batches = stopping.get("executed_batch_ids")
    schedule_complete = bool(
        stopping.get("contract") == DAGGER1_COLLECTION_SCHEDULE_CONTRACT
        and stopping.get("workflow_terminal") is True
        and stopping.get("collection_pass") == "training"
        and stopping.get("training_eligible") is True
        and stopping.get("passed") is False
        and isinstance(stopping.get("planned_episode_count"), int)
        and stopping.get("planned_episode_count") > 0
        and stopping.get("executed_episode_count")
        == stopping.get("planned_episode_count")
        and isinstance(planned_batches, list)
        and isinstance(executed_batches, list)
        and executed_batches == planned_batches
        and stopping.get("unexecuted_batch_ids") == []
        and stopping.get("terminal_failure") in (None, "")
    )
    check(
        "collection_schedule_completed",
        schedule_complete,
        "strict diagnostic collection did not complete its predeclared schedule",
    )

    for field, label in (
        ("class_audit", "state-class audit"),
        ("recovery_label_audit", "recovery-label audit"),
        ("independent_root_support", "independent-root audit"),
        ("targeted_state_coverage", "targeted-state coverage audit"),
        ("deterministic_collection_selection", "collection selection audit"),
        ("recommended_collection_gate", "recommended collection audit"),
        ("rollout_disposition_matrix", "rollout disposition audit"),
    ):
        report = evidence.get(field)
        check(
            f"{field}_passed",
            isinstance(report, Mapping) and report.get("passed") is True,
            f"{label} did not pass",
        )

    quarantine = evidence.get("offline_teacher_target_quarantine_summary")
    quarantine = quarantine if isinstance(quarantine, Mapping) else {}
    check(
        "offline_audit_zero_quarantine",
        quarantine.get("passed") is True
        and quarantine.get("zero_truth_audit_quarantine") is True
        and quarantine.get("quarantined_rows") == 0
        and quarantine.get("invalid_or_missing_audit_rows") == 0
        and quarantine.get("candidate_rows") == len(candidate_rows)
        and quarantine.get("passed_rows") == len(candidate_rows),
        "offline teacher-target audit is not a zero-quarantine candidate ledger",
    )

    diagnostic = evidence.get("diagnostic_artifacts")
    diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
    candidate_binding = diagnostic.get("candidate_recovery_rows")
    candidate_binding = (
        candidate_binding if isinstance(candidate_binding, Mapping) else {}
    )
    actual_candidate_hash = file_sha256(candidate_path)
    check(
        "candidate_file_binding",
        candidate_binding.get("relative_path")
        == "diagnostic.candidate_recovery_rows.jsonl"
        and candidate_binding.get("sha256") == actual_candidate_hash
        and candidate_binding.get("row_count") == len(candidate_rows)
        and evidence.get("candidate_recovery_row_count") == len(candidate_rows),
        "diagnostic candidate file hash/count does not match failure evidence",
    )

    example_ids = [str(row.get("example_id") or "").strip() for row in candidate_rows]
    candidate_shape = bool(
        all(example_ids)
        and len(example_ids) == len(set(example_ids))
        and all(_root(row) for row in candidate_rows)
        and all(row.get("production_label_eligible") is True for row in candidate_rows)
        and all(row.get("collection_role") == "training" for row in candidate_rows)
        and all(row.get("state_origin") == "learner_policy" for row in candidate_rows)
    )
    check(
        "candidate_row_shape",
        candidate_shape,
        "diagnostic candidate ledger has missing/duplicate IDs or ineligible row shape",
    )
    try:
        validate_export_rows_truth_free(candidate_rows)
    except RuntimeError as exc:
        raise ValueError(
            "diagnostic candidate ledger contains private oracle truth"
        ) from exc
    checks["candidate_policy_payload_truth_free"] = True
    recomputed_quarantine = summarize_dagger1_offline_teacher_target_quarantine(
        candidate_rows
    )
    check(
        "candidate_quarantine_recomputed",
        recomputed_quarantine.get("passed") is True
        and recomputed_quarantine.get("quarantined_rows") == 0
        and recomputed_quarantine.get("candidate_rows") == len(candidate_rows),
        "diagnostic candidate ledger no longer recomputes to zero quarantine",
    )
    recomputed_class_audit = audit_target_aware_state_classes(candidate_rows)
    check(
        "candidate_class_audit_recomputed",
        recomputed_class_audit.get("passed") is True,
        "diagnostic candidate ledger no longer passes the state-class audit",
    )
    return {
        "passed": True,
        "checks": checks,
        "candidate_sha256": actual_candidate_hash,
        "candidate_rows": len(candidate_rows),
        "recomputed_candidate_quarantine": recomputed_quarantine,
        "recomputed_candidate_class_audit": recomputed_class_audit,
        "source_report_summaries": {
            name: {
                "passed": evidence[name].get("passed"),
                "stable_json_sha256": stable_json_sha256(evidence[name]),
            }
            for name in (
                "class_audit",
                "recovery_label_audit",
                "independent_root_support",
                "targeted_state_coverage",
                "deterministic_collection_selection",
                "recommended_collection_gate",
                "rollout_disposition_matrix",
                "offline_teacher_target_quarantine_summary",
            )
        },
    }


def validate_d0_generation_binding(
    evidence: Mapping[str, Any],
    *,
    provenance_path: Path,
    provenance: Mapping[str, Any],
    train_path: Path,
    train_rows: Sequence[Mapping[str, Any]],
    validation_path: Path,
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind both BC0 chat splits to the generation provenance named by D1."""

    dataset_hashes = provenance.get("dataset_hashes")
    _require(
        isinstance(dataset_hashes, Mapping),
        "D0 generation provenance lacks dataset hashes",
    )
    actual = {
        "generation_provenance": file_sha256(provenance_path),
        "train": file_sha256(train_path),
        "validation": file_sha256(validation_path),
    }
    expected = {
        "generation_provenance": evidence.get(
            "d0_generation_provenance_sha256"
        ),
        "train": dataset_hashes.get("aggregate.train_view.jsonl"),
        "validation": dataset_hashes.get("aggregate.validation.jsonl"),
    }
    _require(actual == expected, "D0 train/validation provenance hash binding failed")
    _require(
        evidence.get("d0_raw_sha256")
        == dataset_hashes.get("aggregate.raw.jsonl")
        and evidence.get("d0_manifest_sha256")
        == dataset_hashes.get("aggregate.manifest.json"),
        "D1 failure evidence names a different D0 raw corpus or manifest",
    )
    source_state = provenance.get("source_state")
    source_state = source_state if isinstance(source_state, Mapping) else {}
    generation_id = provenance.get("generation_provenance_id")
    _require(
        provenance.get("release_eligible") is True
        and provenance.get("release_failures") == []
        and source_state.get("release_eligible_source") is True
        and source_state.get("source_worktree_dirty") is False
        and _is_commit(source_state.get("source_commit"))
        and _is_sha256(generation_id),
        "D0 generation provenance is not a clean release-eligible source",
    )
    for split_name, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
    ):
        _require(
            all(row.get("generation_provenance_id") == generation_id for row in rows),
            f"D0 {split_name} rows do not bind the named generation provenance",
        )
    train_roots = _roots(train_rows)
    validation_roots = _roots(validation_rows)
    overlap = sorted(train_roots & validation_roots)
    _require(not overlap, "D0 train and validation physical roots overlap")
    return {
        "passed": True,
        "actual_sha256": actual,
        "expected_sha256": expected,
        "generation_provenance_id": generation_id,
        "source_commit": source_state.get("source_commit"),
        "train_validation_root_overlap": overlap,
    }


def _selection_groups(row: Mapping[str, Any]) -> frozenset[str]:
    groups = {
        f"targeted_state_cell:{cell}"
        for cell in dagger1_targeted_state_cells(row)
        if cell in DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS
    }
    recovery_stratum = str(row.get("recovery_stratum") or "").strip()
    if recovery_stratum in DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS:
        groups.add(f"recovery_stratum:{recovery_stratum}")
    return frozenset(groups)


def _gated_selection_floors() -> dict[str, int]:
    result = {
        f"targeted_state_cell:{name}": int(floor)
        for name, floor in sorted(
            DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS.items()
        )
    }
    result.update(
        {
            f"recovery_stratum:{name}": int(floor)
            for name, floor in sorted(
                DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS.items()
            )
            if name not in DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA
        }
    )
    return result


def reserve_evaluation_roots(
    rows: Sequence[Mapping[str, Any]],
    *,
    root_count: int,
    minimum_remaining_rows: int,
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    """Reserve deterministic whole roots without consuming canonical floors."""

    _require(root_count >= 2, "at least two evaluation roots are required")
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        root = _root(row)
        _require(bool(root), "D1 candidate row lacks a physical root")
        by_root[root].append(row)
    _require(
        len(by_root) > root_count,
        "evaluation root request leaves no independent D1 training roots",
    )
    _require(
        len(rows) - sum(
            len(by_root[root])
            for root in sorted(by_root, key=lambda item: (len(by_root[item]), item))[
                :root_count
            ]
        )
        >= minimum_remaining_rows,
        "evaluation root request cannot leave the requested D1 training rows",
    )

    root_groups = {
        root: frozenset(
            group for row in root_rows for group in _selection_groups(row)
        )
        for root, root_rows in by_root.items()
    }
    root_strata = {
        root: frozenset(
            str(row.get("recovery_stratum") or "<missing>")
            for row in root_rows
        )
        for root, root_rows in by_root.items()
    }
    observed_strata = sorted(
        {stratum for strata in root_strata.values() for stratum in strata}
    )
    stratum_root_counts = Counter(
        stratum for strata in root_strata.values() for stratum in strata
    )
    floors = _gated_selection_floors()
    remaining_roots_by_group = {
        group: sum(group in groups for groups in root_groups.values())
        for group in floors
    }
    initial_roots_by_group = dict(sorted(remaining_roots_by_group.items()))
    initial_shortfalls = {
        group: {
            "available": remaining_roots_by_group[group],
            "floor": floor,
        }
        for group, floor in floors.items()
        if remaining_roots_by_group[group] < floor
    }
    _require(
        not initial_shortfalls,
        "D1 candidate ledger is already below canonical gated root floors",
    )

    selected: set[str] = set()
    covered_strata: set[str] = set()
    remaining_row_count = len(rows)
    while len(selected) < root_count:
        choices: list[str] = []
        for root in by_root:
            if root in selected:
                continue
            if remaining_row_count - len(by_root[root]) < minimum_remaining_rows:
                continue
            if any(
                remaining_roots_by_group[group] - 1 < floors[group]
                for group in root_groups[root]
                if group in floors
            ):
                continue
            choices.append(root)
        _require(
            bool(choices),
            "cannot reserve the requested evaluation roots without consuming "
            "canonical D1 training support",
        )

        def choice_key(root: str) -> tuple[Any, ...]:
            new_strata = root_strata[root] - covered_strata
            rarity_score = sum(
                1.0 / stratum_root_counts[stratum] for stratum in new_strata
            )
            return (
                -len(new_strata),
                -rarity_score,
                len(by_root[root]),
                stable_json_sha256(
                    {
                        "contract": PRELIMINARY_DATASET_RECEIPT_CONTRACT,
                        "purpose": "evaluation_root_reservation",
                        "seed": int(seed),
                        "physical_root_fingerprint": root,
                    }
                ),
                root,
            )

        chosen = min(choices, key=choice_key)
        selected.add(chosen)
        covered_strata.update(root_strata[chosen])
        remaining_row_count -= len(by_root[chosen])
        for group in root_groups[chosen]:
            if group in remaining_roots_by_group:
                remaining_roots_by_group[group] -= 1

    remaining_shortfalls = {
        group: {
            "remaining": remaining_roots_by_group[group],
            "floor": floor,
        }
        for group, floor in floors.items()
        if remaining_roots_by_group[group] < floor
    }
    _require(not remaining_shortfalls, "evaluation reservation consumed a gated floor")
    return selected, {
        "contract": "preliminary_floor_aware_whole_root_reservation_v1",
        "passed": True,
        "requested_root_count": root_count,
        "selected_root_count": len(selected),
        "selected_root_set_sha256": _root_set_sha256(selected),
        "selected_roots": sorted(selected),
        "candidate_root_count": len(by_root),
        "candidate_row_count": len(rows),
        "remaining_candidate_row_count": remaining_row_count,
        "gated_root_floors": floors,
        "candidate_roots_by_gated_group": initial_roots_by_group,
        "remaining_roots_by_gated_group": dict(
            sorted(remaining_roots_by_group.items())
        ),
        "observed_recovery_strata": observed_strata,
        "covered_recovery_strata": sorted(covered_strata),
        "omitted_recovery_strata": sorted(set(observed_strata) - covered_strata),
    }


def partition_evaluation_roots(
    rows: Sequence[Mapping[str, Any]],
    evaluation_roots: set[str],
    *,
    validation_root_count: int,
    seed: int,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Split held-out roots while greedily spreading recovery strata."""

    _require(
        0 < validation_root_count < len(evaluation_roots),
        "validation root count must leave at least one held-out test root",
    )
    root_strata: dict[str, set[str]] = {root: set() for root in evaluation_roots}
    for row in rows:
        root = _root(row)
        if root in root_strata:
            root_strata[root].add(str(row.get("recovery_stratum") or "<missing>"))
    all_strata = {stratum for values in root_strata.values() for stratum in values}
    stratum_counts = Counter(
        stratum for values in root_strata.values() for stratum in values
    )
    capacities = {
        "validation": validation_root_count,
        "test": len(evaluation_roots) - validation_root_count,
    }
    assigned: dict[str, set[str]] = {"validation": set(), "test": set()}
    coverage: dict[str, set[str]] = {"validation": set(), "test": set()}
    unassigned = set(evaluation_roots)
    turn = 0
    while unassigned:
        available_partitions = [
            name for name in ("validation", "test") if len(assigned[name]) < capacities[name]
        ]
        partition = available_partitions[turn % len(available_partitions)]

        def root_key(root: str) -> tuple[Any, ...]:
            new_strata = root_strata[root] - coverage[partition]
            rarity_score = sum(1.0 / stratum_counts[item] for item in new_strata)
            return (
                -len(new_strata),
                -rarity_score,
                stable_json_sha256(
                    {
                        "contract": PRELIMINARY_DATASET_RECEIPT_CONTRACT,
                        "purpose": partition,
                        "seed": int(seed),
                        "physical_root_fingerprint": root,
                    }
                ),
                root,
            )

        chosen = min(unassigned, key=root_key)
        assigned[partition].add(chosen)
        coverage[partition].update(root_strata[chosen])
        unassigned.remove(chosen)
        turn += 1
    validation = assigned["validation"]
    test = assigned["test"]
    _require(not validation & test, "D1 validation/test root partition overlaps")
    _require(validation | test == evaluation_roots, "D1 evaluation root was lost")
    return validation, test, {
        "contract": "preliminary_d1_validation_test_root_partition_v1",
        "passed": True,
        "validation_root_count": len(validation),
        "validation_root_set_sha256": _root_set_sha256(validation),
        "validation_roots": sorted(validation),
        "test_root_count": len(test),
        "test_root_set_sha256": _root_set_sha256(test),
        "test_roots": sorted(test),
        "pairwise_overlap": [],
        "observed_recovery_strata": sorted(all_strata),
        "validation_covered_recovery_strata": sorted(coverage["validation"]),
        "validation_omitted_recovery_strata": sorted(
            all_strata - coverage["validation"]
        ),
        "test_covered_recovery_strata": sorted(coverage["test"]),
        "test_omitted_recovery_strata": sorted(all_strata - coverage["test"]),
    }


def _stable_d1_row_order(
    rows: Sequence[Mapping[str, Any]], *, purpose: str, seed: int
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                stable_json_sha256(
                    {
                        "contract": PRELIMINARY_DATASET_RECEIPT_CONTRACT,
                        "purpose": purpose,
                        "seed": int(seed),
                        "example_id": row.get("example_id"),
                        "physical_root_fingerprint": _root(row),
                    }
                ),
                str(row.get("example_id") or ""),
            ),
        )
    ]


def _export_preliminary_d1(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for source in rows:
        row = copy.copy(dict(source))
        row["production_label_eligible"] = False
        row["auxiliary_training_eligible"] = True
        row["preliminary_release_eligible"] = False
        row["preliminary_source_production_label_eligible"] = bool(
            source.get("production_label_eligible") is True
        )
        marked.append(row)
    exported = examples_to_chat_sft(
        marked,
        protocol="canonical",
        allow_ineligible_auxiliary=True,
    )
    _require(len(exported) == len(marked), "canonical D1 export dropped a row")
    for row in exported:
        row["preliminary_release_eligible"] = False
        row["preliminary_source_disposition"] = (
            "strict_failure_diagnostic_candidate"
        )
        metadata = row.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata["preliminary_release_eligible"] = False
        metadata["preliminary_source_disposition"] = (
            "strict_failure_diagnostic_candidate"
        )
        metadata["preliminary_receipt_contract"] = (
            PRELIMINARY_DATASET_RECEIPT_CONTRACT
        )
        row["metadata"] = metadata
    return exported


def _stable_mixed_rows(
    d0_rows: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    tagged: list[tuple[str, int, Mapping[str, Any]]] = [
        ("d0", index, row) for index, row in enumerate(d0_rows)
    ] + [("d1", index, row) for index, row in enumerate(d1_rows)]
    tagged.sort(
        key=lambda item: (
            stable_json_sha256(
                {
                    "contract": PRELIMINARY_DATASET_RECEIPT_CONTRACT,
                    "purpose": "mixed_train_order",
                    "seed": int(seed),
                    "source": item[0],
                    "source_index": item[1],
                    "example_id": item[2].get("example_id"),
                    "physical_root_fingerprint": _root(item[2]),
                }
            ),
            item[0],
            item[1],
        )
    )
    return [copy.deepcopy(dict(row)) for _source, _index, row in tagged]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _split_record(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": file_sha256(path),
        **_distribution(rows),
    }


def build_preliminary_dagger_dataset(
    *,
    failure_evidence_path: Path,
    candidate_rows_path: Path,
    d0_generation_provenance_path: Path,
    d0_train_path: Path,
    d0_validation_path: Path,
    output_dir: Path,
    evaluation_root_count: int = DEFAULT_EVALUATION_ROOTS,
    validation_root_count: int = DEFAULT_VALIDATION_ROOTS,
    d1_training_row_count: int = DEFAULT_D1_TRAINING_ROWS,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the preliminary corpus and return its deterministic receipt."""

    paths = (
        failure_evidence_path,
        candidate_rows_path,
        d0_generation_provenance_path,
        d0_train_path,
        d0_validation_path,
    )
    for path in paths:
        _require(path.is_file(), f"required input does not exist: {path}")
    _require(d1_training_row_count > 0, "D1 training row count must be positive")
    _require(
        evaluation_root_count > validation_root_count > 0,
        "evaluation roots must split into nonempty validation and test sets",
    )

    evidence = _load_mapping(failure_evidence_path)
    candidates = _load_jsonl(candidate_rows_path)
    provenance = _load_mapping(d0_generation_provenance_path)
    d0_train = _load_jsonl(d0_train_path)
    d0_validation = _load_jsonl(d0_validation_path)
    failure_admission = validate_strict_failure_bundle(
        evidence,
        candidate_path=candidate_rows_path,
        candidate_rows=candidates,
    )
    d0_binding = validate_d0_generation_binding(
        evidence,
        provenance_path=d0_generation_provenance_path,
        provenance=provenance,
        train_path=d0_train_path,
        train_rows=d0_train,
        validation_path=d0_validation_path,
        validation_rows=d0_validation,
    )
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    builder_source = _builder_source_attestation(repo_root)
    builder_source_commit = builder_source["source_commit"]

    d0_train_roots = _roots(d0_train)
    d0_validation_roots = _roots(d0_validation)
    d1_candidate_roots = _roots(candidates)
    d0_d1_overlap = sorted(
        d1_candidate_roots & (d0_train_roots | d0_validation_roots)
    )
    _require(
        not d0_d1_overlap,
        "diagnostic D1 candidate roots overlap BC0 train/validation roots",
    )

    eval_roots, reservation = reserve_evaluation_roots(
        candidates,
        root_count=evaluation_root_count,
        minimum_remaining_rows=d1_training_row_count,
        seed=selection_seed,
    )
    validation_roots, test_roots, eval_partition = partition_evaluation_roots(
        candidates,
        eval_roots,
        validation_root_count=validation_root_count,
        seed=selection_seed,
    )
    remaining_candidates = [
        row for row in candidates if _root(row) not in eval_roots
    ]
    selected_d1, selection_report = select_dagger1_collection_rows(
        remaining_candidates,
        target_min_rows=d1_training_row_count,
        target_max_rows=d1_training_row_count,
    )
    _require(
        selection_report.get("contract") == DAGGER1_COLLECTION_SELECTION_CONTRACT
        and selection_report.get("passed") is True
        and len(selected_d1) == d1_training_row_count,
        "canonical D1 training selection failed after evaluation-root reservation",
    )

    validation_raw = _stable_d1_row_order(
        [row for row in candidates if _root(row) in validation_roots],
        purpose="d1_validation",
        seed=selection_seed,
    )
    test_raw = _stable_d1_row_order(
        [row for row in candidates if _root(row) in test_roots],
        purpose="d1_test",
        seed=selection_seed,
    )
    combined_eval_raw = _stable_d1_row_order(
        [*validation_raw, *test_raw],
        purpose="d1_eval_combined",
        seed=selection_seed,
    )
    d1_train = _export_preliminary_d1(selected_d1)
    d1_validation = _export_preliminary_d1(validation_raw)
    d1_test = _export_preliminary_d1(test_raw)
    d1_eval_combined = _export_preliminary_d1(combined_eval_raw)
    for rows in (d1_train, d1_validation, d1_test, d1_eval_combined):
        try:
            validate_export_rows_truth_free(rows)
        except RuntimeError as exc:
            raise ValueError("canonical preliminary D1 export leaked oracle truth") from exc
    mixed_train = _stable_mixed_rows(
        d0_train, d1_train, seed=selection_seed
    )

    split_rows: dict[str, list[dict[str, Any]]] = {
        "bc0_train": [dict(row) for row in d0_train],
        "bc0_validation": [dict(row) for row in d0_validation],
        "d1_train": d1_train,
        "d1_validation": d1_validation,
        "d1_test": d1_test,
        "d1_eval_combined": d1_eval_combined,
        "mixed_train": mixed_train,
    }
    split_roots = {name: _roots(rows) for name, rows in split_rows.items()}
    required_pairs = (
        ("bc0_train", "bc0_validation"),
        ("d1_train", "d1_validation"),
        ("d1_train", "d1_test"),
        ("d1_validation", "d1_test"),
        ("mixed_train", "d1_validation"),
        ("mixed_train", "d1_test"),
        ("bc0_validation", "d1_train"),
        ("bc0_validation", "d1_validation"),
        ("bc0_validation", "d1_test"),
    )
    pairwise_overlaps = {
        f"{left}__{right}": sorted(split_roots[left] & split_roots[right])
        for left, right in required_pairs
    }
    _require(
        not any(pairwise_overlaps.values()),
        "preliminary train/validation/test physical roots overlap",
    )
    expected_eval_content = _row_multiset_sha256([*d1_validation, *d1_test])
    actual_eval_content = _row_multiset_sha256(d1_eval_combined)
    expected_mixed_content = _row_multiset_sha256([*d0_train, *d1_train])
    actual_mixed_content = _row_multiset_sha256(mixed_train)
    _require(
        expected_eval_content == actual_eval_content
        and expected_mixed_content == actual_mixed_content,
        "preliminary combined split content does not match its source rows",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()
    }
    receipt_path = output_dir / RECEIPT_FILENAME
    collisions = [path.name for path in [*output_paths.values(), receipt_path] if path.exists()]
    _require(
        not collisions,
        "refusing to overwrite preliminary artifacts: " + ", ".join(collisions),
    )
    for name, path in output_paths.items():
        _atomic_write_jsonl(path, split_rows[name])

    splits = {
        name: _split_record(output_paths[name], split_rows[name])
        for name in OUTPUT_FILENAMES
    }
    outputs = {
        record["filename"]: {
            key: record[key]
            for key in (
                "sha256",
                "row_count",
                "physical_root_count",
                "physical_root_set_sha256",
            )
        }
        for record in splits.values()
    }
    diagnostic_source = evidence.get("source_state")
    diagnostic_source = (
        diagnostic_source if isinstance(diagnostic_source, Mapping) else {}
    )
    source_commit = diagnostic_source.get("source_commit")
    _require(_is_commit(source_commit), "diagnostic failure evidence lacks its source commit")
    receipt: dict[str, Any] = {
        "contract": PRELIMINARY_DATASET_RECEIPT_CONTRACT,
        "artifact_type": PRELIMINARY_ARTIFACT_TYPE,
        "release_eligible": False,
        "release_ineligibility_reasons": [
            "source D1 strict collection failed round1_replay_capacity",
            "D1 examples were recovered from a diagnostic-only failure ledger",
            "held-out D1 roots are preliminary and are not the frozen release suite",
            "bounded fast-path selection is intended only for preliminary debugging",
        ],
        "intended_use": "preliminary_dagger_signal_and_pipeline_debugging_only",
        "dataset_directory": ".",
        "path_resolution": "resolve output filenames relative to this receipt",
        "parameters": {
            "evaluation_root_count": evaluation_root_count,
            "validation_root_count": validation_root_count,
            "test_root_count": evaluation_root_count - validation_root_count,
            "d1_training_row_count": d1_training_row_count,
            "selection_seed": selection_seed,
        },
        "source_commits": {
            "d0_generation": d0_binding.get("source_commit"),
            "diagnostic_collection": source_commit,
            "builder": builder_source_commit,
        },
        "builder_source_attestation": builder_source,
        "inputs": {
            "failure_evidence": {
                "filename": failure_evidence_path.name,
                "sha256": file_sha256(failure_evidence_path),
            },
            "diagnostic_candidate_rows": {
                "filename": candidate_rows_path.name,
                "sha256": file_sha256(candidate_rows_path),
                "row_count": len(candidates),
            },
            "d0_generation_provenance": {
                "filename": d0_generation_provenance_path.name,
                "sha256": file_sha256(d0_generation_provenance_path),
            },
            "d0_train": {
                "filename": d0_train_path.name,
                "sha256": file_sha256(d0_train_path),
                "row_count": len(d0_train),
            },
            "d0_validation": {
                "filename": d0_validation_path.name,
                "sha256": file_sha256(d0_validation_path),
                "row_count": len(d0_validation),
            },
        },
        "outputs": outputs,
        "splits": splits,
        "audits": {
            "strict_failure_bundle_admission": failure_admission,
            "d0_generation_binding": d0_binding,
            "evaluation_root_reservation": reservation,
            "evaluation_root_partition": eval_partition,
            "canonical_d1_training_selection": selection_report,
            "content_composition": {
                "passed": True,
                "d1_validation_plus_test_row_multiset_sha256": (
                    expected_eval_content
                ),
                "d1_eval_combined_row_multiset_sha256": actual_eval_content,
                "bc0_plus_d1_train_row_multiset_sha256": expected_mixed_content,
                "mixed_train_row_multiset_sha256": actual_mixed_content,
            },
            "root_disjointness": {
                "passed": True,
                "d0_d1_candidate_overlap": d0_d1_overlap,
                "required_pairwise_overlaps": pairwise_overlaps,
                "d1_train_root_set_sha256": splits["d1_train"][
                    "physical_root_set_sha256"
                ],
                "d1_validation_root_set_sha256": splits["d1_validation"][
                    "physical_root_set_sha256"
                ],
                "d1_test_root_set_sha256": splits["d1_test"][
                    "physical_root_set_sha256"
                ],
            },
        },
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, release-ineligible preliminary DAgger dataset "
            "from one narrowly admitted strict-failure diagnostic bundle."
        )
    )
    parser.add_argument("--failure-evidence", required=True, type=Path)
    parser.add_argument("--candidate-rows", required=True, type=Path)
    parser.add_argument("--d0-generation-provenance", required=True, type=Path)
    parser.add_argument("--d0-train", required=True, type=Path)
    parser.add_argument("--d0-validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluation-roots", type=int, default=DEFAULT_EVALUATION_ROOTS
    )
    parser.add_argument(
        "--validation-roots", type=int, default=DEFAULT_VALIDATION_ROOTS
    )
    parser.add_argument(
        "--d1-training-rows", type=int, default=DEFAULT_D1_TRAINING_ROWS
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_preliminary_dagger_dataset(
            failure_evidence_path=args.failure_evidence,
            candidate_rows_path=args.candidate_rows,
            d0_generation_provenance_path=args.d0_generation_provenance,
            d0_train_path=args.d0_train,
            d0_validation_path=args.d0_validation,
            output_dir=args.output_dir,
            evaluation_root_count=args.evaluation_roots,
            validation_root_count=args.validation_roots,
            d1_training_row_count=args.d1_training_rows,
            selection_seed=args.selection_seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "receipt": str(args.output_dir / RECEIPT_FILENAME),
                "release_eligible": receipt["release_eligible"],
                "d1_training_rows": receipt["splits"]["d1_train"]["row_count"],
                "d1_validation_roots": receipt["splits"]["d1_validation"][
                    "physical_root_count"
                ],
                "d1_test_roots": receipt["splits"]["d1_test"][
                    "physical_root_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "PRELIMINARY_ARTIFACT_TYPE",
    "PRELIMINARY_BUILDER_SOURCE_CONTRACT",
    "PRELIMINARY_DATASET_RECEIPT_CONTRACT",
    "RECEIPT_FILENAME",
    "build_preliminary_dagger_dataset",
    "main",
    "partition_evaluation_roots",
    "reserve_evaluation_roots",
    "validate_d0_generation_binding",
    "validate_strict_failure_bundle",
]
