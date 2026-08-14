"""Release gates for semantic consistency in a D0 plus D1 training corpus.

The collection manifest is an integrity binding, not authority for teacher
semantics.  These helpers deliberately recompute exact and approximate
realizability from the immutable native rows at the final ingestion boundary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from psse_env.dagger.sft_audit import (
    audit_approximate_teacher_realizability,
    audit_teacher_realizability,
)
from psse_env.examples.generate_round0_aggregate import (
    APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
)


STRATIFIED_APPROXIMATE_REALIZABILITY_KWARGS: dict[str, Any] = {
    **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
    # A stratum must contain at least one real same-structure comparison.  The
    # global natural-population gate retains the stronger round-0 floors.
    "minimum_nearest_neighbor_comparisons": 1,
    "minimum_nearest_neighbor_coverage": 0.01,
    "minimum_local_perturbation_comparisons": 1,
    "minimum_local_perturbation_coverage": 0.01,
}


def _row_value(row: Mapping[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is not None:
        return value
    for container_name in ("labels", "metadata"):
        container = row.get(container_name)
        if isinstance(container, Mapping) and container.get(field) is not None:
            return container.get(field)
    return None


def stratified_approximate_realizability(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    required_values: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Audit each observable semantic stratum with nonzero pair coverage."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_value = _row_value(row, field)
        value = str(raw_value if raw_value not in (None, "") else "unknown")
        grouped[value].append(row)
    for value in required_values:
        grouped.setdefault(str(value), [])

    reports: dict[str, dict[str, Any]] = {}
    for value, group in sorted(grouped.items()):
        report = audit_approximate_teacher_realizability(
            group,
            **STRATIFIED_APPROXIMATE_REALIZABILITY_KWARGS,
        )
        roots = {
            str(root)
            for row in group
            if (root := _row_value(row, "physical_root_fingerprint"))
            not in (None, "")
        }
        missing_roots = sum(
            _row_value(row, "physical_root_fingerprint") in (None, "")
            for row in group
        )
        comparison_coverage_passed = bool(
            int(report.get("nearest_neighbor_compared_examples", 0)) >= 1
            and float(report.get("nearest_neighbor_comparison_coverage", 0.0))
            > 0.0
            and int(report.get("local_perturbation_compared_examples", 0)) >= 1
            and float(report.get("local_perturbation_comparison_coverage", 0.0))
            > 0.0
        )
        release_gate_passed = bool(
            group
            and not missing_roots
            and comparison_coverage_passed
            and report.get("passed") is True
        )
        report.update(
            {
                "distinct_physical_roots": len(roots),
                "missing_physical_root_rows": missing_roots,
                "comparison_coverage_passed": comparison_coverage_passed,
                "release_gate_passed": release_gate_passed,
            }
        )
        reports[value] = report
    return reports


def _failed_strata(
    reports: Mapping[str, Mapping[str, Any]], *, dimension: str
) -> list[str]:
    return [
        f"{dimension}={value} approximate realizability failed"
        for value, report in sorted(reports.items())
        if report.get("release_gate_passed") is not True
    ]


def audit_dagger1_union_realizability(
    natural_rows: Sequence[Mapping[str, Any]],
    training_view_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute binding exact/approximate gates on natural and sampled rows."""

    natural = list(natural_rows)
    training_view = list(training_view_rows)
    if not natural or not training_view:
        raise ValueError("natural and training-view rows must both be non-empty")

    natural_exact = audit_teacher_realizability(
        natural, conflict_tolerance=0.0
    )
    training_exact = audit_teacher_realizability(
        training_view, conflict_tolerance=0.0
    )
    natural_approximate = audit_approximate_teacher_realizability(
        natural,
        **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
    )
    # Replay duplicates may be useful optimizer weights, but cannot fabricate
    # comparison coverage.  Approximate evidence therefore uses one immutable
    # representative per source example; exact conflict detection above still
    # audits the complete sampled view.
    unique_training_view: list[Mapping[str, Any]] = []
    seen_example_ids: set[str] = set()
    for index, row in enumerate(training_view):
        example_id = str(row.get("example_id") or f"missing-{index}")
        if example_id in seen_example_ids:
            continue
        seen_example_ids.add(example_id)
        unique_training_view.append(row)
    training_approximate = audit_approximate_teacher_realizability(
        unique_training_view,
        **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
    )
    training_approximate.update(
        {
            "raw_training_view_rows": len(training_view),
            "deduplicated_training_view_rows": len(unique_training_view),
            "comparison_population": "unique_source_example_id",
        }
    )
    approximate_by_family = stratified_approximate_realizability(
        natural, "scenario_family"
    )
    approximate_by_state_class = stratified_approximate_realizability(
        natural, "state_class"
    )
    # Recovery strata are a D1 concept.  D0 rows have no such label, so
    # coercing the complete D0 population into a synthetic ``unknown`` stratum
    # would turn an irrelevant group into a binding release gate.
    natural_recovery_rows = [
        row
        for row in natural
        if _row_value(row, "recovery_stratum") not in (None, "")
    ]
    approximate_by_recovery_stratum = stratified_approximate_realizability(
        natural_recovery_rows, "recovery_stratum"
    )

    failures: list[str] = []
    for label, report in (
        ("natural exact", natural_exact),
        ("training-view exact", training_exact),
        ("natural approximate", natural_approximate),
        ("training-view approximate", training_approximate),
    ):
        if report.get("passed") is not True:
            failures.append(f"{label} teacher realizability failed")
    failures.extend(
        _failed_strata(
            approximate_by_family,
            dimension="scenario_family",
        )
    )
    failures.extend(
        _failed_strata(
            approximate_by_state_class,
            dimension="state_class",
        )
    )
    failures.extend(
        _failed_strata(
            approximate_by_recovery_stratum,
            dimension="recovery_stratum",
        )
    )
    return {
        "schema_version": 1,
        "contract": "dagger1_final_union_semantic_realizability_v1",
        "natural_teacher_realizability": natural_exact,
        "training_view_teacher_realizability": training_exact,
        "natural_approximate_teacher_realizability": natural_approximate,
        "training_view_approximate_teacher_realizability": training_approximate,
        "approximate_teacher_realizability_by_scenario_family": (
            approximate_by_family
        ),
        "approximate_teacher_realizability_by_state_class": (
            approximate_by_state_class
        ),
        "approximate_teacher_realizability_by_recovery_stratum": (
            approximate_by_recovery_stratum
        ),
        "recovery_stratum_comparison_population": {
            "contract": "native_rows_with_explicit_recovery_stratum_v1",
            "rows": len(natural_recovery_rows),
        },
        "failures": failures,
        "passed": not failures,
    }
