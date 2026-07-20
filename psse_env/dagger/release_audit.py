"""Strict offline truth audit for release episode trajectories.

This module is deliberately outside the policy/observation path.  It consumes
hidden scenario truth only after an episode has finished and returns an audit
report suitable for release gating; callers must never merge its inputs or
output into a model-facing observation or training target.

The audit fails closed for resolved episodes.  In addition to checking every
accepted correction against a same-family truth target, it requires explicit
remaining-fault evidence and the final active physical store payload.  Healthy
component preservation and the correctness of any claimed diagnostic
explanation remain mandatory for operator handoff as well: escalation may
leave true faults unresolved, but it may not hide collateral damage or a false
partial diagnosis.  Only the final clean-measurement comparison may be skipped
through a reason-bearing ``not_applicable`` declaration; core physical
invariants are never waivable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
)


AUDIT_VERSION = "strict_offline_episode_truth_v3"

ACCEPTED_TARGETS_CHECK = "accepted_correction_targets"
ACCEPTED_TARGET_NONREGRESSION_CHECK = "accepted_target_nonregression"
REMAINING_FAULTS_CHECK = "remaining_true_faults"
HEALTHY_MEASUREMENTS_CHECK = "healthy_measurements_preserved"
HEALTHY_CASE_CHECK = "healthy_case_components_preserved"
FINAL_MEASUREMENTS_CHECK = "final_measurements_match_clean"
FINAL_CASE_CHECK = "final_case_matches_clean"
DIAGNOSTIC_FAMILY_CHECK = "diagnostic_explanation_family"
DIAGNOSTIC_LOCALIZATION_CHECK = "diagnostic_localization"

_RESOLVED_CHECKS = frozenset(
    {
        REMAINING_FAULTS_CHECK,
        HEALTHY_MEASUREMENTS_CHECK,
        HEALTHY_CASE_CHECK,
        FINAL_MEASUREMENTS_CHECK,
        FINAL_CASE_CHECK,
        DIAGNOSTIC_LOCALIZATION_CHECK,
    }
)
# ``not_applicable`` is deliberately narrower than the set of resolved-only
# checks.  Explanation-only waveform families cannot restore their fundamental
# measurement vector, so they may waive the final clean-vector comparison.
# Target correctness, target non-regression, the remaining-fault ledger, and
# healthy-component preservation are physical invariants and are never
# waivable.  Final-case waivers are also prohibited until an explicit,
# independently validated explanation-only case contract exists.
ALLOWED_NOT_APPLICABLE_CHECKS = frozenset({FINAL_MEASUREMENTS_CHECK})
EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT = (
    "explanation_only_diagnostic_localization_v1"
)
_DIAGNOSTIC_FAMILIES = frozenset(
    {"harmonic", "hif", "three_phase_unbalance"}
)
_CORRECTION_FAMILY = {
    CORRECT_MEASUREMENTS: "measurement",
    "correct_measurements_from_path": "measurement",
    CORRECT_PARAMETERS: "parameter",
    "correct_parameters_from_path": "parameter",
    CORRECT_TOPOLOGY: "topology",
    "correct_topology_from_path": "topology",
}
_PARAMETER_BRANCH_COLUMNS = {
    "r": (2,),
    "x": (3,),
    "rx": (2, 3),
    "b": (4,),
    "tap": (8,),
    "shift": (9,),
}


@dataclass(frozen=True)
class ReleaseAuditTolerances:
    """Declared numeric tolerances used by the strict audit.

    Defaults are intentionally tight.  A release scenario with known sensor
    noise or localization uncertainty should persist its chosen profile and
    pass it to the audit rather than rely on an implicit broad allowance.
    """

    measurement_abs: float = 1e-6
    measurement_rel: float = 1e-6
    case_abs: float = 1e-9
    case_rel: float = 1e-9
    final_case_abs: float = 1e-9
    final_case_rel: float = 1e-9
    harmonic_bus_index: float = 0.0
    hif_branch_rows: float = 0.0
    hif_alpha_abs: float = 0.05
    unbalance_bus_index: float = 0.0
    unbalance_top_k: int = 1


def _coerce_tolerances(
    value: ReleaseAuditTolerances | Mapping[str, Any] | None,
) -> ReleaseAuditTolerances:
    if value is None:
        result = ReleaseAuditTolerances()
    elif isinstance(value, ReleaseAuditTolerances):
        result = value
    elif isinstance(value, Mapping):
        valid = {item.name for item in fields(ReleaseAuditTolerances)}
        unknown = sorted(str(key) for key in value if str(key) not in valid)
        if unknown:
            raise ValueError(f"Unknown release-audit tolerances: {unknown}")
        result = ReleaseAuditTolerances(
            **{str(key): item for key, item in value.items()}
        )
    else:
        raise TypeError("tolerances must be ReleaseAuditTolerances or a mapping")
    for name, raw in asdict(result).items():
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Audit tolerance {name!r} must be numeric") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"Audit tolerance {name!r} must be finite and nonnegative")
        if name.endswith("top_k") and (numeric < 1.0 or not numeric.is_integer()):
            raise ValueError(f"Audit tolerance {name!r} must be a positive integer")
    return result


def _as_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    if isinstance(value, Sequence):
        return list(value)
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, Real):
        try:
            if isinstance(value, (str, bytes, bytearray)) or value is None:
                return None
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        result = int(numeric)
    else:
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return result if result >= 0 else None


def _values_close(
    observed: Any,
    expected: Any,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> bool:
    """Recursively compare JSON-like values and array-like numeric payloads."""
    left_number = _finite_number(observed)
    right_number = _finite_number(expected)
    if left_number is not None or right_number is not None:
        return bool(
            left_number is not None
            and right_number is not None
            and math.isclose(
                left_number,
                right_number,
                abs_tol=abs_tolerance,
                rel_tol=rel_tolerance,
            )
        )
    if isinstance(observed, Mapping) or isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            return False
        if {str(key) for key in observed} != {str(key) for key in expected}:
            return False
        expected_by_key = {str(key): item for key, item in expected.items()}
        return all(
            _values_close(
                item,
                expected_by_key[str(key)],
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            for key, item in observed.items()
        )
    left_sequence = _as_sequence(observed)
    right_sequence = _as_sequence(expected)
    if left_sequence is not None or right_sequence is not None:
        if left_sequence is None or right_sequence is None:
            return False
        return len(left_sequence) == len(right_sequence) and all(
            _values_close(
                left,
                right,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            for left, right in zip(left_sequence, right_sequence)
        )
    return observed == expected


def _accepted_corrections(final_state: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], bool]:
    raw = final_state.get("accepted_corrections")
    if raw is None:
        return [], True
    sequence = _as_sequence(raw)
    if sequence is None:
        return [], False
    return [item for item in sequence if isinstance(item, Mapping)], all(
        isinstance(item, Mapping) for item in sequence
    )


def _correction_action(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    action = item.get("source_action") or item.get("action") or item
    return action if isinstance(action, Mapping) else None


def _measurement_truth_targets(
    scenario: Mapping[str, Any], problems: list[str]
) -> set[int]:
    raw = scenario.get("true_measurement_errors", [])
    rows = _as_sequence(raw)
    if rows is None:
        problems.append("true_measurement_targets_malformed")
        return set()
    targets: set[int] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            problems.append("true_measurement_targets_malformed")
            continue
        index = item.get("index", item.get("index0"))
        index = _nonnegative_integer(index)
        if index is None:
            problems.append("true_measurement_targets_malformed")
            continue
        targets.add(index)
    return targets


def _branch_target(item: Mapping[str, Any]) -> tuple[str, Any] | None:
    rows: list[int] = []
    if item.get("branch_row0") is not None:
        row0 = _nonnegative_integer(item["branch_row0"])
        if row0 is None:
            return None
        rows.append(row0)
    for key in ("line_index1", "line_index"):
        if item.get(key) is not None:
            line1 = _nonnegative_integer(item[key])
            if line1 is None or line1 < 1:
                return None
            rows.append(line1 - 1)
    if rows:
        if len(set(rows)) != 1:
            return None
        return ("branch_row0", rows[0])
    for key in ("branch_id", "cb_name", "dss_element"):
        if item.get(key) is not None and str(item[key]).strip():
            return ("branch_id", str(item[key]).strip())
    return None


def _branch_truth_targets(
    scenario: Mapping[str, Any], key: str, problems: list[str]
) -> set[tuple[str, Any]]:
    raw = scenario.get(key, [])
    rows = _as_sequence(raw)
    if rows is None:
        problems.append(f"{key}_malformed")
        return set()
    targets: set[tuple[str, Any]] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            problems.append(f"{key}_malformed")
            continue
        target = _branch_target(item)
        if target is None:
            problems.append(f"{key}_malformed")
            continue
        targets.add(target)
    return targets


def _measurement_action_targets(arguments: Mapping[str, Any]) -> set[int] | None:
    targets: set[int] = set()
    group = arguments.get("suspect_group")
    if group is not None:
        rows = _as_sequence(group)
        if rows is None:
            return None
        for raw in rows:
            index = _nonnegative_integer(raw)
            if index is None:
                return None
            targets.add(index)
    updates = arguments.get("measurement_updates")
    if updates is not None:
        if isinstance(updates, Mapping):
            raw_indices = list(updates)
        else:
            update_rows = _as_sequence(updates)
            if update_rows is None:
                return None
            raw_indices = []
            for item in update_rows:
                if not isinstance(item, Mapping):
                    return None
                raw_indices.append(item.get("index", item.get("index0")))
        for raw in raw_indices:
            index = _nonnegative_integer(raw)
            if index is None:
                return None
            targets.add(index)
    return targets


def _audit_accepted_targets(
    scenario: Mapping[str, Any], final_state: Mapping[str, Any]
) -> tuple[
    list[str],
    set[int],
    set[tuple[str, Any]],
    set[tuple[str, Any]],
    set[int],
    set[tuple[str, Any]],
    set[tuple[str, Any]],
]:
    problems: list[str] = []
    truth_measurements = _measurement_truth_targets(scenario, problems)
    truth_parameters = _branch_truth_targets(
        scenario, "true_parameter_errors", problems
    )
    truth_topology = _branch_truth_targets(
        scenario, "true_topology_errors", problems
    )
    corrections, well_formed = _accepted_corrections(final_state)
    accepted_measurements: set[int] = set()
    accepted_parameters: set[tuple[str, Any]] = set()
    accepted_topology: set[tuple[str, Any]] = set()
    if not well_formed:
        problems.append("accepted_corrections_malformed")
    for item in corrections:
        action = _correction_action(item)
        if action is None:
            problems.append("accepted_correction_action_malformed")
            continue
        tool = str(action.get("tool") or action.get("name") or "")
        family = _CORRECTION_FAMILY.get(tool)
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping):
            problems.append("accepted_correction_arguments_malformed")
            continue
        if family == "measurement":
            targets = _measurement_action_targets(arguments)
            if targets is None:
                problems.append("accepted_measurement_targets_malformed")
            elif not targets:
                problems.append("accepted_measurement_targets_missing")
            elif not targets.issubset(truth_measurements):
                # This catches both completely wrong and broad grouped
                # corrections that include even one healthy meter.
                problems.append("accepted_measurement_targets_outside_truth")
            else:
                accepted_measurements.update(targets)
        elif family in {"parameter", "topology"}:
            target = _branch_target(arguments)
            if target is None:
                problems.append(f"accepted_{family}_target_missing_or_malformed")
                continue
            family_truth = truth_parameters if family == "parameter" else truth_topology
            if target not in family_truth:
                problems.append(f"accepted_{family}_target_outside_same_family_truth")
            elif family == "parameter":
                accepted_parameters.add(target)
            else:
                accepted_topology.add(target)
        else:
            problems.append("accepted_correction_tool_unsupported")
    return (
        problems,
        truth_measurements,
        truth_parameters,
        truth_topology,
        accepted_measurements,
        accepted_parameters,
        accepted_topology,
    )


def _diagnostic_truth(
    scenario: Mapping[str, Any],
) -> tuple[dict[str, list[Mapping[str, Any]]], list[str]]:
    hidden = scenario.get("hidden_truth")
    hidden = hidden if isinstance(hidden, Mapping) else {}
    problems: list[str] = []
    result: dict[str, list[Mapping[str, Any]]] = {}
    aliases = {
        "harmonic": ("true_harmonic_errors",),
        "hif": ("true_hif_errors",),
        "three_phase_unbalance": (
            "true_three_phase_unbalance_errors",
            "true_unbalance_errors",
        ),
    }
    for family, keys in aliases.items():
        raw: Any = None
        present = False
        for key in keys:
            if key in scenario:
                raw = scenario[key]
                present = True
                break
            if key in hidden:
                raw = hidden[key]
                present = True
                break
        if not present:
            result[family] = []
            continue
        rows = _as_sequence(raw)
        if rows is None or not all(isinstance(item, Mapping) for item in rows):
            problems.append(f"true_{family}_localization_targets_malformed")
            result[family] = []
        else:
            result[family] = [item for item in rows if isinstance(item, Mapping)]
    scenario_family = str(scenario.get("scenario_family") or "").lower()
    expected_by_name = {
        "harmonic": "harmonic",
        "hif": "hif",
        "unbalance": "three_phase_unbalance",
    }
    for token, family in expected_by_name.items():
        if token in scenario_family and not result[family]:
            problems.append(f"true_{family}_localization_targets_missing")
    return result, problems


def _explanations(final_state: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], bool]:
    raw = final_state.get("explained_anomalies")
    if raw is None:
        return [], True
    rows = _as_sequence(raw)
    if rows is None:
        return [], False
    return [item for item in rows if isinstance(item, Mapping)], all(
        isinstance(item, Mapping) for item in rows
    )


def _canonical_diagnostic_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"unbalance", "three-phase-unbalance", "three_phase_unbalance"}:
        return "three_phase_unbalance"
    return text


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_value(sources: Sequence[Mapping[str, Any]], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            if source.get(key) is not None:
                return source[key]
    return None


def _bus_number(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("bus"):
            text = text[3:]
        elif text.startswith("b"):
            text = text[1:]
        value = text
    return _nonnegative_integer(value)


def _declared_tolerance(
    truth: Mapping[str, Any], keys: Sequence[str], default: float
) -> float | None:
    for key in keys:
        if truth.get(key) is not None:
            value = _finite_number(truth[key])
            return value if value is not None and value >= 0.0 else None
    return float(default)


def _diagnostic_match(
    family: str,
    truth: Mapping[str, Any],
    explanation: Mapping[str, Any],
    tolerances: ReleaseAuditTolerances,
) -> bool | None:
    """Return True/False, or None when required localization evidence is absent."""
    detail = _mapping(explanation.get("detail"))
    estimated = _mapping(detail.get("estimated"))
    sources = (detail, estimated, explanation)
    if family == "harmonic":
        expected = _bus_number(
            _first_value((truth,), "bus_1based", "source_bus", "source_bus_1based")
        )
        observed = _bus_number(
            _first_value(
                sources,
                "bus_1based",
                "source_bus",
                "source_bus_1based",
                "best_candidate_bus_1based",
            )
        )
        tolerance = _declared_tolerance(
            truth,
            ("bus_index_tolerance", "localization_tolerance_buses"),
            tolerances.harmonic_bus_index,
        )
        if expected is None or observed is None or tolerance is None:
            return None
        return abs(observed - expected) <= tolerance
    if family == "hif":
        expected_branch = _branch_target(truth)
        observed_branch = _branch_target(
            {
                "branch_row0": _first_value(
                    sources, "candidate_branch_row0", "branch_row0"
                ),
                "line_index1": _first_value(sources, "line_index1"),
                "dss_element": _first_value(sources, "dss_element"),
            }
        )
        branch_tolerance = _declared_tolerance(
            truth,
            ("branch_row_tolerance", "localization_tolerance_branches"),
            tolerances.hif_branch_rows,
        )
        if expected_branch is None or observed_branch is None or branch_tolerance is None:
            return None
        if expected_branch[0] == observed_branch[0] == "branch_row0":
            if abs(int(expected_branch[1]) - int(observed_branch[1])) > branch_tolerance:
                return False
        elif expected_branch != observed_branch:
            return False
        expected_phase = _first_value((truth,), "phase", "candidate_phase")
        if expected_phase is not None:
            observed_phase = _first_value(sources, "phase", "candidate_phase")
            if observed_phase is None:
                return None
            if str(observed_phase).upper() != str(expected_phase).upper():
                return False
        expected_alpha = _finite_number(
            _first_value(
                (truth,),
                "alpha_from_from_bus",
                "split_ratio",
                "fault_position_fraction",
            )
        )
        if expected_alpha is not None:
            observed_alpha = _finite_number(
                _first_value(
                    sources,
                    "alpha_from_from_bus",
                    "split_ratio",
                    "fault_position_fraction",
                )
            )
            alpha_tolerance = _declared_tolerance(
                truth,
                ("alpha_tolerance", "localization_tolerance_alpha"),
                tolerances.hif_alpha_abs,
            )
            if observed_alpha is None or alpha_tolerance is None:
                return None
            if abs(observed_alpha - expected_alpha) > alpha_tolerance:
                return False
        return True
    if family == "three_phase_unbalance":
        expected = _bus_number(
            _first_value(
                (truth,),
                "bus_1based",
                "source_bus",
                "source_bus_1based",
                "unbalance_bus",
                "unbalance_bus_name",
            )
        )
        top_vuf = _as_sequence(detail.get("top_vuf_buses")) or []
        top = top_vuf[0] if top_vuf and isinstance(top_vuf[0], Mapping) else {}
        top_k_raw = _first_value(
            (truth,), "localization_top_k", "top_k_tolerance"
        )
        top_k = _nonnegative_integer(
            tolerances.unbalance_top_k if top_k_raw is None else top_k_raw
        )
        if expected is not None and top_k is not None and top_k >= 1 and top_vuf:
            ranked_buses = [
                _bus_number(item.get("bus"))
                for item in top_vuf[:top_k]
                if isinstance(item, Mapping)
            ]
            if expected in ranked_buses:
                return True
        observed = _bus_number(
            _first_value(
                (*sources, top),
                "bus_1based",
                "source_bus",
                "source_bus_1based",
                "max_vuf_bus_1based",
                "bus",
            )
        )
        tolerance = _declared_tolerance(
            truth,
            ("bus_index_tolerance", "localization_tolerance_buses"),
            tolerances.unbalance_bus_index,
        )
        if (
            expected is None
            or observed is None
            or tolerance is None
            or top_k is None
            or top_k < 1
        ):
            return None
        return abs(observed - expected) <= tolerance
    return None


def _audit_diagnostics(
    scenario: Mapping[str, Any],
    final_state: Mapping[str, Any],
    tolerances: ReleaseAuditTolerances,
    *,
    require_complete: bool,
) -> tuple[list[str], list[str], int, int]:
    truth_by_family, truth_problems = _diagnostic_truth(scenario)
    explanations, explanations_valid = _explanations(final_state)
    family_problems = list(truth_problems)
    localization_problems: list[str] = []
    remaining_diagnostic_faults = 0
    if not explanations_valid:
        family_problems.append("diagnostic_explanations_malformed")
    by_family: dict[str, list[Mapping[str, Any]]] = {
        family: [] for family in _DIAGNOSTIC_FAMILIES
    }
    for explanation in explanations:
        family = _canonical_diagnostic_family(explanation.get("family"))
        if family not in _DIAGNOSTIC_FAMILIES:
            family_problems.append("diagnostic_explanation_family_unsupported")
            continue
        by_family[family].append(explanation)
        if not truth_by_family[family]:
            family_problems.append("diagnostic_explanation_without_same_family_truth")
    for family, truth_rows in truth_by_family.items():
        family_explanations = by_family[family]
        if truth_rows and not family_explanations:
            if require_complete:
                family_problems.append(
                    "diagnostic_truth_has_no_same_family_explanation"
                )
            remaining_diagnostic_faults += len(truth_rows)
            continue
        for truth in truth_rows:
            matches = [
                _diagnostic_match(family, truth, explanation, tolerances)
                for explanation in family_explanations
            ]
            if any(match is True for match in matches):
                continue
            remaining_diagnostic_faults += 1
            if require_complete:
                if matches and any(match is False for match in matches):
                    localization_problems.append(
                        "diagnostic_localization_outside_tolerance"
                    )
                else:
                    localization_problems.append(
                        "diagnostic_localization_evidence_missing"
                    )
        for explanation in family_explanations:
            matches = [
                _diagnostic_match(family, truth, explanation, tolerances)
                for truth in truth_rows
            ]
            if truth_rows and not any(match is True for match in matches):
                if any(match is False for match in matches):
                    localization_problems.append(
                        "diagnostic_explanation_does_not_localize_any_truth"
                    )
                else:
                    localization_problems.append(
                        "diagnostic_explanation_localization_evidence_missing"
                    )
    return (
        family_problems,
        localization_problems,
        remaining_diagnostic_faults,
        len(explanations),
    )


def _ledger_containers(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers = [source]
    hidden = source.get("hidden_truth")
    if isinstance(hidden, Mapping):
        containers.append(hidden)
    return containers


def _ledger_truth_complete(source: Mapping[str, Any]) -> bool:
    declarations = [
        container.get("truth_complete")
        for container in _ledger_containers(source)
        if "truth_complete" in container
    ]
    return bool(declarations) and all(value is True for value in declarations)


def _complete_ledger_count(source: Mapping[str, Any]) -> tuple[int | None, bool]:
    declared_counts: list[int] = []
    for container in _ledger_containers(source):
        count: int | None = None
        if "remaining_true_fault_count" in container:
            raw = container.get("remaining_true_fault_count")
            count = _nonnegative_integer(raw)
            if count is None:
                return None, False
            remaining = container.get("remaining_true_faults")
            if remaining is not None:
                rows = _as_sequence(remaining)
                if rows is None or len(rows) != count:
                    return None, False
        elif "remaining_true_faults" in container:
            rows = _as_sequence(container.get("remaining_true_faults"))
            if rows is None:
                return None, False
            count = len(rows)
        else:
            family_keys = (
                "true_measurement_errors",
                "true_parameter_errors",
                "true_topology_errors",
            )
            if not any(key in container for key in family_keys):
                continue
            rows: list[Any] = []
            for key in family_keys:
                values = _as_sequence(container.get(key, []))
                if values is None:
                    return None, False
                rows.extend(values)
            count = len(rows)
        declared_counts.append(count)
    if not declared_counts or len(set(declared_counts)) != 1:
        return None, False
    return declared_counts[0], True


def _load_case(
    value: Any, case_loader: Callable[[Any], Any] | None
) -> tuple[Any, bool]:
    if isinstance(value, Mapping) or _as_sequence(value) is not None:
        return value, True
    if case_loader is None:
        return value, value is not None
    try:
        return case_loader(value), True
    except Exception:
        return None, False


def _case_values_close(
    observed: Any,
    expected: Any,
    *,
    case_loader: Callable[[Any], Any] | None,
    tolerances: ReleaseAuditTolerances,
) -> bool | None:
    observed_loaded, observed_ok = _load_case(observed, case_loader)
    expected_loaded, expected_ok = _load_case(expected, case_loader)
    if not observed_ok or not expected_ok:
        return None
    if (
        case_loader is None
        and isinstance(observed_loaded, (str, bytes))
        and isinstance(expected_loaded, (str, bytes))
        and observed_loaded != expected_loaded
    ):
        # Different case references are not proof of a physical mismatch.
        return None
    return _values_close(
        observed_loaded,
        expected_loaded,
        abs_tolerance=tolerances.final_case_abs,
        rel_tolerance=tolerances.final_case_rel,
    )


def _target_case_fields_match(
    scenario: Mapping[str, Any],
    observed: Any,
    expected: Any,
    *,
    case_loader: Callable[[Any], Any] | None,
    tolerances: ReleaseAuditTolerances,
) -> bool | None:
    """Compare correction targets while preservation checks guard healthy state.

    Parameter corpora may use a clean-case artifact captured at a different
    operating point than the controller's model.  Comparing that entire file
    would conflate exogenous loads and rating metadata with the corrected r/x
    target.  For branch-error scenarios, compare only declared target fields;
    ``_healthy_case_preserved`` independently compares all healthy branches
    and non-branch case components against the initial active case using the
    tighter preservation tolerances.
    """

    parameter_rows = _as_sequence(scenario.get("true_parameter_errors", []))
    topology_rows = _as_sequence(scenario.get("true_topology_errors", []))
    if parameter_rows is None or topology_rows is None:
        return None
    if not parameter_rows and not topology_rows:
        return _case_values_close(
            observed,
            expected,
            case_loader=case_loader,
            tolerances=tolerances,
        )

    observed_loaded, observed_ok = _load_case(observed, case_loader)
    if not observed_ok or not isinstance(observed_loaded, Mapping):
        return None
    observed_branches = _as_sequence(observed_loaded.get("branch"))
    if observed_branches is None:
        return None

    expected_loaded: Any = None
    expected_branches: list[Any] | None = None

    def expected_branch(row0: int) -> list[Any] | None:
        nonlocal expected_loaded, expected_branches
        if expected_branches is None:
            expected_loaded, expected_ok = _load_case(expected, case_loader)
            if not expected_ok or not isinstance(expected_loaded, Mapping):
                return None
            expected_branches = _as_sequence(expected_loaded.get("branch"))
        if expected_branches is None or not 0 <= row0 < len(expected_branches):
            return None
        return _as_sequence(expected_branches[row0])

    def observed_branch(row0: int) -> list[Any] | None:
        if not 0 <= row0 < len(observed_branches):
            return None
        return _as_sequence(observed_branches[row0])

    comparisons: list[tuple[Any, Any, bool]] = []
    for fault in parameter_rows:
        if not isinstance(fault, Mapping):
            return None
        target = _branch_target(fault)
        if target is None or target[0] != "branch_row0":
            return None
        row0 = int(target[1])
        actual_row = observed_branch(row0)
        if actual_row is None:
            return None
        parameter = str(fault.get("parameter") or "rx").strip().lower()
        columns = _PARAMETER_BRANCH_COLUMNS.get(parameter)
        if columns is None:
            return None
        clean_values = {2: fault.get("clean_r"), 3: fault.get("clean_x")}
        clean_row: list[Any] | None = None
        for column in columns:
            if not 0 <= column < len(actual_row):
                return None
            clean_value = clean_values.get(column)
            if clean_value is None:
                clean_row = clean_row or expected_branch(row0)
                if clean_row is None or not 0 <= column < len(clean_row):
                    return None
                clean_value = clean_row[column]
            comparisons.append((actual_row[column], clean_value, False))

    for fault in topology_rows:
        if not isinstance(fault, Mapping):
            return None
        target = _branch_target(fault)
        if target is None or target[0] != "branch_row0":
            return None
        row0 = int(target[1])
        actual_row = observed_branch(row0)
        if actual_row is None or len(actual_row) <= 10:
            return None
        clean_value = fault.get("expected_status")
        if clean_value is None:
            clean_row = expected_branch(row0)
            if clean_row is None or len(clean_row) <= 10:
                return None
            clean_value = clean_row[10]
        comparisons.append((actual_row[10], clean_value, True))

    if not comparisons:
        return None
    return all(
        _values_close(
            actual,
            clean,
            abs_tolerance=(0.0 if exact else tolerances.final_case_abs),
            rel_tolerance=(0.0 if exact else tolerances.final_case_rel),
        )
        for actual, clean, exact in comparisons
    )


def _healthy_case_preserved(
    initial: Any,
    final: Any,
    mutable_columns_by_row: Mapping[int, set[int]],
    *,
    case_loader: Callable[[Any], Any] | None,
    tolerances: ReleaseAuditTolerances,
) -> bool | None:
    initial_loaded, initial_ok = _load_case(initial, case_loader)
    final_loaded, final_ok = _load_case(final, case_loader)
    if not initial_ok or not final_ok:
        return None
    if not isinstance(initial_loaded, Mapping) or not isinstance(final_loaded, Mapping):
        if initial_loaded == final_loaded:
            return True
        return None if case_loader is None else False
    initial_branch = _as_sequence(initial_loaded.get("branch"))
    final_branch = _as_sequence(final_loaded.get("branch"))
    if initial_branch is None or final_branch is None:
        return None
    if len(initial_branch) != len(final_branch):
        return False
    for index, (before, after) in enumerate(zip(initial_branch, final_branch)):
        mutable_columns = mutable_columns_by_row.get(index)
        if mutable_columns is None:
            if not _values_close(
                after,
                before,
                abs_tolerance=tolerances.case_abs,
                rel_tolerance=tolerances.case_rel,
            ):
                return False
            continue
        before_row = _as_sequence(before)
        after_row = _as_sequence(after)
        if before_row is None or after_row is None or len(before_row) != len(after_row):
            return None
        if any(column < 0 or column >= len(before_row) for column in mutable_columns):
            return None
        for column, (before_value, after_value) in enumerate(
            zip(before_row, after_row)
        ):
            if column in mutable_columns:
                continue
            if not _values_close(
                after_value,
                before_value,
                abs_tolerance=tolerances.case_abs,
                rel_tolerance=tolerances.case_rel,
            ):
                return False
    initial_other = {
        str(key): value
        for key, value in initial_loaded.items()
        if str(key) != "branch"
    }
    final_other = {
        str(key): value
        for key, value in final_loaded.items()
        if str(key) != "branch"
    }
    return _values_close(
        final_other,
        initial_other,
        abs_tolerance=tolerances.case_abs,
        rel_tolerance=tolerances.case_rel,
    )


_PARAMETER_CLEAN_FIELDS = {
    2: "clean_r",
    3: "clean_x",
    4: "clean_b",
    8: "clean_tap",
    9: "clean_shift",
}


def _case_branches(
    value: Any, case_loader: Callable[[Any], Any] | None
) -> list[Any] | None:
    loaded, loaded_ok = _load_case(value, case_loader)
    if not loaded_ok or not isinstance(loaded, Mapping):
        return None
    return _as_sequence(loaded.get("branch"))


def _branch_row(rows: list[Any] | None, row0: int) -> list[Any] | None:
    if rows is None or not 0 <= row0 < len(rows):
        return None
    return _as_sequence(rows[row0])


def _vector_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _distance_tolerance(
    initial: Sequence[float],
    clean: Sequence[float],
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> float:
    # Scale the absolute allowance for a multi-column target while keeping the
    # relative term independent of the final value.  Including the final value
    # in the scale would let a large regression manufacture its own tolerance.
    dimensions = max(1, len(clean))
    return math.sqrt(dimensions) * abs_tolerance + rel_tolerance * max(
        _vector_distance(initial, [0.0] * len(initial)),
        _vector_distance(clean, [0.0] * len(clean)),
    )


def _accepted_target_nonregression(
    scenario: Mapping[str, Any],
    *,
    final_measurements: Any,
    final_case: Any,
    case_loader: Callable[[Any], Any] | None,
    tolerances: ReleaseAuditTolerances,
    target_problems: Sequence[str],
    accepted_measurements: set[int],
    accepted_parameters: set[tuple[str, Any]],
    accepted_topology: set[tuple[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Prove that every committed true target is no worse than at reset.

    This invariant is intentionally outcome-independent.  In particular, an
    operator escalation may retain an unresolved true fault, but it may not
    retain a committed correction that increased that fault's distance from
    clean truth.  Missing or contradictory target-state evidence fails closed.
    """

    problems: list[str] = []
    evidence: list[dict[str, Any]] = []
    if target_problems:
        return ["accepted_target_nonregression_target_evidence_invalid"], evidence

    initial_measurements = _as_sequence(scenario.get("measurements"))
    final_measurement_rows = _as_sequence(final_measurements)
    clean_state = scenario.get("clean_state")
    clean_state = clean_state if isinstance(clean_state, Mapping) else {}
    clean_measurements = _as_sequence(
        clean_state.get("measurements", scenario.get("clean_measurements"))
    )
    measurement_truth_rows = _as_sequence(
        scenario.get("true_measurement_errors", [])
    )

    for index in sorted(accepted_measurements):
        malformed = False
        if (
            initial_measurements is None
            or final_measurement_rows is None
            or not 0 <= index < len(initial_measurements)
            or not 0 <= index < len(final_measurement_rows)
            or measurement_truth_rows is None
        ):
            malformed = True
        initial_value = (
            _finite_number(initial_measurements[index])
            if not malformed and initial_measurements is not None
            else None
        )
        final_value = (
            _finite_number(final_measurement_rows[index])
            if not malformed and final_measurement_rows is not None
            else None
        )
        matching_faults = (
            [
                fault
                for fault in measurement_truth_rows
                if isinstance(fault, Mapping)
                and _nonnegative_integer(fault.get("index", fault.get("index0")))
                == index
            ]
            if measurement_truth_rows is not None
            else []
        )
        explicit_clean: list[float] = []
        for fault in matching_faults:
            if "clean" in fault:
                clean_value = _finite_number(fault.get("clean"))
                if clean_value is None:
                    malformed = True
                else:
                    explicit_clean.append(clean_value)
        clean_vector_value = (
            _finite_number(clean_measurements[index])
            if clean_measurements is not None and 0 <= index < len(clean_measurements)
            else None
        )
        clean_candidates = list(explicit_clean)
        if clean_vector_value is not None:
            clean_candidates.append(clean_vector_value)
        if not matching_faults or not clean_candidates:
            malformed = True
        if clean_measurements is not None and clean_vector_value is None:
            malformed = True
        if clean_candidates and any(
            not _values_close(
                value,
                clean_candidates[0],
                abs_tolerance=tolerances.measurement_abs,
                rel_tolerance=tolerances.measurement_rel,
            )
            for value in clean_candidates[1:]
        ):
            malformed = True
        if initial_value is None or final_value is None:
            malformed = True
        if malformed:
            problems.append(
                "accepted_measurement_nonregression_evidence_missing_or_malformed"
            )
            evidence.append(
                {"family": "measurement", "index0": index, "status": "malformed"}
            )
            continue
        clean_value = clean_candidates[0]
        initial_distance = abs(initial_value - clean_value)
        final_distance = abs(final_value - clean_value)
        allowance = _distance_tolerance(
            [initial_value],
            [clean_value],
            abs_tolerance=tolerances.measurement_abs,
            rel_tolerance=tolerances.measurement_rel,
        )
        passed = final_distance <= initial_distance + allowance
        evidence.append(
            {
                "family": "measurement",
                "index0": index,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "tolerance": allowance,
                "status": "passed" if passed else "regressed",
            }
        )
        if not passed:
            problems.append("accepted_measurement_target_regressed")

    needs_case = bool(accepted_parameters or accepted_topology)
    initial_branches = (
        _case_branches(scenario.get("case"), case_loader) if needs_case else None
    )
    final_branches = _case_branches(final_case, case_loader) if needs_case else None
    clean_case = clean_state.get("case", scenario.get("clean_case"))
    clean_branches = _case_branches(clean_case, case_loader) if needs_case else None

    parameter_truth_rows = _as_sequence(scenario.get("true_parameter_errors", []))
    for target in sorted(accepted_parameters, key=lambda item: (str(item[0]), str(item[1]))):
        malformed = target[0] != "branch_row0" or parameter_truth_rows is None
        row0 = int(target[1]) if target[0] == "branch_row0" else -1
        initial_row = _branch_row(initial_branches, row0)
        final_row = _branch_row(final_branches, row0)
        clean_row = _branch_row(clean_branches, row0)
        matching_faults = (
            [
                fault
                for fault in parameter_truth_rows
                if isinstance(fault, Mapping) and _branch_target(fault) == target
            ]
            if parameter_truth_rows is not None
            else []
        )
        clean_by_column: dict[int, float] = {}
        columns: set[int] = set()
        for fault in matching_faults:
            parameter = str(fault.get("parameter") or "rx").strip().lower()
            fault_columns = _PARAMETER_BRANCH_COLUMNS.get(parameter)
            if fault_columns is None:
                malformed = True
                continue
            for column in fault_columns:
                columns.add(column)
                clean_field = _PARAMETER_CLEAN_FIELDS[column]
                raw_clean = fault.get(clean_field)
                if raw_clean is None and clean_row is not None and column < len(clean_row):
                    raw_clean = clean_row[column]
                numeric_clean = _finite_number(raw_clean)
                if numeric_clean is None:
                    malformed = True
                    continue
                previous = clean_by_column.get(column)
                if previous is not None and not _values_close(
                    previous,
                    numeric_clean,
                    abs_tolerance=tolerances.final_case_abs,
                    rel_tolerance=tolerances.final_case_rel,
                ):
                    malformed = True
                clean_by_column[column] = numeric_clean
        if not matching_faults or not columns or set(clean_by_column) != columns:
            malformed = True
        ordered_columns = sorted(columns)
        if (
            initial_row is None
            or final_row is None
            or any(
                column >= len(initial_row) or column >= len(final_row)
                for column in ordered_columns
            )
        ):
            malformed = True
        initial_values = (
            [_finite_number(initial_row[column]) for column in ordered_columns]
            if initial_row is not None
            and all(column < len(initial_row) for column in ordered_columns)
            else []
        )
        final_values = (
            [_finite_number(final_row[column]) for column in ordered_columns]
            if final_row is not None
            and all(column < len(final_row) for column in ordered_columns)
            else []
        )
        clean_values = [
            clean_by_column[column]
            for column in ordered_columns
            if column in clean_by_column
        ]
        if any(value is None for value in initial_values + final_values):
            malformed = True
        if malformed:
            problems.append(
                "accepted_parameter_nonregression_evidence_missing_or_malformed"
            )
            evidence.append(
                {"family": "parameter", "target": list(target), "status": "malformed"}
            )
            continue
        numeric_initial = [float(value) for value in initial_values if value is not None]
        numeric_final = [float(value) for value in final_values if value is not None]
        initial_distance = _vector_distance(numeric_initial, clean_values)
        final_distance = _vector_distance(numeric_final, clean_values)
        allowance = _distance_tolerance(
            numeric_initial,
            clean_values,
            abs_tolerance=tolerances.final_case_abs,
            rel_tolerance=tolerances.final_case_rel,
        )
        passed = final_distance <= initial_distance + allowance
        evidence.append(
            {
                "family": "parameter",
                "target": list(target),
                "columns": ordered_columns,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "tolerance": allowance,
                "status": "passed" if passed else "regressed",
            }
        )
        if not passed:
            problems.append("accepted_parameter_target_regressed")

    topology_truth_rows = _as_sequence(scenario.get("true_topology_errors", []))
    for target in sorted(accepted_topology, key=lambda item: (str(item[0]), str(item[1]))):
        malformed = target[0] != "branch_row0" or topology_truth_rows is None
        row0 = int(target[1]) if target[0] == "branch_row0" else -1
        initial_row = _branch_row(initial_branches, row0)
        final_row = _branch_row(final_branches, row0)
        clean_row = _branch_row(clean_branches, row0)
        matching_faults = (
            [
                fault
                for fault in topology_truth_rows
                if isinstance(fault, Mapping) and _branch_target(fault) == target
            ]
            if topology_truth_rows is not None
            else []
        )
        clean_candidates: list[float] = []
        for fault in matching_faults:
            raw_clean = fault.get("expected_status")
            if raw_clean is None and clean_row is not None and len(clean_row) > 10:
                raw_clean = clean_row[10]
            numeric_clean = _finite_number(raw_clean)
            if numeric_clean is None:
                malformed = True
            else:
                clean_candidates.append(numeric_clean)
        initial_status = (
            _finite_number(initial_row[10])
            if initial_row is not None and len(initial_row) > 10
            else None
        )
        final_status = (
            _finite_number(final_row[10])
            if final_row is not None and len(final_row) > 10
            else None
        )
        if (
            not matching_faults
            or not clean_candidates
            or initial_status is None
            or final_status is None
            or any(value not in {0.0, 1.0} for value in clean_candidates)
            or initial_status not in {0.0, 1.0}
            or final_status not in {0.0, 1.0}
            or len(set(clean_candidates)) != 1
        ):
            malformed = True
        if malformed:
            problems.append(
                "accepted_topology_nonregression_evidence_missing_or_malformed"
            )
            evidence.append(
                {"family": "topology", "target": list(target), "status": "malformed"}
            )
            continue
        clean_status = clean_candidates[0]
        initial_distance = abs(initial_status - clean_status)
        final_distance = abs(final_status - clean_status)
        passed = final_distance <= initial_distance
        evidence.append(
            {
                "family": "topology",
                "target": list(target),
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "tolerance": 0.0,
                "status": "passed" if passed else "regressed",
            }
        )
        if not passed:
            problems.append("accepted_topology_target_regressed")

    return list(dict.fromkeys(problems)), evidence


def _not_applicable_declarations(
    scenario: Mapping[str, Any], supplied: Mapping[str, str] | None
) -> tuple[dict[str, str], list[str]]:
    declarations: dict[str, str] = {}
    problems: list[str] = []
    release_audit = scenario.get("release_audit")
    embedded = (
        release_audit.get("not_applicable")
        if isinstance(release_audit, Mapping)
        else scenario.get("release_audit_not_applicable")
    )
    if isinstance(release_audit, Mapping):
        for raw_name, declaration in release_audit.items():
            if not isinstance(declaration, Mapping):
                continue
            if str(declaration.get("status") or "").lower() != "not_applicable":
                continue
            name = str(raw_name)
            if name not in ALLOWED_NOT_APPLICABLE_CHECKS:
                problems.append("not_applicable_check_unknown_or_prohibited")
                continue
            reason = declaration.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                problems.append("not_applicable_reason_missing")
            else:
                declarations[name] = reason.strip()
    for raw in (embedded, supplied):
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            problems.append("not_applicable_declarations_malformed")
            continue
        for key, reason in raw.items():
            name = str(key)
            if name not in ALLOWED_NOT_APPLICABLE_CHECKS:
                problems.append("not_applicable_check_unknown_or_prohibited")
                continue
            if not isinstance(reason, str) or not reason.strip():
                problems.append("not_applicable_reason_missing")
                continue
            declarations[name] = reason.strip()
    return declarations, problems


def _validated_not_applicable_declarations(
    scenario: Mapping[str, Any],
    final_state: Mapping[str, Any],
    declarations: Mapping[str, str],
    *,
    terminal: bool,
    resolved: bool,
    target_problems: Sequence[str],
    measurement_targets: set[int],
    parameter_targets: set[tuple[str, Any]],
    topology_targets: set[tuple[str, Any]],
    diagnostic_family_problems: Sequence[str],
    localization_problems: Sequence[str],
    remaining_diagnostic_faults: int,
    diagnostic_claim_count: int,
) -> tuple[dict[str, str], list[str]]:
    """Authorize the sole waiver only for a proven explanation-only result."""

    validated = dict(declarations)
    if not resolved or FINAL_MEASUREMENTS_CHECK not in validated:
        # The declaration has no effect on non-resolved outcomes.  In
        # particular, a generated diagnostic scenario that safely escalates
        # should not be quarantined merely because its resolved-path contract
        # is present but unused.
        return validated, []

    problems: list[str] = []
    release_audit = scenario.get("release_audit")
    release_audit = release_audit if isinstance(release_audit, Mapping) else {}
    if (
        release_audit.get("explanation_only_contract")
        != EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT
    ):
        problems.append(
            "final_measurements_not_applicable_contract_marker_missing_or_invalid"
        )

    expected_truth_family = {
        "harmonic": "harmonic",
        "hif": "hif",
        "three_phase_unbalance": "three_phase_unbalance",
    }.get(str(scenario.get("scenario_family") or "").strip().lower())
    if expected_truth_family is None:
        problems.append(
            "final_measurements_not_applicable_requires_pure_diagnostic_family"
        )

    diagnostic_truth, diagnostic_truth_problems = _diagnostic_truth(scenario)
    populated_diagnostic_families = {
        family for family, rows in diagnostic_truth.items() if rows
    }
    if (
        expected_truth_family is None
        or populated_diagnostic_families != {expected_truth_family}
        or diagnostic_truth_problems
    ):
        problems.append(
            "final_measurements_not_applicable_diagnostic_truth_missing_or_malformed"
        )

    if (
        measurement_targets
        or parameter_targets
        or topology_targets
        or target_problems
    ):
        problems.append(
            "final_measurements_not_applicable_prohibits_correction_truth"
        )

    corrections, corrections_well_formed = _accepted_corrections(final_state)
    if corrections or not corrections_well_formed:
        problems.append(
            "final_measurements_not_applicable_prohibits_accepted_corrections"
        )

    if (
        not terminal
        or diagnostic_claim_count < 1
        or diagnostic_family_problems
        or localization_problems
        or remaining_diagnostic_faults != 0
    ):
        problems.append(
            "final_measurements_not_applicable_requires_resolved_diagnostic_localization"
        )

    if problems:
        validated.pop(FINAL_MEASUREMENTS_CHECK, None)
    return validated, list(dict.fromkeys(problems))


def audit_episode_against_truth(
    scenario: Mapping[str, Any],
    final_state: Mapping[str, Any],
    *,
    terminal: bool,
    terminal_outcome: str | None = None,
    active_physical_state: Mapping[str, Any] | None = None,
    remaining_truth: Mapping[str, Any] | None = None,
    case_loader: Callable[[Any], Any] | None = None,
    tolerances: ReleaseAuditTolerances | Mapping[str, Any] | None = None,
    not_applicable: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit one completed episode against hidden truth, without mutating inputs.

    ``final_state`` is the policy-safe decision summary.  For a resolved
    outcome, ``active_physical_state`` should be the store payload returned by
    ``env.store.get_state(env.store.active_state_id)`` and
    ``remaining_truth`` should be the final oracle truth payload.  A case
    loader is required when initial/final/clean cases use different path
    references rather than directly comparable mappings.
    """

    if not isinstance(scenario, Mapping) or not isinstance(final_state, Mapping):
        raise TypeError("scenario and final_state must be mappings")
    release_audit_config = scenario.get("release_audit")
    if tolerances is None and isinstance(release_audit_config, Mapping):
        embedded_tolerances = release_audit_config.get("tolerances")
        if embedded_tolerances is not None:
            tolerances = embedded_tolerances
    tolerance_profile = _coerce_tolerances(tolerances)
    checks: dict[str, dict[str, Any]] = {}
    all_problems: list[str] = []

    def record(name: str, problems: Sequence[str], *, status: str | None = None) -> None:
        unique = list(dict.fromkeys(str(item) for item in problems))
        resolved_status = status or ("failed" if unique else "passed")
        checks[name] = {"status": resolved_status, "problems": unique}
        all_problems.extend(unique)

    (
        target_problems,
        measurement_targets,
        parameter_targets,
        topology_targets,
        accepted_measurement_targets,
        accepted_parameter_targets,
        accepted_topology_targets,
    ) = _audit_accepted_targets(scenario, final_state)
    record(ACCEPTED_TARGETS_CHECK, target_problems)

    outcome = str(terminal_outcome) if terminal_outcome is not None else None
    resolved = outcome == "resolved"
    (
        diagnostic_family_problems,
        localization_problems,
        remaining_diagnostic_faults,
        diagnostic_claim_count,
    ) = _audit_diagnostics(
        scenario,
        final_state,
        tolerance_profile,
        require_complete=resolved,
    )

    declarations, declaration_problems = _not_applicable_declarations(
        scenario, not_applicable
    )
    declarations, contract_problems = _validated_not_applicable_declarations(
        scenario,
        final_state,
        declarations,
        terminal=terminal,
        resolved=resolved,
        target_problems=target_problems,
        measurement_targets=measurement_targets,
        parameter_targets=parameter_targets,
        topology_targets=topology_targets,
        diagnostic_family_problems=diagnostic_family_problems,
        localization_problems=localization_problems,
        remaining_diagnostic_faults=remaining_diagnostic_faults,
        diagnostic_claim_count=diagnostic_claim_count,
    )
    declaration_problems.extend(contract_problems)
    if declaration_problems:
        record("audit_evidence_contract", declaration_problems)

    if resolved and not terminal:
        record("terminal_contract", ["resolved_outcome_requires_terminal_episode"])

    def resolved_check(name: str, problems: Sequence[str]) -> None:
        if not resolved:
            record(name, [], status="not_required")
        elif name in declarations:
            checks[name] = {
                "status": "not_applicable",
                "reason": declarations[name],
                "problems": [],
            }
        else:
            record(name, problems)

    def diagnostic_claim_check(name: str, problems: Sequence[str]) -> None:
        if resolved:
            resolved_check(name, problems)
        elif diagnostic_claim_count or problems:
            # A handoff need not explain every remaining diagnostic fault, but
            # any explanation it does persist is a release claim and must be
            # same-family and correctly localized.  ``not_applicable`` cannot
            # waive a false partial diagnosis.
            record(name, problems)
        else:
            record(name, [], status="not_required")

    diagnostic_claim_check(DIAGNOSTIC_FAMILY_CHECK, diagnostic_family_problems)
    diagnostic_claim_check(DIAGNOSTIC_LOCALIZATION_CHECK, localization_problems)

    derived_remaining_count = (
        len(measurement_targets - accepted_measurement_targets)
        + len(parameter_targets - accepted_parameter_targets)
        + len(topology_targets - accepted_topology_targets)
        + remaining_diagnostic_faults
    )
    remaining_problems: list[str] = []
    if derived_remaining_count != 0:
        remaining_problems.append("resolved_episode_has_remaining_true_faults")
    if remaining_truth is not None:
        if not isinstance(remaining_truth, Mapping) or not _ledger_truth_complete(
            remaining_truth
        ):
            remaining_problems.append("supplied_remaining_truth_ledger_incomplete")
        else:
            supplied_count, supplied_valid = _complete_ledger_count(remaining_truth)
            if not supplied_valid or supplied_count is None:
                remaining_problems.append(
                    "supplied_remaining_truth_ledger_missing_or_malformed"
                )
            elif supplied_count != derived_remaining_count:
                remaining_problems.append(
                    "supplied_remaining_truth_ledger_disagrees_with_derived"
                )
    resolved_check(REMAINING_FAULTS_CHECK, remaining_problems)
    if resolved and REMAINING_FAULTS_CHECK in checks:
        checks[REMAINING_FAULTS_CHECK][
            "derived_remaining_fault_count"
        ] = derived_remaining_count
        checks[REMAINING_FAULTS_CHECK][
            "evidence_source"
        ] = "offline_scenario_truth_derivation"

    physical = active_physical_state
    if physical is None and (
        "measurements" in final_state or "case" in final_state
    ):
        physical = final_state
    physical = physical if isinstance(physical, Mapping) else {}
    initial_measurements = scenario.get("measurements")
    final_measurements = physical.get("measurements")
    final_case = physical.get("case")
    clean_state = scenario.get("clean_state")
    clean_state = clean_state if isinstance(clean_state, Mapping) else {}
    clean_measurements = clean_state.get(
        "measurements", scenario.get("clean_measurements")
    )

    if terminal:
        nonregression_problems, nonregression_evidence = (
            _accepted_target_nonregression(
                scenario,
                final_measurements=final_measurements,
                final_case=final_case,
                case_loader=case_loader,
                tolerances=tolerance_profile,
                target_problems=target_problems,
                accepted_measurements=accepted_measurement_targets,
                accepted_parameters=accepted_parameter_targets,
                accepted_topology=accepted_topology_targets,
            )
        )
        record(ACCEPTED_TARGET_NONREGRESSION_CHECK, nonregression_problems)
        checks[ACCEPTED_TARGET_NONREGRESSION_CHECK]["target_evidence"] = (
            nonregression_evidence
        )
    else:
        record(ACCEPTED_TARGET_NONREGRESSION_CHECK, [], status="not_required")

    healthy_measurement_problems: list[str] = []
    before = _as_sequence(initial_measurements)
    after = _as_sequence(final_measurements)
    if before is None or after is None or len(before) != len(after):
        healthy_measurement_problems.append(
            "healthy_measurement_preservation_evidence_missing_or_malformed"
        )
    elif any(index >= len(before) for index in measurement_targets):
        healthy_measurement_problems.append("true_measurement_target_out_of_range")
    elif any(
        not _values_close(
            after[index],
            before[index],
            abs_tolerance=tolerance_profile.measurement_abs,
            rel_tolerance=tolerance_profile.measurement_rel,
        )
        for index in range(len(before))
        if index not in measurement_targets
    ):
        healthy_measurement_problems.append("healthy_measurement_modified")
    # Healthy-state preservation is an invariant, not a resolution claim.  A
    # safe operator handoff may retain true faults, but it may not corrupt an
    # unrelated meter.  Resolved diagnostic-only scenarios retain their
    # explicit reason-bearing not-applicable contract for compatibility.
    if resolved and HEALTHY_MEASUREMENTS_CHECK in declarations:
        resolved_check(HEALTHY_MEASUREMENTS_CHECK, healthy_measurement_problems)
    else:
        record(HEALTHY_MEASUREMENTS_CHECK, healthy_measurement_problems)

    mutable_columns_by_row: dict[int, set[int]] = {}
    for fault in _as_sequence(scenario.get("true_parameter_errors", [])) or []:
        if not isinstance(fault, Mapping):
            continue
        target = _branch_target(fault)
        parameter = str(fault.get("parameter") or "rx").strip().lower()
        columns = _PARAMETER_BRANCH_COLUMNS.get(parameter)
        if target is None or target[0] != "branch_row0" or columns is None:
            continue
        mutable_columns_by_row.setdefault(int(target[1]), set()).update(columns)
    for fault in _as_sequence(scenario.get("true_topology_errors", [])) or []:
        if not isinstance(fault, Mapping):
            continue
        target = _branch_target(fault)
        if target is None or target[0] != "branch_row0":
            continue
        mutable_columns_by_row.setdefault(int(target[1]), set()).add(10)
    initial_case = scenario.get("case")
    preserved = _healthy_case_preserved(
        initial_case,
        final_case,
        mutable_columns_by_row,
        case_loader=case_loader,
        tolerances=tolerance_profile,
    )
    healthy_case_problems = (
        []
        if preserved is True
        else [
            "healthy_case_component_modified"
            if preserved is False
            else "healthy_case_preservation_evidence_missing_or_unloadable"
        ]
    )
    if resolved and HEALTHY_CASE_CHECK in declarations:
        resolved_check(HEALTHY_CASE_CHECK, healthy_case_problems)
    else:
        record(HEALTHY_CASE_CHECK, healthy_case_problems)

    clean_measurement_rows = _as_sequence(clean_measurements)
    final_measurement_rows = _as_sequence(final_measurements)
    if clean_measurement_rows is None or final_measurement_rows is None:
        final_measurement_problems = [
            "final_clean_measurement_evidence_missing_or_malformed"
        ]
    elif not _values_close(
        final_measurement_rows,
        clean_measurement_rows,
        abs_tolerance=tolerance_profile.measurement_abs,
        rel_tolerance=tolerance_profile.measurement_rel,
    ):
        final_measurement_problems = ["final_measurements_outside_clean_tolerance"]
    else:
        final_measurement_problems = []
    resolved_check(FINAL_MEASUREMENTS_CHECK, final_measurement_problems)

    clean_case = clean_state.get("case", scenario.get("clean_case"))
    case_match = _target_case_fields_match(
        scenario,
        final_case,
        clean_case,
        case_loader=case_loader,
        tolerances=tolerance_profile,
    )
    final_case_problems = (
        []
        if case_match is True
        else [
            "final_case_outside_clean_tolerance"
            if case_match is False
            else "final_clean_case_evidence_missing_or_unloadable"
        ]
    )
    resolved_check(FINAL_CASE_CHECK, final_case_problems)

    unique_problems = list(dict.fromkeys(all_problems))
    return {
        "audit_version": AUDIT_VERSION,
        "scenario_id": str(scenario.get("scenario_id") or ""),
        "physical_root_fingerprint": str(
            scenario.get("physical_root_fingerprint") or ""
        ),
        "scenario_family": str(scenario.get("scenario_family") or "unknown"),
        "terminal": bool(terminal),
        "terminal_outcome": outcome,
        "checks": checks,
        "tolerances": asdict(tolerance_profile),
        "problems": unique_problems,
        "quarantined": bool(unique_problems),
    }


__all__ = [
    "ACCEPTED_TARGET_NONREGRESSION_CHECK",
    "ACCEPTED_TARGETS_CHECK",
    "ALLOWED_NOT_APPLICABLE_CHECKS",
    "AUDIT_VERSION",
    "DIAGNOSTIC_FAMILY_CHECK",
    "DIAGNOSTIC_LOCALIZATION_CHECK",
    "EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT",
    "FINAL_CASE_CHECK",
    "FINAL_MEASUREMENTS_CHECK",
    "HEALTHY_CASE_CHECK",
    "HEALTHY_MEASUREMENTS_CHECK",
    "REMAINING_FAULTS_CHECK",
    "ReleaseAuditTolerances",
    "audit_episode_against_truth",
]
