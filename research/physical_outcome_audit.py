"""Replay research evaluations and audit physical and sequential outcomes.

The lightweight :mod:`research.evaluate` artifact intentionally stores only
the executed action sequence.  This module deterministically replays that
sequence in the production environment, then performs a hidden-truth audit on
the *final active store state*.  Hidden truth is consumed only after each
trajectory has finished; none of the audit output is fed back to a policy or
training example.

The primary six-way episode outcome uses trajectory termination as the outer
contract: generation aborts and horizon loops take precedence over a physical
snapshot that happens to look repaired.  ``final_active_state_class`` is also
reported so a physically repaired policy that subsequently escalates or loops
can be distinguished from a policy that never repaired the system.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CONTEXT_TOOLS,
    CORRECTION_TOOLS,
    DIAGNOSTIC_TOOLS,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    safe_normalize_action,
)
from psse_env.dagger.evaluator import (
    _active_physical_state,
    _successful_action_advanced,
)
from psse_env.dagger.release_audit import (
    ACCEPTED_TARGET_NONREGRESSION_CHECK,
    ACCEPTED_TARGETS_CHECK,
    FINAL_CASE_CHECK,
    FINAL_MEASUREMENTS_CHECK,
    HEALTHY_CASE_CHECK,
    HEALTHY_MEASUREMENTS_CHECK,
    audit_episode_against_truth,
)
from psse_env.dagger.release_factories import (
    deterministic_case_loader,
    production_environment_factory,
    select_observable_expert_actions,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.private_target_matching import (
    PARAMETER_BRANCH_COLUMNS,
    canonical_branch_target,
    correction_family,
    measurement_action_targets,
    measurement_fault_target,
    private_target_tolerances,
)

from .collect import load_scenarios
from .train import file_sha256


AUDIT_SCHEMA_VERSION = 2

EXACT_PHYSICAL_RECOVERY = "exact_physical_recovery"
PARTIAL_RECOVERY = "partial_recovery"
FALSE_INTERVENTION = "false_intervention"
NO_PHYSICAL_PROGRESS = "no_physical_progress"
GENERATION_ABORT = "not_assessable_generation_abort"
LOOP_BEFORE_STABLE_FINAL_STATE = "loop_before_stable_final_state"

PHYSICAL_CLASSES = frozenset(
    {
        EXACT_PHYSICAL_RECOVERY,
        PARTIAL_RECOVERY,
        FALSE_INTERVENTION,
        NO_PHYSICAL_PROGRESS,
    }
)
OUTCOME_CLASSES = frozenset({*PHYSICAL_CLASSES, GENERATION_ABORT, LOOP_BEFORE_STABLE_FINAL_STATE})

_DISPOSITION_EXPECTATION = {
    "ACCEPT_FINAL": "commit",
    "ACCEPT_PARTIAL": "commit",
    "REJECT": "rollback",
    "INCONCLUSIVE": "rollback",
}

_ROUTINE_READ_ONLY_TOOLS = frozenset(
    {
        *CONTEXT_TOOLS,
        *DIAGNOSTIC_TOOLS,
        RUN_WLS,
        VERIFY_CANDIDATE,
        RUN_ALTERNATIVE_TEST,
    }
)

_PARAMETER_CLEAN_FIELDS = {
    2: "clean_r",
    3: "clean_x",
    4: "clean_b",
    8: "clean_tap",
    9: "clean_shift",
}

_BRANCH_COLUMN_ALIASES = {
    2: ("r", "br_r"),
    3: ("x", "br_x"),
    4: ("b", "br_b"),
    8: ("tap", "ratio"),
    9: ("shift", "angle"),
    10: ("status", "br_status"),
}

_EVIDENCE_INTEGRITY_TOKENS = (
    "missing",
    "malformed",
    "unloadable",
    "unverifiable",
    "incomplete",
    "evidence_invalid",
    "out_of_range",
)


def _scenario_execution(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the runtime envelope used by ``research.evaluate``.

    Replay must not inject grouping metadata, release-audit configuration, or a
    synthetic ``truth_complete`` field that the original policy evaluation did
    not receive.  The richer private envelope is constructed separately by
    :func:`_scenario_audit` after the trajectory has finished.
    """

    execution = scenario.get("execution")
    if not isinstance(execution, Mapping):
        return copy.deepcopy(dict(scenario))
    runtime = copy.deepcopy(dict(execution))
    audit = scenario.get("audit")
    truth = audit.get("truth") if isinstance(audit, Mapping) else None
    if isinstance(truth, Mapping):
        clean_state = truth.get("clean_state")
        if isinstance(clean_state, Mapping):
            if "case" in clean_state:
                runtime.setdefault("clean_case", copy.deepcopy(clean_state["case"]))
            if "measurements" in clean_state:
                runtime.setdefault(
                    "clean_measurements",
                    copy.deepcopy(clean_state["measurements"]),
                )
        for key, value in truth.items():
            if key in {"truth_complete", "clean_state"}:
                continue
            runtime.setdefault(str(key), copy.deepcopy(value))
    return runtime


def _scenario_audit(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return the private offline truth envelope used only after replay."""

    audit_scenario = _scenario_execution(scenario)
    audit = scenario.get("audit")
    truth = audit.get("truth") if isinstance(audit, Mapping) else None
    if isinstance(truth, Mapping):
        clean_state = truth.get("clean_state")
        if isinstance(clean_state, Mapping):
            if "case" in clean_state:
                audit_scenario.setdefault(
                    "clean_case", copy.deepcopy(clean_state["case"])
                )
            if "measurements" in clean_state:
                audit_scenario.setdefault(
                    "clean_measurements", copy.deepcopy(clean_state["measurements"])
                )
        for key, value in truth.items():
            if key == "clean_state":
                continue
            audit_scenario.setdefault(str(key), copy.deepcopy(value))
    elif "truth_complete" in scenario:
        audit_scenario["truth_complete"] = copy.deepcopy(scenario["truth_complete"])
    if isinstance(audit, Mapping) and isinstance(audit.get("release_audit"), Mapping):
        audit_scenario["release_audit"] = copy.deepcopy(dict(audit["release_audit"]))
    grouping = scenario.get("grouping")
    if isinstance(grouping, Mapping):
        for key in (
            "scenario_id",
            "root_scenario_id",
            "scenario_family",
            "physical_root_fingerprint",
            "error_cardinality",
        ):
            if grouping.get(key) is not None:
                audit_scenario.setdefault(key, copy.deepcopy(grouping[key]))
    return audit_scenario


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    return str(
        scenario.get("scenario_id")
        or grouping.get("scenario_id")
        or grouping.get("root_scenario_id")
        or f"index{index}"
    )


def _scenario_family(scenario: Mapping[str, Any]) -> str:
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    value = scenario.get("scenario_family") or grouping.get("scenario_family")
    return str(value or "unknown").strip().lower().replace("_", "-")


def _truth_rows(scenario: Mapping[str, Any], family: str) -> list[Mapping[str, Any]]:
    rows = scenario.get(f"true_{family}_errors") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _fault_count(scenario: Mapping[str, Any]) -> int:
    return sum(
        len(_truth_rows(scenario, family))
        for family in ("measurement", "parameter", "topology")
    )


def _truth_target_key(family: str, fault: Mapping[str, Any]) -> str | None:
    if family == "measurement":
        target = measurement_fault_target(fault)
        return f"measurement:{target}" if target is not None else None
    target = canonical_branch_target(fault)
    if family not in {"parameter", "topology"} or target is None:
        return None
    return f"{family}:{target[0]}:{target[1]}"


def _sequence_list(value: Any) -> list[Any] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001 - malformed evidence fails closed
            return None
    return list(value) if isinstance(value, Sequence) else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _close_final_value(
    observed: Any,
    expected: Any,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> bool | None:
    left = _finite_float(observed)
    right = _finite_float(expected)
    if left is None or right is None:
        return None
    return math.isclose(left, right, abs_tol=abs_tolerance, rel_tol=rel_tolerance)


def _loaded_case(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    try:
        loaded = deterministic_case_loader(value)
    except Exception:  # noqa: BLE001 - malformed evidence fails closed
        return None
    return loaded if isinstance(loaded, Mapping) else None


def _branch_rows(value: Any) -> list[Any] | None:
    loaded = _loaded_case(value)
    return _sequence_list(loaded.get("branch")) if loaded is not None else None


def _branch_row_for_target(
    rows: Sequence[Any] | None,
    target: tuple[str, Any],
) -> Any | None:
    if rows is None:
        return None
    if target[0] == "branch_row0":
        row0 = int(target[1])
        return rows[row0] if 0 <= row0 < len(rows) else None
    if target[0] == "branch_id":
        expected = str(target[1])
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for key in ("branch_id", "id", "name", "cb_name", "dss_element"):
                if row.get(key) is not None and str(row[key]) == expected:
                    return row
    return None


def _branch_column_value(row: Any, column: int) -> Any | None:
    if isinstance(row, Mapping):
        for key in _BRANCH_COLUMN_ALIASES.get(column, ()):
            if row.get(key) is not None:
                return row[key]
        return None
    values = _sequence_list(row)
    return values[column] if values is not None and 0 <= column < len(values) else None


def _final_fault_is_resolved(
    *,
    family: str,
    fault: Mapping[str, Any],
    scenario: Mapping[str, Any],
    active_physical_state: Mapping[str, Any],
) -> bool | None:
    profile = private_target_tolerances(scenario)
    if family == "measurement":
        target = measurement_fault_target(fault)
        final_measurements = _sequence_list(active_physical_state.get("measurements"))
        clean_measurements = _sequence_list(scenario.get("clean_measurements"))
        if target is None or final_measurements is None or not 0 <= target < len(
            final_measurements
        ):
            return None
        expected = next(
            (
                fault[key]
                for key in (
                    "clean",
                    "clean_value",
                    "true_value",
                    "expected_value",
                    "correct_value",
                )
                if fault.get(key) is not None
            ),
            None,
        )
        if expected is None and clean_measurements is not None and 0 <= target < len(
            clean_measurements
        ):
            expected = clean_measurements[target]
        return _close_final_value(
            final_measurements[target],
            expected,
            abs_tolerance=profile.measurement_abs,
            rel_tolerance=profile.measurement_rel,
        )

    target = canonical_branch_target(fault)
    if target is None:
        return None
    final_row = _branch_row_for_target(
        _branch_rows(active_physical_state.get("case")), target
    )
    clean_row = _branch_row_for_target(_branch_rows(scenario.get("clean_case")), target)
    if final_row is None:
        return None
    if family == "parameter":
        raw_parameter = fault.get("parameter", fault.get("field"))
        if raw_parameter is None:
            raw_parameter = (
                "rx"
                if fault.get("clean_r") is not None or fault.get("clean_x") is not None
                else "x"
            )
        columns = PARAMETER_BRANCH_COLUMNS.get(str(raw_parameter).strip().lower())
        if not columns:
            return None
        results: list[bool] = []
        for column in columns:
            expected = fault.get(_PARAMETER_CLEAN_FIELDS[column])
            if expected is None and len(columns) == 1:
                expected = next(
                    (
                        fault[key]
                        for key in (
                            "clean",
                            "clean_value",
                            "true_value",
                            "expected_value",
                            "correct_value",
                        )
                        if fault.get(key) is not None
                    ),
                    None,
                )
            if expected is None and clean_row is not None:
                expected = _branch_column_value(clean_row, column)
            close = _close_final_value(
                _branch_column_value(final_row, column),
                expected,
                abs_tolerance=profile.final_case_abs,
                rel_tolerance=profile.final_case_rel,
            )
            if close is None:
                return None
            results.append(close)
        return all(results)
    if family == "topology":
        expected = fault.get("expected_status")
        if expected is None and clean_row is not None:
            expected = _branch_column_value(clean_row, 10)
        return _close_final_value(
            _branch_column_value(final_row, 10),
            expected,
            abs_tolerance=profile.final_case_abs,
            rel_tolerance=profile.final_case_rel,
        )
    return None


def _direct_fault_resolution(
    *,
    scenario: Mapping[str, Any],
    active_physical_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve every injected target from the final physical store directly.

    The mutable oracle retirement ledger is deliberately absent from this
    calculation.  It is retained later only as a disagreement diagnostic.
    """

    problems: list[str] = []
    evidence: list[dict[str, Any]] = []
    truth_keys: set[str] = set()
    restored_keys: set[str] = set()
    if scenario.get("truth_complete") is not True:
        problems.append("scenario_truth_complete_not_explicit_true")
    if not isinstance(active_physical_state, Mapping):
        problems.append("final_active_physical_state_missing")
    for family in ("measurement", "parameter", "topology"):
        field = f"true_{family}_errors"
        raw_rows = scenario.get(field, [])
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes, bytearray, Mapping)
        ):
            problems.append(f"{field}_missing_or_malformed")
            continue
        for index, raw_fault in enumerate(raw_rows):
            if not isinstance(raw_fault, Mapping):
                problems.append(f"{field}_{index}_malformed")
                continue
            fault = copy.deepcopy(dict(raw_fault))
            key = _truth_target_key(family, fault)
            if key is None:
                problems.append(f"{field}_{index}_target_malformed")
                continue
            if key in truth_keys:
                problems.append(f"duplicate_true_fault_target:{key}")
                continue
            truth_keys.add(key)
            restored: bool | None = None
            if isinstance(active_physical_state, Mapping):
                try:
                    restored = _final_fault_is_resolved(
                        family=family,
                        fault=fault,
                        scenario=scenario,
                        active_physical_state=active_physical_state,
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed with evidence
                    problems.append(
                        f"{field}_{index}_comparison_error:{type(exc).__name__}:{exc}"
                    )
            if restored is None:
                problems.append(f"{field}_{index}_comparison_unavailable")
            elif restored:
                restored_keys.add(key)
            evidence.append(
                {
                    "family": family,
                    "fault_index": index,
                    "target_key": key,
                    "restored_in_final_active_state": restored,
                }
            )
    return {
        "truth_complete": scenario.get("truth_complete") is True,
        "true_target_keys": sorted(truth_keys),
        "restored_true_target_keys": sorted(restored_keys),
        "initial_true_target_count": len(truth_keys),
        "restored_true_target_count": len(restored_keys),
        "remaining_true_target_count": len(truth_keys - restored_keys),
        "problems": sorted(set(problems)),
        "per_fault": evidence,
    }


def _remaining_ledger_diagnostic(
    remaining_truth: Mapping[str, Any] | None,
    *,
    direct_remaining_count: int,
) -> dict[str, Any]:
    problems: list[str] = []
    count: int | None = None
    complete = bool(
        isinstance(remaining_truth, Mapping)
        and remaining_truth.get("truth_complete") is True
    )
    if not isinstance(remaining_truth, Mapping):
        problems.append("oracle_remaining_truth_missing")
    elif not complete:
        problems.append("oracle_remaining_truth_incomplete")
    else:
        count = 0
        for family in ("measurement", "parameter", "topology"):
            rows = remaining_truth.get(f"true_{family}_errors", [])
            if not isinstance(rows, Sequence) or isinstance(
                rows, (str, bytes, bytearray, Mapping)
            ):
                problems.append(f"oracle_remaining_true_{family}_errors_malformed")
                count = None
                break
            if any(not isinstance(row, Mapping) for row in rows):
                problems.append(f"oracle_remaining_true_{family}_errors_malformed")
                count = None
                break
            count += len(rows)
    agrees = count == direct_remaining_count if count is not None else None
    if agrees is False:
        problems.append("oracle_remaining_truth_disagrees_with_direct_final_state")
    return {
        "truth_complete": complete,
        "remaining_true_error_count": count,
        "agrees_with_direct_final_state": agrees,
        "problems": sorted(set(problems)),
    }


def _oracle_truth(env: Any, history: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    getter = getattr(env, "get_oracle_state", None)
    if not callable(getter):
        return None
    try:
        state = getter(history)
    except TypeError:
        state = getter()
    truth_dict = getattr(state, "truth_dict", None)
    if callable(truth_dict):
        truth = truth_dict()
    elif isinstance(state, Mapping):
        truth = state
    else:
        return None
    return copy.deepcopy(dict(truth)) if isinstance(truth, Mapping) else None


def _oracle_disposition(env: Any, history: list[Mapping[str, Any]]) -> str | None:
    state = env.current_state() if hasattr(env, "current_state") else {}
    value = state.get("candidate_disposition") if isinstance(state, Mapping) else None
    if value is None and hasattr(env, "get_oracle_state"):
        try:
            oracle = env.get_oracle_state(history)
        except TypeError:
            oracle = env.get_oracle_state()
        value = (
            oracle.get("candidate_disposition")
            if isinstance(oracle, Mapping)
            else getattr(oracle, "candidate_disposition", None)
        )
    if value is None:
        return None
    return str(getattr(value, "value", value)).strip().upper() or None


def _active_hash(env: Any, state: Mapping[str, Any]) -> str | None:
    physical = _active_physical_state(env, state)
    if not isinstance(physical, Mapping):
        return None
    value = physical.get("state_hash")
    return str(value) if value is not None else None


def _runtime_hash(output: Mapping[str, Any], env: Any, state: Mapping[str, Any]) -> str | None:
    metrics = output.get("tool_metrics")
    if isinstance(metrics, Mapping) and metrics.get("state_hash") is not None:
        return str(metrics["state_hash"])
    return _active_hash(env, state)


def _parse_recorded_action(row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the already-internal action persisted by ``research.evaluate``."""

    tool = row.get("tool")
    raw_arguments = row.get("arguments", {})
    if isinstance(raw_arguments, str):
        if raw_arguments.endswith("..."):
            raise ValueError("recorded action arguments were truncated")
        raw_arguments = json.loads(raw_arguments)
    if not isinstance(raw_arguments, Mapping):
        raise ValueError("recorded action arguments are not a mapping")
    return {"tool": tool, "arguments": copy.deepcopy(dict(raw_arguments))}


def _policy_observation(env: Any, history: list[Mapping[str, Any]]) -> dict[str, Any]:
    raw = env.get_policy_observation(history)
    if hasattr(raw, "as_dict"):
        raw = raw.as_dict()
    if not isinstance(raw, Mapping):
        raise TypeError("policy observation is not a mapping")
    return copy.deepcopy(dict(raw))


def _expert_action_signatures(
    env: Any,
    expert: ExpertPolicyOracle,
    history: list[Mapping[str, Any]],
) -> tuple[list[str], str | None]:
    try:
        selection = select_observable_expert_actions(
            policy_observation=_policy_observation(env, history),
            expert_oracle=expert,
        )
        signatures = [action_signature(action) for action in selection.actions]
        return signatures, None
    except Exception as exc:  # noqa: BLE001 - audit records missing comparator evidence
        return [], f"{type(exc).__name__}: {exc}"


def _expected_disposition_action(disposition: str | None) -> str | None:
    return _DISPOSITION_EXPECTATION.get(str(disposition or "").upper())


def _predicted_disposition_action(tool: str) -> str:
    if tool == COMMIT_STATE:
        return "commit"
    if tool == ROLLBACK_STATE:
        return "rollback"
    return "other"


def _committed_target_events(final_state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return committed target-edit events, retaining duplicate targets."""

    targets: list[str] = []
    problems: list[str] = []
    accepted = final_state.get("accepted_corrections") or []
    if not isinstance(accepted, Sequence) or isinstance(accepted, (str, bytes)):
        return [], ["accepted_corrections_malformed"]
    for index, item in enumerate(accepted):
        if not isinstance(item, Mapping):
            problems.append(f"accepted_correction_{index}_malformed")
            continue
        raw = item.get("source_action") or item.get("action") or item
        try:
            action = safe_normalize_action(raw)
        except Exception:  # noqa: BLE001 - malformed audit evidence fails closed
            problems.append(f"accepted_correction_{index}_action_malformed")
            continue
        family = correction_family(action)
        arguments = action.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        if family == "measurement":
            indices = measurement_action_targets(arguments)
            if not indices:
                problems.append(f"accepted_correction_{index}_target_missing")
                continue
            targets.extend(f"measurement:{value}" for value in sorted(indices))
        elif family in {"parameter", "topology"}:
            target = canonical_branch_target(arguments)
            if target is None:
                problems.append(f"accepted_correction_{index}_target_missing")
                continue
            targets.append(f"{family}:{target[0]}:{target[1]}")
        elif action.get("tool") in CORRECTION_TOOLS:
            problems.append(f"accepted_correction_{index}_family_unknown")
    return targets, problems


def _committed_target_keys(final_state: Mapping[str, Any]) -> tuple[set[str], list[str]]:
    """Compatibility helper returning unique committed target keys."""

    events, problems = _committed_target_events(final_state)
    return set(events), problems


def _check(strict: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    checks = strict.get("checks")
    value = checks.get(name) if isinstance(checks, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _strict_physical_summary(
    *,
    scenario: Mapping[str, Any],
    final_state: Mapping[str, Any],
    active_physical_state: Mapping[str, Any] | None,
    remaining_truth: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Audit the active snapshot under a counterfactual resolved contract.

    The counterfactual forces the strict auditor to run all final-state checks
    even for an operator escalation or horizon.  It does not replace the
    observed terminal outcome and is explicitly labelled as snapshot evidence.
    """

    strict = audit_episode_against_truth(
        scenario,
        final_state,
        terminal=True,
        terminal_outcome="resolved",
        active_physical_state=active_physical_state,
        remaining_truth=remaining_truth,
        case_loader=deterministic_case_loader,
    )
    direct = _direct_fault_resolution(
        scenario=scenario,
        active_physical_state=active_physical_state,
    )
    initial_faults = int(direct["initial_true_target_count"])
    remaining_faults = int(direct["remaining_true_target_count"])
    corrected_faults = int(direct["restored_true_target_count"])
    true_target_keys = set(direct["true_target_keys"])
    restored_target_keys = set(direct["restored_true_target_keys"])

    accepted_target_events, target_parse_problems = _committed_target_events(final_state)
    unique_accepted_targets = set(accepted_target_events)
    true_committed_targets = restored_target_keys & unique_accepted_targets
    false_committed_target_events = [
        target for target in accepted_target_events if target not in true_target_keys
    ]
    healthy_checks = {
        HEALTHY_MEASUREMENTS_CHECK: _check(strict, HEALTHY_MEASUREMENTS_CHECK),
        HEALTHY_CASE_CHECK: _check(strict, HEALTHY_CASE_CHECK),
    }
    false_intervention_problems = [
        str(problem)
        for check in healthy_checks.values()
        for problem in check.get("problems") or []
        if str(problem)
        in {"healthy_measurement_modified", "healthy_case_component_modified"}
    ]
    false_intervention_problems.extend(
        f"committed_target_outside_truth:{target}"
        for target in false_committed_target_events
    )
    false_intervention = bool(false_intervention_problems)

    evidence_problems = [*direct["problems"], *target_parse_problems]
    for name in (
        ACCEPTED_TARGETS_CHECK,
        ACCEPTED_TARGET_NONREGRESSION_CHECK,
        HEALTHY_MEASUREMENTS_CHECK,
        HEALTHY_CASE_CHECK,
        FINAL_MEASUREMENTS_CHECK,
        FINAL_CASE_CHECK,
    ):
        check = _check(strict, name)
        for problem in check.get("problems") or []:
            text = str(problem)
            if any(token in text for token in _EVIDENCE_INTEGRITY_TOKENS):
                evidence_problems.append(text)
    physical_assessable = not evidence_problems

    ledger = _remaining_ledger_diagnostic(
        remaining_truth,
        direct_remaining_count=remaining_faults,
    )

    exact_check_names = (
        ACCEPTED_TARGETS_CHECK,
        ACCEPTED_TARGET_NONREGRESSION_CHECK,
        HEALTHY_MEASUREMENTS_CHECK,
        HEALTHY_CASE_CHECK,
        FINAL_MEASUREMENTS_CHECK,
        FINAL_CASE_CHECK,
    )
    exact_checks_passed = all(
        _check(strict, name).get("status") in {"passed", "not_applicable"}
        for name in exact_check_names
    )

    if not physical_assessable:
        physical_class = None
    elif false_intervention:
        physical_class = FALSE_INTERVENTION
    elif remaining_faults == 0 and exact_checks_passed:
        physical_class = EXACT_PHYSICAL_RECOVERY
    elif 0 < corrected_faults < initial_faults:
        physical_class = PARTIAL_RECOVERY
    else:
        physical_class = NO_PHYSICAL_PROGRESS

    event_count = len(accepted_target_events)
    unique_target_count = len(unique_accepted_targets)
    true_committed_count = len(true_committed_targets)

    return {
        "physical_assessable": physical_assessable,
        "final_active_state_class": physical_class,
        "initial_true_error_count": initial_faults,
        "remaining_true_error_count": remaining_faults,
        "true_errors_corrected": corrected_faults,
        "committed_correction_target_count": event_count,
        "committed_correction_target_event_count": event_count,
        "unique_committed_correction_target_count": unique_target_count,
        "true_committed_correction_count": true_committed_count,
        "correction_precision": (
            true_committed_count / event_count
            if physical_assessable and event_count
            else None
        ),
        "unique_target_correction_precision": (
            true_committed_count / unique_target_count
            if physical_assessable and unique_target_count
            else None
        ),
        "correction_recall": (
            corrected_faults / initial_faults
            if physical_assessable and initial_faults
            else None
        ),
        "false_intervention": false_intervention if physical_assessable else None,
        "false_intervention_problems": sorted(set(false_intervention_problems)),
        "exact_physical_checks_passed": exact_checks_passed,
        "physical_evidence_problems": sorted(set(evidence_problems)),
        "direct_final_state_fault_audit": direct,
        "oracle_remaining_truth_ledger_diagnostic": ledger,
        "strict_counterfactual_resolution_audit": strict,
        "metric_unit": "committed_physical_target_events",
        "unique_target_metric_unit": "unique_committed_physical_targets_per_episode",
    }


def classify_episode(
    *,
    generation_abort: bool,
    loop: bool,
    physical_class: str | None,
) -> str | None:
    """Apply the documented precedence for the mutually exclusive outcome."""

    if generation_abort:
        return GENERATION_ABORT
    if loop:
        return LOOP_BEFORE_STABLE_FINAL_STATE
    return physical_class if physical_class in PHYSICAL_CLASSES else None


def _first_loop_divergence(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for row in trace:
        if row.get("candidate_disposition_error"):
            return {
                "step": row.get("step"),
                "class": "candidate_commit_rollback_error",
            }
        if row.get("repeated_action_signature") and row.get("tool") in CONTEXT_TOOLS:
            return {"step": row.get("step"), "class": "repeated_context_request"}
        if row.get("repeated_action_signature") and row.get("tool") in CORRECTION_TOOLS:
            return {"step": row.get("step"), "class": "repeated_correction"}
        if row.get("nonconsecutive_state_hash_revisit"):
            return {
                "step": row.get("step"),
                "class": "nonconsecutive_state_hash_revisit",
            }
        arguments = row.get("arguments")
        if (
            row.get("tool") == ASK_FOR_MORE_EVIDENCE
            and isinstance(arguments, Mapping)
            and arguments.get("request") == RECOVERY_OPTIONS_EXHAUSTED_REQUEST
            and row.get("expert_admissible") is False
        ):
            return {"step": row.get("step"), "class": "premature_escalation"}
        if (
            row.get("controller_no_progress")
            and row.get("expert_admissible") is True
        ):
            return {
                "step": row.get("step"),
                "class": "expert_admissible_valid_action_no_progress",
            }
    for row in trace:
        if row.get("expert_admissible") is False:
            return {"step": row.get("step"), "class": "other_expert_disagreement"}
    return None


def replay_episode(
    env: Any,
    scenario: Mapping[str, Any],
    recorded_episode: Mapping[str, Any],
    *,
    scenario_index: int,
    expert_steps: int | None,
) -> dict[str, Any]:
    """Replay one compact evaluation record and return an auditable episode row."""

    runtime_scenario = _scenario_execution(scenario)
    env.reset(runtime_scenario)
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    replay_problems: list[str] = []
    expected_scenario_id = _scenario_id(scenario, scenario_index)
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    expected_root_id = scenario.get("root_scenario_id") or grouping.get(
        "root_scenario_id"
    )
    expected_fingerprint = scenario.get("physical_root_fingerprint") or grouping.get(
        "physical_root_fingerprint"
    )
    recorded_scenario_id = recorded_episode.get("scenario_id")
    recorded_root_id = recorded_episode.get("root_scenario_id")
    recorded_fingerprint = recorded_episode.get("physical_root_fingerprint")
    if (
        recorded_scenario_id is not None
        and str(recorded_scenario_id).strip()
        and str(recorded_scenario_id) != expected_scenario_id
    ):
        replay_problems.append("scenario_id_mismatch")
    if (
        recorded_root_id is not None
        and str(recorded_root_id).strip()
        and str(recorded_root_id) != str(expected_root_id)
    ):
        replay_problems.append("root_scenario_id_mismatch")
    if (
        recorded_fingerprint is not None
        and str(recorded_fingerprint).strip()
        and str(recorded_fingerprint) != str(expected_fingerprint)
    ):
        replay_problems.append("physical_root_fingerprint_mismatch")
    if recorded_fingerprint is not None and str(recorded_fingerprint).strip():
        scenario_alignment_basis = (
            "validated_physical_root_fingerprint"
            if str(recorded_fingerprint) == str(expected_fingerprint)
            else "mismatched_physical_root_fingerprint"
        )
    elif recorded_root_id is not None and str(recorded_root_id).strip():
        scenario_alignment_basis = (
            "validated_root_scenario_id"
            if str(recorded_root_id) == str(expected_root_id)
            else "mismatched_root_scenario_id"
        )
    elif recorded_scenario_id is not None and str(recorded_scenario_id).strip():
        scenario_alignment_basis = (
            "validated_scenario_id"
            if str(recorded_scenario_id) == expected_scenario_id
            else "mismatched_scenario_id"
        )
    else:
        scenario_alignment_basis = "ordered_index_report_id_missing"
    seen_signatures: set[str] = set()
    state = copy.deepcopy(dict(env.current_state()))
    initial_hash = _active_hash(env, state)
    last_hash_step = {initial_hash: -1} if initial_hash is not None else {}
    no_progress_streak = 0
    max_no_progress_streak = 0
    expert = ExpertPolicyOracle(
        process_oracle=env.process_oracle,
        candidate_oracle=env.candidate_quality_oracle,
    )

    actions = recorded_episode.get("actions") or []
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        actions = []
        replay_problems.append("recorded_actions_malformed")
    for step, stored in enumerate(actions):
        if not isinstance(stored, Mapping):
            replay_problems.append(f"step_{step}_record_malformed")
            break
        try:
            action = _parse_recorded_action(stored)
        except Exception as exc:  # noqa: BLE001
            replay_problems.append(
                f"step_{step}_action_unavailable:{type(exc).__name__}:{exc}"
            )
            break
        action = safe_normalize_action(action)
        signature = action_signature(action)
        repeated_signature = signature in seen_signatures
        seen_signatures.add(signature)
        before = copy.deepcopy(dict(env.current_state()))
        disposition = _oracle_disposition(env, history)
        expected_disposition = _expected_disposition_action(disposition)
        predicted_disposition = _predicted_disposition_action(str(action["tool"]))
        candidate_disposition_error = bool(
            expected_disposition is not None
            and predicted_disposition != expected_disposition
        )
        expert_signatures, expert_error = _expert_action_signatures(env, expert, history)
        expert_admissible = signature in set(expert_signatures) if not expert_error else None
        try:
            after, output = env.step(action)
        except Exception as exc:  # noqa: BLE001
            replay_problems.append(f"step_{step}_execution_error:{type(exc).__name__}:{exc}")
            break
        after = copy.deepcopy(dict(after)) if isinstance(after, Mapping) else {}
        output = copy.deepcopy(dict(output)) if isinstance(output, Mapping) else {}
        status = output.get("execution_status")
        error_code = output.get("error_code")
        if stored.get("status") != status:
            replay_problems.append(f"step_{step}_status_mismatch")
        if stored.get("error_code") != error_code:
            replay_problems.append(f"step_{step}_error_code_mismatch")
        terminal_after = bool(getattr(env, "terminal", False))
        advanced = bool(
            status == "success"
            and _successful_action_advanced(
                before=before,
                after=after,
                output=output,
                terminal=terminal_after,
            )
        )
        valid_no_progress = bool(status == "success" and not advanced)
        controller_progress_eligible = str(action["tool"]) not in _ROUTINE_READ_ONLY_TOOLS
        controller_no_progress = bool(
            controller_progress_eligible and valid_no_progress
        )
        if controller_progress_eligible:
            if controller_no_progress:
                no_progress_streak += 1
                max_no_progress_streak = max(
                    max_no_progress_streak, no_progress_streak
                )
            else:
                no_progress_streak = 0
        runtime_hash = _runtime_hash(output, env, after)
        previous_hash_step = (
            last_hash_step.get(runtime_hash) if runtime_hash is not None else None
        )
        state_revisit = bool(
            previous_hash_step is not None and previous_hash_step < step - 1
        )
        if runtime_hash is not None:
            last_hash_step[runtime_hash] = step
        row = {
            "step": step,
            "tool": action["tool"],
            "arguments": copy.deepcopy(action["arguments"]),
            "action_signature": signature,
            "status": status,
            "error_code": error_code,
            "progress_advanced": advanced,
            "valid_no_progress": valid_no_progress,
            "controller_progress_eligible": controller_progress_eligible,
            "controller_no_progress": controller_no_progress,
            "repeated_action_signature": repeated_signature,
            "runtime_state_hash": runtime_hash,
            "state_hash_revisit": state_revisit,
            "nonconsecutive_state_hash_revisit": state_revisit,
            "candidate_disposition": disposition,
            "expected_candidate_action": expected_disposition,
            "predicted_candidate_action": (
                predicted_disposition if expected_disposition is not None else None
            ),
            "candidate_disposition_error": candidate_disposition_error,
            "expert_admissible": expert_admissible,
            "expert_admissible_action_signatures": expert_signatures,
            "expert_label_error": expert_error,
        }
        trace.append(row)
        history.append({"action": copy.deepcopy(action), "tool_output": copy.deepcopy(output)})

    final_state = copy.deepcopy(dict(env.current_state()))
    terminal = bool(getattr(env, "terminal", False))
    terminal_outcome = getattr(env, "terminal_outcome", None)
    recorded_outcome = recorded_episode.get("terminal_outcome")
    if (str(terminal_outcome) if terminal_outcome is not None else None) != (
        str(recorded_outcome) if recorded_outcome is not None else None
    ):
        replay_problems.append("terminal_outcome_mismatch")
    if len(trace) != int(recorded_episode.get("steps") or 0):
        replay_problems.append("step_count_mismatch")

    first_error = recorded_episode.get("first_error")
    generation_abort = bool(first_error)
    loop = bool(recorded_episode.get("horizon_truncated") and not generation_abort)
    active_physical_state = _active_physical_state(env, final_state)
    remaining_truth = _oracle_truth(env, history)
    audit_scenario = _scenario_audit(scenario)
    physical = _strict_physical_summary(
        scenario=audit_scenario,
        final_state=final_state,
        active_physical_state=active_physical_state,
        remaining_truth=remaining_truth,
    )
    if replay_problems:
        physical["physical_assessable"] = False
        physical["final_active_state_class"] = None
        physical["physical_evidence_problems"] = sorted(
            set(physical["physical_evidence_problems"] + replay_problems)
        )

    outcome_class = classify_episode(
        generation_abort=generation_abort,
        loop=loop,
        physical_class=physical.get("final_active_state_class"),
    )
    valid_steps = sum(row["status"] == "success" for row in trace)
    physical_state_unchanged_steps = sum(
        bool(row["valid_no_progress"]) for row in trace
    )
    controller_steps = sum(
        row["status"] == "success" and row["controller_progress_eligible"]
        for row in trace
    )
    no_progress = sum(bool(row["controller_no_progress"]) for row in trace)
    repeated = sum(bool(row["repeated_action_signature"]) for row in trace)
    revisits = sum(bool(row["state_hash_revisit"]) for row in trace)
    disposition_rows = [
        row for row in trace if row.get("expected_candidate_action") is not None
    ]
    first_state_revisit = next(
        (
            {"step": row["step"], "runtime_state_hash": row["runtime_state_hash"]}
            for row in trace
            if row.get("state_hash_revisit")
        ),
        None,
    )
    first_no_progress = next(
        (
            {"step": row["step"], "action_signature": row["action_signature"]}
            for row in trace
            if row.get("controller_no_progress")
        ),
        None,
    )
    first_repeated_action = next(
        (
            {"step": row["step"], "action_signature": row["action_signature"]}
            for row in trace
            if row.get("repeated_action_signature")
        ),
        None,
    )
    result = {
        "scenario_index": scenario_index,
        "scenario_id": expected_scenario_id,
        "recorded_scenario_id": recorded_scenario_id,
        "root_scenario_id": expected_root_id,
        "recorded_root_scenario_id": recorded_root_id,
        "physical_root_fingerprint": expected_fingerprint,
        "recorded_physical_root_fingerprint": recorded_fingerprint,
        "scenario_alignment_basis": scenario_alignment_basis,
        "scenario_family": _scenario_family(scenario),
        "recorded_terminal_outcome": recorded_outcome,
        "replayed_terminal": terminal,
        "replayed_terminal_outcome": terminal_outcome,
        "generation_abort": generation_abort,
        "generation_abort_error": first_error,
        "loop_before_stable_final_state": loop,
        "episode_outcome_class": outcome_class,
        "replay_matches_record": not replay_problems,
        "replay_problems": replay_problems,
        "steps": len(trace),
        "expert_steps": expert_steps,
        "step_difference_relative_to_expert": (
            len(trace) - expert_steps if expert_steps is not None else None
        ),
        "excess_steps_relative_to_expert": (
            max(len(trace) - expert_steps, 0) if expert_steps is not None else None
        ),
        "valid_action_count": valid_steps,
        "physical_state_unchanged_valid_action_count": (
            physical_state_unchanged_steps
        ),
        "physical_state_unchanged_valid_action_rate": (
            physical_state_unchanged_steps / valid_steps if valid_steps else None
        ),
        "physical_state_unchanged_rate_note": (
            "includes routine read-only actions; use controller no-progress "
            "for unproductive mutation/control actions"
        ),
        "controller_valid_action_count": controller_steps,
        "routine_read_only_valid_action_count": valid_steps - controller_steps,
        "no_progress_valid_action_count": no_progress,
        "no_progress_valid_action_rate": (
            no_progress / controller_steps if controller_steps else None
        ),
        "no_progress_rate_denominator": "controller_valid_actions_excluding_read_only",
        "repeated_action_signature_count": repeated,
        "repeated_action_signature_rate": repeated / len(trace) if trace else None,
        "state_hash_observation_count": sum(
            row.get("runtime_state_hash") is not None for row in trace
        ),
        "state_hash_revisit_count": revisits,
        "state_hash_revisit_rate": (
            revisits
            / sum(row.get("runtime_state_hash") is not None for row in trace)
            if any(row.get("runtime_state_hash") is not None for row in trace)
            else None
        ),
        "max_no_progress_streak": max_no_progress_streak,
        "first_no_progress_valid_action": first_no_progress,
        "first_repeated_action_signature": first_repeated_action,
        "first_state_hash_revisit": first_state_revisit,
        "candidate_disposition_opportunities": len(disposition_rows),
        "candidate_disposition_errors": sum(
            bool(row["candidate_disposition_error"]) for row in disposition_rows
        ),
        "first_loop_divergence": _first_loop_divergence(trace) if loop else None,
        "trace": trace,
        **physical,
    }
    return result


def expert_step_baseline(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
    env_factory: Callable[[], Any] = production_environment_factory,
) -> list[dict[str, Any]]:
    """Run the observable expert once per scenario for a step-count baseline."""

    env = env_factory()
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        env.reset(_scenario_execution(scenario))
        expert = ExpertPolicyOracle(
            process_oracle=env.process_oracle,
            candidate_oracle=env.candidate_quality_oracle,
        )
        history: list[dict[str, Any]] = []
        problem: str | None = None
        for _ in range(max_steps):
            try:
                selection = select_observable_expert_actions(
                    policy_observation=_policy_observation(env, history),
                    expert_oracle=expert,
                )
                action = selection.preferred_action
                if action is None:
                    problem = "expert_returned_no_action"
                    break
                # The rule expert operates on the environment's internal tool
                # schema already (including explicit state identifiers).
                action = safe_normalize_action(action)
                _, output = env.step(action)
                history.append({"action": action, "tool_output": output})
            except Exception as exc:  # noqa: BLE001
                problem = f"{type(exc).__name__}: {exc}"
                break
            if getattr(env, "terminal", False):
                break
        terminal = bool(getattr(env, "terminal", False))
        if not terminal and problem is None:
            problem = "expert_reached_step_horizon"
        rows.append(
            {
                "scenario_index": index,
                "scenario_id": _scenario_id(scenario, index),
                "steps": len(history) if terminal and problem is None else None,
                "terminal": terminal,
                "terminal_outcome": getattr(env, "terminal_outcome", None),
                "problem": problem,
            }
        )
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_counts = Counter(
        str(row.get("episode_outcome_class") or "unclassified") for row in episodes
    )
    snapshot_counts = Counter(
        str(row.get("final_active_state_class") or "not_assessable") for row in episodes
    )
    assessable = [row for row in episodes if row.get("physical_assessable")]
    faulted = [row for row in episodes if int(row.get("initial_true_error_count") or 0) > 0]
    no_error = [
        row
        for row in episodes
        if row.get("physical_assessable")
        and int(row.get("initial_true_error_count") or 0) == 0
    ]
    stable_exact_faulted = sum(
        row.get("episode_outcome_class") == EXACT_PHYSICAL_RECOVERY for row in faulted
    )
    exact_faulted = sum(
        row.get("final_active_state_class") == EXACT_PHYSICAL_RECOVERY for row in faulted
    )
    committed = sum(
        int(row.get("committed_correction_target_event_count") or 0)
        for row in assessable
    )
    unique_committed = sum(
        int(row.get("unique_committed_correction_target_count") or 0)
        for row in assessable
    )
    true_committed = sum(int(row.get("true_committed_correction_count") or 0) for row in assessable)
    true_errors = sum(int(row.get("initial_true_error_count") or 0) for row in assessable)
    corrected = sum(int(row.get("true_errors_corrected") or 0) for row in assessable)
    valid_steps = sum(int(row.get("valid_action_count") or 0) for row in episodes)
    physical_state_unchanged_steps = sum(
        int(row.get("physical_state_unchanged_valid_action_count") or 0)
        for row in episodes
    )
    controller_valid_steps = sum(
        int(row.get("controller_valid_action_count") or 0) for row in episodes
    )
    no_progress = sum(
        int(row.get("no_progress_valid_action_count") or 0) for row in episodes
    )
    action_steps = sum(int(row.get("steps") or 0) for row in episodes)
    repeated = sum(
        int(row.get("repeated_action_signature_count") or 0) for row in episodes
    )
    hash_observations = sum(
        int(row.get("state_hash_observation_count") or 0) for row in episodes
    )
    revisits = sum(int(row.get("state_hash_revisit_count") or 0) for row in episodes)

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    detailed_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        for row in episode.get("trace") or []:
            expected = row.get("expected_candidate_action")
            predicted = row.get("predicted_candidate_action")
            disposition = row.get("candidate_disposition")
            if expected is not None and predicted is not None:
                confusion[str(expected)][str(predicted)] += 1
                detailed_confusion[str(disposition)][str(predicted)] += 1

    excess = [
        int(row["excess_steps_relative_to_expert"])
        for row in episodes
        if row.get("excess_steps_relative_to_expert") is not None
    ]
    step_differences = [
        int(row["step_difference_relative_to_expert"])
        for row in episodes
        if row.get("step_difference_relative_to_expert") is not None
    ]
    return {
        "episodes": len(episodes),
        "six_way_outcome_counts": dict(outcome_counts),
        "final_active_state_counts": dict(snapshot_counts),
        "physical_assessable_episodes": len(assessable),
        "physical_assessment_coverage": _ratio(len(assessable), len(episodes)),
        "correction_precision": _ratio(true_committed, committed),
        "correction_precision_numerator_true_committed_targets": true_committed,
        "correction_precision_denominator_all_committed_target_events": committed,
        "unique_target_correction_precision": _ratio(
            true_committed, unique_committed
        ),
        "unique_target_correction_precision_numerator_true_committed_targets": (
            true_committed
        ),
        "unique_target_correction_precision_denominator_unique_committed_targets": (
            unique_committed
        ),
        "correction_recall": _ratio(corrected, true_errors),
        "correction_recall_numerator_true_errors_corrected": corrected,
        "correction_recall_denominator_true_errors_assessable": true_errors,
        "exact_episode_recovery_rate": _ratio(exact_faulted, len(faulted)),
        "exact_episode_recovery_count": exact_faulted,
        "faulted_episode_count": len(faulted),
        "no_error_episode_count": len(no_error),
        "no_error_clean_preservation_count": sum(
            row.get("final_active_state_class") == EXACT_PHYSICAL_RECOVERY
            for row in no_error
        ),
        "no_error_clean_preservation_rate": _ratio(
            sum(
                row.get("final_active_state_class") == EXACT_PHYSICAL_RECOVERY
                for row in no_error
            ),
            len(no_error),
        ),
        "no_error_false_intervention_count": sum(
            row.get("final_active_state_class") == FALSE_INTERVENTION
            for row in no_error
        ),
        "no_error_false_intervention_rate": _ratio(
            sum(
                row.get("final_active_state_class") == FALSE_INTERVENTION
                for row in no_error
            ),
            len(no_error),
        ),
        "final_active_snapshot_exact_recovery_rate": _ratio(
            exact_faulted, len(faulted)
        ),
        "final_active_snapshot_exact_recovery_count": exact_faulted,
        "stable_terminal_exact_episode_recovery_rate": _ratio(
            stable_exact_faulted, len(faulted)
        ),
        "stable_terminal_exact_episode_recovery_count": stable_exact_faulted,
        "no_progress_valid_action_rate": _ratio(
            no_progress, controller_valid_steps
        ),
        "no_progress_valid_action_count": no_progress,
        "valid_action_count": valid_steps,
        "physical_state_unchanged_valid_action_rate": _ratio(
            physical_state_unchanged_steps, valid_steps
        ),
        "physical_state_unchanged_valid_action_count": (
            physical_state_unchanged_steps
        ),
        "physical_state_unchanged_rate_note": (
            "includes routine read-only actions; use no_progress_valid_action_rate "
            "for controller actions"
        ),
        "controller_valid_action_count": controller_valid_steps,
        "routine_read_only_valid_action_count": valid_steps - controller_valid_steps,
        "no_progress_rate_denominator": "controller_valid_actions_excluding_read_only",
        "repeated_action_signature_rate": _ratio(repeated, action_steps),
        "repeated_action_signature_count": repeated,
        "action_count": action_steps,
        "state_hash_revisit_rate": _ratio(revisits, hash_observations),
        "state_hash_revisit_count": revisits,
        "state_hash_observation_count": hash_observations,
        "state_hash_revisit_definition": "nonconsecutive_runtime_hash_revisit",
        "maximum_no_progress_streak": max(
            (int(row.get("max_no_progress_streak") or 0) for row in episodes),
            default=0,
        ),
        "mean_excess_steps_relative_to_expert": (
            sum(excess) / len(excess) if excess else None
        ),
        "mean_signed_step_difference_relative_to_expert": (
            sum(step_differences) / len(step_differences)
            if step_differences
            else None
        ),
        "excess_steps_relative_to_expert_episode_count": len(excess),
        "candidate_disposition_confusion_matrix": {
            expected: dict(counts) for expected, counts in sorted(confusion.items())
        },
        "candidate_disposition_detailed_confusion_matrix": {
            expected: dict(counts)
            for expected, counts in sorted(detailed_confusion.items())
        },
        "replay_mismatch_episodes": sum(
            not bool(row.get("replay_matches_record")) for row in episodes
        ),
        "oracle_truth_ledger_disagreement_episodes": sum(
            (
                row.get("oracle_remaining_truth_ledger_diagnostic") or {}
            ).get("agrees_with_direct_final_state")
            is False
            for row in episodes
        ),
        "scenario_alignment_counts": dict(
            Counter(str(row.get("scenario_alignment_basis")) for row in episodes)
        ),
        "unclassified_episodes": outcome_counts.get("unclassified", 0),
    }


def audit_evaluation_report(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    expert_baseline: Sequence[Mapping[str, Any]] | None = None,
    env_factory: Callable[[], Any] = production_environment_factory,
) -> dict[str, Any]:
    recorded = report.get("per_episode")
    if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
        raise ValueError("evaluation report has no per_episode sequence")
    if len(recorded) != len(scenarios):
        raise ValueError(
            f"evaluation/scenario length mismatch: {len(recorded)} != {len(scenarios)}"
        )
    expert_steps_by_index = {
        int(row["scenario_index"]): row.get("steps")
        for row in expert_baseline or []
        if isinstance(row, Mapping) and row.get("scenario_index") is not None
    }
    env = env_factory()
    episodes = [
        replay_episode(
            env,
            scenario,
            recorded[index],
            scenario_index=index,
            expert_steps=expert_steps_by_index.get(index),
        )
        for index, scenario in enumerate(scenarios)
    ]
    by_family = {
        family: summarize_episodes(
            [row for row in episodes if row.get("scenario_family") == family]
        )
        for family in sorted({str(row.get("scenario_family")) for row in episodes})
    }
    identity_bound = bool(episodes) and all(
        str(row.get("scenario_alignment_basis", "")).startswith("validated_")
        for row in episodes
    )
    if report.get("scenarios_sha256") and identity_bound:
        source_binding = "sha256_and_per_episode_identity"
    elif report.get("scenarios_sha256"):
        source_binding = "suite_sha256_with_incomplete_per_episode_identity"
    else:
        source_binding = "ordered_replay_without_source_suite_hash"
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "label": report.get("label"),
        "source_adapter": report.get("adapter"),
        "source_release_evidence": report.get("release_evidence"),
        "source_scenarios_sha256": report.get("scenarios_sha256"),
        "source_scenario_binding": source_binding,
        "summary": summarize_episodes(episodes),
        "by_scenario_family": by_family,
        "per_episode": episodes,
        "release_evidence": False,
    }


def _parse_evaluation_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, _, raw_path = value.partition("=")
    if not label.strip() or not raw_path.strip():
        raise ValueError("--evaluation must be LABEL=PATH or PATH")
    return label.strip(), Path(raw_path.strip())


def compact_audit_report(
    report: Mapping[str, Any],
    *,
    full_report_path: str | Path,
) -> dict[str, Any]:
    """Drop per-step replay payloads while preserving results and provenance."""

    evaluations = report.get("evaluations")
    evaluations = evaluations if isinstance(evaluations, Mapping) else {}
    return {
        "audit_schema_version": report.get("audit_schema_version"),
        "scenarios": report.get("scenarios"),
        "scenarios_sha256": report.get("scenarios_sha256"),
        "full_report": str(full_report_path),
        "full_report_sha256": file_sha256(full_report_path),
        "expert_baseline_problem_episodes": sum(
            bool(row.get("problem"))
            for row in report.get("expert_baseline") or []
            if isinstance(row, Mapping)
        ),
        "evaluations": {
            str(label): {
                "source_evaluation": value.get("source_evaluation"),
                "source_scenario_binding": value.get("source_scenario_binding"),
                "summary": value.get("summary"),
                "by_scenario_family": value.get("by_scenario_family"),
                "release_evidence": False,
            }
            for label, value in evaluations.items()
            if isinstance(value, Mapping)
        },
        "release_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Compact research evaluation report; repeat for multiple checkpoints.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--summary-output",
        help="Optional compact report without the per-episode replay traces.",
    )
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument(
        "--skip-expert-baseline",
        action="store_true",
        help="Omit excess-step metrics when an expert replay is too expensive.",
    )
    args = parser.parse_args(argv)

    scenarios_sha256 = file_sha256(args.scenarios)
    scenarios = load_scenarios(args.scenarios)
    baseline = (
        None
        if args.skip_expert_baseline
        else expert_step_baseline(scenarios, max_steps=args.max_steps)
    )
    reports: dict[str, Any] = {}
    for raw in args.evaluation:
        label, path = _parse_evaluation_argument(raw)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} is not a JSON object")
        source_scenarios_sha256 = payload.get("scenarios_sha256")
        if (
            source_scenarios_sha256 is not None
            and str(source_scenarios_sha256) != scenarios_sha256
        ):
            raise ValueError(
                f"{path} was evaluated against a different scenario-suite hash"
            )
        if label in reports:
            raise ValueError(f"duplicate evaluation label: {label}")
        reports[label] = audit_evaluation_report(
            scenarios=scenarios,
            report=payload,
            expert_baseline=baseline,
        )
        reports[label]["source_evaluation"] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    output = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "scenarios": str(args.scenarios),
        "scenarios_sha256": scenarios_sha256,
        "expert_baseline": baseline,
        "evaluations": reports,
        "release_evidence": False,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_output:
        summary_target = Path(args.summary_output)
        summary_target.parent.mkdir(parents=True, exist_ok=True)
        compact = compact_audit_report(output, full_report_path=target)
        summary_target.write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: value["summary"] for key, value in reports.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
