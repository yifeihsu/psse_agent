"""Privileged, post-target correctness audit for DAgger-1 supervision.

The production teacher chooses an action from :class:`PolicyObservation` before
this module is called.  This audit may then inspect the private oracle ledger
and physical transactional state, but it returns only coarse booleans and
controlled reason codes.  It never proposes, rewrites, or ranks an action.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CONTEXT_TOOLS,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    DIAGNOSTIC_TOOLS,
    FINALIZE_DIAGNOSIS,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_WLS,
    VERIFY_CANDIDATE,
    safe_normalize_action,
)
from psse_env.dagger.release_audit import (
    ACCEPTED_TARGET_NONREGRESSION_CHECK,
    HEALTHY_CASE_CHECK,
    HEALTHY_MEASUREMENTS_CHECK,
    audit_episode_against_truth,
)
from psse_env.state_store import OracleState, PolicyObservation
from psse_env.private_target_matching import (
    canonical_branch_target,
    measurement_action_targets,
    measurement_fault_target,
)


_V1_AUDIT_CONTRACT = "dagger1_offline_teacher_target_truth_audit_v1"
_V2_AUDIT_CONTRACT = "dagger1_offline_teacher_target_truth_audit_v2"
OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT = (
    "dagger1_offline_teacher_target_truth_audit_v3"
)
# v2 replaced the operator-escalation check set.  v3 adds the verified terminal
# measurement closure as a narrow alternative to remaining-target membership,
# which widens the commit reason vocabulary; broadening v2 in place would have
# let a v2 reader silently accept codes it cannot interpret.  Records written by
# v1 and v2 remain readable so already-collected artifacts keep validating under
# their own contract; only newly produced records use the current vocabulary.
LEGACY_OFFLINE_TEACHER_TARGET_AUDIT_CONTRACTS = frozenset(
    {_V1_AUDIT_CONTRACT, _V2_AUDIT_CONTRACT}
)
_ESCALATION_REQUESTS = frozenset(
    {
        HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
        RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    }
)
_CORRECTION_TO_FAMILY = {
    CORRECT_MEASUREMENTS: "measurement",
    CORRECT_PARAMETERS: "parameter",
    CORRECT_TOPOLOGY: "topology",
}
_OBSERVABLE_ONLY_ACTIONS = frozenset(
    {
        RUN_WLS,
        VERIFY_CANDIDATE,
        RUN_ALTERNATIVE_TEST,
        ASK_FOR_MORE_EVIDENCE,
        *CONTEXT_TOOLS,
        *DIAGNOSTIC_TOOLS,
    }
)
_AUDIT_FIELDS = frozenset(
    {"contract", "passed", "action_class", "checks", "reason_codes"}
)
# Release-audit checks whose verdict is attributable to a single commit once
# the audit is rebased onto the state that commit mutates.
_COMMIT_ATTRIBUTABLE_CHECKS = frozenset(
    {
        ACCEPTED_TARGET_NONREGRESSION_CHECK,
        HEALTHY_CASE_CHECK,
        HEALTHY_MEASUREMENTS_CHECK,
    }
)
# Release-audit problems that mean "could not verify" rather than "unsafe".
# These must keep failing closed even where inherited harm is excused.
_EVIDENCE_INTEGRITY_MARKERS = (
    "_malformed",
    "_unloadable",
    "_unverifiable",
    "_invalid",
    "_missing",
    "_out_of_range",
    "_prohibited",
    "_unsupported",
    "evidence_missing",
)
_ACTION_CLASS_CHECKS = {
    "missing_target": frozenset({"teacher_target_present"}),
    "invalid_target": frozenset({"teacher_target_well_formed"}),
    "measurement_correction": frozenset(
        {
            "complete_private_ledger",
            "target_is_remaining_family_fault",
            "observable_evidence_gate_passed",
        }
    ),
    "parameter_correction": frozenset(
        {
            "complete_private_ledger",
            "target_is_remaining_family_fault",
            "observable_evidence_gate_passed",
        }
    ),
    "topology_correction": frozenset(
        {
            "complete_private_ledger",
            "target_is_remaining_family_fault",
            "requested_topology_status_matches",
            "observable_evidence_gate_passed",
        }
    ),
    "commit": frozenset(
        {
            "candidate_exists",
            "candidate_verified",
            "candidate_source_truth_evidence_complete",
            "candidate_truth_safe_to_commit",
            "observable_evidence_gate_passed",
        }
    ),
    "rollback": frozenset(
        {
            "candidate_exists",
            "candidate_verified",
            "candidate_source_truth_evidence_complete",
            "candidate_not_truth_safe_to_commit",
            "observable_evidence_gate_passed",
        }
    ),
    "finalize": frozenset(
        {
            "observable_evidence_gate_passed",
            "resolved_claim_matches_private_ledger",
        }
    ),
    "operator_escalation": frozenset(
        {
            "handoff_state_audit_evaluable",
            "observable_evidence_gate_passed",
            "terminal_claim_is_handoff_not_resolution",
        }
    ),
    "read_only": frozenset({"observable_evidence_gate_passed"}),
    "unknown_target": frozenset(
        {"teacher_target_is_known_nonmutating_action"}
    ),
}
_V1_ACTION_CLASS_CHECKS = {
    **_ACTION_CLASS_CHECKS,
    "operator_escalation": frozenset(
        {
            "accepted_state_nonregressive_and_healthy",
            "observable_evidence_gate_passed",
            "terminal_claim_is_handoff_not_resolution",
        }
    ),
}
_ACTION_CLASS_CHECKS_BY_CONTRACT = {
    # v3 keeps v2's check vocabulary: the closure contract changes which
    # commits pass, not which checks are reported.  Map each legacy contract
    # explicitly -- a blanket comprehension over the legacy set would bind v2
    # to v1's operator-escalation checks.
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT: _ACTION_CLASS_CHECKS,
    _V2_AUDIT_CONTRACT: _ACTION_CLASS_CHECKS,
    _V1_AUDIT_CONTRACT: _V1_ACTION_CLASS_CHECKS,
}
_ACTION_CLASSES = frozenset(_ACTION_CLASS_CHECKS)
_FAILURE_ONLY_ACTION_CLASSES = frozenset(
    {"missing_target", "invalid_target", "unknown_target"}
)
_CHECK_NAMES = frozenset(
    check
    for checks_by_class in _ACTION_CLASS_CHECKS_BY_CONTRACT.values()
    for checks in checks_by_class.values()
    for check in checks
)
_COMMON_REASON_CODES = frozenset(
    {
        "teacher_target_missing",
        "teacher_target_invalid",
        "private_ledger_incomplete",
        "target_outside_remaining_family_faults",
        "topology_status_disagrees_with_private_ledger",
        "candidate_source_correction_missing",
        "candidate_missing",
        "candidate_verification_missing",
        "observable_evidence_gate_failed",
        "rollback_would_discard_truth_safe_candidate",
        "resolved_claim_failed_private_release_audit",
        "teacher_target_action_class_unknown",
    }
)
_V2_REASON_CODES = _COMMON_REASON_CODES | {
    "candidate_disposition_not_accepted",
    "candidate_source_target_outside_remaining_truth",
    "candidate_commit_introduces_new_physical_harm",
    "handoff_state_audit_unavailable",
}
_V1_REASON_CODES = _COMMON_REASON_CODES | {
    "candidate_failed_private_commit_safety",
    "handoff_failed_private_safety_audit",
}
#: A rejected closure attempt reports the clause that failed, so a quarantine
#: can be triaged from the bundle without re-running collection.
_CLOSURE_REASON_CODES = frozenset(
    {
        "closure_attestation_malformed",
        "closure_ledger_unreadable",
        "closure_action_does_not_match_attestation",
        "closure_action_not_in_supported_inventory",
        "closure_context_not_state_bound",
        "closure_accepted_set_empty",
        "closure_new_target_count_not_one",
        "closure_does_not_reuse_entire_accepted_set",
        "closure_accepted_target_not_original_truth",
        "closure_new_target_outside_remaining_truth",
        "closure_accepted_target_still_in_remaining_truth",
        "closure_screening_incomplete",
    }
)
_V3_REASON_CODES = _V2_REASON_CODES | _CLOSURE_REASON_CODES
_REASON_CODES_BY_CONTRACT = {
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT: _V3_REASON_CODES,
    _V2_AUDIT_CONTRACT: _V2_REASON_CODES,
    _V1_AUDIT_CONTRACT: _V1_REASON_CODES,
}


def _report(
    *,
    action_class: str,
    checks: Mapping[str, Any],
    reason_codes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the intentionally lossy, non-model audit schema."""

    normalized_checks = {
        str(name): bool(value) for name, value in sorted(checks.items())
    }
    reasons = list(dict.fromkeys(str(value) for value in reason_codes if value))
    passed = bool(normalized_checks) and all(normalized_checks.values()) and not reasons
    report = {
        "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
        "passed": passed,
        "action_class": str(action_class),
        "checks": normalized_checks,
        "reason_codes": reasons,
    }
    return validate_offline_teacher_target_audit_metadata(report)


def validate_offline_teacher_target_audit_metadata(
    value: Any, *, require_passed: bool = False
) -> dict[str, Any]:
    """Validate and clone the deliberately low-bandwidth audit record.

    Exact field/check/reason vocabularies prevent callers from using the
    non-model metadata channel to persist raw oracle truth alongside an SFT
    example.
    """

    if not isinstance(value, Mapping):
        raise ValueError("offline teacher-target audit must be a mapping")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys) or keys != _AUDIT_FIELDS:
        raise ValueError("offline teacher-target audit has unexpected fields")
    contract = value.get("contract")
    if contract not in _ACTION_CLASS_CHECKS_BY_CONTRACT:
        raise ValueError("offline teacher-target audit contract mismatch")
    if not isinstance(value.get("passed"), bool):
        raise ValueError("offline teacher-target audit passed must be boolean")
    action_class = value.get("action_class")
    if not isinstance(action_class, str) or action_class not in _ACTION_CLASSES:
        raise ValueError("offline teacher-target audit action class is invalid")
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("offline teacher-target audit checks must be nonempty")
    if any(not isinstance(key, str) or key not in _CHECK_NAMES for key in checks):
        raise ValueError("offline teacher-target audit check name is invalid")
    if any(not isinstance(item, bool) for item in checks.values()):
        raise ValueError("offline teacher-target audit checks must be boolean")
    expected_checks = _ACTION_CLASS_CHECKS_BY_CONTRACT[contract][action_class]
    if set(checks) != expected_checks:
        raise ValueError(
            "offline teacher-target audit check schema is incomplete or unexpected"
        )
    reasons = value.get("reason_codes")
    if not isinstance(reasons, list):
        raise ValueError("offline teacher-target audit reason codes must be a list")
    if any(
        not isinstance(item, str)
        or item not in _REASON_CODES_BY_CONTRACT[contract]
        for item in reasons
    ):
        raise ValueError("offline teacher-target audit reason code is invalid")
    if len(reasons) != len(set(reasons)):
        raise ValueError("offline teacher-target audit reason codes must be unique")
    recomputed_passed = bool(checks) and all(checks.values()) and not reasons
    if value["passed"] is not recomputed_passed:
        raise ValueError("offline teacher-target audit pass flag is inconsistent")
    if value["passed"] is True and action_class in _FAILURE_ONLY_ACTION_CLASSES:
        raise ValueError(
            "offline teacher-target audit failure-only action class cannot pass"
        )
    if require_passed and value["passed"] is not True:
        raise ValueError("offline teacher-target audit did not pass")
    return copy.deepcopy(dict(value))


def _oracle_truth(oracle_state: Any) -> dict[str, Any]:
    if isinstance(oracle_state, OracleState):
        return oracle_state.truth_dict()
    if isinstance(oracle_state, Mapping):
        hidden = oracle_state.get("hidden_truth")
        truth = copy.deepcopy(dict(hidden)) if isinstance(hidden, Mapping) else {}
        for key in (
            "truth_complete",
            "clean_case",
            "clean_measurements",
            "true_measurement_errors",
            "true_parameter_errors",
            "true_topology_errors",
            "remaining_true_faults",
            "remaining_true_fault_count",
        ):
            if key in oracle_state:
                truth.setdefault(key, copy.deepcopy(oracle_state[key]))
        return truth
    return {}


def _measurement_targets(arguments: Mapping[str, Any]) -> set[int] | None:
    return measurement_action_targets(arguments)


def _truth_measurement_targets(truth: Mapping[str, Any]) -> set[int] | None:
    raw = truth.get("true_measurement_errors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    targets: set[int] = set()
    for fault in raw:
        if not isinstance(fault, Mapping):
            return None
        value = measurement_fault_target(fault)
        if value is None:
            return None
        targets.add(value)
    return targets


def _branch_row0(value: Mapping[str, Any]) -> int | None:
    target = canonical_branch_target(value)
    return int(target[1]) if target is not None and target[0] == "branch_row0" else None


def _named_branch_target(value: Mapping[str, Any]) -> tuple[str, str] | None:
    target = canonical_branch_target(value)
    if target is None or target[0] != "branch_id":
        return None
    return "branch_id", str(target[1])


def _branch_target_matches(
    arguments: Mapping[str, Any], faults: Any
) -> tuple[bool, Mapping[str, Any] | None]:
    if not isinstance(faults, Sequence) or isinstance(faults, (str, bytes)):
        return False, None
    action_target = canonical_branch_target(arguments)
    if action_target is None:
        return False, None
    for fault in faults:
        if not isinstance(fault, Mapping):
            continue
        if canonical_branch_target(fault) == action_target:
            return True, fault
    return False, None


def _status_matches(arguments: Mapping[str, Any], fault: Mapping[str, Any]) -> bool:
    requested = next(
        (
            arguments[key]
            for key in ("desired_status", "status", "expected_status")
            if arguments.get(key) is not None
        ),
        None,
    )
    expected = next(
        (
            fault[key]
            for key in ("expected_status", "clean", "true_value")
            if fault.get(key) is not None
        ),
        None,
    )
    if requested is None or expected is None:
        return False

    def normalized_status(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "closed", "close", "on"}:
                return 1
            if text in {"0", "false", "open", "off"}:
                return 0
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return numeric if numeric in {0, 1} else None

    requested_status = normalized_status(requested)
    expected_status = normalized_status(expected)
    return bool(
        requested_status is not None
        and expected_status is not None
        and requested_status == expected_status
    )


def _correction_target_check(
    action: Mapping[str, Any], truth: Mapping[str, Any]
) -> tuple[dict[str, bool], list[str]]:
    tool = str(action.get("tool") or "")
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    checks: dict[str, bool] = {
        "complete_private_ledger": truth.get("truth_complete") is True,
        "target_is_remaining_family_fault": False,
    }
    reasons: list[str] = []
    if not checks["complete_private_ledger"]:
        reasons.append("private_ledger_incomplete")
        return checks, reasons
    if tool == CORRECT_MEASUREMENTS:
        proposed = _measurement_targets(arguments)
        remaining = _truth_measurement_targets(truth)
        checks["target_is_remaining_family_fault"] = bool(
            proposed and remaining is not None and proposed.issubset(remaining)
        )
    elif tool == CORRECT_PARAMETERS:
        matched, _ = _branch_target_matches(
            arguments, truth.get("true_parameter_errors")
        )
        checks["target_is_remaining_family_fault"] = matched
    elif tool == CORRECT_TOPOLOGY:
        matched, fault = _branch_target_matches(
            arguments, truth.get("true_topology_errors")
        )
        checks["target_is_remaining_family_fault"] = matched
        checks["requested_topology_status_matches"] = bool(
            matched and fault is not None and _status_matches(arguments, fault)
        )
    if not checks["target_is_remaining_family_fault"]:
        reasons.append("target_outside_remaining_family_faults")
    if checks.get("requested_topology_status_matches") is False:
        reasons.append("topology_status_disagrees_with_private_ledger")
    return checks, reasons


VERIFIED_TERMINAL_MEASUREMENT_CLOSURE_CONTRACT = (
    "dagger1_verified_terminal_measurement_closure_v1"
)


def _original_measurement_truth_targets(
    scenario: Mapping[str, Any],
) -> set[int] | None:
    """Original -- not remaining -- measurement truth targets for a scenario."""

    raw = scenario.get("true_measurement_errors", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    targets: set[int] = set()
    for fault in raw:
        if not isinstance(fault, Mapping):
            return None
        target = measurement_fault_target(fault)
        if target is None:
            return None
        targets.add(target)
    return targets


def _accepted_measurement_targets(
    observation: Mapping[str, Any],
) -> set[int] | None:
    """Measurement targets already accepted on the observable ledger."""

    accepted = observation.get("accepted_corrections")
    if not isinstance(accepted, Sequence) or isinstance(accepted, (str, bytes)):
        return None
    targets: set[int] = set()
    for entry in accepted:
        if not isinstance(entry, Mapping):
            return None
        source = entry.get("source_action")
        if not isinstance(source, Mapping):
            return None
        if str(source.get("tool") or "") != CORRECT_MEASUREMENTS:
            continue
        arguments = source.get("arguments")
        proposed = _measurement_targets(
            arguments if isinstance(arguments, Mapping) else {}
        )
        if proposed is None:
            return None
        targets |= proposed
    return targets


def _closure_screening_ordered(
    evidence: Mapping[str, Any], *, new_target: int, closure_targets: set[int]
) -> bool:
    """Require the singleton screen for the new target, then the grouped final.

    Both stages must be ``ACCEPT_FINAL`` with passing physical constraints, and
    the singleton must precede the group: a grouped acceptance recorded before
    its new target was screened alone is not a closure.
    """

    attempts = evidence.get("attempts")
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return False
    singleton_index: int | None = None
    grouped_index: int | None = None
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            return False
        if str(attempt.get("disposition") or "") != "ACCEPT_FINAL":
            continue
        if attempt.get("physical_constraints_ok") is not True:
            continue
        raw_targets = attempt.get("targets")
        if not isinstance(raw_targets, Sequence) or isinstance(
            raw_targets, (str, bytes)
        ):
            continue
        try:
            attempt_targets = {int(item) for item in raw_targets}
        except (TypeError, ValueError):
            return False
        if (
            attempt_targets == {new_target}
            and attempt.get("target_test_passed") is True
            and singleton_index is None
        ):
            singleton_index = index
        elif attempt_targets == closure_targets and grouped_index is None:
            grouped_index = index
    return (
        singleton_index is not None
        and grouped_index is not None
        and singleton_index < grouped_index
    )


def _verified_terminal_measurement_closure_check(
    action: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
    scenario: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    """Narrow alternative to remaining-target membership for a terminal closure.

    The observable provider deliberately authorises one exceptional grouped
    correction -- every previously accepted target plus exactly one new target,
    two-stage screened and state-bound.  The ordinary rule requires every
    proposed target to sit in the *remaining* ledger, so a correctly retired
    accepted target makes an intended closure look mis-targeted.  All four
    DAgger-1 round-2 quarantines were exactly this action.

    Returns ``({}, [])`` when no closure was attested, leaving the ordinary
    verdict untouched.  Otherwise every clause must hold; this admits that one
    action and nothing else.
    """

    context = observation.get("fresh_context_evidence")
    measurement = (
        context.get("measurement") if isinstance(context, Mapping) else None
    )
    if not isinstance(measurement, Mapping):
        return {}, []
    raw_closure = measurement.get("verified_terminal_measurement_closure_targets")
    evidence = measurement.get("verified_terminal_measurement_closure_evidence")
    if raw_closure is None and evidence is None:
        return {}, []

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def record(name: str, ok: bool, reason: str) -> bool:
        checks[name] = bool(ok)
        if not ok:
            reasons.append(reason)
        return bool(ok)

    if not record(
        "closure_attestation_well_formed",
        isinstance(raw_closure, Sequence)
        and not isinstance(raw_closure, (str, bytes))
        and isinstance(evidence, Mapping),
        "closure_attestation_malformed",
    ):
        return checks, reasons

    try:
        closure_targets = {int(item) for item in raw_closure}
    except (TypeError, ValueError):
        record("closure_attestation_well_formed", False, "closure_attestation_malformed")
        return checks, reasons

    arguments = action.get("arguments")
    proposed = _measurement_targets(
        arguments if isinstance(arguments, Mapping) else {}
    )
    accepted = _accepted_measurement_targets(observation)
    remaining = _truth_measurement_targets(truth)
    original = _original_measurement_truth_targets(scenario)
    if proposed is None or accepted is None or remaining is None or original is None:
        record("closure_ledger_readable", False, "closure_ledger_unreadable")
        return checks, reasons
    checks["closure_ledger_readable"] = True

    # Clause 1: the audited action is exactly the attested closure action, and
    # that action is present in the same-state supported-correction inventory.
    record(
        "closure_action_matches_attestation",
        proposed == closure_targets,
        "closure_action_does_not_match_attestation",
    )
    state_id = str(measurement.get("state_id") or "")
    supported = measurement.get("supported_corrections")
    supported_match = False
    if isinstance(supported, Sequence) and not isinstance(supported, (str, bytes)):
        for entry in supported:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("tool") or "") != CORRECT_MEASUREMENTS:
                continue
            entry_arguments = entry.get("arguments")
            entry_arguments = (
                entry_arguments if isinstance(entry_arguments, Mapping) else {}
            )
            entry_targets = _measurement_targets(entry_arguments)
            if (
                entry_targets == closure_targets
                and str(entry_arguments.get("state_id") or "") == state_id
            ):
                supported_match = True
                break
    record(
        "closure_action_in_supported_inventory",
        supported_match,
        "closure_action_not_in_supported_inventory",
    )

    # Clause 2: the context evidence is bound to the current active state.
    record(
        "closure_context_state_bound",
        bool(state_id)
        and state_id == str(observation.get("active_state_id") or "")
        and bool(str(measurement.get("state_hash") or "")),
        "closure_context_not_state_bound",
    )

    # Clauses 3, 4: a non-empty accepted set, entirely reused, plus exactly one
    # new target.
    record("closure_accepted_set_nonempty", bool(accepted), "closure_accepted_set_empty")
    new_targets = proposed - accepted
    record(
        "closure_exactly_one_new_target",
        len(new_targets) == 1,
        "closure_new_target_count_not_one",
    )
    record(
        "closure_reuses_entire_accepted_set",
        bool(accepted) and accepted.issubset(proposed),
        "closure_does_not_reuse_entire_accepted_set",
    )

    # Clause 5: every reused target was an original truth target.  Read from the
    # same private scenario truth the release ledger already uses, so this adds
    # no hidden-data channel.
    record(
        "closure_accepted_targets_are_original_truth",
        bool(accepted) and accepted.issubset(original),
        "closure_accepted_target_not_original_truth",
    )

    # Clauses 6, 7: the new target is still owed, the reused ones are retired.
    record(
        "closure_new_target_in_remaining_truth",
        len(new_targets) == 1 and new_targets.issubset(remaining),
        "closure_new_target_outside_remaining_truth",
    )
    record(
        "closure_accepted_targets_already_retired",
        not (accepted & remaining),
        "closure_accepted_target_still_in_remaining_truth",
    )

    # Clause 8: two-stage screening, singleton before group, both ACCEPT_FINAL.
    # Clause 9 (non-regression, healthy-state) is enforced by the existing
    # physical-safety audit and is deliberately not duplicated here.
    record(
        "closure_screening_two_stage_accept_final",
        len(new_targets) == 1
        and isinstance(evidence, Mapping)
        and _closure_screening_ordered(
            evidence,
            new_target=next(iter(new_targets)),
            closure_targets=closure_targets,
        ),
        "closure_screening_incomplete",
    )
    return checks, reasons


def _candidate(env: Any, action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    candidate_id = arguments.get("candidate_state_id") or getattr(
        env, "current_candidate_id", None
    )
    store = getattr(env, "store", None)
    if candidate_id is None or store is None:
        return None
    try:
        if not store.exists(str(candidate_id)):
            return None
        candidate = store.get_state(str(candidate_id))
    except Exception:
        return None
    return candidate if isinstance(candidate, Mapping) else None


def _active_physical_state(env: Any) -> Mapping[str, Any] | None:
    store = getattr(env, "store", None)
    active_id = getattr(store, "active_state_id", None)
    if store is None or active_id is None:
        return None
    try:
        state = store.get_state(str(active_id))
    except Exception:
        return None
    return state if isinstance(state, Mapping) else None


def _release_safety_check(
    *,
    scenario: Mapping[str, Any],
    policy_observation: Mapping[str, Any],
    active_physical_state: Mapping[str, Any] | None,
    terminal_outcome: str,
    remaining_truth: Mapping[str, Any] | None,
    case_loader: Callable[[Any], Any] | None,
) -> bool:
    try:
        result = audit_episode_against_truth(
            scenario,
            policy_observation,
            terminal=True,
            terminal_outcome=terminal_outcome,
            active_physical_state=active_physical_state,
            remaining_truth=remaining_truth,
            case_loader=case_loader,
        )
    except Exception:
        return False
    return result.get("quarantined") is False


def _release_audit_problems(
    *,
    scenario: Mapping[str, Any],
    policy_observation: Mapping[str, Any],
    active_physical_state: Mapping[str, Any] | None,
    terminal_outcome: str,
    remaining_truth: Mapping[str, Any] | None,
    case_loader: Callable[[Any], Any] | None,
    check_names: frozenset[str] | None = None,
) -> frozenset[str] | None:
    """Return the release-audit problem set, or ``None`` if it could not run.

    ``_release_safety_check`` collapses the audit to one boolean, which cannot
    distinguish harm a teacher target introduces from harm it inherited from
    the learner's already-committed corrections.  Callers that must attribute
    a problem to the target need the problem set itself.  ``check_names``
    narrows the result to named checks so attribution selects a check by
    identity rather than by matching problem text.
    """

    try:
        result = audit_episode_against_truth(
            scenario,
            policy_observation,
            terminal=True,
            terminal_outcome=terminal_outcome,
            active_physical_state=active_physical_state,
            remaining_truth=remaining_truth,
            case_loader=case_loader,
        )
    except Exception:
        return None
    if check_names is None:
        problems = result.get("problems")
        if not isinstance(problems, Sequence) or isinstance(problems, (str, bytes)):
            return None
        return frozenset(str(item) for item in problems)
    checks = result.get("checks")
    if not isinstance(checks, Mapping):
        return None
    collected: set[str] = set()
    for name in check_names:
        entry = checks.get(name)
        if not isinstance(entry, Mapping):
            return None
        problems = entry.get("problems")
        if not isinstance(problems, Sequence) or isinstance(problems, (str, bytes)):
            return None
        collected.update(str(item) for item in problems)
    return frozenset(collected)


def _evidence_integrity_failure(problems: frozenset[str] | None) -> bool:
    """Fail closed when the release audit could not actually verify the state.

    The release-audit vocabulary marks every unverifiable outcome with one of
    these suffixes, so an evidence gap can never be mistaken for a physically
    safe state.
    """

    if problems is None:
        return True
    return any(
        marker in problem
        for problem in problems
        for marker in _EVIDENCE_INTEGRITY_MARKERS
    )


def _declared_truth_target_actions(
    scenario: Mapping[str, Any], *, state_id: str
) -> list[dict[str, Any]] | None:
    """Build an internal audit ledger covering every physical truth target.

    This is private counterfactual evidence only; it is never returned in the
    low-bandwidth audit report.  Covering all targets lets the shared release
    audit detect collateral regression of a different true fault, which the
    healthy-state checks intentionally exclude.
    """

    actions: list[dict[str, Any]] = []
    raw_measurements = scenario.get("true_measurement_errors", [])
    if not isinstance(raw_measurements, Sequence) or isinstance(
        raw_measurements, (str, bytes)
    ):
        return None
    measurement_targets: set[int] = set()
    for fault in raw_measurements:
        if not isinstance(fault, Mapping):
            return None
        target = measurement_fault_target(fault)
        if target is None:
            return None
        measurement_targets.add(target)
    if measurement_targets:
        actions.append(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": state_id,
                    "suspect_group": sorted(measurement_targets),
                },
            }
        )

    for truth_key, tool in (
        ("true_parameter_errors", CORRECT_PARAMETERS),
        ("true_topology_errors", CORRECT_TOPOLOGY),
    ):
        raw_faults = scenario.get(truth_key, [])
        if not isinstance(raw_faults, Sequence) or isinstance(
            raw_faults, (str, bytes)
        ):
            return None
        targets: set[tuple[str, Any]] = set()
        for fault in raw_faults:
            if not isinstance(fault, Mapping):
                return None
            target = canonical_branch_target(fault)
            if target is None:
                return None
            targets.add(target)
        for kind, value in sorted(
            targets, key=lambda item: (str(item[0]), str(item[1]))
        ):
            arguments: dict[str, Any] = {"state_id": state_id}
            arguments[kind] = value
            actions.append({"tool": tool, "arguments": arguments})
    return actions


def _default_case_loader() -> Callable[[Any], Any] | None:
    # Imported only after the model target is fixed.  Keeping this dependency
    # lazy avoids coupling policy construction to the release I/O stack.
    try:
        from psse_env.dagger.release_factories import deterministic_case_loader
    except Exception:
        return None
    return deterministic_case_loader


def offline_teacher_target_audit(
    *,
    preferred_action: Mapping[str, Any] | str | None,
    oracle_state: OracleState | Mapping[str, Any] | Any,
    policy_observation: PolicyObservation | Mapping[str, Any],
    scenario: Mapping[str, Any],
    env: Any,
    observable_evidence_passed: bool,
    case_loader: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Audit a fixed DAgger-1 teacher target against private physical truth.

    The function is deliberately incapable of returning an alternative action.
    Changing hidden truth can therefore change only this quarantine report, not
    the teacher target or anything passed to the learner.
    """

    if preferred_action is None:
        return _report(
            action_class="missing_target",
            checks={"teacher_target_present": False},
            reason_codes=("teacher_target_missing",),
        )
    action = safe_normalize_action(copy.deepcopy(preferred_action))
    observation = (
        policy_observation.as_dict()
        if isinstance(policy_observation, PolicyObservation)
        else copy.deepcopy(dict(policy_observation))
    )
    truth = _oracle_truth(oracle_state)
    tool = action["tool"]
    if tool == "__invalid_action__":
        return _report(
            action_class="invalid_target",
            checks={"teacher_target_well_formed": False},
            reason_codes=("teacher_target_invalid",),
        )

    if tool in _CORRECTION_TO_FAMILY:
        checks, reasons = _correction_target_check(action, truth)
        checks["observable_evidence_gate_passed"] = observable_evidence_passed
        if not observable_evidence_passed:
            reasons.append("observable_evidence_gate_failed")
        return _report(
            action_class=f"{_CORRECTION_TO_FAMILY[tool]}_correction",
            checks=checks,
            reason_codes=reasons,
        )

    if tool in {COMMIT_STATE, ROLLBACK_STATE}:
        candidate = _candidate(env, action)
        source_action = (
            safe_normalize_action(candidate.get("source_action") or {})
            if isinstance(candidate, Mapping)
            else None
        )
        source_truth_checks: dict[str, bool] = {}
        source_reasons: list[str] = []
        source_is_correction = bool(
            source_action is not None
            and source_action["tool"] in _CORRECTION_TO_FAMILY
        )
        if source_is_correction:
            source_truth_checks, source_reasons = _correction_target_check(
                source_action, truth
            )
        else:
            source_reasons.append("candidate_source_correction_missing")
        source_truth_evidence_complete = bool(
            source_is_correction
            and source_truth_checks.get("complete_private_ledger") is True
        )
        verified = bool(
            isinstance(candidate, Mapping)
            and isinstance(candidate.get("verification_output"), Mapping)
            and candidate.get("candidate_disposition") is not None
        )
        accepted_disposition = bool(
            isinstance(candidate, Mapping)
            and str(candidate.get("candidate_disposition") or "")
            in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}
        )
        hypothetical = copy.deepcopy(observation)
        # This counterfactual judges one candidate transition.  Prior accepted
        # entries may already contain a false target; including them makes the
        # shared non-regression audit short-circuit before it evaluates the
        # current target.  Physical inherited effects are neutralized below by
        # rebasing the scenario onto the parent state, while the current source
        # action remains the sole accepted entry in this attribution view.
        accepted: list[dict[str, Any]] = []
        if source_action is not None:
            accepted.append({"source_action": source_action})
        hypothetical["accepted_corrections"] = accepted
        loader = case_loader if case_loader is not None else _default_case_loader()
        # Isolate this commit's own physical effect by rebasing the audit's
        # "initial" state onto the state the commit would mutate.  Corrections
        # the learner already committed cannot be retired by any action
        # available here, so charging their damage to this target would
        # quarantine a correct commit for damage it did not do -- while
        # differencing whole-episode problem *sets* would hide new damage of a
        # kind the inherited state already exhibits.
        parent_state = _active_physical_state(env)
        commit_scenario = dict(scenario)
        if isinstance(parent_state, Mapping):
            for key in ("measurements", "case"):
                if parent_state.get(key) is not None:
                    commit_scenario[key] = copy.deepcopy(parent_state[key])
        current_target_problems = (
            _release_audit_problems(
                scenario=commit_scenario,
                policy_observation=hypothetical,
                active_physical_state=candidate,
                terminal_outcome="operator_escalation",
                remaining_truth=None,
                case_loader=loader,
                check_names=_COMMIT_ATTRIBUTABLE_CHECKS,
            )
            if isinstance(candidate, Mapping) and isinstance(parent_state, Mapping)
            else None
        )
        if current_target_problems is not None:
            # The attribution ledger now contains only this candidate.  A
            # false-target verdict therefore belongs to the independent
            # source-identity check below, while any numeric collateral change
            # still appears in the healthy-state checks.  Removing it here
            # avoids mislabeling an in-tolerance wrong target as new physical
            # harm without allowing an inherited ledger entry to short-circuit
            # current-target non-regression.
            current_target_problems = current_target_problems - {
                "accepted_target_nonregression_false_target"
            }
        declared_actions = (
            _declared_truth_target_actions(
                commit_scenario,
                state_id=str(parent_state.get("state_id") or ""),
            )
            if isinstance(parent_state, Mapping)
            else None
        )
        all_target_problems: frozenset[str] | None = None
        if (
            isinstance(candidate, Mapping)
            and declared_actions is not None
            and isinstance(parent_state, Mapping)
        ):
            all_target_hypothetical = copy.deepcopy(observation)
            all_target_hypothetical["accepted_corrections"] = [
                {"source_action": action} for action in declared_actions
            ]
            all_target_problems = _release_audit_problems(
                scenario=commit_scenario,
                policy_observation=all_target_hypothetical,
                active_physical_state=candidate,
                terminal_outcome="operator_escalation",
                remaining_truth=None,
                case_loader=loader,
                check_names=frozenset({ACCEPTED_TARGET_NONREGRESSION_CHECK}),
            )
        introduced_problems = (
            current_target_problems | all_target_problems
            if current_target_problems is not None
            and all_target_problems is not None
            else None
        )
        physically_safe = bool(
            introduced_problems is not None and not introduced_problems
        )
        target_correct = bool(
            source_truth_checks
            and all(source_truth_checks.values())
            and not source_reasons
        )
        # A verified terminal measurement closure reuses targets it has already
        # retired from the remaining ledger, so ordinary remaining-membership
        # rejects it.  Consult the narrow closure contract only after the
        # ordinary rule has failed, and only for a measurement correction whose
        # private ledger is complete: nothing else can reach this branch.
        closure_checks: dict[str, bool] = {}
        closure_reasons: list[str] = []
        if (
            tool == COMMIT_STATE
            and not target_correct
            and source_is_correction
            and source_truth_evidence_complete
            and source_action is not None
            and source_action["tool"] == CORRECT_MEASUREMENTS
        ):
            closure_checks, closure_reasons = (
                _verified_terminal_measurement_closure_check(
                    source_action,
                    observation=observation,
                    scenario=commit_scenario,
                    truth=truth,
                )
            )
            if (
                closure_checks
                and all(closure_checks.values())
                and not closure_reasons
            ):
                target_correct = True
        truth_safe_to_commit = bool(
            verified and accepted_disposition and target_correct and physically_safe
        )
        checks = {
            "candidate_exists": isinstance(candidate, Mapping),
            "candidate_verified": verified,
            "candidate_source_truth_evidence_complete": (
                source_truth_evidence_complete
            ),
            "observable_evidence_gate_passed": observable_evidence_passed,
            (
                "candidate_truth_safe_to_commit"
                if tool == COMMIT_STATE
                else "candidate_not_truth_safe_to_commit"
            ): (
                truth_safe_to_commit
                if tool == COMMIT_STATE
                else source_truth_evidence_complete and not truth_safe_to_commit
            ),
        }
        reasons = []
        if not isinstance(candidate, Mapping):
            reasons.append("candidate_missing")
        elif not verified:
            reasons.append("candidate_verification_missing")
        if not source_is_correction:
            reasons.append("candidate_source_correction_missing")
        elif not source_truth_evidence_complete:
            reasons.append("private_ledger_incomplete")
        if not observable_evidence_passed:
            reasons.append("observable_evidence_gate_failed")
        if tool == COMMIT_STATE and not truth_safe_to_commit:
            # One umbrella code could not distinguish a mis-targeted candidate
            # from an undispositioned one or from new physical harm, so a
            # quarantine could not be triaged without re-running collection.
            # Each code fires only where it is the non-redundant cause; a
            # missing or unverified candidate is already reported above.
            if verified:
                if not accepted_disposition:
                    reasons.append("candidate_disposition_not_accepted")
                if source_truth_evidence_complete and not target_correct:
                    reasons.append(
                        "candidate_source_target_outside_remaining_truth"
                    )
                    # A rejected closure attempt is reported clause by clause so
                    # the quarantine can be triaged without re-running
                    # collection.
                    reasons.extend(closure_reasons)
                if not physically_safe:
                    reasons.append("candidate_commit_introduces_new_physical_harm")
        if tool == ROLLBACK_STATE and truth_safe_to_commit:
            reasons.append("rollback_would_discard_truth_safe_candidate")
        return _report(
            action_class="commit" if tool == COMMIT_STATE else "rollback",
            checks=checks,
            reason_codes=reasons,
        )

    if tool == FINALIZE_DIAGNOSIS:
        loader = case_loader if case_loader is not None else _default_case_loader()
        safe = _release_safety_check(
            scenario=scenario,
            policy_observation=observation,
            active_physical_state=_active_physical_state(env),
            terminal_outcome="resolved",
            remaining_truth=truth,
            case_loader=loader,
        )
        checks = {
            "observable_evidence_gate_passed": observable_evidence_passed,
            "resolved_claim_matches_private_ledger": safe,
        }
        reasons = []
        if not observable_evidence_passed:
            reasons.append("observable_evidence_gate_failed")
        if not safe:
            reasons.append("resolved_claim_failed_private_release_audit")
        return _report(
            action_class="finalize",
            checks=checks,
            reason_codes=reasons,
        )

    is_escalation = bool(
        tool == ASK_FOR_MORE_EVIDENCE
        and action["arguments"].get("request") in _ESCALATION_REQUESTS
    )
    if is_escalation:
        loader = case_loader if case_loader is not None else _default_case_loader()
        # An escalation mutates nothing: it audits exactly the state it
        # inherited.  Every problem found here therefore came from corrections
        # the learner already committed, and no action available to the teacher
        # retires a committed correction -- ``rollback_state`` discards only an
        # open candidate.  Quarantining the target for that inherited damage
        # would reject the one correct response to an unrecoverable state.  The
        # audit must still be *runnable*: unverifiable evidence fails closed.
        inherited_problems = _release_audit_problems(
            scenario=scenario,
            policy_observation=observation,
            active_physical_state=_active_physical_state(env),
            terminal_outcome="operator_escalation",
            remaining_truth=truth,
            case_loader=loader,
        )
        evaluable = bool(
            truth.get("truth_complete") is True
            and not _evidence_integrity_failure(inherited_problems)
        )
        checks = {
            "handoff_state_audit_evaluable": evaluable,
            "observable_evidence_gate_passed": observable_evidence_passed,
            "terminal_claim_is_handoff_not_resolution": True,
        }
        reasons = []
        if not observable_evidence_passed:
            reasons.append("observable_evidence_gate_failed")
        if not evaluable:
            reasons.append("handoff_state_audit_unavailable")
        return _report(
            action_class="operator_escalation",
            checks=checks,
            reason_codes=reasons,
        )

    if tool in _OBSERVABLE_ONLY_ACTIONS:
        # Information-gathering and diagnostic actions do not need to identify
        # the hidden family in advance.  Their existing observable evidence
        # gate is the complete admission condition for this offline audit.
        return _report(
            action_class="read_only",
            checks={"observable_evidence_gate_passed": observable_evidence_passed},
            reason_codes=(
                ()
                if observable_evidence_passed
                else ("observable_evidence_gate_failed",)
            ),
        )

    return _report(
        action_class="unknown_target",
        checks={"teacher_target_is_known_nonmutating_action": False},
        reason_codes=("teacher_target_action_class_unknown",),
    )


__all__ = [
    "LEGACY_OFFLINE_TEACHER_TARGET_AUDIT_CONTRACTS",
    "OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT",
    "offline_teacher_target_audit",
    "validate_offline_teacher_target_audit_metadata",
]
