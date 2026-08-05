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
from psse_env.dagger.release_audit import audit_episode_against_truth
from psse_env.state_store import OracleState, PolicyObservation


OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT = (
    "dagger1_offline_teacher_target_truth_audit_v1"
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
            "accepted_state_nonregressive_and_healthy",
            "observable_evidence_gate_passed",
            "terminal_claim_is_handoff_not_resolution",
        }
    ),
    "read_only": frozenset({"observable_evidence_gate_passed"}),
    "unknown_target": frozenset(
        {"teacher_target_is_known_nonmutating_action"}
    ),
}
_ACTION_CLASSES = frozenset(_ACTION_CLASS_CHECKS)
_FAILURE_ONLY_ACTION_CLASSES = frozenset(
    {"missing_target", "invalid_target", "unknown_target"}
)
_CHECK_NAMES = frozenset(
    check
    for checks in _ACTION_CLASS_CHECKS.values()
    for check in checks
)
_REASON_CODES = frozenset(
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
        "candidate_failed_private_commit_safety",
        "rollback_would_discard_truth_safe_candidate",
        "resolved_claim_failed_private_release_audit",
        "handoff_failed_private_safety_audit",
        "teacher_target_action_class_unknown",
    }
)


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
    if value.get("contract") != OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT:
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
    expected_checks = _ACTION_CLASS_CHECKS[action_class]
    if set(checks) != expected_checks:
        raise ValueError(
            "offline teacher-target audit check schema is incomplete or unexpected"
        )
    reasons = value.get("reason_codes")
    if not isinstance(reasons, list):
        raise ValueError("offline teacher-target audit reason codes must be a list")
    if any(not isinstance(item, str) or item not in _REASON_CODES for item in reasons):
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
    targets: set[int] = set()
    updates = arguments.get("measurement_updates")
    if updates is not None:
        if not isinstance(updates, Mapping):
            return None
        try:
            targets.update(int(value) for value in updates)
        except (TypeError, ValueError, OverflowError):
            return None
    group = arguments.get("suspect_group")
    if group is not None:
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            return None
        try:
            targets.update(int(value) for value in group)
        except (TypeError, ValueError, OverflowError):
            return None
    for key in (
        "measurement_index",
        "measurement_id",
        "index",
        "index0",
        "target",
        "meter",
    ):
        if arguments.get(key) is None:
            continue
        try:
            targets.add(int(arguments[key]))
        except (TypeError, ValueError, OverflowError):
            return None
    return targets


def _truth_measurement_targets(truth: Mapping[str, Any]) -> set[int] | None:
    raw = truth.get("true_measurement_errors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    targets: set[int] = set()
    for fault in raw:
        if not isinstance(fault, Mapping):
            return None
        value = next(
            (
                fault[key]
                for key in ("index", "index0", "measurement_index")
                if fault.get(key) is not None
            ),
            None,
        )
        if value is None:
            return None
        try:
            targets.add(int(value))
        except (TypeError, ValueError, OverflowError):
            return None
    return targets


def _branch_row0(value: Mapping[str, Any]) -> int | None:
    for key, offset in (
        ("branch_row0", 0),
        ("line_index1", -1),
        ("line_index", -1),
    ):
        if value.get(key) is None:
            continue
        try:
            result = int(value[key]) + offset
        except (TypeError, ValueError, OverflowError):
            return None
        return result if result >= 0 else None
    return None


def _named_branch_target(value: Mapping[str, Any]) -> tuple[str, str] | None:
    for key in ("branch_id", "cb_name"):
        if value.get(key) is not None:
            return key, str(value[key])
    return None


def _branch_target_matches(
    arguments: Mapping[str, Any], faults: Any
) -> tuple[bool, Mapping[str, Any] | None]:
    if not isinstance(faults, Sequence) or isinstance(faults, (str, bytes)):
        return False, None
    action_row = _branch_row0(arguments)
    action_named = _named_branch_target(arguments)
    if action_row is None and action_named is None:
        return False, None
    for fault in faults:
        if not isinstance(fault, Mapping):
            continue
        if action_row is not None and _branch_row0(fault) == action_row:
            return True, fault
        if action_named is not None and _named_branch_target(fault) == action_named:
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
        accepted = [
            copy.deepcopy(dict(item))
            for item in observation.get("accepted_corrections") or []
            if isinstance(item, Mapping)
        ]
        if source_action is not None:
            accepted.append({"source_action": source_action})
        hypothetical["accepted_corrections"] = accepted
        loader = case_loader if case_loader is not None else _default_case_loader()
        physically_safe = bool(
            isinstance(candidate, Mapping)
            and _release_safety_check(
                scenario=scenario,
                policy_observation=hypothetical,
                active_physical_state=candidate,
                terminal_outcome="operator_escalation",
                remaining_truth=None,
                case_loader=loader,
            )
        )
        target_correct = bool(
            source_truth_checks
            and all(source_truth_checks.values())
            and not source_reasons
        )
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
            reasons.append("candidate_failed_private_commit_safety")
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
        safe = _release_safety_check(
            scenario=scenario,
            policy_observation=observation,
            active_physical_state=_active_physical_state(env),
            terminal_outcome="operator_escalation",
            remaining_truth=truth,
            case_loader=loader,
        )
        checks = {
            "accepted_state_nonregressive_and_healthy": safe,
            "observable_evidence_gate_passed": observable_evidence_passed,
            "terminal_claim_is_handoff_not_resolution": True,
        }
        reasons = []
        if not observable_evidence_passed:
            reasons.append("observable_evidence_gate_failed")
        if not safe:
            reasons.append("handoff_failed_private_safety_audit")
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
    "OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT",
    "offline_teacher_target_audit",
    "validate_offline_teacher_target_audit_metadata",
]
