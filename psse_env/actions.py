from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from psse_env.evidence_profile import (
    DEFAULT_EVIDENCE_PROFILE,
    GATED_DIAGNOSTIC_TOOLS,
    allows_diagnostic_tools,
    disabled_requests,
    disabled_tools,
    is_scada_only,
    is_strict_boundary,
    is_wls_gated,
    requires_wls_alarm_for_diagnostics,
)


RUN_WLS = "run_wls"
VERIFY_CANDIDATE = "verify_candidate"
GET_MEASUREMENT_CONTEXT = "get_measurement_context"
GET_PARAMETER_CONTEXT = "get_parameter_context"
GET_TOPOLOGY_CONTEXT = "get_topology_context"
CORRECT_MEASUREMENTS = "correct_measurements"
CORRECT_PARAMETERS = "correct_parameters"
CORRECT_TOPOLOGY = "correct_topology"
COMMIT_STATE = "commit_state"
ROLLBACK_STATE = "rollback_state"
FINALIZE_DIAGNOSIS = "finalize_diagnosis"
ASK_FOR_MORE_EVIDENCE = "ask_for_more_evidence"
RUN_ALTERNATIVE_TEST = "run_alternative_test"
HIF_DIAGNOSTICS_EXHAUSTED_REQUEST = (
    "operator_escalation:hif_diagnostics_exhausted"
)
HIF_CONDITIONING_UNAVAILABLE_REQUEST = (
    "operator_escalation:hif_conditioning_unavailable"
)
RECOVERY_OPTIONS_EXHAUSTED_REQUEST = (
    "operator_escalation:recovery_options_exhausted"
)
RECOVERY_BUDGET_EXHAUSTED_REQUEST = (
    "operator_escalation:recovery_budget_exhausted"
)
# A branch fault whose top-ranked line does not dominate its runner-up under
# the parameter-ranking contract.  The ranked candidates are tested under
# verification; once every one of them is rejected the diagnosis is handed to
# the operator bounded to that candidate set, instead of falling through to a
# meter correction that would only mask the fault.
AMBIGUOUS_BRANCH_CANDIDATES_REQUEST = (
    "operator_escalation:ambiguous_branch_candidates"
)
#: How many ranked lines the parameter context offers when no line dominates.
PARAMETER_RANKING_AMBIGUITY_CANDIDATES = 2


def ambiguous_branch_candidate_lines(observation: Mapping[str, Any]) -> list[int] | None:
    """The ranked parameter candidates that were all rejected on this state.

    ``None`` unless the fresh parameter context on the active state declared
    the ranking ambiguous, named its candidate lines, and every one of those
    lines has a verification-rejected ``correct_parameters`` hypothesis whose
    parent is the active state.  This is the policy-visible precondition of
    the bounded operator handoff; the expert, the escalation provider, and
    the environment audit all read the same evidence.
    """

    contexts = observation.get("fresh_context_evidence")
    contexts = contexts if isinstance(contexts, Mapping) else {}
    evidence = contexts.get("parameter")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if evidence.get("parameter_ranking_ambiguous") is not True:
        return None
    active_id = str(observation.get("active_state_id") or "")
    if active_id and str(evidence.get("state_id") or "") != active_id:
        return None
    candidates = [
        int(line)
        for line in evidence.get("parameter_ranking_candidate_lines") or []
        if isinstance(line, int) and not isinstance(line, bool)
    ]
    if not candidates:
        return None
    rejected: set[int] = set()
    for record in observation.get("rejected_hypotheses") or []:
        if not isinstance(record, Mapping):
            continue
        if record.get("rejection_kind") == "executor_failure":
            # A solver failure closes an attempted recovery route but does
            # not establish a verification-rejected physical hypothesis.
            continue
        parent = record.get("candidate_parent_id")
        if parent is not None and active_id and str(parent) != active_id:
            continue
        source = record.get("source_action")
        if not isinstance(source, Mapping) or source.get("tool") != CORRECT_PARAMETERS:
            continue
        arguments = source.get("arguments")
        line = arguments.get("line_index") if isinstance(arguments, Mapping) else None
        if isinstance(line, int) and not isinstance(line, bool):
            rejected.add(int(line))
    if not set(candidates) <= rejected:
        return None
    return candidates
# An accepted correction can make the candidate WLS statistic quiescent
# without proving that every physical error has been removed.  Production
# mode persists this policy-visible protocol obligation until a same-state
# investigation either supplies another supported correction or justifies an
# operator handoff.  It is deliberately an observable controller marker, not
# a hidden-truth fault label.
POST_CORRECTION_CONFIRMATION_SIGNATURE = (
    "post_correction_resolution_confirmation_required:measurement_context"
)
GET_HARMONIC_CONTEXT = "get_harmonic_context"
GET_THREE_PHASE_CONTEXT = "get_three_phase_context"
RUN_HSE_FROM_PATH = "run_hse_from_path"
RUN_THREE_PHASE_NLM_FROM_PATH = "run_three_phase_nlm_from_path"
ESTIMATE_HIF_FROM_PATH = "estimate_hif_location_magnitude_from_path"
ESTIMATE_HIF_MULTISCAN_FROM_PATH = "estimate_hif_location_magnitude_multiscan_from_path"
INVALID_ACTION = "__invalid_action__"

CORRECTION_TOOLS = {
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
}

CONTEXT_TOOLS = {
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
}

# Failure codes of a process-gate or lifecycle refusal: the action never
# reached a provider or executor, so it tested nothing.  The environment's
# operator-escalation audit does not count such an attempt as an exhausted
# recovery target, and the expert does not count it as a tried action.
PROCESS_REJECTION_ERROR_CODES = frozenset(
    {
        "schema_error",
        "unknown_tool",
        "candidate_lifecycle_violation",
        "unknown_state_id",
        "state_reference_mismatch",
        "missing_precondition",
        "post_correction_confirmation_required",
    }
)

STATE_MANAGEMENT_TOOLS = {
    COMMIT_STATE,
    ROLLBACK_STATE,
    FINALIZE_DIAGNOSIS,
}

# Read-only specialized diagnostics executed through configured evidence
# providers.  They share the canonical deployment tool names so DAgger data
# and the production corpus keep one model-visible surface.
DIAGNOSTIC_TOOLS = {
    GET_HARMONIC_CONTEXT,
    GET_THREE_PHASE_CONTEXT,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
}

MACRO_ACTIONS = {
    RUN_WLS,
    VERIFY_CANDIDATE,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    COMMIT_STATE,
    ROLLBACK_STATE,
    FINALIZE_DIAGNOSIS,
    ASK_FOR_MORE_EVIDENCE,
    RUN_ALTERNATIVE_TEST,
    *DIAGNOSTIC_TOOLS,
}

# Observable anomaly-signature vocabulary shared by expert routing and the
# explained-anomaly termination semantics.  A diagnostic explanation for a
# family accounts for the unresolved signatures matching that family's
# markers; families not listed here can only be resolved by corrections.
ANOMALY_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "harmonic": ("harmonic", "harmonics", "thd", "distortion", "waveform"),
    "three_phase_unbalance": (
        "three_phase_unbalance",
        "voltage_unbalance",
        "unbalance",
        "imbalance",
        "negative_sequence",
        "vuf",
    ),
    "hif": (
        "hif",
        "high_impedance",
        "high_impedance_fault",
        "arc",
        "arcing",
        "downed_conductor",
        "zero_sequence_hif",
    ),
}


#: Families whose anomaly is a physical waveform-level event.  An accepted
#: explanation closes such a signature for termination, but it does not remove
#: the event from the network: the fundamental-frequency operator vector stays
#: inconsistent with the balanced model, so residual-based bad-data routes
#: remain unreliable for as long as the signature stands.
WAVEFORM_ANOMALY_FAMILIES = ("harmonic", "three_phase_unbalance", "hif")


def waveform_anomaly_signatures(unresolved: Any) -> list[str]:
    """Observable waveform-family signatures in ``unresolved``, explained or not.

    Word-boundary marker matching on the lower-cased signature text, the same
    rule the routing helpers use, so a signature that routes to a diagnostic
    is the same signature that blocks the fundamental-frequency routes.
    """
    patterns = tuple(
        re.compile(rf"(?<![a-z0-9]){re.escape(marker.lower())}(?![a-z0-9])")
        for family in WAVEFORM_ANOMALY_FAMILIES
        for marker in ANOMALY_FAMILY_MARKERS[family]
    )
    found: list[str] = []
    for item in unresolved or []:
        text = str(item).lower()
        if any(pattern.search(text) for pattern in patterns):
            found.append(str(item))
    return found


#: Telemetry channels that let the three-phase state be screened directly.
THREE_PHASE_TELEMETRY_CHANNELS = frozenset(
    {"three_phase_voltages", "three_phase_branch_currents"}
)


def _ledger_wls_alarm(context_evidence: Any, active_state_id: Any, *, content_bound: bool) -> bool:
    """A current, successful WLS ledger entry whose detection rule fired.

    The alarm is the pair of provider flags (chi-square at the configured
    level OR the largest normalized residual at its threshold).  ``wls_*``
    signatures are not a substitute: the solve mints none while a waveform
    signature stands, and a chi-square-only alarm may mint none at all.
    """
    if not isinstance(context_evidence, Mapping):
        return False
    evidence = context_evidence.get("wls")
    if not isinstance(evidence, Mapping) or evidence.get("successful") is not True:
        return False
    if str(evidence.get("state_id") or "") != str(active_state_id or ""):
        return False
    if content_bound and not (isinstance(evidence.get("state_hash"), str) and evidence["state_hash"]):
        return False
    return evidence.get("chi_square_alarm") is True or evidence.get("normalized_residual_alarm") is True


def current_wls_alarm(state: Any, history: Any = None) -> bool:
    """Whether the active state's current balanced WLS reports an alarm.

    This is the expert-side mirror of the ``wls_gated_diagnostics`` gate:
    auxiliary streams and their diagnostics may be requested only while a
    bound successful ``run_wls``/``verify_candidate`` on the active state has
    ``chi_square_alarm`` or ``normalized_residual_alarm`` set.  Strict
    profiles read the durable content-bound ledger only; the auxiliary
    profile may also read the latest same-state WLS output in the visible
    history for compact fixtures without a ledger.
    """
    active = str(state.get("active_state_id") or "")
    contexts = state.get("fresh_context_evidence") or {}
    if isinstance(contexts, Mapping) and "wls" in contexts:
        return _ledger_wls_alarm(contexts, active, content_bound=is_strict_boundary(state))
    if is_strict_boundary(state):
        return False
    events = list(history if history is not None else state.get("history_window") or [])
    if state.get("last_tool") in {RUN_WLS, VERIFY_CANDIDATE}:
        events.append({
            "action": {"tool": state.get("last_tool"), "arguments": {"state_id": active}},
            "tool_output": state.get("last_tool_output") or {},
        })
    for event in reversed(events):
        if not isinstance(event, Mapping):
            continue
        action = safe_normalize_action(event.get("action") or event.get("executed_action") or event)
        if action["tool"] not in {RUN_WLS, VERIFY_CANDIDATE}:
            continue
        requested = str(action["arguments"].get("state_id") or "")
        if active and requested and requested != active:
            continue
        output = event.get("tool_output") or event.get("outcome") or {}
        if not isinstance(output, Mapping) or output.get("execution_status") != "success":
            return False
        metrics = output.get("tool_metrics")
        if not isinstance(metrics, Mapping):
            metrics = event.get("observable_metrics")
        if not isinstance(metrics, Mapping):
            return False
        return metrics.get("chi_square_alarm") is True or metrics.get("normalized_residual_alarm") is True
    return False


def _fundamental_anomaly_trigger(
    signatures: Sequence[str], context_evidence: Any, active_state_id: Any, evidence_profile: Any,
) -> bool:
    """Whether a balanced WLS anomaly may open an auxiliary evidence request.

    Under ``wls_gated_diagnostics`` the trigger is the current alarm on the
    active state (the same condition the environment gate enforces).  The
    historical auxiliary profile keeps its ``wls_*``-signature trigger.
    """
    if requires_wls_alarm_for_diagnostics(evidence_profile):
        return _ledger_wls_alarm(context_evidence, active_state_id, content_bound=True)
    return any(item.startswith("wls_") for item in signatures)


def diagnostic_tool_permitted(state: Any, tool: str, request: Any = None, history: Any = None) -> bool:
    """Whether the evidence profile lets the expert propose ``tool`` now.

    Disabled tools and disabled operator requests are never proposed.  Gated
    auxiliary diagnostics need the current balanced alarm under the WLS-gated
    profile; every other tool is unaffected.
    """
    if tool in disabled_tools(state):
        return False
    if tool == ASK_FOR_MORE_EVIDENCE and request is not None and str(request) in disabled_requests(state):
        return False
    if tool in GATED_DIAGNOSTIC_TOOLS and requires_wls_alarm_for_diagnostics(state):
        return current_wls_alarm(state, history)
    return True


def harmonic_screening_pending(
    *, unresolved: Any, tried_action_signatures: Any, active_state_id: Any,
    context_evidence: Any = None, evidence_profile: Any = DEFAULT_EVIDENCE_PROFILE,
) -> bool:
    """An observed WLS anomaly permits one request for spectral evidence.

    This predicate deliberately does not inspect telemetry availability or a
    hidden scenario family. Only a process-valid request enters the request
    ledger. Rejected pre-WLS calls must not bypass later spectral acquisition.
    """
    if not allows_diagnostic_tools(evidence_profile):
        return False
    signatures = [str(item) for item in (unresolved or [])]
    if not _fundamental_anomaly_trigger(signatures, context_evidence, active_state_id, evidence_profile):
        return False
    if waveform_anomaly_signatures(signatures):
        return False
    active = str(active_state_id or "")
    if isinstance(context_evidence, Mapping):
        request = context_evidence.get("harmonic") or {}
        return not (
            isinstance(request, Mapping)
            and str(request.get("state_id") or "") == active
            and request.get("request_attempted") is True
        )
    # Compatibility for legacy compact fixtures without a request ledger.
    for signature in tried_action_signatures or []:
        tool, _, encoded = str(signature).partition(":")
        if tool != GET_HARMONIC_CONTEXT:
            continue
        try:
            arguments = json.loads(encoded) if encoded else {}
        except ValueError:
            continue
        requested = str((arguments or {}).get("state_id") or "")
        if not active or not requested or requested == active:
            return False
    return True


def three_phase_acquisition_pending(
    *, unresolved: Any, tried_action_signatures: Any, active_state_id: Any,
    context_evidence: Any = None, evidence_profile: Any = DEFAULT_EVIDENCE_PROFILE,
) -> bool:
    """Request phase-resolved evidence after a generic WLS anomaly.

    Acquisition depends on observable anomalies and the request ledger, never
    on whether the hidden root happens to carry phase-resolved telemetry.
    Rejected calls do not count as acquisitions.
    """
    if not allows_diagnostic_tools(evidence_profile):
        return False
    signatures = [str(item) for item in (unresolved or [])]
    if not _fundamental_anomaly_trigger(signatures, context_evidence, active_state_id, evidence_profile):
        return False
    if waveform_anomaly_signatures(signatures):
        return False
    active = str(active_state_id or "")
    if isinstance(context_evidence, Mapping):
        request = context_evidence.get("three_phase") or {}
        return not (
            isinstance(request, Mapping)
            and str(request.get("state_id") or "") == active
            and request.get("request_attempted") is True
        )
    # Compatibility for compact fixtures predating the acquisition ledger.
    for signature in tried_action_signatures or []:
        tool, _, encoded = str(signature).partition(":")
        if tool != GET_THREE_PHASE_CONTEXT:
            continue
        try:
            arguments = json.loads(encoded) if encoded else {}
        except ValueError:
            continue
        requested = str((arguments or {}).get("state_id") or "")
        if not active or not requested or requested == active:
            return False
    return True


def three_phase_context_available(context_evidence: Any, active_state_id: Any) -> bool:
    """Whether a successful acquisition exposes current phase telemetry."""
    if not isinstance(context_evidence, Mapping):
        return False
    context = context_evidence.get("three_phase") or {}
    return bool(
        isinstance(context, Mapping)
        and str(context.get("state_id") or "") == str(active_state_id or "")
        and context.get("request_attempted") is True
        and context.get("three_phase_context_status") == "available"
        and set(context.get("available_evidence_channels") or [])
        & THREE_PHASE_TELEMETRY_CHANNELS
    )


#: Share of normalized residuals above the outlier threshold at which a WLS
#: anomaly counts as broad.  On the legacy corpora spectral distortion
#: elevated 75 to 84 of the 122 channels, a load unbalance 7 to 19, a bad
#: meter 1 to 5, and the broadest branch fault 49.  On the regenerated
#: 2026-09-21 corpora a harmonic alarm is narrow too (at most 27% of the
#: channels), so under the WLS-gated contract the rule only orders the two
#: auxiliary requests and three-phase phasors are asked for first in
#: practice; spectra follow when the phasors are unavailable.
BROAD_ANOMALY_BREADTH = 0.5


def wls_anomaly_breadth(state: Any, history: Any = None) -> float | None:
    """Residual breadth of the current-state WLS, from the ledger or history."""
    active = str(state.get("active_state_id") or "")
    contexts = state.get("fresh_context_evidence") or {}
    if isinstance(contexts, Mapping):
        evidence = contexts.get("wls") or {}
        if (
            isinstance(evidence, Mapping)
            and str(evidence.get("state_id") or "") == active
            and evidence.get("successful") is True
            and evidence.get("anomaly_breadth") is not None
        ):
            try:
                return float(evidence["anomaly_breadth"])
            except (TypeError, ValueError):
                return None
    events = list(history if history is not None else state.get("history_window") or [])
    if state.get("last_tool") == RUN_WLS:
        events.append(
            {
                "action": {"tool": RUN_WLS, "arguments": {"state_id": active}},
                "tool_output": state.get("last_tool_output") or {},
            }
        )
    for event in reversed(events):
        if not isinstance(event, Mapping):
            continue
        action = safe_normalize_action(
            event.get("action") or event.get("executed_action") or event
        )
        if action["tool"] != RUN_WLS:
            continue
        requested = str(action["arguments"].get("state_id") or "")
        if active and requested and requested != active:
            continue
        output = event.get("tool_output") or event.get("outcome") or {}
        metrics = output.get("tool_metrics") if isinstance(output, Mapping) else None
        if not isinstance(metrics, Mapping):
            metrics = event.get("observable_metrics")
        if isinstance(metrics, Mapping) and metrics.get("anomaly_breadth") is not None:
            try:
                return float(metrics["anomaly_breadth"])
            except (TypeError, ValueError):
                return None
    return None


def current_gnn_screen(state: Any) -> Mapping[str, Any]:
    """Only current, successful, content-bound WLS can expose a learned screen."""
    contexts = state.get("fresh_context_evidence") or {}
    wls = (contexts.get("wls") or {}) if isinstance(contexts, Mapping) else {}
    screen = (wls.get("gnn_screen") or {}) if isinstance(wls, Mapping) else {}
    if (
        isinstance(screen, Mapping)
        and wls.get("successful") is True
        and bool(wls.get("state_id")) and bool(wls.get("state_hash"))
        and str(wls["state_id"]) == str(state.get("active_state_id") or "")
        and screen.get("state_id") == wls["state_id"]
        and screen.get("state_hash") == wls["state_hash"]
        and screen.get("screen_status") == "valid"
    ):
        return screen
    return {}


def gnn_investigation_pending(state: Any) -> bool:
    """A quiet WLS result cannot skip an outstanding learned investigation.

    These are acquisition/context obligations, not physical fault assertions.
    A reported unavailable phase acquisition satisfies the request; acquired
    telemetry must be investigated before statistical episode closure.
    """
    screen = current_gnn_screen(state)
    if not (screen.get("phase_trigger") is True or screen.get("anomaly_trigger") is True):
        return False
    if is_strict_boundary(state):
        # A learned score derived from these same SCADA/WLS values may request
        # balanced investigation; it does not create a phase sensor or a fault
        # diagnosis. Both anomaly and phase heads use the same generic route.
        # (The WLS-gated profile refuses the learned screen altogether; a
        # legacy record is still read as a balanced request only.)
        contexts = state.get("fresh_context_evidence") or {}
        return any(not (
            isinstance(contexts.get(family), Mapping)
            and contexts[family].get("state_id") == screen["state_id"]
            and contexts[family].get("state_hash") == screen["state_hash"]
        ) for family in ("measurement", "parameter", "topology"))
    waveform = waveform_anomaly_signatures(state.get("unresolved_signatures") or [])
    if waveform:
        # Independent acquired waveform evidence supersedes the screening
        # hypothesis and follows its existing diagnostic ladder. Do not demand
        # balanced contexts or a second phase acquisition after that diagnosis.
        return bool(unexplained_signatures(waveform, state.get("explained_anomalies") or []))
    contexts = state.get("fresh_context_evidence") or {}
    if screen.get("phase_trigger") is True:
        phase = contexts.get("three_phase") or {}
        if not (isinstance(phase, Mapping) and phase.get("state_id") == screen["state_id"]
                and phase.get("state_hash") == screen["state_hash"] and phase.get("request_attempted") is True):
            return True
        return phase.get("three_phase_context_status") != "unavailable" and phase.get("nlm_attempted") is not True
    if screen.get("phase_trigger") is False and screen.get("anomaly_trigger") is True:
        return any(not (
            isinstance(contexts.get(family), Mapping)
            and contexts[family].get("state_id") == screen["state_id"]
            and contexts[family].get("state_hash") == screen["state_hash"]
        ) for family in ("measurement", "parameter", "topology"))
    return False


def preferred_first_request(state: Any, history: Any = None) -> str:
    """Which additional measurement to request first after a WLS anomaly.

    A broad anomaly (most channels inconsistent with the balanced model) is
    the signature of spectral distortion, so spectra are requested first; a
    narrow one points at a phase-resolved event or a meter, so three-phase
    measurements come first.  Either request falls back to the other when it
    returns nothing.  Without a breadth statistic (compact fixtures) the
    spectral request keeps its historical precedence in the auxiliary
    profile; the WLS-gated profile asks for phasors first, because every
    alarm on its corpora is narrow.
    """
    if is_scada_only(state):
        return GET_MEASUREMENT_CONTEXT
    if not is_strict_boundary(state) and current_gnn_screen(state).get("phase_trigger") is True:
        return GET_THREE_PHASE_CONTEXT
    breadth = wls_anomaly_breadth(state, history)
    if breadth is None:
        return GET_THREE_PHASE_CONTEXT if is_wls_gated(state) else GET_HARMONIC_CONTEXT
    if breadth < BROAD_ANOMALY_BREADTH:
        return GET_THREE_PHASE_CONTEXT
    return GET_HARMONIC_CONTEXT


def successful_current_wls(state: Any, history: Any = None) -> bool:
    """Read same-state WLS proof from the durable ledger or visible history."""
    active = str(state.get("active_state_id") or "")
    contexts = state.get("fresh_context_evidence") or {}
    if isinstance(contexts, Mapping) and "wls" in contexts:
        evidence = contexts.get("wls") or {}
        return bool(
            isinstance(evidence, Mapping)
            and str(evidence.get("state_id") or "") == active
            and evidence.get("successful") is True
            and (not is_strict_boundary(state) or (
                isinstance(evidence.get("state_hash"), str) and bool(evidence["state_hash"])
            ))
        )
    if is_strict_boundary(state):
        # New strict episodes always publish a content-bound WLS ledger. A
        # legacy score, family flag or claimed last tool is not that proof.
        return False
    provenance = state.get("semantic_field_provenance") or {}
    source = str(provenance.get("remaining_anomaly_score") or "").lower()
    if state.get("remaining_anomaly_score") is not None and (
        "wls" in source or source.startswith("observable_candidate_verification")
    ):
        return True
    events = list(history if history is not None else state.get("history_window") or [])
    if state.get("last_tool") == RUN_WLS:
        events.append({
            "action": {"tool": RUN_WLS, "arguments": {"state_id": active}},
            "tool_output": state.get("last_tool_output") or {},
        })
    for event in events:
        if not isinstance(event, Mapping):
            continue
        action = safe_normalize_action(event.get("action") or event.get("executed_action") or event)
        output = event.get("tool_output") or event.get("outcome") or {}
        requested = action["arguments"].get("state_id")
        if (
            action["tool"] == RUN_WLS
            and str(requested or "") == active
            and isinstance(output, Mapping)
            and output.get("execution_status") == "success"
        ):
            return True
    return False


def three_phase_screening_pending(
    *,
    unresolved: Any,
    available_evidence: Any,
    tried_action_signatures: Any,
    active_state_id: Any,
    context_evidence: Any = None, evidence_profile: Any = DEFAULT_EVIDENCE_PROFILE,
) -> bool:
    """Whether an unflagged WLS anomaly still awaits its three-phase screening.

    A ``wls_*`` signature (a current balanced alarm under the WLS-gated
    profile) with no waveform-family signature requires NLM once a
    successful acquisition exposes phase telemetry. The durable ledger
    counts process-valid NLM dispatches, so rejected premature attempts never
    suppress later screening, even when their history has been truncated.
    """
    if not allows_diagnostic_tools(evidence_profile):
        return False
    signatures = [str(item) for item in (unresolved or [])]
    if not _fundamental_anomaly_trigger(signatures, context_evidence, active_state_id, evidence_profile):
        return False
    if waveform_anomaly_signatures(signatures):
        return False
    channels = {str(item) for item in (available_evidence or [])}
    if not (channels & THREE_PHASE_TELEMETRY_CHANNELS):
        return False
    active = str(active_state_id or "")
    if isinstance(context_evidence, Mapping):
        if not three_phase_context_available(context_evidence, active):
            return False
        context = context_evidence["three_phase"]
        return context.get("nlm_attempted") is not True
    for signature in tried_action_signatures or []:
        text = str(signature)
        tool, _, encoded = text.partition(":")
        if tool != RUN_THREE_PHASE_NLM_FROM_PATH:
            continue
        try:
            arguments = json.loads(encoded) if encoded else {}
        except ValueError:
            continue
        requested = str((arguments or {}).get("state_id") or "")
        if not active or not requested or requested == active:
            return False
    return True


def unexplained_signatures(
    unresolved: Any,
    explained_records: Any,
) -> list[str]:
    """Unresolved signatures not covered by any recorded diagnostic explanation."""
    explained: set[str] = set()
    for record in explained_records or []:
        if not isinstance(record, Mapping):
            continue
        for signature in record.get("explained_signatures") or []:
            explained.add(str(signature))
    return [str(item) for item in (unresolved or []) if str(item) not in explained]


def terminal_explanation_signatures(unresolved: Any) -> list[str]:
    """Return physical/diagnostic signatures relevant to explanation closure.

    The post-correction confirmation marker is a process obligation rather
    than an anomaly that a diagnostic estimator can explain.  Filtering it
    here preserves valid explanation-only closure while ensuring that the
    marker by itself never authorizes finalization.
    """

    return [
        str(item)
        for item in (unresolved or [])
        if str(item) != POST_CORRECTION_CONFIRMATION_SIGNATURE
    ]


def invalid_action(error_code: str, error_detail: str | None = None) -> dict[str, Any]:
    """Return the canonical learner-action representation for malformed output."""
    arguments: dict[str, Any] = {"error_code": str(error_code)}
    if error_detail:
        arguments["error_detail"] = str(error_detail)
    return {"tool": INVALID_ACTION, "arguments": arguments}


def normalize_action(action: Mapping[str, Any] | str) -> dict[str, Any]:
    """Normalize a tool-call-like object to {"tool": name, "arguments": {...}}."""
    if isinstance(action, str):
        stripped = action.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            decoded = json.loads(stripped)
            if not isinstance(decoded, Mapping):
                raise ValueError("JSON action must decode to an object.")
            return normalize_action(decoded)
        return {"tool": stripped, "arguments": {}}
    if not isinstance(action, Mapping):
        raise TypeError(f"action must be a mapping or string, got {type(action).__name__}")

    function = action.get("function")
    if function is not None and not isinstance(function, Mapping):
        raise ValueError("Action function must be a mapping.")
    tool = (
        action.get("tool")
        or action.get("name")
        or action.get("tool_name")
        or (function or {}).get("name")
    )
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"Action has no tool name: {action!r}")

    arguments = action.get("arguments")
    if arguments is None and isinstance(action.get("function"), Mapping):
        arguments = action["function"].get("arguments")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError(f"Action arguments for {tool} must be a mapping.")

    normalized_arguments = _canonicalize_correction_arguments(tool, dict(arguments))
    normalized = {"tool": tool, "arguments": normalized_arguments}
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action must be JSON-serializable: {exc}") from exc
    return normalized


def _canonicalize_correction_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Flatten the one supported correction payload and reject ambiguity.

    Older traces may wrap physical edits in ``arguments.modification`` while
    newer tool calls put them directly in ``arguments``.  Execution, process
    checks, provenance, and candidate assessment must all see the same fields;
    accepting conflicting copies would let a call claim one target and mutate
    another.
    """
    if tool not in CORRECTION_TOOLS:
        return arguments

    nested = arguments.pop("modification", None)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("Correction modification must be a mapping.")
        for key, value in nested.items():
            if key in arguments and arguments[key] != value:
                raise ValueError(f"Conflicting correction field in modification: {key}")
            arguments.setdefault(str(key), value)

    updates = arguments.get("measurement_updates")
    if isinstance(updates, (list, tuple)):
        normalized_updates: dict[int, Any] = {}
        for item in updates:
            if not isinstance(item, Mapping):
                raise ValueError("measurement_updates list entries must be mappings.")
            raw_index = item.get("index", item.get("index0"))
            if raw_index is None or "value" not in item:
                raise ValueError("measurement_updates entries require index and value.")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("measurement update indices must be integers.") from exc
            if index in normalized_updates:
                raise ValueError(f"Duplicate measurement update index: {index}")
            normalized_updates[index] = item["value"]
        arguments["measurement_updates"] = normalized_updates
    elif isinstance(updates, Mapping):
        normalized_updates = {}
        for raw_index, value in updates.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("measurement update indices must be integers.") from exc
            if index in normalized_updates:
                raise ValueError(f"Duplicate measurement update index: {index}")
            normalized_updates[index] = value
        arguments["measurement_updates"] = normalized_updates

    if tool == CORRECT_MEASUREMENTS and isinstance(arguments.get("measurement_updates"), Mapping):
        declared = set(arguments["measurement_updates"])
        aliases: set[int] = set()
        for key in ("measurement_index", "index", "index0", "target", "meter", "measurement_id"):
            if arguments.get(key) is None:
                continue
            try:
                aliases.add(int(arguments[key]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Measurement target {key!r} must be the numeric updated index."
                ) from exc
        if aliases and aliases != declared:
            raise ValueError("Measurement target aliases conflict with measurement_updates.")

    if tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
        numeric_targets = [
            key
            for key in ("line_index", "line_index1", "branch_row0")
            if arguments.get(key) is not None
        ]
        named_targets = [
            key for key in ("branch_id", "cb_name") if arguments.get(key) is not None
        ]
        # A breaker name identifies the switch of a node/breaker correction; it
        # may accompany exactly one numeric row, the branch that switch affects
        # in the operator's bus-branch model.  Every other combination is
        # ambiguous.
        if (
            len(numeric_targets) > 1
            or len(named_targets) > 1
            or (numeric_targets and named_targets and named_targets != ["cb_name"])
        ):
            raise ValueError("Correction must use exactly one branch target convention.")
    return arguments


def safe_normalize_action(action: Any) -> dict[str, Any]:
    """Normalize arbitrary policy output without raising.

    DAgger must retain malformed learner outputs as recovery examples.  This
    helper therefore converts parsing and schema failures to the same sentinel
    action consumed by the process-validity gate.
    """
    if isinstance(action, bytes):
        try:
            action = action.decode("utf-8")
        except UnicodeDecodeError as exc:
            return invalid_action("argument_decode_error", str(exc))

    if isinstance(action, str):
        stripped = action.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                action = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return invalid_action("json_parse_error", exc.msg)

    if isinstance(action, Mapping):
        arguments = action.get("arguments")
        function = action.get("function")
        if arguments is None and isinstance(function, Mapping):
            arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                decoded_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return invalid_action("argument_decode_error", exc.msg)
            action = dict(action)
            if "arguments" in action:
                action["arguments"] = decoded_arguments
            elif isinstance(function, Mapping):
                action["function"] = dict(function)
                action["function"]["arguments"] = decoded_arguments

    try:
        return normalize_action(action)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return invalid_action("schema_error", str(exc))


def action_signature(action: Mapping[str, Any] | str) -> str:
    normalized = normalize_action(action)
    args_text = json.dumps(normalized["arguments"], sort_keys=True, separators=(",", ":"))
    return f"{normalized['tool']}:{args_text}"


def action_target(action: Mapping[str, Any] | str) -> str | None:
    args = normalize_action(action)["arguments"]
    for key in (
        "target",
        "meter",
        "measurement_id",
        "measurement_index",
        "branch_id",
        "line_index",
        "line_index1",
        "branch_row0",
        "cb_name",
    ):
        if key in args and args[key] is not None:
            return f"{key}={args[key]}"
    return None
