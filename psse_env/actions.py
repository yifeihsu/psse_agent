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
    is_classifier_gated,
    is_scada_only,
    is_strict_boundary,
    is_suspicion_gated,
    is_suspicion_profile,
    is_wls_gated,
    required_suspicion,
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
# A committed correction left the WLS quiet.  The controller does not let the
# agent certify a case resolved after editing its own model (an edit that only
# silences the alarm must not close the case), so the corrected state is handed
# to the operator for confirmation.  Under the suspicion-family profiles this
# is one request right after the commit (2026-10-07); the older path fetched a
# measurement context on the quiet state first and asked with
# RECOVERY_OPTIONS_EXHAUSTED_REQUEST.
POST_CORRECTION_CONFIRMATION_REQUEST = (
    "operator_escalation:post_correction_confirmation"
)
# The balanced routes are exhausted, the phasors acquired on the state came
# back balanced and the spectra showed no distortion (or were refused), yet the
# alarm stands: the diagnosis is handed over as an unexplained discrepancy,
# the honest end of the suspicion-gated ladder (2026-09-30).
UNEXPLAINED_DISCREPANCY_REQUEST = (
    "operator_escalation:unexplained_balanced_discrepancy"
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
#: Bounded comparison of the ranked parameter candidates (2026-10-19).  One
#: policy action fits each requested line on an isolated branch of the active
#: state, runs the same verification solve ``run_wls`` would run on that
#: candidate, and reports the observable verdicts side by side; no candidate is
#: opened and no state changes.  The decision audit of the hard draw found the
#: expert's parameter ordering wrong on 4 of 8 misranked training roots while
#: the runner-up line verified and solved the root; the fixed rule picks the
#: top-ranked line and commits a healthy branch instead.
COMPARE_PARAMETER_CANDIDATES = "compare_parameter_candidates"
PARAMETER_COMPARISON_MIN_CANDIDATES = 2
PARAMETER_COMPARISON_MAX_CANDIDATES = 3
PARAMETER_COMPARISON_CONTRACT = "branch_verified_parameter_comparison_v1"
#: The ``fresh_context_evidence`` family that records the comparison on the
#: active state; dropped with the other balanced contexts when the state changes.
PARAMETER_COMPARISON_FAMILY = "parameter_comparison"
_COMPARISON_DISPOSITION_RANK = {
    "ACCEPT_FINAL": 0, "ACCEPT_PARTIAL": 1, "INCONCLUSIVE": 2, "REJECT": 3,
}


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
def comparison_line_indices(value: Any) -> list[int] | None:
    """The distinct positive 1-based line indices of a comparison request.

    ``None`` when the request is malformed: not a list, a non-integer or
    boolean entry, a non-positive line, a repeated line, or a count outside
    the bounded window.
    """
    if not isinstance(value, (list, tuple)):
        return None
    lines: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item in lines:
            return None
        lines.append(int(item))
    if not PARAMETER_COMPARISON_MIN_CANDIDATES <= len(lines) <= PARAMETER_COMPARISON_MAX_CANDIDATES:
        return None
    return lines


def _comparison_sort_key(candidate: Mapping[str, Any]) -> tuple[int, float, float, int]:
    disposition = candidate.get("verification_verdict")
    if candidate.get("execution_status") != "success" or disposition is None:
        rank = len(_COMPARISON_DISPOSITION_RANK)
    else:
        rank = _COMPARISON_DISPOSITION_RANK.get(str(disposition), len(_COMPARISON_DISPOSITION_RANK))

    def finite(value: Any, default: float) -> float:
        if isinstance(value, bool) or value is None:
            return default
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return number if number == number and abs(number) != float("inf") else default

    progress = finite(candidate.get("global_progress"), float("-inf"))
    statistic = finite(candidate.get("chi_square_statistic"), float("inf"))
    request_rank = candidate.get("request_rank")
    request_rank = int(request_rank) if isinstance(request_rank, int) and not isinstance(request_rank, bool) else 0
    return (rank, -progress, statistic, request_rank)


def rank_parameter_comparison(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order the per-line verdicts of a comparison, best first.

    An accepted verification (final before partial) beats an inconclusive one,
    which beats a rejection or a failed fit; within a class the larger global
    progress of the verification solve wins, then the smaller chi-square
    statistic, then the order of the request.  ``verification_verdict`` is the
    environment's own truth-free verdict on the branch candidate and the other
    keys are the verification metrics the policy reads anyway.
    """
    ordered = sorted((dict(item) for item in candidates), key=_comparison_sort_key)
    for position, item in enumerate(ordered):
        item["comparison_rank"] = position + 1
    return ordered


def parameter_comparison_evidence(observation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The comparison recorded on the active state, else ``None``."""
    contexts = observation.get("fresh_context_evidence")
    contexts = contexts if isinstance(contexts, Mapping) else {}
    evidence = contexts.get(PARAMETER_COMPARISON_FAMILY)
    if not isinstance(evidence, Mapping):
        return None
    active_id = str(observation.get("active_state_id") or "")
    if active_id and str(evidence.get("state_id") or "") != active_id:
        return None
    if not isinstance(evidence.get("candidates"), (list, tuple)):
        return None
    return evidence


def parameter_comparison_ranked_lines(observation: Mapping[str, Any]) -> list[int]:
    """The compared lines of the active state, best verdict first (empty without a comparison)."""
    evidence = parameter_comparison_evidence(observation)
    if evidence is None:
        return []
    lines: list[int] = []
    for item in rank_parameter_comparison([c for c in evidence["candidates"] if isinstance(c, Mapping)]):
        line = item.get("line_index")
        if isinstance(line, int) and not isinstance(line, bool) and line not in lines:
            lines.append(int(line))
    return lines


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



def process_gate_refusal(error_code: Any) -> bool:
    """The step was refused before any provider ran, so it tested nothing.

    The process gate's own rejections, a tool the evidence profile disables,
    and the alarm and suspicion gates of the WLS-gated profiles
    (``diagnostics_require_*``).  A provider's own failure (an unavailable
    scan window, a numerical failure) is an answer and is not one of these.
    """
    code = str(error_code or "")
    return (
        code in PROCESS_REJECTION_ERROR_CODES
        or code == "evidence_profile_tool_unavailable"
        or code.startswith("diagnostics_require_")
    )


STATE_MANAGEMENT_TOOLS = {
    COMMIT_STATE,
    ROLLBACK_STATE,
    FINALIZE_DIAGNOSIS,
}

#: Item 7 step 6 (the user's decision of 2026-10-28, "extend rollback"): a
#: committed correction can be withdrawn by making an ancestor of the active
#: state active again.  The process gate admits it only after the configured
#: number of fruitless correction cycles since that ancestor's commit, with the
#: alarm still standing, at most once per ancestor; it costs an action.  The
#: tool exists only in an environment built with ``revert_fruitless_cycles``,
#: so it is in none of the vocabularies below and every other environment
#: refuses it as an unknown tool.
REVERT_STATE = "revert_state"
REVERT_CONTRACT = "gated_revert_v1"
#: ``rejection_kind`` of the rejected-hypothesis record a revert leaves on the
#: ancestor for the correction it withdrew there.
REVERTED_REJECTION_KIND = "reverted_after_fruitless_cycles"
_CORRECTION_HYPOTHESIS_FAMILY = {
    CORRECT_MEASUREMENTS: "measurement",
    CORRECT_PARAMETERS: "parameter",
    CORRECT_TOPOLOGY: "topology",
}


def correction_hypothesis(action: Any) -> tuple[str, int | None] | None:
    """The (family, line) a correction tests: the line for a branch correction, ``None`` for a meter edit."""
    normalized = safe_normalize_action(action)
    family = _CORRECTION_HYPOTHESIS_FAMILY.get(normalized["tool"])
    if family is None:
        return None
    arguments = normalized["arguments"]
    line = arguments.get("line_index", arguments.get("line_index1"))
    if family == "measurement" or isinstance(line, bool) or not isinstance(line, int):
        line = None
    return family, line


def revert_ledger_of(observation: Mapping[str, Any] | Any) -> Mapping[str, Any] | None:
    """The revert ledger an environment with the gated revert reports, else ``None``."""
    getter = getattr(observation, "get", None)
    ledger = getter("revert_ledger") if callable(getter) else None
    return ledger if isinstance(ledger, Mapping) and ledger.get("contract") == REVERT_CONTRACT else None


def withdrawn_families(observation: Mapping[str, Any] | Any) -> set[str]:
    """Families of the corrections a revert withdrew on the active state (item 7 step 6)."""
    getter = getattr(observation, "get", None)
    if not callable(getter):
        return set()
    active_id = str(getter("active_state_id") or "")
    families: set[str] = set()
    for record in getter("rejected_hypotheses") or []:
        if not isinstance(record, Mapping) or record.get("rejection_kind") != REVERTED_REJECTION_KIND:
            continue
        if active_id and str(record.get("candidate_parent_id") or "") != active_id:
            continue
        hypothesis = correction_hypothesis(record.get("source_action") or {})
        if hypothesis is not None:
            families.add(hypothesis[0])
    return families


#: Item 7 step 6 (the user's amendment of 2026-10-28): a confidence over the
#: three balanced error categories, meter (bad data), parameter (R/X) and
#: topology (breaker), from quantities the solve already produced.  The meter
#: share is the largest normalized residual against the largest normalized
#: branch multiplier, r / (r + lambda), the classical Lagrangian test; the
#: branch share splits into topology and parameter by the open-terminal
#: confidence of step 2.  The ranking uses the bands the routes already use:
#: the meter leads only outside the 1.2 dominance band (r > 1.2 lambda), the
#: open terminal leads the branch block above 0.5.  Two hypotheses compete when
#: the rule that orders them cannot separate them: meter and branch inside the
#: dominance band, parameter and topology inside step 3's 3:1 band.
HYPOTHESIS_FAMILIES = ("measurement", "parameter", "topology")
LAGRANGIAN_DOMINANCE_BAND = 1.2
HYPOTHESIS_CONFIDENCE_CONTRACT = "balanced_hypothesis_confidence_v1"


def hypothesis_ranking(residual_max: float, multiplier_max: float, open_terminal: float) -> list[str]:
    """The route order the vector implies: meter first only when meter dominant, the open terminal first above 0.5."""
    branch = ["topology", "parameter"] if open_terminal > 0.5 else ["parameter", "topology"]
    if residual_max > LAGRANGIAN_DOMINANCE_BAND * multiplier_max:
        return ["measurement", *branch]
    return [*branch, "measurement"]


def competing_hypotheses(residual_max: float, multiplier_max: float, open_terminal: float) -> list[str]:
    """The families the leading one cannot be told apart from by the rule that orders them (empty when it stands clear)."""
    leader = hypothesis_ranking(residual_max, multiplier_max, open_terminal)[0]
    if leader == "measurement":
        return []
    out = []
    if AMBIGUITY_BAND[0] <= open_terminal <= AMBIGUITY_BAND[1]:
        out.append("parameter" if leader == "topology" else "topology")
    if not multiplier_max > LAGRANGIAN_DOMINANCE_BAND * residual_max:
        out.append("measurement")
    return out


def hypothesis_confidence_vector(
    residual_max: float,
    multiplier_max: float,
    route_confidence: Mapping[str, Any] | None,
    *,
    measurement_index: int | None = None,
    parameter_line: int | None = None,
) -> dict[str, Any] | None:
    """The three-category confidence, its ranking and the competing hypotheses, from one solve (None without evidence)."""
    residual_max, multiplier_max = abs(float(residual_max)), abs(float(multiplier_max))
    total = residual_max + multiplier_max
    if not total > 0.0:
        return None
    try:
        open_terminal = float((route_confidence or {}).get("topology") or 0.0)
    except (TypeError, ValueError):
        open_terminal = 0.0
    measurement = residual_max / total
    branch = 1.0 - measurement
    topology_line = (route_confidence or {}).get("line_index")
    return {
        "contract": HYPOTHESIS_CONFIDENCE_CONTRACT,
        "measurement": round(measurement, 3),
        "parameter": round(branch * (1.0 - open_terminal), 3),
        "topology": round(branch * open_terminal, 3),
        "ranking": hypothesis_ranking(residual_max, multiplier_max, open_terminal),
        "competing": competing_hypotheses(residual_max, multiplier_max, open_terminal),
        "residual_max": round(residual_max, 2),
        "multiplier_max": round(multiplier_max, 2),
        "open_terminal_confidence": round(open_terminal, 3),
        "measurement_index": measurement_index,
        "parameter_line": parameter_line,
        "topology_line": topology_line if topology_line is not None else parameter_line,
        "breaker_refined": bool((route_confidence or {}).get("source") == "node_breaker_context"),
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
    COMPARE_PARAMETER_CANDIDATES,
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
#: Signature the balanced screen mints on a current WLS when its HIF class wins
#: (psse_env.providers.hif_screen).  It carries the HIF marker so the routing
#: helpers read it as an HIF suspicion; it is a reason to request phasors, not
#: a waveform event, and since C5 (2026-10-01) it no longer closes the balanced
#: correction routes (``blocking_waveform_signatures``).
HIF_SCREEN_SIGNATURE = "wls_hif_suspected"

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


def blocking_waveform_signatures(unresolved: Any, observation: Any = None) -> list[str]:
    """Waveform signatures that hold the correction routes shut.

    C5 (hypothesis-ranking plan, 2026-10-01): the balanced screen's HIF
    *suspicion* (``wls_hif_suspected``) is the physics rule's reason to admit
    the phasors, not a waveform event on the network.  While it is untested,
    no phasors requested on the state yet, it does not close the balanced
    routes: the ledger expert may test its leading balanced hypothesis before
    the acquisition (one verified candidate per state), and verification, not
    the gate, judges that candidate.  Once phasors were requested on the
    state the suspicion is theirs to settle, and it blocks as before until the
    refreshed WLS withdraws it or the NLM confirms the fault.  Sensor-reported
    and phasor-confirmed waveform signatures block in every case.  Without an
    ``observation`` the untested case is assumed.
    """
    signatures = waveform_anomaly_signatures(unresolved)
    if observation is not None and phasor_ledger(observation):
        return signatures
    return [item for item in signatures if not str(item).startswith(HIF_SCREEN_SIGNATURE)]


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
    Under ``suspicion_gated_diagnostics`` an alarm alone opens nothing: a
    balanced HIF suspicion mints an HIF-marked WLS signature instead, which
    routes to phasor acquisition and holds the balanced corrections until the
    phasors have been examined.
    """
    if is_suspicion_profile(evidence_profile):
        return False
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
        if not current_wls_alarm(state, history):
            return False
        family = required_suspicion(state, tool)
        if family is not None and not current_suspicion(state, family):
            return False
    return True


def current_screen_report(state: Any, family: str) -> Mapping[str, Any]:
    """The balanced screen report for ``family`` on the current, bound WLS.

    Screens run inside ``run_wls`` and ride on its ledger entry, so a report
    is current exactly when that WLS is: successful, on the active state, and
    content-bound.  Only the HIF screen exists (``hif_screen``); a harmonic
    screen would publish ``harmonic_screen`` the same way.
    """
    contexts = state.get("fresh_context_evidence") if isinstance(state, Mapping) else None
    wls = (contexts.get("wls") or {}) if isinstance(contexts, Mapping) else {}
    if not (
        isinstance(wls, Mapping)
        and wls.get("successful") is True
        and isinstance(wls.get("state_hash"), str) and wls["state_hash"]
        and str(wls.get("state_id") or "") == str(state.get("active_state_id") or "")
    ):
        return {}
    report = wls.get(f"{family}_screen")
    return report if isinstance(report, Mapping) else {}


def current_triage_report(state: Any) -> Mapping[str, Any]:
    """The triage classifier's report on the current, bound WLS of the active state, else empty.

    Under classifier_gated_diagnostics the classifier runs inside ``run_wls``
    and its report rides on the WLS ledger entry as ``triage``; it is current
    exactly when that WLS is (``current_screen_report``).
    """
    contexts = state.get("fresh_context_evidence") if isinstance(state, Mapping) else None
    wls = (contexts.get("wls") or {}) if isinstance(contexts, Mapping) else {}
    if not (
        isinstance(wls, Mapping)
        and wls.get("successful") is True
        and isinstance(wls.get("state_hash"), str) and wls["state_hash"]
        and str(wls.get("state_id") or "") == str(state.get("active_state_id") or "")
    ):
        return {}
    report = wls.get("triage")
    return report if isinstance(report, Mapping) else {}


def triage_admits_request(report: Any) -> bool:
    """A valid triage report whose request score reached its threshold."""
    return isinstance(report, Mapping) and report.get("status") == "valid" and report.get("request_admitted") is True


def triage_admits_spectra(report: Any) -> bool:
    """A valid triage report whose admitted request names the harmonic family (2026-10-13).

    The runtime sets ``spectra_admitted`` when the request is admitted and the
    largest of its waveform heads is harmonic; the spectra then open before
    the phasors, which follow only when the spectra come back clean.
    """
    return triage_admits_request(report) and report.get("spectra_admitted") is True


def triage_waveform_family(report: Any) -> str | None:
    """The waveform family a valid triage report favours (hif, unbalance or harmonic), if reported."""
    if not isinstance(report, Mapping) or report.get("status") != "valid":
        return None
    family = report.get("waveform_family")
    return str(family) if family in ("hif", "unbalance", "harmonic") else None


def triage_first_family(report: Any) -> str | None:
    """The balanced family a valid triage report says to investigate first."""
    if not isinstance(report, Mapping) or report.get("status") != "valid":
        return None
    family = report.get("first_family")
    return str(family) if family in ("measurement", "parameter", "topology") else None


def balanced_route_failed(state: Any) -> bool:
    """The classifier profile's fallback: the balanced route was tried on the active state and gave nothing.

    True when a balanced correction bound to the active state was rejected by
    verification (an executor failure tested nothing), or when every balanced
    context fetched on this state offered no correction.  Nothing in it reads
    truth: rejections and context ledgers are policy-visible.
    """
    if not isinstance(state, Mapping):
        return False
    active_id = str(state.get("active_state_id") or "")
    for record in state.get("rejected_hypotheses") or []:
        if not isinstance(record, Mapping) or record.get("rejection_kind") == "executor_failure":
            continue
        parent = record.get("candidate_parent_id")
        if parent is not None and active_id and str(parent) != active_id:
            continue
        source = record.get("source_action")
        if isinstance(source, Mapping) and safe_normalize_action(source)["tool"] in CORRECTION_TOOLS:
            return True
    fetched = [
        family for family in ("measurement", "parameter", "topology")
        if state.get(f"has_fresh_{family}_context") and str(state.get(f"{family}_context_state_id") or "") == active_id
    ]
    if not fetched:
        return False
    contexts = state.get("fresh_context_evidence")
    contexts = contexts if isinstance(contexts, Mapping) else {}
    for family in fetched:
        ledger = contexts.get(family)
        if not isinstance(ledger, Mapping) or str(ledger.get("state_id") or "") != active_id:
            return False
        if ledger.get("supported_corrections"):
            return False
    return True


def screen_phasor_suspicion(report: Any, kind: str) -> bool:
    """Read one suspicion kind off a (compact or full) screen report.

    ``hif``: an HIF won; ``voltage_meter``: a phase-A voltage channel was set
    aside as a bad meter; ``unexplained``: no hypothesis sequence explained
    the alarm; ``phasor``: any of the three.  A report that is not valid
    carries no suspicion.
    """
    if not isinstance(report, Mapping) or report.get("status") != "valid":
        return False
    kinds = report.get("phasor_suspicion")
    kinds = kinds if isinstance(kinds, Mapping) else {}
    if kind == "hif":
        return report.get("suspected") is True
    if kind == "phasor":
        return report.get("suspected") is True or any(bool(value) for value in kinds.values())
    return bool(kinds.get(kind))


def phasor_ledger(state: Any) -> Mapping[str, Any]:
    """The phasor acquisition ledger bound to the active state, else empty."""
    if not isinstance(state, Mapping):
        return {}
    contexts = state.get("fresh_context_evidence")
    context = contexts.get("three_phase") if isinstance(contexts, Mapping) else None
    if not (
        isinstance(context, Mapping)
        and str(context.get("state_id") or "") == str(state.get("active_state_id") or "")
        and context.get("request_attempted") is True
    ):
        return {}
    return context


def phasors_examined(state: Any) -> bool:
    """Phasors were acquired on the active state and the NLM has looked at them."""
    ledger = phasor_ledger(state)
    return bool(ledger) and ledger.get("nlm_attempted") is True


def phasor_nlm_classification(state: Any) -> str | None:
    """What the NLM concluded from the phasors acquired on the active state."""
    ledger = phasor_ledger(state)
    value = ledger.get("nlm_classification") if ledger else None
    return str(value) if value else None


#: NLM classifications that name a diagnosable waveform event.  Phasors that
#: named one cannot close the ladder as an unexplained discrepancy.
DIAGNOSABLE_PHASOR_CLASSIFICATIONS = frozenset({"hif_suspected", "three_phase_unbalance"})


def unexplained_handoff_supported(state: Any) -> bool:
    """Both acquisition tiers answered on the active state and named no event.

    The honest end of the suspicion-gated ladder is an unexplained
    discrepancy only when the phasors acquired and tested on this state named
    no diagnosable event and spectra were requested on this state too; the
    environment audits the handoff against exactly these records.  An alarm
    that outlives a named event (an accepted HIF estimate whose conditioned
    WLS still flags a residual, say) or whose spectra were taken on an
    ancestor state is an exhausted recovery instead.
    """
    if not phasors_examined(state):
        return False
    classification = phasor_nlm_classification(state)
    if classification is None or classification in DIAGNOSABLE_PHASOR_CLASSIFICATIONS:
        return False
    contexts = state.get("fresh_context_evidence")
    harmonic = contexts.get("harmonic") if isinstance(contexts, Mapping) else None
    # The audit needs a successful spectra request on this very state: one
    # still pending or failed, or an answer carried over a commit, is not it.
    return bool(
        isinstance(harmonic, Mapping)
        and str(harmonic.get("state_id") or "") == str(state.get("active_state_id") or "")
        and harmonic.get("request_attempted") is True
        and not harmonic.get("carried_from_state_id")
        and harmonic.get("harmonic_context_status") not in {"pending", "failed"}
    )


def _rejected_correction_targets(state: Mapping[str, Any]) -> dict[str, set[int]]:
    """Verification-rejected correction targets bound to the active state, per screen class."""
    active_id = str(state.get("active_state_id") or "")
    targets: dict[str, set[int]] = {"meter": set(), "parameter": set(), "topology": set()}
    for record in state.get("rejected_hypotheses") or []:
        if not isinstance(record, Mapping) or record.get("rejection_kind") == "executor_failure":
            continue
        parent = record.get("candidate_parent_id")
        if parent is not None and active_id and str(parent) != active_id:
            continue
        source = record.get("source_action")
        if not isinstance(source, Mapping):
            continue
        normalized = safe_normalize_action(source)
        arguments = normalized["arguments"]
        if normalized["tool"] == CORRECT_MEASUREMENTS:
            group = arguments.get("suspect_group")
            if isinstance(group, (list, tuple)):
                targets["meter"].update(int(i) for i in group if isinstance(i, int) and not isinstance(i, bool))
        elif normalized["tool"] == CORRECT_PARAMETERS:
            line = arguments.get("line_index", arguments.get("line_index1"))
            if isinstance(line, int) and not isinstance(line, bool):
                targets["parameter"].add(int(line) - 1)
            row = arguments.get("branch_row0")
            if isinstance(row, int) and not isinstance(row, bool):
                targets["parameter"].add(int(row))
    return targets


def screen_explanation_refuted(state: Any) -> bool:
    """The screen's accepted meter and branch hypotheses were all rejected by verification on this state.

    The screen explained the alarm with a sequence of balanced causes; every
    one of them that names a meter or a branch has been tried as a correction
    on the active state and rejected.  The alarm is then unexplained in fact,
    whatever the screen's own flag says, and the phasor tier opens
    (hypothesis-ranking plan, step 3).  Topology hypotheses name a branch the
    breaker correction does not, so they are not counted either way.
    """
    if not isinstance(state, Mapping):
        return False
    report = current_screen_report(state, "hif")
    if not isinstance(report, Mapping) or report.get("status") != "valid":
        return False
    accepted = [item for item in report.get("accepted_hypotheses") or []
                if isinstance(item, Mapping) and str(item.get("class")) in {"meter", "parameter"}]
    if not accepted:
        return False
    rejected = _rejected_correction_targets(state)
    for item in accepted:
        class_name = str(item["class"])
        target = item.get("channel_index0") if class_name == "meter" else item.get("branch_row0")
        if not isinstance(target, int) or isinstance(target, bool) or int(target) not in rejected[class_name]:
            return False
    return True


def requested_on_earlier_state(state: Any, tool: str) -> bool:
    """``tool`` was requested on a state of the episode other than the active one.

    The active state's own acquisitions are read from its ledger, which only
    a dispatched request creates and only a commit (which changes the active
    state) drops; a tried action on the active state with no ledger entry was
    refused by the gate and tested nothing.  An earlier state's ledger is gone
    after the commit, so its tried actions are the only record left.  A
    request that named no state is not attributed to any.
    """
    if not isinstance(state, Mapping):
        return False
    active = str(state.get("active_state_id") or "")
    for item in state.get("tried_action_signatures") or []:
        name, separator, encoded = str(item).partition(":")
        if name != tool or not separator:
            continue
        try:
            arguments = json.loads(encoded)
        except (TypeError, ValueError):
            continue
        requested = str(arguments.get("state_id") or "") if isinstance(arguments, Mapping) else ""
        if requested and requested != active:
            return True
    return False


def phasors_examined_in_episode(state: Any) -> bool:
    """Phasors were examined on this state, or earlier in the episode with no waveform event found.

    A committed meter or branch correction changes the operator's model, not
    the network, so phasors acquired and tested on an ancestor state remain
    evidence about unbalance and HIFs.  An earlier NLM run is read from the
    episode's tried actions; had it found an event, its waveform signature
    would still stand.  On the active state only the ledger counts: a refused
    request there examined nothing.
    """
    if phasors_examined(state):
        return True
    if not isinstance(state, Mapping):
        return False
    examined = requested_on_earlier_state(state, RUN_THREE_PHASE_NLM_FROM_PATH) and requested_on_earlier_state(
        state, GET_THREE_PHASE_CONTEXT
    )
    return bool(examined and not waveform_anomaly_signatures(state.get("unresolved_signatures") or []))


def phasors_tested_in_episode(state: Any) -> bool:
    """Phasors were acquired and NLM-tested on this state or on an earlier state of the episode.

    Unlike ``phasors_examined_in_episode`` this counts an earlier examination
    whatever it found: a committed meter or branch correction changes the
    operator's model, not the network, so the phasors of an ancestor state
    still show the same event.  Under classifier_gated_diagnostics a request
    the classifier admits on a corrected child state therefore does not
    acquire the phasors again (2026-10-07).
    """
    if phasors_examined(state):
        return True
    return bool(
        isinstance(state, Mapping)
        and requested_on_earlier_state(state, RUN_THREE_PHASE_NLM_FROM_PATH)
        and requested_on_earlier_state(state, GET_THREE_PHASE_CONTEXT)
    )


def spectra_examined_in_episode(state: Any) -> bool:
    """Spectra were requested on this state (its ledger) or on an earlier state of the episode."""
    if not isinstance(state, Mapping):
        return False
    contexts = state.get("fresh_context_evidence")
    harmonic = contexts.get("harmonic") if isinstance(contexts, Mapping) else None
    if (isinstance(harmonic, Mapping) and str(harmonic.get("state_id") or "") == str(state.get("active_state_id") or "")
            and harmonic.get("request_attempted") is True):
        return True
    return requested_on_earlier_state(state, GET_HARMONIC_CONTEXT)


def spectra_ledger(state: Any) -> Mapping[str, Any]:
    """The spectra request dispatched on the active state (its ``harmonic`` ledger entry), else empty."""
    if not isinstance(state, Mapping):
        return {}
    contexts = state.get("fresh_context_evidence")
    harmonic = contexts.get("harmonic") if isinstance(contexts, Mapping) else None
    if (isinstance(harmonic, Mapping) and str(harmonic.get("state_id") or "") == str(state.get("active_state_id") or "")
            and harmonic.get("request_attempted") is True):
        return harmonic
    return {}


def spectra_show_distortion(state: Any) -> bool:
    """The spectra acquired on the active state detected a distortion (the HSE rung is what follows them)."""
    return spectra_ledger(state).get("harmonic_distortion_detected") is True


def current_suspicion(state: Any, family: str) -> bool:
    """Whether the evidence on the active state points at ``family``.

    Under suspicion_gated_diagnostics this, together with the WLS alarm, is
    what admits an auxiliary stream (``DIAGNOSTIC_SUSPICION_REQUIREMENTS``):
    ``phasor`` for phase-resolved measurements (an HIF won, a phase-A voltage
    channel was set aside, or the alarm stayed unexplained), ``hif`` for the
    HIF estimator (the screen's HIF, or an HIF-like line differential the
    acquired phasors showed), ``harmonic`` for spectra (phasors acquired on
    this state came back balanced, or under classifier_gated_diagnostics the
    triage report's harmonic call, 2026-10-13).  A suspicion that the
    acquired phasors later refuted still admits the stream it already opened.
    """
    report = current_screen_report(state, "hif")
    if family == "harmonic":
        if phasor_nlm_classification(state) == "balanced_three_phase":
            return True
        if (isinstance(state, Mapping) and is_classifier_gated(state.get("evidence_profile", DEFAULT_EVIDENCE_PROFILE))
                and triage_admits_spectra(current_triage_report(state))):
            # classifier_gated_diagnostics (2026-10-13): the classifier's
            # harmonic call on the current WLS opens the spectra before any
            # phasors; the phasors follow when the spectra come back clean.
            return True
        # Phasors examined on an ancestor state found no event (its waveform
        # signature would still stand): the spectra tier is open here too.
        return bool(not phasor_ledger(state) and phasors_examined_in_episode(state))
    if isinstance(state, Mapping) and is_classifier_gated(state.get("evidence_profile", DEFAULT_EVIDENCE_PROFILE)):
        # No balanced screen runs here: the phasor tier opens on the triage
        # classifier's admitted request or on the balanced route having
        # failed on this state; the HIF suspicion is what the phasors showed.
        if family == "phasor":
            return triage_admits_request(current_triage_report(state)) or balanced_route_failed(state)
        if family == "hif":
            return phasor_nlm_classification(state) == "hif_suspected"
        return False
    if family == "hif":
        return screen_phasor_suspicion(report, "hif") or phasor_nlm_classification(state) == "hif_suspected"
    if family == "refuted_explanation":
        return screen_explanation_refuted(state)
    if family == "phasor":
        return screen_phasor_suspicion(report, "phasor") or screen_explanation_refuted(state)
    return screen_phasor_suspicion(report, family)


def hif_suspicion_refuted(state: Any) -> bool:
    """The current HIF suspicion was tested on phasors and found no HIF."""
    report = current_screen_report(state, "hif")
    return screen_phasor_suspicion(report, "hif") and report.get("refuted_by_phase_measurements") is True


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


#: Compound-error study, stage 2 item 5 (the user's decision of 2026-10-28,
#: reading C): a handoff may carry explicit abstentions, "not identifiable:
#: line N isolated by breaker M", as a structured ``abstentions`` argument.
#: They are recorded by the environment and reported by the evaluator and
#: the truth audit; no verdict depends on them.
ISOLATED_LINE_ABSTENTION = "line_isolated_by_breaker"
_OPEN_STATUS_TEXT = frozenset({"0", "open", "false"})


def isolated_line_abstentions(observation: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    """The lines that the accepted breaker corrections of the active lineage left isolated, as abstentions.

    An accepted ``correct_topology`` that opened a breaker on a single line
    terminal (``status`` open with a ``line_index``) de-energised that line,
    so its impedance is not identifiable from the snapshot; the statement
    names the line and the breaker.  Read from the policy observation's
    ``accepted_corrections`` only.
    """
    getter = getattr(observation, "get", None)
    accepted = getter("accepted_corrections", []) if callable(getter) else []
    out: dict[tuple[int, str | None], dict[str, Any]] = {}
    for item in accepted or []:
        source = item.get("source_action") if isinstance(item, Mapping) else None
        if source is None:
            continue
        try:
            normalized = normalize_action(source)
        except (TypeError, ValueError):
            continue
        if normalized["tool"] != CORRECT_TOPOLOGY:
            continue
        arguments = normalized["arguments"]
        line = arguments.get("line_index")
        if line is None and arguments.get("line_index1") is not None:
            line = arguments.get("line_index1")
        if isinstance(line, bool) or not isinstance(line, int):
            continue
        if str(arguments.get("status")).strip().lower() not in _OPEN_STATUS_TEXT:
            continue
        cb_name = arguments.get("cb_name")
        key = (int(line), str(cb_name) if cb_name is not None else None)
        out.setdefault(key, {"kind": ISOLATED_LINE_ABSTENTION, "line_index": int(line), "cb_name": key[1]})
    return [out[key] for key in sorted(out, key=lambda k: (k[0], k[1] or ""))]


def with_handoff_statements(action: Mapping[str, Any], statements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """An operator handoff with ``statements`` appended to its structured ``abstentions`` field; anything else unchanged."""
    try:
        normalized = normalize_action(action)
    except (TypeError, ValueError):
        return dict(action)
    request = normalized["arguments"].get("request")
    if normalized["tool"] != ASK_FOR_MORE_EVIDENCE or not str(request or "").startswith("operator_escalation:"):
        return normalized
    if not statements:
        return normalized
    arguments = dict(normalized["arguments"])
    existing = arguments.get("abstentions")
    arguments["abstentions"] = [*(existing if isinstance(existing, list) else []), *[dict(item) for item in statements]]
    return {"tool": normalized["tool"], "arguments": arguments}


def with_isolated_line_abstentions(action: Mapping[str, Any], observation: Mapping[str, Any] | Any) -> dict[str, Any]:
    """The handoff action with its abstention statements attached; any other action, or none to declare, unchanged."""
    return with_handoff_statements(action, isolated_line_abstentions(observation))


#: Item 7 step 3 (2026-10-28): when the teacher hands off with the alarm still
#: standing and the solve cannot clearly tell an impedance error from an open
#: terminal on the leading line (route confidence between 0.25 and 0.75, so
#: neither hypothesis has three-to-one support), the handoff states
#: "ambiguous: line N, parameter or breaker" with the confidence, in the same
#: structured field.  Recorded and reported; no verdict depends on it.
AMBIGUOUS_ROUTE_STATEMENT = "ambiguous_parameter_or_breaker"
AMBIGUITY_BAND = (0.25, 0.75)


def _route_confidence_of(observation: Mapping[str, Any] | Any) -> tuple[Mapping[str, Any] | None, bool]:
    """The latest bound route confidence (the topology context's refinement when it names the same line) and whether the solve alarms."""
    getter = getattr(observation, "get", None)
    contexts = getter("fresh_context_evidence", {}) if callable(getter) else {}
    contexts = contexts if isinstance(contexts, Mapping) else {}
    wls = contexts.get("wls")
    if not isinstance(wls, Mapping) or not wls.get("successful"):
        return None, False
    record = wls.get("route_confidence")
    record = record if isinstance(record, Mapping) else None
    topology = contexts.get("topology")
    if record is not None and isinstance(topology, Mapping) and str(topology.get("state_id") or "") == str(wls.get("state_id") or ""):
        refined = topology.get("route_confidence")
        if isinstance(refined, Mapping) and refined.get("line_index") == record.get("line_index"):
            record = refined
    return record, bool(wls.get("anomalous"))


def _routes_attempted(observation: Mapping[str, Any] | Any, line: int) -> list[str]:
    """The hypotheses the episode's corrections tried on ``line`` (committed or rejected), from the observation."""
    getter = getattr(observation, "get", None)
    tried: set[str] = set()
    for field in ("accepted_corrections", "rejected_hypotheses"):
        for item in (getter(field, []) if callable(getter) else []) or []:
            source = item.get("source_action") if isinstance(item, Mapping) else None
            if source is None:
                continue
            try:
                normalized = normalize_action(source)
            except (TypeError, ValueError):
                continue
            target = normalized["arguments"].get("line_index", normalized["arguments"].get("line_index1"))
            if target != line:
                continue
            if normalized["tool"] == CORRECT_PARAMETERS:
                tried.add("parameter")
            elif normalized["tool"] == CORRECT_TOPOLOGY:
                tried.add("topology")
    return sorted(tried)


#: Item 7 step 6: with the three-category confidence on the solve, the
#: statement lists the leading hypothesis and every one competing with it,
#: each with its confidence and target, instead of the parameter-or-breaker
#: pair alone.  Recorded and reported; no verdict depends on it.
AMBIGUOUS_HYPOTHESES_STATEMENT = "ambiguous_hypotheses"
AMBIGUITY_STATEMENT_KINDS = frozenset({AMBIGUOUS_ROUTE_STATEMENT, AMBIGUOUS_HYPOTHESES_STATEMENT})


def hypothesis_vector_of(observation: Mapping[str, Any] | Any) -> tuple[Mapping[str, Any] | None, bool]:
    """The latest bound three-category confidence (the topology context's refinement when bound to the same state) and whether the solve alarms."""
    getter = getattr(observation, "get", None)
    contexts = getter("fresh_context_evidence", {}) if callable(getter) else {}
    contexts = contexts if isinstance(contexts, Mapping) else {}
    wls = contexts.get("wls")
    if not isinstance(wls, Mapping) or not wls.get("successful"):
        return None, False
    record = wls.get("hypothesis_confidence")
    record = record if isinstance(record, Mapping) else None
    topology = contexts.get("topology")
    if record is not None and isinstance(topology, Mapping) and str(topology.get("state_id") or "") == str(wls.get("state_id") or ""):
        refined = topology.get("hypothesis_confidence")
        if isinstance(refined, Mapping):
            record = refined
    return record, bool(wls.get("anomalous"))


def _hypothesis_target(vector: Mapping[str, Any], family: str) -> dict[str, Any]:
    if family == "measurement":
        return {"measurement_index": vector.get("measurement_index")}
    return {"line_index": vector.get("topology_line" if family == "topology" else "parameter_line")}


def _families_attempted(observation: Mapping[str, Any] | Any) -> list[str]:
    """The families the active lineage committed or the active state rejected, from the observation."""
    getter = getattr(observation, "get", None)
    tried: set[str] = set()
    for field in ("accepted_corrections", "rejected_hypotheses"):
        for item in (getter(field, []) if callable(getter) else []) or []:
            source = item.get("source_action") if isinstance(item, Mapping) else None
            hypothesis = correction_hypothesis(source) if source else None
            if hypothesis is not None:
                tried.add(hypothesis[0])
    return sorted(tried)


def hypothesis_statements(observation: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    """The competing hypotheses, with their confidences, that the latest solve supports at a handoff, else none."""
    vector, anomalous = hypothesis_vector_of(observation)
    if vector is None or not anomalous or not vector.get("competing"):
        return []
    ranking = [str(family) for family in vector.get("ranking") or []]
    if not ranking:
        return []
    families = [ranking[0], *[str(family) for family in vector["competing"]]]
    return [{
        "kind": AMBIGUOUS_HYPOTHESES_STATEMENT,
        "hypotheses": [
            {"family": family, "confidence": vector.get(family), **_hypothesis_target(vector, family)}
            for family in families
        ],
        "ranking": ranking,
        "breaker_refined": bool(vector.get("breaker_refined")),
        "attempted": _families_attempted(observation),
        "withdrawn": sorted(withdrawn_families(observation)),
    }]


def ambiguity_statements(observation: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    """The ambiguity statement the latest solve supports at a handoff, else none.

    With the three-category confidence on the solve (item 7 step 6), the
    competing hypotheses; otherwise step 3's "ambiguous: line N, parameter
    or breaker".
    """
    vector, _ = hypothesis_vector_of(observation)
    if vector is not None:
        return hypothesis_statements(observation)
    record, anomalous = _route_confidence_of(observation)
    if record is None or not anomalous or record.get("line_index") is None:
        return []
    try:
        topology = float(record["topology"])
        line = int(record["line_index"])
    except (KeyError, TypeError, ValueError):
        return []
    if not AMBIGUITY_BAND[0] <= topology <= AMBIGUITY_BAND[1]:
        return []
    return [{
        "kind": AMBIGUOUS_ROUTE_STATEMENT,
        "line_index": line,
        "topology_confidence": round(topology, 3),
        "parameter_confidence": round(1.0 - topology, 3),
        "basis": record.get("basis"),
        "breaker_evidence": record.get("breaker_evidence"),
        "attempted": _routes_attempted(observation, line),
    }]


def with_ambiguity_statements(action: Mapping[str, Any], observation: Mapping[str, Any] | Any) -> dict[str, Any]:
    """The handoff action with its ambiguity statement attached (item 7 step 3); any other action unchanged."""
    return with_handoff_statements(action, ambiguity_statements(observation))


def abstention_record(declared: Any, observation: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    """The declared statements of a handoff, each marked supported by what the observation shows.

    An isolated-line abstention is supported when an accepted breaker
    correction isolated that line; an ambiguity statement (item 7 step 3)
    when the latest solve's route confidence names the same line inside the
    ambiguity band.
    """
    supported = {(item["line_index"], item["cb_name"]) for item in isolated_line_abstentions(observation)}
    supported_lines = {line for line, _ in supported}
    current = ambiguity_statements(observation)
    ambiguous_lines = {item["line_index"] for item in current if item.get("kind") == AMBIGUOUS_ROUTE_STATEMENT}
    competing_sets = [
        sorted(str(h.get("family")) for h in item.get("hypotheses") or [])
        for item in current if item.get("kind") == AMBIGUOUS_HYPOTHESES_STATEMENT
    ]
    records: list[dict[str, Any]] = []
    for item in declared if isinstance(declared, (list, tuple)) else []:
        if not isinstance(item, Mapping):
            records.append({"kind": None, "line_index": None, "cb_name": None, "supported": False, "malformed": True})
            continue
        if str(item.get("kind")) == AMBIGUOUS_HYPOTHESES_STATEMENT:
            hypotheses = [dict(h) for h in item.get("hypotheses") or [] if isinstance(h, Mapping)]
            records.append({
                "kind": AMBIGUOUS_HYPOTHESES_STATEMENT,
                "hypotheses": hypotheses,
                "ranking": list(item.get("ranking") or []),
                "attempted": list(item.get("attempted") or []),
                "withdrawn": list(item.get("withdrawn") or []),
                "supported": sorted(str(h.get("family")) for h in hypotheses) in competing_sets,
            })
            continue
        if str(item.get("kind")) == AMBIGUOUS_ROUTE_STATEMENT:
            line = item.get("line_index")
            valid_line = isinstance(line, int) and not isinstance(line, bool)
            records.append({
                "kind": AMBIGUOUS_ROUTE_STATEMENT,
                "line_index": int(line) if valid_line else None,
                "topology_confidence": item.get("topology_confidence"),
                "parameter_confidence": item.get("parameter_confidence"),
                "basis": item.get("basis"),
                "attempted": list(item.get("attempted") or []),
                "supported": bool(valid_line and int(line) in ambiguous_lines),
            })
            continue
        line = item.get("line_index")
        cb_name = item.get("cb_name")
        kind = item.get("kind")
        ok = (
            str(kind) == ISOLATED_LINE_ABSTENTION
            and isinstance(line, int) and not isinstance(line, bool)
            and ((int(line), str(cb_name)) in supported if cb_name is not None else int(line) in supported_lines)
        )
        records.append({
            "kind": str(kind) if kind is not None else None,
            "line_index": int(line) if isinstance(line, int) and not isinstance(line, bool) else None,
            "cb_name": str(cb_name) if cb_name is not None else None,
            "supported": bool(ok),
        })
    return records


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
