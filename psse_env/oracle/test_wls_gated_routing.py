"""Expert routing under ``wls_gated_diagnostics`` (the research default).

The contract: detection uses only balanced SCADA and its WLS (chi-square at
alpha 0.01 OR max normalized residual >= 4.0).  After a current alarm on the
active state the expert may request the auxiliary streams and run the
matching diagnostics; ``unavailable`` answers are normal and the balanced
ladder continues.  No seeded flag, hint or precomputed diagnosis is used, the
alternative test and any learned screen stay blocked, and only supported
operator requests are ever proposed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, CORRECT_MEASUREMENTS, DIAGNOSTIC_TOOLS, ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH, FINALIZE_DIAGNOSIS, GET_HARMONIC_CONTEXT,
    GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT, GET_THREE_PHASE_CONTEXT,
    GET_TOPOLOGY_CONTEXT, HIF_CONDITIONING_UNAVAILABLE_REQUEST,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST, RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_ALTERNATIVE_TEST, RUN_HSE_FROM_PATH, RUN_THREE_PHASE_NLM_FROM_PATH, RUN_WLS,
    action_signature, current_wls_alarm, diagnostic_tool_permitted,
    harmonic_screening_pending, preferred_first_request, three_phase_acquisition_pending,
    three_phase_screening_pending,
)
from psse_env.evidence_profile import (
    AUXILIARY_EVIDENCE_PROFILE, DEFAULT_EVIDENCE_PROFILE, GATED_DIAGNOSTIC_TOOLS,
    SCADA_ONLY_PROFILE, WLS_GATED_PROFILE, disabled_requests, disabled_tools,
    is_strict_boundary, is_wls_gated,
)
from psse_env.oracle import DiagnosticsExpert, ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.termination_expert import TerminationExpert

ACTIVE = "gated:s0"
HASH = "content-current"
WLS_SIGNATURES = [
    "wls_residual_outlier_dominant index=26 channel=Pinj",
    "wls_branch_multiplier line_status_or_parameter line=4",
]
MINTED_HIF = "hif_suspected_line_differential"
MINTED_UNBALANCE = "three_phase_unbalance localized_by_diagnostic"
MINTED_HARMONIC = "harmonic distortion_detected_by_context"
PHASE_CHANNELS = ["three_phase_voltages", "three_phase_branch_currents"]
SUPPORTED_REQUESTS = {
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST, HIF_CONDITIONING_UNAVAILABLE_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST, "operator_escalation:recovery_budget_exhausted",
    "operator_escalation:ambiguous_branch_candidates",
}


def _wls(*, solved=True, alarm=True, breadth=0.05, chi=None, residual=None):
    if not solved:
        return {"successful": False}
    chi = alarm if chi is None else chi
    residual = alarm if residual is None else residual
    return {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_wls:test",
        "successful": True, "anomalous": alarm, "anomaly_breadth": breadth,
        "chi_square_alarm": chi, "normalized_residual_alarm": residual,
        "normalized_residual_threshold": 4.0, "max_normalized_residual": 8.0 if residual else 2.0,
    }


def _state(*, solved=True, alarm=True, signatures=None, profile=WLS_GATED_PROFILE, **wls_fields):
    """A compact policy observation after (or before) the opening WLS."""
    if signatures is None:
        signatures = list(WLS_SIGNATURES) if solved and alarm else []
    return {
        "evidence_profile": profile, "active_state_id": ACTIVE, "has_open_candidate": False,
        "remaining_budget": 40,
        "remaining_anomaly_score": (2.0 if alarm else 0.5) if solved else None,
        "no_material_anomaly_remaining": solved and not alarm,
        "unresolved_signatures": list(signatures), "explained_anomalies": [],
        "accepted_corrections": [], "rejected_hypotheses": [], "available_evidence": [],
        "last_tool": RUN_WLS if solved else None, "last_tool_status": "success" if solved else None,
        "tried_action_signatures": (
            [action_signature({"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}})] if solved else []
        ),
        "fresh_context_evidence": {"wls": _wls(solved=solved, alarm=alarm, **wls_fields)},
    }


def _telemetry(state, family, *, available, channels=(), nlm_attempted=None, **extra):
    """Record an answered auxiliary request on the active state."""
    record = {
        "state_id": ACTIVE, "state_hash": HASH,
        "evidence_source": f"deployment_context:{family}_measurements",
        "request_attempted": True,
        f"{family}_context_status": "available" if available else "unavailable",
        "available_evidence_channels": list(channels),
    }
    if family == "harmonic":
        record["harmonic_distortion_detected"] = bool(available)
    if nlm_attempted is not None:
        record["nlm_attempted"] = nlm_attempted
    record.update(extra)
    state["fresh_context_evidence"][family] = record
    if available:
        state["available_evidence"] = sorted(set(state["available_evidence"]) | set(channels))


def _context(state, family, targets=()):
    tool = {"measurement": "correct_measurements", "parameter": "correct_parameters",
            "topology": "correct_topology"}[family]
    state[f"has_fresh_{family}_context"] = True
    state[f"{family}_context_state_id"] = ACTIVE
    state["fresh_context_evidence"][family] = {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": f"deployment_context:{family}",
        "route_status": "actionable" if targets else "complete_negative",
        "supported_corrections": [
            {"tool": tool, "arguments": {"state_id": ACTIVE, **target}} for target in targets
        ],
    }


def _nlm_step(summary, *, accepted=None, explanation=None):
    metrics = {"state_id": ACTIVE, "state_hash": HASH, "nlm_summary": summary,
               "evidence_source": "deployment_diagnostic:three_phase_nlm"}
    if accepted is not None:
        metrics["diagnostic_acceptance"] = {"accepted": accepted}
    if explanation is not None:
        metrics["anomaly_explanation"] = explanation
    return {"action": {"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": ACTIVE}},
            "tool_output": {"execution_status": "success", "tool_metrics": metrics}}


def _estimator_step(tool, *, accepted):
    return {"action": {"tool": tool, "arguments": {"state_id": ACTIVE, "candidate_branch_row0": 3}},
            "tool_output": {"execution_status": "success", "tool_metrics": {
                "state_id": ACTIVE, "state_hash": HASH, "diagnostic_acceptance": {"accepted": accepted},
                "evidence_source": "deployment_diagnostic:hif"}}}


def _oracle():
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))


def _first(state, history=None):
    actions = _oracle().next_actions(state, history or [])
    assert actions, state
    return actions[0]


def _assert_no_forbidden(state, history=None):
    """Nothing disabled, and nothing gated before the alarm, is ever labelled."""
    for proposal in _oracle().next_action_proposals(state, history or []):
        tool = proposal.action["tool"]
        assert tool not in disabled_tools(state), proposal
        assert tool != RUN_ALTERNATIVE_TEST, proposal
        request = proposal.action["arguments"].get("request")
        if tool == ASK_FOR_MORE_EVIDENCE and request is not None:
            assert request in SUPPORTED_REQUESTS, proposal
            assert request not in disabled_requests(state), proposal
        if tool in GATED_DIAGNOSTIC_TOOLS:
            assert current_wls_alarm(state), proposal


# --------------------------------------------------------------------------- profile


def test_default_profile_is_wls_gated_and_strict():
    assert DEFAULT_EVIDENCE_PROFILE == WLS_GATED_PROFILE
    assert is_wls_gated(_state()) and is_strict_boundary(_state())
    assert disabled_tools(_state()) == frozenset({RUN_ALTERNATIVE_TEST})
    assert disabled_requests(_state()) == frozenset()
    state = _state()
    state.pop("evidence_profile")
    assert is_wls_gated(state)


# ------------------------------------------------------------------ WLS-first opening


@pytest.mark.parametrize("fields", [
    {},
    {"unresolved_signatures": ["hif_suspected_zero_sequence"], "available_evidence": ["hif_scan_window", "nlm_diagnostic"]},
    {"unresolved_signatures": [MINTED_HIF], "available_evidence": PHASE_CHANNELS},
    {"unresolved_signatures": ["harmonic_distortion_detected"], "available_evidence": ["harmonic_measurements"]},
    {"unresolved_signatures": ["three_phase_unbalance vuf_threshold_exceeded"], "available_evidence": PHASE_CHANNELS},
    {"unresolved_signatures": ["wls_branch_multiplier_dominant line_status_or_parameter line=3"]},
    {"no_material_anomaly_remaining": True, "remaining_anomaly_score": 0.0, "oracle_terminal_eligible": True},
    {"explained_anomalies": [{"family": "hif", "explained_signatures": ["hif_suspected_zero_sequence"]}],
     "unresolved_signatures": ["hif_suspected_zero_sequence"]},
])
def test_every_family_opens_with_wls_and_nothing_gated_before_it(fields):
    state = _state(solved=False)
    state.update(fields)
    assert _first(state) == {"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}}
    assert not current_wls_alarm(state)
    for tool in sorted(GATED_DIAGNOSTIC_TOOLS):
        assert not diagnostic_tool_permitted(state, tool)
    _assert_no_forbidden(state)
    assert TerminationExpert().propose(state, []) == []


def test_unsuccessful_or_unbound_wls_is_not_a_current_alarm():
    for mutate in ("failed", "old_state", "missing_hash", "missing"):
        state = _state()
        wls = state["fresh_context_evidence"]["wls"]
        if mutate == "failed":
            wls["successful"] = False
        elif mutate == "old_state":
            wls["state_id"] = "previous:s0"
        elif mutate == "missing_hash":
            wls.pop("state_hash")
        else:
            state["fresh_context_evidence"].pop("wls")
        assert not current_wls_alarm(state), mutate
        assert not diagnostic_tool_permitted(state, GET_THREE_PHASE_CONTEXT), mutate
        assert _first(state)["tool"] == RUN_WLS, mutate
        _assert_no_forbidden(state)


# ------------------------------------------------------------ the alarm gate itself


def test_quiet_wls_never_requests_auxiliary_streams():
    state = _state(alarm=False)
    assert not current_wls_alarm(state)
    assert _first(state) == {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
    _assert_no_forbidden(state)
    # Residual signatures below the alarm thresholds do not open the gate.
    state = _state(alarm=False, signatures=["wls_residual_outlier index=26 channel=Pinj"])
    state.update(remaining_anomaly_score=0.9, no_material_anomaly_remaining=False)
    assert not current_wls_alarm(state)
    for stage in (DiagnosticsExpert().three_phase_screening_proposals, DiagnosticsExpert().harmonic_screening_proposals):
        assert stage(state, []) == []
    assert _first(state)["tool"] not in GATED_DIAGNOSTIC_TOOLS
    _assert_no_forbidden(state)


@pytest.mark.parametrize("chi,residual", [(True, False), (False, True), (True, True)])
def test_either_alarm_flag_opens_the_three_phase_request(chi, residual):
    state = _state(chi=chi, residual=residual)
    assert current_wls_alarm(state)
    for tool in sorted(GATED_DIAGNOSTIC_TOOLS):
        assert diagnostic_tool_permitted(state, tool)
    assert _first(state) == {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _assert_no_forbidden(state)


def test_chi_square_only_alarm_without_wls_signatures_still_discovers():
    """A chi-square alarm may mint no wls_* signature; the flags drive routing."""
    state = _state(chi=True, residual=False, signatures=[])
    arguments = {"unresolved": [], "tried_action_signatures": [], "active_state_id": ACTIVE,
                 "context_evidence": state["fresh_context_evidence"], "evidence_profile": WLS_GATED_PROFILE}
    assert three_phase_acquisition_pending(**arguments)
    assert harmonic_screening_pending(**arguments)
    assert not three_phase_acquisition_pending(**{**arguments, "evidence_profile": AUXILIARY_EVIDENCE_PROFILE})
    assert _first(state) == {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": ACTIVE}}


def test_three_phase_is_requested_first_whatever_the_breadth_says_when_absent():
    assert preferred_first_request(_state(breadth=None)) == GET_THREE_PHASE_CONTEXT
    assert preferred_first_request(_state(breadth=0.27)) == GET_THREE_PHASE_CONTEXT
    assert preferred_first_request(_state(breadth=None, profile=AUXILIARY_EVIDENCE_PROFILE)) == GET_HARMONIC_CONTEXT
    assert preferred_first_request(_state(profile=SCADA_ONLY_PROFILE)) == GET_MEASUREMENT_CONTEXT


# ------------------------------------------------- unavailable -> harmonic -> balanced


def test_unavailable_phasors_then_spectra_then_balanced_ladder():
    state = _state()
    assert _first(state)["tool"] == GET_THREE_PHASE_CONTEXT
    _telemetry(state, "three_phase", available=False)
    assert _first(state) == {"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _telemetry(state, "harmonic", available=False)
    assert _first(state) == {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _assert_no_forbidden(state)
    _context(state, "measurement", [{"suspect_group": [26]}])
    expected = {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [26]}}
    assert _first(state) == expected
    assert _oracle().process_oracle.check(state, expected)["process_valid"]


@pytest.mark.parametrize("family,target", [("parameter", {"line_index": 3}), ("topology", {"line_index": 3, "status": 0})])
def test_balanced_branch_routes_after_both_streams_are_unavailable(family, target):
    state = _state(signatures=["wls_branch_multiplier_dominant line_status_or_parameter line=3"])
    _telemetry(state, "three_phase", available=False)
    _telemetry(state, "harmonic", available=False)
    assert _first(state)["tool"] in {GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT}
    _context(state, family, [target])
    action = _first(state)
    assert action == {"tool": f"correct_{'parameters' if family == 'parameter' else 'topology'}",
                      "arguments": {"state_id": ACTIVE, **target}}
    assert _oracle().process_oracle.check(state, action)["process_valid"]


def test_balanced_ladder_exhaustion_hands_off_with_a_supported_request_only():
    state = _state(signatures=["wls_model_discrepancy"])
    _telemetry(state, "three_phase", available=False)
    _telemetry(state, "harmonic", available=False)
    for family in ("measurement", "parameter", "topology"):
        _context(state, family)
    proposal = _oracle().next_action_proposals(state, [])[0]
    assert proposal.action == {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {
        "state_id": ACTIVE, "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST}}
    assert "unresolved_balanced_model_discrepancy_requires_operator_handoff" in proposal.evidence_codes
    assert not any("hif" in code for code in proposal.evidence_codes)


# ------------------------------------------------------- three-phase available -> NLM


def _screened_state(**overrides):
    state = _state(**overrides)
    _telemetry(state, "three_phase", available=True, channels=PHASE_CHANNELS, nlm_attempted=False)
    return state


def test_acquired_phasors_are_screened_by_nlm_before_any_correction():
    state = _screened_state()
    assert three_phase_screening_pending(
        unresolved=state["unresolved_signatures"], available_evidence=state["available_evidence"],
        tried_action_signatures=[], active_state_id=ACTIVE,
        context_evidence=state["fresh_context_evidence"], evidence_profile=WLS_GATED_PROFILE)
    assert _first(state) == {"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": ACTIVE}}
    _context(state, "measurement", [{"suspect_group": [26]}])
    assert _first(state)["tool"] == RUN_THREE_PHASE_NLM_FROM_PATH
    _assert_no_forbidden(state)


def test_balanced_nlm_result_returns_to_the_balanced_ladder():
    state = _screened_state()
    state["fresh_context_evidence"]["three_phase"]["nlm_attempted"] = True
    history = [_nlm_step({"screening_mode": True, "diagnostic_classification": "balanced_three_phase",
                          "top_hif_groups": []}, accepted=False)]
    # A balanced screen is a negative answer from the phasors: the spectra are
    # asked for once (the gate holds corrections until then), then the
    # balanced ladder resumes.
    assert _first(state, history) == {"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _telemetry(state, "harmonic", available=False)
    assert _first(state, history) == {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _assert_no_forbidden(state, history)


def test_unbalance_explanation_finalizes_without_any_correction():
    state = _screened_state()
    state["fresh_context_evidence"]["three_phase"]["nlm_attempted"] = True
    state["unresolved_signatures"].append(MINTED_UNBALANCE)
    state["explained_anomalies"] = [{
        "family": "three_phase_unbalance", "tool": RUN_THREE_PHASE_NLM_FROM_PATH, "state_id": ACTIVE,
        "explained_signatures": [MINTED_UNBALANCE, *WLS_SIGNATURES], "minted_signature": MINTED_UNBALANCE,
        "detail": {"bus_1based": 9},
    }]
    assert _first(state) == {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
    assert TerminationExpert().propose(state, [])[0].evidence_codes[0] == "anomalies_explained_by_diagnostics"
    # The same explanation closes nothing under scada_only, where it cannot exist.
    assert TerminationExpert().propose({**state, "evidence_profile": SCADA_ONLY_PROFILE}, []) == []


# ------------------------------------------------------------------ the HIF ladder


def _hif_suspected_state(*, scan_window=True):
    state = _screened_state()
    state["fresh_context_evidence"]["three_phase"]["nlm_attempted"] = True
    state["unresolved_signatures"].append(MINTED_HIF)
    if scan_window:
        state["available_evidence"] = sorted(set(state["available_evidence"]) | {"hif_scan_window"})
    return state


NLM_HIF = _nlm_step({"screening_mode": True, "diagnostic_classification": "hif_suspected",
                     "top_hif_groups": [{"branch_row0": 3}], "suspected_phase": "B"}, accepted=False)


def test_minted_hif_signature_routes_to_multiscan_then_single_scan_then_handoff():
    state = _hif_suspected_state()
    first = _first(state, [NLM_HIF])
    assert first == {"tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH, "arguments": {
        "state_id": ACTIVE, "candidate_branch_row0": 3, "candidate_phase": "B"}}
    history = [NLM_HIF, _estimator_step(ESTIMATE_HIF_MULTISCAN_FROM_PATH, accepted=False)]
    assert _first(state, history)["tool"] == ESTIMATE_HIF_FROM_PATH
    history.append(_estimator_step(ESTIMATE_HIF_FROM_PATH, accepted=False))
    handoff = _first(state, history)
    assert handoff == {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {
        "state_id": ACTIVE, "request": HIF_DIAGNOSTICS_EXHAUSTED_REQUEST}}
    assert HIF_DIAGNOSTICS_EXHAUSTED_REQUEST not in disabled_requests(state)
    for events in ([NLM_HIF], history):
        _assert_no_forbidden(state, events)
        assert not any(action["tool"] in {GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS}
                       for action in _oracle().next_actions(state, events))


def test_without_a_scan_window_the_single_scan_estimator_is_the_ladder():
    state = _hif_suspected_state(scan_window=False)
    assert _first(state, [NLM_HIF])["tool"] == ESTIMATE_HIF_FROM_PATH
    history = [NLM_HIF, _estimator_step(ESTIMATE_HIF_FROM_PATH, accepted=False)]
    assert _first(state, history)["arguments"]["request"] == HIF_DIAGNOSTICS_EXHAUSTED_REQUEST


def test_hif_ladder_waits_for_the_alarm_after_a_commit_drops_the_ledger():
    """After a commit the phasors are re-acquired only behind a fresh alarm."""
    state = _hif_suspected_state()
    state["fresh_context_evidence"]["wls"] = {"successful": False}
    assert _first(state, [NLM_HIF]) == {"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}}
    _assert_no_forbidden(state, [NLM_HIF])
    state["fresh_context_evidence"]["wls"] = _wls(alarm=False)
    assert not diagnostic_tool_permitted(state, ESTIMATE_HIF_MULTISCAN_FROM_PATH)
    assert all(action["tool"] not in GATED_DIAGNOSTIC_TOOLS for action in _oracle().next_actions(state, [NLM_HIF]))


def _conditioning(*, candidates=(), status="ready"):
    return {"status": status, "state_id": ACTIVE, "state_hash": HASH,
            "method": "paired_opendss_effect_compensation", "physical_fault_still_present": True,
            "remaining_meter_candidate_indices": list(candidates),
            "failure_reasons": [] if status == "ready" else ["prediction_envelope_unavailable"]}


def _accepted_hif_state(*, alarm, conditioning=None):
    state = _hif_suspected_state()
    state["fresh_context_evidence"]["wls"] = _wls(alarm=alarm)
    state["remaining_anomaly_score"] = 2.0 if alarm else 0.5
    state["no_material_anomaly_remaining"] = not alarm
    if not alarm:
        state["unresolved_signatures"] = [MINTED_HIF]
    state["explained_anomalies"] = [{"family": "hif", "tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH,
        "state_id": ACTIVE, "explained_signatures": [MINTED_HIF],
        "detail": {"conditioning_fit": {"success": True}, "candidate_branch_row0": 3}}]
    if conditioning is not None:
        state["fresh_context_evidence"]["hif_conditioning"] = deepcopy(conditioning)
    return state


def test_accepted_hif_runs_the_conditioned_wls_then_finalizes_when_quiet():
    state = _accepted_hif_state(alarm=False)
    state["fresh_context_evidence"]["wls"]["successful"] = True
    # No conditioning record yet: the fit must be checked against current sensors.
    assert _first(state) == {"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}}
    state = _accepted_hif_state(alarm=False, conditioning=_conditioning())
    assert _first(state) == {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
    assert _oracle().process_oracle.check(state, {"tool": FINALIZE_DIAGNOSIS, "arguments": {}})["process_valid"]


def test_accepted_hif_with_remaining_meter_candidates_repairs_the_meter():
    state = _accepted_hif_state(alarm=True, conditioning=_conditioning(candidates=[26]))
    assert _first(state) == {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _context(state, "measurement", [{"suspect_group": [26]}])
    state["fresh_context_evidence"]["measurement"]["state_hash"] = HASH
    assert _first(state) == {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [26]}}
    _assert_no_forbidden(state)


def test_unavailable_conditioning_hands_off_with_the_supported_hif_request():
    state = _accepted_hif_state(alarm=False, conditioning=_conditioning(status="unavailable"))
    assert _first(state) == {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {
        "state_id": ACTIVE, "request": HIF_CONDITIONING_UNAVAILABLE_REQUEST}}
    assert HIF_CONDITIONING_UNAVAILABLE_REQUEST not in disabled_requests(state)
    # scada_only can neither hold the explanation nor make this request.
    strict = {**deepcopy(state), "evidence_profile": SCADA_ONLY_PROFILE}
    assert DiagnosticsExpert().hif_continuation_proposals(strict) == []


def test_post_commit_meter_repair_hands_off_with_recovery_options_exhausted():
    from psse_env.actions import POST_CORRECTION_CONFIRMATION_SIGNATURE

    state = _accepted_hif_state(alarm=False, conditioning=_conditioning())
    state["unresolved_signatures"].append(POST_CORRECTION_CONFIRMATION_SIGNATURE)
    state["accepted_corrections"] = [{"candidate_state_id": ACTIVE, "source_action": {
        "tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": "gated:previous", "suspect_group": [26]}}}]
    _context(state, "measurement")
    state["fresh_context_evidence"]["measurement"]["state_hash"] = HASH
    assert _first(state) == {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {
        "state_id": ACTIVE, "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST}}


# ------------------------------------------------------------------ harmonic roots


def test_harmonic_root_runs_hse_after_phasors_are_unavailable_then_finalizes():
    state = _state(breadth=0.27)
    assert _first(state)["tool"] == GET_THREE_PHASE_CONTEXT
    _telemetry(state, "three_phase", available=False)
    assert _first(state)["tool"] == GET_HARMONIC_CONTEXT
    _telemetry(state, "harmonic", available=True, channels=["harmonic_measurements"])
    state["unresolved_signatures"].append(MINTED_HARMONIC)
    history = [{"action": {"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": ACTIVE}},
                "tool_output": {"execution_status": "success", "tool_metrics": {"harmonic_context_status": "available"}}}]
    assert _first(state, history) == {"tool": RUN_HSE_FROM_PATH, "arguments": {"state_id": ACTIVE}}
    _assert_no_forbidden(state, history)
    state["explained_anomalies"] = [{"family": "harmonic", "tool": RUN_HSE_FROM_PATH, "state_id": ACTIVE,
        "explained_signatures": [MINTED_HARMONIC, *WLS_SIGNATURES], "detail": {"bus_1based": 5}}]
    assert _first(state, history) == {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}


# ------------------------------------------------------- strict boundary invariants


def test_private_family_truth_and_hints_do_not_change_the_gated_route():
    for state in (_state(), _screened_state(), _hif_suspected_state()):
        _context(state, "measurement", [{"suspect_group": [26]}])
        expected = _oracle().next_actions(state, [NLM_HIF])
        for family in ("measurement", "parameter", "topology", "hif"):
            oracle_like = {"policy_observation": deepcopy(state), f"true_{family}_errors": [{"index": 55}],
                "oracle_action_hints": [{"tool": CORRECT_MEASUREMENTS, "arguments": {"suspect_group": [55]}}],
                "hidden_truth": {"oracle_terminal_eligible": True}}
            assert _oracle().next_actions(oracle_like, [NLM_HIF]) == expected


def test_stored_nlm_diagnostic_is_not_a_channel_under_strict_profiles():
    state = _state(signatures=["hif_suspected_zero_sequence", *WLS_SIGNATURES])
    state["available_evidence"] = ["nlm_diagnostic"]
    assert DiagnosticsExpert().propose(state, []) == []
    state["evidence_profile"] = AUXILIARY_EVIDENCE_PROFILE
    assert DiagnosticsExpert().propose(state, [])[0].action["tool"] == RUN_THREE_PHASE_NLM_FROM_PATH


def test_synthetic_terminal_eligibility_is_ignored_under_strict_profiles():
    from psse_env.state_store import SYNTHETIC_TERMINAL_COMPATIBILITY_KEY

    state = _state()
    state.update({"accepted_corrections": [{"candidate_state_id": ACTIVE}],
                  SYNTHETIC_TERMINAL_COMPATIBILITY_KEY: True, "oracle_terminal_eligible": True})
    assert TerminationExpert().propose(state, []) == []


@pytest.mark.parametrize("tool", sorted(DIAGNOSTIC_TOOLS | {RUN_ALTERNATIVE_TEST}))
def test_scada_only_still_refuses_every_auxiliary_tool_in_the_expert(tool):
    state = _state(profile=SCADA_ONLY_PROFILE)
    state["available_evidence"] = ["hif_scan_window", "nlm_diagnostic", "harmonic_measurements", *PHASE_CHANNELS]
    assert not diagnostic_tool_permitted(state, tool)
    assert all(action["tool"] not in DIAGNOSTIC_TOOLS for action in _oracle().next_actions(state, []))


def test_alternative_test_is_never_proposed_under_the_gated_profile():
    for state in (_state(), _screened_state(), _hif_suspected_state(), _state(alarm=False)):
        assert not diagnostic_tool_permitted(state, RUN_ALTERNATIVE_TEST)
        _assert_no_forbidden(state, [NLM_HIF])


# ----------------------------------------------------------- real environment


_HIF_CORPUS = (Path(__file__).resolve().parents[2] / "artifacts" / "measurements"
               / "hif_physical69_main_valid_detectable_7x10_20260921" / "samples.jsonl")


def _research_environment():
    from scripts.run_dagger_research import research_diagnostic_environment_factory

    env = research_diagnostic_environment_factory(evidence_profile=WLS_GATED_PROFILE)
    return env, ExpertPolicyOracle(process_oracle=env.process_oracle, candidate_oracle=env.candidate_quality_oracle)


def _drive(env, oracle, *, max_steps=40):
    tools = []
    while not env.is_terminal() and len(tools) < max_steps:
        actions = oracle.next_actions(env.get_oracle_state(), env.history)
        assert actions, env.current_state()
        action = actions[0]
        if action["tool"] in GATED_DIAGNOSTIC_TOOLS:
            assert current_wls_alarm(env.get_policy_observation().as_dict()), action
        if action["tool"] == ASK_FOR_MORE_EVIDENCE:
            assert action["arguments"].get("request") in SUPPORTED_REQUESTS, action
        assert action["tool"] != RUN_ALTERNATIVE_TEST
        env.assert_training_decision_evidence(action)
        _, output = env.step(action)
        assert output["execution_status"] == "success", (action, output)
        tools.append(action["tool"])
    return tools


@pytest.fixture(scope="module")
def gated_scenarios():
    pytest.importorskip("opendssdirect")
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator

    generator = Round0ScenarioGenerator(seed=20260719, evidence_profile=WLS_GATED_PROFILE,
        hif_sample_paths=[_HIF_CORPUS] if _HIF_CORPUS.is_file() else None)
    plan = {"harmonic": 1, "measurement": 1, "no_error": 1, "three_phase_unbalance": 1}
    if _HIF_CORPUS.is_file():
        plan["hif"] = 1
    return {row["scenario_family"]: row for row in generator.build(plan)}


def test_real_root_observation_carries_no_flag_and_expert_opens_with_wls(gated_scenarios):
    for family, scenario in gated_scenarios.items():
        env, oracle = _research_environment()
        env.reset(deepcopy(scenario))
        observation = env.get_policy_observation().as_dict()
        assert observation["evidence_profile"] == WLS_GATED_PROFILE, family
        assert observation["unresolved_signatures"] == [], family
        assert observation["available_evidence"] == [], family
        assert observation["explained_anomalies"] == [], family
        assert not current_wls_alarm(observation), family
        assert oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"] == RUN_WLS, family


def test_real_clean_root_closes_after_wls_without_any_request(gated_scenarios):
    env, oracle = _research_environment()
    env.reset(deepcopy(gated_scenarios["no_error"]))
    assert _drive(env, oracle) == [RUN_WLS, FINALIZE_DIAGNOSIS]
    assert env.terminal_outcome == "resolved"


def test_real_harmonic_root_discovers_the_source_through_unavailable_phasors(gated_scenarios):
    env, oracle = _research_environment()
    env.reset(deepcopy(gated_scenarios["harmonic"]))
    tools = _drive(env, oracle)
    assert tools == [RUN_WLS, GET_THREE_PHASE_CONTEXT, GET_HARMONIC_CONTEXT, RUN_HSE_FROM_PATH, FINALIZE_DIAGNOSIS]
    assert env.terminal_outcome == "resolved"
    record = env.get_policy_observation().explained_anomalies[0]
    assert record["family"] == "harmonic"
    assert record["detail"]["bus_1based"] == gated_scenarios["harmonic"]["hidden_truth"]["true_harmonic_errors"][0]["bus_1based"]


def test_real_measurement_root_falls_back_to_the_balanced_ladder(gated_scenarios):
    env, oracle = _research_environment()
    env.reset(deepcopy(gated_scenarios["measurement"]))
    tools = _drive(env, oracle)
    assert tools[:3] == [RUN_WLS, GET_THREE_PHASE_CONTEXT, GET_HARMONIC_CONTEXT]
    assert tools[3:7] == [GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS, RUN_WLS, "commit_state"]
    assert tools[-1] == ASK_FOR_MORE_EVIDENCE
    assert env.terminal_outcome == "operator_escalation"
    assert not env.get_policy_observation().explained_anomalies


def test_real_unbalance_root_is_explained_by_nlm_after_the_alarm(gated_scenarios):
    env, oracle = _research_environment()
    env.reset(deepcopy(gated_scenarios["three_phase_unbalance"]))
    tools = _drive(env, oracle)
    assert tools == [RUN_WLS, GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH, FINALIZE_DIAGNOSIS]
    assert env.terminal_outcome == "resolved"
    record = env.get_policy_observation().explained_anomalies[0]
    assert record["family"] == "three_phase_unbalance"


@pytest.mark.skipif(not _HIF_CORPUS.is_file(), reason="physical HIF corpus is not checked out")
def test_real_discovered_hif_root_runs_the_full_ladder(gated_scenarios):
    env, oracle = _research_environment()
    env.reset(deepcopy(gated_scenarios["hif"]))
    for tool in sorted(GATED_DIAGNOSTIC_TOOLS):
        assert not diagnostic_tool_permitted(env.get_policy_observation().as_dict(), tool)
    tools = _drive(env, oracle)
    assert tools[:4] == [RUN_WLS, GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH,
                         ESTIMATE_HIF_MULTISCAN_FROM_PATH]
    observation = env.get_policy_observation().as_dict()
    assert MINTED_HIF in observation["unresolved_signatures"]
    if env.terminal_outcome == "resolved":
        # Accepted fit: conditioned WLS, then closure without a correction.
        assert tools[-2:] == [RUN_WLS, FINALIZE_DIAGNOSIS]
        assert observation["explained_anomalies"][0]["family"] == "hif"
    else:
        # Both estimators rejected the fit: single-scan fallback, then the
        # explicit exhaustion handoff, never a meter or branch correction.
        assert env.terminal_outcome == "operator_escalation"
        assert tools[4:] == [ESTIMATE_HIF_FROM_PATH, ASK_FOR_MORE_EVIDENCE]
        assert env.history[-1]["action"]["arguments"]["request"] == HIF_DIAGNOSTICS_EXHAUSTED_REQUEST
    assert not any(tool.startswith("correct_") for tool in tools)
