"""A refused or off-target diagnostic tests nothing (2026-10-02).

An off-path probe of the 2026-10-01 ranked cell's round-2 roots (a scripted
learner that follows the expert except for one action) found three ways a
learner's diagnostic call could hide the expert's own rung:

* a harmonic state estimate requested before any spectra was refused by the
  suspicion gate, yet counted as done, so after the spectra came back the
  expert handed off and the label audit refused the handoff (no HSE ran);
* an HIF estimate on a line outside the phasor localization was a real test
  of another hypothesis, yet it retired the estimator rung, so the expert
  declared the HIF diagnostics exhausted on a line the audit does not accept;
* a spectra request refused before the phasors counted as "spectra examined",
  so the expert never asked for them again and walked the balanced routes.

The rules now: a gate refusal never completes a rung or enters the expert's
seen set; the estimator rung is judged on the localized line; on the active
state an acquisition counts only through its ledger, and an HSE attempt
counts only after the latest spectra request.
"""
from __future__ import annotations

from copy import deepcopy

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, ESTIMATE_HIF_FROM_PATH, ESTIMATE_HIF_MULTISCAN_FROM_PATH, GET_HARMONIC_CONTEXT,
    GET_THREE_PHASE_CONTEXT, HIF_DIAGNOSTICS_EXHAUSTED_REQUEST, RUN_HSE_FROM_PATH, RUN_THREE_PHASE_NLM_FROM_PATH,
    RUN_WLS, action_signature, phasors_examined_in_episode, process_gate_refusal, requested_on_earlier_state,
    spectra_examined_in_episode,
)
from psse_env.oracle import DiagnosticsExpert
from psse_env.oracle.test_wls_gated_routing import (
    ACTIVE, HASH, NLM_HIF, WLS_SIGNATURES, MINTED_HARMONIC, _first, _hif_suspected_state, _oracle, _state, _telemetry,
)

ANCESTOR = "gated:s-1"


def _refused(tool, code, **arguments):
    return {"action": {"tool": tool, "arguments": {"state_id": ACTIVE, **arguments}},
            "tool_output": {"execution_status": "failure", "error_code": code, "tool_metrics": {}}}


def _estimate(tool, row, *, accepted, phase=None):
    arguments = {"state_id": ACTIVE, "candidate_branch_row0": row}
    if phase is not None:
        arguments["candidate_phase"] = phase
    return {"action": {"tool": tool, "arguments": arguments},
            "tool_output": {"execution_status": "success", "tool_metrics": {
                "state_id": ACTIVE, "state_hash": HASH, "diagnostic_acceptance": {"accepted": accepted},
                "evidence_source": "deployment_diagnostic:hif"}}}


WINDOW_MISSING = {"action": {"tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH,
                             "arguments": {"state_id": ACTIVE, "candidate_branch_row0": 3}},
                  "tool_output": {"execution_status": "failure", "error_code": "hif_scan_window_missing",
                                  "tool_metrics": {}}}


def _tried(state, *events):
    state["tried_action_signatures"] = [*state["tried_action_signatures"],
                                        *(action_signature(event["action"]) for event in events)]
    return list(events)


def test_gate_refusals_are_not_answers():
    for code in ("missing_precondition", "diagnostics_require_wls_alarm", "diagnostics_require_harmonic_suspicion",
                 "diagnostics_require_hif_suspicion", "evidence_profile_tool_unavailable", "state_reference_mismatch"):
        assert process_gate_refusal(code), code
    for code in ("hif_scan_window_missing", "hif_search_budget_invalid", None, ""):
        assert not process_gate_refusal(code), code
    completed = DiagnosticsExpert._completed_diagnostics([
        _refused(RUN_HSE_FROM_PATH, "diagnostics_require_harmonic_suspicion"),
        _refused(ESTIMATE_HIF_FROM_PATH, "diagnostics_require_hif_suspicion", candidate_branch_row0=3),
        WINDOW_MISSING,
    ], active_state_id=ACTIVE)
    # The provider's own answer (no scan window) retires multiscan; the refusals test nothing.
    assert set(completed) == {ESTIMATE_HIF_MULTISCAN_FROM_PATH}
    assert completed[ESTIMATE_HIF_MULTISCAN_FROM_PATH]["_arguments"]["candidate_branch_row0"] == 3


def test_a_refused_estimate_on_the_localized_line_is_asked_again():
    state = _hif_suspected_state(scan_window=False)
    refused = _refused(ESTIMATE_HIF_FROM_PATH, "diagnostics_require_hif_suspicion",
                       candidate_branch_row0=3, candidate_phase="B")
    history = _tried(state, refused, NLM_HIF, WINDOW_MISSING)
    assert _first(state, history) == {"tool": ESTIMATE_HIF_FROM_PATH, "arguments": {
        "state_id": ACTIVE, "candidate_branch_row0": 3, "candidate_phase": "B"}}


def test_an_estimate_on_another_line_leaves_the_localized_rung_open():
    state = _hif_suspected_state(scan_window=False)
    history = _tried(state, NLM_HIF, WINDOW_MISSING, _estimate(ESTIMATE_HIF_FROM_PATH, 7, accepted=False))
    assert _first(state, history) == {"tool": ESTIMATE_HIF_FROM_PATH, "arguments": {
        "state_id": ACTIVE, "candidate_branch_row0": 3, "candidate_phase": "B"}}
    # Once the localized line is rejected too, the diagnostics are exhausted.
    history += _tried(state, _estimate(ESTIMATE_HIF_FROM_PATH, 3, accepted=False, phase="B"))
    assert _first(state, history) == {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {
        "state_id": ACTIVE, "request": HIF_DIAGNOSTICS_EXHAUSTED_REQUEST}}


def _harmonic_state(*, hse_before_spectra):
    """Spectra with a distortion are held on the active state; one HSE attempt is on record."""
    state = _state(signatures=[*WLS_SIGNATURES, MINTED_HARMONIC])
    _telemetry(state, "three_phase", available=False)
    hse = {"tool": RUN_HSE_FROM_PATH, "arguments": {"state_id": ACTIVE}}
    spectra = {"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": ACTIVE}}
    order = [hse, spectra] if hse_before_spectra else [spectra, hse]
    state["tried_action_signatures"] += [action_signature(action) for action in order]
    _telemetry(state, "harmonic", available=True, channels=["harmonic_measurements"])
    return state


def test_an_hse_refused_before_the_spectra_is_run_once_they_arrive():
    # The refusal has left the bounded history window; only the tried list remains.
    state = _harmonic_state(hse_before_spectra=True)
    assert _first(state, []) == {"tool": RUN_HSE_FROM_PATH, "arguments": {"state_id": ACTIVE}}
    # While the refusal is still in the window it is discarded by its code as well.
    refused = _refused(RUN_HSE_FROM_PATH, "diagnostics_require_harmonic_suspicion")
    assert _first(state, [refused]) == {"tool": RUN_HSE_FROM_PATH, "arguments": {"state_id": ACTIVE}}


def test_an_hse_run_on_the_held_spectra_is_not_repeated():
    state = _harmonic_state(hse_before_spectra=False)
    ran = {"action": {"tool": RUN_HSE_FROM_PATH, "arguments": {"state_id": ACTIVE}},
           "tool_output": {"execution_status": "success", "tool_metrics": {
               "state_id": ACTIVE, "state_hash": HASH, "diagnostic_acceptance": {"accepted": False}}}}
    actions = _oracle().next_actions(deepcopy(state), [ran])
    assert actions and all(action["tool"] != RUN_HSE_FROM_PATH for action in actions), actions


def test_acquisitions_on_the_active_state_count_only_through_its_ledger():
    def state(tried, *, spectra_ledger=False, phase_ledger=False):
        contexts = {}
        if spectra_ledger:
            contexts["harmonic"] = {"state_id": ACTIVE, "request_attempted": True}
        if phase_ledger:
            contexts["three_phase"] = {"state_id": ACTIVE, "request_attempted": True, "nlm_attempted": True}
        return {"active_state_id": ACTIVE, "unresolved_signatures": [], "fresh_context_evidence": contexts,
                "tried_action_signatures": [action_signature({"tool": tool, "arguments": arguments})
                                            for tool, arguments in tried]}

    here, there, nowhere = {"state_id": ACTIVE}, {"state_id": ANCESTOR}, {}
    # Spectra: a refused request on the active state leaves no ledger and counts for nothing.
    assert not spectra_examined_in_episode(state([(GET_HARMONIC_CONTEXT, here)]))
    assert spectra_examined_in_episode(state([(GET_HARMONIC_CONTEXT, here)], spectra_ledger=True))
    assert spectra_examined_in_episode(state([(GET_HARMONIC_CONTEXT, there)]))
    assert not spectra_examined_in_episode(state([(GET_HARMONIC_CONTEXT, nowhere)]))
    # Phasors: both the acquisition and its test, on an earlier state, with no event standing.
    both_here = [(GET_THREE_PHASE_CONTEXT, here), (RUN_THREE_PHASE_NLM_FROM_PATH, here)]
    both_there = [(GET_THREE_PHASE_CONTEXT, there), (RUN_THREE_PHASE_NLM_FROM_PATH, there)]
    assert not phasors_examined_in_episode(state(both_here))
    assert phasors_examined_in_episode(state(both_here, phase_ledger=True))
    assert phasors_examined_in_episode(state(both_there))
    assert not phasors_examined_in_episode(state(both_there[:1]))
    standing = state(both_there)
    standing["unresolved_signatures"] = ["hif_suspected_line_differential"]
    assert not phasors_examined_in_episode(standing)
    assert requested_on_earlier_state(state([(RUN_WLS, there)]), RUN_WLS)
    assert not requested_on_earlier_state(state([(RUN_WLS, here)]), RUN_WLS)
