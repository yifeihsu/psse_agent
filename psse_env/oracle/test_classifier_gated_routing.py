"""Expert routing and the request gate under ``classifier_gated_diagnostics``.

The contract (docs/classifier_triage_plan_20261004.md, decision G1): the
balanced screen is replaced by a triage classifier whose report rides on the
WLS ledger as ``triage``.  Phasors follow its admitted request, or the
fallbacks after a balanced miss: a correction tried on the state was
rejected by verification, or every balanced context fetched on the state
offered no correction.  Spectra follow phasors that came back balanced.
"""
from __future__ import annotations

import pytest

from psse_env.actions import (
    CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, GET_HARMONIC_CONTEXT, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT,
    GET_THREE_PHASE_CONTEXT, GET_TOPOLOGY_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH, RUN_WLS, action_signature,
    balanced_route_failed, current_suspicion, current_triage_report, diagnostic_tool_permitted, triage_admits_request,
    triage_first_family,
)
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT, CLASSIFIER_GATED_PROMPT_PARAGRAPH, system_prompt_for_observation,
    tool_schemas_for_observation,
)
from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas
from psse_env.evidence_profile import (
    CLASSIFIER_GATED_PROFILE, SUSPICION_GATED_DISABLED_TOOLS, disabled_tools, is_classifier_gated, is_strict_boundary,
    is_wls_gated, required_suspicion, requires_wls_alarm_for_diagnostics,
)
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.expert_policy import _triage_first_order
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.process_validity import current_family_suspicion

ACTIVE = "classifier:s0"
HASH = "content-current"
WLS_SIGNATURES = [
    "wls_residual_outlier_dominant index=26 channel=Pinj",
    "wls_branch_multiplier line_status_or_parameter line=4",
]


def _report(*, admitted, score=None, first="measurement", status="valid"):
    threshold = 0.2782
    if score is None:
        score = 0.91 if admitted else 0.03
    return {"method": "triage_gnn", "model_id": "triage_gnn:test", "status": status, "request_score": score,
            "request_threshold": threshold, "request_admitted": bool(admitted), "first_family": first,
            "first_family_scores": {"measurement": 0.5, "parameter": 0.3, "topology": 0.2},
            "state_id": ACTIVE, "state_hash": HASH}


def _state(*, admitted=False, first="measurement", alarm=True, report=True, profile=CLASSIFIER_GATED_PROFILE):
    """A compact policy observation after the opening WLS, with the triage report on its ledger."""
    wls = {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_wls:test", "successful": True,
        "anomalous": alarm, "anomaly_breadth": 0.05, "chi_square_alarm": alarm, "normalized_residual_alarm": alarm,
        "normalized_residual_threshold": 4.0, "max_normalized_residual": 8.0 if alarm else 2.0, "bus_count": 14,
    }
    if report:
        wls["triage"] = _report(admitted=admitted, first=first)
    return {
        "evidence_profile": profile, "active_state_id": ACTIVE, "candidate_state_id": None, "has_open_candidate": False,
        "remaining_budget": 40, "remaining_anomaly_score": 2.0 if alarm else 0.5, "no_material_anomaly_remaining": not alarm,
        "unresolved_signatures": list(WLS_SIGNATURES) if alarm else [], "explained_anomalies": [],
        "accepted_corrections": [], "rejected_hypotheses": [], "available_evidence": [],
        "last_tool": RUN_WLS, "last_tool_status": "success",
        "tried_action_signatures": [action_signature({"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}})],
        "fresh_context_evidence": {"wls": wls},
    }


def _phasors(state, *, nlm_attempted=None, classification=None):
    record = {"state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_context:three_phase_measurements",
              "request_attempted": True, "three_phase_context_status": "available",
              "available_evidence_channels": ["three_phase_voltages", "three_phase_branch_currents"]}
    if nlm_attempted is not None:
        record["nlm_attempted"] = nlm_attempted
    if classification is not None:
        record["nlm_classification"] = classification
    state["fresh_context_evidence"]["three_phase"] = record
    state["available_evidence"] = ["three_phase_voltages", "three_phase_branch_currents"]
    state["tried_action_signatures"].append(action_signature({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": ACTIVE}}))


def _context(state, family, targets=()):
    tool = {"measurement": CORRECT_MEASUREMENTS, "parameter": CORRECT_PARAMETERS, "topology": "correct_topology"}[family]
    state[f"has_fresh_{family}_context"] = True
    state[f"{family}_context_state_id"] = ACTIVE
    state["fresh_context_evidence"][family] = {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": f"deployment_context:{family}",
        "route_status": "actionable" if targets else "complete_negative",
        "supported_corrections": [{"tool": tool, "arguments": {"state_id": ACTIVE, **target}} for target in targets],
    }


def _rejected(state, tool=CORRECT_MEASUREMENTS, arguments=None, kind="verification"):
    state["rejected_hypotheses"].append({
        "candidate_parent_id": ACTIVE, "rejection_kind": kind,
        "source_action": {"tool": tool, "arguments": {"state_id": ACTIVE, **(arguments or {"suspect_group": [26]})}},
    })


def _oracle():
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))


def _first(state, history=None):
    actions = _oracle().next_actions(state, history or [])
    assert actions, state
    return actions[0]


# --------------------------------------------------------------------------- profile


def test_profile_is_strict_alarm_gated_and_hides_the_same_tools_as_the_suspicion_profile():
    assert is_classifier_gated(CLASSIFIER_GATED_PROFILE) and is_strict_boundary(CLASSIFIER_GATED_PROFILE)
    assert is_wls_gated(CLASSIFIER_GATED_PROFILE) and requires_wls_alarm_for_diagnostics(CLASSIFIER_GATED_PROFILE)
    assert disabled_tools(CLASSIFIER_GATED_PROFILE) == SUSPICION_GATED_DISABLED_TOOLS
    assert required_suspicion(CLASSIFIER_GATED_PROFILE, GET_THREE_PHASE_CONTEXT) == "phasor"
    assert required_suspicion(CLASSIFIER_GATED_PROFILE, GET_HARMONIC_CONTEXT) == "harmonic"
    visible = {row["function"]["name"] for row in tool_schemas_for_observation(canonical_prompt_tool_schemas(), _state())}
    assert "run_alternative_test" not in visible and "estimate_hif_location_magnitude_multiscan_from_path" not in visible
    assert GET_THREE_PHASE_CONTEXT in visible
    prompt = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, _state())
    assert prompt == CANONICAL_DAGGER_SYSTEM_PROMPT + CLASSIFIER_GATED_PROMPT_PARAGRAPH
    assert "wls.triage" in prompt and "rejected by verification" in prompt


# ------------------------------------------------------------------------------ gate


def test_the_report_is_read_off_the_current_bound_wls_only():
    state = _state(admitted=True)
    assert triage_admits_request(current_triage_report(state)) and triage_first_family(current_triage_report(state)) == "measurement"
    stale = _state(admitted=True)
    stale["fresh_context_evidence"]["wls"]["state_id"] = "classifier:s1"
    assert current_triage_report(stale) == {}
    unavailable = _state(admitted=True)
    unavailable["fresh_context_evidence"]["wls"]["triage"]["status"] = "unavailable"
    assert not triage_admits_request(current_triage_report(unavailable)) and triage_first_family(current_triage_report(unavailable)) is None
    assert not triage_admits_request(current_triage_report(_state(report=False)))


def test_phasors_are_admitted_by_the_report_or_by_a_failed_balanced_route():
    assert current_suspicion(_state(admitted=True), "phasor")
    assert diagnostic_tool_permitted(_state(admitted=True), GET_THREE_PHASE_CONTEXT)
    quiet = _state(admitted=False)
    assert not current_suspicion(quiet, "phasor") and not diagnostic_tool_permitted(quiet, GET_THREE_PHASE_CONTEXT)
    # No alarm: nothing is admitted whatever the report says.
    assert not diagnostic_tool_permitted(_state(admitted=True, alarm=False), GET_THREE_PHASE_CONTEXT)
    # A verification-rejected correction on this state opens the tier; an executor failure does not.
    rejected = _state(admitted=False)
    _rejected(rejected)
    assert balanced_route_failed(rejected) and current_suspicion(rejected, "phasor")
    failed = _state(admitted=False)
    _rejected(failed, kind="executor_failure")
    assert not balanced_route_failed(failed)
    elsewhere = _state(admitted=False)
    _rejected(elsewhere)
    elsewhere["rejected_hypotheses"][0]["candidate_parent_id"] = "classifier:s9"
    assert not balanced_route_failed(elsewhere)
    # Contexts fetched on this state that offered nothing open it; one that offered a correction does not.
    empty = _state(admitted=False)
    _context(empty, "measurement")
    _context(empty, "parameter")
    assert balanced_route_failed(empty)
    offered = _state(admitted=False)
    _context(offered, "measurement", targets=[{"suspect_group": [26]}])
    _context(offered, "parameter")
    assert not balanced_route_failed(offered)
    assert not balanced_route_failed(_state(admitted=False))


def test_hif_and_harmonic_suspicions_come_from_the_phasors_not_from_a_screen():
    state = _state(admitted=True)
    assert not current_suspicion(state, "hif") and not current_suspicion(state, "harmonic")
    _phasors(state, nlm_attempted=True, classification="hif_suspected")
    assert current_suspicion(state, "hif") and not current_suspicion(state, "harmonic")
    balanced = _state(admitted=True)
    _phasors(balanced, nlm_attempted=True, classification="balanced_three_phase")
    assert current_suspicion(balanced, "harmonic") and not current_suspicion(balanced, "hif")
    assert not current_suspicion(_state(admitted=True), "voltage_meter") and not current_suspicion(_state(admitted=True), "unexplained")


def test_candidate_verification_carries_the_report_for_the_gate():
    state = _state(admitted=False)
    state["candidate_state_id"] = "classifier:c1"
    state["last_verification"] = {"state_id": "classifier:c1", "triage": _report(admitted=True)}
    assert current_family_suspicion(state, "phasor", "classifier:c1")
    state["last_verification"]["triage"]["request_admitted"] = False
    assert not current_family_suspicion(state, "phasor", "classifier:c1")


# ---------------------------------------------------------------------------- expert


def test_admitted_request_acquires_phasors_then_tests_them_then_opens_spectra_when_balanced():
    state = _state(admitted=True)
    first = _first(state)
    assert first["tool"] == GET_THREE_PHASE_CONTEXT, first
    _phasors(state, nlm_attempted=False)
    assert _first(state)["tool"] == RUN_THREE_PHASE_NLM_FROM_PATH
    state["fresh_context_evidence"]["three_phase"]["nlm_attempted"] = True
    state["fresh_context_evidence"]["three_phase"]["nlm_classification"] = "balanced_three_phase"
    assert _first(state)["tool"] == GET_HARMONIC_CONTEXT


def test_without_an_admitted_request_the_balanced_ladder_runs_first_family_first():
    # Inside the Lagrangian dead band (neither tag dominant) every balanced
    # route is proposed and the triage's first family leads.
    undecided = ["wls_residual_outlier index=26 channel=Pinj", "wls_branch_multiplier line_status_or_parameter line=4"]
    for family, tool in (("measurement", GET_MEASUREMENT_CONTEXT), ("parameter", GET_PARAMETER_CONTEXT), ("topology", GET_TOPOLOGY_CONTEXT)):
        state = _state(admitted=False, first=family)
        state["unresolved_signatures"] = list(undecided)
        proposals = _oracle().next_action_proposals(state, [])
        assert proposals[0].action["tool"] == tool, (family, proposals[0])
        assert all(p.action["tool"] != GET_THREE_PHASE_CONTEXT for p in proposals)
    # A dominant residual still suppresses the branch routes (the physics
    # rule of the baseline experts); the triage orders what is proposed.
    dominant = _state(admitted=False, first="parameter")
    assert [p.action["tool"] for p in _oracle().next_action_proposals(dominant, [])] == [GET_MEASUREMENT_CONTEXT]


def test_triage_order_falls_back_to_structural_first_without_a_report_or_a_match():
    def proposal(tool):
        return ExpertActionProposal(action={"tool": tool, "arguments": {"state_id": ACTIVE}}, source_expert="test",
                                    confidence=0.5, evidence_codes=[], admissible=True)

    ranked = [proposal(GET_MEASUREMENT_CONTEXT), proposal(GET_PARAMETER_CONTEXT), proposal(GET_TOPOLOGY_CONTEXT)]
    assert [p.action["tool"] for p in _triage_first_order(ranked, "topology")] == [GET_TOPOLOGY_CONTEXT, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT]
    assert [p.action["tool"] for p in _triage_first_order(ranked, None)] == [GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT, GET_MEASUREMENT_CONTEXT]
    assert [p.action["tool"] for p in _triage_first_order(ranked[:1], "parameter")] == [GET_MEASUREMENT_CONTEXT]


def test_a_rejected_balanced_correction_opens_the_phasor_tier_as_the_exhaustion_step():
    state = _state(admitted=False)
    _context(state, "measurement", targets=[{"suspect_group": [26]}])
    _rejected(state)
    proposals = _oracle().diagnostics_expert.unexplained_acquisition_proposals(state, [])
    assert proposals and proposals[0].action["tool"] == GET_THREE_PHASE_CONTEXT
    assert "balanced_routes_exhausted" in proposals[0].evidence_codes
    untouched = _state(admitted=False)
    _context(untouched, "measurement", targets=[{"suspect_group": [26]}])
    assert _oracle().diagnostics_expert.unexplained_acquisition_proposals(untouched, []) == []


@pytest.mark.parametrize("admitted", [True, False])
def test_nothing_gated_is_proposed_without_the_alarm(admitted):
    state = _state(admitted=admitted, alarm=False)
    for proposal in _oracle().next_action_proposals(state, []):
        assert proposal.action["tool"] not in {GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH, GET_HARMONIC_CONTEXT}, proposal
