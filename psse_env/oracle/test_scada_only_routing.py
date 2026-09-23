"""Strict routing uses balanced WLS and never turns labels into sensor evidence."""
from copy import deepcopy

import pytest

from psse_env.actions import (
    DIAGNOSTIC_TOOLS, GET_MEASUREMENT_CONTEXT, RUN_WLS, action_signature,
    harmonic_screening_pending, three_phase_acquisition_pending, three_phase_screening_pending,
    gnn_investigation_pending, successful_current_wls,
)
from psse_env.evidence_profile import (
    AUXILIARY_EVIDENCE_PROFILE, DEFAULT_EVIDENCE_PROFILE, SCADA_ONLY_PROFILE, WLS_GATED_PROFILE,
    is_strict_boundary,
)
from psse_env.oracle import DiagnosticsExpert, ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.termination_expert import TerminationExpert


# These invariants describe the literal scada_only profile.  The research
# default moved to wls_gated_diagnostics on 2026-09-23, so the profile is
# named explicitly here; the shared strict-boundary invariants (WLS first, no
# hints, no synthetic closure) are checked for both strict profiles below.
def _state(*, solved=True, alarm=True, profile=SCADA_ONLY_PROFILE):
    active = "strict:s0"
    return {"evidence_profile": profile, "active_state_id": active, "has_open_candidate": False,
        "remaining_budget": 40, "remaining_anomaly_score": (2.0 if alarm else .5) if solved else None,
        "no_material_anomaly_remaining": solved and not alarm,
        "unresolved_signatures": ["wls_residual_outlier_dominant index=26 channel=Pinj"] if solved and alarm else [],
        "explained_anomalies": [], "accepted_corrections": [], "available_evidence": [],
        "last_tool": RUN_WLS if solved else None, "last_tool_status": "success" if solved else None,
        "tried_action_signatures": [action_signature({"tool": RUN_WLS, "arguments": {"state_id": active}})] if solved else [],
        "fresh_context_evidence": {"wls": {"state_id": active, "state_hash": "current-snapshot",
            "evidence_source": "deployment_wls:test", "successful": solved, "anomalous": alarm,
            "chi_square_alarm": alarm, "normalized_residual_alarm": alarm,
            "normalized_residual_threshold": 4., "max_normalized_residual": 8. if alarm else 2.}}}


def _context(state, family, targets=()):
    active = state["active_state_id"]
    tool = {"measurement": "correct_measurements", "parameter": "correct_parameters", "topology": "correct_topology"}[family]
    state[f"has_fresh_{family}_context"] = True
    state[f"{family}_context_state_id"] = active
    state["fresh_context_evidence"][family] = {
        "state_id": active, "state_hash": "current-snapshot", "evidence_source": f"deployment_context:{family}",
        "route_status": "actionable" if targets else "complete_negative",
        "supported_corrections": [{"tool": tool, "arguments": {"state_id": active, **target}} for target in targets],
    }


def _oracle():
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))


@pytest.mark.parametrize("profile", [SCADA_ONLY_PROFILE, WLS_GATED_PROFILE])
@pytest.mark.parametrize("fields", [
    {}, {"unresolved_signatures": ["hif_suspected_zero_sequence"], "available_evidence": ["hif_scan_window", "nlm_diagnostic"]},
    {"unresolved_signatures": ["harmonic_distortion_detected"], "available_evidence": ["harmonic_measurements"]},
    {"unresolved_signatures": ["three_phase_unbalance"], "available_evidence": ["three_phase_voltages"]},
    {"no_material_anomaly_remaining": True, "remaining_anomaly_score": 0., "oracle_terminal_eligible": True},
])
def test_strict_opening_is_wls_even_with_legacy_flags_or_quiet_claim(fields, profile):
    state = _state(solved=False, profile=profile)
    state.update(fields)
    assert is_strict_boundary(state)
    assert _oracle().next_actions(state, [])[0] == {"tool": RUN_WLS, "arguments": {"state_id": state["active_state_id"]}}
    proposals = DiagnosticsExpert().propose(state, [], harmonic_fault_present=True, hif_fault_present=True)
    if profile == SCADA_ONLY_PROFILE:
        assert proposals == []
    else:
        # The WLS-gated ladder may only ask for the missing balanced baseline.
        assert all(proposal.action["tool"] == RUN_WLS for proposal in proposals)
    assert TerminationExpert().propose(state, []) == []


def test_missing_profile_defaults_to_a_strict_wls_first_profile():
    state = _state(solved=False)
    state.pop("evidence_profile")
    assert DEFAULT_EVIDENCE_PROFILE == WLS_GATED_PROFILE
    assert is_strict_boundary(state)
    state.update(unresolved_signatures=["hif_suspected_zero_sequence"], available_evidence=["nlm_diagnostic"])
    assert _oracle().next_actions(state, [])[0]["tool"] == RUN_WLS


@pytest.mark.parametrize("tool", sorted(DIAGNOSTIC_TOOLS | {"run_alternative_test"}))
def test_auxiliary_tools_remain_illegal_even_when_legacy_payload_claims_available(tool):
    state = _state()
    state["available_evidence"] = ["hif_scan_window", "nlm_diagnostic", "harmonic_measurements", "three_phase_voltages"]
    action = {"tool": tool, "arguments": {"state_id": state["active_state_id"], "candidate_branch_row0": 1}}
    checked = _oracle().process_oracle.check(state, action)
    assert not checked["process_valid"]
    assert checked["error_code"] == "tool_disabled_by_evidence_profile"
    assert not any(item["tool"] in DIAGNOSTIC_TOOLS for item in checked["valid_next_actions"])


@pytest.mark.parametrize("escalation_request", ["operator_escalation:hif_diagnostics_exhausted", "operator_escalation:hif_conditioning_unavailable"])
def test_strict_cannot_claim_a_specific_hif_escalation(escalation_request):
    state = _state()
    result = _oracle().process_oracle.check(state, {"tool": "ask_for_more_evidence",
        "arguments": {"state_id": state["active_state_id"], "request": escalation_request}})
    assert result["error_code"] == "tool_disabled_by_evidence_profile"


def test_current_scada_wls_routes_directly_to_balanced_context_and_supported_meter():
    state = _state()
    assert _oracle().next_actions(state, [])[0]["tool"] == GET_MEASUREMENT_CONTEXT
    _context(state, "measurement", [{"suspect_group": [26]}])
    expected = {"tool": "correct_measurements", "arguments": {"state_id": state["active_state_id"], "suspect_group": [26]}}
    assert _oracle().next_actions(state, [])[0] == expected
    assert _oracle().process_oracle.check(state, expected)["process_valid"]
    for family in ("measurement", "parameter", "topology"):
        assert _oracle().process_oracle.check(state, {"tool": f"get_{family}_context",
            "arguments": {"state_id": state["active_state_id"]}})["process_valid"]


def test_private_oracle_family_and_target_hints_do_not_change_strict_action():
    state = _state()
    _context(state, "measurement", [{"suspect_group": [26]}])
    expected = _oracle().next_actions(state, [])
    for family in ("measurement", "parameter", "topology", "hif"):
        oracle_like = {"policy_observation": deepcopy(state), f"true_{family}_errors": [{"index": 55}],
            "oracle_action_hints": [{"tool": "correct_measurements", "arguments": {"suspect_group": [55]}}]}
        assert _oracle().next_actions(oracle_like, []) == expected


@pytest.mark.parametrize("family,target", [("parameter", {"line_index": 3}), ("topology", {"line_index": 3, "status": 0})])
def test_balanced_branch_corrections_remain_available_without_auxiliary_requests(family, target):
    state = _state()
    state["unresolved_signatures"] = ["wls_branch_multiplier_dominant line_status_or_parameter line=3"]
    first = _oracle().next_actions(state, [])[0]
    assert first["tool"] in {"get_parameter_context", "get_topology_context"}
    _context(state, family, [target])
    action = _oracle().next_actions(state, [])[0]
    assert action == {"tool": f"correct_{'parameters' if family == 'parameter' else 'topology'}",
                      "arguments": {"state_id": state["active_state_id"], **target}}
    assert _oracle().process_oracle.check(state, action)["process_valid"]


def test_quiet_current_wls_can_close_without_claiming_a_waveform_family():
    state = _state(alarm=False)
    assert _oracle().next_actions(state, [])[0] == {"tool": "finalize_diagnosis", "arguments": {}}
    assert _oracle().process_oracle.check(state, {"tool": "finalize_diagnosis", "arguments": {}})["process_valid"]


def test_old_explanation_cannot_close_a_current_balanced_residual_alarm():
    state = _state()
    state["explained_anomalies"] = [{"family": "hif", "explained_signatures": list(state["unresolved_signatures"])}]
    assert not _oracle().process_oracle.check(state, {"tool": "finalize_diagnosis", "arguments": {}})["process_valid"]
    assert TerminationExpert().propose(state, []) == []
    assert DiagnosticsExpert().hif_continuation_proposals(state) == []


@pytest.mark.parametrize("mutate", ["missing", "failed", "old_state", "missing_hash"])
def test_balanced_context_correction_and_finalization_need_current_bound_wls(mutate):
    state = _state(alarm=False)
    _context(state, "measurement", [{"suspect_group": [26]}])
    wls = state["fresh_context_evidence"]["wls"]
    if mutate == "missing":
        state["fresh_context_evidence"].pop("wls")
    elif mutate == "failed":
        wls["successful"] = False
    elif mutate == "old_state":
        wls["state_id"] = "previous:s0"
    else:
        wls.pop("state_hash")
    state["semantic_field_provenance"] = {"remaining_anomaly_score": "deployment_wls:claimed"}
    assert not successful_current_wls(state)
    for tool, arguments in (("get_measurement_context", {}), ("correct_measurements", {"suspect_group": [26]}),
                            ("finalize_diagnosis", {})):
        checked = _oracle().process_oracle.check(state, {"tool": tool, "arguments": arguments})
        assert not checked["process_valid"], tool


def test_unresolved_balanced_model_hands_off_without_inventing_a_fault_family():
    state = _state()
    state["unresolved_signatures"] = ["wls_model_discrepancy"]
    for family in ("measurement", "parameter", "topology"):
        _context(state, family)
    proposal = _oracle().next_action_proposals(state, [])[0]
    assert proposal.action == {"tool": "ask_for_more_evidence", "arguments": {"state_id": state["active_state_id"],
        "request": "operator_escalation:recovery_options_exhausted"}}
    assert "unresolved_balanced_model_discrepancy_requires_operator_handoff" in proposal.evidence_codes
    assert not any("hif" in value for value in proposal.evidence_codes)


def test_same_scada_gnn_phase_score_can_request_only_balanced_contexts():
    state = _state(alarm=False)
    state["fresh_context_evidence"]["wls"]["gnn_screen"] = {
        "state_id": state["active_state_id"], "state_hash": "current-snapshot", "screen_status": "valid",
        "phase_trigger": True, "anomaly_trigger": True, "family_scores": {"measurement": .8, "hif": .99}}
    assert gnn_investigation_pending(state)
    assert _oracle().next_actions(state, [])[0]["tool"] == GET_MEASUREMENT_CONTEXT
    assert DiagnosticsExpert().three_phase_screening_proposals(state, []) == []


def test_auxiliary_acquisition_predicates_are_explicitly_opt_in():
    arguments = {"unresolved": ["wls_residual_outlier index=26"], "tried_action_signatures": [], "active_state_id": "episode:s0"}
    for profile in (SCADA_ONLY_PROFILE, WLS_GATED_PROFILE):
        # scada_only never acquires; the WLS-gated profile needs the current
        # alarm from the ledger, which this compact fixture does not carry.
        assert not harmonic_screening_pending(**arguments, evidence_profile=profile)
        assert not three_phase_acquisition_pending(**arguments, evidence_profile=profile)
        assert not three_phase_screening_pending(**arguments, available_evidence=["three_phase_voltages"], evidence_profile=profile)
    assert harmonic_screening_pending(**arguments, evidence_profile=AUXILIARY_EVIDENCE_PROFILE)
    assert three_phase_acquisition_pending(**arguments, evidence_profile=AUXILIARY_EVIDENCE_PROFILE)


def test_explicit_auxiliary_profile_preserves_historical_flagged_hif_ladder():
    state = _state(solved=False, profile=AUXILIARY_EVIDENCE_PROFILE)
    state.update(unresolved_signatures=["hif_suspected_zero_sequence"], available_evidence=["nlm_diagnostic", "hif_scan_window"])
    assert _oracle().next_actions(state, [])[0]["tool"] == "run_three_phase_nlm_from_path"


def test_real_wls_rejects_historical_partial_setup_with_insufficient_progress():
    """A known clean replacement does not override weak observable progress."""
    import json
    from pathlib import Path

    from psse_env.dagger import evaluator, release_factories

    suite_path = Path(release_factories.__file__).with_name("suites") / "bc0_eval_suite_v1.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    row = next(item for item in suite["partial_success_retention"]
               if item["grouping"]["root_scenario_id"] == "r0_1e06979fa21e")
    contract = evaluator.evaluation_intervention_contract(
        "partial_success_retention", row, required=False,
    )
    env = release_factories.production_environment_factory(seed=20260719)
    initial = env.reset(deepcopy(evaluator.strip_offline_truth(row)))
    original_active = initial["active_state_id"]
    bootstrap = {"tool": RUN_WLS, "arguments": {"state_id": "$active"}}
    for raw_action in [bootstrap, *contract["setup_actions"][:3]]:
        current = env.current_state()
        action = deepcopy(raw_action)
        bindings = {"$active": current["active_state_id"],
                    "$candidate": current.get("candidate_state_id")}
        action["arguments"] = {
            key: bindings.get(value, value) if isinstance(value, str) else value
            for key, value in action["arguments"].items()
        }
        # The frozen contract supplies this value solely as an audited setup
        # intervention. Runtime still judges its result from observed WLS.
        execute = (env.apply_audited_evaluation_setup_correction
                   if action["tool"] == "correct_measurements" else env.step)
        _, output = execute(action)
        assert output["execution_status"] == "success"

    candidate_id = env.current_state()["candidate_state_id"]
    candidate = env.store.get_state(candidate_id)
    verification = candidate["verification_output"]
    assert verification["target_fixed"] is True
    assert verification["physical_constraints_ok"] is True
    assert 0 < verification["global_progress"] < .30
    assert verification["globally_resolved"] is False
    assert candidate["candidate_disposition"] == "REJECT"
    assessment = env.get_oracle_state().candidate_assessment
    assert assessment["progress_class"] == "insufficient_global_progress"
    assert assessment["rationale_codes"] == ["partial_global_progress_below_threshold"]
    assert assessment["remaining_true_fault_count"] is None
    _, refused = env.step({"tool": "commit_state", "arguments": {"candidate_state_id": candidate_id}})
    assert refused["execution_status"] == "failure"
    assert env.current_state()["active_state_id"] == original_active
    assert not env.current_state().get("accepted_corrections")
