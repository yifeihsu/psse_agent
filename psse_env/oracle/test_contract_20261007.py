"""The 2026-10-07 contract changes of the suspicion-family profiles.

1. Under classifier_gated_diagnostics, phasors acquired and tested on an
   earlier state of the episode are not acquired again on a corrected child.
2. A committed correction that leaves the WLS quiet is handed to the operator
   in one request (``POST_CORRECTION_CONFIRMATION_REQUEST``), with no extra
   measurement context; other profiles keep the context-then-handoff path.
3. The WLS conditioned on an accepted HIF estimate runs inside the estimate's
   step (controller-run, outside the policy budget and history window).
4. The triage report carries no waveform family scores (test_runtime.py).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, ESTIMATE_HIF_FROM_PATH, GET_MEASUREMENT_CONTEXT, GET_THREE_PHASE_CONTEXT,
    POST_CORRECTION_CONFIRMATION_REQUEST, POST_CORRECTION_CONFIRMATION_SIGNATURE, RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_THREE_PHASE_NLM_FROM_PATH, RUN_WLS, action_signature, phasors_tested_in_episode,
)
from psse_env.evidence_profile import CLASSIFIER_GATED_PROFILE, SUSPICION_GATED_PROFILE, WLS_GATED_PROFILE
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle import test_classifier_gated_routing as routing

REPO = Path(__file__).resolve().parents[2]
DEVELOPMENT_SUITE = REPO / "output" / "classifier_triage_20261004" / "suites" / "development.json"


def _oracle():
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))


# ------------------------------------------------------------------------- 1


def test_an_admitted_request_on_a_corrected_child_does_not_reacquire_tested_phasors():
    state = routing._state(admitted=True)
    parent = "classifier:s-parent"
    state["tried_action_signatures"] += [
        action_signature({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": parent}}),
        action_signature({"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": parent}}),
    ]
    assert phasors_tested_in_episode(state)
    proposals = _oracle().next_action_proposals(state, [])
    assert all(p.action["tool"] != GET_THREE_PHASE_CONTEXT for p in proposals), proposals[:2]
    assert _oracle().diagnostics_expert.unexplained_acquisition_proposals(state, []) == []
    # Without an earlier test the admitted request still acquires them.
    assert _oracle().next_action_proposals(routing._state(admitted=True), [])[0].action["tool"] == GET_THREE_PHASE_CONTEXT


# ------------------------------------------------------------------------- 2


def _quiet_corrected_state(profile):
    state = routing._state(admitted=False, alarm=False, profile=profile)
    if profile != CLASSIFIER_GATED_PROFILE:
        state["fresh_context_evidence"]["wls"].pop("triage", None)
    state.update(
        no_material_anomaly_remaining=False, remaining_anomaly_score=0.4,
        unresolved_signatures=[POST_CORRECTION_CONFIRMATION_SIGNATURE],
        accepted_corrections=[{"candidate_state_id": routing.ACTIVE,
                               "source_action": {"tool": "correct_measurements",
                                                 "arguments": {"state_id": "classifier:s-parent", "suspect_group": [26]}}}],
    )
    return state


@pytest.mark.parametrize("profile", [CLASSIFIER_GATED_PROFILE, SUSPICION_GATED_PROFILE])
def test_a_quiet_corrected_state_is_handed_to_the_operator_in_one_request(profile):
    first = _oracle().next_action_proposals(_quiet_corrected_state(profile), [])[0]
    assert first.action["tool"] == ASK_FOR_MORE_EVIDENCE
    assert first.action["arguments"]["request"] == POST_CORRECTION_CONFIRMATION_REQUEST
    assert "committed_correction_left_wls_quiet" in first.evidence_codes


def test_other_profiles_keep_the_context_then_handoff_path():
    proposals = _oracle().next_action_proposals(_quiet_corrected_state(WLS_GATED_PROFILE), [])
    assert proposals[0].action["tool"] == GET_MEASUREMENT_CONTEXT
    assert all(p.action["arguments"].get("request") != POST_CORRECTION_CONFIRMATION_REQUEST for p in proposals)


@pytest.mark.parametrize("profile, expected", [
    (CLASSIFIER_GATED_PROFILE, [ASK_FOR_MORE_EVIDENCE]),
    (WLS_GATED_PROFILE, [GET_MEASUREMENT_CONTEXT]),
])
def test_the_process_repair_of_a_refused_correction_names_the_profiles_confirmation(profile, expected):
    state = _quiet_corrected_state(profile)
    repairs = ProcessValidityOracle(executor_hydrated_corrections=True).repair_actions(
        state, "post_correction_confirmation_required", None)
    assert [item["tool"] for item in repairs] == expected
    if profile == CLASSIFIER_GATED_PROFILE:
        assert repairs[0]["arguments"]["request"] == POST_CORRECTION_CONFIRMATION_REQUEST


def test_the_provider_reports_the_confirmation_only_for_a_quiet_corrected_state():
    from psse_env.providers.matpower import MatpowerDeploymentProviders

    providers = MatpowerDeploymentProviders(evidence_profile=SUSPICION_GATED_PROFILE)
    observation = _quiet_corrected_state(SUSPICION_GATED_PROFILE)
    state = {"state_id": routing.ACTIVE, "state_hash": routing.HASH, "evidence_profile": SUSPICION_GATED_PROFILE,
             "evidence_request": POST_CORRECTION_CONFIRMATION_REQUEST, "policy_observation": observation}
    report = providers.request_additional_evidence(state)
    assert report["request"] == POST_CORRECTION_CONFIRMATION_REQUEST and report["family"] == "post_correction_confirmation"
    assert report["additional_evidence_available"] is False and report["operator_review_required"] is True
    loud = copy.deepcopy(state)
    loud["policy_observation"]["remaining_anomaly_score"] = 2.0
    assert providers.request_additional_evidence(loud)["execution_status"] == "failure"
    unmarked = copy.deepcopy(state)
    unmarked["policy_observation"]["unresolved_signatures"] = []
    assert providers.request_additional_evidence(unmarked)["error_code"] == "post_correction_confirmation_unsupported"


# ------------------------------------------------------------------------- 2 and 3, closed loop


def _hif_root(family):
    if not DEVELOPMENT_SUITE.is_file():
        pytest.skip("development suite not present (output/classifier_triage_20261004/suites/development.json)")
    roots = json.loads(DEVELOPMENT_SUITE.read_text(encoding="utf-8"))
    return next(root for root in roots if root["grouping"]["scenario_family"] == family)


def _run(root, profile=CLASSIFIER_GATED_PROFILE, max_steps=30):
    import scripts.run_dagger_research as research
    from psse_env.dagger.release_factories import select_observable_expert_actions

    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = profile
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    env = research.resolve_environment_factory("research", profile)()
    oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
    env.reset(copy.deepcopy(root["execution"]))
    history, steps = [], []
    for _ in range(max_steps):
        observation = env.get_policy_observation(history).as_dict()
        action = select_observable_expert_actions(policy_observation=observation, expert_oracle=oracle).preferred_action
        budget_before = observation["remaining_budget"]
        _, output = env.step(action)
        history.append({"state_id": observation["active_state_id"], "candidate_state_id": observation.get("candidate_state_id"),
                        "action": copy.deepcopy(action), "tool_output": copy.deepcopy(output)})
        steps.append((action, output, budget_before, env.get_policy_observation(history).as_dict()))
        if env.terminal:
            break
    return env, steps


def test_the_conditioned_wls_runs_inside_the_accepted_hif_estimate_step():
    env, steps = _run(_hif_root("hif"))
    tools = [action["tool"] for action, *_ in steps]
    assert tools == [RUN_WLS, GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH, ESTIMATE_HIF_FROM_PATH, "finalize_diagnosis"]
    action, output, budget_before, after = steps[3]
    conditioned = output["tool_metrics"]["conditioned_wls"]
    assert conditioned["execution_status"] == "success" and conditioned["hif_conditioning"]["status"] == "ready"
    assert conditioned["normalized_residual_alarm"] is False and conditioned["hif_conditioning"]["remaining_meter_candidate_indices"] == []
    # One controller-run WLS in the controller history, none in the policy's window, one action off the budget.
    controller = [event for event in env.history if event.get("initiated_by")]
    assert len(controller) == 1 and controller[0]["action"]["tool"] == RUN_WLS
    assert after["remaining_budget"] == budget_before - 1
    assert after["last_tool"] == ESTIMATE_HIF_FROM_PATH and "conditioned_wls" in after["last_tool_output"]["tool_metrics"]
    assert all(not event.get("initiated_by") for event in env.get_policy_observation().as_dict()["history_window"])
    assert env.terminal_outcome == "resolved"


def test_a_quiet_corrected_root_closes_with_the_confirmation_request():
    env, steps = _run(_hif_root("measurement"))
    action, output, *_ = steps[-1]
    assert [a["tool"] for a, *_ in steps][-2:] == ["commit_state", ASK_FOR_MORE_EVIDENCE]
    assert action["arguments"]["request"] == POST_CORRECTION_CONFIRMATION_REQUEST
    audit = output["tool_metrics"]["operator_escalation_audit"]
    assert audit["family"] == "post_correction_confirmation" and audit["post_correction_confirmation_handoff"] is True
    assert env.terminal_outcome == "operator_escalation"
    assert RECOVERY_OPTIONS_EXHAUSTED_REQUEST not in json.dumps([a for a, *_ in steps])
