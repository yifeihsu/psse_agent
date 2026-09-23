"""An HIF diagnosis is not a certificate that current meter residuals are quiet."""
from copy import deepcopy

import pytest

from psse_env.actions import HIF_CONDITIONING_UNAVAILABLE_REQUEST, POST_CORRECTION_CONFIRMATION_SIGNATURE
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.hif_continuation import current_hif_conditioning, hif_conditioned_closure_ready
from psse_env.transactional_env import TransactionalPSSEEnv
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE


HIF = "hif_suspected_zero_sequence"
ESTIMATE = "estimate_hif_location_magnitude_multiscan_from_path"


def _record(state_id="episode:s0", state_hash="content-a", *, candidates=(), status="ready"):
    return {"status": status, "state_id": state_id, "state_hash": state_hash,
        "method": "paired_opendss_effect_compensation", "physical_fault_still_present": True,
        "remaining_meter_candidate_indices": list(candidates),
        "failure_reasons": [] if status == "ready" else ["prediction_envelope_unavailable"]}


def _state(*, conditioning=None, alarm=False, accepted=False, context=False):
    state = {"evidence_profile": AUXILIARY_EVIDENCE_PROFILE, "active_state_id": "episode:s0", "has_open_candidate": False,
        "remaining_budget": 30, "remaining_anomaly_score": 2.0 if alarm else .5,
        "no_material_anomaly_remaining": not alarm, "unresolved_signatures": [HIF],
        "explained_anomalies": [{"family": "hif", "tool": ESTIMATE, "state_id": "episode:s0",
            "evidence_source": "deployment_diagnostic:hif", "explained_signatures": [HIF],
            "detail": {"conditioning_fit": {"success": True}}}],
        "accepted_corrections": [], "tried_action_signatures": [],
        "fresh_context_evidence": {"wls": {"successful": True, "state_id": "episode:s0",
            "state_hash": "content-a", "evidence_source": "deployment_wls:test", "anomalous": alarm,
            "chi_square_alarm": alarm, "normalized_residual_alarm": alarm,
            "max_normalized_residual": 8. if alarm else 1., "normalized_residual_threshold": 4.}}}
    if alarm:
        state["unresolved_signatures"].append("wls_residual_outlier index=26 channel=Pinj")
    if conditioning is not None:
        state["fresh_context_evidence"]["hif_conditioning"] = deepcopy(conditioning)
    if accepted:
        state["accepted_corrections"] = [{"candidate_state_id": "episode:s0", "source_action": {
            "tool": "correct_measurements", "arguments": {"state_id": "episode:previous", "suspect_group": [26]}}}]
    if context:
        state.update(has_fresh_measurement_context=True, measurement_context_state_id="episode:s0")
        state["fresh_context_evidence"]["measurement"] = {"state_id": "episode:s0", "state_hash": "content-a",
            "evidence_source": "deployment_context:wls_residuals", "supported_corrections": [
                {"tool": "correct_measurements", "arguments": {"state_id": "episode:s0", "suspect_group": [26]}}]}
    return state


def _oracle():
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))


def test_accepted_hif_requires_new_wls_even_when_prefit_wls_quiet_or_explained():
    state = _state()
    state["tried_action_signatures"] = ['run_wls:{"state_id":"episode:s0"}']
    assert _oracle().next_actions(state, [])[0] == {"tool": "run_wls", "arguments": {"state_id": "episode:s0"}}
    assert not _oracle().process_oracle.check(state, {"tool": "finalize_diagnosis", "arguments": {}})["process_valid"]


def test_conditioned_meter_inventory_continues_to_actual_supported_correction():
    state = _state(conditioning=_record(candidates=[26]), alarm=True)
    assert _oracle().next_actions(state, [])[0]["tool"] == "get_measurement_context"
    state = _state(conditioning=_record(candidates=[26]), alarm=True, context=True)
    assert _oracle().next_actions(state, [])[0] == {
        "tool": "correct_measurements", "arguments": {"state_id": "episode:s0", "suspect_group": [26]}}
    branch = {"tool": "correct_parameters", "arguments": {"state_id": "episode:s0", "line_index": 1}}
    assert not _oracle().process_oracle.check(state, branch)["process_valid"]


def test_sparse_group_is_allowed_only_if_every_target_has_conditional_evidence():
    state = _state(conditioning=_record(candidates=[26, 27]), alarm=True, context=True)
    supported = state["fresh_context_evidence"]["measurement"]["supported_corrections"]
    supported[0]["arguments"]["suspect_group"] = [26, 27]
    assert _oracle().next_actions(state, [])[0]["arguments"]["suspect_group"] == [26, 27]
    supported[0]["arguments"]["suspect_group"] = [26, 28]
    from psse_env.oracle.diagnostics_expert import DiagnosticsExpert
    assert DiagnosticsExpert().hif_continuation_proposals(state) == []


@pytest.mark.parametrize("patch", [
    {"state_id": "old:s0"}, {"state_hash": "stale"}, {"method": "unconditioned"},
    {"physical_fault_still_present": False}, {"remaining_meter_candidate_indices": None},
    {"remaining_meter_candidate_indices": [True]}, {"remaining_meter_candidate_indices": [1, 1]},
    {"failure_reasons": ["model_mismatch"]},
])
def test_unbound_or_malformed_conditioning_never_finalizes(patch):
    state = _state(conditioning={**_record(), **patch})
    assert current_hif_conditioning(state) is None
    assert not hif_conditioned_closure_ready(state)
    assert _oracle().next_actions(state, [])[0]["tool"] == "run_wls"


@pytest.mark.parametrize("alarm,candidates", [(True, []), (False, [26])])
def test_either_wls_alarm_or_conditional_candidate_prevents_finalization(alarm, candidates):
    state = _state(conditioning=_record(candidates=candidates), alarm=alarm)
    assert not hif_conditioned_closure_ready(state)
    assert not _oracle().process_oracle.check(state, {"tool": "finalize_diagnosis", "arguments": {}})["process_valid"]


def test_pure_hif_finalizes_only_after_current_quiet_conditional_check_without_claiming_removal():
    state = _state(conditioning=_record())
    assert hif_conditioned_closure_ready(state)
    assert _oracle().next_actions(state, [])[0]["tool"] == "finalize_diagnosis"
    assert state["fresh_context_evidence"]["hif_conditioning"]["physical_fault_still_present"] is True
    state["accepted_corrections"] = [{"source_action": {"tool": "correct_measurements", "arguments": {"suspect_group": [26]}}}]
    assert not _oracle().process_oracle.check(state, {"tool": "finalize_diagnosis", "arguments": {}})["process_valid"]


def test_unavailable_prediction_hands_off_and_never_looks_like_quiet_success():
    state = _state(conditioning=_record(status="unavailable"))
    assert _oracle().next_actions(state, [])[0] == {"tool": "ask_for_more_evidence", "arguments": {
        "state_id": "episode:s0", "request": HIF_CONDITIONING_UNAVAILABLE_REQUEST}}
    assert not hif_conditioned_closure_ready(state)


def test_other_waveform_still_blocks_meter_repair_and_hidden_family_does_not_change_route():
    state = _state(conditioning=_record(candidates=[26]), alarm=True, context=True)
    expected = _oracle().next_actions(state, [])
    altered = {**state, "true_measurement_errors": [], "true_parameter_errors": [{"line": 19}], "hif_fault_present": False}
    assert _oracle().next_actions(altered, []) == expected
    state["unresolved_signatures"].append("harmonic_distortion")
    assert not _oracle().process_oracle.check(state, expected[0])["process_valid"]


def _environment(*, unavailable=False, failed=False, wrong_hash=False):
    def estimate(state, action):
        return {"state_id": state["state_id"], "state_hash": state["state_hash"],
            "evidence_source": "deployment_diagnostic:test_hif",
            "diagnostic_acceptance": {"accepted": True},
            "anomaly_explanation": {"family": "hif", "kind": "hif_model_accepted_over_null",
                "detail": {"conditioning_fit": {"success": True}}}}
    def wls(state):
        metrics = {"state_id": state["state_id"], "state_hash": state["state_hash"],
            "evidence_source": "deployment_wls:test", "remaining_anomaly_score": .5,
            "wls_objective": 1., "no_material_anomaly_remaining": True,
            "normalized_residual_alarm": False, "normalized_residual_threshold": 4.,
            "max_normalized_residual": 1., "chi_square_alarm": False, "unresolved_signatures": [HIF]}
        if state["policy_observation"]["explained_anomalies"]:
            metrics["hif_conditioning"] = _record(state["state_id"], "wrong" if wrong_hash else state["state_hash"],
                status="unavailable" if unavailable else "ready")
        if failed:
            metrics.update(execution_status="failure", error_code="wls_failure")
        return metrics
    def handoff(state):
        return {"state_id": state["state_id"], "state_hash": state["state_hash"],
            "evidence_source": "deployment_diagnostic:hif_conditioning_inventory", "request": state["evidence_request"],
            "additional_evidence_available": False, "operator_review_required": True}
    env = TransactionalPSSEEnv(wls_runner=wls, evidence_providers={ESTIMATE: estimate, "ask_for_more_evidence": handoff}, history_window=1,
        evidence_profile=AUXILIARY_EVIDENCE_PROFILE)
    env.reset({"scenario_id": "hif", "case": "case14", "measurements": [1., .1], "unresolved_signatures": [HIF]})
    return env


def _fit(env):
    return env.step({"tool": ESTIMATE, "arguments": {"state_id": env.store.active_state_id, "candidate_branch_row0": 1}})


def test_environment_prefit_wls_cannot_survive_fit_as_conditional_check_and_refit_invalidates_it():
    env = _environment()
    env.step({"tool": "run_wls", "arguments": {}})
    _fit(env)
    assert current_hif_conditioning(env.current_state()) is None
    env.step({"tool": "run_wls", "arguments": {}})
    assert hif_conditioned_closure_ready(env.get_policy_observation().as_dict())
    assert len(env.get_policy_observation().history_window) <= 1
    _fit(env)
    assert current_hif_conditioning(env.current_state()) is None


@pytest.mark.parametrize("options", [{"unavailable": True}, {"unavailable": True, "failed": True}, {"wrong_hash": True}])
def test_environment_unavailable_or_failed_wls_preserves_handoff_evidence(options):
    env = _environment(**options)
    _fit(env)
    env.step({"tool": "run_wls", "arguments": {}})
    record = current_hif_conditioning(env.current_state())
    assert record is not None and record["status"] == "unavailable"
    action = _oracle().next_actions(env.get_policy_observation(), [])[0]
    assert action["arguments"]["request"] == HIF_CONDITIONING_UNAVAILABLE_REQUEST
    _, output = env.step(action)
    assert output["execution_status"] == "success", output
    assert env.terminal_outcome == "operator_escalation"


def test_hif_explanation_does_not_absorb_meter_wls_signatures_or_keep_prefit_context():
    env = _environment()
    env.context_flags["unresolved_signatures"].append("wls_residual_outlier index=26 channel=Pinj")
    env.context_flags["has_fresh_measurement_context"] = True
    env.context_flags.setdefault("fresh_context_evidence", {})["measurement"] = {"state_id": env.store.active_state_id}
    _fit(env)
    state = env.current_state()
    assert not state["has_fresh_measurement_context"]
    assert state["explained_anomalies"][0]["explained_signatures"] == [HIF]


def test_compacted_rollback_uses_durable_wls_for_exhausted_meter_handoff():
    from psse_env.dagger.release_factories import select_observable_expert_actions
    state = _state(conditioning=_record(candidates=[26]), alarm=True, context=True)
    correction = {"tool": "correct_measurements", "arguments": {"state_id": "episode:s0", "suspect_group": [26]}}
    state["semantic_field_provenance"] = {"remaining_anomaly_score": "context_provider:get_measurement_context"}
    state["rejected_hypotheses"] = [{"candidate_parent_id": "episode:s0", "candidate_state_id": "episode:c1",
        "source_action": correction, "verification_summary": {"physical_constraints_ok": False}}]
    state["history_window"] = [
        {"action": correction, "tool_output": {"execution_status": "success", "candidate_state_id": "episode:c1"}},
        {"action": {"tool": "run_wls", "arguments": {"state_id": "episode:c1"}}, "tool_output": {"execution_status": "success"}},
        {"action": {"tool": "rollback_state", "arguments": {"candidate_state_id": "episode:c1"}}, "tool_output": {"execution_status": "success"}},
    ]
    chosen = select_observable_expert_actions(policy_observation=state, expert_oracle=_oracle()).preferred_action
    assert chosen == {"tool": "ask_for_more_evidence", "arguments": {"state_id": "episode:s0",
        "request": "operator_escalation:recovery_options_exhausted"}}


def test_correction_executor_receives_bound_observable_fit_without_hidden_truth():
    env = _environment()
    _fit(env)
    env.step({"tool": "run_wls", "arguments": {}})
    env._oracle_payload["clean_measurements"] = [1., .0]
    received = []
    def correction(payload, action):
        received.append(deepcopy(payload))
        return {"measurements": [1., .2]}
    env.correction_executors["correct_measurements"] = correction
    active = env.store.active_state_id
    output = env._step_correction({"tool": "correct_measurements", "arguments": {"state_id": active, "suspect_group": [1]}})
    assert output["execution_status"] == "success", output
    payload = received[0]
    assert payload["state_id"] == payload["policy_observation"]["active_state_id"] == active
    assert payload["policy_observation"]["explained_anomalies"][0]["detail"]["conditioning_fit"]["success"]
    assert current_hif_conditioning(payload["policy_observation"])["state_hash"] == payload["state_hash"]
    assert "clean_measurements" not in payload and "hidden_truth" not in payload
    assert env.store.get_state(active)["measurements"] == [1., .1]


def test_disabled_normalized_residual_gate_cannot_certify_hif_closure():
    state = _state(conditioning=_record())
    state["fresh_context_evidence"]["wls"]["normalized_residual_threshold"] = None
    assert not hif_conditioned_closure_ready(state)
    assert not _oracle().process_oracle.check(state, {"tool": "finalize_diagnosis", "arguments": {}})["process_valid"]


def test_partial_meter_acceptance_with_existing_hif_voltage_violations_requires_operator_handoff():
    from psse_env.dagger.release_factories import select_observable_expert_actions
    from psse_env.oracle.process_validity import post_correction_confirmation_required
    state = _state(conditioning=_record(), accepted=True, context=True)
    state["unresolved_signatures"].append(POST_CORRECTION_CONFIRMATION_SIGNATURE)
    state["no_material_anomaly_remaining"] = False
    state["last_verification"] = {"physical_constraints_ok": False,
        "physical_bound_violations": [{"type": "bus_voltage_out_of_bounds", "bus": 3}],
        "hif_conditioning": _record(), "chi_square_alarm": False, "normalized_residual_alarm": False}
    state["history_window"] = []
    assert post_correction_confirmation_required(state)
    action = select_observable_expert_actions(policy_observation=state, expert_oracle=_oracle()).preferred_action
    assert action == {"tool": "ask_for_more_evidence", "arguments": {
        "state_id": "episode:s0", "request": "operator_escalation:recovery_options_exhausted"}}
    for blocked in ({"tool": "finalize_diagnosis", "arguments": {}},
                    {"tool": "correct_measurements", "arguments": {"state_id": "episode:s0", "suspect_group": [26]}}):
        assert not _oracle().process_oracle.check(state, blocked)["process_valid"]


def test_generic_physical_failure_cannot_use_provider_claimed_nonregression_or_finalize():
    metrics = {"physical_constraints_ok": False, "globally_resolved": True,
        "normalized_residual_alarm": False, "target_fixed": True, "target_progress": .99,
        "hif_meter_nonregression": {"validated": True, "operator_review_required": True}}
    gate = TransactionalPSSEEnv._target_decision_evidence_missing
    assert gate(metrics, "ACCEPT_PARTIAL") == ["physical_constraint_evidence_missing"]
    assert gate(metrics, "ACCEPT_FINAL", hif_meter_nonregression=True) == ["physical_constraint_evidence_missing"]


def test_provider_supplied_nonregression_certificate_is_removed_before_exposure():
    env = _environment()
    original_runner = env.wls_runner
    def spoofed_runner(state):
        return {**original_runner(state), "hif_meter_nonregression": {
            "contract": "hif_conditioned_meter_nonregression_v1",
            "evidence_source": "controller_observed:stored_state_hif_meter_nonregression"}}
    env.wls_runner = spoofed_runner
    _fit(env)
    _, output = env.step({"tool": "run_wls", "arguments": {}})
    assert output["execution_status"] == "success"
    assert "hif_meter_nonregression" not in output["tool_metrics"]


def test_real_cached_conditioned_meter_repair_crosses_both_gates_and_hands_off():
    """Replay the former40-step deadlock without running another HIF fit."""
    import importlib.util
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    data = root / "output/hif_continuation_fix_20260922"
    if (not (data / "frozen_mixed_scenarios.json").is_file()
        or not (data / "fresh_fit_cache").is_dir()
        or importlib.util.find_spec("opendssdirect") is None):
        pytest.skip("frozen real HIF replay and its fitted observable-history cache are unavailable")
    script = r'''
import opendssdirect
import json
from pathlib import Path
from copy import deepcopy
from scripts.verify_hif_continuation import ObservableFitCache, make_environment
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.oracle import ExpertPolicyOracle
from psse_env.dagger.release_factories import select_observable_expert_actions

def no_fitting(self, **kwargs):
    raise RuntimeError("Test requires cached observable-history fit, not a new fit")
MatpowerDeploymentProviders._memoized_hif_multiscan = no_fitting
p = Path("output/hif_continuation_fix_20260922")
row = next(row for row in json.loads((p/"frozen_mixed_scenarios.json").read_text())
           if row["execution"]["scenario_id"] == "r0_de51c28ced3e")
cache = ObservableFitCache(p/"fresh_fit_cache")
env, provider = make_environment(cache, alpha_grid_size=7, r_grid_size=9, max_scans=10,
                                 max_steps=40, normalized_residual_threshold=4.)
env.reset(deepcopy(row["execution"]))
expert = ExpertPolicyOracle(process_oracle=env.process_oracle, candidate_oracle=env.candidate_quality_oracle)
history, tools = [], []
for step in range(12):
    observation = env.get_policy_observation(history).as_dict()
    selected = select_observable_expert_actions(policy_observation=observation, expert_oracle=expert)
    action = selected.preferred_action
    assert action is not None
    if step == 6:
        assert action["tool"] == "commit_state"
        verification = observation["last_verification"]
        assert "target_fixed" not in verification  # Cannot rely on a stripped private shortcut.
        assert verification["physical_constraints_ok"] is False
        proof = verification["hif_meter_nonregression"]
        assert proof["operator_review_required"] and proof["physical_fault_still_present"]
        candidate = env.store.get_state(env.current_candidate_id)
        assert candidate["candidate_disposition"] == "ACCEPT_PARTIAL"
        assert env.candidate_decision_evidence(env.current_candidate_id)["sufficient"]
        for field in ("parent_state_hash", "candidate_state_hash"):
            tampered = deepcopy(observation)
            tampered["last_verification"]["hif_meter_nonregression"][field] = "stale"
            refused = select_observable_expert_actions(policy_observation=tampered, expert_oracle=expert)
            assert refused.preferred_action["tool"] != "commit_state"
        missing = deepcopy(observation)
        missing["last_verification"].pop("hif_meter_nonregression")
        assert select_observable_expert_actions(policy_observation=missing, expert_oracle=expert).preferred_action["tool"] != "commit_state"
    env.assert_training_decision_evidence(action)
    _, output = env.step(action)
    assert output["execution_status"] == "success", output
    tools.append(action["tool"])
    history.append({"action": action, "tool_output": output})
    if env.terminal:
        break
assert env.terminal_outcome == "operator_escalation"
assert tools == ["run_three_phase_nlm_from_path", "estimate_hif_location_magnitude_multiscan_from_path",
                 "run_wls", "get_measurement_context", "correct_measurements", "run_wls", "commit_state",
                 "get_measurement_context", "ask_for_more_evidence"]
assert all(call["origin"] == "cached_fresh_observable_fit" for call in cache.calls)
after = env.store.get_state(env.store.active_state_id)["measurements"]
before = row["execution"]["measurements"]
assert [i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]] == [76]
print(json.dumps({"steps": len(tools), "outcome": env.terminal_outcome, "changed_indices": [76]}))
'''
    completed = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True,
                               text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr[-6000:] + completed.stdout[-3000:]
    assert json.loads(completed.stdout.splitlines()[-1])["steps"] == 9
