"""Read-only balanced investigation from a separately calibrated anomaly head."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE
from psse_env.actions import action_signature, gnn_investigation_pending
from psse_env.dagger.dataset_builder import prepare_model_policy_observation
from psse_env.oracle import ExpertPolicyOracle
from psse_env.oracle.diagnostics_expert import DiagnosticsExpert
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.transactional_env import TransactionalPSSEEnv
from research.gnn_screen.tests.test_protocol_adapter import constant_screen


def screened_state():
    return {
        "active_state_id": "s0", "has_open_candidate": False,
        "evidence_profile": AUXILIARY_EVIDENCE_PROFILE,
        "unresolved_signatures": [], "tried_action_signatures": [],
        "fresh_context_evidence": {"wls": {
            "successful": True, "state_id": "s0", "state_hash": "hash0",
            "gnn_screen": {
                "screen_status": "valid", "state_id": "s0", "state_hash": "hash0",
                "phase_trigger": False, "anomaly_trigger": True,
                "family_scores": {"measurement": .3, "parameter": .7, "topology": .9},
                "disabled_family_heads": [],
            },
        }},
    }


def requested_tool(state):
    proposals = DiagnosticsExpert().gnn_balanced_screening_proposals(state)
    return proposals[0].action["tool"] if proposals else None


def test_family_scores_order_only_read_only_contexts_once_each():
    state = screened_state()
    assert requested_tool(state) == "get_topology_context"
    state["fresh_context_evidence"]["topology"] = {"state_id": "s0", "state_hash": "hash0"}
    assert requested_tool(state) == "get_parameter_context"
    state["tried_action_signatures"].append(action_signature({"tool": "get_parameter_context", "arguments": {"state_id": "s0"}}))
    assert requested_tool(state) == "get_measurement_context"
    state["fresh_context_evidence"]["measurement"] = {"state_id": "s0", "state_hash": "hash0"}
    assert requested_tool(state) is None
    assert state["unresolved_signatures"] == []


@pytest.mark.parametrize("change", [
    "no_screen", "failed_wls", "old_state", "old_hash", "missing_hash", "invalid_screen",
    "phase_positive", "phase_unknown", "anomaly_negative", "anomaly_unknown", "candidate", "waveform",
])
def test_only_current_valid_balanced_positive_screen_routes(change):
    state = screened_state()
    wls = state["fresh_context_evidence"]["wls"]
    screen = wls["gnn_screen"]
    if change == "no_screen":
        wls.pop("gnn_screen")
    elif change == "failed_wls":
        wls["successful"] = False
    elif change == "old_state":
        state["active_state_id"] = "s1"
    elif change == "old_hash":
        screen["state_hash"] = "old"
    elif change == "missing_hash":
        screen.pop("state_hash")
        wls.pop("state_hash")
    elif change == "invalid_screen":
        screen["screen_status"] = "model_failure"
    elif change == "phase_positive":
        screen["phase_trigger"] = True
    elif change == "phase_unknown":
        screen["phase_trigger"] = None
    elif change == "anomaly_negative":
        screen["anomaly_trigger"] = False
    elif change == "anomaly_unknown":
        screen["anomaly_trigger"] = None
    elif change == "candidate":
        state["has_open_candidate"] = True
    else:
        state["unresolved_signatures"] = ["three_phase_unbalance"]
    assert requested_tool(state) is None


def test_disabled_heads_cannot_rank_and_stale_contexts_do_not_suppress_requests():
    state = screened_state()
    state["fresh_context_evidence"]["wls"]["gnn_screen"]["disabled_family_heads"] = ["topology"]
    state["fresh_context_evidence"]["parameter"] = {"state_id": "s0", "state_hash": "old"}
    state["tried_action_signatures"] = [action_signature({"tool": "get_parameter_context", "arguments": {"state_id": "old"}})]
    assert requested_tool(state) == "get_parameter_context"


def test_acquired_waveform_diagnosis_supersedes_screening_context_obligation():
    state = screened_state()
    signature = "three_phase_unbalance"
    state["unresolved_signatures"] = [signature]
    assert gnn_investigation_pending(state)
    assert requested_tool(state) is None
    state["explained_anomalies"] = [{"explained_signatures": [signature]}]
    assert not gnn_investigation_pending(state)
    state["fresh_context_evidence"]["wls"]["gnn_screen"]["phase_trigger"] = True
    assert not gnn_investigation_pending(state)


def test_anomaly_only_screen_prevents_expert_immediate_wls_based_finalization():
    case = _load_python_case("case14")
    provider = MatpowerDeploymentProviders(evidence_profile=AUXILIARY_EVIDENCE_PROFILE, screen_checkpoint="test.pt", screen_calibration="cal.json")
    screen = constant_screen(phase=-5., anomaly=5.)
    with patch("research.gnn_screen.protocol_adapter.load_screen", return_value=screen):
        env = TransactionalPSSEEnv(**provider.env_kwargs(), production_dataset_mode=True, max_steps=12)
        env.reset({"case": "case14", "measurements": build_measurement_vector(case).tolist(), "metadata": {}})
        _, output = env.step({"tool": "run_wls", "arguments": {"state_id": env.store.active_state_id}})
        metrics = output["tool_metrics"]
        assert metrics["no_material_anomaly_remaining"] is True
        assert metrics["unresolved_signatures"] == []
        observation = env.get_policy_observation().as_dict()
        learner, _ = prepare_model_policy_observation(observation)
        assert learner["fresh_context_evidence"]["wls"]["gnn_screen"]["anomaly_trigger"] is True
        unsupported_correction = {"tool": "correct_parameters", "arguments": {
            "state_id": env.store.active_state_id, "line_index": 1, "r": .1,
        }}
        assert not env.process_oracle.check(env.current_state(), unsupported_correction, store=env.store)["process_valid"]
        assert gnn_investigation_pending(env.current_state())
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        requested = []
        for _ in range(3):
            action = oracle.next_actions(env.get_oracle_state(), env.history)[0]
            assert action["tool"] in {"get_measurement_context", "get_parameter_context", "get_topology_context"}
            requested.append(action["tool"])
            env.assert_training_decision_evidence(action)
            _, result = env.step(action)
            assert result["execution_status"] == "success", result
        assert len(set(requested)) == 3
        assert not gnn_investigation_pending(env.current_state())
        assert not env.get_policy_observation().accepted_corrections
        assert not env.get_policy_observation().explained_anomalies
