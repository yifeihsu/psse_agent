"""Read-only routing, invalid screens, and identical expert/learner evidence."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mcp_server.matpower_server import _load_python_case
from psse_env.actions import preferred_first_request
from psse_env.dagger.dataset_builder import prepare_model_policy_observation, summarize_history
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.transactional_env import TransactionalPSSEEnv
from research.gnn_screen.feature_schema import FAMILY_NAMES
from research.gnn_screen.model import WLSScreenGNN
from research.gnn_screen.protocol_adapter import FrozenScreen, snapshot_binding, unavailable_report
from research.gnn_screen.wls_features import configured_case, default_measurement_sigma
from tools.lagrangian_port import lagrangian_m_singlephase_details


def constant_screen(*, phase=5.0, anomaly=5.0, mask=None):
    model = WLSScreenGNN(hidden_dim=8, layers=1, dropout=0)
    def forward(batch):
        count = len(batch["u"])
        return {"phase_screen_logit": torch.full((count,), phase),
                "anomaly_logit": torch.full((count,), anomaly),
                "family_logits": torch.zeros((count, 5))}
    model.forward = forward
    checkpoint = {"model_id": "test", "family_names": FAMILY_NAMES,
                  "trained_heads": {"phase": True, "anomaly": True},
                  "trained_family_mask": mask or [True] * 5}
    calibration = {"model_id": "test", "phase_threshold": .9, "anomaly_threshold": .9,
                   "threshold_comparison": ">"}
    return FrozenScreen(model, None, checkpoint, calibration)


def test_scores_are_hypotheses_and_binding_tracks_measurement_and_model():
    case = _load_python_case("case14")
    z = build_measurement_vector(case)
    screen = constant_screen(mask=[True, True, False, False, False])
    report = screen.screen(case, z, state_id="s0", state_hash="h0")
    assert report["screen_status"] == "valid", report
    assert report["phase_trigger"] is True
    assert report["recommendation"] == "acquire_three_phase_context"
    assert set(report["family_scores"]) == {"hif", "unbalance"}
    assert report["read_only"] and "harmonic" in report["unsupported_families"]
    assert "probabilities" in report["score_interpretation"]
    assert not {"target_fixed", "globally_resolved", "faulted_phase", "resistance"} & report.keys()
    changed = z.copy()
    changed[0] += .001
    assert snapshot_binding(case, z)["measurement_window_hash"] != snapshot_binding(case, changed)["measurement_window_hash"]
    changed_case = copy.deepcopy(case)
    changed_case["branch"][0, 2] *= 1.01
    assert snapshot_binding(case, z)["configured_model_hash"] != snapshot_binding(changed_case, z)["configured_model_hash"]


def test_invalid_and_failed_wls_never_emit_negative_screen():
    case = _load_python_case("case14")
    z = build_measurement_vector(case)
    screen = constant_screen()
    for values, details in ((z[:2], None), (z, {"success": False})):
        report = screen.screen(case, values, wls_details=details)
        assert report["screen_status"] != "valid"
        assert report["phase_trigger"] is None and report["anomaly_trigger"] is None
    z[0] = np.nan
    assert screen.screen(case, z)["screen_status"] == "unsupported_input"


def test_report_binds_actual_covariance_of_supplied_wls():
    case = configured_case(_load_python_case("case14"))
    z = build_measurement_vector(case)
    screen = constant_screen()
    sigma = default_measurement_sigma(len(case["bus"]), len(case["branch"]))
    reports = []
    for multiplier in (1, 2):
        details = lagrangian_m_singlephase_details(
            z, case, 0, case["bus"], measurement_sigma=sigma * multiplier)
        reports.append(screen.screen(case, z, wls_details=details))
    assert all(r["screen_status"] == "valid" for r in reports), reports
    assert reports[0]["measurement_window_hash"] == reports[1]["measurement_window_hash"]
    assert reports[0]["covariance_hash"] != reports[1]["covariance_hash"]
    assert reports[0]["solver_settings"] == {"max_it": 20, "tol": 1e-5}


def test_negative_screen_preserves_independent_coverage():
    case = _load_python_case("case14")
    report = constant_screen(phase=-5., anomaly=-5.).screen(case, build_measurement_vector(case))
    assert report["phase_trigger"] is False
    assert "retain_independent_coverage" in report["recommendation"]
    report = constant_screen(phase=-5.).screen(case, build_measurement_vector(case))
    assert report["recommendation"] == "continue_balanced_investigation"


def test_phase_screen_routes_negative_wls_without_acquiring_telemetry_or_certifying_fault():
    case = _load_python_case("case14")
    # The learned screen is historical auxiliary behaviour: the strict
    # profiles refuse a screen checkpoint by construction.
    provider = MatpowerDeploymentProviders(
        evidence_profile=AUXILIARY_EVIDENCE_PROFILE, screen_checkpoint="test.pt", screen_calibration="cal.json")
    baseline = MatpowerDeploymentProviders(evidence_profile=AUXILIARY_EVIDENCE_PROFILE)
    z = build_measurement_vector(case).tolist()
    scenario = {"case": "case14", "measurements": z, "metadata": {}}
    # Use real WLS evidence; only neural scores are mocked in this protocol test.
    screen = constant_screen()
    with patch("research.gnn_screen.protocol_adapter.load_screen", return_value=screen):
        env = TransactionalPSSEEnv(**provider.env_kwargs(), production_dataset_mode=True, max_steps=12)
        env.reset(scenario)
        active = env.store.active_state_id
        _, output = env.step({"tool": "run_wls", "arguments": {"state_id": active}})
        assert output["execution_status"] == "success", output
        metrics = output["tool_metrics"]
        assert metrics["gnn_screen"]["screen_status"] == "valid", metrics["gnn_screen"]
        assert metrics["chi_square_alarm"] is False
        clean = baseline.run_wls({"case": "case14", "measurements": z})
        assert metrics["globally_resolved"] == clean["globally_resolved"]
        assert metrics["remaining_anomaly_score"] == pytest.approx(clean["remaining_anomaly_score"])
        observation = env.get_policy_observation().as_dict()
        assert observation["available_evidence"] == []
        assert "wls_gnn_phase_investigation" in observation["unresolved_signatures"]
        premature = {"tool": "finalize_diagnosis", "arguments": {}}
        with pytest.raises(ValueError, match="pending GNN investigation"):
            env.assert_training_decision_evidence(premature)
        verdict = env.process_oracle.check(env.get_oracle_state(), premature, store=env.store)
        assert verdict["process_valid"] is False
        learner, _ = prepare_model_policy_observation(observation)
        assert learner["fresh_context_evidence"]["wls"]["gnn_screen"]["phase_trigger"] is True
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        action = oracle.next_actions(env.get_oracle_state(), env.history)[0]
        assert action["tool"] == "get_three_phase_context"
        env.assert_training_decision_evidence(action)
        _, result = env.step(action)
        assert result["execution_status"] == "success", result
        assert result["tool_metrics"]["three_phase_context_status"] == "unavailable"
        assert not env.get_policy_observation().explained_anomalies


def test_no_checkpoint_preserves_default_and_missing_artifact_is_unavailable():
    case = _load_python_case("case14")
    state = {"case": "case14", "measurements": build_measurement_vector(case).tolist()}
    assert "gnn_screen" not in MatpowerDeploymentProviders().run_wls(state)
    provider = MatpowerDeploymentProviders(
        evidence_profile=AUXILIARY_EVIDENCE_PROFILE, screen_checkpoint="missing.pt", screen_calibration="missing.json")
    report = provider.run_wls(state)["gnn_screen"]
    assert report["screen_status"] == "model_unavailable"
    assert report["phase_trigger"] is None


def test_stale_screen_cannot_override_acquisition_order():
    evidence = {"successful": True, "state_id": "s0", "state_hash": "h0", "anomaly_breadth": .9,
                "gnn_screen": {"screen_status": "valid", "phase_trigger": True,
                               "state_id": "s0", "state_hash": "old"}}
    state = {"active_state_id": "s0", "evidence_profile": AUXILIARY_EVIDENCE_PROFILE,
             "fresh_context_evidence": {"wls": evidence}}
    assert preferred_first_request(state) == "get_harmonic_context"
    evidence["gnn_screen"]["state_hash"] = "h0"
    assert preferred_first_request(state) == "get_three_phase_context"


def test_history_retains_binding_and_calibration_policy():
    report = {**unavailable_report("unsupported_input", "bad input"),
              "state_id": "s0", "state_hash": "h0", "model_id": "model",
              "measurement_window_hash": "window", "configured_model_hash": "network"}
    history = [{"action": {"tool": "run_wls", "arguments": {"state_id": "s0"}},
                "tool_output": {"tool_metrics": {"gnn_screen": report}}}]
    assert "configured_model_hash" in str(summarize_history(history))
