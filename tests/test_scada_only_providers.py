from __future__ import annotations

import copy

import numpy as np
import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector


def snapshot():
    z = build_measurement_vector(_load_python_case("case14"))
    z[25] += .2
    return {
        "case": "case14", "state_id": "s0", "state_hash": "hash0",
        "measurements": z.tolist(),
        "metadata": {"sigma_z": [.001] * 14 + [.01] * 108,
            "three_phase_voltages": [{"unavailable_sensor": 999}],
            "three_phase_branch_currents": [{"unavailable_sensor": 888}],
            "harmonic_measurements": [{"unavailable_sensor": 777}],
            "hif_runtime": {"z_obs": [0.] * 122, "op_point": {"load_scale": .1}},
            "hif_scan_window": {"scans": [{"label": "HIF"}]},
            "nlm_diagnostic": {"success": True, "top_hif_groups": [{"branch_row0": 2}]},
            "substation_telemetry": {"reported_statuses": {"hidden": True}},
            "hidden_truth": {"true_family": "hif"}},
        "policy_observation": {
            "unresolved_signatures": ["hif_suspected_zero_sequence"],
            "explained_anomalies": [{"family": "hif", "detail": {"conditioning_fit": {"success": True}}}],
        },
    }


def test_wls_is_invariant_to_every_auxiliary_stream_and_does_not_compensate(monkeypatch):
    provider = MatpowerDeploymentProviders(normalized_residual_threshold=4.)
    state = snapshot()
    original = copy.deepcopy(state)
    def forbidden(*args, **kwargs):
        raise AssertionError("Strict WLS must not invoke HIF replay")
    monkeypatch.setattr("psse_env.providers.matpower.conditioned_prediction", forbidden)
    contaminated = provider.run_wls(state)
    clean_input = copy.deepcopy(state)
    clean_input["metadata"] = {"sigma_z": state["metadata"]["sigma_z"]}
    clean_input["policy_observation"] = {}
    clean = provider.run_wls(clean_input)
    assert contaminated == clean
    assert contaminated["chi_square_alarm"] or contaminated["normalized_residual_alarm"]
    assert "hif_conditioning" not in contaminated
    assert not any("hif" in item for item in contaminated["unresolved_signatures"])
    assert state == original
    np.testing.assert_array_equal(provider._solve(state)["wls_measurements"], state["measurements"])


@pytest.mark.parametrize("method", ["get_three_phase_context", "get_harmonic_context", "run_hse",
                                   "run_three_phase_nlm", "estimate_hif", "estimate_hif_multiscan"])
def test_unavailable_auxiliary_tools_never_consume_hidden_data(method):
    state = snapshot()
    result = getattr(MatpowerDeploymentProviders(), method)(state, {"arguments": {"candidate_branch_row0": 2}})
    assert result["execution_status"] == "failure"
    assert result["error_code"] == "evidence_unavailable_in_scada_only_profile"
    assert "anomaly_explanation" not in result


def test_auxiliary_provider_cannot_override_strict_controller():
    provider = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics")
    state = snapshot()
    state["evidence_profile"] = "scada_only"
    result = provider.estimate_hif_multiscan(state, {"arguments": {"candidate_branch_row0": 2}})
    assert result["execution_status"] == "failure"


def test_scada_parameter_solver_receives_only_observed_history(monkeypatch):
    state = snapshot()
    state["metadata"]["parameter_scans"] = {
        "z_scans": [state["measurements"], state["measurements"]],
        "sigma_z": state["metadata"]["sigma_z"],
        "op_point": {"hidden_load_scale": 1234},
        "initial_states": [[999.] * 27] * 2,
        "three_phase_branch_currents": [{"secret": 99}],
    }
    captured = {}
    def solver(case_path, line, scans, initial_states, **kwargs):
        captured.update(scans=scans, initial_states=initial_states, kwargs=kwargs)
        return {"success": False, "error": "fixture_no_mutation"}
    monkeypatch.setattr("psse_env.providers.matpower._param_correction_json", solver)
    provider = MatpowerDeploymentProviders()
    result = provider.correct_parameters(state, {"arguments": {"line_index": 1}})
    assert result["execution_status"] == "failure"
    assert captured["scans"] == state["metadata"]["parameter_scans"]["z_scans"]
    assert np.max(np.asarray(captured["initial_states"])) < 999.
    assert "op_point" not in captured["kwargs"]


def test_env_bundle_declares_scada_only_by_default():
    provider = MatpowerDeploymentProviders()
    assert provider.env_kwargs()["evidence_profile"] == "scada_only"
    with pytest.raises(ValueError, match="evidence_profile"):
        MatpowerDeploymentProviders(evidence_profile="flagged")
