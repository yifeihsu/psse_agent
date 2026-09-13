from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.stats import chi2

from scripts.validate_ieee57_disturbances import (
    PROFILES, circuit_physics, instrument_observations, make_flat_case, screen_wls,
)


def _telemetry():
    """Dimensionally complete telemetry with conspicuous offline-only fields."""
    return {
        "measurement_vector": [1.0] * 57 + [0.0] * (491 - 57),
        "three_phase_voltages": [{
            "external_bus": row + 1, "bus": f"b{row+1}", "row0": row,
            "vln_pu_rect": [[1.0, 0.0], [-0.5, -np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]],
            "true_fault_bus": "private_target", "reference_voltage": "private_target",
        } for row in range(57)],
        "three_phase_branch_currents": [{
            "asset_id": f"case57:branch:{row+1:04d}", "branch_row0": row,
            "from_bus": row % 56 + 1, "to_bus": row % 56 + 2,
            "i_from_pu_rect": [[0.1, 0.0], [-0.05, -0.08], [-0.05, 0.08]],
            "i_to_pu_rect": [[-0.1, 0.0], [0.05, 0.08], [0.05, -0.08]],
            "hidden_node": "private_target", "fault_resistance": "private_target",
        } for row in range(80)],
        "truth": {"family": "hif", "bus": "private_target"},
        "injection": {"fault_element": "Fault.private_target"},
        "source_injection": {"reference_slack_power_pu": "private_target"},
    }


def test_instrument_allowlist_excludes_offline_truth_and_never_mutates_input():
    original = _telemetry()
    before = deepcopy(original)
    observed = instrument_observations(original, seed=19)
    assert original == before
    assert set(observed) == {"measurement_vector", "three_phase_voltages", "three_phase_branch_currents"}
    assert set(observed["three_phase_voltages"][0]) == {"external_bus", "bus", "row0", "vln_pu_rect"}
    assert set(observed["three_phase_branch_currents"][0]) == {
        "asset_id", "branch_row0", "from_bus", "to_bus", "i_from_pu_rect", "i_to_pu_rect",
    }
    assert "private_target" not in json.dumps(observed)
    assert observed["measurement_vector"] == original["measurement_vector"]


def test_noise_is_reproducible_profiles_share_draws_and_scada_weights_stay_fixed():
    raw = _telemetry()
    nominal = instrument_observations(raw, seed=[20260911, 4], profile=PROFILES["nominal"])
    repeat = instrument_observations(raw, seed=[20260911, 4], profile=PROFILES["nominal"])
    precise = instrument_observations(raw, seed=[20260911, 4], profile=PROFILES["precision_sensitivity"])
    assert nominal == repeat
    assert nominal["measurement_vector"] == precise["measurement_vector"]
    for family, fields in (("three_phase_voltages", ("vln_pu_rect",)),
                          ("three_phase_branch_currents", ("i_from_pu_rect", "i_to_pu_rect"))):
        for field in fields:
            base = np.asarray([row[field] for row in raw[family]])
            delta_nom = np.asarray([row[field] for row in nominal[family]]) - base
            delta_precise = np.asarray([row[field] for row in precise[family]]) - base
            np.testing.assert_allclose(delta_precise, 0.1 * delta_nom, atol=1e-15, rtol=1e-9)
    samples = np.asarray([
        instrument_observations(raw, seed=seed, profile=PROFILES["nominal"])["measurement_vector"]
        for seed in range(128)
    ]) - np.asarray(raw["measurement_vector"])
    assert np.std(samples[:, :57]) == pytest.approx(.001, rel=.05)
    assert np.std(samples[:, 57:]) == pytest.approx(.01, rel=.03)


@pytest.mark.parametrize("malformation", ("short_vector", "nested_vector", "nan_vector", "short_phasor", "infinite_phasor"))
def test_instrument_rejects_incomplete_or_nonfinite_sensor_evidence(malformation):
    raw = _telemetry()
    if malformation == "short_vector":
        raw["measurement_vector"].pop()
    elif malformation == "nested_vector":
        raw["measurement_vector"] = [raw["measurement_vector"]]
    elif malformation == "nan_vector":
        raw["measurement_vector"][90] = float("nan")
    elif malformation == "short_phasor":
        raw["three_phase_voltages"][0]["vln_pu_rect"].pop()
    else:
        raw["three_phase_branch_currents"][0]["i_to_pu_rect"][1][0] = float("inf")
    with pytest.raises(ValueError):
        instrument_observations(raw, seed=12, profile=PROFILES["nominal"])


@pytest.mark.parametrize("sigma", (-1.0, float("nan"), float("inf")))
def test_instrument_rejects_invalid_noise_standard_deviation(sigma):
    with pytest.raises(ValueError):
        instrument_observations(_telemetry(), seed=12, profile={"voltage_sigma_pu": sigma, "current_sigma_pu": 1e-3})


@pytest.fixture(scope="module")
def ieee57_snapshot(tmp_path_factory):
    from psse_env.systems import resolve_system
    from three_phase_model.exporter import export_model
    from three_phase_model.measurements import extract_measurements
    from three_phase_model.runtime import compile_model

    directory = tmp_path_factory.mktemp("disturbance_validation")
    built = export_model(resolve_system("case57").load_case(), directory / "model", case_id="case57")
    dss = compile_model(directory / "model" / "Master.dss")
    measurements = extract_measurements(dss, built["registry"], built["assumptions"])
    return directory, built, measurements


def test_flat_case_preserves_network_and_real_wls_uses_both_alarm_tests(ieee57_snapshot):
    from mcp_server.matpower_server import _load_python_case
    from psse_env.providers.matpower import MatpowerDeploymentProviders

    directory, built, measurements = ieee57_snapshot
    reference = built["reference"]
    original = deepcopy(reference)
    path = make_flat_case(reference, directory)
    parsed = _load_python_case(str(path))
    np.testing.assert_equal(reference["bus"], original["bus"])
    np.testing.assert_equal(parsed["bus"][:, 7], np.ones(57))
    np.testing.assert_equal(parsed["bus"][:, 8], np.zeros(57))
    np.testing.assert_allclose(parsed["branch"][:, :13], reference["branch"][:, :13], atol=1e-11, rtol=0)
    np.testing.assert_allclose(parsed["bus"][:, 2:6], reference["bus"][:, 2:6], atol=1e-11, rtol=0)
    provider = MatpowerDeploymentProviders(chi2_alpha=.05, normalized_residual_threshold=4.0)
    observed = instrument_observations(measurements, seed=1)
    clean = screen_wls(provider, path, observed)
    assert clean["converged"] is True
    assert clean["chi_square_dof"] == 378
    assert clean["chi_square_threshold"] == pytest.approx(chi2.ppf(.95, 378), rel=1e-12)
    assert clean["normalized_residual_threshold"] == 4.0
    assert clean["anomaly_detection_rule"] == "chi_square_or_normalized_residual"
    assert clean["alarm"] is False
    observed["measurement_vector"][57] += .1
    disturbed = screen_wls(provider, path, observed)
    assert disturbed["converged"] is True
    assert disturbed["normalized_residual_alarm"] is True
    assert disturbed["chi_square_alarm"] is False
    assert disturbed["alarm"] is True


def test_failed_wls_is_unknown_not_a_negative_alarm():
    class FailedProvider:
        def run_wls(self, state):
            return {"execution_status": "failure", "error_code": "wls_failure", "error_detail": "unconverged"}

    result = screen_wls(FailedProvider(), "unused.m", {"measurement_vector": [0.0]})
    assert result["alarm"] is None
    assert result["error_code"] == "wls_failure"


def test_full_circuit_physics_covers_hif_hidden_nodes_and_actual_unbalanced_loads(ieee57_snapshot):
    from three_phase_model.disturbances import inject_midspan_hif
    from three_phase_model.runtime import compile_model, redistribute_load

    directory, built, _ = ieee57_snapshot
    registry, assumptions = built["registry"], built["assumptions"]
    dss = compile_model(directory / "model" / "Master.dss")
    healthy = circuit_physics(dss, registry, assumptions)
    assert healthy["passed"]
    assert healthy["phase_node_count"] == 171
    assert healthy["enabled_fault_count"] == 0
    redistribute_load(dss, registry, bus=12, delta=.2)
    unbalanced = circuit_physics(dss, registry, assumptions)
    assert unbalanced["passed"]
    assert unbalanced["checks"]["constant_pq_device_power_pu"]["passed"]
    dss = compile_model(directory / "model" / "Master.dss")
    inject_midspan_hif(dss, registry, assumptions, branch_row0=0, phase=3, alpha=.2, resistance_pu=10)
    faulted = circuit_physics(dss, registry, assumptions)
    assert faulted["passed"]
    assert faulted["phase_node_count"] == 174
    assert faulted["enabled_fault_count"] == 1
    assert faulted["checks"]["fault_ohms_law_current_pu"]["passed"]
