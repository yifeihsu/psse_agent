"""Covariance declarations must reach numerical solves without reinterpretation."""
from unittest.mock import patch

import numpy as np
import pytest
from pypower.api import case14, ppoption, runpf

from Transmission.generate_measurements import compute_measurements_pu
from mcp_server.matpower_server import _wls_json
from psse_env.noise_contract import resolve_state_measurement_noise
from psse_env.providers.matpower import MatpowerDeploymentProviders


def test_resolver_rejects_undeclared_zero_variance_and_conflicting_weights():
    with pytest.raises(ValueError, match="Zero measurement sigma"):
        resolve_state_measurement_noise({"metadata": {"sigma_z": [0., .1]}}, 2)
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_state_measurement_noise({"sigma_z": [.1, .2], "metadata": {"sigma_z": [.1, .3]}}, 2)


def test_resolver_rejects_correlations_in_diagonal_solver_instead_of_dropping_them():
    with pytest.raises(ValueError, match="correlated"):
        resolve_state_measurement_noise({"metadata": {"sigma_z": [.1, .1], "operator_noise": {
            "measurement_covariance": [[.01, .002], [.002, .01]],
        }}}, 2)


def test_scaled_sensor_variance_reaches_runtime_wls():
    solved, ok = runpf(case14(), ppoption(VERBOSE=0, OUT_ALL=0))
    assert ok
    nominal = np.r_[np.full(14, .001), np.full(108, .01)]
    sigma = 1.7 * nominal
    observed = compute_measurements_pu(solved) + np.random.default_rng(731).normal(size=122) * sigma
    state = {"state_id": "scaled", "case": "case14", "measurements": observed.tolist(),
             "metadata": {"sigma_z": sigma.tolist()}, "policy_observation": {}}
    provider = MatpowerDeploymentProviders()
    actual = provider.run_wls(state)
    weighted = _wls_json("case14", observed.tolist(), measurement_sigma=sigma.tolist())
    wrong_nominal = _wls_json("case14", observed.tolist())
    assert weighted["success"] and wrong_nominal["success"]
    assert actual["chi_square_statistic"] == pytest.approx(weighted["global_residual_sum"])
    assert wrong_nominal["global_residual_sum"] / weighted["global_residual_sum"] == pytest.approx(1.7**2, rel=1e-8)
    np.testing.assert_array_equal(state["metadata"]["sigma_z"], sigma)


def test_meter_corrector_receives_variance_not_standard_deviation():
    sigma = np.r_[np.full(14, .002), np.full(108, .017)]
    state = {"state_id": "scaled", "case": "case14", "measurements": [1.] * 122,
             "metadata": {"sigma_z": sigma.tolist()}}
    with patch("psse_env.providers.matpower._meas_correction_json", return_value={"success": False}) as correct:
        MatpowerDeploymentProviders().correct_measurements(state, {"arguments": {"suspect_group": [7]}})
    np.testing.assert_array_equal(correct.call_args.kwargs["R_variances_full"], sigma**2)


def test_parameter_scans_keep_their_explicit_covariance():
    sigma = np.r_[np.full(14, .002), np.full(108, .02)]
    scans = {"sigma_z": sigma.tolist()}
    result = MatpowerDeploymentProviders._parameter_noise_options(
        {"metadata": {}}, scans, [[1.] * 122, [1.] * 122])
    np.testing.assert_array_equal(result["R_variances_full"], sigma**2)


def test_round0_rejects_weight_only_waveform_inputs_without_noise_provenance():
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator, ScenarioRejected
    source = {"id": "unverified", "z_obs": [1.] * 122,
              "sigma_z": [.001] * 14 + [.01] * 108, "three_phase_sigma": .005}
    with pytest.raises(ScenarioRejected, match="Weighting sigmas alone"):
        Round0ScenarioGenerator(validate=False)._aligned_waveform_row(source, "hif")


def test_single_scan_hif_numerics_consume_declared_scada_sigma():
    from three_phase_nlm.hif_parameter_estimator import (
        estimate_hif_location_magnitude, simulate_hif_candidate,
    )
    simulation = simulate_hif_candidate(candidate_branch_row0=0, alpha=.05, phase="A", r_hif_pu=5.)
    sigma = np.r_[np.full(14, .001), np.full(108, .01)]
    observed = np.asarray(simulation["z"]) + np.random.default_rng(67).normal(size=122) * sigma
    kwargs = dict(candidate_branch_row0=0, candidate_phase="A", z_obs=observed,
                  alpha_grid_size=2, r_grid_size=2, refine_top_n=0, seed_from_terminal_currents=False)
    first = estimate_hif_location_magnitude(**kwargs, sigma_z=sigma)
    second = estimate_hif_location_magnitude(**kwargs, sigma_z=1.7 * sigma)
    assert first["success"] and second["success"]
    assert first["fit"]["weighted_residual_norm"] / second["fit"]["weighted_residual_norm"] == pytest.approx(1.7)
