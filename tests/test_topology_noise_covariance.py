from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pypower.api import case14, ppoption, runpf

from Transmission.ieee14_full_measurements import fixed_operator_layout
from Transmission.ieee14_full_substation import (
    TELEMETRY_SIGMA, add_telemetry_noise, injection_metered_nodes,
    operator_model_from_map, operator_noise_for_layout, operator_observation_for_layout,
    operator_sigma_for_layout, operator_vector_for_layout,
)
from Transmission.ieee14_full_topology import build_full_topology
from Transmission.generate_measurements import compute_measurements_pu
from tools.lagrangian_port import lagrangian_m_singlephase_details


@pytest.fixture(scope="module")
def meter_setup():
    reference = case14()
    model = build_full_topology()
    _, layout = operator_model_from_map(reference, model, {})
    nodes = list(model.nodes)
    metered = injection_metered_nodes(model, reference)
    # Covariance does not depend on the means. Use a simple deterministic meter
    # snapshot on the real physical deployment, without claiming a power flow.
    mean = {
        "node_vm": {node: 1.0 for node in nodes},
        "node_pinj": {node: 0.0 for node in metered},
        "node_qinj": {node: 0.0 for node in metered},
        **{key: [0.0] * 20 for key in ("branch_pf", "branch_qf", "branch_pt", "branch_qt")},
        "cb_p": {}, "cb_q": {}, "sigma": dict(TELEMETRY_SIGMA),
        "nominal_sensor_sigma": dict(TELEMETRY_SIGMA),
        "measurement_kind": "noiseless_mean", "noise_draw_count": 0,
        "applied_noise_scale": 0.0, "dead_nodes": [],
    }
    return reference, model, layout, mean


def test_operator_noise_keeps_sum_variance_and_structural_constraints(meter_setup):
    _, _, layout, mean = meter_setup
    before = deepcopy(mean)
    observed = add_telemetry_noise(mean, np.random.default_rng(6), scale=1.7)
    result = operator_observation_for_layout(observed, layout)
    sigma = np.array(result["measurement_sigma"])
    assert sigma.shape == (122,)
    assert sigma[16] == pytest.approx(np.sqrt(2) * 0.017)  # Pinj, bus 3.
    assert sigma[30] == pytest.approx(np.sqrt(2) * 0.017)  # Qinj, bus 3.
    assert result["structural_zero_indices"] == [20, 34]
    assert sigma[20] == sigma[34] == 0
    assert result["measurements"][20] == result["measurements"][34] == 0
    covariance = np.asarray(result["measurement_covariance"])
    np.testing.assert_array_equal(covariance, np.diag(sigma ** 2))
    np.testing.assert_array_equal(result["applied_noise_covariance"], covariance)
    np.testing.assert_array_equal(operator_sigma_for_layout(observed, layout), sigma)
    assert len(set(result["measurement_ids"])) == 122
    assert set(result["measurement_sources"][16]) == {"Pinj:node:3B1", "Pinj:node:3B2"}
    assert result["measurement_ids"][42] == "Pf:branch_row0:0"
    assert mean == before
    mean_contract = operator_noise_for_layout(mean, layout)
    assert mean_contract["measurement_kind"] == "noiseless_mean"
    assert not np.any(mean_contract["applied_noise_covariance"])


def test_actual_draws_match_covariance_on_normal_fourteen_bus_layout(meter_setup):
    _, _, layout, mean = meter_setup
    rng = np.random.default_rng(20260916)
    initial = operator_vector_for_layout(mean, layout)
    draws = np.asarray([
        operator_vector_for_layout(add_telemetry_noise(mean, rng, scale=0.6), layout) - initial
        for _ in range(4000)
    ])
    expected = operator_sigma_for_layout(add_telemetry_noise(mean, rng, scale=0.6), layout)
    active = expected > 0
    np.testing.assert_allclose(draws[:, active].std(axis=0, ddof=1), expected[active], rtol=0.075)
    assert np.array_equal(draws[:, ~active], np.zeros((4000, 2)))
    assert abs(np.corrcoef(draws[:, 16], draws[:, 30])[0, 1]) < 0.05


def test_shared_sensor_reuse_produces_full_covariance_and_keeps_identity(meter_setup):
    reference, model, layout, mean = meter_setup
    sampled = add_telemetry_noise(mean, np.random.default_rng(1))
    repeated = deepcopy(layout)
    repeated["sections"]["2"]["meter_node"] = repeated["sections"]["1"]["meter_node"]
    result = operator_noise_for_layout(sampled, repeated)
    covariance = np.asarray(result["measurement_covariance"])
    assert covariance[0, 1] == pytest.approx(1e-6)
    assert result["measurement_ids"][0] == result["measurement_ids"][1]
    meters = {int(bus): section["meter_node"] for bus, section in layout["sections"].items()}
    fixed = fixed_operator_layout(reference, model, meters)
    np.testing.assert_array_equal(operator_vector_for_layout(sampled, fixed), operator_vector_for_layout(sampled, layout))
    assert operator_noise_for_layout(sampled, fixed)["measurement_ids"] == operator_noise_for_layout(sampled, layout)["measurement_ids"]


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_noisy_generation_rejects_invalid_scale(meter_setup, scale):
    with pytest.raises(ValueError, match="finite positive"):
        add_telemetry_noise(meter_setup[3], np.random.default_rng(1), scale=scale)


def test_noise_is_not_drawn_twice(meter_setup):
    observed = add_telemetry_noise(meter_setup[3], np.random.default_rng(1))
    with pytest.raises(ValueError, match="drawn once"):
        add_telemetry_noise(observed, np.random.default_rng(2))


@pytest.fixture(scope="module")
def exact_problem():
    solved, success = runpf(case14(), ppoption(VERBOSE=0, OUT_ALL=0))
    assert success
    z = compute_measurements_pu(solved)
    exact = np.array([20, 34])
    z[exact] = 0.0
    sigma = np.r_[np.full(14, 0.001), np.full(108, 0.01)]
    sigma[[16, 30]] *= np.sqrt(2)
    sigma[exact] = 0
    return solved, z, sigma, exact


def _solve(problem, z=None):
    case, mean, sigma, exact = problem
    return lagrangian_m_singlephase_details(
        mean if z is None else z, case, 0, case["bus"], measurement_sigma=sigma,
        exact_measurement_indices=exact, tol=1e-10, max_it=30,
    )


def test_exact_injections_use_constraints_and_correct_residual_covariance(exact_problem):
    case, mean, sigma, exact = exact_problem
    before = deepcopy(case)
    observed = mean + np.random.default_rng(8).normal(size=122) * sigma
    out = _solve(exact_problem, observed)
    assert out["success"]
    assert out["n_measurements"] == 120
    assert out["constraint_rank"] == 2
    assert out["effective_state_count"] == 25
    assert out["dof"] == 95
    assert out["exact_measurement_indices"].tolist() == [20, 34]
    assert set(out["measurement_rows"]) == set(range(122)) - {20, 34}
    assert np.max(np.abs(out["constraint_residual"])) < 1e-9
    omega = out["residual_covariance"]
    variances = out["measurement_variance_diag"]
    np.testing.assert_allclose(omega, omega.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(omega), out["residual_covariance_diag"], atol=1e-15)
    np.testing.assert_allclose((omega / variances[None, :]) @ omega, omega, atol=1e-13)
    assert np.sum(np.diag(omega) / variances) == pytest.approx(out["dof"], abs=1e-9)
    assert np.linalg.eigvalsh(omega).min() > -1e-14
    np.testing.assert_array_equal(case["bus"], before["bus"])
    np.testing.assert_array_equal(case["gen"], before["gen"])
    assert np.array_equal(observed[exact], np.zeros(2))


def test_constrained_covariance_predicts_measured_residual_sensitivity(exact_problem):
    _, mean, sigma, _ = exact_problem
    base = _solve(exact_problem)
    for external_row in (2, 16, 54):
        step = sigma[external_row] * 1e-3
        plus, minus = mean.copy(), mean.copy()
        plus[external_row] += step
        minus[external_row] -= step
        derivative = (_solve(exact_problem, plus)["raw_residual"] - _solve(exact_problem, minus)["raw_residual"]) / (2 * step)
        local_row = list(base["measurement_rows"]).index(external_row)
        expected = base["residual_covariance"][:, local_row] / sigma[external_row] ** 2
        np.testing.assert_allclose(derivative, expected, atol=2e-5, rtol=1e-4)


def test_zero_sigma_needs_genuine_explicit_zero_injection(exact_problem):
    case, mean, sigma, _ = exact_problem
    with pytest.raises(ValueError, match="declared structural"):
        lagrangian_m_singlephase_details(mean, case, 0, case["bus"], measurement_sigma=sigma)
    bad = mean.copy()
    bad[20] = 1e-8
    with pytest.raises(ValueError, match="observations must equal zero"):
        _solve(exact_problem, bad)
    for exact in ([0], [15]):
        sig = np.r_[np.full(14, 0.001), np.full(108, 0.01)]
        sig[exact] = 0
        with pytest.raises(ValueError, match="injection rows|structurally zero"):
            lagrangian_m_singlephase_details(mean, case, 0, case["bus"], measurement_sigma=sig,
                                             exact_measurement_indices=exact)


def test_empty_exact_constraint_option_preserves_default_results(exact_problem):
    case, mean, _, _ = exact_problem
    default = lagrangian_m_singlephase_details(mean, case, 0, case["bus"])
    explicit_empty = lagrangian_m_singlephase_details(mean, case, 0, case["bus"], exact_measurement_indices=[])
    for key in ("lambdaN", "r_norm", "wls_objective", "vm_est_pu", "residual_covariance_diag"):
        np.testing.assert_array_equal(default[key], explicit_empty[key])


def test_default_state_tolerance_does_not_relax_exact_constraint_accuracy(exact_problem):
    case, mean, sigma, exact = exact_problem
    observed = mean + np.random.default_rng(19).normal(size=122) * sigma
    observed[2] += 10 * sigma[2]
    result = lagrangian_m_singlephase_details(observed, case, 0, case["bus"],
        measurement_sigma=sigma, exact_measurement_indices=exact)
    assert result["success"]
    assert np.max(np.abs(result["constraint_residual"])) <= 1e-9
    limited = lagrangian_m_singlephase_details(observed, case, 0, case["bus"],
        measurement_sigma=sigma, exact_measurement_indices=exact, max_it=0)
    assert not limited["success"]


@pytest.fixture(scope="module", params=["dangling_line_terminal", "bus_split", "merge"])
def topology_provider_case(request, tmp_path_factory):
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator
    from psse_env.providers.matpower import MatpowerDeploymentProviders
    generator = Round0ScenarioGenerator(seed=31, topology_effects=(request.param,),
        require_branch_dominant_topology=request.param != "merge",
        derived_case_dir=tmp_path_factory.mktemp(f"topology_{request.param}"))
    scenario = generator.build({"topology": 1})[0]
    state = {"state_id": "test:s0", "state_hash": "initial", "status": "active",
             "case": scenario["case"], "measurements": scenario["measurements"],
             "metadata": deepcopy(scenario["metadata"])}
    providers = MatpowerDeploymentProviders(chi2_alpha=0.01,
        derived_case_dir=tmp_path_factory.mktemp(f"candidate_{request.param}"))
    return scenario, state, providers


def _candidate(providers, state, truth):
    result = providers.correct_topology(state, {"tool": "correct_topology", "arguments": {
        "cb_name": truth["cb_name"], "status": int(truth["true_cb_closed"]), "state_id": state["state_id"]}})
    assert "modification" in result, result
    change = result["modification"]
    return {**deepcopy(state), "state_id": "test:s1", "state_hash": "candidate", "status": "candidate",
            "case": change["case"], "measurements": change.get("measurements", state["measurements"]),
            "metadata": {**deepcopy(state["metadata"]), **change["metadata_updates"]}}, result


def test_generated_topology_provider_consumes_covariance_before_and_after_candidate(topology_provider_case):
    scenario, state, providers = topology_provider_case
    original = deepcopy(state)
    metadata = state["metadata"]
    assert metadata["sigma_z"][16] == pytest.approx(np.sqrt(2) * 0.01)
    assert metadata["structural_zero_indices"] == [20, 34]
    source = providers._solve(state)["payload"]
    assert source["success"], source
    assert source["measurement_sigma"] == metadata["sigma_z"]
    assert source["exact_measurement_indices"] == [20, 34]
    assert source["dof"] == 95
    candidate, result = _candidate(providers, state, scenario["true_topology_errors"][0])
    noise = candidate["metadata"]["operator_noise"]
    assert len(candidate["metadata"]["sigma_z"]) == len(candidate["measurements"])
    assert candidate["metadata"]["sigma_z"] == noise["measurement_sigma"]
    assert candidate["metadata"]["structural_zero_indices"] == noise["structural_zero_indices"]
    solved = providers._solve(candidate)["payload"]
    assert solved["success"], solved
    assert solved["measurement_sigma"] == noise["measurement_sigma"]
    assert solved["exact_measurement_indices"] == noise["structural_zero_indices"]
    assert len(solved["r"]) == len(candidate["measurements"])
    if result["breaker_effect"] == "dangling_line_terminal":
        assert noise == metadata["operator_noise"]
    else:
        assert len(noise["measurement_sigma"]) != len(metadata["sigma_z"])
        assert candidate["metadata"]["substation_telemetry"] == state["metadata"]["substation_telemetry"]
    assert state == original


def test_aggregate_meter_error_survives_when_source_identity_is_unchanged(topology_provider_case):
    scenario, source, providers = topology_provider_case
    if scenario["true_topology_errors"][0]["physical_effect"] != "bus_split":
        pytest.skip("the split fixture exercises aggregation identity")
    state = deepcopy(source)
    state["measurements"][16] += 0.25
    before = deepcopy(state)
    candidate, _ = _candidate(providers, state, scenario["true_topology_errors"][0])
    physical_id = state["metadata"]["operator_noise"]["measurement_ids"][16]
    row = candidate["metadata"]["operator_noise"]["measurement_ids"].index(physical_id)
    projected = operator_vector_for_layout(candidate["metadata"]["substation_telemetry"], candidate["metadata"]["operator_layout"])
    assert candidate["measurements"][row] - projected[row] == pytest.approx(0.25, abs=1e-10)
    assert state == before


def test_aggregate_meter_error_is_not_distributed_across_split_sources(meter_setup, tmp_path):
    from psse_env.providers.matpower import MatpowerDeploymentProviders
    from Transmission.ieee14_full_substation import status_labels
    reference, model, layout, mean = meter_setup
    telemetry = add_telemetry_noise(mean, np.random.default_rng(31))
    noise = operator_noise_for_layout(telemetry, layout)
    measurements = operator_vector_for_layout(telemetry, layout).tolist()
    measurements[16] += 0.25
    state = {"case": "case14", "measurements": measurements, "metadata": {
        "substation_telemetry": telemetry, "reported_breaker_status": status_labels(model),
        "operator_voltage_meter_nodes": {key: value["meter_node"] for key, value in layout["sections"].items()},
        "operator_layout": layout, "operator_noise": noise, "sigma_z": noise["measurement_sigma"],
        "structural_zero_indices": noise["structural_zero_indices"]}}
    providers = MatpowerDeploymentProviders(derived_case_dir=tmp_path)
    # Bus 3 has separately metered load and generation on its two busbars.
    breaker = next(cb for cb in model.breakers if cb.name == "CB_3_L34_B2")
    result = providers.correct_topology(state, {"tool": "correct_topology", "arguments": {
        "cb_name": breaker.name, "status": int(not breaker.closed)}})
    assert "aggregate_meter_identity_ambiguous" in result["error_detail"]
    assert result["execution_status"] == "failure"
    assert "modification" not in result
