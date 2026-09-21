from __future__ import annotations

import json

import numpy as np
import pytest
from pypower.api import case14, ppoption, runpf

from Transmission.generate_measurements import compute_measurements_pu
from mcp_server.matpower_server import _load_python_case, _meas_correction_json
from tools.lagrangian_port import lagrangian_m_singlephase_details


@pytest.fixture
def physical_snapshot():
    solved, success = runpf(case14(), ppoption(VERBOSE=0, OUT_ALL=0))
    assert success
    clean = compute_measurements_pu(solved)
    sigma = np.r_[np.full(14, 0.001), np.full(108, 0.01)]
    sigma[[2, 16, 30]] *= 2  # Actual sensor covariance differs at bus 3.
    sigma[[20, 34]] = 0      # Equipment-free bus 7 injection equations are exact.
    observed = clean + np.random.default_rng(431).normal(0, sigma)
    observed[[20, 34]] = 0
    return clean, observed, sigma


def _correct(observed, sigma, group, **kwargs):
    return _meas_correction_json(
        "case14", observed.tolist(), suspect_group=group,
        R_variances_full=(sigma**2).tolist(), exact_measurement_indices=[20, 34],
        **kwargs,
    )


def test_real_noisy_meter_recovery_retains_exact_constraints_and_external_indices(physical_snapshot):
    clean, observed, sigma = physical_snapshot
    observed[2] += 10 * sigma[2]
    original = observed.copy()
    result = _correct(observed, sigma, [2], max_correction_iterations=3, error_tolerance=1e-8)
    assert result["success"]
    assert result["applied_any_correction"]
    assert 1 <= result["iterations_performed"] <= 3
    assert result["exact_measurement_indices"] == [20, 34]
    assert np.max(np.abs(result["constraint_residual"])) < 1e-8
    assert result["dof"] == 95
    assert len(result["r_norm"]) == len(result["resid_raw"]) == 122
    assert len(result["measurement_rows"]) == 120
    assert result["r_norm"][20] == result["r_norm"][34] == 0
    repaired = observed.copy()
    assert [item["index0"] for item in result["corrected_measurements"]] == [2]
    for item in result["corrected_measurements"]:
        repaired[item["index0"]] = item["corrected"]
        assert item["original"] == original[item["index0"]]
        assert item["estimated_error"] == pytest.approx(item["original"] - item["corrected"])
    assert abs(repaired[2] - clean[2]) / sigma[2] < 3
    np.testing.assert_array_equal(repaired[np.arange(122) != 2], original[np.arange(122) != 2])
    assert repaired[20] == repaired[34] == 0
    np.testing.assert_array_equal(observed, original)
    stochastic = np.array(result["measurement_rows"])
    raw = np.array(result["resid_raw"])[stochastic]
    assert result["wls_objective"] == pytest.approx(np.sum(raw**2 / sigma[stochastic]**2))
    json.dumps(result, allow_nan=False)


def test_group_update_uses_actual_supplied_variances_and_final_resolve(physical_snapshot):
    _clean, observed, sigma = physical_snapshot
    observed[2] += 10 * sigma[2]
    observed[16] -= 10 * sigma[16]
    ppc = _load_python_case("case14")
    details = lagrangian_m_singlephase_details(
        observed, ppc, 0, ppc["bus"], measurement_sigma=sigma,
        exact_measurement_indices=[20, 34], max_it=50, tol=1e-9,
    )
    group = np.array([2, 16])
    rows = np.array(details["measurement_rows"])
    local = np.array([int(np.flatnonzero(rows == index)[0]) for index in group])
    omega = details["residual_covariance"][np.ix_(local, local)]
    expected_error = sigma[group]**2 * np.linalg.solve(omega, details["raw_residual"][local])
    result = _correct(observed, sigma, [16, 2], max_correction_iterations=1, error_tolerance=0)
    assert result["success"]
    actual_error = np.array([item["estimated_error"] for item in result["corrected_measurements"]])
    np.testing.assert_allclose(actual_error, expected_error, rtol=1e-10, atol=1e-12)
    # Reusing the former default variances would be four times too small here.
    assert not np.allclose(actual_error, expected_error / 4)
    repaired = observed.copy()
    repaired[group] -= actual_error
    final = lagrangian_m_singlephase_details(
        repaired, ppc, 0, ppc["bus"], measurement_sigma=sigma,
        exact_measurement_indices=[20, 34], max_it=50, tol=1e-9,
    )
    assert result["wls_objective"] == pytest.approx(final["wls_objective"], rel=1e-10)
    assert result["iterations_performed"] == 1


def test_exact_constraints_cannot_be_meter_repair_targets(physical_snapshot):
    _clean, observed, sigma = physical_snapshot
    with pytest.raises(ValueError, match="must not include exact"):
        _correct(observed, sigma, [20])


def test_rank_deficient_target_group_is_rejected_without_pseudoinverse(physical_snapshot):
    _clean, observed, sigma = physical_snapshot
    observed[2] += 10 * sigma[2]
    result = _correct(observed, sigma, np.flatnonzero(sigma > 0).tolist())
    assert not result["success"]
    assert result["error"] == "rank_deficient_or_ill_conditioned_residual_covariance_group"
    assert not result["applied_any_correction"]
    assert result["corrected_measurements"] == []
    assert result["iterations_performed"] == 0


def test_undeclared_zero_variance_is_rejected(physical_snapshot):
    _clean, observed, sigma = physical_snapshot
    with pytest.raises(ValueError, match="Zero variances require exactly"):
        _meas_correction_json("case14", observed.tolist(), R_variances_full=(sigma**2).tolist())


def test_disabled_correction_still_solves_same_exact_covariance(physical_snapshot):
    _clean, observed, sigma = physical_snapshot
    result = _correct(observed, sigma, [2], enable_correction=False)
    assert result["success"]
    assert result["corrected_measurements"] == []
    assert not result["applied_any_correction"]
    assert result["iterations_performed"] == 0
    assert np.isfinite(result["wls_objective"])
    assert np.max(np.abs(result["constraint_residual"])) < 1e-8


def test_failed_final_solve_discards_the_proposal(monkeypatch, physical_snapshot):
    from tools import lagrangian_port

    _clean, observed, sigma = physical_snapshot
    observed[2] += 10 * sigma[2]
    real_solver = lagrangian_port.lagrangian_m_singlephase_details
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(kwargs)
        return real_solver(*args, **kwargs) if len(calls) == 1 else {"success": 0}

    monkeypatch.setattr(lagrangian_port, "lagrangian_m_singlephase_details", fail_second)
    result = _correct(observed, sigma, [2], max_correction_iterations=1)
    assert len(calls) == 2
    assert not result["success"]
    assert result["error"] == "constrained_wls_did_not_converge"
    assert result["corrected_measurements"] == []
    assert not result["applied_any_correction"]


def test_ordinary_positive_covariance_keeps_existing_solver_and_supplied_variance(monkeypatch):
    from tools import lagrangian_correct_port, lagrangian_port

    variance = np.r_[np.full(14, 4e-6), np.full(108, 9e-4)]
    received = []

    def legacy(**kwargs):
        received.append(kwargs["R_variances_full_in"].copy())
        return np.zeros(40), 1, np.zeros(122), np.eye(122), np.ones(122) * 0.001, {}

    def wrong_path(*args, **kwargs):
        raise AssertionError("The ordinary covariance path must retain its existing solver")

    monkeypatch.setattr(lagrangian_correct_port, "lagrangian_m_correct", legacy)
    monkeypatch.setattr(lagrangian_port, "lagrangian_m_singlephase_details", wrong_path)
    result = _meas_correction_json("case14", np.ones(122).tolist(), R_variances_full=variance.tolist())
    assert result["success"]
    np.testing.assert_array_equal(received[0], variance)
    assert result["wls_objective"] == pytest.approx(np.sum(1e-6 / variance))
