"""The effect-targeted magnitude sampler (research.classifier_triage.effect_sampler, 2026-10-12)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from research.classifier_triage import effect_sampler as es


def test_the_ratio_is_one_at_the_alarm_threshold_on_both_networks():
    for dof in (95, 863):
        threshold = dof + es.threshold_excess(dof)
        assert es.effect_ratio(threshold, dof) == pytest.approx(1.0)
        assert es.effect_ratio(dof, dof) == pytest.approx(0.0)
    assert es.threshold_excess(863) > 2.5 * es.threshold_excess(95)  # grows like sqrt(dof)


def test_targets_are_log_uniform_in_the_band_and_bands_parse():
    rng = np.random.default_rng(0)
    draws = np.array([es.draw_target(rng, (1.5, 20.0)) for _ in range(4000)])
    assert draws.min() >= 1.5 and draws.max() <= 20.0
    assert abs(np.mean(np.log(draws)) - 0.5 * (math.log(1.5) + math.log(20.0))) < 0.05
    assert es.parse_band("1.5,20") == (1.5, 20.0) and es.parse_band((2, 4)) == (2.0, 4.0) and es.parse_band(None) is None
    with pytest.raises(ValueError):
        es.parse_band("20,1.5")


def _quadratic_solver(dof: int = 95, gain: float = 40.0):
    """A stand-in WLS whose excess is the squared norm of the scaled deviation plus the noise's."""
    def solve(z):
        energy = float(np.sum(z ** 2))
        return dof + gain * energy, dof
    return solve


def test_fit_scale_reaches_the_target_within_tolerance_and_reports_unreachable_bands():
    solve = _quadratic_solver()
    reference = np.zeros(10)
    deviation = np.zeros(10); deviation[3] = 1.0
    noise = np.zeros(10)
    for target in (1.5, 5.0, 20.0):
        fit = es.fit_scale(solve, reference, deviation, noise, target)
        assert fit["reachable"] and abs(fit["ratio"] - target) <= es.DEFAULT_TOLERANCE * target
        assert fit["iterations"] <= es.MAX_ITERATIONS + 2
    weak = es.fit_scale(solve, reference, deviation, noise, 20.0, scale_bounds=(0.01, 0.1))
    assert weak["reachable"] is False and weak["reason"] == "too_weak_within_bounds"
    strong = es.fit_scale(solve, reference, deviation, noise, 1.5, scale_bounds=(10.0, 50.0))
    assert strong["reachable"] is False and strong["reason"] == "too_strong_within_bounds"
    assert es.fit_scale(solve, reference, np.zeros(10), noise, 2.0)["reachable"] is False


def test_implied_magnitudes_follow_the_family_laws():
    k = 4.0
    meter = es.implied_magnitude("measurement", {"amplitude": 0.05}, k)
    assert meter["amplitude"] == pytest.approx(0.2)
    harmonic = es.implied_magnitude("harmonic", {"thd_target": 0.1}, k)
    assert harmonic["thd"] == pytest.approx(0.2)
    hif = es.implied_magnitude("hif", {"r_hif_ohm": 400.0, "nominal_fault_current_pu": 0.25}, k)
    assert hif["r_hif_ohm"] == pytest.approx(100.0) and hif["nominal_fault_current_pu"] == pytest.approx(1.0)
    unbalance = es.implied_magnitude("three_phase_unbalance", {"load_split": {"fractions": {"a": 0.4, "b": 0.3, "c": 0.3}}}, 0.5)
    assert unbalance["fractions"]["a"] == pytest.approx(1 / 3 + 0.5 * (0.4 - 1 / 3))
    parameter = es.implied_magnitude("parameter", {"r_factor": 0.25, "x_factor": 1.0}, 0.5)
    assert parameter["r_factor"] == pytest.approx(0.5) and parameter["x_factor"] == pytest.approx(1.0)


def test_scaled_phasor_rows_interpolate_between_clean_and_disturbed():
    clean = [{"vln_pu": [1.0, 1.0, 1.0], "ang_deg": [0.0, -120.0, 120.0]}]
    disturbed = [{"vln_pu": [0.9, 1.0, 1.0], "ang_deg": [0.0, -120.0, 120.0]}]
    half = es.scale_phasor_rows(disturbed, clean, 0.5, es.VOLTAGE_PHASOR_FIELDS)
    assert half[0]["vln_pu"][0] == pytest.approx(0.95) and half[0]["vln_pu"][1] == pytest.approx(1.0)
    double = es.scale_phasor_rows(disturbed, clean, 2.0, es.VOLTAGE_PHASOR_FIELDS)
    assert double[0]["vln_pu"][0] == pytest.approx(0.8)


def test_a_single_meter_on_ieee14_lands_in_the_band_with_the_real_wls():
    from psse_env.systems import resolve_system
    from Transmission.generate_measurements import compute_measurements_pu, solve_ac_opf

    solved = solve_ac_opf(resolve_system("case14").load_case())
    z_true = np.asarray(compute_measurements_pu(solved), dtype=float)
    sigma = np.where(np.arange(z_true.size) < 14, 1e-3, 1e-2)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(z_true.size) * sigma
    gross = np.zeros_like(z_true); gross[40] = 10.0 * sigma[40]  # a 10-sigma injection meter
    solve = es.wls_solver("case14", sigma)
    for target in (1.5, 8.0):
        fit = es.fit_scale(solve, z_true, gross, noise, target)
        assert fit["reachable"], fit
        assert abs(fit["ratio"] - target) <= es.DEFAULT_TOLERANCE * target
        objective, dof = solve(z_true + fit["scale"] * gross + noise)
        assert es.effect_ratio(objective, dof) == pytest.approx(fit["ratio"])
