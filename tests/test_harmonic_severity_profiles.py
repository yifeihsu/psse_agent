from __future__ import annotations

import numpy as np
import pytest
from pypower.api import case14, ppoption, runopf

from Harmonics.ieee14_verification import (
    BASE_MVA, BRANCH, BUS, branch_terminal_currents_both, build_ybus,
    build_ybus_h, fundamental_bus_voltages,
)
from Transmission.generate_hse_traces import (
    HARMONIC_THD_PROFILES, build_trace, draw_harmonic_target_thd,
)
from Transmission.generate_measurements import compute_measurements_pu


@pytest.fixture(scope="module")
def opf_case():
    case = runopf(case14(), ppoption(VERBOSE=0, OUT_ALL=0))
    assert case["success"]
    return case


def test_fundamental_voltage_channels_do_not_change_with_harmonic_severity(opf_case):
    before_bus, before_branch = opf_case["bus"].copy(), opf_case["branch"].copy()
    traces = [build_trace(9, severity, 1928, bus=opf_case["bus"], branch=opf_case["branch"],
                          voltage_measurement="fundamental") for severity in (0.0, 0.01, 0.05, 0.2)]
    for trace in traces:
        np.testing.assert_array_equal(trace["z_scada_true"][:14], traces[0]["z_scada_true"][:14])
        # Same seed means exactly the same voltage noise is applied as well.
        np.testing.assert_array_equal(trace["z_scada_meas"][:14], traces[0]["z_scada_meas"][:14])
        assert trace["measurement_semantics"]["voltage_measurement"] == "fundamental"
    np.testing.assert_array_equal(opf_case["bus"], before_bus)
    np.testing.assert_array_equal(opf_case["branch"], before_branch)


@pytest.mark.parametrize("target", [0.01, 0.025, 0.05, 0.10, 0.20])
def test_true_rms_channels_obey_orthogonal_harmonic_formula(target, opf_case):
    trace = build_trace(9, target, 92, bus=opf_case["bus"], branch=opf_case["branch"])
    fundamental = np.abs(fundamental_bus_voltages(opf_case["bus"]))
    harmonic_squared = np.zeros(14)
    for rows in trace["harmonic_phasors"].values():
        phasors = np.array([complex(*row["V_complex_true"]) for row in rows])
        harmonic_squared += np.abs(phasors) ** 2
    expected = np.sqrt(fundamental ** 2 + harmonic_squared)
    np.testing.assert_allclose(trace["z_scada_true"][:14], expected, rtol=0, atol=5e-16)
    np.testing.assert_allclose(expected / fundamental,
                               np.sqrt(1 + np.array(trace["physical_severity"]["voltage_thd_by_bus"]) ** 2),
                               rtol=0, atol=5e-16)
    assert trace["actual_thd"] == pytest.approx(target, abs=1e-14)
    assert expected[8] / fundamental[8] - 1 == pytest.approx(np.sqrt(1 + target ** 2) - 1, abs=5e-16)
    assert trace["physical_severity"]["maximum_voltage_thd"] >= target - 1e-14
    assert trace["measurement_semantics"]["voltage_measurement"] == "true_rms"


@pytest.mark.parametrize("mode", ["true_rms", "fundamental"])
def test_zero_thd_uses_fundamental_in_all_injection_and_branch_power_channels(mode, opf_case):
    trace = build_trace(9, 0, 78, bus=opf_case["bus"], branch=opf_case["branch"], voltage_measurement=mode)
    expected = compute_measurements_pu(opf_case)
    np.testing.assert_allclose(trace["z_scada_true"], expected, rtol=0, atol=1e-11)
    assert np.max(np.abs(expected[42:])) > 0.1
    assert trace["physical_severity"]["source_count"] == 0
    assert trace["physical_severity"]["maximum_voltage_thd"] == 0


def test_voltage_mode_does_not_silently_change_legacy_power_meter_convention():
    rms = build_trace(9, 0.15, 94)
    fundamental = build_trace(9, 0.15, 94, voltage_measurement="fundamental")
    np.testing.assert_array_equal(rms["z_scada_true"][14:], fundamental["z_scada_true"][14:])
    assert "all-pass quadrature" in rms["measurement_semantics"]["reactive_power"]
    voltage_by_order = {1: fundamental_bus_voltages(BUS)}
    for order, rows in rms["harmonic_phasors"].items():
        voltage_by_order[int(order)] = np.array([complex(*row["V_complex_true"]) for row in rows])
    bus_p, bus_q = np.zeros(14), np.zeros(14)
    branch_pf, branch_qf, branch_pt, branch_qt = [np.zeros(20) for _ in range(4)]
    for order, voltage in voltage_by_order.items():
        ybus = build_ybus(BUS, BRANCH, BASE_MVA) if order == 1 else build_ybus_h(BUS, BRANCH, order, BASE_MVA)
        current = ybus @ voltage
        shift = (1 - 1j * order) / (1 + 1j * order)
        bus_p += (voltage * np.conj(current)).real
        bus_q += (shift * voltage * np.conj(current)).real
        from_current, to_current = branch_terminal_currents_both(voltage, BRANCH, order)
        from_voltage = voltage[BRANCH[:, 0].astype(int) - 1]
        to_voltage = voltage[BRANCH[:, 1].astype(int) - 1]
        branch_pf += (from_voltage * np.conj(from_current)).real
        branch_qf += (shift * from_voltage * np.conj(from_current)).real
        branch_pt += (to_voltage * np.conj(to_current)).real
        branch_qt += (shift * to_voltage * np.conj(to_current)).real
    # The harmonic solve uses a 1e-9 diagonal numerical regularizer; direct
    # physical YV differs from its imposed injection by that tiny term.
    np.testing.assert_allclose(rms["z_scada_true"][14:],
        np.r_[bus_p, bus_q, branch_pf, branch_qf, branch_pt, branch_qt], rtol=0, atol=1e-10)


def test_supplied_sigmas_are_used_in_draws_and_preserve_complex_component_contract():
    voltage_noise, power_noise = [], []
    for seed in range(40):
        trace = build_trace(9, 0.03, seed, sigma_vm=0.0025, sigma_pq=0.004)
        assert trace["sigma_z"] == [0.0025] * 14 + [0.004] * 108
        standardized = (np.array(trace["z_scada_meas"]) - trace["z_scada_true"]) / trace["sigma_z"]
        voltage_noise.extend(standardized[:14])
        power_noise.extend(standardized[14:])
        for rows in trace["harmonic_phasors"].values():
            for row in rows:
                assert row["sigma"] == pytest.approx(1e-4 / np.sqrt(2))
                assert row["sigma_complex_rms"] == 1e-4
    assert abs(np.mean(voltage_noise)) < 0.1
    assert 0.9 < np.std(voltage_noise) < 1.1
    assert abs(np.mean(power_noise)) < 0.05
    assert 0.95 < np.std(power_noise) < 1.05
    baseline = build_trace(9, 0.03, 77)
    scaled = build_trace(9, 0.03, 77, sigma_vm=0.0025, sigma_pq=0.004)
    np.testing.assert_allclose(
        (np.array(baseline["z_scada_meas"]) - baseline["z_scada_true"]) / baseline["sigma_z"],
        (np.array(scaled["z_scada_meas"]) - scaled["z_scada_true"]) / scaled["sigma_z"], atol=1e-12,
    )
    assert baseline["harmonic_phasors"] == scaled["harmonic_phasors"]


@pytest.mark.parametrize("profile", ["sensitivity", "stress"])
def test_severity_profile_draws_reach_declared_source_thd(profile):
    rng = np.random.default_rng(73)
    lower, upper = HARMONIC_THD_PROFILES[profile]
    for seed in range(4):
        target = draw_harmonic_target_thd(rng, profile)
        assert lower <= target <= upper
        trace = build_trace(9, target, seed)
        assert trace["actual_thd"] == pytest.approx(target, abs=1e-14)
        assert lower <= trace["physical_severity"]["source_voltage_thd"] <= upper
        assert trace["physical_severity"]["harmonic_orders"] == [5, 7, 11, 13, 17, 19]
        assert trace["physical_severity"]["source_count"] == 1


@pytest.mark.parametrize("kwargs", [
    {"sigma_vm": 0}, {"sigma_pq": -1}, {"sigma_vm": float("nan")},
    {"voltage_measurement": "ambiguous"}, {"harmonic_orders": [5, 5]},
    {"harmonic_orders": [1, 5]},
])
def test_invalid_meter_or_harmonic_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_trace(9, 0.1, 12, **kwargs)


def test_positive_target_requires_an_effective_source_and_profile_is_explicit():
    with pytest.raises(ValueError, match="no harmonic response"):
        build_trace(1, 0.1, 12)
    with pytest.raises(ValueError, match="unknown harmonic"):
        draw_harmonic_target_thd(np.random.default_rng(3), "unspecified")
