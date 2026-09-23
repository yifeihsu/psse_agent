"""Applying an HIF operating point must keep generator reactive limits and voltage regulation.

Writing a generator's kW makes OpenDSS recompute maxkvar/minkvar from the
nominal power factor. Before the fix, the 1 kW synchronous condensers at buses
3, 6 and 8 were left with about +-1 kvar, so they could not regulate and bus 8
(condenser only, no load) injected zero reactive power in every HIF corpus.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("opendssdirect")

from three_phase_nlm.hif_operating_point import (  # noqa: E402
    apply_hif_operating_point,
    canonicalize_ieee14_operating_point,
    capture_operating_point_baseline,
)
from three_phase_nlm.hif_parameter_estimator import (  # noqa: E402
    _compile_base_model,
    _resolve_model_dir,
    _simulate_base,
    _solve_or_raise,
)

# Maxkvar / Minkvar declared in IEEE_14_OpenDSS/IEEE14Gen.DSS (MATPOWER case14 Qmax/Qmin).
MODEL_LIMITS_KVAR = {"b2": (50000.0, -40000.0), "b3": (40000.0, 0.0), "b6": (24000.0, -6000.0), "b8": (24000.0, -6000.0)}
QINJ_BUS8 = 28 + 7
DIVERSE_POINT = {
    "load_scale": 0.93,
    "generator_dispatch_kw": {"b2": 39744.8, "b3": 1.0, "b6": 1.0, "b8": 1.0},
    "voltage_setpoints_pu": {"b2": 1.039, "b3": 1.0155, "b6": 1.06, "b8": 1.084},
    "source_voltage_pu": 1.0534,
}


def _live_limits() -> dict[str, tuple[float, float]]:
    import opendssdirect as dss

    limits = {}
    for name in MODEL_LIMITS_KVAR:
        dss.Circuit.SetActiveElement(f"Generator.{name}")
        limits[name] = (float(dss.Properties.Value("maxkvar")), float(dss.Properties.Value("minkvar")))
    return limits


def _pv_voltage_errors(op_point) -> dict[str, float]:
    import opendssdirect as dss

    setpoints = canonicalize_ieee14_operating_point(op_point)["voltage_setpoints_pu"]
    errors = {}
    for name in MODEL_LIMITS_KVAR:
        dss.Circuit.SetActiveBus(name)
        errors[name] = float(np.mean(dss.Bus.puVmagAngle()[0::2][:3])) - setpoints[name]
    return errors


@pytest.fixture()
def model_dir():
    return _resolve_model_dir(None, "case14")


def test_dispatch_write_keeps_model_reactive_limits(model_dir):
    _compile_base_model(model_dir)
    baseline = capture_operating_point_baseline()
    assert {g["key"]: (g["maxkvar"], g["minkvar"]) for g in baseline["generators"]} == MODEL_LIMITS_KVAR
    applied = apply_hif_operating_point(baseline, DIVERSE_POINT)
    assert _live_limits() == MODEL_LIMITS_KVAR
    assert {k.lower(): (v["maxkvar"], v["minkvar"]) for k, v in applied["reactive_limits_kvar"].items()} == MODEL_LIMITS_KVAR


def test_legacy_baseline_without_limits_still_keeps_them(model_dir):
    _compile_base_model(model_dir)
    baseline = capture_operating_point_baseline()
    for generator in baseline["generators"]:
        generator.pop("maxkvar")
        generator.pop("minkvar")
    apply_hif_operating_point(baseline, DIVERSE_POINT)
    assert _live_limits() == MODEL_LIMITS_KVAR


@pytest.mark.parametrize("op_point", [None, DIVERSE_POINT])
def test_pv_buses_regulate_and_bus8_condenser_supplies_reactive_power(model_dir, op_point):
    _compile_base_model(model_dir)
    apply_hif_operating_point(capture_operating_point_baseline(), op_point)
    _solve_or_raise()
    errors = _pv_voltage_errors(op_point)
    assert max(abs(v) for v in errors.values()) < 1e-3, errors
    import opendssdirect as dss

    dss.Circuit.SetActiveElement("Generator.b8")
    q_kvar = -sum(dss.CktElement.Powers()[1::2][:3])
    assert 1000.0 < q_kvar <= MODEL_LIMITS_KVAR["b8"][0] + 1e-6


def test_hif_reference_solve_carries_bus8_reactive_injection(model_dir):
    z = _simulate_base(model_dir, op_point=DIVERSE_POINT, shunt_convention="ybus")["z"]
    assert z[QINJ_BUS8] > 0.05


def test_generator_at_its_limit_stops_at_the_model_limit(model_dir):
    """At high load with a high setpoint, bus 8 binds at 24 MVAr (not at the +-1.08 kvar reset value)."""
    import opendssdirect as dss

    heavy = {**DIVERSE_POINT, "load_scale": 1.25, "voltage_setpoints_pu": {"b2": 1.045, "b3": 1.01, "b6": 1.07, "b8": 1.10}}
    _compile_base_model(model_dir)
    apply_hif_operating_point(capture_operating_point_baseline(), heavy)
    _solve_or_raise()
    dss.Circuit.SetActiveElement("Generator.b8")
    q_kvar = -sum(dss.CktElement.Powers()[1::2][:3])
    assert abs(q_kvar - MODEL_LIMITS_KVAR["b8"][0]) < 10.0
    assert _pv_voltage_errors(heavy)["b8"] < -1e-3  # setpoint unattainable, so voltage sits below it


def test_injected_hif_candidate_keeps_regulation(model_dir):
    """The candidate simulator (corpus scans and the estimator's forward model) regulates too."""
    from three_phase_nlm.hif_parameter_estimator import simulate_hif_candidate

    result = simulate_hif_candidate(candidate_branch_row0=4, alpha=0.5, phase="A", r_hif_pu=4.0,
                                    op_point=DIVERSE_POINT, shunt_convention="ybus")
    assert result["z"][QINJ_BUS8] > 0.05
    assert _live_limits() == MODEL_LIMITS_KVAR
