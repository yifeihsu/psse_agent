"""Actual multivoltage OpenDSS HIF circuits, local bases, and paired controls."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from psse_env.systems import resolve_system
from three_phase_model.diagnostics import capture_nominal_model, screen_measurements
from three_phase_model.disturbances import (
    audit_disturbed_circuit, eligible_hif_branch_rows, inject_midspan_hif,
    restore_midspan_hif, set_hif_enabled,
)
from three_phase_model.exporter import export_model, load_assumptions
from three_phase_model.measurements import extract_measurements
from three_phase_model.runtime import compile_model
from three_phase_model.voltage_bases import IEEE14_VOLTAGE_BASE_PROFILE_ID, hif_resistance_class


@pytest.fixture(scope="module", params=("normalized_diagonal", "coupled_sensitivity"))
def physical_model(request, tmp_path_factory):
    return export_model(resolve_system("case14").load_case(),
        tmp_path_factory.mktemp("physical_hif") / request.param, case_id="case14",
        assumptions=load_assumptions(request.param), voltage_profile=IEEE14_VOLTAGE_BASE_PROFILE_ID)


def _context(model):
    return compile_model(Path(model["output_dir"]) / "Master.dss")


def _telemetry(dss, model, receipt=None):
    return extract_measurements(dss, model["registry"], model["assumptions"],
        branch_overrides=None if receipt is None else receipt["branch_overrides"])


@pytest.mark.parametrize("branch_row0,kv", [(0, 69.0), (10, 13.8)])
@pytest.mark.parametrize("resistance_ohm", [50.0, 100.0, 500.0, 1000.0, 5000.0])
def test_physical_resistance_uses_faulted_line_voltage_and_engine_ohms_law(
    physical_model, branch_row0, kv, resistance_ohm,
):
    model = physical_model
    dss = _context(model)
    receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"],
        branch_row0=branch_row0, phase=2, alpha=.37, resistance_ohm=resistance_ohm)
    assert receipt["resistance_input_unit"] == "ohm"
    assert not receipt["resistance_defaulted"]
    assert receipt["resistance_ohm"] == resistance_ohm
    assert receipt["local_base_kv_ll"] == kv
    assert receipt["zbase_ohm"] == pytest.approx(kv**2 / 100)
    assert receipt["resistance_pu"] == pytest.approx(resistance_ohm / (kv**2 / 100))
    assert receipt["local_voltage_base_ln_v"] == pytest.approx(kv * 1000 / math.sqrt(3))
    # Physical-ohm class is voltage-agnostic: identical at 69 and 13.8 kV for the same ohms.
    assert receipt["resistance_class"] == hif_resistance_class(resistance_ohm)
    assert receipt["resistance_class"] == {50.0: "moderately_resistive", 100.0: "moderately_high_resistance",
        500.0: "weak_hif", 1000.0: "extreme_weak_hif", 5000.0: "near_open_circuit"}[resistance_ohm]
    audit = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert audit["passed"], audit
    voltage, current = complex(*audit["fault_voltage_v"]), complex(*audit["fault_current_a"])
    assert current == pytest.approx(voltage / resistance_ohm, abs=1e-9)
    assert audit["fault_real_power_w"] == pytest.approx(abs(voltage)**2 / resistance_ohm, abs=1e-6)
    assert abs(voltage) > .5 * receipt["local_voltage_base_ln_v"]
    bases = audit["node_current_bases_a"]
    for name, local_kv in (("b1", 69), ("b6", 13.8), ("b8", 18), (receipt["hidden_bus"], kv)):
        assert bases[name] == pytest.approx(100_000 / (math.sqrt(3) * local_kv))


def test_all_sixteen_physical_lines_preserve_no_fault_split_and_restoration(physical_model):
    model = physical_model
    eligible = eligible_hif_branch_rows(model["registry"])
    assert len(eligible) == 16
    assert set(eligible) == set(range(20)) - {7, 8, 9, 13}
    assert model["registry"]["branches"][13]["dss_element"].lower().startswith("transformer.")
    dss = _context(model)
    before = _telemetry(dss, model)["measurement_vector"]
    for row in eligible:
        receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"],
            branch_row0=row, alpha=.37, resistance_ohm=100, enabled=False)
        init = receipt["numerical_initialization"]
        assert init["no_fault_external_voltage_max_deviation_pu"] < 1e-8
        assert init["no_fault_hidden_voltage_max_deviation_pu"] < 1e-8
        np.testing.assert_allclose(_telemetry(dss, model, receipt)["measurement_vector"], before, atol=2e-8, rtol=0)
        assert audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])["passed"]
        set_hif_enabled(dss, receipt, True)
        assert audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])["passed"]
        set_hif_enabled(dss, receipt, False)
        np.testing.assert_allclose(_telemetry(dss, model, receipt)["measurement_vector"], before, atol=2e-8, rtol=0)
        restore_midspan_hif(dss, receipt)
        np.testing.assert_allclose(_telemetry(dss, model)["measurement_vector"], before, atol=2e-8, rtol=0)
        assert audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])["passed"]


def test_cross_voltage_asset_is_ineligible_even_if_mislabeled_line(physical_model):
    registry = copy.deepcopy(physical_model["registry"])
    registry["branches"][13]["dss_element"] = "Line.mislabeled_cross_voltage"
    assert 13 not in eligible_hif_branch_rows(registry)
    dss = _context(physical_model)
    before = dss.Circuit.AllElementNames()
    with pytest.raises(ValueError, match="same-voltage"):
        inject_midspan_hif(dss, registry, physical_model["assumptions"], branch_row0=13, resistance_ohm=100)
    assert dss.Circuit.AllElementNames() == before


@pytest.mark.parametrize("kwargs", [
    {"resistance_ohm": 100, "resistance_pu": 100 / 47.61},
    {"resistance_ohm": 0}, {"resistance_ohm": -100}, {"resistance_ohm": float("nan")},
    {"resistance_ohm": float("inf")}, {"resistance_ohm": True}, {"resistance_pu": np.bool_(True)},
])
def test_ambiguous_or_invalid_resistance_rejected_before_mutation(physical_model, kwargs):
    dss = _context(physical_model)
    before = dss.Circuit.AllElementNames()
    with pytest.raises(ValueError):
        inject_midspan_hif(dss, physical_model["registry"], physical_model["assumptions"], branch_row0=0, **kwargs)
    assert dss.Circuit.AllElementNames() == before


@pytest.mark.parametrize("branch_row0,kv", [(0, 69.0), (10, 13.8)])
def test_pu_and_ohm_are_same_physical_circuit_and_omitted_resistance_is_rejected(physical_model, branch_row0, kv):
    outputs = []
    for kwargs in ({"resistance_ohm": 100}, {"resistance_pu": 100 / (kv**2 / 100)}):
        dss = _context(physical_model)
        receipt = inject_midspan_hif(dss, physical_model["registry"], physical_model["assumptions"],
            branch_row0=branch_row0, **kwargs)
        assert not receipt["resistance_defaulted"]
        assert receipt["resistance_class"] == "moderately_high_resistance"
        outputs.append(_telemetry(dss, physical_model, receipt)["measurement_vector"])
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-10, rtol=0)
    # 2026-09-19: the silent 10 pu default would be 476 ohm at 69 kV but 19.04 ohm at
    # 13.8 kV, so a multi-voltage registry refuses an omitted resistance before mutation.
    registry_kv = {float(bus["kv_ll"]) for bus in physical_model["registry"]["buses"]}
    assert registry_kv == {69.0, 13.8, 18.0}
    dss = _context(physical_model)
    before = dss.Circuit.AllElementNames()
    with pytest.raises(ValueError, match="resistance_ohm or resistance_pu is required for a multi-voltage registry"):
        inject_midspan_hif(dss, physical_model["registry"], physical_model["assumptions"], branch_row0=branch_row0)
    assert dss.Circuit.AllElementNames() == before
    np.testing.assert_allclose(_telemetry(dss, physical_model)["measurement_vector"],
        _telemetry(_context(physical_model), physical_model)["measurement_vector"], atol=1e-12, rtol=0)


def test_uniform_normalized_ieee14_registry_keeps_legacy_ten_pu_default(tmp_path):
    # The legacy exporter default (no voltage_profile) is a uniform 1 kV base: the
    # historical 10 pu default (0.1 model-ohm) is retained there for backward compatibility.
    legacy = export_model(resolve_system("case14").load_case(), tmp_path / "legacy_uniform",
        case_id="case14", assumptions=load_assumptions("normalized_diagonal"))
    assert {float(bus["kv_ll"]) for bus in legacy["registry"]["buses"]} == {1.0}
    dss = _context(legacy)
    receipt = inject_midspan_hif(dss, legacy["registry"], legacy["assumptions"], branch_row0=0)
    assert receipt["resistance_defaulted"] and receipt["resistance_input_unit"] == "pu"
    assert receipt["resistance_pu"] == pytest.approx(10)
    assert receipt["resistance_ohm"] == pytest.approx(0.1)
    assert receipt["local_base_kv_ll"] == 1.0 and receipt["zbase_ohm"] == pytest.approx(0.01)
    # Model ohms on a normalized base are not physical ohms; the receipt states the scope.
    assert receipt["resistance_class"] == hif_resistance_class(0.1) == "low_resistance_fault"
    assert "declared local base" in receipt["resistance_class_scope"]
    assert audit_disturbed_circuit(dss, receipt, legacy["registry"], legacy["assumptions"])["passed"]


def test_pristine_multivoltage_nominal_screen_is_quiet_including_transformers(physical_model):
    dss = _context(physical_model)
    nominal = capture_nominal_model(dss, physical_model["registry"], physical_model["assumptions"])
    telemetry = _telemetry(dss, physical_model)
    result = screen_measurements(telemetry, nominal)
    assert result["classification"] == "no_detectable_anomaly", result
    assert result["max_branch_normalized_residual"] < 1e-5
    assert telemetry["max_kcl_mismatch_pu"] < 1e-7
    assert nominal["branches"][13]["from_base_kv_ll"] == 13.8
    assert nominal["branches"][13]["to_base_kv_ll"] == 18


@pytest.mark.parametrize("branch_row0,resistance_ohm", [(0, 50), (0, 100), (10, 100), (10, 500)])
def test_analytic_screen_recovers_physical_resistance_without_legacy_pu_bounds(physical_model, branch_row0, resistance_ohm):
    dss = _context(physical_model)
    nominal = capture_nominal_model(dss, physical_model["registry"], physical_model["assumptions"])
    receipt = inject_midspan_hif(dss, physical_model["registry"], physical_model["assumptions"],
        branch_row0=branch_row0, alpha=.37, phase=3, resistance_ohm=resistance_ohm)
    result = screen_measurements(_telemetry(dss, physical_model, receipt), nominal)
    assert result["classification"] == "hif_like_branch_mismatch", result
    fit = result["hif_candidate"]
    assert (fit["branch_row0"], fit["phase"]) == (branch_row0, 3)
    assert fit["alpha_estimate"] == pytest.approx(.37, abs=1e-7)
    assert fit["resistance_ohm_estimate"] == pytest.approx(resistance_ohm, rel=1e-7)
    assert fit["resistance_pu_estimate"] == pytest.approx(receipt["resistance_pu"], rel=1e-7)
    assert fit["resistance_sigma_linearized_ohm"] == pytest.approx(
        fit["resistance_sigma_linearized_pu"] * receipt["zbase_ohm"])
