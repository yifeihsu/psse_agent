from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from pypower.api import case14

from three_phase_model.exporter import export_model, load_assumptions, validate_assumptions
from three_phase_model.measurements import extract_measurements, audit_full_circuit_kcl
from three_phase_model.runtime import compile_model
from three_phase_model.validation import validate_model, positive_sequence_terminal_admittance, reference_terminal_admittance
from three_phase_model.voltage_bases import IEEE14_NOMINAL_KV, IEEE14_VOLTAGE_BASE_PROFILE_ID
from Transmission.generate_measurements import compute_measurements_pu


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("ieee14_voltage_profiles")
    source = case14()
    before = copy.deepcopy(source)
    legacy = export_model(source, root / "legacy", case_id="case14")
    physical = export_model(source, root / "physical", case_id="case14", voltage_profile=IEEE14_VOLTAGE_BASE_PROFILE_ID)
    for key in ("bus", "branch", "gen"):
        np.testing.assert_array_equal(source[key], before[key])
    return source, legacy, physical


def _engine(build):
    from pathlib import Path
    return compile_model(Path(build["output_dir"]) / "Master.dss")


def test_explicit_profile_preserves_per_unit_network_and_compiled_measurements(models):
    source, legacy, physical = models
    reference = physical["reference"]
    np.testing.assert_array_equal(reference["branch"][:, :13], source["branch"][:, :13])
    np.testing.assert_allclose(reference["bus"][:, [7, 8]], legacy["reference"]["bus"][:, [7, 8]], rtol=0, atol=0)
    assert reference["bus"][:, 9].tolist() == [IEEE14_NOMINAL_KV[i] for i in range(1, 15)]
    assert physical["assumptions"]["base_kv_ll"] == 69
    assert physical["assumptions"]["voltage_profile"] == IEEE14_VOLTAGE_BASE_PROFILE_ID
    assert physical["manifest"]["line_count"] == 16
    assert physical["manifest"]["transformer_count"] == 4
    dss = _engine(physical)
    result = validate_model(dss, reference, physical["registry"], physical["assumptions"])
    assert result["passed"], result["failed_checks"]
    actual = extract_measurements(dss, physical["registry"], physical["assumptions"])
    expected = compute_measurements_pu(reference)
    np.testing.assert_allclose(actual["measurement_vector"], expected, rtol=0, atol=1e-7)
    assert audit_full_circuit_kcl(dss, physical["assumptions"])["max_kcl_mismatch_pu"] < 1e-7
    legacy_actual = extract_measurements(_engine(legacy), legacy["registry"], legacy["assumptions"])
    np.testing.assert_allclose(actual["measurement_vector"], legacy_actual["measurement_vector"], rtol=0, atol=1e-7)


def test_load_generator_and_bus_bases_are_local_and_explicit(models):
    _, _, physical = models
    dss = _engine(physical)
    for row in physical["registry"]["buses"]:
        dss.Circuit.SetActiveBus(row["dss_bus"])
        assert row["kv_ll"] == IEEE14_NOMINAL_KV[row["external_bus"]]
        assert dss.Bus.kVBase() * math.sqrt(3) == pytest.approx(row["kv_ll"], abs=1e-10)
    for kind, interface in (("loads", dss.Loads), ("generators", dss.Generators)):
        for row in physical["registry"][kind]:
            interface.Name(row["element"].split(".", 1)[1])
            assert interface.kV() == pytest.approx(IEEE14_NOMINAL_KV[row["bus"]] / math.sqrt(3))
    dss.Capacitors.Name("bs_009")
    assert dss.Capacitors.kV() == pytest.approx(13.8)
    dss.Vsources.Name("source")
    assert dss.Vsources.BasekV() == pytest.approx(69)


def test_lines_use_own_impedance_base_and_cross_voltage_line_becomes_transformer(models):
    source, _, physical = models
    dss = _engine(physical)
    for asset in physical["registry"]["branches"]:
        row = source["branch"][asset["branch_row0"]]
        kv_from, kv_to = IEEE14_NOMINAL_KV[asset["from_bus"]], IEEE14_NOMINAL_KV[asset["to_bus"]]
        assert asset["from_zbase_ohm"] == pytest.approx(kv_from ** 2 / 100)
        assert asset["to_zbase_ohm"] == pytest.approx(kv_to ** 2 / 100)
        if asset["dss_element"].startswith("Line."):
            assert kv_from == kv_to
            dss.Lines.Name(asset["dss_element"].split(".", 1)[1])
            np.testing.assert_allclose(np.array(dss.Lines.RMatrix()).reshape(3, 3), np.eye(3) * row[2] * kv_from ** 2 / 100, rtol=1e-13, atol=1e-13)
        else:
            dss.Transformers.Name(asset["dss_element"].split(".", 1)[1])
            dss.Transformers.Wdg(1)
            assert dss.Transformers.kV() == pytest.approx(kv_from)
            assert dss.Transformers.Tap() == pytest.approx(row[8] or 1)
            dss.Transformers.Wdg(2)
            assert dss.Transformers.kV() == pytest.approx(kv_to)
            assert dss.Transformers.Tap() == 1
    cross = physical["registry"]["branches"][13]
    assert (cross["from_bus"], cross["to_bus"]) == (7, 8)
    assert cross["dss_element"] == "Transformer.br_0014"
    assert cross["tap"] == 1 and cross["source_tap"] == 0
    actual_y = positive_sequence_terminal_admittance(dss, cross["dss_element"], "b7", "b8",
        terminal_kv_ll=(13.8, 18), base_mva=100)
    np.testing.assert_allclose(actual_y, reference_terminal_admittance(source["branch"][13]), rtol=0, atol=1e-8)


def test_independent_primitive_audit_detects_wrong_transformer_winding_voltage(models):
    _, _, physical = models
    dss = _engine(physical)
    dss.Text.Command("Edit Transformer.br_0014 Wdg=2 kV=17")
    dss.Solution.Solve()
    result = validate_model(dss, physical["reference"], physical["registry"], physical["assumptions"])
    assert not result["passed"]
    assert "branch_positive_sequence_terminal_admittance" in result["failed_checks"]


def test_compiled_base_assignment_is_checked_independently(models):
    _, _, physical = models
    dss = _engine(physical)
    dss.Text.Command("SetkVBase Bus=b8 kVLL=13.8")
    result = validate_model(dss, physical["reference"], physical["registry"], physical["assumptions"])
    assert not result["passed"]
    assert "compiled_bus_voltage_bases" in result["failed_checks"]


def test_legacy_default_and_assumptions_roundtrip_remain_supported(models, tmp_path):
    _, legacy, physical = models
    assert legacy["assumptions"]["base_kv_ll"] == 1
    assert "bus_base_kv_ll" not in legacy["assumptions"]
    assert legacy["manifest"]["line_count"] == 17
    assert legacy["manifest"]["transformer_count"] == 3
    assert all(row["kv_ll"] == 1 for row in legacy["registry"]["buses"])
    validate_assumptions(physical["assumptions"])
    replay = export_model(case14(), tmp_path / "replay", assumptions=physical["assumptions"], case_id="case14")
    assert replay["assumptions"] == physical["assumptions"]
    assert validate_model(_engine(replay), replay["reference"], replay["registry"], replay["assumptions"])["passed"]


def test_explicit_per_bus_assumptions_cover_exactly_the_case(tmp_path):
    spec = load_assumptions()
    spec["bus_base_kv_ll"] = dict(IEEE14_NOMINAL_KV)
    good = export_model(case14(), tmp_path / "custom", assumptions=spec)
    assert good["assumptions"]["voltage_profile"] == "explicit_bus_voltage_bases_v1"
    spec["bus_base_kv_ll"].pop(14)
    with pytest.raises(ValueError, match="cover exactly"):
        export_model(case14(), tmp_path / "incomplete", assumptions=spec)
    assert not (tmp_path / "incomplete").exists()


def test_cross_voltage_transformer_charging_and_local_resistive_inductive_shunts(tmp_path):
    case = case14()
    case["branch"][7, 2] = 0.015  # A nonzero winding resistance on 69/13.8 kV.
    case["branch"][7, 4] = 0.025  # Endpoint charging must use each winding's kV.
    case["bus"][7, 4] = 0.2      # Resistive shunt at the 18-kV generator bus.
    case["bus"][13, 5] = -0.2    # Inductive shunt at 13.8 kV.
    built = export_model(case, tmp_path / "local_shunts", case_id="case14",
                         voltage_profile=IEEE14_VOLTAGE_BASE_PROFILE_ID)
    dss = _engine(built)
    result = validate_model(dss, built["reference"], built["registry"], built["assumptions"])
    assert result["passed"], result["checks"]
    measurements = extract_measurements(dss, built["registry"], built["assumptions"])
    np.testing.assert_allclose(measurements["measurement_vector"], compute_measurements_pu(built["reference"]),
                               rtol=0, atol=1e-7)
