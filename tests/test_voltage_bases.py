"""Local physical-unit conversion for the explicitly selected IEEE14 variant."""
from copy import deepcopy
import json
import math

import numpy as np
from pypower.case14 import case14
from pypower.idx_brch import BR_R, BR_STATUS, F_BUS, T_BUS, TAP
from pypower.idx_bus import BASE_KV, BUS_I
import pytest

from three_phase_model.voltage_bases import (
    IEEE14_NOMINAL_KV, IEEE14_VOLTAGE_BASE_PROFILE_ID,
    apply_ieee14_voltage_bases, eligible_ieee14_hif_branch_rows,
    hif_resistance_spec, ieee14_hif_branch_eligibility,
    ieee14_voltage_base_profile, impedance_base_ohm,
)


def test_explicit_profile_copies_case_and_maps_external_bus_ids_without_changing_pu_data():
    original = case14()
    original["bus"] = original["bus"][::-1].copy()
    before = deepcopy(original)
    assert np.count_nonzero(original["bus"][:, BASE_KV]) == 0
    configured = apply_ieee14_voltage_bases(original)
    assert configured is not original
    for source, row in zip(before["bus"], configured["bus"]):
        assert row[BASE_KV] == IEEE14_NOMINAL_KV[int(source[BUS_I])]
        np.testing.assert_array_equal(np.delete(row, BASE_KV), np.delete(source, BASE_KV))
    for key in ("branch", "gen", "gencost"):
        np.testing.assert_array_equal(configured[key], before[key])
        assert not np.shares_memory(configured[key], original[key])
    assert configured["baseMVA"] == before["baseMVA"] == 100
    np.testing.assert_array_equal(original["bus"], before["bus"])
    assert "voltage_base_profile" not in original
    receipt = configured["voltage_base_profile"]
    assert receipt["profile_id"] == IEEE14_VOLTAGE_BASE_PROFILE_ID
    assert not receipt["universal_ieee14_variant_claim"]
    assert not receipt["canonical_source_modified"]
    assert set(receipt["original_bus_base_kv_ll"].values()) == {0.0}
    json.dumps(receipt, allow_nan=False)
    receipt["bus_base_kv_ll"][1] = 1
    assert ieee14_voltage_base_profile()["bus_base_kv_ll"][1] == 69


@pytest.mark.parametrize("kv,expected", [(69, 47.61), (13.8, 1.9044), (18, 3.24)])
def test_impedance_bases_use_line_to_line_kv_and_three_phase_mva(kv, expected):
    assert impedance_base_ohm(kv, 100) == pytest.approx(expected)
    assert impedance_base_ohm(kv, 50) == pytest.approx(2 * expected)


@pytest.mark.parametrize("resistance", [50, 100, 200, 500, 1000, 2000, 5000])
def test_same_physical_resistance_remains_separate_between_voltage_strata(resistance):
    high = hif_resistance_spec(resistance, 69, 100)
    low = hif_resistance_spec(resistance, 13.8, 100)
    assert low["resistance_pu"] == pytest.approx(25 * high["resistance_pu"])
    assert low["nominal_fault_current_a"] == pytest.approx(high["nominal_fault_current_a"] / 5)
    assert low["nominal_fault_current_pu"] == pytest.approx(high["nominal_fault_current_pu"] / 25)
    for spec in (high, low):
        assert spec["resistance_ohm"] == resistance
        assert spec["resistance_pu"] * spec["impedance_base_ohm"] == pytest.approx(resistance)
        voltage_ln = spec["kv_ll"] * 1000 / math.sqrt(3)
        assert spec["nominal_fault_current_a"] == pytest.approx(voltage_ln / resistance)
        assert spec["nominal_fault_current_a"] / spec["current_base_a"] == pytest.approx(spec["nominal_fault_current_pu"])
        assert spec["nominal_single_phase_fault_power_mw"] / 100 == pytest.approx(1 / (3 * spec["resistance_pu"]))
        assert "nominal" in spec["approximation"] and "arcing" in spec["approximation"]
        json.dumps(spec, allow_nan=False)
    assert hif_resistance_spec(100, 69, 100)["resistance_pu"] == pytest.approx(2.1003990758)
    assert hif_resistance_spec(1000, 13.8, 100)["resistance_pu"] == pytest.approx(525.0997689561)


def test_eligibility_excludes_zero_tap_cross_voltage_branch_and_retains_same_voltage_zero_r():
    case = case14()
    before = deepcopy(case)
    result = ieee14_hif_branch_eligibility(case)
    assert result["voltage_profile"] == IEEE14_VOLTAGE_BASE_PROFILE_ID
    assert result["eligible_branch_rows0"] == [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 16, 17, 18, 19]
    assert eligible_ieee14_hif_branch_rows(case) == result["eligible_branch_rows0"]
    cross_voltage = result["branch_rows"][13]
    assert (cross_voltage["from_bus"], cross_voltage["to_bus"]) == (7, 8)
    assert cross_voltage["raw_tap"] == 0
    assert cross_voltage["exclusion_reasons"] == ["cross_voltage_branch"]
    zero_r_line = result["branch_rows"][14]
    assert (zero_r_line["from_bus"], zero_r_line["to_bus"]) == (7, 9)
    assert case["branch"][14, BR_R] == 0 and zero_r_line["eligible"]
    eligible = [row for row in result["branch_rows"] if row["eligible"]]
    assert sum(row["kv_ll"] == 69 for row in eligible) == 7
    assert sum(row["kv_ll"] == 13.8 for row in eligible) == 9
    for key in ("bus", "branch"):
        np.testing.assert_array_equal(case[key], before[key])
    # Bus row order and zero/irrelevant BASE_KV declarations cannot change selection.
    case["bus"] = case["bus"][::-1].copy()
    case["bus"][:, BASE_KV] = 1
    assert eligible_ieee14_hif_branch_rows(case) == result["eligible_branch_rows0"]


def test_inactive_and_explicit_transformer_rows_cannot_be_hif_lines():
    case = case14()
    case["branch"][0, BR_STATUS] = 0
    case["branch"][1, TAP] = 1
    result = ieee14_hif_branch_eligibility(case)
    assert result["branch_rows"][0]["exclusion_reasons"] == ["inactive_branch"]
    assert result["branch_rows"][1]["exclusion_reasons"] == ["transformer_tap"]
    assert len(result["eligible_branch_rows0"]) == 14


@pytest.mark.parametrize("bad", [0, -1, True, np.bool_(True), float("nan"), float("inf"), "69", [69]])
def test_invalid_physical_units_fail_closed(bad):
    with pytest.raises(ValueError):
        impedance_base_ohm(bad, 100)
    with pytest.raises(ValueError):
        impedance_base_ohm(69, bad)
    for args in ((bad, 69, 100), (100, bad, 100), (100, 69, bad)):
        with pytest.raises(ValueError):
            hif_resistance_spec(*args)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "fractional", "nonfinite", "short"])
def test_bus_profile_rejects_unknown_or_ambiguous_bus_identity(mutation):
    case = case14()
    if mutation == "missing":
        case["bus"][0, BUS_I] = 15
    elif mutation == "duplicate":
        case["bus"][0, BUS_I] = 2
    elif mutation == "fractional":
        case["bus"][0, BUS_I] = 1.5
    elif mutation == "nonfinite":
        case["bus"][0, BASE_KV] = np.nan
    else:
        case["bus"] = case["bus"][:, :BASE_KV]
    with pytest.raises(ValueError):
        apply_ieee14_voltage_bases(case)
    with pytest.raises(ValueError):
        ieee14_hif_branch_eligibility(case)


@pytest.mark.parametrize("bus", [0, 15, 1.5, float("nan")])
def test_eligibility_rejects_unknown_endpoint_identity(bus):
    case = case14()
    case["branch"][0, T_BUS] = bus
    with pytest.raises(ValueError):
        ieee14_hif_branch_eligibility(case)


def test_unrepresentable_extreme_base_and_resistance_fail_closed():
    for kv in (1e308, 1e-308):
        with pytest.raises(ValueError):
            impedance_base_ohm(kv, 100)
    with pytest.raises(ValueError):
        hif_resistance_spec(1e308, 1e-150, 100)
