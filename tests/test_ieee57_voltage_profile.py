"""Selected IEEE57 physical reconstruction; canonical pu data remains intact."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from psse_env.systems import resolve_system
from three_phase_model.exporter import export_model, load_assumptions
from three_phase_model.measurements import extract_measurements
from three_phase_model.runtime import compile_model
from three_phase_model.validation import validate_model
from three_phase_model.voltage_bases import (
    IEEE14_VOLTAGE_BASE_PROFILE_ID, IEEE57_VOLTAGE_BASE_PROFILE_ID,
    IEEE57_RECONSTRUCTION_KV, apply_ieee57_voltage_bases,
    eligible_ieee57_hif_branch_rows, get_voltage_base_profile,
    ieee57_hif_branch_eligibility, ieee57_voltage_base_profile,
)


@pytest.fixture(scope="module")
def source():
    return resolve_system("case57").load_case()


@pytest.fixture(scope="module")
def models(tmp_path_factory, source):
    root = tmp_path_factory.mktemp("ieee57_voltage_profile")
    return (
        export_model(source, root / "legacy", case_id="case57"),
        export_model(source, root / "reconstructed", case_id="case57",
                     voltage_profile=IEEE57_VOLTAGE_BASE_PROFILE_ID),
    )


def _engine(model):
    return compile_model(Path(model["output_dir"]) / "Master.dss")


def test_live_source_is_unspecified_and_reconstruction_is_explicit_detached_and_nonmutating(source):
    path = Path(__file__).resolve().parents[1] / "mcp_server" / "case57.m"
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before = deepcopy(source)
    assert len(source["bus"]) == 57 and len(source["branch"]) == 80 and source["baseMVA"] == 100
    np.testing.assert_array_equal(source["bus"][:, 9], np.zeros(57))
    shuffled = deepcopy(source)
    shuffled["bus"] = shuffled["bus"][::-1].copy()
    result = apply_ieee57_voltage_bases(shuffled)
    assert {int(row[0]): row[9] for row in result["bus"]} == IEEE57_RECONSTRUCTION_KV
    assert list(IEEE57_RECONSTRUCTION_KV.values()).count(138.) == 17
    assert list(IEEE57_RECONSTRUCTION_KV.values()).count(69.) == 40
    np.testing.assert_array_equal(np.delete(result["bus"], 9, axis=1), np.delete(shuffled["bus"], 9, axis=1))
    for field in ("branch", "gen", "gencost"):
        np.testing.assert_array_equal(result[field], source[field])
        assert not np.shares_memory(result[field], source[field])
    assert result["baseMVA"] == 100
    profile = result["voltage_base_profile"]
    assert profile["profile_id"] == IEEE57_VOLTAGE_BASE_PROFILE_ID
    assert profile["reference_base_mva"] == 100
    assert not profile["canonical_nominal_voltage_claim"] and not profile["canonical_source_modified"]
    assert set(profile["original_bus_base_kv_ll"].values()) == {0.}
    profile["bus_base_kv_ll"][1] = 1
    assert ieee57_voltage_base_profile()["bus_base_kv_ll"][1] == 138
    assert get_voltage_base_profile(IEEE57_VOLTAGE_BASE_PROFILE_ID) == ieee57_voltage_base_profile()
    for field in ("bus", "gen", "branch"):
        np.testing.assert_array_equal(source[field], before[field])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    json.dumps(profile, allow_nan=False)


def test_eligibility_derives_both_voltage_strata_and_preserves_parallel_row_ids(source):
    result = ieee57_hif_branch_eligibility(source)
    expected = [i for i, row in enumerate(source["branch"])
                if row[10] == 1 and row[8] == 0 and row[9] == 0
                and IEEE57_RECONSTRUCTION_KV[int(row[0])] == IEEE57_RECONSTRUCTION_KV[int(row[1])]]
    assert result["voltage_profile"] == IEEE57_VOLTAGE_BASE_PROFILE_ID
    assert result["eligible_branch_rows0"] == eligible_ieee57_hif_branch_rows(source) == expected
    assert len(expected) == 63
    eligible = [row for row in result["branch_rows"] if row["eligible"]]
    assert sum(row["kv_ll"] == 138 for row in eligible) == 26
    assert sum(row["kv_ll"] == 69 for row in eligible) == 37
    for first, second in ((18, 19), (34, 35)):
        a, b = result["branch_rows"][first], result["branch_rows"][second]
        assert (a["from_bus"], a["to_bus"]) == (b["from_bus"], b["to_bus"])
        assert a["branch_row0"] != b["branch_row0"]
        assert not a["eligible"] and not b["eligible"]


def test_cross_voltage_zero_tap_and_other_non_line_assets_are_excluded(source):
    changed = deepcopy(source)
    assert tuple(changed["branch"][18, :2]) == (4, 18)
    changed["branch"][18, 8] = 0
    changed["branch"][0, 10] = 0
    changed["branch"][1, 9] = 5
    result = ieee57_hif_branch_eligibility(changed)["branch_rows"]
    assert result[18]["exclusion_reasons"] == ["cross_voltage_branch"]
    assert result[0]["exclusion_reasons"] == ["inactive_branch"]
    assert result[1]["exclusion_reasons"] == ["phase_shifting_branch"]


def test_named_profile_preserves_pu_reference_and_independent_compiled_validation(source, models):
    from Transmission.generate_measurements import compute_measurements_pu

    legacy, physical = models
    for key in ("bus", "branch", "gen"):
        np.testing.assert_array_equal(source[key], resolve_system("case57").load_case()[key])
    np.testing.assert_array_equal(physical["reference"]["branch"][:, :13], source["branch"][:, :13])
    np.testing.assert_array_equal(physical["reference"]["bus"][:, [7, 8]], legacy["reference"]["bus"][:, [7, 8]])
    assert physical["assumptions"]["voltage_profile"] == IEEE57_VOLTAGE_BASE_PROFILE_ID
    assert physical["assumptions"]["base_kv_ll"] == 138
    expected_lines = len(eligible_ieee57_hif_branch_rows(source))
    assert physical["manifest"]["line_count"] == expected_lines
    assert physical["manifest"]["transformer_count"] == len(source["branch"]) - expected_lines
    dss = _engine(physical)
    validation = validate_model(dss, physical["reference"], physical["registry"], physical["assumptions"])
    assert validation["passed"], validation["failed_checks"]
    assert validation["checks"]["phase_node_kcl"]["node_count"] == 171
    measured = extract_measurements(dss, physical["registry"], physical["assumptions"])
    assert len(measured["measurement_vector"]) == 491
    assert len(measured["three_phase_voltages"]) == 57
    assert len(measured["three_phase_branch_currents"]) == 80
    np.testing.assert_allclose(measured["measurement_vector"], compute_measurements_pu(physical["reference"]), rtol=0, atol=1e-7)
    uniform = extract_measurements(_engine(legacy), legacy["registry"], legacy["assumptions"])
    np.testing.assert_allclose(measured["measurement_vector"], uniform["measurement_vector"], rtol=0, atol=1e-7)
    assert all(bus["kv_ll"] == 1 for bus in legacy["registry"]["buses"])
    assert "voltage_profile" not in legacy["assumptions"]


def test_actual_bus_load_generator_shunt_transformer_and_line_bases_are_local(source, models):
    _, physical = models
    dss, registry = _engine(physical), physical["registry"]
    for row in registry["buses"]:
        kv = IEEE57_RECONSTRUCTION_KV[row["external_bus"]]
        dss.Circuit.SetActiveBus(row["dss_bus"])
        assert dss.Bus.kVBase() * math.sqrt(3) == pytest.approx(kv, abs=1e-10)
        assert row["source_base_kv"] == 0
    for kind, interface in (("loads", dss.Loads), ("generators", dss.Generators)):
        for row in registry[kind]:
            interface.Name(row["element"].split(".", 1)[1])
            assert interface.kV() == pytest.approx(IEEE57_RECONSTRUCTION_KV[row["bus"]] / math.sqrt(3))
    for row in registry["shunts"]:
        dss.Capacitors.Name(row["element"].split(".", 1)[1])
        assert dss.Capacitors.kV() == pytest.approx(IEEE57_RECONSTRUCTION_KV[row["bus"]])
    for row in registry["branches"]:
        native = source["branch"][row["branch_row0"]]
        from_kv, to_kv = IEEE57_RECONSTRUCTION_KV[row["from_bus"]], IEEE57_RECONSTRUCTION_KV[row["to_bus"]]
        assert row["from_zbase_ohm"] == pytest.approx(from_kv**2 / 100)
        assert row["to_zbase_ohm"] == pytest.approx(to_kv**2 / 100)
        if row["dss_element"].startswith("Line."):
            assert from_kv == to_kv
            dss.Lines.Name(row["dss_element"].split(".", 1)[1])
            np.testing.assert_allclose(np.asarray(dss.Lines.RMatrix()).reshape(3, 3), np.eye(3) * native[2] * from_kv**2 / 100,
                                       rtol=1e-13, atol=1e-13)
        else:
            dss.Transformers.Name(row["dss_element"].split(".", 1)[1])
            for winding, kv, tap in ((1, from_kv, native[8] or 1), (2, to_kv, 1)):
                dss.Transformers.Wdg(winding)
                assert dss.Transformers.kV() == pytest.approx(kv)
                assert dss.Transformers.Tap() == pytest.approx(tap)
    dss.Vsources.Name("source")
    assert dss.Vsources.BasekV() == pytest.approx(138)


def test_balanced_wls_and_measurement_covariance_do_not_depend_on_physical_voltage_units(models):
    from research.gnn_screen.wls_features import build_wls_features, configured_case, default_measurement_sigma

    _, physical = models
    measured = extract_measurements(_engine(physical), physical["registry"], physical["assumptions"])
    sigma = default_measurement_sigma(57, 80)
    assert sigma.shape == (491,)
    z = np.asarray(measured["measurement_vector"]) + np.random.default_rng(57).normal(size=491) * sigma
    case = configured_case(physical["reference"])
    unspecified = deepcopy(case)
    unspecified["bus"][:, 9] = 0
    actual = build_wls_features(case, z, measurement_sigma=sigma)
    legacy = build_wls_features(unspecified, z, measurement_sigma=sigma)
    assert actual["n_states"] == 113 and actual["dof"] == 378
    np.testing.assert_array_equal(actual["variance"], sigma**2)
    for key in ("fitted", "raw_residual", "signed_normalized_residual", "theta_est_rad", "vm_est_pu"):
        np.testing.assert_array_equal(actual[key], legacy[key])
    assert actual["wls_objective"] == legacy["wls_objective"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "fractional"])
def test_apply_profile_rejects_missing_or_ambiguous_bus_identity(source, kind):
    changed = deepcopy(source)
    changed["bus"][0, 0] = {"missing": 58, "duplicate": 2, "fractional": 1.5}[kind]
    with pytest.raises(ValueError):
        apply_ieee57_voltage_bases(changed)
    with pytest.raises(ValueError):
        ieee57_hif_branch_eligibility(changed)


def test_named_profile_rejects_wrong_system_or_conflicting_assumptions_before_writing(source, tmp_path):
    with pytest.raises(ValueError, match="cover exactly"):
        export_model(source, tmp_path / "wrong_system", voltage_profile=IEEE14_VOLTAGE_BASE_PROFILE_ID)
    assert not (tmp_path / "wrong_system").exists()
    spec = load_assumptions()
    spec["bus_base_kv_ll"] = dict(IEEE57_RECONSTRUCTION_KV)
    spec["bus_base_kv_ll"][18] = 138
    with pytest.raises(ValueError, match="conflicts"):
        export_model(source, tmp_path / "conflict", assumptions=spec, voltage_profile=IEEE57_VOLTAGE_BASE_PROFILE_ID)
    assert not (tmp_path / "conflict").exists()
