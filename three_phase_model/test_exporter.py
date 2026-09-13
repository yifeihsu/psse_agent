"""Regression checks at the canonical-case/OpenDSS conversion boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from psse_env.systems import resolve_system
from three_phase_model.exporter import export_model, load_assumptions
from three_phase_model.runtime import compile_model
from three_phase_model.validation import (
    positive_sequence_terminal_admittance,
    reference_terminal_admittance,
    validate_model,
)


@pytest.fixture(scope="module")
def diagonal57(tmp_path_factory):
    return export_model(
        resolve_system("case57").load_case(), tmp_path_factory.mktemp("dss57") / "model",
        case_id="case57", assumptions=load_assumptions("normalized_diagonal"),
    )


def _compile(model):
    return compile_model(Path(model["output_dir"]) / "Master.dss")


def test_case57_preserves_parallel_and_nominal_tap_transformer_assets(diagonal57):
    source = resolve_system("case57").load_case()
    branches = diagonal57["registry"]["branches"]
    assert len(branches) == 80
    assert len({branch["asset_id"] for branch in branches}) == 80
    assert [branch["asset_id"] for branch in branches] == [
        branch.asset_id for branch in resolve_system("case57").branches
    ]
    assert len({branch["dss_element"] for branch in branches}) == 80
    assert diagonal57["manifest"]["base_case_hash"] == resolve_system("case57").base_case_hash
    assert diagonal57["manifest"]["line_count"] == 63
    assert diagonal57["manifest"]["transformer_count"] == 17
    for rows, pair, taps in (((18, 19), (4, 18), (0.970, 0.978)), ((34, 35), (24, 25), (1.0, 1.0))):
        first, second = (branches[row] for row in rows)
        assert (first["from_bus"], first["to_bus"]) == pair
        assert (second["from_bus"], second["to_bus"]) == pair
        assert (first["circuit_ordinal"], second["circuit_ordinal"]) == (1, 2)
        for branch, row, tap in zip((first, second), rows, taps):
            assert branch["dss_element"].startswith("Transformer.")
            assert branch["branch_row0"] == row
            assert branch["source_tap"] == tap
            assert branch["tap"] == tap
            assert branch["x_pu"] == source["branch"][row, 3]
            assert branch["from_terminal"] == 1 and branch["to_terminal"] == 2


def test_phase_voltage_power_and_shunt_bases_match_source_totals(diagonal57):
    reference = diagonal57["reference"]
    registry = diagonal57["registry"]
    engine = _compile(diagonal57)
    assert diagonal57["manifest"]["external_phase_node_count"] == 171
    assert all(bus["kv_ll"] == 1.0 for bus in registry["buses"])
    assert all(bus["source_base_kv"] == 0.0 for bus in registry["buses"])
    for bus in reference["bus"]:
        rows = [item for item in registry["loads"] if item["bus"] == int(bus[0])]
        assert sum(item["kw"] for item in rows) == pytest.approx(bus[2] * 1000)
        assert sum(item["kvar"] for item in rows) == pytest.approx(bus[3] * 1000)
        if rows:
            assert {item["phase"] for item in rows} == {1, 2, 3}
            for item in rows:
                engine.Loads.Name(item["element"].split(".", 1)[1])
                assert engine.Loads.kV() == pytest.approx(1 / np.sqrt(3))
    assert {item["bus"]: item["bs_mvar"] for item in registry["shunts"]} == {18: 10.0, 25: 5.9, 53: 6.3}
    for item in registry["shunts"]:
        engine.Capacitors.Name(item["element"].split(".", 1)[1])
        assert engine.Capacitors.kV() == pytest.approx(1.0)
        assert engine.Capacitors.kvar() == pytest.approx(item["bs_mvar"] * 1000)
    for branch in registry["branches"]:
        if branch["dss_element"].startswith("Transformer."):
            engine.Transformers.Name(branch["dss_element"].split(".", 1)[1])
            for winding, tap in ((1, branch["source_tap"]), (2, 1.0)):
                engine.Transformers.Wdg(winding)
                assert engine.Transformers.kV() == pytest.approx(1.0)
                assert engine.Transformers.kVA() == pytest.approx(100000.0)
                assert engine.Transformers.Tap() == pytest.approx(tap)
    report = validate_model(engine, reference, registry, diagonal57["assumptions"])
    assert report["passed"], report["failed_checks"]


def test_coupled_completion_preserves_actual_positive_sequence_yprim(tmp_path, diagonal57):
    coupled = export_model(
        resolve_system("case57").load_case(), tmp_path / "coupled",
        case_id="case57", assumptions=load_assumptions("coupled_sensitivity"),
    )
    engine = _compile(coupled)
    reference = coupled["reference"]
    assert any(abs(row["xmatrix_ohm"][0][1]) > 0 for row in coupled["registry"]["branches"] if "xmatrix_ohm" in row)
    for branch in coupled["registry"]["branches"]:
        actual = positive_sequence_terminal_admittance(
            engine, branch["dss_element"], f"b{branch['from_bus']}", f"b{branch['to_bus']}",
            zbase_ohm=0.01,
        )
        expected = reference_terminal_admittance(reference["branch"][branch["branch_row0"]])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    report = validate_model(engine, reference, coupled["registry"], coupled["assumptions"])
    assert report["passed"], report["failed_checks"]
    np.testing.assert_allclose(reference["bus"], diagonal57["reference"]["bus"], rtol=0, atol=0)


def test_source_case_and_assumptions_are_not_mutated(tmp_path):
    source = resolve_system("case57").load_case()
    source_before = copy.deepcopy(source)
    assumptions = load_assumptions()
    assumptions_before = copy.deepcopy(assumptions)
    result = export_model(source, tmp_path / "immutable", case_id="case57", assumptions=assumptions)
    for key, value in source.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, source_before[key])
        else:
            assert value == source_before[key]
    assert assumptions == assumptions_before
    assert result["manifest"]["snapshot_equivalence_only"] is True
    assert result["manifest"]["pv_control_equivalence"] is False
    assert result["manifest"]["validation_performed"] is False
    serialized = json.loads((tmp_path / "immutable" / "source_case.json").read_text())
    assert all(row[9] == 0 for row in serialized["bus"])


def test_existing_output_is_refused_without_changes(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_bytes(b"existing model evidence")
    with pytest.raises(FileExistsError):
        export_model(resolve_system("case57").load_case(), destination, case_id="case57")
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing model evidence"


@pytest.mark.parametrize("key,value", [
    ("transformer_connection", "delta_wye"), ("negative_sequence", "infer_from_bus_count"),
    ("generator_control", "pv_control"), ("grounding", "unknown"),
    ("line_zero_sequence_r_ratio", 0.0), ("frequency_hz", float("nan")),
    ("transformer_antifloat_ppm", 1.0), ("transformer_magnetizing_percent", 0.1),
    ("constant_pq_voltage_range", [1.0, 1.5]), ("source_z0_pu", [0.0, 0.0]),
    ("transformer_vector_group", "Dyn11"),
])
def test_unsupported_assumptions_fail_before_output(tmp_path, key, value):
    assumptions = load_assumptions()
    assumptions[key] = value
    with pytest.raises(ValueError):
        export_model(resolve_system("case57").load_case(), tmp_path / "invalid", assumptions=assumptions)
    assert not (tmp_path / "invalid").exists()


def test_explicit_empty_assumptions_do_not_silently_select_defaults(tmp_path):
    with pytest.raises(ValueError):
        export_model(resolve_system("case57").load_case(), tmp_path / "invalid", assumptions={})
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("column,value", [(9, 10.0), (10, 2.0), (8, -0.5), (3, 0.0)])
def test_unsupported_branch_fields_fail_before_output(tmp_path, column, value):
    source = resolve_system("case57").load_case()
    source["branch"][0, column] = value
    with pytest.raises(ValueError):
        export_model(source, tmp_path / "invalid", case_id="case57")
    assert not (tmp_path / "invalid").exists()


def test_missing_original_slack_generator_fails_before_output(tmp_path):
    source = resolve_system("case57").load_case()
    source["gen"][0, 7] = 0
    with pytest.raises(ValueError):
        export_model(source, tmp_path / "invalid", case_id="case57")
    assert not (tmp_path / "invalid").exists()


def test_generator_rows_stay_mapped_when_slack_is_not_first(tmp_path):
    source = resolve_system("case57").load_case()
    order = [6, 0, 4, 2, 5, 1, 3]
    source["gen"] = source["gen"][order].copy()
    source["gencost"] = source["gencost"][order].copy()
    result = export_model(source, tmp_path / "shuffled", case_id="case57")
    assert result["registry"]["source"]["gen_rows0"] == [1]
    for index, row in enumerate(source["gen"]):
        assert result["reference"]["gen"][index, 0] == row[0]
        if index == 1:
            continue
        devices = [item for item in result["registry"]["generators"] if item["gen_row0"] == index]
        assert len(devices) == 3
        assert {item["bus"] for item in devices} == {int(row[0])}
        assert sum(item["kw"] for item in devices) == pytest.approx(result["reference"]["gen"][index, 1] * 1000)
        assert sum(item["kvar"] for item in devices) == pytest.approx(result["reference"]["gen"][index, 2] * 1000)
    report = validate_model(_compile(result), result["reference"], result["registry"], result["assumptions"])
    assert report["passed"], report["failed_checks"]


def test_generic_ieee14_resize_does_not_touch_existing_dss_files(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fixtures = [repo / "IEEE_14_OpenDSS" / name for name in (
        "IEEE14Lines.DSS", "IEEE14Trafo.DSS", "IEEE14Loads.DSS", "IEEE14BusMaster.dss",
    )]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fixtures}
    result = export_model(resolve_system("case14").load_case(), tmp_path / "generic14", case_id="case14")
    assert result["manifest"]["external_bus_count"] == 14
    assert result["manifest"]["physical_branch_count"] == 20
    assert result["manifest"]["external_phase_node_count"] == 42
    assert result["manifest"]["transformer_count"] == 3
    report = validate_model(_compile(result), result["reference"], result["registry"], result["assumptions"])
    assert report["passed"], report["failed_checks"]
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fixtures} == before


def test_compilation_uses_independent_contexts(diagonal57):
    first = _compile(diagonal57)
    first.Text.Command("Disable Line.br_0001")
    first.Circuit.SetActiveElement("Line.br_0001")
    assert not first.CktElement.Enabled()
    second = _compile(diagonal57)
    second.Circuit.SetActiveElement("Line.br_0001")
    assert second.CktElement.Enabled()
    first.Circuit.SetActiveElement("Line.br_0001")
    assert not first.CktElement.Enabled()
