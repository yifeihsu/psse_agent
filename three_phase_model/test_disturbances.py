"""Physical HIF injection, exact split controls, and complete restoration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from psse_env.systems import resolve_system
from three_phase_model.disturbances import (
    audit_disturbed_circuit,
    eligible_hif_branch_rows,
    inject_midspan_hif,
    restore_midspan_hif,
    set_hif_enabled,
)
from three_phase_model.exporter import export_model, load_assumptions
from three_phase_model.measurements import extract_measurements
from three_phase_model.runtime import compile_model, solve


@pytest.fixture(scope="module", params=tuple((name, scale)
    for name in ("normalized_diagonal", "coupled_sensitivity") for scale in (.8, 1.0)))
def model(request, tmp_path_factory):
    name, scale = request.param
    source = resolve_system("case57").load_case()
    source["bus"][:, 2:4] *= scale
    return export_model(
        source, tmp_path_factory.mktemp(f"{name}_{scale}") / "model",
        case_id="case57", assumptions=load_assumptions(name),
    )


def _engine(model):
    return compile_model(Path(model["output_dir"]) / "Master.dss")


def _telemetry(dss, model, receipt=None):
    return extract_measurements(dss, model["registry"], model["assumptions"],
                                branch_overrides=None if receipt is None else receipt["branch_overrides"])


@pytest.mark.parametrize("row,alpha,phase", [(0, 0.2, 1), (28, 0.5, 2), (78, 0.8, 3), (42, 0.2, 2)])
def test_split_null_fault_ohms_law_and_full_restoration(model, row, alpha, phase):
    dss = _engine(model)
    before = _telemetry(dss, model)
    receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"],
                                branch_row0=row, alpha=alpha, phase=phase, resistance_pu=10, enabled=False)
    nofault = _telemetry(dss, model, receipt)
    np.testing.assert_allclose(nofault["measurement_vector"], before["measurement_vector"], atol=2e-8, rtol=0)
    assert len(nofault["measurement_vector"]) == 491
    assert receipt["hidden_bus"] not in json.dumps(nofault)
    assert receipt["fault_element"] not in json.dumps(nofault)
    assert "numerical_initialization" not in json.dumps(nofault)
    original_branch = model["registry"]["branches"][row]
    assert set(receipt["branch_overrides"]) == {original_branch["asset_id"]}
    for end, terminal in (("from", 1), ("to", 2)):
        assert receipt["external_terminals"][end]["terminal"] == terminal
    null_audit = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert null_audit["passed"], null_audit
    assert null_audit["active_phase_node_count"] == 174
    assert null_audit["checks"]["no_fault_split_full_abc_admittance"]["max_error_pu"] < 1e-10
    set_hif_enabled(dss, receipt, True)
    faulted = _telemetry(dss, model, receipt)
    assert np.max(np.abs(np.asarray(faulted["measurement_vector"]) - before["measurement_vector"])) > 1e-5
    audit = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert audit["passed"], audit
    assert receipt["resistance_ohm"] == pytest.approx(0.1)
    voltage, current = complex(*audit["fault_voltage_v"]), complex(*audit["fault_current_a"])
    assert current == pytest.approx(voltage / receipt["resistance_ohm"], abs=1e-7)
    assert abs(current) > 1
    set_hif_enabled(dss, receipt, False)
    cleared = _telemetry(dss, model, receipt)
    np.testing.assert_allclose(cleared["measurement_vector"], nofault["measurement_vector"], atol=2e-8, rtol=0)
    restore_midspan_hif(dss, receipt)
    restored = _telemetry(dss, model)
    np.testing.assert_allclose(restored["measurement_vector"], before["measurement_vector"], atol=2e-8, rtol=0)
    restore_audit = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert restore_audit["passed"], restore_audit
    assert restore_audit["active_phase_node_count"] == 171
    assert restore_audit["checks"]["restored_original_full_abc_admittance"]["max_error_pu"] == 0
    with pytest.raises(ValueError, match="restored"):
        set_hif_enabled(dss, receipt, True)


def test_uniform_case57_registry_keeps_ten_pu_default_and_multivoltage_copy_refuses_it(model):
    # case57 is exported on the legacy uniform normalized base: every bus shares one kv_ll,
    # so omitting the resistance still means the historical 10 pu (0.1 model-ohm).
    registry = model["registry"]
    assert len({float(bus["kv_ll"]) for bus in registry["buses"]}) == 1
    dss = _engine(model)
    receipt = inject_midspan_hif(dss, registry, model["assumptions"], branch_row0=0, enabled=False)
    assert receipt["resistance_defaulted"] and receipt["resistance_input_unit"] == "pu"
    assert receipt["resistance_pu"] == pytest.approx(10)
    assert receipt["resistance_ohm"] == pytest.approx(0.1)
    assert receipt["resistance_class"] == "low_resistance_fault"
    assert "declared local base" in receipt["resistance_class_scope"]
    json.dumps(receipt, allow_nan=False)
    restore_midspan_hif(dss, receipt)
    # A registry with more than one voltage base must be told the resistance unit.
    mixed = copy.deepcopy(registry)
    mixed["buses"][-1]["kv_ll"] = 2.0 * float(mixed["buses"][-1]["kv_ll"])
    dss = _engine(model)
    before = dss.Circuit.AllElementNames()
    with pytest.raises(ValueError, match="resistance_ohm or resistance_pu is required for a multi-voltage registry"):
        inject_midspan_hif(dss, mixed, model["assumptions"], branch_row0=0)
    assert dss.Circuit.AllElementNames() == before
    explicit = inject_midspan_hif(dss, mixed, model["assumptions"], branch_row0=0, resistance_pu=10, enabled=False)
    assert not explicit["resistance_defaulted"] and explicit["resistance_pu"] == pytest.approx(10)
    restore_midspan_hif(dss, explicit)


def test_series_sections_retain_full_phase_coupling_and_no_internal_charging(model):
    dss = _engine(model)
    receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"],
                                branch_row0=0, alpha=0.37, enabled=False)
    for end, fraction in (("from", 0.37), ("to", 0.63)):
        dss.Lines.Name(receipt["segments"][end].split(".", 1)[1])
        assert dss.Lines.Length() == pytest.approx(receipt["original_line"]["length"] * fraction)
        np.testing.assert_allclose(np.asarray(dss.Lines.RMatrix()).reshape((3, 3)), receipt["original_line"]["rmatrix"], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.asarray(dss.Lines.XMatrix()).reshape((3, 3)), receipt["original_line"]["xmatrix"], rtol=1e-14, atol=1e-14)
        assert not np.any(dss.Lines.CMatrix())
        assert len(receipt["external_terminals"][end]["charging_elements"]) == 1
    audit = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert audit["passed"], audit


def test_all_63_line_targets_support_exact_split_controls(model):
    rows = eligible_hif_branch_rows(model["registry"])
    assert len(rows) == 63
    assert not set(rows) & {18, 19, 34, 35}  # includes nominal-tap parallel transformer exclusion
    dss = _engine(model)
    before = _telemetry(dss, model)["measurement_vector"]
    for ordinal, row in enumerate(rows):
        receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"],
                                    branch_row0=row, alpha=(0.2, 0.5, 0.8)[ordinal % 3], enabled=False)
        telemetry = _telemetry(dss, model, receipt)
        np.testing.assert_allclose(telemetry["measurement_vector"], before, atol=2e-8, rtol=0,
                                   err_msg=f"No-fault split changed external telemetry on branch row {row}")
        report = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
        assert report["passed"], (row, report)
        restore_midspan_hif(dss, receipt)


@pytest.mark.parametrize("kwargs", [
    {"branch_row0": 34}, {"branch_row0": -1}, {"branch_row0": 0.5},
    {"alpha": 0}, {"alpha": 1}, {"alpha": float("nan")},
    {"phase": 0}, {"phase": 4}, {"phase": True},
    {"resistance_pu": 0}, {"resistance_pu": float("inf")}, {"enabled": "yes"},
])
def test_invalid_injection_is_rejected_before_mutation(model, kwargs):
    dss = _engine(model)
    original_names = dss.Circuit.AllElementNames()
    params = {"branch_row0": 0, **kwargs}
    with pytest.raises(ValueError):
        inject_midspan_hif(dss, model["registry"], model["assumptions"], **params)
    assert dss.Circuit.AllElementNames() == original_names
    dss.Circuit.SetActiveElement("Line.br_0001")
    assert dss.CktElement.Enabled()


def test_engine_only_receipt_replay_and_source_files_unchanged(model):
    directory = Path(model["output_dir"])
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    registry_before = copy.deepcopy(model["registry"])
    first = _engine(model)
    receipt = inject_midspan_hif(first, model["registry"], model["assumptions"],
                                branch_row0=11, alpha=0.41, phase=2, resistance_pu=100)
    expected = _telemetry(first, model, receipt)
    second = _engine(model)
    for command in receipt["commands"]:
        second.Text.Command(command)
    assert second.Solution.Converged()
    actual = _telemetry(second, model, receipt)
    np.testing.assert_allclose(actual["measurement_vector"], expected["measurement_vector"], atol=1e-10, rtol=0)
    assert model["registry"] == registry_before
    assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    json.dumps(receipt, allow_nan=False)


def test_audit_detects_parallel_original_and_wrong_resistor(model):
    dss = _engine(model)
    receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"], branch_row0=0)
    dss.Text.Command(f"Edit {receipt['original_element']} Enabled=yes")
    solve(dss)
    report = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert not report["passed"]
    assert not report["checks"]["original_line_status"]["passed"]
    assert not report["checks"]["active_element_coverage"]["passed"]
    dss.Text.Command(f"Edit {receipt['original_element']} Enabled=no")
    dss.Text.Command(f"Edit {receipt['fault_element']} R={receipt['resistance_ohm'] * 2}")
    solve(dss)
    report = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert not report["passed"]
    assert not report["checks"]["fault_ohms_law"]["passed"]
    assert not report["checks"]["fault_resistive_power"]["passed"]


def test_audit_detects_lost_endpoint_charging(model):
    dss = _engine(model)
    receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"], branch_row0=0, enabled=False)
    capacitor = receipt["external_terminals"]["from"]["charging_elements"][0]["element"]
    dss.Text.Command(f"Edit {capacitor} Cmatrix=[0 | 0 0 | 0 0 0]")
    solve(dss)
    report = audit_disturbed_circuit(dss, receipt, model["registry"], model["assumptions"])
    assert not report["checks"]["no_fault_split_full_abc_admittance"]["passed"]


def test_known_new_node_regression_seeds_actual_healthy_solution_before_solving(model):
    # Full-sweep coupled 0.8, row42/alpha.2 once selected another low-voltage
    # nonlinear solution despite identical pi admittance. Check the initializer
    # explicitly instead of depending on reproducing allocator-history luck.
    dss = _engine(model)
    raw = np.asarray(dss.Circuit.YNodeVArray()).reshape((-1, 2))
    healthy = {name.lower(): complex(*value) for name, value in zip(dss.Circuit.YNodeOrder(), raw)}
    branch = model["registry"]["branches"][42]
    calls = []

    def assert_seed_then_solve(engine):
        actual = np.asarray(engine.Circuit.YNodeVArray()).reshape((-1, 2))
        names = engine.Circuit.YNodeOrder()
        for name, value in zip(names, actual):
            name = name.lower()
            if name in healthy:
                expected = healthy[name]
            else:
                phase = int(name.rsplit(".", 1)[1])
                expected = .8 * healthy[f"b{branch['from_bus']}.{phase}"] + .2 * healthy[f"b{branch['to_bus']}.{phase}"]
            assert complex(*value) == expected
        calls.append(True)
        solve(engine)

    with patch("three_phase_model.disturbances.solve", side_effect=assert_seed_then_solve):
        receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"],
                                    branch_row0=42, alpha=.2, phase=2, resistance_pu=10, enabled=False)
    assert calls == [True]
    assert "CalcVoltageBases" in receipt["commands"]
    assert receipt["numerical_initialization"]["no_fault_external_voltage_max_deviation_pu"] <= 1e-6
    replay = _engine(model)
    for command in receipt["commands"]:
        replay.Text.Command(command)
    first = _telemetry(dss, model, receipt)
    second = _telemetry(replay, model, receipt)
    np.testing.assert_allclose(second["measurement_vector"], first["measurement_vector"], rtol=0, atol=2e-8)
    assert audit_disturbed_circuit(replay, receipt, model["registry"], model["assumptions"])["passed"]


def test_failure_after_new_node_creation_restores_original_and_preserves_error(model):
    dss = _engine(model)
    before = _telemetry(dss, model)["measurement_vector"]
    calls = []

    def fail_once_then_restore(engine):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("forced solve failure after topology mutation")
        solve(engine)

    with patch("three_phase_model.disturbances.solve", side_effect=fail_once_then_restore):
        with pytest.raises(RuntimeError, match="forced solve failure after topology mutation"):
            inject_midspan_hif(dss, model["registry"], model["assumptions"],
                               branch_row0=42, alpha=.2, phase=2, enabled=False)
    assert len(calls) == 2
    dss.Circuit.SetActiveElement("Line.br_0043")
    assert dss.CktElement.Enabled()
    for name in dss.Circuit.AllElementNames():
        if "hif_b0043" in name:
            dss.Circuit.SetActiveElement(name)
            assert not dss.CktElement.Enabled()
    np.testing.assert_allclose(_telemetry(dss, model)["measurement_vector"], before, atol=2e-8, rtol=0)
