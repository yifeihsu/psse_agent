from __future__ import annotations

import copy
import json
import math

import numpy as np
import opendssdirect as odd
import pytest

from three_phase_model.measurements import audit_full_circuit_kcl, extract_measurements, write_layout


def _fixture(*, unbalanced=False):
    dss = odd.NewContext()
    commands = [
        "clear",
        "set defaultbasefrequency=60",
        "new circuit.measurements bus1=n10.1.2.3 bus2=n10.0.0.0 basekv=1 pu=1.03 angle=17 phases=3 r1=0.000001 x1=0.000001 r0=0.000001 x0=0.000001",
        "new linecode.diag nphases=3 units=none rmatrix=[0.002 | 0 0.002 | 0 0 0.002] xmatrix=[0.004 | 0 0.004 | 0 0 0.004] cmatrix=[0 | 0 0 | 0 0 0]",
        "new line.br_0001 phases=3 bus1=n10.1.2.3 bus2=n40.1.2.3 linecode=diag length=1",
        "new line.br_0002 phases=3 bus1=n10.1.2.3 bus2=n40.1.2.3 linecode=diag length=1",
        "new capacitor.sh_40 phases=3 bus1=n40.1.2.3 bus2=n40.0.0.0 conn=wye kv=1 kvar=200",
        "new capacitor.ch_from phases=3 bus1=n10.1.2.3 bus2=n10.0.0.0 conn=wye kv=1 kvar=50",
        "new capacitor.ch_to phases=3 bus1=n40.1.2.3 bus2=n40.0.0.0 conn=wye kv=1 kvar=50",
    ]
    powers = [1800.0, 600.0, 600.0] if unbalanced else [1000.0] * 3
    for phase, power in enumerate(powers, 1):
        commands.append(
            f"new load.ld_{phase} phases=1 bus1=n40.{phase}.0 conn=wye kv={1 / math.sqrt(3):.16g} kw={power} kvar={power/10} model=1 vminpu=0.2 vmaxpu=2"
        )
    commands += ["set voltagebases=[1]", "calcvoltagebases", "set maxiterations=100 tolerance=1e-12 controlmode=off", "solve"]
    for command in commands:
        dss.Text.Command(command)
    registry = {
        "buses": [
            {"row0": 1, "external_bus": 40, "dss_bus": "n40", "kv_ll": 1.0},
            {"row0": 0, "external_bus": 10, "dss_bus": "n10", "kv_ll": 1.0},
        ],
        "branches": [
            {"asset_id": f"br_{row+1:04d}", "branch_row0": row, "from_bus": 10, "to_bus": 40,
             "dss_element": f"Line.br_{row+1:04d}", "from_terminal": 1, "to_terminal": 2,
             "charging_elements": {"from": ["Capacitor.ch_from"], "to": ["Capacitor.ch_to"]} if row else {"from": [], "to": []}}
            for row in range(2)
        ],
        "loads": [{"bus": 40, "phase": phase, "element": f"Load.ld_{phase}"} for phase in (1, 2, 3)],
        "generators": [],
        "shunts": [{"bus": 40, "element": "Capacitor.sh_40", "gs_mw": 0, "bs_mvar": 0.2}],
        "source": {"element": "Vsource.source", "bus": 10, "gen_rows0": [0]},
    }
    return dss, registry, {"base_mva": 100.0, "base_kv_ll": 1.0, "frequency_hz": 60.0}


def _complex(rect):
    return np.asarray(rect)[:, 0] + 1j * np.asarray(rect)[:, 1]


def test_registry_order_parallel_assets_and_layout_do_not_assume_ieee14():
    dss, registry, assumptions = _fixture()
    layout = write_layout(registry, assumptions)
    assert layout["measurement_count"] == 14
    assert layout["bus_order"] == [10, 40]
    assert layout["branch_order"] == ["br_0001", "br_0002"]
    assert [(row["start_index0"], row["stop_index0_exclusive"]) for row in layout["channels"]] == [
        (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14),
    ]
    result = extract_measurements(dss, registry, assumptions)
    assert len(result["measurement_vector"]) == 14
    assert [row["external_bus"] for row in result["three_phase_voltages"]] == [10, 40]
    assert len(result["three_phase_branch_currents"]) == 2
    json.dumps(result, allow_nan=False)


def test_balanced_phase_sequence_power_and_kcl_conventions():
    dss, registry, assumptions = _fixture()
    result = extract_measurements(dss, registry, assumptions)
    assert result["max_kcl_mismatch_pu"] < 1e-9
    for row in result["three_phase_voltages"]:
        assert max(row["vln_sequence_pu"][0], row["vln_sequence_pu"][2]) < 1e-8
        assert row["vln_pu"][0] == pytest.approx(row["vm_positive_sequence_pu"], abs=1e-8)
    for row in result["branch_powers"]:
        for end in ("from", "to"):
            total = complex(**{k: row[f"s_{end}_total_pu"][k] for k in ("real", "imag")})
            phases = _complex(row[f"s_{end}_phase_pu_rect"])
            sequences = _complex(row[f"s_{end}_sequence_pu_rect"])
            assert total == pytest.approx(sum(phases) / 3.0, abs=1e-12)
            assert total == pytest.approx(sum(sequences), abs=1e-12)
            assert total == pytest.approx(sequences[1], abs=1e-10)
    load_bus = result["bus_injections"][1]
    assert load_bus["p_inj_pu"] == pytest.approx(-0.03, abs=1e-10)
    assert load_bus["q_inj_pu"] == pytest.approx(-0.003, abs=1e-10)
    assert load_bus["q_net_into_branches_pu"] > load_bus["q_inj_pu"]
    assert result["legacy_ieee14_compatible_vector"][5] == load_bus["q_net_into_branches_pu"]
    assert result["measurement_vector"][5] == load_bus["q_inj_pu"]


def test_unbalance_total_power_keeps_sequence_contributions_and_node_order():
    dss, registry, assumptions = _fixture(unbalanced=True)
    result = extract_measurements(dss, registry, assumptions)
    assert result["max_kcl_mismatch_pu"] < 1e-9
    assert result["bus_injections"][1]["p_inj_pu"] == pytest.approx(-0.03, abs=1e-10)
    difference = []
    for row in result["branch_powers"]:
        for end in ("from", "to"):
            total = complex(row[f"p_{end}_pu"], row[f"q_{end}_pu"])
            sequences = _complex(row[f"s_{end}_sequence_pu_rect"])
            assert total == pytest.approx(sum(sequences), abs=1e-12)
            difference.append(abs(total - sequences[1]))
    assert max(difference) > 1e-7
    # Phase B's device must land in index1; conductor0 (ground) is excluded.
    phase_b = result["load_powers"][1]
    assert phase_b["phase_nodes"] == [2]
    assert _complex(phase_b["phase_injection_pu_rect"])[0] == 0j
    assert _complex(phase_b["phase_injection_pu_rect"])[1].real < 0
    assert _complex(phase_b["phase_injection_pu_rect"])[2] == 0j


def test_missing_or_misoriented_assets_fail_closed():
    dss, registry, assumptions = _fixture()
    bad = copy.deepcopy(registry)
    bad["branches"][0]["from_terminal"] = 2
    with pytest.raises(ValueError, match="orientation mismatch"):
        extract_measurements(dss, bad, assumptions)
    bad = copy.deepcopy(registry)
    bad["branches"][0]["dss_element"] = "Line.missing"
    with pytest.raises(ValueError, match="Missing OpenDSS element"):
        extract_measurements(dss, bad, assumptions)
    bad = copy.deepcopy(registry)
    bad["branches"][1]["asset_id"] = bad["branches"][0]["asset_id"]
    with pytest.raises(ValueError, match="Duplicate branch asset"):
        write_layout(bad, assumptions)


def test_generated_ieee57_vector_matches_solved_positive_sequence_reference(tmp_path):
    from psse_env.systems import resolve_system
    from three_phase_model.exporter import export_model
    from three_phase_model.runtime import compile_model

    built = export_model(resolve_system("case57").load_case(), tmp_path / "model", case_id="case57")
    dss = compile_model(tmp_path / "model" / "Master.dss")
    result = extract_measurements(dss, built["registry"], built["assumptions"])
    reference = built["reference"]
    bus, branch, gen = reference["bus"], reference["branch"], reference["gen"]
    rows = {int(row[0]): index for index, row in enumerate(bus)}
    injection = -bus[:, 2].astype(complex) - 1j * bus[:, 3]
    for generator in gen:
        if generator[7] > 0:
            injection[rows[int(generator[0])]] += generator[1] + 1j * generator[2]
    base = float(reference["baseMVA"])
    expected = np.r_[bus[:, 7], injection.real / base, injection.imag / base,
                     branch[:, 13] / base, branch[:, 14] / base,
                     branch[:, 15] / base, branch[:, 16] / base]
    assert len(result["three_phase_voltages"]) == 57
    assert len(result["three_phase_branch_currents"]) == 80
    assert len(result["measurement_vector"]) == 491
    np.testing.assert_allclose(result["measurement_vector"], expected, atol=1e-8, rtol=0)
    assert result["max_kcl_mismatch_pu"] < 1e-8
    # Under this balanced operating point, the source shunts must remain the
    # only distinction between the explicit canonical and legacy injections.
    difference = np.asarray(result["legacy_ieee14_compatible_vector"]) - np.asarray(result["measurement_vector"])
    assert set(np.flatnonzero(np.abs(difference) > 1e-8)) == {2 * 57 + rows[bus_id] for bus_id in (18, 25, 53)}


def _split_fixture_line(dss, *, fault_enabled):
    for command in (
        "edit line.br_0001 enabled=no",
        "new line.hidden_from phases=3 bus1=n10.1.2.3 bus2=hidden_mid.1.2.3 linecode=diag length=0.37",
        "new line.hidden_to phases=3 bus1=hidden_mid.1.2.3 bus2=n40.1.2.3 linecode=diag length=0.63",
        f"new fault.hidden_fault phases=1 bus1=hidden_mid.2 bus2=hidden_mid.0 r=0.1 enabled={'yes' if fault_enabled else 'no'}",
        "solve",
    ):
        dss.Text.Command(command)
    return {"br_0001": {
        "from": {"element": "Line.hidden_from", "terminal": 1, "charging_elements": []},
        "to": {"element": "Line.hidden_to", "terminal": 2, "charging_elements": []},
    }}


def test_split_line_external_measurements_preserve_order_and_hide_fault_metadata():
    dss, registry, assumptions = _fixture()
    reference = extract_measurements(dss, registry, assumptions)
    overrides = _split_fixture_line(dss, fault_enabled=False)
    null = extract_measurements(dss, registry, assumptions, branch_overrides=overrides)
    np.testing.assert_allclose(null["measurement_vector"], reference["measurement_vector"], atol=1e-10, rtol=0)
    dss.Text.Command("edit fault.hidden_fault enabled=yes")
    dss.Text.Command("solve")
    faulted = extract_measurements(dss, registry, assumptions, branch_overrides=overrides)
    assert faulted["measurement_layout"] == reference["measurement_layout"]
    assert len(faulted["measurement_vector"]) == 14
    assert [row["branch"] for row in faulted["three_phase_branch_currents"]] == ["Line.br_0001", "Line.br_0002"]
    assert len(faulted["three_phase_voltages"]) == 2
    serialized = json.dumps(faulted, allow_nan=False).lower()
    for forbidden in ("hidden_mid", "hidden_from", "hidden_to", "hidden_fault", "branch_overrides", "fault_enabled"):
        assert forbidden not in serialized
    assert faulted["max_kcl_mismatch_pu"] < 1e-8
    full_kcl = audit_full_circuit_kcl(dss, assumptions)
    assert full_kcl["node_count"] == 9
    assert full_kcl["policy_observable"] is False
    assert full_kcl["max_kcl_mismatch_pu"] < 1e-8
    assert "all_active_dss_nodes_kcl_offline_v1" not in serialized
    changed = np.max(np.abs(np.asarray(faulted["measurement_vector"]) - np.asarray(null["measurement_vector"])))
    assert changed > 1e-4


def test_hidden_internal_terminal_cannot_replace_external_meter_terminal():
    dss, registry, assumptions = _fixture()
    overrides = _split_fixture_line(dss, fault_enabled=True)
    overrides["br_0001"]["from"]["terminal"] = 2
    with pytest.raises(ValueError, match="orientation mismatch"):
        extract_measurements(dss, registry, assumptions, branch_overrides=overrides)
    with pytest.raises(ValueError, match="unregistered asset"):
        extract_measurements(dss, registry, assumptions, branch_overrides={"unknown_asset": {}})
    with pytest.raises(ValueError, match="Incomplete branch override"):
        extract_measurements(dss, registry, assumptions, branch_overrides={"br_0001": {}})


def test_overridden_endpoint_charging_is_included_in_original_asset_currents():
    dss, registry, assumptions = _fixture()
    reference = extract_measurements(dss, registry, assumptions)
    # Asset2 already has explicit end charging. Reassign equivalent capacitor
    # elements and require the override to preserve total terminal I and S.
    for command in (
        "edit capacitor.ch_from enabled=no",
        "edit capacitor.ch_to enabled=no",
        "new capacitor.hidden_cap_from phases=3 bus1=n10.1.2.3 bus2=n10.0.0.0 conn=wye kv=1 kvar=50",
        "new capacitor.hidden_cap_to phases=3 bus1=n40.1.2.3 bus2=n40.0.0.0 conn=wye kv=1 kvar=50",
        "solve",
    ):
        dss.Text.Command(command)
    overrides = {"br_0002": {
        "from": {"element": "Line.br_0002", "terminal": 1,
                 "charging_elements": [{"element": "Capacitor.hidden_cap_from", "terminal": 1}]},
        "to": {"element": "Line.br_0002", "terminal": 2,
               "charging_elements": [{"element": "Capacitor.hidden_cap_to", "terminal": 1}]},
    }}
    result = extract_measurements(dss, registry, assumptions, branch_overrides=overrides)
    np.testing.assert_allclose(result["measurement_vector"], reference["measurement_vector"], atol=1e-10, rtol=0)
    for side in ("from", "to"):
        np.testing.assert_allclose(result["three_phase_branch_currents"][1][f"i_{side}_pu_rect"],
                                   reference["three_phase_branch_currents"][1][f"i_{side}_pu_rect"], atol=1e-10, rtol=0)
    assert "hidden_cap" not in json.dumps(result)
    assert audit_full_circuit_kcl(dss, assumptions)["max_kcl_mismatch_pu"] < 1e-8


def test_ieee57_charged_split_hif_exports_only_canonical_491_channels(tmp_path):
    from psse_env.systems import resolve_system
    from three_phase_model.disturbances import inject_midspan_hif, set_hif_enabled, restore_midspan_hif
    from three_phase_model.exporter import export_model
    from three_phase_model.runtime import compile_model

    built = export_model(resolve_system("case57").load_case(), tmp_path / "model", case_id="case57")
    dss = compile_model(tmp_path / "model" / "Master.dss")
    registry, assumptions = built["registry"], built["assumptions"]
    pristine = extract_measurements(dss, registry, assumptions)
    receipt = inject_midspan_hif(dss, registry, assumptions, branch_row0=0, alpha=0.37,
                                 phase=2, resistance_pu=10.0, enabled=False)
    overrides = receipt["branch_overrides"]
    null = extract_measurements(dss, registry, assumptions, branch_overrides=overrides)
    np.testing.assert_allclose(null["measurement_vector"], pristine["measurement_vector"], atol=1e-8, rtol=0)
    set_hif_enabled(dss, receipt, True)
    faulted = extract_measurements(dss, registry, assumptions, branch_overrides=overrides)
    assert len(faulted["measurement_vector"]) == 491
    assert len(faulted["three_phase_voltages"]) == 57
    assert len(faulted["three_phase_branch_currents"]) == 80
    assert faulted["measurement_layout"] == pristine["measurement_layout"]
    encoded = json.dumps(faulted, allow_nan=False).lower()
    for hidden_identifier in (receipt["hidden_bus"], receipt["fault_element"], *receipt["created_elements"]):
        assert str(hidden_identifier).lower() not in encoded
    full_kcl = audit_full_circuit_kcl(dss, assumptions)
    assert full_kcl["node_count"] == 174
    assert full_kcl["max_kcl_mismatch_pu"] < 1e-7
    assert faulted["max_kcl_mismatch_pu"] < 1e-7
    restore_midspan_hif(dss, receipt)
    restored = extract_measurements(dss, registry, assumptions)
    np.testing.assert_allclose(restored["measurement_vector"], pristine["measurement_vector"], atol=1e-8, rtol=0)
