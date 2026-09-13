"""Real OPF/PF source tests and label/evidence separation for logical topology."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pypower.api import ppoption, runpf
from pypower.makeYbus import makeYbus

from psse_env.systems import resolve_system
from logical_topology.inventory import build_inventory, process_topology
from logical_topology.measurements import expected_measurements
from logical_topology import scenarios


def _read(path):
    return json.loads(Path(path).read_text())


def _case(value):
    result = copy.deepcopy(value)
    for key in ("bus", "gen", "branch", "gencost"):
        result[key] = np.asarray(result[key], dtype=float)
    return result


@pytest.fixture(scope="module", params=("case14", "case57"))
def corpus(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("logical_scenarios") / request.param
    original_solver = scenarios.solve_true_world
    calls = []

    def frozen_then_solve(source, inventory, statuses):
        assert len(list((directory / "inventories").glob("*.json"))) == 2
        assert len(list((directory / "sensors").glob("*.json"))) == 8
        calls.append((inventory["layout_hash"], scenarios.digest(statuses)))
        return original_solver(source, inventory, statuses)

    # A wrong-model WLS/teacher failure must never gate generation of true physics.
    with patch("logical_topology.scenarios.solve_true_world", side_effect=frozen_then_solve), \
            patch("logical_topology.estimation.estimate", side_effect=AssertionError("WLS must not gate physical source admission")), \
            patch("psse_env.oracle.expert_policy.ExpertPolicyOracle.__init__", side_effect=AssertionError("Teacher must not gate physical source admission")):
        manifest = scenarios.build_corpus(directory, system=request.param, load_scales=(.8,), seed=31415, smoke=True)
    assert len(calls) == len(set(calls)) == manifest["physical_world_count"]
    return directory, manifest


def _admitted_rows(corpus):
    return [row for row in corpus[1]["rows"] if row["physical_admission"]["admitted"]]


def test_smoke_build_retains_candidates_and_admits_only_checked_true_worlds(corpus):
    directory, manifest = corpus
    assert manifest["scenario_count"] == len(manifest["rows"]) >= 60
    assert manifest["physical_world_count"] == len({row["parent_physical_root"] for row in manifest["rows"]}) >= 8
    assert manifest["config"]["physical_admission_uses_wls_or_teacher"] is False
    assert manifest["all_candidates_retained_in_manifest"] is True
    admitted = _admitted_rows(corpus)
    assert admitted and manifest["scenario_count_physically_admitted"] == len(admitted)
    assert any(row["family"] == "inclusion" for row in admitted)
    assert any(row["family"] == "merging" for row in admitted)
    for row in manifest["rows"]:
        physical = _read(directory / row["physical_audit_path"])
        assert physical["admitted"] is row["physical_admission"]["admitted"]
        if physical["admitted"]:
            assert physical["solution"]["success"] is True
            assert physical["physics"]["passed"] is True
            assert all(value["passed"] for value in physical["physics"]["checks"].values())
            assert "execution" in row and "observations_hash" in row
        else:
            assert "execution" not in row
            assert physical["reason"] in {
                "islanding_outside_connected_operating_scope", "opf_nonconvergence_not_a_proof_of_infeasibility",
                "post_opf_power_flow_nonconvergence", "physical_solution_contract_failed", "physical_solver_exception",
            }


def test_both_branch_error_directions_keep_true_physics_and_offline_zero_flows(corpus):
    directory, _ = corpus
    for row in _admitted_rows(corpus):
        if row["family"] not in {"inclusion", "exclusion"}:
            continue
        execution = row["execution"]
        inventory = _read(directory / execution["inventory_path"])
        sensors = _read(directory / execution["measurement_inventory_path"])
        physical = _read(directory / row["physical_audit_path"])
        model_case = _case(_read(directory / execution["base_case_path"]))
        target = next(item for item in inventory["branches"] if item["device_id"] == row["error_device_ids"][0])
        index = target["row0"]
        if row["family"] == "inclusion":
            assert row["true_statuses"][target["device_id"]] == 0
            assert row["model_statuses"][target["device_id"]] == 1
            assert physical["solution"]["branch"][index][10] == 0
            np.testing.assert_array_equal(physical["solution"]["branch"][index][13:17], [0, 0, 0, 0])
        else:
            assert row["true_statuses"][target["device_id"]] == 1
            assert row["model_statuses"][target["device_id"]] == 0
            assert physical["solution"]["branch"][index][10] == 1
        assert model_case["branch"][index, 10] == row["model_statuses"][target["device_id"]]
        source = _case(physical["operating_case"])
        preserved = [column for column in range(model_case["branch"].shape[1]) if column != 10]
        np.testing.assert_array_equal(model_case["branch"][:, preserved], source["branch"][:, preserved])
        np.testing.assert_array_equal(model_case["gen"], source["gen"])
        expected = expected_measurements(source, inventory, row["true_statuses"], _case(physical["solution"]), sensors)
        meter_indices = [i for i, meter in enumerate(sensors["records"]) if meter.get("branch_row0") == index]
        assert len(meter_indices) == 4
        if row["family"] == "inclusion":
            assert all(expected[index] == 0.0 for index in meter_indices)
        observations = _read(directory / execution["observations_path"])
        assert len(observations["values"]) == len(sensors["records"])
        if row["measurement_profile"] == "direct":
            assert all(sensors["records"][index]["available"] for index in meter_indices)
        else:
            assert not any(sensors["records"][index]["available"] for index in meter_indices)
            assert all(observations["values"][index] is None for index in meter_indices)


def test_reported_labels_never_regenerate_physics_or_measurement_noise(corpus):
    directory, _ = corpus
    grouped = defaultdict(list)
    for row in _admitted_rows(corpus):
        if not row.get("measurement_error"):
            grouped[(row["parent_physical_root"], row["measurement_profile"])].append(row)
    compared = 0
    for rows in grouped.values():
        hashes = {row["observations_hash"] for row in rows}
        assert len(hashes) == 1
        if len(rows) > 1:
            compared += 1
            first = _read(directory / rows[0]["execution"]["observations_path"])
            for row in rows[1:]:
                assert _read(directory / row["execution"]["observations_path"]) == first
    assert compared >= 2


def test_execution_observations_redact_unavailable_values_and_preserve_common_sensor_draws(corpus):
    directory, _ = corpus
    rows = _admitted_rows(corpus)
    for row in rows:
        sensors = _read(directory / row["execution"]["measurement_inventory_path"])
        observed = _read(directory / row["execution"]["observations_path"])
        assert set(observed) == {"contract", "values", "sensor_ids", "sensor_inventory_hash"}
        assert observed["sensor_ids"] == [sensor["sensor_id"] for sensor in sensors["records"]]
        for sensor, value in zip(sensors["records"], observed["values"]):
            if sensor["available"]:
                assert isinstance(value, (int, float)) and np.isfinite(value)
            else:
                assert value is None
        if row["measurement_profile"] in {"direct", "voltage_only"} or row.get("measurement_error"):
            continue
        direct = next(other for other in rows if other["parent_physical_root"] == row["parent_physical_root"]
                      and other["measurement_profile"] == "direct" and not other.get("measurement_error"))
        direct_values = _read(directory / direct["execution"]["observations_path"])["values"]
        for index, sensor in enumerate(sensors["records"]):
            if sensor["available"]:
                assert observed["values"][index] == direct_values[index]


def test_mixed_overlays_touch_one_declared_meter_and_active_parameter_only(corpus):
    directory, _ = corpus
    admitted = _admitted_rows(corpus)
    for row in admitted:
        if row.get("measurement_error"):
            baseline = next(other for other in admitted if other["parent_physical_root"] == row["parent_physical_root"]
                            and other["measurement_profile"] == row["measurement_profile"] and not other.get("measurement_error"))
            values = np.asarray(_read(directory / row["execution"]["observations_path"])["values"])
            original = np.asarray(_read(directory / baseline["execution"]["observations_path"])["values"])
            error = row["measurement_error"]
            different = np.flatnonzero(values != original).tolist()
            assert different == [error["index0"]]
            assert values[error["index0"]] - original[error["index0"]] == pytest.approx(error["bias_pu"])
            assert error["sigma_multiple"] == 10
            assert error["graph_distance"] == 0 if error["relationship"] == "nearby" else error["graph_distance"] >= 3
        if row.get("parameter_error"):
            error = row["parameter_error"]
            inventory = _read(directory / row["execution"]["inventory_path"])
            target = inventory["branches"][error["branch_row0"]]
            assert row["true_statuses"][target["device_id"]] == 1
            assert error["physically_identifiable_label"] is True
            current = _case(_read(directory / row["execution"]["base_case_path"]))
            physical = _read(directory / row["physical_audit_path"])
            np.testing.assert_array_equal(current["branch"][error["branch_row0"], 2:4], np.asarray(error["true_r_x"]) * error["factor"])
            assert scenarios.audit_physical_solution(_case(physical["solution"]))["passed"]


def test_dispatch_copy_reproduces_true_pf_without_manufacturing_pv_controls(corpus):
    directory, _ = corpus
    # One true open-coupler operating point tests the canonical/expanded row boundary.
    row = next(row for row in _admitted_rows(corpus) if row["family"] == "merging")
    physical = _read(directory / row["physical_audit_path"])
    operating, truth = _case(physical["operating_case"]), _case(physical["solution"])
    spec = resolve_system(corpus[1]["config"]["system"]["case_id"])
    np.testing.assert_array_equal(operating["gen"][:, 0], spec.load_case()["gen"][:, 0])
    np.testing.assert_array_equal(operating["gen"][:, 1:21], truth["gen"][:, 1:21])
    inventory = _read(directory / row["execution"]["inventory_path"])
    rebuilt = process_topology(operating, inventory, row["true_statuses"])["case"]
    resolved, success = runpf(rebuilt, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
    assert success
    np.testing.assert_allclose(resolved["bus"][:, 7:9], truth["bus"][:, 7:9], atol=1e-7, rtol=0)
    np.testing.assert_allclose(resolved["gen"][:, 1:3], truth["gen"][:, 1:3], atol=1e-5, rtol=0)
    corrupt = copy.deepcopy(truth)
    corrupt["gen"][1, 1] += 3.0
    assert not scenarios.audit_physical_solution(corrupt)["passed"]


def test_source_parent_groups_do_not_cross_default_or_structural_splits(corpus):
    _, manifest = corpus
    for field in ("split", "structural_split"):
        by_parent = defaultdict(set)
        for row in manifest["rows"]:
            by_parent[row["parent_physical_root"]].add(row[field])
        assert all(len(splits) == 1 for splits in by_parent.values())
    heldout = set(device for devices in manifest["config"]["structural_holdout_device_ids"].values() for device in devices)
    parents = {row["parent_physical_root"] for row in manifest["rows"] if heldout.intersection(row["error_device_ids"])}
    assert all(row["structural_split"] == "structural_test" for row in manifest["rows"] if row["parent_physical_root"] in parents)


def test_seed_changes_scenario_noise_view_but_not_parent_or_physical_identity(corpus, tmp_path):
    directory, initial = corpus
    changed = scenarios.build_corpus(tmp_path / "new_seed", system=initial["config"]["system"]["case_id"],
                                     load_scales=(.8,), seed=27182, smoke=True)
    for before, after in zip(initial["rows"], changed["rows"]):
        assert before["parent_physical_root"] == after["parent_physical_root"]
        assert before["split"] == after["split"]
        assert before["scenario_id"] != after["scenario_id"]
        if before["physical_admission"]["admitted"] and after["physical_admission"]["admitted"]:
            assert before["physical_root_fingerprint"] == after["physical_root_fingerprint"]
            assert before["observations_hash"] != after["observations_hash"]


def test_true_islands_are_retained_as_rejections_without_calling_opf():
    inventory = build_inventory("case57", split_buses=[])
    source = resolve_system("case57").load_case()
    statuses = {device: 0 for device in inventory["normal_statuses"]}
    with patch("logical_topology.scenarios.solve_ac_opf", side_effect=AssertionError("Must not invent island sources")) as solver:
        result = scenarios.solve_true_world(source, inventory, statuses)
    assert result["admitted"] is False
    assert result["reason"] == "islanding_outside_connected_operating_scope"
    assert result["connectivity"]["component_count"] == 57
    solver.assert_not_called()


def test_ieee14_normal_logical_bridge_matches_detailed_schematic_electrically():
    from Transmission.ieee14_full_topology import topology_to_matpower

    source = resolve_system("case14").load_case()
    inventory = build_inventory("case14")
    logical = process_topology(source, inventory, inventory["normal_statuses"])["case"]
    detailed, info = topology_to_matpower(source)
    assert info["inactive_empty_busbars"] == []
    for key in ("bus", "gen", "branch", "gencost"):
        np.testing.assert_array_equal(logical[key], detailed[key])
    matrices = []
    solutions = []
    for case in (logical, detailed):
        bus, branch = case["bus"].copy(), case["branch"].copy()
        bus[:, 0] -= 1
        branch[:, :2] -= 1
        matrices.append(tuple(matrix.toarray() for matrix in makeYbus(case["baseMVA"], bus, branch)))
        solved, success = runpf(case, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
        assert success
        solutions.append(solved)
    for first, second in zip(*matrices):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(solutions[0]["bus"], solutions[1]["bus"])
    np.testing.assert_array_equal(solutions[0]["branch"], solutions[1]["branch"])
    # This comparison intentionally covers NORMAL configurations only. Detailed
    # one-end-open lines retain charging and are not whole-asset CB equivalents.
