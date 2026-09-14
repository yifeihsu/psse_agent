"""Logical branch/coupler identities and electrical contraction regressions."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from pypower.makeYbus import makeYbus

from psse_env.systems import resolve_system
from logical_topology.inventory import (
    UnknownStatusError, build_inventory, inventory_hash, process_topology,
    validate_inventory, validate_statuses,
)


def _admittances(case):
    bus, branch = case["bus"].copy(), case["branch"].copy()
    bus[:, 0] -= 1
    branch[:, :2] -= 1
    return tuple(matrix.toarray() for matrix in makeYbus(case["baseMVA"], bus, branch))


def _terminal_admittance(row):
    y, charging = 1 / complex(row[2], row[3]), .5j * row[4]
    tap = (row[8] or 1) * np.exp(1j * np.deg2rad(row[9]))
    return np.array([[(y + charging)/abs(tap)**2, -y/np.conj(tap)],
                     [-y/tap, y + charging]])


@pytest.mark.parametrize("system,couplers,nodes", [("case57", 13, 70), ("case14", 5, 19)])
def test_frozen_inventory_closed_contraction_is_exact_canonical_case(system, couplers, nodes):
    spec = resolve_system(system)
    source = spec.load_case()
    inventory = build_inventory(system)
    assert len(inventory["branches"]) == spec.nl
    assert len(inventory["couplers"]) == couplers
    assert len(inventory["nodes"]) == nodes
    assert [r["asset_id"] for r in inventory["branches"]] == [b.asset_id for b in spec.branches]
    assert inventory_hash(inventory) == inventory["layout_hash"]
    assert build_inventory(system) == inventory
    assert json.loads(json.dumps(inventory)) == inventory
    processed = process_topology(source, inventory, inventory["normal_statuses"])
    for key in ("bus", "branch", "gen", "gencost"):
        np.testing.assert_array_equal(processed["case"][key], source[key])
    for actual, expected in zip(_admittances(processed["case"]), _admittances(source)):
        np.testing.assert_array_equal(actual, expected)
    assert processed["connectivity"]["connected"] is True
    for node in inventory["nodes"]:
        assert processed["node_to_bus"][node["node_id"]] == node["base_bus"]
        assert processed["node_to_row0"][node["node_id"]] == node["base_bus"] - 1


def test_mandated_bus4_partition_and_parallel_assets_survive_open_coupler():
    source, inventory = resolve_system("case57").load_case(), build_inventory("case57")
    status = dict(inventory["normal_statuses"], **{"case57:bus:4:coupler": 0})
    result = process_topology(source, inventory, status)
    assert result["case"]["bus"].shape == (58, 13)
    assert result["case"]["branch"].shape == source["branch"].shape
    assert result["node_to_bus"]["case57:bus:4:A"] == 4
    assert result["node_to_bus"]["case57:bus:4:B"] == 58
    assert result["bus_to_nodes"][58] == ["case57:bus:4:B"]
    branch = result["case"]["branch"]
    assert branch[2, 1] == 4  # 3-4 on A
    assert branch[3, 0] == 4  # 4-5 on A
    np.testing.assert_array_equal(branch[[4, 18, 19], 0], [58, 58, 58])
    assert inventory["branches"][18]["asset_id"] != inventory["branches"][19]["asset_id"]
    np.testing.assert_array_equal(branch[:, 2:], source["branch"][:, 2:])
    np.testing.assert_array_equal(result["case"]["gen"], source["gen"])
    assert result["connectivity"]["connected"] is True


@pytest.mark.parametrize("row", [0, 18, 19, 34, 35])
def test_asset_status_removes_complete_terminal_admittance_with_charging_and_taps(row):
    source, inventory = resolve_system("case57").load_case(), build_inventory("case57", split_buses=[])
    status = dict(inventory["normal_statuses"])
    status[inventory["branches"][row]["device_id"]] = 0
    compiled = process_topology(source, inventory, status)["case"]
    original_y, _, _ = _admittances(source)
    switched_y, from_y, to_y = _admittances(compiled)
    expected_delta = np.zeros_like(original_y)
    endpoints = source["branch"][row, :2].astype(int) - 1
    expected_delta[np.ix_(endpoints, endpoints)] = _terminal_admittance(source["branch"][row])
    np.testing.assert_allclose(original_y - switched_y, expected_delta, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(from_y[row], np.zeros(57))
    np.testing.assert_array_equal(to_y[row], np.zeros(57))
    kept_columns = [i for i in range(source["branch"].shape[1]) if i != 10]
    np.testing.assert_array_equal(compiled["branch"][:, kept_columns], source["branch"][:, kept_columns])
    if row in (18, 19, 34, 35):
        partner = {18: 19, 19: 18, 34: 35, 35: 34}[row]
        assert compiled["branch"][partner, 10] == 1
        assert np.linalg.norm(from_y[partner]) > 0


def test_cumulative_multiple_couplers_and_asset_status_do_not_reset_other_edits():
    source, inventory = resolve_system("case57").load_case(), build_inventory()
    source["branch"][12, 2:4] *= [1.4, .7]
    source["gen"][0, 1:3] += [5., 3.]
    source["gencost"][0, -1] += 10
    status = dict(inventory["normal_statuses"])
    status["case57:bus:1:coupler"] = status["case57:bus:4:coupler"] = 0
    status["case57:branch:20:status_cb"] = 0
    first = process_topology(source, inventory, status)
    assert first["case"]["bus"].shape[0] == 59
    assert first["node_to_bus"]["case57:bus:1:B"] == 58
    assert first["node_to_bus"]["case57:bus:4:B"] == 59
    assert np.count_nonzero(first["case"]["bus"][:, 1] == 3) == 1
    assert first["case"]["bus"][57, 1] == 1
    np.testing.assert_array_equal(first["case"]["gen"], source["gen"])
    np.testing.assert_array_equal(first["case"]["gencost"], source["gencost"])
    np.testing.assert_array_equal(first["case"]["branch"][:, 2:4], source["branch"][:, 2:4])
    status["case57:bus:1:coupler"] = 1
    second = process_topology(source, inventory, status)
    assert second["case"]["bus"].shape[0] == 58
    assert second["node_to_bus"]["case57:bus:4:B"] == 58  # current numeric map, fixed logical identity
    assert second["case"]["branch"][19, 10] == 0
    assert second["statuses"]["case57:bus:4:coupler"] == 0
    np.testing.assert_array_equal(second["case"]["branch"][:, 2:4], source["branch"][:, 2:4])


def test_ieee14_shunt_is_intact_and_load_allocation_preserves_both_p_and_q():
    source, inventory = resolve_system("case14").load_case(), build_inventory("case14")
    source["bus"][:, 2:4] *= .85
    status = dict(inventory["normal_statuses"], **{"case14:bus:9:coupler": 0})
    result = process_topology(source, inventory, status)
    section_a, section_b = (result["node_to_row0"][f"case14:bus:9:{s}"] for s in ("A", "B"))
    np.testing.assert_array_equal(result["case"]["bus"][section_a, 4:6], source["bus"][8, 4:6])
    np.testing.assert_array_equal(result["case"]["bus"][section_b, 4:6], [0, 0])
    for row in (section_a, section_b):
        np.testing.assert_array_equal(result["case"]["bus"][row, 2:4], source["bus"][8, 2:4] / 2)
    np.testing.assert_allclose(result["case"]["bus"][:, 2:6].sum(axis=0), source["bus"][:, 2:6].sum(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result["case"]["gen"], source["gen"])


def test_islanding_is_reported_without_new_reference_sources_or_shedding():
    source, inventory = resolve_system("case57").load_case(), build_inventory(split_buses=[])
    status = dict(inventory["normal_statuses"])
    for row, branch in zip(inventory["branches"], source["branch"]):
        if 57 in branch[:2]:
            status[row["device_id"]] = 0
    result = process_topology(source, inventory, status)
    assert result["connectivity"]["connected"] is False
    assert result["connectivity"]["component_count"] == 2
    island = next(c for c in result["connectivity"]["components"] if c["bus_ids"] == [57])
    assert island["has_load"] is True and island["online_generator_rows0"] == []
    assert island["reference_bus_ids"] == []
    assert island["pd_mw"] == source["bus"][56, 2]
    np.testing.assert_array_equal(result["case"]["bus"], source["bus"])
    np.testing.assert_array_equal(result["case"]["gen"], source["gen"])


@pytest.mark.parametrize("value", [None, "unknown", .5, True, 2])
def test_unknown_or_invalid_status_never_defaults_closed(value):
    inventory = build_inventory()
    statuses = dict(inventory["normal_statuses"])
    statuses["case57:bus:4:coupler"] = value
    with pytest.raises(UnknownStatusError if value is None else ValueError):
        process_topology(resolve_system("case57").load_case(), inventory, statuses)


def test_status_vector_must_be_complete_and_have_no_unknown_device():
    inventory = build_inventory()
    statuses = dict(inventory["normal_statuses"])
    statuses.pop("case57:branch:1:status_cb")
    with pytest.raises(UnknownStatusError):
        validate_statuses(inventory, statuses)
    statuses = {**inventory["normal_statuses"], "case57:branch:unknown": 1}
    with pytest.raises(ValueError, match="Unregistered"):
        validate_statuses(inventory, statuses)


def test_inventory_mutation_and_unrelated_bus_merge_fail_closed():
    inventory = build_inventory()
    inventory["loads"][0]["fraction"] = .25
    with pytest.raises(ValueError, match="hash"):
        validate_inventory(inventory)
    inventory = build_inventory()
    inventory["couplers"][0]["node_b"] = "case57:bus:4:B"
    inventory["layout_hash"] = inventory_hash(inventory)
    with pytest.raises(ValueError, match="one base bus"):
        validate_inventory(inventory)


def test_branch_only_bridge_and_input_objects_are_not_mutated():
    for system in ("case14", "case57"):
        source, inventory = resolve_system(system).load_case(), build_inventory(system, split_buses=[])
        before_case, before_inventory = copy.deepcopy(source), copy.deepcopy(inventory)
        assert not inventory["couplers"]
        assert len(inventory["nodes"]) == len(source["bus"])
        process_topology(source, inventory, inventory["normal_statuses"])
        assert inventory == before_inventory
        for key in ("bus", "branch", "gen", "gencost"):
            np.testing.assert_array_equal(source[key], before_case[key])


def test_expanded_electrical_case_cannot_be_mistaken_for_canonical_equipment_basis():
    source, inventory = resolve_system("case57").load_case(), build_inventory()
    statuses = dict(inventory["normal_statuses"], **{"case57:bus:4:coupler": 0})
    expanded = process_topology(source, inventory, statuses)["case"]
    with pytest.raises(ValueError, match="canonical equipment/base-bus basis"):
        process_topology(expanded, inventory, inventory["normal_statuses"])
