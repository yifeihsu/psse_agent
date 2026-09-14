"""Frozen logical asset-status and ideal bus-coupler inventories.

An asset CB multiplies the entire canonical branch admittance, including
charging. A closed coupler contracts two synthetic sections of ONE base bus;
it never creates a small-impedance branch. Input cases remain on the canonical
equipment/base-bus basis so cumulative status edits preserve other corrections.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, PQ, REF
from pypower.idx_gen import GEN_BUS, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS

from psse_env.systems import SystemSpec, resolve_system


class UnknownStatusError(ValueError):
    """A complete electrical topology cannot be compiled from unknown statuses."""

    condition = "unsupported_unknown_status"


def inventory_hash(inventory: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in inventory.items() if key != "layout_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_inventory(
    system: str | SystemSpec = "case57", *, split_buses: Sequence[int] | None = None,
    layout_id: str = "logical_two_section_v1",
) -> dict[str, Any]:
    """Freeze terminal/equipment assignments before operating points or labels.

    By default every bus with at least four canonical branch terminals gets
    two sections, with at least two branch terminals on each. ``split_buses=[]``
    selects the branch-only bridge. Loads are explicitly split 50/50, while
    generators, original voltage controls and shunts stay intact on section A.
    """
    spec = system if isinstance(system, SystemSpec) else resolve_system(system)
    source = spec.load_case()
    if not isinstance(layout_id, str) or not layout_id.strip():
        raise ValueError("layout_id must be a nonempty versioned identifier")
    incident: dict[int, list[str]] = defaultdict(list)
    for branch in spec.branches:
        incident[branch.from_bus].append(branch.asset_id)
        incident[branch.to_bus].append(branch.asset_id)
    eligible = {bus.external_bus for bus in spec.buses if len(incident[bus.external_bus]) >= 4}
    if split_buses is None:
        selected = eligible
    else:
        if any(isinstance(bus, bool) or not isinstance(bus, Integral) for bus in split_buses):
            raise ValueError("split_buses must contain integer base-bus identifiers")
        if len(set(split_buses)) != len(split_buses):
            raise ValueError("split_buses contains duplicates")
        selected = set(map(int, split_buses))
        if not selected <= eligible:
            raise ValueError(f"Every sectioned bus needs at least four terminals: {sorted(selected-eligible)}")
    node = lambda bus, section: f"{spec.case_id}:bus:{bus}:{section}"
    assignments: dict[tuple[int, str], str] = {}
    nodes, couplers, loads = [], [], []
    for bus in spec.buses:
        number = bus.external_bus
        sections = ("A", "B") if number in selected else ("A",)
        for section in sections:
            nodes.append({"node_id": node(number, section), "base_bus": number, "section": section})
            # This is an allocation of the aggregate bus-demand input, including
            # nominally zero inputs; changing operating load does not change layout.
            loads.append({"base_bus": number, "node_id": node(number, section), "fraction": 1.0 / len(sections)})
        a_assets = set(incident[number][:len(incident[number]) // 2]) if number in selected else set(incident[number])
        if spec.case_id == "case57" and number == 4 and number in selected:
            # Required frozen bus-4 partition, expressed using immutable assets.
            a_assets = {branch.asset_id for branch in spec.branches
                        if {branch.from_bus, branch.to_bus} in ({3, 4}, {4, 5})}
        for asset in incident[number]:
            assignments[(number, asset)] = node(number, "A" if asset in a_assets else "B")
        if number in selected:
            couplers.append({"device_id": f"{spec.case_id}:bus:{number}:coupler", "device_kind": "bus_coupler",
                             "base_bus": number, "node_a": node(number, "A"), "node_b": node(number, "B")})
    branches = [{"asset_id": branch.asset_id, "device_id": branch.asset_id + ":status_cb",
                 "device_kind": "branch_status", "row0": branch.row0,
                 "from_node": assignments[(branch.from_bus, branch.asset_id)],
                 "to_node": assignments[(branch.to_bus, branch.asset_id)]}
                for branch in spec.branches]
    statuses = {branch["device_id"]: int(source["branch"][branch["row0"], BR_STATUS]) for branch in branches}
    statuses.update({coupler["device_id"]: 1 for coupler in couplers})
    inventory = {
        "schema": "logical_topology_inventory_v1", "layout_id": layout_id, "case_id": spec.case_id,
        "base_case_hash": spec.base_case_hash, "base_bus_ids": list(spec.row0_to_external_bus),
        "layout_kind": "two_section_couplers" if selected else "branch_status_only",
        "sectioned_buses": sorted(selected), "nodes": nodes, "branches": branches, "couplers": couplers,
        "loads": loads,
        "generators": [{"gen_row0": index, "node_id": node(int(row[GEN_BUS]), "A")}
                       for index, row in enumerate(source["gen"])],
        "shunts": [{"base_bus": int(row[BUS_I]), "node_id": node(int(row[BUS_I]), "A")}
                   for row in source["bus"] if row[GS] != 0 or row[BS] != 0],
        "normal_statuses": statuses,
        "assumptions": {
            "layout_origin": "synthetic_logical_configuration_not_reconstructed_switchgear",
            "branch_status": "whole_asset_at_both_ends_including_charging",
            "one_end_energized_open_line": "not_represented",
            "coupler": "ideal_closed_switch_contraction_no_finite_impedance",
            "control_attachment": "original_bus_type_and_all_generators_on_section_A",
            "load_allocation": "explicit_aggregate_demand_fractions_frozen_before_operating_points",
            "islands": "reported_without_slack_creation_or_load_shedding",
        },
    }
    inventory["layout_hash"] = inventory_hash(inventory)
    validate_inventory(inventory)
    return inventory


def validate_inventory(inventory: Mapping[str, Any]) -> None:
    """Reject stale/tampered layouts and inconsistent equipment assignments."""
    if inventory.get("schema") != "logical_topology_inventory_v1":
        raise ValueError("Unsupported logical inventory schema")
    if not isinstance(inventory.get("layout_id"), str) or not inventory["layout_id"].strip():
        raise ValueError("Missing versioned layout identifier")
    if inventory.get("layout_hash") != inventory_hash(inventory):
        raise ValueError("Logical layout hash does not match its frozen contents")
    spec = resolve_system(inventory["case_id"])
    source = spec.load_case()
    if inventory.get("base_case_hash") != spec.base_case_hash or inventory["base_bus_ids"] != list(spec.row0_to_external_bus):
        raise ValueError("Inventory does not match the pinned base case")
    nodes = {row["node_id"]: row for row in inventory["nodes"]}
    if len(nodes) != len(inventory["nodes"]):
        raise ValueError("Duplicate logical node identifiers")
    base_nodes: dict[int, set[str]] = defaultdict(set)
    for identity, row in nodes.items():
        base, section = row["base_bus"], row["section"]
        if base not in spec.row0_to_external_bus or section not in ("A", "B") or identity != f"{spec.case_id}:bus:{base}:{section}":
            raise ValueError("Invalid logical bus-section identity")
        base_nodes[base].add(section)
    selected = set(inventory["sectioned_buses"])
    if len(selected) != len(inventory["sectioned_buses"]) or set(base_nodes) != set(spec.row0_to_external_bus):
        raise ValueError("Invalid sectioned-bus inventory")
    if any(base_nodes[bus] != ({"A", "B"} if bus in selected else {"A"}) for bus in base_nodes):
        raise ValueError("Each base bus needs A, and only declared sectioned buses may have B")
    branches = inventory["branches"]
    if len(branches) != spec.nl or [row["row0"] for row in branches] != list(range(spec.nl)):
        raise ValueError("Canonical branch rows must appear exactly once in order")
    devices, terminal_counts = set(), defaultdict(int)
    for record, source_branch in zip(branches, spec.branches):
        if (record["asset_id"] != source_branch.asset_id or record["device_id"] != source_branch.asset_id + ":status_cb"
                or record["device_kind"] != "branch_status"):
            raise ValueError("Asset-status device identity does not match its canonical branch")
        for end, base in (("from", source_branch.from_bus), ("to", source_branch.to_bus)):
            attached = record[f"{end}_node"]
            if attached not in nodes or nodes[attached]["base_bus"] != base:
                raise ValueError("Branch terminal attached to an unrelated base bus")
            terminal_counts[attached] += 1
        devices.add(record["device_id"])
    if spec.case_id == "case57" and 4 in selected:
        actual_a = {row["asset_id"] for row in branches
                    if f"{spec.case_id}:bus:4:A" in (row["from_node"], row["to_node"])}
        if actual_a != {"case57:branch:3", "case57:branch:4"}:
            raise ValueError("The frozen IEEE57 bus-4 A partition must contain only branches 3-4 and 4-5")
    coupled_buses = []
    for record in inventory["couplers"]:
        base = record["base_bus"]
        if (record["device_id"] != f"{spec.case_id}:bus:{base}:coupler" or record["device_kind"] != "bus_coupler"
                or record["node_a"] != f"{spec.case_id}:bus:{base}:A" or record["node_b"] != f"{spec.case_id}:bus:{base}:B"
                or any(record[key] not in nodes for key in ("node_a", "node_b"))):
            raise ValueError("A coupler must connect the declared A/B sections of one base bus")
        if min(terminal_counts[record["node_a"]], terminal_counts[record["node_b"]]) < 2:
            raise ValueError("Each synthetic bus section requires at least two branch terminals")
        devices.add(record["device_id"])
        coupled_buses.append(base)
    if set(coupled_buses) != selected or len(coupled_buses) != len(selected):
        raise ValueError("Sectioned buses require exactly one coupler each")
    load_fractions: dict[int, float] = defaultdict(float)
    seen_loads = set()
    for record in inventory["loads"]:
        attached, base, fraction = record["node_id"], record["base_bus"], record["fraction"]
        if attached not in nodes or nodes[attached]["base_bus"] != base or attached in seen_loads:
            raise ValueError("Invalid aggregate load allocation")
        if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Load fractions must be finite and positive")
        seen_loads.add(attached)
        load_fractions[base] += fraction
    if set(load_fractions) != set(base_nodes) or any(abs(value - 1) > 1e-12 for value in load_fractions.values()):
        raise ValueError("Load allocation must preserve each base bus's complete demand")
    generators = inventory["generators"]
    if [row["gen_row0"] for row in generators] != list(range(len(source["gen"]))):
        raise ValueError("Generator rows must appear exactly once in canonical order")
    if any(record["node_id"] != f"{spec.case_id}:bus:{int(source['gen'][index, GEN_BUS])}:A"
           for index, record in enumerate(generators)):
        raise ValueError("Original generator and voltage controls must remain on section A")
    shunt_buses = {int(row[BUS_I]) for row in source["bus"] if row[GS] != 0 or row[BS] != 0}
    if {row["base_bus"] for row in inventory["shunts"]} != shunt_buses or len(inventory["shunts"]) != len(shunt_buses):
        raise ValueError("Canonical shunts must be assigned intact exactly once")
    if any(row["node_id"] not in nodes or nodes[row["node_id"]]["base_bus"] != row["base_bus"] for row in inventory["shunts"]):
        raise ValueError("Shunt attached to an unrelated base bus")
    if len(devices) != len(branches) + len(inventory["couplers"]):
        raise ValueError("Duplicate logical device identifiers")
    validate_statuses(inventory, inventory["normal_statuses"])
    if any(inventory["normal_statuses"][row["device_id"]] != int(source["branch"][row["row0"], BR_STATUS]) for row in branches):
        raise ValueError("Normal asset statuses must reproduce the pinned canonical case")
    if any(inventory["normal_statuses"][row["device_id"]] != 1 for row in inventory["couplers"]):
        raise ValueError("The normal frozen coupler state must be closed")


def validate_statuses(inventory: Mapping[str, Any], statuses: Mapping[str, Any]) -> dict[str, int]:
    """Require every logical device to have an explicit known integer 0/1."""
    if not isinstance(statuses, Mapping):
        raise UnknownStatusError("A complete status mapping is required")
    devices = {row["device_id"] for row in (*inventory["branches"], *inventory["couplers"])}
    missing, extra = devices - set(statuses), set(statuses) - devices
    if extra:
        raise ValueError(f"Unregistered logical statuses: {sorted(extra)}")
    if missing or any(statuses[device] is None for device in devices if device in statuses):
        raise UnknownStatusError(f"Unknown logical statuses prevent compilation: {sorted(missing | {d for d in devices if statuses.get(d) is None})}")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1) for value in statuses.values()):
        raise ValueError("Logical statuses must be explicit integer 0 or 1")
    return {device: int(statuses[device]) for device in sorted(devices)}


def connectivity_report(case: Mapping[str, Any], *, bus_to_nodes: Mapping[int, list[str]] | None = None,
                        node_to_base_bus: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Report retained components and their inputs; never modify the network."""
    bus, gen, branch = (np.asarray(case[key]) for key in ("bus", "gen", "branch"))
    remaining = set(map(int, bus[:, BUS_I]))
    adjacency: dict[int, set[int]] = {number: set() for number in remaining}
    for row in branch:
        if row[BR_STATUS] == 1:
            f, t = int(row[F_BUS]), int(row[T_BUS])
            adjacency[f].add(t)
            adjacency[t].add(f)
    components = []
    while remaining:
        stack, found = [min(remaining)], set()
        while stack:
            number = stack.pop()
            if number in found:
                continue
            found.add(number)
            stack.extend(adjacency[number] - found)
        remaining -= found
        rows = bus[np.isin(bus[:, BUS_I], list(found))]
        generator_rows = [index for index, row in enumerate(gen) if int(row[GEN_BUS]) in found]
        online = [index for index in generator_rows if gen[index, GEN_STATUS] > 0]
        refs = list(map(int, rows[rows[:, BUS_TYPE] == REF, BUS_I]))
        node_ids = sorted(node for number in found for node in (bus_to_nodes or {}).get(number, []))
        components.append({"bus_ids": sorted(found), "node_ids": node_ids,
            "base_buses": sorted({(node_to_base_bus or {})[node] for node in node_ids}) if node_ids else sorted(found),
            "generator_rows0": generator_rows, "online_generator_rows0": online, "reference_bus_ids": refs,
            "has_online_generation": bool(online), "has_reference": bool(refs),
            "has_load": bool(np.any(rows[:, [PD, QD]] != 0)),
            "pd_mw": float(np.sum(rows[:, PD])), "qd_mvar": float(np.sum(rows[:, QD]))})
    return {"connected": len(components) == 1, "component_count": len(components), "components": components,
            "policy": "report_only_no_additional_slack_or_load_shedding"}


def process_topology(case: Mapping[str, Any], inventory: Mapping[str, Any],
                     complete_statuses: Mapping[str, Any]) -> dict[str, Any]:
    """Compile ideal contractions while retaining canonical equipment row order.

    ``case`` must remain on the inventory's BASE-bus basis. Keep this mutable
    operating/parameter case separate from the compiled result. Numerical bus
    IDs for open B sections are temporary; use returned node maps for sensors.
    """
    validate_inventory(inventory)
    statuses = validate_statuses(inventory, complete_statuses)
    spec = resolve_system(inventory["case_id"])
    canonical = spec.load_case()
    source = copy.deepcopy(dict(case))
    for key, count, columns in (("bus", spec.nb, 13), ("gen", len(canonical["gen"]), 10), ("branch", spec.nl, 13)):
        array = np.asarray(source[key], dtype=float)
        if array.ndim != 2 or len(array) != count or array.shape[1] < columns or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}; compiler input must use the canonical equipment/base-bus basis")
        source[key] = array.copy()
    if (source["baseMVA"] != canonical["baseMVA"] or not np.array_equal(source["bus"][:, BUS_I], canonical["bus"][:, BUS_I])
            or not np.array_equal(source["branch"][:, [F_BUS, T_BUS]], canonical["branch"][:, [F_BUS, T_BUS]])
            or not np.array_equal(source["gen"][:, GEN_BUS], canonical["gen"][:, GEN_BUS])):
        raise ValueError("Compiler input has remapped/reordered base equipment; retain a separate canonical-basis case")
    if not np.isin(source["bus"][:, BUS_TYPE], [1, 2, 3, 4]).all():
        raise ValueError("Unsupported base bus control type")
    nodes = {row["node_id"]: row for row in inventory["nodes"]}
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for coupler in inventory["couplers"]:
        if statuses[coupler["device_id"]] == 1:
            parent[find(coupler["node_b"])] = find(coupler["node_a"])
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    primary = [f"{spec.case_id}:bus:{bus}:A" for bus in inventory["base_bus_ids"]]
    extras = [f"{spec.case_id}:bus:{bus}:B" for bus in inventory["base_bus_ids"]
              if f"{spec.case_id}:bus:{bus}:B" in nodes and find(f"{spec.case_id}:bus:{bus}:B") != find(f"{spec.case_id}:bus:{bus}:A")]
    ordered_groups = [groups[find(node)] for node in primary + extras]
    original_row = spec.external_bus_to_row0
    bus_rows, node_to_bus, node_to_row0, bus_to_nodes = [], {}, {}, {}
    for index, group in enumerate(ordered_groups):
        base_bus = nodes[group[0]]["base_bus"]
        row = source["bus"][original_row[base_bus]].copy()
        number = base_bus if index < spec.nb else max(spec.row0_to_external_bus) + index - spec.nb + 1
        row[BUS_I] = number
        row[[PD, QD, GS, BS]] = 0
        if not any(nodes[node]["section"] == "A" for node in group):
            row[BUS_TYPE] = PQ
        bus_rows.append(row)
        bus_to_nodes[number] = list(group)
        for node in group:
            node_to_bus[node], node_to_row0[node] = number, index
    output = copy.deepcopy(source)
    output["bus"] = np.asarray(bus_rows)
    weights: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for load in inventory["loads"]:
        weights[load["base_bus"]][node_to_row0[load["node_id"]]] += float(load["fraction"])
    for base, allocated in weights.items():
        original = source["bus"][original_row[base], [PD, QD]]
        for index, fraction in allocated.items():
            output["bus"][index, [PD, QD]] = original if len(allocated) == 1 else original * fraction
    shunts = {row["base_bus"]: row["node_id"] for row in inventory["shunts"]}
    for base in inventory["base_bus_ids"]:
        values = source["bus"][original_row[base], [GS, BS]]
        if np.any(values != 0):
            if base not in shunts:
                raise ValueError("An unregistered shunt needs an explicit frozen section assignment")
            output["bus"][node_to_row0[shunts[base]], [GS, BS]] = values
    for branch in inventory["branches"]:
        row = branch["row0"]
        output["branch"][row, [F_BUS, T_BUS]] = [node_to_bus[branch["from_node"]], node_to_bus[branch["to_node"]]]
        output["branch"][row, BR_STATUS] = statuses[branch["device_id"]]
    for generator in inventory["generators"]:
        output["gen"][generator["gen_row0"], GEN_BUS] = node_to_bus[generator["node_id"]]
    connectivity = connectivity_report(output, bus_to_nodes=bus_to_nodes,
                                        node_to_base_bus={name: row["base_bus"] for name, row in nodes.items()})
    return {"case": output, "node_to_bus": node_to_bus, "node_to_row0": node_to_row0,
            "bus_to_nodes": bus_to_nodes, "connectivity": connectivity,
            "layout_hash": inventory["layout_hash"], "statuses": statuses}


__all__ = ["UnknownStatusError", "build_inventory", "validate_inventory", "validate_statuses",
           "inventory_hash", "process_topology", "connectivity_report"]
