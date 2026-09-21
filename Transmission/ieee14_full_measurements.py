"""Fixed-identity SCADA synthesis and single-switch classification for the full IEEE-14
node/breaker model (``ieee14_full_schematic_v1``).

The operator's telemetry keeps a fixed physical identity no matter how the yard is
switched. This module maps a solved contracted case (any bus count) back into the
14-bus operator order ``[Vm(14) Pinj(14) Qinj(14) Pf(20) Qf(20) Pt(20) Qt(20)]``:

* ``Vm[b]``: voltage at one meter node per planning bus. ``main_section_nodes`` puts
  that meter on the energized section holding the most line terminals, so a dead
  injection bay never masquerades as a dead bus.
* ``Pinj/Qinj[b]``: unit output minus load for the equipment attached to planning bus
  ``b``. Bus shunts live inside Ybus in the pipeline's measurement model and are not
  part of this channel. A shed load or dropped unit reads zero.
* Flows: the 20 original branches in reference orientation. A branch inside a
  de-energized section reads zero.

Sections without a slack are de-energized by ``deenergize_unsupplied`` before the
solve; the shed load and dropped units are reported, never hidden.

``single_flip_catalogue`` classifies every one of the 73 single-switch status errors
by what the flipped partition does to line terminals and equipment. Only
``dangling_line_terminal`` flips are representable by the pipeline's branch-status
correction; ``dangling_terminal_errors`` returns those with the equivalent branch row.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from pypower.idx_brch import BR_STATUS, F_BUS, PF, PT, QF, QT, T_BUS
from pypower.idx_bus import BS, BUS_I, BUS_TYPE, GS, NONE, PD, QD, REF, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG

try:  # package import (psse_env, scripts) or direct module import (Transmission/)
    from Transmission.ieee14_full_topology import (
        MODEL_ID,
        FullTopology,
        build_full_topology,
        topology_to_matpower,
    )
except ImportError:  # pragma: no cover - direct execution inside Transmission/
    from ieee14_full_topology import (  # type: ignore
        MODEL_ID,
        FullTopology,
        build_full_topology,
        topology_to_matpower,
    )

NB, NL = 14, 20
NZ = 3 * NB + 4 * NL

__all__ = [
    "MODEL_ID",
    "NB",
    "NL",
    "NZ",
    "classify_flip",
    "dangling_terminal_errors",
    "deenergize_unsupplied",
    "flipped_case",
    "main_section_nodes",
    "operator_measurements",
    "fixed_operator_layout",
    "single_flip_catalogue",
]


# ------------------------------------------------------------------------------ physics


def deenergize_unsupplied(case: Mapping[str, Any]) -> dict[str, Any]:
    """Mark every branch-connected component without a slack bus as de-energized.

    Mutates ``case`` in place: those buses become type ``NONE`` (PYPOWER's ext2int
    then drops them, the branches inside them and their generators), their loads are
    shed and their units dropped. Returns what was removed.
    """
    bus, gen, branch = case["bus"], case["gen"], case["branch"]
    idx = {int(r[BUS_I]): i for i, r in enumerate(bus)}
    parent = list(range(len(bus)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in branch:
        if r[BR_STATUS] > 0:
            a, b = find(idx[int(r[F_BUS])]), find(idx[int(r[T_BUS])])
            if a != b:
                parent[b] = a
    comps: dict[int, list[int]] = {}
    for i in range(len(bus)):
        comps.setdefault(find(i), []).append(i)
    dead: list[int] = []
    shed_p = shed_q = 0.0
    dropped: list[int] = []
    for members in comps.values():
        if any(bus[i, BUS_TYPE] == REF for i in members):
            continue
        member_set = set(members)
        for i in members:
            if bus[i, BUS_TYPE] != NONE:
                shed_p += float(bus[i, PD])
                shed_q += float(bus[i, QD])
            bus[i, BUS_TYPE] = NONE
            dead.append(int(bus[i, BUS_I]))
        for k, g in enumerate(gen):
            if idx[int(g[GEN_BUS])] in member_set and g[GEN_STATUS] > 0:
                g[GEN_STATUS] = 0
                dropped.append(k)
    return {
        "dead_buses": sorted(dead),
        "shed_p_mw": shed_p,
        "shed_q_mvar": shed_q,
        "dropped_gen_rows": dropped,
    }


def flipped_case(
    reference: Mapping[str, Any],
    status_map: Mapping[str, Any],
    *,
    model: FullTopology | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Contract the node/breaker model under ``status_map`` and de-energize islands.

    Returns ``(case, info, removed)`` where ``case`` is ready for a PYPOWER solve,
    ``info`` is the contraction metadata (``node_to_bus``, ``components``...) and
    ``removed`` is the ``deenergize_unsupplied`` report.
    """
    model = model or build_full_topology()
    case, info = topology_to_matpower(reference, dict(status_map), model=model)
    removed = deenergize_unsupplied(case)
    return case, info, removed


def main_section_nodes(
    model: FullTopology, node_to_bus: Mapping[str, int], dead: Sequence[int]
) -> dict[int, str]:
    """Voltage-meter node per planning bus on its energized section with the most
    line terminals (ties: most equipment, then the ``B1`` busbar, then the anchor)."""
    dead_set = set(int(b) for b in dead)
    terminal_nodes = set(model.terminals.values())
    equipment_nodes = {node for table in model.equipment.values() for node in table.values()}
    out: dict[int, str] = {}
    for b in range(1, NB + 1):
        groups: dict[int, list[str]] = {}
        for node, meta in model.nodes.items():
            if meta.planning_bus == b:
                groups.setdefault(int(node_to_bus[node]), []).append(node)

        def score(tb: int) -> tuple:
            nodes = groups[tb]
            return (
                tb not in dead_set,
                sum(n in terminal_nodes for n in nodes),
                sum(n in equipment_nodes for n in nodes),
                f"{b}B1" in nodes,
                model.anchors[b] in nodes,
            )

        nodes = groups[max(groups, key=score)]
        if f"{b}B1" in nodes:
            out[b] = f"{b}B1"
        elif model.anchors[b] in nodes:
            out[b] = model.anchors[b]
        else:
            out[b] = sorted(nodes)[0]
    return out


def operator_measurements(
    solved: Mapping[str, Any],
    reference: Mapping[str, Any],
    model: FullTopology,
    node_to_bus: Mapping[str, int],
    dead: Sequence[int],
    vm_nodes: Mapping[int, str],
) -> np.ndarray:
    """Noiseless fixed-identity SCADA mean in 14-bus operator order.

    ``solved`` is the PYPOWER result (``runpf``/``runopf``) of a case produced by
    ``flipped_case``; ``reference`` is the 14-bus case that was contracted, whose gen
    row order and per-bus loads identify the equipment. ``vm_nodes`` names the
    voltage-meter node of each planning bus.
    This deterministic helper does not draw sensor noise. For observed telemetry
    and its propagated covariance use ``operator_observation_for_layout`` from
    ``ieee14_full_substation`` with the fixed layout below.
    """
    bus = np.asarray(solved["bus"], dtype=float)
    gen = np.asarray(solved["gen"], dtype=float)
    branch = np.asarray(solved["branch"], dtype=float)
    ref_bus = np.asarray(reference["bus"], dtype=float)
    ref_gen = np.asarray(reference["gen"], dtype=float)
    if gen.shape[0] != ref_gen.shape[0]:
        raise ValueError("solved and reference generator tables must have the same rows")
    idx = {int(r[BUS_I]): i for i, r in enumerate(bus)}
    base = float(solved["baseMVA"])
    dead_set = set(int(b) for b in dead)
    v = bus[:, VM].copy()
    for b in dead_set:
        v[idx[b]] = 0.0
    vm = np.array([v[idx[int(node_to_bus[vm_nodes[b]])]] for b in range(1, NB + 1)])
    pinj = np.zeros(NB)
    qinj = np.zeros(NB)
    for k, g in enumerate(gen):
        b = int(ref_gen[k, GEN_BUS])
        if g[GEN_STATUS] > 0:
            pinj[b - 1] += float(g[PG]) / base
            qinj[b - 1] += float(g[QG]) / base
    for b in range(1, NB + 1):
        row = ref_bus[b - 1]
        if int(node_to_bus[model.equipment["load"][b]]) not in dead_set:
            pinj[b - 1] -= float(row[PD]) / base
            qinj[b - 1] -= float(row[QD]) / base
    pf = np.zeros(NL)
    qf = np.zeros(NL)
    pt = np.zeros(NL)
    qt = np.zeros(NL)
    for k, r in enumerate(branch):
        if int(r[F_BUS]) in dead_set or int(r[T_BUS]) in dead_set:
            continue
        pf[k] = float(r[PF]) / base
        qf[k] = float(r[QF]) / base
        pt[k] = float(r[PT]) / base
        qt[k] = float(r[QT]) / base
    return np.r_[vm, pinj, qinj, pf, qf, pt, qt]


def fixed_operator_layout(
    reference: Mapping[str, Any], model: FullTopology, vm_nodes: Mapping[int, str]
) -> dict[str, Any]:
    """Bind the original 122 channels to physical meters without resampling.

    The layout selects one voltage meter and the distinct generator/load meter
    nodes for each planning bus. It is independent of any candidate breaker
    map; retain it with the raw physical observation for candidate comparisons.
    """
    gen_buses = set(np.asarray(reference["gen"])[:, GEN_BUS].astype(int).tolist())
    ref_bus = np.asarray(reference["bus"], dtype=float)
    sections = {}
    for b in range(1, NB + 1):
        equipment = set()
        if b in gen_buses:
            equipment.add(model.equipment["gen"][b])
        if ref_bus[b - 1, PD] != 0 or ref_bus[b - 1, QD] != 0:
            equipment.add(model.equipment["load"][b])
        sections[str(b)] = {
            "meter_node": str(vm_nodes[b]), "equipment_nodes": sorted(equipment),
            "nodes": sorted(node for node, meta in model.nodes.items() if meta.planning_bus == b),
            "planning_buses": [b],
        }
    return {"bus_count": NB, "sections": sections,
            "contract": "fixed_planning_bus_physical_meter_layout_v1"}


# ------------------------------------------------------------------------ classification


def _branch_row_lookup(reference: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for k, r in enumerate(np.asarray(reference["branch"], dtype=float)):
        f, t = int(r[F_BUS]), int(r[T_BUS])
        lookup[(f, t)] = k
        lookup[(t, f)] = k
    return lookup


def _branch_name(reference: Mapping[str, Any], k: int) -> str:
    r = np.asarray(reference["branch"], dtype=float)[k]
    return f"br{k}:{int(r[F_BUS])}-{int(r[T_BUS])}"


def _group_contents(group: Sequence[str], model: FullTopology, reference: Mapping[str, Any]) -> dict[str, Any]:
    nodes = set(group)
    rows = _branch_row_lookup(reference)
    terminals = sorted({rows[(f, t)] for (f, t), node in model.terminals.items() if node in nodes})
    ref_bus = np.asarray(reference["bus"], dtype=float)
    gen_buses = set(np.asarray(reference["gen"], dtype=float)[:, GEN_BUS].astype(int))
    equipment: list[str] = []
    for b, node in model.equipment["gen"].items():
        if node in nodes and b in gen_buses:
            equipment.append(f"gen@{b}")
    for b, node in model.equipment["load"].items():
        if node in nodes and (ref_bus[b - 1, PD] != 0 or ref_bus[b - 1, QD] != 0):
            equipment.append(f"load@{b}")
    for b, node in model.equipment["shunt"].items():
        if node in nodes and (ref_bus[b - 1, GS] != 0 or ref_bus[b - 1, BS] != 0):
            equipment.append(f"shunt@{b}")
    return {
        "nodes": sorted(nodes),
        "terminal_rows": terminals,
        "terminals": [_branch_name(reference, k) for k in terminals],
        "equipment": sorted(equipment),
    }


def classify_flip(
    model: FullTopology,
    reference: Mapping[str, Any],
    flipped_groups: Sequence[tuple[str, ...]],
    normal_groups: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    """Describe what a switch flip does to the partition.

    Categories: ``equivalent`` (partition unchanged), ``merge_<buses>``,
    ``dangling_line_terminal`` (one line end isolated, nothing else),
    ``unsupplied_island`` (equipment cut off with no line), ``bus_split`` (a section
    keeps two or more terminals, or a terminal plus equipment) and ``empty_busbar``.
    """
    flipped_set, normal_set = set(flipped_groups), set(normal_groups)
    if flipped_set == normal_set:
        return {"effect": "equivalent", "category": "equivalent", "affected_planning_buses": []}
    if len(flipped_groups) < len(normal_groups):
        merged = [g for g in flipped_groups if g not in normal_set]
        buses = sorted({model.nodes[n].planning_bus for g in merged for n in g})
        return {
            "effect": "merge",
            "category": "merge_" + "_".join(map(str, buses)),
            "affected_planning_buses": buses,
        }
    new_groups = [g for g in flipped_groups if g not in normal_set]
    contents = [_group_contents(g, model, reference) for g in new_groups]
    order = sorted(
        range(len(contents)),
        key=lambda i: len(contents[i]["terminal_rows"]) + len(contents[i]["equipment"]),
    )
    minor, major = contents[order[0]], contents[order[-1]]
    n_t, n_e = len(minor["terminal_rows"]), len(minor["equipment"])
    if n_t == 0 and n_e == 0:
        category = "empty_busbar"
    elif n_t == 1 and n_e == 0:
        category = "dangling_line_terminal"
    elif n_t == 0:
        category = "unsupplied_island"
    else:
        category = "bus_split"
    return {
        "effect": "split",
        "category": category,
        "affected_planning_buses": sorted({model.nodes[n].planning_bus for g in new_groups for n in g}),
        "minor_section": minor,
        "major_section": major,
    }


def single_flip_catalogue(
    model: FullTopology | None = None, reference: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Classify each of the 73 single-switch status errors against the normal state.

    ``reference`` supplies branch rows and equipment presence (default PYPOWER
    ``case14``); load scaling does not change the classification.
    """
    model = model or build_full_topology()
    if reference is None:
        from pypower.api import case14

        reference = case14()
    normal_groups = model.components()
    catalogue: list[dict[str, Any]] = []
    for cb in model.breakers:
        true_closed = not cb.closed
        flipped_groups = model.components({cb.name: true_closed})
        entry = {
            "cb_name": cb.name,
            "yard": cb.yard,
            "reported_closed": bool(cb.closed),
            "true_closed": bool(true_closed),
            "topological_buses_true": len(flipped_groups),
            **classify_flip(model, reference, flipped_groups, normal_groups),
        }
        entry["partition_changed"] = entry["effect"] != "equivalent"
        if entry["category"] == "dangling_line_terminal":
            entry["equivalent_branch_row0"] = int(entry["minor_section"]["terminal_rows"][0])
        catalogue.append(entry)
    return catalogue


def dangling_terminal_errors(
    model: FullTopology | None = None, reference: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Single-switch errors that isolate exactly one line terminal.

    Each entry carries the switch, its reported and true states, and the
    ``equivalent_branch_row0`` whose out-of-service status is the bus-branch fix.
    """
    return [
        entry
        for entry in single_flip_catalogue(model, reference)
        if entry["category"] == "dangling_line_terminal"
    ]
