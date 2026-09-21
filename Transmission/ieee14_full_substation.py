"""Substation-level telemetry for the full IEEE-14 node/breaker model.

The operator's SCADA snapshot (14-bus order) is what the bus-branch WLS sees. To
identify a wrong breaker status the operator additionally requests the substation
measurements the RTUs already have: the voltage at every connectivity node, the unit
and load meters at the equipment nodes, the terminal flows of the physical branches,
and the P/Q flow through every breaker.

Truth is a PYPOWER power flow of the 65-node network in which every closed breaker is
a tiny series impedance (the values the historical pocket model used) and every open
breaker is out of service, run at the dispatch of an AC-OPF solved on the ideal
contracted topology. Breaker flows are therefore determinate even in yards with
parallel closed paths, and the operator vector is derived from the same physical
solution as the substation telemetry so every shared meter reads the same value.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    T_BUS,
    TAP,
)
from pypower.idx_bus import (
    BASE_KV,
    BS,
    BUS_AREA,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    PQ,
    PV,
    QD,
    REF,
    VA,
    VM,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG

try:
    from Transmission.ieee14_full_measurements import NB, NL, deenergize_unsupplied
    from Transmission.ieee14_full_topology import MODEL_ID, FullTopology, build_full_topology, parse_status
except ImportError:  # pragma: no cover - direct execution inside Transmission/
    from ieee14_full_measurements import NB, NL, deenergize_unsupplied  # type: ignore
    from ieee14_full_topology import MODEL_ID, FullTopology, build_full_topology, parse_status  # type: ignore

# Series impedance of a closed breaker in the physical truth (historical pocket values).
CB_R_PU = 5e-6
CB_X_PU = 5e-5

# Nominal raw-sensor accuracies; summed operator injections propagate variance.
TELEMETRY_SIGMA = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2, "cb": 1e-2}

__all__ = [
    "CB_R_PU",
    "CB_X_PU",
    "TELEMETRY_SIGMA",
    "add_telemetry_noise",
    "injection_metered_nodes",
    "node_breaker_ppc",
    "operator_vector_from_telemetry",
    "operator_model_from_map",
    "operator_vector_for_layout",
    "operator_noise_for_layout",
    "operator_sigma_for_layout",
    "operator_observation_for_layout",
    "STRUCTURAL_EFFECTS",
    "solve_node_breaker",
    "status_map_from_labels",
    "status_labels",
    "substation_telemetry",
]


def status_labels(model: FullTopology, status_map: Mapping[str, Any] | None = None) -> dict[str, str]:
    """``{cb_name: "closed"|"open"}`` for the model's normal state with overrides."""
    return {name: ("closed" if closed else "open") for name, closed in model.states(status_map).items()}


def status_map_from_labels(labels: Mapping[str, Any]) -> dict[str, bool]:
    return {str(name): parse_status(value) for name, value in labels.items()}


def injection_metered_nodes(model: FullTopology, reference: Mapping[str, Any]) -> list[str]:
    """Nodes that host a unit or a load in the reference case: the injection meters."""
    ref_bus = np.asarray(reference["bus"], dtype=float)
    gen_buses = set(np.asarray(reference["gen"], dtype=float)[:, GEN_BUS].astype(int))
    metered: set[str] = set()
    for b in range(1, NB + 1):
        if b in gen_buses:
            metered.add(model.equipment["gen"][b])
        if ref_bus[b - 1, PD] != 0 or ref_bus[b - 1, QD] != 0:
            metered.add(model.equipment["load"][b])
    return sorted(metered, key=lambda n: list(model.nodes).index(n))


def node_breaker_ppc(
    model: FullTopology,
    reference: Mapping[str, Any],
    status_map: Mapping[str, Any] | None = None,
    dispatch: Mapping[str, Any] | None = None,
    node_to_bus: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """PYPOWER case of the 65-node network with breakers as tiny impedances.

    ``dispatch`` is an AC-OPF solution of the contracted case built from the same
    ``reference`` (generator rows in reference order): its PG and the solved voltage
    at each unit's bus become the PV setpoints of the physical power flow. With
    ``node_to_bus`` (the contraction map of that dispatch) every node also starts the
    Newton iteration at its topological bus's solved voltage; near-zero breaker
    impedances make a flat start diverge.
    """
    states = model.states(status_map)
    nodes = list(model.nodes)
    index = {name: i for i, name in enumerate(nodes)}
    ref_bus = np.asarray(reference["bus"], dtype=float)
    ref_gen = np.asarray(reference["gen"], dtype=float)
    ref_branch = np.asarray(reference["branch"], dtype=float)
    base = float(reference["baseMVA"])
    n = len(nodes)

    bus = np.zeros((n, 13))
    for name, meta in model.nodes.items():
        i = index[name]
        src = ref_bus[meta.planning_bus - 1]
        bus[i, BUS_I] = i + 1
        bus[i, BUS_TYPE] = PQ
        bus[i, BUS_AREA] = src[BUS_AREA]
        bus[i, VM] = 1.0
        bus[i, VA] = 0.0
        bus[i, BASE_KV] = src[BASE_KV]
        bus[i, ZONE] = src[ZONE]
        bus[i, VMAX] = src[VMAX]
        bus[i, VMIN] = src[VMIN]
    for b in range(1, NB + 1):
        row = ref_bus[b - 1]
        bus[index[model.equipment["load"][b]], PD] += row[PD]
        bus[index[model.equipment["load"][b]], QD] += row[QD]
        bus[index[model.equipment["shunt"][b]], GS] += row[GS]
        bus[index[model.equipment["shunt"][b]], BS] += row[BS]

    gen = ref_gen.copy()
    dispatch_gen = np.asarray(dispatch["gen"], dtype=float) if dispatch is not None else None
    dispatch_bus = np.asarray(dispatch["bus"], dtype=float) if dispatch is not None else None
    dispatch_vm = (
        {int(r[BUS_I]): float(r[VM]) for r in dispatch_bus} if dispatch_bus is not None else {}
    )
    dispatch_va = (
        {int(r[BUS_I]): float(r[VA]) for r in dispatch_bus} if dispatch_bus is not None else {}
    )
    if dispatch_bus is not None and node_to_bus is not None:
        for name in nodes:
            number = int(node_to_bus[name])
            if number in dispatch_vm:
                bus[index[name], VM] = dispatch_vm[number]
                bus[index[name], VA] = dispatch_va[number]
    for k, row in enumerate(ref_gen):
        b = int(row[GEN_BUS])
        node = index[model.equipment["gen"][b]]
        gen[k, GEN_BUS] = node + 1
        if dispatch_gen is not None:
            gen[k, PG] = dispatch_gen[k, PG]
            gen[k, QG] = dispatch_gen[k, QG]
            gen[k, VG] = dispatch_vm.get(int(dispatch_gen[k, GEN_BUS]), float(dispatch_gen[k, VG]))
        if row[GEN_STATUS] > 0:
            bus[node, BUS_TYPE] = REF if ref_bus[b - 1, BUS_TYPE] == REF else PV
            bus[node, VM] = gen[k, VG]

    branch = np.zeros((NL + len(model.breakers), 13))
    branch[:NL, :] = ref_branch[:, :13]
    for k in range(NL):
        f, t = int(ref_branch[k, F_BUS]), int(ref_branch[k, T_BUS])
        branch[k, F_BUS] = index[model.terminals[(f, t)]] + 1
        branch[k, T_BUS] = index[model.terminals[(t, f)]] + 1
    cb_rows: dict[str, int] = {}
    for j, cb in enumerate(model.breakers):
        k = NL + j
        cb_rows[cb.name] = k
        branch[k, F_BUS] = index[cb.a] + 1
        branch[k, T_BUS] = index[cb.b] + 1
        branch[k, BR_R] = CB_R_PU
        branch[k, BR_X] = CB_X_PU
        branch[k, BR_B] = 0.0
        branch[k, RATE_A] = branch[k, RATE_B] = branch[k, RATE_C] = 0.0
        branch[k, TAP] = 0.0
        branch[k, SHIFT] = 0.0
        branch[k, BR_STATUS] = 1.0 if states[cb.name] else 0.0
        branch[k, ANGMIN] = -360.0
        branch[k, ANGMAX] = 360.0

    ppc = {"version": "2", "baseMVA": base, "bus": bus, "gen": gen, "branch": branch}
    if "gencost" in reference:
        ppc["gencost"] = np.asarray(reference["gencost"], dtype=float).copy()
    info = {"node_index": index, "nodes": nodes, "cb_rows": cb_rows, "states": states}
    return ppc, info


def solve_node_breaker(
    model: FullTopology,
    reference: Mapping[str, Any],
    status_map: Mapping[str, Any] | None = None,
    dispatch: Mapping[str, Any] | None = None,
    node_to_bus: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Power-flow solution of the physical node/breaker network.

    Returns ``(solution or None, info, removed)``; ``removed`` reports de-energized
    sections (loads shed, units dropped) exactly as ``deenergize_unsupplied`` does.
    """
    from pypower.api import ppoption, runpf

    ppc, info = node_breaker_ppc(model, reference, status_map, dispatch, node_to_bus)
    removed = deenergize_unsupplied(ppc)
    solution, converged = runpf(deepcopy(ppc), ppoption(VERBOSE=0, OUT_ALL=0, PF_MAX_IT=30))
    return (solution if converged else None), info, removed


def substation_telemetry(
    solution: Mapping[str, Any],
    model: FullTopology,
    reference: Mapping[str, Any],
    info: Mapping[str, Any],
    removed: Mapping[str, Any],
) -> dict[str, Any]:
    """Noiseless substation measurement set of a solved node/breaker network.

    ``node_vm`` covers every node (0 where de-energized); ``node_pinj``/``node_qinj``
    cover the injection-metered nodes (unit minus load; shunts sit inside the
    estimator's admittance model); ``branch_*`` are the 20 physical branches in
    reference orientation; ``cb_p``/``cb_q`` are the flows from each breaker's
    ``a`` node to its ``b`` node (0 when open or de-energized).
    """
    bus = np.asarray(solution["bus"], dtype=float)
    gen = np.asarray(solution["gen"], dtype=float)
    branch = np.asarray(solution["branch"], dtype=float)
    ref_bus = np.asarray(reference["bus"], dtype=float)
    ref_gen = np.asarray(reference["gen"], dtype=float)
    base = float(solution["baseMVA"])
    index: Mapping[str, int] = info["node_index"]
    nodes: Sequence[str] = info["nodes"]
    dead = {int(b) for b in removed.get("dead_buses", [])}
    row_of = {int(r[BUS_I]): i for i, r in enumerate(bus)}

    node_vm = {}
    for name in nodes:
        number = index[name] + 1
        node_vm[name] = 0.0 if number in dead else float(bus[row_of[number], VM])

    metered = injection_metered_nodes(model, reference)
    node_pinj = {name: 0.0 for name in metered}
    node_qinj = {name: 0.0 for name in metered}
    for k, row in enumerate(gen):
        b = int(ref_gen[k, GEN_BUS])
        node = model.equipment["gen"][b]
        if row[GEN_STATUS] > 0 and node in node_pinj:
            node_pinj[node] += float(row[PG]) / base
            node_qinj[node] += float(row[QG]) / base
    for b in range(1, NB + 1):
        node = model.equipment["load"][b]
        if node in node_pinj and (index[node] + 1) not in dead:
            node_pinj[node] -= float(ref_bus[b - 1, PD]) / base
            node_qinj[node] -= float(ref_bus[b - 1, QD]) / base

    def flow(k: int, column: int) -> float:
        r = branch[k]
        if int(r[F_BUS]) in dead or int(r[T_BUS]) in dead or r[BR_STATUS] <= 0:
            return 0.0
        return float(r[column]) / base

    cb_p = {name: flow(k, PF) for name, k in info["cb_rows"].items()}
    cb_q = {name: flow(k, QF) for name, k in info["cb_rows"].items()}
    return {
        "model_id": MODEL_ID,
        "model_fingerprint": model.fingerprint(),
        "nodes": list(nodes),
        "injection_metered_nodes": list(metered),
        "node_vm": node_vm,
        "node_pinj": node_pinj,
        "node_qinj": node_qinj,
        "branch_pf": [flow(k, PF) for k in range(NL)],
        "branch_qf": [flow(k, QF) for k in range(NL)],
        "branch_pt": [flow(k, PT) for k in range(NL)],
        "branch_qt": [flow(k, QT) for k in range(NL)],
        "cb_p": cb_p,
        "cb_q": cb_q,
        "sigma": dict(TELEMETRY_SIGMA),
        "nominal_sensor_sigma": dict(TELEMETRY_SIGMA),
        "measurement_kind": "noiseless_mean",
        "applied_noise_scale": 0.0,
        "noise_draw_count": 0,
        "dead_nodes": [nodes[b - 1] for b in sorted(dead)],
    }


def add_telemetry_noise(
    telemetry: Mapping[str, Any], rng: np.random.Generator, scale: float = 1.0
) -> dict[str, Any]:
    """Draw sensor noise once, recording the exact applied standard deviations.

    Deterministic controls use ``substation_telemetry`` directly. A noisy draw
    requires a finite positive scale and cannot be applied to an observed
    snapshot a second time. De-energized voltage readings remain explicit zeros.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("noisy telemetry requires a finite positive noise scale")
    if telemetry.get("measurement_kind") == "observed" or telemetry.get("noise_draw_count", 0):
        raise ValueError("telemetry noise must be drawn once from noiseless means")
    nominal = dict(telemetry.get("nominal_sensor_sigma", telemetry["sigma"]))
    if set(nominal) != set(TELEMETRY_SIGMA) or any(
        not np.isfinite(float(value)) or float(value) <= 0 for value in nominal.values()
    ):
        raise ValueError("telemetry sensor sigmas must be finite positive vm/inj/flow/cb values")
    sigma = {key: float(value) * scale for key, value in nominal.items()}
    noisy = deepcopy(dict(telemetry))
    noisy["node_vm"] = {
        name: (value + rng.normal(0.0, sigma["vm"]) if value > 0 else 0.0)
        for name, value in telemetry["node_vm"].items()
    }
    for key in ("node_pinj", "node_qinj"):
        noisy[key] = {name: value + rng.normal(0.0, sigma["inj"]) for name, value in telemetry[key].items()}
    for key in ("branch_pf", "branch_qf", "branch_pt", "branch_qt"):
        noisy[key] = [value + rng.normal(0.0, sigma["flow"]) for value in telemetry[key]]
    for key in ("cb_p", "cb_q"):
        noisy[key] = {name: value + rng.normal(0.0, sigma["cb"]) for name, value in telemetry[key].items()}
    noisy["sigma"] = sigma
    noisy["nominal_sensor_sigma"] = {key: float(value) for key, value in nominal.items()}
    noisy["measurement_kind"] = "observed"
    noisy["applied_noise_scale"] = scale
    noisy["noise_draw_count"] = 1
    return noisy


def operator_vector_from_telemetry(
    telemetry: Mapping[str, Any],
    model: FullTopology,
    vm_nodes: Mapping[int, str],
) -> np.ndarray:
    """The 122-entry operator vector read from the same meters as the telemetry.

    ``Vm[b]`` is the node voltage at the planning bus's designated meter node,
    ``Pinj/Qinj[b]`` sum the injection meters of the bus's equipment nodes, and
    the flows are the branch terminal readings.
    """
    vm = np.array([float(telemetry["node_vm"][vm_nodes[b]]) for b in range(1, NB + 1)])
    pinj = np.zeros(NB)
    qinj = np.zeros(NB)
    for b in range(1, NB + 1):
        seen: set[str] = set()
        for table in ("gen", "load"):
            node = model.equipment[table][b]
            if node in seen or node not in telemetry["node_pinj"]:
                continue
            seen.add(node)
            pinj[b - 1] += float(telemetry["node_pinj"][node])
            qinj[b - 1] += float(telemetry["node_qinj"][node])
    return np.r_[
        vm,
        pinj,
        qinj,
        np.asarray(telemetry["branch_pf"], dtype=float),
        np.asarray(telemetry["branch_qf"], dtype=float),
        np.asarray(telemetry["branch_pt"], dtype=float),
        np.asarray(telemetry["branch_qt"], dtype=float),
    ]


# ------------------------------------------------------------------ operator rendering

STRUCTURAL_EFFECTS = frozenset({"bus_split", "merge"})


def operator_model_from_map(
    reference: Mapping[str, Any],
    model: FullTopology,
    reported_status: Mapping[str, Any] | None = None,
    meter_nodes: Mapping[int, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the operator's bus-branch case for a reported breaker map.

    Every connected node group of the reported map becomes a bus, the way an EMS
    topology processor renders it: a section holding exactly one line terminal and
    no equipment is a dangling terminal, so that line goes out of service and its
    row stays on the main section of the terminal's planning bus; an empty section
    is dropped; a section with equipment but no terminal is an unsupplied island,
    so its units are switched off and its loads dropped. Bus numbers 1..14 go to
    each planning bus's main section in planning order (a merged section takes the
    first number it meets), and further energized sections follow in planning
    order, so the schematic-normal map reproduces the reference numbering exactly.

    ``reference`` is the operator's current case: its branch parameters are kept
    and a line the operator already holds out of service stays out. ``meter_nodes``
    is the operator's voltage-meter node per planning bus; the section holding it
    is that bus's main section.
    """
    states = model.states(reported_status)
    ref_bus = np.asarray(reference["bus"], dtype=float)
    ref_gen = np.asarray(reference["gen"], dtype=float)
    ref_branch = np.asarray(reference["branch"], dtype=float)
    node_order = {name: i for i, name in enumerate(model.nodes)}
    groups = [tuple(g) for g in model.components(states)]
    group_of = {node: gi for gi, group in enumerate(groups) for node in group}

    row_of: dict[tuple[int, int], int] = {}
    for k in range(NL):
        f, t = int(ref_branch[k, F_BUS]), int(ref_branch[k, T_BUS])
        row_of[(f, t)] = k
        row_of[(t, f)] = k
    terminal_rows = {node: row_of[(f, t)] for (f, t), node in model.terminals.items()}
    equipment_nodes = set(injection_metered_nodes(model, reference))
    for b, node in model.equipment["shunt"].items():
        if ref_bus[int(b) - 1, GS] != 0 or ref_bus[int(b) - 1, BS] != 0:
            equipment_nodes.add(node)

    info = []
    for group in groups:
        rows = sorted({terminal_rows[n] for n in group if n in terminal_rows})
        equipment = sorted((n for n in group if n in equipment_nodes), key=node_order.get)
        buses = sorted({model.nodes[n].planning_bus for n in group})
        if rows and (len(rows) > 1 or equipment):
            kind = "energized"
        elif len(rows) == 1:
            kind = "dangling"
        elif equipment:
            kind = "island"
        else:
            kind = "empty"
        info.append(
            {"nodes": list(group), "rows": rows, "equipment": equipment,
             "planning_buses": buses, "kind": kind}
        )

    meters = {int(k): str(v) for k, v in (meter_nodes or {}).items()}
    main: dict[int, int | None] = {}
    for b in range(1, NB + 1):
        candidates = [gi for gi, g in enumerate(info) if b in g["planning_buses"]]
        energized = [gi for gi in candidates if info[gi]["kind"] == "energized"]
        pool = energized or candidates
        if not pool:
            main[b] = None
            continue
        meter = meters.get(b)
        holding = [gi for gi in pool if meter is not None and meter in info[gi]["nodes"]]
        if holding:
            main[b] = holding[0]
        else:
            main[b] = max(
                pool,
                key=lambda gi: (
                    len(info[gi]["rows"]),
                    len(info[gi]["equipment"]),
                    f"{b}B1" in info[gi]["nodes"],
                    -min(node_order[n] for n in info[gi]["nodes"]),
                ),
            )

    number: dict[int, int] = {}
    for b in range(1, NB + 1):
        gi = main[b]
        if gi is not None and gi not in number:
            number[gi] = len(number) + 1
    extras = sorted(
        (gi for gi, g in enumerate(info) if g["kind"] == "energized" and gi not in number),
        key=lambda gi: (
            info[gi]["planning_buses"][0],
            min(node_order[n] for n in info[gi]["nodes"]),
        ),
    )
    for gi in extras:
        number[gi] = len(number) + 1
    n = len(number)
    numbered = sorted(number, key=number.get)

    bus = np.zeros((n, ref_bus.shape[1]))
    for gi in numbered:
        i = number[gi] - 1
        representative = info[gi]["planning_buses"][0]
        bus[i] = ref_bus[representative - 1]
        bus[i, BUS_I] = number[gi]
        bus[i, BUS_TYPE] = PQ
        bus[i, PD] = bus[i, QD] = bus[i, GS] = bus[i, BS] = 0.0
    dropped_loads: list[int] = []
    for b in range(1, NB + 1):
        row = ref_bus[b - 1]
        load_group = group_of[model.equipment["load"][b]]
        if load_group in number:
            bus[number[load_group] - 1, PD] += row[PD]
            bus[number[load_group] - 1, QD] += row[QD]
        elif row[PD] != 0 or row[QD] != 0:
            dropped_loads.append(b)
        shunt_group = group_of[model.equipment["shunt"][b]]
        if shunt_group in number:
            bus[number[shunt_group] - 1, GS] += row[GS]
            bus[number[shunt_group] - 1, BS] += row[BS]

    gen = ref_gen.copy()
    dropped_gens: list[int] = []
    for k, row in enumerate(ref_gen):
        b = int(row[GEN_BUS])
        gi = group_of[model.equipment["gen"][b]]
        if gi in number:
            gen[k, GEN_BUS] = number[gi]
            if row[GEN_STATUS] > 0:
                i = number[gi] - 1
                bus[i, BUS_TYPE] = (
                    REF if ref_bus[b - 1, BUS_TYPE] == REF else max(bus[i, BUS_TYPE], PV)
                )
                bus[i, VM] = row[VG]
        else:
            fallback = main[b] if main[b] is not None and main[b] in number else numbered[0]
            gen[k, GEN_BUS] = number[fallback]
            gen[k, GEN_STATUS] = 0
            dropped_gens.append(k)

    branch = ref_branch.copy()
    dangling_rows: list[int] = []
    for k in range(NL):
        f, t = int(ref_branch[k, F_BUS]), int(ref_branch[k, T_BUS])
        endpoints = []
        out = ref_branch[k, BR_STATUS] <= 0
        for planning_bus, node in ((f, model.terminals[(f, t)]), (t, model.terminals[(t, f)])):
            gi = group_of[node]
            if gi in number:
                endpoints.append(number[gi])
            else:
                out = True
                if info[gi]["kind"] == "dangling":
                    dangling_rows.append(k)
                anchor = main[planning_bus]
                endpoints.append(
                    number[anchor] if anchor is not None and anchor in number else numbered[0]
                )
        if endpoints[0] == endpoints[1]:
            out = True
        branch[k, F_BUS], branch[k, T_BUS] = endpoints
        branch[k, BR_STATUS] = 0.0 if out else 1.0

    case = deepcopy(dict(reference))
    case.update(bus=bus, gen=gen, branch=branch)
    case.pop("order", None)
    case.pop("success", None)
    if "bus_name" in case:
        case["bus_name"] = [" / ".join(info[gi]["nodes"]) for gi in numbered]

    sections = {}
    for gi in numbered:
        g = info[gi]
        meter = None
        for b in g["planning_buses"]:
            if meters.get(b) in g["nodes"]:
                meter = meters[b]
                break
        if meter is None:
            busbars = [x for x in g["nodes"] if model.nodes[x].kind == "busbar"]
            meter = min(busbars or g["nodes"], key=node_order.get)
        sections[str(number[gi])] = {
            "nodes": sorted(g["nodes"], key=node_order.get),
            "planning_buses": g["planning_buses"],
            "meter_node": meter,
            "equipment_nodes": g["equipment"],
        }
    layout = {
        "bus_count": n,
        "sections": sections,
        "node_to_bus": {node: number.get(group_of[node]) for node in model.nodes},
        "dangling_rows": sorted(set(dangling_rows)),
        "dropped_sections": [
            {"kind": g["kind"], "nodes": g["nodes"], "planning_buses": g["planning_buses"]}
            for gi, g in enumerate(info)
            if gi not in number
        ],
        "dropped_loads": dropped_loads,
        "dropped_gen_rows": dropped_gens,
        "main_section_by_bus": {
            str(b): (number.get(main[b]) if main[b] is not None else None)
            for b in range(1, NB + 1)
        },
    }
    return case, layout


def operator_vector_for_layout(
    telemetry: Mapping[str, Any], layout: Mapping[str, Any]
) -> np.ndarray:
    """The operator vector of a rendered layout, read from the substation meters.

    Each bus's voltage is its section's meter node, its injection the sum of the
    injection meters in the section (zero for a section without equipment), and
    the flows are the branch terminal readings.
    """
    n = int(layout["bus_count"])
    sections = layout["sections"]
    vm = np.zeros(n)
    pinj = np.zeros(n)
    qinj = np.zeros(n)
    for i in range(n):
        section = sections[str(i + 1)]
        vm[i] = float(telemetry["node_vm"][section["meter_node"]])
        for node in section["equipment_nodes"]:
            if node in telemetry["node_pinj"]:
                pinj[i] += float(telemetry["node_pinj"][node])
                qinj[i] += float(telemetry["node_qinj"][node])
    return np.r_[
        vm,
        pinj,
        qinj,
        np.asarray(telemetry["branch_pf"], dtype=float),
        np.asarray(telemetry["branch_qf"], dtype=float),
        np.asarray(telemetry["branch_pt"], dtype=float),
        np.asarray(telemetry["branch_qt"], dtype=float),
    ]


def operator_noise_for_layout(
    telemetry: Mapping[str, Any], layout: Mapping[str, Any]
) -> dict[str, Any]:
    """Propagate fixed physical-meter covariance through one operator layout.

    Each voltage row selects one named physical sensor; it is never averaged.
    An injection is a sum of independent equipment-node meters. Shared source
    meters, if a layout repeats them, produce the corresponding off-diagonal
    covariance. Rows without injection meters are exact structural constraints,
    not fictitious noisy sensors. This function does not alter observations or
    draw new noise and is equally applicable to a stored snapshot's layout.

    For explicit ``noiseless_mean`` telemetry, measurement covariance describes
    the target sensor model; ``applied_noise_covariance`` is zero. Observed
    telemetry uses the actual scaled sigmas recorded by ``add_telemetry_noise``.
    """
    n = int(layout["bus_count"])
    sections = layout["sections"]
    sigmas = {key: float(telemetry["sigma"][key]) for key in TELEMETRY_SIGMA}
    if any(not np.isfinite(value) or value <= 0 for value in sigmas.values()):
        raise ValueError("physical sensor sigmas must be finite and positive")
    if set(telemetry["node_pinj"]) != set(telemetry["node_qinj"]):
        raise ValueError("P/Q injection meter identities must match")
    kind = str(telemetry.get("measurement_kind", "observed"))
    if kind not in {"observed", "noiseless_mean"}:
        raise ValueError("measurement_kind must be observed or noiseless_mean")
    dead_nodes = {str(node) for node in telemetry.get("dead_nodes", [])}
    sources: list[list[str]] = []
    measurement_ids: list[str] = []
    source_variances: dict[str, float] = {}
    structural: list[int] = []
    deterministic: list[int] = []

    def append_row(channel: str, source_ids: list[str], variance: float, *, zero_identity: str = "") -> None:
        index = len(sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("an operator row cannot count one physical sensor twice")
        sources.append(source_ids)
        measurement_ids.append(
            source_ids[0] if len(source_ids) == 1 else
            f"{channel}:sum:" + "+".join(sorted(source_ids)) if source_ids else
            f"{channel}:structural_zero:{zero_identity}"
        )
        for source_id in source_ids:
            if source_id in source_variances and source_variances[source_id] != variance:
                raise ValueError("one physical sensor has conflicting declared variances")
            source_variances[source_id] = variance
        if not source_ids:
            structural.append(index)
        elif variance == 0:
            deterministic.append(index)

    for i in range(n):
        section = sections[str(i + 1)]
        node = str(section["meter_node"])
        if node not in telemetry["node_vm"]:
            raise ValueError(f"voltage meter {node} is absent from the physical snapshot")
        variance = 0.0 if node in dead_nodes else sigmas["vm"] ** 2
        append_row("Vm", [f"Vm:node:{node}"], variance)
    for channel, key in (("Pinj", "node_pinj"), ("Qinj", "node_qinj")):
        for i in range(n):
            section = sections[str(i + 1)]
            nodes = [str(node) for node in section["equipment_nodes"] if node in telemetry[key]]
            append_row(channel, [f"{channel}:node:{node}" for node in nodes], sigmas["inj"] ** 2,
                       zero_identity="+".join(sorted(str(node) for node in section.get("nodes", [section["meter_node"]]))))
    for channel, key in (("Pf", "branch_pf"), ("Qf", "branch_qf"), ("Pt", "branch_pt"), ("Qt", "branch_qt")):
        if len(telemetry[key]) != NL:
            raise ValueError(f"{key} must retain the {NL} original physical branch identities")
        for row0 in range(NL):
            append_row(channel, [f"{channel}:branch_row0:{row0}"], sigmas["flow"] ** 2)
    raw_ids = list(source_variances)
    raw_index = {source_id: i for i, source_id in enumerate(raw_ids)}
    aggregation = np.zeros((len(sources), len(raw_ids)))
    for i, source_ids in enumerate(sources):
        for source_id in source_ids:
            aggregation[i, raw_index[source_id]] = 1.0
    raw_variance = np.asarray([source_variances[source_id] for source_id in raw_ids])
    covariance = (aggregation * raw_variance) @ aggregation.T
    diagonal = np.diag(covariance)
    return {
        "contract": "physical_topology_meter_covariance_v1",
        "measurement_kind": kind,
        "measurement_ids": measurement_ids,
        "measurement_sources": sources,
        "measurement_sigma": np.sqrt(diagonal).tolist(),
        "measurement_covariance": covariance.tolist(),
        "applied_noise_covariance": (covariance if kind == "observed" else np.zeros_like(covariance)).tolist(),
        "covariance_interpretation": "applied_observation_covariance" if kind == "observed" else "target_sensor_covariance_for_noiseless_mean",
        "structural_zero_indices": structural,
        "deterministic_sensor_indices": deterministic,
        "stochastic_measurement_indices": np.flatnonzero(diagonal > 0).tolist(),
        "raw_sensor_ids": raw_ids,
        "raw_sensor_variances": raw_variance.tolist(),
        "aggregation_matrix": aggregation.tolist(),
        "covariance_propagation": "A @ diag(raw_sensor_variances) @ A.T",
        "candidate_comparison_requires_retained_raw_observations": True,
        "applied_noise_scale": telemetry.get("applied_noise_scale"),
    }


def operator_sigma_for_layout(telemetry: Mapping[str, Any], layout: Mapping[str, Any]) -> np.ndarray:
    """Per-row standard deviations, with exact structural rows left at zero."""
    return np.asarray(operator_noise_for_layout(telemetry, layout)["measurement_sigma"], dtype=float)


def operator_observation_for_layout(telemetry: Mapping[str, Any], layout: Mapping[str, Any]) -> dict[str, Any]:
    """Store one unchanged observation projection together with its covariance."""
    return {"measurements": operator_vector_for_layout(telemetry, layout).tolist(),
            **operator_noise_for_layout(telemetry, layout)}
