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

# Nominal sensor accuracies (pu, rad) shared by the operator vector and the telemetry.
TELEMETRY_SIGMA = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2, "cb": 1e-2}

__all__ = [
    "CB_R_PU",
    "CB_X_PU",
    "TELEMETRY_SIGMA",
    "add_telemetry_noise",
    "injection_metered_nodes",
    "node_breaker_ppc",
    "operator_vector_from_telemetry",
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
        "dead_nodes": [nodes[b - 1] for b in sorted(dead)],
    }


def add_telemetry_noise(
    telemetry: Mapping[str, Any], rng: np.random.Generator, scale: float = 1.0
) -> dict[str, Any]:
    """Independent Gaussian sensor noise on every channel (sigmas scaled by ``scale``)."""
    sigma = {key: float(value) * float(scale) for key, value in telemetry["sigma"].items()}
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
