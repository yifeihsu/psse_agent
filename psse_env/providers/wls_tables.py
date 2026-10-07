"""Per-bus and per-branch tables of a balanced WLS solve, for the agent's WLS summary.

The summary the agent reads lists the few largest residuals and multipliers.
What that list loses is the relations a graph model reads for free: the two
ends of one line, the line's own R and X multipliers, and the injections at
its buses.  These tables put the same solve in that form, for the buses and
branches near the alarm:

* ``bus_table``: one row per bus with the signed normalized residuals of
  its voltage magnitude and its P and Q injections (``null`` at a
  zero-injection bus, whose injection rows are structural and would
  otherwise show which operator model the root came from);
* ``branch_table``: one row per branch with the signed normalized residuals
  of the P and Q flows at both ends and the normalized Lagrange multipliers
  of its series R and X.

The neighbourhood is every bus or branch with a channel at or above
``residual_threshold`` sigma (or a multiplier that large) plus the buses at
the ends of those branches and the branches at those buses, sorted by the
largest magnitude and capped, so the tables stay a few hundred tokens on any
network.  Pure functions of the solve and the loaded case: the provider
can emit them on every ``run_wls`` and the offline prompt builder renders the
same text.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

BUS_ID, BUS_TYPE, PD, QD, GS, BS = 0, 1, 2, 3, 4, 5
F_BUS, T_BUS, TAP, SHIFT, BR_STATUS = 0, 1, 8, 9, 10
GEN_BUS, GEN_STATUS = 0, 7
BUS_TYPE_NAMES = {1: "PQ", 2: "PV", 3: "ref"}
DEFAULT_RESIDUAL_THRESHOLD = 2.0
DEFAULT_MAX_BUSES = 10
DEFAULT_MAX_BRANCHES = 12
DECIMALS = 2


def zero_injection_rows(ppc: Mapping[str, Any]) -> list[int]:
    """Bus rows with no load, no shunt and no in-service generator."""
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc.get("gen") if ppc.get("gen") is not None else np.empty((0, 8)), dtype=float)
    row_of = {int(bus_id): row for row, bus_id in enumerate(bus[:, BUS_ID].astype(int))}
    generating: set[int] = set()
    if gen.size:
        status = gen[:, GEN_STATUS] if gen.shape[1] > GEN_STATUS else np.ones(gen.shape[0])
        generating = {row_of[int(b)] for b, s in zip(gen[:, GEN_BUS], status) if s > 0 and int(b) in row_of}
    return [
        row for row in range(bus.shape[0])
        if bus[row, PD] == 0.0 and bus[row, QD] == 0.0 and bus[row, GS] == 0.0 and bus[row, BS] == 0.0
        and row not in generating
    ]


def _value(x: float) -> float:
    return float(round(float(x), DECIMALS))


def wls_tables(signed_residuals: Sequence[float], lambdas: Sequence[float], ppc: Mapping[str, Any], *,
               residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD, max_buses: int = DEFAULT_MAX_BUSES,
               max_branches: int = DEFAULT_MAX_BRANCHES) -> dict[str, Any]:
    """The bus and branch tables of one solve (see the module docstring).

    ``signed_residuals`` is the full measurement vector's signed normalized
    residual in the channel order Vm, Pinj, Qinj, Pf, Qf, Pt, Qt;
    ``lambdas`` the normalized multipliers interleaved R, X per branch.
    """
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    nb, nl = bus.shape[0], branch.shape[0]
    residual = np.asarray(signed_residuals, dtype=float)
    if residual.size != 3 * nb + 4 * nl:
        raise ValueError(f"expected {3 * nb + 4 * nl} residual channels, got {residual.size}")
    lam = np.asarray(lambdas, dtype=float)
    if lam.size != 2 * nl:
        lam = np.zeros(2 * nl)
    vm, pinj, qinj = residual[:nb], residual[nb:2 * nb], residual[2 * nb:3 * nb]
    flows = residual[3 * nb:].reshape(4, nl)  # Pf, Qf, Pt, Qt
    lam_r, lam_x = lam[0::2], lam[1::2]
    zero = set(zero_injection_rows(ppc))
    bus_ids = bus[:, BUS_ID].astype(int)
    row_of = {int(bus_id): row for row, bus_id in enumerate(bus_ids)}
    source = np.asarray([row_of[int(b)] for b in branch[:, F_BUS]], dtype=int)
    destination = np.asarray([row_of[int(b)] for b in branch[:, T_BUS]], dtype=int)

    bus_magnitude = np.abs(vm).copy()
    injection_magnitude = np.maximum(np.abs(pinj), np.abs(qinj))
    for row in range(nb):
        if row not in zero:
            bus_magnitude[row] = max(bus_magnitude[row], injection_magnitude[row])
    branch_magnitude = np.max(np.abs(np.vstack((flows, lam_r[None, :], lam_x[None, :]))), axis=0)

    hot_buses = {int(row) for row in np.flatnonzero(bus_magnitude >= residual_threshold)}
    hot_branches = {int(row) for row in np.flatnonzero(branch_magnitude >= residual_threshold)}
    buses = set(hot_buses)
    branches = set(hot_branches)
    for row in hot_branches:
        buses.update((int(source[row]), int(destination[row])))
    for row in range(nl):
        if int(source[row]) in hot_buses or int(destination[row]) in hot_buses:
            branches.add(row)

    bus_rows = sorted(buses, key=lambda row: (-bus_magnitude[row], row))
    branch_rows = sorted(branches, key=lambda row: (-branch_magnitude[row], row))
    tap = branch[:, TAP] if branch.shape[1] > TAP else np.zeros(nl)
    shift = branch[:, SHIFT] if branch.shape[1] > SHIFT else np.zeros(nl)
    status = branch[:, BR_STATUS] if branch.shape[1] > BR_STATUS else np.ones(nl)

    bus_table = []
    for row in bus_rows[:max_buses]:
        item: dict[str, Any] = {"bus": int(bus_ids[row]), "type": BUS_TYPE_NAMES.get(int(bus[row, BUS_TYPE]), "PQ"),
                                "vm": _value(vm[row])}
        if row in zero:
            item.update(p=None, q=None)
        else:
            item.update(p=_value(pinj[row]), q=_value(qinj[row]))
        bus_table.append(item)
    branch_table = []
    for row in branch_rows[:max_branches]:
        item = {"line": int(row) + 1, "from": int(bus_ids[source[row]]), "to": int(bus_ids[destination[row]]),
                "pf": _value(flows[0, row]), "qf": _value(flows[1, row]), "pt": _value(flows[2, row]),
                "qt": _value(flows[3, row]), "lr": _value(lam_r[row]), "lx": _value(lam_x[row])}
        if tap[row] != 0.0 or shift[row] != 0.0:
            item["xfmr"] = True
        if status[row] <= 0:
            item["out"] = True
        branch_table.append(item)
    return {
        "bus_table": bus_table,
        "branch_table": branch_table,
        "omitted": {"buses": max(0, len(bus_rows) - len(bus_table)), "branches": max(0, len(branch_rows) - len(branch_table))},
    }


__all__ = ["DEFAULT_MAX_BRANCHES", "DEFAULT_MAX_BUSES", "DEFAULT_RESIDUAL_THRESHOLD", "wls_tables", "zero_injection_rows"]
