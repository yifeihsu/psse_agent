"""Balanced physics screen for high-impedance faults (HIF), SCADA and WLS only.

After a balanced WLS alarm the operator's own balanced model is refitted under
competing single-cause explanations, and the explanation with the largest
complexity-penalized chi-square reduction wins:

* ``meter``      one of the top normalized-residual channels left out (its
                 bias is free);
* ``parameter``  one branch's series R, X, or R and X jointly re-estimated
                 inside a plausible box (0 <= R <= max(20 R0, X0 / 2),
                 X0 / 20 <= X <= 20 X0); the joint variant pays for two
                 continuous parameters (2026-09-30: a corpus parameter root
                 has both wrong, and one re-estimated parameter left 47 of
                 250 roots alarmed);
* ``topology``   one branch switched out (never one whose loss islands a bus);
* ``hif``        one eligible line split at ``alpha`` into two pi sections,
                 with a new unmeasured zero-injection bus that carries a
                 shunt conductance ``G >= 0`` estimated jointly with the
                 state; optionally (off by default) also with the two end buses'
                 phase-A voltage channels set aside (a single-phase fault
                 shifts the phase-A magnitude by its sequence effect, which
                 the balanced shunt cannot carry), paying one continuous
                 parameter per channel.

    score = J0 - J - (continuous_penalty * n_continuous
                      + discrete_penalty * ln(n_candidates))

A win is applied and the test runs again on what remains: a meter win sets
the channel aside, a parameter win installs the estimate, a topology win
opens the branch; up to ``max_rounds`` comparisons, stopping as soon as the
remaining model clears the alarm rule or an HIF wins (a suspicion opens
phase-resolved measurements; the balanced rounds end there).  The HIF class
is compared only while at most one meter has been set aside, which keeps the
original two-round false-flag rate.  The outcome names every accepted
hypothesis in order (``meter>hif``, ``parameter>meter``).

Three outputs feed the suspicion-gated acquisition rule
(docs/hypothesis_ranking_plan_20260930.md, step 2): ``suspected`` (an HIF
won), ``voltage_meter_channels`` (phase-A voltage channels the rounds set
aside: on balanced SCADA a one-bus unbalance is indistinguishable from a bad
voltage meter, so a leading voltage-meter hypothesis is confirmed on phasors
before the meter is edited), and ``unexplained`` (the rounds ran out or no
hypothesis converged while the alarm stood, and no HIF won).  Each is a
reason to request phase-resolved measurements, not a diagnosis.

Verified on the 2026-09-26 feasibility dataset (IEEE 14, 1,324 alarmed roots,
penalties tuned on its design half): held-out HIF roots 64 of 65 flagged with
the faulted line correct in every flag, 2 of 1,062 non-HIF roots flagged with
the conductance-only shunt, about 0.3 s per round on one core.  Known limits:
faults off the candidate lines or at a bus read as meter errors; two bad flow
meters reading high at the two ends of one line look like an HIF (16-26% in a
synthetic test); the shunt model matches the simulator's linear resistive
HIF, so arcing faults are untested.

The measurement model is the repository's balanced WLS model
(``tools/lagrangian_port``): measurement order ``[Vm, P, Q, Pf, Qf, Pt, Qt]``
over the operator's buses and branches, the declared sigmas, and the exact
structural-zero injection rows as KKT equality constraints.  The dense
admittance and Jacobian below are the same formulas as ``make_ybus`` and
``make_jaco`` (checked against them in the tests).
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA, BASE_KV = 0, 1, 2, 3, 4, 5, 7, 8, 9
F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS = 0, 1, 2, 3, 4, 8, 9, 10
REF = 3

HIF_SCREEN_METHOD = "balanced_sequential_refits_v2"
#: Policy-visible signature of a current suspicion; the line is 1-based.
from psse_env.actions import HIF_SCREEN_SIGNATURE  # the signature lives with the routing helpers
#: Policy-visible signatures of the other two phasor suspicions (2026-09-30):
#: a phase-A voltage channel the rounds set aside, and an alarm no balanced
#: hypothesis sequence explains.  Neither carries a waveform-family marker,
#: so neither blocks the balanced routes by itself.
VOLTAGE_METER_SCREEN_SIGNATURE = "wls_voltage_meter_suspected"
UNEXPLAINED_SCREEN_SIGNATURE = "wls_unexplained_balanced_discrepancy"
HIF_SCREEN_CLASSES = ("meter", "parameter", "topology", "hif")
#: External endpoints of the 16 same-voltage IEEE 14 lines an HIF can sit on
#: (``three_phase_nlm.hif_units.PHYSICAL_ELIGIBLE_HIF_BRANCHES``): every line
#: except the 13.8/18 kV branch 7-8; the three transformers are not lines.
IEEE14_HIF_LINE_ENDPOINTS = frozenset({
    (1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5), (6, 11), (6, 12),
    (6, 13), (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14),
})


@dataclass(frozen=True)
class HifScreenConfig:
    """Decision constants of the verified screen (tuned 2026-09-26)."""

    continuous_penalty: float = 13.5
    discrete_penalty: float = 0.25
    meter_candidates: int = 6
    alphas: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))
    chi2_alpha: float = 0.01
    normalized_residual_threshold: float = 4.0
    #: Comparisons per screen (2026-09-30: three, so a physical win can be
    #: followed by the meter it hid, or three meters set aside in turn).
    max_rounds: int = 3
    #: Joint R and X refit per branch beside the single-parameter refits.
    parameter_rx_variant: bool = True
    #: Split-line shunt with the end buses' phase-A voltage channels set aside,
    #: tried on this many top-ranked candidate lines.  Off by default: on the
    #: 2026-09-30 IEEE 14 dataset it added no HIF detection (130 of 131 with
    #: or without it) and flagged 61 non-HIF roots (31 unbalance, 16 harmonic,
    #: 7 multi-meter, 7 topology); the voltage-meter suspicion covers the
    #: phase-A offset case instead.  Kept for the IEEE 57 study.
    hif_vm_offset_variant: bool = False
    hif_vm_offset_lines: int = 3


DEFAULT_HIF_SCREEN_CONFIG = HifScreenConfig()


# --------------------------------------------------------------- measurement model


def _ybus_dense(base_mva: float, bus: np.ndarray, branch: np.ndarray):
    """Dense copy of ``lagrangian_port.make_ybus`` (same formulas)."""
    nb = bus.shape[0]
    nl = branch.shape[0]
    stat = branch[:, BR_STATUS]
    ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])
    bc = stat * branch[:, BR_B]
    tap = np.ones(nl, dtype=complex)
    tapped = np.flatnonzero(branch[:, TAP] != 0.0)
    tap[tapped] = branch[tapped, TAP]
    tap = tap * np.exp(1j * np.pi / 180.0 * branch[:, SHIFT])
    ytt = ys + 1j * bc / 2.0
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    ysh = (bus[:, GS] + 1j * bus[:, BS]) / base_mva
    f = branch[:, F_BUS].astype(int)
    t = branch[:, T_BUS].astype(int)
    rows = np.arange(nl)
    yf = np.zeros((nl, nb), dtype=complex)
    yt = np.zeros((nl, nb), dtype=complex)
    yf[rows, f] += yff
    yf[rows, t] += yft
    yt[rows, f] += ytf
    yt[rows, t] += ytt
    ybus = np.zeros((nb, nb), dtype=complex)
    np.add.at(ybus, f, yf)
    np.add.at(ybus, t, yt)
    ybus[np.arange(nb), np.arange(nb)] += ysh
    return ybus, yf, yt, f, t


def _h_full(ybus, yf, yt, f, t, va, vm) -> np.ndarray:
    v = vm * np.exp(1j * va)
    sinj = v * np.conj(ybus @ v)
    sf = v[f] * np.conj(yf @ v)
    st = v[t] * np.conj(yt @ v)
    return np.r_[vm, sinj.real, sinj.imag, sf.real, sf.imag, st.real, st.imag]


def _h_and_jac(ybus, yf, yt, f, t, va, vm) -> tuple[np.ndarray, np.ndarray]:
    """h(x) and dh/d[Va, Vm], identical to ``lagrangian_port.make_jaco``."""
    nb = vm.size
    nl = f.size
    v = vm * np.exp(1j * va)
    vnorm = np.exp(1j * va)
    ibus = ybus @ v
    i_from = yf @ v
    i_to = yt @ v
    sinj = v * np.conj(ibus)
    sf = v[f] * np.conj(i_from)
    st = v[t] * np.conj(i_to)
    cy = np.conj(ybus)
    ds_dvm = v[:, None] * cy * np.conj(vnorm)[None, :]
    ds_dvm[np.arange(nb), np.arange(nb)] += np.conj(ibus) * vnorm
    ds_dva = -1j * (v[:, None] * cy * np.conj(v)[None, :])
    ds_dva[np.arange(nb), np.arange(nb)] += 1j * v * np.conj(ibus)
    rows = np.arange(nl)
    cyf = np.conj(yf)
    cyt = np.conj(yt)
    dsf_dva = -1j * (v[f][:, None] * cyf * np.conj(v)[None, :])
    dsf_dva[rows, f] += 1j * np.conj(i_from) * v[f]
    dsf_dvm = v[f][:, None] * cyf * np.conj(vnorm)[None, :]
    dsf_dvm[rows, f] += np.conj(i_from) * vnorm[f]
    dst_dva = -1j * (v[t][:, None] * cyt * np.conj(v)[None, :])
    dst_dva[rows, t] += 1j * np.conj(i_to) * v[t]
    dst_dvm = v[t][:, None] * cyt * np.conj(vnorm)[None, :]
    dst_dvm[rows, t] += np.conj(i_to) * vnorm[t]
    jac = np.zeros((3 * nb + 4 * nl, 2 * nb))
    jac[0:nb, nb:] = np.eye(nb)
    jac[nb:2 * nb, :nb] = ds_dva.real
    jac[nb:2 * nb, nb:] = ds_dvm.real
    jac[2 * nb:3 * nb, :nb] = ds_dva.imag
    jac[2 * nb:3 * nb, nb:] = ds_dvm.imag
    o = 3 * nb
    jac[o:o + nl, :nb] = dsf_dva.real
    jac[o:o + nl, nb:] = dsf_dvm.real
    jac[o + nl:o + 2 * nl, :nb] = dsf_dva.imag
    jac[o + nl:o + 2 * nl, nb:] = dsf_dvm.imag
    jac[o + 2 * nl:o + 3 * nl, :nb] = dst_dva.real
    jac[o + 2 * nl:o + 3 * nl, nb:] = dst_dvm.real
    jac[o + 3 * nl:o + 4 * nl, :nb] = dst_dva.imag
    jac[o + 3 * nl:o + 4 * nl, nb:] = dst_dvm.imag
    h = np.r_[vm, sinj.real, sinj.imag, sf.real, sf.imag, st.real, st.imag]
    return h, jac


# --------------------------------------------------------------- hypothesis models


@dataclass
class _Model:
    """A (possibly modified) copy of the operator model and its channel map."""

    base_mva: float
    bus: np.ndarray
    branch: np.ndarray
    #: Row of the model's full measurement vector for each operator channel;
    #: -1 marks a channel the hypothesis predicts as constant zero.
    sel: np.ndarray
    #: Full-vector rows forced to zero (the injections of a new, empty bus).
    extra_constraint_rows: list[int] = field(default_factory=list)
    #: Estimated parameters: ("R" | "X", branch) or ("G", bus).
    params: list[tuple[str, int]] = field(default_factory=list)
    #: New bus -> (from bus, to bus, alpha) for its warm start.
    init_from: dict[int, tuple[int, int, float]] = field(default_factory=dict)

    @property
    def nb(self) -> int:
        return int(self.bus.shape[0])

    @property
    def nl(self) -> int:
        return int(self.branch.shape[0])

    def full_row(self, block: str, index: int) -> int:
        nb, nl = self.nb, self.nl
        offsets = {"Vm": 0, "P": nb, "Q": 2 * nb, "Pf": 3 * nb, "Qf": 3 * nb + nl,
                   "Pt": 3 * nb + 2 * nl, "Qt": 3 * nb + 3 * nl}
        return offsets[block] + int(index)


class _Operator:
    """The operator's balanced model and the hypothesis copies built from it."""

    def __init__(self, base_mva: float, bus: np.ndarray, branch: np.ndarray) -> None:
        self.base_mva = float(base_mva)
        self.bus = np.asarray(bus, dtype=float).copy()
        self.branch = np.asarray(branch, dtype=float).copy()
        self.nb = int(self.bus.shape[0])
        self.nl = int(self.branch.shape[0])
        self.nz = 3 * self.nb + 4 * self.nl
        refs = np.flatnonzero(self.bus[:, BUS_TYPE].astype(int) == REF)
        if refs.size != 1:
            raise ValueError(f"expected exactly one reference bus, found {refs.size}")
        self.ref = int(refs[0])

    def sel(self, nb: int, nl: int, branch_map: Mapping[tuple[str, int], int] | None = None) -> np.ndarray:
        """Operator channels -> rows of a model with ``nb`` buses and ``nl`` branches.

        ``branch_map[(end, k)]`` names the model branch that carries original
        branch ``k``'s flow meter at end ``"f"`` or ``"t"`` (``-1``: constant 0).
        """
        branch_map = branch_map or {}
        nb0, nl0 = self.nb, self.nl
        sel = np.empty(self.nz, dtype=int)
        for i in range(nb0):
            sel[i] = i
            sel[nb0 + i] = nb + i
            sel[2 * nb0 + i] = 2 * nb + i
        for k in range(nl0):
            kf = branch_map.get(("f", k), k)
            kt = branch_map.get(("t", k), k)
            sel[3 * nb0 + k] = 3 * nb + kf if kf >= 0 else -1
            sel[3 * nb0 + nl0 + k] = 3 * nb + nl + kf if kf >= 0 else -1
            sel[3 * nb0 + 2 * nl0 + k] = 3 * nb + 2 * nl + kt if kt >= 0 else -1
            sel[3 * nb0 + 3 * nl0 + k] = 3 * nb + 3 * nl + kt if kt >= 0 else -1
        return sel

    def base_model(self) -> _Model:
        return _Model(self.base_mva, self.bus.copy(), self.branch.copy(), self.sel(self.nb, self.nl))

    def parameter_model(self, k: int, which: str) -> _Model:
        model = self.base_model()
        model.params = [(which, int(k))]
        return model

    def outage_model(self, k: int) -> _Model:
        model = self.base_model()
        model.branch[int(k), BR_STATUS] = 0.0
        return model

    def split_model(self, k: int, alpha: float) -> _Model:
        """Line ``k`` split at ``alpha`` from its from-bus; the new bus carries shunt G."""
        new_bus = self.nb
        bus = np.vstack([self.bus, self.bus[0:1]])
        bus[new_bus, BUS_I] = new_bus
        bus[new_bus, BUS_TYPE] = 1
        bus[new_bus, PD] = bus[new_bus, QD] = bus[new_bus, GS] = bus[new_bus, BS] = 0.0
        row = self.branch[int(k)].copy()
        from_bus, to_bus = int(row[F_BUS]), int(row[T_BUS])
        first = row.copy()
        second = row.copy()
        for column in (BR_R, BR_X, BR_B):
            first[column] = alpha * row[column]
            second[column] = (1.0 - alpha) * row[column]
        first[T_BUS] = new_bus
        second[F_BUS] = new_bus
        branch = np.vstack([self.branch, second[None, :]])
        branch[int(k)] = first
        branch_map = {("f", int(k)): int(k), ("t", int(k)): self.nl}
        model = _Model(self.base_mva, bus, branch, self.sel(self.nb + 1, self.nl + 1, branch_map))
        model.extra_constraint_rows = [model.full_row("P", new_bus), model.full_row("Q", new_bus)]
        model.params = [("G", new_bus)]
        model.init_from = {new_bus: (from_bus, to_bus, float(alpha))}
        return model

    def islanding_branches(self) -> set[int]:
        """In-service branches whose outage disconnects a bus."""
        live = [k for k in range(self.nl) if self.branch[k, BR_STATUS] > 0]
        islanding: set[int] = set()
        for k in live:
            if not self._connected([j for j in live if j != k]):
                islanding.add(k)
        return islanding

    def _connected(self, branches: Sequence[int]) -> bool:
        adjacency: dict[int, set[int]] = {b: set() for b in range(self.nb)}
        for k in branches:
            f, t = int(self.branch[k, F_BUS]), int(self.branch[k, T_BUS])
            adjacency[f].add(t)
            adjacency[t].add(f)
        seen = {self.ref}
        stack = [self.ref]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == self.nb


# ----------------------------------------------------------------------- solver


def _apply_params(model: _Model, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not model.params:
        return model.bus, model.branch
    bus = model.bus.copy()
    branch = model.branch.copy()
    for (kind, index), value in zip(model.params, values):
        if kind == "R":
            branch[index, BR_R] = value
        elif kind == "X":
            branch[index, BR_X] = value
        elif kind == "G":
            bus[index, GS] = value * model.base_mva
        else:
            raise ValueError(f"unknown screen parameter {kind!r}")
    return bus, branch


def _parameter_columns(model: _Model, bus, branch, va, vm, values, rows_total: int) -> np.ndarray:
    columns = np.zeros((rows_total, len(model.params)))
    for j, (kind, index) in enumerate(model.params):
        if kind == "G":
            columns[model.full_row("P", index), j] = vm[index] ** 2
            continue
        column = BR_R if kind == "R" else BR_X
        step = 1e-6 * max(1.0, abs(values[j]))
        plus = branch.copy()
        minus = branch.copy()
        plus[index, column] = values[j] + step
        minus[index, column] = values[j] - step
        columns[:, j] = (
            _h_full(*_ybus_dense(model.base_mva, bus, plus), va, vm)
            - _h_full(*_ybus_dense(model.base_mva, bus, minus), va, vm)
        ) / (2.0 * step)
    return columns


def _initial_state(model: _Model, nb0: int, va0: np.ndarray, vm0: np.ndarray):
    va = np.zeros(model.nb)
    vm = np.ones(model.nb)
    va[:nb0] = va0
    vm[:nb0] = vm0
    for bus, (from_bus, to_bus, alpha) in model.init_from.items():
        va[bus] = (1.0 - alpha) * va[from_bus] + alpha * va[to_bus]
        vm[bus] = (1.0 - alpha) * vm[from_bus] + alpha * vm[to_bus]
    return va, vm


def _initial_params(model: _Model) -> np.ndarray:
    values = []
    for kind, index in model.params:
        if kind == "R":
            values.append(model.branch[index, BR_R])
        elif kind == "X":
            values.append(model.branch[index, BR_X])
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


@dataclass
class _Problem:
    """One WLS problem: operator vector, sigmas, exact rows, dropped channels."""

    z: np.ndarray
    sigma: np.ndarray
    exact: tuple[int, ...]
    nb0: int
    ref: int

    def stochastic(self, drop: Sequence[int]) -> np.ndarray:
        excluded = set(self.exact) | {int(i) for i in drop}
        return np.array([i for i in range(self.z.size) if i not in excluded], dtype=int)


def _fit(model: _Model, problem: _Problem, drop: Sequence[int], va0, vm0,
         bounds: Mapping[int, tuple[float, float]] | None = None,
         max_iterations: int = 80, tolerance: float = 1e-9) -> dict[str, Any]:
    """Constrained Gauss-Newton WLS on a (modified) model.

    Converged when the step falls below ``tolerance``, or (large-residual
    linear convergence) when J stalls to 1e-10 relative for three iterations
    with steps below 1e-5.  ``bounds`` keeps a parameter inside its box by an
    active-set projection.
    """
    bounds = dict(bounds or {})
    nb = model.nb
    ref = problem.ref
    stochastic = problem.stochastic(drop)
    rows = model.sel[stochastic]
    zs = problem.z[stochastic]
    weights = 1.0 / problem.sigma[stochastic] ** 2
    constraint_rows = np.asarray(
        [int(model.sel[i]) for i in sorted(problem.exact)] + list(model.extra_constraint_rows), dtype=int
    )
    va, vm = _initial_state(model, problem.nb0, np.asarray(va0, float), np.asarray(vm0, float))
    values = _initial_params(model)
    for j, (low, high) in bounds.items():
        values[j] = min(max(values[j], low), high)
    free = list(range(len(model.params)))
    state_columns = np.r_[np.arange(ref), np.arange(ref + 1, 2 * nb)]
    ns = state_columns.size
    zero_rows = rows < 0
    rows_clipped = np.where(zero_rows, 0, rows)
    success = False
    stalled = 0
    previous = None
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        bus, branch = _apply_params(model, values)
        admittance = _ybus_dense(model.base_mva, bus, branch)
        h, jac = _h_and_jac(*admittance, va, vm)
        jac_state = jac[:, state_columns]
        jac_params = _parameter_columns(model, bus, branch, va, vm, values, h.size) if free else None
        predicted = np.where(zero_rows, 0.0, h[rows_clipped])
        error = zs - predicted
        objective = float(np.sum(weights * error * error))
        constraint = h[constraint_rows] if constraint_rows.size else np.zeros(0)
        active: list[int] = []
        while True:
            used = [j for j in free if j not in active]
            full = np.hstack([jac_state, jac_params[:, used]]) if used else jac_state
            measured = full[rows_clipped]
            measured[zero_rows] = 0.0
            gain = measured.T @ (measured * weights[:, None])
            rhs = measured.T @ (weights * error)
            try:
                if constraint_rows.size:
                    c_matrix = full[constraint_rows]
                    n = gain.shape[0]
                    kkt = np.zeros((n + constraint_rows.size, n + constraint_rows.size))
                    kkt[:n, :n] = gain
                    kkt[:n, n:] = c_matrix.T
                    kkt[n:, :n] = c_matrix
                    step = np.linalg.solve(kkt, np.r_[rhs, -constraint])[:n]
                else:
                    step = np.linalg.solve(gain, rhs)
            except np.linalg.LinAlgError:
                return {"success": False, "error": "singular", "iterations": iteration}
            if not np.all(np.isfinite(step)):
                return {"success": False, "error": "nonfinite_step", "iterations": iteration}
            blocked = []
            for position, j in enumerate(used):
                if j in bounds:
                    low, high = bounds[j]
                    delta = step[ns + position]
                    if (values[j] <= low + 1e-14 and delta < 0) or (values[j] >= high - 1e-14 and delta > 0):
                        blocked.append(j)
            if not blocked:
                break
            active.extend(blocked)
        largest = np.max(np.abs(step[:ns]))
        scale = 1.0 if largest < 0.5 else 0.5 / largest
        state_old = np.r_[va, vm]
        values_old = values.copy()
        state = state_old.copy()
        state[state_columns] += scale * step[:ns]
        va, vm = state[:nb], state[nb:]
        for position, j in enumerate(used):
            values[j] += scale * step[ns + position]
            if j in bounds:
                low, high = bounds[j]
                values[j] = min(max(values[j], low), high)
        if np.any(vm < 0.3) or np.any(vm > 2.0) or not np.all(np.isfinite(values)):
            return {"success": False, "error": "diverged", "iterations": iteration}
        change = max(np.max(np.abs(state - state_old)),
                     np.max(np.abs(values - values_old)) if values.size else 0.0)
        if change < tolerance and scale == 1.0:
            success = True
            break
        if previous is not None and abs(previous - objective) <= 1e-10 * max(objective, 1.0) and change < 1e-5:
            stalled += 1
            if stalled >= 3:
                success = True
                break
        else:
            stalled = 0
        previous = objective
    bus, branch = _apply_params(model, values)
    h = _h_full(*_ybus_dense(model.base_mva, bus, branch), va, vm)
    predicted = np.where(zero_rows, 0.0, h[rows_clipped])
    error = zs - predicted
    residual_constraint = float(np.max(np.abs(h[constraint_rows]))) if constraint_rows.size else 0.0
    if residual_constraint > 1e-6:
        success = False
    return {
        "success": bool(success), "J": float(np.sum(weights * error * error)), "va": va, "vm": vm,
        "params": values.copy(), "iterations": iteration, "residual": error, "stochastic": stochastic,
    }


def _normalized_residuals(model: _Model, problem: _Problem, fit: Mapping[str, Any]) -> np.ndarray:
    """|e_i| / sqrt(Omega_ii) at a converged fit, over the full operator vector."""
    from scipy.linalg import null_space

    nb = model.nb
    ref = problem.ref
    stochastic = fit["stochastic"]
    rows = model.sel[stochastic]
    zero_rows = rows < 0
    rows_clipped = np.where(zero_rows, 0, rows)
    constraint_rows = np.asarray(
        [int(model.sel[i]) for i in sorted(problem.exact)] + list(model.extra_constraint_rows), dtype=int
    )
    bus, branch = _apply_params(model, fit["params"])
    h, jac = _h_and_jac(*_ybus_dense(model.base_mva, bus, branch), fit["va"], fit["vm"])
    state_columns = np.r_[np.arange(ref), np.arange(ref + 1, 2 * nb)]
    full = jac[:, state_columns]
    if model.params:
        full = np.hstack([full, _parameter_columns(model, bus, branch, fit["va"], fit["vm"], fit["params"], h.size)])
    measured = full[rows_clipped]
    measured[zero_rows] = 0.0
    variance = problem.sigma[stochastic] ** 2
    reduced = measured @ null_space(full[constraint_rows]) if constraint_rows.size else measured
    gain = reduced.T @ (reduced / variance[:, None])
    projection = reduced @ np.linalg.solve(gain, reduced.T)
    omega = variance - np.diag(projection)
    out = np.zeros(problem.z.size)
    out[stochastic] = np.abs(fit["residual"]) / np.sqrt(np.clip(omega, 1e-300, None))
    return out


# ------------------------------------------------------------------------ screen


def ieee14_hif_lines(branch: np.ndarray) -> list[int]:
    """In-service rows of the 16 same-voltage IEEE 14 lines (external numbering)."""
    return [
        k for k in range(branch.shape[0])
        if branch[k, BR_STATUS] > 0
        and (int(branch[k, F_BUS]), int(branch[k, T_BUS])) in IEEE14_HIF_LINE_ENDPOINTS
    ]


def default_hif_lines(case: Mapping[str, Any]) -> list[int]:
    """Candidate HIF lines of an external-numbered MATPOWER case.

    IEEE 14 (identified by its bus numbering and the 1-2 ... 13-14 lines) uses
    the 16 same-voltage lines of the HIF corpora.  Any other case uses its
    in-service untapped branches whose two ends share a nonzero base kV.
    """
    bus = np.asarray(case["bus"], dtype=float)
    branch = np.asarray(case["branch"], dtype=float)
    numbers = sorted(int(value) for value in bus[:, BUS_I])
    if numbers == list(range(1, 15)):
        lines = ieee14_hif_lines(branch)
        if lines:
            return lines
    base_kv = {int(row[BUS_I]): float(row[BASE_KV]) for row in bus} if bus.shape[1] > BASE_KV else {}
    lines = []
    for k in range(branch.shape[0]):
        if branch[k, BR_STATUS] <= 0 or branch[k, TAP] not in (0.0, 1.0) or branch[k, SHIFT] != 0.0:
            continue
        f, t = int(branch[k, F_BUS]), int(branch[k, T_BUS])
        kv_f, kv_t = base_kv.get(f, 0.0), base_kv.get(t, 0.0)
        if kv_f > 0.0 and math.isclose(kv_f, kv_t, rel_tol=1e-9):
            lines.append(k)
    return lines


def _plausible_box(branch: np.ndarray, kind: str, k: int) -> tuple[float, float]:
    r0 = float(branch[k, BR_R])
    x0 = float(branch[k, BR_X])
    return (0.0, max(20.0 * r0, 0.5 * x0)) if kind == "R" else (0.05 * x0, 20.0 * x0)


def _penalty(config: HifScreenConfig, continuous: int, candidates: int) -> float:
    return config.continuous_penalty * continuous + config.discrete_penalty * math.log(max(candidates, 1))


def _refined_alpha(profile: np.ndarray, index: int, alphas: Sequence[float]) -> float:
    """Parabolic refinement of the grid minimum of a line's J(alpha) profile."""
    alpha = float(alphas[index])
    if 0 < index < len(alphas) - 1 and np.all(np.isfinite(profile[index - 1:index + 2])):
        low, mid, high = profile[index - 1], profile[index], profile[index + 1]
        curvature = low - 2.0 * mid + high
        spacing = float(alphas[index + 1] - alphas[index])
        if curvature > 0:
            alpha = float(alpha + spacing * 0.5 * (low - high) / curvature)
    return alpha


def _round(operator: _Operator, problem: _Problem, drop: list[int], base: Mapping[str, Any],
           r_norm: np.ndarray, lines: Sequence[int], config: HifScreenConfig, *,
           hif_enabled: bool = True) -> dict[str, Any]:
    """Every hypothesis refit on one channel set; class scores and their best candidates.

    The parameter class is the best of three variants per live branch (R, X,
    or R and X jointly, the joint one charged two continuous parameters).  The
    HIF class is the best of the split-line shunt and, on the top-ranked
    lines, the shunt with the end buses' phase-A voltage channels set aside
    (one continuous parameter each).  ``hif_enabled`` is false once two
    meters have been set aside.
    """
    j0 = float(base["J"])
    va, vm = base["va"], base["vm"]
    n_stochastic = int(base["stochastic"].size)
    base_model = operator.base_model()
    best: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    fits = 0

    order = [int(i) for i in np.argsort(-r_norm)[:config.meter_candidates] if r_norm[int(i)] > 0.0]
    meter = []
    for channel in order:
        fit = _fit(base_model, problem, drop + [channel], va, vm)
        fits += 1
        meter.append((float(fit["J"]) if fit.get("success") else math.inf, channel))
    if meter:
        j_meter, channel = min(meter)
        if math.isfinite(j_meter):
            scores["meter"] = j0 - j_meter - _penalty(config, 1, n_stochastic)
            best["meter"] = {
                "channel_index0": channel, "J": j_meter,
                "ranked": [{"channel_index0": int(c), "J": float(j)} for j, c in sorted(meter) if math.isfinite(j)][:5],
            }

    live = [k for k in range(operator.nl) if operator.branch[k, BR_STATUS] > 0]
    parameter: list[tuple[float, float, int, str, Any]] = []
    for k in live:
        for which in ("R", "X"):
            fit = _fit(operator.parameter_model(k, which), problem, drop, va, vm,
                       bounds={0: _plausible_box(operator.branch, which, k)})
            fits += 1
            if fit.get("success"):
                parameter.append((j0 - float(fit["J"]) - _penalty(config, 1, 2 * len(live)), float(fit["J"]), k, which,
                                  float(fit["params"][0])))
        if config.parameter_rx_variant:
            model = operator.base_model()
            model.params = [("R", k), ("X", k)]
            fit = _fit(model, problem, drop, va, vm,
                       bounds={0: _plausible_box(operator.branch, "R", k), 1: _plausible_box(operator.branch, "X", k)})
            fits += 1
            if fit.get("success"):
                parameter.append((j0 - float(fit["J"]) - _penalty(config, 2, len(live)), float(fit["J"]), k, "RX",
                                  {"R": float(fit["params"][0]), "X": float(fit["params"][1])}))
    if parameter:
        parameter.sort(key=lambda item: -item[0])
        score, j_param, k, which, estimate = parameter[0]
        scores["parameter"] = float(score)
        per_branch: dict[int, tuple[float, float, int, str, Any]] = {}
        for item in parameter:
            per_branch.setdefault(int(item[2]), item)
        best["parameter"] = {
            "branch_row0": int(k), "parameter": which, "estimate": estimate, "J": j_param,
            "ranked": [
                {"branch_row0": int(kk), "parameter": w, "estimate": e, "J": float(j), "score": float(s)}
                for s, j, kk, w, e in sorted(per_branch.values(), key=lambda item: item[1])
            ][:5],
        }

    islanding = operator.islanding_branches()
    outages = [k for k in live if k not in islanding]
    topology = []
    for k in outages:
        fit = _fit(operator.outage_model(k), problem, drop, va, vm)
        fits += 1
        if fit.get("success"):
            topology.append((float(fit["J"]), k))
    if topology:
        j_topo, k = min(topology)
        scores["topology"] = j0 - j_topo - _penalty(config, 0, len(outages))
        best["topology"] = {
            "branch_row0": int(k), "J": j_topo,
            "ranked": [{"branch_row0": int(k_row), "J": float(j_row)} for j_row, k_row in sorted(topology)][:5],
        }

    alphas = tuple(float(a) for a in config.alphas)
    if hif_enabled and lines:
        grid = np.full((len(lines), len(alphas)), np.nan)
        conductance = np.full((len(lines), len(alphas)), np.nan)
        for li, k in enumerate(lines):
            for ai, alpha in enumerate(alphas):
                fit = _fit(operator.split_model(k, alpha), problem, drop, va, vm, bounds={0: (0.0, math.inf)})
                fits += 1
                if fit.get("success"):
                    grid[li, ai] = float(fit["J"])
                    conductance[li, ai] = float(fit["params"][0])
        if np.any(np.isfinite(grid)):
            finite = np.where(np.isfinite(grid), grid, np.inf)
            per_line = np.min(finite, axis=1)
            ranked_positions = [int(i) for i in np.argsort(per_line) if np.isfinite(per_line[i])]
            li, ai = (int(v) for v in np.unravel_index(int(np.argmin(finite)), grid.shape))
            j_hif = float(grid[li, ai])
            score_hif = j0 - j_hif - _penalty(config, 1, grid.size)
            best_hif: dict[str, Any] = {
                "variant": "shunt", "branch_row0": int(lines[li]), "alpha_grid": float(alphas[ai]),
                "alpha": _refined_alpha(grid[li], ai, alphas), "shunt_conductance_pu": float(conductance[li, ai]),
                "J": j_hif,
                "runner_up_branch_row0": int(lines[ranked_positions[1]]) if len(ranked_positions) > 1 else None,
                "ranked": [{"branch_row0": int(lines[i]), "J": float(per_line[i])} for i in ranked_positions][:5],
            }
            if config.hif_vm_offset_variant:
                excluded = set(drop) | set(problem.exact)
                for position in ranked_positions[: max(int(config.hif_vm_offset_lines), 0)]:
                    k = int(lines[position])
                    ai_k = int(np.argmin(finite[position]))
                    alpha = float(alphas[ai_k])
                    # The Vm channel of bus row b is operator channel b.
                    offsets = [int(b) for b in (operator.branch[k, F_BUS], operator.branch[k, T_BUS]) if int(b) not in excluded]
                    if not offsets:
                        continue
                    fit = _fit(operator.split_model(k, alpha), problem, drop + offsets, va, vm, bounds={0: (0.0, math.inf)})
                    fits += 1
                    if not fit.get("success"):
                        continue
                    score = j0 - float(fit["J"]) - _penalty(config, 1 + len(offsets), grid.size)
                    if score > score_hif:
                        score_hif = score
                        best_hif = {
                            **best_hif, "variant": "shunt+vm_offsets", "branch_row0": k, "alpha_grid": alpha,
                            "alpha": _refined_alpha(grid[position], ai_k, alphas),
                            "shunt_conductance_pu": float(fit["params"][0]), "J": float(fit["J"]),
                            "vm_offset_channels": offsets,
                        }
            scores["hif"] = float(score_hif)
            best["hif"] = best_hif
    winner = max(scores, key=scores.__getitem__) if scores else None
    return {
        "dropped_channels": list(drop),
        "J0": j0,
        "stochastic_channels": n_stochastic,
        "scores": {name: float(value) for name, value in scores.items()},
        "best": best,
        "winner": winner,
        "hif_compared": bool(hif_enabled and lines),
        "fits": fits,
    }


#: Free continuous parameters each class's best refit adds to the balanced
#: fit (the same count the penalty charges).
def _class_continuous(name: str, item: Mapping[str, Any]) -> int:
    if name == "meter":
        return 1
    if name == "parameter":
        return 2 if item.get("parameter") == "RX" else 1
    if name == "topology":
        return 0
    return 1 + len(item.get("vm_offset_channels") or [])


def _class_model(operator: _Operator, name: str, item: Mapping[str, Any], drop: Sequence[int]):
    """The model, dropped channels and bounds of a class's best refit."""
    fit_drop = list(drop)
    bounds = None
    if name == "meter":
        model = operator.base_model()
        fit_drop = fit_drop + [int(item["channel_index0"])]
    elif name == "parameter":
        k = int(item["branch_row0"])
        if item.get("parameter") == "RX":
            model = operator.base_model()
            model.params = [("R", k), ("X", k)]
            bounds = {0: _plausible_box(operator.branch, "R", k), 1: _plausible_box(operator.branch, "X", k)}
        else:
            model = operator.parameter_model(k, str(item["parameter"]))
            bounds = {0: _plausible_box(operator.branch, str(item["parameter"]), k)}
    elif name == "topology":
        model = operator.outage_model(int(item["branch_row0"]))
    else:
        model = operator.split_model(int(item["branch_row0"]), float(item["alpha_grid"]))
        fit_drop = fit_drop + [int(c) for c in item.get("vm_offset_channels") or []]
        bounds = {0: (0.0, math.inf)}
    return model, fit_drop, bounds


def _class_tests(operator: _Operator, problem: _Problem, drop: Sequence[int], base: Mapping[str, Any],
                 result: Mapping[str, Any], config: HifScreenConfig, round_dof: int) -> dict[str, Any]:
    """Offline study fields for one compared round.

    Each class's best refit is re-run once and tested against the alarm rule
    it would have to clear (chi-square at its remaining degrees of freedom,
    or the normalized-residual limit); the winner is also tried with one more
    meter set aside.  Nothing here changes the decision.
    """
    from scipy.stats import chi2

    va, vm = base["va"], base["vm"]
    limit = config.normalized_residual_threshold

    def tested(model: _Model, fit_drop: list[int], bounds, dof_after: int) -> dict[str, Any]:
        entry: dict[str, Any] = {"dof_after": int(dof_after),
                                 "chi2_threshold_after": float(chi2.ppf(1.0 - config.chi2_alpha, max(dof_after, 1)))}
        fit = _fit(model, problem, fit_drop, va, vm, bounds=bounds)
        if fit.get("success"):
            residuals = _normalized_residuals(model, problem, fit)
            entry.update(J_refit=float(fit["J"]), max_normalized_residual_after=float(np.max(residuals)),
                         explains_alarm=bool(float(fit["J"]) < entry["chi2_threshold_after"]
                                             and float(np.max(residuals)) < limit),
                         _residuals=residuals)
        else:
            entry.update(J_refit=None, max_normalized_residual_after=None, explains_alarm=False,
                         refit_error=str(fit.get("error")))
        return entry

    classes: dict[str, dict[str, Any]] = {}
    kept: dict[str, tuple[Any, list[int], Any, dict[str, Any]]] = {}
    for name, item in (result.get("best") or {}).items():
        model, fit_drop, bounds = _class_model(operator, name, item, drop)
        entry = tested(model, fit_drop, bounds, round_dof - _class_continuous(name, item))
        entry["J_after"] = float(item["J"])
        kept[name] = (model, fit_drop, bounds, entry)
        classes[name] = {key: value for key, value in entry.items() if not key.startswith("_")}
    extras: dict[str, Any] = {"winner_plus_one_meter": None}
    winner = result.get("winner")
    if winner in kept:
        model, fit_drop, bounds, entry = kept[winner]
        residuals = entry.get("_residuals")
        if residuals is not None:
            excluded = set(fit_drop) | set(problem.exact)
            order = [int(i) for i in np.argsort(-residuals) if int(i) not in excluded and residuals[int(i)] > 0.0]
            for channel in order[: config.meter_candidates]:
                trial = tested(model, fit_drop + [channel], bounds,
                               round_dof - _class_continuous(winner, result["best"][winner]) - 1)
                if trial.get("J_refit") is None:
                    continue
                trial["channel_index0"] = channel
                current = extras["winner_plus_one_meter"]
                if current is None or trial["J_refit"] < current["J_refit"]:
                    extras["winner_plus_one_meter"] = {key: value for key, value in trial.items() if not key.startswith("_")}
    scores = {str(name): float(value) for name, value in (result.get("scores") or {}).items()}
    ordered = sorted(scores.values(), reverse=True)
    return {
        "classes": classes,
        "any_class_explains_alarm": any(entry["explains_alarm"] for entry in classes.values()),
        "winner_explains_alarm": bool(classes.get(str(winner), {}).get("explains_alarm", False)),
        "score_margin": (float(ordered[0] - ordered[1]) if len(ordered) > 1 else None),
        "extras": extras,
    }


def screen_hif(
    base_mva: float,
    bus: np.ndarray,
    branch: np.ndarray,
    z: Sequence[float],
    sigma: Sequence[float],
    exact_indices: Sequence[int] = (),
    *,
    lines: Sequence[int],
    va0: Sequence[float] | None = None,
    vm0: Sequence[float] | None = None,
    dof: int | None = None,
    config: HifScreenConfig = DEFAULT_HIF_SCREEN_CONFIG,
) -> dict[str, Any]:
    """Run the screen on an internal-indexed balanced model (0-based buses).

    ``lines`` are the candidate HIF branch rows; ``va0``/``vm0`` warm-start
    the first refit (the operator's WLS state, else the case voltages) and
    each later round starts from the previous base fit; ``dof`` is the base
    solve's chi-square degrees of freedom, used for the alarm test between
    rounds.  Returns a policy-safe report: the compared rounds with their
    class scores, the accepted hypotheses in order, and for a suspicion the
    line, position and shunt conductance of the best split-line fit.
    """
    from scipy.stats import chi2

    operator = _Operator(base_mva, bus, branch)  # working copy: accepted hypotheses modify it
    z_vector = np.asarray(z, dtype=float).reshape(-1)
    sigma_vector = np.asarray(sigma, dtype=float).reshape(-1)
    if z_vector.size != operator.nz or sigma_vector.size != operator.nz:
        raise ValueError(f"screen expects {operator.nz} channels, got z={z_vector.size} sigma={sigma_vector.size}")
    exact = tuple(sorted(int(i) for i in exact_indices))
    problem = _Problem(z_vector, sigma_vector, exact, operator.nb, operator.ref)
    if va0 is None or vm0 is None:
        va0 = np.deg2rad(operator.bus[:, VA] - operator.bus[operator.ref, VA])
        vm0 = operator.bus[:, VM]
    lines = [int(k) for k in lines if 0 <= int(k) < operator.nl and operator.branch[int(k), BR_STATUS] > 0]
    report: dict[str, Any] = {
        "method": HIF_SCREEN_METHOD,
        "status": "valid",
        "suspected": False,
        "candidate_lines": len(lines),
        "penalties": {"continuous": config.continuous_penalty, "discrete_log": config.discrete_penalty},
        "rounds": [],
        "accepted_hypotheses": [],
    }
    drop: list[int] = []
    base_dof = None if dof is None else int(dof)
    outcome: list[str] = []
    accepted_continuous = 0
    va_start = np.asarray(va0, dtype=float)
    vm_start = np.asarray(vm0, dtype=float)
    hif_round: dict[str, Any] | None = None
    explained = False
    tests: dict[str, Any] | None = None
    final_base: dict[str, Any] = {}
    max_rounds = max(1, int(config.max_rounds))
    for round_index in range(max_rounds + 1):
        base = _fit(operator.base_model(), problem, drop, va_start, vm_start)
        if not base.get("success"):
            report.update(status="base_fit_failed")
            break
        r_norm = _normalized_residuals(operator.base_model(), problem, base)
        va_start, vm_start = base["va"], base["vm"]
        round_dof = (base_dof if base_dof is not None else int(base["stochastic"].size) - (2 * operator.nb - 1)) \
            - len(drop) - accepted_continuous
        threshold = float(chi2.ppf(1.0 - config.chi2_alpha, max(round_dof, 1)))
        max_r = float(np.max(r_norm)) if np.size(r_norm) else 0.0
        alarmed = bool(float(base["J"]) >= threshold or max_r >= config.normalized_residual_threshold)
        final_base = {"round_index": round_index, "set_aside_channels": [int(i) for i in drop], "J0": float(base["J"]),
                      "dof": int(round_dof), "chi2_threshold": threshold, "max_normalized_residual": max_r,
                      "alarm": alarmed}
        if round_index > 0 and not alarmed:
            # The accepted hypotheses explain the whole alarm.
            report["rounds"].append({"dropped_channels": list(drop), "J0": float(base["J"]), "clean_after_removal": True})
            explained = True
            break
        if round_index == max_rounds:
            break  # comparisons exhausted while the alarm stands
        live_lines = [k for k in lines if operator.branch[k, BR_STATUS] > 0]
        result = _round(operator, problem, drop, base, r_norm, live_lines, config, hif_enabled=len(drop) <= 1)
        tests = _class_tests(operator, problem, drop, base, result, config, round_dof)
        result["class_tests"] = tests
        report["rounds"].append(result)
        winner = result["winner"]
        if winner is None:
            report.update(status="no_hypothesis_converged")
            break
        outcome.append(winner)
        item = result["best"][winner]
        if winner == "hif":
            hif_round = result
            break
        if winner == "meter":
            drop = drop + [int(item["channel_index0"])]
            report["accepted_hypotheses"].append({"class": "meter", "channel_index0": int(item["channel_index0"])})
        elif winner == "parameter":
            k = int(item["branch_row0"])
            if item["parameter"] == "RX":
                operator.branch[k, BR_R] = float(item["estimate"]["R"])
                operator.branch[k, BR_X] = float(item["estimate"]["X"])
                accepted_continuous += 2
            elif item["parameter"] == "R":
                operator.branch[k, BR_R] = float(item["estimate"])
                accepted_continuous += 1
            else:
                operator.branch[k, BR_X] = float(item["estimate"])
                accepted_continuous += 1
            report["accepted_hypotheses"].append({"class": "parameter", "branch_row0": k,
                                                  "parameter": item["parameter"], "estimate": item["estimate"]})
        else:
            k = int(item["branch_row0"])
            operator.branch[k, BR_STATUS] = 0.0
            report["accepted_hypotheses"].append({"class": "topology", "branch_row0": k})
    report["outcome"] = ">".join(outcome) if outcome else None
    report["explained"] = bool(explained)
    report["voltage_meter_channels"] = [int(c) for c in drop if int(c) < operator.nb]
    if hif_round is not None:
        hif = hif_round["best"]["hif"]
        others = [value for name, value in hif_round["scores"].items() if name != "hif"]
        report.update(
            suspected=True,
            branch_row0=int(hif["branch_row0"]),
            alpha=float(hif["alpha"]),
            shunt_conductance_pu=float(hif["shunt_conductance_pu"]),
            hif_variant=str(hif.get("variant") or "shunt"),
            meter_set_aside_index0=(int(drop[-1]) if drop else None),
            score_margin=float(hif_round["scores"]["hif"] - max(others, default=-math.inf)),
        )
    report["unexplained"] = bool(report["status"] == "valid" and not report["suspected"] and not explained)
    report["phasor_suspicion"] = {
        "hif": bool(report["suspected"]),
        "voltage_meter": bool(report["voltage_meter_channels"]),
        "unexplained": bool(report["unexplained"]),
    }
    _attach_offline_summary(report, final_base, tests, explained)
    return report


def _attach_offline_summary(report: dict[str, Any], final_base: Mapping[str, Any],
                            tests: Mapping[str, Any] | None, explained: bool) -> None:
    """Offline study fields (hypothesis-ranking plan, steps 1 and 2).

    ``final`` describes the last base solve and the last compared round's
    class tests; ``unexplained_variants`` gives the production flag beside two
    reference definitions (the first round alone; the winner with one more
    meter set aside).  The compact policy-visible report copies none of this.
    """
    if report.get("status") != "valid" or not report.get("rounds"):
        report["final"] = None
        report["unexplained_variants"] = None
        return
    compared = [item for item in report["rounds"] if item.get("winner") is not None or item.get("scores")]
    last = compared[-1] if compared else {}
    first_tests = (compared[0].get("class_tests") if compared else None) or {}
    final: dict[str, Any] = dict(final_base)
    final.update(
        winner=last.get("winner"),
        scores=dict(last.get("scores") or {}),
        explained_by_meter_removal=bool(explained and report["accepted_hypotheses"]
                                        and all(h["class"] == "meter" for h in report["accepted_hypotheses"])),
        explained=bool(explained),
    )
    if tests:
        final.update(classes=tests["classes"], any_class_explains_alarm=tests["any_class_explains_alarm"],
                     winner_explains_alarm=tests["winner_explains_alarm"], score_margin=tests["score_margin"],
                     extras=tests["extras"])
    else:
        final.update(classes={}, any_class_explains_alarm=False, winner_explains_alarm=False, score_margin=None,
                     extras={"winner_plus_one_meter": None})
    report["final"] = final
    production = bool(report["unexplained"])
    one_more = ((tests or {}).get("extras") or {}).get("winner_plus_one_meter") or {}
    report["unexplained_variants"] = {
        "production": production,
        "first_round_single_cause": bool(report["rounds"] and compared and not first_tests.get("any_class_explains_alarm", False)
                                         and not report["suspected"]),
        "with_one_more_meter": bool(production and not one_more.get("explains_alarm")),
    }


def compact_screen_targets(best: Mapping[str, Any]) -> dict[str, Any]:
    """Each class's best target with its refit objective, in compact form."""
    compact: dict[str, Any] = {}
    for name, item in best.items():
        if not isinstance(item, Mapping):
            continue
        entry: dict[str, Any] = {"J": round(float(item["J"]), 3)} if item.get("J") is not None else {}
        if name == "meter":
            entry["channel_index0"] = int(item["channel_index0"])
        elif name == "parameter":
            entry["branch_row0"] = int(item["branch_row0"])
            entry["parameter"] = str(item.get("parameter") or "")
        elif name == "topology":
            entry["branch_row0"] = int(item["branch_row0"])
        else:
            entry["branch_row0"] = int(item["branch_row0"])
            entry["alpha_from_from_bus"] = round(float(item.get("alpha", item.get("alpha_grid", 0.0))), 3)
        compact[str(name)] = entry
    return compact


def compact_screen_ranking(best: Mapping[str, Any], top: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Top-ranked alternatives per class (targets and refit objectives only)."""
    compact: dict[str, list[dict[str, Any]]] = {}
    for name, item in best.items():
        ranked = item.get("ranked") if isinstance(item, Mapping) else None
        if not isinstance(ranked, (list, tuple)):
            continue
        rows: list[dict[str, Any]] = []
        for candidate in list(ranked)[:top]:
            if not isinstance(candidate, Mapping):
                continue
            row: dict[str, Any] = {"J": round(float(candidate["J"]), 3)} if candidate.get("J") is not None else {}
            if "channel_index0" in candidate:
                row["channel_index0"] = int(candidate["channel_index0"])
            if "branch_row0" in candidate:
                row["branch_row0"] = int(candidate["branch_row0"])
            if candidate.get("parameter") is not None:
                row["parameter"] = str(candidate["parameter"])
            rows.append(row)
        compact[str(name)] = rows
    return compact


def compact_screen_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """The policy-visible form of a screen report, as it rides on the WLS ledger.

    One function for the provider (``MatpowerDeploymentProviders._hif_screen``)
    and the offline study, so a ranker trained on the study's states reads
    exactly the fields the policy sees: status, the HIF suspicion and its
    outcome, whether the accepted hypotheses explain the alarm, the
    phase-A voltage channels set aside, the three phasor suspicions, and
    per round the winner, the class scores, each class's best target and
    its top-ranked alternatives (objectives rounded to three decimals).
    None of the offline fields (``final``, ``class_tests``,
    ``unexplained_variants``) is copied.
    """
    phasor_kinds = report.get("phasor_suspicion")
    compact: dict[str, Any] = {
        "method": report.get("method", HIF_SCREEN_METHOD),
        "status": report.get("status"),
        "suspected": bool(report.get("suspected")),
        "outcome": report.get("outcome"),
        "explained": bool(report.get("explained")),
        "unexplained": bool(report.get("unexplained")),
        "accepted_hypotheses": [dict(item) for item in report.get("accepted_hypotheses") or []],
        "voltage_meter_channels": [int(c) for c in report.get("voltage_meter_channels") or []],
        "phasor_suspicion": (
            {str(k): bool(v) for k, v in phasor_kinds.items()} if isinstance(phasor_kinds, Mapping)
            else {"hif": bool(report.get("suspected")), "voltage_meter": False, "unexplained": False}
        ),
        "rounds": [
            {"winner": item.get("winner"), "clean_after_removal": item.get("clean_after_removal"),
             "set_aside_channels": list(item.get("dropped_channels") or []),
             "scores": {name: round(float(value), 3) for name, value in (item.get("scores") or {}).items()},
             "best": compact_screen_targets(item.get("best") or {}),
             "ranked": compact_screen_ranking(item.get("best") or {})}
            for item in report.get("rounds") or []
            if isinstance(item, Mapping)
        ],
    }
    if report.get("error"):
        compact["error"] = report["error"]
    if compact["suspected"]:
        compact.update(
            branch_row0=int(report["branch_row0"]),
            line_index1=int(report["branch_row0"]) + 1,
            alpha_from_from_bus=round(float(report["alpha"]), 3),
            meter_set_aside_index0=report.get("meter_set_aside_index0"),
            hif_variant=str(report.get("hif_variant") or "shunt"),
        )
    return compact


class HifScreenCache:
    """Bounded memo of screen reports keyed by every input the screen reads."""

    def __init__(self, limit: int = 64) -> None:
        self.limit = int(limit)
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @staticmethod
    def key(*arrays: Any, config: HifScreenConfig) -> str:
        digest = hashlib.sha256()
        for value in arrays:
            array = np.ascontiguousarray(np.asarray(value, dtype=float))
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(repr(config).encode())
        return digest.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: str, report: dict[str, Any]) -> None:
        self._items[key] = report
        self._items.move_to_end(key)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)


__all__ = [
    "DEFAULT_HIF_SCREEN_CONFIG", "HIF_SCREEN_CLASSES", "HIF_SCREEN_METHOD", "HIF_SCREEN_SIGNATURE",
    "HifScreenCache", "HifScreenConfig", "IEEE14_HIF_LINE_ENDPOINTS", "UNEXPLAINED_SCREEN_SIGNATURE",
    "VOLTAGE_METER_SCREEN_SIGNATURE", "compact_screen_ranking", "compact_screen_report", "compact_screen_targets",
    "default_hif_lines", "ieee14_hif_lines", "screen_hif",
]
