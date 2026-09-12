"""Node/breaker generalized state estimation with normalized Lagrange multipliers.

Given the operator's *reported* breaker statuses and the substation telemetry, this
estimator solves the state of the full IEEE-14 node/breaker model and tests every
reported status. The state is ``[theta(nodes), V(nodes), P_cb, Q_cb]``: breaker flows
are explicit state variables and the physical branches keep their admittances. Each
reported status enters as a pair of equality constraints,

* closed breaker ``k = (a, b)``: ``theta_a - theta_b = 0`` and ``V_a - V_b = 0``;
* open breaker: ``P_k = 0`` and ``Q_k = 0``,

together with zero-injection constraints at the switching nodes that host no unit or
load. The constrained weighted least-squares problem is solved with the Hachtel
(KKT) system

    [ H'WH   C' ] [dx]   [ H'W (z - h(x)) ]
    [ C    -eI  ] [ l ] = [ -c(x)          ]

whose lower-right block is a tiny regularization that absorbs the linearly dependent
constraints of closed-breaker loops. At convergence the multipliers ``l`` and their
covariance, minus the lower-right block of the KKT inverse, give the normalized
multipliers ``l / sqrt(cov)``. The breaker whose constraints carry the largest
normalized multiplier is the prime suspect for a wrong reported status (Clements and
Simões Costa, 1998). Constraints made redundant by a loop get a near-infinite
variance and therefore a normalized multiplier of zero: flipping such a breaker does
not change the connectivity partition, so no analog evidence can implicate it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BS, BUS_TYPE, GS, REF
from pypower.idx_gen import GEN_BUS

try:
    from Transmission.ieee14_full_measurements import NB, NL
    from Transmission.ieee14_full_substation import TELEMETRY_SIGMA, injection_metered_nodes
    from Transmission.ieee14_full_topology import FullTopology
    from tools.lagrangian_port import dSbr_dV1, dSbus_dV_polar, make_ybus
except ImportError:  # pragma: no cover - direct execution inside Transmission/
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    from Transmission.ieee14_full_measurements import NB, NL  # type: ignore
    from Transmission.ieee14_full_substation import TELEMETRY_SIGMA, injection_metered_nodes  # type: ignore
    from Transmission.ieee14_full_topology import FullTopology  # type: ignore
    from tools.lagrangian_port import dSbr_dV1, dSbus_dV_polar, make_ybus  # type: ignore

# Regularization of the constraint block (pu^2 scale of the multiplier covariance).
DEFAULT_REGULARIZATION = 1e-9
# Multiplier variances above this multiple of the regularization ceiling are treated
# as unidentifiable (loop-dependent) and reported as zero.
_UNIDENTIFIABLE_FRACTION = 0.5

__all__ = [
    "DEFAULT_REGULARIZATION",
    "gse_topology_nlm",
    "node_network",
    "rank_breakers",
    "screen_breaker_flips",
]


def rank_breakers(multipliers: Mapping[str, Mapping[str, float]], *, tie_tolerance: float = 1e-3) -> list[str]:
    """Order breakers by their largest normalized multiplier, breaking ties by the other one.

    Hard equality constraints in series carry the same first-order multiplier (the
    tension of a chain), so along a string of closed breakers the angle multipliers
    tie exactly. The second constraint of the pair (voltage equality for a closed
    breaker, reactive flow for an open one) is not shared along the chain and
    resolves the tie toward the breaker whose relaxation explains the data.
    """
    items = []
    for name, comps in multipliers.items():
        values = sorted((abs(float(v)) for v in comps.values()), reverse=True)
        items.append((name, values[0] if values else 0.0, values[1] if len(values) > 1 else 0.0))
    items.sort(key=lambda t: (-t[1], -t[2], t[0]))
    ordered: list[str] = []
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and abs(items[j + 1][1] - items[i][1]) <= tie_tolerance * max(items[i][1], 1e-9):
            j += 1
        ordered.extend(name for name, _, _ in sorted(items[i : j + 1], key=lambda t: (-t[2], t[0])))
        i = j + 1
    return ordered


def screen_breaker_flips(
    model: FullTopology,
    reference: Mapping[str, Any],
    reported_status: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    candidates: Sequence[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Re-estimate with each candidate breaker's reported status flipped.

    The breaker whose flip brings the node/breaker chi-square below its threshold
    explains the substation telemetry; the multiplier ranking only proposes.
    """
    states = model.states(reported_status)
    results = []
    for name in candidates:
        flipped = dict(reported_status)
        flipped[name] = not states[name]
        outcome = gse_topology_nlm(model, reference, flipped, telemetry, **kwargs)
        results.append(
            {
                "cb_name": name,
                "reported_status": "closed" if states[name] else "open",
                "proposed_status": "open" if states[name] else "closed",
                "success": bool(outcome["success"]),
                "chi_square": float(outcome["chi_square"]),
                "dof": int(outcome["dof"]),
            }
        )
    return results


def node_network(model: FullTopology, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Internal-numbered node network: physical branches, shunts, breaker incidence."""
    nodes = list(model.nodes)
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    ref_bus = np.asarray(reference["bus"], dtype=float)
    ref_branch = np.asarray(reference["branch"], dtype=float)
    ref_gen = np.asarray(reference["gen"], dtype=float)
    bus = np.zeros((n, 13))
    bus[:, 0] = np.arange(n)
    bus[:, 1] = 1.0
    bus[:, 7] = 1.0
    for b in range(1, NB + 1):
        bus[index[model.equipment["shunt"][b]], GS] += ref_bus[b - 1, GS]
        bus[index[model.equipment["shunt"][b]], BS] += ref_bus[b - 1, BS]
    branch = ref_branch[:, :13].copy()
    for k in range(NL):
        f, t = int(ref_branch[k, F_BUS]), int(ref_branch[k, T_BUS])
        branch[k, F_BUS] = index[model.terminals[(f, t)]]
        branch[k, T_BUS] = index[model.terminals[(t, f)]]
    slack_bus = int(ref_bus[ref_bus[:, BUS_TYPE] == REF][0, 0])
    slack_node = index[model.equipment["gen"][slack_bus]]
    return {
        "nodes": nodes,
        "index": index,
        "bus": bus,
        "branch": branch,
        "baseMVA": float(reference["baseMVA"]),
        "slack_node": slack_node,
        "cb_names": [cb.name for cb in model.breakers],
        "cb_a": np.array([index[cb.a] for cb in model.breakers], dtype=int),
        "cb_b": np.array([index[cb.b] for cb in model.breakers], dtype=int),
        "metered_nodes": injection_metered_nodes(model, reference),
        "gen_buses": sorted(set(ref_gen[:, GEN_BUS].astype(int))),
    }


def gse_topology_nlm(
    model: FullTopology,
    reference: Mapping[str, Any],
    reported_status: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    max_iterations: int = 30,
    tolerance: float = 1e-7,
    regularization: float = DEFAULT_REGULARIZATION,
) -> dict[str, Any]:
    """Estimate the node/breaker state under ``reported_status`` and rank breakers.

    Returns ``success``, ``iterations``, ``chi_square`` (analog rows), ``dof``
    (analog rows plus independent constraints minus states), ``n_measurements``,
    ``n_constraints``, ``n_states``, ``max_normalized_residual``,
    ``breaker_multipliers`` (per breaker, the normalized multiplier of each of its two
    constraints), ``breaker_scores`` (the larger absolute one), ``ranking`` (breakers
    by score, descending), and the estimated node voltages and breaker flows.
    """
    net = node_network(model, reference)
    tsig = {**TELEMETRY_SIGMA, **dict(telemetry.get("sigma") or {})}
    nodes: Sequence[str] = net["nodes"]
    index: Mapping[str, int] = net["index"]
    n = len(nodes)
    m_cb = len(model.breakers)
    states = model.states(reported_status)
    closed = np.array([states[name] for name in net["cb_names"]], dtype=bool)
    slack = int(net["slack_node"])
    cb_a = net["cb_a"]
    cb_b = net["cb_b"]

    # A closed breaker that closes a loop in the closed-breaker graph adds
    # constraints implied by the others: its multipliers are structurally
    # undefined, and flipping it leaves the connectivity partition unchanged.
    parent = {name: name for name in nodes}

    def _find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    loop_breakers: list[int] = []
    for j, cb in enumerate(model.breakers):
        if not closed[j]:
            continue
        a, b = _find(cb.a), _find(cb.b)
        if a == b:
            loop_breakers.append(j)
        else:
            parent[b] = a

    ybus, yf, yt = make_ybus(net["baseMVA"], net["bus"], net["branch"])
    fbus = net["branch"][:, F_BUS].astype(int)
    tbus = net["branch"][:, T_BUS].astype(int)
    incidence = np.zeros((m_cb, n))
    incidence[np.arange(m_cb), cb_a] += 1.0
    incidence[np.arange(m_cb), cb_b] -= 1.0

    # ------------------------------------------------------------ analog rows
    z_rows: list[float] = []
    sig_rows: list[float] = []
    kinds: list[tuple[str, int]] = []
    live_nodes = [name for name in nodes if float(telemetry["node_vm"].get(name, 0.0)) > 0.0]
    for name in live_nodes:
        z_rows.append(float(telemetry["node_vm"][name]))
        sig_rows.append(tsig["vm"])
        kinds.append(("vm", index[name]))
    metered = [name for name in net["metered_nodes"] if name in telemetry.get("node_pinj", {})]
    for name in metered:
        z_rows.append(float(telemetry["node_pinj"][name]))
        sig_rows.append(tsig["inj"])
        kinds.append(("pinj", index[name]))
    for name in metered:
        z_rows.append(float(telemetry["node_qinj"][name]))
        sig_rows.append(tsig["inj"])
        kinds.append(("qinj", index[name]))
    for key, kind in (("branch_pf", "pf"), ("branch_qf", "qf"), ("branch_pt", "pt"), ("branch_qt", "qt")):
        for k, value in enumerate(telemetry[key]):
            z_rows.append(float(value))
            sig_rows.append(tsig["flow"])
            kinds.append((kind, k))
    for j, name in enumerate(net["cb_names"]):
        z_rows.append(float(telemetry["cb_p"][name]))
        sig_rows.append(tsig["cb"])
        kinds.append(("cb_p", j))
    for j, name in enumerate(net["cb_names"]):
        z_rows.append(float(telemetry["cb_q"][name]))
        sig_rows.append(tsig["cb"])
        kinds.append(("cb_q", j))
    z = np.asarray(z_rows, dtype=float)
    sigma = np.asarray(sig_rows, dtype=float)
    weights = 1.0 / sigma**2
    m = len(z)

    # ----------------------------------------------------------- constraints
    cons: list[tuple[str, int]] = []
    metered_set = set(metered)
    for name in nodes:
        if name not in metered_set:
            cons.append(("zi_p", index[name]))
    for name in nodes:
        if name not in metered_set:
            cons.append(("zi_q", index[name]))
    constraint_rows: dict[str, list[int]] = {name: [] for name in net["cb_names"]}
    for j, name in enumerate(net["cb_names"]):
        for kind in (("c_theta", "c_vm") if closed[j] else ("c_p", "c_q")):
            constraint_rows[name].append(len(cons))
            cons.append((kind, j))
    p = len(cons)

    # State layout: theta (n, slack column dropped), V (n), P_cb (m_cb), Q_cb (m_cb)
    theta_cols = np.r_[np.arange(slack), np.arange(slack + 1, n)]
    theta_pos = {int(node): pos for pos, node in enumerate(theta_cols)}
    off_v = n - 1
    off_p = off_v + n
    off_q = off_p + m_cb
    n_state = off_q + m_cb

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.zeros(n)
        theta[theta_cols] = x[:off_v]
        return theta, x[off_v:off_p], x[off_p:off_q], x[off_q:]

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta, vm, p_cb, q_cb = unpack(x)
        vc = vm * np.exp(1j * theta)
        s_inj = vc * np.conj(ybus @ vc)
        p_inj = s_inj.real + incidence.T @ p_cb
        q_inj = s_inj.imag + incidence.T @ q_cb
        d_va, d_vm = dSbus_dV_polar(ybus, vc)
        dsf_va, dsf_vm, dst_va, dst_vm, s_f, s_t = dSbr_dV1(yf, yt, vc, n, NL, fbus, tbus)
        d_va = d_va.toarray()
        d_vm = d_vm.toarray()
        dsf_va = dsf_va.toarray()
        dsf_vm = dsf_vm.toarray()
        dst_va = dst_va.toarray()
        dst_vm = dst_vm.toarray()

        def injection_rows(kind: str, node: int, target: np.ndarray, row: int) -> None:
            if kind == "p":
                target[row, :off_v] = d_va[node, theta_cols].real
                target[row, off_v:off_p] = d_vm[node, :].real
                target[row, off_p:off_q] = incidence[:, node]
            else:
                target[row, :off_v] = d_va[node, theta_cols].imag
                target[row, off_v:off_p] = d_vm[node, :].imag
                target[row, off_q:] = incidence[:, node]

        h = np.zeros(m)
        H = np.zeros((m, n_state))
        for row, (kind, ref) in enumerate(kinds):
            if kind == "vm":
                h[row] = vm[ref]
                H[row, off_v + ref] = 1.0
            elif kind == "pinj":
                h[row] = p_inj[ref]
                injection_rows("p", ref, H, row)
            elif kind == "qinj":
                h[row] = q_inj[ref]
                injection_rows("q", ref, H, row)
            elif kind == "pf":
                h[row] = s_f[ref].real
                H[row, :off_v] = dsf_va[ref, theta_cols].real
                H[row, off_v:off_p] = dsf_vm[ref, :].real
            elif kind == "qf":
                h[row] = s_f[ref].imag
                H[row, :off_v] = dsf_va[ref, theta_cols].imag
                H[row, off_v:off_p] = dsf_vm[ref, :].imag
            elif kind == "pt":
                h[row] = s_t[ref].real
                H[row, :off_v] = dst_va[ref, theta_cols].real
                H[row, off_v:off_p] = dst_vm[ref, :].real
            elif kind == "qt":
                h[row] = s_t[ref].imag
                H[row, :off_v] = dst_va[ref, theta_cols].imag
                H[row, off_v:off_p] = dst_vm[ref, :].imag
            elif kind == "cb_p":
                h[row] = p_cb[ref]
                H[row, off_p + ref] = 1.0
            elif kind == "cb_q":
                h[row] = q_cb[ref]
                H[row, off_q + ref] = 1.0
        c = np.zeros(p)
        C = np.zeros((p, n_state))
        for row, (kind, ref) in enumerate(cons):
            if kind == "zi_p":
                c[row] = p_inj[ref]
                injection_rows("p", ref, C, row)
            elif kind == "zi_q":
                c[row] = q_inj[ref]
                injection_rows("q", ref, C, row)
            elif kind == "c_theta":
                a, b = int(cb_a[ref]), int(cb_b[ref])
                c[row] = theta[a] - theta[b]
                if a != slack:
                    C[row, theta_pos[a]] += 1.0
                if b != slack:
                    C[row, theta_pos[b]] -= 1.0
            elif kind == "c_vm":
                a, b = int(cb_a[ref]), int(cb_b[ref])
                c[row] = vm[a] - vm[b]
                C[row, off_v + a] += 1.0
                C[row, off_v + b] -= 1.0
            elif kind == "c_p":
                c[row] = p_cb[ref]
                C[row, off_p + ref] = 1.0
            elif kind == "c_q":
                c[row] = q_cb[ref]
                C[row, off_q + ref] = 1.0
        return h, H, c, C

    def kkt(H: np.ndarray, C: np.ndarray) -> np.ndarray:
        K = np.zeros((n_state + p, n_state + p))
        K[:n_state, :n_state] = H.T @ (weights[:, None] * H)
        K[:n_state, n_state:] = C.T
        K[n_state:, :n_state] = C
        K[n_state:, n_state:] = -regularization * np.eye(p)
        return K

    x = np.r_[np.zeros(n - 1), np.ones(n), np.zeros(m_cb), np.zeros(m_cb)]
    success = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        h, H, c, C = evaluate(x)
        rhs = np.r_[H.T @ (weights * (z - h)), -c]
        try:
            step = np.linalg.solve(kkt(H, C), rhs)
        except np.linalg.LinAlgError:
            break
        dx = step[:n_state]
        x = x + dx
        if not np.all(np.isfinite(x)):
            break
        if np.max(np.abs(dx)) < tolerance:
            success = True
            break
    failure = {
        "success": False,
        "iterations": iterations,
        "chi_square": float("nan"),
        "dof": int(m + p - n_state),
        "n_measurements": int(m),
        "n_constraints": int(p),
        "n_states": int(n_state),
        "breaker_multipliers": {},
        "breaker_scores": {},
        "ranking": [],
    }
    if not success:
        return failure
    h, H, c, C = evaluate(x)
    K = kkt(H, C)
    try:
        k_inv = np.linalg.inv(K)
        step = np.linalg.solve(K, np.r_[H.T @ (weights * (z - h)), -c])
    except np.linalg.LinAlgError:
        return failure
    residual = z - h
    sigma_x = k_inv[:n_state, :n_state]
    omega = np.clip(sigma**2 - np.einsum("ij,jk,ik->i", H, sigma_x, H), np.finfo(float).eps, None)
    normalized = residual / np.sqrt(omega)
    lam = step[n_state:]
    cov_lam = -np.diag(k_inv[n_state:, n_state:])
    structurally_dependent = np.zeros(p, dtype=bool)
    for j in loop_breakers:
        for row in constraint_rows[net["cb_names"][j]]:
            structurally_dependent[row] = True
    unidentifiable = structurally_dependent | (cov_lam > _UNIDENTIFIABLE_FRACTION / regularization)
    identifiable = ~unidentifiable & (cov_lam > 0)
    lam_n = np.zeros(p)
    lam_n[identifiable] = lam[identifiable] / np.sqrt(cov_lam[identifiable])
    chi_square = float(np.sum(weights * residual**2))
    theta, vm, p_cb, q_cb = unpack(x)

    multipliers: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    for j, name in enumerate(net["cb_names"]):
        rows = constraint_rows[name]
        labels = ("theta", "vm") if closed[j] else ("p", "q")
        multipliers[name] = {label: float(lam_n[row]) for label, row in zip(labels, rows)}
        scores[name] = float(max(abs(lam_n[row]) for row in rows))
    ranking = rank_breakers(multipliers)
    return {
        "success": True,
        "iterations": iterations,
        "chi_square": chi_square,
        "dof": int(m + int(np.sum(~unidentifiable)) - n_state),
        "n_measurements": int(m),
        "n_constraints": int(p),
        "n_dependent_constraints": int(np.sum(unidentifiable)),
        "loop_dependent_breakers": [net["cb_names"][j] for j in loop_breakers],
        "n_states": int(n_state),
        "max_normalized_residual": float(np.max(np.abs(normalized))),
        "breaker_multipliers": multipliers,
        "breaker_scores": scores,
        "ranking": [
            {
                "cb_name": name,
                "score": scores[name],
                "multipliers": multipliers[name],
                "reported_status": ("closed" if states[name] else "open"),
            }
            for name in ranking
        ],
        "node_vm_est": {name: float(vm[index[name]]) for name in nodes},
        "node_theta_est": {name: float(theta[index[name]]) for name in nodes},
        "cb_flow_est": {name: [float(p_cb[j]), float(q_cb[j])] for j, name in enumerate(net["cb_names"])},
    }
