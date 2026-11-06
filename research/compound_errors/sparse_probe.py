"""Separability probe for sparse multi-error scenarios with cluster decomposition (2026-10-28).

    python -m research.compound_errors.sparse_probe \\
        --suite output/compound_errors_20261027/suite.json \\
        --ieee118-dir output/classifier_triage_20261012/ieee118_effect \\
        --output-dir output/sparse_multi_error_probe_20261028 --workers 8

The design of variable-count sparse error scenarios (``docs/sparse_multi_error_design_20261028.md``) rests on
one assumption: errors are sparse and mostly localized, so the evidence of one WLS solve splits into clusters
that can be checked and corrected one at a time, or side by side.  This probe measures that assumption on data
that exists, with the deployed estimator's own linearization and nothing else:

* the WLS is the deployed Lagrangian solve (``tools.lagrangian_port.lagrangian_m_singlephase_details``, the
  solve ``psse_env.providers.matpower`` runs through ``_wls_json``); its Jacobian, residual covariance and
  exact structural rows give the whitened residual projector ``S = R^-1/2 Omega R^-1/2`` (rank dof), and the
  shared branch R/X Jacobian (``tools.branch_param_jacobian``) gives every line's normalized-multiplier
  directions, so the multiplier covariance of any two hypotheses (meters and lines) is a product of columns;
* an atom (a gross meter error, a line R/X error, an open line terminal) is a deviation ``d`` of the measured
  vector from the reported model's prediction at the true state; its expected whitened residual is
  ``u = S R^-1/2 d``, its noiseless chi-square ``|u|^2``, its expected normalized residuals and multipliers
  follow, and atoms add (at a fixed state the measurement function is additive over branches); the
  ``suite14_natural`` records check the linear chi-square against the recorded nonlinear solo effects;
* noise goes through the same projector, so alarms, flags and clusters are those of one noisy solve; a
  correction removes its atom's deviation and absorbs what its hypothesis can explain of the rest (a meter
  correction projects out the meter's direction, a line correction its R/X tangent, a breaker flip removes
  the deviation exactly and absorbs nothing).

Atoms come from the IEEE 14 compound suite (meters with their offsets, lines with their R/X factors, dangling
terminals as line outages; bus splits are left out, a re-rendered layout is not a branch deviation) and from
the IEEE 118 effect-targeted single-error roots (the meters and lines of the ``measurement``,
``multi_measurement`` and ``parameter`` families).  IEEE 118 has no node/breaker model, so its outage atoms are
synthesized at its recorded operating points on every non-radial line.  The operating points are recorded
ones (IEEE 14 control-meter roots, IEEE 118 healthy corpus samples).

Measured, per system (the design note's tables cite the keys of ``summary.json``):

* (a) footprints: the meters (|normalized residual| >= 4, the deployed alarm level) and lines (|normalized
  multiplier| >= 3, the deployed lambda threshold) an atom's expected evidence flags (``footprints``);
* (b) footprint overlap, residual cosine, own-hypothesis correlation and the cross-cluster block of the
  multiplier covariance against the bus-graph distance of two atoms (``sets.pairs_by_distance``), and the
  correlation decay of the whole hypothesis set (``curves.correlation_decay``);
* (c) how often an atom's own effect is below the global alarm (chi-square at alpha 0.01 or a normalized
  residual of 4) alone, with the others present (its own evidence unflagged) and after the others are
  corrected; (d) what a cluster-local chi-square (the cluster's measurements, its own degrees of freedom,
  a Satterthwaite threshold at alpha 0.01) flags in those cases;
* the separability criterion (``separable``) on random pairs and triples at matched effect ratios
  (log-uniform in the effect-targeted band 1.5 to 20), on variable-count draws at a density per hundred
  branches, and the progress the true correction and the best wrong line and meter corrections would show
  under the global floor (J basis and excess basis) and under a cluster-local floor;
* the IEEE 14 suite's own roots at their natural magnitudes and states (``suite14_natural``).

The separability criterion, on the noiseless expected evidence: (S1) every atom's own element is flagged
with the others present, (S2) the atoms' own elements fall in distinct clusters and no cluster holds flags
owned by two atoms (a flag belongs to the atom whose expected evidence on it is largest), and (S3) correcting
the other atoms keeps at least 90% of each atom's own chi-square energy.  The noisy reading
(``noisy_partition_recovered``) is S1 and S2 on the flags of noisy solves.

Research only.  CPU only.  No provider, oracle, verifier or audit code is changed; only the solve is called.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server.matpower_server import _load_python_case  # noqa: E402
from tools import lagrangian_port as lp  # noqa: E402
from tools.branch_param_jacobian import _hx_internal, all_branch_rx_jacobian, branch_rx_jacobian  # noqa: E402

CONTRACT = "sparse_multi_error_probe_v1"
CHI2_ALPHA = 0.01
RESIDUAL_FLAG = 4.0          # the deployed normalized-residual alarm level
LAMBDA_FLAG = 3.0            # the deployed branch-multiplier threshold (lambda_threshold)
_CONFIG = {"lambda_flag": LAMBDA_FLAG}   # the run's line-flag level (``--lambda-flag``), set in every worker
LINK_DISTANCE = 1            # flags whose bus sets are within one branch share a cluster
CLOSURE_HOPS = 1             # a cluster's measurement set: its buses closed by one hop
BAND = (1.5, 20.0)           # the effect-targeted band (research/classifier_triage/effect_sampler.py)
MASKING_KEEP = 0.9           # separable only if correcting the others keeps 90% of an atom's own energy
BRANCH_FLOOR, METER_FLOOR = 0.30, 0.20   # the verifier's partial-progress floors
CHANNELS = ("Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt")
KINDS = ("meter", "parameter", "outage")
DISTANCE_BINS = (0, 1, 2, 3, 4, 5, 6)    # 6 reads "6 or more"
SWEEP_RATIOS = (0.3, 0.6, 1.0, 1.5, 3.0, 10.0, 20.0)


# --------------------------------------------------------------------------- the deployed solve, linearized

def _as_case(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (float(case[key]) if key == "baseMVA" else np.array(case[key], dtype=float))
            for key in ("baseMVA", "bus", "branch", "gen")}


def _h(case: Mapping[str, Any], theta: np.ndarray, vm: np.ndarray) -> np.ndarray:
    return np.asarray(_hx_internal(lp._copy_result_to_internal(case), theta, vm), dtype=float)


def _solve(case: Mapping[str, Any], z: Sequence[float], sigma: Sequence[float], exact: Sequence[int]) -> dict[str, Any]:
    return lp.lagrangian_m_singlephase_details(
        z=np.asarray(z, dtype=float), result=case, ind=0, bus_data=np.asarray(case["bus"], dtype=float),
        measurement_sigma=np.asarray(sigma, dtype=float), exact_measurement_indices=list(exact) or None)


def channel_of(index: int, nb: int, nl: int) -> tuple[str, int]:
    """(channel, bus or line) of a full measurement-vector index ``[Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]``."""
    if index < 3 * nb:
        return CHANNELS[index // nb], index % nb
    return CHANNELS[3 + (index - 3 * nb) // nl], (index - 3 * nb) % nl


class Linearization:
    """The reported model's WLS at one state: whitened residual projector and hypothesis directions."""

    def __init__(self, case: Mapping[str, Any], theta: np.ndarray, vm: np.ndarray, sigma: Sequence[float],
                 exact: Sequence[int], *, bank: int = 0, seed: int = 0):
        self.case = case
        self.nb, self.nl = int(case["bus"].shape[0]), int(case["branch"].shape[0])
        z = _h(case, theta, vm)
        exact = [int(i) for i in exact]
        if exact:
            z[exact] = 0.0
        det = _solve(case, z, sigma, exact)
        if not det["success"]:
            raise RuntimeError("linearization solve did not converge")
        self.theta = np.asarray(det["theta_est_rad"], dtype=float)
        self.vm = np.asarray(det["vm_est_pu"], dtype=float)
        self.rows = np.asarray(det["measurement_rows"], dtype=int)
        self.exact = np.asarray(det["exact_measurement_indices"], dtype=int)
        self.m = int(self.rows.size)
        self.Hu = np.asarray(det["measurement_jacobian"], dtype=float)
        self.wh = 1.0 / np.sqrt(np.asarray(det["measurement_variance_diag"], dtype=float))
        omega = np.asarray(det["residual_covariance"], dtype=float)
        shat = (self.wh[:, None] * omega) * self.wh[None, :]
        self.S = 0.5 * (shat + shat.T)
        self.dof = int(det["dof"])
        self.threshold = float(chi2.ppf(1.0 - CHI2_ALPHA, self.dof))
        self.excess = self.threshold - self.dof
        C = np.asarray(det["constraint_jacobian"], dtype=float)
        self.Cpinv = np.linalg.pinv(C) if C.size else None
        self.case_int = lp._copy_result_to_internal(case)
        self.Upar = self.directions(all_branch_rx_jacobian(self.case_int, self.theta, self.vm))
        self.upar_norm = np.linalg.norm(self.Upar, axis=0)
        self._upar_div = np.where(self.upar_norm > 1e-9, self.upar_norm, np.inf)
        self.umet_norm = np.sqrt(np.clip(np.diag(self.S), 0.0, None))
        self._umet_div = np.where(self.umet_norm > 1e-9, self.umet_norm, np.inf)
        self.h0 = _h(case, self.theta, self.vm)
        self.row_of_index = {int(i): j for j, i in enumerate(self.rows)}
        self.row_channel = [channel_of(int(i), self.nb, self.nl) for i in self.rows]
        self.bank = None
        if bank:
            rng = np.random.default_rng(seed)
            self.bank = self.S @ rng.standard_normal((self.m, bank))

    def directions(self, columns_full: np.ndarray) -> np.ndarray:
        """Whitened projected directions of full-vector deviation columns (exact rows folded in)."""
        A = columns_full[self.rows]
        if self.Cpinv is not None:
            A = A - self.Hu @ (self.Cpinv @ columns_full[self.exact])
        return self.S @ (self.wh[:, None] * A)

    def u_of(self, d_full: np.ndarray) -> np.ndarray:
        return self.directions(np.asarray(d_full, dtype=float)[:, None])[:, 0]

    def residual_scores(self, w: np.ndarray) -> np.ndarray:
        """Normalized residuals (signed) of whitened residual vector(s)."""
        return w / (self._umet_div[:, None] if w.ndim == 2 else self._umet_div)

    def line_scores(self, w: np.ndarray) -> np.ndarray:
        """max(|lambdaN_R|, |lambdaN_X|) per line of whitened residual vector(s)."""
        w2 = w if w.ndim == 2 else w[:, None]
        lam = np.abs((self.Upar.T @ w2) / self._upar_div[:, None]).reshape(self.nl, 2, -1).max(axis=1)
        return lam if w.ndim == 2 else lam[:, 0]

    def line_tangent(self, k: int, r: float, x: float) -> np.ndarray:
        """Whitened projected R/X directions of line ``k`` with its R/X set to ``(r, x)``."""
        case_int = copy.deepcopy(self.case_int)
        case_int["branch"][k, lp.BR_R], case_int["branch"][k, lp.BR_X] = float(r), float(x)
        return self.directions(branch_rx_jacobian(case_int, k, self.theta, self.vm))

    def chi_square(self, w: np.ndarray) -> np.ndarray:
        w2 = w if w.ndim == 2 else w[:, None]
        return np.einsum("ij,ij->j", w2, w2)

    def global_alarm(self, w: np.ndarray) -> np.ndarray:
        """The deployed alarm: chi-square at alpha or a normalized residual at the residual level."""
        w2 = w if w.ndim == 2 else w[:, None]
        alarm = (self.chi_square(w2) >= self.threshold) | (np.abs(self.residual_scores(w2)).max(axis=0) >= RESIDUAL_FLAG)
        return alarm if w.ndim == 2 else alarm[0]


# --------------------------------------------------------------------------- the bus graph

class Geometry:
    def __init__(self, lin: Linearization):
        self.nb, self.nl = lin.nb, lin.nl
        ci = lin.case_int
        self.f = ci["branch"][:, lp.F_BUS].astype(int)
        self.t = ci["branch"][:, lp.T_BUS].astype(int)
        adj = [set() for _ in range(self.nb)]
        for f, t in zip(self.f, self.t):
            adj[f].add(int(t))
            adj[t].add(int(f))
        self.degree = np.array([len(a) for a in adj])
        D = np.full((self.nb, self.nb), 10 ** 6, dtype=int)
        for s in range(self.nb):
            D[s, s] = 0
            frontier = [s]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in adj[u]:
                        if D[s, v] > D[s, u] + 1:
                            D[s, v] = D[s, u] + 1
                            nxt.append(v)
                frontier = nxt
        self.D = D
        self.line_buses = [(int(f), int(t)) for f, t in zip(self.f, self.t)]
        self.row_buses = [(idx,) if channel in ("Vm", "Pinj", "Qinj") else self.line_buses[idx]
                          for channel, idx in lin.row_channel]
        self.row_b1 = np.array([b[0] for b in self.row_buses])
        self.row_b2 = np.array([b[-1] for b in self.row_buses])
        self.flow_rows_of_line: dict[int, list[int]] = defaultdict(list)
        for j, (channel, idx) in enumerate(lin.row_channel):
            if channel in ("Pf", "Qf", "Pt", "Qt"):
                self.flow_rows_of_line[idx].append(j)
        self.pq_rows = [j for j, (channel, _) in enumerate(lin.row_channel) if channel != "Vm"]
        # A line whose removal islands a bus (a radial line) cannot be an outage atom.
        self.radial = {k for k, (f, t) in enumerate(self.line_buses) if self.degree[f] == 1 or self.degree[t] == 1}

    def distance(self, a: Iterable[int], b: Iterable[int]) -> int:
        return int(self.D[np.ix_(list(a), list(b))].min())

    def closed(self, buses: Iterable[int], hops: int) -> set[int]:
        buses = list(buses)
        return set(np.flatnonzero(self.D[buses].min(axis=0) <= hops).tolist()) if buses else set()

    def measurement_set(self, buses: set[int]) -> np.ndarray:
        """Bus meters at the buses and flows of lines with both terminals among them."""
        inside = np.zeros(self.nb, dtype=bool)
        inside[list(buses)] = True
        return np.flatnonzero(inside[self.row_b1] & inside[self.row_b2])


# --------------------------------------------------------------------------- atoms

@dataclass
class Atom:
    kind: str
    element: int                 # used-row index for a meter, branch row for a line
    buses: tuple[int, ...]
    u: np.ndarray                # expected whitened residual
    tangent: np.ndarray | None   # orthonormal basis of what its correction fits; None for a status flip
    natural_ratio: float
    ratio: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def energy(self) -> float:
        return float(self.u @ self.u)

    @property
    def hypothesis(self) -> tuple[str, int]:
        return ("m" if self.kind == "meter" else "l", int(self.element))

    def scaled(self, ratio: float) -> "Atom":
        return Atom(self.kind, self.element, self.buses, self.u * math.sqrt(ratio / self.ratio), self.tangent,
                    self.natural_ratio, ratio, dict(self.meta))


def _orth(T: np.ndarray | None) -> np.ndarray | None:
    if T is None or T.size == 0:
        return None
    q, r = np.linalg.qr(T)
    keep = np.abs(np.diag(r)) > 1e-9 * max(1.0, float(np.abs(r).max()))
    return q[:, keep] if keep.any() else None


def meter_atom(lin: Linearization, geo: Geometry, index: int, offset: float, *, ratio: float | None) -> Atom | None:
    j = lin.row_of_index.get(int(index))
    if j is None:
        return None
    d = np.zeros(lin.nb * 3 + lin.nl * 4)
    d[int(index)] = float(offset)
    u = lin.u_of(d)
    natural = float(u @ u) / lin.excess
    if natural <= 1e-12:
        return None
    atom = Atom("meter", j, geo.row_buses[j], u, _orth(lin.S[:, [j]]), natural, natural,
                {"index": int(index), "offset": float(offset)})
    return atom if ratio is None else atom.scaled(ratio)


def _deviation(lin: Linearization, case: Mapping[str, Any], state: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """The measured-minus-predicted vector of a changed network, at the true state (the linearization's by default).

    At a state of the reported model an open line into a zero-injection bus breaks that bus's exact
    constraint, which a measured vector never does; such an atom is read at its own root's true state.
    """
    theta, vm = (lin.theta, lin.vm) if state is None else state
    return _h(case, theta, vm) - _h(lin.case, theta, vm)


def parameter_atom(lin: Linearization, geo: Geometry, k: int, r_factor: float, x_factor: float, *,
                   ratio: float | None, exact: bool = False,
                   state: tuple[np.ndarray, np.ndarray] | None = None) -> Atom | None:
    """A line R/X error: matched atoms are first order (the reported tangent), natural ones exact."""
    r0, x0 = float(lin.case["branch"][k, lp.BR_R]), float(lin.case["branch"][k, lp.BR_X])
    if exact:
        case = copy.deepcopy(lin.case)
        case["branch"][k, lp.BR_R], case["branch"][k, lp.BR_X] = r0 * r_factor, x0 * x_factor
        u = lin.u_of(_deviation(lin, case, state))
        tangent = lin.line_tangent(k, r0 * r_factor, x0 * x_factor)
    else:
        tangent = lin.Upar[:, 2 * k:2 * k + 2]
        u = tangent @ np.array([r0 * (r_factor - 1.0), x0 * (x_factor - 1.0)])
    natural = float(u @ u) / lin.excess
    if natural <= 1e-12:
        return None
    atom = Atom("parameter", int(k), geo.line_buses[k], u, _orth(tangent), natural, natural,
                {"r_factor": float(r_factor), "x_factor": float(x_factor), "exact": bool(exact)})
    return atom if ratio is None else atom.scaled(ratio)


def outage_atom(lin: Linearization, geo: Geometry, k: int, *,
                state: tuple[np.ndarray, np.ndarray] | None = None) -> Atom | None:
    """An open terminal read on the bus-branch model: the line out at the true state (natural magnitude)."""
    if k in geo.radial:
        return None
    case = copy.deepcopy(lin.case)
    case["branch"][k, lp.BR_STATUS] = 0.0
    u = lin.u_of(_deviation(lin, case, state))
    natural = float(u @ u) / lin.excess
    if natural <= 1e-12:
        return None
    return Atom("outage", int(k), geo.line_buses[k], u, None, natural, natural, {})


def hypothesis_atom(lin: Linearization, geo: Geometry, element: tuple[str, int]) -> Atom:
    """A wrong correction on an element: a fit of that element's hypothesis, no true deviation removed."""
    kind, idx = element
    zero = np.zeros(lin.m)
    if kind == "m":
        return Atom("meter", idx, geo.row_buses[idx], zero, _orth(lin.S[:, [idx]]), 0.0, 0.0)
    return Atom("parameter", idx, geo.line_buses[idx], zero, _orth(lin.Upar[:, 2 * idx:2 * idx + 2]), 0.0, 0.0)


def own_evidence(atom: Atom, geo: Geometry) -> list[tuple[str, int]]:
    """The elements whose flag shows the atom itself: its meter, or its line and the line's flow meters."""
    if atom.kind == "meter":
        return [("m", atom.element)]
    return [("l", atom.element)] + [("m", j) for j in geo.flow_rows_of_line.get(atom.element, [])]


def fix(w: np.ndarray, atom: Atom, *, remove: bool = True) -> np.ndarray:
    """Correct ``atom``: remove its deviation (a true correction) and absorb what its hypothesis explains."""
    out = (w - (atom.u[:, None] if w.ndim == 2 else atom.u)) if remove else w.copy()
    if atom.tangent is not None:
        out = out - atom.tangent @ (atom.tangent.T @ out)
    return out


# --------------------------------------------------------------------------- flags and clusters

def flags_of(rscore: np.ndarray, lscore: np.ndarray, *, residual_flag: float = RESIDUAL_FLAG,
             lambda_flag: float | None = None) -> list[tuple[str, int]]:
    out = [("m", int(j)) for j in np.flatnonzero(np.abs(rscore) >= residual_flag)]
    lam = _CONFIG["lambda_flag"] if lambda_flag is None else lambda_flag
    return out + [("l", int(k)) for k in np.flatnonzero(lscore >= lam)]


def element_buses(geo: Geometry, element: tuple[str, int]) -> tuple[int, ...]:
    return geo.row_buses[element[1]] if element[0] == "m" else geo.line_buses[element[1]]


def clusters_of(geo: Geometry, flags: Sequence[tuple[str, int]], link: int) -> np.ndarray:
    """Connected components of the flags; two flags are linked when their bus sets are within ``link``."""
    if not flags:
        return np.zeros(0, dtype=int)
    b1 = np.array([element_buses(geo, e)[0] for e in flags])
    b2 = np.array([element_buses(geo, e)[-1] for e in flags])
    D = np.minimum(np.minimum(geo.D[np.ix_(b1, b1)], geo.D[np.ix_(b1, b2)]),
                   np.minimum(geo.D[np.ix_(b2, b1)], geo.D[np.ix_(b2, b2)]))
    _, labels = connected_components(csr_matrix(D <= link), directed=False)
    return labels


def satterthwaite_threshold(lin: Linearization, rows: np.ndarray) -> tuple[float, float, float]:
    """(threshold, mean, equivalent dof) of the local statistic, the whitened residual energy on ``rows``."""
    if rows.size == 0:
        return math.inf, 0.0, 0.0
    block = lin.S[np.ix_(rows, rows)]
    t1, t2 = float(np.trace(block)), float(np.sum(block * block))
    if t1 <= 1e-9:
        return math.inf, t1, 0.0
    g, h = t2 / t1, t1 * t1 / t2
    return float(g * chi2.ppf(1.0 - CHI2_ALPHA, h)), t1, h


class Evidence:
    """Each atom's expected normalized residuals and line scores, computed once per combination."""

    def __init__(self, lin: Linearization, atoms: Sequence[Atom]):
        self.r = [lin.residual_scores(a.u) for a in atoms]
        self.l = [lin.line_scores(a.u) for a in atoms]

    def owner(self, flags: Sequence[tuple[str, int]]) -> list[int]:
        """The atom whose expected evidence on a flag is largest (-1: none reaches one sigma)."""
        out = []
        for kind, idx in flags:
            contrib = [abs(r[idx]) if kind == "m" else l[idx] for r, l in zip(self.r, self.l)]
            best = int(np.argmax(contrib))
            out.append(best if contrib[best] >= 1.0 else -1)
        return out


def partition(geo: Geometry, atoms: Sequence[Atom], ev: Evidence, rscore: np.ndarray, lscore: np.ndarray, *,
              residual_flag: float = RESIDUAL_FLAG, lambda_flag: float | None = None, link: int = LINK_DISTANCE
              ) -> dict[str, Any]:
    flags = flags_of(rscore, lscore, residual_flag=residual_flag, lambda_flag=lambda_flag)
    labels = clusters_of(geo, flags, link)
    owner = ev.owner(flags)
    index = {e: i for i, e in enumerate(flags)}
    atom_cluster, visible = [], []
    for a in atoms:
        hit = [index[e] for e in own_evidence(a, geo) if e in index]
        visible.append(bool(hit))
        if hit:
            score = [abs(rscore[flags[i][1]]) if flags[i][0] == "m" else lscore[flags[i][1]] for i in hit]
            atom_cluster.append(int(labels[hit[int(np.argmax(score))]]))
        else:
            atom_cluster.append(None)
    owners: dict[int, set[int]] = defaultdict(set)
    for lab, o in zip(labels, owner):
        if o >= 0:
            owners[int(lab)].add(o)
    pure = all(len(v) <= 1 for v in owners.values())
    placed = [c for c in atom_cluster if c is not None]
    distinct = len(placed) == len(atoms) and len(set(placed)) == len(placed)
    alone = [c is not None and owners.get(c, set()) <= {i} and atom_cluster.count(c) == 1
             for i, c in enumerate(atom_cluster)]
    owned = {int(lab) for lab, o in zip(labels, owner) if o >= 0}
    return {"flags": flags, "labels": labels, "owner": owner, "atom_cluster": atom_cluster, "visible": visible,
            "pure": pure, "distinct": distinct, "alone": alone, "n_clusters": int(len(set(labels.tolist()))),
            "spurious_clusters": int(len(set(labels.tolist()) - owned))}


def local_set(lin: Linearization, geo: Geometry, part: Mapping[str, Any] | None, cluster: int | None,
              atom: Atom) -> dict[str, Any]:
    """The measurement set of a cluster (its flagged buses closed by one hop), else of the atom's buses."""
    if part is None or cluster is None:
        buses = geo.closed(atom.buses, CLOSURE_HOPS)
    else:
        buses = geo.closed({b for e, lab in zip(part["flags"], part["labels"]) if lab == cluster
                            for b in element_buses(geo, e)}, CLOSURE_HOPS)
    rows = geo.measurement_set(buses)
    thr, dof, eq = satterthwaite_threshold(lin, rows)
    return {"buses": buses, "rows": rows, "threshold": thr, "dof": dof, "eq_dof": eq}


# --------------------------------------------------------------------------- one combination

def masking_keep(atoms: Sequence[Atom], i: int) -> float:
    """Share of atom ``i``'s noiseless energy left after the others are corrected, strongest first."""
    w = atoms[i].u.copy()
    for j in sorted((j for j in range(len(atoms)) if j != i), key=lambda j: -atoms[j].energy):
        w = fix(w, atoms[j], remove=False)
    return float(w @ w) / atoms[i].energy


def _hyp_vectors(lin: Linearization, flags: Sequence[tuple[str, int]]) -> list[np.ndarray]:
    out = []
    for kind, idx in flags:
        if kind == "m":
            if lin.umet_norm[idx] > 1e-9:
                out.append(lin.S[:, idx] / lin.umet_norm[idx])
        else:
            out += [lin.Upar[:, c] / lin.upar_norm[c] for c in (2 * idx, 2 * idx + 1) if lin.upar_norm[c] > 1e-9]
    return out


def evaluate(lin: Linearization, geo: Geometry, atoms: list[Atom], *, draws: int, rng: np.random.Generator,
             sensitivity: bool = False) -> dict[str, Any]:
    n = len(atoms)
    ev = Evidence(lin, atoms)
    u_tot = np.sum([a.u for a in atoms], axis=0)
    rs_tot, ls_tot = lin.residual_scores(u_tot), lin.line_scores(u_tot)
    nf = partition(geo, atoms, ev, rs_tot, ls_tot)
    keep = [masking_keep(atoms, i) for i in range(n)] if n > 1 else [1.0]
    rec: dict[str, Any] = {
        "n_atoms": n, "kinds": [a.kind for a in atoms], "elements": [int(a.element) for a in atoms],
        "buses": [list(a.buses) for a in atoms], "ratio": [round(a.ratio, 4) for a in atoms],
        "natural_ratio": [round(a.natural_ratio, 4) for a in atoms],
        "max_degree": int(max(geo.degree[b] for a in atoms for b in a.buses)),
        "noiseless": {k: nf[k] for k in ("visible", "pure", "distinct", "alone", "n_clusters")},
        "masking_keep": [round(k, 4) for k in keep],
    }
    rec["noiseless"]["n_flags"] = len(nf["flags"])
    foot = []
    for i, a in enumerate(atoms):
        fl = flags_of(ev.r[i], ev.l[i])
        fb = {b for e in fl for b in element_buses(geo, e)}
        foot.append({"flags": fl, "buses": fb,
                     "radius": max((geo.distance(a.buses, element_buses(geo, e)) for e in fl), default=None)})
    rec["footprint"] = [{"n_flags": len(f["flags"]), "n_buses": len(f["buses"]), "radius": f["radius"]} for f in foot]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = atoms[i], atoms[j]
            fa, fb = foot[i]["buses"], foot[j]["buses"]
            Qa = a.tangent if a.tangent is not None else _orth(a.u[:, None])
            Qb = b.tangent if b.tangent is not None else _orth(b.u[:, None])
            # The cross-cluster block: hypotheses of one footprint against those of the other, shared ones
            # left out (a shared flag is overlap, correlated with itself).
            shared = set(foot[i]["flags"]) & set(foot[j]["flags"])
            ha = _hyp_vectors(lin, [e for e in foot[i]["flags"] if e not in shared])
            hb = _hyp_vectors(lin, [e for e in foot[j]["flags"] if e not in shared])
            pairs.append({
                "atoms": [i, j], "distance": geo.distance(a.buses, b.buses),
                "overlap": round(len(fa & fb) / max(1, min(len(fa), len(fb))), 4) if fa and fb else 0.0,
                "cosine": round(float(a.u @ b.u) / math.sqrt(a.energy * b.energy), 4),
                "own_corr": round(float(np.linalg.svd(Qa.T @ Qb, compute_uv=False).max()), 4),
                "cross_block_max": round(float(np.abs(np.array(ha) @ np.array(hb).T).max()), 4) if ha and hb else None,
            })
    rec["pairs"] = pairs
    rec["separable"] = bool(all(nf["visible"]) and nf["distinct"] and nf["pure"] and min(keep) >= MASKING_KEEP) if n > 1 else None
    rec["atom_isolated"] = [bool(nf["visible"][i] and nf["alone"][i] and keep[i] >= MASKING_KEEP) for i in range(n)]
    if sensitivity and n > 1:
        sens = {}
        for name, kw in (("link0", {"link": 0}), ("link2", {"link": 2}), ("lambda4", {"lambda_flag": 4.0}),
                         ("residual3", {"residual_flag": 3.0})):
            p = partition(geo, atoms, ev, rs_tot, ls_tot, **kw)
            sens[name] = bool(all(p["visible"]) and p["distinct"] and p["pure"] and min(keep) >= MASKING_KEEP)
        rec["separable_sensitivity"] = sens
    # The deployed context ranks lines globally; a cluster ranks its own.  Rank of each line atom's line.
    rec["own_line_rank_global"] = [_rank(ls_tot, range(lin.nl), a.element) if a.kind != "meter" else None for a in atoms]
    rec["own_line_rank_cluster"] = _cluster_ranks(geo, atoms, nf, ls_tot)
    local = [local_set(lin, geo, nf, nf["atom_cluster"][i], a) for i, a in enumerate(atoms)]
    rec["local"] = [{"rows": int(l["rows"].size), "dof": round(l["dof"], 2), "eq_dof": round(l["eq_dof"], 2),
                     "threshold": round(l["threshold"], 3) if math.isfinite(l["threshold"]) else None,
                     "concentration": round(float(a.u[l["rows"]] @ a.u[l["rows"]]) / a.energy, 4) if l["rows"].size else 0.0}
                    for a, l in zip(atoms, local)]
    rec["peel"] = peel(lin, geo, atoms)
    if draws:
        rec["noisy"] = _noisy(lin, geo, atoms, ev, u_tot, local, draws, rng, nf["alone"])
    return rec


def peel(lin: Linearization, geo: Geometry, atoms: Sequence[Atom]) -> dict[str, Any]:
    """Sequential decomposition on the noiseless evidence, one solve per round.

    Every round re-partitions what remains; an atom alone in its cluster is corrected in that round (side by
    side with the other lone atoms), a cluster holding several atoms corrects its strongest one; an atom
    whose own evidence is not flagged waits.  Rounds against the atom count is the concurrency on offer;
    atoms left unflagged when nothing else remains are the ones the tool would never see.
    """
    remaining = list(range(len(atoms)))
    rounds, alone_fixes, widths = 0, 0, []
    while remaining:
        sub = [atoms[i] for i in remaining]
        ev = Evidence(lin, sub)
        u = np.sum([a.u for a in sub], axis=0)
        p = partition(geo, sub, ev, lin.residual_scores(u), lin.line_scores(u))
        groups: dict[int, list[int]] = defaultdict(list)
        for pos, c in enumerate(p["atom_cluster"]):
            if c is not None:
                groups[c].append(pos)
        if not groups:
            break
        chosen = []
        for c, members in groups.items():
            if len(members) == 1 and p["alone"][members[0]]:
                chosen.append(members[0])
                alone_fixes += 1
            else:
                chosen.append(max(members, key=lambda pos: sub[pos].energy))
        widths.append(len(chosen))
        remaining = [remaining[pos] for pos in range(len(sub)) if pos not in set(chosen)]
        rounds += 1
    return {"rounds": rounds, "fixed_alone": alone_fixes, "never_flagged": len(remaining),
            "max_width": max(widths, default=0)}


def _rank(scores: np.ndarray, members: Iterable[int], element: int) -> int | None:
    """1-based rank of ``element`` among ``members`` by score (None when it is not a member)."""
    members = list(members)
    if element not in members:
        return None
    return 1 + sum(1 for k in members if scores[k] > scores[element])


def _cluster_ranks(geo, atoms, part, ls) -> list[int | None]:
    """Rank of a line atom's line among the lines of its cluster (lines with both terminals in it)."""
    out = []
    for i, a in enumerate(atoms):
        c = part["atom_cluster"][i]
        if a.kind == "meter" or c is None:
            out.append(None)
            continue
        buses = {b for e, lab in zip(part["flags"], part["labels"]) if lab == c for b in element_buses(geo, e)}
        members = [k for k in range(geo.nl) if geo.f[k] in buses and geo.t[k] in buses]
        out.append(_rank(ls, members, a.element))
    return out


def _progress(J0: np.ndarray, J1: np.ndarray, base: float) -> np.ndarray:
    den = J0 - base
    return np.where(den > 0, (J0 - J1) / np.where(den > 0, den, 1.0), np.nan)


def _noisy(lin, geo, atoms, ev, u_tot, local, draws, rng, noiseless_alone) -> dict[str, Any]:
    n = len(atoms)
    E = lin.bank[:, rng.choice(lin.bank.shape[1], size=draws, replace=False)]
    W = u_tot[:, None] + E
    rs_all, ls_all = lin.residual_scores(W), lin.line_scores(W)
    parts = [partition(geo, atoms, ev, rs_all[:, k], ls_all[:, k]) for k in range(draws)]
    out: dict[str, Any] = {
        "alarm_all": float(lin.global_alarm(W).mean()),
        "partition_recovered": float(np.mean([all(p["visible"]) and p["distinct"] and p["pure"] for p in parts])) if n > 1 else None,
        "visible_with_others": [float(np.mean([p["visible"][i] for p in parts])) for i in range(n)],
        "alone": [float(np.mean([p["alone"][i] for p in parts])) for i in range(n)],
        "spurious_clusters_mean": float(np.mean([p["spurious_clusters"] for p in parts])),
        "n_clusters_mean": float(np.mean([p["n_clusters"] for p in parts])),
        "own_alarm_alone": [float(lin.global_alarm(a.u[:, None] + E).mean()) for a in atoms],
    }
    if n > 1:
        after_alarm, after_local, after_oracle, silent_rows = [], [], [], []
        for i, a in enumerate(atoms):
            Wf = W
            for j in sorted((j for j in range(n) if j != i), key=lambda j: -atoms[j].energy):
                Wf = fix(Wf, atoms[j])
            g = lin.global_alarm(Wf)
            rows = local[i]["rows"]
            hit_or = (lin.chi_square(Wf[rows]) >= local[i]["threshold"]) if rows.size else np.zeros(draws, dtype=bool)
            hit = np.zeros(draws, dtype=bool)
            for k in range(draws):
                c = parts[k]["atom_cluster"][i]
                if c is None:
                    continue
                ls = local_set(lin, geo, parts[k], c, a)
                if ls["rows"].size:
                    v = Wf[ls["rows"], k]
                    hit[k] = float(v @ v) >= ls["threshold"]
            after_alarm.append(float(g.mean()))
            after_local.append(float(hit.mean()))
            after_oracle.append(float(hit_or.mean()))
            silent = ~g
            silent_rows.append({"silent": int(silent.sum()), "local_flags": int((hit & silent).sum()),
                                "oracle_local_flags": int((hit_or & silent).sum())})
        out.update({"after_others_fixed_alarm": after_alarm, "after_others_fixed_local": after_local,
                    "after_others_fixed_local_oracle": after_oracle, "after_others_fixed_silent": silent_rows})
    out["progress"] = _progress_study(lin, geo, atoms, ev, W, local, u_tot, noiseless_alone)
    return out


def _best_wrong(lin, geo, atoms, local_buses, rs, ls) -> dict[str, tuple[str, int] | None]:
    """The strongest expected line and P/Q meter inside a cluster's buses that no atom's correction names."""
    true = {a.hypothesis for a in atoms}
    inside = np.zeros(geo.nb, dtype=bool)
    inside[list(local_buses)] = True
    out: dict[str, tuple[str, int] | None] = {"wrong_line": None, "wrong_meter": None}
    cand_l = [k for k in range(geo.nl) if inside[geo.f[k]] and inside[geo.t[k]] and ("l", k) not in true]
    if cand_l:
        out["wrong_line"] = ("l", int(max(cand_l, key=lambda k: ls[k])))
    cand_m = [j for j in geo.pq_rows if inside[geo.row_b1[j]] and inside[geo.row_b2[j]] and ("m", j) not in true]
    if cand_m:
        out["wrong_meter"] = ("m", int(max(cand_m, key=lambda j: abs(rs[j]))))
    return out


def _progress_study(lin, geo, atoms, ev, W, local, u_tot, alone) -> list[dict[str, Any]]:
    """What the verifier's rules would keep: the true correction against the best wrong line and meter.

    The floors alone, on the global J basis (the default ``global_progress``), the global excess basis
    (``chi_square_excess``) and a cluster-local excess basis over the atom's noiseless cluster's
    measurements; and the dispositions: the deployed rule (final when the global alarm is silent after the
    correction, else partial on the J-basis floor), the same on the excess basis, and the cluster-resolved
    rule (the cluster's local alarm silent after the correction and the global J not risen).  The single
    target test of the deployed verifier is passed by every fit here (a fit clears its own multiplier), so
    these rules are what separates a true correction from a wrong one in this model, as in stage-2 item 4.
    """
    J0 = lin.chi_square(W)
    rs, ls = lin.residual_scores(u_tot), lin.line_scores(u_tot)
    out = []
    for i, a in enumerate(atoms):
        rows = local[i]["rows"]
        if not rows.size:
            out.append({"available": False})
            continue
        JC0 = lin.chi_square(W[rows])
        res: dict[str, Any] = {"available": True, "alone": bool(alone[i]), "n_atoms": len(atoms)}
        cands: dict[str, Atom] = {"true": a}
        for name, e in _best_wrong(lin, geo, atoms, local[i]["buses"], rs, ls).items():
            if e is not None:
                cands[name] = hypothesis_atom(lin, geo, e)
                res[f"{name}_element"] = list(e)
        local_after, resolved = {}, {}
        for name, cand in cands.items():
            W1 = fix(W, cand, remove=(name == "true"))
            J1, JC1 = lin.chi_square(W1), lin.chi_square(W1[rows])
            floor = METER_FLOOR if cand.kind == "meter" else BRANCH_FLOOR
            pj, pe, pc = _progress(J0, J1, 0.0), _progress(J0, J1, float(lin.dof)), _progress(JC0, JC1, local[i]["dof"])
            pj, pe, pc_ = (np.nan_to_num(p, nan=-1.0) for p in (pj, pe, pc))
            silent = ~lin.global_alarm(W1)
            nonreg = J1 <= J0
            local_after[name] = JC1
            resolved[name] = (JC1 < local[i]["threshold"]) & nonreg
            res[name] = {
                "kind": cand.kind, "floor": floor,
                "floor_global_j": float(np.mean(pj >= floor)),
                "floor_global_excess": float(np.mean(pe >= floor)),
                "floor_cluster_local": float(np.mean((pc_ >= floor) & nonreg)),
                "deployed_rule": float(np.mean(silent | (pj >= floor))),
                "excess_rule": float(np.mean(silent | (pe >= floor))),
                "cluster_resolved": float(np.mean(resolved[name])),
                "global_silent_after": float(np.mean(silent)),
                "median": [_med(pj), _med(pe), _med(pc)],
            }
        # The in-cluster comparison (as the bounded comparison on cloned states): every candidate is fitted, the
        # one that leaves the lowest local statistic is kept if it resolves the cluster.
        best = np.min(np.vstack(list(local_after.values())), axis=0)
        for name in cands:
            res[name]["resolved_and_best"] = float(np.mean(resolved[name] & (local_after[name] <= best)))
        out.append(res)
    return out


def _med(x: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), 4) if x.size else None


# --------------------------------------------------------------------------- systems, operating points, pools

@dataclass
class System:
    name: str
    case: dict[str, Any]
    sigma: np.ndarray
    exact: list[int]
    states: list[tuple[np.ndarray, np.ndarray, str]]
    meters: list[tuple[int, float]]
    parameters: list[tuple[int, float, float]]
    outage_lines: list[int] | None          # None: every non-radial line


def load_ieee14(suite_path: Path, n_states: int) -> tuple[System, list[dict[str, Any]]]:
    roots = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    compact = []
    for r in roots:
        c, tr = r["audit"]["compound"], r["audit"]["truth"]
        noise = r["execution"]["metadata"]["operator_noise"]
        compact.append({"scenario_id": r["execution"]["scenario_id"], "class": c["class"], "atoms": c["atoms"],
                        "solo": c["solo_effects"], "relations": c["relations"], "z_clean": tr["clean_measurements"],
                        "clean_case": tr["clean_case"], "sigma": noise["measurement_sigma"],
                        "exact": noise["structural_zero_indices"]})
    del roots
    case = _as_case(_load_python_case("case14"))
    sigma = np.asarray(compact[0]["sigma"], dtype=float)
    exact = [int(i) for i in compact[0]["exact"]]
    states = []
    for r in [r for r in compact if r["class"] == "control_meter"][:n_states]:
        det = _solve(_as_case(r["clean_case"]), r["z_clean"], sigma, exact)
        states.append((np.asarray(det["theta_est_rad"]), np.asarray(det["vm_est_pu"]), r["scenario_id"]))
    meters, parameters, outages = [], [], set()
    for r in compact:
        for a in r["atoms"]:
            if a["kind"] == "meter":
                meters.append((int(a["index"]), float(a["offset"])))
            elif a["kind"] == "parameter":
                parameters.append((int(a["branch_row0"]), float(a["r_factor"]), float(a["x_factor"])))
            elif a["kind"] == "breaker" and a.get("physical_effect") == "dangling_line_terminal":
                outages.add(int(a["branch_row0"]))
    return System("ieee14", case, sigma, exact, states, meters, parameters, sorted(outages)), compact


def load_ieee118(dataset_dir: Path, n_states: int) -> System:
    dataset_dir = Path(dataset_dir)
    meters, parameters, sigma = [], [], None
    with (dataset_dir / "roots.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["family"] not in ("measurement", "multi_measurement", "parameter"):
                continue
            if sigma is None:
                sigma = np.asarray(r["metadata"]["sigma_z"], dtype=float)
            for m in r["truth"].get("measurement") or []:
                meters.append((int(m["index"]), float(m["observed"]) - float(m["clean"])))
            for p in r["truth"].get("parameter") or []:
                parameters.append((int(p["branch_row0"]), float(p["r_factor"]), float(p["x_factor"])))
    case = _as_case(_load_python_case("case118"))
    states = []
    for path in sorted(glob.glob(str(dataset_dir / "balanced" / "shard_*" / "corpus" / "samples.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                if r["label"]["error_type"] == "no_error":
                    det = _solve(case, r["z_true"], r["sigma_z"], [])
                    states.append((np.asarray(det["theta_est_rad"]), np.asarray(det["vm_est_pu"]), r["id"]))
                    break
        if len(states) >= n_states:
            break
    return System("ieee118", case, sigma, [], states, meters, parameters, None)


# --------------------------------------------------------------------------- workers

_SYSTEM: System | None = None
_LINS: list[Linearization] = []
_GEO: Geometry | None = None
_OUTAGES: list[list[Atom]] = []


def _init_worker(system: System, bank: int, seed: int, lambda_flag: float = LAMBDA_FLAG) -> None:
    global _SYSTEM, _LINS, _GEO, _OUTAGES
    _SYSTEM = system
    _CONFIG["lambda_flag"] = float(lambda_flag)
    _LINS = [Linearization(system.case, th, vm, system.sigma, system.exact, bank=bank, seed=seed + s)
             for s, (th, vm, _) in enumerate(system.states)]
    _GEO = Geometry(_LINS[0])
    _OUTAGES = []
    # A recorded healthy state cannot carry an open line into a zero-injection bus (the line's flow at that
    # bus would break the exact constraint), so the drawn outages stay off those lines (IEEE 14: bus 7).
    exact_buses = {int((i - _LINS[0].nb) % _LINS[0].nb) for i in _LINS[0].exact}
    for lin in _LINS:
        lines = system.outage_lines if system.outage_lines is not None else range(lin.nl)
        lines = [k for k in lines if not ({int(_GEO.f[k]), int(_GEO.t[k])} & exact_buses)]
        _OUTAGES.append([a for a in (outage_atom(lin, _GEO, k) for k in lines) if a is not None])


def _pool_candidates(kind: str, s: int, anchor: Sequence[int] | None, distance: int | None, in_band: bool) -> list[Any]:
    geo, lin = _GEO, _LINS[s]
    def near(buses):
        if anchor is None:
            return True
        d = min(geo.distance(anchor, buses), DISTANCE_BINS[-1])
        return d == distance
    if kind == "meter":
        return [m for m in _SYSTEM.meters if m[0] in lin.row_of_index and near(geo.row_buses[lin.row_of_index[m[0]]])]
    if kind == "parameter":
        return [p for p in _SYSTEM.parameters if near(geo.line_buses[p[0]])]
    return [a for a in _OUTAGES[s] if (BAND[0] <= a.natural_ratio <= BAND[1] if in_band else a.natural_ratio >= BAND[0])
            and near(a.buses)]


def _draw_atom(s: int, kind: str, rng: np.random.Generator, ratio: float | None, taken: set[tuple[str, int]], *,
               anchor: Sequence[int] | None = None, distance: int | None = None, in_band: bool = True) -> Atom | None:
    lin, geo = _LINS[s], _GEO
    cands = _pool_candidates(kind, s, anchor, distance, in_band)
    for _ in range(40):
        if not cands:
            return None
        c = cands[int(rng.integers(len(cands)))]
        if kind == "meter":
            atom = meter_atom(lin, geo, c[0], c[1], ratio=ratio)
        elif kind == "parameter":
            atom = parameter_atom(lin, geo, c[0], c[1], c[2], ratio=ratio)
        else:
            atom = c
        if atom is not None and atom.hypothesis not in taken:
            taken.add(atom.hypothesis)
            return atom
    return None


def _combination(task: dict[str, Any]) -> dict[str, Any] | None:
    rng = np.random.default_rng(task["seed"])
    s = int(rng.integers(len(_LINS)))
    lin, geo = _LINS[s], _GEO
    n = int(task["n_atoms"])
    kinds = [str(k) for k in rng.choice(task.get("kind_set", KINDS), size=n)]
    taken: set[tuple[str, int]] = set()
    atoms: list[Atom] = []
    if task["mode"] == "matched":
        target = task.get("ratio") or float(math.exp(rng.uniform(math.log(BAND[0]), math.log(BAND[1]))))
        # An outage keeps its natural magnitude (in the band); the other atoms are matched to the first one.
        order = sorted(range(n), key=lambda i: kinds[i] != "outage")
        for pos, i in enumerate(order):
            anchor = atoms[0].buses if (pos == 1 and task.get("distance_bin") is not None) else None
            a = _draw_atom(s, kinds[i], rng, target, taken, anchor=anchor, distance=task.get("distance_bin"))
            if a is None:
                return None
            if a.kind == "outage" and pos == 0:
                target = a.natural_ratio
            atoms.append(a)
        atoms = [a if a.kind == "outage" else a.scaled(target) for a in atoms]
    else:   # independent magnitudes: each atom log-uniform in the band; outages in it, or natural >= 1.5
        for kind in kinds:
            r = float(math.exp(rng.uniform(math.log(BAND[0]), math.log(BAND[1]))))
            a = _draw_atom(s, kind, rng, r, taken, in_band=not task.get("natural_outages", False))
            if a is None:
                return None
            atoms.append(a)
    rec = evaluate(lin, geo, atoms, draws=int(task["draws"]), rng=rng, sensitivity=bool(task.get("sensitivity")))
    rec.update({"set": task["set"], "state": s, "seed": task["seed"]})
    return rec


def _footprint_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Every element of a kind alone at a fixed ratio (first state): extent, alarms, local alarm."""
    lin, geo = _LINS[0], _GEO
    kind, ratio = task["kind"], float(task["ratio"])
    if kind == "meter":
        made = [meter_atom(lin, geo, int(lin.rows[j]), 1.0, ratio=ratio) for j in geo.pq_rows]
    elif kind == "parameter":
        made = [parameter_atom(lin, geo, k, 1.0, 2.0, ratio=ratio) for k in range(lin.nl)]
    else:
        made = [a.scaled(ratio) for a in _OUTAGES[0]]
    E = lin.bank[:, :400]
    out = []
    for a in made:
        if a is None:
            continue
        rs, ls = lin.residual_scores(a.u), lin.line_scores(a.u)
        fl = flags_of(rs, ls)
        fb = {b for x in fl for b in element_buses(geo, x)}
        part = {"flags": fl, "labels": np.zeros(len(fl), dtype=int)}
        loc = local_set(lin, geo, part if fl else None, 0 if fl else None, a)
        Wa = a.u[:, None] + E
        J = lin.chi_square(Wa[loc["rows"]]) if loc["rows"].size else np.zeros(E.shape[1])
        out.append({
            "kind": kind, "ratio": ratio, "element": int(a.element), "n_flags": len(fl), "n_buses": len(fb),
            "bus_share": round(len(fb) / lin.nb, 4),
            "radius": max((geo.distance(a.buses, element_buses(geo, x)) for x in fl), default=None),
            "max_degree": int(max(geo.degree[b] for b in a.buses)),
            "own_flagged": bool(set(own_evidence(a, geo)) & set(fl)),
            "max_residual": round(float(np.abs(rs).max()), 3), "max_lambda": round(float(ls.max()), 3),
            "global_alarm": round(float(lin.global_alarm(Wa).mean()), 4),
            "global_chi_alarm": round(float((lin.chi_square(Wa) >= lin.threshold).mean()), 4),
            "local_alarm": round(float((J >= loc["threshold"]).mean()), 4),
            "local_rows": int(loc["rows"].size), "local_dof": round(loc["dof"], 2),
            "local_ratio": round(float(a.u[loc["rows"]] @ a.u[loc["rows"]]) / max(loc["threshold"] - loc["dof"], 1e-9), 3)
            if loc["rows"].size else 0.0,
            "concentration": round(float(a.u[loc["rows"]] @ a.u[loc["rows"]]) / a.energy, 4) if loc["rows"].size else 0.0,
        })
    return out


def _system_curves(task: dict[str, Any]) -> dict[str, Any]:
    """Correlation decay of the hypothesis set, noise-only flags and clusters, leverage, Satterthwaite check."""
    lin, geo = _LINS[0], _GEO
    MM = np.abs(lin.S) / lin._umet_div[:, None] / lin._umet_div[None, :]
    LM = (np.abs(lin.Upar.T) / lin._upar_div[:, None] / lin._umet_div[None, :]).reshape(lin.nl, 2, lin.m).max(axis=1)
    LL = (np.abs(lin.Upar.T @ lin.Upar) / lin._upar_div[:, None] / lin._upar_div[None, :]).reshape(
        lin.nl, 2, lin.nl, 2).max(axis=(1, 3))
    def dmat(a1, a2, c1, c2):
        return np.minimum(np.minimum(geo.D[np.ix_(a1, c1)], geo.D[np.ix_(a1, c2)]),
                          np.minimum(geo.D[np.ix_(a2, c1)], geo.D[np.ix_(a2, c2)]))
    pq = np.array(geo.pq_rows)
    lf, lt = geo.f, geo.t
    mats = {
        "meter_meter": (MM[np.ix_(pq, pq)], dmat(geo.row_b1[pq], geo.row_b2[pq], geo.row_b1[pq], geo.row_b2[pq]), True),
        "line_meter": (LM[:, pq], dmat(lf, lt, geo.row_b1[pq], geo.row_b2[pq]), False),
        "line_line": (LL, dmat(lf, lt, lf, lt), True),
    }
    curves = {}
    for name, (M, Dm, symmetric) in mats.items():
        mask = np.triu(np.ones(M.shape, dtype=bool), 1) if symmetric else np.ones(M.shape, dtype=bool)
        vals, dist = M[mask], np.minimum(Dm[mask], DISTANCE_BINS[-1])
        curves[name] = {str(d): {"n": int((dist == d).sum()), "median": round(float(np.median(vals[dist == d])), 4),
                                 "p90": round(float(np.percentile(vals[dist == d], 90)), 4),
                                 "p99": round(float(np.percentile(vals[dist == d], 99)), 4),
                                 "max": round(float(vals[dist == d].max()), 4)}
                        for d in DISTANCE_BINS if (dist == d).any()}
    B = lin.bank
    rsB, lsB = lin.residual_scores(B), lin.line_scores(B)
    n_flags, n_flags4, n_clusters, local_fa, local_n = [], [], [], 0, 0
    for k in range(B.shape[1]):
        fl = flags_of(rsB[:, k], lsB[:, k])
        n_flags.append(len(fl))
        n_flags4.append(len(flags_of(rsB[:, k], lsB[:, k], lambda_flag=4.0)))
        labels = clusters_of(geo, fl, LINK_DISTANCE)
        n_clusters.append(int(len(set(labels.tolist()))))
        for c in set(labels.tolist()):
            buses = geo.closed({b for e, lab in zip(fl, labels) if lab == c for b in element_buses(geo, e)}, CLOSURE_HOPS)
            rows = geo.measurement_set(buses)
            thr, _, _ = satterthwaite_threshold(lin, rows)
            local_n += 1
            local_fa += int(float(B[rows, k] @ B[rows, k]) >= thr)
    rng = np.random.default_rng(task["seed"])
    nominal = []
    for b in rng.choice(lin.nb, size=min(30, lin.nb), replace=False):
        rows = geo.measurement_set(geo.closed([int(b)], CLOSURE_HOPS))
        thr, _, _ = satterthwaite_threshold(lin, rows)
        nominal.append(float((lin.chi_square(B[rows]) >= thr).mean()))
    redundancy = np.diag(lin.S)
    return {
        "system": _SYSTEM.name, "m": lin.m, "dof": lin.dof, "threshold": round(lin.threshold, 3),
        "excess": round(lin.excess, 3), "nb": lin.nb, "nl": lin.nl, "degree_max": int(geo.degree.max()),
        "degree_mean": round(float(geo.degree.mean()), 3),
        "bus_distance_max": int(geo.D[geo.D < 10 ** 6].max()),
        "correlation_decay": curves,
        "noise_only": {"draws": int(B.shape[1]), "global_false_alarm": round(float(lin.global_alarm(B).mean()), 4),
                       "global_chi_false_alarm": round(float((lin.chi_square(B) >= lin.threshold).mean()), 4),
                       "flags_mean": round(float(np.mean(n_flags)), 4),
                       "snapshots_with_flag": round(float(np.mean(np.array(n_flags) > 0)), 4),
                       "snapshots_with_flag_lambda4": round(float(np.mean(np.array(n_flags4) > 0)), 4),
                       "clusters_mean": round(float(np.mean(n_clusters)), 4),
                       "noise_cluster_local_false_alarm": round(local_fa / local_n, 4) if local_n else None,
                       "noise_clusters": local_n},
        "satterthwaite_nominal_false_alarm": {"median": round(float(np.median(nominal)), 4),
                                              "max": round(float(np.max(nominal)), 4)},
        "redundancy": {"median": round(float(np.median(redundancy)), 4), "min": round(float(redundancy.min()), 4),
                       "below_0.1": int((redundancy < 0.1).sum()), "below_0.3": int((redundancy < 0.3).sum())},
    }


# --------------------------------------------------------------------------- the IEEE 14 suite at natural magnitudes

def _suite_root(task: dict[str, Any]) -> dict[str, Any]:
    r = task["root"]
    head = {"scenario_id": r["scenario_id"], "class": r["class"]}
    if any(a["kind"] == "breaker" and a.get("physical_effect") != "dangling_line_terminal" for a in r["atoms"]):
        return {**head, "skipped": "bus_split"}
    sigma, exact = np.asarray(r["sigma"], dtype=float), [int(i) for i in r["exact"]]
    det = _solve(_as_case(r["clean_case"]), r["z_clean"], sigma, exact)
    if not det["success"]:
        return {**head, "skipped": "true_solve"}
    case = _as_case(_load_python_case("case14"))
    true_state = (np.asarray(det["theta_est_rad"]), np.asarray(det["vm_est_pu"]))
    lin = Linearization(case, *true_state, sigma, exact, bank=max(256, task["draws"]), seed=task["seed"])
    geo = Geometry(lin)
    h_true_state = _h(case, *true_state)
    # A line whose terminal is open carries no current: its impedance has no effect on any measurement (the
    # compound audit scores it not applicable), so its parameter atom is dropped, not added to the outage.
    open_lines = {int(a["branch_row0"]) for a in r["atoms"] if a["kind"] == "breaker"}
    atoms, kept, nonlinear = [], [], []
    for pos, a in enumerate(r["atoms"]):
        d = np.zeros(lin.h0.size)
        if a["kind"] == "meter":
            atom = meter_atom(lin, geo, int(a["index"]), float(a["offset"]), ratio=None)
            d[int(a["index"])] = float(a["offset"])
        elif a["kind"] == "parameter":
            if int(a["branch_row0"]) in open_lines:
                continue
            atom = parameter_atom(lin, geo, int(a["branch_row0"]), float(a["r_factor"]), float(a["x_factor"]),
                                  ratio=None, exact=True, state=true_state)
            true_case = copy.deepcopy(case)
            true_case["branch"][int(a["branch_row0"]), lp.BR_R] = float(a["clean_r"])
            true_case["branch"][int(a["branch_row0"]), lp.BR_X] = float(a["clean_x"])
            d = _deviation(lin, true_case, true_state)
        else:
            atom = outage_atom(lin, geo, int(a["branch_row0"]), state=true_state)
            out_case = copy.deepcopy(case)
            out_case["branch"][int(a["branch_row0"]), lp.BR_STATUS] = 0.0
            d = _deviation(lin, out_case, true_state)
        if atom is None:
            return {**head, "skipped": "atom_without_effect"}
        # The nonlinear WLS on the same deviation at the same true state: the linearization's own check.
        z = h_true_state + d
        z[exact] = 0.0
        solved = _solve(case, z, sigma, exact)
        nonlinear.append(float(solved["wls_objective"]) if solved["success"] else None)
        atoms.append(atom)
        kept.append(pos)
    rec = evaluate(lin, geo, atoms, draws=task["draws"], rng=np.random.default_rng(task["seed"]))
    rec["nonlinear_fixed_state_chi_square"] = nonlinear
    # The whole root: the true operator model's noiseless vector at the true state plus the meter offsets,
    # solved on the reported model, against the chi-square of the summed atoms (the superposition check).
    z_root = _h(_as_case(r["clean_case"]), *true_state)
    for a in r["atoms"]:
        if a["kind"] == "meter":
            z_root[int(a["index"])] += float(a["offset"])
    z_root[exact] = 0.0
    solved = _solve(case, z_root, sigma, exact)
    u_sum = np.sum([a.u for a in atoms], axis=0)
    rec["root_chi_square_linear_nonlinear"] = [round(float(u_sum @ u_sum), 3),
                                               float(solved["wls_objective"]) if solved["success"] else None]
    solo = {int(s["atom"]): s for s in r["solo"]}
    rec.update(head)
    rec["isolated_parameter_atoms"] = len(r["atoms"]) - len(atoms)
    rec["relations"] = [x.get("relation") for x in r["relations"]]
    rec["recorded_solo_alarmed"] = [bool(solo[p]["alarmed"]) if p in solo else None for p in kept]
    rec["recorded_solo_below_chi"] = [bool(solo[p]["chi_square"] < solo[p]["chi_square_threshold"]) if p in solo else None
                                      for p in kept]
    rec["recorded_solo_chi_square"] = [float(solo[p]["chi_square"]) if p in solo else None for p in kept]
    rec["linear_solo_chi_square"] = [round(a.energy, 3) for a in atoms]
    rec["linear_solo_below_chi"] = [bool(a.energy < lin.threshold) if a.kind != "meter" else None for a in atoms]
    rec["linear_solo_alarmed_noiseless"] = [
        bool(a.energy >= lin.threshold or float(np.abs(lin.residual_scores(a.u)).max()) >= RESIDUAL_FLAG) for a in atoms]
    return rec


# --------------------------------------------------------------------------- aggregation

def _share(values: Iterable[Any]) -> float | None:
    values = [bool(v) for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def _mean(values: Iterable[Any]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def _q(values: Iterable[Any], qs: Sequence[int] = (10, 50, 90)) -> list[float] | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return [round(float(np.percentile(values, q)), 4) for q in qs] if values else None


PROGRESS_RULES = ("floor_global_j", "floor_global_excess", "floor_cluster_local", "deployed_rule", "excess_rule",
                  "cluster_resolved", "resolved_and_best", "global_silent_after")


def _progress_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    prog = [p for r in records if "noisy" in r for p in r["noisy"]["progress"] if p.get("available")]
    if not prog:
        return None
    out: dict[str, Any] = {}
    for split, keep in (("all", lambda p: True), ("alone", lambda p: p["alone"]), ("merged", lambda p: not p["alone"])):
        part = [p for p in prog if keep(p)]
        entry = {}
        for name in ("true", "wrong_line", "wrong_meter"):
            ps = [p[name] for p in part if name in p]
            if ps:
                entry[name] = {"n": len(ps), **{b: _mean(p[b] for p in ps) for b in PROGRESS_RULES},
                               "median_progress_j_excess_local": [(_q((p["median"][i] for p in ps), (50,)) or [None])[0]
                                                                  for i in range(3)]}
        out[split] = entry
    return out


def _peel_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    recs = [r for r in records if r["n_atoms"] > 1]
    if not recs:
        return None
    return {"rounds_over_atoms": _q(r["peel"]["rounds"] / r["n_atoms"] for r in recs),
            "fixed_alone_share": round(sum(r["peel"]["fixed_alone"] for r in recs) / sum(r["n_atoms"] for r in recs), 4),
            "never_flagged_share": round(sum(r["peel"]["never_flagged"] for r in recs) / sum(r["n_atoms"] for r in recs), 4),
            "max_width": _q(r["peel"]["max_width"] for r in recs)}


def summarize_combinations(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    recs = [r for r in records if r and "n_atoms" in r]
    if not recs:
        return {"n": 0}
    multi = [r for r in recs if r["n_atoms"] > 1]
    noisy = [r["noisy"] for r in recs if "noisy" in r]
    out: dict[str, Any] = {"n": len(recs), "atoms": sum(r["n_atoms"] for r in recs)}
    if multi:
        out["separable"] = _share(r["separable"] for r in multi)
        out["criterion_parts"] = {
            "S1_all_visible": _share(all(r["noiseless"]["visible"]) for r in multi),
            "S2_distinct_and_pure": _share(r["noiseless"]["distinct"] and r["noiseless"]["pure"] for r in multi),
            "S3_masking_keep_0.9": _share(min(r["masking_keep"]) >= MASKING_KEEP for r in multi),
        }
        if "separable_sensitivity" in multi[0]:
            out["separable_sensitivity"] = {k: _share(r["separable_sensitivity"][k] for r in multi)
                                            for k in multi[0]["separable_sensitivity"]}
        out["noisy_partition_recovered"] = _mean(x["partition_recovered"] for x in noisy)
    out["noisy_spurious_clusters_mean"] = _mean(x["spurious_clusters_mean"] for x in noisy)
    out["n_clusters_noiseless"] = {str(k): v for k, v in sorted(Counter(r["noiseless"]["n_clusters"] for r in recs).items())}
    out["atoms_isolated"] = _share(v for r in recs for v in r["atom_isolated"])
    out["atoms_isolated_noisy"] = _mean(v for x in noisy for v in x["alone"])
    out["ratio_quantiles"] = _q(x for r in recs for x in r["ratio"])
    pairs = [p for r in recs for p in r["pairs"]]
    out["pair_distance"] = {str(k): v for k, v in sorted(Counter(min(p["distance"], DISTANCE_BINS[-1]) for p in pairs).items())}
    by_d = defaultdict(list)
    for r in recs:
        if r["n_atoms"] == 2:
            by_d[min(r["pairs"][0]["distance"], DISTANCE_BINS[-1])].append(r)
    out["by_distance"] = {}
    for d, rs in sorted(by_d.items()):
        ps = [r["pairs"][0] for r in rs]
        out["by_distance"][str(d)] = {
            "n": len(rs), "separable": _share(r["separable"] for r in rs),
            "S2_distinct_and_pure": _share(r["noiseless"]["distinct"] and r["noiseless"]["pure"] for r in rs),
            "S3_masking": _share(min(r["masking_keep"]) >= MASKING_KEEP for r in rs),
            "overlap": _q((p["overlap"] for p in ps), (50, 90)),
            "cosine_abs": _q((abs(p["cosine"]) for p in ps), (50, 90)),
            "own_corr": _q(p["own_corr"] for p in ps),
            "cross_block_max": _q(p["cross_block_max"] for p in ps if p["cross_block_max"] is not None),
            "masking_keep_min": _q(min(r["masking_keep"]) for r in rs),
            "noisy_partition_recovered": _mean(r["noisy"]["partition_recovered"] for r in rs if "noisy" in r),
        }
    out["by_kinds"] = {k: {"n": len(v), "separable": _share(r["separable"] for r in v)}
                       for k, v in sorted(_group(multi, lambda r: "+".join(sorted(r["kinds"]))).items())}
    out["by_max_degree"] = {k: {"n": len(v), "separable": _share(r["separable"] for r in v)}
                            for k, v in sorted(_group(multi, lambda r: "<=3" if r["max_degree"] <= 3 else
                                                      ("4-5" if r["max_degree"] <= 5 else ">=6")).items())}
    if noisy:
        alone = [x for r in noisy for x in r["own_alarm_alone"]]
        out["c_own_alarm_alone"] = {"mean": _mean(alone), "share_below_0.9": _share(v < 0.9 for v in alone)}
        out["c_visible_with_others"] = _mean(x for r in noisy for x in r["visible_with_others"])
        out["c_unflagged_with_others_share"] = _share(x < 0.5 for r in noisy for x in r["visible_with_others"])
        after = [x for r in noisy for x in r.get("after_others_fixed_alarm", [])]
        if after:
            out["c_after_others_fixed_alarm"] = _mean(after)
            out["c_after_others_fixed_silent_share"] = _share(x < 0.5 for x in after)
            sil = [x for r in noisy for x in r.get("after_others_fixed_silent", [])]
            silent = sum(x["silent"] for x in sil)
            out["d_global_silent_draws"] = silent
            out["d_local_flags_when_global_silent"] = round(sum(x["local_flags"] for x in sil) / silent, 4) if silent else None
            out["d_oracle_local_flags_when_global_silent"] = round(sum(x["oracle_local_flags"] for x in sil) / silent, 4) if silent else None
            out["d_after_fix_local"] = _mean(x for r in noisy for x in r.get("after_others_fixed_local", []))
    prog = _progress_summary(recs)
    if prog:
        out["progress"] = prog
    out["peel"] = _peel_summary(recs)
    ranks = [(g, c) for r in recs for g, c in zip(r["own_line_rank_global"], r["own_line_rank_cluster"]) if g is not None]
    if ranks:
        out["line_atom_rank"] = {
            "n": len(ranks),
            "global_top1": _share(g == 1 for g, _ in ranks), "global_top3": _share(g <= 3 for g, _ in ranks),
            "cluster_top1": _share(c == 1 if c is not None else False for _, c in ranks),
            "cluster_top3": _share(c <= 3 if c is not None else False for _, c in ranks)}
    out["local_dof"] = _q(l["dof"] for r in recs for l in r["local"])
    out["local_concentration"] = _q(l["concentration"] for r in recs for l in r["local"])
    out["footprint_buses"] = _q(f["n_buses"] for r in recs for f in r["footprint"])
    return out


def summarize_footprints(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for (kind, ratio), rs in sorted(_group(records, lambda r: (r["kind"], r["ratio"])).items()):
        out[f"{kind}@{ratio}"] = {
            "n": len(rs), "n_buses": _q(r["n_buses"] for r in rs), "bus_share": _q(r["bus_share"] for r in rs),
            "radius": _q(r["radius"] for r in rs), "own_flagged": _share(r["own_flagged"] for r in rs),
            "global_alarm": _mean(r["global_alarm"] for r in rs), "global_chi_alarm": _mean(r["global_chi_alarm"] for r in rs),
            "local_alarm": _mean(r["local_alarm"] for r in rs), "local_dof": _q(r["local_dof"] for r in rs),
            "local_ratio": _q(r["local_ratio"] for r in rs), "concentration": _q(r["concentration"] for r in rs),
            "max_residual": _q(r["max_residual"] for r in rs), "max_lambda": _q(r["max_lambda"] for r in rs),
            "n_buses_degree>=5": _q(r["n_buses"] for r in rs if r["max_degree"] >= 5),
            "n_buses_degree<=3": _q(r["n_buses"] for r in rs if r["max_degree"] <= 3),
        }
    return out


def summarize_suite(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for cls, rs in sorted(_group(records, lambda r: r["class"]).items()):
        ok = [r for r in rs if "n_atoms" in r]
        multi = [r for r in ok if r["n_atoms"] > 1]
        entry: dict[str, Any] = {"n": len(rs), "evaluated": len(ok),
                                 "skipped": dict(Counter(r["skipped"] for r in rs if "skipped" in r))}
        if ok:
            entry["root_linear_vs_nonlinear_ratio"] = _q(l / s for r in ok for l, s in [r["root_chi_square_linear_nonlinear"]]
                                                         if s)
            entry["linear_vs_nonlinear_fixed_state_ratio"] = _q(
                l / s for r in ok for l, s in zip(r["linear_solo_chi_square"], r["nonlinear_fixed_state_chi_square"]) if s)
            entry["linear_vs_recorded_solo_ratio"] = _q(l / s for r in ok for l, s in
                                                       zip(r["linear_solo_chi_square"], r["recorded_solo_chi_square"]) if s)
            entry["recorded_atom_below_alarm"] = _share(any(v is False for v in r["recorded_solo_alarmed"]) for r in ok)
            entry["linear_atom_below_alarm"] = _share(not all(r["linear_solo_alarmed_noiseless"]) for r in ok)
            # Stage 1's reading: a structural atom's noiseless chi-square alone below the threshold.
            entry["recorded_structural_atom_below_chi"] = sum(any(v is True for v in r["recorded_solo_below_chi"]) for r in ok)
            entry["linear_structural_atom_below_chi"] = sum(any(v is True for v in r["linear_solo_below_chi"]) for r in ok)
            entry["isolated_parameter_atoms_dropped"] = sum(r["isolated_parameter_atoms"] for r in ok)
            entry["own_alarm_alone_mean"] = _mean(x for r in ok for x in r["noisy"]["own_alarm_alone"])
        if multi:
            entry["separable"] = _share(r["separable"] for r in multi)
            entry["S1_all_visible"] = _share(all(r["noiseless"]["visible"]) for r in multi)
            entry["S2_distinct_and_pure"] = _share(r["noiseless"]["distinct"] and r["noiseless"]["pure"] for r in multi)
            entry["S3_masking"] = _share(min(r["masking_keep"]) >= MASKING_KEEP for r in multi)
            entry["noisy_partition_recovered"] = _mean(r["noisy"]["partition_recovered"] for r in multi)
            after = [x for r in multi for x in r["noisy"]["after_others_fixed_alarm"]]
            entry["after_others_fixed_alarm_mean"] = _mean(after)
            entry["atom_silent_after_fix_share"] = _share(x < 0.5 for x in after)
            sil = [x for r in multi for x in r["noisy"]["after_others_fixed_silent"]]
            silent = sum(x["silent"] for x in sil)
            entry["local_flags_when_global_silent"] = round(sum(x["local_flags"] for x in sil) / silent, 4) if silent else None
            entry["oracle_local_flags_when_global_silent"] = round(sum(x["oracle_local_flags"] for x in sil) / silent, 4) if silent else None
            entry["pair_distance"] = {str(k): v for k, v in sorted(Counter(min(p["distance"], DISTANCE_BINS[-1])
                                                                          for r in multi for p in r["pairs"]).items())}
            prog = _progress_summary(multi)
            if prog:
                entry["progress"] = prog
        out[cls] = entry
    return out


def _group(items: Iterable[Any], key: Callable[[Any], Any]) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = defaultdict(list)
    for x in items:
        out[key(x)].append(x)
    return out


# --------------------------------------------------------------------------- main

class Runner:
    """One worker pool per system (each worker builds the system's linearizations once)."""

    def __init__(self, pool_args: tuple, workers: int):
        self.workers = int(workers)
        _init_worker(*pool_args)
        self.pool = (ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker, initargs=pool_args)
                     if self.workers > 1 else None)

    def map(self, fn: Callable, tasks: Sequence[Any]) -> list[Any]:
        if self.pool is None or len(tasks) <= 1:
            return [fn(t) for t in tasks]
        return list(self.pool.map(fn, tasks, chunksize=max(1, len(tasks) // (self.workers * 6))))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()


def _tasks(seed: int, n: int, name: str, **spec: Any) -> list[dict[str, Any]]:
    return [{"seed": seed * 1_000_003 + i, "set": name, **spec} for i in range(n)]


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in records), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--ieee118-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=1200)
    parser.add_argument("--triples", type=int, default=1200)
    parser.add_argument("--per-bin", type=int, default=120)
    parser.add_argument("--variable", type=int, default=300)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--bank", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--systems", nargs="*", default=["ieee14", "ieee118"])
    parser.add_argument("--skip-suite", action="store_true")
    parser.add_argument("--lambda-flag", type=float, default=LAMBDA_FLAG)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def log(**fields: Any) -> None:
        print(json.dumps({"elapsed": round(time.perf_counter() - started, 1), **fields}, default=str), flush=True)

    systems: dict[str, System] = {}
    suite_roots: list[dict[str, Any]] = []
    if "ieee14" in args.systems:
        systems["ieee14"], suite_roots = load_ieee14(args.suite, args.states)
    if "ieee118" in args.systems:
        systems["ieee118"] = load_ieee118(args.ieee118_dir, args.states)
    log(stage="loaded", systems={k: {"states": len(v.states), "meters": len(v.meters), "parameters": len(v.parameters)}
                                 for k, v in systems.items()})
    summary: dict[str, Any] = {"contract": CONTRACT, "seed": args.seed, "argv": list(argv or sys.argv[1:]), "config": {
        "residual_flag": RESIDUAL_FLAG, "lambda_flag": args.lambda_flag, "link_distance": LINK_DISTANCE,
        "closure_hops": CLOSURE_HOPS, "band": BAND, "masking_keep": MASKING_KEEP, "chi2_alpha": CHI2_ALPHA,
        "floors": {"branch": BRANCH_FLOOR, "meter": METER_FLOOR}, "draws": args.draws, "bank": args.bank,
        "states": args.states}, "systems": {}}

    def flush() -> None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    for name, system in systems.items():
        runner = Runner((system, args.bank, args.seed, args.lambda_flag), args.workers)
        sysout: dict[str, Any] = {"states": [s[2] for s in system.states],
                                  "pools": {"meters": len(system.meters), "parameters": len(system.parameters),
                                            "outage_lines": system.outage_lines,
                                            "outage_atoms_in_band": [sum(BAND[0] <= a.natural_ratio <= BAND[1] for a in pool)
                                                                     for pool in _OUTAGES],
                                            "outage_atoms": [len(pool) for pool in _OUTAGES],
                                            "outage_natural_ratio_q10_50_90": _q(a.natural_ratio for pool in _OUTAGES
                                                                                 for a in pool),
                                            "outage_share_below_band": _share(a.natural_ratio < BAND[0] for pool in _OUTAGES
                                                                              for a in pool),
                                            "outage_share_above_band": _share(a.natural_ratio > BAND[1] for pool in _OUTAGES
                                                                              for a in pool)}}
        summary["systems"][name] = sysout
        sysout["curves"] = _system_curves({"seed": args.seed})
        log(stage="curves", system=name)
        fp_tasks = [{"kind": k, "ratio": r} for k in KINDS for r in SWEEP_RATIOS]
        fps = [x for chunk in runner.map(_footprint_task, fp_tasks) for x in chunk]
        _write_jsonl(out / f"footprints_{name}.jsonl", fps)
        sysout["footprints"] = summarize_footprints(fps)
        log(stage="footprints", system=name, n=len(fps))
        flush()
        nl = int(system.case["branch"].shape[0])
        sets: dict[str, list[dict[str, Any]]] = {
            "pairs_random": _tasks(args.seed + 1, args.pairs, "pairs_random", n_atoms=2, mode="matched",
                                   draws=args.draws, sensitivity=True),
            "triples_random": _tasks(args.seed + 2, args.triples, "triples_random", n_atoms=3, mode="matched",
                                     draws=args.draws, sensitivity=True),
            "pairs_meter_line": _tasks(args.seed + 3, args.pairs // 2, "pairs_meter_line", n_atoms=2, mode="matched",
                                       draws=args.draws, kind_set=("meter", "parameter")),
        }
        for ratio in (1.5, 3.0, 10.0):
            sets[f"pairs_ratio_{ratio}"] = _tasks(args.seed + 10 + int(ratio * 10), args.pairs // 3, f"pairs_ratio_{ratio}",
                                                  n_atoms=2, mode="matched", draws=args.draws, ratio=ratio)
        for b in DISTANCE_BINS:
            sets[f"pairs_distance_{b}"] = _tasks(args.seed + 100 + b, args.per_bin, f"pairs_distance_{b}", n_atoms=2,
                                                 mode="matched", draws=args.draws, distance_bin=b)
        for density in (1.0, 2.0, 4.0):
            rng = np.random.default_rng(args.seed + int(density * 100))
            counts = np.clip(rng.poisson(density * nl / 100.0, size=args.variable), 1, 12)
            for suffix, natural in (("", False), ("_natural_outages", True)):
                sets[f"variable_density_{density}{suffix}"] = [
                    {"seed": args.seed * 31 + int(density * 1000) * 10_000 + i, "n_atoms": int(c), "mode": "independent",
                     "draws": max(12, args.draws // 3), "set": f"variable_density_{density}{suffix}",
                     "natural_outages": natural} for i, c in enumerate(counts)]
        sysout["sets"] = {}
        strat = []
        for set_name, tasks in sets.items():
            recs = [r for r in runner.map(_combination, tasks) if r and "n_atoms" in r]
            if set_name.startswith("pairs_distance_"):
                strat += recs
                continue
            _write_jsonl(out / f"combos_{name}_{set_name}.jsonl", recs)
            sysout["sets"][set_name] = summarize_combinations(recs)
            if set_name.startswith("variable"):
                sysout["sets"][set_name]["atoms_per_root"] = _q(r["n_atoms"] for r in recs)
                sysout["sets"][set_name]["by_count"] = {
                    str(c): {"n": len(v), "atoms_isolated": _share(x for r in v for x in r["atom_isolated"]),
                             "atoms_isolated_noisy": _mean(x for r in v for x in r["noisy"]["alone"]),
                             "clusters_noiseless": _q((r["noiseless"]["n_clusters"] for r in v), (50,)),
                             "progress": _progress_summary(v)}
                    for c, v in sorted(_group(recs, lambda r: min(r["n_atoms"], 8)).items())}
            log(stage="set", system=name, set=set_name, n=len(recs))
            flush()
        _write_jsonl(out / f"combos_{name}_pairs_by_distance.jsonl", strat)
        sysout["sets"]["pairs_by_distance"] = summarize_combinations(strat)
        log(stage="set", system=name, set="pairs_by_distance", n=len(strat))
        flush()
        runner.close()

    if suite_roots and not args.skip_suite:
        tasks = [{"root": r, "draws": max(64, args.draws), "seed": args.seed + i} for i, r in enumerate(suite_roots)]
        runner = Runner((systems["ieee14"], 0, args.seed, args.lambda_flag), args.workers)
        recs = runner.map(_suite_root, tasks)
        runner.close()
        _write_jsonl(out / "suite14_natural.jsonl", recs)
        summary["suite14_natural"] = summarize_suite(recs)
        log(stage="suite", n=len(recs))
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)
    flush()
    log(stage="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
