"""Electrical relations of compound-error atoms: graph distance, bus degree, multiplier footprints (2026-10-28).

Two wrong lines are confusable when their measurement footprints overlap: the
WLS residual each leaves is nearly parallel, so the normalized Lagrange
multipliers of both lines rise together and a one-at-a-time correction of the
better-ranked line absorbs part of the other.  The overlap is read from the
covariance of the branch-parameter multipliers the deployed solve computes
(``tools.lagrangian_port``, ``ea``, columns ``[R_0, X_0, R_1, X_1, ...]``).
That covariance is the Gram matrix of the parameters' sensitivity columns
projected on the residual space, so the first canonical correlation of two
lines' (R, X) blocks is the cosine of the smallest angle between their
residual footprints: 1 when no snapshot can tell the two lines apart, 0 when
their footprints are orthogonal.

Distances are hop counts over the planning bus graph (every branch row,
transformers included), between the closest endpoints of the two elements: 0
when two lines share a bus.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: The deployed comparison trigger (``psse_env.oracle.expert_policy.DEFAULT_COMPARISON_DOMINANCE_BAND``):
#: a ranking dominant by less than this ratio is compared.
COMPARISON_BAND = 1.5


def bus_neighbours(branch: Any) -> dict[int, set[int]]:
    rows = np.asarray(branch, dtype=float)
    neighbours: dict[int, set[int]] = {}
    for row in rows:
        f, t = int(row[0]), int(row[1])
        neighbours.setdefault(f, set()).add(t)
        neighbours.setdefault(t, set()).add(f)
    return neighbours


def bus_distances(branch: Any) -> dict[int, dict[int, int]]:
    """All-pairs shortest path lengths (hops) over the bus graph of ``branch``."""
    neighbours = bus_neighbours(branch)
    table: dict[int, dict[int, int]] = {}
    for source in sorted(neighbours):
        distance = {source: 0}
        frontier = [source]
        while frontier:
            nxt = []
            for node in frontier:
                for other in neighbours[node]:
                    if other not in distance:
                        distance[other] = distance[node] + 1
                        nxt.append(other)
            frontier = nxt
        table[source] = distance
    return table


def line_hops(branch: Any, row_a: int, row_b: int, table: Mapping[int, Mapping[int, int]] | None = None) -> int:
    """Hops between the closest endpoints of two branch rows (0 when they share a bus)."""
    rows = np.asarray(branch, dtype=float)
    table = table if table is not None else bus_distances(rows)
    a = (int(rows[int(row_a), 0]), int(rows[int(row_a), 1]))
    b = (int(rows[int(row_b), 0]), int(rows[int(row_b), 1]))
    return min(int(table[x][y]) for x in a for y in b)


def bus_degree(branch: Any) -> dict[int, int]:
    """Branch rows incident to each bus (parallel circuits counted once each)."""
    degree: dict[int, int] = {}
    for row in np.asarray(branch, dtype=float):
        for bus in (int(row[0]), int(row[1])):
            degree[bus] = degree.get(bus, 0) + 1
    return degree


def line_degree(branch: Any, row0: int, degree: Mapping[int, int] | None = None) -> int:
    """The larger endpoint degree of one branch row."""
    rows = np.asarray(branch, dtype=float)
    degree = degree if degree is not None else bus_degree(rows)
    return max(int(degree[int(rows[int(row0), 0])]), int(degree[int(rows[int(row0), 1])]))


def multiplier_view(case: Any, z: Sequence[float], *, sigma: Sequence[float] | None = None,
                    exact_measurement_indices: Iterable[int] | None = None) -> dict[str, Any] | None:
    """The deployed solve's branch-parameter multipliers and their covariance, or None when it diverges."""
    from mcp_server.matpower_server import _load_python_case
    from tools.lagrangian_port import lagrangian_m_singlephase_details

    ppc = _load_python_case(str(case)) if isinstance(case, str) else copy.deepcopy(dict(case))
    options: dict[str, Any] = {}
    if sigma is not None:
        options["measurement_sigma"] = np.asarray(sigma, dtype=float)
    exact = [int(i) for i in (exact_measurement_indices or [])]
    if exact:
        options["exact_measurement_indices"] = exact
    try:
        details = lagrangian_m_singlephase_details(
            z=np.asarray(z, dtype=float), result=ppc, ind=0, bus_data=ppc["bus"], **options
        )
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None
    if not details.get("success"):
        return None
    return {
        "lambdaN": np.asarray(details["lambdaN"], dtype=float),
        "ea": np.asarray(details["ea"], dtype=float),
        "objective": float(details["wls_objective"]),
        "dof": int(details["dof"]),
    }


def line_scores(lambdaN: Sequence[float]) -> np.ndarray:
    """Per line, the larger absolute normalized multiplier of its R and X."""
    values = np.abs(np.asarray(lambdaN, dtype=float)).reshape(-1, 2)
    return values.max(axis=1)


def _whitener(block: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Columns that whiten a 2x2 covariance block (directions of no variance dropped)."""
    values, vectors = np.linalg.eigh(0.5 * (block + block.T))
    top = float(values.max()) if values.size else 0.0
    keep = values > tolerance * max(top, 1e-300)
    if not keep.any():
        return np.zeros((block.shape[0], 0))
    return vectors[:, keep] / np.sqrt(values[keep])[None, :]


def pair_overlap(ea: np.ndarray, row_a: int, row_b: int) -> dict[str, Any]:
    """How far two lines' residual footprints overlap (first canonical correlation of their R/X multipliers)."""
    ea = np.asarray(ea, dtype=float)
    ia, ib = [2 * int(row_a), 2 * int(row_a) + 1], [2 * int(row_b), 2 * int(row_b) + 1]
    whiten_a = _whitener(ea[np.ix_(ia, ia)])
    whiten_b = _whitener(ea[np.ix_(ib, ib)])
    if not whiten_a.shape[1] or not whiten_b.shape[1]:
        return {"overlap": None, "x_correlation": None, "max_abs_correlation": None}
    coupling = whiten_a.T @ ea[np.ix_(ia, ib)] @ whiten_b
    canonical = np.linalg.svd(coupling, compute_uv=False)

    def correlation(i: int, j: int) -> float | None:
        scale = float(ea[i, i] * ea[j, j])
        return float(ea[i, j] / np.sqrt(scale)) if scale > 0 else None

    pairs = [correlation(i, j) for i in ia for j in ib]
    finite = [abs(v) for v in pairs if v is not None]
    return {
        "overlap": float(min(1.0, canonical[0])) if canonical.size else None,
        "x_correlation": correlation(ia[1], ib[1]),
        "max_abs_correlation": float(max(finite)) if finite else None,
    }


def band_view(scores: Sequence[float], *, band: float = COMPARISON_BAND, rows_of_interest: Iterable[int] = ()) -> dict[str, Any]:
    """The lines whose multiplier lies within ``band`` of the top one, and where the named rows rank."""
    values = np.asarray(scores, dtype=float)
    order = [int(i) for i in np.argsort(-values, kind="stable")]
    top = float(values[order[0]]) if order else 0.0
    within = [row + 1 for row in order if top > 0.0 and values[row] > 0.0 and values[row] * float(band) >= top]
    position = {row: rank + 1 for rank, row in enumerate(order)}
    rows = [int(r) for r in rows_of_interest]
    return {
        "band": float(band),
        "top_score": top,
        "top_lines": [row + 1 for row in order[:5]],
        "band_count": len(within),
        "band_lines": within[:12],
        "line_ranks": {str(row + 1): position.get(row) for row in rows},
        "line_scores": {str(row + 1): float(values[row]) for row in rows},
    }


__all__ = [
    "COMPARISON_BAND",
    "band_view",
    "bus_degree",
    "bus_distances",
    "bus_neighbours",
    "line_degree",
    "line_hops",
    "line_scores",
    "multiplier_view",
    "pair_overlap",
]
