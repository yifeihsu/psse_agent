"""Graph features of one alarmed state, computed from the operator's balanced WLS solve.

One node per bus row and two directed edges per branch row, as in
``research.gnn_screen``, but the evidence is the operator's own WLS result
(``data.wls_payload``) and the default view is residual-only:

* ``residual`` (default): signed normalized residuals of every channel,
  the normalized Lagrange multipliers of each branch's R and X, the configured
  network (per-unit branch parameters, taps, status, bus types, shunts, degree)
  and five global statistics.  No observed or fitted value, no estimated
  state, no sigma: those carry the operating point and the corpus format, which
  is where the family cues of September lived, and they tie a model to one
  network's operating range.
* ``values``: the same plus observed and fitted values, the estimated voltage
  and the angle difference across each branch.  Kept as an ablation that
  measures what those inputs add.

Two format cues are neutralized in both views.  The topology roots treat the
zero-injection bus as exact rows (residual identically zero) while every
other root measures it with noise, so the injection residuals of
zero-injection buses are zeroed on every row; sigmas are never read.

``llm_visible_features`` is the information-matched control for the LLM arm:
what the agent's prompt shows of a WLS solve (the five largest normalized
residuals with channel type and magnitude, the five largest signed
multipliers with their parameter, the chi-square ratio), as a flat vector.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from scipy.stats import chi2

BUS_TYPE, PD, QD, GS, BS = 1, 2, 3, 4, 5
F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS = 0, 1, 2, 3, 4, 8, 9, 10
GEN_BUS, GEN_STATUS = 0, 7
VIEWS = ("residual", "values")
RESIDUAL_FLAG = 3.0
CHI2_ALPHA = 0.01
BLOCKS = ("Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt")
LLM_TOP_K = 5

NODE_FEATURES = {
    "residual": ("vm.r", "vm.flag", "pinj.r", "pinj.flag", "qinj.r", "qinj.flag", "bus_pq", "bus_pv", "bus_ref",
                 "zero_injection", "log1p_degree", "g_shunt_pu", "b_shunt_pu"),
}
NODE_FEATURES["values"] = NODE_FEATURES["residual"] + (
    "vm.z", "vm.h", "pinj.z", "pinj.h", "qinj.z", "qinj.h", "vm_estimated")
EDGE_FEATURES = {
    "residual": ("p_src.r", "p_src.flag", "q_src.r", "q_src.flag", "p_dst.r", "p_dst.flag", "q_dst.r", "q_dst.flag",
                 "lambda_r", "lambda_r.flag", "lambda_x", "lambda_x.flag", "r_pu", "x_pu", "b_pu", "tap",
                 "cos_shift", "sin_shift", "status", "is_transformer", "orientation"),
}
EDGE_FEATURES["values"] = EDGE_FEATURES["residual"] + (
    "p_src.z", "p_src.h", "q_src.z", "q_src.h", "p_dst.z", "p_dst.h", "q_dst.z", "q_dst.h",
    "cos_angle_difference", "sin_angle_difference")
GLOBAL_FEATURES = ("log_chi_square_ratio", "log1p_max_residual", "fraction_residual_gt3", "fraction_residual_gt4",
                   "log1p_max_multiplier")


def dims(view: str) -> tuple[int, int, int]:
    return len(NODE_FEATURES[view]), len(EDGE_FEATURES[view]), len(GLOBAL_FEATURES)


def signed_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


@lru_cache(maxsize=64)
def network(case_name: str) -> dict[str, Any]:
    """The configured network of a case by name (cached; see ``network_of``)."""
    from mcp_server.matpower_server import _load_python_case

    return network_of(_load_python_case(case_name))


def network_of(case: Mapping[str, Any]) -> dict[str, Any]:
    """The configured network of a loaded case: topology, per-unit branch data, bus types, zero-injection buses.

    The provider calls this with the operator's current case (a corrected
    model has its own parameters and status), the offline studies through
    ``network`` by name.
    """
    bus = np.asarray(case["bus"], dtype=float)
    branch = np.asarray(case["branch"], dtype=float)
    gen = np.asarray(case.get("gen", np.empty((0, 8))), dtype=float)
    base = float(case["baseMVA"])
    nb, nl = bus.shape[0], branch.shape[0]
    row_of = {int(bus_id): row for row, bus_id in enumerate(bus[:, 0].astype(int))}
    source = np.asarray([row_of[int(b)] for b in branch[:, F_BUS]], dtype=np.int64)
    destination = np.asarray([row_of[int(b)] for b in branch[:, T_BUS]], dtype=np.int64)
    generating = set()
    if gen.size:
        status = gen[:, GEN_STATUS] if gen.shape[1] > GEN_STATUS else np.ones(gen.shape[0])
        generating = {row_of[int(b)] for b, s in zip(gen[:, GEN_BUS], status) if s > 0 and int(b) in row_of}
    zero_injection = np.asarray([
        bus[row, PD] == 0.0 and bus[row, QD] == 0.0 and bus[row, GS] == 0.0 and bus[row, BS] == 0.0 and row not in generating
        for row in range(nb)
    ])
    degree = np.bincount(np.r_[source, destination], minlength=nb)
    tap = np.where(branch[:, TAP] == 0.0, 1.0, branch[:, TAP])
    shift = np.deg2rad(branch[:, SHIFT])
    transformer = (branch[:, TAP] != 0.0) | (branch[:, SHIFT] != 0.0)
    if bus.shape[1] > 9:
        kv_from, kv_to = bus[source, 9], bus[destination, 9]
        transformer = transformer | ((kv_from > 0) & (kv_to > 0) & ~np.isclose(kv_from, kv_to))
    return {
        "nb": nb, "nl": nl, "source": source, "destination": destination, "zero_injection": zero_injection,
        "bus_static": np.column_stack((bus[:, BUS_TYPE] == 1, bus[:, BUS_TYPE] == 2, bus[:, BUS_TYPE] == 3, zero_injection,
                                       np.log1p(degree), bus[:, GS] / base, bus[:, BS] / base)).astype(float),
        "branch_static": np.column_stack((branch[:, BR_R], branch[:, BR_X], branch[:, BR_B], tap, np.cos(shift), np.sin(shift),
                                          branch[:, BR_STATUS], transformer)).astype(float),
    }


def neutral_residuals(payload: Mapping[str, Any], net: Mapping[str, Any]) -> np.ndarray:
    """Signed normalized residuals with the zero-injection buses' injection channels zeroed (format cue)."""
    nb = int(net["nb"])
    residual = np.array(payload["signed_normalized_residual"], dtype=float, copy=True)
    rows = np.flatnonzero(net["zero_injection"])
    residual[nb + rows] = 0.0
    residual[2 * nb + rows] = 0.0
    return residual


def _packets(residual: np.ndarray) -> np.ndarray:
    return np.column_stack((signed_log(residual), np.abs(residual) > RESIDUAL_FLAG)).astype(float)


def build_graph(case_name: str | Mapping[str, Any], payload: Mapping[str, Any], view: str = "residual") -> dict[str, np.ndarray]:
    """Node, directed-edge and global features of one state (see the module docstring).

    ``case_name`` is a case name (cached network) or a loaded case mapping.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown feature view {view!r}")
    net = network_of(case_name) if isinstance(case_name, Mapping) else network(str(case_name))
    nb, nl = int(net["nb"]), int(net["nl"])
    residual = neutral_residuals(payload, net)
    if residual.size != 3 * nb + 4 * nl:
        raise ValueError(f"expected {3 * nb + 4 * nl} channels, got {residual.size}")
    lambdas = np.asarray(payload["lambda_normalized"], dtype=float)
    lam_r, lam_x = (lambdas[0::2], lambdas[1::2]) if lambdas.size == 2 * nl else (np.zeros(nl), np.zeros(nl))
    blocks = [residual[:nb], residual[nb:2 * nb], residual[2 * nb:3 * nb]]
    flows = [residual[3 * nb + k * nl:3 * nb + (k + 1) * nl] for k in range(4)]  # Pf, Qf, Pt, Qt
    x = np.column_stack((*[_packets(block) for block in blocks], net["bus_static"]))
    lam = np.column_stack((_packets(lam_r), _packets(lam_x)))
    static = net["branch_static"]
    forward = np.column_stack((_packets(flows[0]), _packets(flows[1]), _packets(flows[2]), _packets(flows[3]),
                               lam, static, np.ones(nl)))
    reverse = np.column_stack((_packets(flows[2]), _packets(flows[3]), _packets(flows[0]), _packets(flows[1]),
                               lam, static, -np.ones(nl)))
    if view == "values":
        z = np.asarray(payload["z"], dtype=float)
        fitted = z - np.asarray(payload["raw_residual"], dtype=float)
        pair = lambda start, count: np.column_stack((z[start:start + count], fitted[start:start + count]))  # noqa: E731
        x = np.column_stack((x, pair(0, nb), pair(nb, nb), pair(2 * nb, nb), np.asarray(payload["vm"], dtype=float)))
        flow_values = [pair(3 * nb + k * nl, nl) for k in range(4)]
        delta = np.asarray(payload["theta"], dtype=float)[net["source"]] - np.asarray(payload["theta"], dtype=float)[net["destination"]]
        forward = np.column_stack((forward, flow_values[0], flow_values[1], flow_values[2], flow_values[3],
                                   np.cos(delta), np.sin(delta)))
        reverse = np.column_stack((reverse, flow_values[2], flow_values[3], flow_values[0], flow_values[1],
                                   np.cos(delta), -np.sin(delta)))
    edge_attr = np.empty((2 * nl, forward.shape[1]), dtype=float)
    edge_attr[0::2], edge_attr[1::2] = forward, reverse
    edge_index = np.empty((2, 2 * nl), dtype=np.int64)
    edge_index[:, 0::2] = np.stack((net["source"], net["destination"]))
    edge_index[:, 1::2] = np.stack((net["destination"], net["source"]))
    magnitude = np.abs(residual)
    threshold = float(chi2.ppf(1.0 - CHI2_ALPHA, max(int(payload["dof"]), 1)))
    u = np.asarray([
        np.log(max(float(payload["objective"]), 1e-9) / threshold), np.log1p(magnitude.max()),
        float(np.mean(magnitude > 3.0)), float(np.mean(magnitude > 4.0)),
        np.log1p(float(np.max(np.abs(lambdas))) if lambdas.size else 0.0),
    ], dtype=float)
    graph = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "u": u,
             "edge_pair": np.repeat(np.arange(nl, dtype=np.int64), 2)}
    if not all(np.all(np.isfinite(graph[name])) for name in ("x", "edge_attr", "u")):
        raise ValueError("nonfinite triage features")
    return graph


def flat_features(case_name: str, payload: Mapping[str, Any]) -> dict[str, float]:
    """Every channel's residual and every multiplier as one fixed-length vector.

    The whole WLS result with no graph structure: what a tabular model can do
    when nothing is hidden.  The vector length is the network's channel count,
    so a model on it cannot move to another network.
    """
    net = network(str(case_name))
    residual = signed_log(neutral_residuals(payload, net))
    lambdas = signed_log(np.asarray(payload["lambda_normalized"], dtype=float))
    vector = {f"r{index:03d}": float(value) for index, value in enumerate(residual)}
    vector.update({f"l{index:03d}": float(value) for index, value in enumerate(lambdas)})
    return vector


def llm_visible_features(case_name: str, payload: Mapping[str, Any], top_k: int = LLM_TOP_K, *,
                         signed: bool = False) -> dict[str, float]:
    """What the agent's prompt shows of a WLS solve, as a flat feature vector (information-matched control).

    The default is today's prompt: the five largest residual magnitudes and
    the five largest multipliers.  ``top_k`` and ``signed`` describe richer
    prompts (more listed residuals, with their signs) so the gain of showing
    the agent more can be measured before any prompt is changed.
    """
    net = network(str(case_name))
    nb, nl = int(net["nb"]), int(net["nl"])
    signed_residual = np.asarray(payload["signed_normalized_residual"], dtype=float)
    residual = np.abs(signed_residual)
    lambdas = np.asarray(payload["lambda_normalized"], dtype=float)
    threshold = float(chi2.ppf(1.0 - CHI2_ALPHA, max(int(payload["dof"]), 1)))
    features: dict[str, float] = {
        "chi_square_ratio_log": float(np.log(max(float(payload["objective"]), 1e-9) / threshold)),
        "max_residual_log1p": float(np.log1p(residual.max())),
    }
    edges = np.cumsum([0, nb, nb, nb, nl, nl, nl, nl])
    order = np.argsort(-residual)[:top_k]
    assets: list[tuple[int, int]] = []
    for position in range(top_k):
        index = int(order[position]) if position < order.size else -1
        block = int(np.searchsorted(edges, index, side="right") - 1) if index >= 0 else -1
        features[f"r{position}_log1p"] = float(np.log1p(residual[index])) if index >= 0 else 0.0
        if signed:
            features[f"r{position}_sign"] = float(np.sign(signed_residual[index])) if index >= 0 else 0.0
        for b, name in enumerate(BLOCKS):
            features[f"r{position}_{name}"] = float(block == b)
        assets.append((block, index - int(edges[block]) if block >= 0 else -1))
    # Both ends of one line among the listed residuals (the prompt shows the indices, so the pairing is visible).
    flow_assets = {}
    for block, asset in assets:
        if block >= 3:
            flow_assets.setdefault((asset, "P" if block in (3, 5) else "Q"), set()).add("f" if block in (3, 4) else "t")
    features["flow_pair_listed"] = float(any(ends == {"f", "t"} for ends in flow_assets.values()))
    features["distinct_buses_listed"] = float(len({asset for block, asset in assets if 0 <= block < 3}))
    features["distinct_branches_listed"] = float(len({asset for block, asset in assets if block >= 3}))
    lam_order = np.argsort(-np.abs(lambdas))[:top_k] if lambdas.size else np.asarray([], dtype=int)
    for position in range(top_k):
        index = int(lam_order[position]) if position < lam_order.size else -1
        value = float(lambdas[index]) if index >= 0 else 0.0
        features[f"l{position}_log1p"] = float(np.log1p(abs(value)))
        features[f"l{position}_sign"] = float(np.sign(value))
        features[f"l{position}_is_x"] = float(index >= 0 and index % 2 == 1)
    features["distinct_multiplier_branches"] = float(len({int(i) // 2 for i in lam_order}))
    top_l = float(np.abs(lambdas).max()) if lambdas.size else 0.0
    features["residual_over_multiplier_log"] = float(np.log((residual.max() + 1e-6) / (top_l + 1e-6)))
    return features
