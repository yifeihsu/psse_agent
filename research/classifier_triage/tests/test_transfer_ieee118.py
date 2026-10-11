"""The IEEE 118 transfer study's inference-time interventions and the harmonic trace's reference bus."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from research.classifier_triage import features, transfer_ieee118 as transfer


def _payload(case: str, seed: int = 0, objective: float | None = None) -> dict:
    net = features.network(case)
    nb, nl = int(net["nb"]), int(net["nl"])
    rng = np.random.default_rng(seed)
    residual = rng.normal(0.0, 1.0, 3 * nb + 4 * nl)
    dof = 3 * nb + 4 * nl - (2 * nb - 1)
    return {"signed_normalized_residual": residual, "lambda_normalized": rng.normal(0.0, 1.0, 2 * nl),
            "objective": float(dof if objective is None else objective), "dof": dof}


def test_globals14_keeps_the_excess_objective_and_rescales_the_fractions():
    payload = _payload("case118", objective=863.0 + 400.0)
    payload["signed_normalized_residual"][:12] = 8.0
    graph = features.build_graph("case118", payload, "residual")
    out = transfer.globals14(graph, payload, nz=1098)
    threshold14 = chi2.ppf(0.99, transfer.IEEE14_DOF)
    assert out["u"][0] == pytest.approx(np.log((transfer.IEEE14_DOF + 400.0) / threshold14))
    assert out["u"][2] == pytest.approx(min(1.0, graph["u"][2] * 1098 / transfer.IEEE14_NZ))
    assert out["u"][1] == graph["u"][1] and out["u"][4] == graph["u"][4]
    assert np.array_equal(out["x"], graph["x"])


def test_subgraph_takes_a_connected_neighbourhood_with_every_inner_branch():
    payload = _payload("case118", seed=3)
    graph = features.build_graph("case118", payload, "residual")
    center = transfer._center_bus(graph, payload, features.network("case118"))
    sub = transfer.subgraph(graph, center)
    assert sub["x"].shape == (transfer.SUBGRAPH_BUSES, graph["x"].shape[1])
    assert sub["edge_index"].max() < transfer.SUBGRAPH_BUSES and sub["edge_attr"].shape[0] == sub["edge_index"].shape[1]
    # two directed edges per kept branch, branch ids contiguous from zero
    assert sub["edge_index"].shape[1] == 2 * (int(sub["edge_pair"].max()) + 1)
    assert sorted(set(sub["edge_pair"].tolist())) == list(range(int(sub["edge_pair"].max()) + 1))


def test_magnitude_cap_scales_residuals_multipliers_and_the_excess_objective():
    payload = _payload("case118", objective=2000.0)
    payload["signed_normalized_residual"][5] = -40.0
    capped = transfer.magnitude_capped(payload, cap=10.0)
    assert np.max(np.abs(capped["signed_normalized_residual"])) == pytest.approx(10.0)
    assert capped["objective"] == pytest.approx(payload["dof"] + 0.25 ** 2 * (2000.0 - payload["dof"]))
    untouched = transfer.magnitude_capped(_payload("case118"), cap=1e6)
    assert np.array_equal(untouched["signed_normalized_residual"], _payload("case118")["signed_normalized_residual"])


def test_static_clamp_stays_inside_the_ieee14_range():
    graph = features.build_graph("case118", _payload("case118"), "residual")
    ranges = transfer.static_ranges()
    out = transfer.static_clamped(graph, ranges)
    column = features.EDGE_FEATURES["residual"].index("b_pu")
    assert out["edge_attr"][:, column].max() <= ranges["edge"][1][2] + 1e-12
    residual_columns = [features.EDGE_FEATURES["residual"].index(name) for name in ("p_src.r", "lambda_r")]
    assert np.array_equal(out["edge_attr"][:, residual_columns], graph["edge_attr"][:, residual_columns])


def test_thresholds_hit_their_targets():
    score = np.asarray([0.1, 0.2, 0.3, 0.4, 0.9, 0.95, 0.97, 0.99])
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    assert transfer.recall_threshold(score, y, 1.0) == pytest.approx(0.9)
    assert transfer.recall_threshold(score, y, 0.75) == pytest.approx(0.95)
    threshold = transfer.false_rate_threshold(score, y, 0.25)
    assert np.mean(score[y == 0] >= threshold) <= 0.25


def test_harmonic_trace_grounds_the_case_reference_bus():
    from pypower.api import case14

    from Transmission.generate_hse_traces import build_trace

    ppc = case14()
    order = np.r_[np.arange(1, 14), 0]  # the reference bus moves to the last row
    bus = ppc["bus"][order].copy()
    renumber = {int(old): new + 1 for new, old in enumerate(bus[:, 0])}
    bus[:, 0] = np.arange(1, 15)
    branch = ppc["branch"].copy()
    branch[:, 0] = [renumber[int(b)] for b in branch[:, 0]]
    branch[:, 1] = [renumber[int(b)] for b in branch[:, 1]]
    trace = build_trace(4, 0.15, 11, bus=bus, branch=branch)
    reference_row = int(np.flatnonzero(bus[:, 1] == 3)[0])
    assert reference_row == 13
    for phasors in trace["harmonic_phasors"].values():
        assert phasors[reference_row]["V_complex_true"] == [0.0, 0.0]


def test_union_and_combined_globals_read_two_solves_as_one_system():
    from research.classifier_triage import transfer_ieee118_to_14 as reverse

    small, large = _payload("case14", seed=1, objective=180.0), _payload("case118", seed=2)
    g14 = features.build_graph("case14", small, "residual")
    g118 = features.build_graph("case118", large, "residual")
    joined = reverse.union(g14, g118, reverse.combined_globals(small, "case14", large))
    assert joined["x"].shape[0] == g14["x"].shape[0] + g118["x"].shape[0]
    assert joined["edge_index"].shape[1] == g14["edge_index"].shape[1] + g118["edge_index"].shape[1]
    # no edge crosses between the two networks, and branch ids stay contiguous
    assert joined["edge_index"][:, : g14["edge_index"].shape[1]].max() < g14["x"].shape[0]
    assert joined["edge_index"][:, g14["edge_index"].shape[1]:].min() >= g14["x"].shape[0]
    assert sorted(set(joined["edge_pair"].tolist())) == list(range(20 + 186))
    dof = small["dof"] + large["dof"]
    assert joined["u"][0] == pytest.approx(np.log((180.0 + large["objective"]) / chi2.ppf(0.99, dof)))
    kept = reverse.union(g14, g118)
    assert np.array_equal(kept["u"], g14["u"])


def test_globals118_is_the_inverse_rescaling_of_globals14():
    from research.classifier_triage import transfer_ieee118_to_14 as reverse

    payload = _payload("case14", objective=95.0 + 60.0)
    payload["signed_normalized_residual"][:6] = 9.0
    graph = features.build_graph("case14", payload, "residual")
    out = reverse.globals118(graph, payload, nz=122)
    assert out["u"][0] == pytest.approx(np.log((863 + 60.0) / chi2.ppf(0.99, 863)))
    assert out["u"][2] == pytest.approx(graph["u"][2] * 122 / 1098)
    back = transfer.globals14(out, {**payload, "dof": 863, "objective": 863 + 60.0}, nz=1098)
    assert back["u"][0] == pytest.approx(np.log((95 + 60.0) / chi2.ppf(0.99, 95)))
