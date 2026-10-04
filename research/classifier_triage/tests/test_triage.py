"""Triage classifier package: labels, WLS graph features, the size-agnostic model, and the benchmark studies."""
from __future__ import annotations

import json

import numpy as np
import pytest

from research.classifier_triage import benchmark, data, features

torch = pytest.importorskip("torch")

from research.classifier_triage import train_gnn  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES, TriageGNN, collate  # noqa: E402


def _payload(case: str, seed: int = 0, *, scale: float = 1.0) -> dict:
    net = features.network(case)
    nb, nl = net["nb"], net["nl"]
    rng = np.random.default_rng(seed)
    count = 3 * nb + 4 * nl
    residual = rng.normal(0.0, 1.0, count)
    residual[nb + 2] = 9.0  # one loud injection residual
    return {
        "signed_normalized_residual": residual, "lambda_normalized": rng.normal(0.0, 1.0, 2 * nl),
        "theta": rng.normal(0.0, 0.1, nb) * scale, "vm": 1.0 + rng.normal(0.0, 0.02, nb) * scale,
        "raw_residual": residual * 0.01, "z": rng.normal(0.0, 1.0, count) * scale, "sigma": None, "exact": [],
        "objective": 300.0, "dof": count - (2 * nb - 1),
    }


# ----------------------------------------------------------------------- labels

def _record(truth: dict, objective: float, **extra) -> dict:
    base = {"measurement": [], "parameter": [], "topology": [], "hif": [], "unbalance": [], "harmonic": []}
    return {"truth": {**base, **truth}, "analysis": {"wls": {"chi_square_statistic": objective, "chi_square_dof": 95}}, **extra}


def test_triage_labels_follow_truth():
    meter = [{"index": 40}]
    assert data.triage_labels(_record({"measurement": meter}, 300.0)) == {
        "needs_aux": 0, "families": ["measurement"], "waveform": [], "first": "measurement"}
    hif = data.triage_labels(_record({"hif": [{"branch_row0": 2}], "measurement": meter}, 300.0))
    assert hif["needs_aux"] == 1 and hif["first"] == "measurement" and hif["waveform"] == ["hif"]
    healthy = data.triage_labels(_record({}, 150.0))
    assert healthy == {"needs_aux": 0, "families": [], "waveform": [], "first": None}


def test_the_first_family_of_a_mixed_root_is_the_larger_effect():
    mixed = _record({"measurement": [{"index": 40}], "parameter": [{"branch_row0": 3}]}, 800.0)
    # Removing the meter leaves 700; fixing the parameter leaves 200: the parameter explains more.
    children = [_record({}, 700.0, child_kind="remove_meter_overlay"), _record({}, 200.0, child_kind="fix_parameter")]
    assert data.triage_labels(mixed, children)["first"] == "parameter"
    children = [_record({}, 150.0, child_kind="remove_meter_overlay"), _record({}, 700.0, child_kind="fix_parameter")]
    assert data.triage_labels(mixed, children)["first"] == "measurement"
    # A topology overlay has no fixed-branch child: what is left without the meter is the branch effect.
    topo = _record({"measurement": [{"index": 40}], "topology": [{"branch_row0": 5}]}, 800.0)
    assert data.triage_labels(topo, [_record({}, 600.0, child_kind="remove_meter_overlay")])["first"] == "topology"
    assert data.triage_labels(topo, [_record({}, 120.0, child_kind="remove_meter_overlay")])["first"] == "measurement"


# --------------------------------------------------------------------- features

def test_graph_shapes_and_edge_orientation():
    net = features.network("case14")
    nb, nl = net["nb"], net["nl"]
    assert (nb, nl) == (14, 20)
    payload = _payload("case14")
    for view in features.VIEWS:
        graph = features.build_graph("case14", payload, view)
        node_dim, edge_dim, global_dim = features.dims(view)
        assert graph["x"].shape == (nb, node_dim) and graph["edge_attr"].shape == (2 * nl, edge_dim)
        assert graph["u"].shape == (global_dim,) and graph["edge_index"].shape == (2, 2 * nl)
    graph = features.build_graph("case14", payload)
    names = features.EDGE_FEATURES["residual"]
    src, dst = names.index("p_src.r"), names.index("p_dst.r")
    # The reverse edge of a branch swaps its terminal packets and flips the orientation flag.
    assert np.allclose(graph["edge_attr"][0::2, src], graph["edge_attr"][1::2, dst])
    assert np.allclose(graph["edge_attr"][0::2, dst], graph["edge_attr"][1::2, src])
    assert np.all(graph["edge_attr"][0::2, names.index("orientation")] == 1.0)
    assert np.all(graph["edge_attr"][1::2, names.index("orientation")] == -1.0)
    assert np.array_equal(graph["edge_index"][:, 0::2], graph["edge_index"][::-1, 1::2])


def test_the_zero_injection_bus_carries_no_injection_residual():
    net = features.network("case14")
    assert list(np.flatnonzero(net["zero_injection"])) == [6]  # bus 7
    payload = _payload("case14")
    payload["signed_normalized_residual"][14 + 6] = 25.0
    payload["signed_normalized_residual"][28 + 6] = -25.0
    graph = features.build_graph("case14", payload)
    names = features.NODE_FEATURES["residual"]
    for name in ("pinj.r", "pinj.flag", "qinj.r", "qinj.flag"):
        assert graph["x"][6, names.index(name)] == 0.0
    assert graph["x"][2, names.index("pinj.flag")] == 1.0  # the loud residual at another bus stays


def test_the_residual_view_ignores_operating_point_and_format():
    base = _payload("case14", seed=3)
    moved = {**_payload("case14", seed=3, scale=5.0), "sigma": np.full(122, 0.5), "exact": [20, 34]}
    assert not np.allclose(base["z"], moved["z"])
    first, second = features.build_graph("case14", base), features.build_graph("case14", moved)
    for key in ("x", "edge_attr", "u"):
        assert np.allclose(first[key], second[key]), key
    values_first, values_second = features.build_graph("case14", base, "values"), features.build_graph("case14", moved, "values")
    assert not np.allclose(values_first["x"], values_second["x"])


def test_llm_visible_features_show_only_the_listed_evidence():
    payload = _payload("case14", seed=1)
    visible = features.llm_visible_features("case14", payload)
    assert visible["r0_Pinj"] == 1.0 and abs(visible["r0_log1p"] - np.log1p(9.0)) < 1e-12
    # A change below the five largest residuals is invisible to the prompt view.
    quiet = {**payload, "signed_normalized_residual": payload["signed_normalized_residual"].copy()}
    smallest = int(np.argmin(np.abs(quiet["signed_normalized_residual"])))
    quiet["signed_normalized_residual"][smallest] = 0.0
    assert features.llm_visible_features("case14", quiet) == visible
    pair = {**payload, "signed_normalized_residual": np.zeros(122)}
    pair["signed_normalized_residual"][42 + 3] = 12.0   # Pf of branch 3
    pair["signed_normalized_residual"][42 + 40 + 3] = 11.0  # Pt of branch 3
    assert features.llm_visible_features("case14", pair)["flow_pair_listed"] == 1.0


# ------------------------------------------------------------------------ model

def test_one_model_runs_on_networks_of_different_size():
    node_dim, edge_dim, global_dim = features.dims("residual")
    model = TriageGNN(node_dim, edge_dim, global_dim, hidden_dim=16, layers=2).eval()
    graphs = [features.build_graph("case14", _payload("case14")), features.build_graph("case57", _payload("case57")),
              features.build_graph("case14", _payload("case14", seed=5))]
    with torch.no_grad():
        batched = model(collate(graphs))
        single = model(collate(graphs[1:2]))
    assert batched["needs_aux"].shape == (3,) and batched["family"].shape == (3, len(FAMILY_HEADS))
    assert batched["first"].shape == (3, len(FIRST_CLASSES))
    # Batching does not mix graphs: the 57-bus graph scores the same alone.
    assert torch.allclose(batched["needs_aux"][1], single["needs_aux"][0], atol=1e-5)
    assert torch.allclose(batched["family"][1], single["family"][0], atol=1e-5)


def _row(index: int, needs_aux: int, split: str) -> dict:
    payload = _payload("case14", seed=index)
    if needs_aux:  # a learnable cue: loud voltage residuals at three buses
        payload["signed_normalized_residual"][[1, 4, 8]] = 8.0
    truth = {"measurement": [] if needs_aux else [{"index": 16}], "parameter": [], "topology": [],
             "hif": [], "unbalance": [{"bus": 5}] if needs_aux else [], "harmonic": []}
    return {"id": f"row{index}", "kind": "root", "family": "three_phase_unbalance" if needs_aux else "measurement",
            "parent": f"parent{index}", "split": split, "payload": payload, "record": {"case": "case14", "truth": truth},
            "triage": {"needs_aux": needs_aux, "families": [] if needs_aux else ["measurement"],
                       "first": None if needs_aux else "measurement"}}


def test_training_learns_a_separable_cue_and_scores_every_evaluation_row():
    rows = [_row(i, i % 2, "train") for i in range(160)] + [_row(200 + i, i % 2, "calibration") for i in range(20)] \
        + [_row(300 + i, i % 2, "test") for i in range(20)]
    result = train_gnn.train_and_score(rows, view="residual", seeds=(0,), device="cpu",
                                       settings={"hidden_dim": 32, "layers": 2, "max_epochs": 60, "patience": 60,
                                                 "learning_rate": 3e-3}, log=lambda *_: None)
    assert result["ids"] == [f"row{200 + i}" for i in range(20)] + [f"row{300 + i}" for i in range(20)]
    labels = np.asarray([i % 2 for i in range(20)] * 2)
    assert result["needs_aux"].shape == (40,) and result["first"].shape == (40, 3)
    assert np.mean((result["needs_aux"] > 0.5) == labels) > 0.9
    assert result["train_rows"] + result["validation_rows"] == 160 and result["validation_rows"] > 0


# -------------------------------------------------------------------- benchmark

def test_screen_derived_feature_names():
    assert benchmark.screen_derived("s1_hif") and benchmark.screen_derived("meter_rank_gap_log")
    assert benchmark.screen_derived("ct1_any_explains") and benchmark.screen_derived("final_alarm")
    for name in ("lambda_top_gap_log", "r0_log1p", "chi_ratio_log", "flow_pair_same_sign", "top6_Vm"):
        assert not benchmark.screen_derived(name), name


def test_order_study_counts_true_family_picks():
    def row(family, families, first):
        return {"family": family, "parent": f"p{family}{first}", "triage": {"needs_aux": 0, "families": families, "first": first}}

    test = [row("measurement", ["measurement"], "measurement"), row("parameter", ["parameter"], "parameter"),
            row("measurement+parameter", ["measurement", "parameter"], "parameter"),
            {"family": "hif", "parent": "ph", "triage": {"needs_aux": 1, "families": [], "first": None}},
            {"family": "healthy_window", "parent": "pw", "triage": {"needs_aux": 0, "families": [], "first": None}}]
    study = benchmark.order_study(test, ["measurement", "topology", "measurement", "measurement", "measurement"])
    assert study["n"] == 3 and study["n_mixed"] == 1
    assert study["hit"]["point"] == pytest.approx(2 / 3)            # the parameter root got a topology pick
    assert study["meter_or_branch_hit"]["point"] == pytest.approx(1.0)  # topology still means "branch"
    assert study["oracle_first_agreement_mixed"] == 0.0             # a true family, but not the larger effect


def test_probe_study_reports_the_background_effect():
    probe = [{"family": "probe_meter_opendss", "parent": f"a{i}"} for i in range(10)] \
        + [{"family": "probe_meter_opf", "parent": f"b{i}"} for i in range(10)]
    flags = np.asarray([1] * 6 + [0] * 4 + [1] * 1 + [0] * 9)
    study = benchmark.probe_study(probe, flags)
    assert list(study) == ["meter"]
    study = study["meter"]
    assert study["opendss_request_rate"]["point"] == pytest.approx(0.6)
    assert study["opf_request_rate"]["point"] == pytest.approx(0.1)
    assert study["background_effect"]["point"] == pytest.approx(0.5)
    assert study["n_opendss"] == 10 and study["n_opf"] == 10


# ---------------------------------------------------------------------- LLM leg

def test_llm_scoring_reads_first_actions_and_resumes(tmp_path):
    from research.classifier_triage import llm_dataset, llm_score

    def prompt(row_id, tool_hint):
        state = {"active_state_id": "active", "hint": tool_hint}
        return {"id": row_id, "messages": [{"role": "system", "content": llm_dataset.triage_system_prompt()},
                                           {"role": "user", "content": json.dumps({"state": state})}]}

    rows = [prompt("a", "get_three_phase_context"), prompt("b", "get_parameter_context"),
            prompt("c", "finalize_diagnosis"), prompt("d", "boom")]
    calls = []

    def act(state):
        calls.append(state["hint"])
        if state["hint"] == "boom":
            raise ValueError("unparseable generation")
        return {"tool": state["hint"], "arguments": {}}

    output = tmp_path / "scores.json"
    result = llm_score.score_rows(rows, act, output, log=lambda *_: None)["rows"]
    assert {key: value["decision"] for key, value in result.items()} == {
        "a": "request", "b": "parameter", "c": "other", "d": "invalid"}
    assert result["d"]["error"].startswith("ValueError")
    # A second run reuses every finished row.
    llm_score.score_rows(rows, act, output, log=lambda *_: None)
    assert calls == ["get_three_phase_context", "get_parameter_context", "finalize_diagnosis", "boom"]
    # A row that does not carry the triage prompt is refused, not scored under another contract.
    foreign = {"id": "x", "messages": [{"role": "system", "content": "another prompt"}, rows[0]["messages"][1]]}
    with pytest.raises(ValueError):
        llm_score.state_of(foreign)


def test_llm_first_actions_enter_the_benchmark_as_hard_decisions(tmp_path):
    def row(row_id, family, needs_aux, families, first):
        return {"id": row_id, "family": family, "parent": f"parent_{row_id}", "needs_no_phasors_family": not needs_aux,
                "triage": {"needs_aux": needs_aux, "families": families, "first": first}}

    test = [row("h1", "hif", 1, [], None), row("h2", "harmonic", 1, [], None),
            row("m1", "measurement", 0, ["measurement"], "measurement"), row("p1", "parameter", 0, ["parameter"], "parameter")]
    probe = [{**row("q1", "probe_meter_opendss", 0, ["measurement"], "measurement"), "probe_kind": "meter"},
             {**row("q2", "probe_meter_opf", 0, ["measurement"], "measurement"), "probe_kind": "meter"}]
    decisions = {"h1": "request", "h2": "measurement", "m1": "measurement", "p1": "request", "q1": "request", "q2": "measurement"}
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"rows": {key: {"decision": value, "seconds": 2.0} for key, value in decisions.items()}}))
    result = benchmark.evaluate_llm("llm", {"calibration": [], "test": test, "probe": probe}, path)
    assert result["hard"] is True and result["seconds_per_decision"] == 2.0
    assert result["request"]["recall"]["point"] == pytest.approx(0.5)      # h2 was not requested
    assert result["request"]["unneeded"]["point"] == pytest.approx(0.5)    # p1 was
    assert result["order"]["hit"]["point"] == pytest.approx(0.5)           # m1 right, p1 gave no balanced pick
    assert result["probe"]["meter"]["background_effect"]["point"] == pytest.approx(1.0)
    assert result["decisions"] == {"request": 3, "measurement": 3}
