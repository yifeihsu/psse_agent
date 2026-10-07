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


def test_llm_scoring_selects_test_rows_then_a_probe_sample():
    from research.classifier_triage import llm_score

    rows = [{"id": f"r{i}"} for i in range(4)] + [{"id": f"p{i}"} for i in range(6)]
    labels = [{"id": "r0", "kind": "root", "split": "test", "family": "hif"},
              {"id": "r1", "kind": "root", "split": "calibration", "family": "hif"},
              {"id": "r2", "kind": "mimic", "split": "test", "family": "mimic_flow_pair_same_sign"},
              {"id": "r3", "kind": "healthy", "split": "test", "family": "healthy_window"}]
    labels += [{"id": f"p{i}", "kind": "probe", "split": "test", "probe_kind": "meter",
                "family": "probe_meter_opendss" if i < 4 else "probe_meter_opf"} for i in range(6)]
    chosen = [row["id"] for row in llm_score.select_rows(rows, labels, probe_per_cell=2)]
    assert chosen == ["r0", "r2", "r3", "p0", "p1", "p4", "p5"]  # no calibration row, two probes per background
    # A thresholded reading needs the calibration rows too: they follow the test rows.
    chosen = [row["id"] for row in llm_score.select_rows(rows, labels, probe_per_cell=1, calibration=True)]
    assert chosen == ["r0", "r2", "r3", "r1", "p0", "p4"]


def test_candidate_probabilities_charge_the_token_where_the_calls_part():
    from research.classifier_triage import llm_score

    # Tokens 1, 2 open every call; "request" and "topology" share token 7 before they part.
    targets = {"request": [1, 2, 7, 30, 99], "topology": [1, 2, 7, 31, 99], "measurement": [1, 2, 8, 99], "parameter": [1, 2, 9, 99]}
    contexts = []

    def step(ids):
        contexts.append(list(ids))
        if ids[-1] == 2:   # after the shared opening
            return {7: np.log(0.6), 8: np.log(0.3), 9: np.log(0.1)}
        if ids[-1] == 7:   # inside the shared token of request and topology
            return {30: np.log(0.75), 31: np.log(0.25)}
        raise AssertionError(f"unexpected context {ids}")

    log_p = llm_score.candidate_log_probabilities([50, 51], targets, step)
    assert contexts == [[50, 51, 1, 2], [50, 51, 1, 2, 7]]  # one pass per parting point, none for a call that is alone
    assert np.exp(log_p["request"]) == pytest.approx(0.45) and np.exp(log_p["topology"]) == pytest.approx(0.15)
    assert np.exp(log_p["measurement"]) == pytest.approx(0.3) and np.exp(log_p["parameter"]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        llm_score.candidate_log_probabilities([], {"a": [1, 2], "b": [1, 2, 3]}, step)


def test_llm_probabilities_enter_the_benchmark_as_a_thresholded_score(tmp_path):
    def row(row_id, family, needs_aux, families, first, rule):
        return {"id": row_id, "family": family, "parent": f"parent_{row_id}", "needs_no_phasors_family": not needs_aux,
                "labels": {"needs_aux": needs_aux}, "rule_v3": rule,
                "triage": {"needs_aux": needs_aux, "families": families, "first": first}}

    cal = [row("c1", "hif", 1, [], None, True), row("c2", "hif", 1, [], None, True),
           row("c3", "measurement", 0, ["measurement"], "measurement", False)]
    test = [row("h1", "hif", 1, [], None, True), row("h2", "harmonic", 1, [], None, True),
            row("m1", "measurement", 0, ["measurement"], "measurement", False),
            row("p1", "parameter", 0, ["parameter"], "parameter", False)]
    probe = [{**row("q1", "probe_meter_opendss", 0, ["measurement"], "measurement", False), "probe_kind": "meter"},
             {**row("q2", "probe_meter_opf", 0, ["measurement"], "measurement", False), "probe_kind": "meter"},
             {**row("q3", "probe_meter_opf", 0, ["measurement"], "measurement", False), "probe_kind": "meter"}]

    def p(request, measurement, parameter, topology=0.0):
        return {"p": {"request": request, "measurement": measurement, "parameter": parameter, "topology": topology}, "seconds": 1.5}

    table = {"c1": p(0.9, 0.1, 0.0), "c2": p(0.4, 0.6, 0.0), "c3": p(0.1, 0.9, 0.0),
             "h1": p(0.8, 0.2, 0.0), "h2": p(0.3, 0.7, 0.0),       # below the calibration threshold of 0.4
             "m1": p(0.5, 0.4, 0.1), "p1": p(0.05, 0.25, 0.7),
             "q1": p(0.6, 0.4, 0.0), "q2": p(0.1, 0.9, 0.0)}       # q3 was not scored
    path = tmp_path / "probabilities.json"
    path.write_text(json.dumps({"rows": table, "meta": {"rows": 9}}))
    result = benchmark.evaluate_llm_probabilities("llm_p", {"calibration": cal, "test": test, "probe": probe}, path)
    assert result["request"]["thresholds"]["at_rule_recall"] == pytest.approx(0.4)   # the rule recalls both calibration positives
    assert result["request"]["learned_recall_at_rule_recall"]["point"] == pytest.approx(0.5)
    assert result["request"]["learned_false_rate_at_rule_recall"]["point"] == pytest.approx(0.5)  # m1 requested, p1 not
    assert result["order"]["hit"]["point"] == pytest.approx(1.0)       # the most probable balanced action is the true family
    assert result["probe_rows_scored"] == 2 and result["probe"]["meter"]["background_effect"]["point"] == pytest.approx(1.0)
    assert result["largest_probability_test"] == {"request": 2, "measurement": 1, "parameter": 1}
    path.write_text(json.dumps({"rows": {k: v for k, v in table.items() if k != "h2"}}))
    with pytest.raises(ValueError):
        benchmark.evaluate_llm_probabilities("llm_p", {"calibration": cal, "test": test, "probe": probe}, path)


def test_prompt_features_read_only_what_the_prompt_lists():
    from research.classifier_triage import prompt_control

    summary = {
        "top_residuals": [{"channel": "Pt", "channel_offset": 8, "index0": 90, "value": 20.05},
                          {"channel": "Pf", "channel_offset": 8, "index0": 50, "value": -6.5},
                          {"channel": "Pinj", "channel_offset": 3, "index0": 17, "value": 4.3},
                          {"_omitted_items": 2}],
        "top_lagrange": [{"from_bus": 4, "to_bus": 9, "lambda_index0": 17, "line_row0": 8, "parameter": "X", "value": -12.9}],
    }
    state = {"fresh_context_evidence": {"wls": {"anomaly_breadth": 0.05}},
             "last_tool_output": {"observable_metrics": {"chi_square_ratio": 4.0, "max_normalized_residual": 20.05,
                                                         "wls_summary": summary}}}
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": json.dumps({"state": state})}]}
    values = prompt_control.prompt_features(prompt_control.prompt_state(row))
    assert values["residuals_listed"] == 3 and values["residuals_omitted"] == 2 and values["multipliers_listed"] == 1
    assert values["r0_Pt"] == 1 and values["r0_offset"] == 8 and values["r1_sign"] == -1 and values["r3_log1p"] == 0
    assert values["flow_pair_listed"] == 1 and values["flow_pair_opposite_sign"] == 1 and values["flow_pair_same_sign"] == 0
    assert values["distinct_buses_listed"] == 1 and values["distinct_branches_listed"] == 1
    assert values["l0_is_x"] == 1 and values["l0_line"] == 8 and values["l0_sign"] == -1 and values["l1_line"] == -1
    assert values["top_residual_on_top_multiplier_line"] == 1
    assert values["chi_square_ratio_log"] == pytest.approx(np.log(4.0))
    # An empty summary (no residual at the listing threshold) still yields the same feature names.
    empty = prompt_control.prompt_features({"last_tool_output": {"observable_metrics": {"wls_summary": {}}}})
    assert set(empty) == set(values) and empty["residuals_listed"] == 0


class _CharacterProcessor:
    """One token per character, the tool-call format of the trainer's own test processor."""
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, **_kwargs):
        pieces = ["<tools>", json.dumps(tools, sort_keys=True), "</tools>"]
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                function = message["tool_calls"][0]["function"]
                pieces += ["<assistant>", f"<|tool_call|>call:{function['name']}",
                           json.dumps(function["arguments"], sort_keys=True), "<|end_tool_call|></assistant>"]
            else:
                pieces += [f"<{message['role']}>", str(message.get("content", "")), f"</{message['role']}>"]
        if add_generation_prompt:
            pieces.append("<assistant>")
        return "".join(pieces)

    def __call__(self, text=None, return_tensors=None, **_kwargs):
        ids = [ord(char) for char in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(value)) for value in ids)

    def convert_tokens_to_ids(self, _token):
        return None


class _NextCharacterModel:
    """Next-character probabilities that depend only on the text so far."""

    def __init__(self):
        self.embedding = torch.nn.Embedding(2, 2)
        self.contexts = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None):
        raise NotImplementedError

    def generate(self, input_ids=None, max_new_tokens=None, **_kwargs):
        from types import SimpleNamespace

        assert max_new_tokens == 1 and input_ids.shape[0] == 1
        text = "".join(chr(int(value)) for value in input_ids[0])
        self.contexts.append(text)
        probabilities = torch.full((1, 256), 1e-9)
        if text.endswith("call:get_"):
            for char, p in (("t", 0.6), ("m", 0.3), ("p", 0.1)):
                probabilities[0, ord(char)] = p
        elif text.endswith("call:get_t"):
            for char, p in (("h", 0.75), ("o", 0.25)):
                probabilities[0, ord(char)] = p
        else:
            raise AssertionError(f"unexpected context ...{text[-30:]}")
        return SimpleNamespace(logits=(torch.log(probabilities),), scores=None)


def test_the_first_action_scorer_reads_the_policy_prompt_and_the_parting_tokens(tmp_path):
    from types import SimpleNamespace

    from research.classifier_triage import llm_dataset, llm_score

    state = {"active_state_id": "active", "evidence_profile": llm_dataset.TRIAGE_PROFILE, "remaining_budget": 39}
    row = {"id": "a", "messages": [{"role": "system", "content": llm_dataset.triage_system_prompt()},
                                   {"role": "user", "content": json.dumps({"state": state}, sort_keys=True)}]}
    model = _NextCharacterModel()
    scorer = llm_score.FirstActionScorer(SimpleNamespace(model=model, processor=_CharacterProcessor(), model_id="fake"))
    output = tmp_path / "probabilities.json"
    saved = llm_score.probability_rows([row], scorer, output, meta=lambda: scorer.meta, log=lambda *_: None)["rows"]
    assert saved["a"]["p"] == pytest.approx({"request": 0.45, "topology": 0.15, "measurement": 0.3, "parameter": 0.1}, rel=1e-4)
    # Two forward passes: where the four calls part, and inside the token "three" and "topology" share.
    assert len(model.contexts) == 2 and model.contexts[0].endswith("<assistant><|tool_call|>call:get_")
    tools_block = model.contexts[0].split("</tools>")[0]
    assert llm_dataset.triage_system_prompt() in model.contexts[0]
    assert '"name": "get_three_phase_context"' in tools_block and '"name": "run_alternative_test"' not in tools_block
    assert scorer.meta["prompt_differs_from_training_render"] == 0 and scorer.meta["user_text_differs_from_row"] == 0
    written = json.loads(output.read_text())
    assert written["meta"]["rows"] == 1 and written["meta"]["forward_passes"] == 2
    assert "".join(written["meta"]["candidate_tokens"]["active"]["request"]).startswith(
        "<|tool_call|>call:get_three_phase_context")
    assert written["meta"]["training_renders_checked"] == 4
    # A second run reuses the finished row.
    llm_score.probability_rows([row], scorer, output, meta=lambda: scorer.meta, log=lambda *_: None)
    assert len(model.contexts) == 2


def test_prompt_variants_share_the_contract_and_the_tables_variant_explains_its_tables():
    from research.classifier_triage import llm_dataset, llm_score

    prompts = llm_dataset.triage_system_prompts()
    assert set(prompts) == set(llm_dataset.VARIANTS) == {"prompt_top5", "prompt_top10_signed", "prompt_tables"}
    assert prompts["prompt_top5"] == prompts["prompt_top10_signed"] == llm_dataset.triage_system_prompt()
    assert prompts["prompt_tables"] == prompts["prompt_top5"] + llm_dataset.TABLES_PROMPT_SENTENCE
    assert "bus_table" in prompts["prompt_tables"] and "branch_table" in prompts["prompt_tables"]
    row = {"id": "t", "messages": [{"role": "system", "content": prompts["prompt_tables"]},
                                   {"role": "user", "content": json.dumps({"state": {"active_state_id": "active"}})}]}
    assert llm_score.triage_variant_of(row) == "prompt_tables" and llm_score.state_of(row) == {"active_state_id": "active"}


def test_table_features_read_the_rows_and_the_end_buses():
    from research.classifier_triage import prompt_control

    summary = {
        "top_residuals": [], "top_lagrange": [],
        "bus_table": [{"bus": 4, "type": "PQ", "vm": 0.1, "p": 4.3, "q": -0.2},
                      {"bus": 9, "type": "PQ", "vm": -2.5, "p": 1.0, "q": 0.4},
                      {"bus": 7, "type": "PQ", "vm": 1.3, "p": None, "q": None}],
        "branch_table": [{"line": 9, "from": 4, "to": 9, "pf": 20.0, "qf": 0.1, "pt": 6.5, "qt": -0.6, "lr": -2.4, "lx": -12.9, "xfmr": True},
                         {"line": 7, "from": 4, "to": 5, "pf": 1.8, "qf": 0.6, "pt": -2.2, "qt": -2.3, "lr": 0.0, "lx": 1.0}],
        "omitted": {"buses": 2, "branches": 1},
    }
    state = {"last_tool_output": {"observable_metrics": {"chi_square_ratio": 4.0, "max_normalized_residual": 20.0, "wls_summary": summary}}}
    values = prompt_control.prompt_features(state)
    assert values["table_buses"] == 3 and values["table_branches"] == 2 and values["table_branches_omitted"] == 1
    assert values["bt0_same_sign_p"] == 1 and values["bt0_opposite_sign_p"] == 0 and values["bt0_xfmr"] == 1
    assert values["bt1_same_sign_p"] == 0 and values["bt1_opposite_sign_p"] == 0  # the smaller end is below two sigma
    assert values["bt0_from_p"] == pytest.approx(np.log1p(4.3)) and values["bt0_to_vm"] == pytest.approx(-np.log1p(2.5))
    assert values["bt1_to_listed"] == 0 and values["bt1_to_vm"] == 0.0            # bus 5 is not in the table
    assert values["bb2_zero_injection"] == 1 and values["bb2_p"] == 0.0 and values["bb0_PQ"] == 1
    assert values["bt5_pf"] == 0.0 and values["bb5_vm"] == 0.0                     # empty slots
    without = prompt_control.prompt_features({"last_tool_output": {"observable_metrics": {"wls_summary": {}}}})
    assert "bt0_pf" not in without
