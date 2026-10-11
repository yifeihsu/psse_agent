"""The ``residual_v2`` view: network-free globals, and the runtime's acceptance of a v2 export."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import chi2

torch = pytest.importorskip("torch")

from research.classifier_triage import features, runtime  # noqa: E402
from research.classifier_triage.model import TriageGNN  # noqa: E402


def _payload(case: str, excess: float, hot: int, value: float, *, with_chance: bool = False) -> dict:
    """``hot`` voltage channels at ``value`` plus, with ``with_chance``, the network's expected chance 3-sigma channels."""
    net = features.network(case)
    nb, nl = int(net["nb"]), int(net["nl"])
    dof = 3 * nb + 4 * nl - (2 * nb - 1)
    residual = np.zeros(3 * nb + 4 * nl)
    residual[:hot] = value  # voltage channels, never zeroed
    if with_chance:
        live = residual.size - 2 * int(np.count_nonzero(net["zero_injection"]))
        chance = int(round(features.CHANCE_RATE_GT3 * live))
        residual[nb:nb + chance] = 3.5  # a few chance exceedances among the injections
    return {"signed_normalized_residual": residual, "lambda_normalized": np.full(2 * nl, 0.5), "objective": float(dof + excess), "dof": dof}


def test_v2_globals_give_the_same_error_the_same_values_on_both_networks():
    # The same error on both networks: excess 120, five 8-sigma channels, plus what chance adds on each
    # network's channel count (0 channels on IEEE 14, 3 on IEEE 118).
    small, large = _payload("case14", 120.0, 5, 8.0, with_chance=True), _payload("case118", 120.0, 5, 8.0, with_chance=True)
    u14 = features.build_graph("case14", small, "residual_v2")["u"]
    u118 = features.build_graph("case118", large, "residual_v2")["u"]
    assert u14[0] == pytest.approx(u118[0]) and u14[0] == pytest.approx(np.log1p(120.0 / 70.0))
    assert u14[1] == pytest.approx(u118[1]) and u14[4] == pytest.approx(u118[4])
    # after the chance correction both read about five channels above 3 sigma
    assert abs(u14[2] - u118[2]) < 0.1 and abs(u14[3] - u118[3]) < 0.02
    assert u14[2] == pytest.approx(np.log1p(5 - features.CHANCE_RATE_GT3 * (122 - 2)))
    assert np.count_nonzero(np.abs(large["signed_normalized_residual"]) > 3.0) == 8
    legacy14 = features.build_graph("case14", small, "residual")["u"]
    legacy118 = features.build_graph("case118", large, "residual")["u"]
    assert legacy14[0] == pytest.approx(np.log((95 + 120.0) / chi2.ppf(0.99, 95)))
    # the legacy chi-square ratio and >3 sigma fraction read the same error very differently
    assert legacy14[0] - legacy118[0] > 0.4 and legacy14[2] / legacy118[2] > 4


def test_v2_shares_node_and_edge_features_with_the_residual_view():
    assert features.dims("residual_v2") == features.dims("residual")
    assert features.GLOBAL_FEATURES["residual_v2"][0] == "signed_log_excess_energy"
    quiet = _payload("case118", -30.0, 0, 0.0)
    u = features.build_graph("case118", quiet, "residual_v2")["u"]
    assert u[0] == pytest.approx(-np.log1p(30.0 / 70.0)) and u[2] == 0.0 and u[3] == 0.0
    small = features.build_graph("case14", _payload("case14", 10.0, 2, 4.0))
    v2 = features.build_graph("case14", _payload("case14", 10.0, 2, 4.0), "residual_v2")
    assert np.array_equal(small["x"], v2["x"]) and np.array_equal(small["edge_attr"], v2["edge_attr"])


def test_runtime_accepts_a_v2_export_and_rejects_the_values_view(tmp_path):
    node_dim, edge_dim, global_dim = features.dims("residual_v2")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.manual_seed(0)
    model = TriageGNN(node_dim, edge_dim, global_dim, hidden_dim=16, layers=1, dropout=0.0)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "view": "residual_v2", "seed": 0, "settings": {}},
               checkpoints / "seed_0.pt")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"meta": {"dataset_dir": "study"}, "results": [{
        "name": "gnn_residual", "train_rows": 10, "parameters": 1,
        "request": {"auc": {"point": 0.9}, "thresholds": {"at_rule_recall": 0.4, "rule_recall_calibration": 0.99}}}]}), encoding="utf-8")
    manifest = runtime.export_runtime(checkpoints, benchmark, tmp_path / "runtime")
    assert manifest["view"] == "residual_v2"
    classifier = runtime.load_triage_classifier(tmp_path / "runtime")
    payload = _payload("case118", 120.0, 5, 8.0)
    report = classifier.report("case118", {"success": True, "r": np.abs(payload["signed_normalized_residual"]).tolist(),
                                           "raw_residual": payload["signed_normalized_residual"].tolist(),
                                           "lambdaN": payload["lambda_normalized"].tolist(),
                                           "global_residual_sum": payload["objective"], "dof": payload["dof"]})
    assert report["status"] == "valid" and 0.0 <= report["request_score"] <= 1.0
    with pytest.raises(ValueError, match="residual views"):
        runtime.TriageClassifier(classifier.models, {**manifest, "view": "values"})
