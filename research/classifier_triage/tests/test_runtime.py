"""The exported triage classifier: export, load, and a report on a solve."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.classifier_triage import features, runtime  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES, TriageGNN  # noqa: E402


def _checkpoints(directory, seeds=(0, 1)):
    node_dim, edge_dim, global_dim = features.dims("residual")
    for seed in seeds:
        torch.manual_seed(seed)
        model = TriageGNN(node_dim, edge_dim, global_dim, hidden_dim=16, layers=1, dropout=0.0)
        torch.save({"state_dict": model.state_dict(), "config": model.config, "view": "residual", "seed": seed, "settings": {}},
                   directory / f"seed_{seed}.pt")


def _benchmark(path, threshold=0.3):
    path.write_text(json.dumps({"meta": {"dataset_dir": "study"}, "results": [{
        "name": "gnn_residual", "train_rows": 10, "parameters": 123,
        "request": {"auc": {"point": 0.99}, "thresholds": {"at_rule_recall": threshold, "rule_recall_calibration": 0.99}},
    }]}), encoding="utf-8")


def _wls_payload(case="case14", seed=0):
    from mcp_server.matpower_server import _load_python_case

    ppc = _load_python_case(case)
    nb, nl = ppc["bus"].shape[0], ppc["branch"].shape[0]
    rng = np.random.default_rng(seed)
    magnitude = np.abs(rng.normal(0.0, 1.0, 3 * nb + 4 * nl))
    raw = rng.normal(0.0, 0.01, 3 * nb + 4 * nl)
    return ppc, {"success": True, "r": magnitude.tolist(), "raw_residual": raw.tolist(), "lambdaN": rng.normal(0.0, 1.0, 2 * nl).tolist(),
                 "global_residual_sum": 300.0, "dof": 3 * nb + 4 * nl - (2 * nb - 1)}


def test_export_load_and_report(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    _checkpoints(checkpoints)
    benchmark = tmp_path / "benchmark.json"
    _benchmark(benchmark)
    manifest = runtime.export_runtime(checkpoints, benchmark, tmp_path / "runtime")
    assert manifest["contract"] == runtime.RUNTIME_CONTRACT and manifest["checkpoints"] == ["seed_0.pt", "seed_1.pt"]
    assert manifest["request_threshold"] == pytest.approx(0.3) and manifest["model_id"].startswith("triage_gnn:")
    saved = torch.load(tmp_path / "runtime" / "seed_0.pt", map_location="cpu", weights_only=False)
    assert all(value.dtype == torch.float16 for value in saved["state_dict"].values())

    classifier = runtime.load_triage_classifier(tmp_path / "runtime")
    assert classifier is runtime.load_triage_classifier(tmp_path / "runtime")   # cached
    ppc, payload = _wls_payload()
    report = classifier.report(ppc, payload)
    assert report["status"] == "valid" and report["method"] == "triage_gnn" and report["model_id"] == manifest["model_id"]
    assert 0.0 <= report["request_score"] <= 1.0 and report["request_threshold"] == pytest.approx(0.3)
    assert report["request_admitted"] == (report["request_score"] >= 0.3)
    assert report["first_family"] in FIRST_CLASSES and set(report["family_scores"]) == set(FAMILY_HEADS)
    assert sum(report["first_family_scores"].values()) == pytest.approx(1.0, abs=0.01)
    # The report is the mean of the seeds, and the same by name or by loaded case.
    by_name = classifier.report("case14", payload)
    assert by_name["request_score"] == report["request_score"]
    scores = [runtime.TriageClassifier([model], classifier.manifest).scores(ppc, runtime.payload_from_wls(payload))["needs_aux"]
              for model in classifier.models]
    assert report["request_score"] == pytest.approx(float(np.mean(scores)), abs=1e-4)


def test_report_refuses_a_payload_that_does_not_fit_the_case(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    _checkpoints(checkpoints, seeds=(0,))
    benchmark = tmp_path / "benchmark.json"
    _benchmark(benchmark)
    runtime.export_runtime(checkpoints, benchmark, tmp_path / "runtime")
    classifier = runtime.load_triage_classifier(tmp_path / "runtime")
    ppc, payload = _wls_payload()
    payload["r"] = payload["r"][:-1]
    with pytest.raises(ValueError):
        classifier.report(ppc, payload)


def test_the_tracked_model_loads_and_scores_a_solve():
    classifier = runtime.load_triage_classifier(runtime.DEFAULT_TRIAGE_CLASSIFIER)
    assert len(classifier.models) == 5 and classifier.threshold == pytest.approx(0.2782, abs=1e-3)
    ppc, payload = _wls_payload(seed=3)
    report = classifier.report(ppc, payload)
    assert report["status"] == "valid" and report["first_family"] in FIRST_CLASSES
