"""Leakage controls and real-WLS computational smoke; no screening-accuracy claim."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mcp_server.matpower_server import _load_python_case
from research.gnn_screen.calibrate import calibrate, validate_heldout_parents
from research.gnn_screen.dataset import (MEASUREMENT_CONVENTION, Sample, assign_parent_splits,
    jsonable, load_manifest, make_labels, prepare_corpus)
from research.gnn_screen.evaluate import evaluate, grouped_rate
from research.gnn_screen.graph_builder import FeatureScaler, build_graph
from research.gnn_screen.model import WLSScreenGNN
from research.gnn_screen.train import DEFAULT_CONFIG, empirical_threshold, load_trained_model, predict_samples, train
from research.gnn_screen.wls_features import configured_case, default_measurement_sigma, state_measurements_and_jacobian


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def snapshot():
    case = _load_python_case("case14")
    clean = configured_case(case)
    z, _ = state_measurements_and_jacobian(clean, np.deg2rad(case["bus"][:, 8]), case["bus"][:, 7])
    return case, z


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(jsonable(row)) for row in rows) + "\n", encoding="utf-8")
    return path


def row(case, z, **kwargs):
    return {"case": jsonable(case), "z": z.tolist(), "parent_id": "explicit-parent",
            "families": [], "measurement_convention": MEASUREMENT_CONVENTION, **kwargs}


def test_unknown_labels_not_negative_and_weak_labels_preserved():
    partial = make_labels({"hif": None, "unbalance": 0, "parameter": 1})
    assert partial["phase_mask"] == 0
    assert partial["anomaly"] == 1 and partial["anomaly_mask"] == 1
    assert partial["family_mask"] == [0, 1, 0, 1, 0]
    assert make_labels(["hif"])["phase"] == 1
    with pytest.raises(ValueError, match="harmonic"):
        make_labels(["harmonic"])


def test_parent_splits_prevent_paired_overlay_and_noise_leakage():
    rows = [{"parent_id": f"p{i}", "severity": "weak"} for i in range(12) for _ in range(3)]
    mapping = assign_parent_splits(rows, seed=7)
    assert set(mapping.values()) == {"train", "validation", "calibration", "test"}
    assert all(r["split"] == mapping[r["parent_id"]] for r in rows)
    with pytest.raises(ValueError, match="leakage"):
        assign_parent_splits([{"parent_id": "same", "split": "train"}, {"parent_id": "same", "split": "test"}])
    with pytest.raises(ValueError, match="every row"):
        assign_parent_splits([{"parent_id": "p", "split": "train"}, {"parent_id": "q"}])


def test_manifest_rejects_hidden_input_and_undeclared_convention(tmp_path):
    case, z = snapshot()
    manifest = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="unexpected fields"):
        load_manifest(write_manifest(manifest, [row(case, z, three_phase_currents=[1, 2, 3])]))
    with pytest.raises(ValueError, match="convention"):
        load_manifest(write_manifest(manifest, [row(case, z, measurement_convention="legacy")]))
    absent = row(case, z)
    del absent["parent_id"]
    with pytest.raises(ValueError, match="parent_id"):
        load_manifest(write_manifest(manifest, [absent]))


def test_fresh_noise_reruns_current_wls_and_cache_keys_include_covariance_settings(tmp_path):
    case, z = snapshot()
    sigma = default_measurement_sigma(len(case["bus"]), len(case["branch"]))
    manifest = tmp_path / "noise.jsonl"
    source = row(case, z, split="train", measurement_kind="noiseless_mean", noise_replicates=2,
                 noise_seed=71, measurement_sigma=sigma.tolist(), severity="weak", families=["hif"])
    calls = []
    def spy(configured, measurements, **kwargs):
        calls.append(measurements.copy())
        return build_graph(configured, measurements, **kwargs)
    write_manifest(manifest, [source])
    corpus = prepare_corpus(manifest, cache_dir=tmp_path / "cache", graph_builder=spy)
    assert len(calls) == 2 and len(corpus.samples) == 2
    assert not np.array_equal(calls[0], calls[1])
    assert all(s.labels["phase"] == 1 and s.severity == "weak" for s in corpus.samples)
    assert all("families" not in s.graph["metadata"] for s in corpus.samples)
    prepare_corpus(manifest, cache_dir=tmp_path / "cache", graph_builder=spy)
    assert len(calls) == 2
    # Labels do not affect either random noise or graph caching.
    source["families"] = []
    write_manifest(manifest, [source])
    prepare_corpus(manifest, cache_dir=tmp_path / "cache", graph_builder=spy)
    assert len(calls) == 2
    source["solver_settings"] = {"tol": 1e-9}
    write_manifest(manifest, [source])
    prepare_corpus(manifest, cache_dir=tmp_path / "cache", graph_builder=spy)
    assert len(calls) == 4
    source["measurement_sigma"] = (sigma * 1.1).tolist()
    write_manifest(manifest, [source])
    prepare_corpus(manifest, cache_dir=tmp_path / "cache", graph_builder=spy)
    assert len(calls) == 6


def test_cannot_double_noise_observed_snapshot(tmp_path):
    case, z = snapshot()
    path = write_manifest(tmp_path / "bad.jsonl", [row(case, z, split="train", noise_replicates=2)])
    with pytest.raises(ValueError, match="noiseless_mean"):
        prepare_corpus(path)


def test_independent_calibration_and_test_parent_guards():
    checkpoint = {"parent_splits": {"trained": "train", "cal": "calibration", "test": "test"}}
    with pytest.raises(ValueError, match="assigned train"):
        validate_heldout_parents({"trained": "calibration"}, checkpoint, "calibration")
    with pytest.raises(ValueError, match="allow-new-parents"):
        validate_heldout_parents({"new": "test"}, checkpoint, "test")
    assert validate_heldout_parents({"new": "test"}, checkpoint, "test", allow_new_parents=True) == ["new"]


def test_quantile_ties_and_parent_bootstrap():
    assert empirical_threshold([0.1, 0.2, 0.2], 0.01) == 0.2
    assert not any(score > empirical_threshold([0.2] * 4, 0.01) for score in [0.2] * 4)
    one_parent = grouped_rate([{"parent_id": "same"}] * 100, [True] * 100)
    assert one_parent["parent_bootstrap_ci95"] is None
    two_parents = grouped_rate([{"parent_id": "a"}, {"parent_id": "b"}], [True, False], bootstrap_replicates=100)
    assert two_parents["rate"] == 0.5 and two_parents["parents"] == 2


def test_wls_comparator_reproduces_chi_square_only_and_optional_local_gate():
    case, z = snapshot()
    graph = build_graph(case, z)
    # Boundary fixture: the global statistic is quiet, local rN equals four.
    graph["u"][0] = 0.0
    graph["u"][1] = 4.0
    scaler = FeatureScaler().fit([graph])
    model = WLSScreenGNN(hidden_dim=8, layers=1, dropout=0.0)
    sample = Sample(graph, make_labels([]), "parent", "test", "healthy", "window")
    chi_only = predict_samples(model, scaler, [sample], wls_normalized_threshold=None)
    with_local = predict_samples(model, scaler, [sample], wls_normalized_threshold=4.0)
    assert chi_only[0]["wls_alarm"] is False
    assert with_local[0]["wls_alarm"] is True
    assert chi_only[0]["phase_score"] == with_local[0]["phase_score"]


def test_real_wls_train_calibrate_evaluate_roundtrip(tmp_path):
    # Physical operating states vary; positive family labels here are arbitrary
    # harness labels, not simulated HIFs. This verifies plumbing only.
    case, _ = snapshot()
    clean = configured_case(case)
    sigma = default_measurement_sigma(len(case["bus"]), len(case["branch"]))
    rows = []
    for group, split in enumerate(("train", "validation", "calibration", "test")):
        for parent in range(2):
            vm = case["bus"][:, 7] + 0.002 * (2 * group + parent)
            mean, _ = state_measurements_and_jacobian(clean, np.deg2rad(case["bus"][:, 8]), vm)
            for positive in (False, True):
                measurements = mean + np.random.default_rng(1000 + 10 * group + 2 * parent + positive).normal(0, sigma)
                rows.append(row(case, measurements, parent_id=f"{split}-parent-{parent}", split=split,
                    families=["hif"] if positive else [], severity="weak" if positive else "healthy",
                    offline_metadata={"synthetic_smoke_only": True}))
    manifest = write_manifest(tmp_path / "smoke.jsonl", rows)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"] = {"hidden_dim": 8, "layers": 1, "dropout": 0.0}
    config["training"].update(training_seeds=[0], max_epochs=2, batch_size_graphs=4, early_stopping_patience=2)
    cache = tmp_path / "cache"
    report = train(manifest, tmp_path / "trained", config=config, cache_dir=cache)
    assert report["trained_family_mask"] == {"hif": True, "unbalance": False, "measurement": False, "parameter": False, "topology": False}
    checkpoint_path = Path(report["checkpoint"])
    _, scaler, checkpoint = load_trained_model(checkpoint_path)
    assert scaler.fitted and checkpoint["selected_seed"] == 0
    calibration_path = tmp_path / "calibration.json"
    calibration = calibrate(manifest, checkpoint_path, calibration_path, cache_dir=cache, allow_new_parents=True,
                            wls_normalized_threshold=None)
    assert calibration["healthy_count"] == 2
    assert calibration["threshold_comparison"] == ">"
    assert calibration["calibration_kind"] == "source_heldout_healthy"
    assert calibration["wls_comparator"]["normalized_residual_threshold"] is None
    assert calibration["wls_comparator"]["rule"] == "J >= chi2 threshold"
    evaluation = evaluate(manifest, checkpoint_path, calibration_path, tmp_path / "evaluation.json", cache_dir=cache,
                          bootstrap_replicates=25, allow_new_parents=True)
    assert evaluation["valid_test_windows"] == 4
    assert evaluation["wls_comparator_available"]
    assert evaluation["transfer_study"] == "source_parent_heldout"
    assert evaluation["union_healthy_false_trigger_rate"]["count"] == 2
    assert evaluation["by_family_and_severity"]["hif/weak"]["count"] == 2
    assert evaluation["predictions"][0]["family_scores"][1:] == [None] * 4
    assert set(evaluation["test_parent_ids"]).isdisjoint(calibration["calibration_parent_ids"])
    from research.gnn_screen.protocol_adapter import load_screen
    deployed = load_screen(checkpoint_path, calibration_path).screen(rows[-1]["case"], rows[-1]["z"])
    assert deployed["screen_status"] == "valid", deployed
    assert deployed["phase_score"] == pytest.approx(evaluation["predictions"][-1]["phase_score"], abs=1e-6)
    assert set(deployed["family_scores"]) == {"hif"}
    assert deployed["model_id"] == checkpoint["model_id"]
    calibration["checkpoint_sha256"] = "stale"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="different frozen checkpoint"):
        evaluate(manifest, checkpoint_path, calibration_path, tmp_path / "bad-eval.json", cache_dir=cache)
