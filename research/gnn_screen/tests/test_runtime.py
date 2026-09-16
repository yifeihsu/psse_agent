"""Parallel/cache identity and interrupted-training reproducibility checks."""
import copy

import numpy as np
import pytest
import torch

from research.gnn_screen.dataset import prepare_corpus
from research.gnn_screen.graph_builder import build_graph
from research.gnn_screen.model import collate_graphs
from research.gnn_screen.train import DEFAULT_CONFIG, _PreparedGraphs, load_checkpoint, train
from research.gnn_screen.tests.test_pipeline import row, snapshot, write_manifest
from research.gnn_screen.wls_features import default_measurement_sigma


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_parallel_cache_preserves_noise_and_partial_cache_order(tmp_path):
    case, z = snapshot()
    sigma = default_measurement_sigma(len(case["bus"]), len(case["branch"]))
    rows = [row(case, z, parent_id=f"parent-{i}", split="train", window_id=f"w{i}",
                measurement_kind="noiseless_mean", noise_seed=93, noise_replicates=2,
                measurement_sigma=sigma.tolist()) for i in range(3)]
    manifest = write_manifest(tmp_path / "manifest.jsonl", rows)
    reference = prepare_corpus(manifest)
    # Populate the middle row first: hit/miss mixtures must not reorder samples.
    partial = write_manifest(tmp_path / "partial.jsonl", rows[1:2])
    cache = tmp_path / "cache"
    prepare_corpus(partial, cache_dir=cache)
    actual = prepare_corpus(manifest, cache_dir=cache, workers=2)
    assert (cache / "graphs.sqlite3").is_file()
    assert not list(cache.glob("*.json"))
    assert actual.invalid == reference.invalid
    assert [s.window_id for s in actual.samples] == [s.window_id for s in reference.samples]
    for expected, result in zip(reference.samples, actual.samples):
        for field in ("x", "edge_index", "edge_attr", "u", "edge_pair"):
            np.testing.assert_array_equal(expected.graph[field], result.graph[field])


def test_prepared_batches_match_general_collator():
    case, z = snapshot()
    first, second = build_graph(case, z), build_graph(case, z + 1e-5)
    graphs = [first, second]
    prepared = _PreparedGraphs(graphs, "cpu")
    # Repeated sampled graphs must receive distinct graph/branch offsets.
    actual, expected = prepared.batch([1, 0, 1]), collate_graphs([second, first, second])
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def _smoke_manifest(tmp_path):
    # Arbitrary labels deliberately test runtime mechanics, not HIF accuracy.
    case, z = snapshot()
    sigma = default_measurement_sigma(len(case["bus"]), len(case["branch"]))
    rows = []
    for split in ("train", "validation"):
        for i in range(4):
            measured = z + np.random.default_rng(200 + i).normal(0, sigma)
            rows.append(row(case, measured, parent_id=f"{split}-{i}", split=split,
                            families=["hif"] if i % 2 else [], window_id=f"{split}-{i}"))
    return write_manifest(tmp_path / "manifest.jsonl", rows)


def test_epoch_resume_restores_rng_optimizer_and_seed_selection(tmp_path, monkeypatch):
    import research.gnn_screen.train as training_module

    manifest = _smoke_manifest(tmp_path)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"] = {"hidden_dim": 8, "layers": 1, "dropout": 0.15}
    config["training"].update(training_seeds=[2, 3], max_epochs=3,
                               early_stopping_patience=3, batch_size_graphs=2)
    cache = tmp_path / "cache"
    reference = train(manifest, tmp_path / "reference", config=config, cache_dir=cache)
    original_write = training_module.write_json

    def interrupt_after_epoch(path, value):
        original_write(path, value)
        if path.name == "training_progress.json":
            raise InterruptedError("simulated process interruption after durable epoch checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(training_module, "write_json", interrupt_after_epoch)
        with pytest.raises(InterruptedError):
            train(manifest, tmp_path / "resumed", config=config, cache_dir=cache)
    actual = train(manifest, tmp_path / "resumed", config=config, cache_dir=cache, resume=True)
    assert actual["selected"] == reference["selected"]
    for a, b in zip(actual["history"], reference["history"]):
        assert {k: v for k, v in a.items() if k != "epoch_seconds"} == {k: v for k, v in b.items() if k != "epoch_seconds"}
    assert len(actual["history"]) == len(reference["history"]) == 6
    for seed in (2, 3):
        a = load_checkpoint(tmp_path / "resumed" / f"seed_{seed}" / "checkpoint.pt")
        b = load_checkpoint(tmp_path / "reference" / f"seed_{seed}" / "checkpoint.pt")
        for name, value in a["model_state"].items():
            torch.testing.assert_close(value, b["model_state"][name], rtol=0, atol=0)
    altered = copy.deepcopy(config)
    altered["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="same physical corpus"):
        train(manifest, tmp_path / "resumed", config=altered, cache_dir=cache, resume=True)
