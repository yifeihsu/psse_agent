from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from research.gnn_screen.losses import screen_loss
from research.gnn_screen.model import WLSScreenGNN, collate_graphs


@pytest.fixture(autouse=True)
def deterministic_torch():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(241)
    yield
    torch.set_num_threads(previous_threads)


def make_graph(nodes=5, physical_edges=None, seed=7):
    rng = np.random.default_rng(seed)
    if physical_edges is None:
        physical_edges = [(i, i + 1) for i in range(nodes - 1)]
    directed = [edge for src, dst in physical_edges for edge in ((src, dst), (dst, src))]
    return {
        "x": rng.normal(size=(nodes, 27)).astype(np.float32),
        "edge_index": np.asarray(directed, dtype=np.int64).reshape(-1, 2).T,
        "edge_attr": rng.normal(size=(len(directed), 40)).astype(np.float32),
        "edge_pair": np.repeat(np.arange(len(physical_edges), dtype=np.int64), 2),
        "u": rng.normal(size=4).astype(np.float32),
        "metadata": {"bus_ids": [f"bus-{i}" for i in range(nodes)]},
    }


def assert_same_outputs(actual, expected, index=None):
    for key in actual:
        value = actual[key] if index is None else actual[key][index:index + 1]
        torch.testing.assert_close(value, expected[key], rtol=2e-5, atol=2e-6)


def test_default_architecture_and_variable_graph_sizes():
    model = WLSScreenGNN().eval()
    assert len(model.blocks) == 3
    assert model.decoder[0].in_features == 516
    assert model.decoder[0].out_features == 128
    assert model.decoder[3].out_features == 64
    outputs = model(collate_graphs([make_graph(14), make_graph(57)]))
    assert outputs["phase_screen_logit"].shape == (2,)
    assert outputs["anomaly_logit"].shape == (2,)
    assert outputs["family_logits"].shape == (2, 5)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_bus_directed_edge_and_physical_branch_permutations():
    original = make_graph(7, [(0, 1), (0, 1), (1, 3), (3, 6), (4, 6)])
    permuted = copy.deepcopy(original)
    # New node index -> old node index; map the edge endpoints with its inverse.
    node_order = np.array([6, 4, 1, 3, 5, 2, 0])
    node_inverse = np.argsort(node_order)
    edge_order = np.array([7, 2, 5, 8, 0, 9, 1, 3, 6, 4])
    branch_relabel = np.array([3, 1, 4, 0, 2])
    permuted["x"] = original["x"][node_order]
    permuted["edge_index"] = node_inverse[original["edge_index"][:, edge_order]]
    permuted["edge_attr"] = original["edge_attr"][edge_order]
    permuted["edge_pair"] = branch_relabel[original["edge_pair"][edge_order]]
    model = WLSScreenGNN().eval()
    with torch.no_grad():
        assert_same_outputs(model(collate_graphs([permuted])), model(collate_graphs([original])))


def test_single_batch_equivalence_and_cross_graph_isolation():
    graphs = [make_graph(14), make_graph(57, seed=2), make_graph(3, [], seed=3)]
    model = WLSScreenGNN().eval()
    with torch.no_grad():
        batched = model(collate_graphs(graphs))
        for i, graph in enumerate(graphs):
            assert_same_outputs(batched, model(collate_graphs([graph])), index=i)
        changed = copy.deepcopy(graphs)
        changed[1]["x"] *= 37
        changed[1]["edge_attr"] -= 42
        changed[1]["u"] += 100
        changed_outputs = model(collate_graphs(changed))
        for i in (0, 2):
            for key in batched:
                torch.testing.assert_close(changed_outputs[key][i], batched[key][i], rtol=0, atol=0)
        assert not torch.allclose(changed_outputs["family_logits"][1], batched["family_logits"][1])


def test_parallel_branches_and_metadata_stay_distinct():
    graph = make_graph(3, [(0, 1), (0, 1), (1, 2)])
    batch = collate_graphs([graph, graph])
    assert batch["edge_index"].shape == (2, 12)
    assert batch["branch_batch"].tolist() == [0, 0, 0, 1, 1, 1]
    assert batch["edge_pair"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert batch["metadata"] == [graph["metadata"], graph["metadata"]]
    assert torch.equal(batch["edge_index"][:, 6:], batch["edge_index"][:, :6] + 3)


def test_malformed_batches_reject_cross_graph_messages_and_pairs():
    model = WLSScreenGNN(hidden_dim=16, layers=1)
    batch = collate_graphs([make_graph(3), make_graph(4)])
    batch["edge_index"][1, 0] = 3
    with pytest.raises(ValueError, match="Cross-graph"):
        model(batch)
    graph = make_graph(3)
    graph["edge_pair"] = np.array([0, 1, 1, 1])
    with pytest.raises(ValueError, match="exactly two"):
        collate_graphs([graph])
    graph = make_graph(3)
    graph["edge_pair"] = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="opposite"):
        collate_graphs([graph])


def test_no_branch_graph_and_isolated_nodes_have_finite_backward():
    model = WLSScreenGNN(hidden_dim=16, layers=2)
    outputs = model(collate_graphs([make_graph(1, []), make_graph(4, [(0, 1)])]))
    sum(value.square().sum() for value in outputs.values()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mixed_family_masked_loss_matches_independent_bce():
    outputs = {
        "phase_screen_logit": torch.tensor([0.2, -0.8], requires_grad=True),
        "anomaly_logit": torch.tensor([0.1, 0.9], requires_grad=True),
        "family_logits": torch.tensor([[0.3, -0.5, 0.6, 0.8, -0.9], [-0.2, 0.1, 0.4, 0.8, 0.6]], requires_grad=True),
    }
    targets = {
        "phase": torch.tensor([1., 0.]),
        "anomaly": torch.ones(2),
        "family": torch.tensor([[1., 0., 1., 0., float("nan")], [0., 0., 0., 1., float("nan")]]),
        "family_mask": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
    }
    losses = screen_loss(outputs, targets)
    expected_family = F.binary_cross_entropy_with_logits(outputs["family_logits"][:, :4], targets["family"][:, :4])
    torch.testing.assert_close(losses["family_loss"], expected_family)
    expected = F.binary_cross_entropy_with_logits(outputs["phase_screen_logit"], targets["phase"])
    expected += .25 * F.binary_cross_entropy_with_logits(outputs["anomaly_logit"], targets["anomaly"])
    expected += .1 * expected_family
    torch.testing.assert_close(losses["loss"], expected)
    losses["loss"].backward()
    assert torch.equal(outputs["family_logits"].grad[:, 4], torch.zeros(2))
    assert outputs["family_logits"].grad[0, 0] < 0
    assert outputs["family_logits"].grad[0, 2] < 0
    assert outputs["phase_screen_logit"].grad[1] > 0


def test_all_masked_supervision_is_differentiable_zero():
    outputs = {"phase_screen_logit": torch.randn(2, requires_grad=True), "anomaly_logit": torch.randn(2, requires_grad=True), "family_logits": torch.randn(2, 5, requires_grad=True)}
    targets = {
        "phase": torch.full((2,), float("nan")), "phase_mask": torch.zeros(2, dtype=torch.bool),
        "anomaly": torch.full((2,), float("nan")), "anomaly_mask": torch.zeros(2, dtype=torch.bool),
        "family": torch.full((2, 5), float("nan")), "family_mask": torch.zeros(2, 5, dtype=torch.bool),
    }
    loss = screen_loss(outputs, targets)["loss"]
    assert loss.item() == 0
    loss.backward()
    for logits in outputs.values():
        assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_full_loss_backpropagates_to_node_edge_and_global_inputs():
    model = WLSScreenGNN(hidden_dim=32, layers=3, dropout=0)
    batch = collate_graphs([make_graph(5), make_graph(8, seed=11)])
    for key in ("x", "edge_attr", "u"):
        batch[key].requires_grad_()
    labels = {"phase": torch.tensor([1., 0.]), "anomaly": torch.tensor([1., 1.]), "family": torch.tensor([[1., 1., 0., 0., 0.], [0., 0., 0., 1., 0.]])}
    loss = screen_loss(model(batch), labels)["loss"]
    loss.backward()
    for key in ("x", "edge_attr", "u"):
        assert torch.isfinite(batch[key].grad).all()
        assert batch[key].grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_unavailable_and_nonfinite_graphs_do_not_become_negative_screens():
    graph = make_graph()
    graph["screen_status"] = "wls_failure"
    with pytest.raises(ValueError, match="unavailable"):
        collate_graphs([graph])
    graph = make_graph()
    graph["x"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        collate_graphs([graph])
