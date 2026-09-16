"""Torch-only, variable-size WLS screening network.

The model consumes observable graph features, never simulator labels or asset IDs.
Its independent logits are screening scores; they are not calibrated probabilities
or authorization to change the configured electrical model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .feature_schema import FAMILY_NAMES


def _mean_max(values: Tensor, groups: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Pool without padding; a group with no members receives two zero vectors."""
    total = values.new_zeros((count, values.shape[-1]))
    total.index_add_(0, groups, values)
    sizes = torch.bincount(groups, minlength=count).to(values.dtype).unsqueeze(1)
    mean = total / sizes.clamp_min(1)
    maximum = values.new_full((count, values.shape[-1]), -torch.inf)
    maximum.scatter_reduce_(
        0, groups.unsqueeze(1).expand_as(values), values, reduce="amax", include_self=True
    )
    maximum = torch.where(sizes > 0, maximum, torch.zeros_like(maximum))
    return mean, maximum


def _validate_batch(batch: Mapping[str, Any]) -> None:
    """Reject malformed mappings, especially accidental cross-graph edges."""
    required = ("x", "edge_index", "edge_attr", "u", "node_batch", "edge_pair", "branch_batch")
    if any(key not in batch or not isinstance(batch[key], Tensor) for key in required):
        raise ValueError(f"Graph batch requires tensor fields {required}")
    x, edges, attrs, u = (batch[key] for key in required[:4])
    nodes, pairs, branches = (batch[key] for key in required[4:])
    if x.ndim != 2 or x.shape[1] != 27 or x.shape[0] == 0:
        raise ValueError("x must have shape [N,27] with at least one node")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if attrs.shape != (edges.shape[1], 40):
        raise ValueError("edge_attr must have shape [E,40]")
    if u.ndim != 2 or u.shape[1] != 4 or u.shape[0] == 0:
        raise ValueError("u must have shape [B,4] with at least one graph")
    if nodes.shape != (x.shape[0],) or pairs.shape != (edges.shape[1],) or branches.ndim != 1:
        raise ValueError("node_batch, edge_pair and branch_batch have inconsistent shapes")
    if any(batch[key].device != x.device for key in required):
        raise ValueError("All graph batch tensors must be on the same device")
    if any(tensor.dtype != torch.long for tensor in (edges, nodes, pairs, branches)):
        raise ValueError("Graph indices must be int64 tensors")
    if any(not tensor.is_floating_point() or not torch.isfinite(tensor).all() for tensor in (x, attrs, u)):
        raise ValueError("Graph features must be finite floating-point tensors")
    if attrs.dtype != x.dtype or u.dtype != x.dtype:
        raise ValueError("All graph features must have the same dtype")
    graph_count = u.shape[0]
    if (nodes < 0).any() or (nodes >= graph_count).any():
        raise ValueError("node_batch contains an invalid graph ID")
    if (torch.bincount(nodes, minlength=graph_count) == 0).any():
        raise ValueError("Every graph must contain at least one node")
    if (branches < 0).any() or (branches >= graph_count).any():
        raise ValueError("branch_batch contains an invalid graph ID")
    if (edges < 0).any() or (edges >= x.shape[0]).any():
        raise ValueError("edge_index contains an invalid node ID")
    if (pairs < 0).any() or (pairs >= branches.shape[0]).any():
        raise ValueError("edge_pair contains an invalid physical branch ID")
    if (torch.bincount(pairs, minlength=branches.shape[0]) != 2).any():
        raise ValueError("Each physical branch must have exactly two directed edges")
    if not torch.equal(nodes[edges[0]], nodes[edges[1]]):
        raise ValueError("Cross-graph message passing is forbidden")
    if not torch.equal(nodes[edges[0]], branches[pairs]):
        raise ValueError("Physical branch mapping crosses graph boundaries")
    if pairs.numel():
        pair_edges = torch.argsort(pairs).reshape(-1, 2)
        if not torch.equal(edges[:, pair_edges[:, 0]], edges[:, pair_edges[:, 1]].flip(0)):
            raise ValueError("Each physical branch must pair opposite edge directions")


def collate_graphs(
    graphs: Sequence[Mapping[str, Any]], *, device: torch.device | str | None = None
) -> dict[str, Any]:
    """Batch numpy/tensor graphs while explicitly offsetting nodes and branches.

    ``edge_pair`` contains contiguous local physical-branch IDs, each occurring
    exactly twice. Parallel branches keep distinct IDs. Metadata is returned as a
    list and is never converted to a numerical model input. Training labels are
    intentionally supplied separately from this observable-input batch.
    """
    if not graphs:
        raise ValueError("Cannot collate an empty graph sequence")
    target_device = torch.device("cpu" if device is None else device)
    fields: dict[str, list[Tensor]] = {
        key: [] for key in ("x", "edge_index", "edge_attr", "u", "node_batch", "edge_pair", "branch_batch")
    }
    metadata: list[Any] = []
    node_offset = branch_offset = 0
    for graph_id, graph in enumerate(graphs):
        if graph.get("screen_status", "valid") != "valid":
            raise ValueError("An unavailable screen must not enter the GNN as a negative example")
        x = torch.as_tensor(graph["x"], dtype=torch.float32, device=target_device)
        edges = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=target_device)
        attrs = torch.as_tensor(graph["edge_attr"], dtype=torch.float32, device=target_device)
        u = torch.as_tensor(graph["u"], dtype=torch.float32, device=target_device)
        pairs = torch.as_tensor(graph["edge_pair"], dtype=torch.long, device=target_device)
        if u.shape not in ((4,), (1, 4)):
            raise ValueError("Each input graph must have u with shape [4] or [1,4]")
        branch_count = int(pairs.max()) + 1 if pairs.numel() else 0
        local = {
            "x": x, "edge_index": edges, "edge_attr": attrs, "u": u.reshape(1, 4),
            "node_batch": torch.zeros(x.shape[0], dtype=torch.long, device=target_device),
            "edge_pair": pairs,
            "branch_batch": torch.zeros(branch_count, dtype=torch.long, device=target_device),
        }
        _validate_batch(local)
        fields["x"].append(x)
        fields["edge_index"].append(edges + node_offset)
        fields["edge_attr"].append(attrs)
        fields["u"].append(u.reshape(1, 4))
        fields["node_batch"].append(local["node_batch"] + graph_id)
        fields["edge_pair"].append(pairs + branch_offset)
        fields["branch_batch"].append(local["branch_batch"] + graph_id)
        metadata.append(graph.get("metadata", {}))
        node_offset += x.shape[0]
        branch_offset += branch_count
    result = {
        key: torch.cat(parts, dim=1 if key == "edge_index" else 0)
        for key, parts in fields.items()
    }
    result["metadata"] = metadata
    return result


def _mlp(input_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )


class _MessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_update = _mlp(3 * hidden_dim, hidden_dim, dropout)
        self.message = _mlp(3 * hidden_dim, hidden_dim, dropout)
        self.node_update = _mlp(3 * hidden_dim, hidden_dim, dropout)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, e: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        e = self.edge_norm(e + self.edge_update(torch.cat((h[src], h[dst], e), dim=-1)))
        messages = self.message(torch.cat((h[src], h[dst], e), dim=-1))
        mean, maximum = _mean_max(messages, dst, h.shape[0])
        h = self.node_norm(h + self.node_update(torch.cat((h, mean, maximum), dim=-1)))
        return h, e


class WLSScreenGNN(nn.Module):
    """Independent phase, represented-anomaly and five-family screening heads.

    Defaults implement the guide's 27/40/4 inputs, three width-128 blocks and
    516 -> 128 -> 64 graph decoder. ``hidden_dim`` and ``layers`` are exposed for
    small research controls; input feature semantics and the family set are fixed.
    """

    def __init__(
        self, *, hidden_dim: int = 128, layers: int = 3, dropout: float = 0.1,
        node_dim: int = 27, directed_edge_dim: int = 40, global_dim: int = 4,
    ) -> None:
        super().__init__()
        if (node_dim, directed_edge_dim, global_dim) != (27, 40, 4):
            raise ValueError("WLSScreenGNN v1 requires the 27/40/4 observable feature schema")
        if hidden_dim < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim/layers must be positive and dropout must be in [0,1)")
        self.config = {
            "hidden_dim": hidden_dim, "layers": layers, "dropout": dropout,
            "node_dim": node_dim, "directed_edge_dim": directed_edge_dim, "global_dim": global_dim,
        }
        self.node_encoder = _mlp(node_dim, hidden_dim)
        self.edge_encoder = _mlp(directed_edge_dim, hidden_dim)
        self.blocks = nn.ModuleList(_MessagePassingBlock(hidden_dim, dropout) for _ in range(layers))
        self.decoder = nn.Sequential(
            nn.Linear(4 * hidden_dim + global_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64), nn.SiLU(),
        )
        self.phase_head = nn.Linear(64, 1)
        self.anomaly_head = nn.Linear(64, 1)
        self.family_head = nn.Linear(64, len(FAMILY_NAMES))

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        _validate_batch(batch)
        h = self.node_encoder(batch["x"])
        e = self.edge_encoder(batch["edge_attr"])
        for block in self.blocks:
            h, e = block(h, e, batch["edge_index"])
        graph_count = batch["u"].shape[0]
        # Every registered physical branch contributes once to the graph readout.
        branch_e, _ = _mean_max(e, batch["edge_pair"], batch["branch_batch"].shape[0])
        node_mean, node_max = _mean_max(h, batch["node_batch"], graph_count)
        branch_mean, branch_max = _mean_max(branch_e, batch["branch_batch"], graph_count)
        context = self.decoder(torch.cat((node_mean, node_max, branch_mean, branch_max, batch["u"]), dim=-1))
        return {
            "phase_screen_logit": self.phase_head(context).squeeze(-1),
            "anomaly_logit": self.anomaly_head(context).squeeze(-1),
            "family_logits": self.family_head(context),
        }
