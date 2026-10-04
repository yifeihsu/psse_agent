"""Size-agnostic triage classifier: message passing over buses and branches, pooled to graph-level heads.

The blocks are those of ``research.gnn_screen.model`` (edge and node updates
with mean and max aggregation, residual connections, LayerNorm), so nothing
in the network depends on the number of buses or branches: the same weights
run on any case.  The heads differ: one ``needs_aux`` logit (request
phase-resolved measurements), one logit per family, trained as independent
binary labels because mixed roots carry several families, and a three-way
``first`` head for the balanced family to investigate first.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from research.gnn_screen.model import _MessagePassingBlock, _mean_max, _mlp

FAMILY_HEADS = ("measurement", "parameter", "topology", "hif", "unbalance", "harmonic")
#: The balanced family to investigate first (a softmax; rows without a balanced error are masked).
FIRST_CLASSES = ("measurement", "parameter", "topology")


def collate(graphs: Sequence[Mapping[str, Any]], device: torch.device | str = "cpu") -> dict[str, Tensor]:
    """Batch variable-size graphs, offsetting node and branch indices."""
    if not graphs:
        raise ValueError("cannot collate an empty graph sequence")
    parts: dict[str, list[Tensor]] = {key: [] for key in ("x", "edge_index", "edge_attr", "u", "node_batch", "edge_pair", "branch_batch")}
    node_offset = branch_offset = 0
    for graph_id, graph in enumerate(graphs):
        x = torch.as_tensor(graph["x"], dtype=torch.float32)
        pairs = torch.as_tensor(graph["edge_pair"], dtype=torch.long)
        branches = int(pairs.max()) + 1 if pairs.numel() else 0
        parts["x"].append(x)
        parts["edge_index"].append(torch.as_tensor(graph["edge_index"], dtype=torch.long) + node_offset)
        parts["edge_attr"].append(torch.as_tensor(graph["edge_attr"], dtype=torch.float32))
        parts["u"].append(torch.as_tensor(graph["u"], dtype=torch.float32).reshape(1, -1))
        parts["node_batch"].append(torch.full((x.shape[0],), graph_id, dtype=torch.long))
        parts["edge_pair"].append(pairs + branch_offset)
        parts["branch_batch"].append(torch.full((branches,), graph_id, dtype=torch.long))
        node_offset += x.shape[0]
        branch_offset += branches
    return {key: torch.cat(value, dim=1 if key == "edge_index" else 0).to(device) for key, value in parts.items()}


class TriageGNN(nn.Module):
    """Graph-level ``needs_aux`` and family logits from WLS graph features of any network size."""

    def __init__(self, node_dim: int, edge_dim: int, global_dim: int, *, hidden_dim: int = 128, layers: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.config = {"node_dim": node_dim, "edge_dim": edge_dim, "global_dim": global_dim, "hidden_dim": hidden_dim,
                       "layers": layers, "dropout": dropout}
        self.node_encoder = _mlp(node_dim, hidden_dim)
        self.edge_encoder = _mlp(edge_dim, hidden_dim)
        self.blocks = nn.ModuleList(_MessagePassingBlock(hidden_dim, dropout) for _ in range(layers))
        self.decoder = nn.Sequential(
            nn.Linear(4 * hidden_dim + global_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64), nn.SiLU(),
        )
        self.aux_head = nn.Linear(64, 1)
        self.family_head = nn.Linear(64, len(FAMILY_HEADS))
        self.first_head = nn.Linear(64, len(FIRST_CLASSES))

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        h = self.node_encoder(batch["x"])
        e = self.edge_encoder(batch["edge_attr"])
        for block in self.blocks:
            h, e = block(h, e, batch["edge_index"])
        graphs = batch["u"].shape[0]
        branch_e, _ = _mean_max(e, batch["edge_pair"], batch["branch_batch"].shape[0])
        node_mean, node_max = _mean_max(h, batch["node_batch"], graphs)
        branch_mean, branch_max = _mean_max(branch_e, batch["branch_batch"], graphs)
        context = self.decoder(torch.cat((node_mean, node_max, branch_mean, branch_max, batch["u"]), dim=-1))
        return {"needs_aux": self.aux_head(context).squeeze(-1), "family": self.family_head(context),
                "first": self.first_head(context)}
