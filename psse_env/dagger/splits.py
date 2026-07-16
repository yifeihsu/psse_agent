from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping


def grouped_scenario_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Keep all branches of a root scenario in one deterministic split."""
    if train_fraction < 0 or validation_fraction < 0 or train_fraction + validation_fraction > 1:
        raise ValueError("invalid split fractions")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root_id = str(row.get("root_scenario_id", row.get("scenario_id")))
        groups[root_id].append(dict(row))
    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for root_id in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{root_id}".encode("utf-8")).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(2**64)
        split = "train" if fraction < train_fraction else (
            "validation" if fraction < train_fraction + validation_fraction else "test"
        )
        result[split].extend(groups[root_id])
    return result
