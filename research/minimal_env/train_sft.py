#!/usr/bin/env python3
"""research-train on rows of the minimal interface.

    python research/minimal_env/train_sft.py [the options of python -m psse_env.sft research-train]

The stock trainer (``psse_env/sft/research_cli.py``) was written for the old interface:
it requires every row to declare ``metadata.protocol == "canonical"`` and replaces each
row's tool list with the old registry before its row gate.  A row of the minimal
interface carries its own six tools and its own protocol name, so this entry point calls
the same trainer with those two steps replaced:

* a row must declare ``metadata.protocol == "minimal_env_v1"``;
* a row keeps its own tool list, checked as the schema it is, in the form the run-time
  policy renders (``psse_env.minimal.observation.tools``).

Everything else (the tokenizer and mask gate, the optimizer, checkpoints, the reload
check) is research_cli's.  Nothing under ``psse_env/sft`` changes.
"""
from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.rows import MINIMAL_PROTOCOL  # noqa: E402
from psse_env.sft import research_cli  # noqa: E402
from psse_env.sft.gates import GateError, validate_tool_schemas  # noqa: E402
from psse_env.sft.gemma_text import sanitize_tool_schemas  # noqa: E402

NORMALIZATION_CONTRACT = "minimal_row_tools_kept_v1"


def _protocol(row: Mapping[str, Any]) -> Any:
    metadata = row.get("metadata")
    return metadata.get("protocol") if isinstance(metadata, Mapping) else None


def validate_minimal_splits(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The trainer's split check for minimal rows: both sides non-empty, the protocol named, no case on both sides."""
    if not train_rows or not validation_rows:
        raise GateError("training and validation splits must both be non-empty")
    roots: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for index, row in enumerate(rows):
            root = str(row.get("physical_root_fingerprint") or "").strip()
            if not root:
                raise GateError(f"{split}[{index}] lacks physical_root_fingerprint")
            roots[split].add(root)
            if _protocol(row) != MINIMAL_PROTOCOL:
                raise GateError(f"{split}[{index}] is not a row of {MINIMAL_PROTOCOL}")
    overlap = roots["train"] & roots["validation"]
    if overlap:
        raise GateError("train/validation physical roots overlap: " + ", ".join(sorted(overlap)[:8]))
    return {"train_rows": len(train_rows), "validation_rows": len(validation_rows), "train_roots": len(roots["train"]),
            "validation_roots": len(roots["validation"]), "overlap": []}


def keep_row_tools(rows: Sequence[Mapping[str, Any]], *, source_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each row with its own tool list, validated and in the rendered form; the old registry is not put in."""
    if not rows:
        raise GateError(f"{source_label}: row set is empty")
    kept: list[dict[str, Any]] = []
    counts: Counter = Counter()
    changed = 0
    for index, row in enumerate(rows):
        copied = copy.deepcopy(dict(row))
        tools = validate_tool_schemas(copied.get("tools"), row_label=f"{source_label}[{index}]")
        rendered = sanitize_tool_schemas(tools)
        changed += int(rendered != copied.get("tools"))
        copied["tools"] = rendered
        counts[len(rendered)] += 1
        kept.append(copied)
    return kept, {"contract": NORMALIZATION_CONTRACT, "source_label": source_label, "rows": len(kept),
                  "source_registry_replaced": False, "rows_changed": changed,
                  "rows_by_tool_count": {str(key): value for key, value in sorted(counts.items())}}


def main(argv: list[str] | None = None) -> int:
    research_cli.validate_research_splits = validate_minimal_splits
    research_cli.normalize_research_rows = keep_row_tools
    return research_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
