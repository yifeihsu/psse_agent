#!/usr/bin/env python3
"""research-train, rendering every row with the tool list of its run-time prompt.

    train_runtime_tools.py [the options of python -m psse_env.sft research-train]

The trainer replaces each row's tools with the whole sanitized registry before its row gate
(psse_env/sft/research_cli.py prepare_research_examples, psse_env/sft/research_rows.py).  That made
training match inference when inference rendered the whole registry.  Inference now renders
tool_schemas_for_observation(registry, observation) (research_policy_factory.py), so under scada_only
with the lifecycle choice a training prompt would carry 20 tools and the plain rollback description,
and the run-time prompt 13 tools and the description that says a candidate may be declined.

This entry point calls the same trainer with one change: after the trainer's normalization, each row
gets the list the policy would render for that row's own observation.  Everything else (row gate,
optimizer, checkpoints, reload) is research_cli's.  cell.env CLONE_TOOLS=registry runs the stock
command instead.  Nothing under psse_env changes; a fix there would make this file redundant.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.dataset_builder import tool_schemas_for_observation  # noqa: E402
from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas  # noqa: E402
from psse_env.sft import research_cli  # noqa: E402
from psse_env.sft.gates import GateError  # noqa: E402

_STOCK_NORMALIZATION = research_cli.normalize_research_rows


def row_observation(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The observation a row's prompt shows: the state in its user turn (the policy's payload)."""
    for message in row.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            state = json.loads(str(message.get("content"))).get("state")
            if isinstance(state, Mapping):
                return state
    raise GateError(f"{row.get('example_id')}: no state in the row's user turn")


def normalize_with_runtime_tools(rows: Sequence[Mapping[str, Any]], *, source_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The trainer's normalization, then each row's run-time tool list."""
    normalized, report = _STOCK_NORMALIZATION(rows, source_label=source_label)
    registry = canonical_prompt_tool_schemas()
    counts: Counter = Counter()
    for row in normalized:
        row["tools"] = tool_schemas_for_observation(registry, row_observation(row))
        counts[len(row["tools"])] += 1
    report["rendered_tools"] = {"source": "tool_schemas_for_observation", "rows_by_tool_count": {str(k): v for k, v in sorted(counts.items())}}
    return normalized, report


def install() -> None:
    research_cli.normalize_research_rows = normalize_with_runtime_tools


if __name__ == "__main__":
    install()
    raise SystemExit(research_cli.main(sys.argv[1:]))
