"""Research-only normalization from release rows to inference prompt rows.

Release datasets retain the tool registry they were rendered with.  The Gemma
research policies render the current sanitized prompt registry at inference
time, so this module replaces every row's registry with that one in a deep
copy and records the source registry digests as provenance.  A row rendered
under an older registry (for example the classical trace validation set,
which predates the measurement-request tools) is normalized rather than
rejected; the strict/release dataset path is unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas

from .gates import GateError, validate_tool_schemas


RESEARCH_ROW_NORMALIZATION_CONTRACT = "research_prompt_registry_normalization_v1"


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_research_rows(
    rows: Sequence[Mapping[str, Any]], *, source_label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy the inference registry into deep copies of the rows."""

    materialized = [copy.deepcopy(dict(row)) for row in rows]
    if not materialized:
        raise GateError(f"{source_label}: research row set is empty")
    protocol_failures = []
    for index, row in enumerate(materialized):
        metadata = row.get("metadata")
        protocol = metadata.get("protocol") if isinstance(metadata, Mapping) else None
        if protocol != "canonical":
            protocol_failures.append(f"row[{index}]={protocol!r}")
    if protocol_failures:
        raise GateError(
            f"{source_label}: research rows require metadata.protocol='canonical': "
            + ", ".join(protocol_failures[:8])
        )
    prompt_tools = canonical_prompt_tool_schemas()
    validate_tool_schemas(prompt_tools, row_label=f"{source_label}:prompt_registry")
    source_digests = Counter(_digest(row.get("tools")) for row in materialized)
    normalized: list[dict[str, Any]] = []
    changed = 0
    for row in materialized:
        if row.get("tools") != prompt_tools:
            changed += 1
        row["tools"] = copy.deepcopy(prompt_tools)
        normalized.append(row)

    report = {
        "contract": RESEARCH_ROW_NORMALIZATION_CONTRACT,
        "source_label": source_label,
        "rows": len(normalized),
        "source_registry_replaced": True,
        "source_registry_digests": dict(sorted(source_digests.items())),
        "prompt_registry_digest": _digest(prompt_tools),
        "prompt_registry_tool_count": len(prompt_tools),
        "rows_changed": changed,
        "strict_release_rows_mutated": False,
    }
    return normalized, report


__all__ = [
    "RESEARCH_ROW_NORMALIZATION_CONTRACT",
    "normalize_research_rows",
]
