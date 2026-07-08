from __future__ import annotations

import re
from typing import Any

from IEEE_14_OpenDSS.constants import BRANCH_ORDER


ELIGIBLE_HIF_BRANCHES = [
    i for i, name in enumerate(BRANCH_ORDER) if str(name).lower().startswith("line.")
]

DSS_TO_BRANCH_ROW0 = {str(name).lower(): i for i, name in enumerate(BRANCH_ORDER)}


def normalize_dss_element(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("DSS element name must not be empty")
    return text if "." in text else f"Line.{text}"


def branch_row0_for_dss_element(name: str) -> int | None:
    return DSS_TO_BRANCH_ROW0.get(normalize_dss_element(name).lower())


def parse_branch_endpoints(dss_element: str) -> tuple[int | None, int | None]:
    element = normalize_dss_element(dss_element)
    _, raw = element.split(".", 1)
    match = re.search(r"(\d+)\D+(\d+)", raw)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def branch_info_from_dss_element(dss_element: str) -> dict[str, Any]:
    row0 = branch_row0_for_dss_element(dss_element)
    from_bus, to_bus = parse_branch_endpoints(dss_element)
    return {
        "branch_row0": row0,
        "line_index1": row0 + 1 if row0 is not None else None,
        "dss_element": normalize_dss_element(dss_element),
        "from_bus": from_bus,
        "to_bus": to_bus,
    }


def branch_info_for_row0(branch_row0: int) -> dict[str, Any]:
    idx = int(branch_row0)
    if idx < 0 or idx >= len(BRANCH_ORDER):
        raise IndexError(f"branch_row0={idx} outside BRANCH_ORDER")
    return branch_info_from_dss_element(BRANCH_ORDER[idx])
