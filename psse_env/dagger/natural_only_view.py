"""Deterministic probe ablation derived from the canonical Round-1 placement.

The natural-D1-only study arm is not allowed to run a second sampler.  It is
the order-preserving projection of the already materialized 1,880-row
three-source view: every D0 and natural-D1 placement is retained byte-for-byte,
and every recovery-probe placement is excluded.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from psse_env.dagger.round1_view_policy import (
    ROUND1_NATURAL_ONLY_VIEW_POLICY,
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_natural_only_view_policy_digest,
    round1_view_policy_digest,
    validate_round1_natural_only_view_policy,
    validate_round1_view_policy,
)


NATURAL_ONLY_VIEW_BUILD_CONTRACT = (
    "dagger1_round1_natural_d1_only_ordered_projection_v1"
)


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _allocation(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("replay_source") or "") for row in rows)
    return {
        "d0_bc0_rows": counts.pop("d0_bc0", 0),
        "natural_d1_rows": counts.pop("natural_dagger1", 0),
        "observable_recovery_probe_rows": counts.pop(
            "observable_recovery_probe", 0
        ),
        **({"unrecognized_rows": sum(counts.values())} if counts else {}),
    }


def build_round1_natural_only_view(
    full_view_rows: Sequence[Mapping[str, Any]],
    *,
    full_policy: dict[str, Any] | None = None,
    natural_only_policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact D0+natural-D1 subsequence of the full placement.

    Counts are checked before filtering.  Thus an extra, missing, or relabelled
    probe/natural row cannot be hidden by producing a superficially plausible
    1,842-row output.
    """

    parent_policy = (
        ROUND1_THREE_SOURCE_VIEW_POLICY if full_policy is None else full_policy
    )
    ablation_policy = (
        ROUND1_NATURAL_ONLY_VIEW_POLICY
        if natural_only_policy is None
        else natural_only_policy
    )
    validate_round1_view_policy(parent_policy)
    validate_round1_natural_only_view_policy(
        ablation_policy,
        parent_policy=parent_policy,
    )
    if ablation_policy.get("parent_view_policy_digest") != (
        round1_view_policy_digest(parent_policy)
    ):
        raise ValueError(
            "Natural-only view policy is not bound to the supplied full policy"
        )

    full = []
    for index, row in enumerate(full_view_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Full Round-1 view row {index} is not an object")
        full.append(copy.deepcopy(dict(row)))

    expected_parent = dict(parent_policy["allocation"])
    observed_parent = _allocation(full)
    if observed_parent != expected_parent:
        raise ValueError(
            "Full Round-1 placement does not have the frozen parent allocation: "
            f"expected {expected_parent}, got {observed_parent}"
        )
    if len(full) != int(parent_policy["total_rows"]):
        raise ValueError(
            "Full Round-1 placement row count differs from the frozen parent"
        )

    retained_sources = set(ablation_policy["retained_replay_sources"])
    excluded_sources = set(ablation_policy["excluded_replay_sources"])
    retained_indices = [
        index
        for index, row in enumerate(full)
        if row.get("replay_source") in retained_sources
    ]
    excluded_indices = [
        index
        for index, row in enumerate(full)
        if row.get("replay_source") in excluded_sources
    ]
    if len(retained_indices) + len(excluded_indices) != len(full):
        raise ValueError("Full Round-1 placement contains an unapproved replay source")

    derived = [copy.deepcopy(full[index]) for index in retained_indices]
    expected_derived = dict(ablation_policy["allocation"])
    observed_derived = _allocation(derived)
    if observed_derived != expected_derived:
        raise ValueError(
            "Natural-only projection does not have the frozen allocation: "
            f"expected {expected_derived}, got {observed_derived}"
        )
    if len(derived) != int(ablation_policy["total_rows"]):
        raise ValueError("Natural-only projection row count is not frozen")

    # Reconstructing from parent indices is the decisive no-reselection check.
    if derived != [full[index] for index in retained_indices]:
        raise AssertionError("Natural-only projection changed a retained parent row")
    if any(
        row.get("replay_source") == "observable_recovery_probe"
        for row in derived
    ):
        raise AssertionError("Natural-only projection retained a probe placement")

    report = {
        "contract": NATURAL_ONLY_VIEW_BUILD_CONTRACT,
        "schema_version": 1,
        "view_id": "natural-only",
        "derivation": ablation_policy["derivation"],
        "full_view_policy_digest": round1_view_policy_digest(parent_policy),
        "natural_only_view_policy_digest": (
            round1_natural_only_view_policy_digest(ablation_policy)
        ),
        "parent_rows": len(full),
        "parent_allocation": observed_parent,
        "parent_content_sha256": _stable_json_sha256(full),
        "retained_parent_indices_sha256": _stable_json_sha256(retained_indices),
        "excluded_parent_indices_sha256": _stable_json_sha256(excluded_indices),
        "retained_rows": len(derived),
        "retained_allocation": observed_derived,
        "retained_content_sha256": _stable_json_sha256(derived),
        "excluded_probe_rows": len(excluded_indices),
        "identical_parent_row_objects": True,
        "identical_parent_order": True,
        "reselection_performed": False,
        "passed": True,
    }
    return derived, report


__all__ = [
    "NATURAL_ONLY_VIEW_BUILD_CONTRACT",
    "build_round1_natural_only_view",
]
