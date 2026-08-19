"""Deterministic construction of the Round-1 three-source training view.

Reporting a quota is not the same as enforcing one.  ``dagger1_probe_replay_quota``
says how many probe rows *may* be placed; this module actually places them, and
it is the only place where "a probe row can never displace a natural row" stops
being an intention and becomes a property of the emitted view.

Two rules need the union rather than per-source reasoning:

* the physical-root cap.  At least one probe root coincides with a natural
  support root, so two independent samplers could each honour an eight-row cap
  and still breach it after the union.
* the duplicate cap, for the same reason.

D0 occupies a disjoint root space and is capped within itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from psse_env.dagger.round1_view_policy import (
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_view_policy_digest,
    validate_round1_view_policy,
)

THREE_SOURCE_VIEW_CONTRACT = "dagger1_three_source_training_view_v1"


def _order_key(row: Mapping[str, Any]) -> str:
    """Stable, content-derived placement order.

    Deliberately not input order: a view must not change because an upstream
    file was written in a different sequence.
    """

    payload = {
        "example_id": str(row.get("example_id") or ""),
        "physical_root_fingerprint": str(
            row.get("physical_root_fingerprint") or ""
        ),
        "recovery_stratum": str(row.get("recovery_stratum") or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _root(row: Mapping[str, Any]) -> str:
    return str(row.get("physical_root_fingerprint") or "").strip()


def _example(row: Mapping[str, Any]) -> str:
    return str(row.get("example_id") or "").strip()


class _CapLedger:
    """Placement counters shared by every source that shares a root space."""

    def __init__(self, *, max_duplicate_count: int, max_rows_per_root: int) -> None:
        self.duplicate_cap = int(max_duplicate_count)
        self.root_cap = int(max_rows_per_root)
        self.by_example: Counter[str] = Counter()
        self.by_root: Counter[str] = Counter()

    def admits(self, row: Mapping[str, Any]) -> bool:
        return (
            self.by_example[_example(row)] < self.duplicate_cap
            and self.by_root[_root(row)] < self.root_cap
        )

    def place(self, row: Mapping[str, Any]) -> None:
        self.by_example[_example(row)] += 1
        self.by_root[_root(row)] += 1


def _place_probe_bucket(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    ledger: _CapLedger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain every distinct probe root once, then balance duplicates."""

    bucket = policy["probe_bucket"]
    strata = list(policy["incidence_dependent_recovery_strata"])
    placed: list[dict[str, Any]] = []
    report: dict[str, Any] = {}

    for stratum in strata:
        target = int(bucket.get(stratum) or 0)
        rows = sorted(
            (row for row in probe_rows if row.get("recovery_stratum") == stratum),
            key=_order_key,
        )
        # One copy of every distinct root first: unique roots are the scarce
        # resource, and a duplicate must never be placed ahead of one.
        seen_roots: set[str] = set()
        unique: list[Mapping[str, Any]] = []
        for row in rows:
            root = _root(row)
            if root and root not in seen_roots:
                seen_roots.add(root)
                unique.append(row)
        retained = 0
        for row in unique:
            if not ledger.admits(row):
                continue
            ledger.place(row)
            placed.append(copy.deepcopy(dict(row)))
            retained += 1
        # Only now may duplicates compete for the remaining bucket slots.
        duplicates = 0
        while retained + duplicates < target:
            progressed = False
            for row in unique:
                if retained + duplicates >= target:
                    break
                if not ledger.admits(row):
                    continue
                ledger.place(row)
                placed.append(copy.deepcopy(dict(row)))
                duplicates += 1
                progressed = True
            if not progressed:
                break
        report[stratum] = {
            "requested_rows": target,
            "distinct_roots_retained": retained,
            "duplicate_placements": duplicates,
            "placed_rows": retained + duplicates,
            "shortfall": max(target - (retained + duplicates), 0),
        }
    return placed, report


def _place_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: int,
    ledger: _CapLedger,
) -> tuple[list[dict[str, Any]], int]:
    """Fill a source to ``target``, one pass per duplicate round."""

    ordered = sorted(rows, key=_order_key)
    placed: list[dict[str, Any]] = []
    while len(placed) < target:
        progressed = False
        for row in ordered:
            if len(placed) >= target:
                break
            if not ledger.admits(row):
                continue
            ledger.place(row)
            placed.append(copy.deepcopy(dict(row)))
            progressed = True
        if not progressed:
            break
    return placed, max(target - len(placed), 0)


def build_dagger1_three_source_view(
    *,
    d0_rows: Iterable[Mapping[str, Any]],
    natural_d1_rows: Iterable[Mapping[str, Any]],
    probe_rows: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any] = ROUND1_THREE_SOURCE_VIEW_POLICY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Place the frozen allocation, enforcing caps across the shared root space."""

    validate_round1_view_policy(dict(policy))
    allocation = policy["allocation"]
    caps = policy["global_caps"]

    d0 = list(d0_rows)
    natural = list(natural_d1_rows)
    probes = list(probe_rows)
    for name, rows in (("D0", d0), ("natural D1", natural), ("probe", probes)):
        if not rows:
            raise ValueError(f"three-source view requires non-empty {name} rows")

    # Natural D1 and probes share a root space; D0 does not.
    shared = _CapLedger(
        max_duplicate_count=int(caps["max_duplicate_count"]),
        max_rows_per_root=int(caps["max_rows_per_root"]),
    )
    d0_ledger = _CapLedger(
        max_duplicate_count=int(caps["max_duplicate_count"]),
        max_rows_per_root=int(caps["max_rows_per_root"]),
    )

    # Probes are placed first so an auxiliary row can never be crowded out by
    # natural fill; the natural target is then honoured against what remains.
    probe_placed, probe_report = _place_probe_bucket(
        probes, policy=policy, ledger=shared
    )
    natural_placed, natural_short = _place_source(
        natural,
        target=int(allocation["natural_d1_rows"]),
        ledger=shared,
    )
    d0_placed, d0_short = _place_source(
        d0, target=int(allocation["d0_bc0_rows"]), ledger=d0_ledger
    )

    probe_short = sum(entry["shortfall"] for entry in probe_report.values())
    rows = d0_placed + natural_placed + probe_placed
    cap_violations = sorted(
        root
        for root, count in shared.by_root.items()
        if count > int(caps["max_rows_per_root"])
    )
    shared_roots = sorted(
        root
        for root in shared.by_root
        if any(_root(r) == root for r in probe_placed)
        and any(_root(r) == root for r in natural_placed)
    )
    report = {
        "contract": THREE_SOURCE_VIEW_CONTRACT,
        "policy_digest": round1_view_policy_digest(dict(policy)),
        "total_rows": len(rows),
        "requested_total_rows": int(policy["total_rows"]),
        "placed": {
            "d0_bc0_rows": len(d0_placed),
            "natural_d1_rows": len(natural_placed),
            "observable_recovery_probe_rows": len(probe_placed),
        },
        "shortfalls": {
            "d0_bc0_rows": d0_short,
            "natural_d1_rows": natural_short,
            "observable_recovery_probe_rows": probe_short,
        },
        "probe_bucket": probe_report,
        # Reported because it is the reason caps are enforced on the union.
        "roots_shared_between_probe_and_natural": shared_roots,
        "global_root_cap_violations": cap_violations,
        "max_rows_placed_on_one_shared_root": (
            max(shared.by_root.values()) if shared.by_root else 0
        ),
        "passed": bool(
            not cap_violations
            and not probe_short
            and not natural_short
            and not d0_short
            and len(rows) == int(policy["total_rows"])
        ),
    }
    return rows, report
