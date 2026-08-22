"""The preregistered Round-1 three-source training-view allocation.

Shares are derivable but not authoritative: rounding a share at ingestion time
is how two stages disagree about the same view.  The exact row counts are frozen
here, the shares are recorded for provenance only, and every consumer binds this
policy's digest.

Derivation of the frozen counts, at a 1,880-row view:

    probe        = round(1880 * 0.020)  =   38
    natural      = 1880 - 38            = 1842
    natural D1   = round(1842 * 0.285)  =  525
    D0           = 1842 - 525           = 1317

The probe bucket splits evenly between the two incidence-dependent strata: 19
rows each, being all 12 distinct roots retained once plus 7 deterministic
duplicate placements.  That keeps 10 rows of spare capacity against the 48-row
ceiling of the current 24-row probe corpus, so the allocation does not require
generating more probe roots.

A 3% probe share is infeasible with that corpus (56 requested against 48
available).  2% was chosen as the largest feasible share that leaves capacity
margin while modestly upweighting rare recovery supervision.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ROUND1_VIEW_POLICY_CONTRACT = "dagger1_round1_three_source_view_v1"
ROUND1_NATURAL_ONLY_VIEW_POLICY_CONTRACT = (
    "dagger1_round1_natural_d1_only_derived_view_v1"
)

ROUND1_THREE_SOURCE_VIEW_POLICY: dict[str, Any] = {
    "contract": ROUND1_VIEW_POLICY_CONTRACT,
    "schema_version": 1,
    "total_rows": 1880,
    "allocation": {
        "d0_bc0_rows": 1317,
        "natural_d1_rows": 525,
        "observable_recovery_probe_rows": 38,
    },
    # Recorded for provenance and auditability.  The counts above are what
    # ingestion enforces; these are never re-multiplied to derive them.
    "shares_for_provenance_only": {
        "probe_share": 0.020,
        "natural_share": 0.980,
        "natural_d1_share_of_natural_rows": 0.285,
    },
    "probe_bucket": {
        "post_failure_no_candidate": 19,
        "unsupported_correction_recovery": 19,
        "distinct_roots_retained_per_stratum": 12,
        "duplicate_placements_per_stratum": 7,
    },
    # Caps apply to the union of natural D1 and probe rows, not per source: at
    # least one probe root coincides with a natural support root, so two
    # per-source samplers could each honour the cap and still breach it after
    # the union.
    "global_caps": {
        "max_duplicate_count": 2,
        "max_rows_per_root": 8,
        "applies_across_sources": True,
    },
    "incidence_dependent_recovery_strata": [
        "post_failure_no_candidate",
        "unsupported_correction_recovery",
    ],
    "probe_floor_distinct_roots": 10,
    "combined_floor_distinct_roots": 10,
}


def round1_view_policy_digest(policy: dict[str, Any] | None = None) -> str:
    """Stable digest bound by collection, probe generation, and ingestion."""

    payload = ROUND1_THREE_SOURCE_VIEW_POLICY if policy is None else policy
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# This ablation is deliberately specified as a projection of the already
# placed three-source view.  It is not a second replay-buffer selection policy:
# the D0 and natural-D1 rows (including their multiplicity and order) must be
# identical to those in the canonical 1,880-row view.
ROUND1_NATURAL_ONLY_VIEW_POLICY: dict[str, Any] = {
    "contract": ROUND1_NATURAL_ONLY_VIEW_POLICY_CONTRACT,
    "schema_version": 1,
    "view_id": "natural-only",
    "parent_view_contract": ROUND1_VIEW_POLICY_CONTRACT,
    "parent_view_policy_digest": round1_view_policy_digest(),
    "derivation": "ordered_filter_of_canonical_full_placement_v1",
    "retained_replay_sources": ["d0_bc0", "natural_dagger1"],
    "excluded_replay_sources": ["observable_recovery_probe"],
    "total_rows": 1842,
    "allocation": {
        "d0_bc0_rows": 1317,
        "natural_d1_rows": 525,
        "observable_recovery_probe_rows": 0,
    },
    "parent_allocation": dict(ROUND1_THREE_SOURCE_VIEW_POLICY["allocation"]),
    "require_identical_parent_row_objects": True,
    "require_identical_parent_order": True,
    "permit_reselection": False,
}


def round1_natural_only_view_policy_digest(
    policy: dict[str, Any] | None = None,
) -> str:
    """Stable digest for the preregistered probe-ablation projection."""

    payload = ROUND1_NATURAL_ONLY_VIEW_POLICY if policy is None else policy
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_round1_natural_only_view_policy(
    policy: dict[str, Any],
    *,
    parent_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the ablation to be the exact probe-free parent projection."""

    if policy.get("contract") != ROUND1_NATURAL_ONLY_VIEW_POLICY_CONTRACT:
        raise ValueError("Round-1 natural-only view policy contract mismatch")
    if policy.get("schema_version") != 1:
        raise ValueError("Round-1 natural-only view policy schema mismatch")
    if policy.get("view_id") != "natural-only":
        raise ValueError("Round-1 natural-only view policy has the wrong view_id")
    parent_view = (
        ROUND1_THREE_SOURCE_VIEW_POLICY
        if parent_policy is None
        else parent_policy
    )
    validate_round1_view_policy(parent_view)
    if policy.get("parent_view_contract") != parent_view.get("contract"):
        raise ValueError("Round-1 natural-only view has the wrong parent contract")
    if policy.get("parent_view_policy_digest") != round1_view_policy_digest(
        parent_view
    ):
        raise ValueError("Round-1 natural-only view has the wrong parent digest")
    if policy.get("derivation") != "ordered_filter_of_canonical_full_placement_v1":
        raise ValueError("Round-1 natural-only derivation is not approved")
    if policy.get("retained_replay_sources") != [
        "d0_bc0",
        "natural_dagger1",
    ]:
        raise ValueError("Round-1 natural-only retained sources are not exact")
    if policy.get("excluded_replay_sources") != [
        "observable_recovery_probe"
    ]:
        raise ValueError("Round-1 natural-only excluded source is not exact")
    if policy.get("permit_reselection") is not False:
        raise ValueError("Round-1 natural-only policy must forbid reselection")
    if policy.get("require_identical_parent_row_objects") is not True:
        raise ValueError("Round-1 natural-only policy must preserve row objects")
    if policy.get("require_identical_parent_order") is not True:
        raise ValueError("Round-1 natural-only policy must preserve row order")

    parent = policy.get("parent_allocation")
    allocation = policy.get("allocation")
    if parent != parent_view["allocation"]:
        raise ValueError("Round-1 natural-only parent allocation is not frozen")
    expected = {
        "d0_bc0_rows": int(parent["d0_bc0_rows"]),
        "natural_d1_rows": int(parent["natural_d1_rows"]),
        "observable_recovery_probe_rows": 0,
    }
    if allocation != expected:
        raise ValueError("Round-1 natural-only allocation is not the exact projection")
    total = sum(expected.values())
    if policy.get("total_rows") != total:
        raise ValueError("Round-1 natural-only total does not match its allocation")
    return {
        "contract": ROUND1_NATURAL_ONLY_VIEW_POLICY_CONTRACT,
        "digest": round1_natural_only_view_policy_digest(policy),
        "parent_policy_digest": round1_view_policy_digest(parent_view),
        "total_rows": total,
        "allocation": expected,
        "passed": True,
    }


def validate_round1_view_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Check a policy is internally consistent before anything binds it."""

    allocation = policy.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("Round-1 view policy is missing its allocation")
    total = int(policy.get("total_rows") or 0)
    d0 = int(allocation.get("d0_bc0_rows") or 0)
    d1 = int(allocation.get("natural_d1_rows") or 0)
    probe = int(allocation.get("observable_recovery_probe_rows") or 0)
    if min(total, d0, d1, probe) <= 0:
        raise ValueError("Round-1 view allocation counts must all be positive")
    if d0 + d1 + probe != total:
        raise ValueError(
            f"Round-1 view allocation {d0}+{d1}+{probe} does not sum to {total}"
        )
    bucket = policy.get("probe_bucket") or {}
    strata = policy.get("incidence_dependent_recovery_strata") or []
    bucket_total = sum(int(bucket.get(name) or 0) for name in strata)
    if bucket_total != probe:
        raise ValueError(
            f"probe bucket {bucket_total} does not match the probe allocation {probe}"
        )
    retained = int(bucket.get("distinct_roots_retained_per_stratum") or 0)
    duplicates = int(bucket.get("duplicate_placements_per_stratum") or 0)
    for name in strata:
        if int(bucket.get(name) or 0) != retained + duplicates:
            raise ValueError(
                f"probe bucket for {name} does not equal retained + duplicates"
            )
    if retained < int(policy.get("probe_floor_distinct_roots") or 0):
        raise ValueError(
            "probe bucket retains fewer distinct roots than the probe floor"
        )
    return {
        "contract": ROUND1_VIEW_POLICY_CONTRACT,
        "digest": round1_view_policy_digest(policy),
        "total_rows": total,
        "allocation": dict(allocation),
        "passed": True,
    }
