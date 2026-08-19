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
