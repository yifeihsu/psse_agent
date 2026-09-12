"""Shared interpretation of the optional normalized-residual alarm."""

from __future__ import annotations

from typing import Any, Mapping


def normalized_residual_alarm(evidence: Any) -> bool:
    """Read an explicit provider alarm without reinterpreting legacy cutoffs.

    Candidate verification carries the flag directly. Policy observations keep
    it in their successful, current-state WLS ledger so bounded history cannot
    erase the alarm. An absent flag preserves the global-only legacy contract.
    """

    getter = getattr(evidence, "get", None)
    if not callable(getter):
        return False
    if getter("normalized_residual_alarm") is True:
        return True
    contexts = getter("fresh_context_evidence")
    wls = contexts.get("wls") if isinstance(contexts, Mapping) else None
    if not isinstance(wls, Mapping) or wls.get("successful") is not True:
        return False
    active_id = getter("active_state_id")
    if active_id is not None and str(wls.get("state_id")) != str(active_id):
        return False
    return wls.get("normalized_residual_alarm") is True
