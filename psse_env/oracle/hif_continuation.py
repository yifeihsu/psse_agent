"""Observable state-bound HIF accounting, separate from physical fault removal.

An accepted HIF fit accounts for its own diagnostic signature. It never proves
that coexisting meter errors are absent. Only a subsequent provider check of
the current measurements can open the conditioned meter route or close it.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

from psse_env.actions import ANOMALY_FAMILY_MARKERS, unexplained_signatures, waveform_anomaly_signatures
from psse_env.oracle.expert_types import matching_evidence_codes, policy_state_view, state_value
from psse_env.evidence_profile import is_scada_only


HIF_CONDITIONING_METHOD = "paired_opendss_effect_compensation"


def accepted_hif_explanation(state: Any) -> bool:
    state = policy_state_view(state)
    if is_scada_only(state):
        return False
    return any(
        isinstance(record, Mapping) and (
            record.get("family") == "hif"
            or bool(matching_evidence_codes(record.get("explained_signatures") or [], *ANOMALY_FAMILY_MARKERS["hif"]))
        )
        for record in state_value(state, "explained_anomalies", []) or []
    )


def current_hif_conditioning(state: Any) -> Mapping[str, Any] | None:
    """Read controller-bound evidence; stale/malformed claims are unavailable."""
    state = policy_state_view(state)
    contexts = state_value(state, "fresh_context_evidence") or {}
    if not isinstance(contexts, Mapping):
        return None
    record = contexts.get("hif_conditioning")
    active_id = str(state_value(state, "active_state_id") or "")
    if (not isinstance(record, Mapping) or not active_id
        or str(record.get("state_id") or "") != active_id
        or not isinstance(record.get("state_hash"), str) or not record["state_hash"]
        or record.get("method") != HIF_CONDITIONING_METHOD
        or record.get("physical_fault_still_present") is not True
        or record.get("status") not in {"ready", "unavailable"}):
        return None
    candidates = record.get("remaining_meter_candidate_indices")
    reasons = record.get("failure_reasons")
    if (not isinstance(candidates, (list, tuple))
        or any(type(index) is not int or index < 0 for index in candidates)
        or len(set(candidates)) != len(candidates)
        or not isinstance(reasons, (list, tuple))
        or any(not isinstance(reason, str) for reason in reasons)):
        return None
    wls = contexts.get("wls") or {}
    if record["status"] == "ready" and (
        reasons or not isinstance(wls, Mapping) or wls.get("successful") is not True
        or str(wls.get("state_id") or "") != active_id
        or wls.get("state_hash") != record["state_hash"]
    ):
        return None
    return record


def hif_meter_route_ready(state: Any) -> bool:
    """Permit meter work only after HIF accounting, with no other waveform."""
    if not accepted_hif_explanation(state):
        return False
    record = current_hif_conditioning(state)
    if record is None or record["status"] != "ready":
        return False
    state = policy_state_view(state)
    signatures = waveform_anomaly_signatures(state_value(state, "unresolved_signatures", []) or [])
    hif_signatures = matching_evidence_codes(signatures, *ANOMALY_FAMILY_MARKERS["hif"])
    if set(signatures) - set(hif_signatures):
        return False
    return not unexplained_signatures(hif_signatures, state_value(state, "explained_anomalies", []) or [])


def hif_conditioned_closure_ready(state: Any) -> bool:
    """Quiet dual WLS and no conditional meter candidates; not HIF removal."""
    if not hif_meter_route_ready(state):
        return False
    record = current_hif_conditioning(state)
    contexts = state_value(policy_state_view(state), "fresh_context_evidence") or {}
    wls = contexts.get("wls") or {}
    threshold, maximum = wls.get("normalized_residual_threshold"), wls.get("max_normalized_residual")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold) or threshold <= 0
        or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
        or not math.isfinite(maximum) or maximum < 0 or maximum >= threshold):
        return False
    return bool(
        record is not None and not record["remaining_meter_candidate_indices"]
        and wls.get("chi_square_alarm") is False
        and wls.get("normalized_residual_alarm") is False
    )


def recovery_signatures(state: Any) -> list[str]:
    """Remove only HIF signatures whose current physical effect is accounted."""
    state = policy_state_view(state)
    signatures = [str(value) for value in state_value(state, "unresolved_signatures", []) or []]
    if not hif_meter_route_ready(state):
        return signatures
    hif = set(matching_evidence_codes(signatures, *ANOMALY_FAMILY_MARKERS["hif"]))
    return [signature for signature in signatures if signature not in hif]
