from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

from psse_env.actions import safe_normalize_action


@dataclass(frozen=True)
class ExpertActionProposal:
    """A domain expert's evidence-bearing proposal for the next macro-action.

    ``action`` is deliberately the only field that is later exposed as a
    policy target.  Evidence and confidence are oracle-side ranking metadata;
    callers should use :meth:`as_action` when constructing DAgger labels.
    """

    action: dict[str, Any]
    source_expert: str
    confidence: float
    evidence_codes: list[str] = field(default_factory=list)
    admissible: bool = True
    estimated_immediate_risk: float = 0.0

    def __post_init__(self) -> None:
        normalized = safe_normalize_action(self.action)
        object.__setattr__(self, "action", copy.deepcopy(normalized))
        object.__setattr__(self, "source_expert", str(self.source_expert))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "evidence_codes", [str(code) for code in self.evidence_codes])
        object.__setattr__(self, "admissible", bool(self.admissible))
        object.__setattr__(self, "estimated_immediate_risk", float(self.estimated_immediate_risk))

    def as_action(self) -> dict[str, Any]:
        return copy.deepcopy(self.action)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))

    def with_admissibility(self, admissible: bool) -> "ExpertActionProposal":
        return replace(self, admissible=bool(admissible))


def state_value(state: Any, key: str, default: Any = None) -> Any:
    """Mapping/dataclass-compatible access used by independently testable experts."""
    getter = getattr(state, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(state, key, default)


def normalized_hint_actions(
    hints: Sequence[Mapping[str, Any]] | None,
    *,
    allowed_tools: set[str],
    active_state_id: str | None,
) -> list[dict[str, Any]]:
    """Normalize explicitly privileged oracle hints for one domain.

    Hints are consumed only inside the oracle.  This helper copies them and
    never writes them into the policy observation.
    """
    actions: list[dict[str, Any]] = []
    for hint in hints or ():
        action_like: Any = hint.get("action") if isinstance(hint, Mapping) and "action" in hint else hint
        normalized = safe_normalize_action(action_like)
        if normalized["tool"] not in allowed_tools:
            continue
        arguments = dict(normalized["arguments"])
        # Scenario-authored hints commonly use the legacy local id (``s0``).
        # Episode state ids are now namespaced, so all domain hints must bind
        # to the reached active state rather than retain a stale reference.
        if active_state_id is not None:
            arguments["state_id"] = active_state_id
        actions.append({"tool": normalized["tool"], "arguments": arguments})
    return actions


def history_action_tool(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    action = item.get("source_action") or item.get("action") or item.get("executed_action")
    if action is None and item.get("tool"):
        action = item
    if not isinstance(action, Mapping):
        return None
    return safe_normalize_action(action)["tool"]


def recovery_record_applies_to_state(item: Any, active_state_id: Any) -> bool:
    """Return whether state-bound recovery evidence belongs to the active state.

    Older fixture records may omit both bindings; those remain usable for
    backwards compatibility.  Once either the candidate parent or source
    action declares a state, however, a mismatch must not influence routing on
    a later committed state.
    """
    if not isinstance(item, Mapping):
        return False
    parent_id = item.get("candidate_parent_id")
    if parent_id is not None and (
        active_state_id is None or str(parent_id) != str(active_state_id)
    ):
        return False
    action = item.get("source_action") or item.get("action") or item.get(
        "executed_action"
    )
    if isinstance(action, Mapping):
        normalized = safe_normalize_action(action)
        requested = normalized["arguments"].get("state_id")
        if requested is None:
            requested = action.get("state_id")
        if requested is not None and (
            active_state_id is None or str(requested) != str(active_state_id)
        ):
            return False
    return True


def evidence_contains(signatures: Any, *needles: str) -> bool:
    return bool(matching_evidence_codes(signatures, *needles))


def matching_evidence_codes(signatures: Any, *needles: str) -> list[str]:
    matches: list[str] = []
    for value in signatures or []:
        text = str(value)
        lowered = text.lower()
        if any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(needle.lower())}(?![a-z0-9])",
                lowered,
            )
            is not None
            for needle in needles
        ):
            matches.append(text)
    return list(dict.fromkeys(matches))


def policy_state_view(state: Any) -> Any:
    nested = state_value(state, "policy_observation")
    return nested if nested is not None else state


def dominance_confidence(base: float, matched_codes: Sequence[str], boost: float = 0.05) -> float:
    """Raise a family route's confidence when its WLS evidence is dominant.

    The deployment WLS runner tags the signature family whose normalized
    evidence dominates the solve (largest residual vs largest branch
    multiplier) with a ``dominant`` token.  A family expert whose matched
    signatures carry that token outranks the tied baseline confidence of the
    other families; untagged signatures (pilot adapters, sensors) keep the
    base confidence so existing routes are unchanged.
    """
    if any(
        re.search(r"(?<![a-z0-9])dominant(?![a-z0-9])", str(code).lower())
        for code in matched_codes or []
    ):
        return float(base) + float(boost)
    return float(base)
