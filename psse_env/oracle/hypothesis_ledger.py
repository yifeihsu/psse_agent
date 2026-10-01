"""Hypothesis ledger: the balanced screen's ranked hypotheses against what this state has tested.

Step 3 of the hypothesis-ranking plan (docs/hypothesis_ranking_plan_20260930.md).
The balanced screen that runs on every WLS alarm already compares the four
balanced causes, applies the winners in sequence and ranks the targets inside
each class; its compact report rides on the WLS ledger of the active state.
This module reads that report together with the state's recovery records
(verification-rejected candidates, executor failures, accepted corrections)
and turns them into one ranked list of hypotheses with a status each, plus
the budget the expert may spend on verified candidates.  Everything is read
from the policy observation, so a student can learn the same ordering.

The ledger changes the expert's ordering only:

* the family the screen accepted first (or, after it was tested, the next
  one) is investigated first; the static family priority and the dominance
  tags keep ordering the rest;
* inside a family the screen's top-ranked target is tried first (the screen's
  refit ranking puts the true branch first on 5 of the 8 misranked parameter
  roots where the multiplier ranking never does);
* a family with two verification-rejected candidates on this state, or a
  state with four, ranks last: its remaining supported targets are tried only
  after every other family's proposals, never dropped.  The budget orders;
  it does not hand off.  A production handoff label is valid only once every
  supported same-state correction was tested or is safety-blocked
  (TransactionalPSSEEnv.assert_training_decision_evidence), and the 2026-10-01
  cell's stage 0 failed on a root where a dropped family left two supported
  targets outstanding.

A failed execution is not a tested hypothesis and consumes no budget; a
rejected candidate removes only its target; an accepted correction advances
the state and the ledger starts again on the child's own screen.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    current_screen_report,
    safe_normalize_action,
)
from psse_env.oracle.expert_types import ExpertActionProposal, recovery_record_applies_to_state

SCREEN_CLASS_FAMILY = {"meter": "measurement", "parameter": "parameter", "topology": "topology", "hif": "hif"}
FAMILY_TOOLS = {
    "measurement": (GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS),
    "parameter": (GET_PARAMETER_CONTEXT, CORRECT_PARAMETERS),
    "topology": (GET_TOPOLOGY_CONTEXT, CORRECT_TOPOLOGY),
}
TOOL_FAMILY = {tool: family for family, tools in FAMILY_TOOLS.items() for tool in tools}
CORRECTION_FAMILY = {CORRECT_MEASUREMENTS: "measurement", CORRECT_PARAMETERS: "parameter", CORRECT_TOPOLOGY: "topology"}
#: Verified candidates a family may spend on one state, and a state in all.
DEFAULT_FAMILY_BUDGET = 2
DEFAULT_STATE_BUDGET = 4
#: Confidence added to the leading family's proposals and to the screen's top target.
FAMILY_BOOST = 0.12
TARGET_BOOST = 0.01
SECOND_TARGET_BOOST = 0.005
#: Confidence removed from the proposals of a family (or state) over budget: they rank last.
EXHAUSTED_FAMILY_PENALTY = 0.5


def _as_mapping(state: Any) -> Mapping[str, Any]:
    if isinstance(state, Mapping):
        nested = state.get("policy_observation")
        return nested if isinstance(nested, Mapping) else state
    as_dict = getattr(state, "as_dict", None)
    return as_dict() if callable(as_dict) else {}


def _target_of(class_name: str, item: Mapping[str, Any]) -> int | None:
    key = "channel_index0" if class_name == "meter" else "branch_row0"
    value = item.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def screen_hypotheses(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ranked hypotheses from the compact screen report.

    The accepted sequence comes first in its order (``meter>parameter`` means
    a meter then a branch), then an HIF suspicion, then the last compared
    round's other classes by their penalized score.  Each entry names the
    family, the screen's target for it and where it came from.
    """
    if not isinstance(report, Mapping) or report.get("status") != "valid":
        return []
    hypotheses: list[dict[str, Any]] = []
    for position, accepted in enumerate(report.get("accepted_hypotheses") or []):
        if not isinstance(accepted, Mapping):
            continue
        family = SCREEN_CLASS_FAMILY.get(str(accepted.get("class")))
        if family is None:
            continue
        hypotheses.append({
            "family": family, "source": "accepted", "round": position,
            "target": _target_of(str(accepted["class"]), accepted),
            "parameter": accepted.get("parameter"),
        })
    if report.get("suspected") and not any(item["family"] == "hif" for item in hypotheses):
        branch = report.get("branch_row0")
        hypotheses.append({"family": "hif", "source": "suspected", "round": None,
                           "target": int(branch) if isinstance(branch, int) else None, "parameter": None})
    rounds = [item for item in report.get("rounds") or [] if isinstance(item, Mapping) and item.get("scores")]
    last = rounds[-1] if rounds else {}
    scores = last.get("scores") if isinstance(last.get("scores"), Mapping) else {}
    best = last.get("best") if isinstance(last.get("best"), Mapping) else {}
    for class_name, score in sorted(scores.items(), key=lambda pair: -float(pair[1])):
        family = SCREEN_CLASS_FAMILY.get(str(class_name))
        if family is None or any(item["family"] == family for item in hypotheses):
            continue
        item = best.get(class_name) if isinstance(best.get(class_name), Mapping) else {}
        hypotheses.append({"family": family, "source": "last_round", "round": len(rounds) - 1,
                           "target": _target_of(str(class_name), item), "parameter": item.get("parameter"),
                           "score": float(score)})
    for rank, item in enumerate(hypotheses, start=1):
        item["rank"] = rank
    return hypotheses


def screen_targets(report: Mapping[str, Any], family: str) -> list[int]:
    """The screen's ranked targets for ``family``: accepted targets first, then the last round's ranked alternatives."""
    if not isinstance(report, Mapping) or report.get("status") != "valid":
        return []
    class_name = next((name for name, fam in SCREEN_CLASS_FAMILY.items() if fam == family), None)
    if class_name is None:
        return []
    targets: list[int] = []
    for accepted in report.get("accepted_hypotheses") or []:
        if isinstance(accepted, Mapping) and str(accepted.get("class")) == class_name:
            target = _target_of(class_name, accepted)
            if target is not None and target not in targets:
                targets.append(target)
    rounds = [item for item in report.get("rounds") or [] if isinstance(item, Mapping) and item.get("scores")]
    for round_item in reversed(rounds):
        ranked = (round_item.get("ranked") or {}).get(class_name) if isinstance(round_item.get("ranked"), Mapping) else None
        best = (round_item.get("best") or {}).get(class_name) if isinstance(round_item.get("best"), Mapping) else None
        candidates = list(ranked or [])
        if isinstance(best, Mapping):
            candidates.insert(0, best)
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            target = _target_of(class_name, candidate)
            if target is not None and target not in targets:
                targets.append(target)
        if targets:
            break
    return targets


def correction_target(action: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """(family, screen-comparable target) of a correction action; branches as 0-based rows, meters as channel indices."""
    normalized = safe_normalize_action(action)
    tool = normalized["tool"]
    arguments = normalized["arguments"]
    family = CORRECTION_FAMILY.get(tool)
    if family is None:
        return None, None
    if tool == CORRECT_MEASUREMENTS:
        group = arguments.get("suspect_group")
        if isinstance(group, (list, tuple)) and len(group) == 1 and isinstance(group[0], int) and not isinstance(group[0], bool):
            return family, int(group[0])
        return family, None
    if tool == CORRECT_PARAMETERS:
        for key in ("branch_row0",):
            value = arguments.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return family, int(value)
        for key in ("line_index", "line_index1"):
            value = arguments.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return family, int(value) - 1
        return family, None
    return family, None


def tested_hypotheses(policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """What this active state has tried per family: rejected targets, executor failures, accepted targets."""
    active_id = policy.get("active_state_id")
    tested: dict[str, dict[str, Any]] = {
        family: {"rejected": [], "failed": [], "accepted": []} for family in FAMILY_TOOLS
    }
    for record in policy.get("rejected_hypotheses") or []:
        if not isinstance(record, Mapping) or not recovery_record_applies_to_state(record, active_id):
            continue
        source = record.get("source_action") or {}
        family, target = correction_target(source) if isinstance(source, Mapping) else (None, None)
        if family is None:
            continue
        bucket = "failed" if record.get("rejection_kind") == "executor_failure" else "rejected"
        tested[family][bucket].append(target)
    for record in policy.get("accepted_corrections") or []:
        if not isinstance(record, Mapping):
            continue
        source = record.get("source_action") or record.get("action") or record
        family, target = correction_target(source) if isinstance(source, Mapping) else (None, None)
        if family is None:
            continue
        tested[family]["accepted"].append(target)
    return tested


def hypothesis_ledger(
    state: Any, *, family_budget: int = DEFAULT_FAMILY_BUDGET, state_budget: int = DEFAULT_STATE_BUDGET,
) -> dict[str, Any]:
    """The ranked hypotheses of the active state with their status and the remaining budget."""
    policy = _as_mapping(state)
    report = current_screen_report(policy, "hif")
    hypotheses = screen_hypotheses(report)
    tested = tested_hypotheses(policy)
    rejected_total = sum(len(item["rejected"]) for item in tested.values())
    exhausted = sorted(
        family for family, item in tested.items() if len(item["rejected"]) >= int(family_budget)
    )
    state_exhausted = rejected_total >= int(state_budget)
    for item in hypotheses:
        family = item["family"]
        record = tested.get(family) or {"rejected": [], "failed": [], "accepted": []}
        target = item.get("target")
        if family in exhausted or state_exhausted:
            status = "budget_exhausted"
        elif target is not None and target in record["accepted"]:
            status = "accepted"
        elif target is not None and target in record["rejected"]:
            status = "rejected"
        elif target is not None and target in record["failed"]:
            status = "execution_failed"
        else:
            status = "untested"
        item["status"] = status
        item["screen_targets"] = screen_targets(report, family) if family in FAMILY_TOOLS else []
    leading = next((item for item in hypotheses if item["family"] in FAMILY_TOOLS and item["status"] in {"untested", "execution_failed"}), None)
    return {
        "available": bool(hypotheses),
        "hypotheses": hypotheses,
        "leading_family": leading["family"] if leading else None,
        "tested": tested,
        "rejected_total": rejected_total,
        "family_budget": int(family_budget),
        "state_budget": int(state_budget),
        "exhausted_families": exhausted,
        "state_budget_exhausted": state_exhausted,
    }


def rerank_proposals(
    proposals: Sequence[ExpertActionProposal], state: Any, *,
    family_budget: int = DEFAULT_FAMILY_BUDGET, state_budget: int = DEFAULT_STATE_BUDGET,
) -> list[ExpertActionProposal]:
    """Reorder the family experts' proposals by the ledger and apply its budget.

    Proposals of tools outside the three balanced families pass through
    untouched.  Without a valid screen report on the active state the
    proposals are returned unchanged.
    """
    ledger = hypothesis_ledger(state, family_budget=family_budget, state_budget=state_budget)
    if not ledger["available"]:
        return list(proposals)
    policy = _as_mapping(state)
    report = current_screen_report(policy, "hif")
    leading = ledger["leading_family"]
    exhausted = set(ledger["exhausted_families"])
    state_exhausted = bool(ledger["state_budget_exhausted"])
    result: list[ExpertActionProposal] = []
    for proposal in proposals:
        normalized = safe_normalize_action(proposal.action)
        tool = normalized["tool"]
        family = TOOL_FAMILY.get(tool)
        if family is None:
            result.append(proposal)
            continue
        confidence = proposal.confidence
        evidence = list(proposal.evidence_codes)
        if family in exhausted or (state_exhausted and tool in CORRECTION_FAMILY):
            # Two rejected candidates of this family on this state (or four
            # in all): its remaining targets rank behind every other family's
            # proposals.  They stay available, because a handoff is valid only
            # once every supported same-state correction was tested.
            confidence -= EXHAUSTED_FAMILY_PENALTY
            evidence.append(f"ledger_budget_exhausted={family}" if family in exhausted else "ledger_state_budget_exhausted")
            result.append(replace(proposal, confidence=confidence, evidence_codes=evidence))
            continue
        if family == leading:
            confidence += FAMILY_BOOST
            evidence.append(f"ledger_leading_family={family}")
        if tool in CORRECTION_FAMILY:
            _, target = correction_target(normalized)
            ranked = [t for t in screen_targets(report, family) if t not in ledger["tested"][family]["rejected"]]
            if target is not None and ranked:
                if target == ranked[0]:
                    confidence += TARGET_BOOST
                    evidence.append("ledger_top_target")
                elif len(ranked) > 1 and target == ranked[1]:
                    confidence += SECOND_TARGET_BOOST
                    evidence.append("ledger_second_target")
        result.append(replace(proposal, confidence=confidence, evidence_codes=evidence))
    return result


__all__ = [
    "DEFAULT_FAMILY_BUDGET", "DEFAULT_STATE_BUDGET", "EXHAUSTED_FAMILY_PENALTY", "FAMILY_BOOST", "SCREEN_CLASS_FAMILY",
    "TARGET_BOOST",
    "correction_target", "hypothesis_ledger", "rerank_proposals", "screen_hypotheses", "screen_targets",
    "tested_hypotheses",
]
