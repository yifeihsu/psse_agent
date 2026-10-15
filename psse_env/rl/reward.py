"""The graded episode reward (reward contract of docs/rl_hard_cases_plan_20261013.md, section 6, step 2).

The audit decides success and its basis; the evaluator counts what the
episode cost and what it broke.  This module turns one evaluated episode
into a scalar with a breakdown, under predeclared weights:

* basis credit, ranked: a decisive diagnosis above a bounded handoff that
  names the truth, above an unexplained-discrepancy handoff, above a budget
  exhaustion, with every failure at zero;
* costs: a per-call cost, a cost per acquisition the truth did not need, and
  tool regret against the expert's call count on the same root when a
  reference is given;
* penalties: false commits, false finalizations, a healthy component edited,
  a loop, invalid actions.

Nothing here reads the policy's view; the inputs are the evaluator's episode
record (``EpisodeEvaluation.as_dict()``), which the audit built from the
offline truth after the episode ended.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from psse_env.actions import ASK_FOR_MORE_EVIDENCE, GET_HARMONIC_CONTEXT, GET_THREE_PHASE_CONTEXT

REWARD_CONTRACT = "rl_graded_episode_reward_v1"

#: Audit bases, as release_audit reports them, with their credit relative to a
#: decisive diagnosis.  A handoff that names the truth is worth less than the
#: diagnosis it stops short of; one that only says "unexplained" less again.
BASIS_CREDIT: dict[str, float] = {
    "counterfactual_resolution": 1.0,
    "bounded_localization_handoff": 0.6,
    "unexplained_discrepancy_handoff": 0.5,
}
#: A budget-exhaustion handoff that the audit still credits (rare: a quiet
#: corrected state) is scored like a bounded handoff.
BUDGET_EXHAUSTED_REQUEST = "operator_escalation:recovery_budget_exhausted"
BUDGET_EXHAUSTED_CREDIT = 0.4

#: Which auxiliary stream a family's truth needs.  Phasors on a harmonic root
#: are not needed since the spectra-first route of 2026-10-13; spectra on an
#: unbalance root are the routing's misroute and count as unneeded.
NEEDED_STREAMS: dict[str, frozenset[str]] = {
    "hif": frozenset({GET_THREE_PHASE_CONTEXT}),
    "measurement+hif": frozenset({GET_THREE_PHASE_CONTEXT}),
    "three_phase_unbalance": frozenset({GET_THREE_PHASE_CONTEXT}),
    "harmonic": frozenset({GET_HARMONIC_CONTEXT}),
}
ACQUISITION_TOOLS = frozenset({GET_THREE_PHASE_CONTEXT, GET_HARMONIC_CONTEXT})


@dataclass(frozen=True)
class RewardWeights:
    """Predeclared weights.  Change them in a config, never after seeing the draw."""

    success: float = 1.0
    call_cost: float = 0.02
    unneeded_acquisition_cost: float = 0.1
    tool_regret_cost: float = 0.02
    false_commit: float = 0.5
    false_finalization: float = 0.5
    healthy_component_corruption: float = 0.5
    loop: float = 0.25
    invalid_action: float = 0.05
    #: Calls beyond this many are not charged, so a long episode cannot be
    #: pushed below the penalty floor by length alone.
    max_charged_calls: int = 40
    basis_credit: dict[str, float] = field(default_factory=lambda: dict(BASIS_CREDIT))

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "basis_credit":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"reward weight {name} must be a nonnegative number, got {value!r}")
        if self.max_charged_calls <= 0:
            raise ValueError("max_charged_calls must be positive")
        for basis, credit in self.basis_credit.items():
            if not 0.0 <= float(credit) <= 1.0:
                raise ValueError(f"basis credit for {basis} must lie in [0, 1], got {credit!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RewardWeights":
        if not value:
            weights = cls()
        else:
            payload = dict(value)
            basis = payload.pop("basis_credit", None)
            weights = cls(**payload, basis_credit=dict(basis) if basis else dict(BASIS_CREDIT))
        weights.validate()
        return weights


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    credit: float
    basis: str
    escalation_request: str | None
    calls: int
    charged_calls: int
    acquisitions: dict[str, int]
    unneeded_acquisitions: int
    tool_regret: int | None
    false_commits: int
    false_finalizations: int
    healthy_component_corrupted: bool
    loop_detected: bool
    invalid_actions: int
    penalties: float
    costs: float
    contract: str = REWARD_CONTRACT

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_basis(episode: Mapping[str, Any]) -> str:
    """The audit basis of an evaluated episode, ``none`` when it did not succeed."""
    audit = episode.get("audit") if isinstance(episode.get("audit"), Mapping) else {}
    assessment = audit.get("truth_audited_task_assessment") if isinstance(audit, Mapping) else None
    if isinstance(assessment, Mapping) and assessment.get("basis"):
        return str(assessment["basis"])
    return "none"


def escalation_request(episode: Mapping[str, Any]) -> str | None:
    """The request of the episode's last successful ``ask_for_more_evidence``."""
    for step in reversed(list(episode.get("trace") or [])):
        action = step.get("action") if isinstance(step, Mapping) else None
        if not isinstance(action, Mapping) or action.get("tool") != ASK_FOR_MORE_EVIDENCE:
            continue
        output = step.get("policy_tool_output") or step.get("tool_output") or {}
        status = step.get("execution_status")
        if status is None and isinstance(output, Mapping):
            status = output.get("execution_status")
        if status == "success":
            arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
            request = arguments.get("request")
            return str(request) if request is not None else None
    return None


def acquisition_counts(episode: Mapping[str, Any]) -> dict[str, int]:
    """Successful phasor and spectra acquisitions in the episode's trace."""
    counts = {tool: 0 for tool in sorted(ACQUISITION_TOOLS)}
    for step in episode.get("trace") or []:
        if not isinstance(step, Mapping):
            continue
        action = step.get("action") if isinstance(step.get("action"), Mapping) else {}
        tool = str(action.get("tool"))
        if tool in counts and step.get("execution_status") == "success":
            counts[tool] += 1
    return counts


def unneeded_acquisitions(counts: Mapping[str, int], family: str) -> int:
    needed = NEEDED_STREAMS.get(str(family), frozenset())
    return sum(int(count) for tool, count in counts.items() if tool not in needed)


def episode_reward(
    episode: Mapping[str, Any],
    *,
    family: str | None = None,
    expert_reference_calls: int | None = None,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Score one evaluated episode (``EpisodeEvaluation.as_dict()``)."""

    weights = weights or RewardWeights()
    weights.validate()
    family = str(family or episode.get("family") or "unknown")
    basis = episode_basis(episode)
    request = escalation_request(episode)
    success = bool(episode.get("truth_audited_task_success"))
    credit = 0.0
    if success:
        if basis in weights.basis_credit:
            credit = float(weights.basis_credit[basis])
        elif request == BUDGET_EXHAUSTED_REQUEST:
            credit = BUDGET_EXHAUSTED_CREDIT
        else:
            credit = float(weights.basis_credit.get("counterfactual_resolution", 1.0))
        if request == BUDGET_EXHAUSTED_REQUEST and basis == "counterfactual_resolution":
            credit = min(credit, BUDGET_EXHAUSTED_CREDIT)
    calls = int(episode.get("policy_steps") or episode.get("steps") or 0)
    charged = min(calls, int(weights.max_charged_calls))
    counts = acquisition_counts(episode)
    unneeded = unneeded_acquisitions(counts, family)
    regret: int | None = None
    if expert_reference_calls is not None:
        regret = max(0, calls - int(expert_reference_calls))
    false_commits = int(episode.get("false_commit_count") or 0)
    false_finalizations = int(episode.get("false_finalization_count") or 0)
    healthy_known = episode.get("healthy_preservation_known")
    healthy_preserved = episode.get("healthy_components_preserved")
    corrupted = bool(healthy_known is not False and healthy_preserved is False)
    loop = bool(episode.get("loop_detected"))
    invalid = int(episode.get("invalid_action_count") or 0)

    costs = (
        weights.call_cost * charged
        + weights.unneeded_acquisition_cost * unneeded
        + (weights.tool_regret_cost * regret if regret is not None else 0.0)
    )
    penalties = (
        weights.false_commit * false_commits
        + weights.false_finalization * false_finalizations
        + (weights.healthy_component_corruption if corrupted else 0.0)
        + (weights.loop if loop else 0.0)
        + weights.invalid_action * invalid
    )
    total = weights.success * credit - costs - penalties
    return RewardBreakdown(
        total=float(total),
        credit=float(credit),
        basis=basis,
        escalation_request=request,
        calls=calls,
        charged_calls=charged,
        acquisitions=counts,
        unneeded_acquisitions=unneeded,
        tool_regret=regret,
        false_commits=false_commits,
        false_finalizations=false_finalizations,
        healthy_component_corrupted=corrupted,
        loop_detected=loop,
        invalid_actions=invalid,
        penalties=float(penalties),
        costs=float(costs),
    )


def load_expert_reference(path: str | Path | None) -> dict[str, int]:
    """Calls per root from an expert end-to-end run (``episodes.jsonl`` of expert_e2e)."""
    if path is None:
        return {}
    reference: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scenario_id = row.get("scenario_id")
            steps = row.get("steps")
            if scenario_id is None or steps is None:
                continue
            reference[str(scenario_id)] = int(steps)
    return reference


__all__ = [
    "ACQUISITION_TOOLS",
    "BASIS_CREDIT",
    "BUDGET_EXHAUSTED_CREDIT",
    "BUDGET_EXHAUSTED_REQUEST",
    "NEEDED_STREAMS",
    "REWARD_CONTRACT",
    "RewardBreakdown",
    "RewardWeights",
    "acquisition_counts",
    "episode_basis",
    "episode_reward",
    "escalation_request",
    "load_expert_reference",
    "unneeded_acquisitions",
]
