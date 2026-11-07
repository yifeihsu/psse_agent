"""The lifecycle decisions of an episode, labelled by the truth audit (RL plan of 2026-11-05, section 6).

With the environment's ``lifecycle_choice`` option the policy may decline an
accepted candidate: a ``rollback_state`` of a candidate the verifier accepted
is a legal choice, not a false rollback.  The case reward reaches that choice
only through the outcome many steps later, and with four samples per case
most groups show no difference there.  The audit knows, for every accepted
candidate, whether its target is a truly faulted component, so each decision
can be scored on its own:

* ``commit_state`` on an accepted candidate whose target is true, and
  ``rollback_state`` (a decline) on one whose target is wrong, agree with the
  audit;
* the other two disagree.

A decision is read from the evaluator's trace: the step that disposes of the
open candidate carries the verifier's offline verdict
(``candidate_disposition_offline``), and the candidate's correction is the
last successful correction before it.  Only accepted verdicts are decisions:
a rejected or inconclusive candidate must be rolled back (the process gate
refuses its commit), so there is nothing to score.  A refused lifecycle call
is an invalid action, scored by the episode reward already.

The label is supervision by the truth audit, the same source the outcome
credit uses, never by the teacher.  :func:`psse_env.rl.rows.build_rows`
turns it into a per-row term (``RewardWeights.lifecycle_credit``): a decision
row gets ``+credit`` when it agrees and ``-credit`` when it does not, on top
of its group advantage, and a labelled decision is trained even when its
group's samples all scored the same.  For a two-way choice with both outcomes
known this is the per-decision reward the plan describes; it uses the tokens
the policy sampled, as every other row does, instead of a re-rendered target.

Nothing here reads the policy's view.  The truth comes from the scenario's
``audit.truth`` block (the compound and conventional suites carry it), as
``research.compound_errors.rollout.target_is_true`` reads it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import COMMIT_STATE, CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY, ROLLBACK_STATE

LIFECYCLE_CONTRACT = "rl_lifecycle_decisions_v1"
ACCEPTED_VERDICTS = frozenset({"ACCEPT_FINAL", "ACCEPT_PARTIAL"})
CORRECTION_TOOLS = frozenset({CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY})
DECISION_TOOLS = frozenset({COMMIT_STATE, ROLLBACK_STATE})


def scenario_truth(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """The faulted components of a root (``audit.truth``): meters, lines, breakers and dangling lines."""

    audit = scenario.get("audit") if isinstance(scenario.get("audit"), Mapping) else {}
    truth = audit.get("truth") if isinstance(audit.get("truth"), Mapping) else {}
    return {
        "meters": {int(item["index"]) for item in truth.get("true_measurement_errors") or []
                   if isinstance(item, Mapping) and item.get("index") is not None},
        "lines": {int(item["line_index1"]) for item in truth.get("true_parameter_errors") or []
                  if isinstance(item, Mapping) and item.get("line_index1") is not None},
        "breakers": {str(item["cb_name"]) for item in truth.get("true_topology_errors") or []
                     if isinstance(item, Mapping) and item.get("cb_name")},
        "dangling_lines": {int(item["line_index1"]) for item in truth.get("true_topology_errors") or []
                           if isinstance(item, Mapping) and item.get("line_index1")},
        "known": bool(truth),
    }


def target_is_true(tool: str, arguments: Mapping[str, Any], truth: Mapping[str, Any]) -> bool | None:
    """Whether a correction's target is a truly faulted component; ``None`` when the arguments name none."""

    arguments = arguments or {}
    if tool == CORRECT_MEASUREMENTS:
        group = arguments.get("suspect_group")
        if group is None:
            index = arguments.get("index", arguments.get("measurement_index"))
            group = [index] if index is not None else []
        try:
            targets = {int(item) for item in group}
        except (TypeError, ValueError):
            return None
        return bool(targets) and targets <= set(truth.get("meters") or ())
    if tool == CORRECT_PARAMETERS:
        line = arguments.get("line_index")
        return int(line) in set(truth.get("lines") or ()) if isinstance(line, int) and not isinstance(line, bool) else None
    if tool == CORRECT_TOPOLOGY:
        if arguments.get("cb_name") is not None:
            return str(arguments["cb_name"]) in set(truth.get("breakers") or ())
        line = arguments.get("line_index")
        return int(line) in set(truth.get("dangling_lines") or ()) if isinstance(line, int) and not isinstance(line, bool) else None
    return None


def policy_trace_steps(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The policy's own steps of an evaluator trace, in order (interventions left out)."""

    return [step for step in trace if isinstance(step, Mapping) and not step.get("intervention")]


def lifecycle_decisions(trace: Sequence[Mapping[str, Any]], truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every decision of an episode on an accepted candidate, with the audit's label.

    ``position`` indexes the policy steps (the trajectory's step records);
    ``step`` is the trace step's own number.  ``label`` is the tool the audit
    would choose, ``agrees`` whether the policy chose it; both are ``None``
    when the correction names no target the truth can be read for.
    """

    decisions: list[dict[str, Any]] = []
    open_correction: dict[str, Any] | None = None
    for position, step in enumerate(policy_trace_steps(trace)):
        action = step.get("action") if isinstance(step.get("action"), Mapping) else {}
        tool = str(action.get("tool") or "")
        status = step.get("execution_status")
        if tool in CORRECTION_TOOLS:
            if status == "success":
                arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
                open_correction = {
                    "position": position,
                    "tool": tool,
                    "arguments": {str(key): value for key, value in arguments.items() if key != "state_id"},
                }
            continue
        if tool not in DECISION_TOOLS or status != "success":
            continue
        verdict = step.get("candidate_disposition_offline")
        correction = open_correction
        open_correction = None
        if verdict not in ACCEPTED_VERDICTS or correction is None:
            continue
        truthful = target_is_true(correction["tool"], correction["arguments"], truth) if truth.get("known", True) else None
        label = None if truthful is None else (COMMIT_STATE if truthful else ROLLBACK_STATE)
        number = step.get("step")
        decisions.append({
            "contract": LIFECYCLE_CONTRACT,
            "position": position,
            "step": int(number) if isinstance(number, int) and not isinstance(number, bool) else position,
            "tool": tool,
            "verdict": str(verdict),
            "declined": tool == ROLLBACK_STATE,
            "correction_position": correction["position"],
            "correction_tool": correction["tool"],
            "arguments": correction["arguments"],
            "target_is_true": truthful,
            "label": label,
            "agrees": None if label is None else (tool == label),
        })
    return decisions


def decision_sign(decision: Mapping[str, Any]) -> float | None:
    """``+1`` for a decision that agrees with the audit, ``-1`` for one that does not, ``None`` when unlabelled."""

    agrees = decision.get("agrees")
    if agrees is None:
        return None
    return 1.0 if agrees else -1.0


def summarize_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of an episode's (or a store's) decisions: commits, declines, agreement with the audit."""

    summary = {
        "decisions": 0,
        "commits": 0,
        "declines": 0,
        "agreeing": 0,
        "disagreeing": 0,
        "unlabelled": 0,
        "true_target_declined": 0,
        "wrong_target_committed": 0,
        "wrong_target_declined": 0,
    }
    for decision in decisions:
        summary["decisions"] += 1
        declined = bool(decision.get("declined"))
        summary["declines" if declined else "commits"] += 1
        agrees = decision.get("agrees")
        if agrees is None:
            summary["unlabelled"] += 1
        elif agrees:
            summary["agreeing"] += 1
        else:
            summary["disagreeing"] += 1
        truthful = decision.get("target_is_true")
        if declined and truthful is True:
            summary["true_target_declined"] += 1
        if declined and truthful is False:
            summary["wrong_target_declined"] += 1
        if not declined and truthful is False:
            summary["wrong_target_committed"] += 1
    return summary


__all__ = [
    "ACCEPTED_VERDICTS",
    "CORRECTION_TOOLS",
    "DECISION_TOOLS",
    "LIFECYCLE_CONTRACT",
    "decision_sign",
    "lifecycle_decisions",
    "policy_trace_steps",
    "scenario_truth",
    "summarize_decisions",
    "target_is_true",
]
