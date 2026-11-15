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

Where the open rules lift the commit rule
(``psse_env.actions.OPEN_RULE_COMMIT_REJECTED_CANDIDATE``) the policy may also
keep a candidate the acceptance check did not accept, so its disposal is a
choice too: the audit would keep it when its target is truly faulted and roll
it back otherwise.  ``lifecycle_decisions(..., rejected_choice=True)`` returns
those decisions with the accepted ones, in the episode's order, and every
term that reads a decision's label trains them alike (2026-11-14).  They are
told apart by their verdict (:func:`is_rejected_candidate`); the counts of
:func:`summarize_decisions` stay those of the accepted candidates, and
:func:`summarize_rejected` counts the others.

Nothing here reads the policy's view.  The truth comes from the scenario's
``audit.truth`` block (the compound and conventional suites carry it), as
``research.compound_errors.rollout.target_is_true`` reads it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import COMMIT_STATE, CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY, ROLLBACK_STATE

LIFECYCLE_CONTRACT = "rl_lifecycle_decisions_v1"
ACCEPTED_VERDICTS = frozenset({"ACCEPT_FINAL", "ACCEPT_PARTIAL"})
#: The candidates the acceptance check did not accept; a choice only where the open rules lift the commit rule.
REJECTED_CONTRACT = "rl_rejected_candidate_decisions_v1"
REJECTED_VERDICTS = frozenset({"REJECT", "INCONCLUSIVE"})
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


def _candidate_disposals(trace: Sequence[Mapping[str, Any]], truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every successful commit or rollback of the open candidate of a successful correction, with the audit's label.

    The fields both kinds of decision share; the readers below add
    ``contract`` and ``declined`` or ``kept``.  A disposal whose verdict is
    neither accepted nor rejected (none recorded) is no decision of either kind.
    """

    disposals: list[dict[str, Any]] = []
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
        if correction is None or (verdict not in ACCEPTED_VERDICTS and verdict not in REJECTED_VERDICTS):
            continue
        truthful = target_is_true(correction["tool"], correction["arguments"], truth) if truth.get("known", True) else None
        label = None if truthful is None else (COMMIT_STATE if truthful else ROLLBACK_STATE)
        number = step.get("step")
        disposals.append({
            "position": position,
            "step": int(number) if isinstance(number, int) and not isinstance(number, bool) else position,
            "tool": tool,
            "verdict": str(verdict),
            "correction_position": correction["position"],
            "correction_tool": correction["tool"],
            "arguments": correction["arguments"],
            "target_is_true": truthful,
            "label": label,
            "agrees": None if label is None else (tool == label),
        })
    return disposals


def _accepted_record(disposal: Mapping[str, Any]) -> dict[str, Any]:
    return {"contract": LIFECYCLE_CONTRACT, **disposal, "declined": disposal["tool"] == ROLLBACK_STATE}


def _rejected_record(disposal: Mapping[str, Any]) -> dict[str, Any]:
    return {"contract": REJECTED_CONTRACT, **disposal, "kept": disposal["tool"] == COMMIT_STATE}


def lifecycle_decisions(trace: Sequence[Mapping[str, Any]], truth: Mapping[str, Any], *,
                        rejected_choice: bool = False) -> list[dict[str, Any]]:
    """Every decision of an episode on an accepted candidate, with the audit's label.

    ``position`` indexes the policy steps (the trajectory's step records);
    ``step`` is the trace step's own number.  ``label`` is the tool the audit
    would choose, ``agrees`` whether the policy chose it; both are ``None``
    when the correction names no target the truth can be read for.

    ``rejected_choice`` says the episode ran with the commit rule lifted, so
    keeping or rolling back a rejected or inconclusive candidate was a choice
    as well: those decisions are returned too, in the episode's order, as
    :func:`rejected_candidate_decisions` records them (``kept`` in place of
    ``declined``).  With the rule in place every such candidate is rolled
    back and nothing is chosen, which is the default.
    """

    decisions: list[dict[str, Any]] = []
    for disposal in _candidate_disposals(trace, truth):
        if disposal["verdict"] in ACCEPTED_VERDICTS:
            decisions.append(_accepted_record(disposal))
        elif rejected_choice:
            decisions.append(_rejected_record(disposal))
    return decisions


def rejected_candidate_decisions(trace: Sequence[Mapping[str, Any]], truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every step that disposed of a candidate the acceptance check did not accept, with the audit's label.

    Under the commit rule of the open rules (``psse_env.actions.OPEN_RULE_COMMIT_REJECTED_CANDIDATE``)
    the policy may keep such a candidate, so its disposal is a choice like the
    decline of an accepted one: the audit would keep it when its target is a
    truly faulted component and roll it back otherwise.  With the rule closed
    every one of them is rolled back and nothing is chosen, so a caller asks
    only where the rule is lifted.  ``kept`` is a successful ``commit_state``;
    the other fields are those of :func:`lifecycle_decisions`.

    The evaluation records read these apart from the accepted candidates; a
    sampled rollout under the lifted rule carries them among its decisions
    (``lifecycle_decisions(..., rejected_choice=True)``), where the row terms
    train them.
    """

    return [_rejected_record(disposal) for disposal in _candidate_disposals(trace, truth)
            if disposal["verdict"] in REJECTED_VERDICTS]


def is_rejected_candidate(decision: Mapping[str, Any]) -> bool:
    """Whether a decision record is of a candidate the acceptance check did not accept (a choice under the open rules)."""

    return decision.get("contract") == REJECTED_CONTRACT or str(decision.get("verdict")) in REJECTED_VERDICTS


def accepted_decisions(decisions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The decisions on accepted candidates, in order."""

    return [decision for decision in decisions if not is_rejected_candidate(decision)]


def rejected_decisions(decisions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The decisions on rejected or inconclusive candidates, in order."""

    return [decision for decision in decisions if is_rejected_candidate(decision)]


def accepted_before(decisions: Sequence[Mapping[str, Any]], index: int) -> int:
    """How many decisions on accepted candidates come before ``decisions[index]``: the ordinal the strata and the
    diagnostics count candidates by, which a rejected candidate does not move."""

    return sum(1 for earlier in decisions[:index] if not is_rejected_candidate(earlier))


def kept_wrong_targets(decisions: Sequence[Mapping[str, Any]]) -> int:
    """Corrections an episode kept on components that are not truly faulted: accepted candidates committed and,
    under the lifted commit rule, rejected ones kept.  What the reward's audit of kept targets is given."""

    return (summarize_decisions(decisions)["wrong_target_committed"]
            + summarize_rejected(rejected_decisions(decisions))["wrong_target_kept"])


def summarize_rejected(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of the rejected candidates an episode (or an evaluation) kept or rolled back, by what their target was."""

    summary = {
        "decisions": 0,
        "kept": 0,
        "rolled_back": 0,
        "unlabelled": 0,
        "true_target_kept": 0,
        "wrong_target_kept": 0,
        "true_target_rolled_back": 0,
        "wrong_target_rolled_back": 0,
    }
    for decision in decisions:
        summary["decisions"] += 1
        kept = bool(decision.get("kept"))
        summary["kept" if kept else "rolled_back"] += 1
        truthful = decision.get("target_is_true")
        if truthful is None:
            summary["unlabelled"] += 1
        else:
            summary[("true" if truthful else "wrong") + "_target_" + ("kept" if kept else "rolled_back")] += 1
    return summary


def decision_sign(decision: Mapping[str, Any]) -> float | None:
    """``+1`` for a decision that agrees with the audit, ``-1`` for one that does not, ``None`` when unlabelled."""

    agrees = decision.get("agrees")
    if agrees is None:
        return None
    return 1.0 if agrees else -1.0


def summarize_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of an episode's (or a store's) decisions on accepted candidates: commits, declines, agreement with
    the audit.  Decisions on rejected candidates are left to :func:`summarize_rejected`."""

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
    for decision in accepted_decisions(decisions):
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
    "REJECTED_CONTRACT",
    "REJECTED_VERDICTS",
    "accepted_before",
    "accepted_decisions",
    "decision_sign",
    "is_rejected_candidate",
    "kept_wrong_targets",
    "lifecycle_decisions",
    "policy_trace_steps",
    "rejected_candidate_decisions",
    "rejected_decisions",
    "summarize_rejected",
    "scenario_truth",
    "summarize_decisions",
    "target_is_true",
]
