"""The paired scorecard of an RL checkpoint (workplan of 2026-11-10, "Evaluation and stopping rules").

    python -m research.rl.scorecard \\
        --learner S2=out/arm_b/iteration_001/evaluation_episodes.jsonl \\
        --reference S0=out/arm_b/baseline/evaluation_episodes.jsonl --reference control=out/arm_a/iteration_001/evaluation_episodes.jsonl \\
        --reference expert=data/teacher_map.json --expert expert --matched control --output out/arm_b/scorecard_s2

Every policy is read on the same roots.  The primary number is paired:
roots only the learner solves less roots only the reference solves, with
both counts shown and an exact two-sided sign test on the discordant pairs.
Beside it, for the learner and every reference that records it:

* audited and clean solves (clean: solved with no refused action, no
  refused commit and no loop), by compound class;
* the roots the rule expert fails that the learner solves;
* declines of true candidates, at an episode's first accepted candidate and
  at later ones, and in the classes that hold a breaker and a line fault;
* wrong targets kept, and episodes charged for a corrupted healthy component;
* wrong targets declined, how many were followed by a kept correction on a
  true target, and how many of those episodes were solved;
* refused actions, loops, false finalizations;
* tool calls (on all roots and on the roots both policies solve), the
  solves and corrections the tools ran inside a call, prompt and completion
  tokens, and time.

Three kinds of file are read: the driver's ``evaluation_episodes.jsonl``
(everything above), the ``episodes.jsonl`` of a compound-study run of the
rule expert, and a map of root to ``{success, clean, steps, wrong_kept,
class}`` (the cloning cell's ``teacher_map.json``).  What a file does not
record is left empty, never guessed.

:func:`promotion_check` and :func:`stop_rule` are the workplan's pilot rules
as code.  They allocate compute on small samples; neither is a test of
significance, and the scorecard prints the counts they rest on.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCORECARD_CONTRACT = "rl_paired_scorecard_v1"
MIXED_PREFIX = "rx_cb"
COMMIT, ROLLBACK = "commit_state", "rollback_state"
COST_KEYS = ("input_tokens", "output_tokens", "generation_seconds", "policy_seconds", "tool_seconds", "internal_wls_calls",
             "internal_correction_calls", "pair_refits")


# --------------------------------------------------------------------------- reading


def _driver_record(record: Mapping[str, Any]) -> dict[str, Any]:
    success = bool(record.get("truth_audited_task_success"))
    reward = record.get("reward") if isinstance(record.get("reward"), Mapping) else {}
    decisions = [dict(item) for item in record.get("lifecycle") or [] if isinstance(item, Mapping)]
    return {
        "root": str(record["scenario_id"]),
        "class": record.get("compound_class"),
        "success": success,
        "clean": bool(success and not int(record.get("invalid_action_count") or 0) and not int(record.get("false_commit_count") or 0)
                      and not record.get("loop_detected")),
        "steps": int(record.get("policy_steps") or record.get("steps") or 0),
        "invalid_actions": int(record.get("invalid_action_count") or 0),
        "false_commits": int(record.get("false_commit_count") or 0),
        "false_finalizations": int(record.get("false_finalization_count") or 0),
        "loop": bool(record.get("loop_detected")),
        "healthy_corrupted": bool(reward.get("healthy_component_corrupted")) if "healthy_component_corrupted" in reward else None,
        "decisions": decisions,
        "wrong_kept": sum(1 for item in decisions if item.get("target_is_true") is False and not item.get("declined")),
        "cost": dict(record.get("cost") or {}),
        "prompt_limit_refusals": int(record.get("prompt_limit_refusals") or 0),
    }


def _compound_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """A compound-study episode of the rule expert (``research.compound_errors.rollout.compact_episode``)."""

    success = bool(record.get("success"))
    kept = [item for item in record.get("corrections") or [] if isinstance(item, Mapping) and item.get("lifecycle") == COMMIT]
    return {
        "root": str(record["scenario_id"]),
        "class": record.get("class"),
        "success": success,
        "clean": bool(success and not int(record.get("invalid_action_count") or 0) and not int(record.get("false_commit_count") or 0)
                      and not record.get("loop_detected")),
        "steps": int(record.get("steps") or 0),
        "invalid_actions": int(record.get("invalid_action_count") or 0),
        "false_commits": int(record.get("false_commit_count") or 0),
        "false_finalizations": None,
        "loop": bool(record.get("loop_detected")),
        "healthy_corrupted": (None if record.get("healthy_components_preserved") is None
                              else not bool(record.get("healthy_components_preserved"))),
        "decisions": None,
        "wrong_kept": sum(1 for item in kept if item.get("target_is_true") is False),
        "cost": {},
        "prompt_limit_refusals": 0,
    }


def _map_record(root: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root": str(root), "class": entry.get("class"), "success": bool(entry.get("success")),
        "clean": bool(entry["clean"]) if entry.get("clean") is not None else None,
        "steps": int(entry["steps"]) if entry.get("steps") is not None else None,
        "invalid_actions": None, "false_commits": None, "false_finalizations": None, "loop": None, "healthy_corrupted": None,
        "decisions": None, "wrong_kept": int(entry["wrong_kept"]) if entry.get("wrong_kept") is not None else None,
        "cost": {}, "prompt_limit_refusals": 0,
    }


def load_policy(path: str | Path) -> dict[str, dict[str, Any]]:
    """Root to its normalized record, from any of the three kinds of file."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, Mapping) and "scenario_id" not in whole:
        return {str(root): _map_record(str(root), entry) for root, entry in whole.items() if isinstance(entry, Mapping)}
    records: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = _driver_record(row) if "truth_audited_task_success" in row else _compound_record(row)
        records[record["root"]] = record
    return records


# --------------------------------------------------------------------------- counting


def exact_sign_test(won: int, lost: int) -> float:
    """Two-sided exact sign test on the discordant pairs."""

    n = int(won) + int(lost)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(int(won), int(lost)) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired(learner: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]], key: str = "success") -> dict[str, Any]:
    """The learner against a reference on the roots both were run on."""

    roots = sorted(set(learner) & set(reference))
    usable = [root for root in roots if learner[root].get(key) is not None and reference[root].get(key) is not None]
    won = sorted(root for root in usable if learner[root][key] and not reference[root][key])
    lost = sorted(root for root in usable if not learner[root][key] and reference[root][key])
    both = [root for root in usable if learner[root][key] and reference[root][key]]
    return {
        "roots": len(usable), "roots_missing_in_one": len(set(learner) ^ set(reference)),
        "both": len(both), "learner_only": len(won), "reference_only": len(lost),
        "neither": len(usable) - len(both) - len(won) - len(lost),
        "net": len(won) - len(lost), "p_exact_two_sided": exact_sign_test(len(won), len(lost)),
        "learner_only_roots": won, "reference_only_roots": lost,
    }


def _total(records: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = [record.get(key) for record in records if record.get(key) is not None]
    return sum(int(value) for value in values) if values else None


def summarize_policy(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One policy on its roots: solves, process failures, the lifecycle decisions and what they led to, and the cost."""

    rows = list(records.values())
    decisions = [(record, index, decision) for record in rows for index, decision in enumerate(record.get("decisions") or [])]
    labelled = [item for item in decisions if item[2].get("target_is_true") is not None]
    true = [item for item in labelled if item[2]["target_is_true"]]
    wrong = [item for item in labelled if not item[2]["target_is_true"]]

    def declined(items: Sequence[tuple]) -> dict[str, Any]:
        count = sum(1 for _record, _index, decision in items if decision.get("declined"))
        return {"candidates": len(items), "declined": count, "rate": (count / len(items)) if items else None}

    first = lambda items: [item for item in items if item[1] == 0]  # noqa: E731
    later = lambda items: [item for item in items if item[1] > 0]  # noqa: E731
    mixed = lambda items: [item for item in items if str(item[0].get("class") or "").startswith(MIXED_PREFIX)]  # noqa: E731
    # A wrong target declined, then a correction on a true target kept, then the case solved.
    recoveries = {"wrong_target_declined": 0, "followed_by_a_kept_true_correction": 0, "and_solved": 0, "episodes": 0,
                  "episodes_solved": 0}
    for record in rows:
        own = record.get("decisions") or []
        hit = False
        for index, decision in enumerate(own):
            if decision.get("target_is_true") is False and decision.get("declined"):
                hit = True
                recoveries["wrong_target_declined"] += 1
                useful = any(later_one.get("target_is_true") is True and not later_one.get("declined") for later_one in own[index + 1:])
                recoveries["followed_by_a_kept_true_correction"] += int(useful)
                recoveries["and_solved"] += int(useful and record["success"])
        recoveries["episodes"] += int(hit)
        recoveries["episodes_solved"] += int(hit and record["success"])
    known_decisions = any(record.get("decisions") is not None for record in rows)
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"roots": 0, "solved": 0, "clean": 0})
    for record in rows:
        entry = by_class[str(record.get("class"))]
        entry["roots"] += 1
        entry["solved"] += int(record["success"])
        entry["clean"] += int(bool(record.get("clean")))
    solved = [record for record in rows if record["success"]]
    steps = [record["steps"] for record in rows if record.get("steps") is not None]
    costs = [record.get("cost") or {} for record in rows]
    corrupted = [record.get("healthy_corrupted") for record in rows if record.get("healthy_corrupted") is not None]
    return {
        "roots": len(rows),
        "solved": len(solved),
        "clean": sum(1 for record in rows if record.get("clean")) if any(record.get("clean") is not None for record in rows) else None,
        "by_class": {name: dict(entry) for name, entry in sorted(by_class.items())},
        "process": {"invalid_actions": _total(rows, "invalid_actions"), "false_commits": _total(rows, "false_commits"),
                    "false_finalizations": _total(rows, "false_finalizations"),
                    "loop_episodes": sum(1 for record in rows if record.get("loop")) if any(record.get("loop") is not None for record in rows) else None,
                    "prompt_limit_refusals": _total(rows, "prompt_limit_refusals")},
        "wrong_targets_kept": _total(rows, "wrong_kept"),
        "episodes_keeping_a_wrong_target": (sum(1 for record in rows if (record.get("wrong_kept") or 0) > 0)
                                            if any(record.get("wrong_kept") is not None for record in rows) else None),
        "healthy_component_corrupted_episodes": sum(1 for value in corrupted if value) if corrupted else None,
        "decisions": None if not known_decisions else {
            "labelled": len(labelled),
            "true_target": declined(true), "true_first_candidate": declined(first(true)), "true_later_candidate": declined(later(true)),
            "true_in_mixed_classes": declined(mixed(true)),
            "wrong_target": declined(wrong), "wrong_first_candidate": declined(first(wrong)),
            "wrong_targets_kept": sum(1 for _record, _index, decision in wrong if not decision.get("declined")),
            "recovery": recoveries,
        },
        "cost": {
            "mean_tool_calls": (sum(steps) / len(steps)) if steps else None,
            "mean_tool_calls_solved": (sum(record["steps"] for record in solved if record.get("steps") is not None) / len(solved)) if solved else None,
            **{key: (sum(cost[key] for cost in costs if cost.get(key) is not None) if any(cost.get(key) is not None for cost in costs) else None)
               for key in COST_KEYS},
        },
    }


def tool_calls_on_shared_solves(learner: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Tool calls of the two policies on the roots both solve, where a difference is efficiency and not a failure's length."""

    roots = [root for root in set(learner) & set(reference) if learner[root]["success"] and reference[root]["success"]
             and learner[root].get("steps") is not None and reference[root].get("steps") is not None]
    if not roots:
        return {"roots": 0, "learner": None, "reference": None, "fewer": 0, "same": 0, "more": 0}
    mine = [learner[root]["steps"] for root in roots]
    theirs = [reference[root]["steps"] for root in roots]
    return {"roots": len(roots), "learner": sum(mine) / len(roots), "reference": sum(theirs) / len(roots),
            "fewer": sum(1 for a, b in zip(mine, theirs) if a < b), "same": sum(1 for a, b in zip(mine, theirs) if a == b),
            "more": sum(1 for a, b in zip(mine, theirs) if a > b)}


def build_scorecard(
    learners: Mapping[str, Mapping[str, Mapping[str, Any]]],
    references: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expert: str | None = None,
    matched: str | None = None,
) -> dict[str, Any]:
    """The scorecard of one or more learner checkpoints (or seeds) against the references."""

    if expert is not None and expert not in references:
        raise ValueError(f"the expert reference {expert!r} is not among the references {sorted(references)}")
    if matched is not None and matched not in references:
        raise ValueError(f"the matched reference {matched!r} is not among the references {sorted(references)}")
    card: dict[str, Any] = {"contract": SCORECARD_CONTRACT, "expert": expert, "matched_reference": matched,
                            "references": {name: summarize_policy(records) for name, records in references.items()}, "learners": {}}
    for label, records in learners.items():
        entry: dict[str, Any] = {"summary": summarize_policy(records), "against": {}}
        for name, reference in references.items():
            entry["against"][name] = {
                "solved": paired(records, reference, "success"),
                "clean": paired(records, reference, "clean"),
                "tool_calls_on_shared_solves": tool_calls_on_shared_solves(records, reference),
            }
        if expert is not None:
            failed = sorted(root for root, record in references[expert].items() if not record["success"])
            recovered = [root for root in failed if root in records and records[root]["success"]]
            entry["expert_failures_recovered"] = {"expert_failed_roots": len(failed), "recovered": len(recovered), "roots": recovered}
        if matched is not None:
            entry["promotion"] = promotion_check(entry, card["references"][matched], matched=matched)
        card["learners"][label] = entry
    if len(learners) > 1:
        solved = [entry["summary"]["solved"] for entry in card["learners"].values()]
        card["across_learners"] = {"labels": list(learners), "solved": solved, "min": min(solved), "max": max(solved),
                                   "mean": sum(solved) / len(solved)}
    return card


# --------------------------------------------------------------------------- the pilot rules


def promotion_check(entry: Mapping[str, Any], reference_summary: Mapping[str, Any], *, matched: str) -> dict[str, Any]:
    """The workplan's promotion rule for a pilot, on one learner's scorecard entry.

    Promote only with positive paired net solves over the matched
    reference, at least one expert-failed root recovered, a nonnegative
    paired clean-success change, and no increase in healthy-component
    corruption.  ``inspect`` lists what the workplan says to read before
    the budget grows: declines of true candidates in the mixed classes and
    at later candidates.
    """

    against = entry["against"][matched]
    recovered = (entry.get("expert_failures_recovered") or {}).get("recovered")
    mine = entry["summary"].get("healthy_component_corrupted_episodes")
    theirs = reference_summary.get("healthy_component_corrupted_episodes")
    conditions = {
        "positive_paired_net_solves": against["solved"]["net"] > 0,
        "expert_failed_roots_recovered": None if recovered is None else recovered > 0,
        "nonnegative_paired_clean_change": against["clean"]["net"] >= 0 if against["clean"]["roots"] else None,
        "no_more_healthy_component_corruption": None if mine is None or theirs is None else mine <= theirs,
    }
    decisions = entry["summary"].get("decisions") or {}
    return {
        "matched_reference": matched,
        "conditions": conditions,
        "promote": all(value is True for value in conditions.values()),
        "undetermined": sorted(name for name, value in conditions.items() if value is None),
        "counts": {"learner_only": against["solved"]["learner_only"], "reference_only": against["solved"]["reference_only"],
                   "p_exact_two_sided": against["solved"]["p_exact_two_sided"], "clean_net": against["clean"]["net"],
                   "healthy_component_corrupted_episodes": [mine, theirs]},
        "inspect": {"true_candidates_declined_in_mixed_classes": decisions.get("true_in_mixed_classes"),
                    "true_candidates_declined_at_later_candidates": decisions.get("true_later_candidate")},
        "note": "a rule for allocating compute on a small sample, not a test of significance",
    }


def stop_rule(baseline_to_iterations: Sequence[Mapping[str, Any]], *, screen_iterations: int = 2, extra_iterations: int = 1,
              stop_after_without_gain: int = 3) -> dict[str, Any]:
    """The new series' stopping rule, from an arm's paired results against its own baseline, in order.

    Each entry is :func:`paired` of a completed, valid iteration's
    evaluation against the arm's baseline under its own observation mode.
    The screen is ``screen_iterations``; at most ``extra_iterations`` more
    may be run to settle a concrete uncertain signal; expansion stops after
    ``stop_after_without_gain`` completed iterations in a row without a
    positive paired net.  Nothing else (a better local decline rate, a
    lower loss) resets the count.
    """

    gains = [int(item["net"]) > 0 for item in baseline_to_iterations]
    streak = 0
    for gain in gains:
        streak = 0 if gain else streak + 1
    completed = len(gains)
    return {
        "completed_valid_iterations": completed,
        "paired_net_by_iteration": [int(item["net"]) for item in baseline_to_iterations],
        "iterations_without_gain_in_a_row": streak,
        "screen_complete": completed >= screen_iterations,
        "further_iterations_allowed": max(0, screen_iterations + extra_iterations - completed),
        "stop": streak >= stop_after_without_gain or (completed >= screen_iterations + extra_iterations and not any(gains)),
        "rule": f"screen of {screen_iterations} iterations, at most {extra_iterations} more for a concrete uncertain signal; stop after "
                f"{stop_after_without_gain} completed valid iterations without end-to-end improvement over the arm's own baseline",
    }


# --------------------------------------------------------------------------- writing


def _cell(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(card: Mapping[str, Any], *, title: str = "Paired scorecard") -> str:
    policies = {**{f"{name} (reference)": summary for name, summary in card["references"].items()},
                **{label: entry["summary"] for label, entry in card["learners"].items()}}
    names = list(policies)
    lines = [f"# {title}", "", "## Solves and process", "",
             "| | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]

    def row(label: str, read) -> None:
        lines.append(f"| {label} | " + " | ".join(_cell(read(policies[name])) for name in names) + " |")

    row("roots", lambda s: s["roots"])
    row("solved", lambda s: s["solved"])
    row("clean solves", lambda s: s["clean"])
    row("wrong targets kept", lambda s: s["wrong_targets_kept"])
    row("episodes charged for a corrupted healthy component", lambda s: s["healthy_component_corrupted_episodes"])
    row("refused actions", lambda s: s["process"]["invalid_actions"])
    row("refused commits", lambda s: s["process"]["false_commits"])
    row("false finalizations", lambda s: s["process"]["false_finalizations"])
    row("loop episodes", lambda s: s["process"]["loop_episodes"])
    row("steps refused at the prompt limit", lambda s: s["process"]["prompt_limit_refusals"])
    row("tool calls per root", lambda s: s["cost"]["mean_tool_calls"])
    row("tool calls per solved root", lambda s: s["cost"]["mean_tool_calls_solved"])
    row("solves inside tool calls", lambda s: s["cost"]["internal_wls_calls"])
    row("corrections inside tool calls", lambda s: s["cost"]["internal_correction_calls"])
    row("pair refits", lambda s: s["cost"]["pair_refits"])
    row("prompt tokens", lambda s: s["cost"]["input_tokens"])
    row("completion tokens", lambda s: s["cost"]["output_tokens"])
    row("policy seconds", lambda s: s["cost"]["policy_seconds"])
    row("tool seconds", lambda s: s["cost"]["tool_seconds"])
    lines += ["", "## Lifecycle decisions", "", "| | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]

    def decided(label: str, key: str) -> None:
        def read(summary):
            entry = (summary.get("decisions") or {}).get(key)
            return None if entry is None else f"{entry['declined']} of {entry['candidates']}"
        row(label, read)

    decided("true candidates declined", "true_target")
    decided("at the first accepted candidate", "true_first_candidate")
    decided("at a later candidate", "true_later_candidate")
    decided("in classes with a breaker and a line fault", "true_in_mixed_classes")
    decided("wrong candidates declined", "wrong_target")
    row("wrong declines followed by a kept true correction", lambda s: ((s.get("decisions") or {}).get("recovery") or {}).get("followed_by_a_kept_true_correction"))
    row("and the case solved", lambda s: ((s.get("decisions") or {}).get("recovery") or {}).get("and_solved"))
    for label, entry in card["learners"].items():
        lines += ["", f"## {label}, paired", "",
                  "| against | roots | both | only the learner | only the reference | net | exact p | clean net | calls on shared solves (learner / reference) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, against in entry["against"].items():
            solved, clean, calls = against["solved"], against["clean"], against["tool_calls_on_shared_solves"]
            lines.append(f"| {name} | {solved['roots']} | {solved['both']} | {solved['learner_only']} | {solved['reference_only']} | "
                         f"{solved['net']:+d} | {solved['p_exact_two_sided']:.3f} | {clean['net']:+d} | "
                         f"{_cell(calls['learner'])} / {_cell(calls['reference'])} on {calls['roots']} |")
        recovered = entry.get("expert_failures_recovered")
        if recovered is not None:
            lines += ["", f"Roots the expert fails that the learner solves: {recovered['recovered']} of {recovered['expert_failed_roots']}"
                          + (f" ({', '.join(recovered['roots'])})." if recovered["roots"] else ".")]
        promotion = entry.get("promotion")
        if promotion is not None:
            verdict = "promote" if promotion["promote"] else "do not promote"
            lines += ["", f"Promotion rule against {promotion['matched_reference']}: {verdict}. "
                          + "; ".join(f"{name.replace('_', ' ')}: {'yes' if value else 'not recorded' if value is None else 'no'}"
                                      for name, value in promotion["conditions"].items()) + f". {promotion['note'].capitalize()}."]
    classes = sorted({name for summary in policies.values() for name in summary["by_class"]})
    lines += ["", "## Solved by compound class", "", "| class | roots | " + " | ".join(names) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in names) + " |"]
    for name in classes:
        roots = next((summary["by_class"][name]["roots"] for summary in policies.values() if name in summary["by_class"]), 0)
        lines.append(f"| {name} | {roots} | " + " | ".join(_cell((policies[policy]["by_class"].get(name) or {}).get("solved")) for policy in names) + " |")
    if "across_learners" in card:
        across = card["across_learners"]
        lines += ["", f"Across {', '.join(across['labels'])}: solved {across['solved']} (mean {across['mean']:.1f}, "
                      f"lowest {across['min']}, highest {across['max']}); each is a separate run, shown separately above."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learner", action="append", required=True, metavar="LABEL=FILE")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--expert", default=None, help="the name of the reference that is the rule expert")
    parser.add_argument("--matched", default=None, help="the name of the reference the promotion rule compares with")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Paired scorecard")
    args = parser.parse_args(argv)

    def split(items: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
        return {item.partition("=")[0]: load_policy(item.partition("=")[2]) for item in items}

    card = build_scorecard(split(args.learner), split(args.reference), expert=args.expert, matched=args.matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(card, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    text = markdown(card, title=args.title)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
