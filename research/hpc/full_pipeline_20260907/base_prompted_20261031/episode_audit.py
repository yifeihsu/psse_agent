#!/usr/bin/env python3
"""How the prompted, untuned base model's episodes go (2026-10-31; note section 11.5).

    episode_audit.py --cell-dir PIPE [--label base_prompted] [--output OUT.json]

Read-only, over the full records (``bc0_eval.json``; the base model sits in the
evaluator's first slot) of the base model's seven test evaluations and its dry
run: actions, refused actions and the outputs that did not parse (raw names);
successes by how the episode ended and by the truth audit's basis; failures by
family, ending and the audit's first reason; on the 40-action sets, how many
actions a success that ran to the budget spent after its last state change and
on which tools; and how often the action after a refusal is the refusal's own
``valid_next_actions`` suggestion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

SETS = {"draw1@40": ("test_40", 40), "draw2@40": ("test2_40", 40), "draw1@12": ("test_budget12", 12),
        "draw1@16": ("test_budget16", 16), "draw1@20": ("test_budget20", 20), "draw2@12": ("test2_budget12", 12),
        "draw2@20": ("test2_budget20", 20), "dryrun@40": ("validation_dryrun", 40)}
BREAKERS = {"evaluation_repeated_nonadvancing_failure", "evaluation_repeated_nonadvancing_diagnostic",
            "evaluation_specialized_tool_budget_exhausted"}
INVALID_ACTION = "__invalid_action__"


def ending(episode: dict[str, Any], horizon: int) -> str:
    trace = episode.get("trace") or []
    if trace and trace[-1].get("error_code") in BREAKERS:
        return "breaker"
    if episode.get("terminal"):
        return f"terminal {episode.get('terminal_outcome')}"
    return "budget" if len(trace) >= horizon else "other"


def keyed(counter: Counter) -> dict[str, int]:
    return {" | ".join(map(str, key)) if isinstance(key, tuple) else str(key): value
            for key, value in sorted(counter.items(), key=str)}


def audit(path: Path, horizon: int) -> dict[str, Any]:
    episodes = json.loads(path.read_text(encoding="utf-8"))["suite_metrics"]["episodes"]
    steps = refused = 0
    unparsed, basis, failures, first_reason, after_tools = (Counter() for _ in range(5))
    suggestion, closing = Counter(), Counter()
    after: list[int] = []
    for episode in episodes:
        trace = episode.get("trace") or []
        steps += len(trace)
        for index, step in enumerate(trace):
            action = step.get("action") or {}
            if action.get("tool") == INVALID_ACTION:
                unparsed[str((action.get("arguments") or {}).get("error_detail"))[:200]] += 1
            if step.get("execution_status") == "success":
                continue
            refused += 1
            hints = (step.get("policy_tool_output") or {}).get("valid_next_actions") or []
            if hints and index + 1 < len(trace):
                following = (trace[index + 1].get("action") or {}).get("tool")
                suggested = {hint.get("tool") for hint in hints}
                suggestion[(str(action.get("tool")), "followed" if following in suggested else "not followed")] += 1
        assessment = (episode.get("audit") or {}).get("truth_audited_task_assessment") or {}
        end = ending(episode, horizon)
        success = bool(episode.get("truth_audited_task_success"))
        if success:
            basis[(end, assessment.get("basis"))] += 1
        else:
            failures[(episode.get("family"), end)] += 1
            first_reason[(end, (assessment.get("reasons") or ["none"])[0])] += 1
        if success and end == "budget":
            last = max((i for i, step in enumerate(trace)
                        if step.get("execution_status") == "success" and step.get("state_mutated")), default=-1)
            after.append(len(trace) - 1 - last)
            for step in trace[last + 1:]:
                tool = (step.get("action") or {}).get("tool")
                after_tools[f"{tool}: {step.get('execution_status')} {step.get('error_code') or ''}".strip()] += 1
            closing["tried finalize_diagnosis" if any((s.get("action") or {}).get("tool") == "finalize_diagnosis"
                                                      for s in trace) else "never tried finalize_diagnosis"] += 1
    result: dict[str, Any] = {
        "episodes": len(episodes), "steps": steps, "refused_steps": refused,
        "unparsed_outputs": dict(unparsed.most_common()),
        "successes_by_ending_and_basis": keyed(basis),
        "failures_by_family_and_ending": keyed(failures),
        "failures_by_ending_and_first_audit_reason": keyed(first_reason),
        "next_action_after_a_refusal_with_a_suggestion": keyed(suggestion),
    }
    if after:
        result["budget_successes"] = {
            "episodes": len(after), "actions_after_last_state_change_mean": round(sum(after) / len(after), 2),
            "min": min(after), "max": max(after), "closing": dict(closing),
            "tools_after_last_state_change": dict(after_tools.most_common(12))}
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--label", default="base_prompted")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    evaluations = args.cell_dir / "out" / "extra" / "evaluations"
    result = {name: audit(evaluations / f"{stem}_{args.label}" / "bc0_eval.json", horizon)
              for name, (stem, horizon) in SETS.items()}
    text = json.dumps(result, indent=1, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
