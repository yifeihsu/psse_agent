#!/usr/bin/env python3
"""Per-episode outcomes of one evaluation, in the compound study's record form.

    episode_outcomes.py --suite SUITE.json --evaluation-dir OUT/eval/NAME

evaluate_suite.py keeps the evaluator's whole report ({slot}_eval.json, with every observation,
hundreds of MB per policy) and a short per-root file without the lifecycle fields.  This script
reads the report and the suite and writes, for each policy in policies.json:

* outcomes_LABEL.jsonl: one ``research.compound_errors.rollout.compact_episode`` record per episode
  (scenario id, class from the suite's audit.compound.class, success, actions, slim trace, every
  correction with its verdict, lifecycle step and whether its target is a true one), plus the
  evaluator's ``decline_accounting`` and ``false_rollback_count``, and each decline as a
  ``gate_decisions`` entry;
* outcomes.json: per policy the measures of the spec (solved, wrong corrections kept, actions,
  declines against the audit), overall, per class, and for the breaker classes apart from the rest.

The records have the fields ``research.compound_errors.commit_reader_loop compare`` reads, so the
before/after pairing and the comparison with a teacher's stored episodes need no other code.
Everything here is post-processing: the class and the truth come from the suite and never reach the
model.  A root's truth names breakers exactly on the breaker classes (conventional suites included).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.compound_errors.rollout import compact_episode, load_suite, truth_of  # noqa: E402

OUTCOMES_CONTRACT = "compound_cloning_outcomes_v1"
ACCEPTED = ("ACCEPT_FINAL", "ACCEPT_PARTIAL")
COMMIT, ROLLBACK = "commit_state", "rollback_state"


def outcome_record(episode: Mapping[str, Any], root: Mapping[str, Any]) -> dict[str, Any]:
    record = compact_episode(episode, root)
    truth = truth_of(root)
    record["physical_root"] = episode.get("physical_root")
    record["group"] = "breaker" if truth["breakers"] or truth["dangling_lines"] else "other"
    record["policy_steps"] = episode.get("policy_steps")
    record["false_rollback_count"] = int(episode.get("false_rollback_count") or 0)
    record["decline_accounting"] = dict(episode.get("decline_accounting") or {})
    record["evaluator_error"] = episode.get("evaluator_error")
    if episode.get("evaluator_error"):
        record["error"] = str(episode["evaluator_error"])
    record["gate_decisions"] = [
        {"correction_step": c["step"], "tool": c["tool"], "arguments": c["arguments"], "verdict": c["verdict"],
         "declined": True, "target_is_true": c.get("target_is_true")}
        for c in record["corrections"] if c.get("lifecycle") == ROLLBACK and c.get("verdict") in ACCEPTED
    ]
    record["declines"] = len(record["gate_decisions"])
    return record


def _measures(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kept = [c for r in records for c in r.get("corrections") or [] if c.get("lifecycle") == COMMIT]
    declined = [d for r in records for d in r.get("gate_decisions") or []]
    solved = [r for r in records if r.get("success")]
    return {
        "episodes": len(records),
        "solved": len(solved),
        "wrong_corrections_kept": sum(1 for c in kept if c.get("target_is_true") is False),
        "episodes_keeping_wrong": sum(1 for r in records if any(
            c.get("lifecycle") == COMMIT and c.get("target_is_true") is False for c in r.get("corrections") or [])),
        "true_corrections_kept": sum(1 for c in kept if c.get("target_is_true") is True),
        "declines": len(declined),
        "true_target_declined": sum(1 for d in declined if d.get("target_is_true") is True),
        "wrong_target_declined": sum(1 for d in declined if d.get("target_is_true") is False),
        "episodes_with_decline": sum(1 for r in records if r.get("gate_decisions")),
        "mean_actions": sum(int(r.get("steps") or 0) for r in records) / max(1, len(records)),
        "mean_actions_solved": sum(int(r.get("steps") or 0) for r in solved) / max(1, len(solved)),
        "invalid_action_episodes": sum(1 for r in records if int(r.get("invalid_action_count") or 0) > 0),
        "false_commit_episodes": sum(1 for r in records if int(r.get("false_commit_count") or 0) > 0),
        "false_rollbacks": sum(int(r.get("false_rollback_count") or 0) for r in records),
        "loop_episodes": sum(1 for r in records if r.get("loop_detected")),
        "errors": sum(1 for r in records if r.get("error")),
        # The evaluator's own count against the lifecycle steps read from the trace; these must agree.
        "episodes_decline_count_differs": sum(
            1 for r in records if int((r.get("decline_accounting") or {}).get("declines") or 0) != int(r.get("declines") or 0)),
        "terminal_outcomes": dict(Counter(str(r.get("terminal_outcome") or "nonterminal") for r in records)),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[str(record.get("class") or record.get("family"))].append(record)
        by_group[str(record.get("group"))].append(record)
    return {
        "overall": _measures(records),
        "by_group": {name: _measures(rows) for name, rows in sorted(by_group.items())},
        "by_class": {name: _measures(rows) for name, rows in sorted(by_class.items())},
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def process(suite: Path, evaluation_dir: Path) -> dict[str, Any]:
    identity = json.loads((evaluation_dir / "policies.json").read_text(encoding="utf-8"))
    roots = {str(root["execution"]["scenario_id"]): root for root in load_suite(suite)}
    summary: dict[str, Any] = {"contract": OUTCOMES_CONTRACT, "suite": identity.get("suite"), "seed": identity.get("seed"),
                               "max_steps": identity.get("max_steps"), "research_profile": identity.get("research_profile"), "policies": {}}
    for slot, policy in sorted((identity.get("policies") or {}).items()):
        payload = json.loads((evaluation_dir / f"{slot}_eval.json").read_text(encoding="utf-8"))
        records = []
        for episode in payload["suite_metrics"]["episodes"]:
            scenario_id = str(episode.get("scenario_id"))
            try:
                records.append(outcome_record(episode, roots[scenario_id]))
            except Exception as exc:  # the report stays on disk; the record names what could not be read
                records.append({"scenario_id": scenario_id, "success": bool(episode.get("truth_audited_task_success")),
                                "error": f"{type(exc).__name__}: {exc}", "steps": int(episode.get("policy_steps") or 0),
                                "corrections": [], "tools": [], "gate_decisions": []})
        label = str(policy["label"])
        _atomic_text(evaluation_dir / f"outcomes_{label}.jsonl",
                     "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records))
        summary["policies"][label] = {"slot": slot, "adapter": policy.get("adapter"), "tree_sha256": policy.get("tree_sha256"), **summarize(records)}
    _atomic_text(evaluation_dir / "outcomes.json", json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = process(args.suite, args.evaluation_dir)
    print(json.dumps({label: block["overall"] for label, block in summary["policies"].items()}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
