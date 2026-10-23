#!/usr/bin/env python3
"""Per-policy table of a finished cell from its two round summaries.

    python policy_table.py --r1 out/r1/round_summary.json --r2 out/r2/round_summary.json \\
        [--families measurement multi_measurement ...] [--output table.md]

Expert, BC0 and R1 are read from round 1, R2 from round 2 (R1 is scored again
in round 2; the table states whether the two evaluations agree).  For each
policy: truth-audited successes per family, then over the selected families
the episodes with an invalid action, with a false commit and with a loop,
the mean actions and the mean WLS calls per episode (run_wls and
verify_candidate, summarize.py's ``wls_calls``; "n/a" for summaries written
before that field existed) and the bounded comparisons.  The pattern of
docs/papers/pesgm2027/evidence/conventional_subset_from_round_summaries.py;
the default families are the seven conventional ones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

CONVENTIONAL = (
    "measurement", "multi_measurement", "parameter", "topology",
    "measurement+parameter", "measurement+topology", "no_error",
)
LABELS = {
    "measurement": "Single bad meter",
    "multi_measurement": "Multiple bad meters",
    "parameter": "Line R/X error",
    "topology": "Breaker status error",
    "measurement+parameter": "Meter + R/X error",
    "measurement+topology": "Meter + breaker status",
    "no_error": "Clean controls",
}
COMPARISONS = "tool_counts.compare_parameter_candidates"


def _episodes_with(histogram: Mapping[str, int]) -> int:
    """Episodes whose count is non-zero (or whose flag is true)."""
    return sum(int(v) for k, v in histogram.items() if str(k) not in ("0", "False", "None"))


def _total(histogram: Mapping[str, int]) -> float:
    return sum(float(k) * int(v) for k, v in histogram.items())


def policy_metrics(outcomes: Mapping[str, Any], families: Sequence[str]) -> dict[str, Any]:
    per_family = {}
    n = successes = invalid = false_commit = loops = 0
    steps = 0.0
    wls: float | None = 0.0
    comparisons = 0
    for family in families:
        block = outcomes.get(family)
        if block is None:
            continue
        success = block["truth_audited_task_success"]
        count = sum(int(v) for v in success.values())
        per_family[family] = {"episodes": count, "successes": int(success.get("True", 0))}
        n += count
        successes += int(success.get("True", 0))
        invalid += _episodes_with(block["invalid_action_count"])
        false_commit += _episodes_with(block["false_commit_count"])
        loops += int(block["loop_detected"].get("True", 0))
        steps += _total(block["steps"])
        if "wls_calls" in block and wls is not None:
            wls += _total(block["wls_calls"])
        else:
            wls = None
        comparisons += int(_total(block.get(COMPARISONS) or {}))
    return {
        "per_family": per_family,
        "episodes": n,
        "successes": successes,
        "invalid_action_episodes": invalid,
        "false_commit_episodes": false_commit,
        "loop_episodes": loops,
        "mean_actions": steps / n if n else None,
        "mean_wls_calls": (wls / n) if (n and wls is not None) else None,
        "comparisons": comparisons,
    }


def table(r1: Mapping[str, Any], r2: Mapping[str, Any], families: Sequence[str]) -> dict[str, Any]:
    policies = {
        "expert": r1["per_family"]["expert"]["outcomes"],
        "bc0": r1["per_family"]["bc0"]["outcomes"],
        "r1": r1["per_family"]["r1"]["outcomes"],
        "r2": r2["per_family"]["r2"]["outcomes"],
    }
    metrics = {name: policy_metrics(outcomes, families) for name, outcomes in policies.items()}
    rescored = policy_metrics(r2["per_family"]["r1"]["outcomes"], families)
    return {
        "families": list(families),
        "policies": metrics,
        "r1_rescored_in_round_2": rescored,
        "r1_consistent_across_rounds": rescored["per_family"] == metrics["r1"]["per_family"],
    }


def markdown(result: Mapping[str, Any]) -> str:
    names = ("expert", "bc0", "r1", "r2")
    header = "| Cohort | N | Expert | BC0 | R1 | R2 |"
    lines = [header, "| --- | --- | --- | --- | --- | --- |"]
    policies = result["policies"]
    for family in result["families"]:
        cells = [policies[name]["per_family"].get(family) for name in names]
        if not any(cells):
            continue
        n = next(cell["episodes"] for cell in cells if cell)
        lines.append(f"| {LABELS.get(family, family)} | {n} | "
                     + " | ".join(str(cell["successes"]) if cell else "-" for cell in cells) + " |")
    total = policies["expert"]["episodes"]
    lines.append(f"| All roots | {total} | " + " | ".join(str(policies[name]["successes"]) for name in names) + " |")

    def row(label: str, key: str, fmt: str = "{}") -> str:
        values = [policies[name][key] for name in names]
        return f"| {label} | | " + " | ".join("n/a" if v is None else fmt.format(v) for v in values) + " |"

    lines += [
        row("Episodes with an invalid action", "invalid_action_episodes"),
        row("Episodes with a false commit", "false_commit_episodes"),
        row("Loop episodes", "loop_episodes"),
        row("Mean actions per episode", "mean_actions", "{:.2f}"),
        row("Mean WLS calls per episode", "mean_wls_calls", "{:.2f}"),
        row("Bounded comparisons", "comparisons"),
    ]
    lines.append("")
    lines.append(f"R1 rescored in round 2 identical per family: {result['r1_consistent_across_rounds']}.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--families", nargs="*", default=list(CONVENTIONAL))
    parser.add_argument("--output", type=Path, default=None, help="also write the markdown table and a JSON beside it")
    args = parser.parse_args(argv)
    r1 = json.loads(args.r1.read_text(encoding="utf-8"))
    r2 = json.loads(args.r2.read_text(encoding="utf-8"))
    result = table(r1, r2, args.families)
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
