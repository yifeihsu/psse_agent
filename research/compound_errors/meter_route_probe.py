"""The meter-route rule probe against ``baseline_compare``, root by root (2026-10-29).

    python -m research.compound_errors.meter_route_probe --threshold 11 \\
        --pair ieee118:40=output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl,output/compound_errors_ieee118_20261028/meter_route11_40/episodes.jsonl \\
        --pair ieee118:16=output/compound_errors_ieee118_20261028/budget_16/episodes.jsonl,output/compound_errors_ieee118_20261028/meter_route11_16/episodes.jsonl ... \\
        --output-dir output/compound_errors_ieee118_20261028/meter_route_probe

The rule (``ExpertPolicyOracle(meter_route_residual=...)``, run as
``rollout.py --expert baseline_compare --expert-option meter_route_residual=11``):
after a committed meter correction, while the solve still reports a maximum
normalized residual of at least the threshold, the meter route leads and the
measurement expert's post-commit branch guard stands down.

Per system, budget and class: the roots each teacher solves, the roots the
rule gains and loses against ``baseline_compare`` on the same budget, the mean
actions on the roots both solve, and the corrections each committed on a
healthy target (a committed wrong meter is the masking the lifted guard
risks).  Per run: every committed meter correction with the alarm still on,
whether the rule fires on its verifying solve, and the next true correction
the episode committed after it (meter, line or none), which is how often the
cue was right in the states the rule acted on.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.compound_errors.budget_headroom import BUDGET_HANDOFF, CLASSES, load_jsonl  # noqa: E402

CONTRACT = "compound_meter_route_probe_v1"
METER = "correct_measurements"
LINE_TOOLS = ("correct_parameters", "correct_topology")


def parse_pairs(items: Sequence[str]) -> dict[str, dict[int, tuple[Path, Path]]]:
    """``system:budget=baseline,rule`` items -> {system: {budget: (baseline, rule)}}."""
    out: dict[str, dict[int, tuple[Path, Path]]] = defaultdict(dict)
    for item in items:
        key, _, paths = item.partition("=")
        system, _, budget = key.partition(":")
        baseline, _, rule = paths.partition(",")
        out[system][int(budget)] = (Path(baseline), Path(rule))
    return out


def meter_commits(episode: Mapping[str, Any], threshold: float) -> list[dict[str, Any]]:
    """Every committed meter correction whose verifying solve left the alarm on.

    The reading is the last ``run_wls`` before the ``commit_state``; ``fires``
    is the rule's test on it, ``next_true`` the tool of the first true
    correction the episode committed afterwards (None if none), ``true``
    whether the committed meter correction itself was on a true target.
    """
    trace = list(episode.get("trace") or [])
    committed = [c for c in episode.get("corrections") or [] if c.get("lifecycle") == "commit_state"]
    out = []
    metrics: Mapping[str, Any] | None = None
    last_correction = None
    last_correction_step = None
    for index, step in enumerate(trace):
        tool = str(step.get("tool"))
        if tool == "run_wls" and step.get("metrics"):
            metrics = step["metrics"]
        elif tool.startswith("correct_"):
            last_correction, last_correction_step = tool, index
        elif tool == "commit_state" and step.get("status") == "success":
            if last_correction == METER and metrics and not metrics.get("globally_resolved"):
                residual = metrics.get("max_normalized_residual")
                following = next((str(c.get("tool")) for c in committed if int(c.get("step", -1)) > index and c.get("target_is_true")), None)
                own = next((c for c in committed if int(c.get("step", -1)) == last_correction_step), None)
                out.append({"step": index, "max_normalized_residual": residual,
                            "fires": residual is not None and float(residual) >= threshold, "next_true": following,
                            "true": None if own is None else bool(own.get("target_is_true"))})
    return out


def wrong_commits(episode: Mapping[str, Any]) -> Counter:
    """Committed corrections on a healthy target, by tool."""
    return Counter(str(c.get("tool")) for c in episode.get("corrections") or []
                   if c.get("lifecycle") == "commit_state" and c.get("target_is_true") is False)


def cue_outcomes(episodes: Sequence[Mapping[str, Any]], threshold: float, *, committed_true: bool | None = None) -> dict[str, Any]:
    """Next true correction after each alarm-on meter commit, split by whether the rule fires.

    ``committed_true`` keeps only the commits of a true meter (True) or of a healthy one (False).
    """
    table: dict[str, Counter] = {"fires": Counter(), "quiet": Counter()}
    for episode in episodes:
        for commit in meter_commits(episode, threshold):
            if committed_true is not None and commit["true"] is not committed_true:
                continue
            following = commit["next_true"]
            kind = "meter" if following == METER else "line" if following in LINE_TOOLS else "none"
            table["fires" if commit["fires"] else "quiet"][kind] += 1
    return {key: dict(counter) for key, counter in table.items()}


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _root_note(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {"steps": episode.get("steps"), "terminal_outcome": episode.get("terminal_outcome"),
            "escalation_request": episode.get("escalation_request"), "wrong_commits": dict(wrong_commits(episode)),
            "true_commits": sum(1 for c in episode.get("corrections") or [] if c.get("lifecycle") == "commit_state" and c.get("target_is_true"))}


def compare(baseline: Mapping[str, Mapping[str, Any]], rule: Mapping[str, Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for name in (*CLASSES, "all"):
        ids = sorted(i for i, r in baseline.items() if i in rule and (name == "all" or r.get("class") == name))
        gained = [i for i in ids if rule[i].get("success") and not baseline[i].get("success")]
        lost = [i for i in ids if baseline[i].get("success") and not rule[i].get("success")]
        both = [i for i in ids if baseline[i].get("success") and rule[i].get("success")]
        base_wrong: Counter = Counter()
        rule_wrong: Counter = Counter()
        for i in ids:
            base_wrong.update(wrong_commits(baseline[i]))
            rule_wrong.update(wrong_commits(rule[i]))
        cells[name] = {
            "n": len(ids),
            "baseline_solved": sum(1 for i in ids if baseline[i].get("success")),
            "rule_solved": sum(1 for i in ids if rule[i].get("success")),
            "gained": len(gained),
            "lost": len(lost),
            "mean_actions_both_solved": {"baseline": _mean([int(baseline[i].get("steps") or 0) for i in both]),
                                         "rule": _mean([int(rule[i].get("steps") or 0) for i in both]), "roots": len(both)},
            "budget_handoffs": {"baseline": sum(1 for i in ids if baseline[i].get("escalation_request") == BUDGET_HANDOFF),
                                "rule": sum(1 for i in ids if rule[i].get("escalation_request") == BUDGET_HANDOFF)},
            "wrong_commits": {"baseline": dict(base_wrong), "rule": dict(rule_wrong)},
            "episodes_with_wrong_meter_commit": {"baseline": sum(1 for i in ids if wrong_commits(baseline[i]).get(METER)),
                                                 "rule": sum(1 for i in ids if wrong_commits(rule[i]).get(METER))},
            "rule_fired_episodes": sum(1 for i in ids if any(c["fires"] for c in meter_commits(rule[i], threshold))),
            "gained_roots": {i: {"baseline": _root_note(baseline[i]), "rule": _root_note(rule[i])} for i in gained},
            "lost_roots": {i: {"baseline": _root_note(baseline[i]), "rule": _root_note(rule[i])} for i in lost},
        }
    return cells


def summarize(pairs: Mapping[str, Mapping[int, tuple[Path, Path]]], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system, by_budget in pairs.items():
        system_out: dict[str, Any] = {}
        for budget, (base_path, rule_path) in sorted(by_budget.items()):
            baseline = {str(r["scenario_id"]): r for r in load_jsonl(base_path)}
            rule = {str(r["scenario_id"]): r for r in load_jsonl(rule_path)}
            runs = {"baseline": list(baseline.values()), "rule": list(rule.values())}
            system_out[str(budget)] = {
                "classes": compare(baseline, rule, threshold),
                "cue": {run: cue_outcomes(episodes, threshold) for run, episodes in runs.items()},
                "cue_by_commit": {run: {"true": cue_outcomes(episodes, threshold, committed_true=True),
                                        "healthy": cue_outcomes(episodes, threshold, committed_true=False)}
                                  for run, episodes in runs.items()},
            }
        out[system] = system_out
    return out


def _solved(cell: Mapping[str, Any] | None, n: int | None = None) -> str:
    if not cell:
        return ""
    size = "" if n is None or cell["n"] == n else f" of {cell['n']}"
    return f"{cell['baseline_solved']} / {cell['rule_solved']}{size} (+{cell['gained']}, -{cell['lost']})"


def against_reference(summary: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    """Per system: each class's solved counts (baseline_compare / rule, gained, lost), reference probe beside this one."""
    lines = ["## Against the reference probe", ""]
    for system, by_budget in summary.items():
        budgets = [b for b in by_budget if b in (reference.get(system) or {})]
        if not budgets:
            continue
        header = " | ".join(f"{b} actions, reference | {b} actions, this draw" for b in budgets)
        lines += [f"### {system}", "", f"| class | {header} |", "| --- |" + " --- |" * (2 * len(budgets))]
        for name in (*CLASSES, "all"):
            here = {b: by_budget[b]["classes"].get(name) for b in budgets}
            n = next((cell["n"] for cell in here.values() if cell), None)
            if n is None:
                continue
            cells = [text for b in budgets for text in (_solved(reference[system][b]["classes"].get(name), n), _solved(here[b], n))]
            lines.append(f"| {name} ({n}) | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def markdown(summary: Mapping[str, Any], threshold: float, reference: Mapping[str, Any] | None = None) -> str:
    fmt = lambda v: "" if v is None else f"{v:.1f}"  # noqa: E731
    count = lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "0"  # noqa: E731
    lines = [f"# Meter-route rule (max normalized residual >= {threshold:g}) against baseline_compare", ""]
    for system, by_budget in summary.items():
        for budget, entry in by_budget.items():
            lines += [f"## {system}, {budget} actions", "",
                      "| class | n | baseline_compare | rule | gained | lost | mean actions, both solved (baseline / rule) | "
                      "wrong commits baseline | wrong commits rule | budget handoffs (baseline / rule) | rule fired (episodes) |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for name, cell in entry["classes"].items():
                both = cell["mean_actions_both_solved"]
                lines.append(f"| {name} | {cell['n']} | {cell['baseline_solved']} | {cell['rule_solved']} | {cell['gained']} | {cell['lost']} | "
                             f"{fmt(both['baseline'])} / {fmt(both['rule'])} | {count(cell['wrong_commits']['baseline'])} | "
                             f"{count(cell['wrong_commits']['rule'])} | {cell['budget_handoffs']['baseline']} / {cell['budget_handoffs']['rule']} | "
                             f"{cell['rule_fired_episodes']} |")
            lines += ["", "Next true correction after each committed meter correction with the alarm on:", "",
                      "| run | rule fires: meter / line / none | rule quiet: meter / line / none |", "| --- | --- | --- |"]
            for run, cue in entry["cue"].items():
                split = (entry.get("cue_by_commit") or {}).get(run) or {}
                for label, table in ((run, cue), (f"{run}, after a true meter commit", split.get("true")),
                                     (f"{run}, after a healthy meter commit", split.get("healthy"))):
                    if table is None:
                        continue
                    cells = [f"{table[k].get('meter', 0)} / {table[k].get('line', 0)} / {table[k].get('none', 0)}" for k in ("fires", "quiet")]
                    lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
            lines.append("")
    if reference:
        lines += against_reference(summary, reference)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", action="append", required=True, metavar="SYSTEM:BUDGET=BASELINE,RULE")
    parser.add_argument("--threshold", type=float, default=11.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="an earlier probe's meter_route_probe.json to set beside this one (a fresh draw against the in-sample probe)")
    args = parser.parse_args(argv)
    summary = summarize(parse_pairs(args.pair), args.threshold)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["summary"] if args.reference else None
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {"contract": CONTRACT, "threshold": args.threshold, "pairs": args.pair, "summary": summary,
              "reference": None if args.reference is None else str(args.reference)}
    (out / "meter_route_probe.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = markdown(summary, args.threshold, reference)
    (out / "meter_route_probe.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
