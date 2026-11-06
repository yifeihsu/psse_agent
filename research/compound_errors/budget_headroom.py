"""Success under an action budget, the roots a budget loses, and what a different decision recovers (2026-10-29).

    python -m research.compound_errors.budget_headroom \\
        --run ieee118:40=output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl \\
        --run ieee118:12=output/compound_errors_ieee118_20261028/budget_12/episodes.jsonl ... \\
        --audit ieee118:16=output/compound_errors_ieee118_20261028/audit_budget16/roots.jsonl \\
        --output-dir output/compound_errors_ieee118_20261028/budget_headroom

Per system, class and budget: the expert's successes, the episodes that ran
to the budget, the handoffs the teacher announced because the budget could not
close another correction (`recovery_budget_exhausted`), and the roots the
budget lost (solved at the largest budget, failed at this one).  With a
decision audit of the lost roots under the same budget
(`audit.py --max-steps b --reference-episodes <40-action run>`), each lost
root is binned: recovered by a legal alternative at one of the audited
decision states (route-choice headroom), recovered only by the truth probe,
or by neither; the lost roots' paths are summarized by what the largest budget
spent (actions, rolled-back corrections) against what the truth probe needed.

Per system, from the largest budget's run: the WLS cue the route choice after
the first committed correction can read (the max normalized residual of the
verifying `run_wls`, alarm still on), grouped by the committed correction's
tool and the next true correction the episode committed.
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

CONTRACT = "compound_budget_headroom_v1"
CLASSES = ("rx2_adjacent", "rx2_distant", "meter_rx2", "meter_on_rx", "control_rx", "control_meter")
BUDGET_HANDOFF = "operator_escalation:recovery_budget_exhausted"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_keyed(items: Sequence[str]) -> dict[str, dict[int, Path]]:
    """``system:budget=path`` items -> {system: {budget: path}}."""
    out: dict[str, dict[int, Path]] = defaultdict(dict)
    for item in items:
        key, _, path = item.partition("=")
        system, _, budget = key.partition(":")
        out[system][int(budget)] = Path(path)
    return out


def alternative_solves(root: Mapping[str, Any]) -> list[str]:
    """The tools of the menu alternatives that solved the root (any audited decision state)."""
    tools = []
    for state in root.get("decision_states") or []:
        for branch in state.get("alternatives") or []:
            if (branch.get("outcome") or {}).get("success"):
                tools.append(str((branch.get("action") or {}).get("tool")))
    return tools


def truth_outcome(root: Mapping[str, Any]) -> Mapping[str, Any]:
    """The outcome of the root's truth probe (one per root), or {}."""
    for state in root.get("decision_states") or []:
        truth = state.get("truth")
        if isinstance(truth, Mapping):
            return truth.get("outcome") or {}
    return {}


def truth_solves(root: Mapping[str, Any]) -> bool:
    return bool(truth_outcome(root).get("success"))


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"n": 0}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"n": len(values), "min": values[0], "p10": pick(0.1), "median": pick(0.5), "p90": pick(0.9),
            "share_at_least_10": sum(v >= 10 for v in values) / len(values)}


def post_commit_cue(episode: Mapping[str, Any]) -> dict[str, Any] | None:
    """The WLS reading after the first committed correction, when the alarm is still on.

    The metrics of the last ``run_wls`` before the first ``commit_state`` (the
    verification of the committed correction), the committed correction's
    tool and the tool of the next true correction the episode committed.
    """
    trace = list(episode.get("trace") or [])
    metrics = None
    first_commit = None
    for index, step in enumerate(trace):
        if step.get("tool") == "run_wls" and step.get("metrics"):
            metrics = step["metrics"]
        if step.get("tool") == "commit_state":
            first_commit = index
            break
    if first_commit is None or not metrics or metrics.get("globally_resolved") or float(metrics.get("chi_square_ratio") or 0.0) < 1.0:
        return None
    committed = next((str(s.get("tool")) for s in reversed(trace[:first_commit]) if str(s.get("tool")).startswith("correct_")), None)
    following = next((str(c.get("tool")) for c in episode.get("corrections") or []
                      if int(c.get("step", -1)) > first_commit and c.get("target_is_true") and c.get("lifecycle") == "commit_state"), None)
    return {"committed": committed, "next_true": following, "max_normalized_residual": metrics.get("max_normalized_residual")}


def cue_table(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        cue = post_commit_cue(episode)
        if cue and cue["max_normalized_residual"] is not None:
            groups[f"{cue['committed']} -> {cue['next_true'] or 'none'}"].append(float(cue["max_normalized_residual"]))
    return {key: _quantiles(values) for key, values in sorted(groups.items(), key=lambda kv: -len(kv[1]))}


def lost_paths(lost: Sequence[str], reference: Mapping[str, Mapping[str, Any]], audited: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """What the largest budget spent on the lost roots, against what the truth probe needed."""
    steps = [int(reference[i].get("steps") or 0) for i in lost]
    rollbacks = [sum(1 for t in reference[i].get("tools") or [] if t == "rollback_state") for i in lost]
    truth = [truth_outcome(audited[i]) for i in lost if i in audited]
    truth_steps = [int(t.get("steps") or 0) for t in truth if t.get("success")]
    mean = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731
    return {"roots": len(lost), "mean_reference_actions": mean(steps), "mean_reference_rollbacks": mean(rollbacks),
            "truth_probe_solves": len(truth_steps), "mean_truth_probe_actions": mean(truth_steps)}


def summarize(runs: Mapping[str, Mapping[int, Path]], audits: Mapping[str, Mapping[int, Path]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system, by_budget in runs.items():
        episodes = {budget: {str(r["scenario_id"]): r for r in load_jsonl(path)} for budget, path in by_budget.items()}
        top = max(episodes)
        audited = {budget: {str(r["root"]): r for r in load_jsonl(path)} for budget, path in (audits.get(system) or {}).items()}
        system_out: dict[str, Any] = {"budgets": sorted(episodes), "reference_budget": top, "classes": {}}
        for name in (*CLASSES, "all"):
            cell: dict[str, Any] = {}
            ids = [i for i, r in episodes[top].items() if name == "all" or r.get("class") == name]
            for budget in sorted(episodes):
                rows = [episodes[budget][i] for i in ids if i in episodes[budget]]
                lost = [i for i in ids if episodes[top][i].get("success") and i in episodes[budget] and not episodes[budget][i].get("success")]
                gained = [i for i in ids if not episodes[top][i].get("success") and i in episodes[budget] and episodes[budget][i].get("success")]
                entry: dict[str, Any] = {
                    "n": len(rows),
                    "solved": sum(1 for r in rows if r.get("success")),
                    "ran_to_budget": sum(1 for r in rows if not r.get("terminal")),
                    "budget_handoffs": sum(1 for r in rows if r.get("escalation_request") == BUDGET_HANDOFF),
                    "mean_actions": (sum(int(r.get("steps") or 0) for r in rows) / len(rows)) if rows else None,
                    "lost_against_reference": len(lost),
                    "gained_against_reference": len(gained),
                }
                if budget in audited:
                    bins: Counter = Counter()
                    tools: Counter = Counter()
                    for i in lost:
                        record = audited[budget].get(i)
                        if record is None:
                            bins["not_audited"] += 1
                            continue
                        solving = alternative_solves(record)
                        if solving:
                            bins["legal_alternative"] += 1
                            tools.update(set(solving))
                        elif truth_solves(record):
                            bins["truth_probe_only"] += 1
                        else:
                            bins["neither"] += 1
                    entry["lost_roots_audited"] = dict(bins)
                    entry["solving_alternative_tools"] = dict(tools)
                    entry["lost_paths"] = lost_paths(lost, episodes[top], audited[budget])
                cell[str(budget)] = entry
            system_out["classes"][name] = cell
        system_out["post_commit_cue"] = cue_table(list(episodes[top].values()))
        out[system] = system_out
    return out


def markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Success under an action budget", ""]
    for system, entry in summary.items():
        budgets = entry["budgets"]
        lines += [f"## {system}", "", "| class | " + " | ".join(f"{b} actions: solved (ran to budget / budget handoffs)" for b in budgets) + " |",
                  "| --- | " + " | ".join("---" for _ in budgets) + " |"]
        for name, cell in entry["classes"].items():
            row = [f"{cell[str(b)]['solved']} of {cell[str(b)]['n']} ({cell[str(b)]['ran_to_budget']} / {cell[str(b)]['budget_handoffs']})" for b in budgets]
            lines.append(f"| {name} | " + " | ".join(row) + " |")
        audited = [b for b in budgets if any("lost_roots_audited" in c[str(b)] for c in entry["classes"].values())]
        top = entry["reference_budget"]
        fmt = lambda v: "" if v is None else f"{v:.1f}"  # noqa: E731
        for budget in audited:
            lines += ["", f"Roots lost at {budget} actions (solved at {top}), audited under {budget}:", "",
                      f"| class | lost | legal alternative recovers | truth probe only | neither | solving alternatives | "
                      f"mean actions at {top} | mean rollbacks at {top} | mean truth-probe actions (solved) |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for name, cell in entry["classes"].items():
                e = cell[str(budget)]
                bins = e.get("lost_roots_audited") or {}
                paths = e.get("lost_paths") or {}
                lines.append(f"| {name} | {e['lost_against_reference']} | {bins.get('legal_alternative', 0)} | {bins.get('truth_probe_only', 0)} | "
                             f"{bins.get('neither', 0)} | {e.get('solving_alternative_tools') or ''} | {fmt(paths.get('mean_reference_actions'))} | "
                             f"{fmt(paths.get('mean_reference_rollbacks'))} | {fmt(paths.get('mean_truth_probe_actions'))} |")
        lines += ["", f"Max normalized residual after the first committed correction, alarm still on ({top} actions), "
                      "by committed correction -> next true correction:", "",
                  "| committed -> next true | n | min | p10 | median | p90 | share >= 10 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for key, q in entry["post_commit_cue"].items():
            lines.append(f"| {key} | {q['n']} | {q['min']:.1f} | {q['p10']:.1f} | {q['median']:.1f} | {q['p90']:.1f} | {q['share_at_least_10']:.0%} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="SYSTEM:BUDGET=EPISODES")
    parser.add_argument("--audit", action="append", default=[], metavar="SYSTEM:BUDGET=AUDIT_ROOTS")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = summarize(parse_keyed(args.run), parse_keyed(args.audit))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {"contract": CONTRACT, "runs": args.run, "audits": args.audit, "summary": summary}
    (out / "budget_headroom.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = markdown(summary)
    (out / "budget_headroom.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
