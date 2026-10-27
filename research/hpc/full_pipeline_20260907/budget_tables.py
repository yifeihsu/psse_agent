#!/usr/bin/env python3
"""Success under a fixed action budget (conventional-only cell, 2026-10-27; note section 7).

    budget_tables.py --cell-dir PIPE [--budgets 12 16 20] [--output OUT.json]

Reads the budget-limited evaluations ``out/extra/evaluations/test_budget<b>``
(BC0, R1 and the teacher on the 192 test roots with the evaluation horizon
set to <b> actions; submit_extras_20261027.sh) beside the round-1 evaluation
at the cell's 40-action horizon, and joins them by physical root.  For each
budget and policy: successes, safety-clean successes (solved with no invalid
action, false commit or loop), episodes with an invalid action, a false
commit, a loop or any of the three, episodes that ran to the budget, mean
actions and WLS calls, per family as well; the exact two-sided McNemar tests
between the policies at each budget on success, on safety-clean success and
on process violations; and, against the measured counts, the approximation
read off the 40-action episodes (a root counts as solved within b actions
when its 40-action episode succeeded in at most b actions), with the roots
on which the two disagree.  ``policies.json`` of every budget-limited
evaluation must record that budget, and its adapters must be the cell's BC0
and R1 (tree hashes of the stage receipts).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
POLICIES = ("expert", "bc0", "r1")
PAIRS = (("bc0", "r1"), ("bc0", "expert"), ("r1", "expert"))
DEFAULT_BUDGETS = (12, 16, 20)


def _tables():
    spec = importlib.util.spec_from_file_location("followup_tables", HERE / "followup_tables.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_identity(evaluation_dir: Path, budget: int, receipts: Mapping[str, str | None]) -> dict[str, Any]:
    """The evaluation's recorded horizon and adapters against the budget and the cell's receipts."""
    identity = _read(evaluation_dir / "policies.json")
    if int(identity.get("max_steps") or 0) != int(budget):
        raise ValueError(f"{evaluation_dir.name} recorded max_steps {identity.get('max_steps')}, not {budget}")
    policies = identity.get("policies") or {}
    labels = {slot: (entry or {}).get("label") for slot, entry in policies.items()}
    hashes = {(entry or {}).get("label"): (entry or {}).get("tree_sha256") for entry in policies.values()}
    for label, expected in receipts.items():
        if expected and hashes.get(label) and hashes[label] != expected:
            raise ValueError(f"{evaluation_dir.name}: adapter {label} tree {hashes[label][:12]} is not the cell's {expected[:12]}")
    return {"max_steps": int(identity["max_steps"]), "seed": identity.get("seed"), "slots": labels,
            "adapter_trees": hashes, "expert": identity.get("expert"),
            "adapters_match_receipts": all(hashes.get(l) == e for l, e in receipts.items() if e and hashes.get(l))}


def cell_receipts(cell: Path) -> dict[str, str | None]:
    out = cell / "out"
    bc0 = out / "bc0.done"
    r1 = out / "r1" / "training.done"
    return {
        "bc0": _read(bc0).get("adapter_tree_sha256") if bc0.is_file() else None,
        "r1": _read(r1).get("candidate_tree_sha256") if r1.is_file() else None,
    }


def load_outcomes(cell: Path, budgets: Sequence[int]) -> tuple[dict[int, dict[str, dict[str, dict[str, Any]]]], dict[str, Any]]:
    """budget -> policy -> root -> outcome; the round-1 evaluation under its recorded (40-action) horizon."""
    tables = _tables()
    out = cell / "out"
    collection = out / "r1" / "collection"
    roots = tables.describe_roots(_read(collection / "development_scenarios.json"))
    comparison = _read(collection / "evaluation" / "comparison.json")
    full = int(comparison.get("max_steps") or 40)
    outcomes: dict[int, dict[str, dict[str, dict[str, Any]]]] = {full: {}}
    for policy in POLICIES:
        path = collection / "evaluation" / f"{policy}_eval.json"
        if path.is_file():
            outcomes[full][policy] = tables.records(path, roots)
    identities: dict[str, Any] = {str(full): {"max_steps": full, "source": str(collection / "evaluation")}}
    receipts = cell_receipts(cell)
    for budget in budgets:
        evaluation_dir = out / "extra" / "evaluations" / f"test_budget{budget}"
        if not (evaluation_dir / "summary.json").is_file():
            continue
        identities[str(budget)] = check_identity(evaluation_dir, budget, receipts)
        outcomes[int(budget)] = {}
        for policy in POLICIES:
            path = evaluation_dir / f"{policy}_eval.json"
            if path.is_file():
                outcomes[int(budget)][policy] = tables.records(path, roots)
    return outcomes, {"full_budget": full, "evaluations": identities, "roots": len(roots)}


def approximate(full: Mapping[str, Mapping[str, Any]], budget: int) -> dict[str, dict[str, bool]]:
    """The reading off the 40-action episodes: solved within ``budget`` actions."""
    return {root: {"success": bool(item["success"] and item["steps"] <= budget),
                   "clean": bool(item["clean"] and item["steps"] <= budget)} for root, item in full.items()}


def compare_with_approximation(measured: Mapping[str, Mapping[str, Any]], approx: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    shared = sorted(set(measured) & set(approx))
    result = {}
    for field in ("success", "clean"):
        only_measured = [r for r in shared if measured[r][field] and not approx[r][field]]
        only_approx = [r for r in shared if approx[r][field] and not measured[r][field]]
        result[field] = {
            "approximated": sum(1 for r in shared if approx[r][field]),
            "measured": sum(1 for r in shared if measured[r][field]),
            "only_measured": len(only_measured),
            "only_approximated": len(only_approx),
            "only_measured_roots": only_measured,
            "only_approximated_roots": only_approx,
            "paired_roots": len(shared),
        }
    return result


def build(cell: Path, budgets: Sequence[int] = DEFAULT_BUDGETS) -> dict[str, Any]:
    tables = _tables()
    outcomes, meta = load_outcomes(cell, budgets)
    full = meta["full_budget"]
    result: dict[str, Any] = {"contract": "conventional_only_budget_tables_v1", **meta, "budgets": {}}
    for budget in sorted(outcomes):
        block: dict[str, Any] = {"policies": {}, "paired": {}, "approximation_from_full_budget": {}}
        for policy, items in outcomes[budget].items():
            summary = tables.summarize_policy(items)
            summary["budget_exhausted_episodes"] = sum(1 for item in items.values() if item["steps"] >= budget)
            summary["non_terminal_episodes"] = sum(1 for item in items.values() if not item["terminal"])
            block["policies"][policy] = summary
        for first, second in PAIRS:
            if first in outcomes[budget] and second in outcomes[budget]:
                a, b = outcomes[budget][first], outcomes[budget][second]
                block["paired"][f"{first}_vs_{second}"] = {
                    "success_failures": tables.paired(tables._failures(a, "success"), tables._failures(b, "success"), "f"),
                    "clean_success_failures": tables.paired(tables._failures(a, "clean"), tables._failures(b, "clean"), "f"),
                    "violations": tables.paired(a, b, "violation"),
                }
        if budget != full:
            for policy, items in outcomes[budget].items():
                if policy in outcomes[full]:
                    block["approximation_from_full_budget"][policy] = compare_with_approximation(
                        items, approximate(outcomes[full][policy], budget))
        result["budgets"][str(budget)] = block
    return result


def markdown(result: Mapping[str, Any]) -> str:
    budgets = sorted(result["budgets"], key=int)
    names = [p for p in POLICIES if any(p in result["budgets"][b]["policies"] for b in budgets)]
    lines = [f"### Success under the action budget ({result['roots']} test roots; successes, safety-clean successes in parentheses)",
             "| Budget | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for b in budgets:
        cells = []
        for p in names:
            s = result["budgets"][b]["policies"].get(p)
            cells.append(f"{s['successes']} ({s['clean_successes']})" if s else "-")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines += ["", "### Process-violation episodes (invalid action / false commit / loop); episodes at the budget",
              "| Budget | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for b in budgets:
        cells = []
        for p in names:
            s = result["budgets"][b]["policies"].get(p)
            cells.append(f"{s['violation_episodes']} ({s['invalid_action_episodes']}/{s['false_commit_episodes']}/{s['loop_episodes']}); {s['budget_exhausted_episodes']}" if s else "-")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines += ["", "### Mean actions; mean WLS calls", "| Budget | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for b in budgets:
        cells = []
        for p in names:
            s = result["budgets"][b]["policies"].get(p)
            cells.append(f"{s['mean_actions']:.2f}; {s['mean_wls_calls']:.2f}" if s else "-")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    for b in budgets:
        block = result["budgets"][b]
        families = sorted({f for p in names if p in block["policies"] for f in block["policies"][p]["by_family"]})
        lines += ["", f"### Budget {b}, per family (successes/episodes, safety-clean in parentheses)",
                  "| Family | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
        for family in families:
            cells = []
            for p in names:
                cell = (block["policies"].get(p) or {}).get("by_family", {}).get(family)
                cells.append(f"{cell['successes']}/{cell['episodes']} ({cell['clean_successes']})" if cell else "-")
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        if block["approximation_from_full_budget"]:
            lines += ["", f"Budget {b} against the approximation from the {result['full_budget']}-action episodes (solved within {b} actions):"]
            for p in names:
                cmp = block["approximation_from_full_budget"].get(p)
                if cmp:
                    s, c = cmp["success"], cmp["clean"]
                    lines.append(f"- {p}: approximated {s['approximated']}, measured {s['measured']} "
                                 f"(only measured {s['only_measured']}, only approximated {s['only_approximated']}); "
                                 f"safety-clean approximated {c['approximated']}, measured {c['measured']} "
                                 f"(only measured {c['only_measured']}, only approximated {c['only_approximated']})")
        if block["paired"]:
            lines.append("")
            for name, tests in block["paired"].items():
                first, second = name.split("_vs_")
                s, c, v = tests["success_failures"], tests["clean_success_failures"], tests["violations"]
                lines.append(f"- budget {b}, {first} vs {second}: failures only {first} {s['only_first']} / only {second} "
                             f"{s['only_second']} (p = {s['p']:.3g}); clean-success failures {c['only_first']} / {c['only_second']} "
                             f"(p = {c['p']:.3g}); violations {v['only_first']} / {v['only_second']} (p = {v['p']:.3g})")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.cell_dir, args.budgets)
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
