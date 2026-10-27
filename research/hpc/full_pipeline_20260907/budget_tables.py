#!/usr/bin/env python3
"""Success under a fixed action budget (conventional-only cell, 2026-10-27; note section 7).

    budget_tables.py --cell-dir PIPE [--budgets 12 16 20] [--output OUT.json]

Reads the budget-limited evaluations ``out/extra/evaluations/test_budget<b>``
(BC0, R1 and the teacher on the 192 test roots with the evaluation horizon
set to <b> actions; submit_extras_20261027.sh) and, when present,
``test_budget<b>_r2`` (R2 alone, its record in the evaluator's first adapter
slot), beside the round-1 and round-2 evaluations at the cell's 40-action
horizon, and joins them by physical root.  For each budget and policy:
successes, safety-clean successes (solved with no invalid action, false
commit or loop), episodes with an invalid action, a false commit, a loop or
any of the three, episodes that ran to the budget, mean actions and WLS
calls, per family and per error cohort (the 80 multi-error roots: multiple
meters, meter + R/X, meter + breaker; the 112 single-error and clean roots)
as well; the exact two-sided McNemar tests between the policies at each
budget on success, on safety-clean success and on process violations; and,
against the measured counts, the approximation read off the 40-action
episodes (a root counts as solved within b actions when its 40-action
episode succeeded in at most b actions), with the roots on which the two
disagree.  ``policies.json`` of every budget-limited evaluation must record
that budget, and its adapters must be the cell's BC0, R1 and R2 (tree hashes
of the stage receipts).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
POLICIES = ("expert", "bc0", "r1", "r2")
PAIRS = (("bc0", "r1"), ("bc0", "expert"), ("r1", "expert"), ("bc0", "r2"), ("r1", "r2"), ("r2", "expert"))
DEFAULT_BUDGETS = (12, 16, 20)
MULTI_ERROR_FAMILIES = ("multi_measurement", "measurement+parameter", "measurement+topology")
COHORTS = {"multi_error": "multi-error roots", "single_or_clean": "single-error and clean roots"}


def _tables():
    spec = importlib.util.spec_from_file_location("followup_tables", HERE / "followup_tables.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cohort(family: str | None) -> str:
    return "multi_error" if family in MULTI_ERROR_FAMILIES else "single_or_clean"


def check_identity(evaluation_dir: Path, budget: int, receipts: Mapping[str, str | None],
                   expected_labels: Sequence[str] = ()) -> dict[str, Any]:
    """The evaluation's recorded horizon and adapters against the budget and the cell's receipts."""
    identity = _read(evaluation_dir / "policies.json")
    if int(identity.get("max_steps") or 0) != int(budget):
        raise ValueError(f"{evaluation_dir.name} recorded max_steps {identity.get('max_steps')}, not {budget}")
    policies = identity.get("policies") or {}
    labels = {slot: (entry or {}).get("label") for slot, entry in policies.items()}
    hashes = {(entry or {}).get("label"): (entry or {}).get("tree_sha256") for entry in policies.values()}
    for label in expected_labels:
        if label not in hashes:
            raise ValueError(f"{evaluation_dir.name} does not carry an adapter labelled {label}")
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
    r2 = out / "r2" / "training.done"
    return {
        "bc0": _read(bc0).get("adapter_tree_sha256") if bc0.is_file() else None,
        "r1": _read(r1).get("candidate_tree_sha256") if r1.is_file() else None,
        "r2": _read(r2).get("candidate_tree_sha256") if r2.is_file() else None,
    }


def load_outcomes(cell: Path, budgets: Sequence[int]) -> tuple[dict[int, dict[str, dict[str, dict[str, Any]]]], dict[str, Any]]:
    """budget -> policy -> root -> outcome; the round evaluations under their recorded (40-action) horizon."""
    tables = _tables()
    out = cell / "out"
    collection = out / "r1" / "collection"
    roots = tables.describe_roots(_read(collection / "development_scenarios.json"))
    comparison = _read(collection / "evaluation" / "comparison.json")
    full = int(comparison.get("max_steps") or 40)
    outcomes: dict[int, dict[str, dict[str, dict[str, Any]]]] = {full: {}}
    for policy in ("expert", "bc0", "r1"):
        path = collection / "evaluation" / f"{policy}_eval.json"
        if path.is_file():
            outcomes[full][policy] = tables.records(path, roots)
    identities: dict[str, Any] = {str(full): {"max_steps": full, "source": str(collection / "evaluation")}}
    round2 = out / "r2" / "collection" / "evaluation"
    if (round2 / "r1_eval.json").is_file():
        # R2 is the candidate slot of the round-2 evaluation, on the same roots and horizon.
        horizon = int(_read(round2 / "comparison.json").get("max_steps") or full) if (round2 / "comparison.json").is_file() else full
        if horizon != full:
            raise ValueError(f"the round-2 evaluation ran at {horizon} actions, the round-1 one at {full}")
        outcomes[full]["r2"] = tables.records(round2 / "r1_eval.json", roots)
        identities[str(full)]["r2_source"] = str(round2)
    receipts = cell_receipts(cell)
    for budget in budgets:
        evaluation_dir = out / "extra" / "evaluations" / f"test_budget{budget}"
        if not (evaluation_dir / "summary.json").is_file():
            continue
        identities[str(budget)] = check_identity(evaluation_dir, budget, receipts, ("bc0", "r1"))
        outcomes[int(budget)] = {}
        for policy in ("expert", "bc0", "r1"):
            path = evaluation_dir / f"{policy}_eval.json"
            if path.is_file():
                outcomes[int(budget)][policy] = tables.records(path, roots)
        r2_dir = out / "extra" / "evaluations" / f"test_budget{budget}_r2"
        if (r2_dir / "summary.json").is_file() and (r2_dir / "bc0_eval.json").is_file():
            identities[f"{budget}_r2"] = check_identity(r2_dir, budget, receipts, ("r2",))
            outcomes[int(budget)]["r2"] = tables.records(r2_dir / "bc0_eval.json", roots)
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


def cohort_summary(items: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Successes, safety-clean successes and violation episodes per error cohort."""
    result = {name: {"episodes": 0, "successes": 0, "clean_successes": 0, "violation_episodes": 0} for name in COHORTS}
    for item in items.values():
        cell = result[cohort(item.get("family"))]
        cell["episodes"] += 1
        cell["successes"] += int(item["success"])
        cell["clean_successes"] += int(item["clean"])
        cell["violation_episodes"] += int(item["violation"])
    return result


def build(cell: Path, budgets: Sequence[int] = DEFAULT_BUDGETS) -> dict[str, Any]:
    tables = _tables()
    outcomes, meta = load_outcomes(cell, budgets)
    full = meta["full_budget"]
    result: dict[str, Any] = {"contract": "conventional_only_budget_tables_v2", **meta,
                              "multi_error_families": list(MULTI_ERROR_FAMILIES), "budgets": {}}
    for budget in sorted(outcomes):
        block: dict[str, Any] = {"policies": {}, "paired": {}, "approximation_from_full_budget": {}}
        for policy, items in outcomes[budget].items():
            summary = tables.summarize_policy(items)
            summary["budget_exhausted_episodes"] = sum(1 for item in items.values() if item["steps"] >= budget)
            summary["non_terminal_episodes"] = sum(1 for item in items.values() if not item["terminal"])
            summary["by_cohort"] = cohort_summary(items)
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
    header = ["| Budget | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]

    def cells(b: str, fmt) -> str:
        parts = []
        for p in names:
            s = result["budgets"][b]["policies"].get(p)
            parts.append(fmt(s) if s else "-")
        return f"| {b} | " + " | ".join(parts) + " |"

    lines = [f"### Success under the action budget ({result['roots']} test roots; successes, safety-clean successes in parentheses)"] + header
    lines += [cells(b, lambda s: f"{s['successes']} ({s['clean_successes']})") for b in budgets]
    lines += ["", "### Process-violation episodes (invalid action / false commit / loop); episodes at the budget"] + header
    lines += [cells(b, lambda s: f"{s['violation_episodes']} ({s['invalid_action_episodes']}/{s['false_commit_episodes']}/{s['loop_episodes']}); {s['budget_exhausted_episodes']}") for b in budgets]
    lines += ["", "### Mean actions; mean WLS calls"] + header
    lines += [cells(b, lambda s: f"{s['mean_actions']:.2f}; {s['mean_wls_calls']:.2f}") for b in budgets]
    for name, label in COHORTS.items():
        lines += ["", f"### {label.capitalize()}: successes/episodes (safety-clean); violation episodes"] + header
        lines += [cells(b, lambda s, name=name: f"{s['by_cohort'][name]['successes']}/{s['by_cohort'][name]['episodes']} "
                                               f"({s['by_cohort'][name]['clean_successes']}); {s['by_cohort'][name]['violation_episodes']}")
                  for b in budgets]
    for b in budgets:
        block = result["budgets"][b]
        families = sorted({f for p in names if p in block["policies"] for f in block["policies"][p]["by_family"]})
        lines += ["", f"### Budget {b}, per family (successes/episodes, safety-clean in parentheses)",
                  "| Family | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
        for family in families:
            parts = []
            for p in names:
                cell = (block["policies"].get(p) or {}).get("by_family", {}).get(family)
                parts.append(f"{cell['successes']}/{cell['episodes']} ({cell['clean_successes']})" if cell else "-")
            lines.append(f"| {family} | " + " | ".join(parts) + " |")
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
