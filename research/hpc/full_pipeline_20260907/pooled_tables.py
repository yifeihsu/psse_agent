#!/usr/bin/env python3
"""Per-draw and pooled tables of the cell's two fresh test draws (2026-10-29; note section 8).

    pooled_tables.py --cell-dir PIPE [--draw2 test_draw2] [--prefix test2] [--budgets 12 20] [--output OUT.json]

Draw 1 is the cell's test set (192 roots): expert, BC0 and R1 from the
round-1 evaluation, R2 from the round-2 evaluation (its candidate slot), and
the budget-limited evaluations ``test_budget<b>`` and ``test_budget<b>_r2`` of
section 7.  Draw 2 is ``out/extra/<draw2>/suite/development.json`` with the
evaluations ``<prefix>_<full>``, ``<prefix>_<full>_r2``, ``<prefix>_budget<b>``
and ``<prefix>_budget<b>_r2`` of ``submit_test_draw2_20261029.sh`` (R2 alone in
the evaluator's first adapter slot).  For each draw, and for the pool of the
two (their roots are disjoint, so the pool is the union, formed only where
both draws carry the policy at that horizon): per family and per error
cohort, successes, safety-clean successes (solved with no invalid action,
false commit or loop), violation episodes, episodes that ran to the budget,
mean actions and WLS calls; the exact two-sided McNemar tests BC0 vs R1, BC0
vs expert and R1 vs R2 (and the other pairs) on success, on safety-clean
success and on violations; and, per family and policy, draw 1 against draw 2
at the full horizon.  The recorded horizon and adapter tree hashes of every
evaluation with a ``policies.json`` are checked against the budget and the
cell's receipts (``budget_tables.check_identity``).  Writes ``OUT.json``,
``OUT.md`` and ``OUT_episodes.jsonl``: one line per draw, horizon, policy and
root, the record ``evaluate_suite.py`` writes to ``episodes.jsonl`` plus
``draw`` and ``budget``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
POLICIES = ("expert", "bc0", "r1", "r2")
PAIRS = (("bc0", "r1"), ("bc0", "expert"), ("r1", "r2"), ("r1", "expert"), ("bc0", "r2"), ("r2", "expert"))
DEFAULT_BUDGETS = (12, 20)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, HERE / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_sources(cell: Path, draw2: str, prefix: str, budgets: Sequence[int]) -> dict[str, dict[str, Any]]:
    """Where each draw's rows and evaluations live, and which evaluations carry an identity to check."""
    out = cell / "out"
    extra = out / "extra" / "evaluations"
    collection = out / "r1" / "collection"
    full = int(_read(collection / "evaluation" / "comparison.json").get("max_steps") or 40)
    first: dict[str, Any] = {"rows": collection / "development_scenarios.json", "full_budget": full,
                             "evaluations": {}, "identity": {}}
    first["evaluations"][full] = {
        "expert": collection / "evaluation" / "expert_eval.json",
        "bc0": collection / "evaluation" / "bc0_eval.json",
        "r1": collection / "evaluation" / "r1_eval.json",
        "r2": out / "r2" / "collection" / "evaluation" / "r1_eval.json",
    }
    for budget in budgets:
        b = int(budget)
        first["evaluations"][b] = {
            "expert": extra / f"test_budget{b}" / "expert_eval.json",
            "bc0": extra / f"test_budget{b}" / "bc0_eval.json",
            "r1": extra / f"test_budget{b}" / "r1_eval.json",
            "r2": extra / f"test_budget{b}_r2" / "bc0_eval.json",
        }
        first["identity"][b] = [(extra / f"test_budget{b}", ("bc0", "r1")), (extra / f"test_budget{b}_r2", ("r2",))]
    second: dict[str, Any] = {"rows": out / "extra" / draw2 / "suite" / "development.json", "full_budget": full,
                              "evaluations": {}, "identity": {}}
    names = {full: (f"{prefix}_{full}", f"{prefix}_{full}_r2")}
    for budget in budgets:
        names[int(budget)] = (f"{prefix}_budget{int(budget)}", f"{prefix}_budget{int(budget)}_r2")
    for horizon, (three, alone) in names.items():
        second["evaluations"][horizon] = {
            "expert": extra / three / "expert_eval.json",
            "bc0": extra / three / "bc0_eval.json",
            "r1": extra / three / "r1_eval.json",
            "r2": extra / alone / "bc0_eval.json",
        }
        second["identity"][horizon] = [(extra / three, ("bc0", "r1")), (extra / alone, ("r2",))]
    return {"draw1": first, "draw2": second}


def load_draw(spec: Mapping[str, Any], tables, budget_module, evaluate_module, receipts: Mapping[str, str | None], draw: str):
    """horizon -> policy -> root -> outcome, the identity receipts, and the per-episode records."""
    rows_path: Path = spec["rows"]
    if not rows_path.is_file():
        return {}, {"missing_rows": str(rows_path)}, []
    rows = _read(rows_path)
    roots = tables.describe_roots(rows)
    by_root = {tables._root(row): row for row in rows}
    outcomes: dict[int, dict[str, dict[str, Any]]] = {}
    identities: dict[str, Any] = {"roots": len(roots), "rows": str(rows_path)}
    episodes: list[dict[str, Any]] = []
    for horizon, files in sorted(spec["evaluations"].items()):
        present = {policy: path for policy, path in files.items() if path.is_file()}
        if not present:
            continue
        for directory, labels in spec["identity"].get(horizon, []):
            if (directory / "policies.json").is_file() and (directory / "summary.json").is_file():
                identities[f"{horizon}:{directory.name}"] = budget_module.check_identity(directory, horizon, receipts, labels)
        outcomes[int(horizon)] = {}
        for policy, path in present.items():
            outcomes[int(horizon)][policy] = tables.records(path, roots)
            for episode in tables._episodes(_read(path)):
                root = str(episode.get("physical_root") or "")
                record = evaluate_module._root_record(episode, by_root.get(root), policy)
                episodes.append({"draw": draw, "budget": int(horizon), **record})
    return outcomes, identities, episodes


def pool(draws: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]]) -> dict[int, dict[str, dict[str, Any]]]:
    """The union of the draws' outcomes, for the policies every draw carries at that horizon."""
    horizons: set[int] = set()
    for outcomes in draws.values():
        horizons |= set(outcomes)
    pooled: dict[int, dict[str, dict[str, Any]]] = {}
    for horizon in sorted(horizons):
        blocks = [outcomes[horizon] for outcomes in draws.values() if horizon in outcomes]
        if len(blocks) != len(draws):
            continue
        shared = set.intersection(*(set(block) for block in blocks))
        for policy in shared:
            merged: dict[str, Any] = {}
            for block in blocks:
                overlap = set(merged) & set(block[policy])
                if overlap:
                    raise ValueError(f"root {sorted(overlap)[0]} appears in two draws")
                merged.update(block[policy])
            pooled.setdefault(horizon, {})[policy] = merged
    return pooled


def summarize(outcomes: Mapping[int, Mapping[str, Mapping[str, Any]]], tables, budget_module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in sorted(outcomes):
        policies = outcomes[horizon]
        block: dict[str, Any] = {"policies": {}, "paired": {}}
        for policy, items in policies.items():
            summary = tables.summarize_policy(items)
            summary["budget_exhausted_episodes"] = sum(1 for item in items.values() if item["steps"] >= horizon)
            summary["by_cohort"] = budget_module.cohort_summary(items)
            block["policies"][policy] = summary
        for first, second in PAIRS:
            if first in policies and second in policies:
                a, b = policies[first], policies[second]
                block["paired"][f"{first}_vs_{second}"] = {
                    "success_failures": tables.paired(tables._failures(a, "success"), tables._failures(b, "success"), "f"),
                    "clean_success_failures": tables.paired(tables._failures(a, "clean"), tables._failures(b, "clean"), "f"),
                    "violations": tables.paired(a, b, "violation"),
                }
        result[str(horizon)] = block
    return result


def differences(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Per policy and family, draw 1 against draw 2 at one horizon."""
    result: dict[str, Any] = {}
    for policy in POLICIES:
        a, b = first.get("policies", {}).get(policy), second.get("policies", {}).get(policy)
        if not a or not b:
            continue
        families = sorted(set(a["by_family"]) | set(b["by_family"]))
        result[policy] = {
            "successes": [a["successes"], b["successes"]],
            "clean_successes": [a["clean_successes"], b["clean_successes"]],
            "violation_episodes": [a["violation_episodes"], b["violation_episodes"]],
            "mean_actions": [a["mean_actions"], b["mean_actions"]],
            "by_family": {
                family: {
                    key: [a["by_family"].get(family, {}).get(key), b["by_family"].get(family, {}).get(key)]
                    for key in ("successes", "clean_successes", "episodes")
                }
                for family in families
            },
        }
    return result


def build(cell: Path, draw2: str = "test_draw2", prefix: str = "test2", budgets: Sequence[int] = DEFAULT_BUDGETS) -> dict[str, Any]:
    tables = _load("followup_tables.py")
    budget_module = _load("budget_tables.py")
    evaluate_module = _load("evaluate_suite.py")
    receipts = budget_module.cell_receipts(cell)
    sources = draw_sources(cell, draw2, prefix, budgets)
    outcomes: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    identities: dict[str, Any] = {}
    episodes: list[dict[str, Any]] = []
    for name, spec in sources.items():
        outcomes[name], identities[name], records = load_draw(spec, tables, budget_module, evaluate_module, receipts, name)
        episodes.extend(records)
    full = sources["draw1"]["full_budget"]
    result: dict[str, Any] = {
        "contract": "conventional_only_pooled_test_draws_v1",
        "full_budget": full,
        "draw2": draw2,
        "prefix": prefix,
        "identities": identities,
        "draws": {name: summarize(blocks, tables, budget_module) for name, blocks in outcomes.items()},
    }
    result["draws"]["pooled"] = summarize(pool({name: blocks for name, blocks in outcomes.items() if blocks}), tables, budget_module)
    result["draw_differences_at_full_budget"] = differences(result["draws"].get("draw1", {}).get(str(full), {}),
                                                            result["draws"].get("draw2", {}).get(str(full), {}))
    result["episodes"] = episodes
    return result


def markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    full = str(result["full_budget"])
    for name in ("draw1", "draw2", "pooled"):
        draw = result["draws"].get(name) or {}
        if not draw:
            continue
        horizons = sorted(draw, key=int)
        shown = full if full in draw else horizons[-1]
        block = draw[shown]
        names = [p for p in POLICIES if p in block["policies"]]
        if not names:
            continue
        roots = block["policies"][names[0]]["episodes"]
        lines.append(f"### {name} ({roots} roots), {shown} actions: successes/episodes (safety-clean)")
        lines.append("| Family | " + " | ".join(names) + " |")
        lines.append("| --- |" + " --- |" * len(names))
        families = sorted({f for p in names for f in block["policies"][p]["by_family"]})
        for family in families:
            cells = []
            for p in names:
                cell = block["policies"][p]["by_family"].get(family)
                cells.append(f"{cell['successes']}/{cell['episodes']} ({cell['clean_successes']})" if cell else "-")
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        rows = (
            ("**successes**", lambda s: str(s["successes"])),
            ("safety-clean successes", lambda s: str(s["clean_successes"])),
            ("violation episodes (invalid / false commit / loop)",
             lambda s: f"{s['violation_episodes']} ({s['invalid_action_episodes']}/{s['false_commit_episodes']}/{s['loop_episodes']})"),
            ("mean actions; mean WLS calls", lambda s: f"{s['mean_actions']:.2f}; {s['mean_wls_calls']:.2f}"),
            ("comparisons", lambda s: str(s["comparisons"])),
            ("multi-error roots: successes/episodes (clean); violations",
             lambda s: "{successes}/{episodes} ({clean_successes}); {violation_episodes}".format(**s["by_cohort"]["multi_error"])),
            ("single-error and clean roots: successes/episodes (clean); violations",
             lambda s: "{successes}/{episodes} ({clean_successes}); {violation_episodes}".format(**s["by_cohort"]["single_or_clean"])),
        )
        for label, fmt in rows:
            lines.append(f"| {label} | " + " | ".join(fmt(block["policies"][p]) for p in names) + " |")
        lines += ["", f"{name}: successes (safety-clean); violation episodes; episodes at the budget, per horizon",
                  "| Budget | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
        for horizon in horizons:
            cells = []
            for p in names:
                s = draw[horizon]["policies"].get(p)
                cells.append(f"{s['successes']} ({s['clean_successes']}); {s['violation_episodes']}; {s['budget_exhausted_episodes']}" if s else "-")
            lines.append(f"| {horizon} | " + " | ".join(cells) + " |")
        lines.append("")
        for horizon in horizons:
            for pair, tests in draw[horizon]["paired"].items():
                first, second = pair.split("_vs_")
                s, c, v = tests["success_failures"], tests["clean_success_failures"], tests["violations"]
                lines.append(f"- {name}, {horizon} actions, {first} vs {second}: failures only {first} {s['only_first']} / only {second} "
                             f"{s['only_second']} (p = {s['p']:.3g}); clean-success failures {c['only_first']} / {c['only_second']} "
                             f"(p = {c['p']:.3g}); violations {v['only_first']} / {v['only_second']} (p = {v['p']:.3g}); "
                             f"{s['paired_roots']} paired roots")
        lines.append("")
    diff = result.get("draw_differences_at_full_budget") or {}
    if diff:
        names = [p for p in POLICIES if p in diff]
        lines.append(f"### Draw 1 against draw 2 at {full} actions (successes draw 1 / draw 2, safety-clean in parentheses)")
        lines.append("| Family | " + " | ".join(names) + " |")
        lines.append("| --- |" + " --- |" * len(names))
        families = sorted({f for p in names for f in diff[p]["by_family"]})
        for family in families:
            cells = []
            for p in names:
                cell = diff[p]["by_family"].get(family)
                cells.append(f"{cell['successes'][0]} / {cell['successes'][1]} ({cell['clean_successes'][0]} / {cell['clean_successes'][1]})"
                             if cell else "-")
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        lines.append("| **all** | " + " | ".join(
            f"{diff[p]['successes'][0]} / {diff[p]['successes'][1]} ({diff[p]['clean_successes'][0]} / {diff[p]['clean_successes'][1]})"
            for p in names) + " |")
        lines.append("| violation episodes | " + " | ".join(
            f"{diff[p]['violation_episodes'][0]} / {diff[p]['violation_episodes'][1]}" for p in names) + " |")
        lines.append("| mean actions | " + " | ".join(
            f"{diff[p]['mean_actions'][0]:.2f} / {diff[p]['mean_actions'][1]:.2f}" for p in names) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--draw2", default="test_draw2")
    parser.add_argument("--prefix", default="test2")
    parser.add_argument("--budgets", nargs="*", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.cell_dir, args.draw2, args.prefix, args.budgets)
    episodes = result.pop("episodes")
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
        records = args.output.with_name(args.output.stem + "_episodes.jsonl")
        records.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in episodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
