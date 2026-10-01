"""Paired comparison of two expert arms on the same roots (hypothesis-ranking plan, step 3).

Reads two ``expert_e2e`` output directories built from the same generator seed
and plan, pairs the episodes by scenario id and reports, per family and in
total: truth-audited successes, steps, phasor and spectra acquisitions,
corrections attempted and rejected, false commits, and for roots with a
single true branch or meter whether the first correction of that family hit
the true target (the ordering question the ledger is meant to answer).

    python -m research.hypothesis_ranking.compare_arms --a output/.../step3_baseline --b output/.../step3_ledger --output output/.../step3_compare.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with (path / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[str(row["scenario_id"])] = row
    return rows


def _first_hit(row: Mapping[str, Any]) -> dict[str, bool | None]:
    """Whether the first attempted correction of each family targeted a true fault."""
    truth = row.get("truth") or {}
    result: dict[str, bool | None] = {}
    first: dict[str, Mapping[str, Any]] = {}
    for correction in row.get("corrections") or []:
        tool = str(correction.get("tool"))
        first.setdefault(tool, correction)
    parameter = first.get("correct_parameters")
    if parameter is not None and truth.get("parameter_lines"):
        line = (parameter.get("arguments") or {}).get("line_index")
        result["parameter"] = line in set(truth["parameter_lines"])
    meter = first.get("correct_measurements")
    if meter is not None and truth.get("measurement_indices"):
        group = (meter.get("arguments") or {}).get("suspect_group") or []
        result["measurement"] = bool(group) and all(index in set(truth["measurement_indices"]) for index in group)
    return result


def _metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    tools = row.get("tools") or []
    corrections = row.get("corrections") or []
    return {
        "success": bool(row.get("truth_audited_task_success")),
        "steps": len(tools),
        "phasors": "get_three_phase_context" in tools,
        "spectra": "get_harmonic_context" in tools,
        "corrections": len(corrections),
        "rollbacks": tools.count("rollback_state"),
        "commits": tools.count("commit_state"),
        "false_commits": int(row.get("false_commit_count") or 0),
        "healthy_preserved": row.get("healthy_components_preserved", True) is not False,
        "first_hit": _first_hit(row),
    }


def compare(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    shared = sorted(set(a) & set(b))
    per_family: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "a": Counter(), "b": Counter(), "first_hit": Counter()})
    flips: list[dict[str, Any]] = []
    for scenario_id in shared:
        row_a, row_b = a[scenario_id], b[scenario_id]
        family = str(row_a.get("family"))
        entry = per_family[family]
        entry["n"] += 1
        for label, metrics in (("a", _metrics(row_a)), ("b", _metrics(row_b))):
            counter = entry[label]
            counter["success"] += metrics["success"]
            counter["steps"] += metrics["steps"]
            counter["phasors"] += metrics["phasors"]
            counter["spectra"] += metrics["spectra"]
            counter["corrections"] += metrics["corrections"]
            counter["rollbacks"] += metrics["rollbacks"]
            counter["commits"] += metrics["commits"]
            counter["false_commits"] += metrics["false_commits"]
            counter["healthy_touched"] += 0 if metrics["healthy_preserved"] else 1
            for fam, hit in metrics["first_hit"].items():
                entry["first_hit"][f"{label}_{fam}_n"] += 1
                entry["first_hit"][f"{label}_{fam}_hit"] += bool(hit)
        if _metrics(row_a)["success"] != _metrics(row_b)["success"] or _metrics(row_a)["steps"] != _metrics(row_b)["steps"]:
            flips.append({"scenario_id": scenario_id, "family": family,
                          "a": {"success": _metrics(row_a)["success"], "steps": _metrics(row_a)["steps"], "tools": row_a.get("tools")},
                          "b": {"success": _metrics(row_b)["success"], "steps": _metrics(row_b)["steps"], "tools": row_b.get("tools")}})
    totals = {"a": Counter(), "b": Counter(), "first_hit": Counter(), "n": len(shared)}
    for entry in per_family.values():
        for label in ("a", "b"):
            totals[label].update(entry[label])
        totals["first_hit"].update(entry["first_hit"])
    return {"shared": len(shared), "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a)),
            "per_family": {k: {"n": v["n"], "a": dict(v["a"]), "b": dict(v["b"]), "first_hit": dict(v["first_hit"])} for k, v in sorted(per_family.items())},
            "totals": {"n": totals["n"], "a": dict(totals["a"]), "b": dict(totals["b"]), "first_hit": dict(totals["first_hit"])},
            "differences": flips}


def render(result: Mapping[str, Any], name_a: str, name_b: str) -> str:
    lines = [f"# Paired comparison: {name_a} (A) versus {name_b} (B)\n",
             f"{result['shared']} shared roots" + (f"; only in A: {len(result['only_a'])}, only in B: {len(result['only_b'])}" if result["only_a"] or result["only_b"] else "") + ".\n",
             "| family | n | success A | success B | mean steps A | mean steps B | phasors A/B | spectra A/B | corrections A/B | rollbacks A/B | false commits A/B |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for family, entry in result["per_family"].items():
        a, b, n = entry["a"], entry["b"], entry["n"]
        lines.append(f"| {family} | {n} | {a.get('success', 0)} | {b.get('success', 0)} | {a.get('steps', 0) / max(1, n):.1f} | {b.get('steps', 0) / max(1, n):.1f} | "
                     f"{a.get('phasors', 0)}/{b.get('phasors', 0)} | {a.get('spectra', 0)}/{b.get('spectra', 0)} | {a.get('corrections', 0)}/{b.get('corrections', 0)} | "
                     f"{a.get('rollbacks', 0)}/{b.get('rollbacks', 0)} | {a.get('false_commits', 0)}/{b.get('false_commits', 0)} |")
    t = result["totals"]
    n = max(1, t["n"])
    lines.append(f"| **all** | {t['n']} | {t['a'].get('success', 0)} | {t['b'].get('success', 0)} | {t['a'].get('steps', 0) / n:.1f} | {t['b'].get('steps', 0) / n:.1f} | "
                 f"{t['a'].get('phasors', 0)}/{t['b'].get('phasors', 0)} | {t['a'].get('spectra', 0)}/{t['b'].get('spectra', 0)} | {t['a'].get('corrections', 0)}/{t['b'].get('corrections', 0)} | "
                 f"{t['a'].get('rollbacks', 0)}/{t['b'].get('rollbacks', 0)} | {t['a'].get('false_commits', 0)}/{t['b'].get('false_commits', 0)} |")
    fh = t["first_hit"]
    lines.append("\nFirst correction of a family on the true target (roots with that truth): "
                 f"parameter A {fh.get('a_parameter_hit', 0)}/{fh.get('a_parameter_n', 0)}, B {fh.get('b_parameter_hit', 0)}/{fh.get('b_parameter_n', 0)}; "
                 f"measurement A {fh.get('a_measurement_hit', 0)}/{fh.get('a_measurement_n', 0)}, B {fh.get('b_measurement_hit', 0)}/{fh.get('b_measurement_n', 0)}.\n")
    if result["differences"]:
        lines.append(f"\n## Roots where the arms differ in success or length ({len(result['differences'])})\n")
        for item in result["differences"]:
            lines.append(f"- {item['family']} `{item['scenario_id']}`: A success={item['a']['success']} steps={item['a']['steps']}; "
                         f"B success={item['b']['success']} steps={item['b']['steps']}")
            lines.append(f"  - A: {' > '.join(item['a']['tools'] or [])}")
            lines.append(f"  - B: {' > '.join(item['b']['tools'] or [])}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare(_read(Path(args.a)), _read(Path(args.b)))
    markdown = render(result, Path(args.a).name, Path(args.b).name)
    Path(args.output).write_text(markdown, encoding="utf-8")
    Path(args.output).with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
