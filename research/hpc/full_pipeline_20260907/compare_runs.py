#!/usr/bin/env python3
"""Compare two pipeline summaries family by family (and by stratum).

    python compare_runs.py --previous PREV/pipeline_summary.json \
        --current CUR/pipeline_summary.json [--families topology,measurement+topology]

Prints, for every adapter present in both summaries (bc0, r1, r2, expert), the
successes over episodes per family in the previous and the current run, the
overall rates, and the aggregate's family counts, so a change confined to one
family can be read against an otherwise identical run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(entry: Mapping[str, Any] | None) -> str:
    if not isinstance(entry, Mapping):
        return "-"
    return f"{int(entry.get('successes', 0))}/{int(entry.get('episodes', 0))}"


def _rate(summary: Mapping[str, Any], adapter: str) -> str:
    overall = summary.get("overall") or {}
    entry = overall.get(adapter)
    if isinstance(entry, Mapping):
        successes = entry.get("successes")
        episodes = entry.get("episodes")
        rate = entry.get("rate", entry.get("success_rate"))
        if successes is not None and episodes:
            return f"{int(successes)}/{int(episodes)}" + (f" ({float(rate):.3f})" if rate is not None else "")
    return "-"


def compare(previous: Mapping[str, Any], current: Mapping[str, Any], families: list[str] | None) -> str:
    lines: list[str] = []
    prev_by = previous.get("success_by_family") or {}
    cur_by = current.get("success_by_family") or {}
    adapters = [a for a in ("bc0", "r1", "r2", "expert") if a in prev_by and a in cur_by]
    all_families = sorted(set().union(*(set(prev_by[a]) | set(cur_by[a]) for a in adapters))) if adapters else []
    selected = families or all_families
    lines.append("| Family | " + " | ".join(f"{a} prev | {a} now" for a in adapters) + " |")
    lines.append("|---|" + "---|" * (2 * len(adapters)))
    for family in selected:
        cells = []
        for adapter in adapters:
            cells.append(_cell(prev_by[adapter].get(family)))
            cells.append(_cell(cur_by[adapter].get(family)))
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| Adapter | previous overall | current overall |")
    lines.append("|---|---|---|")
    for adapter in adapters:
        lines.append(f"| {adapter} | {_rate(previous, adapter)} | {_rate(current, adapter)} |")
    lines.append("")
    prev_strata = previous.get("success_by_family_and_stratum") or {}
    cur_strata = current.get("success_by_family_and_stratum") or {}
    for family in selected:
        for adapter in adapters:
            prev_f = (prev_strata.get(adapter) or {}).get(family) or {}
            cur_f = (cur_strata.get(adapter) or {}).get(family) or {}
            strata = sorted(set(prev_f) | set(cur_f))
            if len(strata) > 1 or (strata and strata != ["not_applicable"]):
                cells = ", ".join(f"{s}: {_cell(prev_f.get(s))} -> {_cell(cur_f.get(s))}" for s in strata)
                lines.append(f"- {family} / {adapter}: {cells}")
    lines.append("")
    for label, summary in (("previous", previous), ("current", current)):
        d0 = summary.get("d0") or {}
        lines.append(
            f"- {label} aggregate: {d0.get('raw_roots')} roots, {d0.get('raw_rows')} rows; "
            f"families {json.dumps(d0.get('raw_families'))}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--families", default=None, help="comma-separated subset of families to print")
    args = parser.parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else None
    print(compare(_load(args.previous), _load(args.current), families))


if __name__ == "__main__":
    main()
