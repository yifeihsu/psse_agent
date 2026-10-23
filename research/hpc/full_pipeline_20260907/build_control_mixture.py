#!/usr/bin/env python3
"""The matched cloning control's training mixture (conventional-only cell, 2026-10-23).

    build_control_mixture.py --cell-dir PIPE --fresh-aggregate DIR [--exclude-suite SUITE ...] \\
        --seed N --output MIXTURE.jsonl

R1 trained on the round-1 mixture: the 437 learner-visited rows of the
round-1 collection and 437 D0 rows (874 rows, 219 steps).  The control keeps
the D0 rows in their places and puts in the place of each collected row a
fresh expert row of the same family, drawn from the balanced training view of
an aggregate generated with a new seed under the D0 settings: the expert's
path and its forced recovery branches, balanced and duplicated the way D0's
training view is, the distribution BC0 was cloned from and R1's D0 rows were
drawn from.  Trained like R1 (BC0 as the initial adapter, the same step
count, learning rate, seed and validation rows), the control differs from R1
only in where the new states come from: the expert's own distribution instead
of the student's.  Fresh roots that are D0 roots, round or test roots, or
roots of the extra evaluation suites are excluded.  Training views repeat
rows (up to two copies), so example ids repeat in R1's mixture and here.
Writes the mixture and ``<output>.report.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
#: The balanced training view only: D0's rows in R1's mixture come from D0's.
VIEWS = ("aggregate.train_view.jsonl",)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _family(row: Mapping[str, Any]) -> str:
    return str(row.get("scenario_family") or (row.get("metadata") or {}).get("scenario_family") or "")


def _root(row: Mapping[str, Any]) -> str:
    for holder in (row, row.get("metadata") or {}, row.get("grouping") or {}):
        if isinstance(holder, Mapping) and holder.get("physical_root_fingerprint"):
            return str(holder["physical_root_fingerprint"])
    return ""


def replace_collected_rows(
    mixture: Sequence[Mapping[str, Any]],
    collected_ids: set[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Put a fresh row of the same family in the place of every collected row.

    Each family's candidates are sorted by example id and shuffled with
    ``f"{seed}:{family}"``; the collected rows of that family take them in
    mixture order.  Candidates are a multiset (a balanced view repeats rows).
    A family without enough candidates is an error: the control must keep
    R1's row count and family mix.
    """
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_family[_family(row)].append(dict(row))
    for family, rows in by_family.items():
        rows.sort(key=lambda row: str(row.get("example_id")))
        random.Random(f"{seed}:{family}").shuffle(rows)
    needed = Counter(_family(row) for row in mixture if str(row.get("example_id")) in collected_ids)
    short = {family: (count, len(by_family.get(family, []))) for family, count in needed.items()
             if len(by_family.get(family, [])) < count}
    if short:
        raise ValueError(f"fresh expert rows short per family (needed, available): {dict(sorted(short.items()))}")
    taken: Counter = Counter()
    result: list[dict[str, Any]] = []
    for row in mixture:
        if str(row.get("example_id")) in collected_ids:
            family = _family(row)
            result.append(by_family[family][taken[family]])
            taken[family] += 1
        else:
            result.append(dict(row))
    kept = [row for row in mixture if str(row.get("example_id")) not in collected_ids]
    fresh_roots = {_root(row) for row, original in zip(result, mixture) if str(original.get("example_id")) in collected_ids}
    if fresh_roots & {_root(row) for row in kept}:
        raise ValueError("a fresh expert root is also a root of the kept D0 rows")
    fresh_ids = [str(row.get("example_id")) for row, original in zip(result, mixture)
                 if str(original.get("example_id")) in collected_ids]
    return result, {
        "rows": len(result),
        "replaced_rows": int(sum(needed.values())),
        "replaced_by_family": dict(sorted(needed.items())),
        "kept_rows": len(kept),
        "fresh_roots": len(fresh_roots),
        "fresh_rows_repeated_ids": len(fresh_ids) - len(set(fresh_ids)),
        "candidates_by_family": {family: len(rows) for family, rows in sorted(by_family.items())},
    }


def _exclusion():
    spec = importlib.util.spec_from_file_location("check_suite_exclusion", HERE / "check_suite_exclusion.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--fresh-aggregate", required=True, type=Path)
    parser.add_argument("--exclude-suite", action="append", default=[], type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--evidence-profile", default="scada_only")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    cell = args.cell_dir.resolve()
    source = cell / "source"
    sys.path.insert(0, str(source if (source / "scripts" / "run_dagger_research.py").is_file() else HERE.parents[2]))
    import scripts.run_dagger_research as research

    collection = cell / "out" / "r1" / "collection"
    mixture = _jsonl(collection / "round1.train.jsonl")
    collected_ids = {str(row.get("example_id")) for row in _jsonl(collection / "d1.chat.jsonl")}
    fresh_raw = _jsonl(args.fresh_aggregate / "aggregate.raw.jsonl")
    fresh_chat = [row for view in VIEWS if (args.fresh_aggregate / view).is_file()
                  for row in _jsonl(args.fresh_aggregate / view)]
    fresh_rows, view_report = research.refresh_d0_training_view(fresh_raw, fresh_chat, evidence_profile=args.evidence_profile)
    check = _exclusion()
    excluded = {"d0": check.d0_roots(cell / "out" / "d0" / "aggregate.raw.jsonl")}
    for path in sorted((cell / "out" / "suite").glob("*.json")):
        roots = check.suite_roots([path]).get(path.name)
        if roots:
            excluded[f"suite:{path.name}"] = roots
    for path in args.exclude_suite:
        excluded[str(path)] = check.reference_roots(path)
    blocked = set().union(*excluded.values())
    candidates = [row for row in fresh_rows if _root(row) and _root(row) not in blocked]
    result, report = replace_collected_rows(mixture, collected_ids, candidates, seed=args.seed)
    research.validate_training_evidence_profile(result, evidence_profile=args.evidence_profile, source="control mixture")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in result), encoding="utf-8")
    report.update({
        "contract": "conventional_only_matched_cloning_control_v1",
        "seed": int(args.seed),
        "r1_mixture": str(collection / "round1.train.jsonl"),
        "r1_mixture_sha256": hashlib.sha256((collection / "round1.train.jsonl").read_bytes()).hexdigest(),
        "fresh_aggregate": str(args.fresh_aggregate),
        "fresh_rows": len(fresh_rows),
        "fresh_roots": len({_root(row) for row in fresh_rows}),
        "fresh_rows_excluded": len(fresh_rows) - len(candidates),
        "excluded_sets": {name: len(roots) for name, roots in excluded.items()},
        "view_refresh": view_report,
        "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    })
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "replaced_rows", "replaced_by_family", "fresh_rows_excluded")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
