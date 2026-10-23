#!/usr/bin/env python3
"""The conventional-only cell's extra evaluation suites (2026-10-23).

    build_eval_suites.py --cell-dir PIPE --validation-source SUITE --hard-source SUITE --output-dir OUT

``validation``: the conventional roots of the 2026-10-07 classifier cell's
development suite (the 96 roots of the paper's Table I), less any root of
this cell's D0, round suites or test set.  It selects among the DAgger
iterates (BC0, R1, R2) without touching the test set.

``hard``: the conventional roots of the 2026-10-15 hard draw's development
set (its harmonic roots dropped: scada_only has no spectra), less any root of
this cell's D0, round suites, test set or the validation suite.  A stress test
outside the training distribution: physical admission instead of the
teacher-solvable one, meter errors drawn by WLS effect ratio (1.5 to 3)
instead of the 10-sigma floor, parameter rankings drawn by stratum (many
ambiguous and misranked), and flow-meter pairs that mimic a fault on a line.

Writes ``validation.json``, ``hard.json`` and ``report.json`` (families,
strata, mimic variants, dropped roots and why, sha256s).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
CONVENTIONAL = (
    "no_error", "measurement", "multi_measurement", "parameter", "topology",
    "measurement+parameter", "measurement+topology",
)


def _exclusion():
    spec = importlib.util.spec_from_file_location("check_suite_exclusion", HERE / "check_suite_exclusion.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _family(row: Mapping[str, Any]) -> str:
    return str((row.get("grouping") or {}).get("scenario_family") or row.get("scenario_family") or "")


def select(
    rows: Sequence[Mapping[str, Any]], excluded: Mapping[str, set[str]], fingerprint
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the conventional rows whose root is in no excluded set; report the rest."""
    kept: list[dict[str, Any]] = []
    dropped: dict[str, list[str]] = {}
    other_families: Counter = Counter()
    for row in rows:
        family = _family(row)
        if family not in CONVENTIONAL:
            other_families[family] += 1
            continue
        root = fingerprint(row)
        reasons = [name for name, roots in excluded.items() if root in roots]
        if reasons:
            dropped[root] = reasons
            continue
        kept.append(dict(row))
    return kept, {"dropped": dropped, "other_families": dict(sorted(other_families.items()))}


def describe(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    strata: Counter = Counter()
    mimic: Counter = Counter()
    for row in rows:
        audit = row.get("audit") or {}
        ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
        if "parameter" in _family(row):
            strata[f"{_family(row)}:{ranking.get('stratum')}"] += 1
        variant = (audit.get("mimic") or {}).get("variant") if isinstance(audit.get("mimic"), Mapping) else None
        if variant:
            mimic[str(variant)] += 1
    return {
        "roots": len(rows),
        "families": dict(sorted(Counter(_family(row) for row in rows).items())),
        "parameter_strata": dict(sorted(strata.items())),
        "mimic_variants": dict(sorted(mimic.items())),
    }


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.write_text(json.dumps(list(rows), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--hard-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    check = _exclusion()
    out = args.cell_dir / "out"
    excluded = {"d0": check.d0_roots(out / "d0" / "aggregate.raw.jsonl")}
    for name in ("r1_training", "r2_training", "development"):
        excluded[f"suite:{name}"] = check.reference_roots(out / "suite" / f"{name}.json")
    validation_rows = json.loads(args.validation_source.read_text(encoding="utf-8"))
    validation, validation_drop = select(validation_rows, excluded, check._fingerprint)
    excluded["validation"] = {check._fingerprint(row) for row in validation}
    hard_rows = json.loads(args.hard_source.read_text(encoding="utf-8"))
    hard, hard_drop = select(hard_rows, excluded, check._fingerprint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": "conventional_only_extra_suites_v1",
        "excluded_sets": {name: len(roots) for name, roots in excluded.items()},
        "validation": {
            "source": str(args.validation_source),
            "source_sha256": hashlib.sha256(args.validation_source.read_bytes()).hexdigest(),
            **describe(validation), **validation_drop,
            "sha256": _write(args.output_dir / "validation.json", validation),
        },
        "hard": {
            "source": str(args.hard_source),
            "source_sha256": hashlib.sha256(args.hard_source.read_bytes()).hexdigest(),
            **describe(hard), **hard_drop,
            "sha256": _write(args.output_dir / "hard.json", hard),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: block[k] for k in ("roots", "families", "dropped")}
                      for name, block in report.items() if isinstance(block, dict) and "roots" in block}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
