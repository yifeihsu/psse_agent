"""Check a suite copied from another cell against this cell's expert aggregate.

    python check_suite_exclusion.py --d0-raw OUT/d0/aggregate.raw.jsonl --suite-dir OUT/suite \\
        [--reference-d0-raw OTHER/out/d0/aggregate.raw.jsonl]

The suite builder excludes every D0 root from the suites it draws.  A suite
copied from another cell was drawn around that cell's aggregate, so once this
cell has regenerated its own aggregate the exclusion has to be checked again:
no root of the copied suites may be a D0 root here.  With --reference-d0-raw
the two aggregates' root sets are compared as well (a regenerated aggregate
with the same seed and plan is expected to draw the same roots).  Exits 1 on
an overlap, 0 otherwise; prints a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _fingerprint(container: Any) -> str:
    if not isinstance(container, dict):
        return ""
    for holder in (container, container.get("metadata") or {}, container.get("grouping") or {}):
        if isinstance(holder, dict) and holder.get("physical_root_fingerprint"):
            return str(holder["physical_root_fingerprint"])
    return ""


def d0_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            root = _fingerprint(json.loads(line))
            if not root:
                raise SystemExit(f"a D0 raw row of {path} has no physical_root_fingerprint")
            roots.add(root)
    return roots


def suite_roots(files: Iterable[Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        roots = {_fingerprint(row) for row in payload if isinstance(row, dict)}
        roots.discard("")
        if roots:
            out[path.name] = roots
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d0-raw", required=True, type=Path)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--reference-d0-raw", type=Path, default=None)
    args = parser.parse_args(argv)
    here = d0_roots(args.d0_raw)
    suites = suite_roots(sorted(args.suite_dir.glob("*.json")))
    report: dict[str, Any] = {"d0_roots": len(here), "suites": {}, "overlap": {}}
    overlap = False
    for name, roots in suites.items():
        shared = sorted(roots & here)
        report["suites"][name] = len(roots)
        if shared:
            overlap = True
            report["overlap"][name] = shared
    if args.reference_d0_raw is not None:
        reference = d0_roots(args.reference_d0_raw)
        report["reference"] = {
            "d0_roots": len(reference),
            "same_root_set": here == reference,
            "only_here": len(here - reference),
            "only_in_reference": len(reference - here),
        }
    report["passed"] = not overlap
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if not overlap else 1


if __name__ == "__main__":
    sys.exit(main())
