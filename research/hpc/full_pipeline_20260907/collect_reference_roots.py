#!/usr/bin/env python3
"""Every physical root of a list of earlier sets, for a fresh draw to avoid (2026-10-29).

    collect_reference_roots.py --output ROOTS.json [--exclude-dir DIR ...] PATH [PATH ...]

PATH is a suite (a JSON list of scenario rows), an aggregate (JSONL rows), or a
glob of them, expanded here so that a pattern matching nothing is recorded
instead of silently vanishing.  Every row's physical root is read the way
check_suite_exclusion.py reads it (the row, its ``metadata``, its
``grouping``); a JSON list of fingerprint strings, and rows that carry the
fingerprint under another key (``physical_root``, ``fp``, ``root``: the
conventional-only cell's list of local-preflight roots), are read as roots
too, though check_suite_exclusion.py cannot read such a file itself.
Writes ROOTS.json, a JSON list of ``{"physical_root_fingerprint": ...}``
(sorted, unique) that build_suite.py takes as a ``--protected-suite``, and
ROOTS.report.json: per file, its size, sha256, rows, roots, rows without a
root and whether check_suite_exclusion.py can read it as a
``--reference-suite`` (``checkable_paths``).  Files under an ``--exclude-dir``
(the draw's own output) are skipped, and a file reached twice (a symlinked
stage of a cell that reused another cell's) is read once.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
#: Keys under which a hand-written root list stores the fingerprint.
ALIAS_KEYS = ("physical_root", "fp", "root")


def _exclusion():
    spec = importlib.util.spec_from_file_location("check_suite_exclusion", HERE / "check_suite_exclusion.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 23), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_roots(path: Path, fingerprint: Callable[[Any], str]) -> dict[str, Any]:
    """Rows, roots and root-less rows of one suite or aggregate file."""
    roots: set[str] = set()
    rows = rootless = row_roots = 0

    def take(item: Any) -> None:
        nonlocal rows, rootless, row_roots
        rows += 1
        if isinstance(item, str) and item.startswith("physical_"):
            roots.add(item)
            return
        root = fingerprint(item) if isinstance(item, dict) else ""
        if root:
            roots.add(root)
            row_roots += 1
            return
        alias = next((item[key] for key in ALIAS_KEYS if isinstance(item, dict)
                      and isinstance(item.get(key), str) and item[key].startswith("physical_")), "")
        if alias:
            roots.add(alias)
        else:
            rootless += 1

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    take(json.loads(line))
        checkable = rows > 0 and rootless == 0
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload if isinstance(payload, list) else []:
            take(item)
        # check_suite_exclusion.suite_roots reads dict rows only.
        checkable = row_roots > 0
    return {"rows": rows, "roots": roots, "rows_without_root": rootless, "checkable": checkable}


def collect(patterns: Sequence[str], exclude_dirs: Sequence[Path] = ()) -> tuple[set[str], dict[str, Any]]:
    check = _exclusion()
    excluded = [Path(d).resolve() for d in exclude_dirs]
    report: dict[str, Any] = {"contract": "reference_roots_v1", "patterns": {}, "files": {}, "checkable_paths": []}
    union: set[str] = set()
    read: dict[Path, str] = {}
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        report["patterns"][pattern] = len(matches)
        for match in matches:
            path = Path(match)
            if not path.is_file():
                report["files"][match] = {"error": "missing"}
                continue
            resolved = path.resolve()
            if any(resolved == d or d in resolved.parents for d in excluded):
                report["files"][match] = {"skipped": "under an excluded directory"}
                continue
            if resolved in read:
                if match != read[resolved]:
                    report["files"][match] = {"duplicate_of": read[resolved]}
                continue
            read[resolved] = match
            try:
                info = read_roots(path, check._fingerprint)
            except (OSError, ValueError) as exc:
                report["files"][match] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                continue
            union |= info["roots"]
            report["files"][match] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "rows": info["rows"],
                "roots": len(info["roots"]),
                "rows_without_root": info["rows_without_root"],
                "checkable": info["checkable"],
            }
            if info["checkable"]:
                report["checkable_paths"].append(match)
    report["roots"] = len(union)
    report["files_read"] = len(read)
    report["files_with_roots"] = sum(1 for item in report["files"].values() if item.get("roots"))
    return union, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--exclude-dir", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    roots, report = collect(args.paths, args.exclude_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([{"physical_root_fingerprint": root} for root in sorted(roots)], indent=0) + "\n",
                           encoding="utf-8")
    report["output"] = str(args.output)
    report["output_sha256"] = _sha256(args.output)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("roots", "files_read", "files_with_roots")} | {"checkable": len(report["checkable_paths"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
