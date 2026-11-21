"""The rows manifest of the minimal environment's supervised pass: what the deploy uploads, checked against what the
selection wrote.

    python -m research.hpc.minimal_sft_20261121.rows_manifest --selection-dir DIR --commit COMMIT --output manifest.json

The two views are read whole and refused when they are not the files the selection's report describes (a row count
or a checksum that differs), when a row is not a row of the minimal interface (its protocol, its six tools), when a
case has rows in both views, or when a file holds a carriage return (the views go to the cluster byte for byte).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.actions import TOOLS  # noqa: E402
from psse_env.minimal.rows import MINIMAL_PROTOCOL, ROW_CONTRACT  # noqa: E402
from research.minimal_env.select_rows import SELECTION_CONTRACT  # noqa: E402

MANIFEST_CONTRACT = "minimal_sft_rows_manifest_v1"
TRAIN_VIEW, VALIDATION_VIEW, REPORT = "train.jsonl", "validation.jsonl", "selection_report.json"


class ManifestError(ValueError):
    """The selection folder is not what the pass may train on."""


def _view(path: Path) -> dict[str, Any]:
    """Row count, checksum, cases and action counts of one view; refuses a row of another interface."""
    if not path.is_file():
        raise ManifestError(f"{path} is missing")
    digest = hashlib.sha256()
    rows = 0
    cases: set[str] = set()
    actions: dict[str, int] = {}
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            digest.update(raw)
            if b"\r" in raw:
                raise ManifestError(f"{path.name} line {number} holds a carriage return")
            if not raw.strip():
                continue
            row = json.loads(raw)
            label = str(row.get("example_id") or f"line {number}")
            metadata = row.get("metadata") or {}
            if metadata.get("protocol") != MINIMAL_PROTOCOL or metadata.get("row_contract") != ROW_CONTRACT:
                raise ManifestError(f"{path.name}: {label} is not a row of {MINIMAL_PROTOCOL}")
            names = [item.get("function", {}).get("name") for item in row.get("tools") or []]
            if names != list(TOOLS):
                raise ManifestError(f"{path.name}: {label} does not carry the six tools of the minimal interface, but {names}")
            case = str(row.get("physical_root_fingerprint") or "")
            if not case:
                raise ManifestError(f"{path.name}: {label} names no case (physical_root_fingerprint)")
            cases.add(case)
            call = row["messages"][-1]["tool_calls"][0]["function"]["name"]
            actions[call] = actions.get(call, 0) + 1
            rows += 1
    return {"file": path.name, "rows": rows, "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "cases": cases,
            "actions": dict(sorted(actions.items()))}


def build(selection_dir: Path, *, commit: str) -> dict[str, Any]:
    selection_dir = Path(selection_dir)
    report_path = selection_dir / REPORT
    if not report_path.is_file():
        raise ManifestError(f"{report_path} is missing: this is not a folder select_rows wrote")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("contract") != SELECTION_CONTRACT:
        raise ManifestError(f"the selection report's contract is {report.get('contract')!r}, not {SELECTION_CONTRACT!r}")
    views = {"train": _view(selection_dir / TRAIN_VIEW), "validation": _view(selection_dir / VALIDATION_VIEW)}
    for name, expected_rows in (("train", report.get("train_rows")), ("validation", report.get("validation_rows"))):
        if views[name]["rows"] != expected_rows:
            raise ManifestError(f"the {name} view holds {views[name]['rows']} rows, the selection report says {expected_rows}")
        if views[name]["sha256"] != (report.get("sha256") or {}).get(name):
            raise ManifestError(f"the {name} view is not the file the selection report describes (checksum)")
    shared = views["train"]["cases"] & views["validation"]["cases"]
    if shared:
        raise ManifestError(f"{len(shared)} cases have rows in both views")
    return {
        "contract": MANIFEST_CONTRACT,
        "commit": str(commit),
        "selection_contract": SELECTION_CONTRACT,
        "protocol": MINIMAL_PROTOCOL,
        "selection": {key: report.get(key) for key in ("settings", "cases", "train_rows_by_kind", "rows_dropped", "train_prompt_characters")},
        "views": {name: {**{key: value for key, value in view.items() if key != "cases"}, "cases": len(view["cases"])}
                  for name, view in views.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = build(args.selection_dir, commit=args.commit)
    except ManifestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({name: {"rows": view["rows"], "sha256": view["sha256"][:12]} for name, view in manifest["views"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
