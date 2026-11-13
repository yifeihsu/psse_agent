"""The rows manifest of the open-rules clone cell: what the deploy uploads, checked against what the selection wrote.

    python -m research.hpc.clone_open_rules_20261112.rows_manifest --selection-dir DIR --commit COMMIT --output manifest.json

The selection (``research.compound_errors.open_rules_selection``) writes the two views and its report; the local
row checks (``checks/row_checks.json``: the message checks of every row against its run-time tool list) sit beside
them.  The manifest records each view's digest and row count with the selection's settings, and refuses views that
are not the ones the report and the checks describe: a row count that differs, a failure the checks name, a
carriage return in a view, a selection of another contract.  The training stage copies the manifest into its
receipt.
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

MANIFEST_CONTRACT = "compound_cloning_open_rules_rows_manifest_v1"
SELECTION_CONTRACT = "compound_open_rules_selection_v1"
TRAIN_VIEW, VALIDATION_VIEW = "selection.train_view.jsonl", "selection.validation.jsonl"
REPORT, CHECKS = "selection_report.json", "checks/row_checks.json"
#: The selection's settings the manifest carries, so a receipt says which selection a pass trained on.
SETTINGS = ("truth_roots", "thin", "keep_absent_targets", "keep_true_rejected_rollbacks", "demonstration_contrast", "reserved_roots",
            "validation_cases", "validation_rows", "seed", "system")


class ManifestError(ValueError):
    pass


def describe(path: Path) -> dict[str, Any]:
    """Digest and row count of a view; a carriage return anywhere refuses it (the cluster reads LF)."""
    digest, rows = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for line in stream:
            if b"\r" in line:
                raise ManifestError(f"{path}: carriage return in row {rows + 1}")
            digest.update(line)
            rows += bool(line.strip())
    return {"file": Path(path).name, "sha256": digest.hexdigest(), "rows": rows}


def build(selection_dir: Path, *, commit: str) -> dict[str, Any]:
    selection_dir = Path(selection_dir)
    report_path, checks_path = selection_dir / REPORT, selection_dir / CHECKS
    for path in (selection_dir / TRAIN_VIEW, selection_dir / VALIDATION_VIEW, report_path, checks_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ManifestError(f"missing: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if report.get("contract") != SELECTION_CONTRACT:
        raise ManifestError(f"{report_path}: contract {report.get('contract')!r}, not {SELECTION_CONTRACT!r}")
    views = {"train": describe(selection_dir / TRAIN_VIEW), "validation": describe(selection_dir / VALIDATION_VIEW)}
    expected = {"train": report.get("rows_train"), "validation": report.get("rows_validation_view")}
    counts = checks.get("counts") or {}
    for name, found in views.items():
        if found["rows"] != expected[name]:
            raise ManifestError(f"the {name} view has {found['rows']} rows, the selection report says {expected[name]}")
        if counts.get(f"rows | {name}") != found["rows"] or counts.get(f"rows passing the message checks | {name}") != found["rows"]:
            raise ManifestError(f"the row checks do not cover the {name} view as it is ({found['rows']} rows; checks {counts})")
    if checks.get("failures"):
        raise ManifestError(f"the row checks name failures: {checks['failures']}")
    if not any(int(value) for value in (checks.get("cases") or {}).values()):
        raise ManifestError("the row checks count no case")
    return {
        "contract": MANIFEST_CONTRACT,
        "views": views,
        "selection_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "row_checks_sha256": hashlib.sha256(checks_path.read_bytes()).hexdigest(),
        "selection": {key: report.get(key) for key in SETTINGS if key in report},
        "cases": report.get("cases"),
        "cases_in_views": checks.get("cases"),
        "contrast_rows": checks.get("contrast_rows"),
        "prompt_characters": checks.get("prompt_characters"),
        "source_commit": commit,
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
