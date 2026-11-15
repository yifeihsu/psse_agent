"""The rows manifest of the complete-evidence clone cell: what the deploy uploads, checked against what the
selection wrote and against the view its rows were checked for.

    python -m research.hpc.clone_complete_evidence_20261115.rows_manifest --selection-dir DIR --commit COMMIT --output manifest.json

The selection is the open-rules selection's script run on rows rendered under ``complete_evidence``, so the views,
the report and the refusals are the open-rules clone cell's (``clone_open_rules_20261112/rows_manifest.py``: a row
count that differs, a failure the checks name, a carriage return, a selection of another contract).  One refusal
more: the row checks must say they checked the rows for the complete-evidence view (``checks/row_checks.json`` key
``complete_evidence``), so the rows of the open-rules selection cannot be deployed into this cell by a wrong path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import COMPLETE_EVIDENCE_CONTRACT  # noqa: E402
from research.hpc.clone_open_rules_20261112 import rows_manifest as _open_rules  # noqa: E402

MANIFEST_CONTRACT = "compound_cloning_complete_evidence_rows_manifest_v1"
SELECTION_CONTRACT = _open_rules.SELECTION_CONTRACT
TRAIN_VIEW, VALIDATION_VIEW = _open_rules.TRAIN_VIEW, _open_rules.VALIDATION_VIEW
REPORT, CHECKS = _open_rules.REPORT, _open_rules.CHECKS
ManifestError = _open_rules.ManifestError


def build(selection_dir: Path, *, commit: str) -> dict[str, Any]:
    manifest = _open_rules.build(selection_dir, commit=commit)
    checks = json.loads((Path(selection_dir) / CHECKS).read_text(encoding="utf-8"))
    view = checks.get("complete_evidence")
    if not isinstance(view, dict) or view.get("contract") != COMPLETE_EVIDENCE_CONTRACT:
        raise ManifestError(f"the row checks do not say the rows were checked for {COMPLETE_EVIDENCE_CONTRACT!r}: "
                            "these are not the rows of the complete-evidence selection, or their checks are another view's")
    manifest["contract"] = MANIFEST_CONTRACT
    manifest["complete_evidence"] = dict(view)
    return manifest


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
