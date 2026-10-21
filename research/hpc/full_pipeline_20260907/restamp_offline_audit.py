#!/usr/bin/env python3
"""Recompute the offline teacher-target audit of stored collection rows for one tool.

    python restamp_offline_audit.py --collection-dir OUT/r1/collection \\
        --tool compare_parameter_candidates --receipt OUT/r1/collection/restamp_receipt.json

The resumable collector stores every finished episode under ``rollouts/`` and,
on resume, reloads those rows, re-marks their research eligibility and
rebuilds the D1 files and the mixture (scripts/run_dagger_research.py,
``collect_resumable`` and ``export_research_rows``).  The offline
teacher-target audit, however, is computed once at rollout time and stored in
each row.  When the audit's own vocabulary changes (2026-10-21: the bounded
candidate comparison admitted as a read-only target, after the 2026-10-21
cell's collection quarantined all 81 of its comparison labels as an unknown
action class), the stored verdicts of that tool's rows are recomputed here
with the current audit function from the row's own fields, the row's
research eligibility is re-marked the same way the runner does, every
modified file is backed up beside the rollouts, and a receipt records what
changed.  The read-only audit branch reads nothing but the preferred action
and the observable evidence gate, so no oracle state is needed to recompute
it; a row whose tool takes any other branch is left untouched and counted.
Delete the assembled D1 files and the stage receipt afterwards and resubmit
the collection stage: with every root complete it rolls nothing out and only
reassembles.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.offline_teacher_target_audit import offline_teacher_target_audit  # noqa: E402

READ_ONLY_TOOLS_SAFE_TO_RESTAMP = frozenset({"compare_parameter_candidates"})


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def restamp_file(path: Path, *, tool: str, mark_eligibility) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {"rows": len(rows), "tool_rows": 0, "audit_changed": 0, "newly_eligible": 0, "newly_ineligible": 0}
    updated: list[dict[str, Any]] = []
    for row in rows:
        preferred = row.get("preferred_action") if isinstance(row.get("preferred_action"), dict) else None
        if preferred is None or str(preferred.get("tool")) != tool:
            updated.append(row)
            continue
        counts["tool_rows"] += 1
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        gate = labels.get("training_decision_evidence_verified")
        if gate is None:
            gate = row.get("training_decision_evidence_verified")
        audit = offline_teacher_target_audit(
            preferred_action=preferred,
            oracle_state={},
            policy_observation=row.get("policy_observation") or {},
            scenario={},
            env=None,
            observable_evidence_passed=bool(gate),
        )
        new_row = copy.deepcopy(row)
        if new_row.get("offline_teacher_target_audit") != audit:
            counts["audit_changed"] += 1
            new_row["offline_teacher_target_audit"] = audit
        before = bool(row.get("research_label_eligible"))
        new_row = mark_eligibility(new_row)
        after = bool(new_row.get("research_label_eligible"))
        counts["newly_eligible"] += int(after and not before)
        counts["newly_ineligible"] += int(before and not after)
        updated.append(new_row)
    if counts["audit_changed"] or counts["newly_eligible"] or counts["newly_ineligible"]:
        backup = path.parent.parent / "rollouts_before_restamp" / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        _write_jsonl_atomic(path, updated)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection-dir", required=True, type=Path)
    parser.add_argument("--tool", default="compare_parameter_candidates")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.tool not in READ_ONLY_TOOLS_SAFE_TO_RESTAMP:
        parser.error(f"{args.tool} takes an audit branch that needs oracle state; only {sorted(READ_ONLY_TOOLS_SAFE_TO_RESTAMP)} can be restamped offline")
    from scripts.run_dagger_research import mark_research_label_eligibility

    rollouts = args.collection_dir / "rollouts"
    files = sorted(rollouts.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no rollout files under {rollouts}")
    totals = {"files": 0, "files_changed": 0, "rows": 0, "tool_rows": 0, "audit_changed": 0, "newly_eligible": 0, "newly_ineligible": 0}
    changed_files: list[str] = []
    for path in files:
        counts = restamp_file(path, tool=args.tool, mark_eligibility=mark_research_label_eligibility)
        totals["files"] += 1
        for key in ("rows", "tool_rows", "audit_changed", "newly_eligible", "newly_ineligible"):
            totals[key] += counts[key]
        if counts["audit_changed"]:
            totals["files_changed"] += 1
            changed_files.append(path.name)
    receipt = {"contract": "research_full_pipeline_restamp_offline_audit_v1", "tool": args.tool,
               "collection_dir": str(args.collection_dir.resolve()), **totals, "changed_files": changed_files}
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "changed_files"}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
