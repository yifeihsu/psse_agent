#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from three_phase_nlm.ieee14_adapter import ELIGIBLE_HIF_BRANCHES, branch_info_for_row0  # noqa: E402


def _finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except Exception:
        return False


def _top_rows(nlm: Mapping[str, Any]) -> list[int]:
    groups = nlm.get("top_hif_groups")
    if not isinstance(groups, list):
        return []
    rows = []
    for group in groups:
        if isinstance(group, Mapping) and group.get("branch_row0") is not None:
            rows.append(int(group["branch_row0"]))
    return rows


def validate_row(
    row: Mapping[str, Any],
    *,
    allow_non_top1: bool,
    allow_legacy_log_tail: bool,
) -> list[str]:
    issues: list[str] = []
    if row.get("scenario") != "high_impedance_fault":
        issues.append("not_high_impedance_fault")
    if row.get("case") != "IEEE14":
        issues.append("not_ieee14")
    if not _finite_vector(row.get("z_true"), 122):
        issues.append("bad_z_true")
    if not _finite_vector(row.get("z_obs"), 122):
        issues.append("bad_z_obs")

    scans = row.get("scans")
    if scans is not None:
        if not isinstance(scans, list) or not scans:
            issues.append("bad_hif_scan_window")
        else:
            if row.get("scan_count") != len(scans):
                issues.append("scan_count_mismatch")
            topology_ids = set()
            op_points = []
            for position, scan in enumerate(scans):
                if not isinstance(scan, Mapping):
                    issues.append("bad_hif_scan")
                    continue
                if scan.get("scan_index") != position:
                    issues.append("nonsequential_scan_index")
                if not _finite_vector(scan.get("z_obs"), 122):
                    issues.append("bad_scan_z_obs")
                if not isinstance(scan.get("three_phase_voltages"), list):
                    issues.append("missing_scan_three_phase_voltages")
                if not isinstance(scan.get("op_point"), Mapping):
                    issues.append("missing_scan_op_point")
                else:
                    op_points.append(json.dumps(scan.get("op_point"), sort_keys=True))
                if scan.get("topology_id") is not None:
                    topology_ids.add(str(scan.get("topology_id")))
                if "label" in scan or "shared_label" in scan:
                    issues.append("label_leakage_inside_scan")
            if len(topology_ids) > 1:
                issues.append("scan_topology_changed")
            if scans and row.get("z_obs") != scans[0].get("z_obs"):
                issues.append("reference_scan_z_mismatch")
            window_metadata = row.get("window_metadata")
            mode = window_metadata.get("operating_point_mode") if isinstance(window_metadata, Mapping) else None
            if mode == "identical_noise" and len(set(op_points)) > 1:
                issues.append("identical_noise_op_points_differ")
            if mode == "diverse" and len(scans) > 1 and len(set(op_points)) < 2:
                issues.append("diverse_op_points_not_diverse")
            sigma_z = row.get("sigma_z")
            if sigma_z is not None:
                if not _finite_vector(sigma_z, 122) or any(float(value) <= 0.0 for value in sigma_z):
                    issues.append("bad_scan_sigma_z")

    label = row.get("label") if isinstance(row.get("label"), Mapping) else {}
    target = label.get("branch_row0")
    if target is None:
        issues.append("missing_label_branch_row0")
    else:
        target = int(target)

    nlm = row.get("nlm_diagnostic") if isinstance(row.get("nlm_diagnostic"), Mapping) else {}
    if nlm.get("method") == "metadata_fallback":
        issues.append("metadata_fallback")
    if nlm.get("method") != "legacy_three_phase_nlm":
        issues.append("unexpected_method")
    if not bool(nlm.get("success")):
        issues.append("nlm_not_success")
    if not bool(nlm.get("converged")):
        issues.append("nlm_not_converged")
    if nlm.get("legacy_log_tail") is not None and not allow_legacy_log_tail:
        issues.append("legacy_log_tail_present")

    top_rows = _top_rows(nlm)
    groups = nlm.get("top_hif_groups")
    if isinstance(groups, list) and any(
        isinstance(group, Mapping) and group.get("legacy_line_group_index0") is not None
        for group in groups
    ):
        issues.append("legacy_line_group_index_present")
    if not top_rows:
        issues.append("missing_top_hif_groups")
    elif target is not None:
        if top_rows[0] != target and not allow_non_top1:
            issues.append("target_not_top1")
        if target not in top_rows[:3]:
            issues.append("target_not_top3")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-non-top1", action="store_true")
    parser.add_argument("--allow-legacy-log-tail", action="store_true")
    parser.add_argument("--require-all-eligible-branches", action="store_true")
    args = parser.parse_args()

    methods: Counter[str] = Counter()
    branches: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    top1_count = 0
    top3_count = 0
    legacy_log_tail_rows = 0
    rows = 0

    with args.path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            label = row.get("label") if isinstance(row.get("label"), Mapping) else {}
            nlm = row.get("nlm_diagnostic") if isinstance(row.get("nlm_diagnostic"), Mapping) else {}
            methods[str(nlm.get("method"))] += 1
            if label.get("dss_element") is not None:
                branches[str(label["dss_element"])] += 1
            if label.get("phase") is not None:
                phases[str(label["phase"])] += 1
            if nlm.get("legacy_log_tail") is not None:
                legacy_log_tail_rows += 1

            target = label.get("branch_row0")
            top_rows = _top_rows(nlm)
            if target is not None and top_rows:
                target = int(target)
                top1_count += int(top_rows[0] == target)
                top3_count += int(target in top_rows[:3])

            issues = validate_row(
                row,
                allow_non_top1=bool(args.allow_non_top1),
                allow_legacy_log_tail=bool(args.allow_legacy_log_tail),
            )
            for issue in issues:
                issue_counts[issue] += 1
            if issues:
                print(f"{args.path}:{line_no}: {','.join(issues)}", file=sys.stderr)

    eligible_by_element = {
        branch_info_for_row0(int(idx))["dss_element"]: int(idx) for idx in ELIGIBLE_HIF_BRANCHES
    }
    covered_label_rows = {
        eligible_by_element[element] for element in branches if element in eligible_by_element
    }
    missing_eligible = sorted(set(int(i) for i in ELIGIBLE_HIF_BRANCHES) - covered_label_rows)
    if args.require_all_eligible_branches and missing_eligible:
        issue_counts["missing_eligible_branch"] = len(missing_eligible)

    result = {
        "rows": rows,
        "methods": dict(methods),
        "top1_count": top1_count,
        "top3_count": top3_count,
        "legacy_log_tail_rows": legacy_log_tail_rows,
        "branch_counts": dict(sorted(branches.items())),
        "phase_counts": dict(sorted(phases.items())),
        "missing_eligible_branch_row0": missing_eligible,
        "error_count": sum(issue_counts.values()),
        "issues": dict(sorted(issue_counts.items())),
    }
    print(json.dumps(result, sort_keys=True))
    if issue_counts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
