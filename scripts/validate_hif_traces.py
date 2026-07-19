#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _tool_payload(messages: list[Mapping[str, Any]], tool_name: str) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == tool_name:
            return _json_obj(message.get("content"))
    return {}


def validate_row(
    row: Mapping[str, Any],
    *,
    allow_metadata_fallback: bool = False,
    allow_mock_estimator: bool = False,
) -> list[str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["missing_messages"]

    final = _json_obj(messages[-1].get("content") if isinstance(messages[-1], Mapping) else None)
    if not final:
        return ["missing_final_json"]

    verdict = final.get("verdict") if isinstance(final.get("verdict"), Mapping) else {}
    if verdict.get("error_family") != "high_impedance_fault":
        return []

    issues: list[str] = []
    action = final.get("action") if isinstance(final.get("action"), Mapping) else {}
    applied_tools = action.get("applied_tools") if isinstance(action.get("applied_tools"), list) else []
    estimator_tools = {
        "estimate_hif_location_magnitude_from_path",
        "estimate_hif_location_magnitude_multiscan_from_path",
    }
    if action.get("applied_tool") not in estimator_tools:
        issues.append("missing_hif_parameter_estimator_action")
    if "run_three_phase_nlm_from_path" not in applied_tools:
        issues.append("missing_nlm_tool_action")
    if not any(tool in applied_tools for tool in estimator_tools):
        issues.append("missing_hif_parameter_estimator_tool_action")

    nlm = _tool_payload(messages, "run_three_phase_nlm_from_path")
    hif_est = {}
    for estimator_tool in estimator_tools:
        hif_est = _tool_payload(messages, estimator_tool)
        if hif_est:
            break
    top_groups = nlm.get("top_hif_groups") if isinstance(nlm.get("top_hif_groups"), list) else []
    if not top_groups:
        issues.append("missing_nlm_top_hif_groups")
    if any(isinstance(group, Mapping) and group.get("legacy_line_group_index0") is not None for group in top_groups):
        issues.append("legacy_line_group_index_visible")

    if nlm.get("method") == "metadata_fallback" and not allow_metadata_fallback:
        issues.append("metadata_fallback_nlm")
    if hif_est.get("synthetic_oracle") and not allow_mock_estimator:
        issues.append("synthetic_oracle_hif_estimator")
    if not hif_est:
        issues.append("missing_hif_parameter_estimator_payload")
    elif hif_est.get("success") is False:
        issues.append("failed_hif_parameter_estimator")
    hif_estimated = hif_est.get("estimated") if isinstance(hif_est.get("estimated"), Mapping) else {}
    if hif_est and not isinstance(hif_estimated.get("alpha_from_from_bus"), (int, float)):
        issues.append("missing_hif_alpha_estimate")
    if hif_est and not isinstance(hif_estimated.get("r_hif_pu"), (int, float)):
        issues.append("missing_hif_resistance_estimate")
    if hif_est.get("method") == "multiscan_augmented_hif_parameter_estimation":
        observability = hif_est.get("observability") if isinstance(hif_est.get("observability"), Mapping) else {}
        if not isinstance(observability.get("effective_rank"), int):
            issues.append("missing_hif_multiscan_effective_rank")
        if not observability.get("status"):
            issues.append("missing_hif_multiscan_observability_status")

    evidence = final.get("evidence") if isinstance(final.get("evidence"), Mapping) else {}
    final_groups = evidence.get("top_hif_groups") if isinstance(evidence.get("top_hif_groups"), list) else []
    final_estimate = (
        evidence.get("hif_parameter_estimate")
        if isinstance(evidence.get("hif_parameter_estimate"), Mapping)
        else {}
    )
    details = (
        final.get("suspect_location", {}).get("details", {})
        if isinstance(final.get("suspect_location"), Mapping)
        else {}
    )
    if top_groups and isinstance(details, Mapping):
        if details.get("branch_row0") != top_groups[0].get("branch_row0"):
            issues.append("final_location_not_nlm_top1")
    if top_groups and final_groups and final_groups[0].get("branch_row0") != top_groups[0].get("branch_row0"):
        issues.append("final_evidence_not_nlm_top1")
    if hif_estimated and isinstance(details, Mapping):
        observability = hif_est.get("observability") if isinstance(hif_est.get("observability"), Mapping) else {}
        fit = hif_est.get("fit") if isinstance(hif_est.get("fit"), Mapping) else {}
        report_point = bool(
            hif_est.get("method") != "multiscan_augmented_hif_parameter_estimation"
            or (
                observability.get("parameter_identifiable")
                and not fit.get("ambiguity")
                and observability.get("status")
                in {"full_rank_well_conditioned", "full_rank_weakly_conditioned"}
            )
        )
        if report_point:
            if details.get("alpha_from_from_bus") != hif_estimated.get("alpha_from_from_bus"):
                issues.append("final_alpha_not_estimator")
            if details.get("r_hif_pu") != hif_estimated.get("r_hif_pu"):
                issues.append("final_resistance_not_estimator")
        else:
            if not isinstance(details.get("near_best_alpha_interval"), list):
                issues.append("missing_weak_observability_alpha_interval")
            if "alpha_from_from_bus" in details or "distance_percent_from_from_bus" in details:
                issues.append("unsupported_weak_observability_point_alpha")
    if hif_estimated and isinstance(final_estimate, Mapping):
        observability = hif_est.get("observability") if isinstance(hif_est.get("observability"), Mapping) else {}
        fit = hif_est.get("fit") if isinstance(hif_est.get("fit"), Mapping) else {}
        report_point = bool(
            hif_est.get("method") != "multiscan_augmented_hif_parameter_estimation"
            or (
                observability.get("parameter_identifiable")
                and not fit.get("ambiguity")
                and observability.get("status")
                in {"full_rank_well_conditioned", "full_rank_weakly_conditioned"}
            )
        )
        if report_point:
            if final_estimate.get("alpha_from_from_bus") != hif_estimated.get("alpha_from_from_bus"):
                issues.append("final_evidence_alpha_not_estimator")
        else:
            if "alpha_from_from_bus" in final_estimate or "distance_percent_from_from_bus" in final_estimate:
                issues.append("unsupported_weak_observability_evidence_point_alpha")
            if not isinstance(final_estimate.get("near_best_alpha_interval"), list):
                issues.append("missing_weak_observability_evidence_alpha_interval")

    if isinstance(details, Mapping) and "phase" in details:
        has_phase_evidence = bool(
            nlm.get("suspected_phase")
            or nlm.get("phase_scores")
            or hif_estimated.get("phase")
            or hif_est.get("phase_scores")
        )
        if not has_phase_evidence:
            issues.append("unsupported_final_phase")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-metadata-fallback", action="store_true")
    parser.add_argument(
        "--allow-mock-estimator",
        action="store_true",
        help="Allow label-backed mock HIF estimates in smoke-test traces only.",
    )
    args = parser.parse_args()

    counts: dict[str, int] = {}
    rows = 0
    with args.path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            issues = validate_row(
                row,
                allow_metadata_fallback=bool(args.allow_metadata_fallback),
                allow_mock_estimator=bool(args.allow_mock_estimator),
            )
            for issue in issues:
                counts[issue] = counts.get(issue, 0) + 1
            if issues:
                print(f"{args.path}:{line_no}: {','.join(issues)}")

    if counts:
        print(json.dumps({"rows": rows, "issues": counts}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps({"rows": rows, "issues": {}}, sort_keys=True))


if __name__ == "__main__":
    main()
