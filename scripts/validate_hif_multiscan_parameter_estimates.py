#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from three_phase_nlm.hif_multiscan_estimator import (  # noqa: E402
    estimate_hif_location_magnitude_multiscan,
)


PRODUCTION_GATES = {
    "line_top1_accuracy": ("==", 1.0),
    "phase_accuracy": (">=", 0.95),
    "full_effective_rank_rate": ("==", 1.0),
    "median_alpha_absolute_error": ("<=", 0.05),
    "p90_alpha_absolute_error": ("<=", 0.10),
    "median_r_hif_relative_error": ("<=", 0.20),
    "false_precision_rate": ("<=", 0.05),
    "top_k_parameter_contains_truth": (">=", 0.95),
    "near_best_alpha_interval_coverage": (">=", 0.90),
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def _maybe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _top_nlm_branch(rec: Mapping[str, Any]) -> int | None:
    diagnostic = rec.get("nlm_diagnostic")
    groups = diagnostic.get("top_hif_groups") if isinstance(diagnostic, Mapping) else None
    if not isinstance(groups, list) or not groups or not isinstance(groups[0], Mapping):
        return None
    try:
        return int(groups[0]["branch_row0"])
    except Exception:
        return None


def _gate_passes(value: Any, operator: str, threshold: float) -> bool:
    parsed = _maybe_float(value)
    if parsed is None:
        return False
    if operator == "==":
        return abs(parsed - threshold) <= 1e-12
    if operator == "<=":
        return parsed <= threshold
    if operator == ">=":
        return parsed >= threshold
    raise ValueError(f"Unsupported gate operator {operator!r}")


def _gate_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        metric: {
            "value": summary.get(metric),
            "op": operator,
            "threshold": threshold,
            "passed": _gate_passes(summary.get(metric), operator, threshold),
        }
        for metric, (operator, threshold) in PRODUCTION_GATES.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-scan IEEE-14 HIF parameter estimates.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20, help="Maximum HIF windows; use 0 for all.")
    parser.add_argument("--scan-count", type=int, default=0, help="Use this many leading scans; 0 uses the full window.")
    parser.add_argument("--max-scans", type=int, default=20)
    parser.add_argument(
        "--scan-selection",
        choices=["all", "diversity_greedy", "information_greedy"],
        default="information_greedy",
    )
    parser.add_argument(
        "--resistance-mode",
        choices=["shared", "scan_specific_smooth"],
        default="shared",
    )
    parser.add_argument("--alpha-grid-size", type=int, default=31)
    parser.add_argument("--r-grid-size", type=int, default=35)
    parser.add_argument("--r-hif-pu-min", type=float, default=5.0)
    parser.add_argument("--r-hif-pu-max", type=float, default=1000.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--refine-top-n", type=int, default=3)
    parser.add_argument("--local-max-nfev", type=int, default=40)
    parser.add_argument("--alpha-hit-tol", type=float, default=0.05)
    parser.add_argument("--r-relative-hit-tol", type=float, default=0.20)
    parser.add_argument("--condition-number-limit", type=float, default=1e6)
    parser.add_argument("--absolute-correlation-limit", type=float, default=0.98)
    parser.add_argument("--use-label-phase", action="store_true", help="Oracle-phase diagnostic only; disabled by default.")
    parser.add_argument("--results-output", type=Path, help="Optional JSONL output with one estimator result per event.")
    parser.add_argument("--output", type=Path, help="Optional path for the aggregate JSON summary.")
    parser.add_argument("--enforce-production-gates", action="store_true")
    args = parser.parse_args()
    if args.enforce_production_gates and args.use_label_phase:
        parser.error("Production gates must search phase; --use-label-phase is diagnostic-only")

    rows_seen = 0
    line_top1_correct = 0
    evaluated = 0
    estimator_failures = 0
    alpha_errors: list[float] = []
    r_errors: list[float] = []
    phase_correct = 0
    smallest_singular_values: list[float] = []
    condition_numbers: list[float] = []
    absolute_correlations: list[float] = []
    top_k_hits = 0
    near_alpha_coverage = 0
    false_precision = 0
    full_rank = 0
    condition_pass = 0
    correlation_pass = 0
    status_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []

    for line_no, rec in _iter_jsonl(args.samples):
        if rec.get("scenario") != "high_impedance_fault":
            continue
        if args.limit and rows_seen >= int(args.limit):
            break
        rows_seen += 1
        label = rec.get("shared_label", rec.get("label"))
        label = label if isinstance(label, Mapping) else {}
        raw_scans = rec.get("scans")
        if not isinstance(raw_scans, list) or not raw_scans:
            estimator_failures += 1
            failure = {"line_no": line_no, "id": rec.get("id"), "status": "missing_scans"}
            examples.append(failure)
            event_results.append(failure)
            continue
        selected_input_scans = raw_scans[: int(args.scan_count)] if int(args.scan_count) > 0 else raw_scans

        truth_branch = int(label["branch_row0"])
        candidate_branch = _top_nlm_branch(rec)
        if candidate_branch is None:
            failure = {"line_no": line_no, "id": rec.get("id"), "status": "missing_nlm_top1"}
            examples.append(failure)
            event_results.append(failure)
            continue
        if candidate_branch != truth_branch:
            failure = {
                "line_no": line_no,
                "id": rec.get("id"),
                "status": "line_top1_miss",
                "truth_branch_row0": truth_branch,
                "candidate_branch_row0": candidate_branch,
            }
            examples.append(failure)
            event_results.append(failure)
            continue
        line_top1_correct += 1

        payload = estimate_hif_location_magnitude_multiscan(
            candidate_branch_row0=candidate_branch,
            scans=selected_input_scans,
            sigma_z=rec.get("sigma_z"),
            candidate_phase=str(label.get("phase")) if args.use_label_phase else None,
            resistance_mode=args.resistance_mode,
            max_scans=int(args.max_scans),
            scan_selection=args.scan_selection,
            top_k=int(args.top_k),
            alpha_grid_size=int(args.alpha_grid_size),
            r_grid_size=int(args.r_grid_size),
            r_hif_pu_min=float(args.r_hif_pu_min),
            r_hif_pu_max=float(args.r_hif_pu_max),
            condition_number_limit=float(args.condition_number_limit),
            absolute_correlation_limit=float(args.absolute_correlation_limit),
            refine_top_n=max(0, int(args.refine_top_n)),
            local_max_nfev=max(0, int(args.local_max_nfev)),
        )
        if not payload.get("success"):
            estimator_failures += 1
            failure = {
                "line_no": line_no,
                "id": rec.get("id"),
                "status": "estimator_failed",
                "error": payload.get("error"),
            }
            examples.append(failure)
            event_results.append(failure)
            continue

        evaluated += 1
        estimated = payload.get("estimated") if isinstance(payload.get("estimated"), Mapping) else {}
        fit = payload.get("fit") if isinstance(payload.get("fit"), Mapping) else {}
        uncertainty = payload.get("uncertainty") if isinstance(payload.get("uncertainty"), Mapping) else {}
        observability = payload.get("observability") if isinstance(payload.get("observability"), Mapping) else {}
        status_counts[str(observability.get("status"))] += 1

        alpha_truth = _maybe_float(label.get("split_ratio"))
        alpha_est = _maybe_float(estimated.get("alpha_from_from_bus"))
        r_truth = _maybe_float(label.get("r_hif_pu"))
        r_est = _maybe_float(estimated.get("r_hif_pu"))
        phase_truth = str(label.get("phase") or "").upper()
        phase_est = str(estimated.get("phase") or "").upper()
        phase_correct += int(bool(phase_truth) and phase_est == phase_truth)
        alpha_error = None
        if alpha_truth is not None and alpha_est is not None:
            alpha_error = abs(alpha_est - alpha_truth)
            alpha_errors.append(alpha_error)
        if r_truth is not None and r_truth > 0.0 and r_est is not None:
            r_errors.append(abs(r_est - r_truth) / r_truth)

        rank = observability.get("effective_rank")
        full_rank += int(rank == 2)
        singular = _maybe_float(observability.get("smallest_singular_value"))
        if singular is not None:
            smallest_singular_values.append(singular)
        condition = _maybe_float(observability.get("condition_number"))
        if condition is not None:
            condition_numbers.append(condition)
            condition_pass += int(condition <= float(args.condition_number_limit))
        correlation = _maybe_float(observability.get("alpha_log_r_correlation"))
        if correlation is not None:
            absolute_correlations.append(abs(correlation))
            correlation_pass += int(abs(correlation) <= float(args.absolute_correlation_limit))

        point_claim_allowed = bool(payload.get("parameter_identifiable") and not fit.get("ambiguity"))
        if alpha_error is not None and alpha_error > 0.10 and point_claim_allowed:
            false_precision += 1

        interval = uncertainty.get("near_best_alpha_interval")
        if (
            alpha_truth is not None
            and isinstance(interval, list)
            and len(interval) == 2
            and _maybe_float(interval[0]) is not None
            and _maybe_float(interval[1]) is not None
        ):
            near_alpha_coverage += int(float(interval[0]) <= alpha_truth <= float(interval[1]))

        candidates = payload.get("top_parameter_candidates")
        if isinstance(candidates, list) and alpha_truth is not None and r_truth is not None and r_truth > 0.0:
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                cand_alpha = _maybe_float(candidate.get("alpha_from_from_bus"))
                cand_r = _maybe_float(candidate.get("r_hif_pu"))
                if cand_alpha is None or cand_r is None:
                    continue
                if (
                    abs(cand_alpha - alpha_truth) <= float(args.alpha_hit_tol)
                    and abs(cand_r - r_truth) / r_truth <= float(args.r_relative_hit_tol)
                ):
                    top_k_hits += 1
                    break

        event_results.append(
            {
                "id": rec.get("id"),
                "line_no": line_no,
                "truth": {
                    "branch_row0": truth_branch,
                    "phase": phase_truth,
                    "alpha_from_from_bus": alpha_truth,
                    "r_hif_pu": r_truth,
                },
                "errors": {
                    "alpha_absolute_error": alpha_error,
                    "r_hif_relative_error": (
                        abs(r_est - r_truth) / r_truth
                        if r_truth is not None and r_truth > 0.0 and r_est is not None
                        else None
                    ),
                    "phase_correct": bool(phase_truth and phase_est == phase_truth),
                },
                "estimator": payload,
            }
        )

    summary = {
        "rows_seen": rows_seen,
        "evaluated": evaluated,
        "estimator_failures": estimator_failures,
        "line_top1_accuracy": line_top1_correct / rows_seen if rows_seen else None,
        "phase_accuracy": phase_correct / evaluated if evaluated else None,
        "median_alpha_absolute_error": _percentile(alpha_errors, 50),
        "p90_alpha_absolute_error": _percentile(alpha_errors, 90),
        "median_r_hif_relative_error": _percentile(r_errors, 50),
        "p90_r_hif_relative_error": _percentile(r_errors, 90),
        "top_k_parameter_contains_truth": top_k_hits / evaluated if evaluated else None,
        "near_best_alpha_interval_coverage": near_alpha_coverage / evaluated if evaluated else None,
        "false_precision_rate": false_precision / evaluated if evaluated else None,
        "full_effective_rank_rate": full_rank / evaluated if evaluated else None,
        "median_smallest_singular_value": _percentile(smallest_singular_values, 50),
        "p90_condition_number": _percentile(condition_numbers, 90),
        "p90_absolute_alpha_log_r_correlation": _percentile(absolute_correlations, 90),
        "condition_number_pass_rate": condition_pass / evaluated if evaluated else None,
        "correlation_pass_rate": correlation_pass / evaluated if evaluated else None,
        "observability_status_counts": dict(sorted(status_counts.items())),
        "configuration": {
            "scan_count": int(args.scan_count) or "full_window",
            "max_scans": int(args.max_scans),
            "scan_selection": args.scan_selection,
            "resistance_mode": args.resistance_mode,
            "alpha_grid_size": int(args.alpha_grid_size),
            "r_grid_size": int(args.r_grid_size),
            "refine_top_n": int(args.refine_top_n),
            "local_max_nfev": int(args.local_max_nfev),
            "oracle_phase": bool(args.use_label_phase),
        },
        "examples": examples[:10],
    }
    gates = _gate_report(summary)
    summary["production_gates"] = gates
    summary["production_gates_passed"] = all(item["passed"] for item in gates.values())
    if args.results_output:
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        with args.results_output.open("w", encoding="utf-8") as handle:
            for item in event_results:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.enforce_production_gates and not summary["production_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
