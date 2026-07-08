#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from three_phase_nlm.hif_parameter_estimator import estimate_hif_location_magnitude


PRODUCTION_GATES = {
    "line_top1_accuracy": ("==", 1.0),
    "median_alpha_absolute_error": ("<=", 0.05),
    "p90_alpha_absolute_error": ("<=", 0.10),
    "median_r_hif_relative_error": ("<=", 0.20),
    "false_precision_rate": ("<=", 0.05),
    "top_k_parameter_contains_truth": (">=", 0.95),
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def top_nlm_branch(rec: Mapping[str, Any]) -> int | None:
    diagnostic = rec.get("nlm_diagnostic")
    groups = diagnostic.get("top_hif_groups") if isinstance(diagnostic, Mapping) else None
    if not isinstance(groups, list) or not groups or not isinstance(groups[0], Mapping):
        return None
    try:
        return int(groups[0]["branch_row0"])
    except Exception:
        return None


def truth_power_kw(label: Mapping[str, Any]) -> float | None:
    r_ohm = maybe_float(label.get("r_hif_ohm"))
    kv_ln = maybe_float(label.get("kv_ln")) or (1.0 / math.sqrt(3.0))
    if r_ohm is None or r_ohm <= 0:
        return None
    return (float(kv_ln) * 1000.0) ** 2 / r_ohm / 1000.0


def gate_passes(value: Any, op: str, threshold: float) -> bool:
    parsed = maybe_float(value)
    if parsed is None:
        return False
    if op == "==":
        return abs(parsed - float(threshold)) <= 1e-12
    if op == "<=":
        return parsed <= float(threshold)
    if op == ">=":
        return parsed >= float(threshold)
    raise ValueError(f"Unsupported gate operator: {op}")


def production_gate_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    for metric, (op, threshold) in PRODUCTION_GATES.items():
        value = summary.get(metric)
        gates[metric] = {
            "value": value,
            "op": op,
            "threshold": threshold,
            "passed": gate_passes(value, op, threshold),
        }
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model-based HIF parameter estimates on synthetic samples.")
    parser.add_argument("--samples", type=Path, default=Path("artifacts/measurements/out_measurements_single_error_hif_bridge_500/samples.jsonl"))
    parser.add_argument("--limit", type=int, default=20, help="Maximum HIF rows to evaluate; use 0 for all rows.")
    parser.add_argument("--alpha-grid-size", type=int, default=15)
    parser.add_argument("--r-grid-size", type=int, default=17)
    parser.add_argument("--r-hif-pu-min", type=float, default=5.0)
    parser.add_argument("--r-hif-pu-max", type=float, default=1000.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha-hit-tol", type=float, default=0.05)
    parser.add_argument("--r-rel-hit-tol", type=float, default=0.20)
    parser.add_argument("--pristine-model-dir", default=str(REPO_ROOT / "IEEE_14_OpenDSS"))
    parser.add_argument(
        "--enforce-production-gates",
        action="store_true",
        help="Exit nonzero unless the synthetic HIF parameter-estimation batch passes production gates.",
    )
    args = parser.parse_args()

    rows = 0
    evaluated = 0
    line_top1_correct = 0
    skipped_no_nlm_top1 = 0
    estimator_failures = 0
    alpha_errors: list[float] = []
    r_rel_errors: list[float] = []
    p_rel_errors: list[float] = []
    alpha_within_5 = 0
    alpha_within_10 = 0
    top_k_contains_truth = 0
    ambiguous = 0
    false_precision = 0
    examples: list[dict[str, Any]] = []

    for line_no, rec in iter_jsonl(args.samples):
        if rec.get("scenario") != "high_impedance_fault":
            continue
        if args.limit and rows >= int(args.limit):
            break
        rows += 1
        label = rec.get("label") if isinstance(rec.get("label"), Mapping) else {}
        truth_branch = int(label.get("branch_row0"))
        candidate_branch = top_nlm_branch(rec)
        if candidate_branch is None:
            skipped_no_nlm_top1 += 1
            continue
        if candidate_branch == truth_branch:
            line_top1_correct += 1
        else:
            examples.append(
                {
                    "line_no": line_no,
                    "id": rec.get("id"),
                    "status": "line_top1_miss",
                    "truth_branch_row0": truth_branch,
                    "candidate_branch_row0": candidate_branch,
                }
            )
            continue

        payload = estimate_hif_location_magnitude(
            case_path="case14",
            candidate_branch_row0=candidate_branch,
            z_obs=rec.get("z_obs"),
            three_phase_voltages=rec.get("three_phase_voltages"),
            pristine_model_dir=args.pristine_model_dir,
            load_scale=float((rec.get("op_point") or {}).get("load_scale", 1.0)),
            top_k=int(args.top_k),
            alpha_grid_size=int(args.alpha_grid_size),
            r_grid_size=int(args.r_grid_size),
            r_hif_pu_min=float(args.r_hif_pu_min),
            r_hif_pu_max=float(args.r_hif_pu_max),
        )
        if not payload.get("success"):
            estimator_failures += 1
            examples.append({"line_no": line_no, "id": rec.get("id"), "status": "estimator_failed", "error": payload.get("error")})
            continue

        evaluated += 1
        est = payload.get("estimated") if isinstance(payload.get("estimated"), Mapping) else {}
        fit = payload.get("fit") if isinstance(payload.get("fit"), Mapping) else {}
        alpha_truth = maybe_float(label.get("split_ratio"))
        r_truth = maybe_float(label.get("r_hif_pu"))
        p_truth = truth_power_kw(label)
        alpha_est = maybe_float(est.get("alpha_from_from_bus"))
        r_est = maybe_float(est.get("r_hif_pu"))
        p_est = maybe_float(est.get("p_hif_kw"))
        if alpha_truth is not None and alpha_est is not None:
            err = abs(alpha_est - alpha_truth)
            alpha_errors.append(err)
            alpha_within_5 += int(err <= 0.05)
            alpha_within_10 += int(err <= 0.10)
        if r_truth is not None and r_truth > 0 and r_est is not None:
            r_rel_errors.append(abs(r_est - r_truth) / r_truth)
        if p_truth is not None and p_truth > 0 and p_est is not None:
            p_rel_errors.append(abs(p_est - p_truth) / p_truth)
        if fit.get("ambiguity"):
            ambiguous += 1
        if alpha_errors and alpha_errors[-1] > 0.10 and not fit.get("ambiguity"):
            false_precision += 1

        candidates = payload.get("top_parameter_candidates")
        if isinstance(candidates, list) and alpha_truth is not None and r_truth is not None and r_truth > 0:
            for cand in candidates:
                if not isinstance(cand, Mapping):
                    continue
                cand_alpha = maybe_float(cand.get("alpha_from_from_bus"))
                cand_r = maybe_float(cand.get("r_hif_pu"))
                if cand_alpha is None or cand_r is None:
                    continue
                if abs(cand_alpha - alpha_truth) <= args.alpha_hit_tol and abs(cand_r - r_truth) / r_truth <= args.r_rel_hit_tol:
                    top_k_contains_truth += 1
                    break

    summary = {
        "rows_seen": rows,
        "evaluated": evaluated,
        "skipped_no_nlm_top1": skipped_no_nlm_top1,
        "estimator_failures": estimator_failures,
        "line_top1_accuracy": line_top1_correct / rows if rows else None,
        "median_alpha_absolute_error": percentile(alpha_errors, 50),
        "p90_alpha_absolute_error": percentile(alpha_errors, 90),
        "alpha_within_5_percent": alpha_within_5 / evaluated if evaluated else None,
        "alpha_within_10_percent": alpha_within_10 / evaluated if evaluated else None,
        "median_r_hif_relative_error": percentile(r_rel_errors, 50),
        "median_p_hif_relative_error": percentile(p_rel_errors, 50),
        "top_k_parameter_contains_truth": top_k_contains_truth / evaluated if evaluated else None,
        "ambiguous_case_rate": ambiguous / evaluated if evaluated else None,
        "false_precision_rate": false_precision / evaluated if evaluated else None,
        "examples": examples[:10],
    }
    gates = production_gate_report(summary)
    summary["production_gates"] = gates
    summary["production_gates_passed"] = all(item["passed"] for item in gates.values())
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.enforce_production_gates and not summary["production_gates_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
