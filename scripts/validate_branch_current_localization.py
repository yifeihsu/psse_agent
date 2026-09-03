#!/usr/bin/env python3
"""Score closed-form branch-current localization against hidden labels.

Reads an IEEE-14 measurement corpus (``samples.jsonl``) that carries the
``three_phase_branch_currents`` channel and evaluates, per row:

* three-phase-imbalance rows: does the per-phase shunt-power-spread ranking
  put the labeled unbalance bus first (and within top-k), and does the line
  differential null test stay quiet;
* high-impedance-fault rows: per scan, does the two-terminal differential
  rank the labeled line first, name the labeled phase, and how close are the
  closed-form position and resistance; across the window, the per-scan
  medians.

Labels are used only for scoring.  The script never writes them into the
corpus and never feeds them to the analysis.
"""

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

from three_phase_nlm.branch_current_analysis import (  # noqa: E402
    BRANCH_CURRENT_CHANNEL,
    BRANCH_CURRENT_SIGMA_KEY,
    DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    line_differential_null_test,
    terminal_current_hif_localization,
    terminal_current_hif_localization_multiscan,
    unbalance_source_localization,
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _sigma(row: Mapping[str, Any], scan: Mapping[str, Any] | None, override: float | None) -> float:
    if override is not None and override > 0.0:
        return float(override)
    for source in (scan or {}, row):
        value = source.get(BRANCH_CURRENT_SIGMA_KEY)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            return parsed
    return DEFAULT_BRANCH_CURRENT_SIGMA_PU


def score_unbalance(rows: list[Mapping[str, Any]], *, top_k: int, sigma_override: float | None) -> dict[str, Any]:
    scored = 0
    top1 = 0
    topk = 0
    null_quiet = 0
    significant = 0
    separations: list[float] = []
    misses: Counter[str] = Counter()
    for row in rows:
        label = row.get("label") or {}
        expected = label.get("unbalance_bus")
        voltages = row.get("three_phase_voltages")
        currents = row.get(BRANCH_CURRENT_CHANNEL)
        if expected is None or not voltages or not currents:
            continue
        localization = unbalance_source_localization(
            voltages, currents, top_k=top_k, sigma_pu=_sigma(row, None, sigma_override)
        )
        if localization is None:
            misses["localization_none"] += 1
            continue
        scored += 1
        ranked = [item["bus"] for item in localization["top_unbalance_source_buses"]]
        top1 += int(localization["bus_1based"] == int(expected))
        topk += int(int(expected) in ranked[:top_k])
        significant += int(bool(localization["significant"]))
        separations.append(float(localization["separation_ratio"]))
        null = line_differential_null_test(
            voltages, currents, sigma_pu=_sigma(row, None, sigma_override)
        )
        null_quiet += int(bool(null) and not null["hif_like_differential_present"])
        if localization["bus_1based"] != int(expected):
            misses[f"expected_b{expected}_got_b{localization['bus_1based']}"] += 1
    return {
        "family": "three_phase_unbalance",
        "rows_scored": scored,
        "top1_accuracy": top1 / scored if scored else None,
        f"top{top_k}_accuracy": topk / scored if scored else None,
        "line_differential_null_quiet_rate": null_quiet / scored if scored else None,
        "top1_significant_rate": significant / scored if scored else None,
        "separation_ratio_median": _percentile(separations, 50),
        "separation_ratio_p10": _percentile(separations, 10),
        "misses": dict(misses),
    }


def score_hif(rows: list[Mapping[str, Any]], *, top_k: int, sigma_override: float | None) -> dict[str, Any]:
    scans_scored = 0
    line_top1 = 0
    line_topk = 0
    phase_ok = 0
    detected = 0
    alpha_errors: list[float] = []
    r_rel_errors: list[float] = []
    windows = 0
    window_line_top1 = 0
    window_phase_ok = 0
    window_alpha_errors: list[float] = []
    window_r_rel_errors: list[float] = []
    coherent_line_top1 = 0
    coherent_phase_ok = 0
    coherent_detected = 0
    coherent_alpha_errors: list[float] = []
    coherent_r_rel_errors: list[float] = []
    for row in rows:
        label = row.get("shared_label") or row.get("label") or {}
        expected_row0 = label.get("branch_row0")
        expected_phase = str(label.get("phase") or "").upper()
        expected_alpha = label.get("split_ratio", label.get("alpha_from_from_bus"))
        expected_r = label.get("r_hif_pu")
        if expected_row0 is None:
            continue
        scans = row.get("scans") or [row]
        per_scan_alpha: list[float] = []
        per_scan_r: list[float] = []
        per_scan_phase: list[str] = []
        per_scan_line: list[int] = []
        for scan in scans:
            voltages = scan.get("three_phase_voltages")
            currents = scan.get(BRANCH_CURRENT_CHANNEL)
            if not voltages or not currents:
                continue
            payload = terminal_current_hif_localization(
                voltages, currents, top_k=top_k, sigma_pu=_sigma(row, scan, sigma_override)
            )
            if payload is None:
                continue
            scans_scored += 1
            ranked = [item["branch_row0"] for item in payload["top_hif_groups"]]
            line_top1 += int(ranked[0] == int(expected_row0))
            line_topk += int(int(expected_row0) in ranked[:top_k])
            phase_ok += int(payload["suspected_phase"] == expected_phase)
            detected += int(bool(payload["differential_detected"]))
            estimate = payload.get("terminal_current_estimate") or {}
            per_scan_line.append(int(ranked[0]))
            per_scan_phase.append(str(payload["suspected_phase"]))
            if ranked[0] == int(expected_row0) and expected_alpha is not None:
                error = abs(float(estimate["alpha_from_from_bus"]) - float(expected_alpha))
                alpha_errors.append(error)
                per_scan_alpha.append(float(estimate["alpha_from_from_bus"]))
            if ranked[0] == int(expected_row0) and expected_r:
                r_est = float(estimate.get("r_hif_pu") or math.nan)
                if math.isfinite(r_est) and r_est > 0.0:
                    r_rel_errors.append(abs(r_est - float(expected_r)) / float(expected_r))
                    per_scan_r.append(r_est)
        if per_scan_line:
            windows += 1
            majority_line = Counter(per_scan_line).most_common(1)[0][0]
            majority_phase = Counter(per_scan_phase).most_common(1)[0][0]
            window_line_top1 += int(majority_line == int(expected_row0))
            window_phase_ok += int(majority_phase == expected_phase)
            if per_scan_alpha and expected_alpha is not None:
                window_alpha_errors.append(abs(float(np.median(per_scan_alpha)) - float(expected_alpha)))
            if per_scan_r and expected_r:
                window_r_rel_errors.append(
                    abs(float(np.exp(np.median(np.log(per_scan_r)))) - float(expected_r)) / float(expected_r)
                )
            # The deployment NLM tool averages the complex differential across
            # the whole window before ranking; score that path as well.
            coherent = terminal_current_hif_localization_multiscan(
                [scan for scan in scans if isinstance(scan, Mapping)],
                top_k=top_k,
                sigma_pu=_sigma(row, None, sigma_override),
            )
            if coherent is not None:
                coherent_top = int(coherent["top_hif_groups"][0]["branch_row0"])
                coherent_line_top1 += int(coherent_top == int(expected_row0))
                coherent_phase_ok += int(coherent["suspected_phase"] == expected_phase)
                coherent_detected += int(bool(coherent["differential_detected"]))
                estimate = coherent.get("terminal_current_estimate") or {}
                if coherent_top == int(expected_row0) and expected_alpha is not None and estimate:
                    coherent_alpha_errors.append(
                        abs(float(estimate["alpha_from_from_bus"]) - float(expected_alpha))
                    )
                    r_est = estimate.get("r_hif_pu")
                    if expected_r and r_est:
                        coherent_r_rel_errors.append(
                            abs(float(r_est) - float(expected_r)) / float(expected_r)
                        )
    return {
        "family": "hif",
        "scans_scored": scans_scored,
        "line_top1_accuracy": line_top1 / scans_scored if scans_scored else None,
        f"line_top{top_k}_accuracy": line_topk / scans_scored if scans_scored else None,
        "phase_accuracy": phase_ok / scans_scored if scans_scored else None,
        "differential_detected_rate": detected / scans_scored if scans_scored else None,
        "alpha_abs_error_median": _percentile(alpha_errors, 50),
        "alpha_abs_error_p90": _percentile(alpha_errors, 90),
        "r_rel_error_median": _percentile(r_rel_errors, 50),
        "r_rel_error_p90": _percentile(r_rel_errors, 90),
        "windows_scored": windows,
        "window_line_top1_accuracy": window_line_top1 / windows if windows else None,
        "window_phase_accuracy": window_phase_ok / windows if windows else None,
        "window_median_alpha_abs_error_median": _percentile(window_alpha_errors, 50),
        "window_median_alpha_abs_error_p90": _percentile(window_alpha_errors, 90),
        "window_median_r_rel_error_median": _percentile(window_r_rel_errors, 50),
        "window_median_r_rel_error_p90": _percentile(window_r_rel_errors, 90),
        "coherent_window_line_top1_accuracy": coherent_line_top1 / windows if windows else None,
        "coherent_window_phase_accuracy": coherent_phase_ok / windows if windows else None,
        "coherent_window_detected_rate": coherent_detected / windows if windows else None,
        "coherent_window_alpha_abs_error_median": _percentile(coherent_alpha_errors, 50),
        "coherent_window_alpha_abs_error_p90": _percentile(coherent_alpha_errors, 90),
        "coherent_window_r_rel_error_median": _percentile(coherent_r_rel_errors, 50),
        "coherent_window_r_rel_error_p90": _percentile(coherent_r_rel_errors, 90),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--sigma-pu",
        type=float,
        default=None,
        help="Override the declared branch-current sigma used for detection floors.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = list(_iter_jsonl(args.samples))
    by_scenario: Counter[str] = Counter(str(row.get("scenario")) for row in rows)
    unbalance_rows = [row for row in rows if row.get("scenario") == "three_phase_imbalance"]
    hif_rows = [row for row in rows if row.get("scenario") == "high_impedance_fault"]
    result: dict[str, Any] = {
        "samples": str(args.samples),
        "rows": len(rows),
        "rows_by_scenario": dict(by_scenario),
        "rows_with_branch_currents": sum(
            1
            for row in rows
            if row.get(BRANCH_CURRENT_CHANNEL)
            or any(scan.get(BRANCH_CURRENT_CHANNEL) for scan in (row.get("scans") or []) if isinstance(scan, Mapping))
        ),
    }
    if unbalance_rows:
        result["unbalance"] = score_unbalance(
            unbalance_rows, top_k=int(args.top_k), sigma_override=args.sigma_pu
        )
    if hif_rows:
        result["hif"] = score_hif(hif_rows, top_k=int(args.top_k), sigma_override=args.sigma_pu)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
