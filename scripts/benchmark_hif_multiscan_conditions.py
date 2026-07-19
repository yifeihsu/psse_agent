#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_hif_multiscan_parameter_estimates.py"
LABEL_KEYS = ("branch_row0", "split_ratio", "phase", "r_hif_pu")


def _rows(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_matched_events(diverse: Path, identical: Path) -> int:
    diverse_rows = _rows(diverse)
    identical_rows = _rows(identical)
    if len(diverse_rows) != len(identical_rows):
        raise ValueError("Diverse and identical datasets have different event counts")
    for position, (left, right) in enumerate(zip(diverse_rows, identical_rows)):
        if left.get("id") != right.get("id"):
            raise ValueError(f"Event id mismatch at row {position}")
        left_label = left.get("shared_label", left.get("label", {}))
        right_label = right.get("shared_label", right.get("label", {}))
        if any(left_label.get(key) != right_label.get(key) for key in LABEL_KEYS):
            raise ValueError(f"Shared HIF label mismatch at row {position}")
    return len(diverse_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched IEEE-14 HIF scan-count conditions A-E.")
    parser.add_argument("--diverse-samples", type=Path, required=True)
    parser.add_argument("--identical-samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--alpha-grid-size", type=int, default=7)
    parser.add_argument("--r-grid-size", type=int, default=9)
    parser.add_argument("--refine-top-n", type=int, default=2)
    parser.add_argument("--local-max-nfev", type=int, default=20)
    args = parser.parse_args()

    event_count = _assert_matched_events(args.diverse_samples, args.identical_samples)
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("A_single", args.diverse_samples, 1, 1, "all"),
        ("B_identical_20", args.identical_samples, 20, 20, "all"),
        ("C_diverse_5", args.diverse_samples, 20, 5, "information_greedy"),
        ("D_diverse_10", args.diverse_samples, 20, 10, "information_greedy"),
        ("E_diverse_20", args.diverse_samples, 20, 20, "all"),
    ]
    summaries: dict[str, Any] = {}
    for name, samples, scan_count, max_scans, selection in conditions:
        summary_path = args.out / f"{name}.summary.json"
        results_path = args.out / f"{name}.results.jsonl"
        command = [
            sys.executable,
            str(VALIDATOR),
            "--samples",
            str(samples),
            "--limit",
            str(max(0, int(args.limit))),
            "--scan-count",
            str(scan_count),
            "--max-scans",
            str(max_scans),
            "--scan-selection",
            selection,
            "--alpha-grid-size",
            str(args.alpha_grid_size),
            "--r-grid-size",
            str(args.r_grid_size),
            "--refine-top-n",
            str(args.refine_top_n),
            "--local-max-nfev",
            str(args.local_max_nfev),
            "--results-output",
            str(results_path),
            "--output",
            str(summary_path),
        ]
        print(f"Running {name}...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
        elapsed = time.perf_counter() - started
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["elapsed_seconds"] = elapsed
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries[name] = summary

    combined = {
        "matched_event_count": event_count,
        "phase_source": "searched_A_B_C; synthetic phase used only for evaluation",
        "configuration": {
            "alpha_grid_size": args.alpha_grid_size,
            "r_grid_size": args.r_grid_size,
            "refine_top_n": args.refine_top_n,
            "local_max_nfev": args.local_max_nfev,
            "limit": args.limit or "all",
        },
        "conditions": summaries,
    }
    output = args.out / "parameter_metrics.json"
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "conditions": list(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
