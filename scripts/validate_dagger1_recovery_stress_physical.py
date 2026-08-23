#!/usr/bin/env python3
"""Run the real-environment admission pass for a recovery-stress parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from psse_env.dagger.build_dagger1_recovery_stress import (
    build_recovery_stress_payload,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-holdout", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = json.loads(
        args.development_holdout.read_text(encoding="utf-8")
    )
    rows = payload.get("dagger1_development")
    if not isinstance(rows, list):
        raise ValueError(
            "development holdout must contain dagger1_development rows"
        )
    suites, report = build_recovery_stress_payload(rows)
    roots = {
        row["grouping"]["physical_root_fingerprint"]
        for suite_rows in suites.values()
        for row in suite_rows
    }
    print(
        json.dumps(
            {
                "passed": True,
                "rows": sum(len(value) for value in suites.values()),
                "distinct_physical_roots": len(roots),
                "validation_records": len(report["validation_records"]),
                "rejected_candidates": report["rejected_candidates"],
                "suite_counts": {
                    key: len(value) for key, value in sorted(suites.items())
                },
                "selected_roots_by_family": report[
                    "selected_roots_by_family"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
