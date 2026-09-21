#!/usr/bin/env python3
"""Audit current WLS waveform visibility without tuning or changing detectors.

The canonical operator is the current runtime. The branch-net operator is a
paired measurement-convention check for capacitor-inclusive OpenDSS exports;
it is not silently substituted into the agent environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psse_env.providers.matpower import MatpowerDeploymentProviders
from scripts.evaluate_hif_measurement_recovery import wls_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "output/waveform_detection_20260916")
    args = parser.parse_args()
    provider = MatpowerDeploymentProviders(chi2_alpha=.01, normalized_residual_threshold=4.)
    settings = SimpleNamespace(chi_square_alpha=.01, detection_sigma=4.)
    inputs = [(args.directory / "prepared_faults.jsonl", None),
              (args.directory / "healthy_controls/samples.jsonl", "healthy_controls")]
    details = []
    for path, cohort_override in inputs:
        for ordinal, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            cohort = cohort_override or row["audit_cohort"]
            observed = row.get("z_obs", row.get("measurements"))
            sigma = row["sigma_z"]
            state = {"state_id": f"audit:{cohort}:{ordinal}", "case": "case14",
                     "measurements": observed, "metadata": {"sigma_z": sigma}, "policy_observation": {}}
            try:
                solved = provider._solve(state)
                canonical = provider._wls_detection_metrics(solved)
                canonical["converged"] = bool(solved["payload"]["success"])
            except Exception as exc:
                canonical = {"converged": False, "error": f"{type(exc).__name__}: {exc}"}
            matched = wls_metrics(np.asarray(observed), np.asarray(sigma), settings,
                                  exported_injections=True, input_role="noisy_sensor_snapshot")
            details.append({"cohort": cohort, "source_id": row.get("id", ordinal),
                            "source_ordinal": ordinal, "runtime_canonical": canonical,
                            "branch_net_convention_check": matched})
    summary = {}
    groups = defaultdict(list)
    for row in details:
        groups[row["cohort"]].append(row)
    for cohort, rows in groups.items():
        summary[cohort] = {"root_or_snapshot_count": len(rows)}
        for key in ("runtime_canonical", "branch_net_convention_check"):
            entries = [row[key] for row in rows]
            ok = [row for row in entries if row.get("converged")]
            js = [row["chi_square_statistic"] for row in ok]
            local = [row["max_normalized_residual"] for row in ok]
            summary[cohort][key] = {
                "converged": len(ok), "failed": len(entries) - len(ok),
                "global_alarm": sum(bool(row["chi_square_alarm"]) for row in ok),
                "normalized_residual_alarm": sum(bool(row["normalized_residual_alarm"]) for row in ok),
                "either_alarm": sum(bool(row["chi_square_alarm"] or row["normalized_residual_alarm"]) for row in ok),
                "J_range": [min(js), max(js)] if js else None,
                "J_median": float(np.median(js)) if js else None,
                "max_normalized_residual_range": [min(local), max(local)] if local else None,
            }
    output = {"scope": "fresh fixed-threshold visibility audit; no threshold tuning, no HIF fit acceptance or episode-success claim",
              "settings": {"chi_square_alpha": .01, "nominal_dof": 95,
                           "chi_square_threshold_95dof": float(chi2.ppf(.99, 95)),
                           "normalized_residual_threshold": 4., "rule": "inclusive global OR local alarm"},
              "operator_note": "Runtime canonical case14 is contrasted with the matching branch-net injection convention for DSS capacitor-inclusive observations; observations/noise/covariance are identical in both solves.",
              "summary": summary, "rows": details}
    target = args.directory / "wls_audit.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(target), "settings": output["settings"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
