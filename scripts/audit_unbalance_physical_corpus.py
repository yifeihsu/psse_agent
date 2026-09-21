#!/usr/bin/env python3
"""Audit an IEEE-14 three-phase unbalance corpus against the operator WLS and scenario admission.

For every unbalance row: refit the observed SCADA vector with the operator WLS
(chi-square 0.01 OR normalized residual 4.0), record the voltage unbalance
factor at the labeled bus, and run the scenario generator's discovered-mode
admission (anomaly margin as configured, default 1.25). Nothing is refit or
relabeled to force admission. Writes wls_observations.jsonl, an admission
manifest and summary.json into a fresh output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.providers.scenario_generator import (  # noqa: E402
    Round0ScenarioGenerator, ScenarioRejected, _wls_json, chi2_threshold, voltage_unbalance_factors,
)

VUF_BINS = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 10.0)]


def fit(z, sigma):
    payload = _wls_json("case14", [float(v) for v in z], measurement_sigma=list(sigma))
    if not payload.get("success"):
        return {"success": False, "alarm": None, "error": str(payload.get("error"))}
    residuals = np.abs(np.asarray(payload.get("r") or [], dtype=float))
    dof = int(payload.get("dof", 95))
    statistic = float(payload.get("global_residual_sum") or 0.0)
    threshold = float(chi2_threshold(dof, 0.01))
    max_r = float(residuals.max()) if residuals.size else 0.0
    return {"success": True, "chi_square_statistic": statistic, "chi_square_threshold": threshold,
            "chi_square_ratio": statistic / threshold, "max_normalized_residual": max_r,
            "normalized_residual_threshold": 4.0, "argmax_residual_index": int(np.argmax(residuals)) if residuals.size else -1,
            "alarm": statistic > threshold or max_r >= 4.0,
            "alarm_margin_1p25": statistic > 1.25 * threshold or max_r >= 5.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, help="directory holding samples.jsonl and meta.json")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--anomaly-margin", type=float, default=1.25)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.corpus / "samples.jsonl"
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    generator = Round0ScenarioGenerator(
        seed=args.seed, imbalance_sample_path=path, chi2_alpha=0.01, normalized_residual_threshold=4.0,
        anomaly_margin=args.anomaly_margin, waveform_signature_mode={"three_phase_unbalance": "discovered"},
    )
    rows = generator._imbalance_rows()
    healthy = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line).get("scenario") == "no_error"]
    counts: Counter = Counter()
    by_vuf = defaultdict(Counter)
    by_bus = defaultdict(Counter)
    rejections: Counter = Counter()
    manifest = []
    with (args.out / "wls_observations.jsonl").open("w", encoding="utf-8") as observations:
        for row in healthy:
            result = fit(row["z_obs"], row["sigma_z"])
            counts["healthy_rows"] += 1
            counts["healthy_alarms"] += result.get("alarm") is True
            counts["healthy_alarms_margin_1p25"] += result.get("alarm_margin_1p25") is True
            observations.write(json.dumps({"kind": "healthy", "id": row["id"], "wls": result}) + "\n")
        for ordinal, row in enumerate(rows):
            sigma = row["sigma_z"]
            result = fit(row["z_obs"], sigma)
            factors = voltage_unbalance_factors(row.get("three_phase_voltages_clean") or row.get("three_phase_voltages"))
            labeled_bus = str(row["label"]["unbalance_bus_name"]).lower()
            vuf = next((float(f["vuf"]) for f in factors if str(f.get("bus", "")).lower() == labeled_bus), None)
            if vuf is None and factors:
                vuf = float(factors[0]["vuf"])
            counts["unbalance_rows"] += 1
            counts["alarms"] += result.get("alarm") is True
            counts["alarms_margin_1p25"] += result.get("alarm_margin_1p25") is True
            try:
                scenario = generator._unbalance_scenario(row, ordinal)
                admitted, reason = True, None
                counts["scenario_admitted"] += 1
            except ScenarioRejected as exc:
                admitted, reason = False, exc.reason
                rejections[exc.reason] += 1
            for lo, hi in VUF_BINS:
                if vuf is not None and lo <= vuf < hi:
                    key = f"[{lo},{hi})"
                    by_vuf[key]["rows"] += 1
                    by_vuf[key]["alarms"] += result.get("alarm") is True
                    by_vuf[key]["admitted"] += admitted
            by_bus[row["label"]["unbalance_bus"]]["rows"] += 1
            by_bus[row["label"]["unbalance_bus"]]["admitted"] += admitted
            observations.write(json.dumps({"kind": "unbalance", "id": row["id"], "bus": row["label"]["unbalance_bus"],
                                           "vuf_labeled_bus": vuf, "wls": result, "admitted": admitted, "reason": reason}) + "\n")
            manifest.append({"source": str(path), "source_sha256": source_sha, "id": row["id"], "admitted": admitted,
                             "reason": reason, "vuf_labeled_bus": vuf,
                             "criterion": f"recoverable/discovered WLS admission, margin {args.anomaly_margin:.2f}",
                             "expert_episode_verified": False})
    with (args.out / "unbalance_admission_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item) + "\n")
    summary = {"corpus": str(path), "source_sha256": source_sha, "counts": dict(counts),
               "admission_rejections": dict(rejections), "by_vuf_labeled_bus": {k: dict(v) for k, v in by_vuf.items()},
               "by_unbalance_bus": {str(k): dict(v) for k, v in sorted(by_bus.items())},
               "settings": {"sigma_vm": 0.001, "sigma_power": 0.01, "detector": "chi_square_0.01_or_normalized_residual_4.0",
                            "admission_mode": "recoverable/discovered", "admission_anomaly_margin": args.anomaly_margin,
                            "operator_vm_channel": "phase_a_line_to_neutral_voltage_magnitude_pu"},
               "scope": "Fixed synthetic corpus; scenario admission is not expert success or training completion."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "rejections": dict(rejections)}, indent=1))


if __name__ == "__main__":
    main()
