#!/usr/bin/env python3
"""Render an inspectable same-channel HIF/meter decomposition from saved fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from three_phase_nlm.conditioned_meter_recovery import diagnose_conditioned_meter_errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    folder = args.experiment.resolve()
    config = json.loads((folder / "config.json").read_text())
    roots = json.loads((folder / "root_results.json").read_text())
    root = max((r for r in roots if "prediction_audit" in r),
               key=lambda r: r["prediction_audit"]["max_event_effect_sigma"])
    samples = [json.loads(line) for line in Path(config["samples"]).read_text().splitlines() if line.strip()]
    row = samples[root["source_ordinal"]]
    replay = json.loads((folder / "roots" / root["root_key"] / "replay.json").read_text())
    observed = np.asarray(row["scans"][0]["z_obs"])
    sigma = np.asarray(row["sigma_z"])
    present = np.asarray(replay["predicted_hif_measurements"])
    absent = np.asarray(replay["predicted_base_measurements"])
    effect = present - absent
    index = int(np.argmax(np.abs(effect / sigma)))
    mixed = observed.copy()
    mixed[index] += config["bias_sigma"] * sigma[index]
    result = diagnose_conditioned_meter_errors(
        mixed, present, sigma, prediction_lower=replay["prediction_lower"],
        prediction_upper=replay["prediction_upper"], detection_sigma=config["detection_sigma"],
        max_envelope_width_sigma=config["max_envelope_width_sigma"],
    )
    corrected = np.asarray(result["proposed_measurements"])
    residuals = [(mixed - absent) / sigma, (mixed - present) / sigma, (corrected - present) / sigma]
    titles = ["Against the HIF-absent prediction: physical effect plus meter error",
              "Against the fitted HIF-present prediction: meter error remains",
              "After repairing that meter: physical HIF retained in the prediction"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    colors = ["#b45309", "#1d4ed8", "#047857"]
    for ax, values, title, color in zip(axes, residuals, titles, colors):
        ax.plot(values, color=color, linewidth=1.3)
        ax.scatter([index], [values[index]], color="#dc2626", zorder=4, s=30)
        ax.axhline(0, color="#64748b", linewidth=0.7)
        for limit in (-config["detection_sigma"], config["detection_sigma"]):
            ax.axhline(limit, color="#94a3b8", linestyle="--", linewidth=0.9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Discrepancy / sensor sigma")
        ax.grid(axis="y", alpha=0.17)
        ax.set_xlim(-1, 122)
        for boundary in (14, 28, 42, 62, 82, 102):
            ax.axvline(boundary - 0.5, color="#cbd5e1", linewidth=0.6)
    axes[-1].set_xlabel("Original external measurement index (0–121); red dot = corrupted meter")
    fig.suptitle(f"HIF + measurement error on the same channel ({index})\n"
                 "Synthetic OpenDSS stress case; independently fitted history, held-out snapshot",
                 fontsize=14, fontweight="bold")
    fig.savefig(folder / "same_channel_recovery.png", dpi=180)
    fig.savefig(folder / "same_channel_recovery.svg")
    evidence = {"root_key": root["root_key"], "source_id": root["source_id"], "measurement_index": index,
                "bias_sigma": config["bias_sigma"], "max_hif_effect_sigma": float(np.max(np.abs(effect / sigma))),
                "candidate_indices": result["candidate_indices"], "recovery_supported": result["recovery_supported"],
                "score_semantics": "measurement-standardized prediction discrepancy; not WLS normalized residual"}
    (folder / "same_channel_recovery.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
