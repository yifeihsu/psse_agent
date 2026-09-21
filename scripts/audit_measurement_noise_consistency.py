#!/usr/bin/env python3
"""Separate HIF statistic roles and calibrate a declared noise configuration.

This audit does not relabel legacy data as consistent or modify any corpus.
The declared-channel examples come from source inspection; empirical checks
and Monte Carlo controls are reported separately from those declarations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psse_env.noise_contract import validate_noise_channel
from scripts.evaluate_hif_measurement_recovery import wls_metrics


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "minimum": float(a.min()), "median": float(np.median(a)),
            "maximum": float(a.max()), "mean": float(a.mean()), "std": float(a.std())} if a.size else {"count": 0}


def artifact_audit(folder: Path):
    roots = json.loads((folder / "root_results.json").read_text())
    traces = [json.loads(line) for p in sorted((folder / "roots").glob("*/traces.jsonl"))
              for line in p.read_text().splitlines() if line.strip()]
    result = {}
    for key, role in (("event_absent", "noiseless_model_compatibility"),
                      ("compensated", "noisy_event_only_compensated")):
        result[role] = stats([r["wls"][key]["chi_square_statistic"] for r in roots if "wls" in r])
    by_root = {r["root_key"]: r for r in roots}
    for mode in ("transient", "persistent"):
        selected = [t for t in traces if t["mode"] == mode and "conditioned_wls" in t]
        result[f"actual_noisy_postrepair_{mode}"] = stats([
            t["conditioned_wls"]["after_meter_repair"]["chi_square_statistic"] for t in selected])
        result[f"changed_channel_count_{mode}"] = sorted({len(t["physical_model"]["changed_indices"]) for t in selected})
        if mode == "transient":
            result["postrepair_minus_noisy_event_control_J"] = stats([
                t["conditioned_wls"]["after_meter_repair"]["chi_square_statistic"]
                - by_root[t["root_key"]]["wls"]["compensated"]["chi_square_statistic"] for t in selected])
    result["nominal_dof"] = 95
    result["postrepair_nominal_chi_square_distribution_verified"] = False
    return result


def declarations():
    cases = [
        ("balanced_SCADA_voltage", .001, .001, "scalar", "gaussian", "per_component"),
        ("balanced_SCADA_power", .01, .01, "scalar", "gaussian", "per_component"),
        ("legacy_HIF_phase_voltage", 0., .005, "complex_rectangular", "none", "per_component"),
        ("legacy_unbalance_SCADA_power", 0., .01, "scalar", "none", "per_component"),
        ("topology_bus3_summed_injection", np.sqrt(2) * .01, .01, "scalar", "gaussian", "per_component"),
        ("harmonic_voltage_phasor", 1e-4, 1e-4, "complex_rectangular", "gaussian", "complex_rms"),
        ("HIF_stress_phase_voltage", .005, .005, "complex_rectangular", "gaussian", "per_component"),
    ]
    receipts = {}
    for name, applied, assumed, representation, distribution, semantics in cases:
        kw = dict(channel=name, role="sensor_observation", distribution=distribution,
                  applied_sigma=applied, estimator_sigma=assumed, representation=representation,
                  applied_sigma_semantics=semantics, estimator_sigma_semantics="per_component")
        receipt = validate_noise_channel(**kw)
        try:
            validate_noise_channel(**kw, require_matched_gaussian=True)
            receipt["strict_admission"] = "passed"
        except ValueError as exc:
            receipt["strict_admission"] = "rejected"
            receipt["strict_rejection"] = str(exc)
        receipts[name] = receipt
    return receipts


def empirical_hif_scada(path):
    blocks = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        sigma = np.asarray(row["sigma_z"])
        for scan in row["scans"]:
            blocks.append((np.asarray(scan["z_obs"]) - scan["z_clean"]) / sigma)
    matrix = np.vstack(blocks)
    return {"source": str(path), "paired_reference": "same_state_HIF_present_scan_z_clean",
            "standardized_voltage_noise": stats(matrix[:, :14].ravel()),
            "standardized_power_noise": stats(matrix[:, 14:].ravel()),
            "standardized_all_noise": stats(matrix.ravel())}


def monte_carlo(baseline, sigma, draws, seed):
    rng = np.random.default_rng(seed)
    args = SimpleNamespace(chi_square_alpha=.05, detection_sigma=5.)
    samples = {"matched_gaussian": [], "half_actual_sigma_nominal_weights": [], "one_channel_replaced_by_model_mean": []}
    baseline_probe = wls_metrics(baseline, sigma, args, input_role="noiseless_model_prediction")
    for _ in range(draws):
        epsilon = rng.normal(size=len(sigma)) * sigma
        observed = baseline + epsilon
        replaced = observed.copy()
        replaced[13] = baseline[13]
        for name, value, role in (
            ("matched_gaussian", observed, "noisy_sensor_snapshot"),
            ("half_actual_sigma_nominal_weights", baseline + .5 * epsilon, "noisy_sensor_snapshot"),
            ("one_channel_replaced_by_model_mean", replaced, "noisy_postrepair_snapshot"),
        ):
            samples[name].append(wls_metrics(value, sigma, args, input_role=role))
    result = {"seed": seed, "draws_per_noisy_condition": draws, "unfiltered_draws": True,
              "same_epsilon_paired_across_conditions": True, "nominal_dof": 95,
              "matched_gaussian_expected_J_approximately": 95.,
              "half_sigma_expected_J_approximately": .25 * 95.,
              "noiseless_model_compatibility": baseline_probe}
    for name, rows in samples.items():
        result[name] = {"J": stats([r["chi_square_statistic"] for r in rows]),
                        "converged_count": sum(r["converged"] for r in rows),
                        "global_alarm_count": sum(r["chi_square_alarm"] for r in rows),
                        "normalized_alarm_count": sum(r["normalized_residual_alarm"] for r in rows),
                        "either_alarm_count": sum(not r["no_material_anomaly"] for r in rows)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/noise_consistency_20260916/audit.json")
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    experiment = ROOT / "output/hif_measurement_recovery_20260916"
    roots = json.loads((experiment / "development_17/root_results.json").read_text())
    replay = json.loads((experiment / "development_17/roots" / roots[0]["root_key"] / "replay.json").read_text())
    sigma = np.r_[np.full(14, .001), np.full(108, .01)]
    result = {
        "repository": str(ROOT),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "historical_cross_family_consistency_guaranteed": False,
        "declaration_checks_do_not_certify_source_sampling": True,
        "artifacts": {name: artifact_audit(experiment / name) for name in ("development_17", "stress_6")},
        "source_inspection_declarations": declarations(),
        "declaration_examples_role": "historical_pre_alignment_mismatches_and_positive_controls_not_current_generator_status",
        "empirical_SCADA": {
            "original_HIF": empirical_hif_scada(ROOT / "artifacts/measurements/hif_multiscan_currents_17x10_20260903/samples.jsonl"),
            "HIF_stress": empirical_hif_scada(experiment / "stress_corpus/samples.jsonl"),
        },
        "monte_carlo": monte_carlo(np.asarray(replay["predicted_base_measurements"]), sigma, args.draws, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "artifacts": result["artifacts"], "monte_carlo": result["monte_carlo"]}, indent=2))


if __name__ == "__main__":
    main()
