#!/usr/bin/env python3
"""Exploratory HIF-conditioned meter recovery, with held-out snapshot audits.

This is a research probe, not an environment success-contract override. Runtime
functions receive only whitelisted observations. Clean arrays and labels enter
the offline audit after decisions have been made. Every attempted root remains
in the denominator; repeated meter placements are not independent roots.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from three_phase_nlm.branch_current_analysis import terminal_current_hif_localization_multiscan
from three_phase_nlm.conditioned_meter_recovery import (
    diagnose_conditioned_meter_errors,
    diagnose_nonoverlap_meter_errors,
)
from three_phase_nlm.hif_conditioned_recovery import replay_hif_measurement_effect
from three_phase_nlm.hif_multiscan_estimator import estimate_hif_location_magnitude_multiscan
from psse_env.noise_contract import validate_shared_scada_covariance
from three_phase_nlm.measurement_noise import align_legacy_waveform_row, generated_noise_contract


NOISE_PREPARATION_VERSION = "matched_hif_sensor_noise_v1"


OBSERVABLE_SCAN_KEYS = (
    "measurement_convention",
    "scan_index", "z_obs", "three_phase_voltages", "three_phase_branch_currents",
    "branch_current_sigma_pu", "three_phase_sigma", "op_point", "topology_id", "sigma_z",
)


def observable_scan(scan: Mapping[str, Any]) -> dict[str, Any]:
    """Never send z_clean, clean currents, or source labels to the estimator."""
    return {key: copy.deepcopy(scan[key]) for key in OBSERVABLE_SCAN_KEYS if key in scan}


def prepare_trial_row(
    row: Mapping[str, Any], *, source_metadata: Mapping[str, Any] | None, seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare matched noisy observations once before any bad-meter trial.

    The independent preparation stream depends on observable source identity,
    never on clean references or source labels. Existing explicitly contracted
    draws are preserved. The known stress-generator metadata can document its
    older already-noisy schema without drawing a second realization.
    """
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("noise preparation seed must be a nonnegative integer")
    observable_source = {"id": row.get("id"), "scans": [observable_scan(scan) for scan in row.get("scans", [])]}
    source_digest = hashlib.sha256(json.dumps(jsonable(observable_source), sort_keys=True, allow_nan=False).encode()).hexdigest()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *[int(source_digest[i:i + 8], 16) for i in range(0, 32, 8)]]))
    prepared = align_legacy_waveform_row(row, "hif", rng, legacy_metadata=source_metadata)
    provenance = "existing_noise_contract" if row.get("noise_contract") else "known_legacy_noise_alignment"
    if not prepared.get("noise_contract"):
        metadata = source_metadata if isinstance(source_metadata, Mapping) else {}
        noise = metadata.get("noise") or {}
        if not (
            metadata.get("corpus_kind") == "synthetic_matched_model_stress"
            and metadata.get("generator") == "scripts/build_hif_recovery_stress.py"
            and noise.get("scada") == "independent zero-mean Gaussian using original 122-channel sigma_z"
            and noise.get("voltage_phasors") == "independent Gaussian per real/imag component, then convert to original magnitude/angle format"
            and noise.get("branch_currents") == "existing add_branch_current_noise helper; original per-real/imag component sigma retained (normally .001 pu)"
            and float(noise.get("three_phase_sigma_pu", -1)) == float(prepared["three_phase_sigma"])
        ):
            raise ValueError("explicit waveform weights require an applied-noise contract or known stress-generator metadata")
        contract = generated_noise_contract(
            prepared["sigma_z"], noise_scale=1.0,
            three_phase_sigma=prepared["three_phase_sigma"],
            branch_current_sigma_pu=prepared.get("branch_current_sigma_pu"),
        )
        prepared["noise_contract"] = contract
        for scan in prepared["scans"]:
            scan["noise_contract"] = copy.deepcopy(contract)
        provenance = "known_stress_noise_metadata_preserved_without_redraw"
    # Revalidate the effective applied/estimator declarations after any migration.
    prepared = align_legacy_waveform_row(prepared, "hif", rng)
    scans = [observable_scan(scan) for scan in prepared["scans"]]
    validate_shared_scada_covariance(prepared["sigma_z"], scans)
    prepared_digest = hashlib.sha256(json.dumps(jsonable(scans), sort_keys=True, allow_nan=False).encode()).hexdigest()
    return prepared, {
        "version": NOISE_PREPARATION_VERSION, "seed": int(seed), "provenance": provenance,
        "observable_source_sha256": source_digest, "prepared_observable_scans_sha256": prepared_digest,
        "noise_alignment": copy.deepcopy(prepared.get("noise_alignment")),
        "noise_contract": copy.deepcopy(prepared["noise_contract"]),
        "scan_count": len(scans), "runtime_fields": list(OBSERVABLE_SCAN_KEYS),
        "source_labels_and_clean_references_used_for_preparation_rng": False,
        "statistical_scope": "matched noise declarations and source preparation; not recovery accuracy or false-alarm calibration",
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_history(history: list[dict[str, Any]], sigma: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    validate_shared_scada_covariance(sigma, history)
    localization = terminal_current_hif_localization_multiscan(
        history, sigma_pu=float(history[0].get("branch_current_sigma_pu", 1e-3))
    )
    if not localization or not localization.get("differential_detected"):
        return {"success": False, "error": "no_observable_hif_detection", "localization": localization}
    branch = int(localization["top_hif_groups"][0]["branch_row0"])
    fit = estimate_hif_location_magnitude_multiscan(
        candidate_branch_row0=branch, scans=history, sigma_z=sigma.tolist(),
        alpha_grid_size=args.alpha_grid_size, r_grid_size=args.r_grid_size,
        max_scans=len(history), scan_selection="all", resistance_mode="shared",
        refine_top_n=2, local_max_nfev=25, workers=args.workers,
        **{key: getattr(args, key, None) for key in ("r_hif_pu_min", "r_hif_pu_max", "r_hif_ohm_min", "r_hif_ohm_max")},
        pristine_model_dir=str(REPO_ROOT / "IEEE_14_OpenDSS"),
    )
    fit["observable_localization"] = localization
    return fit


def prediction(fit: dict[str, Any], target: dict[str, Any], snapshot_id: str) -> dict[str, Any]:
    replay = replay_hif_measurement_effect(
        fit, op_point=target["op_point"], scan_index=int(target["scan_index"]),
        snapshot_id=snapshot_id, pristine_model_dir=str(REPO_ROOT / "IEEE_14_OpenDSS"),
    )
    # Sample the profile rectangle corners plus optimum. This sensitivity
    # sample is neither a guaranteed nonlinear bound nor a confidence interval.
    uncertainty = fit.get("uncertainty", {})
    alphas = uncertainty.get("near_best_alpha_interval", [fit["estimated"]["alpha_from_from_bus"]])
    resistances = uncertainty.get("near_best_r_hif_pu_interval", [fit["estimated"]["r_hif_pu"]])
    vectors = [np.asarray(replay["predicted_hif_measurements"], dtype=float)]
    for alpha in sorted(set(alphas)):
        for resistance in sorted(set(resistances)):
            varied = copy.deepcopy(fit)
            varied["estimated"].update(alpha_from_from_bus=float(alpha), r_hif_pu=float(resistance))
            candidate = replay_hif_measurement_effect(
                varied, op_point=target["op_point"], scan_index=int(target["scan_index"]),
                snapshot_id=snapshot_id, pristine_model_dir=str(REPO_ROOT / "IEEE_14_OpenDSS"),
            )
            vectors.append(np.asarray(candidate["predicted_hif_measurements"], dtype=float))
    replay["prediction_lower"] = np.min(vectors, axis=0).tolist()
    replay["prediction_upper"] = np.max(vectors, axis=0).tolist()
    replay["envelope_method"] = "near_best_profile_rectangle_sensitivity_not_confidence_interval"
    return replay


def diagnosis_audit(fit: Mapping[str, Any], label: Mapping[str, Any]) -> dict[str, Any]:
    from three_phase_nlm.hif_units import label_physical_ohm, label_local_kv_ll
    estimated = fit.get("estimated", {})
    if not fit.get("success") or not estimated:
        return {"correct": False, "error": fit.get("error", "fit_failed")}
    alpha_error = abs(float(estimated["alpha_from_from_bus"]) - float(label["split_ratio"]))
    resistance_error = abs(float(estimated["r_hif_pu"]) / float(label["r_hif_pu"]) - 1)
    branch_correct = int(fit["candidate_branch_row0"]) == int(label["branch_row0"])
    phase_correct = str(estimated["phase"]).upper() == str(label["phase"]).upper()
    physical_ohm, local_kv = label_physical_ohm(label), label_local_kv_ll(label)
    return {
        "correct": bool(branch_correct and phase_correct and alpha_error <= 0.15 and resistance_error <= 0.20),
        "truth_physical_ohm": physical_ohm, "truth_local_kv_ll": local_kv,
        "truth_nominal_p_hif_kw": ((local_kv*1000)**2 / (3*physical_ohm) / 1000 if physical_ohm and local_kv else None),
        "branch_correct": branch_correct, "phase_correct": phase_correct,
        "alpha_absolute_error": alpha_error, "resistance_relative_error": resistance_error,
        "alpha_tolerance": 0.15, "resistance_relative_tolerance": 0.20,
        "parameter_identifiable": bool(fit.get("parameter_identifiable")),
    }


def audit_decision(decision: Mapping[str, Any], active: np.ndarray, original: np.ndarray,
                   noiseless: np.ndarray, sigma: np.ndarray, index: int | None) -> dict[str, Any]:
    proposed = np.asarray(decision["proposed_measurements"], dtype=float)
    changed = np.flatnonzero(proposed != active).tolist()
    target_set = [] if index is None else [index]
    exact_support = changed == target_set
    noise_error = None if index is None else abs(float((proposed[index] - noiseless[index]) / sigma[index]))
    original_error = None if index is None else abs(float((proposed[index] - original[index]) / sigma[index]))
    recovered = bool(decision["recovery_supported"] and exact_support and (noise_error is None or noise_error <= 3.0))
    return {
        "success": recovered, "changed_indices": changed, "exact_write_support": exact_support,
        "target_error_to_noiseless_sigma": noise_error, "target_error_to_preinjection_sigma": original_error,
        "off_target_write_count": len(set(changed) - set(target_set)),
        "failure_reasons": list(decision.get("failure_reasons", [])),
        "candidate_indices": decision["candidate_indices"],
        "recovery_supported": bool(decision["recovery_supported"]),
    }


def run_decisions(active: np.ndarray, original: np.ndarray, noiseless: np.ndarray, sigma: np.ndarray,
                  replay: Mapping[str, Any], index: int | None, args: argparse.Namespace) -> dict[str, Any]:
    effect = np.asarray(replay["measurement_effect"])
    physical = diagnose_conditioned_meter_errors(
        active, replay["predicted_hif_measurements"], sigma,
        prediction_lower=replay["prediction_lower"], prediction_upper=replay["prediction_upper"],
        event_effect=effect, detection_sigma=args.detection_sigma,
        max_envelope_width_sigma=args.max_envelope_width_sigma,
    )
    fallback = diagnose_nonoverlap_meter_errors(
        active, replay["predicted_base_measurements"], sigma, event_effect=effect,
        detection_sigma=args.detection_sigma, support_sigma=args.support_sigma,
    )
    physical_audit = audit_decision(physical, active, original, noiseless, sigma, index)
    fallback_audit = audit_decision(fallback, active, original, noiseless, sigma, index)
    absent_only = diagnose_conditioned_meter_errors(
        active, replay["predicted_base_measurements"], sigma,
        detection_sigma=args.detection_sigma,
    )
    absent_audit = audit_decision(absent_only, active, original, noiseless, sigma, index)
    # Truth used ONLY here, after both methods produced their decisions.
    true_effect = noiseless - np.asarray(replay["predicted_base_measurements"])
    overlap = bool(index is not None and abs(true_effect[index] / sigma[index]) >= args.support_sigma)
    if overlap:
        fallback_audit["success"] = False
        fallback_audit["failure_reasons"].append("offline_audit_true_hif_meter_overlap")
    return {"true_overlap": overlap, "physical_model": physical_audit,
            "nonoverlap_fallback": fallback_audit, "absent_model_ablation": absent_audit}


def wls_metrics(z: np.ndarray, sigma: np.ndarray, args: argparse.Namespace,
                *, exported_injections: bool = True,
                input_role: str = "noisy_sensor_snapshot") -> dict[str, Any]:
    # Use the actual corpus variances, not a provider's unrelated default profile.
    from pypower.api import case14
    from tools.lagrangian_port import lagrangian_m_singlephase_details
    interpretations = {
        "noiseless_model_prediction": "deterministic_model_compatibility_only_not_postrepair_or_false_alarm_evidence",
        "noisy_sensor_snapshot": "noisy_snapshot_diagnostic_with_declared_sensor_covariance",
        "noisy_compensated_snapshot": "fitted_effect_uncertainty_not_in_sensor_covariance_no_calibrated_p_value",
        "noisy_postrepair_snapshot": "replacement_and_fitted_effect_uncertainty_not_calibrated_no_chi_square_p_value",
    }
    if input_role not in interpretations:
        raise ValueError(f"Unknown WLS input role: {input_role}")
    case = case14()
    # DSS exported injections include capacitor supply: they are net injections
    # into branches. Omit shunts only from this *measurement operator*. The
    # physical replay retains all capacitors. Do not transform a noisy Vm datum
    # into Qinj, which would spread a Vm meter fault into another channel.
    if exported_injections:
        case["bus"][:, 4:6] = 0.0
    try:
        details = lagrangian_m_singlephase_details(
            z, case, 0, case["bus"], measurement_sigma=sigma,
        )
        dof = int(details["dof"])
        statistic = float(details["wls_objective"])
        local = float(np.max(details["r_norm"]))
        threshold = float(chi2.ppf(1 - args.chi_square_alpha, dof))
        return {
            "input_role": input_role, "statistical_interpretation": interpretations[input_role],
            "converged": bool(details["success"]), "chi_square_statistic": statistic,
            "chi_square_threshold": threshold, "chi_square_dof": dof,
            "chi_square_alpha": args.chi_square_alpha, "chi_square_alarm": statistic >= threshold,
            "max_normalized_residual": local, "normalized_residual_threshold": args.detection_sigma,
            "normalized_residual_alarm": local >= args.detection_sigma,
            "anomaly_rule": "chi_square_or_normalized_residual",
            "measurement_convention": "branch_net_injections_including_capacitors" if exported_injections else "canonical_matpower_injections",
            "no_material_anomaly": bool(details["success"] and statistic < threshold and local < args.detection_sigma),
        }
    except Exception as exc:
        return {"converged": False, "error": f"{type(exc).__name__}: {exc}", "no_material_anomaly": False,
                "input_role": input_role, "statistical_interpretation": interpretations[input_role]}


def continuation_wls(active: np.ndarray, replay: Mapping[str, Any], decision_audit: Mapping[str, Any],
                     sigma: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    effect = np.asarray(replay["measurement_effect"])
    corrected = active.copy()
    center = np.asarray(replay["predicted_hif_measurements"])
    for index in decision_audit["changed_indices"]:
        corrected[index] = center[index]
    return {"before_meter_repair": wls_metrics(active - effect, sigma, args, input_role="noisy_compensated_snapshot"),
            "after_meter_repair": wls_metrics(corrected - effect, sigma, args, input_role="noisy_postrepair_snapshot")}


def summarize(roots: list[dict[str, Any]], traces: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    attempted = len(roots)
    subsets: dict[str, list[dict[str, Any]]] = {
        "all_transient": [row for row in traces if row["mode"] == "transient"],
        "transient_overlap": [row for row in traces if row["mode"] == "transient" and row["true_overlap"]],
        "transient_nonoverlap": [row for row in traces if row["mode"] == "transient" and not row["true_overlap"]],
        "persistent": [row for row in traces if row["mode"] == "persistent"],
    }
    results = {}
    for name, rows in subsets.items():
        results[name] = {
            "trace_count": len(rows), "unique_root_count": len({row["root_key"] for row in rows}),
            "physical_model_meter_recovery": sum(row["physical_model"]["success"] for row in rows),
            "physical_model_joint_success": sum(row["physical_model"]["success"] and row["diagnosis_correct"] for row in rows),
            "fallback_meter_recovery": sum(row["nonoverlap_fallback"]["success"] for row in rows),
            "fallback_joint_success": sum(row["nonoverlap_fallback"]["success"] and row["diagnosis_correct"] for row in rows),
            "off_target_writes": sum(row["physical_model"]["off_target_write_count"] for row in rows),
            "absent_model_meter_recovery": sum(row.get("absent_model_ablation", {}).get("success", False) for row in rows),
            "absent_model_off_target_writes": sum(row.get("absent_model_ablation", {}).get("off_target_write_count", 0) for row in rows),
            "wls_sampled_trace_count": sum("conditioned_wls" in row for row in rows),
            "wls_quiet_after_repair_count": sum(row.get("conditioned_wls", {}).get("after_meter_repair", {}).get("no_material_anomaly", False) for row in rows),
        }
    successful_roots = [root for root in roots if root.get("fit_success")]
    return {
        "status": "exploratory_physical_model_pilot", "config": config,
        "attempted_root_count": attempted, "fitted_root_count": len(successful_roots),
        "failed_root_count": attempted - len(successful_roots),
        "expected_transient_trace_count": attempted * 122 * 2,
        "unexecuted_transient_traces_counted_as_failure": attempted * 122 * 2 - len(subsets["all_transient"]),
        "expected_persistent_trace_count": attempted * int(config["persistent_targets"]),
        "unexecuted_persistent_traces_counted_as_failure": attempted * int(config["persistent_targets"]) - len(subsets["persistent"]),
        "diagnosis_correct_root_count": sum(root.get("diagnosis_audit", {}).get("correct", False) for root in roots),
        "event_only_evaluated_root_count": sum("event_only" in root for root in roots),
        "event_only_false_alarm_root_count": sum(not root["event_only"]["physical_model"]["success"] for root in roots if "event_only" in root),
        "event_absent_wls_quiet_root_count": sum(root.get("wls", {}).get("event_absent", {}).get("no_material_anomaly", False) for root in roots),
        "compensated_event_only_wls_quiet_root_count": sum(root.get("wls", {}).get("compensated", {}).get("no_material_anomaly", False) for root in roots),
        "strata": results, "physical_fault_removed": False,
        "limits": [
            "Known operating-point metadata and the same OpenDSS model family as synthesis.",
            "Gross meter biases target 122 SCADA channels; auxiliary phase-voltage/current sensors carry declared Gaussian noise.",
            "History snapshots exclude active target snapshot; shared HIF parameters persist across the window.",
            "Transient placements reuse a root fit and are dependent trials, not independent roots.",
            "Sensitivity envelope is not a calibrated confidence interval; no inferred covariance is claimed.",
            "Conventional compensated WLS statistics omit fitted-effect uncertainty and are diagnostic probes only.",
            "Direct physical-model prediction checks use measurement-standardized residuals, not WLS normalized residuals.",
            "No agent rollout, DAgger training, production lifecycle, or release-audit success override is enabled.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=REPO_ROOT / "artifacts/measurements/hif_multiscan_currents_17x10_20260903/samples.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--history-scans", type=int, default=4)
    parser.add_argument("--alpha-grid-size", type=int, default=7)
    parser.add_argument("--r-grid-size", type=int, default=9)
    parser.add_argument("--workers", type=int, default=1)
    for unit in ("pu", "ohm"):
        for bound in ("min", "max"):
            parser.add_argument(f"--r-hif-{unit}-{bound}", type=float)
    parser.add_argument("--persistent-targets", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--bias-sigma", type=float, default=10.0)
    parser.add_argument("--detection-sigma", type=float, default=5.0)
    parser.add_argument("--support-sigma", type=float, default=1.0)
    parser.add_argument("--max-envelope-width-sigma", type=float, default=2.0)
    parser.add_argument("--chi-square-alpha", type=float, default=0.05)
    parser.add_argument("--noise-preparation-seed", type=int, default=20260916)
    parser.add_argument("--reuse-fits", action="store_true", help="Re-evaluate saved fits/replays without rerunning expensive fits.")
    args = parser.parse_args()
    if not 1 <= args.history_scans <= 9:
        parser.error("--history-scans must be in [1, 9]")
    if args.noise_preparation_seed < 0:
        parser.error("--noise-preparation-seed must be nonnegative")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "reuse_fits"}
    source_metadata_path = args.samples.with_name("meta.json")
    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else None
    source_metadata_sha256 = hashlib.sha256(source_metadata_path.read_bytes()).hexdigest() if source_metadata_path.exists() else None
    config.update(
        source_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        repository=str(REPO_ROOT), runtime_truth_fields="excluded_by_whitelist",
        meter_accuracy_sigma=3.0, thresholds="predeclared exploratory; not tuned by root outcomes",
        noise_preparation_version=NOISE_PREPARATION_VERSION,
        source_metadata_sha256=source_metadata_sha256,
    )
    config_path = args.output_dir / "config.json"
    original_config = json.loads(config_path.read_text()) if config_path.exists() else None
    if original_config is None and any(args.output_dir.iterdir()):
        raise ValueError("Nonempty output directory has no matching configuration; use a new output directory")
    if original_config is not None:
        previous, current = dict(original_config), dict(config)
        if args.reuse_fits:
            # Reanalysis may add diagnostics at a newer code revision. Keep
            # fit provenance separately; still reject changed data/settings.
            previous.pop("git_head", None)
            current.pop("git_head", None)
        if previous != current:
            raise ValueError("Output experimental configuration differs; use a new output directory")
    if args.reuse_fits and original_config is not None:
        config["fit_generation_git_head"] = original_config["git_head"]
        write_json(args.output_dir / "reanalysis_config.json", config)
    else:
        write_json(config_path, config)
    all_roots: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    for ordinal, line in enumerate(args.samples.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if args.limit and len(all_roots) >= args.limit:
            break
        row = json.loads(line)
        source_root_key = hashlib.sha256(line.encode()).hexdigest()[:16]
        root_identity = f"{NOISE_PREPARATION_VERSION}:{args.noise_preparation_seed}:{source_metadata_sha256}:{source_root_key}"
        root_key = hashlib.sha256(root_identity.encode()).hexdigest()[:16]
        root_dir = args.output_dir / "roots" / root_key
        root_dir.mkdir(parents=True, exist_ok=True)
        result_path = root_dir / "result.json"
        traces_path = root_dir / "traces.jsonl"
        if result_path.exists() and traces_path.exists() and not args.reuse_fits:
            root = json.loads(result_path.read_text())
            trace_rows = [json.loads(item) for item in traces_path.read_text().splitlines()]
            all_roots.append(root)
            all_traces.extend(trace_rows)
            print(json.dumps({"root": row["id"], "status": "resumed_completed"}), flush=True)
            continue
        started = time.monotonic()
        root = {"root_key": root_key, "source_root_key": source_root_key, "source_id": row["id"],
                "source_ordinal": ordinal, "fit_success": False,
                "noise_preparation_version": NOISE_PREPARATION_VERSION,
                "noise_preparation_seed": args.noise_preparation_seed}
        trace_rows = []
        try:
            row, preparation = prepare_trial_row(row, source_metadata=source_metadata, seed=args.noise_preparation_seed)
            write_json(root_dir / "noise_preparation.json", preparation)
            write_json(root_dir / "prepared_observable_scans.json", [observable_scan(scan) for scan in row["scans"]])
            raw_target = row["scans"][0]
            target = observable_scan(raw_target)
            history = [observable_scan(scan) for scan in row["scans"][1:1 + args.history_scans]]
            if len(history) != args.history_scans:
                raise ValueError("insufficient_disjoint_history")
            original = np.asarray(target["z_obs"], dtype=float)
            noiseless = np.asarray(raw_target["z_clean"], dtype=float)  # Offline audit only.
            sigma = np.asarray(row["sigma_z"], dtype=float)
            covariance_receipt = validate_shared_scada_covariance(sigma, [target, *history])
            write_json(root_dir / "scada_covariance_contract.json", covariance_receipt)
            fit_path = root_dir / "fit.json"
            fit = json.loads(fit_path.read_text()) if args.reuse_fits and fit_path.exists() else fit_history(history, sigma, args)
            write_json(root_dir / "fit.json", fit)
            if not fit.get("success"):
                raise ValueError(fit.get("error", "fit_failed"))
            replay_path = root_dir / "replay.json"
            replay = json.loads(replay_path.read_text()) if args.reuse_fits and replay_path.exists() else prediction(fit, target, root_key)
            write_json(root_dir / "replay.json", replay)
            diag_audit = diagnosis_audit(fit, row.get("shared_label", row["label"]))
            root.update(fit_success=True, diagnosis_audit=diag_audit,
                        history_scan_indices=[scan["scan_index"] for scan in history],
                        target_scan_index=target["scan_index"])
            center = np.asarray(replay["predicted_hif_measurements"])
            base = np.asarray(replay["predicted_base_measurements"])
            effect = np.asarray(replay["measurement_effect"])
            model_error = (center - noiseless) / sigma
            root["prediction_audit"] = {
                "rms_error_to_noiseless_sigma": float(np.sqrt(np.mean(model_error**2))),
                "max_error_to_noiseless_sigma": float(np.max(np.abs(model_error))),
                "max_event_effect_sigma": float(np.max(np.abs(effect / sigma))),
                "true_overlap_channel_count": int(np.sum(np.abs((noiseless - base) / sigma) >= args.support_sigma)),
                "max_envelope_width_sigma": float(np.max((np.asarray(replay["prediction_upper"]) - replay["prediction_lower"]) / sigma)),
            }
            root["event_only"] = run_decisions(original, original, noiseless, sigma, replay, None, args)
            # Direct inverse-variance innovation probe, no WLS state/parameter df assertion.
            innovation = (original - center) / sigma
            root["event_only_innovation"] = {"sum_squared_standardized": float(innovation @ innovation),
                                             "max_abs_standardized": float(np.max(np.abs(innovation)))}
            root["wls"] = {"raw": wls_metrics(original, sigma, args),
                           "compensated": wls_metrics(original - effect, sigma, args, input_role="noisy_compensated_snapshot"),
                           "event_absent": wls_metrics(base, sigma, args, input_role="noiseless_model_prediction"),
                           "unadapted_event_absent": wls_metrics(base, sigma, args, exported_injections=False, input_role="noiseless_model_prediction")}
            wls_indices = list(dict.fromkeys([int(np.argmax(np.abs(effect / sigma))), int(np.argmin(np.abs(effect / sigma)))]))
            for index in range(len(original)):
                for sign in (-1, 1):
                    active = original.copy()
                    active[index] += sign * args.bias_sigma * sigma[index]
                    result = run_decisions(active, original, noiseless, sigma, replay, index, args)
                    if index in wls_indices:
                        result["conditioned_wls"] = continuation_wls(active, replay, result["physical_model"], sigma, args)
                    trace_rows.append({"root_key": root_key, "mode": "transient", "measurement_index": index,
                                       "bias_sigma": sign * args.bias_sigma, "diagnosis_correct": diag_audit["correct"], **result})
            persistent_indices = list(dict.fromkeys([int(np.argmax(np.abs(effect / sigma))), int(np.argmin(np.abs(effect / sigma)))]))[:args.persistent_targets]
            for index in persistent_indices:
                try:
                    mixed_history = copy.deepcopy(history)
                    for scan in mixed_history:
                        scan["z_obs"][index] += args.bias_sigma * sigma[index]
                    active = original.copy()
                    active[index] += args.bias_sigma * sigma[index]
                    persistent_fit_path = root_dir / f"persistent_{index}_fit.json"
                    persistent_fit = json.loads(persistent_fit_path.read_text()) if args.reuse_fits and persistent_fit_path.exists() else fit_history(mixed_history, sigma, args)
                    write_json(root_dir / f"persistent_{index}_fit.json", persistent_fit)
                    if not persistent_fit.get("success"):
                        raise ValueError(f"persistent_fit_failed:{index}:{persistent_fit.get('error', 'fit_failed')}")
                    persistent_replay_path = root_dir / f"persistent_{index}_replay.json"
                    persistent_replay = json.loads(persistent_replay_path.read_text()) if args.reuse_fits and persistent_replay_path.exists() else prediction(persistent_fit, target, f"{root_key}:persistent:{index}")
                    write_json(root_dir / f"persistent_{index}_replay.json", persistent_replay)
                    persistent_audit = diagnosis_audit(persistent_fit, row.get("shared_label", row["label"]))
                    result = run_decisions(active, original, noiseless, sigma, persistent_replay, index, args)
                    result["conditioned_wls"] = continuation_wls(active, persistent_replay, result["physical_model"], sigma, args)
                    trace_rows.append({"root_key": root_key, "mode": "persistent", "measurement_index": index,
                                       "bias_sigma": args.bias_sigma, "diagnosis_correct": persistent_audit["correct"],
                                       "diagnosis_audit": persistent_audit, "status": "completed", **result})
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    failed_audit = {
                        "success": False, "changed_indices": [], "exact_write_support": False,
                        "target_error_to_noiseless_sigma": None,
                        "target_error_to_preinjection_sigma": None,
                        "off_target_write_count": 0, "failure_reasons": ["persistent_trial_failed", failure],
                        "candidate_indices": [], "recovery_supported": False,
                    }
                    # The event-free reference is independent of the failed fit;
                    # source truth determines this label only in the offline audit.
                    true_overlap = bool(abs((noiseless[index] - base[index]) / sigma[index]) >= args.support_sigma)
                    trace_rows.append({"root_key": root_key, "mode": "persistent", "measurement_index": index,
                                       "bias_sigma": args.bias_sigma, "diagnosis_correct": False,
                                       "status": "failed", "error": failure, "true_overlap": true_overlap,
                                       "physical_model": copy.deepcopy(failed_audit),
                                       "nonoverlap_fallback": copy.deepcopy(failed_audit)})
        except Exception as exc:
            root["error"] = f"{type(exc).__name__}: {exc}"
        root["elapsed_seconds"] = time.monotonic() - started
        traces_path.write_text("".join(json.dumps(jsonable(item), allow_nan=False) + "\n" for item in trace_rows), encoding="utf-8")
        write_json(result_path, root)
        all_roots.append(root)
        all_traces.extend(trace_rows)
        write_json(args.output_dir / "summary.json", summarize(all_roots, all_traces, config))
        print(json.dumps({"root": row["id"], "fit_success": root["fit_success"], "error": root.get("error"),
                          "diagnosis": root.get("diagnosis_audit"), "prediction": root.get("prediction_audit"),
                          "seconds": root["elapsed_seconds"]}), flush=True)
    write_json(args.output_dir / "summary.json", summarize(all_roots, all_traces, config))
    write_json(args.output_dir / "root_results.json", all_roots)


if __name__ == "__main__":
    main()
