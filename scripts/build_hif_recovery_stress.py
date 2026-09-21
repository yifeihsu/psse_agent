#!/usr/bin/env python3
"""Build a reproducible matched-model HIF stress corpus with noisy telemetry.

Source labels are used solely to specify generation parameters. All ten source
operating points are preserved, while HIF-present and HIF-absent measurements
are solved afresh in OpenDSS. This is a synthetic matched-model stress test,
not independent validation of HIF physics or a population evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from three_phase_nlm.branch_current_analysis import add_branch_current_noise
from three_phase_nlm.hif_parameter_estimator import _resolve_model_dir, _simulate_base, simulate_hif_candidate
from three_phase_nlm.measurement_noise import generated_noise_contract


def add_voltage_phasor_noise(rows: list[dict[str, Any]], rng: np.random.Generator,
                            sigma_pu: float) -> list[dict[str, Any]]:
    """Apply independent Gaussian real/imag noise in the estimator's units."""
    noisy = copy.deepcopy(rows)
    for row in noisy:
        magnitude = np.asarray(row["vln_pu"], dtype=float)
        angle = np.deg2rad(np.asarray(row["ang_deg"], dtype=float))
        if magnitude.shape != (3,) or angle.shape != (3,):
            raise ValueError("Voltage telemetry must have three phase phasors per bus")
        phasors = magnitude * np.exp(1j * angle)
        phasors += rng.normal(0, sigma_pu, 3) + 1j * rng.normal(0, sigma_pu, 3)
        row["vln_pu"] = np.abs(phasors).tolist()
        row["ang_deg"] = np.rad2deg(np.angle(phasors)).tolist()
    return noisy


def select_sources(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], float]]:
    """Select first two distinct source branches per phase, without outcome selection."""
    selected = []
    used_branches: set[int] = set()
    for phase in ("A", "B", "C"):
        matches = [
            (index, row) for index, row in enumerate(rows)
            if row.get("label", {}).get("phase") == phase
        ]
        phase_count = 0
        for index, row in matches:
            branch = int(row["label"]["branch_row0"])
            if branch in used_branches:
                continue
            selected.append((index, row, (5.0, 10.0)[phase_count]))
            used_branches.add(branch)
            phase_count += 1
            if phase_count == 2:
                break
        if phase_count != 2:
            raise ValueError(f"Need two source roots on distinct branches for phase {phase}")
    return selected


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.samples.resolve()
    output_dir = args.output_dir.resolve()
    # Resolve the existing node-local model override once so both forward
    # solves use the same files when PSSE_OPENDSS_MODEL_DIR is configured.
    model_dir = _resolve_model_dir(str(REPO_ROOT / "IEEE_14_OpenDSS"), "case14").resolve()
    source_bytes = source_path.read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    selections = select_sources(source_rows)
    if not np.isfinite(args.three_phase_sigma) or args.three_phase_sigma <= 0:
        raise ValueError("three_phase_sigma must be finite and positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if output_dir / "samples.jsonl" == source_path:
        raise ValueError("Output must not overwrite the source corpus")

    generated = []
    receipts = []
    for source_index, source, resistance in selections:
        label = copy.deepcopy(source["label"])
        branch = int(label["branch_row0"])
        phase = str(label["phase"])
        alpha = float(label["split_ratio"])
        if len(source["scans"]) != 10:
            raise ValueError(f"Source {source['id']} must have exactly ten scans")
        sigma = np.asarray(source["sigma_z"], dtype=float)
        if sigma.shape != (122,) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("Each source requires 122 finite positive measurement sigmas")
        root_id = f"hif_recovery_stress_{phase}_br{branch:02d}_r{int(resistance):02d}_src{source_index:02d}"
        scans = []
        support_by_scan = []
        for raw_scan in source["scans"]:
            scan_index = int(raw_scan["scan_index"])
            operating_point = copy.deepcopy(raw_scan["op_point"])
            current_sigma = float(raw_scan.get("branch_current_sigma_pu", source.get("branch_current_sigma_pu", 1e-3)))
            if not np.isfinite(current_sigma) or current_sigma <= 0:
                raise ValueError("Source branch-current sigma must be finite and positive")
            # The label is a generation input, never an estimator input.
            present = simulate_hif_candidate(
                candidate_branch_row0=branch, alpha=alpha, phase=phase,
                r_hif_pu=resistance, op_point=operating_point,
                pristine_model_dir=str(model_dir),
            )
            absent = _simulate_base(model_dir, op_point=operating_point)
            clean = np.asarray(present["z"], dtype=float)
            absent_clean = np.asarray(absent["z"], dtype=float)
            if clean.shape != sigma.shape or absent_clean.shape != sigma.shape:
                raise ValueError("Forward solver changed the external measurement layout")
            scada_rng, voltage_rng, current_rng = [
                np.random.default_rng(np.random.SeedSequence([args.seed, source_index, scan_index, stream]))
                for stream in (0, 1, 2)
            ]
            scan = {
                "scan_index": scan_index,
                "z_clean": clean.tolist(),
                "z_absent_clean": absent_clean.tolist(),
                "z_obs": (clean + scada_rng.normal(0, sigma)).tolist(),
                "sigma_z": sigma.tolist(),
                "three_phase_voltages": add_voltage_phasor_noise(
                    present["three_phase_voltages"], voltage_rng, args.three_phase_sigma),
                "three_phase_voltages_clean": copy.deepcopy(present["three_phase_voltages"]),
                "three_phase_sigma": args.three_phase_sigma,
                "noise_contract": generated_noise_contract(
                    sigma.tolist(), noise_scale=1.0, three_phase_sigma=args.three_phase_sigma,
                    branch_current_sigma_pu=current_sigma,
                ),
                "three_phase_branch_currents": add_branch_current_noise(
                    present["three_phase_branch_currents"], current_rng, current_sigma),
                "three_phase_branch_currents_clean": copy.deepcopy(present["three_phase_branch_currents"]),
                "branch_current_sigma_pu": current_sigma,
                "op_point": operating_point,
                "topology_id": str(raw_scan.get("topology_id", "ieee14_base")),
            }
            standardized_effect = np.abs(clean - absent_clean) / sigma
            support_by_scan.append({
                "scan_index": scan_index,
                "max_abs_effect_sigma": float(np.max(standardized_effect)),
                "support_count_ge_1sigma": int(np.sum(standardized_effect >= 1)),
                "support_count_ge_3sigma": int(np.sum(standardized_effect >= 3)),
                "support_count_ge_5sigma": int(np.sum(standardized_effect >= 5)),
                "support_indices_ge_1sigma": np.flatnonzero(standardized_effect >= 1).tolist(),
            })
            scans.append(scan)
        # Recompute descriptive physical fields when a physical source supplied them,
        # then keep the stress generator's explicit legacy model-ohm label contract.
        from three_phase_nlm.hif_units import hif_resistance_record
        if label.get("resistance_units") == "ohm_local_base":
            label.update(hif_resistance_record(branch_row0=branch, r_hif_pu=resistance))
        label.update(resistance_units="pu_legacy_normalized_model", r_hif_pu=resistance,
                     r_hif_ohm=float(present["r_hif_ohm"]), r_hif_model_ohm=float(present["r_hif_ohm"]), fault_bus="FaultEst")
        first = scans[0]
        generated.append({
            "id": root_id,
            "scenario": "high_impedance_fault",
            "case": "IEEE14",
            "label": label,
            "shared_label": copy.deepcopy(label),
            "scans": scans,
            "scan_count": len(scans),
            "sigma_z": sigma.tolist(),
            "z_obs": copy.deepcopy(first["z_obs"]),
            "z_true": copy.deepcopy(first["z_absent_clean"]),
            "three_phase_voltages": copy.deepcopy(first["three_phase_voltages"]),
            "three_phase_branch_currents": copy.deepcopy(first["three_phase_branch_currents"]),
            "branch_current_sigma_pu": first["branch_current_sigma_pu"],
            "three_phase_sigma": args.three_phase_sigma,
            "noise_contract": copy.deepcopy(first["noise_contract"]),
            "topology_id": first["topology_id"],
            "op_point": copy.deepcopy(first["op_point"]),
            "window_metadata": {
                "corpus_kind": "synthetic_matched_model_stress",
                "independent_physical_validation": False,
                "operating_point_mode": "preserved_source_diverse",
                "persistent_hif": True,
                "seed": args.seed,
                "source_index0": source_index,
                "source_id": source["id"],
                "z_true_semantics": "same-model HIF-absent reference for scan zero; hidden audit only",
                "source_label_before_stress": copy.deepcopy(source["label"]),
            },
        })
        receipts.append({
            "id": root_id, "source_index0": source_index, "source_id": source["id"],
            "branch_row0": branch, "phase": phase, "alpha_from_from_bus": alpha,
            "shared_r_hif_pu": resistance, "scan_support": support_by_scan,
        })
        print(json.dumps({"root": root_id, "scan0_support": support_by_scan[0]}), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "samples.jsonl"
    sample_text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in generated)
    sample_path.write_text(sample_text, encoding="utf-8")
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    receipt = {
        "corpus_kind": "synthetic_matched_model_stress",
        "independent_physical_validation": False,
        "purpose": "Stronger real OpenDSS HIF effects for same-channel HIF-plus-meter recovery experiments",
        "repository": str(REPO_ROOT), "git_head": git_head,
        "resolved_model_dir": str(model_dir),
        "generator": "scripts/build_hif_recovery_stress.py",
        "source_samples": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_samples": str(sample_path),
        "output_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "seed": args.seed, "root_count": len(generated), "scans_per_root": 10,
        "source_selection": "first two distinct-branch source rows in original file order for each phase A,B,C; no outcome-based selection",
        "resistance_selection": "first root per phase: shared 5 pu; second: shared 10 pu; search lower limit is 5 pu",
        "operating_points": "all ten source op_points preserved exactly",
        "generation_physics": "simulate_hif_candidate and _simulate_base at identical op_point; no additive synthetic HIF approximation",
        "noise": {
            "scada": "independent zero-mean Gaussian using original 122-channel sigma_z",
            "voltage_phasors": "independent Gaussian per real/imag component, then convert to original magnitude/angle format",
            "three_phase_sigma_pu": args.three_phase_sigma,
            "voltage_difference_from_original_generator": "original phase voltages were noise-free; this stress corpus adds noise using the estimator's declared component-sigma convention",
            "branch_currents": "existing add_branch_current_noise helper; original per-real/imag component sigma retained (normally .001 pu)",
            "streams": "NumPy SeedSequence([seed, source_index0, scan_index, sensor_stream]); streams 0 SCADA, 1 voltage, 2 current",
        },
        "hidden_audit_fields": ["label", "shared_label", "z_true", "z_clean", "z_absent_clean", "three_phase_voltages_clean", "three_phase_branch_currents_clean", "window_metadata.source_label_before_stress"],
        "estimator_input_boundary": "Use evaluate_hif_measurement_recovery.observable_scan whitelist; no labels or clean arrays enter fit",
        "measurement_error_injection": "No bad-meter biases in this base corpus; evaluator overlays declared biases on held-out or persistent observations",
        "roots": receipts,
    }
    write_json(output_dir / "generation_receipt.json", receipt)
    write_json(output_dir / "meta.json", {key: value for key, value in receipt.items() if key != "roots"})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=REPO_ROOT / "artifacts/measurements/hif_multiscan_currents_17x10_20260903/samples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/hif_measurement_recovery_20260916/stress_corpus")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--three-phase-sigma", type=float, default=5e-3)
    args = parser.parse_args()
    receipt = build(args)
    print(json.dumps({"samples": receipt["output_samples"], "roots": receipt["root_count"], "output_sha256": receipt["output_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
