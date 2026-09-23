"""Unfiltered physical IEEE14/IEEE57 HIF and paired noisy balanced-WLS audit.

This is a steady-state resistive-surrogate experiment, not a protection test.
No WLS result selects a physical case. Failed solves remain unavailable records.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

# Bound native pools before importing NumPy/SciPy when invoked as a script.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from psse_env.fault_profiles import measurement_sigma
from psse_env.systems import resolve_system
from research.gnn_screen.feature_schema import MEASUREMENT_CONVENTION
from research.gnn_screen.wls_features import build_wls_features
from three_phase_model.disturbances import audit_disturbed_circuit, eligible_hif_branch_rows, inject_midspan_hif
from three_phase_model.exporter import export_model, load_assumptions, write_json
from three_phase_model.measurements import extract_measurements
from three_phase_model.runtime import compile_model
from three_phase_model.validation import validate_model
from three_phase_model.voltage_bases import (
    IEEE14_VOLTAGE_BASE_PROFILE_ID, apply_ieee14_voltage_bases, eligible_ieee14_hif_branch_rows,
)

CONTRACT = "ieee14_physical_hif_unfiltered_sweep_v1"
RESISTANCES_OHM = (50., 100., 200., 500., 1000., 2000., 5000.)
NOISE_PROFILES = ("baseline", "accuracy_005", "accuracy_002")
CHI_SQUARE_ALPHA, NORMALIZED_RESIDUAL_THRESHOLD = .01, 4.
IMPLEMENTATION = (
    "scripts/audit_ieee14_hif_physical_sweep.py", "three_phase_model/exporter.py",
    "three_phase_model/disturbances.py", "three_phase_model/measurements.py",
    "three_phase_model/runtime.py", "three_phase_model/validation.py",
    "three_phase_model/voltage_bases.py", "psse_env/fault_profiles.py",
    "research/gnn_screen/wls_features.py", "tools/lagrangian_port.py",
)


def numeric_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _json(value) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _write_row(handle, value):
    handle.write(_json(value) + "\n")
    handle.flush()


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def standard_noise(seed, parent_ordinal, branch_row0, phase, replicate, size=122):
    """One independent group, paired across resistance, accuracy and controls."""
    entropy = [int(seed), int(parent_ordinal), int(branch_row0), int(phase), int(replicate)]
    return np.random.default_rng(np.random.SeedSequence(entropy)).normal(size=size), entropy


def rethreshold_wls(audit, chi_square_alpha):
    """Compare a second chi-square gate using the SAME fitted observation/J/NR."""
    if not math.isfinite(chi_square_alpha) or not 0 < chi_square_alpha < 1:
        raise ValueError("chi_square_alpha must be strictly between zero and one")
    if not audit.get("success"):
        return {"success": False, "alarm": None, "chi_square_alpha": chi_square_alpha,
                "error": audit.get("error", "original WLS unavailable")}
    threshold = float(chi2.ppf(1 - chi_square_alpha, audit["dof"]))
    return {"success": True, "chi_square_alpha": chi_square_alpha,
        "chi_square_threshold": threshold, "chi_square_alarm": audit["J"] >= threshold,
        "normalized_residual_alarm": audit["max_normalized_residual"] >= NORMALIZED_RESIDUAL_THRESHOLD,
        "alarm": bool(audit["J"] >= threshold or audit["max_normalized_residual"] >= NORMALIZED_RESIDUAL_THRESHOLD),
        "same_fitted_observation": True}


def wls_audit(case, observed, sigma, *, chi_square_alpha=CHI_SQUARE_ALPHA, comparison_alphas=()):
    """Actual flat-start WLS with declared covariance and unchanged dual alarm."""
    try:
        fit = build_wls_features(case, observed, measurement_sigma=sigma, max_it=30, tol=1e-8)
        objective = float(fit["wls_objective"])
        local = float(np.max(np.abs(fit["signed_normalized_residual"])))
        threshold = float(chi2.ppf(1 - chi_square_alpha, fit["dof"]))
        result = {"success": True, "J": objective, "chi_square_threshold": threshold,
            "dof": int(fit["dof"]), "max_normalized_residual": local,
            "chi_square_alarm": objective >= threshold,
            "normalized_residual_alarm": local >= NORMALIZED_RESIDUAL_THRESHOLD,
            "alarm": bool(objective >= threshold or local >= NORMALIZED_RESIDUAL_THRESHOLD),
            "iterations": int(fit["iterations"])}
    except Exception as exc:
        result = {"success": False, "alarm": None, "error": f"{type(exc).__name__}: {exc}"}
    if comparison_alphas:
        result["chi_square_comparisons"] = {
            f"{alpha:g}": rethreshold_wls(result, alpha) for alpha in comparison_alphas}
    return result


def _external_vector(telemetry):
    return np.concatenate([np.asarray(telemetry["measurement_vector"]),
        *[np.asarray(row["vln_pu_rect"]).ravel() for row in telemetry["three_phase_voltages"]],
        *[np.asarray(row[key]).ravel() for row in telemetry["three_phase_branch_currents"]
          for key in ("i_from_pu_rect", "i_to_pu_rect")]])


def _snapshot(build, *, hif=None):
    dss = compile_model(Path(build["output_dir"]) / "Master.dss")
    receipt = None if hif is None else inject_midspan_hif(dss, build["registry"], build["assumptions"], **hif)
    telemetry = extract_measurements(dss, build["registry"], build["assumptions"],
        branch_overrides=None if receipt is None else receipt["branch_overrides"])
    audit = (validate_model(dss, build["reference"], build["registry"], build["assumptions"])
             if receipt is None else audit_disturbed_circuit(dss, receipt, build["registry"], build["assumptions"]))
    # The actual engine must retain the requested constant-PQ operating snapshot.
    power = {row["element"].lower(): complex(row["power_into_element_pu"]["real"],
        row["power_into_element_pu"]["imag"]) for row in telemetry["load_powers"]}
    pq_errors = []
    for row in build["registry"]["loads"]:
        expected = complex(row["kw"], row["kvar"]) / (1000 * build["assumptions"]["base_mva"])
        error = abs(power[row["element"].lower()] - expected)
        pq_errors.append(error)
        if error > 1e-8 + 1e-5 * abs(expected):
            raise RuntimeError(f"Constant-PQ load fallback: {row['element']}")
    if not audit["passed"]:
        raise RuntimeError(f"Physical circuit audit failed: {audit['failed_checks']}")
    return telemetry, receipt, audit, max(pq_errors, default=0.)


def _compact_fault(receipt, audit):
    voltage, current = complex(*audit["fault_voltage_v"]), complex(*audit["fault_current_a"])
    return {"resistance_pu": receipt["resistance_pu"], "zbase_ohm": receipt["zbase_ohm"],
        "fault_current_a": abs(current), "fault_current_pu": abs(current) / receipt["local_current_base_a"],
        "fault_voltage_ln_v": abs(voltage),
        "fault_voltage_ln_pu": abs(voltage) / receipt["local_voltage_base_ln_v"],
        "fault_power_mw": audit["fault_real_power_w"] / 1e6,
        "physical_audit": {"passed": audit["passed"], "checks": audit["checks"],
            "normalization": audit["normalization"], "active_phase_node_count": audit["active_phase_node_count"]},
        "split_initialization": {key: value for key, value in receipt["numerical_initialization"].items()
            if key in ("no_fault_external_voltage_max_deviation_pu", "no_fault_hidden_voltage_max_deviation_pu",
                       "voltage_normalization")}}


def summarize(cases, observations, parents, *, expected_physical_cases, noise_replicates, contract=CONTRACT):
    groups = defaultdict(list)
    for row in observations:
        key = (row["kind"], row["local_base_kv_ll"], row.get("resistance_ohm"), row["noise_profile"])
        groups[key].append(row)
    aggregates = []
    for (kind, kv, resistance, profile), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        successes = [row for row in rows if row["wls"]["success"]]
        alarms = sum(row["wls"].get("alarm") is True for row in successes)
        physical_ids = {row["case_id"] for row in rows}
        aggregates.append({"kind": kind, "local_base_kv_ll": kv, "resistance_ohm": resistance,
            "noise_profile": profile, "physical_case_count": len(physical_ids),
            "operating_parent_count": len({row["parent_id"] for row in rows}),
            "independent_noise_groups": len({row["noise_group_id"] for row in rows}),
            "observations": len(rows), "wls_succeeded": len(successes), "wls_failed": len(rows)-len(successes),
            "dual_alarms": alarms, "quiet": len(successes)-alarms,
            "alarm_fraction_of_successful_wls": alarms / len(successes) if successes else None,
            "chi_square_alarms": sum(row["wls"].get("chi_square_alarm") is True for row in successes),
            "normalized_residual_alarms": sum(row["wls"].get("normalized_residual_alarm") is True for row in successes),
            "J_min": min((row["wls"]["J"] for row in successes), default=None),
            "J_max": max((row["wls"]["J"] for row in successes), default=None)})
    failed = [row["case_id"] for row in cases if not row["physical_success"]]
    fault_observations = [row for row in observations if row["kind"] == "hif"]
    result = {"contract": contract, "selection": "unfiltered_all_requested_physical_cases",
        "operating_parent_count": len(parents), "parents_passed": sum(row["success"] for row in parents),
        "expected_physical_hif_cases": expected_physical_cases, "recorded_physical_hif_cases": len(cases),
        "physical_hif_succeeded": len(cases)-len(failed), "physical_hif_failed": len(failed), "failed_case_ids": failed,
        "complete": len(cases) == expected_physical_cases,
        "noise_replicates_per_parent_line_phase": noise_replicates,
        "fault_noisy_observations_expected": expected_physical_cases * noise_replicates * len(NOISE_PROFILES),
        "fault_noisy_observations_attempted": len(fault_observations),
        "fault_wls_failures": sum(not row["wls"]["success"] for row in fault_observations),
        "control_noisy_observations_attempted": len(observations)-len(fault_observations),
        "all_noisy_observations": len(observations), "by_voltage_resistance_and_noise": aggregates,
        "interpretation": "Steady-state synthetic resistive HIF and balanced-model residual detection; no protection, arcing, empirical error-rate, physical repair, or training-performance claim.",
        "dependence": "Resistance, accuracy, healthy and no-fault comparisons share unit-noise groups. Repeated windows and phase/line interventions share operating parents; aggregate cells are not independent population trials."}
    return result, aggregates


def configure_system(system="case14", voltage_profile=None):
    """Apply the explicitly supported local-base map before export or WLS use.

    Planned eligibility uses the corresponding profile helper; every compiled
    parent's actual registry is checked against it before any fault is run.
    """
    if system == "case14":
        profile = IEEE14_VOLTAGE_BASE_PROFILE_ID
        apply_bases, planned_rows = apply_ieee14_voltage_bases, eligible_ieee14_hif_branch_rows
    elif system == "case57":
        from three_phase_model.voltage_bases import (
            IEEE57_VOLTAGE_BASE_PROFILE_ID, apply_ieee57_voltage_bases, eligible_ieee57_hif_branch_rows,
        )
        profile = IEEE57_VOLTAGE_BASE_PROFILE_ID
        apply_bases, planned_rows = apply_ieee57_voltage_bases, eligible_ieee57_hif_branch_rows
    else:
        raise ValueError("Physical sweep supports system='case14' or system='case57'")
    if voltage_profile is not None and voltage_profile != profile:
        raise ValueError(f"Voltage profile {voltage_profile!r} is incompatible with {system}")
    source = apply_bases(resolve_system(system).load_case())
    return source, profile, list(planned_rows(source))


def run_sweep(output, *, load_scales=(.8, 1.), noise_replicates=1, seed=20260918,
              assumptions="normalized_diagonal", resistances_ohm=RESISTANCES_OHM,
              branch_rows=None, phases=(1, 2, 3), system="case14", voltage_profile=None,
              chi_square_alpha=None, comparison_alphas=None):
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"A fresh output directory is required: {output}")
    if (not load_scales or any(not math.isfinite(scale) or scale <= 0 for scale in load_scales)
        or len(set(load_scales)) != len(load_scales)):
        raise ValueError("load_scales must contain distinct finite positive values")
    if isinstance(noise_replicates, bool) or not isinstance(noise_replicates, int) or noise_replicates < 1:
        raise ValueError("noise_replicates must be a positive integer")
    if not resistances_ohm or any(not math.isfinite(r) or r <= 0 for r in resistances_ohm) or len(set(resistances_ohm)) != len(resistances_ohm):
        raise ValueError("resistances_ohm must contain distinct finite positive values")
    if not phases or len(set(phases)) != len(phases) or any(phase not in (1, 2, 3) for phase in phases):
        raise ValueError("phases must be a nonempty subset of ABC")
    source, voltage_profile, eligible = configure_system(system, voltage_profile)
    chi_square_alpha = (.05 if system == "case57" else CHI_SQUARE_ALPHA) if chi_square_alpha is None else float(chi_square_alpha)
    comparison_alphas = ((.01,) if system == "case57" else ()) if comparison_alphas is None else tuple(comparison_alphas)
    if any(not math.isfinite(alpha) or not 0 < alpha < 1 for alpha in (chi_square_alpha, *comparison_alphas)):
        raise ValueError("Chi-square alpha values must be strictly between zero and one")
    comparison_alphas = tuple(dict.fromkeys(alpha for alpha in comparison_alphas if alpha != chi_square_alpha))
    nb, nl = len(source["bus"]), len(source["branch"])
    measurement_count = 3 * nb + 4 * nl
    bus_kv = {int(row[0]): float(row[9]) for row in source["bus"]}
    contract = CONTRACT if system == "case14" else "ieee57_physical_hif_unfiltered_sweep_v1"
    selected = eligible if branch_rows is None else list(branch_rows)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(eligible):
        raise ValueError("branch_rows must be distinct eligible same-voltage lines")
    expected = len(load_scales) * len(selected) * len(phases) * len(resistances_ohm)
    output.mkdir(parents=True)
    sigmas = {profile: measurement_sigma(nb, nl, noise_profile=profile) for profile in NOISE_PROFILES}
    implementation = (*IMPLEMENTATION, "scripts/audit_ieee57_hif_physical_sweep.py") if system == "case57" else IMPLEMENTATION
    source_hashes = {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in implementation}
    config = {"contract": contract, "created_utc": datetime.now(timezone.utc).isoformat(),
        "system": system, "bus_count": nb, "branch_count": nl, "measurement_count": measurement_count,
        "voltage_profile": voltage_profile, "assumptions": assumptions,
        "load_scales": list(load_scales), "load_scaling": "multiply_bus_PD_QD; retain_non_slack_PG; solve_slack_PQ_and_PV_Q",
        "phases": list(phases), "alpha": .5, "resistances_ohm": list(resistances_ohm),
        "eligible_branch_rows0": eligible, "selected_branch_rows0": selected, "expected_physical_hif_cases": expected,
        "seed": int(seed), "noise_replicates": noise_replicates, "native_blas_threads": 1,
        "measurement_convention": MEASUREMENT_CONVENTION,
        "noise": {"distribution": "independent_zero_mean_Gaussian_per_real_SCADA_channel",
            "sigma_z": {key: value.tolist() for key, value in sigmas.items()},
            "sigma_vm_pu": .001, "sigma_power_pu": {key: float(value[nb]) for key, value in sigmas.items()},
            "unit_noise_seed_order": ["seed", "parent_ordinal", "branch_row0", "phase", "replicate"],
            "pairing": "same unit noise across all R, all accuracy profiles, and healthy/no-fault controls",
            "auxiliary_noise": "not_used_by_this_balanced_WLS_audit; offline_physical_VI_are_exact_engine_values"},
        "wls": {"chi_square_alpha": chi_square_alpha, "normalized_residual_threshold": NORMALIZED_RESIDUAL_THRESHOLD,
                "rule": "inclusive_chi_square_OR_absolute_normalized_residual", "flat_initialization": True},
        "selection": "none; retain quiet and failed cases", "implementation_sha256": source_hashes}
    if comparison_alphas:
        config["wls"]["comparison_chi_square_alphas"] = list(comparison_alphas)
        config["wls"]["comparison_method"] = "rethreshold_same_fitted_J_and_normalized_residual_no_resimulation_or_renoising"
    write_json(output/"experiment_config.json", config)
    for name in implementation:
        destination = output/"implementation_snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, destination)
    cases, observations, parent_rows, controls = [], [], [], []
    with threadpool_limits(limits=1), ExitStack() as stack:
        handles = {name: stack.enter_context((output/f"{name}.jsonl").open("w", encoding="utf-8"))
                   for name in ("cases", "controls", "wls_observations", "noise_groups")}

        def observe(record, z, phase, replicate, unit, group_id):
            for profile, sigma in sigmas.items():
                measured = np.asarray(z) + unit * sigma
                row = {key: record[key] for key in ("case_id", "parent_id", "kind", "load_scale", "local_base_kv_ll")}
                row.update(branch_row0=branch, phase=phase, noise_replicate=replicate,
                    resistance_ohm=record.get("resistance_ohm"), noise_profile=profile, noise_group_id=group_id,
                    sigma_vm_pu=float(sigma[0]), sigma_power_pu=float(sigma[nb]),
                    observed_sha256=numeric_hash(measured), wls=wls_audit(build["reference"], measured, sigma,
                        chi_square_alpha=chi_square_alpha, comparison_alphas=comparison_alphas))
                observations.append(row)
                _write_row(handles["wls_observations"], row)

        for parent_index, scale in enumerate(load_scales):
            parent_id = f"parent_{parent_index:02d}_load_{scale:g}"
            parent = {"parent_id": parent_id, "load_scale": scale, "success": False}
            parent_rows.append(parent)
            build, healthy = None, None
            try:
                physical = deepcopy(source)
                physical["bus"][:, 2:4] *= scale
                build = export_model(physical, output/"parents"/parent_id, assumptions=load_assumptions(assumptions),
                    case_id=system, voltage_profile=voltage_profile)
                if list(eligible_hif_branch_rows(build["registry"])) != eligible:
                    raise ValueError("Compiled asset eligibility differs from declared voltage-profile eligibility")
                healthy, _, validation, _ = _snapshot(build)
                write_json(output/"parents"/parent_id/"balanced_validation.json", validation)
                parent.update(success=True, noiseless_wls={profile: wls_audit(build["reference"], healthy["measurement_vector"], sigma,
                    chi_square_alpha=chi_square_alpha, comparison_alphas=comparison_alphas)
                                                          for profile, sigma in sigmas.items()})
            except Exception as exc:
                parent["error"] = f"{type(exc).__name__}: {exc}"
            print(f"parent {parent_id}: physical_success={parent['success']} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
            healthy_control = {"case_id": parent_id+"_healthy", "parent_id": parent_id, "kind": "healthy",
                "load_scale": scale, "physical_success": parent["success"],
                "mean_measurement_vector": None if healthy is None else healthy["measurement_vector"]}
            controls.append(healthy_control)
            _write_row(handles["controls"], healthy_control)
            for branch in selected:
                fb, tb = map(int, source["branch"][branch, :2])
                kv = bus_kv[fb]
                line = {"parent_id": parent_id, "load_scale": scale, "branch_row0": branch,
                        "from_bus": fb, "to_bus": tb, "local_base_kv_ll": kv}
                split_control = {**line, "case_id": f"{parent_id}_line_{branch:02d}_split", "kind": "no_fault_split",
                                 "physical_success": False}
                split = None
                try:
                    if not parent["success"]:
                        raise RuntimeError(f"Parent unavailable: {parent.get('error')}")
                    split, receipt, audit, _ = _snapshot(build, hif={"branch_row0": branch, "alpha": .5,
                        "phase": 1, "resistance_ohm": 100, "enabled": False})
                    deviation = float(np.max(np.abs(_external_vector(split)-_external_vector(healthy))))
                    if deviation > 2e-8:
                        raise RuntimeError(f"No-fault split differs from healthy telemetry: {deviation}")
                    split_control.update(physical_success=True, external_telemetry_max_difference_pu=deviation,
                        mean_measurement_vector=split["measurement_vector"],
                        physical_audit={"passed": audit["passed"], "checks": audit["checks"]})
                except Exception as exc:
                    split_control["error"] = f"{type(exc).__name__}: {exc}"
                controls.append(split_control)
                _write_row(handles["controls"], split_control)
                for phase in phases:
                    draws = []
                    for replicate in range(noise_replicates):
                        unit, entropy = standard_noise(seed, parent_index, branch, phase, replicate, size=measurement_count)
                        group_id = f"{parent_id}_line_{branch:02d}_phase_{phase}_noise_{replicate}"
                        draws.append((replicate, unit, group_id))
                        _write_row(handles["noise_groups"], {"noise_group_id": group_id, "seed_sequence_entropy": entropy,
                            "unit_noise": unit.tolist(), "unit_noise_sha256": numeric_hash(unit)})
                        if parent["success"]:
                            observe({**healthy_control, "local_base_kv_ll": kv}, healthy["measurement_vector"], phase, replicate, unit, group_id)
                        if split_control["physical_success"]:
                            observe(split_control, split["measurement_vector"], phase, replicate, unit, group_id)
                    for resistance in resistances_ohm:
                        row = {**line, "case_id": f"{parent_id}_line_{branch:02d}_phase_{phase}_R_{resistance:g}",
                            "kind": "hif", "phase": phase, "alpha": .5, "resistance_ohm": resistance,
                            "physical_success": False}
                        try:
                            if not parent["success"] or not split_control["physical_success"]:
                                raise RuntimeError("Parent or paired no-fault split failed its physical audit")
                            hif = {"branch_row0": branch, "phase": phase, "alpha": .5, "resistance_ohm": resistance}
                            telemetry, receipt, audit, pq_error = _snapshot(build, hif=hif)
                            equivalent, _, eq_audit, _ = _snapshot(build, hif={"branch_row0": branch, "phase": phase,
                                "alpha": .5, "resistance_pu": receipt["resistance_pu"]})
                            difference = float(np.max(np.abs(_external_vector(telemetry)-_external_vector(equivalent))))
                            current_error = abs(complex(*audit["fault_current_a"])-complex(*eq_audit["fault_current_a"])) / receipt["local_current_base_a"]
                            if max(difference, current_error) > 2e-8:
                                raise RuntimeError("Ohm and local-pu injection circuits disagree")
                            row.update(_compact_fault(receipt, audit), physical_success=True,
                                maximum_load_pq_error_system_pu=pq_error,
                                pu_equivalence={"passed": True, "external_max_error_pu": difference,
                                                "fault_current_error_pu": current_error},
                                mean_measurement_vector=telemetry["measurement_vector"])
                            for replicate, unit, group_id in draws:
                                observe(row, telemetry["measurement_vector"], phase, replicate, unit, group_id)
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        cases.append(row)
                        _write_row(handles["cases"], row)
                print(f"cases {len(cases)}/{expected}: {parent_id} line={branch} physical_failed={sum(not row['physical_success'] for row in cases)} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    summary, aggregates = summarize(cases, observations, parent_rows, expected_physical_cases=expected,
                                     noise_replicates=noise_replicates, contract=contract)
    if comparison_alphas:
        summary["chi_square_alpha_comparisons"] = {}
        for alpha in comparison_alphas:
            key = f"{alpha:g}"
            comparison_rows = [{**row, "wls": {**row["wls"], **row["wls"]["chi_square_comparisons"][key]}}
                               for row in observations]
            _, compared = summarize(cases, comparison_rows, parent_rows, expected_physical_cases=expected,
                                    noise_replicates=noise_replicates, contract=contract)
            summary["chi_square_alpha_comparisons"][key] = compared
            suffix = "alpha_" + key.replace(".", "p")
            _write_csv(output/f"detection_by_voltage_resistance_{suffix}.csv", [row for row in compared if row["kind"] == "hif"])
            _write_csv(output/f"controls_summary_{suffix}.csv", [row for row in compared if row["kind"] != "hif"])
    summary.update(wall_seconds=time.perf_counter()-started, parents=parent_rows,
        healthy_physical_controls=sum(row["kind"] == "healthy" for row in controls),
        no_fault_split_physical_controls=sum(row["kind"] == "no_fault_split" for row in controls),
        failed_physical_controls=[row["case_id"] for row in controls if not row["physical_success"]],
        implementation_unchanged_during_run=all(hashlib.sha256((REPO/name).read_bytes()).hexdigest() == digest
                                               for name, digest in source_hashes.items()))
    write_json(output/"summary.json", summary)
    _write_csv(output/"detection_by_voltage_resistance.csv", [row for row in aggregates if row["kind"] == "hif"])
    _write_csv(output/"controls_summary.csv", [row for row in aggregates if row["kind"] != "hif"])
    _write_csv(output/"physical_cases.csv", [{key: value for key, value in row.items()
        if key not in ("physical_audit", "split_initialization", "pu_equivalence", "mean_measurement_vector")} for row in cases])
    print(_json({"output": str(output), "physical_passed": summary["physical_hif_succeeded"],
        "physical_failed": summary["physical_hif_failed"], "fault_observations": summary["fault_noisy_observations_attempted"],
        "fault_wls_failures": summary["fault_wls_failures"], "wall_seconds": summary["wall_seconds"]}), flush=True)
    return summary


def argument_parser(*, default_system="case14", default_seed=20260918):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-scales", nargs="+", type=float, default=[.8, 1.])
    parser.add_argument("--noise-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--system", choices=("case14", "case57"), default=default_system)
    parser.add_argument("--voltage-profile")
    parser.add_argument("--chi-square-alpha", type=float)
    parser.add_argument("--comparison-chi-square-alphas", nargs="+", type=float)
    parser.add_argument("--assumptions", choices=("normalized_diagonal", "coupled_sensitivity"), default="normalized_diagonal")
    parser.add_argument("--resistances-ohm", nargs="+", type=float, default=list(RESISTANCES_OHM))
    parser.add_argument("--branch-rows", nargs="+", type=int)
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3])
    return parser


def main(argv=None, *, default_system="case14", default_seed=20260918):
    parser = argument_parser(default_system=default_system, default_seed=default_seed)
    args = parser.parse_args(argv)
    summary = run_sweep(args.output_dir, load_scales=args.load_scales, noise_replicates=args.noise_replicates,
        seed=args.seed, assumptions=args.assumptions, resistances_ohm=args.resistances_ohm,
        branch_rows=args.branch_rows, phases=args.phases, system=args.system, voltage_profile=args.voltage_profile,
        chi_square_alpha=args.chi_square_alpha, comparison_alphas=args.comparison_chi_square_alphas)
    return 0 if summary["complete"] and not summary["physical_hif_failed"] and not summary["failed_physical_controls"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
