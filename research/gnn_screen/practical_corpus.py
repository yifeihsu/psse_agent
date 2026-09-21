"""Practical physical IEEE-14 cohort with an explicit positive boundary ledger.

Parameter/status errors modify the physical network while the reported model
stays fixed. Admission uses physical descriptors and paired noiseless means;
neither fitted WLS statistics nor learned scores participate in admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from three_phase_model.exporter import export_model
from three_phase_model.runtime import compile_model, solve
from three_phase_model.measurements import extract_measurements
from three_phase_model.disturbances import (audit_disturbed_circuit, eligible_hif_branch_rows,
                                           inject_midspan_hif)
from three_phase_nlm.branch_current_analysis import line_differential_null_test
from .dataset import content_hash, file_sha256, jsonable, write_json
from .feature_schema import MEASUREMENT_CONVENTION
from .generate_corpus import (SPLITS, connected, healthy_audit, sample_parent,
                             solve_operating_parent, status_candidates)
from .scenario_policy import POLICY_VERSION, classify_scenario, paired_visibility
from .wls_features import configured_case, default_measurement_sigma, state_measurements_and_jacobian

PHYSICAL_HIF_PROFILE = "ieee14_physical_hif_v1"
SCENARIO_PROFILES = ("legacy_v1", "reviewed_v1", PHYSICAL_HIF_PROFILE)
NOISE_PROFILES = ("baseline", "accuracy_005", "accuracy_002")
REVIEWED_AUXILIARY_MANIFESTS = {
    "weak_hif_evaluation": "weak_hif_evaluation_manifest.jsonl",
    "parameter_sensitivity": "parameter_sensitivity_manifest.jsonl",
    "unbalance_sensitivity": "unbalance_sensitivity_manifest.jsonl",
    "unbalance_stress": "unbalance_stress_manifest.jsonl",
}

MAIN_MIN_DISTANCE = 5.0
CURRENT_COMPONENT_SIGMA_PU = .001
DIFFERENTIAL_COMPONENT_SIGMA_PU = np.sqrt(2.) * CURRENT_COMPONENT_SIGMA_PU
SLOTS = (
    ("hif_5to10", ("hif",), {"resistance_range": (5., 10.)}),
    ("hif_10to20", ("hif",), {"resistance_range": (10., 20.)}),
    ("hif_20to40", ("hif",), {"resistance_range": (20., 40.)}),
    ("unbalance_0", ("unbalance",), {}),
    ("unbalance_1", ("unbalance",), {}),
    ("unbalance_2", ("unbalance",), {}),
    ("measurement_single", ("measurement",), {"meter_count": 1}),
    ("measurement_multi_0", ("measurement",), {"meter_count": None}),
    ("measurement_multi_1", ("measurement",), {"meter_count": None}),
    ("parameter_R", ("parameter",), {"components": ("R",)}),
    ("parameter_X", ("parameter",), {"components": ("X",)}),
    ("parameter_RX", ("parameter",), {"components": ("R", "X")}),
    ("topology_0", ("topology",), {}),
    ("topology_1", ("topology",), {}),
    ("measurement_parameter", ("measurement", "parameter"), {"components": ("R", "X"), "meter_count": 1}),
    ("measurement_topology", ("measurement", "topology"), {"meter_count": 1}),
    ("measurement_hif", ("measurement", "hif"), {"resistance_range": (5., 40.), "meter_count": 1}),
)


def meter_overlay(z: np.ndarray, rng: np.random.Generator, *, count: int | None = None,
                  measurement_sigma: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """SFT-supported single or 2–5 same-measurement-type additive meter errors."""
    count = int(rng.integers(2, 6)) if count is None else int(count)
    if not 1 <= count <= 5:
        raise ValueError("Meter count must be between 1 and 5")
    blocks = (("Vm", 0, 14), ("Pinj", 14, 28), ("Qinj", 28, 42),
              ("Pf", 42, 62), ("Qf", 62, 82), ("Pt", 82, 102), ("Qt", 102, 122))
    channel, start, stop = blocks[int(rng.integers(len(blocks)))]
    indices = np.sort(rng.choice(np.arange(start, stop), size=count, replace=False))
    magnitudes = rng.uniform(10., 15., size=count)
    signs = rng.choice([-1., 1.], size=count)
    sigma = (default_measurement_sigma(14, 20) if measurement_sigma is None
             else np.asarray(measurement_sigma, dtype=float))
    if sigma.shape != (122,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("meter overlay requires 122 finite positive measurement sigmas")
    changed = np.asarray(z).copy()
    changed[indices] += magnitudes * signs * sigma[indices]
    return changed, {"kind": "measurement", "measurement_channel": channel,
        "channel_indices0": indices.tolist(), "sigma_multiples": (magnitudes * signs).tolist(),
        "minimum_absolute_sigma_multiple": float(min(magnitudes)),
        "maximum_absolute_sigma_multiple": float(max(magnitudes)), "meter_count": count}


def perturb_physical_case(case: dict, rng: np.random.Generator, *, family: str,
                          components: tuple[str, ...] = ("R", "X"),
                          parameter_cohort: str = "gross", scenario_profile: str = "legacy_v1") -> tuple[dict, dict]:
    """Mutate actual parameters/status, preserving external sensor identity."""
    physical = copy.deepcopy(case)
    if family == "parameter":
        if not components or set(components) - {"R", "X"}:
            raise ValueError("Parameter components must be R and/or X")
        columns = {"R": 2, "X": 3}
        eligible = [i for i, row in enumerate(physical["branch"])
                    if row[10] and row[8] == 0 and row[2] > 1e-9 and row[3] > 1e-9]
        if not eligible:
            raise ValueError("No eligible active line with nonzero R and X for SFT parameter mutation")
        row = int(rng.choice(eligible))
        if scenario_profile in {"reviewed_v1", PHYSICAL_HIF_PROFILE}:
            from psse_env.fault_profiles import parameter_factor
            factors = {c: parameter_factor(rng, cohort=parameter_cohort) for c in components}
        else:
            if parameter_cohort != "gross":
                raise ValueError("moderate sensitivity requires reviewed_v1")
            factors = {c: float(rng.uniform(.1, .5) if rng.random() < .5 else rng.uniform(2., 5.))
                       for c in components}
        before = {c: float(physical["branch"][row, columns[c]]) for c in components}
        for component, factor in factors.items():
            physical["branch"][row, columns[component]] *= factor
        audit = {"kind": family, "branch_row0": row, "physical_factors": factors,
            "factor_convention": "physical_actual_over_reported_parent", "reported_values": before,
            "actual_values": {c: float(physical["branch"][row, columns[c]]) for c in components}}
    elif family == "topology":
        eligible = status_candidates(physical)
        row = int(rng.choice(eligible))
        before = int(physical["branch"][row, 10])
        physical["branch"][row, 10] = 1 - before
        audit = {"kind": family, "branch_row0": row, "reported_status": before,
            "physical_status": 1 - before, "changed_connectivity": True,
            "physical_graph_connected": connected(physical),
            "scope": "branch_status_only_not_full_node_breaker_topology"}
    else:
        raise ValueError("Only physical parameter/status mutations are supported")
    audit["reported_model_unchanged"] = True
    return physical, audit


def mixed_meter_overlay(z: np.ndarray, rng: np.random.Generator, sigma: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reviewed mixed-root corruption: one power meter, fixed 0.10–0.30 pu."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (122,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("mixed meter overlay requires 122 finite positive measurement sigmas")
    index = int(rng.integers(14, 122))
    offset = float(rng.uniform(.10, .30)) * float(rng.choice([-1., 1.]))
    observed = np.asarray(z).copy()
    observed[index] += offset
    multiple = offset / sigma[index]
    return observed, {"kind": "measurement", "measurement_channel": "power_only",
        "channel_indices0": [index], "meter_count": 1, "additive_offsets_pu": [offset],
        "sigma_multiples": [float(multiple)], "minimum_absolute_sigma_multiple": abs(float(multiple)),
        "maximum_absolute_sigma_multiple": abs(float(multiple)),
        "scope": "reviewed_mixed_fixed_absolute_power_bias_0p10_to_0p30_pu",
        "baseline_sigma_interpretation": "10_to_30_sigma_when_power_sigma_is_0p01"}


def phase_snapshot(build: dict, *, hif: dict | None = None,
                   unbalance: dict | None = None, reviewed_audit: bool = False) -> tuple[np.ndarray, dict]:
    dss = compile_model(Path(build["output_dir"]) / "Master.dss")
    disturbance = None
    if unbalance:
        fractions = np.asarray(unbalance["fractions"], dtype=float)
        if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
            raise ValueError("Three positive phase fractions must sum to one")
        rows = [row for row in build["registry"]["loads"] if row["bus"] == unbalance["bus"]]
        if len(rows) != 3:
            raise ValueError("Unbalance bus needs three generated phase loads")
        commands = []
        totals = np.asarray([[row["kw"], row["kvar"]] for row in rows]).sum(axis=0)
        actual = np.zeros(2)
        for row in rows:
            factor = float(fractions[row["phase"] - 1] * 3)
            command = f"Edit {row['element']} kW={row['kw']*factor:.16g} kvar={row['kvar']*factor:.16g}"
            dss.Text.Command(command)
            commands.append(command)
            actual += np.asarray([row["kw"], row["kvar"]]) * factor
        if not np.allclose(totals, actual, atol=1e-8, rtol=1e-12):
            raise RuntimeError("Phase redistribution changed total load")
        solve(dss)
        disturbance = {"kind": "unbalance", **unbalance, "commands": commands,
            "total_kw_kvar_before": totals.tolist(), "total_kw_kvar_after": actual.tolist()}
    receipt = None
    if hif:
        receipt = inject_midspan_hif(dss, build["registry"], build["assumptions"], **hif)
        audit = audit_disturbed_circuit(dss, receipt, build["registry"], build["assumptions"])
        if not audit["passed"]:
            raise RuntimeError(f"Physical HIF audit failed: {audit['failed_checks']}")
        disturbance = {"kind": "hif", "settings": hif, "receipt": receipt, "audit": audit}
    telemetry = extract_measurements(dss, build["registry"], build["assumptions"],
        branch_overrides=receipt["branch_overrides"] if receipt else None)
    z = np.asarray(telemetry["measurement_vector"], dtype=float)
    if z.shape != (122,) or not np.isfinite(z).all() or telemetry["max_kcl_mismatch_pu"] > 1e-7:
        raise RuntimeError("Physical external measurement/KCL audit failed")
    pq_audit = audit_constant_pq(telemetry, build["registry"], build["assumptions"]["base_mva"], unbalance)
    params = {int(row["branch_row0"]): {"is_line": row["status"] == 1 and row["dss_element"].lower().startswith("line."),
        "b": row["b_pu"]} for row in build["registry"]["branches"]}
    differential = line_differential_null_test(telemetry["three_phase_voltages"],
        telemetry["three_phase_branch_currents"], sigma_pu=CURRENT_COMPONENT_SIGMA_PU, line_parameters=params)
    phase_magnitudes = [value for row in telemetry["three_phase_voltages"] for value in row["vln_pu"]]
    diagnostics = {"minimum_phase_voltage_pu": float(min(phase_magnitudes)),
        "maximum_phase_voltage_pu": float(max(phase_magnitudes)),
        "load_pq_audit": pq_audit,
        "maximum_voltage_negative_positive_ratio": max(
        float(row["vln_sequence_pu"][2]) / float(row["vln_sequence_pu"][1]) for row in telemetry["three_phase_voltages"]),
        "line_differential": differential, "current_component_sigma_pu": CURRENT_COMPONENT_SIGMA_PU,
        "differential_component_sigma_pu": float(DIFFERENTIAL_COMPONENT_SIGMA_PU),
        "current_noise_semantics": "independent real/imaginary components per terminal; two-terminal difference sigma=sqrt(2)*0.001pu",
        "current_channel_used_for_admission_audit_only": True}
    if reviewed_audit:
        from psse_env.fault_profiles import vuf_stratum
        bus_vuf = {int(row["external_bus"]): float(row["vln_sequence_pu"][2]) / float(row["vln_sequence_pu"][1])
                   for row in telemetry["three_phase_voltages"]}
        maximum_bus = max(bus_vuf, key=bus_vuf.get)
        source_bus = int(unbalance["bus"]) if unbalance else None
        powers = {row["element"].lower(): row["power_into_element_pu"] for row in telemetry["load_powers"]}
        source_powers = [
            {"phase": row["phase"], "p_pu": float(powers[row["element"].lower()]["real"]),
             "q_pu": float(powers[row["element"].lower()]["imag"])}
            for row in build["registry"]["loads"] if row["bus"] == source_bus
        ]
        diagnostics.update(vuf_stratum=vuf_stratum(bus_vuf[maximum_bus]), maximum_vuf_bus=maximum_bus,
                           source_bus=source_bus, source_bus_vuf=bus_vuf.get(source_bus),
                           source_phase_powers_system_pu=source_powers, per_bus_vuf=bus_vuf)
    if hif and hif.get("enabled", True):
        branch_asset = build["registry"]["branches"][int(hif["branch_row0"])]
        local_kv = next(float(bus["kv_ll"]) for bus in build["registry"]["buses"]
                        if bus["external_bus"] == branch_asset["from_bus"])
        ibase = float(build["assumptions"]["base_mva"]) * 1000 / (np.sqrt(3) * local_kv)
        actual_current = abs(complex(*disturbance["audit"]["fault_current_a"])) / ibase
        # The exact split retains canonical endpoint charging, so the exported
        # charge-corrected two-terminal differential equals the fault-path current.
        diff = float(differential["max_line_differential_pu"])
        diagnostics.update(hif_fault_path_current_pu=float(actual_current),
            hif_phase_current_sigma=float(actual_current / CURRENT_COMPONENT_SIGMA_PU),
            hif_differential_current_sigma=float(diff / DIFFERENTIAL_COMPONENT_SIGMA_PU),
            hif_differential_current_pu=diff,
            hif_current_evidence="charging_corrected_two_terminal_phase_current_differential",
            current_values_are_normalized_not_equipment_ampere_claims=True)
        if reviewed_audit:
            fault_voltage = complex(*disturbance["audit"]["fault_voltage_v"])
            fault_current = complex(*disturbance["audit"]["fault_current_a"])
            line = build["reference"]["branch"][int(hif["branch_row0"])]
            line_z = float(np.hypot(line[2], line[3]))
            diagnostics["physical_hif"] = {
                "resistance_pu": float(disturbance["receipt"]["resistance_pu"]),
                "resistance_ohm": float(disturbance["audit"]["resistance_ohm"]),
                "line_series_impedance_magnitude_pu": line_z,
                "resistance_to_line_impedance_ratio": float(disturbance["receipt"]["resistance_pu"]) / line_z,
                "fault_current_a_in_normalized_model": float(abs(fault_current)),
                "fault_voltage_v_in_normalized_model": float(abs(fault_voltage)),
                "fault_power_w_in_normalized_model": float((fault_voltage * fault_current.conjugate()).real),
                "equipment_ampere_or_voltage_claim": False,
            }
            if build["assumptions"].get("voltage_profile"):
                diagnostics["physical_hif"].update(
                    voltage_profile=build["assumptions"]["voltage_profile"], local_voltage_base_kv_ll=local_kv,
                    impedance_base_ohm=local_kv**2 / float(build["assumptions"]["base_mva"]),
                    fault_current_a=float(abs(fault_current)), fault_voltage_ln_v=float(abs(fault_voltage)),
                    fault_power_mw=float((fault_voltage * fault_current.conjugate()).real / 1e6),
                    quantities_follow_declared_voltage_bases=True)
                diagnostics["current_values_are_normalized_not_equipment_ampere_claims"] = False
                diagnostics["physical_hif"].pop("fault_current_a_in_normalized_model", None)
                diagnostics["physical_hif"].pop("fault_voltage_v_in_normalized_model", None)
                diagnostics["physical_hif"].pop("fault_power_w_in_normalized_model", None)
    return z, {"converged": True, "engine": dss.Basic.Version(),
        "exporter": "three_phase_model.measurements.extract_measurements:measurement_vector",
        "external_kcl_max_mismatch_pu": float(telemetry["max_kcl_mismatch_pu"]),
        "disturbance": disturbance, "diagnostics": diagnostics}


def audit_constant_pq(telemetry: dict, registry: dict, base_mva: float,
                      unbalance: dict | None = None) -> dict:
    """Reject the DSS voltage-envelope impedance fallback, not just bad KCL."""
    actual = {row["element"].lower(): complex(row["power_into_element_pu"]["real"],
        row["power_into_element_pu"]["imag"]) for row in telemetry["load_powers"]}
    max_relative = max_absolute = 0.
    for row in registry["loads"]:
        factor = (3 * unbalance["fractions"][row["phase"] - 1]
                  if unbalance and row["bus"] == unbalance["bus"] else 1.)
        requested = complex(row["kw"], row["kvar"]) * factor / (float(base_mva) * 1000.)
        measured = actual[row["element"].lower()]
        error = abs(measured - requested)
        relative = error / max(abs(requested), 1e-12)
        max_relative, max_absolute = max(max_relative, relative), max(max_absolute, error)
        if error > 1e-8 + 1e-5 * abs(requested):
            raise RuntimeError(f"Actual phase load P/Q differs from requested constant-PQ setting; voltage-envelope fallback: {row['element']}, relative_error={relative:.6g}")
    return {"passed": True, "maximum_relative_complex_power_error": max_relative,
        "maximum_absolute_complex_power_error_system_pu": max_absolute,
        "absolute_tolerance_system_pu": 1e-8, "relative_tolerance": 1e-5,
        "checked_load_count": len(registry["loads"]), "voltage_envelope_fallback_allowed": False}


def noiseless_wls_audit(case: dict, z: Any, sigma: Any) -> dict[str, Any]:
    """Offline fitted-model residual energy on a noiseless mean, never admission.

    ``J_exact`` is the actual nonlinear WLS objective at the converged numerical
    solution, not the linearized projection and not a proof of a global optimum.
    Positive diagonal sensor variances are required; exact channels need their
    own constrained audit. A failed solve stays unavailable, never zero energy.
    """
    from psse_env.fault_profiles import signal_energy_stratum
    from .wls_features import build_wls_features
    try:
        evidence = build_wls_features(case, z, measurement_sigma=sigma)
        objective = float(evidence["wls_objective"])
        if not np.isfinite(objective) or objective < 0:
            raise ValueError("noiseless WLS returned invalid residual energy")
        return {"success": True, "converged": True, "J_exact": objective,
            "residual_visible_energy_bin": signal_energy_stratum(objective),
            "degrees_of_freedom": int(evidence["dof"]),
            "solved_state_offline_only": {"theta_rad": evidence["theta_est_rad"].tolist(),
                                           "vm_pu": evidence["vm_est_pu"].tolist()},
            "role": "offline_noiseless_signal_energy_audit", "used_for_admission": False,
            "interpretation": "nonlinear_WLS_fitted_model_discrepancy_not_noise_realization_or_global_optimum_certificate"}
    except Exception as exc:
        return {"success": False, "converged": False, "J_exact": None,
            "residual_visible_energy_bin": "unavailable", "error": f"{type(exc).__name__}: {exc}",
            "role": "offline_noiseless_signal_energy_audit", "used_for_admission": False}


def reviewed_slots(*, stage: str, split: str) -> list[tuple[str, tuple[str, ...], dict]]:
    """Explicit curriculum and achieved-VUF targets; no weak-HIF training rows."""
    if stage not in {"early", "full"}:
        raise ValueError("stage must be early or full")
    from psse_env.fault_profiles import get_fault_profile
    weights = get_fault_profile("reviewed_v1")["hif"]["stage_weights"][stage]
    hif = [(f"hif_band{band}_{rep}", ("hif",), {"hif_band": band})
           for band, copies in enumerate(weights)
           for rep in range(copies)]
    ordinary = [row for row in SLOTS if not row[0].startswith(("hif_", "unbalance_"))]
    core_unbalance = [("unbalance_1to2", ("unbalance",), {"vuf_target": "1_to_2pct"}),
                      ("unbalance_2to3", ("unbalance",), {"vuf_target": "2_to_3pct"})]
    # Extra evaluation cohorts use held-out parents only, so loading these files
    # cannot accidentally introduce weak/sensitivity cases into split='train'.
    evaluation = [] if split == "train" else [
        ("hif_weak_evaluation", ("hif",), {"hif_band": "weak", "destination": "weak_hif_evaluation"}),
        ("unbalance_below1", ("unbalance",), {"vuf_target": "below_1pct", "destination": "unbalance_sensitivity"}),
        ("unbalance_above3", ("unbalance",), {"vuf_target": "above_3pct", "destination": "unbalance_stress"}),
        *[(f"parameter_moderate_{''.join(components)}", ("parameter",), {
            "components": components, "parameter_cohort": "moderate", "destination": "parameter_sensitivity"})
          for components in (("R",), ("X",), ("R", "X"))],
    ]
    return [] if split == "calibration" else hif + core_unbalance + ordinary + evaluation


def physical_hif_slots(*, stage: str, split: str) -> list[tuple[str, tuple[str, ...], dict]]:
    """69-kV main HIF curriculum, ohmic evaluation at both line voltages, and the
    evaluation-only 69 kV detection-limit cohort (1000--5000 ohm, 21--105 pu).

    The mixed ``measurement_hif`` slot reuses an accepted physical HIF component,
    so its legacy pu ``resistance_range`` option is dead metadata here and is
    dropped; the shared ``SLOTS`` tuple itself is unchanged for the other profiles.
    """
    from psse_env.fault_profiles import get_fault_profile
    if split == "calibration":
        return []
    profile = get_fault_profile(PHYSICAL_HIF_PROFILE)["hif"]
    ordinary = [(name, families, {key: value for key, value in options.items() if key != "resistance_range"})
                for name, families, options in reviewed_slots(stage=stage, split=split)
                if not name.startswith("hif_")]
    main = [(f"hif_69kv_band{band}_{rep}", ("hif",), {"hif_ohm_band": band, "voltage_kv": 69.0})
            for band, copies in enumerate(profile["stage_weights"][stage]) for rep in range(copies)]
    sweep = [] if split == "train" else [
        (f"hif_{kv:g}kv_{resistance:g}ohm", ("hif",), {
            "resistance_ohm": float(resistance), "voltage_kv": kv,
            "destination": "hif_resistance_evaluation", "resistance_sweep": True})
        for kv in (69.0, 13.8)
        for resistance in [*profile["evaluation_sweep_ohm"], *profile["extreme_evaluation_ohm"]]]
    # Detection-limit rows are retained independently of WLS detection and never
    # proposed for split='train' (the persist guard enforces this a second time).
    detection_limit = [] if split == "train" else [
        (f"hif_69kv_detection_limit_{rep}", ("hif",), {
            "hif_ohm_band": "detection_limit",
            "voltage_kv": float(profile["detection_limit_voltage_stratum_kv_ll"]),
            "destination": "hif_detection_limit_evaluation"})
        for rep in range(2)]
    return main + ordinary + sweep + detection_limit


def physical_impact_audit(case: dict, healthy_z: np.ndarray, physical_z: np.ndarray,
                          sigma: np.ndarray, perturbation: dict | None) -> dict:
    """Same-parent physical differences before any bad-meter overlay."""
    delta = np.asarray(physical_z) - np.asarray(healthy_z)
    audit = {"maximum_voltage_change_pu": float(np.max(np.abs(delta[:14]))),
             "maximum_power_channel_change_pu": float(np.max(np.abs(delta[14:]))),
             "physical_changed_channel_count_above_one_sigma": int(np.sum(np.abs(delta / sigma) >= 1)),
             "physical_paired_mean_distance": float(np.linalg.norm(delta / sigma))}
    if not perturbation:
        return audit
    row0 = int(perturbation["branch_row0"])
    base = float(case["baseMVA"])
    loading = max(np.hypot(healthy_z[42 + row0], healthy_z[62 + row0]),
                  np.hypot(healthy_z[82 + row0], healthy_z[102 + row0])) * base
    rating = float(case["branch"][row0, 5])
    audit.update(branch_row0=row0, parent_branch_loading_mva=float(loading),
                 parent_branch_rate_a_mva=rating,
                 parent_branch_loading_fraction=(float(loading / rating) if rating > 0 else None))
    if perturbation["kind"] == "parameter":
        audit["absolute_parameter_deltas_pu"] = {
            key: abs(float(value) - float(perturbation["reported_values"][key]))
            for key, value in perturbation["actual_values"].items()}
    else:
        audit["topology_impact"] = {"reported_status": perturbation["reported_status"],
                                    "physical_status": perturbation["physical_status"],
                                    "physical_graph_connected": perturbation["physical_graph_connected"]}
    return audit


def cohort_decision(families: list[str], physics: dict, visibility: dict, *, min_distance: float = MAIN_MIN_DISTANCE,
                    reviewed_mixed: bool = False) -> dict:
    diagnostics = physics["diagnostics"]
    perturbation = physics.get("physical_perturbation", {})
    measurement = physics.get("measurement_corruption", {})
    classification = classify_scenario(families,
        measurement_sigma_multiple=measurement.get("minimum_absolute_sigma_multiple"),
        parameter_physical_factors=perturbation.get("physical_factors"),
        max_vuf=diagnostics["maximum_voltage_negative_positive_ratio"],
        hif_injected=True if "hif" in families else None,
        hif_phase_current_sigma=diagnostics.get("hif_phase_current_sigma"),
        hif_differential_current_sigma=diagnostics.get("hif_differential_current_sigma"),
        topology_changed_connectivity=perturbation.get("changed_connectivity"), visibility=visibility)
    if reviewed_mixed:
        if "measurement" not in families or len(families) < 2:
            raise ValueError("reviewed mixed measurement scope requires a mixed measurement family")
        offsets = measurement.get("additive_offsets_pu", [])
        indices = measurement.get("channel_indices0", [])
        valid_meter = (len(offsets) == len(indices) == 1 and 14 <= indices[0] < 122
                       and .1 <= abs(offsets[0]) <= .3)
        classification["family_rules"]["measurement"] = {
            "cohort": "core" if valid_meter else "out_of_scope",
            "reason": "reviewed_mixed_power_meter_fixed_absolute_range",
            "additive_offsets_pu": offsets, "sigma_multiples": measurement.get("sigma_multiples"),
        }
        states = [rule["cohort"] for rule in classification["family_rules"].values()]
        classification["physical_cohort"] = ("out_of_scope" if "out_of_scope" in states else
                                             "boundary" if "boundary" in states else "core")
    physical_core = classification["physical_cohort"] == "core"
    visible = visibility["mean_separation_d"] >= min_distance
    admitted = not families or (physical_core and visible)
    return {**classification, "main_minimum_paired_distance": min_distance,
        "admitted_main": admitted, "admission_uses_noisy_scores": False,
        "rejection_reasons": ([] if admitted else
            (["physical_core_rule_not_met"] if not physical_core else []) +
            (["paired_mean_distance_below_main_margin"] if not visible else []))}


def generate_corpus(output_dir: str | Path, *, parents_by_split: dict[str, int], seed: int = 20260918,
                    noise_replicates: int = 2, healthy_calibration_replicates: int = 80,
                    healthy_replicates_by_split: dict[str, int] | None = None,
                    attempt_cap: int = 24, scenario_profile: str = "legacy_v1",
                    noise_profile: str = "baseline", stage: str = "full") -> dict:
    if scenario_profile not in SCENARIO_PROFILES or noise_profile not in NOISE_PROFILES:
        raise ValueError("Unsupported scenario_profile or noise_profile")
    if stage not in {"early", "full"} or (scenario_profile == "legacy_v1" and stage != "full"):
        raise ValueError("early curriculum requires reviewed_v1; stage must be early or full")
    physical_voltage = scenario_profile == PHYSICAL_HIF_PROFILE
    reviewed = scenario_profile in {"reviewed_v1", PHYSICAL_HIF_PROFILE}
    from psse_env.fault_profiles import get_fault_profile, measurement_sigma
    profile = get_fault_profile(scenario_profile) if reviewed else {"profile_id": "legacy_v1"}
    voltage_options = {"voltage_profile": profile["voltage_profile"]} if physical_voltage else {}
    auxiliary_files = dict(REVIEWED_AUXILIARY_MANIFESTS)
    if physical_voltage:
        auxiliary_files.pop("weak_hif_evaluation")
        auxiliary_files["hif_resistance_evaluation"] = "hif_resistance_evaluation_manifest.jsonl"
        auxiliary_files["hif_detection_limit_evaluation"] = "hif_detection_limit_evaluation_manifest.jsonl"
    if set(parents_by_split) - set(SPLITS) or any(int(n) != n or n < 0 for n in parents_by_split.values()):
        raise ValueError("Invalid split parent counts")
    if min(noise_replicates, healthy_calibration_replicates, attempt_cap) < 1:
        raise ValueError("Replicates/attempt cap must be positive")
    healthy_reps = dict.fromkeys(SPLITS, noise_replicates)
    healthy_reps["calibration"] = healthy_calibration_replicates
    healthy_reps.update(healthy_replicates_by_split or {})
    if set(healthy_reps) != set(SPLITS) or min(healthy_reps.values()) < 1:
        raise ValueError("Invalid healthy replicate counts")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sigma = (default_measurement_sigma(14, 20) if noise_profile == "baseline"
             else measurement_sigma(14, 20, noise_profile=noise_profile))
    write_json(out / "measurement_sigma.json", sigma)
    profile_identity = {"scenario_profile": scenario_profile, "noise_profile": noise_profile,
                        "stage": stage, "profile": profile, "measurement_sigma": sigma.tolist()}
    write_json(out / "scenario_profile.json", profile_identity)
    rng = np.random.default_rng(seed)
    counts: Counter = Counter()
    source_files = [Path(__file__), Path(__file__).with_name("scenario_policy.py"),
        Path(__file__).with_name("generate_corpus.py"),
        *sorted((Path(__file__).parents[2] / "three_phase_model").glob("*.py"))]
    if reviewed or noise_profile != "baseline":
        source_files.append(Path(__file__).parents[2] / "psse_env" / "fault_profiles.py")
    source_hashes = {str(p.relative_to(Path(__file__).parents[2])): file_sha256(p) for p in source_files}
    started = time.monotonic()
    audits = []
    with ExitStack() as stack:
        main = stack.enter_context((out / "manifest.jsonl").open("w", encoding="utf-8"))
        boundary = stack.enter_context((out / "boundary_manifest.jsonl").open("w", encoding="utf-8"))
        auxiliary = {cohort: stack.enter_context((out / filename).open("w", encoding="utf-8"))
                         for cohort, filename in auxiliary_files.items()} if reviewed else {}
        ledger = stack.enter_context((out / "proposal_ledger.jsonl").open("w", encoding="utf-8"))

        def log(record):
            ledger.write(json.dumps(jsonable(record), allow_nan=False, separators=(",", ":")) + "\n")

        ordinal = 0
        for split in SPLITS:
            for split_ordinal in range(int(parents_by_split.get(split, 0))):
                parent_id = f"practical_ieee14_{seed}_{split}_{split_ordinal:05d}"
                if reviewed or noise_profile != "baseline":
                    parent_id += f"_{scenario_profile}_{stage}_{noise_profile}"
                parent_seed = int(rng.integers(0, 2**63 - 1))
                parent_rng = np.random.default_rng(parent_seed)
                parent_dir = out / "parents" / parent_id
                for attempt in range(20):
                    proposed, operating = sample_parent(parent_rng, ordinal)
                    if physical_voltage:
                        from three_phase_model.voltage_bases import apply_ieee14_voltage_bases
                        proposed = apply_ieee14_voltage_bases(proposed)
                    try:
                        physical_parent, opf_audit = solve_operating_parent(proposed)
                        break
                    except (ValueError, RuntimeError) as exc:
                        log({"event": "parent_source_rejected", "parent_id": parent_id, "split": split,
                            "attempt": attempt, "reason": str(exc), "operating_point": operating})
                        counts["parent_source_rejections"] += 1
                else:
                    raise RuntimeError(f"No feasible source parent after 20 attempts: {parent_id}")
                build = export_model(physical_parent, parent_dir / "model", case_id="case14",
                    **voltage_options,
                    source_provenance={"parent_id": parent_id, "seed": parent_seed, "operating_point": operating, "source_opf": opf_audit})
                reported = configured_case(physical_parent)
                case_path = parent_dir / "configured_case.json"
                write_json(case_path, reported)
                healthy_z, healthy_physics = phase_snapshot(build, reviewed_audit=reviewed)
                audit = healthy_audit(build, healthy_z)
                if reviewed:
                    healthy_physics["noiseless_wls_audit"] = noiseless_wls_audit(reported, healthy_z, sigma)
                    audit["declared_noise_profile_wls"] = healthy_physics["noiseless_wls_audit"]
                audits.append(audit)
                write_json(parent_dir / "healthy_audit.json", audit)
                write_json(parent_dir / "source_opf_audit.json", opf_audit)
                _, jacobian = state_measurements_and_jacobian(reported,
                    np.deg2rad(build["reference"]["bus"][:, 8]), build["reference"]["bus"][:, 7])
                saved_boundary_families: set[str] = set()
                split_checked = False
                accepted_components: dict[str, list[dict]] = {family: [] for family in ("hif", "parameter", "topology")}

                def persist(name, families, z, physics, visibility, decision, cohort, settings):
                    metadata_path = parent_dir / f"{name}_physical_audit.json"
                    write_json(metadata_path, physics)
                    d = physics["diagnostics"]
                    hif = settings.get("hif")
                    imbalance = settings.get("unbalance")
                    severity = ("healthy" if not families else cohort if cohort in auxiliary else "sft_resistance_challenge" if cohort == "hif_sft_challenge"
                                else "practical_core" if decision["admitted_main"] else "positive_boundary")
                    record = {"case": case_path.relative_to(out).as_posix(), "z": z.tolist(),
                        "parent_id": parent_id, "window_id": f"{parent_id}:{name}", "split": split,
                        "families": families, "severity": severity,
                        "measurement_kind": "noiseless_mean", "measurement_convention": MEASUREMENT_CONVENTION,
                        "measurement_sigma": "measurement_sigma.json",
                        "noise_seed": int(parent_rng.integers(0, 2**63 - 1)),
                        "noise_replicates": healthy_reps[split] if not families else noise_replicates,
                        "offline_metadata": {"physical_audit_path": metadata_path.relative_to(out).as_posix(),
                            "parent_seed": parent_seed, "physical_case_hash": build["manifest"]["base_case_hash"],
                            "reported_case_hash": content_hash(reported), "physical_source": (
                                "declared_voltage_opendss_fixed_pq_snapshot" if physical_voltage else "normalized_diagonal_opendss_fixed_pq_snapshot"),
                            "measurement_source": healthy_physics["exporter"], "variant": name, "cohort": cohort,
                            "maximum_voltage_negative_positive_ratio": d["maximum_voltage_negative_positive_ratio"],
                            "hif_phase": hif["phase"] if hif else None,
                            "affected_phase": str(hif["phase"]) if hif else
                                ("fractions_" + "_".join(f"{v:.4g}" for v in imbalance["fractions"])) if imbalance else "none",
                            "paired_visibility": visibility, "scenario_policy": decision,
                            "component_core_audit": physics.get("component_core_audit"),
                            "settings": settings, "diagnostic_currents": {k: v for k, v in d.items() if "current" in k or "differential" in k},
                            "full_provenance_hash": content_hash(physics)}}
                    if reviewed or noise_profile != "baseline":
                        record["offline_metadata"].update(
                            scenario_profile=scenario_profile, noise_profile=noise_profile, curriculum_stage=stage,
                            profile_identity_sha256=content_hash(profile_identity),
                            physical_severity={"families": families, "vuf_stratum": d.get("vuf_stratum"),
                                "parameter_cohort": settings.get("parameter_cohort"),
                                "physical_rules": decision.get("family_rules")},
                            noiseless_wls_audit=physics.get("noiseless_wls_audit"),
                            physical_impact=physics.get("physical_impact"), physical_hif=d.get("physical_hif"),
                            actual_source_bus=d.get("source_bus"), source_bus_vuf=d.get("source_bus_vuf"),
                            source_phase_powers_system_pu=d.get("source_phase_powers_system_pu"),
                            vuf_stratum=d.get("vuf_stratum"), requested_vuf_stratum=settings.get("requested_vuf_stratum"),
                            training_eligible=bool(cohort == "main" and split == "train"),
                        )
                    if cohort in {"weak_hif_evaluation", "hif_resistance_evaluation",
                                  "hif_detection_limit_evaluation"} and split == "train":
                        raise ValueError(f"{cohort} rows are evaluation-only and cannot enter a training split")
                    target = main if cohort == "main" else auxiliary[cohort] if cohort in auxiliary else boundary
                    target.write(json.dumps(jsonable(record), allow_nan=False, separators=(",", ":")) + "\n")
                    scope = "main" if cohort == "main" else cohort if cohort in auxiliary else "boundary"
                    counts[f"{scope}:{split}:rows"] += 1
                    counts[f"{scope}:{split}:noise_windows"] += record["noise_replicates"]
                    for family in families or ["healthy"]:
                        counts[f"{scope}:{split}:{family}"] += 1
                    if reviewed:
                        if "unbalance" in families:
                            counts[f"vuf_stratum:{scope}:{split}:{d['vuf_stratum']}"] += 1
                        energy = physics.get("noiseless_wls_audit", {}).get("residual_visible_energy_bin", "unavailable")
                        counts[f"offline_energy_bin:{scope}:{split}:{energy}"] += 1

                visibility = paired_visibility(healthy_z, healthy_z, sigma, jacobian)
                decision = cohort_decision([], healthy_physics, visibility)
                persist("healthy", [], healthy_z, healthy_physics, visibility, decision, "main", {})
                slots = (physical_hif_slots(stage=stage, split=split) if physical_voltage else
                         reviewed_slots(stage=stage, split=split) if reviewed else
                         [] if split == "calibration" else list(SLOTS) + [
                    ("hif_sft_challenge", ("hif",), {"resistance_range": (20., 200.), "uniform_resistance": True})])
                for slot, family_tuple, options in slots:
                    families = list(family_tuple)
                    challenge = slot == "hif_sft_challenge"
                    accepted = False
                    component_family = next((f for f in families if f != "measurement"), None) if len(families) > 1 else None
                    if component_family and not accepted_components.get(component_family):
                        counts[f"{split}:unfilled_slots"] += 1
                        log({"event": "slot_unfilled", "parent_id": parent_id, "split": split,
                            "slot": slot, "families": families, "reason": "no_individually_accepted_core_component"})
                        continue
                    for attempt in range(attempt_cap):
                        name = f"{slot}_draw{attempt:02d}"
                        settings = {}
                        event = {"event": "proposal", "parent_id": parent_id, "split": split,
                            "slot": slot, "attempt": attempt, "families": families}
                        try:
                            variant_build = build
                            perturbation = None
                            reused = (accepted_components[component_family][int(parent_rng.integers(len(accepted_components[component_family])))]
                                      if component_family else None)
                            if reused:
                                settings = copy.deepcopy(reused["settings"])
                            if not reused and ("parameter" in families or "topology" in families):
                                family = "parameter" if "parameter" in families else "topology"
                                changed, perturbation = perturb_physical_case(physical_parent, parent_rng,
                                    family=family, components=options.get("components", ("R", "X")),
                                    parameter_cohort=options.get("parameter_cohort", "gross"), scenario_profile=scenario_profile)
                                if reviewed and family == "parameter":
                                    settings["parameter_cohort"] = options.get("parameter_cohort", "gross")
                                settings["physical_perturbation"] = perturbation
                                changed, variant_opf = solve_operating_parent(changed)
                                variant_build = export_model(changed, parent_dir / "physical_variants" / name,
                                    **voltage_options,
                                    case_id="case14", source_provenance={"parent_id": parent_id,
                                        "reported_model_unchanged": True, "perturbation": perturbation, "source_opf": variant_opf})
                            if not reused and "hif" in families:
                                if physical_voltage:
                                    from psse_env.fault_profiles import hif_resistance_ohm
                                    from three_phase_model.voltage_bases import hif_resistance_class, hif_resistance_spec
                                    bus_kv = {int(bus["external_bus"]): float(bus["kv_ll"]) for bus in build["registry"]["buses"]}
                                    eligible = [row0 for row0 in eligible_hif_branch_rows(build["registry"])
                                        if np.isclose(bus_kv[build["registry"]["branches"][row0]["from_bus"]], options.get("voltage_kv", 69.0))]
                                    if not eligible:
                                        raise ValueError("No active same-voltage line in the requested HIF voltage stratum")
                                    row0 = int(parent_rng.choice(eligible))
                                    resistance = (float(options["resistance_ohm"]) if "resistance_ohm" in options else
                                        hif_resistance_ohm(parent_rng, band=options.get("hif_ohm_band"), stage=stage))
                                    settings["hif"] = {"branch_row0": row0, "phase": int(parent_rng.integers(1, 4)),
                                        "alpha": float(parent_rng.uniform(.25, .75)), "resistance_ohm": resistance}
                                    kv = bus_kv[build["registry"]["branches"][row0]["from_bus"]]
                                    settings["hif_units"] = hif_resistance_spec(resistance, kv, physical_parent["baseMVA"])
                                    # Voltage-agnostic physical-ohm class; hif_units carries the local pu.
                                    settings["hif_resistance_class"] = hif_resistance_class(resistance)
                                    settings["hif_population"] = (
                                        "69kv_detection_limit_1000_to_5000_ohm" if options.get("hif_ohm_band") == "detection_limit"
                                        else "voltage_stratified_resistance_sweep" if options.get("resistance_sweep")
                                        else "69kv_main_100_to_1000_ohm")
                                elif reviewed:
                                    from psse_env.fault_profiles import hif_resistance
                                    resistance = hif_resistance(parent_rng, band=options.get("hif_band"), stage=stage)
                                else:
                                    low, high = options["resistance_range"]
                                    resistance = (float(parent_rng.uniform(low, high)) if options.get("uniform_resistance")
                                                  else float(np.exp(parent_rng.uniform(np.log(low), np.log(high)))))
                                if not physical_voltage:
                                    settings["hif"] = {"branch_row0": int(parent_rng.choice(eligible_hif_branch_rows(build["registry"]))),
                                        "phase": int(parent_rng.integers(1, 4)), "alpha": float(parent_rng.uniform(.25, .75)),
                                        "resistance_pu": resistance}
                                if not split_checked:
                                    null_z, null_physics = phase_snapshot(build, hif={**settings["hif"], "enabled": False})
                                    null_audit = healthy_audit(build, null_z)
                                    null_audit.update(maximum_unsplit_measurement_difference_pu=float(np.max(np.abs(null_z-healthy_z))), physics=null_physics)
                                    write_json(parent_dir / "disabled_hif_split_control_audit.json", null_audit)
                                    split_checked = True
                            if "unbalance" in families:
                                buses = sorted({r["bus"] for r in build["registry"]["loads"]})
                                settings["unbalance"] = {"bus": int(parent_rng.choice(buses)),
                                    "fractions": parent_rng.dirichlet([3., 3., 3.]).tolist(), "distribution": "Dirichlet(3,3,3)"}
                                if reviewed:
                                    settings["requested_vuf_stratum"] = options.get("vuf_target")
                            z, physics = ((reused["z"].copy(), copy.deepcopy(reused["physics"])) if reused else
                                phase_snapshot(variant_build, hif=settings.get("hif"), unbalance=settings.get("unbalance"), reviewed_audit=reviewed)
                                if perturbation or settings else (healthy_z.copy(), copy.deepcopy(healthy_physics)))
                            if reused:
                                physics["component_core_audit"] = {
                                    "source_window_id": f"{parent_id}:{reused['name']}", "family": component_family,
                                    "individually_accepted_before_meter_overlay": True,
                                    "paired_visibility": reused["visibility"], "scenario_policy": reused["decision"],
                                    "component_mean_sha256": content_hash(reused["z"])}
                            if perturbation:
                                physics["physical_perturbation"] = perturbation
                                physics["variant_source_opf"] = variant_opf
                                physics["actual_physical_model_path"] = Path(variant_build["output_dir"]).relative_to(out).as_posix()
                                healthy_audit(variant_build, z)
                            if reviewed:
                                physics["physical_impact"] = physical_impact_audit(
                                    reported, healthy_z, z, sigma, physics.get("physical_perturbation"))
                            if "measurement" in families:
                                if reviewed and len(families) > 1:
                                    z, measurement = mixed_meter_overlay(z, parent_rng, sigma)
                                    meter_only = healthy_z.copy()
                                    meter_only[measurement["channel_indices0"]] += measurement["additive_offsets_pu"]
                                    physics["mixed_counterfactuals"] = {
                                        "source_component_window_id": physics["component_core_audit"]["source_window_id"],
                                        "source_component_mean_sha256": physics["component_core_audit"]["component_mean_sha256"],
                                        "meter_only_mean": meter_only.tolist(),
                                        "meter_only_mean_sha256": content_hash(meter_only),
                                        "shared_additive_offsets_pu": measurement["additive_offsets_pu"],
                                        "role": "offline_paired_component_meter_and_mixed_attribution",
                                    }
                                else:
                                    z, measurement = meter_overlay(z, parent_rng, count=options.get("meter_count"), measurement_sigma=sigma)
                                physics["measurement_corruption"] = measurement
                                settings["measurement"] = measurement
                            visibility = paired_visibility(healthy_z, z, sigma, jacobian, same_configured_model=True)
                            decision = cohort_decision(families, physics, visibility,
                                reviewed_mixed=reviewed and "measurement" in families and len(families) > 1)
                            target_matches = True
                            destination = options.get("destination") if reviewed else None
                            if reviewed:
                                physics["noiseless_wls_audit"] = noiseless_wls_audit(reported, z, sigma)
                                target_matches = not options.get("vuf_target") or (
                                    physics["diagnostics"]["vuf_stratum"] == options["vuf_target"])
                                decision.update(target_vuf_stratum=options.get("vuf_target"),
                                    achieved_vuf_stratum=physics["diagnostics"].get("vuf_stratum"),
                                    target_stratum_reached=bool(target_matches),
                                    selection_rule="historical_physical_core_and_paired_mean_d_ge5_plus_explicit_target_stratum",
                                    admission_uses_J_exact=False)
                                if not target_matches:
                                    decision["admitted_main"] = False
                                    decision["rejection_reasons"].append("requested_vuf_stratum_not_reached")
                                if destination:
                                    decision["admitted_main"] = False
                                    decision["evaluation_cohort_retention_independent_of_gross_or_paired_d_gate"] = True
                            event.update(settings=settings, physics_valid=True, paired_visibility=visibility,
                                scenario_policy=decision, noiseless_mean_sha256=content_hash(z),
                                diagnostics=physics["diagnostics"])
                            counts[f"proposal:{split}:valid"] += 1
                            if challenge or decision["admitted_main"] or (destination and target_matches):
                                persist(name, families, z, physics, visibility, decision,
                                    destination or ("hif_sft_challenge" if challenge else "main"), settings)
                                event["outcome"] = "evaluation_cohort_retained" if destination else "challenge_retained" if challenge else "main_accepted"
                                log(event)
                                accepted = True
                                if not challenge and not destination and len(families) == 1 and families[0] in accepted_components:
                                    accepted_components[families[0]].append({"name": name, "z": z.copy(),
                                        "physics": copy.deepcopy(physics), "visibility": visibility,
                                        "decision": decision, "settings": copy.deepcopy(settings)})
                                break
                            event["outcome"] = "valid_positive_below_main_criteria"
                            counts[f"proposal:{split}:below_main"] += 1
                            # Bound storage while retaining at least the first
                            # positive boundary example for each family/parent.
                            if reviewed or not any(f in saved_boundary_families for f in families):
                                boundary_destination = "positive_boundary"
                                if reviewed and split != "train" and families == ["unbalance"]:
                                    boundary_destination = {"below_1pct": "unbalance_sensitivity",
                                        "above_3pct": "unbalance_stress"}.get(physics["diagnostics"]["vuf_stratum"], "positive_boundary")
                                persist(name, families, z, physics, visibility, decision, boundary_destination, settings)
                                saved_boundary_families.update(families)
                                event["boundary_snapshot_retained"] = True
                            else:
                                event["boundary_snapshot_retained"] = False
                            log(event)
                        except (RuntimeError, ValueError) as exc:
                            counts[f"proposal:{split}:simulation_failure"] += 1
                            log({**event, "settings": settings, "outcome": "simulation_failure", "reason": str(exc)})
                    if not accepted:
                        counts[f"{split}:unfilled_slots"] += 1
                        log({"event": "slot_unfilled", "parent_id": parent_id, "split": split,
                            "slot": slot, "families": families, "attempt_cap": attempt_cap})
                ordinal += 1
                counts[f"{split}:parents"] += 1
                for handle in (main, boundary, ledger, *auxiliary.values()):
                    handle.flush()
                print(json.dumps({"event": "parent_complete", "parent_id": parent_id, "completed_parents": ordinal,
                    "main_rows": sum(v for k, v in counts.items() if k.startswith("main:") and k.endswith(":rows")),
                    "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    report = {"schema": "practical_physical_wls_screen_corpus_v1", "policy_version": POLICY_VERSION,
        "manifest": str(out / "manifest.jsonl"), "boundary_manifest": str(out / "boundary_manifest.jsonl"),
        "proposal_ledger": str(out / "proposal_ledger.jsonl"), "seed": seed, "parents_by_split": parents_by_split,
        "noise_replicates": noise_replicates, "healthy_replicates_by_split": healthy_reps,
        "attempt_cap_per_slot": attempt_cap, "intended_main_slots_per_noncalibration_parent": (
            1 + sum(not item[2].get("destination") for item in (
                physical_hif_slots(stage=stage, split="validation") if physical_voltage else reviewed_slots(stage=stage, split="validation")))
            if reviewed else 1+len(SLOTS)),
        "scenario_profile": scenario_profile, "noise_profile": noise_profile, "curriculum_stage": stage,
        "profile_identity": profile_identity, "profile_identity_sha256": content_hash(profile_identity),
        "main_minimum_paired_distance": MAIN_MIN_DISTANCE, "counts": dict(counts),
        "manifest_sha256": file_sha256(out / "manifest.jsonl"),
        "boundary_manifest_sha256": file_sha256(out / "boundary_manifest.jsonl"),
        "implementation_sha256": source_hashes, "elapsed_seconds": time.monotonic()-started,
        "healthy_max_balanced_equation_error_pu": max((a["maximum_balanced_equation_error_pu"] for a in audits), default=None),
        "admission_uses_wls_alarm": False, "admission_uses_noisy_or_learned_scores": False,
        "reported_model_stays_identical_within_parent": True,
        "profiles": {"meter": "Uniform10–15sigma; one or2–5same-channel meters",
            "parameter": "PhysicalR/X/RX actual/reported factorsUniform(.1,.5) orUniform(2,5); sourceACOPFresolved",
            "unbalance": "Dirichlet(3,3,3) phase fractions; main maxVUF>=.01",
            "hif_main": "Loguniform5–10,10–20,20–40pu slots,ABC,alphaUniform(.25,.75); stronger5–20explicitlyextendsSFT",
            "hif_challenge": "Uniform20–200pu SFT-supported resistance envelope retained independentofmaincriteria",
            "topology": "Physicalconnectedbranchstatustoggle withreportedparentstatusunchanged; notfullnodebreaker"},
        "limitations": ["Main is a selected practical cohort, not unrestricted fault-population coverage",
            "Known-parent paired clean distance is an offline oracle audit, not guaranteed learnability under unknown operating state",
            "Low-visibility/low-VUF HIF/unbalance remain positive in a separately bounded boundary corpus and full proposal ledger",
            "Fixed-PQ normalized diagonal three-phase snapshots; no arcing, harmonic, PV regulation or equipment-ampere claim",
            "Status scope is connected branch switching, not full node-breaker topology"]}
    if reviewed:
        report.update(
            schema="practical_physical_wls_screen_corpus_v2",
            auxiliary_manifests={cohort: {"path": str(out / filename), "sha256": file_sha256(out / filename),
                "training_eligible": False, "parent_splits": ["validation", "test"],
                "row_count": sum(counts.get(f"{cohort}:{split}:rows", 0) for split in SPLITS),
                "rows_by_split": {split: counts.get(f"{cohort}:{split}:rows", 0) for split in SPLITS}}
                for cohort, filename in auxiliary_files.items()},
            positive_boundary_storage="all_valid_off_target_or_below_main_proposals_are_retained_with_positive_labels",
            exact_wls_audit_used_for_admission=False,
            target_stratum_shortfalls="unfilled_slots and proposal ledger; no outcome-based threshold relaxation",
        )
        report["profiles"].update(
            meter="PureUniform10–15sigma; mixedonepower-onlysignedUniform(.10,.30)pu offset with actual selected covariance recorded",
            hif_main=f"Uniform within reviewed bands; {stage} slot weights {profile['hif']['stage_weights'][stage]}; ABC uniform; alphaUniform(.25,.75)",
            hif_challenge="Uniform20–200pu; validation/test parents only; dedicated weak_hif_evaluation_manifest",
            parameter="Gross factors preserved; separate evaluation-only moderate factor sensitivity cohorts",
            unbalance="SymmetricDirichlet(3,3,3); bounded achieved-VUF slots1–2pct/2–3pct main, below1pct sensitivity andabove3pct stress separately",
        )
        evaluation_slot_note = "Dedicated weak/sensitivity/stress slots use validation/test parents only; train off-target positives remain non-main boundary evidence"
        if physical_voltage:
            report["schema"] = "practical_physical_wls_screen_corpus_v3"
            report["voltage_profile"] = profile["voltage_profile"]
            classes = "; ".join(
                f"{row['name']} [{row['lower']:g}, {'inf' if row['upper'] is None else format(row['upper'], 'g')}) ohm"
                for row in profile["hif"]["resistance_classification_ohm"])
            detection_low, detection_high = profile["hif"]["detection_limit_band_ohm"]
            pu_low, pu_high = profile["hif"]["pu_equivalents_69kv"]["detection_limit_band_pu"]
            report["profiles"].update(
                hif_main="69-kV line faults; uniform within 100-200/200-500/500-1000 ohm bands (2.1-21.0 pu on the 69 kV base), converted using each line's voltage base",
                hif_challenge="Unfiltered physical resistance evaluation at 69 and 13.8 kV: 50/100/200/500/1000/2000/5000 ohm",
                hif_detection_limit=(
                    f"Evaluation-only 69 kV detection-limit cohort: uniform {detection_low:g}-{detection_high:g} ohm "
                    f"({pu_low:.1f}-{pu_high:.1f} pu on the 69 kV base), two slots per validation/test parent, "
                    "never proposed for train; retained independently of WLS detection in hif_detection_limit_evaluation_manifest.jsonl"),
                hif_resistance_classification=(
                    "Every physical HIF row carries settings.hif_resistance_class from the voltage-agnostic physical-ohm table "
                    f"(lower bound inclusive, upper exclusive): {classes}"),
            )
            report["limitations"] = [text for text in report["limitations"] if "normalized diagonal" not in text]
            report["limitations"].append("Declared IEEE14 nominal voltage realization; grounded-wye and sequence assumptions remain explicit research choices, not field validation")
            report["limitations"].append("Detection-limit rows (1000-5000 ohm at 69 kV) are evaluation evidence only; the recorded sweep shows the baseline balanced WLS alarms on none of them")
            evaluation_slot_note = "Dedicated resistance-sweep/detection-limit/sensitivity/stress slots use validation/test parents only; train off-target positives remain non-main boundary evidence"
        report["limitations"] = [text for text in report["limitations"] if "separately bounded boundary" not in text] + [
            "Every reviewed valid positive outside main is retained in boundary/auxiliary manifests and proposal ledger",
            "J_exact is offline noiseless fitted-model residual energy; failed solves remain unavailable and never gate admission",
            evaluation_slot_note,
        ]
    write_json(out / "generation_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    for split, default in zip(SPLITS, (100, 30, 100, 50)):
        parser.add_argument(f"--{split}-parents", type=int, default=default)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--noise-replicates", type=int, default=2)
    parser.add_argument("--healthy-calibration-replicates", type=int, default=80)
    parser.add_argument("--healthy-validation-replicates", type=int)
    parser.add_argument("--healthy-test-replicates", type=int)
    parser.add_argument("--attempt-cap", type=int, default=24)
    parser.add_argument("--scenario-profile", choices=SCENARIO_PROFILES, default="legacy_v1")
    parser.add_argument("--noise-profile", choices=NOISE_PROFILES, default="baseline")
    parser.add_argument("--stage", choices=("early", "full"), default="full")
    args = parser.parse_args(argv)
    report = generate_corpus(args.output_dir, parents_by_split={s: getattr(args, f"{s}_parents") for s in SPLITS},
        seed=args.seed, noise_replicates=args.noise_replicates,
        healthy_calibration_replicates=args.healthy_calibration_replicates, attempt_cap=args.attempt_cap,
        healthy_replicates_by_split={s: n for s, n in (("validation", args.healthy_validation_replicates),
            ("test", args.healthy_test_replicates)) if n is not None},
        scenario_profile=args.scenario_profile, noise_profile=args.noise_profile, stage=args.stage)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
