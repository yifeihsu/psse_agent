"""Versioned research cohorts implementing the September 2026 severity review.

The five-family GNN corpus, harmonic meter-semantics experiment, and full
node/breaker experiment remain explicitly different adapters. Existing corpora
and trained models are never overwritten or silently assigned new labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from psse_env.fault_profiles import (
    get_fault_profile, measurement_chain_error, measurement_sigma, signal_energy_stratum,
)
from research.gnn_screen.dataset import content_hash, file_sha256, jsonable, write_json
from research.gnn_screen.scenario_policy import paired_visibility
from research.gnn_screen.wls_features import build_wls_features


PROFILE = "reviewed_v1"
DEFAULT_PROFILE = "ieee14_physical_hif_v1"
NOISE_PROFILES = ("baseline", "accuracy_005", "accuracy_002")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), allow_nan=False, separators=(",", ":")) + "\n")


def _load_case(path: Path) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    for key in ("bus", "branch", "gen", "gencost"):
        if key in case:
            case[key] = np.asarray(case[key], dtype=float)
    return case


def exact_wls_audit(case: dict, mean: Any, sigma: Any, *, exact_rows=()) -> dict:
    """Offline fitted residual energy; a failed fit is unavailable, never zero."""
    try:
        if exact_rows:
            from tools.lagrangian_port import lagrangian_m_singlephase_details
            details = lagrangian_m_singlephase_details(np.asarray(mean), case, 0, case["bus"],
                measurement_sigma=np.asarray(sigma), exact_measurement_indices=list(exact_rows),
                max_it=60, tol=1e-9)
            if not details["success"]:
                raise ValueError("constrained noiseless WLS did not converge")
        else:
            details = build_wls_features(case, mean, measurement_sigma=sigma, max_it=60, tol=1e-9)
        value = float(details["wls_objective"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("invalid noiseless WLS objective")
        return {"success": True, "J_exact": value,
                "residual_visible_energy_bin": signal_energy_stratum(value),
                "dof": int(details["dof"]), "used_for_admission": False,
                "interpretation": "converged nonlinear WLS fit; no global-minimum guarantee"}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return {"success": False, "J_exact": None, "residual_visible_energy_bin": "unavailable",
                "error": str(exc), "used_for_admission": False}


def _healthy_parents(core: Path) -> list[dict]:
    rows = [row for row in read_jsonl(core / "manifest.jsonl") if not row["families"]]
    for row in rows:
        if isinstance(row["z"], str):
            row["z"] = json.loads((core / row["z"]).read_text(encoding="utf-8"))
    return rows


def build_harmonic_companion(core: Path, out: Path, *, seed: int, noise_profiles=NOISE_PROFILES,
                             scenario_profile: str = PROFILE) -> dict:
    """Same solved parents, two meter definitions, lower-THD and stress cohorts."""
    from Transmission.generate_hse_traces import build_trace, draw_harmonic_target_thd
    rng = np.random.default_rng(seed)
    rows = []
    for parent in _healthy_parents(core):
        source = _load_case(core / "parents" / parent["parent_id"] / "model" / "source_case.json")
        reported = _load_case(core / parent["case"])
        source_bus = int(rng.choice([2, 3, 4, 5, 9, 10, 11, 12, 13, 14]))
        targets = [("healthy", 0.0)]
        if parent["split"] != "calibration":
            targets += [(cohort, draw_harmonic_target_thd(rng, profile=cohort))
                        for cohort in ("sensitivity", "stress")]
        # Common unit-noise vectors allow a paired accuracy comparison, without
        # altering the physical event or drawing noise twice.
        for cohort, target in targets:
            trace_seed = int(rng.integers(0, 2**31 - 1))
            for convention in ("fundamental", "true_rms"):
                trace = build_trace(source_bus, target, trace_seed, bus=source["bus"],
                    branch=source["branch"], voltage_measurement=convention)
                harmonic_observations = {order: [
                    {key: value for key, value in item.items() if key != "V_complex_true"}
                    for item in measurements] for order, measurements in trace["harmonic_phasors"].items()}
                mean = np.asarray(trace["z_scada_true"])
                standard_noise = np.random.default_rng(trace_seed + 1).standard_normal(mean.size)
                if target == 0:
                    audit = exact_wls_audit(reported, mean, measurement_sigma(14, 20))
                    if not audit["success"] or audit["J_exact"] > 1e-5:
                        raise ValueError(f"Harmonic zero-THD control/export mismatch: {audit}")
                for noise_profile in noise_profiles:
                    sigma = measurement_sigma(14, 20, noise_profile=noise_profile)
                    rows.append({
                        "schema": "harmonic_meter_semantics_scenario_v1", "profile_id": scenario_profile,
                        "parent_id": parent["parent_id"], "split": parent["split"],
                        "window_id": f"{parent['parent_id']}:harmonic:{cohort}:{convention}:{noise_profile}",
                        "families": [] if target == 0 else ["harmonic"], "cohort": cohort,
                        "case": jsonable(reported), "measurements": (mean + standard_noise * sigma).tolist(),
                        "sigma_z": sigma.tolist(), "noise_profile": noise_profile,
                        "measurement_semantics": trace["measurement_semantics"],
                        "harmonic_phasors": harmonic_observations,
                        "noise_contract": {"distribution": "independent_gaussian",
                            "applied_sigma": sigma.tolist(), "estimator_sigma": sigma.tolist(),
                            "draw_count": 1, "paired_standard_noise_across_accuracy_profiles": True},
                        "offline_audit": {"z_exact": mean.tolist(), "physical_severity": trace["physical_severity"],
                            "observable_strength": exact_wls_audit(reported, mean, sigma),
                            "source_case_sha256": content_hash(source), "noise_seed": trace_seed + 1,
                            "harmonic_solver": "linear frequency-domain source/network solve on solved OPF fundamental",
                            "not_a_fully_coupled_nonlinear_power_flow": True},
                    })
    path = out / "harmonic_scenarios.jsonl"
    write_jsonl(path, rows)
    return {"path": str(path), "rows": len(rows), "sha256": file_sha256(path),
            "adapter": "harmonic_meter_semantics; deliberately excluded from five-family GNN labels",
            "counts": dict(Counter(f"{r['split']}:{r['cohort']}:{r['noise_profile']}" for r in rows))}


def build_measurement_chain_companion(core: Path, out: Path, *, seed: int, scenario_profile: str = PROFILE) -> dict:
    """Declared CT sensitivity cases, separate from the additive gross errors."""
    rng = np.random.default_rng(seed)
    rows = []
    # These are explicit sensitivity assumptions, not instrument specifications.
    mechanisms = (("ct_gain_minus5pct", {"ct_gain": .95}),
                  ("ct_gain_plus5pct", {"ct_gain": 1.05}),
                  ("ct_angle_minus1deg", {"ct_angle_rad": -np.pi / 180}),
                  ("ct_angle_plus1deg", {"ct_angle_rad": np.pi / 180}))
    for parent in _healthy_parents(core):
        if parent["split"] == "calibration":
            continue
        case = _load_case(core / parent["case"])
        branch = int(rng.choice(np.flatnonzero(case["branch"][:, 10])))
        p_index, q_index = 42 + branch, 62 + branch
        for name, setting in mechanisms:
            z = np.asarray(parent["z"], dtype=float).copy()
            z[p_index], z[q_index] = measurement_chain_error(z[p_index], z[q_index], **setting)
            row = copy.deepcopy(parent)
            row.update(case=jsonable(case), z=z.tolist(), families=["measurement"],
                window_id=f"{parent['parent_id']}:{name}", severity="measurement_chain_sensitivity",
                measurement_sigma=measurement_sigma(14, 20).tolist(),
                noise_seed=int(rng.integers(0, 2**63 - 1)))
            row["offline_metadata"] = {"profile_id": scenario_profile, "cohort": "measurement_chain_sensitivity",
                "measurement_chain": {"mechanism": name, **setting, "branch_row0": branch,
                    "indices0": [p_index, q_index], "field_calibrated": False,
                    "power_transform": "S_measured=kV*kI*exp(j*(deltaV-deltaI))*S_true",
                    "noise_applied_after_systematic_transform": True},
                "observable_strength": exact_wls_audit(case, z, measurement_sigma(14, 20)),
                "healthy_window_id": parent["window_id"], "additive_floor_applied": False}
            rows.append(row)
    path = out / "measurement_chain_manifest.jsonl"
    write_jsonl(path, rows)
    return {"path": str(path), "rows": len(rows), "sha256": file_sha256(path),
            "included_in_main_training_manifest": False}


def _relative_reference(value: str, source: Path, destination: Path) -> str:
    return Path(os.path.relpath((source / value).resolve(), destination.resolve())).as_posix()


def build_attribution_pairs(core: Path, out: Path, *, scenario_profile: str = PROFILE) -> dict:
    """Publish healthy/physical-only/meter-only/mixed matched diagnostic arms."""
    source_rows = read_jsonl(core / "manifest.jsonl")
    for row in source_rows:
        if isinstance(row["z"], str):
            row["z"] = json.loads((core / row["z"]).read_text(encoding="utf-8"))
    by_window = {row["window_id"]: row for row in source_rows}
    healthy = {row["parent_id"]: row for row in source_rows if not row["families"]}
    rows, groups = [], []
    for mixed in source_rows:
        if len(mixed["families"]) != 2 or "measurement" not in mixed["families"]:
            continue
        component = mixed["offline_metadata"]["component_core_audit"]["source_window_id"]
        physical = by_window[component]
        control = healthy[mixed["parent_id"]]
        offset = np.asarray(mixed["z"]) - np.asarray(physical["z"])
        meter_only = copy.deepcopy(control)
        meter_only.update(z=(np.asarray(control["z"]) + offset).tolist(), families=["measurement"],
                          severity="matched_meter_only")
        group_id = f"{mixed['window_id']}:attribution"
        for role, source in (("healthy", control), ("physical_only", physical),
                             ("meter_only", meter_only), ("mixed", mixed)):
            row = copy.deepcopy(source)
            row.update(window_id=f"{group_id}:{role}", noise_group_id=group_id, noise_seed=mixed["noise_seed"],
                       noise_replicates=mixed["noise_replicates"])
            case = _load_case(core / row["case"]) if isinstance(row["case"], str) else row["case"]
            row["case"] = jsonable(case)
            sigma = measurement_sigma(14, 20)
            row["measurement_sigma"] = sigma.tolist()
            row["offline_metadata"] = {"profile_id": scenario_profile, "cohort": "paired_error_attribution",
                "attribution_group_id": group_id, "attribution_role": role,
                "source_mixed_window_id": mixed["window_id"],
                "source_physical_window_id": component, "source_healthy_window_id": control["window_id"],
                "meter_offsets_pu": offset.tolist(), "same_standard_noise_across_four_arms": True,
                "noiseless_wls_audit": exact_wls_audit(case, row["z"], sigma)}
            rows.append(row)
        groups.append({"group_id": group_id, "physical_family": physical["families"][0],
                       "split": mixed["split"], "parent_id": mixed["parent_id"]})
    path = out / "paired_attribution_manifest.jsonl"
    write_jsonl(path, rows)
    return {"path": str(path), "rows": len(rows), "groups": groups, "sha256": file_sha256(path),
            "included_in_main_training_manifest": False}


def build_energy_balanced_training_view(core: Path, out: Path, *, seed: int) -> dict:
    """Optional offline sampling view; the complete population remains intact."""
    bins = ("below_1", "1_to_9", "9_to_25", "above_25")
    groups = {key: [] for key in bins}
    healthy, unavailable = [], []
    for row in read_jsonl(core / "manifest.jsonl"):
        if row["split"] != "train":
            continue
        if not row["families"]:
            healthy.append(row)
            continue
        key = (row["offline_metadata"].get("noiseless_wls_audit") or {}).get("residual_visible_energy_bin")
        (groups[key] if key in groups else unavailable).append(row)
    nonempty = [values for values in groups.values() if values]
    count = min(map(len, nonempty)) if nonempty else 0
    rng = np.random.default_rng(seed)
    selected = list(healthy)
    for key, candidates in groups.items():
        if candidates:
            for index in rng.choice(len(candidates), size=count, replace=False):
                selected.append(candidates[int(index)])
    rewritten = []
    for original in selected:
        row = copy.deepcopy(original)
        if isinstance(row["z"], str):
            row["z"] = json.loads((core / row["z"]).read_text(encoding="utf-8"))
        if isinstance(row["case"], str):
            row["case"] = _relative_reference(row["case"], core, out)
        row["measurement_sigma"] = measurement_sigma(14, 20).tolist()
        metadata = row["offline_metadata"]
        if metadata.get("physical_audit_path"):
            metadata["physical_audit_path"] = _relative_reference(metadata["physical_audit_path"], core, out)
        metadata["offline_sampling_view"] = "equal_count_per_available_noiseless_energy_bin"
        rewritten.append(row)
    path = out / "energy_balanced_training_manifest.jsonl"
    write_jsonl(path, rewritten)
    report = {"path": str(path), "rows": len(rewritten), "sha256": file_sha256(path),
        "source_fault_counts_by_bin": {key: len(values) for key, values in groups.items()},
        "selected_faults_per_available_bin": count, "missing_bins": [key for key in bins if not groups[key]],
        "healthy_rows_retained": len(healthy), "unavailable_energy_rows_retained_in_full_source": len(unavailable),
        "all_four_bins_populated": all(groups.values()), "seed": seed,
        "selected_family_counts": dict(Counter(f for row in rewritten for f in row["families"])),
        "automatic_training_enabled": False, "main_population_modified": False}
    write_json(out / "energy_balanced_training_report.json", report)
    return report


def build_accuracy_views(core: Path, out: Path, *, noise_profiles=NOISE_PROFILES[1:]) -> dict:
    """Freeze physical means, split membership and fault labels across sensor profiles."""
    reports = {}
    manifests = sorted(core.glob("*manifest.jsonl"))
    healthy = {row["parent_id"]: row["z"] for row in _healthy_parents(core)}
    for profile in noise_profiles:
        destination = out / profile
        destination.mkdir(parents=True, exist_ok=False)
        sigma = measurement_sigma(14, 20, noise_profile=profile)
        write_json(destination / "measurement_sigma.json", sigma)
        report = {"noise_profile": profile, "physical_population_frozen": True,
                  "main_admission_frozen_from": "baseline", "manifests": {}}
        for manifest in manifests:
            rows = read_jsonl(manifest)
            rewritten = []
            for original in rows:
                row = copy.deepcopy(original)
                if row.get("measurement_kind") != "noiseless_mean":
                    raise ValueError("Accuracy views require exact means; observed data must never be re-noised")
                case = _load_case(core / row["case"]) if isinstance(row["case"], str) else row["case"]
                if isinstance(row["case"], str):
                    row["case"] = _relative_reference(row["case"], core, destination)
                if isinstance(row.get("z"), str):
                    row["z"] = json.loads((core / row["z"]).read_text())
                row["measurement_sigma"] = "measurement_sigma.json"
                audit = row.setdefault("offline_metadata", {})
                if audit.get("physical_audit_path"):
                    audit["physical_audit_path"] = _relative_reference(audit["physical_audit_path"], core, destination)
                audit["source_population_noise_profile"] = "baseline"
                audit["noise_profile"] = profile
                audit["accuracy_view"] = {"source_manifest_sha256": file_sha256(manifest),
                    "physical_mean_sha256": content_hash(row["z"]),
                    "cohort_membership_reselected": False,
                    "same_standard_noise_seed_as_baseline": True,
                    "injected_gross_error_pu_unchanged": True}
                # Old baseline audit stays explicitly named; no stale J is
                # presented as if computed under the smaller covariance.
                audit["source_observable_strength"] = audit.pop("observable_strength", None)
                audit["source_noiseless_wls_audit"] = audit.pop("noiseless_wls_audit", None)
                audit["observable_strength"] = exact_wls_audit(case, row["z"], sigma)
                audit["noiseless_wls_audit"] = copy.deepcopy(audit["observable_strength"])
                audit["residual_visible_energy_bin"] = audit["observable_strength"]["residual_visible_energy_bin"]
                audit["source_paired_visibility"] = audit.pop("paired_visibility", None)
                audit["paired_visibility"] = paired_visibility(healthy[row["parent_id"]], row["z"], sigma)
                audit["admission_covariance_profile"] = "baseline"
                audit["settings_sigma_reference_noise_profile"] = "baseline"
                rewritten.append(row)
            target = destination / manifest.name
            write_jsonl(target, rewritten)
            report["manifests"][manifest.name] = {"rows": len(rewritten), "sha256": file_sha256(target)}
        write_json(destination / "accuracy_view_report.json", report)
        reports[profile] = report
    return reports


def build_node_breaker_companion(out: Path, *, seed: int, parents_by_split: dict[str, int],
                                 noise_profiles=NOISE_PROFILES, attempts: int = 12,
                                 scenario_profile: str = PROFILE) -> dict:
    """Keep binary breaker physics and low-impact valid events, with exact covariance."""
    from pypower.api import case14
    from Transmission.generate_measurements import solve_ac_opf
    from Transmission.ieee14_full_topology import build_full_topology
    from Transmission.ieee14_full_measurements import flipped_case, main_section_nodes, single_flip_catalogue
    from Transmission.ieee14_full_substation import (
        add_telemetry_noise, operator_model_from_map, operator_noise_for_layout,
        operator_vector_for_layout, solve_node_breaker, substation_telemetry,
    )
    rng = np.random.default_rng(seed)
    model = build_full_topology()
    catalogue = single_flip_catalogue(model)
    rows, failures = [], []
    for split, count in parents_by_split.items():
        for ordinal in range(count):
            parent_id = f"reviewed_node_breaker_{seed}_{split}_{ordinal:05d}"
            reference = case14()
            if scenario_profile == DEFAULT_PROFILE:
                from three_phase_model.voltage_bases import apply_ieee14_voltage_bases
                reference = apply_ieee14_voltage_bases(reference)
            load_scale = float(rng.uniform(.8, 1.25))
            reference["bus"][:, 2:4] *= load_scale
            normal_case, normal_info, _ = flipped_case(reference, {}, model=model)
            meter_nodes = main_section_nodes(model, normal_info["node_to_bus"], [])
            reported, layout = operator_model_from_map(reference, model, {}, meter_nodes)
            targets = [("healthy", {})]
            if split != "calibration":
                targets += [(category, None) for category in ("dangling_line_terminal", "bus_split")]
            for category, fixed_status in targets:
                candidates = [None] if fixed_status is not None else [
                    e for e in catalogue if e["category"] == category]
                if fixed_status is None:
                    rng.shuffle(candidates)
                for candidate in candidates[:attempts]:
                    status = {} if candidate is None else {candidate["cb_name"]: candidate["true_closed"]}
                    try:
                        true_case, info, removed = flipped_case(reference, status, model=model)
                        if removed["dead_buses"]:
                            raise ValueError("breaker flip islands equipment")
                        dispatch = solve_ac_opf(copy.deepcopy(true_case))
                        if dispatch is None or not dispatch.get("success"):
                            raise ValueError("true contracted topology OPF failed")
                        solved, physical_info, removed = solve_node_breaker(
                            model, reference, status, dispatch, info["node_to_bus"])
                        if solved is None or removed["dead_buses"]:
                            raise ValueError("detailed node/breaker power flow failed or islands equipment")
                        exact = substation_telemetry(solved, model, reference, physical_info, removed)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        failures.append({"parent_id": parent_id, "category": category,
                                         "candidate": candidate, "reason": str(exc)})
                        continue
                    mean = operator_vector_for_layout(exact, layout)
                    noise_seed = int(rng.integers(0, 2**63 - 1))
                    for noise_profile in noise_profiles:
                        source = copy.deepcopy(exact)
                        pq_sigma = float(measurement_sigma(14, 20, noise_profile)[14])
                        source["sigma"].update(vm=.001, inj=pq_sigma, flow=pq_sigma)
                        source["nominal_sensor_sigma"] = dict(source["sigma"])
                        telemetry = add_telemetry_noise(source, np.random.default_rng(noise_seed))
                        contract = operator_noise_for_layout(telemetry, layout)
                        observation = operator_vector_for_layout(telemetry, layout)
                        exact_rows = contract["structural_zero_indices"]
                        sigma = contract["measurement_sigma"]
                        audit = exact_wls_audit(reported, mean, sigma, exact_rows=exact_rows)
                        row = {"schema": "reviewed_node_breaker_scenario_v1", "profile_id": scenario_profile,
                            "parent_id": parent_id, "split": split, "noise_profile": noise_profile,
                            "window_id": f"{parent_id}:{category}:{noise_profile}",
                            "families": [] if candidate is None else ["topology"],
                            "case": jsonable(reported), "measurements": observation.tolist(),
                            "sigma_z": sigma, "measurement_covariance": contract["measurement_covariance"],
                            "structural_zero_indices": exact_rows,
                            "measurement_ids": contract["measurement_ids"],
                            "substation_telemetry": telemetry,
                            "operator_layout": layout,
                            "offline_audit": {"z_exact": mean.tolist(), "observable_strength": audit,
                                "physical_severity": {"effect": category, "breaker": candidate,
                                    "load_scale": load_scale, "status_is_binary": True},
                                "physical_solve_success": True, "admission_uses_wls_alarm": False,
                                "low_signal_events_retained": True, "source_telemetry": source,
                                "operator_layout": layout, "noise_seed": noise_seed}}
                        rows.append(row)
                    break
                else:
                    failures.append({"parent_id": parent_id, "category": category, "coverage_shortfall": True})
    path = out / "node_breaker_scenarios.jsonl"
    write_jsonl(path, rows)
    write_json(out / "node_breaker_proposal_failures.json", failures)
    return {"path": str(path), "rows": len(rows), "sha256": file_sha256(path),
            "proposal_failures": len(failures),
            "coverage_shortfalls": sum(bool(f.get("coverage_shortfall")) for f in failures),
            "scope": "full node/breaker companion, not silently reduced to GNN branch-status labels"}


def admit_training(bundle: Path) -> dict:
    """Default training view requires current WLS evidence and expert action."""
    from research.filter_reviewed_training import filter_bundle_training
    return filter_bundle_training(bundle, bundle / "training_admission")


def build_bundle(output_dir: str | Path, *, parents_by_split: dict[str, int], seed: int = 20260917,
                 stage: str = "early", attempt_cap: int = 24, noise_replicates: int = 2,
                 include_node_breaker: bool = True, training_admission: bool = True,
                 scenario_profile: str = DEFAULT_PROFILE) -> dict:
    from research.gnn_screen.practical_corpus import generate_corpus
    if stage not in {"early", "full"}:
        raise ValueError("stage must be early or full")
    if set(parents_by_split) != {"train", "validation", "calibration", "test"} or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in parents_by_split.values()
    ):
        raise ValueError("bundle requires a positive parent count in all four disjoint splits")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "fault_profile.json", get_fault_profile(scenario_profile))
    core = out / "baseline" / "core"
    core_report = generate_corpus(core, parents_by_split=parents_by_split, seed=seed,
        noise_replicates=noise_replicates, healthy_calibration_replicates=max(10, noise_replicates),
        attempt_cap=attempt_cap, scenario_profile=scenario_profile, stage=stage, noise_profile="baseline")
    harmonic = build_harmonic_companion(core, out, seed=seed + 1, scenario_profile=scenario_profile)
    chains = build_measurement_chain_companion(core, out, seed=seed + 2, scenario_profile=scenario_profile)
    attribution = build_attribution_pairs(core, out, scenario_profile=scenario_profile)
    energy_balanced = build_energy_balanced_training_view(core, out, seed=seed + 4)
    node_breaker = (build_node_breaker_companion(out, seed=seed + 3, parents_by_split=parents_by_split, scenario_profile=scenario_profile)
                    if include_node_breaker else {"skipped_by_request": True})
    accuracy = build_accuracy_views(core, out / "accuracy_views")
    repository = Path(__file__).resolve().parents[1]
    source_files = ["research/reviewed_fault_scenarios.py", "psse_env/fault_profiles.py",
        "research/gnn_screen/dataset.py", "Transmission/generate_hse_traces.py",
        "Transmission/ieee14_full_substation.py", "Transmission/ieee14_full_topology.py",
        "Transmission/ieee14_full_measurements.py", "tools/lagrangian_port.py"]
    report = {"schema": "reviewed_fault_scenario_bundle_v1", "profile_id": scenario_profile, "seed": seed,
        "generation_completed": True,
        "implementation_sha256": {name: file_sha256(repository / name) for name in source_files},
        "curriculum_stage": stage, "parents_by_split": parents_by_split, "core": core_report,
        "harmonic": harmonic, "measurement_chain": chains, "paired_attribution": attribution,
        "node_breaker": node_breaker, "energy_balanced_training": energy_balanced,
        "accuracy_views": accuracy, "training_performed": False, "detector_improvement_claimed": False,
        "coverage_complete": (not any(value for key, value in core_report.get("counts", {}).items()
                                      if key.endswith(":unfilled_slots"))
                              and not node_breaker.get("coverage_shortfalls", 0)),
        "historical_benchmarks_overwritten": False,
        "comparison_contract": {"population_effect": "evaluate a frozen model and threshold across separately named populations",
            "training_effect": "compare old and revised models on identical frozen test manifests, including weak cases",
            "noise_effect": "same physical means and gross-error values; matched noise and covariance for every profile"},
        "limitations": ["This is a reproducible scenario revision, not a field-event distribution or accuracy certificate",
            "Harmonic and node/breaker companions have explicit adapters; the existing five-family GNN label schema is unchanged",
            "Noiseless WLS energy is offline and unavailable on solver failure; it never becomes an online feature"]}
    write_json(out / "bundle_report.json", report)
    if training_admission:
        report["training_admission"] = admit_training(out)
        report["default_training_inputs"] = {name: details["outputs"]["filtered_manifest"]
            for name, details in report["training_admission"]["profiles"].items()}
        write_json(out / "bundle_report.json", report)
    else:
        report["training_admission"] = {"performed": False, "raw_reviewed_manifests_are_not_training_qualified": True}
        write_json(out / "bundle_report.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    for split in ("train", "validation", "calibration", "test"):
        parser.add_argument(f"--{split}-parents", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--scenario-profile", choices=(PROFILE, DEFAULT_PROFILE), default=DEFAULT_PROFILE)
    parser.add_argument("--curriculum-stage", choices=("early", "full"), default="early")
    parser.add_argument("--attempt-cap", type=int, default=24)
    parser.add_argument("--noise-replicates", type=int, default=2)
    parser.add_argument("--skip-node-breaker", action="store_true")
    parser.add_argument("--no-training-admission", action="store_true",
                        help="Generate the raw physical population only; it remains unqualified for training")
    args = parser.parse_args(argv)
    report = build_bundle(args.output_dir, parents_by_split={
        split: getattr(args, f"{split}_parents") for split in ("train", "validation", "calibration", "test")},
        seed=args.seed, stage=args.curriculum_stage, attempt_cap=args.attempt_cap,
        noise_replicates=args.noise_replicates, include_node_breaker=not args.skip_node_breaker,
        training_admission=not args.no_training_admission, scenario_profile=args.scenario_profile)
    print(json.dumps({"profile_id": report["profile_id"], "report": str(Path(args.output_dir) / "bundle_report.json")}, indent=2))


if __name__ == "__main__":
    main()
