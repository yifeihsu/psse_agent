"""Training-only admission from actual WLS observations and executed expert actions.

Evaluation parents and weak-fault labels remain unchanged. Accepted noisy
replicas are frozen as observations so loading cannot redraw their noise and
invalidate the evidence used to admit them. This certifies an executable
observable expert prefix, not a completed physical diagnosis or repair.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2

from research.gnn_screen.dataset import ROW_KEYS, content_hash, file_sha256, jsonable, load_manifest, write_json
from research.gnn_screen.wls_features import build_wls_features
from research.reviewed_fault_scenarios import read_jsonl, write_jsonl

CONTRACT = "wls_observable_expert_prefix_v1"
THRESHOLDS = {"chi_square_alpha": .01, "normalized_residual": 4.0}


def wls_detection(case: dict, observed: Any, sigma: Any, *, exact_rows=()) -> dict:
    """The unchanged inclusive dual alarm on the actual noisy observation."""
    try:
        if exact_rows:
            from tools.lagrangian_port import lagrangian_m_singlephase_details
            numeric = copy.deepcopy(case)
            for key in ("bus", "branch", "gen"):
                if key in numeric:
                    numeric[key] = np.asarray(numeric[key], dtype=float)
            numeric["bus"][:, 7], numeric["bus"][:, 8] = 1.0, 0.0
            result = lagrangian_m_singlephase_details(np.asarray(observed), numeric, 0, numeric["bus"],
                measurement_sigma=np.asarray(sigma), exact_measurement_indices=list(exact_rows), max_it=30, tol=1e-8)
            if not result["success"]:
                raise ValueError("constrained WLS did not converge")
            residuals = result["r_norm"]
        else:
            result = build_wls_features(case, observed, measurement_sigma=sigma, max_it=30, tol=1e-8)
            residuals = result["signed_normalized_residual"]
        J = float(result["wls_objective"])
        local = float(np.max(np.abs(residuals)))
        threshold = float(chi2.ppf(1 - THRESHOLDS["chi_square_alpha"], result["dof"]))
        return {"success": True, "J": J, "threshold": threshold, "dof": int(result["dof"]),
                "max_normalized_residual": local, "chi_square_alarm": J >= threshold,
                "normalized_residual_alarm": local >= THRESHOLDS["normalized_residual"],
                "alarm": bool(J >= threshold or local >= THRESHOLDS["normalized_residual"])}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return {"success": False, "alarm": None, "error": str(exc)}


def materialize_training_windows(row: dict) -> list[dict]:
    """Match dataset.prepare_corpus's stream, then remove resampling controls."""
    source = {key: copy.deepcopy(value) for key, value in row.items() if key in ROW_KEYS}
    kind = source.get("measurement_kind", "observed")
    reps = source.get("noise_replicates", 1)
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        raise ValueError("noise_replicates must be a positive integer")
    z, sigma = np.asarray(source["z"], dtype=float), np.asarray(source["measurement_sigma"], dtype=float)
    if kind == "observed" and (reps != 1 or "noise_seed" in source or "noise_group_id" in source):
        raise ValueError("observed rows cannot carry fresh-noise settings")
    if kind not in {"observed", "noiseless_mean"}:
        raise ValueError("unsupported measurement_kind")
    if kind == "noiseless_mean" and "noise_seed" not in source:
        raise ValueError("noiseless means require an explicit noise seed")
    seed = int(content_hash({"seed": source.get("noise_seed", 0), "parent": source["parent_id"],
                            "window": source.get("noise_group_id", source["window_id"])})[:16], 16)
    rng = np.random.default_rng(seed)
    windows = []
    for replicate in range(reps):
        observed = z + rng.normal(size=z.shape) * sigma if kind == "noiseless_mean" else z.copy()
        fixed = copy.deepcopy(source)
        fixed.update(z=observed.tolist(), measurement_kind="observed", noise_replicates=1,
                     window_id=source["window_id"] + (f":noise{replicate}" if kind == "noiseless_mean" else ""))
        fixed.pop("noise_seed", None)
        fixed.pop("noise_group_id", None)
        fixed.setdefault("offline_metadata", {})["admission_source"] = {
            "source_window_id": source["window_id"], "source_measurement_kind": kind,
            "source_noise_seed": source.get("noise_seed"), "source_noise_group_id": source.get("noise_group_id"),
            "replicate": replicate, "source_mean_sha256": content_hash(source["z"]),
            "noise_redraw_allowed": False}
        windows.append(fixed)
    return windows


def mixed_component_checks(source: dict, fixed: dict, by_window: dict, healthy_by_parent: dict) -> dict:
    """Offline counterfactual alarms; these vectors never enter expert input."""
    families = source["families"]
    if len(families) < 2:
        return {"passed": True, "required": False}
    if "measurement" not in families or len(families) != 2:
        return {"passed": False, "required": True, "reason": "unsupported_mixed_component_decomposition"}
    component = source.get("offline_metadata", {}).get("component_core_audit") or {}
    physical = by_window.get(component.get("source_window_id"))
    healthy = healthy_by_parent.get(source["parent_id"])
    if physical is None or healthy is None or source.get("measurement_kind") != "noiseless_mean":
        return {"passed": False, "required": True, "reason": "mixed_component_reference_missing"}
    if physical.get("measurement_kind") != "noiseless_mean" or healthy.get("measurement_kind") != "noiseless_mean":
        return {"passed": False, "required": True, "reason": "mixed_counterfactual_requires_noiseless_references"}
    if physical["parent_id"] != source["parent_id"] or physical["split"] != source["split"]:
        raise ValueError("mixed component source crosses physical parent or dataset split")
    if content_hash(physical["case"]) != content_hash(source["case"]) or content_hash(healthy["case"]) != content_hash(source["case"]):
        raise ValueError("mixed counterfactuals must share the reported model")
    noise = np.asarray(fixed["z"]) - np.asarray(source["z"])
    offset = np.asarray(source["z"]) - np.asarray(physical["z"])
    physical_observation = np.asarray(physical["z"]) + noise
    meter_observation = np.asarray(healthy["z"]) + offset + noise
    p = wls_detection(source["case"], physical_observation, fixed["measurement_sigma"])
    m = wls_detection(source["case"], meter_observation, fixed["measurement_sigma"])
    passed = bool(p["success"] and m["success"] and p["alarm"] and m["alarm"])
    return {"passed": passed, "required": True, "physical_only": p, "meter_only": m,
            "same_noise_as_mixed": True, "source_physical_window_id": physical["window_id"],
            "reason": None if passed else "at_least_one_mixed_component_is_wls_quiet_or_unavailable"}


def _export_action(event: dict, fixed: dict) -> tuple[dict, dict]:
    """Use the real controller/canonical SFT converter and its alias validation."""
    from psse_env.dagger.dataset_builder import examples_to_chat_sft, validate_policy_payload
    observation = copy.deepcopy(event["policy_observation"])
    validate_policy_payload(observation)
    action = copy.deepcopy(event["preferred_action"])
    example = {"example_id": fixed["window_id"] + ":evidenced_action", "parent_id": fixed["parent_id"],
               "scenario_id": fixed["window_id"], "root_scenario_id": fixed["parent_id"],
               "physical_root_fingerprint": content_hash({"parent": fixed["parent_id"],
                   "source_window": fixed.get("offline_metadata", {}).get("admission_source", {}).get("source_window_id", fixed["window_id"])}),
               "dataset_split": "train", "dataset_source": CONTRACT,
               "policy_observation": observation, "preferred_action": action,
               "valid_next_actions": copy.deepcopy(event["ordered_actions"]),
               "training_scope": "executed_expert_prefix_only"}
    rows = examples_to_chat_sft([example], protocol="canonical", alias_before_compaction=True,
                               require_derived_provenance=True)
    if len(rows) != 1:
        raise ValueError("qualified expert action did not produce exactly one canonical SFT row")
    rows[0].setdefault("metadata", {}).update(
        scope="executed_expert_prefix_only", complete_repair_verified=False,
        parent_id=fixed["parent_id"], split="train", training_admission_contract=CONTRACT,
        measurement_sha256=content_hash(fixed["z"]), case_sha256=content_hash(fixed["case"]),
        sigma_sha256=content_hash(fixed["measurement_sigma"]))
    return example, rows[0]


def reported_runtime_case(manifest: Path, row: dict) -> tuple[dict, dict]:
    """Restore public operator bounds omitted by the GNN input whitelist.

    Every variant of a parent uses the same reported parent model. Never use
    actual_physical_model_path (the private changed network) for this purpose.
    Estimated/OPF bus voltages and angles are discarded before the WLS probe.
    """
    configured = row["case"]
    runtime = copy.deepcopy(configured)
    parent_path = None
    audit_reference = row.get("offline_metadata", {}).get("physical_audit_path")
    if audit_reference:
        candidate = (manifest.parent / audit_reference).resolve().parent / "model/source_case.json"
        if candidate.is_file():
            runtime = json.loads(candidate.read_text(encoding="utf-8"))
            parent_path = str(candidate)
            source_bus = np.asarray(runtime["bus"], dtype=float)
            configured_bus = np.asarray(configured["bus"], dtype=float)
            if (source_bus.shape[0] != configured_bus.shape[0]
                or not np.array_equal(source_bus[:, [0, 1, 4, 5]], configured_bus[:, [0, 1, 4, 5]])
                or not np.array_equal(np.asarray(runtime["branch"])[:, :13], np.asarray(configured["branch"])[:, :13])):
                raise ValueError("reported parent model disagrees with configured network; refusing private-case substitution")
            known_voltage = configured_bus[:, 9] > 0
            if np.any(known_voltage) and not np.array_equal(source_bus[known_voltage, 9], configured_bus[known_voltage, 9]):
                raise ValueError("reported parent voltage bases disagree with the configured network")
            runtime["branch"] = copy.deepcopy(configured["branch"])
    bus = np.asarray(runtime["bus"], dtype=float).copy()
    bus[:, 7], bus[:, 8] = 1.0, 0.0
    runtime["bus"] = bus.tolist()
    return runtime, {"reported_parent_model": parent_path, "private_fault_model_used": False,
                     "bus_state_initialization": "flat", "runtime_case_sha256": content_hash(runtime)}


def _serializable_row(row: dict, *, source_dir: Path | None = None, destination: Path | None = None) -> dict:
    # load_manifest has already resolved case/z/sigma file references, keeping
    # held-out means, IDs, splits and seeds unchanged when the manifest moves.
    result = jsonable({key: copy.deepcopy(value) for key, value in row.items() if key in ROW_KEYS})
    audit = result.get("offline_metadata", {})
    if source_dir is not None and destination is not None and audit.get("physical_audit_path"):
        audit["physical_audit_path"] = Path(os.path.relpath(
            (source_dir / audit["physical_audit_path"]).resolve(), destination)).as_posix()
    return result


def filter_training_manifest(input_manifest: str | Path, output_dir: str | Path, *,
                             probe: Callable | None = None, context_builder: Callable | None = None) -> dict:
    if probe is None:
        from research.reviewed_expert_admission import probe_expert_action
        probe = probe_expert_action
    if context_builder is None:
        from research.reviewed_observable_context import build_observable_context
        context_builder = build_observable_context
    source_path = Path(input_manifest).resolve()
    rows = load_manifest(source_path)
    if any(row.get("split") not in {"train", "validation", "calibration", "test"} for row in rows):
        raise ValueError("training admission requires explicit preassigned physical-parent splits")
    if any(not isinstance(row["families"], list) for row in rows):
        raise ValueError("training admission requires exhaustive family labels for offline positive/control selection")
    parent_splits = {}
    noise_groups = {}
    for row in rows:
        if row["parent_id"] in parent_splits and parent_splits[row["parent_id"]] != row["split"]:
            raise ValueError("physical parent crosses dataset splits")
        parent_splits[row["parent_id"]] = row["split"]
        if "noise_group_id" in row:
            group = (row["parent_id"], row["noise_group_id"])
            contract = (row.get("noise_seed"), row.get("noise_replicates", 1), len(row["z"]))
            if group in noise_groups and noise_groups[group] != contract:
                raise ValueError("paired noise group disagrees on seed, replicate count, or sensor dimension")
            noise_groups[group] = contract
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    by_window = {row["window_id"]: row for row in rows}
    healthy = {row["parent_id"]: row for row in rows if not row["families"]}
    kept, excluded, ledger, examples, chat = [], [], [], [], []
    counts = Counter()
    held_out = []
    for source in rows:
        if source["split"] != "train":
            held_out.append(_serializable_row(source, source_dir=source_path.parent, destination=out))
            continue
        for fixed in materialize_training_windows(source):
            family = "+".join(sorted(fixed["families"])) or "healthy"
            counts[f"attempted:{family}"] += 1
            detection = wls_detection(fixed["case"], fixed["z"], fixed["measurement_sigma"])
            record = {"window_id": fixed["window_id"], "parent_id": fixed["parent_id"], "split": "train",
                      "families": fixed["families"], "detection": detection, "eligible": False}
            reason = None
            if not detection["success"]:
                reason = "wls_failed"
            elif fixed["families"] and not detection["alarm"]:
                reason = "wls_quiet_fault"
            elif not fixed["families"] and detection["alarm"]:
                reason = "healthy_false_alarm"
            component_checks = {"passed": True, "required": False}
            if reason is None and fixed["families"]:
                component_checks = mixed_component_checks(source, fixed, by_window, healthy)
                if not component_checks["passed"]:
                    reason = "mixed_component_not_detectable"
            record["component_checks"] = component_checks
            receipt = None
            event = None
            if reason is None:
                try:
                    sensor_seed = int(content_hash({"window": fixed["window_id"], "purpose": "auxiliary_sensor_noise"})[:16], 16)
                    metadata, context_receipt = context_builder(source_path, source, noise_seed=sensor_seed)
                    record["context_generation"] = context_receipt
                    runtime_case, runtime_receipt = reported_runtime_case(source_path, source)
                    record["runtime_model"] = runtime_receipt
                    receipt = probe(runtime_case, fixed["z"], fixed["measurement_sigma"],
                                    observable_metadata=metadata, max_actions=40)
                    record["expert_probe"] = receipt
                    observed_wls = receipt["initial_wls"]
                    if (not observed_wls["success"] or observed_wls["alarm"] != detection["alarm"]
                        or not np.isclose(observed_wls["J"], detection["J"], rtol=1e-5, atol=1e-6)):
                        reason = "expert_runtime_wls_disagrees_with_admission"
                    elif fixed["families"] and not receipt["fault_actionable"]:
                        reason = "expert_no_executable_evidenced_action"
                    elif not fixed["families"] and not receipt["healthy_completion_valid"]:
                        reason = "healthy_expert_completion_unverified"
                    else:
                        event = receipt.get("actionable_event")
                        if event is None and receipt["healthy_completion_valid"]:
                            event = receipt["events"][-1]
                        if not event or not all(event.get(key) for key in ("process_evidence_valid", "action_executed", "execution_success")):
                            reason = "expert_action_evidence_incomplete"
                        else:
                            example, sft = _export_action(event, fixed)
                except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                    reason = "expert_or_export_unavailable"
                    record["error"] = str(exc)
            if reason is None:
                kind = "fault_actionable" if fixed["families"] else "healthy_completion"
                admission = {"contract": CONTRACT, "scope": "executed_expert_prefix_only", "eligible": True,
                    "kind": kind, "wls_alarm": detection["alarm"], "expert_valid": True,
                    "action_executed": True, "execution_success": True,
                    "safe_finalize": bool(receipt["healthy_completion_valid"]),
                    "component_checks_passed": component_checks["passed"], "thresholds": dict(THRESHOLDS),
                    "measurement_sha256": content_hash(fixed["z"]), "case_sha256": content_hash(fixed["case"]),
                    "sigma_sha256": content_hash(fixed["measurement_sigma"]),
                    "selected_action": event["preferred_action"],
                    "complete_diagnosis_or_repair_verified": False}
                metadata = fixed.setdefault("offline_metadata", {})
                metadata.update(scenario_profile=metadata.get("scenario_profile", metadata.get("profile_id", "reviewed_v1")),
                                training_admission=admission)
                kept.append(_serializable_row(fixed, source_dir=source_path.parent, destination=out))
                examples.append(example)
                chat.append(sft)
                record.update(eligible=True, reason=kind, admission=admission)
                counts[f"kept:{family}"] += 1
            else:
                fixed.setdefault("offline_metadata", {})["training_exclusion"] = {
                    "contract": CONTRACT, "reason": reason, "physical_fault_labels_retained": True,
                    "held_out_evaluation": False}
                excluded.append(_serializable_row(fixed, source_dir=source_path.parent, destination=out))
                record["reason"] = reason
                counts[f"excluded:{family}:{reason}"] += 1
            ledger.append(record)
    paths = {"training_manifest": out / "training_manifest.jsonl", "filtered_manifest": out / "filtered_manifest.jsonl",
             "held_out_manifest": out / "held_out_manifest.jsonl", "excluded_training_manifest": out / "excluded_training_manifest.jsonl",
             "admission_ledger": out / "admission_ledger.jsonl", "expert_examples": out / "expert_examples.jsonl",
             "expert_chat_sft": out / "expert_prefixes.chat_sft.jsonl"}
    for name, payload in (("training_manifest", kept), ("filtered_manifest", kept + held_out),
                          ("held_out_manifest", held_out), ("excluded_training_manifest", excluded),
                          ("admission_ledger", ledger), ("expert_examples", examples), ("expert_chat_sft", chat)):
        write_jsonl(paths[name], payload)
    report = {"contract": CONTRACT, "scope": "executed_expert_prefix_only", "thresholds": dict(THRESHOLDS),
        "source_manifest": str(source_path), "source_manifest_sha256": file_sha256(source_path),
        "training_noisy_windows_considered": len(kept) + len(excluded), "training_windows_kept": len(kept),
        "training_windows_excluded": len(excluded), "held_out_source_windows_unchanged": len(held_out),
        "expert_prefix_targets": len(chat), "counts": dict(counts), "outputs": {k: str(v) for k, v in paths.items()},
        "output_sha256": {k: file_sha256(v) for k, v in paths.items()},
        "evaluation_selection_applied": False, "excluded_training_parents_are_not_held_out": True,
        "training_performed": False, "complete_episode_success_claimed": False,
        "auxiliary_measurements": "fresh declared-noise observations from replayed same physical circuit; no labels or external fault flags in expert input"}
    report["retained_training_family_counts"] = dict(Counter(
        family for row in kept for family in (row["families"] or ["healthy"])))
    report["gnn_required_head_coverage"] = {
        "phase_positive": any(set(row["families"]) & {"hif", "unbalance"} for row in kept),
        "phase_negative": any(not set(row["families"]) & {"hif", "unbalance"} for row in kept),
        "anomaly_positive": any(row["families"] for row in kept),
        "healthy_negative": any(not row["families"] for row in kept)}
    report["gnn_training_head_coverage_sufficient"] = all(report["gnn_required_head_coverage"].values())
    report["sft_prefix_training_nonempty"] = bool(chat)
    write_json(out / "admission_report.json", report)
    return report


def filter_bundle_training(bundle_dir: str | Path, output_dir: str | Path) -> dict:
    bundle, out = Path(bundle_dir).resolve(), Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {"baseline": bundle / "baseline/core/manifest.jsonl",
               "accuracy_005": bundle / "accuracy_views/accuracy_005/manifest.jsonl",
               "accuracy_002": bundle / "accuracy_views/accuracy_002/manifest.jsonl"}
    reports = {profile: filter_training_manifest(path, out / profile) for profile, path in sources.items()}
    companions = filter_companion_training(bundle, out / "companions")
    report = {"contract": CONTRACT, "source_bundle": str(bundle), "profiles": reports,
              "companions": companions,
              "thresholds_retuned": False, "noise_profile_selected_automatically": False,
              "scope": "main and companion training-only views; original evaluation/sensitivity/stress files retained"}
    write_json(out / "bundle_training_admission.json", report)
    return report


def filter_companion_training(bundle: Path, out: Path, *, probe: Callable | None = None) -> dict:
    """Harmonic/full-breaker schemas stay separate from the five-family GNN."""
    if probe is None:
        from research.reviewed_expert_admission import probe_expert_action
        probe = probe_expert_action
    from Transmission.ieee14_full_topology import build_full_topology
    from Transmission.ieee14_full_substation import status_labels, operator_noise_for_layout
    out.mkdir(parents=True, exist_ok=False)
    core_manifest = bundle / "baseline/core/manifest.jsonl"
    parents = {row["parent_id"]: row for row in load_manifest(core_manifest) if not row["families"]}
    reports = {}
    for name in ("harmonic", "node_breaker"):
        source = bundle / f"{name}_scenarios.jsonl"
        if not source.is_file():
            reports[name] = {"source_available": False, "training_rows_kept": 0,
                             "reason": "companion_not_generated_in_source_bundle"}
            continue
        records = read_jsonl(source)
        kept, rejected, held, ledger, chat = [], [], [], [], []
        counts = Counter()
        for row in records:
            if row["split"] != "train":
                held.append(row)
                continue
            counts[f"attempted:{row['noise_profile']}:{'fault' if row['families'] else 'healthy'}"] += 1
            exact = row.get("structural_zero_indices", [])
            detection = wls_detection(row["case"], row["measurements"], row["sigma_z"], exact_rows=exact)
            decision = {"window_id": row["window_id"], "noise_profile": row["noise_profile"], "detection": detection,
                        "eligible": False, "families": row["families"]}
            reason = None
            if not detection["success"]:
                reason = "wls_failed"
            elif row["families"] and not detection["alarm"]:
                reason = "wls_quiet_fault"
            elif not row["families"] and detection["alarm"]:
                reason = "healthy_false_alarm"
            if reason is None:
                try:
                    if name == "harmonic":
                        runtime, model_receipt = reported_runtime_case(core_manifest, parents[row["parent_id"]])
                        metadata = {"harmonic_orders": sorted(map(int, row["harmonic_phasors"])),
                                    "harmonic_measurements": []}
                        for order, sensors in row["harmonic_phasors"].items():
                            for sensor in sensors:
                                vr, vi = sensor["V_complex_noisy"]
                                metadata["harmonic_measurements"].append({"h": int(order),
                                    "bus": int(sensor["bus_1based"]), "V_real": vr, "V_imag": vi,
                                    "sigma": sensor["sigma"], "sigma_semantics": "per_component"})
                    else:
                        runtime = copy.deepcopy(row["case"])
                        metadata = {key: copy.deepcopy(row[key]) for key in (
                            "substation_telemetry", "operator_layout", "structural_zero_indices", "measurement_ids")}
                        metadata.update(reported_breaker_status=status_labels(build_full_topology()),
                            topology_model_id=row["substation_telemetry"]["model_id"],
                            topology_model_fingerprint=row["substation_telemetry"]["model_fingerprint"],
                            operator_noise=operator_noise_for_layout(row["substation_telemetry"], row["operator_layout"]))
                        model_receipt = {"private_fault_status_used": False, "reported_status": "schematic_normal"}
                    metadata["sigma_z"] = row["sigma_z"]
                    receipt = probe(runtime, row["measurements"], row["sigma_z"], observable_metadata=metadata, max_actions=40)
                    decision.update(expert_probe=receipt, runtime_model=model_receipt)
                    measured = receipt["initial_wls"]
                    if (not measured["success"] or measured["alarm"] != detection["alarm"]
                        or not np.isclose(measured["J"], detection["J"], rtol=1e-5, atol=1e-6)):
                        reason = "expert_runtime_wls_disagrees_with_admission"
                    elif row["families"] and not receipt["fault_actionable"]:
                        reason = "expert_no_executable_evidenced_action"
                    elif not row["families"] and not receipt["healthy_completion_valid"]:
                        reason = "healthy_expert_completion_unverified"
                    else:
                        event = receipt.get("actionable_event") or receipt["events"][-1]
                        if not all(event.get(k) for k in ("process_evidence_valid", "action_executed", "execution_success")):
                            reason = "expert_action_evidence_incomplete"
                        else:
                            fixed = {"window_id": row["window_id"], "parent_id": row["parent_id"],
                                "case": row["case"], "z": row["measurements"], "measurement_sigma": row["sigma_z"]}
                            _, sft = _export_action(event, fixed)
                except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                    reason = "expert_or_export_unavailable"
                    decision["error"] = str(exc)
            if reason is None:
                admitted = copy.deepcopy(row)
                admitted["training_admission"] = {"contract": CONTRACT, "scope": "executed_expert_prefix_only",
                    "eligible": True, "thresholds": dict(THRESHOLDS), "wls_alarm": detection["alarm"],
                    "selected_action": event["preferred_action"], "complete_episode_success_claimed": False}
                kept.append(admitted)
                chat.append(sft)
                decision.update(eligible=True, reason="evidenced_expert_prefix" if row["families"] else "healthy_completion")
                counts[f"kept:{row['noise_profile']}:{'fault' if row['families'] else 'healthy'}"] += 1
            else:
                excluded = copy.deepcopy(row)
                excluded["training_exclusion"] = {"contract": CONTRACT, "reason": reason,
                    "physical_fault_labels_retained": True, "held_out_evaluation": False}
                rejected.append(excluded)
                decision["reason"] = reason
                counts[f"excluded:{row['noise_profile']}:{reason}"] += 1
            ledger.append(decision)
        destination = out / name
        destination.mkdir()
        for filename, rows in (("eligible_scenarios.jsonl", kept), ("excluded_training_scenarios.jsonl", rejected),
            ("held_out_scenarios.jsonl", held), ("admission_ledger.jsonl", ledger), ("expert_prefixes.chat_sft.jsonl", chat)):
            write_jsonl(destination / filename, rows)
        report = {"contract": CONTRACT, "schema_adapter": name, "training_rows_kept": len(kept),
            "training_rows_excluded": len(rejected), "held_out_rows_unchanged": len(held),
            "expert_prefix_targets": len(chat), "counts": dict(counts),
            "source_sha256": file_sha256(source), "scope": "executed_expert_prefix_only",
            "not_a_gnn_five_family_manifest": True}
        write_json(destination / "admission_report.json", report)
        reports[name] = report
    return reports


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--bundle")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = (filter_bundle_training(args.bundle, args.output_dir) if args.bundle
              else filter_training_manifest(args.manifest, args.output_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
