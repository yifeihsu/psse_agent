"""Offline physical, observability, and status-identifiability scenario audits.

Only the execution bundle enters the model runtime. True statuses and the
physical operating case are opened after candidate scanning/application, for
scoring and a separately labeled observability calculation.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .estimation import estimate
from .inventory import process_topology, validate_statuses
from .runtime import LogicalTopologyRuntime, evidence_hash


def _read(root: Path, relative: str) -> Any:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _without_branch_status(case: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(case)
    branch = np.asarray(result["branch"]).copy()
    branch[:, 10] = 0
    result["branch"] = branch
    return result


def _estimation_summary(metrics: Mapping[str, Any] | None, *, reason: str | None = None) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        return {"available": False, "converged": None, "observable": None, "plausible": None,
                "failure_reason": reason or "no_numerical_estimate"}
    keys = ("contract", "converged", "observable", "plausible", "failure_reason", "error_detail",
            "candidate_connected", "excluded_by_declared_scope", "rank", "state_dimension",
            "available_measurement_count", "raw_measurement_count", "electrical_bus_count",
            "closed_coupler_nuisance_count", "chi_square_dof", "chi_square_alpha", "wls_objective",
            "chi_square_threshold", "max_normalized_residual", "normalized_residual_threshold",
            "chi_square_alarm", "normalized_residual_alarm", "initialization", "function_evaluations", "numerical_fit_execution")
    return {"available": True, **{key: deepcopy(metrics[key]) for key in keys if key in metrics}}


def _state_error(metrics, model_case, modeled_statuses, physical, true_statuses, inventory, sensors):
    """Compare at physical nodes with available voltage sensors; truth stays offline."""
    state = metrics.get("state") if isinstance(metrics, Mapping) else None
    if not isinstance(state, Mapping) or any(value is None for value in modeled_statuses.values()):
        return {"available": False, "reason": "no_complete_numerical_state_estimate"}
    observed_nodes = [record["node_id"] for record, available in zip(sensors["records"], sensors["available_mask"])
                      if available and record["kind"] == "Vm"]
    observed_nodes = list(dict.fromkeys(observed_nodes))
    if not observed_nodes:
        return {"available": False, "reason": "no_available_voltage_sensor_nodes"}
    true_map = process_topology(physical["operating_case"], inventory, true_statuses)["node_to_row0"]
    candidate_map = process_topology(model_case, inventory, modeled_statuses)["node_to_row0"]
    physical_bus = np.asarray(physical["solution"]["bus"], dtype=float)
    vm = state.get("node_voltage_magnitude_pu", {})
    angles = state.get("electrical_voltage_angles_rad", [])
    try:
        estimated_vm = np.asarray([vm[node] for node in observed_nodes], dtype=float)
        true_vm = np.asarray([physical_bus[true_map[node], 7] for node in observed_nodes])
        estimated_angle = np.asarray([angles[candidate_map[node]] for node in observed_nodes], dtype=float)
        true_angle = np.deg2rad([physical_bus[true_map[node], 8] for node in observed_nodes])
        if not all(np.isfinite(values).all() for values in (estimated_vm, true_vm, estimated_angle, true_angle)):
            raise ValueError("nonfinite state comparison")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return {"available": False, "reason": f"state_comparison_evidence_incomplete:{exc}"}
    references = [index for index, node in enumerate(observed_nodes) if int(physical_bus[true_map[node], 1]) == 3]
    reference_index = references[0] if references else 0
    vm_error = estimated_vm - true_vm
    angle_difference = (estimated_angle - estimated_angle[reference_index]) - (true_angle - true_angle[reference_index])
    angle_error = np.rad2deg(np.angle(np.exp(1j * angle_difference)))
    return {"available": True, "scope": "offline_comparison_at_nodes_with_available_voltage_magnitude_sensors",
            "compared_node_count": len(observed_nodes), "angle_reference_node_id": observed_nodes[reference_index],
            "voltage_magnitude_max_abs_error_pu": float(np.max(np.abs(vm_error))),
            "voltage_magnitude_rmse_pu": float(np.sqrt(np.mean(vm_error**2))),
            "relative_angle_max_abs_error_deg": float(np.max(np.abs(angle_error))),
            "relative_angle_rmse_deg": float(np.sqrt(np.mean(angle_error**2))),
            "state_observable": metrics.get("observable") is True}


def evaluate_scenario(corpus_dir, row, *, max_pairs=5000, scan_pairs=True, estimator=None) -> dict[str, Any]:
    """Audit one retained corpus row, preserving failures and all logical-CB rivals."""
    root = Path(corpus_dir).resolve(strict=True)
    scenario_id = str(row["scenario_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario_id):
        raise ValueError("scenario_id must be a plain artifact identifier")
    admitted = row.get("physical_admission", {}).get("admitted") is True
    report = {"contract": "logical_topology_three_axis_offline_audit_v1", "scenario_id": scenario_id,
              "family": row.get("family"), "layout": row.get("layout"),
              "measurement_profile": row.get("measurement_profile"), "load_scale": row.get("load_scale"),
              "physical_root_fingerprint": row.get("physical_root_fingerprint"),
              "parent_physical_root": row.get("parent_physical_root"), "physical_admitted": admitted,
              "correction_applied": False, "detailed_audit_path": None, "detailed_audit_sha256": None,
              "audit_outcomes": {"physical_feasibility": deepcopy(row.get("physical_admission", {}))},
              "scope": "offline_topology_engineering_audit; no learned_policy_or_legacy_provider_loop"}
    if not admitted:
        report.update(runtime_decision="physical_reject", status_audit=None, preservation=None,
                      initial_estimation=_estimation_summary(None, reason="physical_reject_no_wls"),
                      final_estimation=_estimation_summary(None, reason="physical_reject_no_wls"),
                      state_estimation_error={"initial": None, "final": None}, supported_topology_resolution=False)
        report["audit_outcomes"].update(
            state_observability={"evaluated": False, "reason": "physical_admission_rejected"},
            status_identifiability={"evaluated": False, "reason": "physical_admission_rejected"})
        return report

    # This allowlist is the entire online input boundary. No row truth,
    # erroneous-device list, or solved physical states enter the runtime.
    execution = row["execution"]
    inventory = _read(root, execution["inventory_path"])
    sensors = _read(root, execution["measurement_inventory_path"])
    observations = _read(root, execution["observations_path"])
    model_case = _read(root, execution["base_case_path"])
    numerical_estimator = estimate if estimator is None else estimator
    runtime_options = {} if estimator is None else {"estimator": estimator}
    runtime = LogicalTopologyRuntime(
        inventory=inventory, current_case=model_case, current_statuses=execution["current_statuses"],
        measurement_inventory=sensors, observations=observations,
        chi2_alpha=.05, normalized_residual_threshold=4., **runtime_options,
    )
    before = runtime.snapshot()
    cardinality = int(row.get("hypothesis_cardinality", 1))
    if cardinality not in (1, 2):
        raise ValueError("Only explicitly declared single/pair hypothesis strata are supported")
    include_pairs = cardinality == 2
    # Disabling pairs does not quietly change a declared two-error prior into
    # a singleton prior: preserve its untested alternatives and block a
    # uniqueness certificate until the declared pair scope is completed.
    scan = runtime.scan_candidates(include_pairs=include_pairs, max_pairs=max_pairs if scan_pairs else 0)
    initial_candidate = scan["current"]
    selected = initial_candidate
    if scan["unique_candidate_id"] is not None:
        runtime.apply(scan["unique_candidate_id"])
        selected = next(candidate for candidate in scan["candidates"] if candidate["candidate_id"] == scan["unique_candidate_id"])
        report["correction_applied"] = True
    after = runtime.snapshot()

    # Offline scoring begins only after the runtime has completed its decisions.
    true_statuses = validate_statuses(inventory, row["true_statuses"])
    physical = _read(root, row["physical_audit_path"])
    true_fit = numerical_estimator(physical["operating_case"], inventory, true_statuses, observations, sensors,
                                  chi2_alpha=.05, normalized_residual_threshold=4.)
    same_parameters_as_truth = evidence_hash(_without_branch_status(before["current_case"])) == evidence_hash(_without_branch_status(physical["operating_case"]))
    model_true_status_fit = true_fit if same_parameters_as_truth else numerical_estimator(
        before["current_case"], inventory, true_statuses, observations, sensors,
        chi2_alpha=.05, normalized_residual_threshold=4.)
    initial_statuses, final_statuses = before["current_statuses"], after["current_statuses"]
    initially_healthy = [device for device, actual in true_statuses.items() if initial_statuses[device] == actual]
    changed = {device: final_statuses[device] for device in true_statuses if final_statuses[device] != initial_statuses[device]}
    false_changes = [device for device, value in changed.items() if value != true_statuses[device]]
    initial_unknown = sum(value is None for value in initial_statuses.values())
    final_unknown = sum(value is None for value in final_statuses.values())
    initial_errors = sum(value is not None and value != true_statuses[device] for device, value in initial_statuses.items())
    final_errors = sum(value is not None and value != true_statuses[device] for device, value in final_statuses.items())
    correct = sum(final_statuses[device] == actual for device, actual in true_statuses.items())
    preservation = {
        "observations_unchanged": evidence_hash(before["observations"]) == evidence_hash(after["observations"]),
        "covariance_unchanged": evidence_hash(before["measurement_inventory"]["covariance"]) == evidence_hash(after["measurement_inventory"]["covariance"]),
        "measurement_inventory_unchanged": evidence_hash(before["measurement_inventory"]) == evidence_hash(after["measurement_inventory"]),
        "nonstatus_case_unchanged": evidence_hash(_without_branch_status(before["current_case"])) == evidence_hash(_without_branch_status(after["current_case"])),
        "fixed_evidence_hash_unchanged": before["fixed_evidence_hash"] == after["fixed_evidence_hash"],
    }
    preservation["passed"] = all(preservation.values())
    exact_statuses = correct == len(true_statuses)
    healthy_preserved = all(final_statuses[device] == initial_statuses[device] for device in initially_healthy)
    status_audit = {"initial_error_count": initial_errors, "initial_unknown_count": initial_unknown,
                    "final_error_count": final_errors, "final_unknown_count": final_unknown,
                    "correct_count": correct, "device_count": len(true_statuses),
                    "accuracy": correct / len(true_statuses), "exact_status_recovery": exact_statuses,
                    "false_correction_count": len(false_changes), "false_correction_device_ids": false_changes,
                    "healthy_cb_count": len(initially_healthy), "healthy_cb_preserved": healthy_preserved,
                    "changed_statuses": changed}
    true_in_plausible = any(candidate["statuses"] == true_statuses for candidate in scan["plausible_candidates"])
    compatible = scan.get("comparison_compatible_candidates", [])
    comparison_guard = scan.get("comparison_guard", {})
    identifiability = {"evaluated": True, "classification": scan["decision"], "reason": scan["reason"],
                       "scope_complete": scan["scope_complete"], "hypothesis_scope": deepcopy(scan["hypothesis_scope"]),
                       "tested_candidate_count": scan["tested_candidate_count"],
                       "plausible_candidate_count": len(scan["plausible_candidates"]),
                       "unresolved_candidate_count": scan["unresolved_candidate_count"],
                       "true_statuses_in_plausible_set": true_in_plausible,
                       "comparison_compatible_candidate_count": len(compatible),
                       "true_statuses_in_comparison_compatible_set": any(candidate["statuses"] == true_statuses for candidate in compatible),
                       "comparison_compatible_set_scope": scan.get("comparison_compatible_set_scope"),
                       "comparison_guard": {key: deepcopy(comparison_guard.get(key)) for key in
                                            ("contract", "allowed", "decision", "candidate_id", "familywise_alpha",
                                             "rival_count", "blocking_candidates", "requires_full_fit",
                                             "exact_nonlinear_false_positive_guarantee", "global_minima_verified",
                                             "common_model_regularity_verified")},
                       "absolute_unique_candidate_id": scan.get("absolute_unique_candidate_id"),
                       "full_fit_refinement_count": len(scan.get("full_fit_refinement_candidate_ids", [])),
                       "unique_candidate_correct": exact_statuses if report["correction_applied"] else None,
                       "global_uniqueness_claimed": False,
                       "pair_stratum_fully_searched": cardinality != 2 or (scan_pairs and scan["scope_complete"])}
    initial_fit, final_fit = initial_candidate.get("estimation"), selected.get("estimation")
    report.update(
        runtime_decision=scan["decision"], status_audit=status_audit, preservation=preservation,
        initial_estimation=_estimation_summary(initial_fit, reason=initial_candidate["reason"]),
        final_estimation=_estimation_summary(final_fit, reason=selected["reason"]),
        initial_hypothesis_resolution=initial_candidate["resolution"],
        final_hypothesis_resolution=selected["resolution"],
        supported_topology_resolution=bool(exact_statuses and selected["plausible"] and healthy_preserved and preservation["passed"]),
        wrong_model_wls_failed=bool((initial_errors or initial_unknown) and isinstance(initial_fit, Mapping) and initial_fit.get("converged") is not True),
        state_estimation_error={
            "initial": _state_error(initial_fit, before["current_case"], initial_statuses, physical, true_statuses, inventory, sensors),
            "final": _state_error(final_fit, after["current_case"], final_statuses, physical, true_statuses, inventory, sensors),
        },
        offline_true_status_model_fit=_estimation_summary(model_true_status_fit),
        offline_observability_fit_used_for_runtime_decisions=False,
        execution_input_hashes={key: evidence_hash(value) for key, value in
                                (("inventory", inventory), ("measurement_inventory", sensors), ("observations", observations), ("current_case", model_case))},
    )
    report["audit_outcomes"]["state_observability"] = {
        "evaluated": True, "basis": "true_physical_operating_case_and_true_statuses_with_same_fixed_sensor_evidence",
        "true_parameter_case_used": True, "model_parameters_match_truth": same_parameters_as_truth,
        "assessment": ("unresolved_numerical_fit" if true_fit.get("converged") is not True else
                       "observable" if true_fit.get("observable") is True else "unobservable"),
        **_estimation_summary(true_fit),
    }
    report["audit_outcomes"]["status_identifiability"] = identifiability
    detailed = root / "audits" / f"{scenario_id}.json.gz"
    detailed.parent.mkdir(parents=True, exist_ok=True)
    if detailed.exists():
        raise FileExistsError(f"Refusing to replace an existing detailed audit: {detailed}")
    payload = {"contract": report["contract"], "runtime_scan": scan,
               "offline_true_physical_status_fit": true_fit, "offline_true_status_model_fit": model_true_status_fit,
               "offline_scoring": {"status_audit": status_audit, "state_estimation_error": report["state_estimation_error"]}}
    with gzip.open(detailed, "wt", encoding="utf-8") as stream:
        json.dump(_plain(payload), stream, allow_nan=False, separators=(",", ":"))
    report["detailed_audit_path"] = str(detailed.relative_to(root)).replace("\\", "/")
    report["detailed_audit_sha256"] = hashlib.sha256(detailed.read_bytes()).hexdigest()
    return _plain(report)
