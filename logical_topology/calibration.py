"""Post-selection separation guard for a fixed logical-topology model family.

Absolute WLS goodness of fit is not a model-identification test. If both models
lie in a common relaxation, J(rival)-J(chosen) <= J(rival)-J(common). Under the
regular rival-null assumptions, the latter has an asymptotic chi-square law.
The common relaxation adds at most four terminal P/Q variables for each
differing branch status, or releases two operational constraints for each
differing ideal coupler. The structural upper df and Bonferroni family budget
therefore give an asymptotic guard, not an exact nonlinear finite-sample claim.

Gaussian zero-flow witnesses have a separate, exact marginal-tail argument
conditional on known covariance and an otherwise correct measurement model.
Either method can reject a rival, so each receives half the family alpha before
allocating across rival models (and, for flow witnesses, measured rows). This
union bound does not require independence or containment of the two tests.
No truth or empirical minimum-gain threshold is used here.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

import numpy as np
from scipy.stats import chi2, norm

from .runtime import evidence_hash


ASSUMPTIONS = {
    "known_measurement_covariance": True,
    "zero_mean_gaussian_analog_noise": True,
    "no_unmodeled_gross_analog_or_parameter_error_under_each_tested_null": True,
    "candidate_family_predeclared_independently_of_analog_values": True,
    "rival_and_selected_fit_use_same_available_raw_observations_and_covariance": True,
    "nonlinear_common_embedding_regular_and_locally_identifiable": True,
    "objectives_represent_the_relevant_global_or_regular_consistent_minima": True,
    "no_active_numerical_voltage_bounds_in_fitted_models": True,
}

# Fixed before inspecting measurements. A weak Gaussian witness followed by a
# fitted comparison is an OR of two tests, not a free second use of the budget.
COMPARISON_METHOD_COUNT = 2


def _devices(inventory):
    result = {}
    for collection, kind in (("branches", "branch_status"), ("couplers", "bus_coupler")):
        for row in inventory[collection]:
            name = row["device_id"]
            if name in result or row.get("device_kind", kind) != kind:
                raise ValueError("invalid or duplicate logical-device identity")
            result[name] = kind
    if not result:
        raise ValueError("empty logical-device inventory")
    return result


def _statuses(values, devices):
    if not isinstance(values, Mapping) or set(values) != set(devices):
        raise ValueError("candidate needs one status for every logical device")
    if any(not isinstance(value, Real) or not math.isfinite(float(value)) or value not in (0, 1) for value in values.values()):
        raise ValueError("candidate status must be known binary 0/1")
    return {key: int(value) for key, value in values.items()}


def common_envelope_df(inventory, rival_statuses, chosen_statuses) -> dict[str, Any]:
    """Structural upper bound; the actual relaxation Jacobian rank may be lower."""
    devices = _devices(inventory)
    rival, chosen = _statuses(rival_statuses, devices), _statuses(chosen_statuses, devices)
    branches = sorted(key for key in devices if rival[key] != chosen[key] and devices[key] == "branch_status")
    couplers = sorted(key for key in devices if rival[key] != chosen[key] and devices[key] == "bus_coupler")
    return {"df_upper_bound": 4*len(branches)+2*len(couplers),
            "differing_branch_device_ids": branches, "differing_coupler_device_ids": couplers,
            "relaxation": "four free real branch-terminal P/Q discrepancies per changed branch; two released ideal-coupler operational constraints per changed coupler",
            "actual_common_model_rank_gain_measured": False}


def _evidence(inventory, sensors, observations):
    records = sensors["records"]
    count = len(records)
    mask = np.asarray(sensors["available_mask"], dtype=bool)
    covariance = np.asarray(sensors["covariance"], dtype=float)
    values = observations["values"]
    if (sensors["layout_hash"] != inventory["layout_hash"]
        or observations["sensor_inventory_hash"] != sensors["sensor_inventory_hash"]
        or observations["sensor_ids"] != [row["sensor_id"] for row in records]
        or len({row["sensor_id"] for row in records}) != count or len(values) != count
        or mask.shape != (count,) or covariance.shape != (count, count)
        or not np.array_equal(mask, [bool(row["available"]) for row in records])
        or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-14, rtol=0)):
        raise ValueError("inconsistent fixed physical measurement evidence")
    if any(values[index] is not None for index in np.flatnonzero(~mask)):
        raise ValueError("unavailable values must remain None")
    available = np.asarray([values[index] for index in np.flatnonzero(mask)], dtype=float)
    if not len(available) or not np.isfinite(available).all():
        raise ValueError("available observations must be finite")
    r = covariance[np.ix_(mask, mask)]
    if np.array_equal(r, np.diag(np.diag(r))):
        if np.min(np.diag(r)) <= 0:
            raise ValueError("available noise covariance must be positive definite")
    else:
        np.linalg.cholesky(r)
    return records, mask, covariance


def _zero_flow(inventory, statuses, records, mask, covariance, observations, rival_count, alpha, *, method_count=1):
    by_row = {int(row["row0"]): row["device_id"] for row in inventory["branches"]}
    witnesses = []
    for index, record in enumerate(records):
        if not mask[index] or record["kind"] not in {"Pf", "Qf", "Pt", "Qt"}:
            continue
        device = by_row[int(record["branch_row0"])]
        if statuses[device] != 0:
            continue
        variance = float(covariance[index, index])
        score = abs(float(observations["values"][index]))/math.sqrt(variance)
        witnesses.append({"sensor_id": record["sensor_id"], "device_id": device,
                          "normalized_absolute_value": score, "marginal_variance": variance,
                          "observed_value": float(observations["values"][index])})
    if not witnesses:
        return {"passed": False, "reason": "no_available_zero_flow_witness_rows", "available_zero_flow_row_count": 0}
    witness = max(witnesses, key=lambda item: item["normalized_absolute_value"])
    z = witness["normalized_absolute_value"]
    log_bound = min(0.0, math.log(2.0)+float(norm.logsf(z))+math.log(len(witnesses))
                    +math.log(rival_count)+math.log(method_count))
    bound = float(math.exp(log_bound))
    return {"passed": log_bound <= math.log(alpha), "method": "gaussian_zero_mean_flow_union_bound",
            "available_zero_flow_row_count": len(witnesses), "family_rival_count": rival_count,
            "method_count": method_count, "method_family_alpha": alpha/method_count,
            "pairwise_method_alpha": alpha/(method_count*rival_count),
            "multiplicity_adjusted_p_upper_bound": bound, "log_adjusted_p_upper_bound": log_bound,
            "witness": witness, "alpha": alpha,
            "reason": "multiplicity_adjusted_zero_flow_rejection" if log_bound <= math.log(alpha)
            else "ordinary_NR_alarm_does_not_meet_family_adjusted_tail_bound",
            "correlation_handling": "marginal Gaussian tails and union bound; independence not required"}


def gaussian_zero_flow_rejection(inventory, statuses, sensors, observations, *, rival_count, alpha=.05):
    """Recompute the witness from fixed raw data rather than trust an NR4 receipt."""
    if not isinstance(rival_count, int) or isinstance(rival_count, bool) or rival_count < 1:
        raise ValueError("rival_count must be a positive integer")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    status = _statuses(statuses, _devices(inventory))
    records, mask, covariance = _evidence(inventory, sensors, observations)
    return _zero_flow(inventory, status, records, mask, covariance, observations, rival_count, alpha)


def _fit(candidate, available_count):
    fit = candidate.get("estimation")
    if not isinstance(fit, Mapping):
        return None, "full_fit_missing"
    if fit.get("converged") is not True:
        return None, "rival_WLS_nonconvergence"
    try:
        m, rank, dimension = int(fit["available_measurement_count"]), int(fit["rank"]), int(fit["state_dimension"])
        objective, maximum = float(fit["wls_objective"]), float(fit["max_normalized_residual"])
        if (m != available_count or not 0 < dimension <= m or not 0 <= rank <= dimension
            or not all(math.isfinite(value) and value >= 0 for value in (objective, maximum))):
            raise ValueError("invalid fit dimensions or residuals")
        if rank != dimension or fit.get("observable") is not True:
            return None, "rival_state_unobservable"
        if m-rank <= 0:
            return None, "no_residual_redundancy"
        volts = fit.get("state", {}).get("node_voltage_magnitude_pu", {})
        if volts and any(not math.isfinite(float(v)) or float(v) <= .200001 or float(v) >= 1.999999 for v in volts.values()):
            return None, "numerical_voltage_bound_invalidates_regular_fit_calibration"
        return {"objective": objective, "rank": rank, "state_dimension": dimension,
                "available_measurement_count": m, "max_normalized_residual": maximum}, None
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return None, f"invalid_full_fit:{exc}"


def _scope_exclusion(candidate, scan, inventory, statuses):
    if scan.get("hypothesis_scope", {}).get("connected_energized_models_only") is not True:
        return False
    if candidate.get("connectivity", {}).get("connected") is not False:
        return False
    nodes = {row["node_id"] for row in inventory.get("nodes", [])}
    if not nodes:
        return False
    parent = {node: node for node in nodes}
    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for collection, first, second in (("branches", "from_node", "to_node"), ("couplers", "node_a", "node_b")):
        for device in inventory[collection]:
            if statuses[device["device_id"]]:
                a, b = root(device[first]), root(device[second])
                parent[a] = b
    return len({root(node) for node in nodes}) > 1


def calibrate_scan(scan, inventory, sensors, observations, *, alpha=.05) -> dict[str, Any]:
    """Require multiplicity-aware separation from every declared rival model.

Weak analytical witnesses request a full fit. Numerical/observability failures
remain blocking. This adds a guard after unique absolute plausibility; it never
selects a replacement candidate or relaxes chi-square/NR admission thresholds.
"""
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    result = {"contract": "logical_topology_pairwise_separation_guard_v2", "allowed": False,
              "decision": "not_a_unique_correction", "candidate_id": scan.get("unique_candidate_id"),
              "familywise_alpha": alpha, "comparisons": [], "requires_full_fit": [], "blocking_candidates": [],
              "method_count": COMPARISON_METHOD_COUNT, "method_budget_allocation": "equal_bonferroni_split",
              "method_family_alpha": alpha/COMPARISON_METHOD_COUNT,
              "exact_nonlinear_false_positive_guarantee": False, "calibration_assumptions": dict(ASSUMPTIONS),
              "global_minima_verified": False, "common_model_regularity_verified": False,
              "scope": "retrospective asymptotic guard within a predeclared fixed-evidence candidate family"}
    if not scan.get("unique_candidate_id") or scan.get("scope_complete") is not True:
        return result
    try:
        records, mask, covariance = _evidence(inventory, sensors, observations)
        expected_hash = evidence_hash({"measurement_inventory": sensors, "observations": observations})
        if scan.get("fixed_evidence_hash") != expected_hash:
            raise ValueError("scan does not bind these fixed observations and covariance")
        devices = _devices(inventory)
        candidates = list(scan["candidates"])
        lookup = {row["candidate_id"]: row for row in candidates}
        if len(lookup) != len(candidates):
            raise ValueError("duplicate candidate IDs")
        current = scan.get("current")
        if isinstance(current, Mapping) and current.get("candidate_id") not in lookup:
            # An unresolved indication containing None is not itself a binary
            # rival: its explicitly enumerated complete assignments are rivals.
            if all(value is not None for value in current.get("statuses", {}).values()):
                lookup[current["candidate_id"]] = current
                candidates.append(current)
        for candidate in candidates:
            if (candidate.get("fixed_evidence_hash") != expected_hash
                or candidate.get("parent_model_hash") != scan.get("parent_model_hash")):
                raise ValueError("candidate evidence or parent-model binding differs from scan")
            _statuses(candidate["statuses"], devices)
        chosen = lookup[scan["unique_candidate_id"]]
        chosen_fit, failure = _fit(chosen, int(mask.sum()))
        if failure or chosen.get("plausible") is not True or chosen.get("current_model") is True:
            result.update(decision="invalid_selected_full_fit", selected_fit_problem=failure or "not_a_changed_plausible_candidate")
            return result
        fit_payload = chosen["estimation"]
        fit_alpha = float(fit_payload.get("chi_square_alpha", .05))
        fit_nr = float(fit_payload.get("normalized_residual_threshold", 4.0))
        if (not 0 < fit_alpha < 1 or not math.isfinite(fit_nr) or fit_nr <= 0
            or chosen_fit["objective"] >= chi2.ppf(1-fit_alpha, chosen_fit["available_measurement_count"]-chosen_fit["rank"])
            or chosen_fit["max_normalized_residual"] >= fit_nr):
            result.update(decision="selected_model_fails_absolute_fit_tests")
            return result
        rivals = [row for row in candidates if row["candidate_id"] != chosen["candidate_id"]]
        rival_count = len(rivals)
        if not rival_count:
            result.update(decision="no_rival_models_tested")
            return result
        result.update(candidate_family_size=len(candidates), rival_count=rival_count,
                      pairwise_alpha=alpha/rival_count,
                      pairwise_method_alpha=alpha/(COMPARISON_METHOD_COUNT*rival_count),
                      fixed_evidence_hash=expected_hash,
                      parent_model_hash=scan["parent_model_hash"], selected_objective=chosen_fit["objective"])
        for rival in rivals:
            cid = rival["candidate_id"]
            comparison = {"candidate_id": cid, "passed": False, "rival_resolution": rival.get("resolution")}
            if rival.get("resolution") == "excluded":
                passed = _scope_exclusion(rival, scan, inventory, rival["statuses"])
                comparison.update(passed=passed, method="declared_connected_scope_exclusion",
                                  reason="verified_disconnected_logical_graph" if passed else "scope_exclusion_not_proven")
            elif rival.get("resolution") == "unresolved":
                comparison.update(method="unresolved_alternative", reason=rival.get("reason", "rival_unresolved"))
            elif rival.get("plausible") is True:
                comparison.update(method="competing_absolute_fit", reason="another_model_is_absolutely_plausible")
            else:
                analytic = rival.get("resolution") == "analytical_rejection"
                if analytic:
                    flow = _zero_flow(inventory, rival["statuses"], records, mask, covariance, observations,
                                      rival_count, alpha, method_count=COMPARISON_METHOD_COUNT)
                    comparison["zero_flow_evidence"] = flow
                    if flow["passed"]:
                        comparison.update(passed=True, method="gaussian_zero_mean_flow_union_bound", reason=flow["reason"])
                if not comparison["passed"]:
                    rival_fit, problem = _fit(rival, int(mask.sum()))
                    if rival_fit is None:
                        if analytic and problem == "full_fit_missing":
                            result["requires_full_fit"].append(cid)
                            comparison.update(method="full_fit_required", reason="analytical_rejection_not_family_adjusted")
                        else:
                            comparison.update(method="unresolved_fit", reason=problem)
                    else:
                        envelope = common_envelope_df(inventory, rival["statuses"], chosen["statuses"])
                        df = envelope["df_upper_bound"]
                        gain = rival_fit["objective"]-chosen_fit["objective"]
                        if df <= 0:
                            comparison.update(method="equivalent_status_model", reason="no_status_difference_to_identify")
                        else:
                            critical = float(chi2.isf(alpha/(COMPARISON_METHOD_COUNT*rival_count), df))
                            upper_p = min(1.0, COMPARISON_METHOD_COUNT*rival_count*float(chi2.sf(max(gain, 0), df)))
                            comparison.update(method="asymptotic_common_relaxation_envelope", passed=gain >= critical,
                                              reason="rival_materially_separated" if gain >= critical else "absolute_threshold_crossing_without_material_separation",
                                              objective_gain=gain, critical_gain=critical,
                                              rival_objective=rival_fit["objective"], selected_objective=chosen_fit["objective"],
                                              multiplicity_adjusted_asymptotic_p_upper_bound=upper_p, **envelope)
            result["comparisons"].append(comparison)
            if not comparison["passed"]:
                result["blocking_candidates"].append(cid)
        result["allowed"] = not result["blocking_candidates"]
        result["decision"] = ("passes_conditional_pairwise_separation_guard" if result["allowed"] else
                              "requires_additional_candidate_fits" if result["requires_full_fit"] else
                              "insufficient_calibrated_status_identifiability")
        return result
    except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        result.update(decision="invalid_fixed_evidence_or_scan", error_detail=f"{type(exc).__name__}: {exc}")
        return result
