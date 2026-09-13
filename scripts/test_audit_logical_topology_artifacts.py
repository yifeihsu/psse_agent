"""Negative checks for logical-topology artifact proof boundaries."""
from copy import deepcopy
import math

import pytest
import numpy as np
from scipy.stats import chi2, norm

from logical_topology.inventory import build_inventory
from logical_topology.measurements import build_measurement_inventory, sample_measurements
from scripts.audit_logical_topology_artifacts import (
    FixedNumericalInputCache, compact_summary, comparison_certificate_errors, comparison_guard_errors, content_hash, numerical_semantic_hash, status_metrics, validate_observation_boundary, validate_sensor_deployment,
)


def fixture():
    inventory = build_inventory("case14", split_buses=[])
    sensors = build_measurement_inventory(inventory, "indirect_even")
    observations = sample_measurements([0.] * len(sensors["records"]), sensors, noise=False)
    return inventory, sensors, observations


def test_valid_fixed_profile_and_redacted_observations_are_accepted():
    inventory, sensors, observations = fixture()
    validate_sensor_deployment(sensors, inventory)
    validate_observation_boundary(observations, sensors)


@pytest.mark.parametrize("kind", ["masked_value", "truth_metadata", "sensor_id", "nonfinite_available"])
def test_observation_boundary_rejects_available_data_leaks_or_mismatches(kind):
    _, sensors, observations = fixture()
    if kind == "masked_value":
        index = sensors["available_mask"].index(False)
        observations["values"][index] = 0.
    elif kind == "truth_metadata":
        observations["true_statuses"] = {"cb": 1}
    elif kind == "sensor_id":
        observations["sensor_ids"][0] = "unregistered"
    else:
        observations["values"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_observation_boundary(observations, sensors)


def test_fault_specific_mask_change_cannot_pass_as_a_fixed_indirect_profile():
    inventory, sensors, _ = fixture()
    record = next(r for r in sensors["records"] if r.get("branch_row0") == 2)
    record["available"] = True
    with pytest.raises(ValueError, match="whole-deployment"):
        validate_sensor_deployment(sensors, inventory)


def test_noise_change_requires_a_different_declared_experiment():
    inventory, sensors, _ = fixture()
    sensors["records"][0]["sigma"] = .1
    with pytest.raises(ValueError, match="noise"):
        validate_sensor_deployment(sensors, inventory)


def test_status_scoring_counts_false_return_to_normal_as_a_false_correction():
    actual = {"open_healthy": 0, "bad_report": 1}
    initial = {"open_healthy": 0, "bad_report": 0}
    final = {"open_healthy": 1, "bad_report": 1}
    metrics = status_metrics(initial, final, actual)
    assert metrics["false_correction_count"] == 1
    assert metrics["false_correction_device_ids"] == ["open_healthy"]
    assert metrics["healthy_cb_preserved"] is False
    assert metrics["exact_status_recovery"] is False


def test_compact_outcomes_keep_runtime_failures_and_false_corrections():
    rows = [{"physical_admitted": True, "audit_execution_failure": "bad input"},
            {"physical_admitted": True, "correction_applied": True,
             "status_audit": {"false_correction_count": 1, "healthy_cb_preserved": False, "exact_status_recovery": False},
             "preservation": {"passed": False}}, {"physical_admitted": False}]
    result = compact_summary(rows)
    assert result["planned"] == 3 and result["physically_admitted"] == 2
    assert result["audit_execution_failures"] == 1
    assert result["false_corrections"] == 1
    assert result["fixed_evidence_or_parameter_preservation_failures"] == 1


def _comparison_fixture(*, gaussian=False):
    from logical_topology.test_calibration import fixture as fixed_family, scan_for
    from logical_topology.calibration import calibrate_scan
    inventory, sensors, observations, statuses = fixed_family()
    if gaussian:
        chosen, rival = dict(statuses), {**statuses, "b1": 0}
        observations["values"][0] = 8.
        scan = scan_for(inventory, sensors, observations, chosen, rival, rival_resolution="analytical_rejection", rival_fit=False)
    else:
        chosen, rival = {**statuses, "b1": 0}, dict(statuses)
        scan = scan_for(inventory, sensors, observations, chosen, rival)
    return calibrate_scan(scan, inventory, sensors, observations), scan, inventory, sensors, observations


@pytest.mark.parametrize("gaussian", [False, True])
def test_independent_checker_accepts_valid_comparison_guard_arithmetic(gaussian):
    values = _comparison_fixture(gaussian=gaussian)
    assert values[0]["allowed"] is True
    assert comparison_guard_errors(*values) == []


@pytest.mark.parametrize("field,value", [("df_upper_bound", 1), ("critical_gain", 0.), ("objective_gain", 9999.)])
def test_comparison_guard_rejects_forged_common_envelope_arithmetic(field, value):
    values = _comparison_fixture()
    values[0]["comparisons"][0][field] = value
    assert "comparison_common_envelope_arithmetic" in comparison_guard_errors(*values)


def test_comparison_guard_rejects_missing_witness_multiplicity_and_false_exact_claim():
    values = _comparison_fixture(gaussian=True)
    values[0]["comparisons"][0]["zero_flow_evidence"]["available_zero_flow_row_count"] = 1
    assert "comparison_gaussian_tail_or_multiplicity" in comparison_guard_errors(*values)
    values = _comparison_fixture()
    values[0]["exact_nonlinear_false_positive_guarantee"] = True
    assert "unsupported_exact_nonlinear_guarantee" in comparison_guard_errors(*values)


def test_independent_numeric_cache_key_matches_exact_inputs_and_changes_with_evidence():
    from psse_env.systems import resolve_system
    from logical_topology.fit_cache import semantic_hash
    inventory, sensors, observations = fixture()
    payload = {"case": resolve_system("case14").load_case(), "inventory": inventory,
               "statuses": inventory["normal_statuses"], "observations": observations, "sensors": sensors,
               "parameters": {"chi2_alpha": .05, "normalized_residual_threshold": 4., "max_nfev": 100},
               "numerical_source_sha256": {"estimation.py": "a"*64}}
    expected = semantic_hash(payload)
    assert numerical_semantic_hash(payload) == expected
    changed = deepcopy(payload)
    changed["observations"]["values"][0] += 1e-12
    assert numerical_semantic_hash(changed) != expected
    changed = deepcopy(payload)
    changed["case"]["branch"][0, 2] += 1e-12
    assert numerical_semantic_hash(changed) != expected


def _certificate_fixture():
    guard, scan, *_ = _comparison_fixture()
    scan["plausible_candidates"] = [c for c in scan["candidates"] if c.get("plausible")]
    scan["unresolved_candidate_count"] = sum(c.get("resolution") == "unresolved" for c in scan["candidates"])
    scan["comparison_guard"] = guard
    scan["certificate"] = {
        "comparison_guard": deepcopy(guard), "comparison_guard_hash": content_hash(guard),
        "candidate_id": scan["unique_candidate_id"], "hypothesis_scope": deepcopy(scan["hypothesis_scope"]),
        "plausible_candidate_count": len(scan["plausible_candidates"]),
        "unresolved_candidate_count": scan["unresolved_candidate_count"],
    }
    return scan


def test_guarded_run_requires_a_guard_but_historical_prototype_does_not():
    assert comparison_certificate_errors({}, require_guard=False) == []
    assert comparison_certificate_errors({}, require_guard=True) == ["required_comparison_guard_missing"]


def test_independent_checker_accepts_bound_guard_certificate():
    assert comparison_certificate_errors(_certificate_fixture(), require_guard=True) == []


@pytest.mark.parametrize("kind", ["guard_hash", "scope", "family_alpha"])
def test_independent_checker_rejects_certificate_or_config_tampering(kind):
    scan = _certificate_fixture()
    if kind == "guard_hash":
        scan["certificate"]["comparison_guard_hash"] = "0"*64
        expected = "comparison_certificate_guard_binding"
    elif kind == "scope":
        scan["certificate"]["hypothesis_scope"] = {}
        expected = "comparison_certificate_scope_binding"
    else:
        scan["comparison_guard"]["familywise_alpha"] = .25
        expected = "comparison_alpha_differs_from_declared_configuration"
    assert expected in comparison_certificate_errors(scan, require_guard=True)


def test_new_archived_guard_contract_cannot_be_replaced_with_legacy_contract():
    scan = _certificate_fixture()
    scan["comparison_guard"]["contract"] = "logical_topology_pairwise_separation_guard_v1"
    assert "comparison_guard_differs_from_archived_contract" in comparison_certificate_errors(
        scan, require_guard=True, expected_guard_contract="logical_topology_pairwise_separation_guard_v2")


@pytest.mark.parametrize("field,value", [("method_count", 1), ("method_budget_allocation", "select_after_observing_data"),
                                         ("method_family_alpha", .05), ("pairwise_method_alpha", .05)])
def test_v2_guard_rejects_missing_shared_method_budget(field, value):
    values = _comparison_fixture()
    assert values[0]["contract"] == "logical_topology_pairwise_separation_guard_v2"
    values[0][field] = value
    expected = "comparison_pairwise_method_budget" if field == "pairwise_method_alpha" else "comparison_method_budget"
    assert expected in comparison_guard_errors(*values)


@pytest.mark.parametrize("field", ["critical_gain", "multiplicity_adjusted_asymptotic_p_upper_bound"])
def test_v2_envelope_rejects_legacy_single_method_threshold_or_p_bound(field):
    values = _comparison_fixture()
    guard = values[0]
    comparison = guard["comparisons"][0]
    comparison[field] = (chi2.isf(guard["familywise_alpha"]/guard["rival_count"], comparison["df_upper_bound"])
                         if field == "critical_gain" else
                         min(1., guard["rival_count"]*chi2.sf(comparison["objective_gain"], comparison["df_upper_bound"])))
    assert "comparison_common_envelope_arithmetic" in comparison_guard_errors(*values)


def test_v2_gaussian_rejects_legacy_single_method_p_bound():
    values = _comparison_fixture(gaussian=True)
    guard = values[0]
    flow = guard["comparisons"][0]["zero_flow_evidence"]
    legacy_log = min(0., math.log(2) + norm.logsf(flow["witness"]["normalized_absolute_value"])
                     + math.log(flow["available_zero_flow_row_count"]) + math.log(guard["rival_count"]))
    flow["log_adjusted_p_upper_bound"] = legacy_log
    flow["multiplicity_adjusted_p_upper_bound"] = math.exp(legacy_log)
    assert "comparison_gaussian_tail_or_multiplicity" in comparison_guard_errors(*values)


@pytest.mark.parametrize("field,value", [("method_count", 1), ("method_family_alpha", .05), ("pairwise_method_alpha", .05)])
def test_v2_gaussian_rejects_inconsistent_method_budget_receipt(field, value):
    values = _comparison_fixture(gaussian=True)
    values[0]["comparisons"][0]["zero_flow_evidence"][field] = value
    assert "comparison_gaussian_method_budget" in comparison_guard_errors(*values)


@pytest.mark.parametrize("gaussian", [False, True])
def test_legacy_guard_remains_inspectable_as_unshared_budget_history(gaussian):
    values = _comparison_fixture(gaussian=gaussian)
    guard = values[0]
    guard["contract"] = "logical_topology_pairwise_separation_guard_v1"
    for field in ("method_count", "method_budget_allocation", "method_family_alpha", "pairwise_method_alpha"):
        guard.pop(field, None)
    comparison = guard["comparisons"][0]
    if gaussian:
        flow = comparison["zero_flow_evidence"]
        for field in ("method_count", "method_family_alpha", "pairwise_method_alpha"):
            flow.pop(field, None)
        legacy_log = min(0., math.log(2) + norm.logsf(flow["witness"]["normalized_absolute_value"])
                         + math.log(flow["available_zero_flow_row_count"]) + math.log(guard["rival_count"]))
        flow["log_adjusted_p_upper_bound"] = legacy_log
        flow["multiplicity_adjusted_p_upper_bound"] = math.exp(legacy_log)
    else:
        comparison["critical_gain"] = chi2.isf(guard["familywise_alpha"]/guard["rival_count"], comparison["df_upper_bound"])
        comparison["multiplicity_adjusted_asymptotic_p_upper_bound"] = min(1., guard["rival_count"]*chi2.sf(comparison["objective_gain"], comparison["df_upper_bound"]))
    assert comparison_guard_errors(*values) == []


def _fixed_cache_fixture():
    from psse_env.systems import resolve_system
    inventory, sensors, observations = fixture()
    payload = {"case": resolve_system("case14").load_case(), "inventory": inventory,
               "statuses": deepcopy(inventory["normal_statuses"]), "observations": observations, "sensors": sensors,
               "parameters": {"chi2_alpha": .05, "normalized_residual_threshold": 4., "max_nfev": 100},
               "numerical_source_sha256": {"estimation.py": "a"*64}}
    return payload


def _fixed_cache(payload):
    return FixedNumericalInputCache(**{key:payload[key] for key in ("inventory", "observations", "sensors", "numerical_source_sha256")})


def _cached_hash(cache, payload):
    return cache.hash(**{key:payload[key] for key in ("case", "statuses", "parameters")})


@pytest.mark.parametrize("variant", ["unchanged", "parameter", "status", "solver", "lists", "signed_zero"])
def test_cached_fixed_encoding_preserves_every_dynamic_semantic_key(variant):
    payload = _fixed_cache_fixture()
    cache = _fixed_cache(payload)
    original = numerical_semantic_hash(payload)
    if variant == "parameter":
        payload["case"]["branch"][0, 2] += 1e-12
    elif variant == "status":
        payload["statuses"][next(iter(payload["statuses"]))] = 0
    elif variant == "solver":
        payload["parameters"]["max_nfev"] += 1
    elif variant == "lists":
        payload["case"] = {key:value.tolist() if isinstance(value, np.ndarray) else value for key,value in payload["case"].items()}
    elif variant == "signed_zero":
        branch = payload["case"]["branch"]
        branch[branch == 0] = -0.
    assert _cached_hash(cache, payload) == numerical_semantic_hash(payload)
    if variant in {"parameter", "status", "solver"}:
        assert _cached_hash(cache, payload) != original
    else:
        assert _cached_hash(cache, payload) == original
    cache.assert_fixed_unchanged()


@pytest.mark.parametrize("field", ["observation", "covariance", "sensor_id", "mask", "inventory", "source_hash"])
def test_cached_fixed_encoding_detects_changed_fixed_input_at_row_end(field):
    payload = _fixed_cache_fixture()
    cache = _fixed_cache(payload)
    original = _cached_hash(cache, payload)
    if field == "observation":
        payload["observations"]["values"][0] += 1e-12
    elif field == "covariance":
        payload["sensors"]["covariance"][0][0] += 1e-12
    elif field == "sensor_id":
        payload["sensors"]["records"][0]["sensor_id"] += "_changed"
    elif field == "mask":
        payload["sensors"]["available_mask"][0] = False
    elif field == "inventory":
        payload["inventory"]["layout_id"] += "_changed"
    else:
        payload["numerical_source_sha256"]["estimation.py"] = "b"*64
    assert numerical_semantic_hash(payload) != original
    with pytest.raises(ValueError, match="Fixed numerical audit inputs changed"):
        cache.assert_fixed_unchanged()


def test_original_and_new_fixed_input_caches_have_separate_mutation_checks():
    original = _fixed_cache_fixture()
    copied = deepcopy(original)
    old_cache, new_cache = _fixed_cache(original), _fixed_cache(copied)
    assert _cached_hash(old_cache, original) == _cached_hash(new_cache, copied)
    original["observations"]["values"][0] += 1e-12
    with pytest.raises(ValueError, match="Fixed numerical audit inputs changed"):
        old_cache.assert_fixed_unchanged()
    new_cache.assert_fixed_unchanged()
