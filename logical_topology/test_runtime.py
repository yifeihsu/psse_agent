from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import chi2

from logical_topology.runtime import LogicalTopologyRuntime, evidence_hash


def _metrics(objective=0.0, maximum=0.0, **updates):
    return {"converged": True, "observable": True, "rank": 5, "state_dimension": 5,
            "available_measurement_count": 20, "wls_objective": objective,
            "max_normalized_residual": maximum, **updates}


def _fixture(*, statuses=None, values=None, estimator=None, processor=None):
    inventory = {
        "branches": [{"device_id": f"b{row}", "device_kind": "branch_status", "row0": row,
                      "asset_id": f"branch:{row}", "from_node": "a", "to_node": "b"} for row in range(3)],
        "couplers": [{"device_id": "c", "device_kind": "bus_coupler", "node_a": "a", "node_b": "b", "base_bus": 4}],
        "normal_statuses": {"b0": 1, "b1": 1, "b2": 1, "c": 1},
        "layout_hash": "unit_test_layout",
        "nodes": [{"node_id": node} for node in ("a", "b", "end")],
    }
    inventory["branches"][2].update(from_node="b", to_node="end")
    case = {"baseMVA": 100., "bus": np.zeros((2, 13)), "gen": np.zeros((1, 21)),
            "branch": np.zeros((3, 13))}
    case["branch"][:, 2:4] = [[.1234, .5678], [.04, .2], [.05, .3]]
    case["branch"][:, 10] = 1
    records = [{"sensor_id": f"unit_sensor_{index}", "kind": "Vm", "node_id": "a", "available": True} for index in range(20)]
    sensors = {"covariance": np.diag([1e-6]+[1e-4]*19).tolist(), "available_mask": [True]*20,
               "records": records, "layout_hash": inventory["layout_hash"], "sensor_inventory_hash": "unit_sensors"}
    observations = {"values": ([1, 0, 1, 1] if values is None else list(values)) + [0.0]*16,
                    "sensor_ids": [row["sensor_id"] for row in records], "sensor_inventory_hash": "unit_sensors"}
    calls = []

    def default_estimator(current_case, devices, state, observed, deployed, **kwargs):
        calls.append(deepcopy((current_case, state, observed, deployed, kwargs)))
        mismatch = sum(abs(state[key]-observed["values"][i]) for i, key in enumerate(("b0", "b1", "b2", "c")))
        return _metrics(100.*mismatch, 10.*mismatch)

    runtime = LogicalTopologyRuntime(
        inventory=inventory, current_case=case,
        current_statuses={"b0": 0, "b1": 0, "b2": 1, "c": 1} if statuses is None else statuses,
        measurement_inventory=sensors, observations=observations,
        estimator=estimator or default_estimator,
        processor=processor or (lambda case, inventory, statuses: {"case": case, "connectivity": {"connected": True}}),
    )
    return runtime, case, observations, sensors, calls


def test_uniform_cb_interface_preserves_different_electrical_operations():
    processed = []

    def processor(case, inventory, statuses):
        processed.append((deepcopy(case), dict(statuses)))
        return {"case": case, "connectivity": {"connected": True}}

    runtime, _, _, _, _ = _fixture(processor=processor)
    before = runtime.snapshot()
    assert runtime.inspect_cb("b0")["electrical_operation"] == "whole_branch_terminal_admittance_multiplier"
    assert runtime.inspect_cb("c")["electrical_operation"] == "ideal_bus_section_contraction_or_separation"
    result = runtime.test_cb("c", 0)
    assert result["device_kinds"] == {"c": "bus_coupler"}
    np.testing.assert_array_equal(processed[-1][0]["branch"], before["current_case"]["branch"])
    assert processed[-1][1]["c"] == 0
    assert runtime.snapshot()["current_statuses"] == before["current_statuses"]


def test_unique_apply_preserves_current_parameters_measurement_fix_and_healthy_outage():
    runtime, input_case, observations, sensors, calls = _fixture()
    fixed = runtime.snapshot()
    # Caller-side edits do not alter the runtime's immutable evidence snapshot.
    input_case["branch"][0, 2] = 999
    observations["values"][1] = 1
    sensors["covariance"][0][0] = 999
    audit = runtime.scan_candidates()
    assert audit["decision"] == "unique_within_declared_scope"
    result = runtime.apply(audit["unique_candidate_id"])
    assert result["current_statuses"] == {"b0": 1, "b1": 0, "b2": 1, "c": 1}
    assert result["current_case"]["branch"][0, 2] == .1234
    assert result["current_case"]["branch"][1, 10] == 0  # correctly represented non-normal outage
    assert result["observations"] == fixed["observations"]
    assert result["measurement_inventory"] == fixed["measurement_inventory"]
    assert result["fixed_evidence_hash"] == fixed["fixed_evidence_hash"]
    assert all(call[2] == fixed["observations"] and call[3] == fixed["measurement_inventory"] for call in calls)
    assert "truth" not in result


def test_current_plausible_open_branch_is_kept_without_restoring_normal_statuses():
    runtime, _, _, _, _ = _fixture(statuses={"b0": 1, "b1": 0, "b2": 1, "c": 1})
    result = runtime.scan_candidates(include_pairs=True, max_pairs=0)
    assert result["decision"] == "keep_current"
    assert result["unique_candidate_id"] is None
    assert runtime.snapshot()["current_statuses"]["b1"] == 0
    candidate = runtime.test_cb("b1", 1)
    with pytest.raises(ValueError):
        runtime.apply(candidate["candidate_id"])


def test_cumulative_pair_correction_keeps_an_unrelated_existing_outage():
    runtime, _, _, _, _ = _fixture(statuses={"b0": 0, "b1": 0, "b2": 0, "c": 1})
    singles = runtime.scan_candidates()
    assert singles["decision"] == "request_measurement_or_parameter_investigation"
    audit = runtime.scan_candidates(include_pairs=True, pair_devices=["b0", "b2"], max_pairs=1)
    assert audit["scope_complete"]
    assert audit["decision"] == "unique_within_declared_scope"
    assert audit["hypothesis_scope"]["global_status_uniqueness_claimed"] is False
    applied = runtime.apply(audit["unique_candidate_id"])
    assert applied["applied_changes"] == {"b0": 1, "b2": 1}
    assert applied["current_statuses"]["b1"] == 0
    np.testing.assert_equal(applied["current_case"]["branch"][:, 10], [1, 0, 1])


def test_truncated_pair_search_cannot_certify_the_best_plausible_candidate():
    runtime, _, _, _, _ = _fixture(values=[1, 1, 1, 1])
    audit = runtime.scan_candidates(include_pairs=True, max_pairs=1)
    assert not audit["scope_complete"]
    assert len(audit["plausible_candidates"]) == 1
    assert audit["unique_candidate_id"] is None
    assert audit["hypothesis_scope"]["untested_pairs_per_assignment"] == 5
    with pytest.raises(ValueError, match="certificate"):
        runtime.apply(audit["plausible_candidates"][0]["candidate_id"])


def test_two_good_fits_remain_ambiguous_even_when_objectives_differ():
    def ambiguous(case, inventory, statuses, observed, sensors, **kwargs):
        changed = statuses["b0"] + (1-statuses["b2"])
        return _metrics(.1 if statuses["b0"] else .2, .1) if changed == 1 else _metrics(100, 10)

    runtime, _, _, _, _ = _fixture(estimator=ambiguous)
    audit = runtime.scan_candidates()
    assert audit["decision"] == "ambiguous_candidate_set"
    assert len(audit["plausible_candidates"]) == 2
    assert audit["unique_candidate_id"] is None


def test_numerical_failure_is_retained_and_other_hypotheses_are_still_tested():
    calls = []

    def estimator(case, inventory, statuses, observed, sensors, **kwargs):
        calls.append(dict(statuses))
        if statuses["b2"] == 0:
            return {"converged": False, "failure_reason": "wrong_model_numerical_failure"}
        return _metrics(0, 0) if statuses["b0"] else _metrics(100, 10)

    runtime, _, _, _, _ = _fixture(estimator=estimator)
    result = runtime.scan_candidates()
    assert len(calls) == 5
    assert result["decision"] == "request_additional_investigation"
    assert result["unresolved_candidate_count"] == 1
    assert any(row["reason"] == "wrong_model_numerical_failure" for row in result["candidates"])
    assert result["plausible_candidates"]
    assert result["unique_candidate_id"] is None


def test_proven_disconnected_candidate_is_excluded_by_declared_scope_not_hidden():
    def processor(case, inventory, statuses):
        return {"case": case, "connectivity": {"connected": bool(statuses["b2"]),
                                                "components": [[1], [2]] if not statuses["b2"] else [[1, 2]]}}

    runtime, _, _, _, calls = _fixture(processor=processor)
    result = runtime.scan_candidates()
    assert len(calls) == 4
    assert result["decision"] == "unique_within_declared_scope"
    assert any(row["resolution"] == "excluded" and not row["connectivity"]["connected"] for row in result["candidates"])


def test_unknown_is_enumerated_not_replaced_with_normal_or_wrong_binary():
    runtime, _, _, _, calls = _fixture(statuses={"b0": None, "b1": 0, "b2": 1, "c": 1})
    assert runtime.inspect_cb("b0")["status_known"] is False
    initial = runtime.test_statuses({})
    assert initial["reason"] == "unknown_statuses_require_binary_hypotheses"
    assert calls == []
    audit = runtime.scan_candidates(device_ids=[])
    assert audit["hypothesis_scope"]["unknown_binary_assignment_count"] == 2
    assert audit["decision"] == "unique_within_declared_scope"
    assert {call[1]["b0"] for call in calls} == {0, 1}
    assert runtime.apply(audit["unique_candidate_id"])["current_statuses"]["b0"] == 1
    many, _, _, _, calls = _fixture(statuses={key: None for key in ("b0", "b1", "b2", "c")})
    result = many.scan_candidates(max_unknown_hypotheses=2)
    assert result["decision"] == "request_status_information"
    assert not result["scope_complete"]
    assert calls == []


def test_cached_trials_and_stale_parent_or_evidence_are_guarded():
    runtime, _, _, _, calls = _fixture()
    first = runtime.test_cb("b0", 1)
    second = runtime.test_cb("b0", 1)
    assert first == second and len(calls) == 1
    audit = runtime.scan_candidates()
    runtime._case["branch"][0, 2] += .01
    with pytest.raises(ValueError, match="parent model changed"):
        runtime.apply(audit["unique_candidate_id"])
    runtime, _, _, _, _ = _fixture()
    audit = runtime.scan_candidates()
    runtime._sensors["covariance"][0][0] *= 2
    with pytest.raises(ValueError, match="observations or covariance changed"):
        runtime.apply(audit["unique_candidate_id"])


def test_absolute_nr_and_rank_calibration_override_any_reported_good_fit():
    runtime, _, _, _, _ = _fixture()
    local_alarm = runtime._calibrate(_metrics(1, 4.01, plausible=True))
    assert not local_alarm["plausible"] and local_alarm["normalized_residual_alarm"]
    assert local_alarm["chi_square_dof"] == 15
    assert local_alarm["chi_square_threshold"] == pytest.approx(chi2.ppf(.95, 15))
    unobservable = runtime._calibrate(_metrics(0, 0, rank=4, observable=False))
    assert unobservable["resolution"] == "unresolved"
    assert unobservable["chi_square_dof"] == 16


def test_real_section_model_tests_coupler_without_any_branch_status_edit():
    from pypower.api import ppoption, runpf
    from psse_env.systems import resolve_system
    from logical_topology.inventory import build_inventory, process_topology
    from logical_topology.measurements import build_measurement_inventory, expected_measurements, sample_measurements

    case = resolve_system("case14").load_case()
    inventory = build_inventory("case14", split_buses=[4])
    true_statuses = dict(inventory["normal_statuses"])
    physical = process_topology(case, inventory, true_statuses)
    solution, success = runpf(physical["case"], ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
    assert success
    sensors = build_measurement_inventory(inventory)
    observations = sample_measurements(expected_measurements(case, inventory, true_statuses, solution, sensors), sensors, noise=False)
    device = inventory["couplers"][0]["device_id"]
    modeled = {**true_statuses, device: 0}
    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses=modeled,
                                     measurement_inventory=sensors, observations=observations)
    before = runtime.snapshot()
    correct = runtime.test_cb(device, 1)
    assert correct["plausible"], correct
    assert correct["device_kinds"][device] == "bus_coupler"
    np.testing.assert_array_equal(runtime.snapshot()["current_case"]["branch"], before["current_case"]["branch"])
    assert evidence_hash(runtime.snapshot()["observations"]) == evidence_hash(observations)
    assert correct["estimation"]["raw_measurement_count"] == 125
    assert correct["estimation"]["electrical_bus_count"] == 14
    supported = runtime.scan_candidates(device_ids=[device])
    assert supported["comparison_guard"]["allowed"], supported["comparison_guard"]
    assert runtime.apply(supported["unique_candidate_id"])["current_statuses"][device] == 1


def _real_branch_fixture(*, status=1, profile="direct"):
    from pypower.api import ppoption, runpf
    from psse_env.systems import resolve_system
    from logical_topology.inventory import build_inventory, process_topology
    from logical_topology.measurements import build_measurement_inventory, expected_measurements, sample_measurements

    case = resolve_system("case14").load_case()
    inventory = build_inventory("case14", split_buses=[])
    statuses = dict(inventory["normal_statuses"])
    device_id = inventory["branches"][0]["device_id"]
    statuses[device_id] = status
    processed = process_topology(case, inventory, statuses)
    solution, success = runpf(processed["case"], ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
    assert success
    sensors = build_measurement_inventory(inventory, profile)
    observations = sample_measurements(expected_measurements(case, inventory, statuses, solution, sensors), sensors, noise=False)
    return case, inventory, statuses, sensors, observations, device_id


def test_off_branch_necessary_condition_matches_independent_full_wls():
    from logical_topology.estimation import estimate

    case, inventory, statuses, sensors, observations, device = _real_branch_fixture()
    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses=statuses,
                                     measurement_inventory=sensors, observations=observations)
    rejected = runtime.test_cb(device, 0)
    assert rejected["resolution"] == "analytical_rejection"
    assert rejected["solver_executed"] is False
    assert rejected["estimation"] is None  # no fabricated J or rank
    witness = rejected["analytical_proof"]
    assert witness["normalized_residual_lower_bound"] >= 4
    independent = estimate(case, inventory, {**statuses, device: 0}, observations, sensors)
    assert not independent["plausible"]
    index = observations["sensor_ids"].index(witness["sensor_id"])
    assert independent["predicted_values"][index] == 0
    assert independent["normalized_residuals"][index] == pytest.approx(witness["normalized_residual_lower_bound"], rel=1e-10)
    # The wrong initial model still gets a real full WLS report, followed by
    # this state-independent necessary-condition rejection if needed.
    wrong = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses={**statuses, device: 0},
                                   measurement_inventory=sensors, observations=observations)
    baseline = wrong.test_statuses({})
    assert baseline["solver_executed"] is True and baseline["estimation"] is not None
    assert baseline["resolution"] == "analytical_rejection"
    scan = wrong.scan_candidates(device_ids=[device])
    assert scan["decision"] == "unique_within_declared_scope"


def test_zero_flow_control_and_masked_flow_rows_do_not_trigger_analytical_rejection():
    case, inventory, statuses, sensors, observations, device = _real_branch_fixture(status=0)
    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses=statuses,
                                     measurement_inventory=sensors, observations=observations)
    healthy = runtime.test_statuses({})
    assert healthy["plausible"]
    assert healthy["solver_executed"] is True
    assert "analytical_proof" not in healthy
    assert runtime.scan_candidates(device_ids=[device])["decision"] == "keep_current"
    case, inventory, statuses, sensors, observations, device = _real_branch_fixture(profile="indirect_even")
    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses=statuses,
                                     measurement_inventory=sensors, observations=observations)
    masked = runtime.test_cb(device, 0)
    assert masked["solver_executed"] is True
    assert "analytical_proof" not in masked
    masked_indices = [index for index, available in enumerate(sensors["available_mask"]) if not available]
    assert masked_indices
    assert all(runtime.snapshot()["observations"]["values"][index] is None for index in masked_indices)
    # Missing telemetry for one branch must not disable valid proof witnesses
    # from the remaining independently available branch-flow sensors.
    available_device = inventory["branches"][1]["device_id"]
    proved = runtime.test_cb(available_device, 0)
    assert proved["resolution"] == "analytical_rejection"
    assert proved["analytical_proof"]["branch_row0"] == 1


def test_actual_open_branch_with_unknown_indication_is_not_defaulted_closed():
    case, inventory, statuses, sensors, observations, device = _real_branch_fixture(status=0)
    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case,
                                     current_statuses={**statuses, device: None},
                                     measurement_inventory=sensors, observations=observations)
    # This declared scope asks only for the unknown indication's two values;
    # it does not assume either value or generate different observations.
    result = runtime.scan_candidates(device_ids=[])
    assert result["hypothesis_scope"]["unknown_binary_assignment_count"] == 2
    assert result["comparison_guard"]["allowed"], result["comparison_guard"]
    final = runtime.apply(result["unique_candidate_id"])
    assert final["current_statuses"][device] == 0
    assert final["current_case"]["branch"][0, 10] == 0
    assert final["fixed_evidence_hash"] == result["fixed_evidence_hash"]


def test_necessary_condition_remains_valid_with_correlated_sensor_covariance():
    import hashlib
    import json
    from logical_topology.estimation import estimate

    case, inventory, statuses, sensors, observations, device = _real_branch_fixture()
    flow = next(index for index, row in enumerate(sensors["records"]) if row["kind"] == "Pf" and row["branch_row0"] == 0)
    covariance = sensors["covariance"]
    covariance[0][flow] = covariance[flow][0] = .2 * np.sqrt(covariance[0][0] * covariance[flow][flow])
    updated = {key: value for key, value in sensors.items() if key != "sensor_inventory_hash"}
    sensors["sensor_inventory_hash"] = hashlib.sha256(json.dumps(updated, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    observations["sensor_inventory_hash"] = sensors["sensor_inventory_hash"]
    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses=statuses,
                                     measurement_inventory=sensors, observations=observations)
    proof = runtime.test_cb(device, 0)["analytical_proof"]
    fit = estimate(case, inventory, {**statuses, device: 0}, observations, sensors)
    index = observations["sensor_ids"].index(proof["sensor_id"])
    assert fit["normalized_residuals"][index] == pytest.approx(proof["normalized_residual_lower_bound"], rel=1e-10)


def test_threshold_crossing_without_model_separation_does_not_authorize_a_flip():
    threshold = float(chi2.ppf(.95, 15))

    def near_threshold(case, inventory, statuses, observations, sensors, **kwargs):
        if statuses == {"b0": 0, "b1": 0, "b2": 1, "c": 1}:
            return _metrics(threshold + .5, 3.)
        if statuses == {"b0": 1, "b1": 0, "b2": 1, "c": 1}:
            return _metrics(threshold - .5, 3.)
        return _metrics(200, 10.)

    runtime, _, _, _, _ = _fixture(estimator=near_threshold)
    result = runtime.scan_candidates()
    assert result["absolute_unique_candidate_id"] is not None
    assert result["unique_candidate_id"] is None
    assert result["decision"] == "insufficient_calibrated_status_identifiability"
    assert not result["comparison_guard"]["allowed"]
    assert result["current"]["candidate_id"] in result["comparison_guard"]["blocking_candidates"]
    assert any(row["candidate_id"] == result["current"]["candidate_id"] for row in result["comparison_compatible_candidates"])
    with pytest.raises(ValueError, match="certificate"):
        runtime.apply(result["absolute_unique_candidate_id"])


def test_basic_or_tampered_certificates_cannot_bypass_the_comparison_guard():
    runtime, _, _, _, _ = _fixture()
    result = runtime.scan_candidates()
    assert result["comparison_guard"]["allowed"]
    del runtime._certificate["comparison_guard"]
    with pytest.raises(ValueError, match="comparison-guard"):
        runtime.apply(result["unique_candidate_id"])
    result = runtime.scan_candidates()
    runtime._certificate["comparison_guard"]["familywise_alpha"] = .5
    with pytest.raises(ValueError, match="comparison-guard"):
        runtime.apply(result["unique_candidate_id"])
    result = runtime.scan_candidates()
    runtime._certificate["comparison_guard"]["contract"] = "logical_topology_pairwise_separation_guard_v1"
    runtime._certificate["comparison_guard_hash"] = evidence_hash(runtime._certificate["comparison_guard"])
    with pytest.raises(ValueError, match="comparison-guard"):
        runtime.apply(result["unique_candidate_id"])


@pytest.mark.parametrize("full_fit_outcome", ("rejected", "plausible", "unresolved"))
def test_weak_analytic_witness_gets_full_fit_and_refreshes_original_scope_membership(full_fit_outcome):
    case, inventory, _, sensors, observations, weak_device = _real_branch_fixture()
    for index, row in enumerate(sensors["records"]):
        if row["kind"] in {"Pf", "Qf", "Pt", "Qt"} and row["branch_row0"] == 0:
            observations["values"][index] = .04001
    actual = dict(inventory["normal_statuses"])
    corrected_device = inventory["branches"][1]["device_id"]
    reported = {**actual, corrected_device: 0}
    weak_rival = {**actual, weak_device: 0}
    called = []

    def fit(case, inventory, statuses, observations, sensors, **kwargs):
        called.append(dict(statuses))
        base = {"rank": 27, "state_dimension": 27, "available_measurement_count": 122}
        if statuses == actual:
            return _metrics(1., 1., **base)
        if statuses == weak_rival:
            if full_fit_outcome == "unresolved":
                return {"converged": False, "failure_reason": "explicit_refinement_failure"}
            return _metrics(50. if full_fit_outcome == "rejected" else 2.,
                            5. if full_fit_outcome == "rejected" else 1., **base)
        return _metrics(1000., 30., **base)

    runtime = LogicalTopologyRuntime(inventory=inventory, current_case=case, current_statuses=reported,
                                     measurement_inventory=sensors, observations=observations, estimator=fit)
    result = runtime.scan_candidates(include_pairs=True, max_pairs=5000)
    assert result["tested_candidate_count"] == 211
    assert result["full_fit_refinement_candidate_ids"]
    assert weak_rival in called
    refined = next(row for row in result["candidates"] if row["statuses"] == weak_rival)
    assert refined["estimation"] is not None
    if full_fit_outcome == "rejected":
        assert result["comparison_guard"]["allowed"], result["comparison_guard"]
        assert runtime.apply(result["unique_candidate_id"])["current_statuses"] == actual
    else:
        assert result["unique_candidate_id"] is None
        assert not result["comparison_guard"]["allowed"]
        assert refined["candidate_id"] in result["comparison_guard"]["blocking_candidates"]
        if full_fit_outcome == "plausible":
            assert len(result["plausible_candidates"]) == 2
        else:
            assert result["unresolved_candidate_count"] == 1
