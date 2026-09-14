from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json

import pytest

import logical_topology.audit as audit_module
from logical_topology.audit import evaluate_scenario
from logical_topology.inventory import build_inventory
from logical_topology.measurements import build_measurement_inventory, expected_measurements, sample_measurements
from logical_topology.scenarios import solve_true_world, write_json
from psse_env.systems import resolve_system


@pytest.fixture(scope="module")
def worlds():
    case = resolve_system("case14").load_case()
    inventory = build_inventory("case14", split_buses=[])
    closed = dict(inventory["normal_statuses"])
    outage = {**closed, inventory["branches"][-1]["device_id"]: 0}
    result = {}
    for name, statuses in (("closed", closed), ("outage", outage)):
        physical = solve_true_world(case, inventory, statuses)
        assert physical["admitted"], physical
        result[name] = (inventory, statuses, physical)
    return result


def _row(root, worlds, *, name="scenario", world="closed", changes=None, profile="direct", parameter_overlay=False, cardinality=1):
    inventory, true_statuses, physical = deepcopy(worlds[world])
    sensors = build_measurement_inventory(inventory, profile)
    values = expected_measurements(physical["operating_case"], inventory, true_statuses, physical["solution"], sensors)
    observations = sample_measurements(values, sensors, noise=False)
    model_statuses = {**true_statuses, **(changes or {})}
    case = deepcopy(physical["operating_case"])
    if parameter_overlay:
        case["branch"][1, 2:4] *= 2
    prefix = root / name
    for filename, payload in (("inventory.json", inventory), ("sensors.json", sensors),
                              ("observations.json", observations), ("case.json", case),
                              ("physical.json", {**physical, "true_statuses": true_statuses, "offline_secret": "DO_NOT_SEND_TO_RUNTIME"})):
        write_json(prefix / filename, payload)
    row = {"scenario_id": name, "family": "test", "layout": "branch_status",
           "measurement_profile": profile, "load_scale": 1., "hypothesis_cardinality": cardinality,
           "physical_admission": {"admitted": True, "physics": physical["physics"]},
           "true_statuses": true_statuses, "physical_audit_path": f"{name}/physical.json",
           "error_device_ids": ["DO_NOT_USE_THIS_LABEL_TO_SELECT_HYPOTHESES"],
           "execution": {"inventory_path": f"{name}/inventory.json", "measurement_inventory_path": f"{name}/sensors.json",
                         "observations_path": f"{name}/observations.json", "base_case_path": f"{name}/case.json",
                         "current_statuses": model_statuses}}
    return row


def test_physical_reject_never_constructs_runtime_or_estimates(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Physically rejected root reached an estimator")
    monkeypatch.setattr(audit_module, "LogicalTopologyRuntime", forbidden)
    monkeypatch.setattr(audit_module, "estimate", forbidden)
    report = evaluate_scenario(tmp_path, {"scenario_id": "rejected", "physical_admission": {
        "admitted": False, "reason": "islanding_outside_connected_operating_scope"}})
    assert report["runtime_decision"] == "physical_reject"
    assert not report["audit_outcomes"]["state_observability"]["evaluated"]
    assert report["initial_estimation"]["failure_reason"] == "physical_reject_no_wls"
    assert report["detailed_audit_path"] is None


def test_runtime_boundary_precedes_truth_scoring_and_all_cb_alternatives_are_tested(tmp_path, worlds, monkeypatch):
    device = worlds["closed"][0]["branches"][0]["device_id"]
    row = _row(tmp_path, worlds, changes={device: 0})
    original_runtime, original_estimate = audit_module.LogicalTopologyRuntime, audit_module.estimate
    events = []

    class ObservedRuntime(original_runtime):
        def __init__(self, **kwargs):
            assert set(kwargs) == {"inventory", "current_case", "current_statuses", "measurement_inventory",
                                   "observations", "chi2_alpha", "normalized_residual_threshold"}
            assert "DO_NOT_SEND_TO_RUNTIME" not in json.dumps(kwargs)
            assert "DO_NOT_USE_THIS_LABEL" not in json.dumps(kwargs)
            super().__init__(**kwargs)

        def scan_candidates(self, **kwargs):
            assert "device_ids" not in kwargs and "pair_devices" not in kwargs
            value = super().scan_candidates(**kwargs)
            events.append("scan_complete")
            return value

        def apply(self, candidate_id):
            result = super().apply(candidate_id)
            events.append("apply_complete")
            return result

    def offline_estimate(*args, **kwargs):
        assert events[:2] == ["scan_complete", "apply_complete"]
        events.append("offline_truth_fit")
        return original_estimate(*args, **kwargs)

    monkeypatch.setattr(audit_module, "LogicalTopologyRuntime", ObservedRuntime)
    monkeypatch.setattr(audit_module, "estimate", offline_estimate)
    report = evaluate_scenario(tmp_path, row)
    assert report["runtime_decision"] == "unique_within_declared_scope"
    assert report["correction_applied"]
    assert report["status_audit"]["exact_status_recovery"]
    assert report["status_audit"]["false_correction_count"] == 0
    assert report["preservation"]["passed"]
    assert report["audit_outcomes"]["state_observability"]["observable"]
    assert report["audit_outcomes"]["state_observability"]["assessment"] == "observable"
    assert report["audit_outcomes"]["status_identifiability"]["tested_candidate_count"] == 21
    assert report["state_estimation_error"]["final"]["voltage_magnitude_max_abs_error_pu"] < 1e-7
    detailed = tmp_path / report["detailed_audit_path"]
    assert hashlib.sha256(detailed.read_bytes()).hexdigest() == report["detailed_audit_sha256"]
    with gzip.open(detailed, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert "true_statuses" not in json.dumps(payload["runtime_scan"])
    assert "offline_true_physical_status_fit" in payload


def test_correct_outage_is_retained_and_sparse_unknown_remains_an_audited_condition(tmp_path, worlds):
    healthy = evaluate_scenario(tmp_path, _row(tmp_path, worlds, name="healthy_outage", world="outage"))
    assert healthy["runtime_decision"] == "keep_current"
    assert healthy["status_audit"]["healthy_cb_preserved"]
    assert healthy["status_audit"]["exact_status_recovery"]
    assert healthy["status_audit"]["changed_statuses"] == {}
    device = worlds["closed"][0]["branches"][0]["device_id"]
    unknown = evaluate_scenario(tmp_path, _row(tmp_path, worlds, name="unknown", changes={device: None}))
    assert unknown["status_audit"]["initial_unknown_count"] == 1
    assert unknown["status_audit"]["initial_error_count"] == 0
    # This operating point also permits weakly excited alternatives on other
    # assets. The all-device audit must retain ambiguity instead of narrowing
    # its search to the offline label and forcing the unknown indication.
    assert unknown["runtime_decision"] == "ambiguous_candidate_set"
    assert unknown["status_audit"]["final_unknown_count"] == 1
    assert unknown["initial_estimation"]["available"] is False
    sparse = evaluate_scenario(tmp_path, _row(tmp_path, worlds, name="sparse", profile="voltage_only", changes={device: 0}))
    assert sparse["physical_admitted"]
    assert sparse["audit_outcomes"]["state_observability"]["observable"] is False
    assert sparse["audit_outcomes"]["state_observability"]["assessment"] == "unobservable"
    assert not sparse["correction_applied"]
    assert sparse["preservation"]["observations_unchanged"]


def test_pairs_disabled_or_budget_truncated_never_earn_a_pair_scope_certificate(tmp_path, worlds):
    inventory = worlds["closed"][0]
    changes = {inventory["branches"][index]["device_id"]: 0 for index in (0, 2)}
    row = _row(tmp_path, worlds, name="pair_disabled", changes=changes, cardinality=2)
    report = evaluate_scenario(tmp_path, row, scan_pairs=False)
    assert not report["correction_applied"]
    assert not report["audit_outcomes"]["status_identifiability"]["scope_complete"]
    assert not report["audit_outcomes"]["status_identifiability"]["pair_stratum_fully_searched"]
    full = evaluate_scenario(tmp_path, _row(tmp_path, worlds, name="pair_complete", changes=changes, cardinality=2))
    assert full["status_audit"]["exact_status_recovery"]
    assert full["audit_outcomes"]["status_identifiability"]["tested_candidate_count"] == 211
    assert full["audit_outcomes"]["status_identifiability"]["scope_complete"]


def test_observability_uses_true_physical_parameters_but_topology_apply_preserves_model_overlay(tmp_path, worlds):
    device = worlds["closed"][0]["branches"][0]["device_id"]
    row = _row(tmp_path, worlds, name="parameter_overlay", changes={device: 0}, parameter_overlay=True)
    model_path = tmp_path / row["execution"]["base_case_path"]
    original = model_path.read_bytes()
    report = evaluate_scenario(tmp_path, row)
    true_model = report["audit_outcomes"]["state_observability"]
    assert true_model["true_parameter_case_used"]
    assert true_model["model_parameters_match_truth"] is False
    assert true_model["observable"] and true_model["plausible"]
    assert report["offline_true_status_model_fit"]["plausible"] is False
    assert report["preservation"]["nonstatus_case_unchanged"]
    assert model_path.read_bytes() == original
    assert report["offline_observability_fit_used_for_runtime_decisions"] is False


def test_wrong_model_numerical_failure_is_not_a_physical_root_rejection(tmp_path, worlds, monkeypatch):
    device = worlds["closed"][0]["branches"][0]["device_id"]
    row = _row(tmp_path, worlds, name="wrong_model_failure", changes={device: 0})
    original_runtime, actual_estimate = audit_module.LogicalTopologyRuntime, audit_module.estimate

    class RuntimeWithWrongModelFailure(original_runtime):
        def __init__(self, **kwargs):
            def controlled_failure(case, inventory, statuses, observations, sensors, **settings):
                if statuses[device] == 0:
                    return {"converged": False, "observable": False, "plausible": False,
                            "failure_reason": "injected_wrong_model_nonconvergence"}
                return actual_estimate(case, inventory, statuses, observations, sensors, **settings)
            super().__init__(estimator=controlled_failure, **kwargs)

    monkeypatch.setattr(audit_module, "LogicalTopologyRuntime", RuntimeWithWrongModelFailure)
    report = evaluate_scenario(tmp_path, row)
    assert report["physical_admitted"]
    assert report["wrong_model_wls_failed"]
    assert report["initial_estimation"]["converged"] is False
    assert report["initial_hypothesis_resolution"] == "analytical_rejection"
    assert report["correction_applied"] and report["status_audit"]["exact_status_recovery"]
    assert report["audit_outcomes"]["state_observability"]["observable"]
