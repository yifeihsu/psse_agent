from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pypower.api import case14, ppoption, runpf
from threadpoolctl import threadpool_limits

from psse_env.dagger.dataset_builder import validate_policy_payload
from psse_env.fault_profiles import measurement_sigma
from research import reviewed_expert_admission as admission
from Transmission.generate_measurements import compute_measurements_pu


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def snapshot():
    case = case14()
    solved, ok = runpf(case, ppoption(VERBOSE=0, OUT_ALL=0))
    assert ok
    sigma = measurement_sigma(14, 20)
    observed = compute_measurements_pu(solved) + np.random.default_rng(91).normal(0, sigma)
    return case, observed, sigma, solved


def test_quiet_healthy_snapshot_executes_observable_finalize_without_fault_eligibility(snapshot):
    case, z, sigma, _ = snapshot
    receipt = admission.probe_expert_action(case, z, sigma)
    assert receipt["contract"] == "wls_observable_expert_prefix_v1"
    assert receipt["initial_wls"]["success"] and receipt["initial_wls"]["alarm"] is False
    assert receipt["selected_action"]["tool"] == "finalize_diagnosis"
    assert receipt["healthy_completion_valid"] and receipt["safe_finalize"]
    assert receipt["action_executed"] and receipt["process_evidence_valid"]
    assert not receipt["fault_actionable"] and not receipt["training_eligible"]
    assert not receipt["full_repair_validated"]
    assert not receipt["gnn_enabled"] and not receipt["external_family_flags_used"]
    assert len(receipt["events"]) == 2
    assert receipt["events"][0]["preferred_action"]["tool"] == "run_wls"
    assert receipt["events"][0]["policy_observation"]["remaining_budget"] == 40
    assert receipt["request_counts"]["attempted"] == 0
    json.dumps(receipt, allow_nan=False)


def test_real_meter_fault_keeps_negative_acquisitions_and_executes_supported_context(snapshot):
    case, z, sigma, _ = snapshot
    z[45] += 15 * sigma[45]
    receipt = admission.probe_expert_action(case, z, sigma)
    assert receipt["initial_wls"]["success"] and receipt["initial_wls"]["alarm"]
    assert receipt["initial_wls"]["J"] > receipt["initial_wls"]["threshold"]
    assert receipt["fault_actionable"] and receipt["training_eligible"]
    assert not receipt["healthy_completion_valid"]
    assert receipt["selected_action"]["tool"] == "get_measurement_context"
    assert receipt["available_context_tool"] == "get_measurement_context"
    assert receipt["acquired_evidence_available"]
    assert receipt["actionable_event_index"] == 3
    assert receipt["actionable_event"] == receipt["events"][3]
    assert receipt["request_counts"]["unavailable_or_empty"] == 2
    assert receipt["request_counts"]["available"] == 1
    for event in receipt["events"]:
        validate_policy_payload(event["policy_observation"])
        assert event["preferred_action"] in event["ordered_actions"]
        assert event["process_evidence_valid"] and event["execution_success"]
        assert "production_label_eligible" not in event
    for event in receipt["events"][1:3]:
        assert event["action_executed"] and not event["available"]
        assert not event["qualifying_action"]
    assert any(45 in rec["arguments"].get("suspect_group", []) for rec in receipt["supported_recommendations"])
    assert receipt["reason"] == "evidenced_expert_prefix"


def test_available_phase_acquisition_is_a_prefix_not_a_fault_identification(snapshot):
    case, z, sigma, solved = snapshot
    z[45] += .15
    voltage = solved["bus"][0, 7] * np.exp(1j * np.deg2rad([0., -120., 120.]))
    rng = np.random.default_rng(511)
    measured = voltage + rng.normal(0, .005, 3) + 1j * rng.normal(0, .005, 3)
    metadata = {"three_phase_voltages": [{"bus": "b1", "vln_pu": abs(measured).tolist(),
                                         "ang_deg": np.rad2deg(np.angle(measured)).tolist()}],
                "three_phase_sigma": .005}
    receipt = admission.probe_expert_action(case, z, sigma, observable_metadata=metadata)
    assert receipt["fault_actionable"]
    assert receipt["selected_action"]["tool"] == "get_three_phase_context"
    assert receipt["step_result"]["tool_metrics"]["measured_buses"] == [1]
    assert receipt["request_counts"]["available"] == 1
    assert receipt["request_counts"]["attempted"] == 1
    assert not receipt["full_repair_validated"]
    assert receipt["supported_recommendations"] == []


def test_flat_initialization_and_input_snapshot_are_preserved(snapshot, monkeypatch):
    case, z, sigma, _ = snapshot
    case["bus"][:, 7] = np.linspace(.5, 1.7, 14)
    case["bus"][:, 8] = np.linspace(-150, 150, 14)
    before = deepcopy(case)
    original_z, original_sigma = z.copy(), sigma.copy()
    captured = []
    real_write = admission.write_ppc_as_matpower_m

    def write(configured, *args, **kwargs):
        captured.append(deepcopy(configured))
        return real_write(configured, *args, **kwargs)

    monkeypatch.setattr(admission, "write_ppc_as_matpower_m", write)
    receipt = admission.probe_expert_action(case, z, sigma)
    assert receipt["healthy_completion_valid"]
    np.testing.assert_array_equal(captured[0]["bus"][:, 7], np.ones(14))
    np.testing.assert_array_equal(captured[0]["bus"][:, 8], np.zeros(14))
    np.testing.assert_array_equal(captured[0]["branch"], case["branch"])
    for key in ("bus", "branch", "gen"):
        np.testing.assert_array_equal(case[key], before[key])
    np.testing.assert_array_equal(z, original_z)
    np.testing.assert_array_equal(sigma, original_sigma)


def test_unavailable_contexts_cannot_qualify_when_short_budget_ends(snapshot):
    case, z, sigma, _ = snapshot
    z[45] += .15
    receipt = admission.probe_expert_action(case, z, sigma, max_actions=3)
    assert receipt["wls_alarm"]
    assert not receipt["fault_actionable"]
    assert not receipt["healthy_completion_valid"]
    assert not receipt["acquired_evidence_available"]
    assert receipt["executed_action_count"] == 3
    assert receipt["request_counts"]["unavailable_or_empty"] == 1
    # The real expert sees the short budget and chooses its existing handoff
    # after the first negative acquisition; the probe must not invent a route.
    assert receipt["reason"] == "operator_handoff_not_actionable_training"
    assert receipt["events"][0]["policy_observation"]["remaining_budget"] == 3


@pytest.mark.parametrize("failure", ["wls", "action", "evidence"])
def test_failed_wls_failed_execution_or_invalid_evidence_never_qualifies(snapshot, monkeypatch, failure):
    case, z, sigma, _ = snapshot
    z[45] += .15
    real_factory = admission.research_diagnostic_environment_factory

    def factory(**kwargs):
        env = real_factory(**kwargs)
        real_step, real_evidence = env.step, env.assert_training_decision_evidence
        def step(action):
            should_fail = (failure == "wls" and action["tool"] == "run_wls") or (failure == "action" and action["tool"] != "run_wls")
            if should_fail:
                return env.current_state(), {"execution_status": "failure", "error_code": "test_backend_failure", "tool_metrics": {"converged": False}}
            return real_step(action)
        def evidence(action):
            if failure == "evidence" and action["tool"] != "run_wls":
                raise ValueError("test_missing_observable_evidence")
            return real_evidence(action)
        env.step, env.assert_training_decision_evidence = step, evidence
        return env

    monkeypatch.setattr(admission, "research_diagnostic_environment_factory", factory)
    receipt = admission.probe_expert_action(case, z, sigma)
    assert not receipt["fault_actionable"] and not receipt["healthy_completion_valid"]
    assert receipt["actionable_event"] is None
    if failure == "wls":
        assert not receipt["initial_wls"]["success"]
        assert receipt["wls_alarm"] is None
    elif failure == "action":
        assert receipt["process_evidence_valid"] and not receipt["action_executed"]
    else:
        assert not receipt["process_evidence_valid"]


def test_expert_handoff_or_no_action_is_not_replaced_by_a_fabricated_action(snapshot, monkeypatch):
    case, z, sigma, _ = snapshot
    z[45] += .15
    real_select = admission.select_observable_expert_actions
    calls = []
    def choose(*, policy_observation, expert_oracle):
        calls.append(deepcopy(policy_observation))
        if len(calls) == 1:
            return real_select(policy_observation=policy_observation, expert_oracle=expert_oracle)
        return SimpleNamespace(preferred_action=None, actions=(), selection_basis="test_no_action")
    monkeypatch.setattr(admission, "select_observable_expert_actions", choose)
    receipt = admission.probe_expert_action(case, z, sigma)
    assert receipt["reason"] == "observable_expert_returned_no_action"
    assert not receipt["fault_actionable"]
    assert len(receipt["events"]) == 1
    for observed in calls:
        validate_policy_payload(observed)


@pytest.mark.parametrize("metadata", [
    {"families": ["hif"]}, {"offline_audit": {"J_exact": 999}},
    {"unresolved_signatures": ["hif_detected"]}, {"measurement_kind": "noiseless_mean"},
    {"three_phase_voltages": [{"bus": "b1", "z_clean": [1]}]},
    {"parameter_scans": {"initial_states": [[1., 0.]]}}, {"three_phase_sigma": 0},
])
def test_private_flags_noiseless_inputs_and_latent_initialization_are_rejected(snapshot, metadata):
    case, z, sigma, _ = snapshot
    with pytest.raises(ValueError):
        admission.probe_expert_action(case, z, sigma, observable_metadata=metadata)


def test_wls_feature_whitelist_cannot_be_misrepresented_as_full_operator_execution_case(snapshot):
    from research.gnn_screen.wls_features import configured_case
    case, z, sigma, _ = snapshot
    with pytest.raises(ValueError, match="full public operator case|voltage bounds"):
        admission.probe_expert_action(configured_case(case), z, sigma)


def test_auxiliary_only_noise_contract_is_validated_against_acquired_sigmas(snapshot):
    from three_phase_nlm.measurement_noise import generated_noise_contract

    case, z, sigma, _ = snapshot
    contract = generated_noise_contract(sigma, noise_scale=1., three_phase_sigma=.005, branch_current_sigma_pu=.001)
    contract["channels"].pop("scada")
    contract.update(scada_noise_drawn_here=False, generation_scope="auxiliary_phase_sensors_only")
    metadata = {"noise_contract": contract, "three_phase_sigma": .005, "branch_current_sigma_pu": .001,
                "sigma_z": sigma.tolist()}
    result = admission.probe_expert_action(case, z, sigma, observable_metadata=metadata)
    assert result["healthy_completion_valid"]
    metadata["three_phase_sigma"] = .007
    with pytest.raises(ValueError, match="estimator sigma disagrees"):
        admission.probe_expert_action(case, z, sigma, observable_metadata=metadata)


def test_exact_injection_rows_are_preserved_as_constraints_not_floored_variances(snapshot):
    case, z, sigma, _ = snapshot
    z[[20, 34]] = 0
    sigma[[20, 34]] = 0
    result = admission.probe_expert_action(case, z, sigma,
        observable_metadata={"structural_zero_indices": [20, 34]})
    assert result["initial_wls"]["success"]
    metrics = result["initial_wls"]["tool_output"]["tool_metrics"]
    assert metrics["chi_square_dof"] == 95
    assert result["healthy_completion_valid"]
    with pytest.raises(ValueError, match="Zero sigmas"):
        admission.probe_expert_action(case, z, sigma)


def test_saved_reviewed_context_builder_to_actual_probe_integration():
    """Use the saved real canary when present; keep native DSS in its own process."""
    root = Path(__file__).resolve().parents[1]
    manifest = root / "output/reviewed_fault_scenarios_20260917/validated_bundle/baseline/core/manifest.jsonl"
    if not manifest.is_file():
        pytest.skip("saved reviewed physical canary is not present")
    script = """
import opendssdirect
from pathlib import Path
import json, sys
from research.gnn_screen.dataset import load_manifest
from research.reviewed_observable_context import build_observable_context
from research.filter_reviewed_training import reported_runtime_case, materialize_training_windows
from research.reviewed_expert_admission import probe_expert_action
path = Path(sys.argv[1])
source = next(row for row in load_manifest(path) if row['split'] == 'train' and row['families'] == ['measurement'])
metadata, private_receipt = build_observable_context(path, source, noise_seed=9123)
case, operator_receipt = reported_runtime_case(path, source)
observed = materialize_training_windows(source)[0]
receipt = probe_expert_action(case, observed['z'], observed['measurement_sigma'], observable_metadata=metadata)
assert receipt['wls_success'] and receipt['wls_alarm']
assert receipt['fault_actionable'] and receipt['actionable_event_index'] == 1
assert receipt['selected_action']['tool'] == 'get_three_phase_context'
assert receipt['actionable_event']['process_evidence_valid']
assert receipt['actionable_event']['execution_success'] and receipt['actionable_event']['available']
assert receipt['request_counts']['available'] == 1
assert not receipt['full_repair_validated']
print(json.dumps({'integration_passed': True, 'selected_tool': receipt['selected_action']['tool']}))
"""
    completed = subprocess.run([sys.executable, "-c", script, str(manifest)], cwd=root,
                               capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(completed.stdout.splitlines()[-1])["integration_passed"]
