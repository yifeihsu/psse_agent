from copy import deepcopy
import json

import numpy as np
import pytest
from pypower.api import case57, ppoption, runpf
from threadpoolctl import threadpool_limits

from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.dagger.ieee57_training import (
    collect_episode, export_audited_rows, replay_episode, supervision_reasons, _detector_outputs,
)


@pytest.fixture(scope="module")
def scenarios():
    with threadpool_limits(limits=1):
        solved, ok = runpf(case57(), ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
    assert ok
    z = build_measurement_vector(solved).tolist()
    base = {"scenario_id": "ieee57_training_unit_clean", "scenario_family": "no_error",
            "network_case": "case57", "case": "case57", "measurements": z,
            "source_tier": "unit_physical_pf",
            "clean_case": "case57", "clean_measurements": z, "metadata": {},
            "true_measurement_errors": [], "true_parameter_errors": [],
            "true_topology_errors": [], "error_cardinality": 0}
    clean = partition_release_scenario_v1(base, split="train")
    clean["grouping"].update(source_realization_id="parent_one", parent_construction_fingerprint="parent_one",
                             parent_plan_sha256="fixed_plan", dataset_split="train")
    hidden = deepcopy(clean)
    hidden["audit"]["truth"]["true_parameter_errors"] = [
        {"branch_row0": 31, "line_index1": 32, "parameter": "rx", "clean_r": .2, "clean_x": .3}
    ]
    return clean, hidden


@pytest.fixture(scope="module")
def collected(scenarios):
    with threadpool_limits(limits=1):
        return [collect_episode(s, max_steps=8) for s in scenarios]


def test_observable_targets_ignore_hidden_truth_and_quarantine_stopping(collected):
    clean, hidden = collected
    assert [r["preferred_action"]["tool"] for r in clean] == ["run_wls", "finalize_diagnosis"]
    assert [r["canonical_action"] for r in clean] == [r["canonical_action"] for r in hidden]
    assert clean[0]["model_view"] == hidden[0]["model_view"]
    assert all(r["production_label_eligible"] for r in clean)
    assert hidden[0]["production_label_eligible"]
    assert not hidden[-1]["production_label_eligible"]
    assert "offline_teacher_target_audit_failed" in hidden[-1]["quarantine_reasons"]


def test_export_keeps_valid_prefix_and_parent_without_truth_leak(collected):
    exported, audit = export_audited_rows(collected[1])
    assert audit["accepted"] == 1 and audit["quarantined"] == 1
    assert len(exported) == 1
    row = exported[0]
    assert row["dataset_split"] == "train" and row["source_realization_id"] == "parent_one"
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "wls_from_path"
    visible = json.dumps(row["messages"])
    assert "true_parameter_errors" not in visible and "clean_r" not in visible
    assert row["metadata"]["ieee57_supervision"]["target_audit_passed"]


def test_validation_and_test_rows_cannot_enter_training_export(collected):
    rows = deepcopy(collected[0])
    rows[0]["dataset_split"] = "validation"
    rows[1]["dataset_split"] = "test"
    exported, audit = export_audited_rows(rows)
    assert not exported and audit["accepted"] == 0


def test_replay_rebinds_fresh_ids_and_quarantined_prefix_context(scenarios, collected):
    exported, _ = export_audited_rows(collected[1])
    with threadpool_limits(limits=1):
        replay = replay_episode(scenarios[1], collected[1], exported)
    assert replay["passed"] and replay["steps"] == 2
    assert replay["exported_targets_replayed"] == 1
    assert all(c["fresh_alias_rebinding"] for c in replay["checks"])


def test_invalid_execution_never_exports_and_tampered_ledger_fails(collected):
    row = deepcopy(collected[0][0])
    row["tool_output"]["execution_status"] = "invalid_action"
    assert "protocol_execution_not_successful" in supervision_reasons(row)
    with pytest.raises(ValueError, match="ledger"):
        export_audited_rows([row])
    row["quarantine_reasons"] = supervision_reasons(row)
    row["production_label_eligible"] = False
    exported, audit = export_audited_rows([row])
    assert not exported and audit["quarantined"] == 1


def test_tampered_export_target_rejected_on_replay(scenarios, collected):
    exported, _ = export_audited_rows(collected[0])
    exported[0]["messages"][2]["tool_calls"][0]["function"]["name"] = "get_parameter_context"
    with threadpool_limits(limits=1), pytest.raises(ValueError, match="differs from fixed"):
        replay_episode(scenarios[0], collected[0], exported)


def test_actual_wls_evidence_cannot_omit_residual_detector(collected):
    out = deepcopy(collected[0][0]["tool_output"])
    out["tool_metrics"].pop("normalized_residual_threshold")
    with pytest.raises(ValueError, match="pinned"):
        _detector_outputs(out, "run_wls")
    out = deepcopy(collected[0][0]["tool_output"])
    out["tool_metrics"]["no_material_anomaly_remaining"] = False
    with pytest.raises(ValueError, match="OR rule"):
        _detector_outputs(out, "run_wls")
    with pytest.raises(ValueError, match="pinned"):
        _detector_outputs({"execution_status": "success", "tool_metrics": {}}, "verify_candidate")


def test_replay_rejects_unbound_extra_rows_and_final_state_tampering(scenarios, collected):
    exported, _ = export_audited_rows(collected[0])
    duplicate = deepcopy(exported[0]); duplicate["example_id"] = "unbound_extra"
    with pytest.raises(ValueError, match="absent"):
        replay_episode(scenarios[0], collected[0], exported+[duplicate])
    raw = deepcopy(collected[0]); raw[-1]["terminal_after_action"] = False
    with threadpool_limits(limits=1), pytest.raises(ValueError, match="post-action"):
        replay_episode(scenarios[0], raw, exported)


def test_replay_rejects_changed_authoritative_terminal_outcome(scenarios, collected):
    from psse_env.dagger.ieee57_runtime import ieee57_environment_factory
    exported, _ = export_audited_rows(collected[0])
    assert collected[0][-1]["terminal_outcome"] == "resolved"
    def wrong_terminal_factory():
        env = ieee57_environment_factory()
        original_step = env.step
        def step(action):
            result = original_step(action)
            if env.terminal_outcome == "resolved":
                env.terminal_outcome = "inconclusive"
            return result
        env.step = step
        return env
    with threadpool_limits(limits=1), pytest.raises(ValueError, match="terminal outcome"):
        replay_episode(scenarios[0], collected[0], exported, environment_factory=wrong_terminal_factory)


@pytest.mark.parametrize("failure", ["late_source_drift", "persisted_trace_tamper"])
def test_failed_release_cannot_leave_qualified_training_file(tmp_path, monkeypatch, scenarios, collected, failure):
    from scripts import run_ieee57_training_pilot as pilot
    output = tmp_path/"release"
    scenario = deepcopy(scenarios[0])
    def parents(path, **kwargs):
        path.mkdir()
        return {"complete": True, "raw_source_generation_complete": True}, [scenario]
    monkeypatch.setattr(pilot, "generate_parent_assigned_scenarios", parents)
    monkeypatch.setattr(pilot, "validate_parent_assignment", lambda *a: {})
    monkeypatch.setattr(pilot, "collect_episode", lambda s: deepcopy(collected[0]))
    monkeypatch.setattr(pilot, "evaluate_scenarios", lambda *a, **k: {
        "infrastructure_errors": [], "episodes": [{}], "family_summary": {},
        "evaluation": {"suite_metrics": {"episodes": [{"scenario_id": scenario["execution"]["scenario_id"],
            "trace": [{"step": row["step"], "action": row["preferred_action"],
                       "policy_observation": row["policy_observation"]} for row in collected[0]]}]}}})
    def replay(s, rows, exported):
        if failure == "persisted_trace_tamper":
            with (output/"raw_targets.jsonl").open("a") as f:
                f.write('{}\n')
        return {"passed": True, "exported_targets_replayed": len(exported)}
    monkeypatch.setattr(pilot, "replay_episode", replay)
    versions = iter([{"source": "a"}]*4+[{"source": "b"}])
    monkeypatch.setattr(pilot, "source_hashes", (lambda: next(versions)) if failure == "late_source_drift" else lambda: {"source": "a"})
    with pytest.raises(ValueError, match="Source changed|Persisted artifact"):
        pilot.build_pilot(output)
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["complete"] is False and manifest["mechanics_release_passed"] is False
    assert not (output/"train.canonical.jsonl").exists()


def test_evaluation_projection_preserves_exact_root_and_parent(scenarios):
    from scripts.run_ieee57_training_pilot import evaluation_view
    from psse_env.dagger.evaluator import validate_release_scenario_suites
    original = deepcopy(scenarios[0])
    projected = evaluation_view(original)
    validate_release_scenario_suites({"standard_success": [projected]})
    assert projected["grouping"]["physical_root_fingerprint"] == original["grouping"]["physical_root_fingerprint"]
    assert projected["grouping"]["source_realization_id"] == original["grouping"]["source_realization_id"]
    assert "parent_plan_sha256" in original["grouping"] and "parent_plan_sha256" not in projected["grouping"]


def test_alias_before_budget_is_id_length_and_json_order_invariant():
    from psse_env.dagger.dataset_builder import prepare_model_policy_observation
    def observation(prefix):
        return {
            "active_state_id": prefix+":s2", "candidate_state_id": prefix+":s3",
            "accepted_corrections": [{"candidate_state_id": prefix+":s1"}],
            "history_window": [
                {"state_id": prefix+":s2", "action": {"tool": "run_wls", "arguments": {"state_id": prefix+":s3"}},
                 "tool_output": {"execution_status": "success", "tool_metrics": {
                     "wls_objective": 300.+i, "normalized_residual_threshold": 4.,
                     "supported_corrections": [{"tool": "correct_measurements", "arguments": {
                         "state_id": prefix+":s2", "suspect_group": [7]}}]}}}
                for i in range(4)]}
    short, long = observation("short_controller"), observation("long_replay_controller_"*12)
    view, bindings = prepare_model_policy_observation(short, max_history_chars=1100, alias_before_compaction=True)
    reloaded = json.loads(json.dumps(long, sort_keys=True))
    replay, fresh = prepare_model_policy_observation(reloaded, max_history_chars=1100, alias_before_compaction=True)
    assert view == replay
    assert view["candidate_state_id"] == "candidate"
    assert bindings["state_aliases"]["candidate"] == short["candidate_state_id"]
    assert fresh["state_aliases"]["candidate"] == long["candidate_state_id"]
    legacy, _ = prepare_model_policy_observation(short)
    explicit_legacy, _ = prepare_model_policy_observation(short, alias_before_compaction=False)
    assert legacy == explicit_legacy
