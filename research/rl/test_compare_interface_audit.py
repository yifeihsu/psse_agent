from copy import deepcopy

import pytest

from research.rl.compare_interface_audit import COMPARE, audit, trace_digest


def report(successes):
    episodes = []
    for root, success in successes.items():
        episodes.append({
            "physical_root": root, "scenario_id": root, "episode_key": root, "family": "measurement",
            "seed": 1, "truth_audited_task_success": success, "invalid_action_count": 0,
            "loop_detected": False, "policy_steps": 2, "steps": 2, "tool_counts": {"run_wls": 1},
            "false_commit_count": 0, "false_finalization_count": 0, "evaluator_error": None, "trace": [],
        })
    return {"suite_metrics": {"configuration": {"seed": 1, "max_steps": 40,
                                               "action_budget_scope": "all_episode_actions"},
                              "episodes": episodes}}


def test_net_gain_does_not_hide_lost_success_or_expert_failure():
    before = report({"a": True, "b": False, "c": False, "d": False})
    after = report({"a": False, "b": True, "c": True, "d": False})
    teacher = report({"a": True, "b": True, "c": False, "d": False})
    result = audit(before, after, teacher, expected_roots=4)
    assert result["gained_roots"] == ["b", "c"]
    assert result["regressed_roots"] == ["a"]
    assert result["expert_only_success_roots"] == ["a"]
    assert result["candidate_only_success_roots"] == ["c"]
    assert result["both_failed_roots"] == ["d"]


@pytest.mark.parametrize("change", ["root", "duplicate", "seed", "budget", "missing_success"])
def test_refuses_unpaired_or_incomplete_results(change):
    before = report({"a": True, "b": False})
    after = deepcopy(before)
    rows = after["suite_metrics"]["episodes"]
    if change == "root":
        rows[0]["physical_root"] = "elsewhere"
    elif change == "duplicate":
        rows[0]["physical_root"] = "b"
    elif change == "seed":
        rows[0]["seed"] = 2
    elif change == "budget":
        after["suite_metrics"]["configuration"]["max_steps"] = 80
    else:
        rows[0].pop("truth_audited_task_success")
        rows[0]["terminal_outcome"] = "resolved"
    with pytest.raises(ValueError):
        audit(before, after, before, expected_roots=2)


def test_counts_internal_work_and_missing_telemetry_separately():
    payload = report({"a": True})
    episode = payload["suite_metrics"]["episodes"][0]
    episode["tool_counts"][COMPARE] = 2
    episode["trace"] = [
        {"action": {"tool": COMPARE}, "policy_tool_output": {"tool_metrics": {
            "internal_wls_calls": 2, "internal_correction_calls": 3}}},
        {"action": {"tool": COMPARE}, "policy_tool_output": {"error_code": "missing_precondition"}},
    ]
    result = audit(payload, payload, payload, expected_roots=1)["summaries"]["candidate"]
    assert result["comparison_calls"] == 2
    assert result["comparison_internal_wls_calls"] == 2
    assert result["comparison_internal_correction_calls"] == 3
    assert result["comparison_calls_without_internal_counts"] == 1


def test_repetition_inspection_resets_on_recorded_progress():
    step = {"action": {"tool": "get_harmonic_context", "arguments": {"state_id": "s0"}},
            "state_before": {"active_state_id": "s0"}, "advanced": False,
            "policy_observation": {"fresh_context_evidence": {"harmonic": {"request_attempted": True}}}}
    progress = {"action": {"tool": "commit_state"}, "advanced": True}
    rows = trace_digest({"trace": [step, step, progress, step]})
    assert [row["repeated_since_last_advance"] for row in rows] == [False, True, False, False]
    assert rows[0]["contexts_before"]["harmonic"]["request_attempted"] is True
