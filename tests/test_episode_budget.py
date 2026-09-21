from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from psse_env.episode_budget import (
    DEFAULT_EPISODE_ACTION_LIMIT, bind_env_action_limit, validate_episode_action_limit,
)
from psse_env.transactional_env import TransactionalPSSEEnv


def _scenario(*, quiet=False):
    return {"scenario_id": "budget-test", "case": "case14", "measurements": [1.0],
            "no_material_anomaly_remaining": quiet}


def test_default_forty_and_explicit_runner_limit_reach_policy_before_reset():
    env = TransactionalPSSEEnv()
    assert DEFAULT_EPISODE_ACTION_LIMIT == env.max_steps == 40
    env.reset(_scenario())
    assert env.get_policy_observation().remaining_budget == 40
    assert bind_env_action_limit(env, 3) == 3
    env.reset(_scenario())
    assert env.get_policy_observation().remaining_budget == 3
    env.step({"tool": "unknown_tool", "arguments": {}})
    assert env.get_policy_observation().remaining_budget == 2
    # A new episode uses the runner's newly bound horizon, not stale history.
    bind_env_action_limit(env, 2)
    env.reset(_scenario())
    assert env.get_policy_observation().remaining_budget == 2
    assert not env.history


def test_failures_spend_budget_and_extra_attempts_neither_dispatch_nor_append(monkeypatch):
    env = TransactionalPSSEEnv(max_steps=3)
    env.reset(_scenario())
    for expected in (2, 1, 0):
        state, output = env.step({"tool": "unknown_tool", "arguments": {}})
        assert output["execution_status"] == "failure"
        assert state["remaining_budget"] == expected
    recorded = deepcopy(env.history)
    store_hash = env.store.episode_hash()

    def forbidden_dispatch(_action):
        pytest.fail("an exhausted episode must not dispatch another action")

    monkeypatch.setattr(env, "dispatch_valid_action", forbidden_dispatch)
    for tool in ("run_wls", "finalize_diagnosis", "correct_measurements"):
        state, output = env.step({"tool": tool, "arguments": {}})
        assert output["error_code"] == "episode_action_limit_reached"
        assert output["execution_status"] == "failure"
        assert output["tool_metrics"]["executed_action_count"] == 3
        assert state["remaining_budget"] == 0
        assert not output["state_mutated"]
    assert env.history == recorded
    assert env.store.episode_hash() == store_hash
    assert not env.is_terminal()
    assert env.terminal_outcome is None


def test_successful_provider_call_forty_executes_but_forty_one_does_not():
    calls = []

    def runner(state):
        calls.append(state["state_id"])
        return {"remaining_anomaly_score": 2.0, "wls_objective": 2.0,
                "no_material_anomaly_remaining": False}

    env = TransactionalPSSEEnv(wls_runner=runner)
    env.reset(_scenario())
    for action_count in range(1, 41):
        state, output = env.step({"tool": "run_wls", "arguments": {}})
        assert output["execution_status"] == "success"
        assert len(env.history) == action_count
        assert state["remaining_budget"] == 40 - action_count
    _, rejected = env.step({"tool": "run_wls", "arguments": {}})
    assert rejected["error_code"] == "episode_action_limit_reached"
    assert len(calls) == len(env.history) == 40
    assert not env.is_terminal()
    assert env.terminal_outcome is None


def test_finalization_is_valid_as_the_fortieth_action():
    env = TransactionalPSSEEnv()
    env.reset(_scenario(quiet=True))
    for _ in range(39):
        env.step({"tool": "unknown_tool", "arguments": {}})
    state, output = env.step({"tool": "finalize_diagnosis", "arguments": {}})
    assert output["execution_status"] == "success"
    assert len(env.history) == 40
    assert state["remaining_budget"] == 0
    assert env.is_terminal()
    assert env.terminal_outcome == "resolved"
    _, rejected = env.step({"tool": "run_wls", "arguments": {}})
    assert rejected["error_code"] == "episode_action_limit_reached"
    assert len(env.history) == 40
    assert env.terminal_outcome == "resolved"


def test_exhaustion_does_not_grant_a_free_finalization():
    env = TransactionalPSSEEnv(max_steps=1)
    env.reset(_scenario(quiet=True))
    env.step({"tool": "unknown_tool", "arguments": {}})
    _, rejected = env.step({"tool": "finalize_diagnosis", "arguments": {}})
    assert rejected["error_code"] == "episode_action_limit_reached"
    assert not env.is_terminal()
    assert env.terminal_outcome is None
    assert len(env.history) == 1


def test_synthetic_setup_attempt_counts_without_fabricating_tool_history():
    env = TransactionalPSSEEnv()
    env.reset(_scenario())
    physical_hash = env.store.episode_hash()
    flags = deepcopy(env.context_flags)
    env.account_setup_actions(1)
    assert env.get_policy_observation().remaining_budget == 39
    assert env.history == []
    assert env.store.episode_hash() == physical_hash
    assert env.context_flags == flags
    assert not env.is_terminal()
    for _ in range(39):
        env.step({"tool": "unknown_tool", "arguments": {}})
    assert len(env.history) == 39
    assert env.current_state()["remaining_budget"] == 0
    _, output = env.step({"tool": "run_wls", "arguments": {}})
    assert output["error_code"] == "episode_action_limit_reached"
    assert output["tool_metrics"]["executed_action_count"] == 39
    assert output["tool_metrics"]["non_dispatched_action_count"] == 1
    assert output["tool_metrics"]["counted_action_count"] == 40
    env.reset(_scenario())
    assert env.current_state()["remaining_budget"] == 40


def test_setup_accounting_rejects_overspend_without_partial_mutation():
    env = TransactionalPSSEEnv(max_steps=2)
    env.reset(_scenario())
    env.step({"tool": "unknown_tool", "arguments": {}})
    with pytest.raises(ValueError, match="exceed"):
        env.account_setup_actions(2)
    assert env.current_state()["remaining_budget"] == 1
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            env.account_setup_actions(count)
    assert env.current_state()["remaining_budget"] == 1


def test_counterfactual_clone_keeps_setup_cost_without_sharing_bookkeeping():
    env = TransactionalPSSEEnv(max_steps=5)
    env.reset(_scenario())
    env.account_setup_actions(1)
    branch = env.clone()
    assert branch.current_state()["remaining_budget"] == 4
    branch.account_setup_actions(1)
    assert branch.current_state()["remaining_budget"] == 3
    assert env.current_state()["remaining_budget"] == 4


@pytest.mark.parametrize("tool", [
    "run_wls", "get_three_phase_context", "get_harmonic_context", "get_measurement_context",
    "run_three_phase_nlm_from_path", "estimate_hif_location_magnitude_multiscan_from_path",
    "correct_measurements", "verify_candidate", "commit_state", "rollback_state", "finalize_diagnosis",
])
def test_each_attempted_environment_tool_uses_one_action_including_rejections(tool):
    env = TransactionalPSSEEnv(max_steps=2)
    env.reset(_scenario())
    state, output = env.step({"tool": tool, "arguments": {}})
    assert output["execution_status"] in {"success", "failure"}
    assert len(env.history) == 1
    assert state["remaining_budget"] == 1


def test_binding_supports_minimal_fake_environments_without_inventing_budget_state():
    env = SimpleNamespace()
    assert bind_env_action_limit(env, 4) == 4
    assert not hasattr(env, "max_steps")

    class ReadOnly:
        @property
        def max_steps(self):
            return 4

    assert bind_env_action_limit(ReadOnly(), 4) == 4
    with pytest.raises(ValueError, match="not writable"):
        bind_env_action_limit(ReadOnly(), 5)


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, 4.0, "4", None])
def test_invalid_horizons_are_rejected(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        validate_episode_action_limit(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        TransactionalPSSEEnv(max_steps=invalid)
    with pytest.raises(ValueError, match="positive integer"):
        bind_env_action_limit(SimpleNamespace(), invalid)
