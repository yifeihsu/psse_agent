from __future__ import annotations

from copy import deepcopy

from psse_env.dagger.evaluator import evaluate_rollout_suites
from psse_env.dagger.rollout_collector import DaggerRolloutCollector
from psse_env.state_store import PolicyObservation
from psse_env.transactional_env import TransactionalPSSEEnv


class StaleBudgetEnv:
    """External environment without a writable episode-budget API."""
    def reset(self, scenario):
        self.calls = 0
        return self.current_state()

    def current_state(self):
        return {"active_state_id": f"s{self.calls}", "remaining_budget": 999,
                "case": "case14", "measurements": []}

    def get_policy_observation(self, history):
        return PolicyObservation(active_state_id=f"s{self.calls}", remaining_budget=999,
                                 history_window=deepcopy(history[-2:]))

    def step(self, action):
        self.calls += 1
        return self.current_state(), {"execution_status": "success", "state_mutated": True,
                                      "active_state_id": f"s{self.calls}", "tool_metrics": {}}

    def is_terminal(self, state=None):
        return False


def test_collector_corrects_next_state_budget_and_transition_label_for_external_env():
    next_budgets = []
    policy_budgets = []

    class Policy:
        def act(self, observation):
            policy_budgets.append(observation["remaining_budget"])
            return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}

    class Expert:
        def next_actions(self, observation, history):
            return [{"tool": "run_wls", "arguments": {"state_id": observation.get("active_state_id")}}]

        def label_transition(self, **kwargs):
            next_budgets.append(kwargs["next_state"]["remaining_budget"])
            return {"process_valid": True, "action_admissible": True}

    env = StaleBudgetEnv()
    collector = DaggerRolloutCollector(env=env, policy=Policy(), expert_oracle=Expert())
    rows = collector.collect_iteration(
        scenarios=[{"scenario_id": "external-budget", "case": "case14", "measurements": []}],
        iteration=0, beta=0.0, max_steps=3,
    )
    assert policy_budgets == [3, 2, 1]
    assert next_budgets == [2, 1, 0]
    assert [row["next_state_summary"]["remaining_budget"] for row in rows] == [2, 1, 0]
    assert env.calls == 3
    assert not hasattr(env, "max_steps")


def test_evaluator_circuit_breaker_attempt_updates_real_environment_budget_without_fake_evidence():
    env = TransactionalPSSEEnv()
    observations = []

    class Policy:
        def act(self, observation):
            observations.append(observation["remaining_budget"])
            tool = "run_hse_from_path" if len(observations) == 1 else "get_harmonic_context"
            return {"tool": tool, "arguments": {"state_id": observation["active_state_id"]}}

    scenario = {
        "scenario_schema_version": 1,
        "execution": {"scenario_id": "circuit-budget", "case": "case14", "measurements": [1.0]},
        "audit": {"evaluation_intervention": {
            "intervention_schema_version": 1, "kind": "efficiency_budget",
            "limits": {"maximum_policy_steps": 2, "maximum_wls_calls": 2,
                       "maximum_specialized_tool_calls": 1}},
            "truth": {"truth_complete": True, "true_measurement_errors": [],
                      "true_parameter_errors": [], "true_topology_errors": []}},
        "grouping": {"scenario_family": "no_error", "case_id": "case14", "error_cardinality": 0,
                     "physical_root_fingerprint": "budget-only-fixture", "source_tier": "synthetic_pilot",
                     "split": "validation"},
    }
    result = evaluate_rollout_suites({"efficiency": [scenario]}, env_factory=lambda: env,
                                    policy_factory=Policy, max_steps=2)
    episode = result.suite_metrics["episodes"][0]
    assert observations == [2, 1]
    assert episode["steps"] == episode["policy_steps"] == 2
    assert episode["trace"][-1]["error_code"] == "evaluation_specialized_tool_budget_exhausted"
    assert env.current_state()["remaining_budget"] == 0
    assert len(env.history) == 1  # The evaluator rejected the second attempt itself.
    assert env.context_flags["last_tool"] == "run_hse_from_path"
    assert not env.is_terminal()
    assert env.terminal_outcome is None
    _, rejected = env.step({"tool": "run_wls", "arguments": {}})
    assert rejected["error_code"] == "episode_action_limit_reached"
    assert rejected["tool_metrics"]["non_dispatched_action_count"] == 1
