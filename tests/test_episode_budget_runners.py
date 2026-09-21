"""Collector/evaluator horizons and visible budgets must describe one limit."""
import copy

from psse_env.dagger.evaluator import evaluate_rollout_suites
from psse_env.dagger.rollout_collector import DaggerRolloutCollector
from psse_env.state_store import PolicyObservation


class AdvancingEnv:
    def __init__(self):
        self.max_steps = 12  # Deliberately stale factory setting.
        self.calls = 0

    def reset(self, scenario):
        self.calls = 0
        self.limit_at_reset = self.max_steps
        return self.current_state()

    def current_state(self):
        return {"active_state_id": f"s{self.calls}", "remaining_budget": self.max_steps - self.calls,
                "no_material_anomaly_remaining": False, "case": "case14", "measurements": []}

    def get_policy_observation(self, history):
        state = self.current_state()
        return PolicyObservation(active_state_id=state["active_state_id"],
                                 remaining_budget=state["remaining_budget"], history_window=copy.deepcopy(history[-2:]))

    def step(self, action):
        assert self.calls < self.max_steps
        self.calls += 1
        return self.current_state(), {"execution_status": "success", "state_mutated": True,
                                      "active_state_id": f"s{self.calls}", "tool_metrics": {}}

    def is_terminal(self, state=None):
        return False


class WlsPolicy:
    def __init__(self):
        self.budgets = []

    def act(self, observation):
        self.budgets.append(observation["remaining_budget"])
        return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}


class WlsExpert:
    def label_transition(self, **kwargs):
        return {"process_valid": True, "action_admissible": True}

    def next_actions(self, observation, history):
        state_id = observation.get("active_state_id")
        return [{"tool": "run_wls", "arguments": {"state_id": state_id}}]


def test_evaluation_executes_at_most_forty_and_reports_matching_remaining_budget():
    env, policy = AdvancingEnv(), WlsPolicy()
    result = evaluate_rollout_suites({"standard_success": [{"scenario_id": "budget", "case": "case14", "measurements": []}]},
                                    env_factory=lambda: env, policy_factory=lambda: policy)
    episode = result.suite_metrics["episodes"][0]
    assert env.limit_at_reset == env.calls == episode["steps"] == episode["policy_steps"] == 40
    assert policy.budgets == list(range(40, 0, -1))
    assert not episode["terminal"]
    assert not episode["final_physical_success"]


def test_collection_binds_the_limit_and_keeps_the_same_visible_budget():
    env, policy = AdvancingEnv(), WlsPolicy()
    collector = DaggerRolloutCollector(env=env, policy=policy, expert_oracle=WlsExpert())
    rows = collector.collect_iteration(scenarios=[{"scenario_id": "budget", "case": "case14", "measurements": []}],
                                       iteration=0, beta=0., max_steps=40)
    assert env.limit_at_reset == env.calls == len(rows) == 40
    assert policy.budgets == list(range(40, 0, -1))


def test_deliberate_short_evaluation_override_also_binds_the_environment():
    env, policy = AdvancingEnv(), WlsPolicy()
    result = evaluate_rollout_suites({"standard_success": [{"scenario_id": "short", "case": "case14", "measurements": []}]},
                                    env_factory=lambda: env, policy_factory=lambda: policy, max_steps=3)
    assert env.limit_at_reset == env.calls == 3
    assert policy.budgets == [3, 2, 1]
    assert result.suite_metrics["episodes"][0]["steps"] == 3
