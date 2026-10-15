from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from psse_env.actions import FINALIZE_DIAGNOSIS, RUN_WLS
from psse_env.rl.reward import RewardWeights
from psse_env.rl.rollout import (
    RecordingPolicy,
    Trajectory,
    TrajectoryStore,
    distinct_trajectories,
    pair_records_with_trace,
    rollout_group,
    rollout_roots,
    sample_seed,
    scenario_family,
    scenario_root,
    store_summary,
)


class _DeploymentOracle:
    mode = "deployment"


class _PhysicalStore:
    def __init__(self, env: "_ScriptEnv") -> None:
        self.env = env
        self.active_state_id = "active"

    def get_state(self, state_id: str) -> dict[str, Any]:
        if state_id != self.active_state_id:
            raise KeyError(state_id)
        return copy.deepcopy(self.env.physical_state)


class _ScriptEnv:
    """A healthy root: the policy runs a WLS, then finalizes; nothing is edited."""

    production_dataset_mode = True
    candidate_quality_oracle = _DeploymentOracle()

    def __init__(self, *, seed: int | None = None, rng: Any = None) -> None:
        self.seed = seed
        self.rng = rng
        self.cursor = 0
        self.terminal = False
        self.terminal_outcome: str | None = None
        self.scenario: dict[str, Any] = {}
        self.physical_state: dict[str, Any] = {}
        self.store = _PhysicalStore(self)

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        self.scenario = copy.deepcopy(dict(scenario))
        self.cursor = 0
        self.terminal = False
        self.terminal_outcome = None
        self.physical_state = copy.deepcopy(dict(self.scenario.get("initial_physical_state") or {}))
        return self.current_state()

    def get_policy_observation(self, history):
        return {"active_state_id": "active", "candidate_state_id": None, "remaining_budget": 6 - self.cursor,
                "phase": "wls" if self.cursor == 0 else "finalize", "history_window": copy.deepcopy(history[-2:])}

    def get_oracle_state(self, history):
        return {"candidate_disposition": None, "candidate_assessment": {},
                "hidden_truth": {"truth_complete": True, "remaining_true_fault_count": 0}}

    def step(self, action):
        tool = action.get("tool")
        self.cursor += 1
        metrics: dict[str, Any] = {}
        if tool == RUN_WLS:
            metrics = {"state_id": "active", "state_hash": hashlib.sha256(f"active:{self.cursor}".encode()).hexdigest()}
        if tool == FINALIZE_DIAGNOSIS:
            self.terminal = True
            self.terminal_outcome = "resolved"
        return self.current_state(), {"execution_status": "success", "error_code": None,
                                      "state_mutated": False, "tool_metrics": metrics}

    def current_state(self):
        return {"active_state_id": "active", "candidate_state_id": None, "accepted_corrections": [], "explained_anomalies": []}

    def is_terminal(self, state=None):
        return self.terminal


class _SamplingPolicy:
    """A scripted policy that pretends to sample: it records a generation per call."""

    def __init__(self, seed: int, *, extra_wls: bool = False) -> None:
        self.seed = seed
        self.extra_wls = extra_wls
        self.calls = 0
        self._last_generation = None
        self.research_policy_identity = {"adapter_path": "fake"}

    @property
    def last_generation(self):
        return copy.deepcopy(self._last_generation)

    def act(self, observation):
        phase = observation["phase"]
        if phase == "wls" or (self.extra_wls and self.calls == 1):
            action = {"tool": RUN_WLS, "arguments": {"state_id": "active"}}
        else:
            action = {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
        self._last_generation = {"model_observation": dict(observation), "prompt_tokens": 10 + self.calls,
                                 "original_prompt_tokens": 10 + self.calls, "completion_token_ids": [self.seed % 7, self.calls],
                                 "completion_token_probabilities": [1.0, 0.5 if self.extra_wls else 1.0],
                                 "text": action["tool"], "generation": {"seed": self.seed}, "call_index": self.calls}
        self.calls += 1
        return action


def _scenario(root: str = "r0_fake") -> dict[str, Any]:
    clean_case = {"baseMVA": 100.0, "branch": []}
    clean_measurements = [0.0] * 8
    return {
        "scenario_id": root, "root_scenario_id": root, "physical_root_fingerprint": f"fp-{root}",
        "scenario_family": "no_error", "error_cardinality": 0, "case_id": "case14",
        "case": copy.deepcopy(clean_case), "clean_case": copy.deepcopy(clean_case),
        "measurements": list(clean_measurements), "clean_measurements": list(clean_measurements),
        "initial_physical_state": {"case": copy.deepcopy(clean_case), "measurements": list(clean_measurements)},
        "split": "development", "source_tier": "test", "true_measurement_errors": [], "hidden_truth": {},
        "final_remaining": 0,
    }


class RolloutTests(unittest.TestCase):
    def test_recording_policy_keeps_generation_and_action_per_step(self) -> None:
        policy = RecordingPolicy(_SamplingPolicy(3))
        policy.act({"phase": "wls"})
        policy.act({"phase": "finalize"})
        self.assertEqual([record["action"]["tool"] for record in policy.records], [RUN_WLS, FINALIZE_DIAGNOSIS])
        self.assertEqual(policy.records[0]["completion_token_ids"], [3, 0])
        self.assertEqual(policy.research_policy_identity, {"adapter_path": "fake"})

    def test_recording_keeps_the_sampled_tokens_of_an_invalid_call(self) -> None:
        class _Invalid(_SamplingPolicy):
            def act(self, observation):
                super().act(observation)
                if self.calls == 2:
                    raise ValueError("unparseable tool call")
                return {"tool": RUN_WLS, "arguments": {"state_id": "active"}}

        policy = RecordingPolicy(_Invalid(5))
        policy.act({"phase": "wls"})
        with self.assertRaises(ValueError):
            policy.act({"phase": "finalize"})
        self.assertEqual(len(policy.records), 2)
        self.assertIsNone(policy.records[1]["action"])
        self.assertEqual(policy.records[1]["policy_exception"], "ValueError")
        self.assertEqual(policy.records[1]["completion_token_ids"], [5, 1])

    def test_a_failure_before_sampling_never_reuses_the_previous_generation(self) -> None:
        class _FailsEarly(_SamplingPolicy):
            def act(self, observation):
                if observation.get("phase") == "broken":
                    raise RuntimeError("observation rejected before generation")
                return super().act(observation)

        policy = RecordingPolicy(_FailsEarly(2))
        policy.act({"phase": "wls"})
        with self.assertRaises(RuntimeError):
            policy.act({"phase": "broken"})
        self.assertNotIn("completion_token_ids", policy.records[1])

    def test_pairing_requires_one_record_per_policy_step(self) -> None:
        trace = [{"step": 0, "intervention": True, "action": {"tool": RUN_WLS}},
                 {"step": 1, "action": {"tool": RUN_WLS, "arguments": {}}, "execution_status": "success", "error_code": None}]
        records = [{"action": {"tool": RUN_WLS}, "model_observation": {"a": 1}, "completion_token_ids": [1], "prompt_tokens": 1,
                    "original_prompt_tokens": 1, "text": "x"}]
        steps, ok, note = pair_records_with_trace(records, trace)
        self.assertTrue(ok)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].step, 1)
        _steps, ok, note = pair_records_with_trace(records * 2, trace)
        self.assertFalse(ok)
        self.assertIn("2 generation records", note)
        _steps, ok, note = pair_records_with_trace([{"action": {"tool": RUN_WLS}}], trace)
        self.assertFalse(ok)
        self.assertIn("without a recorded generation", note)

    def test_rollout_group_runs_k_seeded_episodes_through_the_evaluator(self) -> None:
        scenario = _scenario()
        created: list[_SamplingPolicy] = []

        def policy_factory(seed: int) -> _SamplingPolicy:
            policy = _SamplingPolicy(seed, extra_wls=(len(created) % 2 == 1))
            created.append(policy)
            return policy

        group = rollout_group(scenario, samples=2, base_seed=7, env_factory=lambda **kwargs: _ScriptEnv(**kwargs),
                              policy_factory=policy_factory, max_steps=6, case_loader=None,
                              weights=RewardWeights(call_cost=0.1))
        self.assertEqual(len(group), 2)
        self.assertEqual([trajectory.sample_index for trajectory in group], [0, 1])
        self.assertEqual([trajectory.episode_seed for trajectory in group], [sample_seed(7, "r0_fake", 0), sample_seed(7, "r0_fake", 1)])
        self.assertEqual([len(trajectory.steps) for trajectory in group], [2, 3])
        self.assertTrue(all(trajectory.pairing_ok for trajectory in group), [t.pairing_note for t in group])
        self.assertTrue(all(trajectory.evaluation["truth_audited_task_success"] for trajectory in group),
                        [t.evaluation.get("audit", {}).get("truth_audited_task_assessment") for t in group])
        self.assertGreater(group[0].total_reward, group[1].total_reward)
        self.assertEqual(group[0].steps[0].tool, RUN_WLS)
        self.assertEqual(group[0].steps[0].completion_token_ids, [created[0].seed % 7, 0])
        self.assertEqual(group[0].policy_identity, {"adapter_path": "fake"})
        self.assertEqual(group[0].steps[0].token_probabilities, [1.0, 1.0])
        self.assertEqual(group[1].steps[0].min_token_probability, 0.5)
        self.assertEqual(distinct_trajectories(group), 2)
        self.assertEqual(distinct_trajectories([group[0], group[0]]), 1)
        self.assertEqual(scenario_root(scenario), "r0_fake")
        self.assertEqual(scenario_family(scenario), "no_error")

    def test_store_round_trips_groups_and_resumes_complete_ones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(directory)
            scenarios = [_scenario("r0_a"), _scenario("r0_b")]
            summary = rollout_roots(scenarios, store=store, samples=2, base_seed=1,
                                    env_factory=lambda **kwargs: _ScriptEnv(**kwargs),
                                    policy_factory=lambda seed: _SamplingPolicy(seed), max_steps=6,
                                    case_loader=None, weights=RewardWeights(), expert_reference={"r0_a": 2})
            self.assertEqual(summary["completed_groups"], 2)
            self.assertEqual(sorted(store.roots()), ["r0_a", "r0_b"])
            again = rollout_roots(scenarios, store=store, samples=2, base_seed=1,
                                  env_factory=lambda **kwargs: _ScriptEnv(**kwargs),
                                  policy_factory=lambda seed: _SamplingPolicy(seed), max_steps=6,
                                  case_loader=None, weights=RewardWeights())
            self.assertEqual(again["completed_groups"], 0)
            self.assertEqual(again["skipped_complete_groups"], 2)
            groups = dict(store.iter_groups())
            self.assertEqual(len(groups["r0_a"]), 2)
            self.assertEqual(groups["r0_a"][0].reward["tool_regret"], 0)
            self.assertIsNone(groups["r0_b"][0].reward["tool_regret"])
            restored = Trajectory.from_dict(groups["r0_a"][1].as_dict())
            self.assertEqual(restored.steps[0].completion_token_ids, groups["r0_a"][1].steps[0].completion_token_ids)
            self.assertTrue((Path(directory) / "groups" / "r0_a.jsonl").is_file())
            summary = store_summary(store)
            self.assertEqual(summary["groups"], 2)
            self.assertEqual(summary["episodes"], 4)
            self.assertEqual(summary["success_rate"], 1.0)
            self.assertEqual(summary["sampled_steps_with_probabilities"], 8)
            self.assertEqual(summary["fraction_of_steps_with_every_token_above_0_99"], 1.0)


if __name__ == "__main__":
    unittest.main()
