from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from psse_env.dagger.ieee57_runtime import (
    IEEE57_RUNTIME_CONFIG, ieee57_environment_factory, ieee57_runtime_manifest,
    validate_ieee57_runtime, validate_ieee57_wls_metrics,
    ieee57_expert_oracle_factory,
)
from psse_env.dagger.release_factories import (
    EXPERT_POLICY_IDENTITY, observable_expert_policy_factory,
    production_environment_factory,
    select_observable_expert_actions,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.oracle.measurement_recovery_evidence import measurement_targets_predating_branch_repair


class IEEE57RuntimeTests(unittest.TestCase):
    def test_contract_is_immutable_and_manifest_is_detached(self):
        with self.assertRaises(FrozenInstanceError):
            IEEE57_RUNTIME_CONFIG.chi2_alpha = 0.001
        manifest = ieee57_runtime_manifest()
        manifest["normalized_residual_threshold"] = None
        self.assertEqual(ieee57_runtime_manifest()["normalized_residual_threshold"], 4.0)

    def test_pinned_factory_and_legacy_factory_are_distinct(self):
        env = ieee57_environment_factory(seed=5)
        self.assertEqual(validate_ieee57_runtime(env), ieee57_runtime_manifest())
        legacy = production_environment_factory()
        self.assertIsNone(legacy.wls_runner.__self__.normalized_residual_threshold)
        with self.assertRaisesRegex(ValueError, "detector mismatch"):
            validate_ieee57_runtime(legacy)
        env.wls_runner.__self__.normalized_residual_threshold = None
        with self.assertRaisesRegex(ValueError, "detector mismatch"):
            validate_ieee57_runtime(env)

    def test_explicit_action_horizon_does_not_relax_detector_contract(self):
        env = ieee57_environment_factory()
        env.max_steps = 8
        with self.assertRaisesRegex(ValueError, "max_steps"):
            validate_ieee57_runtime(env)
        self.assertEqual(validate_ieee57_runtime(env, expected_max_steps=8)["max_steps"], 8)
        env.wls_runner.__self__.normalized_residual_threshold = 3.0
        with self.assertRaisesRegex(ValueError, "detector mismatch"):
            validate_ieee57_runtime(env, expected_max_steps=8)

    def test_inclusive_alarm_boundaries_and_missing_or_nonfinite_evidence(self):
        metrics = {
            "chi_square_alpha": .05, "normalized_residual_threshold": 4.0,
            "anomaly_detection_rule": "chi_square_or_normalized_residual",
            "chi_square_statistic": 20.0, "chi_square_threshold": 100.0,
            "max_normalized_residual": 4.0, "chi_square_alarm": False,
            "normalized_residual_alarm": True, "no_material_anomaly_remaining": False,
        }
        validate_ieee57_wls_metrics(metrics)
        metrics.update(chi_square_statistic=100.0, chi_square_alarm=True,
                       max_normalized_residual=3.0, normalized_residual_alarm=False)
        validate_ieee57_wls_metrics(metrics)
        metrics["chi_square_alarm"] = False
        with self.assertRaisesRegex(ValueError, "OR rule"):
            validate_ieee57_wls_metrics(metrics)
        metrics["chi_square_statistic"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_ieee57_wls_metrics(metrics)

    def test_exact_supported_target_observation_uses_production_teacher_contract(self):
        path = Path(__file__).resolve().parents[2] / "research/ieee57/evidence/ieee57_hydrated_correction_regression.json"
        regression = json.loads(path.read_text(encoding="utf-8"))
        observation = regression["policy_observation"]
        # Reproduce the integration failure: a nondeployment process contract
        # filters every target-only correction as an empty payload.
        wrong = select_observable_expert_actions(policy_observation=observation, expert_oracle=ExpertPolicyOracle())
        self.assertEqual(wrong.preferred_action, regression["historical_action"])
        expert = ieee57_expert_oracle_factory()
        selected = select_observable_expert_actions(policy_observation=observation, expert_oracle=expert)
        expected = observation["last_tool_output"]["tool_metrics"]["supported_corrections"][0]
        self.assertEqual(selected.preferred_action, expected)
        self.assertTrue(expert.process_oracle.check(observation, expected)["process_valid"])
        self.assertEqual(ieee57_runtime_manifest()["executor_hydrated_corrections"], True)

    def test_collector_and_replay_construct_the_pinned_teacher(self):
        from psse_env.dagger.ieee57_training import collect_episode, replay_episode
        path = Path(__file__).resolve().parents[2] / "research/ieee57/evidence/training_readiness_scenarios_20260912.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scenario = json.load(handle)[0]
        scenario["grouping"].update(
            parent_construction_fingerprint="historical_fixture_for_factory_test",
            parent_plan_sha256="unit_factory_contract_only",
            dataset_split="train",
        )
        with patch("psse_env.dagger.ieee57_runtime.ieee57_expert_oracle_factory", wraps=ieee57_expert_oracle_factory) as factory:
            rows = collect_episode(scenario, max_steps=1)
            replay = replay_episode(scenario, rows, [])
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(rows[0]["tool_output"]["execution_status"], "success")
        self.assertTrue(replay["passed"])


class MixedNoChangeRegressionTests(unittest.TestCase):
    def test_exact_historical_visible_state_does_not_repeat_repaired_meter(self):
        path = Path(__file__).resolve().parents[2] / "research/ieee57/training_readiness_failure_audit_20260912.json"
        failure = json.loads(path.read_text(encoding="utf-8"))["invalid_actions"][0]
        observation = failure["historical_policy_observation"]
        original = copy.deepcopy(observation)
        policy = observable_expert_policy_factory(policy_identity=EXPERT_POLICY_IDENTITY)
        action = policy.act(observation)
        self.assertEqual(action, {
            "tool": "ask_for_more_evidence", "arguments": {
                "state_id": observation["active_state_id"],
                "request": "operator_escalation:recovery_options_exhausted",
            },
        })
        self.assertEqual(observation, original)
        self.assertNotEqual(action, failure["historical_action"])
        # The anomaly remains available while the no-op action is excluded.
        self.assertGreater(observation["remaining_anomaly_score"], 1.0)
        self.assertFalse(observation["no_material_anomaly_remaining"])

    def test_exact_mixed_prefix_real_protocol_retains_anomaly_after_commit(self):
        root = Path(__file__).resolve().parents[2]
        audit = json.loads((root / "research/ieee57/training_readiness_failure_audit_20260912.json").read_text(encoding="utf-8"))
        failure = audit["invalid_actions"][0]
        evidence = root / "research/ieee57/evidence"
        with gzip.open(evidence / "training_readiness_scenarios_20260912.json.gz", "rt", encoding="utf-8") as handle:
            scenarios = json.load(handle)
        with gzip.open(evidence / "training_readiness_20260912.json.gz", "rt", encoding="utf-8") as handle:
            historical = json.load(handle)
        scenario = next(row for row in scenarios if row["execution"]["scenario_id"] == failure["scenario_id"])
        episode = next(row for row in historical["evaluation"]["suite_metrics"]["episodes"] if row["scenario_id"] == failure["scenario_id"])
        env = ieee57_environment_factory()
        env.reset(copy.deepcopy(scenario["execution"]))  # No audit/truth is supplied.
        post_commit_observations = []
        for transition in episode["trace"]:
            if transition["step"] >= failure["step"]:
                break
            _, output = env.step(copy.deepcopy(transition["action"]))
            self.assertEqual(output["execution_status"], "success")
            validate_ieee57_runtime(env)
            if transition["action"]["tool"] == "run_wls":
                validate_ieee57_wls_metrics(output["tool_metrics"])
            if transition["action"]["tool"] == "commit_state":
                post_commit_observations.append(env.get_policy_observation().as_dict())
        self.assertEqual(len(post_commit_observations), 2)
        for observation in post_commit_observations:
            self.assertFalse(observation["no_material_anomaly_remaining"])
            self.assertGreater(observation["remaining_anomaly_score"], 1.0)
        observation = env.get_policy_observation().as_dict()
        self.assertFalse(observation["last_tool_output"]["tool_metrics"]["accepted_target_refinement"])
        self.assertEqual(len(observation["accepted_corrections"]), 2)
        policy = observable_expert_policy_factory(policy_identity=EXPERT_POLICY_IDENTITY)
        action = policy.act(observation)
        self.assertEqual(action["tool"], "ask_for_more_evidence")
        _, output = env.step(action)
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(len(env.store.get_state(env.store.active_state_id)["measurements"]), 491)

    def test_ordering_generalizes_to_ids_and_repeated_reestimation(self):
        meter = {"source_action": {"tool": "correct_measurements", "arguments": {"suspect_group": [10]}}}
        branch = {"source_action": {"tool": "correct_parameters", "arguments": {"line_index": 70}}}
        for records, expected in [([branch, meter], set()), ([meter, branch], {10}),
                                  ([meter, branch, meter], set()), ([meter, branch, meter, branch], {10})]:
            with self.subTest(records=records):
                self.assertEqual(measurement_targets_predating_branch_repair({"accepted_corrections": records}), expected)


if __name__ == "__main__":
    unittest.main()
