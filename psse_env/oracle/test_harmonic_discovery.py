from __future__ import annotations

import copy
import json
import unittest

from psse_env.dagger.dataset_builder import examples_to_chat_sft
from psse_env.dagger.rollout_collector import (
    BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
    DaggerRolloutCollector,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.transactional_env import TransactionalPSSEEnv
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE


class _BaselinePolicy:
    @staticmethod
    def act(observation: dict) -> dict:
        return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}


class HarmonicDiscoveryTests(unittest.TestCase):
    """Additional spectra are acquired after an observable WLS anomaly."""

    @classmethod
    def setUpClass(cls) -> None:
        generator = Round0ScenarioGenerator(seed=20260719, evidence_profile=AUXILIARY_EVIDENCE_PROFILE)
        cls.scenarios = {
            row["scenario_family"]: row
            for row in generator.build({"harmonic": 1, "measurement": 1, "no_error": 1})
        }

    def _environment(self, scenario: dict) -> tuple[TransactionalPSSEEnv, ExpertPolicyOracle]:
        env = TransactionalPSSEEnv(
            **MatpowerDeploymentProviders(evidence_profile=AUXILIARY_EVIDENCE_PROFILE).env_kwargs(),
            production_dataset_mode=True,
            max_steps=20,
        )
        env.reset(copy.deepcopy(scenario))
        return env, ExpertPolicyOracle(process_oracle=env.process_oracle)

    def _expert_step(self, env: TransactionalPSSEEnv, oracle: ExpertPolicyOracle) -> tuple[str, dict]:
        actions = oracle.next_actions(env.get_oracle_state(), env.history)
        self.assertTrue(actions, env.current_state())
        action = actions[0]
        env.assert_training_decision_evidence(action)
        _, output = env.step(action)
        self.assertEqual(output["execution_status"], "success", (action, output))
        return action["tool"], output

    def test_initial_observation_and_first_action_do_not_reveal_private_spectra(self) -> None:
        with_spectra = copy.deepcopy(self.scenarios["harmonic"])
        without_spectra = copy.deepcopy(with_spectra)
        without_spectra["metadata"].pop("harmonic_measurements", None)
        without_spectra["metadata"].pop("harmonic_orders", None)
        observations = []
        for scenario in (with_spectra, without_spectra):
            env, oracle = self._environment(scenario)
            observation = env.get_policy_observation().as_dict()
            self.assertEqual(observation["available_evidence"], [])
            self.assertEqual(observation["unresolved_signatures"], [])
            self.assertNotIn("harmonic_measurements", json.dumps(observation))
            self.assertNotIn("harmonic_distortion_detected", json.dumps(observation))
            self.assertEqual(oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"], "run_wls")
            observations.append(observation)
        self.assertEqual(observations[0], observations[1])

    def test_learner_cannot_skip_wls_or_acquisition_before_hse(self) -> None:
        for tool in ("get_harmonic_context", "run_hse_from_path"):
            with self.subTest(tool=tool):
                env, oracle = self._environment(self.scenarios["harmonic"])
                action = {"tool": tool, "arguments": {"state_id": env.store.active_state_id}}
                with self.assertRaises(ValueError):
                    env.assert_training_decision_evidence(action)
                _, output = env.step(action)
                self.assertEqual(output["execution_status"], "failure")
                self.assertEqual(output["error_code"], "missing_precondition")
                self.assertFalse(output["state_mutated"])
                self.assertEqual(oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"], "run_wls")
        env, oracle = self._environment(self.scenarios["harmonic"])
        self.assertEqual(self._expert_step(env, oracle)[0], "run_wls")
        _, output = env.step({"tool": "run_hse_from_path", "arguments": {"state_id": env.store.active_state_id}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_detail"], "hse_requires_acquired_harmonic_context")

    def test_same_wls_anomaly_requests_context_with_or_without_spectra(self) -> None:
        actions = []
        for available in (True, False):
            scenario = copy.deepcopy(self.scenarios["harmonic"])
            if not available:
                scenario["metadata"].pop("harmonic_measurements", None)
                scenario["metadata"].pop("harmonic_orders", None)
            env, oracle = self._environment(scenario)
            self.assertEqual(self._expert_step(env, oracle)[0], "run_wls")
            self.assertNotIn("harmonic_measurements", env.get_policy_observation().available_evidence)
            actions.append(oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"])
        # This frozen source has a narrow WLS residual pattern (9/122), so
        # the explicit auxiliary profile asks for phase telemetry first.
        # Private availability of spectra cannot alter that opening request.
        self.assertEqual(actions, ["get_three_phase_context", "get_three_phase_context"])

    def test_real_source_episode_and_export_observe_evidence_only_after_request(self) -> None:
        scenario = self.scenarios["harmonic"]
        env, oracle = self._environment(scenario)
        rows = DaggerRolloutCollector(
            env=env,
            policy=_BaselinePolicy(),
            expert_oracle=oracle,
            supervision_policy=BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
        ).collect_iteration(scenarios=[scenario], iteration=0, beta=1.0, max_steps=8)
        self.assertEqual(
            [row["preferred_action"]["tool"] for row in rows],
            ["run_wls", "get_three_phase_context", "get_harmonic_context", "run_hse_from_path", "finalize_diagnosis"],
        )
        self.assertTrue(env.is_terminal())
        self.assertEqual(rows[1]["tool_output"]["tool_metrics"]["three_phase_context_status"], "unavailable")
        hse = rows[3]["policy_observation"]
        self.assertIn("harmonic_measurements", hse["available_evidence"])
        self.assertTrue(hse["fresh_context_evidence"]["harmonic"]["harmonic_distortion_detected"])
        record = env.get_policy_observation().explained_anomalies[0]
        self.assertEqual(record["family"], "harmonic")
        self.assertEqual(
            record["detail"]["bus_1based"], scenario["hidden_truth"]["true_harmonic_errors"][0]["bus_1based"]
        )
        self.assertTrue(any(str(item).startswith("wls_") for item in record["explained_signatures"]))
        exported = examples_to_chat_sft(rows, protocol="canonical", allow_ineligible_auxiliary=True)
        self.assertEqual(len(exported), 5)
        self.assertEqual(
            [row["messages"][2]["tool_calls"][0]["function"]["name"] for row in exported],
            ["wls_from_path", "get_three_phase_context", "get_harmonic_context", "run_hse_from_path", "finalize_diagnosis"],
        )
        states = [json.loads(row["messages"][1]["content"])["state"] for row in exported]
        for state in states[:3]:
            self.assertNotIn("harmonic_measurements", state["available_evidence"])
            self.assertFalse(any("harmonic" in str(item) for item in state["unresolved_signatures"]))
        self.assertIn("harmonic_measurements", states[3]["available_evidence"])
        self.assertTrue(any("harmonic" in str(item) for item in states[3]["unresolved_signatures"]))
        for row in exported:
            self.assertNotIn("true_harmonic_errors", json.dumps(row["messages"]))
            self.assertNotIn("sensor_signatures_withheld", json.dumps(row["messages"]))

    def test_unavailable_context_keeps_anomaly_and_allows_classical_investigation(self) -> None:
        env, oracle = self._environment(self.scenarios["measurement"])
        self.assertEqual(self._expert_step(env, oracle)[0], "run_wls")
        prior = list(env.get_policy_observation().unresolved_signatures)
        self.assertTrue(prior)
        # A gross meter error is a narrow anomaly, so the phase-resolved
        # request comes first and returns nothing; spectra are requested next.
        tool, output = self._expert_step(env, oracle)
        self.assertEqual(tool, "get_three_phase_context")
        self.assertFalse(
            {"three_phase_voltages", "three_phase_branch_currents"}
            & set(env.get_policy_observation().available_evidence)
        )
        tool, output = self._expert_step(env, oracle)
        self.assertEqual(tool, "get_harmonic_context")
        metrics = output["tool_metrics"]
        self.assertEqual(metrics["harmonic_context_status"], "unavailable")
        self.assertEqual(metrics["available_evidence_channels"], [])
        self.assertFalse(metrics["harmonic_distortion_detected"])
        observation = env.get_policy_observation()
        self.assertEqual(observation.unresolved_signatures, prior)
        self.assertFalse(observation.explained_anomalies)
        self.assertFalse(observation.no_material_anomaly_remaining)
        self.assertFalse(env.is_terminal())
        executed = []
        for _ in range(12):
            tool, _ = self._expert_step(env, oracle)
            executed.append(tool)
            if tool.startswith("correct_") or env.is_terminal():
                break
        self.assertTrue(any(tool.startswith("correct_") for tool in executed), executed)
        self.assertNotIn("run_hse_from_path", executed)
        self.assertNotIn("get_harmonic_context", executed)
        self.assertFalse(env.get_policy_observation().explained_anomalies)

    def test_clean_control_never_requests_harmonics(self) -> None:
        env, oracle = self._environment(self.scenarios["no_error"])
        executed = [self._expert_step(env, oracle)[0], self._expert_step(env, oracle)[0]]
        self.assertEqual(executed, ["run_wls", "finalize_diagnosis"])
        self.assertTrue(env.is_terminal())
        self.assertFalse(env.get_policy_observation().explained_anomalies)


if __name__ == "__main__":
    unittest.main()
