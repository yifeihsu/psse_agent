from __future__ import annotations

import copy
import json
import unittest

from psse_env.dagger.dataset_builder import (
    examples_to_chat_sft,
    prepare_model_policy_observation,
)
from psse_env.dagger.rollout_collector import (
    BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
    DaggerRolloutCollector,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.transactional_env import TransactionalPSSEEnv


PHASE_CHANNELS = {"three_phase_voltages", "three_phase_branch_currents"}
PHASE_CONTEXT = "get_three_phase_context"
NLM = "run_three_phase_nlm_from_path"


class _BaselinePolicy:
    @staticmethod
    def act(observation: dict) -> dict:
        return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}


class UnbalanceDiscoveryTests(unittest.TestCase):
    """Three-phase diagnosis requires an explicit acquired-measurement result."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = {
            row["scenario_family"]: row
            for row in Round0ScenarioGenerator(seed=20260719).build(
                {"three_phase_unbalance": 1, "measurement": 1, "no_error": 1}
            )
        }

    def _environment(self, scenario: dict) -> tuple[TransactionalPSSEEnv, ExpertPolicyOracle]:
        env = TransactionalPSSEEnv(
            **MatpowerDeploymentProviders().env_kwargs(),
            production_dataset_mode=True,
            max_steps=24,
        )
        env.reset(copy.deepcopy(scenario))
        return env, ExpertPolicyOracle(process_oracle=env.process_oracle)

    def _expert_step(self, env, oracle) -> tuple[str, dict]:
        actions = oracle.next_actions(env.get_oracle_state(), env.history)
        self.assertTrue(actions, env.current_state())
        action = actions[0]
        env.assert_training_decision_evidence(action)
        _, output = env.step(action)
        self.assertEqual(output["execution_status"], "success", (action, output))
        return action["tool"], output

    def _reach_phase_request(self, env, oracle) -> None:
        # A load unbalance is a narrow WLS anomaly (a handful of injection
        # residuals), so the phase-resolved request comes first; spectra would
        # only be requested if the phase request returned nothing.
        tool, output = self._expert_step(env, oracle)
        self.assertEqual(tool, "run_wls")
        breadth = output["tool_metrics"]["anomaly_breadth"]
        self.assertLess(breadth, 0.5, breadth)
        self.assertEqual(
            oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"],
            PHASE_CONTEXT,
        )

    def test_initial_observation_has_no_private_phase_inventory_or_flag(self) -> None:
        with_phases = copy.deepcopy(self.scenarios["three_phase_unbalance"])
        without_phases = copy.deepcopy(with_phases)
        for key in PHASE_CHANNELS:
            without_phases["metadata"].pop(key, None)
        observations = []
        for scenario in (with_phases, without_phases):
            env, oracle = self._environment(scenario)
            observation = env.get_policy_observation().as_dict()
            self.assertEqual(observation["available_evidence"], [])
            self.assertEqual(observation["unresolved_signatures"], [])
            contexts = observation["fresh_context_evidence"]
            self.assertNotIn("harmonic", contexts)
            self.assertNotIn("three_phase", contexts)
            self.assertFalse(contexts["wls"]["successful"])
            self.assertNotIn("true_unbalance_errors", json.dumps(observation))
            for channel in PHASE_CHANNELS:
                self.assertNotIn(channel, json.dumps(observation))
            self.assertEqual(
                oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"],
                "run_wls",
            )
            observations.append(observation)
        self.assertEqual(observations[0], observations[1])

    def test_failed_early_actions_do_not_reveal_phase_channels(self) -> None:
        for tool in (PHASE_CONTEXT, NLM, "get_harmonic_context", "run_hse_from_path"):
            with self.subTest(tool=tool):
                env, _ = self._environment(self.scenarios["three_phase_unbalance"])
                action = {"tool": tool, "arguments": {"state_id": env.store.active_state_id}}
                with self.assertRaises(ValueError):
                    env.assert_training_decision_evidence(action)
                _, output = env.step(action)
                self.assertEqual(output["execution_status"], "failure")
                self.assertEqual(output["error_code"], "missing_precondition")
                self.assertFalse(output["state_mutated"])
                observation = env.get_policy_observation()
                self.assertFalse(PHASE_CHANNELS & set(observation.available_evidence))
                self.assertNotIn("three_phase", observation.fresh_context_evidence)
                self.assertFalse(observation.explained_anomalies)

    def test_wls_and_spectral_request_do_not_acquire_phase_measurements(self) -> None:
        env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
        self.assertEqual(self._expert_step(env, oracle)[0], "run_wls")
        self.assertFalse(PHASE_CHANNELS & set(env.get_policy_observation().available_evidence))
        # A student may still ask for spectra first; that request is legal,
        # returns nothing here, and acquires no phase measurements.
        spectral = {"tool": "get_harmonic_context", "arguments": {"state_id": env.store.active_state_id}}
        env.assert_training_decision_evidence(spectral)
        _, output = env.step(spectral)
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(output["tool_metrics"]["harmonic_context_status"], "unavailable")
        self.assertFalse(PHASE_CHANNELS & set(env.get_policy_observation().available_evidence))
        self.assertNotIn("three_phase", env.get_policy_observation().fresh_context_evidence)
        self.assertEqual(
            oracle.next_actions(env.get_oracle_state(), env.history)[0]["tool"],
            PHASE_CONTEXT,
        )

    def test_same_observable_anomaly_requests_phases_without_private_inventory_hint(self) -> None:
        views = []
        actions = []
        for available in (True, False):
            scenario = copy.deepcopy(self.scenarios["three_phase_unbalance"])
            if not available:
                for key in PHASE_CHANNELS:
                    scenario["metadata"].pop(key, None)
            env, oracle = self._environment(scenario)
            self._reach_phase_request(env, oracle)
            action = oracle.next_actions(env.get_oracle_state(), env.history)[0]
            actions.append(action)
            view, _ = prepare_model_policy_observation(env.get_policy_observation().as_dict())
            views.append(view)
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[0]["tool"], PHASE_CONTEXT)
        self.assertEqual(views[0], views[1])

    def test_nlm_requires_context_even_after_successful_wls(self) -> None:
        env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
        self._reach_phase_request(env, oracle)
        action = {"tool": NLM, "arguments": {"state_id": env.store.active_state_id}}
        with self.assertRaises(ValueError):
            env.assert_training_decision_evidence(action)
        _, output = env.step(action)
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "missing_precondition")
        self.assertFalse(env.get_policy_observation().explained_anomalies)
        self.assertFalse(PHASE_CHANNELS & set(env.get_policy_observation().available_evidence))
        self.assertEqual(self._expert_step(env, oracle)[0], PHASE_CONTEXT)
        self.assertEqual(self._expert_step(env, oracle)[0], NLM)
        self.assertEqual(self._expert_step(env, oracle)[0], "finalize_diagnosis")

    def test_premature_failed_attempts_recover_through_the_full_diagnostic_sequence(self) -> None:
        env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
        for tool in (PHASE_CONTEXT, NLM, PHASE_CONTEXT, NLM):
            _, output = env.step({"tool": tool, "arguments": {"state_id": env.store.active_state_id}})
            self.assertEqual(output["execution_status"], "failure")
        executed = []
        for _ in range(8):
            if env.is_terminal():
                break
            executed.append(self._expert_step(env, oracle)[0])
        self.assertEqual(executed, ["run_wls", PHASE_CONTEXT, NLM, "finalize_diagnosis"])
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "resolved")

    def test_full_sequence_survives_omitted_tool_history(self) -> None:
        env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
        for tool in (PHASE_CONTEXT, NLM):
            _, output = env.step({"tool": tool, "arguments": {"state_id": env.store.active_state_id}})
            self.assertEqual(output["execution_status"], "failure")
        executed = []
        for _ in range(8):
            if env.is_terminal():
                break
            # The expert receives only the same durable observable state as
            # the model; even the premature failures have left the window.
            observation = env.get_policy_observation(history=[])
            self.assertEqual(observation.history_window, [])
            actions = oracle.next_actions(observation, [])
            self.assertTrue(actions, observation.as_dict())
            env.assert_training_decision_evidence(actions[0])
            _, output = env.step(actions[0])
            self.assertEqual(output["execution_status"], "success", (actions[0], output))
            executed.append(actions[0]["tool"])
        self.assertEqual(executed, ["run_wls", PHASE_CONTEXT, NLM, "finalize_diagnosis"])
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "resolved")

    def test_failed_latest_wls_invalidates_historical_diagnostic_prerequisite(self) -> None:
        env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
        self._reach_phase_request(env, oracle)
        self.assertEqual(self._expert_step(env, oracle)[0], PHASE_CONTEXT)
        self.assertTrue(env.get_policy_observation().fresh_context_evidence["wls"]["successful"])

        def failed_wls(state):
            del state
            return {"execution_status": "failure", "error_code": "wls_measurements_unavailable"}

        env.wls_runner = failed_wls
        _, output = env.step({"tool": "run_wls", "arguments": {"state_id": env.store.active_state_id}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "wls_measurements_unavailable")
        observation = env.get_policy_observation(history=[])
        # Historical signatures may remain useful evidence, but cannot stand
        # in for the unsuccessful latest assessment of this active state.
        self.assertTrue(any(str(item).startswith("wls_") for item in observation.unresolved_signatures))
        self.assertFalse(observation.fresh_context_evidence["wls"]["successful"])
        for tool in (PHASE_CONTEXT, NLM):
            action = {"tool": tool, "arguments": {"state_id": env.store.active_state_id}}
            with self.assertRaises(ValueError):
                env.assert_training_decision_evidence(action)
            _, rejected = env.step(action)
            self.assertEqual(rejected["execution_status"], "failure")
            self.assertEqual(rejected["error_code"], "missing_precondition")
            self.assertFalse(rejected["state_mutated"])
        self.assertFalse(env.get_policy_observation().explained_anomalies)

    def test_unavailable_phase_request_preserves_anomaly_and_classical_investigation(self) -> None:
        env, oracle = self._environment(self.scenarios["measurement"])
        self._reach_phase_request(env, oracle)
        before = list(env.get_policy_observation().unresolved_signatures)
        self.assertTrue(any(str(item).startswith("wls_") for item in before))
        tool, output = self._expert_step(env, oracle)
        self.assertEqual(tool, PHASE_CONTEXT)
        metrics = output["tool_metrics"]
        self.assertEqual(metrics["three_phase_context_status"], "unavailable")
        self.assertEqual(metrics["available_evidence_channels"], [])
        observation = env.get_policy_observation()
        self.assertEqual(observation.unresolved_signatures, before)
        self.assertFalse(observation.explained_anomalies)
        self.assertFalse(observation.no_material_anomaly_remaining)
        self.assertFalse(env.is_terminal())
        self.assertTrue(observation.fresh_context_evidence["three_phase"]["request_attempted"])
        executed = []
        for _ in range(12):
            tool, _ = self._expert_step(env, oracle)
            executed.append(tool)
            if tool.startswith("correct_") or env.is_terminal():
                break
        self.assertTrue(any(tool.startswith("correct_") for tool in executed), executed)
        self.assertNotIn(NLM, executed)
        self.assertNotIn(PHASE_CONTEXT, executed)
        self.assertFalse(env.get_policy_observation().explained_anomalies)

    def test_successful_phase_refresh_reopens_nlm_after_an_earlier_attempt(self) -> None:
        for old_status in ("success", "failure"):
            for omit_history in (False, True):
                with self.subTest(old_status=old_status, omit_history=omit_history):
                    env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
                    self._reach_phase_request(env, oracle)
                    self.assertEqual(self._expert_step(env, oracle)[0], PHASE_CONTEXT)
                    provider = env.evidence_providers[NLM]

                    def inconclusive(state, action):
                        return {
                            "execution_status": old_status,
                            "error_code": "temporary_sensor_failure" if old_status == "failure" else None,
                            "state_id": state["state_id"],
                            "state_hash": state["state_hash"],
                            "evidence_source": "deployment_diagnostic:three_phase_nlm",
                            "diagnostic_acceptance": {"accepted": False, "reason": "inconclusive_scan"},
                            "nlm_summary": {"note": "No explanation accepted for this scan."},
                        }

                    env.evidence_providers[NLM] = inconclusive
                    active = env.store.active_state_id
                    _, output = env.step({"tool": NLM, "arguments": {"state_id": active}})
                    self.assertEqual(output["execution_status"], old_status)
                    observation = env.get_policy_observation().as_dict()
                    self.assertTrue(observation["fresh_context_evidence"]["three_phase"]["nlm_attempted"])
                    compact, _ = prepare_model_policy_observation(observation, history=[])
                    self.assertTrue(compact["fresh_context_evidence"]["three_phase"]["nlm_attempted"])
                    self.assertFalse(observation["explained_anomalies"])
                    env.evidence_providers[NLM] = provider
                    _, output = env.step({"tool": PHASE_CONTEXT, "arguments": {"state_id": active}})
                    self.assertEqual(output["execution_status"], "success")
                    self.assertFalse(env.current_state()["fresh_context_evidence"]["three_phase"].get("nlm_attempted", False))
                    history = [] if omit_history else env.history
                    actions = oracle.next_actions(env.get_policy_observation(history=history), history)
                    self.assertTrue(actions)
                    self.assertEqual(actions[0]["tool"], NLM)
                    env.assert_training_decision_evidence(actions[0])
                    _, output = env.step(actions[0])
                    self.assertEqual(output["execution_status"], "success")
                    self.assertEqual(self._expert_step(env, oracle)[0], "finalize_diagnosis")
                    self.assertEqual(env.terminal_outcome, "resolved")

    def test_unbound_provider_context_cannot_enable_nlm(self) -> None:
        env, oracle = self._environment(self.scenarios["three_phase_unbalance"])
        self._reach_phase_request(env, oracle)
        provider = env.evidence_providers[PHASE_CONTEXT]

        def unbound(state, action):
            result = dict(provider(state, action))
            result["state_hash"] = "different-physical-state"
            return result

        env.evidence_providers[PHASE_CONTEXT] = unbound
        _, output = env.step({"tool": PHASE_CONTEXT, "arguments": {"state_id": env.store.active_state_id}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertFalse(PHASE_CHANNELS & set(env.get_policy_observation().available_evidence))
        action = {"tool": NLM, "arguments": {"state_id": env.store.active_state_id}}
        with self.assertRaises(ValueError):
            env.assert_training_decision_evidence(action)
        _, output = env.step(action)
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "missing_precondition")

    def test_context_from_another_active_state_cannot_enable_nlm(self) -> None:
        scenario = self.scenarios["three_phase_unbalance"]
        env, oracle = self._environment(scenario)
        self._reach_phase_request(env, oracle)
        self.assertEqual(self._expert_step(env, oracle)[0], PHASE_CONTEXT)
        old = env.get_policy_observation().fresh_context_evidence["three_phase"]
        self.assertEqual(old["state_hash"], env.store.state_hash(env.store.active_state_id))
        new_state = env.store.create_root(
            case=copy.deepcopy(scenario["case"]),
            measurements=copy.deepcopy(scenario["measurements"]),
            metadata=copy.deepcopy(scenario["metadata"]),
            episode_id="new-acquisition-required",
        )
        self.assertNotEqual(old["state_id"], new_state)
        self.assertFalse(PHASE_CHANNELS & set(env.get_policy_observation().available_evidence))
        action = {"tool": NLM, "arguments": {"state_id": new_state}}
        with self.assertRaises(ValueError):
            env.assert_training_decision_evidence(action)
        _, output = env.step(action)
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "missing_precondition")

    def test_real_source_episode_and_canonical_export_use_only_acquired_evidence(self) -> None:
        scenario = self.scenarios["three_phase_unbalance"]
        env, oracle = self._environment(scenario)
        rows = DaggerRolloutCollector(
            env=env,
            policy=_BaselinePolicy(),
            expert_oracle=oracle,
            supervision_policy=BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
        ).collect_iteration(scenarios=[scenario], iteration=0, beta=1.0, max_steps=10)
        expected = ["run_wls", PHASE_CONTEXT, NLM, "finalize_diagnosis"]
        self.assertEqual([row["preferred_action"]["tool"] for row in rows], expected)
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "resolved")
        record = env.get_policy_observation().explained_anomalies[0]
        self.assertEqual(record["family"], "three_phase_unbalance")
        self.assertEqual(record["detail"]["bus_1based"], scenario["hidden_truth"]["true_unbalance_errors"][0]["unbalance_bus"])
        self.assertTrue(any(str(item).startswith("wls_") for item in record["explained_signatures"]))
        # Row 2 is the NLM decision: phase measurements are acquired, nothing
        # is explained yet.
        acquired = rows[2]["policy_observation"]
        self.assertTrue(PHASE_CHANNELS <= set(acquired["available_evidence"]))
        self.assertFalse(acquired["explained_anomalies"])
        context = acquired["fresh_context_evidence"]["three_phase"]
        self.assertTrue(context["request_attempted"])
        self.assertEqual(context["three_phase_context_status"], "available")
        self.assertEqual(context["state_id"], acquired["active_state_id"])
        exported = examples_to_chat_sft(rows, protocol="canonical", allow_ineligible_auxiliary=True)
        self.assertEqual(len(exported), 4)
        self.assertEqual(
            [row["messages"][2]["tool_calls"][0]["function"]["name"] for row in exported],
            ["wls_from_path", *expected[1:]],
        )
        states = [json.loads(row["messages"][1]["content"])["state"] for row in exported]
        # The WLS and phase-request decisions see no phase channels and no
        # unbalance signature; acquisition happens after the request.
        for state in states[:2]:
            self.assertFalse(PHASE_CHANNELS & set(state["available_evidence"]))
            self.assertFalse(any("unbalance" in str(item) for item in state["unresolved_signatures"]))
        self.assertTrue(PHASE_CHANNELS <= set(states[2]["available_evidence"]))
        context = states[2]["fresh_context_evidence"]["three_phase"]
        self.assertTrue(context["request_attempted"])
        self.assertEqual(context["state_id"], "active")
        self.assertTrue(str(context["state_hash"]).startswith("h"))
        # Fresh evidence remains available if conversation history is omitted.
        compact, _ = prepare_model_policy_observation(acquired, history=[])
        self.assertTrue(compact["fresh_context_evidence"]["three_phase"]["request_attempted"])
        for row in exported:
            tools = {tool["function"]["name"] for tool in row["tools"]}
            self.assertIn(PHASE_CONTEXT, tools)
            text = json.dumps(row["messages"])
            self.assertNotIn("true_unbalance_errors", text)
            self.assertNotIn("sensor_signatures_withheld", text)
            self.assertNotIn("hidden_truth", text)

    def test_clean_control_does_not_request_additional_measurements(self) -> None:
        env, oracle = self._environment(self.scenarios["no_error"])
        executed = [self._expert_step(env, oracle)[0], self._expert_step(env, oracle)[0]]
        self.assertEqual(executed, ["run_wls", "finalize_diagnosis"])
        self.assertTrue(env.is_terminal())
        contexts = env.get_policy_observation().fresh_context_evidence
        self.assertNotIn("harmonic", contexts)
        self.assertNotIn("three_phase", contexts)
        self.assertTrue(contexts["wls"]["successful"])
        self.assertFalse(contexts["wls"]["anomalous"])
        self.assertFalse(env.get_policy_observation().explained_anomalies)


if __name__ == "__main__":
    unittest.main()
