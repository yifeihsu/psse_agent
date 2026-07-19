from __future__ import annotations

import unittest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.transactional_env import (
    TransactionalPSSEEnv,
    _semantic_correction_signature,
)


def _truth_free(scenario: dict) -> dict:
    scenario = dict(scenario)
    for key in list(scenario):
        if key.startswith(("true_", "clean_")) or key in {
            "hidden_truth",
            "oracle_action_hints",
        }:
            scenario.pop(key, None)
    return scenario


class _UnboundEvidenceProvider:
    provider_kind = "deployment"

    def __call__(self, state):
        return {
            "state_id": "wrong-state",
            "state_hash": "wrong-hash",
            "evidence_source": "deployment_diagnostic:test_unbound",
            "request": "operator_escalation:recovery_options_exhausted",
            "additional_evidence_available": False,
            "operator_review_required": True,
        }


class OperatorEscalationContractTests(unittest.TestCase):
    @staticmethod
    def _environment(seed: int = 20260719):
        scenario = _truth_free(
            Round0ScenarioGenerator(seed=seed).build({"multi_measurement": 1})[0]
        )
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=24
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(scenario)
        return env, oracle

    def test_premature_handoff_fails_closed_and_does_not_terminalize(self) -> None:
        env, _ = self._environment()
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": env.current_state()["active_state_id"],
                "request": "operator_escalation:recovery_options_exhausted",
            },
        }

        _, output = env.step(action)

        self.assertEqual(output["execution_status"], "failure")
        self.assertFalse(env.is_terminal())
        self.assertIsNone(env.terminal_outcome)

    def test_handoff_rejects_untried_same_state_supported_corrections(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        _, context_output = env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        self.assertTrue(context_output["tool_metrics"]["supported_corrections"])

        _, output = env.step(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": active_id,
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(
            output["error_code"], "operator_escalation_precondition_not_met"
        )
        self.assertIn(
            "same_state_supported_corrections_unexhausted",
            output["error_detail"],
        )
        self.assertFalse(env.is_terminal())

    def test_handoff_rejects_unbound_supported_correction_inventory(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        supported = env.history[-1]["tool_output"]["tool_metrics"][
            "supported_corrections"
        ]
        supported[0]["arguments"].pop("state_id")
        env.context_flags["rejected_hypotheses"] = [
            {
                "candidate_parent_id": active_id,
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": active_id,
                        "suspect_group": supported[0]["arguments"]["suspect_group"],
                    },
                },
            }
        ]

        _, output = env.step(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": active_id,
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertIn("supported_correction_state_unbound", output["error_detail"])
        self.assertFalse(env.is_terminal())

    def test_handoff_rejects_empty_supported_correction_contract(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        env.history[-1]["tool_output"]["tool_metrics"]["supported_corrections"] = [
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": active_id, "suspect_group": []},
            }
        ]

        _, output = env.step(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": active_id,
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertIn("supported_correction_malformed", output["error_detail"])
        self.assertFalse(env.is_terminal())

    def test_measurement_target_signature_canonicalizes_group_order(self) -> None:
        left = _semantic_correction_signature(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": "s0", "suspect_group": [7, 2]},
            }
        )
        right = _semantic_correction_signature(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": "s1", "suspect_group": [2, 7]},
            }
        )

        self.assertEqual(left, right)

    def test_handoff_is_illegal_with_an_open_candidate(self) -> None:
        env, oracle = self._environment(seed=29)
        for _ in range(3):
            action = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]
            env.step(action)
        self.assertTrue(env.current_state()["has_open_candidate"])
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": env.current_state()["candidate_state_id"],
                "request": "operator_escalation:recovery_options_exhausted",
            },
        }

        _, output = env.step(action)

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "candidate_lifecycle_violation")
        self.assertFalse(env.is_terminal())

    def test_budget_handoff_closes_before_an_unfinishable_candidate_lifecycle(self) -> None:
        scenario = _truth_free(
            Round0ScenarioGenerator(seed=20260719).build(
                {"multi_measurement": 1}
            )[0]
        )
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=3
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(scenario)

        first = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]
        self.assertEqual(first["tool"], "run_wls")
        env.step(first)
        second = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]
        self.assertEqual(second["tool"], "get_measurement_context")
        env.step(second)
        handoff = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]

        self.assertEqual(handoff["tool"], ASK_FOR_MORE_EVIDENCE)
        self.assertEqual(
            handoff["arguments"]["request"],
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        )
        _, output = env.step(handoff)
        self.assertEqual(output["execution_status"], "success")
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "operator_escalation")

    def test_unbound_inventory_response_cannot_terminalize(self) -> None:
        env, oracle = self._environment()
        escalation = None
        for _ in range(24):
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions)
            if actions[0]["tool"] == ASK_FOR_MORE_EVIDENCE:
                escalation = actions[0]
                break
            env.step(actions[0])
        self.assertIsNotNone(escalation)
        env.evidence_providers[ASK_FOR_MORE_EVIDENCE] = _UnboundEvidenceProvider()

        _, output = env.step(escalation)

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertFalse(env.is_terminal())
        self.assertIsNone(env.terminal_outcome)


if __name__ == "__main__":
    unittest.main()
