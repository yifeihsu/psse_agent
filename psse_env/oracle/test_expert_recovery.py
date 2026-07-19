from __future__ import annotations

import unittest

from psse_env.actions import action_signature
from psse_env.oracle import ExpertPolicyOracle


def _context_event(tool: str, state_id: str, corrections: list[dict]) -> dict:
    return {
        "action": {"tool": tool, "arguments": {"state_id": state_id}},
        "tool_output": {
            "execution_status": "success",
            "tool_metrics": {
                "state_id": state_id,
                "supported_corrections": corrections,
            },
        },
    }


class SequentialRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.oracle = ExpertPolicyOracle()

    def test_rollback_uses_next_observable_context_alternative(self) -> None:
        state_id = "episode:s0"
        wrong = {
            "tool": "correct_topology",
            "arguments": {"state_id": state_id, "line_index": 6, "status": 0},
        }
        alternative = {
            "tool": "correct_topology",
            "arguments": {"state_id": state_id, "line_index": 3, "status": 0},
        }
        history = [
            _context_event("get_topology_context", state_id, [wrong, alternative]),
            {
                "action": wrong,
                "tool_output": {"execution_status": "success"},
            },
            {
                "action": {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": "episode:s1"},
                },
                "tool_output": {"execution_status": "success"},
            },
        ]
        state = {
            "active_state_id": state_id,
            "last_tool": "rollback_state",
            "last_tool_status": "success",
            "last_tool_output": {"execution_status": "success"},
            "unresolved_signatures": [
                "wls_branch_multiplier_dominant line_status_or_parameter line=3"
            ],
            "has_fresh_topology_context": True,
            "topology_context_state_id": state_id,
            "rejected_hypotheses": [{"source_action": wrong}],
        }

        actions = self.oracle.next_actions(state, history)

        self.assertTrue(actions)
        self.assertEqual(actions[0], alternative)
        self.assertNotIn(wrong, actions)

    def test_missing_parameter_scans_suppresses_entire_family_on_same_state(self) -> None:
        state_id = "episode:s0"
        parameter_actions = [
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": state_id, "line_index": line},
            }
            for line in (6, 3, 7)
        ]
        topology = {
            "tool": "correct_topology",
            "arguments": {"state_id": state_id, "line_index": 3, "status": 0},
        }
        history = [
            _context_event("get_parameter_context", state_id, parameter_actions),
            {
                "action": parameter_actions[0],
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "parameter_scans_missing",
                    "error_detail": "metadata.parameter_scans is unavailable",
                },
            },
            _context_event("get_topology_context", state_id, [topology]),
        ]
        state = {
            "active_state_id": state_id,
            "last_tool": "get_topology_context",
            "last_tool_status": "success",
            "last_tool_output": history[-1]["tool_output"],
            "unresolved_signatures": [
                "wls_branch_multiplier_dominant line_status_or_parameter line=3"
            ],
            "has_fresh_parameter_context": True,
            "parameter_context_state_id": state_id,
            "has_fresh_topology_context": True,
            "topology_context_state_id": state_id,
        }

        actions = self.oracle.next_actions(state, history)

        self.assertTrue(actions)
        self.assertEqual(actions[0], topology)
        self.assertNotIn("correct_parameters", {action["tool"] for action in actions})

    def test_context_alternatives_are_not_reused_after_active_state_changes(self) -> None:
        old_state = "episode:s0"
        active_state = "episode:s1"
        stale = {
            "tool": "correct_topology",
            "arguments": {"state_id": old_state, "line_index": 3, "status": 0},
        }
        history = [_context_event("get_topology_context", old_state, [stale])]
        state = {
            "active_state_id": active_state,
            "last_tool": "commit_state",
            "last_tool_status": "success",
            "last_tool_output": {"execution_status": "success"},
            "unresolved_signatures": [
                "wls_branch_multiplier_dominant line_status_or_parameter line=3"
            ],
            "has_fresh_topology_context": False,
            "topology_context_state_id": None,
        }

        actions = self.oracle.next_actions(state, history)

        self.assertTrue(actions)
        self.assertIn(
            actions[0]["tool"],
            {"get_parameter_context", "get_topology_context"},
        )
        self.assertEqual(actions[0]["arguments"]["state_id"], active_state)
        self.assertNotIn("correct_topology", {action["tool"] for action in actions})
        self.assertNotIn(stale, actions)

    def test_old_accepted_target_does_not_block_fresh_state_target(self) -> None:
        old_state = "episode:s0"
        active_state = "episode:s1"
        old_action = {
            "tool": "correct_topology",
            "arguments": {"state_id": old_state, "line_index": 3, "status": 0},
        }
        refreshed_action = {
            "tool": "correct_topology",
            "arguments": {"state_id": active_state, "line_index": 3, "status": 0},
        }
        history = [
            {"action": old_action, "tool_output": {"execution_status": "success"}},
            _context_event(
                "get_topology_context", active_state, [refreshed_action]
            ),
        ]
        state = {
            "active_state_id": active_state,
            "last_tool": "get_topology_context",
            "last_tool_status": "success",
            "last_tool_output": history[-1]["tool_output"],
            "unresolved_signatures": [
                "wls_branch_multiplier_dominant line_status_or_parameter line=3"
            ],
            "has_fresh_topology_context": True,
            "topology_context_state_id": active_state,
            "accepted_corrections": [{"source_action": old_action}],
        }

        actions = self.oracle.next_actions(state, history)

        self.assertIn(refreshed_action, actions)

    def test_inconclusive_evidence_actions_are_bounded_by_rollback(self) -> None:
        for evidence_tool in ("ask_for_more_evidence", "run_alternative_test"):
            with self.subTest(evidence_tool=evidence_tool):
                policy_observation = {
                    "active_state_id": "episode:s0",
                    "candidate_state_id": "episode:s1",
                    "candidate_status": "verified",
                    "candidate_lifecycle": "VERIFIED",
                    "has_open_candidate": True,
                    "has_verified_candidate": True,
                    "last_tool": evidence_tool,
                    "last_tool_status": "success",
                    "last_tool_output": {"execution_status": "success"},
                }
                state = {
                    "policy_observation": policy_observation,
                    "candidate_disposition": "INCONCLUSIVE",
                }

                actions = self.oracle.next_actions(state, [])

                self.assertTrue(actions)
                self.assertEqual(actions[0]["tool"], "rollback_state")

    def test_exhausted_hif_diagnostics_do_not_silently_finalize(self) -> None:
        state_id = "episode:s0"
        history = [
            {
                "action": {
                    "tool": "run_three_phase_nlm_from_path",
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "nlm_summary": {
                            "top_hif_groups": [{"branch_row0": 4, "score": 0.9}]
                        }
                    },
                },
            },
            {
                "action": {
                    "tool": "estimate_hif_location_magnitude_multiscan_from_path",
                    "arguments": {
                        "state_id": state_id,
                        "candidate_branch_row0": 4,
                    },
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "diagnostic_acceptance": {"accepted": False}
                    },
                },
            },
            {
                "action": {
                    "tool": "estimate_hif_location_magnitude_from_path",
                    "arguments": {
                        "state_id": state_id,
                        "candidate_branch_row0": 4,
                    },
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "diagnostic_acceptance": {"accepted": False}
                    },
                },
            },
            {
                "action": {"tool": "run_wls", "arguments": {"state_id": state_id}},
                "tool_output": {"execution_status": "success"},
            },
        ]
        state = {
            "active_state_id": state_id,
            "last_tool": "run_wls",
            "last_tool_status": "success",
            "last_tool_output": {"execution_status": "success"},
            "unresolved_signatures": ["hif_suspected_zero_sequence"],
            "available_evidence": ["nlm_diagnostic", "hif_scan_window"],
            "remaining_anomaly_score": 2.0,
            "no_material_anomaly_remaining": False,
        }

        actions = self.oracle.next_actions(state, history)
        final = {"tool": "finalize_diagnosis", "arguments": {}}

        self.assertNotIn("finalize_diagnosis", {action["tool"] for action in actions})
        self.assertFalse(self.oracle.process_oracle.check(state, final)["process_valid"])


class ReleaseMixedErrorRecoveryTests(unittest.TestCase):
    @staticmethod
    def _without_privileged_targets(scenario: dict) -> dict:
        scenario = dict(scenario)
        for key in list(scenario):
            if key.startswith("true_") or key.startswith("clean_") or key in {
                "hidden_truth",
                "oracle_action_hints",
            }:
                scenario.pop(key, None)
        return scenario

    def test_default_measurement_topology_matrix_is_truth_free_and_terminal(self) -> None:
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator
        from psse_env.transactional_env import TransactionalPSSEEnv

        scenarios = Round0ScenarioGenerator(seed=20260719).build(
            {"measurement+topology": 3}
        )
        self.assertEqual(len(scenarios), 3)
        for source in scenarios:
            providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
            env = TransactionalPSSEEnv(
                **providers.env_kwargs(),
                production_dataset_mode=True,
                max_steps=24,
            )
            oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
            env.reset(self._without_privileged_targets(source))
            correction_signatures: list[str] = []
            for _ in range(24):
                if env.is_terminal():
                    break
                actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
                self.assertTrue(
                    actions,
                    f"expert stalled in {source['scenario_id']} after {env.history}",
                )
                action = actions[0]
                if action["tool"].startswith("correct_"):
                    correction_signatures.append(action_signature(action))
                env.step(action)

            self.assertTrue(env.is_terminal(), source["scenario_id"])
            self.assertEqual(
                len(correction_signatures),
                len(set(correction_signatures)),
                source["scenario_id"],
            )
            # Missing scan metadata is state-wide.  Retrying a different line
            # cannot make the same parameter executor contract valid.
            self.assertLessEqual(
                sum(
                    signature.startswith("correct_parameters:")
                    for signature in correction_signatures
                ),
                1,
                source["scenario_id"],
            )

    def test_near_tied_healthy_topology_partial_is_not_committed(self) -> None:
        from psse_env.examples.generate_round0_aggregate import (
            audit_episode_against_truth,
        )
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator
        from psse_env.transactional_env import TransactionalPSSEEnv

        source = Round0ScenarioGenerator(seed=31).build(
            {"measurement+topology": 1}
        )[0]
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=24
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(self._without_privileged_targets(source))

        for _ in range(24):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions)
            env.step(actions[0])

        audit = audit_episode_against_truth(
            source,
            env.current_state(),
            terminal=env.is_terminal(),
            terminal_outcome=env.terminal_outcome,
        )
        self.assertTrue(env.is_terminal())
        self.assertEqual(audit["problems"], [])

    def test_measurement_parameter_ambiguity_hands_off_before_budget_deadlock(self) -> None:
        from psse_env.actions import RECOVERY_BUDGET_EXHAUSTED_REQUEST
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator
        from psse_env.transactional_env import TransactionalPSSEEnv

        source = Round0ScenarioGenerator(seed=37).build(
            {"measurement+parameter": 1}
        )[0]
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=24
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(self._without_privileged_targets(source))

        for _ in range(24):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions)
            env.step(actions[0])

        self.assertTrue(env.is_terminal())
        self.assertFalse(env.current_state()["has_open_candidate"])
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(
            env.history[-1]["action"]["arguments"]["request"],
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        )


if __name__ == "__main__":
    unittest.main()
