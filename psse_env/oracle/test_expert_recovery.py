from __future__ import annotations

import unittest

from psse_env.actions import action_signature
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle


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

    def test_rollback_uses_durable_context_inventory_after_bounded_history_drops_context(
        self,
    ) -> None:
        state_id = "episode:s0"
        rejected = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [3]},
        }
        untried = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [12]},
        }
        # The context event is deliberately absent: a release observation keeps
        # only four transitions, while the same-state provider contract remains
        # fresh after rollback.  Its already-visible action inventory is the
        # bounded-memory bridge to the remaining alternative.
        history = [
            {
                "action": rejected,
                "tool_output": {
                    "execution_status": "success",
                    "candidate_state_id": "episode:s1",
                },
            },
            {
                "action": {"tool": "run_wls", "arguments": {"state_id": "episode:s1"}},
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
            "remaining_anomaly_score": 5.0,
            "unresolved_signatures": [
                "wls_residual_outlier_dominant index=12 channel=Vm"
            ],
            "has_fresh_measurement_context": True,
            "measurement_context_state_id": state_id,
            "fresh_context_evidence": {
                "measurement": {
                    "state_id": state_id,
                    "supported_corrections": [rejected, untried],
                }
            },
            "rejected_hypotheses": [{"source_action": rejected}],
        }

        release_oracle = ExpertPolicyOracle(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            )
        )
        actions = release_oracle.next_actions(state, history)

        self.assertTrue(actions)
        self.assertEqual(actions[0], untried)
        self.assertNotIn(rejected, actions)

    def test_durable_context_inventory_rejects_stale_state_binding(self) -> None:
        stale = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s0", "suspect_group": [3]},
        }
        state = {
            "active_state_id": "episode:s1",
            "last_tool": "rollback_state",
            "remaining_anomaly_score": 5.0,
            "unresolved_signatures": ["measurement_residual_outlier"],
            "has_fresh_measurement_context": True,
            "measurement_context_state_id": "episode:s1",
            "fresh_context_evidence": {
                "measurement": {
                    "state_id": "episode:s0",
                    "supported_corrections": [stale],
                }
            },
        }

        actions = self.oracle.next_actions(state, [])

        self.assertNotIn(stale, actions)

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

    def test_locally_fixed_rejected_branch_diversifies_to_meter_probe(self) -> None:
        state_id = "r0_5e9369fe6e44:s0"
        candidate_id = "r0_5e9369fe6e44:s1"
        first_branch = {
            "tool": "correct_parameters",
            "arguments": {"state_id": state_id, "line_index": 20},
        }
        next_branch = {
            "tool": "correct_parameters",
            "arguments": {"state_id": state_id, "line_index": 17},
        }
        meter = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [16]},
        }
        history = [
            _context_event(
                "get_parameter_context", state_id, [first_branch, next_branch]
            ),
            {"action": first_branch, "tool_output": {"execution_status": "success"}},
            {
                "action": {
                    "tool": "run_wls",
                    "arguments": {"state_id": candidate_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": candidate_id,
                        "target_metric_value": 1.418,
                        "target_metric_threshold": 3.0,
                        "global_progress": 0.2889,
                        "globally_resolved": False,
                        "physical_constraints_ok": True,
                    },
                },
            },
            {
                "action": {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": candidate_id},
                },
                "tool_output": {"execution_status": "success"},
            },
        ]
        state = {
            "active_state_id": state_id,
            "last_tool": "rollback_state",
            "unresolved_signatures": [
                "wls_residual_outlier index=16 channel=Pinj",
                "wls_branch_multiplier line_status_or_parameter line=20",
            ],
            "has_fresh_parameter_context": True,
            "parameter_context_state_id": state_id,
            "has_fresh_measurement_context": False,
            "requires_measurement_context": True,
            "rejected_hypotheses": [
                {
                    "candidate_parent_id": state_id,
                    "candidate_state_id": candidate_id,
                    "source_action": first_branch,
                }
            ],
            "remaining_budget": 12,
        }
        oracle = ExpertPolicyOracle(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            )
        )

        actions = oracle.next_actions(state, history)
        self.assertEqual(actions[0]["tool"], "get_measurement_context")
        self.assertNotIn(next_branch, actions)

        measurement_event = _context_event(
            "get_measurement_context", state_id, [meter]
        )
        history.append(measurement_event)
        state.update(
            {
                "last_tool": "get_measurement_context",
                "last_tool_output": measurement_event["tool_output"],
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
            }
        )
        actions = oracle.next_actions(state, history)
        self.assertEqual(actions[0], meter)

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
            self.assertEqual(env.terminal_outcome, "resolved", source["scenario_id"])
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

        final_state = env.current_state()
        audit = audit_episode_against_truth(
            source,
            final_state,
            terminal=env.is_terminal(),
            terminal_outcome=env.terminal_outcome,
            active_physical_state=env.store.get_state(
                str(final_state["active_state_id"])
            ),
            remaining_truth=None,
        )
        self.assertTrue(env.is_terminal())
        self.assertEqual(audit["problems"], [])
        remaining_check = audit["checks"]["remaining_true_faults"]
        self.assertEqual(remaining_check["status"], "passed")
        self.assertEqual(remaining_check["derived_remaining_fault_count"], 0)
        self.assertEqual(
            remaining_check["evidence_source"],
            "offline_scenario_truth_derivation",
        )

    def test_measurement_parameter_ambiguity_resolves_with_exact_targets(self) -> None:
        from psse_env.examples.generate_round0_aggregate import (
            audit_episode_against_truth,
        )
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
        final_state = env.current_state()
        self.assertFalse(final_state["has_open_candidate"])
        self.assertEqual(env.terminal_outcome, "resolved")
        audit = audit_episode_against_truth(
            source,
            final_state,
            terminal=True,
            terminal_outcome=env.terminal_outcome,
            active_physical_state=env.store.get_state(
                str(final_state["active_state_id"])
            ),
            remaining_truth=None,
        )
        self.assertEqual(audit["problems"], [])


if __name__ == "__main__":
    unittest.main()
