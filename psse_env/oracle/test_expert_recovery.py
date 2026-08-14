from __future__ import annotations

import unittest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    action_signature,
)
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

    def test_post_correction_confirmation_uses_last_step_budget_handoff(self) -> None:
        state_id = "episode:s1"
        state = {
            "active_state_id": state_id,
            "accepted_corrections": [
                {
                    "candidate_state_id": state_id,
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {
                            "state_id": "episode:s0",
                            "suspect_group": [3],
                        },
                    },
                }
            ],
            "remaining_anomaly_score": 0.5,
            "no_material_anomaly_remaining": False,
            "unresolved_signatures": [
                POST_CORRECTION_CONFIRMATION_SIGNATURE
            ],
            "remaining_budget": 1,
            "last_tool": "commit_state",
            "semantic_field_provenance": {
                "remaining_anomaly_score": "deployment_wls:candidate_verification"
            },
        }
        history = [
            {
                "action": {
                    "tool": "run_wls",
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {"execution_status": "success"},
            }
        ]

        action = self.oracle.next_actions(state, history)[0]

        self.assertEqual(action["tool"], ASK_FOR_MORE_EVIDENCE)
        self.assertEqual(
            action["arguments"]["request"],
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        )

    def test_post_correction_confirmation_spends_two_steps_on_investigation(self) -> None:
        state_id = "episode:s1"
        state = {
            "active_state_id": state_id,
            "accepted_corrections": [
                {
                    "candidate_state_id": state_id,
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"state_id": "episode:s0", "suspect_group": [3]},
                    },
                }
            ],
            "remaining_anomaly_score": 0.5,
            "no_material_anomaly_remaining": False,
            "unresolved_signatures": [
                POST_CORRECTION_CONFIRMATION_SIGNATURE
            ],
            "remaining_budget": 2,
            "last_tool": "commit_state",
            "semantic_field_provenance": {
                "remaining_anomaly_score": "deployment_wls:candidate_verification"
            },
        }
        history = [
            {
                "action": {
                    "tool": "run_wls",
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {"execution_status": "success"},
            }
        ]

        action = self.oracle.next_actions(state, history)[0]

        self.assertEqual(action["tool"], "get_measurement_context")

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

    def _assert_truth_safe_operator_handoff(self, env, source: dict) -> None:
        from psse_env.dagger.release_audit import (
            ACCEPTED_TARGET_NONREGRESSION_CHECK,
            ACCEPTED_TARGETS_CHECK,
            REMAINING_FAULTS_CHECK,
        )
        from psse_env.examples.generate_round0_aggregate import (
            audit_episode_against_truth,
        )

        self.assertTrue(env.is_terminal(), source["scenario_id"])
        self.assertEqual(
            env.terminal_outcome,
            "operator_escalation",
            source["scenario_id"],
        )
        executed_tools = [
            str((record.get("action") or {}).get("tool") or "")
            for record in env.history
        ]
        self.assertNotIn("finalize_diagnosis", executed_tools)

        final_state = env.current_state()
        active_physical_state = env.store.get_state(
            str(final_state["active_state_id"])
        )
        handoff_audit = audit_episode_against_truth(
            source,
            final_state,
            terminal=True,
            terminal_outcome=env.terminal_outcome,
            active_physical_state=active_physical_state,
            remaining_truth=None,
        )
        # The operator handoff itself must contain no false correction or
        # physical-regression decision.
        self.assertEqual(handoff_audit["problems"], [])
        for check in (
            ACCEPTED_TARGETS_CHECK,
            ACCEPTED_TARGET_NONREGRESSION_CHECK,
        ):
            self.assertEqual(handoff_audit["checks"][check]["status"], "passed")
        self.assertEqual(
            handoff_audit["checks"][REMAINING_FAULTS_CHECK]["status"],
            "not_required",
        )

        # This is a truth-side regression check only; it does not mint an
        # online finalize label.  Zero derived remaining targets, combined
        # with the accepted-target check above, proves exact target coverage.
        physical_resolution_audit = audit_episode_against_truth(
            source,
            final_state,
            terminal=True,
            terminal_outcome="resolved",
            active_physical_state=active_physical_state,
            remaining_truth=None,
        )
        self.assertEqual(physical_resolution_audit["problems"], [])
        remaining_check = physical_resolution_audit["checks"][
            REMAINING_FAULTS_CHECK
        ]
        self.assertEqual(remaining_check["status"], "passed")
        self.assertEqual(remaining_check["derived_remaining_fault_count"], 0)

    def test_default_measurement_topology_matrix_is_truth_free_and_hands_off(self) -> None:
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

            self._assert_truth_safe_operator_handoff(env, source)
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

    def test_near_tied_healthy_topology_partial_hands_off_safely(self) -> None:
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

        self._assert_truth_safe_operator_handoff(env, source)

    def test_measurement_parameter_ambiguity_repairs_then_hands_off(self) -> None:
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

        final_state = env.current_state()
        self.assertFalse(final_state["has_open_candidate"])
        self._assert_truth_safe_operator_handoff(env, source)


class MeasurementTargetIndexContractTests(unittest.TestCase):
    def test_measurement_updates_keys_survive_json_round_trip(self) -> None:
        """JSON object keys are always strings; frozen-suite setup actions
        arrive in that form and their indices must still count as targets."""
        from psse_env.oracle.measurement_recovery_evidence import (
            accepted_measurement_indices,
            measurement_target_indices,
        )

        action = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "episode:s0",
                "measurement_updates": {"19": -0.027},
            },
        }
        self.assertEqual(measurement_target_indices(action), {19})
        self.assertEqual(
            measurement_target_indices(
                {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "measurement_updates": {"nonsense": 1.0, "-3": 1.0},
                    },
                }
            ),
            set(),
        )
        state = {"accepted_corrections": [{"source_action": action}]}
        self.assertEqual(accepted_measurement_indices(state), {19})


class NoEvidenceInvestigationTests(unittest.TestCase):
    def test_exact_provider_no_evidence_context_counts_as_investigation(self) -> None:
        oracle = ExpertPolicyOracle()
        state_id = "episode:s1"
        history = [
            {
                "action": {"tool": "run_wls", "arguments": {"state_id": state_id}},
                "tool_output": {"execution_status": "success"},
            },
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "insufficient_observable_evidence",
                    "error_detail": (
                        "get_measurement_context_provider_returned_no_evidence"
                    ),
                },
            },
        ]
        wls_seen, investigation_seen = oracle._observable_recovery_prerequisites(
            {"active_state_id": state_id}, history, active_id=state_id
        )
        self.assertTrue(wls_seen)
        self.assertTrue(investigation_seen)

    def test_other_insufficient_evidence_details_do_not_count(self) -> None:
        oracle = ExpertPolicyOracle()
        state_id = "episode:s1"
        invalid_details = (
            None,
            "get_measurement_context_provider_evidence_unbound",
            "get_measurement_context_supported_correction_contract_invalid",
            "get_measurement_context_route_contract_invalid",
            "get_measurement_context_terminal_closure_contract_invalid",
            "get_parameter_context_provider_returned_no_evidence",
        )
        for detail in invalid_details:
            with self.subTest(error_detail=detail):
                output = {
                    "execution_status": "failure",
                    "error_code": "insufficient_observable_evidence",
                }
                if detail is not None:
                    output["error_detail"] = detail
                history = [
                    {
                        "action": {
                            "tool": "get_measurement_context",
                            "arguments": {"state_id": state_id},
                        },
                        "tool_output": output,
                    },
                ]
                _, investigation_seen = oracle._observable_recovery_prerequisites(
                    {"active_state_id": state_id}, history, active_id=state_id
                )
                self.assertFalse(investigation_seen)

    def test_latest_integrity_failure_shadows_older_no_evidence(self) -> None:
        oracle = ExpertPolicyOracle()
        state_id = "episode:s1"
        action = {
            "tool": "get_measurement_context",
            "arguments": {"state_id": state_id},
        }
        history = [
            {
                "action": action,
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "insufficient_observable_evidence",
                    "error_detail": (
                        "get_measurement_context_provider_returned_no_evidence"
                    ),
                },
            },
            {
                "action": action,
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "insufficient_observable_evidence",
                    "error_detail": (
                        "get_measurement_context_provider_evidence_unbound"
                    ),
                },
            },
        ]
        _, investigation_seen = oracle._observable_recovery_prerequisites(
            {"active_state_id": state_id}, history, active_id=state_id
        )
        self.assertFalse(investigation_seen)

    def test_failed_context_on_other_state_does_not_count(self) -> None:
        oracle = ExpertPolicyOracle()
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s0"},
                },
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "insufficient_observable_evidence",
                    "error_detail": (
                        "get_measurement_context_provider_returned_no_evidence"
                    ),
                },
            },
        ]
        _, investigation_seen = oracle._observable_recovery_prerequisites(
            {"active_state_id": "episode:s1"}, history, active_id="episode:s1"
        )
        self.assertFalse(investigation_seen)

    def test_no_evidence_provider_outcome_supports_audited_handoff(self) -> None:
        """A provider observably reporting no evidence for the active state
        must open the safe-handoff path end to end: the expert proposes the
        escalation and the environment's audit accepts it as terminal."""
        from psse_env.actions import GET_MEASUREMENT_CONTEXT
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator
        from psse_env.transactional_env import TransactionalPSSEEnv

        source = Round0ScenarioGenerator(seed=7).build({"measurement": 1})[0]
        scenario = {
            key: value
            for key, value in source.items()
            if not key.startswith(("true_", "clean_"))
            and key not in {"hidden_truth", "oracle_action_hints"}
        }
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        kwargs = providers.env_kwargs()

        def no_evidence_context(payload: dict) -> dict:
            del payload
            return {
                "context_tool": GET_MEASUREMENT_CONTEXT,
                "execution_status": "failure",
                "error_code": "insufficient_observable_evidence",
                "error_detail": (
                    f"{GET_MEASUREMENT_CONTEXT}_provider_returned_no_evidence"
                ),
            }

        no_evidence_context.provider_kind = "deployment"
        kwargs["context_providers"] = {
            **kwargs["context_providers"],
            GET_MEASUREMENT_CONTEXT: no_evidence_context,
        }
        env = TransactionalPSSEEnv(
            **kwargs, production_dataset_mode=True, max_steps=24
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(scenario)
        active_id = str(env.current_state()["active_state_id"])
        _, wls = env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        self.assertEqual(wls["execution_status"], "success")
        _, context = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )
        self.assertEqual(context["execution_status"], "failure")
        self.assertEqual(
            context["error_code"], "insufficient_observable_evidence"
        )
        self.assertEqual(
            context["error_detail"],
            f"{GET_MEASUREMENT_CONTEXT}_provider_returned_no_evidence",
        )

        executed: list[str] = []
        for _ in range(6):
            if env.is_terminal():
                break
            proposals = oracle.next_actions(
                env.get_oracle_state(env.history), env.history
            )
            self.assertTrue(
                proposals,
                f"expert returned no action after {executed}; deadlock",
            )
            action = proposals[0]
            executed.append(action["tool"])
            _, output = env.step(action)
            self.assertEqual(output["execution_status"], "success", executed)

        self.assertTrue(env.is_terminal(), executed)
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(executed[-1], "ask_for_more_evidence")


if __name__ == "__main__":
    unittest.main()
