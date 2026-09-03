from __future__ import annotations

import json
import unittest
from pathlib import Path

from psse_env.oracle import DiagnosticsExpert, ExpertPolicyOracle

FIXTURE = Path(__file__).parent.parent / "providers" / "fixtures" / "case14_z.json"


def _policy_state(**overrides) -> dict:
    state = {
        "active_state_id": "episode:s0",
        "candidate_state_id": None,
        "episode_id": "episode",
        "remaining_budget": 8,
        "history_window": [],
        "unresolved_signatures": [],
        "tried_action_signatures": [],
        "accepted_corrections": [],
        "rejected_hypotheses": [],
        "available_evidence": [],
        "remaining_anomaly_score": 3.5,
        "no_material_anomaly_remaining": False,
        "last_tool": "run_wls",
        "last_tool_status": "success",
        "last_tool_output": {},
    }
    state.update(overrides)
    return state


def _successful_step(tool: str, metrics: dict | None = None) -> dict:
    return {
        "action": {"tool": tool, "arguments": {"state_id": "episode:s0"}},
        "tool_output": {"execution_status": "success", "tool_metrics": metrics or {}},
    }


def _failed_step(tool: str, error_code: str) -> dict:
    return {
        "action": {"tool": tool, "arguments": {"state_id": "episode:s0"}},
        "tool_output": {
            "execution_status": "failure",
            "error_code": error_code,
            "tool_metrics": {},
        },
    }


class DiagnosticsExpertRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expert = DiagnosticsExpert()

    def test_no_route_without_telemetry_channel(self) -> None:
        state = _policy_state(unresolved_signatures=["harmonic_distortion_detected"])
        self.assertEqual(self.expert.propose(state, []), [])

    def test_no_route_without_observable_or_privileged_signal(self) -> None:
        state = _policy_state(available_evidence=["harmonic_measurements"])
        self.assertEqual(self.expert.propose(state, []), [])

    def test_harmonic_ladder_context_then_hse(self) -> None:
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            available_evidence=["harmonic_measurements"],
        )
        first = self.expert.propose(state, [])
        self.assertEqual(first[0].action["tool"], "get_harmonic_context")
        follow_up = self.expert.propose(
            state, [_successful_step("get_harmonic_context", {"harmonic_orders": [5]})]
        )
        self.assertEqual(follow_up[0].action["tool"], "run_hse_from_path")
        done = self.expert.propose(
            state,
            [
                _successful_step("get_harmonic_context"),
                _successful_step("run_hse_from_path", {"best_candidate_bus_1based": 14}),
            ],
        )
        self.assertEqual([p.action["tool"] for p in done], [])

    def test_privileged_flags_and_hints_cannot_create_a_route(self) -> None:
        state = _policy_state(
            available_evidence=["harmonic_measurements", "nlm_diagnostic"]
        )
        proposals = self.expert.propose(
            state,
            [],
            oracle_hints=[
                {
                    "tool": "run_three_phase_nlm_from_path",
                    "arguments": {"state_id": "episode:s0"},
                }
            ],
            harmonic_fault_present=True,
            hif_fault_present=True,
        )
        self.assertEqual(proposals, [])

    def test_privileged_flags_do_not_change_an_observable_route(self) -> None:
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            available_evidence=["harmonic_measurements"],
        )
        baseline = self.expert.propose(state, [])
        privileged = self.expert.propose(
            state, [], harmonic_fault_present=True, hif_fault_present=True
        )
        self.assertEqual(
            [proposal.action for proposal in privileged],
            [proposal.action for proposal in baseline],
        )

    def test_hif_ladder_nlm_then_estimator(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic"],
        )
        first = self.expert.propose(state, [])
        self.assertEqual(first[0].action["tool"], "run_three_phase_nlm_from_path")

        nlm_metrics = {
            "nlm_summary": {"top_hif_groups": [{"rank": 1, "branch_row0": 12, "score": 0.91}]}
        }
        history = [_successful_step("run_three_phase_nlm_from_path", nlm_metrics)]
        single = self.expert.propose(state, history)
        self.assertEqual(
            single[0].action["tool"], "estimate_hif_location_magnitude_from_path"
        )
        self.assertEqual(single[0].action["arguments"]["candidate_branch_row0"], 12)

        with_window = self.expert.propose(
            _policy_state(
                unresolved_signatures=["hif_suspected_zero_sequence"],
                available_evidence=["nlm_diagnostic", "hif_scan_window"],
            ),
            history,
        )
        self.assertEqual(
            with_window[0].action["tool"],
            "estimate_hif_location_magnitude_multiscan_from_path",
        )
        self.assertEqual(with_window[0].action["arguments"]["candidate_branch_row0"], 12)

    def test_rejected_hif_ladder_requests_explicit_operator_handoff(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic"],
        )
        nlm_metrics = {
            "nlm_summary": {
                "top_hif_groups": [{"rank": 1, "branch_row0": 12, "score": 0.91}]
            }
        }
        rejected = {"diagnostic_acceptance": {"accepted": False}}
        proposals = self.expert.propose(
            state,
            [
                _successful_step("run_three_phase_nlm_from_path", nlm_metrics),
                _successful_step(
                    "estimate_hif_location_magnitude_from_path", rejected
                ),
            ],
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].action["tool"], "ask_for_more_evidence")
        self.assertEqual(
            proposals[0].action["arguments"]["request"],
            "operator_escalation:hif_diagnostics_exhausted",
        )

    def test_multiscan_hif_handoff_waits_for_single_scan_rejection(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic", "hif_scan_window"],
        )
        nlm_metrics = {
            "nlm_summary": {
                "top_hif_groups": [{"rank": 1, "branch_row0": 12, "score": 0.91}]
            }
        }
        rejected = {"diagnostic_acceptance": {"accepted": False}}
        history = [
            _successful_step("run_three_phase_nlm_from_path", nlm_metrics),
            _successful_step(
                "estimate_hif_location_magnitude_multiscan_from_path", rejected
            ),
        ]
        follow_up = self.expert.propose(state, history)
        self.assertEqual(
            follow_up[0].action["tool"],
            "estimate_hif_location_magnitude_from_path",
        )

        history.append(
            _successful_step("estimate_hif_location_magnitude_from_path", rejected)
        )
        handoff = self.expert.propose(state, history)
        self.assertEqual(handoff[0].action["tool"], "ask_for_more_evidence")

    def test_failed_hif_estimators_fall_back_without_false_handoff(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic", "hif_scan_window"],
        )
        history = [
            _successful_step(
                "run_three_phase_nlm_from_path",
                {
                    "nlm_summary": {
                        "top_hif_groups": [{"rank": 1, "branch_row0": 12}]
                    }
                },
            ),
            _failed_step(
                "estimate_hif_location_magnitude_multiscan_from_path",
                "hif_multiscan_failure",
            ),
        ]
        follow_up = self.expert.propose(state, history)
        self.assertEqual(
            follow_up[0].action["tool"],
            "estimate_hif_location_magnitude_from_path",
        )

        history.append(
            _failed_step(
                "estimate_hif_location_magnitude_from_path",
                "hif_estimation_failure",
            )
        )
        # Two solver failures are an infrastructure defect.  They do not
        # satisfy the environment's audited handoff contract, which requires
        # successful, state-bound estimates with explicit rejected fits.
        self.assertEqual(self.expert.propose(state, history), [])

    def test_summarized_failed_multiscan_history_uses_the_same_safe_fallback(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic", "hif_scan_window"],
        )
        history = [
            {
                "tool": "run_three_phase_nlm_from_path",
                "arguments": {"state_id": "episode:s0"},
                "outcome": {"execution_status": "success"},
                "observable_metrics": {
                    "nlm_summary": {
                        "top_hif_groups": [{"rank": 1, "branch_row0": 12}]
                    }
                },
            },
            {
                "tool": "estimate_hif_location_magnitude_multiscan_from_path",
                "arguments": {
                    "state_id": "episode:s0",
                    "candidate_branch_row0": 12,
                },
                "outcome": {
                    "execution_status": "failure",
                    "error_code": "hif_multiscan_failure",
                },
                "observable_metrics": {},
            },
        ]

        follow_up = self.expert.propose(state, history)

        self.assertEqual(
            follow_up[0].action["tool"],
            "estimate_hif_location_magnitude_from_path",
        )
        history.append(
            {
                "tool": "estimate_hif_location_magnitude_from_path",
                "arguments": {
                    "state_id": "episode:s0",
                    "candidate_branch_row0": 12,
                },
                "outcome": {
                    "execution_status": "failure",
                    "error_code": "hif_estimation_failure",
                },
                "observable_metrics": {},
            }
        )
        self.assertEqual(self.expert.propose(state, history), [])

    def test_accepted_hif_estimator_never_requests_operator_handoff(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic"],
        )
        history = [
            _successful_step(
                "run_three_phase_nlm_from_path",
                {
                    "nlm_summary": {
                        "top_hif_groups": [{"branch_row0": 12, "score": 0.91}]
                    }
                },
            ),
            _successful_step(
                "estimate_hif_location_magnitude_from_path",
                {"diagnostic_acceptance": {"accepted": True}},
            ),
        ]
        self.assertEqual(self.expert.propose(state, history), [])

    def test_unbalance_runs_nlm_but_never_escalates_to_hif_estimation(self) -> None:
        state = _policy_state(
            unresolved_signatures=["three_phase_unbalance vuf=0.05"],
            available_evidence=["three_phase_voltages"],
        )
        first = self.expert.propose(state, [])
        self.assertEqual(first[0].action["tool"], "run_three_phase_nlm_from_path")

        nlm_metrics = {
            "nlm_summary": {
                "top_hif_groups": [{"rank": 1, "branch_row0": 12, "score": 0.91}]
            }
        }
        done = self.expert.propose(
            state, [_successful_step("run_three_phase_nlm_from_path", nlm_metrics)]
        )
        self.assertEqual(done, [])

    def test_summarized_history_window_shape_is_understood(self) -> None:
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            available_evidence=["harmonic_measurements"],
        )
        window_event = {
            "tool": "get_harmonic_context",
            "arguments": {"state_id": "active"},
            "outcome": {"execution_status": "success"},
            "observable_metrics": {"harmonic_orders": [5]},
        }
        follow_up = self.expert.propose(state, [window_event])
        self.assertEqual(follow_up[0].action["tool"], "run_hse_from_path")


class OrchestratorRoutingTests(unittest.TestCase):
    def test_harmonic_route_suppresses_redundant_baseline_wls(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            available_evidence=["harmonic_measurements"],
        )
        actions = oracle.next_actions(state, [])
        self.assertEqual(
            [action["tool"] for action in actions], ["get_harmonic_context"]
        )

    def test_hif_route_suppresses_redundant_baseline_wls(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic", "hif_scan_window"],
        )
        actions = oracle.next_actions(state, [])
        self.assertEqual(
            [action["tool"] for action in actions],
            ["run_three_phase_nlm_from_path"],
        )

    def test_specific_context_routes_suppress_redundant_baseline_wls(self) -> None:
        oracle = ExpertPolicyOracle()
        cases = {
            "parameter": (
                "wls_branch_multiplier_dominant line_status_or_parameter line=3",
                {"get_parameter_context", "get_topology_context"},
            ),
            "topology": (
                "breaker_status_mismatch line=3",
                {"get_topology_context"},
            ),
        }
        for name, (signature, expected_tools) in cases.items():
            with self.subTest(name=name):
                actions = oracle.next_actions(
                    _policy_state(unresolved_signatures=[signature]), []
                )
                tools = {action["tool"] for action in actions}
                self.assertEqual(tools, expected_tools)
                self.assertNotIn("run_wls", tools)

    def test_wls_remains_the_only_fallback_without_a_domain_proposal(self) -> None:
        actions = ExpertPolicyOracle().next_actions(_policy_state(), [])
        self.assertEqual([action["tool"] for action in actions], ["run_wls"])

    def test_mandatory_initial_baseline_wls_is_not_suppressed(self) -> None:
        state = _policy_state(
            last_tool=None,
            remaining_anomaly_score=None,
            available_evidence=["harmonic_measurements"],
        )
        actions = ExpertPolicyOracle().next_actions(state, [])
        self.assertEqual([action["tool"] for action in actions], ["run_wls"])

    def test_measurement_markers_still_route_to_measurement_context(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=["large_residual meter_31"],
            available_evidence=["harmonic_measurements"],
        )
        actions = oracle.next_actions(state, [])
        self.assertEqual(actions[0]["tool"], "get_measurement_context")

    def test_hidden_hif_truth_does_not_change_the_teacher_label(self) -> None:
        oracle = ExpertPolicyOracle()
        observation = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic", "hif_scan_window"],
        )
        without_truth = {
            "policy_observation": observation,
            "hidden_truth": {},
            "oracle_action_hints": [],
        }
        with_truth = {
            "policy_observation": observation,
            "hidden_truth": {"true_hif_errors": [{"branch_row0": 12}]},
            "oracle_action_hints": [],
        }
        baseline_actions = oracle.next_actions(without_truth, [])
        privileged_actions = oracle.next_actions(with_truth, [])
        self.assertEqual(privileged_actions, baseline_actions)
        self.assertEqual(
            baseline_actions[0]["tool"], "run_three_phase_nlm_from_path"
        )

    def test_hidden_hif_truth_cannot_create_a_teacher_label(self) -> None:
        oracle = ExpertPolicyOracle()
        observation = _policy_state(available_evidence=["nlm_diagnostic"])
        baseline = oracle.next_actions(
            {"policy_observation": observation, "hidden_truth": {}}, []
        )
        privileged = oracle.next_actions(
            {
                "policy_observation": observation,
                "hidden_truth": {"true_hif_errors": [{"branch_row0": 12}]},
            },
            [],
        )
        self.assertEqual(privileged, baseline)
        self.assertNotEqual(baseline[0]["tool"], "run_three_phase_nlm_from_path")


class ResolutionSemanticsTests(unittest.TestCase):
    @staticmethod
    def _explained_record(*signatures: str, family: str = "harmonic") -> dict:
        return {
            "tool": "run_hse_from_path",
            "family": family,
            "kind": "harmonic_source_localized",
            "evidence_source": "deployment_diagnostic:harmonic_state_estimation",
            "explained_signatures": list(signatures),
        }

    def test_unexplained_signatures_helper(self) -> None:
        from psse_env.actions import unexplained_signatures

        signatures = ["harmonic_distortion_detected", "large_residual meter_31"]
        records = [self._explained_record("harmonic_distortion_detected")]
        self.assertEqual(
            unexplained_signatures(signatures, records), ["large_residual meter_31"]
        )
        self.assertEqual(unexplained_signatures(signatures, []), signatures)
        self.assertEqual(unexplained_signatures([], records), [])

    def test_finalize_becomes_legal_once_all_signatures_are_explained(self) -> None:
        from psse_env.oracle import ProcessValidityOracle

        gate = ProcessValidityOracle()
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            explained_anomalies=[self._explained_record("harmonic_distortion_detected")],
        )
        check = gate.check(state, {"tool": "finalize_diagnosis", "arguments": {}})
        self.assertTrue(check["process_valid"])

    def test_finalize_stays_blocked_while_a_signature_is_unexplained(self) -> None:
        from psse_env.oracle import ProcessValidityOracle

        gate = ProcessValidityOracle()
        state = _policy_state(
            unresolved_signatures=[
                "harmonic_distortion_detected",
                "large_residual meter_31",
            ],
            explained_anomalies=[self._explained_record("harmonic_distortion_detected")],
        )
        check = gate.check(state, {"tool": "finalize_diagnosis", "arguments": {}})
        self.assertFalse(check["process_valid"])
        self.assertEqual(check["error_code"], "terminal_condition_not_met")

    def test_termination_expert_proposes_finalize_from_explanations(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            available_evidence=["harmonic_measurements"],
            explained_anomalies=[self._explained_record("harmonic_distortion_detected")],
        )
        actions = oracle.next_actions(state, [])
        self.assertEqual(actions[0]["tool"], "finalize_diagnosis")


class EndToEndHarmonicRoutingTests(unittest.TestCase):
    def test_expert_drives_harmonic_investigation_in_real_environment(self) -> None:
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.transactional_env import TransactionalPSSEEnv

        data = json.loads(FIXTURE.read_text())
        providers = MatpowerDeploymentProviders()
        env = TransactionalPSSEEnv(**providers.env_kwargs(), production_dataset_mode=True)
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(
            {
                "scenario_id": "harmonic_route",
                "case": data["case_path"],
                "measurements": list(data["z_obs"]),
                "unresolved_signatures": ["harmonic_distortion_detected"],
                "semantic_field_provenance": {
                    "unresolved_signatures": "deployment_sensor:power_quality"
                },
                "metadata": {
                    "harmonic_measurements": [
                        {
                            "h": 5,
                            "bus": bus,
                            "Vm": 0.02 + 0.001 * bus,
                            "Va_deg": 10.0 * bus,
                            "sigma": 1e-4,
                        }
                        for bus in range(1, 15)
                    ]
                },
            }
        )
        observation = env.get_policy_observation()
        self.assertIn("harmonic_measurements", observation.available_evidence)

        executed: list[str] = []
        for _ in range(5):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(), env.history)
            self.assertTrue(actions, f"expert returned no action after {executed}")
            env.assert_training_decision_evidence(actions[0])
            _, output = env.step(actions[0])
            self.assertEqual(
                output["execution_status"], "success", f"{actions[0]} -> {output}"
            )
            executed.append(actions[0]["tool"])
        self.assertEqual(
            executed,
            ["get_harmonic_context", "run_hse_from_path", "finalize_diagnosis"],
        )
        self.assertTrue(env.is_terminal())
        # The recorded explanation is model-visible and covers the signature.
        final_observation = env.get_policy_observation()
        self.assertTrue(final_observation.explained_anomalies)
        record = final_observation.explained_anomalies[0]
        self.assertEqual(record["family"], "harmonic")
        self.assertEqual(
            record["explained_signatures"], ["harmonic_distortion_detected"]
        )


class ProductionDiagnosticEvidenceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.transactional_env import TransactionalPSSEEnv

        self.data = json.loads(FIXTURE.read_text())
        providers = MatpowerDeploymentProviders()
        self.env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True
        )

    def _scenario(self, **overrides) -> dict:
        scenario = {
            "scenario_id": "diagnostic_gate",
            "case": self.data["case_path"],
            "measurements": list(self.data["z_obs"]),
            "metadata": {},
        }
        scenario.update(overrides)
        return scenario

    def test_hidden_truth_and_channel_cannot_bypass_signature_gate(self) -> None:
        state = self.env.reset(
            self._scenario(
                metadata={
                    "nlm_diagnostic": {
                        "success": True,
                        "method": "observable_test",
                        "top_hif_groups": [{"branch_row0": 12}],
                    }
                },
                hidden_truth={"true_hif_errors": [{"branch_row0": 12}]},
            )
        )
        with self.assertRaisesRegex(ValueError, "observable .*signature"):
            self.env.assert_training_decision_evidence(
                {
                    "tool": "run_three_phase_nlm_from_path",
                    "arguments": {"state_id": state["active_state_id"]},
                }
            )

    def test_hif_estimator_target_must_come_from_latest_nlm_output(self) -> None:
        state = self.env.reset(
            self._scenario(
                unresolved_signatures=["hif_suspected_zero_sequence"],
                semantic_field_provenance={
                    "unresolved_signatures": "deployment_sensor:waveform_capture"
                },
                metadata={
                    "nlm_diagnostic": {
                        "success": True,
                        "converged": True,
                        "method": "observable_test",
                        "top_hif_groups": [
                            {"rank": 1, "branch_row0": 12, "score": 0.9}
                        ],
                    }
                },
            )
        )
        nlm_action = {
            "tool": "run_three_phase_nlm_from_path",
            "arguments": {"state_id": state["active_state_id"]},
        }
        self.env.assert_training_decision_evidence(nlm_action)
        _, output = self.env.step(nlm_action)
        self.assertEqual(output["execution_status"], "success")

        unsupported = {
            "tool": "estimate_hif_location_magnitude_from_path",
            "arguments": {
                "state_id": state["active_state_id"],
                "candidate_branch_row0": 11,
            },
        }
        with self.assertRaisesRegex(ValueError, "not supported by the latest"):
            self.env.assert_training_decision_evidence(unsupported)

        supported = {
            "tool": "estimate_hif_location_magnitude_from_path",
            "arguments": {
                "state_id": state["active_state_id"],
                "candidate_branch_row0": 12,
            },
        }
        self.env.assert_training_decision_evidence(supported)


class TerminalCurrentRoutingTests(unittest.TestCase):
    """Per-phase branch-current telemetry as an observable NLM channel."""

    def setUp(self) -> None:
        self.expert = DiagnosticsExpert()

    def test_branch_currents_alone_enable_nlm_for_hif_signature(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["three_phase_voltages", "three_phase_branch_currents"],
        )
        proposals = self.expert.propose(state, [])
        self.assertEqual(proposals[0].action["tool"], "run_three_phase_nlm_from_path")
        self.assertIn(
            "three_phase_branch_current_telemetry_available", proposals[0].evidence_codes
        )

    def test_branch_currents_enable_nlm_for_unbalance_signature(self) -> None:
        state = _policy_state(
            unresolved_signatures=["three_phase_unbalance vuf_threshold_exceeded"],
            available_evidence=["three_phase_branch_currents"],
        )
        proposals = self.expert.propose(state, [])
        self.assertEqual(proposals[0].action["tool"], "run_three_phase_nlm_from_path")

    def test_suspected_phase_from_nlm_is_forwarded_to_estimator(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["three_phase_voltages", "three_phase_branch_currents"],
        )
        history = [
            _successful_step(
                "run_three_phase_nlm_from_path",
                {
                    "nlm_summary": {
                        "method": "terminal_current_differential",
                        "top_hif_groups": [{"branch_row0": 2}],
                        "suspected_phase": "B",
                    }
                },
            )
        ]
        proposals = self.expert.propose(state, history)
        estimator = proposals[0]
        self.assertEqual(estimator.action["tool"], "estimate_hif_location_magnitude_from_path")
        self.assertEqual(estimator.action["arguments"]["candidate_branch_row0"], 2)
        self.assertEqual(estimator.action["arguments"]["candidate_phase"], "B")

    def test_no_phase_is_forwarded_without_observable_phase_evidence(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["nlm_diagnostic"],
        )
        history = [
            _successful_step(
                "run_three_phase_nlm_from_path",
                {"nlm_summary": {"top_hif_groups": [{"branch_row0": 2}]}},
            )
        ]
        proposals = self.expert.propose(state, history)
        self.assertEqual(proposals[0].action["tool"], "estimate_hif_location_magnitude_from_path")
        self.assertNotIn("candidate_phase", proposals[0].action["arguments"])

    def test_invalid_phase_token_is_ignored(self) -> None:
        state = _policy_state(
            unresolved_signatures=["hif_suspected_zero_sequence"],
            available_evidence=["three_phase_branch_currents"],
        )
        history = [
            _successful_step(
                "run_three_phase_nlm_from_path",
                {"nlm_summary": {"top_hif_groups": [{"branch_row0": 2}], "suspected_phase": "N"}},
            )
        ]
        proposals = self.expert.propose(state, history)
        self.assertNotIn("candidate_phase", proposals[0].action["arguments"])


if __name__ == "__main__":
    unittest.main()
