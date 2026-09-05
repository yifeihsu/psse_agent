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


def _record_acquired_discovery_context(state: dict) -> None:
    """Start a routing fixture after WLS and explicit measurement requests."""
    active = state["active_state_id"]
    channels = [
        item for item in state.get("available_evidence", [])
        if item in {"three_phase_voltages", "three_phase_branch_currents"}
    ]
    contexts = state.setdefault("fresh_context_evidence", {})
    contexts["wls"] = {
        "state_id": active,
        "state_hash": "fixture-current-state",
        "evidence_source": "deployment_wls:fixture",
        "successful": True,
        "anomalous": bool(state.get("unresolved_signatures")),
    }
    contexts["harmonic"] = {
        "state_id": active,
        "state_hash": "fixture-current-state",
        "evidence_source": "deployment_context:harmonic_measurements",
        "request_attempted": True,
        "harmonic_context_status": "unavailable",
        "available_evidence_channels": [],
        "harmonic_distortion_detected": False,
    }
    contexts["three_phase"] = {
        "state_id": active,
        "state_hash": "fixture-current-state",
        "evidence_source": "deployment_context:three_phase_measurements",
        "request_attempted": True,
        "three_phase_context_status": "available" if channels else "unavailable",
        "available_evidence_channels": channels,
        "nlm_attempted": False,
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

    def test_flagged_legacy_root_can_request_an_unadvertised_channel(self) -> None:
        state = _policy_state(unresolved_signatures=["harmonic_distortion_detected"])
        self.assertEqual(self.expert.propose(state, [])[0].action["tool"], "get_harmonic_context")

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
        self.assertEqual(first[0].action["tool"], "run_wls")
        state["fresh_context_evidence"] = {
            "wls": {"state_id": state["active_state_id"], "successful": True}
        }
        request = self.expert.propose(state, [])
        self.assertEqual(request[0].action["tool"], "get_three_phase_context")
        _record_acquired_discovery_context(state)
        acquired = self.expert.propose(state, [])
        self.assertEqual(acquired[0].action["tool"], "run_three_phase_nlm_from_path")

        nlm_metrics = {
            "nlm_summary": {
                "top_hif_groups": [{"rank": 1, "branch_row0": 12, "score": 0.91}]
            }
        }
        state["fresh_context_evidence"]["three_phase"]["nlm_attempted"] = True
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
                state = _policy_state(
                    unresolved_signatures=[signature],
                )
                _record_acquired_discovery_context(state)
                actions = oracle.next_actions(state, [])
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
                "unresolved_signatures": [],
                "semantic_field_provenance": {
                    "unresolved_signatures": "controller_default"
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
        self.assertEqual(observation.available_evidence, [])

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
        # This fixture carries the clean positive-sequence snapshot with
        # synthetic spectra attached, so the WLS anomaly is narrow and the
        # phase-resolved request goes first; it returns nothing, the spectral
        # request follows, and the harmonic route completes.  Real harmonic
        # rows are broad and request spectra directly (see
        # test_harmonic_discovery).
        self.assertEqual(
            executed,
            [
                "run_wls",
                "get_three_phase_context",
                "get_harmonic_context",
                "run_hse_from_path",
                "finalize_diagnosis",
            ],
        )
        self.assertTrue(env.is_terminal())
        # The recorded explanation is model-visible and covers the signature.
        final_observation = env.get_policy_observation()
        self.assertTrue(final_observation.explained_anomalies)
        record = final_observation.explained_anomalies[0]
        self.assertEqual(record["family"], "harmonic")
        self.assertIn("harmonic distortion_detected_by_context", record["explained_signatures"])
        self.assertTrue(any(item.startswith("wls_") for item in record["explained_signatures"]))


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

    def test_hidden_truth_and_channel_cannot_bypass_wls_evidence_gate(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "successful current-state WLS"):
            self.env.assert_training_decision_evidence(
                {
                    "tool": "run_three_phase_nlm_from_path",
                    "arguments": {"state_id": state["active_state_id"]},
                }
            )

    def test_wls_anomaly_requires_phase_acquisition_before_screening_target(self) -> None:
        from three_phase_nlm.synthetic_branch_telemetry import synthetic_unbalance_rows

        voltages, currents = synthetic_unbalance_rows(source_bus=2, split=(0.5, 0.3, 0.2))
        anomalous = list(self.data["z_obs"])
        anomalous[5] += 5.0
        state = self.env.reset(
            self._scenario(
                measurements=anomalous,
                metadata={
                    "three_phase_voltages": voltages,
                    "three_phase_branch_currents": currents,
                    "branch_current_sigma_pu": 1e-3,
                },
            )
        )
        nlm_action = {
            "tool": "run_three_phase_nlm_from_path",
            "arguments": {"state_id": state["active_state_id"]},
        }
        # Before the baseline solve there is no observable anomaly to screen.
        with self.assertRaisesRegex(ValueError, "successful current-state WLS"):
            self.env.assert_training_decision_evidence(nlm_action)
        _, wls = self.env.step({"tool": "run_wls", "arguments": {"state_id": state["active_state_id"]}})
        self.assertEqual(wls["execution_status"], "success")
        self.assertNotIn("three_phase_voltages", self.env.get_policy_observation().available_evidence)
        with self.assertRaisesRegex(ValueError, "fresh acquired three-phase measurements"):
            self.env.assert_training_decision_evidence(nlm_action)
        for tool in ("get_harmonic_context", "get_three_phase_context"):
            action = {"tool": tool, "arguments": {"state_id": state["active_state_id"]}}
            self.env.assert_training_decision_evidence(action)
            _, output = self.env.step(action)
            self.assertEqual(output["execution_status"], "success", output)
        self.assertIn("three_phase_voltages", self.env.get_policy_observation().available_evidence)
        self.env.assert_training_decision_evidence(nlm_action)

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
        _record_acquired_discovery_context(state)
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


class ThreePhaseScreeningTests(unittest.TestCase):
    """Discovery: an unflagged WLS anomaly is screened against three-phase telemetry first."""

    CHANNELS = ["three_phase_voltages", "three_phase_branch_currents"]

    def setUp(self) -> None:
        self.expert = DiagnosticsExpert()

    def _anomalous_state(self, **overrides) -> dict:
        state = _policy_state(
            unresolved_signatures=["wls_residual_outlier_dominant index=5 channel=Vm"],
            available_evidence=list(self.CHANNELS),
            last_tool="run_wls",
            last_tool_status="success",
        )
        state.update(overrides)
        # These fixtures isolate NLM screening after completed same-state
        # WLS, harmonic acquisition, and three-phase acquisition.
        _record_acquired_discovery_context(state)
        return state

    def test_screening_proposes_nlm_on_wls_anomaly_with_telemetry(self) -> None:
        proposals = self.expert.three_phase_screening_proposals(self._anomalous_state(), [])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].action["tool"], "run_three_phase_nlm_from_path")
        self.assertIn("three_phase_screening_before_correction", proposals[0].evidence_codes)

    def test_screening_needs_telemetry_an_anomaly_and_no_flag(self) -> None:
        self.assertEqual(
            self.expert.three_phase_screening_proposals(
                self._anomalous_state(available_evidence=["nlm_diagnostic"]), []
            ),
            [],
        )
        self.assertEqual(
            self.expert.three_phase_screening_proposals(
                self._anomalous_state(unresolved_signatures=[]), []
            ),
            [],
        )
        # A flagged root is owned by the ordinary ladder, not by screening.
        self.assertEqual(
            self.expert.three_phase_screening_proposals(
                self._anomalous_state(
                    unresolved_signatures=["hif_suspected_zero_sequence", "wls_residual_outlier index=5"]
                ),
                [],
            ),
            [],
        )
        screened = self._anomalous_state()
        screened["fresh_context_evidence"]["three_phase"]["nlm_attempted"] = True
        self.assertEqual(
            self.expert.three_phase_screening_proposals(
                screened, [_successful_step("run_three_phase_nlm_from_path")]
            ),
            [],
        )

    def test_orchestrator_screens_before_any_correction_route(self) -> None:
        oracle = ExpertPolicyOracle()
        actions = oracle.next_actions(
            self._anomalous_state(), [_successful_step("run_wls")]
        )
        self.assertTrue(actions)
        self.assertEqual(actions[0]["tool"], "run_three_phase_nlm_from_path")

    def test_orchestrator_without_telemetry_keeps_the_classical_route(self) -> None:
        oracle = ExpertPolicyOracle()
        actions = oracle.next_actions(
            self._anomalous_state(available_evidence=[]), [_successful_step("run_wls")]
        )
        self.assertTrue(actions)
        self.assertNotEqual(actions[0]["tool"], "run_three_phase_nlm_from_path")

    def test_screening_outranks_the_recovery_retry_after_a_failed_correction(self) -> None:
        oracle = ExpertPolicyOracle()
        state = self._anomalous_state(
            has_fresh_measurement_context=True,
            last_tool="correct_measurements",
            last_tool_status="failure",
            last_tool_output={
                "execution_status": "failure",
                "error_code": "correction_not_supported_by_current_context",
                "error_detail": "measurement_target_not_in_context",
            },
        )
        actions = oracle.next_actions(
            state,
            [
                _successful_step("run_wls"),
                _successful_step("get_measurement_context"),
                _failed_step("correct_measurements", "correction_not_supported_by_current_context"),
            ],
        )
        self.assertTrue(actions)
        self.assertEqual(actions[0]["tool"], "run_three_phase_nlm_from_path")

    def test_gate_refuses_corrections_while_screening_is_pending(self) -> None:
        from psse_env.actions import three_phase_screening_pending
        from psse_env.oracle import ProcessValidityOracle

        gate = ProcessValidityOracle()
        state = self._anomalous_state(has_fresh_measurement_context=True)
        verdict = gate.check(
            state,
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "episode:s0", "measurement_updates": {5: 1.0}},
            },
        )
        self.assertFalse(verdict["process_valid"])
        self.assertEqual(verdict["error_code"], "correction_route_not_actionable")
        self.assertEqual(verdict["error_detail"], "measurement_three_phase_screening_pending")
        # Once the NLM check has been tried on this state, the rule lifts.
        tried = 'run_three_phase_nlm_from_path:{"state_id":"episode:s0"}'
        self.assertFalse(
            three_phase_screening_pending(
                unresolved=state["unresolved_signatures"],
                available_evidence=state["available_evidence"],
                tried_action_signatures=[tried],
                active_state_id="episode:s0",
            )
        )
        # A screening bound to an earlier active state does not count.
        self.assertTrue(
            three_phase_screening_pending(
                unresolved=state["unresolved_signatures"],
                available_evidence=state["available_evidence"],
                tried_action_signatures=['run_three_phase_nlm_from_path:{"state_id":"episode:s9"}'],
                active_state_id="episode:s0",
            )
        )


class WaveformRouteStandDownTests(unittest.TestCase):
    """A waveform-family signature keeps the fundamental-frequency routes shut.

    Observed in the 2026-09-03 diagnostic round: after a student's failed
    escalation on an explained unbalance root, the generic WLS recovery
    fallback became a teacher target, WLS then minted residual signatures from
    the still-unbalanced operator vector, and the classical route chased them
    into false commits.  Explanation closes the obligation, not the event.
    """

    UNBALANCE = "three_phase_unbalance phase_current_spread_detected"

    def _explained_unbalance_state(self, **overrides) -> dict:
        state = _policy_state(
            unresolved_signatures=[self.UNBALANCE],
            explained_anomalies=[
                {
                    "family": "three_phase_unbalance",
                    "explained_signatures": [self.UNBALANCE],
                    "evidence_source": "deployment_diagnostic:sequence_voltage_unbalance+branch_currents",
                }
            ],
            available_evidence=["three_phase_voltages", "three_phase_branch_currents"],
            remaining_anomaly_score=12.0,
            last_tool="ask_for_more_evidence",
            last_tool_status="failure",
            last_tool_output={
                "execution_status": "failure",
                "error_code": "recovery_evidence_inventory_incomplete",
            },
        )
        state.update(overrides)
        return state

    def test_waveform_signature_helper_uses_word_boundaries(self) -> None:
        from psse_env.actions import waveform_anomaly_signatures

        self.assertEqual(
            waveform_anomaly_signatures(
                [self.UNBALANCE, "hif_suspected_zero_sequence", "wls_residual_outlier", "search_marker"]
            ),
            [self.UNBALANCE, "hif_suspected_zero_sequence"],
        )
        self.assertEqual(waveform_anomaly_signatures([]), [])

    def test_recovery_expert_defers_on_waveform_roots_without_a_candidate(self) -> None:
        from psse_env.oracle.recovery_expert import RecoveryExpert

        expert = RecoveryExpert()
        self.assertEqual(expert.propose(self._explained_unbalance_state(), []), [])
        classical = _policy_state(
            unresolved_signatures=["wls_residual_outlier_dominant"],
            last_tool="ask_for_more_evidence",
            last_tool_status="failure",
            last_tool_output={
                "execution_status": "failure",
                "error_code": "recovery_evidence_inventory_incomplete",
            },
        )
        fallback = expert.propose(classical, [])
        self.assertTrue(fallback)
        self.assertEqual(fallback[0].action["tool"], "run_wls")

    def test_process_gate_refuses_corrections_under_a_waveform_signature(self) -> None:
        from psse_env.oracle import ProcessValidityOracle

        gate = ProcessValidityOracle()
        state = self._explained_unbalance_state(has_fresh_measurement_context=True)
        verdict = gate.check(
            state,
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "episode:s0", "measurement_updates": {8: 0.5}},
            },
        )
        self.assertFalse(verdict["process_valid"])
        self.assertEqual(verdict["error_code"], "correction_route_not_actionable")
        self.assertEqual(
            verdict["error_detail"],
            "measurement_fundamental_route_blocked_by_waveform_anomaly",
        )

    def test_orchestrator_finalizes_an_explained_root_after_a_failed_action(self) -> None:
        oracle = ExpertPolicyOracle()
        actions = oracle.next_actions(
            self._explained_unbalance_state(),
            [
                _successful_step("run_three_phase_nlm_from_path"),
                _failed_step("ask_for_more_evidence", "recovery_evidence_inventory_incomplete"),
            ],
        )
        self.assertTrue(actions)
        self.assertEqual(actions[0]["tool"], "finalize_diagnosis")

    def test_orchestrator_routes_unexplained_waveform_roots_to_diagnostics_after_failure(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=[self.UNBALANCE],
            available_evidence=["three_phase_voltages", "three_phase_branch_currents"],
            last_tool="get_measurement_context",
            last_tool_status="failure",
            last_tool_output={"execution_status": "failure", "error_code": "unknown_state_id"},
        )
        _record_acquired_discovery_context(state)
        actions = oracle.next_actions(
            state, [_failed_step("get_measurement_context", "unknown_state_id")]
        )
        self.assertTrue(actions)
        self.assertEqual(actions[0]["tool"], "run_three_phase_nlm_from_path")


class RequestOrderTests(unittest.TestCase):
    """The WLS residual breadth decides which measurement request goes first."""

    def _state(self, breadth, **overrides) -> dict:
        state = _policy_state(
            unresolved_signatures=["wls_residual_outlier_dominant index=5 channel=Qinj"],
            fresh_context_evidence={
                "wls": {
                    "state_id": "episode:s0",
                    "successful": True,
                    "anomalous": True,
                    "anomaly_breadth": breadth,
                }
            },
        )
        state.update(overrides)
        return state

    def test_breadth_selects_the_first_request(self) -> None:
        from psse_env.actions import preferred_first_request, wls_anomaly_breadth

        self.assertEqual(wls_anomaly_breadth(self._state(0.66)), 0.66)
        self.assertEqual(preferred_first_request(self._state(0.66)), "get_harmonic_context")
        self.assertEqual(preferred_first_request(self._state(0.07)), "get_three_phase_context")
        self.assertEqual(preferred_first_request(self._state(0.5)), "get_harmonic_context")
        # Without a breadth statistic the spectral request keeps precedence.
        self.assertEqual(preferred_first_request(self._state(None)), "get_harmonic_context")
        self.assertEqual(
            preferred_first_request(self._state(0.07, fresh_context_evidence={"wls": {"successful": False}})),
            "get_harmonic_context",
        )

    def test_breadth_is_read_from_the_last_wls_output_when_the_ledger_is_absent(self) -> None:
        from psse_env.actions import preferred_first_request

        state = _policy_state(
            unresolved_signatures=["wls_residual_outlier index=5 channel=Qinj"],
            last_tool="run_wls",
            last_tool_status="success",
            last_tool_output={"execution_status": "success", "tool_metrics": {"anomaly_breadth": 0.08}},
        )
        state.pop("fresh_context_evidence", None)
        self.assertEqual(preferred_first_request(state), "get_three_phase_context")

    def test_orchestrator_orders_the_requests_by_breadth(self) -> None:
        oracle = ExpertPolicyOracle()
        narrow = oracle.next_actions(self._state(0.07), [_successful_step("run_wls")])
        self.assertEqual(narrow[0]["tool"], "get_three_phase_context")
        broad = oracle.next_actions(self._state(0.66), [_successful_step("run_wls")])
        self.assertEqual(broad[0]["tool"], "get_harmonic_context")


if __name__ == "__main__":
    unittest.main()
