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

    def test_privileged_family_flag_ranks_but_needs_telemetry(self) -> None:
        state = _policy_state(available_evidence=["harmonic_measurements"])
        proposals = self.expert.propose(state, [], harmonic_fault_present=True)
        self.assertEqual(proposals[0].action["tool"], "get_harmonic_context")
        self.assertIn("privileged_harmonic_ranking", proposals[0].evidence_codes)
        without_channel = self.expert.propose(
            _policy_state(), [], harmonic_fault_present=True
        )
        self.assertEqual(without_channel, [])

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
    def test_harmonic_route_outranks_baseline_wls(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=["harmonic_distortion_detected"],
            available_evidence=["harmonic_measurements"],
        )
        actions = oracle.next_actions(state, [])
        self.assertEqual(actions[0]["tool"], "get_harmonic_context")

    def test_measurement_markers_still_route_to_measurement_context(self) -> None:
        oracle = ExpertPolicyOracle()
        state = _policy_state(
            unresolved_signatures=["large_residual meter_31"],
            available_evidence=["harmonic_measurements"],
        )
        actions = oracle.next_actions(state, [])
        self.assertEqual(actions[0]["tool"], "get_measurement_context")

    def test_hif_route_via_hidden_family_and_telemetry(self) -> None:
        oracle = ExpertPolicyOracle()
        state = {
            "policy_observation": _policy_state(
                available_evidence=["nlm_diagnostic", "hif_scan_window"]
            ),
            "hidden_truth": {"true_hif_errors": [{"branch_row0": 12}]},
            "oracle_action_hints": [],
        }
        actions = oracle.next_actions(state, [])
        self.assertEqual(actions[0]["tool"], "run_three_phase_nlm_from_path")


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


if __name__ == "__main__":
    unittest.main()
