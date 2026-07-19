from __future__ import annotations

import json
import unittest
from pathlib import Path

from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import (
    BASE_FAMILIES,
    COMPOSED_FAMILIES,
    Round0ScenarioGenerator,
    ScenarioRejected,
)

FIXTURE = Path(__file__).parent / "fixtures" / "case14_z.json"

_TRUTH_KEY_PREFIXES = ("true_", "clean_", "hidden_")
_FAMILY_TOKENS = (
    "measurement",
    "parameter",
    "topology",
    "harmonic",
    "hif",
    "no_error",
    "meas",
    "topo",
    "param",
)


def _quick_plan() -> dict[str, int]:
    return {
        "no_error": 1,
        "measurement": 1,
        "multi_measurement": 1,
        "parameter": 1,
        "topology": 1,
        "harmonic": 1,
        "hif": 1,
        "measurement+parameter": 1,
        "measurement+topology": 1,
        "measurement+hif": 1,
    }


class ScenarioConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Round0ScenarioGenerator(seed=20260719)
        cls.scenarios = cls.generator.build(_quick_plan())
        cls.by_family = {
            scenario["scenario_family"]: scenario for scenario in cls.scenarios
        }

    def test_every_requested_family_is_built(self) -> None:
        self.assertEqual(
            sorted(self.by_family),
            sorted(BASE_FAMILIES + COMPOSED_FAMILIES),
        )

    def test_scenario_ids_are_opaque(self) -> None:
        for scenario in self.scenarios:
            identifier = scenario["scenario_id"].lower()
            self.assertTrue(identifier.startswith("r0_"), identifier)
            for token in _FAMILY_TOKENS:
                self.assertNotIn(token, identifier)

    def test_policy_visible_metadata_carries_no_truth(self) -> None:
        for scenario in self.scenarios:
            for key in scenario["metadata"]:
                for prefix in _TRUTH_KEY_PREFIXES:
                    self.assertFalse(
                        str(key).startswith(prefix),
                        f"{scenario['scenario_family']}: metadata leaks {key}",
                    )
            self.assertNotIn("scenario_family", scenario["metadata"])

    def test_measurement_truth_matches_vector(self) -> None:
        scenario = self.by_family["measurement"]
        fault = scenario["true_measurement_errors"][0]
        index = int(fault["index"])
        self.assertEqual(fault["observed"], scenario["measurements"][index])
        self.assertEqual(fault["clean"], scenario["clean_measurements"][index])
        self.assertNotEqual(fault["observed"], fault["clean"])

    def test_multi_measurement_carries_multiple_indices(self) -> None:
        scenario = self.by_family["multi_measurement"]
        indices = {fault["index"] for fault in scenario["true_measurement_errors"]}
        self.assertGreater(len(indices), 1)

    def test_parameter_scenario_carries_scans_and_stale_model(self) -> None:
        scenario = self.by_family["parameter"]
        self.assertEqual(scenario["case"], "case14")
        self.assertNotEqual(scenario["clean_case"], "case14")
        scans = scenario["metadata"]["parameter_scans"]
        self.assertTrue(scans["z_scans"] and scans["initial_states"])
        fault = scenario["true_parameter_errors"][0]
        self.assertIn("clean_r", fault)
        self.assertIn("clean_x", fault)

    def test_topology_scenario_is_synthesized_with_derived_clean_case(self) -> None:
        scenario = self.by_family["topology"]
        fault = scenario["true_topology_errors"][0]
        self.assertEqual(scenario["case"], "case14")
        self.assertTrue(Path(scenario["clean_case"]).is_file())
        self.assertEqual(int(fault["expected_status"]), 0)
        self.assertEqual(int(fault["line_index1"]), int(fault["branch_row0"]) + 1)

    def test_harmonic_scenario_routes_by_sensor_signature(self) -> None:
        scenario = self.by_family["harmonic"]
        self.assertEqual(
            scenario["unresolved_signatures"], ["harmonic_distortion_detected"]
        )
        self.assertTrue(
            scenario["semantic_field_provenance"]["unresolved_signatures"].startswith(
                "deployment_sensor"
            )
        )
        self.assertTrue(scenario["metadata"]["harmonic_measurements"])
        self.assertTrue(scenario["hidden_truth"]["true_harmonic_errors"])

    def test_hif_scenario_carries_all_diagnostic_channels(self) -> None:
        scenario = self.by_family["hif"]
        metadata = scenario["metadata"]
        self.assertTrue(metadata["nlm_diagnostic"].get("success"))
        self.assertTrue(metadata["hif_runtime"]["three_phase_voltages"])
        scans = metadata["hif_scan_window"]["scans"]
        self.assertTrue(scans)
        self.assertLessEqual(len(scans), self.generator.hif_max_scans)
        self.assertEqual(
            scenario["unresolved_signatures"], ["hif_suspected_zero_sequence"]
        )
        self.assertTrue(scenario["hidden_truth"]["true_hif_errors"])

    def test_composition_overlays_extra_measurement_faults(self) -> None:
        scenario = self.by_family["measurement+parameter"]
        self.assertTrue(scenario["true_parameter_errors"])
        overlays = scenario["true_measurement_errors"]
        self.assertTrue(overlays)
        for fault in overlays:
            index = int(fault["index"])
            self.assertEqual(fault["observed"], scenario["measurements"][index])
            self.assertNotEqual(fault["observed"], fault["clean"])

    def test_composition_never_overlays_the_faulted_line_flows(self) -> None:
        scenario = self.by_family["measurement+topology"]
        row0 = int(scenario["true_topology_errors"][0]["branch_row0"])
        blocked = {3 * 14 + block * 20 + row0 for block in range(4)}
        overlay_indices = {
            int(fault["index"]) for fault in scenario["true_measurement_errors"]
        }
        self.assertFalse(overlay_indices & blocked)

    def test_report_counts_and_manifest_align(self) -> None:
        report = self.generator.report()
        self.assertEqual(sum(report["built_by_family"].values()), len(self.scenarios))
        manifest_ids = {entry["scenario_id"] for entry in self.generator.manifest}
        self.assertEqual(
            manifest_ids, {scenario["scenario_id"] for scenario in self.scenarios}
        )


class DeterminismTests(unittest.TestCase):
    def test_same_seed_reproduces_scenarios(self) -> None:
        plan = {"measurement": 2, "topology": 1}
        first = Round0ScenarioGenerator(seed=99).build(plan)
        second = Round0ScenarioGenerator(seed=99).build(plan)
        self.assertEqual(
            [scenario["scenario_id"] for scenario in first],
            [scenario["scenario_id"] for scenario in second],
        )
        self.assertEqual(
            [scenario["measurements"] for scenario in first],
            [scenario["measurements"] for scenario in second],
        )

    def test_different_seed_changes_selection(self) -> None:
        plan = {"measurement": 2}
        first = Round0ScenarioGenerator(seed=99).build(plan)
        second = Round0ScenarioGenerator(seed=100).build(plan)
        self.assertNotEqual(
            [scenario["scenario_id"] for scenario in first],
            [scenario["scenario_id"] for scenario in second],
        )


class ValidationGateTests(unittest.TestCase):
    def test_clean_vector_is_rejected_as_undetectable_anomaly(self) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        clean = next(
            iter(generator._corpus()["no_error"])
        )  # physically clean snapshot
        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_anomalous("case14", clean["z_obs"], "measurement")
        self.assertEqual(caught.exception.reason, "anomaly_not_detectable")

    def test_anomalous_vector_is_rejected_as_uncorrected(self) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        data = json.loads(FIXTURE.read_text())
        z = list(data["z_obs"])
        z[5] += 5.0
        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_clean("case14", z, "no_error")
        self.assertEqual(
            caught.exception.reason, "corrected_configuration_still_anomalous"
        )


class WlsSignatureEmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        data = json.loads(FIXTURE.read_text())
        cls.clean_z = list(data["z_obs"])

    def _state(self, z, observation=None) -> dict:
        state = {"state_id": "episode:s0", "case": "case14", "measurements": z}
        if observation is not None:
            state["policy_observation"] = observation
        return state

    def test_gross_error_emits_dominant_residual_signatures(self) -> None:
        z = list(self.clean_z)
        z[5] += 5.0
        metrics = self.providers.run_wls(self._state(z))
        signatures = metrics["unresolved_signatures"]
        self.assertTrue(
            any(sig.startswith("wls_residual_outlier_dominant") for sig in signatures),
            signatures,
        )

    def test_sensor_signatures_are_preserved_and_wls_ones_refreshed(self) -> None:
        z = list(self.clean_z)
        z[5] += 5.0
        observation = {
            "unresolved_signatures": [
                "harmonic_distortion_detected",
                "wls_residual_outlier index=99 channel=Qt",
            ],
            "explained_anomalies": [
                {
                    "family": "harmonic",
                    "explained_signatures": ["harmonic_distortion_detected"],
                }
            ],
        }
        metrics = self.providers.run_wls(self._state(z, observation))
        signatures = metrics["unresolved_signatures"]
        self.assertIn("harmonic_distortion_detected", signatures)
        self.assertNotIn("wls_residual_outlier index=99 channel=Qt", signatures)
        self.assertTrue(any("index=5" in sig for sig in signatures))

    def test_unexplained_waveform_signature_suppresses_wls_signatures(self) -> None:
        z = list(self.clean_z)
        z[5] += 5.0
        observation = {
            "unresolved_signatures": ["hif_suspected_zero_sequence"],
            "explained_anomalies": [],
        }
        metrics = self.providers.run_wls(self._state(z, observation))
        self.assertEqual(
            metrics["unresolved_signatures"], ["hif_suspected_zero_sequence"]
        )

    def test_clean_solve_emits_no_wls_signatures(self) -> None:
        # The shared fixture vector carries embedded corpus errors, so a truly
        # clean vector comes from a validated no_error corpus row instead.
        generator = Round0ScenarioGenerator(seed=20260719)
        clean_row = generator._corpus()["no_error"][0]
        metrics = self.providers.run_wls(self._state(list(clean_row["z_obs"])))
        self.assertFalse(
            [s for s in metrics["unresolved_signatures"] if s.startswith("wls_")]
        )


class MeasurementRouteDisciplineTests(unittest.TestCase):
    @staticmethod
    def _policy_state(signatures) -> dict:
        return {
            "active_state_id": "episode:s0",
            "candidate_state_id": None,
            "unresolved_signatures": signatures,
            "requires_measurement_context": False,
            "has_fresh_measurement_context": False,
            "measurement_context_state_id": None,
        }

    def test_measurement_route_stands_down_under_dominant_branch_evidence(self) -> None:
        from psse_env.oracle.measurement_expert import MeasurementExpert

        state = self._policy_state(
            [
                "wls_residual_outlier index=12 channel=Pinj",
                "wls_branch_multiplier_dominant line_status_or_parameter line=7",
            ]
        )
        proposals = MeasurementExpert().propose(state, [])
        tools = [proposal.action["tool"] for proposal in proposals]
        self.assertNotIn("get_measurement_context", tools)

    def test_one_rejected_branch_family_keeps_measurement_suppressed(self) -> None:
        from psse_env.oracle.measurement_expert import MeasurementExpert

        state = self._policy_state(
            [
                "wls_residual_outlier index=12 channel=Pinj",
                "wls_branch_multiplier_dominant line_status_or_parameter line=7",
            ]
        )
        state["rejected_hypotheses"] = [
            {"action_signature": "correct_parameters:{\"line_index\":7}"}
        ]
        proposals = MeasurementExpert().propose(state, [])
        tools = [proposal.action["tool"] for proposal in proposals]
        self.assertNotIn("get_measurement_context", tools)

    def test_both_rejected_branch_families_lift_measurement_suppression(self) -> None:
        from psse_env.oracle.measurement_expert import MeasurementExpert

        state = self._policy_state(
            [
                "wls_residual_outlier index=12 channel=Pinj",
                "wls_branch_multiplier_dominant line_status_or_parameter line=7",
            ]
        )
        state["rejected_hypotheses"] = [
            {"action_signature": "correct_parameters:{\"line_index\":7}"},
            {"action_signature": "correct_topology:{\"line_index\":7,\"status\":0}"},
        ]
        proposals = MeasurementExpert().propose(state, [])
        tools = [proposal.action["tool"] for proposal in proposals]
        self.assertIn("get_measurement_context", tools)

    def test_measurement_route_engages_when_residuals_dominate(self) -> None:
        from psse_env.oracle.measurement_expert import MeasurementExpert

        state = self._policy_state(
            [
                "wls_residual_outlier_dominant index=12 channel=Pinj",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )
        proposals = MeasurementExpert().propose(state, [])
        self.assertEqual(proposals[0].action["tool"], "get_measurement_context")
        self.assertGreater(proposals[0].confidence, 0.9)


class EndToEndRound0EpisodeTests(unittest.TestCase):
    def _run_episode(self, scenario, max_steps: int = 18):
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.transactional_env import TransactionalPSSEEnv

        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(),
            production_dataset_mode=True,
            max_steps=max_steps,
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(scenario)
        executed = []
        for _ in range(max_steps):
            if env.is_terminal():
                break
            actions = oracle.next_actions(
                env.get_oracle_state(env.history), env.history
            )
            self.assertTrue(actions, f"expert stalled after {executed}")
            _, output = env.step(actions[0])
            executed.append((actions[0], output))
        return env, executed

    def test_measurement_scenario_resolves_with_true_index_in_group(self) -> None:
        # Seed 29 is a well-conditioned single-outlier scenario.  Marginal
        # corners (e.g. seed 31's Vm error at radial bus 8) correctly identify
        # the true index but the achievable correction leaves chi-square just
        # above threshold, so the candidate is rolled back and the episode ends
        # nonterminal — the round-0 driver counts those rather than committing
        # a masking correction.
        generator = Round0ScenarioGenerator(seed=29)
        scenario = generator.build({"measurement": 1})[0]
        env, executed = self._run_episode(scenario)
        self.assertTrue(env.is_terminal())
        true_index = int(scenario["true_measurement_errors"][0]["index"])
        groups = [
            action["arguments"].get("suspect_group")
            for action, output in executed
            if action["tool"] == "correct_measurements"
            and output["execution_status"] == "success"
        ]
        self.assertTrue(groups)
        self.assertTrue(any(true_index in group for group in groups))

    def test_topology_scenario_resolves_by_status_flip_on_true_line(self) -> None:
        generator = Round0ScenarioGenerator(seed=31)
        scenario = generator.build({"topology": 1})[0]
        env, executed = self._run_episode(scenario)
        self.assertTrue(env.is_terminal())
        true_line = int(scenario["true_topology_errors"][0]["line_index1"])
        flips = [
            action["arguments"]
            for action, output in executed
            if action["tool"] == "correct_topology"
            and output["execution_status"] == "success"
        ]
        self.assertTrue(flips)
        self.assertEqual(int(flips[-1]["line_index"]), true_line)


class EpisodeTruthAuditTests(unittest.TestCase):
    def test_masking_measurement_commit_is_quarantined(self) -> None:
        from psse_env.examples.generate_round0_aggregate import (
            audit_episode_against_truth,
        )

        scenario = {
            "scenario_id": "r0_x",
            "scenario_family": "topology",
            "true_topology_errors": [{"line_index1": 12}],
        }
        final_state = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [4, 5]},
                    }
                }
            ]
        }
        audit = audit_episode_against_truth(scenario, final_state, terminal=True)
        self.assertTrue(audit["quarantined"])

    def test_true_family_commits_pass(self) -> None:
        from psse_env.examples.generate_round0_aggregate import (
            audit_episode_against_truth,
        )

        scenario = {
            "scenario_id": "r0_x",
            "scenario_family": "measurement+topology",
            "true_measurement_errors": [{"index": 100}],
            "true_topology_errors": [{"line_index1": 17}],
        }
        final_state = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_topology",
                        "arguments": {"line_index": 17, "status": 0},
                    }
                },
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [25, 100]},
                    }
                },
            ]
        }
        audit = audit_episode_against_truth(scenario, final_state, terminal=True)
        self.assertFalse(audit["quarantined"])


if __name__ == "__main__":
    unittest.main()
