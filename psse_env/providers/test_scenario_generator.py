from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import (
    BASE_FAMILIES,
    COMPOSED_FAMILIES,
    DEFAULT_HIF_FALLBACK_SAMPLE_PATHS,
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
        "three_phase_unbalance": 1,
        "telemetry_no_disturbance": 1,
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

    def test_measurement_tolerance_uses_combined_estimator_reference_noise(self) -> None:
        for family in ("measurement", "multi_measurement"):
            with self.subTest(family=family):
                scenario = self.by_family[family]
                indices = [
                    int(fault["index"])
                    for fault in scenario["true_measurement_errors"]
                ]
                expected = max(
                    1e-6,
                    3.0
                    * math.sqrt(2.0)
                    * max(float(self.generator.noise_profile()[index]) for index in indices),
                )
                release_audit = scenario["release_audit"]
                self.assertAlmostEqual(
                    release_audit["tolerances"]["measurement_abs"], expected
                )
                self.assertEqual(
                    release_audit["measurement_tolerance_basis"],
                    "three_sigma_independent_estimator_and_reference_noise",
                )

    def test_measurement_release_targets_are_globally_clean(self) -> None:
        for family in ("measurement", "multi_measurement"):
            with self.subTest(family=family):
                scenario = self.by_family[family]
                statistic = self.generator._chi2_statistic(
                    "case14", scenario["clean_measurements"]
                )
                self.assertLess(statistic, self.generator.chi2_limit)

    def test_each_multi_measurement_fault_is_individually_detectable(self) -> None:
        scenario = self.by_family["multi_measurement"]
        for fault in scenario["true_measurement_errors"]:
            probe = list(scenario["clean_measurements"])
            probe[int(fault["index"])] = float(fault["observed"])
            with self.subTest(index=fault["index"]):
                self.assertGreaterEqual(
                    self.generator._chi2_statistic("case14", probe),
                    self.generator.chi2_limit,
                )

    def test_parameter_scenario_carries_scans_and_stale_model(self) -> None:
        scenario = self.by_family["parameter"]
        self.assertEqual(scenario["case"], "case14")
        self.assertEqual(scenario["network_case"], "case14")
        self.assertNotEqual(scenario["clean_case"], "case14")
        scans = scenario["metadata"]["parameter_scans"]
        self.assertTrue(scans["z_scans"])
        self.assertNotIn("initial_states", scans)
        self.assertEqual(
            scans["initial_state_strategy"],
            "observed_vm_plus_configured_case_angles_v1",
        )
        fault = scenario["true_parameter_errors"][0]
        self.assertIn("clean_r", fault)
        self.assertIn("clean_x", fault)
        self.assertNotIn("parameter_correction_realizability", scenario)
        self.assertNotIn("parameter_correction_realizability", scenario["metadata"])

    def test_topology_scenario_is_synthesized_with_derived_clean_case(self) -> None:
        scenario = self.by_family["topology"]
        fault = scenario["true_topology_errors"][0]
        self.assertEqual(scenario["case"], "case14")
        self.assertEqual(scenario["network_case"], "case14")
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

    def test_diagnostic_families_emit_explanation_only_audit_contract(self) -> None:
        for family in ("harmonic", "hif", "three_phase_unbalance"):
            with self.subTest(family=family):
                release_audit = self.by_family[family]["release_audit"]
                self.assertEqual(
                    release_audit["explanation_only_contract"],
                    "explanation_only_diagnostic_localization_v1",
                )
                self.assertEqual(
                    set(release_audit["not_applicable"]),
                    {"final_measurements_match_clean"},
                )

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

    def test_unbalance_scenario_uses_distinct_observable_signature(self) -> None:
        scenario = self.by_family["three_phase_unbalance"]
        self.assertEqual(
            scenario["unresolved_signatures"],
            ["three_phase_unbalance vuf_threshold_exceeded"],
        )
        self.assertTrue(scenario["metadata"]["three_phase_voltages"])
        self.assertTrue(scenario["hidden_truth"]["true_unbalance_errors"])
        self.assertEqual(scenario["error_cardinality"], 1)

    def test_telemetry_negative_control_is_balanced_and_truth_clean(self) -> None:
        scenario = self.by_family["telemetry_no_disturbance"]
        self.assertNotIn("unresolved_signatures", scenario)
        self.assertEqual(scenario["error_cardinality"], 0)
        self.assertEqual(scenario["source_tier"], "derived_negative_control")
        for bus in scenario["metadata"]["three_phase_voltages"]:
            self.assertEqual(len(set(bus["vln_pu"])), 1)
            a, b, c = bus["ang_deg"]
            self.assertAlmostEqual(a - b, 120.0)
            self.assertAlmostEqual(c - a, 120.0)

    def test_tracked_hif_fallback_promotes_legacy_rows_to_scan_windows(self) -> None:
        generator = Round0ScenarioGenerator(
            hif_sample_paths=DEFAULT_HIF_FALLBACK_SAMPLE_PATHS,
            seed=20260719,
        )
        scenario = generator.build({"hif": 1})[0]
        window = scenario["metadata"]["hif_scan_window"]
        self.assertEqual(len(window["scans"]), 1)
        self.assertEqual(
            window["window_metadata"]["source_kind"],
            "tracked_single_scan_fallback",
        )

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


class PhysicalSourcePartitionTests(unittest.TestCase):
    @staticmethod
    def _write_corpus(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _row(identifier: str, value: float, *, scenario: str = "no_error") -> dict:
        return {
            "id": identifier,
            "scenario": scenario,
            "z_true": [value, value + 0.1],
            "z_obs": [value + 0.01, value + 0.11],
            "label": {"error_type": scenario},
            "op_point": {"load_scale": value},
        }

    def test_invalid_source_partition_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_partition"):
            Round0ScenarioGenerator(source_partition="validation")

    def test_default_partition_preserves_all_tabular_rows(self) -> None:
        rows = [self._row("a", 0.8), self._row("b", 1.2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            self._write_corpus(corpus_path, rows)
            generator = Round0ScenarioGenerator(
                corpus_path=corpus_path,
                source_partition=None,
                validate=False,
            )
            selected = generator._corpus()["no_error"]

        self.assertEqual(selected, rows)
        metadata = generator.report()["source_partition"]
        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["selected"], None)
        self.assertEqual(metadata["rows_selected_by_corpus_scenario"]["no_error"], 2)

    def test_partitions_are_complementary_and_group_duplicate_physics(self) -> None:
        unique = [self._row(f"root-{index}", 0.7 + index / 100.0) for index in range(20)]
        duplicate = dict(unique[4])
        duplicate["id"] = "renamed-copy"
        duplicate["correction_case_path"] = "/different/staging/name.m"
        rows = [*unique, duplicate]
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            self._write_corpus(corpus_path, rows)
            train = Round0ScenarioGenerator(
                corpus_path=corpus_path,
                source_partition="train",
                validate=False,
            )
            evaluation = Round0ScenarioGenerator(
                corpus_path=corpus_path,
                source_partition="evaluation",
                validate=False,
            )
            train_rows = train._corpus()["no_error"]
            evaluation_rows = evaluation._corpus()["no_error"]

        train_ids = {row["id"] for row in train_rows}
        evaluation_ids = {row["id"] for row in evaluation_rows}
        self.assertFalse(train_ids & evaluation_ids)
        self.assertEqual(train_ids | evaluation_ids, {row["id"] for row in rows})
        duplicate_ids = {"root-4", "renamed-copy"}
        self.assertTrue(
            duplicate_ids <= train_ids or duplicate_ids <= evaluation_ids,
            "duplicate physical rows crossed the source partition",
        )
        train_digests = {
            train._source_physical_digest(row) for row in train_rows
        }
        evaluation_digests = {
            evaluation._source_physical_digest(row) for row in evaluation_rows
        }
        self.assertFalse(train_digests & evaluation_digests)

        train_metadata = train.report()["source_partition"]
        evaluation_metadata = evaluation.report()["source_partition"]
        self.assertEqual(
            train_metadata["algorithm"], "sha256_physical_content_modulo_v1"
        )
        self.assertEqual(train_metadata["modulus"], 5)
        self.assertEqual(train_metadata["evaluation_buckets"], [0])
        self.assertEqual(
            train_metadata["physical_groups_total_by_corpus_scenario"]["no_error"],
            20,
        )
        self.assertEqual(
            train_metadata["physical_groups_selected_by_corpus_scenario"]["no_error"]
            + evaluation_metadata["physical_groups_selected_by_corpus_scenario"][
                "no_error"
            ],
            20,
        )

    def test_only_shared_tabular_sources_are_partitioned(self) -> None:
        no_error_rows = [
            self._row(f"clean-{index}", 0.8 + index / 100.0)
            for index in range(10)
        ]
        topology_rows = [
            self._row(f"topology-{index}", 1.1 + index / 100.0, scenario="topology_error")
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            self._write_corpus(corpus_path, [*no_error_rows, *topology_rows])
            for partition in ("train", "evaluation"):
                with self.subTest(partition=partition):
                    generator = Round0ScenarioGenerator(
                        corpus_path=corpus_path,
                        source_partition=partition,
                        validate=False,
                    )
                    corpus = generator._corpus()
                    self.assertEqual(corpus["topology_error"], topology_rows)

    def test_hif_and_composed_hif_sources_ignore_tabular_partition(self) -> None:
        hif_rows = [
            {"id": "hif-a", "z_obs": [1.0], "z_true": [0.9]},
            {"id": "hif-b", "z_obs": [1.1], "z_true": [0.9]},
        ]
        for partition in ("train", "evaluation"):
            with self.subTest(partition=partition):
                generator = Round0ScenarioGenerator(
                    source_partition=partition,
                    validate=False,
                )
                generator._corpus_by_class = {}
                generator._hif_samples = hif_rows
                direct_source, _ = generator._family_source("hif")
                composed_source, _ = generator._family_source("measurement+hif")
                self.assertEqual(direct_source, hif_rows)
                self.assertEqual(composed_source, hif_rows)

    def test_parameter_and_composed_parameter_share_the_same_partition(self) -> None:
        parameter_rows = [
            self._row(
                f"parameter-{index}",
                0.8 + index / 100.0,
                scenario="parameter_error",
            )
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            self._write_corpus(corpus_path, parameter_rows)
            generator = Round0ScenarioGenerator(
                corpus_path=corpus_path,
                source_partition="evaluation",
                validate=False,
            )
            parameter_source, _ = generator._family_source("parameter")
            composed_source, _ = generator._family_source("measurement+parameter")

        self.assertEqual(parameter_source, composed_source)


class ValidationGateTests(unittest.TestCase):
    @staticmethod
    def _parameter_ranking(
        *line_scores: tuple[int, float],
    ) -> dict[str, object]:
        ranked = [
            {"line_index1": line_index1, "abs_lambda_score": score}
            for line_index1, score in line_scores
        ]
        singleton = len(ranked) == 1
        runner_up = None if singleton else float(ranked[1]["abs_lambda_score"])
        ratio = (
            None
            if singleton
            else float(ranked[0]["abs_lambda_score"]) / float(runner_up)
        )
        return {
            "parameter_ranking_contract": (
                "distinct_line_abs_lambda_dominance_v1"
            ),
            "parameter_ranking_distinct_lines": ranked,
            "parameter_ranking_top_abs_lambda": (
                float(ranked[0]["abs_lambda_score"])
            ),
            "parameter_ranking_runner_up_abs_lambda": runner_up,
            "parameter_ranking_dominance_ratio": ratio,
            "parameter_ranking_dominance_threshold": 1.2,
            "parameter_ranking_singleton": singleton,
            "parameter_ranking_dominant": singleton or bool(ratio >= 1.2),
        }

    def test_parameter_dominance_is_training_admission_not_holdout_filter(
        self,
    ) -> None:
        training = Round0ScenarioGenerator(
            seed=5,
            source_partition="train",
        )
        evaluation = Round0ScenarioGenerator(
            seed=5,
            source_partition="evaluation",
        )

        self.assertIs(training._enforce_parameter_ranking_dominance, True)
        self.assertEqual(
            training._parameter_gate_provider.parameter_ranking_dominance_threshold,
            1.2,
        )
        self.assertIs(evaluation._enforce_parameter_ranking_dominance, False)
        self.assertEqual(
            evaluation._parameter_gate_provider.parameter_ranking_dominance_threshold,
            1.0,
        )

    def test_mixed_parameter_probe_preserves_observable_scan_metadata(self) -> None:
        generator = Round0ScenarioGenerator(seed=5, validate=True)
        provider = Mock()
        provider.run_wls.side_effect = [
            {
                "unresolved_signatures": [
                    "wls_residual_outlier_dominant index=1 channel=Vm"
                ],
                "remaining_anomaly_score": 10.0,
            },
            {
                "target_fixed": True,
                "post_action_resolved": False,
                "physical_constraints_ok": True,
                "remaining_anomaly_score": 5.0,
            },
            {
                "target_fixed": True,
                "post_action_resolved": True,
                "physical_constraints_ok": True,
                "remaining_anomaly_score": 0.0,
            },
        ]
        observed_parameter_states: list[dict] = []

        def parameter_context(state):
            observed_parameter_states.append(state)
            return {
                **self._parameter_ranking((2, 9.0), (1, 3.0)),
                "supported_corrections": [
                    {
                        "tool": "correct_parameters",
                        "arguments": {"state_id": state["state_id"], "line_index": 2},
                    }
                ]
            }

        provider.get_parameter_context.side_effect = parameter_context
        provider.get_measurement_context.return_value = {
            "supported_corrections": [
                {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "offline_mixed_parameter_gate:measurement_context_0",
                        "suspect_group": [1],
                    },
                }
            ]
        }
        provider.correct_measurements.return_value = {
            "modification": {"measurement_updates": {"1": 0.5}}
        }
        generator._parameter_gate_provider = provider
        generator._parameter_gate_results["base-root"] = {
            "corrected_case_path": "corrected_case.py"
        }
        metadata = {"parameter_scans": {"z_scans": [[1.0, 2.0, 3.0]]}}
        scenario = {
            "case": "case14",
            "measurements": [1.0, 1.5],
            "metadata": metadata,
            "true_parameter_errors": [{"line_index1": 2}],
            "true_measurement_errors": [{"index": 1, "clean": 0.5}],
        }

        generator._require_mixed_parameter_recovery_realizable(
            scenario, base_scenario_id="base-root"
        )

        self.assertEqual(len(observed_parameter_states), 1)
        self.assertEqual(observed_parameter_states[0]["metadata"], metadata)
        self.assertIsNot(observed_parameter_states[0]["metadata"], metadata)
        accepted = observed_parameter_states[0]["policy_observation"][
            "accepted_corrections"
        ]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source_action"]["tool"], "correct_measurements")

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

    @patch(
        "psse_env.providers.scenario_generator._param_correction_json",
        return_value={"success": False, "error": "did not converge"},
    )
    def test_parameter_solver_failure_records_offline_metrics(self, mocked) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        scan = json.loads(FIXTURE.read_text())["z_obs"]
        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_parameter_correction_realizable(
                line_row0=6,
                clean_r=0.01,
                clean_x=0.05,
                z_scans=[scan],
                measurements=scan,
                final_case_abs_tolerance=0.02,
            )

        self.assertEqual(
            caught.exception.reason, "parameter_correction_unrealizable"
        )
        self.assertEqual(caught.exception.metrics["line_index1"], 7)
        self.assertFalse(caught.exception.metrics["solver_success"])
        mocked.assert_called_once()

        generator._record_skip(
            "measurement+parameter", "fixture", caught.exception
        )
        self.assertEqual(
            generator.skipped[-1]["metrics"], caught.exception.metrics
        )

    @patch(
        "psse_env.providers.scenario_generator._param_correction_json",
        return_value={"success": True, "corrected_params": [0.05, 0.20]},
    )
    def test_parameter_estimate_must_meet_declared_final_tolerance(self, mocked) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        scan = json.loads(FIXTURE.read_text())["z_obs"]
        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_parameter_correction_realizable(
                line_row0=1,
                clean_r=0.01,
                clean_x=0.05,
                z_scans=[scan],
                measurements=scan,
                final_case_abs_tolerance=0.02,
            )

        self.assertEqual(
            caught.exception.reason,
            "parameter_correction_outside_release_tolerance",
        )
        self.assertGreater(
            caught.exception.metrics["max_abs_error"],
            caught.exception.metrics["final_case_abs_tolerance"],
        )
        mocked.assert_called_once()

    @patch(
        "psse_env.providers.scenario_generator._param_correction_json",
        return_value={"success": True, "corrected_params": [0.01, 0.05]},
    )
    def test_parameter_candidate_must_pass_observable_wls_criteria(self, mocked) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        scan = json.loads(FIXTURE.read_text())["z_obs"]
        generator._parameter_gate_provider.run_wls = Mock(
            return_value={
                "target_fixed": False,
                "target_metric_value": 4.2,
                "target_metric_threshold": 3.0,
                "chi_square_statistic": 100.0,
                "chi_square_threshold": 130.0,
                "post_action_resolved": True,
                "globally_resolved": False,
                "physical_constraints_ok": True,
                "physical_evidence_scope": "observed_snapshot_topology_vm_rate_a",
            }
        )

        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_parameter_correction_realizable(
                line_row0=1,
                clean_r=0.01,
                clean_x=0.05,
                z_scans=[scan],
                measurements=scan,
                final_case_abs_tolerance=0.02,
            )

        self.assertEqual(
            caught.exception.reason,
            "parameter_correction_candidate_unresolved",
        )
        self.assertFalse(caught.exception.metrics["target_fixed"])
        self.assertTrue(caught.exception.metrics["post_action_resolved"])
        self.assertTrue(caught.exception.metrics["physical_constraints_ok"])
        mocked.assert_called_once()

    @patch(
        "psse_env.providers.scenario_generator._param_correction_json",
        return_value={"success": True, "corrected_params": [0.01, 0.05]},
    )
    def test_parameter_context_must_rank_declared_line_first(self, mocked) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        scan = json.loads(FIXTURE.read_text())["z_obs"]
        generator._parameter_gate_provider.run_wls = Mock(
            return_value={
                "target_fixed": True,
                "post_action_resolved": True,
                "globally_resolved": True,
                "physical_constraints_ok": True,
            }
        )
        generator._parameter_gate_provider.get_parameter_context = Mock(
            return_value={
                **self._parameter_ranking((2, 9.0), (1, 3.0)),
                "supported_corrections": [
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": "offline_parameter_context:l1",
                            "line_index": 2,
                        },
                    },
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": "offline_parameter_context:l1",
                            "line_index": 1,
                        },
                    },
                ]
            }
        )

        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_parameter_correction_realizable(
                line_row0=0,
                clean_r=0.01,
                clean_x=0.05,
                z_scans=[scan],
                measurements=scan,
                final_case_abs_tolerance=0.02,
            )

        self.assertEqual(
            caught.exception.reason, "parameter_context_target_ambiguous"
        )
        self.assertEqual(
            caught.exception.metrics["parameter_context_supported_line_indices1"],
            [2, 1],
        )
        self.assertEqual(
            caught.exception.metrics["parameter_context_first_line_index1"], 2
        )
        mocked.assert_called_once()

    @patch(
        "psse_env.providers.scenario_generator._param_correction_json",
        return_value={"success": True, "corrected_params": [0.01, 0.05]},
    )
    def test_parameter_context_true_first_target_remains_eligible(self, mocked) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        scan = json.loads(FIXTURE.read_text())["z_obs"]
        generator._parameter_gate_provider.run_wls = Mock(
            return_value={
                "target_fixed": True,
                "post_action_resolved": True,
                "globally_resolved": True,
                "physical_constraints_ok": True,
            }
        )
        generator._parameter_gate_provider.get_parameter_context = Mock(
            return_value={
                **self._parameter_ranking((1, 9.0), (2, 3.0)),
                "supported_corrections": [
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": "offline_parameter_context:l1",
                            "line_index": 1,
                        },
                    },
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": "offline_parameter_context:l1",
                            "line_index": 2,
                        },
                    },
                ]
            }
        )

        result = generator._require_parameter_correction_realizable(
            line_row0=0,
            clean_r=0.01,
            clean_x=0.05,
            z_scans=[scan],
            measurements=scan,
            final_case_abs_tolerance=0.02,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["corrected_r"], 0.01)
        self.assertEqual(result["corrected_x"], 0.05)
        mocked.assert_called_once()

    @patch(
        "psse_env.providers.scenario_generator._param_correction_json",
        return_value={"success": True, "corrected_params": [0.01, 0.05]},
    )
    def test_parameter_context_requires_observable_dominance_even_when_top1_matches(
        self, mocked
    ) -> None:
        generator = Round0ScenarioGenerator(seed=5)
        scan = json.loads(FIXTURE.read_text())["z_obs"]
        generator._parameter_gate_provider.run_wls = Mock(
            return_value={
                "target_fixed": True,
                "post_action_resolved": True,
                "globally_resolved": True,
                "physical_constraints_ok": True,
            }
        )
        ranking = self._parameter_ranking((1, 11.0), (2, 10.0))
        self.assertIs(ranking["parameter_ranking_dominant"], False)
        generator._parameter_gate_provider.get_parameter_context = Mock(
            return_value={
                **ranking,
                # Keep a nominal top-one action in the mock to prove that
                # admission checks the metrics contract independently rather
                # than inferring dominance from inventory order.
                "supported_corrections": [
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": "offline_parameter_context:l1",
                            "line_index": 1,
                        },
                    }
                ],
            }
        )

        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_parameter_correction_realizable(
                line_row0=0,
                clean_r=0.01,
                clean_x=0.05,
                z_scans=[scan],
                measurements=scan,
                final_case_abs_tolerance=0.02,
            )

        self.assertEqual(
            caught.exception.reason, "parameter_context_target_not_dominant"
        )
        self.assertEqual(
            caught.exception.metrics["parameter_context_ranking"], ranking
        )
        mocked.assert_called_once()

    def test_mixed_parameter_context_requires_observable_dominance(self) -> None:
        generator = Round0ScenarioGenerator(seed=5, validate=True)
        provider = Mock()
        provider.run_wls.side_effect = [
            {
                "unresolved_signatures": [
                    "wls_residual_outlier_dominant index=1 channel=Vm"
                ],
                "remaining_anomaly_score": 10.0,
            },
            {
                "target_fixed": True,
                "post_action_resolved": False,
                "physical_constraints_ok": True,
                "remaining_anomaly_score": 5.0,
            },
        ]
        provider.get_measurement_context.return_value = {
            "supported_corrections": [
                {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": (
                            "offline_mixed_parameter_gate:"
                            "measurement_context_0"
                        ),
                        "suspect_group": [1],
                    },
                }
            ]
        }
        provider.correct_measurements.return_value = {
            "modification": {"measurement_updates": {"1": 0.5}}
        }
        ranking = self._parameter_ranking((2, 11.0), (1, 10.0))
        provider.get_parameter_context.return_value = {
            **ranking,
            "supported_corrections": [
                {
                    "tool": "correct_parameters",
                    "arguments": {
                        "state_id": (
                            "offline_mixed_parameter_gate:"
                            "parameter_context_1"
                        ),
                        "line_index": 2,
                    },
                }
            ],
        }
        generator._parameter_gate_provider = provider
        generator._parameter_gate_results["base-root"] = {
            "corrected_case_path": "corrected_case.py"
        }
        scenario = {
            "case": "case14",
            "measurements": [1.0, 1.5],
            "metadata": {
                "parameter_scans": {"z_scans": [[1.0, 2.0, 3.0]]}
            },
            "true_parameter_errors": [{"line_index1": 2}],
            "true_measurement_errors": [{"index": 1, "clean": 0.5}],
        }

        with self.assertRaises(ScenarioRejected) as caught:
            generator._require_mixed_parameter_recovery_realizable(
                scenario, base_scenario_id="base-root"
            )

        self.assertEqual(
            caught.exception.reason,
            "mixed_parameter_recovery_context_not_dominant",
        )
        self.assertEqual(caught.exception.metrics["stages"][-1]["parameter_ranking"], ranking)

    def test_known_ambiguous_parameter_root_is_rejected_by_dominance_gate(
        self,
    ) -> None:
        generator = Round0ScenarioGenerator(seed=20260719)
        row = next(
            item
            for item in generator._corpus()["parameter_error"]
            if item.get("id") == "pe_428232230768"
        )

        with self.assertRaises(ScenarioRejected) as caught:
            generator._parameter_scenario(row, 68)

        self.assertEqual(
            caught.exception.reason, "parameter_context_target_not_dominant"
        )
        self.assertEqual(caught.exception.metrics["line_index1"], 1)
        ranking = caught.exception.metrics["parameter_context_ranking"]
        self.assertIs(ranking["parameter_ranking_dominant"], False)
        self.assertLess(
            ranking["parameter_ranking_dominance_ratio"],
            ranking["parameter_ranking_dominance_threshold"],
        )
        self.assertEqual(
            [
                item["line_index1"]
                for item in ranking["parameter_ranking_distinct_lines"][:2]
            ],
            [2, 1],
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

    def test_rejected_branch_actions_without_inventories_keep_measurement_suppressed(
        self,
    ) -> None:
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
        self.assertNotIn("get_measurement_context", tools)
        self.assertIn("run_wls", tools)

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
    @staticmethod
    def _without_privileged_targets(scenario):
        scenario = dict(scenario)
        for key in list(scenario):
            if key.startswith("true_") or key.startswith("clean_") or key in {
                "hidden_truth",
                "oracle_action_hints",
            }:
                scenario.pop(key, None)
        return scenario

    def _run_episode(
        self, scenario, max_steps: int = 18, provider_kwargs: dict | None = None
    ):
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.transactional_env import TransactionalPSSEEnv

        providers = MatpowerDeploymentProviders(
            chi2_alpha=0.01, **dict(provider_kwargs or {})
        )
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

    def _assert_scoped_candidate_physical_evidence(self, executed) -> None:
        candidate_checks = [
            output["tool_metrics"]
            for action, output in executed
            if action["tool"] == "run_wls"
            and "steady_state_physical_evidence" in output.get("tool_metrics", {})
        ]
        self.assertEqual(len(candidate_checks), 1)
        metrics = candidate_checks[0]
        self.assertIs(metrics["physical_constraints_ok"], True)
        self.assertNotIn("power_flow_converged", metrics)
        evidence = metrics["steady_state_physical_evidence"]
        self.assertEqual(evidence["scope"], "observed_snapshot_topology_vm_rate_a")
        self.assertTrue(evidence["complete"])

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

    def test_measurement_cycle_is_autonomous_without_truth_or_hints(self) -> None:
        generator = Round0ScenarioGenerator(seed=29)
        source = generator.build({"measurement": 1})[0]
        true_index = int(source["true_measurement_errors"][0]["index"])
        scenario = self._without_privileged_targets(source)
        env, executed = self._run_episode(scenario)

        self.assertTrue(env.is_terminal())
        oracle_state = env.get_oracle_state()
        self.assertFalse(oracle_state.true_measurement_errors)
        self.assertFalse(oracle_state.oracle_action_hints)
        tools = [action["tool"] for action, _ in executed]
        self.assertEqual(
            tools,
            [
                "run_wls",
                "get_measurement_context",
                "correct_measurements",
                "run_wls",
                "commit_state",
                "finalize_diagnosis",
            ],
        )
        group = next(
            action["arguments"]["suspect_group"]
            for action, _ in executed
            if action["tool"] == "correct_measurements"
        )
        self.assertIn(true_index, group)
        self._assert_scoped_candidate_physical_evidence(executed)

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

    def test_branch_family_ambiguity_resolves_without_privileged_truth(self) -> None:
        generator = Round0ScenarioGenerator(seed=31)
        source = generator.build({"topology": 1})[0]
        true_line = int(source["true_topology_errors"][0]["line_index1"])
        env, executed = self._run_episode(
            self._without_privileged_targets(source), max_steps=24
        )

        self.assertTrue(env.is_terminal())
        oracle_state = env.get_oracle_state()
        self.assertFalse(oracle_state.true_parameter_errors)
        self.assertFalse(oracle_state.true_topology_errors)
        self.assertFalse(oracle_state.oracle_action_hints)
        tools = [action["tool"] for action, _ in executed]
        self.assertIn("get_parameter_context", tools)
        self.assertIn("get_topology_context", tools)
        # Missing parameter scans close that hypothesis at the observable
        # context boundary.  The expert must not emit a correction that the
        # provider has already declined to support; topology context then
        # supplies the bounded status flip that verifies and commits.
        parameter_context = next(
            output
            for action, output in executed
            if action["tool"] == "get_parameter_context"
        )
        self.assertEqual(
            parameter_context.get("tool_metrics", {}).get("supported_corrections"),
            [],
        )
        self.assertNotIn("correct_parameters", tools)
        topology_action = next(
            action for action, output in executed
            if action["tool"] == "correct_topology"
            and output["execution_status"] == "success"
        )
        self.assertEqual(int(topology_action["arguments"]["line_index"]), true_line)
        self._assert_scoped_candidate_physical_evidence(executed)

    def test_mixed_measurement_topology_commits_partial_then_final_without_truth(self) -> None:
        generator = Round0ScenarioGenerator(seed=20260719)
        source = generator.build({"measurement+topology": 1})[0]
        env, executed = self._run_episode(
            self._without_privileged_targets(source), max_steps=24
        )

        self.assertTrue(env.is_terminal())
        oracle_state = env.get_oracle_state()
        self.assertFalse(oracle_state.true_measurement_errors)
        self.assertFalse(oracle_state.true_topology_errors)
        successful_corrections = [
            (action, output)
            for action, output in executed
            if action["tool"] in {"correct_measurements", "correct_topology"}
            and output["execution_status"] == "success"
        ]
        self.assertEqual(
            {action["tool"] for action, _ in successful_corrections},
            {"correct_measurements", "correct_topology"},
        )

        candidate_ids = [
            str(output["candidate_state_id"])
            for _, output in successful_corrections
        ]
        candidates = [env.store.get_state(candidate_id) for candidate_id in candidate_ids]
        dispositions = [candidate["candidate_disposition"] for candidate in candidates]
        self.assertIn("ACCEPT_PARTIAL", dispositions)
        self.assertEqual(dispositions[-1], "ACCEPT_FINAL")

        partial = next(
            candidate
            for candidate in candidates
            if candidate["candidate_disposition"] == "ACCEPT_PARTIAL"
        )
        verification = partial["verification_output"]
        self.assertFalse(verification["globally_resolved"])
        self.assertIs(verification["physical_constraints_ok"], True)
        self.assertTrue(verification["physical_evidence_complete"])
        self.assertNotIn("power_flow_converged", verification)
        self.assertEqual(
            verification["steady_state_physical_evidence"]["method"],
            "matpower_case_limits_with_observed_wls_telemetry",
        )

    def test_rejected_mixed_hif_ladder_ends_in_explicit_operator_escalation(self) -> None:
        generator = Round0ScenarioGenerator(seed=20260719, hif_max_scans=3)
        source = generator.build({"measurement+hif": 1})[0]
        env, executed = self._run_episode(
            self._without_privileged_targets(source),
            max_steps=24,
            provider_kwargs={
                "hif_alpha_grid_size": 5,
                "hif_r_grid_size": 7,
                "hif_max_scans": 3,
            },
        )

        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        tools = [action["tool"] for action, _ in executed]
        self.assertNotIn("finalize_diagnosis", tools)
        self.assertEqual(tools[-1], "ask_for_more_evidence")
        final_action, final_output = executed[-1]
        self.assertEqual(
            final_action["arguments"]["request"],
            "operator_escalation:hif_diagnostics_exhausted",
        )
        self.assertEqual(
            final_output["tool_metrics"]["terminal_outcome"],
            "operator_escalation",
        )
        self.assertFalse(env.current_state()["no_material_anomaly_remaining"])
        self.assertTrue(env.current_state()["unresolved_signatures"])


class EpisodeTruthAuditTests(unittest.TestCase):
    def test_incomplete_collection_blocks_release_provenance(self) -> None:
        from psse_env.examples.generate_round0_aggregate import (
            _collection_release_failures,
        )

        failures = _collection_release_failures(
            plan={"measurement": 2, "topology": 1},
            scenarios=[{"scenario_family": "measurement"}],
            nonterminal_episodes=["r0_nonterminal"],
            quarantined_episodes=[{"scenario_id": "r0_bad"}],
        )
        self.assertTrue(any("did not reach a terminal decision" in item for item in failures))
        self.assertTrue(any("truth-side correction audit" in item for item in failures))
        self.assertTrue(any('"measurement": 1' in item for item in failures))
        self.assertTrue(any('"topology": 1' in item for item in failures))

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
            "case": "case14",
            "measurements": [0.0] * 122,
            "clean_measurements": [0.0] * 122,
            "true_measurement_errors": [{"index": 100, "clean": 0.0}],
            "true_topology_errors": [
                {"line_index1": 17, "expected_status": 0}
            ],
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
                        "arguments": {"suspect_group": [100]},
                    }
                },
            ]
        }
        audit = audit_episode_against_truth(
            scenario,
            final_state,
            terminal=True,
            active_physical_state={
                "case": "case14",
                "measurements": [0.0] * 122,
            },
        )
        self.assertFalse(audit["quarantined"])


if __name__ == "__main__":
    unittest.main()
