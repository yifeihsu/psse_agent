"""Real IEEE57 deployment checks, without an LLM or precomputed teacher labels."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pypower.api import case57, ppoption, runopf
from scipy.stats import chi2

from mcp_server.matpower_server import _load_python_case
from psse_env.providers.matpower import _render_matpower_case
from psse_env.providers.scenario_generator import build_measurement_vector
from scripts.validate_balanced_transfer import evaluate_scenarios, normalize_scenarios


class BalancedTransferEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="ieee57_transfer_test_")
        cls.addClassCleanup(cls.temporary.cleanup)
        # The deployment parser intentionally loads only estimation tables;
        # PYPOWER supplies the generator costs needed to solve these OPFs.
        cls.base = case57()
        parsed = _load_python_case("case57")
        for table in ("bus", "branch", "gen"):
            np.testing.assert_allclose(parsed[table], cls.base[table], rtol=0, atol=0)
        cls.options = ppoption(VERBOSE=0, OUT_ALL=0)

        def telemetry(case, scale=1.0):
            operating = copy.deepcopy(case)
            operating["bus"][:, 2:4] *= scale
            solved = runopf(operating, cls.options)
            if not solved["success"]:
                raise AssertionError("IEEE57 test operating point did not solve AC OPF")
            return build_measurement_vector(solved).tolist()

        clean = telemetry(cls.base)
        cls.clean_scenario = {
            "scenario_id": "ieee57_clean_test",
            "scenario_family": "no_error",
            "network_case": "case57",
            "case": "case57",
            "measurements": clean,
            "clean_case": "case57",
            "clean_measurements": copy.deepcopy(clean),
            "metadata": {},
            "error_cardinality": 0,
            "source_tier": "physics_synthesized",
            "true_measurement_errors": [],
            "true_parameter_errors": [],
            "true_topology_errors": [],
        }
        meter = copy.deepcopy(cls.clean_scenario)
        meter.update(
            scenario_id="ieee57_meter450_test", scenario_family="measurement",
            error_cardinality=1,
        )
        meter["measurements"][450] += 2.0
        meter["true_measurement_errors"] = [{
            "index": 450, "observed": meter["measurements"][450], "clean": clean[450],
        }]
        # A noiseless reference permits a tighter bound than the generation
        # contract's three-sigma sensor-noise allowance.
        meter["release_audit"] = {"tolerances": {"measurement_abs": 0.01}}
        residual_only = copy.deepcopy(meter)
        residual_only["scenario_id"] = "ieee57_residual_only_meter450_test"
        residual_only["measurements"][450] = clean[450] + 0.1
        residual_only["true_measurement_errors"][0]["observed"] = residual_only["measurements"][450]
        cls.residual_only_scenario = residual_only

        def parameter(row, factor, name, scales):
            true_case = copy.deepcopy(cls.base)
            true_case["branch"][row, 2:4] *= factor
            path = Path(cls.temporary.name) / f"{name}.m"
            path.write_text(_render_matpower_case(true_case, name), encoding="utf-8")
            scans = [telemetry(true_case, scale) for scale in scales]
            scenario = copy.deepcopy(cls.clean_scenario)
            scenario.update(
                scenario_id=name, scenario_family="parameter", error_cardinality=1,
                clean_case=str(path), measurements=scans[scales.index(1.0)],
                clean_measurements=copy.deepcopy(scans[scales.index(1.0)]),
                metadata={"parameter_scans": {"z_scans": scans}},
            )
            clean_r, clean_x = map(float, true_case["branch"][row, 2:4])
            scenario["true_parameter_errors"] = [{
                "branch_row0": row, "line_index1": row + 1, "parameter": "rx",
                "clean_r": clean_r, "clean_x": clean_x,
            }]
            scenario["release_audit"] = {
                "tolerances": {"final_case_abs": max(0.02, 0.10 * max(clean_r, clean_x))},
                "tolerance_basis": "multi_scan_parameter_estimator_v2",
            }
            return scenario

        recovered = parameter(21, 2.0, "ieee57_line22_test", [0.9, 1.0, 1.05])
        undetected = parameter(26, 1.5, "ieee57_undetected_parameter_test", [1.0])
        cls.scenarios = [cls.clean_scenario, meter, residual_only, recovered, undetected]
        cls.report = evaluate_scenarios(cls.scenarios)
        cls.episodes = {
            episode["scenario_id"]: episode
            for episode in cls.report["evaluation"]["suite_metrics"]["episodes"]
        }

    def test_clean_case57_uses_full_measurements_and_resolves(self) -> None:
        self.assertEqual(len(self.clean_scenario["measurements"]), 491)
        episode = self.episodes["ieee57_clean_test"]
        self.assertTrue(episode["final_physical_success"])
        self.assertTrue(episode["healthy_components_preserved"])
        self.assertEqual(
            [step["action"]["tool"] for step in episode["trace"]],
            ["run_wls", "finalize_diagnosis"],
        )
        self.assertEqual(self.report["infrastructure_errors"], [])
        self.assertTrue(self.report["runtime_source_attestation"]["source_hashes_matched"])

    def test_high_index_meter_recovers_after_unavailable_acquisitions(self) -> None:
        episode = self.episodes["ieee57_meter450_test"]
        self.assertTrue(episode["truth_audited_task_success"])
        self.assertTrue(episode["healthy_components_preserved"])
        self.assertEqual(episode["false_commit_count"], 0)
        self.assertEqual(episode["invalid_action_count"], 0)
        audit = episode["audit"]["accepted_target_audit"]
        self.assertEqual(audit["accepted_targets"]["measurement"], [450])
        self.assertEqual(audit["uncovered_standard_faults"], 0)
        summary = next(item for item in self.report["episodes"] if item["scenario_id"] == episode["scenario_id"])
        self.assertEqual(
            {item["tool"] for item in summary["acquisitions"]},
            {"get_three_phase_context", "get_harmonic_context"},
        )
        for acquisition in summary["acquisitions"]:
            self.assertEqual(acquisition["execution_status"], "success")
            self.assertEqual(acquisition["context_status"], "unavailable")
            self.assertEqual(acquisition["available_evidence_channels"], [])

    def test_effective_thresholds_and_sparse_residual_alarm(self) -> None:
        config = self.report["detection_configuration"]
        self.assertEqual(config["chi2_alpha"], 0.05)
        self.assertEqual(config["normalized_residual_threshold"], 4.0)
        summary = next(item for item in self.report["episodes"] if item["scenario_id"] == "ieee57_residual_only_meter450_test")
        baseline = summary["initial_wls"]
        self.assertAlmostEqual(baseline["chi_square_threshold"], chi2.ppf(0.95, 378))
        self.assertEqual(baseline["normalized_residual_threshold"], 4.0)
        self.assertEqual(baseline["anomaly_detection_rule"], "chi_square_or_normalized_residual")
        self.assertFalse(baseline["chi_square_alarm"])
        self.assertTrue(baseline["normalized_residual_alarm"])
        self.assertFalse(baseline["no_material_anomaly_remaining"])
        self.assertTrue(summary["truth_audited_task_success"])
        self.assertEqual(self.report["initial_detection_summary"]["clean_control_count"], 1)
        self.assertEqual(self.report["initial_detection_summary"]["clean_control_alarm_count"], 0)

    def test_chi_square_only_remains_explicitly_available(self) -> None:
        report = evaluate_scenarios(
            [self.residual_only_scenario], chi2_alpha=0.01,
            normalized_residual_threshold=None,
        )
        self.assertIsNone(report["detection_configuration"]["normalized_residual_threshold"])
        baseline = report["episodes"][0]["initial_wls"]
        self.assertAlmostEqual(baseline["chi_square_threshold"], chi2.ppf(0.99, 378))
        self.assertEqual(baseline["anomaly_detection_rule"], "chi_square_only")
        self.assertTrue(baseline["no_material_anomaly_remaining"])
        self.assertFalse(report["episodes"][0]["truth_audited_task_success"])

    def test_invalid_alarm_configuration_is_rejected(self) -> None:
        for alpha in (0.0, 1.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                evaluate_scenarios([self.clean_scenario], chi2_alpha=alpha)
        for threshold in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                evaluate_scenarios([self.clean_scenario], normalized_residual_threshold=threshold)

    def test_branch_beyond_ieee14_range_recovers_using_observed_scans(self) -> None:
        episode = self.episodes["ieee57_line22_test"]
        self.assertTrue(episode["truth_audited_task_success"])
        self.assertTrue(episode["healthy_components_preserved"])
        self.assertEqual(episode["false_commit_count"], 0)
        self.assertEqual(episode["invalid_action_count"], 0)
        corrections = [step["action"] for step in episode["trace"] if step["action"]["tool"] == "correct_parameters"]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["arguments"]["line_index"], 22)
        self.assertEqual(
            episode["audit"]["accepted_target_audit"]["accepted_targets"]["parameter"], [21],
        )
        self.assertEqual(episode["audit"]["strict_release_audit"]["problems"], [])

    def test_undetected_physical_fault_remains_a_teacher_failure(self) -> None:
        episode = self.episodes["ieee57_undetected_parameter_test"]
        self.assertEqual(episode["terminal_outcome"], "resolved")
        self.assertFalse(episode["truth_audited_task_success"])
        self.assertTrue(episode["truth_audited_task_success_evidence_known"])
        self.assertFalse(episode["final_physical_success"])
        self.assertEqual(episode["false_finalization_count"], 1)
        self.assertIn(
            "resolved_episode_has_remaining_true_faults",
            episode["audit"]["strict_release_audit"]["problems"],
        )
        self.assertEqual(self.report["family_summary"]["parameter"]["physical_roots"], 2)
        self.assertEqual(self.report["family_summary"]["parameter"]["teacher_task_failures"], 1)
        self.assertEqual(self.report["family_summary"]["parameter"]["false_finalizations"], 1)
        self.assertEqual(self.report["family_summary"]["parameter"]["final_physical_correct"], 0)
        self.assertEqual(self.report["family_summary"]["parameter"]["healthy_components_preserved"], 2)
        self.assertTrue(self.report["all_supplied_physical_roots_retained"])

    def test_duplicate_physical_roots_and_out_of_scope_families_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_scenarios([self.clean_scenario, copy.deepcopy(self.clean_scenario)])
        unsupported = copy.deepcopy(self.clean_scenario)
        unsupported["scenario_family"] = "topology"
        with self.assertRaisesRegex(ValueError, "outside balanced transfer scope"):
            normalize_scenarios([unsupported])


if __name__ == "__main__":
    unittest.main()
