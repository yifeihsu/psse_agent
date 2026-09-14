"""Fresh OpenDSS diagnostic checks; no saved fault labels reach the screen."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pypower.api import case57

from .diagnostics import DiagnosticConfig, capture_nominal_model, screen_measurements
from .disturbances import inject_midspan_hif
from .exporter import export_model, load_assumptions
from .measurements import extract_measurements
from .runtime import compile_model, redistribute_load


class PhaseDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="phase_screening_")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.models = {name: export_model(case57(), Path(cls.tmp.name)/name, case_id="case57",
                                        assumptions=load_assumptions(name))
                      for name in ("normalized_diagonal", "coupled_sensitivity")}

    def context(self, name="normalized_diagonal"):
        model = self.models[name]
        dss = compile_model(Path(model["output_dir"])/"Master.dss")
        nominal = capture_nominal_model(dss, model["registry"], model["assumptions"])
        return dss, model, nominal

    @staticmethod
    def measure(dss, model, receipt=None):
        return extract_measurements(dss, model["registry"], model["assumptions"],
                                    branch_overrides=receipt["branch_overrides"] if receipt else None)

    @staticmethod
    def noisy(telemetry, config, seed=412):
        result = copy.deepcopy(telemetry)
        rng = np.random.default_rng(seed)
        for row in result["three_phase_voltages"]:
            row["vln_pu_rect"] = (np.array(row["vln_pu_rect"])+rng.normal(0, config.voltage_sigma_pu, (3, 2))).tolist()
        for row in result["three_phase_branch_currents"]:
            for key in ("i_from_pu_rect", "i_to_pu_rect"):
                row[key] = (np.array(row[key])+rng.normal(0, config.current_sigma_pu, (3, 2))).tolist()
        return result

    def test_full_registry_clean_and_noisy_controls(self):
        config = DiagnosticConfig(voltage_sigma_pu=1e-4, current_sigma_pu=1e-3)
        for name in self.models:
            dss, model, nominal = self.context(name)
            telemetry = self.measure(dss, model)
            for observation in (telemetry, self.noisy(telemetry, config)):
                result = screen_measurements(observation, nominal, config=config)
                self.assertEqual(result["classification"], "no_detectable_anomaly")
                self.assertEqual(len(result["branch_ranking"]), 80)
                self.assertEqual(len(result["nodal_ranking"]), 57)

    def test_midspan_hif_beyond_ieee14_indices_localizes_phase_and_analytic_parameters(self):
        for name in self.models:
            dss, model, nominal = self.context(name)
            fault = inject_midspan_hif(dss, model["registry"], model["assumptions"],
                                       branch_row0=62, phase=2, alpha=.3, resistance_pu=10)
            result = screen_measurements(self.measure(dss, model, fault), nominal)
            self.assertEqual(result["classification"], "hif_like_branch_mismatch")
            candidate = result["hif_candidate"]
            self.assertEqual((candidate["branch_row0"], candidate["phase"]), (62, 2))
            self.assertAlmostEqual(candidate["alpha_estimate"], .3, places=8)
            self.assertAlmostEqual(candidate["resistance_pu_estimate"], 10, places=7)
            self.assertTrue(candidate["parameter_estimates_accepted"])
            self.assertIsNone(result["unbalance_candidate"])

    def test_coupled_unbalance_requires_full_coupled_matrices(self):
        dss, model, nominal = self.context("coupled_sensitivity")
        redistribute_load(dss, model["registry"], bus=12, delta=.2)
        telemetry = self.measure(dss, model)
        result = screen_measurements(telemetry, nominal)
        self.assertEqual(result["classification"], "load_unbalance")
        self.assertEqual(result["unbalance_candidate"]["bus"], 12)
        self.assertLess(result["max_branch_normalized_residual"], 1e-5)
        _, _, wrong_nominal = self.context("normalized_diagonal")
        wrong = screen_measurements(telemetry, wrong_nominal)
        self.assertTrue(wrong["anomaly_detected"])
        self.assertGreater(wrong["max_branch_normalized_residual"], 6)
        self.assertEqual(wrong["classification"], "ambiguous")

    def test_truth_and_device_power_fields_cannot_select_candidates(self):
        dss, model, nominal = self.context()
        redistribute_load(dss, model["registry"], bus=57, delta=.2)
        telemetry = self.measure(dss, model)
        first = screen_measurements(telemetry, nominal)
        altered = copy.deepcopy(telemetry)
        for key in ("load_powers", "generator_injections", "bus_injections", "branch_powers", "source_injection"):
            altered[key] = {"false_bus": 1, "false_branch": 0, "huge": 1e30}
        altered["hidden_truth"] = {"fault_branch": 0, "unbalance_bus": 1}
        altered["measurement_vector"] = [-999]
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(screen_measurements(altered, nominal), sort_keys=True))
        self.assertEqual(first["unbalance_candidate"]["bus"], 57)

    def test_weak_hif_is_retained_as_below_declared_sensor_resolution(self):
        dss, model, nominal = self.context()
        fault = inject_midspan_hif(dss, model["registry"], model["assumptions"],
                                   branch_row0=62, phase=2, alpha=.3, resistance_pu=1000)
        telemetry = self.measure(dss, model, fault)
        main = screen_measurements(telemetry, nominal,
                                   config=DiagnosticConfig(voltage_sigma_pu=1e-4, current_sigma_pu=1e-3))
        self.assertEqual(main["classification"], "no_detectable_anomaly")
        self.assertTrue(main["interpretation"]["no_anomaly_is_not_proof_of_no_fault"])
        precise = screen_measurements(telemetry, nominal)
        self.assertEqual(precise["hif_candidate"]["branch_row0"], 62)
        self.assertFalse(precise["hif_candidate"]["parameter_estimates_accepted"])

    def test_multiple_branch_mismatches_are_ambiguous_not_clean(self):
        dss, model, nominal = self.context()
        telemetry = self.measure(dss, model)
        for index in (0, 62):
            for key in ("i_from_pu_rect", "i_to_pu_rect"):
                telemetry["three_phase_branch_currents"][index][key][0][0] += .1
        result = screen_measurements(telemetry, nominal)
        self.assertTrue(result["anomaly_detected"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["classification"], "ambiguous")
        self.assertIsNone(result["hif_candidate"])

    def test_nominal_capture_rejects_faulted_or_redistributed_context(self):
        dss, model, _ = self.context()
        redistribute_load(dss, model["registry"], bus=12, delta=.2)
        with self.assertRaisesRegex(ValueError, "unchanged registry PQ"):
            capture_nominal_model(dss, model["registry"], model["assumptions"])
        dss, model, _ = self.context()
        inject_midspan_hif(dss, model["registry"], model["assumptions"], branch_row0=62)
        with self.assertRaisesRegex(ValueError, "nominal branch status"):
            capture_nominal_model(dss, model["registry"], model["assumptions"])

    def test_missing_telemetry_and_invalid_noise_are_rejected(self):
        dss, model, nominal = self.context()
        telemetry = self.measure(dss, model)
        incomplete = copy.deepcopy(telemetry)
        incomplete["three_phase_branch_currents"].pop()
        with self.assertRaises(ValueError):
            screen_measurements(incomplete, nominal)
        for sigma in (0, -1, float("nan"), float("inf")):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                screen_measurements(telemetry, nominal, config=DiagnosticConfig(current_sigma_pu=sigma))


if __name__ == "__main__":
    unittest.main()
