"""Physical-source regression checks, distinct from teacher admission."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pypower.idx_gen import PG, PMAX
from pypower.idx_brch import RATE_A

from Transmission import generate_measurements as measurement_builder
from mcp_server.matpower_server import _load_python_case
from psse_env.providers import balanced_corpus as corpus
from psse_env.systems import resolve_system


class BalancedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.manifest = corpus.build_balanced_corpus(
            Path(cls.workspace.name) / "case57", system="case57", seed=20260910,
            counts={"no_error": 1, "measurement_error": 2, "parameter_error": 1}, num_scans=3,
        )
        cls.rows = [json.loads(line) for line in Path(cls.manifest["corpus_path"]).read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_case57_sources_cover_both_meter_subtypes_and_complete_physical_admission(self):
        self.assertTrue(self.manifest["complete"])
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(
            {row["label"]["subtype"] for row in self.rows if row["scenario"] == "measurement_error"},
            {"single_gross_outlier", "multi_gross_outliers"},
        )
        self.assertFalse(self.manifest["clean_noise_chi_square_filtered"])
        self.assertFalse(self.manifest["raw_admission_uses_expert_or_wls"])
        spec = resolve_system("case57")
        for row in self.rows:
            self.assertEqual(len(row["z_obs"]), 491)
            self.assertEqual(row["network_case"], "case57")
            self.assertEqual(row["base_case_hash"], spec.base_case_hash)
            self.assertEqual(row["sigmas"], {"vm": 0.001, "inj": 0.01, "flow": 0.01})
            self.assertTrue(row["physical_validation"]["passed"])
            solved = _load_python_case(row["physical_case_path"])
            np.testing.assert_allclose(corpus.compute_measurements_pu(solved), row["z_true"], atol=1e-8)
        self.assertEqual(len({row["source_realization_id"] for row in self.rows}), 4)

    def test_parameter_model_is_changed_true_network_and_scans_share_one_truth(self):
        row = next(row for row in self.rows if row["scenario"] == "parameter_error")
        base = resolve_system("case57").load_case()
        truth = _load_python_case(row["parameter_error_case_path"])
        branch_row = row["label"]["line_row"]
        self.assertFalse(np.array_equal(truth["branch"][branch_row, 2:4], base["branch"][branch_row, 2:4]))
        others = np.arange(len(base["branch"])) != branch_row
        np.testing.assert_allclose(truth["branch"][others], base["branch"][others])
        self.assertEqual(row["configured_case_path"], resolve_system("case57").case_path)
        self.assertNotIn("initial_states", row)
        self.assertEqual(len(row["z_scans"]), 3)
        self.assertTrue(all(len(scan) == 491 for scan in row["z_scans"]))
        self.assertNotEqual(row["z_scans"][0], row["z_scans"][1])
        # Independent draws should fluctuate around the same noiseless OPF state.
        scan_errors = (np.asarray(row["z_scans"]) - np.asarray(row["z_true"])) / resolve_system("case57").measurement_sigma()
        self.assertLess(abs(float(scan_errors.mean())), 0.15)
        self.assertGreater(float(scan_errors.std()), 0.8)
        self.assertLess(float(scan_errors.std()), 1.2)

    def test_failed_solve_records_attempts_without_calling_it_infeasibility(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(corpus, "solve_ac_opf", return_value=None) as solver:
            manifest = corpus.build_balanced_corpus(
                directory, system="case57", counts={"parameter_error": 1}, max_attempt_multiplier=2,
            )
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["attempted"]["parameter_error"], 2)
            self.assertEqual(manifest["rejected"]["parameter_error"], 2)
            self.assertEqual(solver.call_count, 2)
            self.assertEqual({entry["reason"] for entry in manifest["rejections"]}, {"opf_nonconvergence"})
            self.assertEqual(Path(manifest["corpus_path"]).read_text(), "")
            self.assertEqual(len(Path(manifest["rejection_path"]).read_text().splitlines()), 2)

    def test_clean_draws_are_preserved_without_statistical_admission_on_case14(self):
        # A gross, deliberately substituted noise draw must not be discarded by
        # physical-source admission: that would bias false-alarm evaluation.
        with tempfile.TemporaryDirectory() as directory, patch.object(
            corpus, "base_gaussian_noise", side_effect=lambda z, *_args: np.full_like(z, 0.1),
        ):
            manifest = corpus.build_balanced_corpus(directory, system="case14", counts={"no_error": 1})
            self.assertTrue(manifest["complete"])
            row = json.loads(Path(manifest["corpus_path"]).read_text())
            self.assertEqual(len(row["z_obs"]), 122)
            np.testing.assert_allclose(np.asarray(row["z_obs"]) - row["z_true"], 0.1)

    def test_unrequested_physics_and_invalid_configuration_fail_before_solving(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(corpus, "solve_ac_opf") as solver:
            for kwargs in ({"counts": {"topology_error": 1}}, {"counts": {"no_error": -1}}, {"num_scans": 0}, {"load_scale_range": (1.0, 0.8)}):
                with self.assertRaises(ValueError):
                    corpus.build_balanced_corpus(directory, **kwargs)
            solver.assert_not_called()

    def test_physics_checker_rejects_channel_corruption_and_operating_bound_violation(self):
        row = self.rows[0]
        solved = _load_python_case(row["physical_case_path"])
        solved["success"] = True
        telemetry = np.asarray(row["z_true"])
        corpus.validate_balanced_solution(solved, telemetry)
        corrupted = telemetry.copy()
        corrupted[57] += 0.1
        with self.assertRaisesRegex(corpus.PhysicalAdmissionError, "telemetry_physics_mismatch"):
            corpus.validate_balanced_solution(solved, corrupted)
        invalid = copy.deepcopy(solved)
        invalid["gen"][0, PG] = invalid["gen"][0, PMAX] + 1.0
        with self.assertRaisesRegex(corpus.PhysicalAdmissionError, "operating_bounds_violated"):
            corpus.validate_balanced_solution(invalid, telemetry)

    def test_unlimited_thermal_case_opf_preserves_model_and_matches_nonbinding_reference(self):
        base = corpus.scale_loads(resolve_system("case14").load_case(), 0.9)
        self.assertTrue(np.all(base["branch"][:, RATE_A] == 0))
        solved = corpus.solve_ac_opf(base)
        self.assertIsNotNone(solved)
        self.assertTrue(np.all(solved["branch"][:, RATE_A] == 0))
        self.assertTrue(np.all(base["branch"][:, RATE_A] == 0))
        # A test-only distant nonbinding limit exercises upstream's ordinary
        # constrained-array path, independently of our empty-array workaround.
        reference = copy.deepcopy(base)
        reference["branch"][:, RATE_A] = 9900.0
        solved_reference = corpus.solve_ac_opf(reference)
        self.assertIsNotNone(solved_reference)
        np.testing.assert_allclose(
            corpus.compute_measurements_pu(solved), corpus.compute_measurements_pu(solved_reference), atol=1e-4,
        )

    def test_empty_constraint_solver_hooks_restore_after_exception(self):
        solver_module = importlib.import_module("pypower.pipsopf_solver")
        constraint, hessian = solver_module.opf_consfcn, solver_module.opf_hessfcn
        with patch.object(measurement_builder, "runopf", side_effect=RuntimeError("synthetic solver failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic solver failure"):
                corpus.solve_ac_opf(resolve_system("case14").load_case())
        self.assertIs(solver_module.opf_consfcn, constraint)
        self.assertIs(solver_module.opf_hessfcn, hessian)


if __name__ == "__main__":
    unittest.main()
