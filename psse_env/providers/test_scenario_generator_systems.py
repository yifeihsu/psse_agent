from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcp_server.matpower_server import _load_python_case
from psse_env.dagger.release_factories import deterministic_case_loader
from psse_env.dagger.splits import physical_root_fingerprint
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.balanced_corpus import build_balanced_corpus
from psse_env.providers.matpower import _render_matpower_case
from psse_env.providers.scenario_generator import Round0ScenarioGenerator


class BalancedSystemScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.corpus = build_balanced_corpus(
            cls.root / "corpus", system="case57", seed=20260910,
            counts={"no_error": 2, "measurement_error": 4, "parameter_error": 2},
            num_scans=3,
        )
        cls.raw = [json.loads(line) for line in Path(cls.corpus["corpus_path"]).read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def generator(self, **kwargs):
        return Round0ScenarioGenerator(
            system="case57", corpus_path=self.corpus["corpus_path"],
            balanced_artifact_dir=self.corpus["artifact_dir"],
            derived_case_dir=self.root / "derived", admission_mode="physical",
            min_measurement_error_sigma=10.0, **kwargs,
        )

    def test_requires_explicit_target_corpus_and_rejects_extended_physics(self):
        with self.assertRaisesRegex(ValueError, "explicit fresh corpus"):
            Round0ScenarioGenerator(system="case57")
        for family in ("topology", "hif", "harmonic", "three_phase_unbalance"):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "Unsupported families"):
                self.generator().build({family: 1})

    def test_fresh_source_checks_case_hash_covariance_and_dimensions(self):
        generator = self.generator()
        for key, value in (("network_case", "case14"), ("base_case_hash", "wrong"),
                           ("sigmas", {"vm": 0.02}), ("z_obs", [0.0] * 122),
                           ("physical_validation", {"passed": False})):
            row = copy.deepcopy(self.raw[0])
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generator._validate_source_row(row)

    def test_physical_development_does_not_select_for_solver_or_teacher_success(self):
        generator = self.generator()
        with patch("psse_env.providers.scenario_generator._wls_json", side_effect=AssertionError("WLS admission used")), \
             patch("psse_env.providers.scenario_generator._param_correction_json", side_effect=AssertionError("teacher admission used")):
            rows = generator.build(dict.fromkeys(("no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter"), 1))
        self.assertEqual(len(rows), 5)
        self.assertFalse(generator.report()["parameter_ranking_admission"]["enforced"])
        for row in rows:
            self.assertEqual(row["network_case"], "case57")
            self.assertEqual(len(row["measurements"]), 491)
            self.assertEqual(row["source_tier"], "physics_synthesized_balanced")
            self.assertIn("source_realization_id", row)

    def test_high_numbered_meter_and_parent_lineage_survive_envelope(self):
        generator = self.generator()
        raw = copy.deepcopy(next(row for row in self.raw if row["scenario"] == "measurement_error"))
        raw["z_obs"] = list(raw["z_true"])
        raw["z_obs"][490] += 0.25
        raw["label"] = {"index": 490, "channel": "Qt", "subtype": "single_gross_outlier"}
        scenario = generator._measurement_scenario(raw, 1)
        scenario.update(error_cardinality=1, source_tier="physics_synthesized_balanced",
                        source_realization_id=raw["source_realization_id"],
                        base_case_version=generator.system.base_case_hash,
                        scenario_admission_mode="physical")
        envelope = partition_release_scenario_v1(scenario, split="development")
        self.assertEqual(envelope["audit"]["truth"]["true_measurement_errors"][0]["index"], 490)
        self.assertEqual(envelope["grouping"]["source_realization_id"], raw["source_realization_id"])
        for key in ("source_realization_id", "base_case_version", "scenario_admission_mode"):
            self.assertNotIn(key, envelope["execution"])
            self.assertNotIn(key, envelope["execution"].get("metadata", {}))

    def test_parent_group_stable_but_exact_fingerprint_changes_for_variant(self):
        generator = self.generator()
        original = copy.deepcopy(self.raw[0])
        variant = copy.deepcopy(original)
        variant["z_obs"][300] += 0.5
        self.assertEqual(generator._source_physical_digest(original), generator._source_physical_digest(variant))
        self.assertNotEqual(physical_root_fingerprint(original), physical_root_fingerprint(variant))

    def test_parameter_artifact_rejects_undeclared_shunt_change(self):
        generator = self.generator()
        row = copy.deepcopy(next(row for row in self.raw if row["scenario"] == "parameter_error"))
        case = _load_python_case(row["parameter_error_case_path"])
        case["bus"][0, 5] += 123.0
        path = Path(self.corpus["artifact_dir"]) / "cases_parameter_error" / "tampered.m"
        path.write_text(_render_matpower_case(case, "tampered"), encoding="utf-8")
        row["parameter_error_case_path"] = str(path)
        with self.assertRaisesRegex(ValueError, "undeclared bus or generator"):
            generator._parameter_scenario(row, 1)

    def test_canonical_case_available_to_truth_audit_loader(self):
        case = deterministic_case_loader("case57")
        self.assertEqual(case["bus"].shape[0], 57)
        self.assertEqual(case["branch"].shape[0], 80)


if __name__ == "__main__":
    unittest.main()
