from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psse_env.dagger import ieee57_parents as parents
from psse_env.systems import resolve_system


class FakeGenerator:
    def __init__(self, **kwargs):
        self.options = kwargs

    def build(self, plan):
        family = next(iter(plan))
        records = [json.loads(line) for line in Path(self.options["corpus_path"]).read_text().splitlines()]
        raw_family = "parameter_error" if "parameter" in family else "measurement_error" if "measurement" in family else "no_error"
        raw = next((item for item in records if item["scenario"] == raw_family), None)
        if raw is None:
            return []
        return [{
            "scenario_id": f"scenario_{raw['source_realization_id']}_{family}",
            "root_scenario_id": f"root_{raw['source_realization_id']}",
            "scenario_family": family, "error_cardinality": int(family != "no_error"),
            "case": "case57", "measurements": raw["z_obs"], "clean_case": "case57",
            "clean_measurements": raw["z_obs"], "source_realization_id": raw["source_realization_id"],
            "true_measurement_errors": [], "true_parameter_errors": [], "true_topology_errors": [],
            "network_case": "case57", "source_tier": "physics_synthesized_balanced",
        }]

    def report(self):
        return {"admission_mode": self.options["admission_mode"]}


def fake_corpus(output, **kwargs):
    output = Path(output)
    root = output.parents[2]
    # This assertion runs at the first physical/noise generation boundary.
    assert (root / "parent_plan.json").is_file()
    assert (root / "parent_plan.sha256").is_file()
    plan_bytes = (root / "parent_plan.json").read_bytes()
    assert hashlib.sha256(plan_bytes).hexdigest() == (root / "parent_plan.sha256").read_text().strip()
    scale = kwargs["load_scale_range"][0]
    assert kwargs["load_scale_range"] == (scale, scale)
    output.mkdir(parents=True)
    records = []
    for family, count in kwargs["counts"].items():
        for index in range(count):
            source = f"source_{kwargs['seed']}_{family}_{index}"
            z_true = [scale + (0.001 if family == "parameter_error" else 0.0)] * 491
            records.append({
                "id": source, "source_realization_id": source, "scenario": family,
                "base_case_hash": resolve_system("case57").base_case_hash,
                "op_point": {"load_scale": scale}, "z_true": z_true,
                "z_obs": [value + 0.00001 for value in z_true],
            })
    path = output / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return {"corpus_path": str(path), "artifact_dir": str(output), "accepted": kwargs["counts"], "rejected": {}, "complete": True}


class ParentAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "pilot"
        self.builder = patch.object(parents, "build_balanced_corpus", side_effect=fake_corpus).start()
        self.addCleanup(patch.stopall)
        patch.object(parents, "Round0ScenarioGenerator", FakeGenerator).start()

    def generate(self, **kwargs):
        return parents.generate_parent_assigned_scenarios(self.output, **kwargs)

    def test_plan_precedes_generation_and_all_splits_are_disjoint(self):
        manifest, scenarios = self.generate()
        self.assertTrue(manifest["complete"])
        self.assertEqual(len(scenarios), 15)
        self.assertEqual(manifest["raw_admitted_source_count"], 30)
        self.assertEqual(manifest["raw_support_sources_without_scenario"], 15)
        self.assertEqual(len({row["grouping"]["source_realization_id"] for row in scenarios}), 15)
        self.assertEqual(manifest["independence_validation"]["scenario_count_by_split"], {"train": 5, "validation": 5, "test": 5})
        self.assertFalse(manifest["requested_scenario_population_selected_on_teacher_outcomes"])
        for call in self.builder.call_args_list:
            self.assertEqual(call.kwargs["num_scans"], 3)
        for row in scenarios:
            group = row["grouping"]
            self.assertNotEqual(group["source_realization_id"], group["original_source_realization_id"])
            self.assertNotEqual(group["physical_root_fingerprint"], group["parent_construction_fingerprint"])
            self.assertEqual(group["physical_root_fingerprint"], group["original_physical_root_fingerprint"])
            self.assertNotIn("parent_plan_sha256", row["execution"])

    def test_reproducible_plans_are_path_independent(self):
        manifest, scenarios = self.generate(seed=22, splits=("train",))
        other = Path(self.directory.name) / "other"
        other_manifest, other_rows = parents.generate_parent_assigned_scenarios(other, seed=22, splits=("train",))
        self.assertEqual(manifest["parent_plan_sha256"], other_manifest["parent_plan_sha256"])
        self.assertEqual(scenarios, other_rows)
        self.assertFalse(Path(manifest["scenario_path"]).is_absolute())

    def test_parent_identity_does_not_include_alias_or_seed(self):
        base = resolve_system("case57").base_case_hash
        same = parents.parent_construction_fingerprint(base, 0.91)
        self.assertEqual(same, parents.parent_construction_fingerprint(base, 0.91))
        self.assertNotEqual(same, parents.parent_construction_fingerprint(base, 0.92))
        self.assertNotEqual(same, parents.parent_construction_fingerprint("different base", 0.91))

    def test_relabeling_a_scenario_fails(self):
        manifest, scenarios = self.generate(splits=("train",))
        altered = copy.deepcopy(scenarios)
        altered[0]["grouping"].update(split="test", dataset_split="test")
        with self.assertRaisesRegex(ValueError, "preassigned plan"):
            parents.validate_parent_assignment(self.output, manifest, altered)

    def test_plan_cannot_be_changed_after_assignment(self):
        manifest, scenarios = self.generate(splits=("train",))
        plan = self.output / "parent_plan.json"
        plan.write_text(plan.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "plan changed"):
            parents.validate_parent_assignment(self.output, manifest, scenarios)

    def test_raw_content_change_is_detected(self):
        manifest, scenarios = self.generate(splits=("train",))
        raw = self.output / "parents" / "parent_0000" / "corpus" / "samples.jsonl"
        record = json.loads(raw.read_text())
        record["z_true"][0] += 0.5
        raw.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "raw source content changed"):
            parents.validate_parent_assignment(self.output, manifest, scenarios)

    def test_source_lineage_and_measurement_shape_are_checked(self):
        manifest, scenarios = self.generate(splits=("train",))
        with self.subTest("source"):
            altered = copy.deepcopy(scenarios)
            altered[0]["grouping"]["original_source_realization_id"] = "renamed"
            with self.assertRaisesRegex(ValueError, "original source"):
                parents.validate_parent_assignment(self.output, manifest, altered)
        with self.subTest("dimension"):
            altered = copy.deepcopy(scenarios)
            altered[0]["execution"]["measurements"].pop()
            with self.assertRaisesRegex(ValueError, "491"):
                parents.validate_parent_assignment(self.output, manifest, altered)

    def test_failed_slot_retained_without_drawing_replacement_parent(self):
        def fail_first(output, **kwargs):
            if "parent_0000" in str(output):
                raise RuntimeError("physical solve failed")
            return fake_corpus(output, **kwargs)
        self.builder.side_effect = fail_first
        manifest, scenarios = self.generate(splits=("train",))
        self.assertFalse(manifest["complete"])
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(self.builder.call_count, 5)
        self.assertEqual(manifest["requested_parent_count"], 5)
        self.assertEqual(manifest["failed_or_missing_parent_slots"], ["parent_0000"])
        reports = json.loads((self.output / "slot_results.json").read_text())
        self.assertEqual(reports[0]["status"], "generation_failed")
        self.assertEqual(reports[0]["error"], "physical solve failed")

    def test_output_cannot_overwrite_previous_assignment(self):
        self.generate(splits=("train",))
        with self.assertRaises(FileExistsError):
            self.generate(splits=("test",))

    def test_invalid_inputs_fail_before_directory_creation(self):
        for kwargs in ({"per_family": 0}, {"per_family": True}, {"seed": True}, {"splits": ()}, {"splits": "train"}, {"splits": ("train", "train")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
