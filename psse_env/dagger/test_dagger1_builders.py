from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import psse_env.dagger.build_dagger1_aggregate as aggregate_module
import psse_env.dagger.build_dagger1_scenarios as scenario_module
from psse_env.dagger.collect_dagger1 import (
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_ENV_FACTORY_SPEC,
    DEFAULT_EVALUATION_POLICY,
    DEFAULT_FORBIDDEN_SUITE,
    DEFAULT_POLICY_FACTORY_SPEC,
    frozen_physical_roots,
)
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.sft.provenance import file_sha256


SOURCE_STATE = {
    "source_commit": "a" * 40,
    "release_eligible_source": True,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_d0_inputs(root: Path) -> Path:
    aggregate_dir = root / "d0"
    raw_path = aggregate_dir / "aggregate.raw.jsonl"
    validation_path = aggregate_dir / "aggregate.validation.jsonl"
    test_path = aggregate_dir / "aggregate.test.jsonl"
    _write_jsonl(
        raw_path,
        [
            {
                "dataset_split": "train",
                "production_label_eligible": True,
                "physical_root_fingerprint": "d0-root",
                "example_id": "d0-example",
            }
        ],
    )
    _write_jsonl(validation_path, [{"example_id": "validation-example"}])
    _write_jsonl(test_path, [{"example_id": "test-example"}])
    provenance = {
        "release_eligible": True,
        "generation_descriptor": {"source_state": SOURCE_STATE},
        "generation_provenance_id": "d0-provenance",
        "dataset_hashes": {
            path.name: file_sha256(path)
            for path in (raw_path, validation_path, test_path)
        },
    }
    (aggregate_dir / "aggregate.generation_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate_dir


class _FakeTrainGenerator:
    def __init__(
        self,
        *,
        seed: int,
        source_partition: str,
        parameter_ranking_dominance_threshold: float,
    ) -> None:
        self.seed = seed
        if source_partition != "train":
            raise AssertionError(source_partition)
        if (
            parameter_ranking_dominance_threshold
            != BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ):
            raise AssertionError(parameter_ranking_dominance_threshold)

    def build(self, plan: dict[str, int]) -> list[dict]:
        candidates = {
            "multi_measurement": [
                {
                    "root": "multi-with-scans",
                    "parameter_scans": {"z_scans": [[1.0]]},
                },
                {"root": "multi-fresh", "parameter_scans": {}},
            ],
            "parameter": [
                {"root": "d0-root", "parameter_scans": {}},
                {"root": "parameter-fresh", "parameter_scans": {}},
            ],
            "measurement+parameter": [
                {"root": "frozen-root", "parameter_scans": {}},
                {"root": "mixed-fresh", "parameter_scans": {}},
            ],
        }
        rows: list[dict] = []
        for family, count in sorted(plan.items()):
            selected = candidates[family]
            if len(selected) != count:
                raise AssertionError((family, count))
            for index, row in enumerate(selected):
                rows.append(
                    {
                        **copy.deepcopy(row),
                        "scenario_id": f"{family}-{index}",
                        "scenario_family": family,
                        "error_cardinality": (
                            2 if family == "measurement+parameter" else 1
                        ),
                    }
                )
        return rows

    def report(self) -> dict:
        return {
            "seed": self.seed,
            "source_partition": {"enabled": True, "selected": "train"},
            "parameter_ranking_admission": {
                "contract": "distinct_line_abs_lambda_dominance_v1",
                "enforced": True,
                "threshold": BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
            },
        }


def _fake_partition(row: dict, *, split: str) -> dict:
    return {
        "scenario_schema_version": 1,
        "execution": {
            "scenario_id": row["scenario_id"],
            "case": "case14",
            "measurements": [],
            "metadata": {"parameter_scans": row["parameter_scans"]},
        },
        "audit": {
            "truth": {
                "truth_complete": True,
                "clean_measurements": [],
            },
            "evaluation_intervention": {
                "intervention_schema_version": 1,
                "kind": "none",
            },
        },
        "grouping": {
            "root_scenario_id": row["scenario_id"],
            "physical_root_fingerprint": row["root"],
            "scenario_family": row["scenario_family"],
            "error_cardinality": row["error_cardinality"],
            "case_id": "case14",
            "split": split,
            "source_tier": "test",
        },
    }


class Dagger1ScenarioBuilderTests(unittest.TestCase):
    def test_public_builder_is_deterministic_bound_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            plan = {
                "measurement+parameter": 1,
                "multi_measurement": 1,
                "parameter": 1,
            }
            outputs: list[tuple[Path, Path, dict]] = []
            with (
                patch.object(
                    scenario_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                patch.object(
                    scenario_module,
                    "Round0ScenarioGenerator",
                    _FakeTrainGenerator,
                ),
                patch.object(
                    scenario_module,
                    "partition_release_scenario_v1",
                    side_effect=_fake_partition,
                ),
                patch.object(
                    scenario_module,
                    "frozen_physical_roots",
                    return_value=frozenset({"frozen-root"}),
                ),
            ):
                for run in ("one", "two"):
                    output = root / run / "scenarios.json"
                    report = root / run / "generator.json"
                    manifest = scenario_module.build_dagger1_scenarios(
                        d0_aggregate_dir=d0_dir,
                        output=output,
                        generator_report_path=report,
                        seed=20260720,
                        plan=plan,
                        candidate_multiplier=2,
                    )
                    outputs.append((output, report, manifest))

                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    scenario_module.build_dagger1_scenarios(
                        d0_aggregate_dir=d0_dir,
                        output=outputs[0][0],
                        generator_report_path=outputs[0][1],
                        seed=20260720,
                        plan=plan,
                        candidate_multiplier=2,
                    )

            first_output, first_report, first_manifest = outputs[0]
            second_output, second_report, second_manifest = outputs[1]
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["selected_count_by_family"], plan)
            self.assertEqual(first_manifest["scenario_count"], 3)
            self.assertEqual(first_manifest["physical_root_count"], 3)
            self.assertEqual(first_manifest["filtered_protected_root_count"], 2)
            self.assertEqual(
                first_manifest[
                    "filtered_multi_measurement_with_parameter_scans_root_count"
                ],
                1,
            )
            self.assertEqual(first_manifest["protected_root_overlap"], [])
            selected_roots = {
                row["grouping"]["physical_root_fingerprint"]
                for row in json.loads(first_output.read_text(encoding="utf-8"))
            }
            self.assertEqual(
                selected_roots,
                {"multi-fresh", "parameter-fresh", "mixed-fresh"},
            )


class Dagger1AggregateBuilderTests(unittest.TestCase):
    def _write_d1_inputs(self, root: Path) -> tuple[Path, Path]:
        d1_path = root / "d1.jsonl"
        _write_jsonl(
            d1_path,
            [
                {
                    "physical_root_fingerprint": "d1-root",
                    "production_label_eligible": True,
                    "example_id": "d1-example",
                }
            ],
        )
        scenario_manifest = root / "scenarios.json.manifest.json"
        scenario_manifest.write_text("{}\n", encoding="utf-8")
        expert_source = inspect.getsourcefile(ExpertPolicyOracle)
        if expert_source is None:
            raise AssertionError("ExpertPolicyOracle must be inspectable")
        manifest = {
            "release_evidence_eligible": False,
            "training_eligible": True,
            "output_sha256": file_sha256(d1_path),
            "scenario_builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
            "scenario_manifest": str(scenario_manifest),
            "scenario_manifest_sha256": file_sha256(scenario_manifest),
            "source_state": SOURCE_STATE,
            "factory_identities": {
                "environment": {
                    "import_spec": DEFAULT_ENV_FACTORY_SPEC,
                    "source_sha256": aggregate_module._source_hash_for_import_spec(
                        DEFAULT_ENV_FACTORY_SPEC
                    ),
                },
                "learner_policy": {
                    "import_spec": DEFAULT_POLICY_FACTORY_SPEC,
                    "source_sha256": aggregate_module._source_hash_for_import_spec(
                        DEFAULT_POLICY_FACTORY_SPEC
                    ),
                },
                "expert_oracle": {
                    "source_sha256": file_sha256(expert_source),
                },
            },
            "release_environment_contract": {
                "parameter_ranking_dominance_threshold": (
                    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
                ),
                "production_dataset_mode": True,
                "max_steps": 24,
            },
            "forbidden_suite_sha256": file_sha256(DEFAULT_FORBIDDEN_SUITE),
            "evaluation_policy_sha256": file_sha256(DEFAULT_EVALUATION_POLICY),
        }
        manifest_path = d1_path.with_suffix(d1_path.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return d1_path, manifest_path

    @staticmethod
    def _rebind_d1_manifest(d1_path: Path, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"] = file_sha256(d1_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_public_builder_binds_inputs_rejects_tamper_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            d1_path, d1_manifest_path = self._write_d1_inputs(root)

            def fake_view(d0_rows, d1_rows, **kwargs):
                del kwargs
                self.assertEqual(
                    {row["physical_root_fingerprint"] for row in d0_rows},
                    {"d0-root"},
                )
                self.assertEqual(
                    {row["physical_root_fingerprint"] for row in d1_rows},
                    {"d1-root"},
                )
                return [*copy.deepcopy(d0_rows), *copy.deepcopy(d1_rows)], {
                    "release_ready": True,
                    "source_allocation": {"passed": True},
                }

            def fake_chat(rows, *, protocol):
                self.assertEqual(protocol, "canonical")
                return [
                    {
                        "example_id": row.get("example_id"),
                        "physical_root_fingerprint": row.get(
                            "physical_root_fingerprint"
                        ),
                    }
                    for row in rows
                ]

            output_dir = root / "round1"
            with (
                patch.object(
                    aggregate_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                patch.object(
                    aggregate_module,
                    "build_dagger1_training_view",
                    side_effect=fake_view,
                ),
                patch.object(
                    aggregate_module,
                    "examples_to_chat_sft",
                    side_effect=fake_chat,
                ),
                patch.object(
                    aggregate_module,
                    "tool_schema_hashes",
                    return_value=["schema-hash"],
                ),
            ):
                report = aggregate_module.build_round1_aggregate(
                    d0_aggregate_dir=d0_dir,
                    d1_path=d1_path,
                    d1_manifest_path=d1_manifest_path,
                    output_dir=output_dir,
                    seed=20260719,
                    size=None,
                    d1_share=0.25,
                    minimum_d1_share=0.20,
                    maximum_d1_share=0.30,
                    max_duplicate_count=2,
                    max_rows_per_root=8,
                )
                self.assertTrue(report["release_eligible"])
                self.assertTrue(
                    json.loads(
                        (output_dir / "aggregate.generation_provenance.json").read_text(
                            encoding="utf-8"
                        )
                    )["release_eligible"]
                )
                self.assertTrue((output_dir / "SHA256SUMS").is_file())

                original_d1 = d1_path.read_bytes()
                original_manifest = d1_manifest_path.read_bytes()
                with d1_path.open("ab") as handle:
                    handle.write(b"{}\n")
                with self.assertRaisesRegex(ValueError, "manifest hash"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "tampered",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                d1_path.write_bytes(original_d1)

                frozen_root = sorted(
                    frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE)
                )[0]
                _write_jsonl(
                    d1_path,
                    [
                        {
                            "physical_root_fingerprint": frozen_root,
                            "production_label_eligible": True,
                            "example_id": "forged-frozen-root",
                        }
                    ],
                )
                self._rebind_d1_manifest(d1_path, d1_manifest_path)
                with self.assertRaisesRegex(ValueError, "frozen evaluation suite"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "forged-frozen",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )

                _write_jsonl(
                    d1_path,
                    [
                        {
                            "physical_root_fingerprint": "d1-root",
                            "production_label_eligible": True,
                            "example_id": "forged-private-truth",
                            "labels": {
                                "nested": {
                                    "true_parameter_errors": [
                                        {"line_index1": 1}
                                    ]
                                }
                            },
                        }
                    ],
                )
                self._rebind_d1_manifest(d1_path, d1_manifest_path)
                with self.assertRaisesRegex(RuntimeError, "private oracle truth"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "forged-truth",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )

                d1_path.write_bytes(original_d1)
                d1_manifest_path.write_bytes(original_manifest)

                occupied_dir = root / "occupied"
                occupied_dir.mkdir()
                (occupied_dir / "aggregate.train_view.jsonl").write_text(
                    "do-not-overwrite\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(FileExistsError, "already exist"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=occupied_dir,
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                self.assertEqual(
                    (occupied_dir / "aggregate.train_view.jsonl").read_text(
                        encoding="utf-8"
                    ),
                    "do-not-overwrite\n",
                )


if __name__ == "__main__":
    unittest.main()
