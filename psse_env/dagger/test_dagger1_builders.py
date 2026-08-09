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
from psse_env.dagger.build_dagger1_development_holdout import (
    DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
    DAGGER1_DEVELOPMENT_SPLIT,
    DAGGER1_DEVELOPMENT_SUITE_NAME,
    DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
)
from psse_env.dagger.collect_dagger1 import (
    DAGGER1_COLLECTION_SELECTION_CONTRACT,
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_ENV_FACTORY_SPEC,
    DEFAULT_EVALUATION_POLICY,
    DEFAULT_FORBIDDEN_SUITE,
    DEFAULT_POLICY_FACTORY_SPEC,
    dagger1_production_row_target_contract,
    frozen_physical_roots,
)
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
)
from psse_env.dagger.rollout_collector import (
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    summarize_dagger1_offline_teacher_target_quarantine,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.sft.provenance import file_sha256, stable_json_sha256


SOURCE_STATE = {
    "source_commit": "a" * 40,
    "release_eligible_source": True,
}
LEARNER_REVISION = "b" * 64
LEARNER_MODEL_ID = "/scratch/reviewed-bc0/lora"


def _passed_offline_teacher_target_audit() -> dict:
    return {
        "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
        "passed": True,
        "action_class": "read_only",
        "checks": {"observable_evidence_gate_passed": True},
        "reason_codes": [],
    }


def _d1_training_row(**updates) -> dict:
    row = {
        "physical_root_fingerprint": "d1-root",
        "production_label_eligible": True,
        "example_id": "d1-example",
        "supervision_policy": DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
        "collection_role": "training",
        "state_origin": "learner_policy",
        "recovery_stratum": "post_failure_no_candidate",
        "preferred_action": {
            "tool": "run_wls",
            "arguments": {"state_id": "active"},
        },
        "observable_rank_one_target_proof": {"passed": True},
        "labels": {
            "training_decision_evidence_verified": True,
            "collection_training_eligible": True,
            "collection_disposition": "selected_for_round1_training",
        },
        "offline_teacher_target_audit": (
            _passed_offline_teacher_target_audit()
        ),
        "collection_training_eligible": True,
        "collection_disposition": "selected_for_round1_training",
    }
    row.update(copy.deepcopy(updates))
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mock_production_selection_report(rows: list[dict]) -> dict:
    example_ids = [str(row.get("example_id") or "") for row in rows]
    return {
        "contract": DAGGER1_COLLECTION_SELECTION_CONTRACT,
        "candidate_rows": len(rows),
        "candidate_example_id_set_sha256": stable_json_sha256(
            sorted(example_ids)
        ),
        "target_min_rows": 300,
        "target_max_rows": 600,
        "selected_rows": len(rows),
        "selected_example_id_sequence_sha256": stable_json_sha256(example_ids),
        "passed": True,
    }


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
    manifest_path = aggregate_dir / "aggregate.manifest.json"
    manifest_path.write_text(
        json.dumps({"episode_audits": []}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generation_descriptor = {"source_state": SOURCE_STATE}
    provenance = {
        "release_eligible": True,
        "generation_descriptor": generation_descriptor,
        "generation_provenance_id": stable_json_sha256(generation_descriptor),
        "dataset_hashes": {
            path.name: file_sha256(path)
            for path in (raw_path, validation_path, test_path, manifest_path)
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


class _FakeDefaultPoolGenerator(_FakeTrainGenerator):
    def build(self, plan: dict[str, int]) -> list[dict]:
        if plan != scenario_module.DEFAULT_DAGGER1_CANDIDATE_REQUEST_PLAN:
            raise AssertionError(plan)
        rows: list[dict] = []
        for index in range(108):
            rows.append(
                {
                    "root": f"mixed-{index:03d}",
                    "parameter_scans": {},
                    "scenario_id": f"mixed-{index:03d}",
                    "scenario_family": "measurement+parameter",
                    "error_cardinality": 2,
                }
            )
        for cardinality, count in {2: 19, 3: 21, 4: 18, 5: 33}.items():
            for index in range(count):
                rows.append(
                    {
                        "root": f"multi-{cardinality}-{index:03d}",
                        "parameter_scans": {},
                        "scenario_id": f"multi-{cardinality}-{index:03d}",
                        "scenario_family": "multi_measurement",
                        "error_cardinality": cardinality,
                    }
                )
        for index in range(35):
            rows.append(
                {
                    "root": f"parameter-{index:03d}",
                    "parameter_scans": {},
                    "scenario_id": f"parameter-{index:03d}",
                    "scenario_family": "parameter",
                    "error_cardinality": 1,
                }
            )
        return rows


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
    def test_default_primary_plan_is_reviewed_120_root_cohort(self) -> None:
        self.assertEqual(
            scenario_module.DEFAULT_DAGGER1_ROOT_PLAN,
            {
                "measurement+parameter": 48,
                "multi_measurement": 48,
                "parameter": 24,
            },
        )

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
            self.assertEqual(
                first_manifest["d0_manifest_sha256"],
                file_sha256(d0_dir / "aggregate.manifest.json"),
            )
            selected_roots = {
                row["grouping"]["physical_root_fingerprint"]
                for row in json.loads(first_output.read_text(encoding="utf-8"))
            }
            self.assertEqual(
                selected_roots,
                {"multi-fresh", "parameter-fresh", "mixed-fresh"},
            )

    def test_default_pool_has_quota_bound_primary_reserve_and_holdback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            output = root / "scenarios.json"
            report = root / "generator.json"
            fake_predecessor_roots = {
                *(f"mixed-{index:03d}" for index in range(96)),
                *(f"parameter-{index:03d}" for index in range(24)),
                *(f"multi-2-{index:03d}" for index in range(16)),
                *(f"multi-3-{index:03d}" for index in range(6)),
                *(f"multi-3-{index:03d}" for index in range(9, 21)),
                *(f"multi-4-{index:03d}" for index in range(10)),
                *(f"multi-4-{index:03d}" for index in range(13, 18)),
                *(f"multi-5-{index:03d}" for index in range(16)),
                *(f"multi-5-{index:03d}" for index in range(19, 33)),
            }
            with (
                patch.object(
                    scenario_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                patch.object(
                    scenario_module,
                    "Round0ScenarioGenerator",
                    _FakeDefaultPoolGenerator,
                ),
                patch.object(
                    scenario_module,
                    "partition_release_scenario_v1",
                    side_effect=_fake_partition,
                ),
                patch.object(
                    scenario_module,
                    "frozen_physical_roots",
                    return_value=frozenset(),
                ),
                patch.object(
                    scenario_module,
                    "DAGGER1_PREDECESSOR_TRAINING_ROOT_SET_SHA256",
                    stable_json_sha256(sorted(fake_predecessor_roots)),
                ),
            ):
                manifest = scenario_module.build_dagger1_scenarios(
                    d0_aggregate_dir=d0_dir,
                    output=output,
                    generator_report_path=report,
                    seed=20260720,
                    plan=scenario_module.DEFAULT_DAGGER1_ROOT_PLAN,
                )
                with self.assertRaisesRegex(ValueError, "must remain 2"):
                    scenario_module.build_dagger1_scenarios(
                        d0_aggregate_dir=d0_dir,
                        output=root / "scaled.json",
                        generator_report_path=root / "scaled-report.json",
                        seed=20260720,
                        plan=scenario_module.DEFAULT_DAGGER1_ROOT_PLAN,
                        candidate_multiplier=3,
                    )

            scenarios = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(scenarios), 211)
            self.assertEqual(
                manifest["candidate_request_plan"],
                {
                    "measurement+parameter": 108,
                    "multi_measurement": 176,
                    "parameter": 48,
                },
            )
            self.assertEqual(
                manifest["fresh_candidate_inventory"]["multi_measurement"]
                ["error_cardinality"],
                {"2": 19, "3": 21, "4": 18, "5": 33},
            )
            self.assertEqual(
                manifest["fresh_candidate_inventory"]
                ["measurement+parameter"]["physical_root_count"],
                108,
            )
            self.assertEqual(
                manifest["primary_count_by_family"],
                {
                    "measurement+parameter": 48,
                    "multi_measurement": 48,
                    "parameter": 24,
                },
            )
            self.assertEqual(
                manifest["reserve_count_by_family"],
                {
                    "measurement+parameter": 60,
                    "multi_measurement": 31,
                    "parameter": 0,
                },
            )
            self.assertEqual(
                manifest["base_reserve_plan"],
                {
                    "measurement+parameter": 48,
                    "multi_measurement": 31,
                    "parameter": 0,
                },
            )
            self.assertEqual(
                manifest["topup_reserve_plan"],
                {
                    "measurement+parameter": 12,
                    "multi_measurement": 0,
                    "parameter": 0,
                },
            )
            topup_roots = manifest["topup_reserve_roots_by_family"][
                "measurement+parameter"
            ]
            self.assertEqual(
                topup_roots,
                [f"mixed-{index:03d}" for index in range(96, 108)],
            )
            self.assertEqual(manifest["topup_predecessor_overlap"], [])
            self.assertEqual(
                manifest["topup_reserve_physical_root_set_sha256"],
                stable_json_sha256(topup_roots),
            )
            self.assertEqual(
                manifest["reserve_multi_measurement_cardinality_inventory"],
                {"3": 12, "4": 5, "5": 14},
            )
            self.assertEqual(
                manifest[
                    "withheld_for_development_multi_measurement_cardinality_inventory"
                ],
                {"2": 3, "3": 3, "4": 3, "5": 3},
            )
            held_out = manifest["development_reserved_roots_by_family"][
                "multi_measurement"
            ]
            self.assertEqual(len(held_out), 12)
            self.assertTrue(set(held_out).isdisjoint({
                row["grouping"]["physical_root_fingerprint"]
                for row in scenarios
            }))
            self.assertEqual(
                [row["grouping"]["collection_order"] for row in scenarios],
                list(range(211)),
            )
            self.assertEqual(
                {
                    row["grouping"]["collection_priority"]
                    for row in scenarios
                    if row["grouping"]["collection_cohort"] == "primary"
                },
                {0},
            )
            reserve_priorities = {
                row["grouping"]["scenario_family"]: row["grouping"]
                ["collection_priority"]
                for row in scenarios
                if row["grouping"]["collection_cohort"] == "reserve"
            }
            self.assertEqual(
                reserve_priorities,
                {"multi_measurement": 1, "measurement+parameter": 2},
            )
            self.assertEqual(
                {
                    row["grouping"]["physical_root_fingerprint"]
                    for row in scenarios
                    if row["grouping"]["collection_subcohort"]
                    == "fresh_root_topup"
                },
                set(topup_roots),
            )

    def test_builder_rejects_d0_with_non_release_eligible_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            provenance_path = (
                d0_dir / "aggregate.generation_provenance.json"
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["generation_descriptor"]["source_state"][
                "release_eligible_source"
            ] = False
            provenance["generation_provenance_id"] = stable_json_sha256(
                provenance["generation_descriptor"]
            )
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(
                    scenario_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "source state is not release eligible"
                ),
            ):
                scenario_module.build_dagger1_scenarios(
                    d0_aggregate_dir=d0_dir,
                    output=root / "scenarios.json",
                    generator_report_path=root / "generator.json",
                    seed=20260720,
                    plan={"parameter": 1},
                )

    def test_builder_rejects_d0_with_mismatched_provenance_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            provenance_path = (
                d0_dir / "aggregate.generation_provenance.json"
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["generation_provenance_id"] = "f" * 64
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(
                    scenario_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                self.assertRaisesRegex(RuntimeError, "provenance ID"),
            ):
                scenario_module.build_dagger1_scenarios(
                    d0_aggregate_dir=d0_dir,
                    output=root / "scenarios.json",
                    generator_report_path=root / "generator.json",
                    seed=20260720,
                    plan={"parameter": 1},
                )

    def test_builder_rejects_missing_or_tampered_d0_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            manifest_path = d0_dir / "aggregate.manifest.json"
            manifest_path.unlink()
            with (
                patch.object(
                    scenario_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                self.assertRaisesRegex(FileNotFoundError, "manifest"),
            ):
                scenario_module.build_dagger1_scenarios(
                    d0_aggregate_dir=d0_dir,
                    output=root / "missing-scenarios.json",
                    generator_report_path=root / "missing-generator.json",
                    seed=20260720,
                    plan={"parameter": 1},
                )

            d0_dir = _write_d0_inputs(root)
            manifest_path.write_text("tampered\n", encoding="utf-8")
            with (
                patch.object(
                    scenario_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                self.assertRaisesRegex(RuntimeError, "manifest does not match"),
            ):
                scenario_module.build_dagger1_scenarios(
                    d0_aggregate_dir=d0_dir,
                    output=root / "tampered-scenarios.json",
                    generator_report_path=root / "tampered-generator.json",
                    seed=20260720,
                    plan={"parameter": 1},
                )


class Dagger1AggregateBuilderTests(unittest.TestCase):
    def _write_d1_inputs(self, root: Path) -> tuple[Path, Path]:
        d1_path = root / "d1.jsonl"
        _write_jsonl(
            d1_path,
            [_d1_training_row()],
        )
        all_output = (root / "d1.all.jsonl").resolve()
        all_rows = [_d1_training_row()]
        _write_jsonl(all_output, all_rows)
        scenario_input = root / "scenarios.json"
        scenario_input.write_text(
            json.dumps(
                [
                    {
                        "scenario_id": "d1-training-scenario",
                        "split": "dagger_train",
                        "physical_root_fingerprint": "d1-root",
                        "case": "case14",
                        "measurements": [],
                    }
                ],
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        scenario_manifest = root / "scenarios.json.manifest.json"
        scenario_manifest.write_text("{}\n", encoding="utf-8")
        d0_dir = root / "d0"
        if not (d0_dir / "aggregate.generation_provenance.json").is_file():
            d0_dir = _write_d0_inputs(root)
        d0_raw_path = d0_dir / "aggregate.raw.jsonl"
        d0_provenance_path = (
            d0_dir / "aggregate.generation_provenance.json"
        )
        d0_manifest_path = d0_dir / "aggregate.manifest.json"
        development_rows = []
        for family, count in DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items():
            for index in range(count):
                development_rows.append(
                    {
                        "scenario_schema_version": 1,
                        "execution": {"scenario_id": f"{family}-{index}"},
                        "audit": {"truth": {"truth_complete": True}},
                        "grouping": {
                            "split": DAGGER1_DEVELOPMENT_SPLIT,
                            "scenario_family": family,
                            "physical_root_fingerprint": (
                                f"development-{family}-{index}"
                            ),
                        },
                    }
                )
        development_path = (root / "development.json").resolve()
        development_path.write_text(
            json.dumps(
                {DAGGER1_DEVELOPMENT_SUITE_NAME: development_rows},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        development_generator_report_path = (
            root / "development.generator-report.json"
        ).resolve()
        development_generator_report_path.write_text(
            json.dumps({"source_partition": "train"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        development_roots = sorted(
            row["grouping"]["physical_root_fingerprint"]
            for row in development_rows
        )
        development_manifest = {
            "schema_version": 1,
            "builder_contract": DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
            "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
            "split": DAGGER1_DEVELOPMENT_SPLIT,
            "source_state": SOURCE_STATE,
            "source_bindings": {
                "psse_env/dagger/build_dagger1_development_holdout.py": (
                    file_sha256(
                        Path(__file__).resolve().parents[2]
                        / "psse_env"
                        / "dagger"
                        / "build_dagger1_development_holdout.py"
                    )
                )
            },
            "plan": DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
            "selected_count_by_family": DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
            "scenario_count": 30,
            "physical_root_count": 30,
            "root_set_sha256": {
                "development": stable_json_sha256(development_roots)
            },
            "training_eligible": False,
            "training_collection_eligible": False,
            "release_evidence_eligible": False,
            "promotion_evidence_eligible": False,
            "diagnostic_closed_loop_model_selection_eligible": True,
            "recovery_stratum_qualified_model_selection_eligible": False,
            "development_protected_overlap": {
                "d0": [],
                "frozen": [],
                "d1_training": [],
            },
            "output_sha256": file_sha256(development_path),
            "generator_report_sha256": file_sha256(
                development_generator_report_path
            ),
            "d1_training_scenarios_sha256": file_sha256(scenario_input),
            "d1_training_manifest_sha256": file_sha256(scenario_manifest),
            "d0_raw_sha256": file_sha256(d0_raw_path),
            "d0_generation_provenance_sha256": file_sha256(
                d0_provenance_path
            ),
            "d0_manifest_sha256": file_sha256(d0_manifest_path),
            "frozen_suite_sha256": file_sha256(DEFAULT_FORBIDDEN_SUITE),
            "evaluation_policy_sha256": file_sha256(
                DEFAULT_EVALUATION_POLICY
            ),
        }
        development_manifest_path = (
            root / "development.json.manifest.json"
        ).resolve()
        development_manifest_path.write_text(
            json.dumps(development_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expert_source = inspect.getsourcefile(ExpertPolicyOracle)
        if expert_source is None:
            raise AssertionError("ExpertPolicyOracle must be inspectable")
        manifest = {
            "release_evidence_eligible": False,
            "training_eligible": True,
            "output_sha256": file_sha256(d1_path),
            "all_output": str(all_output),
            "all_output_sha256": file_sha256(all_output),
            "all_output_row_count": len(all_rows),
            "collection_pass": "training",
            "visited_rows": len(all_rows),
            "output_rows": 1,
            "selected_recovery_row_count": 1,
            "candidate_recovery_rows": 1,
            "candidate_recovery_row_count": 1,
            "production_eligible_recovery_rows": 1,
            "production_row_target_contract": (
                dagger1_production_row_target_contract(
                    target_min_rows=300,
                    target_max_rows=600,
                )
            ),
            "deterministic_collection_selection": (
                _mock_production_selection_report(all_rows)
            ),
            "offline_teacher_target_quarantine_summary": (
                summarize_dagger1_offline_teacher_target_quarantine(all_rows)
            ),
            "model_id": LEARNER_MODEL_ID,
            "model_revision": LEARNER_REVISION,
            "learner_seed": {
                "role": "learner_seed_only",
                "collection_model_id": LEARNER_MODEL_ID,
                "collection_model_revision": LEARNER_REVISION,
                "adapter_tree_sha256": LEARNER_REVISION,
                "adapter_file_count": 7,
                "adapter_total_bytes": 1234,
            },
            "scenario_builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
            "input": str(scenario_input),
            "input_sha256": file_sha256(scenario_input),
            "scenario_manifest": str(scenario_manifest),
            "scenario_manifest_sha256": file_sha256(scenario_manifest),
            "d0_manifest_sha256": file_sha256(d0_manifest_path),
            "development_holdout": str(development_path),
            "development_holdout_sha256": file_sha256(development_path),
            "development_holdout_manifest": str(development_manifest_path),
            "development_holdout_manifest_sha256": file_sha256(
                development_manifest_path
            ),
            "development_holdout_generator_report": str(
                development_generator_report_path
            ),
            "development_holdout_generator_report_sha256": file_sha256(
                development_generator_report_path
            ),
            "development_holdout_root_count": len(development_roots),
            "development_physical_root_count": len(development_roots),
            "development_holdout_root_set_sha256": stable_json_sha256(
                development_roots
            ),
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
        rows = [
            json.loads(line)
            for line in d1_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        all_output = Path(manifest["all_output"])
        _write_jsonl(all_output, rows)
        manifest["output_sha256"] = file_sha256(d1_path)
        manifest["all_output_sha256"] = file_sha256(all_output)
        manifest["all_output_row_count"] = len(rows)
        manifest["visited_rows"] = len(rows)
        manifest["output_rows"] = len(rows)
        manifest["selected_recovery_row_count"] = len(rows)
        candidate_count = len(
            [row for row in rows if row.get("production_label_eligible") is True]
        )
        manifest["candidate_recovery_rows"] = candidate_count
        manifest["candidate_recovery_row_count"] = candidate_count
        manifest["production_eligible_recovery_rows"] = candidate_count
        manifest["deterministic_collection_selection"] = (
            _mock_production_selection_report(rows)
        )
        manifest["offline_teacher_target_quarantine_summary"] = (
            summarize_dagger1_offline_teacher_target_quarantine(rows)
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_round1_learner_seed_must_match_collection_aggregate_and_warm_start(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, manifest_path = self._write_d1_inputs(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_hash = file_sha256(manifest_path)
            binding = aggregate_module.validate_round1_learner_seed(
                manifest,
                collection_manifest_sha256=manifest_hash,
                initial_adapter_revision=LEARNER_REVISION.upper(),
            )
            aggregate_module.validate_round1_learner_seed(
                manifest,
                collection_manifest_sha256=manifest_hash,
                aggregate_learner_seed=binding,
                initial_adapter_revision=LEARNER_REVISION,
            )

            with self.assertRaisesRegex(
                ValueError, "INITIAL_ADAPTER_REVISION differs"
            ):
                aggregate_module.validate_round1_learner_seed(
                    manifest,
                    collection_manifest_sha256=manifest_hash,
                    aggregate_learner_seed=binding,
                    initial_adapter_revision="c" * 64,
                )
            with self.assertRaisesRegex(
                ValueError, "aggregate learner_seed differs"
            ):
                aggregate_module.validate_round1_learner_seed(
                    manifest,
                    collection_manifest_sha256=manifest_hash,
                    aggregate_learner_seed={
                        **binding,
                        "adapter_tree_sha256": "c" * 64,
                    },
                    initial_adapter_revision=LEARNER_REVISION,
                )

            forged = copy.deepcopy(manifest)
            forged["learner_seed"]["collection_model_revision"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "one exact 64-hex"):
                aggregate_module.validate_round1_learner_seed(
                    forged,
                    collection_manifest_sha256=manifest_hash,
                )

    def test_quarantine_summary_requires_zero_and_consistent_counts(self) -> None:
        summary = summarize_dagger1_offline_teacher_target_quarantine(
            [_d1_training_row()]
        )
        self.assertEqual(
            aggregate_module.validate_offline_teacher_target_quarantine_summary(
                summary,
                expected_total_rows=1,
                expected_candidate_rows=1,
            ),
            summary,
        )
        mutations = {
            "wrong_contract": {"contract": "unreviewed"},
            "inconsistent_total": {"total_rows": 2},
            "quarantine_not_zero": {
                "passed": False,
                "zero_truth_audit_quarantine": False,
                "passed_rows": 0,
                "quarantined_rows": 1,
                "quarantined_by_action_class": {"rollback": 1},
                "quarantined_by_reason_code": {
                    "candidate_source_correction_missing": 1
                },
                "quarantined_example_ids": ["d1-example"],
            },
        }
        for name, updates in mutations.items():
            with self.subTest(name=name):
                forged = copy.deepcopy(summary)
                forged.update(updates)
                with self.assertRaises(ValueError):
                    aggregate_module.validate_offline_teacher_target_quarantine_summary(
                        forged,
                        expected_total_rows=1,
                        expected_candidate_rows=1,
                    )

    def test_aggregate_recomputes_bounded_selection_from_complete_ledger(self):
        selected = _d1_training_row(
            example_id="selected",
            physical_root_fingerprint="r1",
        )
        unselected_labels = copy.deepcopy(selected["labels"])
        unselected_labels.update(
            {
                "collection_training_eligible": False,
                "collection_disposition": "safe_candidate_not_selected",
            }
        )
        unselected = _d1_training_row(
            example_id="unselected",
            physical_root_fingerprint="r2",
            collection_training_eligible=False,
            collection_disposition="safe_candidate_not_selected",
            labels=unselected_labels,
        )
        ineligible_labels = copy.deepcopy(selected["labels"])
        ineligible_labels.update(
            {
                "collection_training_eligible": False,
                "collection_disposition": "not_safe_candidate",
            }
        )
        ineligible = _d1_training_row(
            example_id="ineligible",
            physical_root_fingerprint="r3",
            production_label_eligible=False,
            collection_training_eligible=False,
            collection_disposition="not_safe_candidate",
            labels=ineligible_labels,
        )
        report = _mock_production_selection_report([selected, unselected])
        report["selected_rows"] = 1
        report["selected_example_id_sequence_sha256"] = stable_json_sha256(
            ["selected"]
        )
        manifest = {
            "production_row_target_contract": (
                dagger1_production_row_target_contract(
                    target_min_rows=300,
                    target_max_rows=600,
                )
            ),
            "deterministic_collection_selection": report,
            "candidate_recovery_rows": 2,
            "candidate_recovery_row_count": 2,
            "production_eligible_recovery_rows": 2,
            "output_rows": 1,
            "selected_recovery_row_count": 1,
        }
        ledger = [selected, unselected, ineligible]
        with patch.object(
            aggregate_module.collect_dagger1_module,
            "select_dagger1_collection_rows",
            return_value=([copy.deepcopy(selected)], copy.deepcopy(report)),
        ) as selector:
            binding = (
                aggregate_module.validate_dagger1_collection_selection_binding(
                    [selected], ledger, manifest
                )
            )
            self.assertTrue(binding["passed"])
            self.assertEqual(binding["candidate_rows"], 2)
            self.assertEqual(binding["selected_rows"], 1)
            self.assertEqual(binding["unselected_safe_candidate_rows"], 1)
            self.assertTrue(unselected["production_label_eligible"])
            self.assertEqual(
                [
                    row["example_id"]
                    for row in selector.call_args.args[0]
                ],
                ["selected", "unselected"],
            )

            forged_selected = copy.deepcopy(selected)
            forged_selected["preferred_action"] = {
                "tool": "forged",
                "arguments": {},
            }
            with self.assertRaisesRegex(ValueError, "exact deterministic"):
                aggregate_module.validate_dagger1_collection_selection_binding(
                    [forged_selected], ledger, manifest
                )

            forged_counts = {**manifest, "candidate_recovery_row_count": 1}
            with self.assertRaisesRegex(ValueError, "safe-candidate row counts"):
                aggregate_module.validate_dagger1_collection_selection_binding(
                    [selected], ledger, forged_counts
                )

    def test_aggregate_rejects_exploratory_collection_bounds(self):
        row = _d1_training_row()
        manifest = {
            "production_row_target_contract": (
                dagger1_production_row_target_contract(
                    target_min_rows=1,
                    target_max_rows=10,
                )
            ),
            "deterministic_collection_selection": {
                **_mock_production_selection_report([row]),
                "target_min_rows": 1,
                "target_max_rows": 10,
            },
        }
        with self.assertRaisesRegex(ValueError, "reviewed production row bounds"):
            aggregate_module.validate_dagger1_collection_selection_binding(
                [row], [row], manifest
            )

    def test_aggregate_requires_development_generator_report_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            d1_path, manifest_path = self._write_d1_inputs(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report_path = Path(
                manifest["development_holdout_generator_report"]
            )
            report_path.unlink()

            with (
                patch.object(
                    aggregate_module,
                    "git_source_state",
                    return_value=SOURCE_STATE,
                ),
                self.assertRaises(FileNotFoundError) as raised,
            ):
                aggregate_module.build_round1_aggregate(
                    d0_aggregate_dir=d0_dir,
                    d1_path=d1_path,
                    d1_manifest_path=manifest_path,
                    output_dir=root / "missing-development-report",
                    seed=20260719,
                    size=None,
                    d1_share=0.25,
                    minimum_d1_share=0.20,
                    maximum_d1_share=0.30,
                    max_duplicate_count=2,
                    max_rows_per_root=8,
                )
            self.assertIn(str(report_path), str(raised.exception))

    def test_aggregate_binds_development_generator_report_to_both_manifests(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            d1_path, manifest_path = self._write_d1_inputs(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report_path = Path(
                manifest["development_holdout_generator_report"]
            )
            with report_path.open("ab") as handle:
                handle.write(b"\n")

            def build(output_name: str) -> None:
                aggregate_module.build_round1_aggregate(
                    d0_aggregate_dir=d0_dir,
                    d1_path=d1_path,
                    d1_manifest_path=manifest_path,
                    output_dir=root / output_name,
                    seed=20260719,
                    size=None,
                    d1_share=0.25,
                    minimum_d1_share=0.20,
                    maximum_d1_share=0.30,
                    max_duplicate_count=2,
                    max_rows_per_root=8,
                )

            with patch.object(
                aggregate_module,
                "git_source_state",
                return_value=SOURCE_STATE,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "generator-report bytes do not match the collection manifest",
                ):
                    build("tampered-development-report")

                manifest["development_holdout_generator_report_sha256"] = (
                    file_sha256(report_path)
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "does not match the development manifest",
                ):
                    build("forged-development-report-binding")

    def test_public_builder_binds_inputs_rejects_tamper_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir = _write_d0_inputs(root)
            d1_path, d1_manifest_path = self._write_d1_inputs(root)
            d1_manifest = json.loads(
                d1_manifest_path.read_text(encoding="utf-8")
            )
            development_payload = json.loads(
                Path(d1_manifest["development_holdout"]).read_text(
                    encoding="utf-8"
                )
            )
            development_roots = frozenset(
                row["grouping"]["physical_root_fingerprint"]
                for row in development_payload[
                    DAGGER1_DEVELOPMENT_SUITE_NAME
                ]
            )

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
                    "validate_development_holdout_binding",
                    return_value=development_roots,
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
                patch.object(
                    aggregate_module,
                    "audit_dagger1_recovery_labels",
                    return_value={"passed": True},
                ),
                patch.object(
                    aggregate_module,
                    "audit_target_aware_state_classes",
                    return_value={"passed": True},
                ),
                patch.object(
                    aggregate_module,
                    "audit_dagger1_independent_root_support",
                    return_value={"passed": True},
                ),
                patch.object(
                    aggregate_module,
                    "audit_dagger1_union_realizability",
                    return_value={"passed": True, "failures": []},
                ),
                patch.object(
                    aggregate_module.collect_dagger1_module,
                    "select_dagger1_collection_rows",
                    side_effect=lambda rows, **kwargs: (
                        copy.deepcopy(list(rows)),
                        _mock_production_selection_report(list(rows)),
                    ),
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
                provenance = json.loads(
                    (output_dir / "aggregate.generation_provenance.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    provenance["generation_descriptor"]["input_artifacts"][
                        "d0_manifest_sha256"
                    ],
                    file_sha256(d0_dir / "aggregate.manifest.json"),
                )
                self.assertEqual(
                    provenance["generation_descriptor"]["learner_seed"],
                    {
                        "role": "learner_seed_only",
                        "collection_model_id": LEARNER_MODEL_ID,
                        "adapter_tree_sha256": LEARNER_REVISION,
                        "collection_model_revision": LEARNER_REVISION,
                        "collection_manifest_sha256": file_sha256(
                            d1_manifest_path
                        ),
                    },
                )
                holdout_binding = provenance["generation_descriptor"][
                    "input_artifacts"
                ]["d1_development_holdout"]
                collection_manifest = json.loads(
                    d1_manifest_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    holdout_binding,
                    {
                        "holdout_sha256": collection_manifest[
                            "development_holdout_sha256"
                        ],
                        "manifest_sha256": collection_manifest[
                            "development_holdout_manifest_sha256"
                        ],
                        "generator_report_sha256": collection_manifest[
                            "development_holdout_generator_report_sha256"
                        ],
                        "physical_root_count": 30,
                        "root_set_sha256": collection_manifest[
                            "development_holdout_root_set_sha256"
                        ],
                    },
                )
                self.assertTrue((output_dir / "SHA256SUMS").is_file())
                for immutable_name in (
                    "aggregate.raw.jsonl",
                    "aggregate.d0.raw.jsonl",
                    "aggregate.d1.raw.jsonl",
                ):
                    self.assertTrue((output_dir / immutable_name).is_file())

                empty_output_dir = root / "round1-existing-empty"
                empty_output_dir.mkdir()
                empty_report = aggregate_module.build_round1_aggregate(
                    d0_aggregate_dir=d0_dir,
                    d1_path=d1_path,
                    d1_manifest_path=d1_manifest_path,
                    output_dir=empty_output_dir,
                    seed=20260719,
                    size=None,
                    d1_share=0.25,
                    minimum_d1_share=0.20,
                    maximum_d1_share=0.30,
                    max_duplicate_count=2,
                    max_rows_per_root=8,
                )
                self.assertTrue(empty_report["release_eligible"])
                self.assertEqual(
                    {path.name for path in empty_output_dir.iterdir()},
                    set(aggregate_module._ROUND1_OUTPUT_FILENAMES),
                )

                d0_provenance_path = (
                    d0_dir / "aggregate.generation_provenance.json"
                )
                original_d0_provenance = d0_provenance_path.read_bytes()
                d0_not_clean = json.loads(original_d0_provenance)
                d0_not_clean["generation_descriptor"]["source_state"][
                    "release_eligible_source"
                ] = False
                d0_not_clean["generation_provenance_id"] = stable_json_sha256(
                    d0_not_clean["generation_descriptor"]
                )
                d0_provenance_path.write_text(
                    json.dumps(d0_not_clean, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError, "source state is not release eligible"
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "d0-not-clean",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )

                d0_provenance_path.write_bytes(original_d0_provenance)
                d0_bad_id = json.loads(original_d0_provenance)
                d0_bad_id["generation_provenance_id"] = "f" * 64
                d0_provenance_path.write_text(
                    json.dumps(d0_bad_id, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "provenance ID"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "d0-bad-provenance-id",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                d0_provenance_path.write_bytes(original_d0_provenance)

                d0_manifest_path = d0_dir / "aggregate.manifest.json"
                original_d0_manifest = d0_manifest_path.read_bytes()
                d0_manifest_path.write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "manifest does not match its provenance"
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "d0-bad-manifest",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                d0_manifest_path.write_bytes(original_d0_manifest)

                original_d1 = d1_path.read_bytes()
                original_manifest = d1_manifest_path.read_bytes()
                d1_wrong_d0 = json.loads(original_manifest)
                d1_wrong_d0["d0_manifest_sha256"] = "f" * 64
                d1_manifest_path.write_text(
                    json.dumps(d1_wrong_d0, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError, "bound to a different D0 aggregate manifest"
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "d1-wrong-d0-manifest",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                d1_manifest_path.write_bytes(original_manifest)
                original_all_output_path = Path(
                    json.loads(original_manifest)["all_output"]
                )
                original_all_output = original_all_output_path.read_bytes()
                original_development_path = Path(
                    json.loads(original_manifest)["development_holdout"]
                )
                original_development = original_development_path.read_bytes()
                with original_development_path.open("ab") as handle:
                    handle.write(b"\n")
                with self.assertRaisesRegex(
                    ValueError,
                    "development holdout bytes",
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "tampered-development",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                original_development_path.write_bytes(original_development)

                with original_all_output_path.open("ab") as handle:
                    handle.write(b"{}\n")
                with self.assertRaisesRegex(ValueError, "all-output ledger hash"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "tampered-all-output",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                original_all_output_path.write_bytes(original_all_output)

                summary_tampered_manifest = json.loads(original_manifest)
                summary_tampered_manifest[
                    "offline_teacher_target_quarantine_summary"
                ]["candidate_definition"]["unreviewed_extra"] = True
                d1_manifest_path.write_text(
                    json.dumps(
                        summary_tampered_manifest, indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError, "summary differs from the all-output ledger"
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "tampered-summary",
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                d1_manifest_path.write_bytes(original_manifest)

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

                development_payload = json.loads(
                    original_development.decode("utf-8")
                )
                development_root = development_payload[
                    DAGGER1_DEVELOPMENT_SUITE_NAME
                ][0]["grouping"]["physical_root_fingerprint"]
                _write_jsonl(
                    d1_path,
                    [
                        _d1_training_row(
                            physical_root_fingerprint=development_root,
                            example_id="forged-development-root",
                        )
                    ],
                )
                self._rebind_d1_manifest(d1_path, d1_manifest_path)
                with self.assertRaisesRegex(ValueError, "development holdout"):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "forged-development",
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
                original_all_output_path.write_bytes(original_all_output)

                frozen_root = sorted(
                    frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE)
                )[0]
                _write_jsonl(
                    d1_path,
                    [
                        _d1_training_row(
                            physical_root_fingerprint=frozen_root,
                            example_id="forged-frozen-root",
                        )
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
                        _d1_training_row(
                            example_id="forged-private-truth",
                            labels={
                                "nested": {
                                    "true_parameter_errors": [
                                        {"line_index1": 1}
                                    ]
                                }
                            },
                        )
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

                smuggled_audit = _passed_offline_teacher_target_audit()
                smuggled_audit["unexpected_extra_field"] = "covert-metadata"
                _write_jsonl(
                    d1_path,
                    [
                        _d1_training_row(
                            example_id="forged-audit-metadata",
                            offline_teacher_target_audit=smuggled_audit,
                        )
                    ],
                )
                self._rebind_d1_manifest(d1_path, d1_manifest_path)
                with self.assertRaisesRegex(
                    ValueError, "invalid offline teacher-target audit"
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=root / "forged-audit-metadata",
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
                original_all_output_path.write_bytes(original_all_output)

                for occupied_name in (
                    "aggregate.train_view.jsonl",
                    "aggregate.generation_provenance.json",
                    "aggregate.preflight.json",
                    "SHA256SUMS",
                ):
                    with self.subTest(occupied_name=occupied_name):
                        occupied_dir = root / (
                            "occupied-" + occupied_name.replace(".", "-")
                        )
                        occupied_dir.mkdir()
                        occupied_path = occupied_dir / occupied_name
                        occupied_path.write_text(
                            "do-not-overwrite\n", encoding="utf-8"
                        )
                        with self.assertRaisesRegex(
                            FileExistsError, "already exist"
                        ):
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
                            occupied_path.read_text(encoding="utf-8"),
                            "do-not-overwrite\n",
                        )

                late_failure_dir = root / "late-failure"
                late_failure_dir.mkdir()
                real_text_writer = aggregate_module._write_text_artifact

                def fail_on_checksum(path, content):
                    if path.name == "SHA256SUMS":
                        raise OSError("simulated late write failure")
                    return real_text_writer(path, content)

                with (
                    patch.object(
                        aggregate_module,
                        "_write_text_artifact",
                        side_effect=fail_on_checksum,
                    ),
                    self.assertRaisesRegex(OSError, "simulated late write failure"),
                ):
                    aggregate_module.build_round1_aggregate(
                        d0_aggregate_dir=d0_dir,
                        d1_path=d1_path,
                        d1_manifest_path=d1_manifest_path,
                        output_dir=late_failure_dir,
                        seed=20260719,
                        size=None,
                        d1_share=0.25,
                        minimum_d1_share=0.20,
                        maximum_d1_share=0.30,
                        max_duplicate_count=2,
                        max_rows_per_root=8,
                    )
                self.assertEqual(list(late_failure_dir.iterdir()), [])
                self.assertEqual(
                    list(root.glob(f".{late_failure_dir.name}.staging-*")), []
                )


if __name__ == "__main__":
    unittest.main()
