from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import psse_env.dagger.build_dagger1_development_holdout as holdout_module
from psse_env.sft.provenance import file_sha256, stable_json_sha256


SOURCE_STATE = {
    "source_commit": "a" * 40,
    "source_worktree_dirty": False,
    "tracked_diff_hash": "b" * 64,
    "untracked_source_files": [],
    "release_eligible_source": True,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _envelope(
    root: str,
    family: str,
    *,
    split: str = "dagger_train",
    scans: bool = False,
) -> dict:
    return {
        "scenario_schema_version": 1,
        "execution": {
            "scenario_id": f"{family}-{root}",
            "case": "case14",
            "measurements": [],
            "metadata": {
                "parameter_scans": {"z_scans": [[1.0]] if scans else []}
            },
        },
        "audit": {
            "truth": {"truth_complete": True, "clean_measurements": []},
            "evaluation_intervention": {
                "intervention_schema_version": 1,
                "kind": "none",
            },
        },
        "grouping": {
            "root_scenario_id": f"root-{root}",
            "physical_root_fingerprint": root,
            "scenario_family": family,
            "error_cardinality": 2 if family == "measurement+parameter" else 1,
            "case_id": "case14",
            "split": split,
            "source_tier": "test",
        },
    }


def _write_boundaries(root: Path, *, training_root: str = "training-root") -> tuple[Path, Path, Path]:
    d0_dir = root / "d0"
    d0_raw = d0_dir / "aggregate.raw.jsonl"
    _write_jsonl(
        d0_raw,
        [
            {
                "physical_root_fingerprint": "d0-root",
                "dataset_split": "train",
                "production_label_eligible": True,
            }
        ],
    )
    d0_provenance = d0_dir / "aggregate.generation_provenance.json"
    d0_descriptor = {"source_state": SOURCE_STATE}
    d0_provenance.write_text(
        json.dumps(
            {
                "release_eligible": True,
                "generation_descriptor": d0_descriptor,
                "generation_provenance_id": stable_json_sha256(d0_descriptor),
                "dataset_hashes": {d0_raw.name: file_sha256(d0_raw)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    training_path = root / "d1-training.json"
    training_rows = [_envelope(training_root, "measurement+parameter")]
    training_path.write_text(
        json.dumps(training_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_manifest = root / "d1-training.json.manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_evidence_eligible": False,
                "source_partition": "train",
                "parameter_ranking_dominance_threshold": 1.0,
                "source_state": SOURCE_STATE,
                "output_sha256": file_sha256(training_path),
                "scenario_count": 1,
                "physical_root_count": 1,
                "protected_root_overlap": [],
                "d0_raw_sha256": file_sha256(d0_raw),
                "d0_generation_provenance_sha256": file_sha256(d0_provenance),
                "frozen_suite_sha256": file_sha256(
                    holdout_module.DEFAULT_FORBIDDEN_SUITE
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return d0_dir, training_path, training_manifest


class _FakeGenerator:
    def __init__(
        self,
        *,
        seed: int,
        source_partition: str,
        parameter_ranking_dominance_threshold: float,
    ) -> None:
        if source_partition != "train":
            raise AssertionError(source_partition)
        if parameter_ranking_dominance_threshold != 1.0:
            raise AssertionError(parameter_ranking_dominance_threshold)
        self.seed = seed

    def build(self, plan: dict[str, int]) -> list[dict]:
        candidates = {
            "measurement+parameter": [
                ("frozen-root", False),
                ("mixed-fresh", False),
                ("mixed-extra", False),
            ],
            "multi_measurement": [
                ("training-root", False),
                ("multi-with-scans", True),
                ("multi-fresh", False),
            ],
            "parameter": [
                ("d0-root", False),
                ("parameter-fresh", False),
                ("parameter-extra", False),
            ],
        }
        rows: list[dict] = []
        for family, count in sorted(plan.items()):
            if count != 3:
                raise AssertionError((family, count))
            for index, (root, scans) in enumerate(candidates[family]):
                rows.append(
                    {
                        "scenario_id": f"candidate-{family}-{index}",
                        "scenario_family": family,
                        "error_cardinality": (
                            2 if family == "measurement+parameter" else 1
                        ),
                        "root": root,
                        "scans": scans,
                    }
                )
        return rows

    def report(self) -> dict:
        return {
            "seed": self.seed,
            "source_partition": {"enabled": True, "selected": "train"},
            "parameter_ranking_admission": {
                "enforced": True,
                "threshold": 1.0,
            },
        }


def _fake_partition(row: dict, *, split: str) -> dict:
    return _envelope(
        row["root"],
        row["scenario_family"],
        split=split,
        scans=row["scans"],
    )


class Dagger1DevelopmentHoldoutTests(unittest.TestCase):
    def _patch_builder(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch.object(
                holdout_module,
                "git_source_state",
                return_value=SOURCE_STATE,
            )
        )
        stack.enter_context(
            patch.object(holdout_module, "Round0ScenarioGenerator", _FakeGenerator)
        )
        stack.enter_context(
            patch.object(
                holdout_module,
                "partition_release_scenario_v1",
                side_effect=_fake_partition,
            )
        )
        stack.enter_context(
            patch.object(
                holdout_module,
                "_frozen_physical_roots",
                return_value=frozenset({"frozen-root"}),
            )
        )
        stack.enter_context(
            patch.object(
                holdout_module,
                "_source_bindings",
                return_value={
                    "psse_env/dagger/build_dagger1_development_holdout.py": "c" * 64
                },
            )
        )
        return stack

    def test_default_plan_reserves_exactly_thirty_roots(self) -> None:
        plan = holdout_module._load_plan(None)
        self.assertEqual(
            plan,
            {
                "measurement+parameter": 12,
                "multi_measurement": 12,
                "parameter": 6,
            },
        )
        self.assertEqual(sum(plan.values()), 30)

    def test_build_is_deterministic_disjoint_and_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir, training_path, training_manifest = _write_boundaries(root)
            runs: list[tuple[Path, Path, dict]] = []
            with self._patch_builder():
                for run in ("one", "two"):
                    output = root / run / "development.json"
                    report = root / run / "generator.json"
                    manifest = holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=output,
                        generator_report_path=report,
                        seed=20260721,
                        plan={
                            "measurement+parameter": 1,
                            "multi_measurement": 1,
                            "parameter": 1,
                        },
                        candidate_multiplier=3,
                    )
                    runs.append((output, report, manifest))
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=runs[0][0],
                        generator_report_path=runs[0][1],
                        seed=20260721,
                        plan={
                            "measurement+parameter": 1,
                            "multi_measurement": 1,
                            "parameter": 1,
                        },
                        candidate_multiplier=3,
                    )

            first_output, first_report, first_manifest = runs[0]
            second_output, second_report, second_manifest = runs[1]
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertFalse(first_manifest["training_eligible"])
            self.assertFalse(first_manifest["training_collection_eligible"])
            self.assertFalse(first_manifest["release_evidence_eligible"])
            self.assertFalse(first_manifest["promotion_evidence_eligible"])
            self.assertFalse(
                first_manifest[
                    "diagnostic_closed_loop_model_selection_eligible"
                ]
            )
            self.assertFalse(
                first_manifest[
                    "recovery_stratum_qualified_model_selection_eligible"
                ]
            )
            self.assertEqual(
                first_manifest["recovery_strata_qualification_status"],
                "pending_teacher_opportunity_trace_instrumentation",
            )
            self.assertEqual(first_manifest["physical_root_count"], 3)
            self.assertEqual(
                first_manifest["development_protected_overlap"],
                {"d0": [], "d1_training": [], "frozen": []},
            )
            payload = json.loads(first_output.read_text(encoding="utf-8"))
            rows = payload[holdout_module.DAGGER1_DEVELOPMENT_SUITE_NAME]
            self.assertEqual(
                {row["grouping"]["physical_root_fingerprint"] for row in rows},
                {"mixed-extra", "multi-fresh", "parameter-extra"},
            )
            self.assertTrue(
                all(
                    row["scenario_schema_version"] == 1
                    and row["grouping"]["split"] == "dagger_development"
                    for row in rows
                )
            )
            self.assertEqual(first_manifest["output_sha256"], file_sha256(first_output))
            self.assertEqual(
                first_manifest["d1_training_manifest_sha256"],
                file_sha256(training_manifest),
            )

    def test_only_approved_plan_is_model_selection_eligible(self) -> None:
        normalized_default = holdout_module._load_plan(None)
        self.assertEqual(
            sum(normalized_default.values()),
            holdout_module.APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT,
        )

        self.assertTrue(
            holdout_module._diagnostic_model_selection_eligible(
                normalized_default,
                physical_root_count=30,
            )
        )
        self.assertFalse(
            holdout_module._diagnostic_model_selection_eligible(
                normalized_default,
                physical_root_count=29,
            )
        )
        self.assertFalse(
            holdout_module._diagnostic_model_selection_eligible(
                {
                    "measurement+parameter": 11,
                    "multi_measurement": 13,
                    "parameter": 6,
                },
                physical_root_count=30,
            )
        )

    def test_dirty_source_and_tampered_training_input_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir, training_path, training_manifest = _write_boundaries(root)
            with patch.object(
                holdout_module,
                "git_source_state",
                return_value={**SOURCE_STATE, "release_eligible_source": False},
            ):
                with self.assertRaisesRegex(RuntimeError, "clean source tree"):
                    holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=root / "dirty.json",
                        generator_report_path=root / "dirty-report.json",
                        seed=1,
                        plan={"parameter": 1},
                    )

            rows = json.loads(training_path.read_text(encoding="utf-8"))
            training_path.write_text(
                json.dumps([*rows, copy.deepcopy(rows[0])], sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self._patch_builder():
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError),
                    "scenario boundary|globally unique",
                ):
                    holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=root / "tampered.json",
                        generator_report_path=root / "tampered-report.json",
                        seed=1,
                        plan={"parameter": 1},
                    )

    def test_existing_training_boundary_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir, training_path, training_manifest = _write_boundaries(
                root,
                training_root="d0-root",
            )
            with self._patch_builder():
                with self.assertRaisesRegex(RuntimeError, "boundaries overlap"):
                    holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=root / "overlap.json",
                        generator_report_path=root / "overlap-report.json",
                        seed=1,
                        plan={"parameter": 1},
                    )

    def test_d0_provenance_id_and_clean_source_are_required(self) -> None:
        for mutation, expected in (
            (
                lambda payload: payload.__setitem__(
                    "generation_provenance_id", "f" * 64
                ),
                "byte-bound",
            ),
            (
                lambda payload: payload["generation_descriptor"][
                    "source_state"
                ].__setitem__("release_eligible_source", False),
                "byte-bound",
            ),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                d0_dir, training_path, training_manifest = _write_boundaries(root)
                provenance_path = (
                    d0_dir / "aggregate.generation_provenance.json"
                )
                payload = json.loads(provenance_path.read_text(encoding="utf-8"))
                mutation(payload)
                provenance_path.write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self._patch_builder(), self.assertRaisesRegex(
                    RuntimeError,
                    expected,
                ):
                    holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=root / "development.json",
                        generator_report_path=root / "generator.json",
                        seed=1,
                        plan={"parameter": 1},
                    )

    def test_substituted_frozen_suite_fails_policy_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0_dir, training_path, training_manifest = _write_boundaries(root)
            substituted_suite = root / "substituted-frozen-suite.json"
            substituted_suite.write_text(
                holdout_module.DEFAULT_FORBIDDEN_SUITE.read_text(encoding="utf-8")
                + "\n",
                encoding="utf-8",
            )
            with self._patch_builder():
                with self.assertRaisesRegex(
                    RuntimeError,
                    "does not match the pinned evaluation policy",
                ):
                    holdout_module.build_dagger1_development_holdout(
                        d0_aggregate_dir=d0_dir,
                        d1_training_scenarios=training_path,
                        d1_training_manifest=training_manifest,
                        output=root / "substituted.json",
                        generator_report_path=root / "substituted-report.json",
                        seed=1,
                        plan={"parameter": 1},
                        forbidden_suite_path=substituted_suite,
                    )


if __name__ == "__main__":
    unittest.main()
