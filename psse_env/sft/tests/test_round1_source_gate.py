"""Executable tests for the Round-1 aggregate source-mix gate."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from psse_env.dagger.collect_dagger1 import DAGGER1_SCENARIO_BUILDER_CONTRACT
from psse_env.dagger.rollout_collector import (
    OFFLINE_TEACHER_TARGET_QUARANTINE_SUMMARY_CONTRACT,
)
from psse_env.sft.gates import GateError
from psse_env.sft.provenance import file_sha256, stable_json_sha256
from psse_env.sft.round1_source_gate import (
    ROUND1_AGGREGATE_BUILDER_CONTRACT,
    ROUND1_IMMUTABLE_VIEW_NAMES,
    main,
    validate_round1_source_mix_gate,
)


SOURCE_COMMIT = "a" * 40
ADAPTER_REVISION = "b" * 64
COLLECTION_MANIFEST_SHA256 = "c" * 64
DEVELOPMENT_HOLDOUT_SHA256 = "1" * 64
DEVELOPMENT_HOLDOUT_MANIFEST_SHA256 = "2" * 64
DEVELOPMENT_HOLDOUT_GENERATOR_REPORT_SHA256 = "4" * 64
DEVELOPMENT_HOLDOUT_ROOT_SET_SHA256 = "3" * 64
DEVELOPMENT_HOLDOUT_ROOT_COUNT = 30
D0_MANIFEST_SHA256 = "0" * 64


class Round1SourceMixGateTests(unittest.TestCase):
    def _write_valid_artifacts(self, root: Path) -> tuple[Path, Path]:
        for index, name in enumerate(ROUND1_IMMUTABLE_VIEW_NAMES):
            (root / name).write_text(
                json.dumps({"view": name, "index": index}) + "\n",
                encoding="utf-8",
            )

        learner_seed = {
            "role": "learner_seed_only",
            "collection_model_id": str((root / "seed_adapter").resolve()),
            "adapter_tree_sha256": ADAPTER_REVISION,
            "collection_model_revision": ADAPTER_REVISION,
            "collection_manifest_sha256": COLLECTION_MANIFEST_SHA256,
        }
        d1_manifest = {
            "training_eligible": True,
            "release_evidence_eligible": False,
            "source_state": {
                "source_commit": SOURCE_COMMIT,
                "release_eligible_source": True,
            },
            "scenario_builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
            "scenario_manifest_sha256": "d" * 64,
            "d0_manifest_sha256": D0_MANIFEST_SHA256,
            "development_holdout_sha256": DEVELOPMENT_HOLDOUT_SHA256,
            "development_holdout_manifest_sha256": (
                DEVELOPMENT_HOLDOUT_MANIFEST_SHA256
            ),
            "development_holdout_generator_report_sha256": (
                DEVELOPMENT_HOLDOUT_GENERATOR_REPORT_SHA256
            ),
            "development_holdout_root_count": DEVELOPMENT_HOLDOUT_ROOT_COUNT,
            "development_physical_root_count": DEVELOPMENT_HOLDOUT_ROOT_COUNT,
            "development_holdout_root_set_sha256": (
                DEVELOPMENT_HOLDOUT_ROOT_SET_SHA256
            ),
            "model_id": learner_seed["collection_model_id"],
            "model_revision": ADAPTER_REVISION,
            "learner_seed": {
                key: value
                for key, value in learner_seed.items()
                if key != "collection_manifest_sha256"
            },
        }
        quarantine_summary = {
            "contract": OFFLINE_TEACHER_TARGET_QUARANTINE_SUMMARY_CONTRACT,
            "candidate_definition": {},
            "total_rows": 25,
            "candidate_rows": 25,
            "non_candidate_rows": 0,
            "passed_rows": 25,
            "quarantined_rows": 0,
            "invalid_or_missing_audit_rows": 0,
            "quarantined_by_action_class": {},
            "quarantined_by_reason_code": {},
            "quarantined_example_ids": [],
            "zero_truth_audit_quarantine": True,
            "passed": True,
        }
        recomputed_d1 = {
            "offline_teacher_target_quarantine_summary": quarantine_summary,
            "recovery_label_audit": {"passed": True},
            "target_aware_state_class_audit": {"passed": True},
            "independent_root_support": {"passed": True},
        }
        semantic = {
            "passed": True,
            "natural_teacher_realizability": {"passed": True},
            "training_view_teacher_realizability": {"passed": True},
            "natural_approximate_teacher_realizability": {"passed": True},
            "training_view_approximate_teacher_realizability": {
                "passed": True
            },
            "approximate_teacher_realizability_by_scenario_family": {
                "family": {"release_gate_passed": True}
            },
            "approximate_teacher_realizability_by_state_class": {
                "state": {"release_gate_passed": True}
            },
            "approximate_teacher_realizability_by_recovery_stratum": {
                "stratum": {"release_gate_passed": True}
            },
        }
        training_view = {
            "builder_contract": ROUND1_AGGREGATE_BUILDER_CONTRACT,
            "source_allocation": {
                "passed": True,
                "observed_d1_share": 0.25,
                "d1_recovery_rows": 25,
            },
        }
        audit_reports = {
            "d1_offline_teacher_target_quarantine_summary": quarantine_summary,
            "d1_recovery_label_audit": recomputed_d1["recovery_label_audit"],
            "d1_target_aware_state_class_audit": recomputed_d1[
                "target_aware_state_class_audit"
            ],
            "d1_independent_root_support": recomputed_d1[
                "independent_root_support"
            ],
            "union_realizability": semantic,
        }
        for key in (
            "natural_teacher_realizability",
            "training_view_teacher_realizability",
            "natural_approximate_teacher_realizability",
            "training_view_approximate_teacher_realizability",
            "approximate_teacher_realizability_by_scenario_family",
            "approximate_teacher_realizability_by_state_class",
            "approximate_teacher_realizability_by_recovery_stratum",
        ):
            audit_reports[f"union_{key}"] = semantic[key]
        audit_hashes = {
            name: stable_json_sha256(report)
            for name, report in sorted(audit_reports.items())
        }
        descriptor = {
            "builder_contract": ROUND1_AGGREGATE_BUILDER_CONTRACT,
            "source_state": {
                "source_commit": SOURCE_COMMIT,
                "release_eligible_source": True,
            },
            "training_view_report_sha256": stable_json_sha256(training_view),
            "audit_report_sha256": audit_hashes,
            "input_artifacts": {
                "d0_manifest_sha256": D0_MANIFEST_SHA256,
                "d1_rows_sha256": "e" * 64,
                "d1_manifest_sha256": COLLECTION_MANIFEST_SHA256,
                "d1_development_holdout": {
                    "holdout_sha256": DEVELOPMENT_HOLDOUT_SHA256,
                    "manifest_sha256": DEVELOPMENT_HOLDOUT_MANIFEST_SHA256,
                    "generator_report_sha256": (
                        DEVELOPMENT_HOLDOUT_GENERATOR_REPORT_SHA256
                    ),
                    "physical_root_count": DEVELOPMENT_HOLDOUT_ROOT_COUNT,
                    "root_set_sha256": DEVELOPMENT_HOLDOUT_ROOT_SET_SHA256,
                },
            },
            "learner_seed": learner_seed,
        }
        provenance_id = stable_json_sha256(descriptor)
        provenance = {
            "release_eligible": True,
            "generation_descriptor": descriptor,
            "generation_provenance_id": provenance_id,
            "dataset_hashes": {
                name: file_sha256(root / name)
                for name in ROUND1_IMMUTABLE_VIEW_NAMES
            },
        }
        preflight = {
            "generation_provenance_id": provenance_id,
            "training_view": training_view,
            "recomputed_d1_audits": recomputed_d1,
            "semantic_realizability": semantic,
            "audit_report_sha256": audit_hashes,
            "d1_collection_manifest": d1_manifest,
            "d1_development_holdout": descriptor["input_artifacts"][
                "d1_development_holdout"
            ],
        }
        provenance_path = root / "aggregate.generation_provenance.json"
        preflight_path = root / "aggregate.preflight.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preflight_path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return provenance_path, preflight_path

    def _rewrite_descriptor(
        self,
        provenance_path: Path,
        preflight_path: Path,
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        mutate(provenance["generation_descriptor"])
        provenance_id = stable_json_sha256(provenance["generation_descriptor"])
        provenance["generation_provenance_id"] = provenance_id
        preflight["generation_provenance_id"] = provenance_id
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preflight_path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _validate(self, provenance: Path, preflight: Path) -> dict[str, object]:
        return validate_round1_source_mix_gate(
            provenance,
            preflight,
            reviewed_source_commit=SOURCE_COMMIT,
            initial_adapter_revision=ADAPTER_REVISION,
        )

    def test_valid_artifacts_pass_function_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            report = self._validate(provenance, preflight)
            self.assertTrue(report["passed"])
            self.assertEqual(report["d1_recovery_rows"], 25)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        "--provenance",
                        str(provenance),
                        "--preflight",
                        str(preflight),
                        "--reviewed-source-commit",
                        SOURCE_COMMIT,
                        "--initial-adapter-revision",
                        ADAPTER_REVISION,
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn("source-mix gate passed", output.getvalue())

    def test_missing_immutable_view_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight = self._write_valid_artifacts(root)
            (root / ROUND1_IMMUTABLE_VIEW_NAMES[0]).unlink()
            with self.assertRaisesRegex(GateError, "immutable source view is missing"):
                self._validate(provenance, preflight)

    def test_tampered_immutable_view_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight = self._write_valid_artifacts(root)
            (root / ROUND1_IMMUTABLE_VIEW_NAMES[1]).write_text(
                "tampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateError, "source view hash mismatch"):
                self._validate(provenance, preflight)

    def test_mismatched_provenance_and_preflight_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight = self._write_valid_artifacts(root)
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["generation_provenance_id"] = "f" * 64
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "preflight and provenance IDs differ",
            ):
                self._validate(provenance, preflight)

    def test_descriptor_id_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight = self._write_valid_artifacts(root)
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            payload["generation_provenance_id"] = "f" * 64
            provenance.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "generation_provenance_id does not hash",
            ):
                self._validate(provenance, preflight)

    def test_d0_manifest_binding_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            self._rewrite_descriptor(
                provenance,
                preflight,
                lambda descriptor: descriptor["input_artifacts"].pop(
                    "d0_manifest_sha256"
                ),
            )
            with self.assertRaisesRegex(
                GateError,
                "bound D0 aggregate manifest hash",
            ):
                self._validate(provenance, preflight)

    def test_d0_manifest_binding_must_match_collection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_collection_manifest"]["d0_manifest_sha256"] = "f" * 64
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "differs from the D1 collection manifest",
            ):
                self._validate(provenance, preflight)

    def test_initial_adapter_seed_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            with self.assertRaisesRegex(
                GateError,
                "INITIAL_ADAPTER_REVISION differs",
            ):
                validate_round1_source_mix_gate(
                    provenance,
                    preflight,
                    reviewed_source_commit=SOURCE_COMMIT,
                    initial_adapter_revision="f" * 64,
                )

    def test_aggregate_source_must_be_release_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            self._rewrite_descriptor(
                provenance,
                preflight,
                lambda descriptor: descriptor["source_state"].update(
                    {"release_eligible_source": False}
                ),
            )
            with self.assertRaisesRegex(
                GateError,
                "aggregate source is not release eligible",
            ):
                self._validate(provenance, preflight)

    def test_d1_source_must_be_release_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_collection_manifest"]["source_state"][
                "release_eligible_source"
            ] = False
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "D1 collection manifest is not approved",
            ):
                self._validate(provenance, preflight)

    def test_development_holdout_binding_shape_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            self._rewrite_descriptor(
                provenance,
                preflight,
                lambda descriptor: descriptor["input_artifacts"][
                    "d1_development_holdout"
                ].update({"unexpected": "field"}),
            )
            with self.assertRaisesRegex(
                GateError,
                "development-holdout binding has an invalid shape",
            ):
                self._validate(provenance, preflight)

    def test_preflight_development_holdout_binding_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_development_holdout"]["root_set_sha256"] = "4" * 64
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "provenance and preflight development-holdout bindings differ",
            ):
                self._validate(provenance, preflight)

    def test_d1_manifest_development_holdout_binding_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_collection_manifest"][
                "development_physical_root_count"
            ] = DEVELOPMENT_HOLDOUT_ROOT_COUNT - 1
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "D1 collection manifest development-holdout binding is invalid",
            ):
                self._validate(provenance, preflight)

    def test_d1_manifest_development_root_set_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_collection_manifest"][
                "development_holdout_root_set_sha256"
            ] = "4" * 64
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "development-holdout binding differs from the D1 collection manifest",
            ):
                self._validate(provenance, preflight)

    def test_d1_manifest_development_generator_report_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_collection_manifest"][
                "development_holdout_generator_report_sha256"
            ] = "5" * 64
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "development-holdout binding differs from the D1 collection manifest",
            ):
                self._validate(provenance, preflight)

    def test_development_holdout_hashes_must_be_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            self._rewrite_descriptor(
                provenance,
                preflight,
                lambda descriptor: descriptor["input_artifacts"][
                    "d1_development_holdout"
                ].update({"holdout_sha256": "not-a-digest"}),
            )
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["d1_development_holdout"][
                "holdout_sha256"
            ] = "not-a-digest"
            preflight.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError,
                "development-holdout hashes or root count are invalid",
            ):
                self._validate(provenance, preflight)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
