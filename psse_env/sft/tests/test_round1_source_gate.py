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
from unittest import mock

from psse_env.dagger.collect_dagger1 import DAGGER1_SCENARIO_BUILDER_CONTRACT
from psse_env.dagger.dataset_builder import examples_to_chat_sft
from psse_env.dagger.round1_view_policy import round1_view_policy_digest
from psse_env.dagger.three_source_view import (
    FINAL_VIEW_SUPPORT_CONTRACT,
    build_dagger1_three_source_view,
)
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
PROBE_PROVENANCE_ID = "f" * 64
TEST_VIEW_POLICY = {
    "contract": "dagger1_round1_three_source_view_v1",
    "schema_version": 1,
    "total_rows": 4,
    "allocation": {
        "d0_bc0_rows": 1,
        "natural_d1_rows": 1,
        "observable_recovery_probe_rows": 2,
    },
    "shares_for_provenance_only": {
        "probe_share": 0.5,
        "natural_share": 0.5,
        "natural_d1_share_of_natural_rows": 0.5,
    },
    "probe_bucket": {
        "post_failure_no_candidate": 1,
        "unsupported_correction_recovery": 1,
        "distinct_roots_retained_per_stratum": 1,
        "duplicate_placements_per_stratum": 0,
    },
    "global_caps": {
        "max_duplicate_count": 2,
        "max_rows_per_root": 8,
        "applies_across_sources": True,
    },
    "incidence_dependent_recovery_strata": [
        "post_failure_no_candidate",
        "unsupported_correction_recovery",
    ],
    "probe_floor_distinct_roots": 1,
    "combined_floor_distinct_roots": 1,
}


class Round1SourceMixGateTests(unittest.TestCase):
    def setUp(self) -> None:
        support = {"contract": "test_support", "passed": True}
        self.support = support
        self.final_support = {
            "contract": FINAL_VIEW_SUPPORT_CONTRACT,
            "natural_rows": 1,
            "probe_rows": 2,
            "training_support": support,
            "probe_source_coverage": {"passed": True},
            "passed": True,
        }
        patchers = (
            mock.patch(
                "psse_env.sft.round1_source_gate.ROUND1_THREE_SOURCE_VIEW_POLICY",
                TEST_VIEW_POLICY,
            ),
            mock.patch(
                "psse_env.sft.round1_source_gate.round1_view_policy_digest",
                side_effect=lambda: round1_view_policy_digest(TEST_VIEW_POLICY),
            ),
            mock.patch(
                "psse_env.sft.round1_source_gate.audit_dagger1_training_support",
                return_value=support,
            ),
            mock.patch(
                "psse_env.dagger.three_source_view."
                "audit_dagger1_final_view_support",
                return_value=self.final_support,
            ),
            mock.patch(
                "psse_env.sft.round1_source_gate."
                "audit_dagger1_final_view_support",
                return_value=self.final_support,
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _source_row(name: str, root: str, **updates: Any) -> dict[str, Any]:
        state_id = f"{name}:s0"
        row = {
            "example_id": name,
            "scenario_id": name,
            "root_scenario_id": name,
            "physical_root_fingerprint": root,
            "dataset_mode": "production",
            "production_label_eligible": True,
            "policy_observation": {
                "active_state_id": state_id,
                "candidate_state_id": None,
                "candidate_parent_id": None,
                "episode_id": name,
                "remaining_budget": 6,
                "history_window": [],
                "unresolved_signatures": [],
                "remaining_anomaly_score": None,
                "no_material_anomaly_remaining": False,
            },
            "history_window": [],
            "preferred_action": {
                "tool": "run_wls",
                "arguments": {"state_id": state_id},
            },
            "labels": {"dataset_mode": "production"},
        }
        row.update(updates)
        return row

    def _write_valid_artifacts(self, root: Path) -> tuple[Path, Path]:
        d0_rows = [self._source_row("d0", "root-d0", replay_source="d0_bc0")]
        d1_rows = [
            self._source_row(
                "d1", "root-d1", replay_source="natural_dagger1"
            )
        ]
        probe_identity = {
            "dataset_source": "observable_recovery_probe",
            "replay_source": "observable_recovery_probe",
            "collector_contract": "dagger1_observable_recovery_probe_v1",
            "state_origin": "observable_recovery_probe",
            "collection_role": "auxiliary_training",
            "state_visited_by": "observable_recovery_probe",
            "auxiliary_training_eligible": True,
            "production_label_eligible": False,
            "natural_on_policy_support_eligible": False,
            "training_decision_evidence_verified": True,
            "generation_provenance_id": PROBE_PROVENANCE_ID,
        }
        probe_rows = [
            self._source_row(
                "probe-a",
                "root-probe-a",
                recovery_stratum="post_failure_no_candidate",
                **probe_identity,
            ),
            self._source_row(
                "probe-b",
                "root-probe-b",
                recovery_stratum="unsupported_correction_recovery",
                **probe_identity,
            ),
        ]
        raw_view, training_view = build_dagger1_three_source_view(
            d0_rows=d0_rows,
            natural_d1_rows=d1_rows,
            probe_rows=probe_rows,
            policy=TEST_VIEW_POLICY,
        )
        validation_rows = [self._source_row("validation", "root-validation")]
        test_rows = [self._source_row("test", "root-test")]

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
            "deterministic_collection_selection_binding": {"passed": True},
            "three_source_training_support": self.support,
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
        audit_reports = {
            "d1_offline_teacher_target_quarantine_summary": quarantine_summary,
            "d1_recovery_label_audit": recomputed_d1["recovery_label_audit"],
            "d1_target_aware_state_class_audit": recomputed_d1[
                "target_aware_state_class_audit"
            ],
            "d1_independent_root_support": recomputed_d1[
                "independent_root_support"
            ],
            "d1_deterministic_collection_selection_binding": recomputed_d1[
                "deterministic_collection_selection_binding"
            ],
            "d1_three_source_training_support": self.support,
            "final_view_support": self.final_support,
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
            "training_support_report_sha256": stable_json_sha256(self.support),
            "final_view_support_report_sha256": stable_json_sha256(
                self.final_support
            ),
            "round1_view_policy": TEST_VIEW_POLICY,
            "round1_view_policy_digest": round1_view_policy_digest(
                TEST_VIEW_POLICY
            ),
            "audit_report_sha256": audit_hashes,
            "input_artifacts": {
                "d0_manifest_sha256": D0_MANIFEST_SHA256,
                "d1_rows_sha256": "e" * 64,
                "d1_manifest_sha256": COLLECTION_MANIFEST_SHA256,
                "d1_manifest_content_sha256": stable_json_sha256(d1_manifest),
                "probe_rows_sha256": "6" * 64,
                "probe_manifest_sha256": "7" * 64,
                "probe_generation_provenance_id": PROBE_PROVENANCE_ID,
                "immutable_source_view_content_sha256": {
                    "d0_bc0": stable_json_sha256(d0_rows),
                    "natural_dagger1": stable_json_sha256(d1_rows),
                    "observable_recovery_probe": stable_json_sha256(probe_rows),
                },
                "immutable_holdout_content_sha256": {
                    "validation": stable_json_sha256(validation_rows),
                    "test": stable_json_sha256(test_rows),
                },
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
        probe_binding = {
            "contract": "dagger1_recovery_probe_binding_v1",
            "passed": True,
            "generation_provenance_id": PROBE_PROVENANCE_ID,
            "probe_rows": len(probe_rows),
            "probe_roots": len(probe_rows),
            "probe_root_set_sha256": "8" * 64,
            "rows_sha256": "6" * 64,
            "manifest_sha256": "7" * 64,
            "view_policy_digest": round1_view_policy_digest(TEST_VIEW_POLICY),
            "support": {"passed": True},
        }
        descriptor["input_artifacts"]["probe_binding_report_sha256"] = (
            stable_json_sha256(probe_binding)
        )
        provenance_id = stable_json_sha256(descriptor)
        for row in raw_view:
            row["generation_provenance_id"] = provenance_id
        for row in [*validation_rows, *test_rows]:
            row["generation_provenance_id"] = provenance_id
        train_rows = examples_to_chat_sft(
            raw_view,
            protocol="canonical",
            allow_ineligible_auxiliary=True,
        )

        def write_rows(name: str, rows: list[dict[str, Any]]) -> None:
            (root / name).write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

        write_rows("aggregate.d0.raw.jsonl", d0_rows)
        write_rows("aggregate.d1.raw.jsonl", d1_rows)
        write_rows("aggregate.probe.raw.jsonl", probe_rows)
        write_rows("aggregate.raw.jsonl", [*d0_rows, *d1_rows, *probe_rows])
        write_rows("aggregate.train_view.raw.jsonl", raw_view)
        write_rows("aggregate.train_view.jsonl", train_rows)
        write_rows("aggregate.validation.jsonl", validation_rows)
        write_rows("aggregate.test.jsonl", test_rows)
        provenance = {
            "release_eligible": True,
            "generation_descriptor": descriptor,
            "generation_provenance_id": provenance_id,
            "dataset_hashes": {
                name: file_sha256(root / name)
                for name in ROUND1_IMMUTABLE_VIEW_NAMES
            },
            "probe_binding": probe_binding,
            "final_view_support": self.final_support,
            "release_checks": {"final_view_support": True},
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
            "probe_binding": probe_binding,
            "three_source_training_support": self.support,
            "final_view_support": self.final_support,
            "release_checks": {"final_view_support": True},
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

    def _forge_all_final_support_records(
        self,
        provenance_path: Path,
        preflight_path: Path,
        forged: dict[str, Any],
    ) -> None:
        """Rewrite every claimed copy/hash while leaving placed rows unchanged."""

        root = provenance_path.parent
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["training_view"]["final_view_support"] = forged
        preflight["final_view_support"] = forged
        report_hash = stable_json_sha256(forged)
        preflight["audit_report_sha256"]["final_view_support"] = report_hash

        descriptor = provenance["generation_descriptor"]
        descriptor["training_view_report_sha256"] = stable_json_sha256(
            preflight["training_view"]
        )
        descriptor["final_view_support_report_sha256"] = report_hash
        descriptor["audit_report_sha256"] = preflight["audit_report_sha256"]
        provenance["final_view_support"] = forged
        provenance_id = stable_json_sha256(descriptor)
        provenance["generation_provenance_id"] = provenance_id
        preflight["generation_provenance_id"] = provenance_id

        def load_rows(name: str) -> list[dict[str, Any]]:
            return [
                json.loads(line)
                for line in (root / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        def write_rows(name: str, rows: list[dict[str, Any]]) -> None:
            (root / name).write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

        raw_rows = load_rows("aggregate.train_view.raw.jsonl")
        for row in raw_rows:
            row["generation_provenance_id"] = provenance_id
        write_rows("aggregate.train_view.raw.jsonl", raw_rows)
        write_rows(
            "aggregate.train_view.jsonl",
            examples_to_chat_sft(
                raw_rows,
                protocol="canonical",
                allow_ineligible_auxiliary=True,
            ),
        )
        for name in ("aggregate.validation.jsonl", "aggregate.test.jsonl"):
            rows = load_rows(name)
            for row in rows:
                row["generation_provenance_id"] = provenance_id
            write_rows(name, rows)
        provenance["dataset_hashes"] = {
            name: file_sha256(root / name)
            for name in ROUND1_IMMUTABLE_VIEW_NAMES
        }
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
            self.assertEqual(report["d1_recovery_rows"], 1)

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
            self.assertIn("source gate passed", output.getvalue())

    def test_source_gate_requires_canonical_sibling_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight = self._write_valid_artifacts(root)
            renamed = root / "renamed-provenance.json"
            renamed.write_bytes(provenance.read_bytes())
            with self.assertRaisesRegex(GateError, "canonical sibling"):
                self._validate(renamed, preflight)

            with self.assertRaisesRegex(GateError, "canonical dataset path"):
                validate_round1_source_mix_gate(
                    provenance,
                    preflight,
                    reviewed_source_commit=SOURCE_COMMIT,
                    initial_adapter_revision=ADAPTER_REVISION,
                    train_path=root / "alternate.jsonl",
                    validation_path=root / "aggregate.validation.jsonl",
                )

    def test_rehashed_holdout_tamper_still_fails_descriptor_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance_path, preflight = self._write_valid_artifacts(root)
            validation_path = root / "aggregate.validation.jsonl"
            row = json.loads(validation_path.read_text(encoding="utf-8"))
            row["forged"] = True
            validation_path.write_text(
                json.dumps(row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            provenance["dataset_hashes"][validation_path.name] = file_sha256(
                validation_path
            )
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateError, "descriptor-bound holdouts"):
                self._validate(provenance_path, preflight)

    def test_embedded_d1_manifest_content_is_descriptor_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight_path = self._write_valid_artifacts(root)
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["d1_collection_manifest"]["unreviewed_claim"] = True
            preflight_path.write_text(
                json.dumps(preflight, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateError, "bind D1 rows and manifest"):
                self._validate(provenance, preflight_path)

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

    def test_rehashed_forged_training_view_still_fails_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provenance, preflight = self._write_valid_artifacts(root)
            view = root / "aggregate.train_view.raw.jsonl"
            view.write_text(json.dumps({"forged": "wrong allocation"}) + "\n")
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            payload["dataset_hashes"][view.name] = file_sha256(view)
            provenance.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateError, "exact deterministic three-source reconstruction"
            ):
                self._validate(provenance, preflight)

    def test_forged_final_support_cannot_be_rehashed_into_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            forged = {
                **self.final_support,
                "forged_final_placement_claim": True,
            }
            self._forge_all_final_support_records(
                provenance,
                preflight,
                forged,
            )
            with self.assertRaisesRegex(
                GateError,
                "final placed-view support does not recompute exactly",
            ):
                self._validate(provenance, preflight)

    def test_placed_natural_floor_shortfall_fails_source_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provenance, preflight = self._write_valid_artifacts(Path(temp_dir))
            scarce_floor_failure = {
                "contract": FINAL_VIEW_SUPPORT_CONTRACT,
                "natural_rows": 1,
                "probe_rows": 2,
                "training_support": {
                    "contract": "test_support",
                    "natural_on_policy_support": {
                        "targeted_state_cell_shortfalls": {
                            "scarce_natural_cell": {
                                "minimum_distinct_physical_roots": 1,
                                "distinct_physical_roots": 0,
                                "root_shortfall": 1,
                                "passed": False,
                            }
                        },
                        "passed": False,
                    },
                    "observable_probe_support": {"passed": True},
                    "combined_training_support": {"passed": True},
                    "passed": False,
                },
                "probe_source_coverage": {
                    "all_unique_probe_source_rows_placed": True,
                    "passed": True,
                },
                "passed": False,
            }
            with mock.patch(
                "psse_env.sft.round1_source_gate."
                "audit_dagger1_final_view_support",
                return_value=scarce_floor_failure,
            ) as recompute:
                with self.assertRaisesRegex(
                    GateError,
                    "final placed-view support does not recompute exactly",
                ):
                    self._validate(provenance, preflight)
            call = recompute.call_args.kwargs
            self.assertEqual(
                [row["example_id"] for row in call["natural_rows"]],
                ["d1"],
            )
            self.assertEqual(len(call["probe_rows"]), 2)
            self.assertEqual(len(call["source_probe_rows"]), 2)

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
