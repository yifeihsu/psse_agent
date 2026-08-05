from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from psse_env.dagger.collect_dagger1 import (
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_FORBIDDEN_SUITE,
    frozen_physical_roots,
    main as collect_dagger1_main,
    recommended_collection_gate,
    targeted_state_coverage,
    validate_collection_pass,
    validate_collection_output_paths,
    validate_d0_provenance_binding,
    validate_development_holdout_binding,
    validate_export_rows_truth_free,
    validate_scenario_builder_manifest,
    validate_training_learner_seed,
    validate_training_source_report,
    validate_training_scenarios,
)
from psse_env.dagger.build_dagger1_development_holdout import (
    DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
    DAGGER1_DEVELOPMENT_SPLIT,
    DAGGER1_DEVELOPMENT_SUITE_NAME,
    DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
)
from psse_env.dagger.rollout_collector import RECOMMENDED_DAGGER1_RECOVERY_STRATA
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
)
from psse_env.dagger.rollout_collector import (
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    summarize_dagger1_offline_teacher_target_quarantine,
)
from psse_env.sft.provenance import stable_json_sha256


class Dagger1CollectionSafetyTests(unittest.TestCase):
    @staticmethod
    def _release_threshold_report():
        return {
            "source_partition": {"enabled": True, "selected": "train"},
            "parameter_ranking_admission": {
                "contract": "distinct_line_abs_lambda_dominance_v1",
                "enforced": True,
                "threshold": 1.0,
            },
        }

    def test_frozen_suite_provides_explicit_nonempty_physical_holdout(self) -> None:
        roots = frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE)
        self.assertEqual(len(roots), 115)

    def test_relabelled_train_scenario_cannot_reuse_frozen_root(self) -> None:
        roots = frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE)
        frozen_root = sorted(roots)[0]
        with self.assertRaisesRegex(ValueError, "protected D0/evaluation root"):
            validate_training_scenarios(
                [
                    {
                        "scenario_id": "relabeled-evaluation-root",
                        "dataset_split": "train",
                        "physical_root_fingerprint": frozen_root,
                        "case": "case14",
                        "measurements": [],
                    }
                ],
                forbidden_roots=roots,
            )

    def test_truth_bearing_collection_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not truth-free"):
            validate_training_scenarios(
                [
                    {
                        "scenario_id": "truth-bearing",
                        "split": "dagger_train",
                        "physical_root_fingerprint": "new-training-root",
                        "case": "case14",
                        "measurements": [],
                        "true_parameter_errors": [{"line_index1": 1}],
                    }
                ],
                forbidden_roots=frozenset({"held-out-root"}),
            )

    def test_nested_metadata_truth_and_schema_audit_fail_closed(self) -> None:
        base = {
            "scenario_id": "nested-truth",
            "split": "dagger_train",
            "physical_root_fingerprint": "new-training-root",
            "case": "case14",
            "measurements": [],
        }
        for injected, expected_path in (
            (
                {"metadata": {"hidden_truth": {"fault": 1}}},
                r"\$\.metadata\.hidden_truth",
            ),
            (
                {"metadata": {"nested": {"true_topology_errors": []}}},
                r"\$\.metadata\.nested\.true_topology_errors",
            ),
            (
                {"metadata": {"nested": {"true_hif_errors": []}}},
                r"\$\.metadata\.nested\.true_hif_errors",
            ),
            (
                {"audit": {"expected_outcome": "resolved"}},
                r"\$\.audit",
            ),
        ):
            with self.subTest(expected_path=expected_path):
                with self.assertRaisesRegex(ValueError, expected_path):
                    validate_training_scenarios(
                        [{**base, **injected}],
                        forbidden_roots=frozenset({"held-out-root"}),
                    )

    def test_non_holdout_truth_free_training_scenario_passes(self) -> None:
        validate_training_scenarios(
            [
                {
                    "scenario_id": "new-training-root",
                    "split": "dagger_train",
                    "physical_root_fingerprint": "new-training-root",
                    "case": "case14",
                    "measurements": [],
                }
            ],
            forbidden_roots=frozenset({"held-out-root"}),
        )

    def test_private_truth_envelope_is_allowed_but_execution_leak_is_not(self):
        envelope = {
            "scenario_schema_version": 1,
            "execution": {
                "scenario_id": "fresh-envelope",
                "case": {},
                "measurements": [1.0],
            },
            "audit": {
                "truth": {
                    "truth_complete": True,
                    "clean_measurements": [1.0],
                    "true_measurement_errors": [{"index": 0}],
                },
                "release_audit": {"generator_only": True},
            },
            "grouping": {
                "root_scenario_id": "fresh-envelope",
                "physical_root_fingerprint": "fresh-envelope-root",
                "scenario_family": "measurement",
                "error_cardinality": 1,
                "case_id": "case14",
                "split": "dagger_train",
                "source_tier": "generated",
            },
        }
        validate_training_scenarios(
            [envelope], forbidden_roots=frozenset({"held-out-root"})
        )
        leaked = {
            **envelope,
            "execution": {
                **envelope["execution"],
                "true_measurement_errors": [{"index": 0}],
            },
        }
        with self.assertRaisesRegex(ValueError, "execution/grouping is not truth-free"):
            validate_training_scenarios(
                [leaked], forbidden_roots=frozenset({"held-out-root"})
            )

    def test_collection_pass_enforces_diagnostic_and_training_beta(self) -> None:
        self.assertTrue(
            validate_collection_pass(
                collection_pass="diagnostic", beta=0.0
            )["passed"]
        )
        self.assertTrue(
            validate_collection_pass(
                collection_pass="training", beta=0.25
            )["passed"]
        )
        self.assertTrue(
            validate_collection_pass(
                collection_pass="training", beta=0.5
            )["passed"]
        )
        with self.assertRaisesRegex(ValueError, "diagnostic.*beta"):
            validate_collection_pass(collection_pass="diagnostic", beta=0.25)
        with self.assertRaisesRegex(ValueError, "training.*beta"):
            validate_collection_pass(collection_pass="training", beta=0.0)

    def test_training_cli_requires_full_all_output_audit_ledger(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            collect_dagger1_main(
                [
                    "--input",
                    "scenarios.json",
                    "--d0-aggregate-dir",
                    "d0",
                    "--scenario-generator-report",
                    "generator.json",
                    "--scenario-manifest",
                    "scenarios.manifest.json",
                    "--output",
                    "eligible.jsonl",
                    "--model-id",
                    "/scratch/adapter",
                    "--model-revision",
                    "a" * 64,
                    "--collection-pass",
                    "training",
                    "--beta",
                    "0.25",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --all-output", stderr.getvalue())

    def test_training_cli_requires_development_holdout_boundary(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            collect_dagger1_main(
                [
                    "--input",
                    "scenarios.json",
                    "--d0-aggregate-dir",
                    "d0",
                    "--scenario-generator-report",
                    "generator.json",
                    "--scenario-manifest",
                    "scenarios.manifest.json",
                    "--output",
                    "eligible.jsonl",
                    "--all-output",
                    "all.jsonl",
                    "--model-id",
                    "/scratch/adapter",
                    "--model-revision",
                    "a" * 64,
                    "--collection-pass",
                    "training",
                    "--beta",
                    "0.25",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --development-holdout", stderr.getvalue())

    def test_d0_provenance_must_be_clean_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "aggregate.raw.jsonl"
            raw_path.write_text("{}\n", encoding="utf-8")
            source_state = {
                "source_commit": "a" * 40,
                "release_eligible_source": True,
            }
            descriptor = {"source_state": dict(source_state)}
            valid = {
                "release_eligible": True,
                "generation_descriptor": descriptor,
                "generation_provenance_id": stable_json_sha256(descriptor),
                "dataset_hashes": {
                    raw_path.name: hashlib.sha256(raw_path.read_bytes()).hexdigest()
                },
            }
            validate_d0_provenance_binding(
                valid,
                raw_path=raw_path,
                source_state=source_state,
            )

            bad_id = copy.deepcopy(valid)
            bad_id["generation_provenance_id"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "generation_provenance_id"):
                validate_d0_provenance_binding(
                    bad_id,
                    raw_path=raw_path,
                    source_state=source_state,
                )

            dirty = copy.deepcopy(valid)
            dirty["generation_descriptor"]["source_state"][
                "release_eligible_source"
            ] = False
            dirty["generation_provenance_id"] = stable_json_sha256(
                dirty["generation_descriptor"]
            )
            with self.assertRaisesRegex(RuntimeError, "release_eligible_source"):
                validate_d0_provenance_binding(
                    dirty,
                    raw_path=raw_path,
                    source_state=source_state,
                )

    def test_development_holdout_is_byte_and_training_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                name: root / name
                for name in (
                    "scenarios.json",
                    "scenarios.manifest.json",
                    "aggregate.raw.jsonl",
                    "aggregate.generation_provenance.json",
                    "suite.json",
                    "policy.json",
                    "development.json",
                    "development.json.manifest.json",
                )
            }
            for index, name in enumerate(
                (
                    "scenarios.json",
                    "scenarios.manifest.json",
                    "aggregate.raw.jsonl",
                    "aggregate.generation_provenance.json",
                    "suite.json",
                    "policy.json",
                )
            ):
                paths[name].write_text(f"fixture-{index}\n", encoding="utf-8")

            rows = []
            for family, count in DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items():
                for index in range(count):
                    rows.append(
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
            paths["development.json"].write_text(
                json.dumps({DAGGER1_DEVELOPMENT_SUITE_NAME: rows}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            roots = sorted(
                row["grouping"]["physical_root_fingerprint"] for row in rows
            )
            source_state = {
                "source_commit": "a" * 40,
                "release_eligible_source": True,
            }

            def digest(name: str) -> str:
                return hashlib.sha256(paths[name].read_bytes()).hexdigest()

            manifest = {
                "schema_version": 1,
                "builder_contract": DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
                "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
                "split": DAGGER1_DEVELOPMENT_SPLIT,
                "source_state": source_state,
                "source_bindings": {
                    "psse_env/dagger/build_dagger1_development_holdout.py": (
                        hashlib.sha256(
                            (
                                Path(__file__).resolve().parents[2]
                                / "psse_env"
                                / "dagger"
                                / "build_dagger1_development_holdout.py"
                            ).read_bytes()
                        ).hexdigest()
                    )
                },
                "plan": DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
                "selected_count_by_family": DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
                "scenario_count": 30,
                "physical_root_count": 30,
                "root_set_sha256": {
                    "development": stable_json_sha256(roots)
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
                "output_sha256": digest("development.json"),
                "d1_training_scenarios_sha256": digest("scenarios.json"),
                "d1_training_manifest_sha256": digest(
                    "scenarios.manifest.json"
                ),
                "d0_raw_sha256": digest("aggregate.raw.jsonl"),
                "d0_generation_provenance_sha256": digest(
                    "aggregate.generation_provenance.json"
                ),
                "frozen_suite_sha256": digest("suite.json"),
                "evaluation_policy_sha256": digest("policy.json"),
            }
            paths["development.json.manifest.json"].write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            kwargs = {
                "source_state": source_state,
                "scenario_input_path": paths["scenarios.json"],
                "scenario_manifest_path": paths["scenarios.manifest.json"],
                "d0_raw_path": paths["aggregate.raw.jsonl"],
                "d0_provenance_path": paths[
                    "aggregate.generation_provenance.json"
                ],
                "forbidden_suite_path": paths["suite.json"],
                "evaluation_policy_path": paths["policy.json"],
                "require_model_selection_eligible": True,
            }
            self.assertEqual(
                len(
                    validate_development_holdout_binding(
                        paths["development.json"],
                        paths["development.json.manifest.json"],
                        **kwargs,
                    )
                ),
                30,
            )

            tampered = dict(manifest)
            tampered["d1_training_manifest_sha256"] = "f" * 64
            paths["development.json.manifest.json"].write_text(
                json.dumps(tampered, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "d1_training_manifest_sha256",
            ):
                validate_development_holdout_binding(
                    paths["development.json"],
                    paths["development.json.manifest.json"],
                    **kwargs,
                )

            tampered_source = copy.deepcopy(manifest)
            tampered_source["source_bindings"][
                "psse_env/dagger/build_dagger1_development_holdout.py"
            ] = "f" * 64
            paths["development.json.manifest.json"].write_text(
                json.dumps(tampered_source, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source_bindings"):
                validate_development_holdout_binding(
                    paths["development.json"],
                    paths["development.json.manifest.json"],
                    **kwargs,
                )

    def test_diagnostic_cli_keeps_all_output_optional(self) -> None:
        with (
            patch(
                "psse_env.dagger.collect_dagger1.git_source_state",
                return_value={
                    "source_commit": "a" * 40,
                    "release_eligible_source": True,
                },
            ),
            self.assertRaises(FileNotFoundError),
        ):
            collect_dagger1_main(
                [
                    "--input",
                    "missing-scenarios.json",
                    "--d0-aggregate-dir",
                    "missing-d0",
                    "--scenario-generator-report",
                    "missing-generator.json",
                    "--scenario-manifest",
                    "missing-scenarios.manifest.json",
                    "--output",
                    "diagnostic.jsonl",
                    "--model-id",
                    "unsloth/gemma-4-31B-it",
                    "--model-revision",
                    "a" * 40,
                    "--collection-pass",
                    "diagnostic",
                    "--beta",
                    "0",
                ]
            )

    def test_training_learner_seed_requires_verified_absolute_adapter(self) -> None:
        revision = "a" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir).resolve() / "lora"
            adapter.mkdir()
            inspection = {
                "path": str(adapter),
                "tree_sha256": revision,
                "file_count": 7,
                "total_bytes": 1234,
            }
            with patch(
                "psse_env.dagger.collect_dagger1.inspect_release_checkpoint",
                return_value=inspection,
            ) as inspect_checkpoint:
                identity = validate_training_learner_seed(
                    model_id=str(adapter),
                    model_revision=revision.upper(),
                )

            inspect_checkpoint.assert_called_once_with(adapter)
            self.assertEqual(
                identity,
                {
                    "role": "learner_seed_only",
                    "collection_model_id": str(adapter),
                    "collection_model_revision": revision,
                    "adapter_tree_sha256": revision,
                    "adapter_file_count": 7,
                    "adapter_total_bytes": 1234,
                },
            )

            with self.assertRaisesRegex(ValueError, "64-hex adapter tree"):
                validate_training_learner_seed(
                    model_id=str(adapter), model_revision="b" * 40
                )
            with self.assertRaisesRegex(ValueError, "absolute local adapter"):
                validate_training_learner_seed(
                    model_id="relative/lora", model_revision=revision
                )
            with (
                patch(
                    "psse_env.dagger.collect_dagger1.inspect_release_checkpoint",
                    return_value={**inspection, "tree_sha256": "c" * 64},
                ),
                self.assertRaisesRegex(ValueError, "tree digest mismatch"),
            ):
                validate_training_learner_seed(
                    model_id=str(adapter), model_revision=revision
                )

    def test_source_report_requires_round0_generator_train_partition(self) -> None:
        validate_training_source_report(self._release_threshold_report())
        for report in (
            {},
            {"source_partition": {"enabled": False, "selected": None}},
            {"source_partition": {"enabled": True, "selected": "evaluation"}},
            {
                "source_partition": {"enabled": True, "selected": "train"},
                "parameter_ranking_admission": {
                    "enforced": True,
                    "threshold": 1.2,
                },
            },
        ):
            with self.subTest(report=report):
                with self.assertRaisesRegex(
                    ValueError, "source_partition|parameter-ranking"
                ):
                    validate_training_source_report(report)

    def test_builder_manifest_cryptographically_binds_input_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                name: root / name
                for name in (
                    "scenarios.json",
                    "generator.json",
                    "aggregate.raw.jsonl",
                    "aggregate.generation_provenance.json",
                    "suite.json",
                    "policy.json",
                )
            }
            for index, path in enumerate(paths.values()):
                path.write_text(f"payload-{index}\n", encoding="utf-8")

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            source_state = {
                "source_commit": "a" * 40,
                "release_eligible_source": True,
            }
            manifest = {
                "schema_version": 1,
                "builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
                "release_evidence_eligible": False,
                "source_state": source_state,
                "source_partition": "train",
                "parameter_ranking_dominance_threshold": 1.0,
                "plan": {"multi_measurement": 1},
                "selected_count_by_family": {"multi_measurement": 1},
                "scenario_count": 1,
                "physical_root_count": 1,
                "protected_root_overlap": [],
                "output_sha256": digest(paths["scenarios.json"]),
                "generator_report_sha256": digest(paths["generator.json"]),
                "d0_raw_sha256": digest(paths["aggregate.raw.jsonl"]),
                "d0_generation_provenance_sha256": digest(
                    paths["aggregate.generation_provenance.json"]
                ),
                "frozen_suite_sha256": digest(paths["suite.json"]),
                "evaluation_policy_sha256": digest(paths["policy.json"]),
            }
            kwargs = {
                "scenarios": [
                    {
                        "grouping": {
                            "scenario_family": "multi_measurement",
                            "physical_root_fingerprint": "fresh-root",
                        }
                    }
                ],
                "input_path": paths["scenarios.json"],
                "generator_report_path": paths["generator.json"],
                "source_state": source_state,
                "d0_raw_path": paths["aggregate.raw.jsonl"],
                "d0_provenance_path": paths[
                    "aggregate.generation_provenance.json"
                ],
                "forbidden_suite_path": paths["suite.json"],
                "evaluation_policy_path": paths["policy.json"],
            }
            validate_scenario_builder_manifest(manifest, **kwargs)
            paths["generator.json"].write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generator_report_sha256"):
                validate_scenario_builder_manifest(manifest, **kwargs)

    def test_collection_outputs_cannot_alias_or_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "scenarios.json"
            input_path.write_text("[]\n", encoding="utf-8")
            output = root / "rows.jsonl"
            with self.assertRaisesRegex(ValueError, "alias protected"):
                validate_collection_output_paths(
                    output=input_path,
                    all_output=None,
                    protected_paths=(input_path,),
                )
            with self.assertRaisesRegex(ValueError, "mutually distinct"):
                validate_collection_output_paths(
                    output=output,
                    all_output=output,
                    protected_paths=(input_path,),
                )
            output.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                validate_collection_output_paths(
                    output=output,
                    all_output=None,
                    protected_paths=(input_path,),
                )

    def test_dynamic_export_truth_leak_fails_closed(self):
        validate_export_rows_truth_free(
            [{"policy_observation": {"active_state_id": "active"}}]
        )
        with self.assertRaisesRegex(RuntimeError, "private oracle truth"):
            validate_export_rows_truth_free(
                [
                    {
                        "labels": {
                            "nested": {
                                "true_parameter_errors": [{"line_index1": 1}]
                            }
                        }
                    }
                ]
            )

    def test_recommended_collection_gate_checks_rows_and_all_strata(self) -> None:
        rows = [
            {"recovery_stratum": stratum}
            for stratum in sorted(RECOMMENDED_DAGGER1_RECOVERY_STRATA)
            for _ in range(50)
        ]
        report = recommended_collection_gate(rows)
        self.assertEqual(len(rows), 300)
        self.assertTrue(report["passed"])
        self.assertTrue(report["recommended_row_target"]["passed"])
        self.assertTrue(report["recovery_strata_passed"])

        incomplete = recommended_collection_gate(rows[:299])
        self.assertFalse(incomplete["passed"])
        self.assertFalse(incomplete["recommended_row_target"]["passed"])

    def test_truth_audit_quarantine_counts_only_pre_audit_training_candidates(self):
        passed_audit = {
            "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
            "passed": True,
            "action_class": "rollback",
            "checks": {
                "candidate_exists": True,
                "candidate_verified": True,
                "candidate_source_truth_evidence_complete": True,
                "candidate_not_truth_safe_to_commit": True,
                "observable_evidence_gate_passed": True,
            },
            "reason_codes": [],
        }
        failed_audit = copy.deepcopy(passed_audit)
        failed_audit["passed"] = False
        failed_audit["checks"][
            "candidate_source_truth_evidence_complete"
        ] = False
        failed_audit["reason_codes"] = [
            "candidate_source_correction_missing"
        ]
        candidate = {
            "supervision_policy": DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            "collection_role": "training",
            "state_origin": "learner_policy",
            "recovery_stratum": "rejected_candidate_rollback",
            "preferred_action": {
                "tool": "rollback_state",
                "arguments": {"candidate_state_id": "candidate"},
            },
            "observable_rank_one_target_proof": {"passed": True},
            "labels": {"training_decision_evidence_verified": True},
        }
        rows = [
            {
                **copy.deepcopy(candidate),
                "example_id": "passed-candidate",
                "production_label_eligible": True,
                "offline_teacher_target_audit": passed_audit,
            },
            {
                **copy.deepcopy(candidate),
                "example_id": "quarantined-candidate",
                "production_label_eligible": False,
                "offline_teacher_target_audit": failed_audit,
            },
            {
                **copy.deepcopy(candidate),
                "example_id": "diagnostic-row",
                "collection_role": "diagnostic",
                "offline_teacher_target_audit": failed_audit,
            },
            {
                **copy.deepcopy(candidate),
                "example_id": "initial-row",
                "state_origin": "initial",
                "offline_teacher_target_audit": failed_audit,
            },
        ]

        report = summarize_dagger1_offline_teacher_target_quarantine(rows)
        self.assertEqual(report["candidate_rows"], 2)
        self.assertEqual(report["non_candidate_rows"], 2)
        self.assertEqual(report["passed_rows"], 1)
        self.assertEqual(report["quarantined_rows"], 1)
        self.assertEqual(
            report["quarantined_example_ids"], ["quarantined-candidate"]
        )
        self.assertFalse(report["zero_truth_audit_quarantine"])
        self.assertFalse(report["passed"])

    def test_targeted_state_coverage_requires_each_independent_root_cell(self):
        rows = [
            {
                "physical_root_fingerprint": f"multi-{cardinality}",
                "scenario_family": "multi_measurement",
                "error_cardinality": cardinality,
                "parameter_scans_available": False,
                "policy_observation": {},
            }
            for cardinality in (2, 4, 5)
        ]
        for route in ("actionable", "complete_negative", "unavailable_or_inconclusive"):
            rows.append(
                {
                    "physical_root_fingerprint": f"route-{route}",
                    "scenario_family": "parameter",
                    "error_cardinality": 1,
                    "policy_observation": {
                        "fresh_context_evidence": {
                            "parameter": {
                                "route_status": route,
                                "parameter_ranking_dominance_ratio": (
                                    1.1 if route == "actionable" else None
                                ),
                            }
                        }
                    },
                }
            )
        for first, second in (("measurement", "parameter"), ("parameter", "measurement")):
            rows.append(
                {
                    "physical_root_fingerprint": f"sequence-{first}",
                    "scenario_family": "measurement+parameter",
                    "error_cardinality": 2,
                    "preferred_action": {
                        "tool": f"get_{second}_context",
                        "arguments": {"state_id": "active"},
                    },
                    "policy_observation": {
                        "history_window": [
                            {
                                "action": {
                                    "tool": f"get_{first}_context",
                                    "arguments": {"state_id": "active"},
                                }
                            }
                        ],
                        "accepted_corrections": (
                            [{"source_action": {"tool": "correct_measurements"}}]
                            if first == "measurement"
                            else []
                        ),
                        "no_material_anomaly_remaining": False,
                    },
                }
            )
        rows = [
            {
                **row,
                "physical_root_fingerprint": (
                    f"{row['physical_root_fingerprint']}-root-{root_index}"
                ),
            }
            for row in rows
            for root_index in range(5)
        ]
        report = targeted_state_coverage(rows)
        self.assertTrue(report["passed"], report)
        with_scans = [dict(row) for row in rows]
        with_scans[0]["parameter_scans_available"] = True
        self.assertFalse(targeted_state_coverage(with_scans)["passed"])
        rows.pop()
        self.assertFalse(targeted_state_coverage(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
