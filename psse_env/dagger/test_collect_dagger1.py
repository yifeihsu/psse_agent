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

import psse_env.dagger.collect_dagger1 as collect_module
from psse_env.dagger.collect_dagger1 import (
    DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
    DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY,
    DAGGER1_RESERVE_FAMILY_PRIORITY,
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_FORBIDDEN_SUITE,
    FAILED_COLLECTION_ALL_ROWS,
    ANALYSIS_COMPLETE_ARTIFACT_TYPE,
    FAILED_COLLECTION_ARTIFACT_TYPE,
    FAILED_COLLECTION_CANDIDATE_ROWS,
    FAILED_COLLECTION_CHECKSUMS,
    FAILED_COLLECTION_EVIDENCE,
    collect_dagger1_rollout_schedule,
    dagger1_production_row_target_contract,
    dagger1_rollout_batches,
    dagger1_rollout_seed,
    evaluate_dagger1_collection_checkpoint,
    failed_strict_collection_gate_names,
    frozen_physical_roots,
    main as collect_dagger1_main,
    recommended_collection_gate,
    select_dagger1_collection_rows,
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
    write_failed_collection_evidence_bundle,
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

    def test_reviewed_production_row_bounds_are_fail_closed(self):
        reviewed = dagger1_production_row_target_contract(
            target_min_rows=300,
            target_max_rows=600,
        )
        exploratory = dagger1_production_row_target_contract(
            target_min_rows=90,
            target_max_rows=100,
        )
        self.assertTrue(reviewed["passed"])
        self.assertFalse(reviewed["exploratory_override"])
        self.assertFalse(exploratory["passed"])
        self.assertTrue(exploratory["exploratory_override"])
        self.assertEqual(exploratory["required_target_min_rows"], 300)
        self.assertEqual(exploratory["required_target_max_rows"], 600)

    def test_round1_publication_contract_is_symmetric_and_fail_closed(self):
        expected_go = {
            "strict_gate_passed": True,
            "round1_aggregate_eligible": True,
            "production_outputs_published": True,
        }
        self.assertEqual(
            collect_module._round1_publication_contract(True),
            expected_go,
        )

        expected_no_go = {key: False for key in expected_go}
        for ineligible in (False, None, 1, "true"):
            with self.subTest(ineligible=ineligible):
                self.assertEqual(
                    collect_module._round1_publication_contract(ineligible),
                    expected_no_go,
                )

    def test_reviewed_mixed_family_coverage_buffer_is_exact(self):
        self.assertEqual(
            DAGGER1_SCENARIO_BUILDER_CONTRACT,
            "fresh_train_partition_dagger1_scenarios_v4",
        )
        self.assertEqual(
            DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
            "dagger1_predeclared_collection_schedule_v2",
        )
        self.assertEqual(
            collect_module.DAGGER1_CANDIDATE_REQUEST_PLAN,
            {
                "measurement+parameter": 108,
                "multi_measurement": 176,
                "parameter": 48,
            },
        )
        self.assertEqual(
            collect_module.DAGGER1_RESERVE_PLAN,
            {
                "measurement+parameter": 60,
                "multi_measurement": 31,
                "parameter": 0,
            },
        )
        self.assertEqual(
            collect_module.DAGGER1_TRAINING_POOL_PLAN,
            {
                "measurement+parameter": 108,
                "multi_measurement": 79,
                "parameter": 24,
            },
        )
        self.assertEqual(
            collect_module.DAGGER1_FRESH_CANDIDATE_COUNT_BY_FAMILY,
            {
                "measurement+parameter": 108,
                "multi_measurement": 91,
                "parameter": 35,
            },
        )
        self.assertEqual(collect_module.DAGGER1_RAW_CANDIDATE_COUNT, 271)
        self.assertEqual(collect_module.DAGGER1_FRESH_CANDIDATE_COUNT, 234)
        self.assertEqual(
            DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY,
            {
                "measurement+parameter": 2,
                "multi_measurement": 3,
                "parameter": 1,
            },
        )
        self.assertEqual(
            collect_module.DAGGER1_BASE_RESERVE_PLAN,
            {
                "measurement+parameter": 48,
                "multi_measurement": 31,
                "parameter": 0,
            },
        )
        self.assertEqual(
            collect_module.DAGGER1_TOPUP_RESERVE_PLAN,
            {
                "measurement+parameter": 12,
                "multi_measurement": 0,
                "parameter": 0,
            },
        )

    @staticmethod
    def _scheduled_scenarios():
        specifications = (
            ("multi-primary", "multi_measurement", "primary", 0),
            ("mixed-primary", "measurement+parameter", "primary", 0),
            ("multi-reserve", "multi_measurement", "reserve", 1),
            ("mixed-reserve", "measurement+parameter", "reserve", 2),
            ("parameter-reserve", "parameter", "reserve", 3),
        )
        return [
            {
                "execution": {"scenario_id": name},
                "audit": {"truth": {"truth_complete": True}},
                "grouping": {
                    "physical_root_fingerprint": name,
                    "scenario_family": family,
                    "collection_cohort": cohort,
                    "collection_subcohort": (
                        "primary" if cohort == "primary" else "base_reserve"
                    ),
                    "collection_priority": priority,
                    "collection_order": order,
                    "split": "dagger_train",
                },
            }
            for order, (name, family, cohort, priority) in enumerate(
                specifications
            )
        ]

    @staticmethod
    def _strict_coverage_rows(extra_rows: int = 0):
        rows = []

        def add(
            prefix,
            count,
            *,
            stratum,
            family="measurement+parameter",
            cardinality=2,
            observation=None,
            preferred_action=None,
            parameter_scans_available=None,
        ):
            for index in range(count):
                rows.append(
                    {
                        "example_id": f"{prefix}-{index}",
                        "physical_root_fingerprint": f"{prefix}-root-{index}",
                        "production_label_eligible": True,
                        "recovery_stratum": stratum,
                        "scenario_family": family,
                        "error_cardinality": cardinality,
                        "parameter_scans_available": parameter_scans_available,
                        "policy_observation": copy.deepcopy(observation or {}),
                        "preferred_action": copy.deepcopy(preferred_action),
                    }
                )

        for cardinality in (2, 4, 5):
            add(
                f"multi-{cardinality}",
                5,
                stratum="multi_measurement_safe_handoff",
                family="multi_measurement",
                cardinality=cardinality,
                parameter_scans_available=False,
            )
        add(
            "route-actionable",
            5,
            stratum="premature_commit_recovery",
            observation={
                "fresh_context_evidence": {
                    "parameter": {
                        "route_status": "actionable",
                        "parameter_ranking_dominance_ratio": 1.1,
                    }
                }
            },
        )
        add(
            "route-negative",
            5,
            stratum="premature_escalation_recovery",
            observation={
                "fresh_context_evidence": {
                    "parameter": {"route_status": "complete_negative"}
                }
            },
        )
        add(
            "route-unavailable",
            5,
            stratum="unsupported_correction_recovery",
            observation={
                "fresh_context_evidence": {
                    "parameter": {
                        "route_status": "unavailable_or_inconclusive"
                    }
                }
            },
        )
        for first, second in (
            ("measurement", "parameter"),
            ("parameter", "measurement"),
        ):
            add(
                f"sequence-{first}",
                5,
                stratum="sequential_measurement_parameter_recovery",
                observation={
                    "history_window": [
                        {"action": {"tool": f"correct_{first}"}}
                    ]
                },
                preferred_action={
                    "tool": f"correct_{second}",
                    "arguments": {},
                },
            )
        add(
            "partial",
            5,
            stratum="post_failure_no_candidate",
            observation={
                "accepted_corrections": [{"target": "measurement:1"}],
                "no_material_anomaly_remaining": False,
            },
        )
        add(
            "unsupported-extra",
            5,
            stratum="unsupported_correction_recovery",
        )
        add("post-extra", 5, stratum="post_failure_no_candidate")
        for index in range(extra_rows):
            rows.append(
                {
                    "example_id": f"extra-{index}",
                    "physical_root_fingerprint": f"extra-root-{index}",
                    "production_label_eligible": True,
                    "recovery_stratum": "multi_measurement_safe_handoff",
                    "scenario_family": "measurement+parameter",
                    "error_cardinality": 2,
                    "policy_observation": {},
                }
            )
        return rows

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

    def test_strict_training_cli_requires_failed_collection_directory(self):
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
                    "--development-holdout",
                    "development.json",
                    "--development-holdout-manifest",
                    "development.manifest.json",
                    "--development-holdout-generator-report",
                    "development.generator.json",
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
                    "--require-recommended-target",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --failed-collection-dir", stderr.getvalue())

    def test_d0_provenance_must_be_clean_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "aggregate.raw.jsonl"
            raw_path.write_text("{}\n", encoding="utf-8")
            manifest_path = Path(temp_dir) / "aggregate.manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
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
                    raw_path.name: hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    manifest_path.name: hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
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

            manifest_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "aggregate_manifest_sha256"):
                validate_d0_provenance_binding(
                    valid,
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
                    "aggregate.manifest.json",
                    "suite.json",
                    "policy.json",
                    "development.json",
                    "development.json.manifest.json",
                    "development.generator.json",
                )
            }
            training_rows = [
                {
                    "scenario_schema_version": 1,
                    "execution": {},
                    "audit": {},
                    "grouping": {
                        "scenario_family": "parameter",
                        "physical_root_fingerprint": "training-root",
                    },
                }
            ]
            paths["scenarios.json"].write_text(
                json.dumps(training_rows, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths["aggregate.raw.jsonl"].write_text(
                json.dumps({"physical_root_fingerprint": "d0-root"}) + "\n",
                encoding="utf-8",
            )
            paths["aggregate.generation_provenance.json"].write_text(
                "{}\n", encoding="utf-8"
            )
            paths["aggregate.manifest.json"].write_text(
                "{}\n", encoding="utf-8"
            )
            paths["suite.json"].write_text(
                json.dumps(
                    {
                        "suite": [
                            {
                                "grouping": {
                                    "physical_root_fingerprint": "frozen-root"
                                }
                            }
                        ]
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            paths["policy.json"].write_text("{}\n", encoding="utf-8")
            paths["development.generator.json"].write_text(
                json.dumps(self._release_threshold_report(), sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            rows = []
            for family, count in DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items():
                for index in range(count):
                    cardinality = (
                        index % 4 + 2
                        if family == "multi_measurement"
                        else (2 if family == "measurement+parameter" else 1)
                    )
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
                                "error_cardinality": cardinality,
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

            reserved_roots = {
                "measurement+parameter": [],
                "multi_measurement": sorted(
                    row["grouping"]["physical_root_fingerprint"]
                    for row in rows
                    if row["grouping"]["scenario_family"]
                    == "multi_measurement"
                ),
                "parameter": [],
            }
            training_manifest = {
                "schema_version": 1,
                "builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
                "development_reserved_roots_by_family": reserved_roots,
                "development_reserved_root_set_sha256_by_family": {
                    family: stable_json_sha256(family_roots)
                    for family, family_roots in reserved_roots.items()
                },
                "withheld_for_development_count_by_family": {
                    family: len(family_roots)
                    for family, family_roots in reserved_roots.items()
                },
            }
            paths["scenarios.manifest.json"].write_text(
                json.dumps(training_manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            def digest(name: str) -> str:
                return hashlib.sha256(paths[name].read_bytes()).hexdigest()

            repo_root = Path(__file__).resolve().parents[2]
            source_bindings = {
                relative: hashlib.sha256(
                    (repo_root / relative).read_bytes()
                ).hexdigest()
                for relative in collect_module.DAGGER1_DEVELOPMENT_SOURCE_BINDINGS
            }
            root_sets = {
                "d0": ["d0-root"],
                "frozen": ["frozen-root"],
                "d1_training": ["training-root"],
                "development": roots,
            }

            manifest = {
                "schema_version": 1,
                "scenario_schema_version": 1,
                "artifact_type": "dagger1_development_holdout_suite",
                "builder_contract": DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
                "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
                "suite_format": "evaluation_suite_mapping_v1",
                "split": DAGGER1_DEVELOPMENT_SPLIT,
                "source_partition": "train",
                "parameter_ranking_dominance_threshold": 1.0,
                "seed": collect_module.DAGGER1_DEVELOPMENT_SEED,
                "source_state": source_state,
                "source_bindings": source_bindings,
                "plan": DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
                "candidate_multiplier": 4,
                "candidate_request_plan": (
                    collect_module.DAGGER1_DEVELOPMENT_CANDIDATE_REQUEST_PLAN
                ),
                "candidate_plan": (
                    collect_module.DAGGER1_DEVELOPMENT_CANDIDATE_REQUEST_PLAN
                ),
                "candidate_count": (
                    collect_module.DAGGER1_DEVELOPMENT_RAW_CANDIDATE_COUNT
                ),
                "filtered_protected_root_count": 118,
                "filtered_multi_measurement_with_parameter_scans_root_count": 0,
                "fresh_candidate_inventory": {
                    family: {
                        "physical_root_count": count,
                        "error_cardinality": (
                            collect_module
                            .DAGGER1_DEVELOPMENT_FRESH_CANDIDATE_CARDINALITY_INVENTORY[
                                family
                            ]
                        ),
                        "physical_root_set_sha256": str(index + 1) * 64,
                    }
                    for index, (family, count) in enumerate(
                        collect_module
                        .DAGGER1_DEVELOPMENT_FRESH_CANDIDATE_COUNT_BY_FAMILY.items()
                    )
                },
                "selected_count_by_family": DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
                "selected_multi_measurement_cardinality_inventory": {
                    "2": 3,
                    "3": 3,
                    "4": 3,
                    "5": 3,
                },
                "training_development_reserved_roots_by_family": (
                    reserved_roots
                ),
                "training_development_reserved_multi_measurement_root_set_sha256": (
                    stable_json_sha256(reserved_roots["multi_measurement"])
                ),
                "selected_multi_measurement_root_set_sha256": (
                    stable_json_sha256(reserved_roots["multi_measurement"])
                ),
                "selected_multi_measurement_matches_training_reservation": True,
                "scenario_count": 30,
                "physical_root_count": 30,
                "root_counts": {
                    name: len(root_values)
                    for name, root_values in root_sets.items()
                },
                "root_set_sha256": {
                    name: stable_json_sha256(root_values)
                    for name, root_values in root_sets.items()
                },
                "pairwise_input_overlap": {
                    "d0_frozen": [],
                    "d0_d1_training": [],
                    "frozen_d1_training": [],
                },
                "training_eligible": False,
                "training_collection_eligible": False,
                "release_evidence_eligible": False,
                "promotion_evidence_eligible": False,
                "diagnostic_closed_loop_model_selection_eligible": True,
                "recovery_stratum_qualified_model_selection_eligible": False,
                "intended_use": (
                    "dagger1_closed_loop_development_model_selection_only"
                ),
                "required_post_evaluation_recovery_strata": list(
                    collect_module.REQUIRED_POST_EVALUATION_RECOVERY_STRATA
                ),
                "recovery_strata_coverage_requires_closed_loop_evaluation": True,
                "recovery_strata_qualification_status": (
                    "pending_teacher_opportunity_trace_instrumentation"
                ),
                "training_development_reserved_boundary_overlap": {
                    "d0": [],
                    "frozen": [],
                    "d1_training": [],
                },
                "development_protected_overlap": {
                    "d0": [],
                    "frozen": [],
                    "d1_training": [],
                },
                "output_sha256": digest("development.json"),
                "generator_report_sha256": digest(
                    "development.generator.json"
                ),
                "d1_training_scenarios_sha256": digest("scenarios.json"),
                "d1_training_manifest_sha256": digest(
                    "scenarios.manifest.json"
                ),
                "d0_raw_sha256": digest("aggregate.raw.jsonl"),
                "d0_generation_provenance_sha256": digest(
                    "aggregate.generation_provenance.json"
                ),
                "d0_manifest_sha256": digest("aggregate.manifest.json"),
                "frozen_suite_sha256": digest("suite.json"),
                "evaluation_policy_sha256": digest("policy.json"),
            }
            paths["development.json.manifest.json"].write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            kwargs = {
                "generator_report_path": paths["development.generator.json"],
                "source_state": source_state,
                "scenario_input_path": paths["scenarios.json"],
                "scenario_manifest_path": paths["scenarios.manifest.json"],
                "d0_raw_path": paths["aggregate.raw.jsonl"],
                "d0_provenance_path": paths[
                    "aggregate.generation_provenance.json"
                ],
                "d0_manifest_path": paths["aggregate.manifest.json"],
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

            for field, mutation, failure in (
                (
                    "selected_multi_measurement_cardinality_inventory",
                    {"2": 2, "3": 3, "4": 3, "5": 3},
                    "selected_multi_measurement_cardinality",
                ),
                (
                    "training_development_reserved_roots_by_family",
                    {
                        **reserved_roots,
                        "multi_measurement": reserved_roots[
                            "multi_measurement"
                        ][1:],
                    },
                    "training_reservation_copy",
                ),
                (
                    "selected_multi_measurement_matches_training_reservation",
                    False,
                    "selected_multi_measurement_reservation",
                ),
            ):
                tampered_field = copy.deepcopy(manifest)
                tampered_field[field] = mutation
                paths["development.json.manifest.json"].write_text(
                    json.dumps(tampered_field, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, failure
                ):
                    validate_development_holdout_binding(
                        paths["development.json"],
                        paths["development.json.manifest.json"],
                        **kwargs,
                    )

            paths["development.json.manifest.json"].write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = self._release_threshold_report()
            report["tampered"] = True
            paths["development.generator.json"].write_text(
                json.dumps(report, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "generator_report_sha256"):
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
                    "aggregate.manifest.json",
                    "suite.json",
                    "policy.json",
                )
            }
            paths["generator.json"].write_text("{}\n", encoding="utf-8")
            paths["aggregate.raw.jsonl"].write_text(
                json.dumps({"physical_root_fingerprint": "d0-root"}) + "\n",
                encoding="utf-8",
            )
            paths["aggregate.generation_provenance.json"].write_text(
                "{}\n", encoding="utf-8"
            )
            paths["aggregate.manifest.json"].write_text(
                "{}\n", encoding="utf-8"
            )
            paths["suite.json"].write_text(
                json.dumps(
                    {
                        "suite": [
                            {
                                "grouping": {
                                    "physical_root_fingerprint": "frozen-root"
                                }
                            }
                        ]
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            paths["policy.json"].write_text("{}\n", encoding="utf-8")

            scenarios = []

            def append_scenarios(
                family: str,
                cohort: str,
                cardinalities: list[int],
                *,
                topup_count: int = 0,
            ) -> None:
                priority = (
                    0
                    if cohort == "primary"
                    else DAGGER1_RESERVE_FAMILY_PRIORITY.index(family) + 1
                )
                for family_index, cardinality in enumerate(cardinalities):
                    order = len(scenarios)
                    is_topup = bool(
                        topup_count
                        and family_index >= len(cardinalities) - topup_count
                    )
                    scenarios.append(
                        {
                            "grouping": {
                                "scenario_family": family,
                                "physical_root_fingerprint": (
                                    f"{cohort}-{family}-{order}"
                                ),
                                "error_cardinality": cardinality,
                                "collection_cohort": cohort,
                                "collection_subcohort": (
                                    collect_module.DAGGER1_TOPUP_SUBCOHORT
                                    if is_topup
                                    else (
                                        "primary"
                                        if cohort == "primary"
                                        else "base_reserve"
                                    )
                                ),
                                "collection_priority": priority,
                                "collection_order": order,
                            }
                        }
                    )

            append_scenarios("measurement+parameter", "primary", [2] * 48)
            append_scenarios(
                "multi_measurement",
                "primary",
                [2] * 16 + [3] * 6 + [4] * 10 + [5] * 16,
            )
            append_scenarios("parameter", "primary", [1] * 24)
            append_scenarios(
                "multi_measurement",
                "reserve",
                [3] * 12 + [4] * 5 + [5] * 14,
            )
            append_scenarios(
                "measurement+parameter",
                "reserve",
                [2] * 60,
                topup_count=12,
            )
            paths["scenarios.json"].write_text(
                json.dumps(scenarios, sort_keys=True) + "\n",
                encoding="utf-8",
            )

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
                "seed": collect_module.DAGGER1_SCENARIO_SEED,
                "plan": collect_module.DAGGER1_PRIMARY_PLAN,
                "primary_plan": collect_module.DAGGER1_PRIMARY_PLAN,
                "primary_count_by_family": collect_module.DAGGER1_PRIMARY_PLAN,
                "reserve_count_by_family": collect_module.DAGGER1_RESERVE_PLAN,
                "base_reserve_plan": collect_module.DAGGER1_BASE_RESERVE_PLAN,
                "topup_reserve_plan": collect_module.DAGGER1_TOPUP_RESERVE_PLAN,
                "selected_count_by_family": (
                    collect_module.DAGGER1_TRAINING_POOL_PLAN
                ),
                "training_pool_plan": collect_module.DAGGER1_TRAINING_POOL_PLAN,
                "candidate_multiplier": 2,
                "candidate_request_plan": (
                    collect_module.DAGGER1_CANDIDATE_REQUEST_PLAN
                ),
                "candidate_plan": collect_module.DAGGER1_CANDIDATE_REQUEST_PLAN,
                "candidate_count": collect_module.DAGGER1_RAW_CANDIDATE_COUNT,
                "fresh_candidate_count": (
                    collect_module.DAGGER1_FRESH_CANDIDATE_COUNT
                ),
                "fresh_candidate_inventory": {
                    family: {
                        "physical_root_count": count,
                        "error_cardinality": (
                            collect_module
                            .DAGGER1_FRESH_CANDIDATE_CARDINALITY_INVENTORY[family]
                        ),
                        "physical_root_set_sha256": str(index + 1) * 64,
                    }
                    for index, (family, count) in enumerate(
                        collect_module
                        .DAGGER1_FRESH_CANDIDATE_COUNT_BY_FAMILY.items()
                    )
                },
                "unused_fresh_candidate_count_by_family": (
                    collect_module
                    .DAGGER1_UNUSED_FRESH_CANDIDATE_COUNT_BY_FAMILY
                ),
                "primary_multi_measurement_cardinality_quota": (
                    collect_module
                    .DAGGER1_PRIMARY_MULTI_MEASUREMENT_CARDINALITY_QUOTA
                ),
                "primary_multi_measurement_cardinality_count": (
                    collect_module
                    .DAGGER1_PRIMARY_MULTI_MEASUREMENT_CARDINALITY_QUOTA
                ),
                "reserve_multi_measurement_cardinality_inventory": (
                    collect_module
                    .DAGGER1_RESERVE_MULTI_MEASUREMENT_CARDINALITY_INVENTORY
                ),
                "collection_schedule": {
                    "contract": DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
                    "cohort_order": ["primary", "reserve"],
                    "reserve_family_priority": list(
                        DAGGER1_RESERVE_FAMILY_PRIORITY
                    ),
                    "priority_field": "grouping.collection_priority",
                    "order_field": "grouping.collection_order",
                    "subcohort_field": "grouping.collection_subcohort",
                    "reserve_subcohort_order": [
                        "base_reserve",
                        collect_module.DAGGER1_TOPUP_SUBCOHORT,
                    ],
                    "maximum_rollout_replicas_by_family": (
                        DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY
                    ),
                },
                "development_reserved_roots_by_family": {
                    "measurement+parameter": [],
                    "multi_measurement": [
                        f"development-multi-{index:02d}" for index in range(12)
                    ],
                    "parameter": [],
                },
                "withheld_for_development_count_by_family": (
                    collect_module
                    .DAGGER1_DEVELOPMENT_RESERVED_COUNT_BY_FAMILY
                ),
                "withheld_for_development_multi_measurement_cardinality_inventory": (
                    collect_module
                    .DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_INVENTORY
                ),
                "scenario_count": len(scenarios),
                "physical_root_count": len(scenarios),
                "filtered_protected_root_count": 37,
                "filtered_multi_measurement_with_parameter_scans_root_count": 0,
                "d0_root_count": 1,
                "frozen_root_count": 1,
                "protected_root_overlap": [],
                "output_sha256": digest(paths["scenarios.json"]),
                "generator_report_sha256": digest(paths["generator.json"]),
                "d0_raw_sha256": digest(paths["aggregate.raw.jsonl"]),
                "d0_generation_provenance_sha256": digest(
                    paths["aggregate.generation_provenance.json"]
                ),
                "d0_manifest_sha256": digest(paths["aggregate.manifest.json"]),
                "frozen_suite_sha256": digest(paths["suite.json"]),
                "evaluation_policy_sha256": digest(paths["policy.json"]),
            }
            reserved = manifest["development_reserved_roots_by_family"]
            manifest["development_reserved_root_set_sha256_by_family"] = {
                family: stable_json_sha256(family_roots)
                for family, family_roots in reserved.items()
            }
            primary_roots = sorted(
                row["grouping"]["physical_root_fingerprint"]
                for row in scenarios
                if row["grouping"]["collection_cohort"] == "primary"
            )
            reserve_roots = sorted(
                row["grouping"]["physical_root_fingerprint"]
                for row in scenarios
                if row["grouping"]["collection_cohort"] == "reserve"
            )
            manifest["primary_physical_root_set_sha256"] = stable_json_sha256(
                primary_roots
            )
            manifest["reserve_physical_root_set_sha256"] = stable_json_sha256(
                reserve_roots
            )
            manifest["training_physical_root_set_sha256"] = stable_json_sha256(
                sorted(primary_roots + reserve_roots)
            )
            topup_roots = sorted(
                row["grouping"]["physical_root_fingerprint"]
                for row in scenarios
                if row["grouping"]["collection_subcohort"]
                == collect_module.DAGGER1_TOPUP_SUBCOHORT
            )
            base_training_roots = sorted(
                set(primary_roots + reserve_roots) - set(topup_roots)
            )
            topup_roots_by_family = {
                "measurement+parameter": topup_roots,
                "multi_measurement": [],
                "parameter": [],
            }
            manifest.update(
                {
                    "topup_reserve_count_by_family": (
                        collect_module.DAGGER1_TOPUP_RESERVE_PLAN
                    ),
                    "topup_reserve_roots_by_family": topup_roots_by_family,
                    "topup_reserve_root_set_sha256_by_family": {
                        family: stable_json_sha256(roots)
                        for family, roots in topup_roots_by_family.items()
                    },
                    "topup_reserve_physical_root_set_sha256": (
                        stable_json_sha256(topup_roots)
                    ),
                    "predecessor_source_commit": (
                        collect_module.DAGGER1_PREDECESSOR_SOURCE_COMMIT
                    ),
                    "predecessor_training_root_count": len(base_training_roots),
                    "predecessor_training_root_set_sha256": (
                        stable_json_sha256(base_training_roots)
                    ),
                    "topup_predecessor_overlap": [],
                    "topup_development_reserved_overlap": [],
                }
            )
            predecessor_patch = patch.object(
                collect_module,
                "DAGGER1_PREDECESSOR_TRAINING_ROOT_SET_SHA256",
                stable_json_sha256(base_training_roots),
            )
            predecessor_patch.start()
            self.addCleanup(predecessor_patch.stop)
            kwargs = {
                "scenarios": scenarios,
                "input_path": paths["scenarios.json"],
                "generator_report_path": paths["generator.json"],
                "source_state": source_state,
                "d0_raw_path": paths["aggregate.raw.jsonl"],
                "d0_provenance_path": paths[
                    "aggregate.generation_provenance.json"
                ],
                "d0_manifest_path": paths["aggregate.manifest.json"],
                "forbidden_suite_path": paths["suite.json"],
                "evaluation_policy_path": paths["policy.json"],
            }
            validate_scenario_builder_manifest(manifest, **kwargs)
            production_batches = dagger1_rollout_batches(
                scenarios, collection_pass="training"
            )
            self.assertEqual(
                sum(len(batch["scenarios"]) for batch in production_batches),
                477,
            )
            self.assertEqual(
                production_batches[-1]["batch_id"],
                "repeat-multi_measurement-r2",
            )

            for field, mutation, failure in (
                (
                    "candidate_request_plan",
                    {
                        **collect_module.DAGGER1_CANDIDATE_REQUEST_PLAN,
                        "multi_measurement": 175,
                    },
                    "candidate_request_plan",
                ),
                (
                    "primary_multi_measurement_cardinality_quota",
                    {"2": 15, "3": 6, "4": 10, "5": 16},
                    "primary_multi_measurement_quota",
                ),
                (
                    "reserve_multi_measurement_cardinality_inventory",
                    {"3": 11, "4": 5, "5": 14},
                    "reserve_multi_measurement_inventory",
                ),
                (
                    "topup_reserve_count_by_family",
                    {
                        "measurement+parameter": 11,
                        "multi_measurement": 0,
                        "parameter": 0,
                    },
                    "topup_reserve_binding",
                ),
            ):
                tampered = copy.deepcopy(manifest)
                tampered[field] = mutation
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, failure
                ):
                    validate_scenario_builder_manifest(tampered, **kwargs)

            tampered_reservation = copy.deepcopy(manifest)
            tampered_reservation["development_reserved_roots_by_family"][
                "multi_measurement"
            ][0] = "tampered-root"
            with self.assertRaisesRegex(ValueError, "development_reservation"):
                validate_scenario_builder_manifest(
                    tampered_reservation, **kwargs
                )

            tampered_schedule = copy.deepcopy(manifest)
            tampered_schedule["collection_schedule"]["cohort_order"] = [
                "reserve",
                "primary",
            ]
            with self.assertRaisesRegex(
                ValueError, "collection_schedule_cohorts"
            ):
                validate_scenario_builder_manifest(tampered_schedule, **kwargs)

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

    def test_failed_collection_directory_is_separate_and_write_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protected = root / "input" / "scenarios.json"
            protected.parent.mkdir()
            protected.write_text("[]\n", encoding="utf-8")
            output = root / "output" / "rows.jsonl"
            with self.assertRaisesRegex(ValueError, "must be separate"):
                validate_collection_output_paths(
                    output=output,
                    all_output=root / "output" / "all.jsonl",
                    failed_collection_dir=protected.parent,
                    protected_paths=(protected,),
                )

            failed_dir = root / "failed-attempt-1"
            failed_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                validate_collection_output_paths(
                    output=output,
                    all_output=root / "output" / "all.jsonl",
                    failed_collection_dir=failed_dir,
                    protected_paths=(protected,),
                )

    def test_failed_strict_collection_bundle_is_atomic_and_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            production_output = root / "training.jsonl"
            production_all = root / "training.all.jsonl"
            production_manifest = root / "training.jsonl.manifest.json"
            failed_dir = root / "attempt-1.failed-collection"
            candidate = {
                "example_id": "candidate-1",
                "production_label_eligible": True,
                "labels": {"production_label_eligible": True},
            }
            quarantined = {
                "example_id": "quarantined-1",
                "production_label_eligible": False,
                "labels": {"production_label_eligible": False},
            }
            evidence = {
                "failed_gate_names": [
                    "offline_teacher_target_quarantine_summary",
                    "recommended_collection_gate",
                ],
                "intended_production_outputs": {
                    "output": str(production_output),
                    "all_output": str(production_all),
                    "manifest": str(production_manifest),
                },
                "source_state": {"source_commit": "a" * 40},
                "model_revision": "b" * 64,
            }

            manifest = write_failed_collection_evidence_bundle(
                failed_dir,
                candidate_rows=[candidate],
                all_rows=[candidate, quarantined],
                evidence=evidence,
            )

            self.assertTrue(failed_dir.is_dir())
            self.assertEqual(
                {path.name for path in failed_dir.iterdir()},
                {
                    FAILED_COLLECTION_CANDIDATE_ROWS,
                    FAILED_COLLECTION_ALL_ROWS,
                    FAILED_COLLECTION_EVIDENCE,
                    FAILED_COLLECTION_CHECKSUMS,
                },
            )
            self.assertFalse(production_output.exists())
            self.assertFalse(production_all.exists())
            self.assertFalse(production_manifest.exists())
            self.assertEqual(
                manifest["artifact_type"], FAILED_COLLECTION_ARTIFACT_TYPE
            )
            for field in (
                "training_eligible",
                "release_evidence_eligible",
                "round1_aggregate_eligible",
                "production_outputs_published",
                "strict_gate_passed",
            ):
                self.assertIs(manifest[field], False)
            self.assertNotIn("output_sha256", manifest)
            self.assertNotIn("all_output", manifest)

            disk_manifest = json.loads(
                (failed_dir / FAILED_COLLECTION_EVIDENCE).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(disk_manifest, manifest)
            for key, filename in (
                ("candidate_recovery_rows", FAILED_COLLECTION_CANDIDATE_ROWS),
                ("all_visited_rows", FAILED_COLLECTION_ALL_ROWS),
            ):
                descriptor = manifest["diagnostic_artifacts"][key]
                path = failed_dir / filename
                self.assertEqual(descriptor["relative_path"], filename)
                self.assertEqual(
                    descriptor["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )

            for filename in (
                FAILED_COLLECTION_CANDIDATE_ROWS,
                FAILED_COLLECTION_ALL_ROWS,
            ):
                rows = [
                    json.loads(line)
                    for line in (failed_dir / filename).read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertTrue(rows)
                for row in rows:
                    self.assertIs(row["collection_training_eligible"], False)
                    self.assertEqual(
                        row["collection_disposition"],
                        "failed_strict_gate_diagnostic_only",
                    )
                    self.assertIs(
                        row["labels"]["collection_training_eligible"], False
                    )

            checksum_lines = (
                failed_dir / FAILED_COLLECTION_CHECKSUMS
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(checksum_lines), 3)
            for line in checksum_lines:
                expected, filename = line.split("  ", 1)
                self.assertEqual(
                    expected,
                    hashlib.sha256((failed_dir / filename).read_bytes()).hexdigest(),
                )
            self.assertNotIn("collection_training_eligible", candidate)

    def test_analysis_bundle_carries_an_unmistakable_top_level_identity(self):
        """A complete-schedule analysis bundle must not read as a strict NO-GO.

        Both bundles are training-ineligible, but they answer different
        questions.  The round-2 analysis bundle carried the strict failure
        artifact type and ``collection_outcome=strict_gate_failed``, so once its
        stdout was gone the only surviving marker was a nested field.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            failed_dir = Path(temp_dir) / "analysis.failed-collection"
            row = {
                "example_id": "candidate-1",
                "production_label_eligible": True,
                "labels": {"production_label_eligible": True},
            }
            manifest = write_failed_collection_evidence_bundle(
                failed_dir,
                candidate_rows=[row],
                all_rows=[row],
                evidence={
                    "failed_gate_names": ["independent_root_support"],
                    "collection_stopping_report": {
                        "stopping_reason": (
                            "analysis_only_complete_schedule_exhausted"
                        ),
                        "analysis_only": True,
                        "executed_episode_count": 477,
                        "planned_episode_count": 477,
                    },
                },
            )

            self.assertEqual(
                manifest["artifact_type"], ANALYSIS_COMPLETE_ARTIFACT_TYPE
            )
            self.assertEqual(
                manifest["collection_outcome"],
                "analysis_only_complete_schedule_exhausted",
            )
            self.assertIs(manifest["analysis_only"], True)
            self.assertIs(manifest["strict_gate_evaluated"], True)
            # Publication safety is identical to a strict NO-GO.
            for field in (
                "training_eligible",
                "round1_aggregate_eligible",
                "production_outputs_published",
                "strict_gate_passed",
            ):
                self.assertIs(manifest[field], False)

    def test_failed_collection_bundle_cleans_partial_staging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "failed-collection"
            evidence = {"failed_gate_names": ["recommended_collection_gate"]}
            with patch(
                "psse_env.dagger.collect_dagger1._write_fsynced_text",
                side_effect=OSError("simulated late write failure"),
            ), self.assertRaisesRegex(OSError, "simulated late write failure"):
                write_failed_collection_evidence_bundle(
                    destination,
                    candidate_rows=[{"example_id": "candidate"}],
                    all_rows=[{"example_id": "candidate"}],
                    evidence=evidence,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_truth_leak_never_publishes_failed_collection_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "failed-collection"
            with self.assertRaisesRegex(RuntimeError, "private oracle truth"):
                write_failed_collection_evidence_bundle(
                    destination,
                    candidate_rows=[
                        {
                            "example_id": "leaked",
                            "labels": {"true_parameter_errors": []},
                        }
                    ],
                    all_rows=[],
                    evidence={
                        "failed_gate_names": ["recommended_collection_gate"]
                    },
                )
            self.assertFalse(destination.exists())

    def test_failed_strict_collection_gate_names_are_structured(self):
        report = failed_strict_collection_gate_names(
            collection_gate={"passed": False},
            targeted_coverage={"passed": True},
            independent_root_support={"passed": False},
            truth_audit_quarantine={"passed": True},
        )
        self.assertEqual(
            report,
            ["independent_root_support", "recommended_collection_gate"],
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

    def test_rollout_schedule_is_finite_and_diagnostic_uses_primary_once(self):
        scenarios = self._scheduled_scenarios()
        diagnostic = dagger1_rollout_batches(
            scenarios, collection_pass="diagnostic"
        )
        self.assertEqual([batch["batch_id"] for batch in diagnostic], ["primary-r0"])
        self.assertEqual(len(diagnostic[0]["scenarios"]), 2)

        training = dagger1_rollout_batches(
            list(reversed(scenarios)), collection_pass="training"
        )
        self.assertEqual(
            [batch["batch_id"] for batch in training],
            [
                "primary-r0",
                "reserve-1-multi_measurement-r0",
                "reserve-2-measurement+parameter-r0",
                "reserve-3-parameter-r0",
                "repeat-multi_measurement-r1",
                "repeat-measurement+parameter-r1",
                "repeat-multi_measurement-r2",
            ],
        )
        self.assertEqual(
            sum(len(batch["scenarios"]) for batch in training), 11
        )

    def test_rollout_seed_and_repeat_ids_are_batch_invariant_and_unique(self):
        scenarios = self._scheduled_scenarios()
        root = scenarios[0]["grouping"]["physical_root_fingerprint"]
        self.assertEqual(
            dagger1_rollout_seed(
                seed=17,
                physical_root_fingerprint=root,
                replica=1,
            ),
            dagger1_rollout_seed(
                seed=17,
                physical_root_fingerprint=root,
                replica=1,
            ),
        )
        self.assertNotEqual(
            dagger1_rollout_seed(
                seed=17,
                physical_root_fingerprint=root,
                replica=0,
            ),
            dagger1_rollout_seed(
                seed=17,
                physical_root_fingerprint=root,
                replica=1,
            ),
        )

        def collect_episode(scenario, replica, rollout_seed, batch_id, order):
            del replica, rollout_seed, batch_id, order
            grouping = scenario["grouping"]
            scenario_id = scenario["execution"]["scenario_id"]
            return [
                {
                    "example_id": f"dagger_iter1_{scenario_id}_step0",
                    "scenario_id": scenario_id,
                    "physical_root_fingerprint": grouping[
                        "physical_root_fingerprint"
                    ],
                    "scenario_family": grouping["scenario_family"],
                    "state_origin": "learner_policy",
                    "step": 0,
                    "terminal_outcome": "resolved",
                }
            ]

        rows, matrix, report, _ = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="training",
            seed=17,
            max_steps=24,
            collect_episode=collect_episode,
            checkpoint=lambda rows, matrix: {
                "candidate_rows": len(rows),
                "selected_rows": [],
                "failed_gate_names": ["not_yet"],
                "passed": False,
            },
        )
        example_ids = [row["example_id"] for row in rows]
        self.assertEqual(len(example_ids), len(set(example_ids)))
        self.assertTrue(
            all(row["state_origin"] == "learner_policy" for row in rows)
        )
        self.assertEqual(report["stopping_reason"], "reserve_exhausted")
        self.assertEqual(report["executed_episode_count"], 11)
        self.assertTrue(matrix["passed"])

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "reserve-exhausted"
            manifest = write_failed_collection_evidence_bundle(
                destination,
                candidate_rows=rows,
                all_rows=rows,
                evidence={
                    "failed_gate_names": ["recommended_collection_gate"],
                    "collection_stopping_report": report,
                    "rollout_disposition_matrix": matrix,
                },
            )
            self.assertEqual(
                manifest["collection_stopping_report"]["stopping_reason"],
                "reserve_exhausted",
            )
            self.assertEqual(
                manifest["diagnostic_artifacts"]["all_visited_rows"][
                    "row_count"
                ],
                11,
            )
            self.assertFalse(manifest["training_eligible"])

    def test_schedule_stops_only_after_first_passing_whole_batch(self):
        scenarios = self._scheduled_scenarios()

        def collect_episode(scenario, replica, rollout_seed, batch_id, order):
            del replica, rollout_seed, batch_id, order
            grouping = scenario["grouping"]
            scenario_id = scenario["execution"]["scenario_id"]
            return [
                {
                    "example_id": f"{scenario_id}-step0",
                    "scenario_id": scenario_id,
                    "physical_root_fingerprint": grouping[
                        "physical_root_fingerprint"
                    ],
                    "scenario_family": grouping["scenario_family"],
                    "step": 0,
                    "terminal_outcome": "resolved",
                }
            ]

        rows, _, report, checkpoint = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="training",
            seed=19,
            max_steps=24,
            collect_episode=collect_episode,
            checkpoint=lambda rows, matrix: {
                "candidate_rows": len(rows),
                "selected_rows": list(rows),
                "failed_gate_names": [] if len(rows) >= 3 else ["row_floor"],
                "passed": len(rows) >= 3,
            },
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(checkpoint["passed"])
        self.assertEqual(
            report["stopped_after_batch"],
            "reserve-1-multi_measurement-r0",
        )
        self.assertEqual(
            report["stopping_reason"], "strict_collection_gate_passed"
        )
        self.assertTrue(report["unexecuted_batch_ids"])

    def test_schedule_stops_at_first_irreversible_truth_quarantine_batch(self):
        scenarios = self._scheduled_scenarios()

        def collect_episode(scenario, replica, rollout_seed, batch_id, order):
            del replica, rollout_seed, batch_id, order
            grouping = scenario["grouping"]
            scenario_id = scenario["execution"]["scenario_id"]
            return [
                {
                    "example_id": f"{scenario_id}-step0",
                    "scenario_id": scenario_id,
                    "physical_root_fingerprint": grouping[
                        "physical_root_fingerprint"
                    ],
                    "scenario_family": grouping["scenario_family"],
                    "step": 0,
                    "terminal_outcome": "resolved",
                }
            ]

        rows, _, report, checkpoint = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="training",
            seed=19,
            max_steps=24,
            collect_episode=collect_episode,
            checkpoint=lambda rows, matrix: {
                "candidate_rows": len(rows),
                "selected_rows": [],
                "failed_gate_names": [
                    "offline_teacher_target_quarantine_summary"
                ],
                "offline_teacher_target_quarantine_summary": {
                    "passed": False,
                    "quarantined_rows": 1,
                },
                "passed": False,
            },
        )

        self.assertEqual(len(rows), 2)
        self.assertFalse(checkpoint["passed"])
        self.assertEqual(report["executed_batch_ids"], ["primary-r0"])
        self.assertEqual(report["stopped_after_batch"], "primary-r0")
        self.assertEqual(
            report["stopping_reason"],
            "irreversible_truth_audit_quarantine",
        )
        self.assertEqual(
            report["terminal_failure"],
            {
                "gate": "offline_teacher_target_quarantine_summary",
                "reason": (
                    "strict_zero_quarantine_gate_is_cumulative_and_"
                    "irreversible"
                ),
                "quarantined_rows": 1,
            },
        )
        self.assertEqual(report["executed_episode_count"], 2)
        self.assertTrue(report["unexecuted_batch_ids"])
        self.assertFalse(report["passed"])
        # Production payload must stay byte-identical: analysis-only bookkeeping
        # never leaks into a strict run.
        self.assertFalse(report["analysis_only"])
        self.assertTrue(report["training_eligible"])

    def test_analysis_only_mode_runs_complete_schedule_past_quarantine(self):
        """A censored run cannot attribute coverage shortfalls to the schedule.

        The DAgger-1 round-2 collection stopped after 2 of 6 batches at 151 of
        477 episodes, so its root-support and replay-capacity shortfalls were
        measured under censorship.  This mode executes every predeclared batch
        so the complete schedule's contribution is observable, while remaining
        incapable of publishing production data.
        """
        scenarios = self._scheduled_scenarios()

        def collect_episode(scenario, replica, rollout_seed, batch_id, order):
            del replica, rollout_seed, batch_id, order
            grouping = scenario["grouping"]
            scenario_id = scenario["execution"]["scenario_id"]
            return [
                {
                    "example_id": f"{scenario_id}-step0",
                    "scenario_id": scenario_id,
                    "physical_root_fingerprint": grouping[
                        "physical_root_fingerprint"
                    ],
                    "scenario_family": grouping["scenario_family"],
                    "step": 0,
                    "terminal_outcome": "resolved",
                }
            ]

        quarantining_checkpoint = lambda rows, matrix: {  # noqa: E731
            "candidate_rows": len(rows),
            "selected_rows": [],
            "failed_gate_names": ["offline_teacher_target_quarantine_summary"],
            "offline_teacher_target_quarantine_summary": {
                "passed": False,
                "quarantined_rows": 1,
            },
            "passed": False,
        }

        strict_rows, _, strict_report, _ = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="training",
            seed=19,
            max_steps=24,
            collect_episode=collect_episode,
            checkpoint=quarantining_checkpoint,
        )
        rows, _, report, _ = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="training",
            seed=19,
            max_steps=24,
            collect_episode=collect_episode,
            checkpoint=quarantining_checkpoint,
            analysis_only_complete_schedule=True,
        )

        # The strict run stops; the analysis run exhausts the schedule.
        self.assertEqual(strict_report["executed_batch_ids"], ["primary-r0"])
        self.assertEqual(
            report["executed_batch_ids"], report["planned_batch_ids"]
        )
        self.assertEqual(report["unexecuted_batch_ids"], [])
        self.assertEqual(
            report["executed_episode_count"], report["planned_episode_count"]
        )
        self.assertGreater(len(rows), len(strict_rows))

        # The quarantine is still recorded and the run is still a failure.
        self.assertIsNotNone(report["terminal_failure"])
        self.assertEqual(
            report["terminal_failure"]["gate"],
            "offline_teacher_target_quarantine_summary",
        )
        self.assertEqual(
            report["terminal_failure"]["first_quarantined_batch"], "primary-r0"
        )
        self.assertFalse(report["passed"])

        # And it can never be mistaken for production data.
        self.assertTrue(report["analysis_only"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(
            report["stopping_reason"],
            "analysis_only_complete_schedule_exhausted",
        )
        self.assertIsNone(report["stopped_after_batch"])

    def test_analysis_only_mode_does_not_stop_on_a_passing_checkpoint(self):
        scenarios = self._scheduled_scenarios()

        def collect_episode(scenario, replica, rollout_seed, batch_id, order):
            del replica, rollout_seed, batch_id, order
            grouping = scenario["grouping"]
            scenario_id = scenario["execution"]["scenario_id"]
            return [
                {
                    "example_id": f"{scenario_id}-step0",
                    "scenario_id": scenario_id,
                    "physical_root_fingerprint": grouping[
                        "physical_root_fingerprint"
                    ],
                    "scenario_family": grouping["scenario_family"],
                    "step": 0,
                    "terminal_outcome": "resolved",
                }
            ]

        _, _, report, _ = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="training",
            seed=19,
            max_steps=24,
            collect_episode=collect_episode,
            checkpoint=lambda rows, matrix: {
                "candidate_rows": len(rows),
                "selected_rows": [],
                "failed_gate_names": [],
                "offline_teacher_target_quarantine_summary": {
                    "passed": True,
                    "quarantined_rows": 0,
                },
                "passed": True,
            },
            analysis_only_complete_schedule=True,
        )

        self.assertEqual(report["unexecuted_batch_ids"], [])
        self.assertIsNone(report["terminal_failure"])
        # A passing checkpoint must not promote an analysis run to production.
        self.assertFalse(report["passed"])
        self.assertTrue(report["analysis_only"])
        self.assertFalse(report["training_eligible"])

    def test_horizon_truncation_is_an_explicit_valid_rollout_disposition(self):
        scenarios = self._scheduled_scenarios()

        def collect_episode(scenario, replica, rollout_seed, batch_id, order):
            del replica, rollout_seed, batch_id, order
            grouping = scenario["grouping"]
            scenario_id = scenario["execution"]["scenario_id"]
            return [
                {
                    "example_id": f"{scenario_id}-step{step}",
                    "scenario_id": scenario_id,
                    "physical_root_fingerprint": grouping[
                        "physical_root_fingerprint"
                    ],
                    "scenario_family": grouping["scenario_family"],
                    "step": step,
                    "terminal_outcome": None,
                }
                for step in range(24)
            ]

        _, matrix, report, _ = collect_dagger1_rollout_schedule(
            scenarios,
            collection_pass="diagnostic",
            seed=23,
            max_steps=24,
            collect_episode=collect_episode,
        )
        self.assertTrue(matrix["passed"])
        self.assertEqual(matrix["environment_terminal_episodes"], 0)
        self.assertEqual(matrix["horizon_truncated_episodes"], 2)
        self.assertEqual(
            report["stopping_reason"], "diagnostic_primary_complete"
        )

    def test_deterministic_selection_caps_rows_and_preserves_root_floors(self):
        rows = self._strict_coverage_rows(extra_rows=70)
        selected, report = select_dagger1_collection_rows(
            rows,
            target_min_rows=90,
            target_max_rows=100,
        )
        selected_reversed, reversed_report = select_dagger1_collection_rows(
            list(reversed(rows)),
            target_min_rows=90,
            target_max_rows=100,
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(selected), 100)
        self.assertEqual(
            [row["example_id"] for row in selected],
            [row["example_id"] for row in selected_reversed],
        )
        self.assertEqual(report, reversed_report)
        self.assertTrue(
            report["selected_independent_root_support"]["passed"]
        )

    def test_selection_preserves_attainable_roots_behind_infeasible_group(self):
        """An infeasible group must not abandon reservation for the rest.

        Regression for the DAgger-1 round-2 selector defect.  The reservation
        loop tested membership against the *required* floor, so a group whose
        candidate pool was smaller than its floor stayed permanently unmet, was
        focused first as the most constrained, exhausted its roots, and then
        broke out of the loop entirely.  Groups ordered behind it kept only the
        roots the general fill stage happened to pick up: in the real
        collection three of six ``post_failure_no_candidate`` roots survived.
        """
        rows = []

        def add(prefix, count, stratum):
            for index in range(count):
                rows.append(
                    {
                        "example_id": f"{prefix}-{index}",
                        "physical_root_fingerprint": f"{prefix}-root-{index}",
                        "production_label_eligible": True,
                        "recovery_stratum": stratum,
                        "scenario_family": "measurement",
                        "error_cardinality": 1,
                    }
                )

        # Group A is intrinsically infeasible (3 candidate roots, floor 10) and
        # sorts first as the most constrained.  Group B is also short of its
        # floor (6 of 10) but must still retain every root it can supply.
        add("ucr", 3, "unsupported_correction_recovery")
        add("pfnc", 6, "post_failure_no_candidate")
        add("filler", 40, "loop_escape")

        selected, report = select_dagger1_collection_rows(
            rows, target_min_rows=10, target_max_rows=12
        )

        ucr = "recovery_stratum:unsupported_correction_recovery"
        pfnc = "recovery_stratum:post_failure_no_candidate"

        self.assertEqual(report["candidate_distinct_roots_by_group"][ucr], 3)
        self.assertEqual(report["candidate_distinct_roots_by_group"][pfnc], 6)
        self.assertEqual(report["attainable_root_targets"][ucr], 3)
        self.assertEqual(report["attainable_root_targets"][pfnc], 6)
        self.assertEqual(report["selected_distinct_roots_by_group"][ucr], 3)
        self.assertEqual(report["selected_distinct_roots_by_group"][pfnc], 6)
        self.assertEqual(report["selected_attainable_root_loss"], {})

        # Preserving attainable support must not soften the release gate: both
        # candidate shortfalls are still reported and the selection still fails.
        self.assertEqual(
            report["candidate_root_group_shortfalls"][ucr]["root_shortfall"], 7
        )
        self.assertEqual(
            report["candidate_root_group_shortfalls"][pfnc]["root_shortfall"], 4
        )
        self.assertFalse(report["passed"])
        self.assertLessEqual(len(selected), 12)

    def test_strict_checkpoint_includes_round1_replay_capacity(self):
        rows = self._strict_coverage_rows(extra_rows=70)

        def d0_rows(count):
            return [
                {
                    "example_id": f"d0-{index}",
                    "physical_root_fingerprint": f"d0-root-{index}",
                    "production_label_eligible": True,
                }
                for index in range(count)
            ]

        passing = evaluate_dagger1_collection_checkpoint(
            rows,
            d0_training_rows=d0_rows(100),
            target_min_rows=90,
            target_max_rows=100,
            rollout_matrix={"passed": True},
        )
        self.assertTrue(passing["passed"], passing["failed_gate_names"])
        self.assertTrue(passing["round1_replay_capacity"]["passed"])

        capacity_failure = evaluate_dagger1_collection_checkpoint(
            rows,
            d0_training_rows=d0_rows(5000),
            target_min_rows=90,
            target_max_rows=100,
            rollout_matrix={"passed": True},
        )
        self.assertFalse(capacity_failure["passed"])
        self.assertIn(
            "round1_replay_capacity",
            capacity_failure["failed_gate_names"],
        )

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
