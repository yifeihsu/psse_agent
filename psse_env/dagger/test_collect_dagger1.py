from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.collect_dagger1 import (
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_FORBIDDEN_SUITE,
    frozen_physical_roots,
    recommended_collection_gate,
    targeted_state_coverage,
    validate_collection_pass,
    validate_collection_output_paths,
    validate_export_rows_truth_free,
    validate_scenario_builder_manifest,
    validate_training_source_report,
    validate_training_scenarios,
)
from psse_env.dagger.rollout_collector import RECOMMENDED_DAGGER1_RECOVERY_STRATA


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
        report = targeted_state_coverage(rows)
        self.assertTrue(report["passed"], report)
        with_scans = [dict(row) for row in rows]
        with_scans[0]["parameter_scans_available"] = True
        self.assertFalse(targeted_state_coverage(with_scans)["passed"])
        rows.pop()
        self.assertFalse(targeted_state_coverage(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
