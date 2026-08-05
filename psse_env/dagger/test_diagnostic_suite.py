from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.diagnostic_suite import (
    audit_failure_diagnostic_evaluation,
    build_failure_diagnostic_suite,
    write_failure_diagnostic_suite,
)
from psse_env.dagger.evaluate_diagnostic import _derive_required_suites


def _scenario(scenario_id: str, family: str) -> dict:
    return {
        "scenario_schema_version": 1,
        "execution": {"scenario_id": scenario_id},
        "audit": {},
        "grouping": {"scenario_family": family},
    }


class DiagnosticSuiteTests(unittest.TestCase):
    def test_diagnostic_runner_requires_every_suite_in_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite_path = Path(temporary) / "diagnostic.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "invalid_action_recovery": [],
                        "standard_success": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _derive_required_suites(["--input", str(suite_path)]),
                [
                    "--required-suite",
                    "invalid_action_recovery",
                    "--required-suite",
                    "standard_success",
                ],
            )
            self.assertEqual(
                _derive_required_suites(
                    [
                        "--input",
                        str(suite_path),
                        "--required-suite",
                        "standard_success",
                    ]
                ),
                [],
            )

    def test_diagnostic_evaluation_audits_exact_roots_and_guard_targets(self) -> None:
        suites = {"invalid_action_recovery": [_scenario("bad", "measurement+parameter")]}
        base_episode = {
            "suite": "invalid_action_recovery",
            "scenario_id": "bad",
            "terminal": True,
            "trace": [
                {
                    "action": {
                        "tool": "correct_parameters",
                        "arguments": {"state_id": "active", "line_index1": 3},
                    },
                    "error_code": "correction_route_not_actionable",
                }
            ],
        }
        artifact = {
            "artifact_type": "closed_loop_diagnostic_evaluation",
            "diagnostic_only": True,
            "release_evidence_eligible": False,
            "training_eligible": False,
            "release_eligible": False,
            "evaluation": {"suite_metrics": {"episodes": [base_episode]}}
        }

        report = audit_failure_diagnostic_evaluation(artifact, suites)

        self.assertTrue(report["passed"])
        self.assertFalse(report["release_evidence_eligible"])
        self.assertEqual(
            report["hard_targets"]["parameter_scans_missing"]["observed"], 0
        )
        self.assertEqual(
            report["recovery_observations"][
                "correction_route_not_actionable"
            ],
            1,
        )

        artifact["evaluation"]["suite_metrics"]["episodes"][0]["trace"].append(
            {
                "action": {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "active", "line_index1": 3},
                },
                "error_code": "parameter_scans_missing",
            }
        )
        failed = audit_failure_diagnostic_evaluation(artifact, suites)
        self.assertFalse(failed["passed"])
        self.assertIn(
            "parameter_scans_missing reached the numerical executor",
            failed["failures"],
        )

        release_typed = dict(artifact)
        release_typed["artifact_type"] = "closed_loop_release_evaluation"
        identity_failure = audit_failure_diagnostic_evaluation(
            release_typed, suites
        )
        self.assertFalse(identity_failure["passed"])
        self.assertIn(
            "evaluation artifact is not irreversibly diagnostic-only",
            identity_failure["failures"],
        )

    def test_selects_only_failed_episode_scenarios(self) -> None:
        suites = {
            "standard_success": [
                _scenario("healthy", "measurement"),
                _scenario("bad", "multi_measurement"),
            ],
            "invalid_action_recovery": [
                _scenario("bad-mixed", "measurement+parameter")
            ],
        }
        artifact = {
            "source_commit": "a" * 40,
            "evaluation": {
                "suite_metrics": {
                    "episodes": [
                        {
                            "suite": "standard_success",
                            "scenario_id": "healthy",
                            "family": "measurement",
                            "terminal": True,
                            "invalid_action_count": 0,
                            "loop_detected": False,
                        },
                        {
                            "suite": "standard_success",
                            "scenario_id": "bad",
                            "family": "multi_measurement",
                            "terminal": False,
                            "invalid_action_count": 4,
                            "false_commit_count": 1,
                            "loop_detected": True,
                        },
                        {
                            "suite": "invalid_action_recovery",
                            "scenario_id": "bad-mixed",
                            "family": "measurement+parameter",
                            "terminal": True,
                            "invalid_action_count": 1,
                            "loop_detected": False,
                        },
                    ]
                }
            },
        }

        selected, report = build_failure_diagnostic_suite(artifact, suites)

        self.assertEqual(
            {
                (suite, row["execution"]["scenario_id"])
                for suite, rows in selected.items()
                for row in rows
            },
            {
                ("standard_success", "bad"),
                ("invalid_action_recovery", "bad-mixed"),
            },
        )
        self.assertEqual(report["selected_scenarios"], 2)
        self.assertFalse(report["release_evidence_eligible"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(
            report["scenario_families"],
            {"measurement+parameter": 1, "multi_measurement": 1},
        )

    def test_writer_refuses_to_overwrite_frozen_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "suite.json"
            artifact = root / "artifact.json"
            frozen.write_text(
                json.dumps({"standard_success": [_scenario("bad", "measurement")]}),
                encoding="utf-8",
            )
            artifact.write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "suite_metrics": {
                                "episodes": [
                                    {
                                        "suite": "standard_success",
                                        "scenario_id": "bad",
                                        "terminal": False,
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                write_failure_diagnostic_suite(
                    artifact_path=artifact,
                    frozen_suite_path=frozen,
                    output_path=frozen,
                )

    def test_writer_checks_reviewed_count_and_nested_source_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "suite.json"
            artifact = root / "artifact.json"
            output = root / "diagnostic.json"
            frozen.write_text(
                json.dumps({"standard_success": [_scenario("bad", "measurement")]}),
                encoding="utf-8",
            )
            artifact.write_text(
                json.dumps(
                    {
                        "provenance": {
                            "source_state": {"source_commit": "b" * 40}
                        },
                        "evaluation": {
                            "suite_metrics": {
                                "episodes": [
                                    {
                                        "suite": "standard_success",
                                        "scenario_id": "bad",
                                        "terminal": False,
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reviewed failure set"):
                write_failure_diagnostic_suite(
                    artifact_path=artifact,
                    frozen_suite_path=frozen,
                    output_path=output,
                    expected_scenarios=13,
                )
            self.assertFalse(output.exists())

            report = write_failure_diagnostic_suite(
                artifact_path=artifact,
                frozen_suite_path=frozen,
                output_path=output,
                expected_scenarios=1,
            )
            self.assertEqual(report["source_commit"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
