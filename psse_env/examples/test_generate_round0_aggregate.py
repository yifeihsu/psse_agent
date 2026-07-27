from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from psse_env.actions import RECOVERY_OPTIONS_EXHAUSTED_REQUEST
from psse_env.dagger.evaluator import fingerprint_evaluation_suites
from psse_env.dagger.rollout_collector import classify_state_example
from psse_env.dagger.splits import PHYSICAL_FINGERPRINT_VERSION
from psse_env.examples.generate_round0_aggregate import (
    BC0_AGGREGATE_SOURCE_PARTITION,
    BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS,
    BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS,
    BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS,
    BC0_FAMILY_RELEASE_POLICY,
    DAGGER_ITERATION_1_RECOVERY_GATE_POLICY,
    DAGGER_ITERATION_1_RECOVERY_MINIMUM_DISTINCT_ROOTS,
    DAGGER_ITERATION_1_RECOVERY_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS,
    DAGGER_ITERATION_1_RECOVERY_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS,
    DEFAULT_EVALUATION_POLICY_PATH,
    DEFAULT_EVALUATION_SUITE_PATH,
    DEFAULT_PLAN,
    _apply_single_label_eligibility,
    _assert_training_view_export_integrity,
    _configured_input_corpora,
    _evaluation_policy_binding,
    _evaluation_suite_binding,
    _family_resolution_release_failures,
    _generation_descriptor,
    _holdout_disjointness_report,
    _holdout_release_failures,
    _input_artifact_release_failures,
    _input_corpus_release_failures,
    _stratified_approximate_realizability,
    _stratified_realizability_release_failures,
    _terminal_scenario_matrix,
    _truth_free_execution_scenario,
    audit_episode_against_truth,
    generate,
    main,
)
from psse_env.sft.provenance import file_sha256, stable_json_sha256


_PHYSICAL_PREFIX = f"physical_v{PHYSICAL_FINGERPRINT_VERSION}_"
_EVAL_PHYSICAL_1 = _PHYSICAL_PREFIX + "1" * 64
_EVAL_PHYSICAL_2 = _PHYSICAL_PREFIX + "2" * 64
_TRAIN_PHYSICAL_1 = _PHYSICAL_PREFIX + "3" * 64
_EASY_EVAL_PHYSICAL = _PHYSICAL_PREFIX + "4" * 64


class GeneratorCliTests(unittest.TestCase):
    @staticmethod
    def _report(*, release_eligible: bool) -> dict:
        return {
            "generation_provenance": {"release_eligible": release_eligible},
            "episode_rows": 1,
            "recovery_rows": 2,
            "quarantined_rows": 0,
            "split_rows": {"train_view": 1, "validation": 1, "test": 1},
            "state_class_distribution": {"diagnosis": 1},
        }

    def test_cli_exits_nonzero_when_written_artifact_is_not_release_eligible(self) -> None:
        with patch.object(sys, "argv", ["generate_round0_aggregate"]), patch(
            "psse_env.examples.generate_round0_aggregate.generate",
            return_value=self._report(release_eligible=False),
        ), self.assertRaises(SystemExit) as raised:
            main()

        self.assertEqual(raised.exception.code, 1)

    def test_cli_returns_normally_for_release_eligible_artifact(self) -> None:
        with patch.object(sys, "argv", ["generate_round0_aggregate"]), patch(
            "psse_env.examples.generate_round0_aggregate.generate",
            return_value=self._report(release_eligible=True),
        ):
            self.assertIsNone(main())


class TerminalScenarioMatrixTests(unittest.TestCase):
    def test_training_view_export_integrity_rejects_dropped_rows(self) -> None:
        selected = [
            {"example_id": "row-a"},
            {"example_id": "row-b"},
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "Balanced training-view export cardinality mismatch: selected=2, exported=1",
        ):
            _assert_training_view_export_integrity(
                selected, [{"example_id": "row-a"}]
            )

    def test_training_view_export_integrity_rejects_substitution(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "Balanced training-view export identity mismatch"
        ):
            _assert_training_view_export_integrity(
                [{"example_id": "row-a"}],
                [{"example_id": "row-b"}],
            )

    def test_terminal_teacher_targets_use_distinct_replay_classes(self) -> None:
        resolved = classify_state_example(
            {}, preferred_action={"tool": "finalize_diagnosis", "arguments": {}}
        )
        escalated = classify_state_example(
            {},
            preferred_action={
                "tool": "ask_for_more_evidence",
                "arguments": {"request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST},
            },
        )

        self.assertEqual(resolved, "terminal_resolved")
        self.assertEqual(escalated, "terminal_operator_escalation")
        self.assertNotEqual(resolved, escalated)

    def test_matrix_requires_every_root_terminal_and_unquarantined(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "m1",
                    "physical_root_fingerprint": "physical-m1",
                    "scenario_family": "measurement",
                    "terminal": True,
                    "terminal_outcome": "resolved",
                    "quarantined": False,
                },
                {
                    "scenario_id": "mt1",
                    "physical_root_fingerprint": "physical-mt1",
                    "scenario_family": "measurement+topology",
                    "terminal": False,
                    "quarantined": False,
                },
                {
                    "scenario_id": "mt2",
                    "physical_root_fingerprint": "physical-mt2",
                    "scenario_family": "measurement+topology",
                    "terminal": True,
                    "terminal_outcome": "operator_escalation",
                    "quarantined": True,
                },
            ]
        )

        self.assertTrue(matrix["measurement"]["release_terminal_coverage"])
        self.assertFalse(matrix["measurement"]["release_resolution_coverage"])
        self.assertEqual(matrix["measurement"]["distinct_physical_roots"], 1)
        self.assertEqual(
            matrix["measurement"]["terminal_outcome_counts"], {"resolved": 1}
        )
        mixed = matrix["measurement+topology"]
        self.assertEqual(mixed["episodes"], 2)
        self.assertEqual(mixed["terminal_episodes"], 1)
        self.assertEqual(mixed["nonterminal_episode_ids"], ["mt1"])
        self.assertEqual(mixed["quarantined_episode_ids"], ["mt2"])
        self.assertEqual(mixed["operator_escalation_episode_ids"], ["mt2"])
        self.assertEqual(
            mixed["terminal_outcome_counts"], {"operator_escalation": 1}
        )
        self.assertFalse(mixed["release_terminal_coverage"])
        self.assertFalse(mixed["release_resolution_coverage"])

    def test_escalation_is_terminal_but_not_resolution_coverage(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "mh1",
                    "physical_root_fingerprint": "physical-mh1",
                    "scenario_family": "measurement+hif",
                    "terminal": True,
                    "terminal_outcome": "operator_escalation",
                    "quarantined": False,
                }
            ]
        )

        hif = matrix["measurement+hif"]
        self.assertTrue(hif["release_terminal_coverage"])
        self.assertFalse(hif["release_resolution_coverage"])
        self.assertEqual(hif["resolution_rate"], 0.0)
        self.assertEqual(hif["operator_escalation_rate"], 1.0)

    def test_release_resolution_coverage_honors_policy_boundary(self) -> None:
        audits = [
            {
                "scenario_id": f"mp{index}",
                "physical_root_fingerprint": f"physical-mp{index}",
                "scenario_family": "measurement+parameter",
                "terminal": True,
                "terminal_outcome": (
                    "resolved" if index < 38 else "operator_escalation"
                ),
                "quarantined": False,
            }
            for index in range(40)
        ]

        entry = _terminal_scenario_matrix(audits)["measurement+parameter"]

        self.assertEqual(entry["audit_verified_resolution_rate"], 0.95)
        self.assertEqual(entry["operator_escalation_rate"], 0.05)
        self.assertTrue(entry["release_terminal_coverage"])
        self.assertTrue(entry["release_resolution_coverage"])

    def test_release_resolution_coverage_requires_policy_root_floor(self) -> None:
        audits = [
            {
                "scenario_id": f"mp{index}",
                "physical_root_fingerprint": f"physical-mp{index}",
                "scenario_family": "measurement+parameter",
                "terminal": True,
                "terminal_outcome": "resolved",
                "quarantined": False,
            }
            for index in range(19)
        ]

        entry = _terminal_scenario_matrix(audits)["measurement+parameter"]

        self.assertTrue(entry["release_terminal_coverage"])
        self.assertFalse(entry["release_resolution_coverage"])

    def test_duplicate_episodes_do_not_inflate_distinct_root_floor_or_rates(self) -> None:
        audits = [
            {
                "scenario_id": f"mp{index}",
                "physical_root_fingerprint": (
                    "physical-mp0" if index == 19 else f"physical-mp{index}"
                ),
                "scenario_family": "measurement+parameter",
                "terminal": True,
                "terminal_outcome": "resolved",
                "quarantined": False,
            }
            for index in range(20)
        ]

        entry = _terminal_scenario_matrix(audits)["measurement+parameter"]

        self.assertEqual(entry["episodes"], 20)
        self.assertEqual(entry["distinct_physical_roots"], 19)
        self.assertEqual(entry["resolution_rate"], 1.0)
        self.assertEqual(
            entry["duplicate_physical_root_fingerprints"],
            {"physical-mp0": ["mp0", "mp19"]},
        )
        self.assertFalse(entry["release_resolution_coverage"])

    def test_missing_physical_root_fingerprint_fails_closed(self) -> None:
        entry = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "missing-root",
                    "scenario_family": "measurement",
                    "terminal": True,
                    "terminal_outcome": "resolved",
                    "quarantined": False,
                }
            ]
        )["measurement"]

        self.assertEqual(entry["distinct_physical_roots"], 0)
        self.assertEqual(
            entry["missing_physical_root_episode_ids"], ["missing-root"]
        )
        self.assertFalse(entry["release_terminal_coverage"])
        self.assertFalse(entry["release_resolution_coverage"])

    def test_quarantined_resolved_claim_is_not_verified_resolution(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "bad-claim",
                    "physical_root_fingerprint": "physical-bad-claim",
                    "scenario_family": "measurement+parameter",
                    "terminal": True,
                    "terminal_outcome": "resolved",
                    "quarantined": True,
                }
            ]
        )

        entry = matrix["measurement+parameter"]
        self.assertEqual(entry["claimed_resolved_episode_ids"], ["bad-claim"])
        self.assertEqual(entry["resolved_episode_ids"], [])
        self.assertEqual(entry["claimed_resolution_rate"], 1.0)
        self.assertEqual(entry["audit_verified_resolution_rate"], 0.0)
        self.assertEqual(entry["resolution_rate"], 0.0)

    def test_unknown_terminal_outcome_is_not_release_terminal_coverage(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "legacy",
                    "physical_root_fingerprint": "physical-legacy",
                    "scenario_family": "measurement",
                    "terminal": True,
                    "quarantined": False,
                }
            ]
        )
        entry = matrix["measurement"]
        self.assertEqual(entry["unknown_terminal_outcome_episode_ids"], ["legacy"])
        self.assertFalse(entry["release_terminal_coverage"])


class FamilyResolutionReleaseTests(unittest.TestCase):
    POLICY = {
        "mixed": {
            "minimum_physical_roots": 20,
            "minimum_resolution_rate": 0.95,
            "maximum_operator_escalation_rate": 0.05,
        }
    }

    def test_root_resolution_and_escalation_shortfalls_are_all_reported(self) -> None:
        failures = _family_resolution_release_failures(
            {
                "mixed": {
                    "episodes": 19,
                    "distinct_physical_roots": 19,
                    "resolution_rate": 18 / 19,
                    "operator_escalation_rate": 1 / 19,
                }
            },
            policy=self.POLICY,
        )

        self.assertEqual(len(failures), 3)
        self.assertIn(
            "mixed: 19 distinct physical roots < required 20", failures
        )
        self.assertTrue(
            any("resolution rate" in failure for failure in failures), failures
        )
        self.assertTrue(
            any("operator-escalation rate" in failure for failure in failures),
            failures,
        )

    def test_policy_accepts_rates_exactly_on_release_boundaries(self) -> None:
        failures = _family_resolution_release_failures(
            {
                "mixed": {
                    "episodes": 20,
                    "distinct_physical_roots": 20,
                    "resolution_rate": 0.95,
                    "operator_escalation_rate": 0.05,
                }
            },
            policy=self.POLICY,
        )

        self.assertEqual(failures, [])

    def test_missing_family_fails_closed(self) -> None:
        failures = _family_resolution_release_failures({}, policy=self.POLICY)

        self.assertEqual(len(failures), 2)
        self.assertTrue(any("physical roots" in failure for failure in failures))
        self.assertTrue(any("resolution rate" in failure for failure in failures))

    def test_positive_count_planned_family_without_policy_fails_closed(self) -> None:
        failures = _family_resolution_release_failures(
            {},
            policy={},
            plan={"future_family": 1, "disabled_family": 0},
        )

        self.assertEqual(
            failures,
            [
                "positive-count planned families lack BC0 release policy: "
                "future_family"
            ],
        )


class OfflineTruthBoundaryTests(unittest.TestCase):
    @staticmethod
    def _case() -> dict[str, object]:
        return {
            "baseMVA": 100.0,
            "bus": [[1.0, 3.0], [2.0, 1.0]],
            "branch": [[1.0, 2.0, 0.01, 0.02, 1.0]],
        }

    def test_truth_free_execution_scenario_strips_every_offline_truth_field(self) -> None:
        scenario = {
            "scenario_id": "mixed-root",
            "root_scenario_id": "mixed-root",
            "scenario_family": "measurement+parameter",
            "case": "case14",
            "measurements": [1.0, 2.0],
            "true_measurement_errors": [{"index": 1}],
            "true_parameter_errors": [{"branch_row0": 0}],
            "true_topology_errors": [],
            "true_custom_future_family": [{"target": 7}],
            "clean_case": "clean-case14",
            "clean_measurements": [1.0, 1.0],
            "clean_state": {"case": "clean-case14", "measurements": [1.0, 1.0]},
            "hidden_truth": {"true_hif_errors": [{"branch_row0": 1}]},
            "oracle_action_hints": [{"tool": "correct_measurements"}],
            "release_audit": {"tolerances": {"measurement_abs": 0.01}},
            "metadata": {"observable_source": "tracked"},
        }
        original = copy.deepcopy(scenario)

        execution = _truth_free_execution_scenario(scenario)

        self.assertEqual(scenario, original)
        self.assertEqual(
            set(execution),
            {
                "scenario_id",
                "root_scenario_id",
                "scenario_family",
                "case",
                "measurements",
                "release_audit",
                "metadata",
            },
        )
        self.assertIsNot(execution["measurements"], scenario["measurements"])

    def test_resolved_truth_audit_fails_closed_without_final_physical_evidence(self) -> None:
        case = self._case()
        result = audit_episode_against_truth(
            {
                "scenario_id": "resolved-without-store-payload",
                "physical_root_fingerprint": "physical-resolved-without-store",
                "scenario_family": "measurement",
                "case": case,
                "clean_case": copy.deepcopy(case),
                "measurements": [1.0, 2.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [],
                "true_parameter_errors": [],
                "true_topology_errors": [],
            },
            {"accepted_corrections": []},
            terminal=True,
            terminal_outcome="resolved",
        )

        self.assertTrue(result["quarantined"])
        self.assertEqual(
            result["physical_root_fingerprint"],
            "physical-resolved-without-store",
        )
        remaining_check = result["checks"]["remaining_true_faults"]
        self.assertEqual(remaining_check["status"], "passed")
        self.assertEqual(remaining_check["derived_remaining_fault_count"], 0)
        self.assertEqual(
            remaining_check["evidence_source"],
            "offline_scenario_truth_derivation",
        )
        self.assertIn(
            "healthy_measurement_preservation_evidence_missing_or_malformed",
            result["problems"],
        )
        self.assertIn(
            "healthy_measurement_preservation_evidence_missing_or_malformed",
            result["problems"],
        )
        self.assertIn(
            "final_clean_measurement_evidence_missing_or_malformed",
            result["problems"],
        )
        self.assertIn(
            "final_clean_case_evidence_missing_or_unloadable", result["problems"]
        )


class AggregateReleaseContractTests(unittest.TestCase):
    @staticmethod
    def _holdout_payload(
        *, scenario_id: str = "eval-root-1", physical_root: str = _EVAL_PHYSICAL_1
    ) -> dict[str, list[dict[str, object]]]:
        frozen = json.loads(DEFAULT_EVALUATION_SUITE_PATH.read_text(encoding="utf-8"))
        row = copy.deepcopy(frozen["standard_success"][0])
        row["execution"]["scenario_id"] = scenario_id
        row["grouping"]["root_scenario_id"] = scenario_id
        row["grouping"]["physical_root_fingerprint"] = physical_root
        return {"standard_success": [row]}

    @staticmethod
    def _write_matching_policy(suite_path: Path, policy_path: Path) -> None:
        suites = json.loads(suite_path.read_text(encoding="utf-8"))
        fingerprint = fingerprint_evaluation_suites(
            suites,
            seed=20260719,
            required_suites=["standard_success"],
            minimum_suites=1,
            minimum_episodes_per_suite=1,
            minimum_roots_per_suite={"standard_success": 1},
        )
        manifest_fields = {
            name: fingerprint[name]
            for name in (
                "suite_manifest",
                "suite_content_hashes",
                "suite_root_set_hashes",
                "suite_content_sha256",
                "root_set_sha256",
            )
        }
        policy = json.loads(
            DEFAULT_EVALUATION_POLICY_PATH.read_text(encoding="utf-8")
        )
        policy["suite_policy"].update(
            {
                "status": "pinned",
                "approved_suite_sha256": file_sha256(suite_path),
                "approved_suite_manifest": manifest_fields,
                "required_suites": ["standard_success"],
                "minimum_physical_roots_per_suite": {"standard_success": 1},
            }
        )
        policy_path.write_text(json.dumps(policy), encoding="utf-8")

    def test_failed_approximate_family_stratum_blocks_release(self) -> None:
        reports = {
            "measurement+topology": {
                "passed": False,
                "labeled_examples": 24,
                "nearest_neighbor_compared_examples": 2,
                "nearest_neighbor_comparison_coverage": 2 / 24,
                "local_perturbation_compared_examples": 1,
                "local_perturbation_comparison_coverage": 1 / 24,
                "multi_action_cost_margin_coverage": 1.0,
            },
            "measurement+parameter": {"passed": True},
        }

        failures = _stratified_realizability_release_failures(
            reports, dimension="scenario_family"
        )

        self.assertEqual(len(failures), 1)
        self.assertIn(
            "scenario_family=measurement+topology", failures[0]
        )

    @staticmethod
    def _safe_underpowered_approximate_report() -> dict[str, object]:
        return {
            "passed": False,
            "labeled_examples": 1,
            "invalid_examples": [],
            "approximate_conflict_rate": 0.0,
            "conflict_tolerance": 0.05,
            "nearest_neighbor_action_disagreement_rate": 0.0,
            "nearest_neighbor_tolerance": 0.10,
            "nearest_neighbor_compared_examples": 0,
            "nearest_neighbor_comparison_coverage": 0.0,
            "local_perturbation_action_disagreement_rate": 0.0,
            "local_perturbation_tolerance": 0.10,
            "local_perturbation_compared_examples": 0,
            "local_perturbation_comparison_coverage": 0.0,
            "multi_action_cost_margin_coverage": 1.0,
        }

    def test_underpowered_family_reports_neighbor_gate_not_applicable(self) -> None:
        row = {
            "scenario_family": "hif",
            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
        }
        with patch(
            "psse_env.examples.generate_round0_aggregate."
            "audit_approximate_teacher_realizability",
            return_value=self._safe_underpowered_approximate_report(),
        ):
            reports = _stratified_approximate_realizability(
                [row],
                "scenario_family",
                required_values=("hif",),
                neighbor_gate_minimum_distinct_roots=20,
            )

        report = reports["hif"]
        self.assertTrue(report["stratified_safety_passed"])
        self.assertFalse(report["neighbor_stability_gate_applicable"])
        self.assertTrue(report["release_gate_passed"])
        self.assertEqual(
            report["release_gate_status"],
            "safety_passed_neighbor_underpowered",
        )
        self.assertEqual(
            _stratified_realizability_release_failures(
                reports, dimension="scenario_family"
            ),
            [],
        )

    def test_family_with_enough_roots_retains_neighbor_stability_gate(self) -> None:
        rows = [
            {
                "scenario_family": "multi_measurement",
                "physical_root_fingerprint": f"{_PHYSICAL_PREFIX}{index:064x}",
            }
            for index in range(20)
        ]
        with patch(
            "psse_env.examples.generate_round0_aggregate."
            "audit_approximate_teacher_realizability",
            return_value=self._safe_underpowered_approximate_report(),
        ):
            reports = _stratified_approximate_realizability(
                rows,
                "scenario_family",
                neighbor_gate_minimum_distinct_roots=20,
            )

        report = reports["multi_measurement"]
        self.assertTrue(report["neighbor_stability_gate_applicable"])
        self.assertFalse(report["release_gate_passed"])
        self.assertEqual(report["release_gate_status"], "failed_neighbor_stability")

    def test_state_class_neighbor_comparisons_are_diagnostic_only(self) -> None:
        row = {
            "state_class": "diagnosis",
            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
        }
        with patch(
            "psse_env.examples.generate_round0_aggregate."
            "audit_approximate_teacher_realizability",
            return_value=self._safe_underpowered_approximate_report(),
        ):
            reports = _stratified_approximate_realizability(
                [row], "state_class", neighbor_gate_minimum_distinct_roots=None
            )

        report = reports["diagnosis"]
        self.assertTrue(report["release_gate_passed"])
        self.assertEqual(
            report["release_gate_status"],
            "safety_passed_neighbor_diagnostic_only",
        )

    def test_empty_required_family_fails_stratified_safety(self) -> None:
        reports = _stratified_approximate_realizability(
            [],
            "scenario_family",
            required_values=("measurement+hif",),
            neighbor_gate_minimum_distinct_roots=20,
        )

        report = reports["measurement+hif"]
        self.assertFalse(report["stratified_safety_passed"])
        self.assertFalse(report["release_gate_passed"])
        self.assertEqual(report["release_gate_status"], "failed_safety")

    def test_release_gate_result_takes_precedence_over_raw_audit_result(self) -> None:
        self.assertEqual(
            _stratified_realizability_release_failures(
                {
                    "hif": {
                        "passed": False,
                        "release_gate_passed": True,
                    }
                },
                dimension="scenario_family",
            ),
            [],
        )
        failures = _stratified_realizability_release_failures(
            {
                "multi_measurement": {
                    "passed": True,
                    "release_gate_passed": False,
                    "release_gate_status": "failed_safety",
                }
            },
            dimension="scenario_family",
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("status=failed_safety", failures[0])

    def test_stratified_safety_failure_is_binding_without_neighbor_gate(self) -> None:
        unsafe = self._safe_underpowered_approximate_report()
        unsafe["invalid_examples"] = [
            {"example_id": "bad-row", "reason": "missing target"}
        ]
        row = {
            "state_class": "diagnosis",
            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
        }
        with patch(
            "psse_env.examples.generate_round0_aggregate."
            "audit_approximate_teacher_realizability",
            return_value=unsafe,
        ):
            reports = _stratified_approximate_realizability(
                [row], "state_class", neighbor_gate_minimum_distinct_roots=None
            )

        report = reports["diagnosis"]
        self.assertFalse(report["stratified_safety_passed"])
        self.assertFalse(report["release_gate_passed"])
        self.assertEqual(report["release_gate_status"], "failed_safety")

    def test_failed_approximate_state_stage_blocks_release(self) -> None:
        failures = _stratified_realizability_release_failures(
            {
                "terminal_resolved": {
                    "passed": False,
                    "labeled_examples": 5,
                    "nearest_neighbor_compared_examples": 0,
                    "nearest_neighbor_comparison_coverage": 0.0,
                    "local_perturbation_compared_examples": 0,
                    "local_perturbation_comparison_coverage": 0.0,
                    "multi_action_cost_margin_coverage": 1.0,
                }
            },
            dimension="state_class",
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("state_class=terminal_resolved", failures[0])

    @staticmethod
    def _ambiguous_row() -> dict[str, object]:
        state_id = "episode:test:s0"
        observation = {
            "active_state_id": state_id,
            "candidate_state_id": None,
            "candidate_parent_id": None,
            "episode_id": "episode:test",
            "remaining_budget": 4,
            "history_window": [],
            "unresolved_signatures": [],
            "remaining_anomaly_score": None,
            "no_material_anomaly_remaining": False,
        }
        actions = [
            {"tool": "run_wls", "arguments": {"state_id": state_id}},
            {
                "tool": "get_measurement_context",
                "arguments": {"state_id": state_id},
            },
        ]
        return {
            "example_id": "ambiguous",
            "policy_observation": observation,
            "history_window": [],
            "preferred_action": actions[0],
            "valid_next_actions": actions,
            "labels": {},
        }

    def test_unranked_multi_action_row_is_auxiliary(self) -> None:
        row = self._ambiguous_row()

        self.assertEqual(_apply_single_label_eligibility(row), 2)

        self.assertIs(row["production_label_eligible"], False)
        self.assertEqual(
            row["dataset_source"], "dagger_unranked_multi_action_auxiliary"
        )
        self.assertEqual(
            row["labels"]["production_ineligibility_reason"],
            "multiple_semantic_actions_without_cost_margin",
        )

    def test_ranked_multi_action_row_remains_production_eligible(self) -> None:
        row = self._ambiguous_row()
        row["cost_margin"] = 0.2

        self.assertEqual(_apply_single_label_eligibility(row), 2)

        self.assertNotIn("production_label_eligible", row)
        self.assertNotIn("production_ineligibility_reason", row)

    def test_deferred_actions_require_exact_bc0_sequential_contract(self) -> None:
        row = self._ambiguous_row()
        row["valid_next_actions"] = [row["preferred_action"]]
        row["deferred_expert_actions"] = [
            {
                "tool": "get_measurement_context",
                "arguments": {
                    "state_id": row["policy_observation"]["active_state_id"]
                },
            }
        ]
        row["dataset_mode"] = "production"

        self.assertEqual(_apply_single_label_eligibility(row), 1)

        self.assertIs(row["production_label_eligible"], False)
        self.assertEqual(
            row["production_ineligibility_reason"],
            "deferred_actions_without_bc0_sequential_contract",
        )

    def test_bc0_sequential_contract_keeps_rank_one_label_eligible(self) -> None:
        row = self._ambiguous_row()
        row["valid_next_actions"] = [row["preferred_action"]]
        row["deferred_expert_actions"] = [
            {
                "tool": "get_measurement_context",
                "arguments": {
                    "state_id": row["policy_observation"]["active_state_id"]
                },
            }
        ]
        row["dataset_mode"] = "production"
        row["supervision_policy"] = "bc0_observable_sequential_v1"

        self.assertEqual(_apply_single_label_eligibility(row), 1)

        self.assertNotIn("production_label_eligible", row)
        self.assertNotIn("production_ineligibility_reason", row)

    def test_default_plan_can_meet_family_and_evaluation_split_minima(self) -> None:
        evaluation_floor = 5 + 5
        self.assertEqual(
            set(DEFAULT_PLAN),
            set(BC0_FAMILY_RELEASE_POLICY),
        )
        for family, requirements in BC0_FAMILY_RELEASE_POLICY.items():
            with self.subTest(family=family):
                self.assertGreaterEqual(
                    DEFAULT_PLAN[family],
                    int(requirements["minimum_physical_roots"]),
                )
                if int(requirements["minimum_physical_roots"]) >= evaluation_floor:
                    self.assertGreaterEqual(DEFAULT_PLAN[family], evaluation_floor)

    def test_tracked_release_plan_is_ten_family_row_budget_plan(self) -> None:
        plan_path = Path(__file__).resolve().parents[2] / "data" / "round0_plan_20260719.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertEqual(set(plan), set(BC0_FAMILY_RELEASE_POLICY))
        self.assertEqual(sum(plan.values()), 263)
        self.assertEqual(plan["hif"], 17)
        self.assertEqual(plan["measurement+parameter"], 36)
        for family, minimum in DEFAULT_PLAN.items():
            with self.subTest(family=family):
                self.assertGreaterEqual(plan[family], minimum)

    def test_hif_bearing_families_have_explicit_handoff_allowance(self) -> None:
        for family in ("hif", "measurement+hif"):
            with self.subTest(family=family):
                requirements = BC0_FAMILY_RELEASE_POLICY[family]
                self.assertEqual(requirements["minimum_resolution_rate"], 0.0)
                self.assertEqual(
                    requirements["maximum_operator_escalation_rate"], 1.0
                )

    def test_critical_tool_family_root_floors_fit_default_family_plans(
        self,
    ) -> None:
        for tool, family_floors in (
            BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS.items()
        ):
            for family, floor in family_floors.items():
                with self.subTest(tool=tool, family=family):
                    self.assertIn(family, DEFAULT_PLAN)
                    self.assertLessEqual(floor, DEFAULT_PLAN[family])

    def test_measurement_hif_is_handoff_only_not_correction_supervision(
        self,
    ) -> None:
        family_floors = (
            BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS
        )

        self.assertEqual(
            family_floors["ask_for_more_evidence"]["measurement+hif"],
            2,
        )
        self.assertNotIn(
            "measurement+hif",
            family_floors["correct_measurements"],
        )

    def test_bc0_and_dagger_recovery_gates_are_phase_separated(self) -> None:
        self.assertNotIn(
            "rollback_state",
            BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS,
        )
        self.assertNotIn(
            "rollback_state",
            BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS,
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_MINIMUM_DISTINCT_ROOTS,
            10,
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY["phase"],
            "dagger_iteration_1",
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY["target_tool"],
            "rollback_state",
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY["required_state_class"],
            "rejected_candidate_recovery",
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY[
                "minimum_distinct_physical_roots"
            ],
            10,
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS,
            {"rollback_state": 10},
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS,
            {
                "rollback_state": {
                    "rejected_candidate_recovery": 10,
                }
            },
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY[
                "target_tool_minimum_distinct_physical_roots"
            ],
            DAGGER_ITERATION_1_RECOVERY_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS,
        )
        self.assertEqual(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY[
                "target_tool_state_class_minimum_distinct_physical_roots"
            ],
            (
                DAGGER_ITERATION_1_RECOVERY_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS
            ),
        )
        self.assertIs(
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY[
                "require_production_label_eligible"
            ],
            True,
        )
        self.assertIn(
            "synthetic_counterfactual",
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY[
                "prohibited_dataset_sources"
            ],
        )
        self.assertNotIn(
            "synthetic_counterfactual",
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY[
                "allowed_dataset_sources"
            ],
        )

    def test_partial_continuation_floors_follow_reachable_protocol_direction(
        self,
    ) -> None:
        floors = BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS
        self.assertEqual(
            floors["correct_measurements"]["accepted_partial_continuation"],
            5,
        )
        self.assertEqual(
            floors["correct_parameters"]["accepted_partial_continuation"],
            5,
        )
        self.assertEqual(
            floors["get_measurement_context"][
                "accepted_partial_continuation"
            ],
            10,
        )
        for impossible_direction in (
            "correct_topology",
            "get_parameter_context",
            "get_topology_context",
        ):
            with self.subTest(tool=impossible_direction):
                self.assertNotIn(impossible_direction, floors)

    def test_aggregate_family_policy_matches_evaluation_policy(self) -> None:
        evaluation_policy = json.loads(
            DEFAULT_EVALUATION_POLICY_PATH.read_text(encoding="utf-8")
        )

        self.assertEqual(
            BC0_FAMILY_RELEASE_POLICY,
            evaluation_policy["family_policy"],
        )

    def test_generation_descriptor_records_split_and_training_view_contracts(self) -> None:
        args = SimpleNamespace(
            protocol="canonical",
            seed=20260719,
            max_steps=24,
            counterfactuals_per_scenario=3,
            chi2_alpha=0.01,
            hif_alpha_grid=5,
            hif_r_grid=7,
            hif_max_scans=3,
        )
        with (
            patch(
                "psse_env.examples.generate_round0_aggregate.git_source_state",
                return_value={"source_commit": "test", "release_eligible_source": True},
            ),
            patch(
                "psse_env.examples.generate_round0_aggregate.file_sha256",
                return_value="sha256-test",
            ),
            patch(
                "psse_env.examples.generate_round0_aggregate.unified_tool_schemas",
                return_value=[],
            ),
            patch(
                "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                return_value=True,
            ),
            patch(
                "psse_env.examples.generate_round0_aggregate._tracked_parameter_artifact_binding",
                return_value={
                    "root": "artifacts/measurements/out_measurements_balanced/cases_parameter_error",
                    "file_count": 1,
                    "files": {"artifacts/parameter.m": "sha256-test"},
                    "tree_sha256": "tree-sha256-test",
                },
            ),
            patch(
                "psse_env.examples.generate_round0_aggregate.validate_builder_environment",
                return_value={
                    "python_version": "3.12.3",
                    "packages": {
                        "numpy": "2.3.5",
                        "scipy": "1.16.3",
                        "PYPOWER": "5.1.19",
                        "fastmcp": "2.12.4",
                    },
                },
            ),
        ):
            descriptor = _generation_descriptor(args, DEFAULT_PLAN)

        config = descriptor["generation_config"]
        self.assertEqual(
            descriptor["builder_environment"]["python_version"], "3.12.3"
        )
        self.assertEqual(BC0_AGGREGATE_SOURCE_PARTITION, "train")
        self.assertEqual(
            config["source_partition"], BC0_AGGREGATE_SOURCE_PARTITION
        )
        self.assertEqual(
            config["critical_split_minimums"], {"validation": 5, "test": 5}
        )
        self.assertEqual(
            config["supervision_policy"],
            "bc0_observable_sequential_v1",
        )
        self.assertEqual(
            config["dagger_iteration_1_recovery_gate"],
            DAGGER_ITERATION_1_RECOVERY_GATE_POLICY,
        )
        self.assertEqual(
            config["stratified_approximate_realizability"],
            {
                "global_population_gate": "full_release_thresholds",
                "family_neighbor_gate_minimum_distinct_roots": 20,
                "underpowered_family_status": "reported_not_applicable",
                "state_class_neighbor_gate": "diagnostic_only",
                "always_binding_checks": [
                    "row_validity",
                    "physical_root_presence",
                    "quantized_conflict_rate",
                    "observed_neighbor_disagreement",
                    "observed_local_disagreement",
                    "multi_action_cost_margin_coverage",
                ],
            },
        )
        self.assertEqual(
            config["training_view"],
            {
                "size_policy": (
                    "natural_target_bearing_train_row_count_with_bounded_replacement"
                ),
                "strict_target_axes": [
                    "target_tool_distinct_physical_roots",
                    "target_tool_x_state_class_distinct_physical_roots",
                    "target_tool_x_scenario_family_distinct_physical_roots",
                ],
                "deviation_gated_target_axes": ["tool_category"],
                "capacity_aware_target_axes": [
                    "tool_category",
                    "state_class",
                    "target_tool",
                    "scenario_family",
                    "error_cardinality",
                    "terminal_outcome",
                ],
                "capacity_aware_policy": (
                    "weighted_then_clip_and_redistribute_v1"
                ),
                "requirement_aware_reservation_policy": (
                    "constrained_first_distinct_physical_root_preselection_v1"
                ),
                "configured_tool_category_weights": {
                    "baseline_diagnostics": 0.20,
                    "context_acquisition": 0.15,
                    "corrections": 0.20,
                    "verification_lifecycle": 0.20,
                    "terminal_or_handoff": 0.10,
                    "specialized_diagnostics": 0.15,
                },
                "tool_category_natural_support_floor": {
                    "minimum_natural_target_bearing_rows": 16,
                    "minimum_distinct_roots": 10,
                },
                "target_tool_minimum_distinct_physical_roots": {
                    "correct_measurements": 10,
                    "correct_parameters": 10,
                    "correct_topology": 10,
                    "get_measurement_context": 10,
                    "get_parameter_context": 10,
                    "get_topology_context": 10,
                },
                "target_tool_state_class_minimum_distinct_physical_roots": {
                    "commit_state": {
                        "accepted_final_commit": 10,
                        "accepted_partial_commit": 10,
                    },
                    "correct_measurements": {
                        "accepted_partial_continuation": 5,
                    },
                    "correct_parameters": {
                        "accepted_partial_continuation": 5,
                    },
                    "get_measurement_context": {
                        "accepted_partial_continuation": 10,
                    },
                },
                "target_tool_scenario_family_minimum_distinct_physical_roots": {
                    "ask_for_more_evidence": {
                        "hif": 5,
                        "measurement+hif": 2,
                        "multi_measurement": 10,
                    },
                    "correct_measurements": {
                        "measurement": 5,
                        "measurement+parameter": 10,
                        "measurement+topology": 10,
                        "multi_measurement": 10,
                    },
                    "correct_parameters": {
                        "measurement+parameter": 10,
                        "parameter": 5,
                    },
                    "correct_topology": {
                        "measurement+topology": 10,
                        "topology": 5,
                    },
                },
                "production_label_eligibility_policy": (
                    "explicit_true_required"
                ),
                "max_duplicate_count": 2,
                "low_cost_margin_threshold": 0.05,
                "maximum_tool_category_target_deviation": 0.10,
            },
        )

        self.assertEqual(
            descriptor["input_artifacts"]["parameter_cases"]["file_count"], 1
        )
        self.assertEqual(
            descriptor["evaluation_holdout"]["path"],
            "psse_env/dagger/suites/bc0_eval_suite_v1.json",
        )
        self.assertTrue(descriptor["evaluation_holdout"]["schema_valid"])
        self.assertEqual(descriptor["evaluation_holdout"]["episode_count"], 115)
        self.assertEqual(descriptor["evaluation_holdout"]["physical_root_count"], 115)
        self.assertEqual(descriptor["evaluation_holdout"]["scenario_id_count"], 115)
        self.assertEqual(
            descriptor["evaluation_policy"]["path"],
            "psse_env/dagger/bc0_evaluation_policy.json",
        )
        self.assertTrue(descriptor["evaluation_policy"]["schema_valid"])
        self.assertEqual(
            descriptor["evaluation_policy"]["suite_policy_status"], "pinned"
        )
        self.assertTrue(descriptor["evaluation_policy"]["approved_suite_sha256"])
        self.assertEqual(config["family_release_policy"], BC0_FAMILY_RELEASE_POLICY)
        self.assertIn(
            "psse_env/dagger/replay_buffer.py", descriptor["generator_hashes"]
        )
        self.assertIn("psse_env/dagger/splits.py", descriptor["generator_hashes"])
        self.assertIn(
            "psse_env/dagger/suite_builder.py", descriptor["generator_hashes"]
        )
        self.assertEqual(
            descriptor["input_corpora"],
            {
                "hif_corpus_0": {
                    "path": (
                        "artifacts/measurements/"
                        "hif_multiscan_benchmark_fixed_diverse_17x20_20260714/"
                        "samples.jsonl"
                    ),
                    "sha256": "sha256-test",
                    "exists": True,
                    "git_tracked": True,
                },
                "hif_quality_0": {
                    "path": (
                        "artifacts/measurements/"
                        "hif_multiscan_benchmark_fixed_diverse_17x20_20260714/"
                        "meta.json"
                    ),
                    "sha256": "sha256-test",
                    "exists": True,
                    "git_tracked": True,
                },
                "hif_quality_1": {
                    "path": (
                        "artifacts/measurements/"
                        "hif_multiscan_benchmark_fixed_diverse_17x20_20260714/"
                        "quality_report.json"
                    ),
                    "sha256": "sha256-test",
                    "exists": True,
                    "git_tracked": True,
                },
                "measurement_corpus": {
                    "path": "data/measurements_5class_merged.jsonl",
                    "sha256": "sha256-test",
                    "exists": True,
                    "git_tracked": True,
                },
            },
        )

    def test_generation_rejects_unapproved_runtime_before_reading_plan(self) -> None:
        with patch(
            "psse_env.examples.generate_round0_aggregate.validate_builder_environment",
            side_effect=RuntimeError("unapproved numerical runtime"),
        ), self.assertRaisesRegex(RuntimeError, "unapproved numerical runtime"):
            generate(SimpleNamespace())

    def test_release_defaults_to_multiscan_hif_training_corpus_and_qa_bindings(
        self,
    ) -> None:
        args = SimpleNamespace(
            measurement_corpus=None,
            hif_corpus=None,
            imbalance_corpus=None,
        )

        configured = _configured_input_corpora(args, DEFAULT_PLAN)

        self.assertEqual(
            [
                path.name
                for name, path in configured.items()
                if name.startswith("hif_corpus_")
            ],
            ["samples.jsonl"],
        )
        self.assertIn(
            "hif_multiscan_benchmark_fixed_diverse_17x20_20260714",
            configured["hif_corpus_0"].as_posix(),
        )
        self.assertEqual(
            {
                name: path.name
                for name, path in configured.items()
                if name.startswith("hif_quality_")
            },
            {"hif_quality_0": "meta.json", "hif_quality_1": "quality_report.json"},
        )

    def test_evaluation_suite_binding_records_file_and_semantic_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite_path = Path(temporary) / "holdout.json"
            suite_path.write_text(
                json.dumps(self._holdout_payload()), encoding="utf-8"
            )
            args = SimpleNamespace(evaluation_suite=suite_path)
            with patch(
                "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                return_value=True,
            ):
                binding = _evaluation_suite_binding(
                    args, repo_root=Path(temporary)
                )

        self.assertTrue(binding["schema_valid"])
        self.assertTrue(binding["git_tracked"])
        self.assertEqual(binding["episode_count"], 1)
        self.assertEqual(binding["physical_root_count"], 1)
        self.assertEqual(
            binding["physical_fingerprint_version"], PHYSICAL_FINGERPRINT_VERSION
        )
        self.assertEqual(binding["invalid_physical_root_version_entries"], [])
        self.assertEqual(
            binding["physical_root_set_sha256"],
            stable_json_sha256([_EVAL_PHYSICAL_1]),
        )
        self.assertEqual(binding["scenario_id_count"], 1)
        self.assertEqual(
            binding["scenario_id_set_sha256"],
            stable_json_sha256(["eval-root-1"]),
        )

    def test_obsolete_holdout_fingerprint_version_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            suite_path = repo_root / "holdout.json"
            suite_path.write_text(
                json.dumps(
                    self._holdout_payload(
                        physical_root="physical_v2_" + "5" * 64
                    )
                ),
                encoding="utf-8",
            )
            policy_path = repo_root / "evaluation_policy.json"
            self._write_matching_policy(suite_path, policy_path)
            with (
                patch(
                    "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                    return_value=True,
                ),
                patch(
                    "psse_env.examples.generate_round0_aggregate.DEFAULT_EVALUATION_POLICY_PATH",
                    policy_path,
                ),
            ):
                args = SimpleNamespace(
                    evaluation_suite=suite_path,
                    evaluation_policy=policy_path,
                )
                binding = _evaluation_suite_binding(args, repo_root=repo_root)
                policy_binding = _evaluation_policy_binding(
                    args,
                    evaluation_holdout=binding,
                    repo_root=repo_root,
                )
                report = _holdout_disjointness_report(
                    [
                        {
                            "scenario_id": "train-root-1",
                            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
                        }
                    ],
                    evaluation_holdout=binding,
                    evaluation_policy=policy_binding,
                    evaluation_suite_path=suite_path,
                    evaluation_policy_path=policy_path,
                    repo_root=repo_root,
                )

        self.assertFalse(report["passed"])
        self.assertEqual(
            binding["invalid_physical_root_version_entries"],
            ["standard_success[0]"],
        )
        self.assertIn(
            "evaluation suite contains physical roots from an obsolete or invalid "
            "fingerprint version",
            report["failures"],
        )

    def test_holdout_disjointness_passes_for_distinct_roots_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            suite_path = repo_root / "holdout.json"
            suite_path.write_text(
                json.dumps(self._holdout_payload()), encoding="utf-8"
            )
            policy_path = repo_root / "evaluation_policy.json"
            self._write_matching_policy(suite_path, policy_path)
            with (
                patch(
                    "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                    return_value=True,
                ),
                patch(
                    "psse_env.examples.generate_round0_aggregate.DEFAULT_EVALUATION_POLICY_PATH",
                    policy_path,
                ),
            ):
                args = SimpleNamespace(
                    evaluation_suite=suite_path,
                    evaluation_policy=policy_path,
                )
                binding = _evaluation_suite_binding(args, repo_root=repo_root)
                policy_binding = _evaluation_policy_binding(
                    args,
                    evaluation_holdout=binding,
                    repo_root=repo_root,
                )
                report = _holdout_disjointness_report(
                    [
                        {
                            "scenario_id": "train-root-1",
                            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
                        }
                    ],
                    evaluation_holdout=binding,
                    evaluation_policy=policy_binding,
                    evaluation_suite_path=suite_path,
                    evaluation_policy_path=policy_path,
                    repo_root=repo_root,
                )

        self.assertTrue(report["passed"])
        self.assertEqual(report["physical_root_intersection_count"], 0)
        self.assertEqual(report["scenario_id_intersection_count"], 0)
        self.assertEqual(_holdout_release_failures(report), [])

    def test_holdout_disjointness_rejects_physical_or_id_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            suite_path = repo_root / "holdout.json"
            suite_path.write_text(
                json.dumps(self._holdout_payload()), encoding="utf-8"
            )
            policy_path = repo_root / "evaluation_policy.json"
            self._write_matching_policy(suite_path, policy_path)
            with patch(
                "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                return_value=True,
            ):
                args = SimpleNamespace(
                    evaluation_suite=suite_path,
                    evaluation_policy=policy_path,
                )
                binding = _evaluation_suite_binding(args, repo_root=repo_root)
                policy_binding = _evaluation_policy_binding(
                    args,
                    evaluation_holdout=binding,
                    repo_root=repo_root,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "Frozen evaluation holdout overlaps"
                ):
                    _holdout_disjointness_report(
                        [
                            {
                                "scenario_id": "eval-root-1",
                                "physical_root_fingerprint": _EVAL_PHYSICAL_1,
                            }
                        ],
                        evaluation_holdout=binding,
                        evaluation_policy=policy_binding,
                        evaluation_suite_path=suite_path,
                        evaluation_policy_path=policy_path,
                        repo_root=repo_root,
                    )

    def test_holdout_suite_change_after_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            suite_path = repo_root / "holdout.json"
            suite_path.write_text(
                json.dumps(self._holdout_payload()), encoding="utf-8"
            )
            policy_path = repo_root / "evaluation_policy.json"
            self._write_matching_policy(suite_path, policy_path)
            with patch(
                "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                return_value=True,
            ):
                args = SimpleNamespace(
                    evaluation_suite=suite_path,
                    evaluation_policy=policy_path,
                )
                binding = _evaluation_suite_binding(args, repo_root=repo_root)
                policy_binding = _evaluation_policy_binding(
                    args,
                    evaluation_holdout=binding,
                    repo_root=repo_root,
                )
                suite_path.write_text(
                    json.dumps(
                        self._holdout_payload(
                            scenario_id="eval-root-2",
                            physical_root=_EVAL_PHYSICAL_2,
                        )
                    ),
                    encoding="utf-8",
                )
                report = _holdout_disjointness_report(
                    [
                        {
                            "scenario_id": "train-root-1",
                            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
                        }
                    ],
                    evaluation_holdout=binding,
                    evaluation_policy=policy_binding,
                    evaluation_suite_path=suite_path,
                    evaluation_policy_path=policy_path,
                    repo_root=repo_root,
                )

        self.assertFalse(report["passed"])
        self.assertIn("file_sha256", report["binding_mismatches"])
        self.assertTrue(_holdout_release_failures(report))

    def test_tracked_alternate_suite_must_match_policy_approved_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            suite_path = repo_root / "alternate_holdout.json"
            suite_path.write_text(
                json.dumps(self._holdout_payload()), encoding="utf-8"
            )
            policy_path = repo_root / "evaluation_policy.json"
            self._write_matching_policy(suite_path, policy_path)
            suite_path.write_text(
                json.dumps(
                    self._holdout_payload(
                        scenario_id="easy-eval-root",
                        physical_root=_EASY_EVAL_PHYSICAL,
                    )
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                evaluation_suite=suite_path,
                evaluation_policy=policy_path,
            )
            with patch(
                "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                return_value=True,
            ):
                binding = _evaluation_suite_binding(args, repo_root=repo_root)
                policy_binding = _evaluation_policy_binding(
                    args,
                    evaluation_holdout=binding,
                    repo_root=repo_root,
                )
                report = _holdout_disjointness_report(
                    [
                        {
                            "scenario_id": "train-root-1",
                            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
                        }
                    ],
                    evaluation_holdout=binding,
                    evaluation_policy=policy_binding,
                    evaluation_suite_path=suite_path,
                    evaluation_policy_path=policy_path,
                    repo_root=repo_root,
                )

        self.assertFalse(policy_binding["suite_file_sha256_matches_approval"])
        self.assertFalse(policy_binding["suite_root_set_sha256_matches_approval"])
        self.assertFalse(report["passed"])
        self.assertIn(
            "evaluation suite file SHA-256 does not match the policy approval",
            report["failures"],
        )

    def test_unpinned_evaluation_policy_blocks_holdout_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            suite_path = repo_root / "holdout.json"
            suite_path.write_text(
                json.dumps(self._holdout_payload()), encoding="utf-8"
            )
            policy_path = repo_root / "evaluation_policy.json"
            self._write_matching_policy(suite_path, policy_path)
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["suite_policy"].update(
                {
                    "status": "unconfigured",
                    "approved_suite_sha256": None,
                    "approved_suite_manifest": None,
                }
            )
            policy["approved_factories"] = {
                role: [] for role in policy["approved_factories"]
            }
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            args = SimpleNamespace(
                evaluation_suite=suite_path,
                evaluation_policy=policy_path,
            )
            with patch(
                "psse_env.examples.generate_round0_aggregate._git_tracks_file",
                return_value=True,
            ):
                binding = _evaluation_suite_binding(args, repo_root=repo_root)
                policy_binding = _evaluation_policy_binding(
                    args,
                    evaluation_holdout=binding,
                    repo_root=repo_root,
                )
                report = _holdout_disjointness_report(
                    [
                        {
                            "scenario_id": "train-root-1",
                            "physical_root_fingerprint": _TRAIN_PHYSICAL_1,
                        }
                    ],
                    evaluation_holdout=binding,
                    evaluation_policy=policy_binding,
                    evaluation_suite_path=suite_path,
                    evaluation_policy_path=policy_path,
                    repo_root=repo_root,
                )

        self.assertTrue(policy_binding["schema_valid"])
        self.assertEqual(policy_binding["suite_policy_status"], "unconfigured")
        self.assertFalse(report["passed"])
        self.assertIn(
            "evaluation policy suite status is not pinned", report["failures"]
        )

    def test_untracked_or_missing_input_corpus_blocks_release(self) -> None:
        failures = _input_corpus_release_failures(
            {
                "input_corpora": {
                    "untracked": {
                        "path": "artifacts/private/samples.jsonl",
                        "sha256": "a" * 64,
                        "exists": True,
                        "git_tracked": False,
                    },
                    "missing": {
                        "path": "data/missing.jsonl",
                        "sha256": None,
                        "exists": False,
                        "git_tracked": False,
                    },
                }
            }
        )

        self.assertIn(
            "input corpus is not repository-tracked: untracked "
            "(artifacts/private/samples.jsonl)",
            failures,
        )
        self.assertIn(
            "input corpus is missing or unhashed: missing (data/missing.jsonl)",
            failures,
        )
        self.assertIn(
            "input corpus is not repository-tracked: missing (data/missing.jsonl)",
            failures,
        )

    def test_parameter_artifact_binding_must_be_content_consistent(self) -> None:
        files = {"artifacts/parameter.m": "a" * 64}
        valid = {
            "input_artifacts": {
                "parameter_cases": {
                    "file_count": 1,
                    "files": files,
                    "tree_sha256": stable_json_sha256(files),
                }
            }
        }
        self.assertEqual(_input_artifact_release_failures(valid), [])
        invalid = copy.deepcopy(valid)
        invalid["input_artifacts"]["parameter_cases"]["file_count"] = 2
        self.assertIn(
            "parameter artifact file count is inconsistent",
            _input_artifact_release_failures(invalid),
        )


if __name__ == "__main__":
    unittest.main()
