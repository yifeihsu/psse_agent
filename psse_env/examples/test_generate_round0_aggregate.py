from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from psse_env.actions import RECOVERY_OPTIONS_EXHAUSTED_REQUEST
from psse_env.dagger.rollout_collector import classify_state_example
from psse_env.examples.generate_round0_aggregate import (
    BC0_FAMILY_RELEASE_POLICY,
    DEFAULT_PLAN,
    _apply_single_label_eligibility,
    _family_resolution_release_failures,
    _generation_descriptor,
    _stratified_realizability_release_failures,
    _terminal_scenario_matrix,
    _truth_free_execution_scenario,
    audit_episode_against_truth,
)


class TerminalScenarioMatrixTests(unittest.TestCase):
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
                    "resolved" if index < 19 else "operator_escalation"
                ),
                "quarantined": False,
            }
            for index in range(20)
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

    def test_hif_bearing_families_have_explicit_handoff_allowance(self) -> None:
        for family in ("hif", "measurement+hif"):
            with self.subTest(family=family):
                requirements = BC0_FAMILY_RELEASE_POLICY[family]
                self.assertEqual(requirements["minimum_resolution_rate"], 0.0)
                self.assertEqual(
                    requirements["maximum_operator_escalation_rate"], 1.0
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
        ):
            descriptor = _generation_descriptor(args, DEFAULT_PLAN)

        config = descriptor["generation_config"]
        self.assertEqual(
            config["critical_split_minimums"], {"validation": 5, "test": 5}
        )
        self.assertEqual(
            config["training_view"],
            {
                "size_policy": "natural_train_row_count_with_bounded_replacement",
                "strict_target_axes": ["tool_category"],
                "capacity_aware_target_axes": [
                    "state_class",
                    "target_tool",
                    "scenario_family",
                    "error_cardinality",
                    "terminal_outcome",
                ],
                "capacity_aware_policy": "uniform_then_clip_and_redistribute_v1",
                "max_duplicate_count": 2,
                "low_cost_margin_threshold": 0.05,
                "maximum_tool_category_target_deviation": 0.10,
            },
        )
        self.assertEqual(config["family_release_policy"], BC0_FAMILY_RELEASE_POLICY)
        self.assertIn(
            "psse_env/dagger/replay_buffer.py", descriptor["generator_hashes"]
        )
        self.assertIn("psse_env/dagger/splits.py", descriptor["generator_hashes"])


if __name__ == "__main__":
    unittest.main()
