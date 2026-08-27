from __future__ import annotations

import unittest
from unittest.mock import patch

from research.physical_outcome_audit import (
    EXACT_PHYSICAL_RECOVERY,
    FALSE_INTERVENTION,
    GENERATION_ABORT,
    LOOP_BEFORE_STABLE_FINAL_STATE,
    NO_PHYSICAL_PROGRESS,
    PARTIAL_RECOVERY,
    _committed_target_events,
    _committed_target_keys,
    _direct_fault_resolution,
    _first_loop_divergence,
    _parse_recorded_action,
    _scenario_audit,
    _scenario_execution,
    _scenario_family,
    _scenario_id,
    _strict_physical_summary,
    audit_evaluation_report,
    classify_episode,
    summarize_episodes,
)


class PhysicalOutcomeAuditTests(unittest.TestCase):
    def test_six_way_classification_gives_abort_and_loop_precedence(self) -> None:
        self.assertEqual(
            classify_episode(
                generation_abort=True,
                loop=True,
                physical_class=EXACT_PHYSICAL_RECOVERY,
            ),
            GENERATION_ABORT,
        )
        self.assertEqual(
            classify_episode(
                generation_abort=False,
                loop=True,
                physical_class=EXACT_PHYSICAL_RECOVERY,
            ),
            LOOP_BEFORE_STABLE_FINAL_STATE,
        )
        for physical_class in (
            EXACT_PHYSICAL_RECOVERY,
            PARTIAL_RECOVERY,
            FALSE_INTERVENTION,
            NO_PHYSICAL_PROGRESS,
        ):
            with self.subTest(physical_class=physical_class):
                self.assertEqual(
                    classify_episode(
                        generation_abort=False,
                        loop=False,
                        physical_class=physical_class,
                    ),
                    physical_class,
                )

    def test_scenario_alignment_uses_root_id_and_family(self) -> None:
        scenario = {
            "grouping": {
                "root_scenario_id": "root-17",
                "scenario_family": "multi_measurement",
            }
        }
        self.assertEqual(_scenario_id(scenario, 3), "root-17")
        self.assertEqual(_scenario_family(scenario), "multi-measurement")

    def test_truncated_recorded_action_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "truncated"):
            _parse_recorded_action(
                {
                    "tool": "correct_measurements",
                    "arguments": '{"measurement_updates":{"1":2.0}...'
                    "...",
                }
            )

    def test_report_and_scenario_length_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            audit_evaluation_report(
                scenarios=[{}],
                report={"per_episode": []},
                env_factory=lambda: object(),
            )

    def test_source_binding_requires_actual_per_episode_identity(self) -> None:
        with (
            patch(
                "research.physical_outcome_audit.replay_episode",
                return_value={
                    "scenario_family": "topology",
                    "scenario_alignment_basis": "ordered_index_report_id_missing",
                },
            ),
            patch(
                "research.physical_outcome_audit.summarize_episodes",
                return_value={},
            ),
        ):
            result = audit_evaluation_report(
                scenarios=[{}],
                report={"scenarios_sha256": "abc", "per_episode": [{}]},
                env_factory=lambda: object(),
            )
        self.assertEqual(
            result["source_scenario_binding"],
            "suite_sha256_with_incomplete_per_episode_identity",
        )

        with (
            patch(
                "research.physical_outcome_audit.replay_episode",
                return_value={
                    "scenario_family": "topology",
                    "scenario_alignment_basis": (
                        "validated_physical_root_fingerprint"
                    ),
                },
            ),
            patch(
                "research.physical_outcome_audit.summarize_episodes",
                return_value={},
            ),
        ):
            result = audit_evaluation_report(
                scenarios=[{}],
                report={"scenarios_sha256": "abc", "per_episode": [{}]},
                env_factory=lambda: object(),
            )
        self.assertEqual(
            result["source_scenario_binding"],
            "sha256_and_per_episode_identity",
        )

    def test_committed_corrections_are_counted_at_unique_target_level(self) -> None:
        final_state = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {
                            "state_id": "s0",
                            "suspect_group": [4, 7],
                        },
                    }
                },
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {
                            "state_id": "s1",
                            "suspect_group": [7],
                        },
                    }
                },
                {
                    "source_action": {
                        "tool": "correct_parameters",
                        "arguments": {"state_id": "s1", "line_index": 3},
                    }
                },
            ]
        }
        targets, problems = _committed_target_keys(final_state)
        self.assertEqual(
            targets,
            {"measurement:4", "measurement:7", "parameter:branch_row0:2"},
        )
        self.assertEqual(problems, [])
        events, event_problems = _committed_target_events(final_state)
        self.assertEqual(
            events,
            [
                "measurement:4",
                "measurement:7",
                "measurement:7",
                "parameter:branch_row0:2",
            ],
        )
        self.assertEqual(event_problems, [])

    def test_replay_envelope_matches_evaluate_and_audit_truth_is_separate(self) -> None:
        scenario = {
            "execution": {"case": "case14", "measurements": [3.0]},
            "audit": {
                "truth": {
                    "truth_complete": True,
                    "clean_measurements": [1.0],
                    "true_measurement_errors": [{"index": 0, "clean": 1.0}],
                    "true_parameter_errors": [],
                    "true_topology_errors": [],
                },
                "release_audit": {"tolerances": {"measurement_abs": 0.1}},
            },
            "grouping": {"root_scenario_id": "root-1"},
        }
        runtime = _scenario_execution(scenario)
        self.assertNotIn("truth_complete", runtime)
        self.assertNotIn("release_audit", runtime)
        self.assertNotIn("root_scenario_id", runtime)
        audit = _scenario_audit(scenario)
        self.assertIs(audit["truth_complete"], True)
        self.assertIn("release_audit", audit)
        self.assertEqual(audit["root_scenario_id"], "root-1")

    def test_direct_final_state_overrides_stale_remaining_truth_ledger(self) -> None:
        case = {
            "baseMVA": 100.0,
            "bus": [],
            "gen": [],
            "branch": [],
        }
        scenario = {
            "scenario_id": "measurement-ledger-regression",
            "truth_complete": True,
            "case": case,
            "measurements": [3.0, 2.0],
            "clean_case": case,
            "clean_measurements": [1.0, 2.0],
            "true_measurement_errors": [{"index": 0, "clean": 1.0}],
            "true_parameter_errors": [],
            "true_topology_errors": [],
        }
        active = {"case": case, "measurements": [1.0, 2.0]}
        stale_ledger = {
            "truth_complete": True,
            "true_measurement_errors": [{"index": 0, "clean": 1.0}],
            "true_parameter_errors": [],
            "true_topology_errors": [],
        }
        final_state = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [0]},
                    }
                }
            ]
        }
        result = _strict_physical_summary(
            scenario=scenario,
            final_state=final_state,
            active_physical_state=active,
            remaining_truth=stale_ledger,
        )
        self.assertTrue(result["physical_assessable"])
        self.assertEqual(result["final_active_state_class"], EXACT_PHYSICAL_RECOVERY)
        self.assertEqual(result["true_errors_corrected"], 1)
        self.assertEqual(result["remaining_true_error_count"], 0)
        self.assertFalse(
            result["oracle_remaining_truth_ledger_diagnostic"][
                "agrees_with_direct_final_state"
            ]
        )

    def test_direct_fault_resolution_requires_explicit_complete_truth(self) -> None:
        result = _direct_fault_resolution(
            scenario={
                "measurements": [],
                "case": {"branch": []},
                "clean_measurements": [],
                "clean_case": {"branch": []},
                "true_measurement_errors": [],
                "true_parameter_errors": [],
                "true_topology_errors": [],
            },
            active_physical_state={"measurements": [], "case": {"branch": []}},
        )
        self.assertFalse(result["truth_complete"])
        self.assertIn("scenario_truth_complete_not_explicit_true", result["problems"])

    def test_direct_fault_resolution_checks_parameter_and_topology_final_case(self) -> None:
        clean_parameter = [1.0, 2.0, 0.1, 0.2, 0.0, 0, 0, 0, 0, 0, 1, -360, 360]
        clean_topology = [2.0, 3.0, 0.2, 0.4, 0.0, 0, 0, 0, 0, 0, 1, -360, 360]
        clean_case = {"branch": [clean_parameter, clean_topology]}
        scenario = {
            "truth_complete": True,
            "case": clean_case,
            "clean_case": clean_case,
            "measurements": [],
            "clean_measurements": [],
            "true_measurement_errors": [],
            "true_parameter_errors": [
                {
                    "branch_row0": 0,
                    "parameter": "rx",
                    "clean_r": 0.1,
                    "clean_x": 0.2,
                }
            ],
            "true_topology_errors": [
                {"branch_row0": 1, "expected_status": 1}
            ],
        }
        result = _direct_fault_resolution(
            scenario=scenario,
            active_physical_state={"case": clean_case, "measurements": []},
        )
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["initial_true_target_count"], 2)
        self.assertEqual(result["restored_true_target_count"], 2)

    def test_loop_divergence_prefers_candidate_disposition_error(self) -> None:
        trace = [
            {
                "step": 0,
                "tool": "commit_state",
                "candidate_disposition_error": True,
                "repeated_action_signature": False,
                "status": "success",
                "progress_advanced": True,
            },
            {
                "step": 1,
                "tool": "get_measurement_context",
                "candidate_disposition_error": False,
                "repeated_action_signature": True,
                "status": "success",
                "progress_advanced": False,
                "expert_admissible": True,
            },
        ]
        self.assertEqual(
            _first_loop_divergence(trace),
            {"step": 0, "class": "candidate_commit_rollback_error"},
        )

    def test_loop_divergence_ignores_routine_read_only_no_progress(self) -> None:
        trace = [
            {
                "step": 0,
                "tool": "run_wls",
                "candidate_disposition_error": False,
                "repeated_action_signature": False,
                "nonconsecutive_state_hash_revisit": False,
                "controller_no_progress": False,
                "expert_admissible": True,
            },
            {
                "step": 1,
                "tool": "get_measurement_context",
                "candidate_disposition_error": False,
                "repeated_action_signature": False,
                "nonconsecutive_state_hash_revisit": True,
                "controller_no_progress": False,
                "expert_admissible": True,
            },
        ]
        self.assertEqual(
            _first_loop_divergence(trace),
            {"step": 1, "class": "nonconsecutive_state_hash_revisit"},
        )

    def test_summary_keeps_stable_outcome_separate_from_active_snapshot(self) -> None:
        episodes = [
            {
                "episode_outcome_class": EXACT_PHYSICAL_RECOVERY,
                "final_active_state_class": EXACT_PHYSICAL_RECOVERY,
                "physical_assessable": True,
                "initial_true_error_count": 2,
                "true_errors_corrected": 2,
                "true_committed_correction_count": 2,
                "committed_correction_target_count": 2,
                "committed_correction_target_event_count": 2,
                "unique_committed_correction_target_count": 2,
                "valid_action_count": 2,
                "controller_valid_action_count": 2,
                "no_progress_valid_action_count": 1,
                "steps": 2,
                "repeated_action_signature_count": 1,
                "state_hash_observation_count": 2,
                "state_hash_revisit_count": 1,
                "max_no_progress_streak": 1,
                "step_difference_relative_to_expert": 1,
                "excess_steps_relative_to_expert": 1,
                "replay_matches_record": True,
                "oracle_remaining_truth_ledger_diagnostic": {
                    "agrees_with_direct_final_state": True
                },
                "trace": [
                    {
                        "expected_candidate_action": "commit",
                        "predicted_candidate_action": "commit",
                        "candidate_disposition": "ACCEPT_FINAL",
                    }
                ],
            },
            {
                "episode_outcome_class": LOOP_BEFORE_STABLE_FINAL_STATE,
                "final_active_state_class": EXACT_PHYSICAL_RECOVERY,
                "physical_assessable": True,
                "initial_true_error_count": 1,
                "true_errors_corrected": 1,
                "true_committed_correction_count": 1,
                "committed_correction_target_count": 1,
                "committed_correction_target_event_count": 1,
                "unique_committed_correction_target_count": 1,
                "valid_action_count": 1,
                "controller_valid_action_count": 1,
                "no_progress_valid_action_count": 1,
                "steps": 1,
                "repeated_action_signature_count": 0,
                "state_hash_observation_count": 1,
                "state_hash_revisit_count": 1,
                "max_no_progress_streak": 1,
                "step_difference_relative_to_expert": -1,
                "excess_steps_relative_to_expert": 0,
                "replay_matches_record": True,
                "oracle_remaining_truth_ledger_diagnostic": {
                    "agrees_with_direct_final_state": False
                },
                "trace": [],
            },
        ]
        summary = summarize_episodes(episodes)
        self.assertEqual(summary["exact_episode_recovery_count"], 2)
        self.assertEqual(summary["exact_episode_recovery_rate"], 1.0)
        self.assertEqual(summary["final_active_snapshot_exact_recovery_count"], 2)
        self.assertEqual(summary["final_active_snapshot_exact_recovery_rate"], 1.0)
        self.assertEqual(summary["stable_terminal_exact_episode_recovery_count"], 1)
        self.assertEqual(summary["stable_terminal_exact_episode_recovery_rate"], 0.5)
        self.assertEqual(summary["correction_precision"], 1.0)
        self.assertEqual(summary["unique_target_correction_precision"], 1.0)
        self.assertEqual(summary["correction_recall"], 1.0)
        self.assertEqual(summary["oracle_truth_ledger_disagreement_episodes"], 1)
        self.assertEqual(summary["mean_excess_steps_relative_to_expert"], 0.5)
        self.assertEqual(
            summary["mean_signed_step_difference_relative_to_expert"], 0.0
        )
        self.assertEqual(
            summary["candidate_disposition_confusion_matrix"],
            {"commit": {"commit": 1}},
        )


if __name__ == "__main__":
    unittest.main()
