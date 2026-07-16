from __future__ import annotations

import copy
import unittest

from psse_env.verifier.dataset import build_verifier_example
from psse_env.verifier.features import summarize_history
from psse_env.verifier.rules import RuleBasedVerifier


def correction_transition(
    *,
    tool: str = "correct_measurements",
    arguments: dict | None = None,
) -> dict:
    if arguments is None:
        arguments = {
            "state_id": "episode:s0",
            "measurement_updates": {0: 1.0},
        }
    return {
        "parent_state_summary": {
            "episode_id": "episode",
            "active_state_id": "episode:s0",
            "candidate_state_id": None,
            "has_open_candidate": False,
            "has_fresh_parameter_context": True,
            "parameter_context_state_id": "episode:s0",
            "has_fresh_topology_context": True,
            "topology_context_state_id": "episode:s0",
        },
        "action": {"tool": tool, "arguments": arguments},
        "tool_output": {
            "execution_status": "success",
            "state_mutated": True,
            "active_state_id": "episode:s0",
            "candidate_state_id": "episode:s1",
        },
        "candidate_state_summary": {
            "episode_id": "episode",
            "active_state_id": "episode:s0",
            "candidate_state_id": "episode:s1",
            "candidate_parent_id": "episode:s0",
            "has_open_candidate": True,
            "has_unverified_candidate": True,
        },
        "verification_metrics": {},
        "history_summary": {},
    }


class VerifierIdentityHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = RuleBasedVerifier()

    def assert_invalid(self, transition: dict, reason: str) -> None:
        decision = self.verifier.verify(transition)
        self.assertFalse(decision["process_valid"])
        self.assertIn(reason, decision["rationale_codes"])

    def test_valid_namespaced_correction_remains_process_valid(self) -> None:
        decision = self.verifier.verify(correction_transition())
        self.assertTrue(decision["process_valid"])

    def test_cross_episode_successor_is_rejected(self) -> None:
        transition = correction_transition()
        transition["candidate_state_summary"].update(
            {
                "episode_id": "other",
                "active_state_id": "other:s0",
                "candidate_state_id": "other:s1",
                "candidate_parent_id": "other:s0",
            }
        )
        transition["tool_output"].update(
            {"active_state_id": "other:s0", "candidate_state_id": "other:s1"}
        )
        self.assert_invalid(transition, "cross_episode_transition")

    def test_state_id_namespace_must_match_episode(self) -> None:
        transition = correction_transition()
        transition["candidate_state_summary"]["candidate_state_id"] = "other:s1"
        transition["tool_output"]["candidate_state_id"] = "other:s1"
        self.assert_invalid(transition, "state_id_crosses_episode_boundary")

    def test_cross_episode_ids_are_rejected_without_explicit_episode_field(self) -> None:
        transition = correction_transition()
        transition["parent_state_summary"].pop("episode_id")
        transition["candidate_state_summary"].pop("episode_id")
        transition["candidate_state_summary"]["candidate_state_id"] = "other:s1"
        transition["tool_output"]["candidate_state_id"] = "other:s1"
        self.assert_invalid(transition, "state_id_crosses_episode_boundary")

    def test_candidate_parent_must_be_active_state(self) -> None:
        transition = correction_transition()
        transition["candidate_state_summary"]["candidate_parent_id"] = "episode:s9"
        self.assert_invalid(transition, "candidate_parent_mismatch")

    def test_candidate_parent_cannot_exist_without_candidate(self) -> None:
        transition = correction_transition()
        transition["candidate_state_summary"]["candidate_state_id"] = None
        transition["candidate_state_summary"]["candidate_parent_id"] = "episode:s0"
        transition["tool_output"]["candidate_state_id"] = None
        self.assert_invalid(transition, "dangling_candidate_parent_id")

    def test_output_candidate_id_must_match_successor(self) -> None:
        transition = correction_transition()
        transition["tool_output"]["candidate_state_id"] = "episode:s2"
        self.assert_invalid(transition, "output_candidate_state_id_mismatch")

    def test_output_active_id_must_match_successor(self) -> None:
        transition = correction_transition()
        transition["tool_output"]["active_state_id"] = "episode:s2"
        self.assert_invalid(transition, "output_active_state_id_mismatch")

    def test_nested_output_provenance_must_match_transition(self) -> None:
        transition = correction_transition()
        transition["tool_output"]["tool_metrics"] = {
            "episode_id": "other",
            "candidate_state_id": "other:s1",
            "parent_state_id": "other:s0",
        }
        self.assert_invalid(transition, "state_id_crosses_episode_boundary")

    def test_failed_transition_cannot_change_candidate(self) -> None:
        transition = correction_transition()
        transition["tool_output"].update(
            {"execution_status": "failure", "candidate_state_id": "episode:s1"}
        )
        transition["candidate_state_summary"].update(
            {"candidate_state_id": "episode:s1", "candidate_parent_id": "episode:s0"}
        )
        self.assert_invalid(transition, "failed_transition_changed_candidate_state")

    def test_commit_requires_current_candidate_id(self) -> None:
        transition = {
            "parent_state_summary": {
                "episode_id": "episode",
                "active_state_id": "episode:s0",
                "candidate_state_id": None,
                "has_verified_candidate": True,
            },
            "action": {
                "tool": "commit_state",
                "arguments": {"candidate_state_id": "episode:s1"},
            },
            "tool_output": {
                "execution_status": "success",
                "active_state_id": "episode:s1",
                "candidate_state_id": None,
            },
            "candidate_state_summary": {
                "episode_id": "episode",
                "active_state_id": "episode:s1",
                "candidate_state_id": None,
            },
        }
        self.assert_invalid(transition, "commit_without_candidate_state")


class VerifierCorrectionSchemaHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = RuleBasedVerifier()

    def assert_invalid_payload(self, tool: str, arguments: dict, reason: str) -> None:
        decision = self.verifier.verify(
            correction_transition(tool=tool, arguments=copy.deepcopy(arguments))
        )
        self.assertFalse(decision["process_valid"])
        self.assertIn(reason, decision["rationale_codes"])

    def test_empty_measurement_correction_is_rejected(self) -> None:
        self.assert_invalid_payload(
            "correct_measurements",
            {"state_id": "episode:s0"},
            "empty_correction_payload",
        )

    def test_empty_nested_modification_is_rejected(self) -> None:
        self.assert_invalid_payload(
            "correct_measurements",
            {"state_id": "episode:s0", "modification": {}},
            "empty_correction_payload",
        )

    def test_invalid_measurement_update_container_is_rejected(self) -> None:
        self.assert_invalid_payload(
            "correct_measurements",
            {"state_id": "episode:s0", "measurement_updates": "not-a-container"},
            "empty_correction_payload",
        )

    def test_whole_measurement_vector_replacement_is_rejected(self) -> None:
        self.assert_invalid_payload(
            "correct_measurements",
            {"state_id": "episode:s0", "measurements": [1.0, 2.0]},
            "empty_correction_payload",
        )

    def test_malformed_nested_modification_is_rejected(self) -> None:
        self.assert_invalid_payload(
            "correct_measurements",
            {"state_id": "episode:s0", "modification": "not-a-mapping"},
            "invalid_correction_payload",
        )

    def test_malformed_case_updates_are_rejected(self) -> None:
        self.assert_invalid_payload(
            "correct_parameters",
            {"state_id": "episode:s0", "case_updates": None},
            "invalid_correction_payload",
        )

    def test_parameter_correction_requires_target_and_value(self) -> None:
        self.assert_invalid_payload(
            "correct_parameters",
            {"state_id": "episode:s0", "line_index": 0},
            "empty_correction_payload",
        )

    def test_topology_correction_requires_target_and_status(self) -> None:
        self.assert_invalid_payload(
            "correct_topology",
            {"state_id": "episode:s0", "branch_id": "b0"},
            "empty_correction_payload",
        )


class VerifierHistoryBoundaryTests(unittest.TestCase):
    def test_privileged_history_disposition_does_not_change_verifier_input(self) -> None:
        base = correction_transition()
        observable_history = {
            "action": {
                "tool": "run_wls",
                "arguments": {"state_id": "episode:s0"},
            },
            "tool_output": {"execution_status": "success"},
        }
        rejected = copy.deepcopy(base)
        rejected.pop("history_summary", None)
        rejected["history_window"] = [
            {
                **copy.deepcopy(observable_history),
                "transition_label": {"candidate_disposition": "REJECT"},
            }
        ]
        accepted = copy.deepcopy(base)
        accepted.pop("history_summary", None)
        accepted["history_window"] = [
            {
                **copy.deepcopy(observable_history),
                "transition_label": {"candidate_disposition": "ACCEPT_FINAL"},
            }
        ]

        rejected_example = build_verifier_example(rejected)
        accepted_example = build_verifier_example(accepted)
        self.assertEqual(
            rejected_example["history_summary"], accepted_example["history_summary"]
        )
        self.assertEqual(rejected_example["features"], accepted_example["features"])

    def test_disposition_counts_use_observable_successful_actions(self) -> None:
        summary = summarize_history(
            [
                {
                    "action": {"tool": "commit_state", "arguments": {}},
                    "tool_output": {"execution_status": "success"},
                    "transition_label": {"candidate_disposition": "REJECT"},
                },
                {
                    "action": {"tool": "rollback_state", "arguments": {}},
                    "tool_output": {"execution_status": "success"},
                    "transition_label": {"candidate_disposition": "ACCEPT_FINAL"},
                },
            ]
        )
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["rejected_count"], 1)


if __name__ == "__main__":
    unittest.main()
