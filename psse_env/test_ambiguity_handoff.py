"""Ranked hypothesis testing under parameter-ranking ambiguity.

When no line dominates the parameter ranking, the context offers the top
candidates in rank order, the teacher tests them under verification, and an
exhausted candidate set ends in an operator handoff bounded to those lines.
The audit credits that handoff when the true branch is among them; the
research label rule keeps the deterministic ladder choices at such states.
"""

from __future__ import annotations

import unittest

from psse_env.actions import (
    AMBIGUOUS_BRANCH_CANDIDATES_REQUEST,
    CORRECT_PARAMETERS,
    ambiguous_branch_candidate_lines,
)
from psse_env.dagger.release_audit import _bounded_branch_handoff
from psse_env.dagger.rollout_collector import observable_rank_one_target_proof


def _evidence(active: str = "s0", ambiguous: bool = True, lines=(5, 6)) -> dict:
    return {
        "active_state_id": active,
        "fresh_context_evidence": {
            "parameter": {
                "state_id": "s0",
                "state_hash": "a" * 64,
                "context_tool": "get_parameter_context",
                "context_binding": "direct_context",
                "evidence_source": "deployment_context:wls_lagrange",
                "route_status": "actionable",
                "parameter_ranking_contract": "distinct_line_abs_lambda_dominance_v1",
                "parameter_ranking_ambiguous": ambiguous,
                "parameter_ranking_dominant": not ambiguous,
                "parameter_ranking_candidate_lines": list(lines),
            }
        },
        "rejected_hypotheses": [],
    }


def _rejection(line: int, parent: str = "s0") -> dict:
    return {
        "candidate_state_id": f"{parent}:c{line}",
        "candidate_parent_id": parent,
        "source_action": {
            "tool": CORRECT_PARAMETERS,
            "arguments": {"state_id": parent, "line_index": line},
        },
    }


class AmbiguousCandidateEvidenceTests(unittest.TestCase):
    def test_candidates_need_ambiguity_state_binding_and_every_rejection(self) -> None:
        observation = _evidence()
        self.assertIsNone(ambiguous_branch_candidate_lines(observation))
        observation["rejected_hypotheses"] = [_rejection(5)]
        self.assertIsNone(ambiguous_branch_candidate_lines(observation))
        observation["rejected_hypotheses"] = [_rejection(5), _rejection(6)]
        self.assertEqual(ambiguous_branch_candidate_lines(observation), [5, 6])
        # Rejections from an earlier active state do not count.
        stale = _evidence()
        stale["rejected_hypotheses"] = [_rejection(5, parent="s_old"), _rejection(6)]
        self.assertIsNone(ambiguous_branch_candidate_lines(stale))
        # A dominant ranking is never an ambiguous handoff.
        dominant = _evidence(ambiguous=False)
        dominant["rejected_hypotheses"] = [_rejection(5), _rejection(6)]
        self.assertIsNone(ambiguous_branch_candidate_lines(dominant))
        # Evidence bound to another state is stale.
        moved = _evidence(active="s1")
        moved["rejected_hypotheses"] = [_rejection(5, "s1"), _rejection(6, "s1")]
        self.assertIsNone(ambiguous_branch_candidate_lines(moved))


class BoundedHandoffAuditTests(unittest.TestCase):
    @staticmethod
    def _final_state(request: str, lines) -> dict:
        return {
            "last_tool_output": {
                "tool_metrics": {
                    "operator_escalation_audit": {"request": request, "candidate_lines": list(lines)}
                }
            }
        }

    def test_true_branch_within_candidates_is_credited(self) -> None:
        scenario = {"true_parameter_errors": [{"branch_row0": 4, "parameter": "X"}]}
        state = self._final_state(AMBIGUOUS_BRANCH_CANDIDATES_REQUEST, [5, 6])
        verdict = _bounded_branch_handoff(
            scenario, state, ["resolved_episode_has_remaining_true_faults", "final_case_outside_clean_tolerance"]
        )
        self.assertTrue(verdict["passed"], verdict)
        self.assertEqual(verdict["true_lines"], [5])

    def test_wrong_pair_or_modified_healthy_component_is_refused(self) -> None:
        scenario = {"true_parameter_errors": [{"branch_row0": 4, "parameter": "X"}]}
        wrong = _bounded_branch_handoff(
            scenario,
            self._final_state(AMBIGUOUS_BRANCH_CANDIDATES_REQUEST, [7, 8]),
            ["resolved_episode_has_remaining_true_faults"],
        )
        self.assertFalse(wrong["passed"])
        self.assertIn("true_branch_not_among_candidates", wrong["reasons"])
        masked = _bounded_branch_handoff(
            scenario,
            self._final_state(AMBIGUOUS_BRANCH_CANDIDATES_REQUEST, [5, 6]),
            ["resolved_episode_has_remaining_true_faults", "healthy_measurement_modified"],
        )
        self.assertFalse(masked["passed"])
        generic = _bounded_branch_handoff(
            scenario,
            self._final_state("operator_escalation:recovery_options_exhausted", []),
            ["resolved_episode_has_remaining_true_faults"],
        )
        self.assertFalse(generic["passed"])
        too_wide = _bounded_branch_handoff(
            scenario,
            self._final_state(AMBIGUOUS_BRANCH_CANDIDATES_REQUEST, [5, 6, 7]),
            ["resolved_episode_has_remaining_true_faults"],
        )
        self.assertFalse(too_wide["passed"])
        two_faults = _bounded_branch_handoff(
            {
                "true_parameter_errors": [{"branch_row0": 4}],
                "true_topology_errors": [{"branch_row0": 9}],
            },
            self._final_state(AMBIGUOUS_BRANCH_CANDIDATES_REQUEST, [5, 6]),
            ["resolved_episode_has_remaining_true_faults"],
        )
        self.assertFalse(two_faults["passed"])


class RankOneProofTests(unittest.TestCase):
    def test_non_correction_ladder_choice_is_a_deterministic_target(self) -> None:
        first = {"tool": "get_parameter_context", "arguments": {"state_id": "s0"}}
        second = {"tool": "get_topology_context", "arguments": {"state_id": "s0"}}
        proof = observable_rank_one_target_proof(
            {"active_state_id": "s0"}, preferred_action=first, expert_actions=[first, second]
        )
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["basis"], "deterministic_non_correction_target")
        not_first = observable_rank_one_target_proof(
            {"active_state_id": "s0"}, preferred_action=second, expert_actions=[first, second]
        )
        self.assertFalse(not_first["passed"])

    def test_ambiguous_parameter_ranking_still_proves_the_top_candidate(self) -> None:
        actions = [
            {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 5}},
            {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 6}},
        ]
        observation = {
            "active_state_id": "s0",
            "has_fresh_parameter_context": True,
            "parameter_context_state_id": "s0",
            "fresh_context_evidence": {
                "parameter": {
                    "state_id": "s0",
                    "state_hash": "b" * 64,
                    "context_tool": "get_parameter_context",
                    "context_binding": "direct_context",
                    "evidence_source": "deployment_context:wls_lagrange",
                    "route_status": "actionable",
                    "parameter_ranking_contract": "distinct_line_abs_lambda_dominance_v1",
                    "parameter_ranking_distinct_lines": [
                        {"line_index1": 5, "abs_lambda_score": 6.0},
                        {"line_index1": 6, "abs_lambda_score": 5.5},
                    ],
                    "parameter_ranking_top_abs_lambda": 6.0,
                    "parameter_ranking_runner_up_abs_lambda": 5.5,
                    "parameter_ranking_dominance_ratio": 6.0 / 5.5,
                    "parameter_ranking_dominance_threshold": 1.2,
                    "parameter_ranking_dominant": False,
                    "parameter_ranking_ambiguous": True,
                    "supported_corrections": actions,
                }
            },
        }
        proof = observable_rank_one_target_proof(
            observation, preferred_action=actions[0], expert_actions=actions
        )
        self.assertTrue(proof["passed"], proof)
        self.assertEqual(proof["basis"], "ranked_ambiguity_candidate")
        tied = dict(observation)
        tied["fresh_context_evidence"] = {
            "parameter": {
                **observation["fresh_context_evidence"]["parameter"],
                "parameter_ranking_distinct_lines": [
                    {"line_index1": 5, "abs_lambda_score": 6.0},
                    {"line_index1": 6, "abs_lambda_score": 6.0},
                ],
                "parameter_ranking_runner_up_abs_lambda": 6.0,
                "parameter_ranking_dominance_ratio": 1.0,
            }
        }
        self.assertFalse(
            observable_rank_one_target_proof(
                tied, preferred_action=actions[0], expert_actions=actions
            )["passed"]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
