from __future__ import annotations

import copy
import unittest

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.oracle.parameter_expert import ParameterExpert
from psse_env.oracle.topology_expert import TopologyExpert


_BETA025_NOGO_ARCHIVE_SHA256 = (
    "fd9f559e7a3adbeb2e54ceda47ab8e7798cd001aaa754eae2ff20fe34663b53b"
)
_ARCHIVED_WRONG_PARAMETER_TARGET_RANK_COUNTS = {
    1: 18,
    2: 23,
    3: 26,
    4: 4,
}


def _parameter_hints() -> list[dict]:
    return [
        {
            "tool": CORRECT_PARAMETERS,
            "arguments": {"state_id": "episode:s0", "line_index": line1},
        }
        for line1 in (4, 2, 6, 8)
    ]


def _measurement_dominant_state() -> dict:
    return {
        "active_state_id": "episode:s0",
        "last_tool": "get_parameter_context",
        "last_tool_status": "success",
        "last_tool_output": {"execution_status": "success"},
        "remaining_budget": 20,
        "requires_measurement_context": True,
        "unresolved_signatures": [
            "wls_residual_outlier_dominant index=32 channel=Qinj",
            "wls_branch_multiplier line_status_or_parameter line=4",
        ],
        "has_fresh_parameter_context": True,
        "parameter_context_state_id": "episode:s0",
        "fresh_context_evidence": {
            "parameter": {
                "state_id": "episode:s0",
                "route_status": "actionable",
                "supported_corrections": _parameter_hints(),
            }
        },
        "accepted_corrections": [],
        "rejected_hypotheses": [],
        "tried_action_signatures": [],
        "history_window": [],
    }


class MixedFamilyRoutingTests(unittest.TestCase):
    def test_off_policy_parameter_context_cannot_override_measurement_dominance(
        self,
    ) -> None:
        actions = ExpertPolicyOracle().next_actions(
            _measurement_dominant_state(), []
        )

        self.assertEqual(
            actions,
            [
                {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": "episode:s0"},
                }
            ],
        )

    def test_all_71_archived_wrong_line_rank_fixtures_are_suppressed(
        self,
    ) -> None:
        # Compact diagnostic fixtures from the immutable beta=0.25 NO-GO
        # archive.  Every failure shared the same observable cross-family
        # condition; the rank counts retain the full 71-row audit inventory.
        self.assertEqual(len(_BETA025_NOGO_ARCHIVE_SHA256), 64)
        exercised = 0
        for wrong_rank, count in sorted(
            _ARCHIVED_WRONG_PARAMETER_TARGET_RANK_COUNTS.items()
        ):
            for fixture_index in range(count):
                with self.subTest(
                    wrong_rank=wrong_rank,
                    fixture_index=fixture_index,
                ):
                    state = _measurement_dominant_state()
                    hints = _parameter_hints()
                    # Put the archived wrong rank first in the provider's
                    # deferred inventory; cross-family suppression must not
                    # depend on within-parameter ordering.
                    hints = hints[wrong_rank - 1 :] + hints[: wrong_rank - 1]
                    self.assertEqual(
                        ParameterExpert().propose(
                            state,
                            [],
                            oracle_hints=hints,
                        ),
                        [],
                    )
                    exercised += 1
        self.assertEqual(exercised, 71)

    def test_lower_ranked_parameter_hints_remain_suppressed_after_rejection(
        self,
    ) -> None:
        state = _measurement_dominant_state()
        state["rejected_hypotheses"] = [
            {
                "candidate_parent_id": "episode:s0",
                "source_action": _parameter_hints()[0],
                "action_signature": (
                    'correct_parameters:{"line_index":4,"state_id":"episode:s0"}'
                ),
                "verification_summary": {
                    "state_id": "episode:s1",
                    "target_metric_value": 9.0,
                    "target_metric_threshold": 3.0,
                    "target_progress": 0.1,
                    "global_progress": 0.0,
                    "physical_constraints_ok": True,
                },
            }
        ]

        actions = ExpertPolicyOracle().next_actions(state, [])

        self.assertEqual([action["tool"] for action in actions], [GET_MEASUREMENT_CONTEXT])

    def test_branch_dominance_or_explicit_sensor_evidence_keeps_parameter_route(
        self,
    ) -> None:
        for extra_signature in (
            "wls_branch_multiplier_dominant line=4",
            "parameter_sensor_impedance_anomaly line=4",
        ):
            with self.subTest(extra_signature=extra_signature):
                state = _measurement_dominant_state()
                state["unresolved_signatures"].append(extra_signature)
                proposals = ParameterExpert().propose(
                    state,
                    [],
                    oracle_hints=_parameter_hints(),
                )
                self.assertEqual(
                    proposals[0].action["tool"], CORRECT_PARAMETERS
                )

    def test_accepted_partial_measurement_keeps_sequential_parameter_route(
        self,
    ) -> None:
        state = _measurement_dominant_state()
        state["accepted_corrections"] = [
            {
                "candidate_parent_id": "episode:s0",
                "candidate_state_id": "episode:s0",
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [32],
                    },
                },
            }
        ]
        proposals = ParameterExpert().propose(
            state,
            [],
            oracle_hints=_parameter_hints(),
        )

        self.assertEqual(proposals[0].action["tool"], CORRECT_PARAMETERS)

    def test_topology_hints_follow_the_same_cross_family_guard(self) -> None:
        state = _measurement_dominant_state()
        state.update(
            {
                "has_fresh_topology_context": True,
                "topology_context_state_id": "episode:s0",
            }
        )
        hints = [
            {
                "tool": CORRECT_TOPOLOGY,
                "arguments": {
                    "state_id": "episode:s0",
                    "line_index1": 4,
                    "status": 0,
                },
            }
        ]

        self.assertEqual(TopologyExpert().propose(state, [], oracle_hints=hints), [])

        explicit = copy.deepcopy(state)
        explicit["unresolved_signatures"].append(
            "topology_breaker_status_disagreement line=4"
        )
        proposals = TopologyExpert().propose(explicit, [], oracle_hints=hints)
        self.assertEqual(proposals[0].action["tool"], CORRECT_TOPOLOGY)


if __name__ == "__main__":
    unittest.main()
