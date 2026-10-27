"""The topology-first structural order (compound-error study, 2026-10-27) and its variant."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    COMPARE_PARAMETER_CANDIDATES,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle, _structural_first_order
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.expert_variants import EXPERT_VARIANTS, expert_variant_options


def _proposal(tool: str, **arguments) -> ExpertActionProposal:
    return ExpertActionProposal(
        action={"tool": tool, "arguments": {"state_id": "s0", **arguments}},
        source_expert="test", confidence=0.9, evidence_codes=[], admissible=True, estimated_immediate_risk=0.1,
    )


def _tools(proposals) -> list[str]:
    return [proposal.action["tool"] for proposal in proposals]


class TopologyFirstOrderTests(unittest.TestCase):
    def test_default_order_is_unchanged(self) -> None:
        ranked = [
            _proposal(CORRECT_PARAMETERS, line_index=3),
            _proposal(CORRECT_MEASUREMENTS, suspect_group=[4]),
            _proposal(CORRECT_TOPOLOGY, cb_name="CB_4_N1_N2", status=0),
            _proposal(GET_PARAMETER_CONTEXT),
            _proposal(GET_TOPOLOGY_CONTEXT),
            _proposal(GET_MEASUREMENT_CONTEXT),
        ]
        self.assertEqual(
            _tools(_structural_first_order(ranked)),
            [CORRECT_PARAMETERS, CORRECT_TOPOLOGY, GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT,
             CORRECT_MEASUREMENTS, GET_MEASUREMENT_CONTEXT],
        )
        self.assertEqual(_tools(_structural_first_order(ranked, topology_first=False)), _tools(_structural_first_order(ranked)))

    def test_topology_first_leads_the_structural_tier_only(self) -> None:
        ranked = [
            _proposal(COMPARE_PARAMETER_CANDIDATES, line_indices=[3, 6]),
            _proposal(CORRECT_PARAMETERS, line_index=3),
            _proposal(CORRECT_MEASUREMENTS, suspect_group=[4]),
            _proposal(CORRECT_TOPOLOGY, cb_name="CB_4_N1_N2", status=0),
            _proposal(GET_PARAMETER_CONTEXT),
            _proposal(GET_TOPOLOGY_CONTEXT),
            _proposal(GET_MEASUREMENT_CONTEXT),
        ]
        self.assertEqual(
            _tools(_structural_first_order(ranked, topology_first=True)),
            [CORRECT_TOPOLOGY, GET_TOPOLOGY_CONTEXT, COMPARE_PARAMETER_CANDIDATES, CORRECT_PARAMETERS,
             GET_PARAMETER_CONTEXT, CORRECT_MEASUREMENTS, GET_MEASUREMENT_CONTEXT],
        )
        # Without a structural proposal nothing moves, whatever the flag.
        meters = [_proposal(CORRECT_MEASUREMENTS, suspect_group=[4]), _proposal(RUN_WLS)]
        self.assertEqual(_tools(_structural_first_order(meters, topology_first=True)), _tools(meters))

    def test_variant_selects_the_flag_and_the_conventional_teacher_does_not(self) -> None:
        self.assertIn("baseline_compare_topology_first", EXPERT_VARIANTS)
        self.assertEqual(expert_variant_options("baseline_compare"), {"candidate_comparison": True})
        self.assertEqual(expert_variant_options("baseline_compare_topology_first"),
                         {"candidate_comparison": True, "topology_first": True})
        self.assertFalse(ExpertPolicyOracle().topology_first)
        self.assertFalse(ExpertPolicyOracle(**expert_variant_options("baseline_compare")).topology_first)
        self.assertTrue(ExpertPolicyOracle(**expert_variant_options("baseline_compare_topology_first")).topology_first)


if __name__ == "__main__":
    unittest.main()
