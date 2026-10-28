"""Stage 2 item 5 (decision C, 2026-10-28): the explicit isolated-line abstention at handoff, recorded and never required."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    ISOLATED_LINE_ABSTENTION,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    abstention_record,
    isolated_line_abstentions,
    with_isolated_line_abstentions,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.oracle.expert_variants import EXPERT_VARIANTS, expert_variant_options


def _accepted(tool: str, **arguments) -> dict:
    return {"candidate_state_id": "s1", "candidate_parent_id": "s0",
            "source_action": {"tool": tool, "arguments": {"state_id": "s0", **arguments}}}


OBSERVATION = {
    "active_state_id": "s2",
    "accepted_corrections": [
        _accepted("correct_topology", cb_name="CB_9_B1_N3", status=0, line_index=17),   # a dangling terminal
        _accepted("correct_topology", cb_name="CB_4_N2_B2", status=0),                  # a bus split: no line
        _accepted("correct_topology", cb_name="CB_2_B1_L23", status=1, line_index=3),   # a reconnected terminal
        _accepted("correct_parameters", line_index=5),
        _accepted("correct_measurements", suspect_group=[8]),
    ],
}
HANDOFF = {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": "s2", "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST}}


class IsolatedLineAbstentionTests(unittest.TestCase):
    def test_statements_name_the_lines_the_accepted_open_breakers_isolated(self) -> None:
        self.assertEqual(
            isolated_line_abstentions(OBSERVATION),
            [{"kind": ISOLATED_LINE_ABSTENTION, "line_index": 17, "cb_name": "CB_9_B1_N3"}],
        )
        self.assertEqual(isolated_line_abstentions({"accepted_corrections": []}), [])
        self.assertEqual(isolated_line_abstentions({}), [])

    def test_only_an_operator_handoff_carries_them(self) -> None:
        attached = with_isolated_line_abstentions(HANDOFF, OBSERVATION)
        self.assertEqual(attached["arguments"]["abstentions"], isolated_line_abstentions(OBSERVATION))
        self.assertEqual(attached["arguments"]["request"], RECOVERY_OPTIONS_EXHAUSTED_REQUEST)
        self.assertNotIn("abstentions", with_isolated_line_abstentions(HANDOFF, {"accepted_corrections": []})["arguments"])
        other = {"tool": "get_measurement_context", "arguments": {"state_id": "s2"}}
        self.assertEqual(with_isolated_line_abstentions(other, OBSERVATION), other)
        plain_request = {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": "s2", "request": "three_phase"}}
        self.assertNotIn("abstentions", with_isolated_line_abstentions(plain_request, OBSERVATION)["arguments"])

    def test_the_environment_record_marks_support_and_never_fails(self) -> None:
        declared = [
            {"kind": ISOLATED_LINE_ABSTENTION, "line_index": 17, "cb_name": "CB_9_B1_N3"},
            {"kind": ISOLATED_LINE_ABSTENTION, "line_index": 17},
            {"kind": ISOLATED_LINE_ABSTENTION, "line_index": 3, "cb_name": "CB_2_B1_L23"},
            {"kind": "something_else", "line_index": 17},
            "not a statement",
        ]
        record = abstention_record(declared, OBSERVATION)
        self.assertEqual([item["supported"] for item in record], [True, True, False, False, False])
        self.assertTrue(record[-1]["malformed"])
        self.assertEqual(abstention_record(None, OBSERVATION), [])
        self.assertEqual(abstention_record([], OBSERVATION), [])

    def test_the_expert_option_is_off_everywhere_and_attaches_when_on(self) -> None:
        for name in EXPERT_VARIANTS:
            self.assertNotIn("abstain_isolated_lines", expert_variant_options(name))
        self.assertFalse(ExpertPolicyOracle().abstain_isolated_lines)
        self.assertTrue(ExpertPolicyOracle(abstain_isolated_lines=True).abstain_isolated_lines)


if __name__ == "__main__":
    unittest.main()
