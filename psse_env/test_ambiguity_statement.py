"""Item 7 step 3 (2026-10-28): the ambiguity statement at handoff, recorded and never required."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    AMBIGUOUS_ROUTE_STATEMENT,
    ASK_FOR_MORE_EVIDENCE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    abstention_record,
    ambiguity_statements,
    with_ambiguity_statements,
    with_isolated_line_abstentions,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.oracle.expert_variants import EXPERT_VARIANTS, expert_variant_options


def _observation(topology: float | None, *, anomalous: bool = True, refined: float | None = None, accepted=(), rejected=()) -> dict:
    wls = {"successful": True, "anomalous": anomalous, "state_id": "s2"}
    if topology is not None:
        wls["route_confidence"] = {"line_index": 6, "topology": topology, "parameter": round(1 - topology, 3),
                                   "basis": "flow_at_noise_without_cue"}
    contexts = {"wls": wls}
    if refined is not None:
        contexts["topology"] = {"state_id": "s2", "route_confidence": {
            "line_index": 6, "topology": refined, "parameter": round(1 - refined, 3), "basis": "open_terminal_cue",
            "breaker_evidence": {"cb_name": "CB_X", "score_ratio": refined, "flip_lowers_objective": True}}}
    return {"active_state_id": "s2", "fresh_context_evidence": contexts,
            "accepted_corrections": [{"source_action": a} for a in accepted],
            "rejected_hypotheses": [{"source_action": a} for a in rejected]}


HANDOFF = {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": "s2", "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST}}


class AmbiguityStatementTests(unittest.TestCase):
    def test_stated_only_inside_the_band_while_the_alarm_stands(self) -> None:
        (statement,) = ambiguity_statements(_observation(0.5))
        self.assertEqual((statement["kind"], statement["line_index"], statement["topology_confidence"],
                          statement["parameter_confidence"]), (AMBIGUOUS_ROUTE_STATEMENT, 6, 0.5, 0.5))
        self.assertEqual(ambiguity_statements(_observation(0.9)), [])
        self.assertEqual(ambiguity_statements(_observation(0.1)), [])
        self.assertEqual(ambiguity_statements(_observation(0.5, anomalous=False)), [])
        self.assertEqual(ambiguity_statements(_observation(None)), [])

    def test_the_topology_context_refines_the_confidence_and_attempts_are_listed(self) -> None:
        self.assertEqual(ambiguity_statements(_observation(0.5, refined=0.9)), [])
        (statement,) = ambiguity_statements(_observation(0.4, refined=0.6, rejected=[
            {"tool": "correct_parameters", "arguments": {"state_id": "s1", "line_index": 6}}]))
        self.assertEqual(statement["topology_confidence"], 0.6)
        self.assertEqual(statement["breaker_evidence"]["cb_name"], "CB_X")
        self.assertEqual(statement["attempted"], ["parameter"])

    def test_statements_merge_in_the_handoff_field_and_the_record_marks_support(self) -> None:
        observation = _observation(0.5, accepted=[
            {"tool": "correct_topology", "arguments": {"state_id": "s0", "cb_name": "CB_Y", "status": 0, "line_index": 9}}])
        action = with_ambiguity_statements(with_isolated_line_abstentions(HANDOFF, observation), observation)
        kinds = [item["kind"] for item in action["arguments"]["abstentions"]]
        self.assertEqual(kinds, ["line_isolated_by_breaker", AMBIGUOUS_ROUTE_STATEMENT])
        record = abstention_record(action["arguments"]["abstentions"], observation)
        self.assertEqual([item["supported"] for item in record], [True, True])
        self.assertEqual(record[1]["topology_confidence"], 0.5)
        stale = abstention_record(action["arguments"]["abstentions"], _observation(0.9))
        self.assertFalse(stale[1]["supported"])
        other = {"tool": "get_measurement_context", "arguments": {"state_id": "s2"}}
        self.assertEqual(with_ambiguity_statements(other, observation), other)

    def test_the_expert_option_is_off_everywhere_and_attaches_when_on(self) -> None:
        for name in EXPERT_VARIANTS:
            self.assertNotIn("state_ambiguity", expert_variant_options(name))
        self.assertFalse(ExpertPolicyOracle().state_ambiguity)
        self.assertTrue(ExpertPolicyOracle(state_ambiguity=True).state_ambiguity)


if __name__ == "__main__":
    unittest.main()
