"""Item 7 step 7: the teacher that compares the impedance edit and the breaker flip where they compete."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    COMPARE_ROUTES,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    ROUTE_COMPARISON_FAMILY,
    hypothesis_confidence_vector,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.expert_variants import EXPERT_VARIANTS, expert_variant_options, variant_environment_options
from psse_env.oracle.process_validity import ProcessValidityOracle


def _observation(vector, *, topology=None, parameter=None, comparison=None, rejected=()) -> dict:
    contexts = {"wls": {"successful": True, "anomalous": True, "state_id": "s0", "state_hash": "h", "chi_square_alarm": True,
                        "hypothesis_confidence": vector}}
    observation = {"active_state_id": "s0", "evidence_profile": "scada_only", "research_tools": [COMPARE_ROUTES],
                   "fresh_context_evidence": contexts, "rejected_hypotheses": list(rejected), "remaining_budget": 30}
    for family, lines in (("topology", topology), ("parameter", parameter)):
        if lines is None:
            continue
        tool = CORRECT_TOPOLOGY if family == "topology" else CORRECT_PARAMETERS
        extra = {"cb_name": "CB_6", "status": 0} if family == "topology" else {}
        contexts[family] = {"state_id": "s0", "state_hash": "h", "route_status": "actionable",
                            "supported_corrections": [{"tool": tool, "arguments": {"state_id": "s0", "line_index": line, **extra}} for line in lines]}
        observation[f"has_fresh_{family}_context"] = True
        observation[f"{family}_context_state_id"] = "s0"
    if comparison is not None:
        contexts[ROUTE_COMPARISON_FAMILY] = comparison
    return observation


def _proposal(tool, **arguments):
    return ExpertActionProposal(action={"tool": tool, "arguments": {"state_id": "s0", **arguments}}, source_expert="parameter_expert",
                                confidence=0.9, evidence_codes=[], admissible=True)


class RouteComparisonTeacherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expert = ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True),
                                         **expert_variant_options("baseline_compare_guarded_routes"))
        self.competing = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.5, "line_index": 6}, parameter_line=6)
        self.clear = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.1, "line_index": 6}, parameter_line=6)

    def first(self, observation, ranked=None):
        ranked = ranked or [_proposal(GET_PARAMETER_CONTEXT)]
        proposal = self.expert._route_comparison_first(ranked, observation, seen_signatures=set(), blocked_correction_tools=set())
        return None if proposal is None else proposal.action

    def test_the_variant_and_its_environment(self) -> None:
        self.assertIn("baseline_compare_guarded_routes", EXPERT_VARIANTS)
        self.assertTrue(self.expert.route_comparison)
        self.assertEqual(variant_environment_options("baseline_compare_guarded_routes"), {"route_comparison": True})
        for name in EXPERT_VARIANTS:
            if name != "baseline_compare_guarded_routes":
                self.assertNotIn("route_comparison", expert_variant_options(name))

    def test_the_breaker_context_comes_first_then_the_parameter_context_then_the_comparison(self) -> None:
        self.assertIsNone(self.first(_observation(self.clear)))
        self.assertEqual(self.first(_observation(self.competing))["tool"], GET_TOPOLOGY_CONTEXT)
        self.assertEqual(self.first(_observation(self.competing, topology=[6]))["tool"], GET_PARAMETER_CONTEXT)
        action = self.first(_observation(self.competing, topology=[6], parameter=[6, 3]))
        self.assertEqual(action, {"tool": COMPARE_ROUTES, "arguments": {"state_id": "s0", "parameter_line": 6, "topology_line": 6}})
        # Nothing to compare when the breaker context offers no flip on the line, or a route was already tried here.
        self.assertIsNone(self.first(_observation(self.competing, topology=[4], parameter=[6])))
        tried = {"candidate_parent_id": "s0", "source_action": {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": "s0", "line_index": 6}}}
        self.assertIsNone(self.first(_observation(self.competing, topology=[6], parameter=[6], rejected=[tried])))
        # Without the tool in the environment, nothing.
        self.assertIsNone(self.first({**_observation(self.competing, topology=[6], parameter=[6]), "research_tools": None}))

    def test_after_the_comparison_the_best_verified_route_leads(self) -> None:
        def record(best):
            other = "parameter" if best == "topology" else "topology"
            correction = {"topology": {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": "s0", "cb_name": "CB_6", "status": 0, "line_index": 6}},
                          "parameter": {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 6}}}
            return {"state_id": "s0", "parameter_line": 6, "topology_line": 6, "best_route": best, "candidates": [
                {"route": best, "correction": correction[best], "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL"},
                {"route": other, "correction": correction[other], "execution_status": "success", "verification_verdict": "REJECT"}]}
        parameter_wins = _observation(self.competing, topology=[6], parameter=[6], comparison=record("parameter"))
        topology_wins = _observation(self.competing, topology=[6], parameter=[6], comparison=record("topology"))
        self.assertFalse(self.expert._topology_first_for(parameter_wins))
        self.assertTrue(self.expert._topology_first_for(topology_wins))
        self.assertIsNone(self.first(topology_wins))
        ranked = [_proposal(CORRECT_PARAMETERS, line_index=6), _proposal(CORRECT_TOPOLOGY, cb_name="CB_6", status=0, line_index=6)]
        reordered = self.expert._route_comparison_order(ranked, topology_wins)
        self.assertEqual([p.action["tool"] for p in reordered], [CORRECT_TOPOLOGY, CORRECT_PARAMETERS])


if __name__ == "__main__":
    unittest.main()
