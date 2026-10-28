"""Item 7 step 6: the confidence over meter, parameter and topology, and the teacher that walks its ranking."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    AMBIGUOUS_HYPOTHESES_STATEMENT,
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_PARAMETER_CONTEXT,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    REVERT_CONTRACT,
    REVERT_STATE,
    REVERTED_REJECTION_KIND,
    abstention_record,
    ambiguity_statements,
    competing_hypotheses,
    hypothesis_confidence_vector,
    hypothesis_ranking,
    withdrawn_families,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.expert_variants import (
    EXPERT_VARIANTS,
    REVERT_FRUITLESS_CYCLES,
    expert_variant_options,
    variant_environment_options,
    variant_provider_options,
)
from psse_env.oracle.parameter_expert import ParameterExpert
from psse_env.providers.test_route_confidence import _solved
from psse_env.providers.matpower import MatpowerDeploymentProviders


class VectorTests(unittest.TestCase):
    def test_the_meter_share_is_the_lagrangian_test_and_the_branch_share_splits_by_the_open_terminal(self) -> None:
        vector = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.6, "line_index": 6}, measurement_index=4, parameter_line=3)
        self.assertEqual((vector["measurement"], vector["parameter"], vector["topology"]), (0.25, 0.3, 0.45))
        self.assertEqual(vector["ranking"], ["topology", "parameter", "measurement"])
        self.assertEqual((vector["topology_line"], vector["parameter_line"], vector["measurement_index"]), (6, 3, 4))
        self.assertIsNone(hypothesis_confidence_vector(0.0, 0.0, None))

    def test_the_ranking_uses_the_bands_the_routes_already_use(self) -> None:
        # The meter leads only outside the 1.2 dominance band; inside it the branch block leads.
        self.assertEqual(hypothesis_ranking(13.0, 10.0, 0.0), ["measurement", "parameter", "topology"])
        self.assertEqual(hypothesis_ranking(11.0, 10.0, 0.0), ["parameter", "topology", "measurement"])
        self.assertEqual(hypothesis_ranking(11.0, 10.0, 0.6)[0], "topology")
        self.assertEqual(hypothesis_ranking(10.0, 30.0, 0.5)[0], "parameter")  # 0.5 is not above 0.5

    def test_competitors_are_what_the_ordering_rule_cannot_separate(self) -> None:
        self.assertEqual(competing_hypotheses(13.0, 10.0, 0.5), [])  # a clear meter leads alone
        self.assertEqual(competing_hypotheses(10.0, 30.0, 0.1), [])  # a clear impedance error
        self.assertEqual(competing_hypotheses(10.0, 30.0, 0.6), ["parameter"])  # open terminal or impedance
        self.assertEqual(competing_hypotheses(10.0, 11.0, 0.1), ["measurement"])  # inside the dominance band
        self.assertEqual(competing_hypotheses(10.0, 11.0, 0.3), ["topology", "measurement"])

    def test_the_provider_reads_the_vector_off_its_own_solve(self) -> None:
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only", flow_at_noise_cue=True)
        solved = _solved({4: {"p_meas": 1.0, "p_resid": 20.0}}, {4: 30.0, 7: 10.0})
        route = providers._route_confidence(solved)
        vector = providers._hypothesis_confidence(solved, route)
        self.assertEqual((vector["residual_max"], vector["multiplier_max"]), (20.0, 30.0))
        self.assertEqual(vector["ranking"][0], "topology")
        self.assertEqual((vector["topology_line"], vector["parameter_line"]), (5, 5))
        refined = providers._breaker_refined_confidences(
            solved, [{"cb_name": "CB_5", "line_index": 5, "score": 9.0, "gse_progress_after_flip": 0.4}])
        self.assertTrue(refined["hypothesis_confidence"]["breaker_refined"])


def _observation(*, vector=None, refined=None, anomalous=True, rejected=(), accepted=(), ledger=None, budget=30) -> dict:
    wls = {"successful": True, "anomalous": anomalous, "state_id": "s2", "state_hash": "h", "chi_square_alarm": anomalous}
    if vector is not None:
        wls["hypothesis_confidence"] = vector
    contexts = {"wls": wls}
    if refined is not None:
        contexts["topology"] = {"state_id": "s2", "hypothesis_confidence": refined}
    observation = {"active_state_id": "s2", "fresh_context_evidence": contexts, "remaining_budget": budget,
                   "evidence_profile": "scada_only",
                   "rejected_hypotheses": list(rejected), "accepted_corrections": list(accepted)}
    if ledger is not None:
        observation["revert_ledger"] = ledger
    return observation


def _reverted(tool: str, line: int | None = 6, parent: str = "s2") -> dict:
    arguments = {"state_id": parent, **({"line_index": line} if line is not None else {"suspect_group": [4]})}
    return {"candidate_parent_id": parent, "rejection_kind": REVERTED_REJECTION_KIND, "source_action": {"tool": tool, "arguments": arguments}}


class StatementTests(unittest.TestCase):
    def test_the_statement_lists_the_competing_hypotheses_with_their_confidences(self) -> None:
        vector = hypothesis_confidence_vector(10.0, 11.0, {"topology": 0.3, "line_index": 6}, measurement_index=4, parameter_line=3)
        (statement,) = ambiguity_statements(_observation(vector=vector, rejected=[_reverted(CORRECT_TOPOLOGY)]))
        self.assertEqual(statement["kind"], AMBIGUOUS_HYPOTHESES_STATEMENT)
        self.assertEqual([h["family"] for h in statement["hypotheses"]], ["parameter", "topology", "measurement"])
        self.assertEqual([h.get("line_index", h.get("measurement_index")) for h in statement["hypotheses"]], [3, 6, 4])
        self.assertEqual((statement["attempted"], statement["withdrawn"]), (["topology"], ["topology"]))
        record = abstention_record([statement], _observation(vector=vector))
        self.assertTrue(record[0]["supported"])
        clear = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.1, "line_index": 6})
        self.assertEqual(ambiguity_statements(_observation(vector=clear)), [])
        self.assertFalse(abstention_record([statement], _observation(vector=clear))[0]["supported"])
        self.assertEqual(ambiguity_statements(_observation(vector=vector, anomalous=False)), [])

    def test_the_topology_context_refines_the_vector(self) -> None:
        vector = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.3, "line_index": 6})
        refined = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.9, "line_index": 6, "source": "node_breaker_context"})
        self.assertEqual(ambiguity_statements(_observation(vector=vector, refined=refined)), [])


def _ledger(*entries, threshold: int = REVERT_FRUITLESS_CYCLES) -> dict:
    return {"contract": REVERT_CONTRACT, "fruitless_cycle_threshold": threshold, "lineage": list(entries),
            "reverted_corrections": [], "reverts": []}


def _entry(state: str, committed: str, tool: str, cycles: int, *, line: int = 6, reverted: bool = False) -> dict:
    return {"state_id": state, "committed_state_id": committed, "fruitless_cycles_since": cycles, "reverted": reverted,
            "source_action": {"tool": tool, "arguments": {"state_id": state, "line_index": line}}}


HANDOFF = [ExpertActionProposal(action={"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": "s2", "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST}},
                                source_expert="recovery_expert", confidence=1.0, evidence_codes=[], admissible=True)]


class WalkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expert = ExpertPolicyOracle(**expert_variant_options("baseline_compare_guarded_walk"))

    def test_the_variant_is_named_and_carries_its_environment(self) -> None:
        self.assertIn("baseline_compare_guarded_walk", EXPERT_VARIANTS)
        self.assertTrue(self.expert.hypothesis_walk)
        self.assertEqual(variant_provider_options("baseline_compare_guarded_walk"), {"flow_at_noise_cue": True})
        self.assertEqual(variant_environment_options("baseline_compare_guarded_walk"), {"revert_fruitless_cycles": 3})
        for name in EXPERT_VARIANTS:
            if name != "baseline_compare_guarded_walk":
                self.assertNotIn("hypothesis_walk", expert_variant_options(name))
                self.assertEqual(variant_environment_options(name), {})

    def test_the_route_order_follows_the_ranking_and_a_withdrawn_family_goes_behind(self) -> None:
        topology_first = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.7, "line_index": 6})
        parameter_first = hypothesis_confidence_vector(10.0, 30.0, {"topology": 0.2, "line_index": 6})
        self.assertTrue(self.expert._topology_first_for(_observation(vector=topology_first)))
        self.assertFalse(self.expert._topology_first_for(_observation(vector=parameter_first)))
        # The breaker context's refinement counts.
        self.assertTrue(self.expert._topology_first_for(_observation(vector=parameter_first, refined=topology_first)))
        # A family withdrawn on this state goes behind the other one.
        self.assertFalse(self.expert._topology_first_for(_observation(vector=topology_first, rejected=[_reverted(CORRECT_TOPOLOGY)])))
        self.assertTrue(self.expert._topology_first_for(_observation(vector=parameter_first, rejected=[_reverted(CORRECT_PARAMETERS)])))
        self.assertEqual(withdrawn_families(_observation(rejected=[_reverted(CORRECT_TOPOLOGY, parent="s1")])), set())

    def test_a_fruitless_hypothesis_is_reverted_to_its_first_commit_instead_of_handed_off(self) -> None:
        ledger = _ledger(_entry("s0", "s1", CORRECT_MEASUREMENTS, 4), _entry("s1", "s2", CORRECT_TOPOLOGY, 3))
        (revert,) = self.expert._hypothesis_switch_proposals(_observation(ledger=ledger), HANDOFF)
        self.assertEqual(revert.action, {"tool": REVERT_STATE, "arguments": {"state_id": "s2", "to_state_id": "s1"}})
        # Two commits of one hypothesis go back to the first.
        ledger = _ledger(_entry("s0", "s1", CORRECT_TOPOLOGY, 4), _entry("s1", "s2", CORRECT_TOPOLOGY, 3, line=9))
        (revert,) = self.expert._hypothesis_switch_proposals(_observation(ledger=ledger), HANDOFF)
        self.assertEqual(revert.action["arguments"]["to_state_id"], "s0")

    def test_no_switch_without_the_gate_the_budget_or_a_handoff(self) -> None:
        few = _ledger(_entry("s1", "s2", CORRECT_TOPOLOGY, 2))
        self.assertEqual(self.expert._hypothesis_switch_proposals(_observation(ledger=few), HANDOFF), [])
        done = _ledger(_entry("s1", "s2", CORRECT_TOPOLOGY, 5, reverted=True))
        self.assertEqual(self.expert._hypothesis_switch_proposals(_observation(ledger=done), HANDOFF), [])
        ready = _ledger(_entry("s1", "s2", CORRECT_TOPOLOGY, 3))
        self.assertEqual(self.expert._hypothesis_switch_proposals(_observation(ledger=ready, budget=4), HANDOFF), [])
        self.assertEqual(self.expert._hypothesis_switch_proposals(_observation(ledger=ready, anomalous=False), HANDOFF), [])
        self.assertEqual(self.expert._hypothesis_switch_proposals(_observation(), HANDOFF), [])
        self.assertEqual(self.expert._hypothesis_switch_proposals(_observation(ledger=ready), []), [])

    def test_a_withdrawn_meter_correction_hands_the_lead_to_the_branch_families(self) -> None:
        meter_dominant = {**_observation(), "unresolved_signatures": [
            "wls_residual_outlier_dominant index=4 channel=Pf", "wls_branch_multiplier line_status_or_parameter line=6"]}
        proposed = lambda observation: [p.action["tool"] for p in ParameterExpert().propose(observation, [])]
        self.assertNotIn(GET_PARAMETER_CONTEXT, proposed(meter_dominant))
        self.assertIn(GET_PARAMETER_CONTEXT, proposed({**meter_dominant, "rejected_hypotheses": [_reverted(CORRECT_MEASUREMENTS, None)]}))


if __name__ == "__main__":
    unittest.main()
