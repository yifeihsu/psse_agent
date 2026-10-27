"""Tests of the decision audit's pure parts: signatures, decision states, bins, features and the scripted policy."""

from __future__ import annotations

import unittest

from research.rl.decision_audit import (
    ScriptedThenExpertPolicy,
    action_signature,
    alternative_features,
    better,
    classify_state,
    distinct_actions,
    is_decision_state,
    root_bin,
    truth_correction_actions,
)


def _outcome(success: bool, steps: int = 10, false_commits: int = 0, healthy: bool = True) -> dict:
    return {"success": success, "steps": steps, "false_commits": false_commits, "healthy_preserved": healthy}


class DecisionAuditTests(unittest.TestCase):
    def test_signatures_ignore_state_ids_and_distinct_actions_dedupe(self) -> None:
        a = {"tool": "correct_parameters", "arguments": {"state_id": "s0", "line_index": 2}}
        b = {"tool": "correct_parameters", "arguments": {"state_id": "s9", "line_index": 2}}
        c = {"tool": "correct_parameters", "arguments": {"state_id": "s0", "line_index": 3}}
        self.assertEqual(action_signature(a), action_signature(b))
        self.assertNotEqual(action_signature(a), action_signature(c))
        self.assertEqual(len(distinct_actions([a, b, c])), 2)

    def test_decision_states_need_a_rule_choice_among_two_actions(self) -> None:
        menu = [{"tool": "correct_parameters", "arguments": {"line_index": 1}}, {"tool": "correct_parameters", "arguments": {"line_index": 2}}]
        self.assertTrue(is_decision_state(menu, menu[0], selection_basis="rule_expert"))
        self.assertFalse(is_decision_state(menu, menu[0], selection_basis="candidate_disposition_reconstruction"))
        self.assertFalse(is_decision_state(menu[:1], menu[0], selection_basis="rule_expert"))
        self.assertFalse(is_decision_state(menu, {"tool": "commit_state", "arguments": {}}, selection_basis="rule_expert"))
        self.assertFalse(is_decision_state(menu, {"tool": "run_wls", "arguments": {}}, selection_basis="rule_expert"))

    def test_better_prefers_success_then_a_clear_step_saving_without_new_harm(self) -> None:
        self.assertTrue(better(_outcome(True, 12), _outcome(False, 6)))
        self.assertFalse(better(_outcome(False, 3), _outcome(True, 12)))
        self.assertTrue(better(_outcome(True, 6), _outcome(True, 9)))
        self.assertFalse(better(_outcome(True, 8), _outcome(True, 9)))  # one step is noise
        self.assertFalse(better(_outcome(True, 6, false_commits=1), _outcome(True, 9)))
        self.assertFalse(better(_outcome(True, 6, healthy=False), _outcome(True, 9)))

    def test_bins_follow_the_plan(self) -> None:
        fail, win = _outcome(False, 40), _outcome(True, 8)
        self.assertEqual(classify_state(fail, [{"outcome": win}], {"outcome": win}), "legal_alternative")
        self.assertEqual(classify_state(win, [{"outcome": _outcome(True, 9)}], None), "rule_best")
        self.assertEqual(classify_state(win, [{"outcome": _outcome(True, 5)}], None), "legal_alternative")
        self.assertEqual(classify_state(fail, [{"outcome": fail}], {"outcome": win}), "interface_limit")
        self.assertEqual(classify_state(fail, [{"outcome": fail}], {"outcome": fail}), "evidence_limit")
        self.assertEqual(classify_state(fail, [], None), "evidence_limit")
        self.assertEqual(root_bin(["rule_best", "interface_limit"], True), "interface_limit")
        self.assertEqual(root_bin(["rule_best"], True), "rule_best")
        self.assertEqual(root_bin([], False), "no_decision")
        self.assertEqual(root_bin(["evidence_limit", "legal_alternative"], False), "legal_alternative")

    def test_features_read_the_fresh_context_as_the_policy_sees_it(self) -> None:
        observation = {
            "remaining_budget": 30, "accepted_corrections": [{}], "rejected_hypotheses": [], "tried_action_signatures": ["x"],
            "remaining_anomaly_score": 12.5, "has_open_candidate": False, "available_evidence": ["wls"],
            "fresh_context_evidence": {
                "parameter": {"parameter_ranking_dominance_ratio": 1.02,
                              "parameter_ranking_distinct_lines": [{"line_index1": 2, "abs_lambda": 10.1}, {"line_index1": 1, "abs_lambda": 9.9}]},
                "wls": {"chi_square_statistic": 150.0, "chi_square_threshold": 60.0},
            },
        }
        features = alternative_features(observation, {"tool": "correct_parameters", "arguments": {"line_index": 1}}, rank=1, step=2)
        self.assertEqual((features["rank"], features["step"], features["tool"], features["line_index"]), (1, 2, "correct_parameters", 1))
        self.assertEqual(features["accepted_corrections"], 1)
        self.assertAlmostEqual(features["parameter_dominance_ratio"], 1.02)
        self.assertEqual(features["parameter_candidates"][0]["line_index1"], 2)
        self.assertEqual(features["wls"]["chi_square_statistic"], 150.0)

    def test_truth_corrections_name_every_target_at_once(self) -> None:
        root = {"audit": {"truth": {
            "true_measurement_errors": [{"index": 4}, {"index0": 9}],
            "true_parameter_errors": [{"branch_row0": 6}],
        }}}
        actions = truth_correction_actions(root, "s0")
        self.assertEqual(actions[0]["tool"], "correct_measurements")
        self.assertEqual(actions[0]["arguments"]["suspect_group"], [4, 9])
        self.assertEqual(actions[1], {"tool": "correct_parameters", "arguments": {"state_id": "s0", "line_index": 7}})
        self.assertEqual(truth_correction_actions({"audit": {"truth": {}}}, "s0"), [])

    def test_breaker_truths_are_probed_by_breaker_name(self) -> None:
        root = {"audit": {"truth": {
            "true_parameter_errors": [{"branch_row0": 2}],
            "true_topology_errors": [
                {"cb_name": "CB_4_N1_N2", "true_cb_closed": False, "physical_effect": "bus_split"},
                {"cb_name": "CB_13R1_13R2", "true_cb_closed": False, "branch_row0": 19, "line_index1": 20,
                 "physical_effect": "dangling_line_terminal"},
            ],
        }}}
        actions = truth_correction_actions(root, "s0")
        self.assertEqual([a["tool"] for a in actions], ["correct_parameters", "correct_topology", "correct_topology"])
        self.assertEqual(actions[1]["arguments"], {"state_id": "s0", "cb_name": "CB_13R1_13R2", "status": 0, "line_index": 20})
        self.assertEqual(actions[2]["arguments"], {"state_id": "s0", "cb_name": "CB_4_N1_N2", "status": 0})
        # A line-level truth without a breaker keeps the branch-status probe.
        line_level = truth_correction_actions({"audit": {"truth": {"true_topology_errors": [{"branch_row0": 5}]}}}, "s0")
        self.assertEqual(line_level, [{"tool": "correct_topology", "arguments": {"state_id": "s0", "branch_row0": 5}}])

    def test_the_scripted_policy_replays_then_defers_to_the_expert_with_live_state_ids(self) -> None:
        class Expert:
            def act(self, observation):
                return {"tool": "ask_for_more_evidence", "arguments": {"state_id": observation["active_state_id"]}}

        prefix = [{"tool": "run_wls", "arguments": {"state_id": "old:s0"}}]
        alternative = [{"tool": "correct_parameters", "arguments": {"state_id": "old:s0", "line_index": 2}}]
        policy = ScriptedThenExpertPolicy(prefix, alternative, Expert())
        first = policy.act({"active_state_id": "new:s0", "candidate_state_id": None, "has_unverified_candidate": False})
        self.assertEqual(first, {"tool": "run_wls", "arguments": {"state_id": "new:s0"}})
        second = policy.act({"active_state_id": "new:s0", "candidate_state_id": None, "has_unverified_candidate": False})
        self.assertEqual(second["arguments"], {"state_id": "new:s0", "line_index": 2})
        third = policy.act({"active_state_id": "new:s0", "candidate_state_id": "new:c1", "has_unverified_candidate": True})
        self.assertEqual(third["tool"], "ask_for_more_evidence")
        # A recorded verification of a candidate rebinds to the live candidate id.
        verify = ScriptedThenExpertPolicy([{"tool": "run_wls", "arguments": {"state_id": "old:c1"}}], [], Expert())
        bound = verify.act({"active_state_id": "new:s0", "candidate_state_id": "new:c1", "has_unverified_candidate": True})
        self.assertEqual(bound["arguments"]["state_id"], "new:c1")


    def test_the_truth_guided_policy_applies_corrections_only_on_clear_states(self) -> None:
        from research.rl.decision_audit import TruthGuidedPolicy

        class Expert:
            def act(self, observation):
                return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}

        truth = [{"tool": "correct_measurements", "arguments": {"state_id": "old", "suspect_group": [4, 9]}},
                 {"tool": "correct_parameters", "arguments": {"state_id": "old", "line_index": 7}}]
        policy = TruthGuidedPolicy([{"tool": "run_wls", "arguments": {"state_id": "old"}}], truth, Expert())
        fresh = {"has_fresh_measurement_context": True, "measurement_context_state_id": "s0",
                 "has_fresh_parameter_context": True, "parameter_context_state_id": "s0"}
        clear = {"active_state_id": "s0", "candidate_state_id": None, "has_open_candidate": False, "has_unverified_candidate": False, **fresh}
        open_candidate = {"active_state_id": "s0", "candidate_state_id": "c1", "has_open_candidate": True, "has_unverified_candidate": True, **fresh}
        self.assertEqual(policy.act(clear)["tool"], "run_wls")  # the prefix
        first = policy.act(clear)
        self.assertEqual((first["tool"], first["arguments"]["suspect_group"], first["arguments"]["state_id"]), ("correct_measurements", [4, 9], "s0"))
        self.assertEqual(policy.act(open_candidate)["tool"], "run_wls")  # the expert verifies the open candidate
        second = policy.act(clear)
        self.assertEqual((second["tool"], second["arguments"]["line_index"]), ("correct_parameters", 7))
        self.assertEqual(policy.act(clear)["tool"], "run_wls")  # nothing pending: the expert
        self.assertEqual(len(policy.applied), 2)
        self.assertEqual(policy.pending, [])

    def test_the_truth_guided_policy_fetches_each_family_context_first_and_retries_a_refused_group_as_singles(self) -> None:
        from research.rl.decision_audit import TruthGuidedPolicy

        class Expert:
            def act(self, observation):
                return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}

        truth = [{"tool": "correct_topology", "arguments": {"state_id": "old", "cb_name": "CB_4_N1_N2", "status": 0}},
                 {"tool": "correct_measurements", "arguments": {"state_id": "old", "suspect_group": [4, 9]}}]
        policy = TruthGuidedPolicy([], truth, Expert())
        clear = {"active_state_id": "s0", "candidate_state_id": None, "has_open_candidate": False, "has_unverified_candidate": False}
        # No fresh topology context on s0: the probe asks for it, then issues the breaker correction.
        self.assertEqual(policy.act(clear), {"tool": "get_topology_context", "arguments": {"state_id": "s0"}})
        breaker = policy.act({**clear, "has_fresh_topology_context": True, "topology_context_state_id": "s0"})
        self.assertEqual((breaker["tool"], breaker["arguments"]["cb_name"]), ("correct_topology", "CB_4_N1_N2"))
        # A context already requested on this state is not requested twice even if it is still not fresh.
        after_commit = {**clear, "active_state_id": "s1", "last_tool": "commit_state", "last_tool_output": {}}
        self.assertEqual(policy.act(after_commit), {"tool": "get_measurement_context", "arguments": {"state_id": "s1"}})
        group = policy.act({**after_commit, "last_tool": "get_measurement_context"})
        self.assertEqual(group["arguments"]["suspect_group"], [4, 9])
        # The group is refused as unsupported: the probe retries the single meters, context first.
        refused = {**after_commit, "last_tool": "correct_measurements",
                   "last_tool_output": {"execution_status": "failure", "error_code": "correction_not_supported_by_current_context"},
                   "has_fresh_measurement_context": True, "measurement_context_state_id": "s1"}
        single = policy.act(refused)
        self.assertEqual((single["tool"], single["arguments"]["suspect_group"]), ("correct_measurements", [4]))
        self.assertEqual(policy.pending[0]["arguments"]["suspect_group"], [9])
        self.assertEqual(len(policy.applied), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
