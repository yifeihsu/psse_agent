from __future__ import annotations

import unittest

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    ROLLBACK_STATE,
    RUN_WLS,
    VERIFY_CANDIDATE,
)
from psse_env.rl.lifecycle import decision_sign, lifecycle_decisions, scenario_truth, summarize_decisions, target_is_true


def _step(tool: str, arguments: dict | None = None, *, status: str = "success", disposition: str | None = None,
          intervention: bool = False, number: int | None = None) -> dict:
    step = {"action": {"tool": tool, "arguments": {"state_id": "s", **(arguments or {})}}, "execution_status": status,
            "error_code": None if status == "success" else "refused", "candidate_disposition_offline": disposition}
    if intervention:
        step["intervention"] = {"kind": "scripted"}
    if number is not None:
        step["step"] = number
    return step


TRUTH = {"meters": {3, 4}, "lines": {7}, "breakers": {"CB_1"}, "dangling_lines": set(), "known": True}


class TargetTests(unittest.TestCase):
    def test_scenario_truth_reads_the_audit_block(self) -> None:
        scenario = {"audit": {"truth": {
            "true_measurement_errors": [{"index": 3}, {"index": 4}],
            "true_parameter_errors": [{"line_index1": 7}],
            "true_topology_errors": [{"cb_name": "CB_1"}, {"line_index1": 2}],
        }}}
        truth = scenario_truth(scenario)
        self.assertEqual((truth["meters"], truth["lines"], truth["breakers"], truth["dangling_lines"], truth["known"]),
                         ({3, 4}, {7}, {"CB_1"}, {2}, True))
        self.assertFalse(scenario_truth({})["known"])
        self.assertFalse(scenario_truth({"audit": {"truth": {}}})["known"])

    def test_target_is_true_by_tool(self) -> None:
        self.assertTrue(target_is_true(CORRECT_MEASUREMENTS, {"suspect_group": [3, 4]}, TRUTH))
        self.assertFalse(target_is_true(CORRECT_MEASUREMENTS, {"suspect_group": [3, 5]}, TRUTH))
        self.assertTrue(target_is_true(CORRECT_MEASUREMENTS, {"index": 4}, TRUTH))
        self.assertFalse(target_is_true(CORRECT_MEASUREMENTS, {"suspect_group": []}, TRUTH))
        self.assertIsNone(target_is_true(CORRECT_MEASUREMENTS, {"suspect_group": ["x"]}, TRUTH))
        self.assertTrue(target_is_true(CORRECT_PARAMETERS, {"line_index": 7}, TRUTH))
        self.assertFalse(target_is_true(CORRECT_PARAMETERS, {"line_index": 8}, TRUTH))
        self.assertIsNone(target_is_true(CORRECT_PARAMETERS, {}, TRUTH))
        self.assertTrue(target_is_true(CORRECT_TOPOLOGY, {"cb_name": "CB_1", "status": 1}, TRUTH))
        self.assertFalse(target_is_true(CORRECT_TOPOLOGY, {"cb_name": "CB_2"}, TRUTH))
        self.assertFalse(target_is_true(CORRECT_TOPOLOGY, {"line_index": 2}, TRUTH))
        self.assertIsNone(target_is_true(RUN_WLS, {}, TRUTH))


class DecisionTests(unittest.TestCase):
    def test_commit_and_decline_are_labelled_against_the_audit(self) -> None:
        trace = [
            _step(RUN_WLS, number=1),
            _step(CORRECT_PARAMETERS, {"line_index": 7}, number=2),
            _step(RUN_WLS, number=3),
            _step(VERIFY_CANDIDATE, number=4),
            _step(COMMIT_STATE, disposition="ACCEPT_PARTIAL", number=5),  # a true target committed: agrees
            _step(CORRECT_MEASUREMENTS, {"suspect_group": [5]}, number=6),
            _step(RUN_WLS, number=7),
            _step(VERIFY_CANDIDATE, number=8),
            _step(ROLLBACK_STATE, disposition="ACCEPT_PARTIAL", number=9),  # a wrong target declined: agrees
            _step(CORRECT_MEASUREMENTS, {"suspect_group": [3]}, number=10),
            _step(ROLLBACK_STATE, disposition="ACCEPT_FINAL", number=11),  # a true target declined: disagrees
            _step(CORRECT_TOPOLOGY, {"cb_name": "CB_2", "status": 0}, number=12),
            _step(COMMIT_STATE, disposition="ACCEPT_FINAL", number=13),  # a wrong target committed: disagrees
        ]
        decisions = lifecycle_decisions(trace, TRUTH)
        self.assertEqual([d["position"] for d in decisions], [4, 8, 10, 12])
        self.assertEqual([d["step"] for d in decisions], [5, 9, 11, 13])
        self.assertEqual([d["tool"] for d in decisions], [COMMIT_STATE, ROLLBACK_STATE, ROLLBACK_STATE, COMMIT_STATE])
        self.assertEqual([d["correction_tool"] for d in decisions],
                         [CORRECT_PARAMETERS, CORRECT_MEASUREMENTS, CORRECT_MEASUREMENTS, CORRECT_TOPOLOGY])
        self.assertEqual([d["correction_position"] for d in decisions], [1, 5, 9, 11])
        self.assertEqual([d["target_is_true"] for d in decisions], [True, False, True, False])
        self.assertEqual([d["label"] for d in decisions], [COMMIT_STATE, ROLLBACK_STATE, COMMIT_STATE, ROLLBACK_STATE])
        self.assertEqual([d["agrees"] for d in decisions], [True, True, False, False])
        self.assertEqual([d["declined"] for d in decisions], [False, True, True, False])
        self.assertEqual([decision_sign(d) for d in decisions], [1.0, 1.0, -1.0, -1.0])
        self.assertEqual(decisions[0]["arguments"], {"line_index": 7})
        self.assertEqual(summarize_decisions(decisions), {
            "decisions": 4, "commits": 2, "declines": 2, "agreeing": 2, "disagreeing": 2, "unlabelled": 0,
            "true_target_declined": 1, "wrong_target_committed": 1, "wrong_target_declined": 1,
        })

    def test_forced_and_refused_lifecycle_steps_are_not_decisions(self) -> None:
        trace = [
            _step(CORRECT_PARAMETERS, {"line_index": 8}),
            _step(ROLLBACK_STATE, disposition="REJECT"),  # forced: the verifier rejected the candidate
            _step(CORRECT_PARAMETERS, {"line_index": 7}),
            _step(COMMIT_STATE, disposition="REJECT", status="failure"),  # refused by the process gate
            _step(ROLLBACK_STATE, disposition="INCONCLUSIVE"),
            _step(CORRECT_PARAMETERS, {"line_index": 7}, status="failure"),  # a refused correction opens nothing
            _step(COMMIT_STATE, disposition="ACCEPT_PARTIAL"),  # nothing open: not a decision
            _step(RUN_WLS, intervention=True),  # interventions are not policy steps
            _step(CORRECT_PARAMETERS, {"line_index": 7}),
            _step(COMMIT_STATE, disposition="ACCEPT_PARTIAL"),
        ]
        decisions = lifecycle_decisions(trace, TRUTH)
        self.assertEqual(len(decisions), 1)
        self.assertEqual((decisions[0]["position"], decisions[0]["step"], decisions[0]["agrees"]), (8, 8, True))
        self.assertEqual(lifecycle_decisions([], TRUTH), [])

    def test_unknown_truth_leaves_decisions_unlabelled(self) -> None:
        trace = [_step(CORRECT_PARAMETERS, {"line_index": 7}), _step(COMMIT_STATE, disposition="ACCEPT_FINAL")]
        unlabelled = lifecycle_decisions(trace, scenario_truth({}))
        self.assertEqual(len(unlabelled), 1)
        self.assertIsNone(unlabelled[0]["agrees"])
        self.assertIsNone(unlabelled[0]["label"])
        self.assertIsNone(decision_sign(unlabelled[0]))
        self.assertEqual(summarize_decisions(unlabelled)["unlabelled"], 1)
        # A correction whose arguments name no target the truth can be read for.
        trace = [_step(CORRECT_PARAMETERS, {}), _step(COMMIT_STATE, disposition="ACCEPT_FINAL")]
        self.assertIsNone(lifecycle_decisions(trace, TRUTH)[0]["agrees"])
        # A healthy root with its truth block: every accepted candidate is on a wrong target.
        healthy = scenario_truth({"audit": {"truth": {"true_measurement_errors": []}}})
        trace = [_step(CORRECT_MEASUREMENTS, {"suspect_group": [2]}), _step(ROLLBACK_STATE, disposition="ACCEPT_PARTIAL")]
        self.assertEqual(lifecycle_decisions(trace, healthy)[0]["agrees"], True)


if __name__ == "__main__":
    unittest.main()
