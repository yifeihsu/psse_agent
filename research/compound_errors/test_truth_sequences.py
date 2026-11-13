"""Truth-built sequences: the scripted policy on the toy network, the order search, the labels and the counts.

The toy network of ``psse_env/test_gated_revert`` in the open-rules
environment of ``psse_env/test_open_rules``: the parameter context lists lines
3 to 6; line 3 is a partial step, line 4 is rejected, line 5's tool fails,
line 6 clears the alarm and line 1 is not listed.  The toy never mints the
confirmation marker of a quiet estimator, so the handoff is exercised with
the alarm still on.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    OPEN_RULE_COMMIT_REJECTED_CANDIDATE,
    OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT,
    OPEN_RULE_HANDOFF_WITHOUT_EXHAUSTION,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.test_gated_revert import SCENARIO
from psse_env.test_open_rules import _env
from research.compound_errors import truth_sequences as ts


def _line(index: int) -> dict:
    return {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "S", "line_index": index}}


def _drive(env, policy: ts.TruthSequencePolicy, limit: int = 20) -> list[dict]:
    """The policy against the toy environment; one trace step per action, as the evaluator records them."""
    env.reset(SCENARIO)
    trace = []
    while not env.terminal and len(trace) < limit:
        observation = env.get_policy_observation().as_dict()
        action = policy.act(observation)
        _, output = env.step(action)
        step = {"action": action, "policy_observation": observation, "execution_status": output["execution_status"],
                "error_code": output.get("error_code"), "policy_tool_output": output}
        if env.last_open_rules_used:
            step["open_rules_used"] = list(env.last_open_rules_used)
        trace.append(step)
    return trace


def _tools(trace) -> list[str]:
    return [step["action"]["tool"] for step in trace]


class OrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meters = {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": "S", "suspect_group": [4, 9]}}
        self.breaker = {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": "S", "cb_name": "CB_7", "status": 1}}

    def test_the_first_order_is_breakers_lines_meters(self) -> None:
        first = ts.orders([self.meters, _line(3), self.breaker, _line(2)], 1)
        self.assertEqual([[action["tool"] for action in order] for order in first],
                         [[CORRECT_TOPOLOGY, CORRECT_PARAMETERS, CORRECT_PARAMETERS, CORRECT_MEASUREMENTS]])
        # Inside a family the truth's own order stays.
        self.assertEqual([action["arguments"]["line_index"] for action in first[0][1:3]], [3, 2])

    def test_every_other_permutation_follows_once_up_to_the_limit(self) -> None:
        every = ts.orders([self.meters, _line(3), self.breaker], 6)
        self.assertEqual(len(every), 6)
        self.assertEqual(len({json.dumps([ts.signature(action) for action in order]) for order in every}), 6)
        self.assertEqual(len(ts.orders([self.meters, _line(3), self.breaker], 4)), 4)
        self.assertEqual(len(ts.orders([self.meters, _line(3)], 6)), 2)
        # A root without a correction still has its one, empty, sequence.
        self.assertEqual(ts.orders([], 6), [[]])

    def test_an_action_is_named_without_the_state_it_was_bound_to(self) -> None:
        bound = {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "episode:s3", "line_index": 3}}
        self.assertEqual(ts.strip(bound), {"tool": CORRECT_PARAMETERS, "arguments": {"line_index": 3}})
        self.assertEqual(ts.signature(bound), ts.signature(_line(3)))
        self.assertEqual(ts.strip({"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "c"}}), {"tool": COMMIT_STATE, "arguments": {}})


class PolicyTests(unittest.TestCase):
    def test_a_listed_correction_is_verified_kept_and_handed_off(self) -> None:
        env, policy = _env(True), ts.TruthSequencePolicy([_line(3)])
        trace = _drive(env, policy)
        self.assertEqual(policy.kinds, [ts.KIND_FIRST_SOLVE, ts.KIND_CONTEXT, ts.KIND_CORRECTION, ts.KIND_VERIFY, ts.KIND_COMMIT,
                                        ts.KIND_CLOSING_CONTEXT, ts.KIND_HANDOFF])
        self.assertEqual(_tools(trace), [RUN_WLS, GET_PARAMETER_CONTEXT, CORRECT_PARAMETERS, RUN_WLS, COMMIT_STATE,
                                         GET_MEASUREMENT_CONTEXT, ASK_FOR_MORE_EVIDENCE])
        self.assertTrue(all(step["execution_status"] == "success" for step in trace))
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(policy.failed, [])
        # Only the handoff leaned on a lifted rule: the context still lists corrections nobody tried.
        self.assertEqual([step.get("open_rules_used") for step in trace],
                         [None, None, None, None, None, None, [OPEN_RULE_HANDOFF_WITHOUT_EXHAUSTION]])

    def test_an_unlisted_target_is_corrected_and_a_rejected_candidate_kept(self) -> None:
        env, policy = _env(True), ts.TruthSequencePolicy([_line(1), _line(6)])
        trace = _drive(env, policy)
        self.assertEqual(policy.kinds, [ts.KIND_FIRST_SOLVE, ts.KIND_CONTEXT, ts.KIND_CORRECTION, ts.KIND_VERIFY, ts.KIND_COMMIT,
                                        ts.KIND_CONTEXT, ts.KIND_CORRECTION, ts.KIND_VERIFY, ts.KIND_COMMIT,
                                        ts.KIND_CLOSING_CONTEXT, ts.KIND_HANDOFF])
        self.assertTrue(all(step["execution_status"] == "success" for step in trace))
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        # The context is asked again on the state the first commit made active.
        self.assertNotEqual(trace[1]["action"]["arguments"]["state_id"], trace[5]["action"]["arguments"]["state_id"])
        self.assertEqual(trace[2].get("open_rules_used"), [OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT])
        self.assertEqual(trace[4].get("open_rules_used"), [OPEN_RULE_COMMIT_REJECTED_CANDIDATE])
        self.assertIsNone(trace[6].get("open_rules_used"))
        self.assertEqual([c["source_action"]["arguments"]["line_index"] for c in env.current_state()["accepted_corrections"]], [1, 6])

    def test_a_failed_correction_is_tried_once_more_after_the_others(self) -> None:
        env, policy = _env(True), ts.TruthSequencePolicy([_line(5), _line(3)])
        trace = _drive(env, policy)
        corrections = [(step["action"]["arguments"]["line_index"], step["execution_status"])
                       for kind, step in zip(policy.kinds, trace) if kind == ts.KIND_CORRECTION]
        self.assertEqual(corrections, [(5, "failure"), (3, "success"), (5, "failure")])
        self.assertEqual([entry["action"] for entry in policy.failed], [ts.strip(_line(5))] * 2)
        self.assertEqual(policy.failed[0]["error_code"], "parameter_correction_failure")
        self.assertEqual(policy.kinds[-2:], [ts.KIND_CLOSING_CONTEXT, ts.KIND_HANDOFF])
        self.assertEqual(env.terminal_outcome, "operator_escalation")

    def test_a_commit_refused_twice_is_withdrawn(self) -> None:
        # With the rules closed a rejected candidate cannot be kept: the policy gives it up and goes on.
        env, policy = _env(None), ts.TruthSequencePolicy([_line(4), _line(6)])
        trace = _drive(env, policy, limit=11)
        self.assertEqual(policy.kinds[:11], [ts.KIND_FIRST_SOLVE, ts.KIND_CONTEXT, ts.KIND_CORRECTION, ts.KIND_VERIFY, ts.KIND_COMMIT,
                                             ts.KIND_COMMIT, ts.KIND_ROLLBACK, ts.KIND_CORRECTION, ts.KIND_VERIFY, ts.KIND_COMMIT,
                                             ts.KIND_CLOSING_CONTEXT])
        self.assertEqual([step["execution_status"] for step in trace[4:7]], ["failure", "failure", "success"])
        self.assertEqual(trace[6]["action"]["tool"], ROLLBACK_STATE)
        self.assertEqual(policy.failed, [{"action": ts.strip(_line(4)), "error_code": "commit_refused:candidate_lifecycle_violation",
                                          "error_detail": "commit_rejected_or_inconclusive_candidate"}])
        self.assertFalse(any(step.get("open_rules_used") for step in trace))

    def test_no_correction_is_the_solve_the_closing_context_and_the_handoff(self) -> None:
        env, policy = _env(True), ts.TruthSequencePolicy([])
        _drive(env, policy)
        self.assertEqual(policy.kinds, [ts.KIND_FIRST_SOLVE, ts.KIND_CLOSING_CONTEXT, ts.KIND_HANDOFF])


class TidyingTests(unittest.TestCase):
    def test_a_target_whose_tool_never_succeeds_is_dropped(self) -> None:
        order = [_line(5), _line(3)]
        env, policy = _env(True), ts.TruthSequencePolicy(order)
        trace = _drive(env, policy)
        self.assertEqual([ts.strip(action) for action in ts.tidied_order(order, {"trace": trace}, policy)], [ts.strip(_line(3))])

    def test_a_target_that_needed_a_second_attempt_goes_last(self) -> None:
        order = [_line(5), _line(3)]
        policy = ts.TruthSequencePolicy(order)
        policy.kinds = [ts.KIND_CORRECTION, ts.KIND_CORRECTION, ts.KIND_CORRECTION]
        policy.failed = [{"action": ts.strip(_line(5)), "error_code": "parameter_correction_failure", "error_detail": None}]
        trace = [{"action": _line(5), "execution_status": "failure"}, {"action": _line(3), "execution_status": "success"},
                 {"action": _line(5), "execution_status": "success"}]
        self.assertEqual([ts.strip(action) for action in ts.tidied_order(order, {"trace": trace}, policy)],
                         [ts.strip(_line(3)), ts.strip(_line(5))])
        # Already last: nothing to tidy.
        self.assertIsNone(ts.tidied_order([_line(3), _line(5)], {"trace": trace}, policy))

    def test_a_clean_sequence_is_left_alone(self) -> None:
        order = [_line(3)]
        env, policy = _env(True), ts.TruthSequencePolicy(order)
        trace = _drive(env, policy)
        self.assertIsNone(ts.tidied_order(order, {"trace": trace}, policy))


class LabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env, self.policy = _env(True), ts.TruthSequencePolicy([_line(1), _line(6)])
        trace = _drive(self.env, self.policy)
        # The evaluator records the acceptance check's verdict on the step that disposes of the candidate.
        for kind, step in zip(self.policy.kinds, trace):
            if kind == ts.KIND_COMMIT:
                step["candidate_disposition_offline"] = "REJECT"
        self.episode = {"trace": trace, "truth_audited_task_success": True, "terminal": True, "terminal_outcome": "operator_escalation",
                        "keep_accounting": {"kept": 2, "events": []},
                        "open_rule_accounting": {"counts": {OPEN_RULE_COMMIT_REJECTED_CANDIDATE: 2, OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT: 1}}}
        self.root = {"execution": {"scenario_id": "toy"}, "grouping": {"scenario_family": "parameter", "error_cardinality": 2},
                     "audit": {"compound": {"class": "rx2_distant"},
                               "truth": {"true_parameter_errors": [{"line_index1": 1}, {"line_index1": 6}]}}}

    def test_rows_carry_what_a_selection_needs(self) -> None:
        # A row's user turn shows the state the policy acted on; the evidence label is read from it.
        rows = [{"cloning": {"step": index, "tool": step["action"]["tool"]},
                 "messages": [{"role": "system", "content": "s"},
                              {"role": "user", "content": json.dumps({"state": step["policy_observation"]}, default=str)}]}
                for index, step in enumerate(self.episode["trace"])]
        shown = [row["messages"][1]["content"] for row in rows]
        ts.label_rows(rows, self.episode, self.policy.kinds, passes=True, rule_expert_solves=False)
        labels = [row["cloning"]["truth_sequence"] for row in rows]
        # Line 1 is not in the context's list nor in a signature; line 6 is listed.
        self.assertEqual([label["target_in_evidence"] for label in labels],
                         [None, None, ts.EVIDENCE_ABSENT, None, None, None, ts.EVIDENCE_LISTED, None, None, None, None])
        self.assertEqual([row["messages"][1]["content"] for row in rows], shown)
        self.assertEqual([label["kind"] for label in labels], self.policy.kinds)
        self.assertTrue(all(label["contract"] == ts.SEQUENCE_CONTRACT and label["passes_truth_audit"] is True
                            and label["rule_expert_solves"] is False and label["execution_status"] == "success" for label in labels))
        self.assertEqual(labels[2]["open_rules_used"], [OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT])
        self.assertEqual(labels[4]["open_rules_used"], [OPEN_RULE_COMMIT_REJECTED_CANDIDATE])
        self.assertEqual(labels[0]["open_rules_used"], [])
        # The verdict sits on the lifecycle step alone.
        self.assertEqual([label["verdict"] for label in labels],
                         [None, None, None, None, "REJECT", None, None, None, "REJECT", None, None])
        # The label is beside the cloning label, never in a message.
        self.assertEqual(set(rows[0]), {"cloning", "messages"})

    def test_these_rows_name_their_own_id_tag(self) -> None:
        from research.compound_errors import cloning_rows

        mine = cloning_rows.episode_examples(self.episode, self.root, behaviour=ts.BEHAVIOUR, tag=ts.ROW_TAG)
        cloned = cloning_rows.episode_examples(self.episode, self.root, behaviour="teacher")
        self.assertEqual([example["example_id"] for example in mine[:2]], ["toy:truth:00", "toy:truth:01"])
        self.assertEqual([example["example_id"] for example in cloned[:2]], ["toy:clone:00", "toy:clone:01"])
        self.assertFalse({example["example_id"] for example in mine} & {example["example_id"] for example in cloned})
        self.assertEqual({example["dataset_source"] for example in mine}, {ts.BEHAVIOUR})

    def test_the_record_names_the_order_the_kinds_and_the_accounting(self) -> None:
        built = {"episode": self.episode, "policy": self.policy, "order": [_line(1), _line(6)], "runs": [{"success": True}],
                 "tidied": False, "truth_corrections": [ts.strip(_line(1)), ts.strip(_line(6))], "orders_tried": 1,
                 "orders_available": 2}
        record = ts.sequence_record(self.root, built, rule_expert_solves=False)
        self.assertTrue(record["success"] and record["passes_truth_audit"])
        self.assertIs(record["rule_expert_solves"], False)
        self.assertEqual(record["class"], "rx2_distant")
        self.assertEqual(record["order"], [ts.strip(_line(1)), ts.strip(_line(6))])
        self.assertEqual(record["kinds"], self.policy.kinds)
        self.assertEqual([entry["kind"] for entry in record["trace"]], self.policy.kinds)
        self.assertEqual(record["trace"][2]["open_rules_used"], [OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT])
        self.assertNotIn("open_rules_used", record["trace"][0])
        self.assertEqual(record["keep_accounting"]["kept"], 2)
        self.assertEqual(record["open_rule_accounting"]["counts"][OPEN_RULE_COMMIT_REJECTED_CANDIDATE], 2)
        self.assertEqual(len(record["open_rules"]), 5)
        # Every correction of the sequence names a true target, and the verdicts are the acceptance check's.
        self.assertEqual([(c["target_is_true"], c["verdict"], c["lifecycle"]) for c in record["corrections"]],
                         [(True, "REJECT", COMMIT_STATE)] * 2)
        # What the cloning pass's selection reads.
        self.assertEqual((record["declines"], record["gate_decisions"]), (0, []))
        json.dumps(record)

    def test_the_counts_follow_the_records(self) -> None:
        passing = ts.sequence_record(self.root, {"episode": self.episode, "policy": self.policy, "order": [], "runs": [], "tidied": True,
                                                 "truth_corrections": [], "orders_tried": 2, "orders_available": 2},
                                     rule_expert_solves=False)
        passing["rows_refused"] = {"commit_state | ValueError: x": 1}
        failing = {"scenario_id": "other", "class": "control_rx", "success": False, "rule_expert_solves": True, "steps": 1,
                   "audit_reasons": ["final_case_outside_clean_tolerance"], "orders_tried": 1, "terminal_outcome": "operator_escalation",
                   "trace": [{"kind": ts.KIND_FIRST_SOLVE, "status": "success"}]}
        summary = ts.summarize([passing, failing])
        self.assertEqual((summary["roots"], summary["pass"], summary["tidied"], summary["errors"]), (2, 1, 1, 0))
        self.assertEqual((summary["kept_commits"], summary["episodes_needing_a_lifted_rule"], summary["closed_by_handoff"]), (2, 1, 2))
        self.assertEqual(summary["orders_tried"], {1: 1, 2: 1})
        self.assertEqual(summary["audit_reasons_of_failures"], {"final_case_outside_clean_tolerance": 1})
        self.assertEqual(summary["by_class"], {"control_rx": {"roots": 1, "pass": 0}, "rx2_distant": {"roots": 1, "pass": 1}})
        self.assertEqual(summary["by_rule_expert_outcome"]["rule expert fails"], {"roots": 1, "pass": 1, "steps": 11})
        self.assertEqual(summary["by_rule_expert_outcome"]["rule expert solves"], {"roots": 1, "pass": 0, "steps": 1})
        steps = summary["steps"]
        self.assertEqual((steps["steps"], steps["steps_in_passing_episodes"]), (12, 11))
        self.assertEqual((steps["kind:commit"], steps["kind:first_solve"], steps["kind:handoff"]), (2, 2, 1))
        # The unlisted correction, the two kept commits and the handoff.
        self.assertEqual(steps["steps_needing_a_lifted_rule"], 4)
        self.assertEqual(steps["steps_needing_a_lifted_rule_in_passing_episodes"], 4)
        self.assertEqual(steps[f"rule:{OPEN_RULE_COMMIT_REJECTED_CANDIDATE}"], 2)
        self.assertEqual(steps[f"rule:{OPEN_RULE_HANDOFF_WITHOUT_EXHAUSTION}"], 1)
        self.assertEqual(summary["rows_refused"], {"commit_state | ValueError: x": 1})


class EvidenceTests(unittest.TestCase):
    """Whether the shown state names a correction's target (the layouts are the deployment contexts')."""

    @staticmethod
    def _state(family: str, context: dict, *, signatures=(), wls=None) -> dict:
        return {"fresh_context_evidence": {family: context, "wls": wls or {}}, "unresolved_signatures": list(signatures)}

    def test_a_line(self) -> None:
        action = _line(7)
        listed = self._state("parameter", {"supported_corrections": [{"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "active", "line_index": 7}}]})
        self.assertEqual(ts.target_evidence(listed, action), ts.EVIDENCE_LISTED)
        other = {"supported_corrections": [{"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "active", "line_index": 3}}]}
        for named in (
            self._state("parameter", dict(other, parameter_ranking_distinct_lines=[{"abs_lambda_score": 2.0, "line_index1": 7}])),
            self._state("parameter", dict(other, parameter_findings=[{"line_row0": 6, "parameter": "X", "value": 3.1}])),
            self._state("parameter", other, signatures=["wls_branch_multiplier line_status_or_parameter line=7"]),
            self._state("parameter", other, wls={"flow_at_noise_lines": [7]}),
            self._state("parameter", other, wls={"hypothesis_confidence": {"parameter_line": 7, "topology_line": 2}}),
        ):
            self.assertEqual(ts.target_evidence(named, action), ts.EVIDENCE_NAMED)
        for absent in (
            self._state("parameter", other, signatures=["wls_branch_multiplier line_status_or_parameter line=17"]),
            self._state("parameter", {"supported_corrections": [], "route_status": "complete_negative", "parameter_ranking_distinct_lines": []}),
            # A meter index is not a line.
            self._state("parameter", other, signatures=["wls_residual_outlier index=7 channel=Pf"]),
            {},
        ):
            self.assertEqual(ts.target_evidence(absent, action), ts.EVIDENCE_ABSENT)

    def test_a_meter_group(self) -> None:
        action = {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": "S", "suspect_group": [44, 78]}}

        def context(*groups):
            return {"supported_corrections": [{"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": "active", "suspect_group": list(group)}}
                                              for group in groups]}

        self.assertEqual(ts.target_evidence(self._state("measurement", context([78, 44])), action), ts.EVIDENCE_LISTED)
        self.assertEqual(ts.target_evidence(self._state("measurement", context([44], [78], [9])), action), ts.EVIDENCE_LISTED_SINGLY)
        named = self._state("measurement", dict(context([78]), measurement_findings=[{"channel": "Pf", "index0": 44, "value": 5.0}]))
        self.assertEqual(ts.target_evidence(named, action), ts.EVIDENCE_NAMED)
        by_signature = self._state("measurement", context([9]), signatures=["wls_residual_outlier index=44 channel=Pf",
                                                                               "wls_residual_outlier_dominant index=78 channel=Qf"])
        self.assertEqual(ts.target_evidence(by_signature, action), ts.EVIDENCE_NAMED)
        self.assertEqual(ts.target_evidence(self._state("measurement", context([78])), action), ts.EVIDENCE_PARTLY_NAMED)
        self.assertEqual(ts.target_evidence(self._state("measurement", context([9])), action), ts.EVIDENCE_ABSENT)

    def test_a_breaker(self) -> None:
        action = {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": "S", "cb_name": "CB_2R2_2R3", "status": 0, "line_index": 5}}
        listed = self._state("topology", {"supported_corrections": [{"tool": CORRECT_TOPOLOGY, "arguments": {
            "cb_name": "CB_2R2_2R3", "line_index": 5, "state_id": "active", "status": 0}}]})
        self.assertEqual(ts.target_evidence(listed, action), ts.EVIDENCE_LISTED)
        named = self._state("topology", {"supported_corrections": [], "route_confidence": {"breaker_evidence": {"cb_name": "CB_2R2_2R3"}}})
        self.assertEqual(ts.target_evidence(named, action), ts.EVIDENCE_NAMED)
        line_only = self._state("topology", {"supported_corrections": [], "topology_findings": [{"line_row0": 4, "parameter": "X", "value": 9.0}]})
        self.assertEqual(ts.target_evidence(line_only, action), ts.EVIDENCE_LINE_NAMED)
        self.assertEqual(ts.target_evidence(self._state("topology", {"supported_corrections": []}), action), ts.EVIDENCE_ABSENT)
        # A breaker without a line of its own (a bus split) is named or absent, never line named.
        split = {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": "S", "cb_name": "CB_4_BUS", "status": 1}}
        self.assertEqual(ts.target_evidence(line_only, split), ts.EVIDENCE_ABSENT)

    def test_only_a_correction_has_a_target(self) -> None:
        self.assertIsNone(ts.target_evidence({}, {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "c"}}))
        self.assertIsNone(ts.target_evidence({}, {"tool": RUN_WLS, "arguments": {"state_id": "s"}}))

    def test_the_shown_state_is_the_user_turns(self) -> None:
        row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": json.dumps({"state": {"a": 1}})},
                            {"role": "assistant", "content": ""}]}
        self.assertEqual(ts.shown_state(row), {"a": 1})
        self.assertEqual(ts.shown_state({"messages": []}), {})


class ReferenceTests(unittest.TestCase):
    def test_both_record_layouts_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in (
                {"scenario_id": "a", "success": True},
                {"id": "b", "episode": {"scenario_id": "b", "success": False}, "audit": None},
                {"id": "c", "episode": {"scenario_id": "c"}},
            )) + "\n", encoding="utf-8")
            self.assertEqual(ts.read_reference(path), {"a": True, "b": False})
        self.assertEqual(ts.read_reference(None), {})


if __name__ == "__main__":
    unittest.main()
