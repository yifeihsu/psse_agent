"""Item 7 step 6 (the user's decision of 2026-10-28): the gated revert of a committed correction.

A toy network of six lines: correcting line 3 lowers the WLS objective
without clearing it (a viable partial step that turns out wrong), every
correction tried after it on top of it fails, and line 6 alone clears the
alarm.  The revert makes the root active again once enough fruitless cycles
accumulated, restores what the controller knew there, and records the
withdrawn correction as a rejected hypothesis of the root.
"""
from __future__ import annotations

import unittest

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    REVERT_CONTRACT,
    REVERT_STATE,
    REVERTED_REJECTION_KIND,
    ROLLBACK_STATE,
    RUN_WLS,
    correction_hypothesis,
)
from psse_env.dagger.evaluator import update_revert_events
from psse_env.evidence_profile import SCADA_ONLY_PROFILE
from psse_env.oracle.candidate_quality import CandidateAssessment, CandidateDisposition
from psse_env.oracle.process_validity import ProcessValidityOracle, revert_refusal
from psse_env.state_store import PolicyObservation, PowerSystemStateStore, StateStoreError
from psse_env.transactional_env import TransactionalPSSEEnv

ALARM = 10.0
_OBJECTIVE = {frozenset(): 40.0, frozenset({3}): 30.0, frozenset({3, 6}): 50.0, frozenset({3, 4}): 45.0,
              frozenset({6}): 0.0, frozenset({4}): 41.0}


def _corrected(state) -> frozenset:
    rows = (state.get("case") or {}).get("branch") or []
    return frozenset(index + 1 for index, row in enumerate(rows) if float(row.get("x", 1.0)) != 1.0)


def _wls(state):
    objective = _OBJECTIVE.get(_corrected(state), 60.0)
    alarm = objective >= ALARM
    return {
        "wls_objective": objective, "remaining_anomaly_score": objective / ALARM, "chi_square_statistic": objective,
        "chi_square_threshold": ALARM, "chi_square_alarm": alarm, "converged": True, "globally_resolved": not alarm,
        "no_material_anomaly_remaining": not alarm,
        "unresolved_signatures": ["wls_branch_multiplier_dominant line_status_or_parameter line=3"] if alarm else [],
    }


def _parameter_context(state):
    return {
        "state_id": state["state_id"], "state_hash": state["state_hash"], "route_status": "actionable",
        "supported_corrections": [{"tool": CORRECT_PARAMETERS, "arguments": {"state_id": state["state_id"], "line_index": line}}
                                  for line in (3, 4, 5, 6)],
    }


def _empty_context(state):
    return {"state_id": state["state_id"], "state_hash": state["state_hash"], "supported_corrections": [],
            "route_status": "complete_negative"}


def _correct_parameters(state, action):
    del state
    line = int(action["arguments"]["line_index"])
    if line == 5:
        return {"execution_status": "failure", "error_code": "parameter_correction_failure"}
    return {"modification": {"line_index1": line, "parameter": "x", "value": 0.5}}


class _Quality:
    """Verdicts from the verification solve alone: quiet is final, a lower objective partial, else rejected."""

    def label_candidate(self, *, parent_state, source_action, candidate_state, verification_output, hidden_truth):
        del parent_state, source_action, candidate_state, hidden_truth
        if not verification_output.get("chi_square_alarm"):
            disposition = CandidateDisposition.ACCEPT_FINAL
        elif float(verification_output.get("global_progress") or 0.0) > 0.0:
            disposition = CandidateDisposition.ACCEPT_PARTIAL
        else:
            disposition = CandidateDisposition.REJECT
        return CandidateAssessment(disposition=disposition, progress_class="toy")

    @staticmethod
    def matched_fault_indices(source_action, truth, *, parent_state, candidate_state):
        del parent_state, candidate_state
        line = source_action.get("arguments", {}).get("line_index")
        return [index for index, fault in enumerate(truth.get("true_parameter_errors") or [])
                if fault.get("line_index1") == line]


def _env(cycles: int = 1) -> TransactionalPSSEEnv:
    return TransactionalPSSEEnv(
        evidence_profile=SCADA_ONLY_PROFILE,
        # Corrections name a target and the executor computes the values.
        process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True),
        wls_runner=_wls,
        candidate_quality_oracle=_Quality(),
        context_providers={GET_MEASUREMENT_CONTEXT: _empty_context, GET_PARAMETER_CONTEXT: _parameter_context,
                           GET_TOPOLOGY_CONTEXT: _empty_context},
        correction_executors={CORRECT_PARAMETERS: _correct_parameters},
        max_steps=60,
        revert_fruitless_cycles=cycles,
    )


SCENARIO = {
    "scenario_id": "revert-toy",
    "case": {"branch": [{"x": 1.0} for _ in range(6)]},
    "measurements": [1.0, 1.0],
    "true_parameter_errors": [{"line_index1": 3}, {"line_index1": 6}],
}


def _ok(env: TransactionalPSSEEnv, tool: str, **arguments) -> dict:
    _, output = env.step({"tool": tool, "arguments": arguments})
    assert output["execution_status"] == "success", (tool, output)
    return output


def _try(env: TransactionalPSSEEnv, line: int) -> str:
    """Correct ``line`` on the active state, verify, and commit or roll back; returns the verdict."""
    active = env.store.active_state_id
    output = _ok(env, CORRECT_PARAMETERS, state_id=active, line_index=line)
    candidate = output["candidate_state_id"]
    _ok(env, RUN_WLS, state_id=candidate)
    # The verdict is private to the store; the policy sees only the solve.
    verdict = env.store.get_state(candidate)["candidate_disposition"]
    _ok(env, COMMIT_STATE if verdict.startswith("ACCEPT") else ROLLBACK_STATE, candidate_state_id=candidate)
    return verdict


def _wrong_commit(env: TransactionalPSSEEnv) -> tuple[str, str]:
    """Root -> line 3 committed; returns (root, committed) with WLS and the parameter context fetched on both."""
    root = env.reset(SCENARIO)["active_state_id"]
    _ok(env, RUN_WLS, state_id=root)
    _ok(env, GET_PARAMETER_CONTEXT, state_id=root)
    assert _try(env, 3) == "ACCEPT_PARTIAL"
    committed = env.store.active_state_id
    _ok(env, RUN_WLS, state_id=committed)
    _ok(env, GET_PARAMETER_CONTEXT, state_id=committed)
    return root, committed


def _revert(env: TransactionalPSSEEnv, to_state_id: str) -> dict:
    _, output = env.step({"tool": REVERT_STATE, "arguments": {"state_id": env.store.active_state_id, "to_state_id": to_state_id}})
    return output


class StoreRevertTests(unittest.TestCase):
    def test_revert_marks_the_descendants_and_reactivates_the_ancestor(self) -> None:
        store = PowerSystemStateStore()
        root = store.create_root({"branch": [{"x": 1.0}]}, [1.0], episode_id="e")
        child = store.clone_candidate(root, {"line_index": 1, "value": 0.5}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        store.mark_verified(child, {}, "ACCEPT_PARTIAL")
        store.commit(child)
        grandchild = store.clone_candidate(child, {"line_index": 1, "value": 0.25}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        store.mark_verified(grandchild, {}, "ACCEPT_PARTIAL")
        store.commit(grandchild)
        root_hash = store.state_hash(root)
        with self.assertRaises(StateStoreError):
            store.revert(grandchild)  # the active state itself is no strict ancestor
        self.assertEqual(store.revert(root), [child, grandchild])
        self.assertEqual(store.active_state_id, root)
        self.assertEqual(store.state_hash(root), root_hash)
        self.assertEqual([store.get_state(s)["status"] for s in (child, grandchild)], ["reverted", "reverted"])
        # The withdrawn states stay readable; a new branch grows from the ancestor.
        self.assertEqual(store.source_action(child)["tool"], CORRECT_PARAMETERS)
        branch = store.clone_candidate(root, {"line_index": 1, "value": 0.75}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        self.assertEqual(store.lineage(branch), [root, branch])

    def test_revert_refuses_with_an_open_candidate(self) -> None:
        store = PowerSystemStateStore()
        root = store.create_root({"branch": [{"x": 1.0}]}, [1.0], episode_id="e")
        child = store.clone_candidate(root, {"line_index": 1, "value": 0.5}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        store.mark_verified(child, {}, "ACCEPT_PARTIAL")
        store.commit(child)
        store.clone_candidate(child, {"line_index": 1, "value": 0.25}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        with self.assertRaises(StateStoreError):
            store.revert(root)


class DisabledRevertTests(unittest.TestCase):
    def test_without_the_option_the_tool_is_unknown_and_the_observation_has_no_ledger(self) -> None:
        env = _env(cycles=0)
        root, committed = _wrong_commit(env)
        self.assertNotIn("revert_ledger", env.get_policy_observation().as_dict())
        self.assertNotIn("revert_ledger", env.current_state())
        output = _revert(env, root)
        self.assertEqual(output["error_code"], "unknown_tool")
        self.assertEqual(env.store.active_state_id, committed)

    def test_an_observation_without_a_ledger_serializes_exactly_as_before(self) -> None:
        payload = PolicyObservation(active_state_id="s0").as_dict()
        self.assertNotIn("revert_ledger", payload)
        self.assertIn("revert_ledger", PolicyObservation(active_state_id="s0", revert_ledger={"contract": REVERT_CONTRACT}).as_dict())


class GatedRevertTests(unittest.TestCase):
    def test_the_gate_admits_a_revert_only_after_fruitless_cycles(self) -> None:
        env = _env(cycles=2)
        root, committed = _wrong_commit(env)
        ledger = env.get_policy_observation().as_dict()["revert_ledger"]
        self.assertEqual(ledger["fruitless_cycle_threshold"], 2)
        self.assertEqual([(e["state_id"], e["committed_state_id"], e["fruitless_cycles_since"], e["reverted"])
                          for e in ledger["lineage"]], [(root, committed, 0, False)])
        self.assertEqual(_revert(env, root)["error_detail"], "revert_without_fruitless_cycles")
        self.assertEqual(_try(env, 6), "REJECT")  # one fruitless cycle on the committed state
        self.assertEqual(_revert(env, root)["error_code"], "candidate_lifecycle_violation")
        # An executor failure is a fruitless cycle as well.
        _, failed = env.step({"tool": CORRECT_PARAMETERS, "arguments": {"state_id": committed, "line_index": 5}})
        self.assertEqual(failed["error_code"], "parameter_correction_failure")
        observation = env.get_policy_observation().as_dict()
        self.assertEqual(observation["revert_ledger"]["lineage"][0]["fruitless_cycles_since"], 2)
        # Not a committed ancestor, or an open candidate: refused.
        self.assertEqual(_revert(env, committed)["error_detail"], "revert_target_not_committed_ancestor")
        self.assertEqual(_revert(env, "nowhere")["error_detail"], "revert_target_not_committed_ancestor")
        opened = _ok(env, CORRECT_PARAMETERS, state_id=committed, line_index=4)
        self.assertEqual(_revert(env, root)["error_detail"], "revert_with_open_candidate")
        _ok(env, RUN_WLS, state_id=opened["candidate_state_id"])
        _ok(env, ROLLBACK_STATE, candidate_state_id=opened["candidate_state_id"])
        self.assertEqual(env.store.active_state_id, committed)

    def test_a_revert_restores_the_ancestor_and_records_the_withdrawn_correction(self) -> None:
        env = _env(cycles=1)
        root, committed = _wrong_commit(env)
        before = env.get_policy_observation().as_dict()
        self.assertEqual(len(before["accepted_corrections"]), 1)
        self.assertEqual([f["line_index1"] for f in env._oracle_payload["remaining_true_faults"]], [6])
        self.assertEqual(_try(env, 6), "REJECT")
        output = _revert(env, root)
        self.assertEqual(output["execution_status"], "success")
        self.assertTrue(output["state_mutated"])
        metrics = output["tool_metrics"]
        self.assertEqual((metrics["reverted_to_state_id"], metrics["reverted_state_ids"], metrics["fruitless_cycles_since"]),
                         (root, [committed], 1))
        observation = env.get_policy_observation().as_dict()
        self.assertEqual(observation["active_state_id"], root)
        self.assertEqual(observation["accepted_corrections"], [])
        # The root's WLS and its parameter context are current again, with no new solve.
        self.assertEqual(observation["fresh_context_evidence"]["wls"]["state_id"], root)
        self.assertTrue(observation["fresh_context_evidence"]["wls"]["chi_square_alarm"])
        self.assertTrue(observation["has_fresh_parameter_context"])
        self.assertEqual(observation["parameter_context_state_id"], root)
        # The withdrawn correction is a rejected hypothesis of the root and stays in the ledger.
        (record,) = [r for r in observation["rejected_hypotheses"] if r.get("rejection_kind") == REVERTED_REJECTION_KIND]
        self.assertEqual((record["candidate_parent_id"], record["source_action"]["arguments"]["line_index"]), (root, 3))
        ledger = observation["revert_ledger"]
        self.assertEqual(ledger["lineage"], [])
        self.assertEqual([c["candidate_state_id"] for c in ledger["reverted_corrections"]], [committed])
        self.assertEqual(ledger["reverts"][0]["to_state_id"], root)
        # The private truth is the root's again.
        self.assertEqual([f["line_index1"] for f in env._oracle_payload["remaining_true_faults"]], [3, 6])
        # The other hypothesis clears the alarm from the root, and the root cannot be reverted to twice.
        self.assertEqual(_try(env, 6), "ACCEPT_FINAL")
        final = env.get_policy_observation().as_dict()
        self.assertEqual([c["candidate_state_id"] for c in final["accepted_corrections"]], [env.store.active_state_id])
        self.assertTrue(final["revert_ledger"]["lineage"][0]["reverted"])
        self.assertEqual(_revert(env, root)["error_detail"], "revert_ancestor_already_reverted")

    def test_the_shared_refusal_needs_a_standing_alarm(self) -> None:
        ledger = {"contract": REVERT_CONTRACT, "fruitless_cycle_threshold": 1,
                  "lineage": [{"state_id": "s0", "committed_state_id": "s1", "fruitless_cycles_since": 3, "reverted": False}]}
        quiet = {"active_state_id": "s1", "revert_ledger": ledger,
                 "fresh_context_evidence": {"wls": {"successful": True, "state_id": "s1", "state_hash": "h", "chi_square_alarm": False}}}
        self.assertEqual(revert_refusal(quiet, {"to_state_id": "s0"}), ("missing_precondition", "revert_requires_standing_alarm"))
        alarmed = {**quiet, "fresh_context_evidence": {"wls": {"successful": True, "state_id": "s1", "state_hash": "h",
                                                               "chi_square_alarm": True}}}
        self.assertIsNone(revert_refusal(alarmed, {"to_state_id": "s0"}))
        self.assertEqual(revert_refusal({**alarmed, "revert_ledger": None}, {"to_state_id": "s0"}), ("unknown_tool", REVERT_STATE))


class RevertAccountingTests(unittest.TestCase):
    @staticmethod
    def _revert_output(line: int, tool: str = CORRECT_PARAMETERS) -> dict:
        return {"execution_status": "success", "tool_metrics": {
            "reverted_to_state_id": "s0", "reverted_corrections": [
                {"candidate_state_id": "s1", "source_action": {"tool": tool, "arguments": {"state_id": "s0", "line_index": line}}}]}}

    def test_another_hypothesis_after_a_revert_is_a_recovery_and_the_same_one_a_loop(self) -> None:
        events: list = []
        revert = {"tool": REVERT_STATE, "arguments": {"to_state_id": "s0"}}
        self.assertFalse(update_revert_events(events, step=5, action=revert, status="success", output=self._revert_output(6, CORRECT_TOPOLOGY)))
        # A refused correction tested nothing and leaves the event open.
        refused = {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 6}}
        self.assertFalse(update_revert_events(events, step=6, action=refused, status="failure",
                                              output={"execution_status": "failure", "error_code": "missing_precondition"}))
        self.assertIsNone(events[0]["next_hypothesis"])
        self.assertFalse(update_revert_events(events, step=7, action=refused, status="success", output={"execution_status": "success"}))
        self.assertEqual((events[0]["withdrawn_hypothesis"], events[0]["next_hypothesis"], events[0]["outcome"]),
                         (["topology", 6], ["parameter", 6], "recovery"))
        looped: list = []
        update_revert_events(looped, step=5, action=revert, status="success", output=self._revert_output(6, CORRECT_TOPOLOGY))
        again = {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": "s0", "line_index": 6, "cb_name": "CB_9"}}
        self.assertTrue(update_revert_events(looped, step=6, action=again, status="success", output={"execution_status": "success"}))
        self.assertEqual(looped[0]["outcome"], "loop")

    def test_a_different_meter_is_another_hypothesis(self) -> None:
        revert = {"tool": REVERT_STATE, "arguments": {"to_state_id": "s0"}}
        withdrawn = {"execution_status": "success", "tool_metrics": {"reverted_to_state_id": "s0", "reverted_corrections": [
            {"candidate_state_id": "s1", "source_action": {"tool": "correct_measurements", "arguments": {"state_id": "s0", "suspect_group": [93]}}}]}}
        for group, outcome in (([11], "recovery"), ([93], "loop")):
            events: list = []
            update_revert_events(events, step=5, action=revert, status="success", output=withdrawn)
            update_revert_events(events, step=6, action={"tool": "correct_measurements", "arguments": {"state_id": "s0", "suspect_group": group}},
                                 status="success", output={"execution_status": "success"})
            self.assertEqual((events[0]["withdrawn_hypothesis"], events[0]["outcome"]), (["measurement", [93]], outcome))

    def test_hypotheses_are_family_and_line(self) -> None:
        self.assertEqual(correction_hypothesis({"tool": CORRECT_PARAMETERS, "arguments": {"line_index": 6}}), ("parameter", 6))
        self.assertEqual(correction_hypothesis({"tool": "correct_measurements", "arguments": {"suspect_group": [9, 4]}}), ("measurement", (4, 9)))
        self.assertEqual(correction_hypothesis({"tool": CORRECT_TOPOLOGY, "arguments": {"cb_name": "CB_9_N2_B2", "status": 0}}),
                         ("topology", "CB_9_N2_B2"))
        self.assertIsNone(correction_hypothesis({"tool": RUN_WLS, "arguments": {}}))


if __name__ == "__main__":
    unittest.main()
