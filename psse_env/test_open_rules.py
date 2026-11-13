"""The open rules (the user's direction of 2026-11-12): rules of the controller lifted for the policy.

The toy network of ``test_gated_revert``: the parameter context lists lines 3
to 6; correcting line 3 is a partial step, line 4 is rejected, line 6 clears
the alarm, and line 1 is not listed at all.  Without the option every rule
holds, as it always did: a correction the context does not list is refused,
a rejected candidate can only be rolled back, and the operator handoff waits
until every listed correction was tried.  With it the named rules are lifted
and nothing else changes: the correction tools compute the values and the
verification still reports what it found.
"""
from __future__ import annotations

import unittest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_PARAMETERS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    OPEN_RULE_COMMIT_REJECTED_CANDIDATE,
    OPEN_RULE_CORRECTION_AFTER_ESTIMATOR_PASSES,
    OPEN_RULE_CORRECTION_ON_CLOSED_ROUTE,
    OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT,
    OPEN_RULE_HANDOFF_WITHOUT_EXHAUSTION,
    OPEN_RULES,
    OPEN_RULES_CONTRACT,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_WLS,
    normalize_open_rules,
    open_rules_of,
    open_rules_record,
)
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    LIFECYCLE_CHOICE_PROMPT_PARAGRAPH,
    LIFECYCLE_CHOICE_ROLLBACK_ONLY_SENTENCE,
    OPEN_RULES_COMMIT_DESCRIPTION,
    OPEN_RULES_PROMPT_SENTENCES,
    open_rules_prompt_paragraph,
    system_prompt_for_observation,
    tool_schemas_for_observation,
)
from psse_env.dagger.evaluator import ClosedLoopRolloutEvaluator, commit_keeps_rejected_candidate
from psse_env.dagger.protocol_bridge import CANONICAL_TO_INTERNAL_TOOL, unified_tool_schemas
from psse_env.evidence_profile import SCADA_ONLY_PROFILE
from psse_env.oracle.process_validity import ProcessValidityOracle, post_correction_confirmation_required
from psse_env.state_store import PowerSystemStateStore, StateStoreError
from psse_env.test_gated_revert import SCENARIO, _correct_parameters, _empty_context, _ok, _parameter_context, _Quality, _wls
from psse_env.transactional_env import TransactionalPSSEEnv


def _handoff(state):
    return {"state_id": state["state_id"], "state_hash": state["state_hash"], "evidence_source": "deployment_diagnostic:toy",
            "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST, "additional_evidence_available": False, "operator_review_required": True}


def _env(open_rules=None, *, lifecycle_choice: bool = False, parameter_context=_parameter_context) -> TransactionalPSSEEnv:
    return TransactionalPSSEEnv(
        evidence_profile=SCADA_ONLY_PROFILE,
        process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True),
        wls_runner=_wls,
        candidate_quality_oracle=_Quality(),
        context_providers={GET_MEASUREMENT_CONTEXT: _empty_context, GET_PARAMETER_CONTEXT: parameter_context,
                           GET_TOPOLOGY_CONTEXT: _empty_context},
        correction_executors={CORRECT_PARAMETERS: _correct_parameters},
        evidence_providers={ASK_FOR_MORE_EVIDENCE: _handoff},
        max_steps=60,
        lifecycle_choice=lifecycle_choice,
        open_rules=open_rules,
    )


def _started(env: TransactionalPSSEEnv) -> str:
    """Root with WLS and the parameter context; returns the root id."""
    root = env.reset(SCENARIO)["active_state_id"]
    _ok(env, RUN_WLS, state_id=root)
    _ok(env, GET_PARAMETER_CONTEXT, state_id=root)
    return root


def _verified(env: TransactionalPSSEEnv, root: str, line: int) -> tuple[str, str]:
    """``line`` corrected on ``root`` and verified; (candidate, verdict)."""
    candidate = _ok(env, CORRECT_PARAMETERS, state_id=root, line_index=line)["candidate_state_id"]
    _ok(env, RUN_WLS, state_id=candidate)
    return candidate, env.store.get_state(candidate)["candidate_disposition"]


def _refused(env: TransactionalPSSEEnv, tool: str, **arguments) -> dict:
    _, output = env.step({"tool": tool, "arguments": arguments})
    assert output["execution_status"] == "failure", (tool, output)
    return output


def _schema(schemas, internal: str) -> dict:
    for tool in schemas:
        name = str(tool["function"]["name"])
        if CANONICAL_TO_INTERNAL_TOOL.get(name, name) == internal:
            return tool["function"]
    raise AssertionError(f"no {internal} schema")


class RuleNamesTests(unittest.TestCase):
    def test_a_flag_opens_every_rule_and_names_open_those_named(self) -> None:
        for everything in (True, 1, "1", "all", "true"):
            self.assertEqual(normalize_open_rules(everything), OPEN_RULES)
        for nothing in (None, False, 0, "", "0", "off", [], ()):
            self.assertEqual(normalize_open_rules(nothing), ())
        both = (OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT, OPEN_RULE_COMMIT_REJECTED_CANDIDATE)
        # Whatever the order written, the rules come back in the order of OPEN_RULES.
        self.assertEqual(normalize_open_rules(f"{both[1]}+{both[0]}"), both)
        self.assertEqual(normalize_open_rules(f"{both[1]}, {both[0]}"), both)
        self.assertEqual(normalize_open_rules([both[1], both[0]]), both)
        self.assertEqual(normalize_open_rules(open_rules_record(both)), both)

    def test_an_unknown_rule_is_an_error(self) -> None:
        for bad in ("no_such_rule", f"{OPEN_RULE_COMMIT_REJECTED_CANDIDATE}+typo", 2, {"contract": "other", "rules": []}):
            with self.assertRaises(ValueError):
                normalize_open_rules(bad)
        with self.assertRaises(ValueError):
            _env("no_such_rule")

    def test_only_a_record_of_the_contract_names_rules(self) -> None:
        self.assertEqual(open_rules_of({}), frozenset())
        self.assertEqual(open_rules_of({"open_rules": {"contract": "other", "rules": list(OPEN_RULES)}}), frozenset())
        self.assertEqual(open_rules_of({"open_rules": open_rules_record(True)}), frozenset(OPEN_RULES))
        self.assertEqual(open_rules_record(True), {"contract": OPEN_RULES_CONTRACT, "rules": list(OPEN_RULES)})


class WithoutTheOptionTests(unittest.TestCase):
    def test_every_rule_holds(self) -> None:
        env = _env()
        root = _started(env)
        self.assertNotIn("open_rules", env.current_state())
        self.assertNotIn("open_rules", env.get_policy_observation().as_dict())
        self.assertEqual(open_rules_of(env.current_state()), frozenset())
        # The handoff waits for the listed corrections.
        output = _refused(env, ASK_FOR_MORE_EVIDENCE, state_id=root, request=RECOVERY_OPTIONS_EXHAUSTED_REQUEST)
        self.assertEqual(output["error_code"], "operator_escalation_precondition_not_met")
        self.assertIn("same_state_supported_corrections_unexhausted", output["error_detail"])
        # A correction the context does not list is refused.
        output = _refused(env, CORRECT_PARAMETERS, state_id=root, line_index=1)
        self.assertEqual(output["error_code"], "correction_not_supported_by_current_context")
        # A rejected candidate can only be rolled back.
        candidate, verdict = _verified(env, root, 4)
        self.assertEqual(verdict, "REJECT")
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [ROLLBACK_STATE])
        output = _refused(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(output["error_detail"], "commit_rejected_or_inconclusive_candidate")
        self.assertEqual(env.store.get_state(candidate)["status"], "candidate")

    def test_a_closed_route_refuses_its_corrections(self) -> None:
        env = _env(parameter_context=_empty_context)
        root = _started(env)
        output = _refused(env, CORRECT_PARAMETERS, state_id=root, line_index=6)
        self.assertEqual(output["error_code"], "correction_route_not_actionable")

    def test_the_store_refuses_a_rejected_candidate_unless_kept(self) -> None:
        store = PowerSystemStateStore()
        root = store.create_root({"branch": [{"x": 1.0}]}, [1.0], episode_id="e")
        child = store.clone_candidate(root, {"line_index": 1, "value": 0.5}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        # Unverified: nothing to keep.
        with self.assertRaises(StateStoreError):
            store.commit(child, kept=True)
        store.mark_verified(child, {}, "REJECT")
        with self.assertRaises(StateStoreError):
            store.commit(child)
        self.assertEqual(store.commit(child, kept=True), child)
        self.assertEqual(store.active_state_id, child)
        # The record still shows what the check said.
        self.assertEqual(store.get_state(child)["candidate_disposition"], "REJECT")


class WithTheOptionTests(unittest.TestCase):
    def test_the_observation_names_the_lifted_rules(self) -> None:
        env = _env(True)
        _started(env)
        record = {"contract": OPEN_RULES_CONTRACT, "rules": list(OPEN_RULES)}
        self.assertEqual(env.current_state()["open_rules"], record)
        self.assertEqual(env.get_policy_observation().as_dict()["open_rules"], record)
        self.assertEqual(open_rules_of(env.current_state()), frozenset(OPEN_RULES))

    def test_a_correction_the_context_does_not_list_runs_and_can_be_kept(self) -> None:
        env = _env(True)
        root = _started(env)
        candidate, verdict = _verified(env, root, 1)
        # The tool computed its value and the verification still says what it found.
        self.assertEqual(verdict, "REJECT")
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [ROLLBACK_STATE, COMMIT_STATE])
        output = _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertIs(output["tool_metrics"].get("kept"), True)
        self.assertEqual(env.store.active_state_id, candidate)
        self.assertEqual(env.store.get_state(candidate)["candidate_disposition"], "REJECT")
        accepted = env.current_state()["accepted_corrections"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source_action"]["arguments"]["line_index"], 1)
        self.assertEqual(env.current_state()["rejected_hypotheses"], [])

    def test_a_rejected_candidate_can_still_be_rolled_back(self) -> None:
        env = _env(True)
        root = _started(env)
        candidate, verdict = _verified(env, root, 4)
        self.assertEqual(verdict, "REJECT")
        output = _ok(env, ROLLBACK_STATE, candidate_state_id=candidate)
        self.assertNotIn("kept", output["tool_metrics"])
        self.assertEqual(env.store.active_state_id, root)
        self.assertEqual(len(env.current_state()["rejected_hypotheses"]), 1)

    def test_an_accepted_candidate_is_as_before(self) -> None:
        env = _env(True)
        root = _started(env)
        candidate, verdict = _verified(env, root, 3)
        self.assertEqual(verdict, "ACCEPT_PARTIAL")
        # The commit rule opens nothing on an accepted candidate; declining it is the lifecycle choice's business.
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [COMMIT_STATE])
        output = _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertNotIn("kept", output["tool_metrics"])

    def test_a_closed_route_can_be_tested(self) -> None:
        env = _env(OPEN_RULE_CORRECTION_ON_CLOSED_ROUTE, parameter_context=_empty_context)
        root = _started(env)
        candidate, verdict = _verified(env, root, 6)
        self.assertEqual(verdict, "ACCEPT_FINAL")
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)

    def test_the_handoff_no_longer_waits_for_the_listed_corrections(self) -> None:
        env = _env(OPEN_RULE_HANDOFF_WITHOUT_EXHAUSTION)
        root = _started(env)
        _ok(env, ASK_FOR_MORE_EVIDENCE, state_id=root, request=RECOVERY_OPTIONS_EXHAUSTED_REQUEST)
        self.assertTrue(env.terminal)
        self.assertEqual(env.terminal_outcome, "operator_escalation")

    def test_a_correction_may_follow_a_quiet_estimator(self) -> None:
        # The toy never mints the confirmation marker, so the gate is read on a
        # copy of the state after the alarm cleared, with the marker set.
        env = _env()
        root = _started(env)
        candidate, verdict = _verified(env, root, 6)
        self.assertEqual(verdict, "ACCEPT_FINAL")
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        active = env.store.active_state_id
        _ok(env, GET_PARAMETER_CONTEXT, state_id=active)
        state = dict(env.current_state())
        state["unresolved_signatures"] = [POST_CORRECTION_CONFIRMATION_SIGNATURE]
        self.assertTrue(post_correction_confirmation_required(state))
        action = {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active, "line_index": 3}}
        verdict = env.process_oracle.check(state, action, store=env.store)
        self.assertFalse(verdict["process_valid"])
        self.assertEqual(verdict["error_code"], "post_correction_confirmation_required")
        state["open_rules"] = open_rules_record(OPEN_RULE_COMMIT_REJECTED_CANDIDATE)
        self.assertFalse(env.process_oracle.check(state, action, store=env.store)["process_valid"])
        state["open_rules"] = open_rules_record(OPEN_RULE_CORRECTION_AFTER_ESTIMATOR_PASSES)
        self.assertTrue(env.process_oracle.check(state, action, store=env.store)["process_valid"])

    def test_only_the_named_rules_are_lifted(self) -> None:
        env = _env(OPEN_RULE_COMMIT_REJECTED_CANDIDATE)
        root = _started(env)
        self.assertEqual(env.current_state()["open_rules"]["rules"], [OPEN_RULE_COMMIT_REJECTED_CANDIDATE])
        self.assertEqual(_refused(env, CORRECT_PARAMETERS, state_id=root, line_index=1)["error_code"],
                         "correction_not_supported_by_current_context")
        self.assertEqual(_refused(env, ASK_FOR_MORE_EVIDENCE, state_id=root, request=RECOVERY_OPTIONS_EXHAUSTED_REQUEST)["error_code"],
                         "operator_escalation_precondition_not_met")
        candidate, verdict = _verified(env, root, 4)
        self.assertEqual(verdict, "REJECT")
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)

        env = _env([OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT])
        root = _started(env)
        candidate, verdict = _verified(env, root, 1)
        self.assertEqual(verdict, "REJECT")
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [ROLLBACK_STATE])
        self.assertEqual(_refused(env, COMMIT_STATE, candidate_state_id=candidate)["error_detail"],
                         "commit_rejected_or_inconclusive_candidate")

    def test_a_commit_with_nothing_verified_stays_refused(self) -> None:
        env = _env(True)
        root = _started(env)
        output = _refused(env, COMMIT_STATE, candidate_state_id=root)
        self.assertEqual(output["error_detail"], "commit_without_verified_candidate")
        candidate = _ok(env, CORRECT_PARAMETERS, state_id=root, line_index=4)["candidate_state_id"]
        output = _refused(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(output["error_detail"], "commit_without_verified_candidate")


class AccountingTests(unittest.TestCase):
    def test_a_kept_candidate_is_not_a_false_commit_only_where_the_rule_is_lifted(self) -> None:
        opened = {"open_rules": open_rules_record(OPEN_RULE_COMMIT_REJECTED_CANDIDATE)}
        other = {"open_rules": open_rules_record(OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT)}
        for verdict in ("REJECT", "INCONCLUSIVE"):
            self.assertTrue(commit_keeps_rejected_candidate(opened, verdict))
            self.assertFalse(commit_keeps_rejected_candidate({}, verdict))
            self.assertFalse(commit_keeps_rejected_candidate(other, verdict))
        # A commit with no verified candidate open is a false commit everywhere.
        self.assertFalse(commit_keeps_rejected_candidate(opened, None))

    def test_a_config_may_name_the_option(self) -> None:
        from psse_env.rl.train import ENVIRONMENT_OPTION_KEYS

        self.assertIn("open_rules", ENVIRONMENT_OPTION_KEYS)

    def test_the_environment_names_the_lifted_rules_an_action_needed(self) -> None:
        env = _env(True)
        root = _started(env)
        # A context call needs none.
        self.assertEqual(env.last_open_rules_used, [])
        candidate = _ok(env, CORRECT_PARAMETERS, state_id=root, line_index=1)["candidate_state_id"]
        self.assertEqual(env.last_open_rules_used, [OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT])
        _ok(env, RUN_WLS, state_id=candidate)
        self.assertEqual(env.last_open_rules_used, [])
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(env.last_open_rules_used, [OPEN_RULE_COMMIT_REJECTED_CANDIDATE])
        # A refused action used nothing: the report is about the action just dispatched.
        _refused(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(env.last_open_rules_used, [])

    def test_a_listed_correction_and_an_accepted_commit_need_no_lifted_rule(self) -> None:
        env = _env(True)
        root = _started(env)
        candidate, verdict = _verified(env, root, 3)
        self.assertEqual(verdict, "ACCEPT_PARTIAL")
        self.assertEqual(env.last_open_rules_used, [])
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(env.last_open_rules_used, [])

    def test_a_correction_on_a_closed_route_names_that_rule(self) -> None:
        env = _env(True, parameter_context=_empty_context)
        root = _started(env)
        _ok(env, CORRECT_PARAMETERS, state_id=root, line_index=6)
        self.assertEqual(env.last_open_rules_used, [OPEN_RULE_CORRECTION_ON_CLOSED_ROUTE])

    def test_the_handoff_names_its_rule_only_where_something_was_waived(self) -> None:
        env = _env(True)
        root = _started(env)
        output = _ok(env, ASK_FOR_MORE_EVIDENCE, state_id=root, request=RECOVERY_OPTIONS_EXHAUSTED_REQUEST)
        self.assertEqual(env.last_open_rules_used, [OPEN_RULE_HANDOFF_WITHOUT_EXHAUSTION])
        self.assertIn("same_state_supported_corrections_unexhausted",
                      output["tool_metrics"]["operator_escalation_audit"]["exhaustion_waived"])
        # Nothing listed and every context read: the handoff is the controller's own, with or without the option.
        for rules in (None, True):
            env = _env(rules, parameter_context=_empty_context)
            root = _started(env)
            _ok(env, GET_MEASUREMENT_CONTEXT, state_id=root)
            _ok(env, GET_TOPOLOGY_CONTEXT, state_id=root)
            output = _ok(env, ASK_FOR_MORE_EVIDENCE, state_id=root, request=RECOVERY_OPTIONS_EXHAUSTED_REQUEST)
            self.assertEqual(env.last_open_rules_used, [])
            self.assertNotIn("exhaustion_waived", output["tool_metrics"]["operator_escalation_audit"])

    def test_without_the_option_nothing_is_reported(self) -> None:
        env = _env()
        root = _started(env)
        self.assertEqual(env.last_open_rules_used, [])
        _refused(env, CORRECT_PARAMETERS, state_id=root, line_index=1)
        self.assertEqual(env.last_open_rules_used, [])

    def test_the_evaluator_copies_the_report_onto_the_step_and_the_episode(self) -> None:
        from psse_env.dagger.test_evaluator import _resolved_scenario, _ScriptEnv, _ScriptPolicy

        class _Reporting(_ScriptEnv):
            def step(self, action):
                self.last_open_rules_used = list(self._current_script().get("open_rules_used") or [])
                return super().step(action)

        def episode(env_factory, scenario):
            evaluator = ClosedLoopRolloutEvaluator(env_factory=env_factory, policy_factory=lambda: _ScriptPolicy([]),
                                                   max_steps=8, seed=71)
            return evaluator.evaluate([scenario]).suite_metrics["episodes"][0]

        scenario = _resolved_scenario()
        scenario["script"][2]["open_rules_used"] = [OPEN_RULE_COMMIT_REJECTED_CANDIDATE]
        told = episode(_Reporting, scenario)
        self.assertEqual([step.get("open_rules_used") for step in told["trace"]],
                         [None, None, [OPEN_RULE_COMMIT_REJECTED_CANDIDATE], None, None])
        self.assertEqual(told["open_rule_accounting"], {
            "counts": {OPEN_RULE_COMMIT_REJECTED_CANDIDATE: 1},
            "events": [{"step": 2, "tool": COMMIT_STATE, "rules": [OPEN_RULE_COMMIT_REJECTED_CANDIDATE]}]})
        # An environment that reports nothing leaves every record as it was.
        plain = episode(_ScriptEnv, _resolved_scenario())
        self.assertEqual(plain["open_rule_accounting"], {})
        self.assertFalse(any("open_rules_used" in step for step in plain["trace"]))


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = unified_tool_schemas()
        closed = _env()
        _started(closed)
        self.closed = closed.current_state()
        self.plain = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, self.closed)

    def _observation(self, open_rules, lifecycle_choice: bool = False):
        env = _env(open_rules, lifecycle_choice=lifecycle_choice)
        _started(env)
        return env.current_state()

    def test_the_prompt_and_the_commit_schema_say_so_only_where_rules_are_lifted(self) -> None:
        self.assertNotIn("Open rules", self.plain)
        default = _schema(tool_schemas_for_observation(self.tools, self.closed), COMMIT_STATE)["description"]
        self.assertEqual(default, _schema(self.tools, COMMIT_STATE)["description"])
        opened = self._observation(True)
        text = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, opened)
        self.assertEqual(text, self.plain + open_rules_prompt_paragraph(OPEN_RULES))
        for rule in OPEN_RULES:
            self.assertIn(OPEN_RULES_PROMPT_SENTENCES[rule], text)
        self.assertEqual(_schema(tool_schemas_for_observation(self.tools, opened), COMMIT_STATE)["description"],
                         OPEN_RULES_COMMIT_DESCRIPTION)
        # The shared schema list is never edited in place.
        self.assertEqual(_schema(self.tools, COMMIT_STATE)["description"], default)
        self.assertEqual([t["function"]["name"] for t in tool_schemas_for_observation(self.tools, opened)],
                         [t["function"]["name"] for t in tool_schemas_for_observation(self.tools, self.closed)])

    def test_only_the_lifted_rules_are_named(self) -> None:
        opened = self._observation(OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT)
        text = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, opened)
        self.assertIn(OPEN_RULES_PROMPT_SENTENCES[OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT], text)
        for rule in OPEN_RULES[1:]:
            self.assertNotIn(OPEN_RULES_PROMPT_SENTENCES[rule], text)
        # The commit schema changes only with the commit rule.
        self.assertEqual(_schema(tool_schemas_for_observation(self.tools, opened), COMMIT_STATE)["description"],
                         _schema(self.tools, COMMIT_STATE)["description"])
        self.assertEqual(open_rules_prompt_paragraph(()), "")

    def test_the_lifecycle_paragraph_stops_binding_a_rejected_candidate_to_the_rollback(self) -> None:
        self.assertTrue(LIFECYCLE_CHOICE_PROMPT_PARAGRAPH.endswith(LIFECYCLE_CHOICE_ROLLBACK_ONLY_SENTENCE))
        # The lifecycle choice alone: its paragraph as it was.
        alone = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, self._observation(None, lifecycle_choice=True))
        self.assertEqual(alone, self.plain + LIFECYCLE_CHOICE_PROMPT_PARAGRAPH)
        # With correction rules only, the sentence still holds and stays.
        corrections = system_prompt_for_observation(
            CANONICAL_DAGGER_SYSTEM_PROMPT, self._observation(OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT, lifecycle_choice=True))
        self.assertEqual(corrections, self.plain + LIFECYCLE_CHOICE_PROMPT_PARAGRAPH
                         + open_rules_prompt_paragraph([OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT]))
        # With the commit rule it no longer holds and is dropped.
        both = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, self._observation(True, lifecycle_choice=True))
        self.assertEqual(both, self.plain + LIFECYCLE_CHOICE_PROMPT_PARAGRAPH.removesuffix(LIFECYCLE_CHOICE_ROLLBACK_ONLY_SENTENCE)
                         + open_rules_prompt_paragraph(OPEN_RULES))
        self.assertNotIn("can only be rolled back", both)


if __name__ == "__main__":
    unittest.main()
