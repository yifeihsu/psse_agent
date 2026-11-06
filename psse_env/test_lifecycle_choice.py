"""The lifecycle choice (RL plan of 2026-11-05): the policy may decline a candidate the verifier accepted.

The toy network of ``test_gated_revert``: correcting line 3 lowers the WLS
objective without clearing it, so the verifier accepts it as a partial step,
although only line 6 clears the alarm.  Without the option the verdict fixes
the step after the verification, as it always did; with it the policy may
roll the accepted candidate back, and the environment records a decline.
"""
from __future__ import annotations

import unittest

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_PARAMETERS,
    DECLINED_REJECTION_KIND,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    LIFECYCLE_CHOICE_CONTRACT,
    ROLLBACK_STATE,
    RUN_WLS,
    lifecycle_choice_of,
)
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    LIFECYCLE_CHOICE_PROMPT_PARAGRAPH,
    LIFECYCLE_CHOICE_ROLLBACK_DESCRIPTION,
    system_prompt_for_observation,
    tool_schemas_for_observation,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.evidence_profile import SCADA_ONLY_PROFILE
from psse_env.oracle.process_validity import ProcessValidityOracle
from psse_env.state_store import PowerSystemStateStore, StateStoreError
from psse_env.test_gated_revert import SCENARIO, _correct_parameters, _empty_context, _ok, _parameter_context, _Quality, _wls
from psse_env.transactional_env import TransactionalPSSEEnv


def _env(lifecycle_choice: bool) -> TransactionalPSSEEnv:
    return TransactionalPSSEEnv(
        evidence_profile=SCADA_ONLY_PROFILE,
        process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True),
        wls_runner=_wls,
        candidate_quality_oracle=_Quality(),
        context_providers={GET_MEASUREMENT_CONTEXT: _empty_context, GET_PARAMETER_CONTEXT: _parameter_context,
                           GET_TOPOLOGY_CONTEXT: _empty_context},
        correction_executors={CORRECT_PARAMETERS: _correct_parameters},
        max_steps=60,
        lifecycle_choice=lifecycle_choice,
    )


def _verified(env: TransactionalPSSEEnv, line: int) -> tuple[str, str, str]:
    """Root with WLS and the parameter context, then ``line`` corrected and verified; (root, candidate, verdict)."""
    root = env.reset(SCENARIO)["active_state_id"]
    _ok(env, RUN_WLS, state_id=root)
    _ok(env, GET_PARAMETER_CONTEXT, state_id=root)
    candidate = _ok(env, CORRECT_PARAMETERS, state_id=root, line_index=line)["candidate_state_id"]
    _ok(env, RUN_WLS, state_id=candidate)
    return root, candidate, env.store.get_state(candidate)["candidate_disposition"]


def _rollback_name(schemas) -> dict:
    from psse_env.dagger.protocol_bridge import CANONICAL_TO_INTERNAL_TOOL

    for tool in schemas:
        name = str(tool["function"]["name"])
        if CANONICAL_TO_INTERNAL_TOOL.get(name, name) == ROLLBACK_STATE:
            return tool["function"]
    raise AssertionError("no rollback schema")


class WithoutTheOptionTests(unittest.TestCase):
    def test_the_verdict_still_fixes_the_step(self) -> None:
        env = _env(False)
        root, candidate, verdict = _verified(env, 3)
        self.assertEqual(verdict, "ACCEPT_PARTIAL")
        observation = env.current_state()
        self.assertNotIn("lifecycle_choice", observation)
        self.assertIsNone(lifecycle_choice_of(observation))
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [COMMIT_STATE])
        _, output = env.step({"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "candidate_lifecycle_violation")
        self.assertEqual(output["error_detail"], "rollback_accepted_candidate")
        self.assertEqual(env.store.get_state(candidate)["status"], "candidate")
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(env.store.active_state_id, candidate)
        self.assertNotEqual(root, candidate)

    def test_the_store_refuses_an_accepted_candidate_unless_declined(self) -> None:
        store = PowerSystemStateStore()
        root = store.create_root({"branch": [{"x": 1.0}]}, [1.0], episode_id="e")
        child = store.clone_candidate(root, {"line_index": 1, "value": 0.5}, {"tool": CORRECT_PARAMETERS, "arguments": {}})
        store.mark_verified(child, {}, "ACCEPT_PARTIAL")
        with self.assertRaises(StateStoreError):
            store.rollback(child)
        self.assertEqual(store.rollback(child, declined=True), root)
        self.assertEqual(store.get_state(child)["status"], "rolled_back")
        self.assertEqual(store.get_state(child)["candidate_disposition"], "ACCEPT_PARTIAL")
        self.assertEqual(store.active_state_id, root)


class WithTheOptionTests(unittest.TestCase):
    def test_an_accepted_candidate_can_be_declined(self) -> None:
        env = _env(True)
        root, candidate, verdict = _verified(env, 3)
        self.assertEqual(verdict, "ACCEPT_PARTIAL")
        observation = env.current_state()
        record = lifecycle_choice_of(observation)
        self.assertIsNotNone(record)
        self.assertEqual(record["contract"], LIFECYCLE_CHOICE_CONTRACT)
        self.assertEqual(record["accepted_candidate"], [COMMIT_STATE, ROLLBACK_STATE])
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [COMMIT_STATE, ROLLBACK_STATE])
        root_hash = env.store.state_hash(root)
        output = _ok(env, ROLLBACK_STATE, candidate_state_id=candidate)
        self.assertIs(output["tool_metrics"].get("declined"), True)
        self.assertEqual(env.store.active_state_id, root)
        self.assertEqual(env.store.state_hash(root), root_hash)
        rejected = env.current_state()["rejected_hypotheses"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["rejection_kind"], DECLINED_REJECTION_KIND)
        self.assertEqual(rejected[0]["source_action"]["arguments"]["line_index"], 3)
        self.assertIn("global_progress", rejected[0]["verification_summary"])
        self.assertEqual(env.current_state()["accepted_corrections"], [])
        # Another hypothesis can be tested from the same state, and committed.
        other = _ok(env, CORRECT_PARAMETERS, state_id=root, line_index=6)["candidate_state_id"]
        _ok(env, RUN_WLS, state_id=other)
        self.assertEqual(env.store.get_state(other)["candidate_disposition"], "ACCEPT_FINAL")
        _ok(env, COMMIT_STATE, candidate_state_id=other)
        self.assertEqual(env.store.active_state_id, other)

    def test_committing_an_accepted_candidate_is_unchanged(self) -> None:
        env = _env(True)
        _root, candidate, _verdict = _verified(env, 3)
        _ok(env, COMMIT_STATE, candidate_state_id=candidate)
        self.assertEqual(env.store.active_state_id, candidate)
        self.assertEqual(len(env.current_state()["accepted_corrections"]), 1)
        self.assertEqual(env.current_state()["rejected_hypotheses"], [])

    def test_a_rejected_candidate_still_only_rolls_back_and_names_no_kind(self) -> None:
        env = _env(True)
        root, candidate, verdict = _verified(env, 4)
        self.assertEqual(verdict, "REJECT")
        self.assertEqual([a["tool"] for a in env.enumerate_available_actions()], [ROLLBACK_STATE])
        _, output = env.step({"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_detail"], "commit_rejected_or_inconclusive_candidate")
        output = _ok(env, ROLLBACK_STATE, candidate_state_id=candidate)
        self.assertNotIn("declined", output["tool_metrics"])
        self.assertEqual(env.store.active_state_id, root)
        self.assertNotIn("rejection_kind", env.current_state()["rejected_hypotheses"][0])


class PromptTests(unittest.TestCase):
    def test_the_prompt_and_the_rollback_schema_say_so_only_where_the_step_is_open(self) -> None:
        tools = unified_tool_schemas()
        closed = _env(False)
        _verified(closed, 3)
        opened = _env(True)
        _verified(opened, 3)
        closed_observation, open_observation = closed.current_state(), opened.current_state()
        plain = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, closed_observation)
        self.assertNotIn("Lifecycle choice", plain)
        self.assertEqual(system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, open_observation),
                         plain + LIFECYCLE_CHOICE_PROMPT_PARAGRAPH)
        default = _rollback_name(tool_schemas_for_observation(tools, closed_observation))["description"]
        self.assertEqual(default, _rollback_name(tools)["description"])
        self.assertEqual(_rollback_name(tool_schemas_for_observation(tools, open_observation))["description"],
                         LIFECYCLE_CHOICE_ROLLBACK_DESCRIPTION)
        # The shared schema list is never edited in place.
        self.assertEqual(_rollback_name(tools)["description"], default)
        self.assertEqual([t["function"]["name"] for t in tool_schemas_for_observation(tools, open_observation)],
                         [t["function"]["name"] for t in tool_schemas_for_observation(tools, closed_observation)])


if __name__ == "__main__":
    unittest.main()
