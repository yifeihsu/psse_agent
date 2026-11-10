"""The complete-history observation mode (workplan of 2026-11-10, stage 1), on a synthetic episode.

The episode: a solve, a parameter context, a correction that opens a
candidate, the candidate's solve and verification, then a rollback (the
policy declines) or a commit, then more actions on the resulting active case.
State ids follow the controller's ``<episode>:sN`` form so the alias checks
exercise the real substitution.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import unittest

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_PARAMETERS,
    GET_PARAMETER_CONTEXT,
    ROLLBACK_STATE,
    RUN_WLS,
    VERIFY_CANDIDATE,
)
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    COMPLETE_HISTORY_PROMPT_PARAGRAPH,
    NEWEST_EVENT_EVIDENCE,
    _compact_last_tool_output,
    alias_model_visible_state,
    candidate_experiments,
    compact_tables,
    complete_history_events,
    expand_tables,
    find_model_identifier_leaks,
    records_to_table,
    table_to_records,
    prepare_model_policy_observation,
    system_prompt_for_observation,
    validate_policy_payload,
)
from psse_env.dagger.evaluator import STUDY_POLICY_HISTORY_WINDOW, _canonical_study_policy_observation
from psse_env.dagger.observation_modes import (
    BOUNDED_HISTORY,
    CANDIDATE_EXPERIMENTS_KEY,
    COMPLETE_HISTORY,
    HISTORY_RECORD_KEY,
    OBSERVABLE_HISTORY_KEY,
    complete_history_record,
    observable_history_from_trace,
    observable_history_of,
    restored_observation,
    stored_observation,
    validate_observation_mode,
    with_observable_history,
)

EPISODE = "episode-0001"
PARENT, CANDIDATE, SECOND = f"{EPISODE}:s0", f"{EPISODE}:s1", f"{EPISODE}:s2"


def _hash(tag: str) -> str:
    return hashlib.sha256(tag.encode("ascii")).hexdigest()


def _wls_output(state: str, statistic: float, *, bus: int, active: str = PARENT) -> dict:
    return {
        "execution_status": "success", "state_mutated": False, "active_state_id": active,
        "valid_next_actions": [{"tool": GET_PARAMETER_CONTEXT, "arguments": {"state_id": state}}],
        "tool_metrics": {
            "chi_square_statistic": statistic, "chi_square_threshold": 60.0, "chi_square_alarm": statistic > 60.0,
            "state_hash": _hash(state),
            "wls_summary": {
                "top_residuals": [{"index": 7, "value": 5.1}],
                "bus_table": [{"bus": bus, "type": "PQ", "vm": -3.2, "p": 4.4, "q": None},
                              {"bus": bus + 1, "type": "PV", "vm": 0.4, "p": 1.1, "q": 0.2}],
                "branch_table": [{"line": 9, "from": bus, "to": bus + 1, "pf": 6.1, "qf": 0.2, "pt": -5.9, "qt": 0.1, "lr": 7.5, "lx": 2.2},
                                 {"line": 10, "from": bus + 1, "to": bus + 2, "pf": 0.3, "qf": 0.1, "pt": -0.2, "qt": 0.0, "lr": 0.9,
                                  "lx": 0.4, "xfmr": True}],
                "omitted": {"buses": 0, "branches": 0},
            },
        },
    }


def _event(active: str, candidate: str | None, tool: str, arguments: dict, output: dict) -> dict:
    return {"state_id": active, "candidate_state_id": candidate, "action": {"tool": tool, "arguments": arguments},
            "tool_output": output}


def _prefix() -> list[dict]:
    """Solve, context, correction, candidate solve, verification: a verified candidate is open."""

    return [
        _event(PARENT, None, RUN_WLS, {"state_id": PARENT}, _wls_output(PARENT, 180.0, bus=4)),
        _event(PARENT, None, GET_PARAMETER_CONTEXT, {"state_id": PARENT}, {
            "execution_status": "success", "state_mutated": False,
            "tool_metrics": {"context_tool": GET_PARAMETER_CONTEXT, "finding_count": 2,
                             "supported_corrections": [{"tool": CORRECT_PARAMETERS, "arguments": {"state_id": PARENT, "line_index": 9}}]}}),
        _event(PARENT, None, CORRECT_PARAMETERS, {"state_id": PARENT, "line_index": 9}, {
            "execution_status": "success", "state_mutated": True, "active_state_id": PARENT, "candidate_state_id": CANDIDATE,
            "tool_metrics": {"detailed_evidence": {"contract": "detailed_evidence_v1",
                                                   "edit": [{"line": 9, "r": {"before": 0.03, "after": 0.05, "ratio": 1.667},
                                                             "x": {"before": 0.1, "after": 0.12, "ratio": 1.2}}]}}}),
        _event(PARENT, CANDIDATE, RUN_WLS, {"state_id": CANDIDATE}, _wls_output(CANDIDATE, 41.0, bus=5)),
        _event(PARENT, CANDIDATE, VERIFY_CANDIDATE, {"state_id": CANDIDATE}, {
            "execution_status": "success", "state_mutated": False, "candidate_state_id": CANDIDATE,
            "tool_metrics": {"target_progress": 0.97, "global_progress": 0.77, "globally_resolved": False,
                             "detailed_evidence": {"contract": "detailed_evidence_v1",
                                                   "target_neighbourhood": {"own_metric": {"before": 7.5, "after": 0.4}}}}}),
    ]


def _declined() -> list[dict]:
    return _prefix() + [
        _event(PARENT, CANDIDATE, ROLLBACK_STATE, {"candidate_state_id": CANDIDATE},
               {"execution_status": "success", "state_mutated": True, "active_state_id": PARENT,
                "tool_metrics": {"restored_parent_hash": _hash(PARENT)}}),
        _event(PARENT, None, GET_PARAMETER_CONTEXT, {"state_id": PARENT},
               {"execution_status": "success", "state_mutated": False, "tool_metrics": {"context_tool": GET_PARAMETER_CONTEXT, "finding_count": 1}}),
    ]


def _committed() -> list[dict]:
    # A commit makes the candidate the active case under its own id.
    return _prefix() + [
        _event(PARENT, CANDIDATE, COMMIT_STATE, {"candidate_state_id": CANDIDATE},
               {"execution_status": "success", "state_mutated": True, "active_state_id": CANDIDATE,
                "valid_next_actions": [{"tool": RUN_WLS, "arguments": {"state_id": CANDIDATE}}]}),
        _event(CANDIDATE, None, RUN_WLS, {"state_id": CANDIDATE}, _wls_output(CANDIDATE, 41.0, bus=5, active=CANDIDATE)),
    ]


def _observation(history: list[dict], *, active: str, candidate: str | None = None) -> dict:
    last = history[-1] if history else None
    return {
        "active_state_id": active, "candidate_state_id": candidate, "episode_id": EPISODE, "evidence_profile": "scada_only",
        "remaining_budget": 40 - len(history), "accepted_corrections": [], "rejected_hypotheses": [],
        "last_tool": last["action"]["tool"] if last else None,
        "last_tool_status": last["tool_output"]["execution_status"] if last else None,
        "last_tool_output": copy.deepcopy(last["tool_output"]) if last else {},
        "history_window": copy.deepcopy(history[-STUDY_POLICY_HISTORY_WINDOW:]),
    }


def _views(history: list[dict], *, active: str, candidate: str | None = None):
    bounded = _observation(history, active=active, candidate=candidate)
    complete = with_observable_history(bounded, history)
    return prepare_model_policy_observation(bounded), prepare_model_policy_observation(complete)


class ObservationModeTests(unittest.TestCase):
    def test_modes_are_named_and_validated(self) -> None:
        self.assertEqual(validate_observation_mode(BOUNDED_HISTORY), "bounded_history_v1")
        self.assertEqual(validate_observation_mode(COMPLETE_HISTORY), "complete_history_v1")
        with self.assertRaises(ValueError):
            validate_observation_mode("full")

    def test_the_bounded_mode_renders_what_it_always_rendered(self) -> None:
        history = _declined()
        view, control = prepare_model_policy_observation(_observation(history, active=PARENT))
        # The digest of the view as the code before the complete mode rendered it (recorded 2026-11-10).
        digest = hashlib.sha256(json.dumps(view, sort_keys=True).encode("utf-8")).hexdigest()
        self.assertEqual(digest, "84c853dbb003aec8d80e18d11bbcb1e2399167bac7769bc331b4794d21e07b32")
        self.assertEqual(len(view["history_window"]), STUDY_POLICY_HISTORY_WINDOW)
        self.assertNotIn(HISTORY_RECORD_KEY, view)
        self.assertNotIn(CANDIDATE_EXPERIMENTS_KEY, view)
        self.assertNotIn("observation_mode", control)
        # No history event of the bounded view carries a solve's tables.
        for event in view["history_window"]:
            self.assertNotIn("bus_table", (event.get("observable_metrics") or {}).get("wls_summary") or {})

    def test_an_early_table_and_the_candidate_edit_stay_visible(self) -> None:
        history = _declined()
        (bounded, _), (complete, control) = _views(history, active=PARENT)
        self.assertEqual(control["observation_mode"], COMPLETE_HISTORY)
        events = complete["history_window"]
        self.assertEqual([event["index"] for event in events], list(range(len(history))))
        self.assertEqual(complete[HISTORY_RECORD_KEY], {"contract": COMPLETE_HISTORY, "events": len(history)})
        # The first solve, six actions back, still shows its tables; the bounded view dropped the event.
        written = events[0]["observable_metrics"]["wls_summary"]
        self.assertEqual(written["bus_table"], {"columns": ["bus", "type", "vm", "p", "q"],
                                                "rows": [[4, "PQ", -3.2, 4.4, None], [5, "PV", 0.4, 1.1, 0.2]]})
        self.assertEqual(written["branch_table"]["optional"], ["xfmr"])
        first = expand_tables(events[0]["observable_metrics"])["wls_summary"]
        self.assertEqual(first["bus_table"][0]["bus"], 4)
        self.assertEqual(first["branch_table"][0]["lr"], 7.5)
        self.assertEqual(first["branch_table"][1]["xfmr"], True)
        self.assertNotIn("xfmr", first["branch_table"][0])
        self.assertEqual(events[2]["observable_metrics"]["detailed_evidence"]["edit"][0]["r"]["ratio"], 1.667)
        self.assertEqual(events[4]["observable_metrics"]["detailed_evidence"]["target_neighbourhood"]["own_metric"]["after"], 0.4)
        self.assertEqual(len(bounded["history_window"]), STUDY_POLICY_HISTORY_WINDOW)
        # The bounded view holds neither that solve nor any table, and the correction's edit left it one action ago.
        self.assertNotIn("bus_table", json.dumps(bounded["history_window"]))
        self.assertNotIn("1.667", json.dumps(bounded))

    def test_every_event_carries_the_view_it_had_as_the_last_tool_output(self) -> None:
        history = _committed()
        (_bounded, _), (complete, control) = _views(history, active=CANDIDATE)
        events = complete["history_window"]
        for raw, event in zip(history, events):
            shown_then = _compact_last_tool_output(raw["tool_output"]).get("observable_metrics") or {}
            expected = alias_model_visible_state(shown_then, control["state_aliases"], episode_aliases=control["episode_aliases"],
                                                 hash_aliases=control["hash_aliases"])
            if event is events[-1]:
                # The newest event's evidence is the last tool output; it is not written twice.
                self.assertEqual(event["observable_metrics"], NEWEST_EVENT_EVIDENCE)
                self.assertEqual(complete["last_tool_output"]["observable_metrics"], expected)
            else:
                self.assertEqual(expand_tables(event.get("observable_metrics") or {}), expected)
            # What authorizes the next action is the last tool output's alone.
            self.assertNotIn("valid_next_actions", event)
            self.assertNotIn("valid_next_actions", event["outcome"])
        self.assertIn("valid_next_actions", complete["last_tool_output"])

    def test_a_decline_keeps_the_evidence_and_makes_the_parent_current(self) -> None:
        (_bounded, _), (complete, control) = _views(_declined(), active=PARENT)
        events = complete["history_window"]
        self.assertEqual(control["state_aliases"], {"active": PARENT, "s1": CANDIDATE})
        # Evidence on the parent is evidence on the active case again; the declined candidate is an earlier case.
        self.assertEqual([event["arguments"].get("case_path") for event in events],
                         ["active", "active", "active", "s1", "s1", "s1", "active"])
        self.assertEqual({event["state_id"] for event in events}, {"active"})
        self.assertEqual(events[2]["outcome"]["candidate_state_id"], "s1")
        # The parent's solve and the rollback's restored state carry one hash alias: the same physical state.
        self.assertEqual(events[0]["observable_metrics"]["state_hash"], events[5]["observable_metrics"]["restored_parent_hash"])
        self.assertNotEqual(events[0]["observable_metrics"]["state_hash"], events[3]["observable_metrics"]["state_hash"])
        self.assertEqual(complete[CANDIDATE_EXPERIMENTS_KEY], [{
            "candidate_state_id": "s1", "parent_state_id": "active",
            "correction": {"tool": "correct_parameters_from_path", "arguments": {"line_index": 9}},
            "opened_at": 2, "examined_at": [3, 4], "disposition": "rolled_back", "closed_at": 5,
        }])

    def test_a_commit_moves_the_aliases_without_erasing_anything(self) -> None:
        history = _committed()
        (_bounded, _), (complete, control) = _views(history, active=CANDIDATE)
        events = complete["history_window"]
        self.assertEqual(len(events), len(history))
        self.assertEqual(control["state_aliases"], {"active": CANDIDATE, "s0": PARENT})
        # The committed candidate's evidence now describes the active case; the old parent is an earlier case.
        self.assertEqual([event["arguments"].get("case_path") for event in events],
                         ["s0", "s0", "s0", "active", "active", "active", "active"])
        self.assertEqual([event["state_id"] for event in events], ["s0"] * 6 + ["active"])
        self.assertEqual(events[5]["outcome"]["active_state_id"], "active")
        experiment = complete[CANDIDATE_EXPERIMENTS_KEY][0]
        self.assertEqual((experiment["candidate_state_id"], experiment["parent_state_id"], experiment["disposition"]),
                         ("active", "s0", "committed"))

    def test_an_open_candidate_is_the_candidate_and_listed_as_open(self) -> None:
        (_bounded, _), (complete, control) = _views(_prefix(), active=PARENT, candidate=CANDIDATE)
        self.assertEqual(control["state_aliases"], {"active": PARENT, "candidate": CANDIDATE})
        self.assertEqual([event["arguments"].get("case_path") for event in complete["history_window"]],
                         ["active", "active", "active", "candidate", "candidate"])
        self.assertEqual(complete[CANDIDATE_EXPERIMENTS_KEY][0]["disposition"], "open")
        self.assertNotIn("closed_at", complete[CANDIDATE_EXPERIMENTS_KEY][0])

    def test_an_earlier_case_keeps_its_alias_for_the_rest_of_the_episode(self) -> None:
        # Decline the first candidate, then open, verify and commit a second one, and act once more.
        history = _declined()[:-1] + [
            _event(PARENT, None, CORRECT_PARAMETERS, {"state_id": PARENT, "line_index": 11},
                   {"execution_status": "success", "state_mutated": True, "candidate_state_id": SECOND}),
            _event(PARENT, SECOND, RUN_WLS, {"state_id": SECOND}, _wls_output(SECOND, 12.0, bus=6)),
            _event(PARENT, SECOND, COMMIT_STATE, {"candidate_state_id": SECOND},
                   {"execution_status": "success", "state_mutated": True, "active_state_id": SECOND}),
            _event(SECOND, None, RUN_WLS, {"state_id": SECOND}, _wls_output(SECOND, 12.0, bus=6, active=SECOND)),
        ]
        seen: dict[str, set[str]] = {}
        for length in range(1, len(history) + 1):
            shown = history[:length]
            last = shown[-1]
            after = last["tool_output"]
            active = after.get("active_state_id", last["state_id"])
            candidate = after.get("candidate_state_id", last["candidate_state_id"])
            if last["action"]["tool"] in (COMMIT_STATE, ROLLBACK_STATE):
                candidate = None
            _view, control = prepare_model_policy_observation(with_observable_history(
                _observation(shown, active=active, candidate=candidate), shown))
            for alias, identifier in control["state_aliases"].items():
                if alias not in ("active", "candidate"):
                    seen.setdefault(identifier, set()).add(alias)
        self.assertEqual(seen, {CANDIDATE: {"s1"}, PARENT: {"s0"}})

    def test_no_event_limit_and_no_character_budget(self) -> None:
        history = []
        for number in range(40):
            history.append(_event(PARENT, None, RUN_WLS, {"state_id": PARENT}, _wls_output(PARENT, 100.0 + number, bus=number + 1)))
        (bounded, _), (complete, _) = _views(history, active=PARENT)
        self.assertEqual(len(complete["history_window"]), 40)
        self.assertEqual([expand_tables(event["observable_metrics"])["wls_summary"]["bus_table"][0]["bus"]
                          for event in complete["history_window"][:-1]], list(range(1, 40)))
        self.assertEqual(complete["last_tool_output"]["observable_metrics"]["wls_summary"]["bus_table"][0]["bus"], 40)
        text = json.dumps(complete["history_window"])
        self.assertNotIn("events_omitted_before", text)
        self.assertNotIn("history omitted for size", text)
        self.assertGreater(len(text), 4096)
        self.assertLessEqual(len(bounded["history_window"]), STUDY_POLICY_HISTORY_WINDOW)

    def test_the_view_holds_no_controller_identifier(self) -> None:
        for history, active in ((_declined(), PARENT), (_committed(), CANDIDATE)):
            (_bounded, _), (complete, _control) = _views(history, active=active)
            self.assertEqual(find_model_identifier_leaks(complete), [])
            validate_policy_payload({"state": complete})

    def test_the_system_prompt_gains_its_paragraph_in_the_complete_mode_alone(self) -> None:
        (bounded, _), (complete, _) = _views(_declined(), active=PARENT)
        base = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, bounded)
        self.assertNotIn(COMPLETE_HISTORY_PROMPT_PARAGRAPH, base)
        self.assertEqual(system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, complete),
                         base + COMPLETE_HISTORY_PROMPT_PARAGRAPH)

    def test_an_observation_stored_without_its_events_is_refused_until_they_are_passed(self) -> None:
        history = _declined()
        complete = with_observable_history(_observation(history, active=PARENT), history)
        stored = stored_observation(complete)
        self.assertEqual(stored[OBSERVABLE_HISTORY_KEY], {"contract": COMPLETE_HISTORY, "event_count": len(history)})
        with self.assertRaisesRegex(ValueError, "without its events"):
            prepare_model_policy_observation(stored)
        with self.assertRaisesRegex(ValueError, "without its events"):
            observable_history_of(stored)
        rebuilt, _ = prepare_model_policy_observation(stored, history=history)
        self.assertEqual(rebuilt, prepare_model_policy_observation(complete)[0])
        self.assertIsNone(observable_history_of(_observation(history, active=PARENT)))
        self.assertIsNone(complete_history_record(_observation(history, active=PARENT)))

    def test_the_trace_gives_the_events_back(self) -> None:
        history = _declined()
        trace = []
        for position, event in enumerate(history):
            observation = with_observable_history(_observation(history[:position], active=PARENT), history[:position])
            trace.append({"step": position, "intervention": position == 0,
                          "policy_observation": None if position == 0 else stored_observation(observation),
                          "state_before": {"active_state_id": event["state_id"], "candidate_state_id": event["candidate_state_id"]},
                          "action": event["action"], "policy_tool_output": event["tool_output"]})
        self.assertEqual(observable_history_from_trace(trace), history)
        self.assertEqual(observable_history_from_trace(trace, 3), history[:3])
        self.assertIsNone(restored_observation(trace, 0))
        restored = restored_observation(trace, 4)
        self.assertEqual(observable_history_of(restored), history[:4])
        broken = copy.deepcopy(trace)
        broken[4]["policy_observation"][OBSERVABLE_HISTORY_KEY]["event_count"] = 9
        with self.assertRaisesRegex(ValueError, "records 9 events"):
            restored_observation(broken, 4)

    def test_a_table_is_the_same_records_written_once(self) -> None:
        records = [{"line": 7, "pf": 17.1, "qf": 2.9}, {"line": 4, "pf": 19.8, "qf": 6.1, "xfmr": True}]
        table = records_to_table(records)
        self.assertEqual(table, {"columns": ["line", "pf", "qf", "xfmr"], "optional": ["xfmr"],
                                 "rows": [[7, 17.1, 2.9, None], [4, 19.8, 6.1, True]]})
        self.assertEqual(table_to_records(table), records)
        # Left as records: one record, records with nested values, and a column that is both missing and null.
        for kept in ([{"line": 7}], [{"line": 7, "ends": [4, 5]}, {"line": 4, "ends": [2, 4]}],
                     [{"bus": 7, "p": None}, {"bus": 8}], [1, 2, 3], "text", {"line": 7}):
            self.assertIs(records_to_table(kept), kept)
        evidence = {"wls_summary": {"bus_table": [{"bus": 4, "p": None}, {"bus": 5, "p": 1.0}], "omitted": {"buses": 0},
                                    "top_residuals": [{"index0": 45, "value": 19.8}, {"index0": 86, "value": 18.0}]},
                    "supported_corrections": [{"tool": "correct_parameters", "arguments": {"line_index": 9}},
                                              {"tool": "correct_parameters", "arguments": {"line_index": 4}}],
                    "chi_square_statistic": 180.0}
        compact = compact_tables(evidence)
        self.assertEqual(compact["wls_summary"]["bus_table"], {"columns": ["bus", "p"], "rows": [[4, None], [5, 1.0]]})
        self.assertEqual(compact["supported_corrections"], evidence["supported_corrections"])
        self.assertEqual(expand_tables(compact), evidence)
        self.assertLess(len(json.dumps(compact_tables(_wls_output(PARENT, 1.0, bus=4)))), len(json.dumps(_wls_output(PARENT, 1.0, bus=4))))

    def test_candidate_experiments_and_events_read_the_transitions_alone(self) -> None:
        history = _declined()
        self.assertEqual(len(complete_history_events(history)), len(history))
        self.assertEqual([item["disposition"] for item in candidate_experiments(history)], ["rolled_back"])
        self.assertEqual(candidate_experiments(history[:2]), [])
        # A failed correction opens nothing; a failed rollback closes nothing.
        failed = _prefix()[:2] + [_event(PARENT, None, CORRECT_PARAMETERS, {"state_id": PARENT, "line_index": 3},
                                         {"execution_status": "failure", "error_code": "correction_not_supported_by_current_context"})]
        self.assertEqual(candidate_experiments(failed), [])
        refused = _prefix() + [_event(PARENT, CANDIDATE, ROLLBACK_STATE, {"candidate_state_id": CANDIDATE},
                                      {"execution_status": "failure", "error_code": "rollback_refused"})]
        self.assertEqual(candidate_experiments(refused)[0]["disposition"], "open")


class EvaluatorBindingTests(unittest.TestCase):
    """What the evaluator hands the policy in each mode."""

    def _bind(self, history: list[dict], mode: str) -> dict:
        observation = {"active_state_id": PARENT, "candidate_state_id": None, "history_window": ["whatever the environment said"]}
        return _canonical_study_policy_observation(
            observation, state_before={"active_state_id": PARENT, "candidate_state_id": None}, history=history, observation_mode=mode)

    def test_the_bounded_window_is_the_same_in_both_modes(self) -> None:
        history = _declined()
        bounded, complete = self._bind(history, BOUNDED_HISTORY), self._bind(history, COMPLETE_HISTORY)
        self.assertNotIn(OBSERVABLE_HISTORY_KEY, bounded)
        self.assertEqual(bounded["history_window"], history[-STUDY_POLICY_HISTORY_WINDOW:])
        self.assertEqual({key: value for key, value in complete.items() if key != OBSERVABLE_HISTORY_KEY}, bounded)
        self.assertEqual(observable_history_of(complete), history)
        self.assertEqual(self._bind(history, BOUNDED_HISTORY), _canonical_study_policy_observation(
            {"active_state_id": PARENT, "candidate_state_id": None},
            state_before={"active_state_id": PARENT, "candidate_state_id": None}, history=history))

    def test_what_only_the_audit_knows_never_enters_the_stream(self) -> None:
        clean = _declined()
        marked = copy.deepcopy(clean)
        # The verifier's offline verdict and the truth, as an audit record would carry them.
        marked[4]["tool_output"]["candidate_disposition"] = "ACCEPT_PARTIAL"
        marked[4]["tool_output"]["tool_metrics"]["target_fixed"] = False
        marked[2]["tool_output"]["true_parameter_errors"] = [{"line_index1": 12}]
        marked[0]["tool_output"]["tool_metrics"]["remaining_true_fault_count"] = 2
        self.assertEqual(observable_history_of(self._bind(marked, COMPLETE_HISTORY)), clean)
        self.assertEqual(prepare_model_policy_observation(self._bind(marked, COMPLETE_HISTORY)),
                         prepare_model_policy_observation(self._bind(clean, COMPLETE_HISTORY)))


class PromptLimitTests(unittest.TestCase):
    """A complete-history prompt that does not fit is refused, not cut."""

    def test_the_policy_refuses_an_over_long_complete_prompt_and_still_cuts_a_bounded_one(self) -> None:
        try:
            import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch is not installed")
        import psse_env.dagger.research_policy_factory as factory
        from psse_env.dagger.release_factories import _ModelBundle

        (bounded, _), (complete, _) = _views(_declined(), active=PARENT)
        limit = factory.usable_prompt_tokens()

        class _Policy(factory._CanonicalResearchNativePolicy):
            def render_prompt(self, observation):
                return "rendered", [], {}

        policy = _Policy(_ModelBundle(model=None, processor=None, model_id="m", model_revision="r"), architecture="gemma4_unified")
        original = factory.tokenize_rendered_text
        try:
            factory.tokenize_rendered_text = lambda processor, rendered: {"input_ids": torch.zeros((1, limit + 1), dtype=torch.long)}
            with self.assertRaisesRegex(factory.PromptLimitError, "refused, not cut"):
                policy.generate_text(complete)
            self.assertTrue(policy.last_action_metrics["prompt_limit_refused"])
            self.assertEqual(policy.last_action_metrics["original_prompt_tokens"], limit + 1)
            self.assertIsNone(policy.last_generation)  # nothing was generated, so nothing can be trained on
            # The bounded mode goes on past the check (and fails later only because this test has no model).
            with self.assertRaises(Exception) as raised:
                policy.generate_text(bounded)
            self.assertNotIsInstance(raised.exception, factory.PromptLimitError)
            factory.tokenize_rendered_text = lambda processor, rendered: {"input_ids": torch.zeros((1, limit), dtype=torch.long)}
            with self.assertRaises(Exception) as raised:
                policy.generate_text(complete)
            self.assertNotIsInstance(raised.exception, factory.PromptLimitError)
        finally:
            factory.tokenize_rendered_text = original
        self.assertTrue(factory.is_complete_history_view(complete))
        self.assertFalse(factory.is_complete_history_view(bounded))

    def test_a_complete_prompt_the_device_cannot_hold_is_tried_again_and_then_refused(self) -> None:
        try:
            import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch is not installed")
        from types import SimpleNamespace
        from unittest import mock

        import psse_env.dagger.research_policy_factory as factory
        from psse_env.dagger.release_factories import _ModelBundle

        (bounded, _), (complete, _) = _views(_declined(), active=PARENT)

        class OutOfMemoryError(RuntimeError):  # torch's class by name; older builds raise a RuntimeError with the same words
            pass

        class _Model:
            def __init__(self, failures: int, error: Exception) -> None:
                self.failures, self.error, self.calls, self.seeds = failures, error, 0, []

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.calls += 1
                self.seeds.append(int(torch.initial_seed()))
                if self.calls <= self.failures:
                    raise self.error
                sequences = torch.cat([kwargs["input_ids"], torch.tensor([[7, 8]])], dim=-1)
                if kwargs.get("return_dict_in_generate"):
                    return SimpleNamespace(sequences=sequences, scores=[torch.zeros((1, 16)), torch.zeros((1, 16))])
                return sequences

        class _Policy(factory._CanonicalResearchNativePolicy):
            def render_prompt(self, observation):
                return "rendered", [], {}

        def policy_for(model, **generation):
            options = factory.GenerationOptions(**generation) if generation else None
            return _Policy(_ModelBundle(model=model, processor=None, model_id="m", model_revision="r"), architecture="gemma4_unified",
                           **({"generation": options} if options else {}))

        stand_ins = dict(
            tokenize_rendered_text=lambda processor, rendered: {"input_ids": torch.zeros((1, 6), dtype=torch.long)},
            _model_input_device=lambda model: "cpu", infer_required_side_input_names=lambda *args: (),
            get_stop_token_ids=lambda processor: [1], resolve_pad_token_id=lambda processor: 0, prefill_pad_tokens=lambda length: 0,
            decode_generated_response=lambda processor, ids, pad_token_id=None: ("not a tool call", 2, 0),
            MEMORY_RETRY_PAUSES=(0.0, 0.0),
        )
        with contextlib.ExitStack() as stack:
            for name, stand_in in stand_ins.items():
                stack.enter_context(mock.patch.object(factory, name, stand_in))
            memory = OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")
            # Once: the other worker's long prompt passed, the same draw is taken again.
            model = _Model(1, memory)
            policy = policy_for(model, do_sample=True, temperature=1.0, top_p=1.0, top_k=None, seed=41)
            self.assertEqual((policy.generate_text(complete), model.calls), ("not a tool call", 2))
            self.assertEqual(policy.last_action_metrics["memory_retries"], 1)
            self.assertEqual(model.seeds, [41, 41])  # the repeated attempt is seeded as the first was
            self.assertEqual(policy.last_generation["completion_token_ids"], [7, 8])
            self.assertEqual(policy.last_generation["completion_token_probabilities"], [1 / 16, 1 / 16])
            # A prompt that fits at once reports no retry, as before.
            model = _Model(0, memory)
            policy = policy_for(model)
            self.assertEqual((policy.generate_text(complete), model.calls), ("not a tool call", 1))
            self.assertNotIn("memory_retries", policy.last_action_metrics)
            # After a long complete-history prompt the cached device memory goes back; a short or a bounded one keeps it.
            released: list[int] = []
            with mock.patch.object(factory, "release_device_memory", lambda torch_module: released.append(1)):
                policy_for(_Model(0, memory)).generate_text(complete)
                self.assertEqual(released, [])
                with mock.patch.object(factory, "MEMORY_RELEASE_PROMPT_TOKENS", 6):
                    policy_for(_Model(0, memory)).generate_text(complete)
                    self.assertEqual(released, [1])
                    policy_for(_Model(0, memory), do_sample=True, temperature=1.0, top_p=1.0, top_k=None, seed=3).generate_text(complete)
                    self.assertEqual(released, [1, 1])
                    policy_for(_Model(0, memory)).generate_text(bounded)
                    self.assertEqual(released, [1, 1])
            # Every time: refused as an over-long prompt is, with nothing generated to train on.
            model = _Model(99, RuntimeError("CUDA error: out of memory"))
            policy = policy_for(model)
            with self.assertRaisesRegex(factory.PromptLimitError, "did not fit the device's memory .2 attempts after the first.; it is refused"):
                policy.generate_text(complete)
            self.assertEqual(model.calls, 3)
            self.assertEqual({key: policy.last_action_metrics[key] for key in ("prompt_limit_refused", "prompt_limit_reason", "memory_retries")},
                             {"prompt_limit_refused": True, "prompt_limit_reason": "device_memory", "memory_retries": 2})
            self.assertIsNone(policy.last_generation)
            # The episode's next prompt is at least as long: it is tried once and not waited for again.
            with self.assertRaisesRegex(factory.PromptLimitError, "0 attempts after the first"):
                policy.generate_text(complete)
            self.assertEqual(model.calls, 4)
            # Once the device holds that length again, a later failure is waited for as before.
            model.failures = 0
            self.assertEqual(policy.generate_text(complete), "not a tool call")
            model.failures, model.calls = 99, 0
            with self.assertRaisesRegex(factory.PromptLimitError, "2 attempts after the first"):
                policy.generate_text(complete)
            self.assertEqual(model.calls, 3)
            # The bounded mode is as it was: the error leaves the policy on the first attempt.
            model = _Model(99, memory)
            with self.assertRaises(OutOfMemoryError):
                policy_for(model).generate_text(bounded)
            self.assertEqual(model.calls, 1)
            # Another error is not the device's memory, in either mode.
            model = _Model(99, RuntimeError("shape mismatch"))
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                policy_for(model).generate_text(complete)
            self.assertEqual(model.calls, 1)
        self.assertTrue(factory.is_out_of_memory_error(memory))
        self.assertFalse(factory.is_out_of_memory_error(ValueError("out of memory")))

    def test_the_rollout_summary_tells_a_memory_refusal_from_a_limit_refusal(self) -> None:
        from psse_env.rl.rollout import PROMPT_MEMORY_MARK, prompt_limit_refusals

        def step(detail):
            return {"action": {"tool": "__invalid_action__", "arguments": {"error_detail": detail}}}

        trace = [step("PromptLimitError: the complete-history prompt has 70000 tokens, more than the 65280 that leave room"),
                 step("PromptLimitError: the complete-history prompt of 40000 tokens did not fit the device's memory (2 attempts after the first)"),
                 step("GateError: did not fit the device's memory"), {"action": {"tool": "run_wls", "arguments": {}}}]
        self.assertEqual((prompt_limit_refusals(trace), prompt_limit_refusals(trace, mark=PROMPT_MEMORY_MARK)), (2, 1))


if __name__ == "__main__":
    unittest.main()
