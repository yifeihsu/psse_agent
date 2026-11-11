"""Branching at a lifecycle decision and the outcome credit built on it, on a toy closed loop.

The toy evaluator has the observation and trace shapes of the real one and a
few lines of physics: a line-parameter correction opens a candidate, a solve
on it verifies it (the verifier accepts every candidate, so commit and
decline are both legal), and the case is solved when the corrections kept at
the hand-off are exactly the truly faulted lines.  Every run gets its own
episode id, so recorded state ids never match a replay's and the alias
binding is exercised.
"""
from __future__ import annotations

import copy
import functools
import json
import tempfile
import unittest
from pathlib import Path

from psse_env.actions import ASK_FOR_MORE_EVIDENCE, COMMIT_STATE, CORRECT_PARAMETERS, GET_PARAMETER_CONTEXT, ROLLBACK_STATE, RUN_WLS
from psse_env.dagger.observation_modes import BOUNDED_HISTORY, COMPLETE_HISTORY, stored_observation, with_observable_history
from psse_env.rl.decision_credit import (
    DECISION_CREDIT_CONTRACT,
    DecisionCreditSettings,
    candidate_states,
    check_provenance,
    decision_token_probability,
    select_states,
    state_stratum,
    state_values,
    summarize_values,
    values_map,
)
from psse_env.rl.forks import (
    DecisionState,
    ForkPolicy,
    branch_seed,
    decision_states,
    rebind_action,
    recovery_chain,
    run_branch,
    view_signature,
)
from psse_env.rl.lifecycle import scenario_truth
from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.reward import RewardWeights
from psse_env.rl.rollout import TrajectoryStore, rollout_group

LINES = (3, 7, 9)


def scenario(root: str, true_lines=(7,), *, name: str = "cb2_split", declinable: bool = True, relation: str | None = None) -> dict:
    return {
        "execution": {"scenario_id": root, "lifecycle_choice": declinable},
        "grouping": {"scenario_family": "compound", "physical_root_fingerprint": relation or f"physical_{root}"},
        "audit": {"compound": {"class": name},
                  "truth": {"true_parameter_errors": [{"line_index1": line} for line in true_lines], "truth_complete": True}},
    }


class _Result:
    def __init__(self, episode: dict) -> None:
        self.episode = episode

    def as_dict(self) -> dict:
        return {"suite_metrics": {"episodes": [self.episode]}}


class ToyEvaluator:
    """A closed-loop evaluator over the toy environment (``evaluate_rollout_suites``'s calling convention)."""

    def __init__(self) -> None:
        self.episodes = 0
        self.modes: list[str] = []

    def __call__(self, suites, *, env_factory, policy_factory, max_steps, seed, case_loader, observation_mode=BOUNDED_HISTORY, **contract):
        root = suites["standard_success"][0]
        self.episodes += 1
        self.modes.append(observation_mode)
        episode_id = f"episode-{self.episodes:04d}"
        policy = policy_factory()
        truth = {int(item["line_index1"]) for item in root["audit"]["truth"]["true_parameter_errors"]}
        declinable = bool(root["execution"].get("lifecycle_choice", True))
        active, candidate, candidate_line, verified, counter = f"{episode_id}:s0", None, None, False, 0
        committed: list[int] = []
        rejected: list[int] = []
        states: dict[int, str] = {}  # the candidate state each tried line had, as the real ledgers keep it
        history: list[dict] = []
        trace: list[dict] = []
        invalid = 0
        terminal = False
        for step in range(max_steps):
            last = history[-1] if history else None
            observation = {
                "active_state_id": active, "candidate_state_id": candidate, "episode_id": episode_id, "evidence_profile": "scada_only",
                "remaining_budget": max_steps - step,
                "accepted_corrections": [{"line_index": line, "candidate_state_id": states[line]} for line in committed],
                "rejected_hypotheses": [{"line_index": line, "candidate_state_id": states[line]} for line in rejected],
                "has_open_candidate": candidate is not None, "has_verified_candidate": verified,
                "last_tool": last["action"]["tool"] if last else None,
                "last_tool_status": last["tool_output"]["execution_status"] if last else None,
                "last_tool_output": copy.deepcopy(last["tool_output"]) if last else {},
                "history_window": copy.deepcopy(history[-4:]),
            }
            if observation_mode == COMPLETE_HISTORY:
                observation = with_observable_history(observation, history)
            before = {"active_state_id": active, "candidate_state_id": candidate}
            verdict = "ACCEPT_PARTIAL" if candidate is not None and verified else None
            try:
                action = policy.act(copy.deepcopy(observation))
            except Exception as exc:  # as the real evaluator records a policy exception
                action = {"tool": "__invalid_action__", "arguments": {"error_code": "policy_exception", "error_detail": f"{type(exc).__name__}: {exc}"}}
            tool, arguments = action["tool"], dict(action.get("arguments") or {})
            output: dict = {"execution_status": "success", "state_mutated": False}
            failure = None
            if tool == RUN_WLS:
                if arguments.get("state_id") not in (active, candidate):
                    failure = "unknown_state"
                else:
                    output["tool_metrics"] = {"chi_square_statistic": 100.0 - 30.0 * len(set(committed) & truth) + 20.0 * len(set(committed) - truth)}
                    verified = verified or arguments.get("state_id") == candidate
            elif tool == GET_PARAMETER_CONTEXT:
                output["tool_metrics"] = {"supported_corrections": [
                    {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active, "line_index": line}} for line in LINES if line not in committed]}
            elif tool == CORRECT_PARAMETERS:
                if candidate is not None or arguments.get("state_id") != active:
                    failure = "candidate_open_or_stale_state"
                else:
                    counter += 1
                    candidate, candidate_line, verified = f"{episode_id}:s{counter}", int(arguments["line_index"]), False
                    states[candidate_line] = candidate
                    output.update(state_mutated=True, active_state_id=active, candidate_state_id=candidate)
            elif tool == COMMIT_STATE:
                if candidate is None or not verified or arguments.get("candidate_state_id") != candidate:
                    failure = "commit_without_verified_candidate"
                else:
                    committed.append(candidate_line)
                    active, candidate, verified = candidate, None, False
                    output.update(state_mutated=True, active_state_id=active)
            elif tool == ROLLBACK_STATE:
                if candidate is None or arguments.get("candidate_state_id") != candidate:
                    failure = "rollback_without_candidate"
                elif verified and not declinable:
                    failure = "rollback_of_accepted_candidate_refused"
                else:
                    rejected.append(candidate_line)
                    candidate, verified = None, False
                    output.update(state_mutated=True, active_state_id=active)
            elif tool == ASK_FOR_MORE_EVIDENCE:
                terminal = True
            else:
                failure = "invalid_action"
            if failure is not None:
                output = {"execution_status": "failure", "error_code": failure, "state_mutated": False}
                invalid += 1
            history.append({"state_id": before["active_state_id"], "candidate_state_id": before["candidate_state_id"],
                            "action": copy.deepcopy(action), "tool_output": copy.deepcopy(output)})
            trace.append({"step": step, "intervention": False, "policy_observation": stored_observation(observation),
                          "action": copy.deepcopy(action), "execution_status": output["execution_status"], "error_code": failure,
                          "candidate_disposition_offline": verdict, "policy_tool_output": copy.deepcopy(output), "state_before": before})
            if terminal:
                break
        success = terminal and set(committed) == truth
        return _Result({
            "scenario_id": root["execution"]["scenario_id"], "family": "compound", "trace": trace,
            "truth_audited_task_success": success, "steps": len(trace), "policy_steps": len(trace), "terminal": terminal,
            "false_commit_count": 0, "false_finalization_count": 0, "invalid_action_count": invalid, "loop_detected": False,
            "healthy_components_preserved": set(committed) <= truth, "healthy_preservation_known": True,
            "audit": {"truth_audited_task_assessment": {"basis": "counterfactual_resolution"}} if success else {},
        })


class Scripted:
    """A policy that tries its lines in order and keeps every candidate but the ones it is told to decline.

    It decides from the observation alone, so it can take over an episode
    after a forced step, and it exposes a generation record as a sampling
    policy does: the decision token is the completion's second token.  Its
    record is of one distribution: ``p_choice`` for the token of a commit (and
    of any other call) and the rest for the token of a decline.
    """

    def __init__(self, lines=(7,), *, decline=(), context_first: bool = False, p_choice: float = 0.8) -> None:
        self.lines, self.decline, self.context_first, self.p_choice = list(lines), set(decline), context_first, p_choice
        self.calls = 0
        self.last_generation: dict | None = None
        self._line: int | None = None

    def _choose(self, observation: dict) -> dict:
        active, candidate = observation["active_state_id"], observation.get("candidate_state_id")
        if candidate is not None:
            if not observation["has_verified_candidate"]:
                return {"tool": RUN_WLS, "arguments": {"state_id": candidate}}
            tool = ROLLBACK_STATE if self._line in self.decline else COMMIT_STATE
            return {"tool": tool, "arguments": {"candidate_state_id": candidate}}
        done = {item["line_index"] for item in observation["accepted_corrections"] + observation["rejected_hypotheses"]}
        todo = [line for line in self.lines if line not in done]
        if not todo:
            return {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"request": "operator_escalation:review"}}
        if self.context_first and observation.get("last_tool") != GET_PARAMETER_CONTEXT:
            return {"tool": GET_PARAMETER_CONTEXT, "arguments": {"state_id": active}}
        self._line = todo[0]
        return {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active, "line_index": todo[0]}}

    def act(self, observation: dict) -> dict:
        action = self._choose(observation)
        # The prompt stands for the visible state: the step and the line the open candidate is on.
        prompt = [3, 4, self.calls % 6, (self._line or 0) % 12]
        self.last_generation = {
            "model_observation": {"prompt": prompt}, "prompt_tokens": len(prompt), "original_prompt_tokens": len(prompt),
            "completion_token_ids": [5, 10 if action["tool"] == ROLLBACK_STATE else 11],
            "completion_token_probabilities": [0.99, round(1.0 - self.p_choice, 10) if action["tool"] == ROLLBACK_STATE else self.p_choice],
            "text": f"call:{action['tool']}", "call_index": self.calls,
            "prompt_token_digest": prompt_token_digest(prompt),
        }
        self.calls += 1
        return action


def render_other_choice(observation, text, decision):
    """The other choice's completion, as a target renderer gives it: the same call with the other tool's token."""

    return [5, 10 if decision["label"] == ROLLBACK_STATE else 11]


def rollouts(evaluator: ToyEvaluator, root: dict, policies: list, *, base_seed: int = 1):
    """One group of sampled episodes of ``root``, one policy per sample, as the driver's rollout phase stores them."""

    queue = list(policies)
    return rollout_group(root, samples=len(policies), base_seed=base_seed, env_factory=lambda **_k: None,
                         policy_factory=lambda seed: queue.pop(0), max_steps=12, case_loader=None,
                         weights=RewardWeights(), evaluator=evaluator)


def branch(evaluator: ToyEvaluator, root: dict, state: DecisionState, choice: str, continuation, **options) -> dict:
    return run_branch(root, state, choice, continuation=continuation, continuation_index=0, seed=5, env_factory=lambda **_k: None,
                      max_steps=12, case_loader=None, weights=RewardWeights(), evaluator=evaluator, **options)


class ForkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ToyEvaluator()
        self.root = scenario("r1")
        # The policy keeps a wrong line and hands off: an unsolved episode with one decision on an accepted candidate.
        self.trajectory = rollouts(self.evaluator, self.root, [Scripted(lines=(3,))])[0]
        self.state = decision_states(self.trajectory)[0]

    def test_a_decision_state_holds_what_reaches_the_state_again(self) -> None:
        self.assertTrue(self.trajectory.pairing_ok)
        self.assertFalse(self.trajectory.evaluation["truth_audited_task_success"])
        self.assertEqual(len(decision_states(self.trajectory)), 1)
        state = self.state
        self.assertEqual((state.root, state.sample_index, state.position, state.sampled_tool), ("r1", 0, 2, COMMIT_STATE))
        self.assertEqual([action["tool"] for action in state.prefix_actions], [CORRECT_PARAMETERS, RUN_WLS])
        self.assertEqual(len(state.observations), 3)
        self.assertEqual((state.decision["target_is_true"], state.decision["label"], state.decision["agrees"]), (False, ROLLBACK_STATE, False))
        self.assertEqual((state.ordinal, state.earlier_declines, state.source_success), (0, 0, False))
        self.assertEqual(len(state.transitions), 2)
        self.assertEqual(state.sampled_text, "call:commit_state")
        self.assertEqual(state.sampled_view, self.trajectory.steps[2].model_observation)
        self.assertEqual(state_stratum(state), "wrong_target")
        self.assertEqual(DecisionState.from_dict(json.loads(json.dumps(state.as_dict()))), state)
        # The same state under the two observation modes: the complete one adds the transitions before it.
        self.assertNotIn("observable_history", state.observation(BOUNDED_HISTORY))
        self.assertEqual(state.observation(COMPLETE_HISTORY)["observable_history"]["events"], state.transitions)

    def test_declining_the_wrong_candidate_and_continuing_solves_the_case(self) -> None:
        declined = branch(self.evaluator, self.root, self.state, ROLLBACK_STATE, Scripted(lines=(7,)))
        kept = branch(self.evaluator, self.root, self.state, COMMIT_STATE, Scripted(lines=(7,)))
        for record in (declined, kept):
            self.assertEqual((record["valid"], record["prefix_matched"], record["fork_reached"], record["forced_executed"]),
                             (True, True, True, True))
            self.assertEqual(record["forced_position"], 2)
        self.assertEqual((declined["success"], kept["success"]), (True, False))
        self.assertEqual((declined["forced_is_sampled_choice"], kept["forced_is_sampled_choice"]), (False, True))
        self.assertGreater(declined["reward"]["total"], kept["reward"]["total"])
        self.assertTrue(kept["reward"]["healthy_component_corrupted"])
        # The forced step is the harness's; what the continuation generated starts after it.
        self.assertEqual(declined["tools"], [CORRECT_PARAMETERS, RUN_WLS, ROLLBACK_STATE, CORRECT_PARAMETERS, RUN_WLS, COMMIT_STATE,
                                             ASK_FOR_MORE_EVIDENCE])
        self.assertEqual((declined["continuation_steps"], declined["continuation_records"]), (4, 4))
        # The whole episode counts against the budget and the cost, the replayed prefix included.
        self.assertEqual((declined["policy_steps"], declined["reward"]["calls"]), (7, 7))
        self.assertEqual([decision["declined"] for decision in declined["lifecycle"]], [True, False])

    def test_the_replay_binds_the_recorded_actions_to_the_running_episodes_states(self) -> None:
        recorded = self.state.prefix_actions[1]["arguments"]["state_id"]
        record = branch(self.evaluator, self.root, self.state, ROLLBACK_STATE, Scripted(lines=(7,)), keep_trace=True)
        replayed = record["trace"][1]["action"]["arguments"]["state_id"]
        self.assertNotEqual(replayed, recorded)  # another episode, other ids
        self.assertEqual(replayed.split(":")[1], recorded.split(":")[1])
        digest, aliases = view_signature(self.state.observations[1])
        self.assertEqual(aliases, {"active": self.state.observations[1]["active_state_id"], "candidate": recorded})
        self.assertEqual(rebind_action({"tool": RUN_WLS, "arguments": {"state_id": recorded, "other": recorded}}, aliases,
                                       {"active": "x:s0", "candidate": "x:s1"})["arguments"], {"state_id": "x:s1", "other": recorded})
        self.assertEqual(view_signature(record["trace"][1]["policy_observation"])[0], digest)

    def test_the_state_signature_does_not_depend_on_the_order_of_a_mappings_keys(self) -> None:
        # A record read back from a store has its keys sorted; the running environment has them in its own order.
        observation = self.state.observations[2]
        reordered = {key: observation[key] for key in reversed(list(observation))}
        reordered["last_tool_output"] = {key: observation["last_tool_output"][key] for key in reversed(list(observation["last_tool_output"]))}
        self.assertNotEqual(list(reordered), list(observation))
        self.assertEqual(view_signature(reordered), view_signature(observation))

    def test_the_view_at_a_decision_is_read_from_a_replay_in_either_mode(self) -> None:
        from psse_env.rl.forks import capture_fork_view

        common = dict(env_factory=lambda **_k: None, max_steps=12, case_loader=None, evaluator=self.evaluator)
        bounded = capture_fork_view(self.root, self.state, observation_mode=BOUNDED_HISTORY, **common)
        complete = capture_fork_view(self.root, self.state, observation_mode=COMPLETE_HISTORY, **common)
        self.assertEqual((bounded["matched"], bounded["reached"], complete["matched"]), (True, True, True))
        self.assertNotIn("history_record", bounded["view"])
        self.assertEqual(complete["view"]["history_record"], {"contract": COMPLETE_HISTORY, "events": 2})
        self.assertEqual([event["tool"] for event in complete["view"]["history_window"]], ["correct_parameters_from_path", "wls_from_path"])
        # The toy policy's recorded view is a stand-in, so the bounded view is not that record; a real rollout's is.
        self.assertIs(bounded["equals_sampled_view"], False)
        drifted = DecisionState.from_dict(self.state.as_dict())
        drifted.prefix_actions[0]["arguments"]["line_index"] = 9
        self.assertFalse(capture_fork_view(self.root, drifted, observation_mode=BOUNDED_HISTORY, **common)["matched"])

    def test_a_replay_that_shows_the_policy_something_else_is_marked_and_not_used(self) -> None:
        drifted = DecisionState.from_dict(self.state.as_dict())
        drifted.prefix_actions[0]["arguments"]["line_index"] = 9  # the recorded prefix corrected line 3
        record = branch(self.evaluator, self.root, drifted, ROLLBACK_STATE, Scripted(lines=(7,)))
        self.assertEqual((record["valid"], record["prefix_matched"]), (False, False))
        self.assertEqual([item["position"] for item in record["mismatches"]], [1, 2])
        with self.assertRaisesRegex(ValueError, "an observation for every step"):
            ForkPolicy(DecisionState.from_dict({**self.state.as_dict(), "observations": self.state.as_dict()["observations"][:-1]}),
                       ROLLBACK_STATE, Scripted())
        with self.assertRaisesRegex(ValueError, "the decision state is of r1"):
            branch(self.evaluator, scenario("r2"), self.state, ROLLBACK_STATE, Scripted())
        with self.assertRaisesRegex(ValueError, "forced choice is one of"):
            ForkPolicy(self.state, RUN_WLS, Scripted())

    def test_a_forced_choice_the_environment_refuses_makes_no_valid_branch(self) -> None:
        strict = scenario("r1", declinable=False)
        record = branch(self.evaluator, strict, self.state, ROLLBACK_STATE, Scripted(lines=(7,)))
        self.assertEqual((record["valid"], record["prefix_matched"], record["forced_executed"]), (False, True, False))
        self.assertEqual(record["forced_error_code"], "rollback_of_accepted_candidate_refused")

    def test_both_observation_modes_reach_the_same_state_and_seeds_are_shared(self) -> None:
        bounded = branch(self.evaluator, self.root, self.state, ROLLBACK_STATE, Scripted(lines=(7,)))
        complete = branch(self.evaluator, self.root, self.state, ROLLBACK_STATE, Scripted(lines=(7,)), observation_mode=COMPLETE_HISTORY)
        self.assertEqual(self.evaluator.modes[-2:], [BOUNDED_HISTORY, COMPLETE_HISTORY])
        self.assertEqual((complete["valid"], complete["observation_mode"], complete["tools"]), (True, COMPLETE_HISTORY, bounded["tools"]))
        self.assertEqual(branch_seed(9, self.state, 1), branch_seed(9, self.state, 1))
        self.assertNotEqual(branch_seed(9, self.state, 0), branch_seed(9, self.state, 1))

    def test_what_followed_a_forced_choice_is_read_from_the_branch(self) -> None:
        record = branch(self.evaluator, self.root, self.state, ROLLBACK_STATE, Scripted(lines=(7,), context_first=True), keep_trace=True)
        chain = record["chain"]
        self.assertEqual((chain["productive_target_offered"], chain["next_target_is_true"], chain["next_correction_accepted"],
                          chain["next_correction_committed"]), (True, True, True, True))
        self.assertEqual(chain["next_correction"]["arguments"], {"line_index": 7})
        self.assertEqual(recovery_chain(record["trace"], forced_position=2, truth=scenario_truth(self.root)), chain)
        # A continuation that goes to another wrong line: nothing true was offered to it, and it keeps the wrong one.
        wrong = branch(self.evaluator, self.root, self.state, ROLLBACK_STATE, Scripted(lines=(9,)))
        self.assertEqual((wrong["chain"]["productive_target_offered"], wrong["chain"]["next_target_is_true"], wrong["success"]),
                         (False, False, False))
        ended = branch(self.evaluator, self.root, self.state, COMMIT_STATE, Scripted(lines=()))
        self.assertIsNone(ended["chain"]["next_correction"])


class DecisionCreditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ToyEvaluator()
        self.settings = DecisionCreditSettings(enabled=True, states_per_iteration=4, continuations=1, include_source_return=False)

    def _groups(self):
        """Three roots: one where two samples keep a wrong line and one declines it; a true first candidate; a decline first."""

        first = scenario("r1")
        second = scenario("r2", name="rx2_distant")
        third = scenario("r3", (3, 7), name="rx2_distant")
        return [
            ("r1", rollouts(self.evaluator, first, [Scripted(lines=(3,)), Scripted(lines=(3,)), Scripted(lines=(3, 7), decline=(3,))])),
            ("r2", rollouts(self.evaluator, second, [Scripted(lines=(7,)), Scripted(lines=(7,))])),
            ("r3", rollouts(self.evaluator, third, [Scripted(lines=(9, 7, 3), decline=(9,))])),
        ], {"r1": first, "r2": second, "r3": third}

    def test_states_are_counted_once_per_root_and_chosen_by_kind(self) -> None:
        groups, _roots = self._groups()
        candidates = candidate_states(groups)
        # r1's three samples decide at one visible state; the third's later commit of line 7 is a state of its own.
        self.assertEqual([(group.state.root, group.state.position, group.stratum, len(group.rows)) for group in candidates],
                         [("r1", 2, "wrong_target", 3), ("r1", 5, "true_after_decline", 1), ("r2", 2, "true_first", 2),
                          ("r3", 2, "sampled_decline", 1), ("r3", 5, "true_after_decline", 1), ("r3", 8, "true_after_decline", 1)])
        self.assertEqual(sorted(row["tool"] for row in candidates[0].rows), [COMMIT_STATE, COMMIT_STATE, ROLLBACK_STATE])
        chosen, report = select_states(candidates, self.settings, seed=3)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(report["available"], {"sampled_decline": 1, "wrong_target": 1, "true_after_decline": 3, "true_later": 0, "true_first": 1})
        # The default quotas are for a budget of 32; for a budget of four they are scaled down to it.
        self.assertEqual(report["effective_quotas"], {"sampled_decline": 1, "wrong_target": 1, "true_after_decline": 0, "true_later": 0,
                                                      "true_first": 0})
        # The rest of the budget goes to roots not used yet first, so the retention case of r2 is in.
        self.assertEqual((report["selected"]["sampled_decline"], report["selected"]["wrong_target"], report["selected"]["true_first"]), (1, 1, 1))
        self.assertEqual(report["shortfall"], {name: 0 for name in report["shortfall"]})
        self.assertEqual(report["selection_fraction"]["true_later"], None)
        self.assertEqual(report["roots_selected"], 3)
        # A kind the batch cannot fill is a recorded shortfall.
        short = DecisionCreditSettings(enabled=True, states_per_iteration=4, quotas={"wrong_target": 3})
        self.assertEqual(select_states(candidates, short, seed=3)[1]["shortfall"]["wrong_target"], 2)
        again, _report = select_states(candidate_states(groups), self.settings, seed=3)
        self.assertEqual([group.key for group in again], [group.key for group in chosen])
        with self.assertRaisesRegex(ValueError, "unknown decision_credit fields"):
            DecisionCreditSettings.from_mapping({"branches": 2})
        self.assertEqual(DecisionCreditSettings.from_mapping({"enabled": True, "quotas": {"wrong_target": 3}}).quotas, {"wrong_target": 3})

    def test_the_local_advantage_is_the_choice_value_less_the_policy_weighted_baseline(self) -> None:
        groups, roots = self._groups()
        group = candidate_states(groups)[0]
        for row in group.rows:
            trajectory = next(t for t in groups[0][1] if t.sample_index == row["sample_index"])
            row["probability"], row["probability_source"] = decision_token_probability(
                trajectory.steps[row["position"]], row["tool"], render_other_choice)
        # One policy at one state: 0.8 on the token of the commit, 0.2 on the token of the decline.
        self.assertEqual({(row["tool"], row["probability"], row["probability_source"]) for row in group.rows},
                         {(COMMIT_STATE, 0.8, "decision_token"), (ROLLBACK_STATE, 0.2, "decision_token")})
        branches = [branch(self.evaluator, roots["r1"], group.state, choice, Scripted(lines=(7,))) for choice in (COMMIT_STATE, ROLLBACK_STATE)]
        q_commit, q_decline = (record["reward"]["total"] for record in branches)
        values = state_values(group, branches, self.settings)
        self.assertEqual((values["covered"], values["informative"], values["stratum"]), (True, True, "wrong_target"))
        self.assertAlmostEqual(values["q_gap_commit_minus_decline"], q_commit - q_decline)
        by_tool = {row["tool"]: row for row in values["rows"]}
        # One baseline for the state, V = p(commit) Q(commit) + p(decline) Q(decline), whichever choice a row sampled:
        # A(a) = Q(a) - V = (1 - p(a)) (Q(a) - Q(other)).
        self.assertEqual(len({row["baseline"] for row in values["rows"]}), 1)
        self.assertAlmostEqual(values["baseline"], 0.8 * q_commit + 0.2 * q_decline)
        self.assertAlmostEqual(values["policy"][COMMIT_STATE], 0.8)
        self.assertAlmostEqual(values["policy"][ROLLBACK_STATE], 0.2)
        self.assertEqual(values["probability_basis"], "both_choices_recorded")
        self.assertAlmostEqual(values["mass_on_neither_choice"], 0.0)
        self.assertAlmostEqual(by_tool[COMMIT_STATE]["advantage"], 0.2 * (q_commit - q_decline))
        self.assertAlmostEqual(by_tool[ROLLBACK_STATE]["advantage"], 0.8 * (q_decline - q_commit))
        self.assertLess(by_tool[COMMIT_STATE]["advantage"], 0.0)
        self.assertGreater(by_tool[ROLLBACK_STATE]["advantage"], 0.0)
        # The policy's own expectation of the advantage over the two choices is zero, as a baseline's has to be.
        self.assertAlmostEqual(0.8 * by_tool[COMMIT_STATE]["advantage"] + 0.2 * by_tool[ROLLBACK_STATE]["advantage"], 0.0)
        self.assertEqual(len(values["rows"]), 3)
        mapped = values_map({"states": [values]})
        self.assertEqual(set(mapped), {("r1", 0, 2), ("r1", 1, 2), ("r1", 2, 2)})
        self.assertEqual(mapped[("r1", 2, 2)]["tool"], ROLLBACK_STATE)
        # With the rollouts' own returns counted in Q of the choice they sampled, the baseline is still the branches' alone.
        with_source = state_values(group, branches, DecisionCreditSettings(enabled=True))
        sampled_commit = [row["reward"] for row in group.rows if row["tool"] == COMMIT_STATE]
        self.assertAlmostEqual(with_source["q"][COMMIT_STATE], (q_commit + sum(sampled_commit)) / 3.0)
        self.assertEqual(with_source["q_branch"], values["q_branch"])
        summary = summarize_values([values])
        self.assertEqual((summary["covered"], summary["all"]["decline_better"], summary["sampled_rows_credited"]), (1, 1, 3))
        self.assertEqual(summary["probability_basis"], {"both_choices_recorded": 1})
        self.assertEqual(summary["mass_on_neither_choice"]["states_measured"], 1)
        self.assertAlmostEqual(summary["mass_on_neither_choice"]["max"], 0.0)
        # The audit's label says decline here and the values agree; a label on the other side would be a conflict.
        self.assertEqual(summary["membership_label_against_values"]["label_and_values_disagree"], 0)
        flipped = copy.deepcopy(values)
        flipped["decision"]["label"] = COMMIT_STATE
        self.assertEqual(summarize_values([flipped])["membership_label_against_values"]["roots"], ["r1"])

    def test_the_baseline_is_the_states_and_not_the_rows(self) -> None:
        groups, roots = self._groups()
        group = candidate_states(groups)[0]
        branches = [branch(self.evaluator, roots["r1"], group.state, choice, Scripted(lines=(7,))) for choice in (COMMIT_STATE, ROLLBACK_STATE)]
        q_commit, q_decline = (record["reward"]["total"] for record in branches)

        def credited(p_commit, p_decline):
            for row in group.rows:
                row["probability"] = p_commit if row["tool"] == COMMIT_STATE else p_decline
                row["probability_source"] = "decision_token"
            return state_values(group, branches, self.settings)

        # The policy also called other tools at this step with probability 0.1 (the review's case, 0.7 and 0.2): taking
        # each row's probability for its own choice and the complement for the other gave the commit rows a baseline
        # with 0.3 on the decline and the decline row one with 0.8 on the commit.  The state has one.
        values = credited(0.7, 0.2)
        self.assertEqual(len({row["baseline"] for row in values["rows"]}), 1)
        self.assertAlmostEqual(values["baseline"], (0.7 * q_commit + 0.2 * q_decline) / 0.9)
        self.assertAlmostEqual(values["mass_on_neither_choice"], 0.1)
        self.assertAlmostEqual(values["policy"][COMMIT_STATE] + values["policy"][ROLLBACK_STATE], 1.0)
        by_tool = {row["tool"]: row for row in values["rows"]}
        self.assertAlmostEqual(by_tool[COMMIT_STATE]["advantage"] - by_tool[ROLLBACK_STATE]["advantage"], q_commit - q_decline)
        self.assertEqual((by_tool[COMMIT_STATE]["probability"], by_tool[ROLLBACK_STATE]["probability"]), (0.7, 0.2))
        summary = summarize_values([values])
        self.assertAlmostEqual(summary["mass_on_neither_choice"]["max"], 0.1)
        # Probabilities that cannot be of one distribution credit nothing, and the summary counts the state.
        broken = credited(0.8, 0.8)
        self.assertEqual((broken["covered"], broken["rows"]), (False, []))
        self.assertIn("add up to more than one", broken["reason"])
        self.assertEqual(values_map({"states": [broken]}), {})
        self.assertEqual(summarize_values([broken])["states_with_inconsistent_probabilities"], 1)
        # A state every row left by the same choice: the other choice's probability is not on record and is the complement.
        commits = copy.deepcopy(group)
        commits.rows = [row for row in commits.rows if row["tool"] == COMMIT_STATE]
        for row in commits.rows:
            row["probability"] = 0.7
        single = state_values(commits, branches, self.settings)
        self.assertEqual((single["probability_basis"], single["mass_on_neither_choice"]), ("complement_of_the_sampled_choice", None))
        self.assertAlmostEqual(single["baseline"], 0.7 * q_commit + 0.3 * q_decline)
        self.assertEqual(summarize_values([single])["probability_basis"], {"complement_of_the_sampled_choice": 1})
        # A row whose decision token was not found is not credited; the rows that recorded theirs still are, by the same baseline.
        whole = credited(0.7, 0.2)
        group.rows[0]["probability"], group.rows[0]["probability_source"] = None, "no_decision_token"
        partial = state_values(group, branches, self.settings)
        self.assertEqual((len(whole["rows"]), len(partial["rows"])), (3, 2))
        self.assertEqual({row["baseline"] for row in partial["rows"]}, {whole["baseline"]})

    def test_a_state_without_a_valid_branch_of_each_choice_or_without_a_probability_credits_nothing(self) -> None:
        groups, roots = self._groups()
        group = candidate_states(groups)[0]
        kept = branch(self.evaluator, roots["r1"], group.state, COMMIT_STATE, Scripted(lines=(7,)))
        refused = branch(self.evaluator, scenario("r1", declinable=False), group.state, ROLLBACK_STATE, Scripted(lines=(7,)))
        values = state_values(group, [kept, refused], self.settings)
        self.assertEqual((values["covered"], values["reason"]), (False, "a choice has no valid continuation"))
        self.assertEqual(values["branches"], {"valid": 1, "invalid": 1, "mismatched_prefix": 0, "forced_refused": 1})
        self.assertEqual(values_map({"states": [values]}), {})
        declined = branch(self.evaluator, roots["r1"], group.state, ROLLBACK_STATE, Scripted(lines=(7,)))
        unread = state_values(group, [kept, declined], self.settings)  # no probability was read
        self.assertEqual((unread["rows"], unread["covered"], unread["policy"], unread["baseline"]), ([], True, None, None))
        self.assertEqual(summarize_values([unread])["probability_basis"], {"no_probability_read": 1})
        step = groups[0][1][0].steps[2]
        # Without the other choice's rendering the decision token is not known, and nothing stands in for it: the least
        # likely token of a completion is not the probability of the choice.
        self.assertEqual(decision_token_probability(step, COMMIT_STATE, None), (None, "no_decision_token"))
        self.assertEqual(decision_token_probability(step, COMMIT_STATE, lambda *_args: list(step.completion_token_ids)),
                         (None, "no_decision_token"))  # the two renderings do not part

        def unrenderable(*_args):
            raise ValueError("no rendering of the other choice")

        self.assertEqual(decision_token_probability(step, COMMIT_STATE, unrenderable), (None, "no_decision_token"))
        step.token_probabilities = []
        self.assertEqual(decision_token_probability(step, COMMIT_STATE, render_other_choice), (None, "no_sampling_record"))

    def test_values_are_targets_only_for_the_policy_and_mode_they_were_sampled_under(self) -> None:
        payload = {"contract": DECISION_CREDIT_CONTRACT, "iteration": 2, "adapter": "a2", "observation_mode": BOUNDED_HISTORY}
        check_provenance(payload, iteration=2, adapter="a2", observation_mode=BOUNDED_HISTORY)
        for changes in ({"iteration": 1}, {"adapter": "a1"}, {"observation_mode": COMPLETE_HISTORY}):
            with self.assertRaisesRegex(ValueError, "not targets for this update"):
                check_provenance({**payload, **changes}, iteration=2, adapter="a2", observation_mode=BOUNDED_HISTORY)
        with self.assertRaisesRegex(ValueError, "contract"):
            check_provenance({**payload, "contract": "other"}, iteration=2, adapter="a2", observation_mode=BOUNDED_HISTORY)


class ContinuationPhaseTests(unittest.TestCase):
    """The driver's continuation phase and the update that reads its values, on the toy loop and the tiny model."""

    def test_the_phase_branches_the_sampled_decisions_and_the_update_credits_them(self) -> None:
        try:
            import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch is not installed")
        import psse_env.rl.train as train
        from psse_env.rl.test_train_series import _config, _tiny_model_phase
        from psse_env.rl.train import Manifest, iteration_dir, load_manifest, run_phase

        evaluator = ToyEvaluator()
        first, second = scenario("r1"), scenario("r2", name="rx2_distant")
        originals = {name: getattr(train, name) for name in
                     ("run_branch", "configure_research_environment", "sampling_policy_factory", "_alternative_renderer",
                      "select_iteration_roots", "load_suite")}
        train.run_branch = functools.partial(run_branch, evaluator=evaluator)
        train.configure_research_environment = lambda config: (lambda **_k: None)
        train.sampling_policy_factory = lambda config, adapter, env_factory=None: (lambda seed: Scripted(lines=(7,)))
        train._alternative_renderer = lambda config, adapter: render_other_choice
        train.select_iteration_roots = lambda config, scenarios, iteration: [first, second]
        train.load_suite = lambda path: [first, second]
        try:
            with tempfile.TemporaryDirectory() as directory, _tiny_model_phase({}) as tiny:
                config = _config(directory, max_steps=12, evaluation_budgets=[8], decision_credit={
                    "enabled": True, "weight": 1.5, "states_per_iteration": 3, "continuations": 1, "include_source_return": False})
                store = TrajectoryStore(iteration_dir(config, 0) / "trajectories")
                store.write_group("r1", rollouts(evaluator, first, [Scripted(lines=(3,)), Scripted(lines=(3,)),
                                                                     Scripted(lines=(3, 7), decline=(3,)), Scripted(lines=(7,))]))
                store.write_group("r2", rollouts(evaluator, second, [Scripted(lines=(7,)) for _ in range(4)]))
                Manifest(phase="continuation", current_adapter=config.initial_adapter).save(Path(directory) / "manifest.json")
                shard = run_phase(config, "continuation", shard=(0, 2))
                self.assertEqual((shard["states"], shard["branches_run"]), (2, 4))
                self.assertEqual(load_manifest(config).phase, "continuation")  # a shard closes nothing
                result = run_phase(config, "continuation")
                self.assertEqual(result["branches_run"], 2)  # what the other shard left
                covered = result["selection"]["sampled_rows_covered"]
                # The wrong line's state covers the three samples that decided there, r2's first candidate all four.
                self.assertEqual((result["selection"]["states_selected"], result["selection"]["roots_selected"], covered), (3, 2, 8))
                self.assertEqual(result["probability_sources"], {"decision_token": covered})
                self.assertEqual(load_manifest(config).phase, "update")
                values = json.loads((iteration_dir(config, 0) / "decision_values.json").read_text(encoding="utf-8"))
                self.assertEqual((values["adapter"], values["iteration"], values["observation_mode"]), (config.initial_adapter, 0, BOUNDED_HISTORY))
                self.assertEqual((values["summary"]["states"], values["summary"]["covered"], values["summary"]["branches_invalid"]), (3, 3, 0))
                kept_wrong = next(state for state in values["states"] if state["stratum"] == "wrong_target")
                # Committing the wrong line fails either way; declining it and correcting line 7 solves the case.
                self.assertEqual((kept_wrong["solved"]["forced"][COMMIT_STATE], kept_wrong["solved"]["forced"][ROLLBACK_STATE]), (0, 1))
                self.assertLess(kept_wrong["q_gap_commit_minus_decline"], 0.0)
                self.assertEqual(len(kept_wrong["rows"]), 3)  # the three samples that decided at that state
                # A second run of the phase finds every branch and reruns none.
                Manifest(phase="continuation", current_adapter=config.initial_adapter).save(Path(directory) / "manifest.json")
                self.assertEqual(run_phase(config, "continuation")["branches_run"], 0)
                report = tiny.phase_update(config, Manifest(phase="update", current_adapter=config.initial_adapter))
                rows = report["rows"]
                self.assertEqual((rows["decision_values"], rows["decision_values_unmatched"], rows["decision_credit_weight"]), (covered, 0, 1.5))
                self.assertEqual(rows["decision_credit_rows"] + rows["decision_credit_covered_zero_advantage"], covered)
                self.assertEqual(rows["supervised_rows_replaced_by_credit"], covered)
                self.assertGreaterEqual(rows["decision_credit_rows"], 3)  # the wrong line's three sampled decisions
                self.assertEqual(report["update"]["row_accounting"]["rows_by_kind"]["decision_credit"], rows["decision_credit_rows"])
                self.assertTrue(report["integrity"]["ok"], report["integrity"])
                self.assertEqual(report["decision_credit"]["values"], covered)
                self.assertTrue((iteration_dir(config, 0) / "adapter" / "weights.pt").is_file())
                self.assertTrue(torch.isfinite(torch.load(iteration_dir(config, 0) / "adapter" / "weights.pt")["table"]).all())
        finally:
            for name, original in originals.items():
                setattr(train, name, original)


if __name__ == "__main__":
    unittest.main()
