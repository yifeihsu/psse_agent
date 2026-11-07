from __future__ import annotations

import math
import unittest

from psse_env.rl.advantages import group_advantages
from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.rollout import StepRecord, Trajectory
from psse_env.rl.rows import MIN_SAMPLING_PROBABILITY, RowSettings, build_rows, build_step_row, decision_terms, sampling_logprobs_for


def _render(observation):
    return "P:" + observation["active_state_id"]


def _tokenize(rendered: str):
    ids = [ord(ch) for ch in rendered]
    return ids, len(ids)


def _trajectory(root: str, sample: int, *, steps: int, pairing_ok: bool = True, prompt_tokens: int | None = None,
                with_probabilities: bool = False, digest: str = "none") -> Trajectory:
    """``digest``: ``none`` (no record), ``match`` (of the re-rendered prompt) or ``wrong``."""
    records = []
    for index in range(steps):
        observation = {"active_state_id": f"{root}:s{index}"}
        expected = len(_render(observation)) if prompt_tokens is None else prompt_tokens
        completion = [7, 8, 9][: index + 1]
        recorded = None
        if digest == "match":
            recorded = prompt_token_digest(_tokenize(_render(observation))[0])
        elif digest == "wrong":
            recorded = prompt_token_digest([0])
        records.append(StepRecord(step=index, model_observation=observation, prompt_tokens=expected,
                                  original_prompt_tokens=expected, completion_token_ids=completion,
                                  text="x", action={"tool": "run_wls", "arguments": {}}, tool="run_wls",
                                  execution_status="success" if index else "failure", error_code=None,
                                  token_probabilities=[0.5] * len(completion) if with_probabilities else [],
                                  prompt_digest=recorded))
    return Trajectory(root=root, family="parameter", sample_index=sample, episode_seed=1, steps=records,
                      reward={"total": 1.0}, evaluation={}, pairing_ok=pairing_ok)


def _build(groups, **settings):
    return build_rows(groups, render_prompt=_render, prompt_token_ids=_tokenize, max_length=64,
                      settings=RowSettings(**settings) if settings else None)


class RowTests(unittest.TestCase):
    def test_step_row_masks_the_prompt_and_supervises_the_sampled_tokens(self) -> None:
        row = build_step_row(root="r", sample_index=0, step=0, advantage=0.5, prompt_ids=[1, 2, 3],
                             completion_ids=[9, 8], max_length=100, required_side_inputs=("mm_token_type_ids",))
        self.assertEqual(row.input_ids, [1, 2, 3, 9, 8])
        self.assertEqual(row.labels, [-100, -100, -100, 9, 8])
        self.assertEqual(row.completion_mask, [0, 0, 0, 1, 1])
        self.assertEqual(row.completion_tokens, 2)
        self.assertEqual(row.side_inputs, {"mm_token_type_ids": [0, 0, 0, 0, 0]})
        self.assertFalse(row.prompt_truncated)
        self.assertEqual(set(row.model_record()), {"input_ids", "attention_mask", "labels", "completion_mask", "mm_token_type_ids"})

    def test_long_prompts_are_left_truncated_and_the_completion_survives(self) -> None:
        row = build_step_row(root="r", sample_index=0, step=0, advantage=1.0, prompt_ids=list(range(10)),
                             completion_ids=[99, 98], max_length=5)
        self.assertEqual(row.input_ids, [7, 8, 9, 99, 98])
        self.assertEqual(row.completion_mask, [0, 0, 0, 1, 1])
        self.assertTrue(row.prompt_truncated)
        with self.assertRaises(ValueError):
            build_step_row(root="r", sample_index=0, step=0, advantage=1.0, prompt_ids=[1], completion_ids=[1, 2, 3], max_length=3)
        with self.assertRaises(ValueError):
            build_step_row(root="r", sample_index=0, step=0, advantage=1.0, prompt_ids=[1], completion_ids=[], max_length=3)

    def test_rows_follow_the_group_advantage_and_skip_uninformative_groups(self) -> None:
        informative = ([_trajectory("a", 0, steps=2), _trajectory("a", 1, steps=1)], group_advantages([1.0, 0.0], root="a"))
        flat = ([_trajectory("b", 0, steps=2), _trajectory("b", 1, steps=2)], group_advantages([1.0, 1.0], root="b"))
        rows, report = build_rows([informative, flat], render_prompt=_render, prompt_token_ids=_tokenize, max_length=64)
        self.assertEqual(report["informative_groups"], 1)
        self.assertEqual(report["rows"], 3)
        self.assertEqual([row.root for row in rows], ["a", "a", "a"])
        self.assertEqual([row.sample_index for row in rows], [0, 0, 1])
        self.assertGreater(rows[0].advantage, 0.0)
        self.assertLess(rows[2].advantage, 0.0)
        self.assertEqual(rows[0].input_ids[-1:], [7])
        self.assertEqual(rows[1].input_ids[-2:], [7, 8])
        self.assertEqual(report["completion_tokens"], 1 + 2 + 1)

    def test_unpaired_trajectories_and_prompt_mismatches_are_counted_not_trained(self) -> None:
        unpaired = _trajectory("c", 0, steps=2, pairing_ok=False)
        mismatched = _trajectory("c", 1, steps=2, prompt_tokens=3)
        rows, report = build_rows([([unpaired, mismatched], group_advantages([1.0, 0.0], root="c"))],
                                  render_prompt=_render, prompt_token_ids=_tokenize, max_length=64)
        self.assertEqual(rows, [])
        self.assertEqual(report["trajectories_unpaired"], 1)
        self.assertEqual(report["prompt_length_mismatches"], 2)

    def test_invalid_steps_can_be_excluded(self) -> None:
        group = ([_trajectory("d", 0, steps=2), _trajectory("d", 1, steps=2)], group_advantages([1.0, 0.0], root="d"))
        rows, report = build_rows([group], render_prompt=_render, prompt_token_ids=_tokenize, max_length=64, include_invalid_steps=False)
        self.assertEqual(report["invalid_steps_excluded"], 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.step == 1 for row in rows))

    def test_recorded_sampling_probabilities_become_reference_log_probabilities(self) -> None:
        row = build_step_row(root="r", sample_index=0, step=0, advantage=1.0, prompt_ids=[1, 2], completion_ids=[9, 8, 7],
                             max_length=16, sampling_probabilities=[1.0, 0.5, 0.0])
        self.assertEqual(len(row.sampling_logprobs), 3)
        self.assertAlmostEqual(row.sampling_logprobs[0], 0.0)
        self.assertAlmostEqual(row.sampling_logprobs[1], math.log(0.5))
        self.assertAlmostEqual(row.sampling_logprobs[2], math.log(MIN_SAMPLING_PROBABILITY))
        self.assertIsNone(build_step_row(root="r", sample_index=0, step=0, advantage=1.0, prompt_ids=[1, 2],
                                         completion_ids=[9, 8, 7], max_length=16).sampling_logprobs)
        self.assertIsNone(sampling_logprobs_for([9, 8, 7], [1.0, 0.5]))
        self.assertIsNone(sampling_logprobs_for([9, 8], [0.5, float("nan")]))
        self.assertIsNone(sampling_logprobs_for([9, 8], [0.5, 1.5]))
        self.assertIsNone(sampling_logprobs_for([], []))

    def test_rows_carry_the_recorded_sampling_probabilities(self) -> None:
        recorded = ([_trajectory("e", 0, steps=2, with_probabilities=True), _trajectory("e", 1, steps=1, with_probabilities=True)],
                    group_advantages([1.0, 0.0], root="e"))
        rows, report = build_rows([recorded], render_prompt=_render, prompt_token_ids=_tokenize, max_length=64)
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["rows_with_sampling_probabilities"], 3)
        self.assertEqual(rows[1].sampling_logprobs, [math.log(0.5), math.log(0.5)])
        bare = ([_trajectory("f", 0, steps=1), _trajectory("f", 1, steps=1)], group_advantages([1.0, 0.0], root="f"))
        rows, report = build_rows([bare], render_prompt=_render, prompt_token_ids=_tokenize, max_length=64)
        self.assertEqual(report["rows_with_sampling_probabilities"], 0)
        self.assertTrue(all(row.sampling_logprobs is None for row in rows))

    def test_rows_know_their_trajectorys_token_total(self) -> None:
        group = ([_trajectory("g", 0, steps=2), _trajectory("g", 1, steps=1)], group_advantages([1.0, 0.0], root="g"))
        rows, _report = build_rows([group], render_prompt=_render, prompt_token_ids=_tokenize, max_length=64)
        self.assertEqual([row.completion_tokens for row in rows], [1, 2, 1])
        self.assertEqual([row.trajectory_tokens for row in rows], [3, 3, 1])
        self.assertIsNone(build_step_row(root="r", sample_index=0, step=0, advantage=1.0, prompt_ids=[1],
                                         completion_ids=[2], max_length=8).trajectory_tokens)


class RowSettingsTests(unittest.TestCase):
    def _informative(self, root: str, **kwargs):
        return ([_trajectory(root, 0, steps=2, **kwargs), _trajectory(root, 1, steps=1, **kwargs)],
                group_advantages([1.0, 0.0], root=root))

    def _flat(self, root: str, **kwargs):
        return ([_trajectory(root, 0, steps=2, **kwargs), _trajectory(root, 1, steps=2, **kwargs)],
                group_advantages([1.0, 1.0], root=root))

    def test_defaults_keep_the_old_behaviour_and_the_report_names_the_settings(self) -> None:
        rows, report = _build([self._informative("a"), self._flat("b")])
        self.assertEqual((report["rows"], report["policy_rows"], report["preservation_rows"]), (3, 3, 0))
        self.assertEqual(report["settings"]["preservation"], "none")
        self.assertTrue(all(row.prompt_digest_matches is None and not row.preservation for row in rows))
        self.assertEqual(rows[0].trajectory_key, ("a", 0))

    def test_prompt_digest_is_checked_when_recorded_and_can_be_required(self) -> None:
        rows, report = _build([self._informative("a", digest="match")])
        self.assertTrue(all(row.prompt_digest_matches for row in rows))
        self.assertEqual(report["prompt_digest_mismatches"], 0)
        rows, report = _build([self._informative("a", digest="wrong")])
        self.assertEqual(report["prompt_digest_mismatches"], 3)
        self.assertEqual(len(rows), 3)  # counted, still trained, unless required ...
        rows, report = _build([self._informative("a", digest="wrong")], require_prompt_digest=True)
        self.assertEqual((rows, report["prompt_digest_mismatches"]), ([], 3))
        rows, report = _build([self._informative("a")], require_prompt_digest=True)
        self.assertEqual((rows, report["prompt_digest_missing"]), ([], 3))
        rows, report = _build([self._informative("a", digest="match")], require_prompt_digest=True)
        self.assertEqual(len(rows), 3)

    def test_update_side_truncation_can_be_rejected(self) -> None:
        group = self._informative("a")  # six-token prompts, one- or two-token completions
        rows, report = build_rows([group], render_prompt=_render, prompt_token_ids=_tokenize, max_length=5)
        self.assertEqual(report["prompt_truncated_rows"], 3)
        self.assertEqual(len(rows), 3)
        rows, report = build_rows([group], render_prompt=_render, prompt_token_ids=_tokenize, max_length=5,
                                  settings=RowSettings(reject_truncated_prompts=True))
        self.assertEqual((rows, report["prompt_truncated_rows_rejected"]), ([], 3))

    def test_a_group_with_an_unpaired_trajectory_can_be_skipped_whole(self) -> None:
        group = ([_trajectory("c", 0, steps=2, pairing_ok=False), _trajectory("c", 1, steps=2)],
                 group_advantages([1.0, 0.0], root="c"))
        rows, report = _build([group])
        self.assertEqual((len(rows), report["trajectories_unpaired"], report["groups_incomplete"]), (2, 1, 0))
        rows, report = _build([group], require_complete_groups=True)
        self.assertEqual((rows, report["groups_incomplete"], report["informative_groups"]), ([], 1, 0))

    def test_sampling_records_can_be_required(self) -> None:
        rows, report = _build([self._informative("a")], require_sampling_records=True)
        self.assertEqual((rows, report["rows_without_sampling_record_rejected"]), ([], 3))
        rows, report = _build([self._informative("a", with_probabilities=True)], require_sampling_records=True)
        self.assertEqual(len(rows), 3)

    def test_preservation_rows_come_from_uninformative_groups_with_a_zero_advantage(self) -> None:
        groups = [self._informative("a"), self._flat("b"), self._flat("c")]
        rows, report = _build(groups, preservation="one")
        self.assertEqual((report["policy_rows"], report["preservation_groups"], report["preservation_rows"]), (3, 2, 4))
        kept = [row for row in rows if row.preservation]
        self.assertTrue(all(row.advantage == 0.0 for row in kept))
        self.assertEqual({row.root for row in kept}, {"b", "c"})
        self.assertEqual({row.sample_index for row in kept}, {0})  # one trajectory per group
        self.assertEqual([row.trajectory_tokens for row in kept], [3, 3, 3, 3])
        rows, report = _build(groups, preservation="all")
        self.assertEqual(report["preservation_rows"], 8)
        rows, report = _build(groups, preservation="one", preservation_max_groups=1, seed=5)
        self.assertEqual(report["preservation_groups"], 1)
        again, _ = _build(groups, preservation="one", preservation_max_groups=1, seed=5)
        self.assertEqual([r.root for r in rows], [r.root for r in again])  # seeded draw
        rows, report = _build(groups, preservation="none")
        self.assertEqual(report["preservation_rows"], 0)
        with self.assertRaises(ValueError):
            RowSettings(preservation="some").validate()
        with self.assertRaises(ValueError):
            RowSettings.from_mapping({"preservation": "one", "bogus": 1})


def _with_decisions(trajectory: Trajectory, decisions) -> Trajectory:
    """Mark steps as labelled lifecycle decisions: ``(position, tool, agrees)`` triples."""
    trajectory.lifecycle = []
    for position, tool, agrees in decisions:
        step = trajectory.steps[position]
        step.tool = tool
        step.action = {"tool": tool, "arguments": {}}
        declined = tool == "rollback_state"
        other = "commit_state" if declined else "rollback_state"
        trajectory.lifecycle.append({"position": position, "tool": tool, "declined": declined, "agrees": agrees,
                                     "label": tool if agrees else other, "target_is_true": (not declined) if agrees else declined})
    return trajectory


def _rows(groups, *, settings=None, credit=0.0):
    return build_rows(groups, render_prompt=_render, prompt_token_ids=_tokenize, max_length=64, settings=settings,
                      lifecycle_credit=credit)


class LifecycleRowTests(unittest.TestCase):
    """The lifecycle term (2026-11-07): on policy rows beside the advantage, and rows of its own in uninformative groups."""

    def _groups(self, *, records: bool = True):
        a0 = _trajectory("a", 0, steps=3)
        b0 = _trajectory("b", 0, steps=2)
        if records:
            _with_decisions(a0, [(1, "commit_state", True)])
            _with_decisions(b0, [(0, "rollback_state", False)])
        return [([a0, _trajectory("a", 1, steps=1)], group_advantages([1.0, 0.0], root="a")),
                ([b0, _trajectory("b", 1, steps=2)], group_advantages([1.0, 1.0], root="b"))]

    def test_terms_ride_on_policy_rows_and_make_rows_of_uninformative_groups(self) -> None:
        rows, report = _rows(self._groups(), credit=0.25)
        by_key = {(row.root, row.sample_index, row.step): row for row in rows}
        self.assertEqual(sorted(by_key), [("a", 0, 0), ("a", 0, 1), ("a", 0, 2), ("a", 1, 0), ("b", 0, 0)])
        decision = by_key[("a", 0, 1)]
        self.assertEqual((decision.lifecycle_term, decision.policy_row, decision.lifecycle_only, decision.preservation), (0.25, True, False, False))
        self.assertEqual(decision.advantage, by_key[("a", 0, 0)].advantage)  # the group advantage, unchanged
        self.assertEqual(by_key[("a", 0, 0)].lifecycle_term, 0.0)
        standalone = by_key[("b", 0, 0)]
        self.assertEqual((standalone.advantage, standalone.lifecycle_term, standalone.lifecycle_only, standalone.preservation,
                          standalone.policy_row), (0.0, -0.25, True, False, False))
        self.assertEqual(standalone.trajectory_tokens, standalone.completion_tokens)
        self.assertEqual(
            (report["lifecycle_rows"], report["lifecycle_rows_agreeing"], report["lifecycle_rows_disagreeing"],
             report["lifecycle_only_rows"], report["lifecycle_only_trajectories"], report["lifecycle_decisions_labelled"],
             report["preservation_rows"], report["policy_rows"], report["lifecycle_credit"]),
            (2, 1, 1, 1, 1, 2, 0, 4, 0.25),
        )

    def test_preservation_trajectories_keep_their_decision_rows_as_lifecycle_rows(self) -> None:
        def b_rows(rows):
            return [(row.sample_index, row.step, row.preservation, row.lifecycle_only, row.lifecycle_term) for row in rows if row.root == "b"]

        rows, report = _rows(self._groups(), settings=RowSettings(preservation="all"), credit=0.25)
        self.assertEqual(b_rows(rows), [(0, 0, False, True, -0.25), (0, 1, True, False, 0.0), (1, 0, True, False, 0.0), (1, 1, True, False, 0.0)])
        self.assertEqual((report["preservation_groups"], report["preservation_rows"], report["lifecycle_only_rows"]), (1, 3, 1))
        # ``one`` keeps the first paired trajectory; the second has no decision, so it adds nothing.
        rows, report = _rows(self._groups(), settings=RowSettings(preservation="one"), credit=0.25)
        self.assertEqual(b_rows(rows), [(0, 0, False, True, -0.25), (0, 1, True, False, 0.0)])
        self.assertEqual((report["preservation_rows"], report["lifecycle_only_rows"]), (1, 1))
        # A draw that keeps no group still yields the decision rows.
        rows, report = _rows(self._groups(), settings=RowSettings(preservation="one", preservation_max_groups=0), credit=0.25)
        self.assertEqual(b_rows(rows), [(0, 0, False, True, -0.25)])
        self.assertEqual((report["preservation_groups"], report["preservation_rows"], report["lifecycle_only_rows"]), (0, 0, 1))

    def test_credit_zero_is_the_builder_as_before(self) -> None:
        def key(rows):
            return [(r.root, r.sample_index, r.step, r.advantage, r.preservation, r.lifecycle_only, r.lifecycle_term, r.trajectory_tokens)
                    for r in rows]

        for settings in (None, RowSettings(preservation="all"), RowSettings(preservation="one", preservation_max_groups=1, seed=3),
                         RowSettings(preservation="none")):
            with_records, report = _rows(self._groups(), settings=settings, credit=0.0)
            plain, plain_report = _rows(self._groups(records=False), settings=settings)
            self.assertEqual(key(with_records), key(plain))
            self.assertTrue(all(row.lifecycle_term == 0.0 and not row.lifecycle_only for row in with_records))
            self.assertEqual({k: v for k, v in report.items() if not k.startswith("lifecycle")},
                             {k: v for k, v in plain_report.items() if not k.startswith("lifecycle")})
            self.assertEqual((report["lifecycle_decisions_labelled"], report["lifecycle_rows"]), (2, 0))

    def test_incomplete_informative_groups_still_yield_their_decision_rows(self) -> None:
        a0 = _with_decisions(_trajectory("a", 0, steps=2), [(1, "commit_state", False)])
        a1 = _trajectory("a", 1, steps=1, pairing_ok=False)
        rows, report = _rows([([a0, a1], group_advantages([1.0, 0.0], root="a"))],
                             settings=RowSettings(require_complete_groups=True), credit=0.5)
        self.assertEqual([(row.step, row.advantage, row.lifecycle_term, row.lifecycle_only) for row in rows], [(1, 0.0, -0.5, True)])
        self.assertEqual((report["groups_incomplete"], report["informative_groups"], report["policy_rows"], report["lifecycle_only_rows"]),
                         (1, 0, 0, 1))

    def test_decisions_that_do_not_pair_with_their_steps_are_not_scored(self) -> None:
        trajectory = _trajectory("a", 0, steps=2)
        trajectory.lifecycle = [{"position": 5, "tool": "commit_state", "agrees": True},  # out of range
                                {"position": 1, "tool": "commit_state", "agrees": True},  # the step is a run_wls step
                                {"position": 0, "tool": "rollback_state", "agrees": None}]  # unlabelled
        self.assertEqual(decision_terms(trajectory, 0.25), {})
        trajectory.steps[1].tool = "commit_state"
        self.assertEqual(decision_terms(trajectory, 0.25), {1: 0.25})
        self.assertEqual(decision_terms(trajectory, 0.0), {})

    def test_supervised_form_adds_a_row_per_labelled_decision_with_the_audits_choice(self) -> None:
        asked = []

        def render_target(observation, text, decision):
            asked.append((observation["active_state_id"], text, decision["label"]))
            return [11, 12]

        rows, report = build_rows(self._groups(), render_prompt=_render, prompt_token_ids=_tokenize, max_length=64,
                                  lifecycle_credit=0.25, lifecycle_form="supervised", render_target=render_target)
        supervised = [row for row in rows if row.supervised]
        self.assertEqual([(row.root, row.sample_index, row.step) for row in supervised], [("a", 0, 1), ("b", 0, 0)])
        for row in supervised:
            self.assertEqual((row.advantage, row.lifecycle_term, row.lifecycle_only, row.preservation, row.policy_row), (0.0, 0.25, False, False, False))
            self.assertEqual(row.input_ids[-2:], [11, 12])
            self.assertEqual(row.completion_tokens, 2)
            self.assertEqual(row.trajectory_tokens, 2)
            self.assertIsNone(row.sampling_logprobs)
        # a0 committed a true target (label: commit); b0 declined a true target (label: commit).
        self.assertEqual(asked, [("a:s1", "x", "commit_state"), ("b:s0", "x", "commit_state")])
        # The sampled decision rows carry no term in this form; the policy rows of group a are as without a credit.
        self.assertTrue(all(row.lifecycle_term == 0.0 for row in rows if not row.supervised))
        self.assertEqual(sum(1 for row in rows if row.policy_row), 4)
        self.assertEqual((report["lifecycle_form"], report["supervised_decisions"], report["supervised_rows"], report["supervised_rows_agreeing"],
                          report["supervised_rows_disagreeing"], report["supervised_tokens"], report["lifecycle_rows"], report["lifecycle_only_rows"],
                          report["rows"]), ("supervised", 2, 2, 1, 1, 4, 0, 0, 6))

    def test_supervised_rows_follow_the_prompt_checks_and_count_unrenderable_targets(self) -> None:
        def failing(observation, text, decision):
            if decision["tool"] == "commit_state":  # a0's decision
                raise ValueError("no render")
            return [11]

        rows, report = build_rows(self._groups(), render_prompt=_render, prompt_token_ids=_tokenize, max_length=64,
                                  lifecycle_credit=0.25, lifecycle_form="supervised", render_target=failing)
        self.assertEqual([(row.root, row.step) for row in rows if row.supervised], [("b", 0)])
        self.assertEqual((report["supervised_decisions"], report["supervised_rows"], report["supervised_rows_unrenderable"]), (2, 1, 1))
        # A supervised row needs no sampling record, and the prompt digest check applies to it as to any row.
        a0 = _with_decisions(_trajectory("a", 0, steps=2, with_probabilities=True, digest="wrong"), [(1, "commit_state", True)])
        a1 = _trajectory("a", 1, steps=1, with_probabilities=True, digest="wrong")
        rows, report = build_rows([([a0, a1], group_advantages([1.0, 0.0], root="a"))], render_prompt=_render, prompt_token_ids=_tokenize,
                                  max_length=64, settings=RowSettings(require_sampling_records=True, require_prompt_digest=True),
                                  lifecycle_credit=0.25, lifecycle_form="supervised", render_target=lambda *args: [11])
        self.assertEqual(rows, [])
        self.assertEqual(report["prompt_digest_mismatches"], 4)
        a0 = _with_decisions(_trajectory("a", 0, steps=2, with_probabilities=True, digest="match"), [(1, "commit_state", True)])
        a1 = _trajectory("a", 1, steps=1, with_probabilities=True, digest="match")
        rows, report = build_rows([([a0, a1], group_advantages([1.0, 0.0], root="a"))], render_prompt=_render, prompt_token_ids=_tokenize,
                                  max_length=64, settings=RowSettings(require_sampling_records=True, require_prompt_digest=True),
                                  lifecycle_credit=0.25, lifecycle_form="supervised", render_target=lambda *args: [11])
        self.assertEqual((report["supervised_rows"], report["rows_without_sampling_record_rejected"], report["policy_rows"]), (1, 0, 3))
        with self.assertRaisesRegex(ValueError, "target renderer"):
            build_rows(self._groups(), render_prompt=_render, prompt_token_ids=_tokenize, max_length=64, lifecycle_credit=0.25,
                       lifecycle_form="supervised")
        with self.assertRaisesRegex(ValueError, "lifecycle_form"):
            build_rows(self._groups(), render_prompt=_render, prompt_token_ids=_tokenize, max_length=64, lifecycle_form="bogus")
        # With the credit at zero the supervised form builds nothing extra.
        rows, report = build_rows(self._groups(), render_prompt=_render, prompt_token_ids=_tokenize, max_length=64, lifecycle_form="supervised")
        self.assertEqual((report["supervised_rows"], report["rows"]), (0, 4))

    def test_inconsistent_lifecycle_flags_and_credits_are_refused(self) -> None:
        for bad in (dict(advantage=0.5, lifecycle_term=0.5), dict(advantage=0.0), dict(advantage=0.0, lifecycle_term=0.5, lifecycle_only=True),
                    dict(advantage=0.0, lifecycle_term=0.5, sampling_probabilities=[0.5]), dict(advantage=0.0, lifecycle_term=-0.5)):
            with self.assertRaisesRegex(ValueError, "supervised row"):
                build_step_row(root="r", sample_index=0, step=0, prompt_ids=[1], completion_ids=[2], max_length=8, supervised=True, **bad)
        with self.assertRaises(ValueError):
            build_step_row(root="r", sample_index=0, step=0, advantage=0.0, prompt_ids=[1], completion_ids=[2], max_length=8,
                           preservation=True, lifecycle_term=0.5, supervised=True)
        with self.assertRaises(ValueError):
            build_step_row(root="r", sample_index=0, step=0, advantage=0.0, prompt_ids=[1], completion_ids=[2], max_length=8, lifecycle_only=True)
        with self.assertRaises(ValueError):
            build_step_row(root="r", sample_index=0, step=0, advantage=0.0, prompt_ids=[1], completion_ids=[2], max_length=8,
                           preservation=True, lifecycle_term=0.5)
        with self.assertRaises(ValueError):
            _rows([], credit=-1.0)


if __name__ == "__main__":
    unittest.main()
