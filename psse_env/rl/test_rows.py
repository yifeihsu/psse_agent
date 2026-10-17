from __future__ import annotations

import math
import unittest

from psse_env.rl.advantages import group_advantages
from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.rollout import StepRecord, Trajectory
from psse_env.rl.rows import MIN_SAMPLING_PROBABILITY, RowSettings, build_rows, build_step_row, sampling_logprobs_for


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


if __name__ == "__main__":
    unittest.main()
