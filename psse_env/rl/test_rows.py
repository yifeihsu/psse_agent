from __future__ import annotations

import unittest

from psse_env.rl.advantages import group_advantages
from psse_env.rl.rollout import StepRecord, Trajectory
from psse_env.rl.rows import build_rows, build_step_row


def _render(observation):
    return "P:" + observation["active_state_id"]


def _tokenize(rendered: str):
    ids = [ord(ch) for ch in rendered]
    return ids, len(ids)


def _trajectory(root: str, sample: int, *, steps: int, pairing_ok: bool = True, prompt_tokens: int | None = None) -> Trajectory:
    records = []
    for index in range(steps):
        observation = {"active_state_id": f"{root}:s{index}"}
        expected = len(_render(observation)) if prompt_tokens is None else prompt_tokens
        records.append(StepRecord(step=index, model_observation=observation, prompt_tokens=expected,
                                  original_prompt_tokens=expected, completion_token_ids=[7, 8, 9][: index + 1],
                                  text="x", action={"tool": "run_wls", "arguments": {}}, tool="run_wls",
                                  execution_status="success" if index else "failure", error_code=None))
    return Trajectory(root=root, family="parameter", sample_index=sample, episode_seed=1, steps=records,
                      reward={"total": 1.0}, evaluation={}, pairing_ok=pairing_ok)


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


if __name__ == "__main__":
    unittest.main()
