"""The teacher-forced scorer without a model: its reading of the inputs, its records, its shards and its summary.

The model's side (the loader, the renderers, the forward pass) is the RL update's own and is first exercised by the
cell's job; here stand-ins return fixed tokens and log-probabilities.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from research.minimal_env import score_actions

FINISH_RESOLVED = {"tool": "finish", "arguments": {"outcome": "resolved"}}
FINISH_UNRESOLVED = {"tool": "finish", "arguments": {"outcome": "unresolved"}}
REFIT = {"tool": "correct_parameters", "arguments": {"line": 20}}
UNDO = {"tool": "undo", "arguments": {}}


def _item(case: str, split: str = "dev", step: int = 4, actions=None) -> dict:
    return {"contract": "minimal_scoring_inputs_v1", "case": case, "split": split, "class": "rx2_adjacent", "step": step,
            "observation": {"case": case},
            "actions": actions if actions is not None else [
                {"role": "own", "action": FINISH_RESOLVED}, {"role": "repairing", "action": REFIT},
                {"role": "repairing", "action": UNDO}, {"role": "other_finish", "action": FINISH_UNRESOLVED}]}


#: Per action, its completion tokens' log-probabilities.
LOGPROBS = {"finish": [-0.01, -0.02], "correct_parameters": [-2.0, -0.5, -0.1], "undo": [-3.0]}


def _scorer(item):
    def render_target(observation, action):
        if action["tool"] == "inspect":
            raise ValueError("does not render")
        return list(range(len(LOGPROBS[action["tool"]])))

    def logprobs(prompt, completion):
        assert prompt == [7, 8, 9]
        return LOGPROBS[current["tool"]][: len(completion)]

    current: dict = {}

    def render_target_tracking(observation, action):
        current.clear()
        current.update(action)
        return render_target(observation, action)

    return score_actions.score_state(item, view=lambda observation: {**observation, "viewed": True},
                                     render=lambda observation: json.dumps(observation, sort_keys=True),
                                     prompt_ids=lambda text: ([7, 8, 9], 3), render_target=render_target_tracking,
                                     logprobs=logprobs, tokens=lambda ids: [f"t{value}" for value in ids])


class ScoreStateTests(unittest.TestCase):
    def test_an_actions_probability_is_the_product_over_its_tokens(self) -> None:
        result = _scorer(_item("r0_a"))
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertFalse(result["prompt_cut"])
        by_role = {(entry["role"], entry["action"]["tool"]): entry for entry in result["actions"]}
        refit = by_role[("repairing", "correct_parameters")]
        self.assertAlmostEqual(refit["logprob"], -2.6)
        self.assertAlmostEqual(refit["probability"], math.exp(-2.6))
        self.assertEqual(refit["tokens"], 3)
        self.assertEqual(refit["token_text"], ["t0", "t1", "t2"])
        # The repairing actions are different calls: their probabilities add.
        self.assertAlmostEqual(score_actions.role_probability(result, "repairing"), math.exp(-2.6) + math.exp(-3.0))
        self.assertAlmostEqual(score_actions.role_probability(result, "own"), math.exp(-0.03))

    def test_an_action_that_does_not_render_is_recorded_not_scored(self) -> None:
        item = _item("r0_b", actions=[{"role": "own", "action": FINISH_RESOLVED},
                                      {"role": "repairing", "action": {"tool": "inspect", "arguments": {"view": "breakers"}}}])
        result = _scorer(item)
        self.assertIn("unrenderable", result["actions"][1])
        self.assertNotIn("probability", result["actions"][1])
        self.assertIsNone(score_actions.role_probability(result, "repairing"))

    def test_a_scorer_that_returns_the_wrong_count_of_log_probabilities_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            score_actions.score_state(_item("r0_c"), view=dict, render=str, prompt_ids=lambda text: ([1], 1),
                                      render_target=lambda observation, action: [1, 2], logprobs=lambda prompt, completion: [0.0])


class FileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def _inputs(self, name: str, items) -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        return path

    def test_inputs_are_read_whole_and_a_state_twice_is_refused(self) -> None:
        dev = self._inputs("dev.jsonl", [_item("r0_a"), _item("r0_b")])
        train = self._inputs("train.jsonl", [_item("r0_a", split="train")])
        self.assertEqual([score_actions.state_key(item) for item in score_actions.read_inputs([dev, train])],
                         ["dev_r0_a_4", "dev_r0_b_4", "train_r0_a_4"])
        with self.assertRaises(ValueError):
            score_actions.read_inputs([dev, dev])
        wrong = self._inputs("wrong.jsonl", [{**_item("r0_z"), "contract": "other"}])
        with self.assertRaises(ValueError):
            score_actions.read_inputs([wrong])

    def test_shards_split_the_states_and_a_written_state_is_not_scored_again(self) -> None:
        items = [_item(f"r0_{index}") for index in range(5)]
        directory = self.root / "states"
        calls: list[str] = []

        def scorer(item):
            calls.append(item["case"])
            return _scorer(item)

        self.assertEqual(score_actions.score_all(items, directory=directory, shard=(0, 2), scorer=scorer), 3)
        self.assertEqual(calls, ["r0_0", "r0_2", "r0_4"])
        self.assertEqual(score_actions.score_all(items, directory=directory, shard=None, scorer=scorer), 2)
        self.assertEqual(calls[3:], ["r0_1", "r0_3"])
        self.assertEqual(score_actions.score_all(items, directory=directory, shard=None, scorer=scorer), 0)
        self.assertEqual(len(score_actions.read_results(directory)), 5)
        with self.assertRaises(ValueError):
            score_actions.parse_shard("2/2")

    def test_the_summary_reads_the_repairing_probability_per_split(self) -> None:
        results = [_scorer(_item("r0_a")), _scorer(_item("r0_b", split="train")),
                   _scorer({**_item("r0_a", step=2), "point": "divergence"})]
        summary = score_actions.summarize(results)
        self.assertEqual(sorted(summary["by_split"]), ["dev:divergence", "dev:stop", "train:stop"])
        self.assertEqual(summary["by_split"]["dev:divergence"]["counts"]["states"], 1)
        dev = summary["by_split"]["dev:stop"]
        repairing = math.exp(-2.6) + math.exp(-3.0)
        self.assertEqual(dev["counts"]["states"], 1)
        self.assertAlmostEqual(dev["probability"]["repairing"]["median"], repairing)
        self.assertGreater(repairing, 0.1)  # 0.074 + 0.050
        self.assertEqual(dev["counts"]["repairing_at_least_0.5"], 0)
        self.assertEqual(dev["counts"]["repairing_at_least_0.1"], 1)
        reach = dev["expected_states_where_one_of_k_samples_steps_toward_the_repair"]
        self.assertAlmostEqual(reach["8"], round(1 - (1 - repairing) ** 8, 2))


if __name__ == "__main__":
    unittest.main()
