from __future__ import annotations

import unittest

from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.rollout import StepRecord, Trajectory, pair_records_with_trace


class PromptDigestTests(unittest.TestCase):
    def test_digest_depends_on_the_exact_token_sequence(self) -> None:
        self.assertEqual(prompt_token_digest([1, 2, 3]), prompt_token_digest((1, 2, 3)))
        self.assertEqual(prompt_token_digest(iter([1, 2, 3])), prompt_token_digest([1, 2, 3]))
        self.assertNotEqual(prompt_token_digest([1, 2, 3]), prompt_token_digest([1, 2, 3, 4]))
        self.assertNotEqual(prompt_token_digest([1, 2, 3]), prompt_token_digest([3, 2, 1]))
        self.assertNotEqual(prompt_token_digest([12, 3]), prompt_token_digest([1, 23]))
        self.assertEqual(len(prompt_token_digest([])), 64)

    def test_step_records_carry_the_digest_through_pairing_and_serialization(self) -> None:
        digest = prompt_token_digest([5, 6, 7])
        records = [{"action": {"tool": "run_wls"}, "model_observation": {"a": 1}, "completion_token_ids": [1],
                    "prompt_tokens": 3, "original_prompt_tokens": 3, "text": "x", "prompt_token_digest": digest},
                   {"action": {"tool": "commit_state"}, "model_observation": {"a": 2}, "completion_token_ids": [2],
                    "prompt_tokens": 3, "original_prompt_tokens": 3, "text": "y"}]
        trace = [{"step": 0, "action": {"tool": "run_wls"}, "execution_status": "success"},
                 {"step": 1, "action": {"tool": "commit_state"}, "execution_status": "success"}]
        steps, ok, note = pair_records_with_trace(records, trace)
        self.assertTrue(ok, note)
        self.assertEqual([step.prompt_digest for step in steps], [digest, None])
        trajectory = Trajectory(root="r", family="parameter", sample_index=0, episode_seed=0, steps=steps,
                                reward={"total": 1.0}, evaluation={}, pairing_ok=True)
        reloaded = Trajectory.from_dict(trajectory.as_dict())
        self.assertEqual([step.prompt_digest for step in reloaded.steps], [digest, None])
        self.assertIsNone(StepRecord.from_dict({"step": 0}).prompt_digest)


if __name__ == "__main__":
    unittest.main()
