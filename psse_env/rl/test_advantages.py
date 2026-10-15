from __future__ import annotations

import math
import unittest

from psse_env.rl.advantages import group_advantages, summarize_groups


class AdvantageTests(unittest.TestCase):
    def test_grpo_centers_and_scales_within_the_group(self) -> None:
        group = group_advantages([1.0, 0.0, 1.0, 0.0], root="r")
        self.assertTrue(group.informative)
        self.assertAlmostEqual(sum(group.advantages), 0.0)
        self.assertAlmostEqual(group.std, 0.5)
        self.assertAlmostEqual(group.advantages[0], 0.5 / (0.5 + 1e-6))
        self.assertAlmostEqual(group.advantages[1], -0.5 / (0.5 + 1e-6))

    def test_unscaled_grpo_keeps_reward_units(self) -> None:
        group = group_advantages([0.8, 0.2], scale="none")
        self.assertAlmostEqual(group.advantages[0], 0.3)
        self.assertAlmostEqual(group.advantages[1], -0.3)

    def test_rloo_uses_the_leave_one_out_baseline(self) -> None:
        group = group_advantages([1.0, 0.0, 0.0, 0.0], method="rloo", scale="none")
        self.assertAlmostEqual(group.advantages[0], 1.0)
        self.assertAlmostEqual(group.advantages[1], -1.0 / 3.0)

    def test_a_group_without_variance_carries_no_signal(self) -> None:
        for rewards in ([1.0, 1.0, 1.0], [0.0], [-0.3, -0.3]):
            group = group_advantages(rewards)
            self.assertFalse(group.informative)
            self.assertEqual(set(group.advantages), {0.0})

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            group_advantages([])
        with self.assertRaises(ValueError):
            group_advantages([math.nan, 1.0])
        with self.assertRaises(ValueError):
            group_advantages([1.0, 0.0], method="ppo")
        with self.assertRaises(ValueError):
            group_advantages([1.0, 0.0], scale="batch")

    def test_summary_counts_informative_groups(self) -> None:
        groups = {"a": group_advantages([1.0, 0.0], root="a"), "b": group_advantages([1.0, 1.0], root="b")}
        summary = summarize_groups(groups)
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["informative_groups"], 1)
        self.assertAlmostEqual(summary["informative_fraction"], 0.5)
        self.assertAlmostEqual(summary["mean_reward"], 0.75)
        self.assertEqual(summarize_groups({})["groups"], 0)


if __name__ == "__main__":
    unittest.main()
