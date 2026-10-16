from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.rl.diagnose import diagnose_groups, divergence, format_summary, main, spread_reason
from psse_env.rl.rollout import StepRecord, Trajectory, TrajectoryStore


def _trajectory(root: str, sample: int, tokens: list[list[int]], *, reward: float, success: bool = True, credit: float = 1.0,
                calls: int | None = None, invalid: int = 0, probabilities: list[float] | None = None,
                family: str = "parameter", pairing_ok: bool = True) -> Trajectory:
    steps = []
    for index, ids in enumerate(tokens):
        steps.append(StepRecord(step=index, model_observation={"s": index}, prompt_tokens=5, original_prompt_tokens=5,
                                completion_token_ids=list(ids), text="x", action={"tool": "run_wls"}, tool="run_wls",
                                execution_status="success", error_code=None,
                                token_probabilities=[probabilities[index]] * len(ids) if probabilities else []))
    return Trajectory(root=root, family=family, sample_index=sample, episode_seed=sample, steps=steps,
                      reward={"total": reward, "credit": credit, "calls": calls if calls is not None else len(tokens),
                              "false_commits": 0, "false_finalizations": 0, "healthy_component_corrupted": False,
                              "loop_detected": False, "invalid_actions": invalid},
                      evaluation={"truth_audited_task_success": success}, pairing_ok=pairing_ok)


def _groups():
    flip = [_trajectory("flip", 0, [[1], [2], [3]], reward=0.94, probabilities=[1.0, 0.6, 1.0]),
            _trajectory("flip", 1, [[1], [9], [3], [4]], reward=-0.08, success=False, credit=0.0, probabilities=[1.0, 0.4, 1.0, 1.0])]
    cost = [_trajectory("cost", 0, [[1], [2], [3]], reward=0.94, probabilities=[1.0, 1.0, 1.0]),
            _trajectory("cost", 1, [[1], [2], [7], [3]], reward=0.90, probabilities=[1.0, 1.0, 0.95, 1.0])]
    penalty = [_trajectory("pen", 0, [[1], [2], [3]], reward=0.94),
               _trajectory("pen", 1, [[1], [2], [3], [3]], reward=0.87, invalid=1, pairing_ok=False)]
    same = [_trajectory("same", 0, [[1], [2], [3]], reward=0.94, family="hif"),
            _trajectory("same", 1, [[1], [2], [3]], reward=0.94, family="hif")]
    return [("flip", flip), ("cost", cost), ("pen", penalty), ("same", same)]


class DiagnoseTests(unittest.TestCase):
    def test_spread_reasons_and_divergence(self) -> None:
        groups = dict(_groups())
        self.assertEqual([spread_reason(groups[k]) for k in ("flip", "cost", "pen", "same")],
                         ["success_flip", "cost_only", "penalty", "none"])
        parted = divergence(groups["flip"])
        self.assertEqual((parted["distinct_trajectories"], parted["first_divergence_step"], parted["modal_steps"]), (2, 1, 3))
        self.assertAlmostEqual(parted["shared_prefix_fraction"], 1 / 3)
        self.assertEqual(divergence(groups["cost"])["first_divergence_step"], 2)
        together = divergence(groups["same"])
        self.assertEqual((together["distinct_trajectories"], together["first_divergence_step"]), (1, None))
        self.assertIsNone(together["shared_prefix_fraction"])

    def test_report_counts_signal_by_reason_and_compares_the_advantage_scales(self) -> None:
        report = diagnose_groups(_groups(), method="grpo", scale="group")
        self.assertEqual((report["groups"], report["episodes"]), (4, 8))
        self.assertAlmostEqual(report["success_rate"], 7 / 8)
        self.assertEqual(report["groups_with_distinct_samples"], 3)
        self.assertEqual(report["informative_groups"], 3)
        self.assertEqual(report["groups_by_spread_reason"], {"none": 1, "success_flip": 1, "penalty": 1, "cost_only": 1})
        self.assertEqual(report["unpaired_trajectories"], 1)
        comparison = report["advantage_scale_comparison"]
        # Standardized by the group, a one-call difference and a success flip weigh the same.
        self.assertAlmostEqual(comparison["success_flip"]["mean_abs_advantage_scaled"], 1.0, places=4)
        self.assertAlmostEqual(comparison["cost_only"]["mean_abs_advantage_scaled"], 1.0, places=4)
        self.assertAlmostEqual(comparison["success_flip"]["mean_abs_advantage_unscaled"], 0.51)
        self.assertAlmostEqual(comparison["cost_only"]["mean_abs_advantage_unscaled"], 0.02)
        self.assertAlmostEqual(comparison["cost_only"]["mean_reward_spread"], 0.04)
        self.assertEqual(report["divergence"]["first_step_histogram"], {"1": 1, "2": 1, "3": 1})
        sampling = report["sampling"]
        self.assertEqual(sampling["sampled_steps"], 14)
        self.assertAlmostEqual(sampling["fraction_of_steps_with_every_token_above_0_99"], 11 / 14)
        self.assertAlmostEqual(sampling["min_token_probability_quantiles"]["p10"], 0.6)
        self.assertEqual(report["by_family"]["hif"], {"groups": 1, "informative": 0, "success_flip": 0, "episodes": 2, "successes": 2})
        self.assertEqual(report["by_family"]["parameter"]["success_flip"], 1)
        self.assertEqual(len(report["groups_detail"]), 4)
        text = format_summary(report)
        self.assertIn("success flip 1, penalty 1, cost only 1", text)
        self.assertIn("cost_only: 1 groups", text)

    def test_cli_reads_a_store_and_writes_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "trajectories")
            for root, group in _groups():
                store.write_group(root, group)
            output = Path(directory) / "report.json"
            self.assertEqual(main(["--iteration", directory, "--output", str(output), "--scale", "none"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["scale"], "none")
            self.assertEqual(report["informative_groups"], 3)
            with self.assertRaises(SystemExit):
                main(["--store", str(Path(directory) / "missing")])


if __name__ == "__main__":
    unittest.main()
