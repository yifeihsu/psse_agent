from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.rl.rollout import StepRecord, Trajectory, TrajectoryStore
from research.rl.signal_report import analyze_store, group_signal, main


def trajectory(sample, *, reward=0.8, success=True, credit=1.0, calls=5, target=1, tokens=None, invalid=0):
    step = StepRecord(step=0, model_observation={"state": sample}, prompt_tokens=1, original_prompt_tokens=1,
                      completion_token_ids=tokens or [1, 2], text="", tool="correct_parameters",
                      action={"tool": "correct_parameters", "arguments": {"state_id": f"sample{sample}:s0", "line_index": target}},
                      execution_status="success", error_code=None)
    basis = "counterfactual_resolution" if success else "none"
    evaluation = {"truth_audited_task_success": success, "policy_steps": calls, "invalid_action_count": invalid,
                  "audit": {"truth_audited_task_assessment": {"basis": basis}}}
    return Trajectory(root="r", family="parameter", sample_index=sample, episode_seed=sample,
                      steps=[step], reward={"total": reward, "credit": credit, "basis": basis,
                                           "calls": calls, "invalid_actions": invalid},
                      evaluation=evaluation, pairing_ok=True)


class SignalReportTests(unittest.TestCase):
    def test_cost_spread_is_distinct_from_diagnostic_signal(self):
        record = group_signal([trajectory(0), trajectory(1, reward=0.78, calls=6)])
        self.assertTrue(record["informative"])
        self.assertTrue(record["cost_only_signal"])
        self.assertFalse(record["mixed_success"])
        self.assertFalse(record["mixed_decisive"])
        self.assertEqual(record["semantic_distinct"], 1)

    def test_token_variation_does_not_imply_another_target(self):
        record = group_signal([trajectory(0), trajectory(1, tokens=[9, 8])])
        self.assertEqual(record["token_distinct"], 2)
        self.assertEqual(record["semantic_distinct"], 1)
        self.assertEqual(record["first_correction_targets"], 1)
        changed = group_signal([trajectory(0), trajectory(1, target=2)])
        self.assertEqual(changed["first_correction_targets"], 2)

    def test_constant_failures_have_no_grpo_signal(self):
        record = group_signal([trajectory(i, success=False, credit=0, reward=-0.2) for i in range(4)])
        self.assertTrue(record["all_fail_zero_signal"])
        self.assertEqual(record["advantages"]["grpo_group"], [0.0] * 4)

    def test_mixed_success_and_improvement_over_frozen_are_reported(self):
        record = group_signal([trajectory(0), trajectory(1, reward=-0.2, success=False, credit=0)],
                              greedy={"truth_audited_task_success": False, "reward": {"total": -0.2, "credit": 0}})
        self.assertTrue(record["mixed_success"])
        self.assertTrue(record["mixed_decisive"])
        self.assertFalse(record["cost_only_signal"])
        self.assertTrue(record["greedy_comparison"]["solves_greedy_failure"])

    def test_tiny_cost_difference_gets_the_same_scaled_advantage_as_success(self):
        cheap = group_signal([trajectory(0), trajectory(1, reward=0.78)])
        meaningful = group_signal([trajectory(0), trajectory(1, reward=-0.2, success=False, credit=0)])
        self.assertAlmostEqual(cheap["advantages"]["grpo_group"][0], meaningful["advantages"]["grpo_group"][0], places=3)
        self.assertLess(abs(cheap["advantages"]["rloo_none"][0]), abs(meaningful["advantages"]["rloo_none"][0]) / 40)

    def test_report_reads_saved_groups_and_episode_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "trajectories")
            store.write_group("r", [trajectory(0), trajectory(1, reward=0.78)])
            report = analyze_store(store, strata={"r": "parameter:misranked"})
            self.assertEqual(report["by_stratum"]["parameter:misranked"]["cost_only_signal"], 1)
            baseline = Path(directory) / "evaluation_episodes.jsonl"
            baseline.write_text(json.dumps({"scenario_id": "r", "truth_audited_task_success": True}) + "\n")
            output = Path(directory) / "report.json"
            self.assertEqual(main(["--store", str(store.directory), "--greedy-evaluation", str(baseline), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["summary"]["greedy_comparisons"], 1)


if __name__ == "__main__":
    unittest.main()
