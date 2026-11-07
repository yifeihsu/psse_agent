"""The RL cell's configs load through the driver, name the plan's options, and their class mix matches the training draw."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from psse_env.rl.sampling import RootSampling
from psse_env.rl.train import RLConfig

CELL = Path(__file__).resolve().parent
REPO_ROOT = CELL.parents[2]
#: The training draw's manifest: in this checkout, or in the main checkout when this is one of its worktrees
#: (.claude/worktrees/<name>, three levels below it).
TRAIN_MANIFEST = next(
    (path for path in (root / "output" / "compound_rl_20261106" / "train_ieee14" / "manifest.json"
                       for root in (REPO_ROOT, REPO_ROOT.parents[2] if len(REPO_ROOT.parents) > 2 else REPO_ROOT))
     if path.is_file()),
    REPO_ROOT / "output" / "compound_rl_20261106" / "train_ieee14" / "manifest.json",
)
PLAN_OPTIONS = {"lifecycle_choice": 1, "detailed_evidence": 1, "pair_refit_adjacent": 1, "flow_at_noise_cue": 1, "route_comparison": 0}


class CellConfigTests(unittest.TestCase):
    def test_configs_load_and_name_the_plans_settings(self) -> None:
        for name in ("rl_smoke.json", "rl_ieee14.json"):
            config = RLConfig.load(CELL / name)
            self.assertEqual(config.environment_options, PLAN_OPTIONS, name)
            self.assertEqual(config.evidence_profile, "scada_only")
            self.assertEqual(config.expert_variant, "baseline_compare")
            self.assertEqual((config.max_steps, config.samples_per_root, config.temperature, config.top_p, config.top_k), (40, 4, 1.0, 1.0, None))
            self.assertEqual(config.evaluation_budgets, [16, 20])
            self.assertEqual((config.reward["lifecycle_credit"], config.reward["lifecycle_form"]), (0.25, "supervised"))
            self.assertEqual(config.reward["tool_regret_cost"], 0.0)
            self.assertIsNone(config.expert_reference)
            self.assertFalse(config.load_in_4bit)
            self.assertEqual(config.advantage_scale, "none")
            self.assertEqual((config.update["clip_ratio"], config.update["clip_reference"], config.update["beta_kl"]), (0.2, "training", 0.04))
            self.assertTrue(config.rows["require_sampling_records"])
            self.assertTrue(config.initial_adapter.endswith("/compound_cloning_20261106/out/clone/training/lora"))
        smoke = RLConfig.load(CELL / "rl_smoke.json")
        self.assertEqual((smoke.roots_per_iteration, smoke.development_suite, smoke.evaluate_every_iteration, smoke.baseline_evaluation),
                         (8, None, False, False))
        arm = RLConfig.load(CELL / "rl_ieee14.json")
        self.assertTrue(arm.baseline_evaluation)  # S0 measured under this driver's bf16 loader, not the cloning cell's 4-bit arms
        self.assertTrue(arm.development_suite.endswith("/ieee14_dev300.json"))

    def test_the_comparison_arm_differs_from_the_supervised_arm_in_the_form_alone(self) -> None:
        supervised = json.loads((CELL / "rl_ieee14.json").read_text(encoding="utf-8"))
        sampled = json.loads((CELL / "rl_ieee14_sampled.json").read_text(encoding="utf-8"))
        RLConfig.load(CELL / "rl_ieee14_sampled.json")
        self.assertEqual({key for key in set(supervised) | set(sampled) if supervised.get(key) != sampled.get(key)}, {"output_dir", "reward"})
        self.assertEqual({key for key in supervised["reward"] if supervised["reward"][key] != sampled["reward"][key]}, {"lifecycle_form"})
        self.assertEqual((supervised["reward"]["lifecycle_form"], sampled["reward"]["lifecycle_form"]), ("supervised", "sampled"))
        self.assertNotEqual(supervised["output_dir"], sampled["output_dir"])

    def test_the_stratified_mix_names_every_class_of_the_training_draw(self) -> None:
        config = RLConfig.load(CELL / "rl_ieee14.json")
        sampling = RootSampling.from_mapping(config.root_sampling)
        self.assertEqual((sampling.mode, sampling.key), ("stratified", "compound_class"))
        self.assertEqual(sum(sampling.counts.values()), 64)
        self.assertTrue(all(count >= 2 for count in sampling.counts.values()), sampling.counts)
        if not TRAIN_MANIFEST.is_file():
            self.skipTest("the training draw's manifest is not beside this checkout")
        sizes = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))["roots_by_class"]
        self.assertEqual(set(sampling.counts), set(sizes))
        # In proportion to the draw, within one root per class.
        for name, size in sizes.items():
            self.assertLessEqual(abs(sampling.counts[name] - 64 * size / 2550), 1.0, name)


if __name__ == "__main__":
    unittest.main()
