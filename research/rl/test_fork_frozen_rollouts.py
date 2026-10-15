import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from psse_env.rl.rollout import TrajectoryStore, sample_seed
from psse_env.rl.train import RLConfig, iteration_dir, source_revision
from research.rl.fork_frozen_rollouts import fork
from research.rl.test_signal_report import trajectory


class FrozenForkTests(unittest.TestCase):
    def setup_probe(self, directory):
        directory = Path(directory)
        suite = directory / "training.json"
        suite.write_text(json.dumps([{"execution": {"scenario_id": "r"}}]))
        config = RLConfig(output_dir=str(directory / "probe"), training_suite=str(suite), development_suite=None,
                          initial_adapter=str(directory / "adapter"), base_model="test", base_revision="frozen",
                          samples_per_root=2, reward={"tool_regret_cost": 0})
        store = TrajectoryStore(iteration_dir(config, 0) / "trajectories")
        group = [trajectory(0), trajectory(1)]
        for t in group:
            t.episode_seed = sample_seed(config.seed, "r", t.sample_index)
            t.policy_identity = {"adapter_path": config.initial_adapter}
        store.write_group("r", group)
        path = iteration_dir(config, 0)
        (path / "rollout_config.json").write_text(json.dumps({"config": config.as_dict(), "source_revision": source_revision()}))
        (path / "rollout_summary.json").write_text(json.dumps({"adapter": config.initial_adapter}))
        return config, store

    def test_reward_is_recomputed_without_changing_samples_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, store = self.setup_probe(directory)
            target = replace(source, output_dir=str(Path(directory) / "arm"), reward={"call_cost": 0, "tool_regret_cost": 0})
            receipt = fork(source, target)
            result = TrajectoryStore(iteration_dir(target, 0) / "trajectories").read_group("r")
            self.assertEqual(result[0].total_reward, 1.0)
            self.assertEqual(store.read_group("r")[0].total_reward, 0.8)
            self.assertEqual(result[0].steps[0].completion_token_ids, store.read_group("r")[0].steps[0].completion_token_ids)
            self.assertTrue(receipt["reward_recomputed"])
            self.assertEqual(json.loads((Path(target.output_dir) / "manifest.json").read_text())["phase"], "update")
            with self.assertRaisesRegex(ValueError, "already exists"):
                fork(source, target)

    def test_changed_temperature_or_updated_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.setup_probe(directory)
            target = replace(source, output_dir=str(Path(directory) / "arm"), temperature=1.3)
            with self.assertRaisesRegex(ValueError, "differs"):
                fork(source, target)
            (iteration_dir(source, 0) / "rollout_summary.json").write_text(json.dumps({"adapter": "updated_adapter"}))
            with self.assertRaisesRegex(ValueError, "frozen initial"):
                fork(source, replace(target, temperature=source.temperature))


if __name__ == "__main__":
    unittest.main()
