from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.rl.rollout import StepRecord, Trajectory, TrajectoryStore
from psse_env.rl.train import Manifest, RLConfig, iteration_roots, load_suite, score_groups


def _config(directory: str, **overrides) -> RLConfig:
    payload = {
        "output_dir": directory, "training_suite": str(Path(directory) / "train.json"), "development_suite": None,
        "initial_adapter": str(Path(directory) / "adapter0"), "base_model": "google/gemma-4-12B-it",
        "base_revision": "0" * 40, "samples_per_root": 2,
    }
    payload.update(overrides)
    return RLConfig(**payload)


def _trajectory(root: str, sample: int, reward: float, *, success: bool, calls: int) -> Trajectory:
    step = StepRecord(step=0, model_observation={"a": 1}, prompt_tokens=1, original_prompt_tokens=1,
                      completion_token_ids=[1], text="x", action={"tool": "run_wls"}, tool="run_wls",
                      execution_status="success", error_code=None)
    return Trajectory(root=root, family="parameter", sample_index=sample, episode_seed=sample, steps=[step],
                      reward={"total": reward, "calls": calls}, evaluation={"truth_audited_task_success": success},
                      pairing_ok=True)


class TrainDriverTests(unittest.TestCase):
    def test_config_validation_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            config.validate()
            with self.assertRaises(ValueError):
                _config(directory, samples_per_root=1).validate()
            with self.assertRaises(ValueError):
                _config(directory, update={"learning_rate": -1.0}).validate()
            path = Path(directory) / "rl.json"
            path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
            self.assertEqual(RLConfig.load(path).samples_per_root, 2)
            path.write_text(json.dumps({**config.as_dict(), "bogus": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                RLConfig.load(path)

    def test_manifest_round_trip_defaults_to_the_initial_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = Manifest.load(path, initial_adapter="adapter0")
            self.assertEqual((manifest.iteration, manifest.phase, manifest.current_adapter), (0, "rollout", "adapter0"))
            manifest.iteration = 2
            manifest.phase = "update"
            manifest.current_adapter = "adapter1"
            manifest.save(path)
            reloaded = Manifest.load(path, initial_adapter="adapter0")
            self.assertEqual((reloaded.iteration, reloaded.phase, reloaded.current_adapter), (2, "update", "adapter1"))

    def test_iteration_roots_cycle_through_the_suite_deterministically(self) -> None:
        scenarios = [{"execution": {"scenario_id": f"r{i}"}} for i in range(5)]
        first = iteration_roots(scenarios, iteration=0, roots_per_iteration=2, seed=1)
        second = iteration_roots(scenarios, iteration=1, roots_per_iteration=2, seed=1)
        third = iteration_roots(scenarios, iteration=2, roots_per_iteration=2, seed=1)
        ids = lambda rows: [row["execution"]["scenario_id"] for row in rows]
        self.assertEqual(len(first), 2)
        self.assertEqual(ids(first), ids(iteration_roots(scenarios, iteration=0, roots_per_iteration=2, seed=1)))
        self.assertEqual(len(set(ids(first) + ids(second))), 4)
        self.assertEqual(len(ids(first) + ids(second) + ids(third)), 6)
        self.assertEqual(len(iteration_roots(scenarios, iteration=0, roots_per_iteration=None, seed=1)), 5)

    def test_load_suite_accepts_lists_and_wrapped_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(json.dumps([{"execution": {"scenario_id": "a"}}]), encoding="utf-8")
            self.assertEqual(len(load_suite(path)), 1)
            path.write_text(json.dumps({"scenarios": [{"execution": {"scenario_id": "a"}}, {"execution": {"scenario_id": "b"}}]}), encoding="utf-8")
            self.assertEqual(len(load_suite(path)), 2)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_suite(path)

    def test_score_groups_summarizes_the_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(directory)
            store.write_group("a", [_trajectory("a", 0, 1.0, success=True, calls=6), _trajectory("a", 1, 0.0, success=False, calls=9)])
            store.write_group("b", [_trajectory("b", 0, 1.0, success=True, calls=6), _trajectory("b", 1, 1.0, success=True, calls=6)])
            groups, summary = score_groups(_config(directory), store)
            self.assertEqual(summary["groups"], 2)
            self.assertEqual(summary["informative_groups"], 1)
            self.assertEqual(summary["episodes"], 4)
            self.assertAlmostEqual(summary["success_rate"], 0.75)
            self.assertAlmostEqual(summary["mean_calls"], 27 / 4)
            self.assertEqual(summary["unpaired_trajectories"], 0)
            informative = [advantages for _, advantages in groups if advantages.informative]
            self.assertEqual(len(informative), 1)
            self.assertEqual(informative[0].root, "a")


if __name__ == "__main__":
    unittest.main()
