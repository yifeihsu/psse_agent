"""The baseline register: an entry from a run's evaluation, what may be compared, and no silent overwrite."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.rl.baseline_register import REGISTER_CONTRACT, add, compare, empty_register, entry_from_evaluation, load, main, markdown, save


def _evaluation(run: Path, name: str, *, solved: int, mode: str = "bounded_history_v1", limit: int | None = 49152) -> None:
    directory = run / name
    directory.mkdir(parents=True)
    episodes = []
    for index in range(4):
        success = index < solved
        episodes.append({"scenario_id": f"r{index}", "compound_class": "rx2_distant", "truth_audited_task_success": success,
                         "invalid_action_count": 1 if index == 0 else 0, "false_commit_count": 0, "loop_detected": False, "policy_steps": 10,
                         "reward": {"healthy_component_corrupted": index == 3},
                         "lifecycle": [{"target_is_true": index != 3, "declined": False}]})
    with (directory / "evaluation_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode) + "\n")
    summary = {"adapter": str(run / name / "adapter"), "iteration": 0 if name != "baseline" else None, "baseline": name == "baseline",
               "roots": 4, "truth_audited_task_success": solved, "solved_within_actions": {"16": solved, "20": solved}, "mean_steps": 10.0,
               "mean_reward": 0.2, "invalid_action_count": 1, "false_commit_count": 0, "loop_episodes": 0, "precision": "bf16",
               "observation_mode": mode, "max_steps": 40,
               "lifecycle": {"decisions": 4, "declines": 0, "true_target_declined": 0, "wrong_target_committed": 1, "wrong_target_declined": 0},
               **({"prompt_limits": {"max_input_tokens": limit}} if limit else {})}
    (directory / "evaluation.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")


class BaselineRegisterTests(unittest.TestCase):
    def test_an_entry_is_read_from_a_runs_evaluation_with_how_it_was_taken(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "arm_b"
            suite = Path(name) / "dev.json"
            suite.write_text("[]", encoding="utf-8")
            _evaluation(run, "iteration_000", solved=3, mode="complete_history_v1")
            config = {"development_suite": str(suite), "max_steps": 40, "load_in_4bit": False, "series": {"arm": "B"}}
            entry = entry_from_evaluation(run, "iteration_000", name="arm_b/S1", config=config, source_commit="abc")
            self.assertEqual((entry["results"]["solved"], entry["results"]["clean"], entry["results"]["healthy_component_corrupted"]), (3, 2, 1))
            self.assertEqual(entry["results"]["lifecycle"]["wrong_target_committed"], 1)
            how = entry["how"]
            self.assertEqual((how["precision"], how["observation_mode"], how["max_input_tokens"], how["max_steps"], how["budgets"]),
                             ("bf16", "complete_history_v1", 49152, 40, [16, 20]))
            self.assertEqual((how["source_commit"], len(how["suite_sha256"]), entry["series"]), ("abc", 64, {"arm": "B"}))
            self.assertEqual({len(value) for value in entry["files"].values()}, {64})
            with self.assertRaises(FileNotFoundError):
                entry_from_evaluation(run, "iteration_001", name="x")

    def test_nothing_is_overwritten_silently_and_unlike_evaluations_are_not_compared(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            _evaluation(directory / "a", "baseline", solved=2)
            _evaluation(directory / "a", "iteration_000", solved=3)
            _evaluation(directory / "b", "iteration_000", solved=3, mode="complete_history_v1")
            config = {"development_suite": None, "max_steps": 40}
            register = empty_register()
            first = entry_from_evaluation(directory / "a", "baseline", name="a/S0", config=config, suite_sha256="s" * 64)
            add(register, first)
            add(register, first)  # the same evaluation again changes nothing
            other = entry_from_evaluation(directory / "a", "iteration_000", name="a/S0", config=config, suite_sha256="s" * 64)
            with self.assertRaisesRegex(ValueError, "already has 'a/S0'"):
                add(register, other)
            add(register, {**other, "name": "a/S1"})
            add(register, entry_from_evaluation(directory / "b", "iteration_000", name="b/S1", config=config, suite_sha256="s" * 64))
            add(register, entry_from_evaluation(directory / "b", "iteration_000", name="elsewhere", config=config, suite_sha256="t" * 64))
            path = directory / "register.json"
            save(register, path)
            loaded = load(path)
            self.assertEqual((loaded["contract"], sorted(loaded["entries"])), (REGISTER_CONTRACT, ["a/S0", "a/S1", "b/S1", "elsewhere"]))
            entries = loaded["entries"]
            # A and B differ in the observation mode, which is their treatment; another suite forbids the comparison.
            matched = compare(entries["b/S1"], entries["a/S1"])
            self.assertEqual((matched["comparable"], matched["differs_as_treatment"]),
                             (True, {"observation_mode": ["complete_history_v1", "bounded_history_v1"]}))
            unlike = compare(entries["elsewhere"], entries["a/S1"])
            self.assertEqual((unlike["comparable"], sorted(unlike["not_comparable_because"])), (False, ["suite_sha256"]))
            self.assertEqual(main(["compare", "--register", str(path), "--first", "elsewhere", "--second", "a/S1"]), 1)
            self.assertEqual(main(["compare", "--register", str(path), "--first", "b/S1", "--second", "a/S1"]), 0)
            self.assertIn("| a/S1 | learner | 3 of 4 | 2 |", markdown(loaded))
            self.assertEqual(load(directory / "absent.json"), empty_register())
            (directory / "wrong.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a baseline register"):
                load(directory / "wrong.json")

    def test_the_command_appends_an_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            _evaluation(directory / "run", "baseline", solved=2)
            config = directory / "config.json"
            config.write_text(json.dumps({"development_suite": None, "max_steps": 40, "load_in_4bit": False}), encoding="utf-8")
            path = directory / "register.json"
            arguments = ["add", "--register", str(path), "--name", "run/S0", "--run", str(directory / "run"), "--evaluation", "baseline",
                         "--config", str(config), "--note", "the start adapter"]
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(arguments), 0)
            entry = load(path)["entries"]["run/S0"]
            self.assertEqual((entry["results"]["solved"], entry["policy"]["baseline"], entry["note"]), (2, True, "the start adapter"))


if __name__ == "__main__":
    unittest.main()
