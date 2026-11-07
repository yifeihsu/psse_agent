from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.rl.rollout import StepRecord, Trajectory, TrajectoryStore
from psse_env.rl.train import (
    Manifest,
    RLConfig,
    aggregate_evaluation,
    configure_research_environment,
    environment_record,
    initial_phase,
    iteration_roots,
    load_manifest,
    load_suite,
    phase_evaluate,
    run_phase,
    score_groups,
    status,
)


def _config(directory: str, **overrides) -> RLConfig:
    payload = {
        "output_dir": directory, "training_suite": str(Path(directory) / "train.json"), "development_suite": None,
        "initial_adapter": str(Path(directory) / "adapter0"), "base_model": "google/gemma-4-12B-it",
        "base_revision": "0" * 40, "samples_per_root": 2,
    }
    payload.update(overrides)
    return RLConfig(**payload)


def _trajectory(root: str, sample: int, reward: float, *, success: bool, calls: int, lifecycle=None, pairing_ok: bool = True) -> Trajectory:
    step = StepRecord(step=0, model_observation={"a": 1}, prompt_tokens=1, original_prompt_tokens=1,
                      completion_token_ids=[1], text="x", action={"tool": "run_wls"}, tool="run_wls",
                      execution_status="success", error_code=None)
    return Trajectory(root=root, family="parameter", sample_index=sample, episode_seed=sample, steps=[step],
                      reward={"total": reward, "calls": calls}, evaluation={"truth_audited_task_success": success},
                      pairing_ok=pairing_ok, lifecycle=list(lifecycle or []))


class TrainDriverTests(unittest.TestCase):
    def test_config_validation_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            config.validate()
            with self.assertRaises(ValueError):
                _config(directory, samples_per_root=1).validate()
            with self.assertRaises(ValueError):
                _config(directory, update={"learning_rate": -1.0}).validate()
            for truncated in (dict(top_p=0.95), dict(top_k=64)):
                with self.assertRaisesRegex(ValueError, "biases the policy gradient"):
                    _config(directory, **truncated).validate()
            _config(directory, top_k=0, temperature=1.3).validate()
            with self.assertRaisesRegex(ValueError, "sampling temperature"):
                _config(directory, update={"temperature": 2.0}).validate()
            settings = _config(directory, temperature=1.3, update={"beta_kl": 0.1}).update_settings(4)
            self.assertEqual((settings.temperature, settings.beta_kl, settings.seed), (1.3, 0.1, 20261014 + 4))
            path = Path(directory) / "rl.json"
            path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
            self.assertEqual(RLConfig.load(path).samples_per_root, 2)
            path.write_text(json.dumps({**config.as_dict(), "bogus": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                RLConfig.load(path)

    def test_environment_and_expert_options_and_budgets_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _config(directory, environment_options={"lifecycle_choice": 1, "detailed_evidence": 1, "route_comparison": 0}).validate()
            with self.assertRaisesRegex(ValueError, "unknown environment_options"):
                _config(directory, environment_options={"bogus": 1}).validate()
            with self.assertRaisesRegex(ValueError, "config fields of their own"):
                _config(directory, environment_options={"evidence_profile": "scada_only"}).validate()
            with self.assertRaisesRegex(ValueError, "expert_variant"):
                _config(directory, expert_options={"variant": "baseline"}).validate()
            with self.assertRaisesRegex(ValueError, "lifecycle_choice must open"):
                _config(directory, reward={"lifecycle_credit": 0.25}).validate()
            _config(directory, reward={"lifecycle_credit": 0.25}, environment_options={"lifecycle_choice": 1}).validate()
            _config(directory, reward={"lifecycle_credit": 0.25, "lifecycle_form": "supervised"}, environment_options={"lifecycle_choice": 1}).validate()
            with self.assertRaisesRegex(ValueError, "lifecycle_form"):
                _config(directory, reward={"lifecycle_form": "bogus"}).validate()
            for budgets in ([16, 50], [0], [True], ["16"]):
                with self.assertRaisesRegex(ValueError, "evaluation_budgets"):
                    _config(directory, evaluation_budgets=budgets).validate()
            _config(directory, evaluation_budgets=[]).validate()
            config = _config(directory, environment_options={"lifecycle_choice": 1}, expert_options={"state_ambiguity": 1})
            path = Path(directory) / "rl.json"
            path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
            loaded = RLConfig.load(path)
            self.assertEqual((loaded.environment_options, loaded.expert_options, loaded.evaluation_budgets),
                             ({"lifecycle_choice": 1}, {"state_ambiguity": 1}, [16, 20]))
            record = environment_record(loaded)
            self.assertEqual((record["environment_options"], record["expert_variant"], record["max_steps"]), ({"lifecycle_choice": 1}, "baseline", 40))

    def test_configure_research_environment_applies_and_clears_the_named_options(self) -> None:
        import scripts.run_dagger_research as research

        snapshot = [(options, dict(options)) for options in (research.RESEARCH_ENVIRONMENT_OPTIONS, research.RESEARCH_EXPERT_OPTIONS)]
        try:
            with tempfile.TemporaryDirectory() as directory:
                research.RESEARCH_ENVIRONMENT_OPTIONS["route_comparison"] = True  # left behind by an earlier cell
                research.RESEARCH_EXPERT_OPTIONS["abstain_isolated_lines"] = True
                config = _config(directory, evidence_profile="scada_only", expert_variant="baseline_compare",
                                 environment_options={"lifecycle_choice": 1, "detailed_evidence": 1, "flow_at_noise_cue": 1},
                                 expert_options={"state_ambiguity": 1})
                factory = configure_research_environment(config)
                self.assertTrue(callable(factory))
                options = research.RESEARCH_ENVIRONMENT_OPTIONS
                self.assertEqual((options["lifecycle_choice"], options["detailed_evidence"], options["flow_at_noise_cue"]), (1, 1, 1))
                self.assertNotIn("route_comparison", options)
                self.assertEqual(options["evidence_profile"], "scada_only")
                self.assertEqual(research.RESEARCH_EXPERT_OPTIONS, {"variant": "baseline_compare", "state_ambiguity": 1})
                configure_research_environment(_config(directory))
                self.assertNotIn("lifecycle_choice", options)
                self.assertEqual(research.RESEARCH_EXPERT_OPTIONS, {"variant": "baseline"})
        finally:
            for options, original in snapshot:
                options.clear()
                options.update(original)

    def test_row_settings_are_validated_and_a_clipped_update_requires_sampling_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, rows={"preservation": "one", "preservation_max_groups": 8})
            config.validate()
            settings = config.row_settings(3)
            self.assertEqual((settings.preservation, settings.seed, settings.require_sampling_records), ("one", 20261014 + 3, False))
            with self.assertRaisesRegex(ValueError, "preservation"):
                _config(directory, rows={"preservation": "some"}).validate()
            with self.assertRaisesRegex(ValueError, "unknown rows fields"):
                _config(directory, rows={"bogus": True}).validate()
            clipped = _config(directory, update={"clip_ratio": 0.2})
            self.assertTrue(clipped.row_settings(0).require_sampling_records)

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
            self.assertEqual(summary["lifecycle"]["decisions"], 0)

    def test_score_groups_counts_the_labelled_decisions_of_paired_trajectories(self) -> None:
        decline = {"position": 0, "tool": "rollback_state", "declined": True, "agrees": False, "target_is_true": True}
        commit = {"position": 0, "tool": "commit_state", "declined": False, "agrees": True, "target_is_true": True}
        unlabelled = {"position": 0, "tool": "commit_state", "declined": False, "agrees": None, "target_is_true": None}
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(directory)
            store.write_group("a", [_trajectory("a", 0, 1.0, success=True, calls=6, lifecycle=[commit]),
                                    _trajectory("a", 1, 1.0, success=True, calls=6, lifecycle=[decline, unlabelled], pairing_ok=False)])
            _groups, summary = score_groups(_config(directory), store)
            lifecycle = summary["lifecycle"]
            self.assertEqual((lifecycle["decisions"], lifecycle["declines"], lifecycle["agreeing"], lifecycle["disagreeing"],
                              lifecycle["unlabelled"], lifecycle["true_target_declined"], lifecycle["labelled_decisions_paired"]),
                             (3, 1, 1, 1, 1, 1, 1))

    def test_aggregate_evaluation_reports_budgets_and_lifecycle_decisions(self) -> None:
        records = [
            {"family": "parameter", "truth_audited_task_success": True, "steps": 12, "policy_steps": 12, "reward": {"total": 0.7},
             "lifecycle": [{"declined": False, "agrees": True, "target_is_true": True}]},
            {"family": "parameter", "truth_audited_task_success": True, "steps": 18, "policy_steps": 18, "reward": {"total": 0.6},
             "lifecycle": [{"declined": True, "agrees": True, "target_is_true": False}, {"declined": False, "agrees": True, "target_is_true": True}]},
            {"family": "topology", "truth_audited_task_success": False, "steps": 25, "policy_steps": 25, "reward": {"total": -0.5},
             "lifecycle": [{"declined": True, "agrees": False, "target_is_true": True}]},
        ]
        summary = aggregate_evaluation(records, [16, 20])
        self.assertEqual(summary["truth_audited_task_success"], 2)
        self.assertEqual(summary["solved_within_actions"], {"16": 1, "20": 2})
        self.assertAlmostEqual(summary["mean_steps_solved"], 15.0)
        lifecycle = summary["lifecycle"]
        self.assertEqual((lifecycle["decisions"], lifecycle["declines"], lifecycle["agreeing"], lifecycle["disagreeing"],
                          lifecycle["true_target_declined"], lifecycle["wrong_target_declined"], lifecycle["episodes_with_decline"]),
                         (4, 2, 3, 1, 1, 1, 2))
        self.assertEqual(aggregate_evaluation(records)["solved_within_actions"], {})
        self.assertEqual(aggregate_evaluation([])["mean_steps_solved"], None)


    def test_evaluation_is_skipped_when_the_update_left_the_adapter_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, development_suite=str(Path(directory) / "dev.json"))
            out = Path(directory) / "iteration_000"
            out.mkdir()
            (out / "update_report.json").write_text(json.dumps({"update": None}), encoding="utf-8")
            summary = phase_evaluate(config, Manifest(current_adapter="adapter0"))
            self.assertIn("skipped", summary)
            self.assertIn("skipped", json.loads((out / "evaluation.json").read_text(encoding="utf-8"))["summary"])
            self.assertIsNone(phase_evaluate(_config(directory), Manifest(current_adapter="adapter0")))

    def test_baseline_evaluation_measures_the_starting_adapter_before_the_first_rollout(self) -> None:
        import psse_env.dagger.evaluator as evaluator_module
        import psse_env.dagger.research_policy_factory as policy_module
        import scripts.run_dagger_research as research

        trace = [{"step": 0, "action": {"tool": "run_wls", "arguments": {"state_id": "active"}}, "execution_status": "success",
                  "error_code": None}]

        class _Result:
            def __init__(self, scenario) -> None:
                self.scenario = scenario

            def as_dict(self):
                return {"suite_metrics": {"episodes": [{
                    "scenario_id": self.scenario["execution"]["scenario_id"], "family": self.scenario["grouping"]["scenario_family"],
                    "truth_audited_task_success": True, "audit": {"truth_audited_task_assessment": {"basis": "counterfactual_resolution"}},
                    "trace": trace, "steps": 9, "policy_steps": 9}]}}

        def fake_evaluate(suites, *, env_factory, policy_factory, max_steps, seed, case_loader, **contract):
            return _Result(suites["standard_success"][0])

        loaded = []

        def fake_policy_factory(adapter, **kwargs):
            loaded.append((adapter, kwargs.get("load_in_4bit")))
            return object()

        originals = (evaluator_module.evaluate_rollout_suites, policy_module.research_gemma_policy_factory,
                     policy_module.clear_research_policy_cache)
        snapshot = [(options, dict(options)) for options in (research.RESEARCH_ENVIRONMENT_OPTIONS, research.RESEARCH_EXPERT_OPTIONS)]
        evaluator_module.evaluate_rollout_suites = fake_evaluate
        policy_module.research_gemma_policy_factory = fake_policy_factory
        policy_module.clear_research_policy_cache = lambda: None
        try:
            with tempfile.TemporaryDirectory() as directory:
                dev = Path(directory) / "dev.json"
                dev.write_text(json.dumps([{"execution": {"scenario_id": "d1"}, "grouping": {"scenario_family": "parameter"}},
                                           {"execution": {"scenario_id": "d2"}, "grouping": {"scenario_family": "topology"}}]), encoding="utf-8")
                self.assertEqual(initial_phase(_config(directory, development_suite=str(dev))), "rollout")
                with self.assertRaisesRegex(ValueError, "development suite"):
                    _config(directory, baseline_evaluation=True).validate()
                config = _config(directory, development_suite=str(dev), baseline_evaluation=True, load_in_4bit=False,
                                 evaluation_budgets=[8, 16])
                config.validate()
                self.assertEqual(initial_phase(config), "baseline")
                self.assertEqual((status(config)["phase"], status(config)["baseline"]), ("baseline", None))
                with self.assertRaisesRegex(RuntimeError, "at phase 'baseline'"):
                    run_phase(config, "rollout")
                shard = run_phase(config, "baseline", shard=(0, 2))
                self.assertEqual((shard["evaluated"], shard["roots"], shard["shard"]), (1, 1, "0/2"))
                self.assertEqual(status(config)["phase"], "baseline")  # a shard closes nothing
                summary = run_phase(config, "baseline")
                self.assertEqual((summary["roots"], summary["truth_audited_task_success"], summary["adapter"], summary["baseline"]),
                                 (2, 2, config.initial_adapter, True))
                self.assertEqual(summary["solved_within_actions"], {"8": 0, "16": 2})
                self.assertNotIn("iteration", summary)
                self.assertEqual(loaded, [(config.initial_adapter, False)] * 2)  # the driver's own loader, once per process
                self.assertTrue((Path(directory) / "baseline" / "evaluation.json").is_file())
                lines = (Path(directory) / "baseline" / "evaluation_episodes.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(sorted(json.loads(line)["scenario_id"] for line in lines), ["d1", "d2"])
                manifest = load_manifest(config)
                self.assertEqual((manifest.phase, manifest.iteration, manifest.baseline["adapter"], manifest.current_adapter),
                                 ("rollout", 0, config.initial_adapter, config.initial_adapter))
                self.assertEqual(manifest.baseline["evaluation"]["truth_audited_task_success"], 2)
                with self.assertRaisesRegex(RuntimeError, "at phase 'rollout'"):
                    run_phase(config, "baseline")
                # A run started without the flag is at its first rollout, as before.
                self.assertEqual(Manifest.load(Path(directory) / "none.json", initial_adapter="a").phase, "rollout")
                with self.assertRaises(ValueError):
                    Manifest.load(Path(directory) / "none.json", initial_adapter="a", initial_phase="bogus")
        finally:
            (evaluator_module.evaluate_rollout_suites, policy_module.research_gemma_policy_factory,
             policy_module.clear_research_policy_cache) = originals
            for options, original in snapshot:
                options.clear()
                options.update(original)

    def test_aggregate_evaluation_counts_success_cost_and_safety(self) -> None:
        records = [
            {"family": "parameter", "truth_audited_task_success": True, "steps": 6, "reward": {"total": 0.88},
             "false_commit_count": 0, "loop_detected": False, "invalid_action_count": 0},
            {"family": "parameter", "truth_audited_task_success": False, "steps": 30, "reward": {"total": -0.6},
             "false_commit_count": 2, "loop_detected": True, "invalid_action_count": 3, "false_finalization_count": 1},
            {"family": "hif", "truth_audited_task_success": True, "steps": 5, "reward": {"total": 0.9}},
        ]
        summary = aggregate_evaluation(records)
        self.assertEqual(summary["roots"], 3)
        self.assertEqual(summary["truth_audited_task_success"], 2)
        self.assertAlmostEqual(summary["mean_steps"], 41 / 3)
        self.assertAlmostEqual(summary["mean_reward"], (0.88 - 0.6 + 0.9) / 3)
        self.assertEqual((summary["false_commit_count"], summary["invalid_action_count"], summary["loop_episodes"]), (2, 3, 1))
        self.assertEqual(summary["false_finalization_count"], 1)
        self.assertEqual(summary["by_family"], {"hif": {"roots": 1, "successes": 1}, "parameter": {"roots": 2, "successes": 1}})


if __name__ == "__main__":
    unittest.main()
