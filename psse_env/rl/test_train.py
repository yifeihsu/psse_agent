from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

try:
    import torch

    from psse_env.rl.test_update import _Preempted, _TinyLM
except Exception:  # pragma: no cover - torch is optional for the pure-Python tests
    torch = None


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


@contextlib.contextmanager
def _tiny_update_phase(holder):
    """The update phase on the tiny bigram model of the update tests.

    Stand-ins replace the scoring, the model loader, the prompt functions, the
    row builder and the adapter writer for the length of the block: two groups
    of two samples, one row each.  ``holder['fail_after']`` cuts the model's
    forward passes the way a preemption cuts an update.
    """

    import psse_env.rl.train as train
    from psse_env.rl.advantages import GroupAdvantages
    from psse_env.rl.rows import build_step_row

    tokens = {("a", 0): ([1, 2], [3]), ("a", 1): ([2, 3], [4]), ("b", 0): ([3, 4], [5]), ("b", 1): ([4, 5], [6])}

    def score(config, store):
        groups = [([_trajectory(root, 0, 1.0, success=True, calls=5), _trajectory(root, 1, -1.0, success=False, calls=9)],
                   GroupAdvantages(root, (1.0, -1.0), (1.0, -1.0), 0.0, 1.0, True)) for root in ("a", "b")]
        return groups, {"lifecycle": {"labelled_decisions_paired": 0}}

    def load(config, adapter, *, reference_adapter=None):
        model = _TinyLM(seed=3, fail_after=holder.get("fail_after"))
        weights = Path(adapter) / "weights.pt"
        if weights.is_file():
            model.load_state_dict(torch.load(weights))
        return model, None, ()

    def prompt_functions(config, adapter, processor):
        return (lambda observation: ""), (lambda text: ([], 0)), SimpleNamespace(rendered=0, refused=0)

    def rows_of(groups, **kwargs):
        rows = []
        for trajectories, adv in groups:
            for trajectory, advantage in zip(trajectories, adv.advantages):
                prompt, completion = tokens[trajectory.root, trajectory.sample_index]
                rows.append(build_step_row(root=trajectory.root, sample_index=trajectory.sample_index, step=0,
                                           advantage=advantage, prompt_ids=prompt, completion_ids=completion, max_length=64))
        return rows, {}

    def save(model, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(directory) / "weights.pt")

    names = ("score_groups", "_load_trainable_model", "_prompt_functions", "build_rows", "_save_trained_adapter")
    originals = [getattr(train, name) for name in names]
    for name, stand_in in zip(names, (score, load, prompt_functions, rows_of, save)):
        setattr(train, name, stand_in)
    try:
        yield train
    finally:
        for name, original in zip(names, originals):
            setattr(train, name, original)


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

    def test_the_carried_optimizer_is_the_latest_one_an_earlier_iteration_left(self) -> None:
        from psse_env.rl.train import OPTIMIZER_FILE, _carried_optimizer_file, iteration_dir

        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            self.assertIsNone(_carried_optimizer_file(config, 0))
            self.assertIsNone(_carried_optimizer_file(config, 3))
            # Iterations 0 and 2 took optimizer steps; iteration 1 carried its adapter over and left no state.
            for iteration in (0, 1, 2):
                iteration_dir(config, iteration).mkdir(parents=True)
            for iteration in (0, 2):
                (iteration_dir(config, iteration) / OPTIMIZER_FILE).write_bytes(b"")
            self.assertIsNone(_carried_optimizer_file(config, 0))
            self.assertEqual(_carried_optimizer_file(config, 1), iteration_dir(config, 0) / OPTIMIZER_FILE)
            self.assertEqual(_carried_optimizer_file(config, 2), iteration_dir(config, 0) / OPTIMIZER_FILE)
            self.assertEqual(_carried_optimizer_file(config, 3), iteration_dir(config, 2) / OPTIMIZER_FILE)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_update_phase_carries_the_optimizer_from_one_iteration_to_the_next(self) -> None:
        update = {"chunk_by": "group", "optimizer_steps": 2, "learning_rate": 0.01}
        tables = {}
        for carry in (True, False):
            with tempfile.TemporaryDirectory() as directory, _tiny_update_phase({}) as train:
                # The switch is left out when it is off, as in every config written before it existed.
                config = _config(directory, update={**update, **({"carry_optimizer": True} if carry else {})},
                                 update_checkpoint_rows=1)
                first_dir, second_dir = train.iteration_dir(config, 0), train.iteration_dir(config, 1)
                first = train.phase_update(config, Manifest(phase="update", current_adapter=config.initial_adapter))
                second = train.phase_update(config, Manifest(iteration=1, phase="update", current_adapter=first["adapter_out"]))
                self.assertEqual((first["update"]["optimizer_steps"], second["update"]["optimizer_steps"]), (2, 2))
                self.assertEqual((first["update"]["split_groups"], second["update"]["split_groups"]), (0, 0))
                self.assertFalse((first_dir / "update_checkpoint").exists())
                if carry:
                    self.assertEqual(first["optimizer"], {"carried_from": None, "steps_before": 0, "steps_after": 2,
                                                          "continuity_since_iteration": 0})
                    self.assertEqual(second["optimizer"], {"carried_from": str(first_dir / train.OPTIMIZER_FILE),
                                                           "steps_before": 2, "steps_after": 4, "continuity_since_iteration": 0})
                    written = json.loads((second_dir / "update_report.json").read_text(encoding="utf-8"))
                    self.assertEqual(written["optimizer"], second["optimizer"])
                    self.assertTrue((second_dir / train.OPTIMIZER_FILE).is_file())
                else:
                    self.assertNotIn("optimizer", first)
                    self.assertNotIn("optimizer", second)
                    self.assertFalse((first_dir / train.OPTIMIZER_FILE).exists())
                    self.assertFalse((second_dir / train.OPTIMIZER_FILE).exists())
                tables[carry] = [torch.load(path / "adapter" / "weights.pt")["table"] for path in (first_dir, second_dir)]
        # The first update starts a new optimizer either way; the second continues it only when it is carried.
        self.assertTrue(torch.equal(tables[True][0], tables[False][0]))
        self.assertFalse(torch.allclose(tables[True][1], tables[False][1], atol=1e-4))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_a_preempted_update_phase_resumes_with_the_carried_optimizer(self) -> None:
        update = {"carry_optimizer": True, "chunk_by": "group", "optimizer_steps": 2, "learning_rate": 0.01}
        tables = {}
        for preempted in (False, True):
            holder = {"fail_after": None}
            with tempfile.TemporaryDirectory() as directory, _tiny_update_phase(holder) as train:
                config = _config(directory, update=update, update_checkpoint_rows=1)
                first_dir, out = train.iteration_dir(config, 0), train.iteration_dir(config, 1)
                first = train.phase_update(config, Manifest(phase="update", current_adapter=config.initial_adapter))
                manifest = Manifest(iteration=1, phase="update", current_adapter=first["adapter_out"])
                if preempted:
                    # Two rows per optimizer step: the first step lands, the second is cut after its first row.
                    holder["fail_after"] = 3
                    with self.assertRaises(_Preempted):
                        train.phase_update(config, manifest)
                    holder["fail_after"] = None
                    state = json.loads((out / "update_checkpoint" / "state.json").read_text(encoding="utf-8"))
                    self.assertEqual((state["completed_chunks"], state["chunk_rows_done"]), (1, 1))
                    self.assertFalse((out / train.OPTIMIZER_FILE).exists())
                report = train.phase_update(config, manifest)
                self.assertEqual(report["resumed_from_checkpoint"],
                                 str(out / "update_checkpoint" / "step_1") if preempted else None)
                self.assertEqual(report["update"]["resumed_from"],
                                 {"completed_chunks": 1, "chunk_rows_done": 1} if preempted else None)
                # The state before the update is the earlier iteration's either way; the resumed update
                # continues from its own checkpoint, one step further on.
                self.assertEqual(report["optimizer"], {"carried_from": str(first_dir / train.OPTIMIZER_FILE),
                                                       "steps_before": 2, "steps_after": 4, "continuity_since_iteration": 0})
                self.assertTrue((out / train.OPTIMIZER_FILE).is_file())
                self.assertFalse((out / "update_checkpoint").exists())
                tables[preempted] = torch.load(out / "adapter" / "weights.pt")["table"]
        self.assertTrue(torch.allclose(tables[True], tables[False], atol=1e-7))


if __name__ == "__main__":
    unittest.main()
