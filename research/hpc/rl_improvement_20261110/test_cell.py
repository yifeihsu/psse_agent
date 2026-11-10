"""The series cell: the configs are what series.json gives, the matched arms differ as it says, and its tools work."""
from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.observation_modes import BOUNDED_HISTORY, COMPLETE_HISTORY
from psse_env.rl.rollout import TrajectoryStore
from psse_env.rl.sampling import RootSampling
from psse_env.rl.train import CREDIT_ITERATION_PHASES, ITERATION_PHASES, PHASES, RLConfig, iteration_phases, load_manifest
from research.hpc.rl_improvement_20261110 import calibration, make_configs, provenance, seed_register
from research.rl.baseline_register import compare, load

CELL = Path(__file__).resolve().parent
REPO_ROOT = CELL.parents[2]
#: The training draw's manifest: in this checkout, or in the main checkout when this is one of its worktrees.
TRAIN_MANIFEST = next(
    (path for path in (root / "output" / "compound_rl_20261106" / "train_ieee14" / "manifest.json"
                       for root in (REPO_ROOT, REPO_ROOT.parents[2] if len(REPO_ROOT.parents) > 2 else REPO_ROOT))
     if path.is_file()),
    REPO_ROOT / "output" / "compound_rl_20261106" / "train_ieee14" / "manifest.json",
)
FIRST_ARM = CELL.parent / "compound_rl_20261107" / "rl_ieee14.json"
PLAN_OPTIONS = {"lifecycle_choice": 1, "detailed_evidence": 1, "pair_refit_adjacent": 1, "flow_at_noise_cue": 1, "route_comparison": 0}


def _config(name: str) -> dict:
    return json.loads((CELL / "configs" / name).read_text(encoding="utf-8"))


class SeriesConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.series = make_configs.load_series()
        self.configs = make_configs.build(self.series)

    def test_the_files_on_disk_are_what_the_series_gives(self) -> None:
        self.assertEqual(make_configs.check(self.configs), [])
        for name in self.configs:
            RLConfig.load(CELL / "configs" / name)
            self.assertNotIn(b"\r", (CELL / "configs" / name).read_bytes(), name)

    def test_matched_arms_differ_in_the_named_field_alone(self) -> None:
        self.assertEqual(make_configs.check_matched(self.series, self.configs), [])
        arm = {name: _config(self.series["arms"][name]["file"]) for name in make_configs.EXPERIMENT_ARMS}
        self.assertEqual(make_configs.differing_fields(arm["B"], arm["A"]), ["observation_mode"])
        self.assertEqual(make_configs.differing_fields(arm["C"], arm["A"]), ["decision_credit"])
        self.assertEqual(make_configs.differing_fields(arm["D"], arm["B"]), ["decision_credit"])
        self.assertEqual(make_configs.differing_fields(arm["D"], arm["C"]), ["observation_mode"])
        self.assertEqual([arm[name]["observation_mode"] for name in "ABCD"],
                         [BOUNDED_HISTORY, COMPLETE_HISTORY, BOUNDED_HISTORY, COMPLETE_HISTORY])
        self.assertEqual(len({arm[name]["output_dir"] for name in "ABCD"}), 4)
        # One starting adapter, one optimizer schedule, one root schedule and seed plan, one reward, one budget.
        for field in ("initial_adapter", "update", "root_sampling", "seed", "reward", "max_steps", "samples_per_root", "rows"):
            self.assertEqual(len({json.dumps(arm[name][field], sort_keys=True) for name in "ABCD"}), 1, field)
        # A mismatch is found: an arm that also changes the learning rate is not the series' pair.
        broken = copy.deepcopy(self.configs)
        broken[self.series["arms"]["B"]["file"]]["update"]["learning_rate"] = 2e-5
        self.assertEqual(len(make_configs.check_matched(self.series, broken)), 2)  # B against A, and D against B

    def test_the_arms_run_the_repaired_update_and_the_pilot_the_workplan_names(self) -> None:
        for name in make_configs.EXPERIMENT_ARMS:
            config = RLConfig.load(CELL / "configs" / self.series["arms"][name]["file"])
            update = config.update_settings(0)
            self.assertEqual((update.chunk_by, update.carry_optimizer, update.optimizer_steps, update.fresh_optimizer_steps), ("group", True, 2, 1))
            self.assertEqual((update.max_row_grad_norm, update.clip_ratio, update.clip_reference, update.beta_kl), (1.0, 0.2, "training", 0.04))
            self.assertEqual((update.step_diagnostics, update.diagnostic_components), (True, False))
            # The first step of an update clips against the row's own forward; whole groups are normalized so that
            # what a group's samples share cancels.
            self.assertEqual((update.first_step_baseline, update.loss_normalization), ("own", "dapo"))
            self.assertEqual((config.roots_per_iteration, config.samples_per_root, config.max_steps), (64, 4, 40))
            self.assertEqual(config.environment_options, PLAN_OPTIONS)
            self.assertEqual((config.evidence_profile, config.load_in_4bit, config.baseline_evaluation), ("scada_only", False, True))
            self.assertTrue(config.stop_on_integrity_failure)
            self.assertTrue(config.reserved_roots.endswith("/configs/reserved_roots.json"))
            self.assertEqual(config.series["arm"], name)
            self.assertEqual(iteration_phases(config), CREDIT_ITERATION_PHASES if name in "CD" else ITERATION_PHASES)
        credit = RLConfig.load(CELL / "configs" / self.series["arms"]["D"]["file"]).credit_settings()
        self.assertEqual((credit.enabled, credit.states_per_iteration, credit.continuations, credit.weight), (True, 32, 2, 1.0))
        pilot = self.series["protocol"]["pilot"]
        self.assertEqual(pilot["sampled_episodes_per_arm"], pilot["iterations"] * pilot["roots_per_iteration"] * pilot["samples_per_root"])
        self.assertEqual(pilot["continuation_branches_per_iteration_at_most"], credit.states_per_iteration * 2 * credit.continuations)
        probe = self.series["protocol"]["fork_probe"]
        self.assertEqual(probe["suffix_episodes"], (probe["training_roots"] + probe["validation_roots"]) * probe["choices"]
                         * probe["observation_modes"] * probe["continuations"])

    def test_the_prompt_limit_is_relaxed_for_every_arm_alike_and_fits_the_model(self) -> None:
        note = self.series["prompt_limit"]
        for name, config in self.configs.items():
            self.assertEqual((config["max_input_tokens"], config["max_length"]), (note["max_input_tokens"], note["max_input_tokens"]), name)
        self.assertGreater(note["max_input_tokens"], note["was"])
        self.assertLess(note["max_input_tokens"] + 256, note["model_context_window"])
        # The longest complete-history prompt seen so far fits with room for the completion.
        self.assertGreater(note["max_input_tokens"] - 256, 41500)

    def test_arm_a_is_the_first_arm_with_the_repaired_update(self) -> None:
        first = json.loads(FIRST_ARM.read_text(encoding="utf-8"))
        arm = _config("arm_a_bounded.json")
        new_fields = {"max_input_tokens", "observation_mode", "decision_credit", "reserved_roots", "stop_on_integrity_failure", "series",
                      "policy_source"}
        differing = {key for key in set(first) | set(arm) if first.get(key) != arm.get(key)} - new_fields
        self.assertEqual(differing, {"output_dir", "update", "max_length"})
        changed = {key for key in set(first["update"]) | set(arm["update"]) if first["update"].get(key) != arm["update"].get(key)}
        self.assertEqual(changed, {"chunk_by", "carry_optimizer", "fresh_optimizer_steps", "first_step_baseline", "loss_normalization",
                                   "step_diagnostics", "diagnostic_components"})
        # The same roots in the same order, sampled with the same seeds: arm A's first batch is the first arm's.
        self.assertEqual((arm["seed"], arm["root_sampling"]), (first["seed"], first["root_sampling"]))

    def test_the_stratified_mix_and_the_reservation(self) -> None:
        config = RLConfig.load(CELL / "configs" / "arm_a_bounded.json")
        sampling = RootSampling.from_mapping(config.root_sampling)
        self.assertEqual((sampling.mode, sampling.key, sum(sampling.counts.values())), ("stratified", "compound_class", 64))
        reserved = json.loads((CELL / "reserved_roots.json").read_text(encoding="utf-8"))
        pool = json.loads((CELL / "probe_training_roots.json").read_text(encoding="utf-8"))
        self.assertEqual((len(reserved["roots"]), len(set(reserved["roots"])), reserved["iterations_free_of_reserved_roots"]), (64, 64, 30))
        self.assertEqual(reserved["counts"], dict(sorted(sampling.counts.items())))
        self.assertEqual((reserved["seed"], reserved["suite_sha256"]), (config.seed, self.series["inputs"]["training_suite"]["sha256"]))
        self.assertEqual((len(pool["roots"]), len(set(pool["roots"]) & set(reserved["roots"])), pool["iterations"]), (128, 0, [28, 29]))
        self.assertEqual(config.series["planned_inputs"]["reserved_roots_sha256"], make_configs.file_sha256(CELL / "reserved_roots.json"))
        smoke = RootSampling.from_mapping(_config("smoke_complete.json")["root_sampling"])
        self.assertEqual((sum(smoke.counts.values()), set(smoke.counts)), (8, set(sampling.counts)))
        self.assertEqual({name for name, count in smoke.counts.items() if count}, {"rx_cb_distant", "rx_cb_meter", "rx_cb_same_line", "rx_cb_shared_bus"})
        if not TRAIN_MANIFEST.is_file():
            self.skipTest("the training draw's manifest is not beside this checkout")
        sizes = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))["roots_by_class"]
        self.assertEqual(set(sampling.counts), set(sizes))

    def test_the_calibration_runs_differ_in_the_schedule_alone(self) -> None:
        runs = {name: _config(self.series["arms"][name]["file"]) for name in ("calibration_1", "calibration_2", "calibration_4")}
        schedules = {name: (config["update"]["optimizer_steps"], config["update"]["fresh_optimizer_steps"]) for name, config in runs.items()}
        self.assertEqual(schedules, {"calibration_1": (1, 1), "calibration_2": (2, None), "calibration_4": (4, None)})
        for name, config in runs.items():
            self.assertEqual(make_configs.differing_fields(config, runs["calibration_1"]), [] if name == "calibration_1" else ["update"], name)
            self.assertEqual((config["development_suite"], config["baseline_evaluation"], config["update"]["diagnostic_components"]),
                             (None, False, True))
        # Against arm A a calibration run changes the schedule, drops the evaluation and reads no reserved roots.
        self.assertEqual(make_configs.differing_fields(runs["calibration_1"], _config("arm_a_bounded.json")),
                         ["baseline_evaluation", "development_suite", "evaluate_every_iteration", "reserved_roots", "update"])

    def test_the_probes_expert_is_the_rule_expert_with_its_route_comparison(self) -> None:
        config = RLConfig.load(CELL / "configs" / "probe_expert.json")
        self.assertEqual((config.policy_source, config.expert_variant), ("expert", "baseline_compare_guarded_routes"))
        self.assertEqual(config.expert_options, {"abstain_isolated_lines": 1, "state_ambiguity": 1})
        self.assertEqual(config.environment_options, {**PLAN_OPTIONS, "route_comparison": 1})


class CellToolTests(unittest.TestCase):
    def _run_config(self, directory: Path, **changes) -> Path:
        config = _config("calibration_1step.json")
        suite = directory / "train.json"
        suite.write_text(json.dumps([{"execution": {"scenario_id": "r1"}}]), encoding="utf-8")
        adapter = directory / "adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        config.update(output_dir=str(directory / "run"), training_suite=str(suite), initial_adapter=str(adapter), **changes)
        path = directory / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _batch(self, directory: Path) -> Path:
        batch = directory / "batch"
        groups = batch / "groups"
        groups.mkdir(parents=True)
        (groups / "r1.jsonl").write_text('{"root": "r1"}\n{"root": "r1"}\n', encoding="utf-8")
        (groups / "r2.jsonl").write_text('{"root": "r2"}\n', encoding="utf-8")
        return batch

    def test_the_job_scripts_run_the_drivers_phases_and_name_the_cells_files(self) -> None:
        script = (CELL / "rl.sbatch").read_text(encoding="utf-8")
        self.assertEqual(re.search(r"for phase in ([a-z ]+); do", script).group(1).split(), list(PHASES))
        self.assertIn('"$phase" != update', script)  # the update alone is one process
        probe = (CELL / "fork_probe.sbatch").read_text(encoding="utf-8")
        for name in ("reserved_roots.json", "probe_training_roots.json", "arm_a_bounded.json", "probe_expert.json"):
            self.assertIn(name, probe)
            self.assertTrue((CELL / name).is_file() or (CELL / "configs" / name).is_file(), name)
        for name in ("rl.sbatch", "fork_probe.sbatch", "submit_arm.sh", "deploy_from_windows.sh"):
            self.assertNotIn(b"\r", (CELL / name).read_bytes(), name)

    def test_a_calibration_run_starts_at_the_update_of_the_fixed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            path, batch = self._run_config(directory), self._batch(directory)
            record = calibration.prepare(path, batch)
            config = RLConfig.load(path)
            manifest = load_manifest(config)
            self.assertEqual((manifest.iteration, manifest.phase, manifest.current_adapter), (0, "update", config.initial_adapter))
            self.assertEqual((record["batch"]["groups"], record["batch"]["episodes"]), (2, 3))
            self.assertEqual(calibration.batch_listing(directory / "run" / "iteration_000" / "trajectories"),
                             {key: record["batch"][key] for key in ("groups", "episodes", "sha256")})
            self.assertEqual(TrajectoryStore(directory / "run" / "iteration_000" / "trajectories").roots(), ["r1", "r2"])
            with self.assertRaisesRegex(SystemExit, "already holds a run"):
                calibration.prepare(path, batch)
            (directory / "other").mkdir()
            with self.assertRaisesRegex(SystemExit, "holds no rollout groups"):
                calibration.prepare(self._run_config(directory / "other"), directory / "nothing")

    def test_the_calibration_report_compares_runs_on_one_batch(self) -> None:
        def run(directory: Path, steps: int, digest: str, movement: float) -> Path:
            target = directory / "iteration_000"
            target.mkdir(parents=True)
            step = {"rows": 10, "rows_clipped_fraction": 0.1, "gradient_norm_before_global_clip": 2.0, "global_clip_scale": 0.5,
                    "top_rows_squared_norm_share": 0.4, "components": {"policy": {"rows": 8, "share_of_step": 0.7},
                                                                      "kl": {"rows": 9, "share_of_step": 0.3}},
                    "movement": {"rms_over_learning_rate": movement, "relative_to_weight_norm": 1e-3, "cosine_with_descent": 0.6},
                    "decisions": {"true_target_first_candidate": {"decisions": 3, "mean_p_decline": 0.03},
                                  "true_target_later_candidate": {"decisions": 1, "mean_p_decline": 0.05},
                                  "wrong_target": {"decisions": 2, "mean_p_decline": 0.2}, "auc_wrong_above_true": 0.9}}
            report = {"update": {"optimizer_steps": steps, "planned_steps": steps, "fresh_optimizer": True, "rows": 10, "batch_digest": digest,
                                 "row_accounting": {"each_row_once": True, "rows_by_kind": {"policy": 8}}, "split_groups": 0,
                                 "on_policy_check": [{"mean_abs_log_ratio": 0.004}], "policy_loss": [0.1] * steps, "kl": [0.0] * steps,
                                 "grad_norm": [2.0] * steps, "settings": {"optimizer_steps": steps, "chunk_by": "group"},
                                 "diagnostics": [step] * steps},
                      "integrity": {"ok": True, "failures": []}}
            (target / "update_report.json").write_text(json.dumps(report), encoding="utf-8")
            return directory

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            one, two = run(directory / "one", 1, "d1", 1.0), run(directory / "two", 2, "d1", 0.7)
            report = calibration.build_report([one, two])
            self.assertTrue(report["same_batch"])
            self.assertEqual([len(item["steps"]) for item in report["runs"]], [1, 2])
            text = calibration.markdown(report)
            self.assertIn("| two | 2 | 10 | 0.100 | 0.700 | 0.600 | 0.700 | 0.300 | n/a | 0.0300 | 0.0500 | 0.2000 | 0.900 |", text)
            self.assertNotIn("do not compare", text)
            other = run(directory / "other", 1, "d2", 1.0)
            self.assertFalse(calibration.build_report([one, other])["same_batch"])
            self.assertFalse(calibration.build_report([one, directory / "unfinished"])["same_batch"])
            self.assertIn("not finished", calibration.markdown(calibration.build_report([one, directory / "unfinished"])))

    def test_the_provenance_record_reports_an_input_that_is_not_the_planned_one(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            path = self._run_config(directory)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["series"]["planned_inputs"]["training_suite_sha256"] = make_configs.file_sha256(Path(config["training_suite"]))
            config["series"].pop("expert_reference")
            path.write_text(json.dumps(config), encoding="utf-8")
            entry = provenance.record(path, environment={"SLURM_JOB_ID": "7"})
            self.assertEqual((entry["job"], entry["inputs_not_as_planned"], entry["inputs_missing"]), ("7", [], []))
            self.assertTrue(entry["inputs"]["training_suite"]["as_planned"])
            self.assertEqual(entry["config"]["sha256"], make_configs.file_sha256(path))
            self.assertEqual(entry["prompt_limit"]["max_input_tokens"], 49152)
            self.assertIsNotNone(entry["inputs"]["initial_adapter"]["files"]["adapter_config.json"])
            self.assertIsNone(entry["inputs"]["initial_adapter"]["files"]["adapter_model.safetensors"])
            Path(config["training_suite"]).write_text("[]", encoding="utf-8")
            changed = provenance.record(path, environment={})
            self.assertEqual(changed["inputs_not_as_planned"], ["training_suite"])
            self.assertEqual(provenance.main(["--config", str(path), "--no-write"]), 0)      # reported, not refused
            self.assertEqual(provenance.main(["--config", str(path), "--no-write", "--strict"]), 3)
            written = provenance.append(changed, directory / "run")
            self.assertEqual(len(written.read_text(encoding="utf-8").splitlines()), 1)

    def test_the_register_is_seeded_from_the_read_of_the_first_arm(self) -> None:
        read = json.loads((CELL / "first_arm_read.json").read_text(encoding="utf-8"))
        register = load(CELL / "baseline_register.json")
        self.assertEqual(seed_register.build(read)["entries"], register["entries"])
        solved = {name: entry["results"]["solved"] for name, entry in register["entries"].items()}
        self.assertEqual(solved, {"first_arm/S0": 182, "first_arm/S1": 180, "first_arm/S2": 181, "first_arm/S3": 182,
                                  "rule_expert/route_teacher": 188, "audit_deciding": 200})
        s0, s3 = register["entries"]["first_arm/S0"], register["entries"]["first_arm/S3"]
        self.assertEqual((s0["how"]["precision"], s0["how"]["observation_mode"], s0["how"]["max_input_tokens"]), ("bf16", BOUNDED_HISTORY, 32768))
        self.assertEqual(s0["how"]["suite_sha256"], make_configs.load_series()["inputs"]["development_suite"]["sha256"])
        self.assertEqual((s3["results"]["clean"], s3["results"]["lifecycle"]["wrong_target_committed"]), (173, 43))
        self.assertEqual(compare(s3, s0)["comparable"], True)


if __name__ == "__main__":
    unittest.main()
