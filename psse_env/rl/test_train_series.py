"""The driver's additions for the experiment series of 2026-11-10, on the tiny bigram model and synthetic rollouts.

The update phase here runs through the real row builder (unlike the
stand-in of ``test_train``): the rollout store holds an adversarial root
whose samples contrast success with failure and include an invalid sampled
output, a root whose samples all scored the same, and labelled lifecycle
decisions, and the update is cut by a preemption and resumed.
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from psse_env.dagger.observation_modes import BOUNDED_HISTORY, COMPLETE_HISTORY
from psse_env.rl.integrity import MARKER_FILE, IntegrityError
from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.rollout import StepRecord, Trajectory, TrajectoryStore
from psse_env.rl.train import (
    CREDIT_ITERATION_PHASES,
    ITERATION_PHASES,
    Manifest,
    RLConfig,
    apply_prompt_limits,
    environment_record,
    iteration_dir,
    iteration_phases,
    load_manifest,
    model_context_window,
    reserved_roots,
    run_phase,
)

try:
    import torch

    from psse_env.rl.test_update import _Preempted, _TinyLM
except Exception:  # pragma: no cover - torch is optional for the pure-Python tests
    torch = None

COMMIT, ROLLBACK = "commit_state", "rollback_state"
UPDATE = {"chunk_by": "group", "carry_optimizer": True, "optimizer_steps": 2, "fresh_optimizer_steps": 1, "learning_rate": 0.01,
          "clip_ratio": 0.2, "clip_reference": "training", "max_row_grad_norm": 1.0, "loss_normalization": "dapo",
          "step_diagnostics": True}
ROWS = {"require_complete_groups": True, "require_prompt_digest": True, "reject_truncated_prompts": True, "preservation": "one"}


def _config(directory: str, **overrides) -> RLConfig:
    payload = {
        "output_dir": directory, "training_suite": str(Path(directory) / "train.json"), "development_suite": None,
        "initial_adapter": str(Path(directory) / "adapter0"), "base_model": "google/gemma-4-12B-it", "base_revision": "0" * 40,
        "samples_per_root": 4, "max_length": 64, "advantage_scale": "none",
        "environment_options": {"lifecycle_choice": 1}, "reward": {"lifecycle_credit": 0.25, "lifecycle_form": "supervised"},
        "update": dict(UPDATE), "rows": dict(ROWS), "update_checkpoint_rows": 2, "stop_on_integrity_failure": True,
    }
    payload.update(overrides)
    config = RLConfig(**payload)
    config.validate()
    return config


def _step(number: int, prompt: list[int], completion: list[int], *, tool: str = "run_wls", status: str = "success") -> StepRecord:
    return StepRecord(
        step=number, model_observation={"prompt": list(prompt)}, prompt_tokens=len(prompt), original_prompt_tokens=len(prompt),
        completion_token_ids=list(completion), text=f"call:{tool}", action={"tool": tool, "arguments": {}}, tool=tool,
        execution_status=status, error_code=None if status == "success" else "invalid_action",
        token_probabilities=[0.9] * len(completion), prompt_digest=prompt_token_digest(prompt),
    )


def _decision(position: int, tool: str, *, true: bool) -> dict:
    label = COMMIT if true else ROLLBACK
    return {"position": position, "step": position, "tool": tool, "label": label, "agrees": tool == label, "target_is_true": true,
            "declined": tool == ROLLBACK, "verdict": "ACCEPT_PARTIAL", "correction_tool": "correct_parameters",
            "correction_position": position - 1, "arguments": {"line_index": 9}}


def _trajectory(root: str, sample: int, reward: float, steps: list[StepRecord], *, success: bool, lifecycle=()) -> Trajectory:
    return Trajectory(root=root, family="parameter", sample_index=sample, episode_seed=sample, steps=steps,
                      reward={"total": reward, "calls": len(steps)}, evaluation={"truth_audited_task_success": success},
                      pairing_ok=True, lifecycle=list(lifecycle))


def _write_rollouts(config: RLConfig, iteration: int = 0) -> dict[str, int]:
    """An adversarial root and a flat one; returns how many rows of each kind the update should see."""

    store = TrajectoryStore(iteration_dir(config, iteration) / "trajectories")
    shared = _step(0, [1, 2], [3])  # the four samples of a root part after one common step
    # Two samples commit a true candidate and solve; one declines it and fails; one fails after an invalid output.
    adversarial = [
        _trajectory("adv", 0, 0.8, [shared, _step(1, [2, 3], [4, 5]), _step(2, [3, 4], [11], tool=COMMIT)], success=True,
                    lifecycle=[_decision(2, COMMIT, true=True)]),
        _trajectory("adv", 1, 0.7, [shared, _step(1, [2, 3], [4, 6]), _step(2, [3, 4], [11], tool=COMMIT)], success=True,
                    lifecycle=[_decision(2, COMMIT, true=True)]),
        _trajectory("adv", 2, -1.2, [shared, _step(1, [2, 3], [4, 5]), _step(2, [3, 4], [10], tool=ROLLBACK),
                                     _step(3, [4, 5], [7, 8])], success=False, lifecycle=[_decision(2, ROLLBACK, true=True)]),
        _trajectory("adv", 3, -0.9, [shared, _step(1, [2, 3], [9], tool="__invalid_action__", status="failure"),
                                     _step(2, [3, 9], [4, 5])], success=False),
    ]
    flat = [_trajectory("flat", sample, 0.5, [_step(0, [5, 6], [7]), _step(1, [6, 7], [11], tool=COMMIT)], success=True,
                        lifecycle=[_decision(1, COMMIT, true=False)]) for sample in range(4)]
    store.write_group("adv", adversarial)
    store.write_group("flat", flat)
    return {"policy": 13, "preservation": 2, "supervised": 7}


@contextlib.contextmanager
def _tiny_model_phase(holder):
    """The update phase with the real scoring and row builder, on the tiny model.

    Stand-ins replace the model loader, the prompt functions (a step's model
    observation holds its prompt tokens) and the adapter writer.
    ``holder['fail_after']`` cuts the model's forward passes as a preemption
    does; ``holder['loads']`` counts how often a model was loaded.
    """

    import psse_env.rl.train as train

    def load(config, adapter, *, reference_adapter=None):
        holder["loads"] = holder.get("loads", 0) + 1
        model = _TinyLM(seed=3, fail_after=holder.get("fail_after"))
        weights = Path(adapter) / "weights.pt"
        if weights.is_file():
            model.load_state_dict(torch.load(weights))
        return model, None, ()

    def render_target(observation, text, decision):
        return [11] if decision["label"] == COMMIT else [10]

    render_target.rendered, render_target.refused = 0, 0

    def prompt_functions(config, adapter, processor):
        return (lambda observation: json.dumps(observation["prompt"])), (lambda text: (json.loads(text), len(json.loads(text)))), render_target

    def save(model, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(directory) / "weights.pt")

    names = ("_load_trainable_model", "_prompt_functions", "_save_trained_adapter")
    originals = [getattr(train, name) for name in names]
    for name, stand_in in zip(names, (load, prompt_functions, save)):
        setattr(train, name, stand_in)
    try:
        yield train
    finally:
        for name, original in zip(names, originals):
            setattr(train, name, original)


class SeriesConfigTests(unittest.TestCase):
    def test_the_new_fields_are_validated_and_default_to_what_recorded_runs_did(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = RLConfig(output_dir=directory, training_suite="t", development_suite=None, initial_adapter="a",
                             base_model="m", base_revision="0" * 40)
            plain.validate()
            self.assertEqual((plain.observation_mode, plain.decision_credit, plain.reserved_roots, plain.stop_on_integrity_failure),
                             (BOUNDED_HISTORY, {}, None, False))
            self.assertEqual(plain.evaluator_options(), {})
            self.assertEqual(iteration_phases(plain), ITERATION_PHASES)
            self.assertEqual(environment_record(plain)["observation_mode"], BOUNDED_HISTORY)
            complete = _config(directory, observation_mode=COMPLETE_HISTORY)
            self.assertEqual(complete.evaluator_options(), {"observation_mode": COMPLETE_HISTORY})
            credit = _config(directory, decision_credit={"enabled": True, "weight": 0.5, "states_per_iteration": 8})
            self.assertEqual(iteration_phases(credit), CREDIT_ITERATION_PHASES)
            self.assertEqual(credit.credit_settings().continuations, 2)
            with self.assertRaisesRegex(ValueError, "observation_mode"):
                _config(directory, observation_mode="everything")
            with self.assertRaisesRegex(ValueError, "lifecycle_choice must open"):
                _config(directory, decision_credit={"enabled": True}, environment_options={}, reward={})
            with self.assertRaisesRegex(ValueError, "unknown decision_credit fields"):
                _config(directory, decision_credit={"enabled": True, "branches": 3})
            with self.assertRaisesRegex(ValueError, "unknown kinds"):
                _config(directory, decision_credit={"enabled": True, "quotas": {"easy": 4}})
            with self.assertRaisesRegex(ValueError, "series must be a mapping"):
                _config(directory, series=["a"])

    def test_a_run_with_decision_credit_goes_through_the_continuation_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, decision_credit={"enabled": True})
            manifest = load_manifest(config)
            self.assertEqual(manifest.phase, "rollout")
            with self.assertRaisesRegex(RuntimeError, "at phase 'rollout'"):
                run_phase(config, "continuation")
            with self.assertRaisesRegex(RuntimeError, "decision_credit.enabled"):
                from psse_env.rl.train import phase_continuation

                phase_continuation(_config(directory), manifest)

    def test_reserved_roots_are_read_and_an_iteration_that_draws_one_is_refused(self) -> None:
        import psse_env.rl.train as train

        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "train.json"
            suite.write_text(json.dumps([{"execution": {"scenario_id": f"r{number}"}, "grouping": {"scenario_family": "parameter"}}
                                         for number in range(4)]), encoding="utf-8")
            reserved = Path(directory) / "reserved.json"
            reserved.write_text(json.dumps({"roots": ["r2"], "purpose": "diagnostic validation"}), encoding="utf-8")
            config = _config(directory, reserved_roots=str(reserved), policy_source="expert")
            self.assertEqual(reserved_roots(config), {"r2"})
            self.assertEqual(reserved_roots(_config(directory)), set())
            with self.assertRaisesRegex(RuntimeError, "draws 1 reserved roots"):
                train.phase_rollout(config, Manifest(current_adapter=config.initial_adapter))
            reserved.write_text(json.dumps("r2"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a list of root ids"):
                reserved_roots(config)


class PromptLimitConfigTests(unittest.TestCase):
    """``max_input_tokens``: the run's own prompt limit, set in every phase's process and recorded."""

    def setUp(self) -> None:
        import psse_env.dagger.preliminary_e2b_eval as limits

        self.limits = limits
        # Whatever a test sets, the process's own limits are back afterwards.
        for name in ("MAX_INPUT_TOKENS", "MAX_NEW_TOKENS"):
            patcher = mock.patch.object(limits, name, getattr(limits, name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_config_without_a_limit_runs_under_the_process_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.limits.MAX_INPUT_TOKENS, self.limits.MAX_NEW_TOKENS = 8192, 64
            config = _config(directory)
            self.assertIsNone(config.max_input_tokens)
            record = apply_prompt_limits(config)
            self.assertEqual(self.limits.MAX_INPUT_TOKENS, 8192)
            self.assertEqual(record, {"max_input_tokens": 8192, "max_new_tokens": 64, "usable_complete_history_tokens": 8128,
                                      "max_length": 64, "source": "environment", "model_context_window": None})

    def test_the_configs_limit_is_the_one_the_policy_applies(self) -> None:
        import psse_env.dagger.research_policy_factory as factory

        with tempfile.TemporaryDirectory() as directory:
            self.limits.MAX_INPUT_TOKENS, self.limits.MAX_NEW_TOKENS = 8192, 256
            config = _config(directory, max_input_tokens=49152, max_length=49152)
            record = apply_prompt_limits(config)
            self.assertEqual((record["max_input_tokens"], record["usable_complete_history_tokens"], record["source"]),
                             (49152, 48896, "config"))
            # What a complete-history prompt is refused above, and what a bounded one is cut at.
            self.assertEqual(factory.usable_prompt_tokens(), 48896)
            self.assertEqual(self.limits.MAX_INPUT_TOKENS, 49152)
            # A later config without a limit does not reset another's: a process serves one config.
            self.assertEqual(apply_prompt_limits(_config(directory))["max_input_tokens"], 49152)

    def test_the_training_length_must_hold_the_longest_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "longest training row"):
                _config(directory, max_input_tokens=128, max_length=64)
            for bad in (0, -5, True, 1.5, "4096"):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _config(directory, max_input_tokens=bad, max_length=1 << 20)

    def test_a_limit_beyond_the_models_window_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 1024}}), encoding="utf-8")
            self.assertEqual(model_context_window(str(model_dir)), 1024)
            self.limits.MAX_INPUT_TOKENS, self.limits.MAX_NEW_TOKENS = 8192, 64
            fits = _config(directory, base_model=str(model_dir), max_input_tokens=960, max_length=960)
            self.assertEqual(apply_prompt_limits(fits)["model_context_window"], 1024)
            # 961 prompt tokens and a completion of 64 are one position more than the model has.
            with self.assertRaisesRegex(ValueError, "1024 positions"):
                apply_prompt_limits(_config(directory, base_model=str(model_dir), max_input_tokens=961, max_length=961))
            self.assertEqual(self.limits.MAX_INPUT_TOKENS, 960)
            # A model whose config is not on this machine is not checked (and nothing is fetched for it).
            flat = Path(directory) / "flat"
            flat.mkdir()
            (flat / "config.json").write_text(json.dumps({"max_position_embeddings": 2048}), encoding="utf-8")
            self.assertEqual(model_context_window(str(flat)), 2048)
            self.assertIsNone(model_context_window(str(Path(directory) / "absent")))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_every_phase_summary_carries_the_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _tiny_model_phase({}) as train:
            self.limits.MAX_INPUT_TOKENS, self.limits.MAX_NEW_TOKENS = 8192, 8
            config = _config(directory, max_input_tokens=64, max_length=64)
            _write_rollouts(config)
            report = train.phase_update(config, Manifest(phase="update", current_adapter=config.initial_adapter))
            self.assertEqual(report["prompt_limits"], {"max_input_tokens": 64, "max_new_tokens": 8, "usable_complete_history_tokens": 56,
                                                       "max_length": 64, "source": "config", "model_context_window": None})
            self.assertTrue(report["integrity"]["ok"])


@unittest.skipIf(torch is None, "torch is not installed")
class AdversarialUpdateTests(unittest.TestCase):
    def _run(self, directory: str, holder: dict, *, cut: int | None = None, **overrides):
        with _tiny_model_phase(holder) as train:
            config = _config(directory, **overrides)
            kinds = _write_rollouts(config)
            manifest = Manifest(phase="update", current_adapter=config.initial_adapter)
            if cut is not None:
                holder["fail_after"] = cut
                with self.assertRaises(_Preempted):
                    train.phase_update(config, manifest)
                holder["fail_after"] = None
            report = train.phase_update(config, manifest)
            weights = torch.load(iteration_dir(config, 0) / "adapter" / "weights.pt")["table"]
        return config, kinds, report, weights

    def test_a_root_with_contrasting_samples_an_invalid_output_and_a_preemption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _config_a, kinds, report, expected = self._run(str(Path(directory) / "whole"), {})
            rows, update = report["rows"], report["update"]
            # The failed sample's invalid output is a sampled step like any other and is trained against.
            self.assertEqual((rows["invalid_steps_excluded"], rows["policy_rows"], rows["preservation_rows"], rows["supervised_rows"]),
                             (0, kinds["policy"], kinds["preservation"], kinds["supervised"]))
            self.assertEqual((rows["supervised_rows_agreeing"], rows["supervised_rows_disagreeing"]), (2, 5))
            self.assertEqual(rows["informative_groups"], 1)
            self.assertEqual(update["row_accounting"]["rows_by_kind"],
                             {"policy": 13, "preservation": 2, "lifecycle_only": 0, "decision_credit": 0, "supervised": 7})
            self.assertTrue(update["row_accounting"]["each_row_once"])
            # The first update of a run takes one step whatever the config's steps per update.
            self.assertEqual((update["optimizer_steps"], update["planned_steps"], update["fresh_optimizer"], update["split_groups"]),
                             (1, 1, True, 0))
            self.assertEqual(report["optimizer"], {"carried_from": None, "steps_before": 0, "steps_after": 1,
                                                   "continuity_since_iteration": 0})
            self.assertEqual((report["integrity"]["ok"], report["integrity"]["failures"]), (True, []))
            self.assertEqual(report["observation_mode"], BOUNDED_HISTORY)
            # What the step did: the supervised term's rows, the decline of a true target among the decisions.
            step = update["diagnostics"][0]
            self.assertEqual(step["components"]["supervised"]["rows"], 7)
            self.assertEqual(step["decisions"]["true_target"]["decisions"], 3)
            self.assertEqual(step["decisions"]["wrong_target"]["decisions"], 4)
            self.assertAlmostEqual(step["movement"]["max_abs_over_learning_rate"], 1.0, places=3)
            for number, cut in enumerate((4, 9, 17, 30)):
                holder: dict = {}
                _config_b, _kinds, resumed, weights = self._run(str(Path(directory) / f"cut{number}"), holder, cut=cut)
                self.assertTrue(torch.allclose(weights, expected, atol=1e-7), cut)
                self.assertTrue(resumed["integrity"]["ok"], resumed["integrity"])
                self.assertEqual(resumed["update"]["row_accounting"], update["row_accounting"])
                self.assertEqual(resumed["update"]["batch_digest"], update["batch_digest"])
                self.assertEqual(resumed["optimizer"]["steps_after"], 1)

    def test_the_second_update_takes_its_two_steps_and_continues_the_moments(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _tiny_model_phase({}) as train:
            config = _config(directory)
            _write_rollouts(config, 0)
            _write_rollouts(config, 1)
            first = train.phase_update(config, Manifest(phase="update", current_adapter=config.initial_adapter))
            second = train.phase_update(config, Manifest(iteration=1, phase="update", current_adapter=first["adapter_out"]))
            self.assertEqual((first["update"]["optimizer_steps"], second["update"]["optimizer_steps"]), (1, 2))
            self.assertEqual(second["update"]["fresh_optimizer"], False)
            self.assertEqual(second["optimizer"], {"carried_from": str(iteration_dir(config, 0) / train.OPTIMIZER_FILE),
                                                   "steps_before": 1, "steps_after": 3, "continuity_since_iteration": 0})
            self.assertEqual(second["update"]["split_groups"], 0)
            self.assertTrue(second["integrity"]["ok"])
            # The clip baseline of the second update is its own: nothing of the first update's checkpoint is left.
            self.assertFalse((iteration_dir(config, 0) / "update_checkpoint").exists())
            self.assertFalse((iteration_dir(config, 1) / "update_checkpoint").exists())
            # A run that stepped once without leaving its state starts the continuity over.
            (iteration_dir(config, 0) / train.OPTIMIZER_FILE).unlink()
            self.assertEqual(train._continuity_start(config, 2), 1)
            self.assertEqual(train._continuity_start(config, 1), 1)

    def test_an_update_that_fails_a_check_stops_the_run_and_is_not_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            holder: dict = {}
            with _tiny_model_phase(holder) as train:
                config = _config(directory)
                _write_rollouts(config)
                original = train.update_integrity
                train.update_integrity = lambda report: {"ok": False, "applied": True, "warnings": [], "checks": {},
                                                         "failures": ["likelihood_provenance: a test failure"]}
                try:
                    manifest_path = Path(directory) / "manifest.json"
                    Manifest(phase="update", current_adapter=config.initial_adapter).save(manifest_path)
                    with self.assertRaisesRegex(IntegrityError, "a test failure"):
                        run_phase(config, "update")
                    marker = iteration_dir(config, 0) / MARKER_FILE
                    self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["failures"], ["likelihood_provenance: a test failure"])
                    # The run did not advance, the report is there to read, and a retry does not train again.
                    self.assertEqual(load_manifest(config).phase, "update")
                    written = json.loads((iteration_dir(config, 0) / "update_report.json").read_text(encoding="utf-8"))
                    self.assertFalse(written["integrity"]["ok"])
                    loads = holder["loads"]
                    with self.assertRaisesRegex(IntegrityError, "remove the marker"):
                        run_phase(config, "update")
                    self.assertEqual(holder["loads"], loads)
                    # A config that does not stop on a failure records it and goes on.
                    lenient = _config(str(Path(directory) / "lenient"), stop_on_integrity_failure=False)
                    _write_rollouts(lenient)
                    report = train.phase_update(lenient, Manifest(phase="update", current_adapter=lenient.initial_adapter))
                    self.assertFalse(report["integrity"]["ok"])
                    self.assertFalse((iteration_dir(lenient, 0) / MARKER_FILE).exists())
                finally:
                    train.update_integrity = original

    def test_a_batch_whose_rows_are_all_rejected_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _tiny_model_phase({}) as train:
            config = _config(directory)
            _write_rollouts(config)
            Path(config.initial_adapter).mkdir()
            (Path(config.initial_adapter) / "adapter_config.json").write_text("{}", encoding="utf-8")
            # Every stored step now names another prompt than the one the update renders for it.
            store = TrajectoryStore(iteration_dir(config, 0) / "trajectories")
            for root in store.roots():
                group = store.read_group(root)
                for trajectory in group:
                    for step in trajectory.steps:
                        step.prompt_digest = "f" * 64
                store.write_group(root, group)
            Manifest(phase="update", current_adapter=config.initial_adapter).save(Path(directory) / "manifest.json")
            with self.assertRaisesRegex(IntegrityError, "prompt_digest_mismatches"):
                run_phase(config, "update")
            written = json.loads((iteration_dir(config, 0) / "update_report.json").read_text(encoding="utf-8"))
            self.assertEqual((written["update"], written["rows"]["rows"]), (None, 0))
            self.assertGreater(written["rows"]["prompt_digest_mismatches"], 0)
            self.assertEqual((written["integrity"]["ok"], written["integrity"]["applied"]), (False, False))
            # No step was taken and the run did not move on to the evaluation of an adapter that was never trained.
            self.assertTrue((iteration_dir(config, 0) / MARKER_FILE).is_file())
            self.assertEqual(load_manifest(config).phase, "update")
            with self.assertRaisesRegex(IntegrityError, "remove the marker"):
                train.phase_update(config, Manifest(phase="update", current_adapter=config.initial_adapter))

    def test_decision_values_replace_the_membership_label_on_the_rows_they_cover(self) -> None:
        from psse_env.rl.decision_credit import DECISION_CREDIT_CONTRACT

        with tempfile.TemporaryDirectory() as directory, _tiny_model_phase({}) as train:
            config = _config(directory, decision_credit={"enabled": True, "weight": 2.0})
            _write_rollouts(config)
            manifest = Manifest(phase="update", current_adapter=config.initial_adapter)
            with self.assertRaises(FileNotFoundError):
                train.phase_update(config, manifest)  # the continuation phase has not run
            values = {
                "contract": DECISION_CREDIT_CONTRACT, "iteration": 0, "adapter": config.initial_adapter,
                "observation_mode": BOUNDED_HISTORY, "selection": {"selected": {"true_first": 1}}, "summary": {"covered": 1},
                # The state the three deciding samples of the adversarial root share: committing is worth more.
                "states": [{"root": "adv", "covered": True, "rows": [
                    {"sample_index": 0, "position": 2, "tool": COMMIT, "advantage": 0.2},
                    {"sample_index": 1, "position": 2, "tool": COMMIT, "advantage": 0.2},
                    {"sample_index": 2, "position": 2, "tool": ROLLBACK, "advantage": -1.5}]}],
            }
            path = iteration_dir(config, 0) / "decision_values.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            report = train.phase_update(config, manifest)
            rows = report["rows"]
            self.assertEqual((rows["decision_values"], rows["decision_credit_rows"], rows["supervised_rows_replaced_by_credit"]), (3, 3, 3))
            self.assertEqual((rows["supervised_rows"], rows["decision_values_unmatched"], rows["decision_credit_weight"]), (4, 0, 2.0))
            self.assertEqual(report["update"]["row_accounting"]["rows_by_kind"],
                             {"policy": 10, "preservation": 2, "lifecycle_only": 0, "decision_credit": 3, "supervised": 4})
            self.assertEqual(report["decision_credit"]["values"], 3)
            self.assertTrue(report["integrity"]["ok"])
            self.assertEqual(report["update"]["diagnostics"][0]["components"]["lifecycle"]["rows"], 3)
            # Values of another adapter, iteration or observation mode are records, not targets.
            for field, wrong in (("adapter", "elsewhere"), ("iteration", 3), ("observation_mode", COMPLETE_HISTORY)):
                path.write_text(json.dumps({**values, field: wrong}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not targets for this update"):
                    train.phase_update(config, manifest)


if __name__ == "__main__":
    unittest.main()
