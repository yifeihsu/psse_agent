"""Stage 0 of the workplan of 2026-11-10: the repaired update, verified on the tiny bigram model.

What the workplan asks to be checked before any run of the new series:
continuous training against save-and-resume across step and iteration
boundaries; every admitted row in exactly one step of the declared pass;
the same aggregate objective and gradient however the pass is dealt; the
clip baseline bound to its own batch; and the report-only diagnostics
(gradient by loss term, dominant rows, step size, lifecycle probabilities)
changing nothing they report on.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from psse_env.rl.integrity import update_integrity
from psse_env.rl.rows import build_step_row
from psse_env.rl.update import (
    UpdateCheckpoint,
    UpdateSettings,
    _chunk_weights,
    _chunks,
    batch_digest,
    build_optimizer,
    completion_logprobs,
    decision_summary,
    load_optimizer_state,
    optimizer_steps_taken,
    policy_gradient_update,
    row_accounting,
    row_kind,
    save_optimizer_state,
)

try:
    import torch

    from psse_env.rl.test_update import _group_row, _load_tiny, _Preempted, _save_tiny, _TinyLM
except Exception:  # pragma: no cover - torch is optional for the pure-Python tests
    torch = None


def _groups(count: int = 4, *, shift: int = 0):
    """``count`` groups of two samples and two steps each; a group's advantages sum to zero."""

    rows = []
    for number in range(count):
        root = f"g{number}"
        base = (number + shift) % 6
        for sample, advantage in ((0, 1.0 + 0.5 * number), (1, -(1.0 + 0.5 * number))):
            rows.append(_group_row(root, sample, [base, base + 1], [base + 2 + sample], advantage, step=0))
            rows.append(_group_row(root, sample, [base + 1, base + 2 + sample], [base + 3, base + 4 + sample], advantage, step=1))
    return rows


def _decision(label: str, *, true: bool, ordinal: int = 0, tool: str | None = None):
    tool = tool or label
    return {"label": label, "tool": tool, "target_is_true": true, "agrees": tool == label, "ordinal": ordinal,
            "earlier_declines": 0, "correction_tool": "correct_parameters"}


if torch is not None:

    class _StepRecorder:
        """Stands in for the optimizer: keeps every step's accumulated gradient and leaves the weights alone."""

        def __init__(self, model) -> None:
            self.model = model
            self.grads: list = []

        def zero_grad(self, set_to_none=True):
            for parameter in self.model.parameters():
                parameter.grad = None

        def step(self):
            self.grads.append(self.model.table.grad.detach().clone())

        def state_dict(self):
            return {}

        def load_state_dict(self, value):
            pass


@unittest.skipIf(torch is None, "torch is not installed")
class PlannedStepTests(unittest.TestCase):
    def test_a_new_optimizer_takes_the_fresh_steps_and_a_carried_one_the_full_number(self) -> None:
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=2, fresh_optimizer_steps=1, chunk_by="group")
        rows = _groups()
        model = _TinyLM()
        optimizer = build_optimizer(model, settings)
        first = policy_gradient_update(model, rows, settings, optimizer=optimizer)
        self.assertEqual((first.optimizer_steps, first.planned_steps, first.fresh_optimizer), (1, 1, True))
        self.assertEqual(first.row_accounting["rows_per_step"], [len(rows)])
        carried = build_optimizer(model, settings, optimizer.state_dict())
        second = policy_gradient_update(model, rows, settings, optimizer=carried)
        self.assertEqual((second.optimizer_steps, second.planned_steps, second.fresh_optimizer), (2, 2, False))
        self.assertEqual(optimizer_steps_taken(carried.state_dict()), 3)
        # Without the setting a new optimizer takes every step, as before.
        default = policy_gradient_update(_TinyLM(), rows, replace(settings, fresh_optimizer_steps=None))
        self.assertEqual((default.optimizer_steps, default.planned_steps, default.fresh_optimizer), (2, 2, True))

    def test_two_steps_of_a_new_optimizer_follow_the_first_step_against_the_batch(self) -> None:
        # One weight (the logits after token 2), two whole groups that pull it opposite ways, the second three
        # times as hard, so the batch says: token 4 up, token 3 down.
        rate = 0.01
        rows = [_group_row("a", 0, [1, 2], [3], 1.0), _group_row("a", 1, [1, 2], [4], -1.0),
                _group_row("b", 0, [5, 2], [4], 3.0), _group_row("b", 1, [5, 2], [3], -3.0)]
        seed = next(seed for seed in range(20)
                    if [rows[i].root for i in _chunks(rows, UpdateSettings(optimizer_steps=2, chunk_by="group", seed=seed))[0]] == ["a", "a"])
        two = UpdateSettings(learning_rate=rate, optimizer_steps=2, chunk_by="group", seed=seed)

        def moved(settings):
            model = _TinyLM()
            before = model.table.detach().clone()
            report = policy_gradient_update(model, rows, settings)
            return (model.table.detach() - before)[2], report

        delta, report = moved(two)
        self.assertEqual(report.split_groups, 0)
        # Group a came first: token 3 went up although the batch wants it down.
        self.assertGreater(float(delta[3]), 0.2 * rate)
        self.assertLess(float(delta[4]), -0.2 * rate)
        delta, report = moved(replace(two, fresh_optimizer_steps=1))
        self.assertEqual(report.optimizer_steps, 1)
        self.assertAlmostEqual(float(delta[3]), -rate, delta=1e-4 * rate)
        self.assertAlmostEqual(float(delta[4]), rate, delta=1e-4 * rate)

    def test_a_resumed_update_plans_what_the_interrupted_one_planned(self) -> None:
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, fresh_optimizer_steps=1, chunk_by="group", seed=3)
        rows = _groups()
        with tempfile.TemporaryDirectory() as directory:
            start = Path(directory) / "start"
            _save_tiny(_TinyLM(seed=3), start)
            reference = _load_tiny(start, seed=3)
            expected = policy_gradient_update(reference, rows, settings)
            checkpoint = UpdateCheckpoint(directory=Path(directory) / "checkpoint", save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_load_tiny(start, seed=3, fail_after=5), rows, settings, checkpoint=checkpoint)
            state = checkpoint.load_state()
            self.assertEqual((state["planned_steps"], state["chunk_count"], state["chunk_rows_done"]), (1, 1, 4))
            resumed_model = _load_tiny(start, seed=3)
            resumed = policy_gradient_update(resumed_model, rows, settings, checkpoint=checkpoint)
            self.assertEqual((resumed.optimizer_steps, resumed.planned_steps), (expected.optimizer_steps, 1))
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-7))

    def test_the_first_step_can_clip_against_the_rows_own_forward(self) -> None:
        import psse_env.rl.update as update_module
        from psse_env.rl.test_update import _with_record

        rows = _with_record(_TinyLM(seed=1), _groups(4))
        base = UpdateSettings(learning_rate=0.01, optimizer_steps=2, chunk_by="group", clip_ratio=0.2, clip_reference="training", seed=3)

        def run(settings, *, shift: float = 0.0):
            """The update with the persisted baseline off by ``shift`` nats, as a pass on another GPU type leaves it."""

            original = update_module.completion_logprobs
            model = _TinyLM(seed=1)
            state = {"training": False}

            def shifted(model_, row, **kwargs):
                values = original(model_, row, **kwargs)
                return values if state["training"] or not shift else values + shift

            original_train = model.train

            def train(mode: bool = True):
                state["training"] = bool(mode)
                return original_train(mode)

            model.train = train
            update_module.completion_logprobs = shifted
            try:
                report = policy_gradient_update(model, rows, settings)
            finally:
                update_module.completion_logprobs = original
            return model, report

        # With an exact baseline the two settings are the same update.
        persisted_model, persisted = run(base)
        own_model, own = run(replace(base, first_step_baseline="own"))
        self.assertTrue(torch.allclose(persisted_model.table, own_model.table, atol=1e-7))
        self.assertEqual(persisted.clip_check[0]["tokens_beyond_threshold"], 0)
        # A baseline written 0.5 nats away (beyond the clip of ln 1.2) silences the first step under the persisted
        # baseline; with the row's own forward the first step is the plain one, and the report still shows the gap.
        plain_model, _plain = run(replace(base, clip_ratio=None))
        drifted_model, drifted = run(base, shift=0.5)
        repaired_model, repaired = run(replace(base, first_step_baseline="own"), shift=0.5)
        first_step = lambda model: (model.table - _TinyLM(seed=1).table).abs().max()  # noqa: E731
        self.assertGreater(drifted.clip_check[0]["tokens_beyond_threshold"], 0)
        self.assertEqual(repaired.clip_check[0]["tokens_beyond_threshold"], drifted.clip_check[0]["tokens_beyond_threshold"])
        self.assertFalse(torch.allclose(drifted_model.table, own_model.table, atol=1e-6))
        self.assertGreater(float(first_step(repaired_model)), 0.0)
        # An update of one step runs no baseline pass and is the plain step.
        single = replace(base, optimizer_steps=1, first_step_baseline="own")
        single_model, single_report = run(single, shift=0.5)
        plain_single_model, _ = run(replace(single, clip_ratio=None))
        self.assertTrue(torch.allclose(single_model.table, plain_single_model.table, atol=1e-8))
        self.assertEqual(single_report.clip_check[0]["rows"], 0)
        self.assertEqual(len(single_model.calls), len(plain_single_model.calls))                 # no extra forward per row
        self.assertGreater(len(run(replace(base, optimizer_steps=1))[0].calls), len(single_model.calls))  # the persisted baseline costs one
        self.assertEqual(single_report.optimizer_steps, 1)
        del plain_model
        with self.assertRaisesRegex(ValueError, "first_step_baseline"):
            UpdateSettings(first_step_baseline="mine").validate()

    def test_settings_validation(self) -> None:
        for bad in (dict(fresh_optimizer_steps=0), dict(fresh_optimizer_steps=True), dict(fresh_optimizer_steps=1.5),
                    dict(step_diagnostics="yes"), dict(diagnostic_top_rows=-1)):
            with self.assertRaises(ValueError):
                UpdateSettings(**bad).validate()
        UpdateSettings(fresh_optimizer_steps=1, step_diagnostics=True, diagnostic_top_rows=0).validate()


@unittest.skipIf(torch is None, "torch is not installed")
class ContinuityTests(unittest.TestCase):
    """Continuous training against saving and resuming, across iteration and step boundaries."""

    settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, fresh_optimizer_steps=1, chunk_by="group", carry_optimizer=True)

    def _batches(self):
        return [_groups(shift=0), _groups(shift=1), _groups(shift=2)]

    def _continuous(self):
        """Three updates in one process: one model, one optimizer, nothing saved."""

        model = _TinyLM(seed=3)
        optimizer = build_optimizer(model, self.settings)
        reports = [policy_gradient_update(model, rows, replace(self.settings, seed=7 + number), optimizer=optimizer)
                   for number, rows in enumerate(self._batches())]
        return model, optimizer, reports

    def test_training_on_equals_saving_and_reloading_between_iterations(self) -> None:
        model, optimizer, reports = self._continuous()
        self.assertEqual([report.optimizer_steps for report in reports], [1, 2, 2])
        self.assertEqual(optimizer_steps_taken(optimizer.state_dict()), 5)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current, state = _TinyLM(seed=3), None
            for number, rows in enumerate(self._batches()):
                settings = replace(self.settings, seed=7 + number)
                own = build_optimizer(current, settings, state)
                policy_gradient_update(current, rows, settings, optimizer=own)
                _save_tiny(current, base / f"adapter{number}")
                save_optimizer_state(own, base / f"optimizer{number}.pt")
                # A new process: the weights and the optimizer state come back from disk.
                current = _load_tiny(base / f"adapter{number}", seed=3)
                state = load_optimizer_state(base / f"optimizer{number}.pt")
            self.assertEqual(optimizer_steps_taken(state), 5)
        self.assertTrue(torch.equal(current.table, model.table))

    def test_a_preemption_inside_an_update_changes_nothing(self) -> None:
        model, _optimizer, _reports = self._continuous()
        batches = self._batches()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            # The first two updates, saved as the driver saves them.
            current = _TinyLM(seed=3)
            first = build_optimizer(current, self.settings)
            policy_gradient_update(current, batches[0], replace(self.settings, seed=7), optimizer=first)
            second = build_optimizer(current, self.settings, first.state_dict())
            policy_gradient_update(current, batches[1], replace(self.settings, seed=8), optimizer=second)
            _save_tiny(current, base / "start")
            save_optimizer_state(second, base / "carried.pt")
            settings = replace(self.settings, seed=9)
            # The third update is cut in its first step, in its second step, and right after its first step.
            for cut in (3, 11, 8):
                checkpoint = UpdateCheckpoint(directory=base / f"checkpoint{cut}", save_model=_save_tiny, every_rows=2)
                interrupted = _load_tiny(base / "start", seed=3, fail_after=cut)
                with self.assertRaises(_Preempted):
                    policy_gradient_update(interrupted, batches[2], settings, checkpoint=checkpoint,
                                           optimizer=build_optimizer(interrupted, settings, load_optimizer_state(base / "carried.pt")))
                # As the driver resumes: the checkpoint's weights when a step was committed, the optimizer
                # rebuilt from the state before this update (the checkpoint's own state is loaded over it).
                resumed_model = _load_tiny(checkpoint.model_dir() or (base / "start"), seed=3)
                resumed_optimizer = build_optimizer(resumed_model, settings, load_optimizer_state(base / "carried.pt"))
                report = policy_gradient_update(resumed_model, batches[2], settings, optimizer=resumed_optimizer, checkpoint=checkpoint)
                self.assertTrue(torch.allclose(resumed_model.table, model.table, atol=1e-7), cut)
                self.assertEqual(optimizer_steps_taken(resumed_optimizer.state_dict()), 5)
                self.assertTrue(report.row_accounting["each_row_once"])
                self.assertEqual(report.optimizer_steps, 2)


@unittest.skipIf(torch is None, "torch is not installed")
class AccountingTests(unittest.TestCase):
    def test_every_row_is_in_exactly_one_step_of_the_one_pass(self) -> None:
        rows = _groups() + [_group_row("flat", 0, [4, 5], [8], 0.0, preservation=True),
                            _group_row("flat", 1, [5, 6], [9], 0.0, supervised=True)]
        for steps, chunk_by in ((1, "trajectory"), (2, "trajectory"), (2, "group"), (4, "group")):
            model = _TinyLM()
            report = policy_gradient_update(model, rows, UpdateSettings(optimizer_steps=steps, chunk_by=chunk_by))
            accounting = report.row_accounting
            self.assertEqual((accounting["passes"], accounting["rows"], sum(accounting["rows_per_step"])), (1, len(rows), len(rows)))
            self.assertEqual((accounting["rows_duplicated"], accounting["rows_missing"], accounting["each_row_once"]), (0, 0, True))
            self.assertEqual(accounting["rows_by_kind"], {"policy": 16, "preservation": 1, "lifecycle_only": 0,
                                                         "decision_credit": 0, "supervised": 1})
            self.assertEqual(len(accounting["rows_per_step"]), report.optimizer_steps)
            # One forward pass per row: nothing was computed twice and nothing skipped.
            self.assertEqual(len(model.calls), len(rows))
        self.assertEqual([row_kind(row) for row in rows[-2:]], ["preservation", "supervised"])

    def test_a_deal_that_repeats_or_drops_a_row_is_refused_before_anything_is_trained(self) -> None:
        import psse_env.rl.update as update

        rows = _groups(2)
        self.assertFalse(row_accounting(rows, [[0, 1, 2], [2, 3, 4, 5, 6]])["each_row_once"])
        self.assertEqual(row_accounting(rows, [[0, 1, 2], [2, 3, 4, 5, 6]])["rows_missing"], 1)
        original = update._chunks
        update._chunks = lambda rows, settings: [[0, 1, 2, 3], [3, 4, 5, 6]]
        try:
            model = _TinyLM()
            before = model.table.detach().clone()
            with self.assertRaisesRegex(RuntimeError, "not one pass over the batch"):
                policy_gradient_update(model, rows, UpdateSettings(optimizer_steps=2))
            self.assertTrue(torch.equal(model.table.detach(), before))
            self.assertEqual(model.calls, [])
        finally:
            update._chunks = original

    def test_the_dealt_steps_add_up_to_the_objective_and_gradient_of_the_whole_batch(self) -> None:
        rows = _groups()
        whole = {"dapo": "policy_tokens", "sequence": "policy_rows", "trajectory": "policy_trajectories"}
        for normalization, unit in whole.items():
            base = UpdateSettings(learning_rate=0.01, loss_normalization=normalization, max_grad_norm=1e9, chunk_by="group")
            model = _TinyLM(seed=4)
            recorder = _StepRecorder(model)
            one = policy_gradient_update(model, rows, replace(base, optimizer_steps=1), optimizer=recorder)
            total = _chunk_weights(rows, list(range(len(rows))))[unit]
            for steps in (2, 4):
                settings = replace(base, optimizer_steps=steps)
                model = _TinyLM(seed=4)
                recorder_split = _StepRecorder(model)
                report = policy_gradient_update(model, rows, settings, optimizer=recorder_split)
                shares = [_chunk_weights(rows, chunk)[unit] / total for chunk in _chunks(rows, settings)]
                self.assertAlmostEqual(sum(shares), 1.0)
                self.assertEqual(len(recorder_split.grads), steps)
                combined = sum(share * gradient for share, gradient in zip(shares, recorder_split.grads))
                self.assertTrue(torch.allclose(combined, recorder.grads[0], atol=1e-6), (normalization, steps))
                self.assertAlmostEqual(sum(share * loss for share, loss in zip(shares, report.policy_loss)), one.policy_loss[0], places=6)
                self.assertEqual(report.split_groups, 0)

    def test_a_checkpoint_is_bound_to_its_batch(self) -> None:
        rows, other = _groups(shift=0), _groups(shift=1)
        self.assertEqual(len(rows), len(other))
        self.assertNotEqual(batch_digest(rows), batch_digest(other))
        self.assertEqual(batch_digest(rows), batch_digest(_groups(shift=0)))
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, chunk_by="group", seed=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(fail_after=5), rows, settings, checkpoint=checkpoint)
            self.assertEqual(checkpoint.load_state()["batch_digest"], batch_digest(rows))
            # Same number of rows and steps, another batch: the counts alone would let it through.
            with self.assertRaisesRegex(RuntimeError, "another batch digest"):
                policy_gradient_update(_TinyLM(), other, settings, checkpoint=checkpoint)
            report = policy_gradient_update(_TinyLM(), rows, settings, checkpoint=checkpoint)
            self.assertEqual(report.batch_digest, batch_digest(rows))

    def test_a_checkpoint_is_bound_to_the_order_of_its_rows_and_to_its_settings(self) -> None:
        import psse_env.rl.update as update_module

        rows = _groups()
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=1, chunk_by="group", max_row_grad_norm=1.0, seed=7)
        other = next(seed for seed in range(8, 60) if _chunks(rows, replace(settings, seed=seed)) != _chunks(rows, settings))
        self.assertNotEqual(update_module.plan_digest(_chunks(rows, settings)),
                            update_module.plan_digest(_chunks(rows, replace(settings, seed=other))))
        with tempfile.TemporaryDirectory() as directory:
            start = Path(directory) / "start"
            _save_tiny(_TinyLM(seed=3), start)
            reference = _load_tiny(start, seed=3)
            expected = policy_gradient_update(reference, rows, settings)
            checkpoint = UpdateCheckpoint(directory=Path(directory) / "checkpoint", save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_load_tiny(start, seed=3, fail_after=6), rows, settings, checkpoint=checkpoint)
            state = checkpoint.load_state()
            self.assertEqual((state["chunk_rows_done"], state["plan_digest"]), (6, expected.row_accounting["plan_digest"]))
            self.assertEqual(state["settings"], settings.as_dict())
            # Another seed deals the rows in another order: the six rows done are not the first six of that order, so
            # a resume would run rows twice and leave others out while the plan still says each row once.
            with self.assertRaisesRegex(RuntimeError, r"other settings \(then, now\): \{'seed': \(7, %d\)\}" % other):
                policy_gradient_update(_load_tiny(start, seed=3), rows, replace(settings, seed=other), checkpoint=checkpoint)
            # So is any other setting the gradient in the checkpoint was accumulated under.
            for changes in ({"max_row_grad_norm": 5.0}, {"loss_normalization": "trajectory"}, {"learning_rate": 0.01},
                            {"step_diagnostics": True}):
                with self.assertRaisesRegex(RuntimeError, "other settings"):
                    policy_gradient_update(_load_tiny(start, seed=3), rows, replace(settings, **changes), checkpoint=checkpoint)
            # The same settings and another deal (the dealing rule itself changed): the order's digest catches it.
            original = update_module._group_chunks
            update_module._group_chunks = lambda *args: [list(reversed(chunk)) for chunk in original(*args)]
            try:
                with self.assertRaisesRegex(RuntimeError, "another order of the rows"):
                    policy_gradient_update(_load_tiny(start, seed=3), rows, settings, checkpoint=checkpoint)
            finally:
                update_module._group_chunks = original
            # The refusals consumed nothing: the unchanged call resumes and ends where the uninterrupted update did.
            self.assertEqual(checkpoint.load_state(), state)
            resumed_model = _load_tiny(start, seed=3)
            resumed = policy_gradient_update(resumed_model, rows, settings, checkpoint=checkpoint)
            self.assertEqual(resumed.resumed_from, {"completed_chunks": 0, "chunk_rows_done": 6})
            self.assertEqual(resumed.row_accounting, expected.row_accounting)
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-7))

    def test_a_checkpoint_from_before_the_order_was_recorded_resumes_with_a_warning(self) -> None:
        rows = _groups()
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=1, chunk_by="group", seed=7)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(seed=3, fail_after=6), rows, settings, checkpoint=checkpoint)
            state = checkpoint.load_state()
            for name in ("plan_digest", "settings"):
                state.pop(name)
            checkpoint.state_path.write_text(json.dumps(state), encoding="utf-8")
            resumed = policy_gradient_update(_TinyLM(seed=3), rows, settings, checkpoint=checkpoint)
            self.assertEqual(resumed.resumed_from, {"completed_chunks": 0, "chunk_rows_done": 6, "plan_verified": False})
            result = update_integrity({"rows": {"rows": len(rows)}, "update": resumed.as_dict()})
            self.assertTrue(result["ok"], result["failures"])
            self.assertTrue(any("taken on trust" in warning for warning in result["warnings"]))
            self.assertEqual(result["checks"]["recovery"]["plan_digest"], resumed.row_accounting["plan_digest"])

    def test_a_clip_baseline_cannot_be_continued_on_another_batch(self) -> None:
        from psse_env.rl.test_update import _with_record

        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=1, clip_ratio=0.2, clip_reference="training")
        rows = _with_record(_TinyLM(), _groups(shift=0))
        other = _with_record(_TinyLM(), _groups(shift=1))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=4)
            with self.assertRaises(_Preempted):  # cut inside the baseline pass, before any optimizer step
                policy_gradient_update(_TinyLM(fail_after=6), rows, settings, checkpoint=checkpoint)
            self.assertIsNone(checkpoint.load_state())
            self.assertTrue((Path(directory) / "old_logprobs.partial.pt").is_file())
            # The other batch has the same number of rows and the same completion lengths.
            with self.assertRaisesRegex(RuntimeError, "partial clip baseline was written for a different batch"):
                policy_gradient_update(_TinyLM(), other, settings, checkpoint=checkpoint)
            resumed = policy_gradient_update(_TinyLM(), rows, settings, checkpoint=checkpoint)
            self.assertEqual(resumed.resumed_from["baseline_rows_done"], 4)


@unittest.skipIf(torch is None, "torch is not installed")
class DiagnosticsTests(unittest.TestCase):
    def _batch(self):
        rows = _groups(3)
        rows.append(_group_row("flat", 0, [4, 5], [8], 0.0, preservation=True))
        for number, (label, true, tool) in enumerate((("commit_state", True, "commit_state"), ("rollback_state", False, "commit_state"),
                                                      ("commit_state", True, "rollback_state"))):
            rows.append(build_step_row(root=f"g{number}", sample_index=0, step=5, advantage=0.0, prompt_ids=[number + 1, number + 2],
                                       completion_ids=[10 if label == "rollback_state" else 11], max_length=64, lifecycle_term=0.25,
                                       supervised=True, decision=_decision(label, true=true, ordinal=number, tool=tool)))
        return rows

    def _reference(self):
        frozen = _TinyLM(seed=9)
        return lambda row: completion_logprobs(frozen, row).detach()

    def test_the_diagnostics_change_neither_the_weights_nor_the_rest_of_the_report(self) -> None:
        rows = self._batch()
        for bound in (0.05, None):
            base = UpdateSettings(learning_rate=0.01, optimizer_steps=2, chunk_by="group", beta_kl=0.5, max_row_grad_norm=bound, seed=2)
            plain_model, traced_model = _TinyLM(seed=1), _TinyLM(seed=1)
            plain = policy_gradient_update(plain_model, rows, base, reference_logprobs=self._reference())
            traced = policy_gradient_update(traced_model, rows, replace(base, step_diagnostics=True), reference_logprobs=self._reference())
            self.assertTrue(torch.equal(plain_model.table, traced_model.table), bound)
            self.assertEqual(plain.diagnostics, [])
            self.assertEqual(len(traced.diagnostics), 2)
            for name in ("policy_loss", "kl", "supervised_loss", "grad_norm", "row_grad_norms", "row_accounting", "batch_digest"):
                self.assertEqual(getattr(plain, name), getattr(traced, name), name)
            self.assertEqual(len(plain_model.calls), len(traced_model.calls))  # extra backward passes, no extra forward

    def test_without_the_split_the_diagnostics_cost_no_backward_pass(self) -> None:
        from unittest import mock

        rows = self._batch()
        base = UpdateSettings(learning_rate=0.01, optimizer_steps=2, chunk_by="group", beta_kl=0.5, max_row_grad_norm=0.05, seed=2)
        original = torch.Tensor.backward

        def run(settings):
            passes: list[int] = []

            def counted(tensor, *args, **kwargs):
                passes.append(1)
                return original(tensor, *args, **kwargs)

            model = _TinyLM(seed=1)
            with mock.patch.object(torch.Tensor, "backward", counted):
                report = policy_gradient_update(model, rows, settings, reference_logprobs=self._reference())
            return model, report, len(passes)

        plain_model, _plain, plain_passes = run(base)
        lean_model, lean, lean_passes = run(replace(base, step_diagnostics=True, diagnostic_components=False))
        _full_model, full, full_passes = run(replace(base, step_diagnostics=True))
        self.assertEqual(lean_passes, plain_passes)
        self.assertGreater(full_passes, plain_passes)
        self.assertTrue(torch.equal(plain_model.table, lean_model.table))
        for lean_step, full_step in zip(lean.diagnostics, full.diagnostics):
            self.assertEqual(lean_step["components"], {})
            self.assertEqual(set(full_step["components"]), {"policy", "kl", "supervised"})
            # Everything but the split is the same record.
            for name in ("rows", "rows_clipped", "rows_by_kind", "top_rows", "movement", "decisions", "gradient_norm_before_global_clip"):
                self.assertEqual(lean_step[name], full_step[name], name)
        with self.assertRaisesRegex(ValueError, "diagnostic_components"):
            UpdateSettings(diagnostic_components="no").validate()

    def test_the_gradient_is_split_by_loss_term_and_the_parts_add_up(self) -> None:
        rows = self._batch()
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1, beta_kl=0.5, max_grad_norm=1e9, step_diagnostics=True)
        report = policy_gradient_update(_TinyLM(seed=1), rows, settings, reference_logprobs=self._reference())
        step = report.diagnostics[0]
        self.assertEqual(set(step["components"]), {"policy", "kl", "supervised"})
        self.assertAlmostEqual(sum(part["share_of_step"] for part in step["components"].values()), 1.0, places=5)
        self.assertEqual({name: part["rows"] for name, part in step["components"].items()}, {"policy": 12, "kl": 13, "supervised": 3})
        self.assertAlmostEqual(step["gradient_norm_before_global_clip"], report.grad_norm[0], places=6)
        # Each part is the gradient of its own term: the same batch with the other terms switched off.
        def alone(batch, **changes):
            model = _TinyLM(seed=1)
            recorder = _StepRecorder(model)
            policy_gradient_update(model, batch, replace(settings, step_diagnostics=False, **changes), optimizer=recorder,
                                   reference_logprobs=self._reference())
            return float(recorder.grads[0].norm())

        sampled = [row for row in rows if not row.supervised]
        self.assertAlmostEqual(step["components"]["policy"]["norm"], alone(sampled, beta_kl=0.0), places=6)
        self.assertAlmostEqual(step["components"]["supervised"]["norm"], alone([row for row in rows if row.supervised]), places=6)
        without_advantage = [build_step_row(root=row.root, sample_index=row.sample_index, step=row.step, advantage=0.0,
                                            prompt_ids=row.input_ids[: len(row.input_ids) - row.completion_tokens],
                                            completion_ids=row.input_ids[-row.completion_tokens:], max_length=64) for row in sampled]
        # The KL term is normalized over all sampled rows, so the policy rows keep their place in it.
        self.assertAlmostEqual(step["components"]["kl"]["norm"], alone(without_advantage), places=6)

    def test_the_rows_that_carry_a_step_are_named(self) -> None:
        rows = _groups(3)
        # One row with a far larger advantage than the rest of its step.
        rows.append(_group_row("heavy", 0, [7, 8], [9], 40.0))
        rows.append(_group_row("heavy", 1, [7, 8], [10], -40.0))
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1, max_row_grad_norm=0.5, step_diagnostics=True,
                                  diagnostic_top_rows=3)
        report = policy_gradient_update(_TinyLM(seed=1), rows, settings)
        step = report.diagnostics[0]
        self.assertEqual([item["root"] for item in step["top_rows"][:2]], ["heavy", "heavy"])
        self.assertEqual(len(step["top_rows"]), 3)
        self.assertTrue(all(item["clipped"] for item in step["top_rows"][:2]))
        self.assertEqual(step["rows_clipped"], report.row_grad_norms[0]["clipped"])
        self.assertAlmostEqual(step["rows_clipped_fraction"], step["rows_clipped"] / len(rows))
        self.assertGreater(step["top_rows_squared_norm_share"], 0.9)
        self.assertEqual(step["rows_by_kind"]["policy"]["rows"], len(rows))
        self.assertEqual({key for key in step["top_rows"][0]} >= {"root", "sample_index", "step", "kind", "advantage", "norm"}, True)
        self.assertAlmostEqual(step["global_clip_scale"], min(1.0, 1.0 / report.grad_norm[0]))

    def test_the_step_size_is_reported_in_units_of_the_learning_rate(self) -> None:
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=2, chunk_by="group", step_diagnostics=True, seed=1)
        model = _TinyLM(seed=1)
        before = model.table.detach().clone()
        report = policy_gradient_update(model, _groups(), settings)
        first, second = (step["movement"] for step in report.diagnostics)
        # The first step of a new AdamW moves every weight it touches by the learning rate, whatever its gradient.
        self.assertAlmostEqual(first["max_abs_over_learning_rate"], 1.0, places=3)
        self.assertLessEqual(first["rms_over_learning_rate"], 1.0)
        self.assertEqual(first["weights"], model.table.numel())
        self.assertGreater(first["cosine_with_descent"], 0.0)
        total = float((model.table.detach() - before).norm())
        self.assertLessEqual(total, first["update_norm"] + second["update_norm"] + 1e-9)
        self.assertGreater(first["relative_to_weight_norm"], 0.0)

    def test_what_the_policy_gives_the_lifecycle_choices_is_read_off_the_decision_rows(self) -> None:
        rows = self._batch()
        model = _TinyLM(seed=1)
        expected = [float(completion_logprobs(model, row).exp().prod()) for row in rows if row.supervised]
        report = policy_gradient_update(model, rows, UpdateSettings(learning_rate=1e-9, optimizer_steps=1, step_diagnostics=True))
        decisions = report.diagnostics[0]["decisions"]
        self.assertEqual((decisions["decisions"], decisions["true_target"]["decisions"], decisions["wrong_target"]["decisions"]), (3, 2, 1))
        # A supervised row's completion is the audit's choice: its probability is the decline probability when the
        # audit says decline, and one minus it when the audit says commit.
        self.assertAlmostEqual(decisions["wrong_target"]["mean_p_decline"], expected[1], places=6)
        self.assertAlmostEqual(decisions["true_target"]["mean_p_decline"], (2.0 - expected[0] - expected[2]) / 2.0, places=6)
        self.assertEqual(decisions["true_target_first_candidate"]["decisions"], 1)
        self.assertEqual(decisions["true_target_later_candidate"]["decisions"], 1)
        self.assertIn(decisions["auc_wrong_above_true"], (0.0, 0.5, 1.0))
        summary = decision_summary([{"p_decline": 0.9, "target_is_true": False, "ordinal": 0},
                                    {"p_decline": 0.2, "target_is_true": True, "ordinal": 0},
                                    {"p_decline": 0.6, "target_is_true": True, "ordinal": 1, "earlier_declines": 1}])
        self.assertEqual(summary["auc_wrong_above_true"], 1.0)
        self.assertEqual(summary["true_target"]["p_decline_above_half"], 1)
        self.assertEqual(summary["true_target_after_a_decline"]["decisions"], 1)

    def test_a_preempted_step_keeps_its_diagnostics(self) -> None:
        rows = self._batch()
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, chunk_by="group", beta_kl=0.5, max_row_grad_norm=0.5,
                                  step_diagnostics=True, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            start = Path(directory) / "start"
            _save_tiny(_TinyLM(seed=1), start)
            reference = _load_tiny(start, seed=1)
            expected = policy_gradient_update(reference, rows, settings, reference_logprobs=self._reference())
            for cut in (3, 12):
                checkpoint = UpdateCheckpoint(directory=Path(directory) / f"checkpoint{cut}", save_model=_save_tiny, every_rows=1)
                with self.assertRaises(_Preempted):
                    policy_gradient_update(_load_tiny(start, seed=1, fail_after=cut), rows, settings, checkpoint=checkpoint,
                                           reference_logprobs=self._reference())
                resumed_model = _load_tiny(checkpoint.model_dir() or start, seed=1)
                resumed = policy_gradient_update(resumed_model, rows, settings, checkpoint=checkpoint, reference_logprobs=self._reference())
                self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-7))
                self.assertEqual(len(resumed.diagnostics), 2)
                for got, want in zip(resumed.diagnostics, expected.diagnostics):
                    self.assertEqual((got["rows"], got["rows_clipped"]), (want["rows"], want["rows_clipped"]))
                    self.assertEqual([item["row"] for item in got["top_rows"]], [item["row"] for item in want["top_rows"]])
                    self.assertEqual(set(got["components"]), set(want["components"]))
                    for name, part in want["components"].items():
                        self.assertEqual(got["components"][name]["rows"], part["rows"])
                        self.assertAlmostEqual(got["components"][name]["norm"], part["norm"], places=5)
                    self.assertEqual(got["decisions"]["decisions"], want["decisions"]["decisions"])


class IntegrityTests(unittest.TestCase):
    def _report(self, **changes):
        report = {
            "rows": {"rows": 8, "prompt_length_mismatches": 0, "prompt_digest_mismatches": 0, "prompt_digest_missing": 0,
                     "rows_without_sampling_record_rejected": 0, "prompt_truncated_rows": 0, "prompts_cut_at_generation": 0,
                     "target_renderer": {"rendered": 3, "refused": 0}},
            "optimizer": {"steps_before": 1, "steps_after": 3},
            "update": {"rows": 8, "optimizer_steps": 2, "planned_steps": 2, "policy_loss": [0.1, -0.1], "kl": [0.0, 0.01],
                       "lifecycle_loss": [0.0, 0.0], "supervised_loss": [0.01, 0.01], "grad_norm": [1.2, 0.8],
                       "row_grad_norms": [{"dropped_nonfinite": 0}, {"dropped_nonfinite": 0}],
                       "on_policy_check": [{"rows_without_record": 0, "mean_abs_log_ratio": 0.001}, {}],
                       "settings": {"max_on_policy_log_ratio": 0.2}, "row_accounting": {"each_row_once": True},
                       "batch_digest": "d", "resumed_from": None},
        }
        for path, value in changes.items():
            target = report
            keys = path.split(".")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        return report

    def test_a_clean_update_passes_and_each_stop_condition_is_named(self) -> None:
        clean = update_integrity(self._report())
        self.assertEqual((clean["ok"], clean["applied"], clean["failures"]), (True, True, []))
        cases = {
            "nonfinite_update": {"update.grad_norm": [1.2, float("nan")]},
            "nonfinite_update: 2 rows": {"update.row_grad_norms": [{"dropped_nonfinite": 2}, {"dropped_nonfinite": 0}]},
            "prompt_digest_mismatches": {"rows.prompt_digest_mismatches": 1},
            "without a sampling record": {"rows.rows_without_sampling_record_rejected": 4},
            "could not be rendered": {"rows.target_renderer": {"rendered": 2, "refused": 1}},
            "off policy": {"update.on_policy_check": [{"rows_without_record": 0, "mean_abs_log_ratio": 0.4}, {}]},
            "does not fit the training length": {"rows.prompt_truncated_rows": 1},
            "cut at its input limit": {"rows.prompts_cut_at_generation": 3},
            "in no optimizer step or in more than one": {"update.row_accounting": {"each_row_once": False}},
            "8 rows were built and 7 were updated": {"update.rows": 7},
            "carried optimizer counts 1 steps": {"optimizer": {"steps_before": 1, "steps_after": 2}},
            "continuation values without a sampled decision row": {"rows.decision_values_unmatched": 1},
        }
        for needle, changes in cases.items():
            result = update_integrity(self._report(**changes))
            self.assertFalse(result["ok"], needle)
            self.assertTrue(any(needle in failure for failure in result["failures"]), (needle, result["failures"]))
        # An update that was not applied has nothing to fail; a resumed one without a digest is a warning, not a stop.
        self.assertEqual(update_integrity({"update": None, "note": "carried over"})["applied"], False)
        resumed = update_integrity(self._report(**{"update.resumed_from": {"completed_chunks": 1}, "update.batch_digest": None}))
        self.assertTrue(resumed["ok"])
        self.assertTrue(any("no batch digest" in warning for warning in resumed["warnings"]))
        self.assertTrue(math.isfinite(clean["checks"]["provenance"]["first_step_mean_abs_log_ratio"]))

    def test_a_batch_whose_rows_were_all_rejected_is_a_stop_and_not_a_pass(self) -> None:
        # No row survived, so no step was taken; the reasons the rows were rejected for are stop conditions all the same.
        rejected = update_integrity({
            "update": None, "note": "no training rows after pairing and rendering checks",
            "rows": {"rows": 0, "prompt_digest_mismatches": 22, "prompts_cut_at_generation": 15,
                     "target_renderer": {"rendered": 0, "refused": 0}}})
        self.assertEqual((rejected["ok"], rejected["applied"]), (False, False))
        self.assertTrue(any("22 steps whose re-rendered prompt is not the sampled one" in failure for failure in rejected["failures"]))
        self.assertTrue(any("15 steps whose prompt the policy cut" in failure for failure in rejected["failures"]))
        self.assertEqual(rejected["checks"]["update_work"], {"rows_built": 0, "rows_updated": None})
        for counter, needle in (("prompt_length_mismatches", "prompt_length_mismatches"), ("prompt_digest_missing", "prompt_digest_missing"),
                                ("rows_without_sampling_record_rejected", "without a sampling record"),
                                ("supervised_rows_unrenderable", "could not be rendered"), ("prompt_truncated_rows", "training length")):
            result = update_integrity({"update": None, "rows": {"rows": 0, counter: 3}})
            self.assertFalse(result["ok"], counter)
            self.assertTrue(any(needle in failure for failure in result["failures"]), (counter, result["failures"]))
        # No row and nothing that stops a run accounts for it: something to read, not a stop.
        empty = update_integrity({"update": None, "note": "no training rows", "rows": {"rows": 0, "empty_completions": 4}})
        self.assertEqual((empty["ok"], empty["applied"], empty["failures"]), (True, False, []))
        self.assertTrue(any("no training row was built" in warning for warning in empty["warnings"]))
        self.assertTrue(any("empty_completions = 4" in warning for warning in empty["warnings"]))
        # An update with nothing to train on never built rows: nothing to check and nothing to warn about.
        carried = update_integrity({"update": None, "note": "carried over"})
        self.assertEqual((carried["ok"], carried["applied"], carried["failures"], carried["warnings"]), (True, False, [], []))
        self.assertEqual(carried["checks"]["note"], "carried over")


if __name__ == "__main__":
    unittest.main()
