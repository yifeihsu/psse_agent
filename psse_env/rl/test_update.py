from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from psse_env.rl.rows import build_step_row
from psse_env.rl.update import (
    RowGradientBound,
    UpdateCheckpoint,
    UpdateSettings,
    _bounded_row_backward,
    completion_logprobs,
    policy_gradient_update,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for the pure-Python tests
    torch = None


if torch is not None:

    class _Outputs:
        def __init__(self, logits):
            self.logits = logits

    class _Preempted(RuntimeError):
        pass

    class _TinyLM(torch.nn.Module):
        """A bigram model: logits depend on the previous token only."""

        vocab = 12

        def __init__(self, seed: int = 0, fail_after: int | None = None) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.table = torch.nn.Parameter(torch.randn(self.vocab, self.vocab, generator=generator) * 0.1)
            self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.calls: list[int | None] = []
            self.fail_after = fail_after

        @property
        def device(self):
            return self.table.device

        def forward(self, input_ids, attention_mask=None, logits_to_keep=None, use_cache=False, **side):
            if self.fail_after is not None and len(self.calls) >= self.fail_after:
                raise _Preempted("preempted")
            self.calls.append(logits_to_keep)
            logits = self.table[input_ids]
            if logits_to_keep:
                logits = logits[:, -int(logits_to_keep) :, :]
            return _Outputs(logits)

    class _NoKeepLM(_TinyLM):
        def forward(self, input_ids, attention_mask=None, use_cache=False, **side):  # no logits_to_keep
            self.calls.append(None)
            return _Outputs(self.table[input_ids])

    def _save_tiny(model, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory / "weights.pt")

    def _load_tiny(directory: Path, **kwargs) -> "_TinyLM":
        model = _TinyLM(**kwargs)
        model.load_state_dict(torch.load(directory / "weights.pt"))
        return model

    class _GradRecorder:
        """Stands in for the optimizer: keeps the accumulated gradient of the one step instead of applying it."""

        def __init__(self, model) -> None:
            self.model = model
            self.grad = None

        def zero_grad(self, set_to_none=True):
            for parameter in self.model.parameters():
                parameter.grad = None

        def step(self):
            self.grad = self.model.table.grad.detach().clone()

        def state_dict(self):
            return {}

        def load_state_dict(self, value):
            pass

    def _with_record(model, rows, *, shift: float = 0.0):
        """Attach each row's own log-probabilities under ``model`` (plus ``shift``) as its sampling record."""
        for row in rows:
            row.sampling_logprobs = [value + shift for value in completion_logprobs(model, row).detach().tolist()]
        return rows


def _row(prompt, completion, advantage, sample=0, preservation=False):
    """One row; rows with different ``sample`` are different trajectories (the update chunks by trajectory)."""
    return build_step_row(root="r", sample_index=sample, step=0, advantage=advantage, prompt_ids=prompt,
                          completion_ids=completion, max_length=64, preservation=preservation)


def _rows():
    return [_row([1, 2, 3], [4, 5], 1.0, 0), _row([2, 3], [6], -1.0, 1), _row([3, 4, 5], [7, 8], 0.5, 2),
            _row([5, 6], [9], -0.5, 3), _row([6, 7, 8], [10, 11], 1.5, 4), _row([8, 9], [2], -1.5, 5)]


def _rows_after_token_3():
    """Rows whose first completion token is predicted from token 3, so one step moves every row's logits."""
    return [_row([1, 2, 3], [4, 5], 1.0, 0), _row([2, 3], [6], -1.0, 1), _row([0, 3], [7, 8], 0.5, 2),
            _row([5, 3], [9], -0.5, 3), _row([6, 7, 3], [10, 11], 1.5, 4), _row([8, 3], [2], -1.5, 5)]


@unittest.skipIf(torch is None, "torch is not installed")
class UpdateTests(unittest.TestCase):
    def test_completion_logprobs_match_a_full_forward_pass(self) -> None:
        model = _TinyLM()
        row = _row([1, 2, 3, 4], [5, 6], 1.0)
        kept = completion_logprobs(model, row, use_logits_to_keep=True)
        full = completion_logprobs(model, row, use_logits_to_keep=False)
        self.assertEqual(model.calls[0], 3)
        self.assertEqual(tuple(kept.shape), (2,))
        self.assertTrue(torch.allclose(kept, full, atol=1e-6))
        ids = torch.tensor([row.input_ids])
        logits = model.table[ids][0]
        expected = torch.log_softmax(logits[-3:-1], dim=-1).gather(-1, ids[0, -2:].unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(kept, expected, atol=1e-6))

    def test_log_probabilities_are_taken_at_the_sampling_temperature(self) -> None:
        model = _TinyLM()
        row = _row([1, 2, 3], [4, 5], 1.0)
        tempered = completion_logprobs(model, row, temperature=2.0)
        ids = torch.tensor([row.input_ids])
        logits = model.table[ids][0] / 2.0
        expected = torch.log_softmax(logits[-3:-1], dim=-1).gather(-1, ids[0, -2:].unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(tempered, expected, atol=1e-6))
        self.assertFalse(torch.allclose(tempered, completion_logprobs(model, row), atol=1e-4))

    def test_falls_back_when_the_model_has_no_logits_to_keep(self) -> None:
        model = _NoKeepLM()
        logprobs = completion_logprobs(model, _row([1, 2, 3], [4], 1.0), use_logits_to_keep=True)
        self.assertEqual(tuple(logprobs.shape), (1,))

    def test_positive_advantage_raises_the_sampled_likelihood_and_negative_lowers_it(self) -> None:
        for advantage in (1.0, -1.0):
            model = _TinyLM()
            row = _row([1, 2, 3], [4, 5], advantage)
            before = completion_logprobs(model, row).sum().item()
            report = policy_gradient_update(model, [row], UpdateSettings(learning_rate=0.1, optimizer_steps=1))
            after = completion_logprobs(model, row).sum().item()
            if advantage > 0:
                self.assertGreater(after, before)
            else:
                self.assertLess(after, before)
            self.assertEqual((report.rows, report.completion_tokens, report.optimizer_steps), (1, 2, 1))
            self.assertTrue(math.isfinite(report.grad_norm[0]))
            self.assertEqual(report.kl, [0.0])
            self.assertIsNone(report.resumed_from)

    def test_zero_advantage_rows_leave_the_parameters_unchanged(self) -> None:
        model = _TinyLM()
        before = model.table.detach().clone()
        report = policy_gradient_update(model, [_row([1, 2], [3], 0.0)], UpdateSettings(learning_rate=0.1))
        self.assertTrue(torch.equal(before, model.table.detach()))
        self.assertEqual(report.grad_norm, [0.0])

    def test_kl_penalty_needs_a_reference_and_uses_the_k3_estimator(self) -> None:
        row = _row([1, 2, 3], [4], 0.0)
        settings = UpdateSettings(learning_rate=0.05, beta_kl=1.0, optimizer_steps=1)
        with self.assertRaisesRegex(ValueError, "starting adapter"):
            policy_gradient_update(_TinyLM(), [row], settings)
        report = policy_gradient_update(_TinyLM(), [row], settings, reference_logprobs=lambda _row: torch.zeros(1))
        self.assertGreater(report.kl[0], 0.0)
        self.assertGreater(report.grad_norm[0], 0.0)
        model = _TinyLM()
        same = lambda r: completion_logprobs(model, r).detach()
        report = policy_gradient_update(model, [row], settings, reference_logprobs=same)
        self.assertAlmostEqual(report.kl[0], 0.0, places=6)

    def test_optimizer_steps_partition_the_rows(self) -> None:
        model = _TinyLM()
        rows = [_row([1, 2], [3], 1.0, 0), _row([2, 3], [4], -1.0, 1), _row([3, 4], [5], 0.5, 2)]
        seen: list[int] = []
        report = policy_gradient_update(model, rows, UpdateSettings(learning_rate=0.01, optimizer_steps=2),
                                        progress=lambda record: seen.append(int(record["rows"])))
        self.assertEqual(report.optimizer_steps, 2)
        self.assertEqual(sorted(seen), [1, 2])
        self.assertAlmostEqual(report.mean_advantage, 0.5 / 3.0)

    def test_an_interrupted_update_resumes_to_the_same_parameters(self) -> None:
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, seed=7)
        reference = _TinyLM(seed=3)
        expected = policy_gradient_update(reference, _rows(), settings)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            # Chunks hold three rows each: the run completes step one, saves the
            # gradients of two rows of step two, and is preempted on the third.
            interrupted = _TinyLM(seed=3, fail_after=5)
            with self.assertRaises(_Preempted):
                policy_gradient_update(interrupted, _rows(), settings, checkpoint=checkpoint)
            state = checkpoint.load_state()
            self.assertEqual((state["completed_chunks"], state["chunk_rows_done"]), (1, 2))
            self.assertIsNotNone(checkpoint.model_dir())
            resumed_model = _load_tiny(checkpoint.model_dir(), seed=3)
            resumed = policy_gradient_update(resumed_model, _rows(), settings, checkpoint=checkpoint)
            self.assertEqual(resumed.resumed_from, {"completed_chunks": 1, "chunk_rows_done": 2})
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-6))
            self.assertEqual(len(resumed.policy_loss), 2)
            for got, want in zip(resumed.policy_loss, expected.policy_loss):
                self.assertAlmostEqual(got, want, places=5)
            files = sorted(path.name for path in Path(directory).iterdir())
            self.assertIn("state.json", files)
            self.assertEqual([name for name in files if name.startswith("grads_")], [])

    def test_a_checkpoint_for_other_rows_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=1)
            policy_gradient_update(_TinyLM(), _rows(), UpdateSettings(optimizer_steps=2), checkpoint=checkpoint)
            with self.assertRaisesRegex(RuntimeError, "different batch"):
                policy_gradient_update(_TinyLM(), _rows()[:3], UpdateSettings(optimizer_steps=2), checkpoint=checkpoint)

    def test_the_row_bound_caps_an_outlier_row(self) -> None:
        rows = [_row([1, 2, 3], [4, 5], 1.0), _row([2, 3], [6], -1.0), _row([3, 4], [7], 0.5),
                _row([5, 6, 7], [8, 9], 400.0)]

        class _Recorder:
            def __init__(self):
                self.grad = None

            def zero_grad(self, set_to_none=True):
                for parameter in model.parameters():
                    parameter.grad = None

            def step(self):
                self.grad = model.table.grad.detach().clone()

            def state_dict(self):
                return {}

            def load_state_dict(self, value):
                pass

        bound = 0.05
        expected = torch.zeros_like(_TinyLM(seed=4).table)
        norms = []
        for row in rows:
            single = _TinyLM(seed=4)
            tokens = sum(r.completion_tokens for r in rows)
            loss = -(float(row.advantage) * completion_logprobs(single, row)).sum() / tokens
            loss.backward()
            norm = float(single.table.grad.norm())
            norms.append(norm)
            expected += single.table.grad * min(1.0, bound / norm)
        self.assertGreater(max(norms), 20 * bound)
        model = _TinyLM(seed=4)
        recorder = _Recorder()
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1, max_grad_norm=1e9, max_row_grad_norm=bound)
        report = policy_gradient_update(model, rows, settings, optimizer=recorder)
        self.assertTrue(torch.allclose(recorder.grad, expected, atol=1e-6))
        summary = report.row_grad_norms[0]
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["clipped"], sum(1 for norm in norms if norm > bound))
        self.assertAlmostEqual(summary["max"], max(norms), places=5)

    def test_the_median_bound_starts_after_the_warmup(self) -> None:
        bound = RowGradientBound(UpdateSettings(row_grad_norm_factor=3.0, row_grad_norm_warmup=3))
        self.assertIsNone(bound.bound())
        for norm in (0.0, 1.0, 2.0):
            bound.record(norm, clipped=False, dropped=False)
        self.assertIsNone(bound.bound())
        bound.record(4.0, clipped=False, dropped=False)
        self.assertAlmostEqual(bound.bound(), 3.0 * 2.0)
        capped = RowGradientBound(UpdateSettings(max_row_grad_norm=5.0, row_grad_norm_factor=3.0, row_grad_norm_warmup=1), norms=[4.0])
        self.assertAlmostEqual(capped.bound(), 5.0)
        capped.record(float("nan"), clipped=False, dropped=True)
        summary = capped.summary()
        self.assertEqual((summary["rows"], summary["dropped_nonfinite"], summary["nonzero_rows"]), (2, 1, 1))

    def test_a_row_with_a_nonfinite_gradient_is_dropped(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.grad = torch.full((3,), 0.5)
        bound = RowGradientBound(UpdateSettings(max_row_grad_norm=10.0))
        _bounded_row_backward((parameter * float("inf")).sum(), [parameter], bound, torch)
        self.assertTrue(torch.equal(parameter.grad, torch.full((3,), 0.5)))
        self.assertEqual(bound.dropped, 1)
        _bounded_row_backward((parameter * 2.0).sum(), [parameter], bound, torch)
        self.assertTrue(torch.allclose(parameter.grad, torch.full((3,), 2.5)))

    def test_a_bounded_update_resumes_to_the_same_parameters(self) -> None:
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, seed=7, max_row_grad_norm=0.5,
                                  row_grad_norm_factor=1.5, row_grad_norm_warmup=1)
        reference = _TinyLM(seed=3)
        expected = policy_gradient_update(reference, _rows(), settings)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(seed=3, fail_after=5), _rows(), settings, checkpoint=checkpoint)
            self.assertEqual(len(checkpoint.load_state()["partial_row_norms"]["norms"]), 2)
            resumed_model = _load_tiny(checkpoint.model_dir(), seed=3)
            resumed = policy_gradient_update(resumed_model, _rows(), settings, checkpoint=checkpoint)
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-6))
            self.assertEqual([s["clipped"] for s in resumed.row_grad_norms], [s["clipped"] for s in expected.row_grad_norms])
            self.assertEqual(len(resumed.row_grad_norms), 2)

    def test_the_on_policy_check_is_exact_at_the_first_step_and_sees_the_drift_after_it(self) -> None:
        model = _TinyLM(seed=2)
        rows = _with_record(model, _rows_after_token_3())
        rows[4].sampling_logprobs = None
        rows[5].sampling_logprobs = None
        report = policy_gradient_update(model, rows, UpdateSettings(learning_rate=0.5, optimizer_steps=2, seed=7))
        self.assertEqual(len(report.on_policy_check), 2)
        first, second = report.on_policy_check
        self.assertEqual(first["rows"] + first["rows_without_record"], 3)
        self.assertEqual(first["rows_without_record"] + second["rows_without_record"], 2)
        self.assertGreater(first["tokens"], 0)
        self.assertAlmostEqual(first["mean_abs_log_ratio"], 0.0, places=5)
        self.assertAlmostEqual(first["max_abs_log_ratio"], 0.0, places=5)
        self.assertEqual(first["tokens_beyond_threshold"], 0)
        self.assertGreater(second["mean_abs_log_ratio"], 1e-3)
        self.assertAlmostEqual(second["threshold_log_ratio"], math.log(1.2))

    def test_the_clipped_ratio_has_the_plain_gradient_on_policy_and_none_beyond_the_clip(self) -> None:
        def gradient(rows, settings):
            model = _TinyLM(seed=4)
            recorder = _GradRecorder(model)
            policy_gradient_update(model, rows, settings, optimizer=recorder)
            return recorder.grad

        plain = UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e9)
        clipped = UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e9, clip_ratio=0.2)
        on_policy = _with_record(_TinyLM(seed=4), _rows_after_token_3())
        self.assertTrue(torch.allclose(gradient(on_policy, clipped), gradient(on_policy, plain), atol=1e-6))
        # The recorded probability is 1.5 times smaller than the current one: ratio 1.5, beyond the clip.
        too_high = _with_record(_TinyLM(seed=4), [_row([1, 2, 3], [4, 5], 1.0)], shift=-math.log(1.5))
        self.assertTrue(torch.equal(gradient(too_high, clipped), torch.zeros(12, 12)))
        self.assertFalse(torch.equal(gradient(too_high, plain), torch.zeros(12, 12)))
        # A negative advantage with a ratio of 1.5 is unclipped (the surrogate still pushes the action down) ...
        too_high_negative = _with_record(_TinyLM(seed=4), [_row([1, 2, 3], [4, 5], -1.0)], shift=-math.log(1.5))
        self.assertFalse(torch.equal(gradient(too_high_negative, clipped), torch.zeros(12, 12)))
        # ... and a ratio of 1/1.5 with a negative advantage is clipped.
        too_low_negative = _with_record(_TinyLM(seed=4), [_row([1, 2, 3], [4, 5], -1.0)], shift=math.log(1.5))
        self.assertTrue(torch.equal(gradient(too_low_negative, clipped), torch.zeros(12, 12)))
        # A clip without a record is refused, never a silent plain step.
        bare = [_row([1, 2, 3], [4, 5], 1.0)]
        with self.assertRaisesRegex(ValueError, "behaviour log-probabilities"):
            gradient(bare, clipped)
        self.assertFalse(torch.equal(gradient(bare, plain), torch.zeros(12, 12)))

    def test_a_clipped_update_with_the_check_resumes_to_the_same_parameters_and_report(self) -> None:
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, seed=7, clip_ratio=0.2)
        reference = _TinyLM(seed=3)
        rows = _with_record(_TinyLM(seed=3), _rows())
        expected = policy_gradient_update(reference, rows, settings)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(seed=3, fail_after=5), rows, settings, checkpoint=checkpoint)
            self.assertEqual(checkpoint.load_state()["partial_on_policy"]["rows"], 2)
            resumed_model = _load_tiny(checkpoint.model_dir(), seed=3)
            resumed = policy_gradient_update(resumed_model, rows, settings, checkpoint=checkpoint)
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-6))
            self.assertEqual([c["rows"] for c in resumed.on_policy_check], [c["rows"] for c in expected.on_policy_check])
            self.assertEqual([c["tokens"] for c in resumed.on_policy_check], [c["tokens"] for c in expected.on_policy_check])
            for got, want in zip(resumed.on_policy_check, expected.on_policy_check):
                self.assertAlmostEqual(got["mean_abs_log_ratio"], want["mean_abs_log_ratio"], places=5)

    def test_trajectory_normalization_gives_every_trajectory_the_same_weight(self) -> None:
        short = [_row([1, 2, 3], [4], 1.0)]
        long = [_row([2, 3], [5, 6], -1.0), _row([7, 3], [8, 9], -1.0), _row([0, 3], [10, 11], -1.0)]
        for row in long:
            row.sample_index = 1
        rows = short + long
        for group in (short, long):
            total = sum(row.completion_tokens for row in group)
            for row in group:
                row.trajectory_tokens = total
        expected = torch.zeros(12, 12)
        for group in (short, long):
            for row in group:
                single = _TinyLM(seed=4)
                loss = -(float(row.advantage) * completion_logprobs(single, row)).sum() / (row.trajectory_tokens * 2)
                loss.backward()
                expected += single.table.grad
        model = _TinyLM(seed=4)
        recorder = _GradRecorder(model)
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1, max_grad_norm=1e9, loss_normalization="trajectory")
        report = policy_gradient_update(model, rows, settings, optimizer=recorder)
        self.assertTrue(torch.allclose(recorder.grad, expected, atol=1e-6))
        self.assertEqual((report.rows, report.completion_tokens), (4, 7))
        # Under the token normalization the long trajectory carries six of the seven tokens.
        dapo = _GradRecorder(_TinyLM(seed=4))
        policy_gradient_update(dapo.model, rows, UpdateSettings(learning_rate=0.01, optimizer_steps=1, max_grad_norm=1e9),
                               optimizer=dapo)
        self.assertFalse(torch.allclose(dapo.grad, expected, atol=1e-4))

    def test_optimizer_steps_keep_whole_trajectories_together_and_balanced(self) -> None:
        from psse_env.rl.update import _chunks

        rows = [_row([1, 2, 3], [4], 1.0, 0)]
        for prompt, completion in (([2, 3], [5, 6]), ([7, 3], [8, 9]), ([0, 3], [10, 11])):
            rows.append(_row(prompt, completion, -1.0, 1))
        rows += [_row([4, 5], [6], 0.5, 2), _row([5, 6], [7], -0.5, 3), _row([6, 7], [8], -0.5, 3)]
        chunks = _chunks(rows, UpdateSettings(optimizer_steps=2, seed=3))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sorted(index for chunk in chunks for index in chunk), list(range(len(rows))))
        for chunk in chunks:
            keys = {rows[index].trajectory_key for index in chunk}
            self.assertEqual(len(keys), 2)  # four trajectories dealt two per step
            for index in chunk:  # every row of a trajectory sits in the chunk that holds it
                self.assertTrue(all(rows[other].trajectory_key != rows[index].trajectory_key
                                    for other_chunk in chunks if other_chunk is not chunk for other in other_chunk))
        # Under the trajectory normalization the last step weighs each of its trajectories 1/2,
        # whatever their lengths (rows in one step cannot be split across steps any more).
        for row in rows:
            row.trajectory_tokens = sum(other.completion_tokens for other in rows if other.trajectory_key == row.trajectory_key)
        last = chunks[-1]
        expected = torch.zeros(12, 12)
        for index in last:
            single = _TinyLM(seed=4)
            loss = -(float(rows[index].advantage) * completion_logprobs(single, rows[index])).sum() / (rows[index].trajectory_tokens * 2)
            loss.backward()
            expected += single.table.grad
        recorder = _GradRecorder(_TinyLM(seed=4))
        policy_gradient_update(recorder.model, rows, UpdateSettings(optimizer_steps=2, seed=3, max_grad_norm=1e9,
                                                                    loss_normalization="trajectory"), optimizer=recorder)
        self.assertTrue(torch.allclose(recorder.grad, expected, atol=1e-6))

    def test_preservation_rows_enter_the_kl_term_but_not_the_policy_normalization(self) -> None:
        def gradient(rows, normalization):
            recorder = _GradRecorder(_TinyLM(seed=4))
            settings = UpdateSettings(optimizer_steps=1, max_grad_norm=1e9, loss_normalization=normalization)
            report = policy_gradient_update(recorder.model, rows, settings, optimizer=recorder)
            return recorder.grad, report

        for normalization in ("dapo", "trajectory", "sequence"):
            policy = _rows_after_token_3()
            kept = _rows_after_token_3() + [_row([9, 3], [10, 11, 4], 0.0, 9, preservation=True)]
            for rows in (policy, kept):
                for row in rows:
                    row.trajectory_tokens = row.completion_tokens
            alone, _ = gradient(policy, normalization)
            with_preservation, report = gradient(kept, normalization)
            self.assertTrue(torch.allclose(alone, with_preservation, atol=1e-6), normalization)
            self.assertEqual((report.policy_rows, report.preservation_rows), (6, 1))
        # With a KL penalty the preservation row is regularized: a reference that disagrees with the
        # model on that row alone moves the parameters, which a zero-advantage row never could.
        reference = _TinyLM(seed=5)
        row = _row([9, 3], [10, 11, 4], 0.0, 9, preservation=True)
        row.trajectory_tokens = row.completion_tokens
        recorder = _GradRecorder(_TinyLM(seed=4))
        policy_gradient_update(recorder.model, [row], UpdateSettings(optimizer_steps=1, max_grad_norm=1e9, beta_kl=1.0),
                               optimizer=recorder, reference_logprobs=lambda r: completion_logprobs(reference, r).detach())
        self.assertFalse(torch.equal(recorder.grad, torch.zeros(12, 12)))

    def test_the_training_pass_clip_baseline_makes_the_first_step_exact_and_survives_a_resume(self) -> None:
        def gradient(rows, settings):
            recorder = _GradRecorder(_TinyLM(seed=4))
            report = policy_gradient_update(recorder.model, rows, settings, optimizer=recorder)
            return recorder.grad, report

        plain = UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e9)
        against_sampler = UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e9, clip_ratio=0.2)
        against_training = UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e9, clip_ratio=0.2,
                                          clip_reference="training")
        # Records off by a factor of 1.5 (a numerical gap between the sampling and training passes):
        # against the sampler the first step is clipped, against the training pass it is the plain step.
        shifted = _with_record(_TinyLM(seed=4), _rows_after_token_3(), shift=-math.log(1.5))
        expected, _ = gradient(_with_record(_TinyLM(seed=4), _rows_after_token_3()), plain)
        clipped, _ = gradient(shifted, against_sampler)
        exact, report = gradient(shifted, against_training)
        self.assertFalse(torch.allclose(clipped, expected, atol=1e-6))
        self.assertTrue(torch.allclose(exact, expected, atol=1e-6))
        self.assertEqual(report.clip_check[0]["tokens_beyond_threshold"], 0)
        self.assertLess(report.clip_check[0]["mean_abs_log_ratio"], 1e-6)
        self.assertGreater(report.on_policy_check[0]["mean_abs_log_ratio"], 0.4)  # the sampler gap is still reported
        self.assertAlmostEqual(report.clip_check[0]["threshold_log_ratio"], math.log(1.2))
        # Resume: the baseline is persisted before the first step, so the second step clips against it.
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, seed=7, clip_ratio=0.2, clip_reference="training")
        reference = _TinyLM(seed=3)
        wanted = policy_gradient_update(reference, _with_record(_TinyLM(seed=3), _rows()), settings)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            # Six baseline forwards come first; the run then completes step one (three rows), saves two rows of
            # step two and is preempted on the third, as in the plain resume test.
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(seed=3, fail_after=11), _with_record(_TinyLM(seed=3), _rows()), settings,
                                       checkpoint=checkpoint)
            self.assertEqual(checkpoint.load_state()["old_logprobs_file"], "old_logprobs.pt")
            self.assertEqual((checkpoint.load_state()["completed_chunks"], checkpoint.load_state()["chunk_rows_done"]), (1, 2))
            resumed_model = _load_tiny(checkpoint.model_dir(), seed=3)
            resumed = policy_gradient_update(resumed_model, _with_record(_TinyLM(seed=3), _rows()), settings, checkpoint=checkpoint)
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-6))
            self.assertEqual(len(resumed.clip_check), 2)
            for got, want in zip(resumed.clip_check, wanted.clip_check):
                self.assertAlmostEqual(got["mean_abs_log_ratio"], want["mean_abs_log_ratio"], places=5)
        with self.assertRaises(ValueError):
            UpdateSettings(clip_reference="bogus").validate()

    def test_a_first_step_off_policy_beyond_the_tolerance_is_refused_before_it_is_applied(self) -> None:
        model = _TinyLM(seed=2)
        rows = _with_record(model, _rows_after_token_3(), shift=0.5)
        recorder = _GradRecorder(model)
        with self.assertRaisesRegex(RuntimeError, "not on policy"):
            policy_gradient_update(model, rows, UpdateSettings(optimizer_steps=1, max_on_policy_log_ratio=0.1), optimizer=recorder)
        self.assertIsNone(recorder.grad)
        exact = _with_record(_TinyLM(seed=2), _rows_after_token_3())
        report = policy_gradient_update(_TinyLM(seed=2), exact, UpdateSettings(optimizer_steps=1, max_on_policy_log_ratio=0.1))
        self.assertLess(report.on_policy_check[0]["mean_abs_log_ratio"], 1e-5)
        self.assertGreater(report.on_policy_check[0]["uncertain_tokens"], 0)
        self.assertLess(report.on_policy_check[0]["uncertain_mean_abs_log_ratio"], 1e-5)
        with self.assertRaisesRegex(RuntimeError, "without a record"):
            missing = _rows_after_token_3()
            policy_gradient_update(_TinyLM(seed=2), missing, UpdateSettings(optimizer_steps=1, max_on_policy_log_ratio=0.1))

    def test_settings_validation(self) -> None:
        for bad in (dict(learning_rate=0.0), dict(beta_kl=-1.0), dict(loss_normalization="token"), dict(temperature=0.0),
                    dict(max_row_grad_norm=0.0), dict(row_grad_norm_factor=-1.0), dict(row_grad_norm_warmup=0),
                    dict(clip_ratio=0.0), dict(clip_ratio=1.0), dict(clip_ratio=float("nan")),
                    dict(max_on_policy_log_ratio=0.0), dict(max_on_policy_log_ratio=float("inf"))):
            with self.assertRaises(ValueError):
                UpdateSettings(**bad).validate()
        with self.assertRaises(ValueError):
            policy_gradient_update(_TinyLM(), [], UpdateSettings())
        with self.assertRaises(ValueError):
            UpdateCheckpoint(directory=Path("."), save_model=lambda m, d: None, every_rows=0)


if __name__ == "__main__":
    unittest.main()
