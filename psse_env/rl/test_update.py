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
    build_optimizer,
    completion_logprobs,
    load_optimizer_state,
    optimizer_steps_taken,
    policy_gradient_update,
    save_optimizer_state,
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


def _group_row(root, sample, prompt, completion, advantage, step=0, preservation=False, supervised=False):
    """One row of the group ``root``: the samples of a root are one group, and their advantages sum to zero."""
    return build_step_row(root=root, sample_index=sample, step=step, advantage=advantage, prompt_ids=prompt,
                          completion_ids=completion, max_length=64, preservation=preservation,
                          lifecycle_term=0.25 if supervised else 0.0, supervised=supervised)


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

    def test_lifecycle_rows_train_their_term_apart_from_the_policy_term(self) -> None:
        for term in (0.5, -0.5):
            model = _TinyLM()
            row = build_step_row(root="r", sample_index=0, step=0, advantage=0.0, prompt_ids=[1, 2, 3], completion_ids=[4, 5],
                                 max_length=64, lifecycle_term=term, lifecycle_only=True)
            before = completion_logprobs(model, row).sum().item()
            report = policy_gradient_update(model, [row], UpdateSettings(learning_rate=0.1, optimizer_steps=1))
            after = completion_logprobs(model, row).sum().item()
            if term > 0:
                self.assertGreater(after, before)
            else:
                self.assertLess(after, before)
            self.assertEqual((report.policy_rows, report.lifecycle_rows, report.preservation_rows, len(report.lifecycle_loss)), (0, 1, 0, 1))
            # The term's loss: the mean over the row's two tokens of -term * logprob, one lifecycle row in the step.
            self.assertAlmostEqual(report.lifecycle_loss[0], -term * before / 2, places=5)
            self.assertEqual(report.policy_loss, [0.0])

    def test_lifecycle_and_policy_terms_add_up_under_their_own_normalizations(self) -> None:
        def gradient(rows):
            model = _TinyLM()
            recorder = _GradRecorder(model)
            policy_gradient_update(model, rows, UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e6), optimizer=recorder)
            return recorder.grad

        policy = _row([1, 2, 3], [4, 5], 1.0, 0)
        lifecycle = build_step_row(root="r", sample_index=1, step=0, advantage=0.0, prompt_ids=[2, 3], completion_ids=[6, 7, 8],
                                   max_length=64, lifecycle_term=-0.25, lifecycle_only=True)
        riding = build_step_row(root="r", sample_index=2, step=0, advantage=0.5, prompt_ids=[3, 4], completion_ids=[9],
                                max_length=64, lifecycle_term=0.25)
        both = gradient([policy, lifecycle])
        self.assertTrue(torch.allclose(both, gradient([policy]) + gradient([lifecycle]), atol=1e-6))
        self.assertFalse(torch.allclose(both, gradient([policy]), atol=1e-6))
        # A term riding on a policy row adds the lifecycle gradient of that row to its policy gradient.
        plain = build_step_row(root="r", sample_index=2, step=0, advantage=0.5, prompt_ids=[3, 4], completion_ids=[9], max_length=64)
        only_term = build_step_row(root="r", sample_index=2, step=0, advantage=0.0, prompt_ids=[3, 4], completion_ids=[9], max_length=64,
                                   lifecycle_term=0.25, lifecycle_only=True)
        self.assertTrue(torch.allclose(gradient([riding]), gradient([plain]) + gradient([only_term]), atol=1e-6))

    def test_supervised_rows_train_the_audits_choice_outside_every_other_term(self) -> None:
        model = _TinyLM()
        target = build_step_row(root="r", sample_index=0, step=0, advantage=0.0, prompt_ids=[1, 2, 3], completion_ids=[4, 5],
                                max_length=64, lifecycle_term=0.5, supervised=True)
        before = completion_logprobs(model, target).sum().item()
        report = policy_gradient_update(model, [target], UpdateSettings(learning_rate=0.1, optimizer_steps=1))
        after = completion_logprobs(model, target).sum().item()
        self.assertGreater(after, before)
        self.assertEqual((report.policy_rows, report.lifecycle_rows, report.supervised_rows, report.preservation_rows), (0, 0, 1, 0))
        self.assertAlmostEqual(report.supervised_loss[0], -0.5 * before / 2, places=5)
        self.assertEqual((report.policy_loss, report.lifecycle_loss), ([0.0], [0.0]))
        # The on-policy check and the clip ignore supervised rows (no record), the reference pass never sees them,
        # and the KL normalization counts the sampled rows only.
        calls = []

        def reference(row):
            calls.append(row.sample_index)
            return completion_logprobs(_TinyLM(seed=1), row).detach()

        model = _TinyLM()
        policy = _with_record(model, [_row([1, 2, 3], [4, 5], 1.0, 0)])[0]
        supervised = build_step_row(root="r", sample_index=1, step=0, advantage=0.0, prompt_ids=[2, 3], completion_ids=[6, 7, 8],
                                    max_length=64, lifecycle_term=0.25, supervised=True)
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=1, beta_kl=0.1, clip_ratio=0.2, clip_reference="training",
                                  max_on_policy_log_ratio=0.5)
        report = policy_gradient_update(model, [policy, supervised], settings, reference_logprobs=reference)
        self.assertEqual(calls, [0])
        self.assertEqual(report.on_policy_check[0]["rows_without_record"], 0)
        self.assertEqual(report.on_policy_check[0]["rows"], 1)
        self.assertEqual((report.policy_rows, report.supervised_rows), (1, 1))
        self.assertNotEqual(report.supervised_loss[0], 0.0)

    def test_supervised_and_policy_gradients_add_up(self) -> None:
        def gradient(rows):
            model = _TinyLM()
            recorder = _GradRecorder(model)
            policy_gradient_update(model, rows, UpdateSettings(learning_rate=0.1, optimizer_steps=1, max_grad_norm=1e6), optimizer=recorder)
            return recorder.grad

        policy = _row([1, 2, 3], [4, 5], 1.0, 0)
        supervised = build_step_row(root="r", sample_index=1, step=0, advantage=0.0, prompt_ids=[2, 3], completion_ids=[6, 7, 8],
                                    max_length=64, lifecycle_term=0.25, supervised=True)
        self.assertTrue(torch.allclose(gradient([policy, supervised]), gradient([policy]) + gradient([supervised]), atol=1e-6))
        # The supervised gradient is the cross-entropy gradient of the target (a unit-advantage row under the token
        # normalization is the mean negative log-likelihood too), scaled by the term.
        plain = build_step_row(root="r", sample_index=1, step=0, advantage=1.0, prompt_ids=[2, 3], completion_ids=[6, 7, 8], max_length=64)
        self.assertTrue(torch.allclose(gradient([supervised]), 0.25 * gradient([plain]), atol=1e-6))

    def test_checkpoint_state_records_the_lifecycle_losses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyLM()
            row = build_step_row(root="r", sample_index=0, step=0, advantage=0.0, prompt_ids=[1, 2], completion_ids=[3, 4],
                                 max_length=64, lifecycle_term=0.5, lifecycle_only=True)
            checkpoint = UpdateCheckpoint(directory=Path(directory) / "ckpt", save_model=_save_tiny, every_rows=4)
            report = policy_gradient_update(model, [row], UpdateSettings(learning_rate=0.1), checkpoint=checkpoint)
            state = checkpoint.load_state()
            self.assertEqual(state["lifecycle_losses"], report.lifecycle_loss)
            self.assertEqual(state["partial_lifecycle"], 0.0)
            self.assertEqual((state["supervised_losses"], state["partial_supervised"]), ([0.0], 0.0))

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

    def test_a_training_pass_preempted_during_the_baseline_resumes_from_its_partial_rows(self) -> None:
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, seed=7, clip_ratio=0.2, clip_reference="training")
        reference = _TinyLM(seed=3)
        wanted = policy_gradient_update(reference, _with_record(_TinyLM(seed=3), _rows()), settings)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=2)
            # The baseline pass persists its rows two at a time; the third row's forward is preempted, so
            # two baseline rows survive and no optimizer step has been applied.
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(seed=3, fail_after=2), _with_record(_TinyLM(seed=3), _rows()), settings,
                                       checkpoint=checkpoint)
            self.assertTrue((Path(directory) / "old_logprobs.partial.pt").is_file())
            self.assertIsNone(checkpoint.load_state())
            self.assertIsNone(checkpoint.model_dir())
            # The model has not moved, so the next incarnation loads the starting weights and continues the pass.
            resumed_model = _TinyLM(seed=3)
            resumed = policy_gradient_update(resumed_model, _with_record(_TinyLM(seed=3), _rows()), settings,
                                             checkpoint=checkpoint)
            self.assertEqual(resumed.resumed_from, {"completed_chunks": 0, "chunk_rows_done": 0, "baseline_rows_done": 2})
            self.assertFalse((Path(directory) / "old_logprobs.partial.pt").exists())
            self.assertEqual(checkpoint.load_state()["old_logprobs_file"], "old_logprobs.pt")
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-6))
            for got, want in zip(resumed.clip_check, wanted.clip_check):
                self.assertAlmostEqual(got["mean_abs_log_ratio"], want["mean_abs_log_ratio"], places=5)
            # The partial file of another batch is refused rather than reused.
            other = UpdateCheckpoint(directory=Path(directory) / "other", save_model=_save_tiny, every_rows=2)
            with self.assertRaises(_Preempted):
                policy_gradient_update(_TinyLM(seed=3, fail_after=2), _with_record(_TinyLM(seed=3), _rows()), settings,
                                       checkpoint=other)
            with self.assertRaises(RuntimeError):
                policy_gradient_update(_TinyLM(seed=3), _with_record(_TinyLM(seed=3), _rows()[:5]), settings,
                                       checkpoint=other)

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

    def test_group_chunking_keeps_every_group_in_one_step(self) -> None:
        from psse_env.rl.update import _chunks, _split_groups

        rows = []
        for number in range(6):  # informative groups: two samples of two steps, plus a labelled lifecycle decision
            root = f"informative{number}"
            for sample, advantage in ((0, 1.0), (1, -1.0)):
                rows.append(_group_row(root, sample, [1, 2], [3], advantage, step=0))
                rows.append(_group_row(root, sample, [2, 3 + sample], [5, 6], advantage, step=1))
            rows.append(_group_row(root, 0, [3, 4], [7], 0.0, step=1, supervised=True))
        for number in range(5):  # flat groups: one preservation row and one supervised row
            root = f"flat{number}"
            rows.append(_group_row(root, 0, [4, 5], [8], 0.0, preservation=True))
            rows.append(_group_row(root, 1, [5, 6], [9], 0.0, supervised=True))
        for seed in range(5):
            chunks = _chunks(rows, UpdateSettings(optimizer_steps=2, seed=seed, chunk_by="group"))
            self.assertEqual(sorted(index for chunk in chunks for index in chunk), list(range(len(rows))))
            roots = [{rows[index].root for index in chunk} for chunk in chunks]
            self.assertEqual(len(roots), 2)
            self.assertFalse(roots[0] & roots[1])  # no group has rows in both steps
            self.assertEqual([sum(1 for root in step if root.startswith("informative")) for step in roots], [3, 3])
            self.assertEqual(sorted(sum(1 for root in step if root.startswith("flat")) for step in roots), [2, 3])
            self.assertEqual(_split_groups(rows, chunks), 0)
            # Dealt by trajectory, the same batch has groups with a sample in each step.
            self.assertGreater(_split_groups(rows, _chunks(rows, UpdateSettings(optimizer_steps=2, seed=seed))), 0)

    def test_split_groups_counts_the_groups_whose_policy_rows_sit_in_more_than_one_step(self) -> None:
        from psse_env.rl.update import _split_groups

        rows = [_group_row("a", 0, [1, 2], [3], 1.0), _group_row("a", 1, [2, 3], [4], -1.0),
                _group_row("b", 0, [3, 4], [5], 0.0, preservation=True), _group_row("b", 1, [4, 5], [6], 0.0, supervised=True),
                _group_row("c", 0, [5, 6], [7], 1.0), _group_row("c", 1, [6, 7], [8], 0.0, supervised=True)]
        self.assertEqual([row.policy_row for row in rows], [True, True, False, False, True, False])
        # Group a's two policy rows part: the one case in which a group's advantages no longer cancel within a step.
        self.assertEqual(_split_groups(rows, [[0, 2, 4], [1, 3, 5]]), 1)
        # Group a whole; b and c are parted only by rows that carry no group advantage.
        self.assertEqual(_split_groups(rows, [[0, 1, 2, 4], [3, 5]]), 0)
        self.assertEqual(_split_groups(rows, [[0, 1, 2, 3, 4, 5]]), 0)

    def test_a_batch_whose_gradient_is_zero_leaves_the_weights_in_place(self) -> None:
        from psse_env.rl.update import _chunks, _split_groups

        # Each group holds the same row twice with opposite advantages, so each group's gradient is zero.
        rows = []
        for root, prompt, completion in (("a", [1, 2], [3, 4]), ("b", [5, 6], [7, 8])):
            rows += [_group_row(root, 0, prompt, completion, 1.0), _group_row(root, 1, prompt, completion, -1.0)]
        # The seeds under which the deal by trajectory puts a group's two samples in different steps:
        # on those the two-step update stays in place only because whole groups are dealt.
        splitting = [seed for seed in range(8)
                     if _split_groups(rows, _chunks(rows, UpdateSettings(optimizer_steps=2, seed=seed)))]
        self.assertGreaterEqual(len(splitting), 3)
        arms = [UpdateSettings(learning_rate=0.01, optimizer_steps=2, chunk_by="group", seed=seed) for seed in splitting]
        arms += [UpdateSettings(learning_rate=0.01, optimizer_steps=2, chunk_by="group"),
                 UpdateSettings(learning_rate=0.01, optimizer_steps=1)]
        for settings in arms:
            model = _TinyLM()
            before = model.table.detach().clone()
            report = policy_gradient_update(model, rows, settings)
            self.assertTrue(torch.equal(before, model.table.detach()))
            self.assertEqual(report.split_groups, 0)
            self.assertEqual(report.grad_norm, [0.0] * settings.optimizer_steps)

    def test_a_group_dealt_across_two_steps_moves_weights_its_gradient_leaves_alone(self) -> None:
        # The defect of the run of 2026-11-08: one group, the same row twice with opposite advantages.
        # Its gradient is zero, but dealt by trajectory each step sees one sample, and a new AdamW
        # takes the learning rate in the first step and gives back about a twentieth in the second.
        rate = 0.01
        rows = [_group_row("a", 0, [1, 2], [3, 4], 1.0), _group_row("a", 1, [1, 2], [3, 4], -1.0)]
        model = _TinyLM()
        before = model.table.detach().clone()
        report = policy_gradient_update(model, rows, UpdateSettings(learning_rate=rate, optimizer_steps=2))
        self.assertEqual(report.split_groups, 1)
        moved = (model.table.detach() - before).abs()
        touched = [2, 3]  # the completion's tokens are predicted from tokens 2 and 3
        others = [index for index in range(_TinyLM.vocab) if index not in touched]
        self.assertTrue(torch.equal(moved[others], torch.zeros_like(moved[others])))
        self.assertTrue(bool((moved[touched] > 0.93 * rate).all()) and bool((moved[touched] < 0.96 * rate).all()))

    def test_what_a_groups_samples_share_cancels_only_when_the_group_is_in_one_step(self) -> None:
        from psse_env.rl.update import _chunks, _split_groups

        rate = 0.01
        rows = []
        # Each group's two samples share their first step (predicted from token 2 in group a, 8 in group b)
        # and part on the second; the advantages sum to zero within the group.
        for root, shared, seconds in (("a", ([1, 2], [3]), (([0, 4], [5]), ([0, 6], [7]))),
                                      ("b", ([1, 8], [9]), (([0, 10], [11]), ([0, 11], [0])))):
            for sample, advantage in ((0, 1.0), (1, -1.0)):
                rows.append(_group_row(root, sample, shared[0], shared[1], advantage, step=0))
                rows.append(_group_row(root, sample, seconds[sample][0], seconds[sample][1], advantage, step=1))
        shared_rows, own_rows = [2, 8], [4, 6, 10, 11]
        # A seed under which the deal by trajectory splits both groups: both deals run on it.
        split = next(settings for settings in (UpdateSettings(learning_rate=rate, optimizer_steps=2, seed=seed) for seed in range(10))
                     if _split_groups(rows, _chunks(rows, settings)) == 2)
        whole = UpdateSettings(learning_rate=rate, optimizer_steps=2, seed=split.seed, chunk_by="group")
        model = _TinyLM()
        before = model.table.detach().clone()
        report = policy_gradient_update(model, rows, whole)
        moved = (model.table.detach() - before).abs()
        self.assertEqual(report.split_groups, 0)
        self.assertTrue(torch.equal(moved[shared_rows], torch.zeros_like(moved[shared_rows])))
        self.assertTrue(bool((moved[own_rows] > 0.5 * rate).all()))
        # Dealt by trajectory, the shared step's weights move almost the learning rate.
        model = _TinyLM()
        report = policy_gradient_update(model, rows, split)
        moved = (model.table.detach() - before).abs()
        self.assertEqual(report.split_groups, 2)
        self.assertTrue(bool((moved[shared_rows] > 0.93 * rate).all()) and bool((moved[shared_rows] < 0.96 * rate).all()))

    def test_a_carried_optimizer_moves_a_weight_by_how_steadily_its_gradient_points(self) -> None:
        rate = 0.01
        settings = UpdateSettings(learning_rate=rate, optimizer_steps=1)
        touched = [2, 5]

        def batch(sign: float):
            return [_group_row("a", 0, [1, 2], [3], sign), _group_row("a", 1, [4, 5], [6], -sign)]

        def twin(source):
            other = _TinyLM()
            other.load_state_dict(source.state_dict())
            return other

        model = _TinyLM()
        start = model.table.detach().clone()
        optimizer = build_optimizer(model, settings)
        self.assertEqual(optimizer_steps_taken(optimizer.state_dict()), 0)
        policy_gradient_update(model, batch(1.0), settings, optimizer=optimizer)
        first = (model.table.detach() - start)[touched]
        self.assertTrue(torch.allclose(first.abs(), torch.full_like(first, rate), rtol=1e-4))
        with tempfile.TemporaryDirectory() as directory:
            save_optimizer_state(optimizer, Path(directory) / "optimizer.pt")
            carried = load_optimizer_state(Path(directory) / "optimizer.pt")
        self.assertEqual(optimizer_steps_taken(carried), 1)
        after_first = model.table.detach().clone()

        def step(rows, state):
            other = twin(model)
            continued = build_optimizer(other, settings, state)
            policy_gradient_update(other, rows, settings, optimizer=continued)
            return (other.table.detach() - after_first)[touched], optimizer_steps_taken(continued.state_dict())

        # The same gradient again: the full learning rate, the same way.
        again, steps = step(batch(1.0), carried)
        self.assertEqual(steps, 2)
        self.assertTrue(torch.allclose(again, first, rtol=0.02))
        # The opposite gradient: about a twentieth of it (0.0526), back the way the first step came.
        back, steps = step(batch(-1.0), carried)
        self.assertEqual(steps, 2)
        self.assertTrue(bool((back * first < 0).all()))
        self.assertTrue(bool((back.abs() > 0.03 * rate).all()) and bool((back.abs() < 0.08 * rate).all()))
        # A new optimizer on that opposite gradient takes the full learning rate back: the sign step.
        fresh, steps = step(batch(-1.0), None)
        self.assertEqual(steps, 1)
        self.assertTrue(torch.allclose(fresh, -first, rtol=1e-4))
        default = twin(model)
        policy_gradient_update(default, batch(-1.0), settings)
        self.assertTrue(torch.equal((default.table.detach() - after_first)[touched], fresh))

    def test_a_carried_optimizer_takes_the_settings_in_force(self) -> None:
        model = _TinyLM()
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1)
        optimizer = build_optimizer(model, settings)
        policy_gradient_update(model, [_row([1, 2], [3], 1.0)], settings, optimizer=optimizer)
        changed = UpdateSettings(learning_rate=0.002, optimizer_steps=1, weight_decay=0.01, adam_beta1=0.8,
                                 adam_beta2=0.99, adam_epsilon=1e-6)
        continued = build_optimizer(model, changed, optimizer.state_dict())
        group = continued.param_groups[0]
        self.assertEqual((group["lr"], group["weight_decay"], group["betas"], group["eps"]), (0.002, 0.01, (0.8, 0.99), 1e-6))
        self.assertEqual(optimizer_steps_taken(continued.state_dict()), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_building_from_a_live_state_leaves_that_state_as_it_was(self) -> None:
        model = _TinyLM(seed=1)
        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1)
        batch = [_row([1, 2], [3], 1.0)]
        first = build_optimizer(model, settings)
        policy_gradient_update(model, batch, settings, optimizer=first)
        moment = first.state_dict()["state"][0]["exp_avg"].clone()
        second = build_optimizer(model, settings, first.state_dict())
        policy_gradient_update(model, batch, settings, optimizer=second)
        self.assertEqual(optimizer_steps_taken(first.state_dict()), 1)
        self.assertTrue(torch.equal(first.state_dict()["state"][0]["exp_avg"], moment))
        self.assertEqual(optimizer_steps_taken(second.state_dict()), 2)

    def test_a_zero_gradient_step_coasts_on_carried_momentum_unless_beta1_is_zero(self) -> None:
        rate = 0.01
        touched = [2, 5]
        others = [index for index in range(_TinyLM.vocab) if index not in touched]
        real = [_group_row("a", 0, [1, 2], [3], 1.0), _group_row("a", 1, [4, 5], [6], -1.0)]
        nothing = [_group_row("b", 0, [1, 2], [3], 1.0), _group_row("b", 1, [1, 2], [3], -1.0)]  # gradient zero
        for beta1 in (0.9, 0.0):
            settings = UpdateSettings(learning_rate=rate, optimizer_steps=1, adam_beta1=beta1)
            model = _TinyLM()
            optimizer = build_optimizer(model, settings)
            policy_gradient_update(model, real, settings, optimizer=optimizer)
            before = model.table.detach().clone()
            continued = build_optimizer(model, settings, optimizer.state_dict())
            report = policy_gradient_update(model, nothing, settings, optimizer=continued)
            moved = (model.table.detach() - before).abs()
            self.assertEqual(report.grad_norm, [0.0])
            self.assertTrue(torch.equal(moved[others], torch.zeros_like(moved[others])))
            if beta1:
                # 0.4737 / 0.7069 of the learning rate: the first step's momentum over its second moment.
                self.assertTrue(bool((moved[touched] > 0.66 * rate).all()) and bool((moved[touched] < 0.68 * rate).all()))
            else:
                self.assertTrue(torch.equal(moved[touched], torch.zeros_like(moved[touched])))

    def test_a_carried_state_for_other_parameters_is_refused(self) -> None:
        class _Wide(_TinyLM):
            vocab = 16

        class _WithBias(_TinyLM):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(self.vocab))

        settings = UpdateSettings(learning_rate=0.01, optimizer_steps=1)
        model = _TinyLM()
        optimizer = build_optimizer(model, settings)
        policy_gradient_update(model, [_row([1, 2], [3], 1.0)], settings, optimizer=optimizer)
        state = optimizer.state_dict()
        with self.assertRaisesRegex(RuntimeError, "other trainable parameters"):
            build_optimizer(_Wide(), settings, state)
        with self.assertRaisesRegex(RuntimeError, "1 trainable parameters, the model has 2"):
            build_optimizer(_WithBias(), settings, state)
        with self.assertRaisesRegex(RuntimeError, "not an optimizer state_dict"):
            build_optimizer(_TinyLM(), settings, {"weights": 1})
        build_optimizer(_TinyLM(), settings, state)  # the same parameters: accepted

    def test_a_checkpoint_dealt_by_the_other_chunking_is_refused(self) -> None:
        rows = [_group_row("a", 0, [1, 2], [3], 1.0), _group_row("a", 1, [2, 3], [4], -1.0),
                _group_row("b", 0, [3, 4], [5], 0.5), _group_row("b", 1, [4, 5], [6], -0.5)]
        by_trajectory = UpdateSettings(optimizer_steps=2)
        by_group = UpdateSettings(optimizer_steps=2, chunk_by="group")
        # Four rows in two steps under either deal, so the row and step counts alone would let the resume through.
        for written, resumed, message in ((by_group, by_trajectory, "dealt by 'group'"),
                                          (by_trajectory, by_group, "dealt by 'trajectory'")):
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = UpdateCheckpoint(directory=Path(directory), save_model=_save_tiny, every_rows=1)
                policy_gradient_update(_TinyLM(), rows, written, checkpoint=checkpoint)
                self.assertEqual(checkpoint.load_state().get("chunk_by"), "group" if written is by_group else None)
                with self.assertRaisesRegex(RuntimeError, message):
                    policy_gradient_update(_TinyLM(), rows, resumed, checkpoint=checkpoint)

    def test_an_interrupted_group_update_with_a_carried_optimizer_resumes_to_the_same_parameters(self) -> None:
        settings = UpdateSettings(learning_rate=0.05, optimizer_steps=2, seed=7, chunk_by="group")
        rows = []
        for root, (first, second), advantage in (("a", (([1, 2], [3]), ([2, 3], [4, 5])), 1.0),
                                                 ("b", (([3, 4], [5]), ([4, 5], [6])), 0.5),
                                                 ("c", (([5, 6], [7, 8]), ([6, 7], [8])), 1.5),
                                                 ("d", (([7, 8], [9]), ([8, 9], [10])), 1.0)):
            rows += [_group_row(root, 0, first[0], first[1], advantage), _group_row(root, 1, second[0], second[1], -advantage)]
        with tempfile.TemporaryDirectory() as directory:
            start, carried_file = Path(directory) / "start", Path(directory) / "optimizer.pt"
            # An earlier update: one step, whose optimizer state the next update continues.
            one_step = UpdateSettings(learning_rate=0.05, optimizer_steps=1)
            earlier = _TinyLM(seed=3)
            optimizer = build_optimizer(earlier, one_step)
            policy_gradient_update(earlier, _rows(), one_step, optimizer=optimizer)
            _save_tiny(earlier, start)
            save_optimizer_state(optimizer, carried_file)
            carried = load_optimizer_state(carried_file)

            reference = _load_tiny(start, seed=3)
            expected = policy_gradient_update(reference, rows, settings, optimizer=build_optimizer(reference, settings, carried))
            self.assertEqual(expected.split_groups, 0)

            checkpoint = UpdateCheckpoint(directory=Path(directory) / "checkpoint", save_model=_save_tiny, every_rows=2)
            # Two groups, four rows, per step: the run completes step one, saves the
            # gradients of two rows of step two, and is preempted on the third.
            interrupted = _load_tiny(start, seed=3, fail_after=6)
            with self.assertRaises(_Preempted):
                policy_gradient_update(interrupted, rows, settings, optimizer=build_optimizer(interrupted, settings, carried),
                                       checkpoint=checkpoint)
            state = checkpoint.load_state()
            self.assertEqual((state["completed_chunks"], state["chunk_rows_done"], state["chunk_by"]), (1, 2, "group"))
            # As the driver does: the optimizer is rebuilt from the state before this update,
            # and the resume loads the checkpoint's state, one step further, over it.
            resumed_model = _load_tiny(checkpoint.model_dir(), seed=3)
            resumed_optimizer = build_optimizer(resumed_model, settings, carried)
            resumed = policy_gradient_update(resumed_model, rows, settings, optimizer=resumed_optimizer, checkpoint=checkpoint)
            self.assertEqual(resumed.resumed_from, {"completed_chunks": 1, "chunk_rows_done": 2})
            self.assertTrue(torch.allclose(resumed_model.table, reference.table, atol=1e-6))
            self.assertEqual(optimizer_steps_taken(resumed_optimizer.state_dict()), 3)
            self.assertEqual(optimizer_steps_taken(carried), 1)
            # Without the carried state the same update ends elsewhere, so the state is what was continued.
            anew = _load_tiny(start, seed=3)
            policy_gradient_update(anew, rows, settings)
            self.assertFalse(torch.allclose(anew.table, reference.table, atol=1e-4))

    def test_settings_validation(self) -> None:
        for bad in (dict(learning_rate=0.0), dict(beta_kl=-1.0), dict(loss_normalization="token"), dict(temperature=0.0),
                    dict(max_row_grad_norm=0.0), dict(row_grad_norm_factor=-1.0), dict(row_grad_norm_warmup=0),
                    dict(clip_ratio=0.0), dict(clip_ratio=1.0), dict(clip_ratio=float("nan")),
                    dict(max_on_policy_log_ratio=0.0), dict(max_on_policy_log_ratio=float("inf")),
                    dict(chunk_by="root"), dict(carry_optimizer="yes")):
            with self.assertRaises(ValueError):
                UpdateSettings(**bad).validate()
        with self.assertRaises(ValueError):
            policy_gradient_update(_TinyLM(), [], UpdateSettings())
        with self.assertRaises(ValueError):
            UpdateCheckpoint(directory=Path("."), save_model=lambda m, d: None, every_rows=0)


if __name__ == "__main__":
    unittest.main()
