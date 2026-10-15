from __future__ import annotations

import contextlib
import math
import unittest

from psse_env.rl.rows import build_step_row
from psse_env.rl.update import UpdateSettings, completion_logprobs, policy_gradient_update

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for the pure-Python tests
    torch = None


if torch is not None:

    class _Outputs:
        def __init__(self, logits):
            self.logits = logits

    class _TinyLM(torch.nn.Module):
        """A bigram model: logits depend on the previous token only."""

        vocab = 12

        def __init__(self, seed: int = 0) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.table = torch.nn.Parameter(torch.randn(self.vocab, self.vocab, generator=generator) * 0.1)
            self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.adapter_enabled = True
            self.calls: list[int | None] = []

        @property
        def device(self):
            return self.table.device

        def forward(self, input_ids, attention_mask=None, logits_to_keep=None, use_cache=False, **side):
            self.calls.append(logits_to_keep)
            table = self.table if self.adapter_enabled else self.table.detach() * 0.0
            logits = table[input_ids]
            if logits_to_keep:
                logits = logits[:, -int(logits_to_keep) :, :]
            return _Outputs(logits)

        @contextlib.contextmanager
        def disable_adapter(self):
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = True

    class _NoKeepLM(_TinyLM):
        def forward(self, input_ids, attention_mask=None, use_cache=False, **side):  # no logits_to_keep
            self.calls.append(None)
            return _Outputs(self.table[input_ids])


def _row(prompt, completion, advantage):
    return build_step_row(root="r", sample_index=0, step=0, advantage=advantage, prompt_ids=prompt,
                          completion_ids=completion, max_length=64)


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

    def test_falls_back_when_the_model_has_no_logits_to_keep(self) -> None:
        model = _NoKeepLM()
        row = _row([1, 2, 3], [4], 1.0)
        logprobs = completion_logprobs(model, row, use_logits_to_keep=True)
        self.assertEqual(tuple(logprobs.shape), (1,))

    def test_positive_advantage_raises_the_sampled_likelihood_and_negative_lowers_it(self) -> None:
        for advantage in (1.0, -1.0):
            model = _TinyLM()
            row = _row([1, 2, 3], [4, 5], advantage)
            before = completion_logprobs(model, row).sum().item()
            settings = UpdateSettings(learning_rate=0.1, beta_kl=0.0, optimizer_steps=1)
            report = policy_gradient_update(model, [row], settings)
            after = completion_logprobs(model, row).sum().item()
            if advantage > 0:
                self.assertGreater(after, before)
            else:
                self.assertLess(after, before)
            self.assertEqual(report.rows, 1)
            self.assertEqual(report.completion_tokens, 2)
            self.assertEqual(report.optimizer_steps, 1)
            self.assertTrue(math.isfinite(report.grad_norm[0]))
            self.assertEqual(report.kl, [0.0])

    def test_zero_advantage_rows_leave_the_parameters_unchanged(self) -> None:
        model = _TinyLM()
        before = model.table.detach().clone()
        report = policy_gradient_update(model, [_row([1, 2], [3], 0.0)], UpdateSettings(learning_rate=0.1, beta_kl=0.0))
        self.assertTrue(torch.equal(before, model.table.detach()))
        self.assertEqual(report.grad_norm, [0.0])

    def test_kl_penalty_uses_the_reference_pass_and_the_k3_estimator(self) -> None:
        model = _TinyLM()
        row = _row([1, 2, 3], [4], 0.0)
        settings = UpdateSettings(learning_rate=0.05, beta_kl=1.0, optimizer_steps=1)
        report = policy_gradient_update(model, [row], settings)
        self.assertEqual(len(report.kl), 1)
        self.assertGreaterEqual(report.kl[0], 0.0)
        self.assertGreater(report.grad_norm[0], 0.0)
        stub = _NoKeepLM()

        def reference(_row):
            return torch.zeros(1)

        report = policy_gradient_update(stub, [row], settings, reference_logprobs=reference)
        self.assertGreater(report.kl[0], 0.0)

    def test_optimizer_steps_partition_the_rows(self) -> None:
        model = _TinyLM()
        rows = [_row([1, 2], [3], 1.0), _row([2, 3], [4], -1.0), _row([3, 4], [5], 0.5)]
        seen: list[int] = []
        report = policy_gradient_update(model, rows, UpdateSettings(learning_rate=0.01, beta_kl=0.0, optimizer_steps=2),
                                        progress=lambda record: seen.append(int(record["rows"])))
        self.assertEqual(report.optimizer_steps, 2)
        self.assertEqual(sorted(seen), [1, 2])
        self.assertAlmostEqual(report.mean_advantage, 0.5 / 3.0)

    def test_settings_validation(self) -> None:
        with self.assertRaises(ValueError):
            UpdateSettings(learning_rate=0.0).validate()
        with self.assertRaises(ValueError):
            UpdateSettings(beta_kl=-1.0).validate()
        with self.assertRaises(ValueError):
            UpdateSettings(loss_normalization="token").validate()
        with self.assertRaises(ValueError):
            policy_gradient_update(_TinyLM(), [], UpdateSettings())


if __name__ == "__main__":
    unittest.main()
