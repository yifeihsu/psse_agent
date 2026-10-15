"""The KL reference adapter on a real (tiny, randomly initialized) PEFT model.

Checks what the update relies on: the reference log-probabilities are the
starting adapter's, the switch leaves the trained adapter active and the set
of trainable parameters unchanged, and an update with the KL term moves only
the trained adapter.  Skips when transformers or peft are not installed.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from psse_env.rl.rows import build_step_row
from psse_env.rl.update import UpdateSettings, completion_logprobs, peft_reference_logprobs, policy_gradient_update

try:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
except Exception:  # pragma: no cover - optional dependencies
    torch = None


def _row(advantage: float = 1.0):
    return build_step_row(root="r", sample_index=0, step=0, advantage=advantage, prompt_ids=[1, 5, 9, 3, 7],
                          completion_ids=[11, 4, 2], max_length=32)


@unittest.skipIf(torch is None, "torch, transformers or peft is not installed")
class PeftReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        config = GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2)
        self.base = GPT2LMHeadModel(config)
        self.base.eval()
        self.lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], fan_in_fan_out=True,
                               init_lora_weights=False, task_type="CAUSAL_LM")
        self.directory = tempfile.TemporaryDirectory()
        self.reference_dir = Path(self.directory.name) / "reference"
        get_peft_model(copy.deepcopy(self.base), self.lora).save_pretrained(str(self.reference_dir))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _trained_model(self):
        model = PeftModel.from_pretrained(copy.deepcopy(self.base), str(self.reference_dir), is_trainable=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.add_(0.5 * torch.randn_like(parameter))
        model.load_adapter(str(self.reference_dir), adapter_name="reference", is_trainable=False)
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                parameter.requires_grad_(False)
        return model

    def test_reference_log_probabilities_are_the_starting_adapters(self) -> None:
        model = self._trained_model()
        trainable_before = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable_before)
        self.assertTrue(all(".default." in name for name in trainable_before))
        reference = peft_reference_logprobs(model)(_row())
        standalone = PeftModel.from_pretrained(copy.deepcopy(self.base), str(self.reference_dir))
        expected = completion_logprobs(standalone, _row()).detach()
        self.assertTrue(torch.allclose(reference, expected, atol=1e-5))
        current = completion_logprobs(model, _row()).detach()
        self.assertFalse(torch.allclose(reference, current, atol=1e-4))
        self.assertEqual(model.active_adapter, "default")
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad}, trainable_before)

    def test_an_update_with_the_kl_term_moves_only_the_trained_adapter(self) -> None:
        model = self._trained_model()
        snapshot = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}
        report = policy_gradient_update(model, [_row(1.0), _row(-0.5)], UpdateSettings(learning_rate=0.05, beta_kl=0.5))
        self.assertGreater(report.kl[0], 0.0)
        moved = {name for name, p in model.named_parameters() if "lora_" in name and not torch.equal(p.detach(), snapshot[name])}
        self.assertTrue(moved)
        self.assertTrue(all(".default." in name for name in moved))

    def test_a_model_without_the_reference_adapter_is_refused(self) -> None:
        model = PeftModel.from_pretrained(copy.deepcopy(self.base), str(self.reference_dir), is_trainable=True)
        with self.assertRaisesRegex(ValueError, "starting adapter"):
            peft_reference_logprobs(model)


if __name__ == "__main__":
    unittest.main()
