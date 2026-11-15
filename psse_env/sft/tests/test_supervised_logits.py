"""The loss from the logits of the supervised span alone is the model's own loss (``psse_env.sft.supervised_logits``).

A small causal model whose forward computes the loss the way the language models do (shifted logits, ``-100``
ignored, the mean over the supervised tokens) and takes ``logits_to_keep``.
"""
from __future__ import annotations

import inspect
import unittest

import torch
from transformers.modeling_outputs import CausalLMOutput

from psse_env.sft.research_cli import ResearchTrainerSettings, parser
from psse_env.sft.supervised_logits import (
    IGNORE_INDEX,
    keep_supervised_logits,
    span_token_accuracy,
    supervised_span_trainer,
    supervised_start,
)

VOCABULARY, HIDDEN = 37, 12


class _Causal(torch.nn.Module):
    """Position t sees tokens 0..t; the head projects every kept position onto the vocabulary."""

    def __init__(self, *, takes_span: bool = True, honours_span: bool = True) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(VOCABULARY, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCABULARY)
        self.takes_span, self.honours_span = takes_span, honours_span
        self.projected: list[int] = []

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if "logits_to_keep" in kwargs and not self.takes_span:
            raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
        keep = int(kwargs.get("logits_to_keep") or 0)
        hidden = torch.tanh(self.embedding(input_ids).cumsum(dim=1))
        if keep and self.honours_span:
            hidden = hidden[:, -keep:, :]
        logits = self.head(hidden)
        self.projected.append(int(logits.shape[1]))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, VOCABULARY), labels[:, 1:].reshape(-1),
                                                     ignore_index=IGNORE_INDEX)
        return CausalLMOutput(loss=loss, logits=logits)


def _batch(lengths=(40,), targets=(5,)):
    """Rows of a prompt and a target, right-padded as the collator pads them."""
    generator = torch.Generator().manual_seed(11)
    width = max(lengths)
    input_ids = torch.zeros((len(lengths), width), dtype=torch.long)
    attention = torch.zeros((len(lengths), width), dtype=torch.long)
    labels = torch.full((len(lengths), width), IGNORE_INDEX, dtype=torch.long)
    for row, (length, target) in enumerate(zip(lengths, targets)):
        tokens = torch.randint(1, VOCABULARY, (length,), generator=generator)
        input_ids[row, :length] = tokens
        attention[row, :length] = 1
        labels[row, length - target:length] = tokens[length - target:]
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def _gradients(model, batch):
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    output.loss.backward()
    return float(output.loss.detach()), [parameter.grad.clone() for parameter in model.parameters()], output


class SupervisedLogitsTests(unittest.TestCase):
    def test_the_supervised_span_starts_at_the_first_supervised_token_of_any_row(self) -> None:
        self.assertEqual(supervised_start(_batch((40,), (5,))["labels"]), 35)
        self.assertEqual(supervised_start(_batch((40, 28), (5, 6))["labels"]), 22)
        self.assertIsNone(supervised_start(torch.full((2, 9), IGNORE_INDEX)))

    def test_the_loss_and_its_gradients_are_the_models_own(self) -> None:
        for lengths, targets in (((40,), (5,)), ((40, 28, 33), (5, 6, 1)), ((12,), (11,))):
            with self.subTest(lengths=lengths):
                batch = _batch(lengths, targets)
                plain, wrapped = _Causal(), _Causal()
                counts = keep_supervised_logits(wrapped)
                loss, gradients, _ = _gradients(plain, batch)
                kept_loss, kept_gradients, output = _gradients(wrapped, batch)
                self.assertAlmostEqual(kept_loss, loss, places=5)
                for mine, theirs in zip(kept_gradients, gradients):
                    self.assertTrue(torch.allclose(mine, theirs, atol=1e-6))
                # Only the span was projected onto the vocabulary: one position before the first supervised token to the end.
                start = supervised_start(batch["labels"])
                keep = max(lengths) - start + 1
                self.assertEqual(wrapped.projected, [keep])
                self.assertEqual(plain.projected, [max(lengths)])
                self.assertEqual(tuple(output.logits.shape), (len(lengths), keep, VOCABULARY))
                self.assertEqual((counts["kept_calls"], counts["full_calls"], counts["ignored_calls"]), (1, 0, 0))
                self.assertEqual((counts["largest_kept"], counts["longest_sequence"]), (keep, max(lengths)))

    def test_an_output_reads_as_the_trainer_reads_it(self) -> None:
        wrapped = _Causal()
        keep_supervised_logits(wrapped)
        output = wrapped(**_batch())
        self.assertIsInstance(output, dict)           # the trainer takes outputs["loss"] of a mapping
        self.assertIs(output["loss"], output.loss)
        self.assertEqual(list(output.keys()), ["loss", "logits"])

    def test_calls_without_a_span_go_to_the_model_as_they_were(self) -> None:
        wrapped = _Causal()
        counts = keep_supervised_logits(wrapped)
        batch = _batch()
        generation = wrapped(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        self.assertIsNone(generation.loss)
        self.assertEqual(generation.logits.shape[1], 40)
        from_the_start = dict(batch, labels=batch["input_ids"].clone())
        self.assertEqual(wrapped(**from_the_start).logits.shape[1], 40)
        nothing = dict(batch, labels=torch.full_like(batch["labels"], IGNORE_INDEX))
        wrapped(**nothing)
        # A caller that names its own span keeps it (the RL update asks for the completion's logits, without labels).
        asked = wrapped(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=3)
        self.assertEqual(asked.logits.shape[1], 3)
        self.assertEqual((counts["full_calls"], counts["kept_calls"]), (4, 0))

    def test_a_forward_without_the_argument_is_called_as_before_and_counted(self) -> None:
        batch = _batch()
        plain, refusing = _Causal(), _Causal(takes_span=False)
        counts = keep_supervised_logits(refusing)
        self.assertAlmostEqual(float(refusing(**batch).loss), float(plain(**batch).loss), places=6)
        self.assertEqual((counts["refused_calls"], counts["full_calls"], counts["kept_calls"]), (1, 1, 0))

    def test_a_forward_that_ignores_the_argument_still_gives_the_right_loss_and_says_so(self) -> None:
        batch = _batch((40, 31), (5, 4))
        plain, ignoring = _Causal(), _Causal(honours_span=False)
        counts = keep_supervised_logits(ignoring)
        self.assertAlmostEqual(float(ignoring(**batch).loss), float(plain(**batch).loss), places=5)
        self.assertEqual((counts["ignored_calls"], counts["kept_calls"]), (1, 0))

    def test_the_signature_the_trainer_inspects_is_the_models(self) -> None:
        plain, wrapped = _Causal(), _Causal()
        keep_supervised_logits(wrapped)
        self.assertEqual(list(inspect.signature(wrapped.forward).parameters), list(inspect.signature(plain.forward).parameters))

    def test_the_option_is_off_unless_named(self) -> None:
        self.assertFalse(ResearchTrainerSettings(model_name="m", revision="0" * 40, output_dir="o").supervised_logits_only)
        base = ["--model-choice", "12b", "--train", "t.jsonl", "--validation", "v.jsonl", "--output-dir", "out"]
        self.assertFalse(parser().parse_args(base).supervised_logits_only)
        self.assertTrue(parser().parse_args([*base, "--supervised-logits-only"]).supervised_logits_only)


class _StockTrainer:
    """What the span trainer needs of TRL's: a model, the log's two stores, and a compute_loss that reads every logit."""

    def __init__(self, model) -> None:
        from collections import defaultdict

        self.model = model
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        raise AssertionError("the stock trainer's loss reads whole-sequence logits and must not be called")


class SpanTrainerTests(unittest.TestCase):
    def test_the_accuracy_on_the_span_is_the_accuracy_on_the_sequence(self) -> None:
        batch = _batch((40, 28, 33), (5, 6, 1))
        plain, wrapped = _Causal(), _Causal()
        keep_supervised_logits(wrapped)
        whole = span_token_accuracy(plain(**batch).logits, batch["labels"])
        span = span_token_accuracy(wrapped(**batch).logits, batch["labels"])
        self.assertIsNotNone(whole)
        self.assertAlmostEqual(span, whole, places=9)
        self.assertIsNone(span_token_accuracy(None, batch["labels"]))
        self.assertIsNone(span_token_accuracy(plain(**batch).logits, torch.full_like(batch["labels"], IGNORE_INDEX)))

    def test_the_span_trainer_takes_the_loss_as_the_stock_transformers_trainer_does(self) -> None:
        from unittest import mock

        import transformers

        wrapped = _Causal()
        keep_supervised_logits(wrapped)
        batch = _batch((40, 28), (5, 6))
        seen = {}

        def stock(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            seen["use_cache"] = inputs.get("use_cache")
            seen["num_items_in_batch"] = num_items_in_batch
            outputs = model(**{name: value for name, value in inputs.items() if name != "use_cache"})
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

        trainer_class = supervised_span_trainer(_StockTrainer)
        self.assertTrue(issubclass(trainer_class, _StockTrainer))
        self.assertEqual(trainer_class.__name__, "SupervisedSpan_StockTrainer")
        trainer = trainer_class(wrapped)
        with mock.patch.object(transformers.Trainer, "compute_loss", stock):
            wrapped.train()
            loss, outputs = trainer.compute_loss(wrapped, dict(batch), return_outputs=True)
            alone = trainer.compute_loss(wrapped, dict(batch))
            wrapped.eval()
            trainer.compute_loss(wrapped, dict(batch))
        self.assertAlmostEqual(float(loss), float(_Causal()(**batch).loss), places=5)
        self.assertAlmostEqual(float(alone), float(loss), places=6)
        self.assertEqual(seen, {"use_cache": False, "num_items_in_batch": None})
        # The log keeps the token count and the accuracy; no entropy over the prompt.
        tokens = int(batch["attention_mask"].sum())
        self.assertEqual(trainer._metrics["train"]["num_tokens"], [2 * tokens])
        self.assertEqual(len(trainer._metrics["train"]["mean_token_accuracy"]), 2)
        self.assertEqual(len(trainer._metrics["eval"]["mean_token_accuracy"]), 1)
        self.assertEqual(trainer._metrics["eval"]["num_tokens"], [2 * tokens])   # evaluation adds no training tokens
        self.assertNotIn("entropy", trainer._metrics["train"])
        self.assertAlmostEqual(trainer._metrics["train"]["mean_token_accuracy"][0],
                               span_token_accuracy(outputs.logits, batch["labels"]), places=9)


if __name__ == "__main__":
    unittest.main()
