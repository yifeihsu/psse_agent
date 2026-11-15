"""Logits only where the labels are supervised (2026-11-15).

A cloning row is a prompt of thousands of tokens and a target of a few dozen:
every label before the assistant's tool call is ``-100``.  The model's own
loss still computes a logit for every position, one row of the vocabulary
each: at 262,144 entries that is about 1 MB a position in float32, 19.9 GiB
for the 20,390-token prompt of the complete-evidence rows, several copies of
which the loss holds at once.  The training smoke of 2026-11-15 ran out of a
95 GB card on that first row.  None of it is needed: a position's logits
enter the loss only where the next token is supervised.

``keep_supervised_logits`` makes a model's forward, when it is given labels,
ask for the logits of the supervised span alone (``logits_to_keep``, which the
RL update has used since 2026-10-14) and compute the same loss from them: the
mean cross-entropy over the supervised tokens.  The prompt is still read in
full; only the vocabulary projection of the unsupervised positions is left
out.  The value is the model's own loss up to floating point (the tests check
the loss and its gradients on a small model).

The trainer needs one change with it.  TRL's ``SFTTrainer.compute_loss`` reads
the whole-sequence logits twice more for its own log, an entropy over every
position and a token accuracy, and would fail on a span.
``supervised_span_trainer`` gives the same trainer with the stock
``transformers.Trainer.compute_loss`` and the accuracy taken on the span, which
is the same number (it only ever counted supervised tokens); the entropy of the
prompt positions is not logged.

Off unless the trainer is run with ``--supervised-logits-only``, so every
earlier pass is reproduced as it was run.  A forward that does not take
``logits_to_keep`` is called as before and counted, so a report says whether
the saving was real.

Checked against the installed stack on torch (2026-11-15: transformers 5.10.4,
TRL 0.24.0): ``Gemma4UnifiedForConditionalGeneration.forward`` projects
``hidden_states[:, -logits_to_keep:, :]`` only, its ``accepts_loss_kwargs`` is
false, so the trainer hands it no ``num_items_in_batch`` and its loss is the
plain mean that is computed here, divided by the accumulation steps by the
trainer as before.
"""
from __future__ import annotations

import functools
from typing import Any

IGNORE_INDEX = -100


def supervised_start(labels: Any) -> int | None:
    """The first position any row of a ``[batch, length]`` label tensor supervises; ``None`` when none does."""
    supervised = (labels != IGNORE_INDEX).any(dim=0)
    if not bool(supervised.any()):
        return None
    return int(supervised.nonzero()[0])


def keep_supervised_logits(model: Any) -> dict[str, Any]:
    """Install the supervised-span forward on ``model``; returns the live counts of how it was called.

    ``kept_calls`` asked for the supervised span and got it; ``ignored_calls`` asked and were handed the whole
    sequence anyway (the loss is still right, the memory is not saved); ``full_calls`` went to the model
    unchanged: no labels, nothing supervised, supervision from the first position, or a forward without
    ``logits_to_keep`` (``refused_calls`` counts those among them).
    """
    import torch
    from transformers.modeling_outputs import CausalLMOutput

    inner = model.forward
    counts: dict[str, Any] = {"enabled": True, "kept_calls": 0, "ignored_calls": 0, "full_calls": 0, "refused_calls": 0,
                              "largest_kept": 0, "longest_sequence": 0}

    @functools.wraps(inner)
    def forward(*args: Any, **kwargs: Any) -> Any:
        labels = kwargs.get("labels")
        start = None
        if labels is not None and kwargs.get("logits_to_keep") is None and getattr(labels, "dim", lambda: 0)() == 2:
            start = supervised_start(labels)
        if start is None or start == 0:
            counts["full_calls"] += 1
            return inner(*args, **kwargs)
        length = int(labels.shape[1])
        # The logits at position t predict token t + 1: the span starts one position before the first supervised token.
        keep = length - start + 1
        rest = {name: value for name, value in kwargs.items() if name != "labels"}
        try:
            outputs = inner(*args, logits_to_keep=keep, **rest)
        except TypeError:
            counts["full_calls"] += 1
            counts["refused_calls"] += 1
            return inner(*args, **kwargs)
        logits = outputs.logits
        if int(logits.shape[1]) == keep:
            counts["kept_calls"] += 1
        elif int(logits.shape[1]) == length:
            counts["ignored_calls"] += 1
            logits = logits[:, -keep:, :]
        else:
            raise ValueError(f"asked for the last {keep} logits of {length} positions, got {int(logits.shape[1])}")
        counts["largest_kept"] = max(counts["largest_kept"], keep)
        counts["longest_sequence"] = max(counts["longest_sequence"], length)
        predicting = logits[:, :-1, :].float()
        targets = labels[:, start:].to(predicting.device)
        loss = torch.nn.functional.cross_entropy(
            predicting.reshape(-1, predicting.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX
        )
        return CausalLMOutput(loss=loss, logits=logits)

    model.forward = forward
    return counts


def span_token_accuracy(logits: Any, labels: Any) -> float | None:
    """The share of supervised tokens the logits predict, for logits of the whole sequence or of its last positions."""
    if logits is None or labels is None or getattr(logits, "dim", lambda: 0)() != 3 or int(logits.shape[1]) < 2:
        return None
    keep, length = int(logits.shape[1]), int(labels.shape[1])
    if keep > length:
        return None
    predictions = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, length - (keep - 1):].to(predictions.device)
    mask = targets != IGNORE_INDEX
    total = int(mask.sum())
    if total == 0:
        return None
    return float(((predictions == targets) & mask).sum()) / total


def supervised_span_trainer(base: type) -> type:
    """``base`` (TRL's ``SFTTrainer``) computing its loss as ``transformers.Trainer`` does, without the two readings
    of whole-sequence logits; the token accuracy and the token count stay in its log."""
    import torch
    from transformers import Trainer

    class SupervisedSpanTrainer(base):  # type: ignore[misc, valid-type]
        def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, num_items_in_batch: Any = None) -> Any:
            mode = "train" if self.model.training else "eval"
            labels = inputs.get("labels")
            inputs["use_cache"] = False
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            metrics = getattr(self, "_metrics", None)
            if isinstance(metrics, dict) and mode in metrics:
                with torch.no_grad():
                    if mode == "train" and "attention_mask" in inputs:
                        self._total_train_tokens = getattr(self, "_total_train_tokens", 0) + int(inputs["attention_mask"].sum())
                    metrics[mode]["num_tokens"] = [getattr(self, "_total_train_tokens", 0)]
                    accuracy = span_token_accuracy(getattr(outputs, "logits", None), labels)
                    if accuracy is not None:
                        metrics[mode]["mean_token_accuracy"].append(accuracy)
            return (loss, outputs) if return_outputs else loss

    SupervisedSpanTrainer.__name__ = f"SupervisedSpan{base.__name__}"
    return SupervisedSpanTrainer


__all__ = ["IGNORE_INDEX", "keep_supervised_logits", "span_token_accuracy", "supervised_span_trainer", "supervised_start"]
