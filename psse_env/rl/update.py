"""The on-policy LoRA update.

One gradient pass per rollout batch, so the importance ratio is one and the
loss is the advantage-weighted log-likelihood of the sampled completion
tokens, normalized by the number of completion tokens in the optimizer step
(the DAPO normalization TRL uses by default), plus an optional KL penalty to
the reference adapter estimated with the k3 estimator on the sampled tokens.
Rows are long prompts with short completions, so each row is its own forward
pass and the model is asked for the completion logits only (``logits_to_keep``).

The model is any causal LM whose forward returns ``.logits``; a PEFT model
supplies ``disable_adapter()`` for the reference pass.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from psse_env.rl.rows import TrainingRow

UPDATE_CONTRACT = "rl_on_policy_lora_update_v1"
LOSS_NORMALIZATIONS = ("dapo", "sequence")


@dataclass(frozen=True)
class UpdateSettings:
    learning_rate: float = 1e-5
    beta_kl: float = 0.02
    optimizer_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    loss_normalization: str = "dapo"
    use_logits_to_keep: bool = True
    seed: int = 20261014
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.beta_kl) or self.beta_kl < 0:
            raise ValueError("beta_kl must be finite and nonnegative")
        if self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive")
        if self.loss_normalization not in LOSS_NORMALIZATIONS:
            raise ValueError(f"loss_normalization must be one of {LOSS_NORMALIZATIONS}")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateReport:
    contract: str
    rows: int
    completion_tokens: int
    optimizer_steps: int
    policy_loss: list[float]
    kl: list[float]
    grad_norm: list[float]
    mean_advantage: float
    seconds: float
    settings: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_tensors(row: TrainingRow, device: Any, torch_module: Any) -> dict[str, Any]:
    record = row.model_record()
    tensors = {
        "input_ids": torch_module.tensor([record["input_ids"]], dtype=torch_module.long, device=device),
        "attention_mask": torch_module.tensor([record["attention_mask"]], dtype=torch_module.long, device=device),
    }
    for key, values in record.items():
        if key in {"input_ids", "attention_mask", "labels", "completion_mask"}:
            continue
        tensors[key] = torch_module.tensor([values], dtype=torch_module.long, device=device)
    return tensors


def completion_logprobs(model: Any, row: TrainingRow, *, use_logits_to_keep: bool = True) -> Any:
    """Log-probabilities of the row's completion tokens under ``model`` (shape ``[n]``).

    The completion is the last ``n`` tokens of the row.  With
    ``logits_to_keep`` the model returns the last ``n + 1`` logits only; the
    first ``n`` of those predict the completion tokens.
    """

    import torch

    device = _model_device(model)
    inputs = _row_tensors(row, device, torch)
    n = row.completion_tokens
    if n <= 0:
        raise ValueError("row has no completion tokens")
    outputs = None
    if use_logits_to_keep:
        try:
            outputs = model(**inputs, logits_to_keep=n + 1, use_cache=False)
        except TypeError:
            outputs = None
    if outputs is None:
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits
    if logits.shape[1] > n + 1:
        logits = logits[:, -(n + 1) :, :]
    predicting = logits[:, :-1, :].float()
    targets = inputs["input_ids"][:, -n:]
    logprobs = torch.log_softmax(predicting, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logprobs[0]


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    for parameter in model.parameters():
        return parameter.device
    raise ValueError("model has no parameters")


def _reference_context(model: Any) -> Any:
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        raise ValueError("a KL penalty needs a PEFT model with disable_adapter(); set beta_kl to 0 otherwise")
    return disable()


def _trainable_parameters(model: Any) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def policy_gradient_update(
    model: Any,
    rows: Sequence[TrainingRow],
    settings: UpdateSettings,
    *,
    optimizer: Any | None = None,
    reference_logprobs: Callable[[TrainingRow], Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> UpdateReport:
    """Apply ``settings.optimizer_steps`` AdamW steps over ``rows`` and report the losses.

    ``reference_logprobs`` overrides the reference pass (tests); by default
    the PEFT adapter is disabled for it.  Rows with a zero advantage still
    count toward the token normalization, as in GRPO.
    """

    import torch

    settings.validate()
    started = time.perf_counter()
    if not rows:
        raise ValueError("no training rows")
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(settings.adam_beta1, settings.adam_beta2),
            eps=settings.adam_epsilon,
        )
    order = list(range(len(rows)))
    random.Random(settings.seed).shuffle(order)
    chunk_count = min(settings.optimizer_steps, len(order))
    chunks = [order[index::chunk_count] for index in range(chunk_count)]
    model.train()
    policy_losses: list[float] = []
    kls: list[float] = []
    grad_norms: list[float] = []
    total_tokens = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_rows = [rows[index] for index in chunk]
        chunk_tokens = sum(row.completion_tokens for row in chunk_rows)
        total_tokens += chunk_tokens
        optimizer.zero_grad(set_to_none=True)
        chunk_policy = 0.0
        chunk_kl = 0.0
        for row in chunk_rows:
            logprobs = completion_logprobs(model, row, use_logits_to_keep=settings.use_logits_to_keep)
            policy_term = -(float(row.advantage) * logprobs)
            kl_term = None
            if settings.beta_kl > 0.0:
                with torch.no_grad():
                    if reference_logprobs is not None:
                        reference = reference_logprobs(row).detach().to(logprobs.device)
                    else:
                        with _reference_context(model):
                            reference = completion_logprobs(
                                model, row, use_logits_to_keep=settings.use_logits_to_keep
                            ).detach()
                delta = reference - logprobs
                kl_term = torch.exp(delta) - delta - 1.0
            if settings.loss_normalization == "dapo":
                scale = 1.0 / max(1, chunk_tokens)
                loss = policy_term.sum() * scale
                if kl_term is not None:
                    loss = loss + settings.beta_kl * kl_term.sum() * scale
            else:
                scale = 1.0 / max(1, len(chunk_rows))
                loss = policy_term.mean() * scale
                if kl_term is not None:
                    loss = loss + settings.beta_kl * kl_term.mean() * scale
            loss.backward()
            chunk_policy += float(policy_term.sum().detach()) * (scale if settings.loss_normalization == "dapo" else scale / max(1, row.completion_tokens))
            if kl_term is not None:
                chunk_kl += float(kl_term.mean().detach()) / max(1, len(chunk_rows))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm))
        optimizer.step()
        policy_losses.append(chunk_policy)
        kls.append(chunk_kl)
        grad_norms.append(grad_norm)
        if progress is not None:
            progress({
                "optimizer_step": chunk_index + 1,
                "rows": len(chunk_rows),
                "completion_tokens": chunk_tokens,
                "policy_loss": chunk_policy,
                "kl": chunk_kl,
                "grad_norm": grad_norm,
                "elapsed_seconds": time.perf_counter() - started,
            })
    model.eval()
    mean_advantage = sum(row.advantage for row in rows) / len(rows)
    return UpdateReport(
        contract=UPDATE_CONTRACT,
        rows=len(rows),
        completion_tokens=total_tokens,
        optimizer_steps=len(chunks),
        policy_loss=policy_losses,
        kl=kls,
        grad_norm=grad_norms,
        mean_advantage=float(mean_advantage),
        seconds=time.perf_counter() - started,
        settings=settings.as_dict(),
    )


__all__ = [
    "LOSS_NORMALIZATIONS",
    "UPDATE_CONTRACT",
    "UpdateReport",
    "UpdateSettings",
    "completion_logprobs",
    "policy_gradient_update",
]
