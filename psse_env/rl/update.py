"""The on-policy LoRA update.

One gradient pass per rollout batch, so the importance ratio is one and the
loss is the advantage-weighted log-likelihood of the sampled completion
tokens, normalized by the number of completion tokens in the optimizer step
(the DAPO normalization TRL uses by default).  Log-probabilities are taken at
the sampling temperature, so the policy being optimized is the one that
produced the samples.

An optional KL penalty (``beta_kl > 0``) keeps the policy near a reference
adapter, estimated with the k3 estimator on the sampled tokens.  The
reference is the starting adapter of the run, loaded beside the trained one
as a second PEFT adapter named ``reference``; it is never the bare base
model, which would pull the student back toward the untuned Gemma.

Rows are long prompts with short completions, so each row is its own forward
pass and the model is asked for the completion logits only
(``logits_to_keep``).  An :class:`UpdateCheckpoint` makes the update survive
preemption: the adapter and optimizer state after every optimizer step, and
the accumulated gradients every few rows inside a step, are written as
immutable files committed by one ``state.json``.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from psse_env.rl.rows import TrainingRow

UPDATE_CONTRACT = "rl_on_policy_lora_update_v2"
LOSS_NORMALIZATIONS = ("dapo", "sequence")
REFERENCE_ADAPTER = "reference"
TRAINED_ADAPTER = "default"


@dataclass(frozen=True)
class UpdateSettings:
    learning_rate: float = 1e-5
    beta_kl: float = 0.0
    optimizer_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    loss_normalization: str = "dapo"
    use_logits_to_keep: bool = True
    temperature: float = 1.0
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
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

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
    resumed_from: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateCheckpoint:
    """Where and how often a running update persists itself.

    ``save_model(model, directory)`` writes the trained weights (a PEFT
    ``save_pretrained`` of the trained adapter in the driver); the caller
    reloads them from :meth:`model_dir` before resuming.
    """

    directory: Path
    save_model: Callable[[Any, Path], None]
    every_rows: int = 32

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.every_rows <= 0:
            raise ValueError("every_rows must be positive")

    @property
    def state_path(self) -> Path:
        return self.directory / "state.json"

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def model_dir(self) -> Path | None:
        """The weights to reload before resuming, or ``None`` before the first optimizer step."""
        state = self.load_state()
        if not state or not state.get("model_dir"):
            return None
        path = self.directory / str(state["model_dir"])
        return path if path.exists() else None


def _commit_state(checkpoint: UpdateCheckpoint, state: Mapping[str, Any]) -> None:
    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(checkpoint.state_path)
    keep = {str(state.get("model_dir") or ""), str(state.get("optimizer_file") or ""), str(state.get("grads_file") or "")}
    for path in checkpoint.directory.iterdir():
        if path.name in {"state.json"} or path.name in keep:
            continue
        if path.name.startswith(("step_", "optimizer_", "grads_")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


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


def completion_logprobs(
    model: Any, row: TrainingRow, *, use_logits_to_keep: bool = True, temperature: float = 1.0
) -> Any:
    """Log-probabilities of the row's completion tokens under ``model`` at ``temperature`` (shape ``[n]``).

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
    predicting = logits[:, :-1, :].float() / float(temperature)
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


def peft_reference_logprobs(
    model: Any,
    *,
    adapter_name: str = REFERENCE_ADAPTER,
    trained_adapter: str = TRAINED_ADAPTER,
    use_logits_to_keep: bool = True,
    temperature: float = 1.0,
) -> Callable[[TrainingRow], Any]:
    """Reference log-probabilities from a second PEFT adapter of the same model.

    PEFT's ``set_adapter`` also flips ``requires_grad`` on the adapters it
    activates and deactivates, so the trainable flags are snapshotted before
    the switch and restored after it; the function asserts that the set of
    trainable parameters is unchanged.
    """

    config = getattr(model, "peft_config", None)
    if not isinstance(config, Mapping) or adapter_name not in config:
        raise ValueError(
            f"a KL penalty needs the starting adapter loaded as PEFT adapter {adapter_name!r}; "
            "set beta_kl to 0 otherwise"
        )
    if trained_adapter not in config:
        raise ValueError(f"the trained adapter {trained_adapter!r} is not loaded")

    def reference(row: TrainingRow) -> Any:
        import torch

        flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        try:
            model.set_adapter(adapter_name)
            with torch.no_grad():
                values = completion_logprobs(
                    model, row, use_logits_to_keep=use_logits_to_keep, temperature=temperature
                ).detach()
        finally:
            model.set_adapter(trained_adapter)
            for name, parameter in model.named_parameters():
                wanted = flags.get(name)
                if wanted is not None and parameter.requires_grad != wanted:
                    parameter.requires_grad_(wanted)
        after = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        before = {name for name, value in flags.items() if value}
        if after != before:
            raise RuntimeError("switching to the reference adapter changed the trainable parameter set")
        return values

    return reference


def _trainable_parameters(model: Any) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _chunks(row_count: int, settings: UpdateSettings) -> list[list[int]]:
    order = list(range(row_count))
    random.Random(settings.seed).shuffle(order)
    chunk_count = min(settings.optimizer_steps, len(order))
    return [order[index::chunk_count] for index in range(chunk_count)]


def policy_gradient_update(
    model: Any,
    rows: Sequence[TrainingRow],
    settings: UpdateSettings,
    *,
    optimizer: Any | None = None,
    reference_logprobs: Callable[[TrainingRow], Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint: UpdateCheckpoint | None = None,
) -> UpdateReport:
    """Apply ``settings.optimizer_steps`` AdamW steps over ``rows`` and report the losses.

    With ``beta_kl > 0`` and no ``reference_logprobs``, the reference is the
    model's PEFT adapter named ``reference``.  Rows with a zero advantage
    still count toward the token normalization, as in GRPO.  With a
    ``checkpoint``, a resumed call (same rows, same settings) continues from
    the last committed state; the caller must have reloaded the model from
    ``checkpoint.model_dir()`` when it is not ``None``.
    """

    import torch

    settings.validate()
    started = time.perf_counter()
    if not rows:
        raise ValueError("no training rows")
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if settings.beta_kl > 0.0 and reference_logprobs is None:
        reference_logprobs = peft_reference_logprobs(
            model, use_logits_to_keep=settings.use_logits_to_keep, temperature=settings.temperature
        )
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(settings.adam_beta1, settings.adam_beta2),
            eps=settings.adam_epsilon,
        )
    chunks = _chunks(len(rows), settings)
    chunk_tokens = [sum(rows[index].completion_tokens for index in chunk) for chunk in chunks]
    policy_losses: list[float] = []
    kls: list[float] = []
    grad_norms: list[float] = []
    start_chunk = 0
    start_row = 0
    partial_policy = 0.0
    partial_kl = 0.0
    resumed_from: dict[str, Any] | None = None
    state = checkpoint.load_state() if checkpoint is not None else None
    if state is not None:
        if int(state.get("row_count", -1)) != len(rows) or int(state.get("chunk_count", -1)) != len(chunks):
            raise RuntimeError("the update checkpoint was written for a different batch of rows")
        start_chunk = int(state["completed_chunks"])
        start_row = int(state.get("chunk_rows_done") or 0)
        policy_losses = [float(v) for v in state.get("policy_losses") or []]
        kls = [float(v) for v in state.get("kls") or []]
        grad_norms = [float(v) for v in state.get("grad_norms") or []]
        partial_policy = float(state.get("partial_policy") or 0.0)
        partial_kl = float(state.get("partial_kl") or 0.0)
        if state.get("optimizer_file"):
            optimizer.load_state_dict(torch.load(checkpoint.directory / state["optimizer_file"], map_location="cpu"))
        resumed_from = {"completed_chunks": start_chunk, "chunk_rows_done": start_row}

    model.train()
    for chunk_index in range(start_chunk, len(chunks)):
        chunk = chunks[chunk_index]
        tokens = chunk_tokens[chunk_index]
        first_row = start_row if chunk_index == start_chunk else 0
        if first_row == 0:
            optimizer.zero_grad(set_to_none=True)
            chunk_policy = 0.0
            chunk_kl = 0.0
        else:
            optimizer.zero_grad(set_to_none=True)
            saved = torch.load(checkpoint.directory / state["grads_file"], map_location="cpu")
            for name, parameter in _named_trainable(model):
                if name in saved:
                    parameter.grad = saved[name].to(parameter.device, dtype=parameter.dtype)
            chunk_policy = partial_policy
            chunk_kl = partial_kl
        for position in range(first_row, len(chunk)):
            row = rows[chunk[position]]
            # The reference pass runs first, with no autograd graph alive while
            # the adapter is switched (gradient checkpointing recomputes the
            # forward during backward with whatever adapter is active then).
            reference = reference_logprobs(row).detach() if settings.beta_kl > 0.0 else None
            logprobs = completion_logprobs(
                model, row, use_logits_to_keep=settings.use_logits_to_keep, temperature=settings.temperature
            )
            policy_term = -(float(row.advantage) * logprobs)
            kl_term = None
            if reference is not None:
                delta = reference.to(logprobs.device) - logprobs
                kl_term = torch.exp(delta) - delta - 1.0
            if settings.loss_normalization == "dapo":
                scale = 1.0 / max(1, tokens)
                loss = policy_term.sum() * scale
                if kl_term is not None:
                    loss = loss + settings.beta_kl * kl_term.sum() * scale
                chunk_policy += float(policy_term.sum().detach()) * scale
            else:
                scale = 1.0 / max(1, len(chunk))
                loss = policy_term.mean() * scale
                if kl_term is not None:
                    loss = loss + settings.beta_kl * kl_term.mean() * scale
                chunk_policy += float(policy_term.mean().detach()) * scale
            loss.backward()
            if kl_term is not None:
                chunk_kl += float(kl_term.mean().detach()) / max(1, len(chunk))
            done = position + 1
            if checkpoint is not None and done < len(chunk) and done % checkpoint.every_rows == 0:
                _save_partial(checkpoint, model, optimizer, state_base=_state_base(rows, chunks, chunk_index, policy_losses, kls, grad_norms),
                              rows_done=done, partial_policy=chunk_policy, partial_kl=chunk_kl, previous=state)
                state = checkpoint.load_state()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm))
        optimizer.step()
        policy_losses.append(chunk_policy)
        kls.append(chunk_kl)
        grad_norms.append(grad_norm)
        if checkpoint is not None:
            _save_step(checkpoint, model, optimizer, rows=rows, chunks=chunks, completed=chunk_index + 1,
                       policy_losses=policy_losses, kls=kls, grad_norms=grad_norms)
            state = checkpoint.load_state()
        if progress is not None:
            progress({
                "optimizer_step": chunk_index + 1,
                "rows": len(chunk),
                "completion_tokens": tokens,
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
        completion_tokens=sum(chunk_tokens),
        optimizer_steps=len(chunks),
        policy_loss=policy_losses,
        kl=kls,
        grad_norm=grad_norms,
        mean_advantage=float(mean_advantage),
        seconds=time.perf_counter() - started,
        settings=settings.as_dict(),
        resumed_from=resumed_from,
    )


def _named_trainable(model: Any) -> list[tuple[str, Any]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _state_base(rows, chunks, chunk_index, policy_losses, kls, grad_norms) -> dict[str, Any]:
    return {
        "contract": UPDATE_CONTRACT,
        "row_count": len(rows),
        "chunk_count": len(chunks),
        "completed_chunks": int(chunk_index),
        "policy_losses": list(policy_losses),
        "kls": list(kls),
        "grad_norms": list(grad_norms),
    }


def _save_partial(checkpoint, model, optimizer, *, state_base, rows_done, partial_policy, partial_kl, previous) -> None:
    import torch

    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    name = f"grads_c{state_base['completed_chunks']}_r{rows_done}.pt"
    temporary = checkpoint.directory / (name + ".tmp")
    torch.save({key: parameter.grad.detach().cpu() for key, parameter in _named_trainable(model) if parameter.grad is not None}, temporary)
    temporary.replace(checkpoint.directory / name)
    previous = previous or {}
    _commit_state(checkpoint, {
        **state_base,
        "chunk_rows_done": int(rows_done),
        "partial_policy": float(partial_policy),
        "partial_kl": float(partial_kl),
        "grads_file": name,
        "model_dir": previous.get("model_dir"),
        "optimizer_file": previous.get("optimizer_file"),
    })


def _save_step(checkpoint, model, optimizer, *, rows, chunks, completed, policy_losses, kls, grad_norms) -> None:
    import torch

    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    model_dir = f"step_{completed}"
    temporary_dir = checkpoint.directory / (model_dir + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    checkpoint.save_model(model, temporary_dir)
    final_dir = checkpoint.directory / model_dir
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary_dir.rename(final_dir)
    optimizer_file = f"optimizer_{completed}.pt"
    temporary = checkpoint.directory / (optimizer_file + ".tmp")
    torch.save(optimizer.state_dict(), temporary)
    temporary.replace(checkpoint.directory / optimizer_file)
    _commit_state(checkpoint, {
        **_state_base(rows, chunks, completed, policy_losses, kls, grad_norms),
        "chunk_rows_done": 0,
        "partial_policy": 0.0,
        "partial_kl": 0.0,
        "grads_file": None,
        "model_dir": model_dir,
        "optimizer_file": optimizer_file,
    })


__all__ = [
    "LOSS_NORMALIZATIONS",
    "REFERENCE_ADAPTER",
    "TRAINED_ADAPTER",
    "UPDATE_CONTRACT",
    "UpdateCheckpoint",
    "UpdateReport",
    "UpdateSettings",
    "completion_logprobs",
    "peft_reference_logprobs",
    "policy_gradient_update",
]
