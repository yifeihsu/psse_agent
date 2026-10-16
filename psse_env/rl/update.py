"""The on-policy LoRA update.

One gradient pass per rollout batch, so the importance ratio is one and the
loss is the advantage-weighted log-likelihood of the sampled completion
tokens, normalized by the number of completion tokens in the optimizer step
(the DAPO normalization TRL uses by default).  Log-probabilities are taken at
the sampling temperature, so the policy being optimized is the one that
produced the samples.  That ratio-one reading is exact for the first
optimizer step of a batch only; with ``optimizer_steps > 1`` the later steps
see parameters the earlier ones moved.  ``clip_ratio`` turns the loss into
the clipped PPO surrogate against the probabilities the policy recorded when
it sampled, which makes those later steps sound.  With or without it, every
optimizer step reports how far its log-probabilities are from the recorded
ones (:class:`OnPolicyCheck`), so a run can tell the numerical gap between
the decoding pass and the training pass from the drift its own steps add.

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from psse_env.rl.rows import TrainingRow

UPDATE_CONTRACT = "rl_on_policy_lora_update_v2"
#: ``dapo``: every completion token of the optimizer step weighs the same, so
#: a trajectory weighs as many tokens as it has (a 40-step looping sample
#: outweighs seven 6-step ones).  ``sequence``: every row (step) weighs the
#: same.  ``trajectory``: every trajectory weighs the same, whatever its
#: length; a row's tokens are divided by its trajectory's tokens and the
#: step by its number of trajectories.
LOSS_NORMALIZATIONS = ("dapo", "sequence", "trajectory")
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
    #: Per-row gradient bound (2026-10-15, after a pilot step whose norm was
    #: 5,255 against 37 to 58 elsewhere).  Each row's gradient is clipped to
    #: ``max_row_grad_norm`` and, once ``row_grad_norm_warmup`` rows with a
    #: nonzero gradient have been seen in the optimizer step, to
    #: ``row_grad_norm_factor`` times their median norm, so no single row
    #: can set a step's direction.  A row whose gradient is not finite is
    #: dropped.  Both unset keeps the plain accumulation.
    #:
    #: Caution (hard-draw run, 2026-10-15): most rows sample tokens of
    #: probability close to 1, so their gradients are nearly zero and the
    #: median row norm was 1e-4 against a 95th percentile of 1.5; five times
    #: the median then clipped 40% of the rows, the ones carrying the signal.
    #: The median is not a usable scale for these rows; prefer the absolute
    #: cap, set well above the largest row norm a step reports.
    max_row_grad_norm: float | None = None
    row_grad_norm_factor: float | None = None
    row_grad_norm_warmup: int = 16
    #: PPO-style clip of the importance ratio against the probabilities the
    #: policy recorded when it sampled (``None``: the plain on-policy loss,
    #: whose ratio-one assumption holds for the first optimizer step of a
    #: batch only).  A row without a sampling record keeps the plain loss.
    #: At ratio one the clipped surrogate has the same gradient as the plain
    #: loss, so the first step is unchanged by it.
    clip_ratio: float | None = None

    @property
    def bounds_rows(self) -> bool:
        return self.max_row_grad_norm is not None or self.row_grad_norm_factor is not None

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
        for name in ("max_row_grad_norm", "row_grad_norm_factor"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} must be finite and positive when set")
        if self.row_grad_norm_warmup < 1:
            raise ValueError("row_grad_norm_warmup must be at least 1")
        if self.clip_ratio is not None and (
            not math.isfinite(float(self.clip_ratio)) or not 0.0 < float(self.clip_ratio) < 1.0
        ):
            raise ValueError("clip_ratio must lie in (0, 1) when set")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RowGradientBound:
    """The per-row bound of one optimizer step and the row norms it has seen."""

    def __init__(self, settings: UpdateSettings, norms: Sequence[float] = (), clipped: int = 0, dropped: int = 0) -> None:
        self.settings = settings
        self.norms = [float(value) for value in norms]
        self.clipped = int(clipped)
        self.dropped = int(dropped)

    def bound(self) -> float | None:
        """The bound for the next row, from the rows before it."""
        bounds: list[float] = []
        if self.settings.max_row_grad_norm is not None:
            bounds.append(float(self.settings.max_row_grad_norm))
        if self.settings.row_grad_norm_factor is not None:
            nonzero = sorted(value for value in self.norms if value > 0.0)
            if len(nonzero) >= self.settings.row_grad_norm_warmup:
                middle = len(nonzero) // 2
                median = nonzero[middle] if len(nonzero) % 2 else 0.5 * (nonzero[middle - 1] + nonzero[middle])
                bounds.append(float(self.settings.row_grad_norm_factor) * median)
        return min(bounds) if bounds else None

    def record(self, norm: float, *, clipped: bool, dropped: bool) -> None:
        if dropped:
            self.dropped += 1
            return
        self.norms.append(float(norm))
        self.clipped += int(clipped)

    def summary(self) -> dict[str, Any]:
        values = sorted(self.norms)
        nonzero = [value for value in values if value > 0.0]

        def quantile(q: float) -> float | None:
            if not values:
                return None
            return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

        return {
            "rows": len(values) + self.dropped,
            "nonzero_rows": len(nonzero),
            "median": quantile(0.5),
            "p95": quantile(0.95),
            "max": values[-1] if values else None,
            "clipped": self.clipped,
            "dropped_nonfinite": self.dropped,
            "bound_after_step": self.bound(),
        }


class OnPolicyCheck:
    """How far one optimizer step's log-probabilities are from the ones the policy sampled with.

    Rows carry the sampling probabilities the policy recorded; the update
    recomputes the same tokens' log-probabilities with the model it trains.
    At the first optimizer step of a batch the parameters are the sampling
    parameters, so any gap is numerical (the training forward pass against
    the cached decoding pass); from the second step on it also holds the
    drift the earlier steps introduced.  ``threshold`` is a log-ratio: the
    tokens beyond it are the ones a PPO clip of that size would clip.
    """

    def __init__(self, *, threshold: float = math.log(1.2), state: Mapping[str, Any] | None = None) -> None:
        state = dict(state or {})
        self.threshold = float(state.get("threshold") or threshold)
        self.rows = int(state.get("rows") or 0)
        self.rows_without_record = int(state.get("rows_without_record") or 0)
        self.tokens = int(state.get("tokens") or 0)
        self.sum_abs = float(state.get("sum_abs") or 0.0)
        self.sum = float(state.get("sum") or 0.0)
        self.max_abs = float(state.get("max_abs") or 0.0)
        self.beyond = int(state.get("beyond") or 0)

    def record(self, logprobs: Any, sampling_logprobs: Any | None) -> None:
        """Compare one row's recomputed log-probabilities (shape ``[n]``) with its recorded ones."""
        if sampling_logprobs is None:
            self.rows_without_record += 1
            return
        delta = (logprobs.detach().float() - sampling_logprobs.detach().float()).reshape(-1)
        self.rows += 1
        count = int(delta.numel())
        self.tokens += count
        if count:
            self.sum_abs += float(delta.abs().sum())
            self.sum += float(delta.sum())
            self.max_abs = max(self.max_abs, float(delta.abs().max()))
            self.beyond += int((delta.abs() > self.threshold).sum())

    def state(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "rows": self.rows,
            "rows_without_record": self.rows_without_record,
            "tokens": self.tokens,
            "sum_abs": self.sum_abs,
            "sum": self.sum,
            "max_abs": self.max_abs,
            "beyond": self.beyond,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "rows_without_record": self.rows_without_record,
            "tokens": self.tokens,
            "mean_log_ratio": (self.sum / self.tokens) if self.tokens else None,
            "mean_abs_log_ratio": (self.sum_abs / self.tokens) if self.tokens else None,
            "max_abs_log_ratio": self.max_abs if self.tokens else None,
            "tokens_beyond_threshold": self.beyond,
            "threshold_log_ratio": self.threshold,
        }


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
    #: Per optimizer step, when the per-row bound is on: row norm quantiles,
    #: rows clipped and rows dropped as non-finite.
    row_grad_norms: list[dict[str, Any]] = field(default_factory=list)
    #: Per optimizer step: the gap between the recomputed and the recorded
    #: sampling log-probabilities (:class:`OnPolicyCheck`).
    on_policy_check: list[dict[str, Any]] = field(default_factory=list)

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
    partial_rows: Mapping[str, Any] = {}
    row_summaries: list[dict[str, Any]] = []
    check_summaries: list[dict[str, Any]] = []
    partial_check: Mapping[str, Any] = {}
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
        partial_rows = state.get("partial_row_norms") or {}
        row_summaries = list(state.get("row_grad_norms") or [])
        partial_check = state.get("partial_on_policy") or {}
        check_summaries = list(state.get("on_policy_checks") or [])
        if state.get("optimizer_file"):
            optimizer.load_state_dict(torch.load(checkpoint.directory / state["optimizer_file"], map_location="cpu"))
        resumed_from = {"completed_chunks": start_chunk, "chunk_rows_done": start_row}

    model.train()
    for chunk_index in range(start_chunk, len(chunks)):
        chunk = chunks[chunk_index]
        tokens = chunk_tokens[chunk_index]
        first_row = start_row if chunk_index == start_chunk else 0
        bound = (
            RowGradientBound(settings, norms=partial_rows.get("norms") or (), clipped=int(partial_rows.get("clipped") or 0),
                             dropped=int(partial_rows.get("dropped") or 0))
            if first_row > 0 else RowGradientBound(settings)
        )
        check = OnPolicyCheck(state=partial_check if first_row > 0 else None)
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
        trajectories = len({(rows[index].root, rows[index].sample_index) for index in chunk})
        for position in range(first_row, len(chunk)):
            row = rows[chunk[position]]
            # The reference pass runs first, with no autograd graph alive while
            # the adapter is switched (gradient checkpointing recomputes the
            # forward during backward with whatever adapter is active then).
            reference = reference_logprobs(row).detach() if settings.beta_kl > 0.0 else None
            logprobs = completion_logprobs(
                model, row, use_logits_to_keep=settings.use_logits_to_keep, temperature=settings.temperature
            )
            old_logprobs = None
            if row.sampling_logprobs is not None and len(row.sampling_logprobs) == int(logprobs.shape[0]):
                old_logprobs = torch.tensor(list(row.sampling_logprobs), dtype=logprobs.dtype, device=logprobs.device)
            check.record(logprobs, old_logprobs)
            policy_term = _policy_term(logprobs, old_logprobs, float(row.advantage), settings.clip_ratio, torch)
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
            elif settings.loss_normalization == "trajectory":
                # A row without a trajectory total (rows built outside build_rows) is its own trajectory.
                scale = 1.0 / (max(1, int(row.trajectory_tokens or row.completion_tokens)) * max(1, trajectories))
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
            if settings.bounds_rows:
                _bounded_row_backward(loss, parameters, bound, torch)
            else:
                loss.backward()
            if kl_term is not None:
                chunk_kl += float(kl_term.mean().detach()) / max(1, len(chunk))
            done = position + 1
            if checkpoint is not None and done < len(chunk) and done % checkpoint.every_rows == 0:
                _save_partial(checkpoint, model, optimizer,
                              state_base={**_state_base(rows, chunks, chunk_index, policy_losses, kls, grad_norms),
                                          "row_grad_norms": list(row_summaries),
                                          "partial_row_norms": {"norms": list(bound.norms), "clipped": bound.clipped,
                                                                "dropped": bound.dropped},
                                          "on_policy_checks": list(check_summaries),
                                          "partial_on_policy": check.state()},
                              rows_done=done, partial_policy=chunk_policy, partial_kl=chunk_kl, previous=state)
                state = checkpoint.load_state()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm))
        optimizer.step()
        policy_losses.append(chunk_policy)
        kls.append(chunk_kl)
        grad_norms.append(grad_norm)
        if settings.bounds_rows:
            row_summaries.append(bound.summary())
        check_summaries.append(check.summary())
        if checkpoint is not None:
            _save_step(checkpoint, model, optimizer, rows=rows, chunks=chunks, completed=chunk_index + 1,
                       policy_losses=policy_losses, kls=kls, grad_norms=grad_norms, row_summaries=row_summaries,
                       check_summaries=check_summaries)
            state = checkpoint.load_state()
        if progress is not None:
            progress({
                "optimizer_step": chunk_index + 1,
                "rows": len(chunk),
                "completion_tokens": tokens,
                "policy_loss": chunk_policy,
                "kl": chunk_kl,
                "grad_norm": grad_norm,
                "row_grad_norms": row_summaries[-1] if settings.bounds_rows else None,
                "on_policy_check": check_summaries[-1],
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
        row_grad_norms=row_summaries,
        on_policy_check=check_summaries,
    )


def _policy_term(logprobs: Any, old_logprobs: Any | None, advantage: float, clip_ratio: float | None, torch_module: Any) -> Any:
    """Per-token policy loss: the plain on-policy term, or the clipped PPO surrogate.

    The surrogate needs both a clip and the row's recorded sampling
    log-probabilities; ``min(r A, clip(r) A)`` with ``r`` the per-token
    ratio.  At ``r = 1`` its gradient is ``A`` times the score, the plain
    term's gradient, so a first optimizer step is the same either way.
    """

    if clip_ratio is None or old_logprobs is None:
        return -(advantage * logprobs)
    ratio = torch_module.exp(logprobs - old_logprobs)
    clipped = torch_module.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
    return -torch_module.minimum(ratio * advantage, clipped * advantage)


def _bounded_row_backward(loss: Any, parameters: Sequence[Any], bound: RowGradientBound, torch_module: Any) -> None:
    """Backpropagate one row alone, clip its gradient to the row bound and add it to the step's sum.

    The step's accumulated gradient is held aside while the row's own
    gradient is computed, so the row's norm is measured on its own; a row
    whose norm is not finite contributes nothing.
    """

    held = [parameter.grad for parameter in parameters]
    for parameter in parameters:
        parameter.grad = None
    loss.backward()
    limit = bound.bound()
    norm = float(torch_module.nn.utils.clip_grad_norm_(parameters, max_norm=limit if limit is not None else float("inf")))
    finite = math.isfinite(norm)
    bound.record(norm, clipped=bool(finite and limit is not None and norm > limit), dropped=not finite)
    for parameter, previous in zip(parameters, held):
        row_gradient = parameter.grad if finite else None
        if previous is None:
            parameter.grad = row_gradient
        elif row_gradient is None:
            parameter.grad = previous
        else:
            parameter.grad = previous.add_(row_gradient)


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


def _save_step(checkpoint, model, optimizer, *, rows, chunks, completed, policy_losses, kls, grad_norms, row_summaries=(),
               check_summaries=()) -> None:
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
        "row_grad_norms": list(row_summaries),
        "partial_row_norms": {},
        "on_policy_checks": list(check_summaries),
        "partial_on_policy": {},
        "chunk_rows_done": 0,
        "partial_policy": 0.0,
        "partial_kl": 0.0,
        "grads_file": None,
        "model_dir": model_dir,
        "optimizer_file": optimizer_file,
    })


__all__ = [
    "LOSS_NORMALIZATIONS",
    "OnPolicyCheck",
    "REFERENCE_ADAPTER",
    "RowGradientBound",
    "TRAINED_ADAPTER",
    "UPDATE_CONTRACT",
    "UpdateCheckpoint",
    "UpdateReport",
    "UpdateSettings",
    "completion_logprobs",
    "peft_reference_logprobs",
    "policy_gradient_update",
]
