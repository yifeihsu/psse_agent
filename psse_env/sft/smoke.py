"""One-batch and tiny-overfit training smoke gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .collator import AssistantOnlyCollator
from .gates import GateError, ParsedToolCall, PreparedExample, parse_tool_call


@dataclass(frozen=True)
class SmokeResult:
    passed: bool
    steps: int
    initial_loss: float
    final_loss: float
    gradients_finite: bool
    parameter_changed: bool
    loss_decreased: bool
    generation_round_trip: bool | None = None
    generated_tool_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loss_from_output(output: Any) -> Any:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, dict):
        loss = output.get("loss")
    if loss is None:
        raise GateError("Model forward output has no loss.")
    return loss


def _model_device(model: Any) -> Any:
    try:
        return next(parameter for parameter in model.parameters()).device
    except StopIteration as exc:
        raise GateError("Model has no parameters for a forward/backward smoke test.") from exc


def run_training_smoke(
    model: Any,
    processor: Any,
    examples: Sequence[PreparedExample],
    *,
    steps: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
    require_loss_decrease: bool | None = None,
) -> SmokeResult:
    """Run real forward/backward optimizer steps over assistant-only labels.

    ``steps=1`` is the one-batch gate.  For ``steps>1`` the default additionally
    requires a lower final loss, making it a tiny-overfit gate.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(f"torch is required for the training smoke gate: {exc}") from exc
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive.")
    if not examples:
        raise GateError("Training smoke gate has no prepared examples.")
    require_loss_decrease = steps > 1 if require_loss_decrease is None else require_loss_decrease
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise GateError("Model has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    before = [parameter.detach().clone() for parameter in trainable]
    collator = AssistantOnlyCollator(processor)
    device = _model_device(model)
    losses: list[float] = []
    gradients_finite = True
    model.train()
    subset = list(examples[:batch_size])
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in collator(subset).items()}
        output = model(**batch)
        loss = _loss_from_output(output)
        numeric_loss = float(loss.detach().cpu())
        if not math.isfinite(numeric_loss):
            raise GateError(f"Training smoke produced non-finite loss {numeric_loss}.")
        losses.append(numeric_loss)
        loss.backward()
        for parameter in trainable:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                gradients_finite = False
        if not gradients_finite:
            raise GateError("Training smoke produced non-finite gradients.")
        optimizer.step()
    changed = any(not torch.equal(previous, current.detach()) for previous, current in zip(before, trainable))
    decreased = losses[-1] < losses[0] if len(losses) > 1 else False
    passed = gradients_finite and changed and (decreased or not require_loss_decrease)
    if not passed:
        raise GateError(
            "Training smoke failed: "
            f"parameter_changed={changed}, gradients_finite={gradients_finite}, "
            f"initial_loss={losses[0]:.6g}, final_loss={losses[-1]:.6g}."
        )
    return SmokeResult(
        passed=True,
        steps=steps,
        initial_loss=losses[0],
        final_loss=losses[-1],
        gradients_finite=gradients_finite,
        parameter_changed=changed,
        loss_decreased=decreased,
    )


def run_generation_tool_call_smoke(
    model: Any,
    processor: Any,
    example: PreparedExample,
    *,
    max_new_tokens: int | None = None,
) -> ParsedToolCall:
    """Greedily generate and require the expected canonical tool call."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(f"torch is required for generation smoke: {exc}") from exc
    expected = example.expected_tool_call
    if expected is None:
        raise GateError("Generation smoke requires a tool-call target example.")
    try:
        completion_start = example.completion_mask.index(1)
    except ValueError as exc:
        raise GateError("Generation smoke example has no completion boundary.") from exc
    if completion_start <= 0:
        raise GateError("Generation smoke example has no retained prompt tokens.")
    device = _model_device(model)
    inputs: dict[str, Any] = {
        "input_ids": torch.tensor([example.input_ids[:completion_start]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [example.attention_mask[:completion_start]], dtype=torch.long, device=device
        ),
    }
    for key, values in example.side_inputs.items():
        inputs[key] = torch.tensor([values[:completion_start]], dtype=torch.long, device=device)
    decoder = processor if callable(getattr(processor, "decode", None)) else getattr(processor, "tokenizer", None)
    if decoder is None or not callable(getattr(decoder, "decode", None)):
        raise GateError("Processor/tokenizer exposes no decode() method for generation smoke.")
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or (example.supervised_tokens + 16),
            do_sample=False,
        )
    output_ids = generated[0][completion_start:].detach().cpu()
    text = decoder.decode(output_ids, skip_special_tokens=False)
    parsed = parse_tool_call(text)
    if parsed != expected:
        raise GateError(
            "Generated tool call does not round-trip to the target: "
            f"expected={expected}, generated={parsed}, text={text[:240]!r}."
        )
    return parsed
