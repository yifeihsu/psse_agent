"""One-batch and tiny-overfit training smoke gates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
)

from .collator import AssistantOnlyCollator
from .gates import (
    GateError,
    ParsedToolCall,
    PreparedExample,
    parse_tool_call,
    verify_assistant_only_mask,
)


TARGETED_RECOVERY_CASES = (
    "parameter_route_without_scans",
    "measurement_parameter_sequence",
    "failed_correction_recovery",
    "premature_commit_recovery",
    "valid_safe_escalation",
)
TARGETED_TINY_OVERFIT_LEARNING_RATES = (1e-4, 3e-4, 1e-3)
TARGETED_MIN_RELATIVE_LOSS_REDUCTION = 0.20

_PARAMETER_CORRECTION_TOOLS = frozenset(
    {"correct_parameters", "correct_parameters_from_path"}
)
_CORRECTION_TOOLS = frozenset(
    {
        "correct_measurements",
        "correct_measurements_from_path",
        "correct_parameters",
        "correct_parameters_from_path",
        "correct_topology",
        "correct_topology_from_path",
    }
)
_PARAMETER_NO_SCAN_FAILURES = frozenset(
    {"correction_route_not_actionable", "parameter_scans_missing"}
)
_SAFE_ESCALATION_REQUESTS = frozenset(
    {
        HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
        RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    }
)


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


@dataclass(frozen=True)
class TargetedRecoveryExample:
    case: str
    example_id: str
    prepared: PreparedExample


@dataclass(frozen=True)
class TargetedSmokeResult:
    passed: bool
    steps: int
    learning_rate: float
    case_example_ids: dict[str, str]
    initial_mean_loss: float
    final_mean_loss: float
    relative_loss_reduction: float
    minimum_relative_loss_reduction: float
    gradients_finite: bool
    gradients_nonzero: bool
    minimum_gradient_norm: float
    maximum_gradient_norm: float
    parameter_changed: bool
    assistant_only_masks: bool
    case_loss_checks: tuple[dict[str, Any], ...]
    generation_checks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetedSweepResult:
    passed: bool
    learning_rates: tuple[float, ...]
    minimum_relative_loss_reduction: float
    required_cases: tuple[str, ...]
    successful_learning_rates: tuple[float, ...]
    best_diagnostic_learning_rate: float | None
    runs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _row_labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = _row_metadata(row).get("labels")
    return labels if isinstance(labels, Mapping) else {}


def _row_recovery_stratum(row: Mapping[str, Any]) -> str:
    value = row.get("recovery_stratum")
    if value is None:
        value = _row_labels(row).get("recovery_stratum")
    return str(value or "").strip()


def _row_model_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {}
    user_messages = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if not user_messages:
        return {}
    content = user_messages[-1].get("content")
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    state = payload.get("state")
    return state if isinstance(state, Mapping) else payload


def _last_tool_and_error(state: Mapping[str, Any]) -> tuple[str, str]:
    tool = str(state.get("last_tool") or "").strip()
    output = state.get("last_tool_output")
    output = output if isinstance(output, Mapping) else {}
    error = str(output.get("error_code") or "").strip()
    history = state.get("history_window")
    if isinstance(history, list) and history:
        event = history[-1]
        if isinstance(event, Mapping):
            action = event.get("action")
            event_output = event.get("tool_output")
            if not tool and isinstance(action, Mapping):
                tool = str(action.get("tool") or "").strip()
            if not error and isinstance(event_output, Mapping):
                error = str(event_output.get("error_code") or "").strip()
    return tool, error


def _target_tool(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    target = messages[-1]
    if not isinstance(target, Mapping) or target.get("role") != "assistant":
        return ""
    calls = target.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return ""
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    return (
        str(function.get("name") or "").strip()
        if isinstance(function, Mapping)
        else ""
    )


def _target_arguments(row: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return {}
    target = messages[-1]
    calls = target.get("tool_calls") if isinstance(target, Mapping) else None
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    function = call.get("function") if isinstance(call, Mapping) else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    return arguments if isinstance(arguments, Mapping) else {}


def _matches_targeted_case(case: str, row: Mapping[str, Any]) -> bool:
    if row.get("production_label_eligible") is not True:
        return False
    metadata = _row_metadata(row)
    if metadata.get("protocol") != "canonical":
        return False
    stratum = _row_recovery_stratum(row)
    state = _row_model_state(row)
    last_tool, last_error = _last_tool_and_error(state)
    scenario_family = str(
        row.get("scenario_family") or metadata.get("scenario_family") or ""
    ).strip()
    if case == "parameter_route_without_scans":
        return bool(
            stratum == "unsupported_correction_recovery"
            and last_tool in _PARAMETER_CORRECTION_TOOLS
            and last_error in _PARAMETER_NO_SCAN_FAILURES
        )
    if case == "measurement_parameter_sequence":
        return bool(
            stratum == "sequential_measurement_parameter_recovery"
            and scenario_family == "measurement+parameter"
        )
    if case == "failed_correction_recovery":
        return bool(
            stratum == "post_failure_no_candidate"
            and last_tool in _CORRECTION_TOOLS
            and last_error
            and not state.get("has_open_candidate")
            and not state.get("candidate_state_id")
        )
    if case == "premature_commit_recovery":
        return bool(
            stratum == "premature_commit_recovery"
            and last_tool == "commit_state"
            and last_error == "candidate_lifecycle_violation"
        )
    if case == "valid_safe_escalation":
        return bool(
            stratum == "multi_measurement_safe_handoff"
            and _target_tool(row) == "ask_for_more_evidence"
            and _target_arguments(row).get("request")
            in _SAFE_ESCALATION_REQUESTS
        )
    raise ValueError(f"Unknown targeted recovery case {case!r}.")


def select_targeted_recovery_slice(
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[PreparedExample],
) -> tuple[TargetedRecoveryExample, ...]:
    """Select one deterministic, distinct canonical row for each recovery case."""

    if len(rows) != len(examples):
        raise GateError(
            "Targeted recovery selection requires one prepared example per train row."
        )
    selection = targeted_recovery_row_selection(rows)
    selected: list[TargetedRecoveryExample] = []
    for case, index, example_id in selection:
        example = examples[index]
        if example.expected_tool_call is None:
            raise GateError(
                f"Targeted case {case!r} has no canonical tool-call target."
            )
        verify_assistant_only_mask(example)
        selected.append(
            TargetedRecoveryExample(
                case=case,
                example_id=example_id,
                prepared=example,
            )
        )
    return tuple(selected)


def targeted_recovery_row_selection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int, str], ...]:
    """Return the stable row identity for each required recovery case."""

    ordered = sorted(
        enumerate(rows),
        key=lambda item: (
            str(item[1].get("example_id") or ""),
            item[0],
        ),
    )
    selected: list[tuple[str, int, str]] = []
    used_indices: set[int] = set()
    for case in TARGETED_RECOVERY_CASES:
        match = next(
            (
                (index, row)
                for index, row in ordered
                if index not in used_indices and _matches_targeted_case(case, row)
            ),
            None,
        )
        if match is None:
            raise GateError(
                "Targeted tiny-overfit slice is missing required case "
                f"{case!r}."
            )
        index, row = match
        used_indices.add(index)
        selected.append(
            (
                case,
                index,
                str(row.get("example_id") or f"row[{index}]"),
            )
        )
    return tuple(selected)


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


def run_targeted_recovery_smoke(
    model: Any,
    processor: Any,
    selected: Sequence[TargetedRecoveryExample],
    *,
    steps: int = 20,
    learning_rate: float = 1e-4,
    minimum_relative_loss_reduction: float = (
        TARGETED_MIN_RELATIVE_LOSS_REDUCTION
    ),
) -> TargetedSmokeResult:
    """Overfit the exact five-case recovery slice at micro-batch size one.

    Loss is measured over every case before and after the balanced cycle. Each
    optimizer step uses one case, which keeps the 31B diagnostic memory bounded
    while giving every case the same number of updates.
    """

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(
            f"torch is required for the targeted training smoke gate: {exc}"
        ) from exc
    if tuple(item.case for item in selected) != TARGETED_RECOVERY_CASES:
        raise GateError(
            "Targeted recovery slice must contain the five required cases in "
            "the reviewed deterministic order."
        )
    if steps < len(selected) or steps % len(selected) != 0:
        raise ValueError(
            "Targeted tiny-overfit steps must be a positive multiple of five."
        )
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("Targeted tiny-overfit learning rate must be positive.")
    if not (
        math.isfinite(float(minimum_relative_loss_reduction))
        and 0.0 < float(minimum_relative_loss_reduction) < 1.0
    ):
        raise ValueError(
            "minimum_relative_loss_reduction must be strictly between zero and one."
        )
    for item in selected:
        verify_assistant_only_mask(item.prepared)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise GateError("Model has no trainable parameters.")
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=0.0,
    )
    collator = AssistantOnlyCollator(processor)
    device = _model_device(model)

    def case_losses() -> list[float]:
        losses: list[float] = []
        model.eval()
        with torch.inference_mode():
            for item in selected:
                batch = {
                    key: value.to(device)
                    for key, value in collator([item.prepared]).items()
                }
                loss = float(_loss_from_output(model(**batch)).detach().cpu())
                if not math.isfinite(loss):
                    raise GateError(
                        f"Targeted case {item.case!r} produced non-finite loss {loss}."
                    )
                losses.append(loss)
        return losses

    initial_losses = case_losses()
    gradient_norms: list[float] = []
    model.train()
    for step in range(steps):
        item = selected[step % len(selected)]
        optimizer.zero_grad(set_to_none=True)
        batch = {
            key: value.to(device)
            for key, value in collator([item.prepared]).items()
        }
        loss = _loss_from_output(model(**batch))
        numeric_loss = float(loss.detach().cpu())
        if not math.isfinite(numeric_loss):
            raise GateError(
                f"Targeted training produced non-finite loss {numeric_loss} "
                f"for case {item.case!r}."
            )
        loss.backward()
        squared_gradient_norm = 0.0
        for parameter in trainable:
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().cpu())
            if not math.isfinite(norm):
                raise GateError(
                    f"Targeted training produced a non-finite gradient for {item.case!r}."
                )
            squared_gradient_norm += norm * norm
        gradient_norm = math.sqrt(squared_gradient_norm)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise GateError(
                f"Targeted training produced a zero or non-finite gradient for {item.case!r}."
            )
        gradient_norms.append(gradient_norm)
        optimizer.step()

    final_losses = case_losses()
    initial_mean = sum(initial_losses) / len(initial_losses)
    final_mean = sum(final_losses) / len(final_losses)
    denominator = max(abs(initial_mean), 1e-12)
    relative_reduction = (initial_mean - final_mean) / denominator
    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, trainable)
    )
    if not changed:
        raise GateError("Targeted tiny-overfit changed no trainable parameter.")
    if relative_reduction < float(minimum_relative_loss_reduction):
        raise GateError(
            "Targeted tiny-overfit loss reduction was not substantial: "
            f"required>={float(minimum_relative_loss_reduction):.3f}, "
            f"observed={relative_reduction:.3f}, "
            f"initial={initial_mean:.6g}, final={final_mean:.6g}."
        )

    case_loss_checks = tuple(
        {
            "case": item.case,
            "example_id": item.example_id,
            "initial_loss": initial,
            "final_loss": final,
            "relative_loss_reduction": (
                (initial - final) / max(abs(initial), 1e-12)
            ),
        }
        for item, initial, final in zip(selected, initial_losses, final_losses)
    )

    generation_checks: list[dict[str, Any]] = []
    for item in selected:
        expected = item.prepared.expected_tool_call
        if expected is None:  # Defensive; selection already rejects this.
            raise GateError(f"Targeted case {item.case!r} has no tool-call target.")
        parsed = run_generation_tool_call_smoke(
            model,
            processor,
            item.prepared,
        )
        generation_checks.append(
            {
                "case": item.case,
                "example_id": item.example_id,
                "expected_tool": expected.name,
                "expected_arguments": dict(expected.arguments),
                "generated_tool": parsed.name,
                "generated_arguments": dict(parsed.arguments),
                "exact_match": parsed == expected,
            }
        )

    return TargetedSmokeResult(
        passed=True,
        steps=int(steps),
        learning_rate=float(learning_rate),
        case_example_ids={item.case: item.example_id for item in selected},
        initial_mean_loss=initial_mean,
        final_mean_loss=final_mean,
        relative_loss_reduction=relative_reduction,
        minimum_relative_loss_reduction=float(minimum_relative_loss_reduction),
        gradients_finite=True,
        gradients_nonzero=True,
        minimum_gradient_norm=min(gradient_norms),
        maximum_gradient_norm=max(gradient_norms),
        parameter_changed=True,
        assistant_only_masks=True,
        case_loss_checks=case_loss_checks,
        generation_checks=tuple(generation_checks),
    )


def _generate_single_tool_call_with_text(
    model: Any,
    processor: Any,
    example: PreparedExample,
    *,
    max_new_tokens: int | None = None,
) -> tuple[ParsedToolCall, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(f"torch is required for generation smoke: {exc}") from exc
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
    return parse_tool_call(text), text


def generate_single_tool_call(
    model: Any,
    processor: Any,
    example: PreparedExample,
    *,
    max_new_tokens: int | None = None,
) -> ParsedToolCall:
    """Greedily generate exactly one parseable tool call."""
    parsed, _text = _generate_single_tool_call_with_text(
        model,
        processor,
        example,
        max_new_tokens=max_new_tokens,
    )
    return parsed


def run_generation_tool_call_smoke(
    model: Any,
    processor: Any,
    example: PreparedExample,
    *,
    max_new_tokens: int | None = None,
) -> ParsedToolCall:
    """Greedily generate and require the expected canonical tool call."""
    expected = example.expected_tool_call
    if expected is None:
        raise GateError("Generation smoke requires a tool-call target example.")
    parsed, text = _generate_single_tool_call_with_text(
        model,
        processor,
        example,
        max_new_tokens=max_new_tokens,
    )
    if parsed != expected:
        raise GateError(
            "Generated tool call does not round-trip to the target: "
            f"expected={expected}, generated={parsed}, text={text[:240]!r}."
        )
    return parsed
