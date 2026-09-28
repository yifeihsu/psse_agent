"""Gemma 4 chat-template text helpers shared by the research policy runtime.

Prompt rendering, tokenization, stop tokens, response decoding and tool-schema
sanitization, moved verbatim from the retired root-level evaluator
(``eval_sft_agent_gemma_v4.py``) and SFT script
(``gpt_oss_power_sft_revised_v3.py``) so that rendering at evaluation time stays
byte-identical to what the adapters were trained on.
"""
from __future__ import annotations

import copy
import json
from typing import Any


GEMMA_TOOL_CALL_CLOSE = "<tool_call|>"
GEMMA_TOOL_RESPONSE_OPEN = "<|tool_response>"
GEMMA_TURN_CLOSE = "<turn|>"
GEMMA_THOUGHT_OPEN = "<|channel>thought"
GEMMA_CHANNEL_CLOSE = "<channel|>"
EMPTY_THOUGHT_CHANNEL = f"{GEMMA_THOUGHT_OPEN}\n{GEMMA_CHANNEL_CLOSE}"


def stringify_message_content_for_template(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    coerced_messages: list[dict[str, Any]] = []
    mutated = False
    for raw_message in messages:
        message = copy.deepcopy(raw_message)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            message["content"] = json.dumps(content, ensure_ascii=False)
            mutated = True
        coerced_messages.append(message)
    return coerced_messages, mutated


def render_text_with_stringified_content_fallback(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    original_exc: Exception,
) -> str:
    coerced_messages, mutated = stringify_message_content_for_template(messages)
    if not mutated:
        raise original_exc
    return tokenizer.apply_chat_template(coerced_messages, **kwargs)


def render_eval_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    enable_thinking: bool,
    inject_empty_thought_channel: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        rendered = render_text_with_stringified_content_fallback(tokenizer, messages, kwargs, exc)

    if inject_empty_thought_channel:
        marker = "<|turn>model\n"
        if marker in rendered:
            pieces: list[str] = []
            cursor = 0
            while True:
                turn_index = rendered.find(marker, cursor)
                if turn_index == -1:
                    pieces.append(rendered[cursor:])
                    break
                model_body_start = turn_index + len(marker)
                pieces.append(rendered[cursor:model_body_start])
                if not rendered.startswith(GEMMA_THOUGHT_OPEN, model_body_start):
                    pieces.append(EMPTY_THOUGHT_CHANNEL)
                cursor = model_body_start
            rendered = "".join(pieces)
    return rendered


def tokenize_rendered_text(tokenizer: Any, prompt: str) -> Any:
    try:
        return tokenizer(text=prompt, return_tensors="pt")
    except TypeError:
        return tokenizer(prompt, return_tensors="pt")


def tokenize_text(tokenizer: Any, text: str, **kwargs: Any) -> Any:
    """Tokenize plain text for both tokenizer and processor-style objects."""
    try:
        return tokenizer(text=text, **kwargs)
    except TypeError:
        return tokenizer(text, **kwargs)


def encode_text(tokenizer: Any, text: str) -> dict[str, list[int]]:
    try:
        encoded = tokenize_text(
            tokenizer,
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
    except TypeError:
        encoded = tokenize_text(
            tokenizer,
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )

    normalized: dict[str, list[int]] = {}
    for key, value in encoded.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(f"Expected a single encoded sample, got batch size {len(value)}")
            normalized[key] = value[0]
        else:
            normalized[key] = value
    return normalized


def _token_id_tokenizer(tokenizer: Any) -> Any:
    """Gemma 4 may load as a Processor; use its inner tokenizer for token-id lookups."""
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        return tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "convert_tokens_to_ids"):
        return inner
    return tokenizer


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    tokenizer = _token_id_tokenizer(tokenizer)
    stop_tokens = [GEMMA_TOOL_CALL_CLOSE, GEMMA_TOOL_RESPONSE_OPEN, GEMMA_TURN_CLOSE]
    stop_ids: list[int] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for token in stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_id:
            stop_ids.append(token_id)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        stop_ids.extend(eos_id)
    elif eos_id is not None:
        stop_ids.append(eos_id)

    out: list[int] = []
    seen: set[int] = set()
    for token_id in stop_ids:
        if token_id not in seen:
            out.append(token_id)
            seen.add(token_id)
    return out


def resolve_pad_token_id(tokenizer: Any) -> int | None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, list):
            pad_token_id = eos_id[0]
        else:
            pad_token_id = eos_id
    return pad_token_id


def trim_trailing_generated_pad_ids(token_ids: Any, pad_token_id: int | None) -> Any:
    if pad_token_id is None:
        return token_ids
    try:
        length = int(token_ids.shape[-1])
    except Exception:
        return token_ids

    end = length
    while end > 0 and int(token_ids[end - 1]) == int(pad_token_id):
        end -= 1
    return token_ids[:end]


def decode_generated_response(
    tokenizer: Any,
    token_ids: Any,
    *,
    pad_token_id: int | None,
) -> tuple[str, int, int]:
    original_count = int(token_ids.shape[-1]) if hasattr(token_ids, "shape") else len(token_ids)
    trimmed_ids = trim_trailing_generated_pad_ids(token_ids, pad_token_id)
    trimmed_count = int(trimmed_ids.shape[-1]) if hasattr(trimmed_ids, "shape") else len(trimmed_ids)
    response_text = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    return response_text, trimmed_count, max(0, original_count - trimmed_count)


def default_schema_description(name: str | None, schema: dict[str, Any]) -> str:
    label = (name or "value").replace("_", " ")
    schema_type = schema.get("type")
    if schema_type == "boolean":
        return f"Whether to set {label}."
    if schema_type == "array":
        return f"List of {label}."
    if schema_type == "object":
        return f"{label.capitalize()} object."
    return f"{label.capitalize()} value."


def fill_schema_descriptions(schema: Any, name: str | None = None) -> Any:
    if isinstance(schema, list):
        return [fill_schema_descriptions(item, name=name) for item in schema]
    if not isinstance(schema, dict):
        return schema

    filled = {key: fill_schema_descriptions(value, name=key) for key, value in schema.items()}
    if any(key in filled for key in ("type", "properties", "items", "anyOf", "oneOf", "allOf")):
        filled.setdefault("description", default_schema_description(name, filled))
    return filled


def sanitize_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue
        fixed_tool = dict(tool)
        function_info = fixed_tool.get("function")
        if isinstance(function_info, dict):
            fixed_function = dict(function_info)
            fixed_function.setdefault(
                "description",
                f"Call the {fixed_function.get('name', 'tool')} tool.",
            )
            parameters = fixed_function.get("parameters")
            if isinstance(parameters, dict):
                fixed_function["parameters"] = fill_schema_descriptions(
                    parameters,
                    name=fixed_function.get("name", "parameters"),
                )
            fixed_tool["function"] = fixed_function
        sanitized.append(fixed_tool)
    return sanitized
