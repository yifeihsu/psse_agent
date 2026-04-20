"""
Gemma 4 evaluator for the fine-tuned PSSE diagnostic agent.

This keeps the hardened runtime/tool loop from the GPT-OSS evaluator, but swaps
the generation parsing logic to Gemma 4's chat-template format. It prefers the
tokenizer's native `parse_response()` helper when available and falls back to a
manual parser for Gemma tool-call blocks when the installed Transformers build
cannot parse them yet.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from trace_protocol import (
    CONTEXT_TOOL_NAMES,
    canonical_tool_schemas,
    extract_conversation_context,
    hydrate_tool_arguments as protocol_hydrate_tool_arguments,
    normalize_instruction_content,
    resolve_case_path_alias,
    summarize_tool_result_for_conversation,
)

from eval_sft_agent_hardened import (
    classify_result_error,
    extract_ground_truth,
    extract_prompt_prefix,
    jsonish_loads,
    normalize_verdict,
    resolve_max_input_tokens,
)
from mcp_server.matpower_server import (
    correct_measurements_from_path,
    correct_parameters_from_path,
    correct_topology_from_path,
    run_hse_from_path,
    wls_from_path,
)


GEMMA_TOOL_CALL_OPEN = "<|tool_call>"
GEMMA_TOOL_CALL_CLOSE = "<tool_call|>"
GEMMA_TOOL_RESPONSE_OPEN = "<|tool_response>"
GEMMA_TOOL_RESPONSE_CLOSE = "<tool_response|>"
GEMMA_TURN_CLOSE = "<turn|>"
GEMMA_THOUGHT_OPEN = "<|channel>thought"
GEMMA_CHANNEL_CLOSE = "<channel|>"
EMPTY_THOUGHT_CHANNEL = f"{GEMMA_THOUGHT_OPEN}\n{GEMMA_CHANNEL_CLOSE}"
GEMMA_QUOTE_TOKEN = '<|"|>'
FINAL_JSON_SCHEMA_MARKER = "Return only strict JSON with this structure:"
PROSE_TOOL_CATALOG_MARKER = "Available tools:"
DECISION_POLICY_MARKER = "Decision policy:"
FIRST_TOOL_PHASE_MESSAGE = (
    "Current phase: first tool selection.\n"
    "Before any tool response exists, emit exactly one native Gemma tool call and nothing else.\n"
    "Do not emit verdict JSON before the first tool response.\n"
    "On the first assistant turn for every snapshot, call `wls_from_path`."
)
LATER_TOOL_PHASE_MESSAGE = (
    "Current phase: intermediate tool use.\n"
    "Emit exactly one native Gemma tool call and nothing else.\n"
    "Do not emit verdict JSON while another tool call is required."
)
FINAL_PHASE_MESSAGE = (
    "Current phase: final answer.\n"
    "Return only the strict verdict JSON. Do not emit a tool call in this phase."
)

TOOL_MAP = {
    "wls_from_path": wls_from_path,
    "correct_measurements_from_path": correct_measurements_from_path,
    "correct_parameters_from_path": correct_parameters_from_path,
    "correct_topology_from_path": correct_topology_from_path,
    "run_hse_from_path": run_hse_from_path,
}
CORE_TOOL_NAMES = frozenset(
    {
        "wls_from_path",
        "correct_measurements_from_path",
        "correct_parameters_from_path",
        "correct_topology_from_path",
        "run_hse_from_path",
    }
)
RUNTIME_HELPER_TOOL_NOTE_PREFIX = "Runtime helper tools available for this snapshot:"
TOKENIZER_ARTIFACT_FILENAMES = frozenset(
    {
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "added_tokens.json",
        "processor_config.json",
        "chat_template.jinja",
    }
)


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_message_content(value: Any, *, parse_json_strings: bool) -> Any:
    if value is None:
        return ""
    parsed = maybe_parse_json_string(value) if parse_json_strings else value
    if isinstance(parsed, (dict, list)):
        return parsed
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, ensure_ascii=False)


def normalize_assistant_tool_responses(tool_responses: Any) -> list[dict[str, Any]] | Any:
    if not isinstance(tool_responses, list):
        return tool_responses

    normalized: list[dict[str, Any]] = []
    for tool_response in tool_responses:
        item = copy.deepcopy(tool_response)
        if not isinstance(item, dict):
            normalized.append(
                {
                    "name": "unknown",
                    "response": normalize_message_content(item, parse_json_strings=True),
                }
            )
            continue

        fixed = dict(item)
        response_value = None
        for key in ("response", "result", "content"):
            if key in fixed:
                response_value = fixed.get(key)
                break
        if response_value is not None:
            fixed["response"] = normalize_message_content(response_value, parse_json_strings=True)
        for stale_key in ("result", "content"):
            fixed.pop(stale_key, None)

        if not isinstance(fixed.get("name"), str) or not fixed.get("name"):
            for alt_key in ("tool_name", "function_name"):
                alt_value = fixed.get(alt_key)
                if isinstance(alt_value, str) and alt_value:
                    fixed["name"] = alt_value
                    break
        fixed.setdefault("name", "unknown")
        normalized.append(fixed)
    return normalized


def normalize_tool_role_content(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if all(isinstance(part, dict) and "type" in part for part in value):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def strip_prose_tool_catalog(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    start = text.find(PROSE_TOOL_CATALOG_MARKER)
    if start == -1:
        return text
    end = text.find(DECISION_POLICY_MARKER, start)
    if end == -1:
        return text
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    if before and after:
        return f"{before}\n\n{after}"
    return before or after


def strip_prose_tool_catalog_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for raw_message in messages:
        message = copy.deepcopy(raw_message)
        if message.get("role") in {"system", "developer"}:
            message["content"] = strip_prose_tool_catalog(message.get("content"))
        stripped.append(message)
    return stripped


def resolve_tool_response_name(tool_message: dict[str, Any], assistant_tool_calls: list[dict[str, Any]]) -> str:
    explicit_name = tool_message.get("name")
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name

    tool_call_id = tool_message.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        for tool_call in assistant_tool_calls:
            if tool_call.get("id") != tool_call_id:
                continue
            function_info = tool_call.get("function")
            if isinstance(function_info, dict):
                resolved_name = function_info.get("name")
                if isinstance(resolved_name, str) and resolved_name:
                    return resolved_name
    return "unknown"


def collapse_openai_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(messages):
        message = copy.deepcopy(messages[cursor])
        role = message.get("role")
        tool_calls = message.get("tool_calls")

        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            merged_tool_responses = normalize_assistant_tool_responses(message.get("tool_responses") or [])
            if not isinstance(merged_tool_responses, list):
                merged_tool_responses = []

            scan = cursor + 1
            found_following_tool = False
            while scan < len(messages) and messages[scan].get("role") == "tool":
                tool_message = messages[scan]
                merged_tool_responses.append(
                    {
                        "name": resolve_tool_response_name(tool_message, tool_calls),
                        "response": normalize_message_content(
                            tool_message.get("content", ""),
                            parse_json_strings=True,
                        ),
                    }
                )
                found_following_tool = True
                scan += 1

            if found_following_tool:
                message["tool_responses"] = normalize_assistant_tool_responses(merged_tool_responses)
                collapsed.append(message)
                cursor = scan
                continue

        if role == "tool":
            message["content"] = normalize_tool_role_content(message.get("content", ""))
        collapsed.append(message)
        cursor += 1
    return collapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Gemma 4 PSSE agent")
    p.add_argument(
        "--adapter",
        default="outputs/gemma4_power_agent/lora",
        help="Path to the LoRA adapter directory",
    )
    p.add_argument(
        "--test-file",
        default="out_traces_balanced/sft_traces.test.jsonl",
        help="JSONL file with test conversations",
    )
    p.add_argument(
        "--model-revision",
        default="",
        help="Optional pinned base-model revision to match the revision used during SFT.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on the number of samples to evaluate",
    )
    p.add_argument(
        "--concurrent-conversations",
        type=int,
        default=1,
        help="Number of conversations to advance together in one batched generation step.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Max generation turns before forcing stop. Set high enough for 5 tool calls plus a final verdict.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max tokens per generation step",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum input context length to render into the model",
    )
    p.add_argument(
        "--output",
        default="outputs/gemma4_power_agent/eval_gemma4_results.jsonl",
        help="Where to write per-sample results",
    )
    p.add_argument(
        "--continue-on-tool-error",
        action="store_true",
        help="Let the model continue after a tool returns success=false (default: stop and mark runtime error)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print every generation step",
    )
    p.add_argument(
        "--include-tool-schemas",
        action="store_true",
        default=True,
        help="Pass the same tool schemas used during SFT into apply_chat_template",
    )
    p.add_argument(
        "--no-include-tool-schemas",
        dest="include_tool_schemas",
        action="store_false",
        help="Do not pass tool schemas to apply_chat_template",
    )
    p.add_argument(
        "--tools-file",
        default="",
        help="Optional JSON file containing a list of tool schemas",
    )
    p.add_argument(
        "--repair-wls-from-user",
        action="store_true",
        default=True,
        help="If a parsed wls_from_path call is missing/short z, repair it from the user's z_obs",
    )
    p.add_argument(
        "--no-repair-wls-from-user",
        dest="repair_wls_from_user",
        action="store_false",
        help="Disable auto-repair of malformed wls_from_path arguments from the user payload",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable Gemma thinking mode during evaluation",
    )
    p.add_argument(
        "--inject-empty-thought-channel",
        action="store_true",
        default=True,
        help="Mirror SFT formatting by inserting an empty Gemma thought channel after each rendered model-turn open.",
    )
    p.add_argument(
        "--no-inject-empty-thought-channel",
        dest="inject_empty_thought_channel",
        action="store_false",
        help="Disable empty-thought injection even when the adapter was trained with it.",
    )
    p.add_argument(
        "--prefer-base-tokenizer",
        action="store_true",
        default=True,
        help="Prefer the pinned base tokenizer over tokenizer files saved inside the adapter directory.",
    )
    p.add_argument(
        "--prefer-adapter-tokenizer",
        dest="prefer_base_tokenizer",
        action="store_false",
        help="Prefer adapter-local tokenizer files when available.",
    )
    p.add_argument(
        "--filter-unavailable-helper-tools",
        action="store_true",
        default=True,
        help="Hide helper tool schemas that are not backed by runtime context for the current sample.",
    )
    p.add_argument(
        "--no-filter-unavailable-helper-tools",
        dest="filter_unavailable_helper_tools",
        action="store_false",
        help="Always expose every helper tool schema, even when a sample lacks that runtime context.",
    )
    p.add_argument(
        "--continue-on-missing-context-tool",
        action="store_true",
        default=True,
        help="If the model calls a helper tool without runtime context, feed the tool error back and let the conversation continue.",
    )
    p.add_argument(
        "--no-continue-on-missing-context-tool",
        dest="continue_on_missing_context_tool",
        action="store_false",
        help="Treat missing runtime context helper-tool calls as terminal errors.",
    )
    p.add_argument(
        "--rolling-batch-scheduler",
        action="store_true",
        default=True,
        help="Keep the batched evaluator full by backfilling finished conversations with new samples.",
    )
    p.add_argument(
        "--no-rolling-batch-scheduler",
        dest="rolling_batch_scheduler",
        action="store_false",
        help="Disable the rolling scheduler and evaluate in fixed microbatches.",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load the base model in 4-bit mode for evaluation",
    )
    p.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading",
    )
    p.add_argument(
        "--load-in-16bit",
        action="store_true",
        default=True,
        help="Load the base model in bf16/16-bit mode for evaluation",
    )
    p.add_argument(
        "--no-load-in-16bit",
        dest="load_in_16bit",
        action="store_false",
        help="Disable 16-bit loading",
    )
    p.add_argument(
        "--gc-collect-every-n-turns",
        type=int,
        default=0,
        help="Run gc.collect() every N generation turns. 0 disables periodic collection.",
    )
    p.add_argument(
        "--empty-cuda-cache-every-n-turns",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N generation turns. 0 disables periodic cache flushes.",
    )
    return p.parse_args()


def normalize_gemma_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = copy.deepcopy(raw_message)
        role = msg.get("role")
        if role is None:
            continue

        if role in {"system", "developer"}:
            content = msg.get("content", "")
            content = normalize_instruction_content(role, content)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            msg["content"] = content

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                fixed_calls = []
                for tool_call in tool_calls:
                    tc = copy.deepcopy(tool_call)
                    function_info = tc.get("function")
                    if isinstance(function_info, dict):
                        arguments = function_info.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                pass
                        if arguments is None:
                            arguments = {}
                        function_info["arguments"] = arguments
                        tc["function"] = function_info
                    fixed_calls.append(tc)
                msg["tool_calls"] = fixed_calls
            else:
                msg.pop("tool_calls", None)
            tool_responses = msg.get("tool_responses")
            if tool_responses is not None:
                msg["tool_responses"] = normalize_assistant_tool_responses(tool_responses)
            if "content" in msg or (not msg.get("tool_calls") and not msg.get("tool_responses")):
                msg["content"] = normalize_message_content(msg.get("content", ""), parse_json_strings=False)
        elif role == "tool":
            msg["content"] = normalize_tool_role_content(msg.get("content", ""))
        elif role in {"user", "system", "developer"}:
            msg["content"] = normalize_message_content(msg.get("content", ""), parse_json_strings=False)

        normalized.append(msg)
    return normalized


def adapter_tokenizer_files(adapter_path: Path) -> list[str]:
    if not adapter_path.is_dir():
        return []
    return sorted(
        child.name for child in adapter_path.iterdir() if child.name in TOKENIZER_ARTIFACT_FILENAMES
    )


def prepare_unsloth_adapter_path(
    adapter_path: Path,
    *,
    prefer_base_tokenizer: bool,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None, list[str]]:
    tokenizer_files = adapter_tokenizer_files(adapter_path)
    if not prefer_base_tokenizer or not tokenizer_files:
        return str(adapter_path), None, tokenizer_files

    tempdir = tempfile.TemporaryDirectory(prefix="unsloth_adapter_notokenizer_")
    temp_path = Path(tempdir.name)
    for child in adapter_path.iterdir():
        if child.name in TOKENIZER_ARTIFACT_FILENAMES:
            continue
        target = temp_path / child.name
        try:
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        except Exception:
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)
    return str(temp_path), tempdir, tokenizer_files



def extract_tool_schema_name(schema: Mapping[str, Any]) -> str | None:
    if not isinstance(schema, Mapping):
        return None
    function_info = schema.get("function")
    if isinstance(function_info, Mapping):
        name = function_info.get("name")
        if isinstance(name, str) and name:
            return name
    name = schema.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def available_helper_tools_from_runtime_context(runtime_context: Mapping[str, Any] | None) -> list[str]:
    tool_context = ((runtime_context or {}).get("tool_context") or {})
    available: list[str] = []
    if isinstance(tool_context.get("parameter_context"), Mapping):
        available.append("get_parameter_context")
    if isinstance(tool_context.get("topology_context"), Mapping):
        available.append("get_topology_context")
    if isinstance(tool_context.get("harmonic_context"), Mapping):
        available.append("get_harmonic_context")
    verification_snapshots = tool_context.get("verification_snapshots")
    if isinstance(verification_snapshots, Mapping) and verification_snapshots:
        available.append("get_verification_snapshot")
    return available


def filter_tool_schemas_for_runtime_context(
    tools: list[dict[str, Any]] | None,
    runtime_context: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if tools is None:
        return None

    allowed_names = set(CORE_TOOL_NAMES)
    allowed_names.update(available_helper_tools_from_runtime_context(runtime_context))

    filtered: list[dict[str, Any]] = []
    for schema in tools:
        name = extract_tool_schema_name(schema)
        if name is None or name in allowed_names:
            filtered.append(schema)
    return filtered


def inject_runtime_tool_availability_note(
    messages: list[dict[str, Any]],
    runtime_context: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    available_helpers = available_helper_tools_from_runtime_context(runtime_context)
    helper_text = ", ".join(available_helpers) if available_helpers else "none"
    note = (
        f"{RUNTIME_HELPER_TOOL_NOTE_PREFIX} {helper_text}. "
        "Do not call helper tools that are not listed here."
    )

    injected: list[dict[str, Any]] = []
    merged = False
    for raw_message in messages:
        message = copy.deepcopy(raw_message)
        if not merged and message.get("role") in {"system", "developer"}:
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if RUNTIME_HELPER_TOOL_NOTE_PREFIX not in content:
                content = f"{content.rstrip()}\n\n{note}".strip() if content else note
            message["content"] = content
            merged = True
        injected.append(message)

    if not merged:
        injected.insert(0, {"role": "system", "content": note})
    return injected


def is_missing_runtime_context_tool_error(tool_result: Mapping[str, Any] | None) -> bool:
    if not isinstance(tool_result, Mapping):
        return False
    if tool_result.get("success") is not False:
        return False
    error = tool_result.get("error")
    return isinstance(error, str) and error.startswith("Missing runtime context for ")


def strip_final_json_schema(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    marker_index = text.find(FINAL_JSON_SCHEMA_MARKER)
    if marker_index == -1:
        return text
    return text[:marker_index].rstrip()


def infer_expected_phase(messages_gt: list[dict[str, Any]], conversation: list[dict[str, Any]]) -> str:
    gt_assistants = [message for message in messages_gt if message.get("role") == "assistant"]
    produced_assistants = sum(1 for message in conversation if message.get("role") == "assistant")
    if produced_assistants >= len(gt_assistants):
        return "final"
    next_assistant = gt_assistants[produced_assistants]
    if next_assistant.get("tool_calls"):
        return "first_tool_call" if produced_assistants == 0 else "later_tool_call"
    return "final"


def apply_phase_gating(messages: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    gated = copy.deepcopy(messages)
    if phase in {"first_tool_call", "later_tool_call"}:
        for message in gated:
            if message.get("role") in {"system", "developer"}:
                message["content"] = strip_final_json_schema(message.get("content"))
        phase_message = (
            FIRST_TOOL_PHASE_MESSAGE if phase == "first_tool_call" else LATER_TOOL_PHASE_MESSAGE
        )
    else:
        phase_message = FINAL_PHASE_MESSAGE

    merged: list[dict[str, Any]] = []
    merged_phase_instruction = False
    for message in gated:
        role = message.get("role")
        if role == "developer":
            message["role"] = "system"
            role = "system"

        if not merged_phase_instruction and role == "system":
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if phase_message not in content:
                content = f"{content.rstrip()}\n\n{phase_message}".strip() if content else phase_message
            message["content"] = content
            merged_phase_instruction = True

        merged.append(message)

    if not merged_phase_instruction:
        merged.insert(0, {"role": "system", "content": phase_message})
    return merged


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
        marker = f"<|turn>model\n"
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


def build_model_inputs(
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    *,
    max_input_tokens: int,
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
    phase: str,
    inject_empty_thought_channel: bool,
    runtime_context: Mapping[str, Any] | None = None,
    filter_unavailable_helper_tools: bool = True,
) -> tuple[dict[str, Any], bool]:
    effective_tools = (
        filter_tool_schemas_for_runtime_context(tools, runtime_context)
        if filter_unavailable_helper_tools
        else tools
    )
    rendered_conversation = collapse_openai_tool_messages(copy.deepcopy(conversation))
    if effective_tools is not None:
        rendered_conversation = strip_prose_tool_catalog_from_messages(rendered_conversation)
    rendered_conversation = normalize_gemma_messages(apply_phase_gating(rendered_conversation, phase))
    rendered_conversation = inject_runtime_tool_availability_note(
        rendered_conversation,
        runtime_context,
    )

    prompt = render_eval_text(
        tokenizer,
        rendered_conversation,
        effective_tools,
        enable_thinking=enable_thinking,
        inject_empty_thought_channel=inject_empty_thought_channel,
    )
    inputs = tokenize_rendered_text(tokenizer, prompt)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    truncated = False

    if max_input_tokens is not None and input_ids.shape[-1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -max_input_tokens:]
        truncated = True

    model_inputs = {"input_ids": input_ids.to(model.device)}
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask.to(model.device)
    return model_inputs, truncated


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


def extract_gemma_thinking(text: str) -> tuple[str, str | None]:
    remaining = text.lstrip()
    chunks: list[str] = []

    while remaining.startswith(GEMMA_THOUGHT_OPEN):
        remaining = remaining[len(GEMMA_THOUGHT_OPEN):]
        if remaining.startswith("\n"):
            remaining = remaining[1:]
        close_index = remaining.find(GEMMA_CHANNEL_CLOSE)
        if close_index == -1:
            break
        chunk = remaining[:close_index].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[close_index + len(GEMMA_CHANNEL_CLOSE):].lstrip()

    thinking = "\n\n".join(chunks).strip() or None
    return remaining, thinking


def strip_trailing_gemma_tokens(text: str) -> str:
    stripped = text.strip()
    for token in (GEMMA_TURN_CLOSE, GEMMA_TOOL_CALL_CLOSE, GEMMA_TOOL_RESPONSE_CLOSE):
        if stripped.endswith(token):
            stripped = stripped[: -len(token)].rstrip()
    return stripped


def quote_unquoted_object_keys(text: str) -> str:
    return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)


def parse_gemma_argument_object(payload: str) -> dict[str, Any]:
    candidate = payload.strip()
    candidate = strip_trailing_gemma_tokens(candidate)
    candidate = candidate.replace(GEMMA_QUOTE_TOKEN, '"')
    candidate = quote_unquoted_object_keys(candidate)
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict-like tool arguments, got {type(obj).__name__}")
    return obj


def manual_parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    match = re.search(
        r"<\|tool_call\>\s*call:([A-Za-z_][A-Za-z0-9_]*)\s*(.*?)\s*(?:<tool_call\|>|$)",
        text,
        re.DOTALL,
    )
    if not match:
        return None

    tool_name = match.group(1)
    arguments_text = match.group(2).strip()
    if not arguments_text.startswith("{"):
        raise ValueError(f"Gemma tool call for {tool_name} is missing an argument object")
    return tool_name, parse_gemma_argument_object(arguments_text)


def extract_json_wrapped_tool_call(obj: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None

    single_call = obj.get("tool_call")
    if isinstance(single_call, dict):
        obj = single_call

    direct_tool_name = obj.get("tool_name")
    direct_arguments = obj.get("arguments")
    if isinstance(direct_tool_name, str):
        arguments = direct_arguments if direct_arguments is not None else {}
        if isinstance(arguments, str):
            arguments = jsonish_loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError(
                f"JSON-wrapped tool arguments are not a dict for {direct_tool_name!r}: {type(arguments).__name__}"
            )
        return direct_tool_name, arguments

    tool_calls = obj.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        if isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            return obj["name"], obj["arguments"]
        return None

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        raise ValueError(f"JSON-wrapped tool call must be an object, got {type(first_call).__name__}")

    function_field = first_call.get("function")
    arguments: Any = None
    tool_name: Any = None

    if isinstance(function_field, dict):
        tool_name = function_field.get("name")
        arguments = function_field.get("arguments")
    elif isinstance(function_field, str):
        tool_name = function_field
        arguments = first_call.get("args", first_call.get("arguments"))
    else:
        tool_name = first_call.get("name")
        arguments = first_call.get("args", first_call.get("arguments"))

    if isinstance(arguments, str):
        arguments = jsonish_loads(arguments)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError(
            f"JSON-wrapped tool arguments are not a dict for {tool_name!r}: {type(arguments).__name__}"
        )
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("JSON-wrapped tool call is missing a function name.")
    return tool_name, arguments


def extract_tool_call_from_jsonish_text(text: str) -> tuple[str, dict[str, Any]] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    name_patterns = [
        r'"tool_name"\s*:\s*"([^"]+)"',
        r'"function"\s*:\s*"([^"]+)"',
        r'"name"\s*:\s*"([^"]+)"',
    ]
    tool_name: str | None = None
    for pattern in name_patterns:
        match = re.search(pattern, cleaned)
        if match:
            tool_name = match.group(1)
            break
    if tool_name is None:
        return None

    arguments: dict[str, Any] = {}
    case_match = re.search(r'"case_path"\s*:\s*"([^"]+)"', cleaned)
    if case_match:
        arguments["case_path"] = case_match.group(1)

    line_match = re.search(r'"line_index"\s*:\s*([0-9]+)', cleaned)
    if line_match:
        arguments["line_index"] = int(line_match.group(1))

    cb_match = re.search(r'"cb_name"\s*:\s*"([^"]+)"', cleaned)
    if cb_match:
        arguments["cb_name"] = cb_match.group(1)

    status_match = re.search(r'"desired_status"\s*:\s*(true|false)', cleaned, flags=re.IGNORECASE)
    if status_match:
        arguments["desired_status"] = status_match.group(1).lower() == "true"

    return tool_name, arguments


def parse_gemma_generation(text: str, tokenizer: Any) -> dict[str, Any]:
    raw = text.strip()
    notes: list[str] = []
    parsed: dict[str, Any] | None = None

    try:
        native = tokenizer.parse_response(raw)
    except Exception:
        native = None
        if GEMMA_TOOL_CALL_OPEN in raw:
            notes.append("tokenizer_parse_response_tool_call_fallback")
        else:
            notes.append("tokenizer_parse_response_failed")

    if isinstance(native, dict):
        parsed = native

    if parsed and parsed.get("tool_calls"):
        tool_calls = parsed.get("tool_calls") or []
        if len(tool_calls) > 1:
            notes.append("multiple_tool_calls_generated")
        first_call = tool_calls[0]
        function_info = first_call.get("function", {})
        arguments = function_info.get("arguments", {})
        if isinstance(arguments, str):
            arguments = jsonish_loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError(f"Parsed tool arguments are not a dict: {type(arguments).__name__}")
        return {
            "type": "tool_call",
            "name": function_info.get("name"),
            "arguments": arguments,
            "thinking": parsed.get("thinking"),
            "notes": notes,
            "raw": raw[:1000],
        }

    body, manual_thinking = extract_gemma_thinking(raw)
    body = strip_trailing_gemma_tokens(body)

    if GEMMA_TOOL_CALL_OPEN in body:
        try:
            manual_result = manual_parse_tool_call(body)
        except (ValueError, json.JSONDecodeError):
            manual_result = None
            notes.append("manual_tool_call_parse_failed")
        tool_name, arguments = manual_result or (None, None)
        if tool_name is not None and isinstance(arguments, dict):
            notes.append("manual_tool_call_parser")
            return {
                "type": "tool_call",
                "name": tool_name,
                "arguments": arguments,
                "thinking": parsed.get("thinking") if parsed else manual_thinking,
                "notes": notes,
                "raw": raw[:1000],
            }

    content = None
    if parsed and isinstance(parsed.get("content"), str):
        content = parsed["content"]
    elif body:
        content = body

    if content:
        try:
            obj = jsonish_loads(content)
            wrapped_tool_call = extract_json_wrapped_tool_call(obj)
            if wrapped_tool_call is not None:
                tool_name, arguments = wrapped_tool_call
                notes.append("json_wrapped_tool_call_fallback")
                return {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": arguments,
                    "thinking": parsed.get("thinking") if parsed else manual_thinking,
                    "notes": notes,
                    "raw": raw[:1000],
                }
            if isinstance(obj, dict) and "verdict" in obj:
                return {
                    "type": "verdict",
                    "content": obj,
                    "thinking": parsed.get("thinking") if parsed else manual_thinking,
                    "notes": notes,
                    "raw": raw[:1000],
                }
            if isinstance(obj, dict):
                notes.append("json_object_without_verdict")
        except Exception:
            notes.append("verdict_json_parse_failed")
            regex_tool_call = extract_tool_call_from_jsonish_text(content)
            if regex_tool_call is not None:
                tool_name, arguments = regex_tool_call
                notes.append("regex_json_tool_call_fallback")
                return {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": arguments,
                    "thinking": parsed.get("thinking") if parsed else manual_thinking,
                    "notes": notes,
                    "raw": raw[:1000],
                }

    return {
        "type": "unparseable",
        "thinking": parsed.get("thinking") if parsed else manual_thinking,
        "notes": notes,
        "raw": raw[:1000],
    }


def resolve_turn_max_new_tokens(turn_index0: int, default_max_new_tokens: int) -> int:
    if turn_index0 == 0:
        return default_max_new_tokens
    return min(default_max_new_tokens, 768)


def maybe_run_turn_housekeeping(
    turn_index0: int,
    *,
    gc_collect_every_n_turns: int,
    empty_cuda_cache_every_n_turns: int,
) -> None:
    should_collect = gc_collect_every_n_turns > 0 and (turn_index0 + 1) % gc_collect_every_n_turns == 0
    should_empty_cache = (
        empty_cuda_cache_every_n_turns > 0
        and (turn_index0 + 1) % empty_cuda_cache_every_n_turns == 0
    )
    if not should_collect and not should_empty_cache:
        return

    if should_empty_cache:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if should_collect:
        gc.collect()


def format_preview(value: Any, limit: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_verbose_generation_block(
    turn: int,
    token_count: int,
    response_text: str,
    *,
    was_truncated: bool,
) -> None:
    print(f"\n  ======== [Turn {turn}] Generated ({token_count} tokens) ========")
    lines = response_text.splitlines() or [""]
    for line in lines:
        print(f"  {line}")
    if was_truncated:
        print("  [prompt was left-truncated to fit the context window]")
    print("  ======================================================\n")


def load_tools(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not args.include_tool_schemas:
        return None
    if args.tools_file:
        with open(args.tools_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("--tools-file must contain a JSON list of tool schemas.")
        return data
    return canonical_tool_schemas()


def repair_tool_arguments(
    tool_name: str,
    arguments: Any,
    conversation: list[dict[str, Any]],
    *,
    hidden_context: Mapping[str, Any] | None = None,
) -> tuple[Any, list[str]]:
    return protocol_hydrate_tool_arguments(tool_name, arguments, conversation, hidden_context=hidden_context)


def execute_context_tool(
    name: str,
    arguments: dict[str, Any],
    runtime_context: Mapping[str, Any] | None,
    hidden_context: dict[str, Any],
) -> dict[str, Any]:
    tool_context = ((runtime_context or {}).get("tool_context") or {})
    payload: Any = None
    if name == "get_parameter_context":
        payload = tool_context.get("parameter_context")
        if isinstance(payload, dict):
            hidden_context["parameter_context"] = payload
    elif name == "get_topology_context":
        payload = tool_context.get("topology_context")
        if isinstance(payload, dict):
            hidden_context["topology_context"] = payload
    elif name == "get_harmonic_context":
        payload = tool_context.get("harmonic_context")
        if isinstance(payload, dict):
            hidden_context["harmonic_context"] = payload
    elif name == "get_verification_snapshot":
        stage = arguments.get("stage")
        payload = (tool_context.get("verification_snapshots") or {}).get(stage)
        if isinstance(payload, dict):
            hidden_context["snapshot_context"] = payload
    if isinstance(payload, dict):
        return payload
    return {"success": False, "error": f"Missing runtime context for {name}", "available_context_tools": available_helper_tools_from_runtime_context(runtime_context)}


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    runtime_context: Mapping[str, Any] | None = None,
    hidden_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name in CONTEXT_TOOL_NAMES:
        return execute_context_tool(name, arguments, runtime_context, hidden_context or {})

    tool_obj = TOOL_MAP.get(name)
    if tool_obj is None:
        return {"success": False, "error": f"Unknown tool: {name}"}
    try:
        fn = getattr(tool_obj, "fn", tool_obj)
        call_args = dict(arguments)
        if "case_path" in call_args:
            call_args["case_path"] = resolve_case_path_alias(call_args["case_path"], hidden_context or runtime_context)
        return fn(**call_args)
    except Exception as exc:  # pragma: no cover - defensive runtime wrapper
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def compact_tool_arguments_for_prompt(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "wls_from_path": {"case_path"},
        "get_parameter_context": {"case_path", "line_index"},
        "get_topology_context": {"case_path"},
        "get_harmonic_context": {"case_path"},
        "get_verification_snapshot": {"case_path", "stage"},
        "correct_measurements_from_path": {
            "case_path",
            "suspect_group",
            "enable_correction",
            "max_correction_iterations",
            "error_tolerance",
        },
        "correct_parameters_from_path": {"case_path", "line_index"},
        "correct_topology_from_path": {"case_path", "cb_name", "desired_status"},
        "run_hse_from_path": {"case_path"},
    }.get(tool_name)
    if keep_keys is None:
        return dict(arguments)
    return {key: value for key, value in arguments.items() if key in keep_keys and value is not None}


@dataclass
class EvalSampleState:
    sample_index: int
    messages_gt: list[dict[str, Any]]
    runtime_context: Mapping[str, Any] | None
    gt_verdict: dict[str, Any] | None
    conversation: list[dict[str, Any]]
    meta: dict[str, Any]
    index_map: dict[str, Any]
    hidden_context: dict[str, Any]
    tool_calls_made: list[str] = field(default_factory=list)
    parse_notes: list[str] = field(default_factory=list)
    predicted_verdict: dict[str, Any] | None = None
    error_msg: str | None = None
    input_truncated: bool = False
    last_raw_generation: str | None = None
    wls_completed_successfully: bool = False
    turn_trace: list[dict[str, Any]] = field(default_factory=list)
    turn_index0: int = 0

    @property
    def finished(self) -> bool:
        return self.error_msg is not None or self.predicted_verdict is not None


def init_eval_sample_state(
    sample_index: int,
    messages_gt: list[dict[str, Any]],
    runtime_context: Mapping[str, Any] | None,
) -> EvalSampleState:
    gt_verdict = normalize_verdict(extract_ground_truth(messages_gt))
    conversation = extract_prompt_prefix(messages_gt)
    meta, index_map = extract_conversation_context(messages_gt)
    hidden_context = dict(runtime_context or {})
    return EvalSampleState(
        sample_index=sample_index,
        messages_gt=messages_gt,
        runtime_context=runtime_context,
        gt_verdict=gt_verdict,
        conversation=conversation,
        meta=meta,
        index_map=index_map,
        hidden_context=hidden_context,
    )



def summarize_turn_trace_runtime(turn_trace: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens_total = sum(int(turn.get("prompt_tokens", 0) or 0) for turn in turn_trace)
    generated_tokens_total = sum(int(turn.get("generated_tokens", 0) or 0) for turn in turn_trace)
    model_generate_seconds_total = sum(float(turn.get("model_generate_seconds", 0.0) or 0.0) for turn in turn_trace)
    tool_seconds_total = sum(float(turn.get("tool_seconds", 0.0) or 0.0) for turn in turn_trace)
    max_prompt_tokens = max((int(turn.get("prompt_tokens", 0) or 0) for turn in turn_trace), default=0)
    model_tokens_per_second = None
    if model_generate_seconds_total > 0:
        model_tokens_per_second = generated_tokens_total / model_generate_seconds_total
    return {
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
        "max_prompt_tokens": max_prompt_tokens,
        "model_generate_seconds_total": round(model_generate_seconds_total, 6),
        "tool_seconds_total": round(tool_seconds_total, 6),
        "estimated_model_tokens_per_second": (
            round(model_tokens_per_second, 3) if model_tokens_per_second is not None else None
        ),
    }


def build_result_from_state(state: EvalSampleState) -> dict[str, Any]:
    gt_family = state.gt_verdict.get("verdict", {}).get("error_family") if state.gt_verdict else None
    pred_family = state.predicted_verdict.get("verdict", {}).get("error_family") if state.predicted_verdict else None

    gt_has_error = state.gt_verdict.get("verdict", {}).get("has_error") if state.gt_verdict else None
    pred_has_error = state.predicted_verdict.get("verdict", {}).get("has_error") if state.predicted_verdict else None

    return {
        "gt_error_family": gt_family,
        "pred_error_family": pred_family,
        "gt_has_error": gt_has_error,
        "pred_has_error": pred_has_error,
        "family_correct": gt_family == pred_family,
        "detection_correct": gt_has_error == pred_has_error,
        "tool_calls": state.tool_calls_made,
        "num_turns": len(state.tool_calls_made) + (1 if state.predicted_verdict else 0),
        "predicted_verdict": state.predicted_verdict,
        "error": state.error_msg,
        "input_truncated": state.input_truncated,
        "parse_notes": state.parse_notes,
        "last_raw_generation": state.last_raw_generation,
        "required_wls_satisfied": state.wls_completed_successfully,
        "runtime": summarize_turn_trace_runtime(state.turn_trace),
        "turn_trace": state.turn_trace,
        "final_conversation": state.conversation,
    }


def build_critical_error_result(messages_gt: list[dict[str, Any]], exc: Exception) -> dict[str, Any]:
    gt_verdict = normalize_verdict(extract_ground_truth(messages_gt))
    return {
        "gt_error_family": gt_verdict.get("verdict", {}).get("error_family") if gt_verdict else None,
        "pred_error_family": None,
        "gt_has_error": gt_verdict.get("verdict", {}).get("has_error") if gt_verdict else None,
        "pred_has_error": None,
        "family_correct": False,
        "detection_correct": False,
        "tool_calls": [],
        "num_turns": 0,
        "predicted_verdict": None,
        "error": f"CRITICAL ERROR: {exc}",
        "input_truncated": False,
        "parse_notes": [],
        "last_raw_generation": None,
        "runtime": {},
    }


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

def pad_model_inputs_for_batch(
    model_inputs_list: list[dict[str, Any]],
    *,
    pad_token_id: int | None,
) -> tuple[dict[str, Any], int]:
    import torch

    max_len = max(model_inputs["input_ids"].shape[-1] for model_inputs in model_inputs_list)
    device = model_inputs_list[0]["input_ids"].device
    dtype = model_inputs_list[0]["input_ids"].dtype
    effective_pad_token_id = 0 if pad_token_id is None else pad_token_id

    batch_size = len(model_inputs_list)
    batched_input_ids = torch.full(
        (batch_size, max_len),
        effective_pad_token_id,
        dtype=dtype,
        device=device,
    )

    has_attention = any("attention_mask" in model_inputs for model_inputs in model_inputs_list)
    batched_attention_mask = None
    if has_attention:
        batched_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=model_inputs_list[0].get("attention_mask", model_inputs_list[0]["input_ids"]).dtype,
            device=device,
        )

    for row_index, model_inputs in enumerate(model_inputs_list):
        input_ids = model_inputs["input_ids"][0]
        length = input_ids.shape[-1]
        batched_input_ids[row_index, -length:] = input_ids
        if batched_attention_mask is not None:
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None:
                batched_attention_mask[row_index, -length:] = 1
            else:
                batched_attention_mask[row_index, -length:] = attention_mask[0]

    batched_inputs = {"input_ids": batched_input_ids}
    if batched_attention_mask is not None:
        batched_inputs["attention_mask"] = batched_attention_mask
    return batched_inputs, max_len


def run_state_turn(
    state: EvalSampleState,
    *,
    turn_index0: int,
    response_text: str,
    token_count: int,
    was_truncated: bool,
    turn_max_new_tokens: int,
    tokenizer: Any,
    continue_on_tool_error: bool,
    continue_on_missing_context_tool: bool,
    repair_wls_from_user: bool,
    verbose: bool,
    prompt_tokens: int,
    model_generate_seconds: float,
) -> None:
    state.last_raw_generation = response_text
    turn_record: dict[str, Any] = {
        "turn": turn_index0 + 1,
        "raw_generation": response_text,
        "prompt_truncated": was_truncated,
        "turn_max_new_tokens": turn_max_new_tokens,
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(token_count),
        "model_generate_seconds": round(float(model_generate_seconds), 6),
    }

    if verbose:
        print_verbose_generation_block(
            turn_index0 + 1,
            token_count,
            response_text,
            was_truncated=was_truncated,
        )

    parsed = parse_gemma_generation(response_text, tokenizer)
    state.parse_notes.extend(parsed.get("notes", []))
    turn_record["parse_type"] = parsed["type"]
    if parsed.get("notes"):
        turn_record["parse_notes"] = list(parsed["notes"])
    if parsed.get("thinking"):
        turn_record["thinking"] = parsed["thinking"]

    if parsed["type"] == "tool_call":
        tool_name = parsed["name"]
        tool_args = parsed["arguments"]
        if repair_wls_from_user:
            tool_args, repair_notes = repair_tool_arguments(
                tool_name,
                tool_args,
                state.conversation,
                hidden_context=state.hidden_context,
            )
            if repair_notes:
                state.parse_notes.extend(repair_notes)
                turn_record.setdefault("parse_notes", [])
                turn_record["parse_notes"].extend(repair_notes)

        if not isinstance(tool_args, dict):
            state.error_msg = f"Parsed tool arguments are not a dict for {tool_name}: {type(tool_args).__name__}"
            turn_record["error"] = state.error_msg
            state.turn_trace.append(turn_record)
            return

        exec_tool_args = tool_args
        render_tool_args = compact_tool_arguments_for_prompt(tool_name, exec_tool_args)
        turn_record["tool_name"] = tool_name
        turn_record["tool_arguments"] = exec_tool_args
        turn_record["tool_arguments_for_prompt"] = render_tool_args

        state.tool_calls_made.append(tool_name)
        call_id = f"call_{tool_name}_{uuid.uuid4().hex[:8]}"

        state.conversation.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call_id,
                        "function": {
                            "name": tool_name,
                            "arguments": render_tool_args,
                        },
                    }
                ],
            }
        )

        tool_start = time.perf_counter()
        tool_result = execute_tool(
            tool_name,
            exec_tool_args,
            runtime_context=state.runtime_context,
            hidden_context=state.hidden_context,
        )
        tool_seconds = time.perf_counter() - tool_start
        turn_record["tool_seconds"] = round(tool_seconds, 6)
        turn_record["tool_result"] = tool_result
        tool_result_for_prompt = summarize_tool_result_for_conversation(
            tool_name,
            tool_result,
            state.meta,
            state.index_map,
        )
        turn_record["tool_result_for_prompt"] = tool_result_for_prompt
        state.conversation.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(tool_result_for_prompt, ensure_ascii=False),
            }
        )

        if verbose:
            print(f"  -> Tool call: {tool_name}")
            print(f"  <- Tool result: {format_preview(tool_result_for_prompt)}")

        if tool_name == "wls_from_path" and tool_result.get("success") is True:
            state.wls_completed_successfully = True

        missing_context_tool_error = is_missing_runtime_context_tool_error(tool_result)
        if tool_result.get("success") is False:
            if missing_context_tool_error and continue_on_missing_context_tool:
                turn_record["continued_after_tool_error"] = True
                turn_record["tool_error_type"] = "missing_runtime_context"
            elif not continue_on_tool_error:
                state.error_msg = f"Tool {tool_name} failed: {tool_result.get('error', 'unknown error')}"
                turn_record["error"] = state.error_msg
                state.turn_trace.append(turn_record)
                return

        state.turn_trace.append(turn_record)
        return

    if parsed["type"] == "verdict":
        if not state.wls_completed_successfully:
            state.error_msg = f"Verdict before required wls_from_path at turn {turn_index0 + 1}"
            turn_record["error"] = state.error_msg
            turn_record["verdict"] = parsed["content"]
            state.turn_trace.append(turn_record)
            return

        state.predicted_verdict = normalize_verdict(parsed["content"])
        turn_record["verdict"] = parsed["content"]
        state.conversation.append(
            {
                "role": "assistant",
                "content": json.dumps(parsed["content"], ensure_ascii=False),
            }
        )
        if verbose:
            print("  -> Final verdict:")
            print(f"  {format_preview(parsed['content'], limit=4000)}")
        state.turn_trace.append(turn_record)
        return

    excerpt = parsed.get("raw", response_text)
    state.error_msg = f"Unparseable output at turn {turn_index0 + 1}: {excerpt[:300]}"
    turn_record["error"] = state.error_msg
    state.turn_trace.append(turn_record)


def run_one_sample(
    messages_gt: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    max_turns: int,
    max_new_tokens: int,
    max_input_tokens: int,
    tools: list[dict[str, Any]] | None,
    continue_on_tool_error: bool,
    continue_on_missing_context_tool: bool,
    repair_wls_from_user: bool,
    enable_thinking: bool,
    verbose: bool,
    runtime_context: Mapping[str, Any] | None,
    inject_empty_thought_channel: bool,
    gc_collect_every_n_turns: int,
    empty_cuda_cache_every_n_turns: int,
    filter_unavailable_helper_tools: bool,
) -> dict[str, Any]:
    import torch

    gt_verdict = normalize_verdict(extract_ground_truth(messages_gt))
    conversation = extract_prompt_prefix(messages_gt)
    meta, index_map = extract_conversation_context(messages_gt)
    hidden_context = dict(runtime_context or {})
    sample_tools = (
        filter_tool_schemas_for_runtime_context(tools, runtime_context)
        if filter_unavailable_helper_tools
        else tools
    )

    tool_calls_made: list[str] = []
    parse_notes: list[str] = []
    predicted_verdict: dict[str, Any] | None = None
    error_msg: str | None = None
    input_truncated = False
    last_raw_generation: str | None = None
    wls_completed_successfully = False
    turn_trace: list[dict[str, Any]] = []

    stop_ids = get_stop_token_ids(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, list):
            pad_token_id = eos_id[0]
        else:
            pad_token_id = eos_id

    for turn in range(max_turns):
        maybe_run_turn_housekeeping(
            turn,
            gc_collect_every_n_turns=gc_collect_every_n_turns,
            empty_cuda_cache_every_n_turns=empty_cuda_cache_every_n_turns,
        )

        phase = infer_expected_phase(messages_gt, conversation)
        model_inputs, was_truncated = build_model_inputs(
            conversation,
            tokenizer,
            model,
            max_input_tokens=max_input_tokens,
            tools=sample_tools,
            enable_thinking=enable_thinking,
            phase=phase,
            inject_empty_thought_channel=inject_empty_thought_channel,
            runtime_context=runtime_context,
            filter_unavailable_helper_tools=filter_unavailable_helper_tools,
        )
        input_truncated = input_truncated or was_truncated

        turn_max_new_tokens = resolve_turn_max_new_tokens(turn, max_new_tokens)

        model_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=turn_max_new_tokens,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=pad_token_id,
            )
        model_generate_seconds = time.perf_counter() - model_start

        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        new_tokens = outputs[0][prompt_tokens:]
        response_text, decoded_token_count, trailing_pad_tokens = decode_generated_response(
            tokenizer,
            new_tokens,
            pad_token_id=pad_token_id,
        )
        last_raw_generation = response_text
        turn_record: dict[str, Any] = {
            "turn": turn + 1,
            "raw_generation": response_text,
            "prompt_truncated": was_truncated,
            "turn_max_new_tokens": turn_max_new_tokens,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(decoded_token_count),
            "trimmed_trailing_pad_tokens": int(trailing_pad_tokens),
            "model_generate_seconds": round(model_generate_seconds, 6),
        }

        if verbose:
            print_verbose_generation_block(
                turn + 1,
                len(new_tokens),
                response_text,
                was_truncated=was_truncated,
            )

        parsed = parse_gemma_generation(response_text, tokenizer)
        parse_notes.extend(parsed.get("notes", []))
        turn_record["parse_type"] = parsed["type"]
        if parsed.get("notes"):
            turn_record["parse_notes"] = list(parsed["notes"])
        if parsed.get("thinking"):
            turn_record["thinking"] = parsed["thinking"]

        if parsed["type"] == "tool_call":
            tool_name = parsed["name"]
            tool_args = parsed["arguments"]
            if repair_wls_from_user:
                tool_args, repair_notes = repair_tool_arguments(
                    tool_name,
                    tool_args,
                    conversation,
                    hidden_context=hidden_context,
                )
                if repair_notes:
                    parse_notes.extend(repair_notes)
                    turn_record.setdefault("parse_notes", [])
                    turn_record["parse_notes"].extend(repair_notes)

            if not isinstance(tool_args, dict):
                error_msg = f"Parsed tool arguments are not a dict for {tool_name}: {type(tool_args).__name__}"
                turn_record["error"] = error_msg
                turn_trace.append(turn_record)
                break

            exec_tool_args = tool_args
            render_tool_args = compact_tool_arguments_for_prompt(tool_name, exec_tool_args)
            turn_record["tool_name"] = tool_name
            turn_record["tool_arguments"] = exec_tool_args
            turn_record["tool_arguments_for_prompt"] = render_tool_args

            tool_calls_made.append(tool_name)
            call_id = f"call_{tool_name}_{uuid.uuid4().hex[:8]}"

            conversation.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": call_id,
                            "function": {
                                "name": tool_name,
                                "arguments": render_tool_args,
                            },
                        }
                    ],
                }
            )

            tool_start = time.perf_counter()
            tool_result = execute_tool(
                tool_name,
                exec_tool_args,
                runtime_context=runtime_context,
                hidden_context=hidden_context,
            )
            tool_seconds = time.perf_counter() - tool_start
            turn_record["tool_seconds"] = round(tool_seconds, 6)
            turn_record["tool_result"] = tool_result
            tool_result_for_prompt = summarize_tool_result_for_conversation(
                tool_name,
                tool_result,
                meta,
                index_map,
            )
            turn_record["tool_result_for_prompt"] = tool_result_for_prompt
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result_for_prompt, ensure_ascii=False),
                }
            )

            if verbose:
                print(f"  -> Tool call: {tool_name}")
                print(f"  <- Tool result: {format_preview(tool_result_for_prompt)}")

            del outputs
            if tool_name == "wls_from_path" and tool_result.get("success") is True:
                wls_completed_successfully = True

            missing_context_tool_error = is_missing_runtime_context_tool_error(tool_result)
            if tool_result.get("success") is False:
                if missing_context_tool_error and continue_on_missing_context_tool:
                    turn_record["continued_after_tool_error"] = True
                    turn_record["tool_error_type"] = "missing_runtime_context"
                elif not continue_on_tool_error:
                    error_msg = f"Tool {tool_name} failed: {tool_result.get('error', 'unknown error')}"
                    turn_record["error"] = error_msg
                    turn_trace.append(turn_record)
                    break

            turn_trace.append(turn_record)
            continue

        if parsed["type"] == "verdict":
            if not wls_completed_successfully:
                error_msg = f"Verdict before required wls_from_path at turn {turn + 1}"
                turn_record["error"] = error_msg
                turn_record["verdict"] = parsed["content"]
                turn_trace.append(turn_record)
                break

            predicted_verdict = normalize_verdict(parsed["content"])
            turn_record["verdict"] = parsed["content"]
            conversation.append(
                {
                    "role": "assistant",
                    "content": json.dumps(parsed["content"], ensure_ascii=False),
                }
            )
            if verbose:
                print("  -> Final verdict:")
                print(f"  {format_preview(parsed['content'], limit=4000)}")
            turn_trace.append(turn_record)
            break

        excerpt = parsed.get("raw", response_text)
        error_msg = f"Unparseable output at turn {turn + 1}: {excerpt[:300]}"
        turn_record["error"] = error_msg
        turn_trace.append(turn_record)
        break

    if predicted_verdict is None and error_msg is None:
        error_msg = f"Max turns reached without final verdict ({max_turns})"

    gt_family = gt_verdict.get("verdict", {}).get("error_family") if gt_verdict else None
    pred_family = predicted_verdict.get("verdict", {}).get("error_family") if predicted_verdict else None

    gt_has_error = gt_verdict.get("verdict", {}).get("has_error") if gt_verdict else None
    pred_has_error = predicted_verdict.get("verdict", {}).get("has_error") if predicted_verdict else None

    return {
        "gt_error_family": gt_family,
        "pred_error_family": pred_family,
        "gt_has_error": gt_has_error,
        "pred_has_error": pred_has_error,
        "family_correct": gt_family == pred_family,
        "detection_correct": gt_has_error == pred_has_error,
        "tool_calls": tool_calls_made,
        "num_turns": len(tool_calls_made) + (1 if predicted_verdict else 0),
        "predicted_verdict": predicted_verdict,
        "error": error_msg,
        "input_truncated": input_truncated,
        "parse_notes": parse_notes,
        "last_raw_generation": last_raw_generation,
        "required_wls_satisfied": wls_completed_successfully,
        "runtime": summarize_turn_trace_runtime(turn_trace),
        "turn_trace": turn_trace,
        "final_conversation": conversation,
    }


def run_sample_batch(
    batch_samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    sample_offset: int,
    max_turns: int,
    max_new_tokens: int,
    max_input_tokens: int,
    tools: list[dict[str, Any]] | None,
    continue_on_tool_error: bool,
    continue_on_missing_context_tool: bool,
    repair_wls_from_user: bool,
    enable_thinking: bool,
    verbose: bool,
    inject_empty_thought_channel: bool,
    gc_collect_every_n_turns: int,
    empty_cuda_cache_every_n_turns: int,
    filter_unavailable_helper_tools: bool,
) -> list[dict[str, Any]]:
    import torch

    states = [
        init_eval_sample_state(sample_offset + batch_index, sample["messages"], sample.get("runtime_context"))
        for batch_index, sample in enumerate(batch_samples)
    ]

    stop_ids = get_stop_token_ids(tokenizer)
    pad_token_id = resolve_pad_token_id(tokenizer)

    for turn in range(max_turns):
        active_states = [state for state in states if not state.finished]
        if not active_states:
            break

        maybe_run_turn_housekeeping(
            turn,
            gc_collect_every_n_turns=gc_collect_every_n_turns,
            empty_cuda_cache_every_n_turns=empty_cuda_cache_every_n_turns,
        )

        turn_max_new_tokens = resolve_turn_max_new_tokens(turn, max_new_tokens)

        model_inputs_list: list[dict[str, Any]] = []
        truncation_flags: list[bool] = []
        prompt_tokens_list: list[int] = []
        for state in active_states:
            phase = infer_expected_phase(state.messages_gt, state.conversation)
            state_tools = (
                filter_tool_schemas_for_runtime_context(tools, state.runtime_context)
                if filter_unavailable_helper_tools
                else tools
            )
            model_inputs, was_truncated = build_model_inputs(
                state.conversation,
                tokenizer,
                model,
                max_input_tokens=max_input_tokens,
                tools=state_tools,
                enable_thinking=enable_thinking,
                phase=phase,
                inject_empty_thought_channel=inject_empty_thought_channel,
                runtime_context=state.runtime_context,
                filter_unavailable_helper_tools=filter_unavailable_helper_tools,
            )
            state.input_truncated = state.input_truncated or was_truncated
            model_inputs_list.append(model_inputs)
            truncation_flags.append(was_truncated)
            prompt_tokens_list.append(int(model_inputs["input_ids"].shape[-1]))

        batched_inputs, padded_input_len = pad_model_inputs_for_batch(
            model_inputs_list,
            pad_token_id=pad_token_id,
        )

        model_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **batched_inputs,
                max_new_tokens=turn_max_new_tokens,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=pad_token_id,
            )
        batch_model_generate_seconds = time.perf_counter() - model_start
        per_state_model_generate_seconds = batch_model_generate_seconds / max(1, len(active_states))

        for row_index, state in enumerate(active_states):
            new_tokens = outputs[row_index][padded_input_len:]
            response_text, decoded_token_count, trailing_pad_tokens = decode_generated_response(
                tokenizer,
                new_tokens,
                pad_token_id=pad_token_id,
            )
            run_state_turn(
                state,
                turn_index0=turn,
                response_text=response_text,
                token_count=decoded_token_count,
                was_truncated=truncation_flags[row_index],
                turn_max_new_tokens=turn_max_new_tokens,
                tokenizer=tokenizer,
                continue_on_tool_error=continue_on_tool_error,
                continue_on_missing_context_tool=continue_on_missing_context_tool,
                repair_wls_from_user=repair_wls_from_user,
                verbose=verbose,
                prompt_tokens=prompt_tokens_list[row_index],
                model_generate_seconds=per_state_model_generate_seconds,
            )

        del outputs

    for state in states:
        if not state.finished:
            state.error_msg = f"Max turns reached without final verdict ({max_turns})"
    return [build_result_from_state(state) for state in states]



def run_samples_with_rolling_scheduler(
    test_samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    concurrent_conversations: int,
    max_turns: int,
    max_new_tokens: int,
    max_input_tokens: int,
    tools: list[dict[str, Any]] | None,
    continue_on_tool_error: bool,
    continue_on_missing_context_tool: bool,
    repair_wls_from_user: bool,
    enable_thinking: bool,
    verbose: bool,
    inject_empty_thought_channel: bool,
    gc_collect_every_n_turns: int,
    empty_cuda_cache_every_n_turns: int,
    filter_unavailable_helper_tools: bool,
    on_result: Any,
) -> None:
    import traceback
    import torch

    stop_ids = get_stop_token_ids(tokenizer)
    pad_token_id = resolve_pad_token_id(tokenizer)

    pending_index = 0
    active_states: list[EvalSampleState] = []

    def top_up() -> None:
        nonlocal pending_index
        while pending_index < len(test_samples) and len(active_states) < concurrent_conversations:
            sample = test_samples[pending_index]
            active_states.append(
                init_eval_sample_state(
                    pending_index,
                    sample["messages"],
                    sample.get("runtime_context"),
                )
            )
            pending_index += 1

    top_up()

    while active_states:
        survivors: list[EvalSampleState] = []
        for state in active_states:
            if not state.finished and state.turn_index0 >= max_turns:
                state.error_msg = f"Max turns reached without final verdict ({max_turns})"
                on_result(build_result_from_state(state))
            else:
                survivors.append(state)
        active_states = survivors
        if not active_states:
            top_up()
            continue

        seen_turns: set[int] = set()
        for state in active_states:
            if state.turn_index0 not in seen_turns:
                maybe_run_turn_housekeeping(
                    state.turn_index0,
                    gc_collect_every_n_turns=gc_collect_every_n_turns,
                    empty_cuda_cache_every_n_turns=empty_cuda_cache_every_n_turns,
                )
                seen_turns.add(state.turn_index0)

        grouped_states: dict[int, list[EvalSampleState]] = {}
        for state in active_states:
            turn_limit = resolve_turn_max_new_tokens(state.turn_index0, max_new_tokens)
            grouped_states.setdefault(turn_limit, []).append(state)

        for turn_max_new_tokens, group_states in sorted(grouped_states.items(), reverse=True):
            if not group_states:
                continue

            model_inputs_list: list[dict[str, Any]] = []
            prompt_tokens_list: list[int] = []
            truncation_flags: list[bool] = []
            for state in group_states:
                state_tools = (
                    filter_tool_schemas_for_runtime_context(tools, state.runtime_context)
                    if filter_unavailable_helper_tools
                    else tools
                )
                model_inputs, was_truncated = build_model_inputs(
                    state.conversation,
                    tokenizer,
                    model,
                    max_input_tokens=max_input_tokens,
                    tools=state_tools,
                    enable_thinking=enable_thinking,
                    phase=infer_expected_phase(state.messages_gt, state.conversation),
                    inject_empty_thought_channel=inject_empty_thought_channel,
                    runtime_context=state.runtime_context,
                    filter_unavailable_helper_tools=filter_unavailable_helper_tools,
                )
                state.input_truncated = state.input_truncated or was_truncated
                model_inputs_list.append(model_inputs)
                prompt_tokens_list.append(int(model_inputs["input_ids"].shape[-1]))
                truncation_flags.append(was_truncated)

            batched_inputs, padded_input_len = pad_model_inputs_for_batch(
                model_inputs_list,
                pad_token_id=pad_token_id,
            )

            try:
                model_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = model.generate(
                        **batched_inputs,
                        max_new_tokens=turn_max_new_tokens,
                        use_cache=True,
                        temperature=0.0,
                        do_sample=False,
                        eos_token_id=stop_ids,
                        pad_token_id=pad_token_id,
                    )
                batch_model_generate_seconds = time.perf_counter() - model_start
                per_state_model_generate_seconds = (
                    batch_model_generate_seconds / max(1, len(group_states))
                )

                for row_index, state in enumerate(group_states):
                    new_tokens = outputs[row_index][padded_input_len:]
                    response_text, decoded_token_count, trailing_pad_tokens = decode_generated_response(
                        tokenizer,
                        new_tokens,
                        pad_token_id=pad_token_id,
                    )
                    run_state_turn(
                        state,
                        turn_index0=state.turn_index0,
                        response_text=response_text,
                        token_count=decoded_token_count,
                        was_truncated=truncation_flags[row_index],
                        turn_max_new_tokens=turn_max_new_tokens,
                        tokenizer=tokenizer,
                        continue_on_tool_error=continue_on_tool_error,
                        continue_on_missing_context_tool=continue_on_missing_context_tool,
                        repair_wls_from_user=repair_wls_from_user,
                        verbose=verbose,
                        prompt_tokens=prompt_tokens_list[row_index],
                        model_generate_seconds=per_state_model_generate_seconds,
                    )
                    state.turn_index0 += 1

                del outputs
            except Exception as exc:
                traceback.print_exc(file=sys.stdout)
                print("Rolling batch step failed; retrying affected states serially.")
                for row_index, state in enumerate(group_states):
                    try:
                        single_inputs = model_inputs_list[row_index]
                        prompt_tokens = prompt_tokens_list[row_index]
                        model_start = time.perf_counter()
                        with torch.inference_mode():
                            outputs = model.generate(
                                **single_inputs,
                                max_new_tokens=turn_max_new_tokens,
                                use_cache=True,
                                temperature=0.0,
                                do_sample=False,
                                eos_token_id=stop_ids,
                                pad_token_id=pad_token_id,
                            )
                        model_generate_seconds = time.perf_counter() - model_start
                        new_tokens = outputs[0][prompt_tokens:]
                        response_text, decoded_token_count, trailing_pad_tokens = decode_generated_response(
                            tokenizer,
                            new_tokens,
                            pad_token_id=pad_token_id,
                        )
                        run_state_turn(
                            state,
                            turn_index0=state.turn_index0,
                            response_text=response_text,
                            token_count=decoded_token_count,
                            was_truncated=truncation_flags[row_index],
                            turn_max_new_tokens=turn_max_new_tokens,
                            tokenizer=tokenizer,
                            continue_on_tool_error=continue_on_tool_error,
                            continue_on_missing_context_tool=continue_on_missing_context_tool,
                            repair_wls_from_user=repair_wls_from_user,
                            verbose=verbose,
                            prompt_tokens=prompt_tokens,
                            model_generate_seconds=model_generate_seconds,
                        )
                        del outputs
                    except Exception as serial_exc:
                        traceback.print_exc(file=sys.stdout)
                        state.error_msg = f"CRITICAL ERROR: {serial_exc}"
                        state.turn_trace.append(
                            {
                                "turn": state.turn_index0 + 1,
                                "error": state.error_msg,
                                "prompt_tokens": prompt_tokens_list[row_index],
                            }
                        )
                    finally:
                        state.turn_index0 += 1

        still_active: list[EvalSampleState] = []
        finished_now: list[EvalSampleState] = []
        for state in active_states:
            if state.finished:
                finished_now.append(state)
            else:
                still_active.append(state)
        active_states = still_active

        for state in sorted(finished_now, key=lambda item: item.sample_index):
            on_result(build_result_from_state(state))

        top_up()


def main() -> None:
    args = parse_args()

    if not Path(args.adapter).exists() and "/" in args.adapter:
        print(f"Downloading adapter from Hugging Face Hub: {args.adapter} ...")
        from huggingface_hub import snapshot_download

        try:
            downloaded_path = snapshot_download(repo_id=args.adapter)
            if (Path(downloaded_path) / "lora").exists():
                args.adapter = str(Path(downloaded_path) / "lora")
            else:
                args.adapter = downloaded_path
        except Exception as exc:
            print(f"ERROR: Failed to download adapter: {exc}")
            sys.exit(1)
    elif not Path(args.adapter).exists():
        print(f"ERROR: Adapter not found locally at {args.adapter}")
        sys.exit(1)

    if not Path(args.test_file).exists():
        print(f"ERROR: Test file not found at {args.test_file}")
        sys.exit(1)

    adapter_cfg_path = Path(args.adapter) / "adapter_config.json"
    if not adapter_cfg_path.exists():
        print(f"ERROR: Adapter config not found at {adapter_cfg_path}")
        sys.exit(1)

    with open(adapter_cfg_path, "r", encoding="utf-8") as handle:
        adapter_cfg = json.load(handle)
    base_model_name = adapter_cfg["base_model_name_or_path"]
    tokenizer_path = base_model_name
    if not args.prefer_base_tokenizer and (Path(args.adapter) / "tokenizer_config.json").exists():
        tokenizer_path = args.adapter

    print(f"Loading adapter from {args.adapter} ...")
    print(f"  Base model: {base_model_name}")
    print(f"  Tokenizer source: {'base model' if tokenizer_path == base_model_name else 'adapter directory'}")
    use_transformers_fallback = False
    fallback_reason: str | None = None
    unsloth_stage = "initialization"
    unsloth_tempdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        from unsloth import FastModel

        unsloth_max_seq = args.max_seq_length if args.max_seq_length is not None else 4096
        unsloth_model_name, unsloth_tempdir, skipped_tokenizer_files = prepare_unsloth_adapter_path(
            Path(args.adapter),
            prefer_base_tokenizer=args.prefer_base_tokenizer,
        )
        if skipped_tokenizer_files:
            print(
                "  Unsloth adapter load: masking adapter-local tokenizer artifacts to prefer the base tokenizer: "
                + ", ".join(skipped_tokenizer_files)
            )
        unsloth_kwargs: dict[str, Any] = {
            "model_name": unsloth_model_name,
            "max_seq_length": unsloth_max_seq,
            "load_in_4bit": args.load_in_4bit,
            "load_in_16bit": args.load_in_16bit,
            "full_finetuning": False,
        }
        # args.adapter has already been resolved to a local PEFT adapter directory.
        # Keep the base-model revision pin for the transformers fallback path, but do
        # not forward a Hub revision into Unsloth's local-adapter loader.
        unsloth_stage = "FastModel.from_pretrained"
        model, tokenizer = FastModel.from_pretrained(**unsloth_kwargs)
        unsloth_stage = "FastModel.for_inference"
        FastModel.for_inference(model)
        print("Model loaded via Unsloth native adapter path.\n")
    except ImportError:
        use_transformers_fallback = True
        fallback_reason = "Unsloth not available"
    except Exception as exc:
        use_transformers_fallback = True
        fallback_reason = (
            f"Unsloth load failed during {unsloth_stage} "
            f"({type(exc).__name__}: {exc})"
        )
    finally:
        if unsloth_tempdir is not None:
            unsloth_tempdir.cleanup()

    if use_transformers_fallback:
        print(f"{fallback_reason}, falling back to transformers + peft ...")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base_model_kwargs: dict[str, Any] = {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if args.load_in_16bit or not args.load_in_4bit else None,
            "trust_remote_code": True,
        }
        if args.model_revision:
            base_model_kwargs["revision"] = args.model_revision
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **base_model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if args.model_revision and tokenizer_path == base_model_name:
            tokenizer_kwargs["revision"] = args.model_revision
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded via transformers + peft.\n")

    resolved_max_input_tokens = resolve_max_input_tokens(args, model, tokenizer)
    max_input_tokens = min(int(resolved_max_input_tokens), int(args.max_seq_length))
    tools = load_tools(args)
    print(f"Using max input context length: {max_input_tokens} (resolver={resolved_max_input_tokens}, cap={args.max_seq_length})")
    print(f"Pinned base revision: {args.model_revision or 'no'}")
    print(f"Tool schemas passed to chat template: {'yes' if tools is not None else 'no'}")
    print(f"Gemma thinking enabled: {'yes' if args.enable_thinking else 'no'}")
    print(f"Inject empty thought channel: {'yes' if args.inject_empty_thought_channel else 'no'}")
    print(f"GC collect every N turns: {max(0, int(args.gc_collect_every_n_turns))}")
    print(f"Empty CUDA cache every N turns: {max(0, int(args.empty_cuda_cache_every_n_turns))}")
    print(f"Filter unavailable helper tools: {'yes' if args.filter_unavailable_helper_tools else 'no'}")
    print(f"Continue on missing context tool: {'yes' if args.continue_on_missing_context_tool else 'no'}")
    print(f"Rolling batch scheduler: {'yes' if args.rolling_batch_scheduler else 'no'}")
    print(f"Concurrent conversations: {max(1, int(args.concurrent_conversations))}\n")

    test_samples: list[dict[str, Any]] = []
    with open(args.test_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    if args.max_samples:
        test_samples = test_samples[: args.max_samples]
    print(f"Loaded {len(test_samples)} test samples.\n")

    results: list[dict[str, Any]] = []
    family_correct = 0
    detection_correct = 0
    errors = 0
    family_counts: Counter[str] = Counter()
    family_correct_counts: Counter[str] = Counter()
    error_kind_counts: Counter[str] = Counter()
    parse_note_counts: Counter[str] = Counter()

    concurrent_conversations = max(1, int(args.concurrent_conversations))

    def record_result(result: dict[str, Any]) -> None:
        nonlocal family_correct, detection_correct, errors

        results.append(result)
        out_file.write(json.dumps(result, default=str, ensure_ascii=False) + "\n")
        out_file.flush()

        gt_family_local = result["gt_error_family"] or "unknown"
        family_counts[gt_family_local] += 1

        if result["family_correct"]:
            family_correct += 1
            family_correct_counts[gt_family_local] += 1
        if result["detection_correct"]:
            detection_correct += 1
        if result["error"]:
            errors += 1
            error_kind = classify_result_error(result["error"])
            if error_kind is not None:
                error_kind_counts[error_kind] += 1
        for note in result.get("parse_notes") or []:
            parse_note_counts[note] += 1

        status = "✓" if result["family_correct"] else "✗"
        pred_fam_str = str(result["pred_error_family"]) if result["pred_error_family"] else "NONE"
        extra = ""
        if result.get("input_truncated"):
            extra += " truncated"
        if result.get("parse_notes"):
            extra += f" notes={result['parse_notes'][:3]}"

        print(
            f"[{len(results)}/{len(test_samples)}] {status}  gt={gt_family_local:<20s}  pred={pred_fam_str:<20s}  "
            f"tools={result['tool_calls']}{extra}"
        )

    with open(args.output, "w", encoding="utf-8") as out_file:
        if args.rolling_batch_scheduler and concurrent_conversations > 1:
            print(
                f"Using rolling microbatch scheduler with capacity {concurrent_conversations}.\n"
            )
            run_samples_with_rolling_scheduler(
                test_samples,
                model,
                tokenizer,
                concurrent_conversations=concurrent_conversations,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=max_input_tokens,
                tools=tools,
                continue_on_tool_error=args.continue_on_tool_error,
                continue_on_missing_context_tool=args.continue_on_missing_context_tool,
                repair_wls_from_user=args.repair_wls_from_user,
                enable_thinking=args.enable_thinking,
                verbose=args.verbose,
                inject_empty_thought_channel=args.inject_empty_thought_channel,
                gc_collect_every_n_turns=args.gc_collect_every_n_turns,
                empty_cuda_cache_every_n_turns=args.empty_cuda_cache_every_n_turns,
                filter_unavailable_helper_tools=args.filter_unavailable_helper_tools,
                on_result=record_result,
            )
        else:
            for batch_start in range(0, len(test_samples), concurrent_conversations):
                batch_samples = test_samples[batch_start : batch_start + concurrent_conversations]
                batch_end = batch_start + len(batch_samples)
                if len(batch_samples) == 1:
                    print(f"[{batch_start + 1}/{len(test_samples)}] ", end="", flush=True)
                else:
                    print(f"[{batch_start + 1}-{batch_end}/{len(test_samples)}] batch size={len(batch_samples)}")

                try:
                    if len(batch_samples) == 1:
                        sample = batch_samples[0]
                        result_batch = [
                            run_one_sample(
                                sample["messages"],
                                model,
                                tokenizer,
                                max_turns=args.max_turns,
                                max_new_tokens=args.max_new_tokens,
                                max_input_tokens=max_input_tokens,
                                tools=tools,
                                continue_on_tool_error=args.continue_on_tool_error,
                                continue_on_missing_context_tool=args.continue_on_missing_context_tool,
                                repair_wls_from_user=args.repair_wls_from_user,
                                enable_thinking=args.enable_thinking,
                                verbose=args.verbose,
                                runtime_context=sample.get("runtime_context"),
                                inject_empty_thought_channel=args.inject_empty_thought_channel,
                                gc_collect_every_n_turns=args.gc_collect_every_n_turns,
                                empty_cuda_cache_every_n_turns=args.empty_cuda_cache_every_n_turns,
                                filter_unavailable_helper_tools=args.filter_unavailable_helper_tools,
                            )
                        ]
                    else:
                        result_batch = run_sample_batch(
                            batch_samples,
                            model,
                            tokenizer,
                            sample_offset=batch_start,
                            max_turns=args.max_turns,
                            max_new_tokens=args.max_new_tokens,
                            max_input_tokens=max_input_tokens,
                            tools=tools,
                            continue_on_tool_error=args.continue_on_tool_error,
                            continue_on_missing_context_tool=args.continue_on_missing_context_tool,
                            repair_wls_from_user=args.repair_wls_from_user,
                            enable_thinking=args.enable_thinking,
                            verbose=args.verbose,
                            inject_empty_thought_channel=args.inject_empty_thought_channel,
                            gc_collect_every_n_turns=args.gc_collect_every_n_turns,
                            empty_cuda_cache_every_n_turns=args.empty_cuda_cache_every_n_turns,
                            filter_unavailable_helper_tools=args.filter_unavailable_helper_tools,
                        )
                except Exception as exc:
                    import traceback

                    traceback.print_exc(file=sys.stdout)
                    print("Batch evaluation failed; falling back to serial evaluation for this batch.")
                    result_batch = []
                    for sample in batch_samples:
                        try:
                            result_batch.append(
                                run_one_sample(
                                    sample["messages"],
                                    model,
                                    tokenizer,
                                    max_turns=args.max_turns,
                                    max_new_tokens=args.max_new_tokens,
                                    max_input_tokens=max_input_tokens,
                                    tools=tools,
                                    continue_on_tool_error=args.continue_on_tool_error,
                                    continue_on_missing_context_tool=args.continue_on_missing_context_tool,
                                    repair_wls_from_user=args.repair_wls_from_user,
                                    enable_thinking=args.enable_thinking,
                                    verbose=args.verbose,
                                    runtime_context=sample.get("runtime_context"),
                                    inject_empty_thought_channel=args.inject_empty_thought_channel,
                                    gc_collect_every_n_turns=args.gc_collect_every_n_turns,
                                    empty_cuda_cache_every_n_turns=args.empty_cuda_cache_every_n_turns,
                                    filter_unavailable_helper_tools=args.filter_unavailable_helper_tools,
                                )
                            )
                        except Exception as serial_exc:
                            traceback.print_exc(file=sys.stdout)
                            result_batch.append(build_critical_error_result(sample["messages"], serial_exc))

                for result in result_batch:
                    record_result(result)

    n = len(results)
    print("\n" + "=" * 60)
    print(f"{'EVALUATION SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total samples:          {n}")
    print(f"  Error detection acc:    {detection_correct}/{n}  ({100 * detection_correct / n:.1f}%)")
    print(f"  Error family acc:       {family_correct}/{n}  ({100 * family_correct / n:.1f}%)")
    print(f"  Parse/runtime errors:   {errors}")
    total_model_seconds = sum(
        float((result.get("runtime") or {}).get("model_generate_seconds_total", 0.0) or 0.0)
        for result in results
    )
    total_tool_seconds = sum(
        float((result.get("runtime") or {}).get("tool_seconds_total", 0.0) or 0.0)
        for result in results
    )
    total_generated_tokens = sum(
        int((result.get("runtime") or {}).get("generated_tokens_total", 0) or 0)
        for result in results
    )
    estimated_decode_tps = None
    if total_model_seconds > 0:
        estimated_decode_tps = total_generated_tokens / total_model_seconds
    print(f"  Estimated model time:   {total_model_seconds:.2f}s")
    print(f"  Estimated tool time:    {total_tool_seconds:.2f}s")
    if estimated_decode_tps is not None:
        print(f"  Estimated decode TPS:   {estimated_decode_tps:.1f}")
    print()
    if error_kind_counts:
        print("  Error breakdown:")
        for kind, count in error_kind_counts.most_common():
            print(f"    {kind:<25s}  {count}")
        print()
    print("  Per-family breakdown:")
    for family in sorted(family_counts.keys()):
        total = family_counts[family]
        correct = family_correct_counts[family]
        print(f"    {family:<25s}  {correct}/{total}  ({100 * correct / total:.1f}%)")
    if parse_note_counts:
        print()
        print("  Top parser notes:")
        for note, count in parse_note_counts.most_common(8):
            print(f"    {note:<35s}  {count}")
    print("=" * 60)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
