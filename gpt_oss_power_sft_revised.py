#!/usr/bin/env python3
"""SFT for Gemma 4 on power-system tool traces.

Design choices:
1) Normalize OpenAI-style tool call arguments back into Python dicts/lists before
   applying the Transformers chat template.
2) Expand each conversation into one training sample per assistant turn so Gemma 4
   always learns from the *final* assistant action of the current sample.
3) Build explicit completion masks from prompt/completion boundaries instead of
   relying on fragile string-marker masking.
4) Truncate only the prompt prefix when sequences exceed max length; never cut the
   target assistant action unless the action itself is longer than the context.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from unsloth import FastModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from trace_protocol import canonical_tool_schemas


DEFAULT_POWER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "wls_from_path",
            "description": "Run weighted least-squares state estimation on a power-system snapshot and return residual diagnostics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "z": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Observed measurement vector.",
                    },
                },
                "required": ["case_path", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_measurements_from_path",
            "description": "Correct suspected bad measurements and optionally rerun diagnostic iterations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string"},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "suspect_group": {"type": "array", "items": {"type": "integer"}},
                    "enable_correction": {"type": "boolean"},
                    "max_correction_iterations": {"type": "integer"},
                    "error_tolerance": {"type": "number"},
                },
                "required": [
                    "case_path",
                    "z",
                    "suspect_group",
                    "enable_correction",
                    "max_correction_iterations",
                    "error_tolerance",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_parameters_from_path",
            "description": "Correct line-parameter errors using repeated measurement scans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string"},
                    "line_index": {"type": "integer"},
                    "z_scans": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                },
                "required": ["case_path", "line_index", "z_scans"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_topology_from_path",
            "description": "Correct a suspected topology mismatch by switching a breaker/circuit breaker status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string"},
                    "cb_name": {"type": "string"},
                    "desired_status": {"type": "boolean"},
                },
            "required": ["case_path", "cb_name", "desired_status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_hse_from_path",
            "description": "Run Harmonic State Estimation (HSE) to identify a single harmonic source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string"},
                    "harmonic_measurements": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "harmonic_orders": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "slack_bus": {"type": "integer"}
                },
                "required": ["case_path", "harmonic_measurements"]
            },
        },
    },
]

LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

GEMMA_TOOL_CALL_OPEN = "<|tool_call>"
GEMMA_THOUGHT_OPEN = "<|channel>thought"
GEMMA_CHANNEL_CLOSE = "<channel|>"
FINAL_JSON_SCHEMA_MARKER = "Return only strict JSON with this structure:"
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


SCRIPT_DIR = Path(__file__).resolve().parent
PREEMPTION_EXIT_CODE = 99

SYSTEM_TEXT_REPLACEMENTS = {
    "Use Harmony/native tool calling only.": "Use the active model chat template's native tool-calling format.",
    "Harmony/native tool calling": "native tool calling",
    "<|call|>": "",
    "<|return|>": "",
}
CHAT_TEMPLATE_CONTENT_FALLBACK_WARNED = False


def normalize_instruction_content(role: str, content: Any) -> Any:
    if content is None:
        return ""
    if not isinstance(content, str):
        return content
    if role not in {"system", "developer"}:
        return content

    text = content
    for source_text, replacement_text in SYSTEM_TEXT_REPLACEMENTS.items():
        text = text.replace(source_text, replacement_text)
    return text.strip()


class SignalState:
    def __init__(self) -> None:
        self.received: int | None = None

    def install(self) -> None:
        for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None)):
            if signum is None:
                continue
            try:
                signal.signal(signum, self._handle_signal)
            except Exception:
                continue

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self.received is not None:
            return
        self.received = signum
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(
            f"Received {signame}; will save a checkpoint at the end of the current step and stop training.",
            flush=True,
        )


class SaveOnSignalCallback(TrainerCallback):
    def __init__(self, signal_state: SignalState) -> None:
        self.signal_state = signal_state

    def _request_stop(self, control: Any) -> Any:
        if self.signal_state.received is None:
            return control
        control.should_save = True
        control.should_training_stop = True
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._request_stop(control)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._request_stop(control)


def make_sft_config_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Filter/rename kwargs so the script works across TRL/SFTConfig versions."""
    parameters = inspect.signature(SFTConfig.__init__).parameters
    supported = {name for name in parameters if name != "self"}
    normalized = dict(kwargs)

    alias_pairs = [
        ("eval_strategy", "evaluation_strategy"),
        ("evaluation_strategy", "eval_strategy"),
    ]
    for source_name, target_name in alias_pairs:
        if source_name in normalized and source_name not in supported and target_name in supported:
            normalized[target_name] = normalized.pop(source_name)

    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in normalized.items():
        if value is None:
            continue
        if key in supported:
            filtered[key] = value
        else:
            dropped.append(key)

    if dropped:
        print(
            "SFTConfig does not support these kwargs in the current environment; dropping: "
            + ", ".join(sorted(dropped)),
            flush=True,
        )
    return filtered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 SFT for power-system tool traces")
    parser.add_argument("--train-file", type=str, default="out_traces_balanced/sft_traces.train.jsonl")
    parser.add_argument("--valid-file", "--val-file", dest="valid_file", type=str, default="out_traces_balanced/sft_traces.valid.jsonl")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Cap the number of training conversations loaded from the JSONL file. 0 keeps all rows.",
    )
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=0,
        help="Cap the number of validation conversations loaded from the JSONL file. 0 keeps all rows.",
    )
    parser.add_argument("--model-name", type=str, default="unsloth/Gemma-4-26B-A4B-it")
    parser.add_argument("--model-revision", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/gemma4_power_agent")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="auto",
        help="Checkpoint path to resume from, or 'auto' to use the latest checkpoint in --output-dir.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", type=str, default="none", help="The integration to report the results to, e.g., 'wandb'")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load-in-16bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-16bit", dest="load_in_16bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-scope",
        type=str,
        choices=("language_model", "all"),
        default="language_model",
        help="Which Gemma 4 towers receive LoRA. Use 'language_model' for text-only tool traces.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--drop-too-long-targets", action="store_true", default=True)
    parser.add_argument("--keep-too-long-targets", dest="drop_too_long_targets", action="store_false")
    parser.add_argument("--include-tool-schemas", action="store_true", default=True)
    parser.add_argument("--no-include-tool-schemas", dest="include_tool_schemas", action="store_false")
    parser.add_argument("--tools-file", type=str, default="")
    parser.add_argument("--phase-gated-prompt", action="store_true", default=True)
    parser.add_argument("--no-phase-gated-prompt", dest="phase_gated_prompt", action="store_false")
    parser.add_argument(
        "--preserve-system-text",
        action="store_true",
        default=False,
        help="Do not rewrite Harmony/GPT-OSS-specific system/developer wording.",
    )
    parser.add_argument(
        "--sanity-check-samples",
        type=int,
        default=3,
        help="Number of first-turn tool-call prompts to greedily decode after training; 0 disables the check.",
    )
    parser.add_argument(
        "--mask-sanity-samples",
        type=int,
        default=3,
        help="Number of pre-train label spans to decode and verify before training starts; 0 disables the check.",
    )
    parser.add_argument(
        "--sanity-check-max-new-tokens",
        type=int,
        default=128,
        help="Generation budget for each post-train sanity decode.",
    )
    parser.add_argument(
        "--sanity-check-fail-on-miss",
        action="store_true",
        default=False,
        help="Exit non-zero if any post-train sanity sample fails to begin with the expected tool call.",
    )
    return parser.parse_args(argv)


def has_nonempty_jsonl(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with p.open("r", encoding="utf-8") as f:
        return any(line.strip() for line in f)


def load_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_num}, got {type(row).__name__}")
            rows.append(row)
    return rows


def latest_checkpoint_dir(output_dir: str) -> str | None:
    root = Path(output_dir)
    if not root.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("checkpoint-"):
            continue
        suffix = name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        checkpoints.append((int(suffix), child))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def resolve_resume_checkpoint(value: str, output_dir: str) -> str | None:
    if not value:
        return None
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        return latest_checkpoint_dir(output_dir)
    return value


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def candidate_roots(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]
    return unique_paths([Path.cwd() / path, SCRIPT_DIR / path])


def resolve_dataset_path(path: str, split_name: str, required: bool = True) -> str | None:
    requested = Path(path).expanduser()

    for candidate in candidate_roots(requested):
        if candidate.exists():
            return str(candidate)

    suggestions: list[Path] = []
    split_patterns = {
        "sft_traces.train.jsonl": ["*.train.jsonl"],
        "sft_traces.valid.jsonl": ["*.valid.jsonl", "*.val.jsonl"],
        "sft_traces.val.jsonl": ["*.val.jsonl", "*.valid.jsonl"],
        "sft_traces.test.jsonl": ["*.test.jsonl"],
    }.get(requested.name)

    search_dirs: list[Path] = []
    if requested.is_absolute():
        search_dirs.append(requested.parent)
    else:
        search_dirs.extend(
            [
                Path.cwd() / requested.parent,
                SCRIPT_DIR / requested.parent,
                Path.cwd(),
                SCRIPT_DIR,
                SCRIPT_DIR / "data",
            ]
        )
    search_dirs = unique_paths(search_dirs)

    if split_patterns is not None:
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            matches: list[Path] = []
            for pattern in split_patterns:
                matches.extend(sorted(directory.glob(pattern)))
            matches = unique_paths(matches)
            suggestions.extend(matches)
            if len(matches) == 1:
                resolved = matches[0]
                print(
                    f"Resolved missing {split_name} dataset path "
                    f"'{path}' -> '{display_path(resolved)}'"
                )
                return str(resolved)

    for directory in search_dirs:
        candidate = directory / requested.name
        if candidate.exists():
            suggestions.append(candidate)

    if not required:
        print(f"Validation dataset not found at '{path}'; continuing without evaluation split.")
        return None

    suggestion_lines = ""
    unique_suggestions = unique_paths(suggestions)
    if unique_suggestions:
        formatted = "\n".join(f"  - {display_path(candidate)}" for candidate in unique_suggestions[:8])
        suggestion_lines = f"\nAvailable candidates:\n{formatted}"

    raise FileNotFoundError(
        f"Unable to find {split_name} dataset file '{path}'.{suggestion_lines}"
    )


def prune_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            cleaned = prune_none(value)
            if cleaned is not None:
                out[key] = cleaned
        return out
    if isinstance(obj, list):
        return [prune_none(v) for v in obj if v is not None]
    return obj


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if s[0] not in "[{":
        return value
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return value


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


def normalize_messages(
    messages: list[dict[str, Any]],
    *,
    preserve_system_text: bool = False,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = prune_none(raw_message)
        role = msg.get("role")
        if role is None:
            continue

        if role in {"system", "developer"}:
            content = msg.get("content", "")
            if not preserve_system_text:
                content = normalize_instruction_content(role, content)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            msg["content"] = content

        elif role == "user":
            content = msg.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            msg["content"] = content

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                fixed_calls = []
                for tool_call in tool_calls:
                    tc = prune_none(tool_call)
                    function_info = tc.get("function")
                    if isinstance(function_info, dict):
                        function_info = dict(function_info)
                        arguments = maybe_parse_json_string(function_info.get("arguments"))
                        if arguments is None:
                            arguments = {}
                        function_info["arguments"] = arguments
                        tc["function"] = function_info
                    fixed_calls.append(tc)
                msg["tool_calls"] = fixed_calls
                msg.pop("content", None)
            else:
                msg.pop("tool_calls", None)
                content = msg.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                msg["content"] = content

        elif role == "tool":
            content = msg.get("content", "")
            if content is None:
                content = ""
            parsed = maybe_parse_json_string(content)
            if isinstance(parsed, (dict, list)):
                msg["content"] = parsed
            elif isinstance(content, str):
                msg["content"] = content
            else:
                msg["content"] = json.dumps(content, ensure_ascii=False)

        normalized.append(msg)
    return normalized


def assistant_turn_indices(messages: list[dict[str, Any]]) -> list[int]:
    return [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]


def explode_conversation(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Create one training sample per assistant turn.

    The model should learn from the final assistant action of each sample. This is
    especially important for tool-calling traces.
    """
    expanded: list[list[dict[str, Any]]] = []
    turns = assistant_turn_indices(messages)
    for idx in turns:
        expanded.append(messages[: idx + 1])
    return expanded


def strip_final_json_schema(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    marker_index = text.find(FINAL_JSON_SCHEMA_MARKER)
    if marker_index == -1:
        return text
    return text[:marker_index].rstrip()


def assistant_phase_bucket(target_kind: str, assistant_turn_index: int) -> str:
    if target_kind == "final":
        return "final"
    if assistant_turn_index == 0:
        return "first_tool_call"
    return "later_tool_call"


def phase_gate_messages(
    messages: list[dict[str, Any]],
    *,
    target_kind: str,
    assistant_turn_index: int,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return [dict(message) for message in messages]

    gated = [dict(message) for message in messages]
    if not gated:
        return gated

    if target_kind == "tool_call":
        for message in gated:
            if message.get("role") not in {"system", "developer"}:
                continue
            message["content"] = strip_final_json_schema(message.get("content"))
        phase_message = (
            FIRST_TOOL_PHASE_MESSAGE
            if assistant_turn_index == 0
            else LATER_TOOL_PHASE_MESSAGE
        )
    else:
        phase_message = FINAL_PHASE_MESSAGE

    insert_index = max(0, len(gated) - 1)
    gated.insert(insert_index, {"role": "developer", "content": phase_message})
    return gated


def load_tools(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not args.include_tool_schemas:
        return None
    if args.tools_file:
        with open(args.tools_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--tools-file must contain a JSON list of tool schemas.")
        return sanitize_tool_schemas(data)
    return canonical_tool_schemas()


def render_text(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, add_generation_prompt: bool) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as exc:
            return render_text_with_stringified_content_fallback(tokenizer, messages, kwargs, exc)
    except Exception as exc:
        return render_text_with_stringified_content_fallback(tokenizer, messages, kwargs, exc)


def stringify_message_content_for_template(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    coerced_messages: list[dict[str, Any]] = []
    mutated = False
    for raw_message in messages:
        message = dict(raw_message)
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

    global CHAT_TEMPLATE_CONTENT_FALLBACK_WARNED
    if not CHAT_TEMPLATE_CONTENT_FALLBACK_WARNED:
        print(
            "Chat template rejected structured message content; retrying with JSON-stringified content.",
            flush=True,
        )
        CHAT_TEMPLATE_CONTENT_FALLBACK_WARNED = True

    return tokenizer.apply_chat_template(coerced_messages, **kwargs)


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


def infer_side_input_plan(model: Any, tokenizer: Any, model_name: str) -> SideInputPlan:
    discovered: set[str] = set()
    model_input_names = getattr(tokenizer, "model_input_names", None)
    if isinstance(model_input_names, (list, tuple)):
        discovered.update(str(name) for name in model_input_names)
    try:
        discovered.update(inspect.signature(model.forward).parameters.keys())
    except Exception:
        pass

    lowered = model_name.lower()
    is_gemma = "gemma" in lowered
    is_gemma4 = "gemma-4" in lowered or "gemma4" in lowered

    need_token_type_ids = "token_type_ids" in discovered or is_gemma
    need_mm_token_type_ids = "mm_token_type_ids" in discovered or is_gemma4
    return SideInputPlan(need_token_type_ids=need_token_type_ids, need_mm_token_type_ids=need_mm_token_type_ids)


def classify_lora_tower(module_name: str) -> str:
    if module_name.startswith("model.language_model.") or ".language_model." in module_name:
        return "language_model"
    if module_name.startswith("model.vision_tower.") or ".vision_tower." in module_name:
        return "vision_tower"
    if module_name.startswith("model.audio_tower.") or ".audio_tower." in module_name:
        return "audio_tower"
    return "other"


def resolve_lora_target_modules(model: Any, scope: str) -> list[str]:
    inventory = Counter()
    selected: list[str] = []

    for module_name, _module in model.named_modules():
        if not module_name.endswith(LORA_TARGET_SUFFIXES):
            continue
        tower = classify_lora_tower(module_name)
        inventory[tower] += 1
        if scope == "all" or tower == scope:
            selected.append(module_name)

    if not selected:
        raise ValueError(
            f"No LoRA target modules matched scope={scope!r}. Inventory by tower: {dict(inventory)}"
        )

    print(f"LoRA module inventory by tower: {dict(inventory)}")
    print(f"Selected {len(selected)} LoRA target modules for scope={scope}.")
    preview = selected[:8]
    for module_name in preview:
        print(f"  - {module_name}")
    remaining = len(selected) - len(preview)
    if remaining > 0:
        print(f"  ... {remaining} more")
    return selected


def slice_or_zeros(values: list[int] | None, start: int, stop: int) -> list[int]:
    if values is None:
        return [0] * (stop - start)
    return values[start:stop]


class BuildStats:
    def __init__(self) -> None:
        self.original_rows = 0
        self.expanded_rows = 0
        self.used_rows = 0
        self.prompt_trimmed = 0
        self.dropped_too_long_target = 0
        self.target_kind = Counter()
        self.original_message_lengths = Counter()


class SideInputPlan:
    def __init__(self, need_token_type_ids: bool, need_mm_token_type_ids: bool) -> None:
        self.need_token_type_ids = need_token_type_ids
        self.need_mm_token_type_ids = need_mm_token_type_ids

    def as_dict(self) -> dict[str, bool]:
        return {
            "token_type_ids": self.need_token_type_ids,
            "mm_token_type_ids": self.need_mm_token_type_ids,
        }



def build_processed_split(
    raw_split: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    default_tools: list[dict[str, Any]] | None,
    drop_too_long_targets: bool,
    stats: BuildStats,
    split_name: str,
    side_inputs: SideInputPlan,
    preserve_system_text: bool,
    phase_gated_prompt: bool,
):
    records: list[dict[str, Any]] = []

    for row_index, row in enumerate(raw_split):
        stats.original_rows += 1
        raw_messages = row["messages"]
        stats.original_message_lengths[len(raw_messages)] += 1
        normalized = normalize_messages(raw_messages, preserve_system_text=preserve_system_text)
        expanded = explode_conversation(normalized)
        stats.expanded_rows += len(expanded)

        row_tools = row.get("tools")
        if row_tools is not None:
            if not isinstance(row_tools, list):
                raise ValueError(f"Expected row['tools'] to be a list in {split_name}, got {type(row_tools).__name__}")
            tools = sanitize_tool_schemas(row_tools)
        else:
            tools = default_tools

        for sample_index, sample_messages in enumerate(expanded):
            target = sample_messages[-1]
            target_kind = "tool_call" if "tool_calls" in target else "final"
            stats.target_kind[target_kind] += 1
            phase_bucket = assistant_phase_bucket(target_kind, sample_index)

            phase_messages = phase_gate_messages(
                sample_messages,
                target_kind=target_kind,
                assistant_turn_index=sample_index,
                enabled=phase_gated_prompt,
            )
            history = phase_messages[:-1]
            full_text = render_text(tokenizer, phase_messages, tools, add_generation_prompt=False)
            prompt_text = render_text(tokenizer, history, tools, add_generation_prompt=True)

            full_enc = encode_text(tokenizer, full_text)
            prompt_ids = encode_text(tokenizer, prompt_text)["input_ids"]
            full_ids = full_enc["input_ids"]

            if len(full_ids) < len(prompt_ids):
                raise ValueError(
                    f"Prompt tokenization longer than full sample in {split_name}; this should not happen."
                )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                mismatch_at = next(
                    (idx for idx, (full_id, prompt_id) in enumerate(zip(full_ids, prompt_ids)) if full_id != prompt_id),
                    min(len(full_ids), len(prompt_ids)),
                )
                prompt_tail = prompt_text[max(0, len(prompt_text) - 160):]
                full_tail = full_text[max(0, len(full_text) - 160):]
                raise ValueError(
                    "Prompt tokenization is not a literal prefix of the full sample. "
                    f"split={split_name} row={row_index} sample={sample_index} mismatch_token_index={mismatch_at}\n"
                    f"prompt_tail={prompt_tail!r}\nfull_tail={full_tail!r}"
                )

            completion_len = len(full_ids) - len(prompt_ids)
            if completion_len <= 0:
                raise ValueError(
                    f"Non-positive completion length in {split_name}; inspect sample rendering."
                )

            orig_length = len(full_ids)
            truncated = False

            if orig_length > max_seq_length:
                if completion_len > max_seq_length:
                    if drop_too_long_targets:
                        stats.dropped_too_long_target += 1
                        continue
                    slice_start = orig_length - max_seq_length
                    input_ids = full_ids[-max_seq_length:]
                    completion_mask = [1] * max_seq_length
                    truncated = True
                else:
                    keep_prompt = max_seq_length - completion_len
                    slice_start = orig_length - (keep_prompt + completion_len)
                    input_ids = full_ids[slice_start:]
                    completion_mask = [0] * keep_prompt + [1] * completion_len
                    stats.prompt_trimmed += 1
                    truncated = True
            else:
                slice_start = 0
                input_ids = full_ids
                completion_mask = [0] * len(prompt_ids) + [1] * completion_len

            slice_stop = slice_start + len(input_ids)
            attention_mask = [1] * len(input_ids)
            if len(input_ids) != len(completion_mask):
                raise ValueError("input_ids/completion_mask length mismatch")
            labels = [token_id if is_completion else -100 for token_id, is_completion in zip(input_ids, completion_mask)]

            record: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "completion_mask": completion_mask,
                "orig_length": orig_length,
                "used_length": len(input_ids),
                "completion_length": completion_len,
                "target_kind": target_kind,
                "phase_bucket": phase_bucket,
                "assistant_turn_index": sample_index,
                "expected_tool_name": (
                    (((target.get("tool_calls") or [{}])[0].get("function") or {}).get("name"))
                    if target_kind == "tool_call"
                    else None
                ),
                "case_path": extract_case_path(history),
                "was_truncated": truncated,
                "text": full_text,  # kept for debugging/preview only
            }

            if side_inputs.need_token_type_ids:
                record["token_type_ids"] = slice_or_zeros(full_enc.get("token_type_ids"), slice_start, slice_stop)

            if side_inputs.need_mm_token_type_ids:
                record["mm_token_type_ids"] = slice_or_zeros(full_enc.get("mm_token_type_ids"), slice_start, slice_stop)

            records.append(record)
            stats.used_rows += 1

    if not records:
        raise ValueError(
            f"No usable samples were built for {split_name}. Increase --max-seq-length, redesign tool payloads, or disable --drop-too-long-targets."
        )
    return records



def percentile(sorted_values: list[int], p: int) -> int:
    if not sorted_values:
        return 0
    idx = int(round((p / 100) * (len(sorted_values) - 1)))
    return sorted_values[idx]



def report_dataset(records: list[dict[str, Any]], split_name: str, stats: BuildStats, max_seq_length: int) -> None:
    lengths = [r["orig_length"] for r in records]
    used_lengths = [r["used_length"] for r in records]
    lengths_sorted = sorted(lengths)
    used_sorted = sorted(used_lengths)
    phase_sample_counts = Counter()
    phase_token_mass = Counter()
    phase_completion_sums = Counter()
    for record in records:
        phase = record.get("phase_bucket") or assistant_phase_bucket(
            str(record.get("target_kind")),
            int(record.get("assistant_turn_index", 0)),
        )
        completion_length = int(record["completion_length"])
        phase_sample_counts[phase] += 1
        phase_token_mass[phase] += completion_length
        phase_completion_sums[phase] += completion_length
    phase_order = ("first_tool_call", "later_tool_call", "final")
    ordered_phase_counts = {phase: phase_sample_counts[phase] for phase in phase_order if phase_sample_counts[phase]}
    ordered_phase_tokens = {phase: phase_token_mass[phase] for phase in phase_order if phase_token_mass[phase]}
    ordered_phase_means = {
        phase: round(phase_completion_sums[phase] / phase_sample_counts[phase], 1)
        for phase in phase_order
        if phase_sample_counts[phase]
    }

    print(f"\n=== {split_name} dataset summary ===")
    print(f"original conversations: {stats.original_rows}")
    print(f"expanded assistant-turn samples: {stats.expanded_rows}")
    print(f"usable samples: {stats.used_rows}")
    print(f"dropped too-long targets: {stats.dropped_too_long_target}")
    print(f"prompt-trimmed samples: {stats.prompt_trimmed}")
    print(f"assistant target kinds: {dict(stats.target_kind)}")
    print(f"assistant phase counts: {ordered_phase_counts}")
    print(f"supervised token mass by phase: {ordered_phase_tokens}")
    print(f"mean completion length by phase: {ordered_phase_means}")
    print(f"original message-count distribution: {dict(stats.original_message_lengths)}")
    print(
        f"orig token lengths -> min={min(lengths)}, p50={percentile(lengths_sorted, 50)}, "
        f"p90={percentile(lengths_sorted, 90)}, p95={percentile(lengths_sorted, 95)}, "
        f"max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}"
    )
    print(
        f"used token lengths -> min={min(used_lengths)}, p50={percentile(used_sorted, 50)}, "
        f"p90={percentile(used_sorted, 90)}, p95={percentile(used_sorted, 95)}, "
        f"max={max(used_lengths)}, mean={sum(used_lengths)/len(used_lengths):.1f}, "
        f"context_limit={max_seq_length}"
    )
    print(f"=== end {split_name} summary ===\n")



def warn_on_schema_mismatch(raw_ds) -> None:
    """Lightweight QA for the canonical nested verdict schema."""
    if len(raw_ds["train"]) == 0:
        return

    checked = min(64, len(raw_ds["train"]))
    malformed = 0
    missing_thresholds = 0
    for i in range(checked):
        messages = raw_ds["train"][i]["messages"]
        final = messages[-1]
        if final.get("role") != "assistant" or "content" not in final:
            continue
        try:
            obj = json.loads(final["content"])
        except Exception:
            continue
        if not isinstance(obj, dict) or "verdict" not in obj or "evidence" not in obj:
            malformed += 1
            continue
        gm = ((obj.get("evidence") or {}).get("global_metrics") or {})
        if gm.get("global_residual_threshold") is None or gm.get("global_residual_ratio") is None:
            missing_thresholds += 1

    if malformed or missing_thresholds:
        print(
            "WARNING: sampled final JSON outputs do not fully match the canonical nested schema. "
            f"malformed={malformed}/{checked}, missing_thresholds={missing_thresholds}/{checked}."
        )



def records_to_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    columns: dict[str, list[Any]] = {key: [r[key] for r in records] for key in records[0].keys()}
    return Dataset.from_dict(columns)


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def decode_token_ids(tokenizer: Any, token_ids: torch.Tensor) -> str:
    decoder = tokenizer if hasattr(tokenizer, "decode") else getattr(tokenizer, "tokenizer", None)
    if decoder is None or not hasattr(decoder, "decode"):
        raise ValueError("Tokenizer does not expose decode() for sanity checks.")
    return decoder.decode(token_ids.detach().cpu(), skip_special_tokens=False)


def build_generation_inputs(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: Any,
) -> dict[str, torch.Tensor]:
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "enable_thinking": False,
    }
    if tools is not None:
        template_kwargs["tools"] = tools

    try:
        inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        prompt = render_text(tokenizer, messages, tools, add_generation_prompt=True)
        inputs = tokenize_text(tokenizer, prompt, return_tensors="pt")
    except Exception:
        prompt = render_text(tokenizer, messages, tools, add_generation_prompt=True)
        inputs = tokenize_text(tokenizer, prompt, return_tensors="pt")

    if isinstance(inputs, str):
        inputs = tokenize_text(tokenizer, inputs, return_tensors="pt")

    if not hasattr(inputs, "items"):
        raise ValueError(f"Unexpected chat-template output type: {type(inputs).__name__}")

    device = model_device(model)
    model_inputs: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)

    if "input_ids" not in model_inputs:
        raise ValueError("Chat template did not produce input_ids for sanity check generation.")
    return model_inputs


def strip_leading_thought_blocks(text: str) -> str:
    remaining = text.lstrip()
    while remaining.startswith(GEMMA_THOUGHT_OPEN):
        remaining = remaining[len(GEMMA_THOUGHT_OPEN):]
        if remaining.startswith("\n"):
            remaining = remaining[1:]
        close_index = remaining.find(GEMMA_CHANNEL_CLOSE)
        if close_index == -1:
            break
        remaining = remaining[close_index + len(GEMMA_CHANNEL_CLOSE):].lstrip()
    return remaining


def completion_token_ids(record: dict[str, Any]) -> list[int]:
    return [
        token_id
        for token_id, is_completion in zip(record["input_ids"], record["completion_mask"])
        if is_completion
    ]


def run_mask_alignment_sanity_check(records: list[dict[str, Any]], tokenizer: Any, sample_count: int) -> None:
    if sample_count <= 0:
        return
    if not records:
        raise ValueError("Mask sanity check requested, but no training records were built.")

    print(f"=== Pre-train mask sanity ({min(sample_count, len(records))} samples) ===")
    checked = 0
    for record in records:
        completion_ids = completion_token_ids(record)
        if not completion_ids:
            raise ValueError("Encountered a training sample with an empty completion span.")

        completion_text = decode_token_ids(tokenizer, torch.tensor(completion_ids, dtype=torch.long))
        candidate = strip_leading_thought_blocks(completion_text).lstrip()
        case_label = record.get("case_path") or "unknown"

        if record.get("target_kind") == "tool_call":
            expected_tool = record.get("expected_tool_name")
            if not isinstance(expected_tool, str) or not expected_tool:
                raise ValueError("Tool-call record is missing expected_tool_name for sanity checking.")
            expected_prefix = f"{GEMMA_TOOL_CALL_OPEN}call:{expected_tool}"
            passed = candidate.startswith(expected_prefix)
            expected_label = expected_prefix
        else:
            expected_prefix = '{"verdict"'
            passed = candidate.startswith(expected_prefix)
            expected_label = expected_prefix

        print(
            f"[mask {checked + 1}/{min(sample_count, len(records))}] "
            f"case={case_label} target={record.get('target_kind')} pass={passed}"
        )
        print(f"  expected={expected_label!r}")
        print(f"  output={candidate[:240]!r}")

        if not passed:
            raise ValueError(
                "Pre-train label sanity failed: masked completion does not start with the expected target prefix. "
                f"case={case_label} target_kind={record.get('target_kind')} expected={expected_label!r} "
                f"got={candidate[:120]!r}"
            )

        checked += 1
        if checked >= sample_count:
            break

    print(f"=== End pre-train mask sanity: {checked}/{checked} passed ===")


def extract_case_path(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        parsed = content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:
                continue
        if isinstance(parsed, dict):
            case_path = parsed.get("case_path")
            if case_path:
                return str(case_path)
    return None


def collect_first_turn_sanity_examples(
    raw_rows: list[dict[str, Any]],
    default_tools: list[dict[str, Any]] | None,
    preserve_system_text: bool,
    phase_gated_prompt: bool,
    sample_count: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if sample_count <= 0:
        return examples

    for row_index, row in enumerate(raw_rows):
        normalized = normalize_messages(row["messages"], preserve_system_text=preserve_system_text)
        expanded = explode_conversation(normalized)
        if not expanded:
            continue

        sample_messages = expanded[0]
        target = sample_messages[-1]
        tool_calls = target.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        function_info = tool_calls[0].get("function")
        if not isinstance(function_info, dict):
            continue
        expected_tool = function_info.get("name")
        if not isinstance(expected_tool, str) or not expected_tool:
            continue

        row_tools = row.get("tools")
        if row_tools is not None:
            if not isinstance(row_tools, list):
                raise ValueError(f"Expected row['tools'] to be a list, got {type(row_tools).__name__}")
            tools = sanitize_tool_schemas(row_tools)
        else:
            tools = default_tools

        phase_messages = phase_gate_messages(
            sample_messages,
            target_kind="tool_call",
            assistant_turn_index=0,
            enabled=phase_gated_prompt,
        )
        history = phase_messages[:-1]
        examples.append(
            {
                "row_index": row_index,
                "case_path": extract_case_path(history),
                "history": history,
                "tools": tools,
                "expected_tool": expected_tool,
            }
        )
        if len(examples) >= sample_count:
            break

    return examples


def run_post_train_sanity_check(
    *,
    model: Any,
    tokenizer: Any,
    raw_rows: list[dict[str, Any]],
    default_tools: list[dict[str, Any]] | None,
    preserve_system_text: bool,
    phase_gated_prompt: bool,
    sample_count: int,
    max_new_tokens: int,
) -> int:
    examples = collect_first_turn_sanity_examples(
        raw_rows,
        default_tools=default_tools,
        preserve_system_text=preserve_system_text,
        phase_gated_prompt=phase_gated_prompt,
        sample_count=sample_count,
    )
    if not examples:
        print("Post-train sanity check skipped: no first-turn tool-call examples were found.")
        return 0

    print(f"=== Post-train first-turn sanity ({len(examples)} samples) ===")
    FastModel.for_inference(model)
    failures = 0

    for index, example in enumerate(examples, start=1):
        model_inputs = build_generation_inputs(tokenizer, example["history"], example["tools"], model)
        input_len = int(model_inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_text = decode_token_ids(tokenizer, output_ids[0][input_len:])
        candidate = strip_leading_thought_blocks(generated_text).lstrip()
        expected_prefix = f"{GEMMA_TOOL_CALL_OPEN}call:{example['expected_tool']}"
        passed = candidate.startswith(expected_prefix)
        case_label = example["case_path"] or f"row{example['row_index']}"
        print(f"[sanity {index}/{len(examples)}] case={case_label} expected={example['expected_tool']} pass={passed}")
        print(f"  output={candidate[:240]!r}")
        if not passed:
            failures += 1

    print(f"=== End post-train sanity: {len(examples) - failures}/{len(examples)} passed ===")
    return failures


def resolve_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)

    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if inner_tokenizer is not None:
        inner_pad_token_id = getattr(inner_tokenizer, "pad_token_id", None)
        if inner_pad_token_id is not None:
            return int(inner_pad_token_id)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None and inner_tokenizer is not None:
        eos_token_id = getattr(inner_tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)

    raise ValueError("Tokenizer does not expose pad_token_id or eos_token_id for batch padding.")


class PretokenizedSFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = resolve_pad_token_id(tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("PretokenizedSFTCollator received an empty batch.")

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        optional_side_inputs = {
            key: any(key in feature for feature in features)
            for key in ("token_type_ids", "mm_token_type_ids")
        }
        side_input_batches: dict[str, list[list[int]]] = {
            key: [] for key, enabled in optional_side_inputs.items() if enabled
        }

        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature.get("attention_mask") or ([1] * len(input_ids)))
            labels = feature.get("labels")
            if labels is None:
                completion_mask = feature.get("completion_mask")
                if completion_mask is None or len(completion_mask) != len(input_ids):
                    raise ValueError("Each feature must provide either labels or a same-length completion_mask.")
                labels = [token_id if is_completion else -100 for token_id, is_completion in zip(input_ids, completion_mask)]
            else:
                labels = list(labels)

            if len(attention_mask) != len(input_ids):
                raise ValueError("attention_mask/input_ids length mismatch in batch feature.")
            if len(labels) != len(input_ids):
                raise ValueError("labels/input_ids length mismatch in batch feature.")

            pad_length = max_length - len(input_ids)
            batch_input_ids.append(input_ids + ([self.pad_token_id] * pad_length))
            batch_attention_mask.append(attention_mask + ([0] * pad_length))
            batch_labels.append(labels + ([-100] * pad_length))
            for key, batch_values in side_input_batches.items():
                raw_values = feature.get(key)
                values = list(raw_values) if raw_values is not None else ([0] * len(input_ids))
                if len(values) != len(input_ids):
                    raise ValueError(f"{key}/input_ids length mismatch in batch feature.")
                batch_values.append(values + ([0] * pad_length))

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        for key, batch_values in side_input_batches.items():
            batch[key] = torch.tensor(batch_values, dtype=torch.long)
        return batch



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal_state = SignalState()
    signal_state.install()
    default_tools = load_tools(args)
    train_file = resolve_dataset_path(args.train_file, "train", required=True)
    valid_file = resolve_dataset_path(args.valid_file, "validation", required=False)
    if not args.model_revision:
        print(
            "WARNING: --model-revision is not pinned. Gemma 4 chat-template and function-calling behavior can drift "
            "across upstream revisions.",
            flush=True,
        )

    print(f"Loading model: {args.model_name}")
    model_kwargs: dict[str, Any] = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "load_in_16bit": args.load_in_16bit,
        "full_finetuning": False,
    }
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model, tokenizer = FastModel.from_pretrained(**model_kwargs)
    lora_target_modules = resolve_lora_target_modules(model, args.lora_target_scope)

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    side_inputs = infer_side_input_plan(model, tokenizer, args.model_name)
    print(f"Gemma side-input plan: {side_inputs.as_dict()}")

    data_files = {"train": train_file}
    if valid_file and has_nonempty_jsonl(valid_file):
        data_files["validation"] = valid_file

    print(f"Loading dataset from: {data_files}")
    raw_ds = {split_name: load_jsonl_rows(path) for split_name, path in data_files.items()}
    if args.max_train_rows > 0:
        original_count = len(raw_ds["train"])
        raw_ds["train"] = raw_ds["train"][: args.max_train_rows]
        print(f"Trimmed train conversations: {len(raw_ds['train'])}/{original_count}")
    if "validation" in raw_ds and args.max_valid_rows > 0:
        original_count = len(raw_ds["validation"])
        raw_ds["validation"] = raw_ds["validation"][: args.max_valid_rows]
        print(f"Trimmed validation conversations: {len(raw_ds['validation'])}/{original_count}")
    warn_on_schema_mismatch(raw_ds)

    train_stats = BuildStats()
    train_records = build_processed_split(
        raw_ds["train"],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        default_tools=default_tools,
        drop_too_long_targets=args.drop_too_long_targets,
        stats=train_stats,
        split_name="train",
        side_inputs=side_inputs,
        preserve_system_text=args.preserve_system_text,
        phase_gated_prompt=args.phase_gated_prompt,
    )
    run_mask_alignment_sanity_check(train_records, tokenizer, args.mask_sanity_samples)
    train_dataset = records_to_dataset(train_records)
    report_dataset(train_records, "train", train_stats, args.max_seq_length)

    eval_dataset = None
    if "validation" in raw_ds:
        eval_stats = BuildStats()
        eval_records = build_processed_split(
            raw_ds["validation"],
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            default_tools=default_tools,
            drop_too_long_targets=args.drop_too_long_targets,
            stats=eval_stats,
            split_name="validation",
            side_inputs=side_inputs,
            preserve_system_text=args.preserve_system_text,
            phase_gated_prompt=args.phase_gated_prompt,
        )
        eval_dataset = records_to_dataset(eval_records)
        report_dataset(eval_records, "validation", eval_stats, args.max_seq_length)

    if len(train_dataset) > 0:
        print("=== Preview of first rendered sample ===")
        print(train_dataset[0]["text"][:2500])
        print("=== End preview ===")

    cpu_count = os.cpu_count() or 1
    dataloader_num_workers = max(0, min(args.dataloader_num_workers, cpu_count))
    training_args = SFTConfig(
        **make_sft_config_kwargs(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        run_name=args.run_name or None,
        max_length=None,
        packing=False,
        completion_only_loss=False,
        save_strategy="steps",
        save_only_model=False,
        save_safetensors=True,
        restore_callback_states_from_checkpoint=True,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        tf32=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        )
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=PretokenizedSFTCollator(tokenizer),
        callbacks=[SaveOnSignalCallback(signal_state)],
    )

    # Workaround: Unsloth randomly replaces custom collators with HF collators in __init__.
    trainer.data_collator = PretokenizedSFTCollator(tokenizer)

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    if resume_checkpoint:
        print(f"Starting training from checkpoint: {resume_checkpoint}")
    else:
        print("Starting training from scratch...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(f"Training metrics: {trainer_stats.metrics}")

    save_dir = Path(args.output_dir) / "lora"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Saved LoRA adapter and tokenizer to: {save_dir}")

    sanity_failures = 0
    if signal_state.received is None and args.sanity_check_samples > 0:
        sanity_failures = run_post_train_sanity_check(
            model=model,
            tokenizer=tokenizer,
            raw_rows=raw_ds["train"],
            default_tools=default_tools,
            preserve_system_text=args.preserve_system_text,
            phase_gated_prompt=args.phase_gated_prompt,
            sample_count=args.sanity_check_samples,
            max_new_tokens=args.sanity_check_max_new_tokens,
        )
        if sanity_failures > 0:
            print(f"WARNING: post-train first-turn sanity failed on {sanity_failures} sample(s).")
            if args.sanity_check_fail_on_miss:
                return 2

    if signal_state.received is not None:
        try:
            signame = signal.Signals(signal_state.received).name
        except Exception:
            signame = str(signal_state.received)
        print(
            f"Training stopped after {signame}; checkpoint and adapter were saved. "
            f"Exiting with code {PREEMPTION_EXIT_CODE} for launcher-side requeue.",
            flush=True,
        )
        return PREEMPTION_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
