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
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--drop-too-long-targets", action="store_true", default=True)
    parser.add_argument("--keep-too-long-targets", dest="drop_too_long_targets", action="store_false")
    parser.add_argument("--include-tool-schemas", action="store_true", default=True)
    parser.add_argument("--no-include-tool-schemas", dest="include_tool_schemas", action="store_false")
    parser.add_argument("--tools-file", type=str, default="")
    parser.add_argument(
        "--preserve-system-text",
        action="store_true",
        default=False,
        help="Do not rewrite Harmony/GPT-OSS-specific system/developer wording.",
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
):
    records: list[dict[str, Any]] = []

    for row in raw_split:
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

        for sample_messages in expanded:
            target = sample_messages[-1]
            target_kind = "tool_call" if "tool_calls" in target else "final"
            stats.target_kind[target_kind] += 1

            history = sample_messages[:-1]
            full_text = render_text(tokenizer, sample_messages, tools, add_generation_prompt=False)
            prompt_text = render_text(tokenizer, history, tools, add_generation_prompt=True)

            prompt_enc = encode_text(tokenizer, prompt_text)
            full_enc = encode_text(tokenizer, full_text)
            prompt_ids = full_enc["input_ids"][:0] + prompt_enc["input_ids"]
            full_ids = full_enc["input_ids"]

            if len(full_ids) < len(prompt_ids):
                raise ValueError(
                    f"Prompt tokenization longer than full sample in {split_name}; this should not happen."
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

    print(f"\n=== {split_name} dataset summary ===")
    print(f"original conversations: {stats.original_rows}")
    print(f"expanded assistant-turn samples: {stats.expanded_rows}")
    print(f"usable samples: {stats.used_rows}")
    print(f"dropped too-long targets: {stats.dropped_too_long_target}")
    print(f"prompt-trimmed samples: {stats.prompt_trimmed}")
    print(f"assistant target kinds: {dict(stats.target_kind)}")
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

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    )
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
