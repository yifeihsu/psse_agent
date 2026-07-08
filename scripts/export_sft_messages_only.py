#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trace_protocol import canonical_tool_schemas


FORBIDDEN_KEYS = {"label", "scenario_model_dir", "legacy_line_group_index0", "runtime_context"}


def _json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _strip_forbidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden(item)
            for key, item in value.items()
            if key not in FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_strip_forbidden(item) for item in value]
    return value


def _maybe_sanitize_json_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return value
    cleaned = _strip_forbidden(parsed)
    return _json_compact(cleaned)


def _normalize_tool_calls(message: dict[str, Any], *, dict_tool_arguments: bool) -> None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        args = function.get("arguments")
        if isinstance(args, str) and dict_tool_arguments:
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                parsed = {}
            function["arguments"] = _strip_forbidden(parsed if isinstance(parsed, dict) else {})
        elif isinstance(args, dict):
            function["arguments"] = _strip_forbidden(args)


def clean_messages(row: Mapping[str, Any], *, dict_tool_arguments: bool) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row missing messages list")
    cleaned_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        cleaned = {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if key in {"role", "content", "tool_calls", "tool_call_id", "name"}
        }
        if "content" in cleaned:
            cleaned["content"] = _maybe_sanitize_json_text(cleaned["content"])
        _normalize_tool_calls(cleaned, dict_tool_arguments=dict_tool_arguments)
        cleaned_messages.append(_strip_forbidden(cleaned))
    return cleaned_messages


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN_KEYS or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--keep-json-string-tool-arguments",
        action="store_true",
        help="Keep OpenAI-style JSON-string tool_call arguments. Default converts to dicts for HF/Transformers.",
    )
    parser.add_argument(
        "--omit-tools",
        action="store_true",
        help="Write only messages. Default includes the canonical tools column for HF/TRL tool-calling SFT.",
    )
    parser.add_argument("--fail-on-forbidden", action="store_true")
    args = parser.parse_args()

    rows = 0
    tools = None if args.omit_tools else canonical_tool_schemas()
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            out_row = {
                "messages": clean_messages(
                    raw,
                    dict_tool_arguments=not bool(args.keep_json_string_tool_arguments),
                )
            }
            if tools is not None:
                out_row["tools"] = tools
            if args.fail_on_forbidden and _contains_forbidden_key(out_row):
                raise ValueError(f"Forbidden metadata key remains after cleaning at input line {line_no}")
            dst.write(_json_compact(out_row) + "\n")
            rows += 1
    print(json.dumps({"rows": rows, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
