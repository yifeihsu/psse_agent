#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from trace_protocol import canonical_tool_schemas, normalize_instruction_content


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export nested chat traces into flat prompt/completion JSONL files for HF training UIs."
    )
    p.add_argument("--train-file", default="out_traces_balanced/sft_traces.train.jsonl")
    p.add_argument("--valid-file", default="out_traces_balanced/sft_traces.valid.jsonl")
    p.add_argument("--test-file", default="out_traces_balanced/sft_traces.test.jsonl")
    p.add_argument("--output-dir", default="out_traces_flat_hf")
    p.add_argument("--model-name", default="unsloth/Gemma-4-26B-A4B-it")
    p.add_argument(
        "--no-include-tool-schemas",
        dest="include_tool_schemas",
        action="store_false",
        help="Disable tool schemas when rendering prompts/completions.",
    )
    p.set_defaults(include_tool_schemas=True)
    return p.parse_args()


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
    if not s or s[0] not in "[{":
        return value
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return value


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = prune_none(raw_message)
        role = msg.get("role")
        if role is None:
            continue

        if "content" in msg:
            msg["content"] = normalize_instruction_content(role, msg.get("content"))
        if "content" in msg and not isinstance(msg["content"], str):
            msg["content"] = json.dumps(msg["content"], ensure_ascii=False)

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                fixed_calls = []
                for tool_call in tool_calls:
                    tc = prune_none(tool_call)
                    function_info = tc.get("function")
                    if isinstance(function_info, dict):
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
                msg.setdefault("content", "")

        elif role == "tool":
            msg.setdefault("content", "")
            if not isinstance(msg["content"], str):
                msg["content"] = json.dumps(msg["content"], ensure_ascii=False)

        elif role in {"user", "system", "developer"}:
            msg.setdefault("content", "")

        normalized.append(msg)
    return normalized


def assistant_turn_indices(messages: list[dict[str, Any]]) -> list[int]:
    return [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]


def explode_conversation(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [messages[: idx + 1] for idx in assistant_turn_indices(messages)]


def render_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def extract_final_verdict(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            obj = json.loads(content)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("verdict"), dict):
            return obj
    return None


def decode_ids(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def flatten_split(
    tokenizer: Any,
    src_path: Path,
    dst_path: Path,
    tools: list[dict[str, Any]] | None,
) -> dict[str, int]:
    rows_in = 0
    rows_out = 0
    target_kind_counts = {"tool_call": 0, "final": 0}

    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for row_index, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            rows_in += 1
            row = json.loads(line)
            raw_messages = row.get("messages")
            if not isinstance(raw_messages, list):
                continue

            final_verdict = extract_final_verdict(raw_messages) or {}
            verdict = final_verdict.get("verdict") or {}

            normalized = normalize_messages(raw_messages)
            expanded = explode_conversation(normalized)

            for turn_index, sample_messages in enumerate(expanded, start=1):
                target = sample_messages[-1]
                history = sample_messages[:-1]
                target_kind = "tool_call" if "tool_calls" in target else "final"

                full_text = render_text(tokenizer, sample_messages, tools, add_generation_prompt=False)
                prompt_text = render_text(tokenizer, history, tools, add_generation_prompt=True)

                full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                if len(full_ids) < len(prompt_ids):
                    raise ValueError(
                        f"Tokenized full text shorter than prompt text in {src_path} row {row_index + 1}."
                    )
                completion_ids = full_ids[len(prompt_ids) :]
                if not completion_ids:
                    raise ValueError(
                        f"Empty completion in {src_path} row {row_index + 1}, assistant turn {turn_index}."
                    )

                flat_row = {
                    "prompt": decode_ids(tokenizer, prompt_ids),
                    "completion": decode_ids(tokenizer, completion_ids),
                    "target_kind": target_kind,
                    "error_family": verdict.get("error_family"),
                    "has_error": verdict.get("has_error"),
                    "confidence": verdict.get("confidence"),
                    "source_row": row_index,
                    "assistant_turn_index": turn_index,
                }
                dst.write(json.dumps(flat_row, ensure_ascii=False) + "\n")
                rows_out += 1
                target_kind_counts[target_kind] += 1

    return {
        "input_rows": rows_in,
        "output_rows": rows_out,
        "tool_call_rows": target_kind_counts["tool_call"],
        "final_rows": target_kind_counts["final"],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tools = canonical_tool_schemas() if args.include_tool_schemas else None

    split_map = {
        "train": (Path(args.train_file), output_dir / "train.jsonl"),
        "validation": (Path(args.valid_file), output_dir / "validation.jsonl"),
        "test": (Path(args.test_file), output_dir / "test.jsonl"),
    }

    report: dict[str, Any] = {
        "model_name": args.model_name,
        "include_tool_schemas": args.include_tool_schemas,
        "splits": {},
    }

    for split_name, (src, dst) in split_map.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source split: {src}")
        stats = flatten_split(tokenizer, src, dst, tools)
        report["splits"][split_name] = {
            "source": str(src),
            "output": str(dst),
            **stats,
        }
        print(
            f"{split_name}: {stats['input_rows']} source rows -> {stats['output_rows']} flat rows "
            f"({stats['tool_call_rows']} tool_call, {stats['final_rows']} final)"
        )

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Flat Prompt/Completion Export",
                "",
                "This directory contains a flattened prompt/completion export derived from the nested PSSE agent traces.",
                "",
                "Files:",
                "- `train.jsonl`",
                "- `validation.jsonl`",
                "- `test.jsonl`",
                "- `export_report.json`",
                "",
                "Each row contains flat scalar columns suitable for HF training UIs:",
                "- `prompt`",
                "- `completion`",
                "- `target_kind`",
                "- `error_family`",
                "- `has_error`",
                "- `confidence`",
                "- `source_row`",
                "- `assistant_turn_index`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
