#!/usr/bin/env python3
"""Dump turn-1 prompt diffs between SFT sanity and eval rendering paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_adapter_loader import resolve_tokenizer_source
from gpt_oss_power_sft_revised_v3 import (
    DEFAULT_POWER_TOOLS,
    dump_turn1_prompt_diagnostics,
    load_jsonl_rows,
    sanitize_tool_schemas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump rendered turn-1 prompt diagnostics")
    parser.add_argument("--adapter", required=True, help="Path to saved LoRA adapter directory")
    parser.add_argument(
        "--train-file",
        default="artifacts/traces/out_traces_balanced/sft_traces.train.jsonl",
        help="Training JSONL used by SFT",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gemma4_power_agent/prompt_diagnostics_manual",
        help="Directory where prompt files and unified diffs are written",
    )
    parser.add_argument(
        "--model-revision",
        default="",
        help="Pinned base-model revision to use when loading the base tokenizer",
    )
    parser.add_argument(
        "--prefer-base-tokenizer",
        action="store_true",
        default=False,
        help="Prefer base tokenizer instead of adapter-local tokenizer artifacts",
    )
    parser.add_argument(
        "--tools-file",
        default="",
        help="Optional JSON file containing a list of tool schemas",
    )
    parser.add_argument("--preserve-system-text", action="store_true", default=False)
    parser.add_argument("--no-phase-gated-prompt", dest="phase_gated_prompt", action="store_false", default=True)
    parser.add_argument("--no-inject-empty-thought-channel", dest="inject_empty_thought_channel", action="store_false", default=True)
    return parser.parse_args()


def load_default_tools(path: str) -> list[dict] | None:
    if not path:
        return DEFAULT_POWER_TOOLS
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected tools file to contain a list, got {type(data).__name__}")
    return sanitize_tool_schemas(data)


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    adapter_path = Path(args.adapter)
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    base_model_name = adapter_config["base_model_name_or_path"]
    tokenizer_name, tokenizer_source, _ = resolve_tokenizer_source(
        adapter_path,
        base_model_name=base_model_name,
        prefer_base_tokenizer=args.prefer_base_tokenizer,
    )

    tokenizer_kwargs = {"trust_remote_code": True}
    if args.model_revision and tokenizer_name == base_model_name:
        tokenizer_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    print(f"Tokenizer source: {tokenizer_source}")
    dump_turn1_prompt_diagnostics(
        raw_rows=load_jsonl_rows(args.train_file),
        default_tools=load_default_tools(args.tools_file),
        tokenizer=tokenizer,
        preserve_system_text=args.preserve_system_text,
        phase_gated_prompt=args.phase_gated_prompt,
        phase_role="system",
        inject_empty_thought_channel=args.inject_empty_thought_channel,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
