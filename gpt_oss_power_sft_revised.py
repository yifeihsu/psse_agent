#!/usr/bin/env python3
"""SFT for GPT-OSS-20B on power-system tool traces.

Design choices:
1) Normalize OpenAI-style tool call arguments back into Python dicts/lists before
   applying the Transformers chat template.
2) Expand each conversation into one training sample per assistant turn so GPT-OSS
   always learns from the *final* assistant action of the current sample.
3) Build explicit completion masks from prompt/completion boundaries instead of
   relying on fragile string-marker masking.
4) Truncate only the prompt prefix when sequences exceed max length; never cut the
   target assistant action unless the action itself is longer than the context.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetDict, load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-OSS SFT for power-system tool traces")
    parser.add_argument("--train-file", type=str, default="data/sft_final.jsonl")
    parser.add_argument("--valid-file", type=str, default="data/split_valid.jsonl")
    parser.add_argument("--model-name", type=str, default="unsloth/gpt-oss-20b")
    parser.add_argument("--output-dir", type=str, default="outputs/gpt_oss_power_agent")
    parser.add_argument("--max-seq-length", type=int, default=8192)
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
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", type=str, default="none", help="The integration to report the results to, e.g., 'wandb'")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--drop-too-long-targets", action="store_true", default=True)
    parser.add_argument("--keep-too-long-targets", dest="drop_too_long_targets", action="store_false")
    parser.add_argument("--include-tool-schemas", action="store_true", default=True)
    parser.add_argument("--no-include-tool-schemas", dest="include_tool_schemas", action="store_false")
    parser.add_argument("--tools-file", type=str, default="")
    return parser.parse_args()


def has_nonempty_jsonl(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with p.open("r", encoding="utf-8") as f:
        return any(line.strip() for line in f)


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
    split_aliases = {"split_train.jsonl": "train", "split_valid.jsonl": "valid", "split_test.jsonl": "test"}
    split_suffix = split_aliases.get(requested.name)

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

    if split_suffix is not None:
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            matches = sorted(directory.glob(f"*.{split_suffix}.jsonl"))
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


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = prune_none(raw_message)
        role = msg.get("role")
        if role is None:
            continue

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
            # Tool outputs should stay strings for chat templates.
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
    """Create one training sample per assistant turn.

    GPT-OSS should learn from the final assistant action of each sample. This is
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
    return sanitize_tool_schemas(DEFAULT_POWER_TOOLS)


def render_text(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, add_generation_prompt: bool) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


class BuildStats:
    def __init__(self) -> None:
        self.original_rows = 0
        self.expanded_rows = 0
        self.used_rows = 0
        self.prompt_trimmed = 0
        self.dropped_too_long_target = 0
        self.target_kind = Counter()
        self.original_message_lengths = Counter()



def build_processed_split(
    raw_split,
    tokenizer,
    max_seq_length: int,
    tools: list[dict[str, Any]] | None,
    drop_too_long_targets: bool,
    stats: BuildStats,
    split_name: str,
):
    records: list[dict[str, Any]] = []

    for row in raw_split:
        stats.original_rows += 1
        raw_messages = row["messages"]
        stats.original_message_lengths[len(raw_messages)] += 1
        normalized = normalize_messages(raw_messages)
        expanded = explode_conversation(normalized)
        stats.expanded_rows += len(expanded)

        for sample_messages in expanded:
            target = sample_messages[-1]
            target_kind = "tool_call" if "tool_calls" in target else "final"
            stats.target_kind[target_kind] += 1

            history = sample_messages[:-1]
            full_text = render_text(tokenizer, sample_messages, tools, add_generation_prompt=False)
            prompt_text = render_text(tokenizer, history, tools, add_generation_prompt=True)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

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
                    # As a fallback, keep the tail of the target only.
                    input_ids = full_ids[-max_seq_length:]
                    completion_mask = [1] * max_seq_length
                    truncated = True
                else:
                    keep_prompt = max_seq_length - completion_len
                    input_ids = full_ids[-(keep_prompt + completion_len):]
                    completion_mask = [0] * keep_prompt + [1] * completion_len
                    stats.prompt_trimmed += 1
                    truncated = True
            else:
                input_ids = full_ids
                completion_mask = [0] * len(prompt_ids) + [1] * completion_len

            attention_mask = [1] * len(input_ids)
            if len(input_ids) != len(completion_mask):
                raise ValueError("input_ids/completion_mask length mismatch")

            records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "completion_mask": completion_mask,
                    "orig_length": orig_length,
                    "used_length": len(input_ids),
                    "completion_length": completion_len,
                    "target_kind": target_kind,
                    "was_truncated": truncated,
                    "text": full_text,  # kept for debugging/preview only
                }
            )
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
    """Lightweight data QA for the uploaded dataset.

    The current power-system dataset often asks for `decision_basis` in the system
    instruction while omitting it in the final JSON. Warn early so the user can
    decide whether to fix the dataset or prompt.
    """
    if len(raw_ds["train"]) == 0:
        return

    system_text = raw_ds["train"][0]["messages"][0].get("content", "")
    if "decision_basis" not in system_text:
        return

    checked = min(64, len(raw_ds["train"]))
    missing = 0
    for i in range(checked):
        messages = raw_ds["train"][i]["messages"]
        final = messages[-1]
        if final.get("role") != "assistant" or "content" not in final:
            continue
        try:
            obj = json.loads(final["content"])
        except Exception:
            continue
        if "decision_basis" not in obj:
            missing += 1

    if missing:
        print(
            f"WARNING: system prompt asks for `decision_basis`, but {missing}/{checked} sampled final JSON outputs omit it. "
            "Either add the field to targets or relax the instruction."
        )



def to_hf_dataset(records: list[dict[str, Any]]):
    columns: dict[str, list[Any]] = {}
    for key in records[0].keys():
        columns[key] = [r[key] for r in records]
    return DatasetDict.from_dict(columns)  # type: ignore[attr-defined]


# datasets.Dataset.from_dict, but keep this helper to avoid accidental column drift.
def records_to_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    columns: dict[str, list[Any]] = {key: [r[key] for r in records] for key in records[0].keys()}
    return Dataset.from_dict(columns)



def main() -> None:
    args = parse_args()
    tools = load_tools(args)
    train_file = resolve_dataset_path(args.train_file, "train", required=True)
    valid_file = resolve_dataset_path(args.valid_file, "validation", required=False)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    data_files = {"train": train_file}
    if valid_file and has_nonempty_jsonl(valid_file):
        data_files["validation"] = valid_file

    print(f"Loading dataset from: {data_files}")
    raw_ds = load_dataset("json", data_files=data_files)
    warn_on_schema_mismatch(raw_ds)

    train_stats = BuildStats()
    train_records = build_processed_split(
        raw_ds["train"],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tools=tools,
        drop_too_long_targets=args.drop_too_long_targets,
        stats=train_stats,
        split_name="train",
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
            tools=tools,
            drop_too_long_targets=args.drop_too_long_targets,
            stats=eval_stats,
            split_name="validation",
        )
        eval_dataset = records_to_dataset(eval_records)
        report_dataset(eval_records, "validation", eval_stats, args.max_seq_length)

    if len(train_dataset) > 0:
        print("=== Preview of first rendered sample ===")
        print(train_dataset[0]["text"][:2500])
        print("=== End preview ===")

    training_args = SFTConfig(
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
        weight_decay=0.001,
        lr_scheduler_type="linear",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        max_length=None,
        packing=False,
        completion_only_loss=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training metrics: {trainer_stats.metrics}")

    save_dir = Path(args.output_dir) / "lora"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Saved LoRA adapter and tokenizer to: {save_dir}")


if __name__ == "__main__":
    main()
