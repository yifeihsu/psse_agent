from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from trace_protocol import (
    canonical_tool_schemas,
    CONTEXT_TOOL_NAMES,
    extract_conversation_context,
    hydrate_tool_arguments as protocol_hydrate_tool_arguments,
    round_tool_result_payload,
    resolve_case_path_alias,
    summarize_tool_result_for_conversation,
)

from mcp_server.matpower_server import (
    correct_measurements_from_path,
    correct_parameters_from_path,
    correct_topology_from_path,
    estimate_hif_location_magnitude_from_path,
    run_hse_from_path,
    run_three_phase_nlm_from_path,
    wls_from_path,
)

SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_TOOL_SCHEMAS: list[dict[str, Any]] = [
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
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "z": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Observed measurement vector.",
                    },
                    "suspect_group": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "0-based measurement indices to correct.",
                    },
                    "enable_correction": {
                        "type": "boolean",
                        "description": "Whether to apply the grouped correction step.",
                    },
                    "max_correction_iterations": {
                        "type": "integer",
                        "description": "Maximum correction passes to run.",
                    },
                    "error_tolerance": {
                        "type": "number",
                        "description": "Stop correcting once the estimated error falls below this tolerance.",
                    },
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
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "line_index": {"type": "integer", "description": "1-based MATPOWER branch row index."},
                    "z_scans": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Measurement snapshots across repeated scans.",
                    },
                    "initial_states": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Initial state vectors aligned to the repeated scans.",
                    },
                },
                "required": ["case_path", "line_index", "z_scans", "initial_states"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_topology_from_path",
            "description": "Correct a suspected topology mismatch by switching a breaker status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "cb_name": {"type": "string", "description": "Circuit-breaker name."},
                    "desired_status": {
                        "type": "boolean",
                        "description": "Target breaker status; true for closed, false for open.",
                    },
                },
                "required": ["case_path", "cb_name", "desired_status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_hse_from_path",
            "description": "Run harmonic state estimation to identify a single harmonic source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "harmonic_measurements": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Per-harmonic complex voltage measurements.",
                    },
                    "harmonic_orders": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Harmonic orders to evaluate.",
                    },
                    "slack_bus": {
                        "type": "integer",
                        "description": "0-based slack bus index.",
                    },
                },
                "required": ["case_path", "harmonic_measurements"],
            },
        },
    },
]

TOOL_MAP = {
    "wls_from_path": wls_from_path,
    "correct_measurements_from_path": correct_measurements_from_path,
    "correct_parameters_from_path": correct_parameters_from_path,
    "correct_topology_from_path": correct_topology_from_path,
    "estimate_hif_location_magnitude_from_path": estimate_hif_location_magnitude_from_path,
    "run_hse_from_path": run_hse_from_path,
    "run_three_phase_nlm_from_path": run_three_phase_nlm_from_path,
}


def resolve_existing_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path]
        script_relative = SCRIPT_DIR / path
        if script_relative != candidates[0]:
            candidates.append(script_relative)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found: {raw_path}. Looked in: {searched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned GPT-OSS agent with local tools.")
    parser.add_argument(
        "--adapter-path",
        default="outputs/gpt_oss_sft_power_agent_4k/lora",
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--trace-file",
        default="tmp_show_trace.json",
        help="JSON or JSONL file containing at least a system+user tool-use trace.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row to load when --trace-file points to a JSONL dataset.",
    )
    parser.add_argument(
        "--initial-messages",
        type=int,
        default=2,
        help="Number of messages to seed the conversation with. Use 2 for system+user-only eval.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/final turns to generate.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation limit per step.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Context window passed to Unsloth.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="GPT-OSS reasoning effort during inference.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 disables sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p used only when temperature > 0.")
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional path to save the full evaluated transcript and raw generations as JSON.",
    )
    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="Print the reference final assistant message from the trace, if present.",
    )
    return parser.parse_args()


def load_trace_messages(trace_file: Path, sample_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    if trace_file.suffix.lower() == ".jsonl":
        with trace_file.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                if row_index == sample_index:
                    payload = json.loads(line)
                    messages = payload.get("messages")
                    if not isinstance(messages, list):
                        raise ValueError(f"Row {sample_index} in {trace_file} does not contain a 'messages' list.")
                    return messages, payload
        raise IndexError(f"Sample index {sample_index} is outside the range of {trace_file}.")

    with trace_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return payload["messages"], payload
    if isinstance(payload, list):
        return payload, {"messages": payload}
    raise ValueError(f"Unsupported trace format in {trace_file}; expected a messages list or object containing one.")


def find_reference_final(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def extract_reference_user_followups(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_messages = [dict(message) for message in messages if message.get("role") == "user"]
    return user_messages[1:]


def pretty_json(value: Any, *, limit: int | None = None) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n...<truncated>..."
    return text


def extract_analysis_block(raw_text: str) -> str | None:
    marker = "<|channel|>analysis<|message|>"
    if marker not in raw_text:
        return None
    start = raw_text.find(marker) + len(marker)
    end = raw_text.find("<|end|>", start)
    if end == -1:
        end = len(raw_text)
    analysis = raw_text[start:end].strip()
    return analysis or None


def parse_generation(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    analysis = extract_analysis_block(raw_text)

    if "to=functions." in raw_text and "<|call|>" in raw_text:
        name_start = raw_text.find("to=functions.") + len("to=functions.")
        name_end = raw_text.find("<|channel|>", name_start)
        if name_end == -1:
            raise ValueError(f"Could not parse tool name from generation:\n{raw_text}")

        tool_name = raw_text[name_start:name_end].strip()
        message_start = raw_text.find("<|message|>", name_end)
        call_end = raw_text.find("<|call|>", message_start)
        if message_start == -1 or call_end == -1:
            raise ValueError(f"Could not parse tool arguments from generation:\n{raw_text}")

        arguments_text = raw_text[message_start + len("<|message|>"):call_end].strip()
        arguments = json.loads(arguments_text)
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool call arguments for {tool_name} are not a JSON object: {arguments_text}")

        return {
            "kind": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "arguments_text": arguments_text,
            "analysis": analysis,
        }

    message_pos = raw_text.rfind("<|message|>")
    content_start = message_pos + len("<|message|>") if message_pos != -1 else 0

    end_positions = []
    for marker in ("<|return|>", "<|end|>", "<|call|>"):
        pos = raw_text.find(marker, content_start)
        if pos != -1:
            end_positions.append(pos)
    content_end = min(end_positions) if end_positions else len(raw_text)
    content = raw_text[content_start:content_end].strip()

    parsed_json = None
    if content.startswith("{") or content.startswith("["):
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            parsed_json = None

    if parsed_json is not None:
        return {
            "kind": "final",
            "content": content,
            "parsed_json": parsed_json,
            "analysis": analysis,
        }

    return {
        "kind": "assistant_message",
        "content": content,
        "parsed_json": None,
        "analysis": analysis,
    }


def build_tool_call_message(step: int, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": f"call_{tool_name}_{step:02d}",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        ],
    }


def build_tool_result_message(step: int, tool_name: str, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": f"call_{tool_name}_{step:02d}",
        "name": tool_name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def execute_context_tool(
    tool_name: str,
    arguments: dict[str, Any],
    runtime_context: dict[str, Any] | None,
    hidden_context: dict[str, Any],
) -> Any:
    tool_context = ((runtime_context or {}).get("tool_context") or {})
    payload: Any = None
    if tool_name == "get_parameter_context":
        payload = tool_context.get("parameter_context")
        if isinstance(payload, dict):
            hidden_context["parameter_context"] = payload
    elif tool_name == "get_topology_context":
        payload = tool_context.get("topology_context")
        if isinstance(payload, dict):
            hidden_context["topology_context"] = payload
    elif tool_name == "get_harmonic_context":
        payload = tool_context.get("harmonic_context")
        if isinstance(payload, dict):
            hidden_context["harmonic_context"] = payload
    elif tool_name == "get_verification_snapshot":
        stage = arguments.get("stage")
        payload = (tool_context.get("verification_snapshots") or {}).get(stage)
        if isinstance(payload, dict):
            hidden_context["snapshot_context"] = payload
    if isinstance(payload, dict):
        return payload
    return {"success": False, "error": f"Missing runtime context for {tool_name}"}


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    runtime_context: dict[str, Any] | None = None,
    hidden_context: dict[str, Any] | None = None,
) -> Any:
    if tool_name in CONTEXT_TOOL_NAMES:
        return execute_context_tool(tool_name, arguments, runtime_context, hidden_context or {})
    if tool_name not in TOOL_MAP:
        return {"success": False, "error": f"Unknown tool requested by model: {tool_name}"}
    tool = TOOL_MAP[tool_name]
    try:
        call_args = dict(arguments)
        if "case_path" in call_args:
            call_args["case_path"] = resolve_case_path_alias(call_args["case_path"], hidden_context or runtime_context)
        if callable(tool):
            return tool(**call_args)
        if hasattr(tool, "fn") and callable(tool.fn):
            return tool.fn(**call_args)
        return {"success": False, "error": f"Tool {tool_name} is not directly callable and exposes no callable fn."}
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def save_run(output_file: Path, payload: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_stop_token_ids(tokenizer) -> list[int]:
    stop_token_ids: list[int] = []
    for token in ("<|call|>", "<|return|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            stop_token_ids.append(token_id)
    return sorted(set(stop_token_ids))


def main() -> None:
    from unsloth import FastLanguageModel

    args = parse_args()
    adapter_path = resolve_existing_path(args.adapter_path, label="LoRA adapter directory")
    trace_file = resolve_existing_path(args.trace_file, label="Trace file")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local GPT-OSS-20B 4-bit evaluation in this script.")

    messages, trace_payload = load_trace_messages(trace_file, args.sample_index)
    if len(messages) < args.initial_messages:
        raise ValueError(
            f"Trace only has {len(messages)} message(s); cannot seed with {args.initial_messages} initial messages."
        )

    seed_messages = messages[: args.initial_messages]
    reference_final = find_reference_final(messages)

    print(f"Loading adapter from: {adapter_path}")
    print(f"Using trace file: {trace_file}")
    if trace_file.suffix.lower() == ".jsonl":
        print(f"Using JSONL sample index: {args.sample_index}")
    print(f"Seeding conversation with {len(seed_messages)} message(s)")
    print("Loading GPT-OSS-20B in 4-bit. Expect roughly 13 GB of VRAM use.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    stop_token_ids = resolve_stop_token_ids(tokenizer)

    conversation = list(seed_messages)
    reference_user_followups = extract_reference_user_followups(messages)
    replayed_user_followups = 0
    run_steps: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    tool_schemas = canonical_tool_schemas()
    runtime_context = trace_payload.get("runtime_context") if isinstance(trace_payload, dict) else None
    hidden_context: dict[str, Any] = {
        "case_aliases": dict(((runtime_context or {}).get("case_aliases") or {})),
    }

    for step in range(1, args.max_steps + 1):
        print(f"\n=== Step {step} ===")
        model_inputs = tokenizer.apply_chat_template(
            conversation,
            tools=tool_schemas,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=args.reasoning_effort,
        ).to("cuda")

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "do_sample": args.temperature > 0,
        }
        if stop_token_ids:
            generate_kwargs["eos_token_id"] = stop_token_ids
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            outputs = model.generate(**model_inputs, **generate_kwargs)

        prompt_length = model_inputs["input_ids"].shape[-1]
        new_tokens = outputs[0][prompt_length:]
        raw_generation = tokenizer.decode(new_tokens, skip_special_tokens=False)

        step_record: dict[str, Any] = {
            "step": step,
            "raw_generation": raw_generation,
        }

        print("[Raw generation]")
        print(raw_generation.strip() or "<empty>")

        parsed = parse_generation(raw_generation)
        step_record["parsed"] = parsed

        if parsed.get("analysis"):
            print("[Analysis block]")
            print(parsed["analysis"])

        if parsed["kind"] == "tool_call":
            tool_name = parsed["tool_name"]
            arguments, hydration_notes = protocol_hydrate_tool_arguments(
                tool_name,
                parsed["arguments"],
                conversation,
                hidden_context=hidden_context,
            )
            step_record["hydration_notes"] = hydration_notes

            print(f"[Tool call] {tool_name}")
            print(pretty_json(arguments, limit=3000))

            tool_result = execute_tool(
                tool_name,
                arguments,
                runtime_context=runtime_context,
                hidden_context=hidden_context,
            )
            meta_context, index_map = extract_conversation_context(conversation)
            try:
                tool_result_compact = summarize_tool_result_for_conversation(
                    tool_name,
                    tool_result,
                    meta_context,
                    index_map,
                )
            except Exception:
                tool_result_compact = tool_result
            stored_tool_result = tool_result_compact if tool_name in CONTEXT_TOOL_NAMES else tool_result
            step_record["tool_result"] = round_tool_result_payload(stored_tool_result)
            step_record["tool_result_compact"] = round_tool_result_payload(tool_result_compact)

            print("[Tool result]")
            print(pretty_json(tool_result_compact, limit=5000))

            conversation.append(build_tool_call_message(step, tool_name, arguments))
            conversation.append(build_tool_result_message(step, tool_name, tool_result_compact))
            run_steps.append(step_record)
            continue

        if parsed["kind"] == "assistant_message":
            assistant_content = parsed["content"]
            step_record["assistant_message"] = assistant_content
            print("[Assistant message]")
            print(assistant_content or "<empty>")
            conversation.append({"role": "assistant", "content": assistant_content})
            if replayed_user_followups >= len(reference_user_followups):
                print("No reference user follow-up remains; stopping.")
                run_steps.append(step_record)
                final_result = step_record
                break
            followup_message = reference_user_followups[replayed_user_followups]
            replayed_user_followups += 1
            conversation.append(followup_message)
            step_record["replayed_user_followup"] = followup_message
            print("[Replayed user follow-up]")
            print(pretty_json(followup_message, limit=3000))
            run_steps.append(step_record)
            continue

        final_content = parsed["content"]
        final_json = parsed["parsed_json"]
        step_record["final_content"] = final_content
        step_record["final_json"] = final_json

        print("[Final response]")
        print(final_content or "<empty>")

        conversation.append({"role": "assistant", "content": final_content})
        run_steps.append(step_record)
        final_result = step_record
        break
    else:
        print(f"Stopped after reaching max_steps={args.max_steps} without a final assistant answer.")

    if args.show_reference and reference_final:
        print("\n=== Reference Final Message ===")
        print(reference_final)

    output_payload = {
        "adapter_path": str(adapter_path),
        "trace_file": str(trace_file),
        "sample_index": args.sample_index,
        "initial_messages": args.initial_messages,
        "reasoning_effort": args.reasoning_effort,
        "timestamp": datetime.now().isoformat(),
        "seed_messages": seed_messages,
        "steps": run_steps,
        "final_result": final_result,
        "reference_final": reference_final,
        "conversation": conversation,
        "trace_metadata": {k: v for k, v in trace_payload.items() if k != "messages"},
    }

    if args.output_file:
        output_file = Path(args.output_file)
        save_run(output_file, output_payload)
        print(f"\nSaved run output to: {output_file}")


if __name__ == "__main__":
    main()
