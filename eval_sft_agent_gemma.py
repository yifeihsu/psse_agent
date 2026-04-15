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
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from trace_protocol import normalize_instruction_content

from eval_sft_agent_hardened import (
    classify_result_error,
    execute_tool,
    extract_ground_truth,
    extract_prompt_prefix,
    extract_user_snapshot,
    jsonish_loads,
    load_tools,
    normalize_verdict,
    repair_tool_arguments,
    resolve_max_input_tokens,
)


GEMMA_TOOL_CALL_OPEN = "<|tool_call>"
GEMMA_TOOL_CALL_CLOSE = "<tool_call|>"
GEMMA_TOOL_RESPONSE_CLOSE = "<tool_response|>"
GEMMA_TURN_CLOSE = "<turn|>"
GEMMA_THOUGHT_OPEN = "<|channel>thought"
GEMMA_CHANNEL_CLOSE = "<channel|>"
GEMMA_QUOTE_TOKEN = '<|"|>'


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
        "--max-samples",
        type=int,
        default=None,
        help="Cap on the number of samples to evaluate",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Max tool-call turns before forcing stop (safety)",
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
        default="eval_gemma4_results.jsonl",
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
    return p.parse_args()


def normalize_gemma_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = copy.deepcopy(raw_message)
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


def build_model_inputs(
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    *,
    max_input_tokens: int,
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
) -> tuple[dict[str, Any], bool]:
    rendered_conversation = normalize_gemma_messages(conversation)

    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        template_kwargs["tools"] = tools

    try:
        inputs = tokenizer.apply_chat_template(rendered_conversation, **template_kwargs)
    except TypeError:
        prompt_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools is not None:
            prompt_kwargs["tools"] = tools
        prompt = tokenizer.apply_chat_template(rendered_conversation, **prompt_kwargs)
        try:
            inputs = tokenizer(text=prompt, return_tensors="pt")
        except TypeError:
            inputs = tokenizer(prompt, return_tensors="pt")
    if isinstance(inputs, str):
        try:
            inputs = tokenizer(text=inputs, return_tensors="pt")
        except TypeError:
            inputs = tokenizer(inputs, return_tensors="pt")

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
    stop_tokens = [GEMMA_TOOL_CALL_CLOSE, GEMMA_TURN_CLOSE]
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
        tool_name, arguments = manual_parse_tool_call(body) or (None, None)
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
            verdict = jsonish_loads(content)
            if isinstance(verdict, dict):
                return {
                    "type": "verdict",
                    "content": verdict,
                    "thinking": parsed.get("thinking") if parsed else manual_thinking,
                    "notes": notes,
                    "raw": raw[:1000],
                }
        except Exception:
            notes.append("verdict_json_parse_failed")

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
    repair_wls_from_user: bool,
    enable_thinking: bool,
    verbose: bool,
) -> dict[str, Any]:
    import torch

    gt_verdict = normalize_verdict(extract_ground_truth(messages_gt))
    conversation = extract_prompt_prefix(messages_gt)
    user_snapshot = extract_user_snapshot(messages_gt)

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
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model_inputs, was_truncated = build_model_inputs(
            conversation,
            tokenizer,
            model,
            max_input_tokens=max_input_tokens,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        input_truncated = input_truncated or was_truncated

        turn_max_new_tokens = resolve_turn_max_new_tokens(turn, max_new_tokens)

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=turn_max_new_tokens,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=pad_token_id,
            )

        new_tokens = outputs[0][model_inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        last_raw_generation = response_text
        turn_record: dict[str, Any] = {
            "turn": turn + 1,
            "raw_generation": response_text,
            "prompt_truncated": was_truncated,
            "turn_max_new_tokens": turn_max_new_tokens,
        }

        if verbose:
            print(f"\n  ======== [Turn {turn+1}] Generated ({len(new_tokens)} tokens) ========")
            print(f"  {response_text}")
            if was_truncated:
                print("  [prompt was left-truncated to fit the context window]")
            print("  ======================================================\n")

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
                tool_args, repair_notes = repair_tool_arguments(tool_name, tool_args, user_snapshot)
                if repair_notes:
                    parse_notes.extend(repair_notes)
                    turn_record.setdefault("parse_notes", [])
                    turn_record["parse_notes"].extend(repair_notes)

            turn_record["tool_name"] = tool_name
            turn_record["tool_arguments"] = tool_args
            if not isinstance(tool_args, dict):
                error_msg = f"Parsed tool arguments are not a dict for {tool_name}: {type(tool_args).__name__}"
                turn_record["error"] = error_msg
                turn_trace.append(turn_record)
                break

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
                                "arguments": tool_args,
                            },
                        }
                    ],
                }
            )

            tool_result = execute_tool(tool_name, tool_args)
            tool_result_str = json.dumps(tool_result, default=str, ensure_ascii=False)
            turn_record["tool_result"] = tool_result
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": tool_result_str,
                }
            )

            if verbose:
                print(f"  -> Tool call: {tool_name}")
                preview = tool_result_str if len(tool_result_str) <= 240 else tool_result_str[:237] + "..."
                print(f"  <- Tool result: {preview}")

            del outputs
            if tool_name == "wls_from_path" and tool_result.get("success") is True:
                wls_completed_successfully = True

            if tool_result.get("success") is False and not continue_on_tool_error:
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
            turn_trace.append(turn_record)
            break

        excerpt = parsed.get("raw", response_text)
        error_msg = f"Unparseable output at turn {turn + 1}: {excerpt[:300]}"
        turn_record["error"] = error_msg
        turn_trace.append(turn_record)
        break

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
        "turn_trace": turn_trace,
        "final_conversation": conversation,
    }


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

    print(f"Loading adapter from {args.adapter} ...")
    try:
        from unsloth import FastModel

        unsloth_max_seq = args.max_seq_length if args.max_seq_length is not None else 4096
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=unsloth_max_seq,
            load_in_4bit=args.load_in_4bit,
            load_in_16bit=args.load_in_16bit,
            full_finetuning=False,
        )
        FastModel.for_inference(model)
        print("Model loaded via Unsloth.\n")
    except ImportError:
        print("Unsloth not available, falling back to transformers + peft ...")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        adapter_cfg_path = Path(args.adapter) / "adapter_config.json"
        with open(adapter_cfg_path, "r", encoding="utf-8") as handle:
            adapter_cfg = json.load(handle)
        base_model_name = adapter_cfg["base_model_name_or_path"]
        print(f"  Base model: {base_model_name}")

        quantization_config = None
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.load_in_16bit or not args.load_in_4bit else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()

        tokenizer_path = args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists() else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded via transformers + peft.\n")

    max_input_tokens = resolve_max_input_tokens(args, model, tokenizer)
    tools = load_tools(args)
    print(f"Using max input context length: {max_input_tokens}")
    print(f"Tool schemas passed to chat template: {'yes' if tools is not None else 'no'}")
    print(f"Gemma thinking enabled: {'yes' if args.enable_thinking else 'no'}\n")

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

    with open(args.output, "w", encoding="utf-8") as out_file:
        for idx, sample in enumerate(test_samples):
            messages = sample["messages"]
            print(f"[{idx + 1}/{len(test_samples)}] ", end="", flush=True)

            try:
                result = run_one_sample(
                    messages,
                    model,
                    tokenizer,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    tools=tools,
                    continue_on_tool_error=args.continue_on_tool_error,
                    repair_wls_from_user=args.repair_wls_from_user,
                    enable_thinking=args.enable_thinking,
                    verbose=args.verbose,
                )
            except Exception as exc:
                import traceback

                traceback.print_exc(file=sys.stdout)
                gt_verdict = normalize_verdict(extract_ground_truth(messages))
                result = {
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
                }

            results.append(result)
            out_file.write(json.dumps(result, default=str, ensure_ascii=False) + "\n")
            out_file.flush()

            gt_family = result["gt_error_family"] or "unknown"
            family_counts[gt_family] += 1

            if result["family_correct"]:
                family_correct += 1
                family_correct_counts[gt_family] += 1
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
                f"{status}  gt={gt_family:<20s}  pred={pred_fam_str:<20s}  "
                f"tools={result['tool_calls']}{extra}"
            )

    n = len(results)
    print("\n" + "=" * 60)
    print(f"{'EVALUATION SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total samples:          {n}")
    print(f"  Error detection acc:    {detection_correct}/{n}  ({100 * detection_correct / n:.1f}%)")
    print(f"  Error family acc:       {family_correct}/{n}  ({100 * family_correct / n:.1f}%)")
    print(f"  Parse/runtime errors:   {errors}")
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
