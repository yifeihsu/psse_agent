"""
Revised evaluator for the fine-tuned PSSE diagnostic agent.

Main fixes versus the original evaluator:
  * Uses the model chat template directly for tokenization, matching SFT formatting.
  * Stops generation only on <|call|> / <|return|> (not <|end|>), so multi-message
    Harmony completions like analysis -> tool call are not truncated mid-turn.
  * Parses Harmony-style completions more robustly, including analysis + tool call,
    stripped chat-template variants, and JSON-string-wrapped tool arguments.
  * Preserves assistant reasoning across tool turns via the `thinking` field, which is
    how the gpt-oss Hugging Face chat template expects analysis history to be passed.
  * Treats tool execution failures as runtime errors by default instead of letting the
    model continue as if the failed tool output were normal evidence.
  * Normalizes verdict fields before scoring (e.g. ["measurement_error"] -> "measurement_error").
  * Does not silently cap evaluation at 8K unless requested.

Usage examples:
  python eval_sft_agent_revised.py \
    --adapter outputs/gpt_oss_sft_power_agent_4k/lora \
    --test-file out_traces_balanced/sft_traces.test.jsonl \
    --max-seq-length 4096 \
    --output eval_4k_revised.jsonl

  python eval_sft_agent_revised.py \
    --adapter outputs/gpt_oss_sft_power_agent_16k/lora \
    --test-file out_traces_balanced/sft_traces.test.jsonl \
    --max-seq-length 16384 \
    --output eval_16k_revised.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 0. Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned PSSE agent")
    p.add_argument(
        "--adapter",
        default="outputs/gpt_oss_sft_power_agent_4k/lora",
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
        default=8,
        help="Max tool-call turns before forcing stop (safety)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens per generation step",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "Maximum input context length. If omitted, use the smallest sane value from "
            "the model/tokenizer config instead of hard-coding 8192."
        ),
    )
    p.add_argument(
        "--output",
        default="eval_results.jsonl",
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Import tools from the MCP server module (direct Python calls)
# ---------------------------------------------------------------------------
# We import the raw Python functions, stripping the FastMCP decorator overhead.
# Each function uses keyword-only arguments and returns a dict.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_server.matpower_server import (  # noqa: E402
    correct_measurements_from_path,
    correct_parameters_from_path,
    correct_topology_from_path,
    run_hse_from_path,
    wls_from_path,
)

TOOL_MAP = {
    "wls_from_path": wls_from_path,
    "correct_measurements_from_path": correct_measurements_from_path,
    "correct_parameters_from_path": correct_parameters_from_path,
    "correct_topology_from_path": correct_topology_from_path,
    "run_hse_from_path": run_hse_from_path,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one of the matpower tools and return its JSON-serialisable result."""
    tool_obj = TOOL_MAP.get(name)
    if tool_obj is None:
        return {"success": False, "error": f"Unknown tool: {name}"}
    try:
        fn = getattr(tool_obj, "fn", tool_obj)
        return fn(**arguments)
    except Exception as exc:  # pragma: no cover - defensive runtime wrapper
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 2. Dataset helpers
# ---------------------------------------------------------------------------

def strip_none_fields(message: dict[str, Any]) -> dict[str, Any]:
    """Mirror the cleaning used during training before apply_chat_template."""
    return {k: v for k, v in message.items() if v is not None}



def extract_prompt_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all non-model messages before the first assistant/tool action."""
    prefix: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role in {"assistant", "tool"}:
            break
        prefix.append(strip_none_fields(msg))
    return prefix



def extract_ground_truth(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the parsed JSON verdict from the last assistant content message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            try:
                obj = json.loads(msg["content"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict) and "verdict" in obj:
                return obj
    return None


# ---------------------------------------------------------------------------
# 3. Verdict normalization / scoring helpers
# ---------------------------------------------------------------------------

FAMILY_SYNONYMS = {
    "negative": "no_error",
    "none": None,
    "noerror": "no_error",
    "no_error": "no_error",
    "measurementerror": "measurement_error",
    "parametererror": "parameter_error",
    "topologyerror": "topology_error",
    "threephaseimbalance": "three_phase_imbalance",
    "harmonicanomaly": "harmonic_anomaly",
}



def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
    return None



def normalize_error_family(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            fam = normalize_error_family(item)
            if fam is not None:
                return fam
        return None
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    key = s.replace("_", "")
    return FAMILY_SYNONYMS.get(s, FAMILY_SYNONYMS.get(key, s))



def normalize_verdict(verdict_obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(verdict_obj, dict):
        return None
    verdict = verdict_obj.get("verdict")
    if not isinstance(verdict, dict):
        return verdict_obj

    norm = json.loads(json.dumps(verdict_obj))  # cheap deep copy for JSON-y content
    norm_verdict = norm.setdefault("verdict", {})
    norm_verdict["has_error"] = normalize_bool(norm_verdict.get("has_error"))
    norm_verdict["error_family"] = normalize_error_family(norm_verdict.get("error_family"))
    return norm


# ---------------------------------------------------------------------------
# 4. Harmony / chat-template parsing helpers
# ---------------------------------------------------------------------------

TERMINATORS = ("<|call|>", "<|return|>", "<|end|>")



def strip_trailing_special_tokens(text: str) -> str:
    s = text.strip()
    changed = True
    while changed:
        changed = False
        for tok in TERMINATORS + ("<|start|>assistant",):
            if s.endswith(tok):
                s = s[: -len(tok)].rstrip()
                changed = True
    return s



def _candidate_json_strings(payload: str) -> list[str]:
    payload = payload.strip()
    candidates: list[str] = [payload]
    stripped = strip_trailing_special_tokens(payload)
    if stripped != payload:
        candidates.append(stripped)

    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
        candidates.append(stripped[1:-1])
    if stripped.startswith('"') and stripped.endswith('}'):
        candidates.append(stripped[1:])
        candidates.append(stripped[1:] + '"')
    if stripped.startswith("'") and stripped.endswith("'") and len(stripped) >= 2:
        candidates.append(stripped[1:-1])

    # De-duplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out



def jsonish_loads(payload: str) -> Any:
    """Parse normal JSON plus a few common malformed/escaped variants seen in the logs."""
    last_error: Exception | None = None

    for candidate in _candidate_json_strings(payload):
        attempts = [candidate]

        unescaped = candidate.replace(r'\"', '"').replace(r'\\', '\\')
        if unescaped != candidate:
            attempts.append(unescaped)
            fixed_double_quotes = re.sub(r'""([^\"]+)"', r'"\1"', unescaped)
            if fixed_double_quotes != unescaped:
                attempts.append(fixed_double_quotes)

        for attempt in attempts:
            try:
                obj = json.loads(attempt)
                if isinstance(obj, str):
                    try:
                        obj = json.loads(obj)
                    except Exception:
                        pass
                return obj
            except Exception as exc:
                last_error = exc

            try:
                return ast.literal_eval(attempt)
            except Exception as exc:
                last_error = exc

    raise ValueError(f"Could not parse JSON-ish payload: {last_error}")



def infer_tool_name_from_arguments(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None

    if "name" in obj and obj.get("name") in TOOL_MAP and "arguments" in obj:
        return obj["name"]
    if "tool_name" in obj and obj.get("tool_name") in TOOL_MAP:
        return obj["tool_name"]

    keys = set(obj.keys())
    if {"case_path", "z"} <= keys and "suspect_group" not in keys:
        return "wls_from_path"
    if {"case_path", "z", "suspect_group"} <= keys:
        return "correct_measurements_from_path"
    if {"case_path", "line_index", "z_scans"} <= keys:
        return "correct_parameters_from_path"
    if {"case_path", "cb_name", "desired_status"} <= keys:
        return "correct_topology_from_path"
    if {"case_path", "harmonic_measurements"} <= keys:
        return "run_hse_from_path"
    return None



def normalize_tool_name(recipient: str | None, inferred_obj: Any = None) -> str | None:
    if recipient:
        candidate = recipient.strip()
        candidate = candidate.replace("functions.", "", 1)
        # Some malformed outputs glue "commentary" onto the tool name.
        for tool_name in TOOL_MAP:
            if candidate == tool_name:
                return tool_name
            if candidate.startswith(tool_name):
                return tool_name
            if tool_name in candidate:
                return tool_name

        candidate = re.sub(r"[^A-Za-z0-9_].*$", "", candidate)
        if candidate in TOOL_MAP:
            return candidate

    return infer_tool_name_from_arguments(inferred_obj)



def parse_harmony_messages(text: str) -> list[dict[str, Any]]:
    """
    Parse the decoded completion into Harmony-style assistant messages.

    The completion may contain multiple assistant messages in one generation step, e.g.
      analysis <|end|> assistant commentary tool_call <|call|>
    """
    s = text.strip()
    idx = 0
    messages: list[dict[str, Any]] = []

    while idx < len(s):
        while idx < len(s) and s[idx].isspace():
            idx += 1

        if s.startswith("<|start|>assistant", idx):
            idx += len("<|start|>assistant")
            continue

        msg_pos = s.find("<|message|>", idx)
        if msg_pos == -1:
            break

        header = s[idx:msg_pos]
        content_start = msg_pos + len("<|message|>")

        term_pos: int | None = None
        term_tok = ""
        for tok in ("<|call|>", "<|return|>", "<|end|>"):
            pos = s.find(tok, content_start)
            if pos != -1 and (term_pos is None or pos < term_pos):
                term_pos = pos
                term_tok = tok

        if term_pos is None:
            content = s[content_start:]
            idx = len(s)
        else:
            content = s[content_start:term_pos]
            idx = term_pos + len(term_tok)

        channel = None
        recipient = None
        constrain = None

        m = re.search(r"<\|channel\|>\s*([A-Za-z_]+)", header)
        if m:
            channel = m.group(1)

        m = re.search(r"to=([^\s<]+)", header)
        if m:
            recipient = m.group(1)

        m = re.search(r"<\|constrain\|>\s*([^\s<]+)", header)
        if m:
            constrain = m.group(1)
        else:
            m = re.search(r"\b(json)\s*$", header)
            if m:
                constrain = m.group(1)

        messages.append(
            {
                "header": header,
                "channel": channel,
                "recipient": recipient,
                "constrain": constrain,
                "content": content,
                "terminator": term_tok,
            }
        )

    return messages



def parse_generation(text: str) -> dict[str, Any]:
    """
    Parse a raw model completion into either:
      - tool_call
      - verdict
      - unparseable

    Returns additional metadata:
      - thinking: concatenated analysis text before the decisive action
      - notes: parser notes / fallbacks used
    """
    raw = text.strip()
    notes: list[str] = []
    messages = parse_harmony_messages(raw)
    analysis_chunks: list[str] = []

    if messages:
        for msg in messages:
            channel = msg.get("channel")
            content = str(msg.get("content", ""))
            terminator = msg.get("terminator")
            recipient = msg.get("recipient")

            if channel == "analysis":
                if content.strip():
                    analysis_chunks.append(content.strip())
                continue

            # Harmony tool call path.
            if terminator == "<|call|>" or recipient:
                parsed_payload: Any = None
                try:
                    parsed_payload = jsonish_loads(content)
                except Exception:
                    parsed_payload = None

                tool_name = normalize_tool_name(recipient, parsed_payload)
                if tool_name:
                    if recipient and not (recipient.startswith("functions.") or tool_name in recipient):
                        notes.append(f"non_canonical_recipient:{recipient}")
                    if recipient and normalize_tool_name(recipient, None) is None:
                        notes.append("tool_inferred_from_payload_signature")

                    args = parsed_payload
                    if isinstance(parsed_payload, dict) and "name" in parsed_payload and "arguments" in parsed_payload:
                        args = parsed_payload["arguments"]
                        notes.append("openai_style_tool_object")
                    elif isinstance(parsed_payload, dict) and "tool_name" in parsed_payload and "arguments" in parsed_payload:
                        args = parsed_payload["arguments"]
                        notes.append("tool_name_arguments_object")

                    return {
                        "type": "tool_call",
                        "name": tool_name,
                        "arguments": args,
                        "thinking": "\n\n".join(analysis_chunks).strip() or None,
                        "notes": notes,
                        "messages": messages,
                    }

            # Harmony final message path.
            try:
                obj = jsonish_loads(content)
            except Exception:
                obj = None
            if isinstance(obj, dict) and "verdict" in obj:
                return {
                    "type": "verdict",
                    "content": obj,
                    "thinking": "\n\n".join(analysis_chunks).strip() or None,
                    "notes": notes,
                    "messages": messages,
                }

    # Fallback 1: stripped chat-template tool form without <|message|>, e.g.
    #   to=functions.wls_from_pathcommentary json{"case_path": ...}
    m = re.search(
        r"to=(?P<recipient>[^\s]+).*?(?:json|function)\s*(?P<payload>(?:\{.*\}|\".*))$",
        raw,
        re.DOTALL,
    )
    if m:
        recipient = m.group("recipient")
        payload = m.group("payload")
        try:
            obj = jsonish_loads(payload)
            tool_name = normalize_tool_name(recipient, obj)
            if tool_name:
                notes.append("stripped_template_tool_fallback")
                args = obj
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                elif isinstance(obj, dict) and "tool_name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                return {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": args,
                    "thinking": None,
                    "notes": notes,
                    "messages": messages,
                }
        except Exception:
            pass

    # Fallback 2: plain or embedded JSON verdict / tool args.
    json_blocks = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
    for block in reversed(json_blocks):
        try:
            obj = jsonish_loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            notes.append("embedded_json_verdict_fallback")
            return {
                "type": "verdict",
                "content": obj,
                "thinking": None,
                "notes": notes,
                "messages": messages,
            }
        tool_name = normalize_tool_name(None, obj)
        if tool_name:
            notes.append("embedded_json_tool_fallback")
            args = obj
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                args = obj["arguments"]
            elif isinstance(obj, dict) and "tool_name" in obj and "arguments" in obj:
                args = obj["arguments"]
            return {
                "type": "tool_call",
                "name": tool_name,
                "arguments": args,
                "thinking": None,
                "notes": notes,
                "messages": messages,
            }

    return {
        "type": "unparseable",
        "raw": raw[:1000],
        "thinking": None,
        "notes": notes,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# 5. Generation helpers
# ---------------------------------------------------------------------------


def resolve_max_input_tokens(args: argparse.Namespace, model: Any, tokenizer: Any) -> int:
    """Pick the smallest sane context limit unless the user requested one explicitly."""
    if args.max_seq_length is not None:
        return args.max_seq_length

    candidates: list[int] = []
    for value in [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]:
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)

    return min(candidates) if candidates else 16384



def build_model_inputs(
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    *,
    max_input_tokens: int,
) -> tuple[dict[str, Any], bool]:
    """Render the chat with the tokenizer chat template and left-truncate if needed."""
    try:
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        # Older tokenizer versions may not support return_dict on apply_chat_template.
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")

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



def get_stop_token_ids(tokenizer: Any) -> list[int]:
    stop_tokens = ["<|call|>", "<|return|>"]
    stop_ids: list[int] = []

    unk_id = getattr(tokenizer, "unk_token_id", None)
    for tok in stop_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id != unk_id:
            stop_ids.append(tok_id)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        stop_ids.extend(eos_id)
    elif eos_id is not None:
        stop_ids.append(eos_id)

    # De-duplicate while preserving order.
    out: list[int] = []
    seen: set[int] = set()
    for item in stop_ids:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


# ---------------------------------------------------------------------------
# 6. Run one sample through the agent loop
# ---------------------------------------------------------------------------

def run_one_sample(
    messages_gt: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    max_turns: int,
    max_new_tokens: int,
    max_input_tokens: int,
    continue_on_tool_error: bool,
    verbose: bool,
) -> dict[str, Any]:
    import torch

    gt_verdict = normalize_verdict(extract_ground_truth(messages_gt))
    conversation = extract_prompt_prefix(messages_gt)

    tool_calls_made: list[str] = []
    parse_notes: list[str] = []
    predicted_verdict: dict[str, Any] | None = None
    error_msg: str | None = None
    input_truncated = False
    last_raw_generation: str | None = None

    stop_ids = get_stop_token_ids(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, list):
            pad_token_id = eos_id[0]
        else:
            pad_token_id = eos_id

    for turn in range(max_turns):
        model_inputs, was_truncated = build_model_inputs(
            conversation,
            tokenizer,
            model,
            max_input_tokens=max_input_tokens,
        )
        input_truncated = input_truncated or was_truncated

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=pad_token_id,
            )

        new_tokens = outputs[0][model_inputs["input_ids"].shape[-1] :]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        last_raw_generation = response_text

        if verbose:
            print(f"\n  ======== [Turn {turn+1}] Generated ({len(new_tokens)} tokens) ========")
            print(f"  {response_text}")
            if was_truncated:
                print("  [prompt was left-truncated to fit the context window]")
            print("  ======================================================\n")

        parsed = parse_generation(response_text)
        parse_notes.extend(parsed.get("notes", []))

        if parsed["type"] == "tool_call":
            tool_name = parsed["name"]
            tool_args = parsed["arguments"]
            if not isinstance(tool_args, dict):
                error_msg = f"Parsed tool arguments are not a dict for {tool_name}: {type(tool_args).__name__}"
                break

            tool_calls_made.append(tool_name)
            call_id = f"call_{tool_name}_{uuid.uuid4().hex[:8]}"

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call_id,
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }
                ],
            }
            if parsed.get("thinking"):
                assistant_message["thinking"] = parsed["thinking"]
            conversation.append(assistant_message)

            if verbose:
                print(f"  -> Tool call: {tool_name}")
                if parsed.get("thinking"):
                    print(f"     thinking: {parsed['thinking'][:200]}")

            tool_result = execute_tool(tool_name, tool_args)
            tool_result_str = json.dumps(tool_result, default=str, ensure_ascii=False)
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": tool_result_str,
                }
            )

            if verbose:
                preview = tool_result_str if len(tool_result_str) <= 240 else tool_result_str[:237] + "..."
                print(f"  <- Tool result: {preview}")

            if tool_result.get("success") is False and not continue_on_tool_error:
                error_msg = f"Tool {tool_name} failed: {tool_result.get('error', 'unknown error')}"
                break

        elif parsed["type"] == "verdict":
            predicted_verdict = normalize_verdict(parsed["content"])
            conversation.append(
                {
                    "role": "assistant",
                    "content": json.dumps(parsed["content"], ensure_ascii=False),
                }
            )
            break

        else:
            excerpt = parsed.get("raw", response_text)
            error_msg = f"Unparseable output at turn {turn+1}: {excerpt[:300]}"
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
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not Path(args.adapter).exists() and not "/" in args.adapter:
        print(f"ERROR: Adapter not found at {args.adapter} (and not a HuggingFace Hub ID)")
        sys.exit(1)
    if not Path(args.test_file).exists():
        print(f"ERROR: Test file not found at {args.test_file}")
        sys.exit(1)

    # ---- Load model ----
    print(f"Loading adapter from {args.adapter} ...")
    try:
        from unsloth import FastLanguageModel

        unsloth_max_seq = args.max_seq_length if args.max_seq_length is not None else 16384
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=unsloth_max_seq,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("Model loaded via Unsloth.\n")
    except ImportError:
        print("Unsloth not available, falling back to transformers + peft ...")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        adapter_cfg_path = Path(args.adapter) / "adapter_config.json"
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
        print(f"  Base model: {base_model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded via transformers + peft.\n")

    max_input_tokens = resolve_max_input_tokens(args, model, tokenizer)
    print(f"Using max input context length: {max_input_tokens}\n")

    # ---- Load test data ----
    test_samples: list[dict[str, Any]] = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    if args.max_samples:
        test_samples = test_samples[: args.max_samples]
    print(f"Loaded {len(test_samples)} test samples.\n")

    # ---- Run evaluation ----
    results: list[dict[str, Any]] = []
    family_correct = 0
    detection_correct = 0
    errors = 0
    family_counts: Counter[str] = Counter()
    family_correct_counts: Counter[str] = Counter()

    with open(args.output, "w", encoding="utf-8") as out_file:
        for idx, sample in enumerate(test_samples):
            messages = sample["messages"]
            print(f"[{idx+1}/{len(test_samples)}] ", end="", flush=True)

            try:
                result = run_one_sample(
                    messages,
                    model,
                    tokenizer,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    continue_on_tool_error=args.continue_on_tool_error,
                    verbose=args.verbose,
                )
            except Exception as exc:  # pragma: no cover - top-level eval guard
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

            gt_fam = result["gt_error_family"] or "unknown"
            family_counts[gt_fam] += 1

            if result["family_correct"]:
                family_correct += 1
                family_correct_counts[gt_fam] += 1
            if result["detection_correct"]:
                detection_correct += 1
            if result["error"]:
                errors += 1

            status = "✓" if result["family_correct"] else "✗"
            pred_fam_str = str(result["pred_error_family"]) if result["pred_error_family"] else "NONE"
            gt_fam_str = str(gt_fam)
            extra = ""
            if result.get("input_truncated"):
                extra += " truncated"
            if result.get("parse_notes"):
                extra += f" notes={result['parse_notes'][:3]}"

            print(
                f"{status}  gt={gt_fam_str:<20s}  pred={pred_fam_str:<20s}  "
                f"tools={result['tool_calls']}{extra}"
            )

    # ---- Print summary ----
    n = len(results)
    print("\n" + "=" * 60)
    print(f"{'EVALUATION SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total samples:          {n}")
    print(f"  Error detection acc:    {detection_correct}/{n}  ({100*detection_correct/n:.1f}%)")
    print(f"  Error family acc:       {family_correct}/{n}  ({100*family_correct/n:.1f}%)")
    print(f"  Parse/runtime errors:   {errors}")
    print()
    print("  Per-family breakdown:")
    for fam in sorted(family_counts.keys()):
        total = family_counts[fam]
        correct = family_correct_counts[fam]
        print(f"    {fam:<25s}  {correct}/{total}  ({100*correct/total:.1f}%)")
    print("=" * 60)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
