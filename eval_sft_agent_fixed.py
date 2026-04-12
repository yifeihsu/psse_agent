"""
Fixed evaluator for the fine-tuned PSSE diagnostic agent.

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
  * Mirrors SFT prompt rendering by passing tool schemas and normalizing tool-call messages before apply_chat_template.

Usage examples:
  python eval_sft_agent_fixed.py \
    --adapter outputs/gpt_oss_sft_power_agent_4k/lora \
    --test-file out_traces_balanced/sft_traces.test.jsonl \
    --max-seq-length 4096 \
    --output eval_4k_revised.jsonl

  python eval_sft_agent_fixed.py \
    --adapter outputs/gpt_oss_sft_power_agent_16k/lora \
    --test-file out_traces_balanced/sft_traces.test.jsonl \
    --max-seq-length 16384 \
    --output eval_16k_revised.jsonl
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from trace_protocol import (
    canonical_tool_schemas,
    CONTEXT_TOOL_NAMES,
    extract_conversation_context,
    hydrate_tool_arguments as protocol_hydrate_tool_arguments,
    looks_like_json,
    resolve_case_path_alias,
    summarize_tool_result_for_conversation,
)

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
        default=8192,
        help="Maximum input context length to render into the model",
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
        help="Hydrate omitted tool arguments from the visible user payloads before tool execution",
    )
    p.add_argument(
        "--no-repair-wls-from-user",
        dest="repair_wls_from_user",
        action="store_false",
        help="Disable hydration of omitted tool arguments from user follow-up payloads",
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
PARSEABLE_TOOL_NAMES = set(TOOL_MAP) | set(CONTEXT_TOOL_NAMES)


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
                    "z": {"type": "array", "items": {"type": "number"}, "description": "Observed measurement vector."},
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
                    "z": {"type": "array", "items": {"type": "number"}, "description": "Observed measurement vector."},
                    "suspect_group": {"type": "array", "items": {"type": "integer"}, "description": "Suspected measurement indices."},
                    "enable_correction": {"type": "boolean", "description": "Whether to apply the correction."},
                    "max_correction_iterations": {"type": "integer", "description": "Maximum correction iterations."},
                    "error_tolerance": {"type": "number", "description": "Correction stopping tolerance."},
                },
                "required": ["case_path", "z", "suspect_group", "enable_correction", "max_correction_iterations", "error_tolerance"],
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
                    "line_index": {"type": "integer", "description": "Suspected line index."},
                    "z_scans": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Repeated measurement scans."},
                },
                "required": ["case_path", "line_index", "z_scans"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_topology_from_path",
            "description": "Correct a suspected topology mismatch by switching a breaker/circuit-breaker status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "cb_name": {"type": "string", "description": "Breaker name."},
                    "desired_status": {"type": "boolean", "description": "Desired breaker status."},
                },
                "required": ["case_path", "cb_name", "desired_status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_hse_from_path",
            "description": "Run Harmonic State Estimation (HSE) to identify a harmonic source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "harmonic_measurements": {"type": "array", "items": {"type": "object"}, "description": "Harmonic measurements for HSE."},
                    "harmonic_orders": {"type": "array", "items": {"type": "integer"}, "description": "Optional harmonic orders."},
                    "slack_bus": {"type": "integer", "description": "Optional slack bus index."},
                },
                "required": ["case_path", "harmonic_measurements"],
            },
        },
    },
]


def sanitize_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue
        fixed = json.loads(json.dumps(tool))
        function_info = fixed.get("function")
        if isinstance(function_info, dict):
            function_info.setdefault("description", f"Call the {function_info.get('name', 'tool')} tool.")
            params = function_info.get("parameters")
            if isinstance(params, dict):
                params.setdefault("type", "object")
                params.setdefault("properties", {})
                for name, spec in params.get("properties", {}).items():
                    if isinstance(spec, dict):
                        spec.setdefault("description", f"Argument {name}.")
            fixed["function"] = function_info
        sanitized.append(fixed)
    return sanitized


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


def prune_none(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not None}


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = prune_none(dict(raw_message))
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
                    tc = prune_none(dict(tool_call))
                    function_info = tc.get("function")
                    if isinstance(function_info, dict):
                        function_info = dict(function_info)
                        function_info["arguments"] = maybe_parse_json_string(function_info.get("arguments"))
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


def extract_user_snapshot(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        obj: Any = content
        if isinstance(content, str):
            try:
                obj = json.loads(content)
            except Exception:
                continue
        if isinstance(obj, dict) and ("z_obs" in obj or "case_path" in obj):
            return obj
    return None


def repair_tool_arguments(
    tool_name: str,
    arguments: Any,
    conversation: list[dict[str, Any]],
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
    return {"success": False, "error": f"Missing runtime context for {name}"}


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    runtime_context: Mapping[str, Any] | None = None,
    hidden_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call one of the matpower tools and return its JSON-serialisable result."""
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


# ---------------------------------------------------------------------------
# 2. Dataset helpers
# ---------------------------------------------------------------------------

def strip_none_fields(message: dict[str, Any]) -> dict[str, Any]:
    """Mirror the cleaning used during training before apply_chat_template."""
    return prune_none(message)



def extract_prompt_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all non-model messages before the first assistant/tool action."""
    prefix: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role in {"assistant", "tool"}:
            break
        prefix.append(strip_none_fields(msg))
    return prefix


def extract_reference_user_followups(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_messages = [strip_none_fields(dict(msg)) for msg in messages if msg.get("role") == "user"]
    return user_messages[1:]



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

    if "name" in obj and obj.get("name") in PARSEABLE_TOOL_NAMES and "arguments" in obj:
        return obj["name"]
    if "function" in obj and obj.get("function") in PARSEABLE_TOOL_NAMES and "arguments" in obj:
        return obj["function"]
    if "tool_name" in obj and obj.get("tool_name") in PARSEABLE_TOOL_NAMES:
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
        for tool_name in PARSEABLE_TOOL_NAMES:
            if candidate == tool_name:
                return tool_name
            if candidate.startswith(tool_name):
                return tool_name
            if tool_name in candidate:
                return tool_name

        candidate = re.sub(r"[^A-Za-z0-9_].*$", "", candidate)
        if candidate in PARSEABLE_TOOL_NAMES:
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
                    elif isinstance(parsed_payload, dict) and "function" in parsed_payload and "arguments" in parsed_payload:
                        args = parsed_payload["arguments"]
                        notes.append("function_arguments_object")
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
                elif isinstance(obj, dict) and "function" in obj and "arguments" in obj:
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
            elif isinstance(obj, dict) and "function" in obj and "arguments" in obj:
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

    plain_text = strip_trailing_special_tokens(raw).strip()
    if messages:
        for msg in reversed(messages):
            content = strip_trailing_special_tokens(str(msg.get("content", ""))).strip()
            if content:
                return {
                    "type": "assistant_message",
                    "content": content,
                    "thinking": "\n\n".join(analysis_chunks).strip() or None,
                    "notes": notes,
                    "messages": messages,
                }
    if plain_text and not looks_like_json(plain_text):
        return {
            "type": "assistant_message",
            "content": plain_text,
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
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], bool]:
    """Render the chat with the tokenizer chat template and left-truncate if needed."""
    rendered_conversation = normalize_messages(conversation)
    try:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if tools is not None:
            kwargs["tools"] = tools
        inputs = tokenizer.apply_chat_template(rendered_conversation, **kwargs)
    except TypeError:
        prompt_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools is not None:
            prompt_kwargs["tools"] = tools
        prompt = tokenizer.apply_chat_template(rendered_conversation, **prompt_kwargs)
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


def classify_result_error(error_msg: str | None) -> str | None:
    if not error_msg:
        return None

    s = error_msg.lower()
    if s.startswith("verdict before required wls_from_path"):
        return "missing_required_wls"
    if s.startswith("unparseable output"):
        return "unparseable_output"
    if s.startswith("tool "):
        if "no module named 'scipy'" in s or 'no module named "scipy"' in s:
            return "tool_missing_scipy"
        if "valueerror: wls input error" in s:
            return "tool_bad_wls_input"
        return "tool_failure"
    if s.startswith("critical error"):
        return "critical_error"
    return "other_error"


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
    tools: list[dict[str, Any]] | None,
    continue_on_tool_error: bool,
    repair_wls_from_user: bool,
    runtime_context: Mapping[str, Any] | None,
    verbose: bool,
) -> dict[str, Any]:
    import torch

    gt_verdict = normalize_verdict(extract_ground_truth(messages_gt))
    conversation = extract_prompt_prefix(messages_gt)
    reference_user_followups = extract_reference_user_followups(messages_gt)
    replayed_user_followups = 0
    hidden_context: dict[str, Any] = {
        "case_aliases": dict(((runtime_context or {}).get("case_aliases") or {})),
    }

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
        turn_record: dict[str, Any] = {
            "turn": turn + 1,
            "raw_generation": response_text,
            "prompt_truncated": was_truncated,
        }

        if verbose:
            print(f"\n  ======== [Turn {turn+1}] Generated ({len(new_tokens)} tokens) ========")
            print(f"  {response_text}")
            if was_truncated:
                print("  [prompt was left-truncated to fit the context window]")
            print("  ======================================================\n")

        parsed = parse_generation(response_text)
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
            turn_record["tool_name"] = tool_name
            turn_record["tool_arguments"] = tool_args
            if not isinstance(tool_args, dict):
                error_msg = f"Parsed tool arguments are not a dict for {tool_name}: {type(tool_args).__name__}"
                turn_record["error"] = error_msg
                turn_trace.append(turn_record)
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
                            "arguments": tool_args,
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

            tool_result = execute_tool(
                tool_name,
                tool_args,
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
            turn_record["tool_result"] = tool_result_compact if tool_name in CONTEXT_TOOL_NAMES else tool_result
            tool_result_str = json.dumps(tool_result_compact, default=str, ensure_ascii=False)
            turn_record["tool_result_compact"] = tool_result_compact
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

            del outputs
            if tool_name == "wls_from_path" and tool_result.get("success") is True:
                wls_completed_successfully = True

            if tool_result.get("success") is False and not continue_on_tool_error:
                error_msg = f"Tool {tool_name} failed: {tool_result.get('error', 'unknown error')}"
                turn_record["error"] = error_msg
                turn_trace.append(turn_record)
                break

            turn_trace.append(turn_record)

        elif parsed["type"] == "assistant_message":
            assistant_content = parsed["content"]
            turn_record["assistant_message"] = assistant_content
            conversation.append({"role": "assistant", "content": assistant_content})
            if replayed_user_followups >= len(reference_user_followups):
                error_msg = f"Assistant requested follow-up data at turn {turn+1}, but no reference user payload remained"
                turn_record["error"] = error_msg
                turn_trace.append(turn_record)
                break
            followup_message = reference_user_followups[replayed_user_followups]
            replayed_user_followups += 1
            conversation.append(followup_message)
            turn_record["replayed_user_followup"] = followup_message
            turn_trace.append(turn_record)
            continue

        elif parsed["type"] == "verdict":
            if not wls_completed_successfully:
                error_msg = f"Verdict before required wls_from_path at turn {turn+1}"
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

        else:
            excerpt = parsed.get("raw", response_text)
            error_msg = f"Unparseable output at turn {turn+1}: {excerpt[:300]}"
            turn_record["error"] = error_msg
            turn_trace.append(turn_record)
            break

    if predicted_verdict is None and error_msg is None:
        error_msg = f"Stopped after max_turns={max_turns} without a final verdict"

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


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

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
        except Exception as e:
            print(f"ERROR: Failed to download adapter: {e}")
            sys.exit(1)
    elif not Path(args.adapter).exists():
        print(f"ERROR: Adapter not found locally at {args.adapter}")
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
    tools = load_tools(args)
    print(f"Using max input context length: {max_input_tokens}")
    print(f"Tool schemas passed to chat template: {'yes' if tools is not None else 'no'}\n")

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
    error_kind_counts: Counter[str] = Counter()
    parse_note_counts: Counter[str] = Counter()

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
                    tools=tools,
                    continue_on_tool_error=args.continue_on_tool_error,
                    repair_wls_from_user=args.repair_wls_from_user,
                    runtime_context=sample.get("runtime_context"),
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
                error_kind = classify_result_error(result["error"])
                if error_kind is not None:
                    error_kind_counts[error_kind] += 1
            for note in result.get("parse_notes") or []:
                parse_note_counts[note] += 1

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
    if error_kind_counts:
        print("  Error breakdown:")
        for kind, count in error_kind_counts.most_common():
            print(f"    {kind:<25s}  {count}")
        print()
    print("  Per-family breakdown:")
    for fam in sorted(family_counts.keys()):
        total = family_counts[fam]
        correct = family_correct_counts[fam]
        print(f"    {fam:<25s}  {correct}/{total}  ({100*correct/total:.1f}%)")
    if parse_note_counts:
        print()
        print("  Top parser notes:")
        for note, count in parse_note_counts.most_common(8):
            print(f"    {note:<35s}  {count}")
    print("=" * 60)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
