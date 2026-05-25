"""
Hardened evaluator for the fine-tuned PSSE diagnostic agent.

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
  * Repairs near-miss JSON by extracting the first balanced JSON fragment and closing small brace/bracket mismatches.
  * Uses a smaller default token budget on post-tool turns to reduce runaway verbose finals.

Usage examples:
  python eval_sft_agent_hardened.py \
    --adapter outputs/gemma4_power_agent/lora \
    --test-file out_traces_balanced/sft_traces.test.jsonl \
    --max-seq-length 4096 \
    --output outputs/gemma4_power_agent/eval_4k_hardened.jsonl

  python eval_sft_agent_hardened.py \
    --adapter outputs/gemma4_power_agent/lora \
    --test-file out_traces_balanced/sft_traces.test.jsonl \
    --max-seq-length 16384 \
    --output outputs/gemma4_power_agent/eval_16k_hardened.jsonl
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
        default=4,
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
        default="outputs/gemma4_power_agent/eval_results.jsonl",
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
    return sanitize_tool_schemas(DEFAULT_POWER_TOOLS)


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


def repair_tool_arguments(tool_name: str, arguments: Any, user_snapshot: dict[str, Any] | None) -> tuple[Any, list[str]]:
    notes: list[str] = []
    if not isinstance(arguments, dict) or not isinstance(user_snapshot, dict):
        return arguments, notes

    repaired = json.loads(json.dumps(arguments))
    user_case = user_snapshot.get("case_path")
    user_z = user_snapshot.get("z_obs")

    if user_case and not repaired.get("case_path"):
        repaired["case_path"] = user_case
        notes.append("filled_case_path_from_user")

    if tool_name in {"wls_from_path", "correct_measurements_from_path"} and isinstance(user_z, list):
        z_val = repaired.get("z")
        if not isinstance(z_val, list):
            repaired["z"] = user_z
            notes.append(f"filled_{tool_name}_z_from_user")
        elif len(z_val) != len(user_z):
            repaired["z"] = user_z
            notes.append(f"replaced_{tool_name}_z_len_{len(z_val)}_with_user_len_{len(user_z)}")

    return repaired, notes


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
REQUIRED_TOOL_BY_FAMILY = {
    "measurement_error": "correct_measurements_from_path",
    "parameter_error": "correct_parameters_from_path",
    "topology_error": "correct_topology_from_path",
    "harmonic_anomaly": "run_hse_from_path",
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


def normalize_error_families(value: Any) -> list[str]:
    raw_values = value if isinstance(value, list) else [value]
    families: list[str] = []
    for item in raw_values:
        family = normalize_error_family(item)
        if family is not None and family != "no_error" and family not in families:
            families.append(family)
    return families


def verdict_error_families(verdict_obj: dict[str, Any] | None) -> list[str]:
    if not isinstance(verdict_obj, dict):
        return []
    verdict = verdict_obj.get("verdict")
    if not isinstance(verdict, dict):
        return []
    families = normalize_error_families(verdict.get("error_families"))
    if not families:
        families = normalize_error_families(verdict.get("error_family"))
    return families


def applied_tools_from_verdict(verdict_obj: dict[str, Any] | None) -> list[str]:
    if not isinstance(verdict_obj, dict):
        return []
    action = verdict_obj.get("action")
    if not isinstance(action, dict):
        return []
    raw = action.get("applied_tools")
    tools: list[str] = []
    if isinstance(raw, list):
        tools.extend(str(item) for item in raw if item)
    elif action.get("applied_tool"):
        tools.append(str(action["applied_tool"]))
    return tools


def required_tools_for_families(families: list[str]) -> list[str]:
    return [tool for family in families if (tool := REQUIRED_TOOL_BY_FAMILY.get(family))]


def multi_metric_fields(
    gt_verdict: dict[str, Any] | None,
    predicted_verdict: dict[str, Any] | None,
    tool_calls_made: list[str],
) -> dict[str, Any]:
    gt_families = verdict_error_families(gt_verdict)
    pred_families = verdict_error_families(predicted_verdict)
    required_tools = required_tools_for_families(gt_families)
    required_called = [tool for tool in required_tools if tool in tool_calls_made]
    required_missing = [tool for tool in required_tools if tool not in tool_calls_made]
    coverage = (len(required_called) / len(required_tools)) if required_tools else None
    return {
        "gt_error_families": gt_families,
        "pred_error_families": pred_families,
        "family_set_correct": predicted_verdict is not None and set(gt_families) == set(pred_families),
        "required_tools": required_tools,
        "required_tools_called": required_called,
        "required_tools_missing": required_missing,
        "required_tool_coverage": coverage,
        "predicted_applied_tools": applied_tools_from_verdict(predicted_verdict),
    }



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
    if "error_families" in norm_verdict:
        norm_verdict["error_families"] = normalize_error_families(norm_verdict.get("error_families"))
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




def extract_first_balanced_json_fragment(text: str) -> str | None:
    """
    Return the first balanced top-level JSON object/array found in text.

    This salvages generations like:
      {"case_path": ...} //commentary to=functions...
    and ignores the trailing junk after the first balanced JSON payload.
    """
    s = text.strip()
    start = None
    opener = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None:
        return None

    stack: list[str] = [opener]
    in_string = False
    escape = False
    closing = {"{": "}", "[": "]"}
    opening = {v: k for k, v in closing.items()}

    for i in range(start + 1, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch in closing:
            stack.append(ch)
            continue
        if ch in opening:
            if not stack:
                break
            top = stack[-1]
            if opening[ch] != top:
                break
            stack.pop()
            if not stack:
                return s[start : i + 1]

    return None


def build_unbalanced_json_candidates(text: str) -> list[str]:
    """
    Build a few repaired candidates for nearly-valid JSON strings.

    Typical evaluator failures here are exactly one missing '}' at the end of a tool call
    or final verdict, so we repair small bracket-count mismatches conservatively.
    """
    s = strip_trailing_special_tokens(text).strip()
    if not s:
        return []

    candidates: list[str] = []

    # Candidate 1: strip trailing junk after the first balanced JSON object/array.
    frag = extract_first_balanced_json_fragment(s)
    if frag and frag != s:
        candidates.append(frag)

    # Candidate 2+: append small numbers of missing closing brackets/braces.
    brace_delta = s.count("{") - s.count("}")
    bracket_delta = s.count("[") - s.count("]")
    if 0 < brace_delta <= 4 and 0 <= bracket_delta <= 8:
        candidates.append(s + ("]" * max(0, bracket_delta)) + ("}" * max(0, brace_delta)))
    if 0 <= brace_delta <= 4 and 0 < bracket_delta <= 8:
        candidates.append(s + ("]" * max(0, bracket_delta)) + ("}" * max(0, brace_delta)))

    # If we found a balanced prefix, also try balancing that prefix.
    if frag:
        brace_delta = frag.count("{") - frag.count("}")
        bracket_delta = frag.count("[") - frag.count("]")
        if 0 <= brace_delta <= 4 and 0 <= bracket_delta <= 8:
            balanced_frag = frag + ("]" * max(0, bracket_delta)) + ("}" * max(0, brace_delta))
            if balanced_frag != frag:
                candidates.append(balanced_frag)

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

    raw_candidates = _candidate_json_strings(payload)
    repair_candidates: list[str] = []
    for candidate in raw_candidates:
        repair_candidates.extend(build_unbalanced_json_candidates(candidate))

    # Preserve order while trying raw candidates first, then repaired candidates.
    all_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates + repair_candidates:
        if candidate not in seen:
            all_candidates.append(candidate)
            seen.add(candidate)

    for candidate in all_candidates:
        attempts = [candidate]

        unescaped = candidate.replace(r'\"', '"').replace(r'\\', '\\')
        if unescaped != candidate:
            attempts.append(unescaped)
            fixed_double_quotes = re.sub(r'""([^\"]+)"', r'"\1"', unescaped)
            if fixed_double_quotes != unescaped:
                attempts.append(fixed_double_quotes)

        # If the candidate still contains trailing junk after a balanced JSON object,
        # try the first balanced fragment as a last-mile salvage.
        frag = extract_first_balanced_json_fragment(candidate)
        if frag and frag not in attempts:
            attempts.append(frag)

        seen_attempts: set[str] = set()
        for attempt in attempts:
            if attempt in seen_attempts:
                continue
            seen_attempts.add(attempt)

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
    if "function" in obj and obj.get("function") in TOOL_MAP and "arguments" in obj:
        return obj["function"]
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




def resolve_turn_max_new_tokens(turn_index0: int, default_max_new_tokens: int) -> int:
    """
    Keep the default budget on turn 1, but use a smaller default on the final-verdict turn
    unless the caller already chose a smaller budget.

    This reduces runaway verbose finals without changing the CLI surface.
    """
    if turn_index0 == 0:
        return default_max_new_tokens
    return min(default_max_new_tokens, 768)

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

        new_tokens = outputs[0][model_inputs["input_ids"].shape[-1] :]
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

    gt_family = gt_verdict.get("verdict", {}).get("error_family") if gt_verdict else None
    pred_family = predicted_verdict.get("verdict", {}).get("error_family") if predicted_verdict else None

    gt_has_error = gt_verdict.get("verdict", {}).get("has_error") if gt_verdict else None
    pred_has_error = predicted_verdict.get("verdict", {}).get("has_error") if predicted_verdict else None
    multi_metrics = multi_metric_fields(gt_verdict, predicted_verdict, tool_calls_made)

    return {
        "gt_error_family": gt_family,
        "pred_error_family": pred_family,
        "gt_has_error": gt_has_error,
        "pred_has_error": pred_has_error,
        "family_correct": gt_family == pred_family,
        "detection_correct": gt_has_error == pred_has_error,
        **multi_metrics,
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
    family_set_correct = 0
    required_tool_coverages: list[float] = []
    multi_tp: Counter[str] = Counter()
    multi_fp: Counter[str] = Counter()
    multi_fn: Counter[str] = Counter()
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
                    **multi_metric_fields(gt_verdict, None, []),
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
            if result.get("family_set_correct"):
                family_set_correct += 1
            coverage = result.get("required_tool_coverage")
            if coverage is not None:
                required_tool_coverages.append(float(coverage))
            gt_family_set = set(result.get("gt_error_families") or [])
            pred_family_set = set(result.get("pred_error_families") or [])
            for fam in gt_family_set | pred_family_set:
                if fam in gt_family_set and fam in pred_family_set:
                    multi_tp[fam] += 1
                elif fam in pred_family_set:
                    multi_fp[fam] += 1
                else:
                    multi_fn[fam] += 1
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
    print(f"  Error family-set acc:   {family_set_correct}/{n}  ({100*family_set_correct/n:.1f}%)")
    if required_tool_coverages:
        mean_coverage = sum(required_tool_coverages) / len(required_tool_coverages)
        print(f"  Required-tool coverage: {100*mean_coverage:.1f}% over {len(required_tool_coverages)} samples")
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
    if multi_tp or multi_fp or multi_fn:
        print()
        print("  Multi-label per-family PRF:")
        for fam in sorted(set(multi_tp) | set(multi_fp) | set(multi_fn)):
            tp = multi_tp[fam]
            fp = multi_fp[fam]
            fn = multi_fn[fam]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            print(f"    {fam:<25s}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    if parse_note_counts:
        print()
        print("  Top parser notes:")
        for note, count in parse_note_counts.most_common(8):
            print(f"    {note:<35s}  {count}")
    print("=" * 60)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
