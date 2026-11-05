#!/usr/bin/env python3
"""Post-process a Gemma evaluation JSONL with a stronger reviewer model.

``eval_sft_agent_gemma_v4.py`` writes one JSON row per evaluated sample. Each
row keeps the full per-turn trace (``turn_trace``), including the raw tool
payload the MATPOWER tools returned (``tool_result``) and the compact view the
agent actually saw (``tool_result_for_prompt``), plus the agent's final
structured verdict (``predicted_verdict``).

This script turns those rows into operator-facing natural-language analyses
written by a stronger model. For every row it builds one request from the
model-visible trace only, submits the requests through the Claude Message
Batches API (half price, no latency requirement), and writes the rows back
with:

- ``analysis``: the reviewer's structured output (verdict restatement,
  independent assessment, explanation, evidence chain, concerns, process
  flags, cited entities);
- ``analysis_checks``: local checks that need no model (restatement matches
  the agent's verdict, cited entities exist in the trace, agreement flag is
  consistent with the reviewer's own assessment);
- ``analysis_scores``: label-based scores computed from the ground-truth
  fields the eval rows already carry;
- ``analysis_meta``: model, batch id, stop reason, token usage, status.

Ground-truth fields (``gt_*``, ``family_correct``, ``required_tools*`` and
friends) are never sent to the model; a leak guard refuses to build a request
that contains them. They are used only to score the reviewer afterwards.

Modes:
  run      submit a batch, wait for it, collect and score (default)
  submit   submit a batch and write a manifest; collect later
  collect  fetch an ended batch from its manifest and merge the results
  sync     call the Messages API row by row (small runs, debugging)
  dry-run  write the requests to a JSONL file without calling the API

Examples:
  python explain_eval_results.py --input outputs/gemma4_power_agent/eval_123.jsonl --mode dry-run
  python explain_eval_results.py --input outputs/gemma4_power_agent/eval_123.jsonl
  python explain_eval_results.py --input outputs/gemma4_power_agent/eval_123.jsonl --mode submit
  python explain_eval_results.py --input outputs/gemma4_power_agent/eval_123.jsonl --mode collect
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trace_protocol import ERROR_FAMILIES, maybe_parse_json_string  # noqa: E402

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_ARRAY_ITEMS = 1024
DEFAULT_POLL_SECONDS = 60
DEFAULT_MAX_WAIT_MINUTES = 24 * 60
# Message Batches bill at half the standard rate; these are the Opus 5.5 batch
# rates in USD per million tokens and are used only for the dry-run estimate.
DEFAULT_PRICE_INPUT_PER_MTOK = 2.0
DEFAULT_PRICE_OUTPUT_PER_MTOK = 10.0
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FLOAT_DECIMALS = 4
MANIFEST_SCHEMA_VERSION = 1
CUSTOM_ID_PREFIX = "sample-"

ANALYST_FAMILIES = list(ERROR_FAMILIES) + ["undetermined"]

PROCESS_FLAGS = [
    "no_wls_before_verdict",
    "correction_without_verification",
    "verification_not_resolved",
    "measurement_correction_on_distributed_residuals",
    "tool_error",
    "input_truncated",
    "parse_issue",
    "no_verdict",
    "ambiguous_localization",
    "other",
]

# Row fields that derive from the hidden label. None of them may reach the
# model; ``required_tools*`` are computed from the ground-truth families.
FORBIDDEN_ROW_KEYS = frozenset(
    {
        "gt_error_family",
        "gt_has_error",
        "gt_error_families",
        "family_correct",
        "detection_correct",
        "family_set_correct",
        "required_tools",
        "required_tools_called",
        "required_tools_missing",
        "required_tool_coverage",
    }
)

# Raw WLS keys that are bulky and add nothing the reviewer can use.
DEFAULT_RAW_DROP_KEYS = frozenset(
    {
        "measurement_jacobian",
        "measurement_variance_diag",
        "fitted_measurement",
        "residual_covariance_diag",
        "measurement_rows",
        "measurement_sigma",
        "raw_residual",
        "theta_est_rad",
    }
)

BREAKER_NAME_RE = re.compile(r"CB_[A-Za-z0-9_]+")


ANALYST_SYSTEM_PROMPT = """You are a senior power-system state-estimation engineer reviewing the trace of an automated diagnostic agent.

You receive, as JSON:
- `case`: the snapshot metadata the agent worked from: the case identifier, the measurement channel order, `index_map` giving the 0-based [start, end) index range of each channel, and `branch_info` listing every branch with its 0-based row `i`, `from_bus`, `to_bus`, and whether it is a line or a transformer.
- `turns`: every tool call the agent made, in order, with the arguments it used, the compact result the agent saw (`result_compact`), and additional raw solver output the agent did not see (`result_raw_extras`), followed by the agent's final verdict turn if it produced one.
- `final_verdict`: the agent's final structured verdict, or null if it never produced one.
- `agent_process`: run metadata such as the tool-call sequence, whether the input was truncated, parse notes, and any runtime error.

Conventions:
- Fields ending in `0` are 0-based. `line_index` and `line_index1` are 1-based branch rows; `line_row0` is the 0-based branch row.
- `r` in a WLS result is the vector of normalized residuals over all measurements; an absolute value above about 3 marks a suspicious measurement.
- `lambdaN` is the vector of normalized Lagrange multipliers in `per_branch_R_X_interleaved` layout: entries 2*k and 2*k+1 belong to branch row k (R then X). Large multipliers concentrated on one branch indicate a parameter error on that branch.
- `global_residual_sum` is the WLS objective, `global_residual_threshold` is its chi-square threshold, and `global_residual_ratio` above 1 means the snapshot fails the global test.
- Verification results report `post_action_global_residual_ratio`, `post_action_improved`, and `post_action_resolved` after a correction.

Diagnostic signatures used by this system:
- measurement_error: one or a few large normalized residuals concentrated on a measurement and its neighbours; correcting that measurement restores the global test.
- parameter_error: large normalized Lagrange multipliers concentrated on one branch, often with a passing or marginal global test; parameter correction on that branch restores the test.
- topology_error: widespread large residuals around a substation; breaker context identifies a breaker whose observed status differs from the expected one; topology correction restores the test.
- three_phase_imbalance: a residual pattern inconsistent with a single bad measurement, confirmed by measured three-phase voltages or three-phase NLM evidence and localized to an unbalance source bus.
- harmonic_anomaly: an elevated global residual without a dominant bad measurement, confirmed by harmonic state estimation ranking a source bus.
- high_impedance_fault: three-phase NLM line-group evidence and HIF parameter estimation on a branch, with location-certainty and observability caveats.
- no_error: the global test passes and no residual or multiplier evidence stands out.
Multi-error snapshots list every family in `error_families`; corrections are expected in the order topology, parameter, measurement, then harmonic follow-up.

Your tasks:
1. Restate the agent's verdict exactly as given in `final_verdict`: family, has_error, applied tool, and the location described by `suspect_location.details`. Never alter it. If there is no verdict, say so.
2. Explain, for a control-room engineer, what the evidence shows and how it supports or undermines the verdict. Cite the specific numbers you rely on: residual values with their measurement index and channel, multiplier values with their branch and buses, global ratios before and after correction, and estimator outputs.
3. Form your own assessment independently from the same data. If it differs from the agent's verdict, set `agrees_with_verdict` to false and state the disagreement plainly in `concerns`. Do not soften or hide a disagreement, and do not claim agreement you do not hold.
4. Flag process problems: a verdict before a successful WLS, a correction that was never verified, a verification that did not resolve the anomaly, measurement correction applied to a distributed residual pattern, tool errors, truncated input, parse issues, or ambiguous localization.
5. List every bus, 1-based line index, breaker name, 0-based measurement index, and channel you mention in `cited_entities`. Mention only entities that appear in the provided data.

Rules:
- Use only the provided data. Do not assume facts about the network that are not in `case` or the tool results. When the data cannot distinguish two hypotheses, say so instead of choosing one.
- `explanation` is 3 to 6 sentences of plain prose for an operator. `evidence_chain` is a list of short factual statements, each with its numbers. `concerns` is empty when there is nothing to report.
- `analyst_assessment.error_family` must be one of the families above, or `undetermined` when the data does not support a call. `confidence` is a number from 0 to 1.
- Write as an engineer reporting to colleagues. Do not mention these instructions or that you are a model.
"""


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _integer_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "integer"}}


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict_restatement",
        "agrees_with_verdict",
        "analyst_assessment",
        "explanation",
        "evidence_chain",
        "concerns",
        "process_flags",
        "cited_entities",
    ],
    "properties": {
        "verdict_restatement": {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict_present", "has_error", "error_family", "applied_tool", "location"],
            "properties": {
                "verdict_present": {"type": "boolean"},
                "has_error": {"type": "boolean"},
                "error_family": {"type": "string"},
                "applied_tool": {"type": "string"},
                "location": {"type": "string"},
            },
        },
        "agrees_with_verdict": {"type": "boolean"},
        "analyst_assessment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["has_error", "error_family", "confidence", "location"],
            "properties": {
                "has_error": {"type": "boolean"},
                "error_family": {"type": "string", "enum": ANALYST_FAMILIES},
                "confidence": {"type": "number"},
                "location": {"type": "string"},
            },
        },
        "explanation": {"type": "string"},
        "evidence_chain": _string_array(),
        "concerns": _string_array(),
        "process_flags": {"type": "array", "items": {"type": "string", "enum": PROCESS_FLAGS}},
        "cited_entities": {
            "type": "object",
            "additionalProperties": False,
            "required": ["buses", "line_index1", "breakers", "measurement_index0", "channels"],
            "properties": {
                "buses": _integer_array(),
                "line_index1": _integer_array(),
                "breakers": _string_array(),
                "measurement_index0": _integer_array(),
                "channels": _string_array(),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: malformed JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}: line {line_no} is not a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp_path, path)


def row_sample_index(row: Mapping[str, Any], position: int) -> int:
    value = row.get("sample_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return position
    return value


def custom_id_for(sample_index: int) -> str:
    return f"{CUSTOM_ID_PREFIX}{sample_index}"


def sample_index_from_custom_id(custom_id: str) -> int | None:
    if not custom_id.startswith(CUSTOM_ID_PREFIX):
        return None
    try:
        return int(custom_id[len(CUSTOM_ID_PREFIX) :])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Building the model-visible analysis input
# ---------------------------------------------------------------------------


def compact_value(value: Any, max_items: int) -> Any:
    """Round floats and cap long arrays so a request stays a few thousand tokens."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, Mapping):
        return {str(key): compact_value(item, max_items) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [compact_value(item, max_items) for item in list(value)[:max_items]]
        dropped = len(value) - len(items)
        if dropped > 0:
            items.append(f"... ({dropped} more items omitted)")
        return items
    return str(value)


def extract_user_payloads(conversation: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in conversation or []:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = maybe_parse_json_string(message.get("content"))
        if isinstance(content, Mapping):
            payloads.append(dict(content))
    return payloads


def build_case_block(
    user_payloads: Sequence[Mapping[str, Any]], *, include_measurements: bool, max_items: int
) -> dict[str, Any] | None:
    if not user_payloads:
        return None
    snapshot = user_payloads[0]
    case: dict[str, Any] = {}
    for key in ("case_path", "measurement_order", "index_map", "branch_info"):
        if key in snapshot:
            case[key] = compact_value(snapshot[key], max_items)
    z_obs = snapshot.get("z_obs")
    if isinstance(z_obs, list):
        case["measurement_count"] = len(z_obs)
        if include_measurements:
            case["z_obs"] = compact_value(z_obs, max_items)
    extra_keys = sorted(
        key for key in snapshot if key not in {"case_path", "measurement_order", "index_map", "branch_info", "z_obs"}
    )
    if extra_keys:
        case["other_snapshot_fields"] = {
            key: compact_value(snapshot[key], max_items) for key in extra_keys
        }
    if len(user_payloads) > 1:
        case["follow_up_user_payloads"] = [
            compact_value(payload, max_items) for payload in user_payloads[1:]
        ]
    return case


def build_turn_blocks(
    turn_trace: Sequence[Mapping[str, Any]] | None,
    *,
    max_items: int,
    raw_drop_keys: frozenset[str],
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for record in turn_trace or []:
        if not isinstance(record, Mapping):
            continue
        block: dict[str, Any] = {"turn": record.get("turn"), "parse_type": record.get("parse_type")}
        if record.get("tool_name"):
            block["tool_name"] = record["tool_name"]
            arguments = record.get("tool_arguments_for_prompt", record.get("tool_arguments"))
            block["arguments"] = compact_value(maybe_parse_json_string(arguments), max_items)
            compact = maybe_parse_json_string(record.get("tool_result_for_prompt"))
            if compact is not None:
                block["result_compact"] = compact_value(compact, max_items)
            raw = maybe_parse_json_string(record.get("tool_result"))
            if isinstance(raw, Mapping):
                compact_keys = set(compact.keys()) if isinstance(compact, Mapping) else set()
                extras = {
                    key: value
                    for key, value in raw.items()
                    if key not in raw_drop_keys and key not in compact_keys
                }
                if extras:
                    block["result_raw_extras"] = compact_value(extras, max_items)
        if record.get("verdict") is not None:
            if record.get("error"):
                # A rejected verdict (for example one issued before WLS) is not
                # the final verdict, so the reviewer needs its full content.
                block["rejected_verdict"] = compact_value(maybe_parse_json_string(record["verdict"]), max_items)
            else:
                block["verdict"] = "see final_verdict"
        for key in ("error", "parse_notes", "prompt_truncated", "continued_after_tool_error", "tool_error_type"):
            if record.get(key):
                block[key] = compact_value(record[key], max_items)
        turns.append(block)
    return turns


def build_analysis_input(
    row: Mapping[str, Any],
    *,
    include_measurements: bool = False,
    max_array_items: int = DEFAULT_MAX_ARRAY_ITEMS,
    raw_drop_keys: frozenset[str] = DEFAULT_RAW_DROP_KEYS,
) -> dict[str, Any]:
    """Assemble what the reviewer sees. Only model-visible fields are copied."""
    user_payloads = extract_user_payloads(row.get("final_conversation"))
    runtime = row.get("runtime") if isinstance(row.get("runtime"), Mapping) else {}
    agent_process = {
        "tool_calls": list(row.get("tool_calls") or []),
        "num_turns": row.get("num_turns"),
        "required_wls_satisfied": row.get("required_wls_satisfied"),
        "input_truncated": bool(row.get("input_truncated")),
        "parse_notes": list(row.get("parse_notes") or []),
        "runtime_error": row.get("error"),
        "predicted_applied_tools": list(row.get("predicted_applied_tools") or []),
        "runtime_summary": compact_value(dict(runtime), max_array_items) if runtime else None,
    }
    analysis_input = {
        "sample_index": row.get("sample_index"),
        "case": build_case_block(
            user_payloads, include_measurements=include_measurements, max_items=max_array_items
        ),
        "turns": build_turn_blocks(row.get("turn_trace"), max_items=max_array_items, raw_drop_keys=raw_drop_keys),
        "final_verdict": compact_value(maybe_parse_json_string(row.get("predicted_verdict")), max_array_items),
        "agent_process": agent_process,
    }
    assert_no_label_leak(analysis_input)
    return analysis_input


def assert_no_label_leak(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            if key_text in FORBIDDEN_ROW_KEYS or key_text.startswith("gt_"):
                raise RuntimeError(f"Label field {key_text!r} would reach the model at {path}")
            assert_no_label_leak(value, f"{path}.{key_text}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_no_label_leak(item, f"{path}[{index}]")


def row_has_trace(row: Mapping[str, Any]) -> bool:
    return bool(row.get("turn_trace")) or row.get("predicted_verdict") is not None


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def user_message_text(analysis_input: Mapping[str, Any], *, structured_output: bool) -> str:
    text = (
        "Review this diagnostic trace and return the analysis as JSON matching the required schema.\n\n"
        + json.dumps(analysis_input, ensure_ascii=False, separators=(",", ":"))
    )
    if not structured_output:
        text += (
            "\n\nReturn only a JSON object, with no prose before or after it, matching this schema:\n"
            + json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False, separators=(",", ":"))
        )
    return text


def build_message_params(
    analysis_input: Mapping[str, Any],
    *,
    model: str,
    max_tokens: int,
    effort: str,
    system_prompt: str,
    structured_output: bool,
) -> dict[str, Any]:
    output_config: dict[str, Any] = {"effort": effort}
    if structured_output:
        output_config["format"] = {"type": "json_schema", "schema": ANALYSIS_SCHEMA}
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": user_message_text(analysis_input, structured_output=structured_output),
            }
        ],
        "output_config": output_config,
    }


def build_batch_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    max_tokens: int,
    effort: str,
    system_prompt: str,
    structured_output: bool,
    include_measurements: bool,
    max_array_items: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    requests: list[dict[str, Any]] = []
    custom_ids: dict[str, int] = {}
    for position, row in enumerate(rows):
        sample_index = row_sample_index(row, position)
        analysis_input = build_analysis_input(
            row, include_measurements=include_measurements, max_array_items=max_array_items
        )
        custom_id = custom_id_for(sample_index)
        if custom_id in custom_ids:
            raise RuntimeError(f"Duplicate sample_index {sample_index} in input rows")
        custom_ids[custom_id] = sample_index
        requests.append(
            {
                "custom_id": custom_id,
                "params": build_message_params(
                    analysis_input,
                    model=model,
                    max_tokens=max_tokens,
                    effort=effort,
                    system_prompt=system_prompt,
                    structured_output=structured_output,
                ),
            }
        )
    return requests, custom_ids


def estimate_tokens(text: str) -> int:
    # Rough: the tokenizer averages near four characters per token on JSON.
    return max(1, len(text) // 4)


def estimate_request_cost(
    requests: Sequence[Mapping[str, Any]],
    *,
    price_input_per_mtok: float,
    price_output_per_mtok: float,
    assumed_output_tokens: int = 700,
) -> dict[str, Any]:
    input_tokens = 0
    for request in requests:
        params = request["params"]
        system_text = " ".join(block["text"] for block in params["system"])
        user_text = " ".join(
            message["content"] if isinstance(message["content"], str) else json.dumps(message["content"])
            for message in params["messages"]
        )
        input_tokens += estimate_tokens(system_text) + estimate_tokens(user_text)
    output_tokens = assumed_output_tokens * len(requests)
    return {
        "requests": len(requests),
        "estimated_input_tokens": input_tokens,
        "assumed_output_tokens": output_tokens,
        "estimated_cost_usd": round(
            input_tokens / 1e6 * price_input_per_mtok + output_tokens / 1e6 * price_output_per_mtok, 2
        ),
        "price_input_per_mtok": price_input_per_mtok,
        "price_output_per_mtok": price_output_per_mtok,
    }


# ---------------------------------------------------------------------------
# Parsing responses
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def message_to_record(message: Any) -> dict[str, Any]:
    """Flatten an SDK Message (or an equivalent dict) into plain JSON data."""
    text_parts: list[str] = []
    for block in _field(message, "content", []) or []:
        if _field(block, "type") == "text":
            text_parts.append(str(_field(block, "text", "")))
    usage = _field(message, "usage")
    usage_record = None
    if usage is not None:
        usage_record = {
            key: _field(usage, key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        }
    stop_details = _field(message, "stop_details")
    return {
        "text": "".join(text_parts),
        "stop_reason": _field(message, "stop_reason"),
        "stop_details": (
            {"category": _field(stop_details, "category"), "explanation": _field(stop_details, "explanation")}
            if stop_details is not None
            else None
        ),
        "model": _field(message, "model"),
        "usage": usage_record,
    }


def first_json_object(text: str) -> Any:
    """Return the first balanced JSON object in ``text`` (tolerates wrapped prose)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.DOTALL)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : index + 1])
                    except json.JSONDecodeError:
                        break
        start = stripped.find("{", start + 1)
    raise ValueError("no JSON object found in model output")


def parse_analysis_record(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return (analysis, status, error) for a flattened message record."""
    stop_reason = record.get("stop_reason")
    if stop_reason == "refusal":
        details = record.get("stop_details") or {}
        return None, "refused", f"refusal:{details.get('category')}:{details.get('explanation')}"
    text = record.get("text") or ""
    try:
        parsed = first_json_object(text)
    except ValueError as exc:
        status = "truncated" if stop_reason == "max_tokens" else "parse_error"
        return None, status, f"{exc}; stop_reason={stop_reason}"
    if not isinstance(parsed, dict):
        return None, "parse_error", "model output is not a JSON object"
    missing = [key for key in ANALYSIS_SCHEMA["required"] if key not in parsed]
    if missing:
        status = "truncated" if stop_reason == "max_tokens" else "schema_error"
        return parsed, status, f"missing keys: {missing}"
    if stop_reason == "max_tokens":
        return parsed, "truncated", "stop_reason=max_tokens"
    return parsed, "ok", None


# ---------------------------------------------------------------------------
# Local checks and scoring (no model involved)
# ---------------------------------------------------------------------------


def entity_universe(analysis_input: Mapping[str, Any]) -> dict[str, Any]:
    case = analysis_input.get("case") or {}
    buses: set[int] = set()
    branch_info = case.get("branch_info") if isinstance(case, Mapping) else None
    n_branches = 0
    if isinstance(branch_info, list):
        for branch in branch_info:
            if not isinstance(branch, Mapping):
                continue
            n_branches += 1
            for key in ("from_bus", "to_bus"):
                value = branch.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    buses.add(value)
    channels: set[str] = set()
    measurement_count = 0
    if isinstance(case, Mapping):
        order = case.get("measurement_order")
        if isinstance(order, list):
            channels.update(str(item) for item in order)
        index_map = case.get("index_map")
        if isinstance(index_map, Mapping):
            channels.update(str(key) for key in index_map)
            for bounds in index_map.values():
                if isinstance(bounds, list) and len(bounds) == 2 and isinstance(bounds[1], int):
                    measurement_count = max(measurement_count, bounds[1])
        count = case.get("measurement_count")
        if isinstance(count, int) and not isinstance(count, bool):
            measurement_count = max(measurement_count, count)
    serialized = json.dumps(analysis_input, ensure_ascii=False)
    breakers = set(BREAKER_NAME_RE.findall(serialized))
    return {
        "known": bool(buses) or n_branches > 0 or measurement_count > 0,
        "buses": buses,
        "n_branches": n_branches,
        "measurement_count": measurement_count,
        "channels": channels,
        "breakers": breakers,
    }


def _verdict_block(verdict_obj: Any) -> dict[str, Any]:
    if isinstance(verdict_obj, Mapping) and isinstance(verdict_obj.get("verdict"), Mapping):
        return dict(verdict_obj["verdict"])
    return {}


def run_local_checks(
    analysis: Mapping[str, Any] | None,
    analysis_input: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(analysis, Mapping):
        return {"checked": False, "passed": False, "problems": ["no_analysis"]}
    problems: list[str] = []
    predicted = analysis_input.get("final_verdict")
    predicted_verdict = _verdict_block(predicted)
    restatement = analysis.get("verdict_restatement") or {}
    verdict_present = predicted_verdict != {}

    if bool(restatement.get("verdict_present")) != verdict_present:
        problems.append("restatement_presence_mismatch")
    if verdict_present:
        pred_family = predicted_verdict.get("error_family")
        if (restatement.get("error_family") or "") != (pred_family or ""):
            problems.append("restatement_family_mismatch")
        pred_has_error = predicted_verdict.get("has_error")
        if isinstance(pred_has_error, bool) and restatement.get("has_error") is not pred_has_error:
            problems.append("restatement_has_error_mismatch")

    assessment = analysis.get("analyst_assessment") or {}
    analyst_family = assessment.get("error_family")
    if analyst_family not in ANALYST_FAMILIES:
        problems.append("analyst_family_invalid")
    confidence = assessment.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        problems.append("analyst_confidence_out_of_range")

    agrees = analysis.get("agrees_with_verdict")
    if verdict_present and isinstance(agrees, bool):
        same_family = analyst_family == predicted_verdict.get("error_family")
        same_detection = (
            not isinstance(predicted_verdict.get("has_error"), bool)
            or assessment.get("has_error") is predicted_verdict.get("has_error")
        )
        if agrees and not (same_family and same_detection):
            problems.append("agreement_inconsistent_with_assessment")
        if not agrees and same_family and same_detection and not analysis.get("concerns"):
            problems.append("disagreement_without_concerns")

    universe = entity_universe(analysis_input)
    cited = analysis.get("cited_entities") or {}
    unknown: dict[str, list[Any]] = {}
    if universe["known"]:
        if universe["buses"]:
            bad = [bus for bus in cited.get("buses") or [] if bus not in universe["buses"]]
            if bad:
                unknown["buses"] = bad
        if universe["n_branches"]:
            bad = [line for line in cited.get("line_index1") or [] if not 1 <= line <= universe["n_branches"]]
            if bad:
                unknown["line_index1"] = bad
        if universe["measurement_count"]:
            bad = [
                index
                for index in cited.get("measurement_index0") or []
                if not 0 <= index < universe["measurement_count"]
            ]
            if bad:
                unknown["measurement_index0"] = bad
        if universe["channels"]:
            bad = [channel for channel in cited.get("channels") or [] if channel not in universe["channels"]]
            if bad:
                unknown["channels"] = bad
    bad_breakers = [name for name in cited.get("breakers") or [] if name not in universe["breakers"]]
    if bad_breakers:
        unknown["breakers"] = bad_breakers
    if unknown:
        problems.append("unknown_entities_cited")

    explanation = analysis.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        problems.append("empty_explanation")

    return {
        "checked": True,
        "passed": not problems,
        "problems": problems,
        "unknown_entities": unknown,
        "entity_universe_known": universe["known"],
    }


def score_row(row: Mapping[str, Any], analysis: Mapping[str, Any] | None) -> dict[str, Any]:
    """Label-based scores. Reads gt_* fields that were never sent to the model."""
    gt_family = row.get("gt_error_family")
    gt_families = set(row.get("gt_error_families") or ([] if gt_family in (None, "no_error") else [gt_family]))
    gt_has_error = row.get("gt_has_error")
    agent_correct = row.get("family_correct")
    scores: dict[str, Any] = {
        "agent_family_correct": agent_correct,
        "analyst_family_correct": None,
        "analyst_family_in_gt_set": None,
        "analyst_detection_correct": None,
        "analyst_agrees_with_agent": None,
        "review_outcome": None,
    }
    if not isinstance(analysis, Mapping):
        return scores
    assessment = analysis.get("analyst_assessment") or {}
    analyst_family = assessment.get("error_family")
    analyst_has_error = assessment.get("has_error")
    agrees = analysis.get("agrees_with_verdict")
    scores["analyst_family_correct"] = analyst_family == gt_family if gt_family is not None else None
    scores["analyst_family_in_gt_set"] = (
        (analyst_family in gt_families) if gt_families else (analyst_family == "no_error")
    )
    if isinstance(gt_has_error, bool) and isinstance(analyst_has_error, bool):
        scores["analyst_detection_correct"] = analyst_has_error == gt_has_error
    scores["analyst_agrees_with_agent"] = agrees if isinstance(agrees, bool) else None
    if isinstance(agent_correct, bool) and isinstance(agrees, bool):
        if agent_correct and agrees:
            scores["review_outcome"] = "confirmed_correct"
        elif agent_correct and not agrees:
            scores["review_outcome"] = "false_alarm"
        elif not agent_correct and not agrees:
            scores["review_outcome"] = "caught_error"
        else:
            scores["review_outcome"] = "missed_error"
    return scores


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str((row.get("analysis_meta") or {}).get("status")) for row in rows)
    analyzed = [row for row in rows if (row.get("analysis_meta") or {}).get("status") == "ok"]
    outcomes = Counter(str((row.get("analysis_scores") or {}).get("review_outcome")) for row in analyzed)
    agent_correct = sum(1 for row in analyzed if row.get("family_correct") is True)
    analyst_correct = sum(1 for row in analyzed if (row.get("analysis_scores") or {}).get("analyst_family_correct"))
    analyst_detection = sum(
        1 for row in analyzed if (row.get("analysis_scores") or {}).get("analyst_detection_correct")
    )
    checks_passed = sum(1 for row in analyzed if (row.get("analysis_checks") or {}).get("passed"))
    problem_counts = Counter(
        problem for row in analyzed for problem in (row.get("analysis_checks") or {}).get("problems") or []
    )
    flag_counts = Counter(
        flag for row in analyzed for flag in (row.get("analysis") or {}).get("process_flags") or []
    )
    per_family: dict[str, dict[str, Any]] = {}
    for row in analyzed:
        family = str(row.get("gt_error_family") or "unknown")
        bucket = per_family.setdefault(family, {"rows": 0, "agent_correct": 0, "analyst_correct": 0})
        bucket["rows"] += 1
        bucket["agent_correct"] += int(row.get("family_correct") is True)
        bucket["analyst_correct"] += int(bool((row.get("analysis_scores") or {}).get("analyst_family_correct")))
    usage_totals: Counter[str] = Counter()
    for row in rows:
        usage = (row.get("analysis_meta") or {}).get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value
    agent_wrong = outcomes["caught_error"] + outcomes["missed_error"]
    agent_right = outcomes["confirmed_correct"] + outcomes["false_alarm"]
    return {
        "rows": len(rows),
        "analyzed": len(analyzed),
        "statuses": dict(statuses),
        "agent_family_accuracy": _rate(agent_correct, len(analyzed)),
        "analyst_family_accuracy": _rate(analyst_correct, len(analyzed)),
        "analyst_detection_accuracy": _rate(analyst_detection, len(analyzed)),
        "reviewer_recall_on_agent_errors": _rate(outcomes["caught_error"], agent_wrong),
        "reviewer_false_alarm_rate": _rate(outcomes["false_alarm"], agent_right),
        "review_outcomes": dict(outcomes),
        "local_check_pass_rate": _rate(checks_passed, len(analyzed)),
        "local_check_problems": dict(problem_counts),
        "process_flags": dict(flag_counts),
        "per_family": per_family,
        "usage_totals": dict(usage_totals),
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print("\n=== Reviewer summary ===")
    print(f"rows: {summary['rows']}  analyzed: {summary['analyzed']}  statuses: {summary['statuses']}")
    for key in (
        "agent_family_accuracy",
        "analyst_family_accuracy",
        "analyst_detection_accuracy",
        "reviewer_recall_on_agent_errors",
        "reviewer_false_alarm_rate",
        "local_check_pass_rate",
    ):
        value = summary.get(key)
        print(f"  {key:<34s} {value if value is not None else 'n/a'}")
    print(f"  review outcomes: {summary['review_outcomes']}")
    if summary.get("local_check_problems"):
        print(f"  local check problems: {summary['local_check_problems']}")
    if summary.get("process_flags"):
        print(f"  process flags: {summary['process_flags']}")
    print("  per family (rows / agent correct / analyst correct):")
    for family, bucket in sorted(summary.get("per_family", {}).items()):
        print(f"    {family:<24s} {bucket['rows']:>4d} / {bucket['agent_correct']:>4d} / {bucket['analyst_correct']:>4d}")
    if summary.get("usage_totals"):
        print(f"  token usage: {summary['usage_totals']}")


# ---------------------------------------------------------------------------
# Merging results into rows
# ---------------------------------------------------------------------------


def attach_result(
    row: Mapping[str, Any],
    *,
    record: Mapping[str, Any] | None,
    status: str,
    error: str | None,
    meta: Mapping[str, Any],
    include_measurements: bool,
    max_array_items: int,
) -> dict[str, Any]:
    out = dict(row)
    analysis: dict[str, Any] | None = None
    final_status = status
    final_error = error
    if record is not None:
        analysis, parse_status, parse_error = parse_analysis_record(record)
        if status == "succeeded":
            final_status, final_error = parse_status, parse_error
    analysis_input = (
        build_analysis_input(row, include_measurements=include_measurements, max_array_items=max_array_items)
        if row_has_trace(row)
        else {"final_verdict": None}
    )
    out["analysis"] = analysis
    out["analysis_checks"] = run_local_checks(analysis if final_status == "ok" else None, analysis_input)
    out["analysis_scores"] = score_row(row, analysis if final_status == "ok" else None)
    out["analysis_meta"] = {
        **dict(meta),
        "status": final_status,
        "error": final_error,
        "stop_reason": record.get("stop_reason") if record else None,
        "model_served": record.get("model") if record else None,
        "usage": record.get("usage") if record else None,
    }
    return out


def merge_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    results_by_index: Mapping[int, Mapping[str, Any]],
    existing_by_index: Mapping[int, Mapping[str, Any]],
    base_meta: Mapping[str, Any],
    include_measurements: bool,
    max_array_items: int,
) -> list[dict[str, Any]]:
    """Overlay batch results on the input rows, keeping earlier good analyses."""
    merged: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        sample_index = row_sample_index(row, position)
        result = results_by_index.get(sample_index)
        existing = existing_by_index.get(sample_index)
        if result is not None:
            merged.append(
                attach_result(
                    row,
                    record=result.get("record"),
                    status=str(result.get("status")),
                    error=result.get("error"),
                    meta={**base_meta, "custom_id": custom_id_for(sample_index)},
                    include_measurements=include_measurements,
                    max_array_items=max_array_items,
                )
            )
        elif existing is not None and "analysis_meta" in existing:
            merged.append(dict(existing))
        else:
            merged.append(
                attach_result(
                    row,
                    record=None,
                    status="not_requested" if row_has_trace(row) else "skipped_no_trace",
                    error=None,
                    meta={**base_meta, "custom_id": None},
                    include_measurements=include_measurements,
                    max_array_items=max_array_items,
                )
            )
    return merged


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def make_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("The anthropic package is required for API modes: pip install anthropic") from exc
    return anthropic.Anthropic()


def submit_batch(client: Any, requests: Sequence[Mapping[str, Any]]) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    typed_requests = [
        Request(custom_id=request["custom_id"], params=MessageCreateParamsNonStreaming(**request["params"]))
        for request in requests
    ]
    batch = client.messages.batches.create(requests=typed_requests)
    return str(batch.id)


def wait_for_batch(client: Any, batch_id: str, *, poll_seconds: int, max_wait_minutes: int) -> Any:
    deadline = time.monotonic() + max_wait_minutes * 60
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"batch {batch_id}: {batch.processing_status} "
            f"(processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired})"
        )
        if batch.processing_status == "ended":
            return batch
        if time.monotonic() >= deadline:
            raise SystemExit(f"batch {batch_id} did not end within {max_wait_minutes} minutes; rerun with --mode collect")
        time.sleep(poll_seconds)


def collect_batch_results(client: Any, batch_id: str) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for item in client.messages.batches.results(batch_id):
        sample_index = sample_index_from_custom_id(str(item.custom_id))
        if sample_index is None:
            print(f"skipping result with unexpected custom_id {item.custom_id!r}")
            continue
        result_type = str(item.result.type)
        if result_type == "succeeded":
            results[sample_index] = {
                "status": "succeeded",
                "record": message_to_record(item.result.message),
                "error": None,
            }
        else:
            error = getattr(item.result, "error", None)
            error_text = None
            if error is not None:
                error_text = f"{getattr(error, 'type', None)}: {getattr(error, 'message', None)}"
            results[sample_index] = {"status": result_type, "record": None, "error": error_text}
    return results


def run_sync_requests(
    client: Any, requests: Sequence[Mapping[str, Any]], *, use_fallbacks: bool
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    total = len(requests)
    for position, request in enumerate(requests, start=1):
        sample_index = sample_index_from_custom_id(request["custom_id"])
        if sample_index is None:
            continue
        params = dict(request["params"])
        try:
            if use_fallbacks:
                message = client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                message = client.messages.create(**params)
            results[sample_index] = {
                "status": "succeeded",
                "record": message_to_record(message),
                "error": None,
            }
            print(f"[{position}/{total}] sample {sample_index}: {message.stop_reason}")
        except Exception as exc:  # noqa: BLE001 - one failed row must not abort the run
            results[sample_index] = {"status": "errored", "record": None, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{position}/{total}] sample {sample_index}: {type(exc).__name__}: {exc}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Evaluation JSONL written by eval_sft_agent_gemma_v4.py")
    parser.add_argument("--output", default=None, help="Output JSONL (default: <input stem>.explained.jsonl)")
    parser.add_argument(
        "--mode",
        choices=["run", "submit", "collect", "sync", "dry-run"],
        default="run",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N eligible rows")
    parser.add_argument(
        "--sample-index", type=int, action="append", default=None, help="Only these sample indices (repeatable)"
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip rows whose existing output already has a successful analysis",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also request analyses for rows with no trace and no verdict (critical eval errors)",
    )
    parser.add_argument(
        "--include-measurements",
        action="store_true",
        help="Include the raw z_obs measurement vector in the request",
    )
    parser.add_argument("--max-array-items", type=int, default=DEFAULT_MAX_ARRAY_ITEMS)
    parser.add_argument("--system-prompt-file", default=None, help="Override the reviewer system prompt")
    parser.add_argument("--manifest", default=None, help="Batch manifest path (default: <output>.batch.json)")
    parser.add_argument("--batch-id", default=None, help="Collect this batch id instead of the manifest's")
    parser.add_argument("--requests-file", default=None, help="Dry-run output (default: <output>.requests.jsonl)")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-wait-minutes", type=int, default=DEFAULT_MAX_WAIT_MINUTES)
    parser.add_argument(
        "--no-fallbacks",
        dest="fallbacks",
        action="store_false",
        help="Sync mode: do not route refusals to a fallback model",
    )
    parser.add_argument(
        "--no-structured-output",
        dest="structured_output",
        action="store_false",
        help="Ask for JSON in the prompt instead of enforcing the schema with output_config",
    )
    parser.add_argument("--price-input", type=float, default=DEFAULT_PRICE_INPUT_PER_MTOK)
    parser.add_argument("--price-output", type=float, default=DEFAULT_PRICE_OUTPUT_PER_MTOK)
    parser.set_defaults(fallbacks=True, structured_output=True)
    return parser.parse_args(argv)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + ".explained.jsonl")


def load_system_prompt(path: str | None) -> str:
    if not path:
        return ANALYST_SYSTEM_PROMPT
    return Path(path).read_text(encoding="utf-8")


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    existing_by_index: Mapping[int, Mapping[str, Any]],
    only_missing: bool,
    include_failed: bool,
    limit: int | None,
    sample_indices: Sequence[int] | None,
) -> list[Mapping[str, Any]]:
    wanted = set(sample_indices) if sample_indices else None
    selected: list[Mapping[str, Any]] = []
    for position, row in enumerate(rows):
        sample_index = row_sample_index(row, position)
        if wanted is not None and sample_index not in wanted:
            continue
        if not include_failed and not row_has_trace(row):
            continue
        if only_missing:
            existing = existing_by_index.get(sample_index) or {}
            if (existing.get("analysis_meta") or {}).get("status") == "ok":
                continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finish(
    rows: Sequence[Mapping[str, Any]],
    *,
    results_by_index: Mapping[int, Mapping[str, Any]],
    existing_by_index: Mapping[int, Mapping[str, Any]],
    base_meta: Mapping[str, Any],
    output_path: Path,
    include_measurements: bool,
    max_array_items: int,
) -> dict[str, Any]:
    merged = merge_results(
        rows,
        results_by_index=results_by_index,
        existing_by_index=existing_by_index,
        base_meta=base_meta,
        include_measurements=include_measurements,
        max_array_items=max_array_items,
    )
    write_jsonl(output_path, merged)
    summary = summarize_rows(merged)
    summary_path = output_path.with_name(output_path.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(merged)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")
    print_summary(summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_name(output_path.stem + ".batch.json")
    rows = load_jsonl(input_path)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")
    existing_rows = load_jsonl(output_path) if output_path.exists() else []
    existing_by_index = {row_sample_index(row, position): row for position, row in enumerate(existing_rows)}
    system_prompt = load_system_prompt(args.system_prompt_file)
    base_meta = {
        "model": args.model,
        "effort": args.effort,
        "structured_output": args.structured_output,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_id": None,
    }

    if args.mode == "collect":
        batch_id = args.batch_id
        manifest: dict[str, Any] = {}
        if batch_id is None:
            if not manifest_path.exists():
                raise SystemExit(f"no manifest at {manifest_path}; pass --batch-id or run --mode submit first")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            batch_id = str(manifest["batch_id"])
        client = make_client()
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            counts = batch.request_counts
            print(
                f"batch {batch_id} is {batch.processing_status} "
                f"(processing={counts.processing} succeeded={counts.succeeded} errored={counts.errored}); "
                "try again later"
            )
            return 2
        results = collect_batch_results(client, batch_id)
        meta = {**base_meta, **{key: manifest.get(key, base_meta[key]) for key in ("model", "effort", "structured_output")}}
        meta["batch_id"] = batch_id
        finish(
            rows,
            results_by_index=results,
            existing_by_index=existing_by_index,
            base_meta=meta,
            output_path=output_path,
            include_measurements=bool(manifest.get("include_measurements", args.include_measurements)),
            max_array_items=int(manifest.get("max_array_items", args.max_array_items)),
        )
        return 0

    selected = select_rows(
        rows,
        existing_by_index=existing_by_index,
        only_missing=args.only_missing,
        include_failed=args.include_failed,
        limit=args.limit,
        sample_indices=args.sample_index,
    )
    if not selected:
        print("no rows selected; nothing to do")
        return 0
    requests, custom_ids = build_batch_requests(
        selected,
        model=args.model,
        max_tokens=args.max_tokens,
        effort=args.effort,
        system_prompt=system_prompt,
        structured_output=args.structured_output,
        include_measurements=args.include_measurements,
        max_array_items=args.max_array_items,
    )
    estimate = estimate_request_cost(
        requests, price_input_per_mtok=args.price_input, price_output_per_mtok=args.price_output
    )
    print(
        f"built {estimate['requests']} requests: ~{estimate['estimated_input_tokens']:,} input tokens, "
        f"~{estimate['assumed_output_tokens']:,} output tokens assumed, "
        f"~${estimate['estimated_cost_usd']:.2f} at batch rates"
    )

    if args.mode == "dry-run":
        requests_path = Path(args.requests_file) if args.requests_file else output_path.with_name(
            output_path.stem + ".requests.jsonl"
        )
        write_jsonl(requests_path, requests)
        print(f"wrote {len(requests)} requests to {requests_path} (no API call made)")
        return 0

    client = make_client()
    if args.mode == "sync":
        if args.fallbacks:
            print(f"sync mode: refusals are routed to a fallback model (beta {FALLBACK_BETA}); use --no-fallbacks to disable")
        results = run_sync_requests(client, requests, use_fallbacks=args.fallbacks)
        finish(
            rows,
            results_by_index=results,
            existing_by_index=existing_by_index,
            base_meta=base_meta,
            output_path=output_path,
            include_measurements=args.include_measurements,
            max_array_items=args.max_array_items,
        )
        return 0

    batch_id = submit_batch(client, requests)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": base_meta["generated_at"],
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "effort": args.effort,
        "structured_output": args.structured_output,
        "include_measurements": args.include_measurements,
        "max_array_items": args.max_array_items,
        "request_count": len(requests),
        "custom_ids": custom_ids,
    }
    write_manifest(manifest_path, manifest)
    print(f"submitted batch {batch_id} with {len(requests)} requests; manifest at {manifest_path}")
    if args.mode == "submit":
        print(f"collect later with: python {Path(__file__).name} --input {input_path} --output {output_path} --mode collect")
        return 0

    wait_for_batch(client, batch_id, poll_seconds=args.poll_seconds, max_wait_minutes=args.max_wait_minutes)
    results = collect_batch_results(client, batch_id)
    finish(
        rows,
        results_by_index=results,
        existing_by_index=existing_by_index,
        base_meta={**base_meta, "batch_id": batch_id},
        output_path=output_path,
        include_measurements=args.include_measurements,
        max_array_items=args.max_array_items,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
