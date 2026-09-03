"""Pinned E4B/12B processor and training/inference prompt-parity audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval_sft_agent_gemma_v4 import render_eval_text, tokenize_rendered_text
from gpt_oss_power_sft_revised_v3 import encode_text

from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    validate_policy_payload,
)
from psse_env.dagger.preliminary_e2b_eval import (
    FORCED_TOOL_PREFIX,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    canonical_prompt_tool_schemas,
)
from psse_env.research_models import (
    GEMMA4_12B,
    GEMMA4_E4B,
    PROMPT_PROFILE_NATIVE,
    PROMPT_PROFILE_SMALL_FORCED,
    ResearchModelSpec,
    get_research_model_spec,
)

from .gates import (
    GateError,
    load_exact_processor,
    load_jsonl,
    prepare_example,
    verify_assistant_only_mask,
)
from .research_rows import normalize_research_rows


AUDIT_CONTRACT = "research_gemma4_processor_parity_v1"
_UNIFIED_GENERATION_SUFFIX = "<|turn>model\n<|channel>thought\n<channel|>"
_SMALL_GENERATION_SUFFIX = "<|turn>model\n"
_PRIORITY_TOOLS = (
    "wls_from_path",
    "get_parameter_context",
    "get_topology_context",
    "get_harmonic_context",
    "get_verification_snapshot",
    "get_measurement_context",
    "correct_measurements_from_path",
    "correct_parameters_from_path",
    "correct_topology_from_path",
    "run_hse_from_path",
    "run_three_phase_nlm_from_path",
    "estimate_hif_location_magnitude_from_path",
    "estimate_hif_location_magnitude_multiscan_from_path",
    "commit_state",
    "rollback_state",
    "finalize_diagnosis",
    "ask_for_more_evidence",
    "run_alternative_test",
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")
    os.replace(temporary, path)


def _target_tool(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    target = messages[-1] if isinstance(messages, list) and messages else None
    calls = target.get("tool_calls") if isinstance(target, Mapping) else None
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    function = call.get("function") if isinstance(call, Mapping) else None
    return str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""


def _state_class(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(row.get("state_class") or metadata.get("state_class") or "unknown")


def _state_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {}
    users = [item for item in messages if isinstance(item, Mapping) and item.get("role") == "user"]
    if not users or not isinstance(users[-1].get("content"), str):
        return {}
    try:
        payload = json.loads(users[-1]["content"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    state = payload.get("state") if isinstance(payload, Mapping) else None
    return state if isinstance(state, Mapping) else {}


def _history_length(row: Mapping[str, Any]) -> int:
    history = _state_payload(row).get("history_window")
    return len(history) if isinstance(history, list) else 0


def select_representative_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 32,
    source_cohort: str = "d0",
) -> list[dict[str, Any]]:
    """Select a deterministic tool/class/long-history audit slice."""

    if limit <= 0:
        raise ValueError("audit row limit must be positive")
    candidates: list[dict[str, Any]] = []
    seen_payloads: dict[str, str] = {}
    for index, source in enumerate(rows):
        row = copy.deepcopy(dict(source))
        source_index = int(row.pop("_research_audit_source_index", index))
        serialized = _stable_json(row)
        fingerprint = str(row.get("example_id") or serialized)
        if fingerprint in seen_payloads:
            if seen_payloads[fingerprint] != serialized:
                raise GateError(
                    "Conflicting rows share audit identity "
                    f"{fingerprint!r} at source index {source_index}"
                )
            continue
        seen_payloads[fingerprint] = serialized
        if not _target_tool(row):
            continue
        row["_research_audit_selection"] = {
            "source_index": source_index,
            "source_cohort": source_cohort,
            "example_id": str(row.get("example_id") or ""),
            "source_row_sha256": _sha256_bytes(serialized.encode("utf-8")),
            "target_tool": _target_tool(row),
            "state_class": _state_class(row),
            "history_length": _history_length(row),
            "serialized_characters": len(serialized),
        }
        candidates.append(row)
    if len(candidates) < limit:
        raise GateError(
            f"Only {len(candidates)} unique canonical tool rows are available; requested {limit}"
        )

    ranked = sorted(
        candidates,
        key=lambda row: (
            int(row["_research_audit_selection"]["history_length"]),
            int(row["_research_audit_selection"]["serialized_characters"]),
            str(row.get("example_id") or ""),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        key = _stable_json({k: v for k, v in row.items() if k != "_research_audit_selection"})
        if key not in selected_ids and len(selected) < limit:
            selected_ids.add(key)
            selected.append(row)

    # First cover every important protocol target that exists in the supplied
    # corpus, choosing its longest-history example.
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_tool[_target_tool(row)].append(row)
        by_class[_state_class(row)].append(row)
    for tool in _PRIORITY_TOOLS:
        if by_tool[tool]:
            add(by_tool[tool][0])
    for state_class in sorted(by_class):
        add(by_class[state_class][0])

    # Preserve a substantial long-history tail, then fill round-robin by tool
    # so frequent WLS/commit targets cannot monopolize the slice.
    for row in ranked[: max(8, limit // 3)]:
        add(row)
    tool_names = sorted(by_tool)
    depth = 0
    while len(selected) < limit:
        progressed = False
        for tool in tool_names:
            bucket = by_tool[tool]
            if depth < len(bucket):
                before = len(selected)
                add(bucket[depth])
                progressed = progressed or len(selected) > before
        if not progressed and depth >= max(len(bucket) for bucket in by_tool.values()):
            break
        depth += 1
    if len(selected) != limit:
        raise GateError(f"Representative selection produced {len(selected)}/{limit} rows")
    return selected


def select_audit_rows(
    d0_rows: Sequence[Mapping[str, Any]],
    rollback_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 32,
) -> list[dict[str, Any]]:
    """Select unique D0 rows plus one explicitly labeled D1 rollback canary."""

    if limit < 2:
        raise ValueError("processor audit requires at least two rows")
    rollback_candidates = []
    for source_index, source in enumerate(rollback_rows):
        if _target_tool(source) != "rollback_state":
            continue
        row = copy.deepcopy(dict(source))
        row["_research_audit_source_index"] = source_index
        rollback_candidates.append(row)
    if not rollback_candidates:
        raise GateError("rollback canary source contains no rollback_state target")
    selected = select_representative_rows(
        d0_rows,
        limit=limit - 1,
        source_cohort="d0",
    )
    selected.append(
        select_representative_rows(
            rollback_candidates,
            limit=1,
            source_cohort="d1_rollback_canary",
        )[0]
    )
    return selected


def _clean_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in row.items() if key != "_research_audit_selection"}


def _flatten_input_ids(encoded: Any) -> list[int]:
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise GateError("inference tokenization did not return input_ids")
    value = encoded["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise GateError("inference input_ids are not one integer sequence")
    if not value:
        raise GateError("inference input_ids are empty")
    return list(value)


def _aligned_token_fields(encoded: Any) -> dict[str, list[int]]:
    if not isinstance(encoded, Mapping):
        raise GateError("inference tokenization did not return a mapping")
    input_ids = _flatten_input_ids(encoded)
    aligned: dict[str, list[int]] = {}
    for key, raw in encoded.items():
        try:
            values = _flatten_input_ids({"input_ids": raw})
        except GateError:
            continue
        if len(values) == len(input_ids):
            aligned[str(key)] = values
    aligned.setdefault("input_ids", input_ids)
    aligned.setdefault("attention_mask", [1] * len(input_ids))
    return aligned


def audit_model_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: ResearchModelSpec,
    processor: Any,
    processor_loader: str,
    max_length: int,
    minimum_headroom_tokens: int,
) -> dict[str, Any]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if minimum_headroom_tokens <= 0:
        raise ValueError("minimum_headroom_tokens must be positive")
    canonical_tools = canonical_prompt_tool_schemas()
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    expected_suffix = (
        _UNIFIED_GENERATION_SUFFIX
        if spec.prompt_profile == PROMPT_PROFILE_NATIVE
        else _SMALL_GENERATION_SUFFIX
    )
    for index, selected in enumerate(rows):
        row = _clean_row(selected)
        label = str(row.get("example_id") or f"row[{index}]")
        row_failures: list[str] = []
        messages = row.get("messages")
        tools = row.get("tools")
        if tools != canonical_tools:
            row_failures.append("tool schemas differ from the research inference registry")
        if not isinstance(messages, list) or len(messages) != 3:
            row_failures.append("messages are not exactly system, user, assistant")
            messages = []
        elif [message.get("role") for message in messages if isinstance(message, Mapping)] != [
            "system",
            "user",
            "assistant",
        ]:
            row_failures.append("message roles are not exactly system, user, assistant")
        first_message = messages[0] if messages else None
        if not isinstance(first_message, Mapping) or first_message.get("role") != "system":
            row_failures.append("first message is not system")
        if (
            not isinstance(first_message, Mapping)
            or first_message.get("content") != CANONICAL_DAGGER_SYSTEM_PROMPT
        ):
            row_failures.append("system prompt differs from research inference")
        state = _state_payload(row)
        if not state:
            row_failures.append("user content is not the canonical serialized state payload")
        else:
            try:
                validate_policy_payload({"state": state})
            except (TypeError, ValueError) as exc:
                row_failures.append(f"observable state payload is invalid: {exc}")
            user_messages = [
                item
                for item in messages
                if isinstance(item, Mapping) and item.get("role") == "user"
            ]
            expected_user = json.dumps(
                {"state": state}, sort_keys=True, allow_nan=False
            )
            if not user_messages or user_messages[-1].get("content") != expected_user:
                row_failures.append("user state serialization differs from inference")
            expected_prompt_messages = [
                {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
                {"role": "user", "content": expected_user},
            ]
            if list(messages[:2]) != expected_prompt_messages:
                row_failures.append(
                    "prompt messages differ from the exact inference message objects"
                )
        try:
            example = prepare_example(row, processor, max_length=max_length, row_label=label)
            verify_assistant_only_mask(example)
        except (GateError, ValueError) as exc:
            row_failures.append(f"prepare:{type(exc).__name__}:{exc}")
            example = None

        if example is not None:
            inference_prompt = render_eval_text(
                processor,
                list(messages[:-1]),
                canonical_tools,
                enable_thinking=False,
                inject_empty_thought_channel=False,
            )
            if inference_prompt != example.rendered_prompt:
                row_failures.append("training and inference prompts differ")
            inference_encoding: dict[str, list[int]] = {}
            try:
                inference_encoding = _aligned_token_fields(
                    tokenize_rendered_text(processor, inference_prompt)
                )
                inference_ids = inference_encoding["input_ids"]
            except (GateError, TypeError, ValueError) as exc:
                inference_ids = []
                row_failures.append(f"inference tokenization failed: {exc}")
            if (
                inference_ids
                and not example.prompt_truncated
                and example.input_ids[: len(inference_ids)] != inference_ids
            ):
                row_failures.append("training and inference prompt token ids differ")
            if inference_ids and not example.prompt_truncated:
                training_encoding = {
                    "input_ids": example.input_ids[: len(inference_ids)],
                    "attention_mask": example.attention_mask[: len(inference_ids)],
                    **{
                        key: values[: len(inference_ids)]
                        for key, values in example.side_inputs.items()
                    },
                }
                if inference_encoding != training_encoding:
                    row_failures.append(
                        "training and inference aligned token fields differ: "
                        f"training={sorted(training_encoding)}, "
                        f"inference={sorted(inference_encoding)}"
                    )
            if not inference_prompt.endswith(expected_suffix):
                row_failures.append("generation prompt has the wrong model-specific suffix")
            if spec.prompt_profile == PROMPT_PROFILE_SMALL_FORCED:
                if not example.rendered_completion.startswith(FORCED_TOOL_PREFIX):
                    row_failures.append("E4B target does not start with the forced tool prefix")
                if not example.rendered_text.startswith(
                    example.rendered_prompt + FORCED_TOOL_PREFIX
                ):
                    row_failures.append("E4B forced conditioning is not a prefix of the SFT row")
                if example.empty_thought_injected:
                    row_failures.append("E4B unexpectedly injected an empty thought channel")
                try:
                    forced_ids = _flatten_input_ids(
                        encode_text(processor, FORCED_TOOL_PREFIX)
                    )
                except (GateError, TypeError, ValueError) as exc:
                    forced_ids = []
                    row_failures.append(f"E4B forced-prefix tokenization failed: {exc}")
                if forced_ids and len(forced_ids) >= MAX_NEW_TOKENS:
                    row_failures.append(
                        "E4B forced prefix leaves no generation budget: "
                        f"{len(forced_ids)} >= {MAX_NEW_TOKENS}"
                    )
                supervised_ids = [
                    token
                    for token, label_value in zip(example.input_ids, example.labels)
                    if label_value != -100
                ]
                if forced_ids and supervised_ids[: len(forced_ids)] != forced_ids:
                    row_failures.append(
                        "E4B inference forced-prefix token ids differ from supervision"
                    )
                supervised_start = next(
                    (
                        offset
                        for offset, label_value in enumerate(example.labels)
                        if label_value != -100
                    ),
                    len(example.labels),
                )
                forced_stop = supervised_start + len(forced_ids)
                if forced_ids and example.attention_mask[
                    supervised_start:forced_stop
                ] != [1] * len(forced_ids):
                    row_failures.append(
                        "E4B forced-prefix attention-mask suffix differs from inference"
                    )
                for side_name, side_values in example.side_inputs.items():
                    if forced_ids and side_values[
                        supervised_start:forced_stop
                    ] != [0] * len(forced_ids):
                        row_failures.append(
                            "E4B forced-prefix side-input suffix differs from inference: "
                            f"{side_name}"
                        )
            elif spec.prompt_profile == PROMPT_PROFILE_NATIVE:
                if not example.empty_thought_injected:
                    row_failures.append("12B did not align the processor-provided empty thought channel")
                # A native Unified tool response legitimately begins with the
                # same ``<|tool_call>call:`` marker used to condition E4B.  The
                # distinction is where it comes from: 12B's inference prompt
                # ends at the processor-provided thought channel and does not
                # manually append those tokens.  Reject conditioning on the
                # prompt surface, not a valid marker in the target response.
                if inference_prompt.endswith(FORCED_TOOL_PREFIX):
                    row_failures.append(
                        "12B inference prompt manually appends the E4B forced prefix"
                    )
            if example.prompt_truncated:
                row_failures.append("prompt was truncated")
            if example.target_truncated:
                row_failures.append("assistant target was truncated")
            if example.supervised_tokens <= 0:
                row_failures.append("no supervised assistant token remains")
            expected = example.expected_tool_call
            if expected is None or expected.name != _target_tool(row):
                row_failures.append("target tool-call round trip changed the canonical action")
            record = {
                "example_id": label,
                "target_tool": _target_tool(row),
                "state_class": _state_class(row),
                "history_length": _history_length(row),
                "original_tokens": example.original_length,
                "used_tokens": example.used_length,
                "supervised_tokens": example.supervised_tokens,
                "prompt_truncated": example.prompt_truncated,
                "target_truncated": example.target_truncated,
                "empty_thought_injected": example.empty_thought_injected,
                "prompt_suffix_ok": inference_prompt.endswith(expected_suffix),
                "inference_prompt_tokens": len(inference_ids),
                "inference_aligned_fields": sorted(inference_encoding),
                "failures": row_failures,
            }
        else:
            record = {
                "example_id": label,
                "target_tool": _target_tool(row),
                "state_class": _state_class(row),
                "history_length": _history_length(row),
                "failures": row_failures,
            }
        records.append(record)
        failures.extend(f"{label}:{failure}" for failure in row_failures)

    maximum = max(
        (int(record.get("original_tokens") or 0) for record in records), default=0
    )
    maximum_prompt = max(
        (int(record.get("inference_prompt_tokens") or 0) for record in records),
        default=0,
    )
    maximum_target = max(
        (int(record.get("supervised_tokens") or 0) for record in records),
        default=0,
    )
    headroom = max_length - maximum
    if headroom < minimum_headroom_tokens:
        failures.append(
            f"maximum rendered length leaves only {headroom} tokens of headroom; "
            f"required {minimum_headroom_tokens}"
        )
    if maximum_prompt > MAX_INPUT_TOKENS:
        failures.append(
            f"maximum prompt length {maximum_prompt} exceeds live inference input "
            f"limit {MAX_INPUT_TOKENS}; set RESEARCH_MAX_INPUT_TOKENS consistently"
        )
    if max_length > MAX_INPUT_TOKENS:
        failures.append(
            f"training max_length {max_length} exceeds live inference input limit "
            f"{MAX_INPUT_TOKENS}; configure the same reviewed envelope"
        )
    if maximum_target > MAX_NEW_TOKENS:
        failures.append(
            f"maximum supervised target {maximum_target} exceeds live generation "
            f"limit {MAX_NEW_TOKENS}"
        )
    return {
        "contract": AUDIT_CONTRACT,
        "passed": not failures,
        "model": {
            "key": spec.key,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "architecture": spec.architecture,
            "prompt_profile": spec.prompt_profile,
            "processor_loader": processor_loader,
        },
        "rows_requested": len(rows),
        "rows_prepared": sum("original_tokens" in record for record in records),
        "tool_round_trips": sum(
            "original_tokens" in record and not any("round trip" in failure for failure in record["failures"])
            for record in records
        ),
        "maximum_rendered_tokens": maximum,
        "maximum_inference_prompt_tokens": maximum_prompt,
        "maximum_supervised_target_tokens": maximum_target,
        "configured_max_length": max_length,
        "headroom_tokens": headroom,
        "minimum_headroom_tokens": minimum_headroom_tokens,
        "live_inference_max_input_tokens": MAX_INPUT_TOKENS,
        "live_inference_max_new_tokens": MAX_NEW_TOKENS,
        "failures": failures,
        "records": records,
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("transformers", "tokenizers", "torch", "peft", "trl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Audit pinned E4B/12B processor and prompt parity on one shared slice"
    )
    result.add_argument("--d0", required=True, type=Path)
    result.add_argument("--rollback-canary", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--model-choice", action="append", choices=("e4b", "12b")
    )
    result.add_argument("--rows", type=int, default=32)
    result.add_argument("--max-length", type=int, default=16384)
    result.add_argument("--minimum-headroom-tokens", type=int, default=1024)
    result.add_argument("--allow-download", action="store_true")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_outputs = [
        path
        for path in (
            output_dir / "processor_audit.json",
            output_dir / "processor_audit_e4b.json",
            output_dir / "processor_audit_12b.json",
            output_dir / "selection.json",
        )
        if path.exists()
    ]
    if stale_outputs:
        raise GateError(
            "Preserving an existing processor-audit run; use a new output directory: "
            + ", ".join(str(path) for path in stale_outputs)
        )
    d0_path = args.d0.expanduser().resolve(strict=True)
    rollback_path = args.rollback_canary.expanduser().resolve(strict=True)
    d0_rows = load_jsonl(d0_path)
    rollback_rows = load_jsonl(rollback_path)
    sources = [
        {
            "cohort": "d0",
            "path": str(d0_path),
            "rows": len(d0_rows),
            "sha256_recorded_only": _file_sha256(d0_path),
        },
        {
            "cohort": "d1_rollback_canary",
            "path": str(rollback_path),
            "rows": len(rollback_rows),
            "sha256_recorded_only": _file_sha256(rollback_path),
        },
    ]
    selected_source = select_audit_rows(d0_rows, rollback_rows, limit=args.rows)
    clean_source = [_clean_row(row) for row in selected_source]
    normalized, normalization = normalize_research_rows(
        clean_source, source_label="processor_audit_selection"
    )
    selected: list[dict[str, Any]] = []
    for source_row, prompt_row in zip(selected_source, normalized):
        prompt_row["_research_audit_selection"] = copy.deepcopy(
            source_row["_research_audit_selection"]
        )
        selected.append(prompt_row)
    _write_jsonl(output_dir / "selected_source_rows.jsonl", clean_source)
    _write_jsonl(
        output_dir / "selected_prompt_rows.jsonl",
        [_clean_row(row) for row in selected],
    )
    _write_json(
        output_dir / "selection.json",
        {
            "contract": AUDIT_CONTRACT,
            "sources": sources,
            "rows_available": len(d0_rows) + len(rollback_rows),
            "rows_selected": len(selected),
            "normalization": normalization,
            "selection": [row["_research_audit_selection"] for row in selected],
        },
    )
    choices = args.model_choice or [GEMMA4_E4B.key, GEMMA4_12B.key]
    reports: dict[str, dict[str, Any]] = {}
    for choice in choices:
        spec = get_research_model_spec(choice)
        processor, loader = load_exact_processor(
            spec.model_id,
            spec.revision,
            local_files_only=not args.allow_download,
            trust_remote_code=spec.trust_remote_code,
        )
        if loader != "AutoProcessor":
            raise GateError(
                f"{choice}: exact research path requires AutoProcessor, got {loader}"
            )
        report = audit_model_rows(
            selected,
            spec=spec,
            processor=processor,
            processor_loader=loader,
            max_length=args.max_length,
            minimum_headroom_tokens=args.minimum_headroom_tokens,
        )
        reports[choice] = report
        _write_json(output_dir / f"processor_audit_{choice}.json", report)
    summary = {
        "contract": AUDIT_CONTRACT,
        "passed": bool(reports) and all(report["passed"] for report in reports.values()),
        "sources": sources,
        "selected_rows": len(selected),
        "models": {
            choice: {
                "passed": report["passed"],
                "maximum_rendered_tokens": report["maximum_rendered_tokens"],
                "maximum_inference_prompt_tokens": report[
                    "maximum_inference_prompt_tokens"
                ],
                "headroom_tokens": report["headroom_tokens"],
                "failure_count": len(report["failures"]),
            }
            for choice, report in reports.items()
        },
        "environment": _environment(),
    }
    _write_json(output_dir / "processor_audit.json", summary)
    _write_json(output_dir / "environment.json", summary["environment"])
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
