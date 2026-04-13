import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = "data/sft_with_tools.jsonl"
DEFAULT_REPORT = "out_traces_balanced/preprocess_report.json"
DEFAULT_TOKENIZER = "unsloth/Gemma-4-26B-A4B-it"
DEFAULT_MAX_SEQ_LENGTH = 4096
RANDOM_SEED = 42
from trace_protocol import (
    BALANCED_SPLIT_COUNTS,
    BALANCED_TOTAL_PER_CLASS,
    ERROR_FAMILIES,
    canonical_tool_schemas,
    looks_like_json,
    maybe_parse_json_string,
    normalize_instruction_content,
    normalize_error_family,
    parse_json_text,
    prune_none,
    round_assistant_payload,
    round_tool_arguments,
    round_user_payload,
)

VALID_ERROR_FAMILIES = set(ERROR_FAMILIES)


def load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    errors = 0
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {line_no}: JSON error - {exc}")
                errors += 1
    if errors:
        print(f"  [warn] skipped {errors} malformed lines")
    return samples


def save_jsonl(samples: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_report(report: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in "{["


def round_numeric_values(value: Any, decimals: int) -> Any:
    if isinstance(value, float):
        rounded = round(value, decimals)
        return 0.0 if rounded == -0.0 else rounded
    if isinstance(value, list):
        return [round_numeric_values(item, decimals) for item in value]
    if isinstance(value, dict):
        return {key: round_numeric_values(item, decimals) for key, item in value.items()}
    return value


def canonicalize_json_text(text: str, decimals: int) -> str:
    payload = json.loads(text)
    payload = round_numeric_values(payload, decimals)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := prune_none(item)) is not None
        }
    if isinstance(value, list):
        return [prune_none(item) for item in value if item is not None]
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


def load_tools(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not args.include_tool_schemas:
        return None
    if args.tools_file:
        with open(args.tools_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("--tools-file must contain a JSON list of tool schemas.")
        return sanitize_tool_schemas(data)
    return canonical_tool_schemas()


def normalize_message(message: dict, decimals: int) -> dict:
    normalized = copy.deepcopy(message)
    role = normalized.get("role")

    if "content" in normalized:
        normalized["content"] = normalize_instruction_content(role, normalized.get("content"))
    content = normalized.get("content")
    if isinstance(content, str) and looks_like_json(content):
        payload = json.loads(content)
        if role == "user":
            payload = round_user_payload(payload)
        else:
            payload = round_assistant_payload(payload)
        normalized["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    tool_calls = normalized.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str) and looks_like_json(arguments):
                function["arguments"] = json.dumps(
                    round_tool_arguments(json.loads(arguments)),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            elif isinstance(arguments, (dict, list)):
                function["arguments"] = round_tool_arguments(arguments)

    return normalized


def normalize_sample(sample: dict, decimals: int) -> dict:
    normalized = copy.deepcopy(sample)
    messages = normalized.get("messages")
    if isinstance(messages, list):
        normalized["messages"] = [normalize_message(message, decimals) for message in messages]
    return normalized


def parse_json_text(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def extract_final_diagnosis(sample: dict) -> dict | None:
    for message in reversed(sample.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not looks_like_json(content):
            continue
        payload = parse_json_text(content)
        if isinstance(payload, dict) and "verdict" in payload:
            return payload
    return None


def extract_label(sample: dict) -> str | None:
    diagnosis = extract_final_diagnosis(sample)
    if not diagnosis:
        return None
    label = normalize_error_family(diagnosis.get("verdict", {}).get("error_family"))
    return label if label in VALID_ERROR_FAMILIES else None


def extract_tool_sequence(sample: dict) -> tuple[str, ...]:
    sequence: list[str] = []
    for message in sample.get("messages", []):
        for tool_call in message.get("tool_calls", []):
            name = tool_call.get("function", {}).get("name")
            if isinstance(name, str) and name:
                sequence.append(name)
    return tuple(sequence)


def extract_user_snapshot(sample: dict) -> dict[str, Any] | None:
    for message in sample.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not looks_like_json(content):
            continue
        payload = parse_json_text(content)
        if isinstance(payload, dict) and "z_obs" in payload:
            return payload
    return None


def validate_sample(sample: dict, index: int) -> list[str]:
    errors: list[str] = []
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return [f"sample {index}: missing messages list"]

    if messages[0].get("role") != "system":
        errors.append(f"sample {index}: first message is not system")

    if not any(message.get("role") == "user" for message in messages):
        errors.append(f"sample {index}: missing user message")

    diagnosis = extract_final_diagnosis(sample)
    if diagnosis is None:
        errors.append(f"sample {index}: missing final assistant diagnosis JSON")
    else:
        missing_keys = sorted({"verdict", "evidence", "suspect_location", "action", "summary"} - diagnosis.keys())
        if missing_keys:
            errors.append(f"sample {index}: diagnosis missing keys {missing_keys}")
        verdict = diagnosis.get("verdict")
        if not isinstance(verdict, dict):
            errors.append(f"sample {index}: diagnosis.verdict must be an object")
        else:
            if "global_metrics" not in diagnosis.get("evidence", {}):
                errors.append(f"sample {index}: diagnosis.evidence missing global_metrics")
            label = normalize_error_family(verdict.get("error_family"))
            if label not in VALID_ERROR_FAMILIES:
                errors.append(f"sample {index}: invalid error_family {verdict.get('error_family')!r}")
            if not isinstance(verdict.get("has_error"), bool):
                errors.append(f"sample {index}: verdict.has_error must be boolean")
            if verdict.get("confidence") is None:
                errors.append(f"sample {index}: verdict.confidence missing")

    for message_index, message in enumerate(messages, 1):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"sample {index}: message {message_index} has invalid role {role!r}")
            continue

        if role == "user":
            payload = parse_json_text(message.get("content"))
            if not isinstance(payload, dict):
                errors.append(f"sample {index}: user message {message_index} is not valid JSON")
            elif "case_path" not in payload and "z_obs" not in payload and "harmonic_measurements" not in payload and "three_phase_voltages" not in payload and "breaker_context" not in payload and "z_scans" not in payload:
                errors.append(f"sample {index}: user message {message_index} is missing canonical payload fields")

        if role == "assistant" and "tool_calls" in message:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                errors.append(f"sample {index}: assistant message {message_index} has empty tool_calls")
            else:
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        errors.append(f"sample {index}: assistant message {message_index} missing tool name")
                    arguments = function.get("arguments")
                    arguments_ok = False
                    if isinstance(arguments, str):
                        arguments_ok = parse_json_text(arguments) is not None
                    elif isinstance(arguments, (dict, list)):
                        arguments_ok = True
                    if not arguments_ok:
                        errors.append(
                            f"sample {index}: assistant message {message_index} has invalid tool arguments"
                        )

        if role == "tool":
            payload = parse_json_text(message.get("content"))
            if payload is None:
                errors.append(f"sample {index}: tool message {message_index} is not valid JSON")
            if not message.get("name"):
                errors.append(f"sample {index}: tool message {message_index} missing tool name")

    return errors


def build_dedupe_key(sample: dict, mode: str) -> str | None:
    if mode == "none":
        return None

    if mode == "user_snapshot":
        snapshot = extract_user_snapshot(sample)
        if snapshot is None:
            return None
        payload = {"case_path": snapshot.get("case_path") or snapshot.get("case"), "z_obs": snapshot.get("z_obs")}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    if mode == "full_trace":
        return json.dumps(sample, sort_keys=True, separators=(",", ":"))

    raise ValueError(f"Unsupported dedupe mode: {mode}")


def deduplicate_samples(samples: list[dict], mode: str) -> tuple[list[dict], int]:
    if mode == "none":
        return samples, 0

    seen: set[str] = set()
    deduped: list[dict] = []
    duplicates = 0

    for sample in samples:
        key = build_dedupe_key(sample, mode)
        if key is None:
            deduped.append(sample)
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(sample)

    return deduped, duplicates


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    exact = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda idx: (exact[idx] - counts[idx], exact[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def distribute_selected_group_exact(
    samples: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        buckets[sample_priority_tuple(sample)[0]].append(sample)

    targets = {
        "train": BALANCED_SPLIT_COUNTS["train"],
        "valid": BALANCED_SPLIT_COUNTS["valid"],
        "test": BALANCED_SPLIT_COUNTS["test"],
    }
    assigned = {"train": [], "valid": [], "test": []}
    remaining = dict(targets)

    def split_rank(name: str) -> tuple[float, int, int]:
        order = {"train": 0, "valid": 1, "test": 2}
        return (remaining[name] / targets[name], remaining[name], -order[name])

    for priority in sorted(buckets.keys(), reverse=True):
        group = list(buckets[priority])
        rng.shuffle(group)
        for sample in group:
            eligible = [name for name, count in remaining.items() if count > 0]
            if not eligible:
                raise ValueError("Split allocation overflow while distributing selected samples.")
            split_name = max(eligible, key=split_rank)
            assigned[split_name].append(sample)
            remaining[split_name] -= 1

    if any(count != 0 for count in remaining.values()):
        raise ValueError(f"Split allocation underflow: {remaining}")
    return assigned["train"], assigned["valid"], assigned["test"]


def stratified_split(
    samples: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)

    for sample in samples:
        label = extract_label(sample) or "__missing__"
        tool_sequence = ">".join(extract_tool_sequence(sample)) or "no_tools"
        stratum = f"{label}::{tool_sequence}"
        by_stratum[stratum].append(sample)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for group in by_stratum.values():
        rng.shuffle(group)
        n_train, n_val, _ = split_counts(len(group), train_ratio, val_ratio, test_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def exact_balanced_split(samples: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        label = extract_label(sample)
        if label is not None:
            by_label[label].append(sample)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    selected_counts: dict[str, int] = {}

    for label in ERROR_FAMILIES:
        group = list(by_label.get(label, []))
        if len(group) < BALANCED_TOTAL_PER_CLASS:
            raise ValueError(
                f"Need at least {BALANCED_TOTAL_PER_CLASS} accepted samples for {label}, found {len(group)}"
            )
        rng.shuffle(group)
        group.sort(key=sample_priority_tuple, reverse=True)
        selected = group[:BALANCED_TOTAL_PER_CLASS]
        train_group, val_group, test_group = distribute_selected_group_exact(selected, seed + len(label))
        train.extend(train_group)
        val.extend(val_group)
        test.extend(test_group)
        selected_counts[label] = len(selected)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return (
        train,
        val,
        test,
        {
            "mode": "exact_balanced",
            "selected_per_class": selected_counts,
            "split_counts_per_class": dict(BALANCED_SPLIT_COUNTS),
        },
    )


def balance_training_set(samples: list[dict], mode: str, seed: int) -> tuple[list[dict], dict[str, Any]]:
    if mode == "none":
        return samples, {"mode": "none"}

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_label[extract_label(sample) or "__missing__"].append(sample)

    if not by_label:
        return samples, {"mode": mode, "applied": False}

    counts = {label: len(group) for label, group in by_label.items()}
    target = max(counts.values()) if mode == "upsample" else min(counts.values())
    balanced: list[dict] = []

    for label, group in sorted(by_label.items()):
        if mode == "upsample":
            if len(group) < target:
                balanced.extend(group)
                balanced.extend(rng.choices(group, k=target - len(group)))
            else:
                balanced.extend(group)
        elif mode == "downsample":
            balanced.extend(rng.sample(group, target))
        else:
            raise ValueError(f"Unsupported balance mode: {mode}")

    rng.shuffle(balanced)
    return balanced, {
        "mode": mode,
        "applied": True,
        "target_per_class": target,
        "before": counts,
        "after": dict(Counter(extract_label(sample) for sample in balanced)),
    }


def label_distribution(samples: list[dict]) -> dict[str, int]:
    return dict(Counter(extract_label(sample) for sample in samples))


def final_payload(sample: dict) -> dict[str, Any] | None:
    diagnosis = extract_final_diagnosis(sample)
    return diagnosis if isinstance(diagnosis, dict) else None


def verification_summary(sample: dict) -> dict[str, Any] | None:
    payload = final_payload(sample)
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if not isinstance(action, dict):
        return None
    summary = action.get("verification_summary")
    return summary if isinstance(summary, dict) else None


def sample_priority_tuple(sample: dict) -> tuple[int, float, str]:
    summary = verification_summary(sample) or {}
    resolved = bool(summary.get("post_action_resolved"))
    improved = bool(summary.get("post_action_improved"))
    ratio = summary.get("post_action_global_residual_ratio")
    try:
        ratio_value = float(ratio) if ratio is not None else math.inf
    except (TypeError, ValueError):
        ratio_value = math.inf
    payload = final_payload(sample) or {}
    summary_text = payload.get("summary") if isinstance(payload.get("summary"), str) else ""
    return (2 if resolved else 1 if improved else 0, -ratio_value, summary_text)


def dataset_qa(samples: list[dict]) -> dict[str, Any]:
    thresholds_null = 0
    ratios_null = 0
    no_error_ratio_violations = 0
    no_error_nonempty_evidence = 0
    post_action_resolved: Counter[str] = Counter()
    post_action_improved: Counter[str] = Counter()
    evidence_lengths: Counter[str] = Counter()
    for sample in samples:
        payload = final_payload(sample)
        if not payload:
            continue
        label = normalize_error_family(payload.get("verdict", {}).get("error_family"))
        evidence = payload.get("evidence", {})
        gm = evidence.get("global_metrics", {})
        if gm.get("global_residual_threshold") is None:
            thresholds_null += 1
        if gm.get("global_residual_ratio") is None:
            ratios_null += 1
        verification = payload.get("action", {}).get("verification_summary") or {}
        if verification.get("post_action_resolved") is True:
            post_action_resolved[label] += 1
        if verification.get("post_action_improved") is True:
            post_action_improved[label] += 1
        top_residuals = evidence.get("top_residuals", []) or []
        top_lagrange = evidence.get("top_lagrange", []) or []
        evidence_lengths[f"top_residuals:{len(top_residuals)}"] += 1
        evidence_lengths[f"top_lagrange:{len(top_lagrange)}"] += 1
        if label == "no_error":
            ratio = gm.get("global_residual_ratio")
            if ratio is None or float(ratio) >= 0.9:
                no_error_ratio_violations += 1
            if top_residuals or top_lagrange:
                no_error_nonempty_evidence += 1

    return {
        "count": len(samples),
        "thresholds_null": thresholds_null,
        "ratios_null": ratios_null,
        "no_error_ratio_violations": no_error_ratio_violations,
        "no_error_nonempty_evidence": no_error_nonempty_evidence,
        "post_action_resolved_by_family": dict(post_action_resolved),
        "post_action_improved_by_family": dict(post_action_improved),
        "evidence_lengths": dict(evidence_lengths),
    }


def print_split_stats(name: str, samples: list[dict]) -> None:
    distribution = label_distribution(samples)
    total = len(samples)
    print(f"\n  {name}: {total} samples")
    for label in sorted(distribution):
        count = distribution[label]
        pct = (count / total * 100) if total else 0.0
        print(f"    {label:<25} {count:>5}  ({pct:5.1f}%)")


def normalize_messages_for_gpt_oss(messages: list[dict[str, Any]], decimals: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        msg = prune_none(copy.deepcopy(raw_message))
        role = msg.get("role")
        if role is None:
            continue

        if "content" in msg and not isinstance(msg["content"], str):
            payload = round_user_payload(msg["content"]) if role == "user" else round_assistant_payload(msg["content"])
            msg["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                fixed_calls = []
                for tool_call in tool_calls:
                    tc = prune_none(copy.deepcopy(tool_call))
                    function = tc.get("function")
                    if isinstance(function, dict):
                        arguments = maybe_parse_json_string(function.get("arguments"))
                        if isinstance(arguments, (dict, list)):
                            arguments = round_tool_arguments(arguments)
                        if arguments is None:
                            arguments = {}
                        function["arguments"] = arguments
                        tc["function"] = function
                    fixed_calls.append(tc)
                msg["tool_calls"] = fixed_calls
                msg.pop("content", None)
            else:
                msg.pop("tool_calls", None)
                msg.setdefault("content", "")
        elif role == "tool":
            msg.setdefault("content", "")
            if not isinstance(msg["content"], str):
                msg["content"] = json.dumps(
                    round_assistant_payload(msg["content"]),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        else:
            msg.setdefault("content", "")

        normalized.append(msg)
    return normalized


def render_gpt_oss_text(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def count_tool_argument_formats(samples: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        for message in sample.get("messages", []):
            for tool_call in message.get("tool_calls", []):
                arguments = tool_call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    counts["string"] += 1
                elif isinstance(arguments, (dict, list)):
                    counts["structured"] += 1
                elif arguments is None:
                    counts["missing"] += 1
                else:
                    counts[type(arguments).__name__] += 1
    return dict(counts)


def percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, math.ceil(len(sorted_values) * pct) - 1)
    return sorted_values[index]


def audit_token_lengths(
    samples: list[dict],
    tokenizer_name: str,
    max_seq_length: int,
    tools: list[dict[str, Any]] | None,
    decimals: int,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, "transformers is not installed; skipping token audit"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - depends on local HF cache/network
        return None, f"failed to load tokenizer {tokenizer_name!r}: {exc}"

    lengths: list[int] = []
    for sample in samples:
        try:
            text = render_gpt_oss_text(
                tokenizer,
                normalize_messages_for_gpt_oss(sample["messages"], decimals),
                tools,
            )
        except Exception as exc:
            return None, f"failed to render the chat template for token audit: {exc}"
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(token_count)

    lengths.sort()
    if not lengths:
        return {
            "tokenizer": tokenizer_name,
            "max_seq_length": max_seq_length,
            "count": 0,
        }, None

    over_limit = sum(length > max_seq_length for length in lengths)
    return {
        "tokenizer": tokenizer_name,
        "max_seq_length": max_seq_length,
        "count": len(lengths),
        "min": lengths[0],
        "mean": round(mean(lengths), 2),
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "max": lengths[-1],
        "over_limit": over_limit,
        "over_limit_pct": round(over_limit / len(lengths) * 100, 2),
        "required_to_avoid_truncation": lengths[-1],
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PSSE agent dataset for SFT")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Round floats inside embedded JSON payloads",
    )
    parser.add_argument(
        "--exact-balanced",
        action="store_true",
        default=True,
        help="Select exactly 500 samples per class and split them into 400/50/50 train/valid/test.",
    )
    parser.add_argument(
        "--no-exact-balanced",
        dest="exact_balanced",
        action="store_false",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=("none", "user_snapshot", "full_trace"),
        default="user_snapshot",
        help="Deduplicate samples before splitting",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer used to audit sequence lengths",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Target max sequence length for token audit",
    )
    parser.add_argument(
        "--skip-token-audit",
        action="store_true",
        help="Skip tokenizer-based sequence length audit",
    )
    parser.add_argument(
        "--include-tool-schemas",
        action="store_true",
        default=True,
        help="Include power-tool schemas when rendering chat templates for token audit",
    )
    parser.add_argument(
        "--no-include-tool-schemas",
        dest="include_tool_schemas",
        action="store_false",
    )
    parser.add_argument(
        "--tools-file",
        default="",
        help="Optional JSON file with tool schemas used for token audit",
    )
    parser.add_argument("--out-train", default="out_traces_balanced/sft_traces.train.jsonl")
    parser.add_argument("--out-val", default="out_traces_balanced/sft_traces.valid.jsonl")
    parser.add_argument("--out-test", default="out_traces_balanced/sft_traces.test.jsonl")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Write preprocessing report JSON here")
    args = parser.parse_args()

    if args.round_decimals < 0:
        raise ValueError("--round-decimals must be non-negative")

    tools = load_tools(args)

    print("=" * 60)
    print("PSSE Agent Dataset Preprocessing")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"\nLoading: {args.input}")
    raw_samples = load_jsonl(args.input)
    print(f"  Loaded {len(raw_samples)} samples")

    print(f"\nNormalizing embedded JSON payloads (round_decimals={args.round_decimals})...")
    normalized_samples = [normalize_sample(sample, args.round_decimals) for sample in raw_samples]

    print("\nValidating samples...")
    validation_errors: list[str] = []
    valid_samples: list[dict] = []
    for index, sample in enumerate(normalized_samples, 1):
        errors = validate_sample(sample, index)
        if errors:
            validation_errors.extend(errors)
            continue
        valid_samples.append(sample)

    if validation_errors:
        print(f"  [warn] dropped {len(normalized_samples) - len(valid_samples)} invalid samples")
        for error in validation_errors[:10]:
            print(f"  [warn] {error}")
        if len(validation_errors) > 10:
            print(f"  [warn] ... and {len(validation_errors) - 10} more validation issues")
    else:
        print("  Validation passed with no dropped samples")

    tool_argument_formats = count_tool_argument_formats(valid_samples)
    print(f"  Tool argument formats: {tool_argument_formats or {'none': 0}}")

    print(f"\nDeduplicating ({args.dedupe_by})...")
    deduped_samples, duplicate_count = deduplicate_samples(valid_samples, args.dedupe_by)
    print(f"  Removed {duplicate_count} duplicates")

    labels = [extract_label(sample) for sample in deduped_samples]
    unlabeled = sum(label is None for label in labels)
    if unlabeled:
        print(f"  [warn] dropping {unlabeled} unlabeled samples")
        deduped_samples = [sample for sample in deduped_samples if extract_label(sample) is not None]

    print(f"  Retained {len(deduped_samples)} labeled samples")
    print(f"  Class distribution: {label_distribution(deduped_samples)}")

    if args.exact_balanced:
        print(
            "\nSelecting exact balanced splits: "
            f"{BALANCED_SPLIT_COUNTS['train']}/{BALANCED_SPLIT_COUNTS['valid']}/{BALANCED_SPLIT_COUNTS['test']} "
            f"per class (seed={args.seed})"
        )
        train, val, test, split_report = exact_balanced_split(deduped_samples, args.seed)
    else:
        print("\nUsing fallback stratified ratio split (seed=%s)" % args.seed)
        train, val, test = stratified_split(
            deduped_samples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=args.seed,
        )
        split_report = {"mode": "fallback_stratified"}

    print_split_stats("Train", train)
    print_split_stats("Val", val)
    print_split_stats("Test", test)
    if args.exact_balanced:
        for split_name, split_samples in (("train", train), ("valid", val), ("test", test)):
            distribution = label_distribution(split_samples)
            expected = BALANCED_SPLIT_COUNTS[split_name]
            for label in ERROR_FAMILIES:
                if distribution.get(label, 0) != expected:
                    raise ValueError(
                        f"Exact balance failed for {split_name}/{label}: expected {expected}, got {distribution.get(label, 0)}"
                    )

    qa_report = {
        "all": dataset_qa(deduped_samples),
        "train": dataset_qa(train),
        "val": dataset_qa(val),
        "test": dataset_qa(test),
    }
    if qa_report["all"]["thresholds_null"] or qa_report["all"]["ratios_null"]:
        raise ValueError("Dataset QA failed: null thresholds or ratios remain in accepted samples.")
    if qa_report["all"]["no_error_ratio_violations"]:
        raise ValueError("Dataset QA failed: borderline no_error traces remain in accepted samples.")
    if qa_report["all"]["no_error_nonempty_evidence"]:
        raise ValueError("Dataset QA failed: accepted no_error traces still contain significant evidence lists.")

    token_audit: dict[str, Any] = {}
    if args.skip_token_audit:
        print("\nSkipping token audit")
    else:
        schema_mode = "with tool schemas" if tools is not None else "without tool schemas"
        print(f"\nAuditing chat-template token lengths with {args.tokenizer_name!r} ({schema_mode})...")
        for split_name, split_samples in (("train", train), ("val", val), ("test", test), ("all", deduped_samples)):
            audit, warning = audit_token_lengths(
                split_samples,
                args.tokenizer_name,
                args.max_seq_length,
                tools,
                args.round_decimals,
            )
            if warning:
                print(f"  [warn] {warning}")
                token_audit["warning"] = warning
                break
            token_audit[split_name] = audit
            print(
                f"  {split_name:<5} count={audit['count']:<4} mean={audit['mean']:<8}"
                f" p95={audit['p95']:<5} max={audit['max']:<5}"
                f" over_limit={audit['over_limit']}"
            )

        overall = token_audit.get("all")
        if overall and overall["over_limit"]:
            print(
                "  [warn] examples exceed max_seq_length; "
                f"{overall['over_limit']} / {overall['count']} would truncate at {args.max_seq_length}"
            )

    print("\nSaving splits...")
    save_jsonl(train, args.out_train)
    save_jsonl(val, args.out_val)
    save_jsonl(test, args.out_test)
    print(f"  {args.out_train}  ({len(train)} samples)")
    print(f"  {args.out_val}    ({len(val)} samples)")
    print(f"  {args.out_test}   ({len(test)} samples)")

    report = {
        "input": args.input,
        "config": {
            "seed": args.seed,
            "round_decimals": args.round_decimals,
            "exact_balanced": args.exact_balanced,
            "dedupe_by": args.dedupe_by,
            "tokenizer_name": None if args.skip_token_audit else args.tokenizer_name,
            "max_seq_length": None if args.skip_token_audit else args.max_seq_length,
            "include_tool_schemas": args.include_tool_schemas,
            "tools_file": args.tools_file or None,
        },
        "counts": {
            "loaded": len(raw_samples),
            "valid": len(valid_samples),
            "duplicates_removed": duplicate_count,
            "retained": len(deduped_samples),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "label_distribution": {
            "all": label_distribution(deduped_samples),
            "train": label_distribution(train),
            "val": label_distribution(val),
            "test": label_distribution(test),
        },
        "tool_sequences": dict(
            Counter(">".join(extract_tool_sequence(sample)) for sample in deduped_samples)
        ),
        "tool_argument_formats": tool_argument_formats,
        "split_strategy": split_report,
        "validation": {
            "issues": len(validation_errors),
            "sample_errors": validation_errors[:25],
        },
        "dataset_qa": qa_report,
        "token_audit": token_audit,
    }
    save_report(report, args.report)
    print(f"  {args.report}")

    print("\n" + "=" * 60)
    print("Done. Dataset is normalized, validated, and split for SFT.")
    print("=" * 60)


if __name__ == "__main__":
    main()
