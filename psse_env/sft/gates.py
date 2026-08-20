"""Tokenizer, schema, masking, and grouped-pilot go/no-go gates.

Rendering is intentionally strict.  There is no hand-written Gemma template
fallback: production approval must use ``apply_chat_template`` from the exact,
pinned processor/tokenizer that training will use.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_TYPE_INPUT_NAMES = ("token_type_ids", "mm_token_type_ids")

# Deliberately duplicated as an ingestion contract instead of importing
# recovery_probes here: recovery_probes imports the release factory, which in
# turn imports this module and the training module.  Keeping this small fixed
# vocabulary local avoids a module cycle while still failing closed on drift.
_ROUND1_RECOVERY_PROBE_CONTRACT = "dagger1_observable_recovery_probe_v1"
_ROUND1_RECOVERY_PROBE_SOURCE = "observable_recovery_probe"
_ROUND1_RECOVERY_PROBE_ROLE = "auxiliary_training"
_ROUND1_RECOVERY_PROBE_STRATA = frozenset(
    {
        "post_failure_no_candidate",
        "unsupported_correction_recovery",
    }
)


def processor_token_type_input_names(processor: Any) -> tuple[str, ...]:
    """Return token-type side inputs advertised by a processor or tokenizer."""
    discovered: set[str] = set()
    for candidate in (processor, getattr(processor, "tokenizer", None)):
        if candidate is None:
            continue
        names = getattr(candidate, "model_input_names", None)
        if isinstance(names, (list, tuple)):
            discovered.update(str(name) for name in names)
    return tuple(name for name in TOKEN_TYPE_INPUT_NAMES if name in discovered)


class GateError(RuntimeError):
    """A required SFT safety gate could not be proven."""


class TargetTruncationError(GateError):
    def __init__(self, message: str, *, original_length: int, target_length: int) -> None:
        super().__init__(message)
        self.original_length = original_length
        self.target_length = target_length


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class PreparedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    completion_mask: list[int]
    original_length: int
    used_length: int
    supervised_tokens: int
    prompt_truncated: bool
    target_truncated: bool
    empty_thought_injected: bool
    target_kind: str
    expected_tool_call: ParsedToolCall | None
    rendered_text: str = field(repr=False)
    rendered_prompt: str = field(repr=False)
    rendered_completion: str = field(repr=False)
    side_inputs: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def model_record(self) -> dict[str, list[int]]:
        record: dict[str, list[int]] = {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
            "completion_mask": list(self.completion_mask),
        }
        record.update({key: list(value) for key, value in self.side_inputs.items()})
        return record


@dataclass(frozen=True)
class LengthAudit:
    rows_seen: int
    rows_prepared: int
    p50: int
    p95: int
    p99: int
    maximum: int
    used_p50: int
    used_p95: int
    used_p99: int
    used_maximum: int
    prompt_truncated_rows: int
    target_truncated_rows: int
    zero_supervision_rows: int
    empty_thought_injected_rows: int
    supervised_token_p50: int
    supervised_token_p95: int
    supervised_token_p99: int
    supervised_token_maximum: int


@dataclass
class DatasetGateReport:
    passed: bool
    failures: list[str]
    warnings: list[str]
    length_audit: LengthAudit
    tool_call_rows: int
    tool_round_trips: int
    action_distribution: dict[str, int]
    class_distribution: dict[str, int]
    prepared: list[PreparedExample] = field(default_factory=list, repr=False)

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "length_audit": asdict(self.length_audit),
            "tool_call_rows": self.tool_call_rows,
            "tool_round_trips": self.tool_round_trips,
            "action_distribution": dict(self.action_distribution),
            "class_distribution": dict(self.class_distribution),
        }
        if include_records:
            payload["prepared_records"] = [record.model_record() for record in self.prepared]
        return payload


@dataclass(frozen=True)
class GroupedPilotReport:
    passed: bool
    failures: tuple[str, ...]
    total_rows: int
    split_rows: dict[str, int]
    split_group_counts: dict[str, int]
    overlapping_groups: dict[str, tuple[str, ...]]
    action_distribution: dict[str, dict[str, int]]
    class_distribution: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise GateError(f"Unable to open dataset {source}: {type(exc).__name__}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise GateError(
                    f"Expected a JSON object at {source}:{line_number}, got {type(value).__name__}."
                )
            rows.append(value)
    if not rows:
        raise GateError(f"Dataset is empty: {source}")
    return rows


def _schema_name(schema: Mapping[str, Any]) -> str | None:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _check_schema_node(schema: Any, *, path: str) -> None:
    if not isinstance(schema, dict):
        raise GateError(f"{path} must be a JSON Schema object.")
    declared_type = schema.get("type")
    valid_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if declared_type is not None:
        # JSON Schema allows a single type name or a union list of them
        # (e.g. ["string", "null"] for nullable arguments).
        if isinstance(declared_type, list):
            if (
                not declared_type
                or len(set(declared_type)) != len(declared_type)
                or any(member not in valid_types for member in declared_type)
            ):
                raise GateError(f"{path}.type is unsupported or invalid: {declared_type!r}.")
        elif declared_type not in valid_types:
            raise GateError(f"{path}.type is unsupported or invalid: {declared_type!r}.")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise GateError(f"{path}.enum must be a non-empty list.")
    numeric_types = (
        {declared_type}
        if isinstance(declared_type, str)
        else set(declared_type or [])
    )
    for keyword in ("minimum", "maximum"):
        if keyword not in schema:
            continue
        bound = schema[keyword]
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
        ):
            raise GateError(f"{path}.{keyword} must be a finite JSON number.")
        if declared_type is not None and not numeric_types & {"integer", "number"}:
            raise GateError(
                f"{path}.{keyword} requires an integer or number schema type."
            )
    if (
        "minimum" in schema
        and "maximum" in schema
        and schema["minimum"] > schema["maximum"]
    ):
        raise GateError(f"{path}.minimum must not exceed {path}.maximum.")
    if declared_type == "object" or "properties" in schema or "required" in schema:
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise GateError(f"{path}.properties must be an object.")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            raise GateError(f"{path}.required must be a string list.")
        unknown_required = set(required) - set(properties)
        if unknown_required:
            raise GateError(f"{path}.required references missing properties: {sorted(unknown_required)}.")
        for key, child in properties.items():
            _check_schema_node(child, path=f"{path}.properties.{key}")
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, (bool, dict)):
            raise GateError(f"{path}.additionalProperties must be boolean or a schema.")
        if isinstance(additional, dict):
            _check_schema_node(additional, path=f"{path}.additionalProperties")
    if declared_type == "array" and "items" in schema:
        _check_schema_node(schema["items"], path=f"{path}.items")


def _is_json_type(value: Any, declared_type: str) -> bool:
    if declared_type == "object":
        return isinstance(value, dict)
    if declared_type == "array":
        return isinstance(value, list)
    if declared_type == "string":
        return isinstance(value, str)
    if declared_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if declared_type == "boolean":
        return isinstance(value, bool)
    if declared_type == "null":
        return value is None
    return False


def _validate_json_instance(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    declared_type = schema.get("type")
    if isinstance(declared_type, str) and not _is_json_type(value, declared_type):
        raise GateError(f"{path} must have JSON type {declared_type}, got {type(value).__name__}.")
    if isinstance(declared_type, list) and not any(
        _is_json_type(value, member) for member in declared_type
    ):
        raise GateError(
            f"{path} must have one of JSON types {declared_type!r}, got {type(value).__name__}."
        )
    if "enum" in schema and value not in schema["enum"]:
        raise GateError(f"{path} must be one of {schema['enum']!r}, got {value!r}.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise GateError(
                f"{path} must be >= {schema['minimum']!r}, got {value!r}."
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise GateError(
                f"{path} must be <= {schema['maximum']!r}, got {value!r}."
            )
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise GateError(f"{path} is missing required arguments: {missing}.")
        # Tool-call supervision is fail-closed.  JSON Schema ordinarily treats
        # an omitted ``additionalProperties`` keyword as ``true``, but that
        # would silently admit misspelled or stale assistant arguments that
        # have no declared tool semantics.  A schema may still opt in to a
        # free-form object explicitly with ``additionalProperties: true`` or
        # constrain dynamic keys with an additional-properties schema.
        additional = schema.get("additionalProperties", False)
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is None:
                if additional is False:
                    raise GateError(f"{path} contains unsupported argument {key!r}.")
                child_schema = additional if isinstance(additional, dict) else None
            if isinstance(child_schema, Mapping):
                _validate_json_instance(child, child_schema, path=f"{path}.{key}")
    elif isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, child in enumerate(value):
            _validate_json_instance(child, schema["items"], path=f"{path}[{index}]")


def validate_tool_schemas(tools: Any, *, row_label: str) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise GateError(f"{row_label}: row-level tools must be a non-empty JSON-schema list.")
    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_schema in enumerate(tools):
        if not isinstance(raw_schema, dict):
            raise GateError(f"{row_label}: tools[{index}] must be an object.")
        if raw_schema.get("type") != "function":
            raise GateError(f"{row_label}: tools[{index}].type must equal 'function'.")
        function = raw_schema.get("function")
        if not isinstance(function, dict):
            raise GateError(f"{row_label}: tools[{index}].function must be an object.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise GateError(f"{row_label}: tools[{index}] has no function name.")
        if name in names:
            raise GateError(f"{row_label}: duplicate tool schema name {name!r}.")
        names.add(name)
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise GateError(
                f"{row_label}: schema {name!r} must have an object-valued JSON Schema parameters field."
            )
        _check_schema_node(parameters, path=f"{row_label}.tools[{index}].function.parameters")
        try:
            json.dumps(raw_schema, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise GateError(f"{row_label}: schema {name!r} is not strict JSON: {exc}") from exc
        validated.append(copy.deepcopy(raw_schema))
    return validated


def validate_current_tool_registry(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require every release row to carry the exact current protocol registry."""
    from psse_env.dagger.dataset_builder import TOOL_JSON_SCHEMAS
    from psse_env.dagger.protocol_bridge import unified_tool_schemas

    expected = {
        "controller": TOOL_JSON_SCHEMAS,
        "canonical": unified_tool_schemas(),
    }
    expected_json = {
        protocol: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for protocol, value in expected.items()
    }
    failures: list[str] = []
    seen_mismatches: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        row_label = str(row.get("example_id") or f"row[{index}]")
        metadata = row.get("metadata")
        protocol = metadata.get("protocol") if isinstance(metadata, Mapping) else None
        if protocol not in expected:
            failures.append(
                f"{row_label}: release row must declare metadata.protocol as canonical or controller."
            )
            continue
        tools = row.get("tools")
        try:
            actual_json = json.dumps(
                tools, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError):
            continue  # The ordinary schema validator reports the malformed row.
        if actual_json == expected_json[protocol]:
            continue
        actual_names = {
            item.get("function", {}).get("name")
            for item in tools or []
            if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
        }
        expected_names = {
            item["function"]["name"] for item in expected[protocol]
        }
        signature = (protocol, hashlib.sha256(actual_json.encode("utf-8")).hexdigest())
        if signature in seen_mismatches:
            continue
        seen_mismatches.add(signature)
        failures.append(
            f"{row_label}: tool registry does not match current {protocol} registry; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}, "
            f"expected_count={len(expected_names)}, actual_count={len(actual_names)}."
        )
    return failures


def validate_messages(
    messages: Any,
    *,
    tools: Sequence[Mapping[str, Any]],
    row_label: str,
) -> list[dict[str, Any]]:
    """Validate model-visible messages and every assistant tool call strictly."""
    if not isinstance(messages, list) or not messages:
        raise GateError(f"{row_label}: messages must be a non-empty list.")
    normalized: list[dict[str, Any]] = []
    known_tools = {_schema_name(schema): schema["function"]["parameters"] for schema in tools}
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, dict):
            raise GateError(f"{row_label}: messages[{message_index}] must be an object.")
        role = raw_message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise GateError(f"{row_label}: messages[{message_index}] has invalid role {role!r}.")
        message = copy.deepcopy(raw_message)
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
                raise GateError(
                    f"{row_label}: messages[{message_index}].tool_calls must be a non-empty assistant list."
                )
            for call_index, call in enumerate(tool_calls):
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise GateError(
                        f"{row_label}: tool_calls[{call_index}] must be a type='function' object."
                    )
                function = call.get("function")
                if not isinstance(function, dict):
                    raise GateError(f"{row_label}: tool_calls[{call_index}].function must be an object.")
                name = function.get("name")
                if name not in known_tools:
                    raise GateError(f"{row_label}: target tool {name!r} has no row-level schema.")
                arguments = function.get("arguments")
                if not isinstance(arguments, dict):
                    raise GateError(
                        f"{row_label}: assistant function.arguments must be a dictionary, "
                        f"not {type(arguments).__name__}."
                    )
                try:
                    json.dumps(arguments, sort_keys=True, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise GateError(f"{row_label}: tool arguments are not strict JSON: {exc}") from exc
                _validate_json_instance(
                    arguments,
                    known_tools[name],
                    path=f"{row_label}.messages[{message_index}].tool_calls[{call_index}].function.arguments",
                )
        normalized.append(message)
    if normalized[-1].get("role") != "assistant":
        raise GateError(f"{row_label}: the final message must be the assistant training target.")
    final = normalized[-1]
    if not final.get("tool_calls") and not (
        isinstance(final.get("content"), str) and final["content"].strip()
    ):
        raise GateError(f"{row_label}: zero supervised semantic assistant target (no tool call or content).")
    return normalized


def _flatten_encoding(value: Any, *, field_name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise GateError(f"Expected one tokenized row for {field_name}, got batch size {len(value)}.")
        value = list(value[0])
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise GateError(f"Tokenizer field {field_name!r} is not a one-dimensional integer list.")
    return list(value)


def _tokenize_rendered(processor: Any, text: str) -> dict[str, list[int]]:
    base_kwargs = {
        "add_special_tokens": False,
        "return_attention_mask": True,
    }
    kwargs = dict(base_kwargs)
    for name in processor_token_type_input_names(processor):
        kwargs[f"return_{name}"] = True

    def encode(call_kwargs: Mapping[str, Any]) -> Any:
        try:
            return processor(text=text, **call_kwargs)
        except TypeError:
            return processor(text, **call_kwargs)

    try:
        encoded = encode(kwargs)
    except TypeError:
        # Some tokenizer-style fallbacks advertise an input without accepting
        # the corresponding ``return_*`` processor option. Required Gemma 4
        # inputs are synthesized after model discovery in the training path.
        encoded = encode(base_kwargs)
    if not hasattr(encoded, "items"):
        raise GateError(f"Processor returned non-mapping tokenization output {type(encoded).__name__}.")
    result: dict[str, list[int]] = {}
    for key, value in encoded.items():
        try:
            flattened = _flatten_encoding(value, field_name=str(key))
        except GateError:
            # Image tensors and other non-token side data are not part of a text-only trace.
            continue
        result[str(key)] = flattened
    if "input_ids" not in result:
        raise GateError("Processor tokenization did not return input_ids.")
    if "attention_mask" not in result:
        result["attention_mask"] = [1] * len(result["input_ids"])
    for key, values in result.items():
        if key in {"input_ids", "attention_mask"} or key.endswith("token_type_ids"):
            if len(values) != len(result["input_ids"]):
                raise GateError(f"Processor field {key!r} is not aligned to input_ids.")
    return result


def _template_kwargs(*, tools: list[dict[str, Any]], add_generation_prompt: bool) -> dict[str, Any]:
    return {
        "tools": tools,
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }


def _render(processor: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    template = getattr(processor, "apply_chat_template", None)
    if not callable(template):
        raise GateError("The selected processor/tokenizer does not expose apply_chat_template().")
    try:
        rendered = template(messages, **_template_kwargs(tools=tools, add_generation_prompt=add_generation_prompt))
    except Exception as exc:
        raise GateError(
            "Exact apply_chat_template rendering failed; no fallback template was used: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(rendered, str) or not rendered:
        raise GateError("apply_chat_template(tokenize=False) returned no rendered text.")
    return rendered


def _target_call(final_message: Mapping[str, Any]) -> ParsedToolCall | None:
    calls = final_message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    if len(calls) != 1:
        raise GateError("Each SFT target must contain exactly one tool call.")
    function = calls[0].get("function")
    if not isinstance(function, Mapping):
        raise GateError("Assistant tool-call target has no function object.")
    return ParsedToolCall(name=str(function["name"]), arguments=copy.deepcopy(function["arguments"]))


def _json_tool_call(value: Any) -> ParsedToolCall | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        return _json_tool_call(value["function"])
    name = value.get("name", value.get("tool_name"))
    arguments = value.get("arguments", value.get("parameters"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if isinstance(name, str) and name and isinstance(arguments, dict):
        return ParsedToolCall(name=name, arguments=arguments)
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) == 1:
        return _json_tool_call(tool_calls[0])
    return None


_CALL_NAME_RE = re.compile(r"(?:<\|tool_call\|>|<tool_call>)?\s*call:([A-Za-z_][A-Za-z0-9_.-]*)")


def _gemma_wire_to_json(text: str) -> str:
    """Quote Gemma wire-format object keys without changing string contents."""
    source = text.replace('<|"|>', '"')
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(source):
        character = source[index]
        output.append(character)
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            index += 1
            continue
        if character not in "{,":
            index += 1
            continue

        scan = index + 1
        while scan < len(source) and source[scan].isspace():
            output.append(source[scan])
            scan += 1
        if scan >= len(source) or source[scan] in {'"', "{", "["}:
            index = scan
            continue
        key_end = scan
        while key_end < len(source) and source[key_end] not in ":,{}[]":
            key_end += 1
        raw_key = source[scan:key_end]
        stripped_key = raw_key.rstrip()
        if key_end < len(source) and source[key_end] == ":" and stripped_key:
            trailing = raw_key[len(stripped_key) :]
            output.extend(['"', stripped_key, '"', trailing, ":"])
            index = key_end + 1
        else:
            index = scan
    return "".join(output)


def parse_tool_call(text: str) -> ParsedToolCall:
    """Parse common Gemma native tool-call renderings into a canonical call."""
    if not isinstance(text, str) or not text.strip():
        raise GateError("Generated tool-call text is empty.")
    decoder = json.JSONDecoder()
    named = _CALL_NAME_RE.search(text)
    if named is not None:
        object_start = text.find("{", named.end())
        if object_start != -1:
            try:
                arguments, _ = decoder.raw_decode(text, object_start)
            except json.JSONDecodeError:
                # Gemma 4's native template uses a JSON-like wire format with
                # bare object keys and <|"|> quote sentinels, for example:
                #   {state_id:<|"|>active<|"|>}
                # Canonicalize only the argument object following call:<name>.
                gemma_json = _gemma_wire_to_json(text[object_start:])
                try:
                    arguments, _ = decoder.raw_decode(gemma_json)
                except json.JSONDecodeError:
                    arguments = None
            if isinstance(arguments, dict):
                return ParsedToolCall(name=named.group(1), arguments=arguments)

    candidates: list[ParsedToolCall] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        parsed = _json_tool_call(value)
        if parsed is not None and parsed not in candidates:
            candidates.append(parsed)
    if len(candidates) != 1:
        detail = "none" if not candidates else "multiple"
        raise GateError(f"Generated output did not contain exactly one parseable tool call ({detail} found).")
    return candidates[0]


def verify_assistant_only_mask(example: PreparedExample) -> None:
    size = len(example.input_ids)
    if not (size == len(example.labels) == len(example.completion_mask) == len(example.attention_mask)):
        raise GateError("Token ids, attention mask, labels, and completion mask have different lengths.")
    supervised = [index for index, label in enumerate(example.labels) if label != -100]
    if not supervised:
        raise GateError("No supervised assistant tokens remain after tokenization.")
    first = supervised[0]
    if any(value != -100 for value in example.labels[:first]):
        raise GateError("A system/user/history token is supervised.")
    if any(value == -100 for value in example.labels[first:]):
        raise GateError("The assistant target mask is not one contiguous suffix.")
    if example.labels[first:] != example.input_ids[first:]:
        raise GateError("Assistant labels do not match assistant input token ids token by token.")
    expected_mask = [0] * first + [1] * (size - first)
    if example.completion_mask != expected_mask:
        raise GateError("Completion mask disagrees with assistant-only labels.")


def prepare_example(
    row: Mapping[str, Any],
    processor: Any,
    *,
    max_length: int,
    row_label: str = "row",
) -> PreparedExample:
    """Render and tokenize one final-assistant target without truncating it."""
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    tools = validate_tool_schemas(row.get("tools"), row_label=row_label)
    messages = validate_messages(
        row.get("messages"),
        tools=tools,
        row_label=row_label,
    )
    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise GateError(f"{row_label}: assistant target has no prompt history.")

    rendered_prompt = _render(processor, prompt_messages, tools, add_generation_prompt=True)
    rendered_full = _render(processor, messages, tools, add_generation_prompt=False)
    empty_thought_injected = False
    if not rendered_full.startswith(rendered_prompt):
        # Some exact Gemma 4 templates add an empty thought channel only to a
        # generation prompt.  Training must mirror that no-thinking prefix so
        # prompt/completion boundaries match inference.  This is the same
        # explicit normalization used by the root Gemma training path; it is
        # not a fallback rendering template.
        model_marker = "<|turn>model\n"
        empty_thought = "<|channel>thought\n<channel|>"
        marker_end = rendered_prompt.rfind(model_marker)
        if marker_end != -1:
            marker_end += len(model_marker)
            prompt_suffix = rendered_prompt[marker_end:]
            base = rendered_prompt[:marker_end]
            if prompt_suffix == empty_thought and rendered_full.startswith(base):
                rendered_full = base + empty_thought + rendered_full[len(base) :]
                empty_thought_injected = True
    if not rendered_full.startswith(rendered_prompt):
        raise GateError(
            f"{row_label}: generation prompt is not an exact prefix of the training render; "
            "assistant token boundaries cannot be proven."
        )

    prompt_encoding = _tokenize_rendered(processor, rendered_prompt)
    full_encoding = _tokenize_rendered(processor, rendered_full)
    prompt_ids = prompt_encoding["input_ids"]
    full_ids = full_encoding["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise GateError(
            f"{row_label}: tokenized generation prompt is not an exact prefix of the full row."
        )
    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise GateError(f"{row_label}: zero supervised assistant tokens.")
    if len(target_ids) > max_length:
        raise TargetTruncationError(
            f"{row_label}: assistant target is {len(target_ids)} tokens, exceeding max_length={max_length}.",
            original_length=len(full_ids),
            target_length=len(target_ids),
        )

    slice_start = max(0, len(full_ids) - max_length)
    prompt_truncated = slice_start > 0
    used_ids = full_ids[slice_start:]
    prompt_kept = len(prompt_ids) - slice_start
    if prompt_kept < 0:
        raise TargetTruncationError(
            f"{row_label}: truncation would remove part of the assistant target.",
            original_length=len(full_ids),
            target_length=len(target_ids),
        )
    completion_mask = [0] * prompt_kept + [1] * len(target_ids)
    if len(completion_mask) != len(used_ids) or used_ids[-len(target_ids) :] != target_ids:
        raise TargetTruncationError(
            f"{row_label}: target-survival check failed after truncation.",
            original_length=len(full_ids),
            target_length=len(target_ids),
        )
    labels = [token if is_completion else -100 for token, is_completion in zip(used_ids, completion_mask)]

    side_inputs: dict[str, list[int]] = {}
    for key, values in full_encoding.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if key.endswith("token_type_ids") and len(values) == len(full_ids):
            side_inputs[key] = values[slice_start:]
    attention_mask = full_encoding["attention_mask"][slice_start:]
    expected = _target_call(messages[-1])
    rendered_completion = rendered_full[len(rendered_prompt) :]
    if expected is not None:
        parsed = parse_tool_call(rendered_completion)
        if parsed != expected:
            raise GateError(
                f"{row_label}: rendered tool-call round trip changed the target: "
                f"expected={expected}, parsed={parsed}."
            )

    example = PreparedExample(
        input_ids=used_ids,
        attention_mask=attention_mask,
        labels=labels,
        completion_mask=completion_mask,
        original_length=len(full_ids),
        used_length=len(used_ids),
        supervised_tokens=len(target_ids),
        prompt_truncated=prompt_truncated,
        target_truncated=False,
        empty_thought_injected=empty_thought_injected,
        target_kind="tool_call" if expected is not None else "assistant_content",
        expected_tool_call=expected,
        rendered_text=rendered_full,
        rendered_prompt=rendered_prompt,
        rendered_completion=rendered_completion,
        side_inputs=side_inputs,
    )
    verify_assistant_only_mask(example)
    return example


def _percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return int(ordered[index])


def _class_label(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        direct = metadata.get("state_class")
        if isinstance(direct, str) and direct:
            return direct
        labels = metadata.get("labels")
        if isinstance(labels, Mapping) and isinstance(labels.get("state_class"), str):
            return str(labels["state_class"])
    return "unknown"


def _target_action(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], Mapping):
        return "unknown"
    calls = messages[-1].get("tool_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], Mapping):
        function = calls[0].get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            return str(function["name"])
    return "assistant_content"


def audit_dataset(
    rows: Iterable[Mapping[str, Any]],
    processor: Any,
    *,
    max_length: int,
    allow_prompt_truncation: bool = False,
    require_current_registry: bool = False,
) -> DatasetGateReport:
    failures: list[str] = []
    warnings: list[str] = []
    prepared: list[PreparedExample] = []
    original_lengths: list[int] = []
    used_lengths: list[int] = []
    supervised_lengths: list[int] = []
    target_truncated = 0
    zero_supervision = 0
    tool_rows = 0
    round_trips = 0
    actions: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    materialized = list(rows)

    if not materialized:
        failures.append("Dataset has no rows.")
    if require_current_registry:
        failures.extend(validate_current_tool_registry(materialized))
    for index, row in enumerate(materialized):
        row_label = str(row.get("example_id") or f"row[{index}]")
        actions[_target_action(row)] += 1
        classes[_class_label(row)] += 1
        try:
            example = prepare_example(row, processor, max_length=max_length, row_label=row_label)
        except TargetTruncationError as exc:
            target_truncated += 1
            original_lengths.append(exc.original_length)
            supervised_lengths.append(exc.target_length)
            failures.append(str(exc))
            continue
        except GateError as exc:
            if "zero supervised" in str(exc).lower():
                zero_supervision += 1
            failures.append(str(exc))
            continue
        prepared.append(example)
        original_lengths.append(example.original_length)
        used_lengths.append(example.used_length)
        supervised_lengths.append(example.supervised_tokens)
        if example.expected_tool_call is not None:
            tool_rows += 1
            round_trips += 1

    prompt_truncated = sum(example.prompt_truncated for example in prepared)
    if prompt_truncated and not allow_prompt_truncation:
        failures.append(
            f"{prompt_truncated} row(s) require prompt truncation at max_length={max_length}; "
            "shorten history or explicitly approve prompt truncation."
        )
    if tool_rows == 0 and prepared:
        warnings.append("No tool-call target rows were present; native tool-call round trip was not exercised.")

    audit = LengthAudit(
        rows_seen=len(materialized),
        rows_prepared=len(prepared),
        p50=_percentile(original_lengths, 50),
        p95=_percentile(original_lengths, 95),
        p99=_percentile(original_lengths, 99),
        maximum=max(original_lengths, default=0),
        used_p50=_percentile(used_lengths, 50),
        used_p95=_percentile(used_lengths, 95),
        used_p99=_percentile(used_lengths, 99),
        used_maximum=max(used_lengths, default=0),
        prompt_truncated_rows=prompt_truncated,
        target_truncated_rows=target_truncated,
        zero_supervision_rows=zero_supervision,
        empty_thought_injected_rows=sum(example.empty_thought_injected for example in prepared),
        supervised_token_p50=_percentile(supervised_lengths, 50),
        supervised_token_p95=_percentile(supervised_lengths, 95),
        supervised_token_p99=_percentile(supervised_lengths, 99),
        supervised_token_maximum=max(supervised_lengths, default=0),
    )
    return DatasetGateReport(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        length_audit=audit,
        tool_call_rows=tool_rows,
        tool_round_trips=round_trips,
        action_distribution=dict(sorted(actions.items())),
        class_distribution=dict(sorted(classes.items())),
        prepared=prepared,
    )


def _round1_auxiliary_probe_failures(
    row: Mapping[str, Any],
    *,
    generation_provenance_id: str,
) -> tuple[str, ...]:
    """Return why one row is not the exact authenticated probe shape.

    This check does not authenticate provenance by itself.  Its caller may
    supply only the generation ID returned by the immutable Round-1 source
    gate; the training entrypoint enforces that ordering before calling the
    grouped-pilot gate.
    """

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return ("metadata is missing",)

    expected = {
        "dataset_mode": "production",
        "dataset_source": _ROUND1_RECOVERY_PROBE_SOURCE,
        "collector_contract": _ROUND1_RECOVERY_PROBE_CONTRACT,
        "state_origin": _ROUND1_RECOVERY_PROBE_SOURCE,
        "collection_role": _ROUND1_RECOVERY_PROBE_ROLE,
        "state_visited_by": _ROUND1_RECOVERY_PROBE_SOURCE,
        "replay_source": _ROUND1_RECOVERY_PROBE_SOURCE,
        "auxiliary_training_eligible": True,
        "production_label_eligible": False,
        "natural_on_policy_support_eligible": False,
        "training_decision_evidence_verified": True,
        "generation_provenance_id": generation_provenance_id,
    }
    failures: list[str] = []
    for key, wanted in expected.items():
        for container_name, container in (("row", row), ("metadata", metadata)):
            actual = container.get(key)
            matches = actual is wanted if isinstance(wanted, bool) else actual == wanted
            if not matches:
                failures.append(
                    f"{container_name}.{key} must be {wanted!r}, got {actual!r}"
                )
    recovery_stratum = row.get("recovery_stratum")
    if recovery_stratum not in _ROUND1_RECOVERY_PROBE_STRATA:
        failures.append(
            "row.recovery_stratum is not a reviewed recovery-probe stratum"
        )
    if metadata.get("recovery_stratum") != recovery_stratum:
        failures.append("metadata.recovery_stratum does not mirror the row")
    return tuple(failures)


def _has_round1_recovery_probe_marker(row: Mapping[str, Any]) -> bool:
    metadata = row.get("metadata")
    containers = (row, metadata if isinstance(metadata, Mapping) else {})
    return any(
        container.get("dataset_source") == _ROUND1_RECOVERY_PROBE_SOURCE
        or container.get("collector_contract") == _ROUND1_RECOVERY_PROBE_CONTRACT
        or container.get("state_origin") == _ROUND1_RECOVERY_PROBE_SOURCE
        or container.get("collection_role") == _ROUND1_RECOVERY_PROBE_ROLE
        or container.get("state_visited_by") == _ROUND1_RECOVERY_PROBE_SOURCE
        or container.get("replay_source") == _ROUND1_RECOVERY_PROBE_SOURCE
        or container.get("auxiliary_training_eligible") is True
        for container in containers
    )


def validate_grouped_pilot(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_key: str = "root_scenario_id",
    minimum_rows: int = 32,
    maximum_rows: int = 128,
    require_validation: bool = True,
    require_production_dataset_mode: bool = True,
    require_production_label_eligible: bool = True,
    required_protocol: str | None = None,
    validated_round1_generation_provenance_id: str | None = None,
) -> GroupedPilotReport:
    failures: list[str] = []
    trusted_round1_id = validated_round1_generation_provenance_id
    if trusted_round1_id is not None and re.fullmatch(
        r"[0-9a-f]{64}", trusted_round1_id
    ) is None:
        failures.append(
            "validated Round-1 generation provenance ID must be lowercase 64-hex."
        )
        trusted_round1_id = None
    total = sum(len(rows) for rows in splits.values())
    if total < minimum_rows or total > maximum_rows:
        failures.append(f"Pilot size must be in [{minimum_rows}, {maximum_rows}], got {total} rows.")
    if not splits.get("train"):
        failures.append("Grouped pilot has no train rows.")
    if require_validation and not splits.get("validation"):
        failures.append("Grouped pilot has no validation rows.")

    ownership: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    action_distribution: dict[str, dict[str, int]] = {}
    class_distribution: dict[str, dict[str, int]] = {}
    for split_name, rows in splits.items():
        groups: set[str] = set()
        actions: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        for index, row in enumerate(rows):
            probe_marked = _has_round1_recovery_probe_marker(row)
            if require_production_label_eligible and (
                row.get("production_label_eligible") is not True or probe_marked
            ):
                auxiliary_failures = (
                    _round1_auxiliary_probe_failures(
                        row,
                        generation_provenance_id=trusted_round1_id,
                    )
                    if trusted_round1_id is not None
                    else ("no validated Round-1 source binding was supplied",)
                )
                if auxiliary_failures:
                    failures.append(
                        f"{split_name}[{index}] is not explicitly production-label "
                        "eligible as a non-probe row and is not an authenticated "
                        "Round-1 recovery probe: "
                        + "; ".join(auxiliary_failures)
                        + "."
                    )
            metadata = row.get("metadata")
            protocol = metadata.get("protocol") if isinstance(metadata, Mapping) else None
            if required_protocol is not None and protocol != required_protocol:
                failures.append(
                    f"{split_name}[{index}] must use {required_protocol!r} protocol, got {protocol!r}."
                )
            if require_production_dataset_mode and (
                row.get("dataset_mode") != "production"
                or not isinstance(metadata, Mapping)
                or metadata.get("dataset_mode") != "production"
            ):
                failures.append(
                    f"{split_name}[{index}] is not tagged as a production dataset row."
                )
            group = row.get(group_key)
            if not isinstance(group, str) or not group:
                failures.append(f"{split_name}[{index}] is missing non-empty {group_key!r}.")
                continue
            groups.add(group)
            ownership.setdefault(group, set()).add(split_name)
            actions[_target_action(row)] += 1
            classes[_class_label(row)] += 1
        split_groups[split_name] = groups
        action_distribution[split_name] = dict(sorted(actions.items()))
        class_distribution[split_name] = dict(sorted(classes.items()))

    overlaps = {
        group: tuple(sorted(owners))
        for group, owners in ownership.items()
        if len(owners) > 1
    }
    if overlaps:
        preview = ", ".join(f"{group}:{'/'.join(owners)}" for group, owners in sorted(overlaps.items())[:8])
        failures.append(f"Root-scenario groups overlap across splits: {preview}.")
    return GroupedPilotReport(
        passed=not failures,
        failures=tuple(failures),
        total_rows=total,
        split_rows={name: len(rows) for name, rows in splits.items()},
        split_group_counts={name: len(groups) for name, groups in split_groups.items()},
        overlapping_groups=overlaps,
        action_distribution=action_distribution,
        class_distribution=class_distribution,
    )


def load_exact_processor(
    model_name: str,
    revision: str,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    auto_processor_cls: Any | None = None,
    auto_tokenizer_cls: Any | None = None,
) -> tuple[Any, str]:
    """Load the pinned Gemma 4 processor/tokenizer or fail closed.

    The two injectable class arguments exist for offline tests; production callers
    leave them unset and therefore use Transformers' AutoProcessor/AutoTokenizer.
    """
    if not isinstance(model_name, str) or "gemma-4" not in model_name.lower():
        raise GateError(f"Exact live gate requires a Gemma 4 model id, got {model_name!r}.")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision.strip()) is None:
        raise GateError("Exact live gate requires a pinned 40-character Hugging Face commit revision.")
    if auto_processor_cls is None or auto_tokenizer_cls is None:
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on the live environment.
            raise GateError(f"Transformers is unavailable for the exact Gemma gate: {exc}") from exc
        auto_processor_cls = auto_processor_cls or AutoProcessor
        auto_tokenizer_cls = auto_tokenizer_cls or AutoTokenizer

    kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    errors: list[str] = []
    for label, loader in (("AutoProcessor", auto_processor_cls), ("AutoTokenizer", auto_tokenizer_cls)):
        try:
            processor = loader.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if not callable(getattr(processor, "apply_chat_template", None)):
            errors.append(f"{label}: loaded object has no apply_chat_template")
            continue
        return processor, label
    raise GateError(
        "Pinned Gemma 4 processor/tokenizer is unavailable; live gate is NO-GO. " + " | ".join(errors)
    )
