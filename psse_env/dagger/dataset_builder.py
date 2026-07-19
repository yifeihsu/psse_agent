from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from psse_env.actions import INVALID_ACTION, MACRO_ACTIONS, safe_normalize_action
from psse_env.state_store import find_forbidden_policy_paths


DEFAULT_DAGGER_SYSTEM_PROMPT = (
    "You are a PSSE diagnostic agent. Given the current state summary and recent "
    "tool history, choose exactly one next valid tool call through the provided "
    "tool interface. Do not return final-answer prose. State references are local "
    "controller aliases such as active and candidate."
)
CANONICAL_DAGGER_SYSTEM_PROMPT = (
    "You are a power-system state-estimation diagnostic agent. Given the current "
    "state summary and recent tool history, choose exactly one next valid tool "
    "call through the provided tool interface. Do not return final-answer prose. "
    "Case references such as active and candidate are local case_path aliases; "
    "use them exactly as provided."
)
SUPPORTED_EXPORT_PROTOCOLS = ("controller", "canonical")

# State identifiers are transaction-controller capabilities, not semantic model
# features.  Only these argument keys are rebound after model generation.
ACTION_STATE_REFERENCE_KEYS = frozenset({"state_id", "candidate_state_id", "parent_state_id"})
POLICY_IDENTIFIER_KEYS = frozenset(
    {
        "active_state_id",
        "candidate_state_id",
        "candidate_parent_id",
        "parent_state_id",
        "state_id",
        "measurement_context_state_id",
        "parameter_context_state_id",
        "topology_context_state_id",
    }
)
EPISODE_IDENTIFIER_KEYS = frozenset({"episode_id"})
HISTORY_KEYS = frozenset(
    {
        "history",
        "history_window",
        "recent_action_observation_history",
        "recent_history",
    }
)
HASH_IDENTIFIER_KEY_ORDER = (
    "state_hash",
    "state_hash_before",
    "state_hash_after",
    "restored_parent_hash",
)
HASH_IDENTIFIER_KEYS = frozenset(HASH_IDENTIFIER_KEY_ORDER)
IDENTIFIER_BEARING_TEXT_KEYS = frozenset(
    {
        # These fields deliberately serialize an action that can contain a
        # controller state reference.  Embedded replacement is safe here; it
        # is not safe in arbitrary strings such as tool names or evidence.
        "action_signature",
        "modification_signature",
        "tried_action_signatures",
    }
)
NON_MODEL_VISIBLE_KEYS = frozenset(
    {
        "semantic_field_provenance",
        "policy_field_provenance",
        "policy_provenance",
        "provenance",
        "scenario_id",
        "root_scenario_id",
        "source_example_id",
    }
)
DERIVED_POLICY_FIELDS = frozenset(
    {
        "unresolved_signatures",
        "remaining_anomaly_score",
        "no_material_anomaly_remaining",
    }
)
FORBIDDEN_PROVENANCE_SOURCES = frozenset(
    {
        "hidden_truth",
        "hidden-truth",
        "oracle_hint",
        "oracle-hint",
        "oracle_action_hint",
        "oracle_action_hints",
        "synthetic_oracle",
        "ground_truth",
    }
)
PROVENANCE_SOURCE_KEYS = frozenset(
    {
        "source",
        "sources",
        "derived_from",
        "origin",
        "upstream_source",
        "source_type",
        "source_kind",
        "kind",
    }
)

HISTORY_METRIC_KEYS = (
    "wls_objective",
    "chi_square_statistic",
    "residual_norm",
    "max_normalized_residual",
    "remaining_anomaly_score",
    "anomaly_threshold",
    "chi_square_threshold",
    "power_flow_converged",
    "topology_feasible",
    "target_progress",
    "global_progress",
    "post_action_resolved",
    "globally_resolved",
    "new_violations",
    "context_tool",
    "finding_count",
    "context_version",
    "measurement_findings",
    "parameter_findings",
    "topology_findings",
    "supported_corrections",
    "wls_summary",
    "harmonic_orders",
    "measured_buses",
    "harmonic_summary",
    "best_candidate_bus_1based",
    "hse_summary",
    "nlm_summary",
    "hif_summary",
)
CONTEXT_DETAIL_KEYS = frozenset(
    {
        "measurement_findings",
        "parameter_findings",
        "topology_findings",
        "supported_corrections",
        "wls_summary",
        "harmonic_summary",
        "hse_summary",
        "nlm_summary",
        "hif_summary",
    }
)


def _object_schema(
    properties: Mapping[str, Any] | None = None,
    *,
    required: Sequence[str] = (),
    additional_properties: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": copy.deepcopy(dict(properties or {})),
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = list(required)
    return schema


_STATE_ID = {
    "type": "string",
    "description": "Controller-local state alias, normally active or candidate.",
}
_BRANCH_TARGET_PROPERTIES = {
    "branch_id": {"type": "string"},
    "cb_name": {"type": "string"},
    "line_index": {"type": "integer"},
    "line_index1": {"type": "integer"},
    "branch_row0": {"type": "integer"},
}


def _tool_schema(name: str, description: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": copy.deepcopy(dict(parameters)),
        },
    }


# Native Transformers/TRL rows need full JSON tool schemas on every row.  The
# schemas intentionally describe the public macro-action surface, not solver
# implementation details or hidden oracle fields.
TOOL_JSON_SCHEMAS: list[dict[str, Any]] = [
    _tool_schema(
        "run_wls",
        "Run observable weighted least-squares diagnostics on the current state.",
        _object_schema({"state_id": _STATE_ID}, required=("state_id",)),
    ),
    _tool_schema(
        "verify_candidate",
        "Verify the open candidate using observable solver evidence.",
        _object_schema({"state_id": _STATE_ID}, required=("state_id",)),
    ),
    *[
        _tool_schema(
            name,
            f"Retrieve observable {name.removeprefix('get_').removesuffix('_context')} evidence.",
            _object_schema({"state_id": _STATE_ID}, required=("state_id",)),
        )
        for name in (
            "get_measurement_context",
            "get_parameter_context",
            "get_topology_context",
        )
    ],
    _tool_schema(
        "correct_measurements",
        "Create a candidate by applying bounded measurement replacements.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                "measurement_updates": {
                    "type": "object",
                    "description": "Map of zero-based measurement indices to replacement values.",
                    "additionalProperties": {"type": "number"},
                },
                "measurement_index": {"type": "integer"},
                "index": {"type": "integer"},
                "index0": {"type": "integer"},
                "target": {"type": "integer"},
                "meter": {"type": "integer"},
                "measurement_id": {"type": "integer"},
            },
            required=("state_id", "measurement_updates"),
        ),
    ),
    _tool_schema(
        "correct_parameters",
        "Create a candidate by correcting one identified branch parameter.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                **_BRANCH_TARGET_PROPERTIES,
                "field": {"type": "string", "enum": ["r", "x", "b"]},
                "parameter": {"type": "string", "enum": ["r", "x", "b"]},
                "value": {"type": "number"},
                "corrected_value": {"type": "number"},
                "new_value": {"type": "number"},
                "multiplier": {"type": "number"},
            },
            required=("state_id",),
        ),
    ),
    _tool_schema(
        "correct_topology",
        "Create a candidate by changing one identified branch or breaker status.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                **_BRANCH_TARGET_PROPERTIES,
                "status": {"type": "integer", "enum": [0, 1]},
                "expected_status": {"type": "integer", "enum": [0, 1]},
                "status_field": {"type": "string"},
            },
            required=("state_id",),
        ),
    ),
    _tool_schema(
        "commit_state",
        "Commit a verified acceptable candidate.",
        _object_schema({"candidate_state_id": _STATE_ID}, required=("candidate_state_id",)),
    ),
    _tool_schema(
        "rollback_state",
        "Rollback a verified rejected or inconclusive candidate.",
        _object_schema({"candidate_state_id": _STATE_ID}, required=("candidate_state_id",)),
    ),
    _tool_schema(
        "finalize_diagnosis",
        "Finish only after observable evidence shows no material anomaly remains.",
        _object_schema(),
    ),
    _tool_schema(
        "ask_for_more_evidence",
        "Request additional observable evidence for the current state.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                "request": {"type": "string"},
            },
            required=("state_id",),
        ),
    ),
    _tool_schema(
        "run_alternative_test",
        "Run an alternative observable diagnostic test on the current state.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                "test_name": {"type": "string"},
            },
            required=("state_id",),
        ),
    ),
    _tool_schema(
        "get_harmonic_context",
        "Retrieve observable harmonic measurement context for the current state.",
        _object_schema({"state_id": _STATE_ID}, required=("state_id",)),
    ),
    _tool_schema(
        "run_hse_from_path",
        "Run harmonic state estimation on the referenced state's harmonic measurements.",
        _object_schema({"state_id": _STATE_ID}, required=("state_id",)),
    ),
    _tool_schema(
        "run_three_phase_nlm_from_path",
        "Run three-phase NLM high-impedance-fault localization on the referenced state.",
        _object_schema({"state_id": _STATE_ID}, required=("state_id",)),
    ),
    _tool_schema(
        "estimate_hif_location_magnitude_from_path",
        "Estimate HIF position and magnitude on a suspected branch of the referenced state.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                "candidate_branch_row0": {"type": "integer"},
                "candidate_phase": {"type": ["string", "null"], "enum": ["A", "B", "C", None]},
                "top_k": {"type": "integer"},
                "alpha_grid_size": {"type": "integer"},
                "r_grid_size": {"type": "integer"},
                "r_hif_pu_min": {"type": "number"},
                "r_hif_pu_max": {"type": "number"},
            },
            required=("state_id", "candidate_branch_row0"),
        ),
    ),
    _tool_schema(
        "estimate_hif_location_magnitude_multiscan_from_path",
        "Estimate shared HIF parameters from the referenced state's persistent scan window.",
        _object_schema(
            {
                "state_id": _STATE_ID,
                "candidate_branch_row0": {"type": "integer"},
                "candidate_phase": {"type": ["string", "null"], "enum": ["A", "B", "C", None]},
                "resistance_mode": {"type": "string", "enum": ["shared", "scan_specific_smooth"]},
                "max_scans": {"type": "integer"},
                "scan_selection": {
                    "type": "string",
                    "enum": ["all", "diversity_greedy", "information_greedy"],
                },
                "top_k": {"type": "integer"},
                "alpha_grid_size": {"type": "integer"},
                "r_grid_size": {"type": "integer"},
                "r_hif_pu_min": {"type": "number"},
                "r_hif_pu_max": {"type": "number"},
                "robust_loss": {"type": "string", "enum": ["linear", "soft_l1", "huber"]},
                "smoothness_lambda": {"type": "number"},
            },
            required=("state_id", "candidate_branch_row0"),
        ),
    ),
]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _normalized_provenance_source(source: Any) -> str:
    return str(source).strip().lower().replace(" ", "_")


def find_forbidden_provenance_paths(value: Any, prefix: str = "$") -> list[str]:
    """Return provenance paths that depend on privileged synthetic sources."""
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if str(key) in PROVENANCE_SOURCE_KEYS:
                sources = item if isinstance(item, (list, tuple, set)) else [item]
                for index, source in enumerate(sources):
                    if isinstance(source, (Mapping, list, tuple, set)):
                        continue
                    normalized = _normalized_provenance_source(source)
                    if _is_forbidden_provenance_source(normalized):
                        suffix = f"[{index}]" if len(sources) > 1 else ""
                        paths.append(f"{path}{suffix}")
            paths.extend(find_forbidden_provenance_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(find_forbidden_provenance_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and _is_forbidden_provenance_source(
        _normalized_provenance_source(value)
    ):
        # Provenance producers sometimes wrap the source as
        # {"source": {"kind": "hidden_truth"}} or use a project-specific
        # source-key alias.  Scanning scalar leaves makes those variants fail
        # closed without imposing one metadata schema on every provider.
        paths.append(prefix)
    return paths


def _is_forbidden_provenance_source(normalized: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    tokens = (
        normalized.replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(":", "_")
        .split("_")
    )
    return (
        normalized in FORBIDDEN_PROVENANCE_SOURCES
        or any(
            marker in compact
            for marker in (
                "hiddentruth",
                "oraclehint",
                "actionhint",
                "groundtruth",
                "synthetic",
            )
        )
        or any(
            token in {"hidden", "oracle", "truth", "synthetic", "hint"}
            for token in tokens
        )
    )


def _find_embedded_forbidden_provenance_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if "provenance" in key_text.lower():
                paths.extend(find_forbidden_provenance_paths(item, path))
            else:
                paths.extend(_find_embedded_forbidden_provenance_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_find_embedded_forbidden_provenance_paths(item, f"{prefix}[{index}]"))
    return paths


def _meaningful_derived_value(field: str, value: Any) -> bool:
    if field == "unresolved_signatures":
        return bool(value)
    if field == "remaining_anomaly_score":
        return value is not None
    if field == "no_material_anomaly_remaining":
        return value is True
    return value not in (None, False, "", [], {})


def _provenance_entry(provenance: Mapping[str, Any], field: str) -> Any:
    candidates: list[Mapping[str, Any]] = [provenance]
    for container_key in ("fields", "policy_fields", "state", "policy_observation"):
        nested = provenance.get(container_key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    path_keys = (field, f"$.{field}", f"$.state.{field}", f"$.policy_observation.{field}")
    for candidate in candidates:
        for key in path_keys:
            if key in candidate:
                return candidate[key]
    return None


def _has_declared_provenance_source(entry: Any) -> bool:
    if isinstance(entry, str):
        return bool(entry.strip())
    if isinstance(entry, Mapping):
        for key, value in entry.items():
            if str(key) in PROVENANCE_SOURCE_KEYS:
                if isinstance(value, str) and value.strip():
                    return True
                if isinstance(value, (list, tuple)) and any(str(item).strip() for item in value):
                    return True
                if _has_declared_provenance_source(value):
                    return True
            elif isinstance(value, Mapping) and _has_declared_provenance_source(value):
                return True
    return False


def validate_policy_provenance(
    payload: Any,
    provenance: Mapping[str, Any] | None,
    *,
    require_derived_provenance: bool = True,
) -> None:
    """Validate semantic policy fields against their non-model provenance.

    Empty/default derived values are safe literals.  Once a derived field is
    informative, export fails closed unless its source is supplied and is
    deployment-observable.  Provenance itself is persisted in row metadata and
    is never placed in the model prompt.
    """
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("policy provenance must be a mapping")
    provenance_map = dict(provenance or {})
    forbidden = find_forbidden_provenance_paths(provenance_map)
    if forbidden:
        raise ValueError(
            "Privileged provenance found for policy-visible fields: " + ", ".join(forbidden)
        )
    if not require_derived_provenance or not isinstance(payload, Mapping):
        return
    state = payload.get("state", payload)
    if not isinstance(state, Mapping):
        return
    missing: list[str] = []
    for field in sorted(DERIVED_POLICY_FIELDS):
        if field not in state or not _meaningful_derived_value(field, state[field]):
            continue
        entry = _provenance_entry(provenance_map, field)
        if entry is None or not _has_declared_provenance_source(entry):
            missing.append(field)
    if missing:
        raise ValueError(
            "Missing observable provenance for derived policy fields: " + ", ".join(missing)
        )


def validate_policy_payload(
    payload: Any,
    *,
    provenance: Mapping[str, Any] | None = None,
    require_derived_provenance: bool = False,
) -> None:
    """Fail closed when oracle-only fields or provenance reach model data."""
    paths = find_forbidden_policy_paths(payload)
    if paths:
        raise ValueError(f"Privileged fields found in policy payload: {', '.join(paths)}")
    embedded_provenance = _find_embedded_forbidden_provenance_paths(payload)
    if embedded_provenance:
        raise ValueError(
            "Privileged provenance found in policy payload: " + ", ".join(embedded_provenance)
        )
    validate_policy_provenance(
        payload,
        provenance,
        require_derived_provenance=require_derived_provenance,
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_tool_schemas(tools: Sequence[Mapping[str, Any]]) -> None:
    """Validate the portable subset required by Transformers chat templates."""
    if not tools:
        raise ValueError("At least one JSON tool schema is required for every SFT row.")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise ValueError(f"Tool schema {index} must be a type=function object.")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"Tool schema {index} has no function object.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tool schema {index} has no function name.")
        if name in names:
            raise ValueError(f"Duplicate tool schema name: {name}")
        names.add(name)
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
            raise ValueError(f"Tool schema {name} parameters must be a JSON object schema.")
        properties = parameters.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"Tool schema {name} parameters.properties must be an object.")
        required = parameters.get("required", [])
        if not isinstance(required, list) or any(key not in properties for key in required):
            raise ValueError(f"Tool schema {name} has invalid required properties.")
    try:
        json.dumps(list(tools), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tool schemas must be finite JSON: {exc}") from exc


def resolve_tool_schemas(
    available_tools: Iterable[str] | Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve tool names or caller-supplied schemas to row-level JSON schemas."""
    requested = list(available_tools) if available_tools is not None else []
    if not requested:
        resolved = copy.deepcopy(TOOL_JSON_SCHEMAS)
    elif all(isinstance(item, str) for item in requested):
        names = [str(item) for item in requested]
        unknown = sorted(set(names) - MACRO_ACTIONS)
        if unknown:
            raise ValueError(f"No DAgger JSON schema for tools: {', '.join(unknown)}")
        by_name = {tool["function"]["name"]: tool for tool in TOOL_JSON_SCHEMAS}
        resolved = [copy.deepcopy(by_name[name]) for name in names]
    elif all(isinstance(item, Mapping) for item in requested):
        resolved = copy.deepcopy(requested)
    else:
        raise ValueError("available_tools must contain either names or JSON schema objects, not both.")
    validate_tool_schemas(resolved)
    return resolved


def _iter_identifier_values(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in POLICY_IDENTIFIER_KEYS and item is not None:
                yield key_text, str(item)
            yield from _iter_identifier_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_identifier_values(item)


def build_state_alias_bindings(*payloads: Any) -> dict[str, str]:
    """Build deterministic local-alias -> controller-state bindings."""
    references: list[tuple[str, str]] = []
    for payload in payloads:
        references.extend(_iter_identifier_values(payload))

    alias_to_id: dict[str, str] = {}
    active = next((value for key, value in references if key == "active_state_id"), None)
    candidate = next((value for key, value in references if key == "candidate_state_id"), None)
    if active is not None:
        alias_to_id["active"] = active
    if candidate is not None and candidate != active:
        alias_to_id["candidate"] = candidate

    assigned = set(alias_to_id.values())
    next_index = 0
    for _, identifier in references:
        if identifier in assigned:
            continue
        while f"s{next_index}" in alias_to_id:
            next_index += 1
        alias_to_id[f"s{next_index}"] = identifier
        assigned.add(identifier)
        next_index += 1
    return alias_to_id


def build_hash_alias_bindings(*payloads: Any) -> dict[str, str]:
    """Build local aliases for opaque state hashes while preserving equality."""
    identifiers: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in HASH_IDENTIFIER_KEYS and item is not None:
                    identifiers.append(str(item))
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for payload in payloads:
        visit(payload)
    return {f"h{index}": identifier for index, identifier in enumerate(dict.fromkeys(identifiers))}


def _episode_bindings(value: Any) -> dict[str, str]:
    bindings: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in EPISODE_IDENTIFIER_KEYS and item is not None:
                bindings.setdefault("episode", str(item))
            else:
                nested = _episode_bindings(item)
                for alias, identifier in nested.items():
                    bindings.setdefault(alias, identifier)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _episode_bindings(item)
            for alias, identifier in nested.items():
                bindings.setdefault(alias, identifier)
    return bindings


def _replace_identifier_text(text: str, replacements: Mapping[str, str]) -> str:
    sanitized = text
    for identifier, alias in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if identifier:
            sanitized = sanitized.replace(identifier, alias)
    return sanitized


def alias_model_visible_state(
    value: Any,
    alias_to_state_id: Mapping[str, str],
    *,
    episode_aliases: Mapping[str, str] | None = None,
    hash_aliases: Mapping[str, str] | None = None,
) -> Any:
    """Replace controller IDs without rewriting unrelated semantic strings.

    Exact identifier values can occur anywhere in a structured payload.  Only
    known serialized-signature fields permit embedded replacement; applying a
    short episode ID as an unconstrained substring rewrite can corrupt tool
    names (for example, episode ``e`` must not rewrite ``verify_candidate``).
    """
    id_to_alias = {str(identifier): str(alias) for alias, identifier in alias_to_state_id.items()}
    embedded_id_to_alias = dict(id_to_alias)
    for alias, identifier in (episode_aliases or {}).items():
        id_to_alias.setdefault(str(identifier), str(alias))
    for alias, identifier in (hash_aliases or {}).items():
        id_to_alias.setdefault(str(identifier), str(alias))
        embedded_id_to_alias.setdefault(str(identifier), str(alias))

    def visit(item: Any, key: str | None = None) -> Any:
        if isinstance(item, Mapping):
            return {str(child_key): visit(child, str(child_key)) for child_key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child, key) for child in item]
        if isinstance(item, str):
            if item in id_to_alias:
                return id_to_alias[item]
            if (
                key in POLICY_IDENTIFIER_KEYS
                or key in EPISODE_IDENTIFIER_KEYS
                or key in HASH_IDENTIFIER_KEYS
            ):
                return id_to_alias.get(item, item)
            if key in IDENTIFIER_BEARING_TEXT_KEYS:
                return _replace_identifier_text(item, embedded_id_to_alias)
            return item
        return copy.deepcopy(item)

    return visit(value)


def bind_controller_action(
    action: Mapping[str, Any] | str,
    alias_to_state_id: Mapping[str, str],
) -> dict[str, Any]:
    """Bind model-generated local state aliases to real controller state IDs."""
    normalized = safe_normalize_action(action)
    if normalized["tool"] == INVALID_ACTION:
        raise ValueError(f"Cannot bind malformed action: {normalized['arguments']}")
    arguments = copy.deepcopy(normalized["arguments"])
    for key in ACTION_STATE_REFERENCE_KEYS:
        if key not in arguments or arguments[key] is None:
            continue
        alias = str(arguments[key])
        if alias not in alias_to_state_id:
            raise ValueError(f"Unknown controller state alias for {key}: {alias}")
        arguments[key] = str(alias_to_state_id[alias])
    return {"tool": normalized["tool"], "arguments": arguments}


def _bounded_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 3,
    max_items: int = 8,
    max_text_chars: int = 160,
) -> Any:
    if depth >= max_depth:
        if isinstance(value, (Mapping, list, tuple)):
            return "<nested_value_summarized>"
    if isinstance(value, Mapping):
        items = list(value.items())
        bounded = {
            str(key): _bounded_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_text_chars=max_text_chars,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            bounded["_omitted_fields"] = len(items) - max_items
        return bounded
    if isinstance(value, (list, tuple)):
        bounded_list = [
            _bounded_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_text_chars=max_text_chars,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            bounded_list.append({"_omitted_items": len(value) - max_items})
        return bounded_list
    if isinstance(value, str) and len(value) > max_text_chars:
        return value[: max_text_chars - 20] + f"...<{len(value)} chars>"
    return copy.deepcopy(value)


def _bounded_history_metric(key: str, value: Any) -> Any:
    return _bounded_value(
        value,
        max_depth=6 if key in CONTEXT_DETAIL_KEYS else 3,
        max_items=8,
        max_text_chars=160,
    )


def _strict_json_clone(value: Any) -> Any:
    """Return JSON-native values (notably string object keys) or fail closed."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SFT value must be finite JSON: {exc}") from exc


def _first_mapping_value(mappings: Sequence[Mapping[str, Any]], key: str) -> Any:
    for mapping in mappings:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def summarize_history(
    history: Iterable[Mapping[str, Any]] | None,
    *,
    max_events: int = 8,
    max_chars: int = 4096,
) -> list[dict[str, Any]]:
    """Create one bounded, structured history window for the model prompt."""
    if max_events < 0 or max_chars < 64:
        raise ValueError("max_events must be nonnegative and max_chars must be at least 64.")
    source = [item for item in (history or []) if isinstance(item, Mapping)]
    selected = source[-max_events:] if max_events else []
    summarized: list[dict[str, Any]] = []
    omitted = max(0, len(source) - len(selected))
    for index, item in enumerate(selected):
        action = safe_normalize_action(item.get("action", item.get("executed_action", {})))
        tool_output = item.get("tool_output")
        if not isinstance(tool_output, Mapping):
            tool_output = {}
        metrics = tool_output.get("tool_metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        transition = item.get("transition_label")
        if not isinstance(transition, Mapping):
            transition = {}
        event: dict[str, Any] = {
            "tool": action["tool"],
            "arguments": _bounded_value(action["arguments"]),
            "outcome": {
                "execution_status": _first_mapping_value(
                    (tool_output, transition), "execution_status"
                ),
                "process_valid": transition.get("process_valid"),
                "error_code": _first_mapping_value((tool_output, transition), "error_code"),
                "state_mutated": tool_output.get("state_mutated"),
            },
        }
        state_id = item.get("state_id")
        if state_id is not None:
            event["state_id"] = state_id
        compact_metrics = {
            key: _bounded_history_metric(
                key, _first_mapping_value((metrics, tool_output), key)
            )
            for key in HISTORY_METRIC_KEYS
            if _first_mapping_value((metrics, tool_output), key) is not None
        }
        if compact_metrics:
            event["observable_metrics"] = compact_metrics
        if index == 0 and omitted:
            event["events_omitted_before"] = omitted
        summarized.append(event)

    while summarized and len(json.dumps(summarized, sort_keys=True)) > max_chars:
        summarized.pop(0)
        omitted += 1
        if summarized:
            summarized[0]["events_omitted_before"] = omitted
    if selected and not summarized:
        summarized = [{"events_omitted_before": len(source), "summary": "history omitted for size"}]
    return summarized


def _without_history(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_history(item)
            for key, item in value.items()
            if str(key) not in HISTORY_KEYS and str(key) not in NON_MODEL_VISIBLE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_history(item) for item in value]
    return copy.deepcopy(value)


def _compact_last_tool_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    metrics = value.get("tool_metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    compact: dict[str, Any] = {
        key: _bounded_value(value[key])
        for key in ("execution_status", "error_code", "state_mutated", "active_state_id", "candidate_state_id")
        if key in value
    }
    compact_metrics = {
        key: _bounded_history_metric(key, _first_mapping_value((metrics, value), key))
        for key in HISTORY_METRIC_KEYS
        if _first_mapping_value((metrics, value), key) is not None
    }
    # A stable key order is required because the first occurrence of each
    # digest determines its local h0/h1 alias.  Iterating the membership set
    # would make exported rows depend on Python's per-process hash seed.
    for key in HASH_IDENTIFIER_KEY_ORDER:
        hash_value = _first_mapping_value((metrics, value), key)
        if hash_value is not None:
            compact_metrics[key] = str(hash_value)
    if compact_metrics:
        compact["observable_metrics"] = compact_metrics
    return compact


def prepare_model_policy_observation(
    raw_observation: Mapping[str, Any],
    *,
    history: Iterable[Mapping[str, Any]] | None = None,
    max_history_events: int = 8,
    max_history_chars: int = 4096,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the exact model-visible sanitization used by export and audits."""
    if not isinstance(raw_observation, Mapping):
        raise ValueError("Policy observation must be a mapping.")
    source_history = list(
        raw_observation.get("history_window", []) if history is None else (history or [])
    )
    observation = _without_history(raw_observation)
    if "last_tool_output" in observation:
        observation["last_tool_output"] = _compact_last_tool_output(
            observation["last_tool_output"]
        )
    for field in (
        "last_verification",
        "accepted_corrections",
        "explained_anomalies",
        "rejected_hypotheses",
    ):
        if field in observation:
            observation[field] = _bounded_value(
                observation[field], max_depth=4, max_items=8, max_text_chars=160
            )
    history_window = summarize_history(
        source_history,
        max_events=max_history_events,
        max_chars=max_history_chars,
    )
    observation["history_window"] = history_window
    state_aliases = build_state_alias_bindings(observation, history_window)
    episode_aliases = _episode_bindings(raw_observation)
    hash_aliases = build_hash_alias_bindings(observation, history_window)
    aliased = alias_model_visible_state(
        observation,
        state_aliases,
        episode_aliases=episode_aliases,
        hash_aliases=hash_aliases,
    )
    if not isinstance(aliased, Mapping):
        raise ValueError("Prepared policy observation must remain a mapping.")
    return dict(aliased), {
        "state_aliases": state_aliases,
        "episode_aliases": episode_aliases,
        "hash_aliases": hash_aliases,
        "source_history_events": len(source_history),
        "exported_history_events": len(history_window),
    }


_OPAQUE_HASH_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_CONTROLLER_STATE_RE = re.compile(r"[A-Za-z0-9_.-]{8,}:s[0-9]+")


def find_model_identifier_leaks(value: Any, prefix: str = "$") -> list[str]:
    """Find long opaque hashes or controller state IDs in model-visible data."""
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            paths.extend(find_model_identifier_leaks(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(find_model_identifier_leaks(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and (
        _OPAQUE_HASH_RE.search(value) or _CONTROLLER_STATE_RE.search(value)
    ):
        paths.append(prefix)
    return paths


def _extract_policy_provenance(example: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    for value in (
        example.get("semantic_field_provenance"),
        example.get("policy_field_provenance"),
        example.get("policy_provenance"),
        observation.get("semantic_field_provenance"),
        observation.get("policy_field_provenance"),
        observation.get("policy_provenance"),
    ):
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
    provenance = example.get("provenance")
    if isinstance(provenance, Mapping):
        nested = provenance.get("policy_fields", provenance.get("policy_observation"))
        if isinstance(nested, Mapping):
            return copy.deepcopy(dict(nested))
    return {}


def _default_provenance(observation: Mapping[str, Any], supplied: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(supplied))
    for field in DERIVED_POLICY_FIELDS:
        if field in observation and not _meaningful_derived_value(field, observation[field]):
            result.setdefault(field, {"source": "default_literal"})
    return result


def _controller_metadata(
    example: Mapping[str, Any],
    alias_to_state_id: Mapping[str, str],
    episode_aliases: Mapping[str, str],
    hash_aliases: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "state_aliases": copy.deepcopy(dict(alias_to_state_id)),
        "episode_aliases": copy.deepcopy(dict(episode_aliases)),
        "hash_aliases": copy.deepcopy(dict(hash_aliases)),
        "scenario_id": example.get("scenario_id"),
        "root_scenario_id": example.get("root_scenario_id", example.get("scenario_id")),
        "source_example_id": example.get("example_id"),
    }


def examples_to_chat_sft(
    examples: Iterable[Mapping[str, Any]],
    *,
    available_tools: Iterable[str] | Iterable[Mapping[str, Any]] | None = None,
    system_prompt: str | None = None,
    assistant_format: str = "tool_calls",
    max_history_events: int = 8,
    max_history_chars: int = 4096,
    require_derived_provenance: bool = True,
    protocol: str = "controller",
) -> list[dict[str, Any]]:
    """Convert DAgger examples to native, controller-bindable chat SFT rows.

    ``protocol="controller"`` exports the internal macro-action surface.
    ``protocol="canonical"`` exports the deployment power-tool surface
    (``wls_from_path``/``case_path``) shared with the production SFT corpus,
    converting each expert target through the protocol bridge while keeping
    the reverse alias bindings in row metadata.
    """
    if protocol not in SUPPORTED_EXPORT_PROTOCOLS:
        raise ValueError(
            f"protocol must be one of {SUPPORTED_EXPORT_PROTOCOLS}, got {protocol!r}."
        )
    bridge = None
    if protocol == "canonical":
        from psse_env.dagger import protocol_bridge as bridge
    if system_prompt is None:
        system_prompt = (
            CANONICAL_DAGGER_SYSTEM_PROMPT
            if protocol == "canonical"
            else DEFAULT_DAGGER_SYSTEM_PROMPT
        )
    if protocol == "canonical" and available_tools is None:
        tools = bridge.unified_tool_schemas()
        validate_tool_schemas(tools)
    else:
        tools = resolve_tool_schemas(available_tools)
    schema_names = {tool["function"]["name"] for tool in tools}
    rows: list[dict[str, Any]] = []
    for example in examples:
        target = example.get("preferred_action")
        if target is None:
            valid = example.get("valid_next_actions") or []
            target = valid[0] if valid else None
        if target is None:
            continue

        raw_observation = example.get("policy_observation", example.get("state_summary", {}))
        if not isinstance(raw_observation, Mapping):
            raise ValueError(f"Policy observation must be a mapping for {example.get('example_id')}")
        raw_history = example.get("history_window")
        if raw_history is None:
            raw_history = raw_observation.get("history_window", [])
        raw_history = list(raw_history or [])
        aliased_observation, alias_metadata = prepare_model_policy_observation(
            raw_observation,
            history=raw_history,
            max_history_events=max_history_events,
            max_history_chars=max_history_chars,
        )

        normalized_target = safe_normalize_action(target)
        if normalized_target["tool"] == INVALID_ACTION:
            raise ValueError(f"Invalid expert target for {example.get('example_id')}: {normalized_target}")

        alias_to_state_id = alias_metadata["state_aliases"]
        episode_aliases = alias_metadata["episode_aliases"]
        hash_aliases = alias_metadata["hash_aliases"]
        aliased_target = alias_model_visible_state(
            normalized_target,
            alias_to_state_id,
            episode_aliases=episode_aliases,
            hash_aliases=hash_aliases,
        )
        for key in ACTION_STATE_REFERENCE_KEYS:
            target_reference = aliased_target["arguments"].get(key)
            if target_reference is not None and str(target_reference) not in alias_to_state_id:
                raise ValueError(
                    f"Expert target references unbound controller state for {key}: {target_reference}"
                )
        if protocol == "canonical":
            exported_target = bridge.internal_to_canonical_action(aliased_target)
            for key in bridge.CANONICAL_STATE_REFERENCE_KEYS:
                target_reference = exported_target["arguments"].get(key)
                if target_reference is not None and str(target_reference) not in alias_to_state_id:
                    raise ValueError(
                        f"Expert target references unbound case alias for {key}: {target_reference}"
                    )
        else:
            exported_target = aliased_target
        if exported_target["tool"] not in schema_names:
            raise ValueError(
                f"Target tool {exported_target['tool']} has no schema in this SFT row."
            )
        user_payload = {"state": aliased_observation}
        identifier_leaks = find_model_identifier_leaks(user_payload)
        if identifier_leaks:
            raise ValueError(
                "Unaliased controller identifiers found in model payload: "
                + ", ".join(identifier_leaks)
            )

        supplied_provenance = _extract_policy_provenance(example, raw_observation)
        policy_provenance = _default_provenance(raw_observation, supplied_provenance)
        validate_policy_payload(
            user_payload,
            provenance=policy_provenance,
            require_derived_provenance=require_derived_provenance,
        )

        if assistant_format == "tool_calls":
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": exported_target["tool"],
                            # Native Transformers/TRL representation: keep this
                            # as an object.  Stringification belongs to a runtime
                            # transport adapter, never the training dataset.
                            "arguments": _strict_json_clone(exported_target["arguments"]),
                        },
                    }
                ],
            }
        elif assistant_format == "json_content":
            assistant_message = {
                "role": "assistant",
                "content": json.dumps(exported_target, sort_keys=True),
            }
        else:
            raise ValueError("assistant_format must be 'tool_calls' or 'json_content'.")

        source_index = len(rows)
        labels = copy.deepcopy(example.get("labels", {}))
        dataset_mode = example.get("dataset_mode")
        if dataset_mode is None and isinstance(labels, Mapping):
            dataset_mode = labels.get("dataset_mode")
        row = {
            # These are non-model dataset metadata.  Preserve real group
            # ownership so grouped root-scenario splitting remains possible
            # after export; dynamic IDs are excluded from messages below.
            "example_id": example.get("example_id") or f"sft_{source_index:06d}",
            "scenario_id": example.get("scenario_id"),
            "root_scenario_id": example.get("root_scenario_id", example.get("scenario_id")),
            "dataset_mode": dataset_mode,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
                assistant_message,
            ],
            "tools": copy.deepcopy(tools),
            "metadata": {
                "iteration": example.get("iteration"),
                "step": example.get("step"),
                "dataset_mode": dataset_mode,
                "protocol": protocol,
                "labels": labels,
                "state_class": example.get("state_class"),
                "controller": _controller_metadata(
                    example,
                    alias_to_state_id,
                    episode_aliases,
                    hash_aliases,
                ),
                "semantic_field_provenance": policy_provenance,
                "history": {
                    "source_events": alias_metadata["source_history_events"],
                    "exported_events": alias_metadata["exported_history_events"],
                    "max_events": max_history_events,
                    "max_chars": max_history_chars,
                },
            },
        }
        validate_tool_schemas(row["tools"])
        validate_policy_payload(
            json.loads(row["messages"][1]["content"]),
            provenance=policy_provenance,
            require_derived_provenance=require_derived_provenance,
        )
        json.dumps(row, sort_keys=True, allow_nan=False)
        rows.append(row)
    return rows


def dataset_statistics(examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    by_disposition: dict[str, int] = {}
    by_executed_by: dict[str, int] = {}
    for example in examples:
        total += 1
        labels = example.get("labels") or {}
        disposition = labels.get("candidate_disposition") or "none"
        by_disposition[disposition] = by_disposition.get(disposition, 0) + 1
        executed_by = example.get("executed_by", "unknown")
        by_executed_by[executed_by] = by_executed_by.get(executed_by, 0) + 1
    return {
        "total_examples": total,
        "by_candidate_disposition": by_disposition,
        "by_executed_by": by_executed_by,
    }
