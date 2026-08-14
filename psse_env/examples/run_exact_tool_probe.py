from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import psse_env.dagger.dataset_builder as dataset_builder
from psse_env.dagger import TOOL_JSON_SCHEMAS
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.sft import audit_dataset, load_exact_processor
from psse_env.sft.provenance import (
    file_sha256,
    git_source_state,
    stable_json_sha256,
)


PINNED_MODELS = (
    (
        "unsloth/gemma-4-31B-it",
        "8a796db4df380b178065ed910849477ff0e99c87",
    ),
    (
        "unsloth/gemma-4-E2B-it",
        "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae",
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _minimal_value(schema: dict[str, Any], *, key: str) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list):
        non_null = [value for value in enum if value is not None]
        if non_null:
            return copy.deepcopy(non_null[0])
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    types = [value for value in types if value != "null"]
    selected = types[0] if types else "null"
    if selected == "string":
        if key in {"candidate_state_id"}:
            return "candidate"
        if key in {"state_id", "case_path", "scan_window_path"}:
            return "active"
        return f"probe_{key}"
    if selected == "integer":
        return 1 if key in {"line_index", "line_index1"} else 0
    if selected == "number":
        return 1.0
    if selected == "boolean":
        return True
    if selected == "array":
        items = schema.get("items")
        return [_minimal_value(dict(items), key=key)] if isinstance(items, dict) else []
    if selected == "object":
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        return {
            required_key: _minimal_value(dict(properties[required_key]), key=required_key)
            for required_key in schema.get("required", [])
        }
    return None


def _registry(protocol: str) -> list[dict[str, Any]]:
    if protocol == "canonical":
        return unified_tool_schemas()
    if protocol == "controller":
        return copy.deepcopy(TOOL_JSON_SCHEMAS)
    raise ValueError("protocol must be canonical or controller")


def target_arguments_from_registry(
    protocol: str = "canonical",
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Generate one valid target from every current registered macro schema."""
    targets: list[tuple[str, dict[str, Any]]] = []
    for schema in _registry(protocol):
        function = schema["function"]
        parameters = function["parameters"]
        arguments = _minimal_value(parameters, key="arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool {function['name']} does not have object arguments.")
        targets.append((str(function["name"]), arguments))
    return tuple(targets)


def build_rows(protocol: str = "canonical") -> list[dict[str, Any]]:
    registry = _registry(protocol)
    rows: list[dict[str, Any]] = []
    for index, (name, arguments) in enumerate(
        target_arguments_from_registry(protocol)
    ):
        rows.append(
            {
                "example_id": f"exact-tool-probe-{protocol}-{index:02d}-{name}",
                "root_scenario_id": f"probe-{protocol}-{index:02d}",
                "tools": copy.deepcopy(registry),
                "messages": [
                    {
                        "role": "system",
                        "content": "Choose exactly one valid PSSE diagnostic tool call.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "episode_id": "e",
                                "active_state_id": "active",
                                "candidate_state_id": "candidate",
                                "probe_target": name,
                            },
                            sort_keys=True,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": copy.deepcopy(arguments),
                                },
                            }
                        ],
                    },
                ],
                "metadata": {
                    "state_class": "exact_tool_surface_probe",
                    "protocol": protocol,
                },
            }
        )
    return rows


def run(
    *,
    local_files_only: bool = True,
    max_length: int = 4096,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    rows_by_protocol = {
        protocol: build_rows(protocol) for protocol in ("canonical", "controller")
    }
    rows = [
        row for protocol_rows in rows_by_protocol.values() for row in protocol_rows
    ]
    source_state = git_source_state(_REPO_ROOT)
    if require_clean_source and not source_state.get("release_eligible_source"):
        raise RuntimeError(
            "Exact processor evidence must be generated from a clean tracked worktree; "
            "use --allow-dirty-source only for a non-release development probe."
        )
    coverage: dict[str, Any] = {}
    for protocol in ("canonical", "controller"):
        registered_tools = [
            schema["function"]["name"] for schema in _registry(protocol)
        ]
        covered_tools = [
            name for name, _ in target_arguments_from_registry(protocol)
        ]
        missing_tools = sorted(set(registered_tools) - set(covered_tools))
        extra_tools = sorted(set(covered_tools) - set(registered_tools))
        passed = (
            not missing_tools
            and not extra_tools
            and len(covered_tools) == len(set(covered_tools))
        )
        coverage[protocol] = {
            "registered_tool_count": len(registered_tools),
            "covered_tool_count": len(covered_tools),
            "missing_tools": missing_tools,
            "extra_tools": extra_tools,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                f"Exact {protocol} probe does not cover its registry exactly once: "
                f"missing={missing_tools}, extra={extra_tools}."
            )
    processors: dict[str, Any] = {}
    for model, revision in PINNED_MODELS:
        processor, loader = load_exact_processor(
            model,
            revision,
            local_files_only=local_files_only,
        )
        report = audit_dataset(
            rows,
            processor,
            max_length=max_length,
            require_current_registry=True,
        )
        report_payload = report.to_dict()
        processors[model] = {
            "revision": revision,
            "processor_loader": loader,
            "passed": report.passed,
            "tool_round_trips": report.tool_round_trips,
            "length_audit": report_payload["length_audit"],
            "failures": report.failures,
        }
        if not report.passed:
            raise RuntimeError(f"Exact tool-surface probe failed for {model}: {report.failures[:3]}")
    return {
        "rows": len(rows),
        "adversarial_episode_id": "e",
        "tool_targets": {
            protocol: [
                name for name, _ in target_arguments_from_registry(protocol)
            ]
            for protocol in ("canonical", "controller")
        },
        "tool_coverage": coverage,
        "provenance": {
            **source_state,
            "schema_registry_hashes": {
                protocol: stable_json_sha256(_registry(protocol))
                for protocol in ("canonical", "controller")
            },
            "exporter_hash": file_sha256(dataset_builder.__file__),
            "probe_source_hash": file_sha256(__file__),
            "generated_probe_rows_hash": stable_json_sha256(rows),
            "processor_revisions": {
                model: revision for model, revision in PINNED_MODELS
            },
        },
        "release_eligible": bool(source_state.get("release_eligible_source")),
        "processors": processors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every DAgger macro target through pinned Gemma 4 processors."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sft_pilot"
        / "all_tools_exact_processor_probe.json",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Development-only probe; generated report is not release eligible.",
    )
    args = parser.parse_args()
    report = run(
        local_files_only=not args.allow_download,
        max_length=args.max_length,
        require_clean_source=not args.allow_dirty_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
