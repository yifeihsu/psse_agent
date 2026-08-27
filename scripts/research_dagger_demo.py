"""Small utilities for the research-only DAgger demonstration."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence


SUMMARY_FIELDS = (
    "episodes",
    "resolved_episodes",
    "resolution_rate",
    "terminal_episodes",
    "terminal_rate",
    "invalid_action_count",
    "episodes_with_invalid_actions",
    "loop_episodes",
    "loop_rate",
    "false_commit_count",
    "false_finalization_count",
    "false_rollback_count",
    "healthy_component_corruption_episodes",
)

PAIR_IDENTITY_FIELDS = (
    "suite",
    "scenario_id",
    "physical_root",
    "seed",
    "split",
    "family",
    "cardinality",
    "case",
    "source_tier",
)

TRACE_PREFLIGHT_CONTRACT = "research_dagger_trace_preflight_v2"
PREFLIGHT_DECISION_CONTRACT = "research_dagger_preflight_decision_v2"
REPAIR_CURRICULUM_CONTRACT = "research_dagger_repair_curriculum_v2"

# This research curriculum preserves the broader closed-loop action repertoire
# while explicitly repairing the five decisions missed by the earlier runs.
# Every observable probe is retained once instead of being replayed four times.
# The counts are a focused pilot, not a replacement for the preregistered
# 1,880-row production view.
REPAIR_D0_BUCKETS: tuple[tuple[str, int], ...] = (
    ("wls_from_path", 77),
    ("get_topology_context", 38),
    ("correct_parameters_from_path", 32),
    ("get_measurement_context", 12),
    ("correct_measurements_from_path", 20),
    ("correct_topology_from_path", 14),
    ("get_parameter_context", 8),
    ("commit_state", 12),
    ("finalize_diagnosis", 8),
    ("ask_for_more_evidence", 6),
    ("get_harmonic_context", 2),
    ("run_hse_from_path", 2),
    ("run_three_phase_nlm_from_path", 2),
    ("estimate_hif_location_magnitude_from_path", 2),
    ("estimate_hif_location_magnitude_multiscan_from_path", 2),
)

REPAIR_D1_CATEGORIES: tuple[dict[str, Any], ...] = (
    {"name": "gettopo", "tool": "get_topology_context", "requested": 2},
    {"name": "commit", "tool": "commit_state", "requested": 6},
    {
        "name": "rollback_rejected",
        "tool": "rollback_state",
        "stratum": "rejected_candidate_rollback",
        "requested": 20,
        "expected": 18,
    },
    {
        "name": "wls_escalation",
        "tool": "wls_from_path",
        "stratum": "premature_escalation_recovery",
        "requested": 27,
        "expected": 22,
    },
    {
        "name": "corrmeas",
        "tool": "correct_measurements_from_path",
        "requested": 9,
    },
    {
        "name": "corrparam",
        "tool": "correct_parameters_from_path",
        "requested": 19,
    },
    {
        "name": "wls_invalid",
        "tool": "wls_from_path",
        "stratum": "invalid_precondition_repair",
        "requested": 2,
    },
    {
        "name": "wls_post",
        "tool": "wls_from_path",
        "stratum": "post_failure_no_candidate",
        "requested": 5,
    },
    {
        "name": "ask_loop",
        "tool": "ask_for_more_evidence",
        "stratum": "loop_escape",
        "requested": 5,
    },
    {
        "name": "getmeas_loop",
        "tool": "get_measurement_context",
        "stratum": "loop_escape",
        "requested": 3,
    },
    {
        "name": "rollback_commit",
        "tool": "rollback_state",
        "stratum": "premature_commit_recovery",
        "requested": 44,
    },
    {
        "name": "wls_loop",
        "tool": "wls_from_path",
        "stratum": "loop_escape",
        "requested": 35,
        "expected": 29,
    },
    {
        "name": "getmeas_seq",
        "tool": "get_measurement_context",
        "stratum": "sequential_measurement_parameter_recovery",
        "requested": 45,
    },
    {
        "name": "ask_multi",
        "tool": "ask_for_more_evidence",
        "stratum": "multi_measurement_safe_handoff",
        "requested": 42,
    },
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _physical_root(row: Mapping[str, Any]) -> str:
    direct = row.get("physical_root") or row.get("physical_root_fingerprint")
    if direct:
        return str(direct)
    grouping = row.get("grouping")
    if isinstance(grouping, Mapping):
        return str(
            grouping.get("physical_root")
            or grouping.get("physical_root_fingerprint")
            or ""
        )
    return ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    return rows


def _physical_roots(value: Any) -> set[str]:
    roots: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"physical_root", "physical_root_fingerprint"}:
                root = str(child or "").strip()
                if root:
                    roots.add(root)
            elif isinstance(child, (Mapping, list, tuple)):
                roots.update(_physical_roots(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            roots.update(_physical_roots(child))
    return roots


def _recovery_stratum(row: Mapping[str, Any]) -> str:
    for container in (
        row,
        row.get("metadata"),
        row.get("labels"),
    ):
        if isinstance(container, Mapping):
            value = container.get("recovery_stratum")
            if value is not None and str(value).strip():
                return str(value).strip()
    return "<unspecified>"


def _target_tool(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                call = calls[0]
                function = call.get("function") if isinstance(call, Mapping) else None
                name = function.get("name") if isinstance(function, Mapping) else None
                if name is not None and str(name).strip():
                    return str(name).strip()
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                try:
                    action = json.loads(content)
                except json.JSONDecodeError:
                    action = None
                if isinstance(action, Mapping):
                    name = action.get("tool") or action.get("name")
                    if name is not None and str(name).strip():
                        return str(name).strip()
    return "<missing>"


def _distribution(
    tagged_rows: Sequence[tuple[str, Mapping[str, Any]]],
    getter: Any,
) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    for source, row in tagged_rows:
        value = getter(row)
        overall[value] += 1
        by_source.setdefault(source, Counter())[value] += 1
    return {
        "overall": dict(sorted(overall.items())),
        "by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source.items())
        },
    }


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _trace_preflight_summary(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize trace-preflight rows without applying a release threshold."""

    count = len(results)

    def truth_count(field: str) -> int:
        return sum(row.get(field) is True for row in results)

    def rate(numerator: int) -> float:
        return float(numerator / count) if count else 0.0

    schema = truth_count("schema_valid")
    state_bound = truth_count("state_bound")
    tool_match = truth_count("target_tool_match")
    exact = truth_count("exact_target_match")
    max_token_hits = truth_count("hit_max_new_tokens")
    errors = sum(bool(row.get("error")) for row in results)
    truncated_rows = 0
    truncated_tokens = 0
    for row in results:
        metrics = row.get("action_metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        value = metrics.get("truncated_input_tokens", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            value = 0
        if value > 0:
            truncated_rows += 1
            truncated_tokens += value
    return {
        "row_count": count,
        "schema_valid_count": schema,
        "schema_valid_rate": rate(schema),
        "state_bound_count": state_bound,
        "state_bound_rate": rate(state_bound),
        "target_tool_match_count": tool_match,
        "target_tool_match_rate": rate(tool_match),
        "exact_target_match_count": exact,
        "exact_target_match_rate": rate(exact),
        "input_truncated_row_count": truncated_rows,
        "input_truncated_row_rate": rate(truncated_rows),
        "truncated_input_token_count": truncated_tokens,
        "max_new_token_hit_count": max_token_hits,
        "max_new_token_hit_rate": rate(max_token_hits),
        "error_count": errors,
        "error_rate": rate(errors),
    }


def run_trace_preflight(
    validation_path: Path,
    adapter_path: Path,
    output_path: Path,
    *,
    stop_on_zero_exact: bool = False,
) -> dict[str, Any]:
    """Greedily score every trace-validation target with a research adapter.

    This is diagnostic evidence only.  In particular, it neither samples the
    validation file nor calls or changes the production generation gate.
    """

    import eval_sft_agent_gemma_v4 as eval_runtime
    import gpt_oss_power_sft_revised_v3 as sft_runtime
    import psse_env.dagger.preliminary_e2b_eval as policy_runtime
    import psse_env.dagger.preliminary_tool_gate as gate_runtime
    import psse_env.dagger.protocol_bridge as protocol_runtime
    from psse_env.dagger.preliminary_e2b_eval import (
        _CanonicalE2BPolicy,
        _cached_bundle,
    )
    from psse_env.dagger.preliminary_tool_gate import (
        _policy_state,
        _state_aliases,
        _target_action,
        evaluate_generation,
    )
    from psse_env.dagger.protocol_bridge import unified_tool_schemas
    from psse_env.dagger.release_factories import checkpoint_tree_sha256

    validation = validation_path.expanduser().resolve(strict=True)
    adapter = adapter_path.expanduser().resolve(strict=True)
    validation_digest = _file_sha256(validation)
    rows = _read_jsonl(validation)
    if _file_sha256(validation) != validation_digest:
        raise ValueError("trace preflight validation file changed while it was read")
    if not rows:
        raise ValueError("trace preflight validation file is empty")

    canonical_registry_sha256 = _stable_sha256(unified_tool_schemas())
    prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_ids: set[str] = set()
    roots: set[str] = set()
    for index, row in enumerate(rows, start=1):
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"trace validation row {index} lacks an example_id")
        if example_id in seen_ids:
            raise ValueError(
                f"trace validation example_id values are not unique: {example_id!r}"
            )
        seen_ids.add(example_id)
        if _stable_sha256(row.get("tools")) != canonical_registry_sha256:
            raise ValueError(
                f"trace validation row {example_id!r} tool registry is not canonical"
            )
        # Reuse the exact gate parsers so this diagnostic observes the same
        # prompt, target, and controller-alias contracts as closed-loop use.
        state = _policy_state(row)
        _target_action(row)
        _state_aliases(row)
        root = str(row.get("physical_root_fingerprint") or "").strip()
        if not root:
            raise ValueError(
                f"trace validation row {example_id!r} lacks a physical root"
            )
        roots.add(root)
        prepared.append((row, state))

    adapter_digest = checkpoint_tree_sha256(adapter)
    behavior_sources = {
        "research_dagger_demo": Path(__file__).resolve(),
        "preliminary_e2b_eval": Path(policy_runtime.__file__).resolve(),
        "preliminary_tool_gate": Path(gate_runtime.__file__).resolve(),
        "protocol_bridge": Path(protocol_runtime.__file__).resolve(),
        "eval_sft_agent_gemma_v4": Path(eval_runtime.__file__).resolve(),
        "gpt_oss_power_sft_revised_v3": Path(sft_runtime.__file__).resolve(),
    }
    behavior_binding = {
        "maximum_input_tokens": int(policy_runtime.MAX_INPUT_TOKENS),
        "maximum_new_tokens": int(policy_runtime.MAX_NEW_TOKENS),
        "source_sha256": {
            name: _file_sha256(path) for name, path in sorted(behavior_sources.items())
        },
    }
    behavior_binding_sha256 = _stable_sha256(behavior_binding)
    policy = _CanonicalE2BPolicy(_cached_bundle(str(adapter), adapter_digest))
    results: list[dict[str, Any]] = []
    for row, state in prepared:
        generated_text = ""
        generation_error: str | None = None
        try:
            generated_text = policy.generate_text(state)
            if not isinstance(generated_text, str):
                raise TypeError("trace preflight policy returned non-text generation")
        except Exception as exc:  # Retain every inference failure in the report.
            generation_error = f"{type(exc).__name__}: {exc}"

        try:
            action_metrics = policy.last_action_metrics
        except Exception as exc:  # A broken metrics property must not lose the row.
            action_metrics = {}
            metrics_error = f"{type(exc).__name__}: {exc}"
            generation_error = (
                f"{generation_error}; metrics: {metrics_error}"
                if generation_error
                else f"metrics: {metrics_error}"
            )
        if not isinstance(action_metrics, Mapping):
            generation_error = (
                f"{generation_error}; metrics are not a mapping"
                if generation_error
                else "metrics are not a mapping"
            )
            action_metrics = {}

        try:
            result = evaluate_generation(
                row,
                generated_text,
                action_metrics=action_metrics,
            )
        except Exception as exc:  # Preserve the row even on an evaluator bug.
            result = {
                "example_id": str(row.get("example_id")),
                "physical_root_fingerprint": str(row.get("physical_root_fingerprint")),
                "expected_action": _target_action(row),
                "generated_action": None,
                "bound_internal_action": None,
                "schema_valid": False,
                "state_bound": False,
                "target_tool_match": False,
                "exact_target_match": False,
                "hit_max_new_tokens": action_metrics.get("hit_max_new_tokens") is True,
                "action_metrics": dict(action_metrics),
                "generated_text_sha256": hashlib.sha256(
                    generated_text.encode("utf-8")
                ).hexdigest(),
                "generated_text_preview": generated_text[:240],
                "error": f"{type(exc).__name__}: {exc}",
            }
        if generation_error:
            evaluation_error = result.get("error")
            result["generation_error"] = generation_error
            result["error"] = (
                f"{generation_error}; evaluation: {evaluation_error}"
                if evaluation_error
                else generation_error
            )
        else:
            result["generation_error"] = None
        results.append(result)

    overall = _trace_preflight_summary(results)
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        expected = result.get("expected_action")
        tool = (
            str(expected.get("tool")) if isinstance(expected, Mapping) else "<missing>"
        )
        by_tool.setdefault(tool, []).append(result)
    per_expected_tool = {
        tool: {
            "expected_count": len(tool_results),
            **_trace_preflight_summary(tool_results),
        }
        for tool, tool_results in sorted(by_tool.items())
    }
    zero_exact_stop_triggered = bool(
        stop_on_zero_exact and overall["exact_target_match_count"] == 0
    )
    report = {
        "contract": TRACE_PREFLIGHT_CONTRACT,
        "artifact_type": "research_only_trace_preflight",
        "research_only": True,
        "diagnostic_only": True,
        "release_eligible": False,
        "release_ineligibility_reasons": [
            "research-only learner-trace diagnostic",
            "validation labels are not a production release gate",
        ],
        "selection": {
            "mode": "all_validation_rows",
            "row_count": len(rows),
            "example_ids": [str(row.get("example_id")) for row in rows],
        },
        "adapter_path": str(adapter),
        "adapter_tree_sha256": adapter_digest,
        "validation_file": str(validation),
        "validation_file_sha256": validation_digest,
        "validation_row_count": len(rows),
        "validation_physical_root_count": len(roots),
        "validation_physical_roots": sorted(roots),
        "canonical_tool_registry_sha256": canonical_registry_sha256,
        "behavior_binding": behavior_binding,
        "behavior_binding_sha256": behavior_binding_sha256,
        "overall": overall,
        "per_expected_tool": per_expected_tool,
        "stop_on_zero_exact": bool(stop_on_zero_exact),
        "zero_exact_stop_triggered": zero_exact_stop_triggered,
        "results": results,
    }
    _write_json_atomic(output_path, report)
    return report


def _validate_trace_preflight_report(
    report: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    """Reject stale, incomplete, or internally impossible preflight reports."""

    if report.get("contract") != TRACE_PREFLIGHT_CONTRACT:
        raise ValueError(f"preflight contract mismatch: {path}")
    if not str(report.get("adapter_path") or "").strip():
        raise ValueError(f"preflight report lacks an adapter path: {path}")
    row_count = report.get("validation_row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError(
            f"preflight report has an invalid validation row count: {path}"
        )
    behavior_binding = report.get("behavior_binding")
    behavior_digest = report.get("behavior_binding_sha256")
    if not isinstance(behavior_binding, Mapping) or behavior_digest != _stable_sha256(
        behavior_binding
    ):
        raise ValueError(f"preflight report has an invalid behavior binding: {path}")

    overall = report.get("overall")
    per_tool = report.get("per_expected_tool")
    results = report.get("results")
    if (
        not isinstance(overall, Mapping)
        or not isinstance(per_tool, Mapping)
        or not isinstance(results, list)
    ):
        raise ValueError(f"preflight report is incomplete: {path}")
    if len(results) != row_count or any(
        not isinstance(result, Mapping) for result in results
    ):
        raise ValueError(
            f"preflight report row-level results do not match its validation count: {path}"
        )

    expected_overall = _trace_preflight_summary(results)
    if dict(overall) != expected_overall:
        raise ValueError(
            f"preflight report overall metrics do not reconcile with results: {path}"
        )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        expected_action = result.get("expected_action")
        tool = (
            str(expected_action.get("tool"))
            if isinstance(expected_action, Mapping)
            else "<missing>"
        )
        grouped.setdefault(tool, []).append(result)
    expected_per_tool = {
        tool: {
            "expected_count": len(tool_results),
            **_trace_preflight_summary(tool_results),
        }
        for tool, tool_results in sorted(grouped.items())
    }
    if dict(per_tool) != expected_per_tool:
        raise ValueError(
            f"preflight report per-tool metrics do not reconcile with results: {path}"
        )

    selection = report.get("selection")
    if not isinstance(selection, Mapping) or selection.get("row_count") != row_count:
        raise ValueError(f"preflight report selection count is inconsistent: {path}")
    selected_ids = selection.get("example_ids")
    result_ids = [str(result.get("example_id")) for result in results]
    if selected_ids != result_ids:
        raise ValueError(f"preflight report selection IDs are inconsistent: {path}")


def choose_repair_checkpoint(
    baseline_report_path: Path,
    candidate_report_paths: Sequence[Path],
    output_path: Path,
    *,
    required_tools: Sequence[str],
    minimum_exact: int,
    minimum_schema_rate: float,
    minimum_state_bound_rate: float,
    require_baseline_improvement: bool,
) -> dict[str, Any]:
    """Choose one checkpoint by greedy validation behavior, not train loss."""

    if not candidate_report_paths:
        raise ValueError("checkpoint selection requires at least one candidate report")
    normalized_tools = [str(tool).strip() for tool in required_tools]
    if not normalized_tools or any(not tool for tool in normalized_tools):
        raise ValueError("required preflight tools must be non-empty")
    if len(set(normalized_tools)) != len(normalized_tools):
        raise ValueError("required preflight tools must be unique")
    if minimum_exact < 0:
        raise ValueError("minimum_exact must be non-negative")
    for name, rate in (
        ("minimum_schema_rate", minimum_schema_rate),
        ("minimum_state_bound_rate", minimum_state_bound_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    baseline_path = baseline_report_path.expanduser().resolve(strict=True)
    candidate_paths = [
        path.expanduser().resolve(strict=True) for path in candidate_report_paths
    ]
    baseline = _read_object(baseline_path)
    candidates = [_read_object(path) for path in candidate_paths]
    reports = [(baseline_path, baseline), *zip(candidate_paths, candidates)]
    for path, report in reports:
        _validate_trace_preflight_report(report, path=path)

    binding_fields = (
        "validation_file_sha256",
        "validation_row_count",
        "canonical_tool_registry_sha256",
        "behavior_binding_sha256",
    )
    for path, report in zip(candidate_paths, candidates):
        mismatches = [
            field
            for field in binding_fields
            if report.get(field) != baseline.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"preflight report {path} uses a different validation binding: "
                + ", ".join(mismatches)
            )

    def integer_metric(report: Mapping[str, Any], field: str) -> int:
        value = report["overall"].get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid preflight integer metric {field!r}")
        return value

    def rate_metric(report: Mapping[str, Any], field: str) -> float:
        value = report["overall"].get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid preflight rate metric {field!r}")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"preflight rate metric {field!r} is outside [0, 1]")
        return value

    def tool_exact(report: Mapping[str, Any], tool: str) -> int:
        per_tool = report["per_expected_tool"].get(tool)
        if not isinstance(per_tool, Mapping):
            return 0
        value = per_tool.get("exact_target_match_count", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid exact count for expected tool {tool!r}")
        return value

    summaries: list[dict[str, Any]] = []
    for ordinal, (path, report) in enumerate(zip(candidate_paths, candidates)):
        exact_by_tool = {tool: tool_exact(report, tool) for tool in normalized_tools}
        summaries.append(
            {
                "ordinal": ordinal,
                "report_path": str(path),
                "report_sha256": _file_sha256(path),
                "adapter_path": str(report["adapter_path"]),
                "adapter_tree_sha256": str(report.get("adapter_tree_sha256") or ""),
                "critical_tool_exact_counts": exact_by_tool,
                "critical_tool_coverage": sum(
                    value > 0 for value in exact_by_tool.values()
                ),
                "exact_target_match_count": integer_metric(
                    report, "exact_target_match_count"
                ),
                "target_tool_match_count": integer_metric(
                    report, "target_tool_match_count"
                ),
                "state_bound_count": integer_metric(report, "state_bound_count"),
                "schema_valid_count": integer_metric(report, "schema_valid_count"),
                "state_bound_rate": rate_metric(report, "state_bound_rate"),
                "schema_valid_rate": rate_metric(report, "schema_valid_rate"),
            }
        )

    selected = max(
        summaries,
        key=lambda summary: (
            summary["critical_tool_coverage"],
            summary["exact_target_match_count"],
            summary["target_tool_match_count"],
            summary["state_bound_count"],
            summary["schema_valid_count"],
            -summary["ordinal"],
        ),
    )
    baseline_exact = integer_metric(baseline, "exact_target_match_count")
    failures: list[str] = []
    missing_tools = [
        tool
        for tool, count in selected["critical_tool_exact_counts"].items()
        if count == 0
    ]
    if missing_tools:
        failures.append(
            "zero exact validation actions for required tools: "
            + ", ".join(missing_tools)
        )
    if selected["exact_target_match_count"] < minimum_exact:
        failures.append(
            f"exact validation actions {selected['exact_target_match_count']} "
            f"are below the minimum {minimum_exact}"
        )
    if selected["schema_valid_rate"] < minimum_schema_rate:
        failures.append(
            f"schema-valid rate {selected['schema_valid_rate']:.6f} is below "
            f"{minimum_schema_rate:.6f}"
        )
    if selected["state_bound_rate"] < minimum_state_bound_rate:
        failures.append(
            f"state-bound rate {selected['state_bound_rate']:.6f} is below "
            f"{minimum_state_bound_rate:.6f}"
        )
    if (
        require_baseline_improvement
        and selected["exact_target_match_count"] <= baseline_exact
    ):
        failures.append(
            f"exact validation actions {selected['exact_target_match_count']} "
            f"do not exceed BC0's {baseline_exact}"
        )

    report = {
        "contract": PREFLIGHT_DECISION_CONTRACT,
        "research_only": True,
        "decision": "evaluate" if not failures else "stop_before_closed_loop",
        "passed": not failures,
        "failures": failures,
        "validation_binding": {field: baseline.get(field) for field in binding_fields},
        "required_tools": normalized_tools,
        "thresholds": {
            "minimum_exact_target_matches": minimum_exact,
            "minimum_schema_valid_rate": minimum_schema_rate,
            "minimum_state_bound_rate": minimum_state_bound_rate,
            "require_exact_improvement_over_baseline": require_baseline_improvement,
        },
        "baseline": {
            "report_path": str(baseline_path),
            "report_sha256": _file_sha256(baseline_path),
            "adapter_path": str(baseline["adapter_path"]),
            "adapter_tree_sha256": str(baseline.get("adapter_tree_sha256") or ""),
            "exact_target_match_count": baseline_exact,
        },
        "selected": selected,
        "candidates": summaries,
    }
    _write_json_atomic(output_path, report)
    return report


def _validated_canonical_sft_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Validate canonical chat rows without changing order or row bytes."""

    from psse_env.dagger.preliminary_tool_gate import (
        _policy_state,
        _state_aliases,
        _target_action,
    )
    from psse_env.dagger.protocol_bridge import unified_tool_schemas

    canonical_registry_sha256 = _stable_sha256(unified_tool_schemas())
    identities: dict[str, tuple[str, str, str]] = {}
    validated: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"{label} row {index} lacks an example_id")
        if _stable_sha256(row.get("tools")) != canonical_registry_sha256:
            raise ValueError(
                f"{label} row {example_id!r} tool registry is not canonical"
            )
        _policy_state(row)
        target = _target_action(row)
        _state_aliases(row)
        root = _physical_root(row)
        if not root:
            raise ValueError(f"{label} row {example_id!r} lacks a physical root")
        tool = str(target.get("tool") or "").strip()
        if not tool or tool != _target_tool(row):
            raise ValueError(f"{label} row {example_id!r} has an ambiguous target")
        identity = (root, tool, _stable_sha256(target))
        previous = identities.setdefault(example_id, identity)
        if previous != identity:
            raise ValueError(
                f"{label} repeats example_id {example_id!r} with a different target"
            )
        validated.append(row)
    return validated


def _protected_roots_from_path(path: Path) -> set[str]:
    if path.suffix.lower() == ".jsonl":
        return _physical_roots(_read_jsonl(path))
    return _physical_roots(json.loads(path.read_text(encoding="utf-8")))


def _repair_order_key(row: Mapping[str, Any], *, seed: int, category: str) -> str:
    example_id = str(row.get("example_id") or "")
    return hashlib.sha256(f"{seed}|{category}|{example_id}".encode("utf-8")).hexdigest()


def _canonical_user_content(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("canonical repair row must contain exactly three messages")
    user = messages[1]
    content = user.get("content") if isinstance(user, Mapping) else None
    if not isinstance(content, str):
        raise ValueError("canonical repair row lacks serialized user content")
    return content


def _place_repair_row(
    row: Mapping[str, Any],
    *,
    example_counts: Counter[str],
    root_counts: Counter[str],
    maximum_rows_per_root: int,
) -> bool:
    example_id = str(row.get("example_id") or "")
    root = _physical_root(row)
    if example_counts[example_id] >= 2 or root_counts[root] >= maximum_rows_per_root:
        return False
    example_counts[example_id] += 1
    root_counts[root] += 1
    return True


def _select_d0_bucket(
    rows: Sequence[Mapping[str, Any]],
    *,
    tool: str,
    quota: int,
    seed: int,
    example_counts: Counter[str],
    root_counts: Counter[str],
) -> list[dict[str, Any]]:
    category = f"d0:{tool}"
    candidates = sorted(
        (dict(row) for row in rows if _target_tool(row) == tool),
        key=lambda row: _repair_order_key(row, seed=seed, category=category),
    )
    accepted: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for row in candidates:
        if len(accepted) >= quota:
            break
        prompt = _canonical_user_content(row)
        if prompt in seen_prompts:
            continue
        if not _place_repair_row(
            row,
            example_counts=example_counts,
            root_counts=root_counts,
            maximum_rows_per_root=8,
        ):
            continue
        seen_prompts.add(prompt)
        accepted.append(row)

    replay_basis = list(accepted)
    while len(accepted) < quota:
        progressed = False
        for row in replay_basis:
            if len(accepted) >= quota:
                break
            if not _place_repair_row(
                row,
                example_counts=example_counts,
                root_counts=root_counts,
                maximum_rows_per_root=8,
            ):
                continue
            accepted.append(dict(row))
            progressed = True
        if not progressed:
            break
    if len(accepted) != quota:
        raise ValueError(
            f"D0 repair bucket {tool!r} placed {len(accepted)} of {quota} rows"
        )
    return accepted


def _select_d1_category(
    rows: Sequence[Mapping[str, Any]],
    *,
    specification: Mapping[str, Any],
    seed: int,
    example_counts: Counter[str],
    shared_root_counts: Counter[str],
) -> list[dict[str, Any]]:
    name = str(specification["name"])
    tool = str(specification["tool"])
    stratum = str(specification.get("stratum") or "")
    requested = int(specification["requested"])
    expected = int(specification.get("expected", requested))
    candidates = sorted(
        (
            dict(row)
            for row in rows
            if _target_tool(row) == tool
            and (not stratum or _recovery_stratum(row) == stratum)
        ),
        key=lambda row: _repair_order_key(row, seed=seed, category=f"d1:{name}"),
    )
    accepted: list[dict[str, Any]] = []
    for row in candidates:
        if len(accepted) >= requested:
            break
        if not _place_repair_row(
            row,
            example_counts=example_counts,
            root_counts=shared_root_counts,
            maximum_rows_per_root=8,
        ):
            continue
        accepted.append(row)

    if name == "gettopo" and len(accepted) < requested:
        for row in list(accepted):
            if len(accepted) >= requested:
                break
            if _place_repair_row(
                row,
                example_counts=example_counts,
                root_counts=shared_root_counts,
                maximum_rows_per_root=8,
            ):
                accepted.append(dict(row))
    if len(accepted) != expected:
        raise ValueError(
            f"natural-D1 repair category {name!r} placed {len(accepted)}; "
            f"expected {expected}"
        )
    return accepted


def make_repair_curriculum(
    d0_path: Path,
    natural_paths: Sequence[Path],
    probe_donor_path: Path,
    probe_audit_path: Path,
    output_path: Path,
    report_output_path: Path,
    *,
    protected_paths: Sequence[Path],
    seed: int,
    require_reference_binding: bool = True,
) -> dict[str, Any]:
    """Build the frozen 512-row decision-balanced research curriculum."""

    from psse_env.dagger.offline_teacher_target_audit import (
        OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
    )
    from psse_env.dagger.protocol_bridge import unified_tool_schemas

    if seed != 3407:
        raise ValueError("the frozen repair curriculum requires seed 3407")
    if len(natural_paths) != 2:
        raise ValueError("repair curriculum requires natural train and validation")
    d0_file = d0_path.expanduser().resolve(strict=True)
    natural_files = [path.expanduser().resolve(strict=True) for path in natural_paths]
    donor_file = probe_donor_path.expanduser().resolve(strict=True)
    audit_file = probe_audit_path.expanduser().resolve(strict=True)
    protected_files = [
        path.expanduser().resolve(strict=True) for path in protected_paths
    ]
    input_hashes = {
        "d0": _file_sha256(d0_file),
        "natural_d1": [_file_sha256(path) for path in natural_files],
        "probe_donor": _file_sha256(donor_file),
        "probe_audit": _file_sha256(audit_file),
        "protected": [_file_sha256(path) for path in protected_files],
    }
    reference_inputs = {
        "d0": "28b733db96c6ce05dbdc8d43484bdbb14445e1105958a78f9a35024aa5b3844a",
        "natural_d1": [
            "7d2f7afa851abe894be8b07d99b269c8fbafdb1ce164c29e08c2e832b775e7c1",
            "3723c467de8e557cc3cc628cde5d329602c980b141b1960a153922bd20f7535b",
        ],
        "probe_donor": "7e4d3e2b568f77d19f9dfa05adb85ba91c769b81474fcd13fb80f68ae747c5f5",
        "probe_audit": "6d9e6771b2c430ecd4a2788ed9e6ebb389c70977c7a559eec33c08b09a9a055a",
        "protected": [
            "f05b944f89fa03f61c11376bd05da513f21a7b747c66bfdf80ab11908290898e",
            "68fec0dfe42c6dc4d0df2877633ba494faefcacfe188c44c2cd35b6272b87280",
        ],
    }
    reference_bound = all(
        input_hashes[field] == expected for field, expected in reference_inputs.items()
    )
    if require_reference_binding and not reference_bound:
        mismatches = [
            field
            for field, expected in reference_inputs.items()
            if input_hashes[field] != expected
        ]
        raise ValueError(
            "repair curriculum source hashes do not match the frozen reference: "
            + ", ".join(mismatches)
        )

    protected_roots: set[str] = set()
    protected_inputs: list[dict[str, Any]] = []
    for path, digest in zip(protected_files, input_hashes["protected"]):
        roots = _protected_roots_from_path(path)
        protected_roots.update(roots)
        protected_inputs.append(
            {
                "path": str(path),
                "sha256": digest,
                "physical_root_count": len(roots),
            }
        )

    d0_all = _validated_canonical_sft_rows(_read_jsonl(d0_file), label="D0")
    natural_all: list[dict[str, Any]] = []
    natural_split_by_id: dict[str, str] = {}
    for path in natural_files:
        split_rows = _validated_canonical_sft_rows(
            _read_jsonl(path), label=f"natural D1 {path.name}"
        )
        for row in split_rows:
            natural_split_by_id[str(row["example_id"])] = path.name
        natural_all.extend(split_rows)

    donor_rows = _read_jsonl(donor_file)
    probes: list[dict[str, Any]] = []
    seen_probe_ids: set[str] = set()
    for row in donor_rows:
        if row.get("dataset_source") != "observable_recovery_probe":
            continue
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen_probe_ids:
            continue
        seen_probe_ids.add(example_id)
        probes.append(dict(row))
    probes = _validated_canonical_sft_rows(probes, label="recovery probe donor")
    if len(probes) != 24:
        raise ValueError(f"probe donor retained {len(probes)} unique rows; expected 24")

    audit_rows = _read_jsonl(audit_file)
    audit_by_id = {str(row.get("example_id") or ""): row for row in audit_rows}
    for probe in probes:
        example_id = str(probe["example_id"])
        audit = audit_by_id.get(example_id)
        if not isinstance(audit, Mapping):
            raise ValueError(f"probe {example_id!r} lacks its raw audit source")
        proof = audit.get("observable_rank_one_target_proof")
        private_audit = audit.get("offline_teacher_target_audit")
        if audit.get("auxiliary_training_eligible") is not True:
            raise ValueError(f"probe {example_id!r} is not auxiliary eligible")
        if audit.get("training_decision_evidence_verified") is not True:
            raise ValueError(f"probe {example_id!r} lacks verified training evidence")
        if not isinstance(proof, Mapping) or proof.get("passed") is not True:
            raise ValueError(f"probe {example_id!r} lacks a passed rank-one proof")
        if (
            not isinstance(private_audit, Mapping)
            or private_audit.get("contract") != OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT
            or private_audit.get("passed") is not True
        ):
            raise ValueError(
                f"probe {example_id!r} lacks a current passed private audit"
            )
        if _physical_root(probe) != _physical_root(audit):
            raise ValueError(f"probe {example_id!r} donor/audit root mismatch")
        if _recovery_stratum(probe) != _recovery_stratum(audit):
            raise ValueError(f"probe {example_id!r} donor/audit stratum mismatch")

    probe_roots = {_physical_root(row) for row in probes}
    if len(probe_roots) != 24:
        raise ValueError("repair curriculum requires 24 distinct probe roots")
    probe_tools = dict(sorted(Counter(map(_target_tool, probes)).items()))
    probe_strata = dict(sorted(Counter(map(_recovery_stratum, probes)).items()))
    if probe_tools != {"get_measurement_context": 12, "wls_from_path": 12}:
        raise ValueError(f"unexpected probe tool distribution: {probe_tools}")
    if probe_strata != {
        "post_failure_no_candidate": 12,
        "unsupported_correction_recovery": 12,
    }:
        raise ValueError(f"unexpected probe stratum distribution: {probe_strata}")
    probe_overlap = sorted(probe_roots & protected_roots)
    if probe_overlap:
        raise ValueError(
            "required probes overlap protected roots: " + ", ".join(probe_overlap)
        )

    d0 = [row for row in d0_all if _physical_root(row) not in protected_roots]
    natural = [row for row in natural_all if _physical_root(row) not in protected_roots]
    example_counts: Counter[str] = Counter()
    d0_root_counts: Counter[str] = Counter()
    shared_root_counts: Counter[str] = Counter()
    for row in probes:
        if not _place_repair_row(
            row,
            example_counts=example_counts,
            root_counts=shared_root_counts,
            maximum_rows_per_root=8,
        ):
            raise ValueError("probe reservation violates example/root caps")

    d0_selected: list[dict[str, Any]] = []
    d0_report: list[dict[str, Any]] = []
    for tool, quota in REPAIR_D0_BUCKETS:
        bucket_rows = _select_d0_bucket(
            d0,
            tool=tool,
            quota=quota,
            seed=seed,
            example_counts=example_counts,
            root_counts=d0_root_counts,
        )
        d0_selected.extend(bucket_rows)
        d0_report.append(
            {
                "tool": tool,
                "quota": quota,
                "placed": len(bucket_rows),
                "example_ids": [str(row["example_id"]) for row in bucket_rows],
            }
        )

    natural_selected: list[dict[str, Any]] = []
    natural_report: list[dict[str, Any]] = []
    for specification in REPAIR_D1_CATEGORIES:
        category_rows = _select_d1_category(
            natural,
            specification=specification,
            seed=seed,
            example_counts=example_counts,
            shared_root_counts=shared_root_counts,
        )
        natural_selected.extend(category_rows)
        natural_report.append(
            {
                **dict(specification),
                "placed": len(category_rows),
                "example_ids": [str(row["example_id"]) for row in category_rows],
            }
        )

    output_rows = [*d0_selected, *natural_selected, *probes]
    if (len(d0_selected), len(natural_selected), len(probes)) != (237, 251, 24):
        raise AssertionError("repair source allocation drifted")
    if len(output_rows) != 512:
        raise AssertionError(f"repair curriculum produced {len(output_rows)} rows")
    final_example_counts = Counter(str(row["example_id"]) for row in output_rows)
    final_root_counts = Counter(_physical_root(row) for row in output_rows)
    prompt_counts = Counter(_canonical_user_content(row) for row in output_rows)
    if max(final_example_counts.values()) > 2:
        raise AssertionError("repair curriculum exceeded the two-copy example cap")
    if final_example_counts != example_counts:
        raise AssertionError("repair curriculum example ledger drifted")
    if max(d0_root_counts.values()) > 8 or max(shared_root_counts.values()) > 8:
        raise AssertionError("repair curriculum exceeded a source-space root cap")
    overlap = sorted(set(final_root_counts) & protected_roots)
    if overlap:
        raise AssertionError("protected-root filtering failed: " + ", ".join(overlap))

    expected_tools = {
        "ask_for_more_evidence": 53,
        "commit_state": 18,
        "correct_measurements_from_path": 29,
        "correct_parameters_from_path": 51,
        "correct_topology_from_path": 14,
        "estimate_hif_location_magnitude_from_path": 2,
        "estimate_hif_location_magnitude_multiscan_from_path": 2,
        "finalize_diagnosis": 8,
        "get_harmonic_context": 2,
        "get_measurement_context": 72,
        "get_parameter_context": 8,
        "get_topology_context": 40,
        "rollback_state": 62,
        "run_hse_from_path": 2,
        "run_three_phase_nlm_from_path": 2,
        "wls_from_path": 147,
    }
    actual_tools = dict(sorted(Counter(map(_target_tool, output_rows)).items()))
    if actual_tools != expected_tools:
        raise AssertionError(f"repair tool allocation drifted: {actual_tools}")

    output_sha256 = _canonical_jsonl_sha256(output_rows)
    ordered_ids_sha256 = hashlib.sha256(
        ("\n".join(str(row["example_id"]) for row in output_rows) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    if reference_bound:
        if (
            output_sha256
            != "2f2b12f697224647506117aaee95e914368548706768109c970753e612964a64"
        ):
            raise AssertionError("reference repair-view output hash drifted")
        if (
            ordered_ids_sha256
            != "1a6ff0f6d03e147c9b7b697f5be7bdfa5a4bdca9bfde19ad521f42819de05c18"
        ):
            raise AssertionError("reference repair-view ordered IDs drifted")
    _write_jsonl_atomic(output_path, output_rows)
    if _file_sha256(output_path) != output_sha256:
        raise AssertionError("repair curriculum changed while it was published")
    output_bytes = output_path.stat().st_size

    selected_natural_split = Counter(
        natural_split_by_id[str(row["example_id"])] for row in natural_selected
    )
    source_rows = {
        "d0_bc0": d0_selected,
        "natural_dagger1": natural_selected,
        "observable_recovery_probe": probes,
    }
    source_roots = {
        source: {_physical_root(row) for row in rows}
        for source, rows in source_rows.items()
    }
    source_root_overlap = {
        "d0_natural_dagger1": len(
            source_roots["d0_bc0"] & source_roots["natural_dagger1"]
        ),
        "d0_observable_recovery_probe": len(
            source_roots["d0_bc0"] & source_roots["observable_recovery_probe"]
        ),
        "natural_dagger1_observable_recovery_probe": len(
            source_roots["natural_dagger1"] & source_roots["observable_recovery_probe"]
        ),
    }
    report = {
        "contract": REPAIR_CURRICULUM_CONTRACT,
        "artifact_type": "research_only_decision_balanced_repair_curriculum",
        "research_only": True,
        "release_eligible": False,
        "release_ineligibility_reasons": [
            "decision-balanced repair curriculum, not the canonical Round-1 view",
            "historical natural-D1 validation rows are admitted to training",
            "historical natural-D1 audit metadata uses the superseded v3 contract",
            "production semantic-realizability and immutable-source gates are not rerun",
        ],
        "seed": seed,
        "inputs": {
            "d0": {"path": str(d0_file), "sha256": input_hashes["d0"]},
            "natural_d1": [
                {"path": str(path), "sha256": digest}
                for path, digest in zip(natural_files, input_hashes["natural_d1"])
            ],
            "probe_donor": {
                "path": str(donor_file),
                "sha256": input_hashes["probe_donor"],
            },
            "probe_audit": {
                "path": str(audit_file),
                "sha256": input_hashes["probe_audit"],
            },
            "protected": protected_inputs,
        },
        "reference_binding": {
            "required": bool(require_reference_binding),
            "matched": reference_bound,
            "expected_output_sha256": (
                "2f2b12f697224647506117aaee95e914368548706768109c970753e612964a64"
            ),
            "expected_ordered_example_ids_sha256": (
                "1a6ff0f6d03e147c9b7b697f5be7bdfa5a4bdca9bfde19ad521f42819de05c18"
            ),
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "bytes": output_bytes,
            "ordered_example_ids_sha256": ordered_ids_sha256,
            "canonical_tool_registry_sha256": _stable_sha256(unified_tool_schemas()),
        },
        "counts": {
            "rows": len(output_rows),
            "unique_examples": len(final_example_counts),
            "distinct_canonical_prompts": len(prompt_counts),
            "physical_roots": len(final_root_counts),
            "protected_roots": len(protected_roots),
            "protected_overlap": 0,
            "maximum_example_copies": max(final_example_counts.values()),
            "maximum_canonical_prompt_copies": max(prompt_counts.values()),
            "maximum_d0_rows_per_root": max(d0_root_counts.values()),
            "maximum_shared_d1_probe_rows_per_root": max(shared_root_counts.values()),
            "d0_candidates_excluded_by_protected_root": len(d0_all) - len(d0),
            "natural_candidates_excluded_by_protected_root": (
                len(natural_all) - len(natural)
            ),
        },
        "source_distribution": {
            "d0_bc0": len(d0_selected),
            "natural_dagger1": len(natural_selected),
            "observable_recovery_probe": len(probes),
        },
        "source_physical_root_count": {
            **{source: len(roots) for source, roots in sorted(source_roots.items())},
            "natural_d1_probe_union": len(
                source_roots["natural_dagger1"]
                | source_roots["observable_recovery_probe"]
            ),
        },
        "source_physical_root_overlap": source_root_overlap,
        "source_tool_distribution": {
            source: dict(sorted(Counter(map(_target_tool, rows)).items()))
            for source, rows in sorted(source_rows.items())
        },
        "tool_distribution": actual_tools,
        "natural_source_split_distribution": dict(
            sorted(selected_natural_split.items())
        ),
        "natural_recovery_stratum_distribution": dict(
            sorted(Counter(map(_recovery_stratum, natural_selected)).items())
        ),
        "probe_tool_distribution": probe_tools,
        "probe_recovery_stratum_distribution": probe_strata,
        "probe_audit": {
            "rows_checked": len(probes),
            "auxiliary_eligible": len(probes),
            "training_evidence_verified": len(probes),
            "rank_one_proof_passed": len(probes),
            "private_audit_passed": len(probes),
        },
        "selection": {"d0_buckets": d0_report, "natural_d1": natural_report},
        "caps": {
            "maximum_example_copies": 2,
            "maximum_d0_rows_per_root": 8,
            "maximum_shared_d1_probe_rows_per_root": 8,
            "passed": True,
        },
        "passed": True,
    }
    _write_json_atomic(report_output_path, report)
    return report


def make_train_view(
    d0_path: Path,
    d1_path: Path,
    probes_path: Path,
    output_path: Path,
    *,
    d0_count: int | None,
    probe_repeat: int,
    seed: int,
    protected_suite_path: Path | None,
) -> dict[str, Any]:
    """Build a small, deterministic three-source research training view."""
    from psse_env.dagger.dataset_builder import examples_to_chat_sft

    d0_rows = _read_jsonl(d0_path)
    d1_rows = _read_jsonl(d1_path)
    raw_probes = _read_jsonl(probes_path)

    selected_d0_count = len(d1_rows) if d0_count is None else d0_count
    if selected_d0_count < 0:
        raise ValueError("d0_count must be non-negative")
    if selected_d0_count > len(d0_rows):
        raise ValueError(
            f"requested {selected_d0_count} D0 rows, but only {len(d0_rows)} exist"
        )
    if probe_repeat < 0:
        raise ValueError("probe_repeat must be non-negative")

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(d0_rows)), selected_d0_count))
    selected_d0 = [d0_rows[index] for index in selected_indices]
    exported_probes = examples_to_chat_sft(
        raw_probes,
        protocol="canonical",
        allow_ineligible_auxiliary=True,
    )

    tagged_rows: list[tuple[str, Mapping[str, Any]]] = (
        [("d0", row) for row in selected_d0]
        + [("natural_d1", row) for row in d1_rows]
        + [
            ("recovery_probe", row)
            for _ in range(probe_repeat)
            for row in exported_probes
        ]
    )
    rng.shuffle(tagged_rows)

    training_roots = _physical_roots([row for _, row in tagged_rows])
    protected_roots: set[str] = set()
    if protected_suite_path is not None:
        protected_value = json.loads(protected_suite_path.read_text(encoding="utf-8"))
        protected_roots = _physical_roots(protected_value)
        overlap = sorted(training_roots & protected_roots)
        if overlap:
            raise ValueError(
                "physical-root overlap with protected suite: " + ", ".join(overlap)
            )

    _write_jsonl_atomic(output_path, [row for _, row in tagged_rows])
    return {
        "output": str(output_path),
        "seed": seed,
        "counts": {
            "d0_available": len(d0_rows),
            "d0_selected": len(selected_d0),
            "natural_d1_included": len(d1_rows),
            "raw_recovery_probes": len(raw_probes),
            "exported_recovery_probes": len(exported_probes),
            "probe_repeat": probe_repeat,
            "recovery_probe_rows_included": len(exported_probes) * probe_repeat,
            "total": len(tagged_rows),
        },
        "physical_roots": {
            "training": len(training_roots),
            "protected": len(protected_roots),
            "overlap": 0,
        },
        "recovery_stratum_distribution": _distribution(tagged_rows, _recovery_stratum),
        "tool_distribution": _distribution(tagged_rows, _target_tool),
    }


def make_suite_complement(
    input_path: Path,
    heldout_path: Path,
    output_path: Path,
    *,
    expected_per_suite: int | None,
) -> dict[str, Any]:
    """Remove held-out physical roots from a suite mapping."""
    suites = _read_object(input_path)
    heldout = _read_object(heldout_path)
    if set(suites) != set(heldout):
        raise ValueError("full and held-out suite names differ")

    selected: dict[str, list[dict[str, Any]]] = {}
    heldout_roots: set[str] = set()
    selected_roots: set[str] = set()
    per_suite: dict[str, dict[str, int]] = {}
    for suite_name in sorted(suites):
        full_rows = suites[suite_name]
        heldout_rows = heldout[suite_name]
        if not isinstance(full_rows, list) or not isinstance(heldout_rows, list):
            raise ValueError(f"suite {suite_name!r} is not a list")
        full_objects = [dict(row) for row in full_rows if isinstance(row, Mapping)]
        heldout_objects = [
            dict(row) for row in heldout_rows if isinstance(row, Mapping)
        ]
        if len(full_objects) != len(full_rows) or len(heldout_objects) != len(
            heldout_rows
        ):
            raise ValueError(f"suite {suite_name!r} contains a malformed row")
        full_root_list = [_physical_root(row) for row in full_objects]
        heldout_root_list = [_physical_root(row) for row in heldout_objects]
        if any(not root for root in full_root_list + heldout_root_list):
            raise ValueError(f"suite {suite_name!r} contains a row without a root")
        if len(set(full_root_list)) != len(full_root_list):
            raise ValueError(f"suite {suite_name!r} repeats a full-suite root")
        if len(set(heldout_root_list)) != len(heldout_root_list):
            raise ValueError(f"suite {suite_name!r} repeats a held-out root")
        missing = sorted(set(heldout_root_list) - set(full_root_list))
        if missing:
            raise ValueError(
                f"suite {suite_name!r} held-out roots are absent from full suite: {missing}"
            )
        heldout_set = set(heldout_root_list)
        remainder = [
            row
            for row, root in zip(full_objects, full_root_list)
            if root not in heldout_set
        ]
        if expected_per_suite is not None and len(remainder) != expected_per_suite:
            raise ValueError(
                f"suite {suite_name!r} produced {len(remainder)} rows; "
                f"expected {expected_per_suite}"
            )
        selected[suite_name] = remainder
        suite_selected_roots = {_physical_root(row) for row in remainder}
        overlap = selected_roots & suite_selected_roots
        if overlap:
            raise ValueError(f"selected roots repeat across suites: {sorted(overlap)}")
        selected_roots.update(suite_selected_roots)
        heldout_roots.update(heldout_set)
        per_suite[suite_name] = {
            "full": len(full_objects),
            "heldout": len(heldout_objects),
            "selected": len(remainder),
        }
    overlap = sorted(selected_roots & heldout_roots)
    if overlap:
        raise ValueError(
            "suite complement overlaps held-out roots: " + ", ".join(overlap)
        )
    _write_json_atomic(output_path, selected)
    return {
        "output": str(output_path),
        "suites": len(selected),
        "selected_episodes": sum(len(rows) for rows in selected.values()),
        "selected_roots": len(selected_roots),
        "heldout_roots": len(heldout_roots),
        "root_overlap": 0,
        "per_suite": per_suite,
    }


def make_root_disjoint_suite_split(
    input_path: Path,
    train_output_path: Path,
    heldout_output_path: Path,
    *,
    heldout_per_suite: int,
) -> dict[str, Any]:
    """Make the smallest deterministic physical-root-closure holdout."""
    if heldout_per_suite <= 0:
        raise ValueError("heldout_per_suite must be positive")
    suites = _read_object(input_path)

    normalized: dict[str, list[dict[str, Any]]] = {}
    root_suites: dict[str, set[str]] = {}
    for suite_name in sorted(suites):
        rows = suites[suite_name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"suite {suite_name!r} is empty or not a list")
        objects = [dict(row) for row in rows if isinstance(row, Mapping)]
        if len(objects) != len(rows):
            raise ValueError(f"suite {suite_name!r} contains a malformed row")
        roots = [_physical_root(row) for row in objects]
        if any(not root for root in roots):
            raise ValueError(f"suite {suite_name!r} contains a row without a root")
        if len(set(roots)) != len(roots):
            raise ValueError(f"suite {suite_name!r} repeats a physical root")
        if len(objects) <= heldout_per_suite:
            raise ValueError(f"suite {suite_name!r} has no room for training rows")
        normalized[suite_name] = objects
        for root in roots:
            root_suites.setdefault(root, set()).add(suite_name)

    ordered_roots = sorted(root_suites)
    heldout_roots: set[str] | None = None
    for size in range(1, len(ordered_roots)):
        for candidate_tuple in itertools.combinations(ordered_roots, size):
            candidate = set(candidate_tuple)
            heldout_counts = {
                suite_name: sum(_physical_root(row) in candidate for row in rows)
                for suite_name, rows in normalized.items()
            }
            if all(count == heldout_per_suite for count in heldout_counts.values()):
                heldout_roots = candidate
                break
        if heldout_roots is not None:
            break
    if heldout_roots is None:
        raise ValueError(
            "no physical-root closure gives exactly "
            f"{heldout_per_suite} held-out rows per suite"
        )

    train: dict[str, list[dict[str, Any]]] = {}
    heldout: dict[str, list[dict[str, Any]]] = {}
    per_suite: dict[str, dict[str, int]] = {}
    for suite_name, rows in normalized.items():
        heldout[suite_name] = [
            row for row in rows if _physical_root(row) in heldout_roots
        ]
        train[suite_name] = [
            row for row in rows if _physical_root(row) not in heldout_roots
        ]
        if len(heldout[suite_name]) != heldout_per_suite or not train[suite_name]:
            raise ValueError(f"suite {suite_name!r} violates the requested split")
        per_suite[suite_name] = {
            "train": len(train[suite_name]),
            "heldout": len(heldout[suite_name]),
        }

    train_roots = _physical_roots(train)
    actual_heldout_roots = _physical_roots(heldout)
    overlap = sorted(train_roots & actual_heldout_roots)
    if overlap:
        raise ValueError("root-disjoint split overlaps: " + ", ".join(overlap))
    if actual_heldout_roots != heldout_roots:
        raise ValueError("root-closure split lost a selected held-out root")

    _write_json_atomic(train_output_path, train)
    _write_json_atomic(heldout_output_path, heldout)
    return {
        "contract": "research_root_disjoint_suite_split_v1",
        "input": str(input_path),
        "train_output": str(train_output_path),
        "heldout_output": str(heldout_output_path),
        "suites": len(normalized),
        "heldout_per_suite": heldout_per_suite,
        "train_episodes": sum(len(rows) for rows in train.values()),
        "heldout_episodes": sum(len(rows) for rows in heldout.values()),
        "train_roots": len(train_roots),
        "heldout_roots": len(actual_heldout_roots),
        "root_overlap": 0,
        "heldout_root_ids": sorted(actual_heldout_roots),
        "per_suite": per_suite,
    }


def _trace_example(
    episode: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    ordinal: int,
) -> dict[str, Any] | None:
    assessment = row.get("objective_action_assessment")
    observation = row.get("policy_observation")
    if not isinstance(assessment, Mapping) or not isinstance(observation, Mapping):
        return None
    expected = assessment.get("expected_action")
    classified_stratum = assessment.get("recovery_stratum")
    stratum = (
        classified_stratum.strip()
        if isinstance(classified_stratum, str) and classified_stratum.strip()
        else "learner_visited_recovery"
    )
    if (
        assessment.get("contract") != "dagger_study_objective_action_assessment_v1"
        or assessment.get("evidence_available") is not True
        or assessment.get("policy_payload_leakage_paths") != []
        or not isinstance(expected, Mapping)
    ):
        return None
    if set(expected) != {"tool", "arguments"} or not isinstance(
        expected.get("arguments"), Mapping
    ):
        return None
    episode_key = str(episode.get("episode_key") or "episode")
    scenario_id = episode.get("scenario_id")
    return {
        "example_id": f"research_trace:{episode_key}:{ordinal}",
        "scenario_id": scenario_id,
        "root_scenario_id": scenario_id,
        "physical_root_fingerprint": episode.get("physical_root"),
        "dataset_source": "research_dagger_learner_trace",
        "collector_contract": "research_dagger_learner_trace_v1",
        "state_origin": "learner_policy",
        "collection_role": "research_dagger_recovery",
        "state_visited_by": "learner_policy",
        "replay_source": "learner_policy",
        "auxiliary_training_eligible": True,
        "natural_on_policy_support_eligible": True,
        # The observable expert fixed this target, but the research path does
        # not pretend that a production private target audit was run.
        "training_decision_evidence_verified": False,
        "recovery_stratum": stratum,
        "production_label_eligible": False,
        "policy_observation": dict(observation),
        "preferred_action": dict(expected),
        "valid_next_actions": [dict(expected)],
        "scenario_family": episode.get("family"),
        "error_cardinality": episode.get("cardinality"),
        "network_case": episode.get("case"),
        "source_tier": episode.get("source_tier"),
        "episode_terminal_outcome": episode.get("terminal_outcome"),
        "labels": {
            "dataset_mode": "research_dagger_learner_trace",
            "learner_action": row.get("action"),
            "learner_execution_status": row.get("execution_status"),
            "learner_error_code": row.get("error_code"),
        },
    }


def make_trace_dagger_view(
    artifact_path: Path,
    d0_path: Path,
    train_output_path: Path,
    validation_output_path: Path,
    *,
    report_output_path: Path | None,
    protected_suite_path: Path | None,
    validation_roots_per_suite: int,
    max_rows_per_episode: int,
    dagger_repeat: int,
    d0_count: int | None,
    seed: int,
) -> dict[str, Any]:
    """Export observable expert labels on states visited by the learner."""
    from psse_env.dagger.dataset_builder import examples_to_chat_sft

    if validation_roots_per_suite <= 0:
        raise ValueError("validation_roots_per_suite must be positive")
    if max_rows_per_episode <= 0:
        raise ValueError("max_rows_per_episode must be positive")
    if dagger_repeat <= 0:
        raise ValueError("dagger_repeat must be positive")

    artifact = _read_object(artifact_path)
    _overall, indexed = _evaluation_view(artifact, label="learner rollout")
    by_suite: dict[str, list[Mapping[str, Any]]] = {}
    for episode in indexed.values():
        suite = str(episode.get("suite") or "")
        root = str(episode.get("physical_root") or "")
        if not suite or not root:
            raise ValueError("learner rollout episode lacks suite/root identity")
        by_suite.setdefault(suite, []).append(episode)

    root_suites: dict[str, set[str]] = {}
    suite_roots: dict[str, set[str]] = {}
    for suite, episodes in sorted(by_suite.items()):
        roots = {str(episode.get("physical_root")) for episode in episodes}
        if len(roots) <= validation_roots_per_suite:
            raise ValueError(f"suite {suite!r} has no room for train roots")
        suite_roots[suite] = roots
        for root in roots:
            root_suites.setdefault(root, set()).add(suite)

    # Select whole physical roots, not individual suite rows. This prevents a
    # root reused by several recovery suites from straddling train/validation.
    ordered_roots = sorted(root_suites)
    validation_roots: set[str] | None = None
    for size in range(1, len(ordered_roots)):
        for candidate_tuple in itertools.combinations(ordered_roots, size):
            candidate = set(candidate_tuple)
            if all(
                len(roots & candidate) >= validation_roots_per_suite
                and bool(roots - candidate)
                for roots in suite_roots.values()
            ):
                validation_roots = candidate
                break
        if validation_roots is not None:
            break
    if validation_roots is None:
        raise ValueError("cannot create a root-disjoint trace train/validation split")

    validation_keys = {
        str(episode.get("episode_key"))
        for episode in indexed.values()
        if str(episode.get("physical_root")) in validation_roots
    }

    protected_roots: set[str] = set()
    if protected_suite_path is not None:
        protected_roots = _physical_roots(
            json.loads(protected_suite_path.read_text(encoding="utf-8"))
        )
    artifact_roots = {str(episode.get("physical_root")) for episode in indexed.values()}
    protected_overlap = sorted(artifact_roots & protected_roots)
    if protected_overlap:
        raise ValueError(
            "learner rollout overlaps protected suite roots: "
            + ", ".join(protected_overlap)
        )

    raw_train: list[dict[str, Any]] = []
    raw_validation: list[dict[str, Any]] = []
    skipped_rows = 0
    seen_payloads: set[str] = set()
    for episode_key in sorted(indexed):
        episode = indexed[episode_key]
        policy_rows = _policy_rows(episode)[:max_rows_per_episode]
        destination = raw_validation if episode_key in validation_keys else raw_train
        for ordinal, row in enumerate(policy_rows):
            example = _trace_example(episode, row, ordinal=ordinal)
            if example is None:
                skipped_rows += 1
                continue
            identity = json.dumps(
                {
                    "root": example.get("physical_root_fingerprint"),
                    "observation": example.get("policy_observation"),
                    "target": example.get("preferred_action"),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if identity in seen_payloads:
                skipped_rows += 1
                continue
            seen_payloads.add(identity)
            destination.append(example)
    if not raw_train or not raw_validation:
        raise ValueError("learner rollout produced an empty train/validation partition")

    exported_train = examples_to_chat_sft(
        raw_train,
        protocol="canonical",
        allow_ineligible_auxiliary=True,
    )
    exported_validation = examples_to_chat_sft(
        raw_validation,
        protocol="canonical",
        allow_ineligible_auxiliary=True,
    )
    if len(exported_train) != len(raw_train) or len(exported_validation) != len(
        raw_validation
    ):
        raise ValueError("canonical trace export dropped a learner state")

    d0_rows = _read_jsonl(d0_path)
    excluded_d0_roots = protected_roots | _physical_roots(exported_validation)
    d0_candidates = [
        row
        for row in d0_rows
        if not (root := _physical_root(row)) or root not in excluded_d0_roots
    ]
    selected_d0_count = len(exported_train) if d0_count is None else d0_count
    if selected_d0_count < 0 or selected_d0_count > len(d0_candidates):
        raise ValueError(
            f"d0_count must be in 0..{len(d0_candidates)}, got {selected_d0_count}"
        )
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(d0_candidates)), selected_d0_count))
    selected_d0 = [d0_candidates[index] for index in selected_indices]
    tagged_train: list[tuple[str, Mapping[str, Any]]] = [
        ("d0", row) for row in selected_d0
    ] + [("learner_trace", row) for _ in range(dagger_repeat) for row in exported_train]
    rng.shuffle(tagged_train)
    rng.shuffle(exported_validation)
    _write_jsonl_atomic(train_output_path, [row for _, row in tagged_train])
    _write_jsonl_atomic(validation_output_path, exported_validation)

    trace_train_roots = _physical_roots(exported_train)
    d0_train_roots = _physical_roots(selected_d0)
    train_roots = trace_train_roots | d0_train_roots
    validation_roots = _physical_roots(exported_validation)
    if train_roots & validation_roots:
        raise ValueError("trace train/validation physical roots overlap")
    if train_roots & protected_roots:
        raise ValueError("research training view overlaps protected roots")
    report = {
        "contract": "research_dagger_learner_trace_v1",
        "artifact": str(artifact_path),
        "train_output": str(train_output_path),
        "validation_output": str(validation_output_path),
        "seed": seed,
        "counts": {
            "rollout_episodes": len(indexed),
            "raw_train_learner_rows": len(raw_train),
            "raw_validation_learner_rows": len(raw_validation),
            "skipped_or_duplicate_policy_rows": skipped_rows,
            "d0_available": len(d0_rows),
            "d0_excluded_by_root": len(d0_rows) - len(d0_candidates),
            "d0_selected": selected_d0_count,
            "dagger_repeat": dagger_repeat,
            "train_rows": len(tagged_train),
            "validation_rows": len(exported_validation),
        },
        "roots": {
            "rollout": len(artifact_roots),
            "train": len(train_roots),
            "trace_train": len(trace_train_roots),
            "d0_train": len(d0_train_roots),
            "validation": len(validation_roots),
            "protected": len(protected_roots),
            "pairwise_overlap": 0,
        },
        "recovery_stratum_distribution": _distribution(tagged_train, _recovery_stratum),
        "tool_distribution": _distribution(tagged_train, _target_tool),
    }
    if report_output_path is not None:
        _write_json_atomic(report_output_path, report)
    return report


def make_subset(input_path: Path, output_path: Path, per_suite: int) -> dict[str, Any]:
    suites = _read_object(input_path)
    if per_suite <= 0:
        raise ValueError("per_suite must be positive")

    selected: dict[str, list[dict[str, Any]]] = {}
    used_roots: set[str] = set()
    for suite_name in sorted(suites):
        rows = suites[suite_name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"suite {suite_name!r} is empty or not a list")
        candidates = [row for row in rows if isinstance(row, dict)]
        preferred = [row for row in candidates if _physical_root(row) not in used_roots]
        remainder = [row for row in candidates if row not in preferred]
        chosen = (preferred + remainder)[:per_suite]
        if len(chosen) != per_suite:
            raise ValueError(
                f"suite {suite_name!r} has {len(chosen)} rows; need {per_suite}"
            )
        selected[suite_name] = chosen
        used_roots.update(root for row in chosen if (root := _physical_root(row)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output_path),
        "suites": len(selected),
        "episodes": sum(len(rows) for rows in selected.values()),
        "distinct_physical_roots": len(used_roots),
        "per_suite": per_suite,
    }


def _evaluation_view(
    artifact: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    evaluation = artifact.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{label} artifact lacks evaluation object")
    suite_metrics = evaluation.get("suite_metrics")
    if not isinstance(suite_metrics, Mapping):
        raise ValueError(f"{label} artifact lacks suite_metrics")
    overall = suite_metrics.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError(f"{label} artifact lacks overall metrics")
    rows = suite_metrics.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} artifact lacks nonempty episodes")
    indexed: dict[str, Mapping[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} episode[{ordinal}] is not an object")
        key = row.get("episode_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} episode[{ordinal}] lacks episode_key")
        if key in indexed:
            raise ValueError(f"{label} has duplicate episode_key {key!r}")
        indexed[key] = row
    if overall.get("episodes") != len(indexed):
        raise ValueError(f"{label} overall episode count disagrees with episodes")
    return dict(overall), indexed


def _paired_episodes(
    bc0: Mapping[str, Mapping[str, Any]],
    dagger: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if set(bc0) != set(dagger):
        raise ValueError(
            "paired episode mismatch: "
            f"BC0-only={sorted(set(bc0) - set(dagger))}, "
            f"DAgger-only={sorted(set(dagger) - set(bc0))}"
        )
    for key in sorted(bc0):
        mismatched = [
            name
            for name in PAIR_IDENTITY_FIELDS
            if bc0[key].get(name) != dagger[key].get(name)
        ]
        if mismatched:
            raise ValueError(
                f"paired episode {key!r} differs in {', '.join(mismatched)}"
            )
    return dict(bc0), dict(dagger)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _policy_rows(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trace = episode.get("trace")
    if not isinstance(trace, list):
        raise ValueError(f"episode {episode.get('episode_key')!r} lacks trace")
    rows = [row for row in trace if isinstance(row, Mapping)]
    if len(rows) != len(trace):
        raise ValueError(f"episode {episode.get('episode_key')!r} has malformed trace")
    if any(row.get("intervention") not in {True, False} for row in rows):
        raise ValueError(
            f"episode {episode.get('episode_key')!r} has invalid intervention flags"
        )
    policy = [row for row in rows if row.get("intervention") is False]
    recorded = episode.get("policy_steps")
    if isinstance(recorded, bool) or not isinstance(recorded, int):
        raise ValueError(f"episode {episode.get('episode_key')!r} lacks policy_steps")
    if recorded != len(policy):
        raise ValueError(
            f"episode {episode.get('episode_key')!r} policy_steps/trace mismatch"
        )
    return policy


def _research_metrics(episodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    counts = {
        name: 0
        for name in (
            "policy",
            "schema",
            "executable",
            "opportunity",
            "evaluable",
            "exact",
            "exact_executed",
            "evaluator_error",
            "control_quarantine",
            "audit_quarantine",
            "any_quarantine",
            "false_commit",
            "false_finalization",
            "false_rollback",
            "corruption",
            "unknown_safety",
        )
    }
    for episode in episodes.values():
        policy = _policy_rows(episode)
        trace = episode["trace"]
        intervention = episode.get("evaluation_intervention")
        intervention = intervention if isinstance(intervention, Mapping) else {}
        pre_steps = intervention.get("pre_policy_step_count")
        if (
            isinstance(pre_steps, bool)
            or not isinstance(pre_steps, int)
            or pre_steps < 0
            or not policy
            or pre_steps >= len(trace)
            or trace[pre_steps] is not policy[0]
            or any(row.get("intervention") is not True for row in trace[:pre_steps])
        ):
            raise ValueError(
                f"episode {episode.get('episode_key')!r} has an invalid first "
                "post-intervention policy row"
            )

        counts["policy"] += len(policy)
        for row in policy:
            action = row.get("action")
            valid = bool(
                isinstance(action, Mapping)
                and isinstance(action.get("tool"), str)
                and action.get("tool") != "__invalid_action__"
                and isinstance(action.get("arguments"), Mapping)
            )
            counts["schema"] += valid
            counts["executable"] += valid and row.get("execution_status") == "success"

        first = policy[0]
        assessment = first.get("objective_action_assessment")
        assessment = assessment if isinstance(assessment, Mapping) else {}
        stratum = assessment.get("recovery_stratum")
        opportunity = isinstance(stratum, str) and bool(stratum.strip())
        expected = assessment.get("expected_action")
        evaluable = bool(
            opportunity
            and assessment.get("evidence_available") is True
            and isinstance(expected, Mapping)
        )
        exact = bool(evaluable and first.get("action") == expected)
        counts["opportunity"] += opportunity
        counts["evaluable"] += evaluable
        counts["exact"] += exact
        counts["exact_executed"] += exact and first.get("execution_status") == "success"

        control = episode.get("control_quarantine")
        audit = episode.get("audit")
        control = isinstance(control, Mapping) and control.get("quarantined") is True
        audit = isinstance(audit, Mapping) and audit.get("quarantined") is True
        counts["control_quarantine"] += control
        counts["audit_quarantine"] += audit
        counts["any_quarantine"] += control or audit
        counts["evaluator_error"] += episode.get("evaluator_error") is not None
        for field, name in (
            ("false_commit_count", "false_commit"),
            ("false_finalization_count", "false_finalization"),
            ("false_rollback_count", "false_rollback"),
        ):
            value = episode.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"episode metric {field!r} is missing or invalid")
            counts[name] += value
        known = episode.get("healthy_preservation_known") is True
        counts["corruption"] += (
            known and episode.get("healthy_components_preserved") is False
        )
        counts["unknown_safety"] += not known

    return {
        "policy_action_count": counts["policy"],
        "schema_valid_action_count": counts["schema"],
        "schema_valid_action_rate": _rate(counts["schema"], counts["policy"]),
        "executable_action_count": counts["executable"],
        "executable_action_rate": _rate(counts["executable"], counts["policy"]),
        "mean_policy_steps": _rate(counts["policy"], total),
        "first_recovery_action_opportunity_episodes": counts["opportunity"],
        "first_recovery_action_evaluable_episodes": counts["evaluable"],
        "first_recovery_action_unevaluable_episodes": total - counts["evaluable"],
        "first_recovery_action_exact_correct_episodes": counts["exact"],
        "first_recovery_action_exact_accuracy": _rate(
            counts["exact"], counts["evaluable"]
        ),
        "first_recovery_action_correct_and_executed_episodes": counts["exact_executed"],
        "first_recovery_action_correct_and_executed_rate": _rate(
            counts["exact_executed"], counts["evaluable"]
        ),
        "evaluator_error_episodes": counts["evaluator_error"],
        "control_quarantined_episodes": counts["control_quarantine"],
        "audit_quarantined_episodes": counts["audit_quarantine"],
        "control_or_audit_quarantined_episodes": counts["any_quarantine"],
        "false_commit_count": counts["false_commit"],
        "false_finalization_count": counts["false_finalization"],
        "false_rollback_count": counts["false_rollback"],
        "healthy_component_corruption_episodes": counts["corruption"],
        "healthy_component_safety_unknown_episodes": counts["unknown_safety"],
    }


def _difference(left: Any, right: Any) -> float | int | None:
    if left is None or right is None:
        return None
    return left - right


def summarize(
    bc0_path: Path, dagger_path: Path, output_path: Path | None
) -> dict[str, Any]:
    bc0_artifact = _read_object(bc0_path)
    dagger_artifact = _read_object(dagger_path)
    bc0, raw_bc0_episodes = _evaluation_view(bc0_artifact, label="BC0")
    dagger, raw_dagger_episodes = _evaluation_view(dagger_artifact, label="DAgger")
    bc0_episodes, dagger_episodes = _paired_episodes(
        raw_bc0_episodes, raw_dagger_episodes
    )
    bc0_research = _research_metrics(bc0_episodes)
    dagger_research = _research_metrics(dagger_episodes)
    report = {
        "prototype": True,
        "pairing": {
            "matched": True,
            "matched_episode_count": len(bc0_episodes),
            "episode_keys": sorted(bc0_episodes),
            "asserted_identity_fields": list(PAIR_IDENTITY_FIELDS),
        },
        "metric_definitions": {
            "first_recovery_action": (
                "exact equality to the evaluator's objective expected_action on "
                "the first post-intervention policy row"
            ),
            "correct_and_executed": (
                "first recovery action is exactly correct and execution_status is success"
            ),
            "schema_valid_action": "normalized policy tool is not __invalid_action__",
            "executable_action": (
                "schema-valid policy action with execution_status success"
            ),
            "healthy_component_safety_unknown": (
                "healthy_preservation_known is not true"
            ),
        },
        "bc0": {
            **{name: bc0.get(name) for name in SUMMARY_FIELDS},
            **bc0_research,
        },
        "dagger": {
            **{name: dagger.get(name) for name in SUMMARY_FIELDS},
            **dagger_research,
        },
        "delta": {
            "resolution_rate": float(dagger.get("resolution_rate") or 0.0)
            - float(bc0.get("resolution_rate") or 0.0),
            "terminal_rate": float(dagger.get("terminal_rate") or 0.0)
            - float(bc0.get("terminal_rate") or 0.0),
            "invalid_action_count": int(dagger.get("invalid_action_count") or 0)
            - int(bc0.get("invalid_action_count") or 0),
            "loop_rate": float(dagger.get("loop_rate") or 0.0)
            - float(bc0.get("loop_rate") or 0.0),
            **{
                name: _difference(dagger_research[name], bc0_research[name])
                for name in (
                    "first_recovery_action_exact_accuracy",
                    "first_recovery_action_correct_and_executed_rate",
                    "schema_valid_action_rate",
                    "executable_action_rate",
                    "mean_policy_steps",
                    "evaluator_error_episodes",
                    "control_or_audit_quarantined_episodes",
                    "false_commit_count",
                    "false_finalization_count",
                    "false_rollback_count",
                    "healthy_component_corruption_episodes",
                    "healthy_component_safety_unknown_episodes",
                )
            },
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("subset", help="select a small balanced suite")
    subset.add_argument("--input", required=True, type=Path)
    subset.add_argument("--output", required=True, type=Path)
    subset.add_argument("--per-suite", type=int, default=1)

    summary = subparsers.add_parser("summary", help="compare BC0 and DAgger metrics")
    summary.add_argument("--bc0", required=True, type=Path)
    summary.add_argument("--dagger", required=True, type=Path)
    summary.add_argument("--output", type=Path)

    train_view = subparsers.add_parser(
        "train-view", help="build a lightweight three-source research SFT view"
    )
    train_view.add_argument("--d0", required=True, type=Path)
    train_view.add_argument("--d1", required=True, type=Path)
    train_view.add_argument("--probes", required=True, type=Path)
    train_view.add_argument("--output", required=True, type=Path)
    train_view.add_argument(
        "--d0-count",
        type=int,
        help="D0 rows to sample without replacement (default: number of D1 rows)",
    )
    train_view.add_argument("--probe-repeat", type=int, default=4)
    train_view.add_argument("--seed", type=int, default=3407)
    train_view.add_argument(
        "--protected-suite",
        type=Path,
        help="optional JSON suite; only physical-root overlap is rejected",
    )

    complement = subparsers.add_parser(
        "suite-complement", help="remove held-out roots from a suite mapping"
    )
    complement.add_argument("--input", required=True, type=Path)
    complement.add_argument("--heldout", required=True, type=Path)
    complement.add_argument("--output", required=True, type=Path)
    complement.add_argument("--expected-per-suite", type=int)

    root_split = subparsers.add_parser(
        "root-split",
        help="make deterministic train/holdout physical-root closures",
    )
    root_split.add_argument("--input", required=True, type=Path)
    root_split.add_argument("--train-output", required=True, type=Path)
    root_split.add_argument("--heldout-output", required=True, type=Path)
    root_split.add_argument("--heldout-per-suite", required=True, type=int)

    trace_view = subparsers.add_parser(
        "trace-view",
        help="export observable expert targets on learner-visited states",
    )
    trace_view.add_argument("--artifact", required=True, type=Path)
    trace_view.add_argument("--d0", required=True, type=Path)
    trace_view.add_argument("--train-output", required=True, type=Path)
    trace_view.add_argument("--validation-output", required=True, type=Path)
    trace_view.add_argument("--report-output", type=Path)
    trace_view.add_argument("--protected-suite", type=Path)
    trace_view.add_argument("--validation-roots-per-suite", type=int, default=1)
    trace_view.add_argument("--max-rows-per-episode", type=int, default=4)
    trace_view.add_argument("--dagger-repeat", type=int, default=2)
    trace_view.add_argument("--d0-count", type=int)
    trace_view.add_argument("--seed", type=int, default=3407)

    repair_curriculum = subparsers.add_parser(
        "repair-curriculum",
        help="build the fixed 512-row decision-balanced repair curriculum",
    )
    repair_curriculum.add_argument("--d0", required=True, type=Path)
    repair_curriculum.add_argument(
        "--natural",
        required=True,
        action="append",
        type=Path,
        help="canonical natural-D1 chat-SFT file; repeat for historical splits",
    )
    repair_curriculum.add_argument("--probe-donor", required=True, type=Path)
    repair_curriculum.add_argument("--probe-audit", required=True, type=Path)
    repair_curriculum.add_argument("--output", required=True, type=Path)
    repair_curriculum.add_argument("--report-output", required=True, type=Path)
    repair_curriculum.add_argument(
        "--protected",
        action="append",
        default=[],
        type=Path,
        help="JSON/JSONL roots excluded from training; repeat as needed",
    )
    repair_curriculum.add_argument("--seed", type=int, default=3407)

    trace_preflight = subparsers.add_parser(
        "trace-preflight",
        help="score every trace-validation row before a closed-loop replay",
    )
    trace_preflight.add_argument("--validation", required=True, type=Path)
    trace_preflight.add_argument("--adapter", required=True, type=Path)
    trace_preflight.add_argument("--output", required=True, type=Path)
    trace_preflight.add_argument(
        "--stop-on-zero-exact",
        action="store_true",
        help="write the diagnostic report, then exit nonzero when exact count is zero",
    )

    preflight_decision = subparsers.add_parser(
        "preflight-decision",
        help="select a repair checkpoint and decide whether closed-loop eval is justified",
    )
    preflight_decision.add_argument("--baseline", required=True, type=Path)
    preflight_decision.add_argument(
        "--candidate", required=True, action="append", type=Path
    )
    preflight_decision.add_argument("--output", required=True, type=Path)
    preflight_decision.add_argument("--required-tool", action="append", default=[])
    preflight_decision.add_argument("--minimum-exact", type=int, default=5)
    preflight_decision.add_argument("--minimum-schema-rate", type=float, default=0.95)
    preflight_decision.add_argument(
        "--minimum-state-bound-rate", type=float, default=0.90
    )
    preflight_decision.add_argument(
        "--allow-no-baseline-improvement",
        action="store_true",
        help="do not require more exact actions than BC0",
    )
    preflight_decision.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="write the decision report, then exit 2 when evaluation is rejected",
    )

    args = parser.parse_args(argv)
    exit_code = 0
    if args.command == "subset":
        result = make_subset(args.input, args.output, args.per_suite)
    elif args.command == "summary":
        result = summarize(args.bc0, args.dagger, args.output)
    elif args.command == "train-view":
        result = make_train_view(
            args.d0,
            args.d1,
            args.probes,
            args.output,
            d0_count=args.d0_count,
            probe_repeat=args.probe_repeat,
            seed=args.seed,
            protected_suite_path=args.protected_suite,
        )
    elif args.command == "suite-complement":
        result = make_suite_complement(
            args.input,
            args.heldout,
            args.output,
            expected_per_suite=args.expected_per_suite,
        )
    elif args.command == "root-split":
        result = make_root_disjoint_suite_split(
            args.input,
            args.train_output,
            args.heldout_output,
            heldout_per_suite=args.heldout_per_suite,
        )
    elif args.command == "trace-view":
        result = make_trace_dagger_view(
            args.artifact,
            args.d0,
            args.train_output,
            args.validation_output,
            report_output_path=args.report_output,
            protected_suite_path=args.protected_suite,
            validation_roots_per_suite=args.validation_roots_per_suite,
            max_rows_per_episode=args.max_rows_per_episode,
            dagger_repeat=args.dagger_repeat,
            d0_count=args.d0_count,
            seed=args.seed,
        )
    elif args.command == "repair-curriculum":
        result = make_repair_curriculum(
            args.d0,
            args.natural,
            args.probe_donor,
            args.probe_audit,
            args.output,
            args.report_output,
            protected_paths=args.protected,
            seed=args.seed,
        )
    elif args.command == "trace-preflight":
        result = run_trace_preflight(
            args.validation,
            args.adapter,
            args.output,
            stop_on_zero_exact=args.stop_on_zero_exact,
        )
        if result["zero_exact_stop_triggered"] is True:
            exit_code = 2
    else:  # preflight-decision
        required_tools = args.required_tool or [
            "wls_from_path",
            "rollback_state",
            "correct_parameters_from_path",
            "get_measurement_context",
            "get_topology_context",
        ]
        result = choose_repair_checkpoint(
            args.baseline,
            args.candidate,
            args.output,
            required_tools=required_tools,
            minimum_exact=args.minimum_exact,
            minimum_schema_rate=args.minimum_schema_rate,
            minimum_state_bound_rate=args.minimum_state_bound_rate,
            require_baseline_improvement=not args.allow_no_baseline_improvement,
        )
        if args.stop_on_fail and result["passed"] is not True:
            exit_code = 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
