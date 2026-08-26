"""Research-only exact-action preflight for Gemma-4-12B adapters.

The input is an exported SFT validation JSONL.  Each row is normalized back to
its model-visible state and rendered by :class:`ResearchGemmaPolicy`, so this
preflight cannot accidentally preserve a stale serialized chat template.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from psse_env.dagger.dataset_builder import validate_policy_payload
from psse_env.dagger.preliminary_tool_gate import _policy_state, _target_action
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.dagger.release_factories import _validated_generated_action
from psse_env.research_models import GEMMA4_12B
from psse_env.sft.research_rows import normalize_research_rows


PREFLIGHT_CONTRACT = "research_gemma4_12b_action_preflight_v1"
COMPARISON_CONTRACT = "research_gemma4_12b_action_preflight_comparison_v1"
DEFAULT_MINIMUM_EXACT = 5


def load_validation_rows(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL and reject missing or duplicate example IDs."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"validation JSONL has a blank line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"validation row {line_number} is not an object")
            row = dict(value)
            example_id = str(row.get("example_id") or "").strip()
            if not example_id:
                raise ValueError(f"validation row {line_number} lacks an example_id")
            if example_id in seen:
                raise ValueError(f"duplicate validation example_id: {example_id!r}")
            seen.add(example_id)
            rows.append(row)
    if not rows:
        raise ValueError("validation JSONL is empty")
    return rows


def normalize_validation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the current model observation and canonical action target.

    ``_policy_state`` deliberately parses the JSON payload rather than passing
    the stored chat text to a processor.  The live 12B policy therefore renders
    that state using its current native prompt contract.
    """

    example_id = str(row.get("example_id") or "").strip()
    if not example_id:
        raise ValueError("validation row lacks an example_id")
    state = _policy_state(row)
    validate_policy_payload({"state": state})
    return {
        "example_id": example_id,
        "physical_root_fingerprint": str(
            row.get("physical_root_fingerprint") or ""
        ).strip(),
        "state": state,
        "expected_action": _target_action(row),
    }


def _canonical_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("policy action is not an object")
    tool = value.get("tool", value.get("name"))
    arguments = value.get("arguments")
    text = json.dumps(
        {"name": tool, "arguments": arguments},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    schemas = {
        str(item["function"]["name"]): item["function"]["parameters"]
        for item in unified_tool_schemas()
    }
    return _validated_generated_action(text, schemas)


def _policy_action(policy: Any, state: Mapping[str, Any]) -> Any:
    method = getattr(policy, "act_model_observation", None)
    if not callable(method):
        method = getattr(policy, "act", None)
    if not callable(method):
        raise TypeError("policy lacks act_model_observation() or act()")
    return method(copy.deepcopy(dict(state)))


def _safe_metrics(policy: Any) -> dict[str, Any]:
    value = getattr(policy, "last_action_metrics", {})
    if not isinstance(value, Mapping):
        raise TypeError("policy last_action_metrics is not an object")
    return copy.deepcopy(dict(value))


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(results)

    def count_true(field: str) -> int:
        return sum(row.get(field) is True for row in results)

    def rate(value: int) -> float:
        return float(value / count) if count else 0.0

    schema = count_true("schema_valid")
    tool = count_true("tool_match")
    exact = count_true("exact_match")
    errors = sum(bool(row.get("error")) for row in results)
    input_truncated = count_true("input_truncated")
    output_truncated = count_true("hit_max_new_tokens")
    truncated = sum(
        row.get("input_truncated") is True
        or row.get("hit_max_new_tokens") is True
        for row in results
    )
    truncated_tokens = sum(
        int(row.get("truncated_input_tokens") or 0) for row in results
    )
    expected_tools = Counter(
        str(row.get("expected_action", {}).get("tool", "<missing>"))
        for row in results
    )
    generated_tools = Counter(
        str(row.get("generated_action", {}).get("tool", "<invalid>"))
        if isinstance(row.get("generated_action"), Mapping)
        else "<invalid>"
        for row in results
    )
    return {
        "example_count": count,
        "schema_valid_count": schema,
        "schema_valid_rate": rate(schema),
        "tool_match_count": tool,
        "tool_match_rate": rate(tool),
        "exact_count": exact,
        "exact_rate": rate(exact),
        "error_count": errors,
        "error_rate": rate(errors),
        "truncation_count": truncated,
        "truncation_rate": rate(truncated),
        "input_truncated_count": input_truncated,
        "output_max_token_hit_count": output_truncated,
        "truncated_input_token_count": truncated_tokens,
        "expected_action_tool_counts": dict(sorted(expected_tools.items())),
        "generated_action_tool_counts": dict(sorted(generated_tools.items())),
    }


def score_validation_rows(
    rows: Sequence[Mapping[str, Any]], policy: Any
) -> dict[str, Any]:
    """Score all rows with a policy; suitable for deterministic fake policies."""

    prompt_rows, normalization = normalize_research_rows(
        rows, source_label="research_action_preflight_validation"
    )
    prepared = [normalize_validation_row(row) for row in prompt_rows]
    ids = [row["example_id"] for row in prepared]
    if len(set(ids)) != len(ids):
        raise ValueError("validation example_id values are not unique")
    results: list[dict[str, Any]] = []
    for row in prepared:
        generated: dict[str, Any] | None = None
        metrics: dict[str, Any] = {}
        error: str | None = None
        try:
            generated = _canonical_action(_policy_action(policy, row["state"]))
            metrics = _safe_metrics(policy)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                metrics = _safe_metrics(policy)
            except Exception as metrics_exc:
                error += f"; metrics: {type(metrics_exc).__name__}: {metrics_exc}"
        truncated_tokens = metrics.get("truncated_input_tokens", 0)
        if isinstance(truncated_tokens, bool) or not isinstance(truncated_tokens, int):
            truncated_tokens = 0
        truncated_tokens = max(0, truncated_tokens)
        expected = row["expected_action"]
        results.append(
            {
                "example_id": row["example_id"],
                "physical_root_fingerprint": row["physical_root_fingerprint"],
                "expected_action": expected,
                "generated_action": generated,
                "schema_valid": generated is not None,
                "tool_match": bool(
                    generated is not None
                    and generated.get("tool") == expected.get("tool")
                ),
                "exact_match": generated == expected,
                "input_truncated": truncated_tokens > 0,
                "truncated_input_tokens": truncated_tokens,
                "hit_max_new_tokens": metrics.get("hit_max_new_tokens") is True,
                "action_metrics": metrics,
                "error": error,
            }
        )
    summary = summarize_results(results)
    return {
        "contract": PREFLIGHT_CONTRACT,
        "research_only": True,
        "model": {
            "model_id": GEMMA4_12B.model_id,
            "revision": GEMMA4_12B.revision,
            "architecture": GEMMA4_12B.architecture,
            "prompt_profile": GEMMA4_12B.prompt_profile,
        },
        "prompt_normalization": normalization,
        "example_ids": ids,
        "summary": summary,
        # Common scalar aliases make reports easy to gate without knowing the
        # complete summary schema.
        **{key: summary[key] for key in (
            "example_count", "schema_valid_count", "schema_valid_rate",
            "tool_match_count", "tool_match_rate", "exact_count", "exact_rate",
            "error_count", "truncation_count",
        )},
        "results": results,
    }


def compare_preflight_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_exact: int = DEFAULT_MINIMUM_EXACT,
) -> dict[str, Any]:
    """Apply the bounded research checkpoint promotion gate."""

    if minimum_exact < 0:
        raise ValueError("minimum_exact must be non-negative")
    baseline_ids = list(baseline.get("example_ids") or [])
    candidate_ids = list(candidate.get("example_ids") or [])
    same_examples = bool(
        baseline_ids
        and len(baseline_ids) == len(set(baseline_ids))
        and len(candidate_ids) == len(set(candidate_ids))
        and set(baseline_ids) == set(candidate_ids)
    )
    baseline_results = baseline.get("results")
    candidate_results = candidate.get("results")
    baseline_results = baseline_results if isinstance(baseline_results, list) else []
    candidate_results = candidate_results if isinstance(candidate_results, list) else []
    baseline_actions = {
        str(row.get("example_id")): row.get("generated_action")
        for row in baseline_results if isinstance(row, Mapping)
    }
    candidate_actions = {
        str(row.get("example_id")): row.get("generated_action")
        for row in candidate_results if isinstance(row, Mapping)
    }
    changed_ids = [
        example_id for example_id in candidate_ids
        if example_id in baseline_actions
        and example_id in candidate_actions
        and baseline_actions[example_id] != candidate_actions[example_id]
    ] if same_examples else []
    baseline_summary = baseline.get("summary")
    candidate_summary = candidate.get("summary")
    baseline_summary = baseline_summary if isinstance(baseline_summary, Mapping) else {}
    candidate_summary = candidate_summary if isinstance(candidate_summary, Mapping) else {}
    baseline_exact = int(baseline_summary.get("exact_count") or 0)
    candidate_exact = int(candidate_summary.get("exact_count") or 0)
    checks = {
        "same_examples": same_examples,
        "candidate_schema_100_percent": (
            int(candidate_summary.get("example_count") or 0) > 0
            and candidate_summary.get("schema_valid_count")
            == candidate_summary.get("example_count")
        ),
        "candidate_zero_errors": int(candidate_summary.get("error_count") or 0) == 0,
        "candidate_zero_truncation": int(
            candidate_summary.get("truncation_count") or 0
        ) == 0,
        "candidate_minimum_exact": candidate_exact >= minimum_exact,
        "candidate_improves_exact": candidate_exact > baseline_exact,
        "generated_action_changed": bool(changed_ids),
    }
    return {
        "contract": COMPARISON_CONTRACT,
        "research_only": True,
        "minimum_exact": minimum_exact,
        "baseline_exact_count": baseline_exact,
        "candidate_exact_count": candidate_exact,
        "exact_count_delta": candidate_exact - baseline_exact,
        "changed_action_count": len(changed_ids),
        "changed_example_ids": changed_ids,
        "checks": checks,
        "passed": all(checks.values()),
        "failure_reasons": [name for name, passed in checks.items() if not passed],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_preflight(
    validation_path: Path,
    adapter_path: Path,
    *,
    policy_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if policy_loader is None:
        from psse_env.dagger.research_policy_factory import research_gemma_policy_factory

        policy_loader = research_gemma_policy_factory
    policy = policy_loader(
        adapter_path.expanduser().resolve(strict=True),
        base_model=GEMMA4_12B.model_id,
        base_revision=GEMMA4_12B.revision,
        architecture=GEMMA4_12B.architecture,
        prompt_profile=GEMMA4_12B.prompt_profile,
        load_in_4bit=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    report = score_validation_rows(load_validation_rows(validation_path), policy)
    report["validation_path"] = str(validation_path.expanduser().resolve())
    report["adapter_path"] = str(adapter_path.expanduser().resolve())
    return report


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"report is not an object: {path}")
    return dict(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="score one adapter")
    run.add_argument("--validation", required=True, type=Path)
    run.add_argument("--adapter", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    compare = subparsers.add_parser("compare", help="compare two preflight reports")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--minimum-exact", type=int, default=DEFAULT_MINIMUM_EXACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        report = run_preflight(args.validation, args.adapter)
        _write_json(args.output, report)
        return 0
    report = compare_preflight_reports(
        _read_report(args.baseline),
        _read_report(args.candidate),
        minimum_exact=args.minimum_exact,
    )
    _write_json(args.output, report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
