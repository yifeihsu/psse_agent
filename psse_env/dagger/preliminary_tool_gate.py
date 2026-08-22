"""Fail-closed generation gate for the diagnostic Gemma 4 E2B fast path.

Finite held-out loss does not prove that an adapter can emit an executable
tool call.  This gate greedily decodes a deterministic, root-spread subset of
the preliminary D1 validation split and requires schema-valid, controller-
bound calls before a preliminary checkpoint may receive its completion
receipt.  It is explicitly non-release evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ``bind_controller_action`` lives with the SFT alias contract. Importing it
# here (rather than accepting a permissive callable) keeps the gate tied to the
# same executable binding used by closed-loop inference.
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    bind_controller_action,
    validate_policy_payload,
)
from psse_env.dagger.protocol_bridge import (
    canonical_to_internal_action,
    unified_tool_schemas,
)
from psse_env.dagger.release_factories import (
    _validated_generated_action,
    checkpoint_tree_sha256,
)
from psse_env.sft.gates import GateError


GATE_CONTRACT = "preliminary_e2b_tool_generation_gate_v1"
GATE_ARTIFACT_TYPE = "preliminary_dagger_nonrelease_generation_gate"
GATE_FILENAME = "preliminary_tool_generation_gate.json"
REQUIRED_CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS = 64
SAMPLE_COUNT = 32
SELECTION_SEED = 3407
MINIMUM_DISTINCT_ROOTS = 10
MINIMUM_SCHEMA_VALID_RATE = 0.99
MINIMUM_STATE_BOUND_RATE = 0.98
MINIMUM_BC0_TARGET_TOOL_RATE = 0.0
MINIMUM_DAGGER_TARGET_TOOL_RATE = 0.50


class PreliminaryToolGateError(ValueError):
    """The preliminary generation gate could not establish its contract."""


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreliminaryToolGateError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PreliminaryToolGateError(f"{label} must be one JSON object")
    return dict(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PreliminaryToolGateError(
                        f"validation JSONL contains a blank line at {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise PreliminaryToolGateError(
                        f"validation row {line_number} is not an object"
                    )
                rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreliminaryToolGateError(f"cannot read validation JSONL: {exc}") from exc
    if len(rows) < SAMPLE_COUNT:
        raise PreliminaryToolGateError(
            f"generation gate requires at least {SAMPLE_COUNT} validation rows"
        )
    return rows


def _target_action(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise PreliminaryToolGateError(
            "gate rows must contain exactly system, user, and assistant messages"
        )
    system, user, target = messages
    if not isinstance(system, Mapping) or system.get("role") != "system":
        raise PreliminaryToolGateError("gate row lacks its canonical system message")
    if system.get("content") != CANONICAL_DAGGER_SYSTEM_PROMPT:
        raise PreliminaryToolGateError("gate row system prompt is not canonical")
    if not isinstance(user, Mapping) or user.get("role") != "user":
        raise PreliminaryToolGateError("gate row lacks its user observation")
    if not isinstance(target, Mapping) or target.get("role") != "assistant":
        raise PreliminaryToolGateError("gate row lacks its assistant target")
    calls = target.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise PreliminaryToolGateError("gate target must contain exactly one tool call")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        raise PreliminaryToolGateError("gate target tool call lacks a function")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
        raise PreliminaryToolGateError("gate target function is malformed")
    return {"tool": name, "arguments": copy.deepcopy(dict(arguments))}


def _policy_state(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    content = messages[1].get("content")
    if not isinstance(content, str):
        raise PreliminaryToolGateError("gate user content must be canonical JSON text")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PreliminaryToolGateError(f"gate user content is invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"state"}:
        raise PreliminaryToolGateError("gate user payload must contain only state")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise PreliminaryToolGateError("gate user state must be an object")
    canonical_content = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if content != canonical_content:
        raise PreliminaryToolGateError(
            "gate user JSON bytes differ from closed-loop prompt serialization"
        )
    validate_policy_payload(payload)
    return copy.deepcopy(dict(state))


def _state_aliases(row: Mapping[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata")
    controller = metadata.get("controller") if isinstance(metadata, Mapping) else None
    aliases = controller.get("state_aliases") if isinstance(controller, Mapping) else None
    if not isinstance(aliases, Mapping) or not aliases:
        raise PreliminaryToolGateError("gate row lacks controller state aliases")
    normalized = {str(alias): str(value) for alias, value in aliases.items()}
    if "active" not in normalized:
        raise PreliminaryToolGateError("gate row lacks the active controller alias")
    return normalized


def _physical_root(row: Mapping[str, Any]) -> str:
    root = str(row.get("physical_root_fingerprint") or "").strip()
    if not root:
        raise PreliminaryToolGateError("gate row lacks a physical root")
    return root


def _example_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("example_id") or "").strip()
    if not value:
        raise PreliminaryToolGateError("gate row lacks an example_id")
    return value


def _target_reference_class(action: Mapping[str, Any]) -> str:
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    for key in ("case_path", "scan_window_path"):
        if key not in arguments:
            continue
        value = str(arguments[key])
        return value if value in {"active", "candidate"} else "historical"
    return "no_reference"


def _row_rank(row: Mapping[str, Any], *, seed: int) -> tuple[str, str]:
    example_id = _example_id(row)
    return (
        _stable_sha256(
            {
                "contract": GATE_CONTRACT,
                "purpose": "validation_selection",
                "seed": int(seed),
                "example_id": example_id,
                "physical_root_fingerprint": _physical_root(row),
            }
        ),
        example_id,
    )


def select_gate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_count: int = SAMPLE_COUNT,
    seed: int = SELECTION_SEED,
) -> list[dict[str, Any]]:
    """Select a deterministic subset covering roots, tools, and alias classes."""

    if sample_count < SAMPLE_COUNT or sample_count > len(rows):
        raise PreliminaryToolGateError(
            f"sample_count must be in [{SAMPLE_COUNT}, {len(rows)}]"
        )
    prepared: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for row in rows:
        example_id = _example_id(row)
        if example_id in seen_ids:
            raise PreliminaryToolGateError("validation example_id values are not unique")
        seen_ids.add(example_id)
        action = _target_action(row)
        _policy_state(row)
        _state_aliases(row)
        _physical_root(row)
        prepared.append((row, action))

    selected: dict[str, Mapping[str, Any]] = {}

    def add_first(groups: Iterable[Iterable[tuple[Mapping[str, Any], dict[str, Any]]]]) -> None:
        for group in groups:
            candidates = sorted(group, key=lambda item: _row_rank(item[0], seed=seed))
            if candidates:
                selected.setdefault(_example_id(candidates[0][0]), candidates[0][0])

    by_root: dict[str, list[tuple[Mapping[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_tool: dict[str, list[tuple[Mapping[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_reference: dict[str, list[tuple[Mapping[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in prepared:
        row, action = item
        by_root[_physical_root(row)].append(item)
        by_tool[str(action["tool"])].append(item)
        by_reference[_target_reference_class(action)].append(item)

    # Independent roots are the first coverage axis, followed by every target
    # tool and every state-reference class present in validation.
    add_first(by_root[root] for root in sorted(by_root))
    add_first(by_tool[tool] for tool in sorted(by_tool))
    add_first(by_reference[name] for name in sorted(by_reference))
    if len(selected) > sample_count:
        raise PreliminaryToolGateError(
            "sample_count is too small for required root/tool/reference coverage"
        )
    for row, _action in sorted(prepared, key=lambda item: _row_rank(item[0], seed=seed)):
        if len(selected) >= sample_count:
            break
        selected.setdefault(_example_id(row), row)
    result = [copy.deepcopy(dict(row)) for row in selected.values()]
    if len(result) != sample_count:
        raise PreliminaryToolGateError("could not select the requested gate sample")
    if len({_physical_root(row) for row in result}) < MINIMUM_DISTINCT_ROOTS:
        raise PreliminaryToolGateError(
            f"gate sample has fewer than {MINIMUM_DISTINCT_ROOTS} physical roots"
        )
    return result


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_generation(
    row: Mapping[str, Any],
    generated_text: str,
    *,
    action_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one raw generation against schema, aliases, and its target."""

    expected = _target_action(row)
    aliases = _state_aliases(row)
    tools = row.get("tools")
    if _stable_sha256(tools) != _stable_sha256(unified_tool_schemas()):
        raise PreliminaryToolGateError("gate row tool registry differs from runtime")
    parameter_schemas = {
        str(tool["function"]["name"]): tool["function"]["parameters"]
        for tool in tools
    }
    schema_valid = False
    state_bound = False
    generated: dict[str, Any] | None = None
    bound_action: dict[str, Any] | None = None
    error: str | None = None
    try:
        generated = _validated_generated_action(generated_text, parameter_schemas)
        schema_valid = True
        internal = canonical_to_internal_action(generated)
        bound_action = bind_controller_action(internal, aliases)
        state_bound = True
    except (GateError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    metrics = dict(action_metrics or {})
    hit_max = metrics.get("hit_max_new_tokens") is True
    return {
        "example_id": _example_id(row),
        "physical_root_fingerprint": _physical_root(row),
        "expected_action": expected,
        "generated_action": generated,
        "bound_internal_action": bound_action,
        "schema_valid": schema_valid,
        "state_bound": state_bound,
        "target_tool_match": bool(
            generated is not None and generated.get("tool") == expected["tool"]
        ),
        "exact_target_match": generated == expected,
        "hit_max_new_tokens": hit_max,
        "action_metrics": metrics,
        "generated_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
        "generated_text_preview": generated_text[:240],
        "error": error,
    }


def _minimum_target_tool_rate(stage: str) -> float:
    if stage == "bc0":
        return MINIMUM_BC0_TARGET_TOOL_RATE
    if stage == "dagger":
        return MINIMUM_DAGGER_TARGET_TOOL_RATE
    raise PreliminaryToolGateError(f"unsupported gate stage: {stage!r}")


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    minimum_target_tool_rate: float = MINIMUM_DAGGER_TARGET_TOOL_RATE,
) -> dict[str, Any]:
    count = len(results)
    if count != SAMPLE_COUNT:
        raise PreliminaryToolGateError(
            f"gate summary requires exactly {SAMPLE_COUNT} results"
        )
    schema = sum(row.get("schema_valid") is True for row in results)
    bound = sum(row.get("state_bound") is True for row in results)
    tool_match = sum(row.get("target_tool_match") is True for row in results)
    exact = sum(row.get("exact_target_match") is True for row in results)
    max_hits = sum(row.get("hit_max_new_tokens") is True for row in results)
    distinct_roots = len(
        {str(row.get("physical_root_fingerprint")) for row in results}
    )
    summary = {
        "sample_count": count,
        "distinct_physical_roots": distinct_roots,
        "schema_valid_count": schema,
        "schema_valid_rate": _rate(schema, count),
        "state_bound_count": bound,
        "state_bound_rate": _rate(bound, count),
        "target_tool_match_count": tool_match,
        "target_tool_match_rate": _rate(tool_match, count),
        "exact_target_match_count": exact,
        "exact_target_match_rate": _rate(exact, count),
        "max_new_token_hit_count": max_hits,
    }
    summary["passed"] = bool(
        distinct_roots >= MINIMUM_DISTINCT_ROOTS
        and summary["schema_valid_rate"] >= MINIMUM_SCHEMA_VALID_RATE
        and summary["state_bound_rate"] >= MINIMUM_STATE_BOUND_RATE
        and summary["target_tool_match_rate"] >= minimum_target_tool_rate
        and max_hits == 0
    )
    return summary


def gate_plan_contract(stage: str) -> dict[str, Any]:
    return {
        "contract": GATE_CONTRACT,
        "sample_count": SAMPLE_COUNT,
        "selection_seed": SELECTION_SEED,
        "minimum_distinct_physical_roots": MINIMUM_DISTINCT_ROOTS,
        "minimum_schema_valid_rate": MINIMUM_SCHEMA_VALID_RATE,
        "minimum_state_bound_rate": MINIMUM_STATE_BOUND_RATE,
        "minimum_target_tool_match_rate": _minimum_target_tool_rate(stage),
        "max_new_tokens": MAX_NEW_TOKENS,
        "required_context_tokens": REQUIRED_CONTEXT_TOKENS,
        "decoding": "greedy",
    }


def _validate_stage_plan(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreliminaryToolGateError("stage plan must be a regular file")
    plan = _load_json_object(path.resolve(strict=True), label="stage plan")
    if plan.get("contract") != "preliminary_dagger_training_stage_plan_v1":
        raise PreliminaryToolGateError("stage plan contract is not preliminary v1")
    if plan.get("release_eligible") is not False:
        raise PreliminaryToolGateError("stage plan must be release_eligible=false")
    stage = str(plan.get("stage") or "")
    if plan.get("generation_gate") != gate_plan_contract(stage):
        raise PreliminaryToolGateError("stage plan generation gate differs from code")
    training = plan.get("training_arguments")
    if not isinstance(training, Mapping) or training.get("max_seq_length") != REQUIRED_CONTEXT_TOKENS:
        raise PreliminaryToolGateError(
            f"stage plan must use {REQUIRED_CONTEXT_TOKENS} context tokens"
        )
    return plan


def _atomic_write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PreliminaryToolGateError(f"gate report already exists: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.chmod(path, 0o400)
    except FileExistsError as exc:
        raise PreliminaryToolGateError(f"gate report publication raced: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_gate(stage_plan_path: Path) -> tuple[Path, dict[str, Any]]:
    """Run the bound validation gate and publish its non-release report."""

    plan_path = stage_plan_path.resolve(strict=True)
    plan = _validate_stage_plan(plan_path)
    output_dir = Path(str(plan.get("output_dir") or "")).resolve(strict=True)
    if plan_path != output_dir / "preliminary_stage_plan.json":
        raise PreliminaryToolGateError("stage plan is not inside its bound output directory")
    validation_path = Path(str(plan.get("validation_file") or "")).resolve(strict=True)
    if _file_sha256(validation_path) != plan.get("validation_file_sha256"):
        raise PreliminaryToolGateError("validation bytes differ from the stage plan")
    adapter = (output_dir / "lora").resolve(strict=True)
    adapter_revision = checkpoint_tree_sha256(adapter)

    rows = _load_jsonl(validation_path)
    selected = select_gate_rows(rows)
    from psse_env.dagger.preliminary_e2b_eval import (
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
        MAX_INPUT_TOKENS,
        MAX_NEW_TOKENS as POLICY_MAX_NEW_TOKENS,
        _CanonicalE2BPolicy,
        _cached_bundle,
    )

    if MAX_INPUT_TOKENS != REQUIRED_CONTEXT_TOKENS:
        raise PreliminaryToolGateError("policy and generation-gate context limits differ")
    if POLICY_MAX_NEW_TOKENS != MAX_NEW_TOKENS:
        raise PreliminaryToolGateError("policy and generation-gate output limits differ")
    policy = _CanonicalE2BPolicy(_cached_bundle(str(adapter), adapter_revision))
    results: list[dict[str, Any]] = []
    for row in selected:
        state = _policy_state(row)
        generated_text = ""
        try:
            generated_text = policy.generate_text(state)
            result = evaluate_generation(
                row,
                generated_text,
                action_metrics=policy.last_action_metrics,
            )
        except Exception as exc:  # The report must retain every model/runtime failure.
            result = {
                "example_id": _example_id(row),
                "physical_root_fingerprint": _physical_root(row),
                "expected_action": _target_action(row),
                "generated_action": None,
                "bound_internal_action": None,
                "schema_valid": False,
                "state_bound": False,
                "target_tool_match": False,
                "exact_target_match": False,
                "hit_max_new_tokens": bool(
                    policy.last_action_metrics.get("hit_max_new_tokens") is True
                ),
                "action_metrics": policy.last_action_metrics,
                "generated_text_sha256": hashlib.sha256(
                    generated_text.encode("utf-8")
                ).hexdigest(),
                "generated_text_preview": generated_text[:240],
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)

    stage = str(plan.get("stage"))
    minimum_target_tool_rate = _minimum_target_tool_rate(stage)
    summary = summarize_results(
        results,
        minimum_target_tool_rate=minimum_target_tool_rate,
    )
    selected_ids = [_example_id(row) for row in selected]
    report = {
        "contract": GATE_CONTRACT,
        "artifact_type": GATE_ARTIFACT_TYPE,
        "release_eligible": False,
        "release_ineligibility_reasons": [
            "small-model preliminary debugging gate",
            "validation roots are not the frozen release evaluation suite",
        ],
        "stage": plan.get("stage"),
        "model": BASE_MODEL_ID,
        "model_revision": BASE_MODEL_REVISION,
        "adapter_path": str(adapter),
        "adapter_tree_sha256": adapter_revision,
        "stage_plan_sha256": _file_sha256(plan_path),
        "validation_file": str(validation_path),
        "validation_file_sha256": _file_sha256(validation_path),
        "selection": {
            "sample_count": len(selected),
            "selection_seed": SELECTION_SEED,
            "selected_example_ids": selected_ids,
            "selected_example_ids_sha256": _stable_sha256(selected_ids),
            "selected_row_multiset_sha256": _stable_sha256(
                sorted(_stable_sha256(row) for row in selected)
            ),
            "selected_physical_roots": sorted(
                {_physical_root(row) for row in selected}
            ),
        },
        "thresholds": gate_plan_contract(stage),
        "summary": summary,
        "results": results,
        "passed": summary["passed"],
    }
    report_path = output_dir / GATE_FILENAME
    _atomic_write_once(report_path, report)
    return report_path, report


def validate_gate_report(
    *,
    report_path: Path,
    stage_plan_path: Path,
    adapter_path: Path,
    validation_path: Path,
    require_passed: bool = True,
) -> dict[str, Any]:
    """Revalidate a published gate against current plan, adapter, and data."""

    if report_path.is_symlink() or not report_path.is_file():
        raise PreliminaryToolGateError("generation gate report is missing or a symlink")
    plan = _validate_stage_plan(stage_plan_path.resolve(strict=True))
    report = _load_json_object(report_path.resolve(strict=True), label="generation gate report")
    expected = {
        "contract": GATE_CONTRACT,
        "artifact_type": GATE_ARTIFACT_TYPE,
        "release_eligible": False,
        "stage": plan.get("stage"),
        "model": plan.get("model"),
        "model_revision": plan.get("model_revision"),
        "adapter_path": str(adapter_path.resolve(strict=True)),
        "adapter_tree_sha256": checkpoint_tree_sha256(adapter_path.resolve(strict=True)),
        "stage_plan_sha256": _file_sha256(stage_plan_path.resolve(strict=True)),
        "validation_file": str(validation_path.resolve(strict=True)),
        "validation_file_sha256": _file_sha256(validation_path.resolve(strict=True)),
        "thresholds": gate_plan_contract(str(plan.get("stage"))),
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise PreliminaryToolGateError(
                f"generation gate report {field} differs from current evidence"
            )
    results = report.get("results")
    if not isinstance(results, list):
        raise PreliminaryToolGateError("generation gate report results must be a list")
    recomputed = summarize_results(
        results,
        minimum_target_tool_rate=_minimum_target_tool_rate(str(plan.get("stage"))),
    )
    if report.get("summary") != recomputed or report.get("passed") != recomputed["passed"]:
        raise PreliminaryToolGateError("generation gate summary does not recompute")
    if require_passed and report.get("passed") is not True:
        raise PreliminaryToolGateError("generation gate did not pass")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-release preliminary E2B tool generation gate."
    )
    parser.add_argument("--stage-plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report_path, report = run_gate(args.stage_plan)
    except (OSError, PreliminaryToolGateError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "report": str(report_path),
                "passed": report["passed"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GATE_ARTIFACT_TYPE",
    "GATE_CONTRACT",
    "GATE_FILENAME",
    "PreliminaryToolGateError",
    "evaluate_generation",
    "gate_plan_contract",
    "main",
    "run_gate",
    "select_gate_rows",
    "summarize_results",
    "validate_gate_report",
]
