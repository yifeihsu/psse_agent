"""Compare multiple Gemma-4 BC0 adapters on one frozen closed-loop suite.

The comparison is intentionally evaluation-only.  Each adapter receives its
own resumable report, and a report is reused only when its adapter bytes,
suite bytes, roots, and evaluation configuration still match.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psse_env.research_models import GEMMA4_12B

from .gates import GateError
from .provenance import stable_json_sha256
from .research_bc0_eval import (
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    REQUIRED_MAX_INPUT_TOKENS,
    RESEARCH_BC0_EVAL_CONTRACT,
    STANDARD_SUITE_NAME,
    _configure_input_ceiling,
    adapter_content_fingerprint,
    evaluate_research_suite,
    summarize_closed_loop_outcomes,
)


CHECKPOINT_REPORT_CONTRACT = "research_gemma4_12b_bc0_checkpoint_eval_v1"
COMPARISON_REPORT_CONTRACT = "research_gemma4_12b_bc0_checkpoint_comparison_v1"
COMPARISON_REPORT_NAME = "research_bc0_checkpoint_comparison.json"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class AdapterSpec:
    label: str
    path: Path


def parse_adapter_spec(value: str) -> AdapterSpec:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("adapter must use LABEL=ADAPTER_PATH")
    if _LABEL.fullmatch(label) is None:
        raise argparse.ArgumentTypeError(
            "adapter label must be 1-64 letters, numbers, dots, dashes, or underscores"
        )
    return AdapterSpec(label=label, path=Path(raw_path))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _scenario_root(row: Mapping[str, Any]) -> str:
    grouping = row.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    return str(grouping.get("physical_root_fingerprint") or "").strip()


def load_frozen_suite(
    path: str | Path,
    *,
    suite_name: str = STANDARD_SUITE_NAME,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"frozen suite JSON is invalid: {exc}") from exc

    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, Mapping):
        raw_rows = payload.get(suite_name)
    else:
        raw_rows = None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise GateError(f"frozen suite needs a non-empty {suite_name!r} list")
    if not all(isinstance(row, Mapping) for row in raw_rows):
        raise GateError("every frozen suite row must be a JSON object")

    scenarios = [copy.deepcopy(dict(row)) for row in raw_rows]
    roots = [_scenario_root(row) for row in scenarios]
    if any(not root for root in roots):
        raise GateError("every frozen suite row needs a physical root")
    duplicates = sorted({root for root in roots if roots.count(root) > 1})
    if duplicates:
        raise GateError(f"frozen suite physical roots are not unique: {duplicates}")
    sorted_roots = sorted(roots)
    identity = {
        "path": str(resolved),
        "suite_name": suite_name,
        "episodes": len(scenarios),
        "content_sha256": stable_json_sha256(scenarios),
        "root_set_sha256": stable_json_sha256(sorted_roots),
        "physical_roots": sorted_roots,
    }
    return scenarios, identity


def _resolve_adapters(specs: Sequence[AdapterSpec]) -> list[dict[str, Any]]:
    if not specs:
        raise GateError("at least one adapter is required")
    labels: set[str] = set()
    fingerprints: dict[str, str] = {}
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        if _LABEL.fullmatch(spec.label) is None:
            raise GateError(f"invalid adapter label: {spec.label!r}")
        folded = spec.label.casefold()
        if folded in labels:
            raise GateError(f"adapter labels must be unique: {spec.label!r}")
        labels.add(folded)
        path = spec.path.expanduser().resolve(strict=True)
        identity = adapter_content_fingerprint(path)
        fingerprint = str(identity["content_sha256"])
        previous = fingerprints.get(fingerprint)
        if previous is not None:
            raise GateError(
                "adapter fingerprints must be unique: "
                f"{previous!r} and {spec.label!r} have {fingerprint}"
            )
        fingerprints[fingerprint] = spec.label
        resolved.append(
            {
                "label": spec.label,
                "path": path,
                "identity": identity,
            }
        )
    return resolved


def _report_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"research_bc0_checkpoint_{label}.json"


def _read_report(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _matching_completed_report(
    payload: Mapping[str, Any] | None,
    *,
    label: str,
    fingerprint: str,
    suite: Mapping[str, Any],
    seed: int,
    max_steps: int,
) -> bool:
    if payload is None:
        return False
    adapter = payload.get("adapter")
    adapter = adapter if isinstance(adapter, Mapping) else {}
    recorded_suite = payload.get("suite")
    recorded_suite = recorded_suite if isinstance(recorded_suite, Mapping) else {}
    configuration = payload.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    behavior = payload.get("policy_behavior")
    behavior = behavior if isinstance(behavior, Mapping) else {}
    evaluation = payload.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    suite_metrics = evaluation.get("suite_metrics")
    suite_metrics = suite_metrics if isinstance(suite_metrics, Mapping) else {}
    overall = suite_metrics.get("overall")
    overall = overall if isinstance(overall, Mapping) else {}
    expected_episodes = int(suite["episodes"])
    required_behavior = {
        "episodes",
        "schema_valid_action_rate",
        "observable_expert_tool_agreement_rate",
        "observable_expert_exact_action_agreement_rate",
    }
    required_outcomes = {
        "episodes",
        "audited_completion_episodes",
        "audited_completion_rate",
        "audited_post_correction_handoff_episodes",
        "audited_post_correction_handoff_rate",
        "final_physical_success_episodes",
        "final_physical_success_rate",
        "invalid_action_count",
        "episodes_with_invalid_actions",
        "false_commit_count",
        "false_commit_episodes",
        "false_commit_rate",
        "loop_episodes",
        "loop_rate",
    }
    return bool(
        payload.get("contract") == CHECKPOINT_REPORT_CONTRACT
        and payload.get("research_eval_contract") == RESEARCH_BC0_EVAL_CONTRACT
        and payload.get("evaluation_completed") is True
        and adapter.get("label") == label
        and adapter.get("content_sha256") == fingerprint
        and recorded_suite.get("content_sha256") == suite.get("content_sha256")
        and recorded_suite.get("root_set_sha256") == suite.get("root_set_sha256")
        and recorded_suite.get("physical_roots") == suite.get("physical_roots")
        and configuration.get("seed") == seed
        and configuration.get("max_steps") == max_steps
        and required_behavior.issubset(behavior)
        and required_outcomes.issubset(overall)
        and behavior.get("episodes") == expected_episodes
        and overall.get("episodes") == expected_episodes
    )


def _count(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _rate(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _outcome_mapping(
    outcomes: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = outcomes.get(key)
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _comparison_entry(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    reused_report: bool,
) -> dict[str, Any]:
    evaluation = report.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    suite_metrics = evaluation.get("suite_metrics")
    suite_metrics = suite_metrics if isinstance(suite_metrics, Mapping) else {}
    overall = suite_metrics.get("overall")
    overall = overall if isinstance(overall, Mapping) else {}
    behavior = report.get("policy_behavior")
    behavior = behavior if isinstance(behavior, Mapping) else {}
    adapter = report.get("adapter")
    adapter = adapter if isinstance(adapter, Mapping) else {}
    outcomes = summarize_closed_loop_outcomes(evaluation)
    return {
        "label": adapter.get("label"),
        "adapter_content_sha256": adapter.get("content_sha256"),
        "report_path": str(report_path),
        "reused_report": reused_report,
        "episodes": _count(overall, "episodes"),
        "primary_success_metric": outcomes.get("primary_success_metric"),
        "success_contract": outcomes.get("success_contract"),
        "contract_execution": outcomes.get("contract_execution"),
        "historical_backfill_lower_bound": outcomes.get(
            "historical_backfill_lower_bound"
        ),
        "truth_audited_task_success": _outcome_mapping(
            outcomes, "truth_audited_task_success"
        ),
        "truth_audited_fault_recovery_success": _outcome_mapping(
            outcomes, "truth_audited_fault_recovery_success"
        ),
        "accepted_true_fault_target_coverage": _outcome_mapping(
            outcomes, "accepted_true_fault_target_coverage"
        ),
        "truth_audited_fault_target_correction": _outcome_mapping(
            outcomes, "truth_audited_fault_target_correction"
        ),
        "safe_completion": _outcome_mapping(outcomes, "safe_completion"),
        "audited_completion": {
            "episodes": _count(overall, "audited_completion_episodes"),
            "rate": _rate(overall, "audited_completion_rate"),
        },
        "audited_post_correction_handoff": {
            "episodes": _count(overall, "audited_post_correction_handoff_episodes"),
            "rate": _rate(overall, "audited_post_correction_handoff_rate"),
        },
        "strict_resolved_physical_success": {
            "episodes": _count(overall, "final_physical_success_episodes"),
            "rate": _rate(overall, "final_physical_success_rate"),
        },
        "schema_valid_action_rate": _rate(behavior, "schema_valid_action_rate"),
        "observable_expert_tool_agreement_rate": _rate(
            behavior, "observable_expert_tool_agreement_rate"
        ),
        "observable_expert_exact_action_agreement_rate": _rate(
            behavior, "observable_expert_exact_action_agreement_rate"
        ),
        "invalid_actions": {
            "count": _count(overall, "invalid_action_count"),
            "episodes": _count(overall, "episodes_with_invalid_actions"),
        },
        "false_commits": {
            "count": _count(overall, "false_commit_count"),
            "episodes": _count(overall, "false_commit_episodes"),
            "rate": _rate(overall, "false_commit_rate"),
        },
        "loops": {
            "episodes": _count(overall, "loop_episodes"),
            "rate": _rate(overall, "loop_rate"),
        },
    }


def _descending(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return -float(value)
    return 1.0


def _ranking_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    task_success = entry.get("truth_audited_task_success")
    task_success = task_success if isinstance(task_success, Mapping) else {}
    fault_recovery = entry.get("truth_audited_fault_recovery_success")
    fault_recovery = fault_recovery if isinstance(fault_recovery, Mapping) else {}
    target_correction = entry.get("truth_audited_fault_target_correction")
    target_correction = (
        target_correction if isinstance(target_correction, Mapping) else {}
    )
    safe_completion = entry.get("safe_completion")
    safe_completion = safe_completion if isinstance(safe_completion, Mapping) else {}
    audited = entry["audited_completion"]
    handoff = entry["audited_post_correction_handoff"]
    strict = entry["strict_resolved_physical_success"]
    invalid = entry["invalid_actions"]
    false_commits = entry["false_commits"]
    loops = entry["loops"]
    return (
        _descending(task_success.get("rate")),
        _descending(fault_recovery.get("rate")),
        _descending(target_correction.get("rate")),
        _descending(safe_completion.get("rate")),
        _descending(audited["rate"]),
        _descending(handoff["rate"]),
        _descending(strict["rate"]),
        _descending(entry["schema_valid_action_rate"]),
        _descending(entry["observable_expert_exact_action_agreement_rate"]),
        _descending(entry["observable_expert_tool_agreement_rate"]),
        int(invalid["count"]),
        int(false_commits["count"]),
        int(loops["episodes"]),
        str(entry["label"]).casefold(),
    )


def _progress(record: Mapping[str, Any]) -> None:
    if record.get("event") == "episode_complete":
        print(json.dumps(dict(record), sort_keys=True), file=sys.stderr, flush=True)


def _release_accelerator_memory() -> None:
    """Release one adapter/model before loading the next checkpoint."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def compare_checkpoints(
    *,
    suite_json: str | Path,
    adapters: Sequence[AdapterSpec],
    output_dir: str | Path,
    suite_name: str = STANDARD_SUITE_NAME,
    seed: int = DEFAULT_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy_loader: Callable[..., Any] | None = None,
    evaluator: Callable[..., Any] | None = None,
    environment_factory: Callable[..., Any] | None = None,
    case_loader: Callable[[Any], Any] | None = None,
    expert_factory: Callable[[], Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = _progress,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    _configure_input_ceiling()
    scenarios, suite = load_frozen_suite(suite_json, suite_name=suite_name)
    resolved_adapters = _resolve_adapters(adapters)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    evaluated_labels: list[str] = []
    reused_labels: list[str] = []
    for item in resolved_adapters:
        label = str(item["label"])
        adapter_path = item["path"]
        adapter_identity = item["identity"]
        report_path = _report_path(output, label)
        existing = _read_report(report_path)
        reusable = _matching_completed_report(
            existing,
            label=label,
            fingerprint=str(adapter_identity["content_sha256"]),
            suite=suite,
            seed=seed,
            max_steps=max_steps,
        )
        if reusable:
            report = dict(existing)
            reused_labels.append(label)
        else:
            evaluation, behavior = evaluate_research_suite(
                scenarios,
                adapter_path=adapter_path,
                seed=seed,
                max_steps=max_steps,
                policy_loader=policy_loader,
                evaluator=evaluator,
                environment_factory=environment_factory,
                case_loader=case_loader,
                expert_factory=expert_factory,
                progress_callback=progress_callback,
            )
            report = {
                "contract": CHECKPOINT_REPORT_CONTRACT,
                "research_eval_contract": RESEARCH_BC0_EVAL_CONTRACT,
                "evaluation_completed": True,
                "adapter": {
                    "label": label,
                    "path": str(adapter_path),
                    **adapter_identity,
                },
                "model": {
                    "model_id": GEMMA4_12B.model_id,
                    "revision": GEMMA4_12B.revision,
                    "architecture": GEMMA4_12B.architecture,
                    "prompt_profile": GEMMA4_12B.prompt_profile,
                },
                "max_input_tokens": REQUIRED_MAX_INPUT_TOKENS,
                "configuration": {"seed": seed, "max_steps": max_steps},
                "suite": copy.deepcopy(suite),
                "policy_behavior": behavior,
                "closed_loop_outcomes": summarize_closed_loop_outcomes(evaluation),
                "evaluation": evaluation,
            }
            _atomic_json(report_path, report)
            evaluated_labels.append(label)
        entries.append(
            _comparison_entry(
                report,
                report_path=report_path,
                reused_report=reusable,
            )
        )
        _release_accelerator_memory()

    ranked = sorted(entries, key=_ranking_key)
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
    aggregate = {
        "contract": COMPARISON_REPORT_CONTRACT,
        "research_eval_contract": RESEARCH_BC0_EVAL_CONTRACT,
        "comparison_completed": True,
        "suite": suite,
        "configuration": {"seed": seed, "max_steps": max_steps},
        "adapter_count": len(ranked),
        "evaluation_order": [str(item["label"]) for item in resolved_adapters],
        "evaluated_labels": evaluated_labels,
        "reused_labels": reused_labels,
        "ranking_criteria": [
            "truth_audited_task_success_rate_desc",
            "truth_audited_fault_recovery_success_rate_desc",
            "truth_audited_fault_target_correction_rate_desc",
            "safe_completion_rate_desc",
            "audited_completion_rate_desc",
            "audited_post_correction_handoff_rate_desc",
            "strict_resolved_physical_success_rate_desc",
            "schema_valid_action_rate_desc",
            "observable_expert_exact_action_agreement_rate_desc",
            "observable_expert_tool_agreement_rate_desc",
            "invalid_action_count_asc",
            "false_commit_count_asc",
            "loop_episode_count_asc",
            "label_asc",
        ],
        "ranking": ranked,
    }
    _atomic_json(output / COMPARISON_REPORT_NAME, aggregate)
    return aggregate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite-json", required=True, type=Path)
    result.add_argument("--suite-name", default=STANDARD_SUITE_NAME)
    result.add_argument(
        "--adapter",
        dest="adapters",
        action="append",
        required=True,
        type=parse_adapter_spec,
        metavar="LABEL=ADAPTER_PATH",
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return result


def run(
    args: argparse.Namespace,
    **kwargs: Any,
) -> dict[str, Any]:
    return compare_checkpoints(
        suite_json=args.suite_json,
        suite_name=args.suite_name,
        adapters=args.adapters,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        comparison = run(parser().parse_args(argv))
    except Exception as exc:
        print(
            json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr
        )
        return 2
    print(
        json.dumps(
            {
                "comparison_completed": comparison["comparison_completed"],
                "evaluated_labels": comparison["evaluated_labels"],
                "reused_labels": comparison["reused_labels"],
                "ranking": [
                    {"rank": row["rank"], "label": row["label"]}
                    for row in comparison["ranking"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AdapterSpec",
    "CHECKPOINT_REPORT_CONTRACT",
    "COMPARISON_REPORT_CONTRACT",
    "COMPARISON_REPORT_NAME",
    "compare_checkpoints",
    "load_frozen_suite",
    "main",
    "parse_adapter_spec",
    "run",
]
