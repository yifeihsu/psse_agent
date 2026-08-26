"""Lightweight research-only Gemma-4-12B BC0 closed-loop baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.actions import INVALID_ACTION, action_signature, safe_normalize_action
from psse_env.dagger.release_audit import (
    TRUTH_AUDITED_TASK_SUCCESS_CONTRACT,
    validate_post_correction_handoff_assessment,
)
from psse_env.research_models import GEMMA4_12B

from .gates import GateError, load_jsonl
from .provenance import file_sha256, stable_json_sha256


RESEARCH_BC0_EVAL_CONTRACT = "research_gemma4_12b_bc0_baseline_v2"
REQUIRED_MAX_INPUT_TOKENS = 32768
DEFAULT_SEED = 20260720
DEFAULT_MAX_STEPS = 24
DEFAULT_CANDIDATE_MULTIPLIER = 3
DEFAULT_D1_PLAN = {
    "measurement+parameter": 6,
    "multi_measurement": 6,
    "parameter": 3,
}
DEFAULT_D0_SUITE = (
    Path(__file__).resolve().parents[1] / "dagger" / "suites" / "bc0_eval_suite_v1.json"
)
STANDARD_SUITE_NAME = "standard_success"
EXPECTED_D0_STANDARD_ROOTS = 21
SCHEMA_VALID_READINESS_RATE = 0.90


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _callable_identity(value: Callable[..., Any]) -> str:
    module = str(getattr(value, "__module__", "") or type(value).__module__)
    qualname = str(getattr(value, "__qualname__", "") or type(value).__qualname__)
    return f"{module}:{qualname}"


def _training_root(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    grouping = row.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    return str(
        row.get("physical_root_fingerprint")
        or metadata.get("physical_root_fingerprint")
        or grouping.get("physical_root_fingerprint")
        or ""
    ).strip()


def _scenario_root(row: Mapping[str, Any]) -> str:
    grouping = row.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    return str(grouping.get("physical_root_fingerprint") or "").strip()


def _scenario_family(row: Mapping[str, Any]) -> str:
    grouping = row.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    return str(grouping.get("scenario_family") or "").strip()


def load_d0_training_roots(path: str | Path) -> tuple[set[str], int]:
    rows = load_jsonl(Path(path).expanduser().resolve(strict=True))
    if not rows:
        raise GateError("canonical D0 training view is empty")
    roots = {_training_root(row) for row in rows}
    if "" in roots:
        raise GateError("every canonical D0 training row needs a physical root")
    return roots, len(rows)


def load_frozen_standard_suite() -> list[dict[str, Any]]:
    payload = json.loads(DEFAULT_D0_SUITE.read_text(encoding="utf-8"))
    rows = payload.get(STANDARD_SUITE_NAME) if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise GateError("packaged D0 suite has no valid standard_success list")
    scenarios = [copy.deepcopy(dict(row)) for row in rows]
    roots = [_scenario_root(row) for row in scenarios]
    if (
        len(roots) != EXPECTED_D0_STANDARD_ROOTS
        or any(not root for root in roots)
        or len(roots) != len(set(roots))
    ):
        raise GateError("packaged D0 standard_success must contain 21 unique roots")
    return scenarios


def adapter_content_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a cheap identity for reporting and D0-to-D1 continuity."""

    adapter = Path(path).expanduser().resolve(strict=True)
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise GateError(
            "adapter needs adapter_config.json and adapter_model.safetensors"
        )
    files = {candidate.name: file_sha256(candidate) for candidate in (config, weights)}
    return {"content_sha256": stable_json_sha256(files), "files": files}


def _configure_input_ceiling() -> None:
    configured = os.environ.get("RESEARCH_MAX_INPUT_TOKENS")
    if configured is not None and configured.strip() != str(REQUIRED_MAX_INPUT_TOKENS):
        raise GateError("RESEARCH_MAX_INPUT_TOKENS must be 32768 for this baseline")
    os.environ["RESEARCH_MAX_INPUT_TOKENS"] = str(REQUIRED_MAX_INPUT_TOKENS)
    loaded = sys.modules.get("psse_env.dagger.preliminary_e2b_eval")
    if (
        loaded is not None
        and getattr(loaded, "MAX_INPUT_TOKENS", None) != REQUIRED_MAX_INPUT_TOKENS
    ):
        raise GateError("start a fresh process so the 32768-token ceiling takes effect")


def build_d1_development_suite(
    *,
    d0_training_roots: set[str],
    frozen_standard_roots: set[str],
    seed: int = DEFAULT_SEED,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    generator_factory: Callable[..., Any] | None = None,
    partitioner: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate the exact deterministic, root-isolated 6/6/3 D1 suite."""

    if not d0_training_roots or not frozen_standard_roots:
        raise GateError("D1 generation needs both D0 training and holdout roots")
    if isinstance(candidate_multiplier, bool) or candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    if generator_factory is None:
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator

        generator_factory = Round0ScenarioGenerator
    if partitioner is None:
        from psse_env.dagger.suite_builder import partition_release_scenario_v1

        partitioner = partition_release_scenario_v1

    requested = {
        family: count * candidate_multiplier
        for family, count in sorted(DEFAULT_D1_PLAN.items())
    }
    generator = generator_factory(
        seed=seed,
        source_partition="train",
        parameter_ranking_dominance_threshold=1.0,
    )
    generated = list(generator.build(requested))
    buckets: dict[str, list[dict[str, Any]]] = {
        family: [] for family in DEFAULT_D1_PLAN
    }
    seen: set[str] = set()
    excluded: Counter[str] = Counter()
    for raw in generated:
        envelope = copy.deepcopy(dict(partitioner(raw, split="development")))
        root = _scenario_root(envelope)
        family = _scenario_family(envelope)
        if family not in buckets or not root:
            excluded["invalid_or_unrequested"] += 1
        elif root in d0_training_roots:
            excluded["d0_training_overlap"] += 1
        elif root in frozen_standard_roots:
            excluded["d0_standard_overlap"] += 1
        elif root in seen:
            excluded["duplicate_root"] += 1
        else:
            seen.add(root)
            buckets[family].append(envelope)

    selected: list[dict[str, Any]] = []
    eligible: dict[str, int] = {}
    for family, required in sorted(DEFAULT_D1_PLAN.items()):
        bucket = sorted(
            buckets[family],
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}:{family}:{_scenario_root(row)}".encode()
                ).hexdigest(),
                _scenario_root(row),
            ),
        )
        eligible[family] = len(bucket)
        if len(bucket) < required:
            raise GateError(
                f"D1 {family} has {len(bucket)}/{required} fresh roots; "
                "increase --candidate-multiplier"
            )
        selected.extend(bucket[:required])
    selected.sort(key=lambda row: (_scenario_family(row), _scenario_root(row)))

    roots = {_scenario_root(row) for row in selected}
    plan = Counter(_scenario_family(row) for row in selected)
    if len(roots) != 15 or dict(plan) != DEFAULT_D1_PLAN:
        raise GateError("D1 selection did not produce the exact 15-root 6/6/3 mix")
    if roots & (d0_training_roots | frozen_standard_roots):
        raise GateError("D1 selection overlaps a protected D0 root")
    from psse_env.dagger.evaluator import validate_release_scenario_suites

    validate_release_scenario_suites({STANDARD_SUITE_NAME: selected})
    return selected, {
        "plan": dict(sorted(plan.items())),
        "physical_roots": sorted(roots),
        "eligible_candidates": eligible,
        "excluded_candidates": dict(sorted(excluded.items())),
        "root_isolation": {
            "d0_training_overlap": [],
            "d0_standard_overlap": [],
        },
    }


class _InstrumentedPolicy:
    def __init__(self, policy: Any, records: list[dict[str, Any]]) -> None:
        self.policy = policy
        self.records = records

    @property
    def last_action_metrics(self) -> Any:
        return getattr(self.policy, "last_action_metrics", {})

    def act(self, observation: Mapping[str, Any]) -> Any:
        try:
            return self.policy.act(copy.deepcopy(dict(observation)))
        finally:
            metrics = getattr(self.policy, "last_action_metrics", {})
            metrics = metrics() if callable(metrics) else metrics
            items = metrics.items() if isinstance(metrics, Mapping) else []
            self.records.append(
                {
                    str(key): value
                    for key, value in items
                    if value is None or isinstance(value, (str, bool, int, float))
                }
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)


def summarize_policy_behavior(
    evaluation: Mapping[str, Any],
    generation_records: Sequence[Mapping[str, Any]],
    *,
    expert_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    suite_metrics = evaluation.get("suite_metrics")
    suite_metrics = suite_metrics if isinstance(suite_metrics, Mapping) else {}
    episodes = suite_metrics.get("episodes")
    episodes = episodes if isinstance(episodes, list) else []
    if expert_factory is None:
        from psse_env.oracle.expert_policy import ExpertPolicyOracle

        expert_factory = ExpertPolicyOracle
    expert = expert_factory()
    steps = valid = comparable = tool_matches = exact_matches = expert_errors = 0
    for episode in episodes:
        trace = episode.get("trace", []) if isinstance(episode, Mapping) else []
        for row in trace if isinstance(trace, list) else []:
            if not isinstance(row, Mapping) or row.get("intervention") is not False:
                continue
            observation = row.get("policy_observation")
            if not isinstance(observation, Mapping):
                continue
            action = safe_normalize_action(row.get("action"))
            steps += 1
            valid += action.get("tool") != INVALID_ACTION
            try:
                history = observation.get("history_window")
                history = list(history) if isinstance(history, list) else []
                proposals = expert.next_actions(
                    copy.deepcopy(dict(observation)), history
                )
                if not proposals:
                    continue
                expected = safe_normalize_action(proposals[0])
                comparable += 1
                tool_matches += action.get("tool") == expected.get("tool")
                exact_matches += action_signature(action) == action_signature(expected)
            except Exception:
                expert_errors += 1

    def rate(value: int, total: int) -> float | None:
        return value / total if total else None

    truncated = [
        int(row.get("truncated_input_tokens") or 0) for row in generation_records
    ]
    original_prompts = [
        int(row["original_prompt_tokens"])
        for row in generation_records
        if isinstance(row.get("original_prompt_tokens"), int)
    ]
    return {
        "episodes": len(episodes),
        "evaluator_error_episodes": sum(
            isinstance(row, Mapping) and row.get("evaluator_error") is not None
            for row in episodes
        ),
        "policy_steps": steps,
        "schema_valid_actions": valid,
        "schema_valid_action_rate": rate(valid, steps),
        "observable_expert_comparable_steps": comparable,
        "observable_expert_comparison_errors": expert_errors,
        "observable_expert_tool_agreement_rate": rate(tool_matches, comparable),
        "observable_expert_exact_action_agreement_rate": rate(
            exact_matches, comparable
        ),
        "maximum_original_prompt_tokens": max(original_prompts, default=None),
        "input_truncated_steps": sum(value > 0 for value in truncated),
        "input_truncated_tokens": sum(max(0, value) for value in truncated),
    }


def evaluate_research_suite(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    adapter_path: Path,
    seed: int,
    max_steps: int,
    policy_loader: Callable[..., Any] | None = None,
    evaluator: Callable[..., Any] | None = None,
    environment_factory: Callable[..., Any] | None = None,
    case_loader: Callable[[Any], Any] | None = None,
    expert_factory: Callable[[], Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy_loader is None:
        from psse_env.dagger.research_policy_factory import (
            research_gemma_policy_factory,
        )

        policy_loader = research_gemma_policy_factory
    if evaluator is None:
        from psse_env.dagger.evaluator import evaluate_rollout_suites

        evaluator = evaluate_rollout_suites
    if environment_factory is None:
        from psse_env.dagger.release_factories import production_environment_factory

        environment_factory = production_environment_factory
    if case_loader is None:
        from psse_env.dagger.release_factories import deterministic_case_loader

        case_loader = deterministic_case_loader

    records: list[dict[str, Any]] = []
    policy = policy_loader(
        adapter_path,
        base_model=GEMMA4_12B.model_id,
        base_revision=GEMMA4_12B.revision,
        architecture=GEMMA4_12B.architecture,
        prompt_profile=GEMMA4_12B.prompt_profile,
        load_in_4bit=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    instrumented = _InstrumentedPolicy(policy, records)
    result = evaluator(
        {STANDARD_SUITE_NAME: [copy.deepcopy(dict(row)) for row in scenarios]},
        env_factory=environment_factory,
        policy_factory=lambda **_kwargs: instrumented,
        max_steps=max_steps,
        seed=seed,
        required_suites=[STANDARD_SUITE_NAME],
        minimum_suites=1,
        minimum_episodes_per_suite=len(scenarios),
        minimum_roots_per_suite=len(scenarios),
        case_loader=case_loader,
        require_release_environment=False,
        require_policy_identity=False,
        progress_callback=progress_callback,
    ).as_dict()
    return result, summarize_policy_behavior(
        result, records, expert_factory=expert_factory
    )


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_95(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    observed = successes / total
    scale = 1.0 + z * z / total
    center = (observed + z * z / (2.0 * total)) / scale
    half_width = (
        z
        * math.sqrt(observed * (1.0 - observed) / total + z * z / (4.0 * total * total))
        / scale
    )
    return {"low": max(0.0, center - half_width), "high": min(1.0, center + half_width)}


def _positive_cardinality(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _legacy_truth_audited_task_success(episode: Mapping[str, Any]) -> bool:
    """Backfill the new primary metric from persisted completion evidence."""

    if episode.get("final_physical_success") is True:
        return True
    audit = episode.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    strict = audit.get("strict_release_audit")
    strict = strict if isinstance(strict, Mapping) else {}
    if (
        episode.get("terminal_outcome") == "resolved"
        and strict.get("quarantined") is False
        and strict.get("problems") == []
    ):
        return True
    assessment = audit.get("post_correction_handoff_assessment")
    valid, _ = validate_post_correction_handoff_assessment(
        assessment,
        str(episode.get("scenario_id") or ""),
        str(episode.get("physical_root") or ""),
        str(episode.get("family") or ""),
    )
    return valid


def _episode_task_success(episode: Mapping[str, Any]) -> tuple[bool, bool]:
    success = episode.get("truth_audited_task_success")
    evidence_known = episode.get("truth_audited_task_success_evidence_known")
    if isinstance(success, bool) and isinstance(evidence_known, bool):
        return success, evidence_known
    backfilled = _legacy_truth_audited_task_success(episode)
    return backfilled, backfilled


def _episode_faulted(episode: Mapping[str, Any]) -> bool:
    audit = episode.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    assessment = audit.get("truth_audited_task_assessment")
    if isinstance(assessment, Mapping) and isinstance(assessment.get("faulted"), bool):
        return bool(assessment["faulted"])
    return _positive_cardinality(episode.get("cardinality"))


def _episode_fault_presence_known(episode: Mapping[str, Any]) -> bool:
    audit = episode.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    assessment = audit.get("truth_audited_task_assessment")
    if isinstance(assessment, Mapping) and isinstance(
        assessment.get("fault_presence_known"), bool
    ):
        return bool(assessment["fault_presence_known"])
    cardinality = episode.get("cardinality")
    return bool(
        isinstance(cardinality, int)
        and not isinstance(cardinality, bool)
        and cardinality >= 0
    )


def _target_identities(value: Any) -> set[tuple[str, int]] | None:
    if not isinstance(value, Mapping):
        return None
    targets: set[tuple[str, int]] = set()
    for family in ("measurement", "parameter", "topology"):
        rows = value.get(family)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return None
        for raw_target in rows:
            if isinstance(raw_target, bool) or not isinstance(raw_target, int):
                return None
            targets.add((family, raw_target))
    return targets


def _target_evidence_identity(raw_row: Mapping[str, Any]) -> tuple[str, int] | None:
    family = raw_row.get("family")
    if family == "measurement":
        raw_target = raw_row.get("index0")
        if isinstance(raw_target, int) and not isinstance(raw_target, bool):
            return family, raw_target
        return None
    if family not in {"parameter", "topology"}:
        return None
    raw_target = raw_row.get("target")
    if (
        isinstance(raw_target, Sequence)
        and not isinstance(raw_target, (str, bytes))
        and len(raw_target) == 2
        and raw_target[0] == "branch_row0"
        and isinstance(raw_target[1], int)
        and not isinstance(raw_target[1], bool)
    ):
        return str(family), raw_target[1]
    return None


def _target_distance_evidence(
    audit: Mapping[str, Any],
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    task = audit.get("truth_audited_task_assessment")
    task = task if isinstance(task, Mapping) else {}
    completion = task.get("counterfactual_completion_audit")
    if not isinstance(completion, Mapping):
        handoff = audit.get("post_correction_handoff_assessment")
        handoff = handoff if isinstance(handoff, Mapping) else {}
        completion = handoff.get("counterfactual_completion_audit")
    if not isinstance(completion, Mapping):
        completion = audit.get("strict_release_audit")
    completion = completion if isinstance(completion, Mapping) else {}
    checks = completion.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    nonregression = checks.get("accepted_target_nonregression")
    nonregression = nonregression if isinstance(nonregression, Mapping) else {}
    evidence = nonregression.get("target_evidence")
    evidence = (
        evidence
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes))
        else []
    )
    observed: set[tuple[str, int]] = set()
    corrected: set[tuple[str, int]] = set()
    for raw_row in evidence:
        if not isinstance(raw_row, Mapping):
            continue
        identity = _target_evidence_identity(raw_row)
        final_distance = raw_row.get("final_distance")
        tolerance = raw_row.get("tolerance")
        if (
            identity is None
            or isinstance(final_distance, bool)
            or not isinstance(final_distance, (int, float))
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(final_distance))
            or not math.isfinite(float(tolerance))
            or float(final_distance) < 0.0
            or float(tolerance) < 0.0
            or raw_row.get("status") not in {"passed", "failed"}
        ):
            continue
        observed.add(identity)
        if raw_row.get("status") == "passed" and float(final_distance) <= float(
            tolerance
        ):
            corrected.add(identity)
    return observed, corrected


def _target_progress(episodes: Sequence[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    true_count = accepted_count = corrected_count = 0
    accepted_evidence_episodes = correction_evidence_episodes = 0
    correction_evidence_targets = eligible_episodes = 0
    for raw_episode in episodes:
        if not isinstance(raw_episode, Mapping) or not _positive_cardinality(
            raw_episode.get("cardinality")
        ):
            continue
        eligible_episodes += 1
        audit = raw_episode.get("audit")
        audit = audit if isinstance(audit, Mapping) else {}
        target_audit = audit.get("accepted_target_audit")
        target_audit = target_audit if isinstance(target_audit, Mapping) else {}
        true_targets = _target_identities(target_audit.get("true_targets"))
        accepted_targets = _target_identities(target_audit.get("accepted_targets"))
        if not true_targets or accepted_targets is None:
            continue
        accepted_evidence_episodes += 1
        true_count += len(true_targets)
        accepted_true_targets = true_targets & accepted_targets
        accepted_count += len(accepted_true_targets)
        observed_targets, clean_targets = _target_distance_evidence(audit)
        corrected_count += len(true_targets & accepted_targets & clean_targets)
        known_correction_targets = (true_targets - accepted_true_targets) | (
            accepted_true_targets & observed_targets
        )
        correction_evidence_targets += len(known_correction_targets)
        correction_evidence_episodes += int(accepted_true_targets <= observed_targets)
    accepted_common = {
        "true_targets": true_count,
        "audited_faulted_episodes": accepted_evidence_episodes,
        "eligible_faulted_episodes": eligible_episodes,
        "evidence_coverage_rate": _rate(accepted_evidence_episodes, eligible_episodes),
        "supported_target_families": ["measurement", "parameter", "topology"],
    }
    accepted = {
        **accepted_common,
        "accepted_true_targets": accepted_count,
        "rate": _rate(accepted_count, true_count),
        "interpretation": "target identification plus accepted correction action",
    }
    corrected = {
        "true_targets": true_count,
        "audited_faulted_episodes": correction_evidence_episodes,
        "eligible_faulted_episodes": eligible_episodes,
        "evidence_coverage_rate": _rate(
            correction_evidence_episodes, eligible_episodes
        ),
        "evidence_covered_true_targets": correction_evidence_targets,
        "target_evidence_coverage_rate": _rate(correction_evidence_targets, true_count),
        "evidence_interpretation": (
            "accepted true targets require valid final-distance evidence; "
            "unaccepted true targets are known correction failures"
        ),
        "supported_target_families": ["measurement", "parameter", "topology"],
        "clean_tolerance_corrected_targets": corrected_count,
        "rate": _rate(corrected_count, true_count),
        "interpretation": "accepted true targets with final distance within tolerance",
    }
    return accepted, corrected


def summarize_closed_loop_outcomes(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose terminal-independent task success and stricter safety outcomes."""

    suite_metrics = evaluation.get("suite_metrics")
    suite_metrics = suite_metrics if isinstance(suite_metrics, Mapping) else {}
    overall = suite_metrics.get("overall")
    overall = overall if isinstance(overall, Mapping) else {}
    raw_episodes = suite_metrics.get("episodes")
    episodes = (
        list(raw_episodes)
        if isinstance(raw_episodes, Sequence)
        and not isinstance(raw_episodes, (str, bytes))
        else []
    )
    derived = [
        (row, *_episode_task_success(row))
        for row in episodes
        if isinstance(row, Mapping)
    ]
    reported_episode_count = _nonnegative_int(overall.get("episodes"))
    episode_count = max(reported_episode_count or 0, len(derived))
    native_task_successes = _nonnegative_int(
        overall.get("truth_audited_task_success_episodes")
    )
    native_task_known = _nonnegative_int(
        overall.get("truth_audited_task_success_evidence_known_episodes")
    )
    native_faulted_episodes = _nonnegative_int(overall.get("faulted_episodes"))
    native_fault_recovery_successes = _nonnegative_int(
        overall.get("truth_audited_fault_recovery_success_episodes")
    )
    native_aggregates_present = all(
        value is not None
        for value in (
            native_task_successes,
            native_task_known,
            native_faulted_episodes,
            native_fault_recovery_successes,
        )
    )
    native_aggregates_valid = bool(
        native_aggregates_present
        and native_task_successes <= native_task_known <= episode_count
        and native_fault_recovery_successes <= native_faulted_episodes <= episode_count
    )
    native_episode_contract = bool(
        len(derived) == episode_count
        and all(
            isinstance(row.get("truth_audited_task_success"), bool)
            and isinstance(row.get("truth_audited_task_success_evidence_known"), bool)
            for row, _success, _known in derived
        )
    )
    if native_aggregates_valid:
        task_successes = int(native_task_successes)
        task_known = int(native_task_known)
        faulted_episodes = int(native_faulted_episodes)
        fault_recovery_successes = int(native_fault_recovery_successes)
        contract_execution = "native_terminal_independent_truth_audit"
    else:
        task_successes = sum(success for _row, success, _known in derived)
        task_known = sum(known for _row, _success, known in derived)
        faulted_episodes = sum(
            _episode_faulted(row) for row, _success, _known in derived
        )
        fault_recovery_successes = sum(
            success and _episode_faulted(row) for row, success, _known in derived
        )
        contract_execution = (
            "native_episode_recompute_after_invalid_aggregates"
            if native_aggregates_present and native_episode_contract
            else "invalid_native_aggregates_fail_closed_backfill"
            if native_aggregates_present
            else "native_terminal_independent_truth_audit"
            if native_episode_contract
            else "conservative_legacy_evidence_backfill"
        )
    native_contract = contract_execution.startswith("native_")
    native_fault_presence_known = _nonnegative_int(
        overall.get("truth_fault_presence_known_episodes")
    )
    fault_presence_known = (
        native_fault_presence_known
        if native_fault_presence_known is not None
        and native_fault_presence_known <= episode_count
        else sum(
            _episode_fault_presence_known(row) for row, _success, _known in derived
        )
    )
    task_by_outcome = overall.get("truth_audited_task_success_by_terminal_outcome")
    if not isinstance(task_by_outcome, Mapping) or not native_aggregates_valid:
        counts: Counter[str] = Counter()
        for row, success, _known in derived:
            if not success:
                continue
            outcome = row.get("terminal_outcome")
            counts[str(outcome) if outcome is not None else "nonterminal"] += 1
        task_by_outcome = dict(sorted(counts.items()))
    accepted_targets, corrected_targets = _target_progress(episodes)
    completion_count = _nonnegative_int(overall.get("audited_completion_episodes"))
    completion_count = (
        completion_count
        if completion_count is not None and completion_count <= episode_count
        else 0
    )
    return {
        "episodes": episode_count,
        "primary_success_metric": "truth_audited_task_success",
        "success_contract": TRUTH_AUDITED_TASK_SUCCESS_CONTRACT,
        "contract_execution": contract_execution,
        "historical_backfill_lower_bound": not native_contract,
        "truth_audited_task_success": {
            "episodes": task_successes,
            "eligible_episodes": episode_count,
            "rate": _rate(task_successes, episode_count),
            "wilson_95_interval": _wilson_95(task_successes, episode_count),
            "evidence_known_episodes": task_known,
            "evidence_coverage_rate": _rate(task_known, episode_count),
            "evidence_coverage_interpretation": (
                "all episodes received the native final-state audit"
                if native_contract
                else "episodes with persisted evidence sufficient to prove success; "
                "historical failures were not reclassified without a native audit"
            ),
            "by_actual_terminal_outcome": dict(task_by_outcome),
            "terminal_label_independent": True,
            "standard_physical_fault_requirement": (
                "all true faults identified and corrected within clean tolerance"
            ),
            "diagnostic_fault_requirement": (
                "correct family and localization under an explicit "
                "explanation-only diagnostic contract"
            ),
            "requires_healthy_components_preserved": True,
        },
        "truth_audited_fault_recovery_success": {
            "episodes": fault_recovery_successes,
            "eligible_faulted_episodes": faulted_episodes,
            "rate": _rate(fault_recovery_successes, faulted_episodes),
            "wilson_95_interval": _wilson_95(
                fault_recovery_successes, faulted_episodes
            ),
            "fault_presence_known_episodes": fault_presence_known,
            "fault_presence_evidence_coverage_rate": _rate(
                fault_presence_known, episode_count
            ),
        },
        "accepted_true_fault_target_coverage": accepted_targets,
        "truth_audited_fault_target_correction": corrected_targets,
        "safe_completion": {
            "episodes": completion_count,
            "rate": _rate(completion_count, episode_count),
            "requires_safety_clean_lifecycle": True,
        },
        "strict_resolved_physical_success": {
            "episodes": int(overall.get("final_physical_success_episodes") or 0),
            "rate": overall.get("final_physical_success_rate"),
            "terminal_outcome": "resolved",
            "requires_strict_physical_audit": True,
        },
        "audited_post_correction_handoff": {
            "episodes": int(
                overall.get("audited_post_correction_handoff_episodes") or 0
            ),
            "rate": overall.get("audited_post_correction_handoff_rate"),
            "terminal_outcome": "operator_escalation",
            "requires_versioned_safety_clean_assessment": True,
        },
        "audited_completion_union": {
            "episodes": completion_count,
            "rate": _rate(completion_count, episode_count),
        },
    }


def _d0_predecessor(path: Path, adapter_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_adapter = payload.get("adapter") if isinstance(payload, Mapping) else {}
    if not (
        isinstance(payload, Mapping)
        and payload.get("contract") == RESEARCH_BC0_EVAL_CONTRACT
        and payload.get("phase") == "d0"
        and payload.get("evaluation_completed") is True
        and payload.get("readiness_gate", {}).get("passed") is True
        and isinstance(recorded_adapter, Mapping)
        and recorded_adapter.get("content_sha256") == adapter_id
    ):
        raise GateError("D1 requires a ready D0 report for the same adapter bytes")


def _progress(record: Mapping[str, Any]) -> None:
    if record.get("event") == "episode_complete":
        print(json.dumps(dict(record), sort_keys=True), file=sys.stderr, flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("d0", "d1"), required=True)
    result.add_argument("--adapter-path", required=True, type=Path)
    result.add_argument("--d0-train", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--d0-report", type=Path)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    result.add_argument(
        "--candidate-multiplier", type=int, default=DEFAULT_CANDIDATE_MULTIPLIER
    )
    return result


def run(
    args: argparse.Namespace,
    *,
    policy_loader: Callable[..., Any] | None = None,
    evaluator: Callable[..., Any] | None = None,
    environment_factory: Callable[..., Any] | None = None,
    case_loader: Callable[[Any], Any] | None = None,
    expert_factory: Callable[[], Any] | None = None,
    generator_factory: Callable[..., Any] | None = None,
    partitioner: Callable[..., Mapping[str, Any]] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = _progress,
) -> dict[str, Any]:
    _configure_input_ceiling()
    adapter = args.adapter_path.expanduser().resolve(strict=True)
    d0_train = args.d0_train.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    adapter_identity = adapter_content_fingerprint(adapter)
    d0_roots, d0_rows = load_d0_training_roots(d0_train)
    frozen = load_frozen_standard_suite()
    frozen_roots = {_scenario_root(row) for row in frozen}
    if d0_roots & frozen_roots:
        raise GateError("D0 training overlaps the packaged standard holdout")

    if args.phase == "d0":
        if args.d0_report is not None:
            raise GateError("--d0-report is only valid for phase=d1")
        scenarios = frozen
        suite = {
            "kind": "packaged_d0_standard_success",
            "path": str(DEFAULT_D0_SUITE),
            "content_sha256": stable_json_sha256(frozen),
            "physical_roots": sorted(frozen_roots),
            "root_isolation": {"d0_training_overlap": []},
        }
        report_path = output / "research_bc0_d0_eval.json"
    else:
        if args.d0_report is None:
            raise GateError("phase=d1 requires --d0-report")
        _d0_predecessor(
            args.d0_report.expanduser().resolve(strict=True),
            adapter_identity["content_sha256"],
        )
        scenarios, suite = build_d1_development_suite(
            d0_training_roots=d0_roots,
            frozen_standard_roots=frozen_roots,
            seed=args.seed,
            candidate_multiplier=args.candidate_multiplier,
            generator_factory=generator_factory,
            partitioner=partitioner,
        )
        suite_payload = {STANDARD_SUITE_NAME: scenarios}
        suite_path = output / "d1_development_suite.json"
        _atomic_json(suite_path, suite_payload)
        suite.update(
            kind="deterministic_d1_development",
            path=str(suite_path),
            content_sha256=stable_json_sha256(suite_payload),
        )
        report_path = output / "research_bc0_d1_eval.json"

    resolved_case_loader = case_loader
    if resolved_case_loader is None:
        from psse_env.dagger.release_factories import deterministic_case_loader

        resolved_case_loader = deterministic_case_loader

    evaluation, behavior = evaluate_research_suite(
        scenarios,
        adapter_path=adapter,
        seed=args.seed,
        max_steps=args.max_steps,
        policy_loader=policy_loader,
        evaluator=evaluator,
        environment_factory=environment_factory,
        case_loader=resolved_case_loader,
        expert_factory=expert_factory,
        progress_callback=progress_callback,
    )
    schema_rate = behavior["schema_valid_action_rate"]
    failures: list[str] = []
    if behavior["episodes"] != len(scenarios):
        failures.append("episode_count_mismatch")
    if behavior["evaluator_error_episodes"]:
        failures.append("evaluator_errors")
    if (
        not isinstance(schema_rate, (int, float))
        or schema_rate < SCHEMA_VALID_READINESS_RATE
    ):
        failures.append("schema_valid_action_rate_below_0.90")

    report = {
        "contract": RESEARCH_BC0_EVAL_CONTRACT,
        "phase": args.phase,
        "evaluation_completed": True,
        "passed": not failures,
        "adapter": {"path": str(adapter), **adapter_identity},
        "model": {
            "model_id": GEMMA4_12B.model_id,
            "revision": GEMMA4_12B.revision,
            "architecture": GEMMA4_12B.architecture,
            "prompt_profile": GEMMA4_12B.prompt_profile,
        },
        "max_input_tokens": REQUIRED_MAX_INPUT_TOKENS,
        "evaluation_wiring": {
            "case_loader": _callable_identity(resolved_case_loader),
        },
        "d0_training": {
            "path": str(d0_train),
            "rows": d0_rows,
            "physical_root_count": len(d0_roots),
        },
        "suite": suite,
        "policy_behavior": behavior,
        "closed_loop_outcomes": summarize_closed_loop_outcomes(evaluation),
        "readiness_gate": {
            "passed": not failures,
            "minimum_schema_valid_action_rate": SCHEMA_VALID_READINESS_RATE,
            "failures": failures,
        },
        "evaluation": evaluation,
    }
    _atomic_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except Exception as exc:
        print(
            json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr
        )
        return 2
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "phase": report["phase"],
                "closed_loop_outcomes": report["closed_loop_outcomes"],
                "readiness_gate": report["readiness_gate"],
                "policy_behavior": report["policy_behavior"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_D1_PLAN",
    "EXPECTED_D0_STANDARD_ROOTS",
    "RESEARCH_BC0_EVAL_CONTRACT",
    "adapter_content_fingerprint",
    "build_d1_development_suite",
    "evaluate_research_suite",
    "load_d0_training_roots",
    "load_frozen_standard_suite",
    "main",
    "run",
    "summarize_closed_loop_outcomes",
    "summarize_policy_behavior",
]
