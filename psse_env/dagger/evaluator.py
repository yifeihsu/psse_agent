"""Reproducible, truth-isolated closed-loop recovery evaluation.

The evaluator deliberately has two phases at every step:

1. build and validate a policy-only observation, then ask the policy to act;
2. after the action is fixed, inspect oracle state for offline safety scoring.

Consequently, scenario truth, candidate disposition, cost labels, and physical
audit callbacks are never passed to the policy.  The same evaluator can be
used for a rule policy, a base model, or a trained checkpoint by changing only
``policy_factory``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    DIAGNOSTIC_TOOLS,
    FINALIZE_DIAGNOSIS,
    INVALID_ACTION,
    ROLLBACK_STATE,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    invalid_action,
    safe_normalize_action,
)
from psse_env.dagger.dataset_builder import TOOL_JSON_SCHEMAS, validate_policy_payload
from psse_env.dagger.release_audit import (
    ACCEPTED_TARGETS_CHECK,
    DIAGNOSTIC_FAMILY_CHECK,
    FINAL_CASE_CHECK,
    FINAL_MEASUREMENTS_CHECK,
    HEALTHY_CASE_CHECK,
    HEALTHY_MEASUREMENTS_CHECK,
    REMAINING_FAULTS_CHECK,
    audit_episode_against_truth as strict_audit_episode_against_truth,
)
from psse_env.state_store import OracleState, PolicyObservation, policy_safe_copy
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256


@dataclass(frozen=True)
class RecoveryMetrics:
    """Aggregate rates used for checkpoint selection.

    Existing fields retain their original meaning.  The additional fields make
    terminal handoff, physical correctness, and operational efficiency visible
    without requiring callers to unpack ``suite_metrics``.
    """

    final_physical_success: float = 0.0
    false_finalization: float = 0.0
    healthy_component_corruption: float = 0.0
    forced_error_recovery: float = 0.0
    tool_regret: float = 0.0
    partial_success_retention: float = 0.0
    false_rollback: float = 0.0
    false_commit: float = 0.0
    loop_rate: float = 0.0
    final_physical_correctness: float = 0.0
    terminal_rate: float = 0.0
    resolution_rate: float = 0.0
    operator_escalation_rate: float = 0.0
    healthy_component_preservation: float = 0.0
    invalid_action_recovery: float = 0.0
    mean_wls_calls: float = 0.0
    mean_specialized_tool_calls: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "final_physical_success": 4.0,
    "false_finalization": -5.0,
    "healthy_component_corruption": -5.0,
    "forced_error_recovery": 2.0,
    "tool_regret": -0.25,
    "partial_success_retention": 2.0,
    "false_rollback": -2.0,
    "false_commit": -4.0,
    "loop_rate": -2.0,
}


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    metrics: RecoveryMetrics
    suite_metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "metrics": self.metrics.as_dict(),
            "suite_metrics": copy.deepcopy(self.suite_metrics),
        }


@dataclass(frozen=True)
class EpisodeEvaluation:
    """Serializable record for one closed-loop physical root."""

    episode_key: str
    scenario_id: str
    suite: str
    family: str
    cardinality: int | str
    case: str
    split: str
    source_tier: str
    physical_root: str
    seed: int
    steps: int
    terminal: bool
    terminal_outcome: str | None
    final_physical_correct: bool
    physical_correctness_known: bool
    final_physical_success: bool
    healthy_components_preserved: bool
    healthy_preservation_known: bool
    false_commit_count: int
    false_rollback_count: int
    false_finalization_count: int
    partial_fix_count: int
    retained_partial_fix_count: int
    invalid_action_count: int
    recovered_invalid_action_count: int
    loop_detected: bool
    wls_calls: int
    specialized_tool_calls: int
    tool_counts: dict[str, int]
    specialized_tool_counts: dict[str, int]
    tool_regret_total: float
    tool_regret_samples: int
    audit: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    evaluator_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def recovery_score(
    metrics: RecoveryMetrics | Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    values = metrics.as_dict() if isinstance(metrics, RecoveryMetrics) else dict(metrics)
    score_weights = dict(weights or DEFAULT_SCORE_WEIGHTS)
    return sum(
        float(values.get(key, 0.0) or 0.0) * weight
        for key, weight in score_weights.items()
    )


def make_evaluation_result(
    metrics: RecoveryMetrics | Mapping[str, Any],
    *,
    suite_metrics: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
) -> EvaluationResult:
    typed = (
        metrics
        if isinstance(metrics, RecoveryMetrics)
        else RecoveryMetrics(
            **{
                key: float(value)
                for key, value in metrics.items()
                if key in RecoveryMetrics.__dataclass_fields__
            }
        )
    )
    return EvaluationResult(
        score=recovery_score(typed, weights=weights),
        metrics=typed,
        suite_metrics=dict(suite_metrics or {}),
    )


EVALUATION_SUITES = (
    "standard_success",
    "forced_error_recovery",
    "partial_success_retention",
    "invalid_action_recovery",
    "efficiency",
)


PhysicalAudit = Callable[[Mapping[str, Any]], Mapping[str, Any] | bool]
ToolCostResolver = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
CaseLoader = Callable[[Any], Any]


class ClosedLoopRolloutEvaluator:
    """Execute fixed scenario suites against freshly constructed policies.

    ``env_factory`` and ``policy_factory`` are called once per episode.  They
    may accept a keyword-only ``seed`` and/or ``rng`` argument, but never
    receive the scenario or its truth.  ``physical_audit_fn`` and
    ``tool_cost_resolver`` run only after the policy action has been selected;
    their single context mapping may therefore contain offline truth.

    A physical audit callback can override the conservative built-in audit by
    returning any of these keys: ``final_physical_correct``,
    ``physical_correctness_known``, ``healthy_components_preserved``,
    ``healthy_preservation_known``, and ``partial_fixes_retained``.
    """

    def __init__(
        self,
        *,
        env_factory: Callable[..., Any],
        policy_factory: Callable[..., Any],
        max_steps: int = 24,
        seed: int = 0,
        weights: Mapping[str, float] | None = None,
        physical_audit_fn: PhysicalAudit | None = None,
        tool_cost_resolver: ToolCostResolver | None = None,
        case_loader: CaseLoader | None = None,
        required_suites: Iterable[str] | None = None,
        minimum_suites: int = 1,
        minimum_episodes_per_suite: int = 1,
        minimum_roots_per_suite: int = 1,
    ) -> None:
        if not callable(env_factory) or not callable(policy_factory):
            raise TypeError("env_factory and policy_factory must be callable.")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        if case_loader is not None and not callable(case_loader):
            raise TypeError("case_loader must be callable when supplied.")
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.weights = dict(weights) if weights is not None else None
        self.physical_audit_fn = physical_audit_fn
        self.tool_cost_resolver = tool_cost_resolver
        self.case_loader = case_loader
        self.required_suites = _normalize_required_suites(required_suites)
        self.minimum_suites = _positive_integer(
            minimum_suites, field="minimum_suites"
        )
        self.minimum_episodes_per_suite = _positive_integer(
            minimum_episodes_per_suite, field="minimum_episodes_per_suite"
        )
        self.minimum_roots_per_suite = _positive_integer(
            minimum_roots_per_suite, field="minimum_roots_per_suite"
        )

    def evaluate(
        self,
        scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
        | Iterable[Mapping[str, Any]],
    ) -> EvaluationResult:
        suites = _normalize_suites(scenario_suites)
        suite_manifest = _validate_and_fingerprint_suites(
            suites,
            required_suites=self.required_suites,
            minimum_suites=self.minimum_suites,
            minimum_episodes_per_suite=self.minimum_episodes_per_suite,
            minimum_roots_per_suite=self.minimum_roots_per_suite,
        )
        episodes: list[EpisodeEvaluation] = []
        for suite_name in sorted(suites):
            ordered = sorted(
                enumerate(suites[suite_name]),
                key=lambda item: (
                    _scenario_id(item[1], item[0]),
                    _stable_hash(item[1]),
                ),
            )
            occurrence_by_id: Counter[str] = Counter()
            for original_index, scenario in ordered:
                scenario_id = _scenario_id(scenario, original_index)
                occurrence = occurrence_by_id[scenario_id]
                occurrence_by_id[scenario_id] += 1
                episode_seed = _episode_seed(
                    self.seed,
                    suite_name,
                    scenario_id,
                    occurrence,
                )
                episodes.append(
                    self._run_episode(
                        suite=suite_name,
                        scenario=scenario,
                        scenario_index=occurrence,
                        episode_seed=episode_seed,
                    )
                )

        overall = summarize_episode_evaluations(episodes)
        grouped = {
            dimension: _group_episodes(episodes, attribute)
            for dimension, attribute in (
                ("suite", "suite"),
                ("family", "family"),
                ("cardinality", "cardinality"),
                ("case", "case"),
                ("split", "split"),
                ("source_tier", "source_tier"),
                ("physical_root", "physical_root"),
            )
        }
        report = {
            "schema_version": 2,
            "configuration": {
                "seed": self.seed,
                "max_steps": self.max_steps,
                "suite_names": sorted(suites),
                "episode_order": [episode.episode_key for episode in episodes],
                "required_suites": list(self.required_suites),
                "minimum_suites": self.minimum_suites,
                "minimum_episodes_per_suite": self.minimum_episodes_per_suite,
                "minimum_roots_per_suite": self.minimum_roots_per_suite,
                **suite_manifest,
            },
            "overall": overall,
            "suites": grouped["suite"],
            "groups": grouped,
            # Named aliases keep JSON reports convenient for downstream jobs.
            **{f"by_{name}": values for name, values in grouped.items()},
            "episodes": [episode.as_dict() for episode in episodes],
        }
        metrics = _recovery_metrics(overall)
        return make_evaluation_result(
            metrics,
            suite_metrics=report,
            weights=self.weights,
        )

    def evaluate_suites(
        self,
        scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
        | Iterable[Mapping[str, Any]],
    ) -> EvaluationResult:
        """Alias retained for call sites that prefer an explicit suite verb."""

        return self.evaluate(scenario_suites)

    def _run_episode(
        self,
        *,
        suite: str,
        scenario: Mapping[str, Any],
        scenario_index: int,
        episode_seed: int,
    ) -> EpisodeEvaluation:
        scenario_copy = copy.deepcopy(dict(scenario))
        env = _call_factory(self.env_factory, episode_seed)
        policy = _call_factory(self.policy_factory, episode_seed)
        env.reset(copy.deepcopy(scenario_copy))
        initial_state = _current_state(env)
        history: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        tool_counts: Counter[str] = Counter()
        specialized_counts: Counter[str] = Counter()
        seen_signatures: set[str] = set()
        loop_detected = False
        invalid_indices: list[int] = []
        advancing_indices: list[int] = []
        false_commits = 0
        false_rollbacks = 0
        false_finalizations = 0
        deferred_finalizations = 0
        partial_candidate_ids: list[str] = []
        partial_action_signatures: list[str] = []
        collateral_commit_seen = False
        regret_total = 0.0
        regret_samples = 0
        evaluator_error: str | None = None

        for step in range(self.max_steps):
            false_finalization_this_step = False
            observation = _policy_observation(env, history)
            # This check is repeated even for PolicyObservation implementations
            # so custom environments cannot accidentally expand the boundary.
            validate_policy_payload(observation)
            try:
                raw_action = _policy_action(policy, observation)
            except Exception as exc:  # malformed learner behavior is measurable
                raw_action = invalid_action(
                    "policy_exception", f"{type(exc).__name__}: {exc}"
                )
            action = safe_normalize_action(raw_action)

            # Offline state is intentionally obtained only after the policy has
            # committed to its action.
            pre_oracle = _oracle_state(env, history)
            disposition = _candidate_disposition(pre_oracle)
            pre_remaining = _remaining_fault_count(pre_oracle)
            state_before_action = _current_state(env)
            tool = action["tool"]
            signature = action_signature(action)
            if signature in seen_signatures:
                loop_detected = True
            seen_signatures.add(signature)
            tool_counts[tool] += 1
            if tool in DIAGNOSTIC_TOOLS:
                specialized_counts[tool] += 1

            if tool == COMMIT_STATE:
                if disposition not in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
                    false_commits += 1
                assessment = _candidate_assessment(pre_oracle)
            elif tool == ROLLBACK_STATE:
                if disposition in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
                    false_rollbacks += 1
            elif tool == FINALIZE_DIAGNOSIS:
                pre_diagnostic = _diagnostic_truth_audit(
                    _scenario_truth(scenario_copy),
                    observation.get("explained_anomalies") or [],
                )
                known_false = pre_remaining is not None and pre_remaining > 0
                known_false = known_false or not pre_diagnostic[
                    "diagnostic_truth_matched"
                ]
                if known_false:
                    false_finalizations += 1
                    false_finalization_this_step = True
                elif pre_remaining is None:
                    deferred_finalizations += 1

            try:
                next_state, tool_output = env.step(copy.deepcopy(action))
                if not isinstance(tool_output, Mapping):
                    raise TypeError("env.step() tool output must be a mapping")
                output = copy.deepcopy(dict(tool_output))
            except Exception as exc:
                evaluator_error = f"env_step:{type(exc).__name__}"
                output = {
                    "execution_status": "failure",
                    "error_code": "evaluator_env_step_exception",
                    "error_detail": type(exc).__name__,
                    "state_mutated": False,
                }
                next_state = _current_state(env)

            status = str(output.get("execution_status") or "failure")
            if (
                tool == FINALIZE_DIAGNOSIS
                and status != "success"
                and not false_finalization_this_step
            ):
                false_finalizations += 1
                false_finalization_this_step = True
                if pre_remaining is None:
                    deferred_finalizations = max(0, deferred_finalizations - 1)
            if tool == COMMIT_STATE and status == "success":
                if disposition == "ACCEPT_PARTIAL":
                    candidate_id = _candidate_id(pre_oracle, observation)
                    if candidate_id:
                        partial_candidate_ids.append(candidate_id)
                    partial_action_signatures.append(signature)
                if assessment.get("collateral_damage") is True or assessment.get(
                    "healthy_component_modified"
                ) is True:
                    collateral_commit_seen = True
            invalid = tool == INVALID_ACTION or status != "success"
            if invalid:
                invalid_indices.append(step)
            advanced = bool(
                not invalid
                and _successful_action_advanced(
                    before=state_before_action,
                    after=next_state,
                    output=output,
                    terminal=_is_terminal(env, next_state),
                )
            )
            if advanced:
                advancing_indices.append(step)

            label = _resolve_cost_label(
                self.tool_cost_resolver,
                scenario=scenario_copy,
                suite=suite,
                step=step,
                observation=observation,
                action=action,
                tool_output=output,
                oracle_state=pre_oracle,
                env=env,
            )
            regret = _tool_regret(label, action)
            if regret is not None:
                regret_total += regret
                regret_samples += 1

            transition = {
                "state_id": observation.get("active_state_id"),
                "candidate_state_id": observation.get("candidate_state_id"),
                "action": policy_safe_copy(action),
                "tool_output": policy_safe_copy(output),
            }
            history.append(transition)
            trace.append(
                {
                    "step": step,
                    "observation_hash": _stable_hash(observation),
                    "action": policy_safe_copy(action),
                    "execution_status": status,
                    "advanced": advanced,
                    "error_code": output.get("error_code"),
                    "candidate_disposition_offline": disposition,
                    "tool_regret": regret,
                    "terminal_outcome": _output_terminal_outcome(output),
                }
            )
            if evaluator_error or _is_terminal(env, next_state):
                break

        final_state = _current_state(env)
        final_oracle = _oracle_state(env, history)
        terminal = _is_terminal(env, final_state)
        outcome = _terminal_outcome(env, trace)
        active_physical_state = _active_physical_state(env, final_state)
        default_audit = _default_physical_audit(
            scenario=scenario_copy,
            initial_state=initial_state,
            final_state=final_state,
            final_oracle=final_oracle,
            history=history,
            collateral_commit_seen=collateral_commit_seen,
            terminal=terminal,
            terminal_outcome=outcome,
            active_physical_state=active_physical_state,
            case_loader=self.case_loader,
        )
        audit = dict(default_audit)
        if self.physical_audit_fn is not None:
            supplied = self.physical_audit_fn(
                {
                    "scenario": copy.deepcopy(scenario_copy),
                    "suite": suite,
                    "initial_state": copy.deepcopy(initial_state),
                    "final_state": copy.deepcopy(final_state),
                    "final_oracle_state": copy.deepcopy(final_oracle),
                    "history": copy.deepcopy(history),
                    "terminal": terminal,
                    "terminal_outcome": outcome,
                    "active_physical_state": copy.deepcopy(active_physical_state),
                    "environment": env,
                    "default_audit": copy.deepcopy(default_audit),
                }
            )
            if isinstance(supplied, bool):
                supplied = {
                    "final_physical_correct": supplied,
                    "physical_correctness_known": True,
                }
            if not isinstance(supplied, Mapping):
                raise TypeError("physical_audit_fn must return a mapping or bool.")
            audit.update(copy.deepcopy(dict(supplied)))

        physical_known = bool(audit.get("physical_correctness_known", False))
        physical_correct = physical_known and bool(
            audit.get("final_physical_correct", False)
        )
        healthy_known = bool(audit.get("healthy_preservation_known", False))
        healthy_preserved = healthy_known and bool(
            audit.get("healthy_components_preserved", False)
        )
        if deferred_finalizations and physical_known and not physical_correct:
            false_finalizations += deferred_finalizations

        accepted = _accepted_corrections(final_state)
        accepted_ids = {
            str(item.get("candidate_state_id"))
            for item in accepted
            if item.get("candidate_state_id") is not None
        }
        accepted_signatures = {
            action_signature(_correction_action(item))
            for item in accepted
            if _correction_action(item).get("tool")
        }
        retained_partial = sum(
            candidate_id in accepted_ids for candidate_id in partial_candidate_ids
        )
        if len(partial_candidate_ids) < len(partial_action_signatures):
            retained_partial += sum(
                signature in accepted_signatures
                for signature in partial_action_signatures[len(partial_candidate_ids) :]
            )
        if audit.get("partial_fixes_retained") is not None:
            override = audit["partial_fixes_retained"]
            if isinstance(override, bool):
                retained_partial = len(partial_action_signatures) if override else 0
            else:
                retained_partial = max(
                    0, min(int(override), len(partial_action_signatures))
                )

        groups = _scenario_groups(scenario_copy)
        scenario_id = _scenario_id(scenario_copy, scenario_index)
        episode_key = f"{suite}:{scenario_id}:{scenario_index}"
        final_success = bool(
            terminal and outcome == "resolved" and physical_correct
        )
        recovered_invalid = (
            sum(
                any(
                    advancing_index > invalid_index
                    for advancing_index in advancing_indices
                )
                for invalid_index in invalid_indices
            )
            if final_success
            else 0
        )
        return EpisodeEvaluation(
            episode_key=episode_key,
            scenario_id=scenario_id,
            suite=suite,
            family=groups["family"],
            cardinality=groups["cardinality"],
            case=groups["case"],
            split=groups["split"],
            source_tier=groups["source_tier"],
            physical_root=groups["physical_root"],
            seed=episode_seed,
            steps=len(trace),
            terminal=terminal,
            terminal_outcome=outcome,
            final_physical_correct=physical_correct,
            physical_correctness_known=physical_known,
            final_physical_success=final_success,
            healthy_components_preserved=healthy_preserved,
            healthy_preservation_known=healthy_known,
            false_commit_count=false_commits,
            false_rollback_count=false_rollbacks,
            false_finalization_count=false_finalizations,
            partial_fix_count=len(partial_action_signatures),
            retained_partial_fix_count=retained_partial,
            invalid_action_count=len(invalid_indices),
            recovered_invalid_action_count=recovered_invalid,
            loop_detected=loop_detected,
            wls_calls=sum(tool_counts[tool] for tool in (RUN_WLS, VERIFY_CANDIDATE)),
            specialized_tool_calls=sum(specialized_counts.values()),
            tool_counts=dict(sorted(tool_counts.items())),
            specialized_tool_counts=dict(sorted(specialized_counts.items())),
            tool_regret_total=regret_total,
            tool_regret_samples=regret_samples,
            audit=copy.deepcopy(audit),
            trace=trace,
            evaluator_error=evaluator_error,
        )


def evaluate_rollout_suites(
    scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
    | Iterable[Mapping[str, Any]],
    *,
    env_factory: Callable[..., Any],
    policy_factory: Callable[..., Any],
    max_steps: int = 24,
    seed: int = 0,
    weights: Mapping[str, float] | None = None,
    physical_audit_fn: PhysicalAudit | None = None,
    tool_cost_resolver: ToolCostResolver | None = None,
    case_loader: CaseLoader | None = None,
    required_suites: Iterable[str] | None = None,
    minimum_suites: int = 1,
    minimum_episodes_per_suite: int = 1,
    minimum_roots_per_suite: int = 1,
) -> EvaluationResult:
    """Functional entry point for closed-loop suite evaluation."""

    return ClosedLoopRolloutEvaluator(
        env_factory=env_factory,
        policy_factory=policy_factory,
        max_steps=max_steps,
        seed=seed,
        weights=weights,
        physical_audit_fn=physical_audit_fn,
        tool_cost_resolver=tool_cost_resolver,
        case_loader=case_loader,
        required_suites=required_suites,
        minimum_suites=minimum_suites,
        minimum_episodes_per_suite=minimum_episodes_per_suite,
        minimum_roots_per_suite=minimum_roots_per_suite,
    ).evaluate(scenario_suites)


# A descriptive alias makes the entry point easy to discover without breaking
# callers that use the shorter name above.
ClosedLoopEvaluator = ClosedLoopRolloutEvaluator
evaluate_closed_loop_rollouts = evaluate_rollout_suites
evaluate_closed_loop = evaluate_rollout_suites


_RELEASE_SOURCE_FAILURE = (
    "source worktree is not a clean tracked commit; use --allow-dirty-source "
    "only for non-release development evidence"
)
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _source_path_descriptor(
    path: str | os.PathLike[str], *, repo_root: Path
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        displayed = str(resolved.relative_to(repo_root))
        location = "repository"
    except ValueError:
        displayed = str(resolved)
        location = "external"
    return {
        "path": displayed,
        "location": location,
        "sha256": file_sha256(resolved) if resolved.is_file() else None,
    }


def _callable_descriptor(
    spec: str,
    value: Callable[..., Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    source_path: str | None = None
    try:
        source_path = inspect.getsourcefile(value) or inspect.getfile(value)
    except (OSError, TypeError):
        source_path = None
    return {
        "import_spec": str(spec).strip(),
        "module": getattr(value, "__module__", None),
        "qualname": getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__qualname__),
        ),
        "source": (
            _source_path_descriptor(source_path, repo_root=repo_root)
            if source_path is not None
            else None
        ),
    }


def _protocol_registry_descriptor(protocol: str) -> dict[str, Any]:
    normalized = str(protocol).strip().lower()
    if normalized == "canonical":
        from psse_env.dagger.protocol_bridge import unified_tool_schemas

        registry = unified_tool_schemas()
    elif normalized == "controller":
        registry = copy.deepcopy(TOOL_JSON_SCHEMAS)
    else:
        raise ValueError("protocol must be canonical or controller")
    names = [str(row["function"]["name"]) for row in registry]
    return {
        "protocol": normalized,
        "registry_sha256": stable_json_sha256(registry),
        "registered_tool_count": len(names),
        "registered_tools": names,
    }


def _evaluation_provenance_failures(provenance: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(provenance, Mapping):
        return ["evaluation identity provenance is missing"]

    failures: list[str] = []
    source_state = provenance.get("source_state")
    if not isinstance(source_state, Mapping) or source_state.get(
        "release_eligible_source"
    ) is not True:
        failures.append(_RELEASE_SOURCE_FAILURE)

    input_suite = provenance.get("input_suite")
    if not isinstance(input_suite, Mapping):
        failures.append("input suite path and hash are missing")
    else:
        if not str(input_suite.get("resolved_path") or "").strip():
            failures.append("input suite resolved path is missing")
        if not _is_sha256(input_suite.get("sha256")):
            failures.append("input suite SHA-256 is missing or invalid")

    factories = provenance.get("factories")
    factories = factories if isinstance(factories, Mapping) else {}
    for field in ("environment", "policy"):
        descriptor = factories.get(field)
        if not isinstance(descriptor, Mapping) or not str(
            descriptor.get("import_spec") or ""
        ).strip():
            failures.append(f"{field} factory import spec is missing")
            continue
        source = descriptor.get("source")
        if not isinstance(source, Mapping) or not _is_sha256(source.get("sha256")):
            failures.append(f"{field} factory source fingerprint is missing")
    case_loader = factories.get("case_loader")
    if case_loader is not None:
        if not isinstance(case_loader, Mapping) or not str(
            case_loader.get("import_spec") or ""
        ).strip():
            failures.append("case-loader import spec is missing")
        else:
            source = case_loader.get("source")
            if not isinstance(source, Mapping) or not _is_sha256(
                source.get("sha256")
            ):
                failures.append("case-loader source fingerprint is missing")

    policy_identity = provenance.get("policy_identity")
    if not isinstance(policy_identity, Mapping):
        failures.append("policy identity is missing")
    else:
        explicit = str(policy_identity.get("explicit_policy_identity") or "").strip()
        model_id = str(policy_identity.get("model_id") or "").strip()
        revision = str(policy_identity.get("model_revision") or "").strip()
        if bool(model_id) != bool(revision):
            failures.append("model ID and immutable model revision must be supplied together")
        if revision and _IMMUTABLE_REVISION.fullmatch(revision) is None:
            failures.append("model revision is not an immutable 40- or 64-hex digest")
        if not explicit and not (model_id and revision):
            failures.append(
                "policy identity requires an explicit identity or a model ID/revision pair"
            )

    protocol_registry = provenance.get("protocol_registry")
    if not isinstance(protocol_registry, Mapping):
        failures.append("model-visible protocol registry identity is missing")
    else:
        if protocol_registry.get("protocol") not in {"canonical", "controller"}:
            failures.append("model-visible protocol is missing or unsupported")
        if not _is_sha256(protocol_registry.get("registry_sha256")):
            failures.append("model-visible protocol registry SHA-256 is missing")

    evaluator_source = provenance.get("evaluator_source")
    if not isinstance(evaluator_source, Mapping) or not _is_sha256(
        evaluator_source.get("sha256")
    ):
        failures.append("evaluator source fingerprint is missing")
    return failures


def build_evaluation_provenance(
    *,
    input_suite_path: str | os.PathLike[str],
    environment_factory_spec: str,
    environment_factory: Callable[..., Any],
    policy_factory_spec: str,
    policy_factory: Callable[..., Any],
    case_loader_spec: str | None = None,
    case_loader: Callable[..., Any] | None = None,
    policy_identity: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    protocol: str = "canonical",
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build the immutable identity envelope for one closed-loop evaluation.

    Library callers may pass this envelope to :func:`write_evaluation_artifact`.
    The CLI always does so and refuses release execution when the envelope is
    incomplete or the source tree is dirty.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    suite_path = Path(input_suite_path).expanduser().resolve(strict=True)
    if bool(case_loader_spec) != (case_loader is not None):
        raise ValueError("case_loader_spec and case_loader must be supplied together")
    explicit_identity = str(policy_identity or "").strip() or None
    normalized_model_id = str(model_id or "").strip() or None
    normalized_revision = str(model_revision or "").strip() or None
    core: dict[str, Any] = {
        "provenance_schema_version": 1,
        "source_state": git_source_state(root),
        "input_suite": {
            "provided_path": str(Path(input_suite_path).expanduser()),
            "resolved_path": str(suite_path),
            "sha256": file_sha256(suite_path),
            "size_bytes": suite_path.stat().st_size,
        },
        "factories": {
            "environment": _callable_descriptor(
                environment_factory_spec,
                environment_factory,
                repo_root=root,
            ),
            "policy": _callable_descriptor(
                policy_factory_spec,
                policy_factory,
                repo_root=root,
            ),
            "case_loader": (
                _callable_descriptor(case_loader_spec, case_loader, repo_root=root)
                if case_loader_spec is not None and case_loader is not None
                else None
            ),
        },
        "policy_identity": {
            "explicit_policy_identity": explicit_identity,
            "model_id": normalized_model_id,
            "model_revision": normalized_revision,
        },
        "protocol_registry": _protocol_registry_descriptor(protocol),
        "evaluator_source": _source_path_descriptor(__file__, repo_root=root),
    }
    core["identity_sha256"] = stable_json_sha256(core)
    failures = _evaluation_provenance_failures(core)
    return {
        **core,
        "release_eligible": not failures,
        "release_failures": failures,
    }


def write_evaluation_artifact(
    result: EvaluationResult,
    output_path: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist a deterministic closed-loop release report.

    The original two-argument library call remains valid.  Such an artifact is
    explicitly non-release because a bare :class:`EvaluationResult` cannot
    identify the executed policy, factories, source tree, or input suite.
    """

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    output = Path(output_path).expanduser()
    if not output.name:
        raise ValueError("output_path must name a JSON artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    recorded_provenance = (
        copy.deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else None
    )
    release_failures = _evaluation_provenance_failures(recorded_provenance)
    if recorded_provenance is not None:
        recorded_provenance["release_eligible"] = not release_failures
        recorded_provenance["release_failures"] = release_failures
    payload: dict[str, Any] = {
        "artifact_schema_version": 2,
        "artifact_type": "closed_loop_release_evaluation",
        "release_eligible": not release_failures,
        "release_failures": release_failures,
        "provenance": recorded_provenance,
        "evaluation": result.as_dict(),
    }
    payload["content_sha256"] = _stable_hash(payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return copy.deepcopy(payload)


def _load_import_spec(spec: str, *, field: str) -> Callable[..., Any]:
    normalized = str(spec).strip()
    module_name, separator, attribute_path = normalized.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"{field} must use MODULE:ATTRIBUTE syntax")
    value: Any = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{field} must resolve to a callable")
    return value


def _load_scenario_suite_file(path: str | os.PathLike[str]) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, (list, dict)):
        raise ValueError("scenario suite JSON must contain a list or object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run closed-loop suites from JSON and persist a release artifact."""

    parser = argparse.ArgumentParser(
        description="Evaluate frozen closed-loop scenario suites and write JSON evidence."
    )
    parser.add_argument("--input", required=True, help="JSON suite mapping or scenario list")
    parser.add_argument("--output", required=True, help="Destination release JSON artifact")
    parser.add_argument(
        "--env-factory", required=True, help="Environment factory as MODULE:ATTRIBUTE"
    )
    parser.add_argument(
        "--policy-factory", required=True, help="Policy factory as MODULE:ATTRIBUTE"
    )
    parser.add_argument(
        "--case-loader", help="Optional physical-case loader as MODULE:ATTRIBUTE"
    )
    parser.add_argument(
        "--policy-identity",
        help=(
            "Explicit immutable policy identity for non-model policies, such as "
            "rule-expert-v3. Required unless --model-id and --model-revision are set."
        ),
    )
    parser.add_argument(
        "--model-id",
        help="Model/checkpoint repository or immutable checkpoint name.",
    )
    parser.add_argument(
        "--model-revision",
        help="Pinned 40-character Git or 64-character content digest.",
    )
    parser.add_argument(
        "--protocol",
        choices=("canonical", "controller"),
        default="canonical",
        help="Exact model-visible tool registry used by the evaluated policy.",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Development only: run from a dirty/untracked source tree, while "
            "marking the persisted artifact release-ineligible."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--required-suite", action="append", default=None)
    parser.add_argument("--minimum-suites", type=int, default=1)
    parser.add_argument("--minimum-episodes-per-suite", type=int, default=1)
    parser.add_argument("--minimum-roots-per-suite", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    explicit_policy_identity = str(args.policy_identity or "").strip()
    model_id = str(args.model_id or "").strip()
    model_revision = str(args.model_revision or "").strip()
    if bool(model_id) != bool(model_revision):
        parser.error("--model-id and --model-revision must be supplied together")
    if model_revision and _IMMUTABLE_REVISION.fullmatch(model_revision) is None:
        parser.error(
            "--model-revision must be an immutable 40- or 64-character hex digest"
        )
    if not explicit_policy_identity and not (model_id and model_revision):
        parser.error(
            "release evaluation requires --policy-identity or both --model-id "
            "and --model-revision"
        )

    required_suites = (
        tuple(args.required_suite)
        if args.required_suite is not None
        else EVALUATION_SUITES
    )
    environment_factory = _load_import_spec(
        args.env_factory, field="env_factory"
    )
    policy_factory = _load_import_spec(
        args.policy_factory, field="policy_factory"
    )
    case_loader = (
        _load_import_spec(args.case_loader, field="case_loader")
        if args.case_loader
        else None
    )
    provenance = build_evaluation_provenance(
        input_suite_path=args.input,
        environment_factory_spec=args.env_factory,
        environment_factory=environment_factory,
        policy_factory_spec=args.policy_factory,
        policy_factory=policy_factory,
        case_loader_spec=args.case_loader,
        case_loader=case_loader,
        policy_identity=explicit_policy_identity,
        model_id=model_id,
        model_revision=model_revision,
        protocol=args.protocol,
    )
    provenance_failures = list(provenance["release_failures"])
    blocking_failures = [
        failure
        for failure in provenance_failures
        if failure != _RELEASE_SOURCE_FAILURE or not args.allow_dirty_source
    ]
    if blocking_failures:
        raise RuntimeError(
            "Closed-loop release identity gate failed: "
            + "; ".join(blocking_failures)
        )

    result = evaluate_rollout_suites(
        _load_scenario_suite_file(args.input),
        env_factory=environment_factory,
        policy_factory=policy_factory,
        max_steps=args.max_steps,
        seed=args.seed,
        case_loader=case_loader,
        required_suites=required_suites,
        minimum_suites=args.minimum_suites,
        minimum_episodes_per_suite=args.minimum_episodes_per_suite,
        minimum_roots_per_suite=args.minimum_roots_per_suite,
    )
    artifact = write_evaluation_artifact(
        result,
        args.output,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "score": result.score,
                "content_sha256": artifact["content_sha256"],
                "release_eligible": artifact["release_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


def summarize_episode_evaluations(
    episodes: Iterable[EpisodeEvaluation],
) -> dict[str, Any]:
    rows = list(episodes)
    total = len(rows)
    terminal = sum(row.terminal for row in rows)
    resolved = sum(
        row.terminal and row.terminal_outcome == "resolved" for row in rows
    )
    escalated = sum(
        row.terminal and row.terminal_outcome == "operator_escalation"
        for row in rows
    )
    physical_known = sum(row.physical_correctness_known for row in rows)
    physical_correct = sum(row.final_physical_correct for row in rows)
    physical_success = sum(row.final_physical_success for row in rows)
    healthy_known = sum(row.healthy_preservation_known for row in rows)
    healthy_preserved = sum(row.healthy_components_preserved for row in rows)
    false_commit_count = sum(row.false_commit_count for row in rows)
    false_rollback_count = sum(row.false_rollback_count for row in rows)
    false_finalization_count = sum(row.false_finalization_count for row in rows)
    partial_count = sum(row.partial_fix_count for row in rows)
    retained_partial = sum(row.retained_partial_fix_count for row in rows)
    invalid_count = sum(row.invalid_action_count for row in rows)
    recovered_invalid = sum(row.recovered_invalid_action_count for row in rows)
    regret_samples = sum(row.tool_regret_samples for row in rows)
    regret_total = sum(row.tool_regret_total for row in rows)
    tool_counts: Counter[str] = Counter()
    specialized_counts: Counter[str] = Counter()
    for row in rows:
        tool_counts.update(row.tool_counts)
        specialized_counts.update(row.specialized_tool_counts)
    return {
        "episodes": total,
        "steps": sum(row.steps for row in rows),
        "mean_steps": _rate(sum(row.steps for row in rows), total),
        "terminal_episodes": terminal,
        "terminal_rate": _rate(terminal, total),
        "resolved_episodes": resolved,
        "resolution_rate": _rate(resolved, total),
        "operator_escalation_episodes": escalated,
        "operator_escalation_rate": _rate(escalated, total),
        "unknown_terminal_outcome_episodes": sum(
            row.terminal and row.terminal_outcome not in {"resolved", "operator_escalation"}
            for row in rows
        ),
        "physical_correctness_known_episodes": physical_known,
        "physical_correct_episodes": physical_correct,
        # The all-episode rate fails closed when truth/audit evidence is absent.
        "final_physical_correctness_rate": _rate(physical_correct, total),
        "final_physical_correctness_known_rate": _rate(
            physical_correct, physical_known
        ),
        "final_physical_success_episodes": physical_success,
        "final_physical_success_rate": _rate(physical_success, total),
        "healthy_preservation_known_episodes": healthy_known,
        "healthy_component_preservation_episodes": healthy_preserved,
        "healthy_component_preservation_rate": _rate(healthy_preserved, total),
        "healthy_component_preservation_known_rate": _rate(
            healthy_preserved, healthy_known
        ),
        "healthy_component_corruption_episodes": sum(
            row.healthy_preservation_known
            and not row.healthy_components_preserved
            for row in rows
        ),
        "healthy_component_corruption_rate": _rate(
            sum(
                row.healthy_preservation_known
                and not row.healthy_components_preserved
                for row in rows
            ),
            total,
        ),
        "false_commit_count": false_commit_count,
        "false_commit_episodes": sum(row.false_commit_count > 0 for row in rows),
        "false_commit_rate": _rate(
            sum(row.false_commit_count > 0 for row in rows), total
        ),
        "false_rollback_count": false_rollback_count,
        "false_rollback_episodes": sum(row.false_rollback_count > 0 for row in rows),
        "false_rollback_rate": _rate(
            sum(row.false_rollback_count > 0 for row in rows), total
        ),
        "false_finalization_count": false_finalization_count,
        "false_finalization_episodes": sum(
            row.false_finalization_count > 0 for row in rows
        ),
        "false_finalization_rate": _rate(
            sum(row.false_finalization_count > 0 for row in rows), total
        ),
        "partial_fix_opportunities": partial_count,
        "retained_partial_fixes": retained_partial,
        "partial_fix_retention_rate": _rate(retained_partial, partial_count),
        "invalid_action_count": invalid_count,
        "recovered_invalid_actions": recovered_invalid,
        "invalid_action_recovery_rate": _rate(recovered_invalid, invalid_count),
        "episodes_with_invalid_actions": sum(
            row.invalid_action_count > 0 for row in rows
        ),
        "loop_episodes": sum(row.loop_detected for row in rows),
        "loop_rate": _rate(sum(row.loop_detected for row in rows), total),
        "wls_calls": sum(row.wls_calls for row in rows),
        "mean_wls_calls": _rate(sum(row.wls_calls for row in rows), total),
        "specialized_tool_calls": sum(row.specialized_tool_calls for row in rows),
        "mean_specialized_tool_calls": _rate(
            sum(row.specialized_tool_calls for row in rows), total
        ),
        "tool_counts": dict(sorted(tool_counts.items())),
        "specialized_tool_counts": dict(sorted(specialized_counts.items())),
        "tool_regret_samples": regret_samples,
        "tool_regret_total": regret_total,
        "mean_tool_regret": (
            regret_total / regret_samples if regret_samples else None
        ),
        "tool_regret_coverage": _rate(
            regret_samples, sum(row.steps for row in rows)
        ),
        "evaluator_error_episodes": sum(row.evaluator_error is not None for row in rows),
    }


def _recovery_metrics(summary: Mapping[str, Any]) -> RecoveryMetrics:
    return RecoveryMetrics(
        final_physical_success=float(summary["final_physical_success_rate"]),
        false_finalization=float(summary["false_finalization_rate"]),
        healthy_component_corruption=float(
            summary["healthy_component_corruption_rate"]
        ),
        forced_error_recovery=float(summary["invalid_action_recovery_rate"]),
        tool_regret=float(summary["mean_tool_regret"] or 0.0),
        partial_success_retention=float(summary["partial_fix_retention_rate"]),
        false_rollback=float(summary["false_rollback_rate"]),
        false_commit=float(summary["false_commit_rate"]),
        loop_rate=float(summary["loop_rate"]),
        final_physical_correctness=float(
            summary["final_physical_correctness_rate"]
        ),
        terminal_rate=float(summary["terminal_rate"]),
        resolution_rate=float(summary["resolution_rate"]),
        operator_escalation_rate=float(summary["operator_escalation_rate"]),
        healthy_component_preservation=float(
            summary["healthy_component_preservation_rate"]
        ),
        invalid_action_recovery=float(summary["invalid_action_recovery_rate"]),
        mean_wls_calls=float(summary["mean_wls_calls"]),
        mean_specialized_tool_calls=float(summary["mean_specialized_tool_calls"]),
    )


def _group_episodes(
    episodes: Sequence[EpisodeEvaluation], attribute: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EpisodeEvaluation]] = {}
    for episode in episodes:
        key = str(getattr(episode, attribute))
        grouped.setdefault(key, []).append(episode)
    return {
        key: summarize_episode_evaluations(grouped[key])
        for key in sorted(grouped)
    }


def _normalize_suites(
    scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
    | Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    raw = (
        dict(scenario_suites)
        if isinstance(scenario_suites, Mapping)
        else {"standard_success": scenario_suites}
    )
    if not raw:
        raise ValueError("evaluation requires at least one non-empty suite")
    suites: dict[str, list[dict[str, Any]]] = {}
    for name, scenarios in raw.items():
        suite_name = str(name).strip()
        if not suite_name:
            raise ValueError("evaluation suite names must be non-empty")
        if suite_name in suites:
            raise ValueError(f"duplicate normalized evaluation suite name: {suite_name}")
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, Mapping):
                raise TypeError(
                    f"Scenario {suite_name}[{index}] must be a mapping."
                )
            rows.append(copy.deepcopy(dict(scenario)))
        if not rows:
            raise ValueError(f"evaluation suite {suite_name!r} is empty")
        suites[suite_name] = rows
    return suites


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed < 1 or not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _normalize_required_suites(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("required_suites must be an iterable of suite names")
    names = [str(item).strip() for item in value]
    if any(not name for name in names):
        raise ValueError("required_suites cannot contain an empty name")
    if len(set(names)) != len(names):
        raise ValueError("required_suites cannot contain duplicates")
    return tuple(sorted(names))


def _validate_and_fingerprint_suites(
    suites: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_suites: Sequence[str],
    minimum_suites: int,
    minimum_episodes_per_suite: int,
    minimum_roots_per_suite: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(suites) < minimum_suites:
        errors.append(
            f"suite_count={len(suites)} < minimum_suites={minimum_suites}"
        )
    missing = sorted(set(required_suites) - set(suites))
    if missing:
        errors.append("missing_required_suites=" + ",".join(missing))

    manifest: dict[str, dict[str, Any]] = {}
    all_roots: set[str] = set()
    for suite_name in sorted(suites):
        rows = list(suites[suite_name])
        ordered = sorted(
            enumerate(rows),
            key=lambda item: (
                _scenario_id(item[1], item[0]),
                _stable_hash(item[1]),
            ),
        )
        ordered_payload = [row for _, row in ordered]
        roots = sorted(
            {_scenario_groups(row)["physical_root"] for row in ordered_payload}
        )
        all_roots.update(roots)
        if len(rows) < minimum_episodes_per_suite:
            errors.append(
                f"{suite_name}: episodes={len(rows)} < "
                f"minimum_episodes_per_suite={minimum_episodes_per_suite}"
            )
        if len(roots) < minimum_roots_per_suite:
            errors.append(
                f"{suite_name}: distinct_roots={len(roots)} < "
                f"minimum_roots_per_suite={minimum_roots_per_suite}"
            )
        manifest[suite_name] = {
            "episodes": len(rows),
            "distinct_physical_roots": len(roots),
            "content_sha256": _stable_hash(ordered_payload),
            "root_set_sha256": _stable_hash(roots),
        }
    if errors:
        raise ValueError("evaluation suite coverage failed: " + "; ".join(errors))

    content_hashes = {
        name: details["content_sha256"] for name, details in manifest.items()
    }
    root_hashes = {
        name: details["root_set_sha256"] for name, details in manifest.items()
    }
    return {
        "suite_manifest": manifest,
        "suite_content_hashes": content_hashes,
        "suite_root_set_hashes": root_hashes,
        "suite_content_sha256": _stable_hash(content_hashes),
        "root_set_sha256": _stable_hash(sorted(all_roots)),
        "suite_coverage_validation": {
            "passed": True,
            "suite_count": len(suites),
            "distinct_physical_roots": len(all_roots),
        },
    }


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    explicit = scenario.get("scenario_id", scenario.get("id"))
    if explicit is not None:
        return str(explicit)
    # A content-derived fallback keeps evaluation invariant to input ordering.
    return f"scenario_{_stable_hash(scenario)[:12]}"


def _episode_seed(base_seed: int, suite: str, scenario_id: str, index: int) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [int(base_seed), str(suite), str(scenario_id), int(index)],
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _call_factory(factory: Callable[..., Any], seed: int) -> Any:
    """Call a factory without ever supplying scenario data."""

    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "seed" in parameters or accepts_kwargs:
        kwargs["seed"] = int(seed)
    if "rng" in parameters or accepts_kwargs:
        kwargs["rng"] = random.Random(seed)
    return factory(**kwargs)


def _policy_action(policy: Any, observation: Mapping[str, Any]) -> Any:
    safe_observation = copy.deepcopy(dict(observation))
    if hasattr(policy, "act"):
        return policy.act(safe_observation)
    if hasattr(policy, "next_actions"):
        # ExpertPolicyOracle supports policy-observation input directly.  Do
        # not substitute an OracleState here: the expert comparator must obey
        # the same observation boundary as learned policies.
        actions = policy.next_actions(
            safe_observation,
            copy.deepcopy(list(safe_observation.get("history_window") or [])),
        )
        return actions[0] if actions else invalid_action("policy_returned_no_action")
    if callable(policy):
        return policy(safe_observation)
    raise TypeError(
        "Policy must be callable or expose .act(observation) / .next_actions(observation)."
    )


def _policy_observation(
    env: Any, history: list[Mapping[str, Any]]
) -> dict[str, Any]:
    if hasattr(env, "get_policy_observation"):
        raw = _call_with_optional_argument(env.get_policy_observation, history)
        if isinstance(raw, PolicyObservation):
            return raw.as_dict()
        if isinstance(raw, Mapping):
            return copy.deepcopy(dict(raw))
        if hasattr(raw, "as_dict"):
            payload = raw.as_dict()
            if isinstance(payload, Mapping):
                return copy.deepcopy(dict(payload))
        raise TypeError("get_policy_observation() must return a mapping.")
    state = _current_state(env)
    return PolicyObservation(
        active_state_id=str(state.get("active_state_id") or "active"),
        candidate_state_id=state.get("candidate_state_id"),
        remaining_budget=int(state.get("remaining_budget") or 0),
        history_window=policy_safe_copy(history),
    ).as_dict()


def _oracle_state(env: Any, history: list[Mapping[str, Any]]) -> Any:
    if not hasattr(env, "get_oracle_state"):
        return None
    return copy.deepcopy(_call_with_optional_argument(env.get_oracle_state, history))


def _current_state(env: Any) -> dict[str, Any]:
    if not hasattr(env, "current_state"):
        return {}
    state = env.current_state()
    if not isinstance(state, Mapping):
        raise TypeError("current_state() must return a mapping.")
    return copy.deepcopy(dict(state))


def _active_physical_state(
    env: Any, final_state: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the final store payload without treating a policy summary as physics."""

    store = getattr(env, "store", None)
    getter = getattr(store, "get_state_for_audit", None)
    if not callable(getter):
        getter = getattr(store, "get_state", None)
    state_id = final_state.get("active_state_id")
    if state_id is None and store is not None:
        state_id = getattr(store, "active_state_id", None)
    if callable(getter) and state_id is not None:
        try:
            payload = getter(str(state_id))
        except (KeyError, TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            return copy.deepcopy(dict(payload))
    if _physical_state_available(final_state):
        return copy.deepcopy(dict(final_state))
    return None


def _successful_action_advanced(
    *,
    before: Mapping[str, Any],
    after: Any,
    output: Mapping[str, Any],
    terminal: bool,
) -> bool:
    """Recognize observable state/decision progress after a successful action."""

    if terminal or output.get("state_mutated") is True:
        return True
    if not isinstance(after, Mapping):
        return False
    for key in ("active_state_id", "candidate_state_id", "phase"):
        if before.get(key) != after.get(key):
            return True
    for key in ("accepted_corrections", "explained_anomalies"):
        before_rows = before.get(key)
        after_rows = after.get(key)
        before_count = (
            len(before_rows)
            if isinstance(before_rows, Sequence)
            and not isinstance(before_rows, (str, bytes))
            else 0
        )
        after_count = (
            len(after_rows)
            if isinstance(after_rows, Sequence)
            and not isinstance(after_rows, (str, bytes))
            else 0
        )
        if after_count > before_count:
            return True
    return False


def _is_terminal(env: Any, state: Mapping[str, Any] | None) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(_call_with_optional_argument(env.is_terminal, state))
    return bool(getattr(env, "terminal", False))


def _call_with_optional_argument(method: Callable[..., Any], value: Any) -> Any:
    """Invoke bound environment methods that support either zero or one input."""

    try:
        parameters = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return method(value)
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return method(value)
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if positional:
        return method(value)
    keyword_only = [
        parameter
        for parameter in parameters
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    if len(keyword_only) == 1:
        return method(**{keyword_only[0].name: value})
    return method()


def _terminal_outcome(env: Any, trace: Sequence[Mapping[str, Any]]) -> str | None:
    outcome = getattr(env, "terminal_outcome", None)
    if outcome:
        return str(outcome)
    for row in reversed(trace):
        candidate = row.get("terminal_outcome")
        if candidate:
            return str(candidate)
    return None


def _output_terminal_outcome(output: Mapping[str, Any]) -> str | None:
    direct = output.get("terminal_outcome")
    metrics = output.get("tool_metrics")
    nested = metrics.get("terminal_outcome") if isinstance(metrics, Mapping) else None
    value = direct if direct is not None else nested
    return str(value) if value is not None else None


def _candidate_disposition(oracle_state: Any) -> str | None:
    value = (
        oracle_state.candidate_disposition
        if isinstance(oracle_state, OracleState)
        else oracle_state.get("candidate_disposition")
        if isinstance(oracle_state, Mapping)
        else getattr(oracle_state, "candidate_disposition", None)
    )
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _candidate_assessment(oracle_state: Any) -> dict[str, Any]:
    value = (
        oracle_state.candidate_assessment
        if isinstance(oracle_state, OracleState)
        else oracle_state.get("candidate_assessment")
        if isinstance(oracle_state, Mapping)
        else getattr(oracle_state, "candidate_assessment", None)
    )
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _candidate_id(
    oracle_state: Any, observation: Mapping[str, Any]
) -> str | None:
    policy_observation = (
        oracle_state.policy_observation
        if isinstance(oracle_state, OracleState)
        else None
    )
    value = (
        policy_observation.candidate_state_id
        if isinstance(policy_observation, PolicyObservation)
        else observation.get("candidate_state_id")
    )
    return str(value) if value is not None else None


def _oracle_truth(oracle_state: Any) -> dict[str, Any]:
    if isinstance(oracle_state, OracleState):
        return oracle_state.truth_dict()
    if isinstance(oracle_state, Mapping):
        hidden = oracle_state.get("hidden_truth")
        truth = copy.deepcopy(dict(hidden)) if isinstance(hidden, Mapping) else {}
        for key, value in oracle_state.items():
            if str(key).startswith("true_") or key in {
                "truth_complete",
                "remaining_true_faults",
                "remaining_true_fault_count",
                "remaining_fault_count",
                "clean_case",
                "clean_measurements",
            }:
                truth[key] = copy.deepcopy(value)
        return truth
    return {}


def _remaining_fault_count(oracle_state: Any) -> int | None:
    truth = _oracle_truth(oracle_state)
    for key in ("remaining_true_fault_count", "remaining_fault_count"):
        if truth.get(key) is not None:
            try:
                return int(truth[key])
            except (TypeError, ValueError):
                return None
    remaining = truth.get("remaining_true_faults")
    if isinstance(remaining, Sequence) and not isinstance(remaining, (str, bytes)):
        return len(remaining)
    return None


def _complete_remaining_truth(oracle_state: Any) -> dict[str, Any] | None:
    truth = _oracle_truth(oracle_state)
    if truth.get("truth_complete") is not True:
        return None
    remaining_rows = truth.get("remaining_true_faults")
    if remaining_rows is not None and (
        not isinstance(remaining_rows, Sequence)
        or isinstance(remaining_rows, (str, bytes))
    ):
        return None
    raw_count = truth.get("remaining_true_fault_count")
    if raw_count is None:
        if remaining_rows is None:
            return None
        count = len(remaining_rows)
        truth["remaining_true_fault_count"] = count
    else:
        try:
            count = int(raw_count)
            numeric_count = float(raw_count)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            count < 0
            or isinstance(raw_count, bool)
            or not math.isfinite(numeric_count)
            or numeric_count != count
        ):
            return None
        if remaining_rows is not None and len(remaining_rows) != count:
            return None
    return truth


def _scenario_truth(scenario: Mapping[str, Any]) -> dict[str, Any]:
    hidden = scenario.get("hidden_truth")
    truth = copy.deepcopy(dict(hidden)) if isinstance(hidden, Mapping) else {}
    metadata = scenario.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    for source in (metadata, scenario):
        for key, value in source.items():
            if str(key).startswith("true_") or key in {
                "truth_complete",
                "clean_case",
                "clean_measurements",
                "remaining_true_faults",
                "remaining_true_fault_count",
            }:
                truth[str(key)] = copy.deepcopy(value)
    if "truth_complete" not in truth:
        truth["truth_complete"] = bool(
            hidden
            or any(str(key).startswith("true_") for key in truth)
            or "clean_case" in truth
            or "clean_measurements" in truth
        )
    return truth


_STRICT_PHYSICAL_EVIDENCE_GAPS = frozenset(
    {
        "healthy_measurement_preservation_evidence_missing_or_malformed",
        "healthy_case_preservation_evidence_missing_or_unloadable",
        "final_clean_measurement_evidence_missing_or_malformed",
        "final_clean_case_evidence_missing_or_unloadable",
        "true_measurement_targets_malformed",
        "true_measurement_target_out_of_range",
    }
)


def _scenario_truth_available(
    scenario: Mapping[str, Any], truth: Mapping[str, Any]
) -> bool:
    containers: list[Mapping[str, Any]] = [scenario]
    for key in ("hidden_truth", "metadata", "clean_state"):
        nested = scenario.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    return bool(
        truth.get("truth_complete") is True
        and (
            any(
                str(key).startswith("true_")
                for container in containers
                for key in container
            )
            or any(
                key in container
                for container in containers
                for key in ("clean_case", "clean_measurements", "clean_state")
            )
        )
    )


def _physical_state_available(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and any(value.get(key) is not None for key in ("case", "measurements"))
    )


def _case_evidence_comparable(
    scenario: Mapping[str, Any],
    active_physical_state: Mapping[str, Any],
    *,
    case_loader: CaseLoader | None,
) -> bool:
    clean_state = scenario.get("clean_state")
    clean_state = clean_state if isinstance(clean_state, Mapping) else {}
    expected = clean_state.get("case", scenario.get("clean_case"))
    observed = active_physical_state.get("case")
    if expected is None or observed is None or case_loader is not None:
        return True
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        return True
    return expected == observed


def _strict_audit_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Promote offline correction truth aliases into the strict audit contract."""

    normalized = copy.deepcopy(dict(scenario))
    hidden = scenario.get("hidden_truth")
    if not isinstance(hidden, Mapping):
        return normalized
    for key in (
        "true_measurement_errors",
        "true_parameter_errors",
        "true_topology_errors",
        "clean_case",
        "clean_measurements",
        "clean_state",
        "truth_complete",
    ):
        if key not in normalized and key in hidden:
            normalized[key] = copy.deepcopy(hidden[key])
    return normalized


def _strict_check_status(checks: Mapping[str, Any], name: str) -> str | None:
    check = checks.get(name)
    return str(check.get("status")) if isinstance(check, Mapping) else None


def _strict_physical_evidence_complete(problems: Sequence[str]) -> bool:
    return not any(
        problem in _STRICT_PHYSICAL_EVIDENCE_GAPS
        or problem.startswith("true_") and problem.endswith("_malformed")
        for problem in problems
    )


def _default_physical_audit(
    *,
    scenario: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    final_oracle: Any,
    history: Sequence[Mapping[str, Any]],
    collateral_commit_seen: bool,
    terminal: bool,
    terminal_outcome: str | None,
    active_physical_state: Mapping[str, Any] | None,
    case_loader: CaseLoader | None,
) -> dict[str, Any]:
    audit_scenario = _strict_audit_scenario(scenario)
    original_truth = _scenario_truth(audit_scenario)
    accepted = _accepted_corrections(final_state)
    target_audit = _accepted_target_audit(original_truth, accepted)
    explanations = list(final_state.get("explained_anomalies") or [])
    diagnostic = _diagnostic_truth_audit(original_truth, explanations)
    base = {
        "physical_correctness_known": False,
        "final_physical_correct": False,
        "healthy_preservation_known": False,
        "healthy_components_preserved": False,
        "remaining_true_fault_count": None,
        "accepted_target_audit": target_audit,
        "diagnostic_truth_audit": diagnostic,
        "initial_active_state_id": initial_state.get("active_state_id"),
        "final_active_state_id": final_state.get("active_state_id"),
        "strict_release_audit": None,
        "audit_mode": "insufficient_evidence",
    }
    evidence_problems: list[str] = []
    if not _scenario_truth_available(audit_scenario, original_truth):
        evidence_problems.append("scenario_truth_unavailable")
    if not _physical_state_available(active_physical_state):
        evidence_problems.append("active_physical_state_unavailable")
    elif not _case_evidence_comparable(
        audit_scenario, active_physical_state, case_loader=case_loader
    ):
        evidence_problems.append("case_loader_required_for_physical_comparison")
    if evidence_problems:
        return {
            **base,
            "problems": evidence_problems,
            "quarantined": True,
        }

    remaining_truth = _complete_remaining_truth(final_oracle)
    strict = strict_audit_episode_against_truth(
        audit_scenario,
        final_state,
        terminal=terminal,
        terminal_outcome=terminal_outcome,
        active_physical_state=active_physical_state,
        remaining_truth=remaining_truth,
        case_loader=case_loader,
    )
    strict_problems = [str(item) for item in strict.get("problems") or []]
    evidence_complete = _strict_physical_evidence_complete(strict_problems)
    resolved = bool(terminal and terminal_outcome == "resolved")
    physical_known = bool(resolved and evidence_complete)
    physical_correct = bool(
        physical_known
        and not strict.get("quarantined", True)
        and not collateral_commit_seen
    )

    checks = strict.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    healthy_statuses = [
        _strict_check_status(checks, HEALTHY_MEASUREMENTS_CHECK),
        _strict_check_status(checks, HEALTHY_CASE_CHECK),
    ]
    healthy_known = bool(
        resolved
        and evidence_complete
        and all(status in {"passed", "failed"} for status in healthy_statuses)
    )
    healthy_preserved = bool(
        healthy_known
        and all(status == "passed" for status in healthy_statuses)
        and target_audit["healthy_targets_preserved"]
        and not collateral_commit_seen
    )
    remaining_check = checks.get(REMAINING_FAULTS_CHECK)
    remaining = (
        remaining_check.get("derived_remaining_fault_count")
        if isinstance(remaining_check, Mapping)
        else None
    )
    return {
        **base,
        "physical_correctness_known": physical_known,
        "final_physical_correct": physical_correct,
        "healthy_preservation_known": healthy_known,
        "healthy_components_preserved": healthy_preserved,
        "remaining_true_fault_count": remaining,
        "strict_release_audit": copy.deepcopy(strict),
        "audit_mode": "strict_release_audit",
        "problems": strict_problems,
        "quarantined": bool(strict.get("quarantined", True)),
        "strict_checks_used": [
            ACCEPTED_TARGETS_CHECK,
            DIAGNOSTIC_FAMILY_CHECK,
            HEALTHY_MEASUREMENTS_CHECK,
            HEALTHY_CASE_CHECK,
            FINAL_MEASUREMENTS_CHECK,
            FINAL_CASE_CHECK,
        ],
        "history_steps_audited": len(history),
    }


def _accepted_corrections(final_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(item))
        for item in final_state.get("accepted_corrections") or []
        if isinstance(item, Mapping)
    ]


def _correction_action(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.get("source_action") or item.get("action") or item
    return safe_normalize_action(raw) if isinstance(raw, Mapping) else invalid_action(
        "accepted_correction_action_missing"
    )


def _measurement_targets(action: Mapping[str, Any]) -> set[int]:
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    targets: set[int] = set()
    group = arguments.get("suspect_group")
    if isinstance(group, Sequence) and not isinstance(group, (str, bytes)):
        for value in group:
            try:
                targets.add(int(value))
            except (TypeError, ValueError):
                continue
    updates = arguments.get("measurement_updates")
    if isinstance(updates, Mapping):
        for value in updates:
            try:
                targets.add(int(value))
            except (TypeError, ValueError):
                continue
    for key in ("measurement_index", "index", "index0", "target"):
        if arguments.get(key) is not None:
            try:
                targets.add(int(arguments[key]))
            except (TypeError, ValueError):
                continue
    return targets


def _branch_row0(value: Mapping[str, Any]) -> int | None:
    for key, offset in (("branch_row0", 0), ("line_index1", -1), ("line_index", -1)):
        if value.get(key) is not None:
            try:
                row = int(value[key]) + offset
            except (TypeError, ValueError):
                return None
            return row if row >= 0 else None
    return None


def _truth_measurement_targets(truth: Mapping[str, Any]) -> set[int]:
    targets: set[int] = set()
    for fault in truth.get("true_measurement_errors") or []:
        if not isinstance(fault, Mapping):
            continue
        for key in ("index", "index0", "measurement_index"):
            if fault.get(key) is not None:
                try:
                    targets.add(int(fault[key]))
                except (TypeError, ValueError):
                    pass
                break
    return targets


def _truth_branch_targets(truth: Mapping[str, Any], family: str) -> set[int]:
    return {
        row
        for item in truth.get(f"true_{family}_errors") or []
        if isinstance(item, Mapping)
        for row in [_branch_row0(item)]
        if row is not None
    }


def _accepted_target_audit(
    truth: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    true_targets: dict[str, set[int]] = {
        "measurement": _truth_measurement_targets(truth),
        "parameter": _truth_branch_targets(truth, "parameter"),
        "topology": _truth_branch_targets(truth, "topology"),
    }
    accepted_targets: dict[str, set[int]] = {
        "measurement": set(),
        "parameter": set(),
        "topology": set(),
    }
    problems: list[str] = []
    for item in accepted:
        action = _correction_action(item)
        tool = action["tool"]
        if tool == CORRECT_MEASUREMENTS:
            family = "measurement"
            targets = _measurement_targets(action)
        elif tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            family = "parameter" if tool == CORRECT_PARAMETERS else "topology"
            row = _branch_row0(action["arguments"])
            targets = {row} if row is not None else set()
        else:
            continue
        accepted_targets[family].update(targets)
        if not targets:
            problems.append(f"{family}_accepted_target_missing")
        elif not targets.issubset(true_targets[family]):
            healthy = sorted(targets - true_targets[family])
            problems.append(f"{family}_healthy_targets_modified:{healthy}")
    uncovered = sum(
        len(true_targets[family] - accepted_targets[family])
        for family in true_targets
    )
    return {
        "true_targets": {
            key: sorted(value) for key, value in true_targets.items()
        },
        "accepted_targets": {
            key: sorted(value) for key, value in accepted_targets.items()
        },
        "healthy_targets_preserved": not problems,
        "uncovered_standard_faults": uncovered,
        "problems": problems,
    }


def _diagnostic_truth_audit(
    truth: Mapping[str, Any], explanations: Sequence[Any]
) -> dict[str, Any]:
    records = [dict(item) for item in explanations if isinstance(item, Mapping)]
    problems: list[str] = []
    checked = 0

    harmonic_truth = [
        item
        for item in truth.get("true_harmonic_errors") or []
        if isinstance(item, Mapping)
    ]
    for item in harmonic_truth:
        checked += 1
        true_bus = item.get("bus_1based", item.get("source_bus"))
        matches = [record for record in records if record.get("family") == "harmonic"]
        if true_bus is not None:
            matches = [
                record
                for record in matches
                if isinstance(record.get("detail"), Mapping)
                and record["detail"].get("bus_1based") is not None
                and int(record["detail"]["bus_1based"]) == int(true_bus)
            ]
        if not matches:
            problems.append("harmonic_localization_mismatch")

    hif_truth = [
        item
        for item in truth.get("true_hif_errors") or []
        if isinstance(item, Mapping)
    ]
    for item in hif_truth:
        checked += 1
        true_row = _branch_row0(item)
        true_phase = item.get("phase")
        matches = [record for record in records if record.get("family") == "hif"]
        if true_row is not None:
            matches = [
                record
                for record in matches
                if isinstance(record.get("detail"), Mapping)
                and record["detail"].get("candidate_branch_row0") is not None
                and int(record["detail"]["candidate_branch_row0"]) == true_row
            ]
        if true_phase is not None:
            phase_matches = []
            for record in matches:
                detail = record.get("detail") or {}
                estimated = detail.get("estimated")
                estimated = estimated if isinstance(estimated, Mapping) else {}
                phase = estimated.get("phase", detail.get("phase"))
                if phase is None or str(phase).upper() == str(true_phase).upper():
                    phase_matches.append(record)
            matches = phase_matches
        if not matches:
            problems.append("hif_localization_mismatch")

    unbalance_keys = (
        "true_three_phase_unbalance_errors",
        "true_unbalance_errors",
        "true_imbalance_errors",
    )
    for key in unbalance_keys:
        for item in truth.get(key) or []:
            checked += 1
            if not any(
                record.get("family") == "three_phase_unbalance"
                for record in records
            ):
                problems.append("three_phase_unbalance_explanation_missing")

    return {
        "checked_diagnostic_faults": checked,
        "diagnostic_truth_matched": not problems,
        "problems": problems,
    }


def _scenario_groups(scenario: Mapping[str, Any]) -> dict[str, Any]:
    metadata = scenario.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    def first(*keys: str, default: Any = "unknown") -> Any:
        for key in keys:
            if scenario.get(key) is not None:
                return scenario[key]
            if metadata.get(key) is not None:
                return metadata[key]
        return default

    cardinality = first("error_cardinality", "cardinality", default=None)
    if cardinality is None:
        truth = _scenario_truth(scenario)
        counted_keys = [
            key
            for key, value in truth.items()
            if key.startswith("true_")
            and key.endswith("_errors")
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ]
        cardinality = sum(len(truth[key]) for key in counted_keys)
        if not counted_keys and not truth.get("truth_complete"):
            cardinality = "unknown"
    case = first("case_id", "network_case", "case", "case_path")
    if isinstance(case, Mapping):
        case = case.get("case_id", case.get("case_path", _stable_hash(case)[:12]))
    scenario_id = _scenario_id(scenario, 0)
    return {
        "family": str(first("scenario_family", "error_family", "family")),
        "cardinality": int(cardinality)
        if isinstance(cardinality, (int, float)) and not isinstance(cardinality, bool)
        else str(cardinality),
        "case": str(case),
        "split": str(first("split", "dataset_split")),
        "source_tier": str(first("source_tier", "data_source_tier")),
        "physical_root": str(
            first(
                "physical_root_fingerprint",
                "physical_root",
                "root_scenario_id",
                default=scenario_id,
            )
        ),
    }


def _resolve_cost_label(
    resolver: ToolCostResolver | None,
    *,
    scenario: Mapping[str, Any],
    suite: str,
    step: int,
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    tool_output: Mapping[str, Any],
    oracle_state: Any,
    env: Any,
) -> Mapping[str, Any] | None:
    context = {
        "scenario": copy.deepcopy(dict(scenario)),
        "suite": suite,
        "step": step,
        "observation": copy.deepcopy(dict(observation)),
        "action": copy.deepcopy(dict(action)),
        "tool_output": copy.deepcopy(dict(tool_output)),
        "oracle_state": copy.deepcopy(oracle_state),
        "environment": env,
    }
    if resolver is not None:
        label = resolver(context)
        if label is not None and not isinstance(label, Mapping):
            raise TypeError("tool_cost_resolver must return a mapping or None.")
        return copy.deepcopy(dict(label)) if isinstance(label, Mapping) else None

    source = scenario.get("evaluation_labels", scenario.get("tool_cost_labels"))
    if source is None and isinstance(scenario.get("metadata"), Mapping):
        source = scenario["metadata"].get(
            "evaluation_labels", scenario["metadata"].get("tool_cost_labels")
        )
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        label = source[step] if step < len(source) else None
    elif isinstance(source, Mapping):
        label = source.get(step, source.get(str(step)))
        if label is None and any(
            key in source
            for key in (
                "action_costs",
                "costs",
                "chosen_cost",
                "action_cost",
                "preferred_action",
            )
        ):
            label = source
    else:
        label = None
    return copy.deepcopy(dict(label)) if isinstance(label, Mapping) else None


def _tool_regret(
    label: Mapping[str, Any] | None, action: Mapping[str, Any]
) -> float | None:
    if not isinstance(label, Mapping):
        return None

    chosen = _first_number(
        label.get("chosen_cost"),
        label.get("action_cost"),
        label.get("executed_cost"),
        label.get("cost"),
    )
    best = _first_number(
        label.get("best_cost"),
        label.get("minimum_cost"),
        label.get("min_cost"),
        label.get("optimal_cost"),
        label.get("expert_cost"),
    )
    costs = label.get("action_costs", label.get("costs"))
    if isinstance(costs, Mapping):
        numeric_costs = {
            str(key): float(value)
            for key, value in costs.items()
            if _is_finite_number(value)
        }
        signature = action_signature(action)
        tool = str(action.get("tool"))
        chosen = numeric_costs.get(signature, numeric_costs.get(tool, chosen))
        if numeric_costs:
            best = min(numeric_costs.values()) if best is None else best

    if chosen is not None and best is not None:
        return max(0.0, chosen - best)

    preferred = label.get("preferred_action")
    margin = _first_number(label.get("cost_margin"), label.get("margin"))
    if preferred is not None and margin is not None:
        preferred_action = safe_normalize_action(preferred)
        return 0.0 if preferred_action == dict(action) else max(0.0, margin)
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if _is_finite_number(value):
            return float(value)
    return None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


__all__ = [
    "ClosedLoopEvaluator",
    "ClosedLoopRolloutEvaluator",
    "DEFAULT_SCORE_WEIGHTS",
    "EVALUATION_SUITES",
    "EpisodeEvaluation",
    "EvaluationResult",
    "RecoveryMetrics",
    "build_evaluation_provenance",
    "evaluate_closed_loop",
    "evaluate_closed_loop_rollouts",
    "evaluate_rollout_suites",
    "main",
    "make_evaluation_result",
    "recovery_score",
    "summarize_episode_evaluations",
    "write_evaluation_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
