"""Pinned balanced IEEE57 deployment contract for collection and replay.

Threshold changes are separate experiments. The generic release factory keeps
its historical IEEE14 defaults; this entry point always selects both tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT, validate_episode_action_limit

from psse_env.dagger.release_factories import production_environment_factory
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.transactional_env import TransactionalPSSEEnv


@dataclass(frozen=True)
class IEEE57RuntimeConfig:
    contract: str = "ieee57_balanced_runtime_v1"
    system: str = "ieee57"
    chi2_alpha: float = 0.05
    normalized_residual_threshold: float = 4.0
    anomaly_rule: str = "chi_square_or_normalized_residual"
    alarm_comparison: str = ">="
    max_steps: int = DEFAULT_EPISODE_ACTION_LIMIT
    history_window: int = 4
    production_dataset_mode: bool = True
    executor_hydrated_corrections: bool = True


IEEE57_RUNTIME_CONFIG = IEEE57RuntimeConfig()


def ieee57_runtime_manifest() -> dict[str, Any]:
    """Return a fresh JSON-serializable copy of the immutable contract."""
    return asdict(IEEE57_RUNTIME_CONFIG)


def validate_ieee57_runtime(
    env: TransactionalPSSEEnv, *, expected_max_steps: int = DEFAULT_EPISODE_ACTION_LIMIT,
) -> dict[str, Any]:
    """Inspect live hooks while allowing an explicit episode-horizon override.

    The ordinary contract remains forty actions. A caller running a deliberate
    shorter episode must name that horizon; detector settings remain pinned.
    """
    config = IEEE57_RUNTIME_CONFIG
    expected_max_steps = validate_episode_action_limit(expected_max_steps)
    owner = getattr(env.wls_runner, "__self__", None)
    if not isinstance(owner, MatpowerDeploymentProviders):
        raise ValueError("IEEE57 runtime requires the MATPOWER deployment WLS provider")
    for field in ("chi2_alpha", "normalized_residual_threshold"):
        if getattr(owner, field, None) != getattr(config, field):
            raise ValueError(f"IEEE57 runtime detector mismatch: {field}")
    if getattr(env, "max_steps", None) != expected_max_steps:
        raise ValueError("IEEE57 runtime environment mismatch: max_steps")
    for field in ("history_window", "production_dataset_mode"):
        if getattr(env, field, None) != getattr(config, field):
            raise ValueError(f"IEEE57 runtime environment mismatch: {field}")
    # Candidate verification invokes this same WLS runner. All other numerical
    # hooks must belong to its provider to prevent split detector configuration.
    for group in (env.context_providers, env.correction_executors):
        for hook in group.values():
            if getattr(hook, "__self__", None) is not owner:
                raise ValueError("IEEE57 runtime has inconsistent deployment providers")
    if getattr(env.candidate_quality_oracle, "mode", None) != "deployment":
        raise ValueError("IEEE57 runtime candidate oracle must use deployment evidence")
    if getattr(env.process_oracle, "executor_hydrated_corrections", None) is not True:
        raise ValueError("IEEE57 runtime requires executor-hydrated correction payloads")
    env.validate_production_configuration()
    manifest = ieee57_runtime_manifest()
    manifest["max_steps"] = expected_max_steps
    return manifest


def validate_ieee57_wls_metrics(metrics: Mapping[str, Any]) -> None:
    """Fail when actual initial/verification/committed WLS drifts from v1."""
    config = IEEE57_RUNTIME_CONFIG
    if (metrics.get("chi_square_alpha") != config.chi2_alpha
            or metrics.get("normalized_residual_threshold") != config.normalized_residual_threshold
            or metrics.get("anomaly_detection_rule") != config.anomaly_rule):
        raise ValueError("WLS metrics do not use the pinned IEEE57 detector")
    for key in ("chi_square_statistic", "chi_square_threshold", "max_normalized_residual"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"WLS metric is not finite nonnegative evidence: {key}")
    if metrics["chi_square_threshold"] <= 0:
        raise ValueError("WLS chi-square threshold must be positive")
    chi_alarm = bool(metrics["chi_square_statistic"] >= metrics["chi_square_threshold"])
    residual_alarm = bool(metrics["max_normalized_residual"] >= config.normalized_residual_threshold)
    if (metrics.get("chi_square_alarm") is not chi_alarm
            or metrics.get("normalized_residual_alarm") is not residual_alarm
            or metrics.get("no_material_anomaly_remaining") is not (not (chi_alarm or residual_alarm))):
        raise ValueError("WLS alarm does not implement the pinned IEEE57 OR rule")


def ieee57_environment_factory(
    *, seed: int | None = None, rng: Any | None = None,
) -> TransactionalPSSEEnv:
    """Uniform evaluator/collector/replay factory with no implicit defaults."""
    env = production_environment_factory(
        seed=seed, rng=rng, chi2_alpha=IEEE57_RUNTIME_CONFIG.chi2_alpha,
        normalized_residual_threshold=IEEE57_RUNTIME_CONFIG.normalized_residual_threshold,
    )
    validate_ieee57_runtime(env)
    return env


def ieee57_expert_oracle_factory() -> ExpertPolicyOracle:
    """Use the production target-only correction contract during selection.

    A default ProcessValidityOracle expects caller-supplied replacement values
    and would reject all provider-supported suspect_group-only actions before
    the actual deployment executor could estimate those values.
    """
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(
        executor_hydrated_corrections=IEEE57_RUNTIME_CONFIG.executor_hydrated_corrections,
    ))
