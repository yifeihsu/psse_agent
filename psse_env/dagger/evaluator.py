from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RecoveryMetrics:
    final_physical_success: float = 0.0
    false_finalization: float = 0.0
    healthy_component_corruption: float = 0.0
    forced_error_recovery: float = 0.0
    tool_regret: float = 0.0
    partial_success_retention: float = 0.0
    false_rollback: float = 0.0
    false_commit: float = 0.0
    loop_rate: float = 0.0

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


def recovery_score(
    metrics: RecoveryMetrics | Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    values = metrics.as_dict() if isinstance(metrics, RecoveryMetrics) else dict(metrics)
    score_weights = dict(weights or DEFAULT_SCORE_WEIGHTS)
    return sum(float(values.get(key, 0.0) or 0.0) * weight for key, weight in score_weights.items())


def make_evaluation_result(
    metrics: RecoveryMetrics | Mapping[str, Any],
    *,
    suite_metrics: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
) -> EvaluationResult:
    typed = metrics if isinstance(metrics, RecoveryMetrics) else RecoveryMetrics(
        **{key: float(value) for key, value in metrics.items() if key in RecoveryMetrics.__dataclass_fields__}
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
