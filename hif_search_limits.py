"""Hard execution limits for model-controlled HIF parameter searches.

The public tool schemas advertise these limits, but schemas are not a trusted
execution boundary.  Provider and estimator entry points import the same
constants and reject out-of-budget values before any OpenDSS work begins.
"""

from __future__ import annotations

import operator
from typing import Any


HIF_ALPHA_GRID_SIZE_MIN = 2
HIF_ALPHA_GRID_SIZE_MAX = 31
HIF_R_GRID_SIZE_MIN = 2
HIF_R_GRID_SIZE_MAX = 35
HIF_MAX_SCANS_MIN = 1
HIF_MAX_SCANS_MAX = 10


def _bounded_integer(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    """Return an integer inside the closed interval or fail without clamping."""

    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be an integer in [{minimum}, {maximum}], got {value!r}"
        )
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{field} must be an integer in [{minimum}, {maximum}], got {value!r}"
        ) from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(
            f"{field} must be in [{minimum}, {maximum}], got {parsed}"
        )
    return parsed


def validate_hif_search_limits(
    *,
    alpha_grid_size: Any,
    r_grid_size: Any,
    max_scans: Any | None = None,
    alpha_grid_size_max: int = HIF_ALPHA_GRID_SIZE_MAX,
    r_grid_size_max: int = HIF_R_GRID_SIZE_MAX,
    max_scans_max: int = HIF_MAX_SCANS_MAX,
) -> tuple[int, int, int | None]:
    """Validate HIF search dimensions against absolute or tighter local caps."""

    if not HIF_ALPHA_GRID_SIZE_MIN <= alpha_grid_size_max <= HIF_ALPHA_GRID_SIZE_MAX:
        raise ValueError(
            "alpha_grid_size_max must be within the absolute HIF search limits"
        )
    if not HIF_R_GRID_SIZE_MIN <= r_grid_size_max <= HIF_R_GRID_SIZE_MAX:
        raise ValueError(
            "r_grid_size_max must be within the absolute HIF search limits"
        )
    if not HIF_MAX_SCANS_MIN <= max_scans_max <= HIF_MAX_SCANS_MAX:
        raise ValueError("max_scans_max must be within the absolute HIF search limits")

    alpha = _bounded_integer(
        alpha_grid_size,
        field="alpha_grid_size",
        minimum=HIF_ALPHA_GRID_SIZE_MIN,
        maximum=alpha_grid_size_max,
    )
    resistance = _bounded_integer(
        r_grid_size,
        field="r_grid_size",
        minimum=HIF_R_GRID_SIZE_MIN,
        maximum=r_grid_size_max,
    )
    scans = (
        None
        if max_scans is None
        else _bounded_integer(
            max_scans,
            field="max_scans",
            minimum=HIF_MAX_SCANS_MIN,
            maximum=max_scans_max,
        )
    )
    return alpha, resistance, scans


__all__ = [
    "HIF_ALPHA_GRID_SIZE_MAX",
    "HIF_ALPHA_GRID_SIZE_MIN",
    "HIF_MAX_SCANS_MAX",
    "HIF_MAX_SCANS_MIN",
    "HIF_R_GRID_SIZE_MAX",
    "HIF_R_GRID_SIZE_MIN",
    "validate_hif_search_limits",
]
