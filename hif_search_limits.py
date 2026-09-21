"""Hard execution limits for model-controlled HIF parameter searches.

The public tool schemas advertise these limits, but schemas are not a trusted
execution boundary.  Provider and estimator entry points import the same
constants and reject out-of-budget values before any OpenDSS work begins.
"""

from __future__ import annotations

import math
import operator
from numbers import Real
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


def _optional_positive_float(value: Any, *, field: str) -> float | None:
    """Return ``None`` or a finite positive float; booleans and strings are rejected."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite positive number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be a finite positive number, got {value!r}")
    return parsed


def validate_hif_resistance_box(
    *,
    r_hif_pu_min: Any | None = None,
    r_hif_pu_max: Any | None = None,
    r_hif_ohm_min: Any | None = None,
    r_hif_ohm_max: Any | None = None,
) -> dict[str, float | None]:
    """Validate a model-supplied HIF resistance search box without resolving it.

    The box may be given in physical ohms (on the candidate line's local
    voltage base) or in per unit, never both. Each unit needs both bounds,
    every bound must be a finite positive number and ``min < max``. Nothing is
    clamped or defaulted here: the estimators resolve an absent box against
    their own defaults. Returns the validated bounds as floats (or ``None``).
    """

    pu_min = _optional_positive_float(r_hif_pu_min, field="r_hif_pu_min")
    pu_max = _optional_positive_float(r_hif_pu_max, field="r_hif_pu_max")
    ohm_min = _optional_positive_float(r_hif_ohm_min, field="r_hif_ohm_min")
    ohm_max = _optional_positive_float(r_hif_ohm_max, field="r_hif_ohm_max")
    pu_given = pu_min is not None or pu_max is not None
    ohm_given = ohm_min is not None or ohm_max is not None
    if pu_given and ohm_given:
        raise ValueError("Supply the HIF resistance search box in ohms or in pu, not both")
    if pu_given and (pu_min is None or pu_max is None):
        raise ValueError("Both r_hif_pu_min and r_hif_pu_max are required")
    if ohm_given and (ohm_min is None or ohm_max is None):
        raise ValueError("Both r_hif_ohm_min and r_hif_ohm_max are required")
    if pu_given and pu_max <= pu_min:
        raise ValueError("Require 0 < r_hif_pu_min < r_hif_pu_max")
    if ohm_given and ohm_max <= ohm_min:
        raise ValueError("Require 0 < r_hif_ohm_min < r_hif_ohm_max")
    return {
        "r_hif_pu_min": pu_min,
        "r_hif_pu_max": pu_max,
        "r_hif_ohm_min": ohm_min,
        "r_hif_ohm_max": ohm_max,
    }


__all__ = [
    "HIF_ALPHA_GRID_SIZE_MAX",
    "HIF_ALPHA_GRID_SIZE_MIN",
    "HIF_MAX_SCANS_MAX",
    "HIF_MAX_SCANS_MIN",
    "HIF_R_GRID_SIZE_MAX",
    "HIF_R_GRID_SIZE_MIN",
    "validate_hif_resistance_box",
    "validate_hif_search_limits",
]
