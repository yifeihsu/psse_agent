"""The balanced WLS alarm and the scenario admission rule, as pure functions of one solve.

The operator's alarm is the deployed detector: the chi-square statistic at
its ``alpha`` threshold or a normalized residual at ``residual_threshold``.
Scenario admission asks for the alarm with a margin to spare.  Two margin
rules exist:

* ``chi_square_margin`` (the 2026-09 rule): J above ``margin`` times the
  chi-square threshold, or max |r| above ``margin`` times the residual
  threshold.  The chi-square branch scales with the network: the excess
  J - dof it demands is 67.5 on IEEE 14 (dof 95) and 340 on IEEE 118 (dof
  863), so the same error passes on the small network and fails on the
  large one.
* ``excess_energy`` (2026-10-11): the alarm, and max |r| above ``margin``
  times the residual threshold or an excess J - dof of at least
  ``excess_floor``.  The excess is the residual energy an error injects,
  which does not depend on the network's size, so a given error is admitted
  on every network or on none.  The floor 70 is the chi-square margin's own
  demand on IEEE 14, so the IEEE 14 populations are unchanged.
"""
from __future__ import annotations

import math

from scipy.stats import chi2

DEFAULT_CHI2_ALPHA = 0.01
DEFAULT_RESIDUAL_THRESHOLD = 4.0
DEFAULT_ANOMALY_MARGIN = 1.25
#: Excess objective J - dof an admitted scenario must inject under the ``excess_energy`` rule.
DEFAULT_EXCESS_FLOOR = 70.0
ADMISSION_RULES = ("chi_square_margin", "excess_energy")


def chi_square_threshold(dof: int, alpha: float = DEFAULT_CHI2_ALPHA) -> float:
    return float(chi2.ppf(1.0 - alpha, max(1, int(dof))))


def operator_alarm(objective: float, dof: int, max_residual: float, *, alpha: float = DEFAULT_CHI2_ALPHA,
                   residual_threshold: float | None = DEFAULT_RESIDUAL_THRESHOLD) -> bool:
    """The deployed detector: chi-square at ``alpha`` or a normalized residual at ``residual_threshold``."""
    if objective >= chi_square_threshold(dof, alpha):
        return True
    return bool(residual_threshold is not None and max_residual >= residual_threshold)


def admitted(objective: float, dof: int, max_residual: float, *, rule: str = "chi_square_margin",
             alpha: float = DEFAULT_CHI2_ALPHA, residual_threshold: float | None = DEFAULT_RESIDUAL_THRESHOLD,
             margin: float = DEFAULT_ANOMALY_MARGIN, excess_floor: float = DEFAULT_EXCESS_FLOOR) -> bool:
    """Whether one solve admits a scenario under ``rule`` (see the module docstring)."""
    if rule not in ADMISSION_RULES:
        raise ValueError(f"admission rule must be one of {ADMISSION_RULES}, got {rule!r}")
    if not all(math.isfinite(v) for v in (objective, max_residual)):
        return False
    residual_margin = bool(residual_threshold is not None and max_residual >= margin * residual_threshold)
    if rule == "chi_square_margin":
        return bool(objective > margin * chi_square_threshold(dof, alpha) or residual_margin)
    if not operator_alarm(objective, dof, max_residual, alpha=alpha, residual_threshold=residual_threshold):
        return False
    return bool(residual_margin or objective - dof >= excess_floor)


def describe(objective: float, dof: int, max_residual: float, *, rule: str = "chi_square_margin",
             alpha: float = DEFAULT_CHI2_ALPHA, residual_threshold: float | None = DEFAULT_RESIDUAL_THRESHOLD,
             margin: float = DEFAULT_ANOMALY_MARGIN, excess_floor: float = DEFAULT_EXCESS_FLOOR) -> str:
    """The solve against the rule's thresholds, for rejection messages."""
    threshold = chi_square_threshold(dof, alpha)
    if rule == "excess_energy":
        text = f"chi2 {objective:.1f} vs {threshold:.1f} (excess {objective - dof:.1f} vs floor {excess_floor:.1f})"
    else:
        text = f"chi2 {objective:.1f} vs {margin:.2f} x {threshold:.1f}"
    if residual_threshold is not None:
        text += f", max|r| {max_residual:.2f} vs {margin:.2f} x {residual_threshold:.2f}"
    return text
