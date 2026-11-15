"""The estimator's output, whole (the user's direction of 2026-11-15).

What the policy read until now was what the rule expert's screens kept of each
solve: the five largest residuals and multipliers above a threshold, the lines
a dominance test admits, the breakers whose flip test clears a bar, and for
each family a list of the corrections those screens support.  Those screens
were written to make the rule expert accurate.  A policy that is to pass the
rule expert has to read what the screens read, not what they concluded.

This module writes the same solves out in full, one row per component, in the
case's own order (so a component keeps its place from one solve to the next),
with no threshold, no cut and no ranking:

``solve_tables``
    every bus and every line of a WLS solve: the estimated voltage magnitude
    and angle of each bus, the signed normalized residuals of the voltage
    magnitude and the injections, of the flows at both ends, and the
    normalized multipliers of the line's series R and X.  Signed means
    measured minus estimated (``signed_residuals``); the summaries the policy
    read before carry the magnitude only.
``meter_table``
    every measurement: its index (what ``correct_measurements`` takes), its
    channel, the bus or line it sits on, the measured value and the signed
    normalized residual.
``line_table``
    every line: its buses, the model's series R and X, and their normalized
    multipliers.
``breaker_table``
    every breaker of the node/breaker model: its reported status, its
    normalized multipliers and their score, what flipping it would do to the
    operator model, and for the breakers the estimator re-estimated the
    chi-square of the substation estimate after the flip.

Pure functions of solves that were made anyway; they read no truth and add no
solve.  The provider emits them only when built with ``complete_evidence`` and
the prompt builder admits them only for an observation that carries the
contract, so every recorded output and prompt is unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.actions import COMPLETE_EVIDENCE_CONTRACT
from psse_env.providers.wls_tables import wls_tables, zero_injection_rows

BUS_ID = 0
F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS = 0, 1, 2, 3, 8, 9, 10
BUS_CHANNELS = ("Vm", "Pinj", "Qinj")
FLOW_CHANNELS = ("Pf", "Qf", "Pt", "Qt")
RESIDUAL_DECIMALS = 2
VALUE_DECIMALS = 4
PARAMETER_DECIMALS = 5
#: The keys under which the complete tables travel, in the provider's outputs and in the observation.
SOLVE_TABLE_KEYS = ("bus_table", "branch_table")
CONTEXT_TABLE_KEYS = ("meter_table", "line_table", "breaker_table")
TABLE_KEYS = SOLVE_TABLE_KEYS + CONTEXT_TABLE_KEYS


def _rounded(value: Any, decimals: int) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(round(number, decimals)) if math.isfinite(number) else None


def signed_residuals(payload: Mapping[str, Any]) -> list[float]:
    """The solve's normalized residuals with their sign: measured minus estimated, in standard deviations.

    The solver reports the magnitude (``r``) and the raw residual ``z - h(x)`` (``raw_residual``) side by
    side; the summaries read the magnitude alone.  Where the raw residual is missing the magnitude stands.
    """
    magnitude = [float(value) for value in payload.get("r") or []]
    raw = payload.get("raw_residual")
    raw = list(raw) if raw is not None else []
    if len(raw) != len(magnitude):
        return magnitude
    return [math.copysign(size, float(error)) if float(error) != 0.0 else size for size, error in zip(magnitude, raw)]


def solve_tables(signed_residuals: Sequence[float], lambdas: Sequence[float], ppc: Mapping[str, Any], *,
                 vm_estimate: Sequence[float] | None = None, angle_estimate_rad: Sequence[float] | None = None) -> dict[str, Any]:
    """Every bus and every line of one solve, in the case's order (the columns of ``wls_tables``).

    With the solve's state estimate, each bus row also carries it: the
    estimated voltage magnitude in per unit (``v``) and angle in degrees
    (``angle``).  An estimate that does not hold one value per bus of the
    case (a bus the solver left out) is not attached.
    """
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    nb, nl = int(bus.shape[0]), int(branch.shape[0])
    tables = wls_tables(signed_residuals, lambdas, ppc, residual_threshold=0.0, max_buses=nb, max_branches=nl)
    place = {int(bus_id): row for row, bus_id in enumerate(bus[:, BUS_ID].astype(int))}
    bus_table = sorted(tables["bus_table"], key=lambda item: place[int(item["bus"])])
    branch_table = sorted(tables["branch_table"], key=lambda item: int(item["line"]))
    if len(bus_table) != nb or len(branch_table) != nl:
        raise ValueError("the solve holds a channel that is not a finite number")
    magnitude = list(vm_estimate) if vm_estimate is not None else []
    angle = list(angle_estimate_rad) if angle_estimate_rad is not None else []
    if len(magnitude) == nb and len(angle) == nb:
        bus_table = [
            {"bus": item["bus"], "type": item["type"], "v": _rounded(magnitude[row], VALUE_DECIMALS),
             "angle": _rounded(math.degrees(float(angle[row])), RESIDUAL_DECIMALS),
             **{key: value for key, value in item.items() if key not in ("bus", "type")}}
            for row, item in enumerate(bus_table)
        ]
    return {"bus_table": bus_table, "branch_table": branch_table}


def meter_table(measurements: Sequence[float], signed_residuals: Sequence[float], ppc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every measurement of the operator's vector, in index order.

    ``at`` is the bus number for Vm, Pinj and Qinj and the line number for the
    flows (Pf, Qf at the from end; Pt, Qt at the to end).  The residual of an
    injection at a zero-injection bus is null, as in the bus table: that row is
    structural, not a meter.
    """
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    nb, nl = int(bus.shape[0]), int(branch.shape[0])
    values = np.asarray(measurements, dtype=float)
    residual = np.asarray(signed_residuals, dtype=float)
    if values.size != 3 * nb + 4 * nl or residual.size != values.size:
        raise ValueError(f"expected {3 * nb + 4 * nl} measurements and residuals, got {values.size} and {residual.size}")
    zero = set(zero_injection_rows(ppc))
    bus_ids = bus[:, BUS_ID].astype(int)
    rows: list[dict[str, Any]] = []
    for block, channel in enumerate(BUS_CHANNELS):
        for row in range(nb):
            index = block * nb + row
            structural = block > 0 and row in zero
            rows.append({"index": index, "channel": channel, "at": int(bus_ids[row]),
                         "value": _rounded(values[index], VALUE_DECIMALS),
                         "residual": None if structural else _rounded(residual[index], RESIDUAL_DECIMALS)})
    for block, channel in enumerate(FLOW_CHANNELS):
        for row in range(nl):
            index = 3 * nb + block * nl + row
            rows.append({"index": index, "channel": channel, "at": row + 1,
                         "value": _rounded(values[index], VALUE_DECIMALS),
                         "residual": _rounded(residual[index], RESIDUAL_DECIMALS)})
    return rows


def line_table(lambdas: Sequence[float], ppc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every line: its buses, the model's series R and X, and their signed normalized multipliers."""
    branch = np.asarray(ppc["branch"], dtype=float)
    nl = int(branch.shape[0])
    lam = np.asarray(lambdas, dtype=float)
    if lam.size != 2 * nl:
        raise ValueError(f"expected {2 * nl} multipliers, got {lam.size}")
    tap = branch[:, TAP] if branch.shape[1] > TAP else np.zeros(nl)
    shift = branch[:, SHIFT] if branch.shape[1] > SHIFT else np.zeros(nl)
    status = branch[:, BR_STATUS] if branch.shape[1] > BR_STATUS else np.ones(nl)
    rows: list[dict[str, Any]] = []
    for row in range(nl):
        item: dict[str, Any] = {
            "line": row + 1, "from": int(branch[row, F_BUS]), "to": int(branch[row, T_BUS]),
            "r": _rounded(branch[row, BR_R], PARAMETER_DECIMALS), "x": _rounded(branch[row, BR_X], PARAMETER_DECIMALS),
            "lr": _rounded(lam[2 * row], RESIDUAL_DECIMALS), "lx": _rounded(lam[2 * row + 1], RESIDUAL_DECIMALS),
        }
        if tap[row] != 0.0 or shift[row] != 0.0:
            item["xfmr"] = True
        if status[row] <= 0:
            item["out"] = True
        rows.append(item)
    return rows


def breaker_table(ranking: Sequence[Mapping[str, Any]], effects: Mapping[str, Mapping[str, Any]],
                  flips: Mapping[str, Mapping[str, Any]], reported_chi_square: float) -> list[dict[str, Any]]:
    """Every breaker of the node/breaker estimate, by name.

    ``ranking`` is the estimator's list (name, reported status, multipliers,
    score); ``effects`` what flipping each breaker does to the operator model;
    ``flips`` the re-estimations the context made, by breaker.  A breaker the
    context did not re-estimate has no ``flip_*`` cell; one whose re-estimation
    did not converge has ``flip_converged`` false and no chi-square.
    """
    rows: list[dict[str, Any]] = []
    for item in sorted(ranking, key=lambda entry: str(entry["cb_name"])):
        name = str(item["cb_name"])
        multipliers = item.get("multipliers") if isinstance(item.get("multipliers"), Mapping) else {}
        row: dict[str, Any] = {
            "cb": name, "status": str(item.get("reported_status")),
            "score": _rounded(item.get("score"), RESIDUAL_DECIMALS),
            "theta": _rounded(multipliers.get("theta"), RESIDUAL_DECIMALS),
            "vm": _rounded(multipliers.get("vm"), RESIDUAL_DECIMALS),
        }
        effect = effects.get(name) or {}
        row["flip_effect"] = str(effect.get("effect") or "unknown")
        if effect.get("branch_row0") is not None:
            row["line"] = int(effect["branch_row0"]) + 1
        buses = effect.get("affected_planning_buses")
        if isinstance(buses, (list, tuple)) and buses:
            row["bus"] = "+".join(str(int(b)) for b in buses)
        flip = flips.get(name)
        if isinstance(flip, Mapping):
            chi_square = _rounded(flip.get("chi_square"), 1) if flip.get("success") else None
            row["flip_converged"] = chi_square is not None
            if chi_square is not None:
                row["flip_chi_square"] = chi_square
                if reported_chi_square > 0.0:
                    row["flip_progress"] = _rounded((float(reported_chi_square) - chi_square) / float(reported_chi_square), 3)
        rows.append(row)
    return rows


__all__ = ["COMPLETE_EVIDENCE_CONTRACT", "CONTEXT_TABLE_KEYS", "SOLVE_TABLE_KEYS", "TABLE_KEYS", "breaker_table",
           "line_table", "meter_table", "signed_residuals", "solve_tables"]
