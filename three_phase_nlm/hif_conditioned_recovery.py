"""Paired physical replay for an already estimated IEEE-14 HIF.

This helper predicts a measurement-space effect. It neither repairs a physical
fault nor certifies that subtracting the effect leaves only measurement noise.
Only observable fitted parameters and an explicitly supplied operating point
are inputs; no clean measurements or injected-error labels are consulted.

OpenDSS uses process-global circuit state, so calls in one process must be
serial. The paired solves deliberately use the same resolved model directory,
operating point, and external channels, including the hidden split-line map.
"""

from __future__ import annotations

from IEEE_14_OpenDSS.measurement_convention import resolve_shunt_convention

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from IEEE_14_OpenDSS.constants import BRANCH_ORDER, BUS_ORDER
from .hif_operating_point import canonicalize_ieee14_operating_point
from .hif_parameter_estimator import (
    _resolve_model_dir,
    _simulate_base,
    simulate_hif_candidate,
)
from .ieee14_adapter import branch_info_for_row0


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _integer(value: Any, field: str) -> int:
    number = _finite_number(value, field)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(number)


def _current_resistance(
    fit: Mapping[str, Any], estimated: Mapping[str, Any], scan_index: int
) -> tuple[str, float]:
    search = fit.get("search")
    search = search if isinstance(search, Mapping) else {}
    per_scan = estimated.get("per_scan_r_hif_pu")
    mode = estimated.get("resistance_model") or search.get("resistance_mode")
    if mode is None:
        mode = "scan_specific_smooth" if per_scan is not None else "single_snapshot"
    if mode not in {"shared", "single_snapshot", "scan_specific_smooth"}:
        raise ValueError(f"unsupported resistance model: {mode!r}")
    if mode == "scan_specific_smooth":
        if not isinstance(per_scan, (list, tuple)) or not per_scan:
            raise ValueError("scan-specific resistance requires per_scan_r_hif_pu")
        values: dict[int, float] = {}
        for item in per_scan:
            if not isinstance(item, Mapping):
                raise ValueError("per_scan_r_hif_pu entries must be objects")
            index = _integer(item.get("scan_index"), "per-scan scan_index")
            if index in values:
                raise ValueError(f"duplicate resistance estimate for scan {index}")
            value = _finite_number(item.get("r_hif_pu"), "per-scan r_hif_pu")
            if value <= 0:
                raise ValueError("per-scan r_hif_pu must be positive")
            values[index] = value
        if scan_index not in values:
            raise ValueError(f"no fitted resistance for current scan {scan_index}")
        resistance = values[scan_index]
    else:
        if per_scan is not None:
            raise ValueError("per-scan resistance conflicts with the declared resistance model")
        resistance = _finite_number(estimated.get("r_hif_pu"), "r_hif_pu")
        if resistance <= 0:
            raise ValueError("r_hif_pu must be positive")
    return str(mode), resistance


def _measurement_vector(payload: Mapping[str, Any], name: str) -> np.ndarray:
    values = np.asarray(payload.get("z"), dtype=float)
    expected = 3 * len(BUS_ORDER) + 4 * len(BRANCH_ORDER)
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} replay must return {expected} finite external measurements")
    return values


def replay_hif_measurement_effect(
    fit_payload: Mapping[str, Any],
    *,
    op_point: Mapping[str, Any],
    scan_index: int,
    snapshot_id: str,
    time_tag: str | None = None,
    pristine_model_dir: str | None = None,
    shunt_convention: str | None = None,
) -> dict[str, Any]:
    """Replay a fitted HIF and its no-HIF counterpart for one named snapshot.

    ``fit_payload`` is the estimator result, not a scenario or hidden label.
    ``op_point`` is the current scan's observable operating-point context. The
    caller is responsible for associating that fit/context with ``snapshot_id``
    and for accepting the diagnostic before using this prediction. A successful
    return means both forward solves completed, not that the fit is identifiable
    or that a conditioned residual has a calibrated statistical distribution.

    For time-varying resistance, an exact ``scan_index`` match is required;
    the median resistance is never substituted. ``time_tag`` is preserved as an
    opaque acquisition-time identifier, without inventing a time interpolation.
    """
    if not isinstance(fit_payload, Mapping) or fit_payload.get("success") is not True:
        raise ValueError("a successful observable HIF estimator payload is required")
    if not isinstance(op_point, Mapping):
        raise ValueError("the current scan operating point must be explicitly supplied")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id must be a nonempty string")
    if time_tag is not None and (not isinstance(time_tag, str) or not time_tag.strip()):
        raise ValueError("time_tag must be a nonempty string when supplied")
    current_scan = _integer(scan_index, "scan_index")
    estimated = fit_payload.get("estimated")
    if not isinstance(estimated, Mapping):
        raise ValueError("HIF estimator payload has no estimated parameters")
    branch = _integer(fit_payload.get("candidate_branch_row0"), "candidate_branch_row0")
    info = branch_info_for_row0(branch)
    if not str(info["dss_element"]).lower().startswith("line."):
        raise ValueError("HIF replay supports IEEE-14 Line.* branches only")
    phase = str(estimated.get("phase") or "").upper()
    if phase not in {"A", "B", "C"}:
        raise ValueError("estimated phase must be A, B, or C")
    alpha = _finite_number(estimated.get("alpha_from_from_bus"), "alpha_from_from_bus")
    if not 0 < alpha < 1:
        raise ValueError("alpha_from_from_bus must lie strictly between zero and one")
    mode, resistance = _current_resistance(fit_payload, estimated, current_scan)
    operating_point = canonicalize_ieee14_operating_point(op_point)
    # The estimator resolver normally falls back to its default model. Do not
    # silently accept a misspelled explicitly requested replay model directory.
    if pristine_model_dir is not None:
        requested = Path(pristine_model_dir).expanduser()
        if not requested.is_absolute():
            requested = Path(__file__).resolve().parents[1] / requested
        if not (requested / "Run_IEEE14Bus.dss").is_file():
            raise ValueError("pristine_model_dir must contain Run_IEEE14Bus.dss")
        pristine_model_dir = str(requested.resolve())
    model_dir = _resolve_model_dir(pristine_model_dir, "case14").resolve()
    convention = resolve_shunt_convention(shunt_convention or (fit_payload.get("search") or {}).get("shunt_convention"), fit_payload)
    present = simulate_hif_candidate(
        candidate_branch_row0=branch,
        alpha=alpha,
        phase=phase,
        r_hif_pu=resistance,
        op_point=deepcopy(operating_point),
        pristine_model_dir=str(model_dir),
        shunt_convention=convention,
    )
    absent = _simulate_base(model_dir, op_point=deepcopy(operating_point), shunt_convention=convention)
    present_z = _measurement_vector(present, "HIF-present")
    absent_z = _measurement_vector(absent, "HIF-absent")
    channel_ids = [f"Vm:{bus}:phase_A" for bus in BUS_ORDER]
    channel_ids += [f"{kind}:{bus}:three_phase_total" for kind in ("Pinj", "Qinj") for bus in BUS_ORDER]
    channel_ids += [f"{kind}:{element}:external_terminal" for kind in ("Pf", "Qf", "Pt", "Qt") for element in BRANCH_ORDER]
    return {
        "success": True,
        "method": "paired_opendss_hif_measurement_replay",
        "binding": {"snapshot_id": snapshot_id, "scan_index": current_scan, "time_tag": time_tag},
        "parameters_used": {
            "candidate_branch_row0": branch,
            "dss_element": info["dss_element"],
            "phase": phase,
            "alpha_from_from_bus": alpha,
            "r_hif_pu": resistance,
            "resistance_model": mode,
            "shunt_convention": convention,
        },
        "op_point": operating_point,
        "predicted_hif_measurements": present_z.tolist(),
        "predicted_base_measurements": absent_z.tolist(),
        "measurement_effect": (present_z - absent_z).tolist(),
        "channel_ids": channel_ids,
        "measurement_semantics": {
            "system": "IEEE14",
            "model_dir": str(model_dir),
            "layout": ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"],
            "voltage": "phase_A_line_to_neutral_magnitude_pu",
            "power": "three_phase_total_pu_on_100MVA",
            "fault_bus": "hidden; external original branch terminals preserved",
            "scope": "one_HIF_on_the_supplied_OpenDSS_model_at_the_supplied_operating_point",
        },
        "diagnostic_fit_evidence": {
            key: deepcopy(fit_payload[key])
            for key in ("method", "fit", "uncertainty", "observability", "parameter_identifiable", "selected_scan_indices", "window_id")
            if key in fit_payload
        },
        "uncertainty_statement": {
            "effect_covariance_available": False,
            "replay_is_conditional_on_estimated_parameters_and_operating_point": True,
            "same_channel_measurement_error_separability": "not_established_by_replay_alone",
            "statistical_calibration": "not_established_by_replay_alone",
        },
        "physical_fault_still_present": True,
    }


__all__ = ["replay_hif_measurement_effect"]
