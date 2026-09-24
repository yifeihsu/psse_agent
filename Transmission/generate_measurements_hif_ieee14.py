#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate IEEE-14 high-impedance-fault measurement records.

The operator-facing vector remains the standard 122-entry IEEE-14 layout:
    z = [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]

The hidden HIF bus is only present inside the copied OpenDSS scenario model.
It is not exposed as a bus in z_obs or z_true.

Resistance units (2026-09-19 reconfiguration)
---------------------------------------------
``--resistance-units ohm`` (default) specifies the fault resistance in
PHYSICAL ohms at the faulted line's actual voltage (buses 1-5: 69 kV,
6-7 and 9-14: 13.8 kV, bus 8: 18 kV; 100 MVA base).  Per line

    r_hif_pu      = resistance_ohm / (kV_local**2 / 100)      (local base)
    r_hif_model_ohm = r_hif_pu * 0.01                         (kV=1 DSS model)

Only ``r_hif_model_ohm`` reaches the injector and the legacy NLM bridge; the
DSS files are never edited (pu invariance).  Labels carry
``resistance_units="ohm_local_base"`` and ``r_hif_ohm`` is then the PHYSICAL
value; ``kv_ln`` stays the model's 0.577 kV (``kv_ln_semantics`` says so).

``--resistance-units pu`` reproduces the legacy corpora: ``r_hif_pu`` is
system pu with no voltage meaning, ``r_hif_ohm = r_hif_pu * 0.01`` (model
ohms), labels gain only ``resistance_units="pu_legacy_normalized_model"``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from dataclasses import dataclass

from pypower.api import case14, ppoption, runopf  # type: ignore
from pypower.idx_brch import BR_STATUS, F_BUS, TAP, T_BUS  # type: ignore
from pypower.idx_bus import PD, QD  # type: ignore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from IEEE_14_OpenDSS.constants import IEEE14_LOAD_BASE_KW  # type: ignore
from IEEE_14_OpenDSS.constants import IEEE14_OPERATING_POINT_KEYS  # type: ignore
from IEEE_14_OpenDSS.export_measurement_series import (  # type: ignore
    BRANCH_ORDER,
    BUS_ORDER,
)
from IEEE_14_OpenDSS.measurement_convention import (  # type: ignore
    MEASUREMENT_CONVENTION_KEY,
    SHUNT_CONVENTION_LEGACY,
    SHUNT_CONVENTION_YBUS,
    SHUNT_CONVENTIONS,
    measurement_convention_payload,
    validate_shunt_convention,
)
from three_phase_nlm import (  # type: ignore
    copy_ieee14_model,
    hif_ohms_from_pu,
    inject_midspan_hif_ieee14,
    run_ieee14_hif_nlm,
    simulate_hif_candidate,
    write_balanced_ieee14_load_override,
)
from three_phase_nlm.ieee14_adapter import ELIGIBLE_HIF_BRANCHES, branch_info_for_row0
from three_phase_nlm.hif_operating_point import (  # type: ignore
    DISPATCH_MODE_CASE14,
    DISPATCH_MODE_OPF,
    DISPATCH_MODES,
    OPF_SOLVER,
    OPFDispatchError,
    OPFOperatingPoint,
    canonicalize_ieee14_operating_point,
    case14_dispatch_receipt,
    ieee14_opf_operating_point,
    opf_dispatch_receipt,
)
from three_phase_nlm.hif_parameter_estimator import _resolve_model_dir, _simulate_base  # type: ignore
from three_phase_nlm.hif_units import (  # type: ignore
    EXCLUDED_CROSS_VOLTAGE_BRANCHES,
    MAIN_HIF_BAND_OHM,
    RESISTANCE_UNITS_OHM_LOCAL_BASE,
    RESISTANCE_UNITS_PU_LEGACY,
    S_BASE_MVA,
    VOLTAGE_BASE_PROFILE_ID,
    VOLTAGE_STRATA,
    eligible_rows_for_stratum,
    hif_resistance_record,
    local_pu_from_ohm,
)
from three_phase_model.voltage_bases import (  # type: ignore
    IEEE14_NOMINAL_KV,
    ieee14_voltage_base_profile,
)
from three_phase_nlm.branch_current_analysis import (  # type: ignore
    BRANCH_CURRENT_CHANNEL,
    BRANCH_CURRENT_SIGMA_KEY,
    DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    add_branch_current_noise,
)
from three_phase_nlm.measurement_noise import (
    DEFAULT_THREE_PHASE_SIGMA_PU,
    add_scada_noise,
    add_voltage_phasor_noise,
    generated_noise_contract,
    positive_sigma,
    scada_noise_sigma,
    scaled_sensor_sigma,
)
from Transmission.generate_measurements import (  # type: ignore
    MEASUREMENT_ORDER,
    compute_measurements_pu,
    make_index_map,
)

RESISTANCE_UNITS_CHOICES = ("ohm", "pu")
VOLTAGE_STRATUM_CHOICES = tuple(VOLTAGE_STRATA)
DEFAULT_R_HIF_OHM_MIN, DEFAULT_R_HIF_OHM_MAX = (float(MAIN_HIF_BAND_OHM[0]), float(MAIN_HIF_BAND_OHM[1]))
DEFAULT_R_HIF_PU_MIN, DEFAULT_R_HIF_PU_MAX = (20.0, 200.0)
PHASES = ("A", "B", "C")

SAMPLING_UNIFORM_OHM = "uniform_ohm"
SAMPLING_BANDS_OHM = "weighted_bands_ohm"
SAMPLING_SWEEP_OHM = "discrete_sweep_ohm"
SAMPLING_UNIFORM_PU_LEGACY = "uniform_pu_legacy"

TELEMETRY_BASES_PHYSICAL = "physical_local_bases"
TELEMETRY_BASES_NORMALIZED = "normalized_model_bases"

KV_LN_SEMANTICS_MODEL = "normalized_model_load_kv_ln"
MODEL_OHM_CONVENTION = (
    "R_model_ohm = r_hif_pu * 0.01 ohm (kV=1 normalized DSS model); "
    "r_hif_pu = resistance_ohm / (local_kv_ll**2 / 100 MVA)"
)

#: Expected legacy-NLM detectability per physical resistance class (brief #6).
EXPECTED_DETECTABILITY_BY_CLASS = {
    "low_resistance_fault": "strong",
    "moderately_resistive": "representative",
    "moderately_high_resistance": "representative",
    "representative_hif": "representative",
    "weak_hif": "weak",
    "extreme_weak_hif": "extreme",
    "near_open_circuit": "extreme",
}

_SQRT3 = math.sqrt(3.0)


def _scale_pypower_loads(ppc: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    ppc2 = deepcopy(ppc)
    ppc2["bus"][:, PD] *= float(alpha)
    ppc2["bus"][:, QD] *= float(alpha)
    return ppc2


def _solve_pypower(ppc: Dict[str, Any]) -> Dict[str, Any] | None:
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    result = runopf(ppc, ppopt)
    return result if result.get("success") else None


def _branch_info_case14() -> list[dict[str, Any]]:
    ppc = case14()
    br = ppc["branch"]
    out = []
    for i in range(br.shape[0]):
        out.append(
            {
                "i": int(i),
                "from_bus": int(br[i, F_BUS]),
                "to_bus": int(br[i, T_BUS]),
                "is_line": bool(float(br[i, TAP]) == 0.0 and float(br[i, BR_STATUS]) > 0.0),
            }
        )
    return out


def _measurement_sigma(length: int = 122) -> np.ndarray:
    if int(length) != 122:
        raise ValueError("IEEE-14 SCADA noise requires 122 channels")
    return scada_noise_sigma()


def _maybe_add_noise(z_obs: list[float], rng: np.random.Generator, noise_scale: float) -> list[float]:
    return add_scada_noise(z_obs, rng, scada_noise_sigma(noise_scale))


# ----------------------------------------------------------------- physical bases
def _bus_number(bus: Any) -> int:
    text = str(bus).strip().lower()
    if text.startswith("b"):
        text = text[1:]
    return int(text.split(".")[0])


def _physical_kvbase_ln(bus: Any) -> float:
    """Physical line-to-neutral base kV of an IEEE-14 bus (kV_LL / sqrt(3))."""
    return float(IEEE14_NOMINAL_KV[_bus_number(bus)]) / _SQRT3


def _physical_ibase_a(bus: Any) -> float:
    """Per-phase current base (S_base/3) / V_LN,base on the physical bus base."""
    return (float(S_BASE_MVA) * 1e6 / 3.0) / (_physical_kvbase_ln(bus) * 1e3)


def _rewrite_voltage_bases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replace ``kvbase_ln`` by the physical local base; pu values untouched."""
    return [{**dict(row), "kvbase_ln": _physical_kvbase_ln(row["bus"])} for row in rows]


def _rewrite_current_bases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replace ``ibase_from_a``/``ibase_to_a`` by the physical terminal bases; pu untouched."""
    return [
        {
            **dict(row),
            "ibase_from_a": _physical_ibase_a(row["from_bus"]),
            "ibase_to_a": _physical_ibase_a(row["to_bus"]),
        }
        for row in rows
    ]


def _rewrite_scan_telemetry_bases(scan: dict[str, Any]) -> dict[str, Any]:
    for key in ("three_phase_voltages", "three_phase_voltages_clean"):
        if isinstance(scan.get(key), list):
            scan[key] = _rewrite_voltage_bases(scan[key])
    for key in (BRANCH_CURRENT_CHANNEL, f"{BRANCH_CURRENT_CHANNEL}_clean"):
        if isinstance(scan.get(key), list):
            scan[key] = _rewrite_current_bases(scan[key])
    return scan


def expected_detectability(resistance_class: str) -> str:
    try:
        return EXPECTED_DETECTABILITY_BY_CLASS[str(resistance_class)]
    except KeyError as exc:
        raise ValueError(f"Unknown HIF resistance class {resistance_class!r}") from exc


# -------------------------------------------------------------- resistance sampling
def _parse_float_list(text: str | None, *, field: str) -> list[float] | None:
    if text is None:
        return None
    items = [part.strip() for part in str(text).split(",") if part.strip()]
    if not items:
        raise ValueError(f"{field} must list at least one value")
    values = [float(part) for part in items]
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError(f"{field} values must be finite and positive")
    return values


def _parse_bands(text: str | None, *, field: str = "--r-hif-ohm-bands") -> list[list[float]] | None:
    if text is None:
        return None
    bands: list[list[float]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"{field} entries must look like lo:hi, got {part!r}")
        lo_text, hi_text = part.split(":", 1)
        lo, hi = float(lo_text), float(hi_text)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi <= lo:
            raise ValueError(f"{field} entries require 0 < lo < hi, got {part!r}")
        bands.append([lo, hi])
    if not bands:
        raise ValueError(f"{field} must list at least one band")
    return bands


def _coerce_bands(bands: Any) -> list[list[float]] | None:
    if bands is None:
        return None
    if isinstance(bands, str):
        return _parse_bands(bands)
    out: list[list[float]] = []
    for band in bands:
        lo, hi = float(band[0]), float(band[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi <= lo:
            raise ValueError(f"r_hif_ohm_bands entries require 0 < lo < hi, got {band!r}")
        out.append([lo, hi])
    if not out:
        raise ValueError("r_hif_ohm_bands must list at least one band")
    return out


def _coerce_float_list(values: Any, *, field: str) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return _parse_float_list(values, field=field)
    out = [float(v) for v in values]
    if not out or any(not math.isfinite(v) or v <= 0.0 for v in out):
        raise ValueError(f"{field} values must be finite and positive")
    return out


def resolve_resistance_sampling(
    *,
    resistance_units: str,
    r_hif_pu_min: float | None,
    r_hif_pu_max: float | None,
    r_hif_ohm_min: float | None,
    r_hif_ohm_max: float | None,
    r_hif_ohm_bands: Any,
    r_hif_ohm_band_weights: Any,
    r_hif_ohm_sweep: Any,
    voltage_stratum: str,
) -> dict[str, Any]:
    """Validate the sampling request and return a self-describing plan.

    Ohm mode accepts exactly one of: uniform ``[r_hif_ohm_min, r_hif_ohm_max]``
    (defaults 100/1000), weighted bands, or a discrete sweep.  Legacy pu mode
    accepts only the pu bounds (defaults 20/200).
    """
    units = str(resistance_units).strip().lower()
    if units not in RESISTANCE_UNITS_CHOICES:
        raise ValueError(f"resistance_units must be one of {RESISTANCE_UNITS_CHOICES}, got {resistance_units!r}")
    pu_given = r_hif_pu_min is not None or r_hif_pu_max is not None
    ohm_uniform_given = r_hif_ohm_min is not None or r_hif_ohm_max is not None
    bands = _coerce_bands(r_hif_ohm_bands)
    sweep = _coerce_float_list(r_hif_ohm_sweep, field="r_hif_ohm_sweep")
    weights = _coerce_float_list(r_hif_ohm_band_weights, field="r_hif_ohm_band_weights")

    if units == "pu":
        if ohm_uniform_given or bands is not None or sweep is not None or weights is not None:
            raise ValueError("Ohm sampling options require resistance_units='ohm'")
        if voltage_stratum not in VOLTAGE_STRATA:
            raise ValueError(f"voltage_stratum must be one of {VOLTAGE_STRATUM_CHOICES}")
        lo = DEFAULT_R_HIF_PU_MIN if r_hif_pu_min is None else float(r_hif_pu_min)
        hi = DEFAULT_R_HIF_PU_MAX if r_hif_pu_max is None else float(r_hif_pu_max)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi <= lo:
            raise ValueError("Require 0 < r_hif_pu_min < r_hif_pu_max")
        return {
            "mode": SAMPLING_UNIFORM_PU_LEGACY,
            "resistance_units": RESISTANCE_UNITS_PU_LEGACY,
            "r_hif_pu_min": lo,
            "r_hif_pu_max": hi,
            "r_hif_pu_range": [lo, hi],
            "r_hif_ohm_range": None,
            "bands_ohm": None,
            "weights": None,
            "sweep_values_ohm": None,
        }

    if pu_given:
        raise ValueError("--r-hif-pu-min/--r-hif-pu-max are only accepted with --resistance-units pu")
    if voltage_stratum not in VOLTAGE_STRATA:
        raise ValueError(f"voltage_stratum must be one of {VOLTAGE_STRATUM_CHOICES}")
    modes_requested = int(ohm_uniform_given) + int(bands is not None) + int(sweep is not None)
    if modes_requested > 1:
        raise ValueError("Choose one ohm sampling mode: uniform min/max, --r-hif-ohm-bands, or --r-hif-ohm-sweep")
    if weights is not None and bands is None:
        raise ValueError("--r-hif-ohm-band-weights requires --r-hif-ohm-bands")

    stratum_kv = VOLTAGE_STRATA[voltage_stratum]

    def _pu_envelope(lo_ohm: float, hi_ohm: float) -> list[float] | None:
        if stratum_kv is None:
            return None
        return [local_pu_from_ohm(lo_ohm, stratum_kv), local_pu_from_ohm(hi_ohm, stratum_kv)]

    if sweep is not None:
        values = sorted(set(sweep))
        return {
            "mode": SAMPLING_SWEEP_OHM,
            "resistance_units": RESISTANCE_UNITS_OHM_LOCAL_BASE,
            "r_hif_pu_min": None,
            "r_hif_pu_max": None,
            "r_hif_pu_range": _pu_envelope(values[0], values[-1]),
            "r_hif_ohm_range": [values[0], values[-1]],
            "bands_ohm": None,
            "weights": None,
            "sweep_values_ohm": values,
            "cell_schedule": "balanced_line_x_resistance_x_phase",
        }
    if bands is not None:
        if weights is None:
            weights = [1.0] * len(bands)
        if len(weights) != len(bands):
            raise ValueError("--r-hif-ohm-band-weights must have one weight per band")
        total = float(sum(weights))
        return {
            "mode": SAMPLING_BANDS_OHM,
            "resistance_units": RESISTANCE_UNITS_OHM_LOCAL_BASE,
            "r_hif_pu_min": None,
            "r_hif_pu_max": None,
            "r_hif_pu_range": _pu_envelope(min(b[0] for b in bands), max(b[1] for b in bands)),
            "r_hif_ohm_range": [min(b[0] for b in bands), max(b[1] for b in bands)],
            "bands_ohm": bands,
            "weights": [float(w) for w in weights],
            "band_probabilities": [float(w) / total for w in weights],
            "sweep_values_ohm": None,
        }
    lo = DEFAULT_R_HIF_OHM_MIN if r_hif_ohm_min is None else float(r_hif_ohm_min)
    hi = DEFAULT_R_HIF_OHM_MAX if r_hif_ohm_max is None else float(r_hif_ohm_max)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        raise ValueError("Require 0 < r_hif_ohm_min < r_hif_ohm_max")
    return {
        "mode": SAMPLING_UNIFORM_OHM,
        "resistance_units": RESISTANCE_UNITS_OHM_LOCAL_BASE,
        "r_hif_pu_min": None,
        "r_hif_pu_max": None,
        "r_hif_pu_range": _pu_envelope(lo, hi),
        "r_hif_ohm_range": [lo, hi],
        "bands_ohm": None,
        "weights": None,
        "sweep_values_ohm": None,
    }


def resolve_shunt_convention_default(resistance_units: str, shunt_convention: str | None) -> str:
    """ybus in ohm mode, legacy_injection in pu mode, unless given explicitly."""
    if shunt_convention is not None:
        return validate_shunt_convention(shunt_convention)
    return SHUNT_CONVENTION_YBUS if str(resistance_units).lower() == "ohm" else SHUNT_CONVENTION_LEGACY


def build_sweep_schedule(
    rng: np.random.Generator,
    *,
    eligible_rows: Sequence[int],
    sweep_values_ohm: Sequence[float],
    n_hif: int,
    phases: Sequence[str] = PHASES,
) -> tuple[list[tuple[int, float, str]], bool]:
    """Balanced schedule over (line x resistance x phase) cells.

    Each full cycle is one permutation of every cell, so every cell is covered
    once per ``len(cells)`` samples.  When ``n_hif`` is not a multiple of the
    cell count a warning is printed and the schedule is truncated (the last
    cycle is partial).  Returns ``(schedule, truncated)``.
    """
    cells = [(int(row0), float(r), str(ph)) for row0, r, ph in itertools.product(eligible_rows, sweep_values_ohm, phases)]
    if not cells:
        raise ValueError("Sweep schedule needs at least one eligible line and one resistance")
    truncated = int(n_hif) % len(cells) != 0
    if truncated:
        print(
            f"WARNING: --n-hif={int(n_hif)} is not a multiple of the {len(cells)} "
            f"(line x resistance x phase) sweep cells; the balanced schedule is truncated.",
            file=sys.stderr,
        )
    schedule: list[tuple[int, float, str]] = []
    while len(schedule) < int(n_hif):
        for index in rng.permutation(len(cells)):
            schedule.append(cells[int(index)])
    return schedule[: int(n_hif)], truncated


@dataclass(frozen=True)
class ScanOperatingPoint:
    """One scan's canonical operating point with the provenance of its dispatch.

    ``op_point`` is the replayable canonical schema (the estimators and the
    balanced reference read only this); ``dispatch`` is the JSON annotation
    stored beside it; ``opf`` keeps the pypower solution in opf mode.
    """

    op_point: dict[str, Any]
    dispatch: dict[str, Any]
    opf: OPFOperatingPoint | None = None


def _validated_dispatch_mode(dispatch_mode: str) -> str:
    normalized = str(dispatch_mode).strip().lower()
    if normalized not in DISPATCH_MODES:
        raise ValueError(f"dispatch_mode must be one of {DISPATCH_MODES}, got {dispatch_mode!r}")
    return normalized


def _dispatched_operating_point(
    *,
    dispatch_mode: str,
    load_scale: float,
    bus_load_scales: Mapping[str, float] | None,
    case14_point: Mapping[str, Any],
) -> ScanOperatingPoint:
    """Attach the dispatch law: the case14 model values, or the AC-OPF at the same loads.

    In opf mode the OPF is solved with exactly the per-bus loads the OpenDSS
    scan will carry (``load_scale * bus_load_scales``); a non-converged OPF
    raises :class:`OPFDispatchError` so the caller skips the window instead of
    quietly keeping the case14 dispatch.
    """
    if dispatch_mode == DISPATCH_MODE_OPF:
        opf = ieee14_opf_operating_point(float(load_scale), bus_load_scales)
        return ScanOperatingPoint(op_point=opf.op_point, dispatch=opf.receipt, opf=opf)
    return ScanOperatingPoint(op_point=canonicalize_ieee14_operating_point(case14_point),
                              dispatch=case14_dispatch_receipt())


def _sample_diverse_operating_point(
    rng: np.random.Generator,
    *,
    event_load_scale: float,
    load_log_std: float,
    dispatch_fraction: float,
    voltage_std: float,
    dispatch_mode: str = DISPATCH_MODE_CASE14,
) -> ScanOperatingPoint:
    """Draw one diverse scan operating point.

    The random draws (spatial load profile, then the case14-mode dispatch and
    setpoint perturbations) are consumed in the same order in every dispatch
    mode, so the load profiles of a seed repeat across modes; in opf mode the
    dispatch and setpoint draws are discarded and the AC-OPF at the drawn
    loads supplies the dispatch, the PV setpoints and the source voltage.
    """
    buses = list(IEEE14_LOAD_BASE_KW)
    raw = np.exp(rng.normal(0.0, float(load_log_std), size=len(buses)))
    weights = np.asarray([IEEE14_LOAD_BASE_KW[bus] for bus in buses], dtype=float)
    raw /= float(np.average(raw, weights=weights))
    bus_scales = {
        bus: float(
            np.clip(
                factor,
                0.65 / float(event_load_scale),
                1.45 / float(event_load_scale),
            )
        )
        for bus, factor in zip(buses, raw)
    }
    dispatch = {
        "b2": float(40000.0 * rng.uniform(1.0 - dispatch_fraction, 1.0 + dispatch_fraction))
    }
    voltage_setpoints = {
        bus: float(np.clip(base + rng.normal(0.0, voltage_std), 0.98, 1.10))
        for bus, base in canonicalize_ieee14_operating_point({})["voltage_setpoints_pu"].items()
    }
    source_voltage = float(np.clip(1.06 + rng.normal(0.0, voltage_std * 0.6), 1.03, 1.08))
    return _dispatched_operating_point(
        dispatch_mode=_validated_dispatch_mode(dispatch_mode),
        load_scale=float(event_load_scale),
        bus_load_scales=bus_scales,
        case14_point={
            "load_scale": float(event_load_scale),
            "bus_load_scales": bus_scales,
            "generator_dispatch_kw": dispatch,
            "voltage_setpoints_pu": voltage_setpoints,
            "source_voltage_pu": source_voltage,
        },
    )


def _resolve_scan_operating_points(
    rng: np.random.Generator,
    *,
    scan_count: int,
    mode: str,
    event_load_scale: float,
    load_log_std: float,
    dispatch_fraction: float,
    voltage_std: float,
    dispatch_mode: str = DISPATCH_MODE_CASE14,
) -> list[ScanOperatingPoint]:
    """Scan operating points of one window with their dispatch provenance.

    Scan 0 is the reference point at the uniform event load scale; in diverse
    mode the later scans perturb the spatial load profile. Raises
    :class:`OPFDispatchError` in opf mode when any scan's AC-OPF fails.
    """
    if int(scan_count) < 1:
        raise ValueError("scans_per_window must be positive")
    normalized = str(mode).strip().lower()
    if normalized not in {"identical_noise", "diverse"}:
        raise ValueError("operating_point_mode must be identical_noise or diverse")
    dispatch_mode = _validated_dispatch_mode(dispatch_mode)
    reference = _dispatched_operating_point(
        dispatch_mode=dispatch_mode,
        load_scale=float(event_load_scale),
        bus_load_scales=None,
        case14_point={"load_scale": float(event_load_scale)},
    )
    if normalized == "identical_noise":
        return [reference for _ in range(int(scan_count))]
    return [
        reference,
        *[
            _sample_diverse_operating_point(
                rng,
                event_load_scale=float(event_load_scale),
                load_log_std=float(load_log_std),
                dispatch_fraction=float(dispatch_fraction),
                voltage_std=float(voltage_std),
                dispatch_mode=dispatch_mode,
            )
            for _ in range(int(scan_count) - 1)
        ],
    ]


def _scan_operating_points(
    rng: np.random.Generator,
    *,
    scan_count: int,
    mode: str,
    event_load_scale: float,
    load_log_std: float,
    dispatch_fraction: float,
    voltage_std: float,
    dispatch_mode: str = DISPATCH_MODE_CASE14,
) -> list[dict[str, Any]]:
    """Canonical scan operating points only (see :func:`_resolve_scan_operating_points`)."""
    return [
        dict(point.op_point)
        for point in _resolve_scan_operating_points(
            rng,
            scan_count=scan_count,
            mode=mode,
            event_load_scale=event_load_scale,
            load_log_std=load_log_std,
            dispatch_fraction=dispatch_fraction,
            voltage_std=voltage_std,
            dispatch_mode=dispatch_mode,
        )
    ]


def _dispatch_meta(dispatch_mode: str) -> dict[str, Any]:
    """meta.json description of how each scan's dispatch and setpoints were chosen."""
    if dispatch_mode == DISPATCH_MODE_OPF:
        return {
            "mode": DISPATCH_MODE_OPF,
            "solver": OPF_SOLVER,
            "load_scaling": "case14 PD/QD at every bus times load_scale times bus_load_scales[bus]; "
                            "the same per-load factors the OpenDSS scan carries",
            "generator_dispatch_kw": "OPF active output of the units at buses 2, 3, 6 and 8",
            "voltage_setpoints_pu": "OPF voltage magnitude at buses 2, 3, 6 and 8 (PV setpoints)",
            "source_voltage_pu": "OPF voltage magnitude at bus 1; the OpenDSS Vsource reproduces the slack "
                                 "and supplies its active and reactive power",
            "random_draws": "the diverse-scan dispatch and setpoint draws are consumed and discarded so that "
                            "bus_load_scales repeat the case14-mode corpora of the same seed; only the load "
                            "profile is random",
            "failure_policy": "a window whose AC-OPF does not converge for any scan is skipped and listed in "
                              "generation.skipped_windows; the case14 dispatch is never substituted",
            "annotation": "each scan and row carries a `dispatch` block beside its canonical op_point "
                          "(mode, objective, slack output, unit reactive outputs)",
            "healthy_controls": "pypower AC-OPF rows in every dispatch mode",
        }
    return {
        "mode": DISPATCH_MODE_CASE14,
        "solver": None,
        "generator_dispatch_kw": "IEEE14Gen.DSS values (bus 2 at 40 MW, 1 kW condensers at 3/6/8); "
                                 "diverse scans perturb bus 2 by +-scan_dispatch_fraction",
        "voltage_setpoints_pu": "IEEE14Gen.DSS setpoints perturbed by scan_voltage_std in diverse scans",
        "source_voltage_pu": "1.06 perturbed by 0.6 * scan_voltage_std in diverse scans",
        "healthy_controls": "pypower AC-OPF rows in every dispatch mode",
    }


def _build_meta(
    *,
    load_scale_min: float,
    load_scale_max: float,
    r_hif_pu_min: float | None,
    r_hif_pu_max: float | None,
    split_min: float,
    split_max: float,
    branch_sampling: str,
    scans_per_window: int,
    operating_point_mode: str,
    noise_scale: float,
    seed: int,
    scan_load_log_std: float,
    scan_dispatch_fraction: float,
    scan_voltage_std: float,
    branch_current_noise_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    three_phase_noise_pu: float = DEFAULT_THREE_PHASE_SIGMA_PU,
    resistance_units: str = RESISTANCE_UNITS_PU_LEGACY,
    r_hif_pu_range: Sequence[float] | None = None,
    r_hif_ohm_range: Sequence[float] | None = None,
    resistance_sampling: Mapping[str, Any] | None = None,
    voltage_stratum: str | None = None,
    eligible_branch_row0: Sequence[int] | None = None,
    excluded_branch_row0: Sequence[Mapping[str, Any]] | None = None,
    eligibility_source: str = "three_phase_nlm.ieee14_adapter.ELIGIBLE_HIF_BRANCHES",
    shunt_convention: str = SHUNT_CONVENTION_LEGACY,
    telemetry_base_semantics: str = TELEMETRY_BASES_NORMALIZED,
    dispatch_mode: str = DISPATCH_MODE_CASE14,
) -> dict[str, Any]:
    nb = 14
    nl = 20
    dispatch_mode = _validated_dispatch_mode(dispatch_mode)
    idx_map = make_index_map(nb, nl)
    sigma_z = scada_noise_sigma(noise_scale).tolist()
    current_sigma = scaled_sensor_sigma(branch_current_noise_pu, noise_scale, field="branch_current_noise_pu")
    voltage_sigma = scaled_sensor_sigma(three_phase_noise_pu, noise_scale, field="three_phase_noise_pu")
    physical = resistance_units == RESISTANCE_UNITS_OHM_LOCAL_BASE
    convention_payload = measurement_convention_payload(shunt_convention)
    if r_hif_pu_range is None and r_hif_pu_min is not None and r_hif_pu_max is not None:
        r_hif_pu_range = [float(r_hif_pu_min), float(r_hif_pu_max)]
    eligible = list(ELIGIBLE_HIF_BRANCHES) if eligible_branch_row0 is None else list(eligible_branch_row0)
    sampling = dict(resistance_sampling or {"mode": SAMPLING_UNIFORM_PU_LEGACY, "bands_ohm": None, "weights": None, "sweep_values_ohm": None})
    shared_parameters = ["branch_row0", "split_ratio", "phase", "r_hif_pu"]
    if physical:
        shared_parameters += ["resistance_ohm", "local_kv_ll", "zbase_ohm", "voltage_base_profile"]
    per_unit_base = "(S_base/3) / V_LN,base at the terminal bus, S_base=100 MVA"
    if physical:
        per_unit_base += (
            "; V_LN,base = physical local kV_LL/sqrt(3) of the terminal bus "
            f"({VOLTAGE_BASE_PROFILE_ID}); pu values are base-invariant"
        )
    nlm_diagnostic_meta: dict[str, Any] = {
        "fields": ["success", "converged", "top_hif_groups", "detected_top1", "detected_top3"],
        "note": "Generated HIF samples use the legacy three-phase NLM bridge when scenario models are available; metadata fallback is only for adapter smoke tests without model-backed evidence.",
    }
    if physical:
        nlm_diagnostic_meta["detection_limit_note"] = (
            "Rows with label.expected_detectability in {weak, extreme} (>= 500 ohm; 69 kV: >= 10.5 pu) "
            "may legitimately miss the legacy NLM top-3; they are kept as detection-limit cases."
        )
    return {
        "case": "case14",
        "baseMVA": 100.0,
        "nb": nb,
        "nl": nl,
        "index_map": {k: [int(v.start), int(v.stop)] for k, v in idx_map.items()},
        "measurement_order": MEASUREMENT_ORDER,
        "branch_info": _branch_info_case14(),
        "sigma_z": sigma_z,
        "three_phase_sigma": voltage_sigma,
        BRANCH_CURRENT_SIGMA_KEY: current_sigma,
        "noise_contract": generated_noise_contract(
            sigma_z, noise_scale=noise_scale, three_phase_sigma=voltage_sigma,
            branch_current_sigma_pu=current_sigma,
        ),
        MEASUREMENT_CONVENTION_KEY: convention_payload,
        "hif": {
            "scenario": "high_impedance_fault",
            "eligible_branch_row0": [int(i) for i in eligible],
            "excluded_branch_row0": [dict(item) for item in (excluded_branch_row0 or [])],
            "eligibility_source": str(eligibility_source),
            "branch_order": BRANCH_ORDER,
            "bus_order": BUS_ORDER,
            "load_scale_range": [float(load_scale_min), float(load_scale_max)],
            "r_hif_pu_range": None if r_hif_pu_range is None else [float(r_hif_pu_range[0]), float(r_hif_pu_range[1])],
            "r_hif_ohm_range": None if r_hif_ohm_range is None else [float(r_hif_ohm_range[0]), float(r_hif_ohm_range[1])],
            "resistance_units": str(resistance_units),
            "resistance_sampling": {
                "mode": sampling.get("mode"),
                "bands_ohm": sampling.get("bands_ohm"),
                "weights": sampling.get("weights"),
                "sweep_values_ohm": sampling.get("sweep_values_ohm"),
                **{k: v for k, v in sampling.items() if k in ("band_probabilities", "cell_schedule", "cell_count", "schedule_truncated")},
            },
            "voltage_stratum": voltage_stratum,
            "voltage_base_profile": ieee14_voltage_base_profile() if physical else None,
            "model_ohm_convention": MODEL_OHM_CONVENTION,
            MEASUREMENT_CONVENTION_KEY: convention_payload,
            "telemetry_base_semantics": str(telemetry_base_semantics),
            "split_ratio_range": [float(split_min), float(split_max)],
            "branch_sampling": str(branch_sampling),
            "dispatch": _dispatch_meta(dispatch_mode),
            "scan_window": {
                "scans_per_window": int(scans_per_window),
                "operating_point_mode": str(operating_point_mode),
                "dispatch_mode": dispatch_mode,
                "shared_parameters": shared_parameters,
                "scan_specific_fields": [
                    "z_clean",
                    "z_obs",
                    "three_phase_voltages",
                    "three_phase_voltages_clean",
                    BRANCH_CURRENT_CHANNEL,
                    f"{BRANCH_CURRENT_CHANNEL}_clean",
                    "sigma_z", "three_phase_sigma", BRANCH_CURRENT_SIGMA_KEY,
                    "op_point", "dispatch",
                ],
                "operating_point_schema": list(IEEE14_OPERATING_POINT_KEYS),
                "bus_load_scale_semantics": "profile_factor_multiplied_by_load_scale",
                "note": ("identical_noise repeats one operating point; diverse varies the spatial load profile "
                         "while preserving the HIF; dispatch and voltage setpoints follow dispatch_mode "
                         "(opf: AC-OPF at each scan's loads; case14: model values with random perturbations)."),
            },
            "generation": {
                "seed": int(seed),
                "noise_scale": float(noise_scale),
                "dispatch_mode": dispatch_mode,
                "three_phase_noise_pu": float(three_phase_noise_pu),
                "branch_current_noise_pu": float(branch_current_noise_pu),
                "applied_three_phase_sigma": voltage_sigma,
                "applied_branch_current_sigma_pu": current_sigma,
                "scan_load_log_std": float(scan_load_log_std),
                "scan_dispatch_fraction": float(scan_dispatch_fraction),
                "scan_voltage_std": float(scan_voltage_std),
                "rng_streams": ("event_labels, operating_points, and measurement_noise are independent; "
                                "the phasor sigmas scale the same standard-normal draws, so SCADA noise does "
                                "not depend on them"),
                "skipped_windows": [],
                "skipped_window_count": 0,
                "skipped_controls": [],
            },
            "phases": list(PHASES),
            "measurement_vector": "operator IEEE-14 122-entry z; hidden fault bus excluded",
            "branch_current_measurements": {
                "channel": BRANCH_CURRENT_CHANNEL,
                "type": "per_phase_terminal_current_phasors",
                "fields": [
                    "branch",
                    "branch_row0",
                    "from_bus",
                    "to_bus",
                    "i_from_pu",
                    "ang_from_deg",
                    "i_to_pu",
                    "ang_to_deg",
                    "ibase_from_a",
                    "ibase_to_a",
                ],
                "sign_convention": "current flowing into the branch from each terminal",
                "per_unit_base": per_unit_base,
                "telemetry_base_semantics": str(telemetry_base_semantics),
                "noise_model": "independent Gaussian per real/imaginary component",
                BRANCH_CURRENT_SIGMA_KEY: current_sigma,
                "applied_noise_sigma_pu": current_sigma,
                "hidden_clean_copy": f"{BRANCH_CURRENT_CHANNEL}_clean",
                "note": (
                    "Two-terminal per-phase currents support fault estimation; "
                    "identifiability still depends on signal and noise. The "
                    "hidden clean copy exists for QA replay only."
                ),
            },
            "nlm_diagnostic": nlm_diagnostic_meta,
        },
    }


def _no_error_measurement_convention() -> dict[str, Any]:
    payload = measurement_convention_payload(SHUNT_CONVENTION_YBUS)
    payload["note"] = (
        "no_error controls come from pypower case14 runopf, whose Pinj/Qinj follow makeSbus "
        "(bus shunts stay in Ybus and are excluded from injections) irrespective of the HIF rows' exporter convention."
    )
    return payload


#: How the row-level balanced reference ``z_true`` is produced. The OpenDSS
#: reference is the same operating point as scan 0 (dispatch, load profile,
#: voltage setpoints, shunt convention) with the fault removed; the pypower
#: OPF reference has a different dispatch and is kept only as ``z_reference_opf``.
BALANCED_REFERENCE_OPENDSS = "opendss_same_operating_point"
BALANCED_REFERENCE_PYPOWER_OPF = "pypower_opf"
BALANCED_REFERENCE_MODES = (BALANCED_REFERENCE_OPENDSS, BALANCED_REFERENCE_PYPOWER_OPF)


def generate_dataset(
    *,
    out_dir: str,
    n_hif: int,
    n_no_error: int,
    seed: int,
    load_scale_min: float,
    load_scale_max: float,
    r_hif_pu_min: float | None = None,
    r_hif_pu_max: float | None = None,
    split_min: float,
    split_max: float,
    noise_scale: float,
    keep_scenarios: bool,
    branch_sampling: str,
    scans_per_window: int = 1,
    operating_point_mode: str = "diverse",
    scan_load_log_std: float = 0.08,
    scan_dispatch_fraction: float = 0.20,
    scan_voltage_std: float = 0.008,
    branch_current_noise_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    three_phase_noise_pu: float = DEFAULT_THREE_PHASE_SIGMA_PU,
    resistance_units: str = "ohm",
    r_hif_ohm_min: float | None = None,
    r_hif_ohm_max: float | None = None,
    r_hif_ohm_bands: Any = None,
    r_hif_ohm_band_weights: Any = None,
    r_hif_ohm_sweep: Any = None,
    voltage_stratum: str = "69kv",
    shunt_convention: str | None = None,
    balanced_reference: str | None = None,
    dispatch_mode: str = DISPATCH_MODE_OPF,
) -> None:
    if int(scans_per_window) < 1:
        raise ValueError("scans_per_window must be positive")
    dispatch_mode = _validated_dispatch_mode(dispatch_mode)
    noise_scale = positive_sigma(noise_scale, field="noise_scale")
    sigma_z = scada_noise_sigma(noise_scale).tolist()
    applied_current_sigma = scaled_sensor_sigma(branch_current_noise_pu, noise_scale, field="branch_current_noise_pu")
    applied_voltage_sigma = scaled_sensor_sigma(three_phase_noise_pu, noise_scale, field="three_phase_noise_pu")
    noise_contract = generated_noise_contract(
        sigma_z, noise_scale=noise_scale, three_phase_sigma=applied_voltage_sigma,
        branch_current_sigma_pu=applied_current_sigma,
    )
    if float(scan_load_log_std) < 0.0:
        raise ValueError("scan_load_log_std must be non-negative")
    if not 0.0 <= float(scan_dispatch_fraction) < 1.0:
        raise ValueError("scan_dispatch_fraction must be in [0, 1)")
    if float(scan_voltage_std) < 0.0:
        raise ValueError("scan_voltage_std must be non-negative")
    if branch_sampling not in {"balanced", "random"}:
        raise ValueError(f"Unknown branch_sampling={branch_sampling!r}")

    sampling = resolve_resistance_sampling(
        resistance_units=resistance_units,
        r_hif_pu_min=r_hif_pu_min,
        r_hif_pu_max=r_hif_pu_max,
        r_hif_ohm_min=r_hif_ohm_min,
        r_hif_ohm_max=r_hif_ohm_max,
        r_hif_ohm_bands=r_hif_ohm_bands,
        r_hif_ohm_band_weights=r_hif_ohm_band_weights,
        r_hif_ohm_sweep=r_hif_ohm_sweep,
        voltage_stratum=voltage_stratum,
    )
    physical = sampling["resistance_units"] == RESISTANCE_UNITS_OHM_LOCAL_BASE
    convention = resolve_shunt_convention_default(resistance_units, shunt_convention)
    convention_payload = measurement_convention_payload(convention)
    balanced_reference = balanced_reference or (
        BALANCED_REFERENCE_OPENDSS if physical else BALANCED_REFERENCE_PYPOWER_OPF)
    if balanced_reference not in BALANCED_REFERENCE_MODES:
        raise ValueError(f"balanced_reference must be one of {BALANCED_REFERENCE_MODES}, got {balanced_reference!r}")
    telemetry_bases = TELEMETRY_BASES_PHYSICAL if physical else TELEMETRY_BASES_NORMALIZED

    if physical:
        eligible_hif_branches = [int(i) for i in eligible_rows_for_stratum(voltage_stratum)]
        excluded_rows = [dict(item) for item in EXCLUDED_CROSS_VOLTAGE_BRANCHES]
        eligibility_source = "three_phase_nlm.hif_units.eligible_rows_for_stratum"
        meta_stratum: str | None = str(voltage_stratum)
    else:
        eligible_hif_branches = [int(i) for i in ELIGIBLE_HIF_BRANCHES]
        excluded_rows = []
        eligibility_source = "three_phase_nlm.ieee14_adapter.ELIGIBLE_HIF_BRANCHES"
        meta_stratum = None
    if not eligible_hif_branches:
        raise ValueError(f"No eligible HIF lines for voltage_stratum={voltage_stratum!r}")

    event_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    no_error_noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    out = Path(os.path.abspath(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    sweep_schedule: list[tuple[int, float, str]] | None = None
    if sampling["mode"] == SAMPLING_SWEEP_OHM:
        sweep_schedule, truncated = build_sweep_schedule(
            event_rng,
            eligible_rows=eligible_hif_branches,
            sweep_values_ohm=sampling["sweep_values_ohm"],
            n_hif=int(n_hif),
        )
        sampling["cell_count"] = len(eligible_hif_branches) * len(sampling["sweep_values_ohm"]) * len(PHASES)
        sampling["schedule_truncated"] = bool(truncated)
        if branch_sampling != "balanced":
            print("NOTE: --r-hif-ohm-sweep always uses the balanced (line x resistance x phase) schedule; "
                  f"--branch-sampling {branch_sampling} is ignored.", file=sys.stderr)

    meta = _build_meta(
        load_scale_min=load_scale_min,
        load_scale_max=load_scale_max,
        r_hif_pu_min=sampling["r_hif_pu_min"],
        r_hif_pu_max=sampling["r_hif_pu_max"],
        split_min=split_min,
        split_max=split_max,
        branch_sampling=branch_sampling,
        scans_per_window=scans_per_window,
        operating_point_mode=operating_point_mode,
        noise_scale=noise_scale,
        seed=seed,
        scan_load_log_std=scan_load_log_std,
        scan_dispatch_fraction=scan_dispatch_fraction,
        scan_voltage_std=scan_voltage_std,
        branch_current_noise_pu=branch_current_noise_pu,
        three_phase_noise_pu=three_phase_noise_pu,
        resistance_units=sampling["resistance_units"],
        r_hif_pu_range=sampling["r_hif_pu_range"],
        r_hif_ohm_range=sampling["r_hif_ohm_range"],
        resistance_sampling=sampling,
        voltage_stratum=meta_stratum,
        eligible_branch_row0=eligible_hif_branches,
        excluded_branch_row0=excluded_rows,
        eligibility_source=eligibility_source,
        shunt_convention=convention,
        telemetry_base_semantics=telemetry_bases,
        dispatch_mode=dispatch_mode,
    )
    meta["hif"]["balanced_reference"] = balanced_reference
    # Physics revision marker: the operating-point path keeps the generator
    # reactive limits declared in IEEE14Gen.DSS. Corpora generated before
    # 2026-09-23 lack this key; their PV units were pinned at +-1.08*kW kvar.
    meta["hif"]["generator_reactive_limits"] = {
        "policy": "model_file_limits_kept",
        "note": "apply_hif_operating_point restores Maxkvar/Minkvar after dispatch and setpoint writes; "
                "PV generators regulate within their limits",
    }
    opf_dispatch = dispatch_mode == DISPATCH_MODE_OPF
    meta["hif"]["z_true_semantics"] = (
        "row-level z_true: balanced OpenDSS solve at scan 0's operating point with the fault removed, "
        "same dispatch/load profile/shunt convention as the scans; z_reference_opf is the pypower OPF vector"
        + (" at the same loads (scan 0's dispatch is that OPF's dispatch)" if opf_dispatch else
           " (different dispatch than the OpenDSS scans)")
        if balanced_reference == BALANCED_REFERENCE_OPENDSS else
        "row-level z_true: pypower OPF balanced case"
        + (" (scan 0's dispatch is that OPF's dispatch)" if opf_dispatch else
           " (different dispatch than the OpenDSS scans)"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    skipped_windows: list[dict[str, Any]] = meta["hif"]["generation"]["skipped_windows"]
    skipped_controls: list[dict[str, Any]] = meta["hif"]["generation"]["skipped_controls"]
    ppc_base = case14()
    base_dss_dir = Path(_REPO_ROOT) / "IEEE_14_OpenDSS"
    scenarios_root = out / "scenarios"
    if keep_scenarios:
        scenarios_root.mkdir(exist_ok=True)

    branch_schedule: list[int] = []
    if branch_sampling == "balanced" and sweep_schedule is None:
        while len(branch_schedule) < int(n_hif):
            branch_schedule.extend(int(i) for i in event_rng.permutation(eligible_hif_branches))
        branch_schedule = branch_schedule[: int(n_hif)]

    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for control_idx in range(int(n_no_error)):
            alpha = float(event_rng.uniform(load_scale_min, load_scale_max))
            solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
            if solved is None:
                skipped_controls.append({"control_index": int(control_idx), "load_scale": alpha,
                                         "reason": "pypower AC-OPF did not converge"})
                print(f"WARNING: skipping healthy control {control_idx} (load_scale={alpha:.6f}): "
                      "pypower AC-OPF did not converge", file=sys.stderr)
                continue
            z = compute_measurements_pu(solved).astype(float).tolist()
            rec = {
                "id": f"nehif_{event_rng.integers(1e12)}",
                "scenario": "no_error",
                "z_true": z,
                "z_clean": z,
                "z_obs": _maybe_add_noise(z, no_error_noise_rng, noise_scale),
                "sigma_z": sigma_z,
                "noise_contract": generated_noise_contract(sigma_z, noise_scale=noise_scale),
                MEASUREMENT_CONVENTION_KEY: _no_error_measurement_convention(),
                "label": {"error_type": "no_error"},
                "op_point": {"load_scale": alpha, "seed": int(seed)},
                "dispatch": {**opf_dispatch_receipt(solved),
                             "note": "healthy controls are pypower AC-OPF rows in every dispatch mode"},
            }
            handle.write(json.dumps(rec) + "\n")

        for sample_idx in range(int(n_hif)):
            scan_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(sample_idx), 2])
            )
            measurement_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(sample_idx), 3])
            )
            sample_id = f"ieee14_hif_{sample_idx:06d}"
            alpha = float(event_rng.uniform(load_scale_min, load_scale_max))

            # --- event parameters: legacy draw order is alpha, [branch], split, phase, resistance
            sweep_cell = sweep_schedule[sample_idx] if sweep_schedule is not None else None
            if sweep_cell is not None:
                branch_row0 = int(sweep_cell[0])
            elif branch_sampling == "balanced":
                branch_row0 = int(branch_schedule[sample_idx])
            else:
                branch_row0 = int(event_rng.choice(eligible_hif_branches))
            dss_element = BRANCH_ORDER[branch_row0]
            split_ratio = float(event_rng.uniform(split_min, split_max))
            phase = str(sweep_cell[2]) if sweep_cell is not None else str(event_rng.choice(list(PHASES)))

            resistance_fields: dict[str, Any] = {}
            if sampling["mode"] == SAMPLING_UNIFORM_PU_LEGACY:
                r_hif_pu = float(event_rng.uniform(sampling["r_hif_pu_min"], sampling["r_hif_pu_max"]))
                r_hif_model_ohm = hif_ohms_from_pu(r_hif_pu, base_mva=100.0, kv_ll=1.0)
                record = None
            else:
                if sampling["mode"] == SAMPLING_SWEEP_OHM:
                    resistance_ohm = float(sweep_cell[1])
                    resistance_fields = {"sweep_value_ohm": resistance_ohm}
                elif sampling["mode"] == SAMPLING_BANDS_OHM:
                    band_index = int(event_rng.choice(len(sampling["bands_ohm"]), p=sampling["band_probabilities"]))
                    band = sampling["bands_ohm"][band_index]
                    resistance_ohm = float(event_rng.uniform(band[0], band[1]))
                    resistance_fields = {"resistance_band_ohm": [float(band[0]), float(band[1])]}
                else:
                    band = sampling["r_hif_ohm_range"]
                    resistance_ohm = float(event_rng.uniform(band[0], band[1]))
                    resistance_fields = {"resistance_band_ohm": [float(band[0]), float(band[1])]}
                record = hif_resistance_record(branch_row0=branch_row0, resistance_ohm=resistance_ohm)
                r_hif_pu = float(record["r_hif_pu"])
                r_hif_model_ohm = float(record["r_hif_model_ohm"])

            # Every event draw is done; the operating points come from their own
            # stream, so a skipped window leaves the labels of later windows intact.
            try:
                scan_points = _resolve_scan_operating_points(
                    scan_rng,
                    scan_count=int(scans_per_window),
                    mode=operating_point_mode,
                    event_load_scale=alpha,
                    load_log_std=scan_load_log_std,
                    dispatch_fraction=scan_dispatch_fraction,
                    voltage_std=scan_voltage_std,
                    dispatch_mode=dispatch_mode,
                )
            except OPFDispatchError as exc:
                skipped_windows.append({"id": sample_id, "sample_index": int(sample_idx), "load_scale": alpha,
                                        "branch_row0": int(branch_row0), "reason": str(exc)})
                print(f"WARNING: skipping {sample_id}: {exc}", file=sys.stderr)
                continue
            scan_op_points = [point.op_point for point in scan_points]

            if keep_scenarios:
                scenario_dir = scenarios_root / sample_id
                if scenario_dir.exists():
                    shutil.rmtree(scenario_dir)
                tmp_context = None
            else:
                tmp_context = tempfile.TemporaryDirectory(prefix=f"{sample_id}_")
                scenario_dir = Path(tmp_context.name)

            try:
                copy_ieee14_model(base_dss_dir, scenario_dir, overwrite=True)
                write_balanced_ieee14_load_override(scenario_dir)
                # The injector always receives MODEL ohms on the kV=1 normalized model.
                injection = inject_midspan_hif_ieee14(
                    scenario_dir,
                    dss_element,
                    split_ratio=split_ratio,
                    phase=phase,
                    r_hif_ohm=r_hif_model_ohm,
                    base_mva=100.0,
                    kv_ll=1.0,
                    fault_bus=f"Fault_{branch_row0 + 1}_{sample_idx:06d}",
                    hif_load_name=f"Load.HIF_{branch_row0 + 1}_{sample_idx:06d}",
                )

                scans = []
                simulation_cache: dict[str, dict[str, Any]] = {}
                for scan_index, scan_point in enumerate(scan_points):
                    scan_op_point = scan_point.op_point
                    op_key = json.dumps(scan_op_point, sort_keys=True, separators=(",", ":"))
                    if op_key not in simulation_cache:
                        simulation_cache[op_key] = simulate_hif_candidate(
                            candidate_branch_row0=branch_row0,
                            alpha=split_ratio,
                            phase=phase,
                            r_hif_pu=r_hif_pu,
                            op_point=scan_op_point,
                            pristine_model_dir=str(base_dss_dir),
                            shunt_convention=convention,
                        )
                    simulated = simulation_cache[op_key]
                    z_scan = simulated["z"]
                    if len(z_scan) != 122:
                        raise RuntimeError(f"Unexpected z_obs length={len(z_scan)}, expected 122")
                    clean_currents = list(simulated[BRANCH_CURRENT_CHANNEL])
                    scan = {
                        "scan_index": int(scan_index),
                        "z_clean": [float(x) for x in z_scan],
                        "z_obs": _maybe_add_noise(
                            [float(x) for x in z_scan], measurement_rng, noise_scale
                        ),
                        "three_phase_voltages": add_voltage_phasor_noise(
                            simulated["three_phase_voltages"], measurement_rng, applied_voltage_sigma
                        ),
                        "three_phase_voltages_clean": deepcopy(simulated["three_phase_voltages"]),
                        BRANCH_CURRENT_CHANNEL: add_branch_current_noise(
                            clean_currents, measurement_rng, applied_current_sigma
                        ),
                        f"{BRANCH_CURRENT_CHANNEL}_clean": clean_currents,
                        BRANCH_CURRENT_SIGMA_KEY: applied_current_sigma,
                        "three_phase_sigma": applied_voltage_sigma,
                        "sigma_z": sigma_z,
                        "noise_contract": noise_contract,
                        MEASUREMENT_CONVENTION_KEY: dict(convention_payload),
                        "op_point": scan_op_point,
                        "dispatch": deepcopy(scan_point.dispatch),
                        "topology_id": "ieee14_base",
                    }
                    if physical:
                        scan = _rewrite_scan_telemetry_bases(scan)
                    scans.append(scan)

                reference_scan = scans[0]

                # In opf mode scan 0's dispatch is this very OPF (uniform load at alpha).
                if scan_points[0].opf is not None:
                    solved = scan_points[0].opf.solution
                else:
                    solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
                if solved is None:
                    skipped_windows.append({"id": sample_id, "sample_index": int(sample_idx), "load_scale": alpha,
                                            "branch_row0": int(branch_row0),
                                            "reason": "pypower AC-OPF reference at the event load scale did not converge"})
                    print(f"WARNING: skipping {sample_id}: pypower AC-OPF reference did not converge", file=sys.stderr)
                    continue
                z_reference_opf = compute_measurements_pu(solved).astype(float).tolist()
                if balanced_reference == BALANCED_REFERENCE_OPENDSS:
                    # Same OpenDSS operating point as the reference scan (dispatch, load
                    # profile, voltage setpoints, shunt convention) with the fault removed:
                    # the paired healthy sensor mean, not a re-dispatched OPF case.
                    balanced_model = _simulate_base(
                        _resolve_model_dir(str(base_dss_dir), "case14"),
                        op_point=scan_op_points[0], shunt_convention=convention,
                    )
                    z_true = [float(x) for x in balanced_model["z"]]
                    z_true_semantics = ("balanced_same_operating_point_opendss_reference; fault removed, same "
                                        "dispatch, load profile and shunt convention as scan 0; z_clean is the "
                                        "HIF sensor mean of scan 0")
                else:
                    z_true = z_reference_opf
                    z_true_semantics = "balanced_reference_pypower_opf; different dispatch than the OpenDSS scans"

                branch_info = branch_info_for_row0(branch_row0)
                # The legacy NLM bridge also works in MODEL ohms (its z_base uses kv_ln=0.577).
                nlm_diagnostic = run_ieee14_hif_nlm(
                    pristine_model_dir=str(base_dss_dir),
                    faulted_model_dir=str(scenario_dir),
                    target_dss_element=dss_element,
                    phase=phase,
                    r_hif_ohm=r_hif_model_ohm,
                    target_branch_row0=branch_row0,
                    load_scale=alpha,
                )
                top_rows = [
                    group.get("branch_row0")
                    for group in nlm_diagnostic.get("top_hif_groups", [])
                    if isinstance(group, Mapping)
                ]
                nlm_diagnostic["detected_top1"] = bool(top_rows and top_rows[0] == branch_row0)
                nlm_diagnostic["detected_top3"] = branch_row0 in top_rows[:3]
                nlm_diagnostic["detected"] = bool(nlm_diagnostic["detected_top3"])

                if record is None:
                    shared_label = {
                        "error_type": "high_impedance_fault",
                        "branch_row0": branch_row0,
                        "line_index1": branch_row0 + 1,
                        "dss_element": dss_element,
                        "from_bus": branch_info["from_bus"],
                        "to_bus": branch_info["to_bus"],
                        "phase": phase,
                        "split_ratio": split_ratio,
                        "fault_bus": injection.fault_bus,
                        "r_hif_pu": r_hif_pu,
                        "r_hif_ohm": r_hif_model_ohm,
                        "kv_ln": injection.kv_ln,
                        "resistance_units": RESISTANCE_UNITS_PU_LEGACY,
                    }
                else:
                    shared_label = {
                        **record,
                        "error_type": "high_impedance_fault",
                        "branch_row0": branch_row0,
                        "line_index1": branch_row0 + 1,
                        "dss_element": dss_element,
                        "from_bus": branch_info["from_bus"],
                        "to_bus": branch_info["to_bus"],
                        "phase": phase,
                        "split_ratio": split_ratio,
                        "fault_bus": injection.fault_bus,
                        "r_hif_pu": r_hif_pu,
                        # PHYSICAL ohms; resistance_units="ohm_local_base" marks the semantics.
                        "r_hif_ohm": float(record["resistance_ohm"]),
                        "resistance_units": RESISTANCE_UNITS_OHM_LOCAL_BASE,
                        # Model load LN kV kept for legacy readers; never a physical value.
                        "kv_ln": injection.kv_ln,
                        "kv_ln_semantics": KV_LN_SEMANTICS_MODEL,
                        "resistance_sampling_mode": sampling["mode"],
                        **resistance_fields,
                        "expected_detectability": expected_detectability(record["resistance_class"]),
                    }
                rec = {
                    "id": sample_id,
                    "scenario": "high_impedance_fault",
                    "case": "IEEE14",
                    "z_true": z_true,
                    "z_clean": reference_scan["z_clean"],
                    "z_obs": reference_scan["z_obs"],
                    "three_phase_voltages": reference_scan["three_phase_voltages"],
                    "three_phase_voltages_clean": reference_scan["three_phase_voltages_clean"],
                    BRANCH_CURRENT_CHANNEL: reference_scan[BRANCH_CURRENT_CHANNEL],
                    f"{BRANCH_CURRENT_CHANNEL}_clean": reference_scan[f"{BRANCH_CURRENT_CHANNEL}_clean"],
                    BRANCH_CURRENT_SIGMA_KEY: applied_current_sigma,
                    "three_phase_sigma": applied_voltage_sigma,
                    "noise_contract": noise_contract,
                    MEASUREMENT_CONVENTION_KEY: dict(convention_payload),
                    "z_true_semantics": z_true_semantics,
                    "z_reference_opf": z_reference_opf,
                    "balanced_reference": balanced_reference,
                    "label": shared_label,
                    "shared_label": shared_label,
                    "nlm_diagnostic": nlm_diagnostic,
                    "scan_count": len(scans),
                    "scans": scans,
                    "sigma_z": sigma_z,
                    "topology_id": "ieee14_base",
                    "op_point": reference_scan["op_point"],
                    "dispatch": deepcopy(reference_scan["dispatch"]),
                    "window_metadata": {
                        "operating_point_mode": str(operating_point_mode),
                        "dispatch_mode": dispatch_mode,
                        "persistent_hif": True,
                        "seed": int(seed),
                        "sample_index": int(sample_idx),
                    },
                }
                if physical:
                    rec["voltage_base_profile"] = VOLTAGE_BASE_PROFILE_ID
                if keep_scenarios:
                    rec["scenario_model_dir"] = str(scenario_dir.relative_to(out))
                handle.write(json.dumps(rec) + "\n")
            finally:
                if tmp_context is not None:
                    tmp_context.cleanup()

    # Record every skipped window and control (an OPF that did not converge is
    # an exclusion, never a silent fallback to the case14 dispatch).
    meta["hif"]["generation"]["skipped_window_count"] = len(skipped_windows)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if skipped_windows or skipped_controls:
        print(f"WARNING: {len(skipped_windows)} HIF windows and {len(skipped_controls)} healthy controls were "
              f"skipped; see meta.json hif.generation.skipped_windows / skipped_controls", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/measurements/out_measurements_hif", help="Output directory")
    parser.add_argument("--n-hif", type=int, default=200)
    parser.add_argument("--n-no-error", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--load-scale-min", type=float, default=0.80)
    parser.add_argument("--load-scale-max", type=float, default=1.25)
    parser.add_argument(
        "--resistance-units",
        choices=list(RESISTANCE_UNITS_CHOICES),
        default="ohm",
        help="ohm: physical ohms on the faulted line's local voltage base (default); pu: legacy system-pu sampling.",
    )
    parser.add_argument("--r-hif-ohm-min", type=float, default=None,
                        help=f"Uniform ohm sampling lower bound (ohm mode; default {DEFAULT_R_HIF_OHM_MIN:g}).")
    parser.add_argument("--r-hif-ohm-max", type=float, default=None,
                        help=f"Uniform ohm sampling upper bound (ohm mode; default {DEFAULT_R_HIF_OHM_MAX:g}).")
    parser.add_argument("--r-hif-ohm-bands", default=None,
                        help='Weighted bands, e.g. "100:200,200:500,500:1000" (ohm mode).')
    parser.add_argument("--r-hif-ohm-band-weights", default=None,
                        help='Band weights matching --r-hif-ohm-bands, e.g. "1,1,1".')
    parser.add_argument("--r-hif-ohm-sweep", default=None,
                        help='Discrete sweep, e.g. "50,100,200,500,1000,2000,5000"; balanced over line x resistance x phase cells.')
    parser.add_argument("--r-hif-pu-min", type=float, default=None,
                        help=f"Legacy pu lower bound; only with --resistance-units pu (default {DEFAULT_R_HIF_PU_MIN:g}).")
    parser.add_argument("--r-hif-pu-max", type=float, default=None,
                        help=f"Legacy pu upper bound; only with --resistance-units pu (default {DEFAULT_R_HIF_PU_MAX:g}).")
    parser.add_argument(
        "--voltage-stratum",
        choices=list(VOLTAGE_STRATUM_CHOICES),
        default="69kv",
        help="Eligible same-voltage lines in ohm mode (Line.7-8 is always excluded); ignored in pu mode.",
    )
    parser.add_argument(
        "--shunt-convention",
        choices=list(SHUNT_CONVENTIONS),
        default=None,
        help="Exporter bus-shunt convention; default ybus in ohm mode, legacy_injection in pu mode.",
    )
    parser.add_argument("--split-min", type=float, default=0.25)
    parser.add_argument("--split-max", type=float, default=0.75)
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="Positive common multiplier for applied noise and declared sigmas; noiseless simulator output is separate.")
    parser.add_argument("--scans-per-window", type=int, default=1)
    parser.add_argument(
        "--operating-point-mode",
        choices=["identical_noise", "diverse"],
        default="diverse",
        help="Use repeated-noise controls or electrically diverse operating points within each persistent HIF event.",
    )
    parser.add_argument("--scan-load-log-std", type=float, default=0.08)
    parser.add_argument("--scan-dispatch-fraction", type=float, default=0.20,
                        help="case14 dispatch mode only: +- fraction on the bus-2 unit in diverse scans "
                             "(the draw is consumed but discarded in opf mode).")
    parser.add_argument("--scan-voltage-std", type=float, default=0.008,
                        help="case14 dispatch mode only: setpoint perturbation in diverse scans "
                             "(the draws are consumed but discarded in opf mode).")
    parser.add_argument(
        "--dispatch-mode",
        choices=list(DISPATCH_MODES),
        default=DISPATCH_MODE_OPF,
        help="opf (default): every scan's generator dispatch, PV setpoints and source voltage come from the "
             "pypower AC-OPF on case14 at the scan's own per-bus loads (the dispatch law of the pypower "
             "scenario families); case14: the checked-in model dispatch with random perturbations. A "
             "non-converged OPF skips the window and is recorded in meta.json.",
    )
    parser.add_argument("--three-phase-noise-pu", type=float, default=DEFAULT_THREE_PHASE_SIGMA_PU,
                        help="Phase-voltage real/imaginary component sigma before multiplying by --noise-scale.")
    parser.add_argument(
        "--branch-current-noise-pu",
        type=float,
        default=DEFAULT_BRANCH_CURRENT_SIGMA_PU,
        help=(
            "Per-component sigma of the per-phase branch-current phasors before "
            "--noise-scale; both applied noise and exported sigma use the scaled value."
        ),
    )
    parser.add_argument("--keep-scenarios", action="store_true")
    parser.add_argument(
        "--branch-sampling",
        choices=["balanced", "random"],
        default="balanced",
        help="Balanced cycles through the eligible lines (stratum-filtered in ohm mode); random samples with replacement. Sweep mode balances (line x resistance x phase) cells.",
    )
    parser.add_argument(
        "--balanced-reference",
        choices=list(BALANCED_REFERENCE_MODES),
        default=None,
        help="Row-level z_true: a balanced OpenDSS solve at scan 0's operating point (default in ohm mode) "
             "or the pypower OPF case (default in legacy pu mode).",
    )
    args = parser.parse_args(argv)

    if args.resistance_units == "ohm" and (args.r_hif_pu_min is not None or args.r_hif_pu_max is not None):
        parser.error("--r-hif-pu-min/--r-hif-pu-max are only accepted with --resistance-units pu")
    if args.resistance_units == "pu" and any(
        value is not None for value in (args.r_hif_ohm_min, args.r_hif_ohm_max, args.r_hif_ohm_bands,
                                         args.r_hif_ohm_band_weights, args.r_hif_ohm_sweep)
    ):
        parser.error("ohm sampling options require --resistance-units ohm")
    try:
        sampling = resolve_resistance_sampling(
            resistance_units=args.resistance_units,
            r_hif_pu_min=args.r_hif_pu_min,
            r_hif_pu_max=args.r_hif_pu_max,
            r_hif_ohm_min=args.r_hif_ohm_min,
            r_hif_ohm_max=args.r_hif_ohm_max,
            r_hif_ohm_bands=args.r_hif_ohm_bands,
            r_hif_ohm_band_weights=args.r_hif_ohm_band_weights,
            r_hif_ohm_sweep=args.r_hif_ohm_sweep,
            voltage_stratum=args.voltage_stratum,
        )
    except ValueError as exc:
        parser.error(str(exc))

    generate_dataset(
        out_dir=args.out,
        n_hif=args.n_hif,
        n_no_error=args.n_no_error,
        seed=args.seed,
        load_scale_min=args.load_scale_min,
        load_scale_max=args.load_scale_max,
        r_hif_pu_min=args.r_hif_pu_min,
        r_hif_pu_max=args.r_hif_pu_max,
        split_min=args.split_min,
        split_max=args.split_max,
        noise_scale=args.noise_scale,
        keep_scenarios=bool(args.keep_scenarios),
        branch_sampling=args.branch_sampling,
        scans_per_window=int(args.scans_per_window),
        operating_point_mode=args.operating_point_mode,
        scan_load_log_std=float(args.scan_load_log_std),
        scan_dispatch_fraction=float(args.scan_dispatch_fraction),
        scan_voltage_std=float(args.scan_voltage_std),
        branch_current_noise_pu=float(args.branch_current_noise_pu),
        three_phase_noise_pu=float(args.three_phase_noise_pu),
        resistance_units=args.resistance_units,
        r_hif_ohm_min=args.r_hif_ohm_min,
        r_hif_ohm_max=args.r_hif_ohm_max,
        r_hif_ohm_bands=args.r_hif_ohm_bands,
        r_hif_ohm_band_weights=args.r_hif_ohm_band_weights,
        r_hif_ohm_sweep=args.r_hif_ohm_sweep,
        voltage_stratum=args.voltage_stratum,
        shunt_convention=args.shunt_convention,
        balanced_reference=args.balanced_reference,
        dispatch_mode=args.dispatch_mode,
    )
    convention = resolve_shunt_convention_default(args.resistance_units, args.shunt_convention)
    if sampling["resistance_units"] == RESISTANCE_UNITS_OHM_LOCAL_BASE:
        eligible = eligible_rows_for_stratum(args.voltage_stratum)
        print(
            f"Wrote IEEE-14 HIF dataset to: {args.out} "
            f"[resistance_units={sampling['resistance_units']} mode={sampling['mode']} "
            f"r_hif_ohm_range={sampling['r_hif_ohm_range']} voltage_stratum={args.voltage_stratum} "
            f"eligible_branch_row0={eligible} shunt_convention={convention} dispatch_mode={args.dispatch_mode} "
            f"three_phase_noise_pu={args.three_phase_noise_pu:g} branch_current_noise_pu={args.branch_current_noise_pu:g}]"
        )
    else:
        print(
            f"Wrote IEEE-14 HIF dataset to: {args.out} "
            f"[resistance_units={sampling['resistance_units']} r_hif_pu_range={sampling['r_hif_pu_range']} "
            f"eligible_branch_row0={[int(i) for i in ELIGIBLE_HIF_BRANCHES]} shunt_convention={convention} "
            f"dispatch_mode={args.dispatch_mode}]"
        )


if __name__ == "__main__":
    main()
