from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from IEEE_14_OpenDSS.constants import (
    IEEE14_GENERATOR_DISPATCH_KW,
    IEEE14_GENERATOR_VOLTAGE_PU,
    IEEE14_LOAD_BASE_KW,
    IEEE14_SOURCE_VOLTAGE_PU,
)


@dataclass(frozen=True)
class IEEE14OperatingPoint:
    load_scale: float
    bus_load_scales: dict[str, float]
    generator_dispatch_kw: dict[str, float]
    voltage_setpoints_pu: dict[str, float]
    source_voltage_pu: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any, *, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def normalize_bus_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.split(r"[.\s]", text, maxsplit=1)[0]


def _normalized_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object mapping names to values")
    return {str(key).strip().lower(): item for key, item in value.items()}


def _canonical_numeric_mapping(
    raw: Any,
    *,
    field: str,
    defaults: Mapping[str, float],
    positive: bool = False,
) -> dict[str, float]:
    normalized = _normalized_mapping(raw, field=field)
    unknown = sorted(set(normalized) - set(defaults))
    if unknown:
        raise ValueError(f"{field} contains unsupported keys: {unknown}")
    result: dict[str, float] = {}
    for name, default in defaults.items():
        parsed = _finite_float(normalized.get(name, default), field=f"{field}.{name}")
        if positive and parsed <= 0.0:
            raise ValueError(f"{field}.{name} must be positive")
        result[str(name)] = parsed
    return result


def canonicalize_ieee14_operating_point(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expand a sparse IEEE-14 operating point into one replayable schema.

    ``bus_load_scales`` are profile factors multiplied by ``load_scale`` when
    applied. Generator dispatch and voltage values are absolute setpoints.
    """
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("operating point must be an object")
    op = dict(raw or {})
    load_scale = _finite_float(op.get("load_scale", 1.0), field="load_scale")
    if load_scale <= 0.0:
        raise ValueError("load_scale must be positive")

    bus_load_scales = _canonical_numeric_mapping(
        op.get("bus_load_scales"),
        field="bus_load_scales",
        defaults={name: 1.0 for name in IEEE14_LOAD_BASE_KW},
        positive=True,
    )
    dispatch_raw = op.get("generator_dispatch_kw", op.get("generator_dispatch"))
    generator_dispatch = _canonical_numeric_mapping(
        dispatch_raw,
        field="generator_dispatch_kw",
        defaults=IEEE14_GENERATOR_DISPATCH_KW,
    )
    if any(value < 0.0 for value in generator_dispatch.values()):
        raise ValueError("generator_dispatch_kw values must be non-negative")

    voltage_raw = op.get("voltage_setpoints_pu", op.get("voltage_setpoints"))
    voltage_setpoints = _canonical_numeric_mapping(
        voltage_raw,
        field="voltage_setpoints_pu",
        defaults=IEEE14_GENERATOR_VOLTAGE_PU,
        positive=True,
    )
    if any(not 0.8 <= value <= 1.2 for value in voltage_setpoints.values()):
        raise ValueError("voltage_setpoints_pu values must be in [0.8, 1.2]")

    source_voltage = _finite_float(
        op.get("source_voltage_pu", IEEE14_SOURCE_VOLTAGE_PU),
        field="source_voltage_pu",
    )
    if not 0.8 <= source_voltage <= 1.2:
        raise ValueError("source_voltage_pu must be in [0.8, 1.2]")

    return IEEE14OperatingPoint(
        load_scale=load_scale,
        bus_load_scales=bus_load_scales,
        generator_dispatch_kw=generator_dispatch,
        voltage_setpoints_pu=voltage_setpoints,
        source_voltage_pu=source_voltage,
    ).to_dict()


# ------------------------------------------------------------------ OPF dispatch
#: How the generator dispatch and voltage setpoints of an operating point are chosen.
#: ``case14`` keeps the checked-in model values (bus 2 at 40 MW, 1 kW condensers at
#: buses 3/6/8, the file setpoints, source 1.06); ``opf`` takes them from the
#: pypower AC-OPF on case14 at the same per-bus load, the dispatch law of the
#: pypower-generated scenario families.
DISPATCH_MODE_OPF = "opf"
DISPATCH_MODE_CASE14 = "case14"
DISPATCH_MODES = (DISPATCH_MODE_OPF, DISPATCH_MODE_CASE14)
#: case14 generator rows 2-5 (the row-1 unit is the slack, folded into the source).
OPF_DISPATCHED_GENERATORS = ("b2", "b3", "b6", "b8")
OPF_SLACK_BUS = "b1"
OPF_SOLVER = ("pypower.api.runopf(case14, ppoption(VERBOSE=0, OUT_ALL=0)); identical to "
              "psse_env.providers.scenario_generator.Round0ScenarioGenerator._solve_ac_opf and "
              "Transmission.generate_measurements_hif_ieee14._solve_pypower (case14 gencost, "
              "RATE_A 9900 MVA, VMIN/VMAX 0.94/1.06)")
#: Dispatch below this (MW) is the interior-point solver's numerical zero, reported as 0 kW.
_OPF_PG_ZERO_MW = 1e-6


class OPFDispatchError(RuntimeError):
    """The AC-OPF gave no valid dispatch for an operating point; callers skip it, never substitute."""


@dataclass(frozen=True)
class OPFOperatingPoint:
    """A canonical operating point taken from one AC-OPF solution."""

    op_point: dict[str, Any]
    receipt: dict[str, Any]
    solution: dict[str, Any] = field(repr=False, compare=False)


def scaled_case14(load_scale: float, bus_load_scales: Mapping[str, float] | None = None) -> dict[str, Any]:
    """pypower case14 with PD/QD scaled per bus by ``load_scale * bus_load_scales[bus]``.

    This is the load law of :func:`apply_hif_operating_point` (every OpenDSS load
    at bus ``bN`` is the case14 load at bus N), so the OPF sees the same loads
    that the OpenDSS solve will carry.
    """
    from pypower.api import case14
    from pypower.idx_bus import BUS_I, PD, QD

    canonical = canonicalize_ieee14_operating_point(
        {"load_scale": load_scale, "bus_load_scales": bus_load_scales})
    ppc = case14()
    for row in ppc["bus"]:
        bus = f"b{int(row[BUS_I])}"
        factor = canonical["load_scale"] * canonical["bus_load_scales"].get(bus, 1.0)
        row[PD] *= factor
        row[QD] *= factor
    return ppc


def solve_ieee14_opf(load_scale: float, bus_load_scales: Mapping[str, float] | None = None) -> dict[str, Any] | None:
    """AC-OPF of case14 at the given per-bus load; ``None`` when the solver does not converge."""
    from pypower.api import ppoption, runopf

    result = runopf(deepcopy(scaled_case14(load_scale, bus_load_scales)), ppoption(VERBOSE=0, OUT_ALL=0))
    return result if result.get("success") else None


def operating_point_from_opf_solution(
    solution: Mapping[str, Any],
    *,
    load_scale: float,
    bus_load_scales: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Canonical operating point carrying the OPF dispatch and voltages.

    ``generator_dispatch_kw`` is the OPF active output of the units at buses 2,
    3, 6 and 8; ``voltage_setpoints_pu`` their OPF bus voltage magnitudes;
    ``source_voltage_pu`` the OPF slack-bus (bus 1) voltage magnitude, so the
    OpenDSS Vsource reproduces the slack and absorbs its active and reactive
    power. The loads are the ones the OPF was solved with.
    """
    from pypower.idx_bus import BUS_I, VM
    from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMIN

    bus_vm = {f"b{int(row[BUS_I])}": float(row[VM]) for row in solution["bus"]}
    dispatch: dict[str, float] = {}
    for row in solution["gen"]:
        bus = f"b{int(row[GEN_BUS])}"
        if bus == OPF_SLACK_BUS or float(row[GEN_STATUS]) <= 0.0:
            continue
        if bus not in OPF_DISPATCHED_GENERATORS:
            raise OPFDispatchError(f"OPF solution dispatches an unexpected unit at {bus}")
        pg_mw = float(row[PG])
        if not math.isfinite(pg_mw) or pg_mw < min(float(row[PMIN]), 0.0) - _OPF_PG_ZERO_MW:
            raise OPFDispatchError(f"OPF active output {pg_mw} MW at {bus} is invalid")
        dispatch[bus] = 0.0 if pg_mw < _OPF_PG_ZERO_MW else pg_mw * 1000.0
    missing = sorted(set(OPF_DISPATCHED_GENERATORS) - set(dispatch))
    if missing:
        raise OPFDispatchError(f"OPF solution has no in-service unit at {missing}")
    try:
        return canonicalize_ieee14_operating_point({
            "load_scale": float(load_scale),
            "bus_load_scales": dict(bus_load_scales or {}),
            "generator_dispatch_kw": dispatch,
            "voltage_setpoints_pu": {bus: bus_vm[bus] for bus in OPF_DISPATCHED_GENERATORS},
            "source_voltage_pu": bus_vm[OPF_SLACK_BUS],
        })
    except ValueError as exc:
        raise OPFDispatchError(f"OPF solution is not a valid operating point: {exc}") from exc


def opf_dispatch_receipt(solution: Mapping[str, Any]) -> dict[str, Any]:
    """JSON annotation of one OPF solution (mode, solver, slack and reactive outputs)."""
    from pypower.idx_gen import GEN_BUS, PG, QG

    slack = next(row for row in solution["gen"] if f"b{int(row[GEN_BUS])}" == OPF_SLACK_BUS)
    return {
        "mode": DISPATCH_MODE_OPF,
        "solver": OPF_SOLVER,
        "objective": float(solution["f"]),
        "slack_bus": OPF_SLACK_BUS,
        "slack_pg_kw": float(slack[PG]) * 1000.0,
        "slack_qg_kvar": float(slack[QG]) * 1000.0,
        "generator_qg_kvar": {f"b{int(row[GEN_BUS])}": float(row[QG]) * 1000.0
                              for row in solution["gen"] if f"b{int(row[GEN_BUS])}" != OPF_SLACK_BUS},
        "slack_folded_into": "OpenDSS Vsource pu = OPF bus-1 voltage magnitude; the source supplies the slack P and Q",
    }


def ieee14_opf_operating_point(
    load_scale: float, bus_load_scales: Mapping[str, float] | None = None
) -> OPFOperatingPoint:
    """Solve the AC-OPF at the given loads and return the canonical operating point.

    Raises :class:`OPFDispatchError` when the OPF does not converge; the caller
    records the failure and skips the window or scan. The case14 dispatch is
    never substituted.
    """
    solution = solve_ieee14_opf(load_scale, bus_load_scales)
    if solution is None:
        raise OPFDispatchError(
            f"AC-OPF did not converge at load_scale={float(load_scale):.6f} "
            f"bus_load_scales={dict(bus_load_scales or {})}")
    return OPFOperatingPoint(
        op_point=operating_point_from_opf_solution(solution, load_scale=load_scale, bus_load_scales=bus_load_scales),
        receipt=opf_dispatch_receipt(solution),
        solution=solution,
    )


def case14_dispatch_receipt() -> dict[str, Any]:
    return {"mode": DISPATCH_MODE_CASE14,
            "solver": None,
            "note": "checked-in IEEE14Gen.DSS dispatch and setpoints (bus 2 at 40 MW, 1 kW condensers at 3/6/8, source 1.06)"}


# ------------------------------------------------------------ OpenDSS application
def _first_bus_name() -> str:
    import opendssdirect as dss  # type: ignore

    names = dss.CktElement.BusNames() or []
    return normalize_bus_name(names[0]) if names else ""


def _generator_reactive_limits() -> tuple[float, float]:
    """Reactive limits (kvar) of the active generator."""
    import opendssdirect as dss  # type: ignore

    return float(dss.Properties.Value("maxkvar")), float(dss.Properties.Value("minkvar"))


def capture_operating_point_baseline() -> dict[str, Any]:
    """Capture enabled non-HIF loads, generators, and source setpoints."""
    import opendssdirect as dss  # type: ignore

    loads: list[dict[str, Any]] = []
    for name in dss.Loads.AllNames() or []:
        dss.Loads.Name(name)
        lower = str(name).strip().lower()
        if lower.startswith(("hif_", "hifest", "hif_est")):
            continue
        if hasattr(dss.CktElement, "Enabled") and not bool(dss.CktElement.Enabled()):
            continue
        loads.append(
            {
                "name": str(name),
                "key": lower,
                "bus": _first_bus_name(),
                "kw": float(dss.Loads.kW()),
                "kvar": float(dss.Loads.kvar()),
            }
        )

    generators: list[dict[str, Any]] = []
    for name in dss.Generators.AllNames() or []:
        dss.Generators.Name(name)
        if hasattr(dss.CktElement, "Enabled") and not bool(dss.CktElement.Enabled()):
            continue
        try:
            vpu = float(dss.Properties.Value("vpu"))
        except Exception:
            vpu = math.nan
        maxkvar, minkvar = _generator_reactive_limits()
        generators.append(
            {
                "name": str(name),
                "key": str(name).strip().lower(),
                "bus": _first_bus_name(),
                "kw": float(dss.Generators.kW()),
                "vpu": vpu,
                "maxkvar": maxkvar,
                "minkvar": minkvar,
            }
        )

    sources: list[dict[str, Any]] = []
    for name in dss.Vsources.AllNames() or []:
        dss.Vsources.Name(name)
        sources.append(
            {
                "name": str(name),
                "key": str(name).strip().lower(),
                "pu": float(dss.Vsources.PU()),
            }
        )
    return {"loads": loads, "generators": generators, "sources": sources}


def _mapping_value(mapping: Mapping[str, Any], *, name: str, bus: str) -> Any:
    for key in (str(name).strip().lower(), normalize_bus_name(bus)):
        if key and key in mapping:
            return mapping[key]
    return None


def apply_ieee14_dispatch_and_setpoints(
    baseline: Mapping[str, Any],
    op_point: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write the generator dispatch, voltage setpoints and source voltage of an operating point.

    Loads are untouched; :func:`apply_hif_operating_point` scales them first.
    The unbalance generator, whose loads are scaled and split by its own code,
    uses this entry point directly. The model's reactive limits are restored
    after every dispatch write.
    """
    import opendssdirect as dss  # type: ignore

    op = canonicalize_ieee14_operating_point(op_point)
    dispatch_kw = op["generator_dispatch_kw"]
    voltage_setpoints = op["voltage_setpoints_pu"]

    applied_dispatch: dict[str, float] = {}
    applied_voltage_setpoints: dict[str, float] = {}
    applied_reactive_limits: dict[str, dict[str, float]] = {}
    for item in baseline.get("generators", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item["name"])
        bus = normalize_bus_name(item.get("bus"))
        raw_kw = _mapping_value(dispatch_kw, name=name, bus=bus)
        if isinstance(raw_kw, Mapping):
            raw_kw = raw_kw.get("kw")
        if raw_kw is not None:
            kw = _finite_float(raw_kw, field=f"generator dispatch for {name}")
        else:
            kw = float(item["kw"])
        if kw < 0.0:
            raise ValueError(f"generator dispatch for {name} must be non-negative")
        dss.Generators.Name(name)
        # Writing kW makes OpenDSS recompute maxkvar/minkvar from the nominal
        # power factor (+-1.08*kW at the default 0.88), which pins the 1 kW
        # synchronous condensers at about 1 kvar and stops voltage regulation.
        # Keep the model's own limits: the baseline's, or the live ones.
        if "maxkvar" in item and "minkvar" in item:
            maxkvar, minkvar = float(item["maxkvar"]), float(item["minkvar"])
        else:
            maxkvar, minkvar = _generator_reactive_limits()
        try:
            dss.Generators.kW(kw)
            applied_dispatch[name] = kw

            raw_vpu = _mapping_value(voltage_setpoints, name=name, bus=bus)
            if raw_vpu is not None or math.isfinite(float(item.get("vpu", math.nan))):
                vpu = (
                    _finite_float(raw_vpu, field=f"voltage setpoint for {name}")
                    if raw_vpu is not None
                    else float(item["vpu"])
                )
                if not 0.8 <= vpu <= 1.2:
                    raise ValueError(f"voltage setpoint for {name} must be in [0.8, 1.2] pu")
                dss.Text.Command(f"Edit Generator.{name} Vpu={vpu:.12g}")
                applied_voltage_setpoints[name] = vpu
        finally:
            dss.Text.Command(f"Edit Generator.{name} Maxkvar={maxkvar:.12g} Minkvar={minkvar:.12g}")
            applied_reactive_limits[name] = {"maxkvar": maxkvar, "minkvar": minkvar}

    source_pu = float(op["source_voltage_pu"])
    applied_source: dict[str, float] = {}
    for item in baseline.get("sources", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item["name"])
        source_value = (
            _finite_float(source_pu, field="source_voltage_pu")
            if source_pu is not None
            else float(item["pu"])
        )
        if not 0.8 <= source_value <= 1.2:
            raise ValueError("source_voltage_pu must be in [0.8, 1.2]")
        dss.Text.Command(f"Edit Vsource.{name} pu={source_value:.12g}")
        applied_source[name] = source_value

    return {
        "generator_dispatch_kw": applied_dispatch,
        "voltage_setpoints_pu": applied_voltage_setpoints,
        "reactive_limits_kvar": applied_reactive_limits,
        "source_voltage_pu": applied_source,
    }


def apply_hif_operating_point(
    baseline: Mapping[str, Any],
    op_point: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply a replayable IEEE-14 operating point to the active OpenDSS circuit.

    ``bus_load_scales`` and ``load_scales`` are multipliers relative to the
    checked-in base model. ``generator_dispatch_kw`` values are absolute kW,
    and ``voltage_setpoints_pu`` values are absolute per-unit setpoints.
    """
    import opendssdirect as dss  # type: ignore

    op = canonicalize_ieee14_operating_point(op_point)
    global_scale = float(op["load_scale"])
    bus_load_scales = op["bus_load_scales"]

    applied_load_scales: dict[str, float] = {}
    for item in baseline.get("loads", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item["name"])
        bus = normalize_bus_name(item.get("bus"))
        profile_scale = _mapping_value(bus_load_scales, name=name, bus=bus)
        if profile_scale is None:
            raise ValueError(f"No canonical load profile factor for {name} at {bus}")
        scale = global_scale * _finite_float(profile_scale, field=f"load scale for {name}")
        if scale <= 0.0:
            raise ValueError(f"load scale for {name} must be positive")
        dss.Loads.Name(name)
        dss.Loads.kW(float(item["kw"]) * scale)
        dss.Loads.kvar(float(item["kvar"]) * scale)
        applied_load_scales[name] = scale

    applied = apply_ieee14_dispatch_and_setpoints(baseline, op)
    return {"load_scales": applied_load_scales, **applied}


__all__ = [
    "DISPATCH_MODES",
    "DISPATCH_MODE_CASE14",
    "DISPATCH_MODE_OPF",
    "IEEE14OperatingPoint",
    "OPFDispatchError",
    "OPFOperatingPoint",
    "OPF_DISPATCHED_GENERATORS",
    "OPF_SLACK_BUS",
    "OPF_SOLVER",
    "apply_hif_operating_point",
    "apply_ieee14_dispatch_and_setpoints",
    "canonicalize_ieee14_operating_point",
    "capture_operating_point_baseline",
    "case14_dispatch_receipt",
    "ieee14_opf_operating_point",
    "normalize_bus_name",
    "operating_point_from_opf_solution",
    "opf_dispatch_receipt",
    "scaled_case14",
    "solve_ieee14_opf",
]
