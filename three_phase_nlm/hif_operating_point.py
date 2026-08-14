from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
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


def _first_bus_name() -> str:
    import opendssdirect as dss  # type: ignore

    names = dss.CktElement.BusNames() or []
    return normalize_bus_name(names[0]) if names else ""


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
        generators.append(
            {
                "name": str(name),
                "key": str(name).strip().lower(),
                "bus": _first_bus_name(),
                "kw": float(dss.Generators.kW()),
                "vpu": vpu,
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
    dispatch_kw = op["generator_dispatch_kw"]
    voltage_setpoints = op["voltage_setpoints_pu"]

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

    applied_dispatch: dict[str, float] = {}
    applied_voltage_setpoints: dict[str, float] = {}
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
        "load_scales": applied_load_scales,
        "generator_dispatch_kw": applied_dispatch,
        "voltage_setpoints_pu": applied_voltage_setpoints,
        "source_voltage_pu": applied_source,
    }


__all__ = [
    "IEEE14OperatingPoint",
    "apply_hif_operating_point",
    "canonicalize_ieee14_operating_point",
    "capture_operating_point_baseline",
    "normalize_bus_name",
]
