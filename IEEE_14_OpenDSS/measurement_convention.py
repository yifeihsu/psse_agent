"""Bus-injection conventions for measurement vectors exported from OpenDSS.

The operator WLS (``mcp_server`` case14) keeps fixed bus shunts (the 19 Mvar
bus-9 capacitor) inside Ybus, so a measured reactive injection must exclude
them: this is the MATPOWER ``makeSbus`` convention (``ybus``). The historical
IEEE-14 exporter instead added capacitor powers to ``Qinj`` (``legacy_injection``),
which made every OpenDSS-derived window disagree with the WLS model at bus 9
regardless of any fault. Regenerated corpora declare their convention so that
estimators simulate candidates under the same convention as the observation.

This module has no OpenDSS dependency and may be imported anywhere.
"""
from __future__ import annotations

from typing import Any, Mapping

SHUNT_CONVENTION_LEGACY = "legacy_injection"
SHUNT_CONVENTION_YBUS = "ybus"
SHUNT_CONVENTIONS = (SHUNT_CONVENTION_LEGACY, SHUNT_CONVENTION_YBUS)
MEASUREMENT_CONVENTION_KEY = "measurement_convention"

_SEMANTICS = {
    SHUNT_CONVENTION_LEGACY: (
        "Capacitor/shunt element powers are added to Pinj/Qinj (historical IEEE-14 exporter). "
        "Disagrees with a WLS model that keeps Bs/Gs in Ybus at bus 9."
    ),
    SHUNT_CONVENTION_YBUS: (
        "Fixed bus shunts stay in Ybus; Pinj/Qinj follow MATPOWER makeSbus "
        "(generation minus demand) and exclude capacitor powers. Matches the operator WLS."
    ),
}


def validate_shunt_convention(value: Any) -> str:
    if not isinstance(value, str) or value not in SHUNT_CONVENTIONS:
        raise ValueError(
            f"shunt_convention must be one of {list(SHUNT_CONVENTIONS)}, got {value!r}"
        )
    return value


def measurement_convention_payload(shunt_convention: str) -> dict[str, Any]:
    """JSON-safe declaration stored on corpus rows, scans and estimator payloads."""
    convention = validate_shunt_convention(shunt_convention)
    return {
        "schema": "ieee14_measurement_convention_v1",
        "shunt_convention": convention,
        "injection_semantics": _SEMANTICS[convention],
        "wls_model_convention": "shunts_in_ybus",
        "consistent_with_operator_wls": convention == SHUNT_CONVENTION_YBUS,
    }


def resolve_shunt_convention(explicit: Any = None, *sources: Any) -> str:
    """Return the first declared convention: explicit, then any source mapping.

    Each source may be a scan/window/row mapping carrying
    ``measurement_convention`` (a payload from :func:`measurement_convention_payload`
    or a bare string). Absent declarations mean the historical convention, so
    tracked corpora keep replaying unchanged.
    """
    if explicit is not None:
        return validate_shunt_convention(explicit)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        declared = source.get(MEASUREMENT_CONVENTION_KEY)
        if isinstance(declared, Mapping):
            declared = declared.get("shunt_convention")
        if declared is not None:
            return validate_shunt_convention(declared)
    return SHUNT_CONVENTION_LEGACY


__all__ = [
    "MEASUREMENT_CONVENTION_KEY",
    "SHUNT_CONVENTIONS",
    "SHUNT_CONVENTION_LEGACY",
    "SHUNT_CONVENTION_YBUS",
    "measurement_convention_payload",
    "resolve_shunt_convention",
    "validate_shunt_convention",
]
