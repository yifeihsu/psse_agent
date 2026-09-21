"""Mid-span HIF injection into the normalized IEEE-14 OpenDSS model.

Units contract
--------------
The checked-in ``IEEE_14_OpenDSS`` model is *normalized*: every bus is 1 kV
line-to-line on a 100 MVA base, so its impedance base is
``MODEL_ZBASE_OHM = 0.01`` ohm everywhere.  Every ``kv_ll`` argument in this
module (``hif_ohms_from_pu``, ``constant_impedance_hif_kw``,
``inject_midspan_hif_ieee14``) is therefore the **model** base (1.0 kV) and
must never be a physical bus voltage: the fault load is written with
``kV = kv_ll/sqrt(3)`` inside a 1 kV network, so ``kv_ll=69`` would place a
39.8 kV load in a 0.577 kV node and draw ~4761x the intended power.  A guard
raises ``ValueError`` for any non-model ``kv_ll`` unless the caller passes
``allow_non_model_kv=True`` (only for a model that really uses that base).

``r_hif_ohm`` everywhere in this module means **normalized-model ohms**
(``R_model_ohm = R_pu * 0.01``).  Physical ohms at the faulted line's actual
voltage (69 / 13.8 / 18 kV) are converted by ``three_phase_nlm.hif_units``
on the line's local base (``R_pu = R_ohm / (kV_local**2 / 100 MVA)``);
``inject_midspan_hif_ieee14`` accepts them through the ``resistance_ohm`` /
``kv_ll_local`` keywords and performs that conversion itself.
"""
from __future__ import annotations

import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ieee14_adapter import branch_info_from_dss_element, normalize_dss_element


IEEE14_DSS_FILES = [
    "Run_IEEE14Bus.dss",
    "IEEE14BusMaster.dss",
    "IEEE14Lines.DSS",
    "IEEE14Trafo.DSS",
    "IEEE14Loads.DSS",
    "IEEE14Cap.DSS",
    "IEEE14Gen.DSS",
]


@dataclass(frozen=True)
class HIFInjectionResult:
    model_dir: str
    dss_element: str
    branch_row0: int | None
    line_index1: int | None
    from_bus: int | None
    to_bus: int | None
    split_ratio: float
    phase: str
    phase_number: int
    fault_bus: str
    hif_load_name: str
    #: Fault resistance in NORMALIZED-MODEL ohms (R_pu * 0.01): the value the
    #: DSS load and the legacy NLM bridge see.  Never physical ohms.
    r_hif_ohm: float
    #: Model line-to-neutral kV of the fault load (kv_ll/sqrt(3) = 0.57735).
    kv_ln: float
    p_kw: float
    line_a: str
    line_b: str
    #: Same value as ``r_hif_ohm`` under its explicit name.
    r_hif_model_ohm: float | None = None
    #: Physical ohms on the line's local voltage base (None for legacy calls).
    resistance_ohm: float | None = None
    #: Local line-to-line kV used for the ohm <-> pu conversion (None for legacy).
    local_kv_ll: float | None = None
    #: Per-unit resistance; base-invariant (local base == normalized model base).
    r_hif_pu: float | None = None
    #: One of the ``three_phase_nlm.hif_units.BASIS_*`` strings.
    resistance_basis: str | None = None
    #: ``"ohm_local_base"`` (physical input) or ``"pu_legacy_normalized_model"``.
    resistance_units: str | None = None

    @property
    def branch_element_overrides(self) -> dict[str, dict[str, Any]]:
        return {
            self.dss_element: {
                "from": self.line_a,
                "from_terminal": 0,
                "to": self.line_b,
                "to_terminal": 1,
            }
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branch_element_overrides"] = self.branch_element_overrides
        return payload


def _require_model_kv(kv_ll: float, allow_non_model_kv: bool, function: str) -> float:
    """Reject physical bus voltages: the injector works in the 1 kV model base."""
    from .hif_units import MODEL_KV_LL  # lazy: hif_units imports the adapter

    value = float(kv_ll)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{function}: kv_ll must be a finite positive model base voltage")
    if allow_non_model_kv or math.isclose(value, MODEL_KV_LL, rel_tol=1e-9, abs_tol=1e-12):
        return value
    raise ValueError(
        f"{function}: kv_ll={value!r} is not the normalized IEEE-14 model base "
        f"({MODEL_KV_LL} kV). Physical bus voltages must never reach the injector: "
        "convert physical ohms with three_phase_nlm.hif_units "
        "(model_ohm_from_physical_ohm / hif_resistance_record) and pass normalized-model "
        "ohms, or use inject_midspan_hif_ieee14(resistance_ohm=..., kv_ll_local=...). "
        "Pass allow_non_model_kv=True only for a model that really uses that base."
    )


def hif_ohms_from_pu(
    r_hif_pu: float,
    *,
    base_mva: float = 100.0,
    kv_ll: float = 1.0,
    allow_non_model_kv: bool = False,
) -> float:
    """Per-unit resistance -> ohms the normalized DSS model needs (``R_pu * 0.01``).

    ``kv_ll`` is the MODEL base (1 kV), never a physical bus voltage; the
    per-unit value is base-invariant, so a local-base pu computed by
    ``three_phase_nlm.hif_units`` converts here without any voltage argument.
    """
    kv = _require_model_kv(kv_ll, allow_non_model_kv, "hif_ohms_from_pu")
    zbase_ohm = (kv * 1000.0) ** 2 / (float(base_mva) * 1e6)
    return float(r_hif_pu) * zbase_ohm


def constant_impedance_hif_kw(
    r_hif_ohm: float, *, kv_ll: float = 1.0, allow_non_model_kv: bool = False
) -> tuple[float, float]:
    """(kv_ln, p_kw) tokens of a constant-impedance (Model=2) fault load.

    ``r_hif_ohm`` is in normalized-model ohms and ``kv_ll`` is the MODEL base
    (1 kV): ``Z = kV_LN**2 / kW`` reproduces ``r_hif_ohm`` exactly in the
    model's own ohms.  Never pass a physical bus voltage here (see module doc).
    """
    if float(r_hif_ohm) <= 0:
        raise ValueError("r_hif_ohm must be positive")
    kv_ll = _require_model_kv(kv_ll, allow_non_model_kv, "constant_impedance_hif_kw")
    kv_ln = float(kv_ll) / math.sqrt(3.0)
    v_ln_volts = kv_ln * 1000.0
    p_kw = (v_ln_volts**2) / float(r_hif_ohm) / 1000.0
    return kv_ln, p_kw


def copy_ieee14_model(source_dir: str | Path, scenario_dir: str | Path, *, overwrite: bool = False) -> Path:
    src = Path(source_dir).resolve()
    dst = Path(scenario_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"IEEE-14 OpenDSS source directory not found: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Scenario directory already exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for filename in IEEE14_DSS_FILES:
        source_file = src / filename
        if not source_file.exists():
            raise FileNotFoundError(f"Required OpenDSS file missing: {source_file}")
        shutil.copy2(source_file, dst / filename)
    return dst


def write_balanced_ieee14_load_override(model_dir: str | Path, *, encoding: str = "utf-8") -> None:
    """
    Restore Bus 3 to the balanced IEEE-14 load in a copied scenario model.

    The checked-in OpenDSS model is currently used for imbalance studies and has
    B3A/B3B/B3C split loads. HIF samples need a clean balanced base unless a
    caller intentionally combines error families.
    """
    loads_path = Path(model_dir) / "IEEE14Loads.DSS"
    if not loads_path.exists():
        raise FileNotFoundError(f"IEEE14Loads.DSS not found in {model_dir}")
    with loads_path.open("a", encoding=encoding) as handle:
        handle.write("\n! ------------- Balanced-load override for HIF scenarios -----------\n")
        handle.write("Edit Load.B3A enabled=no\n")
        handle.write("Edit Load.B3B enabled=no\n")
        handle.write("Edit Load.B3C enabled=no\n")
        handle.write(
            "New Load.__HIF_BAL_B3 Bus1=B3 kV=1 kW=94200 kvar=19000 "
            "vmaxpu=1.06 vminpu=0.94\n"
        )
        handle.write("! ---------------------------------------------------------------\n")


def _phase_number(phase: str) -> int:
    normalized = str(phase).strip().upper()
    mapping = {"A": 1, "B": 2, "C": 3, "1": 1, "2": 2, "3": 3}
    if normalized not in mapping:
        raise ValueError(f"phase must be A/B/C or 1/2/3, got {phase!r}")
    return mapping[normalized]


def _token_key(token: str) -> str | None:
    if "=" not in token:
        return None
    return token.split("=", 1)[0].strip().lower()


def _parse_line_tokens(line: str) -> tuple[list[str], dict[str, str]]:
    tokens = line.strip().split()
    kv = {}
    for token in tokens[2:]:
        key = _token_key(token)
        if key:
            kv[key] = token.split("=", 1)[1]
    return tokens, kv


def _render_replacement_line(
    *,
    original_tokens: list[str],
    new_element: str,
    bus1: str,
    bus2: str,
    length: float,
) -> str:
    rendered = ["New", new_element]
    seen = set()
    overrides = {"bus1": bus1, "bus2": bus2, "length": f"{float(length):.12g}"}
    for token in original_tokens[2:]:
        key = _token_key(token)
        if not key:
            rendered.append(token)
            continue
        lhs = token.split("=", 1)[0]
        value = overrides.get(key, token.split("=", 1)[1])
        rendered.append(f"{lhs}={value}")
        seen.add(key)
    for key in ("bus1", "bus2", "length"):
        if key not in seen:
            rendered.append(f"{key}={overrides[key]}")
    return " ".join(rendered)


def _line_matcher(dss_element: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*new\s+{re.escape(dss_element)}\b.*$", re.IGNORECASE)


def _safe_fault_suffix(dss_element: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", dss_element.split(".", 1)[1]).strip("_")


def _resolve_injection_resistance(
    element: str,
    branch_row0: int | None,
    *,
    r_hif_ohm: float | None,
    resistance_ohm: float | None,
    kv_ll_local: float | None,
) -> dict[str, Any]:
    """Normalized-model ohms plus a self-describing units record."""
    from . import hif_units  # lazy: hif_units imports the adapter

    if (r_hif_ohm is None) == (resistance_ohm is None):
        raise ValueError(
            "Supply exactly one of r_hif_ohm (normalized-model ohms) or "
            "resistance_ohm (physical ohms on the line's local voltage base)"
        )
    if r_hif_ohm is not None:
        if kv_ll_local is not None:
            raise ValueError("kv_ll_local only accompanies resistance_ohm, not r_hif_ohm")
        model_ohm = float(r_hif_ohm)
        if not math.isfinite(model_ohm) or model_ohm <= 0:
            raise ValueError("r_hif_ohm must be positive")
        return {
            "r_hif_model_ohm": model_ohm,
            "resistance_ohm": None,
            "local_kv_ll": None,
            "r_hif_pu": model_ohm / hif_units.MODEL_ZBASE_OHM,
            "resistance_basis": hif_units.BASIS_NORMALIZED_MODEL,
            "resistance_units": hif_units.RESISTANCE_UNITS_PU_LEGACY,
        }
    physical = float(resistance_ohm)
    if not math.isfinite(physical) or physical <= 0:
        raise ValueError("resistance_ohm must be positive")
    if branch_row0 is not None:
        base = hif_units.resolve_line_kv_ll(int(branch_row0), kv_ll_local)
        local_kv_ll = float(base["kv_ll"])
        basis = str(base["resistance_basis"])
    elif kv_ll_local is not None:
        local_kv_ll = float(kv_ll_local)
        basis = hif_units.BASIS_EXPLICIT
    else:
        raise ValueError(
            f"{element!r} is not a BRANCH_ORDER row; pass kv_ll_local with resistance_ohm"
        )
    return {
        "r_hif_model_ohm": hif_units.model_ohm_from_physical_ohm(physical, local_kv_ll),
        "resistance_ohm": physical,
        "local_kv_ll": local_kv_ll,
        "r_hif_pu": hif_units.local_pu_from_ohm(physical, local_kv_ll),
        "resistance_basis": basis,
        "resistance_units": hif_units.RESISTANCE_UNITS_OHM_LOCAL_BASE,
    }


def inject_midspan_hif_ieee14(
    model_dir: str | Path,
    dss_element: str,
    *,
    split_ratio: float,
    phase: str,
    r_hif_ohm: float | None = None,
    base_mva: float = 100.0,
    kv_ll: float = 1.0,
    fault_bus: str | None = None,
    hif_load_name: str | None = None,
    encoding: str = "utf-8",
    resistance_ohm: float | None = None,
    kv_ll_local: float | None = None,
    allow_non_model_kv: bool = False,
) -> HIFInjectionResult:
    """Split ``dss_element`` at ``split_ratio`` and hang a resistive HIF load there.

    Resistance is given in exactly one of two ways:

    * ``r_hif_ohm`` -- NORMALIZED-MODEL ohms (``R_pu * 0.01``), the legacy path.
    * ``resistance_ohm`` -- PHYSICAL ohms at the faulted line's actual voltage,
      optionally with ``kv_ll_local`` (line-to-line kV).  When ``kv_ll_local`` is
      None the line's base is looked up with
      ``three_phase_nlm.hif_units.resolve_line_kv_ll`` (cross-voltage Line.7-8
      uses its from-bus base and is flagged).  The physical value is converted
      with ``hif_units.model_ohm_from_physical_ohm`` before anything touches
      the DSS file.

    ``kv_ll`` is the MODEL base (1 kV) and is guarded; it is never the
    physical voltage (see the module docstring).  The result's ``r_hif_ohm``
    is always model ohms; the physical value, if any, is ``resistance_ohm``.
    """
    del base_mva
    ratio = float(split_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")
    element = normalize_dss_element(dss_element)
    if not element.lower().startswith("line."):
        raise ValueError(f"Only Line.* HIF targets are supported initially, got {element}")

    model_path = Path(model_dir).resolve()
    lines_path = model_path / "IEEE14Lines.DSS"
    if not lines_path.exists():
        raise FileNotFoundError(f"IEEE14Lines.DSS not found in {model_path}")

    lines = lines_path.read_text(encoding=encoding, errors="replace").splitlines()
    matcher = _line_matcher(element)
    matched_line = None
    for line in lines:
        if matcher.match(line):
            matched_line = line
            break
    if matched_line is None:
        raise ValueError(f"Line {element!r} not found in {lines_path}")

    tokens, kv = _parse_line_tokens(matched_line)
    try:
        length = float(kv["length"])
        bus1 = kv["bus1"]
        bus2 = kv["bus2"]
    except KeyError as exc:
        raise ValueError(f"Line {element!r} is missing required token {exc.args[0]!r}") from exc

    suffix = _safe_fault_suffix(element)
    fault_name = fault_bus or f"Fault_{suffix}"
    load_name = hif_load_name or f"Load.HIF_{suffix}"
    line_a = f"{element}_hif_a"
    line_b = f"{element}_hif_b"
    phase_no = _phase_number(phase)
    info = branch_info_from_dss_element(element)
    resistance = _resolve_injection_resistance(
        element,
        info["branch_row0"],
        r_hif_ohm=r_hif_ohm,
        resistance_ohm=resistance_ohm,
        kv_ll_local=kv_ll_local,
    )
    r_model_ohm = float(resistance["r_hif_model_ohm"])
    kv_ln, p_kw = constant_impedance_hif_kw(
        r_model_ohm, kv_ll=kv_ll, allow_non_model_kv=allow_non_model_kv
    )

    len_a = length * ratio
    len_b = length - len_a
    block = [
        "",
        "! ------------- IEEE-14 midspan HIF auto-generated block -----------",
        f"Edit {element} enabled=no",
        _render_replacement_line(
            original_tokens=tokens,
            new_element=line_a,
            bus1=bus1,
            bus2=f"{fault_name}.1.2.3",
            length=len_a,
        ),
        _render_replacement_line(
            original_tokens=tokens,
            new_element=line_b,
            bus1=f"{fault_name}.1.2.3",
            bus2=bus2,
            length=len_b,
        ),
        (
            f"New {load_name} Bus1={fault_name}.{phase_no} Phases=1 Conn=Wye "
            f"Model=2 Status=Fixed kV={kv_ln:.12g} kW={p_kw:.12g} kvar=0"
        ),
        "! ---------------------------------------------------------------",
    ]
    lines_path.write_text("\n".join(lines + block) + "\n", encoding=encoding)

    return HIFInjectionResult(
        model_dir=str(model_path),
        dss_element=element,
        branch_row0=info["branch_row0"],
        line_index1=info["line_index1"],
        from_bus=info["from_bus"],
        to_bus=info["to_bus"],
        split_ratio=ratio,
        phase=str(phase).strip().upper(),
        phase_number=phase_no,
        fault_bus=fault_name,
        hif_load_name=load_name,
        r_hif_ohm=r_model_ohm,
        kv_ln=float(kv_ln),
        p_kw=float(p_kw),
        line_a=line_a,
        line_b=line_b,
        r_hif_model_ohm=r_model_ohm,
        resistance_ohm=resistance["resistance_ohm"],
        local_kv_ll=resistance["local_kv_ll"],
        r_hif_pu=float(resistance["r_hif_pu"]),
        resistance_basis=resistance["resistance_basis"],
        resistance_units=resistance["resistance_units"],
    )
