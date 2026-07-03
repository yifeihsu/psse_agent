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
    r_hif_ohm: float
    kv_ln: float
    p_kw: float
    line_a: str
    line_b: str

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


def hif_ohms_from_pu(r_hif_pu: float, *, base_mva: float = 100.0, kv_ll: float = 1.0) -> float:
    zbase_ohm = (float(kv_ll) * 1000.0) ** 2 / (float(base_mva) * 1e6)
    return float(r_hif_pu) * zbase_ohm


def constant_impedance_hif_kw(r_hif_ohm: float, *, kv_ll: float = 1.0) -> tuple[float, float]:
    if float(r_hif_ohm) <= 0:
        raise ValueError("r_hif_ohm must be positive")
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


def inject_midspan_hif_ieee14(
    model_dir: str | Path,
    dss_element: str,
    *,
    split_ratio: float,
    phase: str,
    r_hif_ohm: float,
    base_mva: float = 100.0,
    kv_ll: float = 1.0,
    fault_bus: str | None = None,
    hif_load_name: str | None = None,
    encoding: str = "utf-8",
) -> HIFInjectionResult:
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
    kv_ln, p_kw = constant_impedance_hif_kw(r_hif_ohm, kv_ll=kv_ll)

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

    info = branch_info_from_dss_element(element)
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
        r_hif_ohm=float(r_hif_ohm),
        kv_ln=float(kv_ln),
        p_kw=float(p_kw),
        line_a=line_a,
        line_b=line_b,
    )
