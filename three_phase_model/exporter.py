"""Export a positive-sequence-preserving, explicit OpenDSS snapshot model."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_bus import BUS_I, BUS_TYPE, REF, PD, QD, GS, BS, BASE_KV, VM, VA
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_assumptions(value: str | Path = "normalized_diagonal") -> dict[str, Any]:
    path = Path(value)
    if not path.is_file():
        path = Path(__file__).parent / "specs" / f"{value}.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate_assumptions(spec)
    return spec


def validate_assumptions(spec: Mapping[str, Any]) -> None:
    expected = {
        "schema": "three_phase_assumptions_v1",
        "voltage_realization": "normalized_uniform_not_equipment_ratings",
        "negative_sequence": "equal_to_positive_sequence",
        "transformer_connection": "grounded_wye_grounded_wye",
        "generator_control": "solved_positive_sequence_snapshot_constant_pq_per_phase",
        "source_boundary": "fixed_thevenin_emf_compensated_at_reference_snapshot",
        "harmonics": "fundamental_only_spectrum_no_validated_harmonic_extension",
        "faults": "not_in_baseline",
        "grounding": "solid_phase_neutrals_to_reference_ground_no_explicit_neutral_conductor",
    }
    numeric_keys = {"base_mva", "base_kv_ll", "frequency_hz", "line_zero_sequence_r_ratio",
                    "line_zero_sequence_x_ratio", "line_zero_sequence_c_ratio"}
    allowed = set(expected) | numeric_keys | {
        "name", "provenance", "constant_pq_voltage_range", "source_z1_pu", "source_z2_pu", "source_z0_pu",
        "transformer_magnetizing_percent", "transformer_antifloat_ppm",
    }
    unknown = set(spec) - allowed
    missing = allowed - set(spec)
    if unknown or missing:
        raise ValueError(f"Unknown or missing assumption fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Unsupported three-phase assumption {key}: {spec.get(key)!r}")
    for key in numeric_keys:
        value = float(spec[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("transformer_magnetizing_percent", "transformer_antifloat_ppm"):
        if float(spec[key]) != 0:
            raise ValueError(f"{key} must be zero for this equivalence model")
    lo, hi = map(float, spec["constant_pq_voltage_range"])
    if not 0 < lo < 1 < hi or not math.isfinite(hi):
        raise ValueError("constant_pq_voltage_range must bracket 1 pu")
    for key in ("source_z1_pu", "source_z2_pu", "source_z0_pu"):
        values = list(map(float, spec[key]))
        if len(values) != 2 or not np.isfinite(values).all() or min(values) < 0 or max(values) <= 0:
            raise ValueError(f"{key} must specify a finite nonzero passive impedance")


def phase_matrix(positive: float, zero_ratio: float) -> np.ndarray:
    """A diag(z0,z1,z1) A^-1 for a reciprocal transposed completion."""
    return np.eye(3) * positive + np.ones((3, 3)) * positive * (zero_ratio - 1.0) / 3.0


def _matrix(values: np.ndarray) -> str:
    return "[" + " | ".join(" ".join(f"{value:.16g}" for value in row[:i + 1])
                              for i, row in enumerate(values)) + "]"


def _serial_case(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in case.items() if key in {"version", "baseMVA", "bus", "gen", "branch", "gencost"}}


def export_model(case: Mapping[str, Any], output_dir: str | Path, *,
                 assumptions: Mapping[str, Any] | None = None,
                 case_id: str = "custom", source_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Solve the canonical positive-sequence case, then export its PQ snapshot.

    The source matrices are never mutated. Cases with shifts, islands, invalid
    branches, or a failed reference solve are rejected before any model is written.
    """
    spec = copy.deepcopy(dict(load_assumptions() if assumptions is None else assumptions))
    validate_assumptions(spec)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", case_id):
        raise ValueError("case_id must be an alphanumeric OpenDSS name beginning with a letter")
    json.dumps(dict(source_provenance or {}), allow_nan=False)
    source = copy.deepcopy(dict(case))
    for name in ("bus", "gen", "branch"):
        source[name] = np.asarray(source[name], dtype=float).copy()
        minimum_columns = {"bus": 13, "gen": 10, "branch": 13}[name]
        if (source[name].ndim != 2 or not len(source[name]) or source[name].shape[1] < minimum_columns
            or not np.isfinite(source[name]).all()):
            raise ValueError(f"Invalid shape or nonfinite source {name}")
    base = float(source["baseMVA"])
    if not math.isfinite(base) or base <= 0 or base != float(spec["base_mva"]):
        raise ValueError("Assumptions must use the source three-phase MVA base")
    buses = source["bus"]
    ids = buses[:, BUS_I].astype(int)
    if len(set(ids)) != len(ids) or not np.array_equal(ids, buses[:, BUS_I]) or np.any(ids <= 0):
        raise ValueError("Bus IDs must be distinct positive integers")
    refs = ids[buses[:, BUS_TYPE] == REF]
    if len(refs) != 1 or not np.isin(buses[:, BUS_TYPE], [1, 2, 3]).all():
        raise ValueError("This realization requires one slack and no isolated buses")
    if not np.isin(source["branch"][:, BR_STATUS], [0, 1]).all():
        raise ValueError("Branch status must be exactly 0 or 1")
    if not np.isin(source["gen"][:, GEN_STATUS], [0, 1]).all():
        raise ValueError("Generator status must be exactly 0 or 1")
    if not np.isin(source["branch"][:, [F_BUS, T_BUS]], ids).all():
        raise ValueError("Every branch endpoint must identify an existing bus")
    if not np.isin(source["gen"][:, GEN_BUS], ids).all():
        raise ValueError("Every generator must identify an existing bus")
    active_gen_buses = set(source["gen"][source["gen"][:, GEN_STATUS] == 1, GEN_BUS])
    if not set(ids[np.isin(buses[:, BUS_TYPE], [2, 3])]).issubset(active_gen_buses):
        raise ValueError("Every PV and slack bus must have an active generator")
    reached = {int(refs[0])}
    edges = [(int(row[F_BUS]), int(row[T_BUS])) for row in source["branch"] if row[BR_STATUS] == 1]
    while True:
        expanded = reached | {t for f, t in edges if f in reached} | {f for f, t in edges if t in reached}
        if expanded == reached:
            break
        reached = expanded
    if reached != set(ids):
        raise ValueError("Active branches must form one connected network")
    if np.any(source["branch"][:, SHIFT] != 0):
        raise ValueError("Phase-shifting transformer conversion is not supported")
    if np.any(source["branch"][:, BR_R] < 0) or np.any(source["branch"][:, BR_X] <= 0):
        raise ValueError("Branches require nonnegative R and positive X")
    if np.any(source["branch"][:, BR_B] < 0) or np.any(source["branch"][:, TAP] < 0):
        raise ValueError("Negative charging or tap ratios are not supported")
    if np.any(buses[:, GS] < 0):
        raise ValueError("Negative shunt conductance is not supported")
    source_hash = hashlib.sha256(json.dumps(_serial_case(source), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    normalized = copy.deepcopy(source)
    kv = float(spec["base_kv_ll"])
    normalized["bus"][:, BASE_KV] = kv
    reference, success = runpf(normalized, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11, PF_MAX_IT=100, ENFORCE_Q_LIMS=0))
    if not success:
        raise ValueError("Positive-sequence reference power flow did not converge")
    lo, hi = map(float, spec["constant_pq_voltage_range"])
    if np.any(reference["bus"][:, VM] <= lo) or np.any(reference["bus"][:, VM] >= hi):
        raise ValueError("Reference voltage is outside the declared constant-PQ envelope")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    zbase = kv * kv / base
    freq = float(spec["frequency_hz"])
    registry: dict[str, Any] = {
        "schema": "three_phase_asset_registry_v1", "case_id": case_id,
        "base_case_hash": source_hash, "buses": [], "branches": [],
        "loads": [], "generators": [], "shunts": [],
    }
    for i, bus in enumerate(buses):
        registry["buses"].append({"row0": i, "external_bus": int(bus[BUS_I]),
                                  "dss_bus": f"b{int(bus[BUS_I])}", "kv_ll": kv,
                                  "source_base_kv": float(bus[BASE_KV]), "phases": [1, 2, 3]})
    files: dict[str, list[str]] = {name: ["! Generated normalized snapshot; see assumptions.json and asset_registry.json."]
                                  for name in ("Lines.dss", "Transformers.dss", "Loads.dss", "Generators.dss", "Shunts.dss")}
    circuits: dict[tuple[int, int], int] = {}
    for i, branch in enumerate(source["branch"]):
        f, t = int(branch[F_BUS]), int(branch[T_BUS])
        pair = tuple(sorted((f, t)))
        circuits[pair] = circuits.get(pair, 0) + 1
        kind = "Transformer" if branch[TAP] != 0 else "Line"
        name = f"br_{i+1:04d}"
        status = int(branch[BR_STATUS])
        enabled = "yes" if status else "no"
        item = {"asset_id": f"{case_id}:branch:{i+1}", "branch_row0": i,
                "from_bus": f, "to_bus": t, "dss_element": f"{kind}.{name}",
                "from_terminal": 1, "to_terminal": 2, "status": status,
                "circuit_ordinal": circuits[pair], "tap": float(branch[TAP] or 1.0),
                "source_tap": float(branch[TAP]), "r_pu": float(branch[BR_R]),
                "x_pu": float(branch[BR_X]), "b_pu": float(branch[BR_B]),
                "charging_elements": {"from": [], "to": []}}
        if kind == "Line":
            r = phase_matrix(branch[BR_R] * zbase, spec["line_zero_sequence_r_ratio"])
            x = phase_matrix(branch[BR_X] * zbase, spec["line_zero_sequence_x_ratio"])
            c = phase_matrix(branch[BR_B] / (2 * math.pi * freq * zbase) * 1e9, spec["line_zero_sequence_c_ratio"])
            files["Lines.dss"].append(
                f"New Line.{name} Phases=3 Bus1=b{f}.1.2.3 Bus2=b{t}.1.2.3 Length=1 Units=none BaseFreq={freq:.16g} "
                f"Rmatrix={_matrix(r)} Xmatrix={_matrix(x)} Cmatrix={_matrix(c)} Rg=0 Xg=0 Enabled={enabled}")
            item.update(rmatrix_ohm=r.tolist(), xmatrix_ohm=x.tolist(), cmatrix_nf=c.tolist())
        else:
            files["Transformers.dss"].append(
                f"New Transformer.{name} Phases=3 Windings=2 Buses=[b{f}.1.2.3.0 b{t}.1.2.3.0] "
                f"Conns=[wye wye] kVs=[{kv:.16g} {kv:.16g}] kVAs=[{base*1000:.16g} {base*1000:.16g}] "
                f"XHL={branch[BR_X]*100:.16g} %Rs=[{branch[BR_R]*50:.16g} {branch[BR_R]*50:.16g}] "
                f"Taps=[{branch[TAP]:.16g} 1] %NoLoadLoss=0 %IMag=0 ppm_Antifloat=0 "
                f"Wdg=1 RNeut=0 XNeut=0 Wdg=2 RNeut=0 XNeut=0 BaseFreq={freq:.16g} Enabled={enabled}")
            # Preserve any transformer charging as endpoint admittances. The
            # source case57 has none, but this avoids a silent generic loss.
            for side, bus_id, ratio in (("from", f, float(branch[TAP]) ** 2), ("to", t, 1.0)):
                if branch[BR_B] != 0:
                    element = f"Capacitor.{name}_{side}_charging"
                    kvar = float(branch[BR_B]) * base * 1000 / (2 * ratio)
                    files["Shunts.dss"].append(f"New {element} Phases=3 Bus1=b{bus_id}.1.2.3 Bus2=b{bus_id}.0.0.0 Conn=wye kV={kv:.16g} kvar={kvar:.16g} BaseFreq={freq:.16g} Enabled={enabled}")
                    item["charging_elements"][side].append(element)
        registry["branches"].append(item)
    for bus in buses:
        number = int(bus[BUS_I])
        if bus[PD] != 0 or bus[QD] != 0:
            for phase, letter in enumerate("abc", 1):
                element = f"Load.ld_{number:03d}_{letter}"
                kw, kvar = float(bus[PD])*1000/3, float(bus[QD])*1000/3
                files["Loads.dss"].append(f"New {element} Phases=1 Bus1=b{number}.{phase}.0 Conn=wye kV={kv/math.sqrt(3):.16g} kW={kw:.16g} kvar={kvar:.16g} Model=1 Status=fixed Vminpu={lo:.16g} Vmaxpu={hi:.16g} Vlowpu={lo/2:.16g} Spectrum=fundamental_only BaseFreq={freq:.16g}")
                registry["loads"].append({"bus": number, "phase": phase, "element": element, "kw": kw, "kvar": kvar})
        if bus[BS] != 0:
            kind = "Capacitor" if bus[BS] > 0 else "Reactor"
            element = f"{kind}.bs_{number:03d}"
            files["Shunts.dss"].append(f"New {element} Phases=3 Bus1=b{number}.1.2.3 Bus2=b{number}.0.0.0 Conn=wye kV={kv:.16g} kvar={abs(bus[BS])*1000:.16g} BaseFreq={freq:.16g}")
            registry["shunts"].append({"bus": number, "element": element, "gs_mw": 0.0, "bs_mvar": float(bus[BS])})
        if bus[GS] > 0:
            element = f"Reactor.gs_{number:03d}"
            files["Shunts.dss"].append(f"New {element} Phases=3 Bus1=b{number}.1.2.3 Bus2=b{number}.0.0.0 R={kv*kv/bus[GS]:.16g} X=0 BaseFreq={freq:.16g}")
            registry["shunts"].append({"bus": number, "element": element, "gs_mw": float(bus[GS]), "bs_mvar": 0.0})
    slack = int(refs[0])
    slack_gen_rows = []
    for i, gen in enumerate(reference["gen"]):
        if gen[GEN_STATUS] <= 0:
            continue
        number = int(gen[GEN_BUS])
        if number == slack:
            slack_gen_rows.append(i)
            continue
        for phase, letter in enumerate("abc", 1):
            element = f"Generator.gen_{i+1:03d}_{letter}"
            kw, kvar = float(gen[PG])*1000/3, float(gen[QG])*1000/3
            files["Generators.dss"].append(f"New {element} Phases=1 Bus1=b{number}.{phase}.0 Conn=wye kV={kv/math.sqrt(3):.16g} kW={kw:.16g} kvar={kvar:.16g} Model=1 Status=fixed Vminpu={lo:.16g} Vmaxpu={hi:.16g} Spectrum=fundamental_only BaseFreq={freq:.16g}")
            registry["generators"].append({"gen_row0": i, "bus": number, "phase": phase, "element": element, "kw": kw, "kvar": kvar})
    if not slack_gen_rows:
        raise ValueError("No active slack generator")
    slack_bus = reference["bus"][np.where(reference["bus"][:, BUS_I] == slack)[0][0]]
    v = slack_bus[VM] * np.exp(1j * np.deg2rad(slack_bus[VA]))
    slack_s = sum((complex(reference["gen"][i, PG], reference["gen"][i, QG]) for i in slack_gen_rows), 0j) / base
    emf = v + complex(*spec["source_z1_pu"]) * np.conj(slack_s / v)
    registry["source"] = {"element": "Vsource.source", "bus": slack, "gen_rows0": slack_gen_rows,
                           "emf_pu": [float(emf.real), float(emf.imag)], "target_voltage_pu": [float(v.real), float(v.imag)],
                           "reference_slack_power_pu": [slack_s.real, slack_s.imag]}
    ztext = " ".join(f"Z{seq}=[{complex(*spec[f'source_z{seq}_pu']).real*zbase:.16g} {complex(*spec[f'source_z{seq}_pu']).imag*zbase:.16g}]" for seq in (1, 2, 0))
    master = ["! Normalized three-phase research model. All physical quantities follow assumptions.json.",
              "Clear", f"Set DefaultBaseFrequency={freq:.16g}",
              f"New Circuit.{case_id}_3p Bus1=b{slack}.1.2.3 Bus2=b{slack}.0.0.0 Phases=3 BasekV={kv:.16g} pu={abs(emf):.16g} Angle={np.rad2deg(np.angle(emf)):.16g} BaseMVA={base:.16g} {ztext} Model=Thevenin",
              "New Spectrum.fundamental_only NumHarm=1 Harmonic=[1] %Mag=[100] Angle=[0]",
              "Edit Vsource.source Spectrum=fundamental_only",
              *[f"Redirect {name}" for name in files],
              f"Set VoltageBases=[{kv:.16g}]", "CalcVoltageBases", "Set ControlMode=off",
              "Set Mode=snapshot Algorithm=Newton LoadModel=Powerflow MaxIterations=1000 Tolerance=1e-12",
              "Solve"]
    (out / "Master.dss").write_text("\n".join(master) + "\n", encoding="utf-8")
    for name, lines in files.items():
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "assumptions.json", spec)
    write_json(out / "asset_registry.json", registry)
    write_json(out / "positive_sequence_reference.json", _serial_case(reference))
    write_json(out / "source_case.json", _serial_case(source))
    manifest = {"schema": "three_phase_model_build_v1", "case_id": case_id,
                "source_provenance": dict(source_provenance or {}), "base_case_hash": source_hash,
                "external_bus_count": len(buses), "external_phase_node_count": 3*len(buses),
                "physical_branch_count": len(registry["branches"]),
                "line_count": sum(row["dss_element"].startswith("Line.") for row in registry["branches"]),
                "transformer_count": sum(row["dss_element"].startswith("Transformer.") for row in registry["branches"]),
                "snapshot_equivalence_only": True, "pv_control_equivalence": False,
                "reference_q_limits_enforced": False, "validation_performed": False,
                "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())}}
    write_json(out / "build_manifest.json", manifest)
    return {"output_dir": str(out), "reference": reference, "registry": registry, "assumptions": spec, "manifest": manifest}
