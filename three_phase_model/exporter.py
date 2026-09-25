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
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, QMAX, QMIN, VG
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS

# Generator realizations. The default keeps every non-slack generator at its
# solved snapshot P/Q on each phase (the IEEE57 contract). The opt-in PV form is
# one three-phase OpenDSS Model=3 element per generator: fixed P, equal Q on the
# three phases, Q adjusted to hold the average phase-to-neutral magnitude at the
# generator setpoint VG and clamped to [QMIN, QMAX]. Its reference solve enforces
# the same limits (the slack stays an unlimited Thevenin source).
CONSTANT_PQ_GENERATORS = "solved_positive_sequence_snapshot_constant_pq_per_phase"
PV_GENERATORS = "voltage_regulated_average_phase_magnitude_with_reactive_limits"
GENERATOR_CONTROLS = {"constant_pq": CONSTANT_PQ_GENERATORS, "pv_q_limits": PV_GENERATORS}


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
        "generator_control": CONSTANT_PQ_GENERATORS,
        "source_boundary": "fixed_thevenin_emf_compensated_at_reference_snapshot",
        "harmonics": "fundamental_only_spectrum_no_validated_harmonic_extension",
        "faults": "not_in_baseline",
        "grounding": "solid_phase_neutrals_to_reference_ground_no_explicit_neutral_conductor",
    }
    numeric_keys = {"base_mva", "base_kv_ll", "frequency_hz", "line_zero_sequence_r_ratio",
                    "line_zero_sequence_x_ratio", "line_zero_sequence_c_ratio"}
    required = set(expected) | numeric_keys | {
        "name", "provenance", "constant_pq_voltage_range", "source_z1_pu", "source_z2_pu", "source_z0_pu",
        "transformer_magnetizing_percent", "transformer_antifloat_ppm",
    }
    allowed = required | {"voltage_profile", "bus_base_kv_ll"}
    unknown = set(spec) - allowed
    missing = required - set(spec)
    if unknown or missing:
        raise ValueError(f"Unknown or missing assumption fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
    for key, value in expected.items():
        if key == "voltage_realization" and spec.get(key) == "declared_per_bus_nominal_voltage":
            continue
        if key == "generator_control" and spec.get(key) == PV_GENERATORS:
            continue
        if spec.get(key) != value:
            raise ValueError(f"Unsupported three-phase assumption {key}: {spec.get(key)!r}")
    if spec.get("voltage_realization") == "declared_per_bus_nominal_voltage":
        voltage_map = spec.get("bus_base_kv_ll")
        if not isinstance(voltage_map, Mapping) or not voltage_map:
            raise ValueError("per-bus voltage realization requires bus_base_kv_ll")
        keys = []
        for key, value in voltage_map.items():
            if isinstance(key, bool) or not str(key).isdigit() or int(key) <= 0:
                raise ValueError("voltage-map bus IDs must be positive integers")
            keys.append(int(key))
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("every bus voltage base must be finite and positive")
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate bus IDs in voltage map")
        if not isinstance(spec.get("voltage_profile"), str) or not spec["voltage_profile"]:
            raise ValueError("per-bus voltage realization requires a profile identity")
    elif "bus_base_kv_ll" in spec or "voltage_profile" in spec:
        raise ValueError("per-bus voltage fields require a declared per-bus realization")
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


def solve_reference(case: Mapping[str, Any], generator_control: str = CONSTANT_PQ_GENERATORS) -> dict[str, Any]:
    """Positive-sequence reference; PV realizations enforce reactive limits.

    Limits are enforced by the usual outer loop around an unlimited Newton PF:
    a violating generator is fixed at its limit (its bus becomes PQ), and a
    fixed generator whose voltage sits on the wrong side of its setpoint is
    released, until the set is stable. The result satisfies complementarity:
    V = VG inside the limits, V <= VG at QMAX, V >= VG at QMIN. The slack
    generator is never limited. Bus types of the returned case are the source's.
    """
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11, PF_MAX_IT=100, ENFORCE_Q_LIMS=0)
    if generator_control == CONSTANT_PQ_GENERATORS:
        reference, success = runpf(case, options)
        if not success:
            raise ValueError("Positive-sequence reference power flow did not converge")
        reference["reactive_limited_gen_rows0"] = {}
        return reference
    if generator_control != PV_GENERATORS:
        raise ValueError(f"Unsupported generator control: {generator_control!r}")
    work = copy.deepcopy(dict(case))
    types = np.asarray(work["bus"][:, BUS_TYPE]).copy()
    row = {int(number): i for i, number in enumerate(work["bus"][:, BUS_I])}
    slack = int(work["bus"][types == REF, BUS_I][0])
    active = [i for i, gen in enumerate(work["gen"]) if gen[GEN_STATUS] > 0 and int(gen[GEN_BUS]) != slack]
    shared = [int(bus) for bus, count in zip(*np.unique([work["gen"][i, GEN_BUS] for i in active], return_counts=True)) if count > 1]
    if shared:
        raise ValueError(f"Reactive-limit enforcement needs one active generator per PV bus: {shared}")
    if any(work["gen"][i, QMIN] > work["gen"][i, QMAX] for i in active):
        raise ValueError("Every regulated generator needs QMIN <= QMAX")
    limited: dict[int, str] = {}
    for _ in range(50):
        trial = copy.deepcopy(work)
        for i, side in limited.items():
            trial["gen"][i, QG] = trial["gen"][i, QMAX if side == "max" else QMIN]
            trial["bus"][row[int(trial["gen"][i, GEN_BUS])], BUS_TYPE] = 1
        reference, success = runpf(trial, options)
        if not success:
            raise ValueError("Reactive-limited positive-sequence reference did not converge")
        changed = False
        for i in active:
            gen = reference["gen"][i]
            voltage = reference["bus"][row[int(gen[GEN_BUS])], VM]
            if i in limited:
                if (limited[i] == "max" and voltage > gen[VG] + 1e-9) or (limited[i] == "min" and voltage < gen[VG] - 1e-9):
                    del limited[i]
                    changed = True
            elif gen[QG] > gen[QMAX] + 1e-7:
                limited[i], changed = "max", True
            elif gen[QG] < gen[QMIN] - 1e-7:
                limited[i], changed = "min", True
        if not changed:
            break
    else:
        raise ValueError("Reactive-limit set did not settle")
    reference["bus"][:, BUS_TYPE] = types
    reference["reactive_limited_gen_rows0"] = {int(i): side for i, side in sorted(limited.items())}
    return reference


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
                 case_id: str = "custom", source_provenance: Mapping[str, Any] | None = None,
                 voltage_profile: str | None = None, generator_control: str | None = None) -> dict[str, Any]:
    """Solve the canonical positive-sequence case, then export its PQ snapshot.

    The source matrices are never mutated. Cases with shifts, islands, invalid
    branches, or a failed reference solve are rejected before any model is written.
    Uniform normalized voltage remains the default. ``voltage_profile`` opts
    into the selected IEEE14, IEEE57 or IEEE118 bases; ``assumptions.bus_base_kv_ll`` permits
    an explicit complete map. Per-unit branch parameters/taps are preserved.
    With a bus map, the scalar ``base_kv_ll`` names the slack/source base only.
    ``generator_control`` ("constant_pq" or "pv_q_limits") overrides the
    assumptions' generator realization; see GENERATOR_CONTROLS.
    """
    spec = copy.deepcopy(dict(load_assumptions() if assumptions is None else assumptions))
    if generator_control is not None:
        if generator_control not in GENERATOR_CONTROLS:
            raise ValueError(f"generator_control must be one of {sorted(GENERATOR_CONTROLS)}")
        spec["generator_control"] = GENERATOR_CONTROLS[generator_control]
    if voltage_profile is not None:
        from .voltage_bases import get_voltage_base_profile
        profile_metadata = get_voltage_base_profile(voltage_profile)
        expected_map = {str(bus): float(value) for bus, value in profile_metadata["bus_base_kv_ll"].items()}
        if spec.get("bus_base_kv_ll") is not None and {
            str(key): float(value) for key, value in spec["bus_base_kv_ll"].items()
        } != expected_map:
            raise ValueError("voltage profile conflicts with assumptions bus voltage bases")
        spec.update(voltage_profile=voltage_profile, bus_base_kv_ll=expected_map,
                    voltage_realization="declared_per_bus_nominal_voltage")
    elif "bus_base_kv_ll" in spec:
        spec["voltage_realization"] = "declared_per_bus_nominal_voltage"
        spec.setdefault("voltage_profile", "explicit_bus_voltage_bases_v1")
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
    bus_kv = ({int(bus): float(value) for bus, value in spec["bus_base_kv_ll"].items()}
              if "bus_base_kv_ll" in spec else {int(bus): float(spec["base_kv_ll"]) for bus in ids})
    if set(bus_kv) != set(ids):
        raise ValueError("voltage profile must cover exactly the source case bus IDs")
    if "bus_base_kv_ll" in spec:
        spec["bus_base_kv_ll"] = {str(number): value for number, value in sorted(bus_kv.items())}
    slack = int(refs[0])
    kv = bus_kv[slack]
    # In a multivoltage model this scalar is the slack/source reference only.
    # All bus and branch conversions use the explicit local map below.
    spec["base_kv_ll"] = kv
    normalized["bus"][:, BASE_KV] = [bus_kv[int(number)] for number in ids]
    regulated = spec["generator_control"] == PV_GENERATORS
    reference = solve_reference(normalized, spec["generator_control"])
    limited = reference.pop("reactive_limited_gen_rows0")
    lo, hi = map(float, spec["constant_pq_voltage_range"])
    if np.any(reference["bus"][:, VM] <= lo) or np.any(reference["bus"][:, VM] >= hi):
        raise ValueError("Reference voltage is outside the declared constant-PQ envelope")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_zbase = kv * kv / base
    freq = float(spec["frequency_hz"])
    registry: dict[str, Any] = {
        "schema": "three_phase_asset_registry_v1", "case_id": case_id,
        "base_case_hash": source_hash, "buses": [], "branches": [],
        "loads": [], "generators": [], "shunts": [],
    }
    for i, bus in enumerate(buses):
        registry["buses"].append({"row0": i, "external_bus": int(bus[BUS_I]),
                                  "dss_bus": f"b{int(bus[BUS_I])}", "kv_ll": bus_kv[int(bus[BUS_I])],
                                  "source_base_kv": float(bus[BASE_KV]), "phases": [1, 2, 3]})
    files: dict[str, list[str]] = {name: ["! Generated normalized snapshot; see assumptions.json and asset_registry.json."]
                                  for name in ("Lines.dss", "Transformers.dss", "Loads.dss", "Generators.dss", "Shunts.dss")}
    circuits: dict[tuple[int, int], int] = {}
    for i, branch in enumerate(source["branch"]):
        f, t = int(branch[F_BUS]), int(branch[T_BUS])
        from_kv, to_kv = bus_kv[f], bus_kv[t]
        zbase = from_kv * from_kv / base
        native_tap = float(branch[TAP] or 1.0)
        pair = tuple(sorted((f, t)))
        circuits[pair] = circuits.get(pair, 0) + 1
        cross_voltage = not math.isclose(from_kv, to_kv, rel_tol=1e-12, abs_tol=0)
        kind = "Transformer" if branch[TAP] != 0 or cross_voltage else "Line"
        name = f"br_{i+1:04d}"
        status = int(branch[BR_STATUS])
        enabled = "yes" if status else "no"
        item = {"asset_id": f"{case_id}:branch:{i+1}", "branch_row0": i,
                "from_bus": f, "to_bus": t, "dss_element": f"{kind}.{name}",
                "from_terminal": 1, "to_terminal": 2, "status": status,
                "circuit_ordinal": circuits[pair], "tap": native_tap,
                "source_tap": float(branch[TAP]), "r_pu": float(branch[BR_R]),
                "x_pu": float(branch[BR_X]), "b_pu": float(branch[BR_B]),
                "from_kv_ll": from_kv, "to_kv_ll": to_kv,
                "from_zbase_ohm": zbase, "to_zbase_ohm": to_kv * to_kv / base,
                "voltage_conversion_required": cross_voltage,
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
                f"Conns=[wye wye] kVs=[{from_kv:.16g} {to_kv:.16g}] kVAs=[{base*1000:.16g} {base*1000:.16g}] "
                f"XHL={branch[BR_X]*100:.16g} %Rs=[{branch[BR_R]*50:.16g} {branch[BR_R]*50:.16g}] "
                f"Taps=[{native_tap:.16g} 1] %NoLoadLoss=0 %IMag=0 ppm_Antifloat=0 "
                f"Wdg=1 RNeut=0 XNeut=0 Wdg=2 RNeut=0 XNeut=0 BaseFreq={freq:.16g} Enabled={enabled}")
            # Preserve any transformer charging as endpoint admittances. The
            # source case57 has none; case118's zero-tap cross-voltage 86-87
            # and 68-116 branches carry line charging under physical bases.
            for side, bus_id, ratio in (("from", f, native_tap ** 2), ("to", t, 1.0)):
                if branch[BR_B] != 0:
                    element = f"Capacitor.{name}_{side}_charging"
                    kvar = float(branch[BR_B]) * base * 1000 / (2 * ratio)
                    files["Shunts.dss"].append(f"New {element} Phases=3 Bus1=b{bus_id}.1.2.3 Bus2=b{bus_id}.0.0.0 Conn=wye kV={bus_kv[bus_id]:.16g} kvar={kvar:.16g} BaseFreq={freq:.16g} Enabled={enabled}")
                    item["charging_elements"][side].append(element)
        registry["branches"].append(item)
    for bus in buses:
        number = int(bus[BUS_I])
        local_kv = bus_kv[number]
        if bus[PD] != 0 or bus[QD] != 0:
            for phase, letter in enumerate("abc", 1):
                element = f"Load.ld_{number:03d}_{letter}"
                kw, kvar = float(bus[PD])*1000/3, float(bus[QD])*1000/3
                files["Loads.dss"].append(f"New {element} Phases=1 Bus1=b{number}.{phase}.0 Conn=wye kV={local_kv/math.sqrt(3):.16g} kW={kw:.16g} kvar={kvar:.16g} Model=1 Status=fixed Vminpu={lo:.16g} Vmaxpu={hi:.16g} Vlowpu={lo/2:.16g} Spectrum=fundamental_only BaseFreq={freq:.16g}")
                registry["loads"].append({"bus": number, "phase": phase, "element": element, "kw": kw, "kvar": kvar, "kv_ll": local_kv})
        if bus[BS] != 0:
            kind = "Capacitor" if bus[BS] > 0 else "Reactor"
            element = f"{kind}.bs_{number:03d}"
            files["Shunts.dss"].append(f"New {element} Phases=3 Bus1=b{number}.1.2.3 Bus2=b{number}.0.0.0 Conn=wye kV={local_kv:.16g} kvar={abs(bus[BS])*1000:.16g} BaseFreq={freq:.16g}")
            registry["shunts"].append({"bus": number, "element": element, "gs_mw": 0.0, "bs_mvar": float(bus[BS])})
        if bus[GS] > 0:
            element = f"Reactor.gs_{number:03d}"
            files["Shunts.dss"].append(f"New {element} Phases=3 Bus1=b{number}.1.2.3 Bus2=b{number}.0.0.0 R={local_kv*local_kv/bus[GS]:.16g} X=0 BaseFreq={freq:.16g}")
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
        if regulated:
            # kW and kvar come before Maxkvar/Minkvar: setting kW re-derives
            # default reactive limits in OpenDSS, so the limits are stated last.
            element = f"Generator.gen_{i+1:03d}"
            kw, kvar = float(gen[PG])*1000, float(gen[QG])*1000
            qmax, qmin = float(gen[QMAX])*1000, float(gen[QMIN])*1000
            files["Generators.dss"].append(f"New {element} Phases=3 Bus1=b{number}.1.2.3 Conn=wye kV={bus_kv[number]:.16g} kW={kw:.16g} kvar={kvar:.16g} Model=3 Vpu={float(gen[VG]):.16g} Maxkvar={qmax:.16g} Minkvar={qmin:.16g} Status=fixed Vminpu={lo:.16g} Vmaxpu={hi:.16g} Spectrum=fundamental_only BaseFreq={freq:.16g}")
            registry["generators"].append({"gen_row0": i, "bus": number, "phases": [1, 2, 3], "element": element,
                                           "kw": kw, "kvar": kvar, "kv_ll": bus_kv[number], "control": "pv",
                                           "vset_pu": float(gen[VG]), "qmax_kvar": qmax, "qmin_kvar": qmin,
                                           "reference_reactive_limit": limited.get(i)})
            continue
        for phase, letter in enumerate("abc", 1):
            element = f"Generator.gen_{i+1:03d}_{letter}"
            kw, kvar = float(gen[PG])*1000/3, float(gen[QG])*1000/3
            files["Generators.dss"].append(f"New {element} Phases=1 Bus1=b{number}.{phase}.0 Conn=wye kV={bus_kv[number]/math.sqrt(3):.16g} kW={kw:.16g} kvar={kvar:.16g} Model=1 Status=fixed Vminpu={lo:.16g} Vmaxpu={hi:.16g} Spectrum=fundamental_only BaseFreq={freq:.16g}")
            registry["generators"].append({"gen_row0": i, "bus": number, "phase": phase, "element": element, "kw": kw, "kvar": kvar, "kv_ll": bus_kv[number]})
    if not slack_gen_rows:
        raise ValueError("No active slack generator")
    slack_bus = reference["bus"][np.where(reference["bus"][:, BUS_I] == slack)[0][0]]
    v = slack_bus[VM] * np.exp(1j * np.deg2rad(slack_bus[VA]))
    slack_s = sum((complex(reference["gen"][i, PG], reference["gen"][i, QG]) for i in slack_gen_rows), 0j) / base
    emf = v + complex(*spec["source_z1_pu"]) * np.conj(slack_s / v)
    registry["source"] = {"element": "Vsource.source", "bus": slack, "gen_rows0": slack_gen_rows,
                           "kv_ll": kv, "zbase_ohm": source_zbase,
                           "emf_pu": [float(emf.real), float(emf.imag)], "target_voltage_pu": [float(v.real), float(v.imag)],
                           "reference_slack_power_pu": [slack_s.real, slack_s.imag]}
    ztext = " ".join(f"Z{seq}=[{complex(*spec[f'source_z{seq}_pu']).real*source_zbase:.16g} {complex(*spec[f'source_z{seq}_pu']).imag*source_zbase:.16g}]" for seq in (1, 2, 0))
    voltage_bases = " ".join(f"{value:.16g}" for value in sorted(set(bus_kv.values())))
    explicit_bus_bases = ([f"SetkVBase Bus=b{number} kVLL={value:.16g}" for number, value in sorted(bus_kv.items())]
                          if "bus_base_kv_ll" in spec else [])
    master = ["! Normalized three-phase research model. All physical quantities follow assumptions.json.",
              "Clear", f"Set DefaultBaseFrequency={freq:.16g}",
              f"New Circuit.{case_id}_3p Bus1=b{slack}.1.2.3 Bus2=b{slack}.0.0.0 Phases=3 BasekV={kv:.16g} pu={abs(emf):.16g} Angle={np.rad2deg(np.angle(emf)):.16g} BaseMVA={base:.16g} {ztext} Model=Thevenin",
              "New Spectrum.fundamental_only NumHarm=1 Harmonic=[1] %Mag=[100] Angle=[0]",
              "Edit Vsource.source Spectrum=fundamental_only",
              *[f"Redirect {name}" for name in files],
              f"Set VoltageBases=[{voltage_bases}]", "CalcVoltageBases", *explicit_bus_bases, "Set ControlMode=off",
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
                "voltage_profile": spec.get("voltage_profile", "legacy_uniform"),
                "bus_base_kv_ll": {str(number): value for number, value in sorted(bus_kv.items())},
                "source_provenance": dict(source_provenance or {}), "base_case_hash": source_hash,
                "external_bus_count": len(buses), "external_phase_node_count": 3*len(buses),
                "physical_branch_count": len(registry["branches"]),
                "line_count": sum(row["dss_element"].startswith("Line.") for row in registry["branches"]),
                "transformer_count": sum(row["dss_element"].startswith("Transformer.") for row in registry["branches"]),
                "snapshot_equivalence_only": not regulated, "pv_control_equivalence": regulated,
                "generator_control": spec["generator_control"],
                "reference_q_limits_enforced": regulated,
                "reference_reactive_limited_gen_rows0": {str(i): side for i, side in limited.items()},
                "validation_performed": False,
                "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())}}
    write_json(out / "build_manifest.json", manifest)
    return {"output_dir": str(out), "reference": reference, "registry": registry, "assumptions": spec, "manifest": manifest}
