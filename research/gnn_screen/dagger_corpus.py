"""WLS-screen corpus built from the DAgger physical HIF and unbalance corpora.

Every HIF and unbalance window comes from the corpora the DAgger pipeline trains
on, so the screen and the agent see the same fault draws, simulator, operating
points, measurement convention and sensor sigmas. Each DAgger window is one
physical parent, and its healthy partner is the balanced solve at the same
operating point on the same simulator path:

* HIF parents keep all ten scans of the window. Scan 0 reuses the stored paired
  healthy reference; later scans are re-solved with the fault removed through
  the HIF generator's own path (``three_phase_nlm.hif_operating_point``).
* Unbalance parents keep the stored unbalanced mean and its stored balanced
  reference, both from the unbalance generator's path.

Competing balanced errors are simulated on the parent's own path and operating
point with DAgger's rules: parameter errors (R, X or RX; each factor uniform in
0.1-0.5 or 2-5; in-service lines with nonzero R and X), a dangling line terminal
(one end opened, the reported model unchanged; the effect DAgger uses for its
mixed topology family), and meter errors of 10-15 sigma on one channel type
(one meter, or two to five). Mixed windows carry one meter error and, like
DAgger, never corrupt the faulted branch's own flow meters. No window is
filtered by WLS detectability; DAgger's detectable training subsets are marked
in offline metadata instead.

Physics: the default corpora (2026-09-23opf) keep generator reactive limits and
run at OPF-driven operating points (unit dispatch, PV setpoints and source
voltage from the pypower AC-OPF at the window's loads), which every row stores
in its ``op_point``. HIF parents replay it through ``apply_hif_operating_point``;
unbalance parents apply the uniform load scale and then the stored dispatch
through the unbalance generator's ``_apply_operating_point_dispatch`` (a no-op
for the load-only op_points of the pre-opf corpora). The earlier 2026-09-19/21
HIF corpora were simulated while dispatch writes reset the reactive limits;
every HIF and unbalance parent is checked to reproduce its stored healthy
reference with the current simulator, so a corpus from different physics is
refused instead of mixed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import warnings

import numpy as np

from .dataset import FAMILY_NAMES, MEASUREMENT_CONVENTION, load_manifest

CONTRACT = "gnn_dagger_aligned_corpus_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts" / "measurements"
# 2026-09-23opf corpora (docs/opf_operating_points_20260923.md): same seeds and
# recipes as the 2026-09-19/21/23b corpora, re-simulated with generator reactive
# limits kept (voltage regulation active), the OpenDSS solve tolerance at 1e-8
# and OPF-driven operating points stored in every row's op_point, so the HIF and
# unbalance paths agree and sit on the pypower families' dispatch; the HIF PMU
# phasors are drawn at sigma 1e-4 (the unbalance corpus keeps 5e-3 / 1e-3).
DEFAULT_HIF_CORPORA = (
    "hif_physical69_main_train_84x10_20260923opf",
    "hif_physical69_main_valid_21x10_20260923opf",
    "hif_physical69_main_train_extra_252x10_20260923opf",
    "hif_physical69_main_valid_extra_63x10_20260923opf",
)
DEFAULT_HIF_DETECTABLE = {
    "hif_physical69_main_train_84x10_20260923opf": "hif_physical69_main_train_detectable_27x10_20260923opf",
    "hif_physical69_main_valid_21x10_20260923opf": "hif_physical69_main_valid_detectable_8x10_20260923opf",
    "hif_physical69_main_train_extra_252x10_20260923opf": "hif_physical69_main_train_extra_detectable_77x10_20260923opf",
    "hif_physical69_main_valid_extra_63x10_20260923opf": "hif_physical69_main_valid_extra_detectable_19x10_20260923opf",
}
DEFAULT_UNBALANCE_CORPUS = "out_measurements_imbalance_currents_ybus_440_20260923opf"
DEFAULT_UNBALANCE_DETECTABLE = "out_measurements_imbalance_currents_ybus_detectable_162_20260923opf"
DEFAULT_EVALUATION_HIF = ("hif_physical_sweep_eval_336x10_20260923opf", "hif_physical69_detection_limit_21x10_20260923opf")
SPLIT_FRACTIONS = {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15}
# Noise replicates per noiseless mean. Calibration parents contribute healthy
# windows only. An unbalance parent has one unbalanced mean against ten HIF
# scans, so its unbalanced mean gets extra noise draws.
REPLICATES = {
    "train": {"healthy": 2, "fault": 2, "unbalance": 8},
    "validation": {"healthy": 6, "fault": 2, "unbalance": 4},
    "calibration": {"healthy": 20, "fault": 0, "unbalance": 0},
    "test": {"healthy": 12, "fault": 3, "unbalance": 6},
    "evaluation": {"healthy": 2, "fault": 3, "unbalance": 3},
}
# train.py balances sampling over (family labels, severity); one training
# severity per family combination weights the nine combinations equally, as in
# practical_v2. Other splits carry the descriptive stratum for reporting.
TRAIN_SEVERITY = "dagger_aligned"
PARAMETER_FACTOR_BANDS = ((0.1, 0.5), (2.0, 5.0))
METER_SIGMA_RANGE = (10.0, 15.0)
METER_BLOCKS = (("Vm", 0, 14), ("Pinj", 14, 28), ("Qinj", 28, 42),
                ("Pf", 42, 62), ("Qf", 62, 82), ("Pt", 82, 102), ("Qt", 102, 122))
FLOW_OFFSETS = (42, 62, 82, 102)
# Competing variants per HIF parent, each on a distinct scan.
HIF_PARENT_VARIANTS = {"measurement_single": 1, "measurement_multi": 1, "measurement+hif": 2, "parameter": 2,
                       "topology": 2, "measurement+parameter": 1, "measurement+topology": 1}
UNBALANCE_PARENT_VARIANTS = {"measurement (single or multi, equal odds)": 1, "parameter": 1, "topology": 1}
NB, NL = 14, 20
SOURCE_FIELDS = ("id", "label", "scans", "z_true", "z_clean", "op_point", "three_phase_voltages_clean")


def _stable_int(*parts: Any) -> int:
    return int(hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).hexdigest()[:16], 16)


def fault_rows(name: str, root: Path = ARTIFACTS) -> list[dict[str, Any]]:
    """Fault rows of one DAgger corpus (its pypower no_error rows are skipped), trimmed to used fields."""
    rows = []
    with (Path(root) / name / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("scenario") != "no_error":
                    rows.append({key: row[key] for key in SOURCE_FIELDS if key in row})
    return rows


def vector_hash(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def configured_case14() -> dict[str, Any]:
    """The nominal reported case14 the DAgger WLS uses."""
    from psse_env.systems import resolve_system
    case = resolve_system("case14").load_case()
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in case.items()}


def measurement_sigma() -> np.ndarray:
    """DAgger's declared SCADA sigmas: 0.001 pu Vm, 0.01 pu injections and flows."""
    return np.r_[np.full(NB, 0.001), np.full(2 * NB + 4 * NL, 0.01)]


def parameter_rows(case: dict[str, Any]) -> list[int]:
    """DAgger's rule: in-service lines (no tap, no shift) with nonzero R and X."""
    branch = np.asarray(case["branch"], dtype=float)
    mask = (branch[:, 10] > 0) & (branch[:, 8] == 0) & (branch[:, 9] == 0)
    mask &= (np.abs(branch[:, 2]) > 1e-9) & (np.abs(branch[:, 3]) > 1e-9)
    return [int(i) for i in np.flatnonzero(mask)]


def topology_rows() -> list[int]:
    """Lines whose single-terminal opening leaves every bus energized (7-8 feeds bus 8 radially)."""
    from IEEE_14_OpenDSS.constants import BRANCH_ORDER
    return [i for i, name in enumerate(BRANCH_ORDER) if name.startswith("Line.") and name != "Line.7-8"]


def sample_parameter(rng: np.random.Generator, rows: list[int]) -> dict[str, Any]:
    """Transmission.generate_measurements.apply_parameter_error_oneline with DAgger's factor bands."""
    row0 = int(rng.choice(rows))
    subtype = str(rng.choice(["R", "X", "RX"]))
    factors = {"R": 1.0, "X": 1.0}
    for component in ("R", "X"):
        if component in subtype:
            low, high = PARAMETER_FACTOR_BANDS[int(rng.integers(len(PARAMETER_FACTOR_BANDS)))]
            factors[component] = float(rng.uniform(low, high))
    changed = [factors[c] for c in ("R", "X") if c in subtype]
    band = ("factor_0p1_0p5" if all(f < 1 for f in changed) else "factor_2_5" if all(f > 1 for f in changed)
            else "factor_opposite_directions")
    return {"branch_row0": row0, "subtype": subtype, "r_factor": factors["R"], "x_factor": factors["X"],
            "stratum": f"parameter_{band}"}


def sample_topology(rng: np.random.Generator, rows: list[int]) -> dict[str, Any]:
    return {"branch_row0": int(rng.choice(rows)), "open_terminal": int(rng.integers(1, 3)),
            "effect": "dangling_line_terminal", "reported_status": 1, "stratum": "dangling_terminal"}


def meter_error(z: list[float], rng: np.random.Generator, sigma: np.ndarray, *, count: int | None,
                blocked: set[int] = frozenset()) -> tuple[list[float], dict[str, Any]]:
    """One channel type, one meter (count=1) or two to five (count=None), each 10-15 sigma.

    The distribution of practical_corpus.meter_overlay and of DAgger's pure
    measurement roots after their 10-sigma floor, with optional blocked indices.
    """
    count = int(rng.integers(2, 6)) if count is None else int(count)
    while True:
        channel, start, stop = METER_BLOCKS[int(rng.integers(len(METER_BLOCKS)))]
        eligible = [i for i in range(start, stop) if i not in blocked]
        if len(eligible) >= count:
            break
    indices = np.sort(rng.choice(eligible, size=count, replace=False))
    multiples = rng.uniform(*METER_SIGMA_RANGE, size=count) * rng.choice([-1.0, 1.0], size=count)
    changed = np.asarray(z, dtype=float).copy()
    changed[indices] += multiples * sigma[indices]
    return changed.tolist(), {"measurement_channel": channel, "channel_indices0": indices.tolist(),
                              "sigma_multiples": multiples.tolist(), "meter_count": count}


def branch_flow_indices(row0: int) -> set[int]:
    return {offset + int(row0) for offset in FLOW_OFFSETS}


# ------------------------------------------------------------------ simulators

class _Simulator:
    """One OpenDSS path per process; every solve recompiles the pristine model."""

    def __init__(self, path: str) -> None:
        if path not in ("hif_operating_point", "imbalance_balanced"):
            raise ValueError(f"unknown simulator path {path!r}")
        self.path = path
        self._base_loads = None

    def _prepare(self, op_point: dict[str, Any]) -> None:
        if self.path == "hif_operating_point":
            from three_phase_nlm.hif_parameter_estimator import _compile_base_model, _resolve_model_dir
            from three_phase_nlm.hif_operating_point import apply_hif_operating_point, capture_operating_point_baseline
            _compile_base_model(_resolve_model_dir(str(REPO_ROOT / "IEEE_14_OpenDSS"), "case14"))
            apply_hif_operating_point(capture_operating_point_baseline(), dict(op_point))
        else:
            from Transmission import generate_measurements_imbalance as gi
            gi._compile_ieee14_opendss(str(REPO_ROOT / "IEEE_14_OpenDSS"))
            if self._base_loads is None:
                self._base_loads = gi._read_base_loads()
            gi._scale_all_loads(self._base_loads, float(op_point["load_scale"]))
            # The generator's order: scale the loads, then write the row's stored
            # dispatch (unit kW, PV setpoints, source pu); a load-only op_point
            # (pre-opf corpora) leaves the checked-in model dispatch in place.
            gi._apply_operating_point_dispatch(op_point)

    def solve(self, op_point: dict[str, Any], *, parameter: dict | None = None,
              topology: dict | None = None) -> list[float]:
        import opendssdirect as dss
        from IEEE_14_OpenDSS.constants import BRANCH_ORDER
        from IEEE_14_OpenDSS.export_measurement_series import extract_measurement_series
        from three_phase_nlm.hif_parameter_estimator import _solve_from_fresh_compile, _solve_or_raise

        def build() -> None:
            self._prepare(op_point)
            if parameter:
                element = BRANCH_ORDER[parameter["branch_row0"]]
                if not element.startswith("Line."):
                    raise ValueError("parameter errors are applied to lines only")
                dss.Lines.Name(element.split(".", 1)[1])
                dss.Lines.RMatrix((np.asarray(dss.Lines.RMatrix()) * parameter["r_factor"]).tolist())
                dss.Lines.XMatrix((np.asarray(dss.Lines.XMatrix()) * parameter["x_factor"]).tolist())
            if topology:
                dss.Text.Command(f"Open {BRANCH_ORDER[topology['branch_row0']]} term={int(topology['open_terminal'])}")
            _solve_or_raise()

        _solve_from_fresh_compile(build)
        z = [float(x) for x in extract_measurement_series(shunt_convention="ybus")[0]]
        if len(z) != 3 * NB + 4 * NL or not np.all(np.isfinite(z)):
            raise RuntimeError("unexpected measurement vector")
        return z


_SIMULATORS: dict[str, _Simulator] = {}


def simulator(path: str) -> _Simulator:
    if path not in _SIMULATORS:
        _SIMULATORS[path] = _Simulator(path)
    return _SIMULATORS[path]


def unbalance_operating_point(op_point: dict[str, Any]) -> dict[str, Any]:
    """The unbalance path's replay point: the uniform load scale plus the dispatch the row stores.

    The 2026-09-23opf rows carry the OPF dispatch (``generator_dispatch_kw``,
    ``voltage_setpoints_pu``, ``source_voltage_pu``) beside ``load_scale``; the
    target bus and any per-bus scale are the unbalanced solve's and are dropped.
    """
    from Transmission.generate_measurements_imbalance import DISPATCH_OP_POINT_KEYS
    op = {"load_scale": float(op_point["load_scale"])}
    op.update({key: op_point[key] for key in DISPATCH_OP_POINT_KEYS if key in op_point})
    return op


# ------------------------------------------------------------------ strata

def max_vuf(voltages: list[dict[str, Any]] | None) -> float | None:
    """Largest bus negative/positive-sequence voltage ratio."""
    if not voltages:
        return None
    a = np.exp(2j * np.pi / 3)
    best = 0.0
    for bus in voltages:
        try:
            v = [m * np.exp(1j * np.deg2rad(ang)) for m, ang in zip(bus["vln_pu"], bus["ang_deg"])]
        except (KeyError, TypeError):
            return None
        v1, v2 = (v[0] + a * v[1] + a * a * v[2]) / 3, (v[0] + a * a * v[1] + a * v[2]) / 3
        best = max(best, abs(v2) / abs(v1))
    return best


def vuf_stratum(vuf: float | None) -> str:
    if vuf is None:
        return "unbalance_vuf_unknown"
    pct = 100 * vuf
    return ("unbalance_vuf_below_0p5pct" if pct < 0.5 else "unbalance_vuf_0p5_1pct" if pct < 1
            else "unbalance_vuf_1_2pct" if pct < 2 else "unbalance_vuf_at_least_2pct")


def hif_stratum(label: dict[str, Any]) -> str:
    """Resistance band in ohms on the local base; 1000 ohm closes DAgger's 500-1000 band."""
    ohm = float(label["resistance_ohm"])
    kv = float(label.get("local_kv_ll") or 69.0)
    edges = (0, 100, 200, 500, 1000, 2000, 5000)
    if ohm >= edges[-1]:
        band = f"at_least_{edges[-1]}"
    else:
        i = 3 if ohm == 1000 else max(j for j in range(len(edges) - 1) if ohm >= edges[j])
        band = f"{edges[i]}_{edges[i + 1]}"
    return f"hif_{kv:g}kv_{band}_ohm".replace(".", "p")


# ------------------------------------------------------------------ parents

class _Parent:
    def __init__(self, parent_id: str, family: str, stratum: str, detectable: bool, sigma: np.ndarray,
                 case: dict[str, Any]) -> None:
        self.parent_id, self.family, self.stratum, self.detectable = parent_id, family, stratum, detectable
        self.sigma, self.case = sigma, case
        self.means: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def add(self, *, name: str, families: list[str], z: list[float], stratum: str, role: str,
            metadata: dict[str, Any], paired: dict[str, Any] | None = None,
            affected_phase: str = "none") -> dict[str, Any]:
        from .practical_corpus import noiseless_wls_audit
        z = [float(x) for x in z]
        audit = noiseless_wls_audit(self.case, z, self.sigma)
        extra = {"stratum": stratum, "affected_phase": affected_phase, "noiseless_wls_J": audit.get("J_exact"),
                 "residual_visible_energy_bin": audit["residual_visible_energy_bin"]}
        if paired is not None:
            extra["paired_healthy_window"] = f"{self.parent_id}:{paired['name']}"
            extra["paired_max_abs_sigma"] = float(np.max(np.abs((np.asarray(z) - paired["z"]) / self.sigma)))
        entry = {"name": name, "families": sorted(families, key=FAMILY_NAMES.index), "z": z, "role": role,
                 "metadata": {**metadata, **extra}}
        self.means.append(entry)
        return entry

    def balanced_variant(self, sim: _Simulator, rng: np.random.Generator, *, kind: str, healthy: dict,
                         source: dict, op: dict, with_meter: bool, name: str) -> None:
        spec = (sample_parameter(rng, parameter_rows(self.case)) if kind == "parameter"
                else sample_topology(rng, topology_rows()))
        try:
            z = sim.solve(op, **{kind: spec})
        except RuntimeError as exc:
            self.failures.append({"parent_id": self.parent_id, "variant": name, kind: spec, "error": str(exc)})
            return
        metadata = {**source, kind: spec}
        if with_meter:
            z, info = meter_error(z, rng, self.sigma, count=1, blocked=branch_flow_indices(spec["branch_row0"]))
            self.add(name=name + "+meter", families=[kind, "measurement"], z=z, stratum=spec["stratum"] + "+meter",
                     role="fault", metadata={**metadata, "measurement": info}, paired=healthy)
        else:
            self.add(name=name, families=[kind], z=z, stratum=spec["stratum"], role="fault", metadata=metadata,
                     paired=healthy)

    def result(self) -> dict[str, Any]:
        return {"parent_id": self.parent_id, "family": self.family, "stratum": self.stratum,
                "detectable": self.detectable, "means": self.means, "failures": self.failures}


def build_hif_parent(task: dict[str, Any]) -> dict[str, Any]:
    row, corpus, sigma = task["row"], task["corpus"], np.asarray(task["sigma"], dtype=float)
    parent_id = f"dagger:{corpus}:{row['id']}"
    rng = np.random.default_rng(np.random.SeedSequence([task["seed"], _stable_int(parent_id)]))
    label, sim = row["label"], simulator("hif_operating_point")
    stratum = hif_stratum(label)
    parent = _Parent(parent_id, "hif", stratum, bool(task["detectable"]), sigma, task["case"])
    hif_info = {"branch_row0": int(label["branch_row0"]), "dss_element": label.get("dss_element"),
                "phase": label["phase"], "split_ratio": float(label["split_ratio"]),
                "resistance_ohm": float(label["resistance_ohm"]), "r_hif_pu_local": float(label["r_hif_pu"]),
                "local_kv_ll": float(label.get("local_kv_ll") or 69.0),
                "resistance_band_ohm": label.get("resistance_band_ohm")}
    scan0 = next(scan for scan in row["scans"] if int(scan["scan_index"]) == 0)
    mismatch = float(np.max(np.abs(np.subtract(sim.solve(scan0["op_point"]), row["z_true"]))))
    if mismatch > 1e-9:
        raise ValueError(f"{parent_id}: the current HIF simulator does not reproduce the stored healthy reference "
                         f"(max |dz| {mismatch:.3g} pu); the corpus was generated with different physics")
    base = {"dagger_corpus": corpus, "dagger_source_id": row["id"], "simulator_path": "hif_operating_point",
            "generator_reactive_limits_reset": False, "dagger_detectable": parent.detectable,
            "evaluation_only": bool(task.get("evaluation_only"))}
    healthy, faulted, sources, ops = {}, {}, {}, {}
    for scan in row["scans"]:
        k = int(scan["scan_index"])
        ops[k] = scan["op_point"]
        sources[k] = {**base, "scan_index": k, "operating_point": scan["op_point"]}
        if k == 0:
            z_healthy, how = row["z_true"], "stored_paired_reference"
        else:
            z_healthy, how = sim.solve(scan["op_point"]), "resimulated_fault_removed"
        healthy[k] = parent.add(name=f"s{k}:healthy", families=[], z=z_healthy, stratum="healthy", role="healthy",
                                metadata={**sources[k], "healthy_source": how})
        faulted[k] = parent.add(name=f"s{k}:hif", families=["hif"], z=scan["z_clean"], stratum=stratum,
                                role="fault", metadata={**sources[k], "hif": hif_info}, paired=healthy[k],
                                affected_phase=str(label["phase"]))
    if not task.get("evaluation_only"):
        variants = [name for name, n in HIF_PARENT_VARIANTS.items() for _ in range(n)]
        scans = [int(k) for k in rng.permutation(sorted(healthy))]
        scans = (scans * (len(variants) // len(scans) + 1))[:len(variants)]
        for index, (variant, k) in enumerate(zip(variants, scans)):
            if variant in ("measurement_single", "measurement_multi"):
                count = 1 if variant == "measurement_single" else None
                z, info = meter_error(healthy[k]["z"], rng, sigma, count=count)
                parent.add(name=f"s{k}:meter{index}", families=["measurement"], z=z,
                           stratum="meter_single_10_15_sigma" if count == 1 else "meter_multi_10_15_sigma",
                           role="fault", metadata={**sources[k], "measurement": info}, paired=healthy[k])
            elif variant == "measurement+hif":
                z, info = meter_error(faulted[k]["z"], rng, sigma, count=1)
                parent.add(name=f"s{k}:hif+meter{index}", families=["hif", "measurement"], z=z,
                           stratum=stratum + "+meter", role="fault",
                           metadata={**sources[k], "hif": hif_info, "measurement": info}, paired=healthy[k],
                           affected_phase=str(label["phase"]))
            else:
                kind = variant.split("+")[-1]
                parent.balanced_variant(sim, rng, kind=kind, healthy=healthy[k], source=sources[k], op=ops[k],
                                        with_meter=variant.startswith("measurement+"), name=f"s{k}:{kind}{index}")
    return parent.result()


def build_unbalance_parent(task: dict[str, Any]) -> dict[str, Any]:
    row, corpus, sigma = task["row"], task["corpus"], np.asarray(task["sigma"], dtype=float)
    parent_id = f"dagger:{corpus}:{row['id']}"
    rng = np.random.default_rng(np.random.SeedSequence([task["seed"], _stable_int(parent_id)]))
    label, sim = row["label"], simulator("imbalance_balanced")
    vuf = max_vuf(row.get("three_phase_voltages_clean"))
    stratum = vuf_stratum(vuf)
    parent = _Parent(parent_id, "unbalance", stratum, bool(task["detectable"]), sigma, task["case"])
    op = unbalance_operating_point(row["op_point"])
    mismatch = float(np.max(np.abs(np.subtract(sim.solve(op), row["z_true"]))))
    if mismatch > 1e-9:
        raise ValueError(f"{parent_id}: the current unbalance simulator does not reproduce the stored balanced reference "
                         f"(max |dz| {mismatch:.3g} pu); the corpus was generated with different physics")
    fractions = [float(label["load_split"]["fractions"][p]) for p in ("a", "b", "c")]
    source = {"dagger_corpus": corpus, "dagger_source_id": row["id"], "simulator_path": "imbalance_balanced",
              "generator_reactive_limits_reset": False, "dagger_detectable": parent.detectable,
              "evaluation_only": False, "scan_index": 0, "operating_point": op}
    unbalance = {"bus": int(label["unbalance_bus"]), "fractions": fractions, "distribution": "Dirichlet(3,3,3)",
                 "max_bus_vuf": vuf}
    healthy = parent.add(name="healthy", families=[], z=row["z_true"], stratum="healthy", role="healthy",
                         metadata={**source, "healthy_source": "stored_paired_reference"})
    parent.add(name="unbalance", families=["unbalance"], z=row["z_clean"], stratum=stratum, role="unbalance",
               metadata={**source, "unbalance": unbalance}, paired=healthy,
               affected_phase="fractions_" + "_".join(f"{v:.4g}" for v in fractions))
    count = 1 if rng.random() < 0.5 else None
    z, info = meter_error(healthy["z"], rng, sigma, count=count)
    parent.add(name="meter", families=["measurement"], z=z,
               stratum="meter_single_10_15_sigma" if count == 1 else "meter_multi_10_15_sigma", role="fault",
               metadata={**source, "measurement": info}, paired=healthy)
    for kind in ("parameter", "topology"):
        parent.balanced_variant(sim, rng, kind=kind, healthy=healthy, source=source, op=op, with_meter=False,
                                name=kind)
    return parent.result()


def dispatch(task: dict[str, Any]) -> dict[str, Any]:
    return build_hif_parent(task) if task["kind"] == "hif" else build_unbalance_parent(task)


# ------------------------------------------------------------------ splits and writing

def assign_splits(parents: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Deterministic parent split, stratified by source family, fault stratum and DAgger admission."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for parent in parents:
        groups[(parent["family"], parent["stratum"], parent["detectable"])].append(parent["parent_id"])
    names = list(SPLIT_FRACTIONS)
    cumulative = np.cumsum([SPLIT_FRACTIONS[name] for name in names])
    mapping = {}
    for members in groups.values():
        ordered = sorted(members, key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest())
        offset = _stable_int(seed, ordered[0]) % 1000 / 1000.0
        for index, pid in enumerate(ordered):
            position = ((index + offset) / len(ordered)) % 1.0
            mapping[pid] = names[min(int(np.searchsorted(cumulative, position, side="right")), len(names) - 1)]
    return mapping


def manifest_rows(parent: dict[str, Any], split: str, seed: int, *,
                  replicate_key: str | None = None) -> list[dict[str, Any]]:
    counts = REPLICATES[replicate_key or split]
    rows = []
    for mean in parent["means"]:
        replicates = counts[mean["role"]]
        if replicates == 0:
            continue
        severity = ("healthy" if mean["role"] == "healthy" else TRAIN_SEVERITY if split == "train"
                    else mean["metadata"]["stratum"])
        rows.append({
            "case": "configured_case14.json", "z": mean["z"], "parent_id": parent["parent_id"],
            "families": mean["families"], "severity": severity, "split": split,
            "window_id": f"{parent['parent_id']}:{mean['name']}", "measurement_sigma": "measurement_sigma.json",
            "measurement_convention": MEASUREMENT_CONVENTION, "measurement_kind": "noiseless_mean",
            "noise_replicates": int(replicates), "noise_seed": int(seed),
            "offline_metadata": {"contract": CONTRACT, **mean["metadata"]},
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _window_counts(rows: list[dict[str, Any]], *, detail: bool = False) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        key = [row["split"], "+".join(row["families"]) or "healthy"]
        if detail:
            key += [row["offline_metadata"]["stratum"], row["offline_metadata"]["residual_visible_energy_bin"]]
        counts["|".join(key)] += row["noise_replicates"]
    return dict(sorted(counts.items()))


def build_corpus(output_dir: str | Path, *, seed: int = 2026092301, workers: int = 1,
                 hif_corpora=DEFAULT_HIF_CORPORA, unbalance_corpus: str = DEFAULT_UNBALANCE_CORPUS,
                 evaluation_hif=DEFAULT_EVALUATION_HIF, max_parents_per_corpus: int | None = None,
                 artifacts: Path = ARTIFACTS, progress: bool = False) -> dict[str, Any]:
    final_output = Path(output_dir)
    if final_output.exists():
        raise FileExistsError(f"{final_output} already exists")
    # Build next to the target and rename at the end, so a refused source corpus
    # or a crash leaves no half-written corpus behind.
    output = final_output.with_name(final_output.name + ".partial")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    case, sigma = configured_case14(), measurement_sigma()
    (output / "configured_case14.json").write_text(json.dumps(case), encoding="utf-8")
    (output / "measurement_sigma.json").write_text(json.dumps(sigma.tolist()), encoding="utf-8")

    def limited(rows):
        return rows[:max_parents_per_corpus] if max_parents_per_corpus else rows

    def detectable_hashes(name):
        return ({vector_hash(r["z_clean"]) for r in fault_rows(name, artifacts)}
                if name and (Path(artifacts) / name).is_dir() else set())

    common = {"seed": seed, "sigma": sigma.tolist(), "case": case}
    tasks, evaluation_tasks, detectable_found = [], [], Counter()
    for corpus in hif_corpora:
        if corpus not in DEFAULT_HIF_DETECTABLE:
            warnings.warn(f"{corpus}: no detectable-subset mapping; dagger_detectable will be False for every parent")
        detectable = detectable_hashes(DEFAULT_HIF_DETECTABLE.get(corpus))
        for row in limited(fault_rows(corpus, artifacts)):
            flag = vector_hash(row["z_clean"]) in detectable
            detectable_found[corpus] += flag
            tasks.append({"kind": "hif", "row": row, "corpus": corpus, "detectable": flag, **common})
    detectable = detectable_hashes(DEFAULT_UNBALANCE_DETECTABLE)
    for row in limited(fault_rows(unbalance_corpus, artifacts)):
        flag = vector_hash(row["z_clean"]) in detectable
        detectable_found[unbalance_corpus] += flag
        tasks.append({"kind": "unbalance", "row": row, "corpus": unbalance_corpus, "detectable": flag, **common})
    for corpus in evaluation_hif or ():
        for row in limited(fault_rows(corpus, artifacts)):
            evaluation_tasks.append({"kind": "hif", "row": row, "corpus": corpus, "detectable": False,
                                     "evaluation_only": True, **common})

    def run(items, label):
        results = []
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            iterator = pool.map(dispatch, items, chunksize=2) if pool else map(dispatch, items)
            for index, result in enumerate(iterator):
                results.append(result)
                if progress and (index + 1) % 50 == 0:
                    print(f"{label}: {index + 1}/{len(items)} parents", flush=True)
        finally:
            if pool:
                pool.shutdown()
        return results

    parents = run(tasks, "main")
    splits = assign_splits(parents, seed)
    rows, failures = [], []
    for parent in parents:
        rows.extend(manifest_rows(parent, splits[parent["parent_id"]], seed))
        failures.extend(parent["failures"])
    _write_jsonl(output / "manifest.jsonl", rows)
    evaluation_rows = []
    if evaluation_tasks:
        for parent in run(evaluation_tasks, "evaluation"):
            evaluation_rows.extend(manifest_rows(parent, "test", seed, replicate_key="evaluation"))
            failures.extend(parent["failures"])
        _write_jsonl(output / "hif_resistance_evaluation_manifest.jsonl", evaluation_rows)
    _write_jsonl(output / "simulation_failures.jsonl", failures)

    summary = {
        "contract": CONTRACT, "seed": seed, "measurement_convention": MEASUREMENT_CONVENTION,
        "sources": {"hif": list(hif_corpora), "unbalance": unbalance_corpus,
                    "hif_evaluation_only": list(evaluation_hif or ())},
        "dagger_detectable_parents_found": dict(detectable_found),
        "split_fractions": SPLIT_FRACTIONS, "noise_replicates": REPLICATES,
        "train_severity": TRAIN_SEVERITY, "hif_parent_variants": HIF_PARENT_VARIANTS,
        "unbalance_parent_variants": UNBALANCE_PARENT_VARIANTS,
        "parents_by_split_and_source": {f"{s}:{f}": n for (s, f), n in sorted(
            Counter((splits[p["parent_id"]], p["family"]) for p in parents).items())},
        "noiseless_means": len(rows), "windows_by_split_and_family": _window_counts(rows),
        "evaluation_only_windows_by_family": _window_counts(evaluation_rows),
        "windows_by_split_family_stratum_energy": _window_counts(rows, detail=True),
        "evaluation_windows_by_family_stratum_energy": _window_counts(evaluation_rows, detail=True),
        "simulation_failures": len(failures),
        "inherited_properties": [
            "HIF parents use the HIF generator path (operating-point profiles, dispatch and PV setpoints) and "
            "unbalance parents the unbalance path (uniform load scale, then the row's stored OPF dispatch and "
            "setpoints); both keep generator reactive limits. Each parent carries its own healthy partners.",
            "No WLS-detectability filtering; offline_metadata.dagger_detectable marks DAgger's training subsets.",
            "The parent split is independent of the DAgger suite split; offline_metadata.dagger_corpus and "
            "dagger_source_id identify every source window.",
        ],
    }
    (output / "corpus_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    load_manifest(output / "manifest.jsonl")
    if evaluation_rows:
        load_manifest(output / "hif_resistance_evaluation_manifest.jsonl")
    output.rename(final_output)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026092301)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-parents-per-corpus", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--no-evaluation-sweep", action="store_true")
    args = parser.parse_args(argv)
    summary = build_corpus(args.output_dir, seed=args.seed, workers=args.workers,
                           max_parents_per_corpus=args.max_parents_per_corpus,
                           evaluation_hif=() if args.no_evaluation_sweep else DEFAULT_EVALUATION_HIF, progress=True)
    print(json.dumps({key: summary[key] for key in ("parents_by_split_and_source", "windows_by_split_and_family",
                                                    "dagger_detectable_parents_found", "simulation_failures")},
                     indent=2))


if __name__ == "__main__":
    main()
