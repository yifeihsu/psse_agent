"""IEEE 118 transfer set for the triage GNN: the pipeline's family recipes on IEEE 118, alarm-admitted.

    python -m research.classifier_triage.ieee118_dataset --output-dir output/classifier_triage_20261007/ieee118 --workers 16

The IEEE 14 triage GNN (``psse_env/oracle/models/triage_gnn_ieee14_20261004``)
was trained on the hypothesis-ranking study rows
(``research.hypothesis_ranking.build_dataset``).  This writes rows of the same
format on IEEE 118, each family drawn with its IEEE 14 recipe:

* operating point: every load scaled by U(0.80, 1.25), generator dispatch and
  voltage setpoints from the AC-OPF at those loads
  (``Transmission.generate_measurements.solve_ac_opf``), as in the IEEE 14
  corpora;
* ``hif``: the IEEE 118 three-phase OpenDSS model (138/161/345 kV bases, PV
  generators with reactive limits) at that dispatch, one resistive fault to
  ground at a fraction U(0.25, 0.75) along a same-voltage line (175 eligible),
  phase uniform, R from the 100-200 / 200-500 / 500-1000 ohm bands with equal
  weight and uniform within the band (the IEEE 14 physical corpora's recipe);
* ``measurement+hif``: one power meter of an admitted HIF window offset by
  max(U(0.10, 0.30), 10 sigma x U(1, 1.5)) with a random sign (the
  generator's overlay); the window is its parent;
* ``three_phase_unbalance``: the same model and dispatch, the load of one
  non-slack load bus split over its phases by Dirichlet(3, 3, 3) fractions,
  bus total kept;
* ``harmonic``: ``Transmission.generate_hse_traces.build_trace`` on the OPF
  point (harmonic reference at the case's slack), source a non-slack load bus,
  source-bus THD U(0.10, 0.20), orders 5 to 19, true-RMS Vm;
* ``measurement``, ``multi_measurement``, ``parameter``,
  ``measurement+parameter``: the round-0 generator on fresh IEEE 118 balanced
  corpora (OPF windows over the same load range), with its own family builders
  and admission (recoverable mode, ten-sigma meters, parameter ranking 1.0
  with rank allowance 2), one corpus and generator per shard;
* healthy windows (the OpenDSS reference of every waveform window and the OPF
  corpus windows, with fresh noise) and flow-pair mimics (both flow meters of
  an eligible line biased 10 to 15 sigma), kept when the WLS alarms.

Roots are admitted by the excess-energy rule (``psse_env.wls_admission``,
2026-10-11): the operator's alarm (chi-square at 1 % or max |r| >= 4) with
max |r| >= 5 or an excess objective J - dof >= 70, the same demand on every
network; the generator's balanced families use the same rule.  Every
waveform and harmonic draw is logged in ``attempts.jsonl`` with its outcome:
``admitted``; ``alarm_only`` (alarms, but without the margin); ``quiet``; or the
exclusion that stopped it (no AC-OPF solution, no three-phase operating point
-- a Newton failure, a constant-power band violation or a failed physical
audit -- or a WLS that does not converge).  The ``alarm_only`` draws are kept
in ``alarm_only.jsonl`` for a secondary look.  SCADA sigmas are the IEEE 14
corpora's (1e-3 on Vm, 1e-2 on power).  Topology roots are not built: the
generator's breaker-level topology model exists for IEEE 14 only.

Each root keeps its WLS payload in ``payloads.pkl`` (the ``data.attach_payloads``
cache format), and the noiseless three-phase voltages and branch currents of
every admitted waveform window go to ``phasors.jsonl.gz`` for later use.
"""
from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import copy
import gzip
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.wls_admission import DEFAULT_EXCESS_FLOOR, admitted as admission_admitted, operator_alarm  # noqa: E402
from research.classifier_triage.effect_sampler import (  # noqa: E402
    draw_target, effect_ratio, fit_scale, implied_magnitude, parse_band, rescale_corpus_row, wls_solver,
)

SYSTEM = "case118"
#: The effect band of a run (research.classifier_triage.effect_sampler): None keeps the physical draws.
EFFECT_BAND: tuple[float, float] | None = None
#: Slack on the band when a root's own solve is checked (the sampler's tolerance plus the noise draw).
BAND_SLACK = 0.15


def set_system(name: str) -> None:
    """Select the registered system of this process (workers take it from their task)."""
    global SYSTEM
    SYSTEM = str(name)


def set_effect_band(band: Any) -> None:
    global EFFECT_BAND
    EFFECT_BAND = parse_band(band)


def _configure(task: Mapping[str, Any]) -> None:
    set_system(task.get("system") or SYSTEM)
    set_effect_band(task.get("effect_band"))


def band_admitted(stats: Mapping[str, Any]) -> bool:
    """Root admission: the operator alarm and, under a band, the solve's own effect ratio in the band."""
    if not stats.get("alarm"):
        return False
    if EFFECT_BAND is None:
        return bool(stats.get("admitted"))
    low, high = EFFECT_BAND
    return low * (1.0 - BAND_SLACK) <= float(stats["effect_ratio"]) <= high * (1.0 + BAND_SLACK)
SEED = 20261007
LOAD_SCALE_RANGE = (0.80, 1.25)
HIF_BANDS_OHM = ((100.0, 200.0), (200.0, 500.0), (500.0, 1000.0))
HIF_SPLIT_RANGE = (0.25, 0.75)
DIRICHLET_CONCENTRATION = 3.0
HARMONIC_THD_RANGE = (0.10, 0.20)
HARMONIC_ORDERS = (5, 7, 11, 13, 17, 19)
CHI2_ALPHA, RESIDUAL_THRESHOLD, ANOMALY_MARGIN = 0.01, 4.0, 1.25
ADMISSION_RULE, ADMISSION_EXCESS_FLOOR = "excess_energy", DEFAULT_EXCESS_FLOOR
OVERLAY_RANGE, OVERLAY_SIGMA_FLOOR, OVERLAY_FLOOR_FACTOR = (0.10, 0.30), 10.0, (1.0, 1.5)
MIMIC_SIGMA_RANGE = (10.0, 15.0)
# Circuit-audit tolerance (pu of the 100 MVA base) for a state OpenDSS accepted only at
# three_phase_model.runtime.ACCEPTANCE_TOLERANCE; 1e-7 otherwise, as in the HIF sweep.
STRICT_AUDIT_TOLERANCE, LOOSE_AUDIT_TOLERANCE = 1e-7, 1e-5
FAMILY_CODES = {"hif": 1, "three_phase_unbalance": 2, "harmonic": 3, "mimic": 5, "corpus_healthy": 6}
DEFAULT_PLAN: dict[str, Any] = {
    "hif_attempts": 700,
    "unbalance_attempts": 500,
    "harmonic_attempts": 500,
    "healthy_replicas": 2,
    "balanced_shards": 16,
    "per_shard": {"measurement": 16, "multi_measurement": 16, "parameter": 16, "measurement+parameter": 16},
    "corpus_per_shard": {"no_error": 24, "measurement_error": 72, "parameter_error": 48},
    "corpus_healthy_replicas": 3,
    "mimic_per_variant": 100,
}
TRUTH_KEYS = ("measurement", "parameter", "topology", "hif", "unbalance", "harmonic")
SPLIT_FRACTIONS = (("train", 0.60), ("calibration", 0.20), ("test", 0.20))


# --------------------------------------------------------------------- helpers

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def split_of(parent_id: str, seed: int = SEED) -> str:
    """60/20/20 by physical parent (the hypothesis-ranking study's rule)."""
    from research.hypothesis_ranking.build_dataset import _split

    return _split(parent_id, seed)


def _empty_truth() -> dict[str, list]:
    return {key: [] for key in TRUTH_KEYS}


def _rng(seed: int, family: str, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), FAMILY_CODES[family], int(index)]))


@lru_cache(maxsize=1)
def context() -> dict[str, Any]:
    """The IEEE 118 source with its physical voltage bases, eligible HIF lines, load buses and SCADA sigma."""
    from psse_env.systems import resolve_system
    from scripts.audit_ieee14_hif_physical_sweep import configure_system

    spec = resolve_system(SYSTEM)
    source, profile, eligible = configure_system(SYSTEM)
    bus = np.asarray(source["bus"], dtype=float)
    slack_bus = int(bus[np.flatnonzero(bus[:, 1] == 3)[0], 0])
    load_buses = [int(row[0]) for row in bus if (row[2] != 0.0 or row[3] != 0.0) and int(row[0]) != slack_bus]
    return {
        "source": source, "profile": profile, "eligible": [int(k) for k in eligible],
        "sigma": np.asarray(spec.measurement_sigma(), dtype=float), "slack_bus": slack_bus, "load_buses": load_buses,
        "bus_kv": {int(row[0]): float(row[9]) for row in bus}, "nb": int(spec.nb), "nl": int(spec.nl),
        "branch_ends": [(int(row[0]), int(row[1])) for row in np.asarray(source["branch"], dtype=float)],
    }


def wls_check(z: Sequence[float], sigma: Sequence[float], metadata: Mapping[str, Any] | None = None
              ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The operator's WLS payload (``data.wls_payload``) and its alarm and admission flags.

    ``metadata`` (a generator root's own: sigma and any exact rows) replaces the plain sigma when given.
    """
    from research.classifier_triage.data import wls_payload

    declared = dict(metadata) if metadata else {"sigma_z": [float(v) for v in sigma]}
    try:
        payload = wls_payload({"case": SYSTEM, "measurements": [float(v) for v in z], "metadata": declared})
    except Exception as exc:  # a solver exception is a non-convergence like any other
        return None, {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    if payload is None:
        return None, {"success": False, "error": "WLS did not converge"}
    objective, dof = float(payload["objective"]), int(payload["dof"])
    threshold = float(chi2.ppf(1.0 - CHI2_ALPHA, max(dof, 1)))
    max_r = float(np.max(np.abs(payload["signed_normalized_residual"])))
    stats = {
        "success": True, "objective": objective, "dof": dof, "chi_square_threshold": threshold,
        "log_chi_square_ratio": float(np.log(max(objective, 1e-9) / threshold)), "max_normalized_residual": max_r,
        "excess_objective": objective - dof,
        "effect_ratio": effect_ratio(objective, dof, CHI2_ALPHA),
        "alarm": operator_alarm(objective, dof, max_r, alpha=CHI2_ALPHA, residual_threshold=RESIDUAL_THRESHOLD),
        "admitted": admission_admitted(objective, dof, max_r, rule=ADMISSION_RULE, alpha=CHI2_ALPHA,
                                       residual_threshold=RESIDUAL_THRESHOLD, margin=ANOMALY_MARGIN,
                                       excess_floor=ADMISSION_EXCESS_FLOOR),
    }
    return payload, stats


def _analysis(stats: Mapping[str, Any]) -> dict[str, Any]:
    """The ``analysis.wls`` block the study loaders read (alarm, statistic, dof)."""
    return {"wls": {"success": True, "alarm": bool(stats["alarm"]), "chi_square_statistic": stats["objective"],
                    "chi_square_dof": stats["dof"], "chi_square_threshold": stats["chi_square_threshold"],
                    "max_normalized_residual": stats["max_normalized_residual"]}}


def record(root_id: str, family: str, parent_id: str, z: Sequence[float], sigma: Sequence[float], truth: Mapping[str, Any],
           stats: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    out = {
        "root_id": root_id, "family": family, "parent_id": parent_id, "case": SYSTEM,
        "measurements": [float(v) for v in z], "metadata": {"sigma_z": [float(v) for v in sigma]},
        "truth": {key: list(truth.get(key, [])) for key in TRUTH_KEYS}, "admission": dict(stats),
        "analysis": _analysis(stats), "split": split_of(parent_id),
    }
    out.update(extra)
    return out


def overlay_meter(z: np.ndarray, sigma: np.ndarray, rng: np.random.Generator, nb: int,
                  base_stats: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """One power-meter gross error: the generator's magnitude law, or under a band an amplitude
    whose increment of the effect ratio over the base solve is a draw from the band."""
    index = int(rng.integers(nb, z.size))  # Pinj .. Qt
    if EFFECT_BAND is not None and base_stats is not None:
        direction = np.zeros(z.size)
        direction[index] = (1.0 if rng.random() < 0.5 else -1.0) * float(sigma[index])
        target = float(base_stats["effect_ratio"]) + draw_target(rng, EFFECT_BAND)
        fit = fit_scale(wls_solver(SYSTEM, sigma), np.asarray(z, dtype=float), direction, np.zeros(z.size), target,
                        scale_bounds=(2.0, 80.0))
        if not fit.get("reachable"):
            return np.array(z, dtype=float, copy=True), {"index": index, "effect_failure": fit.get("reason")}
        out = np.asarray(z, dtype=float) + float(fit["scale"]) * direction
        return out, {"index": index, "observed": float(out[index]), "clean": float(z[index]), "channel": None,
                     "effect": {"target_increment": target - float(base_stats["effect_ratio"]), "scale_sigma": float(fit["scale"]),
                                "ratio": fit["ratio"], "sampler": "effect_targeted_v1"}}
    magnitude = float(rng.uniform(*OVERLAY_RANGE))
    magnitude = max(magnitude, OVERLAY_SIGMA_FLOOR * float(sigma[index]) * float(rng.uniform(*OVERLAY_FLOOR_FACTOR)))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    out = np.array(z, dtype=float, copy=True)
    clean = float(out[index])
    out[index] = clean + sign * magnitude
    return out, {"index": index, "observed": float(out[index]), "clean": clean, "channel": None}


# --------------------------------------------------------- three-phase windows

def dispatched_case(source: Mapping[str, Any], solved: Mapping[str, Any], load_scale: float) -> dict[str, Any]:
    """The source case at the scaled loads with the OPF's generator outputs and voltage setpoints."""
    from pypower.idx_bus import BUS_I, PD, QD, VM
    from pypower.idx_gen import GEN_BUS, PG, VG

    case = copy.deepcopy(dict(source))
    case["bus"] = np.array(case["bus"], dtype=float, copy=True)
    case["gen"] = np.array(case["gen"], dtype=float, copy=True)
    case["bus"][:, PD] *= load_scale
    case["bus"][:, QD] *= load_scale
    row_of = {int(b): i for i, b in enumerate(np.asarray(solved["bus"])[:, BUS_I])}
    case["gen"][:, PG] = np.asarray(solved["gen"])[:, PG]
    case["gen"][:, VG] = [float(np.asarray(solved["bus"])[row_of[int(b)], VM]) for b in case["gen"][:, GEN_BUS]]
    return case


def _check_loads(telemetry: Mapping[str, Any], build: Mapping[str, Any], targets: Mapping[str, complex]) -> None:
    """Every load at its constant-PQ setpoint (a constant-Z fallback misses it by orders of magnitude)."""
    power = {row["element"].lower(): complex(row["power_into_element_pu"]["real"], row["power_into_element_pu"]["imag"])
             for row in telemetry["load_powers"]}
    base_kw = 1000.0 * float(build["assumptions"]["base_mva"])
    for row in build["registry"]["loads"]:
        element = str(row["element"]).lower()
        expected = targets.get(element, complex(float(row["kw"]), float(row["kvar"]))) / base_kw
        if abs(power[element] - expected) > 1e-8 + 1e-5 * abs(expected):
            raise RuntimeError(f"Constant-PQ load fallback: {row['element']}")


def hif_snapshot(build: Mapping[str, Any], hif: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    """The HIF sweep's snapshot (``scripts/audit_ieee14_hif_physical_sweep._snapshot``) with the audit
    tolerance matched to how OpenDSS accepted the state; returns telemetry, receipt, audit, loose flag."""
    from three_phase_model.disturbances import audit_disturbed_circuit, inject_midspan_hif
    from three_phase_model.measurements import extract_measurements
    from three_phase_model.runtime import _LOOSELY_ACCEPTED, compile_model, regulated_generator_deviation

    dss = compile_model(Path(build["output_dir"]) / "Master.dss")
    receipt = inject_midspan_hif(dss, build["registry"], build["assumptions"], **hif)
    telemetry = extract_measurements(dss, build["registry"], build["assumptions"], branch_overrides=receipt["branch_overrides"])
    loose = dss in _LOOSELY_ACCEPTED
    audit = audit_disturbed_circuit(dss, receipt, build["registry"], build["assumptions"],
                                    tolerance_pu=LOOSE_AUDIT_TOLERANCE if loose else STRICT_AUDIT_TOLERANCE)
    regulated = regulated_generator_deviation(dss)
    if not regulated["passed"]:
        raise RuntimeError(f"Regulated-generator control law violated: {regulated['worst']}")
    _check_loads(telemetry, build, {})
    if not audit["passed"]:
        raise RuntimeError(f"Physical circuit audit failed: {audit['failed_checks']}")
    return telemetry, receipt, audit, loose


def unbalance_snapshot(build: Mapping[str, Any], bus: int, fractions: Sequence[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split one bus's load over its phases (total kept), solve, and audit the device laws and KCL."""
    from three_phase_model.measurements import extract_measurements
    from three_phase_model.runtime import _LOOSELY_ACCEPTED, compile_model, regulated_generator_deviation, solve

    registry, assumptions = build["registry"], build["assumptions"]
    loads = [row for row in registry["loads"] if int(row["bus"]) == int(bus)]
    if len(loads) != 3 or {int(row["phase"]) for row in loads} != {1, 2, 3}:
        raise ValueError(f"bus {bus} lacks three single-phase loads")
    p_total = sum(float(row["kw"]) for row in loads)
    q_total = sum(float(row["kvar"]) for row in loads)
    dss = compile_model(Path(build["output_dir"]) / "Master.dss")
    targets: dict[str, complex] = {}
    for row in loads:
        fraction = float(fractions[int(row["phase"]) - 1])
        kw, kvar = p_total * fraction, q_total * fraction
        dss.Text.Command(f"Edit {row['element']} kW={kw:.16g} kvar={kvar:.16g}")
        targets[str(row["element"]).lower()] = complex(kw, kvar)
    solve(dss)
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS snapshot did not converge")
    regulated = regulated_generator_deviation(dss)
    if not regulated["passed"]:
        raise RuntimeError(f"Regulated-generator control law violated: {regulated['worst']}")
    telemetry = extract_measurements(dss, registry, assumptions)
    _check_loads(telemetry, build, targets)
    if float(telemetry["max_kcl_mismatch_pu"]) > 1e-6:
        raise RuntimeError(f"KCL mismatch {telemetry['max_kcl_mismatch_pu']:.2e} pu")
    vuf = {}
    for row in telemetry["three_phase_voltages"]:
        phasors = [complex(*pair) for pair in row["vln_pu_rect"]]
        a = np.exp(2j * np.pi / 3)
        positive = abs(phasors[0] + a * phasors[1] + a * a * phasors[2]) / 3.0
        negative = abs(phasors[0] + a * a * phasors[1] + a * phasors[2]) / 3.0
        vuf[int(row["external_bus"])] = negative / positive if positive > 0 else 0.0
    physical = {"bus": int(bus), "fractions": [float(f) for f in fractions], "total_kw": p_total, "total_kvar": q_total,
                "vuf_at_bus": float(vuf[int(bus)]), "max_vuf": float(max(vuf.values())),
                "max_vuf_bus": int(max(vuf, key=vuf.get)), "loose_acceptance": dss in _LOOSELY_ACCEPTED}
    return telemetry, physical


def _phasor_entry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    return {"three_phase_voltages": [{"external_bus": row["external_bus"], "vln_pu_rect": row["vln_pu_rect"]}
                                     for row in telemetry["three_phase_voltages"]],
            "three_phase_branch_currents": [{"branch_row0": row["branch_row0"], "i_from_pu_rect": row["i_from_pu_rect"],
                                             "i_to_pu_rect": row["i_to_pu_rect"]} for row in telemetry["three_phase_branch_currents"]]}


VOLTAGE_ROW_FIELDS = ("bus", "kvbase_ln", "vln_pu", "ang_deg")
CURRENT_ROW_FIELDS = ("branch", "branch_row0", "from_bus", "to_bus", "i_from_pu", "ang_from_deg", "i_to_pu", "ang_to_deg",
                      "ibase_from_a", "ibase_to_a")
PMU_SIGMA_PU = 1e-4
MEASUREMENT_CONVENTION = {
    "schema": "registry_three_phase_snapshot_measurements_v1", "shunt_convention": "ybus",
    "injection_semantics": "Vm is the phase-A line-neutral magnitude; Pinj/Qinj are the three-phase device totals with bus shunts "
                           "kept in Ybus; flows are three-phase terminal totals (three_phase_model.measurements)",
    "wls_model_convention": "shunts_in_ybus", "consistent_with_operator_wls": True,
}


def _corpus_phasor_rows(telemetry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The exporter's phasor rows in the waveform corpora's field set (``three_phase_nlm`` reads these)."""
    voltages = [{key: row[key] for key in VOLTAGE_ROW_FIELDS} for row in telemetry["three_phase_voltages"]]
    currents = [{key: row[key] for key in CURRENT_ROW_FIELDS} for row in telemetry["three_phase_branch_currents"]]
    return voltages, currents


def corpus_row(kind: str, index: int, *, z_obs: np.ndarray, z_true: np.ndarray, z_clean: np.ndarray, sigma: np.ndarray,
               telemetry: Mapping[str, Any], label: Mapping[str, Any], load_scale: float, rng: np.random.Generator) -> dict[str, Any]:
    """One row of the pipeline's waveform corpus schema (the IEEE 14 corpora's), phasors at the PMU sigma.

    ``z_true`` is the healthy reference of the same operating point (the
    generator's clean measurements), ``z_clean`` the disturbed noiseless mean.
    """
    from three_phase_nlm.branch_current_analysis import add_branch_current_noise
    from three_phase_nlm.measurement_noise import add_voltage_phasor_noise, generated_noise_contract

    voltages_clean, currents_clean = _corpus_phasor_rows(telemetry)
    voltages = add_voltage_phasor_noise(voltages_clean, rng, PMU_SIGMA_PU)
    currents = add_branch_current_noise(currents_clean, rng, PMU_SIGMA_PU)
    sigma_list = [float(v) for v in sigma]
    contract = generated_noise_contract(sigma_list, noise_scale=1.0, three_phase_sigma=PMU_SIGMA_PU,
                                        branch_current_sigma_pu=PMU_SIGMA_PU)
    scenario = "high_impedance_fault" if kind == "hif" else "three_phase_imbalance"
    row: dict[str, Any] = {
        "id": f"{SYSTEM}_{kind}_{index:05d}", "scenario": scenario, "case": SYSTEM, "network_case": SYSTEM,
        "z_obs": [float(v) for v in z_obs], "z_true": [float(v) for v in z_true], "z_clean": [float(v) for v in z_clean],
        "z_true_semantics": "balanced three-phase solve of the same operating point without the disturbance",
        "sigma_z": sigma_list, "three_phase_sigma": PMU_SIGMA_PU, "branch_current_sigma_pu": PMU_SIGMA_PU,
        "three_phase_voltages": voltages, "three_phase_voltages_clean": voltages_clean,
        "three_phase_branch_currents": currents, "three_phase_branch_currents_clean": currents_clean,
        "noise_contract": contract, "label": dict(label), "op_point": {"load_scale": float(load_scale)},
        "dispatch": {"mode": "opf", "solver": "Transmission.generate_measurements.solve_ac_opf at the same loads"},
        "measurement_convention": dict(MEASUREMENT_CONVENTION), "topology_id": f"{SYSTEM}_base",
        "voltage_base_profile": context()["profile"], "balanced_reference": "three_phase_model_same_operating_point",
    }
    if kind == "hif":
        scan = {"scan_index": 0, "z_obs": row["z_obs"], "z_clean": row["z_clean"], "sigma_z": sigma_list,
                "three_phase_sigma": PMU_SIGMA_PU, "branch_current_sigma_pu": PMU_SIGMA_PU,
                "three_phase_voltages": voltages, "three_phase_voltages_clean": voltages_clean,
                "three_phase_branch_currents": currents, "three_phase_branch_currents_clean": currents_clean,
                "noise_contract": contract, "op_point": {"load_scale": float(load_scale)}, "topology_id": f"{SYSTEM}_base",
                "measurement_convention": dict(MEASUREMENT_CONVENTION), "dispatch": row["dispatch"]}
        row.update(scans=[scan], scan_count=1, shared_label=dict(label),
                   window_metadata={"operating_point_mode": "single_snapshot", "dispatch_mode": "opf", "persistent_hif": True,
                                    "source_kind": "ieee118_three_phase_model"})
    return row


def hif_label(disturbance: Mapping[str, Any], physical: Mapping[str, Any]) -> dict[str, Any]:
    branch = int(disturbance["branch_row0"])
    return {
        "error_type": "high_impedance_fault", "branch_row0": branch, "line_index1": branch + 1,
        "dss_element": f"Line.{int(disturbance['from_bus'])}-{int(disturbance['to_bus'])}",
        "from_bus": int(disturbance["from_bus"]), "to_bus": int(disturbance["to_bus"]),
        "phase": "ABC"[int(disturbance["phase"]) - 1], "split_ratio": float(disturbance["alpha"]),
        "resistance_ohm": float(disturbance["resistance_ohm"]), "r_hif_ohm": float(disturbance["resistance_ohm"]),
        "r_hif_pu": float(physical["resistance_pu"]), "kv_ll": float(disturbance["kv_ll"]), "local_kv_ll": float(disturbance["kv_ll"]),
        "zbase_ohm": float(physical["zbase_ohm"]), "resistance_units": "ohm_local_base", "resistance_basis": "local_line_kv_ll",
        "resistance_band_ohm": [float(disturbance["band_low_ohm"]), float(disturbance["band_high_ohm"])],
        "nominal_fault_current_pu": float(physical["fault_current_pu"]), "fault_power_mw": float(physical["fault_power_mw"]),
        "voltage_base_profile": context()["profile"], "cross_voltage_branch": False,
    }


def unbalance_label(disturbance: Mapping[str, Any], physical: Mapping[str, Any]) -> dict[str, Any]:
    bus = int(disturbance["bus"])
    fractions = [float(v) for v in disturbance["fractions"]]
    return {
        "error_type": "three_phase_imbalance", "unbalance_bus": bus, "unbalance_bus_name": f"b{bus}", "kv_ll": float(disturbance["kv_ll"]),
        "load_split": {"bus": f"b{bus}", "source_mode": "split_per_phase_loads",
                       "fractions": {"a": fractions[0], "b": fractions[1], "c": fractions[2]},
                       "total": {"kW": float(physical["total_kw"]), "kvar": float(physical["total_kvar"])}},
        "vuf_at_bus": float(physical["vuf_at_bus"]), "max_vuf": float(physical["max_vuf"]),
    }


def waveform_job(task: Mapping[str, Any]) -> dict[str, Any]:
    """One HIF or unbalance window: OPF point, three-phase model, disturbance, noise, WLS admission.

    Every random draw is made before any solve, so an exclusion never shifts another window's draws.
    """
    from scripts.audit_ieee14_hif_physical_sweep import _compact_fault, _snapshot
    from three_phase_model.exporter import export_model, load_assumptions
    from Transmission.generate_measurements import scale_loads, solve_ac_opf

    _configure(task)
    if EFFECT_BAND is not None:
        raise ValueError("an effect band needs waveform rows from a corpus (--hif-corpus/--unbalance-corpus); "
                         "the three-phase replay path draws physical magnitudes")
    kind, index, seed, work = str(task["kind"]), int(task["index"]), int(task["seed"]), Path(task["work"])
    replicas = int(task.get("healthy_replicas", 2))
    ctx = context()
    sigma, nb = ctx["sigma"], ctx["nb"]
    rng = _rng(seed, kind, index)
    load_scale = float(rng.uniform(*LOAD_SCALE_RANGE))
    if kind == "hif":
        branch = int(ctx["eligible"][int(rng.integers(len(ctx["eligible"])))])
        band = HIF_BANDS_OHM[int(rng.integers(len(HIF_BANDS_OHM)))]
        disturbance = {"branch_row0": branch, "phase": int(rng.integers(1, 4)), "alpha": float(rng.uniform(*HIF_SPLIT_RANGE)),
                       "resistance_ohm": float(rng.uniform(*band)), "band_low_ohm": band[0], "band_high_ohm": band[1]}
        from_bus, to_bus = ctx["branch_ends"][branch]
        disturbance.update(from_bus=from_bus, to_bus=to_bus, kv_ll=ctx["bus_kv"][from_bus])
    elif kind == "three_phase_unbalance":
        bus = int(ctx["load_buses"][int(rng.integers(len(ctx["load_buses"])))])
        disturbance = {"bus": bus, "fractions": [float(v) for v in rng.dirichlet([DIRICHLET_CONCENTRATION] * 3)],
                       "kv_ll": ctx["bus_kv"][bus]}
    else:
        raise ValueError(f"unknown waveform kind {kind!r}")
    unit = rng.standard_normal(sigma.size)
    healthy_units = rng.standard_normal((replicas, sigma.size))
    overlay_rng = np.random.default_rng(rng.integers(2**63 - 1))
    phasor_rng = np.random.default_rng(rng.integers(2**63 - 1))  # the corpus row's PMU noise (drawn last)
    parent = f"{kind}_window:{index:05d}"
    attempt: dict[str, Any] = {"kind": kind, "index": index, "parent_id": parent, "load_scale": load_scale,
                               "disturbance": disturbance}
    out: dict[str, Any] = {"attempt": attempt, "roots": [], "alarm_only": [], "healthy": [], "phasors": {}, "corpus": []}
    started = time.perf_counter()
    model_dir = work / f"{kind}_{index:05d}"

    def excluded(reason: str, exc: BaseException | None = None) -> dict[str, Any]:
        attempt.update(outcome="excluded", reason=reason, seconds=time.perf_counter() - started)
        if exc is not None:
            attempt["error"] = f"{type(exc).__name__}: {exc}"[:600]
        return out

    try:
        try:
            solved = solve_ac_opf(scale_loads(ctx["source"], load_scale))
        except Exception as exc:
            return excluded("opf_nonconvergence", exc)
        if solved is None or not bool(solved.get("success", True)):
            return excluded("opf_nonconvergence")
        try:
            build = export_model(dispatched_case(ctx["source"], solved, load_scale), model_dir,
                                 assumptions=load_assumptions("normalized_diagonal"), case_id=SYSTEM,
                                 voltage_profile=ctx["profile"], generator_control="pv_q_limits")
        except Exception as exc:
            return excluded("model_export_failed", exc)
        try:
            healthy, _, _, _ = _snapshot(build)
        except Exception as exc:
            return excluded("healthy_operating_point_failed", exc)
        try:
            if kind == "hif":
                telemetry, receipt, audit, loose = hif_snapshot(build, {key: disturbance[key] for key in
                                                                        ("branch_row0", "phase", "alpha", "resistance_ohm")})
                compact = _compact_fault(receipt, audit)
                physical = {key: compact[key] for key in ("resistance_pu", "zbase_ohm", "fault_current_a", "fault_current_pu",
                                                         "fault_voltage_ln_pu", "fault_power_mw")}
                physical["loose_acceptance"] = loose
            else:
                telemetry, physical = unbalance_snapshot(build, disturbance["bus"], disturbance["fractions"])
        except Exception as exc:
            return excluded("no_operating_point", exc)
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

    z_true = np.asarray(telemetry["measurement_vector"], dtype=float)
    z_healthy = np.asarray(healthy["measurement_vector"], dtype=float)
    attempt["physical"] = physical
    attempt["max_abs_shift_sigma"] = float(np.max(np.abs(z_true - z_healthy) / sigma))
    z_obs = z_true + unit * sigma
    payload, stats = wls_check(z_obs, sigma)
    if payload is None:
        return excluded("wls_nonconvergence")
    attempt.update(outcome="admitted" if stats["admitted"] else "alarm_only" if stats["alarm"] else "quiet",
                   wls=stats, seconds=time.perf_counter() - started)
    truth = _empty_truth()
    if kind == "hif":
        truth["hif"] = [{**{k: v for k, v in disturbance.items()}, **physical, "split_ratio": disturbance["alpha"],
                         "r_hif_pu": physical["resistance_pu"], "error_type": "high_impedance_fault"}]
    else:
        truth["unbalance"] = [{"bus": disturbance["bus"], "kv_ll": disturbance["kv_ll"], **{k: v for k, v in physical.items() if k != "fractions"},
                               "load_split_fractions": {"a": disturbance["fractions"][0], "b": disturbance["fractions"][1],
                                                        "c": disturbance["fractions"][2]},
                               "error_type": "three_phase_imbalance"}]
    common = {"source_id": parent, "op_point": {"load_scale": load_scale, "dispatch": "ac_opf"}, "physical": physical}
    root = record(f"{kind}:{index:05d}", kind, parent, z_obs, sigma, truth, stats, **common)
    if stats["admitted"]:
        out["roots"].append((root, payload))
        out["phasors"][root["root_id"]] = _phasor_entry(telemetry)
        label = hif_label(disturbance, physical) if kind == "hif" else unbalance_label(disturbance, physical)
        out["corpus"].append(corpus_row(kind, index, z_obs=z_obs, z_true=z_healthy, z_clean=z_true, sigma=sigma,
                                        telemetry=telemetry, label=label, load_scale=load_scale, rng=phasor_rng))
        if kind == "hif":
            z_mixed, meter = overlay_meter(z_obs, sigma, overlay_rng, nb)
            mixed_payload, mixed_stats = wls_check(z_mixed, sigma)
            if mixed_payload is not None and mixed_stats["admitted"]:
                mixed_truth = copy.deepcopy(truth)
                mixed_truth["measurement"] = [meter]
                out["roots"].append((record(f"measurement+hif:{index:05d}", "measurement+hif", parent, z_mixed, sigma,
                                            mixed_truth, mixed_stats, **common), mixed_payload))
            attempt["measurement_hif"] = {"admitted": bool(mixed_payload is not None and mixed_stats["admitted"])}
    elif stats["alarm"]:
        out["alarm_only"].append((root, payload))
    for replica in range(replicas):
        z = z_healthy + healthy_units[replica] * sigma
        h_payload, h_stats = wls_check(z, sigma)
        if h_payload is None or not h_stats["alarm"]:
            continue
        out["healthy"].append((record(f"healthy:{kind}:{index:05d}:{replica}", "healthy_window", parent, z, sigma,
                                      _empty_truth(), h_stats, source="opendss_reference", source_id=parent,
                                      op_point={"load_scale": load_scale, "dispatch": "ac_opf"}), h_payload))
    attempt["healthy_alarms"] = len(out["healthy"])
    return out


def harmonic_job(task: Mapping[str, Any]) -> dict[str, Any]:
    """One harmonic snapshot on an AC-OPF point (``build_trace``), with WLS admission.

    Under an effect band the distortion's effect on the SCADA vector (its deviation from the
    fundamental-only readings) is scaled until the WLS excess lands in the band; the implied
    THD is the drawn one times the square root of the scale.
    """
    from pypower.idx_bus import BUS_I

    _configure(task)

    from Transmission.generate_hse_traces import build_trace
    from Transmission.generate_measurements import scale_loads, solve_ac_opf

    index, seed = int(task["index"]), int(task["seed"])
    ctx = context()
    sigma = ctx["sigma"]
    rng = _rng(seed, "harmonic", index)
    load_scale = float(rng.uniform(*LOAD_SCALE_RANGE))
    source_bus = int(ctx["load_buses"][int(rng.integers(len(ctx["load_buses"])))])
    thd = float(rng.uniform(*HARMONIC_THD_RANGE))
    trace_seed = int(rng.integers(2**31 - 1))
    parent = f"harmonic:{index:05d}"
    attempt: dict[str, Any] = {"kind": "harmonic", "index": index, "parent_id": parent, "load_scale": load_scale,
                               "disturbance": {"bus": source_bus, "thd_target": thd, "kv_ll": ctx["bus_kv"][source_bus]}}
    out: dict[str, Any] = {"attempt": attempt, "roots": [], "alarm_only": [], "healthy": [], "phasors": {}}
    started = time.perf_counter()
    try:
        solved = solve_ac_opf(scale_loads(ctx["source"], load_scale))
    except Exception as exc:
        solved = None
        attempt["error"] = f"{type(exc).__name__}: {exc}"[:600]
    if solved is None or not bool(solved.get("success", True)):
        attempt.update(outcome="excluded", reason="opf_nonconvergence", seconds=time.perf_counter() - started)
        return out
    bus = np.asarray(solved["bus"], dtype=float)
    row = int(np.flatnonzero(bus[:, BUS_I].astype(int) == source_bus)[0])
    try:
        trace = build_trace(row + 1, thd, trace_seed, list(HARMONIC_ORDERS), bus=bus, branch=solved["branch"],
                            sigma_vm=float(sigma[0]), sigma_pq=float(sigma[ctx["nb"]]))
    except Exception as exc:
        attempt.update(outcome="excluded", reason="harmonic_synthesis_failed", error=f"{type(exc).__name__}: {exc}"[:600],
                       seconds=time.perf_counter() - started)
        return out
    z_obs = np.asarray(trace["z_scada_meas"], dtype=float)
    z_true = np.asarray(trace["z_scada_true"], dtype=float)
    from Transmission.generate_measurements import compute_measurements_pu

    fundamental = np.asarray(compute_measurements_pu(solved), dtype=float)
    severity = trace["physical_severity"]
    physical = {"actual_thd": float(trace["actual_thd"]), "maximum_voltage_thd": float(severity["maximum_voltage_thd"]),
                "maximum_voltage_thd_bus": int(severity["maximum_voltage_thd_bus_1based"]),
                "max_abs_shift_sigma": float(np.max(np.abs(z_true - fundamental) / sigma))}
    effect: dict[str, Any] | None = None
    if EFFECT_BAND is not None:
        noise = z_obs - z_true
        target = draw_target(rng, EFFECT_BAND)
        fit = fit_scale(wls_solver(SYSTEM, sigma), fundamental, z_true - fundamental, noise, target)
        if not fit.get("reachable"):
            attempt.update(outcome="excluded", reason="effect_band_unreachable", effect=fit, seconds=time.perf_counter() - started)
            return out
        scale = float(fit["scale"])
        z_true = fundamental + scale * (z_true - fundamental)
        z_obs = z_true + noise
        implied = implied_magnitude("harmonic", {"thd_target": thd}, scale)
        physical = {**physical, "drawn_actual_thd": physical["actual_thd"], "actual_thd": physical["actual_thd"] * math.sqrt(scale),
                    "maximum_voltage_thd": physical["maximum_voltage_thd"] * math.sqrt(scale),
                    "max_abs_shift_sigma": float(np.max(np.abs(z_true - fundamental) / sigma))}
        effect = {"target": target, **{k: v for k, v in fit.items() if k != "reachable"}, "implied": implied,
                  "sampler": "effect_targeted_v1"}
        attempt["effect"] = effect
    attempt["physical"] = physical
    payload, stats = wls_check(z_obs, sigma)
    if payload is None:
        attempt.update(outcome="excluded", reason="wls_nonconvergence", seconds=time.perf_counter() - started)
        return out
    attempt.update(outcome="admitted" if band_admitted(stats) else "alarm_only" if stats["alarm"] else "quiet",
                   wls=stats, seconds=time.perf_counter() - started)
    truth = _empty_truth()
    truth["harmonic"] = [{"bus_1based": source_bus, "thd_target": thd if effect is None else effect["implied"]["thd"],
                          "drawn_thd_target": thd, "load_scale": load_scale, "operating_point": "ac_opf",
                          "kv_ll": ctx["bus_kv"][source_bus], **physical}]
    root = record(f"harmonic:{index:05d}", "harmonic", parent, z_obs, sigma, truth, stats, source_id=parent,
                  op_point={"load_scale": load_scale, "dispatch": "ac_opf"}, physical=physical, effect=effect)
    if band_admitted(stats):
        out["roots"].append((root, payload))
    elif stats["alarm"]:
        out["alarm_only"].append((root, payload))
    return out


# ------------------------------------------------------ waveform rows from a corpus

def corpus_waveform_job(task: Mapping[str, Any]) -> dict[str, Any]:
    """One HIF or unbalance window taken from a corpus row (``psse_env.providers.scenario_generator`` schema).

    The row carries the same operating point without the disturbance (``z_true``), the disturbance
    without sensor noise (``z_clean``) and the clean and disturbed phasors.  Under an effect band the
    row is re-magnituded (``effect_sampler.rescale_corpus_row``); otherwise it only gets a fresh
    sensor-noise draw.  Produces the same records as ``waveform_job`` (root, HIF-plus-meter overlay,
    healthy replicas, the phasor entry and the corpus row for the pipeline).
    """
    _configure(task)
    kind, index, seed = str(task["kind"]), int(task["index"]), int(task["seed"])
    replicas = int(task.get("healthy_replicas", 2))
    row = dict(task["row"])
    ctx = context()
    sigma, nb = ctx["sigma"], ctx["nb"]
    rng = _rng(seed, kind, index)
    row_sigma = np.asarray(row.get("sigma_z") or sigma, dtype=float)
    parent = f"{kind}_window:{index:05d}"
    label = dict(row.get("label") or {})
    attempt: dict[str, Any] = {"kind": kind, "index": index, "parent_id": parent, "source": "corpus_row",
                               "corpus_id": str(row.get("id")), "load_scale": (row.get("op_point") or {}).get("load_scale")}
    out: dict[str, Any] = {"attempt": attempt, "roots": [], "alarm_only": [], "healthy": [], "phasors": {}, "corpus": []}
    started = time.perf_counter()
    z_reference = np.asarray(row["z_true"], dtype=float)
    healthy_units = rng.standard_normal((replicas, row_sigma.size))
    overlay_rng = np.random.default_rng(rng.integers(2**63 - 1))
    if EFFECT_BAND is not None:
        target = draw_target(rng, EFFECT_BAND)
        scaled, fit = rescale_corpus_row(row, SYSTEM, rng, target)
        if scaled is None:
            attempt.update(outcome="excluded", reason="effect_band_unreachable", effect=fit, seconds=time.perf_counter() - started)
            return out
        row = scaled
        attempt["effect"] = row["effect"]
    else:
        z_clean = np.asarray(row["z_clean"], dtype=float)
        row["z_obs"] = (z_clean + rng.standard_normal(row_sigma.size) * row_sigma).tolist()
    z_obs = np.asarray(row["z_obs"], dtype=float)
    payload, stats = wls_check(z_obs, row_sigma)
    if payload is None:
        attempt.update(outcome="excluded", reason="wls_nonconvergence", seconds=time.perf_counter() - started)
        return out
    attempt.update(outcome="admitted" if band_admitted(stats) else "alarm_only" if stats["alarm"] else "quiet",
                   wls=stats, seconds=time.perf_counter() - started)
    implied = {k: v for k, v in (((row.get("effect") or {}).get("implied")) or {}).items() if k not in ("law", "effect_scale")}
    truth = _empty_truth()
    if kind == "hif":
        drawn = {k: label.get(k) for k in ("r_hif_ohm", "r_hif_pu", "fault_power_mw", "nominal_fault_current_pu") if k in label}
        truth["hif"] = [{**label, **implied, "drawn": drawn, "error_type": "high_impedance_fault"}]
    else:
        truth["unbalance"] = [{**label, "implied_fractions": implied.get("fractions"), "error_type": "three_phase_imbalance"}]
    physical = {k: v for k, v in label.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    common = {"source_id": parent, "corpus_id": str(row.get("id")), "op_point": row.get("op_point"), "physical": physical,
              "effect": row.get("effect")}
    root = record(f"{kind}:{index:05d}", kind, parent, z_obs, row_sigma, truth, stats, **common)
    if band_admitted(stats):
        out["roots"].append((root, payload))
        out["phasors"][root["root_id"]] = {"three_phase_voltages": row.get("three_phase_voltages"),
                                           "three_phase_branch_currents": row.get("three_phase_branch_currents"), "format": "corpus"}
        out["corpus"].append(row)
        if kind == "hif":
            z_mixed, meter = overlay_meter(z_obs, row_sigma, overlay_rng, nb, base_stats=stats)
            if "effect_failure" not in meter:
                mixed_payload, mixed_stats = wls_check(z_mixed, row_sigma)
                if mixed_payload is not None and mixed_stats["alarm"]:
                    mixed_truth = copy.deepcopy(truth)
                    mixed_truth["measurement"] = [{k: v for k, v in meter.items() if k != "effect"}]
                    out["roots"].append((record(f"measurement+hif:{index:05d}", "measurement+hif", parent, z_mixed, row_sigma,
                                                mixed_truth, mixed_stats, **common, overlay_effect=meter.get("effect")), mixed_payload))
                attempt["measurement_hif"] = {"admitted": bool(mixed_payload is not None and mixed_stats["alarm"])}
    elif stats["alarm"]:
        out["alarm_only"].append((root, payload))
    for replica in range(replicas):
        z = z_reference + healthy_units[replica] * row_sigma
        h_payload, h_stats = wls_check(z, row_sigma)
        if h_payload is None or not h_stats["alarm"]:
            continue
        out["healthy"].append((record(f"healthy:{kind}:{index:05d}:{replica}", "healthy_window", parent, z, row_sigma,
                                      _empty_truth(), h_stats, source="corpus_reference", source_id=parent,
                                      op_point=row.get("op_point")), h_payload))
    attempt["healthy_alarms"] = len(out["healthy"])
    return out


def _corpus_waveform_safe(task):
    return _safe(corpus_waveform_job, task)


# --------------------------------------------------------------- balanced shards

def build_generator(corpus: Mapping[str, Any], derived: Path, seed: int):
    """The round-0 generator on one fresh corpus of the system, with the study's balanced-family settings.

    Under an effect band the corpus rows already carry the band's magnitudes, so the generator admits
    on the operator alarm alone (margin 1) and the ten-sigma meter floor is lifted; the band check is
    applied to each root's own solve afterwards (``band_admitted``).
    """
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator

    if EFFECT_BAND is not None:
        admission = {"admission_rule": "chi_square_margin", "anomaly_margin": 1.0, "min_measurement_error_sigma": None}
    else:
        admission = {"admission_rule": ADMISSION_RULE, "admission_excess_floor": ADMISSION_EXCESS_FLOOR,
                     "min_measurement_error_sigma": 10.0}
    return Round0ScenarioGenerator(
        system=SYSTEM, admission_mode="recoverable", corpus_path=corpus["corpus_path"],
        balanced_artifact_dir=corpus["artifact_dir"], derived_case_dir=derived, seed=seed,
        normalized_residual_threshold=RESIDUAL_THRESHOLD,
        parameter_ranking_dominance_threshold=1.0, parameter_target_rank_allowance=2, **admission,
    )


def balanced_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    """One corpus, one generator: meter and parameter roots, healthy corpus windows and flow-pair mimics."""
    from psse_env.providers.balanced_corpus import build_balanced_corpus
    from research.hypothesis_ranking.build_dataset import RootBuilder, _root_record

    _configure(task)
    shard, seed, out_dir = int(task["shard"]), int(task["seed"]), Path(task["out_dir"])
    plan = dict(task["per_shard"])
    ctx = context()
    sigma, nb, nl = ctx["sigma"], ctx["nb"], ctx["nl"]
    started = time.perf_counter()
    shard_dir = out_dir / f"shard_{shard:02d}"
    corpus = build_balanced_corpus(shard_dir / "corpus", system=SYSTEM, seed=seed, counts=dict(task["corpus_counts"]),
                                   load_scale_range=LOAD_SCALE_RANGE, num_scans=3, effect_band=EFFECT_BAND)
    if EFFECT_BAND is not None:
        # the composed overlay of measurement+parameter follows the generator's own magnitude law,
        # outside the band: the effect datasets leave that family out
        plan["measurement+parameter"] = 0
    generator = build_generator(corpus, shard_dir / "derived_cases", seed + 1)
    builder = RootBuilder(generator, seed)
    timings = {}
    for family, call in (("measurement", lambda: builder.build_measurement("measurement", plan.get("measurement", 0))),
                         ("multi_measurement", lambda: builder.build_measurement("multi_measurement", plan.get("multi_measurement", 0))),
                         ("parameter", lambda: builder.build_parameter(plan.get("parameter", 0), plan.get("measurement+parameter", 0)))):
        t0 = time.perf_counter()
        try:
            call()
        except Exception as exc:  # keep the shard's other families; the manifest names the failure
            builder.skips.setdefault(family, {})[f"builder_exception: {type(exc).__name__}: {exc}"[:300]] = 1
        timings[family] = time.perf_counter() - t0
    result: dict[str, Any] = {"shard": shard, "roots": [], "healthy": [], "mimic": [], "skips": builder.skips,
                              "corpus": {k: corpus.get(k) for k in ("counts", "complete", "corpus_path")}, "timings": timings}
    for root in builder.roots:
        payload, stats = wls_check(root["measurements"], sigma, root.get("metadata") or None)
        if payload is None or not band_admitted(stats):
            continue
        parent = f"corpus118:{shard:02d}:{root['source_id']}"
        clean = {key: value for key, value in root.items() if key not in ("parent_id",)}
        clean.update(parent_id=parent, admission=stats, analysis=_analysis(stats), split=split_of(parent),
                     case_path=str(root["case"]), case=SYSTEM if Path(str(root["case"])).stem.lower() == SYSTEM else str(root["case"]))
        result["roots"].append((clean, payload))
    # Healthy OPF windows: the corpus's own noisy draw plus fresh replicas of its clean telemetry.
    no_error = list(generator._corpus().get("no_error", []))
    healthy_rng = _rng(seed, "corpus_healthy", shard)
    for row in no_error:
        parent = f"corpus118:{shard:02d}:{row['id']}"
        row_sigma = np.asarray(row.get("sigma_z") or sigma, dtype=float)
        draws = [np.asarray(row["z_obs"], dtype=float)]
        draws += [np.asarray(row["z_true"], dtype=float) + healthy_rng.standard_normal(row_sigma.size) * row_sigma
                  for _ in range(int(task.get("corpus_healthy_replicas", 0)))]
        for replica, z in enumerate(draws):
            payload, stats = wls_check(z, row_sigma)
            if payload is None or not stats["alarm"]:
                continue
            result["healthy"].append((record(f"healthy:corpus118:{shard:02d}:{row['id']}:{replica}", "healthy_window", parent, z,
                                             row_sigma, _empty_truth(), stats, source="opf_corpus", source_id=str(row["id"])),
                                      payload))
    # Flow-pair mimics on eligible HIF lines, from the corpus's noisy healthy draws.
    mimic_rng = _rng(seed, "mimic", shard)
    per_variant = int(task.get("mimic_per_variant", 0))
    variants = (("mimic_flow_pair_same_sign", (1.0, 1.0)), ("mimic_flow_pair_opposite_sign", (1.0, -1.0)))
    order = mimic_rng.permutation(len(no_error)) if no_error else []
    built = {name: 0 for name, _ in variants}
    for position_index, position in enumerate(order):
        name, signs = variants[position_index % 2]
        if built[name] >= per_variant:
            if all(count >= per_variant for count in built.values()):
                break
            continue
        row = no_error[int(position)]
        row_sigma = np.asarray(row.get("sigma_z") or sigma, dtype=float)
        z = np.asarray(row["z_obs"], dtype=float).copy()
        line = int(ctx["eligible"][int(mimic_rng.integers(len(ctx["eligible"])))])
        errors = []
        mimic_effect = None
        if EFFECT_BAND is not None:
            # the flow pair's magnitude is a band draw too, so a mimic is as loud as any root
            direction = np.zeros(z.size)
            for block, sign in ((0, signs[0]), (2, signs[1])):
                channel = 3 * nb + block * nl + line
                direction[channel] = sign * float(row_sigma[channel])
            fit = fit_scale(wls_solver(SYSTEM, row_sigma), z, direction, np.zeros(z.size), draw_target(mimic_rng, EFFECT_BAND),
                            scale_bounds=(2.0, 80.0))
            if not fit.get("reachable"):
                continue
            mimic_effect = {k: v for k, v in fit.items() if k != "reachable"}
            for block, sign in ((0, signs[0]), (2, signs[1])):
                channel = 3 * nb + block * nl + line
                clean = float(z[channel])
                z[channel] = clean + float(fit["scale"]) * direction[channel]
                errors.append({"index": channel, "observed": float(z[channel]), "clean": clean, "channel": None})
        else:
            for block, sign in ((0, signs[0]), (2, signs[1])):  # Pf and Pt
                channel = 3 * nb + block * nl + line
                magnitude = float(mimic_rng.uniform(*MIMIC_SIGMA_RANGE)) * float(row_sigma[channel])
                clean = float(z[channel])
                z[channel] = clean + sign * magnitude
                errors.append({"index": channel, "observed": float(z[channel]), "clean": clean, "channel": None})
        payload, stats = wls_check(z, row_sigma)
        if payload is None:
            continue
        built[name] += 1
        if not stats["alarm"]:
            continue
        parent = f"corpus118:{shard:02d}:{row['id']}"
        truth = _empty_truth()
        truth["measurement"] = errors
        result["mimic"].append((record(f"{name}:{shard:02d}:{row['id']}:{line}", name, parent, z, row_sigma, truth, stats,
                                       source_id=str(row["id"]), mimic={"branch_row0": line, "signs": list(signs)},
                                       effect=mimic_effect), payload))
    result["seconds"] = time.perf_counter() - started
    return result


# ------------------------------------------------------------------------ driver

def _safe(fn, task):
    try:
        return fn(task)
    except Exception as exc:
        return {"crash": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "task": dict(task)}


def _waveform_safe(task):
    return _safe(harmonic_job if task["kind"] == "harmonic" else waveform_job, task)


def _balanced_safe(task):
    return _safe(balanced_shard, task)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--plan", type=json.loads, default=None, help="JSON overrides of DEFAULT_PLAN")
    parser.add_argument("--stages", default="waveform,balanced", help="waveform (HIF, unbalance, harmonic) and/or balanced")
    parser.add_argument("--system", default=SYSTEM, help="registered system (case14, case57, case118)")
    parser.add_argument("--effect-band", default=None,
                        help="low,high: draw every magnitude as the WLS effect ratio in this band "
                             "(research.classifier_triage.effect_sampler); needs the corpus waveform rows")
    parser.add_argument("--hif-corpus", action="append", default=[], help="HIF corpus samples.jsonl (rows replace the three-phase replays)")
    parser.add_argument("--unbalance-corpus", action="append", default=[], help="unbalance corpus samples.jsonl")
    args = parser.parse_args(argv)
    set_system(args.system)
    set_effect_band(args.effect_band)
    if EFFECT_BAND is not None and not (args.hif_corpus or args.unbalance_corpus):
        parser.error("--effect-band needs --hif-corpus and/or --unbalance-corpus")
    task_common = {"system": SYSTEM, "effect_band": list(EFFECT_BAND) if EFFECT_BAND else None}
    plan = copy.deepcopy(DEFAULT_PLAN)
    for key, value in (args.plan or {}).items():
        plan[key] = value
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("roots.jsonl", "manifest.json")):
        raise SystemExit(f"{out} already holds a dataset; choose a fresh directory")
    work = out / "_work"
    work.mkdir(exist_ok=True)
    started = time.perf_counter()
    roots, alarm_only, healthy, mimic, attempts, crashes, shard_reports = [], [], [], [], [], [], []
    payloads: dict[str, Any] = {}
    phasors: dict[str, Any] = {}
    corpus: dict[str, list[dict[str, Any]]] = {"hif": [], "three_phase_unbalance": []}

    def take(result: Mapping[str, Any]) -> None:
        if "crash" in result:
            crashes.append(result)
            return
        for name, sink in (("roots", roots), ("alarm_only", alarm_only), ("healthy", healthy), ("mimic", mimic)):
            for item, payload in result.get(name, []):
                sink.append(item)
                payloads[str(item["root_id"])] = payload
        phasors.update(result.get("phasors") or {})
        # Waveform results carry corpus rows; a balanced shard's "corpus" is its builder report.
        rows = result.get("corpus")
        for row in (rows if isinstance(rows, list) else []):
            corpus["hif" if row["scenario"] == "high_impedance_fault" else "three_phase_unbalance"].append(row)
        if "attempt" in result:
            attempts.append(result["attempt"])

    if "waveform" in stages:
        tasks: list[dict[str, Any]] = []
        if args.hif_corpus or args.unbalance_corpus:
            # waveform rows from corpora (both systems alike); rows are shuffled per kind and capped by the plan
            order_rng = np.random.default_rng(args.seed + 17)
            for kind, files, cap_key in (("hif", args.hif_corpus, "hif_attempts"),
                                         ("three_phase_unbalance", args.unbalance_corpus, "unbalance_attempts")):
                rows: list[dict[str, Any]] = []
                for path in files:
                    with Path(path).open(encoding="utf-8") as stream:
                        rows += [r for r in (json.loads(line) for line in stream if line.strip())
                                 if (r.get("label") or {}).get("error_type") not in (None, "no_error")]
                # more draws than rows cycle the shuffled rows: each pass gets its own target and noise
                order = order_rng.permutation(len(rows)) if rows else np.zeros(0, dtype=int)
                count = int(plan[cap_key]) if rows else 0
                tasks += [{"kind": kind, "index": i, "seed": args.seed, "row": rows[int(order[i % len(rows)])],
                           "healthy_replicas": plan["healthy_replicas"], **task_common} for i in range(count)]
            corpus_tasks = len(tasks)
        else:
            corpus_tasks = 0
            tasks += ([{"kind": "hif", "index": i, "seed": args.seed, "work": str(work), "healthy_replicas": plan["healthy_replicas"],
                        **task_common} for i in range(int(plan["hif_attempts"]))]
                      + [{"kind": "three_phase_unbalance", "index": i, "seed": args.seed, "work": str(work),
                          "healthy_replicas": plan["healthy_replicas"], **task_common} for i in range(int(plan["unbalance_attempts"]))])
        tasks += [{"kind": "harmonic", "index": i, "seed": args.seed, **task_common} for i in range(int(plan["harmonic_attempts"]))]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_corpus_waveform_safe if (k < corpus_tasks) else _waveform_safe, task)
                       for k, task in enumerate(tasks)]
            for done, future in enumerate(as_completed(futures), start=1):
                take(future.result())
                if done % 100 == 0 or done == len(futures):
                    print(f"[waveform] {done}/{len(futures)} draws, {len(roots)} roots, {len(healthy)} healthy alarms, "
                          f"{len(crashes)} crashes, {time.perf_counter() - t0:.0f} s", flush=True)
    if "balanced" in stages:
        shard_tasks = [{"shard": shard, "seed": args.seed + 1000 * (shard + 1), "out_dir": str(out / "balanced"),
                        "per_shard": plan["per_shard"], "corpus_counts": plan["corpus_per_shard"],
                        "corpus_healthy_replicas": plan["corpus_healthy_replicas"],
                        "mimic_per_variant": int(np.ceil(int(plan["mimic_per_variant"]) / int(plan["balanced_shards"]))),
                        **task_common}
                       for shard in range(int(plan["balanced_shards"]))]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(shard_tasks))) as pool:
            futures = [pool.submit(_balanced_safe, task) for task in shard_tasks]
            for future in as_completed(futures):
                result = future.result()
                take(result)
                if "crash" not in result:
                    shard_reports.append({k: result[k] for k in ("shard", "skips", "corpus", "timings", "seconds")})
                print(f"[balanced] shard done ({len(shard_reports)}/{len(shard_tasks)}), roots {len(roots)}, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)

    counts = {"roots": _write_jsonl(out / "roots.jsonl", roots), "alarm_only": _write_jsonl(out / "alarm_only.jsonl", alarm_only),
              "healthy_alarms": _write_jsonl(out / "healthy_alarms.jsonl", healthy), "mimic": _write_jsonl(out / "mimic.jsonl", mimic),
              "attempts": _write_jsonl(out / "attempts.jsonl", attempts), "crashes": _write_jsonl(out / "crashes.jsonl", crashes)}
    with (out / "payloads.pkl").open("wb") as stream:
        pickle.dump(payloads, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(out / "phasors.jsonl.gz", "wt", encoding="utf-8") as stream:
        for root_id, entry in phasors.items():
            stream.write(json.dumps(_jsonable({"root_id": root_id, **entry})) + "\n")
    # The pipeline's waveform corpora: the admitted HIF and unbalance windows in the IEEE 14 corpora's row
    # schema (psse_env.providers.scenario_generator consumes them through --hif-sample-paths / --imbalance-sample-path).
    if corpus["hif"] or corpus["three_phase_unbalance"]:
        corpus_dir = out / "pipeline_corpus"
        corpus_dir.mkdir(exist_ok=True)
        corpus_counts = {}
        for kind, name in (("hif", "hif_samples.jsonl"), ("three_phase_unbalance", "unbalance_samples.jsonl")):
            rows = sorted(corpus[kind], key=lambda row: row["id"])
            corpus_counts[kind] = _write_jsonl(corpus_dir / name, rows)
        (corpus_dir / "meta.json").write_text(json.dumps(_jsonable({
            "schema": "ieee118_waveform_corpus_v1", "system": SYSTEM, "counts": corpus_counts,
            "three_phase_sigma": PMU_SIGMA_PU, "branch_current_sigma_pu": PMU_SIGMA_PU,
            "hif": {"generation": {"dispatch_mode": "opf", "three_phase_noise_pu": PMU_SIGMA_PU, "branch_current_noise_pu": PMU_SIGMA_PU,
                                   "skipped_windows": []}, "scan_window": {"scans_per_window": 1}},
            "imbalance": {"generation": {"dispatch_mode": "opf", "three_phase_noise_pu": PMU_SIGMA_PU,
                                         "branch_current_noise_pu": PMU_SIGMA_PU, "skipped_windows": []}},
            "measurement_convention": MEASUREMENT_CONVENTION, "voltage_base_profile": context()["profile"],
            "admission": {"rule": ADMISSION_RULE, "excess_floor": ADMISSION_EXCESS_FLOOR},
        }), indent=2), encoding="utf-8")
        counts["pipeline_corpus"] = corpus_counts
    outcomes: dict[str, dict[str, int]] = {}
    for attempt in attempts:
        key = attempt.get("reason") if attempt.get("outcome") == "excluded" else attempt.get("outcome")
        bucket = outcomes.setdefault(attempt["kind"], {})
        bucket[str(key)] = bucket.get(str(key), 0) + 1
    by_family: dict[str, int] = {}
    for root in roots:
        by_family[root["family"]] = by_family.get(root["family"], 0) + 1
    manifest = {
        "schema": "classifier_triage_ieee118_transfer_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(), "seed": args.seed, "plan": plan, "stages": sorted(stages), "system": SYSTEM,
        "effect_band": list(EFFECT_BAND) if EFFECT_BAND else None,
        "waveform_source": {"hif_corpora": list(args.hif_corpus), "unbalance_corpora": list(args.unbalance_corpus)}
        if (args.hif_corpus or args.unbalance_corpus) else "three_phase_replay",
        "recipes": {"load_scale_range": LOAD_SCALE_RANGE, "dispatch": "ac_opf", "hif_bands_ohm": HIF_BANDS_OHM,
                    "hif_split_range": HIF_SPLIT_RANGE, "dirichlet_concentration": DIRICHLET_CONCENTRATION,
                    "harmonic_thd_range": HARMONIC_THD_RANGE, "harmonic_orders": HARMONIC_ORDERS,
                    "generator_control": "pv_q_limits", "voltage_profile": context()["profile"],
                    "eligible_hif_lines": len(context()["eligible"]), "load_buses": len(context()["load_buses"]),
                    "overlay": {"range_pu": OVERLAY_RANGE, "sigma_floor": OVERLAY_SIGMA_FLOOR, "floor_factor": OVERLAY_FLOOR_FACTOR},
                    "mimic_sigma_range": MIMIC_SIGMA_RANGE},
        "admission": {"chi2_alpha": CHI2_ALPHA, "normalized_residual_threshold": RESIDUAL_THRESHOLD, "anomaly_margin": ANOMALY_MARGIN,
                      "rule": ADMISSION_RULE, "excess_floor": ADMISSION_EXCESS_FLOOR,
                      "description": "roots: operator alarm (chi-square at alpha OR max|r| >= 4) AND (max|r| >= margin x 4 OR J - dof >= floor); "
                                     "healthy windows and mimics: the operator alarm alone"},
        "counts": counts, "roots_by_family": by_family, "attempt_outcomes": outcomes, "balanced_shards": shard_reports,
        "split_fractions": dict(SPLIT_FRACTIONS), "total_seconds": time.perf_counter() - started, "workers": args.workers,
    }
    (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"counts": counts, "roots_by_family": by_family, "attempt_outcomes": outcomes}, indent=2), flush=True)
    return 0 if not crashes else 2


if __name__ == "__main__":
    raise SystemExit(main())
