#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate IEEE-14 high-impedance-fault measurement records.

The operator-facing vector remains the standard 122-entry IEEE-14 layout:
    z = [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]

The hidden HIF bus is only present inside the copied OpenDSS scenario model.
It is not exposed as a bus in z_obs or z_true.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

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
from three_phase_nlm import (  # type: ignore
    copy_ieee14_model,
    hif_ohms_from_pu,
    inject_midspan_hif_ieee14,
    run_ieee14_hif_nlm,
    simulate_hif_candidate,
    write_balanced_ieee14_load_override,
)
from three_phase_nlm.ieee14_adapter import ELIGIBLE_HIF_BRANCHES, branch_info_for_row0
from three_phase_nlm.hif_operating_point import canonicalize_ieee14_operating_point  # type: ignore
from Transmission.generate_measurements import (  # type: ignore
    MEASUREMENT_ORDER,
    compute_measurements_pu,
    make_index_map,
)


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
    arr = np.zeros(int(length), dtype=float)
    nb = 14
    nl = 20
    sigma = np.zeros_like(arr, dtype=float)
    sigma[0:nb] = 1e-3
    sigma[nb : 3 * nb] = 1e-2
    sigma[3 * nb : 3 * nb + 4 * nl] = 1e-2
    return sigma


def _maybe_add_noise(z_obs: list[float], rng: np.random.Generator, noise_scale: float) -> list[float]:
    if float(noise_scale) <= 0:
        return [float(x) for x in z_obs]
    arr = np.asarray(z_obs, dtype=float)
    sigma = _measurement_sigma(len(arr))
    return (arr + rng.normal(0.0, sigma * float(noise_scale))).astype(float).tolist()


def _sample_diverse_operating_point(
    rng: np.random.Generator,
    *,
    event_load_scale: float,
    load_log_std: float,
    dispatch_fraction: float,
    voltage_std: float,
) -> dict[str, Any]:
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
    return canonicalize_ieee14_operating_point({
        "load_scale": float(event_load_scale),
        "bus_load_scales": bus_scales,
        "generator_dispatch_kw": dispatch,
        "voltage_setpoints_pu": voltage_setpoints,
        "source_voltage_pu": float(np.clip(1.06 + rng.normal(0.0, voltage_std * 0.6), 1.03, 1.08)),
    })


def _scan_operating_points(
    rng: np.random.Generator,
    *,
    scan_count: int,
    mode: str,
    event_load_scale: float,
    load_log_std: float,
    dispatch_fraction: float,
    voltage_std: float,
) -> list[dict[str, Any]]:
    if int(scan_count) < 1:
        raise ValueError("scans_per_window must be positive")
    normalized = str(mode).strip().lower()
    if normalized not in {"identical_noise", "diverse"}:
        raise ValueError("operating_point_mode must be identical_noise or diverse")
    reference = canonicalize_ieee14_operating_point({"load_scale": float(event_load_scale)})
    if normalized == "identical_noise":
        return [dict(reference) for _ in range(int(scan_count))]
    return [
        reference,
        *[
            _sample_diverse_operating_point(
                rng,
                event_load_scale=float(event_load_scale),
                load_log_std=float(load_log_std),
                dispatch_fraction=float(dispatch_fraction),
                voltage_std=float(voltage_std),
            )
            for _ in range(int(scan_count) - 1)
        ],
    ]


def _build_meta(
    *,
    load_scale_min: float,
    load_scale_max: float,
    r_hif_pu_min: float,
    r_hif_pu_max: float,
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
) -> dict[str, Any]:
    nb = 14
    nl = 20
    idx_map = make_index_map(nb, nl)
    return {
        "case": "case14",
        "baseMVA": 100.0,
        "nb": nb,
        "nl": nl,
        "index_map": {k: [int(v.start), int(v.stop)] for k, v in idx_map.items()},
        "measurement_order": MEASUREMENT_ORDER,
        "branch_info": _branch_info_case14(),
        "hif": {
            "scenario": "high_impedance_fault",
            "eligible_branch_row0": [int(i) for i in ELIGIBLE_HIF_BRANCHES],
            "branch_order": BRANCH_ORDER,
            "bus_order": BUS_ORDER,
            "load_scale_range": [float(load_scale_min), float(load_scale_max)],
            "r_hif_pu_range": [float(r_hif_pu_min), float(r_hif_pu_max)],
            "split_ratio_range": [float(split_min), float(split_max)],
            "branch_sampling": str(branch_sampling),
            "scan_window": {
                "scans_per_window": int(scans_per_window),
                "operating_point_mode": str(operating_point_mode),
                "shared_parameters": ["branch_row0", "split_ratio", "phase", "r_hif_pu"],
                "scan_specific_fields": ["z_clean", "z_obs", "three_phase_voltages", "op_point"],
                "operating_point_schema": list(IEEE14_OPERATING_POINT_KEYS),
                "bus_load_scale_semantics": "profile_factor_multiplied_by_load_scale",
                "note": "identical_noise repeats one operating point; diverse varies spatial load, dispatch, and voltage setpoints while preserving the HIF.",
            },
            "generation": {
                "seed": int(seed),
                "noise_scale": float(noise_scale),
                "scan_load_log_std": float(scan_load_log_std),
                "scan_dispatch_fraction": float(scan_dispatch_fraction),
                "scan_voltage_std": float(scan_voltage_std),
                "rng_streams": "event_labels, operating_points, and measurement_noise are independent",
            },
            "phases": ["A", "B", "C"],
            "measurement_vector": "operator IEEE-14 122-entry z; hidden fault bus excluded",
            "nlm_diagnostic": {
                "fields": ["success", "converged", "top_hif_groups", "detected_top1", "detected_top3"],
                "note": "Generated HIF samples use the legacy three-phase NLM bridge when scenario models are available; metadata fallback is only for adapter smoke tests without model-backed evidence.",
            },
        },
    }


def generate_dataset(
    *,
    out_dir: str,
    n_hif: int,
    n_no_error: int,
    seed: int,
    load_scale_min: float,
    load_scale_max: float,
    r_hif_pu_min: float,
    r_hif_pu_max: float,
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
) -> None:
    if int(scans_per_window) < 1:
        raise ValueError("scans_per_window must be positive")
    if float(scan_load_log_std) < 0.0:
        raise ValueError("scan_load_log_std must be non-negative")
    if not 0.0 <= float(scan_dispatch_fraction) < 1.0:
        raise ValueError("scan_dispatch_fraction must be in [0, 1)")
    if float(scan_voltage_std) < 0.0:
        raise ValueError("scan_voltage_std must be non-negative")
    event_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    no_error_noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    out = Path(os.path.abspath(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    meta = _build_meta(
        load_scale_min=load_scale_min,
        load_scale_max=load_scale_max,
        r_hif_pu_min=r_hif_pu_min,
        r_hif_pu_max=r_hif_pu_max,
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
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ppc_base = case14()
    base_dss_dir = Path(_REPO_ROOT) / "IEEE_14_OpenDSS"
    scenarios_root = out / "scenarios"
    if keep_scenarios:
        scenarios_root.mkdir(exist_ok=True)

    eligible_hif_branches = [int(i) for i in ELIGIBLE_HIF_BRANCHES]
    if branch_sampling not in {"balanced", "random"}:
        raise ValueError(f"Unknown branch_sampling={branch_sampling!r}")
    branch_schedule: list[int] = []
    if branch_sampling == "balanced":
        while len(branch_schedule) < int(n_hif):
            branch_schedule.extend(int(i) for i in event_rng.permutation(eligible_hif_branches))
        branch_schedule = branch_schedule[: int(n_hif)]

    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for _ in range(int(n_no_error)):
            alpha = float(event_rng.uniform(load_scale_min, load_scale_max))
            solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
            if solved is None:
                continue
            z = compute_measurements_pu(solved).astype(float).tolist()
            rec = {
                "id": f"nehif_{event_rng.integers(1e12)}",
                "scenario": "no_error",
                "z_true": z,
                "z_obs": _maybe_add_noise(z, no_error_noise_rng, noise_scale),
                "label": {"error_type": "no_error"},
                "op_point": {"load_scale": alpha, "seed": int(seed)},
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
            branch_row0 = (
                int(branch_schedule[sample_idx])
                if branch_sampling == "balanced"
                else int(event_rng.choice(eligible_hif_branches))
            )
            dss_element = BRANCH_ORDER[branch_row0]
            split_ratio = float(event_rng.uniform(split_min, split_max))
            phase = str(event_rng.choice(["A", "B", "C"]))
            r_hif_pu = float(event_rng.uniform(r_hif_pu_min, r_hif_pu_max))
            r_hif_ohm = hif_ohms_from_pu(r_hif_pu, base_mva=100.0, kv_ll=1.0)

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
                injection = inject_midspan_hif_ieee14(
                    scenario_dir,
                    dss_element,
                    split_ratio=split_ratio,
                    phase=phase,
                    r_hif_ohm=r_hif_ohm,
                    base_mva=100.0,
                    kv_ll=1.0,
                    fault_bus=f"Fault_{branch_row0 + 1}_{sample_idx:06d}",
                    hif_load_name=f"Load.HIF_{branch_row0 + 1}_{sample_idx:06d}",
                )

                scan_op_points = _scan_operating_points(
                    scan_rng,
                    scan_count=int(scans_per_window),
                    mode=operating_point_mode,
                    event_load_scale=alpha,
                    load_log_std=scan_load_log_std,
                    dispatch_fraction=scan_dispatch_fraction,
                    voltage_std=scan_voltage_std,
                )
                scans = []
                simulation_cache: dict[str, dict[str, Any]] = {}
                for scan_index, scan_op_point in enumerate(scan_op_points):
                    op_key = json.dumps(scan_op_point, sort_keys=True, separators=(",", ":"))
                    if op_key not in simulation_cache:
                        simulation_cache[op_key] = simulate_hif_candidate(
                            candidate_branch_row0=branch_row0,
                            alpha=split_ratio,
                            phase=phase,
                            r_hif_pu=r_hif_pu,
                            op_point=scan_op_point,
                            pristine_model_dir=str(base_dss_dir),
                        )
                    simulated = simulation_cache[op_key]
                    z_scan = simulated["z"]
                    if len(z_scan) != 122:
                        raise RuntimeError(f"Unexpected z_obs length={len(z_scan)}, expected 122")
                    scans.append(
                        {
                            "scan_index": int(scan_index),
                            "z_clean": [float(x) for x in z_scan],
                            "z_obs": _maybe_add_noise(
                                [float(x) for x in z_scan], measurement_rng, noise_scale
                            ),
                            "three_phase_voltages": simulated["three_phase_voltages"],
                            "op_point": scan_op_point,
                            "topology_id": "ieee14_base",
                        }
                    )

                reference_scan = scans[0]

                solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
                if solved is None:
                    continue
                z_true = compute_measurements_pu(solved).astype(float).tolist()

                branch_info = branch_info_for_row0(branch_row0)
                nlm_diagnostic = run_ieee14_hif_nlm(
                    pristine_model_dir=str(base_dss_dir),
                    faulted_model_dir=str(scenario_dir),
                    target_dss_element=dss_element,
                    phase=phase,
                    r_hif_ohm=r_hif_ohm,
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
                    "r_hif_ohm": r_hif_ohm,
                    "kv_ln": injection.kv_ln,
                }
                rec = {
                    "id": sample_id,
                    "scenario": "high_impedance_fault",
                    "case": "IEEE14",
                    "z_true": z_true,
                    "z_obs": reference_scan["z_obs"],
                    "three_phase_voltages": reference_scan["three_phase_voltages"],
                    "label": shared_label,
                    "shared_label": shared_label,
                    "nlm_diagnostic": nlm_diagnostic,
                    "scan_count": len(scans),
                    "scans": scans,
                    "sigma_z": _measurement_sigma().astype(float).tolist(),
                    "topology_id": "ieee14_base",
                    "op_point": reference_scan["op_point"],
                    "window_metadata": {
                        "operating_point_mode": str(operating_point_mode),
                        "persistent_hif": True,
                        "seed": int(seed),
                        "sample_index": int(sample_idx),
                    },
                }
                if keep_scenarios:
                    rec["scenario_model_dir"] = str(scenario_dir.relative_to(out))
                handle.write(json.dumps(rec) + "\n")
            finally:
                if tmp_context is not None:
                    tmp_context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/measurements/out_measurements_hif", help="Output directory")
    parser.add_argument("--n-hif", type=int, default=200)
    parser.add_argument("--n-no-error", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--load-scale-min", type=float, default=0.80)
    parser.add_argument("--load-scale-max", type=float, default=1.25)
    parser.add_argument("--r-hif-pu-min", type=float, default=20.0)
    parser.add_argument("--r-hif-pu-max", type=float, default=200.0)
    parser.add_argument("--split-min", type=float, default=0.25)
    parser.add_argument("--split-max", type=float, default=0.75)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--scans-per-window", type=int, default=1)
    parser.add_argument(
        "--operating-point-mode",
        choices=["identical_noise", "diverse"],
        default="diverse",
        help="Use repeated-noise controls or electrically diverse operating points within each persistent HIF event.",
    )
    parser.add_argument("--scan-load-log-std", type=float, default=0.08)
    parser.add_argument("--scan-dispatch-fraction", type=float, default=0.20)
    parser.add_argument("--scan-voltage-std", type=float, default=0.008)
    parser.add_argument("--keep-scenarios", action="store_true")
    parser.add_argument(
        "--branch-sampling",
        choices=["balanced", "random"],
        default="balanced",
        help="Balanced cycles through eligible Line.* branches; random samples with replacement.",
    )
    args = parser.parse_args()

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
    )
    print(f"Wrote IEEE-14 HIF dataset to: {args.out}")


if __name__ == "__main__":
    main()
