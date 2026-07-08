#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate IEEE-14 multi-error measurement records for the PSSE agent SFT pipeline.

This script intentionally writes a separate artifact tree and does not modify
the existing single-error datasets.  Each emitted record uses
scenario="multi_error" with backward-compatible component labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Transmission.generate_measurements import (  # noqa: E402
    DEFAULTS,
    DEFAULT_SIGMAS,
    HARMONICS_AVAILABLE,
    MEASUREMENT_ORDER,
    _choose_random_cb_open,
    _nb_to_operator_z,
    base_gaussian_noise,
    branch_line_mask,
    compute_measurements_pu,
    load_case,
    make_harmonic_anomaly_record,
    make_index_map,
    make_initial_state_guess,
    make_no_error_record,
    make_parameter_error_record,
    make_topology_error_record,
    sigma_vector,
    write_ppc_as_matpower_m,
)
from pypower.api import ppoption, runpf  # noqa: E402
from pypower.idx_brch import BR_R, BR_STATUS, BR_X, F_BUS, TAP, T_BUS  # noqa: E402
from pypower.idx_bus import VA, VM  # noqa: E402


ERROR_PRIORITY = [
    "measurement_error",
    "parameter_error",
    "topology_error",
    "harmonic_anomaly",
]
SHORT_NAME = {
    "measurement_error": "measurement",
    "parameter_error": "parameter",
    "topology_error": "topology",
    "harmonic_anomaly": "harmonic",
}
FAMILY_ALIASES = {
    "measurement": "measurement_error",
    "measurement_error": "measurement_error",
    "parameter": "parameter_error",
    "parameter_error": "parameter_error",
    "topology": "topology_error",
    "topology_error": "topology_error",
    "harmonic": "harmonic_anomaly",
    "harmonic_anomaly": "harmonic_anomaly",
}
DEFAULT_COMBOS = [
    ("measurement_error", "parameter_error"),
    ("measurement_error", "topology_error"),
    ("measurement_error", "harmonic_anomaly"),
    ("parameter_error", "topology_error"),
    ("parameter_error", "harmonic_anomaly"),
    ("topology_error", "harmonic_anomaly"),
]
SCADA_STRUCTURAL_FAMILIES = {"parameter_error", "topology_error"}
CASE14_STANDARD_BRANCH_NAMES = [
    "line_1-2",
    "line_1-5",
    "line_2-3",
    "line_2-4",
    "line_2-5",
    "line_3-4",
    "line_4-5",
    "trafo_4-7",
    "trafo_4-9",
    "trafo_5-6",
    "line_6-11",
    "line_6-12",
    "line_6-13",
    "trafo_7-8",
    "trafo_7-9",
    "line_9-10",
    "line_9-14",
    "line_10-11",
    "line_12-13",
    "line_13-14",
]


def combo_key(families: Sequence[str]) -> str:
    return "+".join(SHORT_NAME[family] for family in families)


def canonicalize_combo(families: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    parsed: list[str] = []
    for raw in families:
        key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        family = FAMILY_ALIASES.get(key)
        if family is None:
            raise ValueError(f"Unsupported multi-error family {raw!r}. Use one of: {sorted(FAMILY_ALIASES)}")
        if family not in seen:
            seen.add(family)
            parsed.append(family)
    ordered = tuple(family for family in ERROR_PRIORITY if family in seen)
    if len(ordered) < 2:
        raise ValueError(f"Each multi-error combo must contain at least two distinct families, got {families!r}")
    if all(family == "measurement_error" for family in ordered):
        raise ValueError("A multi-error combo cannot contain only measurement errors.")
    return ordered


def combo_requires_structural_coupling(families: Sequence[str]) -> bool:
    return SCADA_STRUCTURAL_FAMILIES.issubset(set(canonicalize_combo(families)))


def coupling_metadata(families: Sequence[str]) -> dict[str, Any]:
    if combo_requires_structural_coupling(families):
        return {
            "coupling_mode": "curriculum_independent_components",
            "physically_coupled": False,
            "note": (
                "This sample combines independently generated structural components. "
                "Use for tool-use curriculum, not final physical concurrent-fault benchmarking."
            ),
        }
    return {
        "coupling_mode": "scada_coupled_or_separate_harmonic_channel",
        "physically_coupled": True,
        "note": (
            "The SCADA-visible part is coupled by construction, or the harmonic anomaly "
            "is represented in the separate harmonic measurement channel."
        ),
    }


def choose_base_family(families: Sequence[str]) -> str | None:
    canonical = canonicalize_combo(families)
    family_set = set(canonical)
    if SCADA_STRUCTURAL_FAMILIES.issubset(family_set):
        return "topology_error"
    if family_set == {"measurement_error", "harmonic_anomaly"}:
        return "clean_scada"
    return next((family for family in ERROR_PRIORITY if family in family_set and family != "measurement_error"), None)


def _stage_snapshot(
    *,
    case_path: str | None,
    remaining_families: Sequence[str],
    note: str,
    z_obs: Sequence[float] | None = None,
    case_path_policy: str | None = None,
    z_obs_policy: str | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "case_path": case_path,
        "remaining_families": list(remaining_families),
        "note": note,
    }
    if z_obs is not None:
        snapshot["z_obs"] = np.asarray(z_obs, dtype=float).tolist()
    if case_path_policy:
        snapshot["case_path_policy"] = case_path_policy
    if z_obs_policy:
        snapshot["z_obs_policy"] = z_obs_policy
    return snapshot


def _remaining_after_correction(families: Sequence[str], corrected_family: str) -> list[str]:
    correction_order = [
        "topology_error",
        "parameter_error",
        "measurement_error",
        "harmonic_anomaly",
    ]
    ordered = [family for family in correction_order if family in set(families)]
    if corrected_family not in ordered:
        return []
    idx = ordered.index(corrected_family)
    return ordered[idx + 1 :]


def parse_combo_spec(spec: str) -> tuple[str, ...]:
    parts = [part for part in str(spec).replace(",", "+").split("+") if part.strip()]
    return canonicalize_combo(parts)


def resolve_combos(combo_specs: Sequence[str] | None) -> list[tuple[str, ...]]:
    combos = [parse_combo_spec(spec) for spec in combo_specs] if combo_specs else [tuple(combo) for combo in DEFAULT_COMBOS]
    deduped: list[tuple[str, ...]] = []
    for combo in combos:
        if combo not in deduped:
            deduped.append(combo)
    return deduped


def _branch_info(ppc: Mapping[str, Any]) -> list[dict[str, Any]]:
    branch = ppc["branch"]
    return [
        {
            "i": int(i),
            "from_bus": int(branch[i, F_BUS]),
            "to_bus": int(branch[i, T_BUS]),
            "is_line": bool(branch[i, TAP] == 0.0 and branch[i, BR_STATUS] > 0),
        }
        for i in range(branch.shape[0])
    ]


def _apply_measurement_outlier(
    z_obs: np.ndarray,
    idx_map: Mapping[str, slice],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add only the gross measurement component, preserving the base error vector."""
    z_multi = np.asarray(z_obs, dtype=float).copy()
    subtype = str(rng.choice(["single_gross_outlier", "multi_gross_outliers"]))
    channel = str(rng.choice(MEASUREMENT_ORDER))
    sl = idx_map[channel]
    ch_sigma = {
        "Vm": DEFAULT_SIGMAS["vm"],
        "Pinj": DEFAULT_SIGMAS["inj"],
        "Qinj": DEFAULT_SIGMAS["inj"],
        "Pf": DEFAULT_SIGMAS["flow"],
        "Qf": DEFAULT_SIGMAS["flow"],
        "Pt": DEFAULT_SIGMAS["flow"],
        "Qt": DEFAULT_SIGMAS["flow"],
    }[channel]

    if subtype == "single_gross_outlier":
        idx0 = int(rng.integers(sl.start, sl.stop))
        amp = float(rng.choice([-1, 1]) * rng.uniform(5.0, 15.0) * ch_sigma)
        z_multi[idx0] += amp
        label = {
            "error_type": "measurement_error",
            "subtype": subtype,
            "channel": channel,
            "index": idx0,
            "amplitude": amp,
        }
    else:
        n_in_ch = sl.stop - sl.start
        count = int(rng.integers(2 if n_in_ch >= 2 else 1, min(5, n_in_ch) + 1))
        indices = rng.choice(np.arange(sl.start, sl.stop), size=count, replace=False)
        deltas = rng.choice([-1, 1], size=count) * rng.uniform(5.0, 15.0, size=count) * ch_sigma
        z_multi[indices] += deltas
        label = {
            "error_type": "measurement_error",
            "subtype": subtype,
            "channel": channel,
            "indices": [int(i) for i in indices],
            "amplitudes": [float(v) for v in deltas],
        }
    return z_multi, label


def _index_map_for_vector_length(vector_len: int, nb: int) -> dict[str, slice] | None:
    remainder = int(vector_len) - 3 * int(nb)
    if remainder < 0 or remainder % 4 != 0:
        return None
    return make_index_map(int(nb), remainder // 4)


def _project_measurement_label_to_snapshot_with_mapping(
    z_obs: Sequence[float],
    label: Mapping[str, Any],
    source_idx_map: Mapping[str, slice],
    target_idx_map: Mapping[str, slice],
) -> tuple[list[float], dict[str, Any] | None]:
    z_multi = np.asarray(z_obs, dtype=float).copy()
    channel = label.get("channel")
    if not isinstance(channel, str) or channel not in source_idx_map or channel not in target_idx_map:
        return z_multi.tolist(), None

    src_sl = source_idx_map[channel]
    dst_sl = target_idx_map[channel]
    original_indices: list[int] = []
    current_indices: list[int] = []

    def _project_index(index0: Any) -> int | None:
        try:
            offset = int(index0) - int(src_sl.start)
        except Exception:
            return None
        projected = int(dst_sl.start) + offset
        if projected < int(dst_sl.start) or projected >= int(dst_sl.stop) or projected >= len(z_multi):
            return None
        return projected

    if isinstance(label.get("index"), int) and label.get("amplitude") is not None:
        projected = _project_index(label["index"])
        if projected is not None:
            z_multi[projected] += float(label["amplitude"])
            original_indices.append(int(label["index"]))
            current_indices.append(int(projected))
    indices = label.get("indices")
    amplitudes = label.get("amplitudes")
    if isinstance(indices, list) and isinstance(amplitudes, list):
        for index0, amplitude in zip(indices, amplitudes):
            projected = _project_index(index0)
            if projected is not None:
                z_multi[projected] += float(amplitude)
                original_indices.append(int(index0))
                current_indices.append(int(projected))
    if not current_indices:
        return z_multi.tolist(), None
    source_vector_length = max(int(sl.stop) for sl in source_idx_map.values())
    return z_multi.tolist(), {
        "channel": channel,
        "index_space": "post_topology_correction",
        "source_vector_length": source_vector_length,
        "target_vector_length": int(len(z_multi)),
        "original_indices0": original_indices,
        "current_indices0": current_indices,
    }


def _project_measurement_label_to_snapshot(
    z_obs: Sequence[float],
    label: Mapping[str, Any],
    source_idx_map: Mapping[str, slice],
    target_idx_map: Mapping[str, slice],
) -> list[float]:
    projected, _mapping = _project_measurement_label_to_snapshot_with_mapping(
        z_obs,
        label,
        source_idx_map,
        target_idx_map,
    )
    return projected


def _make_parameter_context(
    rng: np.random.Generator,
    ppc_base: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    out_dir: Path,
    *,
    scans: int,
    load_scale_min: float,
    load_scale_max: float,
) -> dict[str, Any] | None:
    return make_parameter_error_record(
        rng,
        ppc_base,
        idx_map,
        load_scale_min,
        load_scale_max,
        DEFAULTS["r_err_range"],
        DEFAULTS["x_err_range"],
        scans,
        out_dir,
    )


def _copy_auxiliary_fields(target: dict[str, Any], family: str, component: Mapping[str, Any]) -> None:
    if family == "parameter_error":
        for key in ("z_scans", "initial_states", "parameter_error_case_path", "correction_case_path"):
            if key in component:
                target[key] = component[key]
    elif family == "topology_error":
        for key in ("z_true_full_model", "corrected_model_path"):
            if key in component:
                target[key] = component[key]
    elif family == "harmonic_anomaly":
        for key in ("harmonic_measurements", "harmonic_orders"):
            if key in component:
                target[key] = component[key]


def _make_component_factories(
    rng: np.random.Generator,
    ppc_base: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    out_dir: Path,
    *,
    scans: int,
    load_scale_min: float,
    load_scale_max: float,
) -> dict[str, Callable[[], dict[str, Any] | None]]:
    return {
        "parameter_error": lambda: _make_parameter_context(
            rng,
            ppc_base,
            idx_map,
            out_dir,
            scans=scans,
            load_scale_min=load_scale_min,
            load_scale_max=load_scale_max,
        ),
        "topology_error": lambda: make_topology_error_record(rng, ppc_base, idx_map, out_dir),
        "harmonic_anomaly": lambda: make_harmonic_anomaly_record(rng),
    }


def _physical_pt_coupling_metadata() -> dict[str, Any]:
    return {
        "coupling_mode": "physical_parameter_on_topology_corrected_model",
        "physically_coupled": True,
        "note": (
            "The initial node-breaker operating snapshot contains both the breaker-status "
            "topology fault and a perturbed in-service line. After topology correction, "
            "the verification snapshot keeps the same parameter fault on the corrected "
            "bus-branch model so parameter correction has real residual evidence."
        ),
    }


def _import_nodebreaker_pp14():
    try:
        from . import nodebreaker_pp14 as nb  # type: ignore
    except Exception:
        import os as _os
        import sys as _sys

        _sys.path.append(_os.path.dirname(__file__))
        import nodebreaker_pp14 as nb  # type: ignore
    return nb


def _import_nb_to_matpower():
    try:
        from . import nb_to_matpower as nb2mp  # type: ignore
    except Exception:
        import os as _os
        import sys as _sys

        _sys.path.append(_os.path.dirname(__file__))
        import nb_to_matpower as nb2mp  # type: ignore
    return nb2mp


def _run_pandapower_pf(net: Any, *, init: str = "dc") -> bool:
    try:
        import pandapower as pp  # type: ignore

        pp.runpp(net, init=init)
        return True
    except Exception:
        return False


def _solve_ppc_power_flow(ppc: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        results, success = runpf(deepcopy(dict(ppc)), ppopt)
    except Exception:
        return None
    return results if success else None


def _ordered_branch_names_for_exported_ppc(net_bb: Any, ppc: Mapping[str, Any]) -> list[str] | None:
    try:
        raw_names = [str(row["name"]) for _, row in net_bb.line.sort_index().iterrows()]
        raw_names += [str(row["name"]) for _, row in net_bb.trafo.sort_index().iterrows()]
    except Exception:
        return None

    present = set(raw_names)
    ordered: list[str] = []
    used: set[str] = set()
    for name in CASE14_STANDARD_BRANCH_NAMES:
        if name in present:
            ordered.append(name)
            used.add(name)
    for name in raw_names:
        if name not in used:
            ordered.append(name)
            used.add(name)
    if len(ordered) != int(ppc["branch"].shape[0]):
        return None
    return ordered


def _build_topology_corrected_ppc(
    net: Any,
    out_dir: Path,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[str]] | None:
    try:
        nb2mp = _import_nb_to_matpower()
        net_bb, _ = nb2mp.topology_processed_busbranch(net)
        if hasattr(nb2mp, "_prune_dangling_buses"):
            nb2mp._prune_dangling_buses(net_bb)
        _run_pandapower_pf(net_bb, init="flat")

        tmp_export_path = out_dir / "_tmp_topology_exports" / f"tmp_topology_{rng.integers(1e12)}.m"
        tmp_export_path.parent.mkdir(parents=True, exist_ok=True)
        ppc_full = nb2mp.export_to_matpower(net_bb, filename_mat=str(tmp_export_path))

        try:
            res_vm = net_bb.res_bus.sort_index()["vm_pu"].values
            res_va = net_bb.res_bus.sort_index()["va_degree"].values
            if len(res_vm) == ppc_full["bus"].shape[0]:
                ppc_full["bus"][:, VM] = res_vm
                ppc_full["bus"][:, VA] = res_va
        except Exception:
            pass

        branch_names = _ordered_branch_names_for_exported_ppc(net_bb, ppc_full)
        if branch_names is None:
            return None
        return ppc_full, branch_names
    except Exception as exc:
        print(f"[warn] Failed to build coupled P+T topology model: {exc}")
        return None


def _sample_factor(rng: np.random.Generator, ranges: Any) -> float:
    if isinstance(ranges, list):
        if not ranges:
            return 1.0
        selected = ranges[int(rng.integers(len(ranges)))]
        return float(rng.uniform(float(selected[0]), float(selected[1])))
    return float(rng.uniform(float(ranges[0]), float(ranges[1])))


def _draw_parameter_perturbation(
    rng: np.random.Generator,
    r_range: Any,
    x_range: Any,
) -> tuple[str, float, float]:
    subtype = str(rng.choice(["R", "X", "RX"]))
    r_factor = _sample_factor(rng, r_range) if subtype in {"R", "RX"} else 1.0
    x_factor = _sample_factor(rng, x_range) if subtype in {"X", "RX"} else 1.0
    return subtype, r_factor, x_factor


def _matching_nodebreaker_line_name(line_idx: Mapping[str, int], branch_name: str) -> str | None:
    if branch_name in line_idx:
        return branch_name
    if not branch_name.startswith("line_") or "-" not in branch_name:
        return None
    left, right = branch_name[len("line_") :].split("-", 1)
    reversed_name = f"line_{right}-{left}"
    return reversed_name if reversed_name in line_idx else None


def _candidate_coupled_parameter_lines(
    ppc_topology: Mapping[str, Any],
    branch_names: Sequence[str],
    line_idx: Mapping[str, int],
) -> list[tuple[int, str, str]]:
    branch = ppc_topology["branch"]
    if len(branch_names) != int(branch.shape[0]):
        return []

    mask_line = branch_line_mask(ppc_topology)
    candidates: list[tuple[int, str, str]] = []
    eps = 1e-9
    for row0, branch_name in enumerate(branch_names):
        if row0 >= int(branch.shape[0]) or not str(branch_name).startswith("line_"):
            continue
        nb_line_name = _matching_nodebreaker_line_name(line_idx, str(branch_name))
        if nb_line_name is None:
            continue
        if (
            bool(mask_line[row0])
            and abs(float(branch[row0, BR_R])) > eps
            and abs(float(branch[row0, BR_X])) > eps
        ):
            candidates.append((int(row0), str(branch_name), nb_line_name))
    return candidates


def _apply_parameter_fault_to_ppc(
    ppc_nominal: Mapping[str, Any],
    line_row0: int,
    r_factor: float,
    x_factor: float,
) -> dict[str, Any]:
    ppc_faulty = deepcopy(dict(ppc_nominal))
    branch = ppc_faulty["branch"]
    branch[line_row0, BR_R] = max(1e-6, float(branch[line_row0, BR_R]) * float(r_factor))
    branch[line_row0, BR_X] = max(1e-6, float(branch[line_row0, BR_X]) * float(x_factor))
    return ppc_faulty


def _apply_parameter_fault_to_nodebreaker_line(
    net: Any,
    line_name: str,
    line_idx: Mapping[str, int],
    r_factor: float,
    x_factor: float,
) -> bool:
    idx = line_idx.get(line_name)
    if idx is None:
        return False
    try:
        net.line.at[idx, "r_ohm_per_km"] = max(
            1e-9,
            float(net.line.at[idx, "r_ohm_per_km"]) * float(r_factor),
        )
        net.line.at[idx, "x_ohm_per_km"] = max(
            1e-9,
            float(net.line.at[idx, "x_ohm_per_km"]) * float(x_factor),
        )
        return True
    except Exception:
        return False


def _make_parameter_scan_context(
    solved_faulty: Mapping[str, Any],
    idx_map_full: Mapping[str, slice],
    rng: np.random.Generator,
    scans: int,
) -> tuple[list[list[float]], list[list[float]]]:
    z_true = compute_measurements_pu(solved_faulty)
    sigma_r = sigma_vector(idx_map_full, DEFAULT_SIGMAS)
    z_scans: list[list[float]] = []
    initial_states: list[list[float]] = []
    for _ in range(int(scans)):
        z_scan = (z_true + rng.standard_normal(z_true.shape[0]) * sigma_r).astype(float).tolist()
        z_scans.append(z_scan)
        initial_states.append(make_initial_state_guess(solved_faulty["bus"], rng))
    return z_scans, initial_states


def _make_physically_coupled_parameter_topology_record(
    rng: np.random.Generator,
    ppc_base: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    out_dir: Path,
    families: Sequence[str],
    *,
    scans: int,
) -> dict[str, Any] | None:
    families = canonicalize_combo(families)
    family_set = set(families)
    try:
        nb = _import_nodebreaker_pp14()
        status_map, topology_label = _choose_random_cb_open(rng, None)
        net, _sec_bus, _cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)
    except Exception:
        return None

    if not _run_pandapower_pf(net, init="dc"):
        return None

    topology_model = _build_topology_corrected_ppc(net, out_dir, rng)
    if topology_model is None:
        return None
    ppc_topology_nominal, topology_branch_names = topology_model

    candidates = _candidate_coupled_parameter_lines(ppc_topology_nominal, topology_branch_names, line_idx)
    if not candidates:
        return None
    line_row0, topology_branch_name, nb_line_name = candidates[int(rng.integers(len(candidates)))]
    subtype, r_factor, x_factor = _draw_parameter_perturbation(
        rng,
        DEFAULTS["r_err_range"],
        DEFAULTS["x_err_range"],
    )

    if not _apply_parameter_fault_to_nodebreaker_line(net, nb_line_name, line_idx, r_factor, x_factor):
        return None
    if not _run_pandapower_pf(net, init="dc"):
        return None

    z_true_initial = _nb_to_operator_z(net, line_idx, trafo_idx, ppc_base)
    z_obs_initial = z_true_initial + base_gaussian_noise(z_true_initial, idx_map, DEFAULT_SIGMAS, rng)

    ppc_param_faulty = _apply_parameter_fault_to_ppc(
        ppc_topology_nominal,
        line_row0,
        r_factor,
        x_factor,
    )
    solved_param_faulty = _solve_ppc_power_flow(ppc_param_faulty)
    solved_nominal = _solve_ppc_power_flow(ppc_topology_nominal)
    if solved_param_faulty is None or solved_nominal is None:
        return None

    z_true_param_faulty = compute_measurements_pu(solved_param_faulty)
    z_true_nominal = compute_measurements_pu(solved_nominal)
    idx_map_full = make_index_map(
        solved_param_faulty["bus"].shape[0],
        solved_param_faulty["branch"].shape[0],
    )
    idx_map_nominal = make_index_map(
        solved_nominal["bus"].shape[0],
        solved_nominal["branch"].shape[0],
    )
    if max(sl.stop for sl in idx_map_full.values()) != max(sl.stop for sl in idx_map_nominal.values()):
        return None
    sigma_full = sigma_vector(idx_map_full, DEFAULT_SIGMAS)
    signal = (z_true_param_faulty - z_true_nominal) / np.maximum(sigma_full, 1e-12)
    # The coupled P+T curriculum should not leave the parameter step as a
    # purely symbolic action. Reject weak draws where the remaining parameter
    # effect after topology correction is at the noise floor.
    if float(np.sum(signal * signal) / max(1, signal.shape[0])) < 0.5:
        return None
    shared_noise = 0.5 * base_gaussian_noise(z_true_nominal, idx_map_full, DEFAULT_SIGMAS, rng)
    z_obs_param_faulty = z_true_param_faulty + shared_noise
    z_obs_nominal = z_true_nominal + shared_noise

    z_scans, initial_states = _make_parameter_scan_context(
        solved_param_faulty,
        idx_map_full,
        rng,
        scans,
    )

    corrected_model_basename = f"case_topology_corrected_pt_{rng.integers(1e12)}"
    corrected_model_path = write_ppc_as_matpower_m(
        solved_nominal,
        out_dir / "models_topology" / f"{corrected_model_basename}.m",
        corrected_model_basename,
    )
    parameter_case_basename = f"case_param_topology_err_{rng.integers(1e12)}"
    parameter_case_path = write_ppc_as_matpower_m(
        solved_param_faulty,
        out_dir / "cases_parameter_error" / f"{parameter_case_basename}.m",
        parameter_case_basename,
    )

    parameter_label = {
        "error_type": "parameter_error",
        "subtype": subtype,
        "line_row": int(line_row0),
        "from_bus": int(solved_param_faulty["branch"][line_row0, F_BUS]),
        "to_bus": int(solved_param_faulty["branch"][line_row0, T_BUS]),
        "r_factor": float(r_factor),
        "x_factor": float(x_factor),
        "topology_corrected_branch_name": topology_branch_name,
        "nodebreaker_line_name": nb_line_name,
        "index_space": "post_topology_correction",
    }
    topology_label = dict(topology_label)
    topology_label["error_type"] = "topology_error"

    labels: dict[str, dict[str, Any]] = {
        "parameter_error": parameter_label,
        "topology_error": topology_label,
    }
    z_obs = np.asarray(z_obs_initial, dtype=float)

    if "measurement_error" in family_set:
        z_obs, measurement_label = _apply_measurement_outlier(z_obs, idx_map, rng)
        labels["measurement_error"] = measurement_label

    post_topology_z = np.asarray(z_obs_param_faulty, dtype=float).tolist()
    post_parameter_z = np.asarray(z_obs_nominal, dtype=float).tolist()
    post_measurement_z = np.asarray(z_obs_nominal, dtype=float).tolist()

    if "measurement_error" in labels:
        projected_topology_z, topology_projection = _project_measurement_label_to_snapshot_with_mapping(
            post_topology_z,
            labels["measurement_error"],
            idx_map,
            idx_map_full,
        )
        projected_parameter_z, parameter_projection = _project_measurement_label_to_snapshot_with_mapping(
            post_parameter_z,
            labels["measurement_error"],
            idx_map,
            idx_map_full,
        )
        if topology_projection is None or parameter_projection is None:
            return None
        post_topology_z = projected_topology_z
        post_parameter_z = projected_parameter_z
        labels["measurement_error"]["index_spaces"] = {
            "original_indices0": topology_projection["original_indices0"],
            "post_topology_correction_indices0": topology_projection["current_indices0"],
            "post_topology_correction": topology_projection,
            "post_parameter_correction_indices0": parameter_projection["current_indices0"],
            "post_parameter_correction": parameter_projection,
        }

    verification_snapshots: dict[str, Any] = {
        "post_topology_correction": _stage_snapshot(
            case_path=str(corrected_model_path),
            z_obs=post_topology_z,
            remaining_families=_remaining_after_correction(families, "topology_error"),
            note=(
                "Topology model corrected; the same physical line-parameter fault remains "
                "on the corrected bus-branch model."
            ),
        ),
        "post_parameter_correction": _stage_snapshot(
            case_path=str(corrected_model_path),
            z_obs=post_parameter_z,
            remaining_families=_remaining_after_correction(families, "parameter_error"),
            note="Parameter model corrected on the topology-corrected network.",
        ),
    }
    if "measurement_error" in labels:
        verification_snapshots["post_topology_correction"]["measurement_index_projection"] = (
            labels["measurement_error"]["index_spaces"]["post_topology_correction"]
        )
        verification_snapshots["post_measurement_correction"] = _stage_snapshot(
            case_path=str(corrected_model_path),
            z_obs=post_measurement_z,
            remaining_families=_remaining_after_correction(families, "measurement_error"),
            note="Gross measurement component removed after structural corrections.",
        )

    component_op_points: dict[str, Any] = {
        "parameter_error": {
            "index_space": "post_topology_correction",
            "measurement_count": int(len(z_obs_param_faulty)),
        },
        "topology_error": {"load_scale": 1.0},
    }

    auxiliary_fields: dict[str, Any] = {}
    if "harmonic_anomaly" in family_set:
        harmonic_component = make_harmonic_anomaly_record(rng)
        if harmonic_component is None:
            return None
        labels["harmonic_anomaly"] = dict(harmonic_component.get("label", {}))
        labels["harmonic_anomaly"]["error_type"] = "harmonic_anomaly"
        component_op_points["harmonic_anomaly"] = harmonic_component.get("op_point", {})
        _copy_auxiliary_fields(auxiliary_fields, "harmonic_anomaly", harmonic_component)

    primary_family = next(family for family in ERROR_PRIORITY if family in families)
    ordered_errors = []
    for family in families:
        item = dict(labels[family])
        item["error_type"] = family
        ordered_errors.append(item)

    record: dict[str, Any] = {
        "id": f"multi_{combo_key(families).replace('+', '_')}_{rng.integers(1_000_000_000_000)}",
        "scenario": "multi_error",
        "z_true": z_true_initial.astype(float).tolist(),
        "z_obs": z_obs.astype(float).tolist(),
        "label": {
            "error_type": "multi_error",
            "combo": combo_key(families),
            "error_families": list(families),
            "primary_error_family": primary_family,
            "errors": ordered_errors,
            **_physical_pt_coupling_metadata(),
        },
        "op_point": {
            "base_family": "coupled_parameter_topology",
            "component_op_points": component_op_points,
        },
        "verification_snapshots": verification_snapshots,
        "z_true_full_model": z_obs_param_faulty.astype(float).tolist(),
        "corrected_model_path": str(corrected_model_path),
        "z_scans": z_scans,
        "initial_states": initial_states,
        "parameter_error_case_path": str(parameter_case_path),
        "correction_case_path": str(parameter_case_path),
        **auxiliary_fields,
    }
    return record


def make_multi_error_record(
    rng: np.random.Generator,
    ppc_base: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    out_dir: Path,
    families: Sequence[str],
    *,
    scans: int,
    load_scale_min: float,
    load_scale_max: float,
    mode: str = "physical",
) -> dict[str, Any] | None:
    factories = _make_component_factories(
        rng,
        ppc_base,
        idx_map,
        out_dir,
        scans=scans,
        load_scale_min=load_scale_min,
        load_scale_max=load_scale_max,
    )
    components: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}

    families = canonicalize_combo(families)
    if mode == "physical" and combo_requires_structural_coupling(families):
        return _make_physically_coupled_parameter_topology_record(
            rng,
            ppc_base,
            idx_map,
            out_dir,
            families,
            scans=scans,
        )
    primary_family = next(family for family in ERROR_PRIORITY if family in families)
    base_family = choose_base_family(families)
    if base_family is None:
        return None

    for family in families:
        if family == "measurement_error":
            continue
        component = factories[family]()
        if component is None:
            return None
        components[family] = component
        labels[family] = dict(component.get("label", {}))
        labels[family]["error_type"] = family

    if base_family == "clean_scada":
        base_component = make_no_error_record(
            rng,
            ppc_base,
            idx_map,
            load_scale_min,
            load_scale_max,
        )
        if base_component is None:
            return None
    else:
        base_component = components[base_family]
    z_true = np.asarray(base_component["z_true"], dtype=float)
    z_obs = np.asarray(base_component["z_obs"], dtype=float)
    z_obs_before_measurement_outlier = z_obs.copy()

    if "measurement_error" in families:
        z_obs, measurement_label = _apply_measurement_outlier(z_obs, idx_map, rng)
        labels["measurement_error"] = measurement_label

    verification_snapshots: dict[str, Any] = {}
    if "measurement_error" in families:
        verification_snapshots["post_measurement_correction"] = _stage_snapshot(
            case_path=None,
            z_obs=z_obs_before_measurement_outlier,
            case_path_policy="preserve_current_case",
            remaining_families=_remaining_after_correction(families, "measurement_error"),
            note="Gross measurement component removed; other active families may remain.",
        )
    if "parameter_error" in families:
        parameter_component = components.get("parameter_error", {})
        parameter_case_path = (
            parameter_component.get("parameter_error_case_path")
            or parameter_component.get("correction_case_path")
        )
        verification_snapshots["post_parameter_correction"] = _stage_snapshot(
            case_path=str(parameter_case_path) if parameter_case_path else None,
            z_obs_policy="preserve_current_z_obs",
            remaining_families=_remaining_after_correction(families, "parameter_error"),
            note="Parameter model corrected; preserve any uncorrected measurement components.",
        )
    if "topology_error" in families:
        topology_component = components.get("topology_error", {})
        if topology_component.get("corrected_model_path") is not None:
            topology_z_obs = topology_component.get("z_true_full_model")
            if topology_z_obs is not None and "measurement_error" in labels:
                target_idx_map = _index_map_for_vector_length(
                    len(topology_z_obs),
                    int(ppc_base["bus"].shape[0]),
                )
                if target_idx_map is None:
                    return None
                topology_z_obs, projection = _project_measurement_label_to_snapshot_with_mapping(
                    topology_z_obs,
                    labels["measurement_error"],
                    idx_map,
                    target_idx_map,
                )
                if projection is None:
                    return None
                labels["measurement_error"]["index_spaces"] = {
                    "original_indices0": projection["original_indices0"],
                    "post_topology_correction_indices0": projection["current_indices0"],
                    "post_topology_correction": projection,
                }
            elif "measurement_error" in labels:
                return None
            verification_snapshots["post_topology_correction"] = _stage_snapshot(
                case_path=topology_component.get("corrected_model_path"),
                z_obs=topology_z_obs,
                z_obs_policy=None if topology_z_obs is not None else "preserve_current_z_obs",
                remaining_families=_remaining_after_correction(families, "topology_error"),
                note="Topology model corrected; preserve remaining data/model faults.",
            )
            if (
                "measurement_error" in labels
                and isinstance(labels["measurement_error"].get("index_spaces"), Mapping)
            ):
                verification_snapshots["post_topology_correction"]["measurement_index_projection"] = (
                    labels["measurement_error"]["index_spaces"]["post_topology_correction"]
                )

    ordered_errors = []
    for family in families:
        item = dict(labels[family])
        item["error_type"] = family
        ordered_errors.append(item)
    component_op_points = {
        family: component.get("op_point", {})
        for family, component in components.items()
    }
    if base_family == "clean_scada":
        component_op_points["clean_scada"] = base_component.get("op_point", {})

    record: dict[str, Any] = {
        "id": f"multi_{combo_key(families).replace('+', '_')}_{rng.integers(1_000_000_000_000)}",
        "scenario": "multi_error",
        "z_true": z_true.astype(float).tolist(),
        "z_obs": z_obs.astype(float).tolist(),
        "label": {
            "error_type": "multi_error",
            "combo": combo_key(families),
            "error_families": list(families),
            "primary_error_family": primary_family,
            "errors": ordered_errors,
            **coupling_metadata(families),
        },
        "op_point": {
            "base_family": base_family,
            "component_op_points": component_op_points,
        },
    }
    if verification_snapshots:
        record["verification_snapshots"] = verification_snapshots
    for family, component in components.items():
        _copy_auxiliary_fields(record, family, component)
    return record


def generate_dataset(
    *,
    case_name: str,
    per_combo: int,
    total: int | None = None,
    out_dir: str,
    seed: int,
    scans: int,
    attempt_mult: int,
    load_scale_min: float,
    load_scale_max: float,
    combos: Sequence[Sequence[str]] | None = None,
    mode: str = "physical",
) -> None:
    if case_name != "14":
        raise ValueError("V1 multi-error generation supports IEEE-14 only.")
    combo_list = [canonicalize_combo(combo) for combo in combos] if combos is not None else resolve_combos(None)
    if any("harmonic_anomaly" in combo for combo in combo_list) and not HARMONICS_AVAILABLE:
        raise RuntimeError("Harmonics modules are required for configured combos containing harmonic_anomaly.")

    rng = np.random.default_rng(seed)
    out = Path(os.path.abspath(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    ppc_base = load_case(case_name)
    idx_map = make_index_map(ppc_base["bus"].shape[0], ppc_base["branch"].shape[0])
    line_mask = branch_line_mask(ppc_base)

    requested_counts = distribute_counts(combo_list, total=total, per_combo=per_combo)

    meta = {
        "case": f"case{case_name}",
        "baseMVA": float(ppc_base["baseMVA"]),
        "nb": int(ppc_base["bus"].shape[0]),
        "nl": int(ppc_base["branch"].shape[0]),
        "index_map": {k: [int(v.start), int(v.stop)] for k, v in idx_map.items()},
        "measurement_order": MEASUREMENT_ORDER,
        "branch_info": _branch_info(ppc_base),
        "sigmas": DEFAULT_SIGMAS,
        "scenarios_emitted": ["multi_error"],
        "omitted_scenarios": ["three_phase_imbalance"],
        "multi_error": {
            "type": "custom_combos",
            "default_combo_set": combo_list == [tuple(combo) for combo in DEFAULT_COMBOS],
            "min_errors_per_snapshot": min(len(combo) for combo in combo_list),
            "max_errors_per_snapshot": max(len(combo) for combo in combo_list),
            "error_count_by_combo": {combo_key(combo): len(combo) for combo in combo_list},
            "combos": [combo_key(combo) for combo in combo_list],
            "combo_families": {combo_key(combo): list(combo) for combo in combo_list},
            "line_candidate_count": int(np.sum(line_mask)),
            "mode": mode,
        },
        "requested_counts": {combo_key(combo): int(count) for combo, count in requested_counts.items()},
        "note": (
            "Multi-error records preserve the scalar primary_error_family "
            "and include label.errors for full multi-label training/evaluation."
        ),
    }

    samples_path = out / "samples.jsonl"
    written_counts: dict[str, int] = {}
    with samples_path.open("w", encoding="utf-8") as handle:
        for combo in combo_list:
            key = combo_key(combo)
            target_count = int(requested_counts[combo])
            written = 0
            attempts = 0
            max_attempts = max(target_count * int(attempt_mult), target_count + 25)
            with tqdm(total=target_count, desc=f"Multi-error {key}") as pbar:
                while written < target_count and attempts < max_attempts:
                    attempts += 1
                    record = make_multi_error_record(
                        rng,
                        ppc_base,
                        idx_map,
                        out,
                        combo,
                        scans=scans,
                        load_scale_min=load_scale_min,
                        load_scale_max=load_scale_max,
                        mode=mode,
                    )
                    if record is None:
                        continue
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    pbar.update(1)
            if written < target_count:
                print(f"[warn] {key}: wrote {written}/{target_count} after {attempts} attempts.")
            written_counts[key] = int(written)

    meta["written_counts"] = written_counts
    with (out / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"\nWrote multi-error dataset to: {out}")
    print(f"Samples: {samples_path}")
    print(f"Meta: {out / 'meta.json'}")


def distribute_counts(
    combo_list: Sequence[Sequence[str]],
    *,
    total: int | None,
    per_combo: int,
) -> dict[tuple[str, ...], int]:
    combos = [canonicalize_combo(combo) for combo in combo_list]
    if total is None:
        return {combo: int(per_combo) for combo in combos}
    if total <= 0:
        raise ValueError("--total must be positive when provided.")
    base = int(total) // len(combos)
    remainder = int(total) % len(combos)
    return {
        combo: base + (1 if idx < remainder else 0)
        for idx, combo in enumerate(combos)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-error PSSE measurements")
    parser.add_argument("--case", choices=["14"], default="14")
    parser.add_argument("--per-combo", type=int, default=500)
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total number of records across all selected combos. Overrides --per-combo when set.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=None,
        help=(
            "Multi-error combo to generate, e.g. 'measurement+parameter+topology'. "
            "May be repeated. Defaults to the six pairwise measurement/parameter/topology/harmonic combos."
        ),
    )
    parser.add_argument("--out", default="artifacts/measurements/out_measurements_multi_error")
    parser.add_argument("--seed", type=int, default=1442)
    parser.add_argument("--scans", type=int, default=8)
    parser.add_argument("--attempt-mult", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["curriculum", "physical"],
        default="physical",
        help=(
            "curriculum: allow independently generated component combinations; "
            "physical: require SCADA-visible parameter+topology faults to keep real residual evidence "
            "across topology and parameter correction snapshots. Defaults to physical."
        ),
    )
    parser.add_argument("--ls-min", type=float, default=DEFAULTS["load_scale_min"])
    parser.add_argument("--ls-max", type=float, default=DEFAULTS["load_scale_max"])
    args = parser.parse_args()

    generate_dataset(
        case_name=args.case,
        per_combo=args.per_combo,
        total=args.total,
        out_dir=args.out,
        seed=args.seed,
        scans=args.scans,
        attempt_mult=args.attempt_mult,
        load_scale_min=args.ls_min,
        load_scale_max=args.ls_max,
        combos=resolve_combos(args.combo),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
