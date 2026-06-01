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
    branch_line_mask,
    load_case,
    make_harmonic_anomaly_record,
    make_index_map,
    make_parameter_error_record,
    make_topology_error_record,
)
from pypower.idx_brch import BR_STATUS, F_BUS, TAP, T_BUS  # noqa: E402


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


def _project_measurement_label_to_snapshot(
    z_obs: Sequence[float],
    label: Mapping[str, Any],
    source_idx_map: Mapping[str, slice],
    target_idx_map: Mapping[str, slice],
) -> list[float]:
    z_multi = np.asarray(z_obs, dtype=float).copy()
    channel = label.get("channel")
    if not isinstance(channel, str) or channel not in source_idx_map or channel not in target_idx_map:
        return z_multi.tolist()

    src_sl = source_idx_map[channel]
    dst_sl = target_idx_map[channel]

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
    indices = label.get("indices")
    amplitudes = label.get("amplitudes")
    if isinstance(indices, list) and isinstance(amplitudes, list):
        for index0, amplitude in zip(indices, amplitudes):
            projected = _project_index(index0)
            if projected is not None:
                z_multi[projected] += float(amplitude)
    return z_multi.tolist()


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
    mode: str = "curriculum",
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
        raise NotImplementedError(
            "Physical parameter+topology coupling is not implemented yet. "
            "Use --mode curriculum for tool-use traces, or exclude P+T combos."
        )
    primary_family = next(family for family in ERROR_PRIORITY if family in families)
    base_family = next((family for family in ERROR_PRIORITY if family in families and family != "measurement_error"), None)
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

    base_component = components[base_family]
    z_true = np.asarray(base_component["z_true"], dtype=float)
    z_obs = np.asarray(base_component["z_obs"], dtype=float)
    z_obs_before_measurement_outlier = z_obs.copy()

    if "measurement_error" in families:
        z_obs, measurement_label = _apply_measurement_outlier(z_obs, idx_map, rng)
        labels["measurement_error"] = measurement_label

    ordered_errors = []
    for family in families:
        item = dict(labels[family])
        item["error_type"] = family
        ordered_errors.append(item)

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
                if target_idx_map is not None:
                    topology_z_obs = _project_measurement_label_to_snapshot(
                        topology_z_obs,
                        labels["measurement_error"],
                        idx_map,
                        target_idx_map,
                    )
            verification_snapshots["post_topology_correction"] = _stage_snapshot(
                case_path=topology_component.get("corrected_model_path"),
                z_obs=topology_z_obs,
                z_obs_policy=None if topology_z_obs is not None else "preserve_current_z_obs",
                remaining_families=_remaining_after_correction(families, "topology_error"),
                note="Topology model corrected; preserve remaining data/model faults.",
            )

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
            "component_op_points": {
                family: component.get("op_point", {})
                for family, component in components.items()
            },
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
    mode: str = "curriculum",
) -> None:
    if case_name != "14":
        raise ValueError("V1 multi-error generation supports IEEE-14 only.")
    combo_list = [canonicalize_combo(combo) for combo in combos] if combos is not None else resolve_combos(None)
    if mode == "physical":
        unsupported = [combo_key(combo) for combo in combo_list if combo_requires_structural_coupling(combo)]
        if unsupported:
            raise NotImplementedError(
                "Physical parameter+topology coupling is not implemented yet. "
                f"Unsupported physical combos: {', '.join(unsupported)}. "
                "Use --mode curriculum for tool-use traces, or exclude P+T combos."
            )
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
    parser.add_argument("--out", default="out_measurements_multi_error")
    parser.add_argument("--seed", type=int, default=1442)
    parser.add_argument("--scans", type=int, default=8)
    parser.add_argument("--attempt-mult", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["curriculum", "physical"],
        default="curriculum",
        help=(
            "curriculum: allow independently generated component combinations; "
            "physical: require all SCADA-visible faults to come from one coupled physical snapshot."
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
