#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IEEE_14_OpenDSS.constants import (  # noqa: E402
    BRANCH_ORDER,
    BUS_ORDER,
    IEEE14_GENERATOR_DISPATCH_KW,
    IEEE14_GENERATOR_VOLTAGE_PU,
    IEEE14_LOAD_BASE_KW,
    IEEE14_OPERATING_POINT_KEYS,
)
from IEEE_14_OpenDSS.export_measurement_series import (  # noqa: E402
    _compile_and_solve,
    element_pq_3ph_per_terminal,
)
from three_phase_nlm.hif_operating_point import (  # noqa: E402
    canonicalize_ieee14_operating_point,
)
from three_phase_nlm.hif_parameter_estimator import (  # noqa: E402
    _line_tokens,
    _resolve_model_dir,
    _simulate_candidate,
)
from three_phase_nlm.ieee14_adapter import branch_info_for_row0  # noqa: E402
from three_phase_nlm.branch_current_analysis import (  # noqa: E402
    BRANCH_CURRENT_CHANNEL,
    branch_current_rows_valid,
)


TRANSFORMER_ROWS = [
    index for index, element in enumerate(BRANCH_ORDER) if element.lower().startswith("transformer.")
]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def _finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except Exception:
        return False


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _canonical_schema_matches(op_point: Any) -> bool:
    if not isinstance(op_point, Mapping) or set(op_point) != set(IEEE14_OPERATING_POINT_KEYS):
        return False
    if set(op_point.get("bus_load_scales", {})) != set(IEEE14_LOAD_BASE_KW):
        return False
    if set(op_point.get("generator_dispatch_kw", {})) != set(IEEE14_GENERATOR_DISPATCH_KW):
        return False
    if set(op_point.get("voltage_setpoints_pu", {})) != set(IEEE14_GENERATOR_VOLTAGE_PU):
        return False
    try:
        return canonicalize_ieee14_operating_point(op_point) == dict(op_point)
    except Exception:
        return False


def _phasors_valid(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != len(BUS_ORDER):
        return False
    buses = []
    for row in value:
        if not isinstance(row, Mapping):
            return False
        buses.append(str(row.get("bus", "")).lower())
        if not _finite_vector(row.get("vln_pu"), 3) or not _finite_vector(row.get("ang_deg"), 3):
            return False
    return buses == BUS_ORDER


def _operating_point_vector(op_point: Mapping[str, Any]) -> list[float]:
    return [
        float(op_point["load_scale"]),
        *[float(op_point["bus_load_scales"][name]) for name in IEEE14_LOAD_BASE_KW],
        *[float(op_point["generator_dispatch_kw"][name]) for name in IEEE14_GENERATOR_DISPATCH_KW],
        *[float(op_point["voltage_setpoints_pu"][name]) for name in IEEE14_GENERATOR_VOLTAGE_PU],
        float(op_point["source_voltage_pu"]),
    ]


def _centered_rank(rows: Sequence[Sequence[float]]) -> int:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not singular.size:
        return 0
    tolerance = max(float(singular[0]) * 1e-8, 1e-10)
    return int(np.sum(singular > tolerance))


def _branch_endpoints(meta: Mapping[str, Any]) -> list[tuple[int, int]]:
    hif = meta.get("hif") if isinstance(meta.get("hif"), Mapping) else {}
    raw = meta.get("branch_info", hif.get("branch_info"))
    if not isinstance(raw, list):
        raise ValueError("meta.hif.branch_info is required")
    by_index = {
        int(item["i"]): (int(item["from_bus"]), int(item["to_bus"]))
        for item in raw
        if isinstance(item, Mapping)
    }
    if set(by_index) != set(range(len(BRANCH_ORDER))):
        raise ValueError("meta.hif.branch_info does not cover all IEEE-14 branch rows")
    return [by_index[index] for index in range(len(BRANCH_ORDER))]


def _measurement_physics(
    vectors: Sequence[Sequence[float]],
    endpoints: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    transformer_ratios: list[float] = []
    secondary_residuals: list[float] = []
    other_residuals: list[float] = []
    for vector in vectors:
        values = np.asarray(vector, dtype=float)
        pinj = values[14:28]
        qinj = values[28:42]
        pf = values[42:62]
        qf = values[62:82]
        pt = values[82:102]
        qt = values[102:122]
        terminal_p = np.zeros(14, dtype=float)
        terminal_q = np.zeros(14, dtype=float)
        for index, (from_bus, to_bus) in enumerate(endpoints):
            terminal_p[from_bus - 1] += pf[index]
            terminal_q[from_bus - 1] += qf[index]
            terminal_p[to_bus - 1] += pt[index]
            terminal_q[to_bus - 1] += qt[index]
        residuals = np.hypot(pinj - terminal_p, qinj - terminal_q)
        secondary_residuals.extend(float(residuals[index - 1]) for index in (6, 7, 9))
        other_residuals.extend(
            float(residuals[index - 1])
            for index in (1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14)
        )
        for index in TRANSFORMER_ROWS:
            denominator = max(abs(float(pf[index])), 1e-12)
            transformer_ratios.append(abs(float(pt[index])) / denominator)

    ratio_median = statistics.median(transformer_ratios) if transformer_ratios else None
    secondary_median = statistics.median(secondary_residuals) if secondary_residuals else None
    other_median = statistics.median(other_residuals) if other_residuals else None
    balance_limit = max(0.03, 2.0 * float(other_median or 0.0))
    return {
        "vector_count": len(vectors),
        "transformer_abs_pt_over_abs_pf_median": ratio_median,
        "transformer_signature_passed": bool(
            ratio_median is not None
            and abs(float(ratio_median) - 1.0) <= 0.10
            and abs(float(ratio_median) - 2.0 / 3.0) > 0.10
        ),
        "secondary_median_balance_residual_pu": secondary_median,
        "other_median_balance_residual_pu": other_median,
        "balance_limit_pu": balance_limit,
        "bus_balance_passed": bool(
            secondary_median is not None and float(secondary_median) <= balance_limit
        ),
    }


def _engine_transformer_loss_check() -> dict[str, Any]:
    import opendssdirect as dss

    _compile_and_solve(str(REPO_ROOT / "IEEE_14_OpenDSS"))
    rows = []
    for element in [BRANCH_ORDER[index] for index in TRANSFORMER_ROWS]:
        dss.Circuit.SetActiveElement(element)
        terminals = element_pq_3ph_per_terminal()
        losses = [float(value) / 1e6 for value in dss.CktElement.Losses()]
        exported = [sum(value[0] for value in terminals), sum(value[1] for value in terminals)]
        errors = [abs(exported[index] - losses[index]) for index in range(2)]
        rows.append(
            {
                "element": element,
                "exported_loss_mw_mvar": exported,
                "opendss_loss_mw_mvar": losses,
                "absolute_error_mw_mvar": errors,
            }
        )
    maximum = max(error for row in rows for error in row["absolute_error_mw_mvar"])
    return {"rows": rows, "maximum_absolute_error_mw_mvar": maximum, "passed": maximum <= 1e-8}


def _replay_clean_scans(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    model_dir = _resolve_model_dir(None, "case14")
    token_cache: dict[int, list[str]] = {}
    attempted = 0
    matched = 0
    missing_clean = 0
    failures: list[dict[str, Any]] = []
    simulation_cache: dict[str, list[float]] = {}
    for row in rows:
        label = row.get("shared_label", row.get("label"))
        if not isinstance(label, Mapping):
            failures.append({"id": row.get("id"), "error": "missing shared label"})
            continue
        branch = int(label["branch_row0"])
        if branch not in token_cache:
            info = branch_info_for_row0(branch)
            token_cache[branch], _ = _line_tokens(model_dir, str(info["dss_element"]))
        info = branch_info_for_row0(branch)
        for scan in row.get("scans", []):
            if limit and attempted >= limit:
                break
            clean = scan.get("z_clean") if isinstance(scan, Mapping) else None
            if not _finite_vector(clean, 122):
                missing_clean += 1
                continue
            attempted += 1
            try:
                cache_key = json.dumps(
                    {
                        "branch": branch,
                        "alpha": float(label["split_ratio"]),
                        "phase": str(label["phase"]),
                        "r_hif_pu": float(label["r_hif_pu"]),
                        "op_point": scan["op_point"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if cache_key not in simulation_cache:
                    simulation_cache[cache_key] = _simulate_candidate(
                        model_dir=model_dir,
                        original_tokens=token_cache[branch],
                        dss_element=str(info["dss_element"]),
                        alpha=float(label["split_ratio"]),
                        phase=str(label["phase"]),
                        r_hif_pu=float(label["r_hif_pu"]),
                        op_point=scan["op_point"],
                    )["z"]
                error = float(
                    np.max(
                        np.abs(
                            np.asarray(simulation_cache[cache_key], dtype=float)
                            - np.asarray(clean, dtype=float)
                        )
                    )
                )
                if error <= 1e-9:
                    matched += 1
                else:
                    failures.append(
                        {
                            "id": row.get("id"),
                            "scan_index": scan.get("scan_index"),
                            "maximum_absolute_error": error,
                        }
                    )
            except Exception as exc:
                failures.append(
                    {"id": row.get("id"), "scan_index": scan.get("scan_index"), "error": str(exc)}
                )
        if limit and attempted >= limit:
            break
    return {
        "attempted": attempted,
        "matched": matched,
        "missing_clean": missing_clean,
        "all_points_reconstructable": bool(attempted > 0 and matched == attempted and not failures),
        "failures": failures[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IEEE-14 multi-scan HIF event windows.")
    parser.add_argument("samples", type=Path)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--strict-physics", action="store_true")
    parser.add_argument("--replay-limit", type=int, default=0, help="Maximum clean scans to replay; 0 means all.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    rows = [row for _line_no, row in _iter_jsonl(args.samples)]
    issues: Counter[str] = Counter()
    expected_scans = int(meta.get("hif", {}).get("scan_window", {}).get("scans_per_window", 20))
    all_ops: list[list[float]] = []
    measurement_ranks: list[int] = []
    control_ranks: list[int] = []
    physics_vectors: list[list[float]] = []
    window_fingerprints: set[str] = set()
    duplicate_windows = 0
    branch_current_scan_count = 0

    schemas_equal = True
    points_unique = True
    values_finite = True
    topology_constant = True
    for row in rows:
        scans = row.get("scans")
        if not isinstance(scans, list) or len(scans) != expected_scans:
            issues["unexpected_scan_count"] += 1
            continue
        op_keys = []
        op_serialized = []
        op_vectors = []
        measurement_vectors = []
        topology_ids = set()
        for position, scan in enumerate(scans):
            if not isinstance(scan, Mapping):
                issues["bad_scan"] += 1
                continue
            if int(scan.get("scan_index", -1)) != position:
                issues["nonsequential_scan_index"] += 1
            op = scan.get("op_point")
            op_keys.append(set(op) if isinstance(op, Mapping) else set())
            if isinstance(op, Mapping):
                op_serialized.append(json.dumps(op, sort_keys=True, separators=(",", ":")))
            schema_ok = _canonical_schema_matches(op)
            schemas_equal = schemas_equal and schema_ok
            if not schema_ok:
                issues["noncanonical_operating_point"] += 1
            else:
                op_vectors.append(_operating_point_vector(op))
            values_finite = values_finite and _finite_tree(op)
            if not _finite_vector(scan.get("z_obs"), 122):
                issues["bad_z_obs"] += 1
                values_finite = False
            if args.strict_physics and not _finite_vector(scan.get("z_clean"), 122):
                issues["missing_z_clean"] += 1
            vector = scan.get("z_clean") if _finite_vector(scan.get("z_clean"), 122) else scan.get("z_obs")
            if _finite_vector(vector, 122):
                physics_vectors.append(vector)
                measurement_vectors.append(vector)
            if not _phasors_valid(scan.get("three_phase_voltages")):
                issues["bad_three_phase_voltages"] += 1
                values_finite = False
            branch_currents = scan.get(BRANCH_CURRENT_CHANNEL)
            if branch_currents is not None:
                branch_current_scan_count += 1
                if not branch_current_rows_valid(branch_currents, expected_branches=20):
                    issues["bad_three_phase_branch_currents"] += 1
                    values_finite = False
                clean_currents = scan.get(f"{BRANCH_CURRENT_CHANNEL}_clean")
                if args.strict_physics and not branch_current_rows_valid(
                    clean_currents, expected_branches=20
                ):
                    issues["missing_clean_branch_currents"] += 1
            topology_ids.add(str(scan.get("topology_id", row.get("topology_id"))))
            if "label" in scan or "shared_label" in scan:
                issues["label_leakage_inside_scan"] += 1

        mode = row.get("window_metadata", {}).get("operating_point_mode")
        unique_count = len(set(op_serialized))
        if mode == "diverse" and unique_count != len(scans):
            points_unique = False
            issues["duplicate_diverse_operating_point"] += 1
        if len(topology_ids) != 1:
            topology_constant = False
            issues["topology_changed"] += 1
        if op_keys and any(keys != op_keys[0] for keys in op_keys[1:]):
            schemas_equal = False
            issues["operating_point_schema_mismatch"] += 1
        if op_vectors:
            control_ranks.append(_centered_rank(op_vectors))
            all_ops.extend(op_vectors)
        if measurement_vectors:
            measurement_ranks.append(_centered_rank(measurement_vectors))
        fingerprint = json.dumps(
            {
                "label": row.get("shared_label", row.get("label")),
                "operating_points": op_serialized,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint in window_fingerprints:
            duplicate_windows += 1
            issues["duplicate_event_window"] += 1
        window_fingerprints.add(fingerprint)

    operating_checks = {
        "all_schemas_equal": schemas_equal,
        "all_points_unique": points_unique,
        "all_values_finite": values_finite,
        "all_points_reconstructable": None,
        "topology_constant_within_event": topology_constant,
    }
    result: dict[str, Any] = {
        "samples": str(args.samples),
        "meta": str(args.meta),
        "event_count": len(rows),
        "snapshot_count": sum(len(row.get("scans", [])) for row in rows),
        "branch_current_scan_count": int(branch_current_scan_count),
        "expected_scans_per_event": expected_scans,
        "operating_point_checks": operating_checks,
        "rank_diagnostics": {
            "control_space_rank_by_event": control_ranks,
            "measurement_effective_rank_by_event": measurement_ranks,
            "minimum_control_space_rank": min(control_ranks) if control_ranks else None,
            "minimum_measurement_effective_rank": min(measurement_ranks) if measurement_ranks else None,
        },
        "duplicate_event_window_count": duplicate_windows,
    }

    if args.strict_physics:
        endpoints = _branch_endpoints(meta)
        data_physics = _measurement_physics(physics_vectors, endpoints)
        engine_physics = _engine_transformer_loss_check()
        replay = _replay_clean_scans(rows, limit=max(0, int(args.replay_limit)))
        operating_checks["all_points_reconstructable"] = replay["all_points_reconstructable"]
        if not data_physics["transformer_signature_passed"]:
            issues["transformer_two_thirds_signature"] += 1
        if not data_physics["bus_balance_passed"]:
            issues["secondary_bus_balance"] += 1
        if not engine_physics["passed"]:
            issues["opendss_transformer_loss_mismatch"] += 1
        if not replay["all_points_reconstructable"]:
            issues["operating_point_replay_mismatch"] += 1
        result["physics_checks"] = {
            "data": data_physics,
            "engine_transformer_losses": engine_physics,
            "clean_replay": replay,
        }

    for name, passed in operating_checks.items():
        if passed is False:
            issues[f"operating_point_{name}"] += 1
    result["issues"] = dict(sorted(issues.items()))
    result["error_count"] = int(sum(issues.values()))
    result["success"] = not issues

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
