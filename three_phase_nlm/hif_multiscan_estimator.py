from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hif_search_limits import validate_hif_search_limits

from .dss_hif_injector import _phase_number
from .hif_parameter_estimator import (
    _candidate_payload,
    _line_tokens,
    _residual_vector,
    _resolve_model_dir,
    _simulate_base,
    _simulate_candidate,
    classify_parameter_certainty,
)
from .hif_operating_point import canonicalize_ieee14_operating_point
from .ieee14_adapter import branch_info_for_row0


_PHASES = ("A", "B", "C")
_SELECTION_MODES = {"all", "diversity_greedy", "information_greedy"}
_RESISTANCE_MODES = {"shared", "scan_specific_smooth"}
_ROBUST_LOSSES = {"linear", "soft_l1", "huber"}


@dataclass(frozen=True)
class HIFScan:
    scan_index: int
    z_obs: np.ndarray
    three_phase_voltages: Any
    op_point: Mapping[str, Any]
    sigma_z: np.ndarray | None = None
    three_phase_sigma: float = 5e-3


def _finite_float(value: Any, *, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _load_window_payload(scan_window_path: str | Path) -> Mapping[str, Any] | Sequence[Any]:
    raw = str(scan_window_path)
    if raw.startswith("bound://"):
        raise ValueError("Bound scan window requires runtime-hydrated scans")
    path_text, separator, selector = raw.partition("#")
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"HIF scan window not found: {path}")

    if path.suffix.lower() == ".jsonl":
        matches: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, Mapping):
                    continue
                if separator and str(item.get("id")) != selector:
                    continue
                matches.append(item)
                if separator:
                    break
        if not matches:
            suffix = f" with id {selector!r}" if separator else ""
            raise ValueError(f"No scan window found in {path}{suffix}")
        if len(matches) > 1 and not separator:
            raise ValueError("JSONL scan window path contains multiple rows; append #<window-id>")
        return matches[0]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (Mapping, list)):
        raise ValueError("HIF scan window JSON must contain an object or array")
    return payload


def _parse_scans(
    *,
    scans: Sequence[Mapping[str, Any]] | None,
    scan_window_path: str | Path | None,
    default_sigma_z: Sequence[float] | None = None,
) -> tuple[list[HIFScan], dict[str, Any]]:
    window: Mapping[str, Any] = {}
    raw_scans: Any = scans
    if raw_scans is None:
        if scan_window_path is None:
            raise ValueError("Provide scans or scan_window_path")
        loaded = _load_window_payload(scan_window_path)
        if isinstance(loaded, Mapping):
            window = loaded
            raw_scans = loaded.get("scans")
        else:
            raw_scans = loaded
    if not isinstance(raw_scans, Sequence) or isinstance(raw_scans, (str, bytes)) or not raw_scans:
        raise ValueError("At least one HIF scan is required")

    default_sigma = window.get("sigma_z", default_sigma_z)
    default_op_point = window.get("op_point") if isinstance(window.get("op_point"), Mapping) else {}
    parsed: list[HIFScan] = []
    vector_length: int | None = None
    topology_ids: set[str] = set()
    for position, item in enumerate(raw_scans):
        if not isinstance(item, Mapping):
            raise ValueError(f"scan {position} must be an object")
        z_raw = item.get("z_obs", item.get("z"))
        if not isinstance(z_raw, Sequence) or isinstance(z_raw, (str, bytes)):
            raise ValueError(f"scan {position} is missing z_obs")
        z = np.asarray(z_raw, dtype=float)
        if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError(f"scan {position} z_obs must be a finite one-dimensional vector")
        if vector_length is None:
            vector_length = int(z.size)
        elif z.size != vector_length:
            raise ValueError(f"scan {position} measurement length {z.size} != {vector_length}")

        sigma_raw = item.get("sigma_z", default_sigma)
        sigma = None
        if sigma_raw is not None:
            sigma = np.asarray(sigma_raw, dtype=float)
            if sigma.shape != z.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
                raise ValueError(f"scan {position} sigma_z must match z_obs with positive finite values")

        op_point = item.get("op_point")
        if op_point is None:
            op_point = default_op_point
        if not isinstance(op_point, Mapping):
            raise ValueError(f"scan {position} op_point must be an object")
        topology_id = item.get("topology_id", window.get("topology_id"))
        if topology_id is not None:
            topology_ids.add(str(topology_id))
        three_phase_sigma = _finite_float(
            item.get("three_phase_sigma", window.get("three_phase_sigma", 5e-3)),
            field=f"scan {position} three_phase_sigma",
        )
        if three_phase_sigma <= 0.0:
            raise ValueError("three_phase_sigma must be positive")
        parsed.append(
            HIFScan(
                scan_index=int(item.get("scan_index", position)),
                z_obs=z,
                three_phase_voltages=item.get("three_phase_voltages"),
                op_point=canonicalize_ieee14_operating_point(op_point),
                sigma_z=sigma,
                three_phase_sigma=three_phase_sigma,
            )
        )
    if len(topology_ids) > 1:
        raise ValueError("All scans in an HIF event window must use the same topology_id")
    return parsed, dict(window)


def _json_cache_key(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [normalize(val) for val in item]
        if isinstance(item, (np.floating, float, np.integer, int)):
            parsed = float(item)
            return round(parsed, 12) if math.isfinite(parsed) else str(parsed)
        if item is None or isinstance(item, (str, bool)):
            return item
        return repr(item)

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))


def _robust_mean_loss(residual: np.ndarray, loss: str) -> float:
    if residual.size == 0:
        return math.inf
    squared = np.square(np.asarray(residual, dtype=float))
    if loss == "linear":
        values = squared
    elif loss == "soft_l1":
        values = 2.0 * (np.sqrt(1.0 + squared) - 1.0)
    elif loss == "huber":
        absolute = np.abs(residual)
        values = np.where(absolute <= 1.0, squared, 2.0 * absolute - 1.0)
    else:  # pragma: no cover - validated by the public entrypoint
        raise ValueError(f"Unsupported loss {loss!r}")
    return float(np.mean(values))


def _scan_residual(scan: HIFScan, simulated: Mapping[str, Any]) -> np.ndarray:
    return _residual_vector(
        observed_z=scan.z_obs,
        simulated_z=simulated["z"],
        observed_three_phase_voltages=scan.three_phase_voltages,
        simulated_three_phase_voltages=simulated.get("three_phase_voltages"),
        sigma_z=scan.sigma_z,
        three_phase_sigma=scan.three_phase_sigma,
    )


def _normalized_joint_residual(blocks: Sequence[np.ndarray]) -> np.ndarray:
    if not blocks:
        raise ValueError("At least one residual block is required")
    return np.concatenate(blocks) / math.sqrt(float(len(blocks)))


def _joint_weighted_residual_norm(
    normalized_residual: np.ndarray,
    blocks: Sequence[np.ndarray],
) -> float:
    mean_block_size = sum(int(block.size) for block in blocks) / float(len(blocks))
    return float(np.linalg.norm(normalized_residual) / math.sqrt(max(mean_block_size, 1.0)))


def _weighted_simulation_difference(
    scan: HIFScan,
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> np.ndarray:
    return _residual_vector(
        observed_z=positive["z"],
        simulated_z=negative["z"],
        observed_three_phase_voltages=positive.get("three_phase_voltages"),
        simulated_three_phase_voltages=negative.get("three_phase_voltages"),
        sigma_z=scan.sigma_z,
        three_phase_sigma=scan.three_phase_sigma,
    )


def _operating_point_features(scans: Sequence[HIFScan]) -> np.ndarray:
    flat_rows: list[dict[str, float]] = []
    keys: set[str] = set()
    for scan in scans:
        row: dict[str, float] = {}
        for field in (
            "load_scale",
            "source_voltage_pu",
        ):
            value = scan.op_point.get(field)
            if value is not None:
                try:
                    row[field] = float(value)
                except Exception:
                    pass
        for field in (
            "bus_load_scales",
            "load_scales",
            "generator_dispatch_kw",
            "generator_dispatch",
            "generator_dispatch_scales",
            "voltage_setpoints_pu",
            "voltage_setpoints",
        ):
            values = scan.op_point.get(field)
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if isinstance(value, Mapping):
                    value = value.get("kw")
                try:
                    row[f"{field}:{str(name).lower()}"] = float(value)
                except Exception:
                    continue
        flat_rows.append(row)
        keys.update(row)
    ordered = sorted(keys)
    if not ordered:
        return np.zeros((len(scans), 1), dtype=float)
    matrix = np.asarray([[row.get(key, 0.0) for key in ordered] for row in flat_rows], dtype=float)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-12] = 1.0
    return (matrix - mean) / scale


def _diversity_greedy_indices(scans: Sequence[HIFScan], count: int) -> list[int]:
    count = min(max(1, int(count)), len(scans))
    features = _operating_point_features(scans)
    distances_from_mean = np.linalg.norm(features, axis=1)
    first = int(np.argmax(distances_from_mean))
    selected = [first]
    while len(selected) < count:
        remaining = [idx for idx in range(len(scans)) if idx not in selected]
        best_idx = max(
            remaining,
            key=lambda idx: min(float(np.linalg.norm(features[idx] - features[chosen])) for chosen in selected),
        )
        selected.append(int(best_idx))
    return selected


def _information_greedy_indices(information: Sequence[np.ndarray], count: int) -> list[int]:
    count = min(max(1, int(count)), len(information))
    trace_scale = max((float(np.trace(item)) for item in information), default=1.0)
    ridge = max(trace_scale * 1e-9, 1e-12)
    selected: list[int] = []
    current = np.zeros((2, 2), dtype=float)
    while len(selected) < count:
        remaining = [idx for idx in range(len(information)) if idx not in selected]

        def objective(idx: int) -> float:
            sign, value = np.linalg.slogdet(current + information[idx] + ridge * np.eye(2))
            return float(value) if sign > 0 else -math.inf

        best = max(remaining, key=objective)
        selected.append(int(best))
        current = current + information[best]
    return selected


def _information_diversity(information: Sequence[np.ndarray]) -> float:
    normalized: list[np.ndarray] = []
    for item in information:
        trace = float(np.trace(item))
        if trace > 1e-18:
            normalized.append(item / trace)
    if len(normalized) < 2:
        return 0.0
    distances = [
        float(np.linalg.norm(normalized[i] - normalized[j], ord="fro") / math.sqrt(2.0))
        for i in range(len(normalized))
        for j in range(i + 1, len(normalized))
    ]
    return float(np.clip(np.mean(distances), 0.0, 1.0)) if distances else 0.0


def _matrix_rank_and_condition(matrix: np.ndarray) -> tuple[int, np.ndarray, float | None]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0, singular, None
    tolerance = max(float(singular[0]) * 1e-8, 1e-12)
    rank = int(np.sum(singular > tolerance))
    condition = None
    if rank == matrix.shape[0] and singular[-1] > 0.0:
        condition = float(singular[0] / singular[-1])
    return rank, singular, condition


def _observability_payload(
    *,
    effective_information: np.ndarray,
    per_scan_information: Sequence[np.ndarray],
    scan_count: int,
    weighted_residual_norm: float,
    residual_reduction: float | None,
    condition_limit: float,
    correlation_limit: float,
    diagnostic_method: str,
) -> dict[str, Any]:
    rank, singular, condition = _matrix_rank_and_condition(effective_information)
    single_ranks = [_matrix_rank_and_condition(item)[0] for item in per_scan_information]
    best_single_rank = max(single_ranks, default=0)
    correlation = None
    if rank == 2:
        covariance = np.linalg.pinv(effective_information, rcond=1e-12)
        denom = math.sqrt(max(float(covariance[0, 0] * covariance[1, 1]), 0.0))
        if denom > 0.0:
            correlation = float(np.clip(covariance[0, 1] / denom, -1.0, 1.0))

    trace_scale = max(
        [float(np.trace(effective_information)), *[float(np.trace(item)) for item in per_scan_information], 1.0]
    )
    ridge = max(trace_scale * 1e-12, 1e-15)
    total_sign, total_logdet = np.linalg.slogdet(effective_information + ridge * np.eye(2))
    single_logdets = []
    for item in per_scan_information:
        sign, logdet = np.linalg.slogdet(item + ridge * np.eye(2))
        if sign > 0:
            single_logdets.append(float(logdet))
    information_gain = None
    if total_sign > 0 and single_logdets:
        information_gain = float(math.exp(min((float(total_logdet) - max(single_logdets)) / 2.0, 50.0)))

    diversity = _information_diversity(per_scan_information)
    model_mismatch = bool(
        weighted_residual_norm > 3.0
        and (residual_reduction is None or float(residual_reduction) < 0.20)
    )
    identifiable = bool(
        rank == 2
        and condition is not None
        and condition <= float(condition_limit)
        and correlation is not None
        and abs(correlation) <= float(correlation_limit)
        and not model_mismatch
    )
    if rank < 2:
        status = "rank_deficient"
    elif model_mismatch:
        status = "model_mismatch_suspected"
    elif scan_count > 1 and diversity < 0.05:
        status = "noise_averaging_only"
    elif identifiable:
        status = "full_rank_well_conditioned"
    else:
        status = "full_rank_weakly_conditioned"

    return {
        "parameter_dimension": 2,
        "parameter_coordinates": ["alpha", "log_r_hif_pu"],
        "effective_rank": int(rank),
        "best_single_scan_rank": int(best_single_rank),
        "rank_gain_vs_best_single_scan": int(rank - best_single_rank),
        "smallest_singular_value": float(singular[-1]) if singular.size else None,
        "largest_singular_value": float(singular[0]) if singular.size else None,
        "condition_number": condition,
        "alpha_log_r_correlation": correlation,
        "information_gain_vs_best_single_scan": information_gain,
        "scan_diversity_score": diversity,
        "status": status,
        "parameter_identifiable": identifiable,
        "condition_number_limit": float(condition_limit),
        "absolute_correlation_limit": float(correlation_limit),
        "diagnostic_method": diagnostic_method,
        "state_treatment": "implicit_power_flow_reduction_at_known_operating_points",
        "model_mismatch_suspected": model_mismatch,
    }


def _with_diagnostic_coverage(
    payload: Mapping[str, Any],
    *,
    requested_scan_count: int,
    successful_scan_indices: Sequence[int],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out = dict(payload)
    out.update(
        {
            "diagnostic_scan_count": len(successful_scan_indices),
            "requested_diagnostic_scan_count": int(requested_scan_count),
            "diagnostic_failed_scan_count": len(failures),
            "diagnostic_failed_scans": [dict(item) for item in failures],
            "diagnostic_complete": not failures,
        }
    )
    if failures:
        out["status"] = "diagnostic_partial"
        out["parameter_identifiable"] = False
    return out


def _profile_scan_specific_information(
    jacobians: Sequence[np.ndarray],
    *,
    smoothness_lambda: float,
) -> np.ndarray:
    scan_count = len(jacobians)
    rows = sum(item.shape[0] for item in jacobians) + max(scan_count - 1, 0)
    raw = np.zeros((rows, scan_count + 1), dtype=float)
    cursor = 0
    for idx, jacobian in enumerate(jacobians):
        width = jacobian.shape[0]
        raw[cursor : cursor + width, 0] = jacobian[:, 0]
        raw[cursor : cursor + width, idx + 1] = jacobian[:, 1]
        cursor += width
    if scan_count > 1 and smoothness_lambda > 0.0:
        scale = math.sqrt(float(smoothness_lambda))
        for idx in range(scan_count - 1):
            raw[cursor + idx, idx + 1] = -scale
            raw[cursor + idx, idx + 2] = scale

    if scan_count == 1:
        return raw.T @ raw
    ones = np.ones(scan_count, dtype=float) / math.sqrt(float(scan_count))
    projector = np.eye(scan_count) - np.outer(ones, ones)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    contrast_basis = eigenvectors[:, eigenvalues > 0.5]
    transform = np.zeros((scan_count + 1, scan_count + 1), dtype=float)
    transform[0, 0] = 1.0
    transform[1:, 1] = 1.0
    transform[1:, 2:] = contrast_basis
    transformed = raw @ transform
    shared = transformed[:, :2]
    nuisance = transformed[:, 2:]
    shared_info = shared.T @ shared
    if nuisance.shape[1] == 0:
        return shared_info
    nuisance_info = nuisance.T @ nuisance
    cross = shared.T @ nuisance
    profiled = shared_info - cross @ np.linalg.pinv(nuisance_info, rcond=1e-10) @ cross.T
    return 0.5 * (profiled + profiled.T)


def _multiscan_candidate_payload(candidate: Mapping[str, Any], rank: int) -> dict[str, Any]:
    payload = _candidate_payload(candidate, rank=rank)
    values = candidate.get("r_hif_pu_by_scan")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
        parsed = [float(value) for value in values]
        payload["r_hif_pu"] = float(np.median(parsed))
        payload["r_hif_pu_range"] = [min(parsed), max(parsed)]
    payload["weighted_residual_norm"] = float(candidate.get("weighted_residual_norm", candidate["score"]))
    return payload


def _distinct_parameter_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    alpha_tolerance: float = 1e-4,
    log_r_tolerance: float = 1e-3,
) -> list[Mapping[str, Any]]:
    """Remove repeated local-optimizer solutions before top-two classification."""

    def resistance_profile(candidate: Mapping[str, Any]) -> np.ndarray:
        values = candidate.get("r_hif_pu_by_scan")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
            return np.log(np.asarray([float(value) for value in values], dtype=float))
        return np.asarray([math.log(float(candidate["r_hif_pu"]))], dtype=float)

    selected: list[Mapping[str, Any]] = []
    for candidate in candidates:
        candidate_alpha = float(candidate["alpha"])
        candidate_phase = str(candidate.get("phase"))
        candidate_r = resistance_profile(candidate)
        repeated = False
        for existing in selected:
            if candidate_phase != str(existing.get("phase")):
                continue
            if abs(candidate_alpha - float(existing["alpha"])) > float(alpha_tolerance):
                continue
            existing_r = resistance_profile(existing)
            if candidate_r.shape == existing_r.shape and np.max(np.abs(candidate_r - existing_r)) <= float(
                log_r_tolerance
            ):
                repeated = True
                break
        if repeated:
            continue
        selected.append(candidate)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def estimate_hif_location_magnitude_multiscan(
    *,
    candidate_branch_row0: int,
    scan_window_path: str | Path | None = None,
    scans: Sequence[Mapping[str, Any]] | None = None,
    sigma_z: Sequence[float] | None = None,
    case_path: str = "case14",
    candidate_phase: str | None = None,
    pristine_model_dir: str | None = None,
    resistance_mode: str = "shared",
    max_scans: int = 10,
    scan_selection: str = "information_greedy",
    top_k: int = 5,
    alpha_grid_size: int = 31,
    r_grid_size: int = 35,
    r_hif_pu_min: float = 5.0,
    r_hif_pu_max: float = 1000.0,
    robust_loss: str = "soft_l1",
    refine_top_n: int = 3,
    local_max_nfev: int = 40,
    smoothness_lambda: float = 0.10,
    uncertainty_tolerance: float = 0.01,
    condition_number_limit: float = 1e6,
    absolute_correlation_limit: float = 0.98,
) -> dict[str, Any]:
    alpha_grid_size, r_grid_size, validated_max_scans = validate_hif_search_limits(
        alpha_grid_size=alpha_grid_size,
        r_grid_size=r_grid_size,
        max_scans=max_scans,
    )
    assert validated_max_scans is not None
    max_scans = validated_max_scans
    mode = str(resistance_mode).strip().lower()
    selection_mode = str(scan_selection).strip().lower()
    loss = str(robust_loss).strip().lower()
    if mode not in _RESISTANCE_MODES:
        raise ValueError(f"resistance_mode must be one of {sorted(_RESISTANCE_MODES)}")
    if selection_mode not in _SELECTION_MODES:
        raise ValueError(f"scan_selection must be one of {sorted(_SELECTION_MODES)}")
    if loss not in _ROBUST_LOSSES:
        raise ValueError(f"robust_loss must be one of {sorted(_ROBUST_LOSSES)}")
    if float(r_hif_pu_min) <= 0.0 or float(r_hif_pu_max) <= float(r_hif_pu_min):
        raise ValueError("Require 0 < r_hif_pu_min < r_hif_pu_max")
    if float(smoothness_lambda) < 0.0:
        raise ValueError("smoothness_lambda must be non-negative")

    parsed_scans, window = _parse_scans(
        scans=scans,
        scan_window_path=scan_window_path,
        default_sigma_z=sigma_z,
    )
    input_scan_count = len(parsed_scans)
    info = branch_info_for_row0(int(candidate_branch_row0))
    dss_element = str(info["dss_element"])
    if not dss_element.lower().startswith("line."):
        return {
            "success": False,
            "method": "multiscan_augmented_hif_parameter_estimation",
            "candidate_branch_row0": int(candidate_branch_row0),
            "dss_element": dss_element,
            "error": "Multi-scan HIF parameter estimation currently supports Line.* branches only.",
        }

    model_dir = _resolve_model_dir(pristine_model_dir, case_path)
    original_tokens, _ = _line_tokens(model_dir, dss_element)
    phase_candidates = [str(candidate_phase).strip().upper()] if candidate_phase else list(_PHASES)
    for phase in phase_candidates:
        _phase_number(phase)

    simulation_cache: dict[tuple[str, float, float, str], dict[str, Any]] = {}
    base_cache: dict[str, dict[str, Any]] = {}
    candidate_errors: list[str] = []

    def simulate(scan: HIFScan, alpha: float, r_hif_pu: float, phase: str) -> dict[str, Any]:
        op_key = _json_cache_key(scan.op_point)
        key = (str(phase), round(float(alpha), 11), round(float(r_hif_pu), 9), op_key)
        if key not in simulation_cache:
            simulation_cache[key] = _simulate_candidate(
                model_dir=model_dir,
                original_tokens=original_tokens,
                dss_element=dss_element,
                alpha=float(alpha),
                phase=str(phase),
                r_hif_pu=float(r_hif_pu),
                op_point=scan.op_point,
            )
        return simulation_cache[key]

    def finite_difference_jacobian(scan: HIFScan, alpha: float, r_hif_pu: float, phase: str) -> np.ndarray:
        alpha_step = min(0.01, max(0.0025, 0.25 / max(int(alpha_grid_size) - 1, 1)))
        alpha_low = max(0.01, float(alpha) - alpha_step)
        alpha_high = min(0.99, float(alpha) + alpha_step)
        rho = math.log(float(r_hif_pu))
        rho_step = 0.04
        rho_low = max(math.log(float(r_hif_pu_min)), rho - rho_step)
        rho_high = min(math.log(float(r_hif_pu_max)), rho + rho_step)
        if alpha_high <= alpha_low or rho_high <= rho_low:
            raise ValueError("Finite-difference point is pinned at a parameter bound")
        alpha_diff = _weighted_simulation_difference(
            scan,
            simulate(scan, alpha_high, r_hif_pu, phase),
            simulate(scan, alpha_low, r_hif_pu, phase),
        ) / (alpha_high - alpha_low)
        rho_diff = _weighted_simulation_difference(
            scan,
            simulate(scan, alpha, math.exp(rho_high), phase),
            simulate(scan, alpha, math.exp(rho_low), phase),
        ) / (rho_high - rho_low)
        return np.column_stack([alpha_diff, rho_diff])

    desired_scans = min(int(max_scans), input_scan_count)
    selection_fallback = None
    pilot_rejected_scan_indices: list[int] = []
    if desired_scans == input_scan_count or selection_mode == "all":
        selected_positions = list(range(input_scan_count))[:desired_scans]
        applied_selection = "all"
    elif selection_mode == "diversity_greedy":
        selected_positions = _diversity_greedy_indices(parsed_scans, desired_scans)
        applied_selection = "diversity_greedy"
    else:
        pilot_alpha = 0.5
        pilot_r = math.sqrt(float(r_hif_pu_min) * float(r_hif_pu_max))
        pilot_positions: list[int] = []
        pilot_information: list[np.ndarray] = []
        for position, scan in enumerate(parsed_scans):
            try:
                phase_information = []
                for pilot_phase in phase_candidates:
                    jacobian = finite_difference_jacobian(
                        scan, pilot_alpha, pilot_r, pilot_phase
                    )
                    phase_information.append(jacobian.T @ jacobian / max(jacobian.shape[0], 1))
                pilot_information.append(np.mean(phase_information, axis=0))
                pilot_positions.append(position)
            except Exception:
                pilot_rejected_scan_indices.append(int(scan.scan_index))
        if pilot_information:
            relative_positions = _information_greedy_indices(
                pilot_information,
                min(desired_scans, len(pilot_information)),
            )
            selected_positions = [pilot_positions[position] for position in relative_positions]
            applied_selection = "information_greedy"
            if pilot_rejected_scan_indices:
                selection_fallback = (
                    "pilot_sensitivity_rejected_scans: "
                    + ",".join(str(index) for index in pilot_rejected_scan_indices)
                )
        else:
            selected_positions = _diversity_greedy_indices(parsed_scans, desired_scans)
            applied_selection = "diversity_greedy"
            selection_fallback = "information_greedy_failed: no scan produced pilot sensitivities"
    selected_scans = [parsed_scans[position] for position in selected_positions]

    def shared_residual(alpha: float, r_hif_pu: float, phase: str) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
        blocks: list[np.ndarray] = []
        simulations: list[dict[str, Any]] = []
        for scan in selected_scans:
            simulated = simulate(scan, alpha, r_hif_pu, phase)
            simulations.append(simulated)
            blocks.append(_scan_residual(scan, simulated))
        return _normalized_joint_residual(blocks), blocks, simulations

    def shared_candidate(alpha: float, r_hif_pu: float, phase: str, stage: str) -> dict[str, Any] | None:
        try:
            residual, blocks, _ = shared_residual(alpha, r_hif_pu, phase)
            raw_residual = np.concatenate(blocks)
            return {
                "alpha": float(alpha),
                "r_hif_pu": float(r_hif_pu),
                "phase": str(phase),
                "score": math.sqrt(max(_robust_mean_loss(raw_residual, loss), 0.0)),
                "weighted_residual_norm": _joint_weighted_residual_norm(residual, blocks),
                "per_scan_weighted_residual_norms": [
                    float(np.linalg.norm(block) / math.sqrt(block.size)) for block in blocks
                ],
                "stage": stage,
            }
        except Exception as exc:
            if len(candidate_errors) < 8:
                candidate_errors.append(str(exc))
            return None

    def varying_residual(
        alpha: float,
        r_values_by_scan: Sequence[float],
        phase: str,
    ) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
        blocks: list[np.ndarray] = []
        simulations: list[dict[str, Any]] = []
        for scan, r_value in zip(selected_scans, r_values_by_scan):
            simulated = simulate(scan, alpha, float(r_value), phase)
            simulations.append(simulated)
            blocks.append(_scan_residual(scan, simulated))
        residual = _normalized_joint_residual(blocks)
        return residual, blocks, simulations

    def varying_candidate(
        alpha: float,
        r_values_by_scan: Sequence[float],
        phase: str,
        stage: str,
    ) -> dict[str, Any] | None:
        try:
            residual, blocks, _ = varying_residual(alpha, r_values_by_scan, phase)
            raw_residual = np.concatenate(blocks)
            r_values = [float(value) for value in r_values_by_scan]
            smooth = 0.0
            if len(r_values) > 1:
                log_values = np.log(np.asarray(r_values, dtype=float))
                smooth = float(smoothness_lambda) * float(np.mean(np.square(np.diff(log_values))))
            score = math.sqrt(max(_robust_mean_loss(raw_residual, loss) + smooth, 0.0))
            return {
                "alpha": float(alpha),
                "r_hif_pu": float(np.median(r_values)),
                "r_hif_pu_by_scan": r_values,
                "phase": str(phase),
                "score": score,
                "weighted_residual_norm": _joint_weighted_residual_norm(residual, blocks),
                "per_scan_weighted_residual_norms": [
                    float(np.linalg.norm(block) / math.sqrt(block.size)) for block in blocks
                ],
                "smoothness_penalty": smooth,
                "stage": stage,
            }
        except Exception as exc:
            if len(candidate_errors) < 8:
                candidate_errors.append(str(exc))
            return None

    alphas = np.linspace(0.05, 0.95, int(alpha_grid_size))
    r_values = np.geomspace(float(r_hif_pu_min), float(r_hif_pu_max), int(r_grid_size))
    all_candidates: list[dict[str, Any]] = []

    if mode == "shared":
        for phase in phase_candidates:
            for alpha in alphas:
                for r_value in r_values:
                    candidate = shared_candidate(float(alpha), float(r_value), phase, "coarse_grid")
                    if candidate is not None:
                        all_candidates.append(candidate)
    else:
        log_r = np.log(r_values)
        scan_count = len(selected_scans)
        for phase in phase_candidates:
            for alpha in alphas:
                loss_matrix = np.full((scan_count, len(r_values)), math.inf, dtype=float)
                for scan_idx, scan in enumerate(selected_scans):
                    for r_idx, r_value in enumerate(r_values):
                        try:
                            residual = _scan_residual(scan, simulate(scan, float(alpha), float(r_value), phase))
                            loss_matrix[scan_idx, r_idx] = _robust_mean_loss(residual, loss)
                        except Exception as exc:
                            if len(candidate_errors) < 8:
                                candidate_errors.append(str(exc))
                if not np.any(np.isfinite(loss_matrix)):
                    continue
                dp = np.full_like(loss_matrix, math.inf)
                back = np.full(loss_matrix.shape, -1, dtype=int)
                dp[0, :] = loss_matrix[0, :]
                for scan_idx in range(1, scan_count):
                    transition = (
                        dp[scan_idx - 1, :, None]
                        + float(smoothness_lambda) * np.square(log_r[:, None] - log_r[None, :])
                    )
                    best_prev = np.argmin(transition, axis=0)
                    dp[scan_idx, :] = loss_matrix[scan_idx, :] + transition[best_prev, np.arange(len(r_values))]
                    back[scan_idx, :] = best_prev
                end = int(np.argmin(dp[-1, :]))
                path = [end]
                for scan_idx in range(scan_count - 1, 0, -1):
                    path.append(int(back[scan_idx, path[-1]]))
                path.reverse()
                selected_r = [float(r_values[idx]) for idx in path]
                candidate = varying_candidate(float(alpha), selected_r, phase, "coarse_grid_profile")
                if candidate is not None:
                    all_candidates.append(candidate)

    if not all_candidates:
        return {
            "success": False,
            "method": "multiscan_augmented_hif_parameter_estimation",
            "candidate_branch_row0": int(candidate_branch_row0),
            "dss_element": dss_element,
            "scan_count": input_scan_count,
            "selected_scan_count": len(selected_scans),
            "error": "No multi-scan HIF parameter candidates solved successfully.",
            "candidate_errors": candidate_errors,
        }

    all_candidates.sort(key=lambda item: float(item["score"]))
    seeds = all_candidates[: max(0, int(refine_top_n))]
    if seeds and int(local_max_nfev) > 0:
        try:
            from scipy.optimize import least_squares  # type: ignore

            for seed in seeds:
                phase = str(seed["phase"])
                if mode == "shared":
                    x0 = np.asarray([float(seed["alpha"]), math.log(float(seed["r_hif_pu"]))], dtype=float)

                    def objective(x: np.ndarray) -> np.ndarray:
                        return shared_residual(float(x[0]), math.exp(float(x[1])), phase)[0]

                    result = least_squares(
                        objective,
                        x0=x0,
                        bounds=(
                            np.asarray([0.01, math.log(float(r_hif_pu_min))]),
                            np.asarray([0.99, math.log(float(r_hif_pu_max))]),
                        ),
                        loss=loss,
                        f_scale=1.0 / math.sqrt(float(len(selected_scans))),
                        max_nfev=int(local_max_nfev),
                    )
                    refined = shared_candidate(
                        float(result.x[0]),
                        math.exp(float(result.x[1])),
                        phase,
                        "bounded_local_refinement",
                    )
                else:
                    seed_r = [float(value) for value in seed["r_hif_pu_by_scan"]]
                    x0 = np.asarray([float(seed["alpha"]), *np.log(seed_r)], dtype=float)

                    def objective(x: np.ndarray) -> np.ndarray:
                        rho = np.asarray(x[1:], dtype=float)
                        data = varying_residual(float(x[0]), np.exp(rho), phase)[0]
                        if rho.size <= 1 or smoothness_lambda <= 0.0:
                            return data
                        smooth = math.sqrt(float(smoothness_lambda)) * np.diff(rho)
                        return np.concatenate([data, smooth])

                    result = least_squares(
                        objective,
                        x0=x0,
                        bounds=(
                            np.asarray([0.01, *([math.log(float(r_hif_pu_min))] * len(seed_r))]),
                            np.asarray([0.99, *([math.log(float(r_hif_pu_max))] * len(seed_r))]),
                        ),
                        loss=loss,
                        f_scale=1.0 / math.sqrt(float(len(selected_scans))),
                        max_nfev=int(local_max_nfev),
                    )
                    refined = varying_candidate(
                        float(result.x[0]),
                        np.exp(result.x[1:]),
                        phase,
                        "bounded_local_refinement",
                    )
                if refined is not None:
                    all_candidates.append(refined)
        except Exception as exc:
            candidate_errors.append(f"local_refinement_skipped: {exc}")

    all_candidates.sort(key=lambda item: float(item["score"]))
    best = all_candidates[0]
    best_score = float(best["score"])
    top_candidates = _distinct_parameter_candidates(all_candidates, limit=max(1, int(top_k)))
    if len(top_candidates) >= 2:
        top2_gap = float(top_candidates[1]["score"]) - best_score
        relative_gap = top2_gap / max(abs(best_score), 1e-9)
    else:
        top2_gap = math.inf
        relative_gap = math.inf
    threshold = best_score + max(abs(best_score) * float(uncertainty_tolerance), 1e-6)
    near_best = [item for item in all_candidates if float(item["score"]) <= threshold] or [best]
    certainty, ambiguity = classify_parameter_certainty(
        relative_gap=relative_gap,
        near_best=near_best,
        top_candidates=top_candidates,
    )

    best_r_values = (
        [float(value) for value in best["r_hif_pu_by_scan"]]
        if mode == "scan_specific_smooth"
        else [float(best["r_hif_pu"])] * len(selected_scans)
    )
    best_residual, best_blocks, best_simulations = varying_residual(
        float(best["alpha"]),
        best_r_values,
        str(best["phase"]),
    )

    base_blocks: list[np.ndarray] = []
    for scan in selected_scans:
        try:
            key = _json_cache_key(scan.op_point)
            if key not in base_cache:
                base_cache[key] = _simulate_base(model_dir, op_point=scan.op_point)
            base_blocks.append(_scan_residual(scan, base_cache[key]))
        except Exception as exc:
            if len(candidate_errors) < 8:
                candidate_errors.append(f"base_simulation: {exc}")
    base_score = None
    residual_reduction = None
    if len(base_blocks) == len(selected_scans):
        base_residual = _normalized_joint_residual(base_blocks)
        base_score = _joint_weighted_residual_norm(base_residual, base_blocks)
        if base_score > 1e-12:
            residual_reduction = (base_score - float(best["weighted_residual_norm"])) / base_score

    jacobians: list[np.ndarray] = []
    diagnostic_scan_indices: list[int] = []
    diagnostic_failures: list[dict[str, Any]] = []
    observability_error = None
    try:
        for scan, r_value in zip(selected_scans, best_r_values):
            try:
                jacobians.append(
                    finite_difference_jacobian(
                        scan,
                        float(best["alpha"]),
                        float(r_value),
                        str(best["phase"]),
                    )
                )
                diagnostic_scan_indices.append(int(scan.scan_index))
            except Exception as exc:
                if mode != "shared":
                    raise
                diagnostic_failures.append(
                    {
                        "scan_index": int(scan.scan_index),
                        "error": str(exc),
                    }
                )
        if not jacobians:
            raise RuntimeError("No selected scan produced finite-difference sensitivities")
        per_scan_information = [jac.T @ jac / max(jac.shape[0], 1) for jac in jacobians]
        if mode == "shared":
            effective_information = np.sum(per_scan_information, axis=0)
            diagnostic_method = "finite_difference_reduced_information"
        else:
            effective_information = _profile_scan_specific_information(
                jacobians,
                smoothness_lambda=float(smoothness_lambda),
            )
            diagnostic_method = "finite_difference_profiled_information_with_resistance_smoothness"
        observability = _with_diagnostic_coverage(
            _observability_payload(
                effective_information=effective_information,
                per_scan_information=per_scan_information,
                scan_count=len(jacobians),
                weighted_residual_norm=float(best["weighted_residual_norm"]),
                residual_reduction=residual_reduction,
                condition_limit=float(condition_number_limit),
                correlation_limit=float(absolute_correlation_limit),
                diagnostic_method=diagnostic_method,
            ),
            requested_scan_count=len(selected_scans),
            successful_scan_indices=diagnostic_scan_indices,
            failures=diagnostic_failures,
        )
    except Exception as exc:
        observability_error = str(exc)
        observability = {
            "parameter_dimension": 2,
            "parameter_coordinates": ["alpha", "log_r_hif_pu"],
            "effective_rank": None,
            "status": "diagnostic_unavailable",
            "parameter_identifiable": False,
            "diagnostic_method": "finite_difference_reduced_information",
            "error": observability_error,
            "diagnostic_scan_count": len(diagnostic_scan_indices),
            "requested_diagnostic_scan_count": len(selected_scans),
            "diagnostic_failed_scan_count": len(diagnostic_failures),
            "diagnostic_failed_scans": diagnostic_failures,
            "diagnostic_complete": False,
        }

    r_ohms = [float(sim["r_hif_ohm"]) for sim in best_simulations]
    fault_volts = [float(sim.get("fault_v_volts") or (1000.0 / math.sqrt(3.0))) for sim in best_simulations]
    currents = [voltage / resistance for voltage, resistance in zip(fault_volts, r_ohms)]
    powers = [voltage**2 / resistance / 1000.0 for voltage, resistance in zip(fault_volts, r_ohms)]
    median_r_pu = float(np.median(best_r_values))
    median_r_ohm = float(np.median(r_ohms))
    phase_scores = {
        phase: min(float(item["score"]) for item in all_candidates if item.get("phase") == phase)
        for phase in phase_candidates
        if any(item.get("phase") == phase for item in all_candidates)
    }
    near_r_values = [float(item["r_hif_pu"]) for item in near_best]

    estimated: dict[str, Any] = {
        "alpha_from_from_bus": float(best["alpha"]),
        "distance_percent_from_from_bus": 100.0 * float(best["alpha"]),
        "phase": best.get("phase"),
        "resistance_model": mode,
        "r_hif_pu": median_r_pu,
        "r_hif_ohm": median_r_ohm,
        "g_hif_siemens": 1.0 / median_r_ohm if median_r_ohm > 0.0 else None,
        "i_hif_amp": float(np.median(currents)),
        "p_hif_kw": float(np.median(powers)),
        "q_hif_kvar": 0.0,
    }
    if mode == "scan_specific_smooth":
        estimated.update(
            {
                "r_hif_pu_median": median_r_pu,
                "r_hif_pu_range": [min(best_r_values), max(best_r_values)],
                "r_hif_ohm_range": [min(r_ohms), max(r_ohms)],
                "i_hif_amp_range": [min(currents), max(currents)],
                "p_hif_kw_range": [min(powers), max(powers)],
                "per_scan_r_hif_pu": [
                    {"scan_index": scan.scan_index, "r_hif_pu": float(value)}
                    for scan, value in zip(selected_scans, best_r_values)
                ],
            }
        )

    return {
        "success": True,
        "parameter_identifiable": bool(observability.get("parameter_identifiable", False)),
        "method": "multiscan_augmented_hif_parameter_estimation",
        "candidate_branch_row0": int(candidate_branch_row0),
        "input_branch_row0": int(candidate_branch_row0),
        "dss_element": dss_element,
        "from_bus": info.get("from_bus"),
        "to_bus": info.get("to_bus"),
        "scan_count": input_scan_count,
        "selected_scan_count": len(selected_scans),
        "selected_scan_indices": [scan.scan_index for scan in selected_scans],
        "estimated": estimated,
        "observability": observability,
        "fit": {
            "multiscan_weighted_residual_norm": float(best["weighted_residual_norm"]),
            "weighted_residual_norm": float(best["weighted_residual_norm"]),
            "robust_score": best_score,
            "robust_loss": loss,
            "no_hif_weighted_residual_norm": base_score,
            "residual_reduction_vs_no_hif": residual_reduction,
            "localization_certainty": certainty,
            "ambiguity": bool(ambiguity),
            "top2_delta_score": top2_gap if math.isfinite(top2_gap) else None,
            "top2_relative_gap": relative_gap if math.isfinite(relative_gap) else None,
            "per_scan_weighted_residual_norms": [
                float(np.linalg.norm(block) / math.sqrt(block.size)) for block in best_blocks
            ],
            "model_mismatch_suspected": bool(observability.get("model_mismatch_suspected", False)),
        },
        "uncertainty": {
            "near_best_alpha_interval": [
                min(float(item["alpha"]) for item in near_best),
                max(float(item["alpha"]) for item in near_best),
            ],
            "near_best_r_hif_pu_interval": [min(near_r_values), max(near_r_values)],
            "interval_method": "near_best_multiscan_score_profile",
            "score_tolerance": float(uncertainty_tolerance),
            "near_best_count": len(near_best),
        },
        "phase_scores": phase_scores,
        "top_parameter_candidates": [
            _multiscan_candidate_payload(item, rank)
            for rank, item in enumerate(top_candidates, start=1)
        ],
        "scan_selection": {
            "requested": selection_mode,
            "applied": applied_selection,
            "selected_scan_indices": [scan.scan_index for scan in selected_scans],
            "input_scan_count": input_scan_count,
            "selected_scan_count": len(selected_scans),
            "fallback": selection_fallback,
            "pilot_rejected_scan_indices": pilot_rejected_scan_indices,
        },
        "search": {
            "resistance_mode": mode,
            "alpha_grid_size": int(alpha_grid_size),
            "r_grid_size": int(r_grid_size),
            "phase_candidates": phase_candidates,
            "coarse_parameter_combinations": int(
                len(alphas) * len(r_values) * len(phase_candidates)
                if mode == "shared"
                else len(alphas) * len(phase_candidates)
            ),
            "total_ranked_candidates": len(all_candidates),
            "unique_opendss_candidate_solves": len(simulation_cache),
            "cache_reused_across_identical_operating_points": True,
            "local_refinement_requested": bool(seeds and int(local_max_nfev) > 0),
            "candidate_errors": candidate_errors[:8],
        },
        "window_id": window.get("id"),
    }


__all__ = ["HIFScan", "estimate_hif_location_magnitude_multiscan"]
