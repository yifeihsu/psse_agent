from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from IEEE_14_OpenDSS.constants import BUS_ORDER

from .dss_hif_injector import (
    _line_matcher,
    _parse_line_tokens,
    _phase_number,
    _render_replacement_line,
    constant_impedance_hif_kw,
    hif_ohms_from_pu,
)
from .ieee14_adapter import branch_info_for_row0


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PRISTINE_MODEL_DIR = _REPO_ROOT / "IEEE_14_OpenDSS"
_PHASES = ("A", "B", "C")


def _measurement_exporters():
    from IEEE_14_OpenDSS.export_measurement_series import (  # type: ignore
        extract_measurement_series,
        extract_three_phase_voltage_measurements,
    )

    return extract_measurement_series, extract_three_phase_voltage_measurements


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _resolve_model_dir(pristine_model_dir: str | None, case_path: str | None) -> Path:
    candidates: list[Path] = []
    if pristine_model_dir:
        candidates.append(Path(pristine_model_dir).expanduser())
    if case_path and str(case_path) not in {"case14", "IEEE14", "ieee14"}:
        candidates.append(Path(str(case_path)).expanduser())
    candidates.append(_DEFAULT_PRISTINE_MODEL_DIR)

    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = (_REPO_ROOT / candidate).resolve()
        if candidate.is_dir() and (candidate / "Run_IEEE14Bus.dss").exists():
            return candidate
    raise FileNotFoundError("No IEEE-14 OpenDSS model directory with Run_IEEE14Bus.dss was found.")


def _line_tokens(model_dir: Path, dss_element: str) -> tuple[list[str], dict[str, str]]:
    path = model_dir / "IEEE14Lines.DSS"
    if not path.exists():
        raise FileNotFoundError(f"IEEE14Lines.DSS not found in {model_dir}")
    matcher = _line_matcher(dss_element)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if matcher.match(line):
            return _parse_line_tokens(line)
    raise ValueError(f"Line {dss_element!r} not found in {path}")


def _compile_base_model(model_dir: Path) -> None:
    import opendssdirect as dss  # type: ignore

    dss.Basic.DataPath(str(model_dir.resolve()))
    dss.Text.Command("Clear")
    dss.Text.Command("Redirect Run_IEEE14Bus.dss")

    # The checked-in IEEE-14 OpenDSS load file may carry the imbalance-study Bus 3
    # split loads. HIF synthetic rows use a balanced base case, so mirror the
    # generator's balanced-load override directly in the active circuit.
    for name in ("B3A", "B3B", "B3C"):
        dss.Text.Command(f"Edit Load.{name} enabled=no")
    dss.Text.Command(
        "New Load.__HIF_BAL_B3 Bus1=B3 kV=1 kW=94200 kvar=19000 "
        "vmaxpu=1.06 vminpu=0.94"
    )


def _enabled_load_snapshot() -> dict[str, dict[str, Any]]:
    import opendssdirect as dss  # type: ignore

    base: dict[str, dict[str, Any]] = {}
    for name in dss.Loads.AllNames() or []:
        dss.Loads.Name(name)
        lower = str(name).lower()
        if lower.startswith("hif_") or lower.startswith("hifest") or lower.startswith("hif_est"):
            continue
        if hasattr(dss.CktElement, "Enabled") and not bool(dss.CktElement.Enabled()):
            continue
        base[lower] = {
            "name": str(name),
            "kW": float(dss.Loads.kW()),
            "kvar": float(dss.Loads.kvar()),
        }
    return base


def _scale_named_loads(loads: Mapping[str, Mapping[str, Any]], load_scale: float) -> None:
    import opendssdirect as dss  # type: ignore

    for item in loads.values():
        dss.Loads.Name(str(item["name"]))
        dss.Loads.kW(float(item["kW"]) * float(load_scale))
        dss.Loads.kvar(float(item["kvar"]) * float(load_scale))


def _solve_or_raise() -> None:
    import opendssdirect as dss  # type: ignore

    dss.Text.Command("Solve")
    if hasattr(dss, "Solution") and not bool(dss.Solution.Converged()):
        raise RuntimeError("OpenDSS solve did not converge")


def _inject_candidate(
    *,
    original_tokens: list[str],
    dss_element: str,
    alpha: float,
    phase: str,
    r_hif_ohm: float,
    fault_bus: str,
) -> dict[str, dict[str, Any]]:
    import opendssdirect as dss  # type: ignore

    _, kv = _parse_line_tokens(" ".join(original_tokens))
    try:
        length = float(kv["length"])
        bus1 = kv["bus1"]
        bus2 = kv["bus2"]
    except KeyError as exc:
        raise ValueError(f"Line {dss_element!r} is missing token {exc.args[0]!r}") from exc

    line_a = f"{dss_element}_hif_est_a"
    line_b = f"{dss_element}_hif_est_b"
    phase_no = _phase_number(phase)
    kv_ln, p_kw = constant_impedance_hif_kw(r_hif_ohm, kv_ll=1.0)
    len_a = float(length) * float(alpha)
    len_b = float(length) - len_a

    dss.Text.Command(f"Edit {dss_element} enabled=no")
    dss.Text.Command(
        _render_replacement_line(
            original_tokens=original_tokens,
            new_element=line_a,
            bus1=bus1,
            bus2=f"{fault_bus}.1.2.3",
            length=len_a,
        )
    )
    dss.Text.Command(
        _render_replacement_line(
            original_tokens=original_tokens,
            new_element=line_b,
            bus1=f"{fault_bus}.1.2.3",
            bus2=bus2,
            length=len_b,
        )
    )
    dss.Text.Command(
        f"New Load.HIF_EST Bus1={fault_bus}.{phase_no} Phases=1 Conn=Wye "
        f"Model=2 Status=Fixed kV={kv_ln:.12g} kW={p_kw:.12g} kvar=0"
    )
    return {
        dss_element: {
            "from": line_a,
            "from_terminal": 0,
            "to": line_b,
            "to_terminal": 1,
        }
    }


def _fault_voltage_volts(fault_bus: str, phase: str, fallback_kv_ln: float) -> float:
    import opendssdirect as dss  # type: ignore

    phase_idx = _phase_number(phase) - 1
    try:
        dss.Circuit.SetActiveBus(fault_bus)
        magang = dss.Bus.VMagAngle() or []
        mag_v = magang[2 * phase_idx] if len(magang) > 2 * phase_idx else None
        if mag_v is not None and float(mag_v) > 0.0:
            return float(mag_v)
    except Exception:
        pass
    return float(fallback_kv_ln) * 1000.0


def _simulate_base(model_dir: Path, *, load_scale: float) -> dict[str, Any]:
    extract_measurement_series, extract_three_phase_voltage_measurements = _measurement_exporters()
    _compile_base_model(model_dir)
    base_loads = _enabled_load_snapshot()
    _scale_named_loads(base_loads, load_scale)
    _solve_or_raise()
    z_sim, _buses, _branches = extract_measurement_series()
    return {
        "z": [float(x) for x in z_sim],
        "three_phase_voltages": extract_three_phase_voltage_measurements(),
    }


def _simulate_candidate(
    *,
    model_dir: Path,
    original_tokens: list[str],
    dss_element: str,
    alpha: float,
    phase: str,
    r_hif_pu: float,
    load_scale: float,
) -> dict[str, Any]:
    extract_measurement_series, extract_three_phase_voltage_measurements = _measurement_exporters()
    r_hif_ohm = hif_ohms_from_pu(r_hif_pu, base_mva=100.0, kv_ll=1.0)
    fault_bus = "FaultEst"
    _compile_base_model(model_dir)
    overrides = _inject_candidate(
        original_tokens=original_tokens,
        dss_element=dss_element,
        alpha=alpha,
        phase=phase,
        r_hif_ohm=r_hif_ohm,
        fault_bus=fault_bus,
    )
    base_loads = _enabled_load_snapshot()
    _scale_named_loads(base_loads, load_scale)
    _solve_or_raise()
    z_sim, _buses, _branches = extract_measurement_series(branch_element_overrides=overrides)
    v3 = extract_three_phase_voltage_measurements()
    kv_ln, _nominal_p_kw = constant_impedance_hif_kw(r_hif_ohm, kv_ll=1.0)
    fault_v = _fault_voltage_volts(fault_bus, phase, kv_ln)
    return {
        "z": [float(x) for x in z_sim],
        "three_phase_voltages": v3,
        "fault_v_volts": float(fault_v),
        "r_hif_ohm": float(r_hif_ohm),
    }


def _measurement_residual(z_obs: Sequence[float], z_sim: Sequence[float]) -> list[float]:
    obs = np.asarray(z_obs, dtype=float)
    sim = np.asarray(z_sim, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"Measurement vector shape mismatch: observed {obs.shape}, simulated {sim.shape}")
    if obs.size == 122:
        sigma = np.empty_like(obs, dtype=float)
        sigma[0:14] = 1e-3
        sigma[14:42] = 1e-2
        sigma[42:122] = 1e-2
    else:
        sigma = np.full_like(obs, 1e-2, dtype=float)
    return ((obs - sim) / sigma).astype(float).tolist()


def _voltage_rows_to_phasors(rows: Any) -> dict[str, list[complex]]:
    out: dict[str, list[complex]] = {}
    if not isinstance(rows, list):
        return out
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        bus = item.get("bus")
        vln_pu = item.get("vln_pu")
        ang_deg = item.get("ang_deg")
        if bus is None or not isinstance(vln_pu, list) or not isinstance(ang_deg, list):
            continue
        vals: list[complex] = []
        for idx in range(min(3, len(vln_pu), len(ang_deg))):
            mag = _maybe_float(vln_pu[idx])
            ang = _maybe_float(ang_deg[idx])
            if mag is None or ang is None:
                vals.append(complex(np.nan, np.nan))
            else:
                vals.append(float(mag) * np.exp(1j * np.deg2rad(float(ang))))
        if len(vals) == 3:
            out[str(bus).lower()] = vals
    return out


def _three_phase_voltage_residual(observed: Any, simulated: Any, *, sigma: float = 5e-3) -> list[float]:
    obs = _voltage_rows_to_phasors(observed)
    sim = _voltage_rows_to_phasors(simulated)
    if not obs or not sim:
        return []
    residuals: list[float] = []
    for bus in [str(b).lower() for b in BUS_ORDER]:
        if bus not in obs or bus not in sim:
            continue
        for vo, vs in zip(obs[bus], sim[bus]):
            if not np.isfinite(vo.real + vo.imag + vs.real + vs.imag):
                continue
            diff = (vo - vs) / float(sigma)
            residuals.extend([float(diff.real), float(diff.imag)])
    return residuals


def _residual_vector(
    *,
    observed_z: Sequence[float],
    simulated_z: Sequence[float],
    observed_three_phase_voltages: Any = None,
    simulated_three_phase_voltages: Any = None,
) -> np.ndarray:
    parts = _measurement_residual(observed_z, simulated_z)
    parts.extend(
        _three_phase_voltage_residual(
            observed_three_phase_voltages,
            simulated_three_phase_voltages,
        )
    )
    return np.asarray(parts, dtype=float)


def _score(residual: np.ndarray) -> float:
    if residual.size == 0:
        return math.inf
    return float(np.linalg.norm(residual) / math.sqrt(float(residual.size)))


def _candidate_payload(candidate: Mapping[str, Any], rank: int | None = None) -> dict[str, Any]:
    out = {
        "alpha_from_from_bus": float(candidate["alpha"]),
        "distance_percent_from_from_bus": 100.0 * float(candidate["alpha"]),
        "phase": candidate.get("phase"),
        "r_hif_pu": float(candidate["r_hif_pu"]),
        "score": float(candidate["score"]),
    }
    if rank is not None:
        out = {"rank": int(rank), **out}
    return out


def _local_refinement_points(best: Mapping[str, Any], *, alpha_step: float, r_ratio: float) -> list[tuple[float, float, str]]:
    alpha0 = float(best["alpha"])
    r0 = float(best["r_hif_pu"])
    phase = str(best["phase"])
    alpha_radius = max(float(alpha_step), 0.025)
    r_low = max(r0 / max(float(r_ratio), 1.01), 1e-6)
    r_high = r0 * max(float(r_ratio), 1.01)
    alphas = np.linspace(max(0.01, alpha0 - alpha_radius), min(0.99, alpha0 + alpha_radius), 9)
    rs = np.geomspace(r_low, r_high, 9)
    return [(float(alpha), float(r), phase) for alpha in alphas for r in rs]


def classify_parameter_certainty(
    *,
    relative_gap: float,
    near_best: Sequence[Mapping[str, Any]],
    top_candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, bool]:
    alpha_near = [
        float(c["alpha"])
        for c in near_best
        if c.get("alpha") is not None and math.isfinite(float(c["alpha"]))
    ]
    alpha_top = [
        float(c["alpha"])
        for c in top_candidates
        if c.get("alpha") is not None and math.isfinite(float(c["alpha"]))
    ]

    near_width = max(alpha_near) - min(alpha_near) if len(alpha_near) >= 2 else 0.0
    top_width = max(alpha_top) - min(alpha_top) if len(alpha_top) >= 2 else 0.0
    top_scores = [
        float(c["score"])
        for c in top_candidates
        if c.get("score") is not None and math.isfinite(float(c["score"]))
    ]
    top2_delta = (
        abs(float(top_scores[1]) - float(top_scores[0]))
        if len(top_scores) >= 2
        else math.inf
    )

    if (
        relative_gap <= 1e-4
        or (top2_delta <= 1e-4 and top_width > 0.02)
        or near_width > 0.10
        or top_width > 0.20
    ):
        return "ambiguous_top2", True
    if relative_gap <= 1e-2 or near_width > 0.05:
        return "moderately_separated", False
    return "well_separated", False


def estimate_hif_location_magnitude(
    *,
    case_path: str = "case14",
    candidate_branch_row0: int,
    candidate_phase: str | None = None,
    z_obs: Sequence[float] | None = None,
    three_phase_voltages: Any = None,
    pristine_model_dir: str | None = None,
    load_scale: float = 1.0,
    top_k: int = 5,
    alpha_grid_size: int = 31,
    r_grid_size: int = 35,
    r_hif_pu_min: float = 5.0,
    r_hif_pu_max: float = 1000.0,
    refine_top_n: int = 3,
    uncertainty_tolerance: float = 0.01,
) -> dict[str, Any]:
    if z_obs is None:
        return {
            "success": False,
            "method": "model_based_hif_parameter_search",
            "error": "z_obs is required for HIF parameter estimation.",
        }
    if int(alpha_grid_size) < 2:
        raise ValueError("alpha_grid_size must be at least 2")
    if int(r_grid_size) < 2:
        raise ValueError("r_grid_size must be at least 2")
    if float(r_hif_pu_min) <= 0.0 or float(r_hif_pu_max) <= float(r_hif_pu_min):
        raise ValueError("Require 0 < r_hif_pu_min < r_hif_pu_max")

    info = branch_info_for_row0(int(candidate_branch_row0))
    dss_element = str(info["dss_element"])
    if not dss_element.lower().startswith("line."):
        return {
            "success": False,
            "method": "model_based_hif_parameter_search",
            "candidate_branch_row0": int(candidate_branch_row0),
            "dss_element": dss_element,
            "error": "HIF parameter estimation currently supports Line.* branches only.",
        }

    model_dir = _resolve_model_dir(pristine_model_dir, case_path)
    original_tokens, _kv = _line_tokens(model_dir, dss_element)

    phase_candidates = [str(candidate_phase).strip().upper()] if candidate_phase else list(_PHASES)
    for phase in phase_candidates:
        _phase_number(phase)

    alphas = np.linspace(0.05, 0.95, int(alpha_grid_size))
    r_values = np.geomspace(float(r_hif_pu_min), float(r_hif_pu_max), int(r_grid_size))
    all_candidates: list[dict[str, Any]] = []
    errors: list[str] = []

    def evaluate(alpha: float, r_hif_pu: float, phase: str, *, stage: str) -> None:
        try:
            sim = _simulate_candidate(
                model_dir=model_dir,
                original_tokens=original_tokens,
                dss_element=dss_element,
                alpha=float(alpha),
                phase=phase,
                r_hif_pu=float(r_hif_pu),
                load_scale=float(load_scale),
            )
            residual = _residual_vector(
                observed_z=z_obs,
                simulated_z=sim["z"],
                observed_three_phase_voltages=three_phase_voltages,
                simulated_three_phase_voltages=sim.get("three_phase_voltages"),
            )
            score = _score(residual)
            all_candidates.append(
                {
                    "alpha": float(alpha),
                    "r_hif_pu": float(r_hif_pu),
                    "phase": phase,
                    "score": float(score),
                    "stage": stage,
                    "fault_v_volts": float(sim.get("fault_v_volts") or 0.0),
                    "r_hif_ohm": float(sim["r_hif_ohm"]),
                }
            )
        except Exception as exc:
            if len(errors) < 5:
                errors.append(str(exc))

    for phase in phase_candidates:
        for alpha in alphas:
            for r_hif_pu in r_values:
                evaluate(float(alpha), float(r_hif_pu), phase, stage="coarse_grid")

    if not all_candidates:
        return {
            "success": False,
            "method": "model_based_hif_parameter_search",
            "candidate_branch_row0": int(candidate_branch_row0),
            "dss_element": dss_element,
            "error": "No HIF parameter candidates solved successfully.",
            "candidate_errors": errors,
        }

    all_candidates.sort(key=lambda item: float(item["score"]))
    coarse_best = list(all_candidates[: max(1, int(refine_top_n))])
    alpha_step = float(alphas[1] - alphas[0]) if len(alphas) > 1 else 0.05
    r_ratio = float(r_values[1] / r_values[0]) if len(r_values) > 1 else 1.25
    seen = {
        (
            round(float(item["alpha"]), 12),
            round(float(item["r_hif_pu"]), 9),
            str(item["phase"]),
        )
        for item in all_candidates
    }
    for seed in coarse_best:
        for alpha, r_hif_pu, phase in _local_refinement_points(seed, alpha_step=alpha_step, r_ratio=r_ratio):
            key = (round(float(alpha), 12), round(float(r_hif_pu), 9), str(phase))
            if key in seen:
                continue
            seen.add(key)
            evaluate(alpha, r_hif_pu, phase, stage="local_grid_refinement")

    all_candidates.sort(key=lambda item: float(item["score"]))
    best = all_candidates[0]
    best_score = float(best["score"])
    top_candidates = all_candidates[: max(1, int(top_k))]

    try:
        base_sim = _simulate_base(model_dir, load_scale=float(load_scale))
        base_residual = _residual_vector(
            observed_z=z_obs,
            simulated_z=base_sim["z"],
            observed_three_phase_voltages=three_phase_voltages,
            simulated_three_phase_voltages=base_sim.get("three_phase_voltages"),
        )
        base_score = _score(base_residual)
    except Exception:
        base_score = math.nan
    residual_reduction = None
    if math.isfinite(base_score) and base_score > 1e-12:
        residual_reduction = (base_score - best_score) / base_score

    if len(top_candidates) >= 2:
        top2_gap = float(top_candidates[1]["score"]) - best_score
        relative_gap = top2_gap / max(abs(best_score), 1e-9)
    else:
        top2_gap = math.inf
        relative_gap = math.inf

    near_threshold = best_score + max(abs(best_score) * float(uncertainty_tolerance), 1e-6)
    near_best = [cand for cand in all_candidates if float(cand["score"]) <= near_threshold]
    if not near_best:
        near_best = [best]
    certainty, ambiguity = classify_parameter_certainty(
        relative_gap=float(relative_gap),
        near_best=near_best,
        top_candidates=top_candidates,
    )

    near_best_alpha_interval = [
        min(float(cand["alpha"]) for cand in near_best),
        max(float(cand["alpha"]) for cand in near_best),
    ]
    near_best_r_interval = [
        min(float(cand["r_hif_pu"]) for cand in near_best),
        max(float(cand["r_hif_pu"]) for cand in near_best),
    ]

    r_ohm = float(best["r_hif_ohm"])
    fault_v = float(best.get("fault_v_volts") or (1.0 / math.sqrt(3.0) * 1000.0))
    i_amp = fault_v / r_ohm if r_ohm > 0 else math.inf
    p_kw = (fault_v**2) / r_ohm / 1000.0 if r_ohm > 0 else math.inf
    phase_scores = {}
    for phase in phase_candidates:
        phase_items = [cand for cand in all_candidates if cand.get("phase") == phase]
        if phase_items:
            phase_scores[phase] = min(float(cand["score"]) for cand in phase_items)

    return {
        "success": True,
        "method": "model_based_hif_parameter_search",
        "candidate_branch_row0": int(candidate_branch_row0),
        "input_branch_row0": int(candidate_branch_row0),
        "dss_element": dss_element,
        "from_bus": info.get("from_bus"),
        "to_bus": info.get("to_bus"),
        "estimated": {
            "alpha_from_from_bus": float(best["alpha"]),
            "distance_percent_from_from_bus": 100.0 * float(best["alpha"]),
            "phase": best.get("phase"),
            "r_hif_pu": float(best["r_hif_pu"]),
            "r_hif_ohm": r_ohm,
            "g_hif_siemens": 1.0 / r_ohm if r_ohm > 0 else math.inf,
            "i_hif_amp": float(i_amp),
            "p_hif_kw": float(p_kw),
            "q_hif_kvar": 0.0,
        },
        "fit": {
            "weighted_residual_norm": best_score,
            "residual_reduction_vs_no_refinement": residual_reduction,
            "relative_residual_improvement": residual_reduction,
            "localization_certainty": certainty,
            "ambiguity": bool(ambiguity),
            "top2_delta_score": top2_gap if math.isfinite(top2_gap) else None,
            "top2_relative_gap": relative_gap if math.isfinite(relative_gap) else None,
        },
        "uncertainty": {
            "near_best_alpha_interval": near_best_alpha_interval,
            "near_best_r_hif_pu_interval": near_best_r_interval,
            "interval_method": "near_best_score_profile",
            "score_tolerance": float(uncertainty_tolerance),
            "near_best_count": len(near_best),
        },
        "phase_scores": phase_scores,
        "top_parameter_candidates": [
            _candidate_payload(candidate, rank=rank)
            for rank, candidate in enumerate(top_candidates, start=1)
        ],
        "search": {
            "alpha_grid_size": int(alpha_grid_size),
            "r_grid_size": int(r_grid_size),
            "phase_candidates": phase_candidates,
            "coarse_candidates_evaluated": int(len(alphas) * len(r_values) * len(phase_candidates)),
            "total_candidates_evaluated": len(all_candidates),
        },
    }
