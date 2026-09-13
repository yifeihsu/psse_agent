"""Full-telemetry phase-domain screening against a pristine nominal model.

This is constitutive/nodal screening, not the legacy three-phase NLM. It reads
only bus voltage and external branch-current observations. Disturbance truth,
online load settings, and exported device powers never select a candidate.
Results establish synthetic model consistency, not empirical HIF validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from .validation import _bus_name, _element, _yprim


@dataclass(frozen=True)
class DiagnosticConfig:
    voltage_sigma_pu: float = 1e-5
    current_sigma_pu: float = 1e-4
    detection_sigmas: float = 6.0
    ambiguity_ratio: float = 1.2


def _rect(values: Any) -> Any:
    arr = np.asarray(values)
    return np.stack((arr.real, arr.imag), axis=-1).tolist()


def _decode(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (*shape, 2) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite complex pairs with shape {shape}")
    return arr[..., 0] + 1j * arr[..., 1]


def _abc_y(dss: Any, name: str, bus_names: list[str], zbase: float) -> np.ndarray:
    element = _element(dss, name)
    size = 3 * len(bus_names)
    if not element["enabled"]:
        return np.zeros((size, size), dtype=complex)
    names = {_bus_name(bus): index for index, bus in enumerate(bus_names)}
    projection = np.zeros((len(element["nodes"]), size))
    for terminal, bus in enumerate(element["buses"]):
        for conductor in range(element["ncond"]):
            index = terminal * element["ncond"] + conductor
            node = element["nodes"][index]
            if node == 0:
                continue
            if node not in (1, 2, 3) or _bus_name(bus) not in names:
                raise ValueError(f"unsupported terminal/node in nominal {name}")
            projection[index, 3 * names[_bus_name(bus)] + node - 1] = 1
    return projection.T @ _yprim(dss, element) @ projection * zbase


def capture_nominal_model(dss: Any, registry: Mapping[str, Any], assumptions: Mapping[str, Any]) -> dict[str, Any]:
    """Capture actual pristine ABC primitives and declared nominal injections."""
    if not dss.Solution.Converged():
        raise ValueError("nominal OpenDSS context must be compiled and solved")
    base, kv = float(assumptions["base_mva"]), float(assumptions["base_kv_ll"])
    if not all(math.isfinite(v) and v > 0 for v in (base, kv)):
        raise ValueError("nominal bases must be positive and finite")
    expected_enabled = {str(registry["source"]["element"]).lower()}
    for row in registry["branches"]:
        element = _element(dss, row["dss_element"])
        if element["enabled"] != bool(row["status"]):
            raise ValueError("nominal branch status differs from registry; capture pristine context before disturbance")
        if row["status"]:
            expected_enabled.add(str(row["dss_element"]).lower())
            for end in ("from", "to"):
                expected_enabled.update(str(item["element"] if isinstance(item, Mapping) else item).lower()
                                        for item in row.get("charging_elements", {}).get(end, []))
    for family in ("loads", "generators", "shunts"):
        expected_enabled.update(str(row["element"]).lower() for row in registry.get(family, []))
    active = {str(name).lower() for name in dss.Circuit.AllElementNames() if _element(dss, name)["enabled"]}
    if active != expected_enabled:
        raise ValueError("nominal capture requires exact pristine active-asset coverage")
    for family, api in (("loads", dss.Loads), ("generators", dss.Generators)):
        for row in registry.get(family, []):
            api.Name(str(row["element"]).split(".", 1)[1])
            for field, actual in (("kw", api.kW()), ("kvar", api.kvar())):
                if not math.isclose(float(row[field]), actual, rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("nominal capture requires unchanged registry PQ device settings")
    zbase = kv**2 / base
    buses = sorted(registry["buses"], key=lambda row: row["row0"])
    bus_names = {int(row["external_bus"]): row["dss_bus"] for row in buses}
    if len(bus_names) != len(buses) or any(float(row["kv_ll"]) != kv for row in buses):
        raise ValueError("nominal registry needs unique buses on the declared uniform voltage base")
    net = {bus: np.zeros(3, dtype=complex) for bus in bus_names}
    load = {bus: np.zeros(3, dtype=complex) for bus in bus_names}
    shunt = {bus: np.zeros((3, 3), dtype=complex) for bus in bus_names}
    for family, sign in (("loads", -1), ("generators", 1)):
        for row in registry.get(family, []):
            bus, phase = int(row["bus"]), int(row["phase"]) - 1
            power = complex(row["kw"], row["kvar"]) / (base * 1000 / 3)
            net[bus][phase] += sign * power
            if family == "loads":
                load[bus][phase] += power
    for row in registry.get("shunts", []):
        bus = int(row["bus"])
        shunt[bus] += _abc_y(dss, row["element"], [bus_names[bus]], zbase)
    source = registry["source"]
    source_bus = int(source["bus"])
    source_y = _abc_y(dss, source["element"], [bus_names[source_bus]], zbase)
    a = np.exp(2j * np.pi / 3)
    emf = complex(*source["emf_pu"]) * np.asarray([1, a*a, a])
    branch_rows = []
    for row in sorted(registry["branches"], key=lambda item: item["branch_row0"]):
        fb, tb = int(row["from_bus"]), int(row["to_bus"])
        y = np.zeros((6, 6), dtype=complex)
        if int(row["status"]):
            names = [row["dss_element"]]
            for end in ("from", "to"):
                names.extend(item["element"] if isinstance(item, Mapping) else item
                             for item in row.get("charging_elements", {}).get(end, []))
            for name in names:
                y += _abc_y(dss, name, [bus_names[fb], bus_names[tb]], zbase)
        branch_rows.append({"asset_id": row["asset_id"], "branch_row0": int(row["branch_row0"]),
                            "from_bus": fb, "to_bus": tb, "status": int(row["status"]),
                            "kind": str(row["dss_element"]).split(".")[0].lower(),
                            "terminal_y_pu_rect": _rect(y)})
    return {"contract": "pristine_compiled_phase_screening_model_v1", "base_mva": base, "base_kv_ll": kv,
            "branches": branch_rows,
            "buses": [{"bus": bus, "dss_bus": bus_names[bus], "nominal_net_pq_pu_rect": _rect(net[bus]),
                       "nominal_load_pq_pu_rect": _rect(load[bus]), "shunt_y_pu_rect": _rect(shunt[bus])}
                      for bus in bus_names],
            "source": {"bus": source_bus, "y_pu_rect": _rect(source_y), "emf_pu_rect": _rect(emf)}}


def _jacobian(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Real Jacobian for H*dV + G*conj(dV), ordered real ABC then imag ABC."""
    return np.block([[(h+g).real, -(h-g).imag], [(h+g).imag, (h-g).real]])


def _scores(values: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    n = len(values)
    sigma = np.sqrt(np.maximum((np.diag(covariance)[:n] + np.diag(covariance)[n:]) / 2, 1e-30))
    return np.abs(values) / sigma


def _statistic(values: np.ndarray, covariance: np.ndarray) -> float:
    real = np.r_[values.real, values.imag]
    return float(real @ np.linalg.pinv(covariance, rcond=1e-12) @ real)


def _line_estimate(y: np.ndarray, volts: np.ndarray, currents: np.ndarray, phase: int, config: DiagnosticConfig) -> dict[str, Any]:
    series = -y[:3, 3:]
    z = np.linalg.inv(series)
    shf, sht = y[:3, :3] - series, y[3:, 3:] - series

    def estimate(x: np.ndarray) -> tuple[float, complex, float]:
        vf, vt, ii, it = x[:3], x[3:6], x[6:9], x[9:12]
        jf, jt = ii-shf@vf, it-sht@vt
        fault = jf + jt
        q, rhs = z@fault, vf-vt+z@jt
        denominator = float(np.vdot(q, q).real)
        if denominator < 1e-24 or abs(fault[phase]) < 1e-12:
            return math.nan, complex(math.nan, math.nan), denominator
        alpha = float(np.vdot(q, rhs).real / denominator)
        vhidden = vf - alpha*z@jf
        return alpha, vhidden[phase] / fault[phase], float(np.linalg.norm(rhs-alpha*q))

    x = np.r_[volts, currents]
    alpha, resistance, mismatch = estimate(x)
    if not math.isfinite(alpha) or not np.isfinite(resistance) or not math.isfinite(mismatch):
        return {"alpha_estimate": None, "resistance_pu_estimate": None,
                "distance_observable": False, "parameter_estimates_accepted": False,
                "shunt_hypothesis_consistent": False, "reason": "insufficient_fault_voltage_drop"}
    variances = np.r_[np.full(6, config.voltage_sigma_pu**2), np.full(6, config.current_sigma_pu**2)]
    alpha_var, resistance_var = 0.0, 0.0
    for index, variance in enumerate(variances):
        for direction in (1, 1j):
            step = 1e-7
            plus, minus = x.copy(), x.copy()
            plus[index] += direction*step
            minus[index] -= direction*step
            a_plus, r_plus, _ = estimate(plus)
            a_minus, r_minus, _ = estimate(minus)
            if not all(math.isfinite(v) for v in (a_plus, a_minus, r_plus.real, r_minus.real)):
                alpha_var = resistance_var = math.inf
                break
            alpha_var += ((a_plus-a_minus)/(2*step))**2 * variance
            resistance_var += ((r_plus.real-r_minus.real)/(2*step))**2 * variance
    alpha_sigma, resistance_sigma = math.sqrt(alpha_var), math.sqrt(resistance_var)
    distance_observable = bool(0 < alpha < 1 and math.isfinite(alpha_sigma) and alpha_sigma <= 0.1)
    resistive = bool(resistance.real > 0 and math.isfinite(resistance_sigma)
                     and abs(resistance.imag) <= max(0.2*abs(resistance.real), 3*resistance_sigma))
    consistent = bool(0 < alpha < 1 and resistive)
    accepted = bool(consistent and distance_observable and resistance_sigma <= 0.25*resistance.real
                    and alpha-3*alpha_sigma > 0 and alpha+3*alpha_sigma < 1)
    return {"alpha_estimate": alpha, "alpha_sigma_linearized": alpha_sigma if math.isfinite(alpha_sigma) else None,
            "resistance_pu_estimate": float(resistance.real), "resistance_imaginary_pu": float(resistance.imag),
            "resistance_sigma_linearized_pu": resistance_sigma if math.isfinite(resistance_sigma) else None,
            "distance_observable": distance_observable,
            "resistive_shunt_consistent": resistive, "shunt_hypothesis_consistent": consistent,
            "parameter_estimates_accepted": accepted,
            "terminal_voltage_fit_residual_pu": mismatch,
            "estimator": "full_matrix_two_terminal_analytic_shunt_fit",
            "uncertainty": "first_order_sensor_noise_propagation_not_empirical_coverage"}


def screen_measurements(telemetry: Mapping[str, Any], nominal_model: Mapping[str, Any], *, config: DiagnosticConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Rank nominal branch and nodal violations without receiving fault truth."""
    config = DiagnosticConfig(**dict(config)) if isinstance(config, Mapping) else config or DiagnosticConfig()
    for name, value in asdict(config).items():
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.ambiguity_ratio < 1:
        raise ValueError("ambiguity_ratio must be at least one")
    bus_specs = {int(row["bus"]): row for row in nominal_model["buses"]}
    voltage_rows = list(telemetry["three_phase_voltages"])
    volts = {int(row["external_bus"]): _decode(row["vln_pu_rect"], (3,), "bus voltages") for row in voltage_rows}
    if len(volts) != len(voltage_rows) or set(volts) != set(bus_specs) or any(np.min(np.abs(v)) < 0.1 for v in volts.values()):
        raise ValueError("complete, unique, energized nominal bus voltage observations are required")
    current_rows = list(telemetry["three_phase_branch_currents"])
    observed = {str(row["asset_id"]): row for row in current_rows}
    nominal_branches = list(nominal_model["branches"])
    if len(observed) != len(current_rows) or set(observed) != {row["asset_id"] for row in nominal_branches}:
        raise ValueError("complete, unique nominal external branch-current observations are required")
    sv2, si2 = config.voltage_sigma_pu**2, config.current_sigma_pu**2
    nodal_current = {bus: np.zeros(3, dtype=complex) for bus in bus_specs}
    degree = {bus: 0 for bus in bus_specs}
    branch_rank = []
    branch_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    d = np.column_stack((np.eye(3), np.eye(3)))
    for nominal in nominal_branches:
        row = observed[nominal["asset_id"]]
        fb, tb = int(nominal["from_bus"]), int(nominal["to_bus"])
        if (int(row["from_bus"]), int(row["to_bus"]), int(row["branch_row0"])) != (fb, tb, int(nominal["branch_row0"])):
            raise ValueError("observed branch terminal identity differs from the nominal registry")
        y = _decode(nominal["terminal_y_pu_rect"], (6, 6), "terminal Y")
        v = np.r_[volts[fb], volts[tb]]
        i = np.r_[_decode(row["i_from_pu_rect"], (3,), "from current"), _decode(row["i_to_pu_rect"], (3,), "to current")]
        nodal_current[fb] += i[:3]
        nodal_current[tb] += i[3:]
        degree[fb] += 1
        degree[tb] += 1
        residual = i-y@v
        covariance = si2*np.eye(6) + sv2*y@y.conj().T
        sigma = np.sqrt(np.maximum(np.diag(covariance).real, 1e-30))
        raw_score = np.abs(residual)/sigma
        differential = d@residual
        # Shared terminal-voltage noise is correlated: D*C*D^H retains series
        # cancellation instead of adding independent terminal residual variances.
        dcovariance = d@covariance@d.T
        dsigma = np.sqrt(np.maximum(np.diag(dcovariance).real, 1e-30))
        differential_score = np.abs(differential)/dsigma
        phase = int(np.argmax(differential_score))
        score = float(max(np.max(raw_score), np.max(differential_score)))
        branch_rank.append({"asset_id": nominal["asset_id"], "branch_row0": nominal["branch_row0"],
                            "from_bus": fb, "to_bus": tb, "kind": nominal["kind"],
                            "normalized_residual": score, "constitutive_normalized_residual": float(np.max(raw_score)),
                            "differential_normalized_residual": float(np.max(differential_score)),
                            "phase": phase+1, "phase_label": "ABC"[phase],
                            "phase_differential_normalized_residuals": differential_score.tolist(),
                            "terminal_residual_pu_rect": _rect(residual), "differential_pu_rect": _rect(differential),
                            "differential_sigma_per_component_pu": dsigma.tolist(),
                            "nominal_linear_gaussian_chi_square_statistic": float(np.vdot(residual, np.linalg.solve(covariance, residual)).real),
                            "nominal_statistic_degrees_of_freedom": 12,
                            "anomalous": score >= config.detection_sigmas})
        branch_data[nominal["asset_id"]] = y, v, i
    branch_rank.sort(key=lambda row: (-row["normalized_residual"], row["branch_row0"]))

    source = nominal_model["source"]
    source_bus = int(source["bus"])
    source_y = _decode(source["y_pu_rect"], (3, 3), "source Y")
    emf = _decode(source["emf_pu_rect"], (3,), "source emf")
    p = np.eye(3)-np.ones((3, 3))/3
    pr = np.block([[p, np.zeros((3, 3))], [np.zeros((3, 3)), p]])
    total_projection = np.asarray([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    node_rank = []
    for bus, nominal in bus_specs.items():
        v, obs = volts[bus], nodal_current[bus]
        s = _decode(nominal["nominal_net_pq_pu_rect"], (3,), "nominal PQ")
        shunt = _decode(nominal["shunt_y_pu_rect"], (3, 3), "shunt Y")
        sy = source_y if bus == source_bus else np.zeros((3, 3), dtype=complex)
        isource = sy@(emf-v) if bus == source_bus else np.zeros(3, dtype=complex)
        expected = np.conj(s/v)-shunt@v+isource
        residual = obs-expected
        jac = _jacobian(shunt+sy, np.diag(np.conj(s)/(v.conj()**2)))
        ccov = degree[bus]*si2*np.eye(6) + sv2*jac@jac.T
        current_scores = _scores(residual, ccov)
        k = shunt+sy
        q = obs+k@v-(sy@emf if bus == source_bus else 0)
        delta = s-v*q.conj()  # additional load power on each phase-power base
        jv = _jacobian(-np.diag(q.conj()), -np.diag(v)@k.conj())
        ji = _jacobian(np.zeros((3, 3)), -np.diag(v))
        pcov = sv2*jv@jv.T + degree[bus]*si2*ji@ji.T
        contrast, contrast_cov = p@delta, pr@pcov@pr.T
        spread_scores = _scores(contrast, contrast_cov)
        total_cov = total_projection@pcov@total_projection.T
        total_sigma = math.sqrt(max(float(np.trace(total_cov))/2, 1e-30))
        total_score = float(abs(sum(delta))/total_sigma)
        spread = float(np.max(spread_scores))
        node_rank.append({"bus": bus, "normalized_residual": float(np.max(current_scores)),
                          "phase_power_spread_normalized_residual": spread,
                          "phase_power_change_pu_rect": _rect(delta), "phase_power_contrast_pu_rect": _rect(contrast),
                          "total_power_change_normalized_residual": total_score,
                          "total_preserving_within_noise": total_score < config.detection_sigmas,
                          "incident_terminal_count": degree[bus], "source_boundary_bus": bus == source_bus,
                          "phase_current_sigma_per_component_pu": np.sqrt((np.diag(ccov)[:3]+np.diag(ccov)[3:])/2).tolist(),
                          "linearized_nodal_chi_square_statistic": _statistic(residual, ccov),
                          "linearized_power_contrast_statistic": _statistic(contrast, contrast_cov),
                          "anomalous": max(float(np.max(current_scores)), spread) >= config.detection_sigmas,
                          "unbalance_like": spread >= config.detection_sigmas and total_score < config.detection_sigmas})
    node_rank.sort(key=lambda row: (-max(row["normalized_residual"], row["phase_power_spread_normalized_residual"]), row["bus"]))
    branch_bad = [row for row in branch_rank if row["anomalous"]]
    node_bad = [row for row in node_rank if row["anomalous"]]
    reasons: list[str] = []
    hif, unbalance = None, None
    if branch_bad:
        top = branch_bad[0]
        runner = branch_rank[1]["normalized_residual"] if len(branch_rank) > 1 else 0
        phase_scores = sorted(top["phase_differential_normalized_residuals"], reverse=True)
        if (len(branch_bad) == 1 and top["kind"] == "line"
            and top["differential_normalized_residual"] >= config.detection_sigmas
            and top["normalized_residual"] >= config.ambiguity_ratio*runner
            and phase_scores[0] >= config.ambiguity_ratio*phase_scores[1]):
            hif = {key: top[key] for key in ("asset_id", "branch_row0", "phase", "phase_label", "normalized_residual")}
            hif.update(_line_estimate(*branch_data[top["asset_id"]], top["phase"]-1, config))
        else:
            reasons.append("branch_mismatch_not_uniquely_consistent_with_one_line_and_phase")
    if node_bad:
        top = node_bad[0]
        top_score = max(top["normalized_residual"], top["phase_power_spread_normalized_residual"])
        runner = max(node_rank[1]["normalized_residual"], node_rank[1]["phase_power_spread_normalized_residual"]) if len(node_rank) > 1 else 0
        if len(node_bad) == 1 and top["unbalance_like"] and top_score >= config.ambiguity_ratio*runner:
            unbalance = {key: top[key] for key in ("bus", "phase_power_change_pu_rect", "phase_power_spread_normalized_residual", "total_preserving_within_noise")}
        else:
            reasons.append("nodal_mismatch_not_uniquely_consistent_with_total_preserving_phase_load_change")
    detected = bool(branch_bad or node_bad)
    ambiguous = bool(reasons)
    classification = ("ambiguous" if ambiguous else "mixed" if hif and unbalance else
                      "hif_like_branch_mismatch" if hif else "load_unbalance" if unbalance else
                      "no_detectable_anomaly")
    return {"contract": "full_telemetry_constitutive_nodal_screen_v1", "classification": classification,
            "anomaly_detected": detected, "ambiguous": ambiguous, "ambiguity_reasons": reasons,
            "hif_candidate": hif, "unbalance_candidate": unbalance,
            "max_branch_normalized_residual": branch_rank[0]["normalized_residual"] if branch_rank else 0,
            "max_nodal_normalized_residual": max((max(row["normalized_residual"], row["phase_power_spread_normalized_residual"]) for row in node_rank), default=0),
            "branch_ranking": branch_rank, "nodal_ranking": node_rank, "configuration": asdict(config),
            "interpretation": {"legacy_nlm_executed": False, "full_phase_telemetry_required": True,
                "no_anomaly_is_not_proof_of_no_fault": True,
                "normalized_residual": "complex magnitude divided by propagated per-real/imaginary component noise scale; gate is not a univariate z test",
                "statistics": "linear nominal branch and first-order nodal Gaussian statistics; not empirically calibrated p-values",
                "source_bus_limitation": "stiff Thevenin boundary amplifies voltage-noise uncertainty and can hide source-bus load redistribution",
                "physics_scope": "fundamental-frequency shunt consistency; no arcing waveform or empirical HIF validation"}}
