"""Balanced WLS on immutable physical sensors and candidate logical statuses.

Closed ideal couplers contract voltages and retain independent P/Q transfer
nuisances. Opening one coupler replaces those two flow unknowns with two voltage
unknowns; changing representation cannot win merely by adding free parameters.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2
from pypower.dSbr_dV import dSbr_dV
from pypower.makeYbus import makeYbus

from .measurements import _internal


class _MeasurementModel:
    def __init__(self, base_case, inventory, statuses, processed, sensors):
        self.case = processed["case"]
        self.bus, self.branch = _internal(self.case)
        self.nb, self.nl = len(self.bus), len(self.branch)
        self.nodes = [row["node_id"] for row in inventory["nodes"]]
        self.node_index = {name: index for index, name in enumerate(self.nodes)}
        self.node_bus = np.asarray([processed["node_to_row0"][name] for name in self.nodes], dtype=int)
        self.ref = int(np.flatnonzero(np.asarray(self.case["bus"])[:, 1] == 3)[0])
        self.angle_rows = np.asarray([row for row in range(self.nb) if row != self.ref], dtype=int)
        self.vm_start = self.nb-1
        self.voltage_states = 2*self.nb-1
        self.closed = [row for row in inventory["couplers"] if statuses[row["device_id"]] == 1]
        self.nstate = self.voltage_states+2*len(self.closed)
        self.shunt = np.zeros(len(self.nodes), dtype=complex)
        source_bus = {int(row[0]): row for row in np.asarray(base_case["bus"])}
        for device in inventory["shunts"]:
            row = source_bus[int(device["base_bus"])]
            self.shunt[self.node_index[device["node_id"]]] += complex(row[4], -row[5])/float(base_case["baseMVA"])
        branches = sorted(inventory["branches"], key=lambda row: row["row0"])
        self.from_node = np.asarray([self.node_index[row["from_node"]] for row in branches])
        self.to_node = np.asarray([self.node_index[row["to_node"]] for row in branches])
        _, self.yf, self.yt = makeYbus(float(base_case["baseMVA"]), self.bus, self.branch)
        self.sensor_selection = []
        for sensor in sensors["records"]:
            kind = sensor["kind"]
            if kind in {"Vm", "Pinj", "Qinj"}:
                index = self.node_index[sensor["node_id"]]
            else:
                index = int(sensor["branch_row0"])
            self.sensor_selection.append((kind, index))
        self.last_x = None

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(self.last_x, x):
            return self.last_h, self.last_jac
        va = np.zeros(self.nb)
        va[self.angle_rows] = x[:self.nb-1]
        vm = x[self.vm_start:self.voltage_states]
        v = vm*np.exp(1j*va)
        fa, fm, ta, tm, sf, st = dSbr_dV(self.branch, self.yf, self.yt, v)
        df = np.zeros((self.nl, self.nstate), dtype=complex)
        dt = np.zeros_like(df)
        df[:, :self.nb-1], df[:, self.vm_start:self.voltage_states] = fa.toarray()[:, self.angle_rows], fm.toarray()
        dt[:, :self.nb-1], dt[:, self.vm_start:self.voltage_states] = ta.toarray()[:, self.angle_rows], tm.toarray()
        snode = self.shunt * vm[self.node_bus]**2
        ds = np.zeros((len(self.nodes), self.nstate), dtype=complex)
        ds[np.arange(len(self.nodes)), self.vm_start+self.node_bus] = 2*self.shunt*vm[self.node_bus]
        np.add.at(snode, self.from_node, sf)
        np.add.at(snode, self.to_node, st)
        np.add.at(ds, self.from_node, df)
        np.add.at(ds, self.to_node, dt)
        for index, coupler in enumerate(self.closed):
            pindex, qindex = self.voltage_states+index, self.voltage_states+len(self.closed)+index
            flow = complex(x[pindex], x[qindex])
            for key, sign in (("node_a", 1), ("node_b", -1)):
                node = self.node_index[coupler[key]]
                snode[node] += sign*flow
                ds[node, pindex] = sign
                ds[node, qindex] = sign*1j
        h = np.empty(len(self.sensor_selection))
        jac = np.zeros((len(h), self.nstate))
        for row, (kind, index) in enumerate(self.sensor_selection):
            if kind == "Vm":
                h[row] = vm[self.node_bus[index]]
                jac[row, self.vm_start+self.node_bus[index]] = 1
            elif kind in ("Pinj", "Qinj"):
                h[row] = snode[index].real if kind == "Pinj" else snode[index].imag
                jac[row] = ds[index].real if kind == "Pinj" else ds[index].imag
            else:
                flow, derivative = (sf, df) if kind in ("Pf", "Qf") else (st, dt)
                h[row] = flow[index].real if kind in ("Pf", "Pt") else flow[index].imag
                jac[row] = derivative[index].real if kind in ("Pf", "Pt") else derivative[index].imag
        self.last_x, self.last_h, self.last_jac = np.array(x), h, jac
        return h, jac


def estimate(case, inventory, statuses, observations, sensors, *, chi2_alpha=.05,
             normalized_residual_threshold=4.0, max_nfev=100) -> dict[str, Any]:
    """Fit the same raw observations for one candidate, retaining numerical failure."""
    from .inventory import process_topology
    if (not math.isfinite(chi2_alpha) or not 0 < chi2_alpha < 1
        or not math.isfinite(normalized_residual_threshold) or normalized_residual_threshold <= 0
        or isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1):
        raise ValueError("invalid WLS numerical or alarm configuration")
    if (observations["sensor_inventory_hash"] != sensors["sensor_inventory_hash"]
        or sensors["layout_hash"] != inventory["layout_hash"]
        or observations["sensor_ids"] != [row["sensor_id"] for row in sensors["records"]]):
        raise ValueError("WLS requires the unchanged raw physical sensor inventory")
    count = len(sensors["records"])
    raw_values = observations["values"]
    mask = np.asarray(sensors["available_mask"], dtype=bool)
    covariance = np.asarray(sensors["covariance"], dtype=float)
    if (not isinstance(raw_values, (list, tuple)) or len(raw_values) != count
        or mask.shape != (count,) or covariance.shape != (count, count)
        or not np.isfinite(covariance).all()
        or not np.array_equal(mask, [bool(row["available"]) for row in sensors["records"]])
        or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14)):
        raise ValueError("invalid fixed raw data, availability, or covariance")
    if any(raw_values[index] is not None for index in np.flatnonzero(~mask)):
        raise ValueError("unavailable physical measurements must be redacted as None")
    if any(raw_values[index] is None for index in np.flatnonzero(mask)):
        raise ValueError("available physical measurements must be finite")
    # The numerical placeholders below never enter the objective or exported
    # residuals. They are not observations or inferred zero-flow evidence.
    z = np.asarray([value if mask[index] else 0.0 for index, value in enumerate(raw_values)], dtype=float)
    if not np.isfinite(z[mask]).all():
        raise ValueError("available physical measurements must be finite")
    basic = {"contract": "physical_section_fixed_evidence_wls_v1", "converged": False,
             "observable": False, "plausible": False, "rank": None, "state_dimension": None,
             "available_measurement_count": int(mask.sum()), "raw_measurement_count": count,
             "sensor_inventory_hash": sensors["sensor_inventory_hash"], "chi_square_alpha": chi2_alpha,
             "normalized_residual_threshold": normalized_residual_threshold,
             "coupler_flows_are_estimated_nuisances": True,
             "wls_objective": None, "chi_square_threshold": None, "chi_square_dof": None,
             "max_normalized_residual": None, "chi_square_alarm": None, "normalized_residual_alarm": None}
    try:
        processed = process_topology(case, inventory, statuses)
    except ValueError as exc:
        return {**basic, "candidate_connected": None, "failure_reason": "invalid_or_unknown_candidate_status", "error_detail": str(exc)}
    bus, branch = _internal(processed["case"])
    live = branch[:, 10] > 0
    from_bus, to_bus = branch[live, 0].astype(int), branch[live, 1].astype(int)
    graph = coo_matrix((np.ones(2*len(from_bus)), (np.r_[from_bus, to_bus], np.r_[to_bus, from_bus])), shape=(len(bus), len(bus)))
    ncomponent, labels = connected_components(graph, directed=False)
    connectivity = {"connected": ncomponent == 1, "component_count": int(ncomponent),
                    "components_bus_rows0": [np.flatnonzero(labels == index).tolist() for index in range(ncomponent)],
                    "proof": "active_finite_branch_graph_after_ideal_closed_switch_contraction"}
    basic.update(candidate_connected=ncomponent == 1, connectivity=connectivity)
    if ncomponent != 1:
        return {**basic, "failure_reason": "excluded_by_declared_connected_scope", "excluded_by_declared_scope": True}
    if not mask.any():
        return {**basic, "failure_reason": "no_available_measurements"}
    try:
        model = _MeasurementModel(case, inventory, statuses, processed, sensors)
        r = covariance[np.ix_(mask, mask)]
        diagonal = np.array_equal(r, np.diag(np.diag(r)))
        if diagonal:
            if np.min(np.diag(r)) <= 0:
                raise ValueError("measurement covariance must be positive definite")
            sigma = np.sqrt(np.diag(r))
            chol = np.diag(sigma)
            def whiten(value):
                return value/sigma if value.ndim == 1 else value/sigma[:, None]
        else:
            chol = np.linalg.cholesky(r)
            def whiten(value):
                return solve_triangular(chol, value, lower=True, check_finite=False)
        initial = np.zeros(model.nstate)
        initial[model.vm_start:model.voltage_states] = 1
        lower, upper = np.full(model.nstate, -np.inf), np.full(model.nstate, np.inf)
        lower[model.vm_start:model.voltage_states], upper[model.vm_start:model.voltage_states] = .2, 2.0
        result = least_squares(lambda x: whiten(model.evaluate(x)[0][mask]-z[mask]), initial,
                               jac=lambda x: whiten(model.evaluate(x)[1][mask]),
                               bounds=(lower, upper), x_scale="jac", method="trf", max_nfev=max_nfev,
                               ftol=1e-10, xtol=1e-10, gtol=1e-8)
        prediction, jac = model.evaluate(result.x)
        residual = z-prediction
        weighted_jac = whiten(jac[mask])
        u, singular, _ = np.linalg.svd(weighted_jac, full_matrices=False)
        rank_threshold = max(weighted_jac.shape)*np.finfo(float).eps*singular[0] if len(singular) else 0
        rank = int(np.sum(singular > max(rank_threshold, (singular[0]*1e-9 if len(singular) else 0))))
        dof = int(mask.sum())-rank
        variance = np.diag(r)-np.sum((chol@u[:, :rank])**2, axis=1)
        floor = np.maximum(np.diag(r)*1e-12, 1e-16)
        standardized = np.abs(residual[mask])/np.sqrt(np.maximum(variance, floor))
        objective = float(np.dot(whiten(residual[mask]), whiten(residual[mask])))
        threshold = float(chi2.ppf(1-chi2_alpha, dof)) if dof > 0 else None
        max_nr = float(np.max(standardized))
        observable = rank == model.nstate
        converged = bool(result.success and np.isfinite(result.x).all() and np.isfinite(objective))
        j_alarm = objective >= threshold if threshold is not None else None
        nr_alarm = max_nr >= normalized_residual_threshold
        plausible = bool(converged and observable and dof > 0 and not j_alarm and not nr_alarm)
        reason = ("wls_nonconvergence" if not converged else "state_unobservable" if not observable else
                  "no_redundancy_for_goodness_of_fit" if dof <= 0 else None)
        all_nr = [None]*count
        for index, value in zip(np.flatnonzero(mask), standardized):
            all_nr[int(index)] = float(value)
        return {**basic, "converged": converged, "observable": observable, "plausible": plausible,
                "failure_reason": reason, "rank": rank, "state_dimension": model.nstate,
                "electrical_bus_count": model.nb, "closed_coupler_nuisance_count": 2*len(model.closed),
                "chi_square_dof": dof, "wls_objective": objective, "chi_square_threshold": threshold,
                "max_normalized_residual": max_nr, "chi_square_alarm": j_alarm, "normalized_residual_alarm": nr_alarm,
                "solver_status": int(result.status), "solver_message": result.message, "function_evaluations": int(result.nfev),
                "weighted_jacobian_singular_values": singular.tolist(), "rank_relative_tolerance": 1e-9,
                "residual_variance_floor_count": int(np.sum(variance < floor)),
                "available_sensor_ids": [sensors["records"][index]["sensor_id"] for index in np.flatnonzero(mask)],
                "predicted_values": prediction.tolist(),
                "raw_residuals": [float(value) if mask[index] else None for index, value in enumerate(residual)],
                "normalized_residuals": all_nr,
                "state": {"node_voltage_magnitude_pu": {node: float(result.x[model.vm_start+model.node_bus[index]]) for index, node in enumerate(model.nodes)},
                          "electrical_voltage_angles_rad": [0.0 if index == model.ref else float(result.x[list(model.angle_rows).index(index)]) for index in range(model.nb)],
                          "closed_coupler_flows_pu": {row["device_id"]: {"p": float(result.x[model.voltage_states+index]),
                                                                        "q": float(result.x[model.voltage_states+len(model.closed)+index])}
                                                        for index, row in enumerate(model.closed)}},
                "initialization": "unit_voltage_zero_angle_zero_coupler_flows_no_truth_state"}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {**basic, "failure_reason": "wls_numerical_failure", "error_detail": str(exc)}
