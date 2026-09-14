"""Fixed physical sensor deployment and covariance-preserving observations."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

import numpy as np
from pypower.makeYbus import makeYbus


def build_measurement_inventory(inventory: Mapping[str, Any], profile: str = "direct", *, covariance=None) -> dict[str, Any]:
    """Freeze availability before an erroneous device is chosen; retain all rows."""
    if profile not in {"direct", "indirect_even", "indirect_odd", "voltage_only"}:
        raise ValueError("unknown fixed measurement profile")
    records = []
    masked = sorted(int(row["row0"]) for row in inventory["branches"]
                    if profile.startswith("indirect_") and int(row["row0"]) % 2 == (0 if profile == "indirect_even" else 1))
    for kind in ("Vm", "Pinj", "Qinj"):
        for node in inventory["nodes"]:
            records.append({"sensor_id": f"{kind}:{node['node_id']}", "kind": kind,
                            "node_id": node["node_id"], "sigma": .001 if kind == "Vm" else .01,
                            "available": profile != "voltage_only" or kind == "Vm"})
    for kind in ("Pf", "Qf", "Pt", "Qt"):
        for branch in sorted(inventory["branches"], key=lambda row: row["row0"]):
            records.append({"sensor_id": f"{kind}:{branch['asset_id']}", "kind": kind,
                            "branch_row0": int(branch["row0"]), "asset_id": branch["asset_id"],
                            "sigma": .01, "available": profile != "voltage_only" and int(branch["row0"]) not in masked})
    variances = np.asarray([row["sigma"]**2 for row in records])
    raw_covariance = np.diag(variances) if covariance is None else np.asarray(covariance, dtype=float)
    if (raw_covariance.shape != (len(records), len(records)) or not np.isfinite(raw_covariance).all()
        or not np.allclose(raw_covariance, raw_covariance.T, rtol=0, atol=1e-14)
        or not np.allclose(np.diag(raw_covariance), variances, rtol=1e-12, atol=1e-16)):
        raise ValueError("raw covariance must preserve declared sensor variances and be symmetric")
    if covariance is not None:
        np.linalg.cholesky(raw_covariance)
    result = {"contract": "fixed_physical_logical_sensor_inventory_v1", "layout_hash": inventory["layout_hash"],
              "profile": profile, "records": records, "available_mask": [row["available"] for row in records],
              "masked_branch_rows0": masked, "masked_sensor_ids": [row["sensor_id"] for row in records if not row["available"]],
              "covariance": raw_covariance.tolist(),
              "availability_basis": "predeclared_deployment_independent_of_true_or_reported_status",
              "section_injection": "actual_allocated_generation_minus_load_excluding_local_shunt_consumption",
              "coupler_flow_sensors": False}
    result["sensor_inventory_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def _internal(case):
    bus = np.asarray(case["bus"], dtype=float).copy()
    branch = np.asarray(case["branch"], dtype=float).copy()
    ids = {int(row[0]): index for index, row in enumerate(bus)}
    bus[:, 0] = np.arange(len(bus))
    branch[:, 0] = [ids[int(value)] for value in branch[:, 0]]
    branch[:, 1] = [ids[int(value)] for value in branch[:, 1]]
    return bus, branch


def expected_measurements(base_operating_case, inventory, true_statuses, physical_solution, sensors) -> list[float]:
    """Read a true PF once: section injections come from allocated equipment."""
    from .inventory import process_topology
    processed = process_topology(base_operating_case, inventory, true_statuses)
    expected_case = processed["case"]
    if not physical_solution.get("success", False):
        raise ValueError("measurement generation requires a successful true physical PF")
    bus, branch, gen = [np.asarray(physical_solution[key], dtype=float) for key in ("bus", "branch", "gen")]
    for table in (bus, branch, gen):
        if not np.all(np.isfinite(table)):
            raise ValueError("physical PF evidence contains non-finite values")
    expected_bus, expected_branch, expected_gen = [np.asarray(expected_case[key]) for key in ("bus", "branch", "gen")]
    if (bus.shape[0] != expected_bus.shape[0] or branch.shape[0] != expected_branch.shape[0]
        or gen.shape[0] != expected_gen.shape[0]
        or not np.allclose(bus[:, [0, 2, 3, 4, 5]], expected_bus[:, [0, 2, 3, 4, 5]], rtol=0, atol=1e-8)
        or not np.allclose(branch[:, :13], expected_branch[:, :13], rtol=0, atol=1e-10)
        or not np.array_equal(gen[:, [0, 7]], expected_gen[:, [0, 7]])
        or float(physical_solution["baseMVA"]) != float(expected_case["baseMVA"])):
        raise ValueError("physical PF topology, equipment allocation, or row identities differ from true statuses")
    if sensors["layout_hash"] != inventory["layout_hash"]:
        raise ValueError("measurement inventory belongs to a different physical layout")
    base = float(physical_solution["baseMVA"])
    source_bus = {int(row[0]): row for row in np.asarray(base_operating_case["bus"])}
    injection = {node["node_id"]: 0j for node in inventory["nodes"]}
    for device in inventory["generators"]:
        row = gen[int(device["gen_row0"])]
        if row[7] > 0:
            injection[device["node_id"]] += complex(row[1], row[2])/base
    for device in inventory["loads"]:
        row = source_bus[int(device["base_bus"])]
        injection[device["node_id"]] -= complex(row[2], row[3])*float(device["fraction"])/base
    internal_bus, internal_branch = _internal(physical_solution)
    ybus, yf, yt = makeYbus(base, internal_bus, internal_branch)
    voltage = bus[:, 7]*np.exp(1j*np.deg2rad(bus[:, 8]))
    sf = voltage[internal_branch[:, 0].astype(int)]*np.conj(yf@voltage)
    st = voltage[internal_branch[:, 1].astype(int)]*np.conj(yt@voltage)
    aggregate = np.zeros(len(bus), dtype=complex)
    for node, value in injection.items():
        aggregate[int(processed["node_to_row0"][node])] += value
    if np.max(np.abs(aggregate-voltage*np.conj(ybus@voltage))) > 1e-6:
        raise ValueError("true physical PF injections do not satisfy allocated-equipment power balance")
    values = []
    for sensor in sensors["records"]:
        kind = sensor["kind"]
        if kind == "Vm":
            value = abs(voltage[int(processed["node_to_row0"][sensor["node_id"]])])
        elif kind in ("Pinj", "Qinj"):
            value = injection[sensor["node_id"]].real if kind == "Pinj" else injection[sensor["node_id"]].imag
        else:
            flow = sf[int(sensor["branch_row0"])] if kind in ("Pf", "Qf") else st[int(sensor["branch_row0"])]
            value = flow.real if kind in ("Pf", "Pt") else flow.imag
        values.append(float(value))
    return values


def sample_measurements(truth_values, sensors, *, seed: int = 0, noise: bool = True) -> dict[str, Any]:
    values = np.asarray(truth_values, dtype=float)
    covariance = np.asarray(sensors["covariance"], dtype=float)
    count = len(sensors["records"])
    if values.shape != (count,) or not np.isfinite(values).all() or covariance.shape != (count, count) or not np.isfinite(covariance).all():
        raise ValueError("raw truth/covariance dimensions or values are invalid")
    if noise:
        rng = np.random.default_rng(seed)
        if np.array_equal(covariance, np.diag(np.diag(covariance))):
            if np.min(np.diag(covariance)) <= 0:
                raise ValueError("measurement noise variances must be positive")
            values = values + np.sqrt(np.diag(covariance))*rng.standard_normal(count)
        else:
            values = values + np.linalg.cholesky(covariance)@rng.standard_normal(count)
    published = [float(value) if available else None for value, available in zip(values, sensors["available_mask"])]
    return {"contract": "fixed_raw_logical_measurement_observations_v1", "values": published,
            "sensor_ids": [row["sensor_id"] for row in sensors["records"]],
            "sensor_inventory_hash": sensors["sensor_inventory_hash"]}


def aggregate_measurements(observations, sensors, aggregation_matrix) -> dict[str, Any]:
    """Preserve original evidence and propagate full covariance as A R A^T."""
    matrix = np.asarray(aggregation_matrix, dtype=float)
    count = len(sensors["records"])
    if matrix.ndim != 2 or matrix.shape[1] != count or not np.isfinite(matrix).all():
        raise ValueError("invalid physical aggregation matrix")
    if observations["sensor_inventory_hash"] != sensors["sensor_inventory_hash"]:
        raise ValueError("observation inventory mismatch")
    for row in matrix:
        used = np.flatnonzero(row)
        if len(used) > 1 and any(sensors["records"][index]["kind"] == "Vm" for index in used):
            raise ValueError("do not aggregate voltage sensors across physical sections")
    used = np.flatnonzero(np.any(matrix != 0, axis=0))
    if any(not sensors["available_mask"][index] for index in used):
        raise ValueError("aggregation cannot use unavailable physical measurements")
    reduced = matrix[:, used]
    values = np.asarray([observations["values"][index] for index in used], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("aggregation needs finite available observations")
    covariance = np.asarray(sensors["covariance"])[np.ix_(used, used)]
    return {"contract": "covariance_preserving_physical_aggregation_v1",
            "values": (reduced@values).tolist(),
            "covariance": (reduced@covariance@reduced.T).tolist(), "aggregation_matrix": matrix.tolist(),
            "raw_observations": copy.deepcopy(observations), "raw_sensor_ids": [row["sensor_id"] for row in sensors["records"]],
            "raw_available_mask": list(sensors["available_mask"]),
            "raw_covariance": copy.deepcopy(sensors["covariance"]),
            "candidate_comparison_requires_retained_raw_observations": True}
