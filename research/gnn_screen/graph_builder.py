"""Bus/registered-branch graph construction with no hidden simulation inputs."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any
import numpy as np
from scipy.stats import chi2

from .feature_schema import (SCHEMA_VERSION, MEASUREMENT_CONVENTION, MEASUREMENT_TYPES, NODE_FEATURES,
                             EDGE_FEATURES, GLOBAL_FEATURES, ScreenInputError)
from .wls_features import build_wls_features, configured_case, default_measurement_sigma


def _packet_locations(kind: int, graph: Mapping[str, Any]):
    if kind < 3:
        yield "x", np.ones(len(graph["x"]), dtype=bool), 7 * kind
    else:
        forward = np.asarray(graph["edge_attr"])[:, 37] > 0
        offset = 7 * (kind - 3)
        yield "edge_attr", forward, offset
        yield "edge_attr", ~forward, (offset + 14) % 28


class FeatureScaler:
    """One shared observed/fitted scaler per channel type across training assets.

    Residual amplitude, leverage, masks and configured physical attributes are
    preserved. Sigma references are shared geometric training means per type.
    Never fit separately per asset, snapshot or target network.
    """

    def __init__(self) -> None:
        self.center = np.zeros(7, dtype=np.float64)
        self.scale = np.ones(7, dtype=np.float64)
        self.sigma_reference = np.array([0.001] + [0.01] * 6, dtype=np.float64)
        self.fitted = False

    def fit(self, graphs: Iterable[Mapping[str, Any]], *, split: str = "train") -> "FeatureScaler":
        if split != "train":
            raise ValueError("FeatureScaler may only be fitted on the training split.")
        values: list[list[np.ndarray]] = [[] for _ in range(7)]
        log_sigmas: list[list[np.ndarray]] = [[] for _ in range(7)]
        references = np.array([0.001] + [0.01] * 6)
        for graph in graphs:
            if graph.get("metadata", {}).get("scaler_applied"):
                raise ValueError("Fit requires unscaled training graphs.")
            for kind in range(7):
                # One native packet per physical measurement; reverse copies do
                # not double-weight branch measurements against bus channels.
                name, rows, offset = next(_packet_locations(kind, graph))
                packets = np.asarray(graph[name], dtype=np.float64)[rows, offset:offset + 7]
                packets = packets[packets[:, 5] == 1]
                if len(packets):
                    values[kind].append(packets[:, 0])
                    log_sigmas[kind].append(packets[:, 3] + np.log(references[kind]))
        for kind in range(7):
            if not values[kind]:
                raise ValueError(f"Training population has no available {MEASUREMENT_TYPES[kind]} channels.")
            population = np.concatenate(values[kind])
            if not np.all(np.isfinite(population)):
                raise ValueError("Nonfinite training features.")
            self.center[kind] = population.mean()
            self.scale[kind] = max(float(population.std()), 1e-8)
            self.sigma_reference[kind] = np.exp(np.concatenate(log_sigmas[kind]).mean())
        self.fitted = True
        return self

    def transform(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        if not self.fitted:
            raise ValueError("Fit or load the training scaler before transformation.")
        if graph.get("metadata", {}).get("scaler_applied"):
            raise ValueError("Graph has already been scaled.")
        result = deepcopy(dict(graph))
        references = np.array([0.001] + [0.01] * 6)
        for kind in range(7):
            for name, rows, offset in _packet_locations(kind, result):
                tensor = result[name]
                available = tensor[:, offset + 5] == 1
                selected = rows & available
                tensor[selected, offset:offset + 2] = (
                    tensor[selected, offset:offset + 2] - self.center[kind]) / self.scale[kind]
                tensor[selected, offset + 3] += np.log(references[kind] / self.sigma_reference[kind])
        result.setdefault("metadata", {})["scaler_applied"] = True
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "measurement_types": list(MEASUREMENT_TYPES),
                "fitted": self.fitted, "center": self.center.tolist(), "scale": self.scale.tolist(),
                "sigma_reference": self.sigma_reference.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureScaler":
        if payload.get("schema_version") != SCHEMA_VERSION or tuple(payload.get("measurement_types", ())) != MEASUREMENT_TYPES:
            raise ValueError("Incompatible feature scaler schema.")
        result = cls()
        for field in ("center", "scale", "sigma_reference"):
            value = np.asarray(payload[field], dtype=np.float64)
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid scaler {field}.")
            if field != "center" and np.any(value <= 0):
                raise ValueError(f"Scaler {field} must be positive.")
            setattr(result, field, value)
        result.fitted = bool(payload.get("fitted"))
        if not result.fitted:
            raise ValueError("Cannot load an unfitted scaler for inference.")
        return result


def _array_digest(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value, dtype="<f8")
        digest.update(json.dumps(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def build_graph(case: Mapping[str, Any], z: Any, *, wls_details: Mapping[str, Any] | None = None,
                scaler: FeatureScaler | None = None, measurement_sigma: Any = None,
                measurement_mask: Any = None, max_it: int = 30, tol: float = 1e-8,
                solver_settings: Mapping[str, Any] | None = None,
                metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the fixed 27/40/4 float64 graph, or raise ScreenInputError.

    There is one node per external bus row and two interleaved edges per
    registered branch row (including open/parallel branches). Asset IDs remain
    metadata. The caller must supply the operator's configured model and the
    exporter's corrected shunt convention; no hidden healthy case is used.
    """
    clean = configured_case(case)
    evidence = build_wls_features(clean, z, wls_details=wls_details, measurement_sigma=measurement_sigma,
                                  measurement_mask=measurement_mask, max_it=max_it, tol=tol,
                                  solver_settings=solver_settings)
    bus, branch = clean["bus"], clean["branch"]
    nb, nl = len(bus), len(branch)
    sigma_reference = default_measurement_sigma(nb, nl)
    packets = np.column_stack((evidence["observed"], evidence["fitted"],
        evidence["signed_normalized_residual"], np.log(evidence["sigma"] / sigma_reference),
        evidence["leverage"], evidence["available"], evidence["residual_usable"]))
    bus_ids = bus[:, 0].astype(np.int64)
    mapping = {int(asset_id): row for row, asset_id in enumerate(bus_ids)}
    source = np.array([mapping[int(asset_id)] for asset_id in branch[:, 0]], dtype=np.int64)
    destination = np.array([mapping[int(asset_id)] for asset_id in branch[:, 1]], dtype=np.int64)
    degree = np.bincount(np.r_[source, destination], minlength=nb)
    x = np.column_stack((packets[:nb], packets[nb:2 * nb], packets[2 * nb:3 * nb],
                         bus[:, 1] == 1, bus[:, 1] == 2, bus[:, 1] == 3,
                         bus[:, 4] / clean["baseMVA"], bus[:, 5] / clean["baseMVA"],
                         np.log1p(degree)))
    flow_packets = [packets[3 * nb + k * nl:3 * nb + (k + 1) * nl] for k in range(4)]
    # Preserve native tap encoding. Known unequal terminal voltage bases also
    # identify a nominal-ratio transformer (IEEE14 row 7-8 has native tap=0).
    shift = np.deg2rad(branch[:, 9])
    known_voltage_transition = ((bus[source, 9] > 0) & (bus[destination, 9] > 0)
                                & ~np.isclose(bus[source, 9], bus[destination, 9], rtol=1e-12, atol=0))
    transformer = (branch[:, 8] != 0) | (branch[:, 9] != 0) | known_voltage_transition
    context = np.column_stack((branch[:, 2:5], branch[:, 8], np.cos(shift), np.sin(shift),
                               branch[:, 10], ~transformer, transformer))
    delta = evidence["theta_est_rad"][source] - evidence["theta_est_rad"][destination]
    edge_attr = np.empty((2 * nl, 40), dtype=np.float64)
    edge_attr[0::2] = np.column_stack((*flow_packets, context, np.ones(nl), np.cos(delta), np.sin(delta)))
    edge_attr[1::2] = np.column_stack((flow_packets[2], flow_packets[3], flow_packets[0], flow_packets[1],
                                      context, -np.ones(nl), np.cos(delta), -np.sin(delta)))
    edge_index = np.empty((2, 2 * nl), dtype=np.int64)
    edge_index[:, 0::2] = np.stack((source, destination))
    edge_index[:, 1::2] = np.stack((destination, source))
    normalized = np.abs(evidence["signed_normalized_residual"])
    u = np.array([evidence["wls_objective"] / evidence["dof"], normalized.max(),
                  np.count_nonzero(normalized > 3) / evidence["n_measurements"],
                  np.count_nonzero(normalized > 4) / evidence["n_measurements"]], dtype=np.float64)
    graph_metadata = dict(metadata or {})
    graph_metadata.update({"schema_version": SCHEMA_VERSION, "screen_status": "valid",
        "bus_ids": bus_ids.tolist(), "branch_rows": list(range(nl)),
        "branch_endpoints": branch[:, :2].astype(np.int64).tolist(),
        "measurement_order": list(MEASUREMENT_TYPES), "measurement_convention": MEASUREMENT_CONVENTION,
        "n_measurements": evidence["n_measurements"], "n_states": evidence["n_states"],
        "dof": evidence["dof"], "wls_objective": evidence["wls_objective"],
        "wls_iterations": evidence["iterations"],
        "wls_alarm": bool(evidence["wls_objective"] >= chi2.ppf(0.95, evidence["dof"])),
        "wls_alarm_definition": "chi_square_only", "wls_chi_square_alpha": 0.05,
        "wls_chi_square_threshold": float(chi2.ppf(0.95, evidence["dof"])),
        "solver_settings": evidence["solver_settings"], "scaler_applied": False,
        "configured_model_hash": _array_digest(clean["baseMVA"],
            bus[:, [0, 1, 4, 5, 9] if np.any(bus[:, 9] > 0) else [0, 1, 4, 5]],
            branch[:, [0, 1, 2, 3, 4, 8, 9, 10]]),
        "measurement_hash": _array_digest(evidence["observed"]),
        "covariance_hash": _array_digest(evidence["variance"]),
        "feature_names": {"x": list(NODE_FEATURES), "edge_attr": list(EDGE_FEATURES), "u": list(GLOBAL_FEATURES)}})
    graph = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "u": u,
             "edge_pair": np.repeat(np.arange(nl, dtype=np.int64), 2), "metadata": graph_metadata}
    if not all(np.all(np.isfinite(graph[name])) for name in ("x", "edge_attr", "u")):
        raise ScreenInputError("Nonfinite screen features.", "wls_failure")
    return graph if scaler is None else scaler.transform(graph)
