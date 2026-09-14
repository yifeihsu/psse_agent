"""Physical IEEE logical-switch scenarios, independent of diagnostic success.

The source stage freezes equipment and sensors first, solves each TRUE topology
once, then derives reported-status errors from the same observations. Cases
rejected by physical admission remain in the manifest. Wrong-model estimation
failures are never physical admission failures.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from itertools import combinations

import numpy as np
from pypower.api import ppoption, runpf
from pypower.makeYbus import makeYbus

from psse_env.systems import resolve_system
from Transmission.generate_measurements import solve_ac_opf
from .inventory import build_inventory, process_topology
from .measurements import build_measurement_inventory, expected_measurements, sample_measurements


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(native(value), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(payload), indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def case_snapshot(case):
    result = {key: deepcopy(case[key]) for key in ("version", "baseMVA", "bus", "gen", "branch", "gencost")}
    if "success" in case:
        result["success"] = bool(case["success"])
    return result


def audit_physical_solution(solution, opf_solution=None):
    """Check explicit generator/load arithmetic against the solved admittances."""
    base = float(solution["baseMVA"])
    bus, branch, gen = (np.asarray(solution[key], dtype=float) for key in ("bus", "branch", "gen"))
    if not all(np.isfinite(a).all() for a in (bus, branch, gen)):
        return {"passed": False, "reason": "nonfinite_physical_solution"}
    rows = {int(row[0]): index for index, row in enumerate(bus)}
    internal_bus, internal_branch = bus.copy(), branch.copy()
    internal_bus[:, 0] = np.arange(len(bus))
    for column in (0, 1):
        internal_branch[:, column] = [rows[int(value)] for value in branch[:, column]]
    ybus, yf, yt = makeYbus(base, internal_bus, internal_branch)
    voltage = bus[:, 7]*np.exp(1j*np.deg2rad(bus[:, 8]))
    network_injection = voltage*np.conj(ybus@voltage)
    equipment_injection = -(bus[:, 2]+1j*bus[:, 3])/base
    active_gen = gen[gen[:, 7] > 0]
    for unit in active_gen:
        equipment_injection[rows[int(unit[0])]] += complex(unit[1], unit[2])/base
    sf = voltage[internal_branch[:, 0].astype(int)]*np.conj(yf@voltage)
    st = voltage[internal_branch[:, 1].astype(int)]*np.conj(yt@voltage)
    flow_error = max(np.max(np.abs(sf-(branch[:, 13]+1j*branch[:, 14])/base)),
                     np.max(np.abs(st-(branch[:, 15]+1j*branch[:, 16])/base)))
    vm_violation = max(0., float(np.max(bus[:, 12]-bus[:, 7])), float(np.max(bus[:, 7]-bus[:, 11])))
    gen_violation = max(0., float(np.max(active_gen[:, 9]-active_gen[:, 1])),
                        float(np.max(active_gen[:, 1]-active_gen[:, 8])),
                        float(np.max(active_gen[:, 4]-active_gen[:, 2])),
                        float(np.max(active_gen[:, 2]-active_gen[:, 3])))/base
    enabled_limits = (branch[:, 10] == 1) & (branch[:, 5] > 0)
    thermal_violation = max(0., float(np.max(np.maximum(np.abs(sf[enabled_limits]), np.abs(st[enabled_limits]))
                                            - branch[enabled_limits, 5]/base))) if enabled_limits.any() else 0.
    off = branch[:, 10] == 0
    off_flow = max(float(np.max(np.abs(sf[off]))), float(np.max(np.abs(st[off])))) if off.any() else 0.
    checks = {
        "generator_minus_load_vs_network_injection_pu": (float(np.max(np.abs(equipment_injection-network_injection))), 1e-6),
        "stored_vs_admittance_branch_flow_pu": (float(flow_error), 1e-7),
        "available_disconnected_branch_flow_pu": (off_flow, 1e-10),
        "voltage_bound_violation_pu": (vm_violation, 1e-5),
        "generator_bound_violation_pu": (gen_violation, 1e-5),
        "declared_rate_a_violation_pu": (thermal_violation, 1e-5),
    }
    if opf_solution is not None:
        opf_bus = np.asarray(opf_solution["bus"])
        checks["opf_followed_by_pf_voltage_consistency_pu"] = (float(np.max(np.abs(bus[:, 7]-opf_bus[:, 7]))), 1e-4)
    return {"passed": bool(solution.get("success") and all(error <= limit for error, limit in checks.values())),
            "checks": {key: {"max_error": error, "limit": limit, "passed": error <= limit}
                       for key, (error, limit) in checks.items()},
            "voltage_min_pu": float(min(bus[:, 7])), "voltage_max_pu": float(max(bus[:, 7])),
            "electrical_bus_count": len(bus), "branch_count": len(branch),
            "rating_scope": "canonical numerical bounds, not verified equipment capabilities"}


def solve_true_world(base_case, inventory, true_statuses):
    """The only OPF/PF entry point; reported statuses never enter this function."""
    compiled = process_topology(base_case, inventory, true_statuses)
    connectivity = compiled["connectivity"]
    connected = connectivity.get("connected", connectivity.get("component_count") == 1)
    if not connected:
        return {"admitted": False, "reason": "islanding_outside_connected_operating_scope",
                "connectivity": connectivity}
    try:
        optimized = solve_ac_opf(compiled["case"])
        if optimized is None or not optimized.get("success"):
            return {"admitted": False, "reason": "opf_nonconvergence_not_a_proof_of_infeasibility",
                    "connectivity": connectivity}
        solved, success = runpf(deepcopy(optimized), ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=40))
        if not success:
            return {"admitted": False, "reason": "post_opf_power_flow_nonconvergence", "connectivity": connectivity}
        audit = audit_physical_solution(solved, optimized)
        if not audit["passed"]:
            return {"admitted": False, "reason": "physical_solution_contract_failed", "physics": audit,
                    "connectivity": connectivity}
        # Operator generation settings are the dispatched physical settings.
        # Their buses remain on the immutable base-equipment basis.
        operating_case = case_snapshot(base_case)
        operating_case["gen"][:, 1:21] = solved["gen"][:, 1:21]
        return {"admitted": True, "operating_case": operating_case, "solution": case_snapshot(solved),
                "physics": audit, "connectivity": connectivity}
    except Exception as exc:
        return {"admitted": False, "reason": "physical_solver_exception", "error": f"{type(exc).__name__}: {exc}",
                "connectivity": connectivity}


def fixed_holdout_devices(inventory):
    """A structural holdout declaration, chosen without measurements or labels."""
    ids = sorted(inventory["normal_statuses"])
    ordered = sorted(ids, key=lambda item: digest([inventory["layout_hash"], "structural_holdout_v1", item]))
    return ordered[:max(1, len(ids)//5)]


def assign_root_splits(rows, *, heldout_device_ids=()):
    """Group all noise/overlay derivatives; reserve entire groups for holdouts.

    A group that contains a held-out error location moves in its entirety to
    structural_test. This may reduce family coverage, which callers must audit;
    it never leaks the same underlying measurements into training and test.
    """
    heldout = set(heldout_device_ids)
    heldout_groups = {row["parent_physical_root"] for row in rows
                      if heldout.intersection(row.get("error_device_ids", []))}
    result = {}
    for row in rows:
        parent = row["parent_physical_root"]
        fraction = int(digest(["root_split_v1", parent])[:12], 16)/16**12
        result[row["scenario_id"]] = ("structural_test" if parent in heldout_groups else
                                       "train" if fraction < .7 else "development" if fraction < .85 else "test")
    return result


def overlay_index(inventory, sensors, device, *, adjacent):
    branch = next((row for row in inventory["branches"] if row["device_id"] == device), None)
    coupler = next((row for row in inventory["couplers"] if row["device_id"] == device), None)
    nodes = {branch["from_node"], branch["to_node"]} if branch else {coupler["node_a"], coupler["node_b"]}
    # Distance is measured in the frozen logical equipment graph, independently
    # of the status error and sensor residuals.
    neighbors = defaultdict(set)
    for row in inventory["branches"]:
        neighbors[row["from_node"]].add(row["to_node"])
        neighbors[row["to_node"]].add(row["from_node"])
    for row in inventory["couplers"]:
        neighbors[row["node_a"]].add(row["node_b"])
        neighbors[row["node_b"]].add(row["node_a"])
    distances = {node: 0 for node in nodes}
    queue = list(nodes)
    for node in queue:
        for nxt in sorted(neighbors[node]):
            if nxt not in distances:
                distances[nxt] = distances[node]+1
                queue.append(nxt)
    choices = [(index, distances.get(record.get("node_id"), 10**6))
               for index, record in enumerate(sensors["records"])
               if record["kind"] == "Pinj" and record["available"]]
    choices = [(index, distance) for index, distance in choices if (distance == 0 if adjacent else distance >= 3)]
    if not choices:
        raise ValueError("Frozen sensor deployment cannot supply the requested overlay relationship")
    return sorted(choices, key=lambda item: (item[1] if adjacent else -item[1], item[0]))[0]


def composition_pairs(inventory, collection):
    """Choose nearby and separated devices from the frozen base-bus graph."""
    devices = inventory[collection]
    node_bus = {row["node_id"]: row["base_bus"] for row in inventory["nodes"]}
    graph = defaultdict(set)
    endpoints = {}
    for row in inventory["branches"]:
        a, b = node_bus[row["from_node"]], node_bus[row["to_node"]]
        graph[a].add(b)
        graph[b].add(a)
        endpoints[row["device_id"]] = {a, b}
    for row in inventory["couplers"]:
        endpoints[row["device_id"]] = {row["base_bus"]}
    distances = {}
    for origin in graph:
        known, queue = {origin: 0}, [origin]
        for bus in queue:
            for nxt in sorted(graph[bus]):
                if nxt not in known:
                    known[nxt] = known[bus]+1
                    queue.append(nxt)
        distances[origin] = known
    candidates = []
    for first, second in combinations(devices, 2):
        ids = [first["device_id"], second["device_id"]]
        distance = min(distances[a].get(b, 10**6) for a in endpoints[ids[0]] for b in endpoints[ids[1]])
        candidates.append((distance, ids))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1]))
    nearest, farthest = candidates[0], candidates[-1]
    result = [("nearby", nearest[1], nearest[0])]
    if nearest[1] != farthest[1]:
        result.append(("separated", farthest[1], farthest[0]))
    return result


def build_corpus(output_dir, *, system="case57", load_scales=(.8, 1.0), seed=20260911, smoke=False,
                 include_compositions=True):
    """Build candidates and admit physical worlds before any WLS or teacher run."""
    load_scales = tuple(float(value) for value in load_scales)
    if not load_scales or any(not np.isfinite(v) or v <= 0 for v in load_scales) or len(set(load_scales)) != len(load_scales):
        raise ValueError("Load scales must be distinct positive finite operating conditions")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = resolve_system(system)
    layouts = {"branch_status": build_inventory(system, split_buses=[], layout_id="logical_branch_status_v1"),
               "bus_sections": build_inventory(system, layout_id="logical_two_section_v1")}
    sensors = {key: {profile: build_measurement_inventory(inventory, profile)
                     for profile in ("direct", "indirect_even", "indirect_odd", "voltage_only")}
               for key, inventory in layouts.items()}
    # These files are emitted before operating points and error directions.
    for layout, inventory in layouts.items():
        write_json(out/"inventories"/f"{layout}.json", inventory)
        for profile, deployment in sensors[layout].items():
            write_json(out/"sensors"/f"{layout}_{profile}.json", deployment)
    config = {"contract": "logical_topology_physical_corpus_v1", "system": spec.to_manifest(),
              "seed": seed, "load_scales": list(load_scales), "smoke": smoke,
              "operating_policy": "same AC OPF followed by PF and bounds checks for each connected true topology",
              "physical_admission_uses_wls_or_teacher": False,
              "masked_branch_selection": "frozen even/odd deployments precede errors; indirect errors selected from their masked assets",
              "structural_holdout_device_ids": {key: fixed_holdout_devices(inv) for key, inv in layouts.items()},
              "root_scope": "engineering_validation; generated split views require independent family/asset-coverage review before training"}
    config["physical_root_fingerprint_contract"] = "noise_invariant_true_world_and_reported_error_v1; parent_physical_root groups all reported-error and sensor variants of one true operating world"
    write_json(out/"config.json", config)
    rows, worlds = [], {}
    observation_cache = {}

    def add(layout, load_index, actual, reported, *, family, direction=None, error_devices=(),
            overlay=None, parameter_row=None, profiles=("direct",), hypothesis_cardinality=1):
        inv = layouts[layout]
        world_id = digest([spec.base_case_hash, inv["layout_hash"], float(load_scales[load_index]), actual])
        if world_id not in worlds:
            source = spec.load_case()
            source["bus"][:, 2:4] *= float(load_scales[load_index])
            physical = solve_true_world(source, inv, actual)
            worlds[world_id] = physical
            write_json(out/"physical_audit"/f"{world_id}.json", {**physical, "true_statuses": actual,
                                                                "load_scale": load_scales[load_index]})
        physical = worlds[world_id]
        for profile in profiles:
            deployment = sensors[layout][profile]
            scenario_id = digest([world_id, profile, reported, family, overlay, parameter_row, int(seed)])
            row = {"scenario_id": scenario_id, "parent_physical_root": world_id, "family": family,
                   "direction": direction, "error_device_ids": list(error_devices), "layout": layout,
                   "measurement_profile": profile, "load_scale": float(load_scales[load_index]),
                   "physical_admission": {key: value for key, value in physical.items() if key not in ("operating_case", "solution")},
                   "true_statuses": deepcopy(actual), "model_statuses": deepcopy(reported),
                   "hypothesis_cardinality": hypothesis_cardinality,
                   "physical_audit_path": f"physical_audit/{world_id}.json"}
            if physical["admitted"]:
                cache_key = (world_id, profile)
                if cache_key not in observation_cache:
                    truth = expected_measurements(physical["operating_case"], inv, actual, physical["solution"], deployment)
                    # Measurements are drawn once per physical world/deployment,
                    # before any reported status or overlay is passed to WLS.
                    # Common sensor draws isolate availability effects when
                    # comparing the predeclared deployment profiles.
                    noise_seed = int(digest([seed, world_id])[:16], 16)
                    observation_cache[cache_key] = sample_measurements(truth, deployment, seed=noise_seed)
                observed = deepcopy(observation_cache[cache_key])
                current_case = deepcopy(physical["operating_case"])
                for branch in inv["branches"]:
                    value = reported[branch["device_id"]]
                    # Unknown is represented only in the logical map, not as a
                    # made-up fractional branch-admittance multiplier.
                    if value is not None:
                        current_case["branch"][branch["row0"], 10] = value
                if overlay:
                    index, distance = overlay_index(inv, deployment, error_devices[0], adjacent=overlay == "nearby")
                    bias = 10 * float(deployment["records"][index]["sigma"])
                    observed["values"][index] += bias
                    row["measurement_error"] = {"index0": index, "sensor_id": deployment["records"][index]["sensor_id"],
                                                "bias_pu": bias, "sigma_multiple": 10, "graph_distance": distance,
                                                "relationship": overlay}
                if parameter_row is not None:
                    true_on = actual[inv["branches"][parameter_row]["device_id"]] == 1
                    row["parameter_error"] = {"branch_row0": parameter_row, "factor": 2.,
                        "physically_identifiable_label": true_on,
                        "true_r_x": current_case["branch"][parameter_row, 2:4].tolist()}
                    current_case["branch"][parameter_row, 2:4] *= 2.
                evidence_id, model_id = digest(observed), digest(current_case)
                write_json(out/"observations"/f"{evidence_id}.json", observed)
                write_json(out/"model_inputs"/f"{model_id}.json", current_case)
                row["execution"] = {"inventory_path": f"inventories/{layout}.json",
                    "measurement_inventory_path": f"sensors/{layout}_{profile}.json",
                    "observations_path": f"observations/{evidence_id}.json", "base_case_path": f"model_inputs/{model_id}.json",
                    "current_statuses": deepcopy(reported)}
                row["observations_hash"] = evidence_id
                row["physical_root_fingerprint"] = digest([world_id, actual, reported, family, overlay, parameter_row])
            rows.append(row)

    for load_index, load_scale in enumerate(load_scales):
        if not np.isfinite(load_scale) or load_scale <= 0:
            raise ValueError("Positive finite load scales required")
        for layout, inv in layouts.items():
            normal = inv["normal_statuses"]
            devices = inv["branches"] if layout == "branch_status" else inv["couplers"]
            if smoke:
                devices = [devices[index] for index in sorted({0, len(devices)//2, len(devices)-1})] if devices else []
            add(layout, load_index, normal, normal, family="healthy_closed")
            for device in devices:
                device_id = device["device_id"]
                opened = {**normal, device_id: 0}
                device_profiles = ["direct"]
                if layout == "branch_status":
                    device_profiles.append("indirect_even" if device["row0"] % 2 == 0 else "indirect_odd")
                else:
                    device_profiles += ["indirect_even", "indirect_odd"]
                add(layout, load_index, normal, opened, family="exclusion" if layout == "branch_status" else "split",
                    direction="true_closed_model_open", error_devices=[device_id], profiles=device_profiles)
                add(layout, load_index, opened, normal, family="inclusion" if layout == "branch_status" else "merging",
                    direction="true_open_model_closed", error_devices=[device_id], profiles=device_profiles)
                add(layout, load_index, opened, opened, family="healthy_outage" if layout == "branch_status" else "healthy_open_coupler",
                    profiles=device_profiles)
            if not include_compositions or not devices:
                continue
            selected = devices[:1] if smoke else [devices[index] for index in sorted({0, len(devices)//2, len(devices)-1})]
            for device in selected:
                device_id = device["device_id"]
                wrong = {**normal, device_id: 0}
                for relationship in ("nearby", "distant"):
                    add(layout, load_index, normal, wrong, family=f"topology+{relationship}_measurement",
                        direction="true_closed_model_open", error_devices=[device_id], overlay=relationship)
                active_parameter = next(row["row0"] for row in inv["branches"]
                                        if row["device_id"] != device_id and row.get("row0") != 0)
                add(layout, load_index, normal, wrong, family="topology+parameter",
                    direction="true_closed_model_open", error_devices=[device_id], parameter_row=active_parameter)
                add(layout, load_index, normal, {**normal, device_id: None}, family="unknown_status",
                    error_devices=[device_id])
                add(layout, load_index, normal, wrong, family="sparse_unobservable",
                    error_devices=[device_id], profiles=("voltage_only",))
            for relationship, pair, distance in composition_pairs(inv, "branches" if layout == "branch_status" else "couplers"):
                family = "two_branch_errors" if layout == "branch_status" else "two_coupler_errors"
                add(layout, load_index, normal, {**normal, **{device: 0 for device in pair}},
                    family=family, direction=f"two_exclusions_{relationship}" if layout == "branch_status" else f"two_splits_{relationship}",
                    error_devices=pair, hypothesis_cardinality=2)
                rows[-1]["composition_graph_distance"] = distance
                # A second composition has both inclusion and exclusion (or
                # merging and split), rather than only restoring closed status.
                actual, reported = {**normal, pair[0]: 0}, {**normal, pair[1]: 0}
                add(layout, load_index, actual, reported, family=family,
                    direction=f"opposite_error_directions_{relationship}", error_devices=pair, hypothesis_cardinality=2)
                rows[-1]["composition_graph_distance"] = distance
            if layout == "bus_sections":
                pair = [inv["branches"][0]["device_id"], devices[0]["device_id"]]
                add(layout, load_index, normal, {**normal, **{device: 0 for device in pair}},
                    family="branch+coupler", error_devices=pair, hypothesis_cardinality=2)
        print(f"Generated true-topology worlds and reported errors for {system}, load={load_scale:g}", flush=True)
    default_splits = assign_root_splits(rows)
    heldout = set(item for ids in config["structural_holdout_device_ids"].values() for item in ids)
    structural_splits = assign_root_splits(rows, heldout_device_ids=heldout)
    for row in rows:
        row["split"] = default_splits[row["scenario_id"]]
        row["structural_split"] = structural_splits[row["scenario_id"]]
    manifest = {"config": config, "rows": rows, "physical_world_count": len(worlds),
                "physical_worlds_admitted": sum(world["admitted"] for world in worlds.values()),
                "scenario_count": len(rows), "scenario_count_physically_admitted": sum(row["physical_admission"]["admitted"] for row in rows),
                "all_candidates_retained_in_manifest": True,
                "split_views_are_not_a_training_readiness_certificate": True}
    write_json(out/"manifest.json", manifest)
    return manifest
