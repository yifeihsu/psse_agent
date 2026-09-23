"""Independent integrity, outcome, and physical-replay audit of logical topology runs.

Truth is used only in this offline checker. It never changes observations or
reruns operating optimization to accommodate a reported topology correction.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import numbers
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from scipy.stats import chi2, norm

REPO = Path(__file__).resolve().parents[1]
NUMERICAL_SOURCE_NAMES = ("logical_topology/estimation.py", "logical_topology/inventory.py", "logical_topology/measurements.py",
                          "psse_env/systems/registry.py", "mcp_server/case14.m", "mcp_server/case57.m")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def content_hash(value):
    def native(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(k): native(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [native(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(native(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def numerical_canonical(item):
    """Reproduce the declared exact float64 cache-input encoding independently."""
    if isinstance(item, dict):
        return {str(key): numerical_canonical(child) for key, child in item.items()}
    if isinstance(item, (list, tuple, np.ndarray)):
        array = np.asarray(item)
        if array.dtype.kind in "fiu":
            array = np.array(array, dtype="<f8", order="C", copy=True)
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite numerical reuse input")
            array[array == 0] = 0.
            return {"numeric_shape": list(array.shape), "float64_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        return [numerical_canonical(child) for child in item]
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, numbers.Real):
        return float(item)
    return item.item() if isinstance(item, np.generic) else item


def numerical_semantic_hash(value):
    return content_hash(numerical_canonical(value))


class FixedNumericalInputCache:
    """Cache canonical fixed inputs; validate their numerical identity per row.

This caches encoding work, not fit results or audit decisions. Original and
copied evidence use separate instances; dynamic cases/statuses/settings are
encoded for every fit. The end-of-row check detects fixed-input mutation.
"""
    def __init__(self, *, inventory, observations, sensors, numerical_source_sha256):
        self._fixed = {"inventory": inventory, "observations": observations, "sensors": sensors,
                       "numerical_source_sha256": numerical_source_sha256}
        self._canonical = numerical_canonical(self._fixed)
        self._identity = content_hash(self._canonical)

    def hash(self, *, case, statuses, parameters):
        payload = {**self._canonical, "case": numerical_canonical(case),
                   "statuses": numerical_canonical(statuses), "parameters": numerical_canonical(parameters)}
        # Canonical fields contain JSON-native values, so another recursive
        # native conversion would repeat fixed-input work without changing bytes.
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def assert_fixed_unchanged(self):
        if numerical_semantic_hash(self._fixed) != self._identity:
            raise ValueError("Fixed numerical audit inputs changed while using their canonical cache")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def under(root, relative):
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError(f"Artifact escapes the run directory: {relative}")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def case_arrays(case):
    result = deepcopy(case)
    for key in ("bus", "gen", "branch", "gencost"):
        result[key] = np.asarray(result[key], dtype=float)
    return result


def validate_observation_boundary(observations, sensors):
    if set(observations) != {"contract", "values", "sensor_ids", "sensor_inventory_hash"}:
        raise ValueError("Unexpected observation fields, including possible truth metadata")
    if (observations["sensor_inventory_hash"] != sensors["sensor_inventory_hash"]
            or observations["sensor_ids"] != [r["sensor_id"] for r in sensors["records"]]
            or len(observations["values"]) != len(sensors["records"])):
        raise ValueError("Observation identities differ from the fixed physical sensor inventory")
    mask = sensors["available_mask"]
    if mask != [row["available"] for row in sensors["records"]]:
        raise ValueError("Sensor availability declarations disagree")
    for available, value in zip(mask, observations["values"]):
        if available:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Available measurements must be finite numbers")
        elif value is not None:
            raise ValueError("Unavailable physical measurements must be redacted to None")


def validate_sensor_deployment(sensors, inventory):
    profile = sensors["profile"]
    if profile not in {"direct", "indirect_even", "indirect_odd", "voltage_only"}:
        raise ValueError("Unsupported fixed sensor profile")
    expected = []
    for kind in ("Vm", "Pinj", "Qinj"):
        for node in inventory["nodes"]:
            expected.append((f"{kind}:{node['node_id']}", kind, node["node_id"], None))
    for kind in ("Pf", "Qf", "Pt", "Qt"):
        for branch in inventory["branches"]:
            expected.append((f"{kind}:{branch['asset_id']}", kind, None, branch["row0"]))
    if len(sensors["records"]) != len(expected):
        raise ValueError("Sensor count differs from the frozen physical inventory")
    for record, (identity, kind, node, branch) in zip(sensors["records"], expected):
        if (record["sensor_id"] != identity or record["kind"] != kind
                or record.get("node_id") != node or record.get("branch_row0") != branch
                or record["sigma"] != (.001 if kind == "Vm" else .01)):
            raise ValueError("Sensor identity, placement, ordering, or declared noise changed")
        available = kind == "Vm" if profile == "voltage_only" else not (
            branch is not None and profile.startswith("indirect_")
            and branch % 2 == (0 if profile == "indirect_even" else 1))
        if record["available"] is not available:
            raise ValueError("Availability is not the predeclared whole-deployment mask")
    covariance = np.asarray(sensors["covariance"], dtype=float)
    variances = np.array([record["sigma"]**2 for record in sensors["records"]])
    if (covariance.shape != (len(expected), len(expected)) or not np.isfinite(covariance).all()
            or not np.array_equal(np.diag(covariance), variances) or not np.array_equal(covariance, covariance.T)):
        raise ValueError("Sensor covariance is inconsistent with declared physical channels")


def overlay_distance(inventory, device_id, sensor_node):
    branch = next((r for r in inventory["branches"] if r["device_id"] == device_id), None)
    coupler = next((r for r in inventory["couplers"] if r["device_id"] == device_id), None)
    starts = [branch["from_node"], branch["to_node"]] if branch else [coupler["node_a"], coupler["node_b"]]
    edges = defaultdict(set)
    for first, second in ([r["from_node"], r["to_node"]] for r in inventory["branches"]):
        edges[first].add(second)
        edges[second].add(first)
    for coupler in inventory["couplers"]:
        first, second = coupler["node_a"], coupler["node_b"]
        edges[first].add(second)
        edges[second].add(first)
    distance, queue = {node: 0 for node in starts}, list(starts)
    for node in queue:
        for neighbor in edges[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance.get(sensor_node)


def status_metrics(initial, final, truth):
    if set(initial) != set(final) or set(final) != set(truth):
        raise ValueError("Status vectors have inconsistent logical devices")
    changed = {key: final[key] for key in truth if initial[key] != final[key]}
    false = [key for key, value in changed.items() if value != truth[key]]
    healthy = [key for key in truth if initial[key] == truth[key]]
    correct = sum(final[key] == value for key, value in truth.items())
    return {"initial_error_count": sum(value is not None and value != truth[key] for key, value in initial.items()),
            "initial_unknown_count": sum(value is None for value in initial.values()),
            "final_error_count": sum(value is not None and value != truth[key] for key, value in final.items()),
            "final_unknown_count": sum(value is None for value in final.values()),
            "correct_count": correct, "device_count": len(truth), "accuracy": correct/len(truth),
            "exact_status_recovery": correct == len(truth), "false_correction_count": len(false),
            "false_correction_device_ids": false, "healthy_cb_count": len(healthy),
            "healthy_cb_preserved": all(final[key] == initial[key] for key in healthy), "changed_statuses": changed}


def compact_summary(rows):
    admitted = [r for r in rows if r.get("physical_admitted")]
    fits = [r.get("initial_estimation") or {} for r in admitted]
    return {"planned": len(rows), "physically_admitted": len(admitted),
        "audit_execution_failures": sum("audit_execution_failure" in r for r in rows),
        "initial_wls_converged": sum(f.get("converged") is True for f in fits),
        "initial_chi_square_alarms": sum(f.get("chi_square_alarm") is True for f in fits),
        "initial_normalized_residual_alarms": sum(f.get("normalized_residual_alarm") is True for f in fits),
        "initial_normalized_residual_only_alarms": sum(f.get("normalized_residual_alarm") is True and f.get("chi_square_alarm") is False for f in fits),
        "runtime_decisions": dict(Counter(r.get("runtime_decision", "not_executed") for r in admitted)),
        "corrections_applied": sum(bool(r.get("correction_applied")) for r in admitted),
        "exact_status_recovery": sum((r.get("status_audit") or {}).get("exact_status_recovery") is True for r in admitted),
        "false_corrections": sum((r.get("status_audit") or {}).get("false_correction_count", 0) for r in admitted),
        "healthy_cb_preservation_failures": sum((r.get("status_audit") or {}).get("healthy_cb_preserved") is False for r in admitted),
        "fixed_evidence_or_parameter_preservation_failures": sum((r.get("preservation") or {}).get("passed") is False for r in admitted)}


def _candidate_fit_errors(fit, observations, sensors, numeric_cache=None):
    """Recompute recorded residual objectives and rank-aware alarm arithmetic."""
    if not fit or "raw_residuals" not in fit:
        return []
    errors = []
    key = sensors["sensor_inventory_hash"]
    numeric_cache = {} if numeric_cache is None else numeric_cache
    if key not in numeric_cache:
        mask = np.asarray(sensors["available_mask"], dtype=bool)
        covariance = np.asarray(sensors["covariance"])[np.ix_(mask, mask)]
        diagonal = np.diag(covariance)
        numeric_cache[key] = (mask, covariance, diagonal, np.array_equal(covariance, np.diag(diagonal)))
    mask, covariance, diagonal, is_diagonal = numeric_cache[key]
    prediction = np.asarray(fit["predicted_values"], dtype=float)
    residual = fit["raw_residuals"]
    normalized = fit["normalized_residuals"]
    if len(residual) != len(mask) or len(normalized) != len(mask) or prediction.shape != mask.shape:
        return ["residual_dimensions"]
    if any(value is not None for values in (residual, normalized) for value, available in zip(values, mask) if not available):
        errors.append("unavailable_residual_exposure")
    available = np.flatnonzero(mask)
    raw = np.asarray([residual[i] for i in available], dtype=float)
    z = np.asarray([observations["values"][i] for i in available], dtype=float)
    if not np.allclose(raw, z-prediction[mask], rtol=0, atol=1e-10):
        errors.append("residual_not_bound_to_observations")
    objective = float(np.sum(raw**2/diagonal)) if is_diagonal else float(raw @ np.linalg.solve(covariance, raw))
    if not math.isclose(objective, fit["wls_objective"], rel_tol=1e-8, abs_tol=1e-8):
        errors.append("weighted_objective")
    maximum = max(float(normalized[i]) for i in available)
    if not math.isclose(maximum, fit["max_normalized_residual"], rel_tol=1e-10, abs_tol=1e-10):
        errors.append("maximum_normalized_residual")
    dof = int(mask.sum()) - fit["rank"]
    if fit["chi_square_dof"] != dof or fit["available_measurement_count"] != int(mask.sum()):
        errors.append("rank_degrees_of_freedom")
    singular = np.asarray(fit.get("weighted_jacobian_singular_values", []), dtype=float)
    if singular.size:
        cutoff = max(max(int(mask.sum()), fit["state_dimension"])*np.finfo(float).eps*singular[0], singular[0]*1e-9)
        if (not np.isfinite(singular).all() or np.any(singular < 0)
                or int(np.count_nonzero(singular > cutoff)) != fit["rank"]
                or fit["observable"] is not (fit["rank"] == fit["state_dimension"])):
            errors.append("jacobian_rank_or_observability")
    if dof > 0:
        threshold = float(chi2.ppf(1-fit["chi_square_alpha"], dof))
        if not math.isclose(threshold, fit["chi_square_threshold"], rel_tol=1e-12):
            errors.append("chi_square_threshold")
        if fit["chi_square_alarm"] is not (fit["wls_objective"] >= threshold):
            errors.append("chi_square_alarm")
    if fit["normalized_residual_alarm"] is not (maximum >= fit["normalized_residual_threshold"]):
        errors.append("normalized_residual_alarm")
    return errors


def comparison_certificate_errors(scan, *, require_guard=False, expected_alpha=.05, expected_guard_contract=None):
    """Bind a guarded scan to its complete application certificate."""
    guard, certificate = scan.get("comparison_guard"), scan.get("certificate")
    if not guard:
        return ["required_comparison_guard_missing"] if require_guard else []
    errors = []
    if expected_guard_contract is not None and guard.get("contract") != expected_guard_contract:
        errors.append("comparison_guard_differs_from_archived_contract")
    if guard.get("familywise_alpha") != expected_alpha:
        errors.append("comparison_alpha_differs_from_declared_configuration")
    if certificate is not None:
        if (guard.get("allowed") is not True or certificate.get("comparison_guard") != guard
                or certificate.get("comparison_guard_hash") != content_hash(guard)):
            errors.append("comparison_certificate_guard_binding")
        if (certificate.get("candidate_id") != scan.get("unique_candidate_id")
                or certificate.get("candidate_id") != guard.get("candidate_id")
                or certificate.get("hypothesis_scope") != scan.get("hypothesis_scope")
                or certificate.get("plausible_candidate_count") != len(scan.get("plausible_candidates", []))
                or certificate.get("unresolved_candidate_count") != scan.get("unresolved_candidate_count")):
            errors.append("comparison_certificate_scope_binding")
    return errors


def comparison_guard_errors(guard, scan, inventory, sensors, observations):
    """Independently check family/method budgets, structural df, and tails."""
    if not guard:
        return []  # Prototype runs predate this additional decision guard.
    errors = []
    contract = guard.get("contract")
    if contract not in {"logical_topology_pairwise_separation_guard_v1", "logical_topology_pairwise_separation_guard_v2"}:
        return ["comparison_guard_contract"]
    method_count = 2 if contract.endswith("_v2") else 1
    def same_number(value, expected):
        return (isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)
                and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15))
    if method_count == 2 and (guard.get("method_count") != 2
            or guard.get("method_budget_allocation") != "equal_bonferroni_split"
            or not same_number(guard.get("method_family_alpha"), guard["familywise_alpha"]/2)):
        errors.append("comparison_method_budget")
    if guard.get("exact_nonlinear_false_positive_guarantee") is not False:
        errors.append("unsupported_exact_nonlinear_guarantee")
    if "rival_count" not in guard:
        return errors + (["allowed_guard_without_comparisons"] if guard.get("allowed") else [])
    candidates = {c["candidate_id"]: c for c in scan["candidates"]}
    current = scan.get("current")
    if current and all(value is not None for value in current["statuses"].values()):
        candidates.setdefault(current["candidate_id"], current)
    chosen = candidates.get(guard.get("candidate_id"))
    if chosen is None:
        return errors + ["comparison_selected_candidate_missing"]
    rivals = {key: value for key, value in candidates.items() if key != chosen["candidate_id"]}
    count, alpha = len(rivals), guard["familywise_alpha"]
    if (not count or not 0 < alpha < 1 or guard["rival_count"] != count
            or guard["candidate_family_size"] != len(candidates)
            or not math.isclose(guard["pairwise_alpha"], alpha/count, rel_tol=1e-12)
            or guard["fixed_evidence_hash"] != scan["fixed_evidence_hash"]
            or guard["parent_model_hash"] != scan["parent_model_hash"]):
        return errors + ["comparison_family_budget_or_binding"]
    if method_count == 2 and not same_number(guard.get("pairwise_method_alpha"), alpha/(2*count)):
        errors.append("comparison_pairwise_method_budget")
    selected_j = chosen["estimation"]["wls_objective"]
    if not math.isclose(selected_j, guard["selected_objective"], rel_tol=1e-12, abs_tol=1e-12):
        errors.append("comparison_selected_objective")
    by_row = {row["row0"]: row for row in inventory["branches"]}
    branch_devices = {row["device_id"] for row in inventory["branches"]}
    coupler_devices = {row["device_id"] for row in inventory["couplers"]}
    comparisons = guard["comparisons"]
    if len(comparisons) != len(rivals) or {c["candidate_id"] for c in comparisons} != set(rivals):
        return errors + ["comparison_rival_coverage"]
    for comparison in comparisons:
        rival = rivals[comparison["candidate_id"]]
        method = comparison["method"]
        flow = comparison.get("zero_flow_evidence")
        flow_pass = False
        if flow is not None:
            witnesses = []
            for index, record in enumerate(sensors["records"]):
                if (sensors["available_mask"][index] and record["kind"] in {"Pf", "Qf", "Pt", "Qt"}
                        and rival["statuses"][by_row[record["branch_row0"]]["device_id"]] == 0):
                    score = abs(observations["values"][index])/math.sqrt(sensors["covariance"][index][index])
                    witnesses.append((score, index))
            if witnesses:
                score, index = max(witnesses, key=lambda value: value[0])
                log_bound = min(0., math.log(2) + float(norm.logsf(score)) + math.log(len(witnesses))
                                + math.log(count) + math.log(method_count))
                flow_pass = log_bound <= math.log(alpha)
                if (flow["available_zero_flow_row_count"] != len(witnesses) or flow["family_rival_count"] != count
                        or not math.isclose(flow["log_adjusted_p_upper_bound"], log_bound, rel_tol=1e-12, abs_tol=1e-12)
                        or not math.isclose(flow["multiplicity_adjusted_p_upper_bound"], math.exp(log_bound), rel_tol=1e-12, abs_tol=1e-300)
                        or flow["witness"]["sensor_id"] != sensors["records"][index]["sensor_id"]
                        or flow["passed"] is not flow_pass):
                    errors.append("comparison_gaussian_tail_or_multiplicity")
                if method_count == 2 and (flow.get("method_count") != 2 or not same_number(flow.get("alpha"), alpha)
                        or not same_number(flow.get("method_family_alpha"), alpha/2)
                        or not same_number(flow.get("pairwise_method_alpha"), alpha/(2*count))):
                    errors.append("comparison_gaussian_method_budget")
            elif flow.get("passed") or flow.get("available_zero_flow_row_count") != 0:
                errors.append("comparison_missing_zero_flow_witness")
        if method == "gaussian_zero_mean_flow_union_bound":
            if flow is None or not flow_pass or comparison["passed"] is not True:
                errors.append("comparison_invalid_gaussian_rejection")
        elif method == "asymptotic_common_relaxation_envelope":
            changed = {key for key in rival["statuses"] if rival["statuses"][key] != chosen["statuses"][key]}
            branches, couplers = sorted(changed & branch_devices), sorted(changed & coupler_devices)
            degrees = 4*len(branches) + 2*len(couplers)
            gain = rival["estimation"]["wls_objective"] - selected_j
            critical = float(chi2.isf(alpha/(method_count*count), degrees))
            probability = min(1., method_count*count*float(chi2.sf(max(gain, 0), degrees)))
            if (degrees <= 0 or comparison["df_upper_bound"] != degrees
                    or comparison["differing_branch_device_ids"] != branches or comparison["differing_coupler_device_ids"] != couplers
                    or not math.isclose(comparison["objective_gain"], gain, rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(comparison["critical_gain"], critical, rel_tol=1e-12)
                    or not math.isclose(comparison["multiplicity_adjusted_asymptotic_p_upper_bound"], probability, rel_tol=1e-12, abs_tol=1e-300)
                    or comparison["passed"] is not (gain >= critical)):
                errors.append("comparison_common_envelope_arithmetic")
        elif method == "declared_connected_scope_exclusion":
            nodes = {node["node_id"] for node in inventory["nodes"]}
            parent = {node: node for node in nodes}
            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node
            for collection, first, second in (("branches", "from_node", "to_node"), ("couplers", "node_a", "node_b")):
                for device in inventory[collection]:
                    if rival["statuses"][device["device_id"]]:
                        parent[find(device[first])] = find(device[second])
            excluded = (scan["hypothesis_scope"].get("connected_energized_models_only") is True
                        and rival.get("connectivity", {}).get("connected") is False
                        and len({find(node) for node in nodes}) > 1)
            if comparison["passed"] is not excluded:
                errors.append("comparison_connectivity_exclusion")
        elif comparison["passed"] is not False:
            errors.append("comparison_unresolved_alternative_passed")
    blocked = [c["candidate_id"] for c in comparisons if not c["passed"]]
    required = [c["candidate_id"] for c in comparisons if c["method"] == "full_fit_required"]
    if (guard["blocking_candidates"] != blocked or guard["requires_full_fit"] != required
            or guard["allowed"] is not (not blocked)):
        errors.append("comparison_guard_final_decision")
    return errors


def audit_run(output_dir, *, report_path=None):
    from logical_topology.inventory import validate_inventory, process_topology
    from logical_topology.measurements import expected_measurements
    from psse_env.systems import resolve_system

    root = Path(output_dir).resolve(strict=True)
    checker_bytes = Path(__file__).read_bytes()
    checker_sha = hashlib.sha256(checker_bytes).hexdigest()
    destination = Path(report_path).resolve() if report_path else root / "independent_artifact_audit.json"
    if destination.exists():
        raise FileExistsError(f"Audit receipt already exists: {destination}")
    receipt, run_config = read_json(root/"run_receipt.json"), read_json(root/"run_config.json")
    corpus = root/"corpus"
    manifest, summary = read_json(corpus/"manifest.json"), read_json(root/"summary.json")
    generation_config = manifest["config"]
    system = run_config.get("system", generation_config["system"]["case_id"])
    seed = run_config.get("seed", generation_config["seed"])
    preset = run_config.get("preset", "smoke" if generation_config["smoke"] else "full")
    load_scales = run_config.get("load_scales", generation_config["load_scales"])
    reports = read_json(root/"audit_results.json")
    failures, counts, cache = [], Counter(), {}
    numeric_cache, fixed_evidence_cache, sensor_input_hashes, inventory_input_hashes = {}, {}, {}, {}

    def check(condition, code, detail):
        if not condition:
            failures.append({"code": code, "detail": detail})

    def read(relative):
        path = under(corpus, relative)
        if path not in cache:
            cache[path] = read_json(path)
        return cache[path]

    def read_source(relative):
        path = under(source_run/"corpus", relative)
        if path not in cache:
            cache[path] = read_json(path)
        return cache[path]

    check(file_hash(corpus/"manifest.json") == receipt["manifest_sha256"], "manifest_sha256", "run receipt")
    check(file_hash(root/"summary.json") == receipt["summary_sha256"], "summary_sha256", "run receipt")
    check(receipt["source_before"] == receipt["source_after"] and receipt.get("all_sources_unchanged_during_run") is True,
          "source_changes_during_run", "all frozen implementation files")
    snapshot = {name: file_hash(under(root/"implementation_snapshot", name)) for name in receipt["source_before"]}
    current = {name: file_hash(under(REPO, name)) for name in receipt["source_before"]}
    check(snapshot == receipt["source_before"], "implementation_snapshot", "frozen sources")
    expected_guard_contract = None
    if "logical_topology/calibration.py" in receipt["source_before"]:
        tree = ast.parse(under(root/"implementation_snapshot", "logical_topology/calibration.py").read_text(encoding="utf-8"))
        contracts = {value.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                     for key, value in zip(node.keys, node.values)
                     if isinstance(key, ast.Constant) and key.value == "contract" and isinstance(value, ast.Constant)
                     and isinstance(value.value, str) and value.value.startswith("logical_topology_pairwise_separation_guard_v")}
        check(len(contracts) == 1, "archived_comparison_guard_contract", sorted(contracts))
        expected_guard_contract = next(iter(contracts)) if len(contracts) == 1 else None
    copy_manifest = None
    source_run = None
    source_receipt = None
    source_defaults, historical_default_budget_proven = {}, False
    if (root/"input_copy_manifest.json").exists():
        copy_manifest = read_json(root/"input_copy_manifest.json")
        source_run = Path(copy_manifest["source_run"]).resolve(strict=True)
        source_receipt = read_json(source_run/"run_receipt.json")
        check(source_receipt.get("all_rows_audited") is True
              and source_receipt.get("all_sources_unchanged_during_run") is True
              and source_receipt["source_before"] == source_receipt["source_after"],
              "reused_source_run_complete", str(source_run))
        check(file_hash(source_run/"corpus/manifest.json") == source_receipt["manifest_sha256"],
              "reused_source_manifest_sha256", str(source_run))
        check(all(file_hash(under(source_run/"implementation_snapshot", name)) == digest
                  for name, digest in source_receipt["source_before"].items()),
              "reused_source_implementation_snapshot", str(source_run))
        source_tree = ast.parse(under(source_run/"implementation_snapshot", "logical_topology/estimation.py").read_text(encoding="utf-8"))
        source_estimator = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == "estimate")
        source_defaults = {arg.arg: ast.literal_eval(default) for arg, default in zip(source_estimator.args.kwonlyargs, source_estimator.args.kw_defaults)}
        historical_default_budget_proven = True
        for source_name in ("logical_topology/runtime.py", "logical_topology/audit.py"):
            for node in ast.walk(ast.parse(under(source_run/"implementation_snapshot", source_name).read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call):
                    called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                    if called in {"estimate", "_estimator"} and any(k.arg not in {"chi2_alpha", "normalized_residual_threshold"} for k in node.keywords):
                        historical_default_budget_proven = False
        check(file_hash(root/"input_copy_manifest.json") == receipt["input_copy_manifest_sha256"], "input_copy_manifest_sha256", "verified run")
        for name, expected in copy_manifest["files_sha256"].items():
            check(file_hash(under(corpus, name)) == expected == file_hash(under(source_run/"corpus", name)), "immutable_input_copy", name)
        check(copy_manifest["source_bytes_preserved"] is True and copy_manifest["copied_bytes_match"] is True
              and receipt["source_corpus_unchanged"] is True and receipt["copied_corpus_unchanged"] is True,
              "input_copy_claims", "source and copied evidence")
    check(compact_summary(reports) == summary["overall"], "overall_summary", "independent compact totals")
    rows = manifest["rows"]
    ids, report_ids = [r["scenario_id"] for r in rows], [r["scenario_id"] for r in reports]
    check(len(ids) == len(set(ids)) == len(report_ids) == len(set(report_ids)) and set(ids) == set(report_ids), "row_coverage", "manifest vs audits")
    by_id = {r["scenario_id"]: r for r in reports}
    layouts = {name: read(f"inventories/{name}.json") for name in ("branch_status", "bus_sections")}
    for inventory in layouts.values():
        validate_inventory(inventory)
        inventory_input_hashes[inventory["layout_hash"]] = content_hash(inventory)
    for name, inventory in layouts.items():
        ordered = sorted(sorted(inventory["normal_statuses"]),
                         key=lambda device: content_hash([inventory["layout_hash"], "structural_holdout_v1", device]))
        expected_heldout = ordered[:max(1, len(ordered)//5)]
        check(manifest["config"]["structural_holdout_device_ids"][name] == expected_heldout,
              "predeclared_structural_holdout", name)
    spec = resolve_system(system)
    parent_splits, structural_splits = defaultdict(set), defaultdict(set)
    source_groups, coverage = defaultdict(set), Counter()
    clean_observations, overlay_rows, checked_deployments = {}, [], set()
    groups = defaultdict(list)
    heldout = {item for devices in manifest["config"]["structural_holdout_device_ids"].values() for item in devices}
    heldout_parents = {row["parent_physical_root"] for row in rows if heldout.intersection(row["error_device_ids"])}
    healthy_open = Counter()
    admitted_worlds = set()
    replay_rows = {}
    detailed_checked = candidate_checked = 0
    reuse_totals = Counter()
    guard_contract_counts = Counter()
    for ordinal, row in enumerate(rows):
        identity, family = row["scenario_id"], row["family"]
        report = by_id[identity]
        counts[family] += 1
        groups[f"{family}/{row['measurement_profile']}"].append(report)
        inventory = layouts[row["layout"]]
        actual, modeled = row["true_statuses"], row["model_statuses"]
        devices = {d["device_id"]: d for key in ("branches", "couplers") for d in inventory[key]}
        check(set(actual) == set(modeled) == set(devices), "status_roster", identity)
        errors = {key for key in actual if actual[key] != modeled[key]}
        check(errors == set(row["error_device_ids"]), "error_device_labels", identity)
        if family in {"inclusion", "exclusion", "split", "merging"}:
            expected_pair = (1, 0) if family in {"exclusion", "split"} else (0, 1)
            check(len(errors) == 1 and all((actual[d], modeled[d]) == expected_pair for d in errors), "error_direction", identity)
            coverage[(row["load_scale"], family, row["error_device_ids"][0], row["measurement_profile"])] += 1
        if family.startswith("healthy"):
            check(actual == modeled, "healthy_status_mismatch", identity)
            if family != "healthy_closed":
                check(any(value == 0 for value in actual.values()), "non_normal_healthy_control", identity)
                healthy_open[family] += 1
        world = content_hash([spec.base_case_hash, inventory["layout_hash"], float(row["load_scale"]), actual])
        check(world == row["parent_physical_root"], "parent_physical_root", identity)
        parent_splits[world].add(row["split"])
        structural_splits[world].add(row["structural_split"])
        fraction = int(content_hash(["root_split_v1", world])[:12], 16)/16**12
        expected_split = "train" if fraction < .7 else "development" if fraction < .85 else "test"
        check(row["split"] == expected_split, "declared_parent_split_rule", identity)
        check(row["structural_split"] == ("structural_test" if world in heldout_parents else expected_split), "declared_structural_split_rule", identity)
        if world in heldout_parents:
            check(row["structural_split"] == "structural_test", "heldout_parent_in_training", identity)
        physical = read(row["physical_audit_path"])
        check(physical["true_statuses"] == actual and physical["admitted"] is row["physical_admission"]["admitted"], "physical_world_binding", identity)
        check(report.get("physical_admitted") is physical["admitted"], "audit_admission_binding", identity)
        check(report == read(f"row_audits/{identity}.json"), "compact_row_binding", identity)
        check("audit_execution_failure" not in report, "audit_execution_failure", identity)
        if not physical["admitted"]:
            check("execution" not in row and report.get("runtime_decision") == "physical_reject", "physical_rejection_retention", identity)
            continue
        check(physical["physics"]["passed"] is True and all(c["passed"] for c in physical["physics"]["checks"].values()), "physical_admission_checks", identity)
        admitted_worlds.add(world)
        execution = row["execution"]
        check(set(execution) == {"inventory_path", "measurement_inventory_path", "observations_path", "base_case_path", "current_statuses"}, "execution_allowlist", identity)
        check(execution["current_statuses"] == modeled, "reported_execution_statuses", identity)
        sensors, observed, model_case = (read(execution[key]) for key in ("measurement_inventory_path", "observations_path", "base_case_path"))
        if sensors["sensor_inventory_hash"] not in checked_deployments:
            validate_sensor_deployment(sensors, inventory)
            checked_deployments.add(sensors["sensor_inventory_hash"])
            sensor_input_hashes[sensors["sensor_inventory_hash"]] = content_hash(sensors)
        validate_observation_boundary(observed, sensors)
        check(set(model_case) <= {"version", "baseMVA", "bus", "gen", "branch", "gencost", "success"}, "model_case_truth_metadata", identity)
        check(content_hash(observed) == row["observations_hash"] == Path(execution["observations_path"]).stem, "observations_content_hash", identity)
        check(content_hash(model_case) == Path(execution["base_case_path"]).stem, "model_content_hash", identity)
        check(content_hash({k:v for k,v in sensors.items() if k != "sensor_inventory_hash"}) == sensors["sensor_inventory_hash"], "sensor_inventory_hash", identity)
        check(sensors["layout_hash"] == inventory["layout_hash"], "sensor_layout_binding", identity)
        if family in {"inclusion", "exclusion"}:
            target = devices[row["error_device_ids"][0]]["row0"]
            meters = [m for m in sensors["records"] if m.get("branch_row0") == target]
            check(len(meters) == 4 and all(m["available"] is (row["measurement_profile"] == "direct") for m in meters), "target_flow_deployment", identity)
        initial_case = deepcopy(model_case)
        expected_case = deepcopy(physical["operating_case"])
        for branch in inventory["branches"]:
            value = modeled[branch["device_id"]]
            if value is not None:
                expected_case["branch"][branch["row0"]][10] = value
                initial_case["branch"][branch["row0"]][10] = value
        if row.get("parameter_error"):
            parameter = row["parameter_error"]
            target = inventory["branches"][parameter["branch_row0"]]
            check(actual[target["device_id"]] == 1 and parameter["physically_identifiable_label"] is True, "inactive_parameter_label", identity)
            check(expected_case["branch"][parameter["branch_row0"]][2:4] == parameter["true_r_x"], "parameter_truth_values", identity)
            expected_case["branch"][parameter["branch_row0"]][2:4] = [v * parameter["factor"] for v in parameter["true_r_x"]]
        # Numeric equality ignores JSON's int/float encoding of the authoritative status column.
        for key in ("bus", "gen", "branch", "gencost"):
            check(np.array_equal(np.asarray(model_case[key]), np.asarray(expected_case[key])), "preserved_model_parameters", f"{identity}/{key}")
        if not row.get("measurement_error"):
            source_groups[(world, row["measurement_profile"])].add(row["observations_hash"])
            clean_observations[(world, row["measurement_profile"])] = observed
        else:
            overlay_rows.append((row, observed, sensors, inventory))
        overlay = (row.get("measurement_error") or {}).get("relationship")
        parameter_row = (row.get("parameter_error") or {}).get("branch_row0")
        check(row["physical_root_fingerprint"] == content_hash([world, actual, modeled, family, overlay, parameter_row]), "topology_error_root_identity", identity)
        check(identity == content_hash([world, row["measurement_profile"], modeled, family, overlay, parameter_row, int(seed)]), "seeded_scenario_identity", identity)
        if report.get("audit_execution_failure"):
            continue
        details_path = under(corpus, report["detailed_audit_path"])
        check(file_hash(details_path) == report["detailed_audit_sha256"], "detailed_audit_sha256", identity)
        with gzip.open(details_path, "rt", encoding="utf-8") as stream:
            detailed = json.load(stream)
        detailed_checked += 1
        scan = detailed["runtime_scan"]
        guard_contract_counts[(scan.get("comparison_guard") or {}).get("contract", "no_comparison_guard")] += 1
        for error in comparison_certificate_errors(scan,
                require_guard=run_config.get("contract") == "fixed_corpus_comparison_guard_revalidation_v1",
                expected_guard_contract=expected_guard_contract):
            check(False, error, identity)
        for error in comparison_guard_errors(scan.get("comparison_guard"), scan, inventory, sensors, observed):
            check(False, error, identity)
        fixed_key = (sensors["sensor_inventory_hash"], row["observations_hash"])
        if fixed_key not in fixed_evidence_cache:
            fixed_evidence_cache[fixed_key] = content_hash({"measurement_inventory": sensors, "observations": observed})
        evidence = fixed_evidence_cache[fixed_key]
        check(report["execution_input_hashes"] == {
            "inventory": inventory_input_hashes[inventory["layout_hash"]],
            "measurement_inventory": sensor_input_hashes[sensors["sensor_inventory_hash"]],
            "observations": row["observations_hash"], "current_case": content_hash(model_case)},
            "compact_execution_input_hashes", identity)
        parent = content_hash({"inventory": inventory, "current_case": initial_case, "current_statuses": modeled,
                               "configuration": {"chi2_alpha": .05, "normalized_residual_threshold": 4., "connected_only": True}})
        check(scan["fixed_evidence_hash"] == evidence and scan["parent_model_hash"] == parent, "runtime_input_binding", identity)
        candidates = {candidate["candidate_id"]: candidate for candidate in [scan["current"], *scan["candidates"]]}
        reuse_receipt = None
        old_payload = None
        new_fixed_inputs = old_fixed_inputs = None
        old_model_case = old_physical_case = None
        if report.get("numeric_fit_provenance"):
            provenance = report["numeric_fit_provenance"]
            path = under(root, provenance["path"])
            check(file_hash(path) == provenance["sha256"], "numeric_fit_provenance_sha256", identity)
            reuse_receipt = read_json(path)
            new_fixed_inputs = FixedNumericalInputCache(inventory=inventory, observations=observed, sensors=sensors,
                                                       numerical_source_sha256=reuse_receipt["numerical_sources"])
            check(reuse_receipt["policy_observable"] is False and reuse_receipt["old_candidate_decisions_or_certificates_reused"] is False
                  and provenance["old_decisions_or_certificates_reused"] is False, "numeric_reuse_boundary", identity)
            log = reuse_receipt["lookups_provenance"]
            expected_counts = {"lookups": len(log), "reused_fits": sum(v["kind"] == "reused_verified_numeric_fit" for v in log),
                               "fresh_estimator_calls": sum(v["kind"] == "fresh_estimator_call" for v in log),
                               "fresh_numerical_solves": sum(v.get("fresh_solve") is True for v in log if v["kind"] == "fresh_estimator_call")}
            check(all(reuse_receipt[key] == value == provenance[key] for key, value in expected_counts.items()), "numeric_reuse_counts", identity)
            check(not expected_counts["reused_fits"] or (reuse_receipt["numerical_sources_match"] is True
                  and all(receipt["source_before"][name] == digest for name, digest in reuse_receipt["numerical_sources"].items())),
                  "reused_numerical_source_versions", identity)
            reuse_totals.update(expected_counts)
            check(source_run is not None and Path(reuse_receipt["source_run"]).resolve() == source_run, "numeric_reuse_source_run", identity)
            if source_receipt is not None:
                historical_versions = source_receipt.get("numerical_versions")
                current_versions = receipt["numerical_versions"]
                version_attestation = ("not_recorded_in_source_run" if historical_versions is None else
                                       "matches_current" if historical_versions == current_versions else "differs_from_current")
                check(reuse_receipt["historical_numerical_versions"] == historical_versions
                      and reuse_receipt["current_numerical_versions"] == current_versions
                      and reuse_receipt["historical_version_attestation"] == version_attestation
                      and receipt["cache_historical_version_attestation"] == version_attestation
                      and (not expected_counts["reused_fits"] or version_attestation != "differs_from_current"),
                      "reused_library_version_attestation", identity)
                check(reuse_receipt["numerical_sources"] == {name:source_receipt["source_before"][name] for name in NUMERICAL_SOURCE_NAMES},
                      "reused_numerical_sources_match_historical_receipt", identity)
            if expected_counts["reused_fits"]:
                source_compact = read_json(under(source_run/"corpus", f"row_audits/{identity}.json"))
                source_detail = under(source_run/"corpus", source_compact["detailed_audit_path"])
                source_sha = file_hash(source_detail)
                check(source_sha == source_compact["detailed_audit_sha256"], "reused_source_audit_sha256", identity)
                check(all(v["source_artifact_sha256"] == source_sha for v in log if v["kind"] == "reused_verified_numeric_fit"), "reused_fit_source_sha256", identity)
                check(source_compact["execution_input_hashes"] == report["execution_input_hashes"]
                      and all(v["source_scenario_id"] == identity for v in log if v["kind"] == "reused_verified_numeric_fit"),
                      "reused_fit_source_scenario_inputs", identity)
                with gzip.open(source_detail, "rt") as stream:
                    old_payload = json.load(stream)
                execution = row["execution"]
                old_fixed_inputs = FixedNumericalInputCache(
                    inventory=read_source(execution["inventory_path"]), observations=read_source(execution["observations_path"]),
                    sensors=read_source(execution["measurement_inventory_path"]),
                    numerical_source_sha256=reuse_receipt["numerical_sources"])
                old_model_case = read_source(execution["base_case_path"])
                old_physical_case = read_source(row["physical_audit_path"])["operating_case"]

        def check_numeric_reuse(fit, fit_case, statuses):
            if not reuse_receipt or not fit or not fit.get("numerical_fit_execution"):
                return
            execution = fit["numerical_fit_execution"]
            allowed = {"kind", "reused", "fresh_solve", "semantic_input_sha256", "estimation_source_sha256", "parameters"}
            check(set(execution) == allowed, "numeric_runtime_provenance_allowlist", identity)
            numerical = deepcopy(fit_case)
            for branch in inventory["branches"]:
                numerical["branch"][branch["row0"]][10] = int(statuses[branch["device_id"]])
            semantic = new_fixed_inputs.hash(case=numerical, statuses={key: int(value) for key, value in statuses.items()},
                                             parameters=execution["parameters"])
            check(execution["semantic_input_sha256"] == semantic, "exact_numerical_reuse_inputs", identity)
            check(execution["estimation_source_sha256"] == receipt["source_before"]["logical_topology/estimation.py"], "numeric_estimator_version", identity)
            events = [event for event in reuse_receipt["lookups_provenance"] if event["semantic_input_sha256"] == semantic and event["kind"] == execution["kind"]]
            check(bool(events), "numeric_reuse_event_missing", identity)
            if execution["reused"]:
                check(execution["fresh_solve"] is False and execution["kind"] == "reused_verified_numeric_fit", "numeric_reuse_execution_flags", identity)
                new_numbers = {key:value for key, value in fit.items() if key != "numerical_fit_execution"}
                matched = False
                for event in events:
                    for slot in event["source_fit_slots"]:
                        match = re.fullmatch(r"runtime_scan\.candidates\[(\d+)\]", slot)
                        if match:
                            old_candidate = old_payload["runtime_scan"]["candidates"][int(match.group(1))]
                            source_fit = old_candidate.get("estimation")
                            old_statuses, old_case = old_candidate["statuses"], deepcopy(old_model_case)
                            check(old_candidate["parent_model_hash"] == parent and old_candidate["fixed_evidence_hash"] == evidence
                                  and old_candidate["candidate_id"] == content_hash({"parent": parent, "statuses": old_statuses, "evidence": evidence}),
                                  "reused_original_candidate_binding", identity)
                        elif slot in {"offline_true_status_model_fit", "offline_true_physical_status_fit"}:
                            source_fit = old_payload.get(slot)
                            old_statuses = actual
                            old_case = deepcopy(old_physical_case if slot == "offline_true_physical_status_fit" else old_model_case)
                        else:
                            check(False, "reused_original_fit_slot", f"{identity}/{slot}")
                            continue
                        if not source_fit:
                            continue
                        old_parameters = (source_fit.get("numerical_fit_execution") or {}).get("parameters")
                        if old_parameters is None and historical_default_budget_proven:
                            old_parameters = {"chi2_alpha": source_fit["chi_square_alpha"],
                                              "normalized_residual_threshold": source_fit["normalized_residual_threshold"],
                                              "max_nfev": source_defaults["max_nfev"]}
                        for branch in inventory["branches"]:
                            old_case["branch"][branch["row0"]][10] = int(old_statuses[branch["device_id"]])
                        old_semantic = old_fixed_inputs.hash(case=old_case, statuses={key:int(value) for key,value in old_statuses.items()},
                                                            parameters=old_parameters)
                        if (old_parameters is not None and old_semantic == semantic
                                and content_hash(new_numbers) == content_hash({key:value for key,value in source_fit.items() if key != "numerical_fit_execution"})):
                            matched = True
                            break
                    if matched:
                        break
                check(matched, "reused_numerical_content_differs_from_source", identity)

        for candidate in candidates.values():
            candidate_checked += 1
            check(candidate["fixed_evidence_hash"] == evidence and candidate["parent_model_hash"] == parent, "candidate_evidence_binding", identity)
            check(candidate["candidate_id"] == content_hash({"parent": parent, "statuses": candidate["statuses"], "evidence": evidence}), "candidate_identity", identity)
            changes = {k:v for k,v in candidate["statuses"].items() if v != modeled[k]}
            check(changes == candidate["changes"], "cumulative_candidate_statuses", identity)
            fit_errors = _candidate_fit_errors(candidate.get("estimation"), observed, sensors, numeric_cache)
            for error in fit_errors:
                check(False, error, f"{identity}/{candidate['candidate_id']}")
            check_numeric_reuse(candidate.get("estimation"), model_case, candidate["statuses"])
            numeric_execution = (candidate.get("estimation") or {}).get("numerical_fit_execution")
            if numeric_execution:
                check(candidate.get("solver_executed") is numeric_execution["fresh_solve"]
                      and candidate.get("numerical_fit_reused") is numeric_execution["reused"]
                      and candidate.get("full_estimate_returned") is True,
                      "candidate_fresh_vs_reused_execution", identity)
            proof = candidate.get("analytical_proof")
            if proof:
                index = observed["sensor_ids"].index(proof["sensor_id"])
                check(sensors["available_mask"][index] and candidate["statuses"][proof["device_id"]] == 0,
                      "analytical_proof_status_availability", identity)
                check(sensors["records"][index]["kind"] in {"Pf", "Qf", "Pt", "Qt"}
                      and sensors["records"][index].get("branch_row0") == devices[proof["device_id"]].get("row0"),
                      "analytical_proof_physical_branch_binding", identity)
                lower = abs(observed["values"][index])/math.sqrt(sensors["covariance"][index][index])
                check(math.isclose(lower, proof["normalized_residual_lower_bound"], rel_tol=1e-12) and lower >= 4,
                      "analytical_proof_bound", identity)
        final = candidates[scan["unique_candidate_id"]]["statuses"] if report["correction_applied"] else modeled
        metrics = status_metrics(modeled, final, actual)
        check(metrics == report["status_audit"] == detailed["offline_scoring"]["status_audit"], "offline_status_outcome", identity)
        check(report["preservation"]["passed"] is True and all(report["preservation"].values()), "runtime_preservation", identity)
        if report["correction_applied"]:
            check(scan["scope_complete"] and scan["unresolved_candidate_count"] == 0 and len(scan["plausible_candidates"]) == 1
                  and scan["certificate"] is not None and scan["certificate"]["candidate_id"] == scan["unique_candidate_id"], "uncertified_application", identity)
            check(scan["certificate"]["parent_model_hash"] == parent and scan["certificate"]["fixed_evidence_hash"] == evidence
                  and scan["certificate"]["scope_complete"] is True, "application_certificate_binding", identity)
            if scan.get("comparison_guard") is not None:
                check(scan["comparison_guard"]["allowed"] is True and scan["certificate"].get("comparison_guard") == scan["comparison_guard"],
                      "application_comparison_guard", identity)
        else:
            check(not report["status_audit"]["changed_statuses"], "unapplied_status_mutation", identity)
        check(report["runtime_decision"] == scan["decision"], "compact_runtime_decision", identity)
        if scan.get("comparison_guard") is not None:
            guard = scan["comparison_guard"]
            compatible = set(guard["blocking_candidates"])
            if guard.get("candidate_id"):
                compatible.add(guard["candidate_id"])
            else:
                compatible.update(c["candidate_id"] for c in scan["plausible_candidates"])
            actual_compatible = scan.get("comparison_compatible_candidates", [])
            check({c["candidate_id"] for c in actual_compatible} == compatible and len(actual_compatible) == len(compatible),
                  "comparison_compatible_set", identity)
            identifiability = report["audit_outcomes"]["status_identifiability"]
            check(identifiability["comparison_compatible_candidate_count"] == len(compatible)
                  and identifiability["true_statuses_in_comparison_compatible_set"] is any(c["statuses"] == actual for c in actual_compatible),
                  "offline_comparison_set_accounting", identity)
        check(scan["tested_candidate_count"] == len(scan["candidates"])
              and scan["unresolved_candidate_count"] == sum(c["resolution"] == "unresolved" for c in scan["candidates"]),
              "candidate_search_accounting", identity)
        check(report.get("offline_observability_fit_used_for_runtime_decisions") is False, "offline_fit_runtime_separation", identity)
        check_numeric_reuse(detailed.get("offline_true_physical_status_fit"), physical["operating_case"], actual)
        check_numeric_reuse(detailed.get("offline_true_status_model_fit"), model_case, actual)
        for fixed_inputs in (new_fixed_inputs, old_fixed_inputs):
            if fixed_inputs is not None:
                fixed_inputs.assert_fixed_unchanged()
        if family in {"inclusion", "exclusion", "split", "merging"} and row["measurement_profile"] == "direct":
            replay_rows.setdefault((row["load_scale"], family), row)
        if (ordinal+1) % 100 == 0:
            print(f"Independent audit checked {ordinal+1}/{len(rows)} scenario rows", flush=True)
    check(all(len(v) == 1 for v in parent_splits.values()) and all(len(v) == 1 for v in structural_splits.values()), "split_parent_leakage", "ordinary and structural views")
    check(all(len(v) == 1 for v in source_groups.values()), "reported_model_regenerated_observations", "physical-world/profile groups")
    for row, observed, sensors, inventory in overlay_rows:
        identity, error = row["scenario_id"], row["measurement_error"]
        baseline = clean_observations[(row["parent_physical_root"], row["measurement_profile"])]
        changed = [i for i, (a, b) in enumerate(zip(observed["values"], baseline["values"])) if a != b]
        index = error["index0"]
        check(changed == [index], "measurement_overlay_preservation", identity)
        record = sensors["records"][index]
        check(record["available"] and record["kind"] == "Pinj" and record["sensor_id"] == error["sensor_id"], "measurement_overlay_sensor", identity)
        check(math.isclose(observed["values"][index]-baseline["values"][index], error["bias_pu"], rel_tol=1e-12, abs_tol=1e-12)
              and math.isclose(error["bias_pu"], 10*record["sigma"], rel_tol=1e-12), "measurement_overlay_magnitude", identity)
        distance = overlay_distance(inventory, row["error_device_ids"][0], record["node_id"])
        check(distance == error["graph_distance"] and (distance == 0 if error["relationship"] == "nearby" else distance >= 3), "measurement_overlay_relation", identity)
    for key, group in groups.items():
        check(compact_summary(group) == summary["groups"][key], "group_summary", key)
    if preset == "full":
        for load in load_scales:
            for branch in layouts["branch_status"]["branches"]:
                profile = "indirect_even" if branch["row0"] % 2 == 0 else "indirect_odd"
                for family in ("inclusion", "exclusion"):
                    for deployment in ("direct", profile):
                        check(coverage[(load, family, branch["device_id"], deployment)] == 1, "planned_all_asset_directions", str((load, family, branch["device_id"], deployment)))
            for coupler in layouts["bus_sections"]["couplers"]:
                for family in ("split", "merging"):
                    for profile in ("direct", "indirect_even", "indirect_odd"):
                        check(coverage[(load, family, coupler["device_id"], profile)] == 1, "planned_all_coupler_directions", str((load, family, coupler["device_id"], profile)))
    check(manifest["physical_world_count"] == len(parent_splits), "physical_world_count", len(parent_splits))
    check(manifest["physical_worlds_admitted"] == len(admitted_worlds), "admitted_physical_world_count", len(admitted_worlds))
    expected_rejections = dict(Counter(row["physical_admission"].get("reason") for row in rows if not row["physical_admission"]["admitted"]))
    check(summary["physical_rejection_reasons"] == expected_rejections, "physical_rejection_summary", "all planned rows")
    if source_run is not None:
        check(dict(reuse_totals) == summary["numerical_fit_provenance"] == receipt["numerical_fit_provenance"], "aggregate_numeric_reuse_counts", "verified totals")
    replays = []
    from pypower.api import ppoption, runpf
    from logical_topology.estimation import _MeasurementModel
    for (_, family), row in sorted(replay_rows.items()):
        inventory, physical = layouts[row["layout"]], read(row["physical_audit_path"])
        sensors = read(row["execution"]["measurement_inventory_path"])
        operating = case_arrays(physical["operating_case"])
        compiled = process_topology(operating, inventory, row["true_statuses"])
        solved, success = runpf(compiled["case"], ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=40))
        original = case_arrays(physical["solution"])
        truth_original = np.asarray(expected_measurements(operating, inventory, row["true_statuses"], original, sensors))
        truth_replayed = np.asarray(expected_measurements(operating, inventory, row["true_statuses"], solved, sensors))
        mask = np.asarray(sensors["available_mask"], dtype=bool)
        error = float(np.max(np.abs(truth_replayed[mask] - truth_original[mask])))
        entry = {"scenario_id": row["scenario_id"], "family": family, "load_scale": row["load_scale"],
                 "pf_converged": bool(success), "available_truth_measurement_max_error_pu": error,
                 "generator_pq_max_error_mva": float(np.max(np.abs(solved["gen"][:, 1:3]-original["gen"][:, 1:3]))),
                 "passed": bool(success and error < 1e-7), "gse_state_replays": []}
        report = by_id[row["scenario_id"]]
        with gzip.open(under(corpus, report["detailed_audit_path"]), "rt") as stream:
            detailed = json.load(stream)
        scan = detailed["runtime_scan"]
        selected = [scan["current"]]
        if scan["unique_candidate_id"]:
            selected.append(next(c for c in scan["candidates"] if c["candidate_id"] == scan["unique_candidate_id"]))
        for candidate in selected:
            fit = candidate.get("estimation") or {}
            if not fit.get("state"):
                continue
            candidate_case = case_arrays(read(row["execution"]["base_case_path"]))
            processed = process_topology(candidate_case, inventory, candidate["statuses"])
            model = _MeasurementModel(candidate_case, inventory, candidate["statuses"], processed, sensors)
            state = fit["state"]
            x = np.zeros(model.nstate)
            x[:model.nb-1] = np.asarray(state["electrical_voltage_angles_rad"])[model.angle_rows]
            for node, busrow in processed["node_to_row0"].items():
                x[model.vm_start + busrow] = state["node_voltage_magnitude_pu"][node]
            for index, coupler in enumerate(model.closed):
                flow = state["closed_coupler_flows_pu"][coupler["device_id"]]
                x[model.voltage_states + index] = flow["p"]
                x[model.voltage_states + len(model.closed) + index] = flow["q"]
            prediction, _ = model.evaluate(x)
            discrepancy = float(np.max(np.abs(prediction - fit["predicted_values"])))
            entry["gse_state_replays"].append({"candidate_id": candidate["candidate_id"], "predicted_measurement_max_error_pu": discrepancy,
                                               "passed": discrepancy < 1e-10, "original_wls_converged": fit["converged"]})
            entry["passed"] &= discrepancy < 1e-10
        check(entry["passed"], "physical_or_gse_replay", row["scenario_id"])
        replays.append(entry)
    bridge_source = REPO/"tmp/logical_topology_independent_review_20260911/bridge_audit.json"
    bridge = read_json(bridge_source)
    bridge_matches = all(file_hash(under(REPO, name)) == expected for name, expected in bridge["source_sha256"].items())
    check(bridge["passed"] and bridge_matches, "ieee14_bridge_source_integrity", str(bridge_source))
    if bridge_matches:
        bridge_target = root/"ieee14_normal_bridge_audit.json"
        if bridge_target.exists():
            check(file_hash(bridge_target) == file_hash(bridge_source), "existing_bridge_audit", str(bridge_target))
        else:
            shutil.copyfile(bridge_source, bridge_target)
    checker_copy = root/"independent_audit_sources"/f"{checker_sha}.py"
    checker_copy.parent.mkdir(parents=True, exist_ok=True)
    if checker_copy.exists():
        check(checker_copy.read_bytes() == checker_bytes, "audit_checker_source_copy", str(checker_copy))
    else:
        checker_copy.write_bytes(checker_bytes)
    result = {"contract": "logical_topology_independent_artifact_audit_v1", "passed": not failures,
              "created_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(root), "failures": failures,
              "source_snapshot_matches": snapshot == receipt["source_before"], "current_sources_match": current == receipt["source_before"],
              "script_sha256": checker_sha, "script_snapshot_path": str(checker_copy.relative_to(root)),
              "manifest_sha256": file_hash(corpus/"manifest.json"),
              "summary_sha256": file_hash(root/"summary.json"), "scenario_counts": dict(counts), "physical_world_count": len(parent_splits),
              "physically_admitted_rows": sum(r["physical_admission"]["admitted"] for r in rows),
              "compact_outcomes_recomputed": compact_summary(reports), "detailed_audits_checked": detailed_checked,
              "candidate_records_checked": candidate_checked, "physical_and_gse_replays": replays,
              "numerical_fit_reuse_totals": dict(reuse_totals), "immutable_corpus_source_run": str(source_run) if source_run else None,
              "comparison_guard_contract_counts": dict(guard_contract_counts), "archived_comparison_guard_contract": expected_guard_contract,
              "legacy_unshared_method_budget_present": bool(guard_contract_counts["logical_topology_pairwise_separation_guard_v1"]),
              "comparison_statistical_scope": "v2 checks a fixed equal Bonferroni allocation across both rejection methods and all rivals; v1 is historical unshared-budget arithmetic and does not support its nominal combined-method error budget. Neither contract establishes exact nonlinear error control.",
              "non_normal_healthy_controls": dict(healthy_open), "heldout_parent_count": len(heldout_parents),
              "ordinary_and_structural_parent_groups_disjoint": all(len(v) == 1 for v in [*parent_splits.values(), *structural_splits.values()]),
              "training_readiness": "not_certified; parent grouping and heldout membership do not guarantee family coverage",
              "physical_root_contract": manifest["config"]["physical_root_fingerprint_contract"],
              "scope": "Artifact integrity and honest outcome accounting, all declared sensor/status bindings, selected true-PF and stored GSE-state replays. No regeneration of noisy observations, teacher-based physical filtering, or global status-uniqueness claim."}
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)
    result = audit_run(args.output_dir, report_path=args.report_path)
    print(json.dumps({"passed": result["passed"], "failure_count": len(result["failures"]),
                      "detailed_audits_checked": result["detailed_audits_checked"], "candidate_records_checked": result["candidate_records_checked"],
                      "replays": len(result["physical_and_gse_replays"])}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
