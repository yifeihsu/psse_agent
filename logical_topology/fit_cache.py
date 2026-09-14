"""Verified reuse of numerical fits, never reuse of old topology decisions.

Cache keys bind the complete canonical numerical case, authoritative statuses,
physical inventory, available observations, full covariance, and solver settings.
Historical fit-slot names stay in an offline receipt: identifying a cache entry
as a true-status fit must not disclose a truth label to the decision runtime.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import gzip
import hashlib
import importlib.metadata
import json
import numbers
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
NUMERICAL_SOURCES = ("logical_topology/estimation.py", "logical_topology/inventory.py", "logical_topology/measurements.py",
                     "psse_env/systems/registry.py", "mcp_server/case14.m", "mcp_server/case57.m")


def numerical_versions() -> dict[str, str | None]:
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": version("scipy"), "pypower": version("PYPOWER")}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {value}")))


def under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return path


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    return value


def legacy_hash(value) -> str:
    return hashlib.sha256(json.dumps(_native(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _canonical(value):
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        try:
            array = np.asarray(value)
            if array.dtype.kind in "fiu":
                array = np.array(array, dtype="<f8", order="C", copy=True)
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite numerical cache input")
                array[array == 0] = 0.0
                return {"numeric_shape": list(array.shape), "float64_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        except (TypeError, ValueError) as exc:
            if "Nonfinite" in str(exc):
                raise
        return [_canonical(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value.item() if isinstance(value, np.generic) else value


def semantic_hash(value) -> str:
    return legacy_hash(_canonical(value))


def authoritative_case(case, inventory, statuses):
    """Preserve numerical parameters while giving the full status map authority."""
    devices = [row["device_id"] for row in inventory["branches"]] + [row["device_id"] for row in inventory["couplers"]]
    if set(statuses) != set(devices) or any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
                                           or value not in (0, 1) for value in statuses.values()):
        raise ValueError("A numerical-fit cache key requires complete binary statuses")
    result = deepcopy(case)
    for row in inventory["branches"]:
        index, value = int(row["row0"]), int(statuses[row["device_id"]])
        if isinstance(result["branch"], np.ndarray):
            result["branch"][index, 10] = value
        else:
            result["branch"][index][10] = value
    return result


def _nonstatus(case):
    result = deepcopy(case)
    branch = np.array(result["branch"], dtype=float, copy=True)
    branch[:, 10] = 0
    result["branch"] = branch
    return result


class VerifiedSourceRun:
    """A completed, archived source run with independently checked byte hashes."""

    def __init__(self, source_run, *, repo_root=REPO):
        self.root = Path(source_run).resolve(strict=True)
        self.corpus = self.root / "corpus"
        self.repo = Path(repo_root).resolve(strict=True)
        self.receipt = _read(self.root / "run_receipt.json")
        self.current_versions = numerical_versions()
        self.historical_versions = self.receipt.get("numerical_versions")
        self.version_attestation = ("not_recorded_in_source_run" if self.historical_versions is None else
                                    "matches_current" if self.historical_versions == self.current_versions else "differs_from_current")
        if self.receipt.get("all_sources_unchanged_during_run") is not True or self.receipt.get("all_rows_audited") is not True:
            raise ValueError("Source run is incomplete or its recorded implementation changed")
        before, after = self.receipt["source_before"], self.receipt["source_after"]
        if before != after:
            raise ValueError("Source run hash receipts disagree")
        for name, digest in before.items():
            if file_sha256(under(self.root / "implementation_snapshot", name)) != digest:
                raise ValueError(f"Archived source hash mismatch: {name}")
        if file_sha256(self.corpus / "manifest.json") != self.receipt["manifest_sha256"]:
            raise ValueError("Source manifest changed since the completed audit")
        self.manifest = _read(self.corpus / "manifest.json")
        self.rows = {row["scenario_id"]: row for row in self.manifest["rows"]}
        if len(self.rows) != len(self.manifest["rows"]):
            raise ValueError("Source manifest has duplicate scenario identifiers")
        self.numerical_hashes = {name: before[name] for name in NUMERICAL_SOURCES}
        source = under(self.root / "implementation_snapshot", "logical_topology/estimation.py").read_text(encoding="utf-8")
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "estimate")
        self.defaults = {arg.arg: ast.literal_eval(default) for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults)}
        self.legacy_default_budget_proven = True
        # The archived first-run call sites must not silently override a
        # missing max_nfev field in old fit payloads.
        for name in ("logical_topology/runtime.py", "logical_topology/audit.py"):
            tree = ast.parse(under(self.root / "implementation_snapshot", name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                if called in ("estimate", "_estimator") and any(keyword.arg not in {"chi2_alpha", "normalized_residual_threshold"} for keyword in node.keywords):
                    self.legacy_default_budget_proven = False

    def sources_match_now(self) -> bool:
        return all(file_sha256(self.repo / name) == digest for name, digest in self.numerical_hashes.items())


class VerifiedEstimatorCache:
    """Per-scenario callable estimator with explicit offline reuse provenance."""

    def __init__(self, source: VerifiedSourceRun, row: Mapping[str, Any], *, estimator=None):
        if estimator is None:
            from .estimation import estimate
            estimator = estimate
        self.source, self.fresh_estimator = source, estimator
        scenario_id = str(row["scenario_id"])
        if scenario_id not in source.rows or legacy_hash(row) != legacy_hash(source.rows[scenario_id]):
            raise ValueError("Revalidation row differs from the immutable source manifest")
        self.row = deepcopy(row)
        self.entries = {}
        self.conflicts = set()
        self.log = []
        self.index_errors = []
        self.lookup_count = self.reused_count = self.fresh_call_count = self.fresh_solve_count = 0
        if not source.sources_match_now() or source.version_attestation == "differs_from_current":
            self.index_errors.append("numerical_source_hash_or_recorded_version_changed; all lookups require new fits")
            return
        if not row["physical_admission"]["admitted"]:
            return
        execution = row["execution"]
        self.inventory = _read(under(source.corpus, execution["inventory_path"]))
        self.sensors = _read(under(source.corpus, execution["measurement_inventory_path"]))
        self.observations = _read(under(source.corpus, execution["observations_path"]))
        self.case = _read(under(source.corpus, execution["base_case_path"]))
        compact_path = source.corpus / "row_audits" / f"{scenario_id}.json"
        if not compact_path.exists():
            self.index_errors.append("old_compact_audit_missing")
            return
        compact = _read(compact_path)
        if not compact.get("detailed_audit_path"):
            self.index_errors.append("old_numerical_audit_unavailable")
            return
        for name, value in (("inventory", self.inventory), ("measurement_inventory", self.sensors),
                            ("observations", self.observations), ("current_case", self.case)):
            if compact["execution_input_hashes"][name] != legacy_hash(value):
                raise ValueError(f"Source execution input changed: {name}")
        detailed = under(source.corpus, compact["detailed_audit_path"])
        if file_sha256(detailed) != compact["detailed_audit_sha256"]:
            raise ValueError("Source detailed audit hash mismatch")
        self.source_artifact_sha256 = compact["detailed_audit_sha256"]
        with gzip.open(detailed, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        fixed_hash = legacy_hash({"measurement_inventory": self.sensors, "observations": self.observations})
        current_statuses = execution["current_statuses"]
        parent_case = deepcopy(self.case)
        for branch in self.inventory["branches"]:
            value = current_statuses[branch["device_id"]]
            if value is not None:
                parent_case["branch"][branch["row0"]][10] = int(value)
        parent_hash = legacy_hash({"inventory": self.inventory, "current_case": parent_case,
                                   "current_statuses": current_statuses,
                                   "configuration": {"chi2_alpha": .05, "normalized_residual_threshold": 4., "connected_only": True}})
        for index, candidate in enumerate(payload["runtime_scan"]["candidates"]):
            statuses = candidate["statuses"]
            candidate_id = legacy_hash({"parent": parent_hash, "statuses": statuses, "evidence": fixed_hash})
            if (candidate["parent_model_hash"] != parent_hash or candidate["fixed_evidence_hash"] != fixed_hash
                or candidate["candidate_id"] != candidate_id):
                raise ValueError("Source candidate model/evidence binding mismatch")
            self._add(self.case, statuses, candidate.get("estimation"), f"runtime_scan.candidates[{index}]")
        true_statuses = row["true_statuses"]
        self._add(self.case, true_statuses, payload.get("offline_true_status_model_fit"), "offline_true_status_model_fit")
        physical = _read(under(source.corpus, row["physical_audit_path"]))
        physical_case = deepcopy(self.case)
        overlay = row.get("parameter_error")
        if overlay:
            index = int(overlay["branch_row0"])
            original_rx = np.asarray(overlay["true_r_x"], dtype=float)
            if not np.allclose(np.asarray(physical_case["branch"][index][2:4]), original_rx*float(overlay["factor"]), rtol=0, atol=1e-12):
                raise ValueError("Declared parameter overlay does not match its bound model input")
            physical_case["branch"][index][2:4] = original_rx.tolist()
        if semantic_hash(_nonstatus(physical_case)) == semantic_hash(_nonstatus(physical["operating_case"])):
            self._add(physical_case, true_statuses, payload.get("offline_true_physical_status_fit"), "offline_true_physical_status_fit")
        else:
            self.index_errors.append("historical_true_physical_case_not_bound; its fit will be recomputed")

    def _key(self, case, inventory, statuses, observations, sensors, parameters):
        normalized = authoritative_case(case, inventory, statuses)
        return semantic_hash({"case": normalized, "inventory": inventory,
                              "statuses": {key: int(value) for key, value in statuses.items()},
                              "observations": observations, "sensors": sensors, "parameters": parameters,
                              "numerical_source_sha256": self.source.numerical_hashes})

    def _add(self, case, statuses, fit, origin):
        if not isinstance(fit, Mapping) or "function_evaluations" not in fit:
            return
        if (fit.get("sensor_inventory_hash") != self.sensors["sensor_inventory_hash"]
            or fit.get("raw_measurement_count") != len(self.sensors["records"])
            or fit.get("available_measurement_count") != sum(self.sensors["available_mask"])):
            raise ValueError("Historical numerical fit sensor binding is inconsistent")
        provenance = fit.get("numerical_fit_execution") or {}
        if provenance.get("parameters"):
            parameters = dict(provenance["parameters"])
        elif self.source.legacy_default_budget_proven:
            parameters = {"chi2_alpha": float(fit["chi_square_alpha"]),
                          "normalized_residual_threshold": float(fit["normalized_residual_threshold"]),
                          "max_nfev": self.source.defaults["max_nfev"]}
        else:
            self.index_errors.append(f"historical_solver_budget_not_proven:{origin}")
            return
        key = self._key(case, self.inventory, statuses, self.observations, self.sensors, parameters)
        numerical = {name: deepcopy(value) for name, value in fit.items() if name != "numerical_fit_execution"}
        if key in self.conflicts:
            return
        if key in self.entries and legacy_hash(self.entries[key]["fit"]) != legacy_hash(numerical):
            self.entries.pop(key)
            self.conflicts.add(key)
            self.index_errors.append(f"conflicting_historical_results_for_exact_input:{key}")
            return
        entry = self.entries.setdefault(key, {"fit": numerical, "origins": []})
        entry["origins"].append(origin)

    def __call__(self, case, inventory, statuses, observations, sensors, **kwargs):
        self.lookup_count += 1
        parameters = {**self.source.defaults, **kwargs}
        compatible_sources = self.source.sources_match_now() and self.source.version_attestation != "differs_from_current"
        try:
            key = self._key(case, inventory, statuses, observations, sensors, parameters)
        except (TypeError, ValueError):
            key = None
        entry = self.entries.get(key) if compatible_sources else None
        if entry is not None:
            self.reused_count += 1
            result = deepcopy(entry["fit"])
            reused, fresh_solve = True, False
            self.log.append({"semantic_input_sha256": key, "kind": "reused_verified_numeric_fit",
                             "source_scenario_id": self.row["scenario_id"], "source_fit_slots": list(entry["origins"]),
                             "source_artifact_sha256": self.source_artifact_sha256})
        else:
            self.fresh_call_count += 1
            # Explicitly pass the recorded defaults as well as overrides so a
            # fallback never silently runs with an unrecorded solver budget.
            try:
                result = dict(self.fresh_estimator(case, inventory, statuses, observations, sensors, **parameters))
            except Exception as exc:
                self.log.append({"semantic_input_sha256": key, "kind": "fresh_estimator_exception",
                                 "error": f"{type(exc).__name__}: {exc}", "fresh_solve": None})
                raise
            reused, fresh_solve = False, "function_evaluations" in result
            self.fresh_solve_count += int(fresh_solve)
            self.log.append({"semantic_input_sha256": key, "kind": "fresh_estimator_call",
                             "reason": "numerical_source_hash_changed" if not compatible_sources else "exact_semantic_input_cache_miss",
                             "fresh_solve": fresh_solve})
        # Generic numerical provenance only. Privileged old slot names are
        # retained in receipt(), which the driver saves after runtime decisions.
        result["numerical_fit_execution"] = {
            "kind": "reused_verified_numeric_fit" if reused else "fresh_estimator_call",
            "reused": reused, "fresh_solve": fresh_solve, "semantic_input_sha256": key,
            "estimation_source_sha256": file_sha256(self.source.repo / "logical_topology/estimation.py"),
            "parameters": parameters,
        }
        return result

    def receipt(self):
        return {"contract": "verified_fixed_input_numerical_fit_reuse_v1", "policy_observable": False,
                "scenario_id": self.row["scenario_id"], "source_run": str(self.source.root),
                "numerical_sources": self.source.numerical_hashes, "numerical_sources_match": self.source.sources_match_now(),
                "current_numerical_versions": self.source.current_versions,
                "historical_numerical_versions": self.source.historical_versions,
                "historical_version_attestation": self.source.version_attestation,
                "reuse_claim": "same checked numerical repository sources and exact fixed inputs; historical library-version equality is not asserted when unrecorded",
                "indexed_exact_inputs": len(self.entries), "conflicting_inputs": len(self.conflicts),
                "lookups": self.lookup_count, "reused_fits": self.reused_count,
                "fresh_estimator_calls": self.fresh_call_count, "fresh_numerical_solves": self.fresh_solve_count,
                "index_notes": list(self.index_errors), "lookups_provenance": deepcopy(self.log),
                "old_candidate_decisions_or_certificates_reused": False}
