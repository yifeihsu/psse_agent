"""Opt-in provider hooks for fixed-sensor logical topology states.

These hooks use the existing provider state and modification conventions. The
legacy protocol bridge and private truth audit still require separate coupler
support; this module does not install a release factory or claim oracle commit
validation. No logical state falls through to the legacy 3*nb+4*nl estimator.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from mcp_server.matpower_server import _load_python_case
from psse_env.actions import CORRECT_TOPOLOGY, GET_TOPOLOGY_CONTEXT
from psse_env.providers.matpower import MatpowerDeploymentProviders

from .inventory import process_topology
from .runtime import LogicalTopologyRuntime, evidence_hash


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _document(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an inline object or a pinned JSON reference")
    if set(value) == {"path", "sha256"}:
        path = Path(value["path"])
        if not path.is_absolute():
            raise ValueError(f"{label} JSON reference must use an absolute path")
        raw = path.resolve(strict=True).read_bytes()
        if hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError(f"{label} JSON content hash changed")
        result = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(result, Mapping):
            raise ValueError(f"{label} JSON must contain an object")
        return copy.deepcopy(dict(result))
    return copy.deepcopy(dict(value))


def _case(value):
    raw = (_document(value, "canonical base case") if isinstance(value, Mapping)
           else _load_python_case(str(value)))
    result = {key: copy.deepcopy(raw[key]) for key in ("version", "baseMVA", "bus", "gen", "branch", "gencost") if key in raw}
    for key in ("bus", "gen", "branch"):
        result[key] = np.asarray(result[key], dtype=float)
        if result[key].ndim != 2 or not np.isfinite(result[key]).all():
            raise ValueError(f"invalid {key} in logical case")
    return result


class LogicalTopologyProviders(MatpowerDeploymentProviders):
    """Inspectable, testable logical-CB provider with evidence-bound proposals."""

    def __init__(self, *, scan_options=None, context_cache_size=16, atomic_actions=False, **kwargs):
        kwargs.setdefault("chi2_alpha", .05)
        kwargs.setdefault("normalized_residual_threshold", 4.0)
        super().__init__(**kwargs)
        if self.normalized_residual_threshold is None:
            raise ValueError("logical topology requires an explicit normalized-residual threshold")
        if context_cache_size < 1:
            raise ValueError("context_cache_size must be positive")
        self.scan_options = copy.deepcopy(dict(scan_options or {}))
        self.context_cache_size = int(context_cache_size)
        self._contexts = OrderedDict()
        self.atomic_actions = bool(atomic_actions)

    def provider_hooks(self):
        """Explicit compatible hooks; caller must supply a logical-aware policy/audit."""
        return {"wls_runner": self.run_wls,
                "context_providers": {GET_TOPOLOGY_CONTEXT: self.get_topology_context},
                "correction_executors": {CORRECT_TOPOLOGY: self.correct_topology}}

    def env_kwargs(self):
        raise NotImplementedError(
            "Use provider_hooks explicitly. Legacy protocol/private truth audits do not yet certify logical bus couplers."
        )

    def _solve(self, state):
        raise ValueError("legacy numerical providers are disabled for logical physical-section sensors")

    def get_measurement_context(self, state):
        return self._failure("logical_measurement_route_not_implemented", **self._binding(state))

    def get_parameter_context(self, state):
        return self._failure("logical_parameter_route_not_implemented", **self._binding(state))

    def correct_measurements(self, state, action):
        return self._failure("logical_measurement_route_not_implemented", **self._binding(state))

    def correct_parameters(self, state, action):
        return self._failure("logical_parameter_route_not_implemented", **self._binding(state))

    def state_payload(self, case, inventory, statuses, sensors, observations):
        """Prepare store.create_root arguments without minting state IDs or truth."""
        base = _case(case)
        loaded_inventory = _document(inventory, "inventory")
        loaded_sensors = _document(sensors, "measurement inventory")
        if (observations["sensor_inventory_hash"] != loaded_sensors["sensor_inventory_hash"]
            or observations["sensor_ids"] != [row["sensor_id"] for row in loaded_sensors["records"]]
            or len(observations["values"]) != len(loaded_sensors["records"])):
            raise ValueError("observations and physical sensor inventory differ")
        for value, available in zip(observations["values"], loaded_sensors["available_mask"]):
            if not available and value is not None:
                raise ValueError("unavailable sensor values must be redacted before creating a state")
            if available and (value is None or not np.isfinite(float(value))):
                raise ValueError("available sensor values must be finite")
        compiled = process_topology(base, loaded_inventory, statuses)
        return {"case": self._derived_case(compiled["case"], "logical_initial"),
                "measurements": copy.deepcopy(observations["values"]),
                "metadata": {"logical_topology": {"contract": "logical_topology_provider_state_v1",
                    "base_case": _native(base), "inventory": _native(inventory),
                    "measurement_inventory": _native(sensors), "current_statuses": copy.deepcopy(statuses)}}}

    def _context(self, state, *, create=True):
        logical = state.get("metadata", {}).get("logical_topology")
        if not isinstance(logical, Mapping) or logical.get("contract") != "logical_topology_provider_state_v1":
            raise ValueError("metadata.logical_topology with the provider contract is required")
        if not state.get("state_id") or not state.get("state_hash"):
            raise ValueError("logical providers require a concrete state ID and state hash")
        inventory = _document(logical["inventory"], "inventory")
        sensors = _document(logical["measurement_inventory"], "measurement inventory")
        base = _case(logical["base_case"])
        statuses = copy.deepcopy(logical["current_statuses"])
        actual = _case(state["case"])
        if actual["branch"].shape[0] != base["branch"].shape[0]:
            raise ValueError("compiled case lost canonical branch-row identities")
        if actual["branch"].shape[1] < 13:
            raise ValueError("compiled case must retain all canonical branch parameter columns")
        if actual["baseMVA"] != base["baseMVA"]:
            raise ValueError("current baseMVA differs from the explicit canonical base case")
        # The legacy case-file projection contains only bus/gen/branch. Cost
        # data stays authoritative in canonical metadata; an inline current
        # case may not silently supply a different version or cost model.
        for key in ("version", "gencost"):
            if key in actual and (key not in base or not np.array_equal(actual[key], base[key])):
                raise ValueError(f"current {key} differs from the explicit canonical base case")
        if all(value is not None for value in statuses.values()):
            expected = process_topology(base, inventory, statuses)["case"]
            if not np.array_equal(actual["branch"][:, [0, 1, 10]], np.asarray(expected["branch"])[:, [0, 1, 10]]):
                raise ValueError("compiled case topology/equipment differs from current logical statuses")
        else:
            # Without a complete topology, a compiled bus-section allocation
            # cannot be inverted safely. Unknown-status callers must supply
            # their current operating parameters on the canonical basis.
            expected = base
            if not np.array_equal(actual["branch"][:, :2], base["branch"][:, :2]):
                raise ValueError("unknown statuses require the explicit canonical equipment basis")
        for key in ("bus", "gen"):
            if not np.array_equal(actual[key], expected[key]):
                raise ValueError(f"current {key} edits require matching canonical base-case metadata and section allocation")
        # Existing parameter edits are authoritative by immutable branch row.
        # Preserve angle limits and any retained trailing fields as well as
        # impedance/tap values. Only endpoints belong to the compiled basis;
        # the complete logical status map is authoritative in the runtime.
        canonical_endpoints = base["branch"][:, :2].copy()
        base["branch"] = actual["branch"].copy()
        base["branch"][:, :2] = canonical_endpoints
        values = copy.deepcopy(state["measurements"])
        if len(values) != len(sensors["records"]):
            raise ValueError("logical state must retain the full physical record layout")
        for value, available in zip(values, sensors["available_mask"]):
            if not available and value is not None:
                raise ValueError("unavailable sensor values must be redacted as None")
            if available and (value is None or not np.isfinite(float(value))):
                raise ValueError("available sensor values must be finite")
        observations = {"values": values, "sensor_ids": [row["sensor_id"] for row in sensors["records"]],
                        "sensor_inventory_hash": sensors["sensor_inventory_hash"]}
        binding = {**self._binding(state), "inventory": inventory, "measurement_inventory": sensors,
                   "case": base, "current_statuses": statuses, "observations": observations,
                   "chi2_alpha": self.chi2_alpha, "nr_threshold": self.normalized_residual_threshold,
                   "scan_options": self.scan_options}
        key = evidence_hash(binding)
        if key in self._contexts:
            self._contexts.move_to_end(key)
            return self._contexts[key]
        if not create:
            raise ValueError("no fresh topology context matches this state, model, and evidence")
        runtime = LogicalTopologyRuntime(inventory=inventory, current_case=base, current_statuses=statuses,
                                         measurement_inventory=sensors, observations=observations,
                                         chi2_alpha=self.chi2_alpha, normalized_residual_threshold=self.normalized_residual_threshold)
        entry = {"binding_key": key, "runtime": runtime, "logical_metadata": copy.deepcopy(dict(logical)),
                 "inventory": inventory, "sensors": sensors, "audit": None}
        self._contexts[key] = entry
        while len(self._contexts) > self.context_cache_size:
            self._contexts.popitem(last=False)
        return entry

    def _artifact(self, payload, prefix):
        encoded = (json.dumps(_native(payload), sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        directory = Path(self.derived_case_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f"{prefix}_{hashlib.sha256(encoded).hexdigest()}.json"
        if not path.is_file() or path.read_bytes() != encoded:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{prefix}_", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return str(path.resolve())

    def run_wls(self, state):
        try:
            entry = self._context(state)
            candidate = entry["runtime"].test_statuses({})
            fit = candidate.get("estimation")
            if not isinstance(fit, Mapping):
                return self._failure("logical_wls_unavailable", candidate["reason"], **self._binding(state),
                                     logical_resolution=candidate["resolution"], candidate_connected=candidate.get("connectivity", {}).get("connected"))
            path = self._artifact(fit, "logical_wls")
            j, threshold, nr = fit["wls_objective"], fit["chi_square_threshold"], fit["max_normalized_residual"]
            ratio = j/threshold if j is not None and threshold is not None and threshold > 0 else None
            score = max(ratio, nr/self.normalized_residual_threshold) if ratio is not None and nr is not None else None
            plausible = candidate["plausible"] is True
            residuals = sorted(({"index0": index, "sensor_id": entry["sensors"]["records"][index]["sensor_id"],
                                  "channel": entry["sensors"]["records"][index]["kind"], "normalized_residual": value}
                                 for index, value in enumerate(fit.get("normalized_residuals", [])) if value is not None),
                                key=lambda row: row["normalized_residual"], reverse=True)[:self.top_k]
            result = {**self._binding(state), "evidence_source": "deployment_wls:logical_physical_sections",
                      "state_estimation_converged": fit["converged"], "converged": fit["converged"],
                      "observable": fit["observable"], "wls_objective": j, "chi_square_statistic": j,
                      "chi_square_threshold": threshold, "chi_square_dof": fit["chi_square_dof"],
                      "chi_square_alpha": self.chi2_alpha, "chi_square_ratio": ratio,
                      "max_normalized_residual": nr, "normalized_residual_threshold": self.normalized_residual_threshold,
                      "chi_square_alarm": fit["chi_square_alarm"], "normalized_residual_alarm": fit["normalized_residual_alarm"],
                      "anomaly_detection_rule": "chi_square_or_normalized_residual", "remaining_anomaly_score": score,
                      "no_material_anomaly_remaining": plausible, "globally_resolved": plausible,
                      "unresolved_signatures": [] if plausible else ["wls_connectivity_or_analog_inconsistency"],
                      "logical_estimation": {key: fit.get(key) for key in ("rank", "state_dimension", "available_measurement_count", "raw_measurement_count", "closed_coupler_nuisance_count", "failure_reason")},
                      "wls_summary": {"global_metrics": {"global_residual_sum": j, "global_residual_threshold": threshold,
                                                            "global_residual_ratio": ratio}, "top_residuals": residuals},
                      "evidence_path": path, "logical_binding_key": entry["binding_key"],
                      "physical_constraints_ok": None, "physical_evidence_scope": "state_estimation_and_connectivity_only"}
            if not fit["converged"] or not fit["observable"]:
                result.update(execution_status="failure", error_code=fit.get("failure_reason") or "logical_wls_not_observable")
            source_action = state.get("source_action", {})
            arguments = source_action.get("arguments", {}) if isinstance(source_action, Mapping) else {}
            target = arguments.get("cb_name")
            if target is not None:
                status = entry["runtime"].inspect_cb(target)["current_status"]
                result["logical_target_status_matches_requested"] = status == arguments.get("status")
                result["target_fixed"] = plausible and status == arguments.get("status")
                result["post_action_resolved"] = plausible
                result["target_evidence_scope"] = "requested_model_status_plus_fixed_evidence_fit_not_private_truth"
            return result
        except Exception as exc:
            return self._failure("logical_wls_input_error", f"{type(exc).__name__}: {exc}", **self._binding(state))

    def inspect_cb(self, state, device_id):
        entry = self._context(state)
        return {**self._binding(state), **entry["runtime"].inspect_cb(device_id), "logical_binding_key": entry["binding_key"]}

    def test_cb(self, state, device_id, status):
        entry = self._context(state)
        candidate = entry["runtime"].test_cb(device_id, status)
        return {**self._binding(state), "candidate": candidate, "logical_binding_key": entry["binding_key"]}

    def get_topology_context(self, state):
        try:
            entry = self._context(state)
            audit = entry["runtime"].scan_candidates(**self.scan_options)
            entry["audit"] = audit
            supported = []
            unique = audit.get("unique_candidate_id")
            chosen = next((row for row in audit["plausible_candidates"] if row["candidate_id"] == unique), None)
            if chosen is not None and self.atomic_actions:
                supported.append({"tool": CORRECT_TOPOLOGY, "arguments": {
                    "state_id": state["state_id"], "candidate_id": unique,
                    "certificate_hash": evidence_hash(audit["certificate"]),
                    "desired_statuses": copy.deepcopy(chosen["changes"])}})
            elif chosen is not None and len(chosen["changes"]) == 1:
                device, status = next(iter(chosen["changes"].items()))
                supported.append({"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": state["state_id"], "cb_name": device, "status": status}})
            return {**self._binding(state), "context_tool": GET_TOPOLOGY_CONTEXT,
                    "evidence_source": "deployment_context:logical_cb_absolute_candidate_audit",
                    "supported_corrections": supported,
                    "route_status": "actionable" if supported else "complete_negative" if audit["decision"] == "keep_current" else "unavailable_or_inconclusive",
                    "route_status_reason": audit["reason"], "logical_decision": audit["decision"],
                    "logical_binding_key": entry["binding_key"], "unique_candidate_id": unique,
                    "tested_candidate_count": audit["tested_candidate_count"],
                    "plausible_candidate_count": len(audit["plausible_candidates"]),
                    "unresolved_candidate_count": audit["unresolved_candidate_count"],
                    "scope_complete": audit["scope_complete"], "hypothesis_scope": audit["hypothesis_scope"],
                    "certificate": copy.deepcopy(audit["certificate"]),
                    "evidence_path": self._artifact(audit, "logical_candidate_audit")}
        except Exception as exc:
            return self._failure("logical_topology_context_error", f"{type(exc).__name__}: {exc}", **self._binding(state))

    def apply_logical_candidate(self, state, candidate_id):
        """Return a transactional modification from the bound complete certificate."""
        entry = self._context(state, create=False)
        audit = entry["audit"]
        if not audit or audit.get("unique_candidate_id") != candidate_id:
            raise ValueError("fresh complete candidate-scan certificate required")
        runtime = copy.deepcopy(entry["runtime"])
        applied = runtime.apply(candidate_id)
        compiled = process_topology(applied["current_case"], applied["inventory"], applied["current_statuses"])
        path = self._derived_case(compiled["case"], f"logical_{candidate_id[:12]}")
        return {"modification": {"case": path, "metadata_updates": {"logical_topology": {
                    "base_case": _native(applied["current_case"]), "current_statuses": applied["current_statuses"],
                    "last_identification_certificate": applied["identification_certificate"]},
                    "last_logical_topology_correction": {"candidate_id": candidate_id, "changes": applied["applied_changes"],
                                                         "parent_binding_key": entry["binding_key"]}}},
                "evidence_source": "deployment_correction:logical_cb_candidate_certificate",
                "logical_candidate_id": candidate_id, "logical_changes": applied["applied_changes"],
                "measurements_and_covariance_preserved": True,
                "compiled_electrical_bus_count": len(compiled["case"]["bus"]),
                "identification_certificate": applied["identification_certificate"]}

    def correct_topology(self, state, action):
        try:
            arguments = action.get("arguments", {})
            if self.atomic_actions:
                from .atomic import validate_atomic_arguments
                validate_atomic_arguments(arguments)
                if action.get("tool") != CORRECT_TOPOLOGY or arguments["state_id"] != state["state_id"]:
                    raise ValueError("logical atomic correction requires this state ID")
                entry = self._context(state, create=False)
                audit = entry["audit"]
                if not audit or audit.get("unique_candidate_id") != arguments["candidate_id"]:
                    raise ValueError("fresh unique candidate context required")
                if arguments["certificate_hash"] != evidence_hash(audit["certificate"]):
                    raise ValueError("submitted certificate hash differs from the complete certificate")
                chosen = next(row for row in audit["plausible_candidates"] if row["candidate_id"] == arguments["candidate_id"])
                if chosen["changes"] != arguments["desired_statuses"]:
                    raise ValueError("atomic desired statuses must exactly match every certified change")
                return self.apply_logical_candidate(state, chosen["candidate_id"])
            if action.get("tool") != CORRECT_TOPOLOGY or set(arguments) != {"state_id", "cb_name", "status"} or arguments["state_id"] != state["state_id"]:
                raise ValueError("logical correction requires this state ID, one cb_name, and status")
            entry = self._context(state, create=False)
            audit = entry["audit"]
            if not audit or not audit.get("unique_candidate_id"):
                raise ValueError("fresh unique candidate context required")
            chosen = next(row for row in audit["plausible_candidates"] if row["candidate_id"] == audit["unique_candidate_id"])
            if chosen["changes"] != {arguments["cb_name"]: arguments["status"]}:
                raise ValueError("correction does not match the certified logical candidate")
            return self.apply_logical_candidate(state, chosen["candidate_id"])
        except Exception as exc:
            return self._failure("logical_topology_correction_rejected", f"{type(exc).__name__}: {exc}", **self._binding(state))
