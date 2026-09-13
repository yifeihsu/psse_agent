"""Evidence-preserving discrete logical-CB hypotheses and guarded application.

The equipment case stays on its canonical bus/branch basis. Asset-status CBs
change their branch row; bus couplers change only logical connectivity and are
processed by contraction/expansion. No power flow, redispatch, truth labels,
measurement regeneration, or normal-status reset belongs in this adapter.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def evidence_hash(value: Any) -> str:
    """Stable content hash for model/evidence binding, including full covariance."""
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _status(value: Any, *, unknown_allowed: bool = False) -> int | None:
    if value is None and unknown_allowed:
        return None
    if not isinstance(value, Real) or not math.isfinite(float(value)) or value not in (0, 1):
        raise ValueError("A CB status must be binary 0/1; unknown status must be explicit None")
    return int(value)


class LogicalTopologyRuntime:
    """A current-model snapshot with immutable observations and scoped certificates."""

    def __init__(
        self, *, inventory: Mapping[str, Any], current_case: Mapping[str, Any],
        current_statuses: Mapping[str, Any], measurement_inventory: Any,
        observations: Any, estimator: Callable[..., Mapping[str, Any]] | None = None,
        processor: Callable[..., Mapping[str, Any]] | None = None,
        chi2_alpha: float = 0.05, normalized_residual_threshold: float = 4.0,
        connected_only: bool = True,
    ) -> None:
        if not math.isfinite(chi2_alpha) or not 0 < chi2_alpha < 1:
            raise ValueError("chi2_alpha must be in (0,1)")
        if not math.isfinite(normalized_residual_threshold) or normalized_residual_threshold <= 0:
            raise ValueError("normalized_residual_threshold must be positive and finite")
        if estimator is None:
            from .estimation import estimate
            estimator = estimate
        if processor is None:
            from .inventory import process_topology
            processor = process_topology
        self._inventory = deepcopy(inventory)
        self._case = deepcopy(current_case)
        self._sensors = deepcopy(measurement_inventory)
        self._observations = deepcopy(observations)
        self._estimator, self._processor = estimator, processor
        self._configuration = {
            "chi2_alpha": float(chi2_alpha),
            "normalized_residual_threshold": float(normalized_residual_threshold),
            "connected_only": bool(connected_only),
        }
        self._devices: dict[str, dict[str, Any]] = {}
        for collection, kind in (("branches", "branch_status"), ("couplers", "bus_coupler")):
            for record in self._inventory.get(collection, []):
                device_id = str(record["device_id"])
                if device_id in self._devices:
                    raise ValueError(f"Duplicate logical device {device_id}")
                if record.get("device_kind") != kind:
                    raise ValueError(f"Incorrect electrical device kind for {device_id}")
                self._devices[device_id] = dict(record)
        if not self._devices:
            raise ValueError("No logical devices were supplied")
        if set(current_statuses) != set(self._devices):
            raise ValueError("current_statuses must contain every logical CB exactly once; use None for unknown")
        self._statuses = {key: _status(current_statuses[key], unknown_allowed=True) for key in self._devices}
        self._case = self._case_for(self._statuses)
        self._fixed_evidence_hash = self._evidence_hash()
        self._flow_proof_rows = self._prepare_flow_proof()
        self._fit_cache: dict[str, dict[str, Any]] = {}
        self._tested: dict[str, dict[str, Any]] = {}
        self._certificate: dict[str, Any] | None = None
        self._last_scan: dict[str, Any] | None = None

    def _evidence_hash(self) -> str:
        return evidence_hash({"measurement_inventory": self._sensors, "observations": self._observations})

    def _model_hash(self) -> str:
        return evidence_hash({"inventory": self._inventory, "current_case": self._case,
                              "current_statuses": self._statuses, "configuration": self._configuration})

    def _assert_fixed_evidence(self) -> None:
        if self._evidence_hash() != self._fixed_evidence_hash:
            raise ValueError("Fixed observations or covariance changed; create a new runtime snapshot")

    def _case_for(self, statuses: Mapping[str, Any]) -> dict[str, Any]:
        case = deepcopy(self._case)
        branches = np.asarray(case["branch"])
        if branches.ndim != 2 or branches.shape[1] <= 10:
            raise ValueError("Canonical equipment case needs a BR_STATUS column")
        for device_id, device in self._devices.items():
            if device["device_kind"] != "branch_status" or statuses[device_id] is None:
                continue
            row = int(device["row0"])
            if row < 0 or row >= len(branches):
                raise ValueError(f"Invalid canonical branch row for {device_id}")
            if isinstance(case["branch"], np.ndarray):
                case["branch"][row, 10] = statuses[device_id]
            else:
                case["branch"][row][10] = statuses[device_id]
        return case

    def snapshot(self) -> dict[str, Any]:
        self._assert_fixed_evidence()
        return {"inventory": deepcopy(self._inventory), "current_case": deepcopy(self._case),
                "current_statuses": deepcopy(self._statuses), "measurement_inventory": deepcopy(self._sensors),
                "observations": deepcopy(self._observations), "model_hash": self._model_hash(),
                "fixed_evidence_hash": self._fixed_evidence_hash,
                "branch_status_authority": "current_statuses; None is never compiled as a normal status"}

    def inspect_cb(self, device_id: str) -> dict[str, Any]:
        if device_id not in self._devices:
            raise ValueError(f"Unknown logical device {device_id}")
        device = self._devices[device_id]
        result = {"device_id": device_id, "device_kind": device["device_kind"],
                  "current_status": self._statuses[device_id], "status_known": self._statuses[device_id] is not None}
        for field in ("asset_id", "row0", "from_node", "to_node", "node_a", "node_b", "base_bus"):
            if field in device:
                result[field] = deepcopy(device[field])
        result["electrical_operation"] = (
            "whole_branch_terminal_admittance_multiplier" if device["device_kind"] == "branch_status"
            else "ideal_bus_section_contraction_or_separation"
        )
        return result

    def _calibrate(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """A small J is insufficient: both tests and full state rank must pass."""
        if metrics.get("converged") is not True:
            return {"resolution": "unresolved", "plausible": False,
                    "reason": metrics.get("failure_reason") or "estimation_did_not_converge"}
        try:
            m = int(metrics["available_measurement_count"])
            rank, dimension = int(metrics["rank"]), int(metrics["state_dimension"])
            statistic = float(metrics["wls_objective"])
            maximum = float(metrics["max_normalized_residual"])
            if not (0 <= rank <= min(m, dimension) and dimension > 0):
                raise ValueError("invalid rank or dimensions")
            if not all(math.isfinite(value) and value >= 0 for value in (statistic, maximum)):
                raise ValueError("invalid residual metrics")
            dof = m - rank
            if dof <= 0:
                raise ValueError("no positive residual degrees of freedom")
            threshold = float(chi2.ppf(1 - self._configuration["chi2_alpha"], dof))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {"resolution": "unresolved", "plausible": False, "reason": f"invalid_estimation_evidence:{exc}"}
        fields = {"chi_square_dof": dof, "chi_square_threshold": threshold,
                  "normalized_residual_threshold": self._configuration["normalized_residual_threshold"],
                  "chi_square_alarm": statistic >= threshold,
                  "normalized_residual_alarm": maximum >= self._configuration["normalized_residual_threshold"]}
        if metrics.get("observable") is not True or rank != dimension:
            return {**fields, "resolution": "unresolved", "plausible": False, "reason": "candidate_state_unobservable"}
        plausible = not (fields["chi_square_alarm"] or fields["normalized_residual_alarm"])
        return {**fields, "resolution": "plausible" if plausible else "rejected", "plausible": plausible,
                "reason": "both_absolute_residual_tests_pass" if plausible else "absolute_residual_test_failed"}

    def _prepare_flow_proof(self) -> list[dict[str, Any]]:
        """Validate the fixed sensor evidence once before using a necessary condition."""
        if not isinstance(self._sensors, Mapping) or not isinstance(self._observations, Mapping):
            return []
        try:
            records = self._sensors["records"]
            if (self._observations["sensor_inventory_hash"] != self._sensors["sensor_inventory_hash"]
                or self._sensors["layout_hash"] != self._inventory["layout_hash"]
                or self._observations["sensor_ids"] != [row["sensor_id"] for row in records]):
                return []
            values = np.asarray(self._observations["values"], dtype=float)
            covariance = np.asarray(self._sensors["covariance"], dtype=float)
            mask = np.asarray(self._sensors["available_mask"], dtype=bool)
            count = len(records)
            if (values.shape != (count,) or covariance.shape != (count, count) or mask.shape != (count,)
                or not np.isfinite(values[mask]).all() or not np.isfinite(covariance).all()
                or not np.array_equal(mask, [bool(row["available"]) for row in records])
                or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14)):
                return []
            diagonal = np.diag(covariance)
            if np.min(diagonal) <= 0:
                return []
            if not np.array_equal(covariance, np.diag(diagonal)):
                np.linalg.cholesky(covariance)
            by_row = {int(device["row0"]): device_id for device_id, device in self._devices.items()
                      if device["device_kind"] == "branch_status"}
            return [{"device_id": by_row[int(row["branch_row0"])] ,
                     "branch_row0": int(row["branch_row0"]), "sensor_id": row["sensor_id"],
                     "observed_value": float(values[index]), "marginal_variance": float(diagonal[index]),
                     "normalized_residual_lower_bound": float(abs(values[index]) / math.sqrt(diagonal[index]))}
                    for index, row in enumerate(records)
                    if mask[index] and row["kind"] in {"Pf", "Qf", "Pt", "Qt"}]
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            # Invalid or unsupported evidence receives the full estimator's
            # explicit input failure; it never earns a mathematical rejection.
            return []

    def _zero_flow_proof(self, statuses: Mapping[str, Any]) -> dict[str, Any] | None:
        rows = [row for row in self._flow_proof_rows if statuses[row["device_id"]] == 0
                and row["normalized_residual_lower_bound"] >= self._configuration["normalized_residual_threshold"]]
        if not rows:
            return None
        witness = max(rows, key=lambda row: row["normalized_residual_lower_bound"])
        return {"contract": "excluded_asset_flow_necessary_condition_v1", **deepcopy(witness),
                "normalized_residual_threshold": self._configuration["normalized_residual_threshold"],
                "fully_evaluated_necessary_condition": True, "model_prediction": 0.0,
                "proof": "whole-branch status zero forces h_i=0 and H_i=0, hence residual variance Omega_ii=R_ii",
                "covariance_and_availability_unchanged": True}

    def test_cb(self, device_id: str, status: int) -> dict[str, Any]:
        return self.test_statuses({device_id: status})

    def test_statuses(self, changes: Mapping[str, Any], *, force_full_fit: bool = False) -> dict[str, Any]:
        """Test model changes against identical observations and full covariance."""
        self._assert_fixed_evidence()
        result = self._test_statuses(changes, force_full_fit=force_full_fit)
        self._assert_fixed_evidence()
        return result

    def _test_statuses(self, changes: Mapping[str, Any], *, force_full_fit: bool = False) -> dict[str, Any]:
        if not isinstance(changes, Mapping) or set(changes) - set(self._devices):
            raise ValueError("Status changes must map registered logical CBs to binary statuses")
        statuses = dict(self._statuses)
        for key, value in changes.items():
            statuses[key] = _status(value)
        changes = {key: value for key, value in statuses.items() if value != self._statuses[key]}
        parent_hash = self._model_hash()
        candidate_id = evidence_hash({"parent": parent_hash, "statuses": statuses, "evidence": self._fixed_evidence_hash})
        base = {"candidate_id": candidate_id, "parent_model_hash": parent_hash,
                "fixed_evidence_hash": self._fixed_evidence_hash, "changes": changes,
                "statuses": statuses, "device_kinds": {key: self._devices[key]["device_kind"] for key in changes},
                "current_model": not changes}
        unknown = [key for key, value in statuses.items() if value is None]
        if unknown:
            result = {**base, "resolution": "unresolved", "plausible": False,
                      "reason": "unknown_statuses_require_binary_hypotheses", "unknown_device_ids": unknown,
                      "estimation": None}
            self._tested[candidate_id] = deepcopy(result)
            return deepcopy(result)
        case = self._case_for(statuses)
        cache_key = evidence_hash({"case": case, "statuses": statuses, "inventory": self._inventory,
                                   "evidence": self._fixed_evidence_hash, "configuration": self._configuration})
        cached = self._fit_cache.get(cache_key)
        if force_full_fit and cached is not None and cached.get("estimation") is None:
            cached = None
        if cached is None:
            proof = self._zero_flow_proof(statuses)
            try:
                if proof is not None and changes and not force_full_fit:
                    cached = {"resolution": "analytical_rejection", "plausible": False,
                              "reason": "available_nonzero_flow_on_excluded_branch",
                              "analytical_proof": proof, "solver_executed": False,
                              "fully_evaluated_necessary_condition": True, "estimation": None}
                else:
                    processed = self._processor(deepcopy(case), deepcopy(self._inventory), deepcopy(statuses))
                    connectivity = deepcopy(processed.get("connectivity", {}))
                    if self._configuration["connected_only"] and connectivity.get("connected") is False:
                        cached = {"resolution": "excluded", "plausible": False,
                                  "reason": "outside_declared_connected_energized_scope", "connectivity": connectivity,
                                  "solver_executed": False, "estimation": None}
                    else:
                        metrics = dict(self._estimator(
                            deepcopy(case), deepcopy(self._inventory), deepcopy(statuses),
                            deepcopy(self._observations), deepcopy(self._sensors),
                            chi2_alpha=self._configuration["chi2_alpha"],
                            normalized_residual_threshold=self._configuration["normalized_residual_threshold"],
                        ))
                        calibrated = self._calibrate(metrics)
                        execution = metrics.get("numerical_fit_execution") or {}
                        cached = {**calibrated, "connectivity": connectivity, "full_estimate_returned": True,
                                  "solver_executed": execution.get("fresh_solve", True),
                                  "numerical_fit_reused": execution.get("reused", False), "estimation": metrics}
                        if proof is not None and (not force_full_fit or calibrated["resolution"] == "rejected"):
                            # Keep the initial full WLS result for diagnosis;
                            # a proof can still rule out its topology when a
                            # wrong-model solve failed numerically.
                            cached.update(resolution="analytical_rejection", plausible=False,
                                          reason="available_nonzero_flow_on_excluded_branch",
                                          analytical_proof=proof, fully_evaluated_necessary_condition=True)
                        elif proof is not None:
                            # A requested full fit can expose a failed solve or
                            # conflicting evidence. Keep its membership rather
                            # than concealing it behind the ordinary NR4 proof.
                            cached["analytical_proof"] = proof
                            cached["ordinary_zero_flow_proof_retained_for_review"] = True
            except Exception as exc:
                cached = {"resolution": "unresolved", "plausible": False,
                          "reason": f"candidate_evaluation_failed:{type(exc).__name__}:{exc}", "estimation": None}
            self._fit_cache[cache_key] = deepcopy(cached)
        result = {**base, **deepcopy(cached)}
        self._tested[candidate_id] = deepcopy(result)
        return deepcopy(result)

    def scan_candidates(
        self, *, device_ids: Sequence[str] | None = None, include_pairs: bool = False,
        pair_devices: Sequence[str] | None = None, max_pairs: int = 64,
        max_unknown_hypotheses: int = 16,
    ) -> dict[str, Any]:
        """Enumerate a declared neighborhood; a truncated search cannot certify uniqueness."""
        self._assert_fixed_evidence()
        self._certificate = None
        self._last_scan = None
        pool = list(self._devices) if device_ids is None else list(device_ids)
        if len(set(pool)) != len(pool) or set(pool) - set(self._devices):
            raise ValueError("device_ids must be distinct registered CB identifiers")
        if isinstance(max_pairs, bool) or int(max_pairs) != max_pairs or max_pairs < 0:
            raise ValueError("max_pairs must be a nonnegative integer")
        if isinstance(max_unknown_hypotheses, bool) or int(max_unknown_hypotheses) != max_unknown_hypotheses or max_unknown_hypotheses < 1:
            raise ValueError("max_unknown_hypotheses must be a positive integer")
        unknown = [key for key, value in self._statuses.items() if value is None]
        known_pool = [key for key in pool if self._statuses[key] is not None]
        pair_pool = known_pool if pair_devices is None else list(pair_devices)
        if pair_devices is not None and not include_pairs:
            raise ValueError("pair_devices requires include_pairs=True")
        if len(set(pair_pool)) != len(pair_pool) or set(pair_pool) - set(known_pool):
            raise ValueError("pair_devices must be distinct known-status members of device_ids")
        all_pairs = list(combinations(pair_pool, 2)) if include_pairs else []
        selected_pairs = all_pairs[:int(max_pairs)]
        assignment_count = 2 ** len(unknown)
        unknown_complete = assignment_count <= max_unknown_hypotheses
        assignments = list(product((0, 1), repeat=len(unknown))) if unknown_complete else []
        scope = {"kind": "declared_discrete_logical_cb_neighborhood",
                 "single_flip_device_ids": pool, "unknown_device_ids": unknown,
                 "unknown_binary_assignment_count": assignment_count,
                 "unknown_assignments_tested": len(assignments),
                 "pair_device_ids": pair_pool if include_pairs else [],
                 "pair_hypotheses_per_assignment": len(all_pairs),
                 "pairs_tested_per_assignment": len(selected_pairs),
                 "untested_pairs_per_assignment": len(all_pairs) - len(selected_pairs),
                 "pair_enumeration_order": "combinations of the supplied device order",
                 "connected_energized_models_only": self._configuration["connected_only"],
                 "global_status_uniqueness_claimed": False}
        current = self._test_statuses({})
        candidates: dict[str, dict[str, Any]] = {}
        for values in assignments:
            completion = dict(zip(unknown, values))
            trials = [completion]
            trials.extend({**completion, key: 1 - self._statuses[key]} for key in known_pool)
            trials.extend({**completion, first: 1 - self._statuses[first], second: 1 - self._statuses[second]}
                          for first, second in selected_pairs)
            for changes in trials:
                candidate = self._test_statuses(changes)
                candidates[candidate["candidate_id"]] = candidate
        self._assert_fixed_evidence()
        complete = unknown_complete and len(selected_pairs) == len(all_pairs)
        def summarize_candidates():
            current_result = candidates.get(current["candidate_id"], current)
            plausible = [row for row in candidates.values() if row["plausible"]]
            unresolved = [row for row in candidates.values() if row["resolution"] == "unresolved"]
            unique = None
            if current_result["plausible"]:
                decision, reason = "keep_current", "current_model_is_plausible; non-normal healthy configurations are retained"
            elif not unknown_complete:
                decision, reason = "request_status_information", "unknown_status_hypothesis_budget_exceeded"
            elif not complete:
                decision, reason = "continue_candidate_search", "declared_hypothesis_scope_not_fully_tested"
            elif unresolved:
                decision, reason = "request_additional_investigation", "numerical_or_observability_failures_leave_alternatives_unresolved"
            elif len(plausible) == 1:
                decision, reason = "unique_within_declared_scope", "one_plausible_candidate_and_all_declared_alternatives_resolved"
                unique = plausible[0]["candidate_id"]
            elif len(plausible) > 1:
                decision, reason = "ambiguous_candidate_set", "multiple_topologies_pass_absolute_goodness_and_normalized_residual_tests"
            else:
                decision, reason = "request_measurement_or_parameter_investigation", "no_tested_topology_passes_absolute_residual_tests"
            return {"contract": "logical_cb_fixed_evidence_candidate_audit_v1", "decision": decision, "reason": reason,
                    "parent_model_hash": self._model_hash(), "fixed_evidence_hash": self._fixed_evidence_hash,
                    "current": current_result, "hypothesis_scope": scope, "scope_complete": complete,
                    "tested_candidate_count": len(candidates), "unresolved_candidate_count": len(unresolved),
                    "plausible_candidates": list(plausible), "unique_candidate_id": unique,
                    "candidates": list(candidates.values())}

        from .calibration import calibrate_scan
        result = summarize_candidates()
        absolute_unique = result["unique_candidate_id"]
        refinements = []
        guard = calibrate_scan(result, self._inventory, self._sensors, self._observations,
                               alpha=self._configuration["chi2_alpha"])
        if guard["requires_full_fit"]:
            for candidate_id in guard["requires_full_fit"]:
                if candidate_id not in candidates:
                    raise ValueError("Comparison guard requested a model outside the declared scope")
                previous = candidates[candidate_id]
                refined = self._test_statuses(previous["changes"], force_full_fit=True)
                if refined["candidate_id"] != candidate_id:
                    raise ValueError("Candidate model binding changed during full-fit refinement")
                candidates[candidate_id] = refined
                refinements.append(candidate_id)
            result = summarize_candidates()
            # Keep the original proposal visible even when new full fits expose
            # another plausible or unresolved rival. The guard evaluates those
            # rivals; it must never select a different winner itself.
            guard = calibrate_scan({**result, "unique_candidate_id": absolute_unique},
                                   self._inventory, self._sensors, self._observations,
                                   alpha=self._configuration["chi2_alpha"])
        result["absolute_unique_candidate_id"] = absolute_unique
        result["absolute_unique_candidate_id_after_refinement"] = result["unique_candidate_id"]
        result["full_fit_refinement_candidate_ids"] = refinements
        result["comparison_guard"] = guard
        compatible_ids = set(guard["blocking_candidates"])
        if guard.get("candidate_id"):
            compatible_ids.add(guard["candidate_id"])
        else:
            compatible_ids.update(row["candidate_id"] for row in result["plausible_candidates"])
        result["comparison_compatible_candidates"] = [row for row in result["candidates"] if row["candidate_id"] in compatible_ids]
        result["comparison_compatible_set_scope"] = (
            "not_rejected_by_comparison_guard; members_need_not_pass_absolute_fit_tests" if absolute_unique else
            "absolute_plausible_set_only; no_unique_proposal_for_pairwise_comparison")
        if result["unique_candidate_id"] is not None:
            if guard["allowed"]:
                result["reason"] = "unique_absolute_fit_and_conditional_pairwise_separation_guard_passed"
                self._certificate = {"candidate_id": result["unique_candidate_id"], "parent_model_hash": self._model_hash(),
                                     "fixed_evidence_hash": self._fixed_evidence_hash,
                                     "hypothesis_scope": deepcopy(scope), "scope_complete": True,
                                     "plausible_candidate_count": 1, "unresolved_candidate_count": 0,
                                     "comparison_guard": deepcopy(guard), "comparison_guard_hash": evidence_hash(guard)}
            else:
                result["decision"] = "insufficient_calibrated_status_identifiability"
                result["reason"] = guard["decision"]
                result["unique_candidate_id"] = None
        result["certificate"] = deepcopy(self._certificate)
        self._assert_fixed_evidence()
        self._last_scan = deepcopy(result) if self._certificate else None
        return deepcopy(result)

    def apply(self, candidate_id: str) -> dict[str, Any]:
        """Commit only a uniquely supported correction in the complete declared scope."""
        self._assert_fixed_evidence()
        candidate = self._tested.get(candidate_id)
        if candidate is None:
            raise ValueError("Unknown or stale candidate; test it on this runtime snapshot")
        if candidate["parent_model_hash"] != self._model_hash():
            raise ValueError("Candidate parent model changed; retest before applying")
        if candidate["fixed_evidence_hash"] != self._fixed_evidence_hash:
            raise ValueError("Candidate observations or covariance changed")
        if not candidate["changes"] or not candidate["plausible"]:
            raise ValueError("Only a changed, absolutely plausible candidate can be applied")
        certificate = self._certificate
        if certificate is None or certificate["candidate_id"] != candidate_id:
            raise ValueError("A complete uniquely identifying candidate-scan certificate is required")
        if certificate["parent_model_hash"] != self._model_hash() or certificate["fixed_evidence_hash"] != self._fixed_evidence_hash:
            raise ValueError("Candidate-scan certificate is stale")
        guard = certificate.get("comparison_guard") or {}
        if (guard.get("contract") != "logical_topology_pairwise_separation_guard_v2"
            or guard.get("allowed") is not True or guard.get("familywise_alpha") != self._configuration["chi2_alpha"]
            or certificate.get("comparison_guard_hash") != evidence_hash(guard) or self._last_scan is None):
            raise ValueError("A verified comparison-guard certificate is required; basic certificates cannot authorize correction")
        from .calibration import calibrate_scan
        verified = calibrate_scan(self._last_scan, self._inventory, self._sensors, self._observations,
                                  alpha=self._configuration["chi2_alpha"])
        selected = next((row for row in self._last_scan["candidates"] if row["candidate_id"] == candidate_id), None)
        if (verified.get("allowed") is not True or evidence_hash(verified) != certificate["comparison_guard_hash"]
            or selected is None or selected["statuses"] != candidate["statuses"]):
            raise ValueError("Comparison-guard evidence no longer verifies this correction")
        self._case = self._case_for(candidate["statuses"])
        self._statuses = deepcopy(candidate["statuses"])
        self._certificate = None
        self._last_scan = None
        self._assert_fixed_evidence()
        return {**self.snapshot(), "applied_candidate_id": candidate_id,
                "applied_changes": deepcopy(candidate["changes"]), "identification_certificate": deepcopy(certificate)}
