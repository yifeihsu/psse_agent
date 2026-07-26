"""Deployment WLS/context/correction providers backed by the MATPOWER runtime.

These adapters replace the deterministic pilot stand-ins with the same pure
Python estimation stack the production MCP server uses
(``mcp_server.matpower_server``): Lagrangian WLS with normalized residuals and
branch Lagrange multipliers, grouped measurement correction, and multi-scan
parameter correction.  Observations therefore carry the exact decision
features the deployed agent sees (top residuals, top multipliers, global
chi-square evidence), summarized with the same ``trace_protocol`` helpers used
by the production SFT corpus.

Physical state convention: ``state["case"]`` is a MATPOWER case path (or a
mapping with a ``case_path`` key) resolvable by the runtime, and
``state["measurements"]`` is the full measurement vector ordered
``[Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)]``.

Case-mutating corrections (parameters, topology) write a content-addressed
derived ``.m`` case under ``derived_case_dir`` and return it as the candidate
case, because a path-valued case cannot be patched in place.  Multi-scan
parameter correction reads repeated observed scans from state metadata and
derives each numerical initial state from measured voltage magnitudes plus the
configured case angles.  It never accepts truth-derived state initializers.
When scans are absent the executor fails closed, which the environment records
as a collectable no-op learner state.
"""

from __future__ import annotations

import cmath
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from typing import Any, Mapping, Sequence

from hif_search_limits import validate_hif_search_limits
from psse_env.actions import (
    ANOMALY_FAMILY_MARKERS,
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    GET_HARMONIC_CONTEXT,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    unexplained_signatures,
)
from psse_env.state_store import apply_modification

from mcp_server.matpower_server import (  # noqa: E402  (repo-root package)
    _estimate_hif_location_magnitude_logic,
    _estimate_hif_location_magnitude_multiscan_logic,
    _infer_harmonic_orders,
    _load_python_case,
    _meas_correction_json,
    _param_correction_json,
    _run_hse_logic,
    _run_three_phase_nlm_logic,
    _wls_json,
)
from trace_protocol import (  # noqa: E402  (repo-root module)
    build_lambda_evidence,
    build_residual_evidence,
    chi2_threshold,
    summarize_harmonic_context_payload,
    summarize_hif_parameter_estimate_payload,
    summarize_hse_payload,
    summarize_measurement_correction_payload,
    summarize_parameter_correction_payload,
    summarize_three_phase_nlm_payload,
    summarize_wls_payload,
)


# When the global chi-square score is only marginally above its threshold,
# independent singleton estimates can leave enough coupled energy for a
# healthy channel to become the largest residual.  A single, bounded
# re-estimation of already accepted targets is safer than expanding the repair
# set in that ambiguity band.  Eight remaining actions preserve room for a
# rejected refinement plus one normal correction/verification transaction.
_COUPLED_REFINEMENT_MAX_ANOMALY_RATIO = 1.10
_COUPLED_REFINEMENT_MIN_REMAINING_BUDGET = 8
_ROUTE_ACTIONABLE = "actionable"
_ROUTE_COMPLETE_NEGATIVE = "complete_negative"
_ROUTE_UNAVAILABLE = "unavailable_or_inconclusive"


def measurement_index_map(nb: int, nl: int) -> dict[str, slice]:
    """Channel layout of the full measurement vector."""
    return {
        "Vm": slice(0, nb),
        "Pinj": slice(nb, 2 * nb),
        "Qinj": slice(2 * nb, 3 * nb),
        "Pf": slice(3 * nb, 3 * nb + nl),
        "Qf": slice(3 * nb + nl, 3 * nb + 2 * nl),
        "Pt": slice(3 * nb + 2 * nl, 3 * nb + 3 * nl),
        "Qt": slice(3 * nb + 3 * nl, 3 * nb + 4 * nl),
    }


def observable_parameter_initial_states(
    ppc: Mapping[str, Any], z_scans: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Build multi-scan solver starts from deployment-observable inputs only."""

    import numpy as np

    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    nb = int(bus.shape[0])
    expected_measurements = 3 * nb + 4 * int(branch.shape[0])
    reference_rows = np.flatnonzero(bus[:, 1].astype(int) == 3)
    reference = int(reference_rows[0]) if reference_rows.size else 0
    configured_angles = bus[:, 8].astype(float)
    configured_angles = configured_angles - configured_angles[reference]
    starts: list[list[float]] = []
    for index, raw_scan in enumerate(z_scans):
        scan = np.asarray(raw_scan, dtype=float).reshape(-1)
        if scan.size != expected_measurements or not np.all(np.isfinite(scan)):
            raise ValueError(
                f"parameter scan {index} must contain {expected_measurements} finite values"
            )
        observed_vm = scan[:nb]
        if np.any(observed_vm <= 0.0):
            raise ValueError(f"parameter scan {index} has non-positive voltage magnitude")
        starts.append(
            np.concatenate((observed_vm, configured_angles)).astype(float).tolist()
        )
    if not starts:
        raise ValueError("parameter correction requires at least one observed scan")
    return starts


def matpower_case_differ(parent_case_path: str, candidate_case_path: str) -> dict[str, Any]:
    """Structural diff between two case files for collateral-damage audits."""
    parent = _load_python_case(parent_case_path)
    candidate = _load_python_case(candidate_case_path)
    if (
        parent["bus"].shape != candidate["bus"].shape
        or parent["gen"].shape != candidate["gen"].shape
        or parent["branch"].shape != candidate["branch"].shape
    ):
        return {"comparable": False}
    changed_branch_rows: dict[int, list[int]] = {}
    for row in range(parent["branch"].shape[0]):
        columns = [
            column
            for column in range(parent["branch"].shape[1])
            if float(parent["branch"][row][column]) != float(candidate["branch"][row][column])
        ]
        if columns:
            changed_branch_rows[row] = columns
    return {
        "comparable": True,
        "base_mva_changed": float(parent["baseMVA"]) != float(candidate["baseMVA"]),
        "bus_changed": bool((parent["bus"] != candidate["bus"]).any()),
        "gen_changed": bool((parent["gen"] != candidate["gen"]).any()),
        "changed_branch_rows": changed_branch_rows,
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _matches_any_marker(text: str, markers: Sequence[str]) -> bool:
    """Word-boundary marker matching, consistent with expert routing."""
    lowered = text.lower()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(marker.lower())}(?![a-z0-9])", lowered)
        for marker in markers
    )


def _three_phase_vuf_evidence(
    three_phase_voltages: Any,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Compute observable negative/positive-sequence voltage ratios."""
    if not isinstance(three_phase_voltages, Sequence) or isinstance(
        three_phase_voltages, (str, bytes)
    ):
        return []
    rotation = cmath.exp(2j * math.pi / 3.0)
    rows: list[dict[str, Any]] = []
    for item in three_phase_voltages:
        if not isinstance(item, Mapping):
            continue
        magnitudes = item.get("vln_pu")
        angles = item.get("ang_deg")
        if (
            not isinstance(magnitudes, Sequence)
            or isinstance(magnitudes, (str, bytes))
            or not isinstance(angles, Sequence)
            or isinstance(angles, (str, bytes))
            or len(magnitudes) < 3
            or len(angles) < 3
        ):
            continue
        try:
            phases = [
                cmath.rect(float(magnitudes[index]), math.radians(float(angles[index])))
                for index in range(3)
            ]
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in phases):
            continue
        va, vb, vc = phases
        positive = (va + rotation * vb + (rotation**2) * vc) / 3.0
        negative = (va + (rotation**2) * vb + rotation * vc) / 3.0
        if abs(positive) <= 1e-12:
            continue
        bus = item.get("bus")
        if isinstance(bus, str):
            match = re.search(r"\d+", bus)
            bus = int(match.group()) if match else bus
        rows.append({"bus": bus, "vuf": float(abs(negative) / abs(positive))})
    rows.sort(key=lambda row: float(row["vuf"]), reverse=True)
    return rows[: max(int(top_k), 0)]


def _render_matpower_case(ppc: Mapping[str, Any], function_name: str) -> str:
    """Render the parsed matrices back to loadable MATPOWER text."""
    lines = [
        f"function mpc = {function_name}",
        "mpc.version = '2';",
        f"mpc.baseMVA = {float(ppc['baseMVA'])};",
    ]
    for name in ("bus", "gen", "branch"):
        lines.append(f"mpc.{name} = [")
        for row in ppc[name]:
            lines.append("\t" + "\t".join(repr(float(value)) for value in row) + ";")
        lines.append("];")
    return "\n".join(lines) + "\n"


class MatpowerDeploymentProviders:
    """Deployment provider bundle for ``TransactionalPSSEEnv``.

    Instances hold only immutable configuration, so bound methods remain
    deepcopy-safe branch collaborators for counterfactual clones.
    """

    provider_kind = "deployment"

    def __init__(
        self,
        *,
        top_k: int = 5,
        residual_threshold: float = 3.0,
        lambda_threshold: float = 3.0,
        chi2_alpha: float = 0.05,
        derived_case_dir: str | None = None,
        max_correction_iterations: int = 2,
        error_tolerance: float = 1e-3,
        hif_alpha_grid_size: int = 31,
        hif_r_grid_size: int = 35,
        hif_max_scans: int = 10,
        harmonic_thd_threshold_percent: float = 1.0,
        unbalance_vuf_threshold: float = 0.02,
        hif_min_residual_reduction: float = 0.20,
        hif_max_weighted_residual_norm: float = 3.0,
        vm_bound_tolerance_pu: float = 0.005,
        branch_rate_tolerance_mva: float = 1e-6,
    ) -> None:
        self.top_k = int(top_k)
        self.residual_threshold = float(residual_threshold)
        self.lambda_threshold = float(lambda_threshold)
        self.chi2_alpha = float(chi2_alpha)
        self.derived_case_dir = str(
            derived_case_dir
            or os.path.join(tempfile.gettempdir(), "psse_derived_cases")
        )
        self.max_correction_iterations = int(max_correction_iterations)
        self.error_tolerance = float(error_tolerance)
        # HIF grid-search resolution.  The 31x35 default matches the
        # production estimator; round-0 collection may configure a coarser
        # grid so the real OpenDSS search stays tractable per episode.
        (
            self.hif_alpha_grid_size,
            self.hif_r_grid_size,
            validated_hif_max_scans,
        ) = validate_hif_search_limits(
            alpha_grid_size=hif_alpha_grid_size,
            r_grid_size=hif_r_grid_size,
            max_scans=hif_max_scans,
        )
        assert validated_hif_max_scans is not None
        self.hif_max_scans = validated_hif_max_scans
        self.harmonic_thd_threshold_percent = float(harmonic_thd_threshold_percent)
        self.unbalance_vuf_threshold = float(unbalance_vuf_threshold)
        self.hif_min_residual_reduction = float(hif_min_residual_reduction)
        self.hif_max_weighted_residual_norm = float(hif_max_weighted_residual_norm)
        self.vm_bound_tolerance_pu = float(vm_bound_tolerance_pu)
        self.branch_rate_tolerance_mva = float(branch_rate_tolerance_mva)
        if self.vm_bound_tolerance_pu < 0.0 or self.branch_rate_tolerance_mva < 0.0:
            raise ValueError("Physical-bound tolerances must be non-negative.")

    # ------------------------------------------------------------------ wiring

    def env_kwargs(self) -> dict[str, Any]:
        """Keyword arguments wiring this bundle into ``TransactionalPSSEEnv``."""
        from psse_env.oracle import ProcessValidityOracle

        return {
            "process_oracle": ProcessValidityOracle(executor_hydrated_corrections=True),
            "candidate_quality_oracle": self._deployment_candidate_quality_oracle(),
            "wls_runner": self.run_wls,
            "context_providers": {
                GET_MEASUREMENT_CONTEXT: self.get_measurement_context,
                GET_PARAMETER_CONTEXT: self.get_parameter_context,
                GET_TOPOLOGY_CONTEXT: self.get_topology_context,
            },
            "correction_executors": {
                CORRECT_MEASUREMENTS: self.correct_measurements,
                CORRECT_PARAMETERS: self.correct_parameters,
                CORRECT_TOPOLOGY: self.correct_topology,
            },
            "evidence_providers": {
                ASK_FOR_MORE_EVIDENCE: self.request_additional_evidence,
                GET_HARMONIC_CONTEXT: self.get_harmonic_context,
                RUN_HSE_FROM_PATH: self.run_hse,
                RUN_THREE_PHASE_NLM_FROM_PATH: self.run_three_phase_nlm,
                ESTIMATE_HIF_FROM_PATH: self.estimate_hif,
                ESTIMATE_HIF_MULTISCAN_FROM_PATH: self.estimate_hif_multiscan,
            },
        }

    @staticmethod
    def _deployment_candidate_quality_oracle() -> Any:
        """Build the one deployment verdict policy used by screen and commit."""

        from psse_env.oracle import CandidateQualityOracle

        return CandidateQualityOracle(
            mode="deployment", case_differ=matpower_case_differ
        )

    def request_additional_evidence(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Report exhaustion of the configured HIF diagnostic inventory.

        This provider does not declare the HIF absent or resolved.  It only
        reports that the controller has already invoked every configured HIF
        diagnostic for the current observable channel inventory.  The
        transactional environment separately checks the full bound history and
        the rejected acceptance tests before this report can end an episode.
        """
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        request = state.get("evidence_request")
        attempted = {
            str(signature).split(":", 1)[0]
            for signature in observation.get("tried_action_signatures") or []
        }
        if request in {
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        }:
            investigation_tools = {
                GET_MEASUREMENT_CONTEXT,
                GET_PARAMETER_CONTEXT,
                GET_TOPOLOGY_CONTEXT,
                CORRECT_MEASUREMENTS,
                CORRECT_PARAMETERS,
                CORRECT_TOPOLOGY,
            }
            score = observation.get("remaining_anomaly_score")
            try:
                score_unresolved = score is not None and float(score) >= 1.0
            except (TypeError, ValueError):
                score_unresolved = False
            if not (
                (observation.get("unresolved_signatures") or score_unresolved)
                and "run_wls" in attempted
                and bool(attempted & investigation_tools)
            ):
                return self._failure(
                    "recovery_evidence_inventory_incomplete",
                    "observable WLS plus investigation history is required",
                )
            available = {
                str(item) for item in observation.get("available_evidence") or []
            }
            if request == RECOVERY_BUDGET_EXHAUSTED_REQUEST:
                try:
                    remaining_budget = int(observation.get("remaining_budget") or 0)
                except (TypeError, ValueError):
                    remaining_budget = 0
                if not 0 < remaining_budget < 4:
                    return self._failure(
                        "recovery_budget_not_exhausted",
                        f"remaining_budget={remaining_budget}",
                    )
                return {
                    **self._binding(state),
                    "evidence_source": "deployment_diagnostic:recovery_budget_inventory",
                    "request": RECOVERY_BUDGET_EXHAUSTED_REQUEST,
                    "family": "recovery_budget",
                    "additional_evidence_available": True,
                    "autonomous_budget_available": False,
                    "operator_review_required": True,
                    "remaining_budget": remaining_budget,
                    "attempted_tools": sorted(attempted),
                    "available_evidence_channels": sorted(available),
                }
            return {
                **self._binding(state),
                "evidence_source": "deployment_diagnostic:recovery_evidence_inventory",
                "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                "family": "mixed_or_unresolved",
                "additional_evidence_available": False,
                "operator_review_required": True,
                "attempted_tools": sorted(attempted),
                "available_evidence_channels": sorted(available),
            }
        if request != HIF_DIAGNOSTICS_EXHAUSTED_REQUEST:
            return self._failure(
                "operator_escalation_request_unsupported", request or "missing_request"
            )
        unresolved = unexplained_signatures(
            observation.get("unresolved_signatures") or [],
            observation.get("explained_anomalies") or [],
        )
        if not self._has_family_signature(
            {"policy_observation": {"unresolved_signatures": unresolved}}, "hif"
        ):
            return self._failure(
                "operator_escalation_not_supported",
                "no unexplained observable HIF signature remains",
            )

        available = {str(item) for item in observation.get("available_evidence") or []}
        required = {RUN_THREE_PHASE_NLM_FROM_PATH, ESTIMATE_HIF_FROM_PATH}
        if "hif_scan_window" in available:
            required.add(ESTIMATE_HIF_MULTISCAN_FROM_PATH)
        missing = sorted(required - attempted)
        if missing:
            return self._failure(
                "hif_diagnostic_ladder_incomplete",
                ",".join(missing),
                required_diagnostics=sorted(required),
                attempted_diagnostics=sorted(required & attempted),
            )
        return {
            **self._binding(state),
            "evidence_source": "deployment_diagnostic:hif_evidence_inventory",
            "request": HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
            "family": "hif",
            "additional_evidence_available": False,
            "operator_review_required": True,
            "required_diagnostics": sorted(required),
            "attempted_diagnostics": sorted(required & attempted),
            "available_evidence_channels": sorted(available),
        }

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _case_path(state: Mapping[str, Any]) -> str:
        case = state.get("case")
        if isinstance(case, Mapping):
            case = case.get("case_path")
        if not isinstance(case, str) or not case:
            raise ValueError(
                "MatpowerDeploymentProviders requires state['case'] to be a case path."
            )
        return case

    @staticmethod
    def _measurements(state: Mapping[str, Any]) -> list[float]:
        measurements = state.get("measurements")
        if not isinstance(measurements, Sequence) or isinstance(measurements, (str, bytes)):
            raise ValueError("state['measurements'] must be the full measurement vector.")
        return [float(value) for value in measurements]

    @staticmethod
    def _binding(state: Mapping[str, Any]) -> dict[str, Any]:
        binding: dict[str, Any] = {}
        if state.get("state_id") is not None:
            binding["state_id"] = str(state["state_id"])
        if state.get("state_hash") is not None:
            binding["state_hash"] = str(state["state_hash"])
        return binding

    def _solve(self, state: Mapping[str, Any]) -> dict[str, Any]:
        case_path = self._case_path(state)
        z = self._measurements(state)
        ppc = _load_python_case(case_path)
        nb = int(ppc["bus"].shape[0])
        nl = int(ppc["branch"].shape[0])
        payload = _wls_json(case_path, z)
        return {
            "case_path": case_path,
            "z": z,
            "ppc": ppc,
            "nb": nb,
            "nl": nl,
            "index_map": measurement_index_map(nb, nl),
            "payload": payload,
        }

    @staticmethod
    def _failure(error_code: str, error_detail: Any = None, **metrics: Any) -> dict[str, Any]:
        result = {"execution_status": "failure", "error_code": error_code, **metrics}
        if error_detail is not None:
            result["error_detail"] = str(error_detail)
        return result

    def _derived_case(self, ppc: Mapping[str, Any], tag: str) -> str:
        text = _render_matpower_case(ppc, f"derived_{tag}")
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        os.makedirs(self.derived_case_dir, exist_ok=True)
        path = os.path.join(self.derived_case_dir, f"{tag}_{digest}.m")
        try:
            with open(path, "rb") as handle:
                if handle.read() == encoded:
                    return path
        except FileNotFoundError:
            pass

        # A content-addressed path must never trust stale bytes from a prior
        # process.  Write and fsync a sibling file before an atomic replace so
        # concurrent evaluators either observe the old complete file or these
        # exact rendered bytes, never a partial case.
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{tag}_{digest}.", suffix=".tmp", dir=self.derived_case_dir
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return path

    @staticmethod
    def _branch_row0(arguments: Mapping[str, Any], nl: int) -> int:
        if arguments.get("branch_row0") is not None:
            row0 = int(arguments["branch_row0"])
        elif arguments.get("line_index1") is not None:
            row0 = int(arguments["line_index1"]) - 1
        elif arguments.get("line_index") is not None:
            row0 = int(arguments["line_index"]) - 1
        else:
            raise ValueError("Correction requires line_index, line_index1, or branch_row0.")
        if not 0 <= row0 < nl:
            raise ValueError(f"Branch row {row0} outside valid range [0, {nl - 1}].")
        return row0

    # --------------------------------------------------------------- WLS runner

    def _target_evidence(
        self,
        source_action: Mapping[str, Any],
        residuals: Sequence[float],
        lambda_values: Sequence[float],
        nl: int,
        *,
        candidate_case: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Observable target-progress evidence for a candidate verification.

        Derived solely from the candidate solve: a measurement target is fixed
        when every corrected index sits below the residual threshold; a
        parameter target is fixed when its Lagrange multipliers sit below the
        multiplier threshold; and a topology target is fixed when the
        candidate case structurally carries the exact requested branch status.
        A remaining independent meter error can keep the corrected branch's
        multiplier elevated, so using that multiplier as the topology target
        test would incorrectly roll back a structurally verified outage repair.
        ``remaining_suspect_count`` counts non-target residual and
        branch-multiplier threshold violations. It is observable diagnostic
        evidence, not an estimate of physical-error cardinality.
        """
        tool = str(source_action.get("tool") or "")
        arguments = source_action.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        target_measurements: set[int] = set()
        target_rows: set[int] = set()
        if tool == CORRECT_MEASUREMENTS:
            group = arguments.get("suspect_group")
            updates = arguments.get("measurement_updates")
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes)):
                target_measurements = {int(index) for index in group}
            elif isinstance(updates, Mapping):
                target_measurements = {int(index) for index in updates}
        elif tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            try:
                target_rows = {self._branch_row0(arguments, nl)}
            except ValueError:
                return None
        else:
            return None
        if not target_measurements and not target_rows:
            return None

        def row_lambda(row0: int) -> float:
            values = [
                abs(float(lambda_values[index]))
                for index in (2 * row0, 2 * row0 + 1)
                if 0 <= index < len(lambda_values)
            ]
            return max(values, default=0.0)

        topology_status_matches: bool | None = None
        if tool == CORRECT_TOPOLOGY:
            requested_status = arguments.get("status", arguments.get("expected_status"))
            if arguments.get("desired_status") is not None and requested_status is None:
                requested_status = int(bool(arguments["desired_status"]))
            try:
                requested_status = int(requested_status)
                row0 = next(iter(target_rows))
                branch = candidate_case["branch"] if candidate_case is not None else None
                raw_candidate_status = float(branch[row0][10])
                candidate_status = int(raw_candidate_status)
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                OverflowError,
                StopIteration,
            ):
                return None
            if (
                requested_status not in {0, 1}
                or not math.isfinite(raw_candidate_status)
                or raw_candidate_status != float(candidate_status)
                or candidate_status not in {0, 1}
            ):
                return None
            topology_status_matches = candidate_status == requested_status

        if target_measurements:
            target_values = [
                abs(float(residuals[index]))
                for index in target_measurements
                if 0 <= index < len(residuals)
            ]
            target_metric_kind = "max_abs_normalized_residual"
            target_metric_threshold = self.residual_threshold
        elif tool == CORRECT_TOPOLOGY:
            target_values = [0.0 if topology_status_matches else 1.0]
            target_metric_kind = "branch_status_mismatch"
            target_metric_threshold = 0.5
        else:
            target_values = [row_lambda(row0) for row0 in target_rows]
            target_metric_kind = "max_abs_branch_multiplier"
            target_metric_threshold = self.lambda_threshold
        if not target_values:
            return None
        target_metric_value = max(target_values)
        target_fixed = target_metric_value < target_metric_threshold
        remaining = sum(
            1
            for index, value in enumerate(residuals)
            if index not in target_measurements and abs(float(value)) >= self.residual_threshold
        ) + sum(
            1
            for row0 in range(nl)
            if row0 not in target_rows and row_lambda(row0) >= self.lambda_threshold
        )
        evidence = {
            "target_fixed": bool(target_fixed),
            "target_progress": 1.0 if target_fixed else 0.0,
            "target_metric_kind": target_metric_kind,
            "target_metric_value": float(target_metric_value),
            "target_metric_threshold": float(target_metric_threshold),
            "remaining_suspect_count": int(remaining),
        }
        if tool == CORRECT_TOPOLOGY:
            topology_multiplier = max(
                (row_lambda(row0) for row0 in target_rows), default=math.inf
            )
            if not math.isfinite(topology_multiplier):
                return None
            evidence.update(
                {
                    # Structural equality proves that the requested mutation
                    # landed on the intended row.  The residual branch
                    # multiplier remains a separate ambiguity signal used by
                    # the deployment quality gate; it is not the topology
                    # target-locality predicate itself.
                    "topology_target_branch_multiplier": float(topology_multiplier),
                    "topology_target_branch_multiplier_threshold": float(
                        self.lambda_threshold
                    ),
                    "topology_target_branch_multiplier_cleared": bool(
                        topology_multiplier < self.lambda_threshold
                    ),
                    "topology_target_status_matches_requested": bool(
                        topology_status_matches
                    ),
                }
            )
        return evidence

    def _steady_state_physical_evidence(
        self, solved: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Check observable snapshot constraints without claiming a power-flow solve.

        The check uses only the current MATPOWER case and the same measured
        ``Vm/Pf/Qf/Pt/Qt`` channels supplied to WLS.  It deliberately has a
        narrow scope: active-network connectivity, bus-voltage limits, and
        active-branch ``RATE_A`` limits.  A zero/non-positive ``RATE_A`` has the
        standard MATPOWER meaning of no applicable limit and is reported as
        unrated rather than silently treated as a passing rated branch.
        """
        import numpy as np

        scope = "observed_snapshot_topology_vm_rate_a"
        violations: list[dict[str, Any]] = []
        input_errors: list[str] = []
        topology: dict[str, Any] = {"checked": False}
        voltage: dict[str, Any] = {
            "checked": False,
            "tolerance_pu": self.vm_bound_tolerance_pu,
        }
        thermal: dict[str, Any] = {
            "checked": False,
            "tolerance_mva": self.branch_rate_tolerance_mva,
            "telemetry_units": "per_unit_on_case_base_mva",
        }

        try:
            ppc = solved["ppc"]
            bus = np.asarray(ppc["bus"], dtype=float)
            branch = np.asarray(ppc["branch"], dtype=float)
            z = np.asarray(solved["z"], dtype=float)
            nb = int(solved["nb"])
            nl = int(solved["nl"])
            index_map = solved["index_map"]
            base_mva = float(ppc["baseMVA"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            input_errors.append(f"physical_evidence_input_invalid:{type(exc).__name__}")
            bus = np.empty((0, 0), dtype=float)
            branch = np.empty((0, 0), dtype=float)
            z = np.asarray([], dtype=float)
            nb = nl = 0
            index_map = {}
            base_mva = math.nan

        if bus.ndim != 2 or bus.shape[0] != nb or bus.shape[1] < 13:
            input_errors.append("matpower_bus_schema_invalid")
        if branch.ndim != 2 or branch.shape[0] != nl or branch.shape[1] < 11:
            input_errors.append("matpower_branch_schema_invalid")
        if not math.isfinite(base_mva) or base_mva <= 0.0:
            input_errors.append("matpower_base_mva_invalid")
        expected_measurements = 3 * nb + 4 * nl
        if z.ndim != 1 or len(z) != expected_measurements or not np.isfinite(z).all():
            input_errors.append("measurement_telemetry_invalid")
        if not isinstance(index_map, Mapping) or not all(
            key in index_map for key in ("Vm", "Pf", "Qf", "Pt", "Qt")
        ):
            input_errors.append("measurement_index_map_invalid")

        schema_ok = not input_errors
        in_service_rows: list[int] = []
        bus_ids: list[int] = []
        if schema_ok:
            raw_ids = bus[:, 0]
            if not np.isfinite(raw_ids).all() or not np.allclose(raw_ids, np.rint(raw_ids)):
                input_errors.append("matpower_bus_ids_invalid")
            else:
                # MATPOWER BUS_TYPE=4 is an explicitly isolated bus and is not
                # part of the energized-network connectivity/limit scope.
                in_service_rows = [
                    row for row in range(nb) if int(round(float(bus[row, 1]))) != 4
                ]
                bus_ids = [int(round(float(bus[row, 0]))) for row in in_service_rows]
                if not bus_ids or len(set(bus_ids)) != len(bus_ids):
                    input_errors.append("matpower_in_service_bus_set_invalid")

        active_rows: list[int] = []
        components: list[list[int]] = []
        if not input_errors:
            bus_set = set(bus_ids)
            adjacency = {bus_id: set() for bus_id in bus_ids}
            for row in range(nl):
                status = float(branch[row, 10])
                if not math.isfinite(status):
                    input_errors.append(f"branch_status_invalid:row0={row}")
                    continue
                if status == 0.0:
                    continue
                active_rows.append(row)
                raw_from = float(branch[row, 0])
                raw_to = float(branch[row, 1])
                if not (
                    math.isfinite(raw_from)
                    and math.isfinite(raw_to)
                    and raw_from.is_integer()
                    and raw_to.is_integer()
                ):
                    input_errors.append(f"active_branch_endpoint_invalid:row0={row}")
                    continue
                from_bus = int(raw_from)
                to_bus = int(raw_to)
                if from_bus not in bus_set or to_bus not in bus_set:
                    input_errors.append(f"active_branch_endpoint_invalid:row0={row}")
                    continue
                adjacency[from_bus].add(to_bus)
                adjacency[to_bus].add(from_bus)

            unseen = set(bus_ids)
            while unseen:
                root = min(unseen)
                stack = [root]
                component: set[int] = set()
                while stack:
                    current = stack.pop()
                    if current in component:
                        continue
                    component.add(current)
                    stack.extend(adjacency[current] - component)
                unseen -= component
                components.append(sorted(component))
            connected = len(components) == 1
            topology = {
                "checked": True,
                "bus_scope": "matpower_bus_type_not_4",
                "in_service_bus_count": len(bus_ids),
                "active_branch_count": len(active_rows),
                "connected": connected,
                "component_count": len(components),
                "components": components,
            }
            if not connected:
                violations.append(
                    {
                        "type": "topology_disconnected",
                        "component_count": len(components),
                        "components": components,
                    }
                )

        if not input_errors:
            vm = z[index_map["Vm"]]
            vm_violations: list[dict[str, Any]] = []
            for row in in_service_rows:
                observed = float(vm[row])
                vmax = float(bus[row, 11])
                vmin = float(bus[row, 12])
                bus_id = int(round(float(bus[row, 0])))
                if (
                    not all(math.isfinite(value) for value in (observed, vmin, vmax))
                    or vmin <= 0.0
                    or vmax < vmin
                ):
                    input_errors.append(f"bus_voltage_limit_invalid:bus={bus_id}")
                    continue
                if (
                    observed < vmin - self.vm_bound_tolerance_pu
                    or observed > vmax + self.vm_bound_tolerance_pu
                ):
                    item = {
                        "type": "bus_voltage_out_of_bounds",
                        "bus": bus_id,
                        "measurement_index0": int(index_map["Vm"].start) + row,
                        "observed_vm_pu": observed,
                        "vmin_pu": vmin,
                        "vmax_pu": vmax,
                    }
                    vm_violations.append(item)
                    violations.append(item)
            voltage = {
                "checked": True,
                "checked_bus_count": len(in_service_rows),
                "within_bounds": not vm_violations,
                "tolerance_pu": self.vm_bound_tolerance_pu,
                "violation_count": len(vm_violations),
            }

        if not input_errors:
            pf = z[index_map["Pf"]]
            qf = z[index_map["Qf"]]
            pt = z[index_map["Pt"]]
            qt = z[index_map["Qt"]]
            rated = 0
            unrated = 0
            thermal_violations: list[dict[str, Any]] = []
            for row in active_rows:
                rate_a = float(branch[row, 5])
                if not math.isfinite(rate_a):
                    input_errors.append(f"branch_rate_a_invalid:row0={row}")
                    continue
                if rate_a <= 0.0:
                    unrated += 1
                    continue
                rated += 1
                from_mva = math.hypot(float(pf[row]), float(qf[row])) * base_mva
                to_mva = math.hypot(float(pt[row]), float(qt[row])) * base_mva
                observed_mva = max(from_mva, to_mva)
                if observed_mva > rate_a + self.branch_rate_tolerance_mva:
                    item = {
                        "type": "active_branch_rate_a_exceeded",
                        "branch_row0": row,
                        "from_bus": int(round(float(branch[row, 0]))),
                        "to_bus": int(round(float(branch[row, 1]))),
                        "from_mva": from_mva,
                        "to_mva": to_mva,
                        "rate_a_mva": rate_a,
                    }
                    thermal_violations.append(item)
                    violations.append(item)
            thermal = {
                "checked": True,
                "active_branch_count": len(active_rows),
                "rated_branch_count": rated,
                "unrated_branch_count": unrated,
                "within_defined_rate_a_bounds": not thermal_violations,
                "tolerance_mva": self.branch_rate_tolerance_mva,
                "telemetry_units": "per_unit_on_case_base_mva",
                "base_mva": base_mva,
                "violation_count": len(thermal_violations),
            }

        complete = not input_errors and all(
            bool(check.get("checked")) for check in (topology, voltage, thermal)
        )
        # Missing/malformed observable inputs are inconclusive, not affirmative
        # evidence of a physical violation.  Downstream acceptance still fails
        # closed because only literal ``True`` is sufficient physical evidence.
        physical_ok: bool | None = None if not complete else not violations
        evidence = {
            "scope": scope,
            "method": "matpower_case_limits_with_observed_wls_telemetry",
            "complete": complete,
            "topology_connectivity": topology,
            "bus_voltage_bounds": voltage,
            "active_branch_rate_a_bounds": thermal,
            "violation_count": len(violations),
            "input_errors": input_errors,
        }
        return {
            "physical_constraints_ok": physical_ok,
            "physical_evidence_scope": scope,
            "physical_evidence_complete": complete,
            "physical_bound_violations": violations,
            "steady_state_physical_evidence": evidence,
        }

    def run_wls(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("wls_input_error", f"{type(exc).__name__}: {exc}")
        payload = solved["payload"]
        if not payload.get("success"):
            return self._failure(
                "wls_failure",
                payload.get("error", "solver_failure"),
                evidence_source="deployment_wls:lagrangian_port",
            )
        residuals = [float(value) for value in payload.get("r") or []]
        nb, nl = solved["nb"], solved["nl"]
        state_count = 2 * nb - 1
        dof = max(1, len(residuals) - state_count)
        statistic = float(payload.get("global_residual_sum") or 0.0)
        threshold = float(chi2_threshold(dof, self.chi2_alpha))
        summary = summarize_wls_payload(
            payload,
            {"nb": nb, "branch_info": payload.get("branch_info") or []},
            solved["index_map"],
        )
        max_abs_residual = max((abs(value) for value in residuals), default=0.0)
        # Observable anomaly signatures drive expert routing.  Signatures this
        # runner derives from its own solve carry the ``wls_`` prefix and are
        # refreshed on every solve; signatures recorded from other sources
        # (power-quality or waveform sensors) are preserved verbatim because
        # the fundamental-frequency solve has no authority to withdraw them.
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        preserved = [
            str(signature)
            for signature in observation.get("unresolved_signatures") or []
            if not str(signature).startswith("wls_")
        ]
        signatures = list(preserved)
        # While an unexplained waveform-level anomaly (harmonic distortion or
        # a suspected HIF) stands, the fundamental-frequency solve's bad-data
        # attributions are physically unreliable: the chi-square elevation is
        # (at least partly) the waveform event itself, and "correcting" SCADA
        # measurements against it would mask the true anomaly.  The solve
        # still reports its metrics but mints no signatures until the
        # specialized diagnostics have explained the sensor-sourced ones.
        waveform_markers = (
            ANOMALY_FAMILY_MARKERS["harmonic"]
            + ANOMALY_FAMILY_MARKERS["three_phase_unbalance"]
            + ANOMALY_FAMILY_MARKERS["hif"]
        )
        unexplained_sensor = [
            signature
            for signature in unexplained_signatures(
                preserved, observation.get("explained_anomalies") or []
            )
            if _matches_any_marker(str(signature), waveform_markers)
        ]
        if statistic >= threshold and not unexplained_sensor:
            lambda_values = [float(value) for value in payload.get("lambdaN") or []]
            max_abs_lambda = max((abs(value) for value in lambda_values), default=0.0)
            # Classical Lagrangian discrimination: a gross measurement error
            # drives the largest normalized residual well above the largest
            # normalized branch multiplier; a branch (parameter/topology)
            # error inverts that.  Dominance requires clear separation in the
            # claimed direction — inside the symmetric dead band neither tag
            # carries the ``dominant`` token, so no family is suppressed and
            # routing falls back to static source priority.
            measurement_dominant = max_abs_residual > 1.2 * max_abs_lambda
            branch_dominant = max_abs_lambda > 1.2 * max_abs_residual
            residual_tag = (
                "wls_residual_outlier_dominant" if measurement_dominant else "wls_residual_outlier"
            )
            branch_tag = (
                "wls_branch_multiplier_dominant line_status_or_parameter"
                if branch_dominant
                else "wls_branch_multiplier line_status_or_parameter"
            )
            for item in build_residual_evidence(
                residuals, solved["index_map"], k=self.top_k, min_abs=self.residual_threshold
            ):
                signatures.append(
                    f"{residual_tag} index={item['index0']} channel={item['channel']}"
                )
            for item in build_lambda_evidence(
                lambda_values,
                payload.get("branch_info") or [],
                k=self.top_k,
                min_abs=self.lambda_threshold,
            ):
                if item.get("line_row0") is not None:
                    signatures.append(f"{branch_tag} line={int(item['line_row0']) + 1}")
        metrics: dict[str, Any] = {
            **self._binding(state),
            "evidence_source": "deployment_wls:lagrangian_port",
            # This is convergence of the state-estimation solve only.  It is
            # not evidence that voltage, thermal, or topology constraints were
            # checked; those remain unknown unless a separate verifier emits
            # narrowly scoped physical evidence.
            "state_estimation_converged": True,
            "converged": True,
            "wls_objective": statistic,
            "chi_square_statistic": statistic,
            "chi_square_threshold": threshold,
            "chi_square_dof": dof,
            "max_normalized_residual": max_abs_residual,
            "anomaly_threshold": 1.0,
            "remaining_anomaly_score": statistic / threshold if threshold else None,
            "no_material_anomaly_remaining": bool(statistic < threshold),
            "globally_resolved": bool(statistic < threshold),
            "unresolved_signatures": _dedupe(signatures),
            "wls_summary": summary,
        }
        source_action = state.get("source_action")
        is_candidate = str(state.get("status") or "") == "candidate"
        if is_candidate and isinstance(source_action, Mapping):
            target_evidence = self._target_evidence(
                source_action,
                residuals,
                [float(value) for value in payload.get("lambdaN") or []],
                nl,
                candidate_case=solved["ppc"],
            )
            if target_evidence is not None:
                resolved = bool(statistic < threshold)
                metrics.update(target_evidence)
                metrics["post_action_resolved"] = resolved
                metrics["globally_resolved"] = resolved and target_evidence["target_fixed"]
        # Physical feasibility is a separate, narrowly scoped observable
        # check.  Run it for every successfully estimated candidate, including
        # a candidate that intentionally leaves another anomaly for the next
        # recovery step.  The result is derived from case topology plus the
        # measured Vm/terminal-flow channels; WLS convergence is only what
        # makes those channels available and is never used as the safety
        # predicate.  The explicit scope prevents a passing snapshot check
        # from being mistaken for global anomaly resolution or an AC
        # power-flow convergence claim.
        if is_candidate:
            metrics.update(self._steady_state_physical_evidence(solved))
        return metrics

    # ----------------------------------------------------------------- contexts

    def get_measurement_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("measurement_context_input_error", f"{type(exc).__name__}: {exc}")
        payload = solved["payload"]
        if not payload.get("success"):
            return self._failure("measurement_context_failure", payload.get("error"))
        residuals = [float(value) for value in payload.get("r") or []]
        evidence = build_residual_evidence(
            residuals, solved["index_map"], k=self.top_k, min_abs=self.residual_threshold
        )
        state_id = str(state.get("state_id") or "")
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        accepted_branch_rows: set[int] = set()
        for accepted in observation.get("accepted_corrections") or []:
            if not isinstance(accepted, Mapping):
                continue
            accepted_action = accepted.get("source_action") or accepted.get("action")
            if not isinstance(accepted_action, Mapping) or str(
                accepted_action.get("tool") or ""
            ) not in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
                continue
            accepted_arguments = accepted_action.get("arguments")
            accepted_arguments = (
                accepted_arguments if isinstance(accepted_arguments, Mapping) else {}
            )
            try:
                if accepted_arguments.get("branch_row0") is not None:
                    accepted_branch_rows.add(int(accepted_arguments["branch_row0"]))
                elif accepted_arguments.get("line_index1") is not None:
                    accepted_branch_rows.add(int(accepted_arguments["line_index1"]) - 1)
                elif accepted_arguments.get("line_index") is not None:
                    accepted_branch_rows.add(int(accepted_arguments["line_index"]) - 1)
            except (TypeError, ValueError, OverflowError):
                continue
        suppressed_colocated = sorted(
            int(item["index0"])
            for item in evidence
            if str(item.get("channel") or "") in {"Pf", "Qf", "Pt", "Qt"}
            and item.get("channel_offset") in accepted_branch_rows
        )
        if accepted_branch_rows:
            evidence = [
                item
                for item in evidence
                if not (
                    str(item.get("channel") or "") in {"Pf", "Qf", "Pt", "Qt"}
                    and item.get("channel_offset") in accepted_branch_rows
                )
            ]
        # Preserve residual-strength ordering for singleton hypotheses.  The
        # only joint action allowed ahead of them is the physically bounded Vm
        # group below; a broad top-k correction could rewrite healthy channels
        # and is never emitted.
        ranked_indices = list(
            dict.fromkeys(int(item["index0"]) for item in evidence)
        )
        physical_vm_joint_targets = self._physical_vm_joint_targets(
            solved, evidence
        )
        supported: list[dict[str, Any]] = []
        for index in ranked_indices:
            supported.append(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {"state_id": state_id, "suspect_group": [index]},
                }
            )
        if len(physical_vm_joint_targets) >= 2:
            supported.append(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": state_id,
                        "suspect_group": physical_vm_joint_targets,
                    },
                }
            )
        highest_remaining_vm_residual = next(
            (
                index
                for index in ranked_indices
                if index not in set(physical_vm_joint_targets)
            ),
            None,
        )
        physical_vm_closure_targets = (
            sorted({*physical_vm_joint_targets, highest_remaining_vm_residual})
            if physical_vm_joint_targets
            and highest_remaining_vm_residual is not None
            else []
        )
        if len(physical_vm_closure_targets) >= 2:
            supported.append(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": state_id,
                        "suspect_group": physical_vm_closure_targets,
                    },
                }
            )
        accepted_records = observation.get("accepted_corrections") or []
        accepted_indices: set[int] = set()
        accepted_index_counts: dict[int, int] = {}
        accepted_joint_groups: set[frozenset[int]] = set()
        for accepted in accepted_records:
            if not isinstance(accepted, Mapping):
                continue
            accepted_action = accepted.get("source_action") or accepted.get("action")
            if not isinstance(accepted_action, Mapping):
                continue
            if str(accepted_action.get("tool") or "") != CORRECT_MEASUREMENTS:
                continue
            accepted_arguments = accepted_action.get("arguments")
            accepted_arguments = (
                accepted_arguments if isinstance(accepted_arguments, Mapping) else {}
            )
            accepted_group = accepted_arguments.get("suspect_group")
            accepted_updates = accepted_arguments.get("measurement_updates")
            raw_indices = (
                accepted_group
                if isinstance(accepted_group, Sequence)
                and not isinstance(accepted_group, (str, bytes))
                else accepted_updates.keys()
                if isinstance(accepted_updates, Mapping)
                else ()
            )
            accepted_group_indices: set[int] = set()
            for raw_index in raw_indices:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError, OverflowError):
                    continue
                accepted_indices.add(index)
                accepted_index_counts[index] = accepted_index_counts.get(index, 0) + 1
                accepted_group_indices.add(index)
            if len(accepted_group_indices) >= 2:
                accepted_joint_groups.add(frozenset(accepted_group_indices))

        statistic = float(payload.get("global_residual_sum") or 0.0)
        dof = max(1, len(residuals) - (2 * int(solved["nb"]) - 1))
        threshold = float(chi2_threshold(dof, self.chi2_alpha))
        colocated_accepted_indices: set[int] = set()
        if accepted_branch_rows:
            for index in accepted_indices:
                for channel in ("Pf", "Qf", "Pt", "Qt"):
                    channel_slice = solved["index_map"].get(channel)
                    if (
                        channel_slice is not None
                        and channel_slice.start <= index < channel_slice.stop
                        and index - channel_slice.start in accepted_branch_rows
                    ):
                        colocated_accepted_indices.add(index)
                        break
        refinement_targets = sorted(accepted_indices - colocated_accepted_indices)
        unaccepted_targets_in_rank_order = [
            index for index in ranked_indices if index not in accepted_indices
        ]
        unaccepted_ranked_targets = sorted(set(unaccepted_targets_in_rank_order))
        dominant_unaccepted_target = bool(
            ranked_indices and ranked_indices[0] not in accepted_indices
        )
        try:
            remaining_budget = int(observation.get("remaining_budget") or 0)
        except (TypeError, ValueError, OverflowError):
            remaining_budget = 0
        anomaly_ratio = statistic / threshold if threshold > 0.0 else math.inf
        near_threshold_refinement_override = bool(
            dominant_unaccepted_target
            and anomaly_ratio <= _COUPLED_REFINEMENT_MAX_ANOMALY_RATIO
            and remaining_budget >= _COUPLED_REFINEMENT_MIN_REMAINING_BUDGET
        )
        refinement_already_accepted = bool(
            refinement_targets
            and (
                frozenset(refinement_targets) in accepted_joint_groups
                or (
                    len(refinement_targets) == 1
                    and accepted_index_counts.get(refinement_targets[0], 0) >= 2
                )
            )
        )
        # A sequence of independently estimated singleton corrections can
        # leave coupled residual energy on an otherwise healthy channel.  Do
        # one joint re-estimation of the already accepted targets after no new
        # residual-dominant singleton remains.  There is one narrow exception:
        # inside a 10% chi-square ambiguity band, and only with enough budget
        # to survive a rejected transaction, refine the accepted set before
        # expanding it.  This observable guard prevents a weak collateral
        # residual from turning a fully repaired episode into a healthy-meter
        # rewrite while still prioritizing clear new faults.
        coupled_refinement_ready = bool(
            len(refinement_targets) >= 2
            and statistic >= threshold
            and (
                not dominant_unaccepted_target
                or near_threshold_refinement_override
            )
            and not refinement_already_accepted
        )
        # A meter corrected before a branch-model repair was estimated against
        # the stale model.  If the global statistic remains above threshold,
        # permit one re-estimation of only those already accepted meter targets
        # on the repaired model.  This cannot introduce a new target, excludes
        # direct-flow channels on the repaired branch, and still has to pass the
        # normal transactional candidate verification before it can commit.
        post_branch_refinement_ready = bool(
            accepted_branch_rows
            and refinement_targets
            and statistic >= threshold
            and not refinement_already_accepted
        )
        refinement_ready = coupled_refinement_ready or post_branch_refinement_ready
        coupled_fallback_targets = (
            sorted(
                set(refinement_targets)
                | set(unaccepted_targets_in_rank_order[:2])
            )
            if len(unaccepted_targets_in_rank_order) >= 2
            else []
        )
        # Keep this legacy grouping only as diagnostic evidence.  Singleton
        # support does not prove that every member of a broad residual group
        # is a faulty meter, so it must never be advertised as executable.
        if refinement_ready:
            all_residual_evidence = build_residual_evidence(
                residuals,
                solved["index_map"],
                k=len(residuals),
                min_abs=0.0,
            )
            present = {int(item["index0"]) for item in evidence}
            evidence.extend(
                item
                for item in all_residual_evidence
                if int(item["index0"]) in accepted_indices
                and int(item["index0"]) not in present
            )
            refinement_action = {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": state_id,
                    "suspect_group": refinement_targets,
                },
            }
            if refinement_action not in supported:
                supported.append(refinement_action)
        branch_route_screening = self._post_measurement_branch_route_screening(
            state,
            accepted_indices=accepted_indices,
            anomaly_unresolved=statistic >= threshold,
        )
        branch_routes_exhausted = bool(
            set(branch_route_screening) == {"parameter", "topology"}
            and all(
                branch_route_screening[family].get("route_status")
                == _ROUTE_COMPLETE_NEGATIVE
                and not branch_route_screening[family]["supported_corrections"]
                for family in ("parameter", "topology")
            )
        )
        terminal_closure_action: dict[str, Any] | None = None
        terminal_closure_evidence: dict[str, Any] = {}
        if branch_routes_exhausted and not accepted_branch_rows:
            terminal_closure_action, terminal_closure_evidence = (
                self._verified_terminal_measurement_closure(
                    state,
                    accepted_indices=accepted_indices,
                    ranked_indices=ranked_indices,
                    parent_score=(statistic / threshold if threshold > 0.0 else None),
                )
            )
        if terminal_closure_action is not None:
            supported.append(terminal_closure_action)
        terminal_closure_targets = (
            list(terminal_closure_action["arguments"]["suspect_group"])
            if terminal_closure_action is not None
            else []
        )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_residuals",
            "context_tool": GET_MEASUREMENT_CONTEXT,
            "finding_count": len(evidence),
            "measurement_findings": evidence,
            "supported_corrections": supported,
            "physical_vm_joint_targets": physical_vm_joint_targets,
            "physical_vm_closure_targets": physical_vm_closure_targets,
            "coupled_measurement_fallback_targets": coupled_fallback_targets,
            "suppressed_colocated_post_branch_indices": suppressed_colocated,
            "accepted_target_refinement": bool(
                refinement_ready
            ),
            "accepted_target_refinement_blocked_by": unaccepted_ranked_targets,
            "accepted_target_refinement_dominant_target_unaccepted": (
                dominant_unaccepted_target
            ),
            "accepted_target_refinement_near_threshold_override": (
                near_threshold_refinement_override
            ),
            "accepted_target_refinement_anomaly_ratio": anomaly_ratio,
            "accepted_target_refinement_remaining_budget": remaining_budget,
            "accepted_target_refinement_already_accepted": (
                refinement_already_accepted
            ),
            "accepted_target_refinement_kind": (
                "post_branch_model_reestimate"
                if post_branch_refinement_ready
                else "coupled_measurement_reestimate"
                if coupled_refinement_ready
                else None
            ),
            "accepted_target_refinement_suppressed_colocated_indices": sorted(
                colocated_accepted_indices
            ),
            "branch_route_screening": branch_route_screening,
            "verified_terminal_measurement_closure_targets": (
                terminal_closure_targets
            ),
            "verified_terminal_measurement_closure_evidence": (
                terminal_closure_evidence
            ),
            "chi_square_statistic": statistic,
            "chi_square_threshold": threshold,
        }

    def _post_measurement_branch_route_screening(
        self,
        state: Mapping[str, Any],
        *,
        accepted_indices: set[int],
        anomaly_unresolved: bool,
    ) -> dict[str, dict[str, Any]]:
        """Bundle current branch inventories after a partial meter commit.

        A fresh measurement solve is already required after every accepted
        partial correction.  At that same immutable active state, collect the
        independently observable parameter and topology inventories so the
        controller need not spend two additional actions merely to prove that
        both routes are empty.  Non-successful or unbound provider responses
        are omitted, which leaves the corresponding route open (fail closed).
        """

        if not accepted_indices or not anomaly_unresolved:
            return {}
        state_id = str(state.get("state_id") or "")
        state_hash = str(state.get("state_hash") or "")
        contexts: dict[str, dict[str, Any]] = {}
        for family, context_tool, provider in (
            ("parameter", GET_PARAMETER_CONTEXT, self.get_parameter_context),
            ("topology", GET_TOPOLOGY_CONTEXT, self.get_topology_context),
        ):
            metrics = provider(copy.deepcopy(dict(state)))
            if (
                not isinstance(metrics, Mapping)
                or metrics.get("execution_status", "success") != "success"
                or str(metrics.get("state_id") or "") != state_id
                or str(metrics.get("state_hash") or "") != state_hash
                or metrics.get("context_tool") != context_tool
                or not isinstance(metrics.get("supported_corrections"), (list, tuple))
            ):
                continue
            contexts[family] = copy.deepcopy(dict(metrics))
        return contexts

    def _verified_terminal_measurement_closure(
        self,
        state: Mapping[str, Any],
        *,
        accepted_indices: set[int],
        ranked_indices: Sequence[int],
        parent_score: float | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Return one preverified accepted-target-plus-singleton final repair.

        The sole new target must first survive an ordinary singleton candidate
        verdict.  Only then may already committed meter targets be jointly
        re-estimated with it, and that exact grouped candidate must pass the
        deployment physical/quality gate as ``ACCEPT_FINAL``.  This is much
        narrower than a top-k residual group: it never introduces two untried
        targets and it is emitted only after both branch inventories are empty.
        """

        from psse_env.oracle import CandidateDisposition

        if not accepted_indices or parent_score is None:
            return None, {}
        state_id = str(state.get("state_id") or "")
        attempted: list[dict[str, Any]] = []
        for raw_target in ranked_indices:
            target = int(raw_target)
            if target in accepted_indices:
                continue
            singleton_action = {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": state_id, "suspect_group": [target]},
            }
            singleton_assessment, singleton_verification, singleton_record = (
                self._assess_measurement_candidate(
                    state,
                    singleton_action,
                    parent_score=parent_score,
                )
            )
            singleton_record["stage"] = "new_target_singleton"
            attempted.append(singleton_record)
            if singleton_assessment is None or singleton_assessment.disposition not in {
                CandidateDisposition.ACCEPT_FINAL,
                CandidateDisposition.ACCEPT_PARTIAL,
            }:
                continue
            closure_targets = sorted(set(accepted_indices) | {target})
            closure_action = {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": state_id,
                    "suspect_group": closure_targets,
                },
            }
            closure_assessment, closure_verification, closure_record = (
                self._assess_measurement_candidate(
                    state,
                    closure_action,
                    parent_score=parent_score,
                )
            )
            closure_record["stage"] = "accepted_targets_plus_singleton"
            attempted.append(closure_record)
            if (
                closure_assessment is not None
                and closure_assessment.disposition
                == CandidateDisposition.ACCEPT_FINAL
                and closure_verification.get("globally_resolved") is True
                and closure_verification.get("target_fixed") is True
                and closure_verification.get("physical_constraints_ok") is True
            ):
                return closure_action, {
                    "eligible": True,
                    "state_id": state_id,
                    "state_hash": str(state.get("state_hash") or ""),
                    "screening_method": (
                        "singleton_then_grouped_deployment_candidate_quality"
                    ),
                    "new_target": target,
                    "closure_targets": closure_targets,
                    "attempts": attempted,
                }
        return None, {
            "eligible": False,
            "state_id": state_id,
            "state_hash": str(state.get("state_hash") or ""),
            "screening_method": "singleton_then_grouped_deployment_candidate_quality",
            "attempts": attempted,
        }

    def _assess_measurement_candidate(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        *,
        parent_score: float,
    ) -> tuple[Any | None, dict[str, Any], dict[str, Any]]:
        """Apply and assess a copied measurement candidate without mutation."""

        arguments = action.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        targets = [int(index) for index in arguments.get("suspect_group") or []]
        record: dict[str, Any] = {
            "targets": targets,
            "screening_method": "deployment_candidate_quality_non_mutating",
        }
        try:
            correction = self.correct_measurements(state, action)
            if correction.get("execution_status", "success") != "success":
                record.update(
                    {
                        "disposition": "REJECT",
                        "progress_class": "correction_execution_failure",
                        "rationale_codes": [
                            str(correction.get("error_code") or "measurement_correction_failure")
                        ],
                    }
                )
                return None, {}, record
            modification = correction.get("modification")
            if not isinstance(modification, Mapping):
                record.update(
                    {
                        "disposition": "REJECT",
                        "progress_class": "candidate_modification_missing",
                        "rationale_codes": ["measurement_candidate_modification_missing"],
                    }
                )
                return None, {}, record
            parent = copy.deepcopy(dict(state))
            candidate = copy.deepcopy(parent)
            candidate_case, candidate_measurements, candidate_metadata = apply_modification(
                case=parent.get("case"),
                measurements=parent.get("measurements"),
                metadata=(
                    parent.get("metadata")
                    if isinstance(parent.get("metadata"), Mapping)
                    else {}
                ),
                modification=modification,
            )
            digest = hashlib.sha256(
                json.dumps(targets, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            candidate.update(
                {
                    "state_id": (
                        f"{str(state.get('state_id') or '')}:measurement-screen:{digest}"
                    ),
                    "parent_state_id": state.get("state_id"),
                    "status": "candidate",
                    "source_action": copy.deepcopy(dict(action)),
                    "modification": copy.deepcopy(dict(modification)),
                    "case": candidate_case,
                    "measurements": candidate_measurements,
                    "metadata": candidate_metadata,
                }
            )
            candidate.pop("state_hash", None)
            verification = self.run_wls(candidate)
            if verification.get("execution_status", "success") != "success":
                record.update(
                    {
                        "disposition": "REJECT",
                        "progress_class": "verification_solver_failure",
                        "rationale_codes": [
                            str(verification.get("error_code") or "wls_failure")
                        ],
                    }
                )
                return None, verification, record
            candidate_score = verification.get("remaining_anomaly_score")
            try:
                denominator = max(abs(float(parent_score)), 1e-12)
                verification["global_progress"] = (
                    float(parent_score) - float(candidate_score)
                ) / denominator
                verification["parent_anomaly_score"] = float(parent_score)
            except (TypeError, ValueError, OverflowError):
                pass
            assessment = self._deployment_candidate_quality_oracle().label_candidate(
                parent_state=parent,
                source_action=action,
                candidate_state=candidate,
                verification_output=verification,
                hidden_truth=None,
            )
            record.update(
                {
                    "disposition": assessment.disposition.value,
                    "progress_class": assessment.progress_class,
                    "global_progress": assessment.global_progress,
                    "target_test_passed": verification.get("target_fixed"),
                    "globally_resolved": verification.get("globally_resolved"),
                    "physical_constraints_ok": verification.get(
                        "physical_constraints_ok"
                    ),
                    "rationale_codes": list(assessment.rationale_codes),
                }
            )
            return assessment, verification, record
        except Exception as exc:
            record.update(
                {
                    "disposition": "REJECT",
                    "progress_class": "candidate_screening_failure",
                    "rationale_codes": [f"screening_{type(exc).__name__}"],
                }
            )
            return None, {}, record

    def _physical_vm_joint_targets(
        self,
        solved: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        """Group only residual-ranked Vm channels outside declared limits.

        Multiple corrupted voltage-magnitude meters can make every singleton
        candidate fail the absolute physical check because the other bad Vm
        channels remain outside VMIN/VMAX.  This bounded proposal contains
        only current residual findings that independently exceed the residual
        threshold and whose raw telemetry violates the corresponding case
        limit by more than the configured physical tolerance.  It never adds
        an in-bound residual or a non-Vm channel, and singleton alternatives
        remain in the context response for transactional fallback.
        """
        try:
            bus = solved["ppc"]["bus"]
            z = solved["z"]
            vm_slice = solved["index_map"]["Vm"]
            nb = int(solved["nb"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return []
        try:
            if len(bus) != nb or len(z) < int(vm_slice.stop):
                return []
        except (TypeError, AttributeError):
            return []

        targets: list[int] = []
        for item in evidence:
            if not isinstance(item, Mapping) or str(item.get("channel") or "") != "Vm":
                continue
            try:
                index = int(item["index0"])
                channel_offset = int(item["channel_offset"])
                residual_value = abs(float(item["value"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (
                residual_value < self.residual_threshold
                or index != int(vm_slice.start) + channel_offset
                or not 0 <= channel_offset < nb
            ):
                continue
            try:
                if int(round(float(bus[channel_offset][1]))) == 4:
                    continue
                observed = float(z[index])
                vmax = float(bus[channel_offset][11])
                vmin = float(bus[channel_offset][12])
            except (IndexError, TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in (observed, vmin, vmax)):
                continue
            if vmin <= 0.0 or vmax < vmin:
                continue
            if (
                observed < vmin - self.vm_bound_tolerance_pu
                or observed > vmax + self.vm_bound_tolerance_pu
            ):
                targets.append(index)
        return sorted(set(targets))

    def _lambda_findings(self, solved: Mapping[str, Any]) -> list[dict[str, Any]]:
        payload = solved["payload"]
        return build_lambda_evidence(
            [float(value) for value in payload.get("lambdaN") or []],
            payload.get("branch_info") or [],
            k=self.top_k,
            min_abs=self.lambda_threshold,
        )

    def get_parameter_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("parameter_context_input_error", f"{type(exc).__name__}: {exc}")
        if not solved["payload"].get("success"):
            return self._failure("parameter_context_failure", solved["payload"].get("error"))
        findings = self._lambda_findings(solved)
        state_id = str(state.get("state_id") or "")
        seen_rows: list[int] = []
        for item in findings:
            row0 = item.get("line_row0")
            if row0 is not None and row0 not in seen_rows:
                seen_rows.append(int(row0))
        metadata = state.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        parameter_scans = metadata.get("parameter_scans")
        z_scans = (
            parameter_scans.get("z_scans")
            if isinstance(parameter_scans, Mapping)
            else None
        )
        scan_count = (
            len(z_scans)
            if isinstance(z_scans, Sequence)
            and not isinstance(z_scans, (str, bytes))
            else 0
        )
        scans_usable = False
        if scan_count > 0:
            try:
                # Use the executor's exact observable-input validator before
                # advertising a correction. Dimension, finiteness, and
                # voltage-magnitude failures must remain diagnostic context,
                # not a guaranteed invalid policy action.
                observable_parameter_initial_states(solved["ppc"], z_scans)
                scans_usable = True
            except (KeyError, TypeError, ValueError, OverflowError):
                scans_usable = False
        # A branch multiplier is useful diagnostic evidence even when the
        # repeated telemetry required by the parameter solver is unavailable.
        # In that case the context must not advertise a correction that the
        # same provider is guaranteed to reject; the expert can proceed to the
        # independently observable topology route without manufacturing an
        # invalid-action recovery example.
        supported = (
            [
                {
                    "tool": CORRECT_PARAMETERS,
                    "arguments": {"state_id": state_id, "line_index": row0 + 1},
                }
                for row0 in seen_rows
            ]
            if scans_usable
            else []
        )
        route_status = (
            _ROUTE_ACTIONABLE
            if supported
            else _ROUTE_COMPLETE_NEGATIVE
            if not findings
            else _ROUTE_UNAVAILABLE
        )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_lagrange",
            "context_tool": GET_PARAMETER_CONTEXT,
            "finding_count": len(findings),
            "parameter_findings": findings,
            "parameter_scans_available": scans_usable,
            "parameter_scan_count": scan_count,
            "supported_corrections": supported,
            "route_status": route_status,
            "route_status_reason": (
                "supported_parameter_candidates"
                if route_status == _ROUTE_ACTIONABLE
                else "no_parameter_findings"
                if route_status == _ROUTE_COMPLETE_NEGATIVE
                else "parameter_findings_require_repeated_scans"
            ),
        }

    def get_topology_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("topology_context_input_error", f"{type(exc).__name__}: {exc}")
        if not solved["payload"].get("success"):
            return self._failure("topology_context_failure", solved["payload"].get("error"))
        findings = self._lambda_findings(solved)
        state_id = str(state.get("state_id") or "")
        branch = solved["ppc"]["branch"]
        proposed: list[dict[str, Any]] = []
        seen_rows: set[int] = set()
        islanding_filtered: list[int] = []
        for item in findings:
            row0 = item.get("line_row0")
            if row0 is None or row0 in seen_rows or not 0 <= int(row0) < solved["nl"]:
                continue
            seen_rows.add(int(row0))
            current_status = int(float(branch[int(row0)][10])) if branch.shape[1] > 10 else 1
            proposed_status = 0 if current_status else 1
            if self._flip_creates_island(branch, int(row0), proposed_status):
                islanding_filtered.append(int(row0) + 1)
                continue
            proposed.append(
                {
                    "tool": CORRECT_TOPOLOGY,
                    "arguments": {
                        "state_id": state_id,
                        "line_index": int(row0) + 1,
                        "status": proposed_status,
                    },
                }
            )
        parent_score = self._remaining_anomaly_score(solved)
        supported: list[dict[str, Any]] = []
        candidate_screening: list[dict[str, Any]] = []
        for action in proposed:
            eligible, evidence = self._screen_topology_correction(
                state,
                action,
                parent_score=parent_score,
            )
            candidate_screening.append(evidence)
            if eligible:
                supported.append(action)
        screening_incomplete = any(
            item.get("screening_complete") is not True
            for item in candidate_screening
        )
        route_status = (
            _ROUTE_UNAVAILABLE
            if screening_incomplete
            else _ROUTE_ACTIONABLE
            if supported
            else _ROUTE_COMPLETE_NEGATIVE
        )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_lagrange_candidate_screened",
            "context_tool": GET_TOPOLOGY_CONTEXT,
            "finding_count": len(findings),
            "topology_findings": findings,
            "supported_corrections": supported,
            "proposed_correction_count": len(proposed),
            "screened_correction_count": len(candidate_screening),
            "topology_candidate_screening": candidate_screening,
            "islanding_filtered_lines": islanding_filtered,
            "route_status": route_status,
            "route_status_reason": (
                "candidate_screening_incomplete"
                if route_status == _ROUTE_UNAVAILABLE
                else "supported_topology_candidates"
                if route_status == _ROUTE_ACTIONABLE
                else "all_topology_findings_observably_rejected"
                if findings
                else "no_topology_findings"
            ),
        }

    def _remaining_anomaly_score(self, solved: Mapping[str, Any]) -> float | None:
        """Return the same normalized chi-square score emitted by ``run_wls``."""

        try:
            residuals = [float(value) for value in solved["payload"].get("r") or []]
            dof = max(1, len(residuals) - (2 * int(solved["nb"]) - 1))
            threshold = float(chi2_threshold(dof, self.chi2_alpha))
            statistic = float(solved["payload"].get("global_residual_sum") or 0.0)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(statistic) or not math.isfinite(threshold) or threshold <= 0.0:
            return None
        return statistic / threshold

    def _screen_topology_correction(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        *,
        parent_score: float | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Fail closed unless a topology hypothesis is observably admissible.

        Screening is a non-mutating lookahead over a copied provider state.  It
        applies the exact correction executor, reruns deployment WLS and the
        scoped physical checks, and delegates the verdict to the same
        deployment ``CandidateQualityOracle`` used by the environment.  The
        real transaction remains authoritative if the policy later selects an
        advertised action.
        """

        from psse_env.oracle import CandidateDisposition

        normalized_action = copy.deepcopy(dict(action))
        arguments = normalized_action.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        line_index = arguments.get("line_index")
        status = arguments.get("status")
        evidence: dict[str, Any] = {
            "state_id": str(state.get("state_id") or ""),
            "state_hash": str(state.get("state_hash") or ""),
            "line_index": line_index,
            "status": status,
            "screening_method": "deployment_candidate_quality_non_mutating",
            "screening_complete": False,
        }
        try:
            correction = self.correct_topology(state, normalized_action)
            if correction.get("execution_status", "success") != "success":
                evidence.update(
                    {
                        "eligible": False,
                        "disposition": "INCONCLUSIVE",
                        "progress_class": "correction_execution_failure",
                        "rationale_codes": [
                            str(
                                correction.get("error_code")
                                or "topology_correction_execution_failure"
                            )
                        ],
                    }
                )
                return False, evidence
            modification = correction.get("modification")
            if not isinstance(modification, Mapping) or not modification.get("case"):
                evidence.update(
                    {
                        "eligible": False,
                        "disposition": "INCONCLUSIVE",
                        "progress_class": "candidate_modification_missing",
                        "rationale_codes": ["topology_candidate_case_missing"],
                    }
                )
                return False, evidence

            parent = copy.deepcopy(dict(state))
            candidate = copy.deepcopy(parent)
            candidate_case, candidate_measurements, candidate_metadata = apply_modification(
                case=parent.get("case"),
                measurements=parent.get("measurements"),
                metadata=(
                    parent.get("metadata")
                    if isinstance(parent.get("metadata"), Mapping)
                    else {}
                ),
                modification=modification,
            )
            screen_suffix = f"l{line_index}s{status}"
            candidate.update(
                {
                    "state_id": (
                        f"{str(state.get('state_id') or '')}:topology-screen:{screen_suffix}"
                    ),
                    "parent_state_id": state.get("state_id"),
                    "status": "candidate",
                    "source_action": normalized_action,
                    "modification": copy.deepcopy(dict(modification)),
                    "case": candidate_case,
                    "measurements": candidate_measurements,
                    "metadata": candidate_metadata,
                }
            )
            candidate.pop("state_hash", None)
            verification = self.run_wls(candidate)
            if verification.get("execution_status", "success") != "success":
                evidence.update(
                    {
                        "eligible": False,
                        "disposition": "REJECT",
                        "screening_complete": True,
                        "progress_class": "verification_solver_failure",
                        "rationale_codes": [
                            str(verification.get("error_code") or "wls_failure")
                        ],
                    }
                )
                return False, evidence

            candidate_score = verification.get("remaining_anomaly_score")
            try:
                if parent_score is not None and candidate_score is not None:
                    denominator = max(abs(float(parent_score)), 1e-12)
                    verification["global_progress"] = (
                        float(parent_score) - float(candidate_score)
                    ) / denominator
                    verification["parent_anomaly_score"] = float(parent_score)
            except (TypeError, ValueError, OverflowError):
                pass
            assessment = self._deployment_candidate_quality_oracle().label_candidate(
                parent_state=parent,
                source_action=normalized_action,
                candidate_state=candidate,
                verification_output=verification,
                hidden_truth=None,
            )
            eligible = assessment.disposition in {
                CandidateDisposition.ACCEPT_FINAL,
                CandidateDisposition.ACCEPT_PARTIAL,
            }
            evidence.update(
                {
                    "eligible": eligible,
                    "disposition": assessment.disposition.value,
                    "screening_complete": (
                        assessment.disposition.value != "INCONCLUSIVE"
                    ),
                    "progress_class": assessment.progress_class,
                    "global_progress": assessment.global_progress,
                    "target_test_passed": verification.get("target_fixed"),
                    "physical_constraints_ok": verification.get(
                        "physical_constraints_ok"
                    ),
                    "topology_target_status_matches_requested": verification.get(
                        "topology_target_status_matches_requested"
                    ),
                    "topology_target_branch_multiplier": verification.get(
                        "topology_target_branch_multiplier"
                    ),
                    "rationale_codes": list(assessment.rationale_codes),
                }
            )
            return eligible, evidence
        except Exception as exc:
            evidence.update(
                {
                    "eligible": False,
                    "disposition": "INCONCLUSIVE",
                    "progress_class": "candidate_screening_failure",
                    "rationale_codes": [f"screening_{type(exc).__name__}"],
                }
            )
            return False, evidence

    @staticmethod
    def _flip_creates_island(branch: Any, row0: int, proposed_status: int) -> bool:
        """Opening a line that is the only path to a bus islands the network.

        A real EMS would never offer that switching action, so the context
        provider filters it from supported corrections (closing a line can
        only improve connectivity and is never filtered).
        """
        if proposed_status != 0:
            return False
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        array = np.asarray(branch, dtype=float)
        statuses = (
            array[:, 10].astype(int).copy()
            if array.shape[1] > 10
            else np.ones(array.shape[0], dtype=int)
        )
        statuses[row0] = 0
        active = statuses != 0
        bus_ids = np.unique(array[:, :2].astype(int))
        index_of = {int(bus): i for i, bus in enumerate(bus_ids)}
        from_idx = [index_of[int(b)] for b in array[active, 0]]
        to_idx = [index_of[int(b)] for b in array[active, 1]]
        n_bus = len(bus_ids)
        adjacency = coo_matrix(
            (np.ones(len(from_idx)), (from_idx, to_idx)), shape=(n_bus, n_bus)
        )
        components, _ = connected_components(adjacency, directed=False)
        return int(components) > 1

    # ---------------------------------------------------------------- executors

    def correct_measurements(
        self, state: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        arguments = dict(action.get("arguments") or {})
        updates = arguments.get("measurement_updates")
        if isinstance(updates, Mapping) and updates:
            return {
                "modification": {
                    "measurement_updates": {int(key): float(value) for key, value in updates.items()}
                },
                "evidence_source": "deployment_correction:explicit_updates",
            }
        suspect_group = arguments.get("suspect_group")
        if not isinstance(suspect_group, Sequence) or isinstance(suspect_group, (str, bytes)):
            return self._failure(
                "measurement_correction_target_missing",
                "correct_measurements requires suspect_group or measurement_updates",
            )
        try:
            case_path = self._case_path(state)
            z = self._measurements(state)
            payload = _meas_correction_json(
                case_path,
                z,
                suspect_group=[int(index) for index in suspect_group],
                enable_correction=True,
                max_correction_iterations=self.max_correction_iterations,
                error_tolerance=self.error_tolerance,
            )
        except Exception as exc:
            return self._failure("measurement_correction_error", f"{type(exc).__name__}: {exc}")
        if not payload.get("success"):
            return self._failure("measurement_correction_failure", payload.get("error"))
        corrected = {
            int(item["index0"]): float(item["corrected"])
            for item in payload.get("corrected_measurements") or []
            if item.get("index0") is not None and item.get("corrected") is not None
        }
        if not corrected:
            return self._failure(
                "measurement_correction_no_change",
                "grouped correction produced no corrected measurements",
            )
        return {
            "modification": {"measurement_updates": corrected},
            "evidence_source": "deployment_correction:lagrangian_correct_port",
            "correction_summary": summarize_measurement_correction_payload(payload),
            "applied_any_correction": bool(payload.get("applied_any_correction")),
            "iterations_performed": payload.get("iterations_performed"),
            "suspect_group": sorted(int(index) for index in suspect_group),
        }

    def correct_parameters(
        self, state: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        arguments = dict(action.get("arguments") or {})
        try:
            case_path = self._case_path(state)
            ppc = _load_python_case(case_path)
            nl = int(ppc["branch"].shape[0])
            row0 = self._branch_row0(arguments, nl)
        except Exception as exc:
            return self._failure("parameter_correction_input_error", f"{type(exc).__name__}: {exc}")
        metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
        scans = metadata.get("parameter_scans")
        if not isinstance(scans, Mapping) or not scans.get("z_scans"):
            return self._failure(
                "parameter_scans_missing",
                "multi-scan parameter correction requires metadata.parameter_scans "
                "with observed z_scans",
            )
        try:
            z_scans = [list(map(float, scan)) for scan in scans["z_scans"]]
            initial_states = observable_parameter_initial_states(ppc, z_scans)
            payload = _param_correction_json(
                case_path,
                row0 + 1,
                z_scans,
                initial_states,
            )
        except Exception as exc:
            return self._failure("parameter_correction_error", f"{type(exc).__name__}: {exc}")
        if not payload.get("success"):
            return self._failure("parameter_correction_failure", payload.get("error"))
        corrected = payload.get("corrected_params") or []
        if len(corrected) < 2:
            return self._failure(
                "parameter_correction_no_change", "solver returned no corrected [r, x] pair"
            )
        updated = copy.deepcopy(ppc)
        updated["branch"][row0][2] = float(corrected[0])
        updated["branch"][row0][3] = float(corrected[1])
        derived_path = self._derived_case(updated, f"param_l{row0 + 1}")
        return {
            "modification": {
                "case": derived_path,
                "metadata_updates": {"last_parameter_correction": {"line_index": row0 + 1}},
            },
            "evidence_source": "deployment_correction:multi_scan_parameter_port",
            "correction_summary": summarize_parameter_correction_payload(payload),
            "line_index": row0 + 1,
            "corrected_r": float(corrected[0]),
            "corrected_x": float(corrected[1]),
        }

    def correct_topology(
        self, state: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        arguments = dict(action.get("arguments") or {})
        try:
            case_path = self._case_path(state)
            ppc = _load_python_case(case_path)
            nl = int(ppc["branch"].shape[0])
            row0 = self._branch_row0(arguments, nl)
        except Exception as exc:
            return self._failure("topology_correction_input_error", f"{type(exc).__name__}: {exc}")
        status = arguments.get("status", arguments.get("expected_status"))
        if arguments.get("desired_status") is not None and status is None:
            status = int(bool(arguments["desired_status"]))
        if status is None:
            return self._failure(
                "topology_correction_target_missing", "correct_topology requires a status"
            )
        new_status = int(status)
        if ppc["branch"].shape[1] <= 10:
            return self._failure(
                "topology_correction_unsupported", "case branch matrix has no status column"
            )
        current_status = int(float(ppc["branch"][row0][10]))
        if current_status == new_status:
            return self._failure(
                "topology_correction_no_change",
                f"branch row {row0} already has status {new_status}",
            )
        updated = copy.deepcopy(ppc)
        updated["branch"][row0][10] = float(new_status)
        derived_path = self._derived_case(updated, f"topo_l{row0 + 1}s{new_status}")
        return {
            "modification": {
                "case": derived_path,
                "metadata_updates": {
                    "last_topology_correction": {"line_index": row0 + 1, "status": new_status}
                },
            },
            "evidence_source": "deployment_correction:branch_status",
            "line_index": row0 + 1,
            "previous_status": current_status,
            "new_status": new_status,
        }


    # ------------------------------------------------- specialized diagnostics

    @staticmethod
    def _metadata(state: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = state.get("metadata")
        return metadata if isinstance(metadata, Mapping) else {}

    def _harmonic_measurements(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        measurements = self._metadata(state).get("harmonic_measurements")
        if not isinstance(measurements, Sequence) or not measurements:
            raise ValueError(
                "state metadata carries no harmonic_measurements; generate the "
                "scenario with harmonic scan data to enable harmonic diagnostics"
            )
        return [dict(item) for item in measurements]

    @staticmethod
    def _observable_signatures(state: Mapping[str, Any]) -> list[str]:
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        return [str(item) for item in observation.get("unresolved_signatures") or []]

    @classmethod
    def _has_family_signature(cls, state: Mapping[str, Any], family: str) -> bool:
        return any(
            _matches_any_marker(signature, ANOMALY_FAMILY_MARKERS[family])
            for signature in cls._observable_signatures(state)
        )

    def _hif_diagnostic_acceptance(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a fail-closed HIF-vs-null goodness-of-fit gate."""
        fit = payload.get("fit")
        fit = fit if isinstance(fit, Mapping) else {}
        observability = payload.get("observability")
        observability = observability if isinstance(observability, Mapping) else {}
        improvement = fit.get("residual_reduction_vs_no_hif")
        if improvement is None:
            # The single-scan estimator uses this older name for improvement
            # over the no-HIF base-model simulation.
            improvement = fit.get("residual_reduction_vs_no_refinement")
        residual = fit.get("multiscan_weighted_residual_norm")
        if residual is None:
            residual = fit.get("weighted_residual_norm")
        try:
            improvement_value = float(improvement)
            residual_value = float(residual)
        except (TypeError, ValueError):
            improvement_value = math.nan
            residual_value = math.nan
        model_mismatch = bool(
            fit.get("model_mismatch_suspected")
            or observability.get("model_mismatch_suspected")
        )
        accepted = bool(
            math.isfinite(improvement_value)
            and math.isfinite(residual_value)
            and improvement_value >= self.hif_min_residual_reduction
            and residual_value <= self.hif_max_weighted_residual_norm
            and not model_mismatch
            and not payload.get("synthetic_oracle", False)
        )
        return {
            "accepted": accepted,
            "null_hypothesis": "no_hif_base_model",
            "residual_reduction_vs_null": (
                improvement_value if math.isfinite(improvement_value) else None
            ),
            "weighted_residual_norm": residual_value if math.isfinite(residual_value) else None,
            "minimum_residual_reduction": self.hif_min_residual_reduction,
            "maximum_weighted_residual_norm": self.hif_max_weighted_residual_norm,
            "model_mismatch_suspected": model_mismatch,
        }

    def get_harmonic_context(
        self, state: Mapping[str, Any], action: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            measurements = self._harmonic_measurements(state)
            case_path = self._case_path(state)
            orders = self._metadata(state).get("harmonic_orders") or _infer_harmonic_orders(
                measurements
            )
        except Exception as exc:
            return self._failure("harmonic_context_missing", f"{type(exc).__name__}: {exc}")
        buses = sorted({int(item.get("bus")) for item in measurements if item.get("bus") is not None})
        summary = summarize_harmonic_context_payload(
            {
                "case_path": case_path,
                "harmonic_measurements": measurements,
                "harmonic_orders": [int(order) for order in orders],
            }
        )
        summary.pop("case_path", None)
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:harmonic_measurements",
            "context_tool": GET_HARMONIC_CONTEXT,
            "finding_count": len(measurements),
            "harmonic_orders": [int(order) for order in orders],
            "measured_buses": buses,
            "harmonic_summary": summary,
        }

    def run_hse(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        try:
            measurements = self._harmonic_measurements(state)
            case_path = self._case_path(state)
            orders = self._metadata(state).get("harmonic_orders") or _infer_harmonic_orders(
                measurements
            )
        except Exception as exc:
            return self._failure("hse_runtime_missing", f"{type(exc).__name__}: {exc}")
        slack_bus = int(self._metadata(state).get("slack_bus", 0))
        payload = _run_hse_logic(case_path, measurements, [int(order) for order in orders], slack_bus)
        if not payload.get("success"):
            return self._failure("hse_failure", payload.get("error"))
        summary = summarize_hse_payload(payload)
        best_bus = payload.get("best_candidate_bus_1based")
        best_thd = summary.get("best_candidate_thd_percent")
        try:
            thd_value = float(best_thd)
        except (TypeError, ValueError):
            thd_value = math.nan
        accepted = bool(
            best_bus is not None
            and math.isfinite(thd_value)
            and thd_value >= self.harmonic_thd_threshold_percent
        )
        metrics = {
            **self._binding(state),
            "evidence_source": "deployment_diagnostic:harmonic_state_estimation",
            "best_candidate_bus_1based": best_bus,
            "hse_summary": summary,
            "diagnostic_acceptance": {
                "accepted": accepted,
                "null_hypothesis": "thd_below_operational_threshold",
                "thd_percent": thd_value if math.isfinite(thd_value) else None,
                "minimum_thd_percent": self.harmonic_thd_threshold_percent,
            },
        }
        if accepted:
            metrics["anomaly_explanation"] = {
                "family": "harmonic",
                "kind": "harmonic_source_localized",
                "detail": {
                    "bus_1based": int(best_bus),
                    "thd_percent": thd_value,
                },
            }
        return metrics

    def run_three_phase_nlm(
        self, state: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        metadata = self._metadata(state)
        diagnostic = metadata.get("nlm_diagnostic")
        pristine_dir = metadata.get("pristine_model_dir")
        faulted_dir = metadata.get("faulted_model_dir")
        unbalance_signal = self._has_family_signature(state, "three_phase_unbalance")
        hif_signal = self._has_family_signature(state, "hif")

        # Pure unbalance is a distinct terminal explanation, not an implicit
        # HIF localization.  Sequence-voltage evidence provides an explicit
        # balanced-system null test and never emits a candidate HIF branch.
        three_phase_voltages = metadata.get("three_phase_voltages")
        if unbalance_signal and not hif_signal and three_phase_voltages:
            vuf_evidence = _three_phase_vuf_evidence(
                three_phase_voltages, top_k=self.top_k
            )
            if not vuf_evidence:
                return self._failure(
                    "unbalance_voltage_evidence_invalid",
                    "three_phase_voltages contains no usable three-phase phasors",
                )
            max_vuf = float(vuf_evidence[0]["vuf"])
            accepted = max_vuf >= self.unbalance_vuf_threshold
            acceptance = {
                "accepted": accepted,
                "null_hypothesis": "balanced_three_phase_voltage",
                "max_vuf": max_vuf,
                "minimum_vuf": self.unbalance_vuf_threshold,
            }
            metrics: dict[str, Any] = {
                **self._binding(state),
                "evidence_source": "deployment_diagnostic:sequence_voltage_unbalance",
                "nlm_summary": {
                    "success": True,
                    "converged": True,
                    "method": "sequence_voltage_unbalance_test",
                    "diagnostic_classification": (
                        "three_phase_unbalance" if accepted else "unresolved"
                    ),
                    "top_hif_groups": [],
                    "top_vuf_buses": vuf_evidence,
                },
                "diagnostic_acceptance": acceptance,
            }
            if accepted:
                metrics["anomaly_explanation"] = {
                    "family": "three_phase_unbalance",
                    "kind": "voltage_unbalance_confirmed",
                    "detail": {
                        "max_vuf": max_vuf,
                        "minimum_vuf": self.unbalance_vuf_threshold,
                        "top_vuf_buses": vuf_evidence,
                    },
                }
            return metrics

        if not isinstance(diagnostic, Mapping) and not (pristine_dir and faulted_dir):
            return self._failure(
                "nlm_runtime_missing",
                "state metadata carries neither usable three-phase voltages, "
                "nlm_diagnostic, nor OpenDSS model dirs",
            )
        try:
            case_path = self._case_path(state)
        except Exception as exc:
            return self._failure("nlm_input_error", f"{type(exc).__name__}: {exc}")
        arguments = dict(action.get("arguments") or {})
        payload = _run_three_phase_nlm_logic(
            case_path=case_path,
            nlm_diagnostic=dict(diagnostic) if isinstance(diagnostic, Mapping) else None,
            target_branch_row0=arguments.get("target_branch_row0"),
            target_dss_element=arguments.get("target_dss_element"),
            pristine_model_dir=pristine_dir,
            faulted_model_dir=faulted_dir,
            phase=arguments.get("phase"),
            r_hif_ohm=arguments.get("r_hif_ohm"),
            load_scale=float(metadata.get("load_scale", 1.0)),
        )
        if not payload.get("success"):
            return self._failure("nlm_failure", payload.get("error"))
        summary = summarize_three_phase_nlm_payload(payload)
        if isinstance(diagnostic, Mapping) and not any(
            key in diagnostic
            for key in ("detected", "detected_top1", "detected_top3")
        ):
            # A sanitized release row carries ranked observable output but not
            # truth-relative localization labels.  Do not turn the runner's
            # compatibility default into a misleading negative observation.
            summary.pop("detected", None)
        metrics = {
            **self._binding(state),
            "evidence_source": "deployment_diagnostic:three_phase_nlm",
            "nlm_summary": summary,
        }
        if unbalance_signal and not hif_signal:
            classification = str(payload.get("diagnostic_classification") or "")
            accepted = bool(
                payload.get("classification_accepted")
                and classification == "three_phase_unbalance"
            )
            metrics["diagnostic_acceptance"] = {
                "accepted": accepted,
                "null_hypothesis": "hif_or_other_unexplained_three_phase_event",
                "diagnostic_classification": classification or None,
            }
            if accepted:
                metrics["anomaly_explanation"] = {
                    "family": "three_phase_unbalance",
                    "kind": "nlm_non_hif_unbalance_classified",
                    "detail": {"diagnostic_classification": classification},
                }
        return metrics

    def estimate_hif(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        arguments = dict(action.get("arguments") or {})
        if arguments.get("candidate_branch_row0") is None:
            return self._failure(
                "hif_target_missing",
                "estimate_hif_location_magnitude requires candidate_branch_row0",
            )
        try:
            alpha_grid_size, r_grid_size, _ = validate_hif_search_limits(
                alpha_grid_size=arguments.get(
                    "alpha_grid_size", self.hif_alpha_grid_size
                ),
                r_grid_size=arguments.get("r_grid_size", self.hif_r_grid_size),
                alpha_grid_size_max=self.hif_alpha_grid_size,
                r_grid_size_max=self.hif_r_grid_size,
            )
        except ValueError as exc:
            return self._failure("hif_search_budget_invalid", exc)
        metadata = self._metadata(state)
        runtime = metadata.get("hif_runtime")
        runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
        try:
            case_path = self._case_path(state)
            z_obs = runtime.get("z_obs") or self._measurements(state)
        except Exception as exc:
            return self._failure("hif_input_error", f"{type(exc).__name__}: {exc}")
        payload = _estimate_hif_location_magnitude_logic(
            case_path=case_path,
            candidate_branch_row0=int(arguments["candidate_branch_row0"]),
            candidate_phase=arguments.get("candidate_phase"),
            z_obs=[float(value) for value in z_obs],
            three_phase_voltages=runtime.get("three_phase_voltages"),
            pristine_model_dir=runtime.get("pristine_model_dir"),
            load_scale=float(runtime.get("load_scale", 1.0)),
            top_k=int(arguments.get("top_k", self.top_k)),
            alpha_grid_size=alpha_grid_size,
            r_grid_size=r_grid_size,
            r_hif_pu_min=float(arguments.get("r_hif_pu_min", 5.0)),
            r_hif_pu_max=float(arguments.get("r_hif_pu_max", 1000.0)),
        )
        if not payload.get("success"):
            return self._failure("hif_estimation_failure", payload.get("error"))
        summary = summarize_hif_parameter_estimate_payload(payload)
        acceptance = self._hif_diagnostic_acceptance(payload)
        metrics: dict[str, Any] = {
            **self._binding(state),
            "evidence_source": "deployment_diagnostic:hif_parameter_estimator",
            "hif_summary": summary,
            "diagnostic_acceptance": acceptance,
        }
        if acceptance["accepted"]:
            metrics["anomaly_explanation"] = {
                "family": "hif",
                "kind": "hif_model_accepted_over_null",
                "detail": {
                    "candidate_branch_row0": int(arguments["candidate_branch_row0"]),
                    "estimated": summary.get("estimated"),
                    "residual_reduction_vs_null": acceptance[
                        "residual_reduction_vs_null"
                    ],
                },
            }
        return metrics

    def estimate_hif_multiscan(
        self, state: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        arguments = dict(action.get("arguments") or {})
        if arguments.get("candidate_branch_row0") is None:
            return self._failure(
                "hif_target_missing",
                "multiscan HIF estimation requires candidate_branch_row0",
            )
        try:
            alpha_grid_size, r_grid_size, max_scans = validate_hif_search_limits(
                alpha_grid_size=arguments.get(
                    "alpha_grid_size", self.hif_alpha_grid_size
                ),
                r_grid_size=arguments.get("r_grid_size", self.hif_r_grid_size),
                max_scans=arguments.get("max_scans", self.hif_max_scans),
                alpha_grid_size_max=self.hif_alpha_grid_size,
                r_grid_size_max=self.hif_r_grid_size,
                max_scans_max=self.hif_max_scans,
            )
        except ValueError as exc:
            return self._failure("hif_search_budget_invalid", exc)
        assert max_scans is not None
        window = self._metadata(state).get("hif_scan_window")
        window = dict(window) if isinstance(window, Mapping) else {}
        scans = window.get("scans")
        if not isinstance(scans, Sequence) or not scans:
            return self._failure(
                "hif_scan_window_missing",
                "state metadata carries no hif_scan_window.scans for multiscan estimation",
            )
        try:
            case_path = self._case_path(state)
        except Exception as exc:
            return self._failure("hif_input_error", f"{type(exc).__name__}: {exc}")
        payload = _estimate_hif_location_magnitude_multiscan_logic(
            scan_window_path=str(window.get("scan_window_path") or state.get("state_id") or "scan_window"),
            candidate_branch_row0=int(arguments["candidate_branch_row0"]),
            scans=[dict(scan) for scan in scans],
            sigma_z=window.get("sigma_z"),
            case_path=case_path,
            candidate_phase=arguments.get("candidate_phase"),
            pristine_model_dir=window.get("pristine_model_dir"),
            resistance_mode=str(arguments.get("resistance_mode", "shared")),
            max_scans=max_scans,
            scan_selection=str(arguments.get("scan_selection", "information_greedy")),
            top_k=int(arguments.get("top_k", self.top_k)),
            alpha_grid_size=alpha_grid_size,
            r_grid_size=r_grid_size,
            r_hif_pu_min=float(arguments.get("r_hif_pu_min", 5.0)),
            r_hif_pu_max=float(arguments.get("r_hif_pu_max", 1000.0)),
            robust_loss=str(arguments.get("robust_loss", "soft_l1")),
            smoothness_lambda=float(arguments.get("smoothness_lambda", 0.10)),
        )
        if not payload.get("success"):
            return self._failure("hif_multiscan_failure", payload.get("error"))
        summary = summarize_hif_parameter_estimate_payload(payload)
        acceptance = self._hif_diagnostic_acceptance(payload)
        metrics: dict[str, Any] = {
            **self._binding(state),
            "evidence_source": "deployment_diagnostic:hif_multiscan_estimator",
            "hif_summary": summary,
            "diagnostic_acceptance": acceptance,
        }
        if acceptance["accepted"]:
            metrics["anomaly_explanation"] = {
                "family": "hif",
                "kind": "hif_model_accepted_over_null",
                "detail": {
                    "candidate_branch_row0": int(arguments["candidate_branch_row0"]),
                    "estimated": summary.get("estimated"),
                    "residual_reduction_vs_null": acceptance[
                        "residual_reduction_vs_null"
                    ],
                },
            }
        return metrics


__all__ = [
    "MatpowerDeploymentProviders",
    "matpower_case_differ",
    "measurement_index_map",
    "observable_parameter_initial_states",
]
