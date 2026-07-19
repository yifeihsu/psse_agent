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
parameter correction reads its repeated scans from state metadata
(``parameter_scans`` with ``z_scans`` and ``initial_states``); when scans are
absent the executor fails closed, which the environment records as a
collectable no-op learner state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
)

from mcp_server.matpower_server import (  # noqa: E402  (repo-root package)
    _load_python_case,
    _meas_correction_json,
    _param_correction_json,
    _wls_json,
)
from trace_protocol import (  # noqa: E402  (repo-root module)
    build_lambda_evidence,
    build_residual_evidence,
    chi2_threshold,
    summarize_measurement_correction_payload,
    summarize_parameter_correction_payload,
    summarize_wls_payload,
)


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

    # ------------------------------------------------------------------ wiring

    def env_kwargs(self) -> dict[str, Any]:
        """Keyword arguments wiring this bundle into ``TransactionalPSSEEnv``."""
        from psse_env.oracle import CandidateQualityOracle, ProcessValidityOracle

        return {
            "process_oracle": ProcessValidityOracle(executor_hydrated_corrections=True),
            "candidate_quality_oracle": CandidateQualityOracle(
                mode="deployment", case_differ=matpower_case_differ
            ),
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
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        os.makedirs(self.derived_case_dir, exist_ok=True)
        path = os.path.join(self.derived_case_dir, f"{tag}_{digest}.m")
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
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
    ) -> dict[str, Any] | None:
        """Observable target-progress evidence for a candidate verification.

        Derived solely from the candidate solve: a measurement target is fixed
        when every corrected index sits below the residual threshold; a branch
        target is fixed when its Lagrange multipliers sit below the multiplier
        threshold.  ``remaining_fault_count`` counts the non-target suspects
        still above threshold, which the deployment candidate oracle uses to
        separate partial from final acceptance.
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

        target_fixed = all(
            abs(float(residuals[index])) < self.residual_threshold
            for index in target_measurements
            if 0 <= index < len(residuals)
        ) and all(row_lambda(row0) < self.lambda_threshold for row0 in target_rows)
        remaining = sum(
            1
            for index, value in enumerate(residuals)
            if index not in target_measurements and abs(float(value)) >= self.residual_threshold
        ) + sum(
            1
            for row0 in range(nl)
            if row0 not in target_rows and row_lambda(row0) >= self.lambda_threshold
        )
        return {
            "target_fixed": bool(target_fixed),
            "target_progress": 1.0 if target_fixed else 0.0,
            "remaining_fault_count": int(remaining),
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
        metrics: dict[str, Any] = {
            **self._binding(state),
            "evidence_source": "deployment_wls:lagrangian_port",
            "converged": True,
            "power_flow_converged": True,
            "wls_objective": statistic,
            "chi_square_statistic": statistic,
            "chi_square_threshold": threshold,
            "chi_square_dof": dof,
            "max_normalized_residual": max_abs_residual,
            "anomaly_threshold": 1.0,
            "remaining_anomaly_score": statistic / threshold if threshold else None,
            "no_material_anomaly_remaining": bool(statistic < threshold),
            "wls_summary": summary,
        }
        source_action = state.get("source_action")
        if str(state.get("status") or "") == "candidate" and isinstance(source_action, Mapping):
            target_evidence = self._target_evidence(
                source_action,
                residuals,
                [float(value) for value in payload.get("lambdaN") or []],
                nl,
            )
            if target_evidence is not None:
                resolved = bool(statistic < threshold)
                metrics.update(target_evidence)
                metrics["physical_constraints_ok"] = True
                metrics["new_constraint_violations"] = 0
                metrics["post_action_resolved"] = resolved
                metrics["globally_resolved"] = resolved and target_evidence["target_fixed"]
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
        suspect_indices = sorted(int(item["index0"]) for item in evidence)
        supported: list[dict[str, Any]] = []
        if suspect_indices:
            supported.append(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {"state_id": state_id, "suspect_group": suspect_indices},
                }
            )
            for index in suspect_indices:
                if len(suspect_indices) > 1:
                    supported.append(
                        {
                            "tool": CORRECT_MEASUREMENTS,
                            "arguments": {"state_id": state_id, "suspect_group": [index]},
                        }
                    )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_residuals",
            "context_tool": GET_MEASUREMENT_CONTEXT,
            "finding_count": len(evidence),
            "measurement_findings": evidence,
            "supported_corrections": supported,
        }

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
        supported = [
            {
                "tool": CORRECT_PARAMETERS,
                "arguments": {"state_id": state_id, "line_index": row0 + 1},
            }
            for row0 in seen_rows
        ]
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_lagrange",
            "context_tool": GET_PARAMETER_CONTEXT,
            "finding_count": len(findings),
            "parameter_findings": findings,
            "supported_corrections": supported,
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
        supported: list[dict[str, Any]] = []
        seen_rows: set[int] = set()
        for item in findings:
            row0 = item.get("line_row0")
            if row0 is None or row0 in seen_rows or not 0 <= int(row0) < solved["nl"]:
                continue
            seen_rows.add(int(row0))
            current_status = int(float(branch[int(row0)][10])) if branch.shape[1] > 10 else 1
            supported.append(
                {
                    "tool": CORRECT_TOPOLOGY,
                    "arguments": {
                        "state_id": state_id,
                        "line_index": int(row0) + 1,
                        "status": 0 if current_status else 1,
                    },
                }
            )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_lagrange",
            "context_tool": GET_TOPOLOGY_CONTEXT,
            "finding_count": len(findings),
            "topology_findings": findings,
            "supported_corrections": supported,
        }

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
        if not isinstance(scans, Mapping) or not scans.get("z_scans") or not scans.get(
            "initial_states"
        ):
            return self._failure(
                "parameter_scans_missing",
                "multi-scan parameter correction requires metadata.parameter_scans "
                "with z_scans and initial_states",
            )
        try:
            payload = _param_correction_json(
                case_path,
                row0 + 1,
                [list(map(float, scan)) for scan in scans["z_scans"]],
                [list(map(float, scan)) for scan in scans["initial_states"]],
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


__all__ = ["MatpowerDeploymentProviders", "measurement_index_map"]
