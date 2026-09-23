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

import numpy as np

from hif_search_limits import validate_hif_search_limits
from psse_env.actions import (
    AMBIGUOUS_BRANCH_CANDIDATES_REQUEST,
    ANOMALY_FAMILY_MARKERS,
    PARAMETER_RANKING_AMBIGUITY_CANDIDATES,
    ambiguous_branch_candidate_lines,
    harmonic_screening_pending,
    three_phase_acquisition_pending,
    three_phase_screening_pending,
    waveform_anomaly_signatures,
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    GET_HARMONIC_CONTEXT,
    GET_THREE_PHASE_CONTEXT,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    HIF_CONDITIONING_UNAVAILABLE_REQUEST,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    unexplained_signatures,
)
from psse_env.state_store import apply_modification
from psse_env.noise_contract import resolve_state_measurement_noise, validate_shared_scada_covariance
from psse_env.evidence_profile import (
    AUXILIARY_EVIDENCE_PROFILE, DEFAULT_EVIDENCE_PROFILE, GATED_DIAGNOSTIC_TOOLS,
    SCADA_ONLY_PROFILE, STRICT_BOUNDARY_PROFILES, WLS_GATED_PROFILE,
    allows_diagnostic_tools, disabled_tools, is_strict_boundary,
    requires_wls_alarm_for_diagnostics, sanitize_gated_metadata,
    sanitize_gated_observation, sanitize_scada_metadata, sanitize_scada_observation,
    validate_evidence_profile,
)
from psse_env.oracle.process_validity import current_wls_alarm
from psse_env.providers.hif_continuation import (
    accepted_fit as accepted_hif_fit, conditioned_prediction, current_scan,
    diagnose as diagnose_hif_meters, fit_receipt, model_fingerprint,
)
from psse_env.oracle.measurement_recovery_evidence import (
    measurement_targets_predating_branch_repair,
)
from psse_env.oracle.expert_types import matching_evidence_codes

from three_phase_nlm.branch_current_analysis import (  # noqa: E402  (repo-root package)
    BRANCH_CURRENT_CHANNEL,
    BRANCH_CURRENT_SIGMA_KEY,
    DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    DEFAULT_UNBALANCE_VUF_THRESHOLD,
    branch_current_rows_to_phasors,
    line_differential_null_test,
    terminal_current_hif_localization,
    terminal_current_hif_localization_multiscan,
    unbalance_source_localization,
)
from three_phase_nlm.measurement_noise import DEFAULT_THREE_PHASE_SIGMA_PU  # noqa: E402
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
#: Signature minted when three-phase screening finds an HIF-like line
#: differential on a root no sensor had flagged; it carries the HIF family
#: marker so the existing estimator ladder takes over.
HIF_SCREENING_SIGNATURE = "hif_suspected_line_differential"
_ROUTE_ACTIONABLE = "actionable"
_ROUTE_COMPLETE_NEGATIVE = "complete_negative"
_ROUTE_UNAVAILABLE = "unavailable_or_inconclusive"
PARAMETER_RANKING_CONTRACT = "distinct_line_abs_lambda_dominance_v1"
PARAMETER_RANKING_DOMINANCE_THRESHOLD = 1.2


def parameter_ranking_contract_is_dominant(
    metrics: Mapping[str, Any],
    *,
    expected_threshold: float = PARAMETER_RANKING_DOMINANCE_THRESHOLD,
) -> bool:
    """Validate the policy-observable parameter-ranking dominance contract.

    ``expected_threshold`` is explicit so a collector can validate the same
    observable ranking contract at its reviewed release threshold without
    weakening the 1.2 default used for round-0 training admission.
    """

    if isinstance(expected_threshold, bool):
        return False
    try:
        normalized_expected_threshold = float(expected_threshold)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not math.isfinite(normalized_expected_threshold)
        or normalized_expected_threshold < 1.0
    ):
        return False

    if metrics.get("parameter_ranking_contract") != PARAMETER_RANKING_CONTRACT:
        return False
    raw_threshold = metrics.get("parameter_ranking_dominance_threshold")
    if isinstance(raw_threshold, bool):
        return False
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not math.isfinite(threshold)
        or not math.isclose(
            threshold,
            normalized_expected_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False

    raw_lines = metrics.get("parameter_ranking_distinct_lines")
    if not isinstance(raw_lines, (list, tuple)) or not raw_lines:
        return False
    ranked_lines: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in raw_lines:
        if not isinstance(item, Mapping):
            return False
        line_index1 = item.get("line_index1")
        score = item.get("abs_lambda_score")
        if (
            not isinstance(line_index1, int)
            or isinstance(line_index1, bool)
            or line_index1 <= 0
            or line_index1 in seen
            or isinstance(score, bool)
        ):
            return False
        try:
            numeric_score = float(score)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(numeric_score) or numeric_score <= 0.0:
            return False
        if ranked_lines and numeric_score > ranked_lines[-1][1]:
            return False
        seen.add(line_index1)
        ranked_lines.append((line_index1, numeric_score))

    raw_top = metrics.get("parameter_ranking_top_abs_lambda")
    if isinstance(raw_top, bool):
        return False
    try:
        top_score = float(raw_top)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isclose(
        top_score, ranked_lines[0][1], rel_tol=1e-12, abs_tol=1e-12
    ):
        return False

    singleton = len(ranked_lines) == 1
    if metrics.get("parameter_ranking_singleton") is not singleton:
        return False
    if singleton:
        if (
            metrics.get("parameter_ranking_runner_up_abs_lambda") is not None
            or metrics.get("parameter_ranking_dominance_ratio") is not None
        ):
            return False
        expected_dominant = True
    else:
        raw_runner_up = metrics.get("parameter_ranking_runner_up_abs_lambda")
        raw_ratio = metrics.get("parameter_ranking_dominance_ratio")
        if isinstance(raw_runner_up, bool) or isinstance(raw_ratio, bool):
            return False
        try:
            runner_up = float(raw_runner_up)
            ratio = float(raw_ratio)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not math.isfinite(runner_up)
            or runner_up <= 0.0
            or not math.isclose(
                runner_up, ranked_lines[1][1], rel_tol=1e-12, abs_tol=1e-12
            )
            or not math.isfinite(ratio)
            or not math.isclose(
                ratio, top_score / runner_up, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            return False
        expected_dominant = ratio >= threshold
    return (
        expected_dominant
        and metrics.get("parameter_ranking_dominant") is True
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


def _residual_breadth_metrics(
    residuals: Sequence[float],
    index_map: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Share of normalized residuals above ``threshold`` and the top block."""
    values = [abs(float(value)) for value in residuals]
    count = sum(1 for value in values if value > float(threshold))
    breadth = count / len(values) if values else 0.0
    dominant = None
    if values:
        top = max(range(len(values)), key=lambda index: values[index])
        for name, bounds in (index_map or {}).items():
            try:
                start, end = int(bounds[0]), int(bounds[1])
            except (TypeError, ValueError, IndexError):
                continue
            if start <= top < end:
                dominant = str(name)
                break
    return {
        "anomaly_breadth": float(breadth),
        "anomaly_breadth_count": int(count),
        "dominant_residual_block": dominant,
    }


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
        normalized_residual_threshold: float | None = None,
        breaker_candidate_count: int = 8,
        min_breaker_flip_progress: float = 0.5,
        derived_case_dir: str | None = None,
        max_correction_iterations: int = 2,
        error_tolerance: float = 1e-3,
        hif_alpha_grid_size: int = 31,
        hif_r_grid_size: int = 35,
        hif_max_scans: int = 10,
        hif_resistance_search: str = "physical_ohm",
        harmonic_thd_threshold_percent: float = 1.0,
        hse_min_sse_reduction: float = 0.5,
        # Shared with the scenario generator so the policy-visible
        # ``vuf_threshold_exceeded`` sensor signature and this gate agree.
        unbalance_vuf_threshold: float = DEFAULT_UNBALANCE_VUF_THRESHOLD,
        hif_min_residual_reduction: float = 0.20,
        hif_max_weighted_residual_norm: float = 3.0,
        hif_terminal_consistency_limit: float = 0.5,
        vm_bound_tolerance_pu: float = 0.005,
        branch_rate_tolerance_mva: float = 1e-6,
        parameter_ranking_dominance_threshold: float = (
            PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ),
        branch_first_partial: bool = False,
        screen_checkpoint: str | None = None,
        screen_calibration: str | None = None,
        evidence_profile: str = DEFAULT_EVIDENCE_PROFILE,
    ) -> None:
        self.evidence_profile = validate_evidence_profile(evidence_profile)
        if bool(screen_checkpoint) != bool(screen_calibration):
            raise ValueError("screen_checkpoint and screen_calibration must be provided together")
        if screen_checkpoint and is_strict_boundary(self.evidence_profile):
            # A learned screen is an additional signal by construction; the
            # strict profiles detect with balanced SCADA and WLS only.
            raise ValueError(
                f"evidence_profile={self.evidence_profile} refuses a learned WLS screen (screen_checkpoint)"
            )
        self.screen_checkpoint = str(screen_checkpoint) if screen_checkpoint else None
        self._hif_prediction_cache: dict[str, Any] = {}
        self.screen_calibration = str(screen_calibration) if screen_calibration else None
        if hif_resistance_search not in {"physical_ohm", "legacy_pu"}:
            raise ValueError("hif_resistance_search must be physical_ohm or legacy_pu")
        self.hif_resistance_search = hif_resistance_search
        self.top_k = int(top_k)
        # Research ablation: waive the branch partial-progress floor when the
        # branch target itself is resolved (see CandidateQualityOracle).
        self.branch_first_partial = bool(branch_first_partial)
        self.residual_threshold = float(residual_threshold)
        self.lambda_threshold = float(lambda_threshold)
        self.chi2_alpha = float(chi2_alpha)
        if not math.isfinite(self.chi2_alpha) or not 0.0 < self.chi2_alpha < 1.0:
            raise ValueError("chi2_alpha must be finite and strictly between 0 and 1")
        # Separate the system-level alarm from the 3-sigma candidate screening
        # cutoff. None preserves historical chi-square-only experiments.
        self.normalized_residual_threshold = (
            None if normalized_residual_threshold is None
            else float(normalized_residual_threshold)
        )
        if self.normalized_residual_threshold is not None and (
            not math.isfinite(self.normalized_residual_threshold)
            or self.normalized_residual_threshold <= 0.0
        ):
            raise ValueError("normalized_residual_threshold must be finite and positive")
        # How many top-ranked breakers of the node/breaker multiplier test are
        # confirmed by re-estimation and offered to the bus-branch screening.
        self.breaker_candidate_count = int(breaker_candidate_count)
        if self.breaker_candidate_count < 1:
            raise ValueError("breaker_candidate_count must be positive")
        # When no single breaker flip leaves the substation estimate clean (another
        # fault remains), a flip is still offered if it removes at least this
        # share of the node/breaker chi-square; the operator-model screening
        # then decides between a partial and a final repair.
        self.min_breaker_flip_progress = float(min_breaker_flip_progress)
        if not 0.0 < self.min_breaker_flip_progress <= 1.0:
            raise ValueError("min_breaker_flip_progress must lie in (0, 1]")
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
        # Minimum fraction of the measured harmonic voltage energy the fitted
        # single source must explain relative to the no-source null model.
        self.hse_min_sse_reduction = float(hse_min_sse_reduction)
        self.unbalance_vuf_threshold = float(unbalance_vuf_threshold)
        self.hif_min_residual_reduction = float(hif_min_residual_reduction)
        self.hif_max_weighted_residual_norm = float(hif_max_weighted_residual_norm)
        # Two-terminal self-consistency bound for accepting an HIF on the
        # strength of a detected differential current (see
        # ``_terminal_current_conclusive``).
        self.hif_terminal_consistency_limit = float(hif_terminal_consistency_limit)
        self.vm_bound_tolerance_pu = float(vm_bound_tolerance_pu)
        self.branch_rate_tolerance_mva = float(branch_rate_tolerance_mva)
        self.parameter_ranking_dominance_threshold = float(
            parameter_ranking_dominance_threshold
        )
        if self.vm_bound_tolerance_pu < 0.0 or self.branch_rate_tolerance_mva < 0.0:
            raise ValueError("Physical-bound tolerances must be non-negative.")
        if (
            not math.isfinite(self.parameter_ranking_dominance_threshold)
            or self.parameter_ranking_dominance_threshold < 1.0
        ):
            raise ValueError(
                "parameter_ranking_dominance_threshold must be finite and >= 1.0"
            )

    # ------------------------------------------------------------------ wiring

    def env_kwargs(self) -> dict[str, Any]:
        """Keyword arguments wiring this bundle into ``TransactionalPSSEEnv``."""
        from psse_env.oracle import ProcessValidityOracle

        return {
            "evidence_profile": self.evidence_profile,
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
                GET_THREE_PHASE_CONTEXT: self.get_three_phase_context,
                RUN_HSE_FROM_PATH: self.run_hse,
                RUN_THREE_PHASE_NLM_FROM_PATH: self.run_three_phase_nlm,
                ESTIMATE_HIF_FROM_PATH: self.estimate_hif,
                ESTIMATE_HIF_MULTISCAN_FROM_PATH: self.estimate_hif_multiscan,
            },
        }

    def _deployment_candidate_quality_oracle(self) -> Any:
        """Build the one deployment verdict policy used by screen and commit."""

        from psse_env.oracle import CandidateQualityOracle

        return CandidateQualityOracle(
            mode="deployment",
            case_differ=matpower_case_differ,
            case_loader=_load_python_case,
            branch_first_partial=self.branch_first_partial,
        )

    def request_additional_evidence(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Report exhaustion of the configured HIF diagnostic inventory.

        This provider does not declare the HIF absent or resolved.  It only
        reports that the controller has already invoked every configured HIF
        diagnostic for the current observable channel inventory.  The
        transactional environment separately checks the full bound history and
        the rejected acceptance tests before this report can end an episode.
        """
        state = self._evidence_state(state)
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        request = state.get("evidence_request")
        if request == HIF_CONDITIONING_UNAVAILABLE_REQUEST:
            condition = (observation.get("fresh_context_evidence") or {}).get("hif_conditioning") or {}
            if not (accepted_hif_fit(state) is not None and condition.get("status") == "unavailable"
                    and all(condition.get(key) == value for key, value in self._binding(state).items())):
                return self._failure("hif_conditioning_handoff_unbound")
            return {**self._binding(state), "request": request, "family": "hif",
                    "evidence_source": "deployment_diagnostic:hif_conditioning_unavailable",
                    "additional_evidence_available": False, "operator_review_required": True,
                    "hif_conditioning": condition}
        attempted = {
            str(signature).split(":", 1)[0]
            for signature in observation.get("tried_action_signatures") or []
        }
        if request == AMBIGUOUS_BRANCH_CANDIDATES_REQUEST:
            candidates = self._ambiguous_branch_candidates(observation)
            if candidates is None:
                return self._failure(
                    "ambiguous_branch_candidates_unsupported",
                    "the parameter route is not ambiguous on this state or its "
                    "ranked candidates have not all been rejected by verification",
                )
            available = {
                str(item) for item in observation.get("available_evidence") or []
            }
            return {
                **self._binding(state),
                "evidence_source": "deployment_diagnostic:ambiguous_branch_candidate_inventory",
                "request": AMBIGUOUS_BRANCH_CANDIDATES_REQUEST,
                "family": "ambiguous_branch",
                "candidate_lines": candidates,
                "additional_evidence_available": False,
                "operator_review_required": True,
                "attempted_tools": sorted(attempted),
                "available_evidence_channels": sorted(available),
            }
        if request in {
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        }:
            investigation_tools = {
                GET_MEASUREMENT_CONTEXT,
                GET_PARAMETER_CONTEXT,
                GET_TOPOLOGY_CONTEXT,
                # The telemetry requests are the operator's first
                # investigation after an anomaly; a budget handoff must be
                # able to close an episode whose budget ends right there.
                GET_HARMONIC_CONTEXT,
                GET_THREE_PHASE_CONTEXT,
                CORRECT_MEASUREMENTS,
                CORRECT_PARAMETERS,
                CORRECT_TOPOLOGY,
            }
            score = observation.get("remaining_anomaly_score")
            try:
                score_unresolved = score is not None and float(score) >= 1.0
            except (TypeError, ValueError):
                score_unresolved = False
            try:
                remaining_budget = int(observation.get("remaining_budget") or 0)
            except (TypeError, ValueError):
                remaining_budget = 0
            unresolved = observation.get("unresolved_signatures") or []
            post_correction_budget_deferral = bool(
                request == RECOVERY_BUDGET_EXHAUSTED_REQUEST
                and remaining_budget == 1
                and observation.get("accepted_corrections")
                and POST_CORRECTION_CONFIRMATION_SIGNATURE in unresolved
            )
            if not (
                (unresolved or score_unresolved)
                and "run_wls" in attempted
                and (
                    bool(attempted & investigation_tools)
                    or post_correction_budget_deferral
                )
            ):
                return self._failure(
                    "recovery_evidence_inventory_incomplete",
                    "observable WLS plus investigation history is required",
                )
            available = {
                str(item) for item in observation.get("available_evidence") or []
            }
            if request == RECOVERY_BUDGET_EXHAUSTED_REQUEST:
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
                    "post_correction_confirmation_deferred": (
                        post_correction_budget_deferral
                    ),
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

    _PROFILE_STRICTNESS = {
        SCADA_ONLY_PROFILE: 0, WLS_GATED_PROFILE: 1, AUXILIARY_EVIDENCE_PROFILE: 2,
    }

    def _effective_profile(self, state: Mapping[str, Any]) -> str:
        """Strictest of the provider's profile and the profile the state declares.

        A permissive historical provider can never loosen a stricter
        controller, and a strict provider stays strict whatever the state says.
        """
        observation = state.get("policy_observation") or {}
        declared = [self.evidence_profile]
        for value in (
            state.get("evidence_profile"),
            observation.get("evidence_profile") if isinstance(observation, Mapping) else None,
        ):
            if isinstance(value, str) and value in self._PROFILE_STRICTNESS:
                declared.append(value)
        return min(declared, key=self._PROFILE_STRICTNESS.__getitem__)

    def _strict_scada(self, state: Mapping[str, Any]) -> bool:
        """Balanced SCADA only: every auxiliary stream and tool is refused."""
        return self._effective_profile(state) == SCADA_ONLY_PROFILE

    def _strict_boundary(self, state: Mapping[str, Any]) -> bool:
        """No seeded signatures, truth handles, precomputed diagnoses or silent sigma defaults."""
        return self._effective_profile(state) in STRICT_BOUNDARY_PROFILES

    def _evidence_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Defend the provider boundary even when called without the controller.

        An auxiliary-capable historical provider cannot override a stricter
        controller. Case and raw SCADA readings are unchanged. Under scada_only
        simulated operating points and auxiliary streams never reach numerical
        routines; under wls_gated_diagnostics the auxiliary streams stay, but
        precomputed diagnoses, truth-side model handles and labels are removed.
        """
        profile = self._effective_profile(state)
        if profile not in STRICT_BOUNDARY_PROFILES:
            result = dict(state)
            result["evidence_profile"] = self.evidence_profile
            return result
        result = dict(state)
        result["evidence_profile"] = profile
        observation = state.get("policy_observation")
        if profile == WLS_GATED_PROFILE:
            result["metadata"] = sanitize_gated_metadata(state.get("metadata") or {})
            if isinstance(observation, Mapping):
                safe = sanitize_gated_observation(observation)
                safe["evidence_profile"] = profile
                result["policy_observation"] = safe
            return result
        result["metadata"] = sanitize_scada_metadata(state.get("metadata") or {})
        if isinstance(observation, Mapping):
            safe = sanitize_scada_observation(observation)
            safe["evidence_profile"] = SCADA_ONLY_PROFILE
            safe["unresolved_signatures"] = [str(s) for s in observation.get("unresolved_signatures") or []
                if str(s).startswith("wls_") or str(s) == POST_CORRECTION_CONFIRMATION_SIGNATURE]
            safe["explained_anomalies"] = []
            safe["available_evidence"] = []
            contexts = observation.get("fresh_context_evidence") or {}
            safe["fresh_context_evidence"] = {k: v for k, v in contexts.items()
                if k in {"wls", "measurement", "parameter", "topology"}}
            result["policy_observation"] = safe
        return result

    def _auxiliary_unavailable(self, state: Mapping[str, Any], tool: str) -> dict[str, Any] | None:
        """Refuse an auxiliary tool the effective profile does not admit right now."""
        profile = self._effective_profile(state)
        if profile == SCADA_ONLY_PROFILE:
            return self._failure("evidence_unavailable_in_scada_only_profile",
                "Only balanced SCADA measurements and their WLS-derived evidence are available.",
                **self._binding(state), evidence_source="deployment_evidence:scada_only_capability",
                evidence_profile=SCADA_ONLY_PROFILE, requested_tool=tool)
        if tool in disabled_tools(profile):
            return self._failure("tool_disabled_by_evidence_profile",
                f"{tool} is not provided under evidence_profile={profile}",
                **self._binding(state), evidence_source="deployment_evidence:profile_capability",
                evidence_profile=profile, requested_tool=tool)
        if requires_wls_alarm_for_diagnostics(profile) and tool in GATED_DIAGNOSTIC_TOOLS:
            # Provider-side copy of the process gate: the bound WLS ledger in
            # the observation must carry a chi-square or normalized-residual
            # alarm on this exact target state.  A direct call without an
            # observation has no alarm and is refused.
            observation = state.get("policy_observation")
            observation = observation if isinstance(observation, Mapping) else {}
            if not current_wls_alarm(observation, state.get("state_id")):
                return self._failure("diagnostics_require_wls_alarm",
                    f"{tool} requires a current balanced WLS alarm on the target state",
                    **self._binding(state), evidence_source="deployment_evidence:wls_alarm_gate",
                    evidence_profile=profile, requested_tool=tool)
        return None

    @staticmethod
    def _finite_positive(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    @classmethod
    def _declared_phasor_sigma(
        cls, metadata: Mapping[str, Any], key: str, channel: str, *extra_sources: Any,
    ) -> float | None:
        """Per-component sigma a phasor channel declares, or None.

        Looks at the metadata, the HIF acquisition block, the scan window and
        any extra source (a scan), then at each source's ``noise_contract``
        channel declaration.  Never substitutes a nominal sensor accuracy.
        """
        runtime = metadata.get("hif_runtime")
        window = metadata.get("hif_scan_window")
        sources = [source for source in (metadata, runtime, window, *extra_sources) if isinstance(source, Mapping)]
        for source in sources:
            value = cls._finite_positive(source.get(key))
            if value is not None:
                return value
        for source in sources:
            contract = source.get("noise_contract")
            channels = contract.get("channels") if isinstance(contract, Mapping) else None
            declaration = channels.get(channel) if isinstance(channels, Mapping) else None
            if not isinstance(declaration, Mapping):
                continue
            for field in ("estimator_sigma_per_component", "applied_sigma_per_component"):
                value = cls._finite_positive(declaration.get(field))
                if value is not None:
                    return value
        return None

    def _phasor_sigma(
        self, state: Mapping[str, Any], key: str, channel: str, legacy_default: float, *extra_sources: Any,
    ) -> float | None:
        """Declared sigma for a phasor channel; legacy default only outside strict profiles."""
        declared = self._declared_phasor_sigma(self._metadata(state), key, channel, *extra_sources)
        if declared is not None:
            return declared
        return None if self._strict_boundary(state) else float(legacy_default)

    @staticmethod
    def _sigma_undeclared(state: Mapping[str, Any], key: str, tool: str) -> dict[str, Any]:
        return {
            "execution_status": "failure",
            "error_code": f"{key}_undeclared",
            "error_detail": (
                f"{tool} requires the declared per-component {key} of the acquired phasors; "
                "no nominal sensor accuracy is substituted under a strict evidence profile"
            ),
            **{k: str(state[k]) for k in ("state_id", "state_hash") if state.get(k) is not None},
        }

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
    def _noise_options(state: Mapping[str, Any], channel_count: int) -> dict[str, Any]:
        contract = resolve_state_measurement_noise(state, channel_count)
        options: dict[str, Any] = {}
        if contract["measurement_sigma"] is not None:
            options["measurement_sigma"] = contract["measurement_sigma"]
        if contract["exact_measurement_indices"]:
            options["exact_measurement_indices"] = contract["exact_measurement_indices"]
        return options

    @classmethod
    def _correction_noise_options(cls, state: Mapping[str, Any], channel_count: int) -> dict[str, Any]:
        options = cls._noise_options(state, channel_count)
        sigma = options.pop("measurement_sigma", None)
        if sigma is not None:
            options["R_variances_full"] = np.square(sigma).tolist()
        return options

    @classmethod
    def _parameter_noise_options(cls, state: Mapping[str, Any], scans: Mapping[str, Any],
                                 z_scans: Sequence[Sequence[float]]) -> dict[str, Any]:
        options = cls._noise_options(state, len(z_scans[0]))
        sigma = scans.get("sigma_z", options.get("measurement_sigma"))
        if sigma is None:
            return {}
        validate_shared_scada_covariance(sigma, [{"z_obs": row} for row in z_scans])
        return {"R_variances_full": np.square(sigma).tolist()}

    @staticmethod
    def _binding(state: Mapping[str, Any]) -> dict[str, Any]:
        binding: dict[str, Any] = {}
        if state.get("state_id") is not None:
            binding["state_id"] = str(state["state_id"])
        if state.get("state_hash") is not None:
            binding["state_hash"] = str(state["state_hash"])
        return binding

    def _solve(self, state: Mapping[str, Any]) -> dict[str, Any]:
        state = self._evidence_state(state)
        case_path = self._case_path(state)
        z = self._measurements(state)
        ppc = _load_python_case(case_path)
        nb = int(ppc["bus"].shape[0])
        nl = int(ppc["branch"].shape[0])
        noise_options = self._noise_options(state, len(z))
        prediction = None if self._strict_scada(state) else conditioned_prediction(state, self._hif_prediction_cache)
        conditional = None
        wls_z = z
        if prediction is not None:
            conditional = diagnose_hif_meters(state, prediction, noise_options.get("measurement_sigma"))
            if self.normalized_residual_threshold is None:
                conditional["conditioning"]["status"] = "unavailable"
                conditional["conditioning"]["failure_reasons"].append("normalized_residual_test_not_configured")
            wls_z = (np.asarray(z) - np.asarray(prediction["measurement_effect"])).tolist()
        if self.screen_checkpoint:
            noise_options["include_screen_evidence"] = True
        payload = _wls_json(case_path, wls_z, **noise_options)
        return {
            "case_path": case_path,
            "z": z,
            "ppc": ppc,
            "nb": nb,
            "nl": nl,
            "index_map": measurement_index_map(nb, nl),
            "payload": payload,
            "hif_prediction": prediction,
            "hif_meter_diagnosis": conditional,
            "wls_measurements": wls_z,
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
        candidate_metadata: Mapping[str, Any] | None = None,
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
        breaker_only = False
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
                # A breaker whose flip splits or merges buses affects no single
                # branch row; its target is the recorded breaker status.
                if tool == CORRECT_TOPOLOGY and arguments.get("cb_name") is not None:
                    breaker_only = True
                else:
                    return None
        else:
            return None
        if not target_measurements and not target_rows and not breaker_only:
            return None

        def row_lambda(row0: int) -> float:
            values = [
                abs(float(lambda_values[index]))
                for index in (2 * row0, 2 * row0 + 1)
                if 0 <= index < len(lambda_values)
            ]
            return max(values, default=0.0)

        topology_status_matches: bool | None = None
        breaker_matches: bool | None = None
        structural_effect: str | None = None
        if tool == CORRECT_TOPOLOGY and breaker_only:
            from Transmission.ieee14_full_topology import parse_status

            requested_breaker = str(arguments["cb_name"]).strip()
            requested_status = arguments.get("status", arguments.get("expected_status"))
            if arguments.get("desired_status") is not None and requested_status is None:
                requested_status = arguments["desired_status"]
            try:
                requested_closed = parse_status(requested_status)
            except ValueError:
                return None
            metadata = candidate_metadata if isinstance(candidate_metadata, Mapping) else {}
            last = metadata.get("last_topology_correction")
            reported = metadata.get("reported_breaker_status")
            breaker_matches = bool(
                isinstance(last, Mapping)
                and str(last.get("cb_name") or "").strip() == requested_breaker
            )
            structural_effect = (
                str(last.get("breaker_effect") or "") if isinstance(last, Mapping) else None
            )
            recorded_closed: bool | None = None
            if isinstance(reported, Mapping) and reported.get(requested_breaker) is not None:
                try:
                    recorded_closed = parse_status(reported[requested_breaker])
                except ValueError:
                    recorded_closed = None
            topology_status_matches = bool(
                breaker_matches and recorded_closed is not None and recorded_closed == requested_closed
            )
        elif tool == CORRECT_TOPOLOGY:
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
            # A breaker-level correction is fixed only when the derived case
            # records that exactly the requested breaker produced the status.
            requested_breaker = arguments.get("cb_name")
            if requested_breaker is not None:
                last = (
                    candidate_metadata.get("last_topology_correction")
                    if isinstance(candidate_metadata, Mapping)
                    else None
                )
                breaker_matches = bool(
                    isinstance(last, Mapping)
                    and str(last.get("cb_name") or "").strip()
                    == str(requested_breaker).strip()
                )
                topology_status_matches = bool(topology_status_matches and breaker_matches)

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
            target_metric_kind = (
                "breaker_status_mismatch" if breaker_only else "branch_status_mismatch"
            )
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
        if tool == CORRECT_TOPOLOGY and breaker_only:
            # No single branch row carries a split or merge, so the branch
            # multiplier ambiguity test does not apply; the structural target
            # is the recorded breaker status in the rendered candidate.
            evidence.update(
                {
                    "topology_target_status_matches_requested": bool(topology_status_matches),
                    "topology_target_breaker": str(arguments["cb_name"]).strip(),
                    "topology_target_breaker_matches_requested": bool(breaker_matches),
                    "topology_target_effect": structural_effect,
                }
            )
            return evidence
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
            if arguments.get("cb_name") is not None:
                evidence["topology_target_breaker"] = str(arguments["cb_name"]).strip()
                evidence["topology_target_breaker_matches_requested"] = bool(breaker_matches)
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

    def _wls_detection_metrics(self, solved: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the configured global and local tests to one observable solve."""
        payload = solved["payload"]
        residuals = [float(value) for value in payload.get("r") or []]
        dof = int(payload.get("dof", max(1, len(residuals) - (2 * int(solved["nb"]) - 1))))
        statistic = float(payload.get("global_residual_sum") or 0.0)
        threshold = float(chi2_threshold(dof, self.chi2_alpha))
        if (not residuals or any(not math.isfinite(value) for value in residuals)
            or not math.isfinite(statistic) or statistic < 0.0
            or not math.isfinite(threshold) or threshold <= 0.0):
            raise ValueError("WLS anomaly evidence must be finite with nonempty residuals")
        maximum = max(abs(value) for value in residuals)
        residual_limit = self.normalized_residual_threshold
        chi_alarm = statistic >= threshold
        residual_alarm = residual_limit is not None and maximum >= residual_limit
        chi_ratio = statistic / threshold
        score = max(chi_ratio, maximum / residual_limit) if residual_limit is not None else chi_ratio
        return {
            "chi_square_statistic": statistic,
            "chi_square_threshold": threshold,
            "chi_square_dof": dof,
            "chi_square_alpha": self.chi2_alpha,
            "chi_square_ratio": chi_ratio,
            "chi_square_alarm": bool(chi_alarm),
            "max_normalized_residual": maximum,
            "normalized_residual_threshold": residual_limit,
            "normalized_residual_alarm": bool(residual_alarm),
            "anomaly_detection_rule": (
                "chi_square_or_normalized_residual" if residual_limit is not None
                else "chi_square_only"
            ),
            "anomaly_threshold": 1.0,
            "remaining_anomaly_score": score,
            "no_material_anomaly_remaining": not (chi_alarm or residual_alarm),
            "globally_resolved": not (chi_alarm or residual_alarm),
        }

    def _screen_wls(self, state: Mapping[str, Any], solved: Mapping[str, Any]) -> dict[str, Any]:
        """Optional learned evidence; never changes WLS/correction certificates."""
        from research.gnn_screen.protocol_adapter import load_screen, unavailable_report

        try:
            screen = load_screen(self.screen_checkpoint, self.screen_calibration)
            return screen.screen(
                solved["ppc"], solved["z"], wls_details=solved["payload"], **self._binding(state)
            )
        except Exception as exc:
            # A missing/incompatible artifact is unavailable, never a negative.
            return unavailable_report("model_unavailable", str(exc), **self._binding(state))

    def run_wls(self, state: Mapping[str, Any]) -> dict[str, Any]:
        state = self._evidence_state(state)
        try:
            solved = self._solve(state)
        except Exception as exc:
            extra = {}
            if accepted_hif_fit(state) is not None:
                extra["hif_conditioning"] = {**self._binding(state), "status": "unavailable",
                    "method": "paired_opendss_effect_compensation",
                    "failure_reasons": [str(exc)], "remaining_meter_candidate_indices": [],
                    "physical_fault_still_present": True}
            return self._failure("wls_input_error", f"{type(exc).__name__}: {exc}", **extra)
        payload = solved["payload"]
        if not payload.get("success"):
            screen_failure = {}
            if solved.get("hif_meter_diagnosis"):
                condition = dict(solved["hif_meter_diagnosis"]["conditioning"])
                condition.update(status="unavailable", failure_reasons=["conditioned_wls_solver_failure"])
                screen_failure["hif_conditioning"] = condition
            if self.screen_checkpoint:
                from research.gnn_screen.protocol_adapter import unavailable_report
                screen_failure["gnn_screen"] = unavailable_report(
                    "wls_failure", str(payload.get("error", "solver_failure")), **self._binding(state)
                )
            return self._failure(
                "wls_failure",
                payload.get("error", "solver_failure"),
                evidence_source="deployment_wls:lagrangian_port",
                **screen_failure,
            )
        residuals = [float(value) for value in payload.get("r") or []]
        nb, nl = solved["nb"], solved["nl"]
        try:
            detection = self._wls_detection_metrics(solved)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            extra = {}
            if solved.get("hif_meter_diagnosis"):
                condition = dict(solved["hif_meter_diagnosis"]["conditioning"])
                condition.update(status="unavailable", failure_reasons=["conditioned_wls_evidence_invalid"])
                extra["hif_conditioning"] = condition
            return self._failure("wls_evidence_error", str(exc), **extra)
        statistic = detection["chi_square_statistic"]
        threshold = detection["chi_square_threshold"]
        resolved = detection["no_material_anomaly_remaining"]
        conditional = solved.get("hif_meter_diagnosis")
        if conditional:
            ready = conditional["conditioning"]["status"] == "ready"
            pending = conditional["candidate_indices"]
            resolved = resolved and ready and not pending
            detection.update(no_material_anomaly_remaining=resolved, globally_resolved=resolved)
            detection["remaining_anomaly_score"] = max(
                detection["remaining_anomaly_score"],
                max((abs(v) / 5.0 for v in conditional["conditional_scores"]), default=0.0),
                1.0 if not ready else 0.0,
            )
        summary = summarize_wls_payload(
            {**payload, "global_residual_threshold": threshold},
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
        # While a waveform-level anomaly (harmonic distortion, three-phase
        # unbalance, or a suspected HIF) stands, the fundamental-frequency
        # solve's bad-data attributions are physically unreliable: the
        # chi-square elevation is (at least partly) the waveform event itself,
        # and "correcting" SCADA measurements against it would mask the true
        # anomaly.  This holds whether or not a diagnostic has explained the
        # signature: an explanation closes the episode's obligation, it does
        # not remove the event from the network.  The solve still reports its
        # metrics but mints no signatures while such a sensor signature stands.
        waveform_sensor = waveform_anomaly_signatures(preserved)
        if conditional and conditional["conditioning"]["status"] == "ready":
            waveform_sensor = [s for s in waveform_sensor if not matching_evidence_codes([s], *ANOMALY_FAMILY_MARKERS["hif"])]
        if not resolved and not waveform_sensor:
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
            **detection,
            # Breadth of the anomaly: the share of normalized residuals above
            # the outlier threshold, and the channel block of the largest one.
            # A waveform-distorted operator vector is inconsistent with the
            # balanced model almost everywhere (spectral distortion elevates
            # well over half the channels), whereas a load unbalance or a bad
            # meter stays narrow.  Observable, so the expert can choose which
            # additional measurement to request first without a family hint.
            **_residual_breadth_metrics(
                residuals, solved["index_map"], threshold=self.residual_threshold
            ),
            "unresolved_signatures": _dedupe(signatures),
            "wls_summary": summary,
        }
        if conditional:
            metrics["hif_conditioning"] = conditional["conditioning"]
            metrics["conditional_meter_scores"] = conditional["conditional_scores"]
            # The independent forward discrepancy supports sparse meter targets
            # even when balanced WLS absorbs part of their bias into its state.
            if conditional["conditioning"]["status"] == "ready":
                metrics["unresolved_signatures"] = _dedupe([
                    *metrics["unresolved_signatures"],
                    *[f"wls_residual_outlier_dominant conditioned_meter index={i}"
                      for i in conditional["candidate_indices"]],
                ])
        source_action = state.get("source_action")
        is_candidate = str(state.get("status") or "") == "candidate"
        if is_candidate and isinstance(source_action, Mapping):
            target_evidence = self._target_evidence(
                source_action,
                residuals,
                [float(value) for value in payload.get("lambdaN") or []],
                nl,
                candidate_case=solved["ppc"],
                candidate_metadata=self._metadata(state),
            )
            if target_evidence is not None:
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
        if self.screen_checkpoint:
            report = self._screen_wls(state, solved)
            metrics["gnn_screen"] = report
            if (report.get("screen_status") == "valid" and report.get("phase_trigger") is True
                and (not self._strict_scada(state) or not resolved)):
                # A request for evidence, not a diagnosed HIF/unbalance signature.
                # Existing acquisition and verification gates remain in force.
                metrics["unresolved_signatures"] = _dedupe([
                    *metrics["unresolved_signatures"], "wls_gnn_phase_investigation",
                ])
        return metrics

    # ----------------------------------------------------------------- contexts

    def get_measurement_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        state = self._evidence_state(state)
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("measurement_context_input_error", f"{type(exc).__name__}: {exc}")
        payload = solved["payload"]
        if not payload.get("success"):
            return self._failure("measurement_context_failure", payload.get("error"))
        residuals = [float(value) for value in payload.get("r") or []]
        conditional = solved.get("hif_meter_diagnosis")
        if conditional:
            condition = conditional["conditioning"]
            indices = conditional["candidate_indices"] if condition["status"] == "ready" else []
            return {
                **self._binding(state), "context_tool": GET_MEASUREMENT_CONTEXT,
                "evidence_source": "deployment_context:hif_conditioned_meter_residuals",
                "hif_conditioning": condition,
                "measurement_findings": build_residual_evidence(residuals, solved["index_map"], k=self.top_k, min_abs=self.residual_threshold),
                "conditional_meter_scores": conditional["conditional_scores"],
                "finding_count": len(indices),
                "supported_corrections": ([{"tool": CORRECT_MEASUREMENTS,
                    "arguments": {"state_id": str(state.get("state_id") or ""), "suspect_group": indices}}] if indices else []),
                **self._wls_detection_metrics(solved),
            }
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
        detection = self._wls_detection_metrics(solved)
        anomaly_unresolved = not detection["no_material_anomaly_remaining"]
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
            and not detection["normalized_residual_alarm"]
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
            and anomaly_unresolved
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
            and set(refinement_targets) <= measurement_targets_predating_branch_repair(observation)
            and anomaly_unresolved
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
            anomaly_unresolved=anomaly_unresolved,
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
        waveform_block = self._waveform_route_block(state)
        screening_pending = self._screening_pending(state)
        branch_dominance_block = self._branch_dominance_block(state)
        if waveform_block or screening_pending or branch_dominance_block:
            # Residual findings stay visible as evidence, but no meter
            # correction is offered: on a waveform-distorted operator vector
            # the residuals attribute the event itself, not a bad sensor;
            # until an unflagged anomaly has been screened against the
            # three-phase telemetry that possibility is still open; and while
            # branch evidence dominates the solve a meter correction can zero
            # the residuals of a wrong model and mask the branch fault, so the
            # meter route stays shut until both branch families have had a
            # hypothesis rejected by verification on this state.
            supported = []
            terminal_closure_action = None
            terminal_closure_targets = []
            terminal_closure_evidence = {}
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_residuals",
            "context_tool": GET_MEASUREMENT_CONTEXT,
            "fundamental_route_blocked_by_waveform_anomaly": waveform_block,
            "three_phase_screening_pending": screening_pending,
            "measurement_route_blocked_by_branch_dominance": branch_dominance_block,
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
            "normalized_residual_threshold": detection["normalized_residual_threshold"],
            "max_normalized_residual": detection["max_normalized_residual"],
            "normalized_residual_alarm": detection["normalized_residual_alarm"],
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
            # Progress stays a reduction in total weighted residual energy;
            # a different meter can dominate the maximum after a valid repair.
            candidate_score = verification.get("chi_square_ratio")
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
        state = self._evidence_state(state)
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("parameter_context_input_error", f"{type(exc).__name__}: {exc}")
        if not solved["payload"].get("success"):
            return self._failure("parameter_context_failure", solved["payload"].get("error"))
        findings = self._lambda_findings(solved)
        state_id = str(state.get("state_id") or "")
        ranked_lines: list[dict[str, Any]] = []
        seen_rows: set[int] = set()
        for item in findings:
            row0 = item.get("line_row0")
            if row0 is None:
                continue
            try:
                normalized_row0 = int(row0)
                abs_lambda_score = abs(float(item["value"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (
                normalized_row0 < 0
                or normalized_row0 in seen_rows
                or not math.isfinite(abs_lambda_score)
            ):
                continue
            seen_rows.add(normalized_row0)
            # ``build_lambda_evidence`` is already ordered by descending
            # absolute multiplier.  Keeping the first R-or-X entry encountered
            # for each physical line therefore produces a deterministic,
            # distinct-line ranking without discarding the remaining raw
            # per-parameter findings below.
            ranked_lines.append(
                {
                    "line_index1": normalized_row0 + 1,
                    "abs_lambda_score": abs_lambda_score,
                }
            )
        top_score = (
            float(ranked_lines[0]["abs_lambda_score"])
            if ranked_lines
            else None
        )
        runner_up_score = (
            float(ranked_lines[1]["abs_lambda_score"])
            if len(ranked_lines) > 1
            else None
        )
        singleton = len(ranked_lines) == 1
        dominance_ratio = (
            top_score / runner_up_score
            if (
                top_score is not None
                and runner_up_score is not None
                and runner_up_score > 0.0
            )
            else None
        )
        ranking_dominant = bool(ranked_lines) and (
            singleton
            or (
                dominance_ratio is not None
                and dominance_ratio
                >= self.parameter_ranking_dominance_threshold
            )
        )
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
        # When no line dominates, the top-ranked candidates are still the
        # only physically plausible parameter hypotheses.  Offer them in rank
        # order so the expert can test them under verification and, if every
        # one is rejected, hand the operator a diagnosis bounded to that set
        # rather than falling through to a masking meter correction.
        ranking_ambiguous = bool(scans_usable and ranked_lines and not ranking_dominant)
        candidate_rows = (
            list(ranked_lines)
            if ranking_dominant
            else list(ranked_lines[:PARAMETER_RANKING_AMBIGUITY_CANDIDATES])
            if ranking_ambiguous
            else []
        )
        supported = (
            [
                {
                    "tool": CORRECT_PARAMETERS,
                    "arguments": {
                        "state_id": state_id,
                        "line_index": int(item["line_index1"]),
                    },
                }
                for item in candidate_rows
            ]
            if scans_usable
            else []
        )
        candidate_lines = [int(item["line_index1"]) for item in candidate_rows]
        route_status = (
            _ROUTE_ACTIONABLE
            if supported
            else _ROUTE_COMPLETE_NEGATIVE
            if not findings
            else _ROUTE_UNAVAILABLE
        )
        waveform_block = self._waveform_route_block(state)
        screening_pending = self._screening_pending(state)
        if waveform_block or screening_pending:
            supported = []
            route_status = (
                _ROUTE_COMPLETE_NEGATIVE if waveform_block else _ROUTE_UNAVAILABLE
            )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_lagrange",
            "context_tool": GET_PARAMETER_CONTEXT,
            "fundamental_route_blocked_by_waveform_anomaly": waveform_block,
            "three_phase_screening_pending": screening_pending,
            "finding_count": len(findings),
            "parameter_findings": findings,
            "parameter_scans_available": scans_usable,
            "parameter_scan_count": scan_count,
            "parameter_ranking_contract": PARAMETER_RANKING_CONTRACT,
            "parameter_ranking_distinct_lines": ranked_lines,
            "parameter_ranking_top_abs_lambda": top_score,
            "parameter_ranking_runner_up_abs_lambda": runner_up_score,
            # A singleton has no finite runner-up.  ``None`` is the
            # JSON-safe representation of its mathematical infinite ratio;
            # the explicit singleton/dominant flags preserve that meaning.
            "parameter_ranking_dominance_ratio": dominance_ratio,
            "parameter_ranking_dominance_threshold": (
                self.parameter_ranking_dominance_threshold
            ),
            "parameter_ranking_singleton": singleton,
            "parameter_ranking_dominant": ranking_dominant,
            "parameter_ranking_ambiguous": ranking_ambiguous,
            "parameter_ranking_candidate_lines": candidate_lines,
            "supported_corrections": supported,
            "route_status": route_status,
            "route_status_reason": (
                "parameter_ranking_ambiguous_top_candidates"
                if route_status == _ROUTE_ACTIONABLE and ranking_ambiguous
                else "supported_parameter_candidates"
                if route_status == _ROUTE_ACTIONABLE
                else "no_parameter_findings"
                if route_status == _ROUTE_COMPLETE_NEGATIVE
                else "parameter_target_not_observably_dominant"
                if scans_usable and ranked_lines and not ranking_dominant
                else "parameter_findings_require_repeated_scans"
            ),
        }

    def get_topology_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        state = self._evidence_state(state)
        try:
            solved = self._solve(state)
        except Exception as exc:
            return self._failure("topology_context_input_error", f"{type(exc).__name__}: {exc}")
        if not solved["payload"].get("success"):
            return self._failure("topology_context_failure", solved["payload"].get("error"))
        binding = self._topology_binding(state)
        if binding is not None:
            if binding.get("error_code"):
                return self._failure(binding["error_code"], binding.get("error_detail"))
            return self._node_breaker_topology_context(state, solved, binding)
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
        # Explicit status-hypothesis enumeration for the direction the R/X
        # multiplier cannot see.  A branch modeled out of service contributes
        # zero parameter sensitivity, so a "modeled open / truly closed" error
        # never appears in ``findings``.  Every out-of-service row is therefore
        # screened as a close hypothesis by the same non-mutating WLS lookahead
        # that vets the multiplier-ranked open hypotheses.
        hypothesis_source: dict[int, str] = {
            int(action["arguments"]["line_index"]) - 1: "branch_multiplier_ranking"
            for action in proposed
        }
        enumerated_close_lines: list[int] = []
        if branch.shape[1] > 10:
            for row0 in range(int(solved["nl"])):
                if row0 in seen_rows or int(float(branch[row0][10])) != 0:
                    continue
                seen_rows.add(row0)
                enumerated_close_lines.append(row0 + 1)
                hypothesis_source[row0] = "out_of_service_status_enumeration"
                proposed.append(
                    {
                        "tool": CORRECT_TOPOLOGY,
                        "arguments": {
                            "state_id": state_id,
                            "line_index": row0 + 1,
                            "status": 1,
                        },
                    }
                )
        parent_score = self._wls_detection_metrics(solved)["chi_square_ratio"]
        supported: list[dict[str, Any]] = []
        candidate_screening: list[dict[str, Any]] = []
        for action in proposed:
            eligible, evidence = self._screen_topology_correction(
                state,
                action,
                parent_score=parent_score,
            )
            evidence["hypothesis_source"] = hypothesis_source.get(
                int(action["arguments"]["line_index"]) - 1
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
        waveform_block = self._waveform_route_block(state)
        screening_pending = self._screening_pending(state)
        if waveform_block or screening_pending:
            supported = []
            route_status = (
                _ROUTE_COMPLETE_NEGATIVE if waveform_block else _ROUTE_UNAVAILABLE
            )
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:wls_lagrange_candidate_screened",
            "context_tool": GET_TOPOLOGY_CONTEXT,
            "fundamental_route_blocked_by_waveform_anomaly": waveform_block,
            "three_phase_screening_pending": screening_pending,
            "finding_count": len(findings),
            "topology_findings": findings,
            "supported_corrections": supported,
            "proposed_correction_count": len(proposed),
            "screened_correction_count": len(candidate_screening),
            "topology_candidate_screening": candidate_screening,
            "enumerated_close_hypotheses": enumerated_close_lines,
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

    # ------------------------------------------------ node/breaker topology

    _NODE_BREAKER_MODEL: Any = None

    @classmethod
    def _node_breaker_model(cls):
        if cls._NODE_BREAKER_MODEL is None:
            from Transmission.ieee14_full_topology import build_full_topology

            cls._NODE_BREAKER_MODEL = build_full_topology()
        return cls._NODE_BREAKER_MODEL

    def _topology_binding(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        """The node/breaker model, reported breaker statuses and substation telemetry
        bound to a state, or None when the state carries no such channel.

        The electrical reference is the state's current case with every physical
        branch in service: line statuses are derived from breakers here, while an
        accepted parameter correction must keep its corrected impedances.
        """
        metadata = self._metadata(state)
        telemetry = metadata.get("substation_telemetry")
        reported = metadata.get("reported_breaker_status")
        if not isinstance(telemetry, Mapping) or not isinstance(reported, Mapping):
            return None
        from Transmission.ieee14_full_topology import MODEL_ID

        model = self._node_breaker_model()
        declared = telemetry.get("model_id") or metadata.get("topology_model_id")
        if declared is not None and str(declared) != MODEL_ID:
            return {
                "error_code": "topology_model_mismatch",
                "error_detail": f"state telemetry declares {declared!r}; provider model is {MODEL_ID!r}",
            }
        fingerprint = telemetry.get("model_fingerprint") or metadata.get("topology_model_fingerprint")
        if fingerprint is not None and str(fingerprint) != model.fingerprint():
            return {
                "error_code": "topology_model_mismatch",
                "error_detail": "state telemetry fingerprint differs from the provider's model",
            }
        try:
            reference = copy.deepcopy(_load_python_case(self._case_path(state)))
        except Exception as exc:
            return {
                "error_code": "topology_context_input_error",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
        if reference["branch"].shape[1] > 10:
            reference["branch"][:, 10] = 1.0
        meters: dict[int, str] = {}
        raw_meters = metadata.get("operator_voltage_meter_nodes")
        if isinstance(raw_meters, Mapping):
            for key, node in raw_meters.items():
                try:
                    meters[int(key)] = str(node)
                except (TypeError, ValueError):
                    continue
        return {
            "model": model,
            "reference": reference,
            "reported_labels": {str(name): str(value) for name, value in reported.items()},
            "telemetry": telemetry,
            "meter_nodes": meters,
        }

    def _synchronized_telemetry(
        self, state: Mapping[str, Any], binding: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Substation telemetry with the meters shared with the operator vector
        overwritten by the state's current measurements.

        The operator's bus voltage, bus injection and terminal-flow channels are
        readings of the same physical meters, so an accepted meter correction (or
        a gross meter error) must reach the node/breaker estimator through them.
        """
        telemetry = copy.deepcopy(dict(binding["telemetry"]))
        try:
            z = self._measurements(state)
        except ValueError:
            return telemetry
        model = binding["model"]
        nl = 20
        node_vm = dict(telemetry.get("node_vm") or {})
        node_p = dict(telemetry.get("node_pinj") or {})
        node_q = dict(telemetry.get("node_qinj") or {})
        # The operator layout: one (meter node, equipment nodes) pair per bus.
        # A rendered layout is recorded after a structural breaker correction;
        # otherwise the buses are the planning buses.
        layout = self._metadata(state).get("operator_layout")
        buses: list[tuple[str | None, list[str]]] = []
        if isinstance(layout, Mapping) and isinstance(layout.get("sections"), Mapping):
            count = int(layout.get("bus_count") or len(layout["sections"]))
            for i in range(count):
                section = layout["sections"].get(str(i + 1))
                if not isinstance(section, Mapping):
                    buses.append((None, []))
                    continue
                buses.append(
                    (
                        section.get("meter_node"),
                        [str(node) for node in (section.get("equipment_nodes") or [])],
                    )
                )
        else:
            for b in range(1, 15):
                equipment = sorted(
                    {model.equipment[table][b] for table in ("gen", "load")}
                )
                buses.append((binding["meter_nodes"].get(b), equipment))
        nb = len(buses)
        if len(z) != 3 * nb + 4 * nl:
            return telemetry
        for i, (meter, equipment) in enumerate(buses):
            if meter in node_vm and float(node_vm[meter]) > 0.0:
                node_vm[meter] = float(z[i])
            metered = [node for node in equipment if node in node_p]
            if len(metered) == 1:
                node_p[metered[0]] = float(z[nb + i])
                node_q[metered[0]] = float(z[2 * nb + i])
        telemetry["node_vm"] = node_vm
        telemetry["node_pinj"] = node_p
        telemetry["node_qinj"] = node_q
        base = 3 * nb
        telemetry["branch_pf"] = [float(v) for v in z[base : base + nl]]
        telemetry["branch_qf"] = [float(v) for v in z[base + nl : base + 2 * nl]]
        telemetry["branch_pt"] = [float(v) for v in z[base + 2 * nl : base + 3 * nl]]
        telemetry["branch_qt"] = [float(v) for v in z[base + 3 * nl : base + 4 * nl]]
        return telemetry

    @staticmethod
    def _breaker_effect(
        model: Any,
        reference: Mapping[str, Any],
        reported_labels: Mapping[str, str],
        cb_name: str,
        new_closed: bool,
    ) -> dict[str, Any]:
        """Bus-branch consequence of changing one reported breaker status.

        Only a change that isolates or reconnects exactly one line terminal maps
        onto the operator's 14-bus model as a branch status; anything else (a bus
        split or merge, an islanded bay, or no partition change) is reported by
        name and refused by the executor.
        """
        from Transmission.ieee14_full_substation import (
            injection_metered_nodes,
            status_map_from_labels,
        )

        old_map = status_map_from_labels(reported_labels)
        new_map = dict(old_map)
        new_map[cb_name] = bool(new_closed)
        old_groups = set(model.components(old_map))
        new_groups = set(model.components(new_map))
        if old_groups == new_groups:
            return {"effect": "equivalent"}
        branch = np.asarray(reference["branch"], dtype=float)
        bus = np.asarray(reference["bus"], dtype=float)
        row_of: dict[tuple[int, int], int] = {}
        for k in range(branch.shape[0]):
            f, t = int(branch[k, 0]), int(branch[k, 1])
            row_of[(f, t)] = k
            row_of[(t, f)] = k
        terminal_rows = {
            node: row_of[(f, t)]
            for (f, t), node in model.terminals.items()
            if (f, t) in row_of
        }
        equipment = set(injection_metered_nodes(model, reference))
        for b, node in model.equipment["shunt"].items():
            if bus[int(b) - 1, 4] != 0 or bus[int(b) - 1, 5] != 0:
                equipment.add(node)

        def contents(group):
            rows = sorted({terminal_rows[n] for n in group if n in terminal_rows})
            return rows, sorted(n for n in group if n in equipment)

        removed = old_groups - new_groups
        added = new_groups - old_groups
        touched = sorted(
            {model.nodes[n].planning_bus for g in (removed | added) for n in g}
        )
        if len(new_groups) > len(old_groups):
            if len(added) != 2 or len(removed) != 1:
                return {"effect": "bus_split", "affected_planning_buses": touched}
            parts = [contents(g) for g in added]
            minor = min(parts, key=lambda c: (len(c[0]) + len(c[1]), len(c[0])))
            if len(minor[0]) == 1 and not minor[1]:
                return {
                    "effect": "dangling_line_terminal",
                    "branch_row0": int(minor[0][0]),
                    "branch_status": 0,
                }
            if not minor[0]:
                return {
                    "effect": "unsupplied_island" if minor[1] else "empty_busbar",
                    "affected_planning_buses": touched,
                }
            return {"effect": "bus_split", "affected_planning_buses": touched}
        if len(new_groups) < len(old_groups):
            if len(removed) != 2 or len(added) != 1:
                return {"effect": "merge", "affected_planning_buses": touched}
            parts = [contents(g) for g in removed]
            minor = min(parts, key=lambda c: (len(c[0]) + len(c[1]), len(c[0])))
            if len(minor[0]) == 1 and not minor[1]:
                return {
                    "effect": "reconnect_line_terminal",
                    "branch_row0": int(minor[0][0]),
                    "branch_status": 1,
                }
            return {"effect": "merge", "affected_planning_buses": touched}
        return {"effect": "reassignment", "affected_planning_buses": touched}

    def _correct_breaker_status(
        self, state: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Set one reported breaker status and derive the operator's bus-branch case."""
        from Transmission.ieee14_full_topology import parse_status

        binding = self._topology_binding(state)
        if binding is None:
            return self._failure(
                "topology_correction_breaker_unsupported",
                "the state carries no node/breaker binding "
                "(substation telemetry and reported breaker statuses)",
            )
        if binding.get("error_code"):
            return self._failure(binding["error_code"], binding.get("error_detail"))
        cb_name = str(arguments.get("cb_name") or "").strip()
        labels = binding["reported_labels"]
        if cb_name not in labels:
            return self._failure(
                "topology_correction_unknown_breaker",
                f"{cb_name!r} is not a breaker of the bound model",
            )
        status = arguments.get("status", arguments.get("expected_status"))
        if arguments.get("desired_status") is not None and status is None:
            status = arguments["desired_status"]
        if status is None:
            return self._failure(
                "topology_correction_target_missing", "correct_topology requires a status"
            )
        try:
            new_closed = parse_status(status)
        except ValueError:
            return self._failure(
                "topology_correction_invalid_status",
                f"status must be 0/1 or open/closed, got {status!r}",
            )
        new_label = "closed" if new_closed else "open"
        try:
            current_closed = parse_status(labels[cb_name])
        except ValueError:
            return self._failure(
                "topology_correction_input_error",
                f"unreadable reported status for {cb_name}",
            )
        if current_closed == new_closed:
            return self._failure(
                "topology_correction_no_change", f"{cb_name} is already reported {new_label}"
            )
        model = binding["model"]
        effect = self._breaker_effect(model, binding["reference"], labels, cb_name, new_closed)
        if effect["effect"] in {"bus_split", "merge"}:
            return self._render_structural_breaker_correction(
                state, binding, arguments, cb_name, new_closed, new_label, effect
            )
        if effect["effect"] not in {"dangling_line_terminal", "reconnect_line_terminal"}:
            return self._failure(
                "topology_correction_unsupported_effect",
                f"setting {cb_name} {new_label} would {effect['effect'].replace('_', ' ')}; "
                "the operator model can carry an isolated or reconnected line "
                "terminal, a bus split or a bus merge, not an equipment outage",
                breaker_effect=effect["effect"],
            )
        row0 = int(effect["branch_row0"])
        branch_status = int(effect["branch_status"])
        requested_row = None
        for key in ("branch_row0", "line_index1", "line_index"):
            if arguments.get(key) is not None:
                try:
                    value = int(arguments[key])
                except (TypeError, ValueError):
                    return self._failure(
                        "topology_correction_input_error",
                        f"invalid {key}={arguments[key]!r}",
                    )
                requested_row = value if key == "branch_row0" else value - 1
        if requested_row is not None and requested_row != row0:
            return self._failure(
                "topology_correction_inconsistent_target",
                f"{cb_name} {new_label} affects line {row0 + 1}, not line {requested_row + 1}",
                breaker_effect=effect["effect"],
                line_index=row0 + 1,
            )
        try:
            case_path = self._case_path(state)
            ppc = _load_python_case(case_path)
        except Exception as exc:
            return self._failure(
                "topology_correction_input_error", f"{type(exc).__name__}: {exc}"
            )
        if ppc["branch"].shape[1] <= 10 or not 0 <= row0 < int(ppc["branch"].shape[0]):
            return self._failure(
                "topology_correction_unsupported", "case branch matrix cannot carry the status"
            )
        current_status = int(float(ppc["branch"][row0][10]))
        if current_status == branch_status:
            return self._failure(
                "topology_correction_no_change",
                f"branch row {row0} already has status {branch_status}",
            )
        updated = copy.deepcopy(ppc)
        updated["branch"][row0][10] = float(branch_status)
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", cb_name)
        derived_path = self._derived_case(updated, f"topo_{tag}_s{int(new_closed)}")
        new_labels = dict(labels)
        new_labels[cb_name] = new_label
        return {
            "modification": {
                "case": derived_path,
                "metadata_updates": {
                    "reported_breaker_status": new_labels,
                    "last_topology_correction": {
                        "line_index": row0 + 1,
                        "status": branch_status,
                        "cb_name": cb_name,
                        "cb_status": new_label,
                        "breaker_effect": effect["effect"],
                        "derived_case": derived_path,
                        "operator_layout_changed": False,
                    },
                },
            },
            "evidence_source": "deployment_correction:breaker_status",
            "line_index": row0 + 1,
            "previous_status": current_status,
            "new_status": branch_status,
            "cb_name": cb_name,
            "cb_previous_status": labels[cb_name],
            "cb_new_status": new_label,
            "breaker_effect": effect["effect"],
        }

    def _render_structural_breaker_correction(
        self,
        state: Mapping[str, Any],
        binding: Mapping[str, Any],
        arguments: Mapping[str, Any],
        cb_name: str,
        new_closed: bool,
        new_label: str,
        effect: Mapping[str, Any],
    ) -> dict[str, Any]:
        """A breaker flip that splits or merges buses re-renders the operator model.

        The topology processor contracts the new reported map into a bus-branch
        case whose bus count differs from the parent's, and the operator vector
        is re-read from the substation meters in that layout, so the candidate
        carries a new case and new measurements together with the layout the
        estimator needs to keep syncing shared meters.
        """
        from Transmission.ieee14_full_substation import (
            operator_model_from_map,
            operator_noise_for_layout,
            operator_vector_for_layout,
            status_map_from_labels,
        )

        if any(
            arguments.get(key) is not None
            for key in ("branch_row0", "line_index1", "line_index")
        ):
            return self._failure(
                "topology_correction_inconsistent_target",
                f"{cb_name} {new_label} would {effect['effect'].replace('_', ' ')} the "
                "operator model; it affects no single line",
                breaker_effect=effect["effect"],
            )
        labels = binding["reported_labels"]
        new_labels = dict(labels)
        new_labels[cb_name] = new_label
        try:
            current = _load_python_case(self._case_path(state))
        except Exception as exc:
            return self._failure(
                "topology_correction_input_error", f"{type(exc).__name__}: {exc}"
            )
        try:
            case, layout = operator_model_from_map(
                current, binding["model"], status_map_from_labels(new_labels), binding["meter_nodes"]
            )
            telemetry = self._synchronized_telemetry(state, binding)
            measurements = operator_vector_for_layout(telemetry, layout)
            operator_noise = operator_noise_for_layout(telemetry, layout)
            covariance = np.asarray(operator_noise["measurement_covariance"], dtype=float)
            if not np.array_equal(covariance, np.diag(np.diag(covariance))):
                raise ValueError("correlated physical meter covariance requires a full-covariance WLS solver")
            old_layout = self._metadata(state).get("operator_layout")
            if isinstance(old_layout, Mapping):
                source_measurements = np.asarray(self._measurements(state), dtype=float)
                projected_source = operator_vector_for_layout(telemetry, old_layout)
                old_noise = operator_noise_for_layout(telemetry, old_layout)
                new_rows = {identity: row for row, identity in enumerate(operator_noise["measurement_ids"])}
                # Single-meter updates already changed their physical source in
                # _synchronized_telemetry. An aggregate meter discrepancy cannot
                # be assigned to either equipment meter without new evidence.
                # Preserve it only when its exact source combination survives.
                for row in np.flatnonzero(np.abs(source_measurements - projected_source) > 1e-10):
                    identity = old_noise["measurement_ids"][row]
                    if identity not in new_rows:
                        raise ValueError(f"aggregate_meter_identity_ambiguous_after_topology_change:{identity}")
                    measurements[new_rows[identity]] += source_measurements[row] - projected_source[row]
        except (KeyError, ValueError) as exc:
            return self._failure(
                "topology_correction_telemetry_incomplete",
                f"{type(exc).__name__}: {exc}",
                breaker_effect=effect["effect"],
            )
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", cb_name)
        derived_path = self._derived_case(case, f"topo_{tag}_s{int(new_closed)}")
        return {
            "modification": {
                "case": derived_path,
                "measurements": [float(value) for value in measurements],
                "metadata_updates": {
                    "reported_breaker_status": new_labels,
                    "operator_layout": layout,
                    "substation_telemetry": telemetry,
                    "sigma_z": operator_noise["measurement_sigma"],
                    "operator_noise": operator_noise,
                    "structural_zero_indices": operator_noise["structural_zero_indices"],
                    "last_topology_correction": {
                        "cb_name": cb_name,
                        "cb_status": new_label,
                        "breaker_effect": effect["effect"],
                        "affected_planning_buses": list(effect.get("affected_planning_buses") or []),
                        "derived_case": derived_path,
                        "operator_bus_count": int(layout["bus_count"]),
                        "operator_layout_changed": True,
                    },
                },
            },
            "evidence_source": "deployment_correction:breaker_status",
            "cb_name": cb_name,
            "cb_previous_status": labels[cb_name],
            "cb_new_status": new_label,
            "breaker_effect": effect["effect"],
            "affected_planning_buses": list(effect.get("affected_planning_buses") or []),
            "operator_bus_count": int(layout["bus_count"]),
        }

    def _node_breaker_topology_context(
        self,
        state: Mapping[str, Any],
        solved: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Request the substation measurements, run the node/breaker estimator
        with normalized Lagrange multipliers, confirm the top-ranked breakers by
        re-estimation, and screen the confirmed breaker flips on the operator
        model exactly as line hypotheses are screened.
        """
        from Transmission.ieee14_full_gse import gse_topology_nlm, screen_breaker_flips
        from Transmission.ieee14_full_substation import status_map_from_labels

        model = binding["model"]
        reference = binding["reference"]
        labels = binding["reported_labels"]
        state_id = str(state.get("state_id") or "")
        findings = self._lambda_findings(solved)
        telemetry = self._synchronized_telemetry(state, binding)
        reported_map = status_map_from_labels(labels)
        estimate = gse_topology_nlm(model, reference, reported_map, telemetry)
        if not estimate.get("success"):
            return self._failure(
                "topology_context_gse_failure",
                "node/breaker estimator did not converge in "
                f"{estimate.get('iterations')} iterations",
            )
        gse_threshold = float(chi2_threshold(max(1, int(estimate["dof"])), self.chi2_alpha))
        top = estimate["ranking"][: self.breaker_candidate_count]
        flips = {
            item["cb_name"]: item
            for item in screen_breaker_flips(
                model, reference, reported_map, telemetry, [item["cb_name"] for item in top]
            )
        }
        breaker_findings: list[dict[str, Any]] = []
        admissible: list[tuple[dict[str, Any], dict[str, Any]]] = []
        reported_chi_square = float(estimate["chi_square"])
        for rank, item in enumerate(top, start=1):
            cb_name = item["cb_name"]
            flip = flips[cb_name]
            proposed_closed = flip["proposed_status"] == "closed"
            effect = self._breaker_effect(model, reference, labels, cb_name, proposed_closed)
            flip_threshold = float(
                chi2_threshold(max(1, int(flip["dof"])), self.chi2_alpha)
            )
            # A flip whose re-estimation diverges (an island, a singular KKT
            # system) reports a NaN chi-square; it explains nothing and must
            # not reach the model-visible context as a non-finite number.
            flip_converged = bool(flip["success"]) and math.isfinite(
                float(flip["chi_square"])
            )
            flip_chi_square = float(flip["chi_square"]) if flip_converged else None
            clean = bool(flip_converged and flip_chi_square < flip_threshold)
            progress = (
                (reported_chi_square - flip_chi_square) / reported_chi_square
                if flip_converged and reported_chi_square > 0.0
                else float("-inf")
            )
            structural = effect["effect"] in {"bus_split", "merge"}
            representable = structural or effect["effect"] in {
                "dangling_line_terminal",
                "reconnect_line_terminal",
            }
            finding = {
                "cb_name": cb_name,
                "rank": rank,
                "score": float(item["score"]),
                "normalized_multipliers": dict(item.get("multipliers") or {}),
                "reported_status": item["reported_status"],
                "proposed_status": flip["proposed_status"],
                "flip_estimate_converged": flip_converged,
                "gse_chi_square_after_flip": flip_chi_square,
                "gse_threshold_after_flip": flip_threshold,
                "gse_progress_after_flip": (
                    float(progress) if math.isfinite(progress) else None
                ),
                "flip_explains_substation_measurements": clean,
                "bus_branch_effect": effect["effect"],
            }
            if structural:
                finding["affected_planning_buses"] = list(
                    effect.get("affected_planning_buses") or []
                )
            elif representable:
                finding["line_index"] = int(effect["branch_row0"]) + 1
                finding["branch_status"] = int(effect["branch_status"])
            breaker_findings.append(finding)
            if representable and (clean or progress >= self.min_breaker_flip_progress):
                action_arguments = {
                    "state_id": state_id,
                    "cb_name": cb_name,
                    "status": int(proposed_closed),
                }
                if not structural:
                    action_arguments["line_index"] = int(effect["branch_row0"]) + 1
                admissible.append(
                    (finding, {"tool": CORRECT_TOPOLOGY, "arguments": action_arguments})
                )
        # A flip that leaves the substation estimate clean explains everything
        # and outranks any partial explanation.  Without one (another fault
        # remains), offer the flips that remove most of the anomaly, best first;
        # the operator-model screening decides between partial and final.
        clean_flips = [pair for pair in admissible if pair[0]["flip_explains_substation_measurements"]]

        def flip_chi_square_key(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
            value = pair[0]["gse_chi_square_after_flip"]
            return float(value) if value is not None else math.inf

        if clean_flips:
            chosen = sorted(clean_flips, key=flip_chi_square_key)
        else:
            chosen = sorted(admissible, key=flip_chi_square_key)
        proposed = [action for _, action in chosen]
        hypothesis_source: dict[str, str] = {
            finding["cb_name"]: (
                "node_breaker_nlm_ranking"
                if finding["flip_explains_substation_measurements"]
                else "node_breaker_nlm_partial_progress"
            )
            for finding, _ in chosen
        }
        parent_score = self._wls_detection_metrics(solved)["chi_square_ratio"]
        supported: list[dict[str, Any]] = []
        candidate_screening: list[dict[str, Any]] = []
        for action in proposed:
            eligible, evidence = self._screen_topology_correction(
                state, action, parent_score=parent_score
            )
            evidence["hypothesis_source"] = hypothesis_source.get(
                action["arguments"]["cb_name"]
            )
            candidate_screening.append(evidence)
            if eligible:
                supported.append(action)
        screening_incomplete = any(
            item.get("screening_complete") is not True for item in candidate_screening
        )
        route_status = (
            _ROUTE_UNAVAILABLE
            if screening_incomplete
            else _ROUTE_ACTIONABLE
            if supported
            else _ROUTE_COMPLETE_NEGATIVE
        )
        waveform_block = self._waveform_route_block(state)
        screening_pending = self._screening_pending(state)
        if waveform_block or screening_pending:
            supported = []
            route_status = (
                _ROUTE_COMPLETE_NEGATIVE if waveform_block else _ROUTE_UNAVAILABLE
            )
        inventory = {
            "node_voltages": len(telemetry.get("node_vm") or {}),
            "injection_meters": len(telemetry.get("node_pinj") or {}),
            "terminal_flow_channels": 4 * len(telemetry.get("branch_pf") or []),
            "breaker_flow_channels": 2 * len(telemetry.get("cb_p") or {}),
        }
        return {
            **self._binding(state),
            "evidence_source": "deployment_context:node_breaker_nlm_candidate_screened",
            "context_tool": GET_TOPOLOGY_CONTEXT,
            "fundamental_route_blocked_by_waveform_anomaly": waveform_block,
            "three_phase_screening_pending": screening_pending,
            "substation_measurements_requested": True,
            "substation_measurement_inventory": inventory,
            "topology_model_id": str(telemetry.get("model_id") or ""),
            "node_breaker_estimate": {
                "method": "generalized_state_estimation_normalized_lagrange_multipliers",
                "chi_square": float(estimate["chi_square"]),
                "chi_square_threshold": gse_threshold,
                "dof": int(estimate["dof"]),
                "anomalous": bool(estimate["chi_square"] >= gse_threshold),
                "iterations": int(estimate["iterations"]),
                "n_measurements": int(estimate["n_measurements"]),
                "n_constraints": int(estimate["n_constraints"]),
                "n_dependent_constraints": int(estimate.get("n_dependent_constraints", 0)),
                "max_normalized_residual": float(estimate.get("max_normalized_residual", 0.0)),
            },
            "breaker_findings": breaker_findings,
            "breaker_candidate_count": int(self.breaker_candidate_count),
            "finding_count": len(findings),
            "topology_findings": findings,
            "supported_corrections": supported,
            "proposed_correction_count": len(proposed),
            "screened_correction_count": len(candidate_screening),
            "topology_candidate_screening": candidate_screening,
            "enumerated_close_hypotheses": [],
            "islanding_filtered_lines": [],
            "route_status": route_status,
            "route_status_reason": (
                "candidate_screening_incomplete"
                if route_status == _ROUTE_UNAVAILABLE
                else "supported_topology_candidates"
                if route_status == _ROUTE_ACTIONABLE
                else "all_breaker_hypotheses_observably_rejected"
                if breaker_findings
                else "no_topology_findings"
            ),
        }

    def _remaining_anomaly_score(self, solved: Mapping[str, Any]) -> float | None:
        """Return the same combined alarm score emitted by ``run_wls``."""

        try:
            return self._wls_detection_metrics(solved)["remaining_anomaly_score"]
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

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
        if arguments.get("cb_name") is not None:
            evidence["cb_name"] = str(arguments["cb_name"]).strip()
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
            if arguments.get("cb_name") is not None:
                screen_suffix += ":" + str(arguments["cb_name"]).strip()
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

            candidate_score = verification.get("chi_square_ratio")
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
        state = self._evidence_state(state)
        arguments = dict(action.get("arguments") or {})
        if not self._strict_scada(state) and accepted_hif_fit(state) is not None:
            try:
                solved = self._solve(state)
                conditional = solved["hif_meter_diagnosis"]
                group = arguments.get("suspect_group") or []
                indices = [int(i) for i in group]
                if (not indices or arguments.get("measurement_updates")
                    or conditional["conditioning"]["status"] != "ready"
                    or not set(indices).issubset(conditional["candidate_indices"])):
                    return self._failure("hif_conditioned_measurement_target_unsupported")
                present = solved["hif_prediction"]["predicted_hif_measurements"]
                return {
                    "modification": {"measurement_updates": {i: float(present[i]) for i in indices}},
                    "evidence_source": "deployment_correction:hif_present_prediction",
                    "suspect_group": sorted(indices), "applied_any_correction": True,
                    "hif_conditioning": conditional["conditioning"],
                    "physical_fault_still_present": True,
                }
            except Exception as exc:
                return self._failure("hif_conditioned_measurement_correction_error", exc)
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
                **self._correction_noise_options(state, len(z)),
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
        state = self._evidence_state(state)
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
                **self._parameter_noise_options(state, scans, z_scans),
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
        state = self._evidence_state(state)
        arguments = dict(action.get("arguments") or {})
        if arguments.get("cb_name") is not None:
            return self._correct_breaker_status(state, arguments)
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
        # Branch status multiplies the series admittance in the estimator's
        # model, so anything but exactly 0 or 1 would scale or reverse the
        # branch instead of opening/closing it.  Fail closed.
        try:
            status_value = float(status)
        except (TypeError, ValueError, OverflowError):
            return self._failure(
                "topology_correction_invalid_status", f"status must be 0 or 1, got {status!r}"
            )
        if not math.isfinite(status_value) or status_value not in (0.0, 1.0):
            return self._failure(
                "topology_correction_invalid_status", f"status must be 0 or 1, got {status!r}"
            )
        new_status = int(status_value)
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
    def _branch_dominance_block(cls, state: Mapping[str, Any]) -> bool:
        """Branch evidence dominates and both branch routes are still open.

        Mirrors the measurement expert's stand-down rule as an environment
        contract: the meter route is not actionable while the solve is
        branch-multiplier dominant (and not residual-outlier dominant) unless
        a parameter hypothesis and a topology hypothesis have each been
        rejected by verification on the active state.
        """
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else state
        signatures = [str(item) for item in cls._observable_signatures(state)]
        branch_dominant = any("wls_branch_multiplier_dominant" in item for item in signatures)
        meter_dominant = any("wls_residual_outlier_dominant" in item for item in signatures)
        if not branch_dominant or meter_dominant:
            return False
        active_id = str(observation.get("active_state_id") or state.get("state_id") or "")
        rejected_families: set[str] = set()
        for record in observation.get("rejected_hypotheses") or []:
            if not isinstance(record, Mapping):
                continue
            if record.get("rejection_kind") == "executor_failure":
                # This gate requires physical candidate verification. A
                # numerical execution failure cannot supply that evidence.
                continue
            parent = record.get("candidate_parent_id")
            if parent is not None and active_id and str(parent) != active_id:
                continue
            source = record.get("source_action")
            tool = source.get("tool") if isinstance(source, Mapping) else None
            if tool == CORRECT_PARAMETERS:
                rejected_families.add("parameter")
            elif tool == CORRECT_TOPOLOGY:
                rejected_families.add("topology")
        return not {"parameter", "topology"} <= rejected_families

    @staticmethod
    def _ambiguous_branch_candidates(
        observation: Mapping[str, Any]
    ) -> list[int] | None:
        """Ranked parameter candidates that were all tested and rejected here."""
        return ambiguous_branch_candidate_lines(observation)

    @classmethod
    def _waveform_route_block(cls, state: Mapping[str, Any]) -> list[str]:
        """Waveform-family signatures that block fundamental-frequency routes.

        Explained or not: the event is still on the network, so residual and
        multiplier evidence cannot be attributed to a meter or a branch.
        """
        if not allows_diagnostic_tools(cls._state_profile(state)):
            return []
        return waveform_anomaly_signatures(cls._observable_signatures(state))

    @staticmethod
    def _state_profile(state: Mapping[str, Any]) -> str:
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        return str(state.get("evidence_profile") or observation.get("evidence_profile") or DEFAULT_EVIDENCE_PROFILE)

    @classmethod
    def _screening_pending(cls, state: Mapping[str, Any]) -> bool:
        """Unflagged WLS anomaly with acquisition or screening still pending."""
        profile = cls._state_profile(state)
        if not allows_diagnostic_tools(profile):
            return False
        observation = state.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        if harmonic_screening_pending(
            unresolved=observation.get("unresolved_signatures") or [],
            tried_action_signatures=observation.get("tried_action_signatures") or [],
            active_state_id=observation.get("active_state_id") or state.get("state_id"),
            context_evidence=observation.get("fresh_context_evidence"),
            evidence_profile=profile,
        ):
            return True
        if three_phase_acquisition_pending(
            unresolved=observation.get("unresolved_signatures") or [],
            tried_action_signatures=observation.get("tried_action_signatures") or [],
            active_state_id=observation.get("active_state_id") or state.get("state_id"),
            context_evidence=observation.get("fresh_context_evidence"),
            evidence_profile=profile,
        ):
            return True
        return three_phase_screening_pending(
            unresolved=observation.get("unresolved_signatures") or [],
            available_evidence=observation.get("available_evidence") or [],
            tried_action_signatures=observation.get("tried_action_signatures") or [],
            active_state_id=observation.get("active_state_id") or state.get("state_id"),
            context_evidence=observation.get("fresh_context_evidence"),
            evidence_profile=profile,
        )

    @classmethod
    def _has_family_signature(cls, state: Mapping[str, Any], family: str) -> bool:
        return any(
            _matches_any_marker(signature, ANOMALY_FAMILY_MARKERS[family])
            for signature in cls._observable_signatures(state)
        )

    def _branch_current_channel(
        self, state: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]] | None, float | None]:
        """Per-phase branch-current telemetry and its declared sigma, if present.

        The channel may sit at the metadata top level (unbalance rows) or inside
        ``hif_runtime`` (HIF rows); the declared sigma follows the same lookup
        and the noise contract.  Outside the strict profiles an undeclared
        sigma falls back to the legacy nominal sensor accuracy; under a strict
        profile it is ``None`` and the caller fails closed.
        """
        metadata = self._metadata(state)
        runtime = metadata.get("hif_runtime")
        runtime = runtime if isinstance(runtime, Mapping) else {}
        rows = metadata.get(BRANCH_CURRENT_CHANNEL) or runtime.get(BRANCH_CURRENT_CHANNEL)
        sigma = self._phasor_sigma(
            state, BRANCH_CURRENT_SIGMA_KEY, BRANCH_CURRENT_CHANNEL, DEFAULT_BRANCH_CURRENT_SIGMA_PU
        )
        if not rows or not branch_current_rows_to_phasors(rows):
            return None, sigma
        return [dict(item) for item in rows if isinstance(item, Mapping)], sigma

    #: Most recent multi-scan HIF searches, keyed by their complete inputs.
    #: The search is deterministic and costs tens of seconds, and a learner
    #: that loops on a state repeats it with identical inputs.
    _HIF_MULTISCAN_MEMO_LIMIT = 16

    def _memoized_hif_multiscan(self, **kwargs: Any) -> dict[str, Any]:
        memo = getattr(self, "_hif_multiscan_memo", None)
        if memo is None:
            memo = {}
            self._hif_multiscan_memo = memo
        try:
            key = hashlib.sha256(
                json.dumps([kwargs, model_fingerprint(kwargs.get("pristine_model_dir"))], sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
        except (TypeError, ValueError):
            return _estimate_hif_location_magnitude_multiscan_logic(**kwargs)
        cached = memo.get(key)
        if cached is None:
            cached = _estimate_hif_location_magnitude_multiscan_logic(**kwargs)
            if isinstance(cached, Mapping) and cached.get("success"):
                if len(memo) >= self._HIF_MULTISCAN_MEMO_LIMIT:
                    memo.pop(next(iter(memo)))
                memo[key] = cached
            return cached
        return copy.deepcopy(cached)

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
        terminal = payload.get("terminal_current_estimate")
        terminal = terminal if isinstance(terminal, Mapping) else {}
        estimated = payload.get("estimated")
        estimated = estimated if isinstance(estimated, Mapping) else {}
        terminal_conclusive, terminal_reason = self._terminal_current_conclusive(
            terminal, estimated
        )
        residual_ok = bool(
            math.isfinite(residual_value)
            and residual_value <= self.hif_max_weighted_residual_norm
        )
        improvement_ok = bool(
            math.isfinite(improvement_value)
            and improvement_value >= self.hif_min_residual_reduction
        )
        accepted = bool(
            residual_ok
            and not model_mismatch
            and not payload.get("synthetic_oracle", False)
            and (improvement_ok or terminal_conclusive)
        )
        if not accepted:
            basis = None
        elif improvement_ok:
            basis = "residual_reduction_vs_null"
        else:
            basis = "terminal_current_differential"
        return {
            "accepted": accepted,
            "acceptance_basis": basis,
            "null_hypothesis": "no_hif_base_model",
            "residual_reduction_vs_null": (
                improvement_value if math.isfinite(improvement_value) else None
            ),
            "weighted_residual_norm": residual_value if math.isfinite(residual_value) else None,
            "minimum_residual_reduction": self.hif_min_residual_reduction,
            "maximum_weighted_residual_norm": self.hif_max_weighted_residual_norm,
            "model_mismatch_suspected": model_mismatch,
            "terminal_current_conclusive": terminal_conclusive,
            "terminal_current_reason": terminal_reason,
            "terminal_consistency_limit": self.hif_terminal_consistency_limit,
        }

    def _terminal_current_conclusive(
        self, terminal: Mapping[str, Any], estimated: Mapping[str, Any]
    ) -> tuple[bool, str]:
        """Is the two-terminal differential evidence conclusive on its own?

        The residual-reduction gate compares whole-vector fits and is diluted
        by sensor noise on hundreds of unaffected entries.  A differential
        current on the candidate line that clears the six-sigma floor *is* the
        fault current; it is conclusive when the closed form also finds a
        positive, predominantly resistive fault impedance, the fault-point
        voltages computed from both ends agree to within the sensor-noise
        share of the line drop, and the model search agrees on the phase.  A
        bad current sensor produces a differential too, but not one that any
        (alpha, R) reconciles from both terminals.
        """
        if not terminal:
            return False, "no_terminal_current_evidence"
        if not bool(terminal.get("differential_detected")):
            return False, "differential_below_detection_floor"
        try:
            r_value = float(terminal.get("r_hif_pu"))
            ratio = float(terminal.get("consistency_ratio"))
            x_value = float(terminal.get("x_hif_pu") or 0.0)
            alpha = float(terminal.get("alpha_from_from_bus"))
        except (TypeError, ValueError):
            return False, "terminal_estimate_incomplete"
        if not (math.isfinite(r_value) and r_value > 0.0):
            return False, "nonpositive_fault_resistance"
        # A fault fitted at a terminal is what a gross error on that
        # terminal's current sensor also looks like; the line's own phasors
        # cannot separate the two, so this evidence alone is not conclusive.
        if bool(terminal.get("endpoint_ambiguous")) or not (
            math.isfinite(alpha) and 0.02 < alpha < 0.98
        ):
            return False, "fault_at_terminal_ambiguous_with_sensor_error"
        if not math.isfinite(ratio) or ratio > self.hif_terminal_consistency_limit:
            return False, "two_terminal_inconsistent"
        if abs(x_value) > 0.5 * r_value:
            return False, "fault_impedance_not_resistive"
        phase = terminal.get("phase")
        model_phase = estimated.get("phase")
        if phase and model_phase and str(phase).upper() != str(model_phase).upper():
            return False, "phase_disagreement_with_model_search"
        return True, "detected_consistent_resistive_differential"

    def get_three_phase_context(
        self, state: Mapping[str, Any], action: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        unavailable = self._auxiliary_unavailable(state, GET_THREE_PHASE_CONTEXT)
        if unavailable is not None:
            return unavailable
        from mcp_server.matpower_server import _get_three_phase_context_logic

        state = self._evidence_state(state)
        metadata = self._metadata(state)
        runtime = metadata.get("hif_runtime")
        runtime = runtime if isinstance(runtime, Mapping) else {}
        metrics = _get_three_phase_context_logic(
            case_path=self._case_path(state),
            three_phase_voltages=(
                metadata.get("three_phase_voltages") or runtime.get("three_phase_voltages")
            ),
            three_phase_branch_currents=(
                metadata.get(BRANCH_CURRENT_CHANNEL) or runtime.get(BRANCH_CURRENT_CHANNEL)
            ),
        )
        metrics.pop("case_path", None)
        return {**self._binding(state), **metrics}

    def get_harmonic_context(
        self, state: Mapping[str, Any], action: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        unavailable = self._auxiliary_unavailable(state, GET_HARMONIC_CONTEXT)
        if unavailable is not None:
            return unavailable
        state = self._evidence_state(state)
        # Requesting a measurement channel does not imply that it exists or
        # that the unknown event is harmonic. Availability is learned here,
        # after WLS, rather than advertised from hidden scenario metadata.
        if not self._metadata(state).get("harmonic_measurements"):
            return {
                **self._binding(state),
                "evidence_source": "deployment_context:harmonic_measurements",
                "context_tool": GET_HARMONIC_CONTEXT,
                "harmonic_context_status": "unavailable",
                "available_evidence_channels": [],
                "harmonic_distortion_detected": False,
                "finding_count": 0,
                "harmonic_summary": {
                    "measurement_count": 0,
                    "note": "No spectral measurements were returned by the measurement request; the cause remains unknown.",
                },
            }
        try:
            measurements = self._harmonic_measurements(state)
            case_path = self._case_path(state)
            orders = self._metadata(state).get("harmonic_orders") or _infer_harmonic_orders(
                measurements
            )
            observed = self._measurements(state)
            nb = int(_load_python_case(case_path)["bus"].shape[0])
            energy: dict[int, float] = {}
            noise_energy: dict[int, float] = {}
            for item in measurements:
                if int(item["h"]) <= 1:
                    continue
                bus = int(item["bus"])
                if "V_real" in item and "V_imag" in item:
                    voltage = complex(float(item["V_real"]), float(item["V_imag"]))
                else:
                    voltage = cmath.rect(float(item["Vm"]), math.radians(float(item.get("Va_deg", 0))))
                sigma = float(item.get("sigma", 1e-4))
                if item.get("sigma_semantics", "per_component") == "complex_rms":
                    sigma /= math.sqrt(2.0)
                elif item.get("sigma_semantics", "per_component") != "per_component":
                    raise ValueError("Unsupported harmonic sigma semantics")
                if not (math.isfinite(abs(voltage)) and math.isfinite(sigma) and sigma > 0):
                    raise ValueError("Harmonic measurement and positive noise scale must be finite")
                energy[bus] = energy.get(bus, 0.0) + abs(voltage) ** 2
                noise_energy[bus] = noise_energy.get(bus, 0.0) + 2 * sigma ** 2
            ratios: dict[int, float] = {}
            for bus, squared in energy.items():
                if not 1 <= bus <= min(nb, len(observed)):
                    raise ValueError("Harmonic monitor has no corresponding SCADA voltage")
                reference = float(observed[bus - 1])
                if not math.isfinite(reference) or reference <= 0:
                    raise ValueError("Spectral screening requires a positive measured voltage")
                ratios[bus] = 100 * math.sqrt(squared) / reference
            detected = any(
                ratio >= self.harmonic_thd_threshold_percent
                and energy[bus] > 9 * noise_energy[bus]
                for bus, ratio in ratios.items()
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
            "harmonic_context_status": "available",
            "available_evidence_channels": ["harmonic_measurements"],
            "harmonic_distortion_detected": detected,
            "harmonic_screening": {
                "detected": detected,
                # The input SCADA Vm may be total RMS. Do not label this
                # acquisition-stage ratio as fundamental-referenced THD.
                "maximum_harmonic_to_scada_voltage_percent": max(ratios.values(), default=0.0),
                "minimum_ratio_percent": self.harmonic_thd_threshold_percent,
                "minimum_noise_energy_ratio": 9.0,
                "source_localized": False,
            },
            "minted_signatures": ["harmonic distortion_detected_by_context"] if detected else [],
            "harmonic_orders": [int(order) for order in orders],
            "measured_buses": buses,
            "harmonic_summary": summary,
        }

    def run_hse(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        unavailable = self._auxiliary_unavailable(state, RUN_HSE_FROM_PATH)
        if unavailable is not None:
            return unavailable
        state = self._evidence_state(state)
        try:
            measurements = self._harmonic_measurements(state)
            case_path = self._case_path(state)
            orders = self._metadata(state).get("harmonic_orders") or _infer_harmonic_orders(
                measurements
            )
        except Exception as exc:
            return self._failure("hse_runtime_missing", f"{type(exc).__name__}: {exc}")
        slack_bus = int(self._metadata(state).get("slack_bus", 0))
        # The observed fundamental Vm is the physically meaningful THD
        # denominator; the case's stored Vm is only a planning value.
        fundamental_vm: list[float] | None = None
        try:
            observed = self._measurements(state)
            nb = int(_load_python_case(case_path)["bus"].shape[0])
            if len(observed) >= nb:
                fundamental_vm = observed[:nb]
        except Exception:
            fundamental_vm = None
        payload = _run_hse_logic(
            case_path,
            measurements,
            [int(order) for order in orders],
            slack_bus,
            fundamental_vm=fundamental_vm,
        )
        if not payload.get("success"):
            return self._failure("hse_failure", payload.get("error"))
        summary = summarize_hse_payload(payload)
        best_bus = payload.get("best_candidate_bus_1based")
        best_thd = summary.get("best_candidate_thd_percent")
        try:
            thd_value = float(best_thd)
        except (TypeError, ValueError):
            thd_value = math.nan
        try:
            sse_reduction = float(payload.get("sse_reduction_vs_null"))
        except (TypeError, ValueError):
            sse_reduction = math.nan
        # Acceptance requires both an operationally material distortion and a
        # single-source model that actually explains the measured harmonic
        # voltages (fails closed when the estimator reports no null-model
        # comparison).
        source_explains_data = bool(
            math.isfinite(sse_reduction) and sse_reduction >= self.hse_min_sse_reduction
        )
        accepted = bool(
            best_bus is not None
            and math.isfinite(thd_value)
            and thd_value >= self.harmonic_thd_threshold_percent
            and source_explains_data
        )
        metrics = {
            **self._binding(state),
            "evidence_source": "deployment_diagnostic:harmonic_state_estimation",
            "best_candidate_bus_1based": best_bus,
            "hse_summary": summary,
            "diagnostic_acceptance": {
                "accepted": accepted,
                "null_hypothesis": "no_harmonic_source_or_thd_below_operational_threshold",
                "thd_percent": thd_value if math.isfinite(thd_value) else None,
                "minimum_thd_percent": self.harmonic_thd_threshold_percent,
                "sse_reduction_vs_null": sse_reduction if math.isfinite(sse_reduction) else None,
                "minimum_sse_reduction": self.hse_min_sse_reduction,
                "source_model_explains_data": source_explains_data,
                "fundamental_voltage_source": payload.get("fundamental_voltage_source"),
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
        unavailable = self._auxiliary_unavailable(state, RUN_THREE_PHASE_NLM_FROM_PATH)
        if unavailable is not None:
            return unavailable
        state = self._evidence_state(state)
        strict = self._strict_boundary(state)
        metadata = self._metadata(state)
        # A stored diagnostic is the hidden sample's own output and the faulted
        # model is the truth; a strict profile computes from the acquired
        # phasors only and fails closed when they are missing.
        diagnostic = None if strict else metadata.get("nlm_diagnostic")
        pristine_dir = None if strict else metadata.get("pristine_model_dir")
        faulted_dir = None if strict else metadata.get("faulted_model_dir")
        unbalance_signal = self._has_family_signature(state, "three_phase_unbalance")
        hif_signal = self._has_family_signature(state, "hif")

        # Pure unbalance is a distinct terminal explanation, not an implicit
        # HIF localization.  Sequence-voltage evidence provides an explicit
        # balanced-system null test and never emits a candidate HIF branch.
        runtime = metadata.get("hif_runtime")
        runtime = runtime if isinstance(runtime, Mapping) else {}
        # HIF rows carry their three-phase voltages inside the runtime block;
        # unbalance rows carry them at the metadata top level.
        three_phase_voltages = metadata.get("three_phase_voltages") or runtime.get(
            "three_phase_voltages"
        )
        branch_currents, current_sigma = self._branch_current_channel(state)
        if branch_currents and current_sigma is None:
            return self._sigma_undeclared(state, BRANCH_CURRENT_SIGMA_KEY, RUN_THREE_PHASE_NLM_FROM_PATH)
        # Screening: no sensor has flagged a waveform anomaly, so the operator
        # only holds the positive-sequence snapshot and a fundamental-frequency
        # anomaly.  Before any residual is attributed to a meter or a branch,
        # the three-phase telemetry is checked for the event the balanced
        # model cannot represent: a source of unbalance, an HIF-like line
        # differential, or neither (balanced, and the classical routes stand).
        screening = not (unbalance_signal or hif_signal or self._waveform_route_block(state))
        screening_hif_like = False
        if (unbalance_signal or screening) and not hif_signal and three_phase_voltages:
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
            summary: dict[str, Any] = {
                "success": True,
                "converged": True,
                "method": "sequence_voltage_unbalance_test",
                "diagnostic_classification": (
                    "three_phase_unbalance" if accepted else "unresolved"
                ),
                "top_hif_groups": [],
                "top_vuf_buses": vuf_evidence,
            }
            detail: dict[str, Any] = {
                "max_vuf": max_vuf,
                "minimum_vuf": self.unbalance_vuf_threshold,
                "top_vuf_buses": vuf_evidence,
            }
            evidence_source = "deployment_diagnostic:sequence_voltage_unbalance"
            if branch_currents:
                # Per-phase branch currents localize the unbalance *source*
                # (negative-sequence voltage alone peaks at weak buses) and
                # supply an explicit non-HIF null: no line may carry a
                # differential current above the sensor floor.
                localization = unbalance_source_localization(
                    three_phase_voltages,
                    branch_currents,
                    top_k=self.top_k,
                    sigma_pu=current_sigma,
                )
                null_test = line_differential_null_test(
                    three_phase_voltages, branch_currents, sigma_pu=current_sigma
                )
                hif_like = bool(
                    null_test is not None
                    and null_test.get("hif_like_differential_present")
                )
                # The per-phase shunt-power spread measures the unbalance
                # source directly, while VUF is a symptom that also depends on
                # network strength; with currents present the explanation
                # rests on a noise-significant source and a quiet line null.
                source_significant = bool(
                    localization is not None and localization.get("significant")
                )
                voltage_gate_passed = bool(accepted)
                accepted = bool(source_significant and not hif_like)
                acceptance.update(
                    {
                        "accepted": accepted,
                        "null_hypothesis": (
                            "no_significant_shunt_power_spread_and_no_line_differential"
                        ),
                        "voltage_gate_passed": voltage_gate_passed,
                        "line_differential_null": null_test,
                        "source_localized": localization is not None,
                        "source_significant": source_significant,
                        "acceptance_basis": (
                            "shunt_power_spread_source" if accepted else None
                        ),
                    }
                )
                summary.update(
                    {
                        "method": "sequence_voltage_unbalance_test+shunt_power_spread",
                        "diagnostic_classification": (
                            "three_phase_unbalance"
                            if accepted
                            else (
                                "hif_suspected"
                                if hif_like
                                else ("balanced_three_phase" if screening else "unresolved")
                            )
                        ),
                        "line_differential_null": null_test,
                    }
                )
                screening_hif_like = bool(screening and hif_like and not accepted)
                if localization is not None:
                    summary["localization"] = {
                        key: localization[key]
                        for key in (
                            "method",
                            "bus_1based",
                            "phase_power_spread_rel",
                            "separation_ratio",
                            "significant",
                            "significant_bus_count",
                        )
                    }
                    summary["top_unbalance_source_buses"] = localization[
                        "top_unbalance_source_buses"
                    ]
                    detail.update(
                        {
                            "bus_1based": int(localization["bus_1based"]),
                            "localization": summary["localization"],
                            "top_unbalance_source_buses": localization[
                                "top_unbalance_source_buses"
                            ],
                        }
                    )
                detail["line_differential_null"] = null_test
                evidence_source = (
                    "deployment_diagnostic:sequence_voltage_unbalance+branch_currents"
                )
            metrics: dict[str, Any] = {
                **self._binding(state),
                "evidence_source": evidence_source,
                "nlm_summary": summary,
                "diagnostic_acceptance": acceptance,
            }
            if screening:
                summary["screening_mode"] = True
                if not accepted and not screening_hif_like:
                    summary["diagnostic_classification"] = (
                        "balanced_three_phase"
                        if branch_currents or max_vuf < self.unbalance_vuf_threshold
                        else summary["diagnostic_classification"]
                    )
            if accepted:
                metrics["anomaly_explanation"] = {
                    "family": "three_phase_unbalance",
                    "kind": (
                        "voltage_unbalance_source_localized"
                        if branch_currents
                        else "voltage_unbalance_confirmed"
                    ),
                    "detail": detail,
                }
            if not screening_hif_like:
                return metrics
            # An HIF-like line differential during screening: fall through to
            # the terminal-current localization so the ladder gets a ranked
            # line and phase, and mint the HIF signature the ladder keys on.

        if three_phase_voltages and branch_currents:
            # Two-terminal differential currents name the faulted line and
            # phase from the telemetry itself; a stored NLM diagnostic is kept
            # only as secondary evidence.  A persistent scan window is averaged
            # coherently so the noise floor drops with the scan count.
            window = metadata.get("hif_scan_window")
            window_scans = (
                window.get("scans") if isinstance(window, Mapping) else None
            )
            current_scans = [
                scan
                for scan in (window_scans or [])
                if isinstance(scan, Mapping)
                and scan.get("three_phase_voltages")
                and scan.get(BRANCH_CURRENT_CHANNEL)
            ]
            if len(current_scans) >= 2:
                localized = terminal_current_hif_localization_multiscan(
                    current_scans,
                    top_k=self.top_k,
                    sigma_pu=current_sigma,
                )
            else:
                localized = terminal_current_hif_localization(
                    three_phase_voltages,
                    branch_currents,
                    top_k=self.top_k,
                    sigma_pu=current_sigma,
                )
            if localized is not None:
                summary = summarize_three_phase_nlm_payload(localized)
                if isinstance(diagnostic, Mapping):
                    legacy_groups = diagnostic.get("top_hif_groups")
                    if isinstance(legacy_groups, Sequence) and legacy_groups:
                        first = legacy_groups[0]
                        if isinstance(first, Mapping) and first.get("branch_row0") is not None:
                            summary["legacy_nlm_top_branch_row0"] = int(first["branch_row0"])
                            summary["legacy_nlm_method"] = diagnostic.get("method")
                localized_metrics: dict[str, Any] = {
                    **self._binding(state),
                    "evidence_source": "deployment_diagnostic:terminal_current_differential",
                    "nlm_summary": summary,
                }
                if screening_hif_like:
                    summary["screening_mode"] = True
                    summary["diagnostic_classification"] = "hif_suspected"
                    localized_metrics["minted_signatures"] = [HIF_SCREENING_SIGNATURE]
                return localized_metrics

        if strict:
            return self._failure(
                "nlm_runtime_missing",
                "the acquired three-phase phasors are missing or unusable; a strict "
                "evidence profile never falls back to a stored diagnostic or a faulted model",
            )
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

    def _hif_search_arguments(self, arguments, *sources):
        from IEEE_14_OpenDSS.measurement_convention import resolve_shunt_convention
        explicit = arguments.get("shunt_convention")
        declared = explicit is not None or any(isinstance(source, Mapping) and source.get("measurement_convention") is not None for source in sources)
        return {
            **{key: arguments.get(key) for key in ("r_hif_pu_min", "r_hif_pu_max", "r_hif_ohm_min", "r_hif_ohm_max", "kv_ll")},
            "resistance_search": self.hif_resistance_search,
            "shunt_convention": resolve_shunt_convention(explicit, *sources) if declared else None,
        }

    def estimate_hif(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        unavailable = self._auxiliary_unavailable(state, ESTIMATE_HIF_FROM_PATH)
        if unavailable is not None:
            return unavailable
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
        state = self._evidence_state(state)
        metadata = self._metadata(state)
        runtime = metadata.get("hif_runtime")
        runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
        try:
            case_path = self._case_path(state)
            z_obs = self._measurements(state)
        except Exception as exc:
            return self._failure("hif_input_error", f"{type(exc).__name__}: {exc}")
        branch_currents, current_sigma = self._branch_current_channel(state)
        if branch_currents and current_sigma is None:
            return self._sigma_undeclared(state, BRANCH_CURRENT_SIGMA_KEY, ESTIMATE_HIF_FROM_PATH)
        voltage_sigma = self._phasor_sigma(
            state, "three_phase_sigma", "three_phase_voltages", DEFAULT_THREE_PHASE_SIGMA_PU
        )
        if voltage_sigma is None:
            return self._sigma_undeclared(state, "three_phase_sigma", ESTIMATE_HIF_FROM_PATH)
        payload = _estimate_hif_location_magnitude_logic(
            case_path=case_path,
            candidate_branch_row0=int(arguments["candidate_branch_row0"]),
            candidate_phase=arguments.get("candidate_phase"),
            z_obs=[float(value) for value in z_obs],
            three_phase_voltages=(
                runtime.get("three_phase_voltages") or metadata.get("three_phase_voltages")
            ),
            pristine_model_dir=runtime.get("pristine_model_dir"),
            load_scale=float(runtime.get("load_scale", 1.0)),
            top_k=int(arguments.get("top_k", self.top_k)),
            alpha_grid_size=alpha_grid_size,
            r_grid_size=r_grid_size,
            **self._hif_search_arguments(arguments, runtime, metadata),
            three_phase_branch_currents=branch_currents,
            branch_current_sigma_pu=current_sigma,
            sigma_z=runtime.get("sigma_z", metadata.get("sigma_z")),
            three_phase_sigma=float(voltage_sigma),
        )
        if not payload.get("success"):
            return self._failure("hif_estimation_failure", payload.get("error"))
        summary = summarize_hif_parameter_estimate_payload(payload)
        acceptance = self._hif_diagnostic_acceptance(payload)
        metrics: dict[str, Any] = {
            **self._binding(state),
            "evidence_source": (
                "deployment_diagnostic:hif_parameter_estimator+branch_currents"
                if branch_currents
                else "deployment_diagnostic:hif_parameter_estimator"
            ),
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
                    "conditioning_fit": fit_receipt(payload, case_path, independent=False),
                    "terminal_current_estimate": summary.get("terminal_current_estimate"),
                    "residual_reduction_vs_null": acceptance[
                        "residual_reduction_vs_null"
                    ],
                },
            }
        return metrics

    def estimate_hif_multiscan(
        self, state: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        unavailable = self._auxiliary_unavailable(state, ESTIMATE_HIF_MULTISCAN_FROM_PATH)
        if unavailable is not None:
            return unavailable
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
        state = self._evidence_state(state)
        strict = self._strict_boundary(state)
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
        # Every scan is estimated with the sigma it declares (or the window's
        # declaration); a strict profile refuses a scan that declares neither.
        metadata = self._metadata(state)
        declared_scans: list[dict[str, Any]] = []
        for scan in scans:
            item = dict(scan) if isinstance(scan, Mapping) else {}
            voltage_sigma = self._declared_phasor_sigma(metadata, "three_phase_sigma", "three_phase_voltages", item)
            if voltage_sigma is None and not strict:
                voltage_sigma = float(DEFAULT_THREE_PHASE_SIGMA_PU)
            if voltage_sigma is None:
                return self._sigma_undeclared(state, "three_phase_sigma", ESTIMATE_HIF_MULTISCAN_FROM_PATH)
            item["three_phase_sigma"] = float(voltage_sigma)
            if item.get(BRANCH_CURRENT_CHANNEL):
                current_sigma = self._declared_phasor_sigma(metadata, BRANCH_CURRENT_SIGMA_KEY, BRANCH_CURRENT_CHANNEL, item)
                if current_sigma is None and not strict:
                    current_sigma = float(DEFAULT_BRANCH_CURRENT_SIGMA_PU)
                if current_sigma is None:
                    return self._sigma_undeclared(state, BRANCH_CURRENT_SIGMA_KEY, ESTIMATE_HIF_MULTISCAN_FROM_PATH)
                item[BRANCH_CURRENT_SIGMA_KEY] = float(current_sigma)
            declared_scans.append(item)
        scans = declared_scans
        independent = False
        try:
            acquisition = current_scan(self._metadata(state))
            history_scans = [scan for scan in scans if scan.get("scan_index") != acquisition["scan_index"]]
            if len(history_scans) >= 2:
                scans = history_scans
                independent = True
        except ValueError:
            # Diagnosis remains available; independent replay/meter repair
            # fails closed if the current acquisition cannot be bound.
            pass
        payload = self._memoized_hif_multiscan(
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
            **self._hif_search_arguments(arguments, window, self._metadata(state)),
            robust_loss=str(arguments.get("robust_loss", "soft_l1")),
            smoothness_lambda=float(arguments.get("smoothness_lambda", 0.10)),
            branch_current_sigma_pu=self._branch_current_channel(state)[1],
            require_declared_sigmas=strict,
        )
        if not payload.get("success"):
            return self._failure("hif_multiscan_failure", payload.get("error"))
        summary = summarize_hif_parameter_estimate_payload(payload)
        acceptance = self._hif_diagnostic_acceptance(payload)
        search = payload.get("search") if isinstance(payload.get("search"), Mapping) else {}
        metrics: dict[str, Any] = {
            **self._binding(state),
            "evidence_source": (
                "deployment_diagnostic:hif_multiscan_estimator+branch_currents"
                if search.get("branch_current_block")
                else "deployment_diagnostic:hif_multiscan_estimator"
            ),
            "hif_summary": summary,
            "diagnostic_acceptance": acceptance,
            "conditioning_fit_independent_of_current_scada": independent,
        }
        if acceptance["accepted"]:
            metrics["anomaly_explanation"] = {
                "family": "hif",
                "kind": "hif_model_accepted_over_null",
                "detail": {
                    "candidate_branch_row0": int(arguments["candidate_branch_row0"]),
                    "estimated": summary.get("estimated"),
                    "conditioning_fit": fit_receipt(payload, case_path, independent=independent, metadata=self._metadata(state)),
                    "terminal_current_estimate": summary.get("terminal_current_estimate"),
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
