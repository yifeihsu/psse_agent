"""The numerical execution layer.

Everything here computes; nothing here decides.  The kernel runs the state
estimator, the substation (node/breaker) estimator and the three correction
tools on the state it is handed and returns their numbers.  It never ranks a
target, never offers or withholds a correction, never accepts or rejects one,
and never says what to do next.

It reuses the numerical routines the old environment runs
(``psse_env.providers.matpower.MatpowerDeploymentProviders``): the WLS solve
and its two tests, the complete tables of ``complete_evidence``, the
measurement repair, the multi-scan line fit, the breaker flip and the
substation estimator.  It reads none of that provider's screens: the
``get_*_context`` methods (top-5 cuts, dominance test, ``supported_corrections``,
``route_status``), ``run_wls`` (signatures, target verdicts) and the candidate
oracle are not called.

A request the tools cannot carry out (a fit that fails, a repair that changes
nothing) comes back as ``{"ok": False, ...}``: an execution failure, never a
judgement of the diagnosis.

``CostMeter`` counts what the calls cost, so an interface that needs fewer
model calls is not mistaken for one that computes less.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from psse_env.providers import complete_evidence as _complete
from psse_env.providers.matpower import MatpowerDeploymentProviders
from trace_protocol import chi2_threshold

#: The options of the compound-error runs (``research/hpc/*/configs``): SCADA-class data only, the
#: chi-square test paired with the largest-normalized-residual test at 4, the alternating pair refit.
DEFAULT_PROVIDER_OPTIONS: dict[str, Any] = {
    "evidence_profile": "scada_only",
    "normalized_residual_threshold": 4.0,
    "pair_refit_adjacent": True,
    "complete_evidence": True,
}
#: Named breakers one ``inspect`` may ask the substation estimator to re-estimate.
MAX_FLIP_TESTS_PER_CALL = 8


@dataclass
class CostMeter:
    """What the numerical tools were asked to do, and how long it took."""

    wls_solves: int = 0
    substation_solves: int = 0
    parameter_fits: int = 0
    measurement_repairs: int = 0
    seconds: dict[str, float] = field(default_factory=dict)

    def add_time(self, kind: str, started: float) -> None:
        self.seconds[kind] = self.seconds.get(kind, 0.0) + (time.perf_counter() - started)

    def snapshot(self) -> dict[str, Any]:
        return {
            "wls_solves": self.wls_solves,
            "substation_solves": self.substation_solves,
            "parameter_fits": self.parameter_fits,
            "measurement_repairs": self.measurement_repairs,
            "seconds": {key: round(value, 4) for key, value in sorted(self.seconds.items())},
            "tool_seconds": round(sum(self.seconds.values()), 4),
        }


def _failure(output: Mapping[str, Any], default: str) -> dict[str, Any]:
    return {"ok": False, "error_code": str(output.get("error_code") or default),
            "error_detail": output.get("error_detail")}


class NumericalKernel:
    """State estimation, substation estimation and the correction tools, with their cost counted."""

    def __init__(self, providers: MatpowerDeploymentProviders | None = None, **provider_options: Any) -> None:
        if providers is not None and provider_options:
            raise ValueError("pass a provider bundle or its options, not both")
        self.providers = providers or MatpowerDeploymentProviders(**{**DEFAULT_PROVIDER_OPTIONS, **provider_options})
        self.cost = CostMeter()

    def reset_cost(self) -> None:
        self.cost = CostMeter()

    # ---------------------------------------------------------------- estimation

    def estimate(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """One WLS solve of ``state``: convergence, the two tests' numbers and every bus, line and meter.

        ``lines`` joins the model's series R and X with the solve's
        multipliers and flow residuals; ``meters`` is the measurement vector
        with each reading's declared standard deviation and signed normalized
        residual.  A solve that does not converge is reported as such, with
        the solver's message and no tables.
        """
        started = time.perf_counter()
        self.cost.wls_solves += 1
        try:
            solved = self.providers._solve(dict(state))
        except Exception as exc:  # an unreadable case or vector: the estimate did not run
            self.cost.add_time("wls", started)
            return {"converged": False, "error": f"{type(exc).__name__}: {exc}"}
        payload = solved["payload"]
        if not payload.get("success"):
            self.cost.add_time("wls", started)
            return {"converged": False, "error": str(payload.get("error") or "solver_failure")}
        try:
            tests = self.providers._wls_detection_metrics(solved)
            signed = _complete.signed_residuals(payload)
            lambdas = payload.get("lambdaN") or []
            ppc = solved["ppc"]
            solve = _complete.solve_tables(signed, lambdas, ppc, vm_estimate=payload.get("vm_est_pu"),
                                           angle_estimate_rad=payload.get("theta_est_rad"))
            flows = {int(row["line"]): row for row in solve["branch_table"]}
            lines = []
            for row in _complete.line_table(lambdas, ppc):
                flow = flows[int(row["line"])]
                lines.append({**row, **{key: flow.get(key) for key in ("pf", "qf", "pt", "qt")}})
            meters = _complete.meter_table(solved["z"], signed, ppc)
            sigma = solved["noise_options"].get("measurement_sigma")
            if sigma is not None and len(sigma) == len(meters):
                meters = [{**row, "sigma": float(sigma[position])} for position, row in enumerate(meters)]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.cost.add_time("wls", started)
            return {"converged": False, "error": f"{type(exc).__name__}: {exc}"}
        self.cost.add_time("wls", started)
        return {
            "converged": True,
            "chi_square": float(tests["chi_square_statistic"]),
            "chi_square_threshold": float(tests["chi_square_threshold"]),
            "chi_square_dof": int(tests["chi_square_dof"]),
            "max_residual": float(tests["max_normalized_residual"]),
            "residual_threshold": tests["normalized_residual_threshold"],
            "bus_count": int(solved["nb"]),
            "line_count": int(solved["nl"]),
            "buses": solve["bus_table"],
            "lines": lines,
            "meters": meters,
        }

    # ------------------------------------------------------------- the substation

    def has_substation_model(self, state: Mapping[str, Any]) -> bool:
        """Whether the state carries a node/breaker model: reported breaker statuses and substation telemetry."""
        metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
        return isinstance(metadata.get("substation_telemetry"), Mapping) and isinstance(
            metadata.get("reported_breaker_status"), Mapping)

    def breaker_names(self, state: Mapping[str, Any]) -> list[str]:
        metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
        reported = metadata.get("reported_breaker_status")
        return sorted(str(name) for name in reported) if isinstance(reported, Mapping) else []

    def _substation_inputs(self, state: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        prepared = self.providers._evidence_state(dict(state))
        binding = self.providers._topology_binding(prepared)
        if binding is None:
            return None, {"ok": False, "error_code": "substation_model_missing",
                          "error_detail": "the state carries no substation telemetry and reported breaker statuses"}
        if binding.get("error_code"):
            return None, _failure(binding, "substation_model_error")
        from Transmission.ieee14_full_substation import status_map_from_labels

        return {
            "binding": binding,
            "telemetry": self.providers._synchronized_telemetry(prepared, binding),
            "reported": status_map_from_labels(binding["reported_labels"]),
        }, None

    def substation(self, state: Mapping[str, Any], flips: Sequence[str] = ()) -> dict[str, Any]:
        """The substation estimate of ``state`` and its table of every breaker.

        One estimate gives every breaker's normalized multipliers.  The
        estimator re-estimates with a breaker's status flipped for the
        ``breaker_candidate_count`` breakers its own multipliers rank first
        (its confirmation stage, as in the old topology context) and for
        every breaker named in ``flips``; those rows carry the chi-square
        after the flip.
        """
        from Transmission.ieee14_full_gse import gse_topology_nlm, screen_breaker_flips

        started = time.perf_counter()
        inputs, failure = self._substation_inputs(state)
        if inputs is None:
            return failure  # type: ignore[return-value]
        binding = inputs["binding"]
        model, reference, labels = binding["model"], binding["reference"], binding["reported_labels"]
        unknown = [name for name in flips if str(name) not in labels]
        if unknown:
            return {"ok": False, "error_code": "unknown_breaker", "error_detail": ", ".join(map(str, unknown))}
        self.cost.substation_solves += 1
        estimate = gse_topology_nlm(model, reference, inputs["reported"], inputs["telemetry"])
        if not estimate.get("success"):
            self.cost.add_time("substation", started)
            return {"ok": True, "converged": False, "iterations": int(estimate.get("iterations") or 0)}
        tested = [str(item["cb_name"]) for item in estimate["ranking"][: self.providers.breaker_candidate_count]]
        tested += [str(name) for name in flips if str(name) not in tested]
        self.cost.substation_solves += len(tested)
        results = {
            item["cb_name"]: item
            for item in screen_breaker_flips(model, reference, inputs["reported"], inputs["telemetry"], tested)
        }
        effects = {
            str(item["cb_name"]): self.providers._breaker_effect(
                model, reference, labels, item["cb_name"], item["reported_status"] != "closed")
            for item in estimate["ranking"]
        }
        chi_square = float(estimate["chi_square"])
        table = _complete.breaker_table(estimate["ranking"], effects, results, chi_square)
        self.cost.add_time("substation", started)
        return {
            "ok": True,
            "converged": True,
            "chi_square": chi_square,
            "chi_square_threshold": float(chi2_threshold(max(1, int(estimate["dof"])), self.providers.chi2_alpha)),
            "chi_square_dof": int(estimate["dof"]),
            "max_residual": float(estimate.get("max_normalized_residual", 0.0)),
            "breakers": table,
        }

    # --------------------------------------------------------------- corrections

    def correct_measurements(self, state: Mapping[str, Any], meters: Sequence[int]) -> dict[str, Any]:
        """Repair the named measurements: the tool computes the corrected readings."""
        started = time.perf_counter()
        self.cost.measurement_repairs += 1
        output = self.providers.correct_measurements(
            dict(state), {"arguments": {"suspect_group": [int(index) for index in meters]}})
        self.cost.add_time("measurement_repair", started)
        if output.get("execution_status") == "failure":
            return _failure(output, "measurement_correction_failure")
        return {"ok": True, "modification": output["modification"],
                "report": {"iterations": output.get("iterations_performed")}}

    def correct_parameters(self, state: Mapping[str, Any], line: int) -> dict[str, Any]:
        """Fit the series R and X of the named line from the repeated scans."""
        started = time.perf_counter()
        self.cost.parameter_fits += 1
        output = self.providers.correct_parameters(dict(state), {"arguments": {"line_index": int(line)}})
        refit = output.get("pair_refit") if isinstance(output.get("pair_refit"), Mapping) else None
        if refit is not None and refit.get("status") == "refit":
            self.cost.parameter_fits += len(refit.get("steps") or [])
        self.cost.add_time("parameter_fit", started)
        if output.get("execution_status") == "failure":
            return _failure(output, "parameter_correction_failure")
        return {"ok": True, "modification": output["modification"], "report": {}}

    def correct_topology(self, state: Mapping[str, Any], *, status: str, breaker: str | None = None,
                         line: int | None = None) -> dict[str, Any]:
        """Set the reported status of a breaker, or the service status of a line where there is no breaker model."""
        started = time.perf_counter()
        if breaker is not None:
            arguments: dict[str, Any] = {"cb_name": str(breaker), "status": status}
        else:
            arguments = {"line_index": int(line), "status": 1 if status == "in" else 0}  # type: ignore[arg-type]
        output = self.providers.correct_topology(dict(state), {"arguments": arguments})
        self.cost.add_time("topology", started)
        if output.get("execution_status") == "failure":
            return _failure(output, "topology_correction_failure")
        report = {key: output[key] for key in ("breaker_effect", "affected_planning_buses") if key in output}
        return {"ok": True, "modification": output["modification"], "report": report}


__all__ = ["CostMeter", "DEFAULT_PROVIDER_OPTIONS", "MAX_FLIP_TESTS_PER_CALL", "NumericalKernel"]
