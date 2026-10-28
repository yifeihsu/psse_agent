"""Identifiability probes for the pairs the compound-errors assessment flagged (2026-10-27).

    python -m research.compound_errors.identifiability --suite output/compound_errors_20261027/suite.json \\
        --output-dir output/compound_errors_20261027/identifiability --workers 14

Three experiments, each on the roots of the classes it concerns, each
candidate in its own research environment (``scada_only``, the deployment
candidate oracle, no expert):

* ``same_line_meter`` (class ``meter_on_rx``): a bad flow meter on the line
  whose impedance is wrong.  The true meter correction alone, the true
  parameter correction alone, and both in either order, with the verifier's
  verdict, the solve after each step, the estimate's distance from the true
  impedance and whether the end state would pass the truth audit's case and
  measurement checks.  A twin of every root with the same meter error moved
  to a healthy channel is probed the same way, so the pair's own effect is
  read against a paired control.
* ``same_terminal_breaker`` (class ``rx_cb_same_line``): the breaker that
  isolates a terminal of the line whose impedance is wrong.  Breaker first,
  parameter first, and each alone; the line's terminal flows in the
  snapshot (the observability of its impedance) and whether the breaker fix
  alone leaves the solve clean.
* ``adjacent_lines`` (classes ``rx2_adjacent`` and ``rx2_distant``): two
  wrong lines.  The deployed ranking's view of both lines, the bounded
  comparison's verdict on the top candidates, and the sequential
  single-line estimates (the first line on the stale model, the second on
  the derived model, then the first again: one round of an alternating pair
  refit) with their distance from the truth.

Every probe is the deployed executor and verifier the policy uses; the truth
is read only to name the targets and to score the estimates.

``--system`` (2026-10-28) follows ``research.compound_errors.rollout``: the
suite's system is read from the roots and a system other than IEEE 14 runs
with the ``chi_square_excess`` progress basis, on the gated environment and in
the ungated provider path's own progress, unless ``--progress-basis`` names
one.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import (  # noqa: E402
    COMMIT_STATE,
    COMPARE_PARAMETER_CANDIDATES,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    ROLLBACK_STATE,
    RUN_WLS,
)

CONTRACT = "compound_identifiability_v1"
EXPERIMENTS = ("same_line_meter", "same_terminal_breaker", "adjacent_lines", "breaker_masking")
EXPERIMENT_CLASSES = {
    "same_line_meter": ("meter_on_rx",),
    "same_terminal_breaker": ("rx_cb_same_line",),
    "adjacent_lines": ("rx2_adjacent", "rx2_distant"),
    # How a parameter correction on the context's top line explains a breaker
    # error (the masking the parameter-first order exposes, section 5).
    "breaker_masking": ("control_cb", "rx_cb_distant", "rx_cb_shared_bus", "cb2_dangling"),
}
CONTEXT_OF = {CORRECT_MEASUREMENTS: GET_MEASUREMENT_CONTEXT, CORRECT_PARAMETERS: GET_PARAMETER_CONTEXT, CORRECT_TOPOLOGY: GET_TOPOLOGY_CONTEXT}
ACCEPTING = {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}
_WLS_KEYS = ("chi_square_statistic", "chi_square_threshold", "chi_square_ratio", "globally_resolved", "global_progress",
             "target_progress", "remaining_suspect_count", "max_normalized_residual", "target_metric_value", "target_metric_threshold")

_FACTORY: Callable[[], Any] | None = None
#: The progress basis of the run (None: the environment's default, the whole statistic).
_PROGRESS_BASIS: str | None = None


# --------------------------------------------------------------------------- environment helpers


def make_factory(profile: str, progress_basis: str | None = None) -> Callable[[], Any]:
    import scripts.run_dagger_research as research

    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = profile
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    research.RESEARCH_ENVIRONMENT_OPTIONS["triage_classifier"] = None
    if progress_basis:
        research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] = progress_basis
    factory = research.resolve_environment_factory("research", profile)
    return lambda: factory()


def _init_worker(profile: str, progress_basis: str | None = None) -> None:
    global _FACTORY, _PROGRESS_BASIS
    _FACTORY = make_factory(profile, progress_basis)
    _PROGRESS_BASIS = progress_basis


def system_size(root: Mapping[str, Any]) -> tuple[int, int]:
    """(buses, branches) of the root's registered system."""
    from psse_env.systems import resolve_system

    spec = resolve_system(str((root.get("grouping") or {}).get("case_id") or "case14"))
    return spec.nb, spec.nl


def active_id(env: Any) -> str:
    return str(env.get_policy_observation().as_dict()["active_state_id"])


def candidate_id(env: Any) -> str | None:
    observation = env.get_policy_observation().as_dict()
    value = observation.get("candidate_state_id")
    return str(value) if value else None


def step(env: Any, tool: str, state_id: str | None = None, **arguments: Any) -> dict[str, Any]:
    _state, output = env.step({"tool": tool, "arguments": {"state_id": state_id or active_id(env), **arguments}})
    return dict(output)


def wls_metrics(output: Mapping[str, Any]) -> dict[str, Any]:
    metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
    return {key: metrics.get(key) for key in _WLS_KEYS if key in metrics}


def disposition_of(env: Any) -> str | None:
    oracle = env.get_oracle_state()
    value = getattr(oracle, "candidate_disposition", None)
    if value is None and isinstance(oracle, Mapping):
        value = oracle.get("candidate_disposition")
    value = getattr(value, "value", value)
    return str(value) if value is not None else None


def branch_rx(env: Any, state_id: str, row0: int) -> tuple[float, float] | None:
    from mcp_server.matpower_server import _load_python_case

    try:
        payload = env.store.get_state(state_id)
    except Exception:
        return None
    case = payload.get("case")
    try:
        ppc = _load_python_case(case) if isinstance(case, str) else case
        row = ppc["branch"][int(row0)]
        return float(row[2]), float(row[3])
    except Exception:
        return None


def new_env(root: Mapping[str, Any]) -> Any:
    assert _FACTORY is not None
    env = _FACTORY()
    env.reset(copy.deepcopy(root["execution"]))
    step(env, RUN_WLS)
    return env


def apply(env: Any, action: Mapping[str, Any], *, row0: int | None = None, lifecycle: str | None = None) -> dict[str, Any]:
    """Fetch the family's context, apply one correction, verify it and dispose of it (commit when accepted)."""
    tool = str(action["tool"])
    arguments = dict(action.get("arguments") or {})
    context = step(env, CONTEXT_OF[tool])
    record: dict[str, Any] = {"tool": tool, "arguments": arguments, "context_status": context.get("execution_status")}
    output = step(env, tool, **arguments)
    record.update(status=output.get("execution_status"), error_code=output.get("error_code"),
                  error_detail=str(output.get("error_detail") or "")[:160])
    if output.get("execution_status") != "success":
        return record
    candidate = candidate_id(env) or output.get("candidate_state_id")
    if candidate is None:
        record["status"] = "no_candidate"
        return record
    verification = step(env, RUN_WLS, state_id=str(candidate))
    record["verification"] = wls_metrics(verification)
    record["disposition"] = disposition_of(env)
    if tool == CORRECT_PARAMETERS and row0 is not None:
        record["estimate"] = branch_rx(env, str(candidate), row0)
    chosen = lifecycle or (COMMIT_STATE if record["disposition"] in ACCEPTING else ROLLBACK_STATE)
    closed = step(env, chosen, candidate_state_id=str(candidate))
    record["lifecycle"] = chosen
    record["lifecycle_status"] = closed.get("execution_status")
    record["lifecycle_error"] = closed.get("error_code")
    return record


def solve(env: Any) -> dict[str, Any]:
    return wls_metrics(step(env, RUN_WLS))


# --------------------------------------------------------------------------- truth helpers


def truth_of(root: Mapping[str, Any]) -> Mapping[str, Any]:
    return (root.get("audit") or {}).get("truth") or {}


def compound_of(root: Mapping[str, Any]) -> Mapping[str, Any]:
    return (root.get("audit") or {}).get("compound") or {}


def parameter_truths(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in truth_of(root).get("true_parameter_errors") or []]


def tolerance_of(root: Mapping[str, Any]) -> float:
    tolerances = ((root.get("audit") or {}).get("release_audit") or {}).get("tolerances") or {}
    try:
        return float(tolerances.get("final_case_abs"))
    except (TypeError, ValueError):
        return 0.02


def estimate_error(estimate: tuple[float, float] | None, truth: Mapping[str, Any], tolerance: float) -> dict[str, Any] | None:
    if estimate is None:
        return None
    r, x = estimate
    clean_r, clean_x = float(truth["clean_r"]), float(truth["clean_x"])
    return {
        "r": r, "x": x, "clean_r": clean_r, "clean_x": clean_x,
        "abs_r": abs(r - clean_r), "abs_x": abs(x - clean_x),
        "rel_r": abs(r - clean_r) / max(abs(clean_r), 1e-9), "rel_x": abs(x - clean_x) / max(abs(clean_x), 1e-9),
        "within_tolerance": bool(abs(r - clean_r) <= tolerance and abs(x - clean_x) <= tolerance),
    }


def meter_action(root: Mapping[str, Any]) -> dict[str, Any]:
    indices = sorted(int(item["index"]) for item in truth_of(root).get("true_measurement_errors") or [])
    return {"tool": CORRECT_MEASUREMENTS, "arguments": {"suspect_group": indices}}


def parameter_action(truth: Mapping[str, Any]) -> dict[str, Any]:
    return {"tool": CORRECT_PARAMETERS, "arguments": {"line_index": int(truth["line_index1"])}}


def breaker_action(truth: Mapping[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {"cb_name": str(truth["cb_name"]), "status": int(bool(truth["true_cb_closed"]))}
    if truth.get("line_index1") is not None:
        # The context offers a dangling-terminal flip with its line; the
        # process gate matches the whole argument set.
        arguments["line_index"] = int(truth["line_index1"])
    return {"tool": CORRECT_TOPOLOGY, "arguments": arguments}


# --------------------------------------------------------------------------- the ungated provider path


class ProviderProbe:
    """The deployed executors and verifier on a root without the environment's process gates.

    The environment refuses a correction its fresh context did not advertise
    (``correction_not_supported_by_current_context``), stands the meter route
    down under branch dominance and demands a confirmation after a commit.
    Those gates are part of the interface; the physics of a correction (what
    the executor estimates, what the solve looks like afterwards, the verdict
    the deployment candidate oracle would give) is read here the way the
    provider's own non-mutating screening reads it.
    """

    def __init__(self, root: Mapping[str, Any], *, profile: str = "scada_only") -> None:
        from psse_env.dagger.release_factories import BC0_CHI2_ALPHA, BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        from psse_env.providers.matpower import MatpowerDeploymentProviders

        self.providers = MatpowerDeploymentProviders(
            chi2_alpha=BC0_CHI2_ALPHA, evidence_profile=profile,
            parameter_ranking_dominance_threshold=BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
            normalized_residual_threshold=4.0,
        )
        execution = root["execution"]
        self.counter = 0
        self.state: dict[str, Any] = {
            "state_id": "probe:s0", "status": "active",
            "case": copy.deepcopy(execution["case"]),
            "measurements": [float(v) for v in execution["measurements"]],
            "metadata": copy.deepcopy(execution.get("metadata") or {}),
        }
        self.parent_solve = self.providers.run_wls(self.state)

    def solve(self) -> dict[str, Any]:
        output = self.providers.run_wls(self.state)
        return {key: output.get(key) for key in _WLS_KEYS if key in output}

    def apply(self, action: Mapping[str, Any], *, row0: int | None = None) -> dict[str, Any]:
        from psse_env.actions import normalize_action
        from psse_env.state_store import apply_modification

        tool = str(action["tool"])
        arguments = {**dict(action.get("arguments") or {}), "state_id": self.state["state_id"]}
        normalized = normalize_action({"tool": tool, "arguments": arguments})
        executor = {CORRECT_MEASUREMENTS: self.providers.correct_measurements, CORRECT_PARAMETERS: self.providers.correct_parameters,
                    CORRECT_TOPOLOGY: self.providers.correct_topology}[tool]
        record: dict[str, Any] = {"tool": tool, "arguments": dict(action.get("arguments") or {}), "path": "provider"}
        correction = executor(self.state, normalized)
        record["status"] = correction.get("execution_status", "success")
        if record["status"] != "success":
            record.update(error_code=correction.get("error_code"), error_detail=str(correction.get("error_detail") or "")[:160])
            return record
        modification = correction.get("modification") or {}
        case, measurements, metadata = apply_modification(
            case=self.state.get("case"), measurements=self.state.get("measurements"),
            metadata=self.state.get("metadata") or {}, modification=modification,
        )
        self.counter += 1
        candidate = {
            **copy.deepcopy(self.state),
            "state_id": f"probe:c{self.counter}", "parent_state_id": self.state["state_id"], "status": "candidate",
            "source_action": normalized, "modification": copy.deepcopy(dict(modification)),
            "case": case, "measurements": measurements, "metadata": metadata,
        }
        candidate.pop("state_hash", None)
        verification = self.providers.run_wls(candidate)
        if verification.get("execution_status", "success") != "success":
            record.update(status="verification_failed", error_code=verification.get("error_code"))
            return record
        parent_score = self.parent_solve.get("chi_square_ratio")
        candidate_score = verification.get("chi_square_ratio")
        try:
            if parent_score is not None and candidate_score is not None:
                denominator = max(abs(float(parent_score)), 1e-12)
                verification["global_progress"] = (float(parent_score) - float(candidate_score)) / denominator
                verification["parent_anomaly_score"] = float(parent_score)
            # The run's progress basis, as the gated environment measures it:
            # the excess of the statistic over the parent's degrees of freedom.
            parent_j, candidate_j = self.parent_solve.get("chi_square_statistic"), verification.get("chi_square_statistic")
            dof = self.parent_solve.get("chi_square_dof")
            if (_PROGRESS_BASIS == "chi_square_excess" and parent_j is not None and candidate_j is not None
                    and dof is not None and float(parent_j) - float(dof) > 0.0):
                excess = float(parent_j) - float(dof)
                verification["global_progress"] = (excess - (float(candidate_j) - float(dof))) / excess
                verification["global_progress_basis"] = "chi_square_excess"
        except (TypeError, ValueError, OverflowError):
            pass
        assessment = self.providers._deployment_candidate_quality_oracle().label_candidate(
            parent_state=self.state, source_action=normalized, candidate_state=candidate,
            verification_output=verification, hidden_truth=None,
        )
        record["verification"] = {key: verification.get(key) for key in _WLS_KEYS if key in verification}
        record["disposition"] = assessment.disposition.value
        record["progress_class"] = assessment.progress_class
        if tool == CORRECT_PARAMETERS:
            record["estimate"] = (float(correction.get("corrected_r")), float(correction.get("corrected_x")))
        if record["disposition"] in ACCEPTING:
            committed = dict(candidate)
            committed.update(state_id=f"probe:s{self.counter}", status="active")
            for key in ("parent_state_id", "source_action", "modification"):
                committed.pop(key, None)
            self.state = committed
            self.parent_solve = verification
            record["lifecycle"] = COMMIT_STATE
        else:
            record["lifecycle"] = ROLLBACK_STATE
        return record

    def final_checks(self, root: Mapping[str, Any]) -> dict[str, Any]:
        return _final_checks_of_state(self.state, root)


def ranking_view(env: Any, lines: Sequence[int]) -> dict[str, Any]:
    output = step(env, GET_PARAMETER_CONTEXT)
    metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
    order = [int(item["line_index1"]) for item in metrics.get("parameter_ranking_distinct_lines") or []]
    return {
        "order": order[:6],
        "dominance_ratio": metrics.get("parameter_ranking_dominance_ratio"),
        "dominant": metrics.get("parameter_ranking_dominant"),
        "route_status": metrics.get("route_status"),
        "supported": [a.get("arguments", {}).get("line_index") for a in metrics.get("supported_corrections") or []],
        "true_line_ranks": {str(line): (order.index(int(line)) + 1 if int(line) in order else None) for line in lines},
    }


def measurement_view(env: Any, indices: Sequence[int]) -> dict[str, Any]:
    output = step(env, GET_MEASUREMENT_CONTEXT)
    metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
    findings = metrics.get("measurement_findings") or []
    order = [int(item.get("index0", item.get("index"))) for item in findings
             if isinstance(item, Mapping) and item.get("index0", item.get("index")) is not None]
    groups = [sorted(int(i) for i in (a.get("arguments", {}).get("suspect_group") or [])) for a in metrics.get("supported_corrections") or []]
    return {
        "residual_order": order[:6],
        "true_meter_ranks": {str(i): (order.index(int(i)) + 1 if int(i) in order else None) for i in indices},
        "supported_groups": groups[:6],
        "true_group_supported": any(set(group) == set(int(i) for i in indices) for group in groups),
    }


def signatures_of(env: Any) -> list[str]:
    observation = env.get_policy_observation().as_dict()
    return [str(s) for s in observation.get("unresolved_signatures") or []]


def dominance_of(signatures: Sequence[str]) -> str:
    if any("wls_residual_outlier_dominant" in s for s in signatures):
        return "residual"
    if any("wls_branch_multiplier_dominant" in s for s in signatures):
        return "branch"
    return "none"


# --------------------------------------------------------------------------- experiments


def _sequence(root: Mapping[str, Any], actions: Sequence[tuple[str, Mapping[str, Any], int | None]],
              *, truths: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One correction sequence on the gated environment and on the ungated provider path."""
    tolerance = tolerance_of(root)
    env = new_env(root)
    steps = []
    for label, action, row0 in actions:
        record = apply(env, action, row0=row0)
        record["label"] = label
        if record.get("estimate") is not None and truths and label in truths:
            record["estimate_error"] = estimate_error(tuple(record["estimate"]), truths[label], tolerance)
        steps.append(record)
    probe = ProviderProbe(root)
    provider_steps = []
    for label, action, row0 in actions:
        record = probe.apply(action, row0=row0)
        record["label"] = label
        if record.get("estimate") is not None and truths and label in truths:
            record["estimate_error"] = estimate_error(tuple(record["estimate"]), truths[label], tolerance)
        provider_steps.append(record)
    return {"steps": steps, "final": solve(env), "checks": _final_checks(env, root),
            "provider": {"steps": provider_steps, "final": probe.solve(), "checks": probe.final_checks(root)}}


def _final_checks(env: Any, root: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the end state would pass the truth audit's final-case and final-measurement checks."""
    return _final_checks_of_state(env.store.get_state(active_id(env)), root)


def _final_checks_of_state(state: Mapping[str, Any], root: Mapping[str, Any]) -> dict[str, Any]:
    from mcp_server.matpower_server import _load_python_case

    truth = truth_of(root)
    tolerance = tolerance_of(root)
    case = state.get("case")
    ppc = _load_python_case(case) if isinstance(case, str) else case
    case_ok = True
    for item in truth.get("true_parameter_errors") or []:
        row = ppc["branch"][int(item["branch_row0"])]
        if abs(float(row[2]) - float(item["clean_r"])) > tolerance or abs(float(row[3]) - float(item["clean_x"])) > tolerance:
            case_ok = False
    for item in truth.get("true_topology_errors") or []:
        if item.get("branch_row0") is not None and int(float(ppc["branch"][int(item["branch_row0"])][10])) != int(item.get("expected_status", 0)):
            case_ok = False
    clean = np.asarray(truth.get("clean_measurements") or [], dtype=float)
    measurements = np.asarray(state.get("measurements") or [], dtype=float)
    sigma = np.asarray((root["execution"].get("metadata") or {}).get("sigma_z") or [], dtype=float)
    measurements_ok = None
    if clean.shape == measurements.shape and clean.size:
        bound = 3.0 * np.sqrt(2.0) * float(np.max(sigma)) if sigma.size else 0.05
        measurements_ok = bool(np.max(np.abs(measurements - clean)) <= bound)
    return {"case_ok": case_ok, "measurements_ok": measurements_ok, "tolerance": tolerance}


def twin_root(root: Mapping[str, Any], *, seed: int) -> dict[str, Any] | None:
    """The same root with its meter error moved from the faulted line to a healthy channel."""
    from psse_env.providers.matpower import measurement_index_map

    truth = truth_of(root)
    meters = truth.get("true_measurement_errors") or []
    if len(meters) != 1 or not truth.get("clean_measurements"):
        return None
    clean = [float(v) for v in truth["clean_measurements"]]
    measurements = [float(v) for v in root["execution"]["measurements"]]
    if len(clean) != len(measurements):
        return None
    error = meters[0]
    i = int(error["index"])
    offset = measurements[i] - clean[i]
    nb, nl = system_size(root)
    index_map = measurement_index_map(nb, nl)
    structural = set(int(v) for v in (root["execution"].get("metadata") or {}).get("structural_zero_indices") or [])
    blocked = set(structural) | {i}
    for item in truth.get("true_parameter_errors") or []:
        for channel in ("Pf", "Qf", "Pt", "Qt"):
            blocked.add(index_map[channel].start + int(item["branch_row0"]))
    eligible = [k for channel in ("Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt") for k in range(index_map[channel].start, index_map[channel].stop)
                if k not in blocked]
    j = random.Random(f"{seed}:{root['execution']['scenario_id']}").choice(eligible)
    twin = copy.deepcopy(dict(root))
    twin["execution"]["measurements"] = list(clean)
    twin["execution"]["measurements"][j] = clean[j] + offset
    channel = next(name for name, span in index_map.items() if span.start <= j < span.stop)
    twin["audit"]["truth"]["true_measurement_errors"] = [{"index": j, "channel": channel, "observed": clean[j] + offset, "clean": clean[j]}]
    twin["execution"]["scenario_id"] = f"{root['execution']['scenario_id']}:twin"
    return twin


def same_line_meter(root: Mapping[str, Any]) -> dict[str, Any]:
    truth = parameter_truths(root)[0]
    row0 = int(truth["branch_row0"])
    tolerance = tolerance_of(root)
    meters = sorted(int(item["index"]) for item in truth_of(root).get("true_measurement_errors") or [])

    def probe(target: Mapping[str, Any]) -> dict[str, Any]:
        env = new_env(target)
        base = {"signatures": signatures_of(env)[:6], "dominance": dominance_of(signatures_of(env)),
                "solve": solve(env), "ranking": ranking_view(env, [truth["line_index1"]]),
                "measurement": measurement_view(env, meters)}
        results: dict[str, Any] = {"base": base, "sequences": {}}
        plans = {
            "meter": [("meter", meter_action(target), None)],
            "parameter": [("parameter", parameter_action(truth), row0)],
            "meter_then_parameter": [("meter", meter_action(target), None), ("parameter", parameter_action(truth), row0)],
            "parameter_then_meter": [("parameter", parameter_action(truth), row0), ("meter", meter_action(target), None)],
        }
        for name, actions in plans.items():
            results["sequences"][name] = _sequence(target, actions, truths={"parameter": truth})
        return results

    out = {"experiment": "same_line_meter", "root": root["execution"]["scenario_id"], "class": compound_of(root).get("class"),
           "line": int(truth["line_index1"]), "meter_indices": meters,
           "solo_effects": compound_of(root).get("solo_effects"), "original": probe(root)}
    twin = twin_root(root, seed=20261027)
    if twin is not None:
        out["twin"] = probe(twin)
        out["twin_meter_indices"] = [int(item["index"]) for item in twin["audit"]["truth"]["true_measurement_errors"]]
    return out


def same_terminal_breaker(root: Mapping[str, Any]) -> dict[str, Any]:
    from psse_env.providers.matpower import measurement_index_map

    line = parameter_truths(root)[0]
    breaker = dict((truth_of(root).get("true_topology_errors") or [])[0])
    row0 = int(line["branch_row0"])
    tolerance = tolerance_of(root)
    nb, nl = system_size(root)
    measurements = [float(v) for v in root["execution"]["measurements"]]
    index_map = measurement_index_map(nb, nl)
    sigma = [float(v) for v in (root["execution"].get("metadata") or {}).get("sigma_z") or []]
    flows = {channel: measurements[index_map[channel].start + row0] for channel in ("Pf", "Qf", "Pt", "Qt")}
    flow_sigmas = {channel: (abs(flows[channel]) / sigma[index_map[channel].start + row0]) if sigma else None for channel in flows}
    env = new_env(root)
    base = {"signatures": signatures_of(env)[:6], "dominance": dominance_of(signatures_of(env)), "solve": solve(env),
            "ranking": ranking_view(env, [line["line_index1"]])}
    plans = {
        "breaker": [("breaker", breaker_action(breaker), None)],
        "parameter": [("parameter", parameter_action(line), row0)],
        "breaker_then_parameter": [("breaker", breaker_action(breaker), None), ("parameter", parameter_action(line), row0)],
        "parameter_then_breaker": [("parameter", parameter_action(line), row0), ("breaker", breaker_action(breaker), None)],
    }
    sequences = {name: _sequence(root, actions, truths={"parameter": line}) for name, actions in plans.items()}
    return {"experiment": "same_terminal_breaker", "root": root["execution"]["scenario_id"], "class": compound_of(root).get("class"),
            "line": int(line["line_index1"]), "breaker": breaker["cb_name"], "line_flows": flows, "line_flow_sigmas": flow_sigmas,
            "solo_effects": compound_of(root).get("solo_effects"), "base": base, "sequences": sequences}


def adjacent_lines(root: Mapping[str, Any]) -> dict[str, Any]:
    truths = parameter_truths(root)
    tolerance = tolerance_of(root)
    lines = [int(t["line_index1"]) for t in truths]
    env = new_env(root)
    base = {"signatures": signatures_of(env)[:6], "dominance": dominance_of(signatures_of(env)), "solve": solve(env),
            "ranking": ranking_view(env, lines)}
    # The bounded comparison on the context's top candidates (what the teacher would compare).
    order = base["ranking"]["order"]
    comparison = None
    # The comparison takes the context's supported candidates (the ambiguous
    # top two, or every ranked line when one dominates), at most three.
    supported = [int(line) for line in base["ranking"]["supported"] if isinstance(line, int)]
    candidates = [line for line in order if line in supported][:3]
    if len(candidates) >= 2:
        output = step(env, COMPARE_PARAMETER_CANDIDATES, line_indices=candidates)
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        comparison = {
            "status": output.get("execution_status"), "error_code": output.get("error_code"),
            "requested": candidates, "best_line": metrics.get("best_line_index"),
            "ranked": metrics.get("ranked_line_indices"),
            "verdicts": {str(item.get("line_index")): item.get("verification_verdict") for item in metrics.get("candidates") or []},
            "best_is_true": (int(metrics["best_line_index"]) in lines) if metrics.get("best_line_index") is not None else None,
        }
    # Sequential single-line estimates: the better-ranked true line first on
    # the stale model, the other on the derived model, then the first again.
    ranked_true = sorted(truths, key=lambda t: (base["ranking"]["true_line_ranks"].get(str(t["line_index1"])) or 99))
    first, second = ranked_true[0], ranked_true[1]
    run = _sequence(root, [
        ("first", parameter_action(first), int(first["branch_row0"])),
        ("second", parameter_action(second), int(second["branch_row0"])),
        ("first_again", parameter_action(first), int(first["branch_row0"])),
    ], truths={"first": first, "second": second, "first_again": first})
    relation = (compound_of(root).get("relations") or [{}])[0]
    return {"experiment": "adjacent_lines", "root": root["execution"]["scenario_id"], "class": compound_of(root).get("class"),
            "lines": lines, "relation": relation.get("relation"), "bus_distance": relation.get("bus_distance"),
            "solo_effects": compound_of(root).get("solo_effects"), "base": base, "comparison": comparison,
            "sequential": {"first_line": int(first["line_index1"]), "second_line": int(second["line_index1"]), **run}}


def breaker_masking(root: Mapping[str, Any]) -> dict[str, Any]:
    """The context's top parameter line corrected on a breaker root: what the estimator does and what the verifier says."""
    from mcp_server.matpower_server import _load_python_case

    truth = truth_of(root)
    breakers = [dict(item) for item in truth.get("true_topology_errors") or []]
    lines = [int(t["line_index1"]) for t in parameter_truths(root)]
    dangling = [int(b["line_index1"]) for b in breakers if b.get("line_index1") is not None]
    env = new_env(root)
    base = {"signatures": signatures_of(env)[:6], "dominance": dominance_of(signatures_of(env)), "solve": solve(env),
            "ranking": ranking_view(env, lines or dangling)}
    top = base["ranking"]["order"][0] if base["ranking"]["order"] else None
    nominal = _load_python_case(root["execution"]["case"])["branch"]
    record: dict[str, Any] = {"experiment": "breaker_masking", "root": root["execution"]["scenario_id"], "class": compound_of(root).get("class"),
                              "breakers": [b["cb_name"] for b in breakers], "effects": [b["physical_effect"] for b in breakers],
                              "dangling_lines": dangling, "true_lines": lines, "top_line": top,
                              "top_is_dangling_line": (top in dangling) if top is not None else None,
                              "top_is_true_line": (top in lines) if top is not None else None,
                              "top_touches_breaker_bus": None, "base": base}
    if top is not None:
        from_bus, to_bus = int(nominal[top - 1][0]), int(nominal[top - 1][1])
        buses = {int(b) for item in breakers for b in (item.get("affected_planning_buses") or [])}
        record["top_touches_breaker_bus"] = bool({from_bus, to_bus} & buses)
        probe = ProviderProbe(root)
        step_record = probe.apply({"tool": CORRECT_PARAMETERS, "arguments": {"line_index": int(top)}}, row0=top - 1)
        estimate = step_record.get("estimate")
        if estimate is not None:
            r0, x0 = float(nominal[top - 1][2]), float(nominal[top - 1][3])
            step_record["estimate_ratio"] = {"r": estimate[0] / r0 if r0 else None, "x": estimate[1] / x0 if x0 else None}
        record["top_line_parameter"] = step_record
        gated = apply(env, {"tool": CORRECT_PARAMETERS, "arguments": {"line_index": int(top)}}, row0=top - 1)
        record["top_line_parameter_gated"] = {k: gated.get(k) for k in ("status", "error_code", "disposition", "verification")}
    breaker_probe = ProviderProbe(root)
    record["breakers_alone"] = [breaker_probe.apply(breaker_action(item)) for item in breakers]
    record["breakers_alone_final"] = breaker_probe.solve()
    return record


def summarize_breaker_masking(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if not r.get("error"):
            groups[str(r.get("class"))].append(r)
    for name, rows in sorted(groups.items()):
        entry: dict[str, Any] = {"roots": len(rows)}
        entry["dominance"] = dict(Counter(r["base"]["dominance"] for r in rows))
        entry["parameter_route_actionable"] = sum(1 for r in rows if r["base"]["ranking"]["route_status"] == "actionable")
        entry["top_is_dangling_line"] = sum(1 for r in rows if r.get("top_is_dangling_line"))
        entry["top_is_true_line"] = sum(1 for r in rows if r.get("top_is_true_line"))
        entry["top_touches_breaker_bus"] = sum(1 for r in rows if r.get("top_touches_breaker_bus"))
        steps = [r.get("top_line_parameter") or {} for r in rows]
        entry["top_line_verdicts"] = dict(Counter(_verdict(s) for s in steps))
        ratios = [s.get("estimate_ratio") for s in steps if s.get("estimate_ratio")]
        entry["top_line_estimates"] = len(ratios)
        entry["top_line_x_ratio_median"] = _median([v["x"] for v in ratios if v.get("x") is not None])
        entry["top_line_x_ratio_above_3"] = sum(1 for v in ratios if (v.get("x") or 0) > 3.0)
        entry["top_line_x_ratio_above_10"] = sum(1 for v in ratios if (v.get("x") or 0) > 10.0)
        entry["top_line_gated_verdicts"] = dict(Counter(_verdict(r.get("top_line_parameter_gated") or {}) for r in rows))
        breaker_steps = [s for r in rows for s in r.get("breakers_alone") or []]
        entry["breaker_alone_verdicts"] = dict(Counter(_verdict(s) for s in breaker_steps))
        entry["breakers_alone_resolved"] = sum(1 for r in rows if (r.get("breakers_alone_final") or {}).get("globally_resolved") is True)
        out[name] = entry
    return out


EXPERIMENT_FUNCTIONS = {"same_line_meter": same_line_meter, "same_terminal_breaker": same_terminal_breaker, "adjacent_lines": adjacent_lines,
                        "breaker_masking": breaker_masking}


def _run(payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        record = EXPERIMENT_FUNCTIONS[payload["experiment"]](payload["root"])
    except Exception as exc:
        record = {"experiment": payload["experiment"], "root": payload["root"]["execution"]["scenario_id"],
                  "class": compound_of(payload["root"]).get("class"), "error": f"{type(exc).__name__}: {exc}"}
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


# --------------------------------------------------------------------------- summaries


def _mean(values: Sequence[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    values = sorted(float(v) for v in values if v is not None)
    return values[len(values) // 2] if values else None


def _view(sequence: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    """The gated environment view of a sequence, or the ungated provider view."""
    return sequence.get("provider") or {} if path == "provider" else sequence


def _step_of(sequence: Mapping[str, Any], label: str, path: str = "env") -> Mapping[str, Any]:
    return next((s for s in _view(sequence, path).get("steps") or [] if s.get("label") == label), {})


def _verdict(step: Mapping[str, Any]) -> str:
    return str(step.get("disposition") or step.get("error_code") or step.get("status") or "failed")


def _sequence_cell(seqs: Sequence[Mapping[str, Any]], labels: Sequence[str], path: str) -> dict[str, Any]:
    views = [_view(s, path) for s in seqs]
    cell: dict[str, Any] = {"n": len(seqs)}
    for label in labels:
        steps = [_step_of(s, label, path) for s in seqs]
        cell[f"{label}_verdicts"] = dict(Counter(_verdict(s) for s in steps))
        estimates = [s.get("estimate_error") for s in steps if s.get("estimate_error")]
        if estimates:
            cell[f"{label}_estimates"] = len(estimates)
            cell[f"{label}_within_tolerance"] = sum(1 for e in estimates if e["within_tolerance"])
            cell[f"{label}_median_rel_r"] = _median([e["rel_r"] for e in estimates])
            cell[f"{label}_median_rel_x"] = _median([e["rel_x"] for e in estimates])
    cell["final_resolved"] = sum(1 for v in views if (v.get("final") or {}).get("globally_resolved") is True)
    cell["case_ok"] = sum(1 for v in views if (v.get("checks") or {}).get("case_ok"))
    cell["measurements_ok"] = sum(1 for v in views if (v.get("checks") or {}).get("measurements_ok"))
    cell["both_ok"] = sum(1 for v in views if (v.get("checks") or {}).get("case_ok") and (v.get("checks") or {}).get("measurements_ok"))
    return cell


def summarize_same_line_meter(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("original", "twin"):
        rows = [r[arm] for r in records if arm in r and not r.get("error")]
        if not rows:
            continue
        entry: dict[str, Any] = {"roots": len(rows)}
        entry["dominance"] = dict(Counter(r["base"]["dominance"] for r in rows))
        entry["true_line_rank_1"] = sum(1 for r in rows if r["base"]["ranking"]["true_line_ranks"] and list(r["base"]["ranking"]["true_line_ranks"].values())[0] == 1)
        entry["true_meter_rank_1"] = sum(1 for r in rows if r["base"]["measurement"]["true_meter_ranks"] and list(r["base"]["measurement"]["true_meter_ranks"].values())[0] == 1)
        entry["true_meter_group_supported"] = sum(1 for r in rows if r["base"]["measurement"]["true_group_supported"])
        entry["parameter_route_actionable"] = sum(1 for r in rows if r["base"]["ranking"]["route_status"] == "actionable")
        for path in ("env", "provider"):
            for name in ("meter", "parameter", "meter_then_parameter", "parameter_then_meter"):
                seqs = [r["sequences"][name] for r in rows if name in r["sequences"]]
                entry[f"{path}:{name}"] = _sequence_cell(seqs, name.split("_then_"), path)
        out[arm] = entry
    return out


def summarize_same_terminal_breaker(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [r for r in records if not r.get("error")]
    entry: dict[str, Any] = {"roots": len(rows)}
    entry["dominance"] = dict(Counter(r["base"]["dominance"] for r in rows))
    entry["true_line_rank_1"] = sum(1 for r in rows if list(r["base"]["ranking"]["true_line_ranks"].values())[0] == 1)
    entry["parameter_route_actionable"] = sum(1 for r in rows if r["base"]["ranking"]["route_status"] == "actionable")
    entry["line_flow_sigmas_median"] = {channel: _median([r["line_flow_sigmas"][channel] for r in rows]) for channel in ("Pf", "Qf", "Pt", "Qt")}
    entry["line_flows_below_3_sigma"] = sum(1 for r in rows if all((r["line_flow_sigmas"][c] or 0) < 3.0 for c in ("Pf", "Qf", "Pt", "Qt")))
    for path in ("env", "provider"):
        for name in ("breaker", "parameter", "breaker_then_parameter", "parameter_then_breaker"):
            seqs = [r["sequences"][name] for r in rows if name in r["sequences"]]
            entry[f"{path}:{name}"] = _sequence_cell(seqs, name.split("_then_"), path)
    return entry


def summarize_adjacent_lines(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if not r.get("error"):
            groups[str(r.get("class"))].append(r)
    for name, rows in sorted(groups.items()):
        entry: dict[str, Any] = {"roots": len(rows)}
        ranks = [sorted(v for v in r["base"]["ranking"]["true_line_ranks"].values() if v is not None) for r in rows]
        entry["both_lines_in_top_2"] = sum(1 for v in ranks if len(v) == 2 and max(v) <= 2)
        entry["both_lines_in_top_3"] = sum(1 for v in ranks if len(v) == 2 and max(v) <= 3)
        entry["one_line_unranked"] = sum(1 for v in ranks if len(v) < 2)
        entry["top_ranked_is_true"] = sum(1 for r in rows if r["base"]["ranking"]["order"] and r["base"]["ranking"]["order"][0] in r["lines"])
        entry["dominant"] = sum(1 for r in rows if r["base"]["ranking"]["dominant"] is True)
        entry["dominance_ratio_median"] = _median([r["base"]["ranking"]["dominance_ratio"] for r in rows])
        comparisons = [r["comparison"] for r in rows if r.get("comparison")]
        entry["comparisons"] = len(comparisons)
        entry["comparison_best_is_true"] = sum(1 for c in comparisons if c.get("best_is_true"))
        entry["comparison_best_none"] = sum(1 for c in comparisons if c.get("best_line") is None)
        seq = [r["sequential"] for r in rows]
        for path in ("env", "provider"):
            entry[f"{path}:sequential"] = _sequence_cell(seq, ("first", "second", "first_again"), path)
        out[name] = entry
    return out


SUMMARIZERS = {"same_line_meter": summarize_same_line_meter, "same_terminal_breaker": summarize_same_terminal_breaker,
               "adjacent_lines": summarize_adjacent_lines, "breaker_masking": summarize_breaker_masking}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="scada_only")
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--system", default=None, help="the suite's registered system (default: read from the roots)")
    parser.add_argument("--progress-basis", default=None)
    args = parser.parse_args(argv)

    from research.compound_errors.rollout import suite_system, system_progress_basis

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    if isinstance(suite, Mapping):
        suite = suite.get("scenarios") or suite.get("roots") or []
    system = suite_system(suite, args.system)
    progress_basis = system_progress_basis(system, args.progress_basis)
    payloads = []
    for experiment in args.experiments:
        taken: Counter = Counter()
        for root in suite:
            name = compound_of(root).get("class")
            if name not in EXPERIMENT_CLASSES[experiment]:
                continue
            if args.per_class and taken[name] >= args.per_class:
                continue
            taken[name] += 1
            payloads.append({"experiment": experiment, "root": root})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[identifiability] {len(payloads)} {system} probes over {args.experiments} under {args.profile}, "
          f"progress basis {progress_basis or 'default'}, workers {args.workers}", flush=True)
    started = time.perf_counter()
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    streams = {experiment: (out / f"{experiment}.jsonl").open("w", encoding="utf-8") for experiment in args.experiments}
    try:
        def emit(record: Mapping[str, Any]) -> None:
            records[record["experiment"]].append(dict(record))
            streams[record["experiment"]].write(json.dumps(record, sort_keys=True, default=str) + "\n")
            streams[record["experiment"]].flush()
            done = sum(len(v) for v in records.values())
            if done % 20 == 0 or done == len(payloads):
                print(json.dumps({"done": done, "elapsed": round(time.perf_counter() - started, 1),
                                  "errors": sum(1 for v in records.values() for r in v if r.get("error"))}), flush=True)

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.profile, progress_basis)) as pool:
                for record in pool.map(_run, payloads, chunksize=1):
                    emit(record)
        else:
            _init_worker(args.profile, progress_basis)
            for payload in payloads:
                emit(_run(payload))
    finally:
        for stream in streams.values():
            stream.close()
    summary = {experiment: SUMMARIZERS[experiment](records[experiment]) for experiment in args.experiments}
    summary.update(contract=CONTRACT, suite=str(args.suite), system=system, progress_basis=progress_basis, profile=args.profile,
                   errors={experiment: sum(1 for r in records[experiment] if r.get("error")) for experiment in args.experiments},
                   total_seconds=round(time.perf_counter() - started, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
