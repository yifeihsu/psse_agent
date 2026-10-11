"""The basis of a candidate's global progress (psse_env.transactional_env, 2026-10-11).

Under the dual anomaly test the environment measures the progress of a correction as the
fraction of the parent's chi-square statistic it removes; ``chi_square_excess`` measures the
fraction of the excess over the degrees of freedom instead, which does not shrink with the
network's size.  A fake WLS adapter plays a 118-bus-sized solve: dof 863, five bad meters
worth about 120 each, so one repair removes 9% of the statistic but 21% of the excess.
"""
from __future__ import annotations

import pytest

from psse_env.actions import (
    CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT, RUN_WLS,
)
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE
from psse_env.transactional_env import DEFAULT_PROGRESS_BASIS, PROGRESS_BASES, TransactionalPSSEEnv

DOF, PER_METER = 863.0, 120.0
REQUIRED = {RUN_WLS, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT,
            CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY}


def _deterministic(function):
    function.provider_kind = "deterministic"
    return function


@_deterministic
def _wls(state):
    """Each remaining bad meter (value 9.0, clean 1.0) adds PER_METER to the statistic."""
    bad = sum(1 for value in (state.get("measurements") or []) if value != 1.0)
    statistic = DOF + PER_METER * bad
    return {
        "state_id": state["state_id"], "state_hash": state["state_hash"], "evidence_source": "deployment_wls:test",
        "wls_objective": statistic, "chi_square_statistic": statistic, "chi_square_dof": int(DOF),
        "chi_square_threshold": 962.6, "chi_square_alarm": bad > 0,
        "max_normalized_residual": 12.0 if bad else 2.0, "normalized_residual_alarm": bad > 0, "normalized_residual_threshold": 4.0,
        "remaining_anomaly_score": 3.0 if bad else 0.5, "anomaly_threshold": 1.0,
        "target_progress": 1.0 if state.get("parent_state_id") else 0.0, "target_fixed": True if state.get("parent_state_id") else None,
        "remaining_suspect_count": bad, "globally_resolved": bad == 0, "physical_constraints_ok": True,
        "new_constraint_violations": 0, "unresolved_signatures": ["measurement_residual_outlier"] if bad else [], "converged": True,
    }


@_deterministic
def _context(state):
    return {"context_rows": [{"state_hash": state["state_hash"]}], "unresolved_signatures": ["measurement_residual_outlier"],
            "supported_corrections": [{"tool": CORRECT_MEASUREMENTS,
                                       "arguments": {"state_id": state["state_id"], "measurement_updates": {0: 1.0}}}]}


@_deterministic
def _correct(state, action):
    del state
    return {"modification": {k: v for k, v in action["arguments"].items() if k not in {"state_id", "candidate_state_id"}},
            "executor_receipt": "progress_basis_test_v1"}


def _env(basis):
    return TransactionalPSSEEnv(
        evidence_profile=AUXILIARY_EVIDENCE_PROFILE, production_dataset_mode=True, approved_deterministic_providers=REQUIRED,
        wls_runner=_wls, context_providers={GET_MEASUREMENT_CONTEXT: _context, GET_PARAMETER_CONTEXT: _context, GET_TOPOLOGY_CONTEXT: _context},
        correction_executors={CORRECT_MEASUREMENTS: _correct, CORRECT_PARAMETERS: _correct, CORRECT_TOPOLOGY: _correct},
        max_steps=8, progress_basis=basis,
    )


def _one_repair(basis):
    env = _env(basis)
    root = env.reset({"scenario_id": f"progress-basis-{basis}", "case": {}, "measurements": [9.0] * 5,
                      "clean_measurements": [1.0] * 5, "true_measurement_errors": [{"index": i, "clean": 1.0} for i in range(5)]})
    active = root["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    env.step({"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active}})
    candidate, output = env.step({"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": active, "measurement_updates": {0: 1.0}}})
    assert output["execution_status"] == "success"
    _, verification = env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate["candidate_state_id"]}})
    return verification["tool_metrics"]


def test_the_default_basis_is_the_whole_statistic():
    assert DEFAULT_PROGRESS_BASIS == "chi_square_statistic" and set(PROGRESS_BASES) == {"chi_square_statistic", "chi_square_excess"}
    metrics = _one_repair("chi_square_statistic")
    assert metrics["global_progress_basis"] == "chi_square_statistic"
    assert metrics["global_progress"] == pytest.approx(PER_METER / (DOF + 5 * PER_METER))  # 8.2%: under every partial floor


def test_the_excess_basis_measures_the_anomaly_energy():
    metrics = _one_repair("chi_square_excess")
    assert metrics["global_progress_basis"] == "chi_square_excess"
    assert metrics["global_progress"] == pytest.approx(1.0 / 5.0)  # one of five equal meters


def test_an_unknown_basis_is_refused():
    with pytest.raises(ValueError, match="progress_basis"):
        _env("excess")
