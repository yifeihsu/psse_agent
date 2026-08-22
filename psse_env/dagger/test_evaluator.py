from __future__ import annotations

import copy
import hashlib
import io
import json
import platform
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest import mock

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    FINALIZE_DIAGNOSIS,
    GET_HARMONIC_CONTEXT,
    INVALID_ACTION,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_HSE_FROM_PATH,
    RUN_WLS,
    VERIFY_CANDIDATE,
)
from psse_env.dagger.evaluator import (
    ClosedLoopRolloutEvaluator,
    STUDY_EVALUATION_SCHEMA_VERSION,
    _runtime_environment_descriptor,
    build_evaluation_provenance,
    evaluate_rollout_suites,
    main as evaluator_main,
    strip_offline_truth,
    write_evaluation_artifact,
)
from psse_env.dagger.evaluation_gate import (
    _episode_safety_ordinal,
    _intervention_failures,
)
from psse_env.sft.release_hardware import normalize_accelerator_class


class _ScriptPolicy:
    def __init__(
        self,
        observations: list[dict[str, Any]],
        *,
        release_policy_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.observations = observations
        if release_policy_identity is not None:
            self.release_policy_identity = copy.deepcopy(
                dict(release_policy_identity)
            )

    def act(self, observation: Mapping[str, Any]) -> Any:
        payload = copy.deepcopy(dict(observation))
        self.observations.append(payload)
        phase = payload["phase"]
        if phase == "malformed":
            return []
        actions = {
            "wls": {"tool": RUN_WLS, "arguments": {"state_id": "active"}},
            "partial_commit": {
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": "candidate-1"},
            },
            "hse": {
                "tool": RUN_HSE_FROM_PATH,
                "arguments": {"state_id": "active"},
            },
            "finalize": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "escalate": {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": "active",
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            },
            "false_commit": {
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": "bad-candidate"},
            },
            "false_rollback": {
                "tool": ROLLBACK_STATE,
                "arguments": {"candidate_state_id": "good-candidate"},
            },
            "parameter": {
                "tool": CORRECT_PARAMETERS,
                "arguments": {"state_id": "active", "line_index1": 2},
            },
        }
        return actions[phase]


def _cli_policy_factory(
    *,
    policy_identity: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> _ScriptPolicy:
    return _ScriptPolicy(
        [],
        release_policy_identity={
            "explicit_policy_identity": policy_identity,
            "model_id": model_id,
            "model_revision": model_revision,
        },
    )


def _unattested_policy_factory(
    *, model_id: str | None = None, model_revision: str | None = None
) -> _ScriptPolicy:
    del model_id, model_revision
    return _ScriptPolicy([])


def _cli_case_loader(value: Any) -> Any:
    return copy.deepcopy(value)


class _PhysicalStore:
    def __init__(self, env: "_ScriptEnv") -> None:
        self.env = env
        self.active_state_id = "active"

    def get_state(self, state_id: str) -> dict[str, Any]:
        if state_id != self.active_state_id:
            raise KeyError(state_id)
        return copy.deepcopy(self.env.physical_state)


class _DeploymentCandidateOracle:
    mode = "deployment"


class _ScriptEnv:
    # This scripted environment is a release-contract test double: its
    # transitions are observable-script driven and its oracle exposes no
    # hidden scenario truth.
    production_dataset_mode = True
    candidate_quality_oracle = _DeploymentCandidateOracle()

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed
        self.cursor = 0
        self.terminal = False
        self.terminal_outcome: str | None = None
        self.accepted: list[dict[str, Any]] = []
        self.explanations: list[dict[str, Any]] = []
        self.scenario: dict[str, Any] = {}
        self.physical_state: dict[str, Any] = {}
        self.store = _PhysicalStore(self)

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        self.scenario = copy.deepcopy(dict(scenario))
        self.cursor = 0
        self.terminal = False
        self.terminal_outcome = None
        self.accepted = []
        self.explanations = []
        self.physical_state = copy.deepcopy(
            dict(self.scenario.get("initial_physical_state") or {})
        )
        return self.current_state()

    def _current_script(self) -> dict[str, Any]:
        script = self.scenario["script"]
        if self.cursor < len(script):
            return script[self.cursor]
        return script[-1]

    def get_policy_observation(self, history: list[Mapping[str, Any]]) -> dict[str, Any]:
        row = self._current_script()
        return {
            "active_state_id": "active",
            "candidate_state_id": row.get("candidate_state_id"),
            "remaining_budget": len(self.scenario["script"]) - self.cursor,
            "phase": row["phase"],
            "history_window": copy.deepcopy(history[-2:]),
        }

    def get_oracle_state(self, history: list[Mapping[str, Any]]) -> dict[str, Any]:
        row = self._current_script()
        truth = {
            "truth_complete": True,
            "remaining_true_fault_count": row.get("remaining"),
        }
        return {
            "candidate_disposition": row.get("disposition"),
            "candidate_assessment": row.get("assessment", {}),
            "hidden_truth": truth,
        }

    def step(self, action: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        row = self._current_script()
        status = row.get("status", "success")
        if (
            action.get("tool") == COMMIT_STATE
            and status == "success"
            and row.get("accepted_action")
        ):
            self.accepted.append(
                {
                    "candidate_state_id": row.get("candidate_state_id"),
                    "source_action": copy.deepcopy(row["accepted_action"]),
                }
            )
        if row.get("explanation"):
            self.explanations.append(copy.deepcopy(row["explanation"]))
        self.cursor += 1
        if row.get("terminal_outcome"):
            self.terminal = True
            self.terminal_outcome = row["terminal_outcome"]
        tool_metrics: dict[str, Any] = {}
        if action.get("tool") in {RUN_WLS, VERIFY_CANDIDATE} and status == "success":
            state_id = str(action.get("arguments", {}).get("state_id") or "")
            tool_metrics = {
                "state_id": state_id,
                "state_hash": hashlib.sha256(
                    f"{self.scenario.get('scenario_id')}:{self.cursor}:{state_id}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        return self.current_state(), {
            "execution_status": status,
            "error_code": row.get("error_code"),
            "state_mutated": status == "success" and action.get("tool") == COMMIT_STATE,
            "tool_metrics": tool_metrics,
        }

    def current_state(self) -> dict[str, Any]:
        return {
            "active_state_id": "active",
            "candidate_state_id": (
                self._current_script().get("candidate_state_id")
                if self.scenario
                else None
            ),
            "accepted_corrections": copy.deepcopy(self.accepted),
            "explained_anomalies": copy.deepcopy(self.explanations),
        }

    def is_terminal(self, state: Mapping[str, Any] | None = None) -> bool:
        return self.terminal


class _AuditedHandoffEnv(_ScriptEnv):
    """Expose the exact observable post-correction handoff certificate."""

    _STATE_HASH = "a" * 64

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        self.transition_history: list[dict[str, Any]] = []
        self.last_action: dict[str, Any] | None = None
        self.last_output: dict[str, Any] = {}
        state = super().reset(scenario)
        self.physical_state["state_id"] = "active"
        self.physical_state["state_hash"] = self._STATE_HASH
        return state

    def step(self, action: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        _, output = super().step(action)
        if action.get("tool") == ASK_FOR_MORE_EVIDENCE and self.terminal:
            output = {
                "execution_status": "success",
                "error_code": None,
                "error_detail": None,
                "state_mutated": False,
                "active_state_id": "active",
                "candidate_state_id": None,
                "tool_metrics": {
                    "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                    "state_id": "active",
                    "state_hash": self._STATE_HASH,
                    "terminal_outcome": "operator_escalation",
                    "operator_review_required": True,
                    "additional_evidence_available": False,
                    "operator_escalation_audit": {
                        "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                        "active_state_id": "active",
                        "active_state_hash": self._STATE_HASH,
                        "post_correction_confirmation_handoff": True,
                        "post_correction_confirmation_deferred": False,
                        "operator_review_required": True,
                        "additional_evidence_available": False,
                        "missing_required_contexts": [],
                        "outstanding_recovery_targets": [],
                        "unexplained_signature_count": 1,
                    },
                },
            }
        self.last_action = copy.deepcopy(dict(action))
        self.last_output = copy.deepcopy(output)
        self.transition_history.append(
            {
                "state_id": "active",
                "action": copy.deepcopy(dict(action)),
                "tool_output": copy.deepcopy(output),
                "transition_label": {
                    "execution_status": output.get("execution_status"),
                    "process_valid": output.get("execution_status") == "success",
                    "error_code": output.get("error_code"),
                    "error_detail": output.get("error_detail"),
                },
            }
        )
        return self.current_state(), copy.deepcopy(output)

    def current_state(self) -> dict[str, Any]:
        state = super().current_state()
        last_action = getattr(self, "last_action", None)
        state.update(
            {
                "has_open_candidate": state.get("candidate_state_id") is not None,
                "has_unverified_candidate": False,
                "has_verified_candidate": False,
                "history_window": copy.deepcopy(
                    getattr(self, "transition_history", [])
                ),
                "last_tool": (
                    last_action.get("tool")
                    if isinstance(last_action, Mapping)
                    else None
                ),
                "last_tool_output": copy.deepcopy(
                    getattr(self, "last_output", {})
                ),
                "last_tool_status": (
                    getattr(self, "last_output", {}).get("execution_status")
                ),
                "unresolved_signatures": (
                    [POST_CORRECTION_CONFIRMATION_SIGNATURE]
                    if self.terminal
                    else []
                ),
            }
        )
        return state


class _StaticHandoffProcessOracle:
    def __init__(self, *, process_valid: bool) -> None:
        self.process_valid = bool(process_valid)

    def check(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        *,
        store: Any | None = None,
    ) -> dict[str, Any]:
        del state, action, store
        return {
            "process_valid": self.process_valid,
            "reason": None if self.process_valid else "independent_rejection",
            "error_code": None if self.process_valid else "invalid_action",
            "error_detail": None if self.process_valid else "independent_rejection",
            "valid_next_actions": [],
        }


class _InconsistentValidHandoffProcessOracle:
    def check(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        *,
        store: Any | None = None,
    ) -> dict[str, Any]:
        del state, action, store
        return {
            "process_valid": True,
            "reason": "inconsistent_success",
            "error_code": None,
            "error_detail": None,
            "valid_next_actions": [
                {"tool": RUN_WLS, "arguments": {"state_id": "active"}}
            ],
        }


class _HistorylessAuditedHandoffEnv(_AuditedHandoffEnv):
    """Match production: current_state omits the recorded transition history."""

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        self.process_oracle = _StaticHandoffProcessOracle(process_valid=True)
        return super().reset(scenario)

    def current_state(self) -> dict[str, Any]:
        state = super().current_state()
        state.pop("history_window", None)
        return state


class _IndependentlyRejectedAuditedHandoffEnv(_AuditedHandoffEnv):
    """Expose a forged-good history label plus an independent rejection."""

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        self.process_oracle = _StaticHandoffProcessOracle(process_valid=False)
        return super().reset(scenario)


class _InconsistentlyCertifiedAuditedHandoffEnv(_AuditedHandoffEnv):
    """Expose a forged-good history label plus an inconsistent success label."""

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        self.process_oracle = _InconsistentValidHandoffProcessOracle()
        return super().reset(scenario)


class _ReleaseScriptEnv(_ScriptEnv):
    """Test-only factory whose executable fixture is code-pinned, not suite data."""

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(dict(scenario))
        fixture = _resolved_scenario()
        payload["initial_physical_state"] = copy.deepcopy(
            fixture["initial_physical_state"]
        )
        payload["script"] = copy.deepcopy(fixture["script"])
        return super().reset(payload)


class _PartialSetupEnv(_ScriptEnv):
    def step(self, action: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        next_state, output = super().step(action)
        if action.get("tool") == CORRECT_MEASUREMENTS:
            output["state_mutated"] = True
        return next_state, output


def _resolved_scenario() -> dict[str, Any]:
    dirty_measurements = [0.0] * 8
    dirty_measurements[7] = 9.0
    clean_measurements = [0.0] * 8
    clean_measurements[7] = 1.0
    clean_case = {"baseMVA": 100.0, "branch": []}
    return {
        "scenario_id": "resolved-root",
        "root_scenario_id": "resolved-root",
        "physical_root_fingerprint": "fp-resolved",
        "scenario_family": "measurement+harmonic",
        "error_cardinality": 2,
        "case_id": "case14",
        "case": copy.deepcopy(clean_case),
        "clean_case": copy.deepcopy(clean_case),
        "measurements": dirty_measurements,
        "clean_measurements": clean_measurements,
        "initial_physical_state": {
            "case": copy.deepcopy(clean_case),
            "measurements": copy.deepcopy(clean_measurements),
        },
        "split": "validation",
        "source_tier": "real",
        "true_measurement_errors": [{"index": 7}],
        "hidden_truth": {
            "true_harmonic_errors": [{"bus_1based": 4}],
        },
        "final_remaining": 0,
        "script": [
            {"phase": "malformed", "status": "failure", "remaining": 1},
            {"phase": "wls", "remaining": 1},
            {
                "phase": "partial_commit",
                "candidate_state_id": "candidate-1",
                "disposition": "ACCEPT_PARTIAL",
                "remaining": 1,
                "accepted_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {"state_id": "active", "suspect_group": [7]},
                },
            },
            {
                "phase": "hse",
                "remaining": 0,
                "explanation": {
                    "family": "harmonic",
                    "detail": {"bus_1based": 4},
                },
            },
            {"phase": "finalize", "remaining": 0, "terminal_outcome": "resolved"},
        ],
    }


def _partitioned_resolved_scenario() -> dict[str, Any]:
    source = _resolved_scenario()
    return {
        "scenario_schema_version": 1,
        "execution": {
            key: copy.deepcopy(source[key])
            for key in (
                "scenario_id",
                "case",
                "measurements",
                "initial_physical_state",
                "script",
            )
        },
        "audit": {
            "evaluation_intervention": {
                "intervention_schema_version": 1,
                "kind": "none",
            },
            "truth": {
                "clean_case": copy.deepcopy(source["clean_case"]),
                "clean_measurements": copy.deepcopy(source["clean_measurements"]),
                "true_measurement_errors": copy.deepcopy(
                    source["true_measurement_errors"]
                ),
                "hidden_truth": copy.deepcopy(source["hidden_truth"]),
                "truth_complete": True,
            }
        },
        "grouping": {
            key: copy.deepcopy(source[key])
            for key in (
                "root_scenario_id",
                "physical_root_fingerprint",
                "scenario_family",
                "error_cardinality",
                "case_id",
                "split",
                "source_tier",
            )
        },
    }


def _release_partitioned_resolved_scenario() -> dict[str, Any]:
    scenario = _partitioned_resolved_scenario()
    scenario["execution"].pop("initial_physical_state")
    scenario["execution"].pop("script")
    return scenario


def _pre_policy_failure_scenario(*, malformed: bool) -> dict[str, Any]:
    scenario = _partitioned_resolved_scenario()
    scenario["execution"]["script"] = copy.deepcopy(_resolved_scenario()["script"][1:])
    scenario["audit"]["evaluation_intervention"] = {
        "intervention_schema_version": 1,
        "kind": "pre_policy_failure",
        "failure_mode": "malformed" if malformed else "well_formed",
        "error_code": (
            "injected_invalid_action"
            if malformed
            else "injected_transient_tool_failure"
        ),
    }
    return scenario


def _partial_retention_scenario() -> dict[str, Any]:
    scenario = _partitioned_resolved_scenario()
    correction = {
        "tool": CORRECT_MEASUREMENTS,
        "arguments": {
            "state_id": "$active",
            "measurement_updates": {"7": 1.0},
        },
    }
    scenario["audit"]["evaluation_intervention"] = {
        "intervention_schema_version": 1,
        "kind": "committed_partial_correction",
        "setup_actions": [
            {
                "tool": "get_measurement_context",
                "arguments": {"state_id": "$active"},
            },
            correction,
            {"tool": RUN_WLS, "arguments": {"state_id": "$candidate"}},
            {
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": "$candidate"},
            },
        ],
        "retention_required": True,
    }
    scenario["execution"]["script"] = [
        {"phase": "unused-context", "remaining": 2},
        {"phase": "unused-correction", "remaining": 2},
        {
            "phase": "unused-verification",
            "candidate_state_id": "candidate-1",
            "remaining": 1,
        },
        {
            "phase": "unused-commit",
            "candidate_state_id": "candidate-1",
            "disposition": "ACCEPT_PARTIAL",
            "remaining": 1,
            "accepted_action": {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": "active",
                    "measurement_updates": {7: 1.0},
                },
            },
        },
        {"phase": "finalize", "remaining": 0, "terminal_outcome": "resolved"},
    ]
    return scenario


def _escalation_scenario() -> dict[str, Any]:
    return {
        "scenario_id": "escalated-root",
        "physical_root_fingerprint": "fp-escalated",
        "scenario_family": "measurement+topology",
        "error_cardinality": 2,
        "case_id": "case14",
        "split": "test",
        "source_tier": "synthetic_physics",
        "hidden_truth": {
            "true_measurement_errors": [{"index": 3}],
            "true_topology_errors": [{"line_index1": 2}],
        },
        "final_remaining": 2,
        "script": [
            {"phase": "wls", "remaining": 2},
            {"phase": "wls", "remaining": 2},
            {
                "phase": "escalate",
                "remaining": 2,
                "terminal_outcome": "operator_escalation",
            },
        ],
    }


def _audited_handoff_scenario() -> dict[str, Any]:
    scenario = _resolved_scenario()
    scenario.update(
        {
            "scenario_id": "audited-handoff-root",
            "root_scenario_id": "audited-handoff-root",
            "physical_root_fingerprint": "fp-audited-handoff",
            "scenario_family": "measurement",
            "error_cardinality": 1,
            "hidden_truth": {},
            "final_remaining": 0,
            "script": [
                {
                    "phase": "partial_commit",
                    "candidate_state_id": "active",
                    "disposition": "ACCEPT_FINAL",
                    "remaining": 0,
                    "accepted_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "active",
                            "suspect_group": [7],
                        },
                    },
                },
                {
                    "phase": "escalate",
                    "remaining": 0,
                    "terminal_outcome": "operator_escalation",
                },
            ],
        }
    )
    return scenario


class ClosedLoopEvaluatorTests(unittest.TestCase):
    def test_policy_exception_is_recorded_as_a_schema_valid_invalid_action(
        self,
    ) -> None:
        class RaisingPolicy:
            def act(self, _observation: Mapping[str, Any]) -> Any:
                raise ValueError("unsupported canonical release argument")

        scenario = _partitioned_resolved_scenario()
        scenario["execution"]["script"] = [
            {
                "phase": "unused",
                "status": "failure",
                "error_code": "invalid_action",
                "remaining": 1,
            }
        ]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=_ScriptEnv,
            policy_factory=RaisingPolicy,
            max_steps=1,
        )
        episode = result.suite_metrics["episodes"][0]
        action = episode["trace"][0]["action"]
        self.assertEqual(action["tool"], INVALID_ACTION)
        self.assertEqual(action["arguments"]["error_code"], "policy_exception")
        self.assertIn("ValueError", action["arguments"]["error_detail"])
        self.assertEqual(episode["invalid_action_count"], 1)

    def test_policy_hidden_failure_interventions_are_injected_before_policy(self) -> None:
        for malformed, suite, expected_tool in (
            (False, "forced_error_recovery", RUN_WLS),
            (True, "invalid_action_recovery", "__invalid_action__"),
        ):
            observations: list[dict[str, Any]] = []
            with self.subTest(suite=suite):
                result = evaluate_rollout_suites(
                    {suite: [_pre_policy_failure_scenario(malformed=malformed)]},
                    env_factory=_ScriptEnv,
                    policy_factory=lambda: _ScriptPolicy(observations),
                    max_steps=8,
                )
                episode = result.suite_metrics["episodes"][0]
                injected = episode["evaluation_intervention"]
                self.assertTrue(injected["applied"])
                self.assertEqual(injected["pre_policy_step_count"], 1)
                self.assertEqual(injected["injected_failure_count"], 1)
                self.assertEqual(
                    injected["injected_invalid_action_count"], int(malformed)
                )
                self.assertEqual(injected["recovered_failure_count"], 1)
                self.assertEqual(episode["invalid_action_count"], 0)
                self.assertEqual(episode["recovered_invalid_action_count"], 0)
                self.assertEqual(episode["policy_steps"], 4)
                self.assertEqual(episode["steps"], 5)
                first_history = observations[0]["history_window"][-1]
                self.assertEqual(first_history["action"]["tool"], expected_tool)
                self.assertEqual(
                    first_history["tool_output"]["execution_status"], "failure"
                )
                self.assertFalse(first_history["tool_output"]["state_mutated"])
                self.assertNotIn("evaluation_intervention", json.dumps(observations))

    def test_partial_retention_intervention_requires_a_real_committed_setup(self) -> None:
        observations: list[dict[str, Any]] = []
        result = evaluate_rollout_suites(
            {"partial_success_retention": [_partial_retention_scenario()]},
            env_factory=_PartialSetupEnv,
            policy_factory=lambda: _ScriptPolicy(observations),
            max_steps=4,
        )
        episode = result.suite_metrics["episodes"][0]
        evidence = episode["evaluation_intervention"]
        self.assertEqual(evidence["pre_policy_step_count"], 4)
        self.assertEqual(evidence["retention_opportunity_count"], 1)
        self.assertEqual(evidence["retained_opportunity_count"], 1)
        self.assertEqual(episode["partial_fix_count"], 1)
        self.assertEqual(episode["retained_partial_fix_count"], 1)
        self.assertEqual(episode["policy_steps"], 1)
        self.assertEqual(
            [row["advanced"] for row in episode["trace"][:4]],
            [False, True, False, True],
        )
        self.assertEqual(len(observations[0]["history_window"]), 4)
        self.assertEqual(
            observations[0]["history_window"][-1]["action"]["tool"], COMMIT_STATE
        )
        observable_payload = json.dumps(observations)
        self.assertNotIn("evaluation_intervention", observable_payload)
        self.assertNotIn("retention_required", observable_payload)

        mislabeled = _partial_retention_scenario()
        mislabeled["execution"]["script"][3]["disposition"] = "ACCEPT_FINAL"
        with self.assertRaisesRegex(ValueError, "requires ACCEPT_PARTIAL"):
            evaluate_rollout_suites(
                {"partial_success_retention": [mislabeled]},
                env_factory=_PartialSetupEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                max_steps=4,
            )

    def test_release_intervention_schema_is_suite_specific_and_fail_closed(self) -> None:
        wrong_kind = _release_partitioned_resolved_scenario()
        wrong_kind["audit"]["evaluation_intervention"] = {
            "intervention_schema_version": 1,
            "kind": "none",
        }
        with self.assertRaisesRegex(ValueError, "requires intervention kind"):
            evaluate_rollout_suites(
                {"efficiency": [wrong_kind]},
                env_factory=_ReleaseScriptEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                require_release_environment=True,
            )

    def test_repeated_nonadvancing_diagnostic_is_recorded_but_not_reexecuted(
        self,
    ) -> None:
        executed_tools: list[str] = []

        class CountingEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_tools.append(str(action.get("tool") or ""))
                return super().step(action)

        scenario = _resolved_scenario()
        scenario["script"] = [
            {"phase": "hse", "remaining": 1},
            {"phase": "hse", "remaining": 1},
            {"phase": "finalize", "remaining": 0, "terminal_outcome": "resolved"},
        ]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=CountingEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(executed_tools, [RUN_HSE_FROM_PATH])
        self.assertEqual(episode["policy_steps"], 2)
        self.assertEqual(episode["specialized_tool_calls"], 2)
        self.assertEqual(episode["invalid_action_count"], 1)
        self.assertTrue(episode["loop_detected"])
        self.assertFalse(episode["terminal"])
        self.assertIsNone(episode["evaluator_error"])
        self.assertEqual(
            episode["trace"][1]["error_code"],
            "evaluation_repeated_nonadvancing_diagnostic",
        )
        self.assertEqual(episode["trace"][1]["execution_status"], "failure")
        self.assertFalse(episode["trace"][1]["advanced"])

    def test_repeated_deterministic_failures_are_bounded_and_quarantined(
        self,
    ) -> None:
        cases = (
            (
                "escalate",
                ASK_FOR_MORE_EVIDENCE,
                "operator_escalation_precondition_not_met",
                "rejected_operator_escalation",
                None,
            ),
            (
                "false_commit",
                COMMIT_STATE,
                "candidate_lifecycle_violation",
                "rejected_commit",
                None,
            ),
            (
                "raise",
                INVALID_ACTION,
                "policy_exception",
                "schema_invalid_action",
                ValueError,
            ),
            (
                "parameter",
                CORRECT_PARAMETERS,
                "correction_not_supported_by_current_context",
                "unsupported_correction",
                None,
            ),
            (
                "parameter",
                CORRECT_PARAMETERS,
                "post_correction_confirmation_required",
                "unsupported_correction",
                None,
            ),
            (
                "parameter",
                CORRECT_PARAMETERS,
                "correction_route_not_actionable",
                "correction_route_not_actionable",
                None,
            ),
            (
                "parameter",
                CORRECT_PARAMETERS,
                "parameter_scans_missing",
                "parameter_scans_missing",
                None,
            ),
        )
        for phase, expected_tool, error_code, failure_kind, raised in cases:
            with self.subTest(failure_kind=failure_kind):
                executed_tools: list[str] = []

                class FixedFailureEnv(_ScriptEnv):
                    def step(
                        self, action: Mapping[str, Any]
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
                        executed_tools.append(str(action.get("tool") or ""))
                        return self.current_state(), {
                            "execution_status": "failure",
                            "error_code": error_code,
                            "state_mutated": False,
                            "tool_metrics": {},
                        }

                if raised is not None:

                    class RaisingPolicy:
                        def act(self, _observation: Mapping[str, Any]) -> Any:
                            raise raised("same schema failure")

                    policy_factory = RaisingPolicy
                else:

                    def policy_factory() -> _ScriptPolicy:
                        return _ScriptPolicy([])

                scenario = _resolved_scenario()
                scenario["script"] = [
                    {
                        "phase": phase,
                        "remaining": 1,
                        "status": "failure",
                        "error_code": error_code,
                    }
                ]
                result = evaluate_rollout_suites(
                    {"standard_success": [scenario]},
                    env_factory=FixedFailureEnv,
                    policy_factory=policy_factory,
                    max_steps=8,
                )

                episode = result.suite_metrics["episodes"][0]
                quarantine = episode["control_quarantine"]
                self.assertEqual(executed_tools, [expected_tool])
                self.assertEqual(episode["policy_steps"], 2)
                self.assertEqual(episode["invalid_action_count"], 2)
                self.assertTrue(episode["loop_detected"])
                self.assertFalse(episode["terminal"])
                self.assertIsNone(episode["evaluator_error"])
                self.assertEqual(episode["trace"][0]["error_code"], error_code)
                self.assertEqual(
                    episode["trace"][1]["error_code"],
                    "evaluation_repeated_nonadvancing_failure",
                )
                self.assertTrue(quarantine["quarantined"])
                self.assertEqual(quarantine["failure_kind"], failure_kind)
                self.assertEqual(quarantine["trigger_error_code"], error_code)
                self.assertEqual(quarantine["action_tool"], expected_tool)
                self.assertEqual(quarantine["executed_failure_count"], 1)
                self.assertEqual(quarantine["attempted_failure_count"], 2)
                self.assertRegex(
                    quarantine["action_signature_sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertEqual(
                    episode["audit"]["control_quarantine"], quarantine
                )
                overall = result.suite_metrics["overall"]
                self.assertEqual(overall["control_quarantined_episodes"], 1)
                self.assertEqual(overall["control_quarantine_rate"], 1.0)
                self.assertEqual(
                    overall["control_quarantine_reason_counts"],
                    {failure_kind: 1},
                )
                self.assertEqual(
                    overall[
                        "repeated_nonadvancing_failure_breaker_episodes"
                    ],
                    1,
                )
                # The breaker is a policy-performance quarantine, not an
                # evaluator infrastructure error, and remains a promotion
                # NO-GO under the paired safety ordinal.
                self.assertEqual(_episode_safety_ordinal(episode), 0)

    def test_family_wide_parameter_failure_blocks_a_different_target(self) -> None:
        executed_actions: list[dict[str, Any]] = []

        class AlternatingParameterPolicy:
            def __init__(self) -> None:
                self.targets = [2, 3]
                self.cursor = 0

            def act(self, _observation: Mapping[str, Any]) -> dict[str, Any]:
                target = self.targets[min(self.cursor, len(self.targets) - 1)]
                self.cursor += 1
                return {
                    "tool": CORRECT_PARAMETERS,
                    "arguments": {
                        "state_id": "active",
                        "line_index1": target,
                    },
                }

        class MissingScansEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_actions.append(copy.deepcopy(dict(action)))
                return self.current_state(), {
                    "execution_status": "failure",
                    "error_code": "parameter_scans_missing",
                    "state_mutated": False,
                    "tool_metrics": {},
                }

        scenario = _resolved_scenario()
        scenario["script"] = [{"phase": "unused", "remaining": 1}]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=MissingScansEnv,
            policy_factory=AlternatingParameterPolicy,
            max_steps=4,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(
            [action["arguments"]["line_index1"] for action in executed_actions],
            [2],
        )
        self.assertEqual(episode["policy_steps"], 2)
        self.assertEqual(
            episode["trace"][1]["error_code"],
            "evaluation_repeated_nonadvancing_failure",
        )
        self.assertTrue(episode["loop_detected"])
        self.assertEqual(
            episode["control_quarantine"]["failure_kind"],
            "parameter_scans_missing",
        )

    def test_real_state_advance_resets_deterministic_failure_bound(self) -> None:
        executed_tools: list[str] = []
        repeated_invalid = {
            "tool": INVALID_ACTION,
            "arguments": {
                "error_code": "policy_exception",
                "error_detail": "same schema failure",
            },
        }

        class SequencePolicy:
            def __init__(self) -> None:
                self.actions = [
                    repeated_invalid,
                    {
                        "tool": RUN_WLS,
                        "arguments": {"state_id": "active"},
                    },
                    repeated_invalid,
                ]
                self.cursor = 0

            def act(self, _observation: Mapping[str, Any]) -> dict[str, Any]:
                action = self.actions[self.cursor]
                self.cursor += 1
                return copy.deepcopy(action)

        class AdvancingEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_tools.append(str(action.get("tool") or ""))
                if action.get("tool") == RUN_WLS:
                    return self.current_state(), {
                        "execution_status": "success",
                        "error_code": None,
                        "state_mutated": True,
                        "tool_metrics": {},
                    }
                return self.current_state(), {
                    "execution_status": "failure",
                    "error_code": "policy_exception",
                    "state_mutated": False,
                    "tool_metrics": {},
                }

        scenario = _resolved_scenario()
        scenario["script"] = [{"phase": "unused", "remaining": 1}]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=AdvancingEnv,
            policy_factory=SequencePolicy,
            max_steps=3,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(
            executed_tools,
            [INVALID_ACTION, RUN_WLS, INVALID_ACTION],
        )
        self.assertEqual(episode["policy_steps"], 3)
        self.assertFalse(episode["loop_detected"])
        self.assertFalse(episode["control_quarantine"]["quarantined"])
        self.assertEqual(
            result.suite_metrics["overall"]["control_quarantined_episodes"], 0
        )

    def test_nonadvancing_intervening_action_does_not_reset_failure_bound(
        self,
    ) -> None:
        executed_tools: list[str] = []

        class CountingEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_tools.append(str(action.get("tool") or ""))
                return super().step(action)

        scenario = _resolved_scenario()
        scenario["script"] = [
            {
                "phase": "escalate",
                "remaining": 1,
                "status": "failure",
                "error_code": "operator_escalation_precondition_not_met",
            },
            {
                "phase": "wls",
                "remaining": 1,
                "status": "success",
            },
            {
                "phase": "escalate",
                "remaining": 1,
                "status": "failure",
                "error_code": "operator_escalation_precondition_not_met",
            },
        ]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=CountingEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(
            executed_tools,
            [ASK_FOR_MORE_EVIDENCE, RUN_WLS],
        )
        self.assertEqual(episode["policy_steps"], 3)
        self.assertEqual(
            episode["trace"][2]["error_code"],
            "evaluation_repeated_nonadvancing_failure",
        )
        self.assertTrue(episode["control_quarantine"]["quarantined"])
        self.assertEqual(
            episode["control_quarantine"]["failure_kind"],
            "rejected_operator_escalation",
        )

    def test_unclassified_failure_is_not_short_circuited(self) -> None:
        executed_tools: list[str] = []

        class TransientFailureEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_tools.append(str(action.get("tool") or ""))
                return self.current_state(), {
                    "execution_status": "failure",
                    "error_code": "transient_provider_failure",
                    "state_mutated": False,
                    "tool_metrics": {},
                }

        scenario = _resolved_scenario()
        scenario["script"] = [{"phase": "escalate", "remaining": 1}]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=TransientFailureEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=3,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(executed_tools, [ASK_FOR_MORE_EVIDENCE] * 3)
        self.assertEqual(episode["policy_steps"], 3)
        self.assertTrue(episode["loop_detected"])
        self.assertFalse(episode["control_quarantine"]["quarantined"])

    def test_diagnostic_can_repeat_after_a_real_state_advance(self) -> None:
        executed_tools: list[str] = []

        class CountingEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_tools.append(str(action.get("tool") or ""))
                return super().step(action)

        scenario = _resolved_scenario()
        scenario["script"] = [
            {
                "phase": "hse",
                "remaining": 1,
                "explanation": {
                    "family": "harmonic",
                    "detail": {"bus_1based": 4},
                },
            },
            {"phase": "hse", "remaining": 1},
        ]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=CountingEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=2,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(
            executed_tools,
            [RUN_HSE_FROM_PATH, RUN_HSE_FROM_PATH],
        )
        self.assertFalse(episode["loop_detected"])
        self.assertEqual(
            [row["execution_status"] for row in episode["trace"]],
            ["success", "success"],
        )

    def test_efficiency_specialized_budget_blocks_first_excess_execution(
        self,
    ) -> None:
        executed_tools: list[str] = []

        class CountingEnv(_ScriptEnv):
            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                executed_tools.append(str(action.get("tool") or ""))
                return super().step(action)

        class SequencePolicy:
            def __init__(self) -> None:
                self.actions = [
                    {
                        "tool": RUN_HSE_FROM_PATH,
                        "arguments": {"state_id": "active"},
                    },
                    {
                        "tool": GET_HARMONIC_CONTEXT,
                        "arguments": {"state_id": "active"},
                    },
                ]
                self.cursor = 0

            def act(self, _observation: Mapping[str, Any]) -> dict[str, Any]:
                action = self.actions[self.cursor]
                self.cursor += 1
                return copy.deepcopy(action)

        scenario = _partitioned_resolved_scenario()
        scenario["audit"]["evaluation_intervention"] = {
            "intervention_schema_version": 1,
            "kind": "efficiency_budget",
            "limits": {
                "maximum_policy_steps": 4,
                "maximum_wls_calls": 4,
                "maximum_specialized_tool_calls": 1,
            },
        }
        scenario["execution"]["script"] = [
            {"phase": "unused-first", "remaining": 1},
            {"phase": "unused-second", "remaining": 1},
        ]
        result = evaluate_rollout_suites(
            {"efficiency": [scenario]},
            env_factory=CountingEnv,
            policy_factory=SequencePolicy,
            max_steps=4,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(executed_tools, [RUN_HSE_FROM_PATH])
        self.assertEqual(episode["policy_steps"], 2)
        self.assertEqual(episode["specialized_tool_calls"], 2)
        self.assertEqual(
            episode["specialized_tool_counts"],
            {GET_HARMONIC_CONTEXT: 1, RUN_HSE_FROM_PATH: 1},
        )
        self.assertEqual(
            episode["trace"][1]["error_code"],
            "evaluation_specialized_tool_budget_exhausted",
        )
        self.assertEqual(episode["trace"][1]["execution_status"], "failure")
        self.assertFalse(episode["trace"][1]["advanced"])
        self.assertFalse(episode["loop_detected"])
        self.assertIsNone(episode["evaluator_error"])
        evidence_failures, performance_failures = _intervention_failures(
            episode,
            scenario["audit"]["evaluation_intervention"],
        )
        self.assertEqual(evidence_failures, [])
        self.assertEqual(
            performance_failures,
            [
                "episode efficiency limit maximum_specialized_tool_calls "
                "failed: observed 2 > allowed 1"
            ],
        )

    def test_progress_callback_reports_policy_safe_step_timing(self) -> None:
        events: list[dict[str, Any]] = []

        class TelemetryPolicy(_ScriptPolicy):
            @property
            def last_action_metrics(self) -> dict[str, Any]:
                return {
                    "prompt_tokens": 120,
                    "generated_tokens": 7,
                    "generation_seconds": 0.25,
                    "hit_max_new_tokens": False,
                    "last_token_id": 50,
                    "private_model_state": "must-not-leak",
                }

        scenario = _resolved_scenario()
        scenario["script"] = [
            {
                "phase": "finalize",
                "remaining": 0,
                "terminal_outcome": "resolved",
            }
        ]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=_ScriptEnv,
            policy_factory=lambda: TelemetryPolicy([]),
            progress_callback=lambda row: events.append(copy.deepcopy(dict(row))),
        )

        self.assertTrue(result.suite_metrics["episodes"][0]["terminal"])
        self.assertEqual(
            [row["event"] for row in events],
            [
                "episode_start",
                "policy_action",
                "step_complete",
                "episode_complete",
            ],
        )
        policy_event = events[1]
        self.assertEqual(
            policy_event["action"],
            {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
        )
        self.assertGreaterEqual(policy_event["policy_seconds"], 0.0)
        self.assertEqual(
            policy_event["policy_metrics"],
            {
                "generated_tokens": 7,
                "generation_seconds": 0.25,
                "hit_max_new_tokens": False,
                "last_token_id": 50,
                "prompt_tokens": 120,
            },
        )
        self.assertGreaterEqual(events[2]["tool_seconds"], 0.0)
        self.assertNotIn("hidden_truth", json.dumps(events))
        self.assertNotIn("private_model_state", json.dumps(events))

    def test_hostile_policy_metrics_cannot_interrupt_evaluation_progress(self) -> None:
        events: list[dict[str, Any]] = []

        class NonCopyableMetric:
            def __deepcopy__(self, _memo: dict[int, Any]) -> Any:
                raise RuntimeError("must not be copied")

        class TelemetryPolicy(_ScriptPolicy):
            @property
            def last_action_metrics(self) -> dict[str, Any]:
                return {
                    "prompt_tokens": NonCopyableMetric(),
                    "private_model_state": NonCopyableMetric(),
                }

        scenario = _resolved_scenario()
        scenario["script"] = [
            {
                "phase": "finalize",
                "remaining": 0,
                "terminal_outcome": "resolved",
            }
        ]
        result = evaluate_rollout_suites(
            {"standard_success": [scenario]},
            env_factory=_ScriptEnv,
            policy_factory=lambda: TelemetryPolicy([]),
            progress_callback=lambda row: events.append(dict(row)),
        )

        self.assertTrue(result.suite_metrics["episodes"][0]["terminal"])
        self.assertEqual(
            [row["event"] for row in events],
            [
                "episode_start",
                "policy_action",
                "step_complete",
                "episode_complete",
            ],
        )
        self.assertEqual(events[1]["policy_metrics"], {})

    def test_executes_suites_and_reports_recovery_safety_and_groups(self) -> None:
        observed: list[dict[str, Any]] = []

        def cost_resolver(context: Mapping[str, Any]) -> dict[str, float]:
            # Every chosen tool costs one more than the labeled best action.
            return {"chosen_cost": 2.0, "best_cost": 1.0}

        result = evaluate_rollout_suites(
            {
                "forced_error_recovery": [_resolved_scenario()],
                "efficiency": [_escalation_scenario()],
            },
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy(observed),
            max_steps=8,
            seed=19,
            tool_cost_resolver=cost_resolver,
        )

        overall = result.suite_metrics["overall"]
        self.assertEqual(
            result.suite_metrics["schema_version"],
            STUDY_EVALUATION_SCHEMA_VERSION,
        )
        self.assertEqual(overall["episodes"], 2)
        self.assertEqual(overall["terminal_rate"], 1.0)
        self.assertEqual(overall["resolution_rate"], 0.5)
        self.assertEqual(overall["operator_escalation_rate"], 0.5)
        self.assertEqual(overall["audited_post_correction_handoff_rate"], 0.0)
        self.assertEqual(overall["audited_completion_rate"], 0.5)
        self.assertEqual(overall["unqualified_operator_escalation_rate"], 0.5)
        self.assertEqual(overall["final_physical_success_rate"], 0.5)
        self.assertEqual(overall["invalid_action_count"], 1)
        self.assertEqual(overall["invalid_action_recovery_rate"], 1.0)
        self.assertEqual(overall["partial_fix_retention_rate"], 1.0)
        self.assertEqual(overall["loop_rate"], 0.5)
        self.assertEqual(overall["wls_calls"], 3)
        self.assertEqual(overall["specialized_tool_calls"], 1)
        self.assertEqual(overall["mean_tool_regret"], 1.0)
        self.assertEqual(overall["tool_regret_samples"], 8)
        self.assertEqual(
            set(result.suite_metrics["by_family"]),
            {"measurement+harmonic", "measurement+topology"},
        )
        self.assertEqual(
            result.suite_metrics["by_cardinality"]["2"]["episodes"], 2
        )
        self.assertEqual(set(result.suite_metrics["by_split"]), {"test", "validation"})
        self.assertEqual(
            set(result.suite_metrics["by_physical_root"]),
            {"fp-escalated", "fp-resolved"},
        )
        for episode in result.suite_metrics["episodes"]:
            terminal_rows = [
                index
                for index, row in enumerate(episode["trace"])
                if row["terminal_outcome"] is not None
            ]
            self.assertEqual(terminal_rows, [len(episode["trace"]) - 1])
            self.assertEqual(
                episode["trace"][-1]["terminal_outcome"],
                episode["terminal_outcome"],
            )
        self.assertTrue(observed)
        for observation in observed:
            self.assertNotIn("hidden_truth", observation)
            self.assertFalse(any(key.startswith("true_") for key in observation))

    def test_safe_operator_escalation_reports_known_healthy_preservation(
        self,
    ) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "safe-escalation"
        scenario["physical_root_fingerprint"] = "fp-safe-escalation"
        scenario["script"] = [
            {
                "phase": "escalate",
                "remaining": 2,
                "terminal_outcome": "operator_escalation",
            }
        ]

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(episode["terminal_outcome"], "operator_escalation")
        self.assertFalse(episode["physical_correctness_known"])
        self.assertFalse(episode["final_physical_correct"])
        self.assertFalse(episode["final_physical_success"])
        self.assertTrue(episode["healthy_preservation_known"])
        self.assertTrue(episode["healthy_components_preserved"])
        assessment = episode["audit"]["post_correction_handoff_assessment"]
        self.assertEqual(assessment["status"], "not_applicable")
        self.assertFalse(assessment["eligible"])
        self.assertEqual(
            result.suite_metrics["overall"][
                "unqualified_operator_escalation_episodes"
            ],
            1,
        )
        self.assertEqual(
            result.suite_metrics["overall"]["audited_completion_episodes"],
            0,
        )
        checks = episode["audit"]["strict_release_audit"]["checks"]
        self.assertEqual(checks["healthy_measurements_preserved"]["status"], "passed")
        self.assertEqual(
            checks["healthy_case_components_preserved"]["status"], "passed"
        )

    def test_only_audited_post_correction_handoff_counts_as_completion(
        self,
    ) -> None:
        result = evaluate_rollout_suites(
            [_audited_handoff_scenario()],
            env_factory=_AuditedHandoffEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        episode = result.suite_metrics["episodes"][0]
        assessment = episode["audit"]["post_correction_handoff_assessment"]
        self.assertEqual(episode["terminal_outcome"], "operator_escalation")
        self.assertEqual(assessment["status"], "passed")
        self.assertTrue(assessment["eligible"])
        self.assertEqual(
            assessment["actual_terminal_outcome"], "operator_escalation"
        )
        self.assertTrue(assessment["runtime_contract"]["passed"])
        self.assertEqual(
            episode["trace"][-1]["runtime_state_hash"],
            assessment["runtime_contract"]["active_state_hash"],
        )
        self.assertFalse(
            assessment["counterfactual_completion_audit"]["quarantined"]
        )
        overall = result.suite_metrics["overall"]
        self.assertEqual(overall["resolution_rate"], 0.0)
        self.assertEqual(overall["audited_post_correction_handoff_rate"], 1.0)
        self.assertEqual(overall["audited_completion_rate"], 1.0)
        self.assertEqual(overall["unqualified_operator_escalation_rate"], 0.0)

    def test_historyless_final_state_binds_independently_checked_handoff(
        self,
    ) -> None:
        result = evaluate_rollout_suites(
            [_audited_handoff_scenario()],
            env_factory=_HistorylessAuditedHandoffEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        episode = result.suite_metrics["episodes"][0]
        assessment = episode["audit"]["post_correction_handoff_assessment"]
        self.assertEqual(assessment["status"], "passed", assessment["reasons"])
        self.assertIs(assessment["eligible"], True)
        self.assertIs(assessment["runtime_contract"]["passed"], True)
        self.assertEqual(assessment["runtime_contract"]["failures"], [])
        self.assertEqual(
            result.suite_metrics["overall"]["audited_completion_episodes"],
            1,
        )

    def test_independent_process_rejection_overrides_forged_good_label(
        self,
    ) -> None:
        result = evaluate_rollout_suites(
            [_audited_handoff_scenario()],
            env_factory=_IndependentlyRejectedAuditedHandoffEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        episode = result.suite_metrics["episodes"][0]
        assessment = episode["audit"]["post_correction_handoff_assessment"]
        self.assertEqual(assessment["status"], "failed")
        self.assertIn("handoff_transition_label_invalid", assessment["reasons"])
        self.assertIsNone(assessment["counterfactual_completion_audit"])
        self.assertEqual(
            result.suite_metrics["overall"][
                "audited_post_correction_handoff_episodes"
            ],
            0,
        )

    def test_inconsistent_valid_process_label_fails_closed(self) -> None:
        result = evaluate_rollout_suites(
            [_audited_handoff_scenario()],
            env_factory=_InconsistentlyCertifiedAuditedHandoffEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        episode = result.suite_metrics["episodes"][0]
        assessment = episode["audit"]["post_correction_handoff_assessment"]
        self.assertEqual(assessment["status"], "failed")
        self.assertIn("handoff_transition_label_invalid", assessment["reasons"])
        self.assertIsNone(assessment["counterfactual_completion_audit"])
        self.assertEqual(
            result.suite_metrics["overall"][
                "audited_post_correction_handoff_episodes"
            ],
            0,
        )

    def test_generic_escalation_does_not_recover_injected_failure(self) -> None:
        scenario = _pre_policy_failure_scenario(malformed=False)
        scenario["execution"]["script"] = [
            {"phase": "wls", "remaining": 1},
            {
                "phase": "escalate",
                "remaining": 1,
                "terminal_outcome": "operator_escalation",
            },
        ]

        result = evaluate_rollout_suites(
            {"forced_error_recovery": [scenario]},
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertEqual(
            episode["evaluation_intervention"]["injected_failure_count"], 1
        )
        self.assertEqual(
            episode["evaluation_intervention"]["recovered_failure_count"], 0
        )
        self.assertEqual(
            result.suite_metrics["overall"]["injected_failure_recovery_rate"],
            0.0,
        )
        self.assertEqual(
            result.suite_metrics["overall"][
                "unqualified_operator_escalation_episodes"
            ],
            1,
        )

    def test_uses_offline_per_step_cost_labels_without_a_resolver(self) -> None:
        scenario = {
            "scenario_id": "cost-labeled",
            "scenario_family": "no_error",
            "hidden_truth": {},
            "final_remaining": 0,
            "evaluation_labels": [
                {"action_costs": {FINALIZE_DIAGNOSIS: 3.0, RUN_WLS: 1.0}}
            ],
            "script": [
                {
                    "phase": "finalize",
                    "remaining": 0,
                    "terminal_outcome": "resolved",
                }
            ],
        }
        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )
        overall = result.suite_metrics["overall"]
        self.assertEqual(overall["tool_regret_samples"], 1)
        self.assertEqual(overall["mean_tool_regret"], 2.0)

    def test_partitioned_audit_cost_labels_are_scored_but_never_executed(
        self,
    ) -> None:
        class RecordingEnv(_ScriptEnv):
            reset_payloads: list[dict[str, Any]] = []

            def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
                self.__class__.reset_payloads.append(copy.deepcopy(dict(scenario)))
                return super().reset(scenario)

        execution = {
            "scenario_id": "partitioned-cost-labeled",
            "case": {"baseMVA": 100.0, "branch": []},
            "measurements": [],
            "initial_physical_state": {
                "case": {"baseMVA": 100.0, "branch": []},
                "measurements": [],
            },
            "script": [
                {
                    "phase": "finalize",
                    "remaining": 0,
                    "terminal_outcome": "resolved",
                }
            ],
        }
        scenario = {
            "scenario_schema_version": 1,
            "execution": execution,
            "audit": {
                "evaluation_labels": [
                    {"action_costs": {FINALIZE_DIAGNOSIS: 3.0, RUN_WLS: 1.0}}
                ]
            },
            "grouping": {
                "physical_root_fingerprint": "root-partitioned-cost",
                "scenario_family": "no_error",
                "error_cardinality": 0,
                "case_id": "case14",
                "split": "test",
                "source_tier": "test",
            },
        }

        RecordingEnv.reset_payloads.clear()
        result = evaluate_rollout_suites(
            [scenario],
            env_factory=RecordingEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        self.assertEqual(RecordingEnv.reset_payloads, [execution])
        overall = result.suite_metrics["overall"]
        self.assertEqual(overall["tool_regret_samples"], 1)
        self.assertEqual(overall["mean_tool_regret"], 2.0)

    def test_partitioned_direct_normalized_cost_label_is_scored(self) -> None:
        scenario = {
            "scenario_schema_version": 1,
            "execution": {
                "scenario_id": "partitioned-direct-cost",
                "case": {"baseMVA": 100.0, "branch": []},
                "measurements": [],
                "initial_physical_state": {
                    "case": {"baseMVA": 100.0, "branch": []},
                    "measurements": [],
                },
                "script": [
                    {
                        "phase": "finalize",
                        "remaining": 0,
                        "terminal_outcome": "resolved",
                    }
                ],
            },
            "audit": {
                "actionCosts": {FINALIZE_DIAGNOSIS: 3.0, RUN_WLS: 1.0}
            },
            "grouping": {
                "physical_root_fingerprint": "root-partitioned-direct-cost",
                "scenario_family": "no_error",
                "error_cardinality": 0,
                "case_id": "case14",
                "split": "test",
                "source_tier": "test",
            },
        }

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )

        overall = result.suite_metrics["overall"]
        self.assertEqual(overall["tool_regret_samples"], 1)
        self.assertEqual(overall["mean_tool_regret"], 2.0)

    def test_scores_false_commit_rollback_and_finalization_intent(self) -> None:
        scenario = {
            "scenario_id": "unsafe",
            "scenario_family": "measurement",
            "hidden_truth": {"true_measurement_errors": [{"index": 2}]},
            "final_remaining": 1,
            "script": [
                {
                    "phase": "false_commit",
                    "candidate_state_id": "bad-candidate",
                    "disposition": "REJECT",
                    "status": "failure",
                    "remaining": 1,
                },
                {
                    "phase": "false_rollback",
                    "candidate_state_id": "good-candidate",
                    "disposition": "ACCEPT_FINAL",
                    "status": "failure",
                    "remaining": 1,
                },
                {
                    "phase": "finalize",
                    "remaining": 1,
                    "terminal_outcome": "resolved",
                },
            ],
        }
        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=4,
        )
        summary = result.suite_metrics["overall"]
        self.assertEqual(summary["false_commit_count"], 1)
        self.assertEqual(summary["false_rollback_count"], 1)
        self.assertEqual(summary["false_finalization_count"], 1)
        self.assertEqual(summary["final_physical_success_rate"], 0.0)

    def test_release_environment_contract_fails_closed(self) -> None:
        class DevelopmentOracle:
            mode = "auto"

        class DevelopmentEnv(_ScriptEnv):
            production_dataset_mode = False
            candidate_quality_oracle = DevelopmentOracle()

        development = evaluate_rollout_suites(
            [_resolved_scenario()],
            env_factory=DevelopmentEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )
        validation = development.suite_metrics["configuration"][
            "release_environment_validation"
        ]
        self.assertFalse(validation["passed"])
        self.assertEqual(validation["episodes_checked"], 1)
        self.assertEqual(
            validation["required"],
            {
                "production_dataset_mode": True,
                "candidate_quality_oracle_mode": "deployment",
            },
        )
        self.assertEqual(
            validation["failures"],
            [
                "candidate_quality_oracle.mode is not 'deployment'",
                "production_dataset_mode is not exactly true",
            ],
        )

        with self.assertRaisesRegex(
            ValueError, "release environment validation failed"
        ):
            evaluate_rollout_suites(
                [_release_partitioned_resolved_scenario()],
                env_factory=DevelopmentEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                max_steps=8,
                require_release_environment=True,
            )

        for callback_kwargs in (
            {"tool_cost_resolver": lambda _context: None},
            {"physical_audit_fn": lambda _context: True},
        ):
            with self.subTest(callback=next(iter(callback_kwargs))):
                with self.assertRaisesRegex(ValueError, "release evaluation forbids"):
                    evaluate_rollout_suites(
                        [_resolved_scenario()],
                        env_factory=_ScriptEnv,
                        policy_factory=lambda: _ScriptPolicy([]),
                        require_release_environment=True,
                        **callback_kwargs,
                    )

        with self.assertRaisesRegex(ValueError, "release evaluation forbids"):
            evaluate_rollout_suites(
                [_resolved_scenario()],
                env_factory=_ScriptEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                expected_policy_identity={
                    "explicit_policy_identity": "test-policy-v1",
                    "model_id": None,
                    "model_revision": None,
                },
                tool_cost_resolver=lambda _context: None,
            )

        with self.assertRaisesRegex(ValueError, "unsupported override fields"):
            evaluate_rollout_suites(
                [_resolved_scenario()],
                env_factory=_ScriptEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                physical_audit_fn=lambda _context: {
                    "strict_release_audit": {"quarantined": False}
                },
            )

        callback_result = evaluate_rollout_suites(
            [_resolved_scenario()],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            tool_cost_resolver=lambda _context: None,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            callback_artifact = write_evaluation_artifact(
                callback_result,
                Path(temporary_directory) / "callback-development.json",
            )
        self.assertFalse(callback_artifact["release_eligible"])
        self.assertTrue(
            any(
                "custom physical-audit or tool-cost callbacks" in failure
                for failure in callback_artifact["release_failures"]
            )
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = write_evaluation_artifact(
                development, Path(temporary_directory) / "development.json"
            )
        self.assertFalse(artifact["release_eligible"])
        self.assertTrue(
            any(
                "executed environment contract is not release-safe" in failure
                for failure in artifact["release_failures"]
            )
        )

    def test_strip_offline_truth_removes_extended_normalized_audit_aliases(
        self,
    ) -> None:
        scenario = {
            "scenario_id": "extended-aliases",
            "case": "case14",
            "measurements": [1.0, 2.0],
            "initial_physical_state": {
                "case": "case14",
                "measurements": [1.0, 2.0],
            },
            "finalPhysicalState": {"measurements": [0.0, 0.0]},
            "Ground-Truth": {"fault": "measurement"},
            "truth": {"fault": "topology"},
            "finalState": {"case": "clean-answer"},
            "labels": {"candidate_disposition": "ACCEPT_FINAL"},
            "preferredAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "teacherAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "expert_action": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "targetAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "goldAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "oracleAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "recommendedAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "validAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "validNextAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "teacherActions": [{"tool": FINALIZE_DIAGNOSIS, "arguments": {}}],
            "expertActions": [{"tool": FINALIZE_DIAGNOSIS, "arguments": {}}],
            "oracleActions": [{"tool": FINALIZE_DIAGNOSIS, "arguments": {}}],
            "recommendedActions": [
                {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
            ],
            "correctAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "optimalAction": {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            "actionCostsByTool": {FINALIZE_DIAGNOSIS: 0.0},
            "qValues": {FINALIZE_DIAGNOSIS: 0.0},
            "referenceSolution": {"case": "clean-answer"},
            "family": "measurement",
            "valid-next-actions": [
                {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
            ],
            "ACTION COSTS": {FINALIZE_DIAGNOSIS: 0.0, RUN_WLS: 1.0},
            "rankingCostLabels": {FINALIZE_DIAGNOSIS: 0.0},
            "metadata": {
                "observable_runtime": {"channel": "pmu"},
                "groundTruth": {"fault": "parameter"},
                "preferred-action": {"tool": RUN_WLS, "arguments": {}},
                "validNextActions": [{"tool": RUN_WLS, "arguments": {}}],
                "action-costs": {RUN_WLS: 0.0},
                "nested": [
                    {
                        "finalPhysicalState": {"case": "clean-answer"},
                        "runtime_value": 7,
                    }
                ],
            },
        }
        original = copy.deepcopy(scenario)

        execution = strip_offline_truth(scenario)

        self.assertEqual(scenario, original)
        self.assertEqual(
            execution,
            {
                "scenario_id": "extended-aliases",
                "case": "case14",
                "measurements": [1.0, 2.0],
                "initial_physical_state": {
                    "case": "case14",
                    "measurements": [1.0, 2.0],
                },
                "metadata": {
                    "observable_runtime": {"channel": "pmu"},
                    "nested": [{"runtime_value": 7}],
                },
            },
        )

    def test_partitioned_scenario_passes_only_explicit_execution_to_reset(
        self,
    ) -> None:
        class RecordingEnv(_ScriptEnv):
            reset_payloads: list[dict[str, Any]] = []

            def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
                self.__class__.reset_payloads.append(copy.deepcopy(dict(scenario)))
                return super().reset(scenario)

        source = _resolved_scenario()
        execution = {
            key: copy.deepcopy(source[key])
            for key in (
                "scenario_id",
                "case",
                "measurements",
                "initial_physical_state",
                "script",
            )
        }
        audit = {
            "truth": {
                "clean_case": copy.deepcopy(source["clean_case"]),
                "clean_measurements": copy.deepcopy(source["clean_measurements"]),
                "true_measurement_errors": copy.deepcopy(
                    source["true_measurement_errors"]
                ),
                "hidden_truth": copy.deepcopy(source["hidden_truth"]),
                "truth_complete": True,
            },
            "final_physical_state": {"audit_sentinel": "never-reset"},
            "labels": {
                "preferred_action": {
                    "tool": FINALIZE_DIAGNOSIS,
                    "arguments": {},
                }
            },
        }
        grouping = {
            key: copy.deepcopy(source[key])
            for key in (
                "root_scenario_id",
                "physical_root_fingerprint",
                "scenario_family",
                "error_cardinality",
                "case_id",
                "split",
                "source_tier",
            )
        }
        partitioned = {
            "scenario_schema_version": 1,
            "execution": execution,
            "audit": audit,
            "grouping": grouping,
        }
        resolver_scenarios: list[dict[str, Any]] = []

        def cost_resolver(context: Mapping[str, Any]) -> None:
            self.assertNotIn("environment", context)
            resolver_scenarios.append(copy.deepcopy(dict(context["scenario"])))
            return None

        RecordingEnv.reset_payloads.clear()
        result = evaluate_rollout_suites(
            [partitioned],
            env_factory=RecordingEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            tool_cost_resolver=cost_resolver,
            max_steps=8,
        )

        self.assertEqual(RecordingEnv.reset_payloads, [execution])
        self.assertTrue(resolver_scenarios)
        self.assertEqual(
            resolver_scenarios[0]["audit"]["final_physical_state"],
            {"audit_sentinel": "never-reset"},
        )
        self.assertTrue(result.suite_metrics["episodes"][0]["final_physical_success"])
        self.assertEqual(
            set(result.suite_metrics["by_family"]), {"measurement+harmonic"}
        )

    def test_partitioned_scenario_rejects_normalized_audit_aliases_in_execution(
        self,
    ) -> None:
        scenario = {
            "scenario_schema_version": 1,
            "execution": {
                "scenario_id": "leaky-explicit-execution",
                "case": "case14",
                "measurements": [],
                "metadata": {
                    "finalPhysicalState": {"case": "answer"},
                    "groundTruth": {"fault": "measurement"},
                    "labels": {"state_class": "terminal_resolved"},
                    "preferredAction": {
                        "tool": FINALIZE_DIAGNOSIS,
                        "arguments": {},
                    },
                    "validNextActions": [
                        {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
                    ],
                    "action-costs": {FINALIZE_DIAGNOSIS: 0.0},
                },
            },
            "audit": {},
            "grouping": {},
        }

        with self.assertRaisesRegex(
            ValueError, r"\$\.metadata\.finalPhysicalState"
        ):
            strip_offline_truth(scenario)

        for key, path in (
            ("z_clean", r"\$\.metadata\.hif_scan_window\.scans\[0\]\.z_clean"),
            ("detected_top1", r"\$\.metadata\.nlm_diagnostic\.detected_top1"),
            ("initial_states", r"\$\.metadata\.parameter_scans\.initial_states"),
        ):
            candidate = _partitioned_resolved_scenario()
            if key == "z_clean":
                candidate["execution"]["metadata"] = {
                    "hif_scan_window": {"scans": [{key: [1.0]}]}
                }
            elif key == "initial_states":
                candidate["execution"]["metadata"] = {
                    "parameter_scans": {key: [[1.0]]}
                }
            else:
                candidate["execution"]["metadata"] = {
                    "nlm_diagnostic": {key: True}
                }
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, path):
                strip_offline_truth(candidate)

    def test_partition_markers_are_versioned_complete_and_collision_safe(self) -> None:
        for malformed in (
            {"audit": {"truth": {"clean_case": "answer"}}, "case": "case14"},
            {"grouping": {"scenario_family": "measurement"}, "case": "case14"},
            {
                "execution": {"case": "case14", "measurements": []},
                "audit": {},
                "grouping": {},
            },
            {
                "scenario_schema_version": True,
                "execution": {"case": "case14", "measurements": []},
                "audit": {},
                "grouping": {},
            },
            {
                "scenario_schema_version": 1.0,
                "execution": {"case": "case14", "measurements": []},
                "audit": {},
                "grouping": {},
            },
            {
                "scenarioSchemaVersion": 1,
                "Execution": {"case": "case14", "measurements": []},
                "Audit": {},
                "Grouping": {},
            },
        ):
            with self.subTest(malformed=sorted(malformed)):
                with self.assertRaisesRegex(ValueError, "partitioned scenario"):
                    strip_offline_truth(malformed)

        collision = {
            "scenario_schema_version": 1,
            "execution": {
                "scenario_id": "collision",
                "case": "dirty-case",
                "measurements": [9.0],
            },
            "audit": {
                "case": "forged-clean-case",
                "truth": {"measurements": [1.0]},
            },
            "grouping": {"scenario_family": "measurement"},
        }
        with self.assertRaisesRegex(ValueError, "collide"):
            strip_offline_truth(collision)

    def test_partitioned_grouping_and_metadata_schema_fail_closed(self) -> None:
        malformed: list[tuple[dict[str, Any], str]] = []
        missing = _partitioned_resolved_scenario()
        missing["grouping"].pop("source_tier")
        malformed.append((missing, "missing required fields"))
        empty = _partitioned_resolved_scenario()
        empty["grouping"]["split"] = ""
        malformed.append((empty, "grouping.split must be non-empty"))
        fractional = _partitioned_resolved_scenario()
        fractional["grouping"]["error_cardinality"] = 1.5
        malformed.append((fractional, "error_cardinality must be a non-negative integer"))
        ambiguous = _partitioned_resolved_scenario()
        ambiguous["grouping"]["family"] = ambiguous["grouping"]["scenario_family"]
        malformed.append((ambiguous, "grouping contains unsupported fields"))
        unknown_metadata = _partitioned_resolved_scenario()
        unknown_metadata["execution"]["metadata"] = {"unknown_runtime": True}
        malformed.append((unknown_metadata, "metadata contains unsupported fields"))

        for scenario, message in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    strip_offline_truth(scenario)

        valid = _partitioned_resolved_scenario()
        valid_metadata = {
            "semantic_field_provenance": {},
            "unresolved_signatures": [],
            "remaining_anomaly_score": 0.0,
            "no_material_anomaly_remaining": False,
            "requires_measurement_context": True,
            "measurement_covariance": [],
            "slack_bus": 1,
            "pristine_model_dir": "/models/pristine",
            "faulted_model_dir": "/models/faulted",
            "load_scale": 1.0,
            "parameter_scans": {},
            "harmonic_measurements": [],
            "harmonic_orders": [],
            "nlm_diagnostic": {},
            "hif_runtime": {},
            "hif_scan_window": {},
            "three_phase_voltages": [],
        }
        valid["execution"]["metadata"] = valid_metadata
        self.assertEqual(strip_offline_truth(valid)["metadata"], valid_metadata)

    def test_conflicting_truth_aliases_fail_closed(self) -> None:
        scenario = _resolved_scenario()
        scenario["ground_truth"] = {
            "true_measurement_errors": [{"index": 6}],
        }

        with self.assertRaisesRegex(
            ValueError, "conflicting offline truth field 'true_measurement_errors'"
        ):
            evaluate_rollout_suites(
                [scenario],
                env_factory=_ScriptEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                max_steps=8,
            )

    def test_normalized_ground_truth_container_drives_strict_audit(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "normalized-ground-truth"
        scenario["physical_root_fingerprint"] = "fp-normalized-ground-truth"
        scenario["Ground-Truth"] = {
            "cleanCase": scenario.pop("clean_case"),
            "cleanMeasurements": scenario.pop("clean_measurements"),
            "trueMeasurementErrors": scenario.pop("true_measurement_errors"),
            "trueHarmonicErrors": scenario.pop("hidden_truth")[
                "true_harmonic_errors"
            ],
            "truthComplete": True,
        }

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )

        episode = result.suite_metrics["episodes"][0]
        self.assertTrue(episode["final_physical_success"])
        self.assertEqual(episode["audit"]["audit_mode"], "strict_release_audit")

    def test_partitioned_ground_truth_locations_drive_strict_audit(self) -> None:
        for nested in (False, True):
            scenario = _partitioned_resolved_scenario()
            truth = scenario["audit"].pop("truth")
            scenario["audit"] = (
                {"truth": {"groundTruth": truth}}
                if nested
                else {"groundTruth": truth}
            )
            with self.subTest(nested=nested):
                result = evaluate_rollout_suites(
                    [scenario],
                    env_factory=_ScriptEnv,
                    policy_factory=lambda: _ScriptPolicy([]),
                    max_steps=8,
                )
                self.assertTrue(
                    result.suite_metrics["episodes"][0]["final_physical_success"]
                )

    def test_mapping_oracle_ground_truth_is_canonicalized(self) -> None:
        class GroundTruthOracleEnv(_ScriptEnv):
            def get_oracle_state(
                self, history: list[Mapping[str, Any]]
            ) -> dict[str, Any]:
                del history
                return {
                    "groundTruth": {
                        "truthComplete": True,
                        "remainingTrueFaults": [{"family": "measurement"}],
                        "remainingTrueFaultCount": 1,
                    }
                }

        scenario = _resolved_scenario()
        scenario["script"] = [
            {
                "phase": "finalize",
                "terminal_outcome": "resolved",
            }
        ]
        result = evaluate_rollout_suites(
            [scenario],
            env_factory=GroundTruthOracleEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )
        self.assertEqual(
            result.suite_metrics["episodes"][0]["false_finalization_count"], 1
        )

    def test_semantically_conflicting_truth_evidence_fails_closed(self) -> None:
        clean_conflict = _resolved_scenario()
        clean_conflict["clean_state"] = {
            "case": copy.deepcopy(clean_conflict["clean_case"]),
            "measurements": [99.0] * len(clean_conflict["clean_measurements"]),
        }
        count_conflict = _resolved_scenario()
        count_conflict["remaining_true_fault_count"] = 2
        count_conflict["remaining_fault_count"] = 1

        for scenario, message in (
            (clean_conflict, "clean_measurements"),
            (count_conflict, "remaining_true_fault_count"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_rollout_suites(
                        [scenario],
                        env_factory=_ScriptEnv,
                        policy_factory=lambda: _ScriptPolicy([]),
                        max_steps=8,
                    )

    def test_legacy_grouping_fields_never_reach_reset(self) -> None:
        scenario = {
            "scenario_id": "legacy-grouping",
            "case": "case14",
            "measurements": [1.0],
            "metadata": {"observable_runtime": True},
            "scenario_family": "measurement+parameter",
            "error_cardinality": 2,
            "split": "held-out",
            "source_tier": "teacher-label",
            "physical_root_fingerprint": "root-secret",
            "root_scenario_id": "root-secret",
        }
        execution = strip_offline_truth(scenario)
        self.assertEqual(
            execution,
            {
                "scenario_id": "legacy-grouping",
                "case": "case14",
                "measurements": [1.0],
                "metadata": {"observable_runtime": True},
            },
        )

    def test_hidden_truth_changes_only_offline_audit_not_trajectory(self) -> None:
        class RecordingEnv(_ScriptEnv):
            instances: list["RecordingEnv"] = []

            def __init__(self, *, seed: int | None = None) -> None:
                super().__init__(seed=seed)
                self.transcript: dict[str, Any] = {}
                self.__class__.instances.append(self)

            def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
                self.transcript = {
                    "reset_scenario": copy.deepcopy(dict(scenario)),
                    "observations": [],
                    "transitions": [],
                }
                state = super().reset(scenario)
                self.transcript["initial_state"] = copy.deepcopy(state)
                return state

            def get_policy_observation(
                self, history: list[Mapping[str, Any]]
            ) -> dict[str, Any]:
                observation = super().get_policy_observation(history)
                self.transcript["observations"].append(copy.deepcopy(observation))
                return observation

            def step(
                self, action: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                state, output = super().step(action)
                self.transcript["transitions"].append(
                    {
                        "action": copy.deepcopy(dict(action)),
                        "tool_output": copy.deepcopy(output),
                        "lifecycle_state": copy.deepcopy(state),
                        "terminal": self.terminal,
                        "terminal_outcome": self.terminal_outcome,
                    }
                )
                return state, output

        first_scenario = _resolved_scenario()
        first_scenario["evaluation_labels"] = [
            {"action_costs": {"invalid_action": 1.0}}
        ]
        first_scenario["metadata"] = {
            "hidden_truth": {"true_nested_secret": "first"},
            "clean_nested_reference": [1.0],
            "oracle_action_hints": [{"tool": "hidden_first"}],
            "tool_cost_labels": {"0": {"best_cost": 0.0}},
        }
        first_scenario["oracle_action_hints"] = [{"tool": "hidden_first"}]
        second_scenario = copy.deepcopy(first_scenario)
        second_scenario["true_measurement_errors"] = [{"index": 6}]
        second_scenario["hidden_truth"] = {
            "true_harmonic_errors": [{"bus_1based": 5}]
        }
        second_scenario["metadata"] = {
            "hidden_truth": {"true_nested_secret": "second"},
            "clean_nested_reference": [2.0],
            "oracle_action_hints": [{"tool": "hidden_second"}],
        }
        second_scenario["oracle_action_hints"] = [{"tool": "hidden_second"}]

        def run(scenario: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            RecordingEnv.instances.clear()
            result = evaluate_rollout_suites(
                [scenario],
                env_factory=RecordingEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                max_steps=8,
                seed=109,
            )
            self.assertEqual(len(RecordingEnv.instances), 1)
            return (
                copy.deepcopy(RecordingEnv.instances[0].transcript),
                result.suite_metrics["episodes"][0],
            )

        first_transcript, first_episode = run(first_scenario)
        second_transcript, second_episode = run(second_scenario)

        # The reset payload, observations, actions, complete tool outputs,
        # candidate lifecycle, and terminal trace are byte-for-byte equivalent.
        self.assertEqual(first_transcript, second_transcript)
        self.assertEqual(first_episode["trace"], second_episode["trace"])
        self.assertEqual(first_episode["terminal"], second_episode["terminal"])
        self.assertEqual(
            first_episode["terminal_outcome"], second_episode["terminal_outcome"]
        )

        reset_payload = first_transcript["reset_scenario"]

        def privileged_keys(value: Any) -> list[str]:
            if isinstance(value, Mapping):
                found = [
                    str(key)
                    for key in value
                    if str(key).startswith(("true_", "clean_"))
                    or str(key)
                    in {
                        "evaluation_labels",
                        "hidden_truth",
                        "oracle_action_hints",
                        "tool_cost_labels",
                    }
                ]
                return found + [
                    key
                    for item in value.values()
                    for key in privileged_keys(item)
                ]
            if isinstance(value, (list, tuple)):
                return [key for item in value for key in privileged_keys(item)]
            return []

        self.assertEqual(privileged_keys(reset_payload), [])
        self.assertTrue(first_episode["final_physical_success"])
        self.assertFalse(second_episode["final_physical_success"])
        self.assertNotEqual(first_episode["audit"], second_episode["audit"])

    def test_rejects_privileged_custom_policy_observation(self) -> None:
        for leaked_key in (
            "hidden_truth",
            "HiddenTruth",
            "GROUND_TRUTH",
            "TrueMeasurementErrors",
        ):
            with self.subTest(leaked_key=leaked_key):
                class LeakyEnv(_ScriptEnv):
                    def get_policy_observation(
                        self, history: list[Mapping[str, Any]]
                    ) -> dict[str, Any]:
                        observation = super().get_policy_observation(history)
                        observation[leaked_key] = ["private"]
                        return observation

                with self.assertRaisesRegex(ValueError, "Privileged fields"):
                    evaluate_rollout_suites(
                        [_resolved_scenario()],
                        env_factory=LeakyEnv,
                        policy_factory=lambda: _ScriptPolicy([]),
                    )

    def test_grouped_correction_touching_one_healthy_meter_fails_preservation(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "broad-correction"
        scenario["scenario_family"] = "measurement"
        scenario["error_cardinality"] = 1
        scenario["hidden_truth"] = {}
        scenario["script"] = [
            {
                "phase": "partial_commit",
                "candidate_state_id": "candidate-1",
                "disposition": "ACCEPT_PARTIAL",
                "remaining": 1,
                "accepted_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "active",
                        "suspect_group": [6, 7],
                    },
                },
            },
            {"phase": "finalize", "remaining": 0, "terminal_outcome": "resolved"},
        ]

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertFalse(episode["healthy_components_preserved"])
        self.assertFalse(episode["final_physical_correct"])
        self.assertIn(
            "measurement_healthy_targets_modified:[6]",
            episode["audit"]["accepted_target_audit"]["problems"],
        )
        self.assertTrue(episode["audit"]["evidence_complete"])
        self.assertIn(
            "accepted_target_nonregression_false_target",
            episode["audit"]["strict_release_audit"]["problems"],
        )

    def test_malformed_accepted_target_remains_an_evidence_gap(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "malformed-accepted-target"
        scenario["scenario_family"] = "measurement"
        scenario["error_cardinality"] = 1
        scenario["hidden_truth"] = {}
        scenario["script"] = [
            {
                "phase": "partial_commit",
                "candidate_state_id": "candidate-1",
                "disposition": "ACCEPT_PARTIAL",
                "remaining": 1,
                "accepted_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "active",
                        "suspect_group": [7.5],
                    },
                },
            },
            {"phase": "finalize", "remaining": 0, "terminal_outcome": "resolved"},
        ]

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertFalse(episode["audit"]["evidence_complete"])
        self.assertIn(
            "accepted_target_nonregression_target_evidence_invalid",
            episode["audit"]["strict_release_audit"]["problems"],
        )

    def test_out_of_range_accepted_target_remains_an_evidence_gap(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "out-of-range-accepted-target"
        scenario["scenario_family"] = "measurement"
        scenario["error_cardinality"] = 1
        scenario["hidden_truth"] = {}
        scenario["script"] = [
            {
                "phase": "partial_commit",
                "candidate_state_id": "candidate-1",
                "disposition": "ACCEPT_PARTIAL",
                "remaining": 1,
                "accepted_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "active",
                        "suspect_group": [7, 99],
                    },
                },
            },
            {"phase": "finalize", "remaining": 0, "terminal_outcome": "resolved"},
        ]

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertFalse(episode["audit"]["evidence_complete"])
        self.assertIn(
            "accepted_measurement_target_out_of_range_or_unverifiable",
            episode["audit"]["strict_release_audit"]["problems"],
        )

    def test_default_audit_rejects_correct_target_with_wrong_final_value(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "wrong-value"
        scenario["physical_root_fingerprint"] = "fp-wrong-value"
        scenario["initial_physical_state"]["measurements"][7] = 2.0

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertTrue(episode["physical_correctness_known"])
        self.assertFalse(episode["final_physical_correct"])
        self.assertFalse(episode["final_physical_success"])
        self.assertEqual(episode["audit"]["audit_mode"], "strict_release_audit")
        self.assertIn(
            "final_measurements_outside_clean_tolerance",
            episode["audit"]["strict_release_audit"]["problems"],
        )

    def test_default_audit_promotes_hidden_standard_truth_for_strict_check(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "hidden-standard-truth"
        scenario["hidden_truth"]["true_measurement_errors"] = scenario.pop(
            "true_measurement_errors"
        )
        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertTrue(episode["final_physical_success"])
        self.assertEqual(
            episode["audit"]["strict_release_audit"]["checks"]
            ["accepted_correction_targets"]["status"],
            "passed",
        )

    def test_default_audit_fails_closed_without_active_physical_state(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "missing-physical-state"
        scenario.pop("initial_physical_state")

        result = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertFalse(episode["physical_correctness_known"])
        self.assertFalse(episode["final_physical_correct"])
        self.assertFalse(episode["final_physical_success"])
        self.assertEqual(episode["audit"]["audit_mode"], "insufficient_evidence")
        self.assertIn(
            "active_physical_state_unavailable", episode["audit"]["problems"]
        )

    def test_case_reference_requires_loader_before_physical_success(self) -> None:
        scenario = _resolved_scenario()
        scenario["scenario_id"] = "case-reference"
        scenario["case"] = "case14"
        scenario["clean_case"] = "case14"
        physical_case = copy.deepcopy(scenario["initial_physical_state"]["case"])

        without_loader = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        ).suite_metrics["episodes"][0]
        self.assertFalse(without_loader["physical_correctness_known"])
        self.assertIn(
            "case_loader_required_for_physical_comparison",
            without_loader["audit"]["problems"],
        )

        with_loader = evaluate_rollout_suites(
            [scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            case_loader=lambda value: (
                copy.deepcopy(physical_case) if value == "case14" else value
            ),
            max_steps=8,
        ).suite_metrics["episodes"][0]
        self.assertTrue(with_loader["final_physical_success"])

    def test_invalid_action_recovery_requires_verified_resolved_terminal(self) -> None:
        resolved = evaluate_rollout_suites(
            [_resolved_scenario()],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        ).suite_metrics["episodes"][0]
        self.assertEqual(resolved["invalid_action_count"], 1)
        self.assertEqual(resolved["recovered_invalid_action_count"], 1)

        escalated_scenario = _escalation_scenario()
        escalated_scenario["script"].insert(
            0, {"phase": "malformed", "status": "failure", "remaining": 2}
        )
        escalated = evaluate_rollout_suites(
            [escalated_scenario],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        ).suite_metrics["episodes"][0]
        self.assertEqual(escalated["invalid_action_count"], 1)
        self.assertFalse(escalated["healthy_preservation_known"])
        self.assertEqual(escalated["recovered_invalid_action_count"], 0)

        physically_wrong = _resolved_scenario()
        physically_wrong["scenario_id"] = "wrong-after-invalid"
        physically_wrong["physical_root_fingerprint"] = "fp-wrong-after-invalid"
        physically_wrong["initial_physical_state"]["measurements"][7] = 3.0
        failed_audit = evaluate_rollout_suites(
            [physically_wrong],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        ).suite_metrics["episodes"][0]
        self.assertEqual(failed_audit["invalid_action_count"], 1)
        self.assertEqual(failed_audit["recovered_invalid_action_count"], 0)

    def test_rejects_empty_and_undercovered_suites(self) -> None:
        kwargs = {
            "env_factory": _ScriptEnv,
            "policy_factory": lambda: _ScriptPolicy([]),
        }
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_rollout_suites([], **kwargs)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_rollout_suites({"standard_success": []}, **kwargs)
        with self.assertRaisesRegex(ValueError, "suite_count"):
            evaluate_rollout_suites(
                [_resolved_scenario()], minimum_suites=2, **kwargs
            )
        with self.assertRaisesRegex(ValueError, "episodes=1"):
            evaluate_rollout_suites(
                [_resolved_scenario()], minimum_episodes_per_suite=2, **kwargs
            )
        duplicate_root = _resolved_scenario()
        duplicate_root["scenario_id"] = "same-root-descendant"
        with self.assertRaisesRegex(ValueError, "distinct_roots=1"):
            evaluate_rollout_suites(
                [_resolved_scenario(), duplicate_root],
                minimum_roots_per_suite=2,
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "missing_required_suites"):
            evaluate_rollout_suites(
                [_resolved_scenario()], required_suites=["efficiency"], **kwargs
            )

    def test_suite_content_and_root_hashes_are_deterministic(self) -> None:
        first_scenario = _resolved_scenario()
        second_scenario = _escalation_scenario()

        def run(rows: list[dict[str, Any]]) -> dict[str, Any]:
            result = evaluate_rollout_suites(
                {"standard_success": rows},
                env_factory=_ScriptEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                max_steps=8,
            )
            return result.suite_metrics["configuration"]

        first = run([first_scenario, second_scenario])
        reordered = run([second_scenario, first_scenario])
        self.assertEqual(first["suite_content_sha256"], reordered["suite_content_sha256"])
        self.assertEqual(first["root_set_sha256"], reordered["root_set_sha256"])
        self.assertTrue(first["suite_coverage_validation"]["passed"])

        changed = copy.deepcopy(first_scenario)
        changed["source_tier"] = "changed-source"
        changed_configuration = run([changed, second_scenario])
        self.assertNotEqual(
            first["suite_content_sha256"],
            changed_configuration["suite_content_sha256"],
        )
        self.assertEqual(first["root_set_sha256"], changed_configuration["root_set_sha256"])

    def test_runtime_environment_records_cuda_device_identity(self) -> None:
        properties = SimpleNamespace(
            name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            total_memory=102_844_334_080,
        )
        fake_torch = SimpleNamespace(
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                is_bf16_supported=lambda: True,
                get_device_properties=lambda index: properties,
                get_device_capability=lambda index: (12, 0),
            ),
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout="570.124.06\n",
        )

        with mock.patch(
            "psse_env.dagger.evaluator.subprocess.run",
            return_value=completed,
        ), mock.patch(
            "psse_env.dagger.evaluator.importlib.import_module",
            return_value=fake_torch,
        ):
            accelerator = _runtime_environment_descriptor()["accelerator"]

        self.assertEqual(accelerator["backend"], "cuda")
        self.assertTrue(accelerator["cuda_available"])
        self.assertEqual(accelerator["torch_cuda_version"], "12.8")
        self.assertEqual(accelerator["driver_version"], "570.124.06")
        self.assertEqual(accelerator["device_count"], 1)
        self.assertTrue(accelerator["bf16_supported"])
        self.assertEqual(
            accelerator["devices"],
            [
                {
                    "index": 0,
                    "name": (
                        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
                    ),
                    "total_memory_bytes": 102_844_334_080,
                    "compute_capability": [12, 0],
                    "accelerator_class": "rtx6000",
                }
            ],
        )
        self.assertEqual(
            normalize_accelerator_class(
                "NVIDIA H100 80GB HBM3",
                85_899_345_920,
            ),
            "h100",
        )

    def test_provenance_rejects_import_spec_callable_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "suite.json"
            input_path.write_text(
                json.dumps(
                    {"standard_success": [_release_partitioned_resolved_scenario()]}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "factory import spec does not resolve to the supplied callable",
            ):
                build_evaluation_provenance(
                    input_suite_path=input_path,
                    environment_factory_spec=(
                        "psse_env.dagger.test_evaluator:_ReleaseScriptEnv"
                    ),
                    environment_factory=_cli_policy_factory,
                    policy_factory_spec=(
                        "psse_env.dagger.test_evaluator:_cli_policy_factory"
                    ),
                    policy_factory=_cli_policy_factory,
                    model_id="test/script-policy",
                    model_revision="a" * 40,
                )

    def test_cli_persists_deterministic_release_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "suite.json"
            output_path = root / "release.json"
            input_path.write_text(
                json.dumps(
                    {"standard_success": [_release_partitioned_resolved_scenario()]}
                ),
                encoding="utf-8",
            )
            arguments = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--env-factory",
                "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
                "--policy-factory",
                "psse_env.dagger.test_evaluator:_cli_policy_factory",
                "--case-loader",
                "psse_env.dagger.test_evaluator:_cli_case_loader",
                "--model-id",
                "test/script-policy",
                "--model-revision",
                "a" * 40,
                "--required-suite",
                "standard_success",
                "--max-steps",
                "8",
            ]
            clean_source = {
                "source_commit": "b" * 40,
                "source_worktree_dirty": False,
                "tracked_diff_hash": hashlib.sha256(b"").hexdigest(),
                "untracked_source_files": [],
                "release_eligible_source": True,
            }
            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=clean_source,
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(evaluator_main(arguments), 0)
            first_bytes = output_path.read_bytes()
            artifact = json.loads(first_bytes)
            self.assertEqual(
                artifact["artifact_type"], "closed_loop_release_evaluation"
            )
            self.assertEqual(
                artifact["artifact_schema_version"],
                STUDY_EVALUATION_SCHEMA_VERSION,
            )
            self.assertEqual(
                artifact["evaluation"]["suite_metrics"]["schema_version"],
                STUDY_EVALUATION_SCHEMA_VERSION,
            )
            self.assertTrue(artifact["release_eligible"])
            self.assertEqual(artifact["release_failures"], [])
            self.assertEqual(len(artifact["content_sha256"]), 64)
            configuration = artifact["evaluation"]["suite_metrics"]["configuration"]
            self.assertTrue(configuration["suite_coverage_validation"]["passed"])
            self.assertEqual(
                configuration["release_environment_validation"],
                {
                    "passed": True,
                    "episodes_checked": 1,
                    "required": {
                        "production_dataset_mode": True,
                        "candidate_quality_oracle_mode": "deployment",
                    },
                    "observed": [
                        {
                            "production_dataset_mode": True,
                            "candidate_quality_oracle_mode": "deployment",
                        }
                    ],
                    "failures": [],
                },
            )
            self.assertEqual(
                configuration["release_scenario_schema_validation"],
                {"passed": True, "scenario_schema_version": 1},
            )
            self.assertEqual(
                configuration["policy_identity_validation"],
                {
                    "passed": True,
                    "episodes_checked": 1,
                    "required": {
                        "explicit_policy_identity": None,
                        "model_id": "test/script-policy",
                        "model_revision": "a" * 40,
                    },
                    "observed": [
                        {
                            "explicit_policy_identity": None,
                            "model_id": "test/script-policy",
                            "model_revision": "a" * 40,
                        }
                    ],
                    "failures": [],
                },
            )

            self.assertIn("standard_success", configuration["suite_content_hashes"])
            provenance = artifact["provenance"]
            self.assertEqual(provenance["source_state"], clean_source)
            self.assertEqual(
                provenance["factories"]["environment"]["import_spec"],
                "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
            )
            self.assertEqual(
                provenance["factories"]["policy"]["import_spec"],
                "psse_env.dagger.test_evaluator:_cli_policy_factory",
            )
            self.assertEqual(
                provenance["factories"]["case_loader"]["import_spec"],
                "psse_env.dagger.test_evaluator:_cli_case_loader",
            )
            for descriptor in provenance["factories"].values():
                self.assertEqual(len(descriptor["source"]["sha256"]), 64)
            self.assertEqual(
                provenance["policy_identity"],
                {
                    "explicit_policy_identity": None,
                    "model_id": "test/script-policy",
                    "model_revision": "a" * 40,
                },
            )
            self.assertEqual(
                provenance["input_suite"]["resolved_path"],
                str(input_path.resolve()),
            )
            self.assertEqual(
                provenance["input_suite"]["sha256"],
                hashlib.sha256(input_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["protocol_registry"]["protocol"], "canonical"
            )
            self.assertEqual(
                provenance["runtime_environment"]["python_implementation"],
                platform.python_implementation(),
            )
            self.assertIn("torch", provenance["runtime_environment"]["packages"])
            self.assertIn(
                "transformers", provenance["runtime_environment"]["packages"]
            )
            self.assertEqual(
                len(provenance["protocol_registry"]["registry_sha256"]), 64
            )
            self.assertEqual(len(provenance["evaluator_source"]["sha256"]), 64)
            self.assertEqual(len(provenance["identity_sha256"]), 64)

            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=clean_source,
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(evaluator_main(arguments), 0)
            self.assertEqual(first_bytes, output_path.read_bytes())

    def test_cli_diagnostic_artifact_can_never_be_release_or_training_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "diagnostic-suite.json"
            output_path = root / "diagnostic-evaluation.json"
            input_path.write_text(
                json.dumps(
                    {
                        "standard_success": [
                            _release_partitioned_resolved_scenario()
                        ]
                    }
                ),
                encoding="utf-8",
            )
            arguments = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--env-factory",
                "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
                "--policy-factory",
                "psse_env.dagger.test_evaluator:_cli_policy_factory",
                "--case-loader",
                "psse_env.dagger.test_evaluator:_cli_case_loader",
                "--model-id",
                "test/script-policy",
                "--model-revision",
                "a" * 40,
                "--required-suite",
                "standard_success",
                "--max-steps",
                "8",
                "--diagnostic-only",
            ]
            clean_source = {
                "source_commit": "b" * 40,
                "source_worktree_dirty": False,
                "tracked_diff_hash": hashlib.sha256(b"").hexdigest(),
                "untracked_source_files": [],
                "release_eligible_source": True,
            }
            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=clean_source,
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(evaluator_main(arguments), 0)

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["artifact_type"],
                "closed_loop_diagnostic_evaluation",
            )
            self.assertTrue(artifact["diagnostic_only"])
            self.assertFalse(artifact["release_evidence_eligible"])
            self.assertFalse(artifact["training_eligible"])
            self.assertFalse(artifact["release_eligible"])
            self.assertIn(
                "diagnostic-only evaluation artifacts are not release evidence",
                artifact["release_failures"],
            )

    def test_cli_requires_policy_or_immutable_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "suite.json"
            input_path.write_text(
                json.dumps(
                    {"standard_success": [_release_partitioned_resolved_scenario()]}
                ),
                encoding="utf-8",
            )
            base = [
                "--input",
                str(input_path),
                "--output",
                str(root / "release.json"),
                "--env-factory",
                "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
                "--policy-factory",
                "psse_env.dagger.test_evaluator:_cli_policy_factory",
                "--required-suite",
                "standard_success",
            ]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluator_main(base)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluator_main(
                    base
                    + [
                        "--model-id",
                        "test/script-policy",
                        "--model-revision",
                        "mutable-main",
                    ]
                )

    def test_cli_rejects_script_policy_relabelled_as_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "suite.json"
            input_path.write_text(
                json.dumps(
                    {"standard_success": [_release_partitioned_resolved_scenario()]}
                ),
                encoding="utf-8",
            )
            arguments = [
                "--input",
                str(input_path),
                "--output",
                str(root / "release.json"),
                "--env-factory",
                "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
                "--policy-factory",
                "psse_env.dagger.test_evaluator:_unattested_policy_factory",
                "--model-id",
                "base/gemma",
                "--model-revision",
                "a" * 40,
                "--required-suite",
                "standard_success",
            ]
            clean_source = {
                "source_commit": "b" * 40,
                "source_worktree_dirty": False,
                "tracked_diff_hash": hashlib.sha256(b"").hexdigest(),
                "untracked_source_files": [],
                "release_eligible_source": True,
            }
            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=clean_source,
            ), self.assertRaisesRegex(ValueError, "policy identity validation failed"):
                evaluator_main(arguments)
            self.assertFalse((root / "release.json").exists())

    def test_cli_dirty_source_requires_override_and_stays_nonrelease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "suite.json"
            output_path = root / "release.json"
            input_path.write_text(
                json.dumps(
                    {"standard_success": [_release_partitioned_resolved_scenario()]}
                ),
                encoding="utf-8",
            )
            arguments = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--env-factory",
                "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
                "--policy-factory",
                "psse_env.dagger.test_evaluator:_cli_policy_factory",
                "--policy-identity",
                "observable-rule-policy-v1",
                "--required-suite",
                "standard_success",
                "--max-steps",
                "8",
            ]
            dirty_source = {
                "source_commit": "c" * 40,
                "source_worktree_dirty": True,
                "tracked_diff_hash": "d" * 64,
                "untracked_source_files": ["policy.py"],
                "release_eligible_source": False,
            }
            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=dirty_source,
            ), self.assertRaisesRegex(RuntimeError, "clean tracked commit"):
                evaluator_main(arguments)
            self.assertFalse(output_path.exists())

            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=dirty_source,
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        evaluator_main(arguments + ["--allow-dirty-source"]), 0
                    )
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(artifact["release_eligible"])
            self.assertTrue(
                any(
                    "clean tracked commit" in failure
                    for failure in artifact["release_failures"]
                )
            )
            self.assertEqual(
                artifact["provenance"]["policy_identity"][
                    "explicit_policy_identity"
                ],
                "observable-rule-policy-v1",
            )

    def test_library_artifact_without_identity_is_backward_compatible_but_nonrelease(
        self,
    ) -> None:
        result = evaluate_rollout_suites(
            [_resolved_scenario()],
            env_factory=_ScriptEnv,
            policy_factory=lambda: _ScriptPolicy([]),
            max_steps=8,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "library.json"
            artifact = write_evaluation_artifact(result, output_path)
            self.assertTrue(output_path.is_file())
            self.assertFalse(artifact["release_eligible"])
            self.assertIsNone(artifact["provenance"])
            self.assertEqual(
                artifact["release_failures"],
                [
                    "evaluation identity provenance is missing",
                    "input suite is not canonical release scenario schema version 1",
                    "instantiated policy identity did not match the release provenance identity",
                ],
            )

    def test_flat_suite_cannot_self_attest_as_release_eligible(self) -> None:
        expected_identity = {
            "explicit_policy_identity": None,
            "model_id": "test/script-policy",
            "model_revision": "a" * 40,
        }
        result = evaluate_rollout_suites(
            [_resolved_scenario()],
            env_factory=_ReleaseScriptEnv,
            policy_factory=_cli_policy_factory,
            max_steps=8,
            expected_policy_identity=expected_identity,
            require_policy_identity=True,
        )
        configuration = result.suite_metrics["configuration"]
        self.assertTrue(configuration["release_environment_validation"]["passed"])
        self.assertTrue(configuration["policy_identity_validation"]["passed"])
        self.assertFalse(
            configuration["release_scenario_schema_validation"]["passed"]
        )

        clean_source = {
            "source_commit": "b" * 40,
            "source_worktree_dirty": False,
            "tracked_diff_hash": hashlib.sha256(b"").hexdigest(),
            "untracked_source_files": [],
            "release_eligible_source": True,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "flat-suite.json"
            input_path.write_text(
                json.dumps({"standard_success": [_resolved_scenario()]}),
                encoding="utf-8",
            )
            with mock.patch(
                "psse_env.dagger.evaluator.git_source_state",
                return_value=clean_source,
            ):
                provenance = build_evaluation_provenance(
                    input_suite_path=input_path,
                    environment_factory_spec=(
                        "psse_env.dagger.test_evaluator:_ReleaseScriptEnv"
                    ),
                    environment_factory=_ReleaseScriptEnv,
                    policy_factory_spec=(
                        "psse_env.dagger.test_evaluator:_cli_policy_factory"
                    ),
                    policy_factory=_cli_policy_factory,
                    model_id="test/script-policy",
                    model_revision="a" * 40,
                )
            artifact = write_evaluation_artifact(
                result, root / "flat-artifact.json", provenance=provenance
            )

        self.assertFalse(artifact["release_eligible"])
        self.assertEqual(
            artifact["release_failures"],
            ["input suite is not canonical release scenario schema version 1"],
        )

    def test_is_reproducible_and_does_not_mutate_supplied_scenarios(self) -> None:
        scenarios = [_escalation_scenario(), _resolved_scenario()]
        original = copy.deepcopy(scenarios)

        def run(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return ClosedLoopRolloutEvaluator(
                env_factory=_ScriptEnv,
                policy_factory=lambda: _ScriptPolicy([]),
                max_steps=8,
                seed=71,
            ).evaluate(rows).as_dict()

        first = run(scenarios)
        second = run(list(reversed(scenarios)))
        self.assertEqual(first, second)
        self.assertEqual(scenarios, original)


if __name__ == "__main__":
    unittest.main()
