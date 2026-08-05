from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINALIZE_DIAGNOSIS,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.dagger.dataset_builder import examples_to_chat_sft
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
    offline_teacher_target_audit,
    validate_offline_teacher_target_audit_metadata,
)
from psse_env.state_store import OracleState, PolicyObservation


def _case() -> dict[str, Any]:
    return {
        "baseMVA": 100.0,
        "bus": [[1, 3]],
        "gen": [[1, 0.0]],
        "branch": [[1, 2, 0.1, 0.2, 0.0, 0, 0, 0, 0, 0, 1]],
    }


def _scenario() -> dict[str, Any]:
    return {
        "scenario_id": "offline-target-audit",
        "physical_root_fingerprint": "offline-target-root",
        "scenario_family": "measurement",
        "case": _case(),
        "clean_case": _case(),
        "measurements": [1.0, 9.0, 3.0],
        "clean_measurements": [1.0, 2.0, 3.0],
        "true_measurement_errors": [{"index": 1, "clean": 2.0}],
        "true_parameter_errors": [
            {
                "branch_row0": 0,
                "line_index1": 1,
                "parameter": "x",
                "clean": 0.2,
            }
        ],
        "true_topology_errors": [
            {"branch_row0": 0, "line_index1": 1, "expected_status": 0}
        ],
        "truth_complete": True,
    }


def _observation(
    *, accepted: list[dict[str, Any]] | None = None, explained=None
) -> PolicyObservation:
    return PolicyObservation(
        active_state_id="episode:s0",
        accepted_corrections=copy.deepcopy(accepted or []),
        explained_anomalies=copy.deepcopy(explained or []),
        remaining_budget=8,
    )


def _oracle(
    truth: dict[str, Any], observation: PolicyObservation | None = None
) -> OracleState:
    return OracleState(
        policy_observation=observation or _observation(),
        clean_case=copy.deepcopy(truth.get("clean_case")),
        clean_measurements=copy.deepcopy(truth.get("clean_measurements")),
        true_measurement_errors=copy.deepcopy(
            truth.get("true_measurement_errors") or []
        ),
        true_parameter_errors=copy.deepcopy(
            truth.get("true_parameter_errors") or []
        ),
        true_topology_errors=copy.deepcopy(
            truth.get("true_topology_errors") or []
        ),
        remaining_true_faults=copy.deepcopy(
            [
                *(truth.get("true_measurement_errors") or []),
                *(truth.get("true_parameter_errors") or []),
                *(truth.get("true_topology_errors") or []),
            ]
        ),
        hidden_truth=copy.deepcopy(truth),
    )


class _Store:
    def __init__(self, states: dict[str, dict[str, Any]], active: str) -> None:
        self.states = copy.deepcopy(states)
        self.active_state_id = active

    def exists(self, state_id: str) -> bool:
        return state_id in self.states

    def get_state(self, state_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.states[state_id])


class _Env:
    def __init__(
        self,
        states: dict[str, dict[str, Any]] | None = None,
        *,
        active: str = "episode:s0",
        candidate: str | None = None,
    ) -> None:
        if states is None:
            states = {
                active: {
                    "state_id": active,
                    "state_hash": "active-hash",
                    "case": _case(),
                    "measurements": [1.0, 9.0, 3.0],
                }
            }
        self.store = _Store(states, active)
        self.current_candidate_id = candidate


class OfflineTeacherTargetAuditTests(unittest.TestCase):
    def _audit(
        self,
        action,
        *,
        truth: dict[str, Any] | None = None,
        observation: PolicyObservation | None = None,
        env: _Env | None = None,
        observable: bool = True,
        scenario: dict[str, Any] | None = None,
    ):
        truth = copy.deepcopy(truth or _scenario())
        observation = observation or _observation()
        return offline_teacher_target_audit(
            preferred_action=action,
            oracle_state=_oracle(truth, observation),
            policy_observation=observation,
            scenario=copy.deepcopy(scenario or truth),
            env=env or _Env(),
            observable_evidence_passed=observable,
            case_loader=None,
        )

    def test_measurement_targets_must_be_nonempty_subset_of_remaining_faults(self):
        good = self._audit(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": "episode:s0",
                    "measurement_updates": {1: 2.0},
                },
            }
        )
        wrong = self._audit(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": "episode:s0",
                    "measurement_updates": {0: 1.0, 1: 2.0},
                },
            }
        )
        self.assertTrue(good["passed"], good)
        self.assertFalse(wrong["passed"])
        self.assertEqual(wrong["action_class"], "measurement_correction")

    def test_parameter_and_topology_target_and_status_are_truth_checked(self):
        parameter_good = self._audit(
            {
                "tool": CORRECT_PARAMETERS,
                "arguments": {"state_id": "episode:s0", "line_index1": 1},
            }
        )
        parameter_bad = self._audit(
            {
                "tool": CORRECT_PARAMETERS,
                "arguments": {"state_id": "episode:s0", "line_index1": 2},
            }
        )
        topology_good = self._audit(
            {
                "tool": CORRECT_TOPOLOGY,
                "arguments": {
                    "state_id": "episode:s0",
                    "line_index1": 1,
                    "status": 0,
                },
            }
        )
        topology_wrong_status = self._audit(
            {
                "tool": CORRECT_TOPOLOGY,
                "arguments": {
                    "state_id": "episode:s0",
                    "line_index1": 1,
                    "status": 1,
                },
            }
        )
        topology_string_open = self._audit(
            {
                "tool": CORRECT_TOPOLOGY,
                "arguments": {
                    "state_id": "episode:s0",
                    "line_index1": 1,
                    "status": "open",
                },
            }
        )
        self.assertTrue(parameter_good["passed"], parameter_good)
        self.assertFalse(parameter_bad["passed"])
        self.assertTrue(topology_good["passed"], topology_good)
        self.assertTrue(topology_string_open["passed"], topology_string_open)
        self.assertFalse(topology_wrong_status["passed"])

    def _candidate_env(self, *, safe: bool) -> _Env:
        source_index = 1 if safe else 0
        candidate_measurements = [1.0, 2.0, 3.0] if safe else [8.0, 9.0, 3.0]
        states = {
            "episode:s0": {
                "state_id": "episode:s0",
                "state_hash": "parent-hash",
                "case": _case(),
                "measurements": [1.0, 9.0, 3.0],
            },
            "episode:s1": {
                "state_id": "episode:s1",
                "state_hash": "candidate-hash",
                "parent_state_id": "episode:s0",
                "case": _case(),
                "measurements": candidate_measurements,
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "episode:s0",
                        "measurement_updates": {source_index: 2.0},
                    },
                },
                "verification_output": {
                    "execution_status": "success",
                    "state_id": "episode:s1",
                    "state_hash": "candidate-hash",
                },
                "candidate_disposition": "ACCEPT_FINAL" if safe else "REJECT",
            },
        }
        return _Env(states, active="episode:s0", candidate="episode:s1")

    def test_commit_requires_truth_safe_candidate_and_rollback_requires_opposite(self):
        commit = {
            "tool": COMMIT_STATE,
            "arguments": {"candidate_state_id": "episode:s1"},
        }
        rollback = {
            "tool": ROLLBACK_STATE,
            "arguments": {"candidate_state_id": "episode:s1"},
        }
        safe_env = self._candidate_env(safe=True)
        unsafe_env = self._candidate_env(safe=False)
        self.assertTrue(self._audit(commit, env=safe_env)["passed"])
        self.assertFalse(self._audit(rollback, env=safe_env)["passed"])
        self.assertFalse(self._audit(commit, env=unsafe_env)["passed"])
        self.assertTrue(self._audit(rollback, env=unsafe_env)["passed"])

    def test_rollback_fails_closed_when_source_or_private_truth_is_unknown(self):
        rollback = {
            "tool": ROLLBACK_STATE,
            "arguments": {"candidate_state_id": "episode:s1"},
        }
        missing_source = self._candidate_env(safe=False)
        del missing_source.store.states["episode:s1"]["source_action"]
        source_report = self._audit(rollback, env=missing_source)
        self.assertFalse(source_report["passed"])
        self.assertFalse(
            source_report["checks"][
                "candidate_source_truth_evidence_complete"
            ]
        )
        self.assertIn(
            "candidate_source_correction_missing",
            source_report["reason_codes"],
        )

        incomplete_truth = _scenario()
        incomplete_truth["truth_complete"] = False
        truth_report = self._audit(
            rollback,
            env=self._candidate_env(safe=False),
            truth=incomplete_truth,
            scenario=incomplete_truth,
        )
        self.assertFalse(truth_report["passed"])
        self.assertFalse(
            truth_report["checks"][
                "candidate_source_truth_evidence_complete"
            ]
        )
        self.assertIn("private_ledger_incomplete", truth_report["reason_codes"])

    def test_finalize_requires_no_remaining_fault_or_correct_diagnostic_explanation(self):
        clean = _scenario()
        for key in (
            "true_measurement_errors",
            "true_parameter_errors",
            "true_topology_errors",
        ):
            clean[key] = []
        clean["measurements"] = copy.deepcopy(clean["clean_measurements"])
        clean_env = _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "clean",
                    "case": _case(),
                    "measurements": copy.deepcopy(clean["measurements"]),
                }
            }
        )
        action = {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
        self.assertTrue(
            self._audit(action, truth=clean, scenario=clean, env=clean_env)["passed"]
        )
        self.assertFalse(self._audit(action)["passed"])

        harmonic = copy.deepcopy(clean)
        harmonic.update(
            {
                "scenario_family": "harmonic",
                "true_harmonic_errors": [{"bus_1based": 5}],
                "release_audit": {
                    "explanation_only_contract": (
                        "explanation_only_diagnostic_localization_v1"
                    ),
                    "not_applicable": {
                        "final_measurements_match_clean": "diagnostic-only"
                    },
                },
            }
        )
        explanation = {
            "family": "harmonic",
            "detail": {"best_candidate_bus_1based": 5},
            "evidence_source": "configured_provider:test",
            "explained_signatures": ["harmonic_distortion"],
        }
        diagnostic_observation = _observation(explained=[explanation])
        self.assertTrue(
            self._audit(
                action,
                truth=harmonic,
                scenario=harmonic,
                observation=diagnostic_observation,
                env=clean_env,
            )["passed"]
        )
        wrong = copy.deepcopy(explanation)
        wrong["detail"]["best_candidate_bus_1based"] = 4
        self.assertFalse(
            self._audit(
                action,
                truth=harmonic,
                scenario=harmonic,
                observation=_observation(explained=[wrong]),
                env=clean_env,
            )["passed"]
        )

    def test_operator_escalation_audits_safety_without_claiming_resolution(self):
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {"request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST},
        }
        accepted = [
            {
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "episode:s0",
                        "measurement_updates": {1: 2.0},
                    },
                }
            }
        ]
        observation = _observation(accepted=accepted)
        corrected = _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "corrected",
                    "case": _case(),
                    "measurements": [1.0, 2.0, 3.0],
                }
            }
        )
        passed = self._audit(action, observation=observation, env=corrected)
        self.assertTrue(passed["passed"], passed)
        self.assertTrue(
            passed["checks"]["terminal_claim_is_handoff_not_resolution"]
        )
        corrupted = _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "corrupted",
                    "case": _case(),
                    "measurements": [99.0, 2.0, 3.0],
                }
            }
        )
        self.assertFalse(
            self._audit(action, observation=observation, env=corrupted)["passed"]
        )

    def test_read_only_actions_use_only_observable_gate(self):
        action = {"tool": RUN_WLS, "arguments": {"state_id": "episode:s0"}}
        self.assertTrue(self._audit(action, observable=True)["passed"])
        self.assertFalse(self._audit(action, observable=False)["passed"])
        unknown = self._audit(
            {"tool": "future_state_mutator", "arguments": {}}, observable=True
        )
        self.assertFalse(unknown["passed"])
        self.assertEqual(unknown["action_class"], "unknown_target")

    def test_hidden_truth_is_post_target_quarantine_only_and_never_chat_input(self):
        action = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {
                "state_id": "episode:s0",
                "measurement_updates": {1: 2.0},
            },
        }
        observation = _observation()
        observation_before = observation.as_dict()
        action_before = copy.deepcopy(action)
        matching = self._audit(action, observation=observation)
        altered_truth = _scenario()
        altered_truth["true_measurement_errors"] = [{"index": 0, "clean": 1.0}]
        altered = self._audit(
            action, truth=altered_truth, observation=observation
        )
        self.assertTrue(matching["passed"])
        self.assertFalse(altered["passed"])
        self.assertEqual(action, action_before)
        self.assertEqual(observation.as_dict(), observation_before)

        row = {
            "example_id": "offline-audit-export",
            "scenario_id": "offline-target-audit",
            "root_scenario_id": "offline-target-audit",
            "policy_observation": observation.as_dict(),
            "history_window": [],
            "preferred_action": {
                "tool": RUN_WLS,
                "arguments": {"state_id": "episode:s0"},
            },
            "production_label_eligible": True,
            "recovery_label_contract": "observable_rank_one_learner_state_v1",
            "offline_teacher_target_audit": {
                "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                "passed": True,
                "action_class": "read_only",
                "checks": {"observable_evidence_gate_passed": True},
                "reason_codes": [],
            },
            "labels": {},
        }
        exported = examples_to_chat_sft([row], protocol="controller")[0]
        self.assertEqual(
            exported["metadata"]["offline_teacher_target_audit"],
            row["offline_teacher_target_audit"],
        )
        visible = json.dumps(exported["messages"], sort_keys=True)
        self.assertNotIn("offline_teacher_target_audit", visible)
        self.assertNotIn(OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT, visible)

        missing = copy.deepcopy(row)
        missing.pop("offline_teacher_target_audit")
        with self.assertRaisesRegex(ValueError, "offline teacher-target audit"):
            examples_to_chat_sft([missing], protocol="controller")

        smuggled = copy.deepcopy(row)
        smuggled["offline_teacher_target_audit"]["hidden_truth"] = {
            "true_measurement_errors": [{"index": 1}]
        }
        with self.assertRaisesRegex(ValueError, "offline teacher-target audit"):
            examples_to_chat_sft([smuggled], protocol="controller")

    def test_passed_metadata_requires_every_action_class_truth_check(self):
        complete_checks = {
            "measurement_correction": {
                "complete_private_ledger": True,
                "target_is_remaining_family_fault": True,
                "observable_evidence_gate_passed": True,
            },
            "parameter_correction": {
                "complete_private_ledger": True,
                "target_is_remaining_family_fault": True,
                "observable_evidence_gate_passed": True,
            },
            "topology_correction": {
                "complete_private_ledger": True,
                "target_is_remaining_family_fault": True,
                "requested_topology_status_matches": True,
                "observable_evidence_gate_passed": True,
            },
            "commit": {
                "candidate_exists": True,
                "candidate_verified": True,
                "candidate_source_truth_evidence_complete": True,
                "candidate_truth_safe_to_commit": True,
                "observable_evidence_gate_passed": True,
            },
            "rollback": {
                "candidate_exists": True,
                "candidate_verified": True,
                "candidate_source_truth_evidence_complete": True,
                "candidate_not_truth_safe_to_commit": True,
                "observable_evidence_gate_passed": True,
            },
            "finalize": {
                "observable_evidence_gate_passed": True,
                "resolved_claim_matches_private_ledger": True,
            },
            "operator_escalation": {
                "accepted_state_nonregressive_and_healthy": True,
                "observable_evidence_gate_passed": True,
                "terminal_claim_is_handoff_not_resolution": True,
            },
            "read_only": {"observable_evidence_gate_passed": True},
        }
        for action_class, checks in complete_checks.items():
            with self.subTest(action_class=action_class):
                valid = {
                    "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                    "passed": True,
                    "action_class": action_class,
                    "checks": copy.deepcopy(checks),
                    "reason_codes": [],
                }
                self.assertEqual(
                    validate_offline_teacher_target_audit_metadata(
                        valid, require_passed=True
                    ),
                    valid,
                )
                incomplete = copy.deepcopy(valid)
                incomplete["checks"].pop(next(iter(checks)))
                with self.assertRaisesRegex(
                    ValueError, "checks must be nonempty|check schema"
                ):
                    validate_offline_teacher_target_audit_metadata(
                        incomplete, require_passed=True
                    )

        for action_class, check in (
            ("missing_target", "teacher_target_present"),
            ("invalid_target", "teacher_target_well_formed"),
            (
                "unknown_target",
                "teacher_target_is_known_nonmutating_action",
            ),
        ):
            with self.subTest(failure_only_action_class=action_class):
                forged = {
                    "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                    "passed": True,
                    "action_class": action_class,
                    "checks": {check: True},
                    "reason_codes": [],
                }
                with self.assertRaisesRegex(ValueError, "cannot pass"):
                    validate_offline_teacher_target_audit_metadata(forged)


if __name__ == "__main__":
    unittest.main()
