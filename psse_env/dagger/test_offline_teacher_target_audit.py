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
    LEGACY_OFFLINE_TEACHER_TARGET_AUDIT_CONTRACTS,
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
    _verified_terminal_measurement_closure_check,
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

    def test_commit_failure_reasons_are_individually_attributable(self):
        # The retired umbrella code could not separate a mis-targeted candidate
        # from an undispositioned one or from new physical harm, so a
        # quarantine could not be triaged without re-running collection.
        report = self._audit(
            {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "episode:s1"}},
            env=self._candidate_env(safe=False),
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            sorted(report["reason_codes"]),
            [
                "candidate_commit_introduces_new_physical_harm",
                "candidate_disposition_not_accepted",
                "candidate_source_target_outside_remaining_truth",
            ],
        )

        # A wrong healthy target that moves only within the declared physical
        # tolerance is still a private target-identity failure, but is not
        # falsely attributed as new numerical harm.
        tolerant_scenario = _scenario()
        tolerant_scenario["release_audit"] = {
            "tolerances": {"measurement_abs": 0.05}
        }
        tolerant_wrong_target = _Env(
            {
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
                    "measurements": [1.01, 9.0, 3.0],
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "episode:s0",
                            "measurement_updates": {0: 1.01},
                        },
                    },
                    "verification_output": {
                        "execution_status": "success",
                        "state_id": "episode:s1",
                        "state_hash": "candidate-hash",
                    },
                    "candidate_disposition": "ACCEPT_FINAL",
                },
            },
            active="episode:s0",
            candidate="episode:s1",
        )
        tolerant_report = self._audit(
            {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "episode:s1"}},
            truth=tolerant_scenario,
            scenario=tolerant_scenario,
            env=tolerant_wrong_target,
        )
        self.assertFalse(tolerant_report["passed"], tolerant_report)
        self.assertEqual(
            tolerant_report["reason_codes"],
            ["candidate_source_target_outside_remaining_truth"],
        )

    def _contaminated_commit_env(self, *, candidate_measurements) -> _Env:
        return _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "parent-hash",
                    "case": _case(),
                    # Index 0 is healthy but already corrupted by a commit the
                    # learner made earlier in this episode.
                    "measurements": [99.0, 9.0, 3.0],
                },
                "episode:s1": {
                    "state_id": "episode:s1",
                    "state_hash": "candidate-hash",
                    "parent_state_id": "episode:s0",
                    "case": _case(),
                    "measurements": list(candidate_measurements),
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "episode:s0",
                            "measurement_updates": {1: 2.0},
                        },
                    },
                    "verification_output": {
                        "execution_status": "success",
                        "state_id": "episode:s1",
                        "state_hash": "candidate-hash",
                    },
                    "candidate_disposition": "ACCEPT_FINAL",
                },
            },
            active="episode:s0",
            candidate="episode:s1",
        )

    def test_commit_is_judged_on_the_harm_it_introduces_not_what_it_inherited(self):
        commit = {
            "tool": COMMIT_STATE,
            "arguments": {"candidate_state_id": "episode:s1"},
        }
        contaminated = _observation(
            accepted=[
                {
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "episode:s0",
                            "measurement_updates": {0: 99.0},
                        },
                    }
                }
            ]
        )
        # Correctly retires the one true measurement fault; the inherited
        # corruption of index 0 is untouched and unfixable from here.
        report = self._audit(
            commit,
            observation=contaminated,
            env=self._contaminated_commit_env(
                candidate_measurements=[99.0, 2.0, 3.0]
            ),
        )
        self.assertTrue(report["passed"], report)

        # The same inherited state, but this commit corrupts index 2 as well.
        worse = self._audit(
            commit,
            observation=contaminated,
            env=self._contaminated_commit_env(
                candidate_measurements=[99.0, 2.0, 42.0]
            ),
        )
        self.assertFalse(worse["passed"], worse)
        self.assertEqual(
            worse["reason_codes"],
            ["candidate_commit_introduces_new_physical_harm"],
        )

        # An inherited false ledger entry must not short-circuit evaluation of
        # the current true target.  This candidate moves the true measurement
        # farther from clean than its parent and must fail attribution.
        regressed_target = self._audit(
            commit,
            observation=contaminated,
            env=self._contaminated_commit_env(
                candidate_measurements=[99.0, 20.0, 3.0]
            ),
        )
        self.assertFalse(regressed_target["passed"], regressed_target)
        self.assertEqual(
            regressed_target["reason_codes"],
            ["candidate_commit_introduces_new_physical_harm"],
        )

        multi_fault_scenario = _scenario()
        multi_fault_scenario.update(
            {
                "measurements": [1.0, 9.0, 8.0],
                "clean_measurements": [1.0, 2.0, 3.0],
                "true_measurement_errors": [
                    {"index": 1, "clean": 2.0},
                    {"index": 2, "clean": 3.0},
                ],
            }
        )
        collateral_true_target_regression = _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "parent-hash",
                    "case": _case(),
                    "measurements": [1.0, 9.0, 8.0],
                },
                "episode:s1": {
                    "state_id": "episode:s1",
                    "state_hash": "candidate-hash",
                    "parent_state_id": "episode:s0",
                    "case": _case(),
                    "measurements": [1.0, 2.0, 100.0],
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "episode:s0",
                            "measurement_updates": {1: 2.0},
                        },
                    },
                    "verification_output": {
                        "execution_status": "success",
                        "state_id": "episode:s1",
                        "state_hash": "candidate-hash",
                    },
                    "candidate_disposition": "ACCEPT_FINAL",
                },
            },
            active="episode:s0",
            candidate="episode:s1",
        )
        collateral_report = self._audit(
            commit,
            truth=multi_fault_scenario,
            scenario=multi_fault_scenario,
            env=collateral_true_target_regression,
        )
        self.assertFalse(collateral_report["passed"], collateral_report)
        self.assertEqual(
            collateral_report["reason_codes"],
            ["candidate_commit_introduces_new_physical_harm"],
        )

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

    def test_handoff_fails_closed_when_private_audit_evidence_is_incomplete(self):
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": "episode:s0",
                "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            },
        }

        incomplete_truth = _scenario()
        incomplete_truth["truth_complete"] = False
        incomplete_report = self._audit(
            action,
            truth=incomplete_truth,
            scenario=incomplete_truth,
        )
        self.assertFalse(incomplete_report["passed"], incomplete_report)
        self.assertEqual(
            incomplete_report["reason_codes"],
            ["handoff_state_audit_unavailable"],
        )

        missing_hif_truth = _scenario()
        missing_hif_truth.update(
            {
                "scenario_family": "hif",
                "true_measurement_errors": [],
                "true_parameter_errors": [],
                "true_topology_errors": [],
            }
        )
        missing_report = self._audit(
            action,
            truth=missing_hif_truth,
            scenario=missing_hif_truth,
        )
        self.assertFalse(missing_report["passed"], missing_report)
        self.assertEqual(
            missing_report["reason_codes"],
            ["handoff_state_audit_unavailable"],
        )

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
        self.assertTrue(passed["checks"]["handoff_state_audit_evaluable"])

    def test_operator_escalation_does_not_inherit_learner_damage(self):
        # An escalation mutates nothing, so damage already committed by the
        # learner is not attributable to this target -- and no action available
        # to the teacher retires a committed correction.  Charging it here
        # would quarantine the only correct response to an unrecoverable state.
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {"request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST},
        }
        observation = _observation(
            accepted=[
                {
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "episode:s0",
                            # Index 0 is a healthy meter: a wrong-target commit.
                            "measurement_updates": {0: 99.0},
                        },
                    }
                }
            ]
        )
        corrupted = _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "corrupted",
                    "case": _case(),
                    "measurements": [99.0, 9.0, 3.0],
                }
            }
        )
        report = self._audit(action, observation=observation, env=corrupted)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reason_codes"], [])

    def test_operator_escalation_fails_closed_on_unverifiable_state(self):
        # Excusing inherited damage must not excuse an audit that could not
        # run: a state whose evidence cannot be verified stays quarantined.
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {"request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST},
        }
        unverifiable = _Env(
            {
                "episode:s0": {
                    "state_id": "episode:s0",
                    "state_hash": "truncated",
                    "case": _case(),
                    # Shorter than the scenario measurement vector.
                    "measurements": [1.0, 9.0],
                }
            }
        )
        report = self._audit(action, env=unverifiable)
        self.assertFalse(report["passed"], report)
        self.assertIn("handoff_state_audit_unavailable", report["reason_codes"])
        self.assertFalse(report["checks"]["handoff_state_audit_evaluable"])

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
                "handoff_state_audit_evaluable": True,
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

    def test_each_contract_validates_only_its_own_escalation_check_set(self):
        # Artifacts collected before v2 must stay readable, and neither
        # contract may borrow the other's check set to claim a pass.
        v1 = sorted(LEGACY_OFFLINE_TEACHER_TARGET_AUDIT_CONTRACTS)[0]
        v1_checks = {
            "accepted_state_nonregressive_and_healthy": True,
            "observable_evidence_gate_passed": True,
            "terminal_claim_is_handoff_not_resolution": True,
        }
        v2_checks = {
            "handoff_state_audit_evaluable": True,
            "observable_evidence_gate_passed": True,
            "terminal_claim_is_handoff_not_resolution": True,
        }

        def record(contract: str, checks: dict[str, bool]) -> dict[str, Any]:
            return {
                "contract": contract,
                "passed": True,
                "action_class": "operator_escalation",
                "checks": copy.deepcopy(checks),
                "reason_codes": [],
            }

        for contract, checks in (
            (v1, v1_checks),
            (OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT, v2_checks),
        ):
            with self.subTest(contract=contract):
                valid = record(contract, checks)
                self.assertEqual(
                    validate_offline_teacher_target_audit_metadata(
                        valid, require_passed=True
                    ),
                    valid,
                )
        for contract, checks in (
            (v1, v2_checks),
            (OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT, v1_checks),
        ):
            with self.subTest(mismatched_contract=contract):
                with self.assertRaisesRegex(ValueError, "check schema"):
                    validate_offline_teacher_target_audit_metadata(
                        record(contract, checks)
                    )

        for contract, checks, retired_reason in (
            (
                v1,
                v1_checks,
                "handoff_state_audit_unavailable",
            ),
            (
                OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                v2_checks,
                "handoff_failed_private_safety_audit",
            ),
        ):
            with self.subTest(cross_version_reason_contract=contract):
                invalid_reason = record(contract, checks)
                invalid_reason["passed"] = False
                invalid_reason["checks"] = {
                    **invalid_reason["checks"],
                    next(iter(checks)): False,
                }
                invalid_reason["reason_codes"] = [retired_reason]
                with self.assertRaisesRegex(ValueError, "reason code"):
                    validate_offline_teacher_target_audit_metadata(
                        invalid_reason
                    )
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_offline_teacher_target_audit_metadata(
                record("dagger1_offline_teacher_target_truth_audit_v0", v2_checks)
            )


if __name__ == "__main__":
    unittest.main()


class VerifiedTerminalMeasurementClosureTests(unittest.TestCase):
    """The audit must admit the designed terminal measurement closure.

    All four DAgger-1 round-2 quarantines were this exact action: every
    previously accepted measurement target plus exactly one new target,
    two-stage screened and state-bound.  Ordinary remaining-target membership
    rejects it because the reused targets were correctly retired from the
    remaining ledger.  These fixtures follow the observed shape of
    r0_9826886a46fd step 17 -- accepted {92, 97}, new {88}.

    Clauses covering non-regression, healthy-state modification, and physical
    constraint failure on the committed candidate are enforced by the existing
    physical-safety audit, not by this helper, and are not duplicated here.
    """

    STATE = "r0_9826886a46fd_episode126:s2"

    def _observation(self) -> dict[str, Any]:
        return {
            "active_state_id": self.STATE,
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {"state_id": "ep:s0", "suspect_group": [92]},
                    }
                },
                {
                    "source_action": {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {"state_id": "ep:s1", "suspect_group": [97]},
                    }
                },
            ],
            "fresh_context_evidence": {
                "measurement": {
                    "state_id": self.STATE,
                    "state_hash": "8508673903d4887d",
                    "verified_terminal_measurement_closure_targets": [88, 92, 97],
                    "verified_terminal_measurement_closure_evidence": {
                        "attempts": [
                            {
                                "stage": "new_target_singleton",
                                "targets": [88],
                                "disposition": "ACCEPT_FINAL",
                                "target_test_passed": True,
                                "physical_constraints_ok": True,
                            },
                            {
                                "stage": "grouped",
                                "targets": [88, 92, 97],
                                "disposition": "ACCEPT_FINAL",
                                "physical_constraints_ok": True,
                            },
                        ]
                    },
                    "supported_corrections": [
                        {
                            "tool": CORRECT_MEASUREMENTS,
                            "arguments": {
                                "state_id": self.STATE,
                                "suspect_group": [88, 92, 97],
                            },
                        }
                    ],
                }
            },
        }

    @classmethod
    def _action(cls, targets: list[int]) -> dict[str, Any]:
        return {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {"state_id": cls.STATE, "suspect_group": list(targets)},
        }

    @staticmethod
    def _faults(indices: list[int]) -> dict[str, Any]:
        return {
            "truth_complete": True,
            "true_measurement_errors": [{"index": i} for i in indices],
        }

    def _check(self, *, action=None, observation=None, scenario=None, truth=None):
        return _verified_terminal_measurement_closure_check(
            action if action is not None else self._action([88, 92, 97]),
            observation=(
                observation if observation is not None else self._observation()
            ),
            scenario=scenario if scenario is not None else self._faults([88, 92, 97]),
            truth=truth if truth is not None else self._faults([88]),
        )

    def _assert_rejected(self, code: str, **kwargs):
        checks, reasons = self._check(**kwargs)
        self.assertIn(code, reasons)
        self.assertFalse(all(checks.values()), checks)

    def test_designed_terminal_closure_is_admitted(self):
        checks, reasons = self._check()
        self.assertEqual(reasons, [])
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()), checks)

    def test_ordinary_correction_without_attestation_is_untouched(self):
        observation = self._observation()
        measurement = observation["fresh_context_evidence"]["measurement"]
        del measurement["verified_terminal_measurement_closure_targets"]
        del measurement["verified_terminal_measurement_closure_evidence"]
        self.assertEqual(self._check(observation=observation), ({}, []))

    def test_missing_state_hash_is_rejected(self):
        observation = self._observation()
        observation["fresh_context_evidence"]["measurement"]["state_hash"] = ""
        self._assert_rejected(
            "closure_context_not_state_bound", observation=observation
        )

    def test_context_bound_to_a_different_state_is_rejected(self):
        observation = self._observation()
        observation["active_state_id"] = "some_other_state:s9"
        self._assert_rejected(
            "closure_context_not_state_bound", observation=observation
        )

    def test_action_absent_from_supported_inventory_is_rejected(self):
        observation = self._observation()
        observation["fresh_context_evidence"]["measurement"][
            "supported_corrections"
        ] = [self._action([88])]
        self._assert_rejected(
            "closure_action_not_in_supported_inventory", observation=observation
        )

    def test_more_than_one_new_target_is_rejected(self):
        observation = self._observation()
        measurement = observation["fresh_context_evidence"]["measurement"]
        measurement["verified_terminal_measurement_closure_targets"] = [
            88,
            89,
            92,
            97,
        ]
        measurement["supported_corrections"] = [self._action([88, 89, 92, 97])]
        self._assert_rejected(
            "closure_new_target_count_not_one",
            action=self._action([88, 89, 92, 97]),
            observation=observation,
            scenario=self._faults([88, 89, 92, 97]),
            truth=self._faults([88, 89]),
        )

    def test_accepted_target_outside_original_truth_is_rejected(self):
        self._assert_rejected(
            "closure_accepted_target_not_original_truth",
            scenario=self._faults([88, 92]),
        )

    def test_new_target_outside_remaining_truth_is_rejected(self):
        self._assert_rejected(
            "closure_new_target_outside_remaining_truth", truth=self._faults([41])
        )

    def test_accepted_target_still_owed_is_rejected(self):
        self._assert_rejected(
            "closure_accepted_target_still_in_remaining_truth",
            truth=self._faults([88, 92]),
        )

    def test_partial_reuse_of_the_accepted_set_is_rejected(self):
        observation = self._observation()
        measurement = observation["fresh_context_evidence"]["measurement"]
        measurement["verified_terminal_measurement_closure_targets"] = [88, 92]
        measurement["supported_corrections"] = [self._action([88, 92])]
        self._assert_rejected(
            "closure_does_not_reuse_entire_accepted_set",
            action=self._action([88, 92]),
            observation=observation,
        )

    def test_reversed_screening_stages_are_rejected(self):
        observation = self._observation()
        observation["fresh_context_evidence"]["measurement"][
            "verified_terminal_measurement_closure_evidence"
        ]["attempts"].reverse()
        self._assert_rejected(
            "closure_screening_incomplete", observation=observation
        )

    def test_grouped_stage_not_accept_final_is_rejected(self):
        observation = self._observation()
        observation["fresh_context_evidence"]["measurement"][
            "verified_terminal_measurement_closure_evidence"
        ]["attempts"][1]["disposition"] = "ACCEPT_PARTIAL"
        self._assert_rejected(
            "closure_screening_incomplete", observation=observation
        )

    def test_singleton_stage_with_failed_constraints_is_rejected(self):
        observation = self._observation()
        observation["fresh_context_evidence"]["measurement"][
            "verified_terminal_measurement_closure_evidence"
        ]["attempts"][0]["physical_constraints_ok"] = False
        self._assert_rejected(
            "closure_screening_incomplete", observation=observation
        )
