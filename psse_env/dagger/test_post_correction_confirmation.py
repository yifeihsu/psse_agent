"""Probe interventions must land in the stratum they claim, and nowhere else."""

from __future__ import annotations

import unittest
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RUN_WLS,
)
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
    validate_offline_teacher_target_audit_metadata,
)
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE
from psse_env.oracle.process_validity import (
    post_correction_confirmation_required,
)

STATE = "r0_probe_episode1:s2"


def _observation(**overrides: Any) -> dict[str, Any]:
    """A pre-intervention state: context is fresh, no candidate is open."""
    observation: dict[str, Any] = {
        "active_state_id": STATE,
        "has_open_candidate": False,
        "candidate_state_id": None,
        "tried_action_signatures": [],
        "accepted_corrections": [],
        "fresh_context_evidence": {
            "measurement": {
                "state_id": STATE,
                "state_hash": "abc123",
                "measurement_findings": [
                    {"channel": "Pinj", "index0": 23, "value": 4.39},
                    {"channel": "Pt", "index0": 41, "value": 2.11},
                ],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {"state_id": STATE, "suspect_group": [41]},
                    }
                ],
            }
        },
    }
    observation.update(overrides)
    return observation


def _after_failure(tool: str, error_code: str, **overrides: Any) -> dict[str, Any]:
    """The real post-intervention observation shape the environment returns."""
    observation = _observation(
        last_tool=tool,
        last_tool_status="failure",
        last_tool_output={"execution_status": "failure", "error_code": error_code},
    )
    observation.update(overrides)
    return observation


def _passed_rank_one_proof(
    observation, *, preferred_action, expert_actions
):
    del observation, preferred_action, expert_actions
    return {
        "contract": "observable_rank_one_target_v1",
        "passed": True,
        "basis": "test_stub",
    }


_PASS_PROOF = _passed_rank_one_proof


def _passed_current_audit(
    observation,
    *,
    preferred_action,
    env,
    history,
    scenario,
    observable_evidence_passed,
):
    del (
        observation,
        preferred_action,
        env,
        history,
        scenario,
        observable_evidence_passed,
    )
    return validate_offline_teacher_target_audit_metadata(
        {
            "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
            "passed": True,
            "action_class": "read_only",
            "checks": {"observable_evidence_gate_passed": True},
            "reason_codes": [],
        },
        require_passed=True,
    )


_PASS_AUDIT = _passed_current_audit


class PostCorrectionConfirmationBoundaryTests(unittest.TestCase):
    @staticmethod
    def _state(**updates: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "active_state_id": STATE,
            # Pre-seeded post-correction evidence without a bound WLS ledger is
            # the historical fixture shape; the strict profiles demand WLS first.
            "evidence_profile": AUXILIARY_EVIDENCE_PROFILE,
            "accepted_corrections": [{"source_action": {"tool": CORRECT_MEASUREMENTS}}],
            "unresolved_signatures": [POST_CORRECTION_CONFIRMATION_SIGNATURE],
            "has_open_candidate": False,
            "has_unverified_candidate": False,
            "has_verified_candidate": False,
        }
        state.update(updates)
        return state

    def test_exact_singleton_signature_after_acceptance_requires_confirmation(self):
        self.assertTrue(post_correction_confirmation_required(self._state()))

    def test_an_extra_signature_does_not_match_the_confirmation_boundary(self):
        self.assertFalse(
            post_correction_confirmation_required(
                self._state(
                    unresolved_signatures=[
                        POST_CORRECTION_CONFIRMATION_SIGNATURE,
                        "another_unresolved_signature",
                    ]
                )
            )
        )

    def test_every_open_candidate_flag_precedes_the_confirmation_boundary(self):
        for flag in (
            "has_open_candidate",
            "has_unverified_candidate",
            "has_verified_candidate",
        ):
            with self.subTest(flag=flag):
                self.assertFalse(
                    post_correction_confirmation_required(self._state(**{flag: True}))
                )


class _FakeEnv:
    """Minimal environment: the intervention always fails as designed."""

    def __init__(
        self, error_code: str = "unknown_state_id", *, confirmation_pending=False
    ):
        self.error_code = error_code
        self.scenario: Any = None
        self.resets = 0
        # The confirmation-violation intervention fires only when an accepted
        # correction exists and the confirmation signature is the sole
        # unresolved signature -- the controller's exact guard condition.
        self.confirmation_pending = confirmation_pending

    def reset(self, scenario):
        self.scenario = scenario
        self.resets += 1

    def assert_training_decision_evidence(self, action):
        """Attest the evidence, as the production environment does."""
        del action

    def get_oracle_state(self, history):
        del history
        return {"hidden_truth": {}}

    def get_policy_observation(self, history):
        observation = _observation()
        if self.confirmation_pending:
            from psse_env.actions import POST_CORRECTION_CONFIRMATION_SIGNATURE

            observation.update(
                {
                    "accepted_corrections": [
                        {
                            "source_action": {
                                "tool": "correct_measurements",
                                "arguments": {
                                    "state_id": observation["active_state_id"],
                                    "suspect_group": [11],
                                },
                            }
                        }
                    ],
                    "unresolved_signatures": [
                        POST_CORRECTION_CONFIRMATION_SIGNATURE
                    ],
                }
            )
        if history:
            action = history[-1]["action"]
            observation.update(
                {
                    "last_tool": action["tool"],
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": self.error_code,
                    },
                }
            )
        return observation

    def step(self, action):
        return None, {"execution_status": "failure", "error_code": self.error_code}


class _FakeOracle:
    def __init__(self, action=None):
        self.action = action or {"tool": RUN_WLS, "arguments": {"state_id": STATE}}

    def next_actions(self, observation, history):
        del observation, history
        return [self.action]


def _scenarios(count: int, family: str = "measurement", cardinality: int = 1):
    return [
        {
            "grouping": {
                "physical_root_fingerprint": f"probe-root-{index}",
                "scenario_family": family,
                "error_cardinality": cardinality,
                "scenario_id": f"scenario-{index}",
            }
        }
        for index in range(count)
    ]


class _VerifiedCandidateEnv:
    """Reaches the confirmation state only through a verified candidate.

    The raw rule expert returns nothing at a verified candidate, so a generator
    that calls it directly stalls here and never reaches the confirmation
    boundary. This fixture therefore fails unless the shared selector's
    commit/rollback reconstruction is in the path -- it pins the architecture,
    not just the outcome.
    """

    def __init__(self):
        self.scenario = None
        self.stage = "unverified"
        self.evidence_calls = []

    def reset(self, scenario):
        self.scenario = scenario
        self.stage = "unverified"

    def assert_training_decision_evidence(self, action):
        self.evidence_calls.append(action)

    def get_oracle_state(self, history):
        return {"hidden_truth": {}}

    def get_policy_observation(self, history):
        from psse_env.actions import POST_CORRECTION_CONFIRMATION_SIGNATURE

        base = {
            "active_state_id": "ep:s1",
            "history_window": [],
            "accepted_corrections": [],
            "unresolved_signatures": [],
        }
        if self.stage == "unverified":
            base.update(
                {
                    "candidate_state_id": "ep:c1",
                    "has_open_candidate": True,
                    "has_verified_candidate": True,
                    "candidate_lifecycle": "VERIFIED_CANDIDATE",
                    "candidate_status": "verified",
                }
            )
        elif self.stage == "confirmation":
            base.update(
                {
                    "accepted_corrections": [
                        {
                            "source_action": {
                                "tool": "correct_measurements",
                                "arguments": {
                                    "state_id": "ep:s1",
                                    "suspect_group": [4],
                                },
                            }
                        }
                    ],
                    "unresolved_signatures": [
                        POST_CORRECTION_CONFIRMATION_SIGNATURE
                    ],
                }
            )
        else:
            base.update(
                {
                    "accepted_corrections": [
                        {
                            "source_action": {
                                "tool": "correct_measurements",
                                "arguments": {
                                    "state_id": "ep:s1",
                                    "suspect_group": [4],
                                },
                            }
                        }
                    ],
                    "unresolved_signatures": [
                        POST_CORRECTION_CONFIRMATION_SIGNATURE
                    ],
                    "last_tool": "correct_measurements",
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": "post_correction_confirmation_required",
                    },
                }
            )
        return base

    def step(self, action):
        # Any observable disposition action closes the candidate; only the
        # reconstruction can produce one here.
        if action.get("tool") in {"commit_state", "rollback_state", ASK_FOR_MORE_EVIDENCE}:
            self.stage = "confirmation"
            return None, {"execution_status": "success"}
        self.stage = "post_intervention"
        return None, {
            "execution_status": "failure",
            "error_code": "post_correction_confirmation_required",
        }
