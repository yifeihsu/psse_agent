"""Probe interventions must land in the stratum they claim, and nowhere else."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    GET_MEASUREMENT_CONTEXT,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_WLS,
)
from psse_env.dagger.recovery_probes import (
    RECOVERY_PROBE_COLLECTION_ROLE,
    RECOVERY_PROBE_DATASET_SOURCE,
    RECOVERY_PROBE_ROOT_FLOORS,
    RECOVERY_PROBE_STATE_ORIGIN,
    audit_recovery_probe_support,
    combined_recovery_support,
    post_correction_confirmation_pending,
    prepare_scenario_envelope,
    probe_audit_scenario,
    post_failure_no_candidate_intervention,
    probe_intervention,
    recovery_probe_manifest,
    stamp_recovery_probe_row,
    verify_probe_stratum,
)
from psse_env.oracle.process_validity import (
    ProcessValidityOracle,
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



_PASS_PROOF = lambda observation, *, preferred_action, expert_actions: {
    "contract": "observable_rank_one_target_v1",
    "passed": True,
    "basis": "test_stub",
}
_PASS_AUDIT = lambda observation, *, preferred_action, env, history, scenario, observable_evidence_passed: {
    "contract": "dagger1_offline_teacher_target_truth_audit_v3",
    "passed": True,
    "action_class": "correct_measurements",
}

class RecoveryProbeInterventionTests(unittest.TestCase):
    def test_post_failure_probe_declines_when_a_candidate_is_open(self):
        self.assertIsNone(
            post_failure_no_candidate_intervention(
                _observation(has_open_candidate=True, candidate_state_id=f"{STATE}x")
            )
        )

    def test_post_failure_probe_binds_an_unresolvable_state(self):
        action = post_failure_no_candidate_intervention(_observation())
        self.assertEqual(action["tool"], GET_MEASUREMENT_CONTEXT)
        state_id = action["arguments"]["state_id"]
        self.assertTrue(state_id.startswith("r0_probe_episode1:"))
        self.assertNotEqual(state_id, STATE)

    def test_intervention_is_deterministic(self):
        for stratum in ("post_failure_no_candidate", "unsupported_correction_recovery"):
            first = probe_intervention(_observation(), stratum=stratum)
            second = probe_intervention(_observation(), stratum=stratum)
            self.assertEqual(first, second, stratum)

    def test_unknown_stratum_is_refused(self):
        with self.assertRaises(ValueError):
            probe_intervention(_observation(), stratum="loop_escape")


class PostCorrectionConfirmationBoundaryTests(unittest.TestCase):
    @staticmethod
    def _state(**updates: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "active_state_id": STATE,
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

    def test_controller_and_probe_share_the_exact_boundary(self):
        oracle = ProcessValidityOracle(executor_hydrated_corrections=True)
        action = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {"state_id": STATE, "suspect_group": [11]},
        }
        cases = (
            self._state(),
            self._state(accepted_corrections=[]),
            self._state(
                unresolved_signatures=[
                    POST_CORRECTION_CONFIRMATION_SIGNATURE,
                    "another_unresolved_signature",
                ]
            ),
            self._state(has_open_candidate=True),
            self._state(has_unverified_candidate=True),
            self._state(has_verified_candidate=True),
        )
        for state in cases:
            with self.subTest(state=state):
                canonical = post_correction_confirmation_required(state)
                probe = post_correction_confirmation_pending(state)
                controller = oracle.check(state, action)
                self.assertEqual(probe, canonical)
                self.assertEqual(
                    controller.get("error_code")
                    == "post_correction_confirmation_required",
                    canonical,
                )


class RecoveryProbeStratumVerificationTests(unittest.TestCase):
    """The classifier decides what a row is; the generator only proposes."""

    def test_unsupported_correction_probe_reaches_its_stratum(self):
        observation = _after_failure(
            CORRECT_MEASUREMENTS, "correction_not_supported_by_current_context"
        )
        result = verify_probe_stratum(
            observation,
            preferred_action={"tool": RUN_WLS, "arguments": {"state_id": STATE}},
            state_class="invalid_precondition_recovery",
            scenario_family="measurement",
            error_cardinality=1,
            expected_stratum="unsupported_correction_recovery",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["actual_stratum"], "unsupported_correction_recovery")

    def test_post_failure_probe_reaches_its_stratum(self):
        observation = _after_failure(GET_MEASUREMENT_CONTEXT, "unknown_state_id")
        result = verify_probe_stratum(
            observation,
            preferred_action={"tool": RUN_WLS, "arguments": {"state_id": STATE}},
            state_class="invalid_precondition_recovery",
            scenario_family="measurement",
            error_cardinality=1,
            expected_stratum="post_failure_no_candidate",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["actual_stratum"], "post_failure_no_candidate")

    def test_probe_landing_in_a_neighbouring_stratum_is_rejected(self):
        """A multi-measurement root whose expert escalates is not a probe row.

        The safe-handoff branch is ordered ahead of post_failure_no_candidate,
        so this state classifies as multi_measurement_safe_handoff. The reserve
        margin above the floor exists for exactly this discard.
        """
        observation = _after_failure(GET_MEASUREMENT_CONTEXT, "unknown_state_id")
        result = verify_probe_stratum(
            observation,
            preferred_action={
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {"request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST},
            },
            state_class="terminal_operator_escalation",
            scenario_family="multi_measurement",
            error_cardinality=3,
            expected_stratum="post_failure_no_candidate",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["actual_stratum"], "multi_measurement_safe_handoff")

    def test_a_probe_that_did_not_fail_is_rejected(self):
        observation = _observation(
            last_tool=GET_MEASUREMENT_CONTEXT,
            last_tool_status="success",
            last_tool_output={"execution_status": "success", "error_code": None},
        )
        result = verify_probe_stratum(
            observation,
            preferred_action={"tool": RUN_WLS, "arguments": {"state_id": STATE}},
            state_class="clean_successful",
            scenario_family="measurement",
            error_cardinality=1,
            expected_stratum="post_failure_no_candidate",
        )
        self.assertFalse(result["passed"])


def _probe_row(stratum: str, root: str) -> dict[str, Any]:
    verification = {"passed": True, "expected_stratum": stratum, "actual_stratum": stratum}
    return stamp_recovery_probe_row(
        {"example_id": f"{stratum}-{root}", "physical_root_fingerprint": root},
        intervention={"tool": GET_MEASUREMENT_CONTEXT, "arguments": {}},
        expected_stratum=stratum,
        verification=verification,
        rank_one_proof={"passed": True},
        teacher_target_audit={"passed": True},
        training_decision_evidence_verified=True,
    )


class RecoveryProbeRowTests(unittest.TestCase):
    def test_stamped_row_is_auxiliary_and_never_learner_visited(self):
        row = _probe_row("post_failure_no_candidate", "root-a")
        self.assertEqual(row["state_origin"], RECOVERY_PROBE_STATE_ORIGIN)
        self.assertEqual(row["dataset_source"], RECOVERY_PROBE_DATASET_SOURCE)
        self.assertEqual(row["collection_role"], RECOVERY_PROBE_COLLECTION_ROLE)
        self.assertNotEqual(row["state_origin"], "learner_policy")
        # A probe row may train, but may not satisfy a natural on-policy floor.
        self.assertIs(row["production_label_eligible"], False)
        self.assertIs(row["natural_on_policy_support_eligible"], False)

    def test_unverified_row_cannot_be_stamped(self):
        with self.assertRaises(ValueError):
            stamp_recovery_probe_row(
                {"example_id": "x", "physical_root_fingerprint": "root-a"},
                intervention={"tool": GET_MEASUREMENT_CONTEXT, "arguments": {}},
                expected_stratum="post_failure_no_candidate",
                verification={"passed": False, "actual_stratum": "loop_escape"},
            )


class RecoveryProbeSupportTests(unittest.TestCase):
    @staticmethod
    def _suite(counts: dict[str, int]) -> list[dict[str, Any]]:
        return [
            _probe_row(stratum, f"{stratum}-root-{index}")
            for stratum, count in counts.items()
            for index in range(count)
        ]

    def test_meeting_both_floors_passes(self):
        report = audit_recovery_probe_support(
            self._suite({s: 10 for s in RECOVERY_PROBE_ROOT_FLOORS})
        )
        self.assertTrue(report["passed"], report)

    def test_shortfall_is_reported_per_stratum(self):
        report = audit_recovery_probe_support(
            self._suite(
                {"post_failure_no_candidate": 10, "unsupported_correction_recovery": 7}
            )
        )
        self.assertFalse(report["passed"])
        strata = report["probe_strata"]
        self.assertEqual(
            strata["unsupported_correction_recovery"]["root_shortfall"], 3
        )
        self.assertEqual(strata["post_failure_no_candidate"]["root_shortfall"], 0)

    def test_one_root_cannot_manufacture_support_with_repeat_rows(self):
        rows = self._suite({s: 10 for s in RECOVERY_PROBE_ROOT_FLOORS})
        rows.append(_probe_row("post_failure_no_candidate", "post_failure_no_candidate-root-0"))
        report = audit_recovery_probe_support(rows)
        self.assertFalse(report["passed"])
        self.assertTrue(report["roots_with_multiple_rows"])

    def test_natural_rows_cannot_be_counted_as_probe_support(self):
        rows = self._suite({s: 10 for s in RECOVERY_PROBE_ROOT_FLOORS})
        rows.append(
            {
                "example_id": "natural-1",
                "physical_root_fingerprint": "natural-root",
                "dataset_source": "dagger_rollout",
                "state_origin": "learner_policy",
                "recovery_stratum": "post_failure_no_candidate",
            }
        )
        report = audit_recovery_probe_support(rows)
        self.assertEqual(report["foreign_rows"], 1)
        self.assertFalse(report["passed"])


class RecoveryProbeManifestTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, Any]]:
        return [
            _probe_row(stratum, f"{stratum}-root-{index}")
            for stratum in RECOVERY_PROBE_ROOT_FLOORS
            for index in range(10)
        ]

    def test_manifest_binds_provenance_and_passes_when_disjoint(self):
        manifest = recovery_probe_manifest(
            self._rows(),
            generator_identity="recovery_probe_generator_v1",
            source_commit="a" * 40,
        )
        self.assertTrue(manifest["passed"], manifest)
        self.assertEqual(manifest["source_commit"], "a" * 40)
        self.assertEqual(manifest["distinct_physical_roots"], 20)
        self.assertIs(manifest["natural_on_policy_support_eligible"], False)

    def test_development_holdout_overlap_fails_the_manifest(self):
        manifest = recovery_probe_manifest(
            self._rows(),
            generator_identity="g",
            source_commit="b" * 40,
            development_roots=["post_failure_no_candidate-root-3"],
        )
        self.assertFalse(manifest["passed"])
        self.assertEqual(
            manifest["root_disjointness"]["development_holdout_overlap"],
            ["post_failure_no_candidate-root-3"],
        )

    def test_frozen_evaluation_overlap_fails_the_manifest(self):
        manifest = recovery_probe_manifest(
            self._rows(),
            generator_identity="g",
            source_commit="c" * 40,
            frozen_evaluation_roots=["unsupported_correction_recovery-root-1"],
        )
        self.assertFalse(manifest["passed"])

    def test_natural_overlap_is_reported_but_not_fatal(self):
        manifest = recovery_probe_manifest(
            self._rows(),
            generator_identity="g",
            source_commit="d" * 40,
            natural_roots=["post_failure_no_candidate-root-0"],
        )
        self.assertTrue(manifest["passed"], manifest)
        self.assertEqual(
            manifest["root_disjointness"]["natural_dagger_overlap"],
            ["post_failure_no_candidate-root-0"],
        )


class CombinedRecoverySupportTests(unittest.TestCase):
    """Combined support must deduplicate roots shared by the two sources."""

    @staticmethod
    def _natural(stratum, count, *, prefix="nat"):
        return [
            {
                "example_id": f"{prefix}-{stratum}-{i}",
                "physical_root_fingerprint": f"physical_v3_{prefix}-{stratum}-{i}",
                "recovery_stratum": stratum,
                "production_label_eligible": True,
            }
            for i in range(count)
        ]

    def test_disjoint_sources_add_up(self):
        natural = self._natural("post_failure_no_candidate", 3)
        probes = [
            _probe_row("post_failure_no_candidate", f"physical_v3_probe-{i}")
            for i in range(10)
        ]
        report = combined_recovery_support(natural, probes)
        combined = report["combined_training_support"]["recovery_strata"]
        self.assertEqual(
            combined["post_failure_no_candidate"]["distinct_physical_roots"], 13
        )

    def test_shared_roots_are_counted_once(self):
        """The additive helper reported 13 here; the real answer is 10.

        Every pilot probe root coincided with a natural support root, so an
        additive combined figure overcounted every one of them.
        """
        shared = [f"physical_v3_shared-{i}" for i in range(3)]
        natural = [
            {
                "example_id": f"nat-{i}",
                "physical_root_fingerprint": root,
                "recovery_stratum": "post_failure_no_candidate",
                "production_label_eligible": True,
            }
            for i, root in enumerate(shared)
        ]
        probes = [
            _probe_row("post_failure_no_candidate", root) for root in shared
        ] + [
            _probe_row("post_failure_no_candidate", f"physical_v3_probe-{i}")
            for i in range(7)
        ]
        report = combined_recovery_support(natural, probes)
        combined = report["combined_training_support"]["recovery_strata"]
        self.assertEqual(
            combined["post_failure_no_candidate"]["distinct_physical_roots"], 10
        )

    def test_natural_figure_is_never_disguised_by_probes(self):
        natural = self._natural("post_failure_no_candidate", 3)
        probes = [
            _probe_row("post_failure_no_candidate", f"physical_v3_probe-{i}")
            for i in range(10)
        ]
        report = combined_recovery_support(natural, probes)
        incidence = report["natural_incidence_report_only"]
        self.assertEqual(
            incidence["post_failure_no_candidate"]["distinct_physical_roots"], 3
        )
        self.assertFalse(incidence["post_failure_no_candidate"]["gated"])

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


class RecoveryProbeGeneratorTests(unittest.TestCase):
    @staticmethod
    def _state_class(observation, preferred_action):
        del observation, preferred_action
        return "invalid_precondition_recovery"

    def test_generator_fills_a_quota_and_stops(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        rows, report = generate_recovery_probes(
            _scenarios(20),
            env=_FakeEnv(),
            expert_oracle=_FakeOracle(),
            rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=self._state_class,
            quotas={"post_failure_no_candidate": 4},
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(report["quota_met"]["post_failure_no_candidate"])
        self.assertEqual(report["roots_admitted"]["post_failure_no_candidate"], 4)
        # One row per physical root, and every row carries the auxiliary identity.
        roots = [row["physical_root_fingerprint"] for row in rows]
        self.assertEqual(len(roots), len(set(roots)))
        for row in rows:
            self.assertEqual(row["state_origin"], RECOVERY_PROBE_STATE_ORIGIN)
            self.assertEqual(row["recovery_stratum"], "post_failure_no_candidate")

    def test_generator_preserves_grouping_for_a_bare_scenario(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        scenario = {
            "physical_root_fingerprint": "probe-root-bare",
            "scenario_family": "measurement",
            "error_cardinality": 1,
            "scenario_id": "scenario-bare",
        }
        env = _FakeEnv()
        rows, report = generate_recovery_probes(
            [scenario],
            env=env,
            expert_oracle=_FakeOracle(),
            rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=self._state_class,
            quotas={"post_failure_no_candidate": 1},
        )

        self.assertEqual(len(rows), 1, report["skipped"])
        self.assertEqual(rows[0]["physical_root_fingerprint"], "probe-root-bare")
        self.assertEqual(rows[0]["scenario_family"], "measurement")
        self.assertEqual(rows[0]["scenario_id"], "scenario-bare")
        self.assertEqual(env.scenario, scenario)

    def test_rows_landing_in_a_neighbouring_stratum_are_discarded(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        # A multi-measurement root whose expert escalates classifies as
        # multi_measurement_safe_handoff, which is ordered ahead of the target.
        rows, report = generate_recovery_probes(
            _scenarios(6, family="multi_measurement", cardinality=3),
            env=_FakeEnv(),
            expert_oracle=_FakeOracle(
                {
                    "tool": ASK_FOR_MORE_EVIDENCE,
                    "arguments": {"request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST},
                }
            ),
            rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=self._state_class,
            quotas={"post_failure_no_candidate": 4},
        )
        self.assertEqual(rows, [])
        self.assertFalse(report["quota_met"]["post_failure_no_candidate"])
        self.assertTrue(
            any("landed_in_" in reason for reason in report["skipped"]),
            report["skipped"],
        )
        self.assertTrue(all(not a["admitted"] for a in report["attempts"]))

    def test_unsupported_correction_quota_uses_its_own_error_path(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        rows, report = generate_recovery_probes(
            _scenarios(5),
            env=_FakeEnv(
                "post_correction_confirmation_required", confirmation_pending=True
            ),
            expert_oracle=_FakeOracle(),
            rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=self._state_class,
            quotas={"unsupported_correction_recovery": 3},
        )
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(
                row["recovery_stratum"], "unsupported_correction_recovery"
            )
            self.assertEqual(row["probe_intervention"]["tool"], CORRECT_MEASUREMENTS)

    def test_unsupported_strata_are_refused(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        with self.assertRaises(ValueError):
            generate_recovery_probes(
                _scenarios(2),
                env=_FakeEnv(),
                expert_oracle=_FakeOracle(),
                rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=self._state_class,
                quotas={"loop_escape": 2},
            )


class ProbeAdmissionParityTests(unittest.TestCase):
    """An unaudited probe row must be impossible to stamp.

    Probe targets are held to the natural DAgger admission standard. Each of
    these fields is a step that was actually executed, not a caller assertion.
    """

    def _stamp(self, **overrides):
        args = {
            "intervention": {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {}},
            "expected_stratum": "post_failure_no_candidate",
            "verification": {
                "passed": True,
                "actual_stratum": "post_failure_no_candidate",
            },
            "rank_one_proof": {"passed": True},
            "teacher_target_audit": {"passed": True},
            "training_decision_evidence_verified": True,
        }
        args.update(overrides)
        return stamp_recovery_probe_row(
            {
                "example_id": "probe-a",
                "physical_root_fingerprint": "physical_v3_root-a",
            },
            **args,
        )

    def test_a_fully_audited_row_is_stamped(self):
        row = self._stamp()
        self.assertTrue(row["auxiliary_training_eligible"])
        self.assertFalse(row["production_label_eligible"])
        self.assertFalse(row["natural_on_policy_support_eligible"])
        self.assertTrue(row["training_decision_evidence_verified"])

    def test_unverified_training_decision_evidence_is_refused(self):
        with self.assertRaises(ValueError):
            self._stamp(training_decision_evidence_verified=False)

    def test_failed_rank_one_proof_is_refused(self):
        with self.assertRaises(ValueError):
            self._stamp(rank_one_proof={"passed": False, "reason": "not_first"})
        with self.assertRaises(ValueError):
            self._stamp(rank_one_proof=None)

    def test_quarantined_teacher_target_is_refused(self):
        with self.assertRaises(ValueError):
            self._stamp(
                teacher_target_audit={"passed": False, "reason_codes": ["x"]}
            )
        with self.assertRaises(ValueError):
            self._stamp(teacher_target_audit=None)

    def test_failed_stratum_verification_is_refused(self):
        with self.assertRaises(ValueError):
            self._stamp(
                verification={"passed": False, "actual_stratum": "loop_escape"}
            )


class ProbeAuditScenarioTests(unittest.TestCase):
    def test_private_truth_reaches_the_audit_scenario(self):
        envelope = {
            "execution": {"scenario_id": "s1", "case": {"x": 1}},
            "audit": {"truth": {"truth_complete": True, "true_measurement_errors": [{"index": 7}]}},
            "grouping": {"physical_root_fingerprint": "physical_v3_root-a"},
        }
        scenario = probe_audit_scenario(envelope)
        # The runtime envelope alone is truth-free; the audit needs the ledger.
        self.assertNotIn("true_measurement_errors", envelope["execution"])
        self.assertTrue(scenario["truth_complete"])
        self.assertEqual(scenario["true_measurement_errors"], [{"index": 7}])
        self.assertEqual(scenario["scenario_id"], "s1")
        self.assertEqual(
            scenario["physical_root_fingerprint"], "physical_v3_root-a"
        )

    def test_bare_scenario_is_reused_for_runtime_audit_and_grouping(self):
        bare = {
            "scenario_id": "bare-s1",
            "physical_root_fingerprint": "physical_v3_root-bare",
            "scenario_family": "measurement",
            "error_cardinality": 1,
        }

        prepared = prepare_scenario_envelope(bare)

        self.assertEqual(prepared["runtime"], bare)
        self.assertEqual(prepared["audit"], bare)
        self.assertEqual(prepared["grouping"], bare)
        self.assertIsNot(prepared["runtime"], bare)
        self.assertIsNot(prepared["grouping"], bare)


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


class ConfirmationPrefixIntegrationTests(unittest.TestCase):
    def test_prefix_reaches_confirmation_only_via_the_shared_selector(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        env = _VerifiedCandidateEnv()

        class _StallingOracle:
            """The raw rule expert: silent at a verified candidate."""

            def next_actions(self, state, history=None):
                del history
                lifecycle = (
                    state.get("candidate_lifecycle")
                    if isinstance(state, Mapping)
                    else None
                )
                if lifecycle == "VERIFIED_CANDIDATE":
                    return []
                return [
                    {
                        "tool": GET_MEASUREMENT_CONTEXT,
                        "arguments": {"state_id": state.get("active_state_id")},
                    }
                ]

        rows, report = generate_recovery_probes(
            _scenarios(1),
            env=env,
            expert_oracle=_StallingOracle(),
            rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=lambda observation, preferred: "invalid_precondition_recovery",
            quotas={"unsupported_correction_recovery": 1},
        )
        self.assertEqual(len(rows), 1, report["skipped"])
        row = rows[0]
        self.assertEqual(
            row["recovery_stratum"], "unsupported_correction_recovery"
        )
        self.assertEqual(
            row["preferred_action"]["tool"], GET_MEASUREMENT_CONTEXT
        )
        # The prefix had to dispose of the verified candidate to get here. The
        # stalling oracle returns nothing at that state and otherwise only ever
        # offers get_measurement_context, so a disposition action in the prefix
        # can only have come from the shared selector's reconstruction.
        self.assertTrue(
            any(
                a["tool"] in {"commit_state", "rollback_state", ASK_FOR_MORE_EVIDENCE}
                for a in row["probe_setup_actions"]
            ),
            row["probe_setup_actions"],
        )
        # And the evidence assertion really ran.
        self.assertTrue(env.evidence_calls)

    def test_absent_evidence_assertion_fails_closed(self):
        from psse_env.dagger.recovery_probes import generate_recovery_probes

        class _NoEvidenceEnv(_VerifiedCandidateEnv):
            assert_training_decision_evidence = None

        env = _NoEvidenceEnv()

        rows, report = generate_recovery_probes(
            _scenarios(1),
            env=env,
            expert_oracle=_FakeOracle(),
            rank_one_proof_for=_PASS_PROOF,
            teacher_target_audit_for=_PASS_AUDIT,
            state_class_for=lambda observation, preferred: "invalid_precondition_recovery",
            quotas={"unsupported_correction_recovery": 1},
        )
        self.assertEqual(rows, [])
        self.assertTrue(
            any(
                "training_decision_evidence_unavailable" in reason
                for reason in report["skipped"]
            ),
            report["skipped"],
        )
