"""Probe interventions must land in the stratum they claim, and nowhere else."""

from __future__ import annotations

import unittest
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    GET_MEASUREMENT_CONTEXT,
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
    post_failure_no_candidate_intervention,
    probe_intervention,
    recovery_probe_manifest,
    stamp_recovery_probe_row,
    unsupported_correction_intervention,
    verify_probe_stratum,
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


class RecoveryProbeInterventionTests(unittest.TestCase):
    def test_unsupported_correction_groups_two_visible_findings(self):
        # A fresh context publishes one supported singleton per finding, so no
        # single finding is ever unsupported; the grouped action is.
        action = unsupported_correction_intervention(_observation())
        self.assertEqual(
            action,
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": STATE, "suspect_group": [23, 41]},
            },
        )

    def test_unsupported_correction_declines_when_the_group_is_supported(self):
        observation = _observation()
        observation["fresh_context_evidence"]["measurement"][
            "supported_corrections"
        ] = [
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": STATE, "suspect_group": [23, 41]},
            }
        ]
        self.assertIsNone(unsupported_correction_intervention(observation))

    def test_unsupported_correction_declines_with_a_single_finding(self):
        observation = _observation()
        observation["fresh_context_evidence"]["measurement"][
            "measurement_findings"
        ] = [{"channel": "Pinj", "index0": 23, "value": 4.39}]
        self.assertIsNone(unsupported_correction_intervention(observation))

    def test_unsupported_correction_declines_on_stale_context(self):
        observation = _observation()
        observation["fresh_context_evidence"]["measurement"]["state_id"] = "other:s1"
        self.assertIsNone(unsupported_correction_intervention(observation))

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
    def test_natural_and_probe_support_stay_separable(self):
        natural = {
            "recovery_strata": {
                "post_failure_no_candidate": {
                    "distinct_physical_roots": 3,
                    "minimum_distinct_physical_roots": 10,
                    "passed": False,
                },
                "unsupported_correction_recovery": {
                    "distinct_physical_roots": 3,
                    "minimum_distinct_physical_roots": 10,
                    "passed": False,
                },
                "multi_measurement_safe_handoff": {
                    "distinct_physical_roots": 70,
                    "minimum_distinct_physical_roots": 10,
                    "passed": True,
                },
            }
        }
        probes = audit_recovery_probe_support(
            [
                _probe_row(stratum, f"{stratum}-root-{index}")
                for stratum in RECOVERY_PROBE_ROOT_FLOORS
                for index in range(10)
            ]
        )
        report = combined_recovery_support(natural, probes)
        combined = report["combined_training_support"]

        # The natural figure survives untouched: a probe cannot disguise it.
        pfnc = combined["post_failure_no_candidate"]
        self.assertEqual(pfnc["natural_distinct_physical_roots"], 3)
        self.assertEqual(pfnc["probe_distinct_physical_roots"], 10)
        self.assertEqual(pfnc["combined_distinct_physical_roots"], 13)
        self.assertTrue(pfnc["natural_floor_is_report_only"])
        self.assertTrue(pfnc["passed"])

        # A stratum with no probe source keeps its natural floor as binding.
        handoff = combined["multi_measurement_safe_handoff"]
        self.assertFalse(handoff["probe_eligible_stratum"])
        self.assertFalse(handoff["natural_floor_is_report_only"])
        self.assertEqual(handoff["probe_distinct_physical_roots"], 0)
        self.assertTrue(report["passed"], report)

    def test_probe_shortfall_still_fails_the_combined_report(self):
        natural = {
            "recovery_strata": {
                "post_failure_no_candidate": {
                    "distinct_physical_roots": 3,
                    "passed": False,
                }
            }
        }
        probes = audit_recovery_probe_support(
            [
                _probe_row("post_failure_no_candidate", f"p-root-{index}")
                for index in range(4)
            ]
        )
        report = combined_recovery_support(natural, probes)
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["combined_training_support"]["post_failure_no_candidate"][
                "combined_distinct_physical_roots"
            ],
            7,
        )


if __name__ == "__main__":
    unittest.main()


class _FakeEnv:
    """Minimal environment: the intervention always fails as designed."""

    def __init__(self, error_code: str = "unknown_state_id"):
        self.error_code = error_code
        self.scenario: Any = None
        self.resets = 0

    def reset(self, scenario):
        self.scenario = scenario
        self.resets += 1

    def get_policy_observation(self, history):
        observation = _observation()
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
            env=_FakeEnv("correction_not_supported_by_current_context"),
            expert_oracle=_FakeOracle(),
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
                state_class_for=self._state_class,
                quotas={"loop_escape": 2},
            )
