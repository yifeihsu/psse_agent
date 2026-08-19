"""Natural, probe, and combined support must stay separately accountable."""

from __future__ import annotations

import unittest

from psse_env.dagger.replay_buffer import (
    DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA,
    DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS,
    audit_dagger1_training_support,
    dagger1_probe_replay_quota,
)

PFNC = "post_failure_no_candidate"
UCR = "unsupported_correction_recovery"


def _rows(stratum: str, count: int, *, prefix: str, probe: bool = False):
    return [
        {
            "example_id": f"{prefix}-{stratum}-{index}",
            "physical_root_fingerprint": f"{prefix}-{stratum}-root-{index}",
            "recovery_stratum": stratum,
            "production_label_eligible": not probe,
            "state_origin": (
                "observable_recovery_probe" if probe else "learner_policy"
            ),
        }
        for index in range(count)
    ]


def _healthy_natural():
    """Natural rows meeting every gated floor, with the rare pair starved."""
    rows = []
    rows += _rows("multi_measurement_safe_handoff", 10, prefix="nat")
    rows += _rows("sequential_measurement_parameter_recovery", 10, prefix="nat")
    rows += _rows("premature_commit_recovery", 5, prefix="nat")
    rows += _rows("premature_escalation_recovery", 5, prefix="nat")
    return rows


class NaturalFloorTests(unittest.TestCase):
    def test_incidence_dependent_strata_carry_no_natural_floor(self):
        self.assertEqual(
            set(DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS)
            & DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA,
            set(),
        )

    def test_natural_support_passes_with_zero_rare_roots(self):
        """A learner that never makes these mistakes must not fail release.

        These fixtures carry no targeted-state-cell coverage, so the audit's
        overall verdict is not the subject here; the stratum shortfalls are.
        """
        report = audit_dagger1_training_support(_healthy_natural(), [])
        natural = report["natural_on_policy_support"]
        self.assertEqual(natural["recovery_stratum_shortfalls"], {})
        for stratum in (PFNC, UCR):
            entry = natural["recovery_strata"].get(stratum)
            if entry is not None:
                self.assertFalse(entry["required_for_release"])

    def test_a_genuine_natural_gap_still_fails(self):
        """Ungating the rare pair must not ungate anything else."""
        rows = [
            row
            for row in _healthy_natural()
            if row["recovery_stratum"] != "multi_measurement_safe_handoff"
        ]
        report = audit_dagger1_training_support(rows, [])
        shortfalls = report["natural_on_policy_support"][
            "recovery_stratum_shortfalls"
        ]
        self.assertIn("multi_measurement_safe_handoff", shortfalls)
        self.assertFalse(report["passed"])


class ProbeAndCombinedTests(unittest.TestCase):
    def test_probe_floor_is_enforced_at_ten_roots(self):
        probes = _rows(PFNC, 9, prefix="probe", probe=True) + _rows(
            UCR, 12, prefix="probe", probe=True
        )
        report = audit_dagger1_training_support(_healthy_natural(), probes)
        probe_support = report["observable_probe_support"]
        self.assertFalse(probe_support["passed"])
        self.assertEqual(
            probe_support["recovery_strata"][PFNC]["root_shortfall"], 1
        )
        self.assertTrue(probe_support["recovery_strata"][UCR]["passed"])
        self.assertFalse(report["passed"])

    def test_combined_support_counts_distinct_roots_across_sources(self):
        natural = _healthy_natural() + _rows(PFNC, 4, prefix="nat")
        probes = _rows(PFNC, 10, prefix="probe", probe=True) + _rows(
            UCR, 10, prefix="probe", probe=True
        )
        report = audit_dagger1_training_support(natural, probes)
        combined = report["combined_training_support"]["recovery_strata"]
        self.assertEqual(combined[PFNC]["distinct_physical_roots"], 14)
        self.assertEqual(combined[UCR]["distinct_physical_roots"], 10)
        self.assertTrue(report["combined_training_support"]["passed"])
        self.assertTrue(report["observable_probe_support"]["passed"])
        self.assertEqual(
            report["natural_on_policy_support"]["recovery_stratum_shortfalls"], {}
        )

    def test_shared_roots_are_not_double_counted(self):
        """A probe on a root the learner also reached adds no new support."""
        natural = _healthy_natural() + _rows(PFNC, 3, prefix="shared")
        probes = _rows(PFNC, 3, prefix="shared", probe=True) + _rows(
            UCR, 10, prefix="probe", probe=True
        )
        combined = audit_dagger1_training_support(natural, probes)[
            "combined_training_support"
        ]["recovery_strata"]
        self.assertEqual(combined[PFNC]["distinct_physical_roots"], 3)

    def test_natural_incidence_is_reported_even_though_ungated(self):
        natural = _healthy_natural() + _rows(PFNC, 2, prefix="nat")
        report = audit_dagger1_training_support(
            natural, _rows(PFNC, 10, prefix="probe", probe=True)
        )
        incidence = report["natural_incidence_report_only"]
        self.assertEqual(incidence[PFNC]["distinct_physical_roots"], 2)
        self.assertFalse(incidence[PFNC]["gated"])


class ProbeReplayQuotaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probes = _rows(PFNC, 12, prefix="probe", probe=True)

    def test_probe_quota_does_not_draw_on_the_natural_share(self):
        quota = dagger1_probe_replay_quota(
            self.probes, total_size=1000, probe_share=0.02
        )
        self.assertEqual(quota["requested_probe_rows"], 20)
        self.assertEqual(quota["natural_rows_remaining"], 980)
        self.assertEqual(quota["natural_share_remaining"], 0.98)

    def test_probe_capacity_shortfall_is_reported(self):
        quota = dagger1_probe_replay_quota(
            self.probes, total_size=10000, probe_share=0.5
        )
        # 12 roots, one example each, duplicate cap 2 -> capacity 24.
        self.assertEqual(quota["available_probe_rows"], 24)
        self.assertEqual(quota["requested_probe_rows"], 5000)
        self.assertEqual(quota["probe_capacity_shortfall"], 4976)
        self.assertFalse(quota["passed"])

    def test_share_has_no_default_and_is_range_checked(self):
        with self.assertRaises(TypeError):
            dagger1_probe_replay_quota(self.probes, total_size=1000)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                dagger1_probe_replay_quota(
                    self.probes, total_size=1000, probe_share=bad
                )

    def test_malformed_probe_rows_fail_closed(self):
        rows = list(self.probes) + [{"example_id": "", "physical_root_fingerprint": ""}]
        quota = dagger1_probe_replay_quota(
            rows, total_size=1000, probe_share=0.02
        )
        self.assertEqual(quota["malformed_probe_rows"], 1)
        self.assertFalse(quota["passed"])

    def test_empty_probe_set_is_refused(self):
        with self.assertRaises(ValueError):
            dagger1_probe_replay_quota([], total_size=1000, probe_share=0.02)


if __name__ == "__main__":
    unittest.main()
