"""The emitted view must honour the frozen allocation and the union caps."""

from __future__ import annotations

import copy
import unittest

from psse_env.dagger.round1_view_policy import (
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_view_policy_digest,
)
from psse_env.dagger.three_source_view import (
    FINAL_VIEW_SUPPORT_CONTRACT,
    audit_dagger1_final_view_support,
    build_dagger1_three_source_view,
)

PFNC = "post_failure_no_candidate"
UCR = "unsupported_correction_recovery"


def _rows(prefix: str, count: int, *, stratum: str | None = None, root=None):
    return [
        {
            "example_id": f"{prefix}-{index}",
            "physical_root_fingerprint": (
                root if root is not None else f"physical_v3_{prefix}-root-{index}"
            ),
            "recovery_stratum": stratum,
        }
        for index in range(count)
    ]


def _natural_rows():
    """Thirty independent roots satisfying every canonical natural floor."""

    rows = _rows("nat", 30)
    strata = (
        ["multi_measurement_safe_handoff"] * 10
        + ["sequential_measurement_parameter_recovery"] * 10
        + ["premature_commit_recovery"] * 5
        + ["premature_escalation_recovery"] * 5
    )
    for index, (row, stratum) in enumerate(zip(rows, strata, strict=True)):
        row["recovery_stratum"] = stratum
        row["preferred_action"] = {"tool": "run_wls", "args": {}}
        row["policy_observation"] = {}
        if index < 15:
            route = ("actionable", "complete_negative", "unavailable")[index // 5]
            row.update(
                {
                    "scenario_family": "multi_measurement",
                    "error_cardinality": (2, 4, 5)[index // 5],
                    "parameter_scans_available": False,
                }
            )
            row["policy_observation"] = {
                "fresh_context_evidence": {
                    "parameter": {
                        "route_status": route,
                        "parameter_ranking_dominance_ratio": 1.1,
                    }
                }
            }
        elif index < 20:
            row["scenario_family"] = "measurement+parameter"
            row["preferred_action"] = {"tool": "correct_parameters", "args": {}}
            row["policy_observation"] = {
                "history_window": [
                    {"action": {"tool": "correct_measurements", "args": {}}}
                ]
            }
        elif index < 25:
            row["scenario_family"] = "measurement+parameter"
            row["preferred_action"] = {"tool": "correct_measurements", "args": {}}
            row["policy_observation"] = {
                "history_window": [
                    {"action": {"tool": "correct_parameters", "args": {}}}
                ]
            }
        else:
            row["policy_observation"] = {
                "accepted_corrections": [{"tool": "correct_measurements"}],
                "no_material_anomaly_remaining": False,
            }
    return rows


def _small_policy(**overrides):
    policy = copy.deepcopy(ROUND1_THREE_SOURCE_VIEW_POLICY)
    policy["total_rows"] = 100
    policy["allocation"] = {
        "d0_bc0_rows": 42,
        "natural_d1_rows": 30,
        "observable_recovery_probe_rows": 28,
    }
    policy["probe_bucket"] = {
        PFNC: 14,
        UCR: 14,
        "distinct_roots_retained_per_stratum": 12,
        "duplicate_placements_per_stratum": 2,
    }
    policy["probe_floor_distinct_roots"] = 10
    policy.update(overrides)
    return policy


class AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _small_policy()
        self.d0 = _rows("d0", 42)
        self.natural = _natural_rows()
        self.probes = _rows("pfnc", 12, stratum=PFNC) + _rows(
            "ucr", 12, stratum=UCR
        )

    def _build(self, **kwargs):
        args = {
            "d0_rows": self.d0,
            "natural_d1_rows": self.natural,
            "probe_rows": self.probes,
            "policy": self.policy,
        }
        args.update(kwargs)
        return build_dagger1_three_source_view(**args)

    def test_exact_allocation_is_placed(self):
        rows, report = self._build()
        self.assertTrue(report["passed"], report["shortfalls"])
        self.assertEqual(len(rows), 100)
        self.assertEqual(report["placed"]["d0_bc0_rows"], 42)
        self.assertEqual(report["placed"]["natural_d1_rows"], 30)
        self.assertEqual(report["placed"]["observable_recovery_probe_rows"], 28)

    def test_every_distinct_probe_root_is_retained_before_any_duplicate(self):
        _, report = self._build()
        for stratum in (PFNC, UCR):
            entry = report["probe_bucket"][stratum]
            self.assertEqual(entry["distinct_roots_retained"], 12)
            self.assertEqual(entry["duplicate_placements"], 2)
            self.assertEqual(entry["shortfall"], 0)

    def test_final_view_support_reaudits_placed_rows_and_probe_sources(self):
        _, report = self._build()
        support = report["final_view_support"]
        self.assertEqual(support["contract"], FINAL_VIEW_SUPPORT_CONTRACT)
        self.assertTrue(support["passed"], support)
        coverage = support["probe_source_coverage"]
        self.assertEqual(coverage["expected_unique_probe_source_rows"], 24)
        self.assertEqual(coverage["source_unique_probe_identities"], 24)
        self.assertEqual(coverage["placed_unique_probe_identities"], 24)
        self.assertTrue(coverage["all_unique_probe_source_rows_placed"])
        training = support["training_support"]
        self.assertTrue(training["natural_on_policy_support"]["passed"])
        self.assertTrue(training["observable_probe_support"]["passed"])
        self.assertTrue(training["combined_training_support"]["passed"])
        for stratum in (PFNC, UCR):
            probe = training["observable_probe_support"]["recovery_strata"][stratum]
            combined = training["combined_training_support"]["recovery_strata"][stratum]
            # Two replay copies lift the row count, never the distinct-root count.
            self.assertEqual(probe["target_bearing_rows"], 14)
            self.assertEqual(probe["distinct_physical_roots"], 12)
            self.assertEqual(combined["target_bearing_rows"], 14)
            self.assertEqual(combined["distinct_physical_roots"], 12)

    def test_final_selection_cannot_drop_a_required_natural_root(self):
        natural = _natural_rows()
        extra = copy.deepcopy(natural[0])
        extra["example_id"] = "nat-extra"
        extra["physical_root_fingerprint"] = "physical_v3_nat-extra-root"
        natural.append(extra)

        # Discover deterministic 30-of-31 placement.  Observation content is
        # absent from the order key, so the same row stays excluded below.
        preliminary_rows, _ = self._build(natural_d1_rows=natural)
        placed_natural_ids = {
            row["example_id"]
            for row in preliminary_rows
            if str(row.get("example_id") or "").startswith("nat")
        }
        dropped = next(
            row for row in natural if row["example_id"] not in placed_natural_ids
        )

        for row in natural:
            observation = dict(row.get("policy_observation") or {})
            observation.pop("accepted_corrections", None)
            observation.pop("no_material_anomaly_remaining", None)
            row["policy_observation"] = observation
        selected_for_cell = [
            row for row in natural if row["example_id"] in placed_natural_ids
        ][:4]
        for row in [*selected_for_cell, dropped]:
            row["policy_observation"] = {
                **row["policy_observation"],
                "accepted_corrections": [{"tool": "correct_measurements"}],
                "no_material_anomaly_remaining": False,
            }

        source_support = audit_dagger1_final_view_support(
            natural_rows=natural,
            probe_rows=self.probes,
            source_probe_rows=self.probes,
            policy=self.policy,
        )
        self.assertTrue(source_support["passed"], source_support)
        _, report = self._build(natural_d1_rows=natural)
        self.assertEqual(
            report["shortfalls"],
            {
                "d0_bc0_rows": 0,
                "natural_d1_rows": 0,
                "observable_recovery_probe_rows": 0,
            },
        )
        self.assertEqual(report["global_root_cap_violations"], [])
        final_support = report["final_view_support"]
        partial = final_support["training_support"]["natural_on_policy_support"][
            "targeted_state_cells"
        ]["partial_success_retention"]
        self.assertEqual(partial["distinct_physical_roots"], 4)
        self.assertEqual(partial["root_shortfall"], 1)
        self.assertFalse(final_support["passed"])
        self.assertFalse(report["passed"])

    def test_every_unique_probe_source_identity_must_survive_placement(self):
        probes = copy.deepcopy(self.probes)
        # Both rows remain unique source identities, but root-first placement
        # can retain only one of them before replay duplicates are drawn.
        probes[1]["physical_root_fingerprint"] = probes[0][
            "physical_root_fingerprint"
        ]
        _, report = self._build(probe_rows=probes)
        training = report["final_view_support"]["training_support"]
        # Eleven independent PFNC roots still clear the canonical ten-root
        # support floor, so only exact source-row coverage catches the loss.
        self.assertTrue(training["observable_probe_support"]["passed"])
        self.assertTrue(training["combined_training_support"]["passed"])
        coverage = report["final_view_support"]["probe_source_coverage"]
        self.assertEqual(len(coverage["missing_source_probe_identities"]), 1)
        self.assertFalse(coverage["all_unique_probe_source_rows_placed"])
        self.assertFalse(coverage["passed"])
        self.assertFalse(report["passed"])

    def test_duplicates_are_balanced_across_the_two_strata(self):
        _, report = self._build()
        self.assertEqual(
            report["probe_bucket"][PFNC]["duplicate_placements"],
            report["probe_bucket"][UCR]["duplicate_placements"],
        )

    def test_policy_digest_is_bound_into_the_report(self):
        _, report = self._build()
        self.assertEqual(
            report["policy_digest"], round1_view_policy_digest(self.policy)
        )

    def test_an_undersupplied_source_reports_a_shortfall(self):
        rows, report = self._build(natural_d1_rows=_rows("nat", 2))
        # 2 distinct rows at a duplicate cap of 2 can supply only 4 placements.
        self.assertEqual(report["placed"]["natural_d1_rows"], 4)
        self.assertEqual(report["shortfalls"]["natural_d1_rows"], 26)
        self.assertFalse(report["passed"])
        self.assertLess(len(rows), 100)

    def test_empty_source_is_refused(self):
        with self.assertRaises(ValueError):
            self._build(probe_rows=[])


class GlobalCapTests(unittest.TestCase):
    """Probe and natural rows share a root space; D0 does not."""

    def test_root_cap_holds_across_the_probe_natural_union(self):
        shared = "physical_v3_shared-root"
        policy = _small_policy()
        # Every natural and probe row sits on ONE root.  Independent samplers
        # would each place up to 8 and breach the cap after the union.
        natural = _rows("nat", 20, root=shared)
        probes = _rows("pfnc", 12, stratum=PFNC, root=shared) + _rows(
            "ucr", 12, stratum=UCR, root=shared
        )
        rows, report = build_dagger1_three_source_view(
            d0_rows=_rows("d0", 42),
            natural_d1_rows=natural,
            probe_rows=probes,
            policy=policy,
        )
        cap = policy["global_caps"]["max_rows_per_root"]
        placed_on_shared = sum(
            1 for row in rows if row["physical_root_fingerprint"] == shared
        )
        self.assertLessEqual(placed_on_shared, cap)
        self.assertEqual(report["global_root_cap_violations"], [])
        self.assertLessEqual(report["max_rows_placed_on_one_shared_root"], cap)
        # The cap binds, so the view cannot be filled: that must be reported,
        # not silently absorbed.
        self.assertFalse(report["passed"])
        self.assertGreater(report["shortfalls"]["natural_d1_rows"], 0)

    def test_probe_rows_are_placed_before_natural_fill_competes(self):
        shared = "physical_v3_shared-root"
        policy = _small_policy()
        natural = _rows("nat", 20, root=shared)
        probes = _rows("pfnc", 1, stratum=PFNC, root=shared) + _rows(
            "ucr", 1, stratum=UCR, root=shared
        )
        rows, report = build_dagger1_three_source_view(
            d0_rows=_rows("d0", 42),
            natural_d1_rows=natural,
            probe_rows=probes,
            policy=policy,
        )
        # The scarce shared root is spent on probes first, so both probe rows
        # appear even though natural rows vastly outnumber them.
        probe_ids = {row["example_id"] for row in rows if row.get("recovery_stratum")}
        self.assertIn("pfnc-0", probe_ids)
        self.assertIn("ucr-0", probe_ids)
        self.assertGreater(
            report["placed"]["observable_recovery_probe_rows"], 0
        )

    def test_d0_is_capped_in_its_own_root_space(self):
        """A D0 root must not be constrained by a natural D1 root of the same name."""
        shared = "physical_v3_shared-root"
        policy = _small_policy()
        policy["allocation"] = {
            "d0_bc0_rows": 8,
            "natural_d1_rows": 8,
            "observable_recovery_probe_rows": 28,
        }
        policy["total_rows"] = 44
        rows, report = build_dagger1_three_source_view(
            d0_rows=_rows("d0", 8, root=shared),
            natural_d1_rows=_rows("nat", 8, root=shared),
            probe_rows=_rows("pfnc", 12, stratum=PFNC)
            + _rows("ucr", 12, stratum=UCR),
            policy=policy,
        )
        self.assertEqual(report["placed"]["d0_bc0_rows"], 8)
        self.assertEqual(report["placed"]["natural_d1_rows"], 8)


if __name__ == "__main__":
    unittest.main()
