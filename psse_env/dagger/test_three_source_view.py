"""The emitted view must honour the frozen allocation and the union caps."""

from __future__ import annotations

import copy
import unittest

from psse_env.dagger.round1_view_policy import (
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_view_policy_digest,
)
from psse_env.dagger.three_source_view import build_dagger1_three_source_view

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


def _small_policy(**overrides):
    policy = copy.deepcopy(ROUND1_THREE_SOURCE_VIEW_POLICY)
    policy["total_rows"] = 100
    policy["allocation"] = {
        "d0_bc0_rows": 60,
        "natural_d1_rows": 30,
        "observable_recovery_probe_rows": 10,
    }
    policy["probe_bucket"] = {
        PFNC: 5,
        UCR: 5,
        "distinct_roots_retained_per_stratum": 3,
        "duplicate_placements_per_stratum": 2,
    }
    policy["probe_floor_distinct_roots"] = 3
    policy.update(overrides)
    return policy


class AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _small_policy()
        self.d0 = _rows("d0", 60)
        self.natural = _rows("nat", 30)
        self.probes = _rows("pfnc", 3, stratum=PFNC) + _rows("ucr", 3, stratum=UCR)

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
        self.assertEqual(report["placed"]["d0_bc0_rows"], 60)
        self.assertEqual(report["placed"]["natural_d1_rows"], 30)
        self.assertEqual(report["placed"]["observable_recovery_probe_rows"], 10)

    def test_every_distinct_probe_root_is_retained_before_any_duplicate(self):
        _, report = self._build()
        for stratum in (PFNC, UCR):
            entry = report["probe_bucket"][stratum]
            self.assertEqual(entry["distinct_roots_retained"], 3)
            self.assertEqual(entry["duplicate_placements"], 2)
            self.assertEqual(entry["shortfall"], 0)

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
        policy["allocation"] = {
            "d0_bc0_rows": 60,
            "natural_d1_rows": 30,
            "observable_recovery_probe_rows": 10,
        }
        # Every natural and probe row sits on ONE root.  Independent samplers
        # would each place up to 8 and breach the cap after the union.
        natural = _rows("nat", 20, root=shared)
        probes = _rows("pfnc", 3, stratum=PFNC, root=shared) + _rows(
            "ucr", 3, stratum=UCR, root=shared
        )
        rows, report = build_dagger1_three_source_view(
            d0_rows=_rows("d0", 60),
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
            d0_rows=_rows("d0", 60),
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
            "observable_recovery_probe_rows": 10,
        }
        policy["total_rows"] = 26
        rows, report = build_dagger1_three_source_view(
            d0_rows=_rows("d0", 8, root=shared),
            natural_d1_rows=_rows("nat", 8, root=shared),
            probe_rows=_rows("pfnc", 3, stratum=PFNC) + _rows("ucr", 3, stratum=UCR),
            policy=policy,
        )
        self.assertEqual(report["placed"]["d0_bc0_rows"], 8)
        self.assertEqual(report["placed"]["natural_d1_rows"], 8)


if __name__ == "__main__":
    unittest.main()
