"""The probe ablation must remain an exact projection of the full view."""

from __future__ import annotations

import copy
import unittest

from psse_env.dagger.natural_only_view import (
    NATURAL_ONLY_VIEW_BUILD_CONTRACT,
    build_round1_natural_only_view,
)
from psse_env.dagger.round1_view_policy import (
    ROUND1_NATURAL_ONLY_VIEW_POLICY,
    round1_natural_only_view_policy_digest,
    round1_view_policy_digest,
    validate_round1_natural_only_view_policy,
)


def _rows(prefix: str, count: int, source: str) -> list[dict]:
    return [
        {
            "example_id": f"{prefix}-{index:04d}",
            "physical_root_fingerprint": f"{prefix}-root-{index % 211:03d}",
            "replay_source": source,
            "payload": {"source_index": index},
        }
        for index in range(count)
    ]


def _canonical_full() -> list[dict]:
    # Interleave source blocks to prove the builder preserves parent order and
    # does not merely concatenate or resample source ledgers.
    d0 = _rows("d0", 1317, "d0_bc0")
    natural = _rows("natural", 525, "natural_dagger1")
    probes = _rows("probe", 38, "observable_recovery_probe")
    return [*d0[:400], *probes[:11], *natural[:200], *d0[400:], *probes[11:], *natural[200:]]


class NaturalOnlyPolicyTests(unittest.TestCase):
    def test_policy_is_versioned_and_bound_to_the_full_policy(self) -> None:
        report = validate_round1_natural_only_view_policy(
            copy.deepcopy(ROUND1_NATURAL_ONLY_VIEW_POLICY)
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["total_rows"], 1842)
        self.assertEqual(
            report["parent_policy_digest"], round1_view_policy_digest()
        )
        self.assertEqual(
            report["digest"], round1_natural_only_view_policy_digest()
        )

    def test_policy_cannot_enable_reselection_or_change_parent_digest(self) -> None:
        for field, value, pattern in (
            ("permit_reselection", True, "forbid reselection"),
            ("parent_view_policy_digest", "0" * 64, "parent digest"),
        ):
            with self.subTest(field=field):
                policy = copy.deepcopy(ROUND1_NATURAL_ONLY_VIEW_POLICY)
                policy[field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_round1_natural_only_view_policy(policy)


class NaturalOnlyProjectionTests(unittest.TestCase):
    def test_exact_parent_subsequence_without_reselection(self) -> None:
        full = _canonical_full()
        derived, report = build_round1_natural_only_view(full)
        expected = [
            copy.deepcopy(row)
            for row in full
            if row["replay_source"] != "observable_recovery_probe"
        ]
        self.assertEqual(derived, expected)
        self.assertEqual(len(derived), 1842)
        self.assertEqual(
            report["contract"], NATURAL_ONLY_VIEW_BUILD_CONTRACT
        )
        self.assertEqual(
            report["retained_allocation"],
            {
                "d0_bc0_rows": 1317,
                "natural_d1_rows": 525,
                "observable_recovery_probe_rows": 0,
            },
        )
        self.assertEqual(report["excluded_probe_rows"], 38)
        self.assertFalse(report["reselection_performed"])
        self.assertTrue(report["identical_parent_row_objects"])
        self.assertTrue(report["identical_parent_order"])

    def assert_rejected(self, rows: list[dict], pattern: str = "allocation") -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            build_round1_natural_only_view(rows)

    def test_forged_probe_relabelled_as_natural_is_rejected(self) -> None:
        rows = _canonical_full()
        probe = next(
            row
            for row in rows
            if row["replay_source"] == "observable_recovery_probe"
        )
        probe["replay_source"] = "natural_dagger1"
        self.assert_rejected(rows)

    def test_extra_and_missing_probe_rows_are_rejected(self) -> None:
        rows = _canonical_full()
        extra = copy.deepcopy(
            next(
                row
                for row in rows
                if row["replay_source"] == "observable_recovery_probe"
            )
        )
        extra["example_id"] = "probe-forged-extra"
        self.assert_rejected([*rows, extra])
        missing = copy.deepcopy(rows)
        del missing[
            next(
                index
                for index, row in enumerate(missing)
                if row["replay_source"] == "observable_recovery_probe"
            )
        ]
        self.assert_rejected(missing)

    def test_extra_and_missing_natural_rows_are_rejected(self) -> None:
        rows = _canonical_full()
        extra = copy.deepcopy(
            next(row for row in rows if row["replay_source"] == "natural_dagger1")
        )
        extra["example_id"] = "natural-forged-extra"
        self.assert_rejected([*rows, extra])
        missing = copy.deepcopy(rows)
        del missing[
            next(
                index
                for index, row in enumerate(missing)
                if row["replay_source"] == "natural_dagger1"
            )
        ]
        self.assert_rejected(missing)

    def test_unknown_source_cannot_be_hidden_in_parent(self) -> None:
        rows = _canonical_full()
        rows[0]["replay_source"] = "forged_source"
        self.assert_rejected(rows)


if __name__ == "__main__":
    unittest.main()
