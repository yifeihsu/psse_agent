"""Tests for the frozen occupancy/exposure screen analysis."""

from __future__ import annotations

from pathlib import Path

from research.exposure_screen_analysis import (
    _exact,
    _stable_exact,
    checkpoint_metrics,
    exact_mcnemar_p,
    load_report,
    paired_table,
    quality_checks,
)


def _episode(
    root: str,
    outcome: str,
    *,
    final_active_state: str | None = None,
    generation_abort: bool = False,
    loop: bool = False,
) -> dict[str, object]:
    return {
        "root_scenario_id": root,
        "initial_true_error_count": 1,
        "episode_outcome_class": outcome,
        "final_active_state_class": final_active_state or outcome,
        "generation_abort": generation_abort,
        "loop_before_stable_final_state": loop,
    }


def test_exact_mcnemar_matches_small_closed_form_cases() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(0, 5) == 0.0625
    assert exact_mcnemar_p(1, 5) == 0.21875
    assert exact_mcnemar_p(3, 3) == 1.0


def test_paired_table_uses_faulted_roots_not_independent_proportions() -> None:
    exact = "exact_physical_recovery"
    partial = "partial_recovery"
    first_rows = [_episode("a", exact), _episode("b", exact), _episode("c", partial)]
    second_rows = [_episode("a", exact), _episode("b", partial), _episode("c", exact)]
    first = {"by_root": {row["root_scenario_id"]: row for row in first_rows}}
    second = {"by_root": {row["root_scenario_id"]: row for row in second_rows}}

    table = paired_table(
        first,
        second,
        _exact,
    )

    assert table == {
        "both_positive": 1,
        "policy_1_only": 1,
        "policy_2_only": 1,
        "neither_positive": 0,
        "discordant": 2,
        "exact_mcnemar_two_sided_p": 1.0,
    }


def test_final_exact_is_distinct_from_repaired_but_unstable_trajectory() -> None:
    repaired_then_aborted = _episode(
        "a",
        "not_assessable_generation_abort",
        final_active_state="exact_physical_recovery",
        generation_abort=True,
    )
    repaired_then_looped = _episode(
        "b",
        "loop_before_stable_final_state",
        final_active_state="exact_physical_recovery",
        loop=True,
    )

    assert _exact(repaired_then_aborted)
    assert not _stable_exact(repaired_then_aborted)
    assert _exact(repaired_then_looped)
    assert not _stable_exact(repaired_then_looped)


def test_tracked_historical_reports_regress_final_stable_and_provenance() -> None:
    report_path = Path(__file__).parent / "results" / "physical_audit_all8_full.json"
    evaluation_keys = {
        "historical_e2b_selective": "e2b_selective",
        "historical_e2b_full": "e2b_full_occupancy",
        "historical_12b_selective": "12b_selective",
        "historical_12b_full": "12b_full_occupancy",
    }
    reports = {
        name: load_report(f"{name}={report_path}::{evaluation_key}")
        for name, evaluation_key in evaluation_keys.items()
    }

    quality = quality_checks(reports)
    assert quality["passed"]
    assert quality["cross_report_same_fault_mask"]
    assert all(
        row["provenance"]["class"] == "legacy_ordered_replay_only"
        for row in quality["per_report"].values()
    )

    e2b_full = checkpoint_metrics(reports["historical_e2b_full"])
    selective_12b = checkpoint_metrics(reports["historical_12b_selective"])
    assert (e2b_full["final_exact_recovery"], e2b_full["stable_exact_recovery"]) == (
        23,
        21,
    )
    assert (
        selective_12b["final_exact_recovery"],
        selective_12b["stable_exact_recovery"],
    ) == (43, 39)

    mislabeled = dict(reports["historical_e2b_selective"])
    mislabeled["evaluation_key"] = "12b_full_occupancy"
    mislabeled_quality = quality_checks(
        {"historical_e2b_selective": mislabeled}
    )
    assert not mislabeled_quality["passed"]
    assert not mislabeled_quality["per_report"]["historical_e2b_selective"][
        "checks"
    ]["evaluation_label_expected"]
