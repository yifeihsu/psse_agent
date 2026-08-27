"""Focused CPU-only tests for research DAgger aggregation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from research.run_dagger import build_aggregate
from research.train import read_jsonl, write_jsonl


def _round1_row(
    example_id: str,
    *,
    state_origin: str,
    production_label_eligible: bool,
    audit_passed: bool = True,
    invalid_learner_action: bool = False,
) -> dict:
    return {
        "example_id": example_id,
        "root_scenario_id": example_id,
        "state_origin": state_origin,
        "production_label_eligible": production_label_eligible,
        "production_label_ineligibility_reason": (
            None if production_label_eligible else "not_recovery_state"
        ),
        "preferred_action": {
            "tool": "run_wls",
            "arguments": {"state_id": f"{example_id}:s0"},
        },
        "executed_action": {
            "tool": "__invalid_action__" if invalid_learner_action else "run_wls",
            "arguments": {},
        },
        "offline_teacher_target_audit": {"passed": audit_passed},
    }


@pytest.mark.parametrize(
    ("inclusion", "expected_ids", "expected_origins", "allows_ineligible"),
    [
        ("selective", {"learner_recovery"}, {"learner_policy": 1}, False),
        (
            "learner_full",
            {"learner_recovery", "learner_non_recovery", "learner_invalid"},
            {"learner_policy": 3},
            True,
        ),
        (
            "full_occupancy",
            {
                "learner_recovery",
                "learner_non_recovery",
                "learner_invalid",
                "expert_visited",
                "initial",
            },
            {"expert_policy": 1, "initial": 1, "learner_policy": 3},
            True,
        ),
    ],
)
def test_inclusion_modes_select_the_intended_audited_occupancy(
    tmp_path: Path,
    inclusion: str,
    expected_ids: set[str],
    expected_origins: dict[str, int],
    allows_ineligible: bool,
) -> None:
    round0_train = tmp_path / "round0.train.jsonl"
    round0_validation = tmp_path / "round0.validation.jsonl"
    round1 = tmp_path / "round1.jsonl"
    write_jsonl(round0_train, [])
    write_jsonl(round0_validation, [])
    write_jsonl(
        round1,
        [
            _round1_row(
                "learner_recovery",
                state_origin="learner_policy",
                production_label_eligible=True,
            ),
            _round1_row(
                "learner_non_recovery",
                state_origin="learner_policy",
                production_label_eligible=False,
            ),
            _round1_row(
                "learner_invalid",
                state_origin="learner_policy",
                production_label_eligible=False,
                invalid_learner_action=True,
            ),
            _round1_row(
                "expert_visited",
                state_origin="expert_policy",
                production_label_eligible=False,
            ),
            _round1_row(
                "initial",
                state_origin="initial",
                production_label_eligible=False,
            ),
            _round1_row(
                "learner_audit_failed",
                state_origin="learner_policy",
                production_label_eligible=False,
                audit_passed=False,
            ),
        ],
    )

    converter_call: dict = {}

    def fake_converter(rows, **kwargs):
        converted = [dict(row) for row in rows]
        converter_call["ids"] = {row["example_id"] for row in converted}
        converter_call.update(kwargs)
        return converted

    output_dir = tmp_path / inclusion
    with patch(
        "psse_env.dagger.dataset_builder.examples_to_chat_sft",
        side_effect=fake_converter,
    ):
        report = build_aggregate(
            round0_train=round0_train,
            round0_validation=round0_validation,
            round1_rows=round1,
            output_dir=output_dir,
            inclusion=inclusion,
        )

    output_ids = {
        row["example_id"]
        for path in (report["train"], report["validation"])
        for row in read_jsonl(path)
    }
    assert converter_call["ids"] == expected_ids
    assert output_ids == expected_ids
    assert converter_call["allow_ineligible_auxiliary"] is allows_ineligible
    assert report["teacher_abstention_rows"] == 1
    assert report["round1_audited_rows"] == 5
    assert report["round1_rows_selected"] == len(expected_ids)
    assert report["round1_audited_state_origin_breakdown"] == {
        "expert_policy": 1,
        "initial": 1,
        "learner_policy": 3,
    }
    assert report["round1_selected_state_origin_breakdown"] == expected_origins
    assert {
        row["example_id"]
        for row in read_jsonl(output_dir / "teacher_abstentions.jsonl")
    } == {"learner_audit_failed"}
    assert report["censored_by_old_rule"] == (
        1 if "learner_invalid" in expected_ids else 0
    )
    assert report["invalid_learner_action_rows_retained"] == (
        1 if "learner_invalid" in expected_ids else 0
    )


def test_unknown_inclusion_lists_all_supported_modes(tmp_path: Path) -> None:
    round0_train = tmp_path / "round0.train.jsonl"
    round0_validation = tmp_path / "round0.validation.jsonl"
    round1 = tmp_path / "round1.jsonl"
    for path in (round0_train, round0_validation, round1):
        write_jsonl(path, [])

    with pytest.raises(ValueError) as excinfo:
        build_aggregate(
            round0_train=round0_train,
            round0_validation=round0_validation,
            round1_rows=round1,
            output_dir=tmp_path / "out",
            inclusion="unknown",
        )

    message = str(excinfo.value)
    assert "selective" in message
    assert "learner_full" in message
    assert "full_occupancy" in message


def test_all_inclusion_modes_share_the_audited_episode_split(tmp_path: Path) -> None:
    round0_train = tmp_path / "round0.train.jsonl"
    round0_validation = tmp_path / "round0.validation.jsonl"
    round1 = tmp_path / "round1.jsonl"
    write_jsonl(round0_train, [])
    write_jsonl(round0_validation, [])
    rows = []
    for index in range(10):
        origin = "learner_policy" if index % 2 else "expert_policy"
        rows.append(
            _round1_row(
                f"episode-{index}",
                state_origin=origin,
                production_label_eligible=index in {1, 3},
            )
        )
    write_jsonl(round1, rows)

    reports = {}
    with patch(
        "psse_env.dagger.dataset_builder.examples_to_chat_sft",
        side_effect=lambda selected, **_kwargs: [dict(row) for row in selected],
    ):
        for inclusion in ("selective", "learner_full", "full_occupancy"):
            reports[inclusion] = build_aggregate(
                round0_train=round0_train,
                round0_validation=round0_validation,
                round1_rows=round1,
                output_dir=tmp_path / inclusion,
                inclusion=inclusion,
                validation_fraction=0.2,
                seed=7,
            )

    assignments = {
        tuple(report["round1_split_assignment_validation_episodes"])
        for report in reports.values()
    }
    assert len(assignments) == 1
    assert all(
        report["round1_split_universe_episodes"] == 10
        for report in reports.values()
    )


def test_split_uses_root_identity_when_converter_drops_episode_id(
    tmp_path: Path,
) -> None:
    round0_train = tmp_path / "round0.train.jsonl"
    round0_validation = tmp_path / "round0.validation.jsonl"
    round1 = tmp_path / "round1.jsonl"
    write_jsonl(round0_train, [])
    write_jsonl(round0_validation, [])
    rows = []
    for index in range(10):
        row = _round1_row(
            f"root-{index}",
            state_origin="learner_policy",
            production_label_eligible=True,
        )
        row["episode_id"] = f"collection-episode-{index}"
        rows.append(row)
    write_jsonl(round1, rows)

    def converter_drops_episode_id(selected, **_kwargs):
        converted = []
        for row in selected:
            exported = dict(row)
            exported.pop("episode_id", None)
            converted.append(exported)
        return converted

    with patch(
        "psse_env.dagger.dataset_builder.examples_to_chat_sft",
        side_effect=converter_drops_episode_id,
    ):
        report = build_aggregate(
            round0_train=round0_train,
            round0_validation=round0_validation,
            round1_rows=round1,
            output_dir=tmp_path / "out",
            inclusion="selective",
            validation_fraction=0.2,
            seed=7,
        )

    held_out = set(report["round1_split_assignment_validation_episodes"])
    assert held_out
    assert all(value.startswith("root-") for value in held_out)
    assert report["round1_rows_added_to_validation"] == len(held_out)
    assert report["round1_rows_added_to_train"] == 10 - len(held_out)


def test_missing_state_origin_fails_closed(tmp_path: Path) -> None:
    round0_train = tmp_path / "round0.train.jsonl"
    round0_validation = tmp_path / "round0.validation.jsonl"
    round1 = tmp_path / "round1.jsonl"
    write_jsonl(round0_train, [])
    write_jsonl(round0_validation, [])
    row = _round1_row(
        "missing-origin",
        state_origin="learner_policy",
        production_label_eligible=False,
    )
    row.pop("state_origin")
    write_jsonl(round1, [row])

    with pytest.raises(ValueError, match="unexpected state_origin"):
        build_aggregate(
            round0_train=round0_train,
            round0_validation=round0_validation,
            round1_rows=round1,
            output_dir=tmp_path / "out",
            inclusion="learner_full",
        )
