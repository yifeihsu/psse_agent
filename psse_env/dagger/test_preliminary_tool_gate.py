from __future__ import annotations

import copy
import json

import pytest

from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT
from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas
from psse_env.dagger.preliminary_tool_gate import (
    MINIMUM_BC0_TARGET_TOOL_RATE,
    MINIMUM_DAGGER_TARGET_TOOL_RATE,
    SAMPLE_COUNT,
    PreliminaryToolGateError,
    evaluate_generation,
    select_gate_rows,
    summarize_results,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from gpt_oss_power_sft_revised_v3 import sanitize_tool_schemas


TARGETS = (
    ("wls_from_path", {"case_path": "active"}),
    ("get_measurement_context", {"case_path": "active"}),
    ("rollback_state", {"case_path": "candidate"}),
    ("commit_state", {"case_path": "candidate"}),
    ("ask_for_more_evidence", {"case_path": "active"}),
    ("get_topology_context", {"case_path": "active"}),
)


def _row(index: int) -> dict:
    tool, arguments = TARGETS[index % len(TARGETS)]
    state = {
        "active_state_id": "active",
        "candidate_state_id": "candidate",
        "history_window": [],
        "remaining_budget": 8,
    }
    payload = {"state": state}
    return {
        "example_id": f"gate-example-{index:03d}",
        "physical_root_fingerprint": f"physical-root-{index % 15:02d}",
        "tools": unified_tool_schemas(),
        "messages": [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": copy.deepcopy(arguments),
                        },
                    }
                ],
            },
        ],
        "metadata": {
            "controller": {
                "state_aliases": {
                    "active": f"controller:s{index}",
                    "candidate": f"controller:s{index + 1}",
                }
            }
        },
    }


def _text(tool: str, arguments: dict) -> str:
    return json.dumps({"name": tool, "arguments": arguments}, sort_keys=True)


def test_inference_reuses_exact_training_tool_schema_sanitizer() -> None:
    assert canonical_prompt_tool_schemas() == sanitize_tool_schemas(
        unified_tool_schemas()
    )
    assert canonical_prompt_tool_schemas() != unified_tool_schemas()


def test_selection_is_deterministic_and_covers_roots_tools_and_references() -> None:
    rows = [_row(index) for index in range(40)]
    first = select_gate_rows(rows)
    second = select_gate_rows(list(reversed(rows)))

    assert [row["example_id"] for row in first] == [
        row["example_id"] for row in second
    ]
    assert len(first) == SAMPLE_COUNT
    assert len({row["physical_root_fingerprint"] for row in first}) == 15
    assert {
        row["messages"][-1]["tool_calls"][0]["function"]["name"]
        for row in first
    } == {name for name, _ in TARGETS}


def test_selection_rejects_noncanonical_user_serialization() -> None:
    rows = [_row(index) for index in range(40)]
    rows[0]["messages"][1]["content"] = '{"state":{"active_state_id":"active"}}'
    with pytest.raises(PreliminaryToolGateError, match="serialization"):
        select_gate_rows(rows)


def test_evaluate_generation_proves_schema_and_controller_binding() -> None:
    row = _row(0)
    result = evaluate_generation(
        row,
        _text("wls_from_path", {"case_path": "active"}),
        action_metrics={"generated_tokens": 20, "hit_max_new_tokens": False},
    )

    assert result["schema_valid"] is True
    assert result["state_bound"] is True
    assert result["target_tool_match"] is True
    assert result["exact_target_match"] is True
    assert result["bound_internal_action"]["tool"] == "run_wls"
    assert result["bound_internal_action"]["arguments"]["state_id"] == "controller:s0"


def test_evaluate_generation_separates_schema_valid_from_unknown_alias() -> None:
    row = _row(0)
    result = evaluate_generation(
        row,
        _text("wls_from_path", {"case_path": "episode"}),
    )

    assert result["schema_valid"] is True
    assert result["state_bound"] is False
    assert result["target_tool_match"] is True
    assert result["exact_target_match"] is False
    assert "Unknown controller state alias" in result["error"]


def test_summary_is_fail_closed_at_objective_action_thresholds() -> None:
    rows = [_row(index) for index in range(SAMPLE_COUNT)]
    passing = [
        evaluate_generation(
            row,
            _text(
                row["messages"][-1]["tool_calls"][0]["function"]["name"],
                row["messages"][-1]["tool_calls"][0]["function"]["arguments"],
            ),
        )
        for row in rows
    ]
    assert summarize_results(passing)["passed"] is True

    one_invalid = copy.deepcopy(passing)
    one_invalid[0]["schema_valid"] = False
    one_invalid[0]["state_bound"] = False
    assert summarize_results(one_invalid)["passed"] is False

    one_max_hit = copy.deepcopy(passing)
    one_max_hit[0]["hit_max_new_tokens"] = True
    assert summarize_results(one_max_hit)["passed"] is False

    no_target_matches = copy.deepcopy(passing)
    for result in no_target_matches:
        result["target_tool_match"] = False
    assert summarize_results(
        no_target_matches,
        minimum_target_tool_rate=MINIMUM_BC0_TARGET_TOOL_RATE,
    )["passed"] is True
    assert summarize_results(
        no_target_matches,
        minimum_target_tool_rate=MINIMUM_DAGGER_TARGET_TOOL_RATE,
    )["passed"] is False
