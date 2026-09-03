from __future__ import annotations

import copy
import json
from pathlib import Path

from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.dagger.research_action_preflight import (
    compare_preflight_reports,
    main,
    score_validation_rows,
)


def _row(example_id: str, tool: str, arguments: dict) -> dict:
    payload = {"state": {"case_id": example_id, "history": []}}
    return {
        "example_id": example_id,
        "physical_root_fingerprint": f"root-{example_id}",
        "metadata": {"protocol": "canonical"},
        "messages": [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, sort_keys=True, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "id": "call_0",
                    "function": {"name": tool, "arguments": arguments},
                }],
            },
        ],
        "tools": unified_tool_schemas(),
    }


class FakePolicy:
    def __init__(self, actions: dict[str, object], metrics: dict[str, dict] | None = None):
        self.actions = actions
        self.metrics = metrics or {}
        self.last_action_metrics: dict = {}
        self.states: list[dict] = []

    def act_model_observation(self, state: dict) -> object:
        self.states.append(copy.deepcopy(state))
        example_id = state["case_id"]
        self.last_action_metrics = copy.deepcopy(self.metrics.get(example_id, {}))
        action = self.actions[example_id]
        if isinstance(action, Exception):
            raise action
        return copy.deepcopy(action)


def test_score_validation_rows_reports_action_quality_and_failures() -> None:
    rows = [
        _row("a", "wls_from_path", {"case_path": "active"}),
        _row("b", "rollback_state", {"case_path": "candidate"}),
        _row("c", "finalize_diagnosis", {}),
    ]
    policy = FakePolicy(
        {
            "a": {"tool": "wls_from_path", "arguments": {"case_path": "active"}},
            "b": {"tool": "commit_state", "arguments": {"case_path": "candidate"}},
            "c": RuntimeError("decode failed"),
        },
        {
            "a": {"truncated_input_tokens": 0, "hit_max_new_tokens": False},
            "b": {"truncated_input_tokens": 2, "hit_max_new_tokens": False},
            "c": {"truncated_input_tokens": 0, "hit_max_new_tokens": True},
        },
    )

    report = score_validation_rows(rows, policy)

    assert [state["case_id"] for state in policy.states] == ["a", "b", "c"]
    assert report["schema_valid_count"] == 2
    assert report["tool_match_count"] == 1
    assert report["exact_count"] == 1
    assert report["error_count"] == 1
    assert report["truncation_count"] == 2
    assert report["summary"]["truncated_input_token_count"] == 2
    assert report["summary"]["generated_action_tool_counts"] == {
        "<invalid>": 1,
        "commit_state": 1,
        "wls_from_path": 1,
    }


def _report(ids: list[str], actions: list[dict], exact: int, **summary_overrides: int) -> dict:
    count = len(ids)
    summary = {
        "example_count": count,
        "schema_valid_count": count,
        "error_count": 0,
        "truncation_count": 0,
        "exact_count": exact,
        **summary_overrides,
    }
    return {
        "example_ids": ids,
        "summary": summary,
        "results": [
            {"example_id": example_id, "generated_action": action}
            for example_id, action in zip(ids, actions)
        ],
    }


def test_compare_requires_improvement_and_changed_action() -> None:
    ids = [f"e{i}" for i in range(6)]
    action = {"tool": "finalize_diagnosis", "arguments": {}}
    baseline = _report(ids, [action] * 6, 4)
    changed = {"tool": "wls_from_path", "arguments": {"case_path": "active"}}
    candidate = _report(ids, [changed, *([action] * 5)], 5)

    decision = compare_preflight_reports(baseline, candidate)

    assert decision["passed"] is True
    assert decision["exact_count_delta"] == 1
    assert decision["changed_example_ids"] == ["e0"]
    unchanged = compare_preflight_reports(baseline, _report(ids, [action] * 6, 5))
    assert unchanged["passed"] is False
    assert "generated_action_changed" in unchanged["failure_reasons"]


def test_compare_rejects_mismatched_examples_schema_errors_and_truncation() -> None:
    action = {"tool": "finalize_diagnosis", "arguments": {}}
    baseline = _report(["a"], [action], 0)
    candidate = _report(
        ["b"], [action], 5,
        schema_valid_count=0, error_count=1, truncation_count=1,
    )
    decision = compare_preflight_reports(baseline, candidate)
    assert decision["passed"] is False
    assert {
        "same_examples",
        "candidate_schema_100_percent",
        "candidate_zero_errors",
        "candidate_zero_truncation",
        "generated_action_changed",
    }.issubset(decision["failure_reasons"])


def test_compare_cli_writes_failed_decision_and_returns_two(tmp_path: Path) -> None:
    action = {"tool": "finalize_diagnosis", "arguments": {}}
    baseline = _report(["a"], [action], 0)
    candidate = _report(["a"], [action], 0)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "decision.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    status = main([
        "compare", "--baseline", str(baseline_path), "--candidate",
        str(candidate_path), "--output", str(output_path), "--minimum-exact", "1",
    ])

    assert status == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is False
