"""CPU-only tests for the research learner-trace generation preflight."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT
from psse_env.dagger.protocol_bridge import unified_tool_schemas
import psse_env.dagger.preliminary_e2b_eval as preliminary_e2b_eval
import psse_env.dagger.release_factories as release_factories
from scripts import research_dagger_demo


def _row(
    index: int,
    *,
    tool: str = "wls_from_path",
    arguments: dict[str, Any] | None = None,
    example_id: str | None = None,
    physical_root: str | None = None,
) -> dict[str, Any]:
    if arguments is None:
        arguments = {"case_path": "active"}
    state = {
        "active_state_id": "active",
        "candidate_state_id": "candidate",
        "history_window": [],
        "remaining_budget": index,
    }
    return {
        "example_id": example_id or f"trace-validation-{index}",
        "physical_root_fingerprint": physical_root or f"root-{index}",
        "tools": unified_tool_schemas(),
        "messages": [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"state": state},
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
                    "active": f"controller:active:{index}",
                    "candidate": f"controller:candidate:{index}",
                }
            }
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _generated(tool: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": tool, "arguments": arguments}, sort_keys=True)


def _mock_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: dict[int, str],
    metrics: dict[int, dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    bundle = object()
    bundle_calls: list[tuple[str, str]] = []

    def cached_bundle(adapter: str, revision: str) -> object:
        bundle_calls.append((adapter, revision))
        return bundle

    class FakePolicy:
        def __init__(self, loaded_bundle: object) -> None:
            assert loaded_bundle is bundle
            self._last_metrics: dict[str, Any] = {}

        @property
        def last_action_metrics(self) -> dict[str, Any]:
            return dict(self._last_metrics)

        def generate_text(self, state: dict[str, Any]) -> str:
            key = int(state["remaining_budget"])
            self._last_metrics = dict((metrics or {}).get(key, {}))
            return outputs[key]

    monkeypatch.setattr(preliminary_e2b_eval, "_cached_bundle", cached_bundle)
    monkeypatch.setattr(preliminary_e2b_eval, "_CanonicalE2BPolicy", FakePolicy)
    monkeypatch.setattr(
        release_factories,
        "checkpoint_tree_sha256",
        lambda _path: "a" * 64,
    )
    return bundle_calls


def test_trace_preflight_scores_every_row_and_reports_tool_breakdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = tmp_path / "validation.jsonl"
    adapter = tmp_path / "adapter"
    output = tmp_path / "preflight.json"
    adapter.mkdir()
    rows = [
        _row(1, physical_root="shared-root"),
        _row(2, physical_root="shared-root"),
        _row(
            3,
            tool="rollback_state",
            arguments={"case_path": "candidate"},
        ),
    ]
    _write_jsonl(validation, rows)
    bundle_calls = _mock_policy(
        monkeypatch,
        outputs={
            1: _generated("wls_from_path", {"case_path": "active"}),
            2: _generated("get_measurement_context", {"case_path": "active"}),
            3: "not-json",
        },
        metrics={
            1: {
                "truncated_input_tokens": 3,
                "hit_max_new_tokens": False,
            },
            2: {
                "truncated_input_tokens": 0,
                "hit_max_new_tokens": True,
            },
            3: {
                "truncated_input_tokens": 0,
                "hit_max_new_tokens": False,
            },
        },
    )

    report = research_dagger_demo.run_trace_preflight(
        validation,
        adapter,
        output,
    )

    assert bundle_calls == [(str(adapter.resolve()), "a" * 64)]
    assert report["selection"]["mode"] == "all_validation_rows"
    assert report["validation_row_count"] == 3
    assert report["validation_physical_root_count"] == 2
    assert report["adapter_tree_sha256"] == "a" * 64
    assert report["validation_file_sha256"] == hashlib.sha256(
        validation.read_bytes()
    ).hexdigest()
    assert report["overall"] == {
        "row_count": 3,
        "schema_valid_count": 2,
        "schema_valid_rate": pytest.approx(2 / 3),
        "state_bound_count": 2,
        "state_bound_rate": pytest.approx(2 / 3),
        "target_tool_match_count": 1,
        "target_tool_match_rate": pytest.approx(1 / 3),
        "exact_target_match_count": 1,
        "exact_target_match_rate": pytest.approx(1 / 3),
        "input_truncated_row_count": 1,
        "input_truncated_row_rate": pytest.approx(1 / 3),
        "truncated_input_token_count": 3,
        "max_new_token_hit_count": 1,
        "max_new_token_hit_rate": pytest.approx(1 / 3),
        "error_count": 1,
        "error_rate": pytest.approx(1 / 3),
    }
    assert report["per_expected_tool"]["wls_from_path"]["row_count"] == 2
    assert (
        report["per_expected_tool"]["wls_from_path"]["exact_target_match_rate"]
        == pytest.approx(0.5)
    )
    assert report["per_expected_tool"]["rollback_state"]["error_count"] == 1
    assert len(report["results"]) == 3
    assert report["results"][2]["error"]
    assert report["stop_on_zero_exact"] is False
    assert report["zero_exact_stop_triggered"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_trace_preflight_zero_exact_option_writes_report_before_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    validation = tmp_path / "validation.jsonl"
    adapter = tmp_path / "adapter"
    output = tmp_path / "preflight.json"
    adapter.mkdir()
    _write_jsonl(
        validation,
        [
            _row(1),
            _row(
                2,
                tool="rollback_state",
                arguments={"case_path": "candidate"},
            ),
        ],
    )
    _mock_policy(
        monkeypatch,
        outputs={
            1: _generated("get_measurement_context", {"case_path": "active"}),
            2: _generated("get_measurement_context", {"case_path": "active"}),
        },
    )

    return_code = research_dagger_demo.main(
        [
            "trace-preflight",
            "--validation",
            str(validation),
            "--adapter",
            str(adapter),
            "--output",
            str(output),
            "--stop-on-zero-exact",
        ]
    )
    capsys.readouterr()

    assert return_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall"]["exact_target_match_count"] == 0
    assert report["zero_exact_stop_triggered"] is True
    assert len(report["results"]) == 2


def test_trace_preflight_rejects_duplicate_validation_ids_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = tmp_path / "validation.jsonl"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_jsonl(
        validation,
        [
            _row(1, example_id="duplicate"),
            _row(2, example_id="duplicate"),
        ],
    )
    monkeypatch.setattr(
        preliminary_e2b_eval,
        "_cached_bundle",
        lambda *_args: pytest.fail("model loader must not run"),
    )

    with pytest.raises(ValueError, match="not unique"):
        research_dagger_demo.run_trace_preflight(
            validation,
            adapter,
            tmp_path / "report.json",
        )


def test_trace_preflight_rejects_noncanonical_tool_registry_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = tmp_path / "validation.jsonl"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    row = _row(1)
    row["tools"] = row["tools"][:-1]
    _write_jsonl(validation, [row])
    monkeypatch.setattr(
        preliminary_e2b_eval,
        "_cached_bundle",
        lambda *_args: pytest.fail("model loader must not run"),
    )

    with pytest.raises(ValueError, match="tool registry is not canonical"):
        research_dagger_demo.run_trace_preflight(
            validation,
            adapter,
            tmp_path / "report.json",
        )
