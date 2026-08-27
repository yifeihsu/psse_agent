"""CPU-only regression tests for auditable research evaluation records."""

from __future__ import annotations

from typing import Any

from research.evaluate import EVALUATION_SCHEMA_VERSION, run_episode


class _Observation:
    def as_dict(self) -> dict[str, Any]:
        return {"state": "visible"}


class _Policy:
    def __init__(self, action: dict[str, Any]) -> None:
        self.action = action

    def act(self, _observation: dict[str, Any]) -> dict[str, Any]:
        return self.action


class _Env:
    terminal = False
    terminal_outcome = None

    def reset(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario

    def get_policy_observation(self, _history):
        return _Observation()

    def step(self, action: dict[str, Any]):
        self.terminal = True
        self.terminal_outcome = "resolved"
        return {}, {
            "execution_status": "success",
            "error_code": None,
            "tool_metrics": {"terminal_outcome": "resolved"},
            "executed": action,
        }


def test_episode_persists_root_binding_and_untruncated_arguments() -> None:
    long_value = "x" * 500
    scenario = {
        "execution": {},
        "grouping": {
            "root_scenario_id": "root-7",
            "physical_root_fingerprint": "sha256:abc",
        },
    }
    action = {"tool": "ask_for_more_evidence", "arguments": {"request": long_value}}
    report = run_episode(_Env(), _Policy(action), scenario, max_steps=2)

    assert EVALUATION_SCHEMA_VERSION == 2
    assert report["scenario_id"] == "root-7"
    assert report["root_scenario_id"] == "root-7"
    assert report["physical_root_fingerprint"] == "sha256:abc"
    assert report["actions"][0]["arguments"] == {"request": long_value}
    assert report["termination_reason"] == "terminal_outcome"


class _BrokenEnv(_Env):
    def step(self, _action: dict[str, Any]):
        raise RuntimeError("executor unavailable")


def test_environment_exception_is_not_silently_labeled_as_horizon() -> None:
    report = run_episode(
        _BrokenEnv(),
        _Policy({"tool": "run_wls", "arguments": {}}),
        {"root_scenario_id": "root-1"},
        max_steps=2,
    )

    assert report["termination_reason"] == "environment_step_exception"
    assert report["first_error"].startswith("env.step RuntimeError")
    assert report["horizon_truncated"] is True
