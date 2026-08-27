from __future__ import annotations

from psse_env.dagger.dataset_builder import prepare_model_policy_observation
from psse_env.dagger.research_dagger_router import bind_observable_router_action


def _verified_candidate(
    *, correction_tool: str = "correct_measurements", **verification: object
) -> dict[str, object]:
    verification_payload: dict[str, object] = {
        "execution_status": "success",
        "state_id": "episode:s1",
        "evidence_source": "deployment_wls:test",
        "evidence_sufficiency": {"sufficient": True},
        **verification,
    }
    return {
        "active_state_id": "episode:s0",
        "candidate_state_id": "episode:s1",
        "candidate_lifecycle": "VERIFIED_CANDIDATE",
        "has_open_candidate": True,
        "has_verified_candidate": True,
        "last_verification": verification_payload,
        "last_tool_output": {"valid_next_actions": []},
        "history_window": [
            {
                "tool": correction_tool,
                "arguments": {"state_id": "episode:s0"},
                "outcome": {"execution_status": "success"},
            }
        ],
    }


def test_canonical_view_preserves_decision_closure_and_valid_next_actions() -> None:
    verification = {
        "state_id": "episode:s1",
        "state_hash": "a" * 64,
        "evidence_source": "deployment_wls:test",
        "chi_square_dof": 95,
        "chi_square_statistic": 120.0,
        "chi_square_threshold": 130.0,
        "converged": True,
        "evidence_sufficiency": {"sufficient": True},
        "globally_resolved": False,
        "post_action_resolved": False,
        "physical_constraints_ok": True,
        "physical_evidence_complete": True,
        "target_metric_value": 0.1,
        "target_metric_threshold": 3.0,
        "target_progress": 0.99,
        "unresolved_signatures": ["measurement_residual_outlier"],
    }
    raw = {
        "active_state_id": "episode:s0",
        "candidate_state_id": "episode:s1",
        "candidate_parent_id": "episode:s0",
        "candidate_lifecycle": "VERIFIED_CANDIDATE",
        "has_open_candidate": True,
        "has_verified_candidate": True,
        "last_verification": verification,
        "last_tool_output": {
            "execution_status": "failure",
            "error_code": "candidate_lifecycle_violation",
            "error_detail": "rollback_accepted_candidate",
            "valid_next_actions": [
                {
                    "tool": "commit_state",
                    "arguments": {"candidate_state_id": "episode:s1"},
                }
            ],
        },
    }

    prepared, _ = prepare_model_policy_observation(raw, history=[])

    closure = prepared["last_verification"]
    assert closure["globally_resolved"] is False
    assert closure["physical_constraints_ok"] is True
    assert closure["target_progress"] == 0.99
    assert closure["target_metric_value"] == 0.1
    assert prepared["last_tool_output"]["error_detail"] == (
        "rollback_accepted_candidate"
    )
    assert prepared["last_tool_output"]["valid_next_actions"] == [
        {
            "tool": "commit_state",
            "arguments": {"candidate_state_id": "candidate"},
        }
    ]


def test_safe_partial_candidate_commits_only_with_complete_metric_closure() -> None:
    observation = _verified_candidate(
        globally_resolved=False,
        physical_constraints_ok=True,
        target_progress=0.99,
        target_metric_value=0.1,
        target_metric_threshold=3.0,
        global_progress=0.20,
    )

    action, guard = bind_observable_router_action("rollback_state", observation)

    assert action == {
        "tool": "commit_state",
        "arguments": {"case_path": "episode:s1"},
    }
    assert guard == "verified_candidate_commit_closure"

    incomplete = _verified_candidate(
        globally_resolved=False,
        physical_constraints_ok=True,
        target_progress=0.99,
        target_metric_value=3.1,
        target_metric_threshold=3.0,
    )
    action, guard = bind_observable_router_action("rollback_state", incomplete)
    assert action["tool"] == "rollback_state"
    assert guard is None


def test_parameter_partial_closure_reuses_canonical_global_progress_floor() -> None:
    accepted = _verified_candidate(
        correction_tool="correct_parameters",
        globally_resolved=False,
        physical_constraints_ok=True,
        target_progress=0.9366973484538216,
        target_metric_value=0.777,
        target_metric_threshold=3.0,
        global_progress=0.31,
    )
    action, guard = bind_observable_router_action("rollback_state", accepted)
    assert action["tool"] == "commit_state"
    assert guard == "verified_candidate_commit_closure"

    rejected = _verified_candidate(
        correction_tool="correct_parameters",
        globally_resolved=False,
        physical_constraints_ok=True,
        target_progress=0.9366973484538216,
        target_metric_value=0.777,
        target_metric_threshold=3.0,
        global_progress=0.29,
    )
    action, guard = bind_observable_router_action("rollback_state", rejected)
    assert action["tool"] == "rollback_state"
    assert guard is None


def test_candidate_bound_advertised_commit_wins_after_invalid_rollback() -> None:
    observation = _verified_candidate(
        globally_resolved=False,
        physical_constraints_ok=True,
    )
    observation["last_tool_output"] = {
        "valid_next_actions": [
            {
                "tool": "commit_state",
                "arguments": {"candidate_state_id": "episode:s1"},
            }
        ]
    }

    action, guard = bind_observable_router_action("rollback_state", observation)

    assert action["tool"] == "commit_state"
    assert guard == "verified_candidate_commit_closure"


def test_unbound_advertised_commit_does_not_authorize_candidate() -> None:
    observation = _verified_candidate(
        globally_resolved=False,
        physical_constraints_ok=True,
    )
    observation["last_tool_output"] = {
        "valid_next_actions": [
            {
                "tool": "commit_state",
                "arguments": {"candidate_state_id": "episode:other"},
            }
        ]
    }

    action, guard = bind_observable_router_action("rollback_state", observation)

    assert action["tool"] == "rollback_state"
    assert guard is None


def test_no_candidate_handoff_uses_controller_accepted_request() -> None:
    observation = {
        "active_state_id": "episode:s0",
        "candidate_state_id": None,
        "candidate_lifecycle": "NO_CANDIDATE",
        "has_open_candidate": False,
    }

    action, guard = bind_observable_router_action(
        "ask_for_more_evidence", observation
    )

    assert action == {
        "tool": "ask_for_more_evidence",
        "arguments": {
            "case_path": "episode:s0",
            "request": "operator_escalation:recovery_options_exhausted",
        },
    }
    assert guard is None


def test_correction_arguments_are_copied_only_from_visible_support() -> None:
    observation = {
        "active_state_id": "episode:s0",
        "candidate_state_id": None,
        "candidate_lifecycle": "NO_CANDIDATE",
        "has_open_candidate": False,
        "fresh_context_evidence": {
            "measurement": {
                "supported_corrections": [
                    {
                        "tool": "correct_measurements",
                        "arguments": {
                            "state_id": "episode:s0",
                            "suspect_group": [7],
                        },
                    }
                ]
            }
        },
    }

    action, guard = bind_observable_router_action(
        "correct_measurements_from_path", observation
    )

    assert action == {
        "tool": "correct_measurements_from_path",
        "arguments": {"case_path": "episode:s0", "suspect_group": [7]},
    }
    assert guard is None
