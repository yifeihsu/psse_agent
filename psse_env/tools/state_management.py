from __future__ import annotations

from typing import Any

from psse_env.state_store import PowerSystemStateStore


def commit_state(store: PowerSystemStateStore, candidate_state_id: str) -> dict[str, Any]:
    active_id = store.commit(candidate_state_id)
    return {
        "execution_status": "success",
        "error_code": None,
        "error_detail": None,
        "state_mutated": True,
        "active_state_id": active_id,
        "candidate_state_id": None,
        "tool_metrics": {},
        "valid_next_actions": [],
    }


def rollback_state(store: PowerSystemStateStore, candidate_state_id: str) -> dict[str, Any]:
    active_id = store.rollback(candidate_state_id)
    return {
        "execution_status": "success",
        "error_code": None,
        "error_detail": None,
        "state_mutated": True,
        "active_state_id": active_id,
        "candidate_state_id": None,
        "tool_metrics": {"rolled_back_state_id": candidate_state_id},
        "valid_next_actions": [],
    }
