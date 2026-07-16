from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import COMMIT_STATE, ROLLBACK_STATE, RUN_WLS
from psse_env.oracle.expert_types import ExpertActionProposal, state_value
from psse_env.oracle.process_validity import ProcessValidityOracle


class RecoveryExpert:
    """Translate a structured learner/tool failure into legal repair actions."""

    source_expert = "recovery_expert"

    def __init__(self, process_oracle: ProcessValidityOracle | None = None) -> None:
        self.process_oracle = process_oracle or ProcessValidityOracle()

    def propose(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        del history
        output = state_value(state, "last_tool_output", {}) or {}
        error_code = output.get("error_code") if isinstance(output, Mapping) else None
        error_detail = output.get("error_detail") if isinstance(output, Mapping) else None
        last_status = state_value(state, "last_tool_status")

        # Compatibility with the original scaffold, which placed the error
        # name directly in last_tool_status.
        if error_code is None and last_status in {
            "json_parse_error",
            "argument_decode_error",
            "schema_error",
            "missing_precondition",
            "state_reference_mismatch",
            "unknown_state_id",
            "candidate_lifecycle_violation",
            "terminal_condition_not_met",
            "unknown_tool",
            "policy_exception",
        }:
            error_code = str(last_status)
        if not error_code and last_status != "failure":
            return []
        if not error_code:
            error_code = "policy_exception"

        actions = self.process_oracle.repair_actions(state, str(error_code), error_detail)
        if not actions:
            actions = self._safe_fallback_actions(state)
        return [
            ExpertActionProposal(
                action=action,
                source_expert=self.source_expert,
                confidence=1.0,
                evidence_codes=["previous_action_invalid", str(error_code)],
                admissible=True,
                estimated_immediate_risk=0.0,
            )
            for action in actions
        ]

    def repair_actions(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        return self.propose(state, history)

    @staticmethod
    def _safe_fallback_actions(state: Any) -> list[dict[str, Any]]:
        candidate_id = state_value(state, "candidate_state_id")
        if state_value(state, "has_unverified_candidate") and candidate_id:
            return [{"tool": RUN_WLS, "arguments": {"state_id": candidate_id}}]
        if state_value(state, "has_verified_candidate") and candidate_id:
            disposition = getattr(
                state_value(state, "candidate_disposition"),
                "value",
                state_value(state, "candidate_disposition"),
            )
            if disposition in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
                return [{"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}]
            return [{"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}]
        active_id = state_value(state, "active_state_id")
        return [{"tool": RUN_WLS, "arguments": {"state_id": active_id}}] if active_id else []
