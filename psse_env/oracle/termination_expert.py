from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    FINALIZE_DIAGNOSIS,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.oracle.expert_types import ExpertActionProposal, state_value


class TerminationExpert:
    """Handle mandatory candidate lifecycle and episode-finalization decisions."""

    source_expert = "termination_expert"

    def __init__(self, *, anomaly_threshold: float = 1.0) -> None:
        self.anomaly_threshold = float(anomaly_threshold)

    def verification_actions(self, state: Any) -> list[ExpertActionProposal]:
        candidate_id = state_value(state, "candidate_state_id")
        if not state_value(state, "has_unverified_candidate") or not candidate_id:
            return []
        return [
            ExpertActionProposal(
                action={"tool": RUN_WLS, "arguments": {"state_id": candidate_id}},
                source_expert=self.source_expert,
                confidence=1.0,
                evidence_codes=["candidate_unverified", "transaction_requires_verification"],
                admissible=True,
                estimated_immediate_risk=0.0,
            )
        ]

    def candidate_disposition_actions(self, state: Any) -> list[ExpertActionProposal]:
        candidate_id = state_value(state, "candidate_state_id")
        if not state_value(state, "has_verified_candidate") or not candidate_id:
            return []
        raw_disposition = state_value(state, "candidate_disposition")
        disposition = getattr(raw_disposition, "value", raw_disposition)
        if disposition in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
            return [
                ExpertActionProposal(
                    action={"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}},
                    source_expert=self.source_expert,
                    confidence=1.0,
                    evidence_codes=["candidate_verified", str(disposition).lower()],
                    admissible=True,
                    estimated_immediate_risk=0.0,
                )
            ]
        if disposition == "INCONCLUSIVE":
            if state_value(state, "last_tool") in {ASK_FOR_MORE_EVIDENCE, "run_alternative_test"}:
                return [
                    ExpertActionProposal(
                        action={"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}},
                        source_expert=self.source_expert,
                        confidence=1.0,
                        evidence_codes=["candidate_inconclusive", "additional_evidence_exhausted"],
                        admissible=True,
                        estimated_immediate_risk=0.0,
                    )
                ]
            return [
                ExpertActionProposal(
                    action={"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": candidate_id}},
                    source_expert=self.source_expert,
                    confidence=0.86,
                    evidence_codes=["candidate_verified", "candidate_inconclusive"],
                    admissible=True,
                    estimated_immediate_risk=0.02,
                ),
                ExpertActionProposal(
                    action={"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}},
                    source_expert=self.source_expert,
                    confidence=0.80,
                    evidence_codes=["candidate_verified", "candidate_inconclusive", "safe_rollback"],
                    admissible=True,
                    estimated_immediate_risk=0.0,
                ),
            ]
        return [
            ExpertActionProposal(
                action={"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}},
                source_expert=self.source_expert,
                confidence=1.0,
                evidence_codes=["candidate_verified", "candidate_rejected"],
                admissible=True,
                estimated_immediate_risk=0.0,
            )
        ]

    def propose(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        del history
        if state_value(state, "has_open_candidate"):
            return []
        no_anomaly = bool(state_value(state, "no_material_anomaly_remaining", False))
        score = state_value(state, "remaining_anomaly_score")
        below_threshold = False
        try:
            below_threshold = score is not None and float(score) < self.anomaly_threshold
        except (TypeError, ValueError):
            below_threshold = False
        if not (no_anomaly or below_threshold):
            return []
        evidence = "no_material_anomaly_remaining" if no_anomaly else "anomaly_score_below_threshold"
        return [
            ExpertActionProposal(
                action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                source_expert=self.source_expert,
                confidence=1.0,
                evidence_codes=[evidence, "terminal_condition_met"],
                admissible=True,
                estimated_immediate_risk=0.0,
            )
        ]
