"""The hypothesis ledger (step 3): screen-ranked hypotheses, tested status, budget, proposal reranking."""
from __future__ import annotations

from psse_env.actions import CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.hypothesis_ledger import (
    FAMILY_BOOST, TARGET_BOOST, hypothesis_ledger, rerank_proposals, screen_hypotheses, screen_targets,
)

ACTIVE = "s0"


def _report(accepted, scores, best, ranked, suspected=False):
    return {
        "status": "valid", "suspected": suspected, "outcome": ">".join(a["class"] for a in accepted) or None,
        "accepted_hypotheses": accepted,
        "rounds": [{"winner": max(scores, key=scores.get), "scores": scores, "best": best, "ranked": ranked,
                    "set_aside_channels": []}],
    }


def _policy(report, rejected=(), accepted=()):
    return {
        "evidence_profile": "suspicion_gated_diagnostics", "active_state_id": ACTIVE, "candidate_state_id": None,
        "has_open_candidate": False,
        "fresh_context_evidence": {"wls": {"state_id": ACTIVE, "state_hash": "h", "successful": True,
                                           "chi_square_alarm": True, "normalized_residual_alarm": True,
                                           "hif_screen": report}},
        "rejected_hypotheses": list(rejected), "accepted_corrections": list(accepted),
    }


def _rejected(tool, **arguments):
    return {"candidate_state_id": "c1", "candidate_parent_id": ACTIVE,
            "source_action": {"tool": tool, "arguments": {"state_id": ACTIVE, **arguments}},
            "verification_summary": {"global_progress": 0.01}}


MIXED = _report(
    accepted=[{"class": "parameter", "branch_row0": 12, "parameter": "RX", "estimate": {"R": 0.1, "X": 0.2}},
              {"class": "meter", "channel_index0": 71}],
    scores={"meter": 40.0, "parameter": 12.0, "topology": -3.0, "hif": -20.0},
    best={"meter": {"channel_index0": 71, "J": 100.0}, "parameter": {"branch_row0": 12, "parameter": "RX", "J": 120.0},
          "topology": {"branch_row0": 5, "J": 140.0}, "hif": {"branch_row0": 3, "alpha_from_from_bus": 0.5, "J": 150.0}},
    ranked={"meter": [{"channel_index0": 71, "J": 100.0}, {"channel_index0": 40, "J": 130.0}],
            "parameter": [{"branch_row0": 12, "parameter": "RX", "J": 120.0}, {"branch_row0": 13, "parameter": "X", "J": 125.0}],
            "topology": [{"branch_row0": 5, "J": 140.0}], "hif": [{"branch_row0": 3, "J": 150.0}]},
)


def test_hypotheses_follow_the_accepted_sequence_then_the_scores():
    hypotheses = screen_hypotheses(MIXED)
    assert [h["family"] for h in hypotheses] == ["parameter", "measurement", "topology", "hif"]
    assert hypotheses[0]["source"] == "accepted" and hypotheses[0]["target"] == 12
    assert hypotheses[1]["target"] == 71 and hypotheses[2]["source"] == "last_round"
    assert screen_targets(MIXED, "parameter") == [12, 13]
    assert screen_targets(MIXED, "measurement") == [71, 40]
    assert screen_hypotheses({"status": "screen_error"}) == []


def test_ledger_statuses_and_budget():
    ledger = hypothesis_ledger(_policy(MIXED))
    assert ledger["leading_family"] == "parameter"
    assert {h["family"]: h["status"] for h in ledger["hypotheses"]}["parameter"] == "untested"
    one_rejected = _policy(MIXED, rejected=[_rejected(CORRECT_PARAMETERS, line_index=13)])
    ledger = hypothesis_ledger(one_rejected)
    assert ledger["tested"]["parameter"]["rejected"] == [12]  # line_index 13 is branch row 12
    assert {h["family"]: h["status"] for h in ledger["hypotheses"]}["parameter"] == "rejected"
    assert ledger["leading_family"] == "measurement"
    two_rejected = _policy(MIXED, rejected=[_rejected(CORRECT_PARAMETERS, line_index=13),
                                            _rejected(CORRECT_PARAMETERS, line_index=14)])
    ledger = hypothesis_ledger(two_rejected)
    assert ledger["exhausted_families"] == ["parameter"]
    failure = dict(_rejected(CORRECT_PARAMETERS, line_index=13), rejection_kind="executor_failure")
    ledger = hypothesis_ledger(_policy(MIXED, rejected=[failure]))
    assert ledger["tested"]["parameter"] == {"rejected": [], "failed": [12], "accepted": []}
    assert ledger["exhausted_families"] == [] and ledger["leading_family"] == "parameter"


def _proposals():
    return [
        ExpertActionProposal(action={"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}},
                             source_expert="measurement_expert", confidence=0.9),
        ExpertActionProposal(action={"tool": GET_PARAMETER_CONTEXT, "arguments": {"state_id": ACTIVE}},
                             source_expert="parameter_expert", confidence=0.87),
        ExpertActionProposal(action={"tool": CORRECT_PARAMETERS, "arguments": {"state_id": ACTIVE, "line_index": 14}},
                             source_expert="parameter_expert", confidence=0.98),
        ExpertActionProposal(action={"tool": CORRECT_PARAMETERS, "arguments": {"state_id": ACTIVE, "line_index": 13}},
                             source_expert="parameter_expert", confidence=0.98),
        ExpertActionProposal(action={"tool": "run_wls", "arguments": {"state_id": ACTIVE}},
                             source_expert="measurement_expert", confidence=0.5),
    ]


def test_rerank_boosts_the_leading_family_and_its_top_target():
    reranked = rerank_proposals(_proposals(), _policy(MIXED))
    by_action = {(p.action["tool"], p.action["arguments"].get("line_index")): p for p in reranked}
    assert by_action[(GET_PARAMETER_CONTEXT, None)].confidence == 0.87 + FAMILY_BOOST
    assert by_action[(GET_MEASUREMENT_CONTEXT, None)].confidence == 0.9
    top = by_action[(CORRECT_PARAMETERS, 13)]
    other = by_action[(CORRECT_PARAMETERS, 14)]
    assert top.confidence == 0.98 + FAMILY_BOOST + TARGET_BOOST and "ledger_top_target" in top.evidence_codes
    assert other.confidence == 0.98 + FAMILY_BOOST + 0.005
    assert by_action[("run_wls", None)].confidence == 0.5


def test_rerank_drops_an_exhausted_family_and_passes_through_without_a_screen():
    exhausted = _policy(MIXED, rejected=[_rejected(CORRECT_PARAMETERS, line_index=13),
                                         _rejected(CORRECT_PARAMETERS, line_index=14)])
    reranked = rerank_proposals(_proposals(), exhausted)
    tools = [p.action["tool"] for p in reranked]
    assert CORRECT_PARAMETERS not in tools and GET_PARAMETER_CONTEXT not in tools
    assert GET_MEASUREMENT_CONTEXT in tools and "run_wls" in tools
    assert [p.confidence for p in rerank_proposals(_proposals(), _policy({"status": "screen_error"}))] == \
        [p.confidence for p in _proposals()]
    reranked = rerank_proposals(_proposals(), _policy(MIXED, rejected=[
        _rejected(CORRECT_MEASUREMENTS, suspect_group=[71]), _rejected(CORRECT_MEASUREMENTS, suspect_group=[40]),
        _rejected(CORRECT_PARAMETERS, line_index=20), _rejected(CORRECT_PARAMETERS, line_index=19),
    ]))
    assert all(p.action["tool"] not in (CORRECT_PARAMETERS, CORRECT_MEASUREMENTS) for p in reranked)
