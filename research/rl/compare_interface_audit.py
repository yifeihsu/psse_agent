"""Pair saved evaluations and inspect the student's remaining expert-only roots.

Offline only: consumes the evaluator's existing truth_audited_task_success
flag; never changes eligibility, generates policy targets, or runs training.
Input files are the full bc0_eval.json/r1_eval.json/expert_eval.json reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

COMPARE = "compare_parameter_candidates"
SUCCESS = "truth_audited_task_success"


def indexed_report(payload: Mapping[str, Any], expected_roots: int) -> tuple[dict, dict]:
    report = payload.get("suite_metrics", payload)
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_roots:
        raise ValueError(f"expected exactly {expected_roots} saved episodes")
    indexed = {}
    for episode in episodes:
        root = episode.get("physical_root")
        if not root or root in indexed:
            raise ValueError("missing or duplicated physical_root")
        if not isinstance(episode.get(SUCCESS), bool):
            raise ValueError(f"{root}: missing audited success flag; refusing to infer success")
        if not isinstance(episode.get("trace"), list):
            raise ValueError(f"{root}: full trace required")
        indexed[root] = episode
    return indexed, report.get("configuration", {})


def summary(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    tools = Counter()
    internal = Counter()
    missing = 0
    for episode in episodes:
        tools.update(episode["tool_counts"])
        for step in episode["trace"]:
            if (step.get("action") or {}).get("tool") != COMPARE:
                continue
            metrics = (step.get("policy_tool_output") or {}).get("tool_metrics") or {}
            if "internal_wls_calls" not in metrics or "internal_correction_calls" not in metrics:
                missing += 1
            for key in ("internal_wls_calls", "internal_correction_calls"):
                internal[key] += int(metrics.get(key, 0))
    counts = [int(row["tool_counts"].get(COMPARE, 0)) for row in episodes]
    return {
        "roots": len(episodes), "successes": sum(row[SUCCESS] for row in episodes),
        "invalid_actions": sum(row["invalid_action_count"] for row in episodes),
        "loop_episodes": sum(bool(row["loop_detected"]) for row in episodes),
        "policy_steps": sum(row["policy_steps"] for row in episodes),
        "all_steps": sum(row["steps"] for row in episodes),
        "mean_steps": sum(row["steps"] for row in episodes) / len(episodes) if episodes else None,
        "tool_counts": dict(sorted(tools.items())),
        "comparison_calls": sum(counts), "comparison_episodes": sum(n > 0 for n in counts),
        "comparison_max_per_episode": max(counts, default=0),
        "comparison_internal_wls_calls": internal["internal_wls_calls"],
        "comparison_internal_correction_calls": internal["internal_correction_calls"],
        "comparison_calls_without_internal_counts": missing,
        "false_commits": sum(row["false_commit_count"] for row in episodes),
        "false_rollbacks": sum(row.get("false_rollback_count", 0) for row in episodes),
        "false_finalizations": sum(row["false_finalization_count"] for row in episodes),
        "evaluator_errors": sum(row.get("evaluator_error") is not None for row in episodes),
    }


def trace_digest(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Observable evidence before each policy action, plus recorded outcomes.

    Repetition is an inspection aid, not a replacement for loop_detected. A
    refused precondition does not establish that an acquisition was answered.
    """
    rows, seen = [], set()
    for step in episode["trace"]:
        if step.get("intervention"):
            continue
        observation = step.get("policy_observation") or {}
        action = step.get("action") or {}
        contexts = observation.get("fresh_context_evidence") or {}
        signature = json.dumps({"action": action, "state": step.get("state_before")}, sort_keys=True)
        context_keys = (
            "state_id", "request_attempted", "three_phase_context_status", "harmonic_context_status",
            "nlm_attempted", "nlm_classification", "hif_suspicion_refuted", "harmonic_distortion_detected",
            "parameter_ranking_dominant", "parameter_ranking_dominance_ratio", "best_line_index",
            "ranked_line_indices", "route_status", "route_status_reason",
            "successful", "chi_square_alarm", "normalized_residual_alarm", "hif_screen",
        )
        rows.append({
            "step": step.get("step"), "action": action,
            "execution_status": step.get("execution_status"), "error_code": step.get("error_code"),
            "advanced": step.get("advanced"), "terminal_outcome": step.get("terminal_outcome"),
            "repeated_since_last_advance": signature in seen,
            "active_state_id": observation.get("active_state_id"),
            "candidate_state_id": observation.get("candidate_state_id"),
            "has_open_candidate": observation.get("has_open_candidate"),
            "phase": observation.get("phase"),
            "no_material_anomaly_remaining": observation.get("no_material_anomaly_remaining"),
            "remaining_budget": observation.get("remaining_budget"),
            "error_detail": (step.get("policy_tool_output") or {}).get("error_detail"),
            "unresolved_signatures": observation.get("unresolved_signatures"),
            "contexts_before": {
                family: {**{key: value[key] for key in context_keys if key in value},
                         "supported_correction_count": len(value.get("supported_corrections") or [])}
                for family, value in contexts.items() if isinstance(value, Mapping)
            },
        })
        if step.get("advanced"):
            seen.clear()
        else:
            seen.add(signature)
    return rows


def audit(baseline: Mapping, candidate: Mapping, expert: Mapping, *, expected_roots: int = 209) -> dict:
    reports = {name: indexed_report(payload, expected_roots)
               for name, payload in (("baseline", baseline), ("candidate", candidate), ("expert", expert))}
    old, new, teacher = (reports[name][0] for name in ("baseline", "candidate", "expert"))
    if set(old) != set(new) or set(old) != set(teacher):
        raise ValueError("physical roots differ; do not compare different draws")
    for name, (rows, config) in reports.items():
        for key in ("seed", "max_steps", "action_budget_scope"):
            if key not in config or config[key] != reports["baseline"][1].get(key):
                raise ValueError(f"{name}: missing or mismatched evaluation configuration: {key}")
        for root, row in rows.items():
            for key in ("seed", "scenario_id", "family", "episode_key"):
                if key not in row or row[key] != old[root].get(key):
                    raise ValueError(f"{name}: mismatched {key} at {root}")
    wins = lambda rows: {root for root, row in rows.items() if row[SUCCESS]}
    old_wins, new_wins, expert_wins = wins(old), wins(new), wins(teacher)
    regressions = sorted(old_wins - new_wins)
    expert_only = sorted(expert_wins - new_wins)
    inspected = sorted((set(old) - old_wins) | (set(new) - new_wins))
    common = sorted(old_wins & new_wins)
    return {
        "contract": "comparison_interface_audit_v1", "success_field": SUCCESS,
        "expected_roots": expected_roots,
        "summaries": {name: summary(list(rows.values())) for name, (rows, _) in reports.items()},
        "gained_roots": sorted(new_wins - old_wins), "regressed_roots": regressions,
        "all_candidate_failures": sorted(set(new) - new_wins),
        "baseline_loop_roots": sorted(root for root, row in old.items() if row["loop_detected"]),
        "candidate_loop_roots": sorted(root for root, row in new.items() if row["loop_detected"]),
        "expert_only_success_roots": expert_only,
        "candidate_only_success_roots": sorted(new_wins - expert_wins),
        "both_failed_roots": sorted(set(new) - new_wins - expert_wins),
        "common_success_costs": {"baseline": summary([old[root] for root in common]),
                                 "candidate": summary([new[root] for root in common])},
        "inspection": {root: {name: trace_digest(rows[root]) for name, (rows, _) in reports.items()}
                       for root in inspected},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "expert", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-roots", type=int, default=209)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("baseline", "candidate", "expert")}
    result = audit(**{name: json.loads(path.read_text()) for name, path in paths.items()},
                   expected_roots=args.expected_roots)
    result["inputs"] = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "inspection"}, indent=2))


if __name__ == "__main__":
    main()
