#!/usr/bin/env python3
"""Reproduce every figure in the DAgger-1 round-2 failure diagnosis.

Run on the Torch login node (paths below are cluster-absolute):

    python3 reproduce_analysis.py > analysis_output.txt

Reads only; writes nothing. Streams the large JSONL files, so memory stays flat.
"""
from __future__ import annotations

import collections
import json
import os

# --- Inputs -----------------------------------------------------------------
RUN = "/scratch/yx3882/dagger1_6b76281c-20260816T035053Z-31541-3666363"
FAILED = f"{RUN}/training_beta025.failed-collection"
EVIDENCE = f"{FAILED}/failure_evidence.json"
VISITED = f"{FAILED}/diagnostic.all_visited_rows.jsonl"
D0 = "/scratch/yx3882/round0_aggregate_6b76281_20260815T143201Z_442254_8568"
D0_RAW = f"{D0}/aggregate.raw.jsonl"
D0_CF = f"{D0}/aggregate.auxiliary_counterfactual.raw.jsonl"

# Predicate constants mirrored from psse_env/dagger/rollout_collector.py
CORRECTION_TOOLS = {
    "correct_measurements", "correct_measurements_from_path",
    "correct_parameters", "correct_parameters_from_path",
    "correct_topology", "correct_topology_from_path",
}
UNSUPPORTED_CODES = {
    "correction_not_supported_by_current_context",
    "correction_route_not_actionable",
    "parameter_scans_missing",
    "post_correction_confirmation_required",
}
UNCLASSIFIED = {None, "", "unclassified"}


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def counter(title: str, c: collections.Counter, n: int = 20) -> None:
    print(f"\n-- {title}")
    for key, val in c.most_common(n):
        print(f"   {val:6d}  {key}")


def previous_failed(obs: dict) -> tuple[bool, str | None]:
    """Exact mirror of rollout_collector._observable_last_failure."""
    out = obs.get("last_tool_output")
    out = out if isinstance(out, dict) else {}
    failed = obs.get("last_tool_status") == "failure" or (
        out.get("execution_status") == "failure" or out.get("error_code") is not None
    )
    return bool(failed), (out.get("error_code") or None)


# --- 1. Gate outcomes -------------------------------------------------------
def section_gates() -> None:
    rule("1. GATE OUTCOMES  (failure_evidence.json)")
    d = json.load(open(EVIDENCE))
    for key in (
        "collection_outcome", "collection_pass", "beta", "iteration", "seed",
        "strict_gate_requested", "strict_gate_passed", "expected_exit_code",
        "production_outputs_published", "training_eligible",
        "round1_aggregate_eligible", "release_evidence_eligible",
        "visited_rows", "candidate_recovery_row_count", "selected_recovery_row_count",
        "d0_physical_root_count", "eligible_physical_root_count",
        "forbidden_physical_root_count", "frozen_evaluation_root_count",
        "development_holdout_root_count",
    ):
        print(f"   {key:38s} {d.get(key)}")

    print(f"\n   failed_gate_names ({len(d['failed_gate_names'])}):")
    for name in d["failed_gate_names"]:
        print(f"      - {name}")

    print("\n-- independent_root_support: per-stratum distinct physical roots")
    strata = d["independent_root_support"]["recovery_strata"]
    print(f"   {'stratum':46s} {'roots':>6s} {'floor':>6s} {'rows':>6s}  required")
    for name, info in sorted(strata.items()):
        print(
            f"   {name:46s} {info['distinct_physical_roots']:6d} "
            f"{info['minimum_distinct_physical_roots']:6d} "
            f"{info['target_bearing_rows']:6d}  {info['required_for_release']}"
            f"{'' if info['passed'] else '   <-- SHORTFALL'}"
        )

    print("\n-- targeted_state_coverage: per-cell distinct physical roots")
    tsc = d["targeted_state_coverage"]
    got, need = tsc["distinct_physical_roots_by_cell"], tsc["minimum_distinct_physical_roots_by_cell"]
    for cell in sorted(got):
        flag = "   <-- SHORTFALL" if got[cell] < need.get(cell, 0) else ""
        print(f"   {cell:46s} {got[cell]:6d} {need.get(cell, 0):6d}{flag}")

    print("\n-- round1_replay_capacity")
    cap = d["round1_replay_capacity"]
    for label in ("requested", "largest_feasible"):
        block = cap[label]
        print(f"   {label:18s} d1={block.get('d1_recovery_rows')} "
              f"d0={block.get('d0_bc0_rows')} total={block.get('total_rows')} "
              f"share={block.get('observed_d1_share')}")
    print(f"   band: [{cap['minimum_d1_share']}, {cap['maximum_d1_share']}]  "
          f"configured={cap['configured_d1_share']}  "
          f"max_duplicate={cap['max_duplicate_count']}  max_rows_per_root={cap['max_rows_per_root']}")
    print(f"   NOTE: largest_feasible share sits INSIDE the band; the gate fails only")
    print(f"         because the requested total (1630) exceeds feasible (1521).")

    print("\n-- offline_teacher_target_quarantine_summary  (zero-tolerance gate)")
    q = d["offline_teacher_target_quarantine_summary"]
    for key in ("total_rows", "candidate_rows", "passed_rows", "quarantined_rows",
                "quarantined_by_action_class", "quarantined_by_reason_code", "passed"):
        print(f"   {key:34s} {q.get(key)}")


# --- 2. Stratum yield and the unclassified mass -----------------------------
def section_visited() -> None:
    rule("2. VISITED-ROW ANALYSIS  (diagnostic.all_visited_rows.jsonl)")
    strat = collections.Counter()
    elig = collections.Counter()
    grid = collections.defaultdict(collections.Counter)
    u_reason = collections.Counter()
    u_origin = collections.Counter()
    detail = collections.Counter()
    combo = collections.Counter()
    total = 0

    for line in open(VISITED):
        r = json.loads(line)
        total += 1
        st = r.get("recovery_stratum")
        strat[st] += 1
        elig[r.get("production_label_eligible")] += 1
        grid[r.get("state_class")][st or "UNCLASSIFIED"] += 1
        if st in UNCLASSIFIED:
            u_reason[r.get("production_label_ineligibility_reason")] += 1
            u_origin[r.get("state_origin")] += 1
            detail[(r.get("state_class"), r.get("state_origin"),
                    r.get("production_label_ineligibility_reason"))] += 1
            if (r.get("state_class") == "invalid_precondition_recovery"
                    and r.get("state_origin") == "learner_policy"):
                obs = r.get("policy_observation") or {}
                pa = r.get("preferred_action") or {}
                failed, _ = previous_failed(obs)
                tl = r.get("transition_label") or {}
                has_cand = bool(obs.get("has_open_candidate") or obs.get("candidate_state_id"))
                combo[(pa.get("tool"), failed, tl.get("process_valid"), has_cand,
                       obs.get("candidate_disposition"))] += 1

    print(f"\n   total rows: {total}")
    counter("recovery_stratum", strat)
    counter("production_label_eligible", elig)
    counter("UNCLASSIFIED -> state_origin", u_origin)
    counter("UNCLASSIFIED -> ineligibility reason", u_reason)

    print("\n-- state_class -> recovery_stratum")
    for sc, c in sorted(grid.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"\n   {sc}  (total {sum(c.values())})")
        for k, v in c.most_common():
            print(f"      {v:6d}  {k}")

    print("\n-- UNCLASSIFIED breakdown (state_class, origin, reason)")
    for k, v in detail.most_common():
        print(f"   {v:6d}  state_class={k[0]}  origin={k[1]}  reason={k[2]}")

    print("\n-- The 161 learner-visited invalid_precondition_recovery rows")
    print("   (preferred.tool, previous_failed, process_valid, has_candidate, disposition)")
    for k, v in combo.most_common():
        print(f"   {v:6d}  {k}")
    print("\n   Every row has previous_failed=False, so classify_dagger1_recovery_stratum")
    print("   falls through to `return None`. The 159 commit_state rows reach")
    print("   state_class=invalid_precondition_recovery only because `disposition` is")
    print("   None from every source -- the catch-all branch, not a real invalidity.")


# --- 3. D0 round-0 aggregate ------------------------------------------------
def section_d0() -> None:
    rule("3. D0 ROUND-0 AGGREGATE: do the starved situations exist there?")

    branch = collections.Counter()
    codes = collections.Counter()
    tool_code = collections.Counter()
    roots_unsupported: set[str] = set()
    roots_postfail: set[str] = set()
    n_cf = 0
    for line in open(D0_CF):
        r = json.loads(line)
        n_cf += 1
        branch[r.get("branch_family")] += 1
        out = r.get("injected_tool_output") or {}
        code = out.get("error_code")
        codes[code] += 1
        tool = (r.get("injected_action") or {}).get("tool")
        obs = r.get("policy_observation") or {}
        has_cand = bool(obs.get("has_open_candidate") or obs.get("candidate_state_id"))
        failed = out.get("execution_status") == "failure" or code is not None
        root = r.get("physical_root_fingerprint")
        if tool in CORRECTION_TOOLS and code in UNSUPPORTED_CODES:
            roots_unsupported.add(root)
            tool_code[(tool, code)] += 1
        if failed and not has_cand:
            roots_postfail.add(root)

    print(f"\n   aggregate.auxiliary_counterfactual.raw.jsonl: {n_cf} rows")
    counter("branch_family", branch)
    counter("injected error_code", codes)
    counter("(injected tool, unsupported code)", tool_code)
    print(f"\n   unsupported_correction_recovery predicate -> "
          f"{len(roots_unsupported)} distinct roots  (floor 10)")
    print(f"   post_failure_no_candidate predicate       -> "
          f"{len(roots_postfail)} distinct roots  (floor 10)")

    raw_unsup: set[str] = set()
    raw_postfail: set[str] = set()
    n_raw = 0
    for line in open(D0_RAW):
        r = json.loads(line)
        n_raw += 1
        obs = r.get("policy_observation") or {}
        failed, code = previous_failed(obs)
        has_cand = bool(obs.get("has_open_candidate") or obs.get("candidate_state_id"))
        root = r.get("physical_root_fingerprint")
        if failed and obs.get("last_tool") in CORRECTION_TOOLS and code in UNSUPPORTED_CODES:
            raw_unsup.add(root)
        if failed and not has_cand:
            raw_postfail.add(root)

    print(f"\n   aggregate.raw.jsonl: {n_raw} ordinary rows")
    print(f"   unsupported_correction_recovery predicate -> {len(raw_unsup)} distinct roots")
    print(f"   post_failure_no_candidate predicate       -> {len(raw_postfail)} distinct roots")
    print("\n   CONCLUSION: both situations exist ONLY in the injected-error")
    print("   counterfactual stream, never in ordinary expert/model rollouts.")
    print("   D0's roots are forbidden to DAgger-1, so the mechanism must be")
    print("   ported to fresh roots -- the rows themselves cannot be reused.")


if __name__ == "__main__":
    for path in (EVIDENCE, VISITED, D0_RAW, D0_CF):
        if not os.path.exists(path):
            raise SystemExit(f"missing input: {path}")
    section_gates()
    section_visited()
    section_d0()
    print("\n[done]")
