#!/usr/bin/env python3
"""Case A vs Case B: was the quarantined commit a wrong target, or a refinement?"""
from __future__ import annotations
import json
from pathlib import Path

V = Path("/scratch/yx3882/dagger1_6b76281c-20260816T035053Z-31541-3666363"
         "/training_beta025.failed-collection/diagnostic.all_visited_rows.jsonl")
Q = {"dagger_iter1_r0_9826886a46fd_step17__order126__replica0": "r0_9826886a46fd_episode126",
     "dagger_iter1_r0_eab044f4881b_step21__order143__replica0": "r0_eab044f4881b_episode143"}
EPS = set(Q.values())

by_ep = {e: [] for e in EPS}
for line in V.open(encoding="utf-8"):
    r = json.loads(line)
    ep = str(r.get("episode_id") or "")
    if ep in EPS:
        by_ep[ep].append(r)
for e in by_ep:
    by_ep[e].sort(key=lambda r: int(r.get("step") or 0))


def src(entry):
    a = (entry or {}).get("source_action") or {}
    return {"tool": a.get("tool"), "args": a.get("arguments")}


for exid, ep in Q.items():
    rows = by_ep[ep]
    qrow = next((r for r in rows if r.get("example_id") == exid), None)
    print("=" * 78)
    print(f"### {exid}")
    print(f"### episode {ep} — {len(rows)} rows, quarantined at step {qrow.get('step') if qrow else '?'}")
    print("=" * 78)
    if qrow is None:
        print("  !! quarantined row not found by example_id")
        continue
    obs = qrow.get("policy_observation") or {}

    print("\n--- THE COMMIT BEING AUDITED ---")
    print(f"preferred_action = {json.dumps(qrow.get('preferred_action'), sort_keys=True)}")
    print(f"executed_action  = {json.dumps(qrow.get('executed_action'), sort_keys=True)}")
    print(f"executed_by      = {qrow.get('executed_by')}   state_class = {qrow.get('state_class')}")
    print(f"lifecycle={obs.get('candidate_lifecycle')} status={obs.get('candidate_status')}")
    print(f"no_material_anomaly_remaining = {obs.get('no_material_anomaly_remaining')}")
    print(f"remaining_anomaly_score       = {obs.get('remaining_anomaly_score')}")

    print("\n--- CANDIDATE UNDER COMMIT (what correction created it) ---")
    print(json.dumps(qrow.get("candidate_state_summary"), indent=1, sort_keys=True)[:1800])

    print("\n--- ACCEPTED CORRECTIONS ALREADY ON THE LEDGER AT THIS STEP ---")
    acc = obs.get("accepted_corrections") or []
    print(f"count = {len(acc)}")
    for i, a in enumerate(acc):
        print(f"  [{i}] parent={a.get('candidate_parent_id')} -> {a.get('candidate_state_id')}")
        print(f"      source={json.dumps(src(a), sort_keys=True)}")

    print("\n--- OFFLINE TEACHER TARGET AUDIT (private) ---")
    print(json.dumps(qrow.get("offline_teacher_target_audit"), indent=1, sort_keys=True)[:2500])

    print("\n--- PRIOR CORRECTION ACTIONS IN THIS EPISODE ---")
    for r in rows:
        pa = r.get("preferred_action") or {}
        if str(pa.get("tool", "")).startswith("correct_"):
            args = pa.get("arguments") or {}
            slim = {k: v for k, v in args.items() if k != "state_id"}
            print(f"  step {r.get('step'):>3} {pa.get('tool')}  {json.dumps(slim, sort_keys=True)[:200]}")
    print("\n")
