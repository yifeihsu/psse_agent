#!/usr/bin/env python3
"""Quarantine forensics for the two failed commit targets (corrected paths)."""
from __future__ import annotations

import json
from pathlib import Path

RUN_ROOT = Path("/scratch/yx3882/dagger1_6b76281c-20260816T035053Z-31541-3666363")
VISITED = RUN_ROOT / "training_beta025.failed-collection" / "diagnostic.all_visited_rows.jsonl"
OUT = RUN_ROOT / "quarantine_forensics"
OUT.mkdir(parents=True, exist_ok=True)

QUARANTINED = {
    "dagger_iter1_r0_9826886a46fd_step17__order126__replica0",
    "dagger_iter1_r0_eab044f4881b_step21__order143__replica0",
}
ROOT_TAGS = {"r0_9826886a46fd", "r0_eab044f4881b"}

rows = []
for line in VISITED.open(encoding="utf-8"):
    r = json.loads(line)
    blob = f"{r.get('example_id')}|{r.get('base_example_id')}|{r.get('episode_id')}|{r.get('scenario_id')}"
    if any(tag in blob for tag in ROOT_TAGS):
        rows.append(r)

rows.sort(key=lambda r: (str(r.get("episode_id") or ""), int(r.get("step") or 0)))
print(f"matched rows: {len(rows)}")
print(f"episodes: {sorted({str(r.get('episode_id')) for r in rows})}\n")

FIELDS = (
    "example_id", "step", "state_origin", "state_class", "recovery_stratum",
    "executed_by", "production_label_eligible", "production_label_ineligibility_reason",
)


def action(a):
    if not isinstance(a, dict):
        return a
    args = a.get("arguments") or {}
    keep = {k: v for k, v in args.items() if k in
            ("state_id", "candidate_state_id", "target", "targets", "element",
             "branch", "bus", "meter", "request", "accepted_target_refinement")}
    return {"tool": a.get("tool"), "args": keep}


for tag in sorted(ROOT_TAGS):
    ep = [r for r in rows if tag in f"{r.get('example_id')}|{r.get('episode_id')}"]
    if not ep:
        continue
    print("=" * 78)
    print(f"### {tag}   ({len(ep)} rows)")
    print("=" * 78)
    for r in ep:
        q = r.get("example_id") in QUARANTINED
        mark = "  <<< QUARANTINED" if q else ""
        obs = r.get("policy_observation") or {}
        aud = r.get("offline_teacher_target_audit")
        print(f"\n-- step {r.get('step')}  {r.get('state_class')} / {r.get('recovery_stratum')}{mark}")
        print(f"   executed_by={r.get('executed_by')}  origin={r.get('state_origin')}")
        print(f"   preferred = {json.dumps(action(r.get('preferred_action')))}")
        print(f"   executed  = {json.dumps(action(r.get('executed_action')))}")
        print(f"   lifecycle={obs.get('candidate_lifecycle')} status={obs.get('candidate_status')} "
              f"verified={obs.get('has_verified_candidate')} unverified={obs.get('has_unverified_candidate')}")
        print(f"   accepted_corrections={json.dumps(obs.get('accepted_corrections'))[:220]}")
        print(f"   explained={json.dumps(obs.get('explained_anomalies'))[:160]} "
              f"remaining_score={obs.get('remaining_anomaly_score')} "
              f"no_material_anomaly={obs.get('no_material_anomaly_remaining')}")
        print(f"   rejected_hypotheses={json.dumps(obs.get('rejected_hypotheses'))[:160]}")
        print(f"   unresolved_signatures={json.dumps(obs.get('unresolved_signatures'))[:200]}")
        if q and isinstance(aud, dict):
            print(f"   AUDIT = {json.dumps(aud, sort_keys=True)[:1500]}")

payload = [{f: r.get(f) for f in FIELDS} | {
    "preferred_action": action(r.get("preferred_action")),
    "executed_action": action(r.get("executed_action")),
    "offline_teacher_target_audit": r.get("offline_teacher_target_audit"),
    "candidate_state_summary": r.get("candidate_state_summary"),
} for r in rows]
(OUT / "quarantined_episodes.compact.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (OUT / "quarantined_episodes.policy_rows.jsonl").open("w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, sort_keys=True) + "\n")
print(f"\n\nwrote {OUT}/quarantined_episodes.compact.json and .policy_rows.jsonl")
