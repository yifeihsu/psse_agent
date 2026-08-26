"""Model-observation identifiability audit (review point 4).

Renders every collected observation the way the policy sees it, canonicalizes
episode-specific identifiers, hashes the result, and asks: do apparently
identical observations ever carry different expert targets or different
candidate-lifecycle states?  A nonzero target-conflict rate means the expert's
action is not a well-defined function of the model-visible state.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict

SCRATCH = (
    r"C:/Users/Holiday/AppData/Local/Temp/claude/"
    r"C--Users-Holiday-Documents-ChatGPT-PSSE-Agent/"
    r"92729157-788a-4c24-ac77-8b1bd2e55183/scratchpad"
)

EPISODE_ID = re.compile(r"r0_[0-9a-f]+_episode\d+")


def canonicalize(value):
    if isinstance(value, str):
        return EPISODE_ID.sub("EP", value)
    if isinstance(value, dict):
        return {k: canonicalize(v) for k, v in sorted(value.items()) if k != "episode_id"}
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    return value


def main() -> int:
    rows = []
    for name in ("round1_rows_big.jsonl", "round2_rows.jsonl"):
        with open(f"{SCRATCH}/research/{name}", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)

    groups = defaultdict(lambda: {"targets": set(), "lifecycles": set(), "n": 0})
    skipped = 0
    for row in rows:
        observation = row.get("policy_observation")
        target = row.get("preferred_action")
        if not isinstance(observation, dict) or not target:
            skipped += 1
            continue
        key = hashlib.sha256(
            json.dumps(canonicalize(observation), sort_keys=True).encode()
        ).hexdigest()
        group = groups[key]
        group["n"] += 1
        group["targets"].add(json.dumps(canonicalize(target), sort_keys=True))
        group["lifecycles"].add(str(observation.get("candidate_lifecycle")))

    total_rows = sum(g["n"] for g in groups.values())
    dup_groups = [g for g in groups.values() if g["n"] > 1]
    dup_rows = sum(g["n"] for g in dup_groups)
    conflict_groups = [g for g in dup_groups if len(g["targets"]) > 1]
    conflict_rows = sum(g["n"] for g in conflict_groups)
    lifecycle_mixed = [g for g in dup_groups if len(g["lifecycles"]) > 1]

    print(f"rows audited: {total_rows} (skipped {skipped})")
    print(f"distinct canonical observations: {len(groups)}")
    print(f"observations seen >1x: {len(dup_groups)} groups / {dup_rows} rows")
    print(f"TARGET-CONFLICT groups: {len(conflict_groups)} / rows {conflict_rows}")
    if dup_rows:
        print(f"target-conflict rate among repeated observations: {conflict_rows/dup_rows:.2%}")
    print(f"lifecycle-mixed duplicate groups: {len(lifecycle_mixed)}")

    # what do conflicts look like?
    shown = 0
    for key, g in groups.items():
        if len(g["targets"]) > 1 and shown < 5:
            tools = Counter(json.loads(t)["tool"] for t in g["targets"])
            print(f"  conflict group n={g['n']} target-tools={dict(tools)} lifecycles={g['lifecycles']}")
            shown += 1

    # precondition reconstructability: can model-visible fields predict when
    # commit/rollback are the expert target while a candidate is open?
    by_lifecycle = defaultdict(Counter)
    for row in rows:
        obs = row.get("policy_observation") or {}
        pref = (row.get("preferred_action") or {}).get("tool")
        if pref:
            by_lifecycle[str(obs.get("candidate_lifecycle"))][pref] += 1
    print("expert target tool by candidate_lifecycle (top 3 each):")
    for lc, counts in sorted(by_lifecycle.items()):
        top = ", ".join(f"{t}:{n}" for t, n in counts.most_common(3))
        print(f"  {lc:<22} {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
