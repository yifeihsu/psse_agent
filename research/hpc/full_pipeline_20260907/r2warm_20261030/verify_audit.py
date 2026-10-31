"""Where verify_candidate appears: in the training mixtures (labels) and in a few named episodes (traces)."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

W, cell = Path(sys.argv[1]), Path(sys.argv[2])
out = cell / "out"
spec = importlib.util.spec_from_file_location("r2rerun_tables", W / "r2rerun_tables.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

result = {"mixtures": {}, "traces": {}}
for policy, relative in (("bc0", "d0/aggregate.train_view.jsonl"), ("r1", "r1/collection/round1.train.jsonl"),
                         ("r2", "r2/collection/round1.train.jsonl")):
    labels, verify_states, after_correction = Counter(), Counter(), Counter()
    for line in (out / relative).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        messages = row.get("messages") or []
        state = json.loads(messages[1]["content"]).get("state", {}) if len(messages) > 1 else {}
        calls = (messages[-1].get("tool_calls") or []) if messages else []
        label = calls[0]["function"]["name"] if calls else None
        source = row.get("research_mixture_source") or "d0"
        labels[label] += 1
        if label == "verify_candidate":
            verify_states[f"{source}|{row.get('scenario_family')}|last {state.get('last_tool')}|open {state.get('has_open_candidate')}"] += 1
        if state.get("has_open_candidate") and str(state.get("last_tool")).startswith("correct_"):
            after_correction[f"{state.get('last_tool')} -> {label}"] += 1
    result["mixtures"][policy] = {"labels": dict(labels.most_common()), "verify_candidate_rows": dict(verify_states),
                                  "after_a_correction_with_open_candidate": dict(after_correction.most_common())}
ev = out / "extra" / "evaluations"
wanted = {
    "draw1@40 r2 (not parameter-first)": (out / "r2" / "collection" / "evaluation" / "r1_eval.json", ["r0_ae094a03e446"]),
    "draw2@40 r2 (not parameter-first)": (ev / "test2_40_r2" / "bc0_eval.json", ["r0_a8864571120b"]),
    "hard r2 (untraced violations)": (ev / "hard_r2" / "bc0_eval.json",
                                      ["r0_cb4bb01d5d5c", "r0_fe4374906638", "r0_b071aa01bfb0", "r0_bb0177c569e5"]),
    "hard expert (a refused-escalation root)": (ev / "hard_bc0_r1" / "expert_eval.json", ["r0_0a6aff40f863", "r0_7d938d8f2977"]),
    "draw1@40 expert (not parameter-first roots)": (out / "r1" / "collection" / "evaluation" / "expert_eval.json", ["r0_ae094a03e446"]),
    "draw2@40 expert (not parameter-first roots)": (ev / "test2_40" / "expert_eval.json", ["r0_a8864571120b"]),
}
for label, (path, sids) in wanted.items():
    result["traces"][label] = mod.compact_traces(path, sids, 16)
(W / "verify_audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
print("verify_audit done")
