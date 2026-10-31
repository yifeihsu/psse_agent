"""Per set and policy: episodes with a verify_candidate call, a context read refused while a candidate is open,
and an operator escalation refused after the final acceptance."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

W, cell = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("r2rerun_tables", W / "r2rerun_tables.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = {}
for name, entry in mod.sources(cell, "", "_pool_20261023", "_bc0init_20261030").items():
    for policy, path in entry["files"].items():
        if not path.is_file():
            continue
        counts, examples = Counter(), {}
        for episode in mod.tables._episodes(json.loads(path.read_text(encoding="utf-8"))):
            trace = episode.get("trace") or []
            tools = [((s.get("action") or {}).get("tool"), s.get("error_code")) for s in trace]
            modes = set()
            if any(t == "verify_candidate" for t, _ in tools):
                modes.add("verify_candidate call")
            if any(str(t).startswith("get_") and e == "candidate_lifecycle_violation" for t, e in tools):
                modes.add("context read refused, candidate open")
            if any(t == "ask_for_more_evidence" and e == "operator_escalation_precondition_not_met" for t, e in tools):
                modes.add("escalation refused, precondition not met")
            for m in modes:
                counts[m] += 1
                examples.setdefault(m, str(episode.get("scenario_id")))
        result[f"{name}|{policy}"] = {"counts": dict(counts), "examples": examples}
(W / "mode_counts.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
print("mode_counts done")
