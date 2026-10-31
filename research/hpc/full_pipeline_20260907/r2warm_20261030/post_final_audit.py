"""What each policy does after its final acceptance, on every set of the R1-start round-2 comparison."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

W = Path(sys.argv[1])
cell = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("r2rerun_tables", W / "r2rerun_tables.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def label(step):
    action = step.get("action") or {}
    tool = action.get("tool")
    args = action.get("arguments") or {}
    text = tool
    if tool == "ask_for_more_evidence":
        text += ":" + str(args.get("request"))
    if step.get("execution_status") != "success":
        text += "!" + str(step.get("error_code"))
    return text

result = {}
for name, entry in mod.sources(cell, "", "_pool_20261023", "_bc0init_20261030").items():
    for policy, path in entry["files"].items():
        if not path.is_file():
            continue
        signatures, refused, examples = Counter(), Counter(), {}
        for episode in mod.tables._episodes(json.loads(path.read_text(encoding="utf-8"))):
            trace = episode.get("trace") or []
            finals = [i for i, s in enumerate(trace) if (s.get("action") or {}).get("tool") == "commit_state"
                      and s.get("candidate_disposition_offline") == "ACCEPT_FINAL"]
            for s in trace:
                if (s.get("action") or {}).get("tool") == "ask_for_more_evidence" and s.get("execution_status") != "success":
                    refused[str(s.get("error_code"))] += 1
            if not finals:
                signatures["no ACCEPT_FINAL"] += 1
                continue
            i = finals[-1]
            before = label(trace[i - 1]) if i > 0 else "-"
            after = " > ".join(label(s) for s in trace[i + 1:]) or "(end)"
            sig = f"via {before} | {after}"
            signatures[sig] += 1
            examples.setdefault(sig, str(episode.get("scenario_id")))
        result[f"{name}|{policy}"] = {"signatures": dict(signatures.most_common()), "refused_evidence_requests": dict(refused),
                                      "examples": examples}
        print(f"done {name} {policy}", flush=True)
(W / "post_final_audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
lines = []
for key, value in result.items():
    lines.append(f"## {key}")
    for sig, n in value["signatures"].items():
        lines.append(f"- {n}: {sig} (e.g. {value['examples'].get(sig, '')})")
    lines.append(f"- refused ask_for_more_evidence: {value['refused_evidence_requests']}")
(W / "post_final_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
