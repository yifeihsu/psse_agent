"""Looks at the breakers and corrections per episode under two evaluations of the same suite.

    python research/hpc/minimal_candidates_20261125/readings/episode_looks.py \\
        output/minimal_candidates_20261125/episodes_base output/minimal_candidates_20261125/episodes_steps17

Each folder holds an evaluation's episode records (`<evaluation dir>/episodes/*.json`, one file a case with every
call the model made).  The first is printed as "supervised", the second as "17 steps".  A look is `inspect` with the
view `breakers`; a look after a correction is one that comes after the episode's first `correct_*` call.

A reading script of the cell, not a tool with tests.
"""
import json, sys
from collections import Counter
from pathlib import Path


def read(directory):
    out = {}
    for path in sorted(Path(directory).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        actions = [step["action"] for step in row["steps"]]
        tools = [a["tool"] for a in actions]
        looks = [i for i, a in enumerate(actions) if a["tool"] == "inspect" and (a.get("arguments") or {}).get("view") == "breakers"]
        corrections = [i for i, t in enumerate(tools) if t.startswith("correct_")]
        first_correction = corrections[0] if corrections else None
        out[row["scenario_id"]] = {
            "class": row["compound_class"], "solved": bool(row["score"]["solved"]), "actions": len(actions),
            "looks": len(looks), "first_is_look": bool(looks and looks[0] == 0),
            "looks_after_a_correction": sum(1 for i in looks if first_correction is not None and i > first_correction),
            "topology": sum(1 for t in tools if t == "correct_topology"), "parameters": sum(1 for t in tools if t == "correct_parameters"),
            "measurements": sum(1 for t in tools if t == "correct_measurements"), "undo": sum(1 for t in tools if t == "undo"),
            "inspect_other": sum(1 for a in actions if a["tool"] == "inspect" and (a.get("arguments") or {}).get("view") != "breakers"),
            "last": tools[-1] if tools else None, "declared": row["score"].get("declared"),
        }
    return out


base, step = read(sys.argv[1]), read(sys.argv[2])
print("episodes", len(base), len(step), "solved", sum(v["solved"] for v in base.values()), sum(v["solved"] for v in step.values()))
keys = ("actions", "looks", "looks_after_a_correction", "topology", "parameters", "measurements", "undo", "inspect_other")


def line(table, cases):
    return (", ".join(f"{key} {sum(table[c][key] for c in cases)}" for key in keys)
            + f"; episodes with a look after a correction {sum(1 for c in cases if table[c]['looks_after_a_correction'])}"
            + f"; first action is the look {sum(1 for c in cases if table[c]['first_is_look'])}")


everything = sorted(base)
groups = {
    "all 300": everything,
    "gained by the 17 steps": [c for c in everything if not base[c]["solved"] and step[c]["solved"]],
    "lost": [c for c in everything if base[c]["solved"] and not step[c]["solved"]],
    "unsolved under both": [c for c in everything if not base[c]["solved"] and not step[c]["solved"]],
    "solved under both": [c for c in everything if base[c]["solved"] and step[c]["solved"]],
}
for label, cases in groups.items():
    print(f"{label}: {len(cases)} cases")
    print("    supervised:", line(base, cases))
    print("    17 steps:  ", line(step, cases))
    moved = Counter((base[c]["looks_after_a_correction"] > 0, step[c]["looks_after_a_correction"] > 0) for c in cases)
    print("    a look after a correction (supervised, 17 steps):", dict(moved))
two = [c for c in everything if base[c]["class"].startswith("cb2")]
with_breaker = [c for c in everything if "cb" in base[c]["class"]]
for label, cases in (("two-breaker classes", two), ("classes with a breaker error", with_breaker)):
    print(f"{label}: {len(cases)} cases; solved {sum(base[c]['solved'] for c in cases)} -> {sum(step[c]['solved'] for c in cases)}")
    print("    supervised:", line(base, cases))
    print("    17 steps:  ", line(step, cases))
print("classes of the gained cases:", dict(Counter(base[c]["class"] for c in groups["gained by the 17 steps"])))
print("classes of the lost cases:", dict(Counter(base[c]["class"] for c in groups["lost"])))
print("last tool of unsolved episodes, supervised:", dict(Counter(base[c]["last"] for c in everything if not base[c]["solved"])),
      "| 17 steps:", dict(Counter(step[c]["last"] for c in everything if not step[c]["solved"])))
