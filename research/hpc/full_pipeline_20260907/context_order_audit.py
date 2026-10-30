#!/usr/bin/env python3
"""Context order on the breaker roots of the conventional-only cell (note section 9.13).

    context_order_audit.py --cell-dir PIPE [--output OUT.json]

On a root with a breaker-status error (the topology and meter + breaker
families) the teacher calls ``run_wls``, reads the parameter context, finds
the parameter route unavailable and reads the topology context next.  This
audit reads every recorded trace of the cell and counts, per evaluation set
and policy, where the policy goes after the parameter context (the topology
context, or the measurement context first: a detour) and what it does from
the state with the parameter and measurement contexts fresh and no topology
context.  It counts the training rows at those two decision states in each
training view (D0's, which trained BC0, and the round mixtures of R1, the
recorded R2 and the rerun's R2), and the learner's and the teacher's action
at them in the two round ledgers (BC0 collecting round 1, R1 round 2), with
the chi-square ratio at the decision on the training states and on the test
roots (both draws at 40 actions, and the new R2's detours among them).

Policies: ``expert``, ``bc0``, ``r1``, ``r2`` (the R2 in ``out/r2``), ``r2_pool``
(the recorded R2, set aside under ``_pool_20261023``), ``r2_bc0init`` (the
rerun's R2 trained from BC0, set aside under ``_bc0init_20261030`` when round 2
was trained again from R1, note section 10; ``r2`` is then the R1-start R2) and
``control`` (the matched cloning control).  Evaluation slots are mapped to
policies by their adapter path in each ``policies.json``; the round evaluations
of the first draw at 40 actions have none and are named here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
BREAKER = ("topology", "measurement+topology")
POOL = "_pool_20261023"
BC0INIT = "_bc0init_20261030"
REFUSED = "correction_not_supported_by_current_context"
POLICIES = ("expert", "bc0", "r1", "r2_pool", "r2_bc0init", "r2", "control")
PAIRS = (("r1", "r2"), ("bc0", "r2"), ("r2_pool", "r2"), ("r2_bc0init", "r2"), ("expert", "r2"), ("bc0", "r1"),
         ("r1", "r2_bc0init"), ("bc0", "r2_bc0init"))
DETOURING = ("r2_bc0init", "r2")
MIXTURES = (("bc0", "d0/aggregate.train_view.jsonl"),
            ("r1", "r1/collection/round1.train.jsonl"),
            ("r2_pool", f"r2{POOL}/collection/round1.train.jsonl"),
            ("r2_bc0init", f"r2{BC0INIT}/collection/round1.train.jsonl"),
            ("r2", "r2/collection/round1.train.jsonl"))
LEDGERS = (("round 1, BC0 collecting", "r1/collection/d1.all.jsonl"),
           ("round 2, R1 collecting", "r2/collection/d1.all.jsonl"))
SET_ORDER = ("draw1@40", "draw2@40", "draw1@12", "draw2@12", "draw1@16", "draw1@20", "draw2@20",
             "validation", "hard")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, HERE / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tables = _load("followup_tables.py")


def _group(family: Any) -> str:
    if family in BREAKER:
        return "breaker"
    return "multi_meter" if family == "multi_measurement" else "other"


def _policy(adapter: str, directory: str) -> str:
    if adapter.endswith("/bc0/lora"):
        return "bc0"
    if adapter.endswith("/r1/training/lora"):
        return "r1"
    if adapter.endswith("/r2/training/lora"):
        return "r2_pool" if POOL in directory else "r2_bc0init" if BC0INIT in directory else "r2"
    if adapter.endswith("/control/training/lora"):
        return "control"
    return adapter


def _set_name(directory: str) -> str:
    base = directory.split("_r2")[0].replace("_bc0_r1", "")
    if base.startswith("test2_budget"):
        return "draw2@" + base[len("test2_budget"):]
    if base.startswith("test_budget"):
        return "draw1@" + base[len("test_budget"):]
    return "draw2@40" if base == "test2_40" else base


def evaluation_files(out: Path) -> list[tuple[str, str, Path]]:
    files = [("draw1@40", "expert", out / "r1/collection/evaluation/expert_eval.json"),
             ("draw1@40", "bc0", out / "r1/collection/evaluation/bc0_eval.json"),
             ("draw1@40", "r1", out / "r1/collection/evaluation/r1_eval.json"),
             ("draw1@40", "r2", out / "r2/collection/evaluation/r1_eval.json"),
             ("draw1@40", "r2_pool", out / f"r2{POOL}/collection/evaluation/r1_eval.json"),
             ("draw1@40", "r2_bc0init", out / f"r2{BC0INIT}/collection/evaluation/r1_eval.json")]
    files = [entry for entry in files if entry[2].is_file()]
    root = out / "extra" / "evaluations"
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if directory.name.startswith(("test_control", "test_repro")):
            continue
        policies = json.loads((directory / "policies.json").read_text(encoding="utf-8"))
        for slot in ("expert", "bc0", "r1"):
            path = directory / f"{slot}_eval.json"
            if path.is_file():
                label = "expert" if slot == "expert" else _policy(policies["policies"][slot]["adapter"], directory.name)
                files.append((_set_name(directory.name), label, path))
    return files


def decision_state(state: Mapping[str, Any]) -> str | None:
    """The two first-stage decision states: right after the parameter context, and P + M fresh without T."""
    if state.get("accepted_corrections") or state.get("has_open_candidate"):
        return None
    p = state.get("has_fresh_parameter_context")
    m = state.get("has_fresh_measurement_context")
    t = state.get("has_fresh_topology_context")
    if p and not m and not t and state.get("last_tool") == "get_parameter_context":
        return "after_parameter_context"
    if p and m and not t:
        return "parameter_and_measurement_without_topology"
    return None


def context_order(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = [((s.get("action") or {}).get("tool"), s.get("error_code")) for s in episode.get("trace") or []]
    tools = [tool for tool, _ in steps]
    after_p = None
    if len(tools) > 2 and tools[:2] == ["run_wls", "get_parameter_context"]:
        after_p = {"get_topology_context": "topology", "get_measurement_context": "measurement"}.get(tools[2], tools[2])
    from_pm = None
    seen: set[str] = set()
    for index, tool in enumerate(tools):
        if tool == "get_topology_context":
            break
        seen.add(tool)
        if {"get_parameter_context", "get_measurement_context"} <= seen:
            if index + 1 < len(steps):
                nxt, error = steps[index + 1]
                from_pm = nxt if error is None else f"{nxt} refused"
            break
        if str(tool).startswith("correct_") or tool == "commit_state":
            break
    return {"after_parameter_context": after_p, "from_parameter_and_measurement": from_pm,
            "refused": sum(1 for _, error in steps if error == REFUSED)}


def _spread(values: list[Any]) -> dict[str, Any]:
    """Median and quartiles of the chi-square ratio (the observation's ``remaining_anomaly_score``)."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    return {"n": len(values), "q25": round(values[len(values) // 4], 1), "median": round(median(values), 1),
            "q75": round(values[(3 * len(values)) // 4], 1)}


def _features(episode: Mapping[str, Any]) -> dict[str, Any]:
    trace = episode.get("trace") or []
    wls = ((trace[0].get("policy_tool_output") or {}).get("tool_metrics") or {}) if trace else {}
    parameter = ((trace[1].get("policy_tool_output") or {}).get("tool_metrics") or {}) if len(trace) > 1 else {}
    signatures = wls.get("unresolved_signatures") or []
    return {"chi_square_ratio": wls.get("chi_square_ratio"),
            "top_abs_lambda": parameter.get("parameter_ranking_top_abs_lambda"),
            "vm_outlier": any("channel=Vm" in s for s in signatures)}


def evaluations(out: Path) -> dict[str, Any]:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    from_pm: dict[tuple[str, str], Counter] = defaultdict(Counter)
    choices: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    features: dict[str, dict[str, Any]] = {}
    for set_name, policy, path in evaluation_files(out):
        for episode in tables._episodes(json.loads(path.read_text(encoding="utf-8"))):
            order = context_order(episode)
            row = counts[(set_name, policy)]
            row["refused_all_roots"] += order["refused"]
            if episode.get("family") not in BREAKER:
                continue
            row["breaker_episodes"] += 1
            row["refused_breaker_roots"] += order["refused"]
            if order["after_parameter_context"] is not None:
                row["after_parameter_context"] += 1
                row["then_" + str(order["after_parameter_context"])] += 1
            if order["from_parameter_and_measurement"] is not None:
                row["parameter_and_measurement_without_topology"] += 1
                from_pm[(set_name, policy)][order["from_parameter_and_measurement"]] += 1
            sid = str(episode.get("scenario_id"))
            choices[set_name][sid][policy] = order["after_parameter_context"]
            if sid not in features:
                features[sid] = {"family": episode.get("family"), **_features(episode)}
    pairs = {}
    for scope, sets in (("draws@40", ("draw1@40", "draw2@40")), ("validation", ("validation",)), ("hard", ("hard",))):
        for a, b in PAIRS:
            only_a = only_b = both = 0
            for set_name in sets:
                for per_policy in choices[set_name].values():
                    if per_policy.get(a) is None or per_policy.get(b) is None:
                        continue
                    da, db = per_policy[a] == "measurement", per_policy[b] == "measurement"
                    only_a += da and not db
                    only_b += db and not da
                    both += da and db
            pairs[f"{scope}: {a} vs {b}"] = {"only_first": only_a, "only_second": only_b, "both": both,
                                            "p": tables.mcnemar(only_a, only_b)}
    split: dict[str, Any] = {}
    for policy in DETOURING:
        for family in BREAKER:
            for label, wanted in (("detour", "measurement"), ("topology", "topology")):
                sids = [sid for set_name in ("draw1@40", "draw2@40") for sid, per_policy in choices[set_name].items()
                        if features[sid]["family"] == family and per_policy.get(policy) == wanted]
                values = [features[sid] for sid in sids]
                ratios = [v["chi_square_ratio"] for v in values if v["chi_square_ratio"] is not None]
                lambdas = [v["top_abs_lambda"] for v in values if v["top_abs_lambda"] is not None]
                split[f"{family}: {policy} {label}"] = {
                    "roots": len(values),
                    "median_chi_square_ratio": round(median(ratios), 2) if ratios else None,
                    "median_top_abs_lambda": round(median(lambdas), 2) if lambdas else None,
                    "vm_outlier_share": round(sum(v["vm_outlier"] for v in values) / len(values), 2) if values else None}
    draws = [choices[s] for s in ("draw1@40", "draw2@40")]
    signal = {"test breaker roots, both draws": _spread([features[sid]["chi_square_ratio"] for d in draws for sid in d])}
    for policy in DETOURING:
        signal[f"of which {policy} detours"] = _spread([features[sid]["chi_square_ratio"] for d in draws
                                                        for sid, per_policy in d.items()
                                                        if per_policy.get(policy) == "measurement"])
    return {"counts": {f"{s}|{p}": dict(c) for (s, p), c in counts.items()},
            "from_parameter_and_measurement": {f"{s}|{p}": dict(c) for (s, p), c in from_pm.items()},
            "detour_pairs": pairs, "r2_detour_features": split, "signal": signal,
            "choices_draws_40": {s: {sid: dict(v) for sid, v in choices[s].items()} for s in ("draw1@40", "draw2@40")}}


def mixtures(out: Path) -> dict[str, Any]:
    result = {}
    for policy, relative in MIXTURES:
        path = out / relative
        if not path.is_file():
            continue
        cells: Counter = Counter()
        total = 0
        ratios = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            total += 1
            messages = row.get("messages") or []
            state = json.loads(messages[1]["content"]).get("state", {}) if len(messages) > 1 else {}
            calls = (messages[-1].get("tool_calls") or []) if messages else []
            kind = decision_state(state)
            if kind:
                label = calls[0]["function"]["name"] if calls else None
                source = row.get("research_mixture_source") or "d0"
                group = _group(row.get("scenario_family"))
                cells[f"{kind}|{group}|{label}|{source}"] += 1
                if kind == "after_parameter_context" and group == "breaker":
                    ratios.append(state.get("remaining_anomaly_score"))
        result[policy] = {"path": relative, "rows": total, "decision_rows": dict(cells),
                          "breaker_decision_signal": _spread(ratios)}
    return result


def ledgers(out: Path) -> dict[str, Any]:
    result = {}
    for name, relative in LEDGERS:
        cells: Counter = Counter()
        rows = eligible = 0
        ratios = []
        for line in (out / relative).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows += 1
            eligible += bool(row.get("research_label_eligible"))
            state = row.get("policy_observation") or {}
            kind = decision_state(state)
            if kind:
                keep = "kept" if row.get("research_label_eligible") else \
                    "dropped (" + ", ".join(row.get("research_label_ineligibility_reasons") or []) + ")"
                group = _group(row.get("scenario_family"))
                cells[f"{kind}|{group}|visited by {row.get('state_visited_by')}|"
                      f"learner {(row.get('model_action') or {}).get('tool')}|"
                      f"teacher {(row.get('preferred_action') or {}).get('tool')}|{keep}"] += 1
                if kind == "after_parameter_context" and group == "breaker":
                    ratios.append(state.get("remaining_anomaly_score"))
        result[name] = {"path": relative, "rows": rows, "eligible": eligible, "decision_states": dict(cells),
                        "breaker_decision_signal": _spread(ratios)}
    return result


def markdown(result: Mapping[str, Any]) -> str:
    ev = result["evaluations"]
    lines = ["Breaker roots: detours (measurement context read right after the parameter context) / "
             "episodes that read the parameter context first", "",
             "| Set | " + " | ".join(POLICIES) + " |", "| --- |" + " --- |" * len(POLICIES)]
    for set_name in SET_ORDER:
        cells = []
        for policy in POLICIES:
            c = ev["counts"].get(f"{set_name}|{policy}")
            cells.append(f"{c.get('then_measurement', 0)}/{c.get('after_parameter_context', 0)}" if c else "")
        lines.append(f"| {set_name} | " + " | ".join(cells) + " |")
    lines += ["", "From the state with the parameter and measurement contexts and no topology context "
              "(breaker roots; next action):", ""]
    for key in sorted(ev["from_parameter_and_measurement"]):
        lines.append(f"- {key}: {ev['from_parameter_and_measurement'][key]}")
    lines += ["", "Refused corrections (`correction_not_supported_by_current_context`), breaker roots / all roots:", ""]
    for set_name in SET_ORDER:
        cells = []
        for policy in POLICIES:
            c = ev["counts"].get(f"{set_name}|{policy}")
            if c:
                cells.append(f"{policy} {c.get('refused_breaker_roots', 0)}/{c.get('refused_all_roots', 0)}")
        lines.append(f"- {set_name}: " + ", ".join(cells))
    lines += ["", "Paired detours (roots where only the first / only the second detours, both; exact McNemar p):", ""]
    for key, v in ev["detour_pairs"].items():
        lines.append(f"- {key}: {v['only_first']} / {v['only_second']}, both {v['both']}; p = {v['p']:.2g}")
    lines += ["", "Each R2's detour roots against its topology-first roots (both draws at 40 actions):", ""]
    for key, v in ev["r2_detour_features"].items():
        lines.append(f"- {key}: {v}")
    lines += ["", "Training rows at the two decision states (state | family group | label | source: rows):", ""]
    for policy, v in result["mixtures"].items():
        lines.append(f"- {policy} ({v['path']}, {v['rows']} rows): {v['decision_rows']}")
    lines += ["", "Round ledgers at the two decision states:", ""]
    for name, v in result["ledgers"].items():
        lines.append(f"- {name} ({v['rows']} rows, {v['eligible']} kept): {v['decision_states']}")
    lines += ["", "Chi-square ratio at the decision on breaker roots (quartiles):", ""]
    for name, v in ev["signal"].items():
        lines.append(f"- {name}: {v}")
    for name, v in list(result["mixtures"].items()) + list(result["ledgers"].items()):
        lines.append(f"- {name}: {v['breaker_decision_signal']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.cell_dir / "out"
    result = {"evaluations": evaluations(out), "mixtures": mixtures(out), "ledgers": ledgers(out)}
    if args.output:
        args.output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(markdown(result), end="")


if __name__ == "__main__":
    main()
