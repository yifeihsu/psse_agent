"""Does the evidence a model reads at a state say what is still wrong there?  A probe on the minimal environment.

    python -m research.minimal_env.evidence_probe --input-dir OUT --train train --evaluate dev

The rows come from ``research.minimal_env.stopping_states``, one set per state an action was chosen on: what the
observation shows at that state, as numbers, and what the truth says (labels).  Small boosted-tree classifiers
are fitted on the rows of one split (folds grouped by case, so a case's states are never on both sides of a fold)
and read on the states where the unrepaired episodes stopped.  Two questions:

``state``
    Is the case still unrepaired here (an error left or a healthy object damaged)?  Read against the states where
    a repaired episode finished: at the threshold that flags 5% of those, how many stopping states of unrepaired
    episodes does the evidence flag?
``candidate``
    Which object is wrong?  Every line, meter and breaker of a state with an error left is a candidate, labelled
    by whether it carries an unrepaired error.  At a stopping state, the place of the best-placed true object
    among all candidates of all kinds.

What a probe sees is the observation's evidence and nothing that names an object: no line number, meter index or
breaker name, so a place it gives reads the evidence and not the scenario generator's habits.  The counts of the
episode so far (actions used, corrections in effect) are a separate set of columns, left out of the ``evidence``
reading and added in the ``with_counts`` one, because they tell a reader how many errors a case of this suite
usually holds rather than what this state shows.  The substation table is computed at every state; the
``shown`` reading keeps it only where the observation showed it, the ``inspect`` reading everywhere (it is one
``inspect`` away).

A probe that reads an error off the evidence says the evidence carries it, for this suite's cases; a probe that
does not is a weak learner on hand-made columns, which bounds nothing.  Nothing here reaches a prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROBE_CONTRACT = "minimal_evidence_probe_v1"
KINDS = ("meter", "line", "breaker")
FLOW_CHANNELS = ("Pf", "Qf", "Pt", "Qt")
CHANNELS = ("Vm", "Pinj", "Qinj") + FLOW_CHANNELS
EFFECTS = ("dangling_line_terminal", "reconnect_line_terminal", "bus_split", "merge", "equivalent", "unsupplied_island")

SUBSTATION_STATE_COLUMNS = ("sub_converged", "sub_chi_ratio", "sub_log_chi_ratio", "sub_max_residual", "breaker_s1",
                            "breaker_s2", "breaker_s3", "breakers_over_3", "breaker_best_progress")
STATE_EVIDENCE_COLUMNS = (
    "converged", "chi_ratio", "log_chi_ratio", "max_residual", "residual_ratio", "passes",
    "meter_r1", "meter_r2", "meter_r3", "meter_r4", "meter_r5", "meters_over_2", "meters_over_3", "meter_rms",
    "line_m1", "line_m2", "line_m3", "lines_over_3", "flow_max", "bus_vm_max", "bus_p_max", "bus_q_max",
) + SUBSTATION_STATE_COLUMNS + (
    "breakers_shown", "edit_max_factor", "last_edit_factor", "last_mod_chi_drop", "last_mod_parameters",
    "last_mod_meters", "last_mod_topology", "tables_shown",
)
STATE_COUNT_COLUMNS = ("steps_used", "mods_in_effect", "mods_undone", "parameters_in_effect", "meters_in_effect",
                       "topology_in_effect", "inspects")
STATE_COLUMNS = STATE_EVIDENCE_COLUMNS + STATE_COUNT_COLUMNS

BREAKER_CANDIDATE_COLUMNS = (
    "br_score", "br_theta", "br_vm", "br_rank", "br_over_top", "br_tested", "br_progress", "br_flip_chi_ratio",
    "br_closed", "br_line_mult", "br_line_flow", "br_bus_res",
) + tuple(f"br_effect_{name}" for name in EFFECTS)
CANDIDATE_COLUMNS = (
    "kind_meter", "kind_line", "kind_breaker",
    "st_chi_ratio", "st_passes", "st_max_residual", "st_sub_chi_ratio", "st_breakers_shown",
    "named_in_effect", "named_undone", "edit_size",
    "ln_mult", "ln_lr", "ln_lx", "ln_mult_rank", "ln_mult_over_top", "ln_flow", "ln_flow_rank", "ln_bus_res",
    "ln_xfmr", "ln_out", "ln_meter_res_max", "ln_meter_res_rank",
    "mt_res", "mt_rank", "mt_over_top", "mt_place_mult", "mt_place_flow", "mt_neighbors_max",
) + tuple(f"mt_channel_{name}" for name in CHANNELS) + BREAKER_CANDIDATE_COLUMNS
#: Candidate columns that read the substation table, blanked in the ``shown`` reading where it was not shown.
SUBSTATION_CANDIDATE_COLUMNS = ("st_sub_chi_ratio",) + BREAKER_CANDIDATE_COLUMNS
#: The statistic a reader looks at first, per kind: the within-kind place by it is printed beside the probe's.
FIRST_STATISTIC = {"meter": "mt_res", "line": "ln_mult", "breaker": "br_score"}


# ------------------------------------------------------------------ reading an observation


def _number(value: Any) -> float:
    try:
        return float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def _magnitude(value: Any) -> float:
    return abs(_number(value))


def _top(values: Sequence[float], count: int) -> list[float]:
    ordered = sorted((value for value in values if not math.isnan(value)), reverse=True)
    return (ordered + [math.nan] * count)[:count]


def _ranks(values: Sequence[float]) -> list[float]:
    """1 for the largest; NaN where the value is missing."""
    order = sorted((index for index, value in enumerate(values) if not math.isnan(value)), key=lambda index: -values[index])
    out = [math.nan] * len(values)
    for place, index in enumerate(order, start=1):
        out[index] = float(place)
    return out


def _current(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    current = int(observation["current_version"])
    return next(entry for entry in observation["versions"] if int(entry["version"]) == current)


def _edit_factor(change: Mapping[str, Any]) -> float:
    before, after = _number(change.get("before")), _number(change.get("after"))
    if before > 0 and after > 0:
        return abs(math.log(after / before))
    return math.nan


def _modification_facts(observation: Mapping[str, Any], sigma: Mapping[int, float]) -> dict[str, Any]:
    """What the corrections so far did, per object: named in effect, named and undone, the size of the change."""
    versions = {int(entry["version"]): entry for entry in observation["versions"]}
    named_in, named_out = set(), set()
    size: dict[tuple[str, str], float] = defaultdict(float)
    edits: list[float] = []
    last: Mapping[str, Any] | None = None
    from research.minimal_env.stopping_states import named_objects

    for record in observation["modifications"]:
        objects = named_objects(record)
        (named_in if record.get("in_effect") else named_out).update(objects)
        if not record.get("in_effect"):
            continue
        last = record
        for change in record.get("changes") or []:
            if change.get("object") == "line" and change.get("field") in ("r", "x"):
                factor = _edit_factor(change)
                if not math.isnan(factor):
                    key = ("line", str(int(change["line"])))
                    size[key] = max(size[key], factor)
                    edits.append(factor)
            elif change.get("object") == "meter":
                index = int(change["meter"])
                scale = sigma.get(index) or math.nan
                moved = abs(_number(change.get("after")) - _number(change.get("before")))
                size[("meter", str(index))] = max(size[("meter", str(index))], moved / scale if scale and scale > 0 else math.nan)
            elif change.get("object") == "breaker":
                size[("breaker", str(change["breaker"]))] = 1.0
    facts: dict[str, Any] = {"named_in": named_in, "named_out": named_out, "size": size,
                             "edit_max_factor": max(edits) if edits else math.nan}
    facts["last_edit_factor"] = math.nan
    facts["last_mod_chi_drop"] = math.nan
    facts["last_tool"] = None
    if last is not None:
        facts["last_tool"] = last.get("tool")
        factors = [_edit_factor(change) for change in last.get("changes") or []
                   if change.get("object") == "line" and change.get("field") in ("r", "x")]
        factors = [factor for factor in factors if not math.isnan(factor)]
        facts["last_edit_factor"] = max(factors) if factors else math.nan
        before = (versions.get(int(last["from_version"])) or {}).get("estimate") or {}
        after = (versions.get(int(last["to_version"])) or {}).get("estimate") or {}
        if before.get("converged") and after.get("converged") and _number(before.get("chi_square")) > 0:
            facts["last_mod_chi_drop"] = 1.0 - _number(after.get("chi_square")) / _number(before.get("chi_square"))
    return facts


def _state_features(observation: Mapping[str, Any], entry: Mapping[str, Any], substation: Mapping[str, Any] | None,
                    facts: Mapping[str, Any]) -> dict[str, float]:
    estimate = entry["estimate"]
    f: dict[str, float] = {name: math.nan for name in STATE_COLUMNS}
    f["converged"] = 1.0 if estimate.get("converged") else 0.0
    f["tables_shown"] = 1.0 if "meters" in estimate else 0.0
    if estimate.get("converged"):
        chi, threshold = _number(estimate.get("chi_square")), _number(estimate.get("chi_square_threshold"))
        f["chi_ratio"] = chi / threshold
        f["log_chi_ratio"] = math.log(max(chi, 1e-9) / threshold)
        f["max_residual"] = _number(estimate.get("max_residual"))
        if estimate.get("residual_threshold") is not None:
            f["residual_ratio"] = f["max_residual"] / _number(estimate["residual_threshold"])
        from psse_env.minimal.scoring import estimator_passes

        f["passes"] = 1.0 if estimator_passes(estimate) else 0.0
    meters = [_magnitude(row.get("residual")) for row in estimate.get("meters") or [] if row.get("residual") is not None]
    if meters:
        f["meter_r1"], f["meter_r2"], f["meter_r3"], f["meter_r4"], f["meter_r5"] = _top(meters, 5)
        f["meters_over_2"] = float(sum(value > 2 for value in meters))
        f["meters_over_3"] = float(sum(value > 3 for value in meters))
        f["meter_rms"] = math.sqrt(sum(value * value for value in meters) / len(meters))
    lines = estimate.get("lines") or []
    if lines:
        multipliers = [max(_magnitude(row.get("lr")), _magnitude(row.get("lx"))) for row in lines]
        f["line_m1"], f["line_m2"], f["line_m3"] = _top(multipliers, 3)
        f["lines_over_3"] = float(sum(value > 3 for value in multipliers if not math.isnan(value)))
        flows = [max((_magnitude(row.get(key)) for key in ("pf", "qf", "pt", "qt")), default=math.nan) for row in lines]
        f["flow_max"] = max((value for value in flows if not math.isnan(value)), default=math.nan)
    buses = estimate.get("buses") or []
    for key in ("vm", "p", "q"):
        values = [_magnitude(row.get(key)) for row in buses]
        values = [value for value in values if not math.isnan(value)]
        if values:
            f[f"bus_{key}_max"] = max(values)
    if substation is not None:
        f["sub_converged"] = 1.0 if substation.get("converged") else 0.0
        if substation.get("converged"):
            chi, threshold = _number(substation.get("chi_square")), _number(substation.get("chi_square_threshold"))
            f["sub_chi_ratio"] = chi / threshold
            f["sub_log_chi_ratio"] = math.log(max(chi, 1e-9) / threshold)
            f["sub_max_residual"] = _number(substation.get("max_residual"))
            scores = [_magnitude(row.get("score")) for row in substation.get("breakers") or []]
            f["breaker_s1"], f["breaker_s2"], f["breaker_s3"] = _top(scores, 3)
            f["breakers_over_3"] = float(sum(value > 3 for value in scores if not math.isnan(value)))
            progress = [_number(row.get("flip_progress")) for row in substation.get("breakers") or [] if row.get("flip_progress") is not None]
            f["breaker_best_progress"] = max(progress) if progress else math.nan
    f["breakers_shown"] = 1.0 if "breakers" in (entry.get("substation") or {}) else 0.0
    f["edit_max_factor"] = facts["edit_max_factor"]
    f["last_edit_factor"] = facts["last_edit_factor"]
    f["last_mod_chi_drop"] = facts["last_mod_chi_drop"]
    f["last_mod_parameters"] = 1.0 if facts["last_tool"] == "correct_parameters" else 0.0
    f["last_mod_meters"] = 1.0 if facts["last_tool"] == "correct_measurements" else 0.0
    f["last_mod_topology"] = 1.0 if facts["last_tool"] == "correct_topology" else 0.0
    modifications = observation["modifications"]
    f["steps_used"] = float(observation["budget"]["used"])
    f["mods_in_effect"] = float(sum(1 for record in modifications if record.get("in_effect")))
    f["mods_undone"] = float(sum(1 for record in modifications if not record.get("in_effect")))
    for name, tool in (("parameters_in_effect", "correct_parameters"), ("meters_in_effect", "correct_measurements"),
                       ("topology_in_effect", "correct_topology")):
        f[name] = float(sum(1 for record in modifications if record.get("in_effect") and record.get("tool") == tool))
    f["inspects"] = float(sum(1 for event in observation["history"] if (event.get("action") or {}).get("tool") == "inspect"))
    return f


def _candidates(entry: Mapping[str, Any], substation: Mapping[str, Any] | None, state: Mapping[str, float],
                facts: Mapping[str, Any]) -> tuple[list[dict[str, float]], list[str], list[str]]:
    estimate = entry["estimate"]
    lines = estimate.get("lines") or []
    meters = [row for row in estimate.get("meters") or [] if row.get("residual") is not None]
    buses = {int(row["bus"]): row for row in estimate.get("buses") or []}
    rows: list[dict[str, float]] = []
    kinds: list[str] = []
    objects: list[str] = []

    def base(kind: str, key: tuple[str, str]) -> dict[str, float]:
        f = {name: math.nan for name in CANDIDATE_COLUMNS}
        for name in KINDS:
            f[f"kind_{name}"] = 1.0 if kind == name else 0.0
        f["st_chi_ratio"], f["st_passes"], f["st_max_residual"] = state["chi_ratio"], state["passes"], state["max_residual"]
        f["st_sub_chi_ratio"], f["st_breakers_shown"] = state["sub_chi_ratio"], state["breakers_shown"]
        f["named_in_effect"] = 1.0 if key in facts["named_in"] else 0.0
        f["named_undone"] = 1.0 if key in facts["named_out"] and key not in facts["named_in"] else 0.0
        f["edit_size"] = facts["size"].get(key, math.nan)
        return f

    def bus_residual(bus: Any) -> float:
        row = buses.get(int(bus)) if bus is not None and str(bus).isdigit() else None
        if row is None:
            return math.nan
        values = [_magnitude(row.get(key)) for key in ("vm", "p", "q")]
        values = [value for value in values if not math.isnan(value)]
        return max(values) if values else math.nan

    multipliers = [max(_magnitude(row.get("lr")), _magnitude(row.get("lx"))) for row in lines]
    flows = [max((_magnitude(row.get(key)) for key in ("pf", "qf", "pt", "qt")), default=math.nan) for row in lines]
    top_multiplier = max((value for value in multipliers if not math.isnan(value)), default=math.nan)
    by_line = {int(row["line"]): (multiplier, flow) for row, multiplier, flow in zip(lines, multipliers, flows)}
    residuals = [_magnitude(row.get("residual")) for row in meters]
    meter_ranks = _ranks(residuals)
    line_meter: dict[int, list[tuple[float, float]]] = defaultdict(list)
    place: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row, value, rank in zip(meters, residuals, meter_ranks):
        if row.get("channel") in FLOW_CHANNELS:
            line_meter[int(row["at"])].append((value, rank))
            place[("line", int(row["at"]))].append(value)
        else:
            place[("bus", int(row["at"]))].append(value)
    for row, multiplier, multiplier_rank, flow, flow_rank in zip(lines, multipliers, _ranks(multipliers), flows, _ranks(flows)):
        key = ("line", str(int(row["line"])))
        f = base("line", key)
        f["ln_mult"], f["ln_lr"], f["ln_lx"] = multiplier, _magnitude(row.get("lr")), _magnitude(row.get("lx"))
        f["ln_mult_rank"] = multiplier_rank
        f["ln_mult_over_top"] = multiplier / top_multiplier if top_multiplier and not math.isnan(top_multiplier) else math.nan
        f["ln_flow"], f["ln_flow_rank"] = flow, flow_rank
        f["ln_bus_res"] = max((value for value in (bus_residual(row.get("from")), bus_residual(row.get("to"))) if not math.isnan(value)),
                              default=math.nan)
        f["ln_xfmr"] = 1.0 if row.get("xfmr") else 0.0
        f["ln_out"] = 1.0 if row.get("out") else 0.0
        own = line_meter.get(int(row["line"])) or []
        if own:
            best = max(own, key=lambda item: item[0])
            f["ln_meter_res_max"], f["ln_meter_res_rank"] = best
        rows.append(f)
        kinds.append("line")
        objects.append(key[1])
    top_residual = max((value for value in residuals if not math.isnan(value)), default=math.nan)
    for row, value, rank in zip(meters, residuals, meter_ranks):
        key = ("meter", str(int(row["index"])))
        f = base("meter", key)
        f["mt_res"], f["mt_rank"] = value, rank
        f["mt_over_top"] = value / top_residual if top_residual and not math.isnan(top_residual) else math.nan
        for name in CHANNELS:
            f[f"mt_channel_{name}"] = 1.0 if row.get("channel") == name else 0.0
        at = int(row["at"])
        if row.get("channel") in FLOW_CHANNELS:
            f["mt_place_mult"], f["mt_place_flow"] = by_line.get(at, (math.nan, math.nan))
            others = list(place[("line", at)])
        else:
            f["mt_place_flow"] = bus_residual(at)
            others = list(place[("bus", at)])
        others.remove(value) if value in others else None
        f["mt_neighbors_max"] = max(others) if others else math.nan
        rows.append(f)
        kinds.append("meter")
        objects.append(key[1])
    breakers = (substation or {}).get("breakers") or [] if (substation or {}).get("converged") else []
    scores = [_magnitude(row.get("score")) for row in breakers]
    top_score = max((value for value in scores if not math.isnan(value)), default=math.nan)
    chi = _number((substation or {}).get("chi_square"))
    for row, score, rank in zip(breakers, scores, _ranks(scores)):
        key = ("breaker", str(row["cb"]))
        f = base("breaker", key)
        f["br_score"], f["br_theta"], f["br_vm"] = score, _magnitude(row.get("theta")), _magnitude(row.get("vm"))
        f["br_rank"] = rank
        f["br_over_top"] = score / top_score if top_score and not math.isnan(top_score) else math.nan
        f["br_tested"] = 1.0 if row.get("flip_chi_square") is not None else 0.0
        f["br_progress"] = _number(row.get("flip_progress"))
        f["br_flip_chi_ratio"] = _number(row.get("flip_chi_square")) / chi if chi and chi > 0 else math.nan
        f["br_closed"] = 1.0 if row.get("status") == "closed" else 0.0
        if row.get("line") is not None:
            f["br_line_mult"], f["br_line_flow"] = by_line.get(int(row["line"]), (math.nan, math.nan))
        if row.get("bus") is not None:
            f["br_bus_res"] = bus_residual(row.get("bus"))
        for name in EFFECTS:
            f[f"br_effect_{name}"] = 1.0 if row.get("flip_effect") == name else 0.0
        rows.append(f)
        kinds.append("breaker")
        objects.append(key[1])
    return rows, kinds, objects


def state_rows(observation: Mapping[str, Any], substation: Mapping[str, Any] | None, scenario: Mapping[str, Any],
               physical: Mapping[str, Any]) -> dict[str, Any]:
    """One state's probe rows: its features and its candidates' features, with the truth's labels."""
    del scenario
    from research.minimal_env.stopping_states import error_object

    entry = _current(observation)
    sigma = {int(row["index"]): _number(row.get("sigma")) for row in entry["estimate"].get("meters") or []}
    facts = _modification_facts(observation, sigma)
    state = _state_features(observation, entry, substation, facts)
    candidates, kinds, objects = _candidates(entry, substation, state, facts)
    left = {error_object(error) for error in physical["errors"] if not error["repaired"]}
    left |= {("line", item) for kind, item in list(left) if kind == "line_status"}
    labels = [1 if (kind, item) in left else 0 for kind, item in zip(kinds, objects)]
    return {
        "state": np.asarray([state[name] for name in STATE_COLUMNS], dtype=np.float32),
        "solved": bool(physical["solved"]),
        "errors_left": int(physical["true_errors"]) - int(physical["repaired"]),
        "damaged": int(physical["damaged"]),
        "candidates": np.asarray([[row[name] for name in CANDIDATE_COLUMNS] for row in candidates], dtype=np.float32).reshape(
            len(candidates), len(CANDIDATE_COLUMNS)),
        "kinds": np.asarray([KINDS.index(kind) for kind in kinds], dtype=np.int8),
        "objects": list(objects),
        "labels": np.asarray(labels, dtype=np.int8),
    }


def pack_states(case: str, split: str, rows: Sequence[Mapping[str, Any]], *, stop_index: int, episode_solved: bool,
                reproduced: bool) -> dict[str, Any]:
    """One episode's rows, stacked."""
    count = len(rows)
    sizes = [len(row["labels"]) for row in rows]
    return {
        "case": case, "split": split, "episode_solved": bool(episode_solved), "reproduced": bool(reproduced),
        "state": (np.stack([row["state"] for row in rows]) if rows else np.zeros((0, len(STATE_COLUMNS)), np.float32)),
        "state_index": np.arange(count, dtype=np.int16),
        "state_is_stop": np.asarray([index == count - 1 for index in range(count)], dtype=bool),
        "state_solved": np.asarray([row["solved"] for row in rows], dtype=bool),
        "state_left": np.asarray([row["errors_left"] for row in rows], dtype=np.int8),
        "state_damaged": np.asarray([row["damaged"] for row in rows], dtype=np.int8),
        "candidates": (np.concatenate([row["candidates"] for row in rows]) if rows else np.zeros((0, len(CANDIDATE_COLUMNS)), np.float32)),
        "cand_state": np.repeat(np.arange(count, dtype=np.int32), sizes),
        "cand_kind": (np.concatenate([row["kinds"] for row in rows]) if rows else np.zeros(0, np.int8)),
        "cand_object": [item for row in rows for item in row["objects"]],
        "cand_label": (np.concatenate([row["labels"] for row in rows]) if rows else np.zeros(0, np.int8)),
    }


def save_packed(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Every episode's rows in one file: states numbered across episodes, candidates pointing at their state."""
    episodes = sorted(episodes, key=lambda item: item["case"])
    offsets = np.cumsum([0] + [len(item["state_index"]) for item in episodes])
    np.savez_compressed(
        path,
        columns_state=np.asarray(STATE_COLUMNS), columns_candidate=np.asarray(CANDIDATE_COLUMNS),
        episode_case=np.asarray([item["case"] for item in episodes]), episode_split=np.asarray([item["split"] for item in episodes]),
        episode_solved=np.asarray([item["episode_solved"] for item in episodes], dtype=bool),
        episode_reproduced=np.asarray([item["reproduced"] for item in episodes], dtype=bool),
        state_x=np.concatenate([item["state"] for item in episodes]) if episodes else np.zeros((0, len(STATE_COLUMNS)), np.float32),
        state_episode=np.concatenate([np.full(len(item["state_index"]), number, dtype=np.int32) for number, item in enumerate(episodes)])
        if episodes else np.zeros(0, np.int32),
        state_index=np.concatenate([item["state_index"] for item in episodes]) if episodes else np.zeros(0, np.int16),
        state_is_stop=np.concatenate([item["state_is_stop"] for item in episodes]) if episodes else np.zeros(0, bool),
        state_solved=np.concatenate([item["state_solved"] for item in episodes]) if episodes else np.zeros(0, bool),
        state_left=np.concatenate([item["state_left"] for item in episodes]) if episodes else np.zeros(0, np.int8),
        state_damaged=np.concatenate([item["state_damaged"] for item in episodes]) if episodes else np.zeros(0, np.int8),
        cand_x=np.concatenate([item["candidates"] for item in episodes]) if episodes else np.zeros((0, len(CANDIDATE_COLUMNS)), np.float32),
        cand_state=np.concatenate([item["cand_state"] + offsets[number] for number, item in enumerate(episodes)])
        if episodes else np.zeros(0, np.int64),
        cand_kind=np.concatenate([item["cand_kind"] for item in episodes]) if episodes else np.zeros(0, np.int8),
        cand_object=np.asarray([value for item in episodes for value in item["cand_object"]]),
        cand_label=np.concatenate([item["cand_label"] for item in episodes]) if episodes else np.zeros(0, np.int8),
    )


# ------------------------------------------------------------------ the probes


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _classifier(seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=seed)


def out_of_fold(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Scores of every row from a classifier fitted on the folds without its group."""
    from sklearn.model_selection import GroupKFold

    scores = np.full(len(y), np.nan, dtype=np.float64)
    for train, test in GroupKFold(n_splits=folds).split(x, y, groups):
        model = _classifier(seed).fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return scores


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if len(positive) == 0 or len(negative) == 0:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score([1] * len(positive) + [0] * len(negative), list(positive) + list(negative)))


def _blank_substation(x: np.ndarray, columns: Sequence[str], shown: np.ndarray, blank: Sequence[str]) -> np.ndarray:
    """The ``shown`` reading: substation columns missing where the observation did not show the table."""
    out = x.copy()
    hidden = ~shown.astype(bool)
    for name in blank:
        out[hidden, list(columns).index(name)] = np.nan
    return out


def state_matrix(data: Mapping[str, np.ndarray], reading: str, counts: bool) -> np.ndarray:
    columns = list(STATE_EVIDENCE_COLUMNS) + (list(STATE_COUNT_COLUMNS) if counts else [])
    index = [list(STATE_COLUMNS).index(name) for name in columns]
    x = data["state_x"][:, index]
    if reading == "shown":
        shown = data["state_x"][:, list(STATE_COLUMNS).index("breakers_shown")] > 0.5
        x = _blank_substation(x, columns, shown, SUBSTATION_STATE_COLUMNS)
    return x


def candidate_matrix(data: Mapping[str, np.ndarray], reading: str) -> np.ndarray:
    x = data["cand_x"]
    if reading == "shown":
        shown = x[:, list(CANDIDATE_COLUMNS).index("st_breakers_shown")] > 0.5
        x = _blank_substation(x, CANDIDATE_COLUMNS, shown, SUBSTATION_CANDIDATE_COLUMNS)
    return x


def _usable(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """States of reproduced episodes."""
    return data["episode_reproduced"][data["state_episode"]]


def _stops(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stopping states of the unrepaired episodes, and the finishing states of the repaired ones (both reproduced)."""
    usable = _usable(data)
    solved_episode = data["episode_solved"][data["state_episode"]]
    stop = data["state_is_stop"] & usable
    return stop & ~solved_episode, stop & solved_episode & data["state_solved"]


def state_probe(train: Mapping[str, np.ndarray], evaluate: Mapping[str, np.ndarray], *, reading: str, counts: bool,
                false_alarms: float = 0.05) -> dict[str, Any]:
    """Is the case unrepaired at a state?  Out of fold on ``train``; fitted on all of it for ``evaluate``."""
    usable = _usable(train)
    x, y = state_matrix(train, reading, counts), (~train["state_solved"]).astype(np.int8)
    groups = train["state_episode"]
    scores = np.full(len(y), np.nan)
    scores[usable] = out_of_fold(x[usable], y[usable], groups[usable])
    model = _classifier(0).fit(x[usable], y[usable])
    evaluated = model.predict_proba(state_matrix(evaluate, reading, counts))[:, 1]
    failed_train, finished_train = _stops(train)
    threshold = float(np.quantile(scores[finished_train], 1.0 - false_alarms)) if finished_train.any() else math.nan

    def read(data: Mapping[str, np.ndarray], values: np.ndarray) -> dict[str, Any]:
        failed, finished = _stops(data)
        passes = data["state_x"][:, list(STATE_COLUMNS).index("passes")] > 0.5
        out = {"auc_all_states": _auc(values[_usable(data) & (data["state_solved"] == 0)], values[_usable(data) & data["state_solved"]]),
               "auc_stopping_vs_finished": _auc(values[failed], values[finished]),
               "unrepaired_stops": int(failed.sum()), "repaired_finishes": int(finished.sum()),
               "unrepaired_stops_flagged": int((values[failed] > threshold).sum()),
               "repaired_finishes_flagged": int((values[finished] > threshold).sum()),
               "unrepaired_stops_with_the_estimate_passing": int((failed & passes).sum()),
               "of_those_flagged": int(((values > threshold) & failed & passes).sum()),
               "repaired_finishes_with_the_estimate_passing": int((finished & passes).sum()),
               "auc_stopping_vs_finished_both_passing": _auc(values[failed & passes], values[finished & passes])}
        return out

    return {"reading": reading, "counts": counts, "threshold": threshold, "false_alarms": false_alarms,
            "train": read(train, scores), "evaluate": read(evaluate, evaluated),
            "scores_train": scores, "scores_evaluate": evaluated}


def candidate_probe(train: Mapping[str, np.ndarray], evaluate: Mapping[str, np.ndarray], *, reading: str) -> dict[str, Any]:
    """Which object is wrong?  Rows of the states with an error left; out of fold on ``train``, fitted for ``evaluate``."""
    def rows(data: Mapping[str, np.ndarray]) -> np.ndarray:
        state = data["cand_state"]
        return _usable(data)[state] & (data["state_left"][state] > 0)

    keep = rows(train)
    x, y = candidate_matrix(train, reading), train["cand_label"].astype(np.int8)
    groups = train["state_episode"][train["cand_state"]]
    scores = np.full(len(y), np.nan)
    scores[keep] = out_of_fold(x[keep], y[keep], groups[keep])
    model = _classifier(0).fit(x[keep], y[keep])
    evaluated = np.full(len(evaluate["cand_label"]), np.nan)
    keep_evaluate = rows(evaluate)
    evaluated[keep_evaluate] = model.predict_proba(candidate_matrix(evaluate, reading)[keep_evaluate])[:, 1]
    return {"reading": reading, "scores_train": scores, "scores_evaluate": evaluated,
            "train_auc": _auc(scores[keep & (y == 1)], scores[keep & (y == 0)]),
            "evaluate_auc": _auc(evaluated[keep_evaluate & (evaluate["cand_label"] == 1)],
                                 evaluated[keep_evaluate & (evaluate["cand_label"] == 0)])}


def places(data: Mapping[str, np.ndarray], scores: np.ndarray, states: np.ndarray) -> list[dict[str, Any]]:
    """At each state of ``states``: the place of the best-placed true object among all candidates by ``scores``,
    its place within its kind, and its within-kind place by the kind's first statistic."""
    out = []
    by_state: dict[int, list[int]] = defaultdict(list)
    for index, state in enumerate(data["cand_state"]):
        by_state[int(state)].append(index)
    for state in np.flatnonzero(states):
        members = np.asarray(by_state.get(int(state), []), dtype=np.int64)
        if not len(members):
            continue
        labels = data["cand_label"][members] == 1
        if not labels.any():
            continue
        values = scores[members]
        kinds = data["cand_kind"][members]
        best = None
        for position in np.flatnonzero(labels):
            overall = 1 + int((values > values[position]).sum())
            same = kinds == kinds[position]
            within = 1 + int((values[same] > values[position]).sum())
            kind = KINDS[int(kinds[position])]
            column = list(CANDIDATE_COLUMNS).index(FIRST_STATISTIC[kind])
            statistic = data["cand_x"][members, column]
            first = 1 + int((np.nan_to_num(statistic[same], nan=-1.0) > np.nan_to_num(statistic[position], nan=-1.0)).sum())
            item = {"state": int(state), "kind": kind, "object": str(data["cand_object"][members[position]]),
                    "place": overall, "place_within_kind": within, "place_by_first_statistic": first,
                    "candidates": int(len(members))}
            if best is None or overall < best["place"]:
                best = item
        out.append(best)
    return out


def _share(items: Sequence[Mapping[str, Any]], key: str, top: int) -> str:
    if not items:
        return "n/a"
    hit = sum(1 for item in items if item[key] <= top)
    return f"{hit} of {len(items)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--train", default="train")
    parser.add_argument("--evaluate", default="dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--evaluate-cases", type=Path, default=None,
                        help="read the evaluated split on these cases only (a reference file: the cases the truth repairs)")
    args = parser.parse_args(argv)
    train = load(args.input_dir / f"probe_{args.train}.npz")
    evaluate = load(args.input_dir / f"probe_{args.evaluate}.npz")
    if args.evaluate_cases is not None:
        from research.minimal_env.stopping_states import case_filter

        kept = case_filter(args.evaluate_cases) or set()
        # A case outside the set is read nowhere: its states count as not reproduced.
        evaluate["episode_reproduced"] = evaluate["episode_reproduced"] & np.isin(evaluate["episode_case"], sorted(kept))
    records = {}
    for split in (args.train, args.evaluate):
        path = args.input_dir / f"episodes_{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[(split, record["case"])] = record
    report: dict[str, Any] = {"contract": PROBE_CONTRACT, "train": args.train, "evaluate": args.evaluate,
                              "evaluate_cases": str(args.evaluate_cases) if args.evaluate_cases else None,
                              "state_columns": {"evidence": list(STATE_EVIDENCE_COLUMNS), "counts": list(STATE_COUNT_COLUMNS)},
                              "candidate_columns": list(CANDIDATE_COLUMNS), "state": [], "candidate": []}
    stop_scores: dict[str, dict[str, Any]] = defaultdict(dict)
    #: Every row's score, out of fold on the training split and fitted on the evaluated one, for readings at other
    #: states than the stopping ones (``probe_scores.npz``).
    kept_scores: dict[str, np.ndarray] = {}
    for reading in ("shown", "inspect"):
        for counts in (False, True):
            result = state_probe(train, evaluate, reading=reading, counts=counts)
            name = f"state_{reading}{'_counts' if counts else ''}"
            kept_scores[f"{name}_{args.train}"] = result["scores_train"]
            kept_scores[f"{name}_{args.evaluate}"] = result["scores_evaluate"]
            report["state"].append({key: value for key, value in result.items() if not key.startswith("scores")})
            print(json.dumps(report["state"][-1], sort_keys=True), flush=True)
            for split, data, scores in ((args.train, train, result["scores_train"]), (args.evaluate, evaluate, result["scores_evaluate"])):
                failed, _finished = _stops(data)
                for state in np.flatnonzero(failed):
                    case = str(data["episode_case"][data["state_episode"][state]])
                    stop_scores[f"{split}:{case}"][f"state_{reading}{'_counts' if counts else ''}"] = {
                        "score": float(scores[state]), "flagged": bool(scores[state] > result["threshold"])}
    for reading in ("shown", "inspect"):
        result = candidate_probe(train, evaluate, reading=reading)
        kept_scores[f"candidate_{reading}_{args.train}"] = result["scores_train"]
        kept_scores[f"candidate_{reading}_{args.evaluate}"] = result["scores_evaluate"]
        entry: dict[str, Any] = {"reading": reading, "train_auc": result["train_auc"], "evaluate_auc": result["evaluate_auc"]}
        for split, data, scores in ((args.train, train, result["scores_train"]), (args.evaluate, evaluate, result["scores_evaluate"])):
            failed, _finished = _stops(data)
            found = places(data, scores, failed)
            by_standing: dict[str, list[dict[str, Any]]] = defaultdict(list)
            by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in found:
                case = str(data["episode_case"][data["state_episode"][item["state"]]])
                record = records.get((split, case)) or {}
                standing = next((error["standing"] for error in (record.get("stop") or {}).get("errors_left") or []
                                 if error["object"] == item["object"] and error["kind"].replace("line_status", "line") == item["kind"]),
                                "unknown")
                item["standing"] = standing
                by_standing[standing].append(item)
                by_kind[item["kind"]].append(item)
                stop_scores[f"{split}:{case}"][f"candidate_{reading}"] = {
                    key: item[key] for key in ("kind", "object", "place", "place_within_kind", "place_by_first_statistic", "standing")}
            entry[split] = {
                "stops": len(found),
                "first": _share(found, "place", 1), "top_3": _share(found, "place", 3), "top_5": _share(found, "place", 5),
                "first_within_kind": _share(found, "place_within_kind", 1),
                "first_within_kind_by_first_statistic": _share(found, "place_by_first_statistic", 1),
                "by_standing": {name: {"stops": len(items), "first": _share(items, "place", 1), "top_3": _share(items, "place", 3)}
                                for name, items in sorted(by_standing.items())},
                "by_kind": {name: {"stops": len(items), "first": _share(items, "place", 1), "top_3": _share(items, "place", 3),
                                   "first_by_first_statistic": _share(items, "place_by_first_statistic", 1)}
                            for name, items in sorted(by_kind.items())},
            }
        report["candidate"].append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
    report["stopping_states"] = dict(sorted(stop_scores.items()))
    output = args.output or args.input_dir / "probe_report.json"
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_name("probe_scores.npz"), **kept_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
