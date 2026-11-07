"""Commit-record check, analysis: does a fuller record of a kept correction tell wrong from true?

    python -m research.compound_errors.commit_record_check \\
        --collected output/commit_record_check_20261105/collect/episodes_full.jsonl \\
        --reference output/item7_step7/compound/expert_baseline_compare_guarded_routes/episodes.jsonl \\
        --output-dir output/commit_record_check_20261105/check

Input: the rollout ``research.compound_errors.commit_record`` collected (the
compound study's teacher with every solve kept in full).  The reference
rollout is the stored one; the trajectories must agree action for action.

Unit: a kept correction (committed, never withdrawn).  Label: wrong when its
target is not a truly faulted component (``target_is_true`` of the rollout,
the study's "wrong commit").  The label is read from the truth audit and is
used only as the label.

Decision moments: an undo is only considered at an active state with no open
candidate whose alarm stands.  The information changes when an attempt ends,
so a row is written for every kept correction after every attempt end (a
commit, a rejected candidate, a refused correction) while the alarm stands.
The row of a correction's own commit is its "at commit" row.

Feature groups, all read from the agent's own solves, none from the truth,
none naming a component (no line, meter or bus identity):

``today``     what the state view already shows about a kept correction: its
              tool, its position, what happened since, the current solve's
              headline numbers and confidence, whether its target is in the
              current top-5 lists.
``verify``    the verification numbers at commit, the record a rejected
              hypothesis already keeps.
``estimate``  what the correction wrote: the impedance change, the meter
              change in sigma, the kind of breaker edit.
``reread``    its own target metric re-read on the later solves, and the
              drift of its estimate under later refits.
``context``   how it was chosen (confidence and rank at the parent solve) and
              the residuals and multipliers next to its target, at commit
              and now, read from the full vectors.
``summary``   the same reading of the target and its neighbourhood from what
              today's tool summaries list (the top five residuals and the top
              five multipliers), assuming the reader knows which lines meet
              at which bus.

Evaluation: grouped cross-validation by root (a root's rows never straddle
folds), gradient boosting, out-of-fold scores; AUC with a root-level
bootstrap interval; an online reading (a correction is flagged when its score
passes the threshold at any decision moment) against the fruitless-cycle rule.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECK_CONTRACT = "compound_commit_record_check_v1"
CHANNELS = ("Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt")
CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY = "correct_measurements", "correct_parameters", "correct_topology"
CORRECTIONS = {CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY}
RUN_WLS, COMMIT, ROLLBACK = "run_wls", "commit_state", "rollback_state"
TARGET_THRESHOLD = 3.0
BUDGET = 40
NAN = float("nan")


# --------------------------------------------------------------------------- one solve


class Solve:
    """One deployed solve: |normalized residual| per channel, the multiplier per line, the solved model."""

    def __init__(self, capture: Mapping[str, Any], metrics: Mapping[str, Any], step: int) -> None:
        self.step = step
        self.metrics = metrics
        self.state_id = str(capture["state_id"])
        self.nb, self.nl = int(capture["nb"]), int(capture["nl"])
        self.r = np.abs(np.asarray(capture["r"], dtype=float))
        lam = np.abs(np.asarray(capture["lam"], dtype=float))
        self.lam = lam.reshape(-1, 2).max(axis=1) if lam.size == 2 * self.nl else np.zeros(self.nl)
        self.z = np.asarray(capture["z"], dtype=float)
        sigma = np.asarray(capture.get("sigma") or [], dtype=float)
        self.sigma = sigma if sigma.size == self.z.size else None
        self.bus = [int(b) for b in capture["bus"]]
        self.row_of_bus = {bus: row for row, bus in enumerate(self.bus)}
        self.branch = np.asarray(capture["branch"], dtype=float)  # from, to, r, x, tap, status

    @property
    def alarm(self) -> bool:
        return not bool(self.metrics.get("no_material_anomaly_remaining"))

    def locate(self, index: int) -> tuple[str, int]:
        nb, nl = self.nb, self.nl
        if index < 3 * nb:
            return CHANNELS[index // nb], index % nb
        block, offset = divmod(index - 3 * nb, nl)
        return CHANNELS[3 + block], offset

    def index(self, channel: str, offset: int) -> int | None:
        block = CHANNELS.index(channel)
        if block < 3:
            return block * self.nb + offset if offset < self.nb else None
        return 3 * self.nb + (block - 3) * self.nl + offset if offset < self.nl else None

    def line_buses(self, row: int) -> tuple[int, int]:
        return int(self.branch[row, 0]), int(self.branch[row, 1])

    def lines_at(self, buses: Iterable[int]) -> list[int]:
        wanted = set(buses)
        return [row for row in range(self.nl) if self.branch[row, 5] > 0 and (int(self.branch[row, 0]) in wanted or int(self.branch[row, 1]) in wanted)]

    def flow_indices(self, row: int) -> list[int]:
        return [3 * self.nb + block * self.nl + row for block in range(4)]

    def bus_indices(self, buses: Iterable[int]) -> list[int]:
        out = []
        for bus in buses:
            row = self.row_of_bus.get(int(bus))
            if row is not None:
                out.extend((row, self.nb + row, 2 * self.nb + row))
        return out

    def conf(self) -> Mapping[str, Any]:
        record = self.metrics.get("hypothesis_confidence")
        return record if isinstance(record, Mapping) else {}

    def listed(self) -> tuple[list[tuple[str, int, float]], dict[int, float]]:
        """What today's WLS summary lists: the top residuals (channel, offset, |value|) and the top multipliers by line row."""
        summary = self.metrics.get("wls_summary") if isinstance(self.metrics.get("wls_summary"), Mapping) else {}
        residuals = [(str(item["channel"]), int(item["channel_offset"]), abs(_f(item.get("value"))))
                     for item in summary.get("top_residuals") or []
                     if item.get("channel") in CHANNELS and item.get("channel_offset") is not None]
        multipliers: dict[int, float] = {}
        for item in summary.get("top_lagrange") or []:
            if item.get("line_row0") is not None:
                row = int(item["line_row0"])
                multipliers[row] = max(multipliers.get(row, 0.0), abs(_f(item.get("value"))))
        return residuals, multipliers


# --------------------------------------------------------------------------- one correction's target


class Target:
    """What a correction edited, in terms that survive a re-rendered model: channels, branch rows, bus ids."""

    def __init__(self, step: Mapping[str, Any], parent: Solve, child: Solve | None) -> None:
        arguments = step.get("arguments") or {}
        self.tool = str(step["tool"])
        self.meters: list[tuple[str, int]] = []
        self.rows: list[int] = []
        self.buses: set[int] = set()
        self.kind = "meter"
        if self.tool == CORRECT_MEASUREMENTS:
            group = arguments.get("suspect_group")
            if group is None:
                index = arguments.get("index", arguments.get("measurement_index"))
                group = [index] if index is not None else []
            self.meters = [parent.locate(int(i)) for i in group]
            for channel, offset in self.meters:
                if channel in CHANNELS[:3]:
                    self.buses.add(parent.bus[offset])
                else:
                    self.buses.update(parent.line_buses(offset))
        elif arguments.get("line_index") is not None or arguments.get("line_index1") is not None or arguments.get("branch_row0") is not None:
            if arguments.get("branch_row0") is not None:
                row = int(arguments["branch_row0"])
            else:
                row = int(arguments.get("line_index1") or arguments.get("line_index")) - 1
            self.rows = [row]
            self.buses.update(parent.line_buses(row))
            self.kind = "parameter" if self.tool == CORRECT_PARAMETERS else "dangling"
        else:
            # A breaker that splits a bus: the lines whose terminal moved, the bus they left and the section they joined.
            self.kind = "split"
            if child is not None:
                for row in range(min(parent.nl, child.nl)):
                    before, after = parent.line_buses(row), child.line_buses(row)
                    if before != after:
                        self.rows.append(row)
                        self.buses.update(set(before) ^ set(after))
                self.buses.update(set(child.bus) - set(parent.bus))

    def own(self, solve: Solve) -> float:
        """The correction's own target metric on a solve: its meters' residual, its line's multiplier, its open line's flows."""
        values: list[float] = []
        if self.kind == "meter":
            for channel, offset in self.meters:
                index = solve.index(channel, offset)
                if index is not None and index < solve.r.size:
                    values.append(float(solve.r[index]))
        elif self.kind == "parameter":
            values.extend(float(solve.lam[row]) for row in self.rows if row < solve.nl)
        elif self.kind == "dangling":
            for row in self.rows:
                if row < solve.nl:
                    values.append(float(solve.lam[row]))
                    values.extend(float(solve.r[i]) for i in solve.flow_indices(row))
        else:
            for row in self.rows:
                if row < solve.nl:
                    values.append(float(solve.lam[row]))
            values.extend(float(solve.r[i]) for i in solve.bus_indices(self.buses) if (i // solve.nb) in (1, 2))
        return max(values, default=NAN)

    def own_indices(self, solve: Solve) -> set[int]:
        out: set[int] = set()
        for channel, offset in self.meters:
            index = solve.index(channel, offset)
            if index is not None:
                out.add(index)
        return out

    def neighbourhood(self, solve: Solve) -> tuple[float, float]:
        """Largest residual and largest multiplier next to the target, the target's own channel and line left out."""
        lines = solve.lines_at(self.buses)
        indices = set(solve.bus_indices(self.buses))
        for row in lines:
            indices.update(solve.flow_indices(row))
        indices -= self.own_indices(solve)
        if self.kind == "dangling":
            for row in self.rows:
                indices -= set(solve.flow_indices(row))
        indices = {i for i in indices if i < solve.r.size}
        residual = max((float(solve.r[i]) for i in indices), default=0.0)
        multiplier = max((float(solve.lam[row]) for row in lines if row not in self.rows or self.kind == "meter"), default=0.0)
        return residual, multiplier

    def listed(self, solve: Solve) -> dict[str, float]:
        """The target and its neighbourhood as today's summary shows them: only the listed top residuals and multipliers."""
        residuals, multipliers = solve.listed()
        own_meters = set(self.meters)
        own_flow_rows = set(self.rows) if self.kind == "dangling" else set()
        own_values, other_residuals = [], []
        for channel, offset, value in residuals:
            if (channel, offset) in own_meters or (channel in CHANNELS[3:] and offset in own_flow_rows):
                own_values.append(value)
                continue
            if channel in CHANNELS[:3]:
                buses = {solve.bus[offset]} if offset < solve.nb else set()
            else:
                buses = set(solve.line_buses(offset)) if offset < solve.nl else set()
            other_residuals.append((value, bool(buses & self.buses)))
        other_multipliers = []
        for row, value in multipliers.items():
            if self.kind != "meter" and row in self.rows:
                own_values.append(value)
                continue
            other_multipliers.append((value, bool(set(solve.line_buses(row)) & self.buses) if row < solve.nl else False))
        own_kind = sorted((v for _c, _o, v in residuals), reverse=True) if self.kind == "meter" else sorted(multipliers.values(), reverse=True)
        own = max(own_values, default=0.0)
        return {
            "own": own, "own_listed": float(bool(own_values)),
            "own_rank": float(own_kind.index(own)) if own_values and own in own_kind else 5.0,
            "adj_r": max((v for v, near in other_residuals if near), default=0.0),
            "adj_r_count": float(sum(near for _v, near in other_residuals)),
            "top_r_adjacent": float(bool(other_residuals) and max(other_residuals)[1]),
            "adj_lam": max((v for v, near in other_multipliers if near), default=0.0),
            "adj_lam_count": float(sum(near for _v, near in other_multipliers)),
            "top_lam_adjacent": float(bool(other_multipliers) and max(other_multipliers)[1]),
        }

    def touches(self, other: "Target") -> bool:
        return bool(self.buses & other.buses)


# --------------------------------------------------------------------------- an episode's rows


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return NAN
    return out if math.isfinite(out) else NAN


def _rank(values: np.ndarray, value: float) -> float:
    return float(np.sum(values > value + 1e-9)) if math.isfinite(value) else NAN


def _top5(solve: Solve, target: Target) -> float:
    summary = solve.metrics.get("wls_summary") if isinstance(solve.metrics.get("wls_summary"), Mapping) else {}
    listed_meters = {int(item["index0"]) for item in summary.get("top_residuals") or [] if item.get("index0") is not None}
    listed_rows = {int(item["line_row0"]) for item in summary.get("top_lagrange") or [] if item.get("line_row0") is not None}
    if target.kind == "meter":
        return float(bool(listed_meters & target.own_indices(solve)))
    return float(bool(listed_rows & set(target.rows)))


def _conf_features(prefix: str, solve: Solve | None) -> dict[str, float]:
    conf = solve.conf() if solve is not None else {}
    return {
        f"{prefix}_conf_meter": _f(conf.get("measurement")), f"{prefix}_conf_parameter": _f(conf.get("parameter")),
        f"{prefix}_conf_topology": _f(conf.get("topology")), f"{prefix}_competing": float(bool(conf.get("competing"))) if conf else NAN,
        f"{prefix}_open_terminal": _f(conf.get("open_terminal_confidence")),
    }


def episode_rows(episode: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decision rows of one episode and its accounting."""
    steps = episode.get("steps_full") or []
    truth_by_step = {int(c["step"]): c.get("target_is_true") for c in episode.get("corrections") or []}
    last_solve: dict[str, Solve] = {}
    solves: list[Solve] = []
    kept: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []  # every attempt end: commit / reject / refused, with its target when known
    rows: list[dict[str, Any]] = []
    active: str | None = None
    pending: dict[str, Any] | None = None
    account = Counter()

    def emit(event_step: int, event: str) -> None:
        # Every kept correction gets its "at commit" row; the later rows exist only while the alarm stands.
        solve = last_solve.get(active or "")
        if solve is None or not kept:
            return
        for position, k in enumerate(kept):
            own_commit = event == "commit" and k["commit_step"] == event_step
            if solve.alarm or own_commit:
                rows.append(decision_row(episode, k, position, kept, attempts, solves, solve, event_step, event))

    for index, step in enumerate(steps):
        tool = str(step.get("tool"))
        arguments = step.get("arguments") or {}
        success = step.get("status") == "success"
        if tool == RUN_WLS and success and isinstance(step.get("solve"), Mapping):
            solve = Solve(step["solve"], step.get("metrics") or {}, index)
            last_solve[solve.state_id] = solve
            solves.append(solve)
            if active is None:
                active = solve.state_id
        elif tool in CORRECTIONS:
            parent = last_solve.get(str(arguments.get("state_id")))
            if not success:
                account["refused"] += 1
                target = Target(step, parent, None) if parent is not None and (tool != CORRECT_TOPOLOGY or arguments.get("line_index") is not None) else None
                attempts.append({"step": index, "event": "refused", "target": target})
                pending = None
                emit(index, "refused")
            else:
                pending = {"step": index, "record": step, "parent": parent, "candidate": step.get("candidate_state_id")}
        elif tool in (COMMIT, ROLLBACK) and success and pending is not None:
            child = last_solve.get(str(pending["candidate"]))
            parent = pending["parent"]
            target = Target(pending["record"], parent, child) if parent is not None else None
            if tool == COMMIT:
                active = str(pending["candidate"])
                if target is not None and child is not None:
                    kept.append({"step": pending["step"], "commit_step": index, "record": pending["record"], "parent": parent,
                                 "child": child, "target": target, "final": str(step.get("disposition")) == "ACCEPT_FINAL",
                                 "wrong": truth_by_step.get(pending["step"]) is False,
                                 "label_known": truth_by_step.get(pending["step"]) is not None,
                                 "attempts_before": len(attempts)})
                else:
                    account["kept_without_solve"] += 1
                attempts.append({"step": index, "event": "commit", "target": target})
                pending = None
                emit(index, "commit")
            else:
                attempts.append({"step": index, "event": "reject", "target": target})
                pending = None
                emit(index, "reject")
    account["kept"] = len(kept)
    account["kept_wrong"] = sum(k["wrong"] for k in kept)
    account["kept_final"] = sum(k["final"] for k in kept)
    account["kept_wrong_final"] = sum(k["wrong"] and k["final"] for k in kept)
    final_solve = last_solve.get(active or "")
    account["alarm_at_end"] = int(final_solve is not None and final_solve.alarm)
    return rows, dict(account)


def decision_row(episode: Mapping[str, Any], k: Mapping[str, Any], position: int, kept: Sequence[Mapping[str, Any]],
                 attempts: Sequence[Mapping[str, Any]], solves: Sequence[Solve], now: Solve, event_step: int,
                 event: str) -> dict[str, Any]:
    target: Target = k["target"]
    parent: Solve = k["parent"]
    child: Solve = k["child"]
    record = k["record"]
    metrics = child.metrics
    tool = target.tool
    features: dict[str, float] = {}

    # ---- today: what the state view already shows about a kept correction
    later_attempts = [a for a in attempts if a["step"] > k["commit_step"]]
    rejected = [a for a in later_attempts if a["event"] == "reject"]
    refused = [a for a in later_attempts if a["event"] == "refused"]
    committed = [a for a in later_attempts if a["event"] == "commit"]
    features.update({
        "today_tool_meter": float(target.kind == "meter"), "today_tool_parameter": float(target.kind == "parameter"),
        "today_tool_dangling": float(target.kind == "dangling"), "today_tool_split": float(target.kind == "split"),
        "today_order": float(position), "today_kept_total": float(len(kept)), "today_commit_step": float(k["commit_step"]),
        "today_steps_since": float(event_step - k["commit_step"]), "today_kept_since": float(len(committed)),
        "today_rejected_since": float(len(rejected)), "today_refused_since": float(len(refused)),
        "today_fruitless_since": float(len(rejected) + len(refused)),
        "today_rejected_adjacent": float(sum(1 for a in rejected + refused if a["target"] is not None and target.touches(a["target"]))),
        "today_kept_adjacent": float(sum(1 for a in committed if a["target"] is not None and target.touches(a["target"]))),
        "today_chi_now": _f(now.metrics.get("chi_square_ratio")), "today_nr_now": float(now.r.max()), "today_lam_now": float(now.lam.max()),
        "today_breadth_now": _f(now.metrics.get("anomaly_breadth")), "today_in_top5_now": _top5(now, target),
        "today_budget_left": float(BUDGET - event_step),
    })
    features.update(_conf_features("today_now", now))

    # ---- verify: the verification numbers at commit (what a rejected hypothesis already keeps)
    features.update({
        "verify_final": float(k["final"]), "verify_target_after": _f(metrics.get("target_metric_value")),
        "verify_target_before": _f(metrics.get("parent_target_metric_value")), "verify_target_progress": _f(metrics.get("target_progress")),
        "verify_global_progress": _f(metrics.get("global_progress")), "verify_chi_after": _f(metrics.get("chi_square_ratio")),
        "verify_chi_before": _f(metrics.get("parent_anomaly_score")), "verify_nr_after": float(child.r.max()),
        "verify_lam_after": float(child.lam.max()), "verify_remaining_suspects": _f(metrics.get("remaining_suspect_count")),
        "verify_breadth_after": _f(metrics.get("anomaly_breadth")), "verify_physical_ok": _f(metrics.get("physical_constraints_ok")),
    })

    # ---- estimate: what the correction wrote
    estimate = {"estimate_log_x": NAN, "estimate_log_r": NAN, "estimate_pair_refit": NAN, "estimate_meter_sigma": NAN,
                "estimate_meter_group": NAN, "estimate_meter_vm": NAN, "estimate_meter_injection": NAN, "estimate_meter_reactive": NAN,
                "estimate_cue_on_line": NAN}
    if target.kind == "parameter":
        row = target.rows[0]
        r0, x0 = float(parent.branch[row, 2]), float(parent.branch[row, 3])
        r1, x1 = float(child.branch[row, 2]), float(child.branch[row, 3])
        estimate["estimate_log_x"] = math.log(abs(x1) / abs(x0)) if x0 and x1 else NAN
        estimate["estimate_log_r"] = math.log(abs(r1) / abs(r0)) if r0 and r1 else NAN
        estimate["estimate_pair_refit"] = float(isinstance((record.get("metrics") or {}).get("pair_refit"), Mapping))
    elif target.kind == "meter":
        changes = []
        for channel, offset in target.meters:
            i, j = parent.index(channel, offset), child.index(channel, offset)
            if i is not None and j is not None and parent.sigma is not None:
                changes.append(abs(float(parent.z[i] - child.z[j])) / max(float(parent.sigma[i]), 1e-12))
        estimate["estimate_meter_sigma"] = max(changes, default=NAN)
        estimate["estimate_meter_group"] = float(len(target.meters))
        estimate["estimate_meter_vm"] = float(any(c == "Vm" for c, _o in target.meters))
        estimate["estimate_meter_injection"] = float(any(c in ("Pinj", "Qinj") for c, _o in target.meters))
        estimate["estimate_meter_reactive"] = float(any(c.startswith("Q") for c, _o in target.meters))
    elif target.kind == "dangling":
        cue = parent.metrics.get("flow_at_noise_lines") or []
        estimate["estimate_cue_on_line"] = float((target.rows[0] + 1) in {int(line) for line in cue})
    features.update(estimate)

    # ---- reread: the correction's own target metric on the later solves, and the drift of its estimate
    later = [s for s in solves if s.step > child.step]
    own_later = [target.own(s) for s in later]
    own_later = [v for v in own_later if math.isfinite(v)]
    own_now = target.own(now)
    drift = NAN
    if target.kind == "parameter":
        row = target.rows[0]
        x_commit, x_now = float(child.branch[row, 3]), float(now.branch[row, 3]) if row < now.nl else NAN
        drift = abs(math.log(abs(x_now) / abs(x_commit))) if x_commit and x_now and math.isfinite(x_now) else NAN
    features.update({
        "reread_own_now": own_now, "reread_own_max_later": max(own_later, default=NAN),
        "reread_own_last": own_later[-1] if own_later else NAN,
        "reread_recrossed": float(max(own_later, default=0.0) >= TARGET_THRESHOLD) if own_later else NAN,
        "reread_own_at_commit": target.own(child), "reread_estimate_drift": drift, "reread_later_solves": float(len(later)),
    })

    # ---- context: how it was chosen, and what stands next to its target
    own_before = target.own(parent)
    if target.kind == "meter":
        rank, kind_max, other_max = _rank(parent.r, own_before), float(parent.r.max()), float(parent.lam.max())
    else:
        rank, kind_max, other_max = _rank(parent.lam, own_before), float(parent.lam.max()), float(parent.r.max())
    conf = parent.conf()
    family = {"meter": "measurement", "parameter": "parameter"}.get(target.kind, "topology")
    ranking = [str(item) for item in conf.get("ranking") or []]
    nbr_before, nbr_after, nbr_now = target.neighbourhood(parent), target.neighbourhood(child), target.neighbourhood(now)
    features.update({
        "context_own_before": own_before, "context_own_rank": rank,
        "context_own_to_kind_max": own_before / kind_max if kind_max > 0 and math.isfinite(own_before) else NAN,
        "context_own_to_other_max": own_before / other_max if other_max > 0 and math.isfinite(own_before) else NAN,
        "context_conf_own": _f(conf.get(family)), "context_conf_rank_own": float(ranking.index(family)) if family in ranking else NAN,
        "context_nr_before": float(parent.r.max()), "context_lam_before": float(parent.lam.max()),
        "context_breadth_before": _f(parent.metrics.get("anomaly_breadth")), "context_attempts_before": float(k["attempts_before"]),
        "context_nbr_r_before": nbr_before[0], "context_nbr_lam_before": nbr_before[1],
        "context_nbr_r_after": nbr_after[0], "context_nbr_lam_after": nbr_after[1],
        "context_nbr_r_now": nbr_now[0], "context_nbr_lam_now": nbr_now[1],
    })
    features.update(_conf_features("context_chosen", parent))

    # ---- summary: the same neighbourhood reading from what today's summaries list (top-5 residuals, top-5 multipliers)
    for moment, solve in (("before", parent), ("after", child), ("now", now)):
        for name, value in target.listed(solve).items():
            features[f"summary_{name}_{moment}"] = value

    return {
        "scenario_id": str(episode["scenario_id"]), "class": episode.get("class"), "success": bool(episode.get("success")),
        "correction_step": int(k["step"]), "event_step": int(event_step), "event": event, "at_commit": bool(event_step == k["commit_step"]),
        "kind": target.kind, "final": bool(k["final"]), "wrong": bool(k["wrong"]), "label_known": bool(k["label_known"]),
        "alarm_now": bool(now.alarm), "features": features,
    }


# --------------------------------------------------------------------------- the check


GROUPS = ("today", "verify", "estimate", "reread", "context")
ARMS = (
    ("today", ("today",)),
    ("+ verification at commit", ("today", "verify")),
    ("+ estimate written", ("today", "verify", "estimate")),
    ("+ own metric re-read", ("today", "verify", "estimate", "reread")),
    ("+ how chosen, neighbourhood", ("today", "verify", "estimate", "reread", "context")),
)


def columns_of(names: Sequence[str], groups: Sequence[str]) -> list[int]:
    return [i for i, name in enumerate(names) if name.split("_", 1)[0] in groups]


def fold_of(roots: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    unique = sorted(set(roots))
    order = np.random.default_rng(seed).permutation(len(unique))
    return {unique[i]: int(position % folds) for position, i in enumerate(order)}


def out_of_fold(x: np.ndarray, y: np.ndarray, roots: Sequence[str], *, folds: int, seed: int, train_mask: np.ndarray | None = None) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier

    assignment = fold_of(roots, folds, seed)
    fold = np.asarray([assignment[root] for root in roots])
    scores = np.full(len(y), np.nan)
    for held in range(folds):
        train = fold != held
        if train_mask is not None:
            train = train & train_mask
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=15,
                                               l2_regularization=1.0, class_weight="balanced", random_state=seed)
        model.fit(x[train], y[train])
        scores[fold == held] = model.predict_proba(x[fold == held])[:, 1]
    return scores


def auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2:
        return NAN
    return float(roc_auc_score(y, score))


def bootstrap_auc(y: np.ndarray, score: np.ndarray, roots: Sequence[str], *, draws: int = 500, seed: int = 0) -> tuple[float, float]:
    index_of: dict[str, list[int]] = defaultdict(list)
    for i, root in enumerate(roots):
        index_of[root].append(i)
    names = sorted(index_of)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        chosen = rng.integers(0, len(names), len(names))
        idx = np.concatenate([index_of[names[i]] for i in chosen])
        value = auc(y[idx], score[idx])
        if math.isfinite(value):
            values.append(value)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (NAN, NAN)


def load(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def identity(collected: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_id = {str(r["scenario_id"]): r for r in reference}
    same = sum(1 for r in collected if str(r["scenario_id"]) in by_id and r.get("tools") == by_id[str(r["scenario_id"])].get("tools")
               and bool(r.get("success")) == bool(by_id[str(r["scenario_id"])].get("success")))
    return {"collected": len(collected), "reference": len(reference), "identical": same,
            "solved_collected": sum(bool(r.get("success")) for r in collected), "errors": sum(bool(r.get("error")) for r in collected)}


def build(collected: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals: Counter = Counter()
    per_episode: dict[str, dict[str, Any]] = {}
    for episode in collected:
        episode_part, account = episode_rows(episode)
        rows.extend(episode_part)
        totals.update(account)
        per_episode[str(episode["scenario_id"])] = account
    return rows, {"totals": dict(totals), "per_episode": per_episode}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collected", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args(argv)

    collected = load(args.collected)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"contract": CHECK_CONTRACT, "collected": str(args.collected)}
    if args.reference:
        report["identity"] = identity(collected, load(args.reference))
        print("[identity]", json.dumps(report["identity"]), flush=True)
    rows, accounting = build(collected)
    report["accounting"] = accounting["totals"]
    print("[accounting]", json.dumps(accounting["totals"]), flush=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[rows] {len(rows)} decision rows -> {out / 'rows.jsonl'}", flush=True)
    report.update(analyze(rows, collected, folds=args.folds, seeds=args.seeds))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    text = render(report)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


# --------------------------------------------------------------------------- analysis


def _matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = sorted(rows[0]["features"])
    return names, np.asarray([[_f(row["features"].get(name)) for name in names] for row in rows], dtype=float)


def _mean_auc(x: np.ndarray, y: np.ndarray, roots: Sequence[str], masks: Mapping[str, np.ndarray], *, folds: int,
              seeds: Sequence[int]) -> tuple[dict[str, float], np.ndarray]:
    """Out-of-fold AUC per evaluation mask, averaged over seeds; the scores of the first seed are returned."""
    values: dict[str, list[float]] = defaultdict(list)
    first: np.ndarray | None = None
    for seed in seeds:
        scores = out_of_fold(x, y, roots, folds=folds, seed=seed)
        if first is None:
            first = scores
        for name, mask in masks.items():
            values[name].append(auc(y[mask], scores[mask]))
    return {name: float(np.nanmean(v)) for name, v in values.items()}, first if first is not None else np.zeros(len(y))


def _operating_points(score: np.ndarray, wrong: np.ndarray, solved: np.ndarray, roots: Sequence[str],
                      precisions: Sequence[float] = (0.5, 0.67, 0.8, 0.9)) -> list[dict[str, Any]]:
    """Corrections flagged at the loosest threshold that still meets each precision."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(wrong[order])
    count = np.arange(1, len(order) + 1)
    precision = hits / count
    out = []
    for target in precisions:
        ok = np.flatnonzero(precision >= target)
        if ok.size == 0:
            out.append({"precision_target": target, "flagged": 0, "wrong": 0, "true": 0, "true_in_solved": 0,
                        "episodes_with_wrong_flagged": 0, "threshold": None})
            continue
        cut = int(ok.max())
        chosen = order[: cut + 1]
        out.append({
            "precision_target": target, "threshold": float(score[order[cut]]), "flagged": int(cut + 1),
            "wrong": int(wrong[chosen].sum()), "true": int((1 - wrong[chosen]).sum()),
            "true_in_solved": int(((1 - wrong[chosen]) * solved[chosen]).sum()),
            "episodes_with_wrong_flagged": len({roots[i] for i in chosen if wrong[i]}),
        })
    return out


def analyze(rows: Sequence[Mapping[str, Any]], collected: Sequence[Mapping[str, Any]], *, folds: int, seeds: Sequence[int]) -> dict[str, Any]:
    names, x_all = _matrix(rows)
    wrong_all = np.asarray([int(row["wrong"]) for row in rows])
    roots_all = [str(row["scenario_id"]) for row in rows]
    at_commit = np.asarray([bool(row["at_commit"]) for row in rows])
    alarm = np.asarray([bool(row["alarm_now"]) for row in rows])
    final = np.asarray([bool(row["final"]) for row in rows])
    solved = np.asarray([int(row["success"]) for row in rows])
    kind = np.asarray([str(row["kind"]) for row in rows])
    keys = [(str(row["scenario_id"]), int(row["correction_step"])) for row in rows]
    report: dict[str, Any] = {"features": len(names), "rows": len(rows)}

    # ---- population
    commit_rows = np.flatnonzero(at_commit)
    population = {
        "kept": int(at_commit.sum()), "kept_wrong": int(wrong_all[at_commit].sum()),
        "kept_wrong_alarm_cleared_at_commit": int((wrong_all * final)[at_commit].sum()),
        "by_kind": {name: {"kept": int((at_commit & (kind == name)).sum()), "wrong": int(wrong_all[at_commit & (kind == name)].sum())}
                    for name in ("meter", "parameter", "dangling", "split")},
        "episodes_failed_with_wrong_kept": len({roots_all[i] for i in commit_rows if wrong_all[i] and not solved[i]}),
        "episodes_solved_with_wrong_kept": len({roots_all[i] for i in commit_rows if wrong_all[i] and solved[i]}),
        "episodes": len(collected), "episodes_solved": sum(bool(e.get("success")) for e in collected),
        "decision_rows_alarm": int(alarm.sum()), "decision_rows_later": int((alarm & ~at_commit).sum()),
    }
    reachable = {keys[i] for i in range(len(rows)) if alarm[i]}
    population["kept_with_a_decision_moment"] = len(reachable)
    population["kept_wrong_with_a_decision_moment"] = len({keys[i] for i in range(len(rows)) if alarm[i] and wrong_all[i]})
    report["population"] = population

    # ---- A. at commit: could the gate have refused it?  One row per kept correction, commit-time features only.
    commit_names = [n for n in names if not n.endswith("_now") and not n.startswith("today_now_") and not n.startswith("reread_")
                    and n not in {"today_steps_since", "today_kept_since", "today_rejected_since", "today_refused_since",
                                  "today_fruitless_since", "today_rejected_adjacent", "today_kept_adjacent"}]
    xa = x_all[at_commit][:, [names.index(n) for n in commit_names]]
    ya, ra = wrong_all[at_commit], [roots_all[i] for i in commit_rows]
    kind_a, final_a = kind[at_commit], final[at_commit]
    masks_a = {"all": np.ones(len(ya), bool), "partial (alarm stands)": ~final_a, "final (alarm cleared)": final_a,
               "meter": kind_a == "meter", "parameter": kind_a == "parameter", "breaker": np.isin(kind_a, ("dangling", "split"))}
    gate: dict[str, Any] = {"counts": {name: {"n": int(mask.sum()), "wrong": int(ya[mask].sum())} for name, mask in masks_a.items()}}
    tool_cols = [commit_names.index(n) for n in commit_names if n.startswith("today_tool_")]
    verify_cols = tool_cols + [commit_names.index(n) for n in commit_names if n.startswith("verify_")]
    gate["tool only"], _ = _mean_auc(xa[:, tool_cols], ya, ra, masks_a, folds=folds, seeds=seeds)
    gate["tool + verification"], _ = _mean_auc(xa[:, verify_cols], ya, ra, masks_a, folds=folds, seeds=seeds)
    gate["all commit-time"], gate_scores = _mean_auc(xa, ya, ra, masks_a, folds=folds, seeds=seeds)
    gate["all commit-time, interval"] = bootstrap_auc(ya, gate_scores, ra)
    gate["operating_points"] = _operating_points(gate_scores, ya, solved[at_commit], ra)
    report["at_commit"] = gate

    # ---- B. later decision moments: the alarm stands, at least the commit has ended; arms add record groups.
    use = alarm
    xb, yb = x_all[use], wrong_all[use]
    rb = [roots_all[i] for i in np.flatnonzero(use)]
    later_b, kind_b = ~at_commit[use], kind[use]
    keys_b = [keys[i] for i in np.flatnonzero(use)]
    last_of: dict[tuple[str, int], int] = {}
    for position, key in enumerate(keys_b):
        last_of[key] = position
    last_b = np.zeros(len(yb), bool)
    last_b[list(last_of.values())] = True
    masks_b = {"all decision moments": np.ones(len(yb), bool), "after the commit": later_b, "last moment of each correction": last_b,
               "after the commit, meter": later_b & (kind_b == "meter"), "after the commit, parameter": later_b & (kind_b == "parameter"),
               "after the commit, breaker": later_b & np.isin(kind_b, ("dangling", "split"))}
    arms: dict[str, Any] = {"counts": {name: {"n": int(mask.sum()), "wrong": int(yb[mask].sum())} for name, mask in masks_b.items()}}
    scores_by_arm: dict[str, np.ndarray] = {}
    for label, groups in ARMS:
        columns = columns_of(names, groups)
        arms[label], scores_by_arm[label] = _mean_auc(xb[:, columns], yb, rb, masks_b, folds=folds, seeds=seeds)
    full_label, today_label = ARMS[-1][0], ARMS[0][0]
    arms["interval, " + full_label] = {name: bootstrap_auc(yb[mask], scores_by_arm[full_label][mask], [rb[i] for i in np.flatnonzero(mask)])
                                      for name, mask in masks_b.items() if name in ("all decision moments", "after the commit")}
    arms["interval, " + today_label] = {name: bootstrap_auc(yb[mask], scores_by_arm[today_label][mask], [rb[i] for i in np.flatnonzero(mask)])
                                       for name, mask in masks_b.items() if name in ("all decision moments", "after the commit")}
    report["decision_moments"] = arms

    # ---- C. online reading: a correction is flagged when its score passes the threshold at any decision moment.
    def per_correction(values: np.ndarray) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
        best: dict[tuple[str, int], float] = {}
        meta: dict[tuple[str, int], tuple[int, int]] = {}
        for position, key in enumerate(keys_b):
            if key not in best or values[position] > best[key]:
                best[key] = float(values[position])
            meta[key] = (int(yb[position]), int(solved[np.flatnonzero(use)[position]]))
        ordered = sorted(best)
        return (ordered, np.asarray([best[k] for k in ordered]), np.asarray([meta[k][0] for k in ordered]),
                np.asarray([meta[k][1] for k in ordered]), [k[0] for k in ordered])

    online: dict[str, Any] = {}
    for label in (today_label, full_label):
        _keys, score, wrong, in_solved, root_names = per_correction(scores_by_arm[label])
        online[label] = {"corrections": len(score), "wrong": int(wrong.sum()), "auc": auc(wrong, score),
                         "operating_points": _operating_points(score, wrong, in_solved, root_names)}
        # Within an episode that keeps a wrong correction: is the top-scored kept correction a wrong one?
        by_root: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for s, w, root in zip(score, wrong, root_names):
            by_root[root].append((float(s), int(w)))
        with_wrong = {root: items for root, items in by_root.items() if any(w for _s, w in items)}
        mixed = {root: items for root, items in with_wrong.items() if any(not w for _s, w in items)}
        online[label]["episodes_with_wrong"] = len(with_wrong)
        online[label]["episodes_with_wrong_and_true"] = len(mixed)
        online[label]["top_scored_is_wrong_in_mixed"] = sum(1 for items in mixed.values() if max(items)[1] == 1)
        online[label]["chance_top_is_wrong_in_mixed"] = float(sum(sum(w for _s, w in items) / len(items) for items in mixed.values()))
    # The fruitless-cycle rule in the same currency: flagged once three attempts ended fruitless since the commit.
    fruitless = xb[:, names.index("today_fruitless_since")]
    _keys, score, wrong, in_solved, root_names = per_correction(fruitless)
    flagged = score >= 3
    online["fruitless cycles >= 3"] = {"flagged": int(flagged.sum()), "wrong": int(wrong[flagged].sum()), "true": int((1 - wrong[flagged]).sum()),
                                       "true_in_solved": int(((1 - wrong[flagged]) * in_solved[flagged]).sum()),
                                       "episodes_with_wrong_flagged": len({r for r, f, w in zip(root_names, flagged, wrong) if f and w})}
    report["online"] = online

    # ---- D. held-out error classes: train on the other thirteen.
    classes = np.asarray([str(row["class"]) for row in rows])[use]
    columns = columns_of(names, ARMS[-1][1])
    held = np.full(len(yb), np.nan)
    from sklearn.ensemble import HistGradientBoostingClassifier

    for name in sorted(set(classes.tolist())):
        test = classes == name
        if yb[~test].sum() == 0:
            continue
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=15,
                                               l2_regularization=1.0, class_weight="balanced", random_state=0)
        model.fit(xb[~test][:, columns], yb[~test])
        held[test] = model.predict_proba(xb[test][:, columns])[:, 1]
    report["held_out_class"] = {"all decision moments": auc(yb, held), "after the commit": auc(yb[later_b], held[later_b]),
                                "per_class": {name: {"n": int((classes == name).sum()), "wrong": int(yb[classes == name].sum()),
                                                     "auc": auc(yb[classes == name], held[classes == name])}
                                              for name in sorted(set(classes.tolist()))}}

    # ---- E. single features: how far one number goes alone (direction-free AUC).
    singles = []
    for column, name in enumerate(names):
        entry = {"feature": name}
        for label, mask, x, y in (("at_commit", np.ones(len(ya), bool), x_all[at_commit], ya), ("after_commit", later_b, xb, yb)):
            values = x[mask][:, column]
            known = np.isfinite(values)
            if known.sum() < 50 or len(set(y[mask][known].tolist())) < 2:
                entry[label] = NAN
                continue
            value = auc(y[mask][known], values[known])
            entry[label] = max(value, 1 - value)
            entry[label + "_n"] = int(known.sum())
        singles.append(entry)
    report["single_features"] = singles
    report["levels"] = information_levels(names, x_all, wrong_all, roots_all, at_commit, alarm, solved, keys, folds=folds, seeds=seeds)
    return report


# --------------------------------------------------------------------------- information levels

#: The evidence the rule expert itself reads: the verifier's numbers and the router's confidence and cue.
EXPERT_FEATURES = {"verify_final", "verify_target_after", "verify_target_before", "verify_target_progress", "verify_global_progress",
                   "verify_chi_after", "verify_chi_before", "verify_physical_ok", "context_conf_own", "context_conf_rank_own",
                   "estimate_cue_on_line"}
#: The agent's own history: when the correction was made and after how many attempts.
POSITION_FEATURES = {"today_order", "today_kept_total", "today_commit_step", "today_budget_left", "context_attempts_before"}
#: Shown in the tool outputs of the moment, beyond what the expert reads.
SUMMARY_FEATURES = {"verify_nr_after", "verify_lam_after", "verify_remaining_suspects", "verify_breadth_after", "context_nr_before",
                    "context_lam_before", "context_breadth_before", "estimate_meter_sigma", "estimate_meter_group", "estimate_meter_vm",
                    "estimate_meter_injection", "estimate_meter_reactive", "estimate_pair_refit"}
#: Only in the complete numerical results: every residual and multiplier, and the model's impedance before the edit.
FULL_FEATURES = {"context_own_before", "context_own_rank", "context_own_to_kind_max", "context_own_to_other_max", "estimate_log_x",
                 "estimate_log_r", "context_nbr_r_before", "context_nbr_lam_before", "context_nbr_r_after", "context_nbr_lam_after"}
#: Cues that lean on how this suite draws faults: no voltage-meter faults, impedance factors outside 0.5 to 2.
PRIOR_FEATURES = {"estimate_meter_vm", "estimate_meter_injection", "estimate_meter_reactive", "estimate_meter_sigma", "estimate_log_x",
                  "estimate_log_r"}
LEVEL_EXPERT, LEVEL_SUMMARY, LEVEL_FULL = "what the rule expert reads", "all the tool summaries show", "full solve numbers"


def level_columns(names: Sequence[str]) -> dict[str, list[str]]:
    expert = [n for n in names if n in EXPERT_FEATURES or n.startswith("today_tool_") or n.startswith("context_chosen_")]
    summary = expert + [n for n in names if n in SUMMARY_FEATURES or (n.startswith("summary_") and n.endswith(("_before", "_after")))]
    full = summary + [n for n in names if n in FULL_FEATURES]
    return {LEVEL_EXPERT: expert, LEVEL_SUMMARY: summary, LEVEL_FULL: full}


def information_levels(names: Sequence[str], x_all: np.ndarray, wrong_all: np.ndarray, roots_all: Sequence[str], at_commit: np.ndarray,
                       alarm: np.ndarray, solved: np.ndarray, keys: Sequence[tuple[str, int]], *, folds: int, seeds: Sequence[int]) -> dict[str, Any]:
    """The same learner on three amounts of information: the expert's evidence, the tool summaries, the full numbers."""
    column = {name: i for i, name in enumerate(names)}
    position = sorted(POSITION_FEATURES)
    out: dict[str, Any] = {}

    # ---- at commit: one row per kept correction
    commit_index = np.flatnonzero(at_commit)
    xa, ya = x_all[at_commit], wrong_all[at_commit]
    ra = [roots_all[i] for i in commit_index]
    everything = {"all": np.ones(len(ya), bool)}
    table: dict[str, Any] = {}
    for label, chosen in level_columns(names).items():
        entry: dict[str, Any] = {}
        variants = {"with history": chosen + position, "without history": chosen,
                    "without fault-size and channel cues": [n for n in chosen + position if n not in PRIOR_FEATURES]}
        for variant, feature_names in variants.items():
            result, scores = _mean_auc(xa[:, [column[n] for n in feature_names]], ya, ra, everything, folds=folds, seeds=seeds)
            entry[variant] = result["all"]
            if variant == "with history":
                entry["interval"] = bootstrap_auc(ya, scores, ra)
                entry["operating_points"] = _operating_points(scores, ya, solved[at_commit], ra)
            if variant == "without fault-size and channel cues":
                entry["operating_points_without_cues"] = _operating_points(scores, ya, solved[at_commit], ra)
        table[label] = entry
    shuffled = []
    full_names = level_columns(names)[LEVEL_FULL] + position
    for seed in seeds:
        permuted = np.random.default_rng(1000 + seed).permutation(ya)
        result, _scores = _mean_auc(xa[:, [column[n] for n in full_names]], permuted, ra, everything, folds=folds, seeds=[seed])
        shuffled.append(result["all"])
    table["shuffled labels, full solve numbers"] = float(np.mean(shuffled))
    out["at_commit"] = table

    # ---- later decision moments: the state view of the moment, then the same with the commit-time record kept
    use = alarm
    xb, yb = x_all[use], wrong_all[use]
    rb = [roots_all[i] for i in np.flatnonzero(use)]
    later = ~at_commit[use]
    masks = {"all decision moments": np.ones(len(yb), bool), "after the commit": later}
    view = [n for n in names if n.startswith("today_") or (n.startswith("summary_") and n.endswith("_now"))]
    levels = level_columns(names)
    record_summary = sorted(set(view) | set(levels[LEVEL_SUMMARY]))
    record_full = sorted(set(record_summary) | set(levels[LEVEL_FULL])
                         | {n for n in names if n.startswith("reread_") or (n.startswith("context_nbr_") and n.endswith("_now"))})
    later_table: dict[str, Any] = {}
    for label, feature_names in (("state view of the moment", view), ("+ the commit-time summaries kept on the record", record_summary),
                                 ("+ full solve numbers, then and now", record_full)):
        result, scores = _mean_auc(xb[:, [column[n] for n in feature_names]], yb, rb, masks, folds=folds, seeds=seeds)
        entry = dict(result)
        entry["interval after the commit"] = bootstrap_auc(yb[later], scores[later], [rb[i] for i in np.flatnonzero(later)])
        without = [n for n in feature_names if n not in PRIOR_FEATURES]
        entry["after the commit, without fault-size and channel cues"] = _mean_auc(
            xb[:, [column[n] for n in without]], yb, rb, masks, folds=folds, seeds=seeds)[0]["after the commit"]
        later_table[label] = entry
    out["later"] = later_table
    out["later_counts"] = {name: {"n": int(mask.sum()), "wrong": int(yb[mask].sum())} for name, mask in masks.items()}
    return out


def render(report: Mapping[str, Any]) -> str:
    lines: list[str] = ["# Commit-record check", ""]
    if "identity" in report:
        ident = report["identity"]
        lines.append(f"Trajectories identical to the stored rollout: {ident['identical']} of {ident['collected']} "
                     f"(solved {ident['solved_collected']}, errors {ident['errors']}).")
    population = report["population"]
    lines += ["", "## Population", "",
              f"- kept corrections {population['kept']}, wrong {population['kept_wrong']} "
              f"({population['kept_wrong_alarm_cleared_at_commit']} of them cleared the alarm at commit)",
              "- by kind: " + ", ".join(f"{name} {v['wrong']}/{v['kept']}" for name, v in population["by_kind"].items()),
              f"- failed episodes keeping a wrong correction {population['episodes_failed_with_wrong_kept']}, solved ones {population['episodes_solved_with_wrong_kept']}",
              f"- kept corrections with a decision moment (alarm standing) {population['kept_with_a_decision_moment']}, "
              f"wrong among them {population['kept_wrong_with_a_decision_moment']}",
              f"- decision rows {population['decision_rows_alarm']} ({population['decision_rows_later']} after the commit)"]
    gate = report["at_commit"]
    subsets = list(gate["counts"])
    lines += ["", "## At commit (one row per kept correction, commit-time numbers only)", "",
              "| features | " + " | ".join(f"{name} ({gate['counts'][name]['wrong']}/{gate['counts'][name]['n']})" for name in subsets) + " |",
              "| --- | " + " | ".join("---" for _ in subsets) + " |"]
    for label in ("tool only", "tool + verification", "all commit-time"):
        lines.append(f"| {label} | " + " | ".join(f"{gate[label][name]:.3f}" for name in subsets) + " |")
    low, high = gate["all commit-time, interval"]
    lines.append(f"\nAll commit-time features, all kept corrections: 95% interval {low:.3f} to {high:.3f} (roots resampled).")
    lines += ["", "| precision target | flagged | wrong | true | true in solved episodes | failed episodes reached |", "| --- | --- | --- | --- | --- | --- |"]
    for point in gate["operating_points"]:
        lines.append(f"| {point['precision_target']:.2f} | {point['flagged']} | {point['wrong']} | {point['true']} | {point['true_in_solved']} | {point['episodes_with_wrong_flagged']} |")
    arms = report["decision_moments"]
    subsets = list(arms["counts"])
    lines += ["", "## Decision moments (alarm standing): AUC, wrong against true kept corrections", "",
              "| record | " + " | ".join(f"{name} ({arms['counts'][name]['wrong']}/{arms['counts'][name]['n']})" for name in subsets) + " |",
              "| --- | " + " | ".join("---" for _ in subsets) + " |"]
    for label, _groups in ARMS:
        lines.append(f"| {label} | " + " | ".join(f"{arms[label][name]:.3f}" for name in subsets) + " |")
    for key, value in arms.items():
        if key.startswith("interval, "):
            lines.append(f"\n{key}: " + "; ".join(f"{name} {low:.3f} to {high:.3f}" for name, (low, high) in value.items()))
    online = report["online"]
    lines += ["", "## Online reading (flag a correction when its score passes the threshold at any decision moment)", ""]
    for label, entry in online.items():
        if "operating_points" not in entry:
            continue
        lines += [f"**{label}**: {entry['corrections']} corrections, {entry['wrong']} wrong, AUC of the per-correction maximum {entry['auc']:.3f}; "
                  f"episodes keeping a wrong and a true correction {entry['episodes_with_wrong_and_true']}, top-scored is wrong in "
                  f"{entry['top_scored_is_wrong_in_mixed']} (chance {entry['chance_top_is_wrong_in_mixed']:.1f})", "",
                  "| precision target | flagged | wrong | true | true in solved episodes | failed episodes reached |", "| --- | --- | --- | --- | --- | --- |"]
        for point in entry["operating_points"]:
            lines.append(f"| {point['precision_target']:.2f} | {point['flagged']} | {point['wrong']} | {point['true']} | {point['true_in_solved']} | {point['episodes_with_wrong_flagged']} |")
        lines.append("")
    rule = online["fruitless cycles >= 3"]
    lines.append(f"Fruitless-cycle rule (three attempts ended fruitless since the commit): flagged {rule['flagged']}, wrong {rule['wrong']}, "
                 f"true {rule['true']} ({rule['true_in_solved']} in solved episodes), failed episodes reached {rule['episodes_with_wrong_flagged']}.")
    levels = report.get("levels")
    if levels:
        commit = levels["at_commit"]
        lines += ["", "## The same learner on three amounts of information, at commit (AUC, one row per kept correction)", "",
                  "| information | with the agent's history | 95% interval | without the history | without fault-size and channel cues |",
                  "| --- | --- | --- | --- | --- |"]
        for label, entry in commit.items():
            if not isinstance(entry, Mapping):
                continue
            low, high = entry["interval"]
            lines.append(f"| {label} | {entry['with history']:.3f} | {low:.3f} to {high:.3f} | {entry['without history']:.3f} | "
                         f"{entry['without fault-size and channel cues']:.3f} |")
        lines += ["", f"Shuffled labels, full solve numbers: {commit['shuffled labels, full solve numbers']:.3f}."]
        for key, title in (("operating_points", "with the agent's history"), ("operating_points_without_cues", "without fault-size and channel cues")):
            lines += ["", f"Corrections a commit-time reader would refuse, {title}:", "",
                      "| information | precision target | refused | wrong | true | true in solved episodes | failed episodes reached |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for label, entry in commit.items():
                if not isinstance(entry, Mapping):
                    continue
                for point in entry[key]:
                    if point["precision_target"] in (0.8, 0.9):
                        lines.append(f"| {label} | {point['precision_target']:.2f} | {point['flagged']} | {point['wrong']} | {point['true']} | "
                                     f"{point['true_in_solved']} | {point['episodes_with_wrong_flagged']} |")
        counts = levels["later_counts"]
        lines += ["", "## The same, at later decision moments (alarm standing)", "",
                  f"| information | all decision moments ({counts['all decision moments']['wrong']}/{counts['all decision moments']['n']}) | "
                  f"after the commit ({counts['after the commit']['wrong']}/{counts['after the commit']['n']}) | 95% interval | "
                  "after the commit, without fault-size and channel cues |", "| --- | --- | --- | --- | --- |"]
        for label, entry in levels["later"].items():
            low, high = entry["interval after the commit"]
            lines.append(f"| {label} | {entry['all decision moments']:.3f} | {entry['after the commit']:.3f} | {low:.3f} to {high:.3f} | "
                         f"{entry['after the commit, without fault-size and channel cues']:.3f} |")
    held = report["held_out_class"]
    lines += ["", "## Held-out error class (trained on the other classes, full record)", "",
              f"All decision moments {held['all decision moments']:.3f}; after the commit {held['after the commit']:.3f}.", "",
              "| class | rows | wrong | AUC |", "| --- | --- | --- | --- |"]
    for name, entry in held["per_class"].items():
        value = entry["auc"]
        lines.append(f"| {name} | {entry['n']} | {entry['wrong']} | {'n/a' if not math.isfinite(value) else f'{value:.3f}'} |")
    lines += ["", "## Single features (direction-free AUC)", "", "| feature | at commit | after the commit |", "| --- | --- | --- |"]
    ranked = sorted(report["single_features"], key=lambda e: -max(e.get("at_commit") if math.isfinite(e.get("at_commit", NAN)) else 0,
                                                                    e.get("after_commit") if math.isfinite(e.get("after_commit", NAN)) else 0))
    for entry in ranked[:22]:
        a, b = entry.get("at_commit", NAN), entry.get("after_commit", NAN)
        lines.append(f"| {entry['feature']} | {'n/a' if not math.isfinite(a) else f'{a:.3f}'} | {'n/a' if not math.isfinite(b) else f'{b:.3f}'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
