"""Detailed evidence of a correction and of the solve that verifies it (RL plan of 2026-11-05).

The tool outputs the policy reads were shaped by what the rule expert uses:
a solve lists its five largest residuals and multipliers, a correction says
which component it edited but not what it wrote, and a verification reports
the target's own metric and the global progress.  The commit-record check of
2026-11-05 found that what tells a wrong accepted correction from a true one
sits elsewhere in the same solves: the size of the edit, and what still stands
on the lines and buses next to the edited component.

This module computes those numbers from solves that were made anyway, for
every correction of every kind (so nothing in their presence names an error
type), and reads no truth:

``edit``
    what a correction wrote: R and X before and after, a meter's value before
    and after with the change in sigma, a breaker's status before and after.
``target_neighbourhood``
    for the verification of a candidate: the edited component's own target
    metric and the largest normalized residual and multiplier on the lines
    and buses that meet it, before the edit (the parent's solve) and after.

The per-bus and per-line tables of a solve are ``psse_env.providers.wls_tables``.
Pure functions; the provider emits them only when built with
``detailed_evidence`` and the prompt admits them under the one key
``detailed_evidence``, so every recorded output and prompt is unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.actions import DETAILED_EVIDENCE_CONTRACT

CHANNELS = ("Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt")
F_BUS, T_BUS, BR_R, BR_X, BR_STATUS = 0, 1, 2, 3, 10
DECIMALS = 2
_MEASUREMENT_TOOL, _PARAMETER_TOOL, _TOPOLOGY_TOOL = "correct_measurements", "correct_parameters", "correct_topology"


def _round(value: float) -> float | None:
    return float(round(float(value), DECIMALS)) if value is not None and math.isfinite(float(value)) else None


def solve_view(payload: Mapping[str, Any], ppc: Mapping[str, Any]) -> dict[str, Any] | None:
    """What the neighbourhood reading needs of one solve: |residual| per channel, the multiplier per line, the model's layout."""
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    nb, nl = int(bus.shape[0]), int(branch.shape[0])
    residual = np.abs(np.asarray(payload.get("r") or [], dtype=float))
    if residual.size != 3 * nb + 4 * nl:
        return None
    lam = np.abs(np.asarray(payload.get("lambdaN") or [], dtype=float))
    lam = lam.reshape(-1, 2).max(axis=1) if lam.size == 2 * nl else np.zeros(nl)
    status = branch[:, BR_STATUS] if branch.shape[1] > BR_STATUS else np.ones(nl)
    return {
        "nb": nb, "nl": nl, "r": residual, "lam": lam, "bus": [int(b) for b in bus[:, 0]],
        "ends": [(int(row[F_BUS]), int(row[T_BUS])) for row in branch], "in_service": [bool(s > 0) for s in status],
    }


def locate(index: int, nb: int, nl: int) -> tuple[str, int]:
    """The channel and the offset inside it (a bus row or a branch row) of a measurement index."""
    if index < 3 * nb:
        return CHANNELS[index // nb], index % nb
    block, offset = divmod(index - 3 * nb, nl)
    return CHANNELS[3 + block], offset


def _index(view: Mapping[str, Any], channel: str, offset: int) -> int | None:
    block = CHANNELS.index(channel)
    if block < 3:
        return block * view["nb"] + offset if offset < view["nb"] else None
    return 3 * view["nb"] + (block - 3) * view["nl"] + offset if offset < view["nl"] else None


def _flow_indices(view: Mapping[str, Any], row: int) -> list[int]:
    return [3 * view["nb"] + block * view["nl"] + row for block in range(4)]


def _bus_indices(view: Mapping[str, Any], buses: Sequence[int]) -> list[int]:
    out: list[int] = []
    for bus in buses:
        if bus in view["bus"]:
            row = view["bus"].index(bus)
            out.extend((row, view["nb"] + row, 2 * view["nb"] + row))
    return out


def correction_target(source_action: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What a correction edited, in terms that survive a re-rendered model: channels, branch rows, bus numbers."""
    tool = str(source_action.get("tool") or "")
    arguments = source_action.get("arguments") if isinstance(source_action.get("arguments"), Mapping) else {}
    meters: list[tuple[str, int]] = []
    rows: list[int] = []
    buses: set[int] = set()
    if tool == _MEASUREMENT_TOOL:
        group = arguments.get("suspect_group")
        if group is None and isinstance(arguments.get("measurement_updates"), Mapping):
            group = list(arguments["measurement_updates"])
        try:
            indices = [int(i) for i in (group or [])]
        except (TypeError, ValueError):
            return None
        if not indices or any(not 0 <= i < before["r"].size for i in indices):
            return None
        meters = [locate(i, before["nb"], before["nl"]) for i in indices]
        for channel, offset in meters:
            buses.update([before["bus"][offset]] if channel in CHANNELS[:3] else before["ends"][offset])
        return {"kind": "meter", "meters": meters, "rows": rows, "buses": buses}
    if tool not in (_PARAMETER_TOOL, _TOPOLOGY_TOOL):
        return None
    row = arguments.get("branch_row0")
    if row is None:
        line = arguments.get("line_index1", arguments.get("line_index"))
        row = int(line) - 1 if isinstance(line, int) and not isinstance(line, bool) else None
    if row is not None:
        if not 0 <= int(row) < before["nl"]:
            return None
        rows = [int(row)]
        buses.update(before["ends"][int(row)])
        return {"kind": "parameter" if tool == _PARAMETER_TOOL else "open_line", "meters": meters, "rows": rows, "buses": buses}
    # A breaker that splits or merges a bus: the lines whose terminal moved, the bus they left and the section they joined.
    if after is None:
        return None
    for index in range(min(before["nl"], after["nl"])):
        if before["ends"][index] != after["ends"][index]:
            rows.append(index)
            buses.update(set(before["ends"][index]) ^ set(after["ends"][index]))
    buses.update(set(after["bus"]) - set(before["bus"]))
    return {"kind": "bus_section", "meters": meters, "rows": rows, "buses": buses}


def own_metric(target: Mapping[str, Any], view: Mapping[str, Any]) -> float | None:
    """The correction's own target metric on a solve: its meters' residual, its line's multiplier, its open line's flows."""
    values: list[float] = []
    kind = target["kind"]
    if kind == "meter":
        for channel, offset in target["meters"]:
            index = _index(view, channel, offset)
            if index is not None:
                values.append(float(view["r"][index]))
    elif kind == "parameter":
        values.extend(float(view["lam"][row]) for row in target["rows"] if row < view["nl"])
    elif kind == "open_line":
        for row in target["rows"]:
            if row < view["nl"]:
                values.append(float(view["lam"][row]))
                values.extend(float(view["r"][i]) for i in _flow_indices(view, row))
    else:
        values.extend(float(view["lam"][row]) for row in target["rows"] if row < view["nl"])
        values.extend(float(view["r"][i]) for i in _bus_indices(view, sorted(target["buses"])) if (i // view["nb"]) in (1, 2))
    return max(values) if values else None


def neighbourhood(target: Mapping[str, Any], view: Mapping[str, Any]) -> tuple[float, float, list[int]]:
    """Largest residual and largest multiplier next to the target (its own channel and line left out), and the lines read."""
    lines = [row for row in range(view["nl"]) if view["in_service"][row] and set(view["ends"][row]) & target["buses"]]
    indices = set(_bus_indices(view, sorted(target["buses"])))
    for row in lines:
        indices.update(_flow_indices(view, row))
    for channel, offset in target["meters"]:
        own = _index(view, channel, offset)
        if own is not None:
            indices.discard(own)
    if target["kind"] == "open_line":
        for row in target["rows"]:
            indices -= set(_flow_indices(view, row))
    residual = max((float(view["r"][i]) for i in indices if i < view["r"].size), default=0.0)
    others = [row for row in lines if target["kind"] == "meter" or row not in target["rows"]]
    multiplier = max((float(view["lam"][row]) for row in others), default=0.0)
    return residual, multiplier, lines


def target_neighbourhood(source_action: Mapping[str, Any], before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The verification's neighbourhood block: own metric and adjacent maxima, before (the parent's solve) and after the edit."""
    if after is None:
        return None
    target = correction_target(source_action, before if before is not None else after, after)
    if target is None:
        return None
    after_residual, after_multiplier, lines = neighbourhood(target, after)
    block: dict[str, Any] = {
        "target": {"kind": target["kind"], "lines": [row + 1 for row in target["rows"]], "buses": sorted(target["buses"])},
        "adjacent_lines": [row + 1 for row in lines if target["kind"] == "meter" or row not in target["rows"]][:12],
        "own_metric": {"after": _round(own_metric(target, after) if own_metric(target, after) is not None else float("nan"))},
        "adjacent_residual_max": {"after": _round(after_residual)},
        "adjacent_multiplier_max": {"after": _round(after_multiplier)},
    }
    if before is not None:
        before_residual, before_multiplier, _lines = neighbourhood(target, before)
        own_before = own_metric(target, before)
        block["own_metric"]["before"] = _round(own_before if own_before is not None else float("nan"))
        block["adjacent_residual_max"]["before"] = _round(before_residual)
        block["adjacent_multiplier_max"]["before"] = _round(before_multiplier)
    return block


def parameter_edit(line_index: int, before: Sequence[float], after: Sequence[float]) -> dict[str, Any]:
    """R and X of one line before and after a parameter correction, with the ratio new over old."""
    out: dict[str, Any] = {"line": int(line_index)}
    for name, old, new in (("r", before[0], after[0]), ("x", before[1], after[1])):
        old, new = float(old), float(new)
        out[name] = {"before": float(round(old, 5)), "after": float(round(new, 5)),
                     "ratio": float(round(new / old, 3)) if old else None}
    return out


def measurement_edit(updates: Mapping[int, float], previous: Sequence[float], sigma: Sequence[float] | None, nb: int, nl: int) -> list[dict[str, Any]]:
    """Each corrected meter: its channel, its value before and after, and the change in sigma."""
    out = []
    for index in sorted(int(i) for i in updates):
        if not 0 <= index < len(previous):
            continue
        channel, offset = locate(index, nb, nl)
        old, new = float(previous[index]), float(updates[index])
        item: dict[str, Any] = {"index0": index, "channel": channel, "channel_offset": int(offset),
                                "before": float(round(old, 5)), "after": float(round(new, 5))}
        if sigma is not None and index < len(sigma) and float(sigma[index]) > 0:
            item["change_sigma"] = _round((new - old) / float(sigma[index]))
        out.append(item)
    return out


def topology_edit(output: Mapping[str, Any]) -> dict[str, Any] | None:
    """A breaker or branch-status correction's own record of what it changed, in one place."""
    edit: dict[str, Any] = {}
    if output.get("cb_name") is not None:
        edit["breaker"] = str(output["cb_name"])
        if output.get("cb_previous_status") is not None or output.get("cb_new_status") is not None:
            edit["breaker_status"] = {"before": output.get("cb_previous_status"), "after": output.get("cb_new_status")}
    if output.get("breaker_effect") is not None:
        edit["effect"] = str(output["breaker_effect"])
    if output.get("line_index") is not None:
        edit["line"] = int(output["line_index"])
    if output.get("previous_status") is not None or output.get("new_status") is not None:
        edit["line_status"] = {"before": output.get("previous_status"), "after": output.get("new_status")}
    return edit or None


__all__ = ["DETAILED_EVIDENCE_CONTRACT", "correction_target", "locate", "measurement_edit", "neighbourhood", "own_metric",
           "parameter_edit", "solve_view", "target_neighbourhood", "topology_edit"]
