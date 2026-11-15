"""The one constructor of what the model reads.

``messages(observation)`` turns the environment's observation into the system
and user turns, and ``tools(observation)`` into the tool list.  Training rows
and the run-time policy both call these two functions, so a model never reads
one layout in training and another when it acts.

The user turn has three parts, as plain text with the tables as comma-separated
rows (a table costs a third of the tokens of the same rows as JSON objects):

* the action history: every call, its result and the version it acted on;
* the modifications: for each correction what actually changed in the state
  (read from the saved states, not from the tool's report), the estimator's
  numbers before and after, and whether the correction is still in effect;
* the versions: each saved state's estimate.  The versions the environment
  shows in full (the most recent ones, ``MinimalDiagnosisEnv.table_versions``)
  carry their tables of buses, lines and meters and, where inspected, of
  breakers; the others keep their summary numbers and are one ``inspect``
  away.  The current version comes last.

Nothing here ranks, filters or advises.  The system prompt says what the task
is and what the columns mean; it names no rule, because the environment has
none.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from psse_env.minimal.actions import tool_schemas
from psse_env.minimal.env import CONTRACT

TASK = (
    "You are responsible for diagnosing and correcting errors in a power grid's model and measurements. The errors "
    "can be {topology_errors}, wrong series parameters of lines, and bad measurements; a case can hold several at "
    "once.\n\n"
    "From the numerical evidence and the outcomes of your earlier actions, choose the next tool call and its target. "
    "Tools return computed results; they do not decide whether your diagnosis is correct. You may inspect, correct, "
    "undo or finish. Every correction is saved as a new version of the state and can be undone. Your goal is to "
    "complete the diagnosis accurately, avoid damaging healthy components, and avoid unnecessary operations."
)
LEGEND = (
    "Reading the state:\n"
    "- Estimate: the state estimator's result on a version. Chi-square is its weighted sum of squared residuals, "
    "given with the threshold of the chi-square test; the largest normalized residual is given with its threshold.\n"
    "- A normalized residual is measured minus estimated, in standard deviations of the measurement.\n"
    "- buses: v and angle are the estimated voltage magnitude (per unit) and angle (degrees); vm, p, q are the "
    "normalized residuals of the bus's voltage magnitude, active injection and reactive injection measurements.\n"
    "- lines: r and x are the model's series resistance and reactance (per unit); lr and lx are their normalized "
    "Lagrange multipliers; pf, qf, pt, qt are the normalized residuals of the active and reactive flow measurements "
    "at the from end and the to end; note marks a transformer (xfmr) or a line out of service (out).\n"
    "- meters: index identifies a measurement; channel is Vm, Pinj or Qinj (at a bus) or Pf, Qf, Pt or Qt (at a "
    "line, from end or to end); value is the reading (per unit); residual is its normalized residual.\n"
    "- An empty cell means the quantity does not exist there.\n"
    "- The tables are shown for the most recent versions. For a version marked '(tables not shown)', inspect "
    "returns them."
)
BREAKER_LEGEND = (
    "\n- breakers (shown for a version once inspected): status is the reported status; theta and vm are the "
    "normalized multipliers of the breaker's two constraints in the substation estimate, and score is the larger "
    "magnitude; flip_effect is what changing the status would do to the operator model, with the line it would "
    "disconnect or reconnect or the bus it would split or merge; flip_chi_square is the substation estimate's "
    "chi-square with that one status changed, for the breakers that were re-estimated, and flip_progress is the "
    "relative reduction."
)
CLOSING = "Reply with exactly one tool call."

BUS_COLUMNS = ("bus", "type", "v", "angle", "vm", "p", "q")
LINE_COLUMNS = ("line", "from", "to", "r", "x", "lr", "lx", "pf", "qf", "pt", "qt", "note")
METER_COLUMNS = ("index", "channel", "at", "value", "residual")
BREAKER_COLUMNS = ("cb", "status", "score", "theta", "vm", "flip_effect", "line", "bus", "flip_chi_square", "flip_progress")


def system_prompt(observation: Mapping[str, Any]) -> str:
    breakers = bool(observation.get("breaker_model"))
    task = TASK.format(topology_errors="wrong breaker statuses" if breakers else "wrong line service statuses")
    return f"{task}\n\n{LEGEND}{BREAKER_LEGEND if breakers else ''}\n\n{CLOSING}"


def tools(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The tool list of an observation, in the form the chat template renders (every schema node described)."""
    from psse_env.sft.gemma_text import sanitize_tool_schemas

    return sanitize_tool_schemas(tool_schemas(breakers=bool(observation.get("breaker_model"))))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [",".join(columns)] + [",".join(_cell(row.get(column)) for column in columns) for row in rows]


def _number(value: Any, decimals: int) -> str:
    return "n/a" if value is None else f"{float(value):.{decimals}f}"


def _estimate_line(estimate: Mapping[str, Any], label: str = "Estimate") -> str:
    if not estimate.get("converged"):
        reason = f" ({estimate['error']})" if estimate.get("error") else ""
        return f"{label}: did not converge{reason}."
    text = (f"{label}: converged; chi-square {_number(estimate.get('chi_square'), 1)}, threshold "
            f"{_number(estimate.get('chi_square_threshold'), 1)} ({estimate.get('chi_square_dof')} degrees of freedom); "
            f"largest normalized residual {_number(estimate.get('max_residual'), 2)}")
    if estimate.get("residual_threshold") is not None:
        text += f", threshold {_number(estimate.get('residual_threshold'), 1)}"
    return text + "."


def _brief(estimate: Mapping[str, Any] | None) -> str:
    if not estimate:
        return "n/a"
    if not estimate.get("converged"):
        return "did not converge"
    return f"chi-square {_number(estimate.get('chi_square'), 1)}, largest normalized residual {_number(estimate.get('max_residual'), 2)}"


def _sigma_text(meters: Sequence[Mapping[str, Any]]) -> str | None:
    """The declared standard deviations: one value per channel, and the meters that differ from their channel's.

    Lossless and short: a column would repeat the same two numbers on nearly
    every row.  A structural row (no residual) is not a meter and is left out.
    """
    if not meters or any("sigma" not in row for row in meters):
        return None
    by_channel: dict[str, list[tuple[int, float]]] = {}
    for row in meters:
        if row.get("residual") is not None:
            by_channel.setdefault(str(row.get("channel")), []).append((int(row["index"]), float(row["sigma"])))
    usual: dict[str, float] = {}
    for channel, items in by_channel.items():
        values = [value for _, value in items]
        usual[channel] = max(sorted(set(values)), key=values.count)
    by_value: dict[float, list[str]] = {}
    for channel, value in usual.items():
        by_value.setdefault(value, []).append(channel)
    parts = [f"{', '.join(channels)} {value:g}" for value, channels in sorted(by_value.items())]
    other: dict[float, list[int]] = {}
    for channel, items in by_channel.items():
        for index, value in items:
            if value != usual[channel]:
                other.setdefault(value, []).append(index)
    for value, indices in sorted(other.items()):
        parts.append(f"meters {', '.join(str(index) for index in sorted(indices))} {value:.6g}")
    return "standard deviation: " + "; ".join(parts)


def _line_rows(lines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in lines:
        note = " ".join(name for name in ("xfmr", "out") if row.get(name))
        rows.append({**row, "note": note or None})
    return rows


def _action_text(action: Mapping[str, Any]) -> str:
    if not action.get("tool"):
        return "(unreadable call)"
    return f"{action['tool']} {json.dumps(action.get('arguments') or {}, sort_keys=True, separators=(',', ':'))}"


def _result_text(result: Mapping[str, Any]) -> str:
    if result.get("status") != "ok":
        detail = f": {result['error_detail']}" if result.get("error_detail") else ""
        return f"error {result.get('error_code')}{detail}"
    if "modification" in result:
        return f"ok: modification {result['modification']}, version {result['from_version']} -> {result['to_version']}"
    if "substation" in result:
        return f"ok: substation estimate of version {result['version']}, {_brief(result['substation'])}"
    if "view" in result:
        return f"ok: tables of version {result['version']}"
    if "outcome" in result:
        return f"ok: finished, {result['outcome']}"
    if "to_version" in result:
        return f"ok: version {result['from_version']} -> {result['to_version']}"
    return "ok"


def _change_text(change: Mapping[str, Any]) -> str:
    kind = change.get("object")
    if kind == "breaker":
        return f"breaker {change['breaker']} {change['before']} -> {change['after']}"
    if kind == "line":
        return f"line {change['line']} {change['field']} {_cell(change['before'])} -> {_cell(change['after'])}"
    if kind == "meter":
        return f"meter {change['meter']} {_cell(change['before'])} -> {_cell(change['after'])}"
    if kind == "operator_model":
        return (f"operator model rebuilt: buses {change['buses_before']} -> {change['buses_after']}, measurements "
                f"{change['measurements_before']} -> {change['measurements_after']}")
    return json.dumps(change, sort_keys=True)


def _version_block(entry: Mapping[str, Any], *, current: bool, created_by: Mapping[int, int]) -> list[str]:
    index = int(entry["version"])
    origin = "initial state" if entry.get("parent") is None else (
        f"version {entry['parent']} with modification {created_by.get(index, '?')}")
    marks = [origin] + (["current"] if current else []) + ([] if entry.get("in_effect") else ["not in effect"])
    lines = [f"== Version {index} ({'; '.join(marks)}) ==", _estimate_line(entry["estimate"])]
    estimate = entry["estimate"]
    if "meters" in estimate:
        lines += ["buses"] + _table(BUS_COLUMNS, estimate.get("buses") or [])
        lines += ["lines"] + _table(LINE_COLUMNS, _line_rows(estimate.get("lines") or []))
        meters = estimate.get("meters") or []
        sigma = _sigma_text(meters)
        lines += ["meters" + (f" ({sigma})" if sigma else "")]
        lines += _table(METER_COLUMNS, meters)
    elif estimate.get("converged"):
        lines.append("(tables not shown)")
    substation = entry.get("substation")
    if substation is not None:
        lines.append(_estimate_line(substation, "Substation estimate"))
        if "breakers" in substation:
            lines += ["breakers"] + _table(BREAKER_COLUMNS, substation["breakers"])
        elif substation.get("converged"):
            lines.append("(breaker table not shown)")
    return lines


def user_text(observation: Mapping[str, Any]) -> str:
    """The user turn of an observation."""
    if observation.get("contract") != CONTRACT:
        raise ValueError(f"not an observation of {CONTRACT}")
    budget = observation["budget"]
    versions = {int(entry["version"]): entry for entry in observation["versions"]}
    current = int(observation["current_version"])
    created_by = {int(record["to_version"]): int(record["id"]) for record in observation["modifications"]}
    out = [f"Actions used: {budget['used']} of {budget['limit']}.", "", "== History =="]
    if not observation["history"]:
        out.append("(no action yet)")
    for event in observation["history"]:
        out.append(f"{event['step']}. {_action_text(event['action'])} -> {_result_text(event['result'])}")
    out += ["", "== Modifications =="]
    if not observation["modifications"]:
        out.append("(none)")
    for record in observation["modifications"]:
        before = versions[int(record["from_version"])]["estimate"]
        after = versions[int(record["to_version"])]["estimate"]
        state = "in effect" if record.get("in_effect") else "undone"
        out.append(f"{record['id']}. step {record['step']}, {_action_text(record)}: version {record['from_version']} -> "
                   f"{record['to_version']}, {state}")
        out.append("   changed: " + "; ".join(_change_text(change) for change in record["changes"]))
        out.append(f"   estimate before: {_brief(before)}; after: {_brief(after)}")
    for index in sorted(versions):
        if index != current:
            out += [""] + _version_block(versions[index], current=False, created_by=created_by)
    out += [""] + _version_block(versions[current], current=True, created_by=created_by)
    return "\n".join(out)


def messages(observation: Mapping[str, Any]) -> list[dict[str, str]]:
    """The system and user turns of an observation: the prompt of training rows and of the run-time policy alike."""
    return [{"role": "system", "content": system_prompt(observation)},
            {"role": "user", "content": user_text(observation)}]


def target_call(action: Mapping[str, Any]) -> dict[str, Any]:
    """An action as the assistant turn of a training row."""
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_0", "type": "function",
        "function": {"name": str(action["tool"]), "arguments": dict(action.get("arguments") or {})},
    }]}


__all__ = ["messages", "system_prompt", "target_call", "tools", "user_text"]
