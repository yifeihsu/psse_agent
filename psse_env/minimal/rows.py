"""Supervised rows of the minimal interface.

A row is one decision: the prompt the model would read at that step and the
action taken, as a native tool call.  The prompt is built by
``psse_env.minimal.observation`` from the observation the environment
returned when the episode ran, the same two functions the run-time policy
calls, so training and acting cannot drift apart.

A row carries labels beside its messages (``row["minimal"]``): which kind of
demonstration and step it is, whether its sequence ended repaired, the
case's class.  A selection reads them; nothing of them is in the messages.
"""
from __future__ import annotations

from typing import Any, Mapping

from psse_env.minimal import observation as view
from psse_env.minimal.truth import KIND_ERROR, KIND_PREFIX, TARGET_KINDS

ROW_CONTRACT = "minimal_sft_row_v1"
#: ``metadata.protocol`` of these rows: the interface their prompts and tool calls belong to.
MINIMAL_PROTOCOL = "minimal_env_v1"


def step_row(root: Mapping[str, Any], step: Mapping[str, Any], *, example_id: str, labels: Mapping[str, Any]) -> dict[str, Any]:
    """The row of one step: the observation it was chosen on, and its action as the target."""
    observation = step["observation"]
    grouping = root.get("grouping") or {}
    execution = root.get("execution") or {}
    return {
        "example_id": example_id,
        "scenario_id": str(execution.get("scenario_id") or ""),
        "root_scenario_id": str(grouping.get("root_scenario_id") or execution.get("scenario_id") or ""),
        "physical_root_fingerprint": str(grouping.get("physical_root_fingerprint") or ""),
        "network_case": str(grouping.get("case_id") or ""),
        "messages": view.messages(observation) + [view.target_call(step["action"])],
        "tools": view.tools(observation),
        "metadata": {"protocol": MINIMAL_PROTOCOL, "row_contract": ROW_CONTRACT},
        "minimal": dict(labels),
    }


def demonstration_rows(root: Mapping[str, Any], built: Mapping[str, Any], *, tag: str) -> list[dict[str, Any]]:
    """The training rows of one demonstration: its target steps whose call the environment carried out."""
    scenario_id = str((root.get("execution") or {}).get("scenario_id") or "")
    rows: list[dict[str, Any]] = []
    for step in built["steps"]:
        if step["kind"] in (KIND_PREFIX, KIND_ERROR) or step["kind"] not in TARGET_KINDS:
            continue
        if step["result"].get("status") != "ok":
            continue  # a call the tool could not carry out is recorded in the episode, not taught
        labels = {
            "demonstration": built["demonstration"],
            "kind": step["kind"],
            "step": int(step["step"]),
            "tool": str(step["action"]["tool"]),
            "class": built.get("class"),
            "sequence_solved": bool(built["solved"]),
            "declared": built["score"].get("declared"),
            "estimator_passes_at_end": bool(built["score"].get("estimator_passes")),
        }
        if built["demonstration"] == "recovery":
            labels["after"] = built.get("after")
            labels["injected_error"] = built.get("injected_error")
        rows.append(step_row(root, step, example_id=f"{scenario_id}:{tag}:{int(step['step']):02d}", labels=labels))
    return rows


def validate_row(row: Mapping[str, Any]) -> None:
    """The schema half of the trainer's row gate: the tools are valid schemas and the target call fits its own."""
    from psse_env.sft.gates import validate_messages, validate_tool_schemas

    label = str(row.get("example_id") or "row")
    tools = validate_tool_schemas(row.get("tools"), row_label=label)
    validate_messages(row.get("messages"), tools=tools, row_label=label)


__all__ = ["MINIMAL_PROTOCOL", "ROW_CONTRACT", "demonstration_rows", "step_row", "validate_row"]
