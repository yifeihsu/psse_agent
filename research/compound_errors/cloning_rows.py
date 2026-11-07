"""Cloning pass: training rows from a full evaluated episode (RL plan of 2026-11-05, start policy B).

The evaluator records, for every policy step, the observation the policy
acted on (``trace[i]["policy_observation"]``) and the action it took.  A
cloning row is that pair, pushed through the same compaction and protocol
bridge as every other SFT row (``examples_to_chat_sft``), so the prompt is the
one the student gets at run time.  Nothing here depends on who chose the
action: the plain teacher, or the teacher with the lifecycle step decided by
the truth audit (``commit_reader_loop.py --decider audit``).

``episode_examples`` is the interface (drafted in the plan session of
2026-11-06): one evaluated episode and its root in, the examples for
``examples_to_chat_sft`` out.  ``episode_chat_rows`` applies the exporter and
labels each row for the later selection by root.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

CLONING_CONTRACT = "compound_cloning_rows_v1"
#: Tools outside the student's canonical tool list; an episode that used one is left out whole.
UNEXPORTABLE_TOOLS = frozenset({"compare_routes", "revert_state"})
#: Observation fields of the behaviour policy's environment that the student's environment does not have.  The
#: route teacher needs ``compare_routes`` offered (without the hint it is the guarded teacher: its breaker-context-first
#: ordering sits behind the same gate, ``ExpertPolicyOracle._route_comparison_first``), the student cannot call that
#: tool and runs without it, so its rows must not name it.
BEHAVIOUR_ONLY_FIELDS = frozenset({"research_tools"})
#: Arguments the student's tool schema does not have.  The handoff statements of the compound study
#: (isolated line, ambiguous hypotheses) ride the closing request and are never required, so the
#: target keeps the request and drops them.
DROPPED_ARGUMENTS = {"ask_for_more_evidence": frozenset({"abstentions"})}


def episode_examples(episode: Mapping[str, Any], root: Mapping[str, Any], *, behaviour: str) -> list[dict[str, Any]]:
    """One example per policy step: the observation the policy acted on, and the action it took."""
    scenario_id = str(root["execution"]["scenario_id"])
    grouping = root.get("grouping") or {}
    compound = (root.get("audit") or {}).get("compound") or {}
    steps = [step for step in episode.get("trace") or [] if isinstance(step, Mapping) and not step.get("intervention")]
    if any(str((step.get("action") or {}).get("tool")) in UNEXPORTABLE_TOOLS for step in steps):
        return []
    examples = []
    for index, step in enumerate(steps):
        action = step.get("action")
        observation = step.get("policy_observation")
        if not isinstance(action, Mapping) or not isinstance(observation, Mapping):
            continue
        examples.append({
            "example_id": f"{scenario_id}:clone:{index:02d}",
            "scenario_id": scenario_id,
            "root_scenario_id": str(grouping.get("root_scenario_id") or scenario_id),
            "physical_root_fingerprint": grouping.get("physical_root_fingerprint"),
            "dataset_mode": "compound_cloning",
            "dataset_source": behaviour,
            "collector_contract": CLONING_CONTRACT,
            "state_origin": "behaviour_policy",
            "state_visited_by": behaviour,
            "labels": {"scenario_family": grouping.get("scenario_family"), "compound_class": compound.get("class"),
                       "step": index, "tool": str(action.get("tool")),
                       "execution_status": step.get("execution_status")},
            "policy_observation": {key: value for key, value in observation.items() if key not in BEHAVIOUR_ONLY_FIELDS},
            "preferred_action": {"tool": action.get("tool"),
                                 "arguments": {key: value for key, value in (action.get("arguments") or {}).items()
                                               if key not in DROPPED_ARGUMENTS.get(str(action.get("tool")), frozenset())}},
        })
    return examples


def episode_chat_rows(episode: Mapping[str, Any], root: Mapping[str, Any], *, behaviour: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The episode's chat rows, each labelled with its case, class, step and tool, and the refused examples by reason."""
    from psse_env.dagger.dataset_builder import examples_to_chat_sft

    rows: list[dict[str, Any]] = []
    refused: Counter = Counter()
    for example in episode_examples(episode, root, behaviour=behaviour):
        try:
            exported = examples_to_chat_sft([example], protocol="canonical")
        except Exception as exc:  # every refusal is reported with its tool
            refused[f"{example['labels']['tool']} | {type(exc).__name__}: {str(exc)[:140]}"] += 1
            continue
        for row in exported:
            row = dict(row)
            row["cloning"] = {"contract": CLONING_CONTRACT, "behaviour": behaviour, "scenario_id": example["scenario_id"],
                              "compound_class": example["labels"]["compound_class"], "step": example["labels"]["step"],
                              "tool": example["labels"]["tool"]}
            rows.append(row)
    return rows, dict(refused)


__all__ = ["BEHAVIOUR_ONLY_FIELDS", "CLONING_CONTRACT", "DROPPED_ARGUMENTS", "UNEXPORTABLE_TOOLS", "episode_chat_rows", "episode_examples"]
