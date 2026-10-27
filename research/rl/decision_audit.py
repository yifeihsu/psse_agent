"""Decision-opportunity audit of a hard-draw suite (docs/decision_audit_plan_20261019.md).

    python -m research.rl.decision_audit --suite output/rl_hard_draw_20261015/hard_training_screen.json \\
        --output-dir output/decision_audit_20261019/pilot --strata parameter:misranked ... --limit 32

For every root the observable expert is rolled out through the evaluator
(the reference trajectory).  At each decision state of that trajectory,
where the expert's own observable menu holds at least two distinct
admissible actions and the choice is a rule-expert choice rather than a
candidate disposition it is obliged to take, the audit replays the
reference actions on a fresh episode, takes one alternative from the menu
and lets the expert finish; the evaluator scores the branch as it scores
every run.  At the first decision state offering a correction it also
takes the correction built from the hidden truth (every true target at
once), a probe of the interface and the evidence that is not a legal
action of the policy.  The states are binned as the plan defines:
``rule_best``, ``legal_alternative``, ``interface_limit``,
``evidence_limit``; the features of every alternative are stored for the
predictability test.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import (  # noqa: E402
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    RUN_WLS,
)
from psse_env.oracle.expert_variants import EXPERT_VARIANTS  # noqa: E402

AUDIT_CONTRACT = "rl_decision_audit_v1"
BINS = ("rule_best", "legal_alternative", "interface_limit", "evidence_limit", "no_decision")
CORRECTION_TOOLS = {CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY}
#: Tools whose choice is forced by the transaction state rather than a diagnostic decision.
LIFECYCLE_TOOLS = {"commit_state", "rollback_state", "finalize_diagnosis"}


# --------------------------------------------------------------------------- pure parts


def action_signature(action: Mapping[str, Any]) -> str:
    """A stable identity of an action: its tool and its target arguments, state ids removed."""

    arguments = dict(action.get("arguments") or {})
    for key in ("state_id", "candidate_state_id"):
        arguments.pop(key, None)
    return json.dumps({"tool": action.get("tool"), "arguments": arguments}, sort_keys=True, default=str)


def distinct_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for action in actions:
        signature = action_signature(action)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(copy.deepcopy(dict(action)))
    return out


def is_decision_state(menu: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any], *, selection_basis: str) -> bool:
    """Whether the policy had a real choice at this state."""

    if selection_basis != "rule_expert":
        return False
    if str(chosen.get("tool")) in LIFECYCLE_TOOLS or str(chosen.get("tool")) == RUN_WLS:
        return False
    return len(distinct_actions(menu)) >= 2


def outcome_of(episode: Mapping[str, Any]) -> dict[str, Any]:
    """The audit's view of an evaluated episode."""

    return {
        "success": bool(episode.get("truth_audited_task_success")),
        "basis": _basis(episode),
        "terminal_outcome": episode.get("terminal_outcome"),
        "false_commits": int(episode.get("false_commit_count") or 0),
        "false_finalizations": int(episode.get("false_finalization_count") or 0),
        "invalid_actions": int(episode.get("invalid_action_count") or 0),
        "loop": bool(episode.get("loop_detected")),
        "steps": int(episode.get("steps") or 0),
        "policy_steps": int(episode.get("policy_steps") or 0),
        "healthy_preserved": bool(episode.get("healthy_components_preserved", True)),
        "tools": [str((step.get("action") or {}).get("tool")) for step in episode.get("trace") or [] if not step.get("intervention")],
    }


def better(candidate: Mapping[str, Any], reference: Mapping[str, Any], *, margin: int = 2) -> bool:
    """Whether ``candidate`` beats ``reference``: success over failure, else the same success with ``margin`` fewer steps and no new harm."""

    if candidate["success"] and not reference["success"]:
        return True
    if candidate["success"] and reference["success"]:
        return (
            candidate["steps"] + margin <= reference["steps"]
            and candidate["false_commits"] <= reference["false_commits"]
            and candidate["healthy_preserved"] >= reference["healthy_preserved"]
        )
    return False


def classify_state(reference: Mapping[str, Any], alternatives: Sequence[Mapping[str, Any]],
                   truth: Mapping[str, Any] | None) -> str:
    """The plan's four bins for one decision state."""

    legal = [alt for alt in alternatives if alt.get("outcome")]
    if any(better(alt["outcome"], reference) for alt in legal):
        return "legal_alternative"
    if reference["success"]:
        return "rule_best"
    if truth is not None and truth.get("outcome") and truth["outcome"]["success"]:
        return "interface_limit"
    return "evidence_limit"


def root_bin(state_bins: Sequence[str], reference_success: bool) -> str:
    """A root's bin is the most actionable bin of its audited states."""

    for name in ("legal_alternative", "interface_limit", "evidence_limit", "rule_best"):
        if name in state_bins:
            return name
    return "rule_best" if reference_success else "no_decision"


def alternative_features(observation: Mapping[str, Any], action: Mapping[str, Any], *, rank: int, step: int) -> dict[str, Any]:
    """What the policy could see about an alternative at the decision state."""

    evidence = observation.get("fresh_context_evidence") or {}
    parameter = evidence.get("parameter") if isinstance(evidence, Mapping) else None
    measurement = evidence.get("measurement") if isinstance(evidence, Mapping) else None
    wls = evidence.get("wls") if isinstance(evidence, Mapping) else None
    arguments = dict(action.get("arguments") or {})
    features: dict[str, Any] = {
        "rank": int(rank),
        "step": int(step),
        "tool": str(action.get("tool")),
        "line_index": arguments.get("line_index", arguments.get("line_index1")),
        "suspect_group": arguments.get("suspect_group"),
        "request": arguments.get("request"),
        "remaining_budget": observation.get("remaining_budget"),
        "accepted_corrections": len(observation.get("accepted_corrections") or []),
        "rejected_hypotheses": len(observation.get("rejected_hypotheses") or []),
        "tried_actions": len(observation.get("tried_action_signatures") or []),
        "remaining_anomaly_score": observation.get("remaining_anomaly_score"),
        "has_open_candidate": bool(observation.get("has_open_candidate")),
        "available_evidence": list(observation.get("available_evidence") or []),
    }
    if isinstance(parameter, Mapping):
        features["parameter_dominance_ratio"] = parameter.get("parameter_ranking_dominance_ratio")
        features["parameter_dominance_threshold"] = parameter.get("parameter_ranking_dominance_threshold")
        features["parameter_dominant"] = parameter.get("parameter_ranking_dominant")
        features["parameter_top_abs_lambda"] = parameter.get("parameter_ranking_top_abs_lambda")
        features["parameter_runner_up_abs_lambda"] = parameter.get("parameter_ranking_runner_up_abs_lambda")
        features["parameter_route_status_reason"] = parameter.get("route_status_reason")
        ranked = parameter.get("parameter_ranking_distinct_lines") or []
        features["parameter_candidates"] = [
            {k: item.get(k) for k in ("line_index1", "abs_lambda_score", "abs_lambda", "lambda", "rank") if isinstance(item, Mapping) and k in item}
            for item in ranked[:6] if isinstance(item, Mapping)
        ]
        line = features.get("line_index")
        if line is not None:
            scores = {int(item["line_index1"]): float(item.get("abs_lambda_score") or item.get("abs_lambda") or 0.0)
                      for item in ranked if isinstance(item, Mapping) and item.get("line_index1") is not None}
            own = scores.get(int(line))
            top = max(scores.values()) if scores else None
            features["candidate_abs_lambda"] = own
            features["candidate_lambda_ratio_to_top"] = (own / top) if own is not None and top else None
            features["candidate_context_rank"] = next((i for i, item in enumerate(ranked) if isinstance(item, Mapping) and item.get("line_index1") == line), None)
    if isinstance(measurement, Mapping):
        findings = [dict(item) for item in (measurement.get("measurement_findings") or []) if isinstance(item, Mapping)]
        features["measurement_findings"] = findings[:10]
        features["measurement_finding_count"] = len(findings)
        group = arguments.get("suspect_group")
        if isinstance(group, list) and group:
            by_index = {}
            for position, item in enumerate(findings):
                for key in ("index0", "index", "measurement_index"):
                    if item.get(key) is not None:
                        by_index[int(item[key])] = (position, item)
                        break
            own = [by_index.get(int(i)) for i in group if by_index.get(int(i)) is not None]
            values = [float(item.get("value") or 0.0) for _p, item in own]
            top = max((float(item.get("value") or 0.0) for item in findings), default=None)
            features["meter_context_rank"] = min((p for p, _i in own), default=None)
            features["meter_value"] = max(values) if values else None
            features["meter_value_ratio_to_top"] = (max(values) / top) if values and top else None
            features["meter_channels"] = [str(item.get("channel")) for _p, item in own]
            features["meter_offsets"] = [item.get("channel_offset") for _p, item in own]
            features["meter_group_size"] = len(group)
            # A flow pair shares one branch: another finding on the same offset with the partner channel.
            partners = 0
            for _p, item in own:
                offset, channel = item.get("channel_offset"), str(item.get("channel"))
                partner = {"Pf": "Pt", "Pt": "Pf", "Qf": "Qt", "Qt": "Qf"}.get(channel)
                if partner and any(other.get("channel_offset") == offset and str(other.get("channel")) == partner for other in findings):
                    partners += 1
            features["meter_same_branch_partner"] = partners
    if isinstance(wls, Mapping):
        features["wls"] = {k: wls.get(k) for k in ("chi_square_statistic", "chi_square_threshold", "chi_square_alarm", "max_normalized_residual",
                                                   "normalized_residual_alarm", "anomaly_breadth", "dominant_residual_block", "dof") if k in wls}
    return features


def truth_correction_actions(root: Mapping[str, Any], active_state_id: str) -> list[dict[str, Any]]:
    """The corrections the hidden truth names, every target of a family at once (the audit's probe, never a policy action)."""

    from psse_env.dagger.evaluator import _truth_branch_targets, _truth_measurement_targets

    audit = root.get("audit") if isinstance(root.get("audit"), Mapping) else {}
    truth = audit.get("truth") if isinstance(audit.get("truth"), Mapping) else {}
    actions: list[dict[str, Any]] = []
    meters = sorted(_truth_measurement_targets(truth))
    if meters:
        actions.append({"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": active_state_id, "suspect_group": [int(i) for i in meters]}})
    for line in sorted(_truth_branch_targets(truth, "parameter")):
        actions.append({"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active_state_id, "line_index": int(line) + 1}})
    # A breaker-status truth (the node/breaker roots: every topology root
    # since 2026-09-12, every compound root) is probed through the breaker
    # route the deployed executor represents it by, bus splits included; a
    # line-level truth without a breaker keeps the branch-status probe.
    named = [item for item in truth.get("true_topology_errors") or [] if isinstance(item, Mapping) and item.get("cb_name")]
    if named:
        for item in sorted(named, key=lambda row: str(row["cb_name"])):
            arguments: dict[str, Any] = {"state_id": active_state_id, "cb_name": str(item["cb_name"]),
                                         "status": int(bool(item.get("true_cb_closed")))}
            # The context offers a dangling-terminal flip with its line, and the
            # process gate matches the whole argument set.
            if item.get("line_index1") is not None:
                arguments["line_index"] = int(item["line_index1"])
            actions.append({"tool": CORRECT_TOPOLOGY, "arguments": arguments})
    else:
        for branch in sorted(_truth_branch_targets(truth, "topology")):
            actions.append({"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": active_state_id, "branch_row0": int(branch)}})
    return actions


def _basis(episode: Mapping[str, Any]) -> str:
    audit = episode.get("audit") if isinstance(episode.get("audit"), Mapping) else {}
    assessment = audit.get("truth_audited_task_assessment") if isinstance(audit, Mapping) else None
    if isinstance(assessment, Mapping) and assessment.get("basis"):
        return str(assessment["basis"])
    return "none" if not episode.get("truth_audited_task_success") else "unknown"


def truth_audit_reasons(episode: Mapping[str, Any]) -> list[str]:
    """The truth audit's reasons for an episode that is not a truth-audited success."""
    audit = episode.get("audit") if isinstance(episode.get("audit"), Mapping) else {}
    assessment = audit.get("truth_audited_task_assessment") if isinstance(audit, Mapping) else None
    if not isinstance(assessment, Mapping):
        return []
    return [str(reason) for reason in assessment.get("reasons") or []]


_VERIFY_METRIC_KEYS = ("global_progress", "target_progress", "globally_resolved", "remaining_suspect_count",
                       "target_metric_value", "target_metric_threshold", "physical_constraints_ok", "chi_square_ratio")


def correction_outcomes(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each correction of a policy trace with its verification metrics and the verdict its lifecycle step carried.

    The evaluator records the offline disposition of the open candidate on the
    step that disposes of it (commit or rollback), so a correction's verdict is
    read from the first lifecycle step after it; the verification metrics come
    from the candidate's own WLS solve in between.
    """
    out: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        action = step.get("action") or {}
        tool = str(action.get("tool"))
        if tool not in CORRECTION_TOOLS:
            continue
        arguments = {k: v for k, v in (action.get("arguments") or {}).items() if k != "state_id"}
        record: dict[str, Any] = {
            "step": index, "tool": tool, "arguments": arguments,
            "execution_status": step.get("execution_status"), "error_code": step.get("error_code"),
            "verification": None, "verdict": None, "lifecycle": None,
        }
        if record["execution_status"] == "success":
            for later in steps[index + 1:]:
                later_action = later.get("action") or {}
                later_tool = str(later_action.get("tool"))
                if later_tool in CORRECTION_TOOLS:
                    break
                output = later.get("policy_tool_output") if isinstance(later.get("policy_tool_output"), Mapping) else {}
                metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
                if later_tool == RUN_WLS and record["verification"] is None and any(key in metrics for key in _VERIFY_METRIC_KEYS):
                    record["verification"] = {key: metrics.get(key) for key in _VERIFY_METRIC_KEYS if key in metrics}
                if later_tool in LIFECYCLE_TOOLS and later_tool != "finalize_diagnosis":
                    record["verdict"] = later.get("candidate_disposition_offline")
                    record["lifecycle"] = later_tool
                    break
        out.append(record)
    return out


# --------------------------------------------------------------------------- policies


class ScriptedThenExpertPolicy:
    """Replay ``prefix``, take ``alternatives`` in order, then follow ``expert``.

    ``rebind`` replaces the ``state_id`` of a scripted action with the live
    active (or candidate) state id, so a prefix recorded on one episode
    replays on a fresh one whose ids may differ.
    """

    def __init__(self, prefix: Sequence[Mapping[str, Any]], alternatives: Sequence[Mapping[str, Any]], expert: Any) -> None:
        self.script = [copy.deepcopy(dict(a)) for a in prefix] + [copy.deepcopy(dict(a)) for a in alternatives]
        self.expert = expert
        self.calls = 0
        self.observations_at_script_end: list[dict[str, Any]] = []

    @property
    def release_policy_identity(self) -> Any:
        return getattr(self.expert, "release_policy_identity", None)

    @staticmethod
    def rebind(action: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
        bound = copy.deepcopy(dict(action))
        arguments = dict(bound.get("arguments") or {})
        if "state_id" in arguments:
            arguments["state_id"] = observation.get("candidate_state_id") if (
                str(bound.get("tool")) == RUN_WLS and observation.get("has_unverified_candidate") and observation.get("candidate_state_id")
                and arguments.get("state_id") not in (observation.get("active_state_id"),)
            ) else observation.get("active_state_id")
            if arguments.get("state_id") is None:
                arguments.pop("state_id")
        if "candidate_state_id" in arguments and observation.get("candidate_state_id"):
            arguments["candidate_state_id"] = observation.get("candidate_state_id")
        bound["arguments"] = arguments
        return bound

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        index = self.calls
        self.calls += 1
        if index < len(self.script):
            return self.rebind(self.script[index], observation)
        return self.expert.act(observation)


class TruthGuidedPolicy(ScriptedThenExpertPolicy):
    """Replay ``prefix``, then apply every truth correction in turn, letting the expert verify and commit between them.

    A correction is only legal on a state without an open candidate, so after
    each scripted correction the expert runs the lifecycle (verification,
    commit or rollback) and the next truth correction follows once the state
    is clear again.  The audit's probe of the interface and the evidence; not a
    policy action.
    """

    def __init__(self, prefix: Sequence[Mapping[str, Any]], truth_actions: Sequence[Mapping[str, Any]], expert: Any) -> None:
        super().__init__(prefix, [], expert)
        self.pending = [copy.deepcopy(dict(a)) for a in truth_actions]
        self.applied: list[dict[str, Any]] = []

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        index = self.calls
        if index < len(self.script):
            return super().act(observation)
        self.calls += 1
        if self.pending and not observation.get("has_open_candidate") and not observation.get("candidate_state_id"):
            action = self.rebind(self.pending.pop(0), observation)
            self.applied.append(action)
            return action
        return self.expert.act(observation)


# --------------------------------------------------------------------------- the audit of one root


class _Harness:
    """Environment factory, expert and evaluator of one worker process."""

    def __init__(self, *, profile: str, expert_variant: str, triage_model: str | None, progress_basis: str | None,
                 max_steps: int, seed: int, reward: Mapping[str, Any] | None) -> None:
        import scripts.run_dagger_research as research
        from psse_env.dagger.evaluator import evaluate_rollout_suites
        from psse_env.dagger.release_factories import deterministic_case_loader
        from psse_env.evidence_profile import validate_evidence_profile
        from psse_env.rl.reward import RewardWeights

        self.research = research
        self.profile = validate_evidence_profile(profile)
        research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = self.profile
        research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
        research.RESEARCH_ENVIRONMENT_OPTIONS["triage_classifier"] = triage_model
        if progress_basis:
            research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] = progress_basis
        research.RESEARCH_EXPERT_OPTIONS["variant"] = expert_variant
        self.env_factory = research.resolve_environment_factory("research", self.profile)
        self.evaluate = evaluate_rollout_suites
        self.case_loader = deterministic_case_loader
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.weights = RewardWeights.from_mapping(reward)
        self.contract = dict(research.PAIRED_EVALUATION_CONTRACT)

    def expert_policy(self) -> Any:
        return self.research.research_expert_policy(self.env_factory)

    def expert_oracle(self) -> Any:
        return self.expert_policy()._expert

    def run(self, root: Mapping[str, Any], policy: Any) -> dict[str, Any]:
        result = self.evaluate(
            {"standard_success": [copy.deepcopy(dict(root))]},
            env_factory=self.env_factory,
            policy_factory=lambda **_kwargs: policy,
            max_steps=self.max_steps,
            seed=self.seed,
            case_loader=self.case_loader,
            **self.contract,
        )
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        episodes = list(payload["suite_metrics"]["episodes"])
        if len(episodes) != 1:
            raise RuntimeError(f"expected one episode, got {len(episodes)}")
        return dict(episodes[0])

    def reward(self, episode: Mapping[str, Any], family: str) -> float:
        from psse_env.rl.reward import episode_reward

        return float(episode_reward(episode, family=family, expert_reference_calls=None, weights=self.weights).total)


_HARNESS: _Harness | None = None


def _init_worker(settings: Mapping[str, Any]) -> None:
    global _HARNESS
    _HARNESS = _Harness(**settings)


def policy_steps(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [step for step in trace if isinstance(step, Mapping) and not step.get("intervention")]


def audit_root(root: Mapping[str, Any], *, harness: _Harness, max_states: int, max_alternatives: int,
               with_truth: bool = True) -> dict[str, Any]:
    """The audit of one root: the reference trajectory, its decision states and their branches."""

    from psse_env.dagger.release_factories import select_observable_expert_actions
    from psse_env.rl.rollout import scenario_family, scenario_root
    from research.rl.hard_draw_report import root_stratum

    started = time.perf_counter()
    root_id = scenario_root(root)
    family = scenario_family(root)
    stratum = root_stratum(root)
    reference_episode = harness.run(root, harness.expert_policy())
    reference = outcome_of(reference_episode)
    reference["reward"] = harness.reward(reference_episode, family)
    steps = policy_steps(reference_episode.get("trace") or [])
    actions = [copy.deepcopy(dict(step.get("action") or {})) for step in steps]
    oracle = harness.expert_oracle()
    states: list[dict[str, Any]] = []
    truth_done = False
    episodes_run = 1
    for index, step in enumerate(steps):
        if len(states) >= max_states:
            break
        observation = step.get("policy_observation")
        if not isinstance(observation, Mapping):
            continue
        try:
            selection = select_observable_expert_actions(policy_observation=observation, expert_oracle=oracle)
        except Exception as exc:  # a malformed stored observation
            states.append({"step": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        menu = distinct_actions(selection.actions)
        chosen = actions[index]
        if not is_decision_state(menu, chosen, selection_basis=selection.selection_basis):
            continue
        prefix = actions[:index]
        chosen_signature = action_signature(chosen)
        alternatives: list[dict[str, Any]] = []
        for rank, action in enumerate(menu):
            if action_signature(action) == chosen_signature:
                continue
            if len(alternatives) >= max_alternatives:
                break
            policy = ScriptedThenExpertPolicy(prefix, [action], harness.expert_policy())
            episode = harness.run(root, policy)
            episodes_run += 1
            branch_steps = policy_steps(episode.get("trace") or [])
            same_state = bool(index < len(branch_steps) and branch_steps[index].get("observation_hash") == step.get("observation_hash"))
            taken = branch_steps[index] if index < len(branch_steps) else {}
            outcome = outcome_of(episode)
            outcome["reward"] = harness.reward(episode, family)
            alternatives.append({
                "rank": rank,
                "action": action,
                "features": alternative_features(observation, action, rank=rank, step=index),
                "same_state": same_state,
                "execution_status": taken.get("execution_status"),
                "error_code": taken.get("error_code"),
                "outcome": outcome,
                "better": better(outcome, reference),
            })
        truth_branch = None
        if with_truth and not truth_done and any(str(a.get("tool")) in CORRECTION_TOOLS for a in menu):
            truth_actions = truth_correction_actions(root, str(observation.get("active_state_id")))
            if truth_actions:
                truth_done = True
                policy = TruthGuidedPolicy(prefix, truth_actions, harness.expert_policy())
                episode = harness.run(root, policy)
                episodes_run += 1
                branch_steps = policy_steps(episode.get("trace") or [])
                taken = branch_steps[index] if index < len(branch_steps) else {}
                outcome = outcome_of(episode)
                outcome["reward"] = harness.reward(episode, family)
                truth_branch = {"action": truth_actions[0], "all_truth_actions": truth_actions, "applied": policy.applied,
                                "unapplied": policy.pending,
                                "execution_status": taken.get("execution_status"), "error_code": taken.get("error_code"),
                                "outcome": outcome, "better": better(outcome, reference),
                                # Why the probe fell short, when it did: the truth audit's
                                # reasons and how each applied correction fared.
                                "audit_reasons": truth_audit_reasons(episode),
                                "corrections": correction_outcomes(branch_steps)}
        state_bin = classify_state(reference, alternatives, truth_branch)
        states.append({
            "step": index,
            "chosen": chosen,
            "chosen_rank": next((i for i, a in enumerate(menu) if action_signature(a) == chosen_signature), None),
            "menu_size": len(menu),
            "menu": menu,
            "observation_hash": step.get("observation_hash"),
            "alternatives": alternatives,
            "truth": truth_branch,
            "bin": state_bin,
        })
    bins = [s["bin"] for s in states if "bin" in s]
    return {
        "contract": AUDIT_CONTRACT,
        "root": root_id,
        "family": family,
        "stratum": stratum,
        "reference": reference,
        "reference_actions": actions,
        "decision_states": states,
        "bin": root_bin(bins, reference["success"]) if bins else ("rule_best" if reference["success"] else "no_decision"),
        "episodes_run": episodes_run,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _audit_root_in_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    assert _HARNESS is not None
    try:
        return audit_root(payload["root"], harness=_HARNESS, max_states=int(payload["max_states"]),
                          max_alternatives=int(payload["max_alternatives"]), with_truth=bool(payload["with_truth"]))
    except Exception as exc:  # keep the batch alive; the error is recorded
        from psse_env.rl.rollout import scenario_root

        return {"contract": AUDIT_CONTRACT, "root": scenario_root(payload["root"]), "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- selection and summary


def select_roots(suite: Sequence[Mapping[str, Any]], *, strata: Sequence[str] | None, per_stratum: Mapping[str, int] | None,
                 tags: Mapping[str, str] | None, prefer_tag: str | None, limit: int | None, roots: Sequence[str] | None) -> list[dict[str, Any]]:
    from psse_env.rl.rollout import scenario_root
    from research.rl.hard_draw_report import root_stratum

    chosen: list[dict[str, Any]] = []
    wanted_roots = set(roots or [])
    counts: Counter = Counter()
    ordered = list(suite)
    if prefer_tag and tags:
        ordered.sort(key=lambda r: 0 if tags.get(scenario_root(r)) == prefer_tag else 1)
    for root in ordered:
        root_id = scenario_root(root)
        stratum = root_stratum(root)
        if wanted_roots and root_id not in wanted_roots:
            continue
        if strata and stratum not in strata:
            continue
        if per_stratum is not None:
            cap = per_stratum.get(stratum)
            if cap is not None and counts[stratum] >= cap:
                continue
        counts[stratum] += 1
        chosen.append(root)
        if limit and len(chosen) >= limit:
            break
    return chosen


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = defaultdict(lambda: {"roots": 0, "reference_success": 0, "bins": Counter(), "state_bins": Counter(),
                                                                 "decision_states": 0, "alternatives": 0, "alternatives_better": 0,
                                                                 "truth_tried": 0, "truth_success": 0, "errors": 0, "episodes": 0, "seconds": 0.0})
    for result in results:
        entry = by_stratum[str(result.get("stratum") or "?")]
        entry["roots"] += 1
        if result.get("error"):
            entry["errors"] += 1
            continue
        entry["reference_success"] += int(bool(result["reference"]["success"]))
        entry["bins"][result["bin"]] += 1
        entry["episodes"] += int(result.get("episodes_run") or 0)
        entry["seconds"] += float(result.get("seconds") or 0.0)
        for state in result.get("decision_states") or []:
            if "bin" not in state:
                continue
            entry["decision_states"] += 1
            entry["state_bins"][state["bin"]] += 1
            entry["alternatives"] += len(state.get("alternatives") or [])
            entry["alternatives_better"] += sum(1 for a in state.get("alternatives") or [] if a.get("better"))
            if state.get("truth"):
                entry["truth_tried"] += 1
                entry["truth_success"] += int(bool(state["truth"]["outcome"]["success"]))
    table = {s: {**{k: v for k, v in e.items() if not isinstance(v, Counter)}, "bins": dict(e["bins"]), "state_bins": dict(e["state_bins"])}
             for s, e in sorted(by_stratum.items())}
    totals = Counter()
    for e in by_stratum.values():
        totals.update(e["bins"])
    return {"contract": AUDIT_CONTRACT, "roots": len(results), "bins": dict(totals), "strata": table}


def markdown(summary: Mapping[str, Any]) -> str:
    lines = ["| stratum | roots | expert solves | rule_best | legal_alternative | interface_limit | evidence_limit | decision states | alternatives (better) | truth branch solves | errors |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for stratum, e in summary["strata"].items():
        b = e["bins"]
        lines.append(f"| {stratum} | {e['roots']} | {e['reference_success']} | {b.get('rule_best', 0)} | {b.get('legal_alternative', 0)} | "
                     f"{b.get('interface_limit', 0)} | {b.get('evidence_limit', 0)} | {e['decision_states']} | {e['alternatives']} ({e['alternatives_better']}) | "
                     f"{e['truth_success']}/{e['truth_tried']} | {e['errors']} |")
    t = summary["bins"]
    lines.append(f"| all | {summary['roots']} | | {t.get('rule_best', 0)} | {t.get('legal_alternative', 0)} | {t.get('interface_limit', 0)} | {t.get('evidence_limit', 0)} | | | | |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="suspicion_gated_diagnostics")
    parser.add_argument("--expert", default="ledger_ranked", choices=EXPERT_VARIANTS)
    parser.add_argument("--triage-model", default=None)
    parser.add_argument("--progress-basis", default=None)
    parser.add_argument("--strata", nargs="*", default=None, help="root strata to audit (hard_draw_report.root_stratum names)")
    parser.add_argument("--per-stratum", default=None, help="JSON object of stratum -> maximum roots")
    parser.add_argument("--tags", default=None, help="identifiability tags.jsonl; with --prefer-tag, those roots are taken first")
    parser.add_argument("--prefer-tag", default=None)
    parser.add_argument("--roots", nargs="*", default=None, help="explicit scenario ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-states", type=int, default=4, help="decision states audited per root")
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--no-truth", action="store_true", help="skip the truth-correction probe")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--reward-json", default=None, help="RewardWeights mapping (default: the contract's defaults)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    if isinstance(suite, Mapping):
        suite = suite.get("scenarios") or suite.get("roots") or []
    tags = None
    if args.tags:
        from psse_env.rl.rollout import scenario_root  # noqa: F401

        tags = {str(row["scenario_id"]): str(row["tag"]) for row in
                (json.loads(line) for line in Path(args.tags).read_text(encoding="utf-8").splitlines() if line.strip())}
    per_stratum = json.loads(args.per_stratum) if args.per_stratum else None
    roots = select_roots(suite, strata=args.strata, per_stratum=per_stratum, tags=tags, prefer_tag=args.prefer_tag,
                         limit=args.limit, roots=args.roots)
    reward = json.loads(Path(args.reward_json).read_text(encoding="utf-8")) if args.reward_json else None
    settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=args.triage_model,
                    progress_basis=args.progress_basis, max_steps=args.max_steps, seed=args.seed, reward=reward)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[audit] {len(roots)} roots from {args.suite}; profile {args.profile}, expert {args.expert}, "
          f"{args.max_states} states x {args.max_alternatives} alternatives, workers {args.workers}", flush=True)
    started = time.perf_counter()
    payloads = [{"root": root, "max_states": args.max_states, "max_alternatives": args.max_alternatives, "with_truth": not args.no_truth}
                for root in roots]
    results: list[dict[str, Any]] = []
    stream = (out / "roots.jsonl").open("w", encoding="utf-8")
    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(settings,)) as pool:
                for result in pool.map(_audit_root_in_worker, payloads, chunksize=1):
                    results.append(result)
                    stream.write(json.dumps(result, sort_keys=True, default=str) + "\n"); stream.flush()
                    print(json.dumps({"root": result.get("root"), "stratum": result.get("stratum"), "bin": result.get("bin"),
                                      "reference_success": (result.get("reference") or {}).get("success"),
                                      "states": len(result.get("decision_states") or []), "episodes": result.get("episodes_run"),
                                      "seconds": result.get("seconds"), "error": result.get("error"), "done": len(results),
                                      "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
        else:
            _init_worker(settings)
            for payload in payloads:
                result = _audit_root_in_worker(payload)
                results.append(result)
                stream.write(json.dumps(result, sort_keys=True, default=str) + "\n"); stream.flush()
                print(json.dumps({"root": result.get("root"), "stratum": result.get("stratum"), "bin": result.get("bin"),
                                  "reference_success": (result.get("reference") or {}).get("success"),
                                  "states": len(result.get("decision_states") or []), "episodes": result.get("episodes_run"),
                                  "seconds": result.get("seconds"), "error": result.get("error"), "done": len(results),
                                  "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        stream.close()
    summary = summarize(results)
    summary.update(suite=str(args.suite), profile=args.profile, expert=args.expert, max_states=args.max_states,
                   max_alternatives=args.max_alternatives, max_steps=args.max_steps, seed=args.seed,
                   total_seconds=round(time.perf_counter() - started, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary) + "\n", encoding="utf-8")
    with (out / "states.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for state in result.get("decision_states") or []:
                row = {"root": result["root"], "family": result.get("family"), "stratum": result.get("stratum"),
                       "reference": result.get("reference"), **state}
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(markdown(summary))
    print(json.dumps({"bins": summary["bins"], "total_seconds": summary["total_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
