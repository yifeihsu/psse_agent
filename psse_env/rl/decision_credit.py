"""Credit for a lifecycle decision from what each choice leads to (workplan of 2026-11-10, stage 4).

The lifecycle term of 2026-11-07 labels a commit or a decline by whether the
candidate's target belongs to the injected faults (:mod:`psse_env.rl.lifecycle`).
That label is a property of the target, not of the outcome: it does not say
that the correction's numbers were good, nor that keeping or declining the
candidate led to a solved case.  Here a decision is credited by the outcome.

For a decision state the policy itself reached in this iteration's rollouts,
the continuation phase of the driver runs both choices again from that state
(:mod:`psse_env.rl.forks`): the prefix replayed, the choice forced, the
current policy sampled to the end, ``continuations`` times per choice.  With
``R`` the episode return of a branch:

* ``Q(c)`` is the mean return of choice ``c``: its forced branches and, when
  ``include_source_return`` is set, the returns of the rollout episodes that
  sampled ``c`` at this state themselves;
* the baseline is ``V = p(commit) Q_b(commit) + p(decline) Q_b(decline)``,
  with ``Q_b`` from the forced branches alone and ``p`` the behaviour
  policy's own probability of the choice at the state.  It is one number per
  state, the same whichever choice an episode happened to sample, which is
  what a baseline has to be for the policy gradient to stay unbiased, and
  its samples are the separate branch samples;
* the fork-local advantage of a rollout episode that sampled choice ``a``
  there is ``A = Q(a) - V``.

Where ``p`` comes from: a rollout records the probability of the tokens it
sampled, so a state's rows give the probability of each choice some row
sampled (:func:`decision_token_probability`).  When both choices were
sampled there, both are on record and ``p`` is the two renormalized to one;
what they leave (``mass_on_neither_choice``: the policy's probability of
calling another tool at that step) is reported.  When every row sampled the
same choice, the other choice's probability is not on record and is taken
as the complement, which assumes that mass is nothing.  On the 1,915
labelled decisions of the first arm and the smoke of 2026-11-11 it was at
most 0.004 at the 64 states where it could be measured.  The baseline never
takes one row's probability for its own choice and the complement for the
other row by row: two rows that sampled different choices would then get
two baselines for one state (the review of 2026-11-11, item 1).

:func:`psse_env.rl.rows.build_rows` puts ``weight * A`` on that episode's own
sampled decision row, in place of the row's episode advantage and of the
supervised membership row for that decision.  Every other row keeps the
objective it had.  For a two-way choice ``A = (1 - p(a)) (Q(a) - Q(other))``
up to sampling noise: a confident, unsampled alternative earns little, which
is the policy gradient's own property and the reason the selection favours
the states where the minority choice was sampled.

What a branch is not: a forced step is an intervention of the harness, and
the steps after it were sampled from states the policy did not reach by
itself, so no branch step is ever a training row.  Values belong to the
adapter, the iteration and the observation mode they were sampled under; the
update refuses others (:func:`check_provenance`).

The selection of states uses the audit's labels (true or wrong target, first
or later candidate) to put more of the budget where the choices can differ.
Those labels shape which states are measured; the policy never sees them,
and the report states how many states of each kind there were and how many
were taken (``selection``), so the enrichment can be weighed.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from psse_env.actions import COMMIT_STATE, ROLLBACK_STATE
from psse_env.rl.forks import CHOICES, DecisionState, decision_states, other_choice
from psse_env.rl.rollout import StepRecord, Trajectory

DECISION_CREDIT_CONTRACT = "rl_decision_credit_v1"
#: The kinds of decision state, in the order the selection fills them: the policy sampled a decline (the minority
#: choice, so the sampled row carries most of the gradient); a wrong target it kept; a true target after an earlier
#: decline, at a later candidate, at an episode's first candidate (the retention cases).
STRATA = ("sampled_decline", "wrong_target", "true_after_decline", "true_later", "true_first")
DEFAULT_QUOTAS = {"sampled_decline": 8, "wrong_target": 12, "true_after_decline": 4, "true_later": 4, "true_first": 4}


@dataclass(frozen=True)
class DecisionCreditSettings:
    """The ``decision_credit`` block of an RL config; empty means off."""

    enabled: bool = False
    #: The weight of the fork-local advantage on a covered decision row.
    weight: float = 1.0
    #: At most this many decision states are branched per iteration.
    states_per_iteration: int = 32
    #: Learner continuations per choice at a state.
    continuations: int = 2
    #: States taken per kind before the rest of the budget is filled from what is left (:data:`STRATA`).
    quotas: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_QUOTAS))
    #: Count the rollout episodes' own returns in ``Q`` of the choice they sampled (never in the baseline).
    include_source_return: bool = True
    #: The behaviour probability of a choice is kept inside ``[floor, 1 - floor]``.
    probability_floor: float = 1e-3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DecisionCreditSettings":
        payload = dict(payload or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown decision_credit fields: {unknown}")
        if "quotas" in payload:
            payload["quotas"] = {str(name): int(value) for name, value in dict(payload["quotas"] or {}).items()}
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.include_source_return, bool):
            raise ValueError("decision_credit.enabled and include_source_return must be true or false")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError("decision_credit.weight must be finite and nonnegative")
        for name in ("states_per_iteration", "continuations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"decision_credit.{name} must be a positive integer")
        unknown = sorted(set(self.quotas) - set(STRATA))
        if unknown:
            raise ValueError(f"decision_credit.quotas name unknown kinds {unknown}; the kinds are {list(STRATA)}")
        if any(int(value) < 0 for value in self.quotas.values()):
            raise ValueError("decision_credit.quotas must be nonnegative")
        if not 0.0 < float(self.probability_floor) < 0.5:
            raise ValueError("decision_credit.probability_floor must lie in (0, 0.5)")

    def as_dict(self) -> dict[str, Any]:
        return {name: (dict(getattr(self, name)) if name == "quotas" else getattr(self, name)) for name in self.__dataclass_fields__}


# --------------------------------------------------------------------------- decision states of a rollout batch


def state_stratum(state: DecisionState) -> str:
    """The kind of a decision state (:data:`STRATA`), from the audit's label of its decision."""

    decision = state.decision
    if state.sampled_tool == ROLLBACK_STATE:
        return "sampled_decline"
    if decision.get("target_is_true") is False:
        return "wrong_target"
    if state.earlier_declines:
        return "true_after_decline"
    return "true_later" if state.ordinal else "true_first"


@dataclass
class StateGroup:
    """A decision state with every rollout episode of its root that decided at the same visible state."""

    state: DecisionState
    stratum: str
    #: One entry per covered sampled decision: ``sample_index``, ``position``, ``tool``, ``reward``, ``success``,
    #: and ``probability`` (the behaviour probability of the sampled choice) once known.
    rows: list[dict[str, Any]]

    @property
    def key(self) -> tuple[str, int, int]:
        return self.state.key


def candidate_states(groups: Iterable[tuple[str, Sequence[Trajectory]]]) -> list[StateGroup]:
    """Every decision state of a rollout batch, the episodes of a root that share one counted once.

    Two episodes of a root share a state when they decoded their decision
    from the same prompt (the recorded prompt digest) at the same step: the
    same visible state, the same remaining budget, the same history.  The state is
    represented by the episode with the lowest sample index; the others are
    its covered rows, whatever they sampled there.
    """

    found: list[StateGroup] = []
    for _root, trajectories in groups:
        by_prompt: dict[tuple[str, str, int], StateGroup] = {}
        for trajectory in sorted(trajectories, key=lambda item: item.sample_index):
            if not trajectory.pairing_ok:
                continue
            for state in decision_states(trajectory):
                row = {"sample_index": state.sample_index, "position": state.position, "tool": state.sampled_tool,
                       "reward": state.source_reward, "success": state.source_success, "probability": None}
                shared = (state.root, state.prompt_digest, state.position) if state.prompt_digest else None
                if shared is not None and shared in by_prompt:
                    by_prompt[shared].rows.append(row)
                    continue
                group = StateGroup(state=state, stratum=state_stratum(state), rows=[row])
                found.append(group)
                if shared is not None:
                    by_prompt[shared] = group
    return found


def select_states(candidates: Sequence[StateGroup], settings: DecisionCreditSettings, *, seed: int) -> tuple[list[StateGroup], dict[str, Any]]:
    """The states an iteration branches, and the record of the choice.

    Each kind takes up to its quota, roots not yet used first; what is left
    of the budget is filled from the remaining states in a seeded order,
    again new roots first.  The order within a kind is seeded, so the same
    batch and seed give the same states.  When the quotas add up to more
    than the budget they are scaled down to it (rounded down), so a small
    budget is not spent on the first kinds alone.
    """

    rng = random.Random(f"{int(seed)}:decision_credit")
    pools: dict[str, list[StateGroup]] = {name: [] for name in STRATA}
    for group in candidates:
        pools[group.stratum].append(group)
    for name in STRATA:
        pools[name].sort(key=lambda group: group.key)
        rng.shuffle(pools[name])
    budget = int(settings.states_per_iteration)
    wanted = {name: int(settings.quotas.get(name, 0)) for name in STRATA}
    if sum(wanted.values()) > budget:
        total = sum(wanted.values())
        wanted = {name: (value * budget) // total for name, value in wanted.items()}
    chosen: list[StateGroup] = []
    used_roots: set[str] = set()

    def take(pool: list[StateGroup], limit: int) -> None:
        for fresh_root in (True, False):
            for group in list(pool):
                if len(chosen) >= budget or limit <= 0:
                    return
                if fresh_root and group.state.root in used_roots:
                    continue
                pool.remove(group)
                chosen.append(group)
                used_roots.add(group.state.root)
                limit -= 1

    for name in STRATA:
        take(pools[name], min(wanted[name], budget - len(chosen)))
    leftover = [group for name in STRATA for group in pools[name]]
    rng.shuffle(leftover)
    take(leftover, budget - len(chosen))
    available = {name: sum(1 for group in candidates if group.stratum == name) for name in STRATA}
    selected = {name: sum(1 for group in chosen if group.stratum == name) for name in STRATA}
    report = {
        "contract": DECISION_CREDIT_CONTRACT,
        "seed": int(seed),
        "budget": budget,
        "quotas": {name: int(settings.quotas.get(name, 0)) for name in STRATA},
        "effective_quotas": wanted,
        "available": available,
        "selected": selected,
        # How much more often than its natural share a kind of state is measured (None when the kind is absent).
        "selection_fraction": {name: (selected[name] / available[name]) if available[name] else None for name in STRATA},
        "shortfall": {name: max(0, wanted[name] - selected[name]) for name in STRATA},
        "states_available": len(candidates),
        "states_selected": len(chosen),
        "roots_selected": len(used_roots),
        "sampled_rows_covered": sum(len(group.rows) for group in chosen),
    }
    chosen.sort(key=lambda group: group.key)
    return chosen, report


# --------------------------------------------------------------------------- the behaviour probability of the choice


def decision_token_probability(
    step: StepRecord, sampled_tool: str, render_alternative: Callable[[Mapping[str, Any], str, Mapping[str, Any]], Sequence[int]] | None,
) -> tuple[float | None, str]:
    """The probability the behaviour policy gave the choice it sampled at a decision step, and how it was read.

    The two choices are the same tool call but for the tool's name, so the
    completion and the other choice's rendering part at one token: the
    recorded probability of that token is the probability of the choice
    (``decision_token``).  ``None`` when that token cannot be found: the step
    has no sampling record (``no_sampling_record``), or there is no renderer,
    the other choice cannot be rendered, or the two renderings do not part
    (``no_decision_token``).  Nothing stands in for it: the least likely token
    of a completion is some token the policy was unsure of, an argument as
    readily as the tool's name, and not the probability of the choice, so a
    row without its decision token is not credited from the continuations and
    keeps the objective it had.
    """

    probabilities = list(step.token_probabilities or [])
    completion = list(step.completion_token_ids or [])
    if not probabilities or len(probabilities) != len(completion):
        return None, "no_sampling_record"
    if render_alternative is not None:
        try:
            alternative = [int(value) for value in render_alternative(
                step.model_observation, step.text, {"label": other_choice(sampled_tool), "tool": sampled_tool})]
        except ValueError:
            alternative = []
        for index, (own, other) in enumerate(zip(completion, alternative)):
            if own != other:
                return float(probabilities[index]), "decision_token"
    return None, "no_decision_token"


# --------------------------------------------------------------------------- values


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


#: How far above one the two choices' recorded probabilities may add up before they are taken as not being of one
#: distribution (the rows of a state decode from one prompt, so beyond numerical noise they cannot).
PROBABILITY_SUM_TOLERANCE = 0.01


def behaviour_policy(rows: Sequence[Mapping[str, Any]], *, floor: float) -> dict[str, Any] | None:
    """The behaviour policy's probability of each choice at a state, from the probabilities its rows recorded.

    Every row of a state decoded its decision from the same prompt, so the
    recorded probability of a choice is one number whichever row sampled it
    (their mean is taken; they differ by numerical noise at most).  With both
    choices on record the two are renormalized to one and what they leave is
    ``mass_on_neither_choice``.  With one, the other is its complement, which
    takes that mass as nothing.  ``None`` when no row recorded a probability.
    Either way the result belongs to the state, not to a row.
    """

    recorded = {choice: [float(row["probability"]) for row in rows if row["tool"] == choice and row.get("probability") is not None]
                for choice in CHOICES}
    known = {choice: sum(values) / len(values) for choice, values in recorded.items() if values}
    if not known:
        return None
    if len(known) == 2:
        total = known[COMMIT_STATE] + known[ROLLBACK_STATE]
        if total > 1.0 + PROBABILITY_SUM_TOLERANCE or total <= 0.0:
            return {"inconsistent": True, "recorded": known}
        commit = known[COMMIT_STATE] / total
        outside: float | None = 1.0 - total
        basis = "both_choices_recorded"
    else:
        (choice, value), = known.items()
        commit = value if choice == COMMIT_STATE else 1.0 - value
        outside = None
        basis = "complement_of_the_sampled_choice"
    commit = min(max(commit, float(floor)), 1.0 - float(floor))
    return {"inconsistent": False, "recorded": known, "probability_basis": basis, "mass_on_neither_choice": outside,
            "policy": {COMMIT_STATE: commit, ROLLBACK_STATE: 1.0 - commit}}


def state_values(group: StateGroup, branches: Sequence[Mapping[str, Any]], settings: DecisionCreditSettings) -> dict[str, Any]:
    """The values of the two choices at a state and the fork-local advantage of every covered sampled row.

    The baseline is the state's: one number, the same for every row whichever
    choice it sampled (:func:`behaviour_policy`).
    """

    valid = [branch for branch in branches if branch.get("valid")]
    forced = {choice: [float(branch["reward"]["total"]) for branch in valid if branch.get("forced_tool") == choice] for choice in CHOICES}
    solved = {choice: sum(1 for branch in valid if branch.get("forced_tool") == choice and branch.get("success")) for choice in CHOICES}
    natural = {choice: [float(row["reward"]) for row in group.rows if row["tool"] == choice and row.get("reward") is not None]
               for choice in CHOICES}
    record: dict[str, Any] = {
        "root": group.state.root, "sample_index": int(group.state.sample_index), "position": int(group.state.position),
        "stratum": group.stratum,
        # The audit's label of the decision, for the reports; the policy never sees it.
        "decision": {key: group.state.decision.get(key) for key in ("tool", "label", "agrees", "target_is_true", "verdict",
                                                                     "correction_tool", "arguments")},
        "ordinal": int(group.state.ordinal), "earlier_declines": int(group.state.earlier_declines),
        "branches": {"valid": len(valid), "invalid": len(branches) - len(valid),
                     "mismatched_prefix": sum(1 for branch in branches if not branch.get("prefix_matched", True)),
                     "forced_refused": sum(1 for branch in branches if branch.get("prefix_matched", True) and not branch.get("forced_executed"))},
        "returns": {"forced": forced, "sampled": natural},
        "solved": {"forced": solved,
                   "sampled": {choice: sum(1 for row in group.rows if row["tool"] == choice and row.get("success")) for choice in CHOICES}},
        "rows": [],
    }
    if any(not forced[choice] for choice in CHOICES):
        record.update(covered=False, reason="a choice has no valid continuation")
        return record
    q_branch = {choice: _mean(forced[choice]) for choice in CHOICES}
    q = {choice: _mean(forced[choice] + (natural[choice] if settings.include_source_return else [])) for choice in CHOICES}
    floor = float(settings.probability_floor)
    behaviour = behaviour_policy(group.rows, floor=floor)
    if behaviour is None:
        # No row's decision token was read: nothing says how likely either choice was, so nothing is credited.
        record.update(covered=True, q=q, q_branch=q_branch, policy=None, baseline=None, probability_basis=None,
                      mass_on_neither_choice=None, q_gap_commit_minus_decline=q_branch[COMMIT_STATE] - q_branch[ROLLBACK_STATE],
                      informative=abs(q_branch[COMMIT_STATE] - q_branch[ROLLBACK_STATE]) > 1e-9)
        return record
    if behaviour["inconsistent"]:
        record.update(covered=False, recorded_probabilities=behaviour["recorded"],
                      reason="the recorded probabilities of the two choices add up to more than one")
        return record
    policy = behaviour["policy"]
    baseline = sum(policy[choice] * q_branch[choice] for choice in CHOICES)
    for row in group.rows:
        probability = row.get("probability")
        if probability is None:
            continue
        record["rows"].append({
            "sample_index": int(row["sample_index"]), "position": int(row["position"]), "tool": row["tool"],
            "probability": min(max(float(probability), floor), 1.0 - floor), "probability_source": row.get("probability_source"),
            "baseline": baseline, "advantage": q[row["tool"]] - baseline,
        })
    record.update(
        covered=True, q=q, q_branch=q_branch, policy=policy, baseline=baseline,
        probability_basis=behaviour["probability_basis"], mass_on_neither_choice=behaviour["mass_on_neither_choice"],
        # The better choice by the forced branches, and whether they tell the choices apart at all.
        q_gap_commit_minus_decline=q_branch[COMMIT_STATE] - q_branch[ROLLBACK_STATE],
        informative=abs(q_branch[COMMIT_STATE] - q_branch[ROLLBACK_STATE]) > 1e-9,
    )
    return record


def summarize_values(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What an iteration's continuation values say: coverage, how often the choices differ, and by kind of state."""

    covered = [state for state in states if state.get("covered")]
    rows = [row for state in covered for row in state["rows"]]

    def kind(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        gaps = [float(item["q_gap_commit_minus_decline"]) for item in items]
        return {
            "states": len(items),
            "informative": sum(1 for item in items if item["informative"]),
            "commit_better": sum(1 for gap in gaps if gap > 1e-9),
            "decline_better": sum(1 for gap in gaps if gap < -1e-9),
            "mean_q_commit": _mean([float(item["q_branch"][COMMIT_STATE]) for item in items]),
            "mean_q_decline": _mean([float(item["q_branch"][ROLLBACK_STATE]) for item in items]),
            "continuations_solved": {choice: sum(int(item["solved"]["forced"][choice]) for item in items) for choice in CHOICES},
            "continuations": {choice: sum(len(item["returns"]["forced"][choice]) for item in items) for choice in CHOICES},
        }

    # Where the audit's membership label and the realized values point different ways.
    conflicts = [state for state in covered if state["informative"] and state["decision"].get("label") in CHOICES
                 and (state["q_gap_commit_minus_decline"] > 0) != (state["decision"]["label"] == COMMIT_STATE)]
    # How the behaviour probabilities were known, and how much of the policy's mass was on neither choice where
    # both were on record (the complement taken elsewhere assumes that mass is nothing).
    bases: dict[str, int] = {}
    for state in covered:
        basis = str(state.get("probability_basis") or "no_probability_read")
        bases[basis] = bases.get(basis, 0) + 1
    outside = [float(state["mass_on_neither_choice"]) for state in covered if state.get("mass_on_neither_choice") is not None]
    return {
        "states": len(states),
        "covered": len(covered),
        "uncovered": len(states) - len(covered),
        "equal_return_states": sum(1 for state in covered if not state["informative"]),
        "sampled_rows_credited": len(rows),
        "mean_abs_advantage": _mean([abs(float(row["advantage"])) for row in rows]),
        "probability_basis": dict(sorted(bases.items())),
        "mass_on_neither_choice": {"states_measured": len(outside), "mean": _mean(outside), "max": max(outside) if outside else None},
        "states_with_inconsistent_probabilities": sum(1 for state in states if state.get("recorded_probabilities") is not None),
        "branches_invalid": sum(int(state["branches"]["invalid"]) for state in states),
        "branches_mismatched_prefix": sum(int(state["branches"]["mismatched_prefix"]) for state in states),
        "all": kind(covered),
        "by_stratum": {name: kind([state for state in covered if state["stratum"] == name]) for name in STRATA},
        "membership_label_against_values": {
            "informative_states_with_a_label": sum(1 for state in covered if state["informative"]
                                                   and state["decision"].get("label") in CHOICES),
            "label_and_values_disagree": len(conflicts),
            "roots": sorted({str(state["root"]) for state in conflicts}),
        },
    }


def values_map(payload: Mapping[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """``(root, sample_index, position)`` of every credited sampled decision to its local advantage, for the row builder."""

    mapped: dict[tuple[str, int, int], dict[str, Any]] = {}
    for state in payload.get("states") or []:
        if not state.get("covered"):
            continue
        for row in state.get("rows") or []:
            mapped[(str(state["root"]), int(row["sample_index"]), int(row["position"]))] = {
                "advantage": float(row["advantage"]), "tool": str(row["tool"]),
            }
    return mapped


def check_provenance(payload: Mapping[str, Any], *, iteration: int, adapter: str, observation_mode: str) -> None:
    """Refuse values that were not sampled under this iteration's policy and observation mode."""

    if payload.get("contract") != DECISION_CREDIT_CONTRACT:
        raise ValueError(f"the decision values carry contract {payload.get('contract')!r}, not {DECISION_CREDIT_CONTRACT!r}")
    wrong = {name: (payload.get(name), wanted) for name, wanted in
             (("iteration", int(iteration)), ("adapter", str(adapter)), ("observation_mode", str(observation_mode)))
             if payload.get(name) != wanted}
    if wrong:
        raise ValueError(
            "the decision values were sampled under another policy or observation mode and are not targets for this "
            f"update (found, wanted): {wrong}"
        )


def load_values(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "DECISION_CREDIT_CONTRACT",
    "DEFAULT_QUOTAS",
    "DecisionCreditSettings",
    "PROBABILITY_SUM_TOLERANCE",
    "STRATA",
    "StateGroup",
    "behaviour_policy",
    "candidate_states",
    "check_provenance",
    "decision_token_probability",
    "load_values",
    "select_states",
    "state_stratum",
    "state_values",
    "summarize_values",
    "values_map",
]
