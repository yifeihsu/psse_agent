"""One-step forks of a policy's own episode: what each other tool would have earned.

For a case, the policy's own episode is run once.  Then, at every state of that episode and for every other
tool, the episode is run again: the actions before the state are replayed through the environment, the tool is
forced (the policy writes the arguments itself, ``act_forced``), and the policy continues to the end under the
budget the episode has left.  A branch is scored like any episode (``scoring.episode_reward``), prefix included,
so its return is the return an ordinary episode with that action would have had.

``finish`` is the one tool whose calls are few enough to try them all: at every state both of its outcomes are
forked as whole calls given by the harness (the own action left out).  No model call is needed, because the
episode ends there.  That is what puts "declare the case unresolved" at the state where the own episode claimed
it resolved; a ``finish`` whose outcome the model writes is not forked.

What it is for (RL plan of 2026-11-22, stage B).  A policy-gradient update learns an action only from episodes
that sampled it, in proportion to p(1 - p): an action the policy gives a probability of 0.0003 is never learned.
A fork measures such an action's outcome directly.  A state whose best branch returned more than going on as the
own episode did can still reach names an action the policy could have taken there and done better with; the RL
update trains that action on that state's prompt, weighted by that excess (``better_branches``,
``research.minimal_env.rl_train``).

Why "can still reach" and not the own return (the probe's smoke of 2026-11-22).  Against the own return alone, a
case that one forced look repairs at its third state also taught "declare it unresolved" at its first two, because
stopping at once costs fewer actions than the own failed episode.  The two lessons contradict each other.  So a
state is compared with the best of the own return and the branches kept at later states: before a later, better
branch the own action is the right one, since it leads there.

What it is not.  Nothing here says which states or tools matter: the forks are at every state and over every
tool, and the truth enters only through the reward, as in every sampled episode.  A forced step is an
intervention of the harness, not a choice of the policy: the steps after it are never training rows.  And the
branching is a training-time measurement: an evaluated episode is one pass, one action at a time.

Replays are checked.  At every replayed step the policy must be shown what it was shown in the own episode (a
digest of the rendered observation); a branch whose state has drifted is marked and never used.

``argument_samples`` adds branches whose forced call is written by a second, sampling policy (every tool but
``finish``, the policy's own included): they try other objects of one tool, which a forced tool with the policy's
own decoding does not.  A sampled call that repeats one already tried at the state is not run again.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from psse_env.minimal import observation as view
from psse_env.minimal.actions import (
    CORRECTION_TOOLS,
    FINISH,
    INSPECT,
    OUTCOMES,
    OUTCOME_UNRESOLVED,
    VIEW_BREAKERS,
    VIEW_ESTIMATE,
    tool_schemas,
)
from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.episode import run_episode
from psse_env.minimal.scoring import RewardWeights
from psse_env.rl.rollout import sample_seed, scenario_root

#: v2 (2026-11-22): both outcomes of ``finish`` given by the harness; ``better`` compares a state with what going
#: on can still reach and carries ``margin``.
FORK_CONTRACT = "minimal_one_step_forks_v2"
#: Who wrote the forced call's arguments: the policy's own decoding, the sampling policy, or the harness (the two
#: outcomes of ``finish``, a whole call each).
ARGUMENTS_GREEDY = "own_decoding"
ARGUMENTS_SAMPLED = "sampled"
ARGUMENTS_GIVEN = "given"
#: A branch that ran to the end of its episode with a valid call of the forced tool.
STATUS_RAN = "ran"
#: The replay showed the policy something else than the own episode did: the branch is not that state's.
STATUS_DRIFTED = "drifted"
#: What the model wrote after the tool's name is not a valid call of that tool.
STATUS_INVALID = "invalid_call"
#: The forced call repeats one already tried at the state (the own action or an earlier branch).
STATUS_DUPLICATE = "duplicate"
#: The policy raised at the forced step (a prompt over its input limit).
STATUS_NO_CALL = "no_call"
STATUSES = (STATUS_RAN, STATUS_DRIFTED, STATUS_INVALID, STATUS_DUPLICATE, STATUS_NO_CALL)
#: ``sample_index`` of the seed a case's argument-sampling policy is built with: beside the sampled episodes.
FORK_SAMPLE = 2000
_GAIN_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ForkSettings:
    """Whether a run forks, and how the update reads the branches."""

    enabled: bool = False
    #: Weight of a fork row, times the branch's margin: how far its return lies above what going on as the own
    #: episode did can still reach (the own return, or a branch kept at a later state).
    weight: float = 1.0
    #: A branch teaches only when its margin is more than this.  0 keeps every margin, the 0.01 of one action
    #: saved included.
    min_gain: float = 0.0
    #: Further branches per tool whose forced call a sampling policy writes (0: none).
    argument_samples: int = 0
    #: Further sampled branches of the own episode's tool alone when that tool is a correction tool
    #: (``correct_topology``, ``correct_parameters``, ``correct_measurements``), so wrong-object turns are
    #: explored without sampling every other tool (0: none).
    own_tool_samples: int = 0
    #: End a branch immediately after a forced call the environment refused (``better_branches`` never teaches
    #: a refused call; ``False`` keeps the continuation for diagnostics such as ``fork_probe``).
    stop_on_refused_call: bool = False
    #: Treat a forced ``inspect`` whose requested tables and breaker flips are already shown at that state as a
    #: no-op: end the branch without running its continuation and never select it in ``better_branches``.
    ignore_noop_inspect: bool = False
    #: Skip forking states preceded by more than this many consecutive refused calls in the own episode, avoiding
    #: 16-state runaway refusal loops (``None``: fork every state).
    max_consecutive_refused_states: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ForkSettings":
        payload = dict(payload or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown forks fields: {unknown}")
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("forks.enabled must be true or false")
        for name in ("weight", "min_gain"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) < float("inf"):
                raise ValueError(f"forks.{name} must be finite and nonnegative")
        if self.enabled and not float(self.weight) > 0.0:
            raise ValueError("forks.weight must be positive when forks are enabled")
        for name in ("argument_samples", "own_tool_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"forks.{name} must be a nonnegative integer")
        for name in ("stop_on_refused_call", "ignore_noop_inspect"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"forks.{name} must be true or false")
        cap = self.max_consecutive_refused_states
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError("forks.max_consecutive_refused_states must be a nonnegative integer when set")


# ------------------------------------------------------------------ states


def state_digest(observation: Mapping[str, Any]) -> str:
    """A digest of what the model reads at a state: the rendered user turn, and whether its tools name breakers."""
    text = ("1" if observation.get("breaker_model") else "0") + view.user_text(observation)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def offered_tools(observation: Mapping[str, Any]) -> list[str]:
    """The names of the tools an observation offers, in the order the model is shown them."""
    return [str(row["function"]["name"]) for row in tool_schemas(breakers=bool(observation.get("breaker_model")))]


def signature(action: Any) -> str:
    """One string per distinct call (or per distinct unreadable text)."""
    if isinstance(action, Mapping):
        return json.dumps({"tool": action.get("tool"), "arguments": action.get("arguments") or {}}, sort_keys=True, default=str)
    return f"text:{action!r}"


def is_noop_inspect(observation: Mapping[str, Any], action: Any) -> bool:
    """Whether ``action`` is an ``inspect`` call whose requested tables and breaker flips are already shown."""
    if not isinstance(action, Mapping) or action.get("tool") != INSPECT:
        return False
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return False
    target_version = arguments.get("version", observation.get("current_version"))
    if isinstance(target_version, bool) or not isinstance(target_version, int):
        return False
    versions = {int(entry["version"]): entry for entry in observation.get("versions") or [] if "version" in entry}
    entry = versions.get(int(target_version))
    if not isinstance(entry, Mapping):
        return False
    kind = arguments.get("view")
    if kind == VIEW_ESTIMATE:
        return "meters" in (entry.get("estimate") or {})
    if kind == VIEW_BREAKERS:
        substation = entry.get("substation")
        if not isinstance(substation, Mapping) or "breakers" not in substation:
            return False
        flips = arguments.get("flip") or []
        if not flips:
            return True
        tested = {
            str(row.get("cb"))
            for row in substation.get("breakers") or []
            if isinstance(row, Mapping) and row.get("flip_chi_square") is not None
        }
        return all(str(name) in tested for name in flips)
    return False


def class_of(root: Mapping[str, Any]) -> str:
    return str(((root.get("audit") or {}).get("compound") or {}).get("class")
               or (root.get("grouping") or {}).get("scenario_family") or "")


class _Recorder:
    """The policy, keeping every state it was asked at and what it answered."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.states: list[dict[str, Any]] = []

    @property
    def last_action_metrics(self) -> Any:
        return getattr(self.policy, "last_action_metrics", None)

    def act(self, observation: Mapping[str, Any]) -> Any:
        state: dict[str, Any] = {"observation": copy.deepcopy(dict(observation)), "action": None, "raised": None}
        self.states.append(state)
        try:
            action = self.policy.act(observation)
        except Exception as exc:
            state["raised"] = f"{type(exc).__name__}: {exc}"
            raise
        state["action"] = copy.deepcopy(action)
        return action


_NOT_CALLED = object()


class ForkPolicy:
    """Replays a prefix, forces one tool, then hands the episode over to the policy.

    ``prefix`` holds the actions before the fork's state as the environment took them in the own episode (the
    call, the model's words where they were not one, ``None`` where the policy raised) and ``digests`` the
    digest of every state up to the fork's, that one included.  ``forced_policy`` writes the forced call
    (``act_forced(observation, tool)``); the policy itself unless another is named.  ``given`` is a whole call
    of ``tool`` that the harness gives instead, with no model call.  ``seen`` holds the signatures of the calls
    already tried at the state.

    After the episode: ``drifted_at`` (the first replayed step whose state was not the own episode's, or
    ``None``), ``forced`` (what the forced step returned), ``status`` and ``forced_generation`` (the tokens of
    the forced call where the policy records them).  A branch that cannot be used ends its episode at once.
    """

    def __init__(self, policy: Any, *, prefix: Sequence[Any], digests: Sequence[str], tool: str,
                 forced_policy: Any | None = None, seen: set[str] | None = None, given: Mapping[str, Any] | None = None,
                 stop_on_refused_call: bool = False, ignore_noop_inspect: bool = False) -> None:
        if len(digests) != len(prefix) + 1:
            raise ValueError("a fork needs the digest of every state up to its own: one more than its prefix has actions")
        if given is not None and given.get("tool") != str(tool):
            raise ValueError("a given call must be a call of the forced tool")
        self._policy = policy
        self._forced_policy = forced_policy if forced_policy is not None else policy
        self._given = None if given is None else copy.deepcopy(dict(given))
        self._prefix = [copy.deepcopy(action) for action in prefix]
        self._digests = [str(value) for value in digests]
        self.tool = str(tool)
        self._seen = seen
        self._stop_on_refused_call = bool(stop_on_refused_call)
        self._ignore_noop_inspect = bool(ignore_noop_inspect)
        self._position = 0
        self._acting: Any = None
        self.drifted_at: int | None = None
        self.forced: Any = _NOT_CALLED
        self.forced_generation: dict[str, Any] | None = None
        self.status: str = STATUS_NO_CALL
        self.noop: bool = False
        self.stopped_early: bool = False

    @property
    def last_action_metrics(self) -> Any:
        return getattr(self._acting, "last_action_metrics", None) if self._acting is not None else None

    @staticmethod
    def _leave() -> dict[str, Any]:
        return {"tool": FINISH, "arguments": {"outcome": OUTCOME_UNRESOLVED}}

    def act(self, observation: Mapping[str, Any]) -> Any:
        index = self._position
        self._position += 1
        self._acting = None
        if self.status in (STATUS_DRIFTED, STATUS_INVALID, STATUS_DUPLICATE) or self.noop or self.stopped_early:
            return self._leave()  # the environment's budget, not this policy, ends an episode that goes on
        if index <= len(self._prefix) and state_digest(observation) != self._digests[index]:
            self.drifted_at = index
            self.status = STATUS_DRIFTED
            return self._leave()
        if index < len(self._prefix):
            return copy.deepcopy(self._prefix[index])
        if index > len(self._prefix):
            if self._stop_on_refused_call and index == len(self._prefix) + 1:
                history = observation.get("history") or []
                if history and ((history[-1].get("result") or {}).get("status") != "ok"):
                    self.stopped_early = True
                    return self._leave()
            self._acting = self._policy
            return self._policy.act(observation)
        if self._given is not None:
            self.forced = copy.deepcopy(self._given)  # the harness's call: no model is asked
        else:
            self._acting = self._forced_policy
            self.forced = self._forced_policy.act_forced(observation, self.tool)
            generation = getattr(self._forced_policy, "last_generation", None)
            if isinstance(generation, Mapping) and generation.get("forced_prefix_tokens"):
                self.forced_generation = {"completion_token_ids": [int(value) for value in generation.get("completion_token_ids") or []],
                                          "forced_prefix_tokens": int(generation["forced_prefix_tokens"])}
        if not isinstance(self.forced, Mapping) or self.forced.get("tool") != self.tool:
            self.status = STATUS_INVALID
            return self._leave()
        if self._seen is not None and signature(self.forced) in self._seen:
            self.status = STATUS_DUPLICATE
            return self._leave()
        if self._ignore_noop_inspect and is_noop_inspect(observation, self.forced):
            self.noop = True
            self.stopped_early = True
        self.status = STATUS_RAN
        return copy.deepcopy(self.forced)


# ------------------------------------------------------------------ one branch, one case


def _brief_action(action: Any) -> Any:
    if isinstance(action, Mapping):
        return {"tool": action.get("tool"), "arguments": copy.deepcopy(dict(action.get("arguments") or {}))}
    return None if action is None else str(action)[:600]


def run_branch(root: Mapping[str, Any], *, env: MinimalDiagnosisEnv, policy: Any, prefix: Sequence[Any],
               digests: Sequence[str], tool: str, weights: RewardWeights | None = None, forced_policy: Any | None = None,
               seen: set[str] | None = None, given: Mapping[str, Any] | None = None,
               stop_on_refused_call: bool = False, ignore_noop_inspect: bool = False) -> dict[str, Any]:
    """One branch: the prefix replayed, ``tool`` forced at the state after it, the policy to the end.

    ``return`` and the outcome are those of the whole episode, as scored for any episode; they mean something
    only for a branch whose ``status`` is ``ran``.  ``executed`` says whether the environment carried the forced
    call out (a call it refused changed nothing).  ``given`` is a whole call the harness gives as the forced one.
    """
    fork = ForkPolicy(policy, prefix=prefix, digests=digests, tool=tool, forced_policy=forced_policy, seen=seen, given=given,
                      stop_on_refused_call=stop_on_refused_call, ignore_noop_inspect=ignore_noop_inspect)
    episode = run_episode(env, root, fork, weights=weights, keep_observations=False)
    at = len(prefix)
    steps = episode["steps"]
    record: dict[str, Any] = {"state": at, "tool": str(tool), "status": fork.status,
                              "action": None if fork.forced is _NOT_CALLED else _brief_action(fork.forced)}
    if fork.drifted_at is not None:
        record["drifted_at"] = int(fork.drifted_at)
    if fork.forced_generation is not None:
        record["forced_generation"] = fork.forced_generation
    if fork.status != STATUS_RAN:
        return record
    score = episode["score"]
    forced_step = steps[at] if at < len(steps) else {}
    result = forced_step.get("result") or {}
    record.update({
        "executed": result.get("status") == "ok",
        "error_code": result.get("error_code"),
        "return": float(episode["reward"]["total"]),
        "solved": bool(score["solved"]),
        "strict_solved": bool((score.get("strict") or {}).get("solved")),
        "declared": score.get("declared"),
        "actions": int(score.get("actions") or 0),
        "outcome": episode["outcome"],
        "continuation": [_brief_action(step.get("action")) for step in steps[at + 1:]],
        "seconds": episode["seconds"],
    })
    if fork.noop:
        record["noop"] = True
    if fork.stopped_early:
        record["stopped_early"] = True
    return record


def better_branches(record: Mapping[str, Any], min_gain: float = 0.0) -> list[dict[str, Any]]:
    """Per state of the own episode, the branch to teach there, where there is one.

    A state's candidate is its branch with the highest return among those that ran with a call the environment
    carried out: a refused call is not an action to teach, and between equal returns the earlier branch is taken
    (a call in the policy's own decoding before a sampled one).

    The candidate is kept when its return exceeds, by more than ``min_gain``, what going on as the own episode did
    can still reach: the own return, or the return of a branch kept at a later state.  So a state before a later,
    better branch teaches nothing, because the own action there leads to that branch.  ``margin`` is that excess
    and the weight of the state's training row; ``ahead`` is what it was measured against; ``gain`` is the
    excess over the own return.  With ``min_gain`` 0 the margins of a case add up to its best return less the own.
    """
    own = float(record["own"]["return"])
    best: dict[int, dict[str, Any]] = {}
    for position, branch in enumerate(record.get("branches") or []):
        if branch.get("status") != STATUS_RAN or not branch.get("executed") or branch.get("noop"):
            continue
        state = int(branch["state"])
        held = best.get(state)
        if held is None or float(branch["return"]) > held["return"] + _GAIN_TOLERANCE:
            best[state] = {
                "state": state, "branch": position, "tool": branch["tool"], "action": copy.deepcopy(branch["action"]),
                "arguments": branch.get("arguments"), "return": float(branch["return"]),
                "solved": bool(branch["solved"]), "actions": int(branch["actions"])}
    kept: list[dict[str, Any]] = []
    ahead = own
    for state in sorted(best, reverse=True):
        item = best[state]
        margin = item["return"] - ahead
        if not margin > float(min_gain) + _GAIN_TOLERANCE:
            continue
        item.update({"gain": item["return"] - own, "margin": margin, "ahead": ahead})
        kept.append(item)
        ahead = item["return"]
    return sorted(kept, key=lambda item: item["state"])


def states_above_the_own_return(record: Mapping[str, Any]) -> list[int]:
    """The states with a carried-out branch above the own return: every state :func:`better_branches` can keep,
    whatever ``min_gain`` it is asked with."""
    own = float(record["own"]["return"])
    return sorted({int(branch["state"]) for branch in record.get("branches") or []
                   if branch.get("status") == STATUS_RAN and branch.get("executed") and not branch.get("noop")
                   and float(branch["return"]) > own + _GAIN_TOLERANCE})


def fork_case(root: Mapping[str, Any], *, env_factory: Callable[[], MinimalDiagnosisEnv], policy: Any,
              weights: RewardWeights | None = None, settings: ForkSettings | None = None, argument_policy: Any | None = None,
              canonical: Callable[[Mapping[str, Any], Mapping[str, Any], Sequence[int]], bool] | None = None) -> dict[str, Any]:
    """The policy's own episode on ``root`` and its one-step forks.

    Returns ``{"record": ..., "observations": {state: observation}}``: the record of the own episode and of every
    branch, with ``better`` (:func:`better_branches`), and the observation of every state with a carried-out
    branch above the own return, which is what a training row of that state is rendered from (every state
    ``better_branches`` can keep, so the update may read the record with another ``min_gain``).

    ``canonical(observation, action, forced_ids)`` is asked, for a forced call whose tokens the policy recorded,
    whether the tokens the harness gave are how that call's own rendering starts; the answer is kept on the
    branch (``prefix_canonical``).
    """
    settings = settings or ForkSettings(enabled=True)
    started = time.perf_counter()
    recorder = _Recorder(policy)
    own = run_episode(env_factory(), root, recorder, weights=weights, keep_observations=False)
    states = recorder.states
    digests = [state_digest(state["observation"]) for state in states]
    replay = [None if state["raised"] else state["action"] for state in states]
    own_steps = own["steps"]
    branches: list[dict[str, Any]] = []
    skipped_refusal_states = 0
    for index, state in enumerate(states):
        if state["raised"]:
            continue  # the policy could not answer here: neither can a forced call
        if settings.max_consecutive_refused_states is not None:
            consecutive = 0
            for earlier in reversed(own_steps[:index]):
                if (earlier.get("result") or {}).get("status") != "ok":
                    consecutive += 1
                else:
                    break
            if consecutive > int(settings.max_consecutive_refused_states):
                skipped_refusal_states += 1
                continue
        own_action = state["action"]
        own_tool = own_action.get("tool") if isinstance(own_action, Mapping) else None
        tools = offered_tools(state["observation"])
        seen = {signature(own_action)}
        # Every other tool with a call the policy writes; ``finish`` with each of its outcomes as a whole call.
        plan: list[tuple[str, str, int | None, dict[str, Any] | None]] = []
        for tool in tools:
            if tool == FINISH:
                calls = [{"tool": FINISH, "arguments": {"outcome": outcome}} for outcome in OUTCOMES]
                plan += [(FINISH, ARGUMENTS_GIVEN, None, call) for call in calls if signature(call) not in seen]
            elif tool != own_tool:
                plan.append((tool, ARGUMENTS_GREEDY, None, None))
        if settings.argument_samples and argument_policy is not None:
            plan += [(tool, ARGUMENTS_SAMPLED, sample, None) for tool in tools if tool != FINISH
                     for sample in range(int(settings.argument_samples))]
        if settings.own_tool_samples and argument_policy is not None and own_tool in CORRECTION_TOOLS:
            offset = int(settings.argument_samples)
            plan += [(str(own_tool), ARGUMENTS_SAMPLED, offset + sample, None)
                     for sample in range(int(settings.own_tool_samples))]
        for tool, arguments, sample, given in plan:
            branch = run_branch(root, env=env_factory(), policy=policy, prefix=replay[:index], digests=digests[: index + 1],
                                tool=tool, weights=weights, forced_policy=argument_policy if arguments == ARGUMENTS_SAMPLED else None,
                                seen=seen, given=given, stop_on_refused_call=settings.stop_on_refused_call,
                                ignore_noop_inspect=settings.ignore_noop_inspect)
            branch["arguments"] = arguments
            if sample is not None:
                branch["sample"] = int(sample)
            generation = branch.pop("forced_generation", None)
            if branch["status"] == STATUS_RAN:
                seen.add(signature(branch["action"]))
            if generation is not None:
                branch["forced_prefix_tokens"] = int(generation["forced_prefix_tokens"])
                if canonical is not None and isinstance(branch.get("action"), Mapping) and branch["status"] in (STATUS_RAN, STATUS_DUPLICATE):
                    forced_ids = generation["completion_token_ids"][: generation["forced_prefix_tokens"]]
                    try:
                        branch["prefix_canonical"] = bool(canonical(state["observation"], branch["action"], forced_ids))
                    except ValueError:
                        branch["prefix_canonical"] = False
            branches.append(branch)
    score = own["score"]
    record: dict[str, Any] = {
        "contract": FORK_CONTRACT,
        "scenario_id": own["scenario_id"],
        "class": class_of(root),
        "settings": asdict(settings),
        "own": {
            "return": float(own["reward"]["total"]),
            "solved": bool(score["solved"]),
            "strict_solved": bool((score.get("strict") or {}).get("solved")),
            "declared": score.get("declared"),
            "actions": int(score.get("actions") or 0),
            "outcome": own["outcome"],
            "steps": [{"step": int(step["step"]), "action": _brief_action(step.get("action")),
                       "status": (step.get("result") or {}).get("status"), "error_code": (step.get("result") or {}).get("error_code")}
                      for step in own["steps"]],
        },
        "states": len(states),
        "branches": branches,
        "seconds": {"own": own["seconds"], "total": round(time.perf_counter() - started, 4)},
    }
    if skipped_refusal_states:
        record["states_skipped_after_repeated_refusals"] = skipped_refusal_states
    record["better"] = better_branches(record, settings.min_gain)
    observations = {str(index): states[index]["observation"] for index in states_above_the_own_return(record)}
    return {"record": record, "observations": observations}


# ------------------------------------------------------------------ the store


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "case"


class ForkStore:
    """One file per case under ``directory/cases``, written whole and in one move."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, case: str) -> Path:
        return self.directory / "cases" / f"{_safe_name(case)}.json"

    def has(self, case: str) -> bool:
        return self._path(case).is_file()

    def write(self, case: str, payload: Mapping[str, Any]) -> Path:
        path = self._path(case)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump({"case": str(case), **dict(payload)}, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
        return path

    def read(self, case: str) -> dict[str, Any]:
        return json.loads(self._path(case).read_text(encoding="utf-8"))

    def iter(self) -> Iterator[tuple[str, dict[str, Any]]]:
        folder = self.directory / "cases"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield str(payload["case"]), payload


def fork_roots(scenarios: Sequence[Mapping[str, Any]], *, store: ForkStore, env_factory: Callable[[], MinimalDiagnosisEnv],
               policy_factory: Callable[[], Any], weights: RewardWeights | None = None, settings: ForkSettings | None = None,
               argument_policy_factory: Callable[[int], Any] | None = None, base_seed: int = 0,
               canonical_factory: Callable[[Any], Any] | None = None,
               progress: Callable[[Mapping[str, Any]], None] | None = None) -> int:
    """Fork every case that has no file yet; returns how many were forked.

    The policy is built only when a case is left, so a start that finds everything done loads no model.
    ``argument_policy_factory(seed)`` builds the sampling policy of one case's sampled forced calls;
    ``canonical_factory(policy)`` the check of :func:`fork_case` (``None`` where the policy cannot give one).
    """
    settings = settings or ForkSettings(enabled=True)
    pending = [scenario for scenario in scenarios if not store.has(scenario_root(scenario))]
    if not pending:
        return 0
    policy = policy_factory()
    canonical = canonical_factory(policy) if canonical_factory is not None else None
    for number, scenario in enumerate(pending, start=1):
        case = scenario_root(scenario)
        argument_policy = None
        if (settings.argument_samples or settings.own_tool_samples) and argument_policy_factory is not None:
            argument_policy = argument_policy_factory(sample_seed(base_seed, case, FORK_SAMPLE))
        payload = fork_case(scenario, env_factory=env_factory, policy=policy, weights=weights, settings=settings,
                            argument_policy=argument_policy, canonical=canonical)
        store.write(case, payload)
        if progress is not None:
            record = payload["record"]
            progress({"fork": case, "done": number, "of": len(pending), "own_solved": record["own"]["solved"],
                      "states": record["states"], "branches": len(record["branches"]),
                      "better": [(item["state"], item["tool"], round(item["margin"], 4)) for item in record["better"]],
                      "seconds": record["seconds"]["total"]})
    return len(pending)


# ------------------------------------------------------------------ reading


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts over forked cases: what the branches did, and how many cases a single forced step repairs."""
    totals: Counter = Counter()
    better_by_tool: Counter = Counter()
    repairing_by_tool: Counter = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    gains: list[float] = []
    margins: list[float] = []
    margin_of_repairs = 0.0
    seconds = 0.0
    for record in records:
        own = record["own"]
        branches = record.get("branches") or []
        cell = by_class[str(record.get("class"))]
        totals["cases"] += 1
        cell["cases"] += 1
        totals["own_episode_repaired"] += int(bool(own["solved"]))
        totals["states"] += int(record["states"])
        totals["branches"] += len(branches)
        for branch in branches:
            totals[f"branches_{branch['status']}"] += 1
            if branch["status"] == STATUS_RAN and not branch.get("executed"):
                totals["forced_calls_the_environment_refused"] += 1
            if branch.get("noop"):
                totals["forced_calls_noop_inspect"] += 1
            if branch.get("stopped_early"):
                totals["forced_calls_stopped_early"] += 1
            if "prefix_canonical" in branch:
                totals["forced_prefixes_checked"] += 1
                totals["forced_prefixes_canonical"] += int(bool(branch["prefix_canonical"]))
        repairing = [branch for branch in branches
                     if branch["status"] == STATUS_RAN and branch.get("executed") and not branch.get("noop")
                     and branch.get("solved")]
        # A refused call changed nothing but the history the model reads.  A repair after one is not that call's
        # doing: it says how easily the own episode's last decisions tip, and is the floor to read the count
        # above against.
        after_a_refusal = [branch for branch in branches
                           if branch["status"] == STATUS_RAN and not branch.get("executed") and branch.get("solved")]
        if not own["solved"]:
            totals["own_episode_unrepaired"] += 1
            cell["own_episode_unrepaired"] += 1
            if repairing:
                totals["unrepaired_cases_one_forced_step_repairs"] += 1
                cell["unrepaired_cases_one_forced_step_repairs"] += 1
                first = min(repairing, key=lambda branch: (branch["state"], -float(branch["return"])))
                repairing_by_tool[str(first["tool"])] += 1
            if any(branch.get("arguments") != ARGUMENTS_SAMPLED for branch in repairing):
                totals["unrepaired_cases_one_forced_step_repairs_in_the_policys_own_decoding"] += 1
            if after_a_refusal:
                totals["unrepaired_cases_repaired_after_a_refused_call"] += 1
                cell["unrepaired_cases_repaired_after_a_refused_call"] += 1
        better = record.get("better") or []
        totals["cases_with_a_better_branch"] += int(bool(better))
        totals["states_with_a_better_branch"] += len(better)
        for item in better:
            repairs = bool(item["solved"]) and not own["solved"]
            better_by_tool[str(item["tool"])] += 1
            gains.append(float(item["gain"]))
            margins.append(float(item["margin"]))
            margin_of_repairs += float(item["margin"]) if repairs else 0.0
            totals["better_branches_that_repair_an_unrepaired_case"] += int(repairs)
            totals["better_branches_at_the_last_state"] += int(item["state"] == int(record["states"]) - 1)
        seconds += float((record.get("seconds") or {}).get("total") or 0.0)

    def spread(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {"count": len(ordered), "min": ordered[0] if ordered else None,
                "median": ordered[len(ordered) // 2] if ordered else None, "max": ordered[-1] if ordered else None,
                "at_most_one_action_cost": sum(1 for value in ordered if value <= 0.05)}

    return {
        "contract": FORK_CONTRACT,
        "totals": dict(sorted(totals.items())),
        "better_branches_by_tool": dict(sorted(better_by_tool.items())),
        "first_repairing_branch_by_tool": dict(sorted(repairing_by_tool.items())),
        # ``margin`` is what a row is weighted by; ``gain`` is the same branches against the own return.
        "margin": {**spread(margins), "total": round(sum(margins), 6),
                   "total_of_branches_that_repair_an_unrepaired_case": round(margin_of_repairs, 6)},
        "gain": spread(gains),
        "by_class": {name: dict(sorted(cell.items())) for name, cell in sorted(by_class.items())},
        "seconds": round(seconds, 1),
    }


__all__ = [
    "ARGUMENTS_GIVEN", "ARGUMENTS_GREEDY", "ARGUMENTS_SAMPLED", "FORK_CONTRACT", "FORK_SAMPLE", "ForkPolicy", "ForkSettings",
    "ForkStore", "STATUSES", "STATUS_DRIFTED", "STATUS_DUPLICATE", "STATUS_INVALID", "STATUS_NO_CALL", "STATUS_RAN",
    "better_branches", "class_of", "fork_case", "fork_roots", "is_noop_inspect", "offered_tools", "run_branch", "signature",
    "state_digest", "states_above_the_own_return", "summarize",
]
