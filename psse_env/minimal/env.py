"""The minimal environment: the model decides, the tools compute, nothing else intervenes.

One episode is a list of saved states (``VersionStore``) and six actions
(``psse_env.minimal.actions``).  A correction is one reversible action: the
environment saves the state, runs the tool on the object the model named,
re-estimates the new state and reports what actually changed and what the
estimator now shows.  Whether to keep the change, undo it, look further or
finish is the model's decision; the environment has no verdict to give.

What the environment checks is the execution contract and nothing more: the
call parses, the tool exists, the arguments have the declared types, the named
objects exist, and the tool produced a usable change.  A check that fails is a
tool error and leaves the state as it was.  A state whose estimate does not
converge is still a state: it is saved and reported as not converged, and the
model may undo it or go on.

The environment never sees the truth.  ``reset`` takes the execution part of a
scenario and passes it through the evidence profile's sanitizer, so whatever a
scenario carries beside it (the audit's truth, the grouping) cannot reach a
tool or an observation.  Scoring is ``psse_env.minimal.scoring``, called by
the trainer or the evaluator on the finished episode.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from mcp_server.matpower_server import _load_python_case
from psse_env.evidence_profile import sanitize_execution_for_profile
from psse_env.minimal.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    ERROR_ARGUMENTS,
    ERROR_NO_CHANGE,
    ERROR_TOOL,
    ERROR_UNKNOWN_OBJECT,
    FINISH,
    INSPECT,
    UNDO,
    VIEW_BREAKERS,
    ActionError,
    parse_action,
    tool_schemas,
    validate,
)
from psse_env.minimal.kernel import MAX_FLIP_TESTS_PER_CALL, NumericalKernel
from psse_env.minimal.versions import VersionError, VersionStore, state_changes

CONTRACT = "minimal_env_v1"
DEFAULT_MAX_ACTIONS = 16
#: The versions whose tables an observation shows: the current one and the one current before it, which is what a
#: before-and-after reading needs.  A table is about 4,000 tokens (every digit is a token), so showing every
#: version's would triple the prompt of a long episode; an older version's tables are one ``inspect`` away, and
#: every version's summary numbers are always shown.
DEFAULT_TABLE_VERSIONS = 2
OUTCOME_BUDGET = "budget_exhausted"
ESTIMATE_SUMMARY_KEYS = ("converged", "error", "chi_square", "chi_square_threshold", "chi_square_dof", "max_residual",
                         "residual_threshold", "bus_count", "line_count")
ESTIMATE_TABLE_KEYS = ("buses", "lines", "meters")
SUBSTATION_SUMMARY_KEYS = ("converged", "chi_square", "chi_square_threshold", "chi_square_dof", "max_residual")


def execution_part(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a scenario the environment may execute on: never the audit's truth."""
    if "execution" in scenario and "scenario_schema_version" in scenario:
        from psse_env.dagger.evaluator import _partitioned_scenario_parts

        return _partitioned_scenario_parts(scenario)[0]
    from psse_env.dagger.evaluator import privileged_execution_paths

    leaked = privileged_execution_paths(scenario)
    if leaked:
        raise ValueError("the scenario handed to the environment carries audit-only fields: " + ", ".join(leaked))
    return copy.deepcopy(dict(scenario))


class MinimalDiagnosisEnv:
    """Saved states, six actions, the numerical kernel; no controller."""

    def __init__(self, kernel: NumericalKernel | None = None, *, max_actions: int = DEFAULT_MAX_ACTIONS,
                 table_versions: int | None = DEFAULT_TABLE_VERSIONS, max_flips: int = MAX_FLIP_TESTS_PER_CALL) -> None:
        if isinstance(max_actions, bool) or int(max_actions) < 1:
            raise ValueError(f"max_actions must be a positive integer, got {max_actions!r}")
        if table_versions is not None and int(table_versions) < 1:
            raise ValueError(f"table_versions must be positive or None, got {table_versions!r}")
        self.kernel = kernel or NumericalKernel()
        self.max_actions = int(max_actions)
        #: How many versions show their tables in the observation, most recently current or inspected first;
        #: ``None`` shows every version's.  A version outside the window is one ``inspect`` away.
        self.table_versions = None if table_versions is None else int(table_versions)
        self.max_flips = int(max_flips)
        self.store: VersionStore | None = None
        self.scenario_id: str | None = None
        self.done = False
        self.outcome: str | None = None
        self._estimates: dict[int, dict[str, Any]] = {}
        self._substations: dict[int, dict[str, Any]] = {}
        self._flips: dict[int, list[str]] = {}
        self._modifications: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._focus: list[int] = []
        self._steps = 0
        self._breakers = False

    # ------------------------------------------------------------------ episode

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        execution = execution_part(scenario)
        self.scenario_id = str(execution.get("scenario_id") or execution.get("id") or "scenario")
        observable = sanitize_execution_for_profile(execution, self.kernel.providers.evidence_profile)
        self.store = VersionStore(
            observable.get("case", observable.get("case_path")),
            observable.get("measurements", observable.get("z_obs", [])),
            observable.get("metadata") or {},
        )
        self.kernel.reset_cost()
        self.done = False
        self.outcome = None
        self._estimates = {0: self.kernel.estimate(self.store.state(0))}
        self._substations = {}
        self._flips = {}
        self._modifications = []
        self._history = []
        self._focus = [0]
        self._steps = 0
        self._breakers = self.kernel.has_substation_model(self.store.state(0))
        return self.observation()

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def breakers(self) -> bool:
        return self._breakers

    def tools(self) -> list[dict[str, Any]]:
        return tool_schemas(breakers=self._breakers, max_flips=self.max_flips)

    def _require_store(self) -> VersionStore:
        if self.store is None:
            raise RuntimeError("reset the environment before using it")
        return self.store

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one action.  Returns the next observation and the action's result."""
        store = self._require_store()
        if self.done:
            raise RuntimeError("the episode is over")
        self._steps += 1
        event: dict[str, Any] = {"step": self._steps, "version": store.current}
        try:
            parsed = parse_action(action)
            event["action"] = parsed
            call = validate(parsed, breakers=self._breakers, max_flips=self.max_flips)
            event["action"] = call
            result = self._dispatch(call)
        except ActionError as error:
            event.setdefault("action", {"tool": None, "arguments": {}})
            result = {"status": "error", "error_code": error.code, "error_detail": error.detail}
        event["result"] = result
        self._history.append(event)
        if not self.done and self._steps >= self.max_actions:
            self.done = True
            self.outcome = OUTCOME_BUDGET
        return self.observation(), copy.deepcopy(result)

    def stop(self, outcome: str) -> None:
        """End the episode from outside, with no declaration: for a runner whose policy cannot act any more."""
        self._require_store()
        if not self.done:
            self.done = True
            self.outcome = str(outcome)

    # ----------------------------------------------------------------- dispatch

    def _dispatch(self, call: Mapping[str, Any]) -> dict[str, Any]:
        tool = call["tool"]
        arguments = call["arguments"]
        if tool == INSPECT:
            return self._inspect(arguments)
        if tool == UNDO:
            return self._undo(arguments)
        if tool == FINISH:
            self.done = True
            self.outcome = str(arguments["outcome"])
            return {"status": "ok", "outcome": self.outcome}
        return self._correct(tool, arguments)

    def _touch(self, version: int) -> None:
        self._focus = [version] + [index for index in self._focus if index != version]

    def _inspect(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        version = int(arguments.get("version", store.current))
        if not store.exists(version):
            raise ActionError(ERROR_UNKNOWN_OBJECT, f"version {version}")
        if arguments["view"] != VIEW_BREAKERS:
            self._touch(version)
            return {"status": "ok", "view": arguments["view"], "version": version}
        state = store.state(version)
        names = set(self.kernel.breaker_names(state))
        unknown = [name for name in arguments.get("flip") or [] if name not in names]
        if unknown:
            raise ActionError(ERROR_UNKNOWN_OBJECT, "breaker " + ", ".join(unknown))
        flips = list(dict.fromkeys(self._flips.get(version, []) + list(arguments.get("flip") or [])))
        output = self.kernel.substation(state, flips)
        if not output.get("ok"):
            raise ActionError(ERROR_TOOL, f"{output.get('error_code')}: {output.get('error_detail')}")
        self._flips[version] = flips
        self._substations[version] = {key: value for key, value in output.items() if key != "ok"}
        self._touch(version)
        summary = {key: output[key] for key in SUBSTATION_SUMMARY_KEYS if key in output}
        return {"status": "ok", "view": VIEW_BREAKERS, "version": version, "substation": summary}

    def _line_count(self, state: Mapping[str, Any]) -> int:
        return int(_load_python_case(str(state["case"]))["branch"].shape[0])

    def _correct(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        before = store.state()
        if "line" in arguments and not 1 <= int(arguments["line"]) <= self._line_count(before):
            raise ActionError(ERROR_UNKNOWN_OBJECT, f"line {arguments['line']}")
        if tool == CORRECT_MEASUREMENTS:
            count = len(before["measurements"])
            unknown = [index for index in arguments["meters"] if not 0 <= int(index) < count]
            if unknown:
                raise ActionError(ERROR_UNKNOWN_OBJECT, "meter " + ", ".join(map(str, unknown)))
            output = self.kernel.correct_measurements(before, arguments["meters"])
        elif tool == CORRECT_PARAMETERS:
            output = self.kernel.correct_parameters(before, int(arguments["line"]))
        elif tool == CORRECT_TOPOLOGY and self._breakers:
            if arguments["breaker"] not in set(self.kernel.breaker_names(before)):
                raise ActionError(ERROR_UNKNOWN_OBJECT, f"breaker {arguments['breaker']}")
            output = self.kernel.correct_topology(before, breaker=arguments["breaker"], status=arguments["status"])
        elif tool == CORRECT_TOPOLOGY:
            output = self.kernel.correct_topology(before, line=int(arguments["line"]), status=arguments["status"])
        else:  # validate() admits no other tool
            raise ActionError(ERROR_ARGUMENTS, str(tool))
        if not output.get("ok"):
            code = str(output.get("error_code") or "")
            raise ActionError(ERROR_NO_CHANGE if code.endswith("_no_change") else ERROR_TOOL,
                              f"{code}: {output.get('error_detail')}")
        parent = store.current
        try:
            child = store.apply(output["modification"], step=self._steps)
        except VersionError as error:
            raise ActionError(ERROR_NO_CHANGE, str(error)) from error
        after = store.state(child)
        self._estimates[child] = self.kernel.estimate(after)
        self._touch(child)
        record = {
            "id": len(self._modifications) + 1,
            "step": self._steps,
            "tool": tool,
            "arguments": copy.deepcopy(dict(arguments)),
            "from_version": parent,
            "to_version": child,
            "changes": state_changes(before, after),
        }
        if output.get("report"):
            record["report"] = copy.deepcopy(output["report"])
        self._modifications.append(record)
        return {"status": "ok", "modification": record["id"], "from_version": parent, "to_version": child}

    def _undo(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        current = store.current
        target = arguments.get("version")
        if target is None:
            target = store.version().parent
            if target is None:
                raise ActionError(ERROR_NO_CHANGE, "no correction is in effect")
        if not store.exists(target):
            raise ActionError(ERROR_UNKNOWN_OBJECT, f"version {target}")
        try:
            store.restore(int(target))
        except VersionError as error:
            raise ActionError(ERROR_NO_CHANGE, str(error)) from error
        self._touch(int(target))
        return {"status": "ok", "from_version": current, "to_version": int(target)}

    # -------------------------------------------------------------- observation

    def observation(self) -> dict[str, Any]:
        """Everything the model may read, as data.  ``psse_env.minimal.observation`` renders it."""
        store = self._require_store()
        lineage = set(store.lineage())
        shown = set(range(len(store))) if self.table_versions is None else set(self._focus[: self.table_versions])
        versions: list[dict[str, Any]] = []
        for index in range(len(store)):
            estimate = self._estimates[index]
            entry: dict[str, Any] = {
                "version": index,
                "parent": store.version(index).parent,
                "in_effect": index in lineage,
                "estimate": {key: estimate[key] for key in ESTIMATE_SUMMARY_KEYS if key in estimate},
            }
            if index in shown:
                entry["estimate"].update({key: estimate[key] for key in ESTIMATE_TABLE_KEYS if key in estimate})
            substation = self._substations.get(index)
            if substation is not None:
                entry["substation"] = {key: substation[key] for key in SUBSTATION_SUMMARY_KEYS if key in substation}
                if index in shown and "breakers" in substation:
                    entry["substation"]["breakers"] = substation["breakers"]
            versions.append(entry)
        modifications = [dict(record, in_effect=record["to_version"] in lineage) for record in self._modifications]
        return copy.deepcopy({
            "contract": CONTRACT,
            "breaker_model": self._breakers,
            "budget": {"limit": self.max_actions, "used": self._steps, "remaining": max(self.max_actions - self._steps, 0)},
            "current_version": store.current,
            "versions": versions,
            "modifications": modifications,
            "history": self._history,
        })

    # ------------------------------------------------- for the trainer, not the model

    def final_state(self) -> dict[str, Any]:
        """The current state's case, measurements and metadata, for scoring."""
        return self._require_store().state()

    def initial_state(self) -> dict[str, Any]:
        return self._require_store().state(0)

    def modifications_in_effect(self) -> list[dict[str, Any]]:
        lineage = set(self._require_store().lineage())
        return [copy.deepcopy(record) for record in self._modifications if record["to_version"] in lineage]

    def history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._history)

    def estimate(self, version: int | None = None) -> dict[str, Any]:
        store = self._require_store()
        return copy.deepcopy(self._estimates[store.current if version is None else int(version)])

    def cost(self) -> dict[str, Any]:
        """Model calls and what the numerical tools were asked to do for them."""
        return {"actions": self._steps, **self.kernel.cost.snapshot()}


__all__ = ["CONTRACT", "DEFAULT_MAX_ACTIONS", "DEFAULT_TABLE_VERSIONS", "MinimalDiagnosisEnv", "OUTCOME_BUDGET", "execution_part"]
