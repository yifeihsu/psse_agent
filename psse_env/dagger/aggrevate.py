"""AggreVaTe-lite top-L counterfactual action ranking.

This module intentionally ranks only actions proposed by an oracle/environment;
it never attempts to enumerate a continuous correction space.  Every action is
evaluated on an isolated branch obtained from ``env.clone_from(state)``,
``env.clone()`` or, as a portable fallback, ``copy.deepcopy(env)``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_action(action: Any) -> dict[str, Any]:
    if isinstance(action, str):
        return {"tool": action, "arguments": {}}
    if not isinstance(action, Mapping):
        return {
            "tool": "__invalid_action__",
            "arguments": {"error_code": "schema_error", "raw_type": type(action).__name__},
        }
    function = action.get("function")
    if function is not None and not isinstance(function, Mapping):
        return {"tool": "__invalid_action__", "arguments": {"error_code": "schema_error"}}
    function = function if isinstance(function, Mapping) else {}
    tool = action.get("tool") or action.get("name") or action.get("tool_name") or function.get("name")
    arguments = action.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"tool": "__invalid_action__", "arguments": {"error_code": "argument_decode_error"}}
    if not isinstance(arguments, Mapping):
        return {"tool": "__invalid_action__", "arguments": {"error_code": "schema_error"}}
    if not tool:
        return {"tool": "__invalid_action__", "arguments": {"error_code": "schema_error"}}
    return {
        "tool": str(tool) if tool else "__invalid_action__",
        "arguments": copy.deepcopy(dict(arguments)),
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def action_key(action: Any) -> str:
    """Canonical action key used for deduplication and deterministic ties."""

    return _stable_json(_normalize_action(action))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number


def _count(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return float(len(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(len(value))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return max(number, 0.0)


def _source_values(*sources: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    expanded: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    nested_keys = (
        "last_verification",
        "verification_output",
        "last_tool_output",
        "tool_metrics",
        "candidate_assessment",
        "counterfactual_verification",
        "metadata",
    )
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        queue: list[tuple[Mapping[str, Any], int]] = [(source, 0)]
        while queue:
            current, depth = queue.pop(0)
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            expanded.append(current)
            if depth >= 3:
                continue
            for key in nested_keys:
                nested = current.get(key)
                if isinstance(nested, Mapping):
                    queue.append((nested, depth + 1))
    return expanded


def _first(sources: Sequence[Mapping[str, Any]], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "success", "resolved"}:
            return True
        if lowered in {"false", "no", "0", "failure", "unresolved"}:
            return False
        return None
    return bool(value)


def _call_with_context(
    function: Callable[..., Any],
    context: Mapping[str, Any],
    positional_fallback: Sequence[Any],
) -> Any:
    """Call an injected hook using named context when its signature permits."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*positional_fallback)

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return function(**dict(context))

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    fallback_index = 0
    aliases = {
        "env": "branch",
        "branch_env": "branch",
        "state": "next_state",
        "observation": "next_state",
        "output": "tool_output",
        "result": "tool_output",
    }
    for parameter in parameters:
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        name = parameter.name
        context_name = name if name in context else aliases.get(name)
        if context_name in context:
            value = context[context_name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        elif fallback_index < len(positional_fallback):
            value = positional_fallback[fallback_index]
            fallback_index += 1
        else:
            raise TypeError(f"Cannot supply required hook parameter {name!r}.")
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    return function(*args, **kwargs)


@dataclass(frozen=True)
class CostWeights:
    """Weights for the six roadmap cost-to-go components."""

    tool: float = 1.0
    remaining_faults: float = 2.0
    healthy_corruption: float = 25.0
    false_commit: float = 50.0
    false_finalization: float = 100.0
    expert_recovery: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _nonnegative(value, f"Cost weight {name}"))


@dataclass(frozen=True)
class BranchCostBreakdown:
    """Weighted terms whose sum is the branch's Q cost."""

    tool_cost: float
    remaining_faults_cost: float
    healthy_corruption_cost: float
    false_commit_cost: float
    false_finalization_cost: float
    expert_recovery_cost: float

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def as_dict(self) -> dict[str, float]:
        result = {key: float(value) for key, value in asdict(self).items()}
        result["total"] = self.total
        return result


@dataclass(frozen=True)
class BranchEvaluation:
    """Result of executing one candidate action on an isolated branch."""

    action: dict[str, Any]
    q_cost: float
    cost_breakdown: BranchCostBreakdown
    raw_cost_components: dict[str, float]
    next_state: dict[str, Any]
    tool_output: dict[str, Any]
    branch_error: str | None = None

    def as_record(self, *, include_branch_result: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "action": copy.deepcopy(self.action),
            "q_cost": float(self.q_cost),
            "cost_breakdown": self.cost_breakdown.as_dict(),
            "raw_cost_components": dict(self.raw_cost_components),
        }
        if include_branch_result:
            record["next_state"] = copy.deepcopy(self.next_state)
            record["tool_output"] = copy.deepcopy(self.tool_output)
        if self.branch_error is not None:
            record["branch_error"] = self.branch_error
        return record


@runtime_checkable
class ActionRanker(Protocol):
    """Interface for a separate, inspectable action ranker."""

    def rank(self, evaluations: Sequence[BranchEvaluation]) -> list[BranchEvaluation]:
        ...


class CostBasedActionRanker:
    """Rank by Q cost, with canonical action JSON as a stable tie-breaker."""

    def rank(self, evaluations: Sequence[BranchEvaluation]) -> list[BranchEvaluation]:
        return sorted(evaluations, key=lambda item: (item.q_cost, action_key(item.action)))

    def best_action(self, evaluations: Sequence[BranchEvaluation]) -> dict[str, Any] | None:
        ranked = self.rank(evaluations)
        return copy.deepcopy(ranked[0].action) if ranked else None


class BranchIsolationError(RuntimeError):
    """Raised when a branch operation mutates or reuses the root environment."""


class TopLBranchEvaluator:
    """Execute and cost a bounded candidate-action set on cloned environments."""

    def __init__(
        self,
        *,
        env: Any,
        expert_recovery_cost: Callable[..., Any] | None = None,
        recovery_cost_fn: Callable[..., Any] | None = None,
        cost_weights: CostWeights | None = None,
        clone_fn: Callable[..., Any] | None = None,
        tool_cost_fn: Callable[..., Any] | None = None,
        failure_recovery_cost: float = 10.0,
        unknown_remaining_fault_cost: float = 1.0,
        seed: int | None = 0,
        verify_root_unchanged: bool = True,
    ) -> None:
        if expert_recovery_cost is not None and recovery_cost_fn is not None:
            raise ValueError("Pass only one of expert_recovery_cost and recovery_cost_fn.")
        self.env = env
        self.recovery_cost_fn = expert_recovery_cost or recovery_cost_fn
        self.cost_weights = cost_weights or CostWeights()
        self.clone_fn = clone_fn
        self.tool_cost_fn = tool_cost_fn
        self.failure_recovery_cost = _nonnegative(failure_recovery_cost, "failure_recovery_cost")
        self.unknown_remaining_fault_cost = _nonnegative(
            unknown_remaining_fault_cost, "unknown_remaining_fault_cost"
        )
        self.seed = seed
        self.verify_root_unchanged = bool(verify_root_unchanged)

    def _current_state(self, env: Any) -> dict[str, Any]:
        method = getattr(env, "current_state", None)
        if callable(method):
            value = method()
            return copy.deepcopy(_mapping(value))
        state = getattr(env, "state", None)
        return copy.deepcopy(_mapping(state))

    def _root_snapshot(self) -> str:
        payload: dict[str, Any] = {"current_state": self._current_state(self.env)}
        for name in ("store", "state", "history", "context_flags", "rng", "terminal", "current_candidate_id"):
            if not hasattr(self.env, name):
                continue
            value = getattr(self.env, name)
            if callable(getattr(value, "getstate", None)):
                payload[name] = value.getstate()
            elif hasattr(value, "__dict__"):
                payload[name] = vars(value)
            else:
                payload[name] = value
        return _stable_json(payload)

    def _assert_branch_isolated(self, branch: Any) -> None:
        for name in ("store", "state", "history", "context_flags", "rng"):
            root_value = getattr(self.env, name, None)
            branch_value = getattr(branch, name, None)
            if root_value is not None and root_value is branch_value:
                raise BranchIsolationError(f"Environment clone shares mutable attribute {name!r} with root.")

    def _clone(self, state: Mapping[str, Any]) -> Any:
        current_state = self._current_state(self.env)
        if self.clone_fn is not None:
            branch = _call_with_context(
                self.clone_fn,
                {"env": self.env, "branch": self.env, "state": state, "next_state": state},
                (self.env, state),
            )
        elif callable(getattr(self.env, "clone_from", None)):
            clone_from = self.env.clone_from
            try:
                parameters = list(inspect.signature(clone_from).parameters.values())
            except (TypeError, ValueError):
                parameters = []
            first_parameter = parameters[0].name if parameters else "state"
            if first_parameter.endswith("_id") or first_parameter in {"state_id", "id"}:
                if _stable_json(state) != _stable_json(current_state):
                    raise BranchIsolationError(
                        "clone_from(state_id) cannot install a non-current state snapshot."
                    )
                reference = state.get("candidate_state_id") or state.get("active_state_id")
                branch = clone_from(str(reference) if reference is not None else None)
            else:
                branch = clone_from(copy.deepcopy(dict(state)))
        elif callable(getattr(self.env, "clone", None)):
            if _stable_json(state) != _stable_json(current_state):
                raise BranchIsolationError("env.clone() can evaluate only the root's current state.")
            clone_method = self.env.clone
            try:
                signature = inspect.signature(clone_method)
                required = [
                    parameter
                    for parameter in signature.parameters.values()
                    if parameter.default is inspect.Parameter.empty
                    and parameter.kind
                    in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
                ]
            except (TypeError, ValueError):
                required = []
            branch = clone_method(copy.deepcopy(dict(state))) if required else clone_method()
        else:
            if _stable_json(state) != _stable_json(current_state):
                raise BranchIsolationError("deepcopy fallback can evaluate only the root's current state.")
            branch = copy.deepcopy(self.env)

        if isinstance(branch, tuple):
            branch = branch[0] if branch else None
        if branch is None:
            raise BranchIsolationError("Environment clone returned None.")
        if branch is self.env:
            raise BranchIsolationError("Environment clone returned the root object.")
        self._assert_branch_isolated(branch)
        return branch

    def _seed_branch(self, branch: Any, state: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        if self.seed is None:
            return
        digest = hashlib.sha256(
            f"{self.seed}:{_stable_json(state)}:{action_key(action)}".encode("utf-8")
        ).hexdigest()
        branch_seed = int(digest[:16], 16)
        rng = getattr(branch, "rng", None)
        if rng is not None and callable(getattr(rng, "seed", None)):
            rng.seed(branch_seed)
            return
        for method_name in ("set_seed", "seed"):
            method = getattr(branch, method_name, None)
            if callable(method):
                method(branch_seed)
                return

    def _step_branch(self, branch: Any, action: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result = branch.step(copy.deepcopy(dict(action)))
        if isinstance(result, tuple):
            if len(result) == 2:
                next_state, output = result
            elif len(result) >= 5:  # Gymnasium: obs, reward, terminated, truncated, info
                next_state, output = result[0], result[4]
            elif result:
                next_state, output = result[0], {}
            else:
                next_state, output = {}, {}
        elif isinstance(result, Mapping):
            next_state, output = result, {}
        else:
            next_state, output = {}, {}
        normalized_state = _mapping(next_state) or self._current_state(branch)
        normalized_output = _mapping(output)
        if (
            action.get("tool") in {"correct_measurements", "correct_parameters", "correct_topology"}
            and normalized_output.get("execution_status") == "success"
            and normalized_state.get("has_unverified_candidate")
            and normalized_state.get("candidate_state_id")
        ):
            verification_action = {
                "tool": "run_wls",
                "arguments": {"state_id": normalized_state["candidate_state_id"]},
            }
            verified_state, verification_output = self._step_branch_once(branch, verification_action)
            normalized_state = verified_state
            normalized_output = dict(normalized_output)
            normalized_output["counterfactual_verification"] = verification_output
        return normalized_state, normalized_output

    def _step_branch_once(
        self, branch: Any, action: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result = branch.step(copy.deepcopy(dict(action)))
        if isinstance(result, tuple):
            next_state = result[0] if result else {}
            output = result[1] if len(result) > 1 else {}
        elif isinstance(result, Mapping):
            next_state, output = result, {}
        else:
            next_state, output = {}, {}
        return _mapping(next_state) or self._current_state(branch), _mapping(output)

    def _tool_cost(
        self,
        *,
        branch: Any,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        next_state: Mapping[str, Any],
        tool_output: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> float:
        if self.tool_cost_fn is None:
            if "counterfactual_verification" in tool_output:
                return 2.0
            return 2.0 if action.get("tool") in {"run_wls", "verify_candidate"} else 1.0
        value = _call_with_context(
            self.tool_cost_fn,
            {
                "branch": branch,
                "state": state,
                "next_state": next_state,
                "action": action,
                "tool_output": tool_output,
                "history": history,
            },
            (action, tool_output),
        )
        return _nonnegative(value, "tool_cost_fn result")

    def _expert_recovery(
        self,
        *,
        branch: Any,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        next_state: Mapping[str, Any],
        tool_output: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        branch_error: str | None,
    ) -> float:
        function = self.recovery_cost_fn
        if function is None:
            oracle = getattr(branch, "oracle", None)
            function = getattr(oracle, "expert_recovery_cost", None)
        if function is not None:
            value = _call_with_context(
                function,
                {
                    "branch": branch,
                    "parent_state": state,
                    "state": next_state,
                    "next_state": next_state,
                    "action": action,
                    "tool_output": tool_output,
                    "history": history,
                },
                (branch, next_state, action, tool_output, history),
            )
            if isinstance(value, Mapping):
                value = (
                    value.get("expert_recovery_cost")
                    if value.get("expert_recovery_cost") is not None
                    else value.get("cost", value.get("estimated_remaining_steps", 0.0))
                )
            recovery = _nonnegative(value, "expert recovery cost")
        else:
            sources = _source_values(
                self._privileged_branch_evidence(branch, next_state), tool_output, next_state
            )
            estimate = _first(sources, "estimated_remaining_steps", "recovery_steps")
            if estimate is not None:
                recovery = max(_finite(estimate), 0.0)
            elif next_state.get("has_verified_candidate"):
                # At least one disposition action remains after verification.
                recovery = 1.0
            else:
                recovery = 0.0
        if branch_error is not None or tool_output.get("execution_status") == "failure":
            recovery = max(recovery, self.failure_recovery_cost)
        return recovery

    def _raw_components(
        self,
        *,
        branch: Any,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
        next_state: Mapping[str, Any],
        tool_output: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        branch_error: str | None,
    ) -> dict[str, float]:
        sources = _source_values(
            self._privileged_branch_evidence(branch, next_state), tool_output, next_state, state
        )
        remaining = _count(
            _first(sources, "remaining_true_fault_count", "remaining_true_faults")
        )
        if remaining is None:
            unresolved_count = _count(_first(sources, "unresolved_signatures"))
            # A nonempty unresolved set is a valid lower bound.  An empty set
            # is not proof that no hidden faults remain.
            remaining = unresolved_count if unresolved_count and unresolved_count > 0.0 else None
        remaining_known = remaining is not None
        if remaining is None:
            remaining = self.unknown_remaining_fault_cost
        healthy_corruption = float(
            _optional_bool(_first(sources, "healthy_component_modified", "collateral_damage"))
            is True
        )

        tool = str(action.get("tool", ""))
        disposition = _first(
            _source_values(state),
            "candidate_disposition",
        )
        false_commit = float(
            tool == "commit_state"
            and (
                disposition not in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}
                or tool_output.get("execution_status") == "failure"
            )
        )

        false_finalization = 0.0
        if tool == "finalize_diagnosis":
            score = _finite(_first(sources, "remaining_anomaly_score"), default=float("nan"))
            threshold = _finite(
                _first(sources, "anomaly_threshold", "chi_square_threshold"),
                default=float("nan"),
            )
            resolved = _optional_bool(
                _first(sources, "post_action_resolved", "no_material_anomaly_remaining")
            )
            unresolved_evidence = (
                (remaining_known and remaining > 0.0)
                or (math.isfinite(score) and math.isfinite(threshold) and score >= threshold)
                or resolved is False
                or tool_output.get("execution_status") == "failure"
            )
            resolution_known = (
                resolved is not None
                or (math.isfinite(score) and math.isfinite(threshold))
                or remaining_known
            )
            # Unknown global status is unsafe for finalization.
            false_finalization = float(unresolved_evidence or not resolution_known)

        tool_cost = self._tool_cost(
            branch=branch,
            state=state,
            action=action,
            next_state=next_state,
            tool_output=tool_output,
            history=history,
        )
        recovery = self._expert_recovery(
            branch=branch,
            state=state,
            action=action,
            next_state=next_state,
            tool_output=tool_output,
            history=history,
            branch_error=branch_error,
        )
        return {
            "tool": tool_cost,
            "remaining_faults": remaining,
            "remaining_faults_known": float(remaining_known),
            "healthy_corruption": healthy_corruption,
            "false_commit": false_commit,
            "false_finalization": false_finalization,
            "expert_recovery": recovery,
        }

    @staticmethod
    def _privileged_branch_evidence(branch: Any, next_state: Mapping[str, Any]) -> dict[str, Any]:
        """Return oracle-only branch labels for Q-cost construction, never policy input."""
        evidence: dict[str, Any] = {}
        oracle_payload = getattr(branch, "_oracle_payload", None)
        if isinstance(oracle_payload, Mapping):
            if oracle_payload.get("oracle_terminal_eligible"):
                evidence["no_material_anomaly_remaining"] = True
            remaining = oracle_payload.get(
                "remaining_true_fault_count", oracle_payload.get("remaining_true_faults")
            )
            if remaining is not None:
                evidence["remaining_true_fault_count"] = (
                    len(remaining) if isinstance(remaining, (list, tuple, set)) else remaining
                )
        candidate_id = next_state.get("candidate_state_id")
        store = getattr(branch, "store", None)
        if candidate_id and store is not None and callable(getattr(store, "get_state", None)):
            try:
                candidate = store.get_state(str(candidate_id))
            except Exception:
                candidate = {}
            if isinstance(candidate, Mapping):
                assessment = candidate.get("candidate_assessment")
                if isinstance(assessment, Mapping):
                    evidence["candidate_assessment"] = dict(assessment)
                if candidate.get("candidate_disposition") is not None:
                    evidence["candidate_disposition"] = candidate.get("candidate_disposition")
        return evidence

    def evaluate_action(
        self,
        *,
        state: Mapping[str, Any],
        action: Mapping[str, Any] | str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> BranchEvaluation:
        normalized_action = _normalize_action(action)
        materialized_state = copy.deepcopy(dict(state))
        materialized_history = [copy.deepcopy(dict(item)) for item in (history or [])]
        root_before = self._root_snapshot()
        branch: Any = None
        branch_error: str | None = None
        next_state: dict[str, Any] = {}
        tool_output: dict[str, Any] = {}
        try:
            branch = self._clone(materialized_state)
            self._seed_branch(branch, materialized_state, normalized_action)
            next_state, tool_output = self._step_branch(branch, normalized_action)
        except Exception as exc:  # A failed candidate remains rankable, never mutates the root.
            branch_error = f"{type(exc).__name__}: {exc}"
            tool_output = {
                "execution_status": "failure",
                "error": "branch_evaluation_error",
                "error_detail": branch_error,
            }
            if branch is not None:
                next_state = self._current_state(branch)

        try:
            raw = self._raw_components(
                branch=branch,
                state=materialized_state,
                action=normalized_action,
                next_state=next_state,
                tool_output=tool_output,
                history=materialized_history,
                branch_error=branch_error,
            )
        except Exception as exc:
            hook_error = f"{type(exc).__name__}: {exc}"
            branch_error = f"{branch_error}; cost_hook_error: {hook_error}" if branch_error else f"cost_hook_error: {hook_error}"
            tool = normalized_action.get("tool")
            raw = {
                "tool": 1.0,
                "remaining_faults": self.unknown_remaining_fault_cost,
                "remaining_faults_known": 0.0,
                "healthy_corruption": 0.0,
                "false_commit": float(tool == "commit_state"),
                "false_finalization": float(tool == "finalize_diagnosis"),
                "expert_recovery": self.failure_recovery_cost,
            }

        root_after = self._root_snapshot()
        if self.verify_root_unchanged and root_before != root_after:
            raise BranchIsolationError("Counterfactual branch or cost hook mutated the root environment.")
        weights = self.cost_weights
        breakdown = BranchCostBreakdown(
            tool_cost=weights.tool * raw["tool"],
            remaining_faults_cost=weights.remaining_faults * raw["remaining_faults"],
            healthy_corruption_cost=weights.healthy_corruption * raw["healthy_corruption"],
            false_commit_cost=weights.false_commit * raw["false_commit"],
            false_finalization_cost=weights.false_finalization * raw["false_finalization"],
            expert_recovery_cost=weights.expert_recovery * raw["expert_recovery"],
        )
        return BranchEvaluation(
            action=normalized_action,
            q_cost=breakdown.total,
            cost_breakdown=breakdown,
            raw_cost_components=raw,
            next_state=next_state,
            tool_output=tool_output,
            branch_error=branch_error,
        )

    def evaluate_actions(
        self,
        *,
        state: Mapping[str, Any],
        actions: Iterable[Mapping[str, Any] | str],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[BranchEvaluation]:
        return [
            self.evaluate_action(state=state, action=action, history=history)
            for action in actions
        ]


class AggreVaTeLite:
    """Enumerate top-L oracle actions, branch them, and store ranked Q costs."""

    def __init__(
        self,
        *,
        env: Any,
        oracle: Any,
        top_l: int = 8,
        near_optimal_tolerance: float = 0.0,
        near_optimal_relative_tolerance: float = 0.0,
        action_ranker: ActionRanker | None = None,
        branch_evaluator: TopLBranchEvaluator | None = None,
        **evaluator_options: Any,
    ) -> None:
        if top_l < 1:
            raise ValueError("top_l must be at least one.")
        near_optimal_tolerance = _nonnegative(near_optimal_tolerance, "near_optimal_tolerance")
        near_optimal_relative_tolerance = _nonnegative(
            near_optimal_relative_tolerance, "near_optimal_relative_tolerance"
        )
        if branch_evaluator is not None and evaluator_options:
            raise ValueError("Evaluator options cannot be used with branch_evaluator.")
        self.env = env
        self.oracle = oracle
        self.top_l = int(top_l)
        self.near_optimal_tolerance = float(near_optimal_tolerance)
        self.near_optimal_relative_tolerance = float(near_optimal_relative_tolerance)
        self.action_ranker = action_ranker or CostBasedActionRanker()
        if branch_evaluator is None:
            recovery = getattr(oracle, "expert_recovery_cost", None)
            if (
                callable(recovery)
                and "expert_recovery_cost" not in evaluator_options
                and "recovery_cost_fn" not in evaluator_options
            ):
                evaluator_options["expert_recovery_cost"] = recovery
        self.branch_evaluator = branch_evaluator or TopLBranchEvaluator(env=env, **evaluator_options)

    def _oracle_actions(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        top_l: int,
    ) -> Iterable[Any]:
        for method_name in ("enumerate_top_actions", "enumerate_admissible_actions", "next_actions"):
            method = getattr(self.oracle, method_name, None)
            if not callable(method):
                continue
            return _call_with_context(
                method,
                {
                    "state": copy.deepcopy(dict(state)),
                    "history": [copy.deepcopy(dict(item)) for item in history],
                    "top_l": top_l,
                },
                (copy.deepcopy(dict(state)),),
            )
        fallback = getattr(self.env, "enumerate_available_actions", None)
        if callable(fallback):
            return _call_with_context(
                fallback,
                {"state": copy.deepcopy(dict(state))},
                (copy.deepcopy(dict(state)),),
            )
        raise TypeError(
            "oracle must expose enumerate_top_actions, enumerate_admissible_actions, or next_actions."
        )

    def enumerate_top_actions(
        self,
        state: Mapping[str, Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        top_l: int | None = None,
    ) -> list[dict[str, Any]]:
        limit = self.top_l if top_l is None else int(top_l)
        if limit < 1:
            raise ValueError("top_l must be at least one.")
        materialized_state = copy.deepcopy(dict(state))
        materialized_history = [copy.deepcopy(dict(item)) for item in (history or [])]
        raw_actions = self._oracle_actions(materialized_state, materialized_history, limit)
        if raw_actions is None:
            return []
        if isinstance(raw_actions, (Mapping, str)):
            raw_actions = [raw_actions]
        elif not isinstance(raw_actions, Iterable):
            raise TypeError("Oracle action enumeration must return an iterable of actions.")
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_action in raw_actions:
            action = _normalize_action(raw_action)
            key = action_key(action)
            if key in seen:
                continue
            seen.add(key)
            result.append(action)
            if len(result) >= limit:
                break
        return result

    def rank_actions(
        self,
        state: Mapping[str, Any],
        *,
        candidate_actions: Iterable[Mapping[str, Any] | str] | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
        top_l: int | None = None,
        include_branch_results: bool = True,
    ) -> dict[str, Any]:
        limit = self.top_l if top_l is None else int(top_l)
        if limit < 1:
            raise ValueError("top_l must be at least one.")
        if candidate_actions is None:
            actions = self.enumerate_top_actions(state, history=history, top_l=limit)
        else:
            actions = []
            seen: set[str] = set()
            action_source: Iterable[Mapping[str, Any] | str]
            if isinstance(candidate_actions, (Mapping, str)):
                action_source = [candidate_actions]
            else:
                action_source = candidate_actions
            for raw_action in action_source:
                action = _normalize_action(raw_action)
                key = action_key(action)
                if key not in seen:
                    actions.append(action)
                    seen.add(key)
                if len(actions) >= limit:
                    break

        evaluations = self.branch_evaluator.evaluate_actions(
            state=state,
            actions=actions,
            history=history,
        )
        ranked = self.action_ranker.rank(copy.deepcopy(evaluations))
        if ranked:
            best_cost = ranked[0].q_cost
            tolerance = max(
                self.near_optimal_tolerance,
                abs(best_cost) * self.near_optimal_relative_tolerance,
            )
            near_optimal = [
                copy.deepcopy(item.action)
                for item in ranked
                if item.q_cost <= best_cost + tolerance + 1e-12
            ]
        else:
            best_cost = None
            tolerance = 0.0
            near_optimal = []
        near_optimal_keys = {action_key(action) for action in near_optimal}
        action_costs = []
        for item in ranked:
            record = item.as_record(include_branch_result=include_branch_results)
            record["near_optimal"] = action_key(item.action) in near_optimal_keys
            record["cost_margin_from_best"] = (
                float(item.q_cost - best_cost) if best_cost is not None else None
            )
            action_costs.append(record)
        return {
            "state": copy.deepcopy(dict(state)),
            "action_costs": action_costs,
            "near_optimal_actions": near_optimal,
            "best_q_cost": float(best_cost) if best_cost is not None else None,
            "near_optimal_cost_tolerance": float(tolerance),
            # Be explicit about what this implementation currently estimates:
            # one candidate action (plus automatic verification for a
            # correction) followed by an injected or heuristic recovery term.
            # It is not a terminal expert rollout unless the injected recovery
            # hook itself performs that rollout.
            "cost_evaluation": {
                "candidate_set": "bounded_oracle_top_l",
                "full_expert_rollout": False,
                "automatic_correction_verification": True,
                "recovery_term": (
                    "injected_callback"
                    if getattr(self.branch_evaluator, "recovery_cost_fn", None) is not None
                    else "one_step_heuristic"
                ),
            },
        }

    collect_ranking_example = rank_actions


def evaluate_top_l_actions(
    *,
    env: Any,
    oracle: Any,
    state: Mapping[str, Any],
    top_l: int = 8,
    history: Sequence[Mapping[str, Any]] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Functional convenience wrapper around :class:`AggreVaTeLite`."""

    return AggreVaTeLite(env=env, oracle=oracle, top_l=top_l, **options).rank_actions(
        state,
        history=history,
    )


def to_pairwise_examples(ranking_example: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert one ranking record to unambiguous chosen/rejected pairs.

    Actions within the recorded near-optimal tolerance are multi-positive: they
    are never emitted as a rejected action against the deterministic rank-one
    choice.  Ambiguous all-near-optimal states therefore produce no pair rather
    than a contradictory preference target.
    """

    costs = ranking_example.get("action_costs")
    if not isinstance(costs, Sequence) or len(costs) < 2:
        return []
    valid: list[dict[str, Any]] = []
    for item in costs:
        if not isinstance(item, Mapping) or item.get("action") is None:
            continue
        try:
            q_cost = float(item.get("q_cost"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(q_cost):
            continue
        copied = dict(item)
        copied["q_cost"] = q_cost
        valid.append(copied)
    valid.sort(key=lambda item: (item["q_cost"], action_key(item["action"])))
    if len(valid) < 2:
        return []
    chosen = valid[0]
    near_optimal_source = ranking_example.get("near_optimal_actions") or []
    if isinstance(near_optimal_source, (Mapping, str)):
        near_optimal_source = [near_optimal_source]
    near_optimal_keys = {
        action_key(action)
        for action in near_optimal_source
        if isinstance(action, (Mapping, str))
    }
    near_optimal_keys.update(
        action_key(item["action"])
        for item in valid
        if item.get("near_optimal") is True
    )
    if not near_optimal_keys:
        tolerance = _finite(
            ranking_example.get("near_optimal_cost_tolerance"), default=0.0
        )
        tolerance = max(tolerance, 0.0)
        near_optimal_keys.update(
            action_key(item["action"])
            for item in valid
            if item["q_cost"] <= chosen["q_cost"] + tolerance + 1e-12
        )
    near_optimal_keys.add(action_key(chosen["action"]))
    pairs: list[dict[str, Any]] = []
    for rejected in valid[1:]:
        if action_key(rejected["action"]) in near_optimal_keys:
            continue
        pairs.append(
            {
                "state": copy.deepcopy(ranking_example.get("state", {})),
                "chosen": copy.deepcopy(chosen["action"]),
                "rejected": copy.deepcopy(rejected["action"]),
                "chosen_q_cost": chosen["q_cost"],
                "rejected_q_cost": rejected["q_cost"],
                "cost_margin": rejected["q_cost"] - chosen["q_cost"],
                "near_optimal_cost_tolerance": ranking_example.get(
                    "near_optimal_cost_tolerance", 0.0
                ),
            }
        )
    return pairs


__all__ = [
    "ActionRanker",
    "AggreVaTeLite",
    "BranchCostBreakdown",
    "BranchEvaluation",
    "BranchIsolationError",
    "CostBasedActionRanker",
    "CostWeights",
    "TopLBranchEvaluator",
    "action_key",
    "evaluate_top_l_actions",
    "to_pairwise_examples",
]
