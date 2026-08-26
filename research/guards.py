"""Inference guards, evaluated as an explicit separate policy arm.

A guarded policy is a different policy: results measured with guards must be
reported as "policy + guards", never merged with unguarded numbers, and any
DAgger collection intended for a guarded deployment should itself run under
the guarded policy.  Measured effect on the frozen development suite: the
silence retry rescued ~1 of 10 firings; the stale-reference guard never fired
(the 12B lock is a commit-discipline failure on a valid reference, not stale
addressing).  Both failure modes are in-principle learnable from retained
failure states with an identifiable observation.

Two measured pathologies motivate this module (traced on the frozen
65-scenario evaluation):

* E2B goes *silent* in deep states -- generation produces zero tool calls
  (9/9 Round-1 failures).  Greedy decoding reproduces the silence
  deterministically, so the guard retries once with sampling enabled.
* 12B *locks onto stale references* -- it keeps addressing a candidate state
  that has already closed (8/12 Round-1 failures, 85% of failed steps).  The
  correct references are printed in the observation every step, so the guard
  checks the emitted action against them and retries once when stale.

Both guards retry at most once and then pass the original result through,
including exceptions: a guard that silently substitutes actions would turn an
honest failure into a hidden one.  Every intervention is recorded so an
evaluation can report exactly how often the guards fired and whether the
retry helped.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Iterator, Mapping

STATE_REFERENCE_KEYS = ("state_id", "candidate_state_id")
_SILENCE_MARKERS = ("exactly one tool call", "found 0")


def _is_silence(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _SILENCE_MARKERS)


def _references(action: Mapping[str, Any]) -> set[str]:
    arguments = action.get("arguments") or {}
    return {
        str(value)
        for key, value in arguments.items()
        if key in STATE_REFERENCE_KEYS and isinstance(value, str) and value
    }


def _valid_targets(observation: Mapping[str, Any]) -> set[str]:
    valid = set()
    active = observation.get("active_state_id")
    if isinstance(active, str) and active:
        valid.add(active)
    if observation.get("has_open_candidate"):
        candidate = observation.get("candidate_state_id")
        if isinstance(candidate, str) and candidate:
            valid.add(candidate)
    return valid


class GuardedPolicy:
    """Wrap a policy with silence-retry and stale-reference-retry guards."""

    def __init__(
        self,
        inner: Any,
        bundle: Any,
        *,
        retry_temperature: float = 0.7,
        retry_top_p: float = 0.9,
    ) -> None:
        self._inner = inner
        self._bundle = bundle
        self._retry_temperature = float(retry_temperature)
        self._retry_top_p = float(retry_top_p)
        self.guard_events: list[dict[str, Any]] = []

    def pop_events(self) -> list[dict[str, Any]]:
        events, self.guard_events = self.guard_events, []
        return events

    @contextlib.contextmanager
    def _sampled_generation(self) -> Iterator[None]:
        model = self._bundle.model
        original = model.generate

        @functools.wraps(original)
        def sampled(*args: Any, **kwargs: Any) -> Any:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self._retry_temperature
            kwargs["top_p"] = self._retry_top_p
            return original(*args, **kwargs)

        model.generate = sampled
        try:
            yield
        finally:
            model.generate = original

    def act(self, observation: Any) -> dict[str, Any]:
        if hasattr(observation, "as_dict"):
            observation = observation.as_dict()

        try:
            action = self._inner.act(observation)
        except Exception as exc:  # noqa: BLE001 - guard on the specific failure
            if not _is_silence(exc):
                raise
            self.guard_events.append(
                {"guard": "silence_retry", "error": str(exc)[:120]}
            )
            with self._sampled_generation():
                action = self._inner.act(observation)

        references = _references(action)
        valid = _valid_targets(observation)
        if references and valid and not references <= valid:
            event = {
                "guard": "stale_reference_retry",
                "tool": action.get("tool"),
                "stale": sorted(references - valid),
                "valid": sorted(valid),
            }
            with self._sampled_generation():
                try:
                    retry = self._inner.act(observation)
                except Exception as exc:  # noqa: BLE001 - keep the original action
                    event["retry"] = f"raised: {str(exc)[:80]}"
                    self.guard_events.append(event)
                    return action
            retry_references = _references(retry)
            if not retry_references or retry_references <= valid:
                event["retry"] = "accepted"
                action = retry
            else:
                event["retry"] = "still_stale_kept_original"
            self.guard_events.append(event)
        return action

    @property
    def release_policy_identity(self) -> Any:
        return getattr(self._inner, "release_policy_identity", None)
