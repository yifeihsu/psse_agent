"""Mock-based tests for the inference guards (no GPU, no model)."""

from __future__ import annotations

from psse_env.sft.gates import GateError

from research.guards import GuardedPolicy


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, *args, **kwargs):  # patched by the guard's context manager
        self.calls.append(kwargs)
        return "output"


class FakeBundle:
    def __init__(self) -> None:
        self.model = FakeModel()


class ScriptedPolicy:
    """Returns queued results; raises queued exceptions; records sampling."""

    def __init__(self, bundle, script) -> None:
        self._bundle = bundle
        self._script = list(script)
        self.sampled_calls = []

    def act(self, observation):
        self._bundle.model.generate()
        sampled = bool(self._bundle.model.calls[-1].get("do_sample"))
        self.sampled_calls.append(sampled)
        result = self._script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


OBS = {
    "active_state_id": "ep:s0",
    "candidate_state_id": "ep:s2",
    "has_open_candidate": False,
}


def test_silence_retry_samples_and_recovers():
    bundle = FakeBundle()
    good = {"tool": "run_wls", "arguments": {"state_id": "ep:s0"}}
    inner = ScriptedPolicy(
        bundle,
        [GateError("Generated output must contain exactly one tool call; found 0"), good],
    )
    guard = GuardedPolicy(inner, bundle)
    assert guard.act(OBS) == good
    assert inner.sampled_calls == [False, True], "retry must enable sampling"
    assert guard.pop_events()[0]["guard"] == "silence_retry"


def test_stale_reference_retry_accepted():
    bundle = FakeBundle()
    stale = {"tool": "rollback_state", "arguments": {"candidate_state_id": "ep:s2"}}
    fresh = {"tool": "run_wls", "arguments": {"state_id": "ep:s0"}}
    inner = ScriptedPolicy(bundle, [stale, fresh])
    guard = GuardedPolicy(inner, bundle)
    assert guard.act(OBS) == fresh
    events = guard.pop_events()
    assert events[0]["guard"] == "stale_reference_retry"
    assert events[0]["retry"] == "accepted"
    assert events[0]["stale"] == ["ep:s2"]


def test_stale_retry_still_stale_keeps_original():
    bundle = FakeBundle()
    stale = {"tool": "rollback_state", "arguments": {"candidate_state_id": "ep:s2"}}
    inner = ScriptedPolicy(bundle, [stale, dict(stale)])
    guard = GuardedPolicy(inner, bundle)
    assert guard.act(OBS) == stale, "guard must not fabricate an action"
    assert guard.pop_events()[0]["retry"] == "still_stale_kept_original"


def test_open_candidate_reference_is_valid():
    bundle = FakeBundle()
    action = {"tool": "commit_state", "arguments": {"candidate_state_id": "ep:s2"}}
    inner = ScriptedPolicy(bundle, [action])
    guard = GuardedPolicy(inner, bundle)
    observation = dict(OBS, has_open_candidate=True)
    assert guard.act(observation) == action
    assert guard.pop_events() == [], "no guard should fire on a valid reference"


def test_unrelated_exception_propagates():
    bundle = FakeBundle()
    inner = ScriptedPolicy(bundle, [RuntimeError("cuda out of memory")])
    guard = GuardedPolicy(inner, bundle)
    try:
        guard.act(OBS)
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-silence exceptions must propagate")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
