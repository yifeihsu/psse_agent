from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT


@pytest.mark.parametrize("module_name, option, field", [
    ("interactive_agent_eval", "--max-steps", "max_steps"),
    ("eval_sft_agent_gemma_v4", "--max-turns", "max_turns"),
    ("eval_sft_agent_hardened", "--max-turns", "max_turns"),
])
def test_legacy_cli_defaults_and_explicit_small_limits(module_name, option, field, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name])
    assert getattr(module.parse_args(), field) == DEFAULT_EPISODE_ACTION_LIMIT == 40
    monkeypatch.setattr(sys, "argv", [module_name, option, "3"])
    assert getattr(module.parse_args(), field) == 3
    monkeypatch.setattr(sys, "argv", [module_name, option, "0"])
    with pytest.raises(SystemExit):
        module.parse_args()


@pytest.fixture
def stubbed_gemma(monkeypatch):
    import torch
    import eval_sft_agent_gemma_v4 as evaluator

    class Model:
        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.full((input_ids.shape[0], 1), 2, dtype=torch.long)], dim=1)

    calls = []
    monkeypatch.setattr(evaluator, "build_model_inputs", lambda *args, **kwargs: ({"input_ids": torch.tensor([[1]])}, False))
    monkeypatch.setattr(evaluator, "get_stop_token_ids", lambda *_: [])
    monkeypatch.setattr(evaluator, "infer_expected_phase", lambda *_: "tool")
    monkeypatch.setattr(evaluator, "decode_generated_response", lambda *args, **kwargs: ("tool", 1, 0))
    monkeypatch.setattr(evaluator, "parse_gemma_generation", lambda *args: {
        "type": "tool_call", "name": "wls_from_path", "arguments": {}})
    monkeypatch.setattr(evaluator, "summarize_tool_result_for_conversation", lambda name, result, *args: result)

    def execute(name, arguments, **kwargs):
        calls.append(kwargs.get("hidden_context", {}).get("sample_id", "single"))
        return {"success": False, "error": "deliberate recoverable tool failure"}

    monkeypatch.setattr(evaluator, "execute_tool", execute)
    settings = {
        "max_new_tokens": 4, "max_input_tokens": 32, "tools": None,
        "continue_on_tool_error": True, "continue_on_missing_context_tool": True,
        "repair_wls_from_user": False, "enable_thinking": False, "verbose": False,
        "inject_empty_thought_channel": False, "gc_collect_every_n_turns": 0,
        "empty_cuda_cache_every_n_turns": 0, "filter_unavailable_helper_tools": False,
        "inject_runtime_helper_note": False,
    }
    return evaluator, Model(), SimpleNamespace(pad_token_id=0, eos_token_id=1), settings, calls


def _sample(index):
    return {"messages": [{"role": "system", "content": "test"}, {"role": "user", "content": "probe"}],
            "runtime_context": {"sample_id": index}}


def test_single_gemma_loop_caps_actual_failed_tool_attempts_at_explicit_limit(stubbed_gemma):
    evaluator, model, tokenizer, settings, calls = stubbed_gemma
    result = evaluator.run_one_sample(_sample(0)["messages"], model, tokenizer,
        max_turns=3, runtime_context=_sample(0)["runtime_context"], **settings)
    assert len(calls) == len(result["tool_calls"]) == len(result["turn_trace"]) == 3
    assert "Max turns reached" in result["error"]


def test_batched_gemma_loop_gives_each_episode_its_own_action_limit(stubbed_gemma):
    evaluator, model, tokenizer, settings, calls = stubbed_gemma
    results = evaluator.run_sample_batch([_sample(0), _sample(1)], model, tokenizer,
        sample_offset=0, max_turns=3, **settings)
    assert len(calls) == 6
    assert all(len(result["tool_calls"]) == 3 for result in results)
    assert all("Max turns reached" in result["error"] for result in results)


def test_rolling_gemma_loop_refill_does_not_reset_an_existing_episode_budget(stubbed_gemma):
    evaluator, model, tokenizer, settings, calls = stubbed_gemma
    results = []
    evaluator.run_samples_with_rolling_scheduler([_sample(i) for i in range(3)], model, tokenizer,
        sample_offset=0, concurrent_conversations=2, max_turns=3, on_result=results.append, **settings)
    assert len(results) == 3
    assert len(calls) == 9
    assert all(len(result["tool_calls"]) == 3 for result in results)


def test_partial_rolling_batch_failure_never_replays_a_dispatched_tool(stubbed_gemma, monkeypatch):
    evaluator, model, tokenizer, settings, calls = stubbed_gemma
    seen = 0

    def fail_after_second_dispatch(name, result, *args):
        nonlocal seen
        seen += 1
        if seen == 2:
            raise RuntimeError("result rendering failed after tool dispatch")
        return result

    monkeypatch.setattr(evaluator, "summarize_tool_result_for_conversation", fail_after_second_dispatch)
    results = []
    evaluator.run_samples_with_rolling_scheduler([_sample(0), _sample(1)], model, tokenizer,
        sample_offset=0, concurrent_conversations=2, max_turns=3, on_result=results.append, **settings)
    by_index = {result["sample_index"]: result for result in results}
    assert calls.count(0) == 3
    assert calls.count(1) == 1
    assert len(by_index[1]["tool_calls"]) == 1
    assert "not replayed" in by_index[1]["error"]
    assert len(by_index[0]["tool_calls"]) == 3


def test_direct_state_turn_cannot_bypass_actual_dispatch_cap_with_repeated_turn_index(stubbed_gemma):
    evaluator, _, tokenizer, _, calls = stubbed_gemma
    sample = _sample(0)
    state = evaluator.init_eval_sample_state(0, sample["messages"], sample["runtime_context"], action_limit=3)
    for _ in range(5):
        evaluator.run_state_turn(state, turn_index0=0, response_text="tool", token_count=1,
            was_truncated=False, turn_max_new_tokens=4, tokenizer=tokenizer,
            continue_on_tool_error=True, continue_on_missing_context_tool=True,
            repair_wls_from_user=False, verbose=False, prompt_tokens=1, model_generate_seconds=0.0)
    assert len(calls) == len(state.tool_calls_made) == 3
    assert "Max turns reached" in state.error_msg
