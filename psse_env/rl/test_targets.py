from __future__ import annotations

import json
import unittest

from psse_env.rl.targets import (
    TargetError,
    TargetRenderer,
    labelled_action,
    normalize_thought_prefix,
    render_target_completion,
)

GENERATION_SUFFIX = "<|turn>model\n<|channel>thought\n<channel|>"
SCHEMAS = {
    name: {"type": "object", "properties": {"case_path": {"type": "string"}}, "required": ["case_path"]}
    for name in ("commit_state", "rollback_state")
}
TOOLS = [{"type": "function", "function": {"name": name, "parameters": schema}} for name, schema in SCHEMAS.items()]


def _wire(name: str, **arguments) -> str:
    body = ",".join(f'{key}:<|"|>{value}<|"|>' for key, value in sorted(arguments.items()))
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


class _Processor:
    """A Gemma-like template: the generation prompt carries the empty thought channel, a finished assistant turn does not."""

    def render_assistant(self, function) -> str:
        return "<|turn>model\n" + _wire(function["name"], **function["arguments"]) + "<turn|>\n"

    def apply_chat_template(self, messages, *, tools=None, tokenize, add_generation_prompt, enable_thinking=False):
        assert tokenize is False and enable_thinking is False
        pieces = ["<tools>" + json.dumps(tools or [], sort_keys=True) + "</tools>"]
        for message in messages:
            if message["role"] == "assistant":
                pieces.append(self.render_assistant(message["tool_calls"][0]["function"]))
            else:
                pieces.append(f"<|turn>{message['role']}\n{message['content']}<turn|>\n")
        if add_generation_prompt:
            pieces.append(GENERATION_SUFFIX)
        return "".join(pieces)

    def __call__(self, text=None, **_kwargs):
        return {"input_ids": [[ord(character) for character in text]]}


class _ArgumentDroppingProcessor(_Processor):
    def render_assistant(self, function) -> str:
        return "<|turn>model\n" + _wire(function["name"]) + "<turn|>\n"


class _OtherTurnProcessor(_Processor):
    def render_assistant(self, function) -> str:
        return "<|turn>assistant\n" + _wire(function["name"], **function["arguments"]) + "<turn|>\n"


class _Policy:
    def __init__(self, processor) -> None:
        self.processor = processor

    def render_messages(self, observation):
        messages = [{"role": "system", "content": "diagnose"}, {"role": "user", "content": json.dumps({"state": dict(observation)}, sort_keys=True)}]
        return messages, [dict(tool) for tool in TOOLS], dict(SCHEMAS)

    def render_prompt(self, observation):
        from psse_env.sft.gemma_text import render_eval_text

        messages, tools, parameters = self.render_messages(observation)
        return render_eval_text(self.processor, messages, tools, enable_thinking=False, inject_empty_thought_channel=False), tools, parameters


DECLINE = {"position": 3, "tool": "rollback_state", "label": "commit_state", "agrees": False, "declined": True}
COMMIT = {"position": 3, "tool": "commit_state", "label": "rollback_state", "agrees": False, "declined": False}


class LabelledActionTests(unittest.TestCase):
    def test_the_label_keeps_the_sampled_arguments_and_swaps_the_tool(self) -> None:
        text = _wire("rollback_state", case_path="candidate")
        self.assertEqual(labelled_action(text, DECLINE), {"tool": "commit_state", "arguments": {"case_path": "candidate"}})
        self.assertEqual(labelled_action('{"name": "commit_state", "arguments": {"case_path": "candidate"}}', COMMIT),
                         {"tool": "rollback_state", "arguments": {"case_path": "candidate"}})
        with self.assertRaisesRegex(TargetError, "not the decision's"):
            labelled_action(_wire("commit_state", case_path="candidate"), DECLINE)
        with self.assertRaisesRegex(TargetError, "no label"):
            labelled_action(text, {**DECLINE, "label": None})
        with self.assertRaisesRegex(TargetError, "does not parse"):
            labelled_action("I would rather not say", DECLINE)


class RenderTests(unittest.TestCase):
    def test_the_thought_channel_of_a_generation_prompt_is_restored_in_the_full_render(self) -> None:
        prompt = "<|turn>user\nx<turn|>\n" + GENERATION_SUFFIX
        full = "<|turn>user\nx<turn|>\n<|turn>model\ncall<turn|>\n"
        normalized, injected = normalize_thought_prefix(full, prompt)
        self.assertTrue(injected)
        self.assertTrue(normalized.startswith(prompt))
        self.assertEqual(normalized[len(prompt):], "call<turn|>\n")
        self.assertEqual(normalize_thought_prefix(prompt + "call", prompt), (prompt + "call", False))
        self.assertEqual(normalize_thought_prefix("elsewhere", prompt), ("elsewhere", False))

    def test_the_completion_is_the_token_suffix_after_the_generation_prompt(self) -> None:
        processor = _Processor()
        policy = _Policy(processor)
        observation = {"active_state_id": "active", "candidate_state_id": "candidate"}
        messages, tools, _parameters = policy.render_messages(observation)
        rendered, _tools, _parameters = policy.render_prompt(observation)
        action = {"tool": "commit_state", "arguments": {"case_path": "candidate"}}
        target = render_target_completion(processor, messages=messages, tools=tools, rendered_prompt=rendered, action=action)
        expected_text = _wire("commit_state", case_path="candidate") + "<turn|>\n"
        self.assertEqual(target.text, expected_text)
        self.assertEqual(target.completion_ids, [ord(character) for character in expected_text])
        self.assertEqual(target.prompt_ids, [ord(character) for character in rendered])
        self.assertTrue(target.empty_thought_injected)
        with self.assertRaisesRegex(TargetError, "not an exact prefix"):
            render_target_completion(_OtherTurnProcessor(), messages=messages, tools=tools, rendered_prompt=rendered, action=action)

    def test_the_renderer_round_trips_the_rendered_call_through_the_policys_parser(self) -> None:
        processor = _Processor()
        renderer = TargetRenderer(_Policy(processor), processor)
        observation = {"active_state_id": "active", "candidate_state_id": "candidate"}
        ids = renderer(observation, _wire("rollback_state", case_path="candidate"), DECLINE)
        self.assertEqual(ids, [ord(character) for character in _wire("commit_state", case_path="candidate") + "<turn|>\n"])
        self.assertEqual((renderer.rendered, renderer.refused), (1, 0))
        # A template that loses the arguments renders a call that no longer parses to the label: refused, never trained.
        broken = _ArgumentDroppingProcessor()
        refusing = TargetRenderer(_Policy(broken), broken)
        with self.assertRaisesRegex(TargetError, "does not parse back"):
            refusing(observation, _wire("rollback_state", case_path="candidate"), DECLINE)
        self.assertEqual((refusing.rendered, refusing.refused), (0, 1))
        with self.assertRaises(TargetError):
            renderer(observation, "no call here", DECLINE)
        self.assertEqual(renderer.refused, 1)


if __name__ == "__main__":
    unittest.main()
