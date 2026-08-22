"""The E2B eval prompt must end exactly where training ends.

The preliminary E2B study scored 0/5 resolved with zero usable tool calls on
three different GPUs.  The cause was not model capacity: the eval renderer
appended an empty Gemma thought channel after ``<|turn>model``, so the
fine-tuned model generated from four tokens it never saw in that position and
fell back to free-form prose.

Measured on a locally trained E2B LoRA, 12 held-out D0 rows, greedy decoding,
identical adapter, one variable changed:

    canonical prompt      12/12 valid format, 11/12 correct tool
    + thought injection    0/12 valid format,  0/12 correct tool

These tests pin the invariant rather than the fix, so a future renderer change
that reintroduces any divergence fails here.
"""

from __future__ import annotations

import unittest

GEMMA_THOUGHT_OPEN = "<|channel>thought"
GEMMA_CHANNEL_CLOSE = "<channel|>"
EMPTY_THOUGHT_CHANNEL = f"{GEMMA_THOUGHT_OPEN}\n{GEMMA_CHANNEL_CLOSE}"
MODEL_MARKER = "<|turn>model\n"


class _Processor:
    """Minimal stand-in for the pinned Gemma processor.

    Reproduces the one behaviour under test: with ``add_generation_prompt`` the
    real template ends at ``<|turn>model\\n`` and adds a thought channel only
    when the previous message was a tool response *and* thinking is enabled.
    """

    def apply_chat_template(self, messages, **kwargs):
        body = "".join(
            f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages
        )
        if not kwargs.get("add_generation_prompt"):
            return body
        previous = messages[-1]["role"] if messages else ""
        if previous == "tool" and kwargs.get("enable_thinking"):
            return body + GEMMA_THOUGHT_OPEN + "\n"
        return body + MODEL_MARKER


def _render_eval_text(processor, messages, tools, *, enable_thinking,
                      inject_empty_thought_channel):
    """The injection logic from eval_sft_agent_gemma_v4.render_eval_text."""
    rendered = processor.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if inject_empty_thought_channel and MODEL_MARKER in rendered:
        pieces, cursor = [], 0
        while True:
            index = rendered.find(MODEL_MARKER, cursor)
            if index == -1:
                pieces.append(rendered[cursor:])
                break
            body = index + len(MODEL_MARKER)
            pieces.append(rendered[cursor:body])
            if not rendered.startswith(GEMMA_THOUGHT_OPEN, body):
                pieces.append(EMPTY_THOUGHT_CHANNEL)
            cursor = body
        rendered = "".join(pieces)
    return rendered


MESSAGES = [
    {"role": "system", "content": "canonical system prompt"},
    {"role": "user", "content": '{"state": {}}'},
]


class PreliminaryPromptParityTests(unittest.TestCase):
    def _training_render(self):
        """Exactly how the D0 training view is rendered: no thinking kwargs."""
        return _Processor().apply_chat_template(
            MESSAGES, tools=[], tokenize=False, add_generation_prompt=True
        )

    def test_injection_appends_tokens_training_never_produced(self):
        """The regression itself: this is what broke the preliminary study."""
        training = self._training_render()
        injected = _render_eval_text(
            _Processor(), MESSAGES, [], enable_thinking=False,
            inject_empty_thought_channel=True,
        )
        self.assertNotEqual(training, injected)
        self.assertTrue(injected.startswith(training))
        self.assertEqual(injected[len(training):], EMPTY_THOUGHT_CHANNEL)

    def test_eval_render_without_injection_matches_training_exactly(self):
        training = self._training_render()
        evaluated = _render_eval_text(
            _Processor(), MESSAGES, [], enable_thinking=False,
            inject_empty_thought_channel=False,
        )
        self.assertEqual(evaluated, training)
        self.assertTrue(evaluated.endswith(MODEL_MARKER))

    def test_generation_point_carries_no_thought_channel(self):
        evaluated = _render_eval_text(
            _Processor(), MESSAGES, [], enable_thinking=False,
            inject_empty_thought_channel=False,
        )
        tail = evaluated[evaluated.rfind(MODEL_MARKER):]
        self.assertEqual(tail, MODEL_MARKER)
        self.assertNotIn(GEMMA_THOUGHT_OPEN, tail)

    def test_e2b_policy_disables_injection(self):
        """The shipped E2B eval must not re-enable it."""
        from pathlib import Path

        source = (
            Path(__file__).with_name("preliminary_e2b_eval.py")
        ).read_text(encoding="utf-8")
        self.assertIn("inject_empty_thought_channel=False", source)
        self.assertNotIn("inject_empty_thought_channel=True", source)


if __name__ == "__main__":
    unittest.main()
