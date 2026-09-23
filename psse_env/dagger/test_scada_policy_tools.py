"""Capability declarations agree across SFT and live prompt renderers per evidence profile."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest

from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT, WLS_GATED_PROMPT_PARAGRAPH, examples_to_chat_sft,
    tool_schemas_for_observation, system_prompt_for_observation,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas, CANONICAL_TO_INTERNAL_TOOL
from psse_env.evidence_profile import (
    AUXILIARY_EVIDENCE_PROFILE, DEFAULT_EVIDENCE_PROFILE, SCADA_ALLOWED_TOOLS, SCADA_ONLY_PROFILE,
    WLS_GATED_DISABLED_TOOLS, WLS_GATED_PROFILE,
)


def _observation(profile=SCADA_ONLY_PROFILE):
    state = {"active_state_id": "capability-test:s0", "remaining_budget": 40, "history_window": []}
    if profile is not None:
        state["evidence_profile"] = profile
    return state


def _row(profile=SCADA_ONLY_PROFILE, tool="run_wls"):
    return examples_to_chat_sft([{
        "policy_observation": _observation(profile),
        "preferred_action": {"tool": tool, "arguments": {"state_id": "capability-test:s0"}},
    }], require_derived_provenance=False)[0]


def _internal_names(schemas):
    names = {tool["function"]["name"] for tool in schemas}
    return {CANONICAL_TO_INTERNAL_TOOL.get(name, name) for name in names}


class _RenderCaptured(Exception):
    pass


class SCADAPolicyToolsTests(unittest.TestCase):
    def _assert_scada_tools(self, schemas):
        names = {tool["function"]["name"] for tool in schemas}
        self.assertEqual(_internal_names(schemas), SCADA_ALLOWED_TOOLS)
        self.assertNotIn("get_three_phase_context", names)
        self.assertNotIn("get_harmonic_context", names)
        self.assertNotIn("run_alternative_test", names)

    def _assert_gated_tools(self, schemas):
        every_tool = _internal_names(unified_tool_schemas())
        self.assertEqual(_internal_names(schemas), every_tool - WLS_GATED_DISABLED_TOOLS)
        names = {tool["function"]["name"] for tool in schemas}
        self.assertIn("get_three_phase_context", names)
        self.assertIn("get_harmonic_context", names)
        self.assertNotIn("run_alternative_test", names)

    def test_explicit_strict_export_records_profile_and_only_scada_tools(self):
        row = _row()
        self.assertEqual(row["evidence_profile"], SCADA_ONLY_PROFILE)
        self.assertEqual(row["metadata"]["evidence_profile"], SCADA_ONLY_PROFILE)
        self._assert_scada_tools(row["tools"])
        self.assertIn("Start with WLS", row["messages"][0]["content"])
        self.assertIn("Auxiliary phase, harmonic and HIF replay data are unavailable", row["messages"][0]["content"])

    def test_gated_default_export_exposes_all_but_disabled_tools_and_states_the_gate(self):
        self.assertEqual(DEFAULT_EVIDENCE_PROFILE, WLS_GATED_PROFILE)
        row = _row(WLS_GATED_PROFILE)
        self.assertEqual(row["evidence_profile"], WLS_GATED_PROFILE)
        self.assertEqual(row["metadata"]["evidence_profile"], WLS_GATED_PROFILE)
        self._assert_gated_tools(row["tools"])
        prompt = row["messages"][0]["content"]
        self.assertTrue(prompt.startswith(CANONICAL_DAGGER_SYSTEM_PROMPT))
        self.assertTrue(prompt.endswith(WLS_GATED_PROMPT_PARAGRAPH))
        self.assertIn("wls_gated_diagnostics", prompt)
        self.assertIn("only after the current WLS on the active state has raised a chi-square or normalized-residual alarm", prompt)
        self.assertIn("unavailable answer is not a diagnosis", prompt)
        self.assertIn("run_alternative_test is unavailable", prompt)

    def test_missing_legacy_profile_is_not_relabelled_on_reexport(self):
        row = _row(None)
        self.assertNotIn("evidence_profile", row)
        self.assertNotIn("evidence_profile", row["metadata"])
        self.assertEqual(row["messages"][0]["content"], CANONICAL_DAGGER_SYSTEM_PROMPT)
        self.assertIn("get_three_phase_context", {tool["function"]["name"] for tool in row["tools"]})

    def test_explicit_auxiliary_profile_keeps_historical_capabilities(self):
        row = _row(AUXILIARY_EVIDENCE_PROFILE)
        self.assertEqual(row["evidence_profile"], AUXILIARY_EVIDENCE_PROFILE)
        self.assertEqual(row["messages"][0]["content"], CANONICAL_DAGGER_SYSTEM_PROMPT)
        self.assertIn("run_hse_from_path", {tool["function"]["name"] for tool in row["tools"]})
        self.assertIn("run_alternative_test", {tool["function"]["name"] for tool in row["tools"]})

    def test_strict_sft_cannot_export_an_auxiliary_teacher_action(self):
        for tool in ("get_three_phase_context", "get_harmonic_context", "run_alternative_test"):
            with self.subTest(tool=tool), self.assertRaisesRegex(ValueError, "no schema"):
                _row(tool=tool)

    def test_gated_sft_exports_gated_diagnostics_but_never_the_disabled_tool(self):
        for tool in ("get_three_phase_context", "get_harmonic_context", "run_three_phase_nlm_from_path"):
            with self.subTest(tool=tool):
                row = _row(WLS_GATED_PROFILE, tool=tool)
                self.assertEqual(row["evidence_profile"], WLS_GATED_PROFILE)
        with self.assertRaisesRegex(ValueError, "no schema"):
            _row(WLS_GATED_PROFILE, tool="run_alternative_test")

    def test_invalid_explicit_profile_fails_closed(self):
        with self.assertRaises(ValueError):
            tool_schemas_for_observation(unified_tool_schemas(), _observation("unknown"))
        with self.assertRaises(ValueError):
            system_prompt_for_observation("prompt", _observation("unknown"))

    def test_native_release_and_small_live_renderers_use_same_filtered_tools_as_sft(self):
        from psse_env.dagger import research_policy_factory, release_factories, preliminary_e2b_eval

        cases = (
            (research_policy_factory._CanonicalResearchNativePolicy, research_policy_factory, "render_eval_text", "generate_text"),
            (release_factories._CanonicalGemmaPolicy, release_factories, "_render", "act"),
            (preliminary_e2b_eval._CanonicalE2BPolicy, preliminary_e2b_eval, "render_eval_text", "generate_text"),
        )
        for cls, module, render_name, method in cases:
            for profile in (SCADA_ONLY_PROFILE, WLS_GATED_PROFILE, AUXILIARY_EVIDENCE_PROFILE):
                with self.subTest(actor=cls.__name__, profile=profile):
                    actor = cls.__new__(cls)
                    actor._tools = unified_tool_schemas()
                    actor._bundle = SimpleNamespace(processor=object())
                    captured = {}
                    def capture(processor, messages, tools, **kwargs):
                        captured.update(messages=messages, tools=tools)
                        raise _RenderCaptured()
                    with patch.object(module, render_name, capture), self.assertRaises(_RenderCaptured):
                        getattr(actor, method)(_observation(profile))
                    self.assertEqual(captured["messages"][0]["content"], _row(profile)["messages"][0]["content"])
                    if profile == SCADA_ONLY_PROFILE:
                        self._assert_scada_tools(captured["tools"])
                    elif profile == WLS_GATED_PROFILE:
                        self._assert_gated_tools(captured["tools"])
                    else:
                        self.assertIn("get_three_phase_context", {tool["function"]["name"] for tool in captured["tools"]})
                        self.assertIn("run_alternative_test", {tool["function"]["name"] for tool in captured["tools"]})


if __name__ == "__main__":
    unittest.main()
