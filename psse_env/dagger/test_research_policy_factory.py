from __future__ import annotations

import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from psse_env.dagger import research_policy_factory as factory
from psse_env.sft.gates import GateError


class ResearchPolicyFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        factory.clear_research_policy_cache()

    def tearDown(self) -> None:
        factory.clear_research_policy_cache()

    def _adapter(
        self,
        root: Path,
        *,
        peft_type: str = "LORA",
        base_model: str = "unsloth/gemma-4-E2B-it",
        processor_assets: bool = False,
        name: str = "adapter",
    ) -> Path:
        adapter = root / name
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": peft_type,
                    "base_model_name_or_path": base_model,
                }
            ),
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        if processor_assets:
            (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")
        return adapter

    def _fake_loaders(self, *, model_type: str = "gemma4"):
        calls: dict[str, list] = {"processor": [], "model": [], "peft": []}

        class Processor:
            def apply_chat_template(self, *_args, **_kwargs):
                return "prompt"

            def decode(self, *_args, **_kwargs):
                return "decoded"

        class AutoProcessor:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls["processor"].append((source, dict(kwargs)))
                return Processor()

        class Base:
            config = types.SimpleNamespace(model_type=model_type)

            def generate(self, *_args, **_kwargs):
                return None

            def get_input_embeddings(self):
                return object()

        class AutoModel:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls["model"].append((source, dict(kwargs)))
                return Base()

        class Peft:
            def __init__(self):
                self.peft_config = {"default": object()}
                self.evaluated = False

            def eval(self):
                self.evaluated = True

        class PeftModel:
            @classmethod
            def from_pretrained(cls, base, source, **kwargs):
                calls["peft"].append((base, source, dict(kwargs)))
                return Peft()

        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = AutoProcessor
        transformers.AutoModelForMultimodalLM = AutoModel
        peft = types.ModuleType("peft")
        peft.PeftModel = PeftModel
        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        return calls, {"transformers": transformers, "peft": peft, "torch": torch}

    def test_factory_accepts_adapter_path_not_tree_revision(self) -> None:
        parameters = inspect.signature(
            factory.research_gemma_policy_factory
        ).parameters
        self.assertIn("adapter_path", parameters)
        self.assertNotIn("adapter_revision", parameters)
        self.assertNotIn("tree_sha256", parameters)

    def test_repetition_detector_flags_only_sustained_runs(self) -> None:
        ordinary = factory._token_repetition_metrics(
            [*range(32), 1, 2, 1, 2, 3]
        )
        runaway = factory._token_repetition_metrics([7, 8, 9] * 8)
        single_token_runaway = factory._token_repetition_metrics([11] * 24)
        self.assertFalse(ordinary["repetition_loop_detected"])
        self.assertTrue(runaway["repetition_loop_detected"])
        self.assertEqual(runaway["repetition_span_tokens"], 24)
        self.assertTrue(single_token_runaway["repetition_loop_detected"])

    def test_structural_adapter_validation_has_no_hash_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(Path(directory))
            resolved, config = factory._validate_adapter_directory(adapter)
        self.assertEqual(resolved, adapter.resolve())
        self.assertEqual(config["peft_type"], "LORA")

    def test_non_lora_and_missing_weights_fail_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(Path(directory), peft_type="IA3")
            with self.assertRaisesRegex(GateError, "LoRA"):
                factory._validate_adapter_directory(adapter)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )
            (adapter / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(GateError, "adapter weights"):
                factory._validate_adapter_directory(adapter)

    def test_factory_returns_non_release_wrapper(self) -> None:
        bundle = object()
        identity = {
            "base_model": "unsloth/gemma-4-31B-it",
            "base_revision": "f" * 40,
            "adapter_path": "/tmp/adapter",
        }

        class FakePolicy:
            def __init__(self, observed_bundle, observed_identity):
                self.bundle = observed_bundle
                self.identity = observed_identity

        with patch.object(
            factory, "_load_research_bundle", return_value=(bundle, identity)
        ), patch.object(factory, "ResearchGemmaPolicy", FakePolicy):
            policy = factory.research_gemma_policy_factory("adapter")
        self.assertIs(policy.bundle, bundle)
        self.assertFalse(hasattr(policy, "release_policy_identity"))

    def test_local_base_and_adapter_processor_load_without_hub_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "models--unsloth--gemma-4-E2B-it" / "snapshots" / ("a" * 40)
            base.mkdir(parents=True)
            adapter = self._adapter(
                root,
                base_model=str(base),
                processor_assets=True,
            )
            calls, modules = self._fake_loaders()
            with patch.dict(sys.modules, modules):
                bundle, identity = factory._load_research_bundle(
                    adapter_path=adapter,
                    base_model=None,
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=False,
                )
        self.assertEqual(calls["model"][0][0], str(base.resolve()))
        self.assertNotIn("revision", calls["model"][0][1])
        self.assertEqual(calls["processor"][0][0], str(adapter.resolve()))
        self.assertNotIn("revision", calls["processor"][0][1])
        self.assertEqual(identity["prompt_profile"], "small_forced")
        self.assertEqual(identity["architecture"], "gemma4")
        self.assertEqual(bundle.model_revision, "local-snapshot")

    def test_non_4bit_load_is_bf16_and_cache_reuses_exact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._adapter(root, name="first")
            second = self._adapter(root, name="second")
            calls, modules = self._fake_loaders()
            with patch.dict(sys.modules, modules):
                first_bundle, _ = factory._load_research_bundle(
                    adapter_path=first,
                    base_model=None,
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=True,
                )
                cached_bundle, _ = factory._load_research_bundle(
                    adapter_path=first,
                    base_model=None,
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=True,
                )
                second_bundle, _ = factory._load_research_bundle(
                    adapter_path=second,
                    base_model=None,
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=True,
                )
        self.assertIs(first_bundle, cached_bundle)
        self.assertIsNot(first_bundle, second_bundle)
        self.assertEqual(len(calls["model"]), 2)
        self.assertIs(
            calls["model"][0][1]["dtype"], modules["torch"].bfloat16
        )
        self.assertNotIn("quantization_config", calls["model"][0][1])

    def test_prompt_profile_requires_known_family_or_explicit_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "opaque_snapshot"
            base.mkdir()
            adapter = self._adapter(root, base_model=str(base))
            with self.assertRaisesRegex(ValueError, "prompt_profile is required"):
                factory._resolve_prompt_profile(
                    None,
                    base_model=str(base),
                    recorded_base=str(base),
                    adapter=adapter,
                )
            self.assertEqual(
                factory._resolve_prompt_profile(
                    "release",
                    base_model=str(base),
                    recorded_base=str(base),
                    adapter=adapter,
                ),
                "release",
            )
            with self.assertRaisesRegex(ValueError, "conflicts"):
                factory._resolve_prompt_profile(
                    "release",
                    base_model="unsloth/gemma-4-E2B-it",
                    recorded_base="",
                    adapter=adapter,
                )
            with self.assertRaisesRegex(ValueError, "conflicting prompt families"):
                factory._resolve_prompt_profile(
                    "e2b",
                    base_model="unsloth/gemma-4-E2B-it",
                    recorded_base="unsloth/gemma-4-31B-it",
                    adapter=adapter,
                )

    def test_12b_uses_unified_multimodal_loader_and_native_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                Path(directory), base_model="google/gemma-4-12B-it"
            )
            calls, modules = self._fake_loaders(model_type="gemma4_unified")
            with patch.dict(sys.modules, modules):
                _, identity = factory._load_research_bundle(
                    adapter_path=adapter,
                    base_model=None,
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=False,
                )
        self.assertEqual(calls["model"][0][0], "google/gemma-4-12B-it")
        self.assertEqual(
            calls["model"][0][1]["revision"],
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        )
        self.assertEqual(identity["architecture"], "gemma4_unified")
        self.assertEqual(identity["prompt_profile"], "native")

    def test_cross_model_adapter_mismatch_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(Path(directory))
            with self.assertRaisesRegex(GateError, "adapter/base mismatch"):
                factory._load_research_bundle(
                    adapter_path=adapter,
                    base_model="google/gemma-4-12B-it",
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=False,
                )

    def test_12b_native_policy_is_not_the_release_decoder(self) -> None:
        bundle = types.SimpleNamespace(model=object(), processor=object())
        policy = factory._canonical_research_policy(
            bundle,
            {
                "prompt_profile": "native",
                "architecture": "gemma4_unified",
            },
        )
        self.assertIsInstance(policy, factory._CanonicalResearchNativePolicy)
        self.assertNotEqual(type(policy).__name__, "_CanonicalGemmaPolicy")

    def test_exported_probe_uses_canonical_surface_without_alias_recompaction(self) -> None:
        observed = {}

        class Canonical:
            last_action_metrics = {}

            def act(self, state):
                observed["state"] = state
                return {"tool": "wls_from_path", "arguments": {"case_path": "active"}}

        exported = {
            "active_state_id": "active",
            "history_window": [{"action": {"tool": "wls_from_path"}}],
        }
        with patch.object(factory, "_canonical_research_policy", return_value=Canonical()):
            policy = factory.ResearchGemmaPolicy(object(), {"prompt_profile": "native"})
        action = policy.act_model_observation(exported)
        self.assertEqual(observed["state"], exported)
        self.assertEqual(action["tool"], "wls_from_path")

    def test_12b_native_generation_uses_processor_thought_prefix_without_forcing(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - test environment guard.
            self.skipTest(str(exc))

        class Embeddings(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

        class Model:
            def __init__(self):
                self.embeddings = Embeddings()

            def get_input_embeddings(self):
                return self.embeddings

            def eval(self):
                return self

            def forward(self, input_ids, mm_token_type_ids=None):
                return input_ids, mm_token_type_ids

            def generate(self, **kwargs):
                suffix = torch.tensor([[99]], dtype=kwargs["input_ids"].dtype)
                return torch.cat((kwargs["input_ids"], suffix), dim=-1)

        bundle = factory._ModelBundle(
            model=Model(),
            processor=object(),
            model_id="google/gemma-4-12B-it",
            model_revision="f" * 40,
            adapter_snapshot_path="adapter",
        )
        policy = factory._CanonicalResearchNativePolicy(
            bundle, architecture="gemma4_unified"
        )
        suffix = policy._UNIFIED_GENERATION_SUFFIX
        with patch.object(
            factory, "render_eval_text", return_value="prompt" + suffix
        ) as render, patch.object(
            factory,
            "tokenize_rendered_text",
            return_value={
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            },
        ), patch.object(
            factory, "get_stop_token_ids", return_value=[1]
        ), patch.object(
            factory, "resolve_pad_token_id", return_value=0
        ), patch.object(
            factory,
            "decode_generated_response",
            return_value=(
                '<|tool_call|>call:wls_from_path{"case_path":"active"}',
                1,
                0,
            ),
        ):
            action = policy.act({"case_alias": "active"})
        self.assertEqual(
            action, {"tool": "wls_from_path", "arguments": {"case_path": "active"}}
        )
        self.assertFalse(render.call_args.kwargs["enable_thinking"])
        self.assertFalse(render.call_args.kwargs["inject_empty_thought_channel"])
        self.assertIsNone(policy.last_action_metrics["forced_tool_prefix"])

    def test_empty_peft_registration_fails_without_raw_base_fallback(self) -> None:
        class EmptyPeft:
            peft_config = {}

            def eval(self):
                pass

        class EmptyPeftModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return EmptyPeft()

        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(Path(directory))
            calls, modules = self._fake_loaders()
            modules["peft"].PeftModel = EmptyPeftModel
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(
                GateError, "no active adapter"
            ):
                factory._load_research_bundle(
                    adapter_path=adapter,
                    base_model=None,
                    base_revision=None,
                    load_in_4bit=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    prompt_profile=None,
                    use_cache=False,
                )
        self.assertEqual(len(calls["model"]), 1)


if __name__ == "__main__":
    unittest.main()
