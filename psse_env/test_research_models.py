from __future__ import annotations

import unittest

from psse_env.research_models import (
    DEFAULT_RESEARCH_MODEL,
    GEMMA4_12B,
    GEMMA4_E4B,
    assert_adapter_model_compatible,
    known_research_model,
    resolve_research_model_spec,
)


class ResearchModelSelectionTests(unittest.TestCase):
    def test_12b_is_the_atomic_default_and_e4b_is_the_smoke_choice(self) -> None:
        self.assertIs(DEFAULT_RESEARCH_MODEL, GEMMA4_12B)
        self.assertEqual(DEFAULT_RESEARCH_MODEL.architecture, "gemma4_unified")
        self.assertEqual(DEFAULT_RESEARCH_MODEL.prompt_profile, "native")
        self.assertEqual(GEMMA4_E4B.architecture, "gemma4")
        self.assertEqual(GEMMA4_E4B.prompt_profile, "small_forced")

    def test_hub_cache_paths_resolve_the_same_pinned_spec(self) -> None:
        source = (
            "/scratch/hf/hub/models--google--gemma-4-12B-it/snapshots/"
            + GEMMA4_12B.revision
        )
        self.assertIs(known_research_model(source), GEMMA4_12B)
        resolved = resolve_research_model_spec(model=source)
        self.assertEqual(resolved.revision, GEMMA4_12B.revision)
        self.assertEqual(resolved.architecture, "gemma4_unified")

    def test_known_architecture_and_prompt_cannot_be_overridden_incoherently(self) -> None:
        with self.assertRaisesRegex(ValueError, "architecture=.*conflicts"):
            resolve_research_model_spec(
                model=GEMMA4_12B.model_id,
                architecture="gemma4",
            )
        with self.assertRaisesRegex(ValueError, "prompt_profile=.*conflicts"):
            resolve_research_model_spec(
                model=GEMMA4_12B.model_id,
                prompt_profile="small_forced",
            )

    def test_known_revision_cannot_drift_from_the_registry(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts with the pinned"):
            resolve_research_model_spec(
                model=GEMMA4_12B.model_id,
                revision="0" * 40,
            )

    def test_known_cross_model_adapter_reuse_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fresh BC0 adapter"):
            assert_adapter_model_compatible(
                GEMMA4_12B.model_id, "unsloth/gemma-4-E2B-it"
            )
        assert_adapter_model_compatible(
            GEMMA4_12B.model_id,
            "/cache/models--google--gemma-4-12B-it/snapshots/abc",
        )


if __name__ == "__main__":
    unittest.main()
