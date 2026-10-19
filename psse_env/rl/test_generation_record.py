from __future__ import annotations

import unittest

from psse_env.dagger.research_policy_factory import GenerationOptions, sampling_summary
from psse_env.rl.train import parse_shard, shard_scenarios

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class GenerationRecordTests(unittest.TestCase):
    def test_sampling_summary_reports_how_peaked_a_completion_was(self) -> None:
        summary = sampling_summary([1.0, 0.999, 0.4, 0.95])
        self.assertAlmostEqual(summary["min_token_probability"], 0.4)
        self.assertAlmostEqual(summary["mean_token_probability"], (1.0 + 0.999 + 0.4 + 0.95) / 4)
        self.assertEqual(summary["tokens_below_0_99"], 2)
        self.assertEqual(sampling_summary([])["tokens_below_0_99"], 0)
        self.assertIsNone(sampling_summary([])["min_token_probability"])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_sampled_token_probabilities_follow_the_processed_scores(self) -> None:
        from psse_env.dagger.research_policy_factory import sampled_token_probabilities

        scores = (torch.tensor([[0.0, 0.0, float("-inf")]]), torch.tensor([[10.0, 0.0, 0.0]]))
        sampled = torch.tensor([1, 0, 2])
        probabilities = sampled_token_probabilities(scores, sampled)
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertGreater(probabilities[1], 0.99)

    def test_the_prefill_pad_moves_prompts_off_the_corrupted_residue_and_can_be_switched_off(self) -> None:
        import os
        from unittest.mock import patch

        from psse_env.dagger.research_policy_factory import prefill_pad_tokens

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PSSE_SDPA_PREFILL_PAD", None)
            self.assertEqual([prefill_pad_tokens(n) for n in (4609, 5857, 7713, 1, 9)], [1, 1, 1, 1, 1])
            self.assertEqual([prefill_pad_tokens(n) for n in (4608, 4610, 2451, 4383, 8)], [0, 0, 0, 0, 0])
        with patch.dict(os.environ, {"PSSE_SDPA_PREFILL_PAD": "0"}):
            self.assertEqual(prefill_pad_tokens(4609), 0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_left_pad_masks_one_leading_position_of_every_aligned_tensor(self) -> None:
        from psse_env.dagger.research_policy_factory import left_pad_model_inputs

        inputs = {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 0]]),
            "scalar": torch.tensor(3),
        }
        padded = left_pad_model_inputs(inputs, pad_tokens=1, pad_token_id=0)
        self.assertEqual(padded["input_ids"].tolist(), [[0, 5, 6, 7]])
        self.assertEqual(padded["attention_mask"].tolist(), [[0, 1, 1, 1]])
        self.assertEqual(padded["mm_token_type_ids"].tolist(), [[0, 0, 0, 0]])
        self.assertEqual(int(padded["scalar"]), 3)
        self.assertEqual(left_pad_model_inputs(inputs, pad_tokens=0, pad_token_id=0)["input_ids"].tolist(), [[5, 6, 7]])
        self.assertEqual(left_pad_model_inputs(inputs, pad_tokens=1, pad_token_id=None)["input_ids"].tolist(), [[0, 5, 6, 7]])

    def test_generation_options_validation(self) -> None:
        GenerationOptions().validate()
        GenerationOptions(do_sample=True, temperature=1.2, top_p=1.0, seed=3).validate()
        GenerationOptions(do_sample=True, top_k=0).validate()
        for bad in (dict(temperature=0.0), dict(top_p=1.5), dict(top_k=-1)):
            with self.assertRaises(ValueError):
                GenerationOptions(do_sample=True, **bad).validate()
        self.assertFalse(GenerationOptions(do_sample=True).truncates)
        self.assertFalse(GenerationOptions(do_sample=True, top_k=0).truncates)
        self.assertTrue(GenerationOptions(do_sample=True, top_p=0.95).truncates)
        self.assertTrue(GenerationOptions(do_sample=True, top_k=64).truncates)
        self.assertFalse(GenerationOptions(top_p=0.5).truncates)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_sampling_always_overrides_the_models_generation_defaults(self) -> None:
        from psse_env.dagger.release_factories import _ModelBundle
        from psse_env.dagger.research_policy_factory import _CanonicalResearchNativePolicy

        bundle = _ModelBundle(model=None, processor=None, model_id="m", model_revision="r")
        sampling = _CanonicalResearchNativePolicy(bundle, architecture="gemma4_unified",
                                                  generation=GenerationOptions(do_sample=True, seed=5))
        self.assertEqual(sampling._generation_kwargs(torch),
                         {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0})
        greedy = _CanonicalResearchNativePolicy(bundle, architecture="gemma4_unified")
        self.assertEqual(greedy._generation_kwargs(torch), {"do_sample": False, "temperature": 0.0})

    def test_shards_partition_the_roots(self) -> None:
        scenarios = [{"execution": {"scenario_id": f"r{i}"}} for i in range(5)]
        self.assertIsNone(parse_shard(None))
        self.assertEqual(parse_shard("1/2"), (1, 2))
        first = shard_scenarios(scenarios, (0, 2))
        second = shard_scenarios(scenarios, (1, 2))
        ids = lambda rows: [row["execution"]["scenario_id"] for row in rows]
        self.assertEqual(ids(first), ["r0", "r2", "r4"])
        self.assertEqual(ids(second), ["r1", "r3"])
        self.assertEqual(len(shard_scenarios(scenarios, None)), 5)
        for bad in ("2/2", "a/b", "-1/2", "0/0"):
            with self.assertRaises(ValueError):
                parse_shard(bad)


if __name__ == "__main__":
    unittest.main()
