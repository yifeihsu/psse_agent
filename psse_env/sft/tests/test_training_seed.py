"""Focused coverage for explicit, fail-closed SFT training seeds."""

from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace
from unittest import mock

from psse_env.sft.gates import GateError
from psse_env.sft.training import (
    TrainerSettings,
    _seed_training_rngs,
    trl_config_kwargs,
)


PINNED_REVISION = "a" * 40
COMMON_TRAIN_ARGS = [
    "train",
    "--revision",
    PINNED_REVISION,
    "--train",
    "train.jsonl",
    "--validation",
    "validation.jsonl",
]


class TestTrainingSeed(unittest.TestCase):


    def test_trainer_settings_reject_invalid_direct_values(self) -> None:
        for value in (-1, 2**32, True, 1.5, "3407"):
            with self.subTest(value=value):
                with self.assertRaises(GateError):
                    TrainerSettings(
                        revision=PINNED_REVISION,
                        seed=value,  # type: ignore[arg-type]
                    ).validate()

    def test_seed_reaches_trl_configuration(self) -> None:
        settings = TrainerSettings(revision=PINNED_REVISION, seed=3409)
        settings.validate()
        self.assertEqual(trl_config_kwargs(settings, has_validation=True)["seed"], 3409)

    def test_runtime_seed_calls_every_engine_in_order(self) -> None:
        events: list[tuple[str, int]] = []
        fake_numpy = SimpleNamespace(
            random=SimpleNamespace(
                seed=lambda seed: events.append(("numpy_random", seed))
            )
        )
        fake_torch = SimpleNamespace(
            manual_seed=lambda seed: events.append(("torch_cpu", seed)),
            cuda=SimpleNamespace(
                manual_seed_all=lambda seed: events.append(("torch_cuda_all", seed))
            ),
        )
        with (
            mock.patch.dict(
                sys.modules,
                {"numpy": fake_numpy, "torch": fake_torch},
            ),
            mock.patch(
                "psse_env.sft.training.random.seed",
                side_effect=lambda seed: events.append(("python_random", seed)),
            ),
        ):
            attestation = _seed_training_rngs(3408)

        self.assertEqual(
            events,
            [
                ("python_random", 3408),
                ("numpy_random", 3408),
                ("torch_cpu", 3408),
                ("torch_cuda_all", 3408),
            ],
        )
        self.assertEqual(attestation["seed"], 3408)
        self.assertEqual(
            attestation["engines"],
            [name for name, _seed in events],
        )

    def test_cold_lora_initial_tensor_is_seed_deterministic(self) -> None:
        import torch

        def initial_tensor(seed: int):
            _seed_training_rngs(seed)
            # Simulate base-model construction consuming the Torch stream.
            torch.rand(31)
            # The cold-LoRA reset must make adapter initialization independent
            # of how much RNG the base-model loader consumed.
            _seed_training_rngs(seed)
            return torch.rand(16)

        first = initial_tensor(3407)
        repeated = initial_tensor(3407)
        different = initial_tensor(3408)

        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
