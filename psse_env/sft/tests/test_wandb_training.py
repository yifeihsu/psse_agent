"""Focused tests for the opt-in W&B Trainer integration."""

from __future__ import annotations

import unittest

from psse_env.sft.gates import GateError
from psse_env.sft.training import TrainerSettings, trl_config_kwargs


PINNED_REVISION = "a" * 40


class TestWandbTrainingConfiguration(unittest.TestCase):
    def test_defaults_disable_reporting(self) -> None:
        settings = TrainerSettings(revision=PINNED_REVISION)
        settings.validate()

        self.assertEqual(settings.report_to, "none")
        self.assertIsNone(settings.run_name)

        kwargs = trl_config_kwargs(settings, has_validation=True)
        self.assertEqual(kwargs["report_to"], "none")
        self.assertIsNone(kwargs["run_name"])


    def test_wandb_settings_are_forwarded_to_trl(self) -> None:
        settings = TrainerSettings(
            revision=PINNED_REVISION,
            report_to="wandb",
            run_name="bc0-round0",
        )
        settings.validate()

        kwargs = trl_config_kwargs(settings, has_validation=True)
        self.assertEqual(kwargs["report_to"], "wandb")
        self.assertEqual(kwargs["run_name"], "bc0-round0")


    def test_invalid_reporting_settings_are_rejected(self) -> None:
        invalid = (
            TrainerSettings(revision=PINNED_REVISION, report_to="tensorboard"),
            TrainerSettings(revision=PINNED_REVISION, report_to=None),  # type: ignore[arg-type]
            TrainerSettings(revision=PINNED_REVISION, run_name=""),
            TrainerSettings(revision=PINNED_REVISION, run_name="   "),
        )
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(GateError):
                    settings.validate()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
