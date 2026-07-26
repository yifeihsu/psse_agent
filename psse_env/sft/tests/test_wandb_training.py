"""Focused tests for the opt-in W&B Trainer integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from psse_env.sft.cli import main as cli_main
from psse_env.sft.cli import parser
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

    def test_train_cli_defaults_and_wandb_options(self) -> None:
        common = [
            "train",
            "--revision",
            PINNED_REVISION,
            "--train",
            "train.jsonl",
            "--validation",
            "validation.jsonl",
        ]

        defaults = parser().parse_args(common)
        self.assertEqual(defaults.report_to, "none")
        self.assertIsNone(defaults.run_name)

        enabled = parser().parse_args(
            [*common, "--report-to", "wandb", "--run-name", "bc0-round0"]
        )
        self.assertEqual(enabled.report_to, "wandb")
        self.assertEqual(enabled.run_name, "bc0-round0")

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

    def test_train_cli_wires_wandb_settings_into_training(self) -> None:
        with (
            mock.patch(
                "psse_env.sft.cli._baseline_evaluation_gate",
                return_value={},
            ),
            mock.patch(
                "psse_env.sft.cli.run_lora_training",
                return_value=SimpleNamespace(metrics={"train_loss": 1.0}),
            ) as run_training,
        ):
            result = cli_main(
                [
                    "train",
                    "--revision",
                    PINNED_REVISION,
                    "--train",
                    "train.jsonl",
                    "--validation",
                    "validation.jsonl",
                    "--report-to",
                    "wandb",
                    "--run-name",
                    "bc0-round0",
                ]
            )

        self.assertEqual(result, 0)
        settings = run_training.call_args.kwargs["settings"]
        self.assertEqual(settings.report_to, "wandb")
        self.assertEqual(settings.run_name, "bc0-round0")

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

    def test_gate_and_smoke_commands_do_not_expose_wandb_options(self) -> None:
        gate = parser().parse_args(
            [
                "gate",
                "--revision",
                PINNED_REVISION,
                "--train",
                "train.jsonl",
                "--validation",
                "validation.jsonl",
            ]
        )
        smoke = parser().parse_args(
            [
                "smoke",
                "--revision",
                PINNED_REVISION,
                "--train",
                "train.jsonl",
                "--validation",
                "validation.jsonl",
                "--mode",
                "one-batch",
            ]
        )

        self.assertFalse(hasattr(gate, "report_to"))
        self.assertFalse(hasattr(gate, "run_name"))
        self.assertFalse(hasattr(smoke, "report_to"))
        self.assertFalse(hasattr(smoke, "run_name"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
