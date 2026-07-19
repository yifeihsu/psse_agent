"""Command-line go/no-go gates and pilot LoRA entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import psse_env.dagger.dataset_builder as dataset_builder

from .gates import GateError, audit_dataset, load_exact_processor, load_jsonl, validate_grouped_pilot
from .provenance import build_gate_provenance, validate_generation_provenance
from .training import LoraSettings, TrainerSettings, run_lora_smoke, run_lora_training


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="unsloth/gemma-4-31B-it")
    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned 40-character Hugging Face commit used for both gate and training.",
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument(
        "--test",
        type=Path,
        help="Optional held-out split to include in schema/template/token audits.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--allow-download", action="store_true", help="Permit Hugging Face downloads; default is cache-only.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-prompt-truncation", action="store_true")
    parser.add_argument("--pilot-min-rows", type=int, default=32)
    parser.add_argument("--pilot-max-rows", type=int, default=128)
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional durable JSON gate report (recommended for HPC release evidence).",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Development-only: allow dirty source or non-release generation "
            "provenance; report remains non-release-eligible."
        ),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Pinned Gemma 4 tool-SFT go/no-go gates")
    commands = result.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="Run exact processor/template/mask/grouped-pilot gates.")
    _common(gate)
    train = commands.add_parser("train", help="Gate, smoke, then run pilot LoRA/TRL SFT.")
    _common(train)
    train.add_argument("--output-dir", default="outputs/dagger_gemma4_pilot")
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument(
        "--eval-strategy", choices=("epoch", "steps"), default="epoch"
    )
    train.add_argument(
        "--save-strategy", choices=("epoch", "steps"), default="epoch"
    )
    train.add_argument("--eval-steps", type=int, default=25)
    train.add_argument("--smoke-steps", type=int, default=1)
    train.add_argument("--load-in-4bit", action="store_true")
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--no-bf16", action="store_true")
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.0)
    smoke = commands.add_parser("smoke", help="Gate and run LoRA optimizer smoke only; never starts TRL training.")
    _common(smoke)
    smoke.add_argument("--mode", choices=("one-batch", "tiny-overfit"), required=True)
    smoke.add_argument("--tiny-overfit-steps", type=int, default=20)
    smoke.add_argument("--output-dir", default="outputs/dagger_gemma4_smoke")
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--gradient-accumulation-steps", type=int, default=1)
    smoke.add_argument("--learning-rate", type=float, default=1e-4)
    smoke.add_argument("--load-in-4bit", action="store_true", default=True)
    smoke.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    smoke.add_argument("--fp16", action="store_true")
    smoke.add_argument("--no-bf16", action="store_true")
    smoke.add_argument("--lora-rank", type=int, default=16)
    smoke.add_argument("--lora-alpha", type=int, default=16)
    smoke.add_argument("--lora-dropout", type=float, default=0.0)
    return result


def _gate_payload(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    test_rows = load_jsonl(args.test) if args.test is not None else []
    splits = {"train": train_rows, "validation": validation_rows}
    if test_rows:
        splits["test"] = test_rows
    all_rows = train_rows + validation_rows + test_rows
    grouped = validate_grouped_pilot(
        splits,
        group_key="physical_root_fingerprint",
        required_protocol="canonical",
        minimum_rows=args.pilot_min_rows,
        maximum_rows=args.pilot_max_rows,
    )
    processor, loader = load_exact_processor(
        args.model,
        args.revision,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    train_gate = audit_dataset(
        train_rows,
        processor,
        max_length=args.max_length,
        allow_prompt_truncation=args.allow_prompt_truncation,
        require_current_registry=True,
    )
    validation_gate = audit_dataset(
        validation_rows,
        processor,
        max_length=args.max_length,
        allow_prompt_truncation=args.allow_prompt_truncation,
        require_current_registry=True,
    )
    test_gate = (
        audit_dataset(
            test_rows,
            processor,
            max_length=args.max_length,
            allow_prompt_truncation=args.allow_prompt_truncation,
            require_current_registry=True,
        )
        if test_rows
        else None
    )
    data_passed = (
        grouped.passed
        and train_gate.passed
        and validation_gate.passed
        and (test_gate is None or test_gate.passed)
    )
    datasets = {
        **{"train": args.train, "validation": args.validation},
        **({"test": args.test} if args.test is not None else {}),
    }
    provenance = build_gate_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        processor_revision=args.revision,
        datasets=datasets,
        rows=all_rows,
        exporter_files=[dataset_builder.__file__, __file__],
    )
    generation = validate_generation_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        datasets=datasets,
        rows=all_rows,
    )
    source_passed = bool(provenance.get("release_eligible_source"))
    provenance_passed = source_passed and generation["passed"]
    passed = data_passed and (provenance_passed or args.allow_dirty_source)
    payload = {
        "passed": passed,
        "release_eligible": data_passed and provenance_passed,
        "source_gate_passed": source_passed,
        "provenance_gate_passed": provenance_passed,
        "provenance_gate_override": bool(
            args.allow_dirty_source and not provenance_passed
        ),
        "processor_loader": loader,
        "model": args.model,
        "revision": args.revision,
        "provenance": provenance,
        "generation_provenance": generation,
        "grouped_pilot": grouped.to_dict(),
        "train": train_gate.to_dict(),
        "validation": validation_gate.to_dict(),
    }
    if test_gate is not None:
        payload["test"] = test_gate.to_dict()
    return payload, passed


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "gate":
            payload, passed = _gate_payload(args)
            rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if args.report_output is not None:
                args.report_output.parent.mkdir(parents=True, exist_ok=True)
                args.report_output.write_text(rendered, encoding="utf-8")
            print(rendered, end="")
            return 0 if passed else 2
        settings = TrainerSettings(
            model_name=args.model,
            revision=args.revision,
            output_dir=args.output_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            epochs=getattr(args, "epochs", 1.0),
            max_steps=getattr(args, "max_steps", -1),
            eval_strategy=getattr(args, "eval_strategy", "epoch"),
            save_strategy=getattr(args, "save_strategy", "epoch"),
            eval_steps=getattr(args, "eval_steps", 25),
            bf16=not args.no_bf16 and not args.fp16,
            fp16=args.fp16,
            load_in_4bit=args.load_in_4bit,
            local_files_only=not args.allow_download,
            trust_remote_code=args.trust_remote_code,
            allow_prompt_truncation=args.allow_prompt_truncation,
            allow_nonrelease_artifacts=args.allow_dirty_source,
        )
        lora = LoraSettings(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        if args.command == "smoke":
            result = run_lora_smoke(
                train_file=args.train,
                validation_file=args.validation,
                settings=settings,
                lora=lora,
                mode=args.mode,
                pilot_minimum_rows=args.pilot_min_rows,
                pilot_maximum_rows=args.pilot_max_rows,
                tiny_overfit_steps=args.tiny_overfit_steps,
            )
            print(json.dumps({"passed": True, "mode": args.mode, "smoke": result.to_dict()}, indent=2))
            return 0
        result = run_lora_training(
            train_file=args.train,
            validation_file=args.validation,
            settings=settings,
            lora=lora,
            pilot_minimum_rows=args.pilot_min_rows,
            pilot_maximum_rows=args.pilot_max_rows,
            smoke_steps=args.smoke_steps,
        )
        metrics = getattr(result, "metrics", {})
        print(json.dumps({"passed": True, "training_metrics": metrics}, indent=2, default=str))
        return 0
    except (GateError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
