"""Command-line go/no-go gates and pilot LoRA entrypoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import psse_env.dagger.dataset_builder as dataset_builder
import psse_env.dagger.evaluation_gate as evaluation_gate
from psse_env.dagger.evaluation_gate import (
    DEFAULT_POLICY_ID,
    DEFAULT_POLICY_PATH,
    current_registry_sha256,
    validate_evaluation_artifact,
)

from .gates import GateError, audit_dataset, load_exact_processor, load_jsonl, validate_grouped_pilot
from .provenance import (
    build_gate_provenance,
    file_sha256,
    git_source_state,
    stable_json_sha256,
    validate_generation_provenance,
)
from .training import (
    LoraSettings,
    TrainerSettings,
    run_lora_smoke,
    run_lora_training,
    run_targeted_lora_smoke_sweep,
    validate_training_seed,
)
from psse_env.sft.historical_expert_closure import (
    validate_historical_expert_closure,
)
from psse_env.dagger.study_manifest import (
    DEFAULT_STUDY_MANIFEST,
    TRAINED_VARIANT_IDS,
    load_study_manifest,
)


def _training_seed(value: str) -> int:
    """Parse a training seed before any expensive pretraining gate runs."""

    try:
        seed = int(value)
        validate_training_seed(seed)
    except (ValueError, GateError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return seed


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
    parser.add_argument(
        "--require-auto-processor",
        action="store_true",
        help=(
            "Fail if the pinned model loads only through AutoTokenizer. "
            "Required by the BC0 release launcher."
        ),
    )
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


def _initial_adapter_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--initial-adapter-path",
        type=Path,
        help=(
            "Optional absolute local LoRA adapter path used as the trainable "
            "warm start. Requires --initial-adapter-revision."
        ),
    )
    parser.add_argument(
        "--initial-adapter-revision",
        help=(
            "Expected immutable 64-hex checkpoint tree SHA-256 for the initial "
            "adapter. Requires --initial-adapter-path."
        ),
    )


def _round1_source_options(
    parser: argparse.ArgumentParser,
    *,
    include_initial_adapter_revision: bool = False,
) -> None:
    parser.add_argument(
        "--round1-provenance",
        type=Path,
        help=(
            "Immutable aggregate.generation_provenance.json authenticated before "
            "any auxiliary recovery-probe row may enter SFT."
        ),
    )
    parser.add_argument(
        "--parent-checkpoint-receipt",
        type=Path,
        help=(
            "Canonical checkpoint_receipt.json for the same-seed BC0 adapter. "
            "Required for Round-1 training and validated before model allocation."
        ),
    )
    parser.add_argument(
        "--round1-preflight",
        type=Path,
        help=(
            "Immutable aggregate.preflight.json paired with --round1-provenance."
        ),
    )
    parser.add_argument(
        "--reviewed-source-commit",
        help="Externally reviewed 40-hex source commit for the Round-1 source gate.",
    )
    parser.add_argument(
        "--round1-view",
        choices=("full", "natural-only"),
        help=(
            "Explicit immutable Round-1 training view. Use full for "
            "natural_dagger_probes and natural-only for natural_dagger."
        ),
    )
    if include_initial_adapter_revision:
        parser.add_argument(
            "--initial-adapter-revision",
            help=(
                "Immutable 64-hex learner-seed adapter revision required to "
                "validate a Round-1 aggregate."
            ),
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Pinned Gemma 4 tool-SFT go/no-go gates")
    commands = result.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="Run exact processor/template/mask/grouped-pilot gates.")
    _common(gate)
    _round1_source_options(gate, include_initial_adapter_revision=True)
    train = commands.add_parser("train", help="Gate, smoke, then run pilot LoRA/TRL SFT.")
    _common(train)
    _initial_adapter_options(train)
    _round1_source_options(train)
    train.add_argument("--output-dir", default="outputs/dagger_gemma4_pilot")
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument(
        "--optimizer",
        choices=("adamw_torch",),
        default="adamw_torch",
    )
    train.add_argument(
        "--lr-scheduler-type",
        choices=("linear",),
        default="linear",
    )
    train.add_argument(
        "--seed",
        type=_training_seed,
        required=True,
        help=(
            "Explicit deterministic training seed (0..4294967295). "
            "Preregister a distinct value for each replicated training run."
        ),
    )
    train.add_argument(
        "--study-variant",
        choices=TRAINED_VARIANT_IDS,
        help=(
            "Preregistered trained variant. If omitted, cold-start D0 is BC0; "
            "warm-start variants are inferred only from an exact source set."
        ),
    )
    train.add_argument(
        "--study-manifest",
        type=Path,
        default=DEFAULT_STUDY_MANIFEST,
        help="Byte-pinned four-variant study manifest bound into the checkpoint receipt.",
    )
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
    train.add_argument(
        "--report-to",
        choices=("none", "wandb"),
        default="none",
        help=(
            "Optional Trainer metrics integration. W&B is disabled by default; "
            "select 'wandb' only when the batch environment is configured."
        ),
    )
    train.add_argument(
        "--run-name",
        help="Optional run name forwarded to the Trainer integration.",
    )
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.0)
    train.add_argument(
        "--expert-baseline-evaluation",
        type=Path,
        help="Release evaluator artifact for the observable expert baseline.",
    )
    train.add_argument(
        "--expert-baseline-gate-receipt",
        type=Path,
        help=(
            "Exact immutable dual-source gate receipt authorizing canonical "
            "reuse of the historical observable-expert baseline."
        ),
    )
    train.add_argument(
        "--base-baseline-evaluation",
        type=Path,
        help="Release evaluator artifact for the exact pinned base model.",
    )
    train.add_argument(
        "--evaluation-suite",
        type=Path,
        help="Frozen JSON suite used by both required baseline evaluations.",
    )
    train.add_argument(
        "--evaluation-policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Versioned hard-constraint policy used for both baselines.",
    )
    train.add_argument(
        "--expert-policy-identity",
        help="Exact immutable explicit identity recorded by the expert artifact.",
    )
    train.add_argument(
        "--baseline-evaluation-report-output",
        type=Path,
        help="Optional durable JSON report for the two pretraining gates.",
    )
    smoke = commands.add_parser("smoke", help="Gate and run LoRA optimizer smoke only; never starts TRL training.")
    _common(smoke)
    _initial_adapter_options(smoke)
    _round1_source_options(smoke)
    smoke.add_argument("--mode", choices=("one-batch", "tiny-overfit"), required=True)
    smoke.add_argument("--tiny-overfit-steps", type=int, default=20)
    smoke.add_argument(
        "--targeted-recovery-sweep",
        action="store_true",
        help=(
            "Run the deterministic five-case recovery tiny-overfit at the fixed "
            "diagnostic LR sweep 1e-4, 3e-4, and 1e-3."
        ),
    )
    smoke.add_argument(
        "--targeted-min-relative-loss-reduction",
        type=float,
        default=0.20,
        help="Minimum mean loss reduction required on the complete five-case slice.",
    )
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


def _round1_source_report_for_gate(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    rows = [*train_rows, *validation_rows, *test_rows]
    has_ineligible_rows = any(
        row.get("production_label_eligible") is not True for row in rows
    )
    from .round1_source_gate import (
        round1_source_binding_required,
        validate_round1_source_mix_gate,
    )

    dataset_paths = [args.train, args.validation]
    if args.test is not None:
        dataset_paths.append(args.test)
    path_requires_binding = round1_source_binding_required(*dataset_paths)
    values = (
        getattr(args, "round1_provenance", None),
        getattr(args, "round1_preflight", None),
        getattr(args, "reviewed_source_commit", None),
        getattr(args, "initial_adapter_revision", None),
        getattr(args, "round1_view", None),
    )
    if any(values) != all(values):
        raise GateError(
            "Round-1 data gate requires --round1-provenance, --round1-preflight, "
            "--reviewed-source-commit, --initial-adapter-revision, and "
            "--round1-view together."
        )
    if not all(values):
        if has_ineligible_rows or path_requires_binding:
            raise GateError(
                "Round-1 or non-production-label SFT rows require the complete "
                "Round-1 source-gate arguments."
            )
        return None
    if args.test is None:
        raise GateError(
            "Round-1 SFT data gate requires the canonical aggregate test split."
        )

    report = validate_round1_source_mix_gate(
        args.round1_provenance,
        args.round1_preflight,
        reviewed_source_commit=str(args.reviewed_source_commit).lower(),
        initial_adapter_revision=str(args.initial_adapter_revision).lower(),
        round1_view=str(args.round1_view),
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
    )
    provenance_id = report.get("generation_provenance_id")
    if report.get("passed") is not True or not isinstance(
        provenance_id, str
    ) or re.fullmatch(r"[0-9a-f]{64}", provenance_id) is None:
        raise GateError(
            "Round-1 source gate did not return a valid generation provenance ID."
        )
    expected_content = report.get("canonical_dataset_content_sha256")
    actual_content = {
        "train": stable_json_sha256(train_rows),
        "validation": stable_json_sha256(validation_rows),
        "test": stable_json_sha256(test_rows),
    }
    if not isinstance(expected_content, dict) or expected_content != actual_content:
        raise GateError(
            "Round-1 source gate authenticated dataset bytes different from "
            "the rows already loaded by the SFT data gate."
        )
    return dict(report)


def _gate_payload(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    test_rows = load_jsonl(args.test) if args.test is not None else []
    splits = {"train": train_rows, "validation": validation_rows}
    if test_rows:
        splits["test"] = test_rows
    all_rows = train_rows + validation_rows + test_rows
    round1_source_report = _round1_source_report_for_gate(
        args,
        train_rows,
        validation_rows,
        test_rows,
    )
    grouped = validate_grouped_pilot(
        splits,
        group_key="physical_root_fingerprint",
        required_protocol="canonical",
        minimum_rows=args.pilot_min_rows,
        maximum_rows=args.pilot_max_rows,
        validated_round1_generation_provenance_id=(
            round1_source_report.get("generation_provenance_id")
            if round1_source_report is not None
            else None
        ),
    )
    processor, loader = load_exact_processor(
        args.model,
        args.revision,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    processor_loader_passed = (
        not args.require_auto_processor or loader == "AutoProcessor"
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
        processor_loader_passed
        and grouped.passed
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
        "processor_loader_requirement": (
            "AutoProcessor" if args.require_auto_processor else None
        ),
        "processor_loader_passed": processor_loader_passed,
        "model": args.model,
        "revision": args.revision,
        "max_length": args.max_length,
        "provenance": provenance,
        "generation_provenance": generation,
        "round1_source_gate": round1_source_report,
        "grouped_pilot": grouped.to_dict(),
        "train": train_gate.to_dict(),
        "validation": validation_gate.to_dict(),
    }
    if test_gate is not None:
        payload["test"] = test_gate.to_dict()
    return payload, passed


def _baseline_evaluation_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate both frozen-suite baselines before any model is loaded."""

    required = {
        "expert_baseline_evaluation": args.expert_baseline_evaluation,
        "base_baseline_evaluation": args.base_baseline_evaluation,
        "evaluation_suite": args.evaluation_suite,
        "expert_policy_identity": args.expert_policy_identity,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise GateError(
            "BC0 training requires frozen-suite expert and base-model evaluation "
            "artifacts; missing: " + ", ".join(missing)
        )
    paths = (
        args.expert_baseline_evaluation,
        args.base_baseline_evaluation,
        args.evaluation_suite,
        args.evaluation_policy,
    )
    missing_paths = [str(path) for path in paths if not Path(path).is_file()]
    if missing_paths:
        raise GateError(
            "BC0 baseline evaluation inputs are missing: " + ", ".join(missing_paths)
        )
    repo_root = Path(__file__).resolve().parents[2]
    current_source = git_source_state(repo_root)
    source_commit = str(current_source.get("source_commit") or "")
    if current_source.get("release_eligible_source") is not True or not source_commit:
        raise GateError(
            "BC0 training requires the current source to be a clean tracked commit."
        )
    suite_hash = file_sha256(args.evaluation_suite)
    registry_hash = current_registry_sha256("canonical")
    common = {
        "policy": args.evaluation_policy,
        "expected_source_commit": source_commit,
        "expected_suite_path": args.evaluation_suite,
        "expected_protocol": "canonical",
        "expected_registry_sha256": registry_hash,
        "required_gate_policy_id": DEFAULT_POLICY_ID,
        "repo_root": repo_root,
        "require_current_clean_source": True,
    }
    expert_receipt_path = getattr(args, "expert_baseline_gate_receipt", None)
    if expert_receipt_path is not None:
        expert_payload = validate_historical_expert_closure(
            expert_receipt_path,
            expert_artifact_path=args.expert_baseline_evaluation,
            repo_root=repo_root,
            expected_suite_path=args.evaluation_suite,
            expected_policy_path=args.evaluation_policy,
            expected_policy_identity=args.expert_policy_identity,
            expected_protocol="canonical",
            expected_registry_sha256=registry_hash,
        )
        expert_facts = expert_payload.get("expert", {})
        artifact_facts = expert_payload.get("artifact", {})
        expert_passed = expert_payload.get("passed") is True
        expert_performance_passed = (
            expert_facts.get("performance_passed", expert_passed) is True
        )
        expert_failures = [
            str(failure) for failure in expert_payload.get("failures", [])
        ]
        expert_artifact_source_commit = str(
            artifact_facts.get("source_commit") or ""
        )
        expert_validator_source_commit = str(
            expert_payload.get("validator_source_commit")
            or expert_payload.get("expert_validator_source_commit")
            or expert_facts.get("validator_source_commit")
            or ""
        )
        expert_validation_mode = "historical-closure-reuse"
    else:
        expert = validate_evaluation_artifact(
            args.expert_baseline_evaluation,
            role="expert-baseline",
            expected_policy_identity=args.expert_policy_identity,
            **common,
        )
        expert_payload = expert.as_dict()
        expert_passed = expert.passed
        expert_performance_passed = expert.performance_passed
        expert_failures = list(expert.failures)
        expert_artifact_source_commit = str(expert.source_commit or "")
        expert_validator_source_commit = source_commit
        expert_validation_mode = "single-source-validation"
    base = validate_evaluation_artifact(
        args.base_baseline_evaluation,
        role="base-baseline",
        expected_model_id=args.model,
        expected_model_revision=args.revision,
        **common,
    )
    blocking_failures = [
        *(f"expert: {failure}" for failure in expert_failures),
        *(f"base evidence: {failure}" for failure in base.evidence_failures),
    ]
    if not expert_passed and not expert_failures:
        blocking_failures.append("expert: full evaluation gate did not pass")
    if not base.evidence_passed and not base.evidence_failures:
        blocking_failures.append("base evidence: evaluation evidence gate did not pass")
    base_performance_findings = [
        f"base performance: {failure}" for failure in base.performance_failures
    ]
    pretraining_gate_passed = expert_passed and base.evidence_passed
    all_baselines_performance_qualified = (
        expert_performance_passed and base.performance_passed
    )
    payload = {
        "passed": pretraining_gate_passed,
        "pretraining_gate_passed": pretraining_gate_passed,
        "expert_full_gate_passed": expert_passed,
        "base_evidence_gate_passed": base.evidence_passed,
        "all_baselines_performance_qualified": all_baselines_performance_qualified,
        # Retain the old field as a strict performance-qualification signal.  A
        # weak but reproducibly measured base may unblock training, but it is
        # never described as release-qualified.
        "release_eligible": (
            pretraining_gate_passed and all_baselines_performance_qualified
        ),
        "override": False,
        "failures": blocking_failures,
        "base_performance_findings": base_performance_findings,
        "source_commit": source_commit,
        "consumer_source_commit": source_commit,
        "base_artifact_source_commit": str(base.source_commit or ""),
        "base_expected_source_commit": source_commit,
        "expert_artifact_source_commit": expert_artifact_source_commit,
        "expert_validator_source_commit": expert_validator_source_commit,
        "expert_validation_mode": expert_validation_mode,
        "frozen_suite_sha256": suite_hash,
        "protocol": "canonical",
        "registry_sha256": registry_hash,
        "expert": expert_payload,
        "base": base.as_dict(),
    }
    if args.baseline_evaluation_report_output is not None:
        report_target = evaluation_gate._prepare_report_output(
            args.baseline_evaluation_report_output,
            repo_root=repo_root,
            protected_inputs=(
                args.expert_baseline_evaluation,
                expert_receipt_path,
                args.base_baseline_evaluation,
                args.evaluation_suite,
                args.evaluation_policy,
                Path(__file__),
            ),
        )
        report_identity = evaluation_gate._publish_new_report(
            report_target,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        publication_failures: list[str] = []
        try:
            if expert_receipt_path is not None:
                fresh_expert_payload = validate_historical_expert_closure(
                    expert_receipt_path,
                    expert_artifact_path=args.expert_baseline_evaluation,
                    repo_root=repo_root,
                    expected_suite_path=args.evaluation_suite,
                    expected_policy_path=args.evaluation_policy,
                    expected_policy_identity=args.expert_policy_identity,
                    expected_protocol="canonical",
                    expected_registry_sha256=registry_hash,
                )
            else:
                fresh_expert_payload = validate_evaluation_artifact(
                    args.expert_baseline_evaluation,
                    role="expert-baseline",
                    expected_policy_identity=args.expert_policy_identity,
                    **common,
                ).as_dict()
            if fresh_expert_payload != expert_payload:
                publication_failures.append(
                    "historical expert closure or expert evidence changed during "
                    "baseline report publication"
                )
            fresh_base_payload = validate_evaluation_artifact(
                args.base_baseline_evaluation,
                role="base-baseline",
                expected_model_id=args.model,
                expected_model_revision=args.revision,
                **common,
            ).as_dict()
            if fresh_base_payload != base.as_dict():
                publication_failures.append(
                    "Base evidence changed during baseline report publication"
                )
            if git_source_state(repo_root) != current_source:
                publication_failures.append(
                    "BC0 baseline source changed during report publication"
                )
        except Exception as exc:
            publication_failures.append(
                "baseline report post-publication re-attestation failed: "
                f"{type(exc).__name__}: {exc}"
            )
        if publication_failures:
            removed = evaluation_gate._unlink_created_report(
                report_target,
                report_identity,
            )
            message = "; ".join(publication_failures)
            if not removed:
                message += "; the newly created report could not be safely removed"
            raise GateError(message)
    if not pretraining_gate_passed:
        raise GateError(
            "BC0 baseline closed-loop evaluation gate failed: "
            + "; ".join(blocking_failures)
        )
    return payload


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
        if args.command == "train":
            # Reject protocol drift before baseline artifact validation or any
            # model/runtime initialization.
            load_study_manifest(args.study_manifest)
        baseline_evaluation_gate = (
            _baseline_evaluation_gate(args) if args.command == "train" else None
        )
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
            optimizer=getattr(args, "optimizer", "adamw_torch"),
            lr_scheduler_type=getattr(args, "lr_scheduler_type", "linear"),
            eval_strategy=getattr(args, "eval_strategy", "epoch"),
            save_strategy=getattr(args, "save_strategy", "epoch"),
            eval_steps=getattr(args, "eval_steps", 25),
            seed=getattr(args, "seed", 3407),
            bf16=not args.no_bf16 and not args.fp16,
            fp16=args.fp16,
            load_in_4bit=args.load_in_4bit,
            local_files_only=not args.allow_download,
            trust_remote_code=args.trust_remote_code,
            allow_prompt_truncation=args.allow_prompt_truncation,
            allow_nonrelease_artifacts=args.allow_dirty_source,
            required_processor_loader=(
                "AutoProcessor"
                if args.command == "train" or args.require_auto_processor
                else None
            ),
            report_to=getattr(args, "report_to", "none"),
            run_name=getattr(args, "run_name", None),
            initial_adapter_path=(
                str(args.initial_adapter_path)
                if getattr(args, "initial_adapter_path", None) is not None
                else None
            ),
            initial_adapter_revision=getattr(
                args,
                "initial_adapter_revision",
                None,
            ),
            parent_checkpoint_receipt_path=(
                str(args.parent_checkpoint_receipt)
                if getattr(args, "parent_checkpoint_receipt", None) is not None
                else None
            ),
            round1_provenance_path=(
                str(args.round1_provenance)
                if getattr(args, "round1_provenance", None) is not None
                else None
            ),
            round1_preflight_path=(
                str(args.round1_preflight)
                if getattr(args, "round1_preflight", None) is not None
                else None
            ),
            reviewed_source_commit=getattr(
                args,
                "reviewed_source_commit",
                None,
            ),
            round1_view=getattr(args, "round1_view", None),
            study_variant=getattr(args, "study_variant", None),
            study_manifest_path=(
                str(args.study_manifest)
                if getattr(args, "study_manifest", None) is not None
                else None
            ),
        )
        lora = LoraSettings(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        if args.command == "smoke":
            if args.targeted_recovery_sweep:
                if args.mode != "tiny-overfit":
                    raise ValueError(
                        "--targeted-recovery-sweep requires --mode tiny-overfit."
                    )
                result = run_targeted_lora_smoke_sweep(
                    train_file=args.train,
                    validation_file=args.validation,
                    settings=settings,
                    lora=lora,
                    pilot_minimum_rows=args.pilot_min_rows,
                    pilot_maximum_rows=args.pilot_max_rows,
                    tiny_overfit_steps=args.tiny_overfit_steps,
                    minimum_relative_loss_reduction=(
                        args.targeted_min_relative_loss_reduction
                    ),
                )
                payload = {
                    "passed": result.passed,
                    "mode": "targeted-tiny-overfit-sweep",
                    "smoke": result.to_dict(),
                }
            else:
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
                payload = {
                    "passed": True,
                    "mode": args.mode,
                    "smoke": result.to_dict(),
                }
            rendered = json.dumps(payload, indent=2) + "\n"
            if args.report_output is not None:
                args.report_output.parent.mkdir(parents=True, exist_ok=True)
                args.report_output.write_text(rendered, encoding="utf-8")
            print(rendered, end="")
            return 0 if payload["passed"] else 2
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
        print(
            json.dumps(
                {
                    "passed": True,
                    "baseline_evaluation_gate": baseline_evaluation_gate,
                    "training_metrics": metrics,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    except (GateError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
