"""Stage 0: the update alone, on one fixed rollout batch, under each schedule (workplan of 2026-11-10).

    python -m research.hpc.rl_improvement_20261110.calibration prepare --config $WORK/configs/calibration_1step.json \\
        --batch /scratch/yx3882/compound_rl_20261107/out/ieee14_v1/iteration_000/trajectories
    sbatch --export=ALL,CONFIG=calibration_1step.json,ITERATIONS=1,TARGET_ITERATIONS=1 --job-name=rli-cal1 rl.sbatch
    python -m research.hpc.rl_improvement_20261110.calibration report \\
        --run $WORK/out/calibration_1step --run $WORK/out/calibration_2chunk --run $WORK/out/calibration_4chunk \\
        --output $WORK/out/calibration_report

``prepare`` makes a run that starts at its update: the batch (an
iteration's ``trajectories`` directory, copied) becomes the run's iteration
0, and the manifest is written at the ``update`` phase with the adapter that
sampled the batch.  The driver then scores the groups, builds the rows and
takes the config's update, exactly as an arm's first update would; nothing
is rolled out and nothing is evaluated.  The batch is the first arm's
iteration 0: S0's samples on the roots arm A draws first, so the three runs
show what A's first update does under one step of a new optimizer, two
whole-root steps and four.

``report`` puts the runs side by side from their update reports: the steps
taken, how far each moved the weights (in units of the learning rate), the
gradient by loss term, the share of rows the row bound clipped, the rows
that carry a step, and what the model gives the lifecycle choices on the
batch's own decisions at each step.  The fixed decision panel of the
workplan is the fork probe's (``research.rl.fork_probe decide`` before and
after, then ``panel``); this report is the batch's own view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CALIBRATION_CONTRACT = "rl_update_calibration_v1"
RECORD_FILE = "calibration.json"


def batch_listing(directory: Path) -> dict[str, Any]:
    """The group files of a trajectory store, with one digest over their names and contents."""

    groups = sorted((Path(directory) / "groups").glob("*.jsonl"))
    digest = hashlib.sha256()
    episodes = 0
    for path in groups:
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
        episodes += sum(1 for line in data.splitlines() if line.strip())
    return {"groups": len(groups), "episodes": episodes, "sha256": digest.hexdigest()}


def prepare(config_path: Path, batch: Path, *, adapter: str | None = None) -> dict[str, Any]:
    """Make ``config_path``'s run start at the update of ``batch``; refuses a run that already exists."""

    from psse_env.rl.train import Manifest, RLConfig, iteration_dir

    config = RLConfig.load(config_path)
    output = Path(config.output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{output} already holds a run (manifest.json); a calibration starts in an empty directory")
    listing = batch_listing(batch)
    if not listing["groups"]:
        raise SystemExit(f"{batch} holds no rollout groups (expected <batch>/groups/*.jsonl)")
    target = iteration_dir(config, 0) / "trajectories"
    if target.exists():
        raise SystemExit(f"{target} exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(batch, target)
    copied = batch_listing(target)
    if copied != listing:
        raise SystemExit(f"the copy of the batch differs from the batch: {copied} against {listing}")
    sampled_by = str(adapter or config.initial_adapter)
    Manifest(iteration=0, phase="update", current_adapter=sampled_by).save(manifest_path)
    record = {
        "contract": CALIBRATION_CONTRACT,
        "prepared_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(config_path),
        "batch": {"source": str(batch), **listing},
        "sampled_by": sampled_by,
        "update": dict(config.update),
        "note": "the run starts at the update phase of iteration 0; its rollouts are the batch, not its own",
    }
    (output / RECORD_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return record


# --------------------------------------------------------------------------- the report


def _step_view(step: Mapping[str, Any]) -> dict[str, Any]:
    movement = step.get("movement") or {}
    decisions = step.get("decisions") or {}
    components = step.get("components") or {}
    return {
        "rows": step.get("rows"),
        "rows_clipped_fraction": step.get("rows_clipped_fraction"),
        "gradient_norm_before_global_clip": step.get("gradient_norm_before_global_clip"),
        "global_clip_scale": step.get("global_clip_scale"),
        "top_rows_squared_norm_share": step.get("top_rows_squared_norm_share"),
        "movement_rms_over_learning_rate": movement.get("rms_over_learning_rate"),
        "movement_relative_to_weight_norm": movement.get("relative_to_weight_norm"),
        "cosine_with_descent": movement.get("cosine_with_descent"),
        "components": {name: {"rows": part.get("rows"), "share_of_step": part.get("share_of_step")} for name, part in components.items()},
        "decisions": decisions,
    }


def run_view(directory: Path) -> dict[str, Any]:
    """What one calibration run did, from its update report."""

    report_path = Path(directory) / "iteration_000" / "update_report.json"
    record_path = Path(directory) / RECORD_FILE
    view: dict[str, Any] = {"run": str(directory), "finished": report_path.is_file()}
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        view["batch_sha256"] = (record.get("batch") or {}).get("sha256")
        view["sampled_by"] = record.get("sampled_by")
    if not report_path.is_file():
        return view
    report = json.loads(report_path.read_text(encoding="utf-8"))
    update = report.get("update") or {}
    settings = update.get("settings") or {}
    first_check = (update.get("on_policy_check") or [{}])[0]
    view.update(
        schedule={key: settings.get(key) for key in ("optimizer_steps", "fresh_optimizer_steps", "chunk_by", "carry_optimizer",
                                                     "learning_rate", "loss_normalization", "max_row_grad_norm", "beta_kl")},
        optimizer_steps=update.get("optimizer_steps"), planned_steps=update.get("planned_steps"),
        fresh_optimizer=update.get("fresh_optimizer"), rows=update.get("rows"), batch_digest=update.get("batch_digest"),
        rows_by_kind=(update.get("row_accounting") or {}).get("rows_by_kind"),
        each_row_once=(update.get("row_accounting") or {}).get("each_row_once"),
        split_groups=update.get("split_groups"),
        first_step_mean_abs_log_ratio=first_check.get("mean_abs_log_ratio"),
        policy_loss=update.get("policy_loss"), kl=update.get("kl"), grad_norm=update.get("grad_norm"),
        integrity=report.get("integrity"),
        steps=[_step_view(step) for step in update.get("diagnostics") or []],
    )
    return view


def build_report(directories: Sequence[Path]) -> dict[str, Any]:
    runs = [run_view(Path(directory)) for directory in directories]
    digests = {run.get("batch_digest") for run in runs if run.get("batch_digest")}
    return {
        "contract": CALIBRATION_CONTRACT,
        "runs": runs,
        # The runs are comparable only on one batch: the update's own digest of its rows says so.
        "same_batch": len(digests) <= 1 and all(run["finished"] for run in runs),
        "batch_digests": sorted(digests),
    }


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _group(decisions: Mapping[str, Any], name: str, key: str) -> Any:
    group = decisions.get(name) if isinstance(decisions.get(name), Mapping) else {}
    return group.get(key)


def markdown(report: Mapping[str, Any], *, title: str = "Update calibration on the fixed batch") -> str:
    lines = [f"# {title}", ""]
    if not report["same_batch"]:
        lines += ["**The runs are not on one finished batch; the table below does not compare like with like.** "
                  f"Batch digests: {report['batch_digests'] or 'none'}.", ""]
    lines += ["| Run | Steps taken (planned) | New optimizer | Rows | Each row once | First-step gap | Integrity |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        name = Path(run["run"]).name
        if not run["finished"]:
            lines.append(f"| {name} | not finished | | | | | |")
            continue
        integrity = run.get("integrity") or {}
        lines.append(f"| {name} | {run['optimizer_steps']} ({run['planned_steps']}) | {_number(run['fresh_optimizer'])} | {run['rows']} | "
                     f"{_number(run['each_row_once'])} | {_number(run['first_step_mean_abs_log_ratio'], 4)} | "
                     f"{'passed' if integrity.get('ok') else 'FAILED: ' + '; '.join(integrity.get('failures') or [])} |")
    lines += ["", "## Each optimizer step", "",
              "Movement is the root mean square change of a trainable weight divided by the learning rate: 1.0 is a step that "
              "moves every weight by the learning rate, which is what the first step of a new AdamW does whatever the gradient. "
              "The decline probabilities are on the batch's own labelled decisions: true targets at an episode's first "
              "accepted candidate and at later ones, and wrong targets. AUC is the chance a wrong target gets a higher decline "
              "probability than a true one.", "",
              "| Run | Step | Rows | Clipped by the row bound | Movement / lr | Cosine with descent | Policy share | KL share | "
              "Supervised share | p(decline) true, first | p(decline) true, later | p(decline) wrong | AUC |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        for number, step in enumerate(run.get("steps") or []):
            share = lambda name: _number((step["components"].get(name) or {}).get("share_of_step"))  # noqa: E731
            decisions = step.get("decisions") or {}
            lines.append(
                f"| {Path(run['run']).name} | {number + 1} | {step['rows']} | {_number(step['rows_clipped_fraction'])} | "
                f"{_number(step['movement_rms_over_learning_rate'])} | {_number(step['cosine_with_descent'])} | {share('policy')} | "
                f"{share('kl')} | {share('supervised')} | "
                f"{_number(_group(decisions, 'true_target_first_candidate', 'mean_p_decline'), 4)} | "
                f"{_number(_group(decisions, 'true_target_later_candidate', 'mean_p_decline'), 4)} | "
                f"{_number(_group(decisions, 'wrong_target', 'mean_p_decline'), 4)} | {_number(decisions.get('auc_wrong_above_true'))} |")
    lines += ["", "A schedule passes stage 0 when its updates are finite and accounted for, the first step is on policy, and the "
              "decline probability moves apart for wrong and true targets (on this batch and on the fork probe's panel) rather "
              "than up for both.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    preparer = commands.add_parser("prepare", help="start a run at the update of a fixed rollout batch")
    preparer.add_argument("--config", required=True, type=Path)
    preparer.add_argument("--batch", required=True, type=Path, help="an iteration's trajectories directory")
    preparer.add_argument("--adapter", default=None, help="the adapter that sampled the batch (default: the config's start adapter)")
    reporter = commands.add_parser("report", help="the calibration runs side by side")
    reporter.add_argument("--run", action="append", required=True, type=Path, help="a calibration run's output directory (repeatable)")
    reporter.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.config, args.batch, adapter=args.adapter), sort_keys=True))
        return 0
    report = build_report(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    text = markdown(report)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0 if report["same_batch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
