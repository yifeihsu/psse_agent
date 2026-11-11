"""What stops an experiment at once (workplan of 2026-11-10, "Evaluation and stopping rules").

    Stop immediately on nonfinite updates, incorrect likelihood provenance,
    silent history truncation, duplicated or missing update work, or invalid
    checkpoint recovery.  A preemption is neither a completed iteration nor
    an algorithmic failure.

:func:`update_integrity` reads an update report (``update_report.json`` of
the driver) and names every one of those conditions the report shows.  A run
whose config sets ``stop_on_integrity_failure`` does not advance past an
update that fails: the driver leaves a marker beside the report and refuses
the phase until someone has looked (:class:`IntegrityError`).

What the report cannot show is caught where it happens and raises there: a
checkpoint continued on another batch, in another order of the rows or under
other settings (the batch digest, the plan digest and the settings a
checkpoint of :mod:`psse_env.rl.update` records), a carried optimizer state
for other parameters,
a first step that is off policy beyond the tolerance, continuation values
sampled under another adapter (:func:`psse_env.rl.decision_credit.check_provenance`).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

INTEGRITY_CONTRACT = "rl_update_integrity_v1"
MARKER_FILE = "integrity_failure.json"


class IntegrityError(RuntimeError):
    """An update showed a condition under which the series stops at once."""


def _finite(values: Sequence[Any]) -> bool:
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in values)


def update_integrity(report: Mapping[str, Any]) -> dict[str, Any]:
    """The stop conditions an update report shows, as ``failures``, with what was checked.

    ``ok`` is whether there is none.  ``warnings`` are things to read that do
    not stop a run by themselves (rows the builder left out for a reason it
    counts, a resumed update).

    ``applied`` is whether an optimizer step was taken.  What the row builder
    counted is read either way.  A batch whose rows were all rejected takes
    no step, and the reasons a row is rejected for (a prompt that is not the
    sampled one, a cut history, a choice that could not be rendered) are stop
    conditions whether or not a row was left: that batch is the case where
    something is wrong with every row, not the case with nothing to stop for.
    """

    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    applied = isinstance(report.get("update"), Mapping)
    update: Mapping[str, Any] = report["update"] if applied else {}
    rows = report.get("rows") if isinstance(report.get("rows"), Mapping) else {}
    if not applied:
        checks["note"] = report.get("note")

    # --- nonfinite updates
    if applied:
        series = [value for name in ("policy_loss", "kl", "lifecycle_loss", "supervised_loss", "grad_norm")
                  for value in (update.get(name) or [])]
        dropped = sum(int(step.get("dropped_nonfinite") or 0) for step in update.get("row_grad_norms") or [])
        checks["finite_losses_and_gradient_norms"] = _finite(series)
        checks["rows_dropped_as_nonfinite"] = dropped
        if not checks["finite_losses_and_gradient_norms"]:
            failures.append("nonfinite_update: a loss or a gradient norm is not finite")
        if dropped:
            failures.append(f"nonfinite_update: {dropped} rows had a gradient that is not finite")

    # --- likelihood provenance: the trained prompt is the sampled one, the trained tokens have their sampling record
    provenance = {name: int(rows.get(name) or 0) for name in (
        "prompt_length_mismatches", "prompt_digest_mismatches", "prompt_digest_missing", "rows_without_sampling_record_rejected",
        "decision_credit_rows_without_record", "decision_values_unmatched", "supervised_rows_unrenderable")}
    refused_targets = int((rows.get("target_renderer") or {}).get("refused") or 0)
    first_check = (update.get("on_policy_check") or [{}])[0]
    checks["provenance"] = {**provenance, "targets_refused": refused_targets,
                            "first_step_rows_without_record": first_check.get("rows_without_record"),
                            "first_step_mean_abs_log_ratio": first_check.get("mean_abs_log_ratio")}
    for name in ("prompt_length_mismatches", "prompt_digest_mismatches", "prompt_digest_missing"):
        if provenance[name]:
            failures.append(f"likelihood_provenance: {provenance[name]} steps whose re-rendered prompt is not the sampled one ({name})")
    if provenance["rows_without_sampling_record_rejected"]:
        failures.append(f"likelihood_provenance: {provenance['rows_without_sampling_record_rejected']} sampled rows without a sampling record")
    if provenance["decision_credit_rows_without_record"] or provenance["decision_values_unmatched"]:
        failures.append("likelihood_provenance: continuation values without a sampled decision row of their own "
                        f"(without record {provenance['decision_credit_rows_without_record']}, unmatched {provenance['decision_values_unmatched']})")
    if refused_targets or provenance["supervised_rows_unrenderable"]:
        failures.append(f"likelihood_provenance: {refused_targets + provenance['supervised_rows_unrenderable']} labelled choices "
                        "could not be rendered as the completion the student emits")
    limit = (update.get("settings") or {}).get("max_on_policy_log_ratio")
    gap = first_check.get("mean_abs_log_ratio")
    if limit is not None and gap is not None and float(gap) > float(limit):
        failures.append(f"likelihood_provenance: the first step is off policy (mean absolute log-ratio {gap}, limit {limit})")

    # --- silent history truncation
    truncation = {name: int(rows.get(name) or 0) for name in ("prompt_truncated_rows", "prompts_cut_at_generation")}
    checks["truncation"] = truncation
    if truncation["prompt_truncated_rows"]:
        failures.append(f"history_truncation: {truncation['prompt_truncated_rows']} rows whose prompt does not fit the training length")
    if truncation["prompts_cut_at_generation"]:
        failures.append(f"history_truncation: {truncation['prompts_cut_at_generation']} steps whose prompt the policy cut at its input limit")

    # --- duplicated or missing update work
    if applied:
        accounting = update.get("row_accounting") if isinstance(update.get("row_accounting"), Mapping) else {}
        steps = update.get("optimizer_steps")
        planned = update.get("planned_steps")
        optimizer = report.get("optimizer") if isinstance(report.get("optimizer"), Mapping) else None
        checks["update_work"] = {
            "each_row_once": accounting.get("each_row_once"),
            "rows_built": rows.get("rows"), "rows_updated": update.get("rows"),
            "optimizer_steps": steps, "planned_steps": planned,
            "optimizer_steps_carried": None if optimizer is None else (optimizer.get("steps_before"), optimizer.get("steps_after")),
        }
        if accounting and not accounting.get("each_row_once"):
            failures.append("update_work: a row is in no optimizer step or in more than one")
        if rows.get("rows") is not None and update.get("rows") is not None and int(rows["rows"]) != int(update["rows"]):
            failures.append(f"update_work: {rows['rows']} rows were built and {update['rows']} were updated")
        if planned is not None and steps is not None and int(steps) > int(planned):
            failures.append(f"update_work: {steps} optimizer steps were taken, {planned} were planned")
        if optimizer is not None and optimizer.get("steps_after") is not None and steps is not None:
            taken = int(optimizer["steps_after"]) - int(optimizer.get("steps_before") or 0)
            if taken != int(steps):
                failures.append(f"update_work: the carried optimizer counts {taken} steps for an update of {steps}")
        for name in ("policy_loss", "kl", "grad_norm"):
            if steps is not None and len(update.get(name) or []) != int(steps):
                failures.append(f"update_work: {len(update.get(name) or [])} values of {name} for {steps} optimizer steps")

        # --- checkpoint recovery
        resumed = update.get("resumed_from")
        checks["recovery"] = {"resumed_from": resumed, "resumed_from_checkpoint": report.get("resumed_from_checkpoint"),
                              "batch_digest": update.get("batch_digest"), "plan_digest": accounting.get("plan_digest")}
        if resumed and not update.get("batch_digest"):
            warnings.append("recovery: the update resumed from a checkpoint that records no batch digest")
        if isinstance(resumed, Mapping) and resumed.get("plan_verified") is False:
            warnings.append("recovery: the update resumed from a checkpoint that records neither the order of its rows nor its "
                            "settings, so the rows before the checkpoint are taken on trust")
    elif rows:
        checks["update_work"] = {"rows_built": rows.get("rows"), "rows_updated": None}
        if not int(rows.get("rows") or 0) and not failures:
            warnings.append("rows: no training row was built, and no stop condition accounts for it")
    for name in ("groups_incomplete", "trajectories_unpaired", "prompt_truncated_rows_rejected", "empty_completions"):
        if int(rows.get(name) or 0):
            warnings.append(f"rows: {name} = {rows[name]}")
    return {"contract": INTEGRITY_CONTRACT, "ok": not failures, "applied": applied, "failures": failures, "warnings": warnings,
            "checks": checks}


__all__ = ["INTEGRITY_CONTRACT", "IntegrityError", "MARKER_FILE", "update_integrity"]
