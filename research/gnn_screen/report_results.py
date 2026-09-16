"""Render a held-out simulation evaluation as a reviewable Markdown report.

This consumes frozen artifacts; it neither selects a checkpoint nor changes any
threshold. Rates count windows while uncertainty resamples operating parents.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import FAMILY_NAMES, write_json
from .evaluate import grouped_rate


def _percent(value):
    return "not available" if value is None else f"{100 * float(value):.2f}%"


def _number(value):
    return "not available" if value is None else f"{float(value):.4f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rate_cells(metric):
    metric = metric or {}
    interval = metric.get("parent_bootstrap_ci95")
    ci = (f"{_percent(interval[0])} to {_percent(interval[1])}"
          if interval is not None else "not available")
    return [_percent(metric.get("rate")), str(metric.get("count", 0)),
            str(metric.get("parents", 0)), ci]


def _table(headers, rows):
    return ["| " + " | ".join(map(_cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(_cell, row)) + " |" for row in rows]]


def audit_artifacts(evaluation, training=None, *, bootstrap_replicates=1000, seed=2026):
    """Check artifact pairing and expose independent-parent coverage explicitly."""
    predictions = evaluation.get("predictions", [])
    calibration = evaluation.get("calibration", {})
    if len(predictions) != evaluation["valid_test_windows"]:
        raise ValueError("evaluation prediction count disagrees with valid_test_windows")
    test_parents = set(evaluation["test_parent_ids"])
    calibration_parents = set(calibration.get("calibration_parent_ids", []))
    if test_parents & calibration_parents:
        raise ValueError("test parents overlap calibration parents")
    if any(row["parent_id"] not in test_parents for row in predictions):
        raise ValueError("predictions contain an undeclared test parent")
    for key in ("model_id", "checkpoint_sha256"):
        if key in evaluation and key in calibration and evaluation[key] != calibration[key]:
            raise ValueError(f"evaluation/calibration {key} mismatch")
    split_counts = {}
    if training is not None:
        mapping = training["parent_splits"]
        used_for_fit = {parent for parent, split in mapping.items() if split in ("train", "validation")}
        if test_parents & used_for_fit:
            raise ValueError("test parents overlap training or validation parents")
        if calibration_parents & used_for_fit:
            raise ValueError("calibration parents overlap training or validation parents")
        if any(parent in mapping and mapping[parent] != "test" for parent in test_parents):
            raise ValueError("test parent assignment changed since training")
        if any(parent in mapping and mapping[parent] != "calibration" for parent in calibration_parents):
            raise ValueError("calibration parent assignment changed since training")
        if (training.get("selected", {}).get("model_id") is not None and
                training["selected"]["model_id"] != evaluation.get("model_id")):
            raise ValueError("training report belongs to a different model")
        split_counts = dict(Counter(mapping.values()))
    phase = [row for row in predictions if row["labels"]["phase_mask"] and row["labels"]["phase"]]
    missed = [row for row in phase if row.get("wls_alarm") is False]
    recovered = grouped_rate(missed,
        [row["phase_score"] > calibration["phase_threshold"] for row in missed],
        bootstrap_replicates=bootstrap_replicates, seed=seed)
    compositions = {}
    for row in phase:
        labels = row["labels"]
        if all(labels["family_mask"]):
            name = "+".join(family for family, present in zip(FAMILY_NAMES, labels["family"]) if present)
        else:
            name = "partially labeled phase-positive"
        compositions.setdefault(name, []).append(row)
    phase_composition = {}
    for name, rows in sorted(compositions.items()):
        subset_missed = [row for row in rows if row.get("wls_alarm") is False]
        metric = lambda subset, values: grouped_rate(subset, values,
            bootstrap_replicates=bootstrap_replicates, seed=seed)
        phase_composition[name] = {
            "gnn_phase_recall": metric(rows, [row["phase_score"] > calibration["phase_threshold"] for row in rows]),
            "gnn_recall_among_wls_misses": metric(subset_missed,
                [row["phase_score"] > calibration["phase_threshold"] for row in subset_missed]),
        }
        if all(row.get("wls_alarm") is not None for row in rows):
            phase_composition[name]["wls_phase_recall"] = metric(rows, [row["wls_alarm"] for row in rows])
    warnings = []
    if calibration.get("healthy_count", 0) < 10000:
        warnings.append("Fewer than 10,000 healthy calibration windows were used; the 1% tail is sparsely sampled.")
    if calibration.get("healthy_parent_count", 0) < 100:
        warnings.append("Fewer than 100 independent healthy calibration parents were used; repeated noise draws do not increase operating-parent coverage.")
    healthy = evaluation.get("healthy_false_trigger_rate", {})
    interval = healthy.get("parent_bootstrap_ci95")
    if interval is not None and interval[0] == interval[1] and healthy.get("rate") in (0, 1):
        warnings.append("The healthy false-trigger bootstrap interval is degenerate because all observed decisions agree. This does not establish a zero population false-trigger rate or certify the 1% target.")
    return {
        "test_parent_count": len(test_parents),
        "valid_test_parent_count": len({row["parent_id"] for row in predictions}),
        "split_parent_counts": split_counts,
        "parent_overlap_check": "passed for supplied artifact IDs; physical provenance must establish those IDs",
        "configured_wls_negative_phase_recall": recovered,
        "phase_composition": phase_composition,
        "warnings": warnings,
    }


def render_report(evaluation, training=None, *, title="WLS screen GNN: held-out simulation pilot",
                  evaluation_path=None, training_path=None,
                  generation=None, generation_path=None,
                  bootstrap_replicates=1000, seed=2026):
    audit = audit_artifacts(evaluation, training,
                           bootstrap_replicates=bootstrap_replicates, seed=seed)
    calibration = evaluation["calibration"]
    lines = [f"# {title}", "",
        "These results measure screening on the supplied held-out simulation distribution. "
        "They do not establish field performance, fault localization, diagnostic recovery, or harmonic detection.", "",
        "## Data and frozen artifacts", ""]
    coverage = [
        ["Valid test windows", evaluation["valid_test_windows"]],
        ["Unavailable test windows", len(evaluation.get("invalid_test_windows", []))],
        ["Declared independent test parents", audit["test_parent_count"]],
        ["Test parents with available screens", audit["valid_test_parent_count"]],
        ["Screen availability", _percent(evaluation.get("screen_availability"))],
        ["Healthy calibration windows", calibration.get("healthy_count", "not available")],
        ["Independent healthy calibration parents", calibration.get("healthy_parent_count", "not available")],
    ]
    if training is not None:
        counts = training.get("valid_graph_counts", {})
        for split in ("train", "validation", "calibration", "test"):
            coverage.append([f"{split.capitalize()} parents / available windows recorded at training",
                f"{audit['split_parent_counts'].get(split, 0)} / {counts.get(split, 0)}"])
        selected = training.get("selected", {})
        coverage.append(["Validation-selected seed / epoch", f"{selected.get('seed')} / {selected.get('epoch')}"])
    lines += _table(["Coverage item", "Value"], coverage)
    if generation is not None:
        lines += ["", "Physical-corpus generation audit:", ""]
        physical_rows = [
            ["Corpus seed", generation.get("seed", "not available")],
            ["Failed physical variants", len(generation.get("failed_variants", []))],
            ["Maximum healthy balanced-equation mismatch (pu)", generation.get("healthy_max_balanced_equation_error_pu", "not available")],
            ["Maximum healthy squared discrepancy in measurement-sigma units", generation.get("healthy_max_squared_sigma_scaled_discrepancy", "not available")],
            ["Measurement convention", generation.get("measurement_convention", "not available")],
            ["Manifest SHA-256", generation.get("manifest_sha256", "not available")],
        ]
        lines += _table(["Physical check", "Recorded result"], physical_rows)
        if generation.get("severity_definitions"):
            lines += ["", "Recorded severity definitions:", "", "```json",
                      json.dumps(generation["severity_definitions"], indent=2, sort_keys=True), "```"]
    lines += ["", f"Model ID: `{evaluation.get('model_id', 'not available')}`. "
        f"Checkpoint SHA-256: `{evaluation.get('checkpoint_sha256', 'not available')}`.", "",
        f"Parent-ID overlap check: {audit['parent_overlap_check']}.", "",
        "Calibration thresholds use healthy parents only. The selected checkpoint and scaler remain frozen. "
        "Noise replicas and related fault variants remain together within an operating parent.", "",
        "## Threshold policy", ""]
    lines += _table(["Policy", "Value"], [
        ["Requested healthy GNN phase false-trigger rate", _percent(calibration.get("requested_healthy_false_trigger_rate"))],
        ["GNN phase score threshold (strict >)", _number(calibration.get("phase_threshold"))],
        ["GNN anomaly score threshold (strict >)", _number(calibration.get("anomaly_threshold"))],
        ["Configured WLS comparator rule", calibration.get("wls_comparator", {}).get("rule", "not available")],
        ["Configured WLS chi-square alpha", calibration.get("wls_comparator", {}).get("chi2_alpha", "not available")],
        ["Configured WLS normalized residual threshold", calibration.get("wls_comparator", {}).get("normalized_residual_threshold")],
        ["Healthy-reference matched WLS score threshold", _number(calibration.get("matched_wls_threshold"))],
        ["Matched WLS scalar definition", calibration.get("matched_wls_score", "not available")],
    ])
    lines += ["", "Sigmoid outputs are scores, not calibrated posterior probabilities. "
        "The GNN/WLS union has its own measured false-trigger rate; a 1% GNN calibration target does not imply a 1% union rate.", "",
        "## Independent test performance", ""]
    metric_specs = [
        ("GNN phase recall", "phase_recall"),
        ("GNN healthy phase false triggers", "healthy_false_trigger_rate"),
        ("GNN phase triggers on non-phase faults", "nonphase_fault_trigger_rate"),
        ("Configured WLS phase recall", "wls_phase_recall"),
        ("Configured WLS healthy false triggers", "wls_healthy_false_trigger_rate"),
        ("Configured WLS triggers on non-phase faults", "wls_nonphase_fault_trigger_rate"),
        ("GNN or configured WLS phase recall", "union_phase_recall"),
        ("GNN or configured WLS healthy false triggers", "union_healthy_false_trigger_rate"),
        ("GNN or configured WLS triggers on non-phase faults", "union_nonphase_fault_trigger_rate"),
        ("GNN minus configured WLS phase recall", "paired_phase_recall_gain_over_wls"),
        ("GNN general anomaly recall", "general_anomaly_recall"),
        ("GNN healthy anomaly false triggers", "healthy_anomaly_trigger_rate"),
        ("GNN phase acquisition requests", "phase_acquisition_fraction"),
    ]
    performance = [[label, *_rate_cells(evaluation[key])] for label, key in metric_specs if key in evaluation]
    performance.append(["GNN recall among phase faults missed by configured WLS",
                        *_rate_cells(audit["configured_wls_negative_phase_recall"])])
    # Additional baseline metrics retain their explicit saved names so their
    # configured/matched operating points are never accidentally conflated.
    known = {key for _, key in metric_specs} | {"phase_recall_among_wls_misses"}
    for key, metric in evaluation.items():
        if key not in known and "wls" in key and isinstance(metric, dict) and "rate" in metric:
            performance.append([key.replace("_", " "), *_rate_cells(metric)])
    lines += _table(["Metric", "Rate", "Windows", "Parents", "Parent bootstrap 95% interval"], performance)
    matched = evaluation.get("matched_budget_wls")
    if matched:
        lines += ["", "The following WLS baseline uses a separate healthy-reference threshold on its saved scalar score, "
                  "calibrated with the same healthy parents and target budget as the GNN. It is distinct from the configured WLS alarm above.", ""]
        matched_specs = [
            ("Matched WLS phase recall", "phase_recall"),
            ("Matched WLS healthy false triggers", "healthy_false_trigger_rate"),
            ("Matched WLS triggers on non-phase faults", "nonphase_fault_trigger_rate"),
            ("GNN recall among phase faults missed by matched WLS", "gnn_recall_among_matched_wls_misses"),
            ("GNN minus matched WLS phase recall", "paired_phase_recall_gain_over_wls"),
        ]
        lines += _table(["Matched-budget metric", "Rate", "Windows", "Parents", "Parent bootstrap 95% interval"],
            [[label, *_rate_cells(matched[key])] for label, key in matched_specs if key in matched])
    lines += ["", "Rates are window weighted. Intervals resample whole independent operating parents, "
        "keeping their windows together. Conditional recall among WLS misses uses only that subset; "
        "a small or empty subset cannot establish reliable incremental detection.", "",
        "Phase recall counting unavailable screens as untriggered: "
        + _percent(evaluation.get("phase_trigger_recall_counting_unavailable_as_untriggered")) + ". "
        "Unavailable screens still require the independent acquisition fallback.", "",
        "## Family scores and severity", ""]
    compositions = audit["phase_composition"]
    if compositions:
        composition_rows = []
        for name, metrics in compositions.items():
            for label, key in (("GNN", "gnn_phase_recall"), ("Configured WLS", "wls_phase_recall"),
                               ("GNN on configured WLS misses", "gnn_recall_among_wls_misses")):
                if key in metrics:
                    composition_rows.append([name, label, *_rate_cells(metrics[key])])
        lines += _table(["Exact phase-positive label combination", "Screen", "Recall", "Windows", "Parents", "Parent bootstrap 95% interval"],
                        composition_rows)
        lines += ["", "Pure HIF and pure unbalance strata exclude competing faults. "
            "Mixed-case recall can reflect evidence of a co-occurring balanced error and must not be presented as pure-fault sensitivity.", ""]
    family_rows = []
    for family in FAMILY_NAMES:
        result = evaluation.get("by_family", {}).get(family, {})
        ranking = result.get("family_score_discrimination") or {}
        family_rows.append([family, "yes" if result.get("head_trained") else "no",
                            *_rate_cells(result.get("phase_trigger")),
                            _number(ranking.get("roc_auc")), _number(ranking.get("average_precision"))])
    lines += _table(["Family", "Head trained", "Phase trigger", "Positive windows", "Parents",
                     "Parent bootstrap 95% interval", "Family score AUROC", "Family score AP"], family_rows)
    lines += ["", "The phase-trigger column measures requests for three-phase investigation. "
        "Measurement, parameter, and topology faults are competing non-phase families, so their phase triggers are not phase-detection successes. "
        "Family scores are independent and mixed labels can activate several families. "
        "AUROC/AP are threshold-free discrimination statistics; AP depends on the experimental family prevalence.", ""]
    family_operating_rows = []
    for family in FAMILY_NAMES:
        result = evaluation.get("by_family", {}).get(family, {})
        for label, key in (("recall", "family_recall"), ("all-negative false positives", "family_false_positive_rate"),
                           ("healthy false triggers", "family_healthy_trigger_rate")):
            if key in result:
                family_operating_rows.append([family, label, _number(result.get("family_threshold")),
                                               *_rate_cells(result[key])])
    if family_operating_rows:
        lines += _table(["Family", "Own head metric", "Threshold", "Rate", "Windows", "Parents", "Parent bootstrap 95% interval"],
                        family_operating_rows)
        lines += ["", "Each family-head threshold is a healthy-reference operating point. "
            "The combined probability of one or more heads firing is not controlled by an individual head's target rate.", ""]
    severity = evaluation.get("by_family_and_severity", {})
    if severity:
        severity_rows = []
        wls_severity = evaluation.get("by_family_and_severity_wls", {})
        for name, metric in sorted(severity.items()):
            severity_rows.append([name, "GNN", *_rate_cells(metric)])
            if name in wls_severity:
                severity_rows.append([name, "Configured WLS", *_rate_cells(wls_severity[name])])
        lines += _table(["Family / severity", "Screen", "Phase trigger rate", "Windows", "Parents", "Parent bootstrap 95% interval"],
                        severity_rows)
        lines += ["", "Severity labels refer to family-specific experimental parameters; an equal label does not imply equal physical magnitude or statistical detectability across families. "
            "A family/severity stratum may include mixed cases; consult the exact-combination rows above and source physical metadata.", ""]
    for key, strata in evaluation.get("offline_phase_strata", {}).items():
        if strata:
            lines += [f"### Offline audit: {key}", ""]
            lines += _table(["Stratum", "Phase trigger rate", "Windows", "Parents", "Parent bootstrap 95% interval"],
                            [[name, *_rate_cells(metric)] for name, metric in sorted(strata.items())])
            lines += [""]
    lines += ["## Interpretation limits", ""]
    lines += [f"- {warning}" for warning in audit["warnings"]]
    if generation is not None:
        lines += [f"- Corpus scope: {limitation}." for limitation in generation.get("limitations", [])]
        if generation.get("failed_variants"):
            lines.append("- Physical simulation failures are excluded from the valid-screen denominator; inspect failed_variants in the generation report before interpreting coverage.")
    lines += [
        "- Independent operating parents are distinct from noisy windows. Generalization claims must use parent-held-out evidence and preserve the physical-parent identity.",
        "- The input contract is phase-A voltage magnitude and total three-phase powers, with injections excluding shunts already modeled by WLS. Export convention and simulator equivalence need physical audits; a manifest label alone does not establish them.",
        "- Simulator-specific artifacts, limited fault mechanisms, and operating-point ranges can create shortcuts. A single-network simulation study does not establish transfer to other networks or real measurements.",
        "- Small or electrically unobservable faults may remain indistinguishable from noise in a single balanced snapshot. A negative screen does not establish absence of a fault.",
        "- Graph connectivity and residual features have not established added value unless matched non-graph and feature-ablation baselines were run with independent selection and calibration.",
        "- Test results must not be reused to select thresholds or tune the model. Any later adaptive redesign requires fresh held-out parents for an unbiased final claim.",
        "- Harmonics are outside this five-family model and retain the existing acquisition and diagnostic path.", "",
        "## Artifact references", ""]
    for label, path in (("Evaluation JSON", evaluation_path), ("Training report JSON", training_path),
                        ("Physical generation report JSON", generation_path)):
        if path is not None:
            lines.append(f"- {label}: `{Path(path).resolve()}`")
    if training is not None and training.get("checkpoint"):
        lines.append(f"- Frozen checkpoint: `{training['checkpoint']}`")
    return "\n".join(lines).rstrip() + "\n", audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--training-report")
    parser.add_argument("--generation-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--title", default="WLS screen GNN: held-out simulation pilot")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    training = (json.loads(Path(args.training_report).read_text(encoding="utf-8"))
                if args.training_report else None)
    generation = (json.loads(Path(args.generation_report).read_text(encoding="utf-8"))
                  if args.generation_report else None)
    report, audit = render_report(evaluation, training, title=args.title,
        evaluation_path=args.evaluation, training_path=args.training_report,
        generation=generation, generation_path=args.generation_report,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    if args.summary_json:
        write_json(args.summary_json, audit)
    print(output.resolve())


if __name__ == "__main__":
    main()
