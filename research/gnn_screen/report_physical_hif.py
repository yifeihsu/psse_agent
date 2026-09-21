"""Standalone scientific plots and Markdown for the frozen physical HIF sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ratio(row, key):
    count, total = row[key + "_count"], row[key + "_available"]
    return f"{count}/{total} ({100 * count / total:.1f}%)" if total else "unavailable"


def render(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    all_rows = summary["tables"]["by_voltage_resistance_noise"]
    faults = [row for row in all_rows if row["kind"] == "hif"]
    profiles = [("baseline", "Power σ = 0.01 pu"), ("accuracy_005", "Power σ = 0.005 pu"),
                ("accuracy_002", "Power σ = 0.002 pu")]
    colors = {"gnn_phase": "#246a98", "wls_dual": "#d68b25", "gnn_or_wls": "#a8517d"}
    labels = {"gnn_phase": "Frozen GNN phase screen", "wls_dual": "Configured WLS", "gnn_or_wls": "GNN OR WLS"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, kv in enumerate((69, 13.8)):
        for j, (profile, title) in enumerate(profiles):
            ax = axes[i, j]
            rows = sorted([r for r in faults if r["local_base_kv_ll"] == kv and r["noise_profile"] == profile],
                          key=lambda row: row["resistance_ohm"])
            for key, marker, style in (("gnn_phase", "o", "-"), ("wls_dual", "s", "--"), ("gnn_or_wls", "^", ":")):
                ax.plot([r["resistance_ohm"] for r in rows], [100 * r[key + "_rate"] for r in rows],
                        marker=marker, linestyle=style, color=colors[key], label=labels[key], linewidth=2, markersize=5)
            ax.set_xscale("log")
            ax.set_ylim(-3, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ticks = [50, 100, 200, 500, 1000, 2000, 5000]
            ax.set_xticks(ticks, [str(value) for value in ticks])
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", alpha=.2)
            ax.set_title(f"{kv:g} kV · {title}\nn = {rows[0]['observations']} per resistance", fontsize=11)
            if j == 0:
                ax.set_ylabel("Fault observations triggering (%)")
            if i == 1:
                ax.set_xlabel("Fault resistance (Ω, log scale)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, .925), frameon=False)
    fig.suptitle("Frozen GNN and WLS on the physical IEEE14 HIF sweep", x=.03, ha="left", fontsize=18)
    fig.text(.03, .02, "Two canonical power-flow parents; 16 lines, ABC phases, midspan faults. Each noise group is paired across resistance and accuracy.\n"
             "Original GNN weights, scaler and thresholds; no target recalibration. Descriptive sweep rates, not population confidence estimates.", fontsize=10)
    fig.tight_layout(rect=(.01, .08, 1, .86))
    for suffix in ("png", "svg"):
        fig.savefig(root / f"detection_by_resistance.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plan, audit = summary["plan"], summary["audit"]
    core = {r["resistance_ohm"]: r for r in faults if r["local_base_kv_ll"] == 69 and r["noise_profile"] == "baseline"}
    lines = ["# Frozen GNN and WLS on the physical IEEE14 HIF sweep", "",
        "This is a fresh inference replay of the saved `ieee14_physical_hif_v1` unfiltered sweep. "
        "The practical V2 GNN weights, training scaler, and original healthy-reference thresholds remain frozen. "
        "No model was retrained and no threshold was fitted to the sweep controls or faults.", "",
        f"At 69 kV and baseline noise, the frozen GNN phase screen triggers on {ratio(core[500], 'gnn_phase')} "
        f"at 500 Ω versus WLS's {ratio(core[500], 'wls_dual')}; at 1000 Ω the corresponding counts are "
        f"{ratio(core[1000], 'gnn_phase')} and {ratio(core[1000], 'wls_dual')}. "
        "The extra alarms come with a higher observed healthy trigger rate and do not establish correct HIF classification.", "",
        "## Experimental scope", "",
        "- 672 physical HIF cases: two canonical power-flow parents (load scales 0.8 and 1.0), "
        "16 eligible lines, ABC phases, seven resistances, fixed midspan location 0.5.",
        "- Seven 69 kV lines and nine 13.8 kV lines. Bus 8 is 18 kV, connected through the 7–8 transformer, "
        "and supplies no eligible same-voltage line fault.",
        "- 2,016 noisy fault observations across three covariance profiles. There are 96 noise groups, "
        "paired across resistance, accuracy, healthy and no-fault split controls.",
        "- Healthy and no-fault split controls each contain 96 observations per noise profile. "
        "They are paired controls, not 192 independent operating parents.",
        "- The sweep is unfiltered by WLS, paired separation, learned scores, or training admission. "
        "The 100/200/500/1000 Ω grid is not the uniformly sampled 100–1000 Ω training proposal distribution.",
        "- Phase-A voltage-magnitude noise is σ=0.001 pu in every profile; power σ is 0.01, 0.005, or 0.002 pu. "
        "Each WLS and GNN graph uses that observation's actual covariance.", "",
        "## Detection results", "",
        "The GNN column is the phase-investigation head: a request for additional three-phase diagnostics. "
        "It is not a correct-HIF-family classification rate. WLS is the configured dual residual detector.", ""]
    for profile, title in profiles:
        lines += [f"### {title}", "", "| Voltage | Resistance Ω | GNN phase | WLS | GNN OR WLS | GNN additions among WLS misses |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in sorted([r for r in faults if r["noise_profile"] == profile],
                          key=lambda row: (-row["local_base_kv_ll"], row["resistance_ohm"])):
            additional = (f"{row['gnn_additional_to_wls']}/{row['wls_misses_with_gnn_available']}"
                          if row['wls_misses_with_gnn_available'] else "No WLS misses")
            lines.append(f"| {row['local_base_kv_ll']:g} kV | {row['resistance_ohm']:g} | {ratio(row, 'gnn_phase')} | "
                f"{ratio(row, 'wls_dual')} | {ratio(row, 'gnn_or_wls')} | "
                f"{additional} |")
        lines.append("")
    lines += ["![Detection by resistance](detection_by_resistance.png)", "", "## Healthy/no-fault controls", "",
              "| Control | Noise profile | GNN phase triggers | WLS triggers | GNN OR WLS |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary["tables"]["controls"]:
        lines.append(f"| {row['kind']} | {row['noise_profile']} | {ratio(row, 'gnn_phase')} | {ratio(row, 'wls_dual')} | {ratio(row, 'gnn_or_wls')} |")
    lines += ["", "These controls are test observations, not new calibration data. The original nominal 1% "
        "calibration target does not guarantee a 1% rate on this transferred population. Fault triggers whose "
        "paired healthy control already triggered must not be called newly detected faults.", "",
        "### Baseline fault triggers absent from their paired healthy control", "",
        "| Voltage | Resistance Ω | Newly triggering GNN | Newly triggering WLS |", "| --- | ---: | ---: | ---: |"]
    for row in sorted([r for r in faults if r["noise_profile"] == "baseline"],
                      key=lambda row: (-row["local_base_kv_ll"], row["resistance_ohm"])):
        lines.append(f"| {row['local_base_kv_ll']:g} kV | {row['resistance_ohm']:g} | "
                     f"{row['gnn_phase_new_vs_healthy_count']}/{row['gnn_phase_paired_available']} | "
                     f"{row['wls_dual_new_vs_healthy_count']}/{row['wls_dual_paired_available']} |")
    lines += ["", "## Thresholds and graph transfer", "",
        f"GNN uses strict score > {summary['thresholds']['phase_threshold']:.16g}. The separate HIF-family "
        f"head threshold remains {summary['thresholds']['family_thresholds']['hif']:.16g}. Scores are uncalibrated sigmoids, not posterior probabilities.", "",
        "WLS uses J = eᵀR⁻¹e ≥129.9726787 (95 residual degrees of freedom, chi-square α=0.01), "
        "OR maximum absolute normalized residual ≥4. "
        "Both comparisons are inclusive. The source-calibrated WLS threshold is retained as a separate "
        "CSV metric; it is not fitted again or asserted to meet a target false-trigger rate here.", "",
        "The current graph correctly identifies branch 7–8 as a transformer from public nominal-voltage "
        "metadata. The old normalized training graphs marked it as a line. The sensitivity column below "
        "restores only the old tap/shift-based line/transformer flags on exactly the same measured vectors; "
        "it does not change the physical network or provide an alternative valid physical interpretation.", "",
        "| Voltage | Resistance Ω | Current physical graph GNN | Old type-flag sensitivity | Changed decisions |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in sorted([r for r in faults if r["noise_profile"] == "baseline"],
                      key=lambda row: (-row["local_base_kv_ll"], row["resistance_ohm"])):
        lines.append(f"| {row['local_base_kv_ll']:g} kV | {row['resistance_ohm']:g} | {ratio(row, 'gnn_phase')} | "
                     f"{ratio(row, 'legacy_type_gnn_phase')} | {row['legacy_type_decision_flips']} |")
    lines += ["", f"Across all {summary['observations']:,} observations, changing only these type flags changes "
        f"{sum(r['legacy_type_decision_flips'] for r in all_rows)} phase-trigger decisions. "
        "No 69 kV phase decision changes in this sweep. This small sensitivity does not explain the main screening pattern."]
    lines += ["", "## Distinguishing the GNN outputs", "",
        "The phase head and general-anomaly head answer different questions. A missed phase trigger does "
        "not mean the general-anomaly head is quiet. The HIF-family score is a third output with its own "
        "source threshold; it must not be conflated with the phase-screening rates above.", "",
        "| Power-noise profile | 69 kV resistance Ω | Phase head | General-anomaly head | HIF-family head |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in sorted([r for r in faults if r["local_base_kv_ll"] == 69 and r["resistance_ohm"] in (100, 200, 500, 1000)],
                      key=lambda row: (row["noise_profile"], row["resistance_ohm"])):
        lines.append(f"| {row['noise_profile']} | {row['resistance_ohm']:g} | {ratio(row, 'gnn_phase')} | "
                     f"{ratio(row, 'gnn_anomaly')} | {ratio(row, 'gnn_hif_family')} |")
    lines += ["", "For example, at power σ=0.002 pu, all 42 of the 69 kV / 100 Ω observations trigger "
        "the general-anomaly head and WLS, while none trigger the phase head. At baseline noise, all "
        "42 observations at 500 Ω and at 1000 Ω rank unbalance above HIF in the family scores. Thus "
        "the frozen model's phase-screening additions are not evidence of correct HIF classification."]
    lines += ["", "## Interpretation limits", "",
        "Only two operating parents were swept. Rates and per-parent tables are descriptive; "
        "no population confidence intervals are reported. Strong dependence among interventions prevents "
        "treating all 672 physical cases or 2,016 noisy faults as independent operating conditions.", "",
        "The saved parents use canonical power-flow dispatch, unlike the OPF-generated V2 training parents. "
        "Their maximum voltage is 1.09 pu against a 1.06 pu case limit; maximum generator reactive-limit "
        "overruns are about 7.30 and 16.55 MVAr. The circuit equations passed the original physical audit, "
        "but these parents are not operating-limit-feasible OPF states. Results therefore combine changes "
        "in fault population, graph type metadata, and operating-point distribution.", "",
        "The 0.005 and 0.002 power-noise profiles also differ from the baseline covariance used for V2 "
        "training/calibration. Better measurements can improve WLS without producing monotonic behavior "
        "from a frozen learned screen. This experiment does not test balanced-error classification, "
        "full episode recovery, arcing faults, or field protection performance.", "",
        "## Verification and reproduction", "",
        f"All {audit['reconstructed_hash_matches']:,} noisy vectors matched the saved measurement hashes. "
        f"Fresh WLS matched {audit['wls_comparisons']:,} saved WLS outcomes, with {audit['wls_alarm_mismatches']} alarm mismatches. "
        f"Maximum objective difference: {audit['max_J_difference']:.3g}; maximum normalized-residual difference: "
        f"{audit['max_normalized_residual_difference']:.3g}. GNN unavailable observations: {len(audit['unavailable'])}.", "",
        f"Checkpoint SHA-256: `{plan['checkpoint_sha256']}`. Original model ID: `{plan['model_id']}`.", "",
        "The run plan records full input-file and graph/solver/model source hashes. Implementation snapshots "
        "preserve the uncommitted physical implementation used for this run; a Git HEAD alone does not describe it.", "",
        "```powershell", "python -m research.gnn_screen.evaluate_physical_hif --sweep output/ieee14_physical_hif_20260918/sweep --checkpoint output/gnn_practical_v2_20260916/training/checkpoint.pt --calibration output/gnn_practical_v2_20260916/evaluation/primary/calibration.json --output-dir output/gnn_physical_hif_replay --device cpu",
        "python -m research.gnn_screen.report_physical_hif output/gnn_physical_hif_replay", "```", "",
        "Artifacts: `predictions.jsonl`, `summary.json`, `by_voltage_resistance_noise.csv`, `by_phase.csv`, "
        "`by_parent.csv`, `controls.csv`, `frozen_calibration.json`, and `run_plan.json`.", "",
        "The focused replay/aggregation suite passed 25 tests. Independent audit receipts and the reproduction "
        "script are saved beside this report as `independent_audit.json` and `audit_frozen_inference.py`.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return root / "README.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory")
    args = parser.parse_args()
    print(render(args.result_directory))
