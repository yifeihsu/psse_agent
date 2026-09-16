"""Standalone scientific figures from saved held-out screening evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"gnn": "#236A9E", "wls": "#D78B28", "matched": "#78814C", "union": "#B95D8B"}


def _draw_rate(ax, y, item, color, label=None, offset=0., height=.5):
    if not item or item.get("rate") is None:
        return
    value = item["rate"] * 100
    interval = item.get("parent_bootstrap_ci95")
    error = np.array([[max(0, value - interval[0] * 100)],
                      [max(0, interval[1] * 100 - value)]]) if interval else None
    ax.barh(y + offset, value, height=height, color=color, label=label,
            edgecolor="#333333", linewidth=.45)
    if error is not None:
        ax.errorbar(value, y + offset, xerr=error, fmt="none", ecolor="#252525", capsize=3, linewidth=1)
    ax.annotate(f"{value:.1f}%", (value, y + offset), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=9)


def render(evaluation_path, output_dir, training_path=None):
    source = Path(evaluation_path)
    evaluation = json.loads(source.read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#262626", "text.color": "#262626",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    rows = [
        ("GNN phase screen", evaluation["phase_recall"], evaluation["healthy_false_trigger_rate"], COLORS["gnn"]),
        ("Configured WLS", evaluation.get("wls_phase_recall"), evaluation.get("wls_healthy_false_trigger_rate"), COLORS["wls"]),
        ("WLS at matched calibration budget", evaluation.get("matched_budget_wls", {}).get("phase_recall"),
         evaluation.get("matched_budget_wls", {}).get("healthy_false_trigger_rate"), COLORS["matched"]),
        ("GNN OR configured WLS", evaluation.get("union_phase_recall"), evaluation.get("union_healthy_false_trigger_rate"), COLORS["union"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (name, recall, false_rate, color) in enumerate(rows):
        _draw_rate(axes[0], i, recall, color)
        _draw_rate(axes[1], i, false_rate, color)
    for ax in axes:
        ax.invert_yaxis()
        ax.set_yticks(range(len(rows)))
        ax.grid(axis="x", color="#E5E5E5", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels([row[0] for row in rows])
    axes[1].set_yticklabels([])
    axes[0].set_xlim(0, 115)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_xlabel("HIF or unbalance windows triggered (%)")
    maximum = max([r[2]["rate"] * 100 for r in rows if r[2] and r[2].get("rate") is not None] + [1.0])
    axes[1].set_xlim(0, max(3., maximum * 1.4))
    axes[1].set_xlabel("Healthy windows triggered (%)")
    axes[1].axvline(1, color="#555555", linestyle="--", linewidth=1)
    fig.suptitle("IEEE-14 held-out screening performance", x=.01, ha="left", fontsize=16)
    healthy = evaluation["healthy_false_trigger_rate"]
    fig.text(.01, .025,
             f"Healthy test: {healthy['count']:,} windows, {healthy['parents']} operating parents. "
             "Bars: test rates; error bars: 95% parent-bootstrap intervals.\n"
             "Dashed line: nominal 1% calibration target. The union is not calibrated to that target.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    paths = []
    for suffix in ("png", "svg"):
        target = output / f"detector_comparison.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        paths.append(str(target.resolve()))
    plt.close(fig)

    severity = evaluation["by_family_and_severity"]
    keys = [key for key in severity if key.startswith(("hif/", "unbalance/"))]
    order = {"weak": 0, "intermediate": 1, "strong": 2}
    keys.sort(key=lambda key: (key.split("/")[0], order.get(key.split("/")[1], 3), key))
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, len(keys) * .65 + 1.9)))
    for idx, key in enumerate(keys):
        _draw_rate(ax, idx, severity[key], COLORS["gnn"], "GNN phase screen" if idx == 0 else None, -.17, .28)
        baseline = evaluation.get("by_family_and_severity_wls", {}).get(key)
        _draw_rate(ax, idx, baseline, COLORS["wls"], "Configured WLS" if idx == 0 else None, .17, .28)
    ax.set_yticks(range(len(keys)), [f"{key.replace('/', ' · ')}  (n={severity[key]['count']:,})" for key in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Phase-investigation recall (%)")
    ax.grid(axis="x", color="#E5E5E5", linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=2, frameon=False)
    fig.suptitle("Detection by fault family and severity", x=.01, ha="left", fontsize=16)
    fig.text(.01, .02, "Held-out physical parents; noisy windows from the same parent remain grouped. "
             "Mixed-fault windows can appear in both families.\n"
             "Error bars: 95% parent-bootstrap intervals. A zero-width interval cannot bound unseen events.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    for suffix in ("png", "svg"):
        target = output / f"severity_recall.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        paths.append(str(target.resolve()))
    plt.close(fig)

    if training_path:
        training = json.loads(Path(training_path).read_text(encoding="utf-8"))
        history = training["history"]
        fig, ax = plt.subplots(figsize=(9.5, 4.7))
        palette = ["#236A9E", "#B89B35", "#D66C33", "#78814C", "#B95D8B"]
        for idx, seed in enumerate(sorted({row["seed"] for row in history})):
            points = [row for row in history if row["seed"] == seed]
            ax.plot([r["epoch"] for r in points], [100 * r["phase_recall"] for r in points],
                    label=f"Seed {seed}", color=palette[idx % len(palette)], linewidth=1.6)
        ax.set(xlabel="Training epoch", ylabel="Validation phase recall (%)", ylim=(0, 100))
        ax.grid(color="#E5E5E5", linewidth=.6)
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(.5, 1), frameon=False)
        fig.suptitle("Validation-only checkpoint selection", x=.01, ha="left", fontsize=15)
        fig.text(.01, .01, "Each epoch uses its validation healthy-reference threshold. "
                 "These are validation metrics; final test data are not used for selection.", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .91))
        for suffix in ("png", "svg"):
            target = output / f"validation_learning_curves.{suffix}"
            fig.savefig(target, dpi=180, bbox_inches="tight")
            paths.append(str(target.resolve()))
        plt.close(fig)
    (output / "figure_sources.json").write_text(json.dumps({
        "evaluation_path": str(source.resolve()), "evaluation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_path": str(Path(training_path).resolve()) if training_path else None,
        "figures": paths, "units": "percentage", "intervals": "95% physical-parent percentile bootstrap",
    }, indent=2) + "\n", encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training")
    args = parser.parse_args()
    print("\n".join(render(args.evaluation, args.output_dir, args.training)))


if __name__ == "__main__":
    main()
