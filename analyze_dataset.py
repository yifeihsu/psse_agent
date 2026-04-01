#!/usr/bin/env python3
"""
Power System Diagnostic Dataset - Analysis & Visualization

This script provides analysis and visualization tools for exploring the dataset:
- Class distribution plots
- Measurement value distributions
- Residual analysis
- Tool call patterns
- Confidence score analysis

Usage:
    python analyze_dataset.py --file data/sft_with_tools.jsonl

Requirements:
    pip install matplotlib seaborn pandas numpy
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)


def load_dataset(file_path: str) -> List[Dict]:
    """Load JSONL dataset."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_features(sample: Dict) -> Dict:
    """Extract features from sample."""
    messages = sample['messages']
    features = {
        'z_obs': None,
        'residuals': None,
        'lambdaN': None,
        'error_family': None,
        'confidence': None,
        'tool_call_count': 0
    }

    for msg in messages:
        if msg['role'] == 'user':
            try:
                user_data = json.loads(msg['content'])
                features['z_obs'] = np.array(user_data['z_obs'])
            except:
                pass

        if msg['role'] == 'assistant' and 'tool_calls' in msg:
            features['tool_call_count'] += len(msg['tool_calls'])

        if msg['role'] == 'tool' and msg.get('name') == 'wls_from_path' and features['residuals'] is None:
            try:
                tool_result = json.loads(msg['content'])
                if 'r' in tool_result:
                    features['residuals'] = np.array(tool_result['r'])
                if 'lambdaN' in tool_result:
                    features['lambdaN'] = np.array(tool_result['lambdaN'])
            except:
                pass

        if msg['role'] == 'assistant' and 'content' in msg:
            try:
                if msg['content'].startswith('{'):
                    diagnosis = json.loads(msg['content'])
                    features['error_family'] = diagnosis.get('error_family')
                    features['confidence'] = diagnosis.get('confidence')
            except:
                pass

    return features


def plot_class_distribution(samples: List[Dict], output_path: str = None):
    """Plot error family distribution."""
    class_counts = defaultdict(int)

    for sample in samples:
        features = extract_features(sample)
        if features['error_family']:
            class_counts[features['error_family']] += 1

    # Sort by count
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    ax1.bar(classes, counts, color=colors[:len(classes)])
    ax1.set_xlabel('Error Family', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Error Family Distribution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for i, (cls, count) in enumerate(zip(classes, counts)):
        ax1.text(i, count + max(counts)*0.01, str(count),
                ha='center', va='bottom', fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=classes, autopct='%1.1f%%',
           colors=colors[:len(classes)], startangle=90)
    ax2.set_title('Error Family Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_measurement_distributions(samples: List[Dict], output_path: str = None):
    """Plot measurement value distributions."""
    all_z_obs = []

    for sample in samples[:100]:  # Sample for speed
        features = extract_features(sample)
        if features['z_obs'] is not None:
            all_z_obs.append(features['z_obs'])

    if not all_z_obs:
        print("No measurement data found")
        return

    z_matrix = np.array(all_z_obs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Voltage magnitudes
    voltages = z_matrix[:, 0:14].flatten()
    axes[0, 0].hist(voltages, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Nominal (1.0 p.u.)')
    axes[0, 0].set_xlabel('Voltage Magnitude (p.u.)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Voltage Magnitude Distribution (Indices 0-13)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()

    # Power injections
    powers_inj = z_matrix[:, 14:42].flatten()
    axes[0, 1].hist(powers_inj, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_xlabel('Power Injection (p.u.)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Power Injection Distribution (Indices 14-41)', fontsize=12, fontweight='bold')

    # Power flows
    powers_flow = z_matrix[:, 42:122].flatten()
    axes[1, 0].hist(powers_flow, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Power Flow (p.u.)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Power Flow Distribution (Indices 42-121)', fontsize=12, fontweight='bold')

    # Box plot comparison
    data_to_plot = [
        z_matrix[:, 0:14].flatten(),
        z_matrix[:, 14:42].flatten(),
        z_matrix[:, 42:122].flatten()
    ]
    axes[1, 1].boxplot(data_to_plot, labels=['Voltage\n(0-13)', 'Power Inj\n(14-41)', 'Power Flow\n(42-121)'])
    axes[1, 1].set_ylabel('Value (p.u.)', fontsize=11)
    axes[1, 1].set_title('Measurement Range Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_residual_analysis(samples: List[Dict], output_path: str = None):
    """Plot residual analysis by error type."""
    residuals_by_class = defaultdict(list)

    for sample in samples[:200]:  # Sample for speed
        features = extract_features(sample)
        if features['residuals'] is not None and features['error_family']:
            max_residual = np.abs(features['residuals']).max()
            residuals_by_class[features['error_family']].append(max_residual)

    if not residuals_by_class:
        print("No residual data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    classes = list(residuals_by_class.keys())
    data = [residuals_by_class[c] for c in classes]

    bp = ax1.boxplot(data, labels=classes, patch_artist=True)
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(classes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Max |Residual|', fontsize=12)
    ax1.set_title('Maximum Residual by Error Family', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Violin plot
    positions = range(1, len(classes) + 1)
    parts = ax2.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors[:len(classes)]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(classes, rotation=45)
    ax2.set_ylabel('Max |Residual|', fontsize=12)
    ax2.set_title('Residual Distribution by Error Family', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_tool_call_patterns(samples: List[Dict], output_path: str = None):
    """Plot tool call patterns."""
    tool_calls_by_class = defaultdict(list)

    for sample in samples:
        features = extract_features(sample)
        if features['error_family']:
            tool_calls_by_class[features['error_family']].append(features['tool_call_count'])

    if not tool_calls_by_class:
        print("No tool call data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Average tool calls
    classes = list(tool_calls_by_class.keys())
    avg_calls = [np.mean(tool_calls_by_class[c]) for c in classes]

    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    ax1.bar(classes, avg_calls, color=colors[:len(classes)], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Tool Calls', fontsize=12)
    ax1.set_title('Average Tool Calls by Error Family', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for i, (cls, avg) in enumerate(zip(classes, avg_calls)):
        ax1.text(i, avg + 0.05, f'{avg:.2f}', ha='center', va='bottom', fontweight='bold')

    # Distribution of tool calls
    all_tool_calls = []
    for samples_list in tool_calls_by_class.values():
        all_tool_calls.extend(samples_list)

    tool_call_counts = defaultdict(int)
    for count in all_tool_calls:
        tool_call_counts[count] += 1

    calls = sorted(tool_call_counts.keys())
    counts = [tool_call_counts[c] for c in calls]

    ax2.bar(calls, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Tool Calls', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Tool Call Counts', fontsize=14, fontweight='bold')
    ax2.set_xticks(calls)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_confidence_analysis(samples: List[Dict], output_path: str = None):
    """Plot confidence score analysis."""
    confidence_by_class = defaultdict(list)

    for sample in samples:
        features = extract_features(sample)
        if features['confidence'] is not None and features['error_family']:
            confidence_by_class[features['error_family']].append(features['confidence'])

    if not confidence_by_class:
        print("No confidence data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    classes = list(confidence_by_class.keys())
    data = [confidence_by_class[c] for c in classes]

    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bp = ax1.boxplot(data, labels=classes, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(classes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.set_ylim([0.95, 1.0])
    ax1.set_title('Confidence Scores by Error Family', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Histogram
    all_confidences = []
    for conf_list in confidence_by_class.values():
        all_confidences.extend(conf_list)

    ax2.hist(all_confidences, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(np.mean(all_confidences), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(all_confidences):.3f}')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def generate_summary_report(samples: List[Dict]):
    """Generate text summary report."""
    print("\n" + "="*60)
    print("DATASET ANALYSIS SUMMARY")
    print("="*60)

    # Basic stats
    print(f"\nTotal Samples: {len(samples)}")

    # Extract features
    all_features = [extract_features(s) for s in samples]

    # Class distribution
    class_counts = defaultdict(int)
    for f in all_features:
        if f['error_family']:
            class_counts[f['error_family']] += 1

    print("\nError Family Distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(samples) * 100
        print(f"  {cls:20s}: {count:5d} ({pct:5.1f}%)")

    # Tool call stats
    tool_calls = [f['tool_call_count'] for f in all_features]
    print(f"\nTool Call Statistics:")
    print(f"  Min:  {min(tool_calls)}")
    print(f"  Max:  {max(tool_calls)}")
    print(f"  Mean: {np.mean(tool_calls):.2f}")

    # Confidence stats
    confidences = [f['confidence'] for f in all_features if f['confidence'] is not None]
    if confidences:
        print(f"\nConfidence Statistics:")
        print(f"  Min:  {min(confidences):.3f}")
        print(f"  Max:  {max(confidences):.3f}")
        print(f"  Mean: {np.mean(confidences):.3f}")

    # Measurement stats
    z_obs_list = [f['z_obs'] for f in all_features[:100] if f['z_obs'] is not None]
    if z_obs_list:
        z_matrix = np.array(z_obs_list)
        print(f"\nMeasurement Statistics (from 100 samples):")
        print(f"  Voltages (0-13):")
        print(f"    Range: [{z_matrix[:, 0:14].min():.4f}, {z_matrix[:, 0:14].max():.4f}] p.u.")
        print(f"    Mean:  {z_matrix[:, 0:14].mean():.4f} p.u.")
        print(f"  Powers (14-121):")
        print(f"    Range: [{z_matrix[:, 14:].min():.4f}, {z_matrix[:, 14:].max():.4f}] p.u.")
        print(f"    Mean:  {z_matrix[:, 14:].mean():.4f} p.u.")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize Power System Diagnostic Dataset'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='data/sft_with_tools.jsonl',
        help='Path to dataset file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/analysis',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Save plots instead of showing them'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.file}...")
    samples = load_dataset(args.file)
    print(f"Loaded {len(samples)} samples")

    # Create output directory
    if args.no_show:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to {output_dir}/")

    # Generate summary
    generate_summary_report(samples)

    # Generate plots
    print("\nGenerating visualizations...")

    output_path = lambda name: str(Path(args.output_dir) / name) if args.no_show else None

    print("  - Class distribution...")
    plot_class_distribution(samples, output_path('class_distribution.png'))

    print("  - Measurement distributions...")
    plot_measurement_distributions(samples, output_path('measurement_distributions.png'))

    print("  - Residual analysis...")
    plot_residual_analysis(samples, output_path('residual_analysis.png'))

    print("  - Tool call patterns...")
    plot_tool_call_patterns(samples, output_path('tool_call_patterns.png'))

    print("  - Confidence analysis...")
    plot_confidence_analysis(samples, output_path('confidence_analysis.png'))

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
