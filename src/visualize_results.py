#!/usr/bin/env python3
"""
Advanced Visualization script for Stability by Design.
Aggregates run CSVs by Phase and creates Control-centric baseline plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import glob
from typing import List

# Use high-quality plotting defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2.5
})
sns.set_theme(style="whitegrid")

def load_results_for_phase(base_dir: str, phase: str) -> pd.DataFrame:
    """Find all summary CSVs specifically for one phase."""
    phase_dir = os.path.join(base_dir, phase)
    if not os.path.exists(phase_dir):
        print(f"Error: Phase directory not found: {phase_dir}")
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(phase_dir, "**", "*.csv"), recursive=True)
    # Filter out detail CSVs
    csv_files = [f for f in csv_files if "detail" not in f]
    
    if not csv_files:
        return pd.DataFrame()
    
    print(f"Phase {phase}: Found {len(csv_files)} result files.")
    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            
    return pd.concat(df_list, ignore_index=True)

def create_control_centric_plot(df, model, dataset, phase, output_dir):
    """
    Creates dual-axis plots with explicit lines from Control to other styles.
    """
    subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
    if subset.empty:
        return

    # Aggregate over seeds
    agg = subset.groupby('prompt_style').agg({
        'accuracy': ['mean', 'std'],
        'variation_ratio': ['mean', 'std']
    }).reset_index()
    agg.columns = ['prompt_style', 'acc_mean', 'acc_std', 'vr_mean', 'vr_std']

    # Normalize style names
    agg['prompt_style'] = agg['prompt_style'].str.capitalize()
    styles = ['Control', 'Metacognition', 'Structure', 'Politeness']
    agg['prompt_style'] = pd.Categorical(agg['prompt_style'], categories=styles, ordered=True)
    agg = agg.sort_values('prompt_style')

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Axis 1: Accuracy (Blue)
    color_acc = '#1f77b4'
    ax1.set_xlabel('Prompt Style')
    ax1.set_ylabel('Accuracy', color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 1.05)

    # Axis 2: Sensitivity (Red/Orange)
    ax2 = ax1.twinx()
    color_vr = '#ff7f0e'
    ax2.set_ylabel('Variation Ratio (Sensitivity)', color=color_vr)
    ax2.tick_params(axis='y', labelcolor=color_vr)
    ax2.set_ylim(0, 1.05)

    # Plot points
    ax1.scatter(agg['prompt_style'], agg['acc_mean'], color=color_acc, s=120, zorder=5, edgecolors='black')
    ax2.scatter(agg['prompt_style'], agg['vr_mean'], color=color_vr, marker='s', s=120, zorder=5, edgecolors='black')

    # Explicit comparison lines from Control
    control_data = agg[agg['prompt_style'] == 'Control']
    if not control_data.empty:
        c_acc = control_data['acc_mean'].values[0]
        c_vr = control_data['vr_mean'].values[0]
        
        for i, row in agg.iterrows():
            if row['prompt_style'] == 'Control':
                continue
            
            # Line for Accuracy (Solid)
            ax1.plot(['Control', row['prompt_style']], [c_acc, row['acc_mean']], 
                    color=color_acc, alpha=0.4, linestyle='-', linewidth=2)
            # Line for Sensitivity (Dashed)
            ax2.plot(['Control', row['prompt_style']], [c_vr, row['vr_mean']], 
                    color=color_vr, alpha=0.4, linestyle='--', linewidth=2)

    plt.title(f"{phase.upper()}: {model.upper()} on {dataset.upper()}\nComparison from Control Baseline")
    
    # Combined Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=color_acc, label='Accuracy', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color=color_vr, label='Sensitivity (VR)', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], color='gray', linestyle='-', label='Acc Change', alpha=0.5),
        Line2D([0], [0], color='gray', linestyle='--', label='VR Change', alpha=0.5)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

    # Save by phase
    phase_fig_dir = os.path.join(output_dir, phase)
    os.makedirs(phase_fig_dir, exist_ok=True)
    
    save_path = os.path.join(phase_fig_dir, f"baseline_compare_{model}_{dataset}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="phase_1", help="Which phase to visualize")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    df = load_results_for_phase(args.results_dir, args.phase)
    if df.empty:
        print(f"No results found for {args.phase}.")
        return

    # Clean data
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['variation_ratio'] = pd.to_numeric(df['variation_ratio'], errors='coerce')

    print(f"Generating Phase-specific analytics for: {args.phase}")

    # Generate plots for every model/dataset combination in this phase
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            create_control_centric_plot(df, model, dataset, args.phase, args.output_dir)

    print(f"Figures saved to {args.output_dir}/{args.phase}/")

if __name__ == "__main__":
    main()
