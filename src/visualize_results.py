#!/usr/bin/env python3
"""
Advanced Visualization script for Stability by Design.
Aggregates run CSVs by Phase and creates Control-centric baseline plots
with unique colors for each prompt style comparison.
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

# Define unique colors for each comparison style
STYLE_COLORS = {
    'Metacognition': '#2ca02c',  # Green
    'Structure': '#d62728',      # Red
    'Politeness': '#9467bd',     # Purple
    'Control': '#7f7f7f'         # Gray for the baseline point
}

def load_results_for_phase(base_dir: str, phase: str) -> pd.DataFrame:
    """Find all summary CSVs specifically for one phase."""
    phase_dir = os.path.join(base_dir, phase)
    if not os.path.exists(phase_dir):
        print(f"Error: Phase directory not found: {phase_dir}")
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(phase_dir, "**", "*.csv"), recursive=True)
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
    Creates dual-axis plots with unique colored lines from Control to other styles.
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
    
    # Axis 1: Accuracy
    ax1.set_xlabel('Prompt Style', fontweight='bold')
    ax1.set_ylabel('Accuracy (Solid Lines)', fontsize=12)
    ax1.set_ylim(0, 1.05)

    # Axis 2: Sensitivity
    ax2 = ax1.twinx()
    ax2.set_ylabel('Variation Ratio / Sensitivity (Dashed Lines)', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False) # Turn off grid for second axis to avoid overlap

    # Get Control baseline values
    control_data = agg[agg['prompt_style'] == 'Control']
    if control_data.empty:
        return
    c_acc = control_data['acc_mean'].values[0]
    c_vr = control_data['vr_mean'].values[0]

    # Plot points and add value labels
    for i, row in agg.iterrows():
        style = row['prompt_style']
        color = STYLE_COLORS.get(style, '#17becf')
        
        # Accuracy points + labels (placed ABOVE point)
        ax1.scatter(style, row['acc_mean'], color=color, s=150, zorder=10, edgecolors='black')
        ax1.text(style, row['acc_mean'] + 0.02, f"A: {row['acc_mean']:.2f}", 
                ha='center', va='bottom', color=color, fontweight='bold', fontsize=9)
        
        # Sensitivity points + labels (placed BELOW point)
        ax2.scatter(style, row['vr_mean'], color=color, marker='s', s=150, zorder=10, edgecolors='black')
        ax2.text(style, row['vr_mean'] - 0.01, f"V: {row['vr_mean']:.2f}", 
                ha='center', va='top', color=color, fontweight='bold', fontsize=9)

    # Draw lines for each style
    for i, row in agg.iterrows():
        style = row['prompt_style']
        if style == 'Control':
            continue
        
        color = STYLE_COLORS.get(style, '#17becf')
        
        # Line for Accuracy (Solid)
        ax1.plot(['Control', style], [c_acc, row['acc_mean']], 
                color=color, alpha=0.7, linestyle='-', linewidth=3)
        # Line for Sensitivity (Dashed)
        ax2.plot(['Control', style], [c_vr, row['vr_mean']], 
                color=color, alpha=0.7, linestyle='--', linewidth=3)

    plt.title(f"{phase.upper()}: {model.upper()} on {dataset.upper()}\nImpact of Prompt Attributes relative to Control", fontsize=14, pad=20)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='Accuracy'),
        Line2D([0], [0], color='black', linestyle='--', label='Sensitivity (VR)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLE_COLORS['Metacognition'], markersize=10, label='Metacognition'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLE_COLORS['Structure'], markersize=10, label='Structure'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLE_COLORS['Politeness'], markersize=10, label='Politeness'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

    # Save by phase
    pdf_phase_fig_dir = os.path.join(output_dir, phase, "pdf")
    png_phase_fig_dir = os.path.join(output_dir, phase, "png")
    os.makedirs(pdf_phase_fig_dir, exist_ok=True)
    os.makedirs(png_phase_fig_dir, exist_ok=True)
    
    base_save_path = os.path.join(png_phase_fig_dir, f"impact_plot_{model}_{dataset}")
    plt.savefig(f"{base_save_path}.png", bbox_inches='tight')
    base_save_path = os.path.join(pdf_phase_fig_dir, f"impact_plot_{model}_{dataset}")
    plt.savefig(f"{base_save_path}.pdf", bbox_inches='tight')
    plt.close()

def create_summary_impact_plot(df: pd.DataFrame, phase: str, output_dir: str):
    """
    Creates a global summary bar chart showing the average delta (change) 
    in Accuracy and Variation Ratio relative to the Control for each style.
    """
    # Filter out styles that aren't in our core set
    valid_styles = ['control', 'metacognition', 'structure', 'politeness']
    df = df[df['prompt_style'].str.lower().isin(valid_styles)].copy()
    df['prompt_style'] = df['prompt_style'].str.capitalize()
    
    # Calculate deltas per (model, dataset, seed) group
    groups = df.groupby(['model', 'dataset', 'seed'])
    delta_list = []
    
    for name, group in groups:
        control_rows = group[group['prompt_style'] == 'Control']
        if control_rows.empty:
            continue
        
        c_acc = control_rows['accuracy'].values[0]
        c_vr = control_rows['variation_ratio'].values[0]
        
        for i, row in group.iterrows():
            if row['prompt_style'] == 'Control':
                continue
            
            delta_list.append({
                'style': row['prompt_style'],
                'delta_acc': row['accuracy'] - c_acc,
                'delta_vr': row['variation_ratio'] - c_vr
            })
    
    if not delta_list:
        print("Warning: No deltas could be calculated for summary plot.")
        return
        
    delta_df = pd.DataFrame(delta_list)
    
    # Melt for grouped plotting
    plot_df = delta_df.melt(id_vars='style', value_vars=['delta_acc', 'delta_vr'],
                           var_name='Metric', value_name='Delta')
    plot_df['Metric'] = plot_df['Metric'].map({
        'delta_acc': 'Change in Accuracy',
        'delta_vr': 'Change in Sensitivity (VR)'
    })

    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x='style', y='Delta', hue='Metric', 
                palette=['#1f77b4', '#ff7f0e'], capsize=.1, errorbar='sd')
    
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.title(f"Global Summary ({phase.upper()}): Average Impact of Prompt Attributes\nRelative to Control Baseline (N={len(delta_df)})", 
              fontsize=16, pad=20)
    plt.ylabel("Absolute Change", fontsize=14)
    plt.xlabel("Prompt Attribute", fontsize=14)
    plt.legend(title="Metric Impact", loc='upper right')
    
    # Save to phase directory
    phase_fig_dir = os.path.join(output_dir, phase)
    os.makedirs(phase_fig_dir, exist_ok=True)
    base_save_path = os.path.join(phase_fig_dir, "summary_impact")
    plt.savefig(f"{base_save_path}.png", bbox_inches='tight')
    plt.savefig(f"{base_save_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Summary plots saved to {base_save_path}.png/pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="phase_1", help="Which phase to visualize")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--summary-only", action="store_true", help="Only generate the global summary plot")
    args = parser.parse_args()

    df = load_results_for_phase(args.results_dir, args.phase)
    if df.empty:
        print(f"No results found for {args.phase}.")
        return

    # Clean data
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['variation_ratio'] = pd.to_numeric(df['variation_ratio'], errors='coerce')
    df = df.dropna(subset=['accuracy', 'variation_ratio'])

    print(f"Generating analytics for: {args.phase}")

    # Global summary plot
    create_summary_impact_plot(df, args.phase, args.output_dir)

    if not args.summary_only:
        # Generate plots for every model/dataset combination
        for model in df['model'].unique():
            for dataset in df['dataset'].unique():
                create_control_centric_plot(df, model, dataset, args.phase, args.output_dir)

    print(f"Figures saved to {args.output_dir}/{args.phase}/")

if __name__ == "__main__":
    main()
