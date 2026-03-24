#!/usr/bin/env python3
"""
Advanced Visualization script for Stability by Design.
Aligns results with NAACL 2024 paper and Project Proposal styles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

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

# Group prompt styles as per proposal
STYLE_GROUPS = {
    'Control': ['Control'],
    'Cognitive': ['Metacognition'],
    'Formatting': ['Structure'],
    'Social': ['Politeness']
}

def create_double_axis_plot(df, model, dataset, output_dir):
    """
    Creates Figure 2 style dual-axis plots: Accuracy (line) and Sensitivity (line).
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

    # Sort to ensure consistent order
    agg['prompt_style'] = pd.Categorical(agg['prompt_style'], categories=['Control', 'Metacognition', 'Structure', 'Politeness'], ordered=True)
    agg = agg.sort_values('prompt_style')

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Axis 1: Accuracy (Blue)
    color_acc = '#1f77b4'
    ax1.set_xlabel('Prompt Style')
    ax1.set_ylabel('Accuracy', color=color_acc)
    ax1.plot(agg['prompt_style'], agg['acc_mean'], marker='o', color=color_acc, label='Accuracy', linewidth=3)
    ax1.fill_between(agg['prompt_style'], agg['acc_mean'] - agg['acc_std'], agg['acc_mean'] + agg['acc_std'], color=color_acc, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 1.05)

    # Axis 2: Sensitivity (Red/Orange)
    ax2 = ax1.twinx()
    color_vr = '#ff7f0e'
    ax2.set_ylabel('Variation Ratio (Sensitivity)', color=color_vr)
    ax2.plot(agg['prompt_style'], agg['vr_mean'], marker='s', color=color_vr, label='Sensitivity', linestyle='--', linewidth=3)
    ax2.fill_between(agg['prompt_style'], agg['vr_mean'] - agg['vr_std'], agg['vr_mean'] + agg['vr_std'], color=color_vr, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color_vr)
    ax2.set_ylim(0, 1.05)

    plt.title(f"{model.upper()} on {dataset.upper()}\nAccuracy vs. Stability Comparison")
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    save_path = os.path.join(output_dir, f"dual_axis_{model}_{dataset}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_style_impact_bar(df, output_dir):
    """
    Creates Figure 3 style bar plot comparing sensitivity across styles.
    """
    plt.figure(figsize=(10, 6))
    
    # Capitalize for labels
    plot_df = df.copy()
    plot_df['prompt_style'] = plot_df['prompt_style'].str.capitalize()
    
    sns.barplot(
        data=plot_df, x='prompt_style', y='variation_ratio', hue='model',
        capsize=.1, errorbar='sd', palette="viridis"
    )
    
    plt.axhline(0.1, ls='--', color='gray', alpha=0.5, label='Stability Baseline')
    plt.title("Impact of Prompt Design on Output Stability (VR)")
    plt.ylabel("Variation Ratio (Lower is Better)")
    plt.xlabel("Independent Variable (Prompt Property)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.savefig(os.path.join(output_dir, "style_sensitivity_bars.png"), bbox_inches='tight')
    plt.close()

def plot_global_correlation(df, output_dir):
    """
    Global scatter plot of VR vs Accuracy with Pearson R.
    """
    plt.figure(figsize=(8, 6))
    
    # Pearson Correlation Calculation
    corr = df['variation_ratio'].corr(df['accuracy'])
    
    sns.regplot(
        data=df, x='variation_ratio', y='accuracy', 
        scatter_kws={'alpha':0.5, 's':100}, line_kws={'color':'red'}
    )
    
    plt.text(0.05, 0.95, f'Pearson r = {corr:.2f}', transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.title("Global Correlation: Sensitivity vs. Performance")
    plt.xlabel("Variation Ratio (Sensitivity)")
    plt.ylabel("Accuracy")
    
    plt.savefig(os.path.join(output_dir, "global_correlation.png"), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default="outputs/results/all_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"File not found: {args.results_csv}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    
    # Ensure numeric and clean
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['variation_ratio'] = pd.to_numeric(df['variation_ratio'], errors='coerce')
    df['prompt_style'] = df['prompt_style'].str.capitalize()

    # 1. Dual Axis plots per Model/Dataset (Figure 2 Style)
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            create_double_axis_plot(df, model, dataset, args.output_dir)

    # 2. Impact Bars (Figure 3 Style)
    plot_style_impact_bar(df, args.output_dir)

    # 3. Global Correlation
    plot_global_correlation(df, args.output_dir)

    print(f"Enhanced figures generated in {args.output_dir}")

if __name__ == "__main__":
    main()
