#!/usr/bin/env python3
"""
Visualization script for NLP Stability experiments.
Generates plots from the consolidated all_results.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_correlation(df, output_dir):
    """Plot Accuracy vs. Variation Ratio correlation."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create the scatter plot with regression line
    sns.regplot(data=df, x='variation_ratio', y='accuracy', scatter_kws={'alpha':0.5})
    
    plt.title("Correlation: Variation Ratio vs. Accuracy", fontsize=14)
    plt.xlabel("Variation Ratio (Higher = More Sensitive)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    save_path = os.path.join(output_dir, "correlation_vr_accuracy.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved correlation plot to {save_path}")
    plt.close()

def plot_by_prompt_style(df, output_dir):
    """Plot Sensitivity by Prompt Style for each model/dataset."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Grouped bar plot
    sns.barplot(data=df, x='prompt_style', y='variation_ratio', hue='model')
    
    plt.title("Variation Ratio by Prompt Style across Models", fontsize=14)
    plt.xlabel("Prompt Style", fontsize=12)
    plt.ylabel("Variation Ratio (Lower is Better)", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path = os.path.join(output_dir, "sensitivity_by_style.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved prompt style comparison to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results.")
    parser.add_argument("--results-csv", type=str, default="outputs/results/all_results.csv",
                        help="Path to the consolidated results CSV")
    parser.add_argument("--output-dir", type=str, default="outputs/figures",
                        help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"Error: Results file not found at {args.results_csv}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.results_csv)
    
    # Ensure types are correct
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['variation_ratio'] = pd.to_numeric(df['variation_ratio'], errors='coerce')
    
    print(f"Found {len(df)} experiment rows in CSV.")
    
    # Generate Plots
    plot_correlation(df, args.output_dir)
    plot_by_prompt_style(df, args.output_dir)
    
    # Optional: Breakdown by Dataset
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        if not subset.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=subset, x='prompt_style', y='variation_ratio', hue='model')
            plt.title(f"Sensitivity on {dataset.upper()}", fontsize=14)
            plt.savefig(os.path.join(args.output_dir, f"sensitivity_{dataset.lower()}.png"), bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()
