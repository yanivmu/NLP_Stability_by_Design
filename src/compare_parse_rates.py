#!/usr/bin/env python3
"""
Compare Parse Rates Utility - Stability by Design

This script analyzes detailed results CSVs to compare the 'parseability' (is_valid_parse)
of model responses between the 'Control' prompt and a specific prompt 'Attribute'.

Usage:
    python src/compare_parse_rates.py --model <model_key> --dataset <dataset_key> --attribute <style_key>

Flag Options:
    --model:
        - flan-t5-base
        - flan-t5-large
        - pythia-410m
        - llama-3.2-1b
        - llama-3.2-1b-instruct
        - phi-3-mini
    
    --dataset:
        - cola (Grammaticality)
        - qasc (Multiple Choice)
        - csqa (Multiple Choice)
    
    --attribute (Prompt Style):
        - metacognition (Induces reasoning/thinking)
        - structure     (Enforces JSON output)
        - politeness    (Adds social framing)

Description:
    The script scans all 'detail.csv' files in the 'outputs/results' directory that match 
    the provided model and dataset. It aggregates the 'is_valid_parse' column to calculate
    what percentage of the model's responses were successfully parsed into a standard
    answer format (e.g., A-H for QASC or Yes/No for CoLA).
"""

import csv
import glob
import os
import argparse

def calculate_parse_comparison(model_key, dataset_key, attribute):
    # Pattern to find all relevant detailed CSV files
    # Scans through phase-specific and root results directories
    pattern = f"outputs/results/**/{model_key}/{dataset_key}/**/*detail.csv"
    files = glob.glob(pattern, recursive=True)

    if not files:
        # Fallback for different directory structures
        alt_pattern = f"outputs/results/{model_key}/{dataset_key}/**/*detail.csv"
        files = glob.glob(alt_pattern, recursive=True)

    if not files:
        print(f"No results found for Model: {model_key}, Dataset: {dataset_key}")
        print(f"Searched pattern: {pattern}")
        return

    stats = {
        'control': {'sum': 0, 'count': 0},
        attribute: {'sum': 0, 'count': 0}
    }

    processed_files = 0
    for file in files:
        processed_files += 1
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                style = row['prompt_style'].lower()
                # Ensure we handle various boolean string representations
                is_valid = row['is_valid_parse'].strip().lower() in ['true', '1', 'yes']
                
                if style in stats:
                    stats[style]['sum'] += 1 if is_valid else 0
                    stats[style]['count'] += 1

    print("=" * 70)
    print(f"PARSE RATE ANALYSIS")
    print(f"Model:     {model_key}")
    print(f"Dataset:   {dataset_key}")
    print(f"Data Source: {processed_files} detailed CSV files")
    print("-" * 70)

    for style in ['control', attribute]:
        s = stats[style]
        if s['count'] > 0:
            rate = (s['sum'] / s['count']) * 100
            print(f"{style.capitalize():<15}: {rate:>6.1f}% ({s['sum']}/{s['count']} responses)")
        else:
            print(f"{style.capitalize():<15}: No data found in the results files.")
    
    print("-" * 70)
    if stats['control']['count'] > 0 and stats[attribute]['count'] > 0:
        diff = (stats['control']['sum']/stats['control']['count']) - (stats[attribute]['sum']/stats[attribute]['count'])
        if diff > 0.1:
            print(f"OBSERVATION: {attribute.capitalize()} reduces parseability by {diff*100:.1f} percentage points.")
        elif diff < -0.1:
            print(f"OBSERVATION: {attribute.capitalize()} improves parseability by {-diff*100:.1f} percentage points.")
        else:
            print(f"OBSERVATION: No significant difference in parseability detected.")
    
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare parse rates between Control and an Attribute.")
    parser.add_argument("--model", required=True, help="Model key (e.g., llama-3.2-1b-instruct)")
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g., cola)")
    parser.add_argument("--attribute", required=True, help="Attribute key (e.g., metacognition, structure, politeness)")
    
    args = parser.parse_args()
    
    # Normalize inputs
    m_key = args.model.lower()
    d_key = args.dataset.lower()
    a_key = args.attribute.lower()
    
    calculate_parse_comparison(m_key, d_key, a_key)
