#!/usr/bin/env python3
"""
Unified Sensitivity Experiment Runner

Runs sensitivity experiments for any supported model and dataset combination.
Generates perturbations on-the-fly and tests prompt properties.

Usage:
    python run_experiment.py --model flan-t5-base --dataset qasc
    python run_experiment.py --model pythia-410m --dataset cola
    python run_experiment.py --model llama-3.2-1b-instruct --dataset qasc --sample-size 500
    python run_experiment.py --model flan-t5-base --dataset qasc --perturbation-method paraphrase

The pipeline is fully **model-agnostic** and **dataset-agnostic**: all
model-specific logic is delegated to ``ModelHandler`` (model_handlers.py)
and all dataset-specific logic to ``DatasetHandler`` (dataset_handlers.py).
Adding a new model or dataset requires *zero* changes to this file.
"""

import argparse
import json
import time
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, EXPERIMENT_SEEDS
from perturbations import (
    set_all_seeds,
    generate_and_validate as generate_synonym_perturbations,
    generate_paraphrase_perturbations,
)
from models import get_device
from model_handlers import get_model_handler, ModelHandler, list_registered_models
from dataset_handlers import get_dataset_handler, DatasetHandler, list_registered_datasets


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into an ExperimentConfig."""
    defaults = ExperimentConfig()

    registered_ds = ", ".join(list_registered_datasets())
    registered_m = ", ".join(list_registered_models())

    parser = argparse.ArgumentParser(description="Run sensitivity experiment")
    parser.add_argument("--model", required=True,
                        help=f"Model key. Registered: {registered_m}")
    parser.add_argument("--dataset", required=True,
                        help=f"Dataset key. Registered: {registered_ds}")
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size,
                        help=f"Number of samples (default: {defaults.sample_size})")
    parser.add_argument("--num-perturbations", type=int, default=defaults.num_perturbations,
                        help=f"Perturbations per item (default: {defaults.num_perturbations})")
    parser.add_argument("--words-to-replace", type=int, default=defaults.words_to_replace,
                        help=f"Synonym swaps per perturbation (default: {defaults.words_to_replace})")
    parser.add_argument("--ootb-size", type=int, default=defaults.ootb_size,
                        help=f"OOTB check samples (default: {defaults.ootb_size})")
    parser.add_argument("--skip-ootb", action="store_true", default=defaults.skip_ootb,
                        help="Skip the OOTB accuracy check")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help=f"Random seed (default: {defaults.seed})")
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help=f"Output directory (default: {defaults.output_dir})")
    parser.add_argument("--perturbation-method", type=str, default=defaults.perturbation_method,
                        choices=["synonym", "paraphrase"],
                        help=f"Perturbation strategy (default: {defaults.perturbation_method})")
    parser.add_argument("--facts", action="store_true", default=defaults.inject_facts,
                        help="Inject supporting facts into QASC prompts (default: off to avoid ceiling effect)")
    args = parser.parse_args()

    return ExperimentConfig(
        model_key=args.model,
        dataset_key=args.dataset,
        sample_size=args.sample_size,
        num_perturbations=args.num_perturbations,
        words_to_replace=args.words_to_replace,
        seed=args.seed,
        ootb_size=args.ootb_size,
        skip_ootb=args.skip_ootb,
        output_dir=args.output_dir,
        perturbation_method=args.perturbation_method,
        inject_facts=args.facts,
    )


# =====================================================================
# OOTB CHECK — fully model- and dataset-agnostic
# =====================================================================

def run_ootb_accuracy_check(
    model_handler: ModelHandler, dataset,
    dataset_handler: DatasetHandler, sample_size: int, seed: int,
) -> Tuple[float, int, int]:
    """Run Out-Of-The-Box accuracy check using Control prompt."""
    sample_items = dataset_handler.sample_items(dataset, sample_size, seed)
    batch_size = model_handler.batch_size
    max_tokens = dataset_handler.get_max_tokens("control")

    correct = 0
    total = 0

    for i in range(0, len(sample_items), batch_size):
        batch = sample_items[i:i + batch_size]
        prompts = [dataset_handler.build_prompt(item, "control") for item in batch]
        responses = model_handler.generate(prompts, max_new_tokens=max_tokens)

        for item, response in zip(batch, responses):
            actual = dataset_handler.get_correct_answer(item)
            predicted = dataset_handler.parse_answer(response)
            if predicted:
                total += 1
                if predicted == actual:
                    correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


# =====================================================================
# SENSITIVITY EXPERIMENT — fully model- and dataset-agnostic
# =====================================================================

def run_sensitivity_experiment(
    model_handler: ModelHandler, dataset,
    dataset_handler: DatasetHandler, style_name: str,
    cfg: ExperimentConfig,
) -> Dict:
    """Run sensitivity experiment for a single prompt style."""
    import random as _random

    sample_items = dataset_handler.sample_items(dataset, cfg.sample_size, cfg.seed)
    max_tokens = dataset_handler.get_max_tokens(style_name)
    is_structured = (style_name == "structure")

    results = []
    total_variation = 0.0

    for idx, item in enumerate(sample_items):
        # Per-item deterministic seed so each item's perturbations are
        # independent of previous items and reproducible across runs.
        item_seed = cfg.seed + idx
        _random.seed(item_seed)
        set_all_seeds(item_seed)

        base_text = dataset_handler.get_item_text(item)

        if cfg.perturbation_method == "paraphrase":
            perturbations = generate_paraphrase_perturbations(
                base_text, num=cfg.num_perturbations,
                device=model_handler.device, seed=item_seed,
            )
        else:
            perturbations = generate_synonym_perturbations(
                base_text, num=cfg.num_perturbations,
                words_to_replace=cfg.words_to_replace,
            )

        prompts = [dataset_handler.build_prompt(item, style_name)]
        for p in perturbations:
            prompts.append(dataset_handler.build_prompt(item, style_name, perturbed_text=p))

        responses = model_handler.generate(prompts, max_new_tokens=max_tokens)

        answers = [dataset_handler.parse_answer(r, is_structured=is_structured) for r in responses]
        valid_answers = [a for a in answers if a]

        if len(valid_answers) >= 2:
            from data_analysis import ResultAnalyzer
            variation_ratio = ResultAnalyzer().calculate_variation_ratio(valid_answers)
        else:
            variation_ratio = 0.0

        original_answer = answers[0] if answers else None
        correct_answer = dataset_handler.get_correct_answer(item)
        is_correct = (original_answer == correct_answer) if original_answer else False

        total_variation += variation_ratio
        results.append({
            "item_idx": idx,
            "correct_answer": correct_answer,
            "original_correct": is_correct,
            "answers": answers,
            "variation_ratio": variation_ratio,
        })

    avg_variation = total_variation / len(results) if results else 0.0
    correct_count = sum(1 for r in results if r["original_correct"])
    accuracy = correct_count / len(results) if results else 0.0

    return {
        "results": results,
        "avg_variation_ratio": avg_variation,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "num_samples": len(results),
        "seed": cfg.seed,
    }


# =====================================================================
# MAIN
# =====================================================================

import csv
from datetime import datetime

def save_to_csv(cfg: ExperimentConfig, all_results: Dict, ootb_acc: float, output_dir: str):
    """
    Saves experiment results in a flat CSV format, one row per prompt style.
    This format is ideal for plotting (Seaborn, Pandas, etc.).
    """
    csv_file = os.path.join(output_dir, "all_results.csv")
    file_exists = os.path.isfile(csv_file)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Header
        if not file_exists:
            writer.writerow([
                "timestamp", "model", "dataset", "method", "words_to_replace", 
                "num_perturbations", "sample_size", "seed", "ootb_accuracy",
                "prompt_style", "variation_ratio", "accuracy"
            ])
        
        # Rows (one per style)
        for style, res in all_results.items():
            writer.writerow([
                timestamp, cfg.model_key, cfg.dataset_key, cfg.perturbation_method,
                cfg.words_to_replace, cfg.num_perturbations, cfg.sample_size,
                cfg.seed, f"{ootb_acc:.4f}",
                style, f"{res['avg_variation_ratio']:.4f}", f"{res['accuracy']:.4f}"
            ])

def main():
    """Main experiment runner."""
    cfg = parse_args()
    set_all_seeds(cfg.seed)

    print("=" * 70)
    print(f"SENSITIVITY EXPERIMENT: {cfg.model_key.upper()} on {cfg.dataset_key.upper()}")
    print(f"Config: {cfg.summary()}")
    print("=" * 70)

    model_handler = get_model_handler(cfg.model_key)
    dataset_handler = get_dataset_handler(cfg.dataset_key, inject_facts=cfg.inject_facts)

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Perturbation method: {cfg.perturbation_method}")
    print(f"Available seeds for multi-run: {EXPERIMENT_SEEDS}")

    if cfg.perturbation_method == "paraphrase":
        print("Loading paraphrase model (flan-t5-small)...")
        from perturbations import get_paraphraser
        get_paraphraser(device=device, seed=cfg.seed)
        print("Paraphraser ready.")

    try:
        model_handler.load(device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        hint = model_handler.auth_hint()
        if hint:
            print(f"\n{hint}")
        sys.exit(1)

    dataset = dataset_handler.load()

    # =========================================================================
    # STEP 1: OOTB ACCURACY CHECK
    # =========================================================================
    accuracy = 0.0
    correct = 0
    total = 0

    if not cfg.skip_ootb:
        print("\n" + "=" * 70)
        print("STEP 1: OUT-OF-THE-BOX (OOTB) ACCURACY CHECK")
        print("=" * 70)
        print(f"\nVerifying model accuracy using Control prompt...")
        print(f"Random baseline: {dataset_handler.CONFIG.random_baseline * 100:.1f}%")
        print(f"Valid signal threshold: {dataset_handler.CONFIG.valid_threshold * 100:.1f}%")

        start_time = time.time()
        accuracy, correct, total = run_ootb_accuracy_check(
            model_handler=model_handler, dataset=dataset,
            dataset_handler=dataset_handler,
            sample_size=cfg.ootb_size, seed=cfg.seed,
        )
        elapsed = time.time() - start_time

        print(f"\n{'=' * 50}")
        print(f"OOTB ACCURACY RESULT: {accuracy * 100:.1f}% ({correct}/{total})")
        print(f"{'=' * 50}")
        print(f"Time: {elapsed:.1f}s")

        if accuracy < dataset_handler.CONFIG.random_baseline:
            print(f"\nWARNING: Accuracy ({accuracy * 100:.1f}%) is BELOW random baseline!")
            print("   Results may not be meaningful, but proceeding anyway...")
        elif accuracy < dataset_handler.CONFIG.valid_threshold:
            print(f"\nWARNING: Accuracy below threshold. Proceeding with caution...")
        else:
            print(f"\nPASS: Accuracy exceeds threshold. Proceeding with experiments.")
    else:
        print("\nSkipping OOTB check (--skip-ootb)")

    # =========================================================================
    # STEP 2: SENSITIVITY EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SENSITIVITY EXPERIMENTS")
    print("=" * 70)

    all_results = {}

    for style_name in cfg.prompt_styles:
        print(f"\n[{style_name.upper()}] Running experiment...")
        start_time = time.time()

        result = run_sensitivity_experiment(
            model_handler=model_handler, dataset=dataset,
            dataset_handler=dataset_handler,
            style_name=style_name, cfg=cfg,
        )

        elapsed = time.time() - start_time
        all_results[style_name] = result

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Average Variation Ratio: {result['avg_variation_ratio']:.4f}")
        print(f"  Accuracy: {result['accuracy'] * 100:.1f}% ({result['correct_count']}/{result['num_samples']})")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    if not cfg.skip_ootb:
        print(f"\nOOTB Accuracy (Control prompt, N={total}): {accuracy * 100:.1f}%")
    print(f"\n{'Prompt Style':<15} | {'VR':<8} | {'Accuracy':<12} | {'Interpretation'}")
    print("-" * 70)

    for style_name, result in all_results.items():
        vr = result["avg_variation_ratio"]
        acc = result["accuracy"]
        interp = "Stable" if vr < 0.2 else "Moderate" if vr < 0.4 else "Unstable"
        print(f"{style_name:<15} | {vr:.4f}   | {acc * 100:5.1f}%       | {interp}")

    if all_results:
        best_vr = min(all_results, key=lambda k: all_results[k]["avg_variation_ratio"])
        worst_vr = max(all_results, key=lambda k: all_results[k]["avg_variation_ratio"])
        best_acc = max(all_results, key=lambda k: all_results[k]["accuracy"])
        worst_acc = min(all_results, key=lambda k: all_results[k]["accuracy"])

        print("\n" + "-" * 70)
        print(f"Most Stable (lowest VR): {best_vr} (VR={all_results[best_vr]['avg_variation_ratio']:.4f})")
        print(f"Least Stable (highest VR): {worst_vr} (VR={all_results[worst_vr]['avg_variation_ratio']:.4f})")
        print(f"Highest Accuracy: {best_acc} ({all_results[best_acc]['accuracy'] * 100:.1f}%)")
        print(f"Lowest Accuracy: {worst_acc} ({all_results[worst_acc]['accuracy'] * 100:.1f}%)")

    # Save results
    output_subdir = os.path.join(
        cfg.output_dir, cfg.model_key, cfg.dataset_key, cfg.perturbation_method,
    )
    os.makedirs(output_subdir, exist_ok=True)

    filename_parts = [
        f"w{cfg.words_to_replace}" if cfg.perturbation_method == "synonym" else "",
        f"n{cfg.num_perturbations}",
        f"s{cfg.seed}",
    ]
    filename = "_".join(p for p in filename_parts if p) + ".json"
    output_file = os.path.join(output_subdir, filename)

    save_data = {
        "model": model_handler.hf_name,
        "model_key": cfg.model_key,
        "dataset": dataset_handler.CONFIG.name,
        "dataset_key": cfg.dataset_key,
        "perturbation_method": cfg.perturbation_method,
        "ootb_accuracy": accuracy,
        "ootb_correct": correct,
        "ootb_total": total,
        "sample_size": cfg.sample_size,
        "num_perturbations": cfg.num_perturbations,
        "words_to_replace": cfg.words_to_replace,
        "seed": cfg.seed,
        "results": {
            k: {
                "avg_variation_ratio": v["avg_variation_ratio"],
                "accuracy": v["accuracy"],
                "correct_count": v["correct_count"],
                "num_samples": v["num_samples"],
            }
            for k, v in all_results.items()
        },
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Also save to consolidated CSV for easy graphing
    save_to_csv(cfg, all_results, accuracy, cfg.output_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
