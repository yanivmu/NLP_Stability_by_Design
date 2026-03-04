#!/usr/bin/env python3
"""
Unified Sensitivity Experiment Runner

Runs sensitivity experiments for any supported model and dataset combination.
Generates perturbations on-the-fly and tests prompt properties.

Usage:
    python run_experiment.py --model flan-t5-base --dataset qasc
    python run_experiment.py --model pythia-410m --dataset cola
    python run_experiment.py --model llama-3.2-1b --dataset qasc --sample-size 50

Supported Models: flan-t5-base, flan-t5-large, pythia-410m, llama-3.2-1b
Supported Datasets: qasc, cola
"""

import argparse
import json
import time
import random
import sys
import os
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    get_model_config, load_model_and_tokenizer, run_inference, get_device, ModelConfig
)
from datasets_config import (
    get_dataset_config, load_dataset, get_item_text, get_correct_answer,
    convert_to_list, DatasetConfig, AnswerType
)
from prompts import get_prompt_styles, get_max_tokens
from data_analysis import ResultAnalyzer, generate_perturbations

# Default configuration
DEFAULT_SEED = 2266
NUM_PERTURBATIONS = 10


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run sensitivity experiment")
    parser.add_argument("--model", required=True, 
                        choices=["flan-t5-base", "flan-t5-large", "pythia-410m", "llama-3.2-1b"],
                        help="Model to use")
    parser.add_argument("--dataset", required=True,
                        choices=["qasc", "cola"],
                        help="Dataset to use")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of samples for sensitivity experiments (default: 30)")
    parser.add_argument("--ootb-size", type=int, default=100,
                        help="Number of samples for OOTB check (default: 100)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--output-dir", type=str, default="outputs/results",
                        help="Output directory for results")
    return parser.parse_args()


def run_ootb_accuracy_check(
    model, tokenizer, dataset, model_config: ModelConfig, dataset_config: DatasetConfig,
    dataset_key: str, prompt_fn, device: str, sample_size: int, seed: int
) -> Tuple[float, int, int]:
    """Run Out-Of-The-Box accuracy check using Control prompt."""
    random.seed(seed)
    analyzer = ResultAnalyzer()
    
    # Sample items
    data_list = convert_to_list(dataset, dataset_key)
    if dataset_key == "cola":
        sample_items = random.sample(data_list, min(sample_size, len(data_list)))
        indices = range(len(sample_items))
    else:
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        sample_items = [dataset[i] for i in indices]
    
    correct = 0
    total = 0
    batch_size = 4 if model_config.use_fp16 else 8
    
    for i in range(0, len(sample_items), batch_size):
        batch_items = sample_items[i:i+batch_size]
        
        # Create prompts based on dataset type
        if dataset_key == "qasc":
            prompts = [prompt_fn(item) for item in batch_items]
        else:  # cola
            prompts = [prompt_fn(item["sentence"]) for item in batch_items]
        
        responses = run_inference(
            model, tokenizer, prompts, device, 
            model_config.model_type, model_config.default_max_tokens
        )
        
        # Check accuracy
        for item, response in zip(batch_items, responses):
            actual = get_correct_answer(item, dataset_key)
            
            if dataset_config.answer_type == AnswerType.LETTER:
                predicted = analyzer.parse_letter_answer(response)
                if predicted == actual:
                    correct += 1
                total += 1
            else:  # YES_NO
                predicted = analyzer.parse_yes_no_answer(response)
                if predicted:
                    total += 1
                    if predicted == actual:
                        correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_sensitivity_experiment(
    model, tokenizer, dataset, model_config: ModelConfig, dataset_config: DatasetConfig,
    dataset_key: str, prompt_fn, style_name: str, device: str, 
    sample_size: int, seed: int, is_structured: bool
) -> Dict:
    """Run sensitivity experiment for a single prompt style."""
    random.seed(seed)
    analyzer = ResultAnalyzer()
    max_tokens = get_max_tokens(style_name, dataset_key)
    
    # Sample items
    data_list = convert_to_list(dataset, dataset_key)
    if dataset_key == "cola":
        sample_items = random.sample(data_list, min(sample_size, len(data_list)))
    else:
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        sample_items = [dataset[i] for i in indices]
    
    results = []
    total_variation = 0.0

    for idx, item in enumerate(sample_items):
        # Get text for perturbation
        base_text = get_item_text(item, dataset_key)
        perturbations = generate_perturbations(base_text, NUM_PERTURBATIONS)

        # Create prompts: original + perturbations
        if dataset_key == "qasc":
            prompts = [prompt_fn(item)]  # Original
            for p in perturbations:
                prompts.append(prompt_fn(item, p))
        else:  # cola
            all_sentences = [base_text] + perturbations
            prompts = [prompt_fn(s) for s in all_sentences]

        # Run inference
        responses = run_inference(
            model, tokenizer, prompts, device,
            model_config.model_type, max_tokens
        )

        # Parse answers
        if dataset_config.answer_type == AnswerType.LETTER:
            answers = [analyzer.parse_letter_answer(r, is_structured=is_structured) for r in responses]
        else:  # YES_NO
            answers = [analyzer.parse_yes_no_answer(r, is_structured=is_structured) for r in responses]

        # Calculate variation ratio
        valid_answers = [a for a in answers if a]
        if len(valid_answers) >= 2:
            variation_ratio = analyzer.calculate_variation_ratio(valid_answers)
        else:
            variation_ratio = 0.0

        # Check accuracy on original (non-perturbed) answer
        original_answer = answers[0] if answers else None
        correct_answer = get_correct_answer(item, dataset_key)
        is_correct = (original_answer == correct_answer) if original_answer else False

        total_variation += variation_ratio
        results.append({
            "item_idx": idx,
            "correct_answer": correct_answer,
            "original_correct": is_correct,
            "answers": answers,
            "variation_ratio": variation_ratio
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
        "seed": seed
    }


def main():
    """Main experiment runner."""
    args = parse_args()

    print("=" * 70)
    print(f"SENSITIVITY EXPERIMENT: {args.model.upper()} on {args.dataset.upper()}")
    print("Testing prompt properties: Control, Metacognition, Structure, Politeness")
    print("=" * 70)

    # Load configurations
    model_config = get_model_config(args.model)
    dataset_config = get_dataset_config(args.dataset)

    # Setup device
    device = get_device()
    print(f"\nDevice: {device}")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(model_config, device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        if "llama" in args.model.lower():
            print("\nTo authenticate, run: huggingface-cli login")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(dataset_config)

    # Get prompt styles
    prompt_styles = get_prompt_styles(args.dataset)

    # =========================================================================
    # STEP 1: OOTB ACCURACY CHECK
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: OUT-OF-THE-BOX (OOTB) ACCURACY CHECK")
    print("=" * 70)
    print(f"\nVerifying model accuracy using Control prompt...")
    print(f"Random baseline: {dataset_config.random_baseline*100:.1f}%")
    print(f"Valid signal threshold: {dataset_config.valid_threshold*100:.1f}%")

    start_time = time.time()
    accuracy, correct, total = run_ootb_accuracy_check(
        model=model, tokenizer=tokenizer, dataset=dataset,
        model_config=model_config, dataset_config=dataset_config,
        dataset_key=args.dataset, prompt_fn=prompt_styles["control"],
        device=device, sample_size=args.ootb_size, seed=args.seed
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"OOTB ACCURACY RESULT: {accuracy*100:.1f}% ({correct}/{total})")
    print(f"{'='*50}")
    print(f"Time: {elapsed:.1f}s")

    # Check validity
    if accuracy < dataset_config.random_baseline:
        print(f"\nCRITICAL: Accuracy ({accuracy*100:.1f}%) is BELOW random baseline!")
        print("   Aborting experiment.")
        sys.exit(1)
    elif accuracy < dataset_config.valid_threshold:
        print(f"\nWARNING: Accuracy below threshold. Proceeding with caution...")
    else:
        print(f"\nPASS: Accuracy exceeds threshold. Proceeding with experiments.")

    # =========================================================================
    # STEP 2: SENSITIVITY EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SENSITIVITY EXPERIMENTS")
    print("=" * 70)

    all_results = {}

    for style_name, prompt_fn in prompt_styles.items():
        print(f"\n[{style_name.upper()}] Running experiment...")
        start_time = time.time()

        is_structured = (style_name == "structure")

        result = run_sensitivity_experiment(
            model=model, tokenizer=tokenizer, dataset=dataset,
            model_config=model_config, dataset_config=dataset_config,
            dataset_key=args.dataset, prompt_fn=prompt_fn, style_name=style_name,
            device=device, sample_size=args.sample_size, seed=args.seed,
            is_structured=is_structured
        )

        elapsed = time.time() - start_time
        all_results[style_name] = result

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Average Variation Ratio: {result['avg_variation_ratio']:.4f}")
        print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct_count']}/{result['num_samples']})")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOOTB Accuracy (Control prompt, N={total}): {accuracy*100:.1f}%")
    print(f"\n{'Prompt Style':<15} | {'VR':<8} | {'Accuracy':<12} | {'Interpretation'}")
    print("-" * 70)

    for style_name, result in all_results.items():
        vr = result['avg_variation_ratio']
        acc = result['accuracy']
        interp = "Stable" if vr < 0.2 else "Moderate" if vr < 0.4 else "Unstable"
        print(f"{style_name:<15} | {vr:.4f}   | {acc*100:5.1f}%       | {interp}")

    # Find best/worst
    best_vr = min(all_results.keys(), key=lambda k: all_results[k]['avg_variation_ratio'])
    worst_vr = max(all_results.keys(), key=lambda k: all_results[k]['avg_variation_ratio'])
    best_acc = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    worst_acc = min(all_results.keys(), key=lambda k: all_results[k]['accuracy'])

    print("\n" + "-" * 70)
    print(f"Most Stable (lowest VR): {best_vr} (VR={all_results[best_vr]['avg_variation_ratio']:.4f})")
    print(f"Least Stable (highest VR): {worst_vr} (VR={all_results[worst_vr]['avg_variation_ratio']:.4f})")
    print(f"Highest Accuracy: {best_acc} ({all_results[best_acc]['accuracy']*100:.1f}%)")
    print(f"Lowest Accuracy: {worst_acc} ({all_results[worst_acc]['accuracy']*100:.1f}%)")

    # Save results to model-specific subfolder
    # Extract model family name (e.g., "flan" from "flan-t5-base", "pythia" from "pythia-410m")
    model_family = args.model.split("-")[0]
    output_subdir = os.path.join(args.output_dir, model_family)
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f"sensitivity_results_{args.model}_{args.dataset}.json")

    save_data = {
        "model": model_config.hf_name,
        "model_key": args.model,
        "dataset": dataset_config.name,
        "dataset_key": args.dataset,
        "ootb_accuracy": accuracy,
        "ootb_correct": correct,
        "ootb_total": total,
        "sample_size": args.sample_size,
        "num_perturbations": NUM_PERTURBATIONS,
        "seed": args.seed,
        "results": {k: {
            "avg_variation_ratio": v["avg_variation_ratio"],
            "accuracy": v["accuracy"],
            "correct_count": v["correct_count"],
            "num_samples": v["num_samples"]
        } for k, v in all_results.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
