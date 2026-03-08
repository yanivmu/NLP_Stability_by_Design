#!/usr/bin/env python3
"""
Unified Sensitivity Experiment Runner

Runs sensitivity experiments for any supported model and dataset combination.
Generates perturbations on-the-fly and tests prompt properties.

Usage:
    python run_experiment.py --model flan-t5-base --dataset qasc
    python run_experiment.py --model pythia-410m --dataset cola
    python run_experiment.py --model llama-3.2-1b --dataset qasc --sample-size 50
    python run_experiment.py --model flan-t5-base --dataset qasc --num-perturbations 5 --words-to-replace 2

Supported Models: flan-t5-base, flan-t5-large, pythia-410m, llama-3.2-1b
Supported Datasets: qasc, cola (extensible via datasets_config.py)
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

from config import ExperimentConfig
from perturbations import (
    set_all_seeds,
    generate_and_validate as generate_synonym_perturbations,
    generate_paraphrase_perturbations,
)
from models import (
    get_model_config, load_model_and_tokenizer, run_inference, get_device, ModelConfig
)
from datasets_config import (
    get_dataset_config, load_dataset, get_item_text, get_correct_answer,
    convert_to_list, DatasetConfig, AnswerType
)
from prompts import get_prompt_styles, get_max_tokens
from data_analysis import ResultAnalyzer


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into an ExperimentConfig."""
    defaults = ExperimentConfig()

    parser = argparse.ArgumentParser(description="Run sensitivity experiment")
    parser.add_argument("--model", required=True,
                        help="Model key (e.g. flan-t5-base, pythia-410m)")
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (e.g. qasc, cola)")
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size,
                        help=f"Number of samples for sensitivity experiments (default: {defaults.sample_size})")
    parser.add_argument("--num-perturbations", type=int, default=defaults.num_perturbations,
                        help=f"Perturbations per item (default: {defaults.num_perturbations})")
    parser.add_argument("--words-to-replace", type=int, default=defaults.words_to_replace,
                        help=f"Synonym swaps per perturbation (default: {defaults.words_to_replace})")
    parser.add_argument("--ootb-size", type=int, default=defaults.ootb_size,
                        help=f"Number of samples for OOTB check (default: {defaults.ootb_size})")
    parser.add_argument("--skip-ootb", action="store_true", default=defaults.skip_ootb,
                        help="Skip the OOTB accuracy check")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help=f"Random seed (default: {defaults.seed})")
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help=f"Output directory for results (default: {defaults.output_dir})")
    parser.add_argument("--perturbation-method", type=str, default=defaults.perturbation_method,
                        choices=["synonym", "paraphrase"],
                        help=f"Perturbation strategy (default: {defaults.perturbation_method})")
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
    )


def run_ootb_accuracy_check(
    model, tokenizer, dataset, model_config: ModelConfig, dataset_config: DatasetConfig,
    dataset_key: str, prompt_fn, device: str, sample_size: int, seed: int
) -> Tuple[float, int, int]:
    """Run Out-Of-The-Box accuracy check using Control prompt."""
    random.seed(seed)
    analyzer = ResultAnalyzer()

    data_list = convert_to_list(dataset, dataset_key)
    if dataset_key == "cola":
        sample_items = random.sample(data_list, min(sample_size, len(data_list)))
    else:
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        sample_items = [dataset[i] for i in indices]

    correct = 0
    total = 0
    batch_size = 4 if model_config.use_fp16 else 8

    for i in range(0, len(sample_items), batch_size):
        batch_items = sample_items[i:i+batch_size]

        if dataset_key == "qasc":
            prompts = [prompt_fn(item) for item in batch_items]
        else:  # cola
            prompts = [prompt_fn(item["sentence"]) for item in batch_items]

        responses = run_inference(
            model, tokenizer, prompts, device,
            model_config.model_type, model_config.default_max_tokens
        )

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
    cfg: ExperimentConfig, is_structured: bool
) -> Dict:
    """Run sensitivity experiment for a single prompt style."""
    random.seed(cfg.seed)
    analyzer = ResultAnalyzer()
    max_tokens = get_max_tokens(style_name, dataset_key)

    data_list = convert_to_list(dataset, dataset_key)
    if dataset_key == "cola":
        sample_items = random.sample(data_list, min(cfg.sample_size, len(data_list)))
    else:
        indices = random.sample(range(len(dataset)), min(cfg.sample_size, len(dataset)))
        sample_items = [dataset[i] for i in indices]

    results = []
    total_variation = 0.0

    for idx, item in enumerate(sample_items):
        base_text = get_item_text(item, dataset_key)
        if cfg.perturbation_method == "paraphrase":
            perturbations = generate_paraphrase_perturbations(
                base_text,
                num=cfg.num_perturbations,
                device=device,
                seed=cfg.seed,
            )
        else:
            perturbations = generate_synonym_perturbations(
                base_text,
                num=cfg.num_perturbations,
                words_to_replace=cfg.words_to_replace,
            )

        # Build prompts: original + perturbations
        if dataset_key == "qasc":
            prompts = [prompt_fn(item)]
            for p in perturbations:
                prompts.append(prompt_fn(item, p))
        else:  # cola
            all_sentences = [base_text] + perturbations
            prompts = [prompt_fn(s) for s in all_sentences]

        responses = run_inference(
            model, tokenizer, prompts, device,
            model_config.model_type, max_tokens
        )

        if dataset_config.answer_type == AnswerType.LETTER:
            answers = [analyzer.parse_letter_answer(r, is_structured=is_structured) for r in responses]
        else:  # YES_NO
            answers = [analyzer.parse_yes_no_answer(r, is_structured=is_structured) for r in responses]

        valid_answers = [a for a in answers if a]
        if len(valid_answers) >= 2:
            variation_ratio = analyzer.calculate_variation_ratio(valid_answers)
        else:
            variation_ratio = 0.0

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
        "seed": cfg.seed
    }


def main():
    """Main experiment runner."""
    cfg = parse_args()

    # Fix ALL random seeds for reproducibility
    set_all_seeds(cfg.seed)

    print("=" * 70)
    print(f"SENSITIVITY EXPERIMENT: {cfg.model_key.upper()} on {cfg.dataset_key.upper()}")
    print(f"Config: {cfg.summary()}")
    print("=" * 70)

    model_config = get_model_config(cfg.model_key)
    dataset_config = get_dataset_config(cfg.dataset_key)

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Perturbation method: {cfg.perturbation_method}")

    # Pre-load paraphraser if needed (so it's ready before the main model)
    if cfg.perturbation_method == "paraphrase":
        print("Loading paraphrase model (flan-t5-small)...")
        from perturbations import get_paraphraser
        get_paraphraser(device=device, seed=cfg.seed)
        print("Paraphraser ready.")

    try:
        model, tokenizer = load_model_and_tokenizer(model_config, device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        if "llama" in cfg.model_key.lower():
            print("\nTo authenticate, run: huggingface-cli login")
        sys.exit(1)

    dataset = load_dataset(dataset_config)

    prompt_styles = get_prompt_styles(cfg.dataset_key)

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
        print(f"Random baseline: {dataset_config.random_baseline*100:.1f}%")
        print(f"Valid signal threshold: {dataset_config.valid_threshold*100:.1f}%")

        start_time = time.time()
        accuracy, correct, total = run_ootb_accuracy_check(
            model=model, tokenizer=tokenizer, dataset=dataset,
            model_config=model_config, dataset_config=dataset_config,
            dataset_key=cfg.dataset_key, prompt_fn=prompt_styles["control"],
            device=device, sample_size=cfg.ootb_size, seed=cfg.seed
        )
        elapsed = time.time() - start_time

        print(f"\n{'='*50}")
        print(f"OOTB ACCURACY RESULT: {accuracy*100:.1f}% ({correct}/{total})")
        print(f"{'='*50}")
        print(f"Time: {elapsed:.1f}s")

        if accuracy < dataset_config.random_baseline:
            print(f"\nWARNING: Accuracy ({accuracy*100:.1f}%) is BELOW random baseline!")
            print("   Results may not be meaningful, but proceeding anyway...")
        elif accuracy < dataset_config.valid_threshold:
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
        if style_name not in prompt_styles:
            print(f"\n[{style_name.upper()}] SKIPPED -- not registered for {cfg.dataset_key}")
            continue

        prompt_fn = prompt_styles[style_name]
        print(f"\n[{style_name.upper()}] Running experiment...")
        start_time = time.time()

        is_structured = (style_name == "structure")

        result = run_sensitivity_experiment(
            model=model, tokenizer=tokenizer, dataset=dataset,
            model_config=model_config, dataset_config=dataset_config,
            dataset_key=cfg.dataset_key, prompt_fn=prompt_fn, style_name=style_name,
            device=device, cfg=cfg, is_structured=is_structured
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
    if not cfg.skip_ootb:
        print(f"\nOOTB Accuracy (Control prompt, N={total}): {accuracy*100:.1f}%")
    print(f"\n{'Prompt Style':<15} | {'VR':<8} | {'Accuracy':<12} | {'Interpretation'}")
    print("-" * 70)

    for style_name, result in all_results.items():
        vr = result['avg_variation_ratio']
        acc = result['accuracy']
        interp = "Stable" if vr < 0.2 else "Moderate" if vr < 0.4 else "Unstable"
        print(f"{style_name:<15} | {vr:.4f}   | {acc*100:5.1f}%       | {interp}")

    if all_results:
        best_vr = min(all_results.keys(), key=lambda k: all_results[k]['avg_variation_ratio'])
        worst_vr = max(all_results.keys(), key=lambda k: all_results[k]['avg_variation_ratio'])
        best_acc = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
        worst_acc = min(all_results.keys(), key=lambda k: all_results[k]['accuracy'])

        print("\n" + "-" * 70)
        print(f"Most Stable (lowest VR): {best_vr} (VR={all_results[best_vr]['avg_variation_ratio']:.4f})")
        print(f"Least Stable (highest VR): {worst_vr} (VR={all_results[worst_vr]['avg_variation_ratio']:.4f})")
        print(f"Highest Accuracy: {best_acc} ({all_results[best_acc]['accuracy']*100:.1f}%)")
        print(f"Lowest Accuracy: {worst_acc} ({all_results[worst_acc]['accuracy']*100:.1f}%)")

    # Save results
    output_subdir = os.path.join(cfg.output_dir, cfg.model_key)
    os.makedirs(output_subdir, exist_ok=True)

    method_suffix = f"_{cfg.perturbation_method}" if cfg.perturbation_method != "synonym" else ""
    output_file = os.path.join(
        output_subdir,
        f"sensitivity_results_{cfg.model_key}_{cfg.dataset_key}{method_suffix}.json",
    )

    save_data = {
        "model": model_config.hf_name,
        "model_key": cfg.model_key,
        "dataset": dataset_config.name,
        "dataset_key": cfg.dataset_key,
        "perturbation_method": cfg.perturbation_method,
        "ootb_accuracy": accuracy,
        "ootb_correct": correct,
        "ootb_total": total,
        "sample_size": cfg.sample_size,
        "num_perturbations": cfg.num_perturbations,
        "words_to_replace": cfg.words_to_replace,
        "seed": cfg.seed,
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
