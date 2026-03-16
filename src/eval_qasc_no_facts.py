#!/usr/bin/env python3
"""
Isolated evaluation: Flan-T5-Large on QASC **without** fact injection.

Motivation
----------
QASC with injected facts produces a ceiling effect (~99% accuracy on
Flan-T5-Large).  This script measures baseline accuracy when the model
must rely on its parametric knowledge alone.

Decision criteria
-----------------
    60–70 % accuracy  →  keep QASC without fact injection
    ≤ 20 % accuracy   →  discard QASC; migrate to CSQA / GSM8K

Usage
-----
    python src/eval_qasc_no_facts.py                   # default N=500, 3 seeds
    python src/eval_qasc_no_facts.py --sample-size 200 # quick sanity check
    python src/eval_qasc_no_facts.py --seeds 2266      # single seed
"""

import argparse
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EXPERIMENT_SEEDS
from perturbations import set_all_seeds
from models import get_device
from model_handlers import get_model_handler, ModelHandler
from dataset_handlers import QASCHandler

MODEL_KEY = "flan-t5-large"


def evaluate_single_seed(
    model_handler: ModelHandler, dataset, handler: QASCHandler,
    sample_size: int, seed: int,
) -> dict:
    """Run QASC evaluation for one seed and return metrics."""
    items = handler.sample_items(dataset, sample_size, seed)
    batch_size = model_handler.batch_size
    max_tokens = handler.get_max_tokens("control")

    correct = 0
    total = 0
    predictions: List[dict] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts = [handler.build_prompt(item, "control") for item in batch]
        responses = model_handler.generate(prompts, max_new_tokens=max_tokens)

        for item, response in zip(batch, responses):
            predicted = handler.parse_answer(response)
            actual = handler.get_correct_answer(item)
            is_correct = predicted == actual
            if predicted:
                total += 1
                if is_correct:
                    correct += 1
            predictions.append({
                "question": item["question"][:80],
                "predicted": predicted,
                "actual": actual,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sample_size": sample_size,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Flan-T5-Large on QASC without fact injection"
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Number of items per seed (default: 500)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(EXPERIMENT_SEEDS),
        help=f"Seeds to evaluate (default: {EXPERIMENT_SEEDS})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/results/qasc_no_facts",
        help="Directory for result JSON files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("QASC BASELINE EVALUATION — NO FACT INJECTION")
    print(f"Model:       {MODEL_KEY}")
    print(f"Sample size: {args.sample_size}")
    print(f"Seeds:       {args.seeds}")
    print("=" * 70)

    handler = QASCHandler(inject_facts=False)
    device = get_device()
    print(f"Device: {device}")

    model_handler = get_model_handler(MODEL_KEY)
    model_handler.load(device)
    dataset = handler.load()

    all_results = []
    for seed in args.seeds:
        set_all_seeds(seed)
        print(f"\n--- Seed {seed} ---")
        start = time.time()
        result = evaluate_single_seed(
            model_handler, dataset, handler, args.sample_size, seed,
        )
        elapsed = time.time() - start
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy'] * 100:.1f}%  ({result['correct']}/{result['total']})  [{elapsed:.1f}s]")

    # Summary across seeds
    accuracies = [r["accuracy"] for r in all_results]
    mean_acc = sum(accuracies) / len(accuracies)
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 70)
    for r in all_results:
        print(f"  Seed {r['seed']:>5}: {r['accuracy'] * 100:5.1f}%")
    print(f"\n  Mean accuracy: {mean_acc * 100:.1f}%")

    if mean_acc >= 0.60:
        print("\n  RECOMMENDATION: Keep QASC without fact injection (accuracy in 60-70%+ range).")
    elif mean_acc <= 0.20:
        print("\n  RECOMMENDATION: Discard QASC — accuracy has plummeted. Migrate to CSQA / GSM8K.")
    else:
        print(f"\n  RECOMMENDATION: Accuracy is {mean_acc * 100:.1f}% — borderline. Investigate further.")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"no_facts_N{args.sample_size}.json")
    save_data = {
        "model": MODEL_KEY,
        "inject_facts": False,
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "mean_accuracy": mean_acc,
        "per_seed": [
            {k: v for k, v in r.items() if k != "predictions"} for r in all_results
        ],
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
