#!/usr/bin/env python3
"""
Sensitivity Experiment: Flan-T5-Base on QASC
Generates perturbations on-the-fly and tests prompt properties.

This script:
1. Loads QASC dataset directly from HuggingFace
2. Generates N=10 semantic-preserving perturbations on-the-fly
3. Runs OOTB accuracy check before sensitivity experiments
4. Uses consolidated functions from data_analysis.py

Prompt properties tested: Control, Metacognition, Structure, Politeness
"""

import json
import torch
import random
from typing import Dict, List, Callable, Tuple
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from data_analysis import ResultAnalyzer, DataManager, generate_perturbations

# Seeds from reference project
SEEDS = [2266, 105, 86379]
NUM_PERTURBATIONS = 10

# Use thresholds from DataManager
QASC_RANDOM_BASELINE = DataManager.QASC_RANDOM_BASELINE  # 0.125 (1/8 for 8 choices)
QASC_VALID_THRESHOLD = DataManager.QASC_VALID_THRESHOLD  # 0.40 = valid signal


# =====================================================================
# PROMPT TEMPLATES FOR QASC (8-way multiple choice)
# =====================================================================

def format_qasc_base(item: Dict) -> str:
    """Format QASC item into base question string."""
    question = item["question"]
    fact1 = item["fact1"]
    fact2 = item["fact2"]
    choices = item["choices"]["text"]
    labels = item["choices"]["label"]
    
    choices_str = "\n".join([f"  {l}) {t}" for l, t in zip(labels, choices)])
    
    return f"""Given:
Fact 1: {fact1}
Fact 2: {fact2}

Question: {question}
Choices:
{choices_str}"""


def create_control_prompt(item: Dict, sentence: str = None) -> str:
    """Standard zero-shot instruction."""
    base = format_qasc_base(item) if sentence is None else sentence
    return f"""{base}

Answer with just the letter (A-H):"""


def add_metacognition(item: Dict, sentence: str = None) -> str:
    """Adds self-check triggers."""
    base = format_qasc_base(item) if sentence is None else sentence
    return f"""{base}

Think carefully about how the facts relate to the question.
Verify your reasoning before answering.
Answer with just the letter (A-H):"""


def add_structure(item: Dict, sentence: str = None) -> str:
    """Enforces strict JSON structured output."""
    base = format_qasc_base(item) if sentence is None else sentence
    return f"""{base}

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "X"}}

Where X is the letter A-H. Output only the JSON, nothing else."""


def add_politeness(item: Dict, sentence: str = None) -> str:
    """Adds conversational fillers."""
    base = format_qasc_base(item) if sentence is None else sentence
    return f"""Hello! I'd really appreciate your help with this question.

{base}

Please provide your answer (just the letter A-H). Thank you!"""


PROMPT_STYLES = {
    "control": create_control_prompt,
    "metacognition": add_metacognition,
    "structure": add_structure,
    "politeness": add_politeness,
}


def run_inference(model, tokenizer, prompts: List[str], device: str, max_new_tokens: int = 20) -> List[str]:
    """Run inference on Flan-T5 (seq2seq model)."""
    inputs = tokenizer(prompts, padding=True, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def run_ootb_accuracy_check(
    model, tokenizer, dataset, device: str,
    sample_size: int = 100, seed: int = 42
) -> Tuple[float, int, int]:
    """
    Run Out-Of-The-Box accuracy check using Control prompt.

    This verifies the model performs significantly better than random guessing
    (12.5% for QASC's 8 choices) before running sensitivity experiments.

    Args:
        model: The loaded Flan-T5 model
        tokenizer: The tokenizer
        dataset: QASC dataset
        device: Device to run on
        sample_size: Number of samples to evaluate
        seed: Random seed for sampling

    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    random.seed(seed)
    analyzer = ResultAnalyzer()

    # Sample items
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    correct = 0
    total = 0

    # Process in batches for efficiency
    batch_size = 8

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_items = [dataset[idx] for idx in batch_indices]

        # Create Control prompts
        prompts = [create_control_prompt(item) for item in batch_items]

        # Run inference
        responses = run_inference(model, tokenizer, prompts, device)

        # Check accuracy
        for item, response in zip(batch_items, responses):
            predicted = analyzer.parse_letter_answer(response)
            actual = item["answerKey"]

            if predicted == actual:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_sensitivity_experiment(
    model, tokenizer, dataset, prompt_fn: Callable,
    sample_size: int = 50, seed: int = 42, device: str = "mps",
    is_structured: bool = False
) -> Dict:
    """
    Run sensitivity experiment for a single prompt style.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dataset: QASC dataset
        prompt_fn: Function to create prompts
        sample_size: Number of samples to test
        seed: Random seed
        device: Device to run on
        is_structured: If True, use JSON parsing for Structure prompt

    Returns:
        Dictionary with results and average variation ratio
    """
    random.seed(seed)
    analyzer = ResultAnalyzer()

    # Sample items
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    results = []
    total_variation = 0.0

    # Use longer max_new_tokens for structured output (needs room for JSON)
    max_tokens = 100 if is_structured else 20

    for idx in indices:
        item = dataset[idx]

        # Generate perturbations of the base question
        base_text = format_qasc_base(item)
        perturbations = generate_perturbations(base_text, NUM_PERTURBATIONS)

        # Create prompts: original + perturbations
        prompts = [prompt_fn(item)]  # Original
        for p in perturbations:
            # For perturbations, we pass the perturbed base text
            prompts.append(prompt_fn(item, p))

        # Run inference
        responses = run_inference(model, tokenizer, prompts, device, max_new_tokens=max_tokens)

        # Parse answers (use consolidated parser with JSON support for Structure prompt)
        answers = [analyzer.parse_letter_answer(r, is_structured=is_structured) for r in responses]

        # Calculate variation ratio
        valid_answers = [a for a in answers if a]
        if len(valid_answers) >= 2:
            variation_ratio = analyzer.calculate_variation_ratio(valid_answers)
        else:
            variation_ratio = 0.0

        # Track accuracy: check if original (non-perturbed) answer is correct
        original_answer = answers[0] if answers else None
        correct_answer = item["answerKey"]
        is_correct = (original_answer == correct_answer) if original_answer else False

        total_variation += variation_ratio
        results.append({
            "idx": idx,
            "question": item["question"][:50],
            "answers": answers,
            "correct_answer": correct_answer,
            "original_correct": is_correct,
            "variation_ratio": variation_ratio
        })

    avg_variation = total_variation / len(results) if results else 0.0

    # Calculate accuracy for this prompt style
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


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    import time
    import sys

    print("=" * 70)
    print("SENSITIVITY EXPERIMENT: Flan-T5-Base on QASC")
    print("Testing prompt properties: Control, Metacognition, Structure, Politeness")
    print("=" * 70)
    print("\nFixes applied:")
    print("  ✓ OOTB accuracy check before sensitivity experiments")
    print("  ✓ Fact injection (fact1 + fact2) in all prompts")
    print("  ✓ Strict JSON parsing for Structure prompt")

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading Flan-T5-Base...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded")

    # Load QASC
    print("\nLoading QASC dataset...")
    dataset = load_dataset("allenai/qasc", split="validation")
    print(f"Loaded {len(dataset)} items")

    # =========================================================================
    # STEP 1: OOTB ACCURACY CHECK (CRUCIAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: OUT-OF-THE-BOX (OOTB) ACCURACY CHECK")
    print("=" * 70)
    print(f"\nVerifying model accuracy using Control prompt...")
    print(f"Random baseline for QASC (8 choices): {QASC_RANDOM_BASELINE*100:.1f}%")
    print(f"Valid signal threshold: {QASC_VALID_THRESHOLD*100:.1f}%")

    OOTB_SAMPLE_SIZE = 100
    SEED = 2266

    start_time = time.time()
    accuracy, correct, total = run_ootb_accuracy_check(
        model=model, tokenizer=tokenizer, dataset=dataset,
        device=device, sample_size=OOTB_SAMPLE_SIZE, seed=SEED
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"OOTB ACCURACY RESULT: {accuracy*100:.1f}% ({correct}/{total})")
    print(f"{'='*50}")
    print(f"Time: {elapsed:.1f}s")

    # Check if accuracy is valid
    if accuracy < QASC_RANDOM_BASELINE:
        print(f"\nCRITICAL: Accuracy ({accuracy*100:.1f}%) is BELOW random baseline ({QASC_RANDOM_BASELINE*100:.1f}%)!")
        print("   Model may be outputting invalid responses or consistently wrong.")
        print("   Aborting experiment.")
        sys.exit(1)
    elif accuracy < QASC_VALID_THRESHOLD:
        print(f"\nWARNING: Accuracy ({accuracy*100:.1f}%) is below valid threshold ({QASC_VALID_THRESHOLD*100:.1f}%).")
        print("   Results may not be meaningful. Proceeding with caution...")
    else:
        print(f"\nPASS: Accuracy ({accuracy*100:.1f}%) exceeds threshold ({QASC_VALID_THRESHOLD*100:.1f}%).")
        print("   Model shows valid signal. Proceeding with sensitivity experiments.")

    # =========================================================================
    # STEP 2: SENSITIVITY EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SENSITIVITY EXPERIMENTS")
    print("=" * 70)

    SAMPLE_SIZE = 30

    all_results = {}

    for style_name, prompt_fn in PROMPT_STYLES.items():
        print(f"\n[{style_name.upper()}] Running experiment...")
        start_time = time.time()

        # Use is_structured=True for Structure prompt to enable JSON parsing
        is_structured = (style_name == "structure")

        result = run_sensitivity_experiment(
            model=model, tokenizer=tokenizer, dataset=dataset,
            prompt_fn=prompt_fn, sample_size=SAMPLE_SIZE, seed=SEED, device=device,
            is_structured=is_structured
        )

        elapsed = time.time() - start_time
        all_results[style_name] = result

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Average Variation Ratio: {result['avg_variation_ratio']:.4f}")
        print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct_count']}/{result['num_samples']})")

    # Summary
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

    # Find best/worst by VR
    best_vr = min(all_results.keys(), key=lambda k: all_results[k]['avg_variation_ratio'])
    worst_vr = max(all_results.keys(), key=lambda k: all_results[k]['avg_variation_ratio'])

    # Find best/worst by accuracy
    best_acc = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    worst_acc = min(all_results.keys(), key=lambda k: all_results[k]['accuracy'])

    print("\n" + "-" * 70)
    print(f"Most Stable (lowest VR): {best_vr} (VR={all_results[best_vr]['avg_variation_ratio']:.4f})")
    print(f"Least Stable (highest VR): {worst_vr} (VR={all_results[worst_vr]['avg_variation_ratio']:.4f})")
    print(f"Highest Accuracy: {best_acc} ({all_results[best_acc]['accuracy']*100:.1f}%)")
    print(f"Lowest Accuracy: {worst_acc} ({all_results[worst_acc]['accuracy']*100:.1f}%)")

    # Save results
    output_file = "sensitivity_results_flan_qasc.json"
    save_data = {
        "model": model_name,
        "dataset": "QASC",
        "ootb_accuracy": accuracy,
        "ootb_correct": correct,
        "ootb_total": total,
        "sample_size": SAMPLE_SIZE,
        "num_perturbations": NUM_PERTURBATIONS,
        "seed": SEED,
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

