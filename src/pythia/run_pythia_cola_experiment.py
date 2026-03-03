#!/usr/bin/env python3
"""
Sensitivity Experiment: Pythia-410M on CoLA
Tests prompt sensitivity using on-the-fly perturbation generation.
Evaluates different prompt properties: Control, Metacognition, Structure, Politeness.

This script:
1. Loads CoLA dataset directly from HuggingFace (no external dependencies)
2. Generates N=10 semantic-preserving perturbations on-the-fly
3. Runs OOTB accuracy check before sensitivity experiments
4. Uses consolidated functions from data_analysis.py
"""

import json
import torch
import random
from typing import Dict, List, Callable, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from data_analysis import ResultAnalyzer, DataManager, generate_perturbations

# Seeds from reference project
SEEDS = [2266, 105, 86379]
NUM_PERTURBATIONS = 10

# Use thresholds from DataManager
COLA_RANDOM_BASELINE = DataManager.COLA_RANDOM_BASELINE  # 0.50 for binary Yes/No
COLA_VALID_THRESHOLD = DataManager.COLA_VALID_THRESHOLD  # 0.60 accuracy = valid signal

# =====================================================================
# PROMPT TEMPLATES (mapping to project's prompt properties)
# =====================================================================

def create_control_prompt(sentence: str) -> str:
    """Standard zero-shot instruction (no special properties)."""
    return f"""Is this sentence grammatically correct? Answer Yes or No.

Sentence: "{sentence}"

Answer:"""


def add_metacognition(sentence: str) -> str:
    """Adds self-check triggers (e.g., 'verify your answer')."""
    return f"""Is this sentence grammatically correct? Answer Yes or No.
Before answering, carefully check the grammar rules. Verify your answer is correct.

Sentence: "{sentence}"

Think about it carefully, then answer:"""


def add_structure(sentence: str) -> str:
    """Enforces strict JSON structured output."""
    return f"""Analyze the grammatical correctness of the following sentence.

Sentence: "{sentence}"

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "Yes or No"}}

Output only the JSON, nothing else."""


def add_politeness(sentence: str) -> str:
    """Adds conversational fillers (e.g., 'please', 'I would appreciate')."""
    return f"""Hello! I would really appreciate your help with this.
Could you please tell me if this sentence is grammatically correct?
Please answer with Yes or No.

Sentence: "{sentence}"

Thank you! Your answer:"""


PROMPT_STYLES = {
    "control": create_control_prompt,
    "metacognition": add_metacognition,
    "structure": add_structure,
    "politeness": add_politeness,
}


def load_cola_dataset(split: str = "validation") -> List[Dict]:
    """
    Load CoLA dataset directly from HuggingFace.

    CoLA (Corpus of Linguistic Acceptability) is a binary classification task
    where sentences are labeled as grammatically acceptable (1) or not (0).

    Returns:
        List of dicts with 'sentence' and 'label' keys
    """
    dataset = load_dataset("nyu-mll/glue", "cola", split=split)

    data_list = []
    for item in dataset:
        data_list.append({
            "idx": item.get("idx", 0),
            "sentence": item.get("sentence", ""),
            "label": item.get("label", 0),  # 0=unacceptable, 1=acceptable
        })

    return data_list


def run_inference_batch(model, tokenizer, prompts: List[str], device: str, max_new_tokens: int = 10) -> List[str]:
    """Run inference on a batch of prompts."""
    inputs = tokenizer(prompts, padding=True, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=False,  # Greedy decoding for determinism
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs['input_ids'][i].shape[0]
        new_tokens = output[input_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def run_ootb_accuracy_check(
    model, tokenizer, dataset: List[Dict], device: str,
    sample_size: int = 100, seed: int = 42
) -> Tuple[float, int, int]:
    """
    Run Out-Of-The-Box accuracy check using Control prompt.

    This verifies the model performs significantly better than random guessing
    (50% for CoLA's binary Yes/No) before running sensitivity experiments.

    Args:
        model: The loaded Pythia model
        tokenizer: The tokenizer
        dataset: CoLA dataset as list of dicts with 'sentence' and 'label'
        device: Device to run on
        sample_size: Number of samples to evaluate
        seed: Random seed for sampling

    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    random.seed(seed)
    analyzer = ResultAnalyzer()

    # Sample items
    sample_items = random.sample(dataset, min(sample_size, len(dataset)))

    correct = 0
    total = 0

    # Process in batches for efficiency
    batch_size = 8

    for i in range(0, len(sample_items), batch_size):
        batch_items = sample_items[i:i+batch_size]

        # Create Control prompts using sentence
        prompts = [create_control_prompt(item["sentence"]) for item in batch_items]

        # Run inference
        responses = run_inference_batch(model, tokenizer, prompts, device)

        # Check accuracy
        for item, response in zip(batch_items, responses):
            predicted = analyzer.parse_yes_no_answer(response)
            # CoLA label: 1 = grammatical (YES), 0 = ungrammatical (NO)
            actual = "YES" if item["label"] == 1 else "NO"

            if predicted:  # Only count if we got a valid parse
                total += 1
                if predicted == actual:
                    correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_sensitivity_experiment(
    model,
    tokenizer,
    dataset: List[Dict],
    prompt_fn: Callable,
    sample_size: int = 50,
    seed: int = 42,
    device: str = "mps",
    is_structured: bool = False
) -> Dict:
    """
    Run sensitivity experiment for a single prompt style.

    Uses on-the-fly perturbation generation instead of pre-generated perturbations.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dataset: CoLA dataset as list of dicts with 'sentence' and 'label'
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
    sample_items = random.sample(dataset, min(sample_size, len(dataset)))

    results = []
    total_variation = 0.0

    # Use longer max_new_tokens for structured output (needs room for JSON)
    max_tokens = 50 if is_structured else 10

    for idx, item in enumerate(sample_items):
        original = item["sentence"]
        label = item["label"]

        # Generate perturbations on-the-fly (N=10)
        perturbations = generate_perturbations(original, NUM_PERTURBATIONS)

        # Create prompts: original + perturbations
        all_sentences = [original] + perturbations
        prompts = [prompt_fn(s) for s in all_sentences]

        # Run inference
        responses = run_inference_batch(model, tokenizer, prompts, device, max_new_tokens=max_tokens)

        # Parse answers (use JSON parsing for Structure prompt)
        answers = [analyzer.parse_yes_no_answer(r, is_structured=is_structured) for r in responses]

        # Calculate variation ratio
        valid_answers = [a for a in answers if a]
        if len(valid_answers) >= 2:
            variation_ratio = analyzer.calculate_variation_ratio(valid_answers)
        else:
            variation_ratio = 0.0

        # Track accuracy: check if original (non-perturbed) answer is correct
        original_answer = answers[0] if answers else None
        # CoLA label: 1 = grammatical (YES), 0 = ungrammatical (NO)
        correct_answer = "YES" if label == 1 else "NO"
        is_correct = (original_answer == correct_answer) if original_answer else False

        total_variation += variation_ratio
        results.append({
            "item_idx": idx,
            "original": original,
            "label": label,
            "correct_answer": correct_answer,
            "original_correct": is_correct,
            "answers": answers,
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
    print("SENSITIVITY EXPERIMENT: Pythia-410M on CoLA")
    print("Testing prompt properties: Control, Metacognition, Structure, Politeness")
    print("=" * 70)
    print("\nFixes applied:")
    print("  ✓ OOTB accuracy check before sensitivity experiments")
    print("  ✓ Strict JSON parsing for Structure prompt")

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading Pythia-410M...")
    model_name = "EleutherAI/pythia-410m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded")

    # Load CoLA dataset from HuggingFace (perturbations generated on-the-fly)
    print("\nLoading CoLA dataset from HuggingFace...")
    dataset = load_cola_dataset()
    print(f"Loaded {len(dataset)} items (perturbations generated on-the-fly)")

    # =========================================================================
    # STEP 1: OOTB ACCURACY CHECK (CRUCIAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: OUT-OF-THE-BOX (OOTB) ACCURACY CHECK")
    print("=" * 70)
    print(f"\nVerifying model accuracy using Control prompt...")
    print(f"Random baseline for CoLA (binary): {COLA_RANDOM_BASELINE*100:.1f}%")
    print(f"Valid signal threshold: {COLA_VALID_THRESHOLD*100:.1f}%")

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
    if accuracy < COLA_RANDOM_BASELINE:
        print(f"\nCRITICAL: Accuracy ({accuracy*100:.1f}%) is BELOW random baseline ({COLA_RANDOM_BASELINE*100:.1f}%)!")
        print("   Model may be outputting invalid responses or consistently wrong.")
        print("   Aborting experiment.")
        sys.exit(1)
    elif accuracy < COLA_VALID_THRESHOLD:
        print(f"\nWARNING: Accuracy ({accuracy*100:.1f}%) is below valid threshold ({COLA_VALID_THRESHOLD*100:.1f}%).")
        print("   Results may not be meaningful. Proceeding with caution...")
    else:
        print(f"\nPASS: Accuracy ({accuracy*100:.1f}%) exceeds threshold ({COLA_VALID_THRESHOLD*100:.1f}%).")
        print("   Model shows valid signal. Proceeding with sensitivity experiments.")

    # =========================================================================
    # STEP 2: SENSITIVITY EXPERIMENTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SENSITIVITY EXPERIMENTS")
    print("=" * 70)

    SAMPLE_SIZE = 30  # Use 30 samples for faster testing

    all_results = {}

    for style_name, prompt_fn in PROMPT_STYLES.items():
        print(f"\n[{style_name.upper()}] Running experiment...")
        start_time = time.time()

        # Use is_structured=True for Structure prompt to enable JSON parsing
        is_structured = (style_name == "structure")

        result = run_sensitivity_experiment(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            prompt_fn=prompt_fn,
            sample_size=SAMPLE_SIZE,
            seed=SEED,
            device=device,
            is_structured=is_structured
        )

        elapsed = time.time() - start_time
        all_results[style_name] = result

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Average Variation Ratio: {result['avg_variation_ratio']:.4f}")
        print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct_count']}/{result['num_samples']})")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOOTB Accuracy (Control prompt, N={total}): {accuracy*100:.1f}%")
    print(f"\n{'Prompt Style':<15} | {'VR':<8} | {'Accuracy':<12} | {'Interpretation'}")
    print("-" * 70)

    for style_name, result in all_results.items():
        vr = result['avg_variation_ratio']
        acc = result['accuracy']
        if vr < 0.2:
            interp = "Stable"
        elif vr < 0.4:
            interp = "Moderate"
        else:
            interp = "Unstable"
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
    output_file = "outputs\\results\\pythia\\sensitivity_results_pythia.json"
    save_data = {
        "model": model_name,
        "dataset": "CoLA",
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

