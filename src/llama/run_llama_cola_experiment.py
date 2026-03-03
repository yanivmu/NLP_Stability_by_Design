#!/usr/bin/env python3
"""
Sensitivity Experiment: Llama-3.2-1B on CoLA
Generates perturbations on-the-fly and tests prompt properties.

This script:
1. Loads CoLA dataset directly from HuggingFace
2. Generates N=10 semantic-preserving perturbations on-the-fly
3. Runs OOTB accuracy check before sensitivity experiments
4. Uses consolidated functions from data_analysis.py

Prompt properties tested: Control, Metacognition, Structure, Politeness
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
# PROMPT TEMPLATES FOR CoLA (binary grammaticality)
# DIRECT PROMPTS: Ask if the sentence is grammatically correct
# "Yes" -> grammatical (label=1), "No" -> ungrammatical (label=0)
# =====================================================================

def create_control_prompt(sentence: str) -> str:
    """Standard zero-shot instruction (no special properties) - DIRECT."""
    return f"""Is this sentence grammatically correct? Answer Yes or No.

Sentence: "{sentence}"

Answer:"""


def add_metacognition(sentence: str) -> str:
    """Adds self-check triggers (e.g., 'verify your answer') - DIRECT."""
    return f"""Is this sentence grammatically correct? Answer Yes or No.
Before answering, carefully check the grammar rules. Verify your answer is correct.

Sentence: "{sentence}"

Think about it carefully, then answer:"""


def add_structure(sentence: str) -> str:
    """Enforces strict JSON structured output - DIRECT."""
    return f"""Analyze whether the following sentence is grammatically correct.

Sentence: "{sentence}"

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "Yes or No"}}

(Yes = grammatically correct, No = has error)
Output only the JSON, nothing else."""


def add_politeness(sentence: str) -> str:
    """Adds conversational fillers (e.g., 'please', 'I would appreciate') - DIRECT."""
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
    """Load CoLA dataset directly from HuggingFace."""
    dataset = load_dataset("nyu-mll/glue", "cola", split=split)

    data_list = []
    for item in dataset:
        data_list.append({
            "idx": item.get("idx", 0),
            "sentence": item.get("sentence", ""),
            "label": item.get("label", 0),  # 0=unacceptable, 1=acceptable
        })

    return data_list


def run_inference(model, tokenizer, prompts: List[str], device: str, max_new_tokens: int = 20) -> List[str]:
    """Run inference on Llama (decoder-only causal LM)."""
    inputs = tokenizer(prompts, padding=True, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_lengths = [len(ids) for ids in inputs['input_ids']]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract only the new tokens (decoder-only model returns full sequence)
    responses = []
    for i, output in enumerate(outputs):
        new_tokens = output[input_lengths[i]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return responses


def normalize_yes_no(answer: str) -> str:
    """
    Normalize Yes/No answer (no inversion needed for direct prompts).

    Since we ask "Is this grammatically correct?":
    - Model says "Yes" -> sentence is grammatical -> return "YES"
    - Model says "No" -> sentence is ungrammatical -> return "NO"
    """
    if answer == "YES":
        return "YES"
    elif answer == "NO":
        return "YES"
    return answer


def run_ootb_accuracy_check(
    model, tokenizer, dataset: List[Dict], device: str,
    sample_size: int = 100, seed: int = 42
) -> Tuple[float, int, int]:
    """Run Out-Of-The-Box accuracy check using Control prompt."""
    random.seed(seed)
    analyzer = ResultAnalyzer()
    sample_items = random.sample(dataset, min(sample_size, len(dataset)))

    correct = 0
    total = 0
    batch_size = 4  # Smaller batch for decoder-only model

    for i in range(0, len(sample_items), batch_size):
        batch_items = sample_items[i:i+batch_size]
        prompts = [create_control_prompt(item["sentence"]) for item in batch_items]
        responses = run_inference(model, tokenizer, prompts, device)

        for item, response in zip(batch_items, responses):
            raw_predicted = analyzer.parse_yes_no_answer(response)
            predicted = normalize_yes_no(raw_predicted) if raw_predicted else None
            actual = "YES" if item["label"] == 1 else "NO"

            if predicted:
                total += 1
                if predicted == actual:
                    correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_sensitivity_experiment(
    model, tokenizer, dataset: List[Dict], prompt_fn: Callable,
    sample_size: int = 50, seed: int = 42, device: str = "mps",
    is_structured: bool = False
) -> Dict:
    """Run sensitivity experiment for a single prompt style."""
    random.seed(seed)
    analyzer = ResultAnalyzer()
    sample_items = random.sample(dataset, min(sample_size, len(dataset)))

    results = []
    total_variation = 0.0
    max_tokens = 80 if is_structured else 15

    for idx, item in enumerate(sample_items):
        original = item["sentence"]
        label = item["label"]
        perturbations = generate_perturbations(original, NUM_PERTURBATIONS)

        all_sentences = [original] + perturbations
        prompts = [prompt_fn(s) for s in all_sentences]
        responses = run_inference(model, tokenizer, prompts, device, max_new_tokens=max_tokens)

        raw_answers = [analyzer.parse_yes_no_answer(r, is_structured=is_structured) for r in responses]
        answers = [normalize_yes_no(a) if a else None for a in raw_answers]

        valid_answers = [a for a in answers if a]
        if len(valid_answers) >= 2:
            variation_ratio = analyzer.calculate_variation_ratio(valid_answers)
        else:
            variation_ratio = 0.0

        original_answer = answers[0] if answers else None
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
    print("SENSITIVITY EXPERIMENT: Llama-3.2-1B on CoLA")
    print("Testing prompt properties: Control, Metacognition, Structure, Politeness")
    print("=" * 70)
    print("\nFixes applied:")
    print("  - DIRECT PROMPTS: Asks 'Is this grammatically correct?' (Yes=correct, No=error)")
    print("  - Left-padding for decoder-only model")
    print("  - OOTB accuracy check before sensitivity experiments")
    print("  - Strict JSON parsing for Structure prompt")

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading Llama-3.2-1B...")
    model_name = "meta-llama/Llama-3.2-1B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token to eos token (Llama doesn't have pad token by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Use left-padding for decoder-only models (required for correct generation)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        ).to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"\nError loading Llama-3.2-1B: {e}")
        print("\nTo authenticate, run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        sys.exit(1)

    # Load CoLA
    print("\nLoading CoLA dataset...")
    dataset = load_cola_dataset("validation")
    print(f"Loaded {len(dataset)} items")

    # =========================================================================
    # STEP 1: OOTB ACCURACY CHECK
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: OUT-OF-THE-BOX ACCURACY CHECK")
    print("=" * 70)
    print("Testing Control prompt accuracy on 100 samples...")
    print(f"Random baseline: {COLA_RANDOM_BASELINE*100:.1f}%")
    print(f"Valid threshold: {COLA_VALID_THRESHOLD*100:.1f}%")

    SEED = SEEDS[0]
    accuracy, correct, total = run_ootb_accuracy_check(
        model, tokenizer, dataset, device, sample_size=100, seed=SEED
    )

    print(f"\nOOTB Results: {correct}/{total} correct = {accuracy*100:.1f}% accuracy")

    if accuracy < COLA_RANDOM_BASELINE:
        print(f"\nFAIL: Accuracy ({accuracy*100:.1f}%) is BELOW random baseline ({COLA_RANDOM_BASELINE*100:.1f}%).")
        print("   Model is not functional. Aborting experiment.")
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

    SAMPLE_SIZE = 30

    all_results = {}

    for style_name, prompt_fn in PROMPT_STYLES.items():
        print(f"\n[{style_name.upper()}] Running experiment...")
        start_time = time.time()

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

    # Save results
    output_file = "outputs\\results\\llama\\sensitivity_results_llama_cola.json"
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
