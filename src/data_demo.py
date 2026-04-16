#!/usr/bin/env python3
"""
Data Demo Script - Shows exactly what's happening in the sensitivity experiments.

This script demonstrates:
1. Loading real models from HuggingFace (Flan-T5-Base, Pythia-410M)
2. Loading real datasets from HuggingFace (QASC, CoLA)
3. How prompts are constructed and perturbed
4. The actual model responses
5. How answers are parsed and evaluated
6. How the Variation Ratio is calculated

Run with: python data_demo.py [--sensitivity-on-parsed]
"""

import argparse
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from data_analysis import generate_perturbations, ResultAnalyzer

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_DEMO_SAMPLES = 3  # Number of samples to demo for each model
NUM_PERTURBATIONS = 5  # Number of perturbations per sample (smaller for demo)

def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")

# =============================================================================
# PART 1: FLAN-T5-BASE + QASC DEMO
# =============================================================================
def demo_flan_qasc(sensitivity_on_raw: bool = True):
    print_section("PART 1: FLAN-T5-BASE + QASC (Question Answering)")
    print(f"\nVariation ratio mode: {'raw decoded strings (strip only)' if sensitivity_on_raw else 'parsed letters (strip + upper)'}")
    
    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load model from HuggingFace
    print_subsection("Loading Model from HuggingFace")
    model_name = "google/flan-t5-base"
    print(f"Model: {model_name}")
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Model loaded! Parameters: {model.num_parameters():,}")
    
    # Load dataset from HuggingFace
    print_subsection("Loading QASC Dataset from HuggingFace")
    dataset = load_dataset("allenai/qasc", split="validation")
    print(f"Dataset: allenai/qasc")
    print(f"Split: validation")
    print(f"Total samples: {len(dataset)}")
    
    # Demo samples
    analyzer = ResultAnalyzer()
    
    for sample_idx in range(NUM_DEMO_SAMPLES):
        item = dataset[sample_idx]
        print_subsection(f"Sample {sample_idx + 1}/{NUM_DEMO_SAMPLES}")
        
        # Show the raw data
        print(f"\n[RAW DATA FROM HUGGINGFACE]")
        print(f"  Question: {item['question']}")
        print(f"  Fact 1: {item['fact1']}")
        print(f"  Fact 2: {item['fact2']}")
        print(f"  Choices:")
        choices = item['choices']
        for i, (label, text) in enumerate(zip(choices['label'], choices['text'])):
            marker = " <-- CORRECT" if label == item['answerKey'] else ""
            print(f"    {label}) {text}{marker}")
        print(f"  Correct Answer: {item['answerKey']}")
        
        # Create prompt with facts
        choices_text = "\n".join([f"{l}) {t}" for l, t in zip(choices['label'], choices['text'])])
        base_prompt = f"""Given these facts:
- {item['fact1']}
- {item['fact2']}

Question: {item['question']}

{choices_text}

Answer with just the letter (A-H):"""
        
        print(f"\n[CONTROL PROMPT]")
        print(base_prompt)
        
        # Generate perturbations
        perturbations = generate_perturbations(item['question'], NUM_PERTURBATIONS)
        print(f"\n[PERTURBATIONS OF THE QUESTION]")
        print(f"  Original: {item['question']}")
        for i, p in enumerate(perturbations, 1):
            print(f"  Perturb {i}: {p}")
        
        # Run inference on original + perturbations
        all_questions = [item['question']] + perturbations
        all_prompts = []
        for q in all_questions:
            prompt = f"""Given these facts:
- {item['fact1']}
- {item['fact2']}

Question: {q}

{choices_text}

Answer with just the letter (A-H):"""
            all_prompts.append(prompt)
        
        print(f"\n[MODEL INFERENCE]")
        answers = []
        raw_responses = []
        for i, prompt in enumerate(all_prompts):
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            raw_responses.append(response)
            parsed = analyzer.parse_letter_answer(response)
            answers.append(parsed)
            
            label = "Original" if i == 0 else f"Perturb {i}"
            correct = "CORRECT" if parsed == item['answerKey'] else "WRONG"
            print(f"  {label}: Raw='{response}' -> Parsed='{parsed}' [{correct}]")
        
        # Calculate variation ratio AND accuracy (VR uses raw or parsed per flag; accuracy uses parsed only)
        print(f"\n[SENSITIVITY & ACCURACY ANALYSIS]")
        valid_parsed = [a for a in answers if a]
        if sensitivity_on_raw:
            vr_inputs = [r.strip() for r in raw_responses if r and r.strip()]
            vr_norm = "raw"
        else:
            vr_inputs = valid_parsed
            vr_norm = "parsed"
        vr = analyzer.calculate_variation_ratio(vr_inputs, normalization=vr_norm)
        counts = Counter(vr_inputs)
        modal = counts.most_common(1)[0] if counts else ("", 0)

        # Accuracy: is the original (non-perturbed) answer correct?
        original_answer = answers[0]
        correct_answer = item['answerKey']
        is_correct = original_answer == correct_answer

        print(f"  All parsed answers: {answers}")
        if sensitivity_on_raw:
            print(f"  VR inputs (raw, stripped): {vr_inputs}")
        print(f"  Correct answer: {correct_answer}")
        print(f"  Original answer: {original_answer} -> {'CORRECT' if is_correct else 'WRONG'}")
        print(f"  Modal answer: '{modal[0]}' (appears {modal[1]} times)")
        print(f"\n  [Variation Ratio]")
        print(f"  Formula: VR = 1 - (f_modal / N_total)")
        print(f"  VR = 1 - ({modal[1]} / {len(answers)}) = {vr:.4f}")
        print(f"  Interpretation: {'Stable' if vr < 0.2 else 'Moderate' if vr < 0.4 else 'Unstable'}")
        print(f"\n  [Key Insight for Research]")
        print(f"  This sample: VR={vr:.2f}, Correct={is_correct}")
        print(f"  Hypothesis: Lower VR (more stable) should correlate with higher accuracy")

    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# =============================================================================
# PART 2: PYTHIA-410M + CoLA DEMO
# =============================================================================
def demo_pythia_cola(sensitivity_on_raw: bool = True):
    print_section("PART 2: PYTHIA-410M + CoLA (Grammaticality Judgment)")
    print(f"\nVariation ratio mode: {'raw decoded strings (strip only)' if sensitivity_on_raw else 'parsed Yes/No (strip + upper)'}")

    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model from HuggingFace
    print_subsection("Loading Model from HuggingFace")
    model_name = "EleutherAI/pythia-410m"
    print(f"Model: {model_name}")
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Model loaded! Parameters: {model.num_parameters():,}")

    # Load dataset from HuggingFace
    print_subsection("Loading CoLA Dataset from HuggingFace")
    cola_dataset = load_dataset("nyu-mll/glue", "cola", split="validation")
    print(f"Dataset: nyu-mll/glue (cola)")
    print(f"Split: validation")
    print(f"Total samples: {len(cola_dataset)}")

    # Demo samples
    analyzer = ResultAnalyzer()

    for sample_idx in range(NUM_DEMO_SAMPLES):
        item = cola_dataset[sample_idx]
        print_subsection(f"Sample {sample_idx + 1}/{NUM_DEMO_SAMPLES}")

        # Show the raw data
        print(f"\n[RAW DATA FROM HUGGINGFACE]")
        print(f"  Sentence: {item['sentence']}")
        print(f"  Label: {item['label']} ({'Grammatical' if item['label'] == 1 else 'Ungrammatical'})")

        # Create prompt
        base_prompt = f"""Is this sentence grammatically correct? Answer Yes or No.

Sentence: "{item['sentence']}"

Answer:"""

        print(f"\n[CONTROL PROMPT]")
        print(base_prompt)

        # Generate perturbations
        perturbations = generate_perturbations(item['sentence'], NUM_PERTURBATIONS)
        print(f"\n[PERTURBATIONS OF THE SENTENCE]")
        print(f"  Original: {item['sentence']}")
        for i, p in enumerate(perturbations, 1):
            print(f"  Perturb {i}: {p}")

        # Run inference on original + perturbations
        all_sentences = [item['sentence']] + perturbations

        print(f"\n[MODEL INFERENCE]")
        answers = []
        raw_responses = []
        for i, sent in enumerate(all_sentences):
            prompt = f"""Is this sentence grammatically correct? Answer Yes or No.

Sentence: "{sent}"

Answer:"""
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            # Get only the new tokens
            response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            raw_responses.append(response)
            parsed = analyzer.parse_yes_no_answer(response)
            answers.append(parsed)

            # Map to expected label
            expected = "YES" if item['label'] == 1 else "NO"
            correct = "CORRECT" if parsed == expected else "WRONG"

            label = "Original" if i == 0 else f"Perturb {i}"
            print(f"  {label}: Raw='{response.strip()[:30]}' -> Parsed='{parsed}' [{correct}]")

        # Calculate variation ratio AND accuracy (VR uses raw or parsed per flag; accuracy uses parsed only)
        print(f"\n[SENSITIVITY & ACCURACY ANALYSIS]")
        valid_parsed = [a for a in answers if a]
        if sensitivity_on_raw:
            vr_inputs = [r.strip() for r in raw_responses if r and r.strip()]
            vr_norm = "raw"
        else:
            vr_inputs = valid_parsed
            vr_norm = "parsed"
        vr = analyzer.calculate_variation_ratio(vr_inputs, normalization=vr_norm)
        counts = Counter(vr_inputs)
        modal = counts.most_common(1)[0] if counts else ("", 0)

        # Accuracy: is the original (non-perturbed) answer correct?
        original_answer = answers[0]
        correct_answer = "YES" if item['label'] == 1 else "NO"
        is_correct = original_answer == correct_answer

        print(f"  All parsed answers: {answers}")
        if sensitivity_on_raw:
            print(f"  VR inputs (raw, stripped): {vr_inputs}")
        print(f"  Correct answer: {correct_answer} (label={item['label']})")
        print(f"  Original answer: {original_answer} -> {'CORRECT' if is_correct else 'WRONG'}")
        print(f"  Modal answer: '{modal[0]}' (appears {modal[1]} times)")
        print(f"\n  [Variation Ratio]")
        print(f"  Formula: VR = 1 - (f_modal / N_total)")
        print(f"  VR = 1 - ({modal[1]} / {len(answers)}) = {vr:.4f}")
        print(f"  Interpretation: {'Stable' if vr < 0.2 else 'Moderate' if vr < 0.4 else 'Unstable'}")
        print(f"\n  [Key Insight for Research]")
        print(f"  This sample: VR={vr:.2f}, Correct={is_correct}")
        print(f"  Hypothesis: Lower VR (more stable) should correlate with higher accuracy")

    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data demo for sensitivity experiments")
    parser.add_argument(
        "--sensitivity-on-parsed",
        action="store_true",
        help="Use extracted answers for VR (strip+upper). Default matches run_experiment: raw decoded strings (strip only)",
    )
    args = parser.parse_args()
    sensitivity_on_raw = not args.sensitivity_on_parsed

    print_section("DATA DEMO - SENSITIVITY EXPERIMENTS")
    print(f"VR mode: {'raw (strip only)' if sensitivity_on_raw else 'parsed (strip + upper)'}")
    print("""
This script demonstrates that we are using:
- REAL models from HuggingFace (google/flan-t5-base, EleutherAI/pythia-410m)
- REAL datasets from HuggingFace (allenai/qasc, nyu-mll/glue cola)
- REAL inference with greedy decoding

You will see:
1. The raw data loaded from HuggingFace
2. How prompts are constructed
3. How perturbations are generated
4. The actual model responses
5. How answers are parsed and evaluated
6. How Variation Ratio is calculated
""")

    # Run both demos
    demo_flan_qasc(sensitivity_on_raw=sensitivity_on_raw)
    demo_pythia_cola(sensitivity_on_raw=sensitivity_on_raw)

    print_section("DEMO COMPLETE")
    print("""
Summary:
- Both models were loaded from HuggingFace Hub
- Both datasets were loaded from HuggingFace Datasets
- We showed the raw data, prompts, model outputs, and calculations
- The Variation Ratio measures output stability across perturbations:
  - VR = 0.0 means perfectly stable (same answer for all perturbations)
  - VR = 1.0 means completely unstable (different answer each time)
""")

