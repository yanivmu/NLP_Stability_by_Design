"""
NLP Final Project: Stability by Design in SLMs
Team Members: 3
Target Completion: March 15th
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

# =====================================================================
# TEAM MEMBER 1: DevOps, Infrastructure & Models
# Responsibilities: Model loading, inference, Slurm optimization, reproducibility.
# =====================================================================

class EnvironmentManager:
    """
    Assigned to: Team Member 1
    Description: Handles the technical setup, including fixing the random seeds 
                 for reproducibility across the entire pipeline.
    """
    def set_seed(self, seed: int) -> None:
        """Fixes the seed for torch, numpy, and random libraries."""
        pass

class ModelManager:
    """
    Assigned to: Team Member 1
    [cite_start]Description: Responsible for loading the SLMs (e.g., Flan-T5-Base, Pythia-410M) [cite: 22] 
                 and generating text safely without Out-Of-Memory (OOM) errors.
    """
    def __init__(self, model_name: str):
        """Loads the model and tokenizer onto the appropriate device (GPU/CPU)."""
        pass

    def generate_responses(self, prompts: List[str], batch_size: int, max_length: int) -> List[str]:
        """
        Takes a list of prompts and returns raw model outputs. 
        Must handle batching and context length limits to prevent OOM on Slurm.
        """
        pass

# =====================================================================
# TEAM MEMBER 2: Data, Parsing & Analysis
# Responsibilities: QASC formatting, Out-Of-The-Box (OOTB) testing, JSON parsing, Math.
# =====================================================================

class DataManager:
    """
    Assigned to: Team Member 2
    Description: Manages the QASC dataset, including the injection of facts.
    """
    def load_qasc_dataset(self, split: str = "validation") -> List[Dict]:
        """Loads the QASC dataset from HuggingFace or local storage."""
        pass

    def format_question_with_facts(self, question: str, fact1: str, fact2: str) -> str:
        """
        Combines Fact 1, Fact 2, and the question into a single cohesive string.
        """
        pass

    def run_ootb_baseline(self, model_manager: ModelManager, dataset: List[Dict]) -> float:
        """
        Evaluates the model's Out-Of-The-Box performance to ensure it performs
        significantly better than random guessing before proceeding with the main experiment.
        """
        pass

class ResultAnalyzer:
    """
    Assigned to: Team Member 2
    Description: Parses the raw text into structured data and calculates sensitivity metrics.
    """
    def extract_json_from_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Uses Regex to find and extract the JSON object (e.g., {"reasoning": "...", "final_answer": "A"})
        from the model's raw output. Handles edge cases where the model adds extra text.
        """
        pass

    def calculate_variation_ratio(self, parsed_answers: List[str]) -> float:
        """
        [cite_start]Calculates the sensitivity metric 's' using the formula: s = 1 - (f_m / (N + 1))[cite: 20].
        Calculations are strictly based on the extracted final answers.
        """
        pass

# =====================================================================
# TEAM MEMBER 3: Prompts & Perturbations
# Responsibilities: Prompt engineering and generating input variations.
# =====================================================================

class PromptEngine:
    """
    Assigned to: Team Member 3
    [cite_start]Description: Wraps the base question with different prompt attributes[cite: 14].
    """
    def create_control_prompt(self, base_question: str) -> str:
        [cite_start]"""Creates the standard zero-shot instruction[cite: 15]."""
        pass

    def add_metacognition(self, base_question: str) -> str:
        [cite_start]"""Adds self-check triggers (e.g., 'verify your answer')[cite: 16]."""
        pass

    def add_structure(self, base_question: str) -> str:
        [cite_start]"""Enforces a JSON output schema (e.g., requesting 'reasoning' and 'final_answer')[cite: 17]."""
        pass

    def add_politeness(self, base_question: str) -> str:
        [cite_start]"""Adds conversational fillers (e.g., 'please', 'I would appreciate')[cite: 18]."""
        pass

class PerturbationGenerator:
    """
    Assigned to: Team Member 3
    [cite_start]Description: Generates the N variants for each prompt to test stability[cite: 19].
    """
    def generate_synonym_variants(self, prompt: str, num_variants: int) -> List[str]:
        """
        Takes an engineered prompt and generates semantic variations 
        (e.g., replacing words with synonyms) without changing the core meaning.
        """
        pass

# =====================================================================
# MAIN ORCHESTRATOR (Collaborative)
# =====================================================================

def main():
    """
    The main execution script to be submitted via sbatch.
    1. Team Member 1 initializes the environment and model.
    2. Team Member 2 loads and prepares the QASC data, then runs OOTB.
    3. Team Member 3 generates the prompts and perturbations.
    4. Team Member 1 runs the generation loop.
    5. Team Member 2 parses the outputs and calculates the metrics.
    """
    pass

if __name__ == "__main__":
    main()