"""
NLP Final Project: Stability by Design in SLMs
Team Member 2: Data Management, Out-Of-The-Box Evaluation, JSON Parsing, and Metric Calculation

This module implements DataManager and ResultAnalyzer classes for handling QASC and CoLA datasets,
performing OOTB sanity checks, parsing LLM outputs, and calculating sensitivity metrics.

Key Features:
- QASC: 8-choice multiple choice with fact injection
- CoLA: Binary grammaticality judgment (Yes/No)
- On-the-fly perturbation generation for both datasets
- Robust JSON parsing for structured prompts
- Variation Ratio calculation for sensitivity analysis
"""

import re
import json
import random
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import Counter

# =====================================================================
# PERTURBATION GENERATION (delegated to perturbations.py)
# =====================================================================
# Team Member 3's WordNet-based perturbation module is the canonical
# implementation.  We re-export its main entry point so that existing
# code that imports from data_analysis keeps working.

from perturbations import generate_and_validate as generate_perturbations  # noqa: F401

# Configure logging for OOTB baseline evaluation
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataManager:
    """
    Assigned to: Team Member 2
    Description: Manages the QASC and CoLA datasets, including loading, formatting,
                 and running Out-Of-The-Box (OOTB) baseline evaluation.
    """

    # QASC has 8 multiple-choice options, so random baseline is 12.5%
    QASC_RANDOM_BASELINE = 0.125
    QASC_VALID_THRESHOLD = 0.40  # 40% accuracy indicates valid signal
    QASC_WARNING_THRESHOLD = 0.15  # 15% accuracy is close to random guessing

    # CoLA is binary classification, so random baseline is 50%
    COLA_RANDOM_BASELINE = 0.50
    COLA_VALID_THRESHOLD = 0.60  # 60% accuracy indicates valid signal
    COLA_WARNING_THRESHOLD = 0.52  # 52% is barely above random

    # Keep legacy names for backwards compatibility
    RANDOM_BASELINE_ACCURACY = QASC_RANDOM_BASELINE
    VALID_SIGNAL_THRESHOLD = QASC_VALID_THRESHOLD
    WARNING_THRESHOLD = QASC_WARNING_THRESHOLD

    def __init__(self):
        """Initialize the DataManager."""
        self._dataset_cache: Dict[str, List[Dict]] = {}

    def load_qasc_dataset(self, split: str = "validation") -> List[Dict]:
        """
        Loads the QASC dataset from HuggingFace.

        Args:
            split: The dataset split to load ('train', 'validation', or 'test').

        Returns:
            A list of dictionaries, each containing a QASC question with facts and choices.

        Raises:
            ImportError: If the datasets library is not installed.
        """
        # Check cache first
        if split in self._dataset_cache:
            return self._dataset_cache[split]

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. Install with: pip install datasets"
            )

        # Load QASC from HuggingFace
        dataset = load_dataset("allenai/qasc", split=split)

        # Convert to list of dicts for easier manipulation
        data_list = []
        for item in dataset:
            data_list.append({
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "fact1": item.get("fact1", ""),
                "fact2": item.get("fact2", ""),
                "choices": item.get("choices", {}).get("text", []),
                "choice_labels": item.get("choices", {}).get("label", []),
                "answer_key": item.get("answerKey", ""),
                "combined_fact": item.get("combinedfact", ""),
            })

        # Cache the result
        self._dataset_cache[split] = data_list
        return data_list

    def format_question_with_facts(self, question: str, fact1: str, fact2: str) -> str:
        """
        Combines Fact 1, Fact 2, and the question into a single cohesive string.

        The QASC dataset requires combining two facts to answer the question correctly.
        This method injects these facts into the prompt to provide necessary context.

        Args:
            question: The original QASC question.
            fact1: The first supporting fact.
            fact2: The second supporting fact.

        Returns:
            A formatted string containing both facts and the question, ready for LLM input.
        """
        formatted_prompt = (
            f"Given the following facts:\n"
            f"Fact 1: {fact1}\n"
            f"Fact 2: {fact2}\n\n"
            f"Question: {question}\n\n"
            f"Please provide your answer in JSON format with the following structure:\n"
            f'{{"reasoning": "your step-by-step reasoning", "final_answer": "A/B/C/D/E/F/G/H"}}'
        )
        return formatted_prompt

    def format_qasc_item(self, item: Dict) -> str:
        """
        Convenience method to format a complete QASC item dictionary.

        Args:
            item: A dictionary from load_qasc_dataset containing question, fact1, fact2, and choices.

        Returns:
            A formatted prompt string including facts, question, and multiple choice options.
        """
        # Build the choices string
        choices_str = ""
        if item.get("choices") and item.get("choice_labels"):
            for label, text in zip(item["choice_labels"], item["choices"]):
                choices_str += f"  {label}. {text}\n"

        # Format with facts
        formatted_prompt = (
            f"Given the following facts:\n"
            f"Fact 1: {item.get('fact1', '')}\n"
            f"Fact 2: {item.get('fact2', '')}\n\n"
            f"Question: {item.get('question', '')}\n"
            f"Choices:\n{choices_str}\n"
            f"Please provide your answer in JSON format:\n"
            f'{{"reasoning": "your step-by-step reasoning", "final_answer": "A/B/C/D/E/F/G/H"}}'
        )
        return formatted_prompt

    def run_ootb_baseline(
        self,
        inference_fn: Callable[[List[str]], List[str]],
        dataset: Optional[List[Dict]] = None,
        sample_size: int = 100,
    ) -> float:
        """
        Evaluates the model's Out-Of-The-Box performance to ensure it performs
        significantly better than random guessing (12.5% for 8-choice QASC).

        Args:
            inference_fn: A callable that takes a list of prompts and returns model outputs.
                         This should be ModelManager.generate_responses or similar.
            dataset: Optional pre-loaded dataset. If None, loads validation split.
            sample_size: Number of questions to sample (100-200 recommended).

        Returns:
            The accuracy as a float between 0.0 and 1.0.
        """
        # Load dataset if not provided
        if dataset is None:
            dataset = self.load_qasc_dataset(split="validation")

        # Sample the dataset
        import random
        sample_size = min(sample_size, len(dataset))
        sample = random.sample(dataset, sample_size)

        # Format all prompts
        prompts = [self.format_qasc_item(item) for item in sample]

        # Run inference
        logger.info(f"Running OOTB baseline evaluation on {sample_size} samples...")
        raw_outputs = inference_fn(prompts)

        # Parse outputs and calculate accuracy
        result_analyzer = ResultAnalyzer()
        correct = 0
        total = 0

        for item, raw_output in zip(sample, raw_outputs):
            parsed = result_analyzer.extract_json_from_output(raw_output)
            predicted_answer = parsed.get("final_answer", "").strip().upper()
            ground_truth = item.get("answer_key", "").strip().upper()

            if predicted_answer and ground_truth:
                total += 1
                if predicted_answer == ground_truth:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        # Log results with appropriate warnings/messages
        self._log_ootb_results(accuracy, total)

        return accuracy

    def _log_ootb_results(self, accuracy: float, total_evaluated: int) -> None:
        """
        Logs the OOTB baseline results with appropriate warnings or success messages.

        Args:
            accuracy: The calculated accuracy.
            total_evaluated: Number of samples that were successfully evaluated.
        """
        logger.info(f"OOTB Baseline Results: {accuracy:.1%} accuracy ({total_evaluated} samples)")

        if accuracy <= self.WARNING_THRESHOLD:
            logger.warning(
                "MODEL IS RANDOMLY GUESSING!\n"
                f"   Accuracy ({accuracy:.1%}) is at or near random baseline ({self.RANDOM_BASELINE_ACCURACY:.1%}).\n"
                "   RECOMMENDATION: Improve the prompt strategy:\n"
                "   - Consider switching to Few-Shot prompting\n"
                "   - Try adding explicit instructions\n"
                "   - Verify the prompt format matches model expectations"
            )
        elif accuracy < self.VALID_SIGNAL_THRESHOLD:
            logger.info(
                f"Moderate signal detected ({accuracy:.1%}).\n"
                "   Model is performing above random, but below optimal threshold.\n"
                "   Consider prompt improvements for better results."
            )
        else:
            logger.info(
                f"VALID SIGNAL DETECTED!\n"
                f"   Accuracy ({accuracy:.1%}) is above {self.VALID_SIGNAL_THRESHOLD:.0%} threshold.\n"
                "   Proceeding with main experiment is recommended."
            )

    # =========================================================================
    # CoLA Dataset Methods
    # =========================================================================

    def load_cola_dataset(self, split: str = "validation") -> List[Dict]:
        """
        Loads the CoLA dataset from HuggingFace (via GLUE).

        CoLA (Corpus of Linguistic Acceptability) is a binary classification task
        where sentences are labeled as grammatically acceptable (1) or not (0).

        Args:
            split: The dataset split to load ('train', 'validation', or 'test').

        Returns:
            A list of dictionaries, each containing a CoLA sentence and label.
        """
        cache_key = f"cola_{split}"
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. Install with: pip install datasets"
            )

        # Load CoLA from GLUE benchmark
        dataset = load_dataset("nyu-mll/glue", "cola", split=split)

        # Convert to list of dicts
        data_list = []
        for item in dataset:
            data_list.append({
                "idx": item.get("idx", 0),
                "sentence": item.get("sentence", ""),
                "label": item.get("label", 0),  # 0=unacceptable, 1=acceptable
            })

        self._dataset_cache[cache_key] = data_list
        return data_list

    def format_cola_item(self, item: Dict, few_shot: bool = False) -> str:
        """
        Formats a CoLA item into a prompt for the model.

        Args:
            item: A dictionary from load_cola_dataset containing sentence and label.
            few_shot: If True, includes few-shot examples in the prompt.

        Returns:
            A formatted prompt string asking the model to judge grammaticality.
        """
        sentence = item.get("sentence", "")

        if few_shot:
            prompt = f'''Is each sentence grammatically correct? Answer Yes or No.

Sentence: "The cat sat on the mat."
Answer: Yes

Sentence: "Him went to store."
Answer: No

Sentence: "She quickly finished her homework."
Answer: Yes

Sentence: "The books is on the table."
Answer: No

Sentence: "{sentence}"
Answer:'''
        else:
            prompt = f'''Is this sentence grammatically correct? Answer Yes or No.

Sentence: "{sentence}"

Answer:'''

        return prompt

    def run_ootb_baseline_cola(
        self,
        inference_fn: Callable[[List[str]], List[str]],
        dataset: Optional[List[Dict]] = None,
        sample_size: int = 100,
        few_shot: bool = False,
    ) -> float:
        """
        Evaluates the model's Out-Of-The-Box performance on CoLA.
        CoLA is binary classification, so random baseline is 50%.

        Args:
            inference_fn: A callable that takes a list of prompts and returns model outputs.
            dataset: Optional pre-loaded dataset. If None, loads validation split.
            sample_size: Number of sentences to sample.
            few_shot: If True, uses few-shot prompting.

        Returns:
            The accuracy as a float between 0.0 and 1.0.
        """
        if dataset is None:
            dataset = self.load_cola_dataset(split="validation")

        import random
        sample_size = min(sample_size, len(dataset))
        sample = random.sample(dataset, sample_size)

        # Format prompts
        prompts = [self.format_cola_item(item, few_shot=few_shot) for item in sample]

        logger.info(f"Running CoLA OOTB evaluation on {sample_size} samples...")
        raw_outputs = inference_fn(prompts)

        # Parse Yes/No responses
        correct = 0
        total = 0

        for item, raw_output in zip(sample, raw_outputs):
            response = raw_output.strip().lower()

            # Extract Yes/No from the first 10 characters
            predicted = None
            if "yes" in response[:10]:
                predicted = 1
            elif "no" in response[:10]:
                predicted = 0

            if predicted is not None:
                total += 1
                if predicted == item["label"]:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        self._log_cola_results(accuracy, total)
        return accuracy

    def _log_cola_results(self, accuracy: float, total_evaluated: int) -> None:
        """Logs the CoLA OOTB baseline results."""
        logger.info(f"CoLA OOTB Results: {accuracy:.1%} accuracy ({total_evaluated} samples)")

        if accuracy <= self.COLA_WARNING_THRESHOLD:
            logger.warning(
                "MODEL IS AT RANDOM LEVEL!\n"
                f"   Accuracy ({accuracy:.1%}) is at or near random baseline ({self.COLA_RANDOM_BASELINE:.1%}).\n"
                "   RECOMMENDATION: Try few-shot prompting or different prompt format."
            )
        elif accuracy < self.COLA_VALID_THRESHOLD:
            logger.info(
                f"Marginal signal detected ({accuracy:.1%}).\n"
                f"   Above random ({self.COLA_RANDOM_BASELINE:.1%}) but below threshold ({self.COLA_VALID_THRESHOLD:.1%})."
            )
        else:
            logger.info(
                f"VALID SIGNAL DETECTED!\n"
                f"   Accuracy ({accuracy:.1%}) is above {self.COLA_VALID_THRESHOLD:.0%} threshold.\n"
                "   Proceeding with CoLA sensitivity experiments is recommended."
            )


class ResultAnalyzer:
    """
    Assigned to: Team Member 2
    Description: Parses the raw text into structured data and calculates sensitivity metrics.

    This class handles:
    1. Robust JSON extraction from noisy LLM outputs
    2. Variation ratio calculation for sensitivity analysis
    """

    # Fallback dictionary when JSON parsing fails completely
    FALLBACK_RESPONSE: Dict[str, Any] = {
        "reasoning": "PARSE_ERROR: Could not extract valid JSON from model output",
        "final_answer": "",
        "parse_success": False
    }

    def __init__(self):
        """Initialize the ResultAnalyzer with compiled regex patterns for efficiency."""
        # Pattern 1: Match complete JSON object with both required fields
        self._json_pattern = re.compile(
            r'\{[^{}]*"reasoning"\s*:\s*"[^"]*"[^{}]*"final_answer"\s*:\s*"[^"]*"[^{}]*\}|'
            r'\{[^{}]*"final_answer"\s*:\s*"[^"]*"[^{}]*"reasoning"\s*:\s*"[^"]*"[^{}]*\}',
            re.DOTALL | re.IGNORECASE
        )

        # Pattern 2: Match any JSON-like object with curly braces
        self._loose_json_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL)

        # Pattern 3: Extract final_answer directly even without proper JSON
        self._answer_pattern = re.compile(
            r'(?:"final_answer"|final_answer)\s*[:\s]+\s*["\']?([A-H])["\']?',
            re.IGNORECASE
        )

        # Pattern 4: Extract reasoning directly
        self._reasoning_pattern = re.compile(
            r'(?:"reasoning"|reasoning)\s*[:\s]+\s*["\']([^"\']+)["\']',
            re.IGNORECASE
        )

    def extract_json_from_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Uses Regex to find and extract the JSON object from the model's raw output.

        Handles edge cases where the model adds extra text, missing brackets,
        or produces malformed JSON. Uses multiple fallback strategies.

        Args:
            raw_text: The raw string output from the LLM.

        Returns:
            A dictionary containing:
            - 'reasoning': The model's reasoning (str)
            - 'final_answer': The extracted answer (str, typically A-H)
            - 'parse_success': Whether parsing was successful (bool)
        """
        if not raw_text or not isinstance(raw_text, str):
            return self.FALLBACK_RESPONSE.copy()

        # Strategy 1: Try to find complete JSON object with both fields
        match = self._json_pattern.search(raw_text)
        if match:
            try:
                parsed = json.loads(match.group())
                # Normalize keys to handle case variations (e.g., "Reasoning" vs "reasoning")
                parsed_lower = {k.lower().replace("_", ""): v for k, v in parsed.items()}
                return {
                    "reasoning": str(parsed_lower.get("reasoning", "")),
                    "final_answer": str(parsed_lower.get("finalanswer", "")).strip().upper(),
                    "parse_success": True
                }
            except json.JSONDecodeError:
                pass

        # Strategy 2: Try to find any JSON-like structure
        match = self._loose_json_pattern.search(raw_text)
        if match:
            try:
                parsed = json.loads(match.group())
                # Normalize keys to lowercase for case-insensitive lookup
                parsed_lower = {k.lower().replace("_", ""): v for k, v in parsed.items()}
                if "finalanswer" in parsed_lower or "reasoning" in parsed_lower:
                    return {
                        "reasoning": str(parsed_lower.get("reasoning", "")),
                        "final_answer": str(parsed_lower.get("finalanswer", "")).strip().upper(),
                        "parse_success": True
                    }
            except json.JSONDecodeError:
                pass

        # Strategy 3: Direct regex extraction of fields (handles malformed JSON)
        answer_match = self._answer_pattern.search(raw_text)
        reasoning_match = self._reasoning_pattern.search(raw_text)

        if answer_match:
            return {
                "reasoning": reasoning_match.group(1) if reasoning_match else "",
                "final_answer": answer_match.group(1).strip().upper(),
                "parse_success": True
            }

        # Strategy 4: Look for standalone letter answer (A-H) as last resort
        standalone_answer = re.search(r'\b([A-H])\b(?:\s*[.)])?', raw_text, re.IGNORECASE)
        if standalone_answer:
            return {
                "reasoning": "",
                "final_answer": standalone_answer.group(1).upper(),
                "parse_success": False  # Partial success
            }

        # All strategies failed
        return self.FALLBACK_RESPONSE.copy()

    def calculate_variation_ratio(self, parsed_answers: List[str]) -> float:
        """
        Calculates the sensitivity metric 's' using the Variation Ratio formula.

        Formula: s = 1 - (f_m / (N + 1))
        Where:
        - f_m is the frequency of the most common (modal) final answer
        - N is the number of perturbed variants
        - The +1 accounts for the original unperturbed prompt

        A higher variation ratio indicates greater sensitivity (less stability).
        - s = 0 means the model always gives the same answer (perfectly stable)
        - s approaching 1 means high variability in answers (unstable)

        IMPORTANT: This calculation uses extracted final_answer values only,
        not raw text strings, to avoid penalizing slight phrasing differences.

        Args:
            parsed_answers: A list of final_answer strings extracted from parsed JSON.
                           These should be normalized (e.g., uppercase single letters).

        Returns:
            The variation ratio 's' as a float between 0.0 and 1.0.
            Returns 0.0 if the input list is empty or has only one element.
        """
        if not parsed_answers:
            return 0.0

        # Filter out empty answers and normalize
        valid_answers = [ans.strip().upper() for ans in parsed_answers if ans and ans.strip()]

        if len(valid_answers) <= 1:
            return 0.0

        # Calculate frequency of each answer
        answer_counts = Counter(valid_answers)

        # Get the frequency of the modal (most common) answer
        f_m = answer_counts.most_common(1)[0][1]

        # N is the number of perturbed variants (excluding original)
        # Total answers = original + N perturbations = N + 1
        # So N = len(valid_answers) - 1
        N = len(valid_answers) - 1

        # Calculate variation ratio: s = 1 - (f_m / (N + 1))
        variation_ratio = 1 - (f_m / (N + 1))

        return variation_ratio

    def batch_extract_and_calculate(
        self,
        raw_outputs: List[str]
    ) -> Dict[str, Any]:
        """
        Convenience method to extract answers from multiple outputs and calculate variation ratio.

        Args:
            raw_outputs: List of raw LLM outputs from perturbed prompts.

        Returns:
            A dictionary containing:
            - 'parsed_results': List of parsed dictionaries
            - 'final_answers': List of extracted final answers
            - 'variation_ratio': The calculated sensitivity metric
            - 'parse_success_rate': Percentage of successfully parsed outputs
        """
        parsed_results = [self.extract_json_from_output(output) for output in raw_outputs]
        final_answers = [p["final_answer"] for p in parsed_results]

        successful_parses = sum(1 for p in parsed_results if p.get("parse_success", False))
        parse_success_rate = successful_parses / len(parsed_results) if parsed_results else 0.0

        variation_ratio = self.calculate_variation_ratio(final_answers)

        return {
            "parsed_results": parsed_results,
            "final_answers": final_answers,
            "variation_ratio": variation_ratio,
            "parse_success_rate": parse_success_rate
        }

    def parse_yes_no_answer(self, response: str, is_structured: bool = False) -> str:
        """
        Parse Yes/No answer from model response (for CoLA and similar binary tasks).

        Args:
            response: Raw model output
            is_structured: If True, attempts JSON parsing first for Structure prompt

        Returns:
            "YES", "NO", or empty string if parsing fails
        """
        response = response.strip()

        # For Structure prompt: Try JSON parsing first
        if is_structured:
            answer = self._parse_yes_no_json(response)
            if answer:
                return answer

        # Fallback: Standard Yes/No extraction
        response_lower = response.lower()[:30]
        if "yes" in response_lower:
            return "YES"
        elif "no" in response_lower:
            return "NO"
        return ""

    def _parse_yes_no_json(self, response: str) -> str:
        """
        Parse final_answer from JSON response for Yes/No tasks.

        Args:
            response: Raw model output (expected to be JSON)

        Returns:
            "YES", "NO", or empty string if parsing fails
        """
        try:
            # Try direct JSON parsing
            data = json.loads(response)
            if isinstance(data, dict) and "final_answer" in data:
                answer = str(data["final_answer"]).strip().lower()
                if "yes" in answer:
                    return "YES"
                elif "no" in answer:
                    return "NO"
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response (model might add extra text)
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "final_answer" in data:
                    answer = str(data["final_answer"]).strip().lower()
                    if "yes" in answer:
                        return "YES"
                    elif "no" in answer:
                        return "NO"
            except json.JSONDecodeError:
                pass

        # Try regex extraction as last resort
        match = re.search(r'"final_answer"\s*:\s*"?(yes|no)"?', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return ""

    def parse_letter_answer(self, response: str, is_structured: bool = False, valid_letters: str = "ABCDEFGH") -> str:
        """
        Parse single letter answer from model response (for QASC and similar multiple choice tasks).

        Args:
            response: Raw model output
            is_structured: If True, attempts JSON parsing first for Structure prompt
            valid_letters: String of valid answer letters (default "ABCDEFGH" for QASC)

        Returns:
            Single uppercase letter or empty string if parsing fails
        """
        response = response.strip()

        # For Structure prompt: Try JSON parsing first
        if is_structured:
            parsed = self.extract_json_from_output(response)
            if parsed.get("parse_success") and parsed.get("final_answer"):
                answer = parsed["final_answer"].upper()
                if answer and answer[0] in valid_letters:
                    return answer[0]

        # Fallback: Standard letter extraction
        response_upper = response.upper()
        pattern = r'\b([' + valid_letters + r'])\b'
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
        # Check first character
        if response_upper and response_upper[0] in valid_letters:
            return response_upper[0]
        return ""


if __name__ == "__main__":
    print("data_analysis.py provides DataManager and ResultAnalyzer.")
    print("For a full demo run: python data_demo.py")
    print("For experiments run:  python run_experiment.py --model flan-t5-base --dataset qasc")
