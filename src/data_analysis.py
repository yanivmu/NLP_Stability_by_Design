"""
NLP Final Project: Stability by Design in SLMs
Team Member 2: Metric Calculation

This module implements the ResultAnalyzer class for calculating sensitivity metrics
(Variation Ratio) used by the experiment pipeline.
"""

from typing import List, Literal
from collections import Counter


class ResultAnalyzer:
    """
    Parses the raw text into structured data and calculates sensitivity metrics.

    The main pipeline uses ``calculate_variation_ratio`` to measure how much
    a model's output changes across perturbations of the same input.
    """

    def calculate_variation_ratio(
        self,
        parsed_answers: List[str],
        normalization: Literal["parsed", "raw"] = "raw",
    ) -> float:
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

        Args:
            parsed_answers: Answer strings to compare (extracted letters when
                ``normalization="parsed"``, or full decoded model outputs when
                ``normalization="raw"``).
            normalization:
                - ``"raw"`` (default): filter empties after ``strip()`` only (no case folding),
                  then modal count—full decoded strings are the outcome.
                - ``"parsed"``: filter empties, apply ``strip().upper()``,
                  then modal count. Use with extracted final answers so minor
                  phrasing differences are not penalized.

        Returns:
            The variation ratio 's' as a float between 0.0 and 1.0.
            Returns 0.0 if the input list is empty or has only one element.
        """
        if not parsed_answers:
            return 0.0

        if normalization == "raw":
            valid_answers = [ans.strip() for ans in parsed_answers if ans and ans.strip()]
        else:
            valid_answers = [ans.strip().upper() for ans in parsed_answers if ans and ans.strip()]

        if len(valid_answers) <= 1:
            return 0.0

        answer_counts = Counter(valid_answers)
        f_m = answer_counts.most_common(1)[0][1]
        N = len(valid_answers) - 1
        variation_ratio = 1 - (f_m / (N + 1))

        return variation_ratio


if __name__ == "__main__":
    print("data_analysis.py provides ResultAnalyzer.")
    print("For experiments run:  python run_experiment.py --model flan-t5-base --dataset qasc")
