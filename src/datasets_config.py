#!/usr/bin/env python3
"""
Dataset configurations for sensitivity experiments.

Supports:
- QASC: 8-way multiple choice science QA
- CoLA: Binary grammaticality judgment
"""

from typing import Dict
from dataclasses import dataclass
from enum import Enum


class AnswerType(Enum):
    """Type of answer expected from the model."""
    LETTER = "letter"    # A-H for multiple choice
    YES_NO = "yes_no"    # Yes/No for binary classification


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str                    # Display name
    hf_name: str                 # HuggingFace dataset identifier
    hf_subset: str = None        # Optional subset name
    split: str = "validation"    # Default split to use
    answer_type: AnswerType = AnswerType.LETTER
    random_baseline: float = 0.5 # Random guessing accuracy
    valid_threshold: float = 0.6 # Threshold for valid signal
    valid_answers: str = "ABCDEFGH"  # Valid answer characters


# Supported dataset configurations
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "qasc": DatasetConfig(
        name="QASC",
        hf_name="allenai/qasc",
        answer_type=AnswerType.LETTER,
        random_baseline=0.125,    # 1/8 for 8 choices
        valid_threshold=0.40,
        valid_answers="ABCDEFGH",
    ),
    "cola": DatasetConfig(
        name="CoLA",
        hf_name="nyu-mll/glue",
        hf_subset="cola",
        answer_type=AnswerType.YES_NO,
        random_baseline=0.50,     # 1/2 for binary
        valid_threshold=0.60,
        valid_answers="",         # Not applicable for Yes/No
    ),
}



