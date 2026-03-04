#!/usr/bin/env python3
"""
Dataset configurations for sensitivity experiments.

Supports:
- QASC: 8-way multiple choice science QA
- CoLA: Binary grammaticality judgment
"""

from typing import Dict, List, Any, Callable
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


def get_dataset_config(dataset_key: str) -> DatasetConfig:
    """Get dataset configuration by key."""
    if dataset_key not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {available}")
    return DATASET_CONFIGS[dataset_key]


def load_dataset(config: DatasetConfig):
    """
    Load dataset from HuggingFace.
    
    Returns the dataset object (HuggingFace Dataset).
    """
    from datasets import load_dataset as hf_load_dataset
    
    print(f"Loading {config.name} dataset...")
    
    if config.hf_subset:
        dataset = hf_load_dataset(config.hf_name, config.hf_subset, split=config.split)
    else:
        dataset = hf_load_dataset(config.hf_name, split=config.split)
    
    print(f"Loaded {len(dataset)} items")
    return dataset


def format_qasc_base(item: Dict) -> str:
    """Format QASC item into base question string with facts."""
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


def get_item_text(item: Dict, dataset_key: str) -> str:
    """Get the main text from a dataset item for perturbation."""
    if dataset_key == "qasc":
        return format_qasc_base(item)
    elif dataset_key == "cola":
        return item["sentence"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")


def get_correct_answer(item: Dict, dataset_key: str) -> str:
    """Get the correct answer for a dataset item."""
    if dataset_key == "qasc":
        return item["answerKey"]
    elif dataset_key == "cola":
        # CoLA: 1 = grammatical (YES), 0 = ungrammatical (NO)
        return "YES" if item["label"] == 1 else "NO"
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")


def convert_to_list(dataset, dataset_key: str) -> List[Dict]:
    """Convert HuggingFace dataset to list of dicts for CoLA compatibility."""
    if dataset_key == "cola":
        return [
            {
                "idx": item.get("idx", i),
                "sentence": item.get("sentence", ""),
                "label": item.get("label", 0),
            }
            for i, item in enumerate(dataset)
        ]
    else:
        # QASC can be used directly as it supports indexing
        return dataset

