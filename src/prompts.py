#!/usr/bin/env python3
"""
Prompt templates for sensitivity experiments.
Team Member 3: Prompt Engineering

Provides four prompt styles (Control, Metacognition, Structure, Politeness)
in a dataset-agnostic way.  Each style is a generic function that accepts
(question_text, answer_format) and returns the full prompt string.

Dataset-specific helpers (QASC, CoLA) wrap the generic templates so that
the existing pipeline in run_experiment.py keeps working without changes.
Adding a new dataset only requires defining its `format_*_base()` and
`answer_format` -- the four styles work automatically.
"""

from typing import Dict, Callable, Optional


# =====================================================================
# GENERIC (dataset-agnostic) PROMPT TEMPLATES
# =====================================================================

def generic_control(question_text: str, answer_format: str) -> str:
    """Standard zero-shot instruction."""
    return f"""{question_text}

Answer with {answer_format}:"""


def generic_metacognition(question_text: str, answer_format: str) -> str:
    """Adds self-check / verification triggers."""
    return f"""{question_text}

Think carefully about the information provided.
Verify your reasoning before answering.
Answer with {answer_format}:"""


def generic_structure(question_text: str, answer_format: str) -> str:
    """Enforces strict JSON structured output."""
    return f"""{question_text}

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "YOUR_ANSWER"}}

Output only the JSON, nothing else."""


def generic_politeness(question_text: str, answer_format: str) -> str:
    """Adds polite conversational markers."""
    return f"""Hello! I would really appreciate your help with this.

{question_text}

Please provide your answer ({answer_format}). Thank you!"""


GENERIC_PROMPTS: Dict[str, Callable] = {
    "control": generic_control,
    "metacognition": generic_metacognition,
    "structure": generic_structure,
    "politeness": generic_politeness,
}


# =====================================================================
# QASC PROMPT TEMPLATES (8-way multiple choice)
# Wrap the generic templates with dataset-specific formatting.
# =====================================================================

def _qasc_base(item: Dict, perturbed_text: Optional[str] = None) -> str:
    """Get the base text for a QASC item, using perturbed text if given."""
    if perturbed_text:
        return perturbed_text
    from datasets_config import format_qasc_base
    return format_qasc_base(item)


QASC_ANSWER_FORMAT = "just the letter (A-H)"


def qasc_control(item: Dict, perturbed_text: Optional[str] = None) -> str:
    return generic_control(_qasc_base(item, perturbed_text), QASC_ANSWER_FORMAT)


def qasc_metacognition(item: Dict, perturbed_text: Optional[str] = None) -> str:
    return generic_metacognition(_qasc_base(item, perturbed_text), QASC_ANSWER_FORMAT)


def qasc_structure(item: Dict, perturbed_text: Optional[str] = None) -> str:
    base = _qasc_base(item, perturbed_text)
    return f"""{base}

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "X"}}

Where X is the letter A-H. Output only the JSON, nothing else."""


def qasc_politeness(item: Dict, perturbed_text: Optional[str] = None) -> str:
    return generic_politeness(_qasc_base(item, perturbed_text), QASC_ANSWER_FORMAT)


QASC_PROMPTS: Dict[str, Callable] = {
    "control": qasc_control,
    "metacognition": qasc_metacognition,
    "structure": qasc_structure,
    "politeness": qasc_politeness,
}


# =====================================================================
# CoLA PROMPT TEMPLATES (binary grammaticality)
# =====================================================================

COLA_ANSWER_FORMAT = "Yes or No"


def cola_control(sentence: str) -> str:
    text = f'Is this sentence grammatically correct? Answer Yes or No.\n\nSentence: "{sentence}"'
    return f"""{text}

Answer:"""


def cola_metacognition(sentence: str) -> str:
    return f"""Is this sentence grammatically correct? Answer Yes or No.
Before answering, carefully check the grammar rules. Verify your answer is correct.

Sentence: "{sentence}"

Think about it carefully, then answer:"""


def cola_structure(sentence: str) -> str:
    return f"""Analyze whether the following sentence is grammatically correct.

Sentence: "{sentence}"

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "Yes or No"}}

(Yes = grammatically correct, No = has error)
Output only the JSON, nothing else."""


def cola_politeness(sentence: str) -> str:
    return f"""Hello! I would really appreciate your help with this.
Could you please tell me if this sentence is grammatically correct?
Please answer with Yes or No.

Sentence: "{sentence}"

Thank you! Your answer:"""


COLA_PROMPTS: Dict[str, Callable] = {
    "control": cola_control,
    "metacognition": cola_metacognition,
    "structure": cola_structure,
    "politeness": cola_politeness,
}


# =====================================================================
# PROMPT STYLE REGISTRY
# =====================================================================

_DATASET_PROMPTS: Dict[str, Dict[str, Callable]] = {
    "qasc": QASC_PROMPTS,
    "cola": COLA_PROMPTS,
}

# Default max-token settings per (style, dataset) combo.
# Structure prompts produce JSON so they need more tokens.
_MAX_TOKENS: Dict[str, Dict[str, int]] = {
    "qasc": {"control": 20, "metacognition": 20, "structure": 100, "politeness": 20},
    "cola": {"control": 10, "metacognition": 10, "structure": 50, "politeness": 10},
}

DEFAULT_MAX_TOKENS = 20
STRUCTURE_DEFAULT_MAX_TOKENS = 100


def get_prompt_styles(dataset_key: str) -> Dict[str, Callable]:
    """
    Get prompt-style functions for a dataset.

    For registered datasets (qasc, cola) returns dataset-tuned templates.
    For unknown datasets returns the generic templates so that new datasets
    work out of the box.
    """
    if dataset_key in _DATASET_PROMPTS:
        return _DATASET_PROMPTS[dataset_key]
    return GENERIC_PROMPTS


def get_max_tokens(style_name: str, dataset_key: str) -> int:
    """Get recommended max new tokens for a prompt style + dataset pair."""
    ds_tokens = _MAX_TOKENS.get(dataset_key, {})
    if style_name in ds_tokens:
        return ds_tokens[style_name]
    return STRUCTURE_DEFAULT_MAX_TOKENS if style_name == "structure" else DEFAULT_MAX_TOKENS


def register_dataset_prompts(
    dataset_key: str,
    prompts: Dict[str, Callable],
    max_tokens: Optional[Dict[str, int]] = None,
) -> None:
    """
    Register prompt templates for a new dataset at runtime.

    Parameters
    ----------
    dataset_key : str
        Short name of the dataset (e.g. "gsm8k").
    prompts : dict
        Mapping of style name -> callable.
    max_tokens : dict or None
        Optional per-style max-token overrides.
    """
    _DATASET_PROMPTS[dataset_key] = prompts
    if max_tokens:
        _MAX_TOKENS[dataset_key] = max_tokens
