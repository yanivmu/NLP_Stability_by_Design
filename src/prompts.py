#!/usr/bin/env python3
"""
Prompt templates for different datasets and prompt styles.

Prompt properties tested:
- Control: Standard zero-shot instruction
- Metacognition: Self-check triggers ("verify your reasoning")  
- Structure: Enforced JSON output
- Politeness: Conversational fillers ("please", "thank you")
"""

from typing import Dict, Callable, Optional
from datasets_config import format_qasc_base


# =====================================================================
# QASC PROMPT TEMPLATES (8-way multiple choice)
# =====================================================================

def qasc_control(item: Dict, perturbed_text: Optional[str] = None) -> str:
    """Standard zero-shot instruction for QASC."""
    base = perturbed_text if perturbed_text else format_qasc_base(item)
    return f"""{base}

Answer with just the letter (A-H):"""


def qasc_metacognition(item: Dict, perturbed_text: Optional[str] = None) -> str:
    """Adds self-check triggers for QASC."""
    base = perturbed_text if perturbed_text else format_qasc_base(item)
    return f"""{base}

Think carefully about how the facts relate to the question.
Verify your reasoning before answering.
Answer with just the letter (A-H):"""


def qasc_structure(item: Dict, perturbed_text: Optional[str] = None) -> str:
    """Enforces strict JSON structured output for QASC."""
    base = perturbed_text if perturbed_text else format_qasc_base(item)
    return f"""{base}

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "X"}}

Where X is the letter A-H. Output only the JSON, nothing else."""


def qasc_politeness(item: Dict, perturbed_text: Optional[str] = None) -> str:
    """Adds conversational fillers for QASC."""
    base = perturbed_text if perturbed_text else format_qasc_base(item)
    return f"""Hello! I'd really appreciate your help with this question.

{base}

Please provide your answer (just the letter A-H). Thank you!"""


# =====================================================================
# CoLA PROMPT TEMPLATES (binary grammaticality)
# =====================================================================

def cola_control(sentence: str) -> str:
    """Standard zero-shot instruction for CoLA."""
    return f"""Is this sentence grammatically correct? Answer Yes or No.

Sentence: "{sentence}"

Answer:"""


def cola_metacognition(sentence: str) -> str:
    """Adds self-check triggers for CoLA."""
    return f"""Is this sentence grammatically correct? Answer Yes or No.
Before answering, carefully check the grammar rules. Verify your answer is correct.

Sentence: "{sentence}"

Think about it carefully, then answer:"""


def cola_structure(sentence: str) -> str:
    """Enforces strict JSON structured output for CoLA."""
    return f"""Analyze whether the following sentence is grammatically correct.

Sentence: "{sentence}"

You MUST respond with valid JSON in this exact format:
{{"reasoning": "your brief explanation here", "final_answer": "Yes or No"}}

(Yes = grammatically correct, No = has error)
Output only the JSON, nothing else."""


def cola_politeness(sentence: str) -> str:
    """Adds conversational fillers for CoLA."""
    return f"""Hello! I would really appreciate your help with this.
Could you please tell me if this sentence is grammatically correct?
Please answer with Yes or No.

Sentence: "{sentence}"

Thank you! Your answer:"""


# =====================================================================
# PROMPT STYLE REGISTRY
# =====================================================================

# QASC prompt functions take (item, optional perturbed_text)
QASC_PROMPTS: Dict[str, Callable] = {
    "control": qasc_control,
    "metacognition": qasc_metacognition,
    "structure": qasc_structure,
    "politeness": qasc_politeness,
}

# CoLA prompt functions take (sentence)
COLA_PROMPTS: Dict[str, Callable] = {
    "control": cola_control,
    "metacognition": cola_metacognition,
    "structure": cola_structure,
    "politeness": cola_politeness,
}


def get_prompt_styles(dataset_key: str) -> Dict[str, Callable]:
    """Get prompt style functions for a dataset."""
    if dataset_key == "qasc":
        return QASC_PROMPTS
    elif dataset_key == "cola":
        return COLA_PROMPTS
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")


def get_max_tokens(style_name: str, dataset_key: str) -> int:
    """Get recommended max tokens for a prompt style."""
    # Structure prompts need more tokens for JSON output
    if style_name == "structure":
        return 100 if dataset_key == "qasc" else 50
    else:
        return 20 if dataset_key == "qasc" else 10

