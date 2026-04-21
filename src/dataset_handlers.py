#!/usr/bin/env python3
"""
Dataset Handler Registry — Factory / Registry pattern for datasets.

Each dataset is encapsulated by a concrete ``DatasetHandler`` subclass that
owns **all** dataset-specific logic:

    * loading from HuggingFace
    * formatting items for perturbation
    * building prompts (4 styles)
    * extracting the correct answer
    * parsing model responses

The main pipeline (``run_experiment.py``) interacts exclusively with the
abstract ``DatasetHandler`` interface, so adding a new dataset requires
only a new subclass — zero changes to the pipeline code.

Usage
-----
    handler = get_dataset_handler("qasc")
    dataset = handler.load()
    text    = handler.get_item_text(item)
    prompt  = handler.build_prompt(item, "control")
    answer  = handler.parse_answer(response)
"""

from __future__ import annotations

import re
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

from datasets_config import DatasetConfig, AnswerType, DATASET_CONFIGS

# =====================================================================
# ABSTRACT BASE CLASS
# =====================================================================


class DatasetHandler(ABC):
    """
    Abstract interface every dataset must implement.

    Subclasses are auto-registered via the ``register_dataset`` decorator.
    """

    # Subclasses MUST set these class-level attributes.
    KEY: str = ""           # short lookup key, e.g. "qasc"
    CONFIG: DatasetConfig = None  # type: ignore[assignment]

    def __init__(self, **kwargs):
        """Accept and ignore unknown kwargs for forward-compatibility.

        Concrete handlers that need specific parameters (e.g. QASC's
        ``inject_facts``) override ``__init__`` and consume them
        before calling ``super().__init__(**remaining)``.
        """

    # ---- loading ----

    @abstractmethod
    def load(self, split: Optional[str] = None) -> Any:
        """Load the dataset from HuggingFace and return the raw dataset object."""

    @abstractmethod
    def sample_items(self, dataset: Any, n: int, seed: int) -> List[Dict]:
        """Return *n* items from *dataset* as a list of dicts."""

    # ---- text / answers ----

    @abstractmethod
    def get_item_text(self, item: Dict) -> str:
        """Return the primary text to be perturbed."""

    @abstractmethod
    def get_correct_answer(self, item: Dict) -> str:
        """Return the normalised ground-truth answer string."""

    # ---- prompts ----

    @abstractmethod
    def build_prompt(
        self,
        item: Dict,
        style: str,
        perturbed_text: Optional[str] = None,
    ) -> str:
        """Build a full prompt for *item* using the named *style*."""

    @abstractmethod
    def get_max_tokens(self, style: str) -> int:
        """Recommended ``max_new_tokens`` for a given prompt style."""

    # ---- answer parsing ----

    @abstractmethod
    def parse_answer(self, response: str, is_structured: bool = False) -> str:
        """Extract a normalised answer from raw model output."""

    def parse_answer_verbose(self, response: str, is_structured: bool = False) -> tuple:
        """Return ``(answer, parse_method)`` — override for dataset-specific detail."""
        return self.parse_answer(response, is_structured), "unknown"


# =====================================================================
# REGISTRY
# =====================================================================

_REGISTRY: Dict[str, type[DatasetHandler]] = {}


def register_dataset(cls: type[DatasetHandler]) -> type[DatasetHandler]:
    """Class decorator that registers a ``DatasetHandler`` subclass."""
    key = cls.KEY
    if not key:
        raise ValueError(f"{cls.__name__} must define a non-empty KEY attribute")
    _REGISTRY[key] = cls
    return cls


def get_dataset_handler(key: str, **kwargs) -> DatasetHandler:
    """Instantiate and return the handler registered under *key*.

    Extra *kwargs* are forwarded to the handler constructor, allowing
    callers to override defaults (e.g. ``inject_facts=True`` for QASC).
    """
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset: {key!r}. Registered: {available}")
    return _REGISTRY[key](**kwargs)


def list_registered_datasets() -> List[str]:
    """Return the keys of all registered dataset handlers."""
    return sorted(_REGISTRY.keys())


# =====================================================================
# HELPER: JSON / letter / yes-no parsing (shared by concrete handlers)
# =====================================================================

_LETTER_PATTERN = re.compile(r'\b([A-H])\b', re.IGNORECASE)
_JSON_ANSWER = re.compile(
    r'"final_answer"\s*:\s*"?([^"}\s,]+)"?', re.IGNORECASE
)
_MARKDOWN_FENCE = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) that some instruct models add."""
    m = _MARKDOWN_FENCE.search(text)
    return m.group(1) if m else text


def _parse_letter(response: str, is_structured: bool = False) -> str:
    """Parse a single letter (A–H) from model output."""
    ans, _ = _parse_letter_verbose(response, is_structured)
    return ans


def _parse_letter_verbose(response: str, is_structured: bool = False) -> tuple:
    """Return ``(answer, parse_method)`` describing how the answer was extracted."""
    text = response.strip()
    if is_structured:
        cleaned = _strip_markdown_fences(text)
        m = _JSON_ANSWER.search(cleaned)
        if m:
            letter = m.group(1).strip().upper()
            if letter and letter[0] in "ABCDEFGH":
                return letter[0], 'json_key "final_answer"'
        jm = re.search(r'\{[^}]+\}', cleaned)
        if jm:
            try:
                data = json.loads(jm.group())
                val = str(data.get("final_answer", "")).strip().upper()
                if val and val[0] in "ABCDEFGH":
                    return val[0], 'json_parse "final_answer"'
            except (json.JSONDecodeError, AttributeError):
                pass
    upper = text.upper()
    m = _LETTER_PATTERN.search(upper)
    if m:
        return m.group(1), r"regex \b[A-H]\b"
    if upper and upper[0] in "ABCDEFGH":
        return upper[0], "first_char A-H"
    return "", "no_match"


def _parse_yes_no(response: str, is_structured: bool = False) -> str:
    """Parse YES / NO from model output."""
    ans, _ = _parse_yes_no_verbose(response, is_structured)
    return ans


_ANSWER_SIGNAL = re.compile(
    r'(?:the\s+answer\s+is|answer\s*:|therefore|thus|so)\s*(yes|no)\b',
    re.IGNORECASE,
)

_NEG_GRAMMAR_KW = (
    "not correct", "incorrect", "ungrammatical",
    "grammatical error", "grammar error", "not grammatical",
)


def _parse_yes_no_verbose(response: str, is_structured: bool = False) -> tuple:
    """Return ``(answer, parse_method)`` describing how the answer was extracted.

    Uses a 4-pass strategy designed to capture the model's *final* answer
    rather than intermediate reasoning tokens:

    1. Explicit answer-signal patterns anywhere (take the last match).
    2. Bare ``yes``/``no`` in the last 50 characters (tail).
    3. Grammaticality keywords in the tail (negatives checked first).
    4. Full-scan fallback for bare ``yes``/``no`` anywhere.
    """
    text = response.strip()

    # --- Structured (JSON) path ---
    if is_structured:
        cleaned = _strip_markdown_fences(text)
        m = _JSON_ANSWER.search(cleaned)
        if m:
            val = m.group(1).strip().lower()
            if "yes" in val:
                return "YES", 'json_regex "final_answer"'
            if "no" in val:
                return "NO", 'json_regex "final_answer"'
        jm = re.search(r'\{[^}]+\}', cleaned)
        if jm:
            try:
                data = json.loads(jm.group())
                val = str(data.get("final_answer", "")).lower()
                if "yes" in val:
                    return "YES", 'json_parse "final_answer"'
                if "no" in val:
                    return "NO", 'json_parse "final_answer"'
            except (json.JSONDecodeError, AttributeError):
                pass

    lower = text.lower()

    # Pass 1 — explicit answer signals (last match wins)
    signals = list(_ANSWER_SIGNAL.finditer(lower))
    if signals:
        last = signals[-1].group(1).lower()
        return ("YES" if last == "yes" else "NO"), "answer_signal_pattern"

    # Pass 2 — bare yes/no in the tail (last 50 chars)
    tail = lower[-50:] if len(lower) > 50 else lower
    if re.search(r'\byes\b', tail):
        return "YES", "text_match tail_50_chars"
    if re.search(r'\bno\b', tail):
        return "NO", "text_match tail_50_chars"

    # Pass 3 — grammaticality keywords in the tail (negatives first)
    if any(kw in tail for kw in _NEG_GRAMMAR_KW):
        return "NO", "text_match grammaticality_keyword_tail"
    if re.search(r'\bcorrect\b', tail) or re.search(r'\bgrammatical(ly)?\b', tail):
        return "YES", "text_match grammaticality_keyword_tail"

    # Pass 4 — full-scan fallback for very short responses
    if re.search(r'\byes\b', lower):
        return "YES", "full_scan"
    if re.search(r'\bno\b', lower):
        return "NO", "full_scan"

    return "", "no_match"


# =====================================================================
# CONCRETE HANDLER: QASC
# =====================================================================

_QASC_ANSWER_FMT = "just the letter (A-H)"


def _format_qasc_base(item: Dict, inject_facts: bool = True) -> str:
    """Build the base QASC question string.

    Parameters
    ----------
    inject_facts : bool
        When *True* (default), Fact 1 and Fact 2 are prepended.
        Set to *False* to evaluate the model without supporting facts.
    """
    question = item["question"]
    choices = item["choices"]["text"]
    labels = item["choices"]["label"]
    choices_str = "\n".join(f"  {l}) {t}" for l, t in zip(labels, choices))

    if inject_facts:
        fact1 = item["fact1"]
        fact2 = item["fact2"]
        return (
            f"Given:\n"
            f"Fact 1: {fact1}\n"
            f"Fact 2: {fact2}\n\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_str}"
        )
    return f"Question: {question}\nChoices:\n{choices_str}"


@register_dataset
class QASCHandler(DatasetHandler):
    KEY = "qasc"
    CONFIG = DATASET_CONFIGS["qasc"]

    def __init__(self, inject_facts: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.inject_facts = inject_facts

    def load(self, split: Optional[str] = None) -> Any:
        from datasets import load_dataset as hf_load
        split = split or self.CONFIG.split
        print(f"Loading {self.CONFIG.name} dataset...")
        ds = hf_load(self.CONFIG.hf_name, split=split)
        print(f"Loaded {len(ds)} items")
        return ds

    def sample_items(self, dataset: Any, n: int, seed: int) -> List[Dict]:
        import random
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(n, len(dataset)))
        return [dataset[i] for i in indices]

    def get_item_text(self, item: Dict) -> str:
        return _format_qasc_base(item, inject_facts=self.inject_facts)

    def get_correct_answer(self, item: Dict) -> str:
        return item["answerKey"]

    # ---- prompts ----

    def build_prompt(self, item: Dict, style: str, perturbed_text: Optional[str] = None) -> str:
        base = perturbed_text if perturbed_text else self.get_item_text(item)
        if style == "control":
            return f"{base}\n\nAnswer with {_QASC_ANSWER_FMT}:"
        if style == "metacognition":
            return (
                f"{base}\n\n"
                "Think carefully about the information provided.\n"
                "Verify your reasoning before answering.\n"
                f"Answer with {_QASC_ANSWER_FMT}:"
            )
        if style == "structure":
            return (
                f"{base}\n\n"
                "You MUST respond with valid JSON in this exact format:\n"
                '{"reasoning": "your brief explanation here", "final_answer": "X"}\n\n'
                "Where X is the letter A-H. Output only the JSON, nothing else."
            )
        if style == "politeness":
            return (
                "Hello! I would really appreciate your help with this.\n\n"
                f"{base}\n\n"
                f"Please provide your answer ({_QASC_ANSWER_FMT}). Thank you!"
            )
        raise ValueError(f"Unknown prompt style: {style!r}")

    def get_max_tokens(self, style: str) -> int:
        return 100 if style == "structure" else 20

    def parse_answer(self, response: str, is_structured: bool = False) -> str:
        return _parse_letter(response, is_structured=is_structured)

    def parse_answer_verbose(self, response: str, is_structured: bool = False) -> tuple:
        return _parse_letter_verbose(response, is_structured=is_structured)


# =====================================================================
# CONCRETE HANDLER: CoLA
# =====================================================================

@register_dataset
class CoLAHandler(DatasetHandler):
    KEY = "cola"
    CONFIG = DATASET_CONFIGS["cola"]

    def load(self, split: Optional[str] = None) -> Any:
        from datasets import load_dataset as hf_load
        split = split or self.CONFIG.split
        print(f"Loading {self.CONFIG.name} dataset...")
        ds = hf_load(self.CONFIG.hf_name, self.CONFIG.hf_subset, split=split)
        print(f"Loaded {len(ds)} items")
        return ds

    def sample_items(self, dataset: Any, n: int, seed: int) -> List[Dict]:
        import random
        random.seed(seed)
        data_list = [
            {"idx": item.get("idx", i), "sentence": item.get("sentence", ""), "label": item.get("label", 0)}
            for i, item in enumerate(dataset)
        ]
        return random.sample(data_list, min(n, len(data_list)))

    def get_item_text(self, item: Dict) -> str:
        return item["sentence"]

    def get_correct_answer(self, item: Dict) -> str:
        return "YES" if item["label"] == 1 else "NO"

    def build_prompt(self, item: Dict, style: str, perturbed_text: Optional[str] = None) -> str:
        sentence = perturbed_text if perturbed_text else item["sentence"]
        if style == "control":
            return (
                f'Is this sentence grammatically correct? Answer Yes or No.\n\n'
                f'Sentence: "{sentence}"\n\n'
                f'Answer:'
            )
        if style == "metacognition":
            return (
                f'Is this sentence grammatically correct? Answer Yes or No.\n'
                f'Before answering, carefully check the grammar rules. '
                f'Verify your answer is correct.\n\n'
                f'Sentence: "{sentence}"\n\n'
                f'Think about it carefully, then answer:'
            )
        if style == "structure":
            return (
                f'Analyze whether the following sentence is grammatically correct.\n\n'
                f'Sentence: "{sentence}"\n\n'
                f'You MUST respond with valid JSON in this exact format:\n'
                f'{{"reasoning": "your brief explanation here", "final_answer": "Yes or No"}}\n\n'
                f'(Yes = grammatically correct, No = has error)\n'
                f'Output only the JSON, nothing else.'
            )
        if style == "politeness":
            return (
                f'Hello! I would really appreciate your help with this.\n'
                f'Could you please tell me if this sentence is grammatically correct?\n'
                f'Please answer with Yes or No.\n\n'
                f'Sentence: "{sentence}"\n\n'
                f'Thank you! Your answer:'
            )
        raise ValueError(f"Unknown prompt style: {style!r}")

    def get_max_tokens(self, style: str) -> int:
        if style == "structure":
            return 150
        if style == "metacognition":
            return 200
        return 10

    def parse_answer(self, response: str, is_structured: bool = False) -> str:
        return _parse_yes_no(response, is_structured=is_structured)

    def parse_answer_verbose(self, response: str, is_structured: bool = False) -> tuple:
        return _parse_yes_no_verbose(response, is_structured=is_structured)


# =====================================================================
# STUB HANDLER: CSQA (CommonsenseQA) — ready for implementation
# =====================================================================

# Register the CSQA config in DATASET_CONFIGS so the rest of the infra
# (prompts, analysis) can reference it.
DATASET_CONFIGS.setdefault(
    "csqa",
    DatasetConfig(
        name="CommonsenseQA",
        hf_name="tau/commonsense_qa",
        split="validation",
        answer_type=AnswerType.LETTER,
        random_baseline=0.20,   # 1/5 for 5 choices
        valid_threshold=0.40,
        valid_answers="ABCDE",
    ),
)

_CSQA_ANSWER_FMT = "just the letter (A-E)"


@register_dataset
class CSQAHandler(DatasetHandler):
    """CommonsenseQA — 5-way multiple choice commonsense reasoning."""

    KEY = "csqa"
    CONFIG = DATASET_CONFIGS["csqa"]

    def load(self, split: Optional[str] = None) -> Any:
        from datasets import load_dataset as hf_load
        split = split or self.CONFIG.split
        print(f"Loading {self.CONFIG.name} dataset...")
        ds = hf_load(self.CONFIG.hf_name, split=split)
        print(f"Loaded {len(ds)} items")
        return ds

    def sample_items(self, dataset: Any, n: int, seed: int) -> List[Dict]:
        import random
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(n, len(dataset)))
        return [dataset[i] for i in indices]

    def get_item_text(self, item: Dict) -> str:
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        choices_str = "\n".join(f"  {l}) {t}" for l, t in zip(labels, choices))
        return f"Question: {question}\nChoices:\n{choices_str}"

    def get_correct_answer(self, item: Dict) -> str:
        return item["answerKey"]

    def build_prompt(self, item: Dict, style: str, perturbed_text: Optional[str] = None) -> str:
        base = perturbed_text if perturbed_text else self.get_item_text(item)
        if style == "control":
            return f"{base}\n\nAnswer with {_CSQA_ANSWER_FMT}:"
        if style == "metacognition":
            return (
                f"{base}\n\n"
                "Think carefully about common sense reasoning.\n"
                "Verify your reasoning before answering.\n"
                f"Answer with {_CSQA_ANSWER_FMT}:"
            )
        if style == "structure":
            return (
                f"{base}\n\n"
                "You MUST respond with valid JSON in this exact format:\n"
                '{"reasoning": "your brief explanation here", "final_answer": "X"}\n\n'
                "Where X is the letter A-E. Output only the JSON, nothing else."
            )
        if style == "politeness":
            return (
                "Hello! I would really appreciate your help with this.\n\n"
                f"{base}\n\n"
                f"Please provide your answer ({_CSQA_ANSWER_FMT}). Thank you!"
            )
        raise ValueError(f"Unknown prompt style: {style!r}")

    def get_max_tokens(self, style: str) -> int:
        return 100 if style == "structure" else 20

    def parse_answer(self, response: str, is_structured: bool = False) -> str:
        return _parse_letter(response, is_structured=is_structured)

    def parse_answer_verbose(self, response: str, is_structured: bool = False) -> tuple:
        return _parse_letter_verbose(response, is_structured=is_structured)


# =====================================================================
# STUB HANDLER: GSM8K — ready for implementation
# =====================================================================

DATASET_CONFIGS.setdefault(
    "gsm8k",
    DatasetConfig(
        name="GSM8K",
        hf_name="openai/gsm8k",
        hf_subset="main",
        split="test",
        answer_type=AnswerType.YES_NO,  # repurposed: free-form numeric
        random_baseline=0.0,
        valid_threshold=0.20,
        valid_answers="",
    ),
)

_GSM8K_NUMERIC = re.compile(r'-?\d[\d,]*\.?\d*')


@register_dataset
class GSM8KHandler(DatasetHandler):
    """GSM8K — grade-school math with numeric free-form answers."""

    KEY = "gsm8k"
    CONFIG = DATASET_CONFIGS["gsm8k"]

    def load(self, split: Optional[str] = None) -> Any:
        from datasets import load_dataset as hf_load
        split = split or self.CONFIG.split
        print(f"Loading {self.CONFIG.name} dataset...")
        ds = hf_load(self.CONFIG.hf_name, self.CONFIG.hf_subset, split=split)
        print(f"Loaded {len(ds)} items")
        return ds

    def sample_items(self, dataset: Any, n: int, seed: int) -> List[Dict]:
        import random
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(n, len(dataset)))
        return [dataset[i] for i in indices]

    def get_item_text(self, item: Dict) -> str:
        return item["question"]

    def get_correct_answer(self, item: Dict) -> str:
        answer_text = item["answer"]
        m = re.search(r'####\s*(.+)', answer_text)
        if m:
            return m.group(1).strip().replace(",", "")
        nums = _GSM8K_NUMERIC.findall(answer_text)
        return nums[-1].replace(",", "") if nums else answer_text.strip()

    def build_prompt(self, item: Dict, style: str, perturbed_text: Optional[str] = None) -> str:
        base = perturbed_text if perturbed_text else self.get_item_text(item)
        if style == "control":
            return f"{base}\n\nProvide only the final numeric answer:"
        if style == "metacognition":
            return (
                f"{base}\n\n"
                "Think step by step. Double-check your arithmetic.\n"
                "Provide only the final numeric answer:"
            )
        if style == "structure":
            return (
                f"{base}\n\n"
                "You MUST respond with valid JSON in this exact format:\n"
                '{"reasoning": "your step-by-step solution", "final_answer": "NUMBER"}\n\n'
                "Output only the JSON, nothing else."
            )
        if style == "politeness":
            return (
                "Hello! Could you please help me with this math problem?\n\n"
                f"{base}\n\n"
                "Please provide only the final numeric answer. Thank you!"
            )
        raise ValueError(f"Unknown prompt style: {style!r}")

    def get_max_tokens(self, style: str) -> int:
        return 150 if style == "structure" else 50

    def parse_answer(self, response: str, is_structured: bool = False) -> str:
        text = response.strip()
        if is_structured:
            m = _JSON_ANSWER.search(text)
            if m:
                return m.group(1).strip().replace(",", "")
            jm = re.search(r'\{[^}]+\}', text)
            if jm:
                try:
                    data = json.loads(jm.group())
                    val = str(data.get("final_answer", ""))
                    return val.strip().replace(",", "")
                except (json.JSONDecodeError, AttributeError):
                    pass
        nums = _GSM8K_NUMERIC.findall(text)
        return nums[-1].replace(",", "") if nums else ""
