#!/usr/bin/env python3
"""
Perturbation generation for sensitivity experiments.
Team Member 3: Prompt Engineering

Two perturbation strategies:
  1. **Synonym** – NLTK WordNet synonym replacement (1-2 words per variant).
  2. **Paraphrase** – A separate small LM (Flan-T5-Small) rewrites the text
     with a "Rewrite without changing meaning" prompt, producing more natural
     and diverse perturbations than rule-based approaches.

Fallback strategies are used when either method cannot produce enough variants.
"""

import os
import random
import re
import logging
from typing import List, Tuple, Optional, Set

import nltk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data bootstrap
# ---------------------------------------------------------------------------

_NLTK_DATA = [
    "wordnet", "averaged_perceptron_tagger_eng", "punkt_tab",
    "stopwords", "omw-1.4",
]


def ensure_nltk_data() -> None:
    """Download required NLTK data if not already present."""
    for resource in _NLTK_DATA:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


ensure_nltk_data()

from nltk.corpus import wordnet as wn, stopwords as sw
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS: Set[str] = set(sw.words("english"))

SKIP_WORDS: Set[str] = STOPWORDS | {
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "given", "fact", "question", "choices", "answer", "sentence",
    "yes", "no", "true", "false",
    "a", "b", "c", "d", "e", "f", "g", "h",
}

# Penn Treebank POS tags -> WordNet POS mapping
_POS_MAP = {
    "NN": wn.NOUN, "NNS": wn.NOUN, "NNP": wn.NOUN, "NNPS": wn.NOUN,
    "VB": wn.VERB, "VBD": wn.VERB, "VBG": wn.VERB, "VBN": wn.VERB,
    "VBP": wn.VERB, "VBZ": wn.VERB,
    "JJ": wn.ADJ, "JJR": wn.ADJ, "JJS": wn.ADJ,
    "RB": wn.ADV, "RBR": wn.ADV, "RBS": wn.ADV,
}

# Minimum word length eligible for replacement
MIN_WORD_LENGTH = 3

# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """
    Fix random seeds across all libraries for full reproducibility.

    Covers: stdlib random, numpy, torch (CPU + GPU), CUDA determinism,
    and Python hash seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# WordNet helpers
# ---------------------------------------------------------------------------


def _to_wordnet_pos(penn_tag: str) -> Optional[str]:
    """Convert a Penn Treebank POS tag to a WordNet POS constant."""
    return _POS_MAP.get(penn_tag)


def get_synonyms(word: str, pos: Optional[str] = None) -> List[str]:
    """
    Return a deduplicated list of *close* WordNet synonyms for *word*.

    Only lemmas from synsets whose first lemma matches or closely relates
    to *word* are returned, which avoids semantically distant alternatives
    like "cat" -> "bozo" or "mat" -> "lusterlessness".

    Parameters
    ----------
    word : str
        The word to look up.
    pos : str or None
        Optional WordNet POS filter (wn.NOUN, wn.VERB, …).

    Returns
    -------
    list[str]
        Synonym lemmas (excluding the word itself), lowercased, single-word only.
    """
    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
    if not synsets:
        return []

    # Only use synsets where the target word appears as a lemma,
    # and cap at the first 3 synsets (most common senses) to avoid
    # obscure meanings.
    synonyms: List[str] = []
    seen: Set[str] = set()
    matched = 0
    for syn in synsets:
        lemma_names = {l.name().lower() for l in syn.lemmas()}
        if word.lower() not in lemma_names:
            continue
        matched += 1
        if matched > 3:
            break
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if (name != word.lower()
                    and " " not in name
                    and name not in seen
                    and len(name) >= 2):
                seen.add(name)
                synonyms.append(name)

    return synonyms


# ---------------------------------------------------------------------------
# Finding replaceable words
# ---------------------------------------------------------------------------


def find_replaceable_words(text: str) -> List[Tuple[str, Optional[str], int]]:
    """
    Identify content words that can be replaced by a synonym.

    Returns a list of (word, wordnet_pos, token_index) tuples for words
    that are nouns / verbs / adjectives / adverbs, long enough, and not
    in the skip list.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    candidates: List[Tuple[str, Optional[str], int]] = []
    for idx, (token, penn_tag) in enumerate(tagged):
        if len(token) < MIN_WORD_LENGTH:
            continue
        if token.lower() in SKIP_WORDS:
            continue
        if not token.isalpha():
            continue

        wn_pos = _to_wordnet_pos(penn_tag)
        if wn_pos is None:
            continue

        syns = get_synonyms(token.lower(), wn_pos)
        if syns:
            candidates.append((token, wn_pos, idx))

    return candidates


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------


def _replace_token(tokens: List[str], idx: int, replacement: str) -> List[str]:
    """Return a new token list with tokens[idx] replaced, preserving case."""
    new_tokens = list(tokens)
    original = tokens[idx]
    if original[0].isupper():
        replacement = replacement.capitalize()
    if original.isupper():
        replacement = replacement.upper()
    new_tokens[idx] = replacement
    return new_tokens


def _tokens_to_text(tokens: List[str]) -> str:
    """Reconstruct text from tokens, fixing common spacing around punctuation."""
    text = " ".join(tokens)
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,;:!?)\]}])', r'\1', text)
    # Fix spaces after opening brackets
    text = re.sub(r'([({\[]\s+)', lambda m: m.group().strip(), text)
    # Fix double spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def generate_perturbations(
    text: str,
    num: int = 10,
    words_to_replace: int = 1,
) -> List[str]:
    """
    Generate *num* semantic-preserving perturbations of *text*.

    Strategy
    --------
    1. POS-tag the input and identify content words with WordNet synonyms.
    2. For each perturbation, randomly pick *words_to_replace* eligible words
       and swap each with a randomly chosen synonym.
    3. Track already-generated variants to ensure uniqueness.
    4. If WordNet cannot produce enough variants, fall back to lightweight
       surface transforms (article swap, punctuation variation).

    Parameters
    ----------
    text : str
        The original text to perturb.
    num : int
        Desired number of perturbations (default 10).
    words_to_replace : int
        How many words to replace per perturbation (default 1).

    Returns
    -------
    list[str]
        Up to *num* unique perturbed strings.
    """
    tokens = word_tokenize(text)
    candidates = find_replaceable_words(text)

    seen: Set[str] = {text}
    perturbations: List[str] = []

    if candidates:
        # Build synonym cache per candidate index
        syn_cache = {}
        for word, wn_pos, idx in candidates:
            syn_cache[idx] = get_synonyms(word.lower(), wn_pos)

        attempts = 0
        max_attempts = num * 8  # prevent infinite loops

        while len(perturbations) < num and attempts < max_attempts:
            attempts += 1

            k = min(words_to_replace, len(candidates))
            chosen = random.sample(candidates, k)

            new_tokens = list(tokens)
            for word, wn_pos, idx in chosen:
                syns = syn_cache.get(idx, [])
                if not syns:
                    continue
                replacement = random.choice(syns)
                new_tokens = _replace_token(new_tokens, idx, replacement)

            variant = _tokens_to_text(new_tokens)
            if variant not in seen:
                seen.add(variant)
                perturbations.append(variant)

    # ------------------------------------------------------------------
    # Fallback: lightweight surface-level transforms if not enough yet
    # ------------------------------------------------------------------
    fallbacks = _generate_fallbacks(text, num - len(perturbations), seen)
    perturbations.extend(fallbacks)

    return perturbations[:num]


# ---------------------------------------------------------------------------
# Fallback perturbations (surface-level)
# ---------------------------------------------------------------------------

_FILLERS = ["Essentially", "Basically", "In other words", "Put simply", "Simply put"]


def _generate_fallbacks(
    text: str, needed: int, seen: Set[str]
) -> List[str]:
    """Generate lightweight surface perturbations when WordNet isn't enough."""
    if needed <= 0:
        return []

    results: List[str] = []

    # 1. Article / determiner swaps
    for pattern, replacement in [
        (r"\bthe\b", "a"), (r"\ba\b", "the"),
        (r"\bThe\b", "A"), (r"\bA\b", "The"),
    ]:
        if len(results) >= needed:
            break
        variant = re.sub(pattern, replacement, text, count=1)
        if variant != text and variant not in seen:
            seen.add(variant)
            results.append(variant)

    # 2. Filler prefix
    for filler in _FILLERS:
        if len(results) >= needed:
            break
        first_lower = text[0].lower() + text[1:] if len(text) > 1 else text
        variant = f"{filler}, {first_lower}"
        if variant not in seen:
            seen.add(variant)
            results.append(variant)

    # 3. Punctuation variation
    if len(results) < needed:
        if text.endswith("."):
            variant = text[:-1]
        elif text.endswith("?"):
            variant = text[:-1] + "."
        else:
            variant = text + "."
        if variant not in seen:
            seen.add(variant)
            results.append(variant)

    return results[:needed]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_perturbation(original: str, perturbed: str) -> bool:
    """
    Basic sanity check that a perturbation is reasonable.

    Checks:
    - Not identical to original
    - Length is within 50 % of original
    - At least 60 % of words are shared (meaning preserved)
    """
    if original == perturbed:
        return False

    orig_len = len(original)
    if orig_len == 0:
        return False

    ratio = len(perturbed) / orig_len
    if ratio < 0.5 or ratio > 1.5:
        return False

    orig_words = set(original.lower().split())
    pert_words = set(perturbed.lower().split())
    if not orig_words:
        return False

    overlap = len(orig_words & pert_words) / len(orig_words)
    return overlap >= 0.6


def generate_and_validate(
    text: str,
    num: int = 10,
    words_to_replace: int = 1,
) -> List[str]:
    """
    Generate perturbations and filter out any that fail validation.

    Attempts to produce extra candidates to compensate for filtered ones.
    """
    raw = generate_perturbations(text, num=num * 2, words_to_replace=words_to_replace)
    valid = [p for p in raw if validate_perturbation(text, p)]
    return valid[:num]


# ---------------------------------------------------------------------------
# LLM-based paraphrase perturbations
# ---------------------------------------------------------------------------

_PARAPHRASE_PROMPTS = [
    "Rewrite this without changing its meaning:\n{text}",
    "Rephrase the following in different words:\n{text}",
    "Say the same thing using different phrasing:\n{text}",
    "Express this differently while keeping the meaning:\n{text}",
    "Paraphrase this text:\n{text}",
    "Reword the following sentence:\n{text}",
    "Write an alternative version of this that means the same thing:\n{text}",
    "State the following in another way:\n{text}",
    "Convey the same idea with different wording:\n{text}",
    "Reformulate this text without altering the meaning:\n{text}",
]


class ParaphraseGenerator:
    """
    Uses a small Flan-T5 model to produce LLM-based paraphrases.

    The paraphraser is always ``google/flan-t5-small`` so that it is
    lightweight and distinct from any model being evaluated.
    """

    _PARAPHRASER_MODEL = "google/flan-t5-small"

    def __init__(self, device: Optional[str] = None, seed: int = 2266):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self._base_seed = seed

        logger.info("Loading paraphraser: %s on %s", self._PARAPHRASER_MODEL, device)
        self.tokenizer = AutoTokenizer.from_pretrained(self._PARAPHRASER_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self._PARAPHRASER_MODEL
        ).to(device)
        self.model.eval()
        logger.info("Paraphraser ready")

    def _generate_one(self, prompt: str, attempt: int = 0, max_new_tokens: int = 256) -> str:
        """Run a single generation with sampling for diversity.

        Seeds torch with (base_seed + attempt) before every call so that
        the same experiment config always produces the same paraphrases,
        even when do_sample=True.
        """
        import torch

        torch.manual_seed(self._base_seed + attempt)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._base_seed + attempt)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate(self, text: str, num: int = 10) -> List[str]:
        """
        Produce up to *num* unique paraphrases of *text*.

        Cycles through diverse prompt templates and retries with sampling
        to maximise variety.
        """
        seen: Set[str] = {text}
        results: List[str] = []
        max_attempts = num * 4

        for attempt in range(max_attempts):
            if len(results) >= num:
                break
            template = _PARAPHRASE_PROMPTS[attempt % len(_PARAPHRASE_PROMPTS)]
            prompt = template.format(text=text)
            candidate = self._generate_one(prompt, attempt=attempt)

            if candidate and candidate not in seen and candidate != text:
                seen.add(candidate)
                results.append(candidate)

        return results[:num]


# Singleton so we only load the model once per process
_paraphraser: Optional[ParaphraseGenerator] = None


def get_paraphraser(device: Optional[str] = None, seed: int = 2266) -> ParaphraseGenerator:
    """Return (and lazily initialise) the shared paraphrase generator."""
    global _paraphraser
    if _paraphraser is None:
        _paraphraser = ParaphraseGenerator(device=device, seed=seed)
    return _paraphraser


def generate_paraphrase_perturbations(
    text: str,
    num: int = 10,
    device: Optional[str] = None,
    seed: int = 2266,
) -> List[str]:
    """
    High-level API: generate *num* paraphrase perturbations of *text*.

    Falls back to synonym-based perturbations for any shortfall.
    """
    para = get_paraphraser(device=device, seed=seed)
    paraphrases = para.generate(text, num=num)

    if len(paraphrases) < num:
        shortfall = num - len(paraphrases)
        logger.info(
            "Paraphraser produced %d/%d; filling %d with synonym fallback",
            len(paraphrases), num, shortfall,
        )
        seen = set(paraphrases) | {text}
        fallback = generate_perturbations(text, num=shortfall * 2, words_to_replace=1)
        for fb in fallback:
            if fb not in seen and len(paraphrases) < num:
                seen.add(fb)
                paraphrases.append(fb)

    return paraphrases[:num]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    set_all_seeds(2266)

    samples = [
        "What is the effect of temperature on the rate of a chemical reaction?",
        "The cat sat on the mat.",
        "Given:\nFact 1: friction causes heat\nFact 2: heat can cause chemical changes\n\nQuestion: What causes chemical changes?\nChoices:\n  A) erosion\n  B) friction\n  C) gravity",
    ]

    method = sys.argv[1] if len(sys.argv) > 1 else "synonym"

    for text in samples:
        print(f"\nOriginal: {text[:80]}{'...' if len(text) > 80 else ''}")
        print("-" * 60)
        if method == "paraphrase":
            perturbs = generate_paraphrase_perturbations(text, num=5)
        else:
            perturbs = generate_and_validate(text, num=5)
        for i, p in enumerate(perturbs, 1):
            print(f"  [{i}] {p[:100]}{'...' if len(p) > 100 else ''}")
        if not perturbs:
            print("  (no perturbations generated)")
