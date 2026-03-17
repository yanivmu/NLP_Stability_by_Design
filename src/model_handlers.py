#!/usr/bin/env python3
"""
Model Handler Registry — Factory / Registry pattern for models.

Each model family is encapsulated by a concrete ``ModelHandler`` subclass
that owns **all** model-specific logic:

    * loading weights and tokenizer from HuggingFace
    * tokenizer configuration (pad tokens, padding side)
    * prompt formatting (chat templates for instruct models)
    * generation and decoding (seq2seq vs causal output slicing)

The main pipeline (``run_experiment.py``) interacts exclusively with the
abstract ``ModelHandler`` interface, so adding a new model family requires
only a new subclass — zero changes to the pipeline code.

Usage
-----
    handler = get_model_handler("flan-t5-base")
    handler.load(device)
    responses = handler.generate(prompts, max_new_tokens=20)

Class hierarchy
---------------
    ModelHandler (ABC)
    ├── Seq2SeqModelHandler          # Flan-T5 family
    ├── CausalModelHandler           # Base causal LMs (Pythia, Llama base)
    └── InstructCausalModelHandler   # Chat / Instruct models (Llama-Instruct, Phi-3)
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from models import ModelConfig, ModelType, MODEL_CONFIGS, get_device


# =====================================================================
# ABSTRACT BASE CLASS
# =====================================================================


class ModelHandler(ABC):
    """Abstract interface every model handler must implement."""

    def __init__(self, key: str, config: ModelConfig):
        self.key = key
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device: Optional[str] = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def hf_name(self) -> str:
        return self.config.hf_name

    @property
    def batch_size(self) -> int:
        """Recommended batch size (smaller for fp16 models to avoid OOM)."""
        return 4 if self.config.use_fp16 else 8

    # ---- lifecycle ----

    @abstractmethod
    def load(self, device: Optional[str] = None) -> "ModelHandler":
        """Load model and tokenizer onto *device*. Returns self for chaining."""

    # ---- prompt formatting ----

    def format_prompts(self, prompts: List[str]) -> List[str]:
        """Optional prompt transformation (e.g. chat template).

        The default implementation is identity — subclasses for instruct
        models override this to wrap prompts in a chat template.
        """
        return prompts

    # ---- generation ----

    @abstractmethod
    def generate(self, prompts: List[str], max_new_tokens: int = 20) -> List[str]:
        """Tokenize, generate, decode — return one string per prompt."""

    # ---- helpers ----

    def auth_hint(self) -> Optional[str]:
        """Return a hint string if authentication is required, else None."""
        return None

    def __repr__(self) -> str:
        loaded = "loaded" if self.model is not None else "not loaded"
        return f"<{self.__class__.__name__} key={self.key!r} ({loaded})>"


# =====================================================================
# CONCRETE: Seq2Seq (Flan-T5 family)
# =====================================================================


class Seq2SeqModelHandler(ModelHandler):
    """Handler for encoder-decoder models (T5, Flan-T5)."""

    def load(self, device: Optional[str] = None) -> "Seq2SeqModelHandler":
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.device = device or get_device()
        print(f"Loading {self.name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hf_name).to(self.device)
        self.model.eval()

        print("Model loaded successfully")
        return self

    def generate(self, prompts: List[str], max_new_tokens: int = 20) -> List[str]:
        formatted = self.format_prompts(prompts)
        inputs = self.tokenizer(
            formatted, padding=True, return_tensors="pt",
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# =====================================================================
# CONCRETE: Causal LM (base decoder-only models)
# =====================================================================


class CausalModelHandler(ModelHandler):
    """Handler for base causal language models (Pythia, Llama base)."""

    def load(self, device: Optional[str] = None) -> "CausalModelHandler":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device or get_device()
        print(f"Loading {self.name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.config.padding_side

        if self.config.use_fp16:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_name, torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_name).to(self.device)

        self.model.eval()
        print("Model loaded successfully")
        return self

    def generate(self, prompts: List[str], max_new_tokens: int = 20) -> List[str]:
        formatted = self.format_prompts(prompts)
        inputs = self.tokenizer(
            formatted, padding=True, return_tensors="pt",
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_lengths = [len(ids) for ids in inputs["input_ids"]]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for i, output in enumerate(outputs):
            new_tokens = output[input_lengths[i]:]
            responses.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return responses


# =====================================================================
# CONCRETE: Instruct Causal LM (chat-template models)
# =====================================================================


class InstructCausalModelHandler(CausalModelHandler):
    """Handler for instruction-tuned causal LMs that use a chat template.

    Inherits loading and generation from ``CausalModelHandler``.
    Overrides ``format_prompts`` to wrap raw text in the tokenizer's
    chat template when one is available.
    """

    def format_prompts(self, prompts: List[str]) -> List[str]:
        if self.tokenizer is None:
            return prompts

        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompts
        if self.tokenizer.chat_template is None:
            return prompts

        formatted = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                formatted.append(text)
            except Exception:
                formatted.append(prompt)
        return formatted

    def auth_hint(self) -> Optional[str]:
        if "llama" in self.hf_name.lower() or "meta-llama" in self.hf_name.lower():
            return "Llama is a gated model. Run: huggingface-cli login"
        return None


# =====================================================================
# REGISTRY
# =====================================================================

# Maps model key -> (handler_class, ModelConfig)
_REGISTRY: Dict[str, tuple[type[ModelHandler], ModelConfig]] = {}


def register_model(
    key: str,
    handler_class: type[ModelHandler],
    config: Optional[ModelConfig] = None,
) -> None:
    """Register a model key with its handler class and config."""
    if config is None:
        config = MODEL_CONFIGS[key]
    _REGISTRY[key] = (handler_class, config)


def get_model_handler(key: str) -> ModelHandler:
    """Instantiate and return the handler registered under *key*."""
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {key!r}. Registered: {available}")
    handler_cls, config = _REGISTRY[key]
    return handler_cls(key=key, config=config)


def list_registered_models() -> List[str]:
    """Return the keys of all registered model handlers."""
    return sorted(_REGISTRY.keys())


# =====================================================================
# REGISTER ALL BUILT-IN MODELS
# =====================================================================

register_model("flan-t5-base", Seq2SeqModelHandler)
register_model("flan-t5-large", Seq2SeqModelHandler)
register_model("pythia-410m", CausalModelHandler)
register_model("llama-3.2-1b", CausalModelHandler)
register_model("llama-3.2-1b-instruct", InstructCausalModelHandler)
register_model("phi-3-mini", InstructCausalModelHandler)
