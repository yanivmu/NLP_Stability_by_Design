#!/usr/bin/env python3
"""
Model configurations for different LLM architectures.

Supports:
- Flan-T5 Base/Large (encoder-decoder seq2seq)
- Pythia-410M (decoder-only causal LM)
- Llama-3.2-1B / Llama-3.2-1B-Instruct (decoder-only causal LM)
- Phi-3-Mini-4K-Instruct (decoder-only causal LM)
"""

import torch
from typing import Dict
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Supported model architectures."""
    SEQ2SEQ = "seq2seq"  # Encoder-decoder (T5, Flan-T5)
    CAUSAL = "causal"    # Decoder-only (GPT, Pythia, Llama)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str                    # Display name
    hf_name: str                 # HuggingFace model identifier
    model_type: ModelType        # Architecture type
    use_fp16: bool = False       # Use half precision
    padding_side: str = "right"  # Padding side for tokenizer
    default_max_tokens: int = 20 # Default max new tokens


# Supported model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "flan-t5-base": ModelConfig(
        name="Flan-T5-Base",
        hf_name="google/flan-t5-base",
        model_type=ModelType.SEQ2SEQ,
        padding_side="right",
        default_max_tokens=20,
    ),
    "flan-t5-large": ModelConfig(
        name="Flan-T5-Large",
        hf_name="google/flan-t5-large",
        model_type=ModelType.SEQ2SEQ,
        padding_side="right",
        default_max_tokens=20,
    ),
    "pythia-410m": ModelConfig(
        name="Pythia-410M",
        hf_name="EleutherAI/pythia-410m",
        model_type=ModelType.CAUSAL,
        padding_side="left",
        default_max_tokens=10,
    ),
    "llama-3.2-1b": ModelConfig(
        name="Llama-3.2-1B",
        hf_name="meta-llama/Llama-3.2-1B",
        model_type=ModelType.CAUSAL,
        use_fp16=True,
        padding_side="left",
        default_max_tokens=15,
    ),
    "llama-3.2-1b-instruct": ModelConfig(
        name="Llama-3.2-1B-Instruct",
        hf_name="meta-llama/Llama-3.2-1B-Instruct",
        model_type=ModelType.CAUSAL,
        use_fp16=True,
        padding_side="left",
        default_max_tokens=20,
    ),
    "phi-3-mini": ModelConfig(
        name="Phi-3-Mini-4K-Instruct",
        hf_name="microsoft/Phi-3-mini-4k-instruct",
        model_type=ModelType.CAUSAL,
        use_fp16=True,
        padding_side="left",
        default_max_tokens=20,
    ),
}


def get_device() -> str:
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

