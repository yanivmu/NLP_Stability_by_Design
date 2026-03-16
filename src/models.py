#!/usr/bin/env python3
"""
Model configurations and inference logic for different LLM architectures.

Supports:
- Flan-T5 Base/Large (encoder-decoder seq2seq)
- Pythia-410M (decoder-only causal LM)
- Llama-3.2-1B / Llama-3.2-1B-Instruct (decoder-only causal LM)
- Phi-3-Mini-4K-Instruct (decoder-only causal LM)
"""

import torch
from typing import List, Dict, Any
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


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key."""
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return MODEL_CONFIGS[model_key]


def load_model_and_tokenizer(config: ModelConfig, device: str):
    """
    Load model and tokenizer based on configuration.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
    
    print(f"Loading {config.name}...")
    
    if config.model_type == ModelType.SEQ2SEQ:
        tokenizer = AutoTokenizer.from_pretrained(config.hf_name)
        model = T5ForConditionalGeneration.from_pretrained(config.hf_name).to(device)
    else:  # CAUSAL
        tokenizer = AutoTokenizer.from_pretrained(config.hf_name)
        # Set pad token for causal models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = config.padding_side
        
        # Load with optional fp16
        if config.use_fp16:
            model = AutoModelForCausalLM.from_pretrained(
                config.hf_name, torch_dtype=torch.float16
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.hf_name).to(device)
    
    model.eval()
    print(f"Model loaded successfully")
    return model, tokenizer


def run_inference(
    model, tokenizer, prompts: List[str], device: str,
    model_type: ModelType, max_new_tokens: int = 20
) -> List[str]:
    """
    Run inference on a batch of prompts.
    
    Handles both seq2seq and causal models appropriately.
    """
    inputs = tokenizer(prompts, padding=True, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if model_type == ModelType.SEQ2SEQ:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:  # CAUSAL
            input_lengths = [len(ids) for ids in inputs['input_ids']]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Extract only new tokens for causal models
            responses = []
            for i, output in enumerate(outputs):
                new_tokens = output[input_lengths[i]:]
                responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
            return responses


def get_device() -> str:
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

