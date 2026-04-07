#!/usr/bin/env python3
"""
Centralized experiment configuration.

All tuneable knobs live here so that run_experiment.py, perturbations.py,
and prompts.py can share a single source of truth.  CLI arguments in
run_experiment.py override the defaults defined below.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

# Three seeds for statistical significance across all experiments.
EXPERIMENT_SEEDS: Tuple[int, ...] = (105, 2266, 86379)


@dataclass
class ExperimentConfig:
    """Full configuration for a sensitivity experiment run."""

    # ---- Model ----
    model_key: str = "flan-t5-base"

    # ---- Dataset ----
    dataset_key: str = "qasc"
    sample_size: int = 500
    inject_facts: bool = False  # QASC: False avoids ceiling effect (~63% vs ~99%)

    # ---- Perturbations ----
    num_perturbations: int = 10
    words_to_replace: int = 1
    perturbation_method: str = "synonym"  # "synonym" or "paraphrase"

    # ---- Reproducibility ----
    seed: int = 2266

    # ---- Prompt styles to evaluate ----
    prompt_styles: Tuple[str, ...] = ("control", "metacognition", "structure", "politeness")

    # ---- Output ----
    output_dir: str = "./outputs/results"
    phase: str = "phase_1"

    # ---- Out-Of-The-Box accuracy check ----
    ootb_size: int = 100
    skip_ootb: bool = False

    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        return (
            f"phase={self.phase}  model={self.model_key}  dataset={self.dataset_key}  "
            f"samples={self.sample_size}  perturbations={self.num_perturbations}  "
            f"method={self.perturbation_method}  "
            f"seed={self.seed}  styles={','.join(self.prompt_styles)}"
        )


# ---- Default config singleton (importable) ----

DEFAULT_CONFIG = ExperimentConfig()
