"""
NLP Final Project: Stability by Design in SLMs
Team Members: 3
Target Completion: March 15th

Architecture Overview
---------------------
This file maps the original team roles to the actual implementation modules.
It can be used as a quick reference or as an import hub.

Module Map:
    interface.py        ->  You are here (orchestrator / reference)
    models.py           ->  Team Member 1: model loading, inference, device management
    datasets_config.py  ->  Team Member 2: dataset configs, loading, formatting
    data_analysis.py    ->  Team Member 2: parsing (ResultAnalyzer), OOTB baselines
    prompts.py          ->  Team Member 3: four prompt styles per dataset
    perturbations.py    ->  Team Member 3: synonym & paraphrase perturbation generation
    config.py           ->  Team Member 3: ExperimentConfig dataclass, seed management
    run_experiment.py   ->  Main CLI entry point (collaborative)
"""

from typing import List, Dict, Any

# =====================================================================
# TEAM MEMBER 1: DevOps, Infrastructure & Models  (models.py)
# =====================================================================
from models import (
    ModelConfig,
    ModelType,
    get_model_config,
    load_model_and_tokenizer,
    run_inference,
    get_device,
)

# =====================================================================
# TEAM MEMBER 2: Data, Parsing & Analysis  (datasets_config.py + data_analysis.py)
# =====================================================================
from datasets_config import (
    DatasetConfig,
    AnswerType,
    get_dataset_config,
    load_dataset,
    format_qasc_base,
    get_item_text,
    get_correct_answer,
    convert_to_list,
)
from data_analysis import DataManager, ResultAnalyzer

# =====================================================================
# TEAM MEMBER 3: Prompts & Perturbations  (prompts.py + perturbations.py + config.py)
# =====================================================================
from config import ExperimentConfig
from prompts import get_prompt_styles, get_max_tokens, register_dataset_prompts
from perturbations import (
    set_all_seeds,
    generate_perturbations,
    generate_and_validate,
    generate_paraphrase_perturbations,
)


def main():
    """
    Quick-start example showing the full pipeline.

    For real experiments use the CLI instead:
        python run_experiment.py --model flan-t5-base --dataset qasc
        python run_experiment.py --model flan-t5-base --dataset cola --perturbation-method paraphrase
    """
    cfg = ExperimentConfig(
        model_key="flan-t5-base",
        dataset_key="qasc",
        sample_size=10,
        num_perturbations=5,
        seed=2266,
    )

    # 1. Reproducibility
    set_all_seeds(cfg.seed)

    # 2. Load model  (Team Member 1)
    model_config = get_model_config(cfg.model_key)
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_config, device)

    # 3. Load dataset  (Team Member 2)
    dataset_config = get_dataset_config(cfg.dataset_key)
    dataset = load_dataset(dataset_config)

    # 4. Get prompts  (Team Member 3)
    prompt_styles = get_prompt_styles(cfg.dataset_key)

    # 5. Generate perturbations  (Team Member 3)
    sample_item = dataset[0]
    text = get_item_text(sample_item, cfg.dataset_key)
    perturbations = generate_and_validate(text, num=cfg.num_perturbations)

    print(f"Model:         {model_config.name}")
    print(f"Dataset:       {dataset_config.name} ({len(dataset)} items)")
    print(f"Prompt styles: {list(prompt_styles.keys())}")
    print(f"Perturbations: {len(perturbations)} generated for first item")
    print("\nUse run_experiment.py for full experiments.")


if __name__ == "__main__":
    main()
