"""
NLP Final Project: Stability by Design in SLMs
Team Members: 3
Target Completion: March 15th

Architecture Overview
---------------------
This file maps the original team roles to the actual implementation modules.
It can be used as a quick reference or as an import hub.

Module Map:
    interface.py          ->  You are here (orchestrator / reference)
    model_handlers.py     ->  Model Handler ABC + Registry (Seq2Seq, Causal, Instruct)
    dataset_handlers.py   ->  Dataset Handler ABC + Registry (CoLA, QASC, CSQA, GSM8K)
    models.py             ->  Team Member 1: ModelConfig definitions, device detection (legacy)
    datasets_config.py    ->  Team Member 2: dataset configs (legacy, still importable)
    data_analysis.py      ->  Team Member 2: parsing (ResultAnalyzer), OOTB baselines
    prompts.py            ->  Team Member 3: four prompt styles per dataset (legacy)
    perturbations.py      ->  Team Member 3: synonym & paraphrase perturbation generation
    config.py             ->  Team Member 3: ExperimentConfig dataclass, seed management
    run_experiment.py     ->  Main CLI entry point (fully model- and dataset-agnostic)
"""

from typing import List, Dict, Any

# =====================================================================
# MODEL HANDLER REGISTRY (the primary API for models)
# =====================================================================
from model_handlers import (
    ModelHandler,
    get_model_handler,
    list_registered_models,
    register_model,
    Seq2SeqModelHandler,
    CausalModelHandler,
    InstructCausalModelHandler,
)

# =====================================================================
# DATASET HANDLER REGISTRY (the primary API for datasets)
# =====================================================================
from dataset_handlers import (
    DatasetHandler,
    get_dataset_handler,
    list_registered_datasets,
    register_dataset,
    QASCHandler,
    CoLAHandler,
    CSQAHandler,
    GSM8KHandler,
)

# =====================================================================
# TEAM MEMBER 1: DevOps, Infrastructure & Models  (models.py)
# Kept for backward compatibility — new code should prefer model_handlers.
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
# Kept for backward compatibility — new code should prefer dataset_handlers.
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
from config import ExperimentConfig, EXPERIMENT_SEEDS
from prompts import get_prompt_styles, get_max_tokens, register_dataset_prompts
from perturbations import (
    set_all_seeds,
    generate_perturbations,
    generate_and_validate,
    generate_paraphrase_perturbations,
)


def main():
    """
    Quick-start example showing the new handler-based pipeline.

    For real experiments use the CLI instead:
        python run_experiment.py --model flan-t5-base --dataset qasc
        python run_experiment.py --model llama-3.2-1b-instruct --dataset cola --perturbation-method paraphrase
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

    # 2. Load model via handler registry
    mh = get_model_handler(cfg.model_key)
    mh.load()

    # 3. Load dataset via handler registry
    dh = get_dataset_handler(cfg.dataset_key)
    dataset = dh.load()

    # 4. Demonstrate handler APIs
    items = dh.sample_items(dataset, n=1, seed=cfg.seed)
    sample_item = items[0]
    text = dh.get_item_text(sample_item)
    perturbations = generate_and_validate(text, num=cfg.num_perturbations)

    print(f"Model:              {mh.name}")
    print(f"Dataset:            {dh.CONFIG.name}")
    print(f"Registered models:   {list_registered_models()}")
    print(f"Registered datasets: {list_registered_datasets()}")
    print(f"Prompt styles:      {list(cfg.prompt_styles)}")
    print(f"Perturbations:      {len(perturbations)} generated for first item")
    print(f"Experiment seeds:   {EXPERIMENT_SEEDS}")
    print("\nUse run_experiment.py for full experiments.")


if __name__ == "__main__":
    main()
