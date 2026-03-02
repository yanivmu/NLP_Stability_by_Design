# Stability by Design in Small Language Models (SLMs)

An NLP research project investigating how **Small Language Models** respond to **semantic-preserving perturbations** in prompts, and whether certain prompt properties (metacognition, structure, politeness) improve model stability and accuracy.

## Research Question

> Do certain prompt design choices make language models more stable when inputs are slightly rephrased?

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Variation Ratio (VR)** | Sensitivity metric: `VR = 1 - (f_modal / N_total)`. VR ≈ 0 means stable, VR ≈ 1 means unstable |
| **Semantic-preserving perturbations** | Input variations that don't change meaning (e.g., adding "basically", synonym swaps) |
| **OOTB Check** | Out-Of-The-Box accuracy validation before measuring stability |
| **Prompt Properties** | Different framing strategies: Control, Metacognition, Structure, Politeness |

## Models & Datasets

| Model | Dataset | Task Type |
|-------|---------|-----------|
| [Flan-T5-Base](https://huggingface.co/google/flan-t5-base) | [QASC](https://huggingface.co/datasets/allenai/qasc) | 8-way multiple choice (science QA) |
| [Pythia-410M](https://huggingface.co/EleutherAI/pythia-410m) | [CoLA](https://huggingface.co/datasets/nyu-mll/glue) | Binary classification (grammaticality) |

## Prompt Styles Tested

| Style | Description |
|-------|-------------|
| **Control** | Standard zero-shot instruction (baseline) |
| **Metacognition** | Adds self-check triggers ("verify your reasoning") |
| **Structure** | Enforces JSON output format |
| **Politeness** | Adds conversational fillers ("please", "thank you") |

## Project Structure

```
├── interface.py                    # Project blueprint/interfaces
├── data_analysis.py                # Core module: data loading, parsing, metrics
├── run_flan_qasc_experiment.py     # Experiment: Flan-T5 on QASC
├── run_pythia_cola_experiment.py   # Experiment: Pythia on CoLA
├── data_demo.py                    # Demo script showing the pipeline
├── sensitivity_results_*.json      # Saved experiment results
└── sensetivity_article_project/    # Reference implementation (not part of main project)
```

## File Descriptions

| File | Purpose |
|------|---------|
| `data_analysis.py` | Contains `DataManager` (dataset loading, OOTB checks), `ResultAnalyzer` (parsing, VR calculation), and `generate_perturbations()` |
| `run_flan_qasc_experiment.py` | Runs full sensitivity experiment with Flan-T5-Base on QASC dataset |
| `run_pythia_cola_experiment.py` | Runs full sensitivity experiment with Pythia-410M on CoLA dataset |
| `data_demo.py` | Interactive demo showing raw data, prompts, model responses, and VR calculation |
| `interface.py` | Abstract interfaces defining the project architecture |

## Results

### Flan-T5-Base on QASC
- **OOTB Accuracy**: 99%

| Prompt Style | Variation Ratio | Accuracy | Stability |
|--------------|-----------------|----------|-----------|
| Control | 0.0000 | 100% | Stable |
| Politeness | 0.0030 | 100% | Stable |
| Structure | 0.0061 | 93.3% | Stable |
| Metacognition | 0.0174 | 93.3% | Stable |

### Pythia-410M on CoLA
- **OOTB Accuracy**: 68.1%

| Prompt Style | Variation Ratio | Accuracy | Stability |
|--------------|-----------------|----------|-----------|
| Structure | 0.0000 | 63.3% | Stable |
| Metacognition | 0.0030 | 6.7% | Stable |
| Control | 0.0152 | 80.0% | Stable |
| Politeness | 0.0394 | 83.3% | Stable |

## Installation

```bash
pip install torch transformers datasets
```

## Usage

### Run Demo (see the pipeline in action)
```bash
python data_demo.py
```

### Run Experiments
```bash
# Flan-T5 on QASC
python run_flan_qasc_experiment.py

# Pythia on CoLA
python run_pythia_cola_experiment.py
```

## Methodology

1. **Load Model & Dataset** from HuggingFace
2. **OOTB Accuracy Check** - Verify model performs above random baseline
3. **Generate Perturbations** - Create N=10 semantic-preserving variants per sample
4. **Run Inference** - Get model responses for original + perturbations
5. **Calculate Variation Ratio** - Measure answer consistency across perturbations
6. **Compare Prompt Styles** - Identify which prompt properties improve stability

## Team Structure

- **Team Member 1**: DevOps, Infrastructure, Model Loading
- **Team Member 2**: Data Management, Parsing, Metric Calculation
- **Team Member 3**: Prompt Engineering, Perturbation Generation

## License

This project is for academic research purposes.

