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
| [Flan-T5-Large](https://huggingface.co/google/flan-t5-large) | [CoLA](https://huggingface.co/datasets/nyu-mll/glue) | Binary classification (grammaticality) |
| [Pythia-410M](https://huggingface.co/EleutherAI/pythia-410m) | [CoLA](https://huggingface.co/datasets/nyu-mll/glue) | Binary classification (grammaticality) |
| [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | [QASC](https://huggingface.co/datasets/allenai/qasc) | 8-way multiple choice (science QA) |

## Prompt Styles Tested

| Style | Description |
|-------|-------------|
| **Control** | Standard zero-shot instruction (baseline) |
| **Metacognition** | Adds self-check triggers ("verify your reasoning") |
| **Structure** | Enforces JSON output format |
| **Politeness** | Adds conversational fillers ("please", "thank you") |

## Project Structure

```text
NLP_Project_Sensitivity/
│
├── outputs/                       # Generated results
│   ├── figures/                   # Generated graphs and plots
│   ├── logs/                      # Slurm .out and .err files
│   └── results/                   # Saved sensitivity_results_*.json files
│       ├── flan/                  # Results for Flan-T5 models
│       ├── llama/                 # Results for LLaMA models
│       └── pythia/                # Results for Pythia models
│
├── reference_paper_code/          # Reference implementation (not part of main execution)
│   └── sensetivity_article_project/
│       ├── datasets/              # Original paper datasets
│       ├── figures/               # Original paper figures
│       ├── prompts/               # Original paper prompts (Control, CoT, etc.)
│       └── ...                    # Original codebase (.py, .ipynb, .csv)
│
├── slurm_scripts/                 # Slurm batch scripts for TAU cluster execution
│
├── src/                           # Core source code modules
│   ├── data_analysis.py           # Data loading, parsing, OOTB checks, VR calculation
│   ├── data_demo.py               # Interactive demo showing the pipeline
│   ├── datasets_config.py         # Dataset configurations (QASC, CoLA)
│   ├── interface.py               # Abstract interfaces defining the project architecture
│   ├── models.py                  # Model configurations (Flan-T5, Pythia, Llama)
│   ├── prompts.py                 # Prompt templates for all styles
│   └── run_experiment.py          # Unified experiment runner for all models/datasets
│
└── README.md                      # Project documentation
```

## File Descriptions

| File | Purpose |
|------|---------|
| `run_experiment.py` | Unified experiment runner - runs any model/dataset combination via CLI arguments |
| `models.py` | Model configurations: HuggingFace names, architecture types, loading logic |
| `datasets_config.py` | Dataset configurations: loading, formatting, thresholds |
| `prompts.py` | Prompt templates for all 4 styles (Control, Metacognition, Structure, Politeness) |
| `data_analysis.py` | Contains `DataManager`, `ResultAnalyzer` (parsing, VR calculation), and `generate_perturbations()` |
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
pip install -r requirements.txt
```

Or install dependencies individually:
```bash
pip install torch transformers datasets huggingface_hub
```

**Note:** For Llama models, you need to authenticate with HuggingFace:
```bash
huggingface-cli login
```

## Usage

### Run Demo (see the pipeline in action)
```bash
cd src
python data_demo.py
```

### Run Experiments

Use the unified `run_experiment.py` script with command-line arguments:

```bash
cd src

# Flan-T5-Base on QASC
python run_experiment.py --model flan-t5-base --dataset qasc

# Flan-T5-Large on QASC
python run_experiment.py --model flan-t5-large --dataset qasc

# Pythia-410M on CoLA
python run_experiment.py --model pythia-410m --dataset cola

# Llama-3.2-1B on QASC
python run_experiment.py --model llama-3.2-1b --dataset qasc

# Llama-3.2-1B on CoLA
python run_experiment.py --model llama-3.2-1b --dataset cola
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model to use: `flan-t5-base`, `flan-t5-large`, `pythia-410m`, `llama-3.2-1b` | Required |
| `--dataset` | Dataset to use: `qasc`, `cola` | Required |
| `--sample-size` | Number of samples for sensitivity experiments | 30 |
| `--ootb-size` | Number of samples for OOTB accuracy check | 100 |
| `--seed` | Random seed for reproducibility | 2266 |
| `--output-dir` | Output directory for results | `outputs/results` |

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

