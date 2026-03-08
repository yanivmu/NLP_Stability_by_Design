# Stability by Design: Prompt Sensitivity Analysis in Small Language Models

An NLP research project investigating how **prompt design choices** affect the **stability** of Small Language Models (SLMs) when inputs are slightly rephrased. Based on the sensitivity framework from [Lu et al., 2024](https://arxiv.org/abs/2311.07230) (*"How are Prompts Different in Terms of Sensitivity?"*, NAACL 2024).

## Research Question

> Do certain prompt properties (metacognition, structure, politeness) make language models more stable when inputs undergo semantic-preserving perturbations?

## Background

The reference paper shows that **sensitivity** (measured via variation ratio) is an unsupervised proxy for model performance and that different prompt designs lead to different sensitivity levels. We adapt this framework to small, locally-runnable models and introduce a cleaner perturbation pipeline based on NLTK WordNet, avoiding the noisy automated perturbations from the original paper (see Appendix A.1 of the reference).

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Variation Ratio (VR)** | `VR = 1 - f_modal / (N + 1)` where `f_modal` is the frequency of the most common prediction across the original + N perturbations. VR = 0 means perfectly stable, VR = 1 means maximally unstable. |
| **Semantic-preserving perturbations** | Input variations that change surface form but not meaning (synonym replacement, paraphrasing). |
| **OOTB accuracy** | Out-Of-The-Box accuracy on unperturbed data, checked before sensitivity experiments to verify the model can perform the task. |
| **Prompt styles** | Four framing strategies tested: Control (baseline), Metacognition (self-check), Structure (JSON output), Politeness (courteous phrasing). |

## Models

| Model | Architecture | Parameters | HuggingFace ID |
|-------|-------------|------------|----------------|
| Flan-T5-Base | Encoder-Decoder (T5) | 248M | `google/flan-t5-base` |
| Flan-T5-Large | Encoder-Decoder (T5) | 783M | `google/flan-t5-large` |
| Pythia-410M | Decoder-only (GPT-NeoX) | 410M | `EleutherAI/pythia-410m` |
| Llama-3.2-1B | Decoder-only (Llama) | 1.24B | `meta-llama/Llama-3.2-1B` |

## Datasets

| Dataset | Task | Labels | Random Baseline |
|---------|------|--------|-----------------|
| [QASC](https://huggingface.co/datasets/allenai/qasc) | 8-way multiple-choice science QA | A-H | 12.5% |
| [CoLA](https://huggingface.co/datasets/nyu-mll/glue) | Binary grammaticality judgment | acceptable / unacceptable | 50% |

## Prompt Styles

| Style | Description | Example Addition |
|-------|-------------|------------------|
| **Control** | Standard zero-shot instruction (baseline) | `"Answer with just the letter:"` |
| **Metacognition** | Self-check triggers for the model | `"Think carefully... Verify your reasoning before answering."` |
| **Structure** | Enforces JSON output format | `"Respond ONLY with valid JSON: {\"answer\": \"...\"}"` |
| **Politeness** | Conversational, courteous phrasing | `"I'd really appreciate your help... Please provide... Thank you!"` |

## Perturbation Methods

### 1. Synonym Replacement (default)

Uses NLTK WordNet to replace 1-2 content words with semantically equivalent synonyms:

1. **POS-tag** the input text (nouns, verbs, adjectives, adverbs)
2. **Filter** out stopwords, question words, answer labels, short words
3. **Look up** WordNet synonyms (top 3 senses only, for quality)
4. **Replace** randomly chosen words with random synonyms
5. **Fallback** to article swaps, filler prefixes, or punctuation variation if WordNet cannot produce enough unique variants

### 2. Paraphrase Generation

Uses a separate small model (`google/flan-t5-small`) to rewrite inputs:

1. Cycle through 10 diverse prompt templates (e.g., "Rewrite this without changing its meaning")
2. Generate with `do_sample=True`, `temperature=0.8`, `top_p=0.9`
3. Deduplicate results; fall back to synonym replacement for any shortfall

## Project Structure

```text
NLP_Stability_by_Design/
├── outputs/
│   └── results/                     # Experiment output JSONs
│       ├── flan-t5-base/            # Results per model
│       ├── flan-t5-large/
│       ├── pythia-410m/
│       └── llama-3.2-1b/
├── reference_paper_code/            # Original paper's code (read-only reference)
├── src/                             # Source code
│   ├── run_experiment.py            # Main CLI entry point
│   ├── config.py                    # ExperimentConfig dataclass
│   ├── perturbations.py             # Synonym + paraphrase perturbation generation
│   ├── prompts.py                   # Four prompt style templates (dataset-agnostic)
│   ├── models.py                    # Model configs, loading, inference
│   ├── datasets_config.py           # Dataset configs, loading, formatting
│   ├── data_analysis.py             # DataManager, ResultAnalyzer, parsing, VR math
│   ├── data_demo.py                 # Standalone demo of the data pipeline
│   └── interface.py                 # Architecture overview and import hub
├── requirements.txt                 # Python dependencies
└── README.md
```

## File Descriptions

| File | Owner | Purpose |
|------|-------|---------|
| `run_experiment.py` | All | Unified CLI runner: loads model, loads dataset, runs OOTB check, generates perturbations, evaluates all prompt styles, saves JSON results |
| `config.py` | TM3 | `ExperimentConfig` dataclass with all tuneable parameters (model, dataset, sample size, perturbation count, seed, etc.) |
| `perturbations.py` | TM3 | WordNet synonym replacement + Flan-T5-Small paraphrase generation, seed management (`set_all_seeds`), validation |
| `prompts.py` | TM3 | Generic and dataset-specific prompt templates for all four styles; extensible via `register_dataset_prompts()` |
| `models.py` | TM1 | `ModelConfig` definitions, `load_model_and_tokenizer()`, `run_inference()`, device detection |
| `datasets_config.py` | TM2 | `DatasetConfig` definitions, HuggingFace dataset loading, item formatting, answer extraction |
| `data_analysis.py` | TM2 | `DataManager` (data loading/sampling), `ResultAnalyzer` (response parsing, VR calculation) |
| `data_demo.py` | TM2 | Interactive demo showing raw data, prompt formatting, model responses |
| `interface.py` | All | Architecture overview, re-exports from all modules, quick-start example |

## Installation

### Prerequisites

- Python 3.9+
- ~5 GB disk space for model weights

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `torch>=2.0.0` -- PyTorch for model inference
- `transformers>=4.30.0` -- HuggingFace model loading and generation
- `datasets>=2.14.0` -- HuggingFace dataset loading
- `huggingface_hub>=0.16.0` -- Model downloading and authentication
- `nltk>=3.8.0` -- WordNet synonym lookup and POS tagging

NLTK data (`wordnet`, `punkt_tab`, `averaged_perceptron_tagger_eng`, `stopwords`, `omw-1.4`) is downloaded automatically on first run.

### Llama Access (required for Llama-3.2-1B only)

Llama is a gated model. To use it:

1. Visit [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and accept Meta's license
2. Create a HuggingFace token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) with "Access to public gated repos" enabled
3. Authenticate locally:

```bash
huggingface-cli login
```

## Usage

All experiments are run from the `src/` directory.

### Basic Experiment

```bash
cd src
python run_experiment.py --model flan-t5-base --dataset qasc
```

### Synonym Mode (default)

Replaces content words with WordNet synonyms. Fast, no extra model needed.

```bash
# Explicit synonym mode (same as default)
python run_experiment.py --model flan-t5-base --dataset qasc --perturbation-method synonym

# Shorthand (synonym is the default, so --perturbation-method can be omitted)
python run_experiment.py --model flan-t5-base --dataset qasc
```

### Paraphrase Mode

Uses a separate small model (`google/flan-t5-small`) to rewrite inputs. Produces more natural, diverse perturbations but is slower (loads an additional model).

```bash
python run_experiment.py --model flan-t5-base --dataset qasc --perturbation-method paraphrase
python run_experiment.py --model flan-t5-large --dataset cola --perturbation-method paraphrase
```

### More Examples

```bash
# 100 samples, 10 perturbations, synonym mode
python run_experiment.py --model flan-t5-base --dataset qasc --sample-size 100 --num-perturbations 10

# Paraphrase mode with custom seed
python run_experiment.py --model pythia-410m --dataset cola --perturbation-method paraphrase --seed 42

# Llama on QASC
python run_experiment.py --model llama-3.2-1b --dataset qasc
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model key: `flan-t5-base`, `flan-t5-large`, `pythia-410m`, `llama-3.2-1b` | *required* |
| `--dataset` | Dataset key: `qasc`, `cola` | *required* |
| `--sample-size` | Number of samples for sensitivity measurement | 30 |
| `--num-perturbations` | Number of perturbations per sample (N) | 10 |
| `--words-to-replace` | Words to replace per synonym perturbation | 1 |
| `--perturbation-method` | `synonym` or `paraphrase` | `synonym` |
| `--ootb-size` | Samples for OOTB accuracy check | 100 |
| `--seed` | Random seed for reproducibility | 2266 |
| `--output-dir` | Output directory | `../outputs/results` |
| `--skip-ootb` | Skip the OOTB accuracy check | false |

### Output

Each experiment produces a JSON file in `outputs/results/<model-key>/`:

```
outputs/results/flan-t5-base/sensitivity_results_flan-t5-base_qasc.json
```

The JSON contains OOTB accuracy, experiment parameters, and per-style results (VR, accuracy, sample count).

## Reproducibility

Seeds are fixed across all libraries via `set_all_seeds()` in `perturbations.py`:

- `random.seed()`
- `numpy.random.seed()`
- `torch.manual_seed()` and `torch.cuda.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`
- `os.environ["PYTHONHASHSEED"]`

The paraphrase generator additionally re-seeds torch before every individual generation call with `base_seed + attempt_index` to ensure deterministic output even with `do_sample=True`.

Default seed: **2266** (from the reference paper).

## Methodology

1. **Fix seeds** across all libraries for reproducibility
2. **Load model & dataset** from HuggingFace
3. **OOTB accuracy check** on unperturbed data (verifies model can perform the task)
4. **For each prompt style** (Control, Metacognition, Structure, Politeness):
   - For each sample in the dataset:
     - Format the question with the prompt style
     - Get the model's answer on the original input
     - Generate N semantic-preserving perturbations
     - Get the model's answer on each perturbation
     - Compute the Variation Ratio across all N+1 predictions
   - Report average VR and accuracy across all samples
5. **Save results** as JSON

## Experiment Results

All experiments: 100 samples, 10 perturbations per sample, synonym mode, seed=2266.

### Flan-T5-Base

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| QASC | 99% | Control | 0.026 | 99% |
| | | Metacognition | 0.030 | 96% |
| | | Structure | 0.031 | 96% |
| | | Politeness | 0.028 | 98% |
| CoLA | 29% | Control | 0.000 | 29% |
| | | Metacognition | 0.000 | 29% |
| | | Structure | 0.116 | 46% |
| | | Politeness | 0.000 | 29% |

### Flan-T5-Large

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| QASC | 100% | Control | 0.012 | 100% |
| | | Metacognition | 0.013 | 100% |
| | | Structure | 0.006 | 100% |
| | | Politeness | 0.012 | 100% |
| CoLA | 64% | Control | 0.124 | 64% |
| | | Metacognition | 0.134 | 63% |
| | | Structure | 0.107 | 66% |
| | | Politeness | 0.132 | 71% |

### Pythia-410M

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| QASC | 3% | Control | 0.000 | 11% |
| | | Metacognition | 0.000 | 13% |
| | | Structure | 0.000 | 13% |
| | | Politeness | 0.000 | 13% |
| CoLA | 77% | Control | 0.000 | 6% |
| | | Metacognition | 0.000 | 1% |
| | | Structure | 0.000 | 5% |
| | | Politeness | 0.000 | 6% |

### Llama-3.2-1B

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| QASC | 1% | Control | 0.000 | 6% |
| | | Metacognition | 0.000 | 0% |
| | | Structure | 0.000 | 0% |
| | | Politeness | 0.000 | 0% |
| CoLA | 77% | Control | 0.000 | 6% |
| | | Metacognition | 0.000 | 6% |
| | | Structure | 0.000 | 0% |
| | | Politeness | 0.000 | 6% |

### Interpretation

- **Flan-T5 models** perform well on QASC (instruction-tuned, encoder-decoder architecture suits QA). Sensitivity is low across all styles, with Structure prompts showing the lowest VR on Flan-T5-Large.
- **Flan-T5-Large on CoLA** shows meaningful sensitivity differences between prompt styles. Politeness achieves the highest accuracy (71%) while Structure achieves the lowest VR (0.107).
- **Pythia and Llama** produce near-random accuracy, and VR = 0.000 reflects consistent (but wrong) answers rather than true stability. These models lack the instruction-tuning needed for zero-shot QA and grammaticality tasks. Sensitivity analysis requires models that can actually perform the task to be meaningful.

## Reference Paper

Lu, S., Schuff, H., & Gurevych, I. (2024). *How are Prompts Different in Terms of Sensitivity?* NAACL 2024. [arXiv:2311.07230](https://arxiv.org/abs/2311.07230)

## Team

- **Team Member 1** -- DevOps & Infrastructure: model loading, device management, inference pipeline
- **Team Member 2** -- Data & Parsing: dataset configs, response parsing, VR calculation
- **Team Member 3** -- Prompt Engineering: prompt templates, perturbation generation, configuration, reproducibility

## License

Academic research project.
