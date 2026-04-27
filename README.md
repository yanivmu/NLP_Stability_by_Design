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
| Llama-3.2-1B-Instruct | Decoder-only (Llama, Instruct) | 1.24B | `meta-llama/Llama-3.2-1B-Instruct` |
| Phi-3-Mini-4K-Instruct | Decoder-only (Phi-3) | 3.8B | `microsoft/Phi-3-mini-4k-instruct` |

Instruct models (Llama-Instruct, Phi-3-Mini) automatically use chat templates via the model handler registry.

## Datasets

| Dataset | Task | Labels | Random Baseline | Status |
|---------|------|--------|-----------------|--------|
| [QASC](https://huggingface.co/datasets/allenai/qasc) | 8-way multiple-choice science QA (no facts) | A-H | 12.5% | Active |
| [CoLA](https://huggingface.co/datasets/nyu-mll/glue) | Binary grammaticality judgment | acceptable / unacceptable | 50% | Active |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | 5-way multiple-choice commonsense | A-E | 20% | Active |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Grade-school math, free-form numeric | numeric | 0% | Registered (stub) |

New datasets are added by implementing a `DatasetHandler` subclass — zero changes to the pipeline code.

**QASC fact injection:** Disabled by default to avoid a ceiling effect (~99% → ~63% accuracy on Flan-T5-Large). Use `--facts` to re-enable for comparison.

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
│   ├── results/                     # Experiment outputs (JSON + CSV)
│   │   ├── phase_1/ … phase_6/     # Results organized by phase
│   │   │   ├── flan-t5-base/       # Per-model directories
│   │   │   │   └── cola/qasc/csqa/ # Per-dataset, contains summary + detail CSVs
│   │   │   ├── flan-t5-large/
│   │   │   ├── pythia-410m/
│   │   │   ├── llama-3.2-1b/
│   │   │   ├── llama-3.2-1b-instruct/
│   │   │   └── phi-3-mini/
│   │   └── qasc_no_facts/          # QASC no-facts baseline eval
│   ├── figures/                     # Generated plots (by phase)
│   └── logs/                        # Slurm job output logs
├── reference_paper_code/            # Original paper's code (read-only reference)
├── scripts/slurm/                   # Slurm job scripts (by phase/model/seed)
├── src/                             # Source code
│   ├── run_experiment.py            # Main CLI entry point (model- and dataset-agnostic)
│   ├── model_handlers.py            # Model Handler ABC + Registry (Seq2Seq, Causal, Instruct)
│   ├── dataset_handlers.py          # Dataset Handler ABC + Registry (CoLA, QASC, CSQA, GSM8K)
│   ├── config.py                    # ExperimentConfig dataclass, experiment seeds
│   ├── perturbations.py             # Synonym + paraphrase perturbation generation
│   ├── visualize_results.py         # Control-centric dual-axis plots (accuracy + VR)
│   ├── eval_qasc_no_facts.py        # Isolated QASC-without-facts evaluation script
│   ├── models.py                    # ModelConfig definitions, device detection (legacy)
│   ├── prompts.py                   # Four prompt style templates + max_tokens per style
│   ├── datasets_config.py           # DatasetConfig definitions (legacy)
│   ├── data_analysis.py             # DataManager, ResultAnalyzer, parsing, VR math
│   ├── data_demo.py                 # Standalone demo of the data pipeline
│   └── interface.py                 # Architecture overview and import hub
├── requirements.txt                 # Python dependencies
└── README.md
```

## File Descriptions

| File | Owner | Purpose |
|------|-------|---------|
| `run_experiment.py` | All | Unified CLI runner — fully model- and dataset-agnostic via handler registries. Produces JSON + detailed per-response CSV. OOTB denominator uses total items (not just parsed). |
| `model_handlers.py` | TM1 | `ModelHandler` ABC + concrete handlers (`Seq2SeqModelHandler`, `CausalModelHandler`, `InstructCausalModelHandler`) + registry |
| `dataset_handlers.py` | TM2 | `DatasetHandler` ABC + concrete handlers (`QASCHandler`, `CoLAHandler`, `CSQAHandler`, `GSM8KHandler`) + registry. Includes multi-pass yes/no parser and letter parser with markdown fence stripping. |
| `config.py` | TM3 | `ExperimentConfig` dataclass with `sensitivity_on_raw` flag, `EXPERIMENT_SEEDS` (3 seeds for statistical significance) |
| `perturbations.py` | TM3 | WordNet synonym replacement + Flan-T5-Small paraphrase generation, seed management (`set_all_seeds`), validation |
| `visualize_results.py` | All | Aggregates phase CSVs and generates control-centric dual-axis plots (accuracy + VR) per model/dataset combination |
| `eval_qasc_no_facts.py` | All | Isolated evaluation of Flan-T5-Large on QASC without fact injection (ceiling-effect investigation) |
| `models.py` | TM1 | `ModelConfig` definitions, `load_model_and_tokenizer()`, `run_inference()`, device detection (legacy, still importable) |
| `prompts.py` | TM3 | Generic and dataset-specific prompt templates for all four styles. Includes per-dataset per-style `max_tokens` settings. |
| `datasets_config.py` | TM2 | `DatasetConfig` definitions, HuggingFace dataset loading (legacy, still importable) |
| `data_analysis.py` | TM2 | `DataManager` (data loading/sampling), `ResultAnalyzer` (response parsing, VR calculation with raw/parsed modes) |
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

### Llama Access (required for Llama models)

Llama models are gated. To use `llama-3.2-1b` or `llama-3.2-1b-instruct`:

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

# Llama Instruct on QASC (chat template applied automatically)
python run_experiment.py --model llama-3.2-1b-instruct --dataset qasc

# Phi-3-Mini on CoLA
python run_experiment.py --model phi-3-mini --dataset cola

# QASC with fact injection (re-enables the old behavior for comparison)
python run_experiment.py --model flan-t5-large --dataset qasc --facts
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model key: `flan-t5-base`, `flan-t5-large`, `pythia-410m`, `llama-3.2-1b`, `llama-3.2-1b-instruct`, `phi-3-mini` | *required* |
| `--dataset` | Dataset key: `qasc`, `cola`, `csqa` (also registered: `gsm8k`) | *required* |
| `--sample-size` | Number of samples for sensitivity measurement | 500 |
| `--num-perturbations` | Number of perturbations per sample (N) | 10 |
| `--words-to-replace` | Words to replace per synonym perturbation | 1 |
| `--perturbation-method` | `synonym` or `paraphrase` | `synonym` |
| `--facts` | Inject supporting facts into QASC prompts (off by default to avoid ceiling effect) | off |
| `--ootb-size` | Samples for OOTB accuracy check | 100 |
| `--seed` | Random seed for reproducibility | 2266 |
| `--output-dir` | Output directory | `../outputs/results` |
| `--skip-ootb` | Skip the OOTB accuracy check | false |
| `--sensitivity-on-parsed` | Use extracted answers (strip+upper) for VR instead of raw decoded strings | off (raw by default) |
| `--phase` | Experiment phase label (e.g., `phase_6`) | `phase_1` |

### Output

Each experiment produces files in `outputs/results/<phase>/<model-key>/<dataset>/<perturbation-method>/`:

```
outputs/results/phase_6/flan-t5-large/cola/synonym/
├── w3_n20_s105.csv               # Summary CSV: one row per prompt style (accuracy, VR, etc.)
├── w3_n20_s105_detail.csv        # Detailed CSV: one row per response (raw_response, parsed_answer, parse_method, is_correct, etc.)
└── w3_n20_s105.json              # Full JSON with OOTB accuracy, experiment config, and per-style results
```

The **summary CSV** contains per-style aggregated metrics (accuracy, VR, correct count, sample size).

The **detail CSV** contains every individual model response with columns: `item_idx`, `prompt_style`, `perturbation_idx`, `correct_answer`, `variation_ratio`, `base_text`, `perturbed_text`, `prompt`, `raw_response`, `parsed_answer`, `parse_method`, `is_valid_parse`, `is_correct`.

The **JSON** contains OOTB accuracy (with `ootb_parsed` count and `parse_rate`), experiment parameters (including `sensitivity_on_raw`), and per-style results.

## Reproducibility

Seeds are fixed across all libraries via `set_all_seeds()` in `perturbations.py`:

- `random.seed()`
- `numpy.random.seed()`
- `torch.manual_seed()` and `torch.cuda.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`
- `os.environ["PYTHONHASHSEED"]`

The paraphrase generator additionally re-seeds torch before every individual generation call with `base_seed + attempt_index` to ensure deterministic output even with `do_sample=True`.

Default seed: **2266** (from the reference paper). Three seeds are used for statistical significance: **105**, **2266**, **86379** (defined in `config.EXPERIMENT_SEEDS`).

## Sensitivity Calculation: Raw vs Parsed

By default, the **Variation Ratio** (sensitivity metric) is computed on the **raw decoded model outputs** (stripped of whitespace), not on the parsed/extracted answers. This is controlled by the `sensitivity_on_raw` flag in `ExperimentConfig` (default: `True`).

| Mode | VR computed on | Accuracy computed on | When to use |
|------|---------------|---------------------|-------------|
| **Raw (default)** | Full decoded strings (strip only) | Parsed answers (strip + upper) | Standard — captures formatting changes as real sensitivity |
| **Parsed** | Extracted final answers (strip + upper) | Parsed answers (strip + upper) | When you only care about the final label, not surface form |

**Why raw is the default:** Two responses like `"A"` and `"The answer is A"` give the same parsed answer but are different raw outputs. Raw mode counts this as variation (the model is sensitive to the perturbation), which is the more conservative and informative measure. Use `--sensitivity-on-parsed` to switch to parsed mode if needed.

## Methodology

1. **Fix seeds** across all libraries for reproducibility
2. **Load model & dataset** from HuggingFace
3. **OOTB accuracy check** on unperturbed data (verifies model can perform the task; denominator is total samples, not just parsed — unparseable responses count as wrong)
4. **For each prompt style** (Control, Metacognition, Structure, Politeness):
   - For each sample in the dataset:
     - Format the question with the prompt style
     - Get the model's answer on the original input
     - Generate N semantic-preserving perturbations
     - Get the model's answer on each perturbation
     - Compute the Variation Ratio across all N+1 predictions (on raw outputs by default)
   - Report average VR and accuracy across all samples
5. **Save results** as JSON + detailed per-response CSV

## Visualization

The `visualize_results.py` script generates **control-centric dual-axis plots** for each model × dataset combination per phase. Each plot shows:

- **Left Y-axis (solid lines):** Accuracy per prompt style
- **Right Y-axis (dashed lines):** Variation Ratio (sensitivity) per prompt style
- **Lines from Control:** Each non-control style is connected to the Control baseline, making it easy to see the impact of each prompt attribute

### Generating Plots

```bash
cd src

# Generate plots for a specific phase
python visualize_results.py --phase phase_6

# Custom results/output directories
python visualize_results.py --phase phase_6 --results-dir outputs/results --output-dir outputs/figures
```

Plots are saved to `outputs/figures/<phase>/impact_plot_<model>_<dataset>.png`.

### How It Works

1. Loads all summary CSVs (non-detail) for the specified phase
2. Aggregates across seeds (mean ± std for accuracy and VR)
3. Creates one plot per model × dataset combination
4. Uses unique colors per style: Metacognition (green), Structure (red), Politeness (purple), Control (gray)

## Code Changes (Phase 5 → Phase 6)

Several bugs were identified and fixed between Phase 5 and Phase 6:

### Fix 1: OOTB Accuracy Denominator
**File:** `run_experiment.py`
The OOTB accuracy denominator was changed from `total_parsed` to `total_items`. Previously, if only 1 out of 100 responses was parseable and that 1 was correct, OOTB showed 100%. Now, unparseable responses count as wrong (accuracy = 1/100 = 1%).

### Fix 2: CoLA Structure `max_tokens`
**File:** `dataset_handlers.py`, `prompts.py`
Increased from 50 → 150. The JSON output format requires more tokens than a bare letter, and 50 tokens was truncating JSON responses mid-field.

### Fix 3: CoLA Metacognition `max_tokens`
**File:** `dataset_handlers.py`, `prompts.py`
Increased from 10 → 200. The metacognition prompt asks models to "think step by step", which produces verbose responses. With only 10 tokens the model was cut off mid-sentence before reaching a conclusion.

### Fix 4: Markdown Fence Stripping
**File:** `dataset_handlers.py`
Models like phi-3-mini wrap JSON output in `` ```json ... ``` `` markdown fences. Added `_strip_markdown_fences()` that runs before JSON parsing in both the letter parser and yes/no parser.

### Fix 5: Improved Yes/No Parser (4-Pass Strategy)
**File:** `dataset_handlers.py`
The `_parse_yes_no_verbose` function was rewritten with a 4-pass strategy for CoLA and similar binary tasks:

1. **Answer signal patterns** — looks for "the answer is Yes/No", "Answer: Yes/No", "therefore Yes/No" anywhere (last match wins)
2. **Tail yes/no** — bare "yes"/"no" in the last 50 characters
3. **Grammaticality keywords in tail** — "correct"/"grammatical" → YES, "incorrect"/"ungrammatical" → NO (negatives checked first, last 50 chars only)
4. **Full-scan fallback** — bare "yes"/"no" anywhere in the response

The tail-focused design avoids the "metacognition trap" where a model says "correct" in its reasoning but concludes "No".

## Experiment Results

### Phase 1 (with fact injection)

All Phase 1 experiments: 100 samples, 10 perturbations per sample, synonym mode, seed=2266, **QASC with fact injection** (`--facts`). Note: QASC with facts produces a ceiling effect (~99-100% accuracy on Flan-T5). Phase 2 experiments use the no-facts default (~63% accuracy on Flan-T5-Large).

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
