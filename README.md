# Stability by Design: Prompt Sensitivity Analysis in Small Language Models

An NLP research project investigating how **prompt design choices** affect the **stability** of Small Language Models (SLMs) when inputs are slightly rephrased. Based on the sensitivity framework from [Lu et al., 2024](https://arxiv.org/abs/2311.07230) (*"How are Prompts Different in Terms of Sensitivity?"*, NAACL 2024).

## Research Question

> Do certain prompt properties (metacognition, structure, politeness) make language models more stable when inputs undergo semantic-preserving perturbations?

## Background

The reference paper shows that **sensitivity** (measured via variation ratio) is an unsupervised proxy for model performance and that different prompt designs lead to different sensitivity levels. We adapt this framework to **small models on shared cluster GPUs**, where hard benchmarks often collapse to chance: the pipeline and dataset choices below are built so that **most reported runs** use tasks on which at least some of our models already show usable accuracy, and so that expensive sensitivity passes only run after an automatic out-of-the-box check. We also use a cleaner perturbation pipeline based on NLTK WordNet.

**Reference code in this repo:** Our pipeline follows the same **variation-ratio on raw model generations** convention as in the original papar (see *Sensitivity (Variation Ratio) and alignment with Lu et al. (2024)* below); the sensitivity definition matches the methodology we adopted from there.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Variation Ratio (VR)** | `VR = 1 - f_modal / (N + 1)` where `f_modal` is the **modal count** in the multiset of **N+1** outputs (default: stripped full decodes; optional: parsed labels — see *Sensitivity*). VR = 0 is stable, VR = 1 is maximally unstable. |
| **Semantic-preserving perturbations** | Input variations that change surface form but not meaning (synonym replacement, paraphrasing). |
| **OOTB accuracy** | Out-Of-The-Box accuracy on unperturbed data with the **Control** prompt, evaluated before sensitivity runs. Used with dataset-specific **random** and **valid-signal** thresholds to decide whether to run the full experiment (see *OOTB screening gate*). Denominator is all OOTB samples; unparseable outputs count as incorrect. |
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
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) CSQA| 5-way multiple-choice commonsense | A-E | 20% | Active |


New datasets are added by implementing a `DatasetHandler` subclass — zero changes to the pipeline code.

**QASC fact injection:** Disabled by default to avoid a ceiling effect (~99% → ~63% accuracy on Flan-T5-Large). Use `--facts` to re-enable for comparison—this turns on **fact1 / fact2** (and related context) in the prompt, similar to minimal-unblocking QASC setups that make the task solvable for smaller LMs.

**Which datasets we lean on:** **QASC** and **CoLA** carry most of the analysis because encoder–decoder instruction-tuned models reach accuracies clearly above chance. **CommonsenseQA** is run when a given model passes the same out-of-the-box gates as the other tasks; small decoder-only checkpoints often fail those gates and **never enter** the sensitivity phase, so we do not treat near-random behavior as prompt-design signal. **GSM8K** stays **registered but stub-level**: free-form numeric answers are difficult for this model class and parsing is fragile, so the project does not anchor conclusions on GSM8K. Together with the mandatory OOTB step in `run_experiment.py` (see *OOTB screening gate*), this keeps experiments on tasks where variation ratio is interpretable rather than a side effect of uniform wrong answers.

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

Experiments can be run from the **repository root** with `python src/run_experiment.py` (recommended: matches Slurm scripts and default `./outputs/results` → `outputs/results/` at repo root). If you `cd src`, the same default resolves to `src/outputs/results/` unless you pass e.g. `--output-dir ../outputs/results`.

### Basic Experiment

```bash
cd src
python run_experiment.py --model flan-t5-base --dataset qasc
```

From repository root (same effect if `output_dir` is unchanged):

```bash
python src/run_experiment.py --model flan-t5-base --dataset qasc
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
| `--output-dir` | Output directory (relative to shell cwd) | `./outputs/results` |
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

## Sensitivity (Variation Ratio) and alignment with Lu et al. (2024)

**Definition (unchanged):** For one item we collect the model’s output on the **original** text plus **N** perturbed variants (\(N+1\) members of a **multiset**). Let \(f_m\) be the **largest frequency** among those members—how many of the \(N+1\) outputs are identical to the mode—after a chosen normalization. Lu et al. use \(s = 1 - f_m / (N+1)\). See the *Key Concepts* table for the intuition (VR = 0 is stable, VR = 1 is maximally unstable).

**Raw outputs vs parsed labels (matches reference methodology):** In `run_experiment.py`, when `sensitivity_on_raw` is true (the default, `ExperimentConfig.sensitivity_on_raw=True`), the multiset passed into `ResultAnalyzer.calculate_variation_ratio` contains the **decoded generation text** for each of the \(N+1\) runs, after **`.strip()` only** — not the dataset handler’s parsed letter / Yes–No / JSON field. That mirrors the spirit of the original sensitivity pipeline in `reference_paper_code/sensetivity_article_project/`: sensitivity is measured over **what the model actually emitted**, so differences in JSON layout, punctuation, or hedging count as instability in the same way as in the paper’s setup (they are not “noise” to be washed out before measuring sensitivity).

**Structured task accuracy:** Regardless of VR mode, **task accuracy** is always computed on **parsed** structured answers (letters, Yes/No, `final_answer` from JSON, etc.) against the gold label. So “did it answer the question correctly?” is always a discrete structured comparison; VR answers whether **generations** (default) or **extracted labels** (optional) stay tied together under paraphrase.

| Mode | VR multiset contains | Accuracy computed on | CLI |
|------|----------------------|----------------------|-----|
| **Raw (default)** | Full decoded strings (strip only) | Parsed answers | *(default)* |
| **Parsed** | Extracted answers (strip + upper), same normalization as accuracy | Parsed answers | `--sensitivity-on-parsed` sets `sensitivity_on_raw=False` |

**Parsed VR:** Use `--sensitivity-on-parsed` when the question is specifically whether the **normalized label** flips across perturbations, ignoring formatting-only churn. Comparing raw and parsed VR on the same runs can separate surface instability from label instability. The report should state which mode was used; primary tables here assume **raw-output VR** unless noted.

## OOTB screening gate

Before any sensitivity loop, `run_experiment.py` runs an **out-of-the-box (OOTB)** check: **Control** prompt, `ootb_size` items (default **100**), same parsing rules as the main run. Accuracy is **correct / total_items** (every drawn item counts; unparseable outputs do not count as “parsed correct” — they lower accuracy), consistent with the reference paper’s use of a sanity pass on unperturbed data. This step always runs first so we **measure performance on a dataset before paying for the full four-style sensitivity sweep**, and so weak model–task pairs exit early instead of producing misleading VR summaries.

**Out-of-the-box threshold values** (from each handler’s `DatasetConfig`: `random_baseline` and `valid_threshold`):

| Dataset | Random baseline | Valid-signal threshold |
|---------|-----------------|------------------------|
| QASC (8-way) | 12.5% | 40% |
| CoLA (binary) | 50% | 60% |
| CommonsenseQA (5-way) | 20% | 40% |
| GSM8K (numeric) | 0% (no meaningful random guess) | 20% |

**Decision logic:**

1. If OOTB accuracy is **strictly below** `random_baseline`, the run **stops** (exit code 0): the model is not better than guessing on that task, so sensitivity numbers would not support meaningful conclusions about prompt design.
2. Else if OOTB accuracy is **strictly below** `valid_threshold`, the run **stops**: there is some signal above chance, but not enough to treat downstream VR/accuracy as a reliable probe of prompt stability (aligned with the “valid signal” idea in `DataManager` / dataset configs).
3. Otherwise the run **continues** with the full four-style sensitivity experiment.

**Why we gate on these thresholds:** Variation ratio is most informative when the model **can** solve a non-trivial fraction of items; otherwise low VR often reflects **repeated wrong or degenerate outputs** rather than desirable robustness (see README interpretation for small base models). Gating saves cluster time and keeps reported cases comparable to the paper’s setting where models are actually engaged with the task. For debugging or forced runs on weak models, use `--skip-ootb` (not recommended for final reported numbers).

## Methodology

1. **Fix seeds** across all libraries for reproducibility (`perturbations.set_all_seeds`, fixed multi-seed grid in `config.EXPERIMENT_SEEDS` for aggregation).
2. **Load model & dataset** from HuggingFace via handler registries.
3. **OOTB accuracy check** (unless `--skip-ootb`): Control prompt; apply **screening gate** above; only if the run passes do we execute sensitivity for all prompt styles.
4. **For each prompt style** (Control, Metacognition, Structure, Politeness):
   - For each sample in the sensitivity subset:
     - Build the prompt for that style
     - Decode outputs on the original text and on **N** perturbations
     - Compute **variation ratio on raw stripped strings** by default (see *Sensitivity* section)
     - Record **accuracy** from parsed answers on the original input
   - Aggregate mean VR and mean accuracy over items
5. **Save results** as JSON + summary CSV + optional detailed per-response CSV

## Visualization

The script `src/visualize_results.py` aggregates **summary** result CSVs from `outputs/results/<phase>/...` and writes **control-centric dual-axis** figures under `outputs/figures/<phase>/`.

### How figures are generated

1. **Discovery:** Recursively glob `*.csv` under `outputs/results/<phase>/`, and **exclude** paths containing `detail` (only one-row-per-style summary files are merged).
2. **Filtering:** Keep rows that match each `(model, dataset)` pair present in the combined table.
3. **Aggregation:** Group by `prompt_style` and compute **mean and std** of `accuracy` and `variation_ratio` across all runs (typically multiple **seeds** and timestamped files from repeated jobs).
4. **Plotting:** One PNG per `(model, dataset)` for the chosen `--phase`. Matplotlib + Seaborn; high DPI (300) for print-quality exports.

### How to read the plots

- **X-axis:** Prompt styles in fixed order: Control, Metacognition, Structure, Politeness.
- **Left axis (accuracy):** **Circles** mark mean accuracy per style. **Solid** colored segments connect **Control → each other style** for accuracy, so you see **Δ accuracy relative to the shared baseline** along each color.
- **Right axis (sensitivity):** **Squares** mark mean **variation ratio (VR)** per style. **Dashed** segments connect Control → each style for VR (**Δ sensitivity**).
- **Inline labels:** `"A: x.xx"` near accuracy points; `"V: x.xx"` near VR points (per-style **means** only; the script also computes per-style standard deviations across files but does **not** draw error bars).
- **Colors:** Control = gray; Metacognition = green; Structure = red; Politeness = purple (non-control lines use that style’s color).
- **Legend:** Explains solid = accuracy, dashed = VR, and the color mapping for the three non-control styles.

**Note:** Summary CSV columns must include `model`, `dataset`, `prompt_style`, `accuracy`, and `variation_ratio` as produced by `run_experiment.py` / `save_to_csv`.

### Generating plots

Run from the **repository root** so default paths match `run_experiment.py` (Slurm jobs use `python src/run_experiment.py` from the project root with `output_dir` defaulting to `./outputs/results`).

```bash
# From repository root — recommended
python src/visualize_results.py --phase phase_6

# Explicit paths (e.g. if your results live elsewhere)
python src/visualize_results.py --phase phase_6 \
  --results-dir outputs/results \
  --output-dir outputs/figures
```

Output files: `outputs/figures/<phase>/impact_plot_<model>_<dataset>.png`.

**Course note on figures:** The NLP project guidelines recommend **vector figures (PDF)** in the final ACL-formatted report for crisp typography; this repository exports **PNG** for quick iteration — regenerate or replot in PDF for the camera-ready paper if required.

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

### Phase 6 (canonical results)

All Phase 6 experiments: 500 samples, 20 perturbations per sample, averaged across synonym (w1/w3/w5) and paraphrase perturbation methods, averaged across 3 seeds (105, 2266, 86379). QASC without fact injection. VR computed on raw decoded outputs (default mode).

**Note on Llama-3.2-1B Structure:** The structure prompt produces `0.000` VR and `0%` accuracy on CoLA for all runs, indicating the model outputs a constant empty/unparseable response for that style — degenerate rather than stable.

### Flan-T5-Base

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| QASC | 44% | Control | 0.263 | 45% |
| | | Metacognition | 0.592 | 41% |
| | | Structure | 0.269 | 40% |
| | | Politeness | 0.271 | 45% |
| CSQA | 73% | Control | 0.249 | 72% |
| | | Metacognition | 0.559 | 69% |
| | | Structure | 0.251 | 66% |
| | | Politeness | 0.248 | 70% |

### Flan-T5-Large

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| CoLA | 63% | Control | 0.162 | 64% |
| | | Metacognition | 0.169 | 68% |
| | | Structure | 0.134 | 70% |
| | | Politeness | 0.178 | 70% |
| QASC | 62% | Control | 0.235 | 62% |
| | | Metacognition | 0.653 | 37% |
| | | Structure | 0.143 | 22% |
| | | Politeness | 0.238 | 62% |
| CSQA | 82% | Control | 0.224 | 83% |
| | | Metacognition | 0.702 | 60% |
| | | Structure | 0.224 | 50% |
| | | Politeness | 0.217 | 83% |

### Pythia-410M

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| CoLA | 64% | Control | 0.568 | 60% |
| | | Metacognition | 0.794 | 27% |
| | | Structure | 0.369 | 63% |
| | | Politeness | 0.525 | 58% |

### Llama-3.2-1B

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| CoLA | 69% | Control | 0.415 | 69% |
| | | Metacognition | 0.641 | 49% |
| | | Structure | 0.000 | 0% |
| | | Politeness | 0.488 | 69% |

### Llama-3.2-1B-Instruct

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| CoLA | 69% | Control | 0.050 | 69% |
| | | Metacognition | 0.807 | 34% |
| | | Structure | 0.453 | 51% |
| | | Politeness | 0.004 | 69% |

### Phi-3-Mini-4K-Instruct

| Dataset | OOTB Acc. | Style | VR | Accuracy |
|---------|-----------|-------|----|----------|
| CoLA | 76% | Control | 0.275 | 76% |
| | | Metacognition | 0.803 | 76% |
| | | Structure | 0.823 | 79% |
| | | Politeness | 0.184 | 76% |
| QASC | 76% | Control | 0.330 | 79% |
| | | Metacognition | 0.579 | 77% |
| | | Structure | 0.818 | 70% |
| | | Politeness | 0.395 | 79% |
| CSQA | 73% | Control | 0.273 | 74% |
| | | Metacognition | 0.432 | 72% |
| | | Structure | 0.881 | 71% |
| | | Politeness | 0.402 | 74% |

### Interpretation

- **Flan-T5-Large on CoLA** shows the clearest prompt-design signal: Structure achieves the lowest VR (0.134) with accuracy comparable to other styles, making it the most stable prompt. Politeness also achieves high accuracy (70%) with moderate VR.
- **Flan-T5-Large on QASC and CSQA** reveals a striking pattern: Metacognition dramatically increases VR (0.65–0.70) while reducing accuracy, suggesting that the "think carefully" prompt causes these models to become erratic rather than more careful. Structure keeps VR low but collapses accuracy (22% on QASC), likely because the JSON format confuses output parsing. Politeness and Control behave nearly identically.
- **Flan-T5-Base** passes the OOTB gate on QASC and CSQA but shows lower overall accuracy than Flan-T5-Large. The Metacognition prompt consistently raises VR across both datasets, mirroring the large model pattern.
- **Phi-3-Mini** achieves the highest raw accuracy (76–79%) across datasets but shows extremely high VR for Structure (0.82–0.88) — reflecting the instability introduced by JSON formatting constraints. Politeness stands out as the most stable style (VR=0.18 on CoLA), matching Control accuracy. Metacognition similarly inflates VR without accuracy gains.
- **Llama-3.2-1B-Instruct** is the most stable model for Control and Politeness (VR ≈ 0.05 and 0.004 respectively on CoLA), but Metacognition completely destabilizes outputs (VR=0.807, accuracy drops to 34%).
- **Llama-3.2-1B (base)** and **Pythia-410M** show high VR overall (0.4–0.8), consistent with the lack of instruction tuning. The Structure prompt for Llama-3.2-1B produces VR=0.000 and accuracy=0%, which reflects degenerate constant outputs rather than genuine stability.
- **Cross-model pattern:** Metacognition consistently raises VR and often hurts accuracy — the "think carefully" instruction appears counterproductive for small models. Politeness is generally the safest style, preserving accuracy while modestly controlling VR. Structure's effect is model-dependent: beneficial for Flan-T5-Large CoLA (low VR, good accuracy) but catastrophic for instruction-tuned models on structured output tasks.

## Reference Paper

Lu, S., Schuff, H., & Gurevych, I. (2024). *How are Prompts Different in Terms of Sensitivity?* NAACL 2024. [arXiv:2311.07230](https://arxiv.org/abs/2311.07230)


## License

Academic research project.
