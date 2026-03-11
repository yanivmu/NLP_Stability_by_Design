# Experiments To Run — Stability by Design

## Overview

Current results are largely unusable due to three compounding problems:
1. **Ceiling effect on QASC** — Flan-T5 scores 96–100%, leaving no room for improvement
2. **Non-instruction-tuned models** — Pythia-410M and Llama-3.2-1B (base) can't follow zero-shot instructions → accuracy ≈ 0–13%, VR = 0.0
3. **Weak perturbations** — `words_to_replace=1` on QASC's long prompts barely changes anything

Below are experiments grouped by priority. **Phase 1** can run immediately with existing code. **Phase 2** requires small code changes (provided below). **Phase 3** is for comprehensive coverage.

---

## Phase 1 — Run Now (No Code Changes)

These use existing datasets (qasc, cola) and models. Focus: stronger perturbation settings, paraphrase method, larger samples, the one promising combination (flan-t5-large + cola).

```bash
# ============================================================
# 1A. Flan-T5-Large on CoLA — BEST RESULT SO FAR, scale it up
# ============================================================
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 1 --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 5 --seed 2266

# Different seeds for statistical robustness
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 105
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 86379

# ============================================================
# 1B. Flan-T5-Base on CoLA — currently degenerate (VR=0.0),
#     scale up to see if signal emerges
# ============================================================
python src/run_experiment.py --model flan-t5-base --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 5 --seed 2266

# ============================================================
# 1C. Paraphrase perturbations — never tested yet
#     These use a small LM to generate more natural perturbations
# ============================================================
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset cola --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset qasc --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset qasc --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266

# ============================================================
# 1D. Stronger perturbations on QASC to break the ceiling
# ============================================================
python src/run_experiment.py --model flan-t5-base --dataset qasc --sample-size 500 --num-perturbations 20 --words-to-replace 5 --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset qasc --sample-size 500 --num-perturbations 20 --words-to-replace 5 --seed 105
python src/run_experiment.py --model flan-t5-base --dataset qasc --sample-size 500 --num-perturbations 20 --words-to-replace 5 --seed 86379
```

---

## Phase 2 — After Code Changes

### Required Code Changes

**Change 1: Add CSQA and GSM8K datasets** — Add to `src/datasets_config.py`:

```python
# In DATASET_CONFIGS dict, add:
"csqa": DatasetConfig(
    name="CSQA",
    hf_name="tau/commonsense_qa",
    answer_type=AnswerType.LETTER,
    random_baseline=0.20,     # 1/5 for 5 choices
    valid_threshold=0.35,
    valid_answers="ABCDE",
),
"gsm8k": DatasetConfig(
    name="GSM8K",
    hf_name="openai/gsm8k",
    hf_subset="main",
    split="test",
    answer_type=AnswerType.NUMBER,  # need to add this enum value
    random_baseline=0.0,
    valid_threshold=0.10,
    valid_answers="",
),
```

You also need to add format/get functions for CSQA and GSM8K in `datasets_config.py`, add `NUMBER = "number"` to the `AnswerType` enum, add CSQA/GSM8K prompt templates to `prompts.py`, and add numeric answer parsing to `data_analysis.py`.

**Change 2: Add instruction-tuned models** — Add to `MODEL_CONFIGS` in `src/models.py`:

```python
"llama-3.2-1b-instruct": ModelConfig(
    name="Llama-3.2-1B-Instruct",
    hf_name="meta-llama/Llama-3.2-1B-Instruct",
    model_type=ModelType.CAUSAL,
    use_fp16=True,
    padding_side="left",
    default_max_tokens=15,
),
"phi-2": ModelConfig(
    name="Phi-2",
    hf_name="microsoft/phi-2",
    model_type=ModelType.CAUSAL,
    use_fp16=True,
    padding_side="left",
    default_max_tokens=20,
),
"gemma-2b-it": ModelConfig(
    name="Gemma-2B-IT",
    hf_name="google/gemma-2b-it",
    model_type=ModelType.CAUSAL,
    use_fp16=True,
    padding_side="left",
    default_max_tokens=20,
),
```

**Change 3: Add temperature parameter** — In `src/models.py` `run_inference()`, add a `temperature` parameter (default 0.0 = greedy). When temperature > 0, set `do_sample=True`. This lets us compare greedy vs. stochastic decoding (matching the reference paper's setup at temperature=0.8).

**Change 4: Add multi-seed support** — Add `--seeds` argument to run_experiment.py that accepts multiple seeds and runs the full experiment for each, then aggregates (mean ± std) across seeds. The reference paper uses 3 seeds (2266, 105, 86379).

---

### Phase 2 Experiments — CSQA (5-way commonsense reasoning)

CSQA is harder than QASC (5-way, no supporting facts given). This should avoid the ceiling effect.

```bash
# Flan-T5 family
python src/run_experiment.py --model flan-t5-base --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 1 --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 1 --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266

# Repeat with different seeds
python src/run_experiment.py --model flan-t5-base --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 105
python src/run_experiment.py --model flan-t5-base --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 86379
python src/run_experiment.py --model flan-t5-large --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 105
python src/run_experiment.py --model flan-t5-large --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 86379

# Instruction-tuned models
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 105
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 86379
python src/run_experiment.py --model phi-2 --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model gemma-2b-it --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
```

### Phase 2 Experiments — GSM8K (math reasoning)

GSM8K is a harder reasoning task. Even large models struggle, making it ideal for measuring sensitivity differences.

```bash
python src/run_experiment.py --model flan-t5-base --dataset gsm8k --sample-size 500 --num-perturbations 20 --words-to-replace 1 --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset gsm8k --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset gsm8k --sample-size 500 --num-perturbations 20 --words-to-replace 1 --seed 2266
python src/run_experiment.py --model flan-t5-large --dataset gsm8k --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset gsm8k --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model phi-2 --dataset gsm8k --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
```

### Phase 2 Experiments — Instruction-tuned models on existing datasets

```bash
# Llama-3.2-1B-Instruct (replaces the useless base Llama)
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset qasc --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266

# Phi-2 (strong 2.7B model, good at reasoning)
python src/run_experiment.py --model phi-2 --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model phi-2 --dataset qasc --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266

# Gemma-2B-IT
python src/run_experiment.py --model gemma-2b-it --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
python src/run_experiment.py --model gemma-2b-it --dataset qasc --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266
```

---

## Phase 3 — Comprehensive / Nice-to-Have

### Paraphrase perturbations on new datasets

```bash
python src/run_experiment.py --model flan-t5-large --dataset csqa --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266
python src/run_experiment.py --model flan-t5-base --dataset csqa --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266
python src/run_experiment.py --model llama-3.2-1b-instruct --dataset csqa --sample-size 500 --num-perturbations 10 --perturbation-method paraphrase --seed 2266
```

### Temperature experiments (compare greedy vs. sampled, matching reference paper)

If temperature parameter is added:
```bash
python src/run_experiment.py --model flan-t5-large --dataset cola --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266 --temperature 0.8
python src/run_experiment.py --model flan-t5-large --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266 --temperature 0.8
python src/run_experiment.py --model flan-t5-base --dataset csqa --sample-size 500 --num-perturbations 20 --words-to-replace 3 --seed 2266 --temperature 0.8
```

---

## Slurm Template

For each command above, use this Slurm wrapper (adjust memory for larger models):

```bash
#!/bin/bash
#SBATCH --job-name=nlp_slm_proj
#SBATCH --output=outputs/logs/nlp_proj_%j.out
#SBATCH --error=outputs/logs/nlp_proj_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

source ~/.bashrc
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
conda activate slm_env

cd /vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Stability_by_Design/

# PASTE COMMAND HERE
```

For Phi-2 or Llama-3.2-1B-Instruct, request `--mem=48000`.

---

## Priority Order (what to submit first)

1. **Phase 1 — 1A** (Flan-T5-Large CoLA, 3 seeds) — strengthen the one good result
2. **Phase 2 code changes** — add CSQA, GSM8K, instruction-tuned models
3. **Phase 2 — CSQA experiments** (all Flan-T5 + instruction-tuned models)
4. **Phase 1 — 1C** (paraphrase perturbations on existing datasets)
5. **Phase 2 — GSM8K experiments**
6. Everything else
