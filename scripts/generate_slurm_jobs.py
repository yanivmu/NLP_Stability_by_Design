import os
from itertools import product

# Configuration
PHASE = "phase_3"
BASE_SLURM_DIR = f"scripts/slurm/{PHASE}"
BASE_LOG_DIR = "outputs/logs"
PROJECT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Stability_by_Design/"

MODELS = [
    "flan-t5-large", "flan-t5-base",
    "llama-3.2-1b", "llama-3.2-1b-instruct",
    "pythia-410m", "phi-3-mini",
]
DATASETS = ["cola", "qasc", "CSQA", "gsm8k"]
SEEDS = [105, 2266, 86379]
PERTURBATION_METHODS = ["synonym", "paraphrase"]
WORDS_TO_REPLACE = [1, 3, 5]  # only relevant for synonym method

# Build the full experiment grid.  For paraphrase, words_to_replace is N/A
# so we use a sentinel value of 0 (ignored by the runner).
experiments = []
for model, dataset, seed, method in product(MODELS, DATASETS, SEEDS, PERTURBATION_METHODS):
    if method == "synonym":
        for words in WORDS_TO_REPLACE:
            experiments.append((model, dataset, words, seed, method))
    else:
        experiments.append((model, dataset, 0, seed, method))

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_subdir}/{job_name}_%j.out
#SBATCH --error={log_subdir}/{job_name}_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --gpus=1

# 1. Activate environment & reproducibility env vars
source ~/.bashrc
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export PYTHONHASHSEED={seed}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Fallback for environment activation if .bashrc sourcing fails
if ! command -v conda &> /dev/null
then
    source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/bin/activate
fi

conda activate slm_env

# 2. Navigate to project
cd {project_dir}

# 3. Run Experiment
python src/run_experiment.py \\
    --model {model} \\
    --dataset {dataset} \\
    --sample-size 500 \\
    --num-perturbations 20 \\
    --perturbation-method {method} \\
    --words-to-replace {words} \\
    --seed {seed}
"""

_MODEL_SHORT = {
    "flan-t5-large": "t5l", "flan-t5-base": "t5b",
    "llama-3.2-1b": "ll1b", "llama-3.2-1b-instruct": "ll1bi",
    "pythia-410m": "py4", "phi-3-mini": "phi3",
}


def generate():
    os.makedirs(BASE_SLURM_DIR, exist_ok=True)

    print(f"Generating SLURM scripts and log directories for {PHASE}...")
    print(f"  Models:  {MODELS}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Methods:  {PERTURBATION_METHODS}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Total jobs: {len(experiments)}")

    for model, dataset, words, seed, method in experiments:
        m_short = _MODEL_SHORT.get(model, model[:4])
        w_tag = f"_w{words}" if method == "synonym" else ""
        job_name = f"p2_{m_short}_{dataset}_{method}{w_tag}_s{seed}"

        # Using forward slashes explicitly for Linux compatibility
        log_subdir = f"{BASE_LOG_DIR}/{PHASE}/{model}"
        os.makedirs(log_subdir, exist_ok=True)
        
        slurm_subdir = f"{BASE_SLURM_DIR}/{model}/seed_{seed}"
        os.makedirs(slurm_subdir, exist_ok=True)

        # Request more memory for Llama/Phi models
        mem = 48000 if ("llama" in model or "phi" in model) else 32000

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            log_subdir=log_subdir,
            project_dir=PROJECT_DIR,
            model=model,
            dataset=dataset,
            method=method,
            words=words,
            seed=seed,
            mem=mem
        )

        file_path = f"{slurm_subdir}/{job_name}.slurm"
        with open(file_path, "wb") as f:
            f.write(slurm_content.encode("utf-8"))

        print(f"  -> Created {file_path}")

if __name__ == "__main__":
    generate()
