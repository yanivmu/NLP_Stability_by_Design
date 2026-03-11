import os

# Configuration
PHASE = "phase_1"
BASE_SLURM_DIR = f"scripts/slurm/{PHASE}"
BASE_LOG_DIR = "outputs/logs"
PROJECT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Stability_by_Design/"

# Phase 1 Experiments (Synonym only)
# Format: (model, dataset, words_to_replace, seed)
experiments = [ 
               (model, dataset, words, seed) 
                for model in ["flan-t5-large", "flan-t5-base", "llama-3.2-1b", "pythia-410m"]
                for dataset in ["cola", "qasc"]
                for words in [1, 3, 5]
                for seed in [105, 2266, 86379]
]

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
#SBATCH --mem=32000
#SBATCH --gpus=1

# 1. Activate environment
source ~/.bashrc
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
conda activate slm_env

# 2. Navigate to project
cd {project_dir}

# 3. Run Experiment
python src/run_experiment.py \\
    --model {model} \\
    --dataset {dataset} \\
    --sample-size 500 \\
    --num-perturbations 20 \\
    --perturbation-method synonym \\
    --words-to-replace {words} \\
    --seed {seed}
"""

def generate():
    os.makedirs(BASE_SLURM_DIR, exist_ok=True)
    
    print(f"Generating Slurm scripts and log directories for {PHASE}...")
    
    for model, dataset, words, seed in experiments:
        # Create a meaningful job name
        m_short = "t5l" if "large" in model else "t5b"
        job_name = f"p1_{m_short}_{dataset}_w{words}_s{seed}"
        
        # Create model-specific log directory
        log_subdir = os.path.join(BASE_LOG_DIR, PHASE, model)
        os.makedirs(log_subdir, exist_ok=True)
        slurm_subdir = os.path.join(BASE_SLURM_DIR, model, "seed_" + str(seed))
        os.makedirs(slurm_subdir, exist_ok=True)
        
        # Fill template
        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            log_subdir=log_subdir,
            project_dir=PROJECT_DIR,
            model=model,
            dataset=dataset,
            words=words,
            seed=seed
        )
        
        # Save to file (using Unix line endings)
        file_path = os.path.join(slurm_subdir, f"{job_name}.slurm")
        with open(file_path, "wb") as f:
            f.write(slurm_content.encode("utf-8"))
            
        print(f"  -> Created {file_path}")
        print(f"     Logs: {log_subdir}/")

if __name__ == "__main__":
    generate()
