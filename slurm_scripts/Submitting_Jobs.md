## Part 2: Submitting Jobs to the Slurm Cluster

This document details how to submit jobs to the TAU CS Slurm Cluster, building upon the environment setup described in `slurm_setup.md`.
It is crucial to avoid conflicting files and permissions when working as a team (`avnerf`, `sharonl4`, `yanivmualem`, `edendaya`).

### Team Workflow (READ THIS FIRST)
* **Code and Data Location:** All project files (`.py`, datasets, output logs) are centrally hosted in Avner's NetApp directory. 
* **Environments:** Each team member **must** use their own Conda environment (`slm_env`) that they set up in Part 1. Do not try to share or activate someone else's environment.
* **The Process:** 1. Connect to the cluster via VS Code Remote-SSH.
   2. Navigate to the shared project directory: `cd /vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Project_Sensitivity`
   3. Pull the latest code from GitHub: `git pull`
   4. Activate your personal environment: `conda activate slm_env`
   5. Run your code or submit your Slurm jobs from this shared folder.

---

### Submitting Jobs (For Avnerf / DevOps)

To run experiments, we use the `sbatch` command to submit jobs to the `studentkillable` partition using the `gpu-students` account.

Create a file named `run_experiment.slurm` in the project root:

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

# 1. Activate the environment (This uses YOUR personal environment)
source ~/.bashrc

# Set HuggingFace cache directory to NetApp to avoid home directory quota errors
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"

conda activate slm_env

# 2. Navigate to the SHARED project directory (Avner's folder)
cd /vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Stability_by_Design/

# 3. Run the Python execution script
python src/run_experiment.py --model flan-t5-base --dataset qasc --prompt_type control

```

### Useful Slurm Commands

* **Submit a job:** `sbatch run_experiment.slurm`
* **Check your jobs:** `squeue --me`
* **Cancel a job:** `scancel <job_id>`
* **Check available servers:** `sinfo`