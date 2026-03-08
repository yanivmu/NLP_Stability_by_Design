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
python src/run_experiment.py --model flan-t5-base --dataset qasc --sample-size 1000 --num-perturbations 20 --perturbation-method synonym --seed 2266 --words-to-replace 5