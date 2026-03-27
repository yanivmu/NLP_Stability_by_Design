#!/bin/bash
#SBATCH --job-name=p2_py4_qasc_synonym_w3_s2266
#SBATCH --output=outputs/logs/phase_3/pythia-410m/p2_py4_qasc_synonym_w3_s2266_%j.out
#SBATCH --error=outputs/logs/phase_3/pythia-410m/p2_py4_qasc_synonym_w3_s2266_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

# 1. Activate environment & reproducibility env vars
source ~/.bashrc
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export PYTHONHASHSEED=2266
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Fallback for environment activation if .bashrc sourcing fails
if ! command -v conda &> /dev/null
then
    source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/bin/activate
fi

conda activate slm_env

# 2. Navigate to project
cd /vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Stability_by_Design/

# 3. Run Experiment
python src/run_experiment.py \
    --model pythia-410m \
    --dataset qasc \
    --sample-size 500 \
    --num-perturbations 20 \
    --perturbation-method synonym \
    --words-to-replace 3 \
    --seed 2266
