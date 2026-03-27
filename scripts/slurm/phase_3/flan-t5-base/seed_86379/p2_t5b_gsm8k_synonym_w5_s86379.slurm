#!/bin/bash
#SBATCH --job-name=p2_t5b_gsm8k_synonym_w5_s86379
#SBATCH --output=outputs/logs/phase_3/flan-t5-base/p2_t5b_gsm8k_synonym_w5_s86379_%j.out
#SBATCH --error=outputs/logs/phase_3/flan-t5-base/p2_t5b_gsm8k_synonym_w5_s86379_%j.err
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
export PYTHONHASHSEED=86379
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
    --model flan-t5-base \
    --dataset gsm8k \
    --sample-size 500 \
    --num-perturbations 20 \
    --perturbation-method synonym \
    --words-to-replace 5 \
    --seed 86379
