#!/bin/bash
#SBATCH --job-name=p4_ll1bi_cola_paraphrase_s86379
#SBATCH --output=outputs/logs/phase_4/llama-3.2-1b-instruct/p4_ll1bi_cola_paraphrase_s86379_%j.out
#SBATCH --error=outputs/logs/phase_4/llama-3.2-1b-instruct/p4_ll1bi_cola_paraphrase_s86379_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --gpus=1
#SBATCH --constraint="geforce_rtx_3090|a100|a5000|a6000|l40s|quadro_rtx_8000"

# 1. Activate environment & reproducibility env vars
source ~/.bashrc
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export PYTHONHASHSEED=86379
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    --phase phase_4 \
    --model llama-3.2-1b-instruct \
    --dataset cola \
    --sample-size 500 \
    --num-perturbations 20 \
    --perturbation-method paraphrase \
    --words-to-replace 0 \
    --seed 86379
