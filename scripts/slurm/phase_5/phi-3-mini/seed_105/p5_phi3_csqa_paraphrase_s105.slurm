#!/bin/bash
#SBATCH --job-name=p5_phi3_csqa_paraphrase_s105
#SBATCH --output=outputs/logs/phase_5/phi-3-mini/p5_phi3_csqa_paraphrase_s105_%j.out
#SBATCH --error=outputs/logs/phase_5/phi-3-mini/p5_phi3_csqa_paraphrase_s105_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --gpus=1

# 1. Activate environment & reproducibility env vars
source ~/.bashrc
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export PYTHONHASHSEED=105
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
    --phase phase_5 \
    --model phi-3-mini \
    --dataset csqa \
    --sample-size 500 \
    --num-perturbations 20 \
    --perturbation-method paraphrase \
    --words-to-replace 0 \
    --seed 105
