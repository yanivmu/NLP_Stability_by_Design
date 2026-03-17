#!/bin/bash
#SBATCH --job-name=p2_ll1b_qasc_paraphrase_s105
#SBATCH --output=outputs/logs\phase_2\llama-3.2-1b/p2_ll1b_qasc_paraphrase_s105_%j.out
#SBATCH --error=outputs/logs\phase_2\llama-3.2-1b/p2_ll1b_qasc_paraphrase_s105_%j.err
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
cd /vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Stability_by_Design/

# 3. Run Experiment
python src/run_experiment.py \
    --model llama-3.2-1b \
    --dataset qasc \
    --sample-size 500 \
    --num-perturbations 20 \
    --perturbation-method paraphrase \
    --words-to-replace 0 \
    --seed 105
