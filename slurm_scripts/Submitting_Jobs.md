## Submitting Jobs (For Avnerf / DevOps)

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

# 1. Activate the environment
source ~/.bashrc
conda activate slm_env

# 2. Navigate to the project directory
# Make sure to clone the repo into your NetApp folder first!
cd /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/NLP_Project_Sensitivity

# 3. Run the Python execution script
python run_experiment.py --model flan-t5-base --dataset qasc --prompt_type control

```

### Useful Slurm Commands

* **Submit a job:** `sbatch run_experiment.slurm`
* **Check your jobs:** `squeue --me`
* **Cancel a job:** `scancel <job_id>`
* **Check available servers:** `sinfo`