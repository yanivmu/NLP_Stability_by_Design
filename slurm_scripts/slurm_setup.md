# TAU CS Cluster Setup Guide

**Team Members:** `avnerf`, `sharonl4`, `yanivmualem`, `edendaya`

This guide provides the exact copy-paste commands to set up the Python environment on the TAU CS Slurm Cluster. **Do not deviate from these paths**, as the university enforces strict storage quotas on the home (`~`) directory.

---

## Part 1: Environment Setup (For All Team Members)

Open your local terminal and follow these steps sequentially.

### Step 1: Connect to the Login Node
Connect to the university cluster using your TAU username *(ensure you are on the university network or VPN)*.

```bash
# Copy the line with your TAU username (avnerf, edendaya, yanivmualem, sharonl4) and the password is your TAU password.
ssh avnerf@slurm-client.cs.tau.ac.il
ssh edendaya@slurm-client.cs.tau.ac.il
ssh yanivmualem@slurm-client.cs.tau.ac.il
ssh sharonl4@slurm-client.cs.tau.ac.il

```

### Step 2: Download Anaconda to NetApp Storage

We must install Anaconda on the shared NetApp drive, **not** in the home directory.

```bash
# Define your personal NetApp path
export MY_NETAPP_PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"

# Create the directory and navigate to it
mkdir -p $MY_NETAPP_PATH
cd $MY_NETAPP_PATH

# Download the Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

```

### Step 3: Install Anaconda

Run the installer:

```bash
bash Anaconda3-2023.09-0-Linux-x86_64.sh

```

⚠️ **CRITICAL INSTALLATION INSTRUCTIONS:**

1. Press `ENTER` to view the license, then press `q` to skip to the end.
2. Type `yes` to accept the terms.
3. **When asked for the installation location, DO NOT press ENTER!** Copy and paste the following path (replace `<username>` with your actual username):
`/vol/joberant_nobck/data/NLP_368307701_2526a/<username>/anaconda3`
4. Type `yes` when asked to initialize Anaconda by running `conda init`.

### Step 4: Configure Package Directories

To prevent heavy pip/conda cache files from crashing your home directory quota:

```bash
# Refresh terminal to enable conda commands
source ~/.bashrc

# Redirect conda packages to NetApp
conda config --add pkgs_dirs /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/conda_pkgs

```

### Step 5: Create the Virtual Environment

We will use Python 3.10 and install the required ML libraries.

```bash
# Create the environment
conda create -y -n slm_env python=3.10

# Activate the environment
conda activate slm_env

# Install PyTorch (CUDA 11.8 compatible) and NLP libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate pandas numpy matplotlib

```

*🎉 Your environment is now fully set up! Remember to run `conda activate slm_env` every time you log in.*

---

## 🚀 Part 2: Submitting Jobs (For Avnerf / DevOps)

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

```

אתה יכול לשמור את זה כ-`CLUSTER_SETUP.md` בתיקייה הראשית של ה-Repository, ופשוט להגיד לשאר חברי הצוות "כנסו לקובץ ההתקנה ותעשו העתק-הדבק לשורות של Part 1".

יש עוד משהו שקשור לתשתיות או לקוד שתרצה שאכין לך כעת?

```