# TAU CS Cluster Setup Guide

**Team Members:** `avnerf`, `sharonl4`, `yanivmualem`, `edendaya`

This guide provides the exact copy-paste commands to set up the Python environment on the TAU CS Slurm Cluster. **Do not deviate from these paths**, as the university enforces strict storage quotas on the home (`~`) directory.

---

## Part 1: Environment Setup (For All Team Members)

Open your local terminal and follow these steps sequentially.

### Step 1: Connect to the Login Node
Connect to the university cluster using your TAU username *(ensure you are on the university network or VPN)*.

```bash
# Copy the line with your TAU username and the password is your TAU password.
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

## Part 2: GitHub & SSH Setup (For All Team Members)

Since we are collaborating using a shared GitHub repository, **every team member must set up an SSH key on the cluster** to be able to pull and push code from the shared project folder.

### Step 1: Generate an SSH Key on the Server

Open a terminal on the server (via VS Code or standard SSH) and run:

```bash
# Replace with your actual email
ssh-keygen -t ed25519 -C "your_email@tau.ac.il"

```

*(Press `ENTER` to accept all default prompts, no passphrase needed).*

### Step 2: Copy Your Public Key

Run the following command to print your newly generated key:

```bash
cat ~/.ssh/id_ed25519.pub

```

*Copy the entire output line (it starts with `ssh-ed25519`).*

### Step 3: Add the Key to GitHub

1. Go to your GitHub account settings in your local web browser.
2. Navigate to **SSH and GPG keys** -> **New SSH key**.
3. Name it something like "TAU Slurm Server" and paste the copied key.

---

## Part 3: Team Workflow & Submitting Jobs

Since you are working as a team, it is crucial to avoid conflicting files and permissions.

### Team Workflow (READ THIS FIRST)

* **Code and Data Location:** All project files (`.py`, datasets, output logs) are centrally hosted in Avner's NetApp directory.
* **Environments:** Each team member **must** use their own Conda environment (`slm_env`) that they set up in Part 1. Do not try to share or activate someone else's environment.
* **The Process:** 1. Connect to the cluster via VS Code Remote-SSH.
2. Navigate to the shared project directory: `cd /vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/NLP_Project_Sensitivity`
3. Pull the latest code from GitHub: `git pull`
4. Activate your personal environment: `conda activate slm_env`
5. Run your code or submit your Slurm jobs from this shared folder.