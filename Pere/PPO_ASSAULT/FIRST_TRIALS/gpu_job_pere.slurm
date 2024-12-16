#!/bin/bash
#SBATCH --partition=tfg           # Replace with the correct partition name
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=16G                 # Memory allocation
#SBATCH --time=12:00:00           # Maximum runtime
#SBATCH --output=job-%j.log       # Log file
#SBATCH --job-name=PPO       # Job name

# Ensure Conda is set up
source ~/miniconda3/etc/profile.d/conda.sh

# Activate Conda environment
conda activate gpu_env

# Run your Python script
python assault_first_trials.py
