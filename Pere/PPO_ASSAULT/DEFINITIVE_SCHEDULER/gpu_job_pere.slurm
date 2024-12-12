#!/bin/bash
#SBATCH --partition=tfg           # Replace with the correct partition name
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=16G                 # Memory allocation
#SBATCH --time=24:00:00           # Maximum runtime
#SBATCH --output=job-%j.log       # Log file
#SBATCH --job-name=PPO_SCHEDULER       # Job name

# Ensure Conda is set up
source ~/miniconda3/etc/profile.d/cuonda.sh

# Activate Conda environment
conda activate gpu_env

# Run your Python script
python kaboom_reinforce_scheduler_test2.py
