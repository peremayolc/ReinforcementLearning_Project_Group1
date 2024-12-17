import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import os  # For file system operations
import json  # For saving data as JSON
from wrappers import (  # Import custom environment wrappers
    MaxAndSkipEnv,
    ProcessFrame84,
    EnhancedRewardWrapper,
    BufferWrapper,
    ScaledFloatFrame,
    EnvCompatibility,
)

def save_training_data(save_dir, hyperparameters, avg_reward_history, loss_history):
    # Prepare data to save
    data = {
        "hyperparameters": hyperparameters,
        "average_reward_history": avg_reward_history,
        "loss_history": loss_history
    }

    # Define the file path
    file_path = os.path.join(save_dir, "training_data.json")

    # Write data to a JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("Training data saved to", file_path)