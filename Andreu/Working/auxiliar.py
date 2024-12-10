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


def plot_results(hyperparameters, loss_history, avg_reward_history, mean_reward, save_dir, smoothing_window=10):
    # Helper function to compute moving average
    def moving_average(data, window_size):  
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    hyperparameter_str = "_".join([f"{k}-{v}".replace(" ", "_") for k, v in hyperparameters.items()])

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Plot Loss Evolution with smoothing
    smoothed_loss = moving_average(loss_history, smoothing_window)
    axes[0].plot(loss_history, label="Loss", alpha=0.5, color='gray')  # Original loss (lighter for comparison)
    axes[0].plot(range(len(smoothed_loss)), smoothed_loss, label=f"Smoothed Loss (window={smoothing_window})", color='blue')  # Smoothed loss
    axes[0].set_title(f"Loss Evolution")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot Average Reward Evolution
    axes[1].plot(avg_reward_history, label="Average Reward")
    axes[1].set_title(f"Average Reward Evolution")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Average Reward")
    axes[1].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(os.path.join(save_dir, f"combined_plots_{hyperparameter_str}.png"))
    plt.close()




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