import torch
import numpy as np
import gymnasium as gym
import json  # Import JSON for saving results
import gymnasium as gym
import ale_py
from ale_py import ALEInterface

from wrappers import make_env
from model import DQN

import os
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get directory of the script
model_paths = [
    os.path.join(script_dir, "models", "model_basic.pth"),
    os.path.join(script_dir, "models", "model_enhanced.pth"),
    os.path.join(script_dir, "models", "model_enhanced.pth")
]


# Set up the environment
env = gym.make("ALE/Kaboom-v5", obs_type="grayscale", render_mode="rgb_array")
env = make_env(env)

# Get input and action dimensions
input_shape = env.observation_space.shape
action_dim = env.action_space.n

def evaluate_model(model, env, episodes=10, log_file=None):
    """
    Evaluate a model's performance in the environment.

    Args:
        model: The DQN model to evaluate.
        env: The environment.
        episodes: Number of episodes to run for evaluation.
        log_file: File handle to log outputs.

    Returns:
        results: A dictionary with rewards for each episode and average reward.
    """
    total_rewards = []
    episode_rewards = {}

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        episode_rewards[f"Episode {episode + 1}"] = episode_reward
        log_line = f"Episode {episode + 1}: Reward = {episode_reward}"
        print(log_line)
        if log_file:
            log_file.write(log_line + "\n")
            log_file.flush()

    avg_reward = np.mean(total_rewards)
    summary_line = f"Average Reward over {episodes} episodes: {avg_reward}"
    print(summary_line)
    if log_file:
        log_file.write(summary_line + "\n")
        log_file.flush()

    # Return results as a dictionary
    episode_rewards["Average Reward"] = avg_reward
    return episode_rewards

# Load and evaluate each model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_file_path = "model_test_results.txt"
json_file_path = "model_test_results.json"

# Initialize JSON results structure
all_results = {}

with open(log_file_path, "w") as log_file:
    for model_path in model_paths:
        header = f"\nEvaluating model: {model_path}\n"
        print(header)
        log_file.write(header + "\n")
        log_file.flush()

        # Initialize the model and load weights
        model = DQN(input_shape, action_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode

        # Evaluate the model
        model_name = model_path.split("\\")[-3]  # Extract a unique name for the model
        results = evaluate_model(model, env, episodes=10, log_file=log_file)

        # Append results to the JSON structure
        all_results[model_name] = results

# Write results to a JSON file (append if the file exists)
try:
    with open(json_file_path, "r") as json_file:
        existing_results = json.load(json_file)
except (FileNotFoundError, json.JSONDecodeError):
    existing_results = {}

existing_results.update(all_results)

with open(json_file_path, "w") as json_file:
    json.dump(existing_results, json_file, indent=4)

env.close()
