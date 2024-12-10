import torch  # For device selection
import collections  # For namedtuple to store experiences
import gymnasium as gym  # For creating the Kaboom environment
from wrappers import make_env  # For environment wrapping
# from tuning import hyperparameter_tuning  # Uncomment if needed
import ale_py
from ale_py import ALEInterface

from model import DQN
import numpy as np
import imageio
import os  # For absolute path
import matplotlib.pyplot as plt  # For optional frame display


if __name__ == "__main__":
    # Initialize the environment with all wrappers
    env = gym.make("ALE/Kaboom-v5", obs_type="grayscale", render_mode="rgb_array")

    print("Environment render mode:", env.render_mode)
    print("Supported render modes:", env.metadata.get('render.modes', []))


    test_env = make_env(env)  # Apply your custom wrappers

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model with the correct input shape and number of actions
    input_shape = test_env.observation_space.shape  # Should be (4, 84, 84) based on your wrappers
    action_dim = test_env.action_space.n
    model = DQN(input_shape, action_dim).to(device)

    # Load the trained weights into the model
    model_path = 'C:/GitHub Repositories/ReinforcementLearning_Project_Group1/Andreu/Working/Runs/experiment_1/model.pth'  # Replace with your model's path
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}.")
    except Exception as e:
        print(f"Failed to load model weights from {model_path}: {e}")
        exit(1)

    # Set the model to evaluation mode
    model.eval()

    # Verify the model's first convolutional layer
    print("Model's First Conv Layer:", model.conv1)


    # Visualize the agent's performance
    visualize_agent(
        test_env,
        model,
        output_filename="C:/GitHub Repositories/ReinforcementLearning_Project_Group1/Andreu/Working/Videos/kaboom_agent_performance.mp4",
        max_steps=10000,
        device=device
    )
