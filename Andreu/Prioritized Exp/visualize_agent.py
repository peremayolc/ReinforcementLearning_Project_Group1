import imageio
import torch
from torch import Functional as F
import numpy as np

from python_Prioritized_experience import *

def visualize_agent(env, model, output_filename="video.mp4"):
    # Reset the environment
    state, _ = env.reset()
    rewards = []
    frames = []

    # Maximum time steps
    for t in range(2000):
        # Get predictions from the model
        pred = model(torch.from_numpy(state).float())
        
        # Convert Q-values to probabilities using softmax
        probabilities = F.softmax(pred, dim=0).data.numpy()
        
        # Select an action based on the probabilities
        action = np.random.choice(np.array([0, 1, 2, 3]), p=probabilities)

        # Get frame from the environment for video
        frame = env.render()
        frames.append(frame)

        # Execute action and get reward and new state
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

        if done:
            print("Total Reward:", sum(rewards))
            break

    # Save the video as MP4
    imageio.mimwrite(output_filename, frames, fps=30)
    print(f"Video saved as {output_filename}")

    # Close the environment
    env.close()




# Initialize the model (REINFORCE uses the same architecture as DQN without softmax)
model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)

try:
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    print(f"Error loading {model_path}: {e}")
                continue
      

visualize_agent(test_env, best_model, output_filename="rein.mp4")