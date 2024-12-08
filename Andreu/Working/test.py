import torch  # For tensor operations
import torch.nn as nn  # For defining and handling neural networks
import torch.nn.functional as F  # For softmax activation
import numpy as np  # For numerical operations
import os  # For file and directory handling
import imageio  # For saving videos
import gymnasium as gym  # For environment interaction
from model import DQN  # Import the Deep Q-Network model

# Define the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def watch_agent(env, models_dir, num_episodes=100, max_experiments=8):
    """
    Evaluates the first 'max_experiments' DQN models in the specified directory and its subdirectories over a number of episodes,
    selects the best-performing model based on average cumulative reward,
    and returns the rewards and action lists for the best model.
    
    Args:
        env (gym.Env): The Gymnasium environment.
        models_dir (str): Directory containing subdirectories of DQN model files (.pth).
        num_episodes (int): Number of episodes to run for each model.
        max_experiments (int): Maximum number of experiments to evaluate.
    
    Returns:
        best_model (nn.Module): The best-performing DQN model.
        best_avg_reward (float): Average cumulative reward of the best model.
        best_rewards (list): List of cumulative rewards for each episode of the best model.
        best_actions (list): List of action sequences for each episode of the best model.
    """
    # Initialize variables to track the best model
    best_avg_reward = -float('inf')
    best_model = None
    best_rewards = []
    best_actions = []
    best_model_name = ""
    
    # Get a sorted list of subdirectories to ensure consistent ordering
    sorted_subdirs = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))])
    
    # Limit to the first 'max_experiments' subdirectories
    selected_subdirs = sorted_subdirs[:max_experiments]
    
    for subdir in selected_subdirs:
        subdir_path = os.path.join(models_dir, subdir)
        model_path = os.path.join(subdir_path, "model.pth")
        
        if os.path.exists(model_path):
            print(f"Evaluating model: {model_path}")
            
            # Initialize the model
            model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
            
            try:
                # Load the model weights
                model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
                continue
            
            # Set the model to evaluation mode
            model.eval()
            
            # Lists to store rewards and actions for this model
            rewards = []
            actions = []
            
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_actions = []
                done = False
                
                while not done:
                    # Convert state to tensor
                    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
                    
                    # Get Q-values from the model
                    with torch.no_grad():
                        q_vals = model(state_tensor).cpu().numpy()[0]
                    
                    # Select the action with the highest Q-value
                    action = np.argmax(q_vals)
                    episode_actions.append(action)
                    
                    # Take the action in the environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
        
                rewards.append(episode_reward)
                actions.append(episode_actions)
            
            # Calculate the average reward for this model
            avg_reward = np.mean(rewards)
            print(f"Average Reward for {model_path}: {avg_reward:.2f}")
            
            # Update the best model if this model has a higher average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model = model
                best_rewards = rewards
                best_actions = actions
                best_model_name = subdir
    
    if best_model is not None:
        print(f"\nBest Model: {best_model_name} with Average Reward: {best_avg_reward:.2f}")
    else:
        print("No valid models were evaluated.")
    
    return best_model, best_avg_reward, best_rewards, best_actions




def visualize_agent(env, model, output_filename="video.mp4", max_steps=10000, device=None):
    """
    Visualizes the agent's performance in the environment by creating a video.
    
    Args:
        env (gym.Env): The environment to visualize.
        model (torch.nn.Module): The trained DQN model.
        output_filename (str): The filename for the saved video.
        max_steps (int): Maximum number of steps to run in the environment.
        device (torch.device, optional): The device to run the model on. If None, uses CUDA if available.
        
    Returns:
        None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()  # Set model to evaluation mode

    frames = []  # List to store frames for the video
    rewards = []  # List to store rewards

    # Reset the environment
    state, info = env.reset()
    
    # Convert the initial state to a tensor and add a batch dimension
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():  # Disable gradient computation
        for t in range(max_steps):
            # Get Q-values from the model
            q_values = model(state)
            
            # Select the action with the highest Q-value (deterministic policy)
            action = torch.argmax(q_values, dim=1).item()
            
            # Render the current frame using the unwrapped environment
            try:
                frame = env.render()
                if frame is None:
                    print(f"Warning: Frame at step {t+1} is None.")
                else:
                    # Convert grayscale to RGB if necessary
                    if len(frame.shape) == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                        print(f"Converted grayscale frame to RGB at step {t+1}.")
                    
                    frames.append(frame)
                    print(f"Captured frame {len(frames)} at step {t+1}.")

                    # Optional: Display the first frame using matplotlib
                    if len(frames) == 1:
                        plt.imshow(frame)
                        plt.title(f"Frame at step {t+1}")
                        plt.axis('off')
                        plt.show()
            except TypeError as e:
                print(f"Render mode 'rgb_array' not supported: {e}")
                break

            # Execute the action in the environment
            try:
                next_state, reward, done, truncated, info = env.step(action)
            except Exception as e:
                print(f"Error during env.step(action): {e}")
                break
            rewards.append(reward)
            
            # Prepare the next state
            if isinstance(next_state, np.ndarray):
                state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                print("Unexpected state type. Expected np.ndarray.")
                break
            
            if done or truncated:
                print(f"Episode finished at step {t+1} with total reward {sum(rewards)}.")
                break

    # Ensure the directory for saving exists
    save_dir = os.path.dirname(output_filename)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Save the frames as a video
    print(f"Saving video to {output_filename}...")
    try:
        if frames:
            imageio.mimwrite(output_filename, frames, fps=30)
            abs_path = os.path.abspath(output_filename)
            print(f"Video saved successfully at {abs_path}.")
        else:
            print("No frames were captured. Video not saved.")
    except Exception as e:
        print(f"Failed to save video: {e}")

    # Print total frames captured
    print(f"Total frames captured: {len(frames)}")

    # Indicate script completion
    print("Finished visualizing agent.")

    # Close the environment
    env.close()
