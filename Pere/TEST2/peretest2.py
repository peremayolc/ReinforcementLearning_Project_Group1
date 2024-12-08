# -*- coding: utf-8 -*-
"""Untitled29.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RcHu2YklENzmmZoneoEEn33snudceY8Y
"""


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gymnasium as gym
from IPython.display import display, Image as IPImage
import time
import numpy as np
import collections
import wandb
import torch.optim as optim
from torchsummary import summary
from itertools import product
import os
import json

import gymnasium as gym

import ale_py
from ale_py import ALEInterface

# chekcs for compatibility of the step return
class EnvCompatibility(gym.Wrapper):
    def step(self, action):
        results = self.env.step(action)
        if len(results) == 4:
            observation, reward, done, info = results
            # Since the base env doesn't return truncated, assume truncated is False
            return observation, reward, done, False, info
        elif len(results) == 5:
            return results
        else:
            raise ValueError("Unexpected number of values returned from env.step()")

    def reset(self, **kwargs):
        results = self.env.reset(**kwargs)
        if isinstance(results, tuple) and len(results) == 2:
            return results
        else:
            # Assume info is an empty dict if not provided
            return results, {}

class FireResetEnv(gym.Wrapper):
    def reset(self, **kwargs):
        # Pass through `seed` and `options` arguments
        self.env.reset(**kwargs)
        # First action
        results = self.env.step(1)
        if len(results) == 5:
            obs, _, term, trunc, info = results
            done = term or trunc
        elif len(results) == 4:
            obs, _, done, info = results
        else:
            raise ValueError("Unexpected number of return values from env.step()")
        if done:
            self.env.reset(**kwargs)
        # Second action
        results = self.env.step(2)
        if len(results) == 5:
            obs, _, term, trunc, info = results
            done = term or trunc
        elif len(results) == 4:
            obs, _, done, info = results
        else:
            raise ValueError("Unexpected number of return values from env.step()")
        if done:
            self.env.reset(**kwargs)
        return obs, info



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self.obs_buffer = collections.deque(maxlen=2)  # Changed from _obs_buffer to obs_buffer
        self.skip = skip  # Changed from _skip to skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):  # Accessing self.skip instead of self._skip
            results = self.env.step(action)
            if len(results) == 5:
                obs, reward, term, trunc, info = results
                terminated = terminated or term
                truncated = truncated or trunc
            elif len(results) == 4:
                obs, reward, done, info = results
                terminated = done
                truncated = False
            else:
                raise ValueError("Unexpected number of return values from env.step()")
            self.obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        return self.process(obs)

    @staticmethod
    def process(frame):
        # Resize and crop the frame
        resized_screen = cv2.resize(frame, (84, 110), interpolation=cv2.INTER_AREA)
        cropped_frame = resized_screen[18:102, :]  # Crop to 84x84
        return cropped_frame.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        self.frames = collections.deque(maxlen=n_steps)
        old_space = env.observation_space

        # Update the observation space to reflect the stacked frames
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_steps, old_space.shape[0], old_space.shape[1]),
            dtype=dtype,
        )

    def reset(self, **kwargs):
        self.frames.clear()
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_steps):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self.observation(obs), info

    def observation(self, observation):
        self.frames.append(observation)
        # Stack frames along the first dimension (channels)
        return np.array(self.frames, dtype=self.dtype)



class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env):
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipEnv(env)
    print("MaxAndSkipEnv        : {}".format(env.observation_space.shape))
    env = FireResetEnv(env)
    print("FireResetEnv         : {}".format(env.observation_space.shape))
    env = ProcessFrame84(env)
    print("ProcessFrame84       : {}".format(env.observation_space.shape))
    env = BufferWrapper(env,4)
    print("BufferWrapper        : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    env = EnvCompatibility(env)
    return env


def print_env_info(name, env):
    obs,info = env.reset()
    print("*** {} Environment ***".format(name))
    print("Observation shape: {}, type: {} and range [{},{}]".format(obs.shape, obs.dtype, np.min(obs), np.max(obs)))
    print("Observation sample:\n{}".format(obs))

#set the environment as Kaboom
env = gym.make("ALE/Kaboom-v5", obs_type="grayscale")


import cv2
import numpy as np
import collections

env = make_env(env)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('cuda')
else:
    device = torch.device("cpu")

class policy_net(nn.Module):
    def __init__(self, input_shape, action_dim):
        """
        Initializes a convolutional policy network for the REINFORCE agent.
        Args:
            input_shape (tuple): The shape of the input observation (e.g., (4, 84, 84)).
            action_dim (int): The number of possible actions in the environment.
        """
        super(policy_net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        # Calculate the flattened size after the convolutional layers
        conv_output_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

        # Apply Xavier initialization
        self._initialize_weights()

    def _get_conv_output_size(self, input_shape):
        """
        Computes the size of the output after passing the input through the convolutional layers.
        Args:
            input_shape (tuple): The shape of the input observation (e.g., (4, 84, 84)).
        Returns:
            int: Flattened size after convolutions.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.numel()

    def _initialize_weights(self):
        """
        Initializes weights of the network using Xavier initialization.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Forward pass of the REINFORCE policy network.
        Args:
            x (torch.Tensor): Input tensor representing the state.
        Returns:
            torch.Tensor: Action probabilities.
        """
        # Pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # Output logits for each action

        # Convert logits to probabilities using softmax
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs

"""class REINFORCE_Agent:
    def __init__(self, env, policy_net, optimizer, gamma=0.99):
        '''
        Initialize the REINFORCE agent.

        Args:
            env: The environment the agent interacts with (e.g., OpenAI Gym environment).
            policy_net: Neural network that outputs action probabilities.
            optimizer: Optimizer for updating the policy network.
            gamma: Discount factor for cumulative rewards.
        '''
        self.env = env  # Environment
        self.policy_net = policy_net  # Policy network
        self.optimizer = optimizer  # Optimizer for policy network
        self.gamma = gamma  # Discount factor
        self._reset()  # Reset agent state

    def _reset(self):
        '''Resets the agent's current state by resetting the environment.'''
        self.current_state, _ = self.env.reset()
        self.total_reward = 0.0
        self.log_probs = []  # Store log probabilities of actions
        self.rewards = []  # Store rewards obtained

    def step(self):
        '''
        Perform a single step in the environment, choosing an action based on the policy network.

        Returns:
            done_reward (float or None): Total reward if the episode ends; otherwise, None.
        '''
        done_reward = None

        # Preprocess the state for the policy network
        state_tensor = torch.tensor([self.current_state], dtype=torch.float32)
        action_probs = self.policy_net(state_tensor)  # Get action probabilities

        # Sample an action based on the policy
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()  # Sample action
        log_prob = action_dist.log_prob(action)  # Get log probability of the action

        # Take the action in the environment
        new_state, reward, terminated, truncated, _ = self.env.step(action.item())
        is_done = terminated or truncated

        # Update agent's state and store log prob and reward
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.current_state = new_state
        self.total_reward += reward

        # Check if episode is done
        if is_done:
            done_reward = self.total_reward
            self._update_policy()
            self._reset()

        return done_reward

    def _update_policy(self):
        '''
        Updates the policy network using the REINFORCE algorithm.
        '''
        # Compute discounted rewards
        discounted_rewards = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards for stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()

        # Update the policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

"""

def train_reinforce(env, policy_net, optimizer, gamma, DEVICE):
    """
    Train a REINFORCE agent in the given environment.

    Args:
        env: The environment to train in.
        policy_net: The policy network for the agent.
        optimizer: Optimizer for updating the policy network.
        hyperparameters: A dictionary of training hyperparameters.
        target_reward: Target average reward to consider the task solved.

    Returns:
        policy_net: The trained policy network.
        total_rewards: List of total rewards per episode.
        avg_reward_history: List of average rewards over time.
    """
    max_episodes = 1000
    target_reward = 20

    wandb.init(
        project="PERE-DEFINITVE-TEST",
        config={
            "learning_rate": learning_rate,
            "gamma": gamma,
            "target_reward": target_reward,
            "number_of_episodes": max_episodes,
            "number_of_rewards_to_average": 10,
        }
    )

    total_rewards = []
    avg_reward_history = []

    for episode in range(max_episodes):
      current_state, _ = env.reset()
      log_probs = []
      rewards = []
      total_reward = 0

      while True:
          state_tensor = torch.tensor([current_state], dtype=torch.float32).to(DEVICE)
          action_probs = policy_net(state_tensor)

          action_dist = torch.distributions.Categorical(action_probs)
          action = action_dist.sample()
          log_prob = action_dist.log_prob(action)

          new_state, reward, terminated, truncated, _ = env.step(action.item())
          is_done = terminated or truncated

          log_probs.append(log_prob)
          rewards.append(reward)
          total_reward += reward
          current_state = new_state

          if is_done:
              total_rewards.append(total_reward)
              break

      # Compute discounted rewards
      discounted_rewards = []
      G = 0
      for reward in reversed(rewards):
          G = reward + gamma * G
          discounted_rewards.insert(0, G)
      discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(DEVICE)

      # Normalize rewards
      discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

      # Compute and apply policy loss
      policy_loss = []
      for log_prob, G in zip(log_probs, discounted_rewards):
          policy_loss.append(-log_prob * G)
      policy_loss = torch.cat(policy_loss).sum()

      optimizer.zero_grad()
      policy_loss.backward()

      optimizer.step()

      # Check mean reward
      mean_reward = np.mean(total_rewards[-10:])
      avg_reward_history.append(mean_reward)

      print(f"Episode {episode}: Total Reward = {total_reward}, Mean Reward 10 episodes= {mean_reward:.3f} Loss: {policy_loss:.3f}")
      wandb.log=({"MEAN REWARD 10 episodes":mean_reward,"TOTAL REWARD PER EPISODE":total_reward, "LOSS":policy_loss})

      if mean_reward >= target_reward:
          print(f"Target reward achieved! Solved in {episode + 1} episodes.")
          break

    wandb.finish()
    return policy_net, total_rewards, avg_reward_history

env = gym.make("ALE/Kaboom-v5", obs_type="grayscale")
env = make_env(env)

wandb.login()

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Policy Network and Optimizer
input_shape = env.observation_space.shape
action_dim = env.action_space.n
gammas=[0.99,0.95]
lrs =[5e-5, 1e-4]

for learning_rate in lrs:
  for gamma in gammas:
    policy_net_model = policy_net(input_shape, action_dim).to(device)
    optimizer = optim.Adam(policy_net_model.parameters(), lr=learning_rate)
    print("Training the REINFORCE Agent...")
    trained_policy, total_rewards, avg_reward_history = train_reinforce(
      env,
      policy_net_model,
      optimizer,
      gamma,
      DEVICE=device
    )

    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward_history, label="Average Reward (last 100 episodes)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Training Process")
    plt.legend()
    plt.show()

    plt.savefig(f'TRAINING-PROCESS[{learning_rate}][{gamma}].png')
    plt.close()