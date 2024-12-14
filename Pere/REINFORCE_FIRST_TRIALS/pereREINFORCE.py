# -*- coding: utf-8 -*-

import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation

import ale_py
from ale_py import ALEInterface

#set the environment as Kaboom
ENV_NAME = "ALE/Kaboom-v5"

#import evenrything needed
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import collections

import datetime
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import wandb
wandb.login()

import os

#check if gpu is available.
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
def make_env(env_name):
    env = gym.make(env_name, obs_type ="grayscale")
    print("Standard Env Grayscaled: {}".format(env.observation_space.shape))
    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, (84, 84))
    print("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    return env


env = make_env(ENV_NAME)

def print_env_info(name, env):
    obs, _ = env.reset()
    print("*** {} Environment ***".format(name))
    print("Environment obs. : {}".format(env.observation_space.shape))
    print("Observation shape: {}, type: {} and range [{},{}]".format(obs.shape, obs.dtype, np.min(obs), np.max(obs)))
    print("Observation sample:\n{}".format(obs))

print_env_info("Wrapped", env)


wandb.login()

#1.Defining the neural network for the policy.
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
HIDDEN_SIZE = 64

import torch.nn as nn
import torch.nn.init as init

def make_policynet(input_shape, output_shape):
    net = nn.Sequential(
        nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape),
        nn.Softmax(dim=1)
    )

    # Initialize weights using Xavier initialization
    for layer in net:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    return net

#just as we did in the DQN part, implement a class for the agent.
class ReinforcementAgent:
    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.current_state = self.env.reset()[0]
        self.total_reward = 0.0
        #create two lists: for storing log probabilities of actions
        self.log_probs = []
        #and one for the rewards at each step.
        self.rewards = []

    def step(self, policy_net, device="cpu"):
        state_tensor = torch.tensor([self.current_state], dtype=torch.float32).to(device)
        #get action probabilities.
        action_probs = policy_net(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        #sample actions based on probability.
        action = dist.sample()
        #store log probability of the action.
        log_prob = dist.log_prob(action)

        new_state, reward, terminated, truncated, _ = self.env.step(action.item())
        is_done = terminated or truncated
        self.total_reward += reward

        #add new probs and rewards to storing lists.
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.current_state = new_state

        if is_done:
            log_probs = self.log_probs
            rewards = self.rewards
            total_reward = self.total_reward
            self._reset()
            return total_reward, log_probs, rewards, action_probs

        return None

LEARNING_RATES = [0.0005,0.001]
GAMMAS = [0.99, 0.95]

for LEARNING_RATE in LEARNING_RATES:
  for GAMMA in GAMMAS:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="DEFINITIVE_TEST_KABOOOM",
        name = f"REINFORCE-TEST({LEARNING_RATE}/{GAMMA})",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.0001,
        "architecture": "REINFORCE",
        "dataset": "KABOOM",
        "epochs": 2500,
        }
    )
    agent = ReinforcementAgent(env)
    policy_net = make_policynet(obs_size, n_actions).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    episodes = 1500
    rolling_window = 10

    total_rewards = []
    rolling_avg_rewards = []
    losses = []

    for episode in range(episodes):
        log_probs = []
        rewards = []
        total_reward = 0

        while True:
            result = agent.step(policy_net, device=device)
            if result is not None:
                total_reward, log_probs, rewards, action_probs = result
                returns = []
                cumulative_return = 0
                for r in reversed(rewards):
                    cumulative_return = r + GAMMA * cumulative_return
                    returns.insert(0, cumulative_return)
                returns = torch.tensor(returns, dtype=torch.float32).to(device)
                baseline = returns.mean()
                std = returns.std() + 1e-9
                standardized_returns = (returns - baseline) / std

                loss = 0
                for log_prob, G in zip(log_probs, standardized_returns):
                    loss -= log_prob * G

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                break

        total_rewards.append(total_reward)
        rolling_avg_rewards.append(np.mean(total_rewards[-rolling_window:]))
        rolling_avg_loss = np.mean(losses[-rolling_window:]) if losses else 0.0

        if episode % 10 == 0:
            print(f'Episode {episode} | Total Reward: {total_reward:.2f} | Rolling Avg Reward: {rolling_avg_rewards[-1]:.2f} | Rolling Avg Loss: {rolling_avg_loss:.4f}')
        wandb.log({"REWARD PER EPISODE TRAINED": total_reward, "LOSS PER GAME":loss, "AVERAGE REWARD":rolling_avg_rewards[-1]})
        if rolling_avg_rewards[-1] >= 200 and episodes > 10:
            print(f'Solved! Environment reward threshold reached in {episode} episodes.')
            break

    torch.save(policy_net.state_dict(), f"REINFORCEMENT:LR({LEARNING_RATE}) and gamma{GAMMA}_MODEL.dat")
wandb.finish()