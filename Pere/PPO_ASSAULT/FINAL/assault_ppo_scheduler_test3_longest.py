import gymnasium as gym

import ale_py
from ale_py import ALEInterface
from ale_py import ALEState

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

import os

#check gpu availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

#imports to use everything needed to train the model. Some of them are not needed.
import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union, Type, Dict

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

#generate the environment and 
env = make_atari_env("Assault-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4) #frame stacking

#create a class to track the models behaviour during training and return rewards, losses and such.
class RewardTrackingCallback(BaseCallback):
    def __init__(self, rolling_window=10, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.rolling_window = rolling_window #how many steps to take before averaging.
        #to store episode rewards, the rolling average of the set window and the loss values.
        self.episode_rewards = [] 
        self.rolling_avg_rewards = []
        self.loss_values = []
    
    #store the information on the created lists.
    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    self.episode_rewards.append(info["episode"]["r"])
                    if len(self.episode_rewards) >= self.rolling_window:
                        avg_reward = np.mean(self.episode_rewards[-self.rolling_window:])
                        self.rolling_avg_rewards.append(avg_reward)
                    else:
                        self.rolling_avg_rewards.append(np.mean(self.episode_rewards))

        #check if loss can be stored (had a problem)
        if "loss" in self.locals:
            self.loss_values.append(self.locals["loss"])

        return True

#call the callback class and set 0 as the rolling window.
callback = RewardTrackingCallback(rolling_window=10)

#generate the learning rate scheduler to update the value at each timestep.
def lr_schedule(progress_remaining):
    return 5e-5 * progress_remaining

#call the model, using the set learning rate.
model = PPO("CnnPolicy", env, verbose=1, learning_rate=lr_schedule, n_steps=2048)
model.learn(total_timesteps=10_000_000, callback=callback) #train the model.

#save the model.
model.save(f"ASSAULT_PPOlong{2048}")

#plot the reward and average reward during the training.
#use the callback, where we have stored all the information during the process.
plt.figure(figsize=(16, 10))
plt.subplot(1, 2, 1)
plt.plot(callback.episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(callback.rolling_avg_rewards, label='Rolling Average Reward (10 episodes)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Rolling Average Reward')
plt.legend()

plt.tight_layout()
plt.show()

#save the training process as a .png.
plt.savefig(f'training_plot_long{2048}.png')
plt.close()