import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import shutil
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
import os
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
from stable_baselines3.common.vec_env import VecVideoRecorder


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

video_dir = "videos"
if os.path.exists(video_dir):
    shutil.rmtree(video_dir)
os.makedirs(video_dir, exist_ok=True)

env = make_atari_env("Assault-v4", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

class RewardTrackingCallback(BaseCallback):
    def __init__(self, rolling_window=10, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.rolling_window = rolling_window
        self.episode_rewards = []
        self.rolling_avg_rewards = []
        self.loss_values = []

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
        if "loss" in self.locals:
            self.loss_values.append(self.locals["loss"])
        return True

callback = RewardTrackingCallback(rolling_window=10)

model = PPO("CnnPolicy", env, verbose=1, learning_rate=lambda progress: 5e-5 * progress, n_steps=2048)
model.learn(total_timesteps=10_000, callback=callback)

model.save("ASSAULT_PPOsecond")

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

plt.savefig(f'training_plot_second_{2048}.png')
plt.close()

env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, 'videos',
                       record_video_trigger=lambda x: x == 50, video_length=6000,
                       name_prefix=f"prueba-")


obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = env.step(action)

# Save the video
env.close()
'''
env_new = make_atari_env("Assault-v4", n_envs=1, seed=0)
env_new = VecFrameStack(env_new, n_stack=4)

untrained_model = PPO("CnnPolicy", env, verbose=1, learning_rate=lambda progress: 5e-5 * progress, n_steps=2048)
for ep in range(1):
  obs = env.reset()
  done = False
  while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
env.close()
print(f"Videos saved in {video_dir}")'''
