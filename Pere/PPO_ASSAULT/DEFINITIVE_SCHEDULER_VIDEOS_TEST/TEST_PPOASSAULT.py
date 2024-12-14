import gymnasium as gym
#import matplotlib.pyplot as plt
import collections
import os
import ale_py
from ale_py import ALEInterface
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO

video_folder = 'videos'
video_length = 6000

env = make_atari_env('Assault-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

env.reset()
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"testtrained-")

model_path = "/content/ASSAULT_PPOsecond2048.zip"
model = PPO.load(model_path, env=env)

obs = env.reset()

for _ in range(6000):
  action, _ = model.predict(obs, deterministic=True)
  obs, _, done, info = env.step(action)

env.close()

env = make_atari_env('Assault-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

env.reset()
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"testuntrained-")

model_untrained = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

obs = env.reset()

for _ in range(6000):
  action, _ = model_untrained.predict(obs, deterministic=True)
  obs, _, done, info = env.step(action)

env.close()