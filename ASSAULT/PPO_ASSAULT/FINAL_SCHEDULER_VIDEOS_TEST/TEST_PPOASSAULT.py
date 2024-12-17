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
import os
#set the video folder and how many steps to take.
video_folder = 'videos'
video_length = 6000

#generate the environment for the trained model.
env = make_atari_env('Assault-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

#reset the environment
#used VecVideoRecorder to record the steps taken and get a video. Set x == 0, to record from the beggining to desired lenght.
env.reset()
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"testtrained-")

#load the best performing model
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get directory of the script
model_path = os.path.join(script_dir, "ASSAULT_PPOlong2048.zip")
model = PPO.load(model_path, env=env)

#generate first observation 
obs = env.reset()
#take the desired number of steps and use the model's trained policy to take actions.
for _ in range(6000):
  action, _ = model.predict(obs, deterministic=True)
  obs, _, done, info = env.step(action)

env.close()

#reset the environemt and do the same but with an untrained model.
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