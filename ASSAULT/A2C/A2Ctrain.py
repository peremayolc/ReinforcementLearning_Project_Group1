import ale_py
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env


TIMESTEPS = 20000
ALGO = f"A2CCnnAssault{TIMESTEPS}"

models_dir = f"models/{ALGO}"
logdir = "logs"

env = make_atari_env('AssaultNoFrameskip-v4', n_envs=16, seed=0)

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=TIMESTEPS, tb_log_name = ALGO)
model.save(f"{models_dir}/{TIMESTEPS}")
