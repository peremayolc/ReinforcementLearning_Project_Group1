import numpy as np
from stable_baselines3 import A2C
import ale_py
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('AssaultNoFrameskip-v4', wrapper_kwargs={"clip_reward":False})
env = VecFrameStack(env, n_stack=4)

# Load the model
model_path = "models/A2CCnnAssault/7000000.zip"
model = A2C.load(model_path, env=env)


episode_rewards = []

for game in range (0, 10):
    done = False
    obs = env.reset()
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        done = done.any()
        episode_reward += reward[0]
    print(episode_reward)
    episode_rewards.append(int(episode_reward))

print("Rewards:", episode_rewards)
print("Average Reward:", np.mean(episode_rewards))