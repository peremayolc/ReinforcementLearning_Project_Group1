import ale_py
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C


video_folder = 'videos'
video_length = 3500
TIMESTEPS = 3500
env = make_atari_env('AssaultNoFrameskip-v4', n_envs=1)

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"prueba-{TIMESTEPS}")

# Load the model
model_path = "models/A2CCnnAssault/7000000.zip"
model = A2C.load(model_path, env=env)

obs = env.reset()

for _ in range(TIMESTEPS):
  action, _ = model.predict(obs, deterministic=True)
  obs, _, done, info = env.step(action)

# Save the video
env.close()