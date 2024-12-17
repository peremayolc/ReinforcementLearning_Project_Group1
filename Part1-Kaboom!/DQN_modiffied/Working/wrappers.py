import gymnasium as gym  # Environment management and wrappers
import collections  # Data structures for buffers
import numpy as np  # Numerical operations
import cv2  # Image processing
import os
from gymnasium.wrappers import RecordVideo
import os
import time

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

        # Initialize frames with zeros
        for _ in range(self.n_steps):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        stacked_obs = self.observation(obs)
        return stacked_obs, info

    def observation(self, observation):
        self.frames.append(observation)
        # Stack frames along the first dimension (channels)
        stacked = np.array(self.frames, dtype=self.dtype)
        return stacked


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    

class EnhancedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EnhancedRewardWrapper, self).__init__(env)
        
        self.real_reward = 0
        self.fire_action = 1

    def reset(self, **kwargs):
        self.real_reward = 0
        obs, info = self.env.reset(**kwargs)
        # Perform FIRE action
        obs, reward, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            # Optionally, perform FIRE action again
            obs, reward, terminated, truncated, info = self.env.step(self.fire_action)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_reward= 0
        self.real_reward += reward 
        #print(f"Modified Reward: {modified_reward} | Terminated: {terminated} | Truncated: {truncated}")
        return obs, reward, terminated, truncated, info



def make_env(env):
    print("Standard Env.        : {}".format(env.observation_space.shape))
    
    # Apply MaxAndSkipEnv first
    env = MaxAndSkipEnv(env)
    print("MaxAndSkipEnv        : {}".format(env.observation_space.shape))
    
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    video_path=os.path.join(script_dir, "new_run","videos")
    
    os.makedirs(video_path, exist_ok=True)  # Ensure the directory exists

    env = RecordVideo(
        env,
        video_folder=video_path,
        episode_trigger=lambda episode_id: (episode_id % 1500 == 0),  # Record every episode
        name_prefix="Kaboom_Episode"
    )

    print("RecordVideo          : Applied")

    # Apply EnhancedRewardWrapper with integrated FIRE action
    env = EnhancedRewardWrapper(env) 

    env = ProcessFrame84(env)
    print("ProcessFrame84       : {}".format(env.observation_space.shape))
    
    env = BufferWrapper(env, 4)
    print("BufferWrapper        : {}".format(env.observation_space.shape))
    
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    env = EnvCompatibility(env)
    print("EnvCompatibility     : Applied")

    
    return env

def print_env_info(name, env):
    obs,info = env.reset()
    print("*** {} Environment ***".format(name))
    print("Observation shape: {}, type: {} and range [{},{}]".format(obs.shape, obs.dtype, np.min(obs), np.max(obs)))
    print("Observation sample:\n{}".format(obs))