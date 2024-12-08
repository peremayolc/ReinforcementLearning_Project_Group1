import gymnasium as gym  # Environment management and wrappers
import collections  # Data structures for buffers
import numpy as np  # Numerical operations
import cv2  # Image processing
import os
from gymnasium.wrappers import RecordVideo

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
        obs, info = self.env.reset(**kwargs)
        # Perform FIRE action if needed at the start
        obs, reward, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Do not step twice here
        return self.env.step(action)

    
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
    def __init__(self, env, initial_lives, penalty, bonus,
                 level_ascend_reward,
                 explosion_cooldown):
        super(EnhancedRewardWrapper, self).__init__(env)
        self.initial_lives = initial_lives
        self.lives = initial_lives
        self.penalty = penalty
        self.bonus = bonus
        self.level_ascend_reward = level_ascend_reward
        self.explosion_cooldown = explosion_cooldown
        self.explosion_timer = 0
        self.previous_frame = None
        self.previous_score = 0
        self.current_level = 1
        self.bombs_cleared = 0
        self.bomb_caught = False  # Track if a bomb was caught in the current level
        self.real_reward = 0
        

        # Bomb counts per level based on the table
        self.level_bomb_counts = [10, 20, 30, 40, 50, 75, 100, 150]

        self.fire_action = 1  # Assuming action '1' is the FIRE action

    def reset(self, **kwargs):
        # Reset internal lives count
        self.lives = self.initial_lives
        self.bombs_cleared = 0
        self.bomb_caught = False
        self.current_level = 1
        self.explosion_timer = 0
        self.previous_frame = None
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
        total_reward = 0.0
        self.real_reward += reward 
          # Corrected to print actual reward

        # Process frame to detect explosion
        current_frame = self.preprocess_frame(obs)

        # Detect explosions and handle life loss
        explosion = False
        if self.explosion_timer <= 0:
            explosion = self.detect_explosion(self.previous_frame, current_frame)

        if explosion:
            self.lives -= 1
            total_reward += self.penalty
            print(f"Explosion detected! Lives remaining: {self.lives}")

            if self.lives > 0:
                # Reset to mid-level if life is lost
                self.reset_to_mid_level()
                print(f"Life lost. Resuming mid-level at bomb count: {self.bombs_cleared}")
                # Override terminated flag to prevent episode from ending
            else:
                # Terminate episode if no lives left
                print("No lives left. Episode terminated.")
                print('GAME SCORE: ', self.real_reward)
                self.lives = self.initial_lives  # Reset lives for the next episode

            self.explosion_timer = self.explosion_cooldown
            #print(f"Explosion Timer Reset to: {self.explosion_timer}")

        # Update explosion cooldown
        if self.explosion_timer > 0:
            self.explosion_timer -= 1
            #print(f"Explosion Timer Decremented to: {self.explosion_timer}")

        # Increment bombs cleared if a bomb was caught
        if reward > 0:  # Assuming catching a bomb gives a positive reward
            self.bombs_cleared += 1
            self.bomb_caught = True  # Set flag to True after the first bomb is caught
            print(f"Bomb Caught! Total Bombs Cleared: {self.bombs_cleared}")

        if not explosion:
            total_reward += self.bonus
            #print(f"Bonus Applied: {self.bonus}")

        # Check if all bombs in the current level are cleared
        if self.bombs_cleared == self.get_level_bomb_thresholds(self.current_level):
            total_reward += self.level_ascend_reward
            print(f"-------------Level Up! Ascended to Level-------------- {self.current_level+1}.")
            self.bomb_caught = False  # Reset flag after clearing all bombs

        # Calculate the current level
        previous_level = self.current_level
        self.current_level = self.calculate_level(self.bombs_cleared)
        if self.current_level != previous_level:
            print(f"Level Changed: {previous_level} -> {self.current_level}")

        self.previous_frame = current_frame

        info['lives'] = self.lives
        info['current_level'] = self.current_level
        info['bombs_cleared'] = self.bombs_cleared

        modified_reward = reward + total_reward
        #print(f"Modified Reward: {modified_reward} | Terminated: {terminated} | Truncated: {truncated}")
        return obs, modified_reward, terminated, truncated, info


    def calculate_level(self, bombs_cleared):
        """
        Calculate the current level based on bombs cleared.
        """
        total_bombs = 0
        for level, bomb_count in enumerate(self.level_bomb_counts, start=1):
            total_bombs += bomb_count
            if bombs_cleared < total_bombs:
                return level
        return len(self.level_bomb_counts)  # Max level if bombs cleared exceeds all thresholds

    def reset_to_mid_level(self):
        """
        Resets the bombs cleared to the midpoint of the previous level.
        """
        if self.current_level > 1:
            previous_level_bombs = sum(self.level_bomb_counts[:self.current_level - 1])
            mid_level_bombs = previous_level_bombs + (self.level_bomb_counts[self.current_level - 2] // 2)
            self.bombs_cleared = mid_level_bombs
        else:
            self.bombs_cleared = 0  # If Level 1, reset to the beginning

    def get_level_bomb_thresholds(self, level):
        """
        Get the total bomb threshold for the given level.
        """
        return sum(self.level_bomb_counts[:level])

    def preprocess_frame(self, frame):
        """
        Preprocess the frame for explosion detection.
        Convert to grayscale and resize if necessary.
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def detect_explosion(self, prev_frame, current_frame, threshold=50):
        """
        Detects an explosion by measuring the change between frames.
        """
        if prev_frame is None:
            return False
        diff = cv2.absdiff(current_frame, prev_frame)
        return np.mean(diff) > threshold



def make_env(env):
    print("Standard Env.        : {}".format(env.observation_space.shape))
    
    # Apply MaxAndSkipEnv first
    env = MaxAndSkipEnv(env)
    print("MaxAndSkipEnv        : {}".format(env.observation_space.shape))
    
    # Apply EnhancedRewardWrapper with integrated FIRE action
    env = EnhancedRewardWrapper(
        env, 
        initial_lives=3, 
        penalty=-2, 
        bonus=0.1,
        level_ascend_reward=+3,
        explosion_cooldown=7  # Adjust based on game mechanics
    )
    video_path = "./Andreu/Working/Runs/videos"
    os.makedirs(video_path, exist_ok=True)  # Ensure the directory exists

    env = RecordVideo(
        env,
        video_folder=video_path,
        episode_trigger=lambda episode_id: True,  # Record every episode
        name_prefix="Kaboom_Episode"
    )

    print("RecordVideo          : Applied")
    
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