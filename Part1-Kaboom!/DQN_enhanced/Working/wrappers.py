import gymnasium as gym  # Environment management and wrappers
import collections  # Data structures for buffers
import numpy as np  # Numerical operations
import cv2  # Image processing
import os
from gymnasium.wrappers import RecordVideo
import os
import time
import wandb  # Import wandb
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
    def __init__(self, env=None, skip=7):
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
                 explosion_cooldown,
                 max_horizontal_distance=84):
        super(EnhancedRewardWrapper, self).__init__(env)
        self.initial_lives = initial_lives
        self.lives = initial_lives
        self.penalty = penalty
        self.max_horizontal_distance = max_horizontal_distance

        self.level_ascend_reward = level_ascend_reward
        self.explosion_cooldown = explosion_cooldown
        self.explosion_timer = 0
        self.previous_frame = None
        self.previous_score = 0
        self.current_level = 1
        self.bombs_cleared = 0
        self.bomb_caught = False  # Track if a bomb was caught in the current level

        # New attribute to track and expose mean reward
        self.mean_reward = 0
        self.last_10_rewards = collections.deque(maxlen=10)
        self.real_list = []

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

        self.real_reward += reward
        done = truncated or terminated

        wandb.log({
                "immidiate_real_reward": self.mean_reward,
            })
        if done:
            self.real_list.append (self.real_reward)
            self.mean_reward = np.mean(self.real_list[-10:])
            wandb.log({
                "real_reward": self.mean_reward,
            })

        
        total_reward = 0.0
          # Corrected to print actual reward

        # Process frame to detect explosion
        current_frame = self.preprocess_frame(obs)

        # Detect explosions and handle life loss
        explosion = False
        if self.explosion_timer <= 0:
            explosion = self.detect_explosion(self.previous_frame, current_frame)

        if explosion:
            total_reward += self.penalty
            #print(f"Explosion detected!")

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
            #print(f"Bomb Caught! Total Bombs Cleared: {self.bombs_cleared}")


         # Horizontal distance-based reward
        bomb_positions, bucket_positions = self.extract_position(current_frame)
        if not bomb_positions or not bucket_positions:
            distance_reward = 0
        else:
            min_dist, _, _ = self.min_horizontal_distance(bomb_positions, bucket_positions)
            distance_reward = self.compute_horizontal_distance_reward(min_dist)
            #print(f"Horizontal Distance Reward: {distance_reward:.2f}")
            total_reward += distance_reward


        self.previous_frame = current_frame

        info['lives'] = self.lives
        info['current_level'] = self.current_level
        info['bombs_cleared'] = self.bombs_cleared

        modified_reward = reward + total_reward
        #print(f"Modified Reward: {modified_reward} | Terminated: {terminated} | Truncated: {truncated}")
        return obs, modified_reward, terminated, truncated, info

    def preprocess_frame(self, frame):
        """
        Preprocess the frame for explosion detection.
        Convert to grayscale, resize, and crop to 84x84.
        """
        # Convert to grayscale if the frame has color channels
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        # Resize the grayscale frame to (84, 110)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)

        # Crop the resized frame to (84, 84) by removing the top 18 pixels
        cropped = resized[18:102, :]  # Rows 18 to 101 (inclusive), all columns

        return cropped.astype(np.uint8)


    def detect_explosion(self, prev_frame, current_frame, threshold=50):
        """
        Detects an explosion by measuring the change between frames.
        """
        if prev_frame is None:
            return False
        diff = cv2.absdiff(current_frame, prev_frame)
        explosion = np.mean(diff) > threshold
        return explosion

    def extract_position(self, frame):
        """
        Updated function for detecting bombs and buckets, using different dilation kernels
        for the top 70% and bottom 30% of the image.
        """
        bomb_positions = []
        bucket_region = []

        # Apply binary threshold
        _, threshold = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)

        # Split the image into two regions
        height, width = frame.shape
        bottom_region_start = int(height * 0.70)  # Define the bottom 30% as the bucket region

        # Top 70% dilation with a 2x2 kernel
        top_threshold = threshold[:bottom_region_start, :]
        kernel_top = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_top = cv2.dilate(top_threshold, kernel_top, iterations=1)

        # Bottom 30% dilation with a 5x5 kernel
        bottom_threshold = threshold[bottom_region_start:, :]
        kernel_bottom = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_bottom = cv2.dilate(bottom_threshold, kernel_bottom, iterations=1)

        # Combine the two regions
        dilated = np.vstack((dilated_top, dilated_bottom))

        # Find contours in the combined dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the frame to draw on
        frame_with_contours = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        # Process the bottom 30% for buckets
        bucket_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Check if the contour is in the bottom region
            if y >= bottom_region_start:
                bucket_contours.append(cnt)

        # Create a single large bounding box that covers all bucket contours
        if bucket_contours:
            x_min = min(cv2.boundingRect(cnt)[0] for cnt in bucket_contours)
            y_min = bottom_region_start  # Fixed to start at the bottom region
            x_max = max(cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in bucket_contours)
            y_max = max(cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in bucket_contours)

            # Add the single bounding box for buckets
            bucket_region = [(x_min, y_min), (x_max, y_max)]
            # Draw a blue rectangle around the entire bucket region
            cv2.rectangle(frame_with_contours, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Process the rest of the frame for bombs
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Detect bombs (outside the bottom region)
            if y < bottom_region_start and 0.5 <= aspect_ratio <= 0.8 and 3 <= h <= 15 and 3 <= w <= 15:  
                bomb_positions.append((x + w // 2, y + h // 2))
                # Draw a red rectangle around the bomb
                cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return bomb_positions, bucket_region


    
    def min_horizontal_distance(self, bomb_positions, bucket_positions):
        """
        Calculate the horizontal distance to the bomb with the lowest y-position (most down).
        Returns the distance and the corresponding bomb and bucket positions.
        """
        max_y = -1  # Initialize with an impossible y-coordinate
        max_dist = 0
        lowest_bomb = None
        closest_bucket = None

        for bomb in bomb_positions:
            if bomb[1] > max_y:  # Check if this bomb has a lower y-coordinate
                max_y = bomb[1]
                lowest_bomb = bomb

        if lowest_bomb is not None:  # Ensure a bomb was found
            for bucket in bucket_positions: 
                dist = abs(lowest_bomb[0] - bucket[0])  # Horizontal distance
                if dist > max_dist:
                    max_dist = dist
                    closest_bucket = bucket 

        return max_dist, lowest_bomb, closest_bucket
        
    
    def compute_horizontal_distance_reward(self, min_dist):
        """
        Compute a reward based on the horizontal distance, scaled between -1 and 1.
        Closer distance yields higher reward.
        """
        # Normalize the distance
        normalized_dist = min_dist / self.max_horizontal_distance  # Scale between 0 and 1

        # Scale to -1 and 1
        distance_reward = -1 + (2 * (1 - normalized_dist))  

        return distance_reward



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
        episode_trigger=lambda episode_id: (episode_id% 1000==0),  # Record every episode
        name_prefix="Kaboom_Episode"
    )

    print("RecordVideo          : Applied")

    # Apply EnhancedRewardWrapper with integrated FIRE action
    env = EnhancedRewardWrapper(
        env, 
        initial_lives=3, 
        penalty=-2, 
        bonus=0.1,
        level_ascend_reward=+3,
        explosion_cooldown=7  # Adjust based on game mechanics
    )
    
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