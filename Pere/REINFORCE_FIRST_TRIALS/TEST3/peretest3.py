import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gymnasium as gym
from IPython.display import display, Image as IPImage
import time
import numpy as np
import collections
import torch.optim as optim
from torchsummary import summary
from itertools import product
import os
import json

import gymnasium as gym

import cv2

import ale_py
from ale_py import ALEInterface

#set all the needed gym wrappers.
#checks for compatibility of the step return
class EnvCompatibility(gym.Wrapper):
    def step(self, action):
        results = self.env.step(action)
        if len(results) == 4:
            observation, reward, done, info = results
            #
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
            return results, {}

class FireResetEnv(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        results = self.env.step(1)
        if len(results) == 5:
            obs, _, term, trunc, info = results
            done = term or trunc
        elif len(results) == 4:
            obs, _, done, info = results
        else:
            raise ValueError("Unexpected number of return values from env.step()")
        if done:
            self.env.reset(**kwargs)
        results = self.env.step(2)
        if len(results) == 5:
            obs, _, term, trunc, info = results
            done = term or trunc
        elif len(results) == 4:
            obs, _, done, info = results
        else:
            raise ValueError("Unexpected number of return values from env.step()")
        if done:
            self.env.reset(**kwargs)
        return obs, info



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self.obs_buffer = collections.deque(maxlen=2)  
        self.skip = skip  

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):  
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
        resized_screen = cv2.resize(frame, (84, 110), interpolation=cv2.INTER_AREA)
        cropped_frame = resized_screen[18:102, :] 
        return cropped_frame.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        self.frames = collections.deque(maxlen=n_steps)
        old_space = env.observation_space

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_steps, old_space.shape[0], old_space.shape[1]),
            dtype=dtype,
        )

    def reset(self, **kwargs):
        self.frames.clear()
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_steps):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self.observation(obs), info

    def observation(self, observation):
        self.frames.append(observation)
        return np.array(self.frames, dtype=self.dtype)



class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

#function to wrap environment using all the wrappers created before.
def make_env(env):
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipEnv(env)
    print("MaxAndSkipEnv        : {}".format(env.observation_space.shape))
    env = FireResetEnv(env)
    print("FireResetEnv         : {}".format(env.observation_space.shape))
    env = ProcessFrame84(env)
    print("ProcessFrame84       : {}".format(env.observation_space.shape))
    env = BufferWrapper(env,4)
    print("BufferWrapper        : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    env = EnvCompatibility(env)
    return env

#print environment information
def print_env_info(name, env):
    obs,info = env.reset()
    print("*** {} Environment ***".format(name))
    print("Observation shape: {}, type: {} and range [{},{}]".format(obs.shape, obs.dtype, np.min(obs), np.max(obs)))
    print("Observation sample:\n{}".format(obs))

class policy_net(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(policy_net, self).__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        #the fully connected layers
        #use funbction to calculate the size of the layer that the convolutions output.
        conv_output_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

        #use xavier initialization
        self._initialize_weights()

    def _get_conv_output_size(self, input_shape):
        #function to get the size at a certain convolution layer.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.numel()

    def _initialize_weights(self):
        #function to initialise the weights based on the xavier uniform idea.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, x):
        #forward pass
        #convolution layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        #flatten and the fully connected
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        #convert the logarithms into probabilities.
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs


#funciton to train the reinforce agent.
def train_reinforce(env, policy_net, optimizer, gamma, DEVICE):
    #set a number of episodes and the target reward desired.
    max_episodes = 6000
    target_reward = 20
    
    #to store results along the way.
    total_rewards = []
    avg_reward_history = []
    
    #for every episode:
    for episode in range(max_episodes):
      #reset the environment and generate lists to store the log_probs and the rewards for each episode.
      current_state, _ = env.reset()
      log_probs = []
      rewards = []
      total_reward = 0
    
        #while the episode runs.
      while True:
          #convert the state to a tensor and get it to the device.
          state_tensor = torch.tensor([current_state], dtype=torch.float32).to(DEVICE)
          #get the probabiliy of each action based on the policy and the state.
          action_probs = policy_net(state_tensor)
          #get action distribuition based on their probabilities.
          action_dist = torch.distributions.Categorical(action_probs)
          #select an actio from the distribution.
          action = action_dist.sample()
          log_prob = action_dist.log_prob(action)
          #generate new state based on chosen action
          new_state, reward, terminated, truncated, _ = env.step(action.item())
          is_done = terminated or truncated #check if episode is done.
          #add all relevant information to set lists.
          log_probs.append(log_prob)
          rewards.append(reward)
          total_reward += reward 
          current_state = new_state

          if is_done:
              total_rewards.append(total_reward)
              break

      #compute the discounted rewards
      discounted_rewards = []
      G = 0
      for reward in reversed(rewards):
          G = reward + gamma * G
          discounted_rewards.insert(0, G)
      discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(DEVICE)

      #normalize the rewards
      discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

      #calculate and apply policy loss. Update the network.
      policy_loss = []
      for log_prob, G in zip(log_probs, discounted_rewards):
          policy_loss.append(-log_prob * G)
      policy_loss = torch.cat(policy_loss).sum()
    
      optimizer.zero_grad()
      policy_loss.backward()
      optimizer.step()

      #get the mean reward
      mean_reward = np.mean(total_rewards[-10:])
      avg_reward_history.append(mean_reward)

      print(f"Episode {episode}: Total Reward = {total_reward}, Mean Reward 10 episodes= {mean_reward:.3f} Loss: {policy_loss:.3f}")
        #check if mean_reward is good enough.
      if mean_reward >= target_reward:
          print(f"Target reward achieved! Solved in {episode + 1} episodes.")
          break
    return policy_net, total_rewards, avg_reward_history

#set the environment as Kaboom
env = gym.make("ALE/Kaboom-v5", obs_type="grayscale")
env = make_env(env)


#configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#train using the hyperparameters we want to explore.
input_shape = env.observation_space.shape
action_dim = env.action_space.n
gammas=[0.99]
lrs =[0.00001, 0.000005]

for learning_rate in lrs:
  for gamma in gammas:
    policy_net_model = policy_net(input_shape, action_dim).to(device)
    optimizer = optim.Adam(policy_net_model.parameters(), lr=learning_rate)
    print("Training the REINFORCE Agent...")
    trained_policy, total_rewards, avg_reward_history = train_reinforce(
      env,
      policy_net_model,
      optimizer,
      gamma,
      DEVICE=device
    )

    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward_history, label="Average Reward (last 100 episodes)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Training Process")
    plt.legend()
    plt.show()

    plt.savefig(f'TRAINING-PROCESS[{learning_rate}][{gamma}].png')
    plt.close()