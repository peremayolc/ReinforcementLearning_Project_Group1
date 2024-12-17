import numpy as np  
import torch
from buffer import ExperienceReplay
import collections
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state','td_error'])

class Agent:
    def __init__(self, env, exp_replay_buffer):
        """
        Initialize the agent.

        Args:
            env: The environment the agent interacts with.
            exp_replay_buffer: ExperienceReplay buffer to store experiences.
        """

        self.env = env  # Reference to the environment

        self.exp_replay_buffer = exp_replay_buffer  # Replay buffer for storing experiences

        self._reset()  # Reset agent state and rewards


    def _reset(self):
        """
        Resets the agent's current state by resetting the environment.
        """

        self.current_state, info = self.env.reset()  # Unpack the tuple

        self.total_reward = 0.0

    def step(self, net, epsilon=0.0, device="cpu"):
        """
        Perform a single step in the environment, choosing an action based on an epsilon-greedy policy.

        Args:
            net: The DQN network for predicting Q-values.
            epsilon: Probability of selecting a random action (for exploration).
            device: Device to perform computations.

        Returns:
            done_reward (float or None): Total reward if the episode ends; otherwise, None.
        """

        done_reward = None  # Initialize reward for the end of the episode

        if np.random.random() < epsilon:  # With probability epsilon, take a random action

            action = self.env.action_space.sample()  # Select random action for exploration

        else:
            state_ = np.array([self.current_state])  # Prepare state as a batch (1, state_dim)

            state = torch.tensor(state_).to(device)  # Convert to tensor and send to device

            q_vals = net(state)  # Get Q-values for the current state

            _, act_ = torch.max(q_vals, dim=1)  # Select the action with the highest Q-value

            action = int(act_.item())  # Convert action from tensor to integer


        # Execute the action in the environment
        try:
            new_state, reward, terminated, truncated, info = self.env.step(action)
            is_done = terminated or truncated

        except ValueError:
            # Fallback for environments returning 4 values
            new_state, reward, done, info = self.env.step(action)
            is_done = done
            terminated = done
            truncated = False
            info = info

        self.total_reward += reward
        exp = Experience(self.current_state, action, reward, is_done, new_state, td_error=1.0)
        self.exp_replay_buffer.append(exp) # store experience to the buffer
        self.current_state = new_state # actualize state

        # reset if episode terminated
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return reward, is_done, done_reward