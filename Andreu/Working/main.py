import torch  # For device selection
import collections  # For namedtuple to store experiences
import gymnasium as gym  # For creating the Kaboom environment
from wrappers import make_env  # For environment wrapping
from tuning import hyperparameter_tuning  # For hyperparameter optimization
import gymnasium as gym
import ale_py
from ale_py import ALEInterface

#set the environment as Kaboom
env = gym.make("ALE/Kaboom-v5")

# Optional: Define or import `hyperparameter_space` if not already defined
hyperparameter_space = {
    "learning_rate": [0.0005],
    "batch_size": [64],
    "gamma": [0.99],
    "epsilon_decay": [0.999985],
    "target_update_freq": [500],
}


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('cuda')
else:
    device = torch.device("cpu")

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state','td_error'])


#set the environment as Kaboom
env = gym.make("ALE/Kaboom-v5", obs_type="grayscale", render_mode = 'rgb_array')

target_reward=200

DQN_dir = "./Andreu/Working/Runs"

print(gym.__version__)

test_env = make_env(env)


best_DQN_model, best_hyperparameters, best_DQN_loss_history, best_DQN_avg_reward_history = hyperparameter_tuning(
    env=test_env,
    hyperparameter_space=hyperparameter_space,  # Maximum number of frames
    target_reward=target_reward,# Reward threshold to consider the task solved
    save_dir=DQN_dir,  # Directory to save the best model
)