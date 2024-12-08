import torch  # For tensor operations
import torch.nn as nn  # For defining the neural network
import torch.optim as optim  # For optimizers
import numpy as np  # For numerical operations
import wandb  # For experiment tracking and logging
from buffer import ExperienceReplay  # For storing and sampling experiences
from agent import Agent  # For interacting with the environment
from model import DQN  # Neural network model
from wrappers import make_env  # For environment preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMBER_OF_REWARDS_TO_AVERAGE = 10

EXPERIENCE_REPLAY_SIZE = 10000

SYNC_TARGET_NETWORK = 500 # syncronize target neuron after 1000 steps

# greedy policy
EPS_START = 1.0
EPS_MIN = 0.02


def train_model(env, hyperparameters, target_reward):
    # Unpack hyperparameters
    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    epsilon_decay = hyperparameters["epsilon_decay"]
    gamma = hyperparameters["gamma"]

    # Initialize wandb for this run
    wandb.init(
        project="DQN_Kaboom",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epsilon_decay": epsilon_decay,
            "gamma": gamma,
            "target_reward": target_reward,
            "experience_replay_size": EXPERIENCE_REPLAY_SIZE,
            "sync_target_network": SYNC_TARGET_NETWORK,
            "eps_start": EPS_START,
            "eps_min": EPS_MIN,
            "number_of_rewards_to_average": NUMBER_OF_REWARDS_TO_AVERAGE,
        }
    )

    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
    agent = Agent(env, buffer)

    epsilon = EPS_START
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_number = 0
    loss_history = []
    avg_reward_history = []

    done = False
    # Optional: Watch the model
    wandb.watch(net, log="all", log_freq=10)

    while len(total_rewards) < 10000:
        frame_number += 1
        epsilon = max(epsilon * epsilon_decay, EPS_MIN)
    
        # Get immediate and cumulative rewards
        immediate_reward, cumulative_reward = agent.step(net, epsilon=epsilon, device=device)

        #print(f"Step: {frame_number} | Immediate Reward: {immediate_reward}")

        # Log immediate reward to wandb
        wandb.log({
            "immediate_reward": immediate_reward,
            "frame_number": frame_number,
        })

        # If an episode ends, log cumulative reward
        if cumulative_reward is not None:
            print('_______________________________Episode ', len(total_rewards)+1,' Ended with Cumulative Reward:______________________________', cumulative_reward)
            total_rewards.append(cumulative_reward)

            mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])

            wandb.log({
                "cumulative_reward": cumulative_reward,
                "episode": len(total_rewards),
                "mean_reward": mean_reward,
                "epsilon": epsilon,
                "frame_number": frame_number,
            })

            if len(total_rewards) % 10 == 0 or mean_reward > target_reward:
                print(f"Frame:{frame_number} | Total games:{len(total_rewards)} | Mean reward: {mean_reward:.3f} (epsilon: {epsilon:.2f})")

            avg_reward_history.append(mean_reward)

            if mean_reward > target_reward:
                #print(f"SOLVED in {frame_number} frames and {len(total_rewards)} games")
                break

        if len(buffer) < batch_size:
            continue

        # Sample a batch
        try:
            states, actions, rewards, dones, next_states, is_weights, indices = buffer.sample(batch_size)
        except ValueError as e:
            print(f"Sampling error: {e}")
            continue

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to(device)

        # Compute current Q values
        Q_values = net(states).gather(1, actions)

        with torch.no_grad():
            # Compute next Q values from target network
            next_Q_values = target_net(next_states).max(1, keepdim=True)[0]
            next_Q_values[dones] = 0.0

        # Compute expected Q values
        expected_Q_values = rewards + (gamma * next_Q_values)

        # Compute TD errors
        td_errors = Q_values - expected_Q_values

        # Compute loss with importance-sampling weights
        loss = (td_errors.pow(2) * is_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Log loss to wandb
        wandb.log({
            "loss": loss.item(),
            "frame_number": frame_number,
        })

        # Update priorities in the buffer
        buffer.update_priorities(indices, td_errors.detach().cpu().numpy().flatten())

        # Synchronize target network
        if frame_number % SYNC_TARGET_NETWORK == 0:
            target_net.load_state_dict(net.state_dict())

    # Finish the wandb run
    wandb.finish()

    

    return net, np.mean(total_rewards[-100:]), loss_history, avg_reward_history

    

