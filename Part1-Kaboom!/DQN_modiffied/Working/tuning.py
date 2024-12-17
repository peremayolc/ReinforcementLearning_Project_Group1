import os  # For file and directory handling
import json  # For saving hyperparameters and training data
import torch  # For saving and loading models
from itertools import product  # For generating hyperparameter combinations

# Import custom functions from your project
from train import train_model  # Training and saving data functions
from auxiliar import save_training_data # For plotting training results

def hyperparameter_tuning(env, hyperparameter_space, target_reward, save_dir):
    # Create directory to save models
    os.makedirs(save_dir, exist_ok=True)

    best_model = None
    best_score = -float("inf")
    best_hyperparameters = None
    best_experiment_index = None

    # Generate all combinations of hyperparameters
    keys = list(hyperparameter_space.keys())  # Extract hyperparameter names
    values = list(hyperparameter_space.values())  # Extract hyperparameter values

    # Use itertools.product to create all combinations of hyperparameter values
    combinations = product(*values)  # Returns tuples of combinations

    # Convert each combination into a dictionary with corresponding hyperparameter names
    hyperparameter_combinations = []
    for combination in combinations:
        hyperparameter_dict = dict(zip(keys, combination))
        hyperparameter_combinations.append(hyperparameter_dict)


    for i, hyperparameters in enumerate(hyperparameter_combinations):
        experiment_dir = os.path.join(save_dir, f"experiment_{i + 1}")
        os.makedirs(experiment_dir, exist_ok=True)

        print(f"Experiment {i + 1}/{len(hyperparameter_combinations)}: {hyperparameters}")
        model, mean_reward, loss_history, avg_reward_history = train_model(env, hyperparameters, target_reward)

        print(f"Mean Reward: {mean_reward:.2f}")
        #plot_results(hyperparameters, loss_history, avg_reward_history, mean_reward, experiment_dir)

        model_path = os.path.join(experiment_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"New model saved with reward {mean_reward:.2f}")

        hyperparameters_save_path = os.path.join(experiment_dir, "hyperparameters.json")
        with open(hyperparameters_save_path, "w") as f:
            json.dump(hyperparameters, f, indent=4)
        print(f"Hyperparameters saved: {hyperparameters}")

        save_training_data(experiment_dir, hyperparameters, avg_reward_history, loss_history)

        # Save the model if it's the best so far
        if mean_reward > best_score:
            best_score = mean_reward
            best_model = model
            best_hyperparameters = hyperparameters

            best_loss_history = loss_history
            best_avg_reward_history = avg_reward_history
            best_experiment_index = i + 1

    best_model_info_path = os.path.join(save_dir, "best_model_info.txt")
    with open(best_model_info_path, "w") as f:
        f.write(f"Best model is experiment {best_experiment_index}\n")
        f.write(f"Best hyperparameters: {best_hyperparameters}\n")
        f.write(f"Best mean reward: {best_score:.2f}\n")
    print(f"Best model information saved at: {best_model_info_path}")
    print(f"Best Hyperparameters: {best_hyperparameters}")
    print(f"Best Mean Reward: {best_score:.2f}")

    return best_model, best_hyperparameters, best_loss_history, best_avg_reward_history