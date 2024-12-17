1. DQN_base, DQN_enhanced, DQN_modified folders contain all the .py files used in training
    and their structure is exactly the same.

2. Runing main.py from each folder creates a 'new_run' folder containing a 'videos' folder
    with videos after n desired episodes (1500 as base stated in wrappers.py) and 'experiment_n'
    folder that will contain the trained model with the given hyperparameters (stated in main.py)
    and all the training data in a json file.
	*notice that videos is emepty because GitHub doesn't allow much storage capacity, instead
	some videos will be available in the report through youtube links

3. It also creates a wandb folder to store all the logs (offline) from the system. It maybe asks
    you to login to your wandb account and paste your key in the terminal.

4. Logs from the terminal can be seen in the wandb folder 'logs'
_____________________________________________________________________________________________

--------------DQN_base (working) -> folder using basic DQN-----------------------

This folder contains some Run with different hyperparameters and all the .py files used during
training.

    1. agent.py -> define the Agent class
    2. auxiliar.py -> some auxiliar functions to save and plot data
    3. buffer.py -> define ExperienceReplay
    4. main.py -> file to run all the others
    5. model.py -> define the neural network class
    6. train.py -> training loop function
    7. tuning.py -> function to run all hyperparameters combinations 
    8. wrappers.py -> gym wrappers used

_____________________________________________________________________________________________

--------------DQN_enhanced (working) -> folder using PER + DDQN + modified Reward---

This folder contains some Run with different hyperparameters and all the .py files used during
training.

    1. agent.py -> define the Agent class
    2. auxiliar.py -> some auxiliar functions to save and plot data
    3. buffer.py -> define ExperienceReplay and SumTree
    4. main.py -> file to run all the others
    5. model.py -> define the neural network class
    6. train.py -> training loop function with DDQN
    7. tuning.py -> function to run all hyperparameters combinations 
    8. wrappers.py -> gym wrappers used with the EnhanceWrapper Class

_____________________________________________________________________________________________

--------------DQN_modified (working)-> folder using PER and DDQN-----------------------

This folder contains some Runs with different hyperparameters and all the .py files used during
training.

    1. agent.py -> define the Agent class
    2. auxiliar.py -> some auxiliar functions to save and plot data
    3. buffer.py -> define ExperienceReplay and SumTree class
    4. main.py -> file to run all the others
    5. model.py -> define the neural network class
    6. train.py -> training loop function with DDQN
    7. tuning.py -> function to run all hyperparameters combinations 
    8. wrappers.py -> gym wrappers used

_____________________________________________________________________________________________

--------------plots -> folder containing plots for the report-----------------------

This folder is not organized since it's unique purpose was doing plots for the report using
csv's from wandb

_____________________________________________________________________________________________

--------------test -> folder containing test file and models to test-----------------------

By running test.py, models from 'models' folder will start a testing over 70 episodes.
Reward in Each episode will be saved in a file with the final mean average reward.
videos each 35 steps will be saved in 'new_run' 'videos' folder