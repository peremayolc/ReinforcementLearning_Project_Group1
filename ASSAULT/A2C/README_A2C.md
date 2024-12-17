# ReinforcementLearning_Project_Group1
#### By Joan Bayona Corbalán (1667446), Pere Mayol Carbonell (1669503) and Andreu Gascón Marzo (1670919)

# FOR ASSAULT
## In here you will find information on how to run the A2C part of the ASSAULT environment. The specific requirements can be found in the [REQUIREMENTSFILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/requirements.txt)

#### For the A2C model used to solve the Assault environment, there are various important parts, including all the models, the videos of the trained final model and also the plots of the final model.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Here you can find the all files and what they do. 
- The [PYTHON FILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/A2Ctrain.py) containing all the code used to train the agent.
- The [VIDEO RECORDER](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/A2Cvisualize.py) records a video of certain length and certain timesteps.
- The [EVALUATOR](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/A2Cevaluate.py) evaluates the model printing a list of rewards and the average reward, you decide the number of episodes.
- The [PLOTS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/plots.ipynb) is simply a notebook which was used to visualize the tensorboard JSON file (ep_len and ep_rew).
  
Moreover, we can find a folder where the performance of the saved final model has been tested -> [FOLDER TEST VIDEOS] 
(https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/videos). 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Finally, we can also find two folders containing most of the trials done along the way, they are separated into two different folders:
1. A folder containing the [INITIAL TRIALS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/tree/main/ASSAULT/PPO_ASSAULT/FIRST_TRIALS), containing the training plots used to determine which the best hyperparameters were and before using the learning scheduler technique to avoid instability during training.
2. A folder containing the [TRIALS USING LEARNING SCHEDULE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/tree/main/ASSAULT/PPO_ASSAULT/TRIALS_SCHEDULER). Here there are two other models, those that came before the ultimate one and that started getting better and more stable results.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
