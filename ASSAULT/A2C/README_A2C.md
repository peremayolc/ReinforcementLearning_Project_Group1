# ReinforcementLearning_Project_Group1
#### By Joan Bayona Corbalán (1667446), Pere Mayol Carbonell (1669503) and Andreu Gascón Marzo (1670919)

# FOR ASSAULT
## In here you will find information on how to run the A2C part of the ASSAULT environment. The specific requirements can be found in the [REQUIREMENTSFILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/requirements.txt)

#### For the A2C model used to solve the Assault environment, there are various important parts, including all the models, the videos of the trained final model and also the plots of the final model.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Here you can find the all files and what they do. 
- The [TRAINING FILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/A2Ctrain.py) containing all the code used to train the agent.
- The [VIDEO RECORDER](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/A2Cvisualize.py) records a video of certain length and certain timesteps.
- The [EVALUATOR](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/A2Cevaluation.py) evaluates the model printing a list of rewards and the average reward, you decide the number of episodes.
- The [PLOT](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/plots.ipynb) is simply a notebook which was used to visualize the tensorboard JSON file (ep_len and ep_rew).
  
Moreover, we can find a folder where the performance of the saved final model has been tested -> [videos](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/videos). 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Finally, we can also find 3 folders containing the plots of the final model (plots directly from tensorboard and one with matplotlib), the models trained and the logs of the 5.000.000 and 7.000.000 timesteps models:
1. A folder containing the [PLOTS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/plots), containing the training plots of the final model.
2. A folder containing the [MODELS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/models). Here there are 3 models, in the [A2C](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/models/A2C) folder we have the first trial with just 20.000 timesteps. In the [A2CCnnAssault](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/models/A2CCnnAssault) we have the 2 models trained in the cluster with their timesteps as name.
3. A folder containing the [logs](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/A2C/logs) from tensorboard of both models (5.000.000 and 7.000.000 timesteps).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
