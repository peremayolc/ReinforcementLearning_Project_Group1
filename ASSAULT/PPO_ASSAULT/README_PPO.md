# ReinforcementLearning_Project_Group1
#### By Joan Bayona Corbalán (1667446), Pere Mayol Carbonell (1669503) and Andreu Gascón Marzo (1670919)

# FOR ASSAULT
## In here you will find information on how to run the PPO part of the ASSAULT environment. The specific requirements can be found in the [REQUIREMENTSFILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/PPO_ASSAULT/requirements.txt)

#### For the PPO model used to solve the Assault environment, there are various important parts, including the final model, the videos of the trained model and also the trials done during the different stages of the project.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THIS IS THE ONLY PART OF THE CODE THAT NEEDS TO BE RUN, other files contain the same concepts but are simply trials for tuning the hyperparameters and checking needed parts of how the enviroment behaves.

The first important thing would be the folder where you can find the final model, the best performing and the one that resulted as the ultimate goal for us. 
Here you can find the all files used and that resulted in the training process. 
- The [PYTHON FILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/PPO_ASSAULT/FINAL/assault_ppo_scheduler_test3_longest.py) containing all the code used.
- The [TRAINING PLOT](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/PPO_ASSAULT/FINAL/training_plot_long2048.png).
- The [MODEL](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/PPO_ASSAULT/FINAL/ASSAULT_PPOlong2048.zip).
- The [JOB FILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/PPO_ASSAULT/FINAL/job-55865.log) containing all the training process information saved along the way.
  
Moreover, we can find a folder where the performance of the saved final model has been tested -> [FOLDER TEST VIDEOS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/tree/main/ASSAULT/PPO_ASSAULT/FINAL_SCHEDULER_VIDEOS_TEST). Here we can find the [VIDEOS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/tree/main/ASSAULT/PPO_ASSAULT/FINAL_SCHEDULER_VIDEOS_TEST/videos) for the trained and untrained model, for comparison and ensure good performance, along with the [PYTHON FILE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/blob/main/ASSAULT/PPO_ASSAULT/FINAL_SCHEDULER_VIDEOS_TEST/TEST_PPOASSAULT.py) used to generate them.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Finally, we can also find two folders containing most of the trials done along the way, they are separated into two different folders:
1. A folder containing the [INITIAL TRIALS](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/tree/main/Pere/PPO_ASSAULT/FIRST_TRIALS), containing the training plots used to determine which the best hyperparameters were and before using the learning scheduler technique to avoid instability during training.
2. A folder containing the [TRIALS USING LEARNING SCHEDULE](https://github.com/peremayolc/ReinforcementLearning_Project_Group1/tree/main/Pere/PPO_ASSAULT/TRIALS_SCHEDULER). Here there are two other models, those that came before the ultimate one and that started getting better and more stable results.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
