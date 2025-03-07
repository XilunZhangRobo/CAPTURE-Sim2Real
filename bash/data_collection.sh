
# This file contains the bash scripts to train the policy for scoop environemnts and collect data for both environments
# You can change name from scoop -> push_one to test the air hockey performance

python -m scripts.train_task_policy --env_name scoop --policy_name sac --n_envs 64  --seed 1

# Once you have a trained policy, you can use the following command to collect data 
# Or you can use random action to collect data. 


# data collection for scoop with policy action and default hyperparameters. (You can swap policy_name with random to use random action)
# If you want to understand what is n_history, please checkout our paper and look for the defination of history length (L=7)
python -m scripts.collect_data --mode train --env_name scoop --n_history 7 --n_data 80000 --n_env 100 --policy_name sac --policy_checkpoint data/policy_model/push_one/sac/0/model_checkpoint_144000_steps.zip
python -m scripts.collect_data --mode eval  --env_name scoop --n_history 7 --n_data 100   --n_env 100 --policy_name sac --policy_checkpoint data/policy_model/push_one/sac/0/model_checkpoint_144000_steps.zip

