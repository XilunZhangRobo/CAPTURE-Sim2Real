# env name: push_one or scoop
# method: ad for CAPTURE, ed for expert distillation(baselines)
# transit method: search_beta_5 for randomized binary search, you can try others to reproduce ablation results
# n_data: how many datapoints that we collected during data generation
# n_data_eval: how many evaluation simulation parameters that we collected. (it is for evaluation logging during training)
# n_envs: how many parallel environment do you want to run (the more the faster)
# batch_sisz: transformer batch size
# evaluation_interval: how often we use evaluation data to evaluate the model performance during training
# loss_fn: loss function for next-token prediction. You can try something else and see the performance difference
# relative_positive: whether to use relative position embedding
# policy_name: we have tested with SAC policy, you can try PPO, random. 
# eval_policy_name: align with trained policy name
# -w: whether to log to wandb



python -m scripts.sim2real \
    --seed 1 \
    --env_name push_one \
    --method ad  \
    --transit_method search_beta_5 \
    --n_data 80000 \
    --n_data_eval 100 \
    --n_envs 64 \
    --batch_size 2048 \
    --evaluation_interval 5 \
    --loss_fn next \
    --relative_position \
    --policy_name sac \
    --eval_policy_name auto \
    --policy_checkpoint data/policy_model/push_one/sac/1/model_checkpoint_144000_steps.zip \
    -w 
