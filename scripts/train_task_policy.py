import os
import time
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from gym_env.utilities import ENV_CONTEXTS, SAC_CONFIG, PPO_CONFIG
from gym_env.utilities import RandomContextStateWrapper, NormalizeActionSpaceWrapper
from scripts.utilities import set_seed


def make_env(env_name, dr):
    def _make_env():
        Env = ENV_CONTEXTS[env_name]["constructor"]
        env = Env(render=False)
        env = RandomContextStateWrapper(env, env_name, dr)
        env = NormalizeActionSpaceWrapper(env)
        env = Monitor(env)
        return env
    return _make_env


def main(args):
    # Set the random seed for reproducibility
    set_seed(args.seed)

    env = SubprocVecEnv([make_env(args.env_name, args.dr) for _ in range(args.n_envs)])

    # Initialize Weights and Biases (W&B) for experiment tracking
    run = wandb.init(
        project=f"train_task_policy_{args.env_name}",
        name=f'{args.policy_name}-{time.strftime("%Y-%m-%d-%H-%M")}',
        sync_tensorboard=True,
        config=args,
    )

    # Select the appropriate policy and its configuration
    if args.policy_name == "ppo":
        Policy = PPO
        param = PPO_CONFIG[args.env_name]
    elif args.policy_name == "sac":
        Policy = SAC
        param = SAC_CONFIG[args.env_name]
    else:
        raise ValueError(f"Unsupported policy {args.policy_name}")

    # Create the directory to save the model
    save_path = f"data/policy_model/{args.env_name}/{args.policy_name}/{args.seed}/"
    os.makedirs(save_path, exist_ok=True)

    # setup tensorboard log
    tensorboard_log = f"data/tensorboard_log/{run.id}"

    # Setup callbacks for checkpointing and W&B
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path=save_path, name_prefix="model_checkpoint"
    )
    wandb_callback = WandbCallback(verbose=2)
    callback = CallbackList([checkpoint_callback, wandb_callback])

    # Initialize the model and start learning
    if args.pre_trained_model:
        model = Policy.load(args.pre_trained_model, env, verbose=1, **param)
    else:
        model = Policy(
            "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, **param
        )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
    )


if __name__ == "__main__":
    # Define available environment names from the context
    env_names = list(ENV_CONTEXTS.keys())

    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, choices=env_names, help="Environment name"
    )
    parser.add_argument(
        "--policy_name", type=str, choices=["ppo", "sac"], help="Policy name"
    )
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=150000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--save_freq", type=int, default=150, help="Checkpoint interval"
    )
    parser.add_argument(
        "--pre_trained_model",type=str, default=None, help="Pre-trained model path"
    )
    
    parser.add_argument(
        "--dr", action="store_true", help="Enable domain randomization"
    )
    # Parse arguments and call the main function
    args = parser.parse_args()
    main(args)
