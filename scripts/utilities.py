import argparse
import os
import random
import torch
import numpy as np
import wandb
from copy import deepcopy
from matplotlib.figure import Figure
from gym_env.utilities import ENV_CONTEXTS
from stable_baselines3 import PPO, SAC
import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_parser():
    env_names = list(ENV_CONTEXTS.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", choices=env_names, help="environment name")
    parser.add_argument(
        "--n_data", type=int, default=64, help="number of sim2real pair for training"
    )
    parser.add_argument(
        "--n_history",
        type=int,
        default=7,
        help="number of iteration we need to transfer from sim to real", # This is from paper L = 7
    )
    parser.add_argument(
        "--n_traj",
        type=int,
        default=1,
        help="number of sim and real trajectories we sample for each sim context. Meaning we first set a context for the sim environment, rollout n trajectories, then we sample n trajectories again in the real environment with same actions.",
    )
    parser.add_argument(
        "--transit_method",
        type=str,
        default="search_beta_5",
        help="how the context changed from sim to real.",
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        default="normal",
        choices=["normal", "interp", "extrap", "cg"],
        help="How we sample the training set and evaluation set to mimic out-of distribution setting, like interpolation, extrapolation, compensation generation.",
    )
    parser.add_argument(
        "--policy_name",
        type=str,
        default="random",
        help="How do we choose action in each environment",
    )
    parser.add_argument(
        "--policy_checkpoint",
        type=str,
        default=None,
        help="The checkpoint that is used for policy to choose action",
    )
    parser.add_argument(
        "--dr", action="store_true", help="Enable domain randomization"
    )
    return parser


def get_dataset_dir(args):
    n_data = args.n_data
    n_traj = args.n_traj
    n_history = args.n_history
    env_name = args.env_name
    transit_method = args.transit_method
    sample_method = args.sample_method
    policy_name = args.policy_name
    mode = args.mode
    return f"data/capture_data/{mode}/{env_name}/{transit_method}/pair_{n_data}/trajectory_{n_traj}/history_{n_history}/{policy_name}/{sample_method}/"


def get_model_dir(args):
    n_data = args.n_data
    n_traj = args.n_traj
    n_history = args.n_history
    env_name = args.env_name
    transit_method = args.transit_method
    sample_method = args.sample_method
    policy_name = args.policy_name
    method = args.method
    loss_fn = args.loss_fn
    return f"data/sim2real_model/{method}/{loss_fn}/{env_name}/{transit_method}/pair_{n_data}/trajectory_{n_traj}/history_{n_history}/{policy_name}/{sample_method}/seed_{args.seed}"


def get_unwrapped_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def load_data(train_args, eval_args):
    train_dir = get_dataset_dir(train_args)
    eval_dir = get_dataset_dir(eval_args)
    # load training data
    train_contexts = np.load(os.path.join(train_dir, f"contexts.npy"))
    train_actions = np.load(os.path.join(train_dir, f"actions.npy"))
    train_real_trajectories = np.load(os.path.join(train_dir, f"real_trajectories.npy"))
    train_sim_trajectories = np.load(os.path.join(train_dir, f"sim_trajectories.npy"))
    # load evaluation data
    eval_contexts = np.load(os.path.join(eval_dir, f"contexts.npy"))
    
    ## manipulate trajectory for push bar env 
    if train_args.env_name == "push_bar":
        ## round the last dimension to one decimal
        
        modified_train_real_trajectories = np.zeros_like(train_real_trajectories)
        modified_train_sim_trajectories = np.zeros_like(train_sim_trajectories)
        threahold = np.radians(8)
        modified_train_real_trajectories[train_real_trajectories < -threahold] = -1
        modified_train_real_trajectories[train_real_trajectories > threahold] = 1
        modified_train_sim_trajectories[train_sim_trajectories < -threahold] = -1
        modified_train_sim_trajectories[train_sim_trajectories > threahold] = 1
        
        train_real_trajectories = modified_train_real_trajectories
        train_sim_trajectories = modified_train_sim_trajectories

    return (
        train_contexts,
        train_actions,
        train_real_trajectories,
        train_sim_trajectories,
        eval_contexts,
    )


def normalize_context(contexts, contexts_range):
    assert contexts_range.shape[0] == contexts.shape[0], f"contexts_range.shape:{contexts_range.shape}, contexts.shape: {contexts.shape}"
    lb = contexts_range[:, 0].reshape([-1, 1, 1])
    ub = contexts_range[:, 1].reshape([-1, 1, 1])
    norm_contexts = (contexts - lb) / (ub - lb)
    norm_contexts = norm_contexts * 2.0 - 1.0
    assert np.all((-1.0 <= norm_contexts) & (norm_contexts <= 1.0))
    return norm_contexts


def to_tensor(data, device, dtype=torch.float32):
    if isinstance(data, (list, tuple)):
        return torch.tensor(data).to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype, device=device)
    else:
        raise TypeError("Input should be a list, tuple, or NumPy array")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_task_policy(policy_name, policy_checkpoint=None):
    if policy_name.lower() == "ppo":
        assert policy_checkpoint is not None
        task_model = PPO.load(policy_checkpoint)
        # task_model.policy.eval()
        policy_args = {"model": task_model}
    elif policy_name.lower() == "sac":
        assert policy_checkpoint is not None
        # task_model = SAC("MlpPolicy", env, verbose=1)
        task_model = SAC.load(policy_checkpoint)
        # task_model.policy.eval()
        policy_args = {"model": task_model}
    elif policy_name.lower() == "random":
        policy_args = {}
    elif policy_name.lower() == "zero":  # debug only
        policy_args = {}
    elif policy_name.lower() == "binary":  # debug only
        policy_args = {}
    else:
        raise ValueError(f"Unsupported policy for sampling data: {policy_name}")
    return policy_args