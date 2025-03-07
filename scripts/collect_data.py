import os
from copy import deepcopy
import numpy as np
import time
import wandb
import warnings
from scipy.stats import beta
import matplotlib.pyplot as plt

from gym_env.utilities import ENV_CONTEXTS
from gym_env.utilities import init_env, collect_data, downsample_state_trajectories
from scripts.utilities import get_parser, get_dataset_dir, get_task_policy


warnings.filterwarnings("ignore")


def sample_value(value_ranges, value_shape):
    if isinstance(value_shape, int):
        value_shape = (value_shape,)
    n_value = int(np.prod(value_shape))

    # compute the total range of data
    total_range = 0
    for lb, ub in value_ranges:
        assert lb <= ub, f"lb: {lb}, ub: {ub}"
        total_range += ub - lb

    # compute the data we need to sample from each range
    value_nums = np.zeros(len(value_ranges), dtype=int)
    for i, (lb, ub) in enumerate(value_ranges):
        num = int(np.round(np.abs(ub - lb) / total_range * n_value))
        value_nums[i] = num
    value_nums[0] = int(n_value - np.sum(value_nums[1:]))

    # sample data from the above range
    samples = []
    for i, (lb, ub) in enumerate(value_ranges):
        samples.append(np.random.uniform(lb, ub, size=value_nums[i]))
    samples = np.concatenate(samples)

    # shuffle array if we sample from multiple ranges
    if len(value_ranges) > 1:
        np.random.shuffle(samples)

    # reshape samples and return
    samples = samples.reshape(value_shape)

    return samples


def sample_context(context_range_dict, size, sample_method="normal"):
    # parameters for sample interpolation and extrapolations
    mean = 0
    p = 0.5
    # sample context from uniform distribution
    samples = {}
    for context_name in context_range_dict:
        lb, ub = context_range_dict[context_name]
        if sample_method == "normal":
            sample_ranges = [(lb, ub)]
        else:
            ValueError(f"Unsupported sample_method {sample_method}")

        samples[context_name] = sample_value(sample_ranges, value_shape=size)
    return samples


def interp_sim2real_values(sim_values, real_values, context_range, n=7, method="linear"):
    # sim values and real values are all (N, 1)  arrays
    sim_values = np.array(sim_values)
    real_values = np.array(real_values)
    # print(sim_values.shape, real_values.shape)
    assert (
        sim_values.shape == real_values.shape
    ), f"sim_values.shape: {sim_values.shape}, real_values.shape: {real_values.shape}"
    assert sim_values.shape[1] == 1
    if method == "linear":
        # sample sim2real trajectory with linear interpolation.
        values = np.linspace(sim_values.flatten(), real_values.flatten(), n, axis=-1)
        condition_1 = (sim_values <= values) & (values <= real_values)
        condition_2 = (real_values <= values) & (values <= sim_values)
        assert np.all(
            condition_1 | condition_2
        ), f"{values}\n{sim_values}\n{real_values}"
        return values
    elif method == "mono_stochastic":
        # sample sim2real trajectory that uniformed sample between sim and real but change monotonously.
        n_values = sim_values.size
        values = np.random.uniform(0, 1, size=(n_values, n))
        values = np.sort(values, axis=1)
        # denormalize the context trajectory
        values = sim_values + values * (real_values - sim_values)
        values[:, 0:1] = sim_values
        values[:, -1:] = real_values
        condition_1 = (sim_values <= values) & (values <= real_values)
        condition_2 = (real_values <= values) & (values <= sim_values)
        assert np.all(
            condition_1 | condition_2
        ), f"{values}\n{sim_values}\n{real_values}"
        return values
    elif method.startswith("search_"):
        # unifrom sample based on the current lower bound and upper bound
        n_values = sim_values.size
        lb = np.ones(n_values) * context_range[0]
        ub = np.ones(n_values) * context_range[1]
        # context trajectory
        values = np.zeros([n_values, n])
        # print(values.shape, real_values.shape)
        values[:, 0] = sim_values.flatten()
        # narrow sampling space based on the upper and lower bound
        for i in range(n - 1):
            # update the lower bound
            mask_lb = values[:, i] < real_values.flatten()
            lb[mask_lb] = values[:, i][mask_lb]
            # update the upper bound
            mask_ub = values[:, i] > real_values.flatten()
            ub[mask_ub] = values[:, i][mask_ub]
            # sample based on the current lower bound and upper bound
            if method == "search_uniform":
                values[:, i + 1] = np.random.uniform(low=lb, high=ub, size=n_values)
            elif method == "search_binary":
                values[:, i+1] = np.mean([lb, ub], axis=0)
            elif method.startswith("search_beta_"):
                a = float(method.replace("search_beta_", ""))
                b = a # shape parameter beta (equal to alpha for symmetric distribution)
                norm_sample = beta.rvs(a, b, size=n_values)
                scaled_sample = lb + (ub - lb) * norm_sample
                values[:, i + 1] = scaled_sample
            else:
                raise ValueError(f"Does not suppot method: {method}")
                
        values[:, n-1] = real_values.flatten()
        condition_1 = (context_range[0] <= values) & (values <= context_range[1])
        assert np.all(
            condition_1), f"{values}\n{sim_values}\n{real_values}"

        return values    
    
    else:
        raise ValueError(f"{method} is not supported.")


def interp_sim2real_context(sim_context, real_context, context_range_dict ,n=7, method="linear"):
    context_traj_dict = {}
    for context_name in sim_context:
        sim_values = sim_context[context_name]
        real_values = real_context[context_name]
        context_range = context_range_dict[context_name]
        context_traj_dict[context_name] = interp_sim2real_values(
            sim_values, real_values, context_range, n=n, method=method
        )
    return context_traj_dict


def sample_sim_and_real_context(context_range_dict, args):
    mode = args.mode
    n_data = args.n_data
    sample_method = args.sample_method
    assert mode in ["train", "eval"]
    # sample sim context
    sim_context = sample_context(context_range_dict, size=(n_data, 1))
    # sample real context
    if sample_method == "normal":
        real_context = sample_context(context_range_dict, size=(n_data, 1))
    else:
        raise ValueError("Not possible case.")

    # validate sim and real context range
    for context_name in context_range_dict:
        lb, ub = context_range_dict[context_name]
        assert np.all(
            (lb <= sim_context[context_name]) & (sim_context[context_name] <= ub)
        )
        assert np.all(
            (lb <= real_context[context_name]) & (real_context[context_name] <= ub)
        )

    return sim_context, real_context


def save_results(
    args, contexts_dict, actions=None, sim_trajectories=None, real_trajectories=None
):
    result_dir = get_dataset_dir(args)
    os.makedirs(result_dir, exist_ok=True)

    # save the context as numpy array
    contexts = list(contexts_dict.values())
    contexts = np.stack(contexts)
    np.save(os.path.join(result_dir, "contexts.npy"), contexts)

    # save action, sim and real trajectories
    if actions is not None:
        np.save(os.path.join(result_dir, f"actions.npy"), actions)
    if sim_trajectories is not None:
        np.save(os.path.join(result_dir, f"sim_trajectories.npy"), sim_trajectories)
    if actions is not None:
        np.save(os.path.join(result_dir, f"real_trajectories.npy"), real_trajectories)


def get_log_func(args, collect_name):
    def fn(n, n_total):
        wandb.log({f"collect/{collect_name}": 100 * n / n_total})

    if args.log_to_wandb:
        return fn
    return None


def main(args):
    n_envs = args.n_envs
    n_data = args.n_data
    n_traj = args.n_traj
    n_history = args.n_history
    env_name = args.env_name
    transit_method = args.transit_method
    mode = args.mode

    # get the interested context range
    context_range_dict = ENV_CONTEXTS[env_name]["context_range"]
    assert len(context_range_dict) > 0, "Need at least one context for the environment"

    # sample context for the each sim2real pair
    sim_context, real_context = sample_sim_and_real_context(context_range_dict, args)

    # get the transit trajectory for each sim2real pair
    context_sim2real_traj_dict = interp_sim2real_context(
        sim_context, real_context, context_range_dict, n=n_history, method=transit_method
    )

    # get the real trajectory for each sim2real pair
    context_real_traj_dict = interp_sim2real_context(
        real_context, real_context, context_range_dict, n=n_history, method="linear"
    )

    # Only need to rollout for training process, the evaluation process needs online roll out
    if mode == "train":
        # save the default context
        env_tmp = init_env(env_name=env_name, n_envs=-1, render=False, dr=args.dr)
        env_tmp.reset()
        default_context = deepcopy(env_tmp.get_context())
        env_tmp.close()

        # sample action and rollout in the sim environment.
        env = init_env(env_name=env_name, n_envs=n_envs, render=False, dr=args.dr)
        policy_args = get_task_policy(args.policy_name, args.policy_checkpoint)

        # sample data
        print("Sample training data")
        print(f"n_data: {n_data} sim2real pairs.")
        print(f"n_traj: {n_traj} sim and real trajectories for each sim to real pair.")
        print(
            f"n_history: {n_history} steps to transit from sim to real with {transit_method} method."
        )

        # Setup wandb
        if args.log_to_wandb:
            exp_prefix = f'{get_dataset_dir(args)}-{time.strftime("%Y-%m-%d-%H-%M")}'
            wandb.init(
                name=exp_prefix,
                group="Collect data",
                project="sim2real-capture-data-collection",
                config=args,
            )
        print("Collecting sim data")
        sim_trajectories, actions, env_params = collect_data(
            env,
            default_context,
            context_range_dict,
            context_sim2real_traj_dict,
            n_traj,
            args.policy_name,
            policy_args,
            log_fn=get_log_func(args, "sim"),
        )
        print("Collecting real data")
        real_trajectories, _, _ = collect_data(
            env,
            default_context,
            context_range_dict,
            context_real_traj_dict,
            n_traj,
            "fix",
            {"actions": actions.copy(), "env_params": env_params.copy()},
            log_fn=get_log_func(args, "real"),
        )

        real_trajectories = downsample_state_trajectories(real_trajectories, env_name)
        sim_trajectories = downsample_state_trajectories(sim_trajectories, env_name)

        save_results(
            args,
            context_sim2real_traj_dict,
            actions,
            sim_trajectories,
            real_trajectories,
        )
    elif mode == "eval":
        save_results(args, context_sim2real_traj_dict)
    else:
        raise ValueError(f"Unsupported mode {mode}.")


def validate_args(args):
    env_names = list(ENV_CONTEXTS.keys())
    assert args.env_name in env_names
    assert args.mode in ["train", "eval"]
    if args.mode == "train" and args.sample_method == "cg":
        raise ValueError("No need to sample training data for composition generation.")


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--n_data_train_cg",
        type=int,
        help="the number of data in the train file, used to load training context to generate compensation samples.",
    )
    parser.add_argument(
        "--n_envs", type=int, default=1, help="number of parallel environment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="sample training data or evaluation data",
    )
    parser.add_argument("--log_to_wandb", "-w", action="store_true")

    args = parser.parse_args()
    validate_args(args)
    main(args)

    # main_test(args.num)
