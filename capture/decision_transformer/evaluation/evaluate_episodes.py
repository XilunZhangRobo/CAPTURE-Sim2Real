import numpy as np
import torch
import wandb
import argparse
import pickle
import random
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from collections import deque

# from capture.decision_transformer.evaluation.evaluate_episodes_bk import (
#     evaluate_episode_delta_theta_batch,
# )
from capture.decision_transformer.models.sim2real_transformer import DecisionTransformer
from capture.decision_transformer.training.seq_trainer_sim2real import SequenceTrainer
from gym_env.utilities import init_env, rollout
import time
import sys

from gym_env.utilities import collect_data, downsample_state_trajectories
from gym_env.utilities import ENV_CONTEXTS
from scripts.utilities import (
    get_dataset_dir,
    get_parser,
    load_data,
    normalize_context,
    to_tensor,
    set_seed,
)


def get_context_traj(batch_contexts_normal, context_range):
    """
    contexts shape:             (batch_size, n_context)
    """
    # the term trajectory is used just to be compatible with the data collection code
    n_data = batch_contexts_normal.shape[0]
    n_history = 1
    batch_size = n_data
    # get the trajectories for the context
    context_traj = {k: np.zeros([n_data, n_history]) for k in context_range}
    for batch_idx in range(batch_size):
        for i, context_name in enumerate(context_range):
            v = batch_contexts_normal[batch_idx, i]
            lb, ub = context_range[context_name]
            context_value = (v + 1.0) / 2
            context_value = context_value * (ub - lb) + lb
            context_traj[context_name][batch_idx, 0] = context_value
            assert (
                lb <= context_traj[context_name][batch_idx, 0] <= ub
            ), f"context_value: {context_traj[context_name][batch_idx, 0]}, lb: {lb}, ub: {ub}"
    return context_traj


def get_context_difference(context_sim_histories, context_real):
    context_sim_histories = np.concatenate(context_sim_histories, axis=1)
    context_diff = context_sim_histories - context_real
    return context_diff


def get_trajectories_difference(sim_traj_histories, real_traj_histories):
    sim_traj_histories = np.concatenate(sim_traj_histories, axis=1)
    real_traj_histories = np.concatenate(real_traj_histories, axis=1)
    diff = sim_traj_histories - real_traj_histories
    diff = np.linalg.norm(diff, axis=-1)
    return diff


def evaluate_batch(
    env,
    model,
    batch_eval_contexts_normal,
    env_name,
    default_context,
    policy_name,
    policy_args,
    window_size,
    n_traj=1,
    n_iteration=6,
    delta_output=False,
    relative_position=False,
    args=None,
):
    """
    Evaluate a batch performance.
    contexts shape:             (batch_size, n_history, n_context)
    """
    device = next(model.parameters()).device

    assert len(batch_eval_contexts_normal.shape) == 3
    context_range = ENV_CONTEXTS[env_name]["context_range"]

    # get the sim and real contexts for this batch
    batch_sim_context_normal = batch_eval_contexts_normal[:, 0, :].copy()
    batch_real_context_normal = batch_eval_contexts_normal[:, -1, :].copy()
    batch_size = batch_eval_contexts_normal.shape[0]
    assert env.num_envs == batch_size

    # iteratively match the real state trajectories
    sim_traj_histories = []
    real_traj_histories = []
    action_histories = []
    context_histories = []
    timestep_histories = []

    for idx in range(n_iteration):

        # get the context trajectories for this batch
        # the output is dictionary to compatible with the collect data function
        batch_sim_contexts_traj = get_context_traj(
            batch_sim_context_normal, context_range
        )
        batch_real_contexts_traj = get_context_traj(
            batch_real_context_normal, context_range
        )
        print(f"Collect trajectories for evaluation after {idx} update.")
        """
        sim_trajectories, actions = collect_data(
            env,
            default_context,
            context_range,
            batch_sim_contexts_traj,
            n_traj,
            policy_name,
            {
                "actions": policy_args["actions"][:, :, idx : idx + 1]
            },  # DEBUG_CODE: evaluate on action set (policy_args)
        )
        """
        sim_trajectories, actions, env_params = collect_data(
            env,
            default_context,
            context_range,
            batch_sim_contexts_traj,
            n_traj,
            policy_name,
            policy_args,
        )
        real_trajectories, _, _ = collect_data(
            env,
            default_context,
            context_range,
            batch_real_contexts_traj,
            n_traj,
            "fix",
            {"actions": actions, "env_params": env_params},
        )
        if args.mode == "eval":
            if args.env_name == "scoop":
                if idx % 3 == 0 and idx > 1:
                    real_trajectories = np.random.choice([-1, 0, 1], size=real_trajectories.shape)
            elif args.env_name == "push_one":
                    real_trajectories = real_trajectories * np.random.uniform(0.9, 1.1, size=real_trajectories.shape)

        # downsample data
        """
        actions shape:              (n_data=batch_size,     n_traj=1, n_history=1, n_action)
        real_trajectories shape:    (n_data=batch_size,     n_traj=1, n_history=1, 10, n_state)
        sim_trajectories shape:     (n_data=batch_size,     n_traj=1, n_history=1, 10, n_state)
        """
        real_trajectories = downsample_state_trajectories(real_trajectories, env_name)
        sim_trajectories = downsample_state_trajectories(sim_trajectories, env_name)

        # reshape data for input to transformer model.
        assert real_trajectories.shape[0] == batch_size
        assert sim_trajectories.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        real_trajectories = real_trajectories.reshape([batch_size, 1, -1])
        sim_trajectories = sim_trajectories.reshape([batch_size, 1, -1])
        actions = actions.reshape([batch_size, 1, -1])
        sim_contexts = batch_sim_context_normal.reshape([batch_size, 1, -1])
        timestep = np.ones([batch_size, 1]) * idx

        sim_traj_histories.append(sim_trajectories)
        real_traj_histories.append(real_trajectories)
        action_histories.append(actions)
        context_histories.append(sim_contexts)
        timestep_histories.append(timestep)

        # input history into transformer model
        input_context_histories = np.concatenate(context_histories, axis=1)
        input_action_histories = np.concatenate(action_histories, axis=1)
        input_sim_traj_histories = np.concatenate(sim_traj_histories, axis=1)
        input_real_traj_histories = np.concatenate(real_traj_histories, axis=1)
        input_timestep_histories = np.concatenate(
            timestep_histories[-window_size:], axis=1
        )
        if relative_position:
            input_timestep_histories = (
                input_timestep_histories - input_timestep_histories[:, 0:1]
            )


        pred, _ = model.get_action(
            to_tensor(input_context_histories, device=device),
            to_tensor(input_action_histories, device=device),
            to_tensor(input_sim_traj_histories, device=device),
            to_tensor(input_real_traj_histories, device=device),
            to_tensor(
                input_timestep_histories,
                device=device,
                dtype=torch.long,
            ),
            batch_size=batch_size,
        )

        if delta_output:
            delta_theta_predict = pred.detach().cpu().numpy()
            batch_sim_context_normal = np.clip(
                batch_sim_context_normal + delta_theta_predict, -1, 1
            )
        else:
            theta_predict = pred.detach().cpu().numpy()
            batch_sim_context_normal = np.clip(theta_predict, -1, 1)

    # TODO: add loss function to evaluate performance
    eval_result = {
        "context_history": np.concatenate(context_histories, axis=1),
        "context_real": batch_real_context_normal.reshape([batch_size, 1, -1]),
        "action_history": np.concatenate(action_histories, axis=1),
        "input_sim_traj_histories": np.concatenate(sim_traj_histories, axis=1),
        "input_real_traj_histories": np.concatenate(real_traj_histories, axis=1),
        "input_timestep_histories": np.concatenate(timestep_histories, axis=1),
    }
    return eval_result
