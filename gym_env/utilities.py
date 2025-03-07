import gym
import numpy as np
from collections import OrderedDict
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gym_env.env_push_box import Push_Box_Goal_Condi
from gym_env.env_push_one import PusherOneSingleAction
from gym_env.env_scoop import Scoop_Balance
from gym_env.env_push_bar import Push_Bar
from stable_baselines3.common.monitor import Monitor
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb


# Environment context, range, and constructor
ENV_CONTEXTS = {
    "push_box": {
        "context_range": {
            "dynamic@cube_g0@friction_sliding": [0.03, 0.3],
            "dynamic@cube_main@inertia_izz": [0.01, 0.3],
            "dynamic@cube_g0@damping_ratio": [-50, -10],
            "dynamic@pusher_g0@damping_ratio": [-50, -10],
            "dynamic@cube_main@mass": [0.4, 2],
        },
        "constructor": Push_Box_Goal_Condi,
        "state_traj_length": 10,
        "n_state_traj": 3,
    },
    "push_one": {
        # "context_range": {
        #     "dynamic@floor1_table_collision@friction_sliding": [0.02, 0.07],
        #     "dynamic@floor2_table_collision@friction_sliding": [0.02, 0.07],
        #     "dynamic@knob_g0@damping_ratio": [-15, -3],
        #     "dynamic@x_left_wall_g0@damping_ratio": [-15, -3],
        #     "dynamic@x_right_wall_g0@damping_ratio": [-15, -3],
        #     "dynamic@block_g0@damping_ratio": [-15, -3],
        # #     # "dynamic@x_left_wall_g0@damping_ratio": [-80, -4],
        # #     # "dynamic@x_right_wall_g0@damping_ratio": [-80, -4],
        # #     # "dynamic@block_g0@damping_ratio": [-80, -4],
        # },
        "context_range": {
            "dynamic@floor1_table_collision@friction_sliding": [0.02, 0.07],
            "dynamic@floor2_table_collision@friction_sliding": [0.02, 0.07],
            "dynamic@knob_g0@damping_ratio": [-15, -3],
            "dynamic@x_left_wall_g0@damping_ratio": [-40, -3],
            "dynamic@x_right_wall_g0@damping_ratio": [-40, -3],
            "dynamic@block_g0@damping_ratio": [-40, -3],
        },
        "constructor": PusherOneSingleAction,
        "state_traj_length": 10,
        "n_state_traj": 2,
    },
    "scoop": {
        "context_range": {
            "init@rodbalance@com": [-1.0, 1.0],
        },
        "constructor": Scoop_Balance,
        "state_traj_length": 1,
        "n_state_traj": 1,  ## only pitch angle
    },
}


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


def linear_schedule_with_warmup(initial_value, warmup_percent=0.2):
    def func(progress_remaining):
        if progress_remaining > (1 - warmup_percent):
            # Warmup phase
            warmup_progress = (1 - progress_remaining) / warmup_percent
            return initial_value * warmup_progress
        else:
            # Decay phase
            decay_progress = progress_remaining / (1 - warmup_percent)
            return initial_value * decay_progress

    return func


def linear_warmup(initial_value, warmup_percent=0.2):
    def func(progress_remaining):
        if progress_remaining > (1 - warmup_percent):
            # Warmup phase
            warmup_progress = (1 - progress_remaining) / warmup_percent
            return initial_value * warmup_progress
        else:
            return initial_value

    return func


SAC_CONFIG = {
    "push_box": {},
    "push_one": {
        "ent_coef": "0.005",
        "learning_starts": 4096,
        "batch_size": 512,
        "tau": 0.001,
        "learning_rate": linear_schedule(7.3e-4),
        # "learning_rate": linear_warmup(3e-4, warmup_percent=0.1),
        "buffer_size": 1000000,
    },
    "scoop": {"ent_coef": "0.1"},
}

PPO_CONFIG = {
    "push_box": {},
    "push_one": {},
    "push_bar": {},
}


class NormalizeActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Store both the high and low arrays in their original forms
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high

        # We normalize action space to a range [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=self.action_space.shape
        )

    def action(self, action):
        # convert action from [-1,1] to original range
        action = self.denormalize_action(action)
        return action

    def reverse_action(self, action):
        # convert action from original range to [-1,1]
        action = self.normalize_action(action)
        return action

    def normalize_action(self, action):
        action = (
            2
            * (
                (action - self.action_space_low)
                / (self.action_space_high - self.action_space_low)
            )
            - 1
        )
        return action

    def denormalize_action(self, action):
        action = (action + 1) / 2 * (
            self.action_space_high - self.action_space_low
        ) + self.action_space_low
        return action


# class ReacherRewardWrapper(gym.Wrapper):
#     def __init__(self, env, env_name):
#         super().__init__(env)
#         self.env_name = env_name

#     def step(self, action):
#         obs, _, terminated, truncated, info = self.env.step(action)
#         reward = (
#             self.reward_dist_weight * info["reward_dist"]
#             + self.reward_ctrl_weight * info["reward_ctrl"]
#         )
#         return obs, reward, terminated, truncated, info


class CustomMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_successes = []

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observation, reward, done, info = super().step(action)

        if done:
            info["episode"]["s"] = info.get("success")
            info["episode"]["action"] = action
        return observation, reward, done, info


class ContextStateWrapper(gym.Wrapper):
    def __init__(self, env, env_name, dr=False):
        super().__init__(env)
        self.interested_context_range = ENV_CONTEXTS[env_name]["context_range"]
        self.interested_context_names = tuple(
            ENV_CONTEXTS[env_name]["context_range"].keys()
        )
        self.action_space = self.env.action_space
        self.observation_shape = self.env.observation_space
        if self.env.observation_space is None:
            observation_shape = (len(self.interested_context_names),)
        else:
            observation_shape = (
                self.env.observation_space.shape[0]
                + len(self.interested_context_names),
            )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )
        self.dr = dr  # flag for domain radomization

    def step(self, action):
        context_obs = self.get_context_obs()
        env_obs, reward, done, info = self.env.step(action)
        obs = self.get_obs(env_obs, context_obs)
        return obs, reward, True, info

    def reset(self):
        env_obs = self.env.reset()
        context_obs = self.get_context_obs()
        obs = self.get_obs(env_obs, context_obs)
        return obs

    def get_context_obs(self):
        interested_context_dict = self.get_interested_context()
        normalized_interested_context_dict = self.normalize_interested_context(
            interested_context_dict
        )
        context_obs = np.array(list(normalized_interested_context_dict.values()))
        return context_obs

    def get_obs(self, env_obs, context_obs):
        # mask out context if is domain radomization
        if self.dr:
            context_obs = np.zeros_like(context_obs)
              
        if env_obs is not None:
            obs = np.concatenate([env_obs.flatten(), context_obs.flatten()])
        else:
            obs = context_obs.flatten()
        return obs.astype(np.float32)

    def get_interested_context(self):
        interested_context = {}
        context_dict = self.env.get_context()
        for context_name in self.interested_context_names:
            interested_context[context_name] = context_dict[context_name]
        return interested_context

    def normalize_interested_context(self, context_dict):
        normalized_contexts = {}
        for context_name in context_dict:
            lb, ub = self.interested_context_range[context_name]
            value = context_dict[context_name]
            assert lb <= value <= ub
            norm_value = (value - lb) / (ub - lb)
            norm_value = norm_value * 2 - 1.0
            assert -1.0 <= norm_value <= 1.0
            normalized_contexts[context_name] = norm_value
        return normalized_contexts


class RandomContextStateWrapper(ContextStateWrapper):
    def __init__(self, env, env_name, dr):
        super(RandomContextStateWrapper, self).__init__(env, env_name, dr)

    def reset(self):
        # sample random context from context range
        random_context_dict = {}
        for context_name, (lb, ub) in self.interested_context_range.items():
            random_context_dict[context_name] = np.random.uniform(lb, ub)
        # reset the environment and set the context
        self.env.reset()
        self.env.set_context(random_context_dict)
        return super().reset()
    


def init_env(env_name, render, n_envs=8, context=None, dr=False):
    assert env_name in ENV_CONTEXTS

    def make_env():
        def _make_env():
            Env = ENV_CONTEXTS[env_name]["constructor"]
            env = Env(render=render)
            env = ContextStateWrapper(env, env_name, dr=dr)
            if context is not None:
                env.set_context(context)
                # np.random.seed(0)
            return NormalizeActionSpaceWrapper(env)

        if n_envs == -1:
            return _make_env()
        else:
            return CustomMonitor(_make_env())
            # return _make_env()

    if n_envs < 1:
        return make_env()
    elif n_envs == 1:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])


def rollout(env, contexts, policy_name, policy_args):
    # set environment context
    env.reset()
    for idx in range(len(contexts)):
        env.env_method("set_context", contexts[idx], indices=idx)
        
    # reset env_param
    for idx in range(env.num_envs):
        env.env_method("set_env_params", None, indices=idx)
    
    # set env params
    env_params = policy_args.get("env_params", None)
    if env_params is not None: 
        for idx in range(len(env_params)):
            env.env_method("set_env_params", env_params[idx], indices=idx)
    
    obs = env.reset()

    # select action based on the policy name
    if policy_name == "random":
        # uniform sample action from the action space
        low = env.action_space.low
        high = env.action_space.high
        actions = np.random.uniform(low, high, (env.num_envs,) + env.action_space.shape)
        # actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    elif policy_name == "zero":
        # set all action to zero, debug only
        actions = np.zeros((env.num_envs,) + env.action_space.shape)
    elif policy_name == "binary":
        # set to all one or all zero, debug only
        actions = np.random.choice([0, 1], (env.num_envs, 1)) * np.ones(
            (env.num_envs,) + env.action_space.shape
        )
    elif policy_name == "ppo":
        # use the trained ppo policy to roll out action.
        model = policy_args["model"]
        actions, _ = model.predict(obs, deterministic=False)
        # actions *= np.random.uniform(0.7, 1.3, size=actions.shape)
        actions = np.clip(actions, -1, 1)
    elif policy_name == "sac":
        # use the trained ppo policy to roll out action.
        model = policy_args["model"]
        # obs[:,:6] = 0
        actions, _ = model.predict(obs, deterministic=False)
        # actions *= np.random.uniform(0.7, 1.3, size=actions.shape)
        actions = np.clip(actions, -1, 1)
        # actions = np.zeros_like(actions)
    elif policy_name == "fix":
        # use the action we provided
        used_actions = np.array(policy_args["actions"])
        assert (
            len(contexts) == used_actions.shape[0]
        ), f"context: {len(contexts)}, actions: { used_actions.shape[0]}"

        if used_actions.shape[0] < env.num_envs:
            actions = np.zeros((env.num_envs,) + env.action_space.shape)
            actions[0 : used_actions.shape[0]] = used_actions
        else:
            actions = used_actions

        assert actions.shape[0] == env.num_envs
    else:
        raise ValueError(f"policy {policy_name} is not allowed.")
    

    # take action and record trajectory
    state_traj = []
    env_params = []
    obs, _, _, info = env.step(actions)
    for i in range(len(contexts)):
        state_traj_per_env = env.env_method("get_state_traj", info[i], indices=i)[0]
        env_params_per_env = env.env_method("get_env_params", info[i], indices=i)[0]
        state_traj.append(state_traj_per_env)
        env_params.append(env_params_per_env)

    state_traj = np.array(state_traj)
    env_params = np.array(env_params)

    return state_traj, actions[: len(contexts)], env_params


def collect_data(
    env,
    default_context_dict,
    context_range_dict,
    context_traj_dict,
    n_traj,
    policy_name,
    policy_args,
    log_fn=None,
):
    """
    context_traj: dictionary:
        key: context_name
        value: context values
        value shape: n_data, n_history
    """

    # get the transit steps (n_history), and the number of data (n_data) from the context_traj
    assert len(context_traj_dict) > 0, "Need trajectory for at least one data. "
    first_context_traj = next(iter(context_traj_dict.values()))
    n_data, n_history = first_context_traj.shape
    n_envs = env.num_envs
    n_action = env.action_space.shape[0]

    # assert the actions shape
    if policy_name == "fix":
        actions = policy_args["actions"]
        env_params = policy_args.get("env_params", None)
        req_shape = (n_data, n_traj, n_history, n_action)
        assert (
            actions.shape == req_shape
        ), f"actions.shape: {actions.shape}, require shape {req_shape}"
    else:
        actions = None
        env_params = None

    # collect data based on the given trajectory
    all_actions = np.zeros((n_data, n_traj, n_history, n_action))
    all_trajectories = None
    all_env_params = None
    
    context_buffer = []
    action_buffer = []
    env_params_buffer = []
    indices_buffer = []
    i = 0
    with tqdm(
        total=n_data * n_traj * n_history,
        desc=f"Rollout with '{policy_name}' policy",
        ascii=" >=",
        smoothing=0,
    ) as pbar:
        for idx_data in range(n_data):
            for idx_traj in range(n_traj):
                for idx_history in range(n_history):
                    pbar.update(1)

                    # set the context for the current sim environment
                    current_context = deepcopy(default_context_dict)
                    for k, v in context_traj_dict.items():
                        current_context[k] = v[idx_data, idx_history]
                        lb, ub = context_range_dict[k]
                        assert lb <= current_context[k] <= ub
                    context_buffer.append(current_context)
                    indices_buffer.append((idx_data, idx_traj, idx_history))

                    # get the action if the action is fixed
                    if policy_name == "fix":
                        action_buffer.append(actions[idx_data, idx_traj, idx_history])
                        env_params_buffer.append(env_params[idx_data, idx_traj, idx_history])

                    # once the buffer is full or at the end of the loop, run the experiment and collect data
                    if len(context_buffer) == n_envs or pbar.n == pbar.total:
                        pbar.refresh()
                        if policy_name == "fix":
                            assert len(action_buffer) == len(context_buffer)
                            assert len(env_params_buffer) == len(context_buffer)
                            policy_args = {"actions": np.array(action_buffer), "env_params": np.array(env_params_buffer)}

                        rollout_trajectories, rollout_actions, rollout_env_params = rollout(
                            env,
                            context_buffer,
                            policy_name=policy_name,
                            policy_args=policy_args,
                        )

                        for i, (i_data, i_traj, i_history) in enumerate(indices_buffer):
                            trajectory = rollout_trajectories[i]
                            r_env_params = rollout_env_params[i]
                            if all_trajectories is None:
                                all_trajectories = np.zeros(
                                    (n_data, n_traj, n_history) + trajectory.shape
                                )
                            if all_env_params is None:
                                all_env_params = np.zeros(
                                    (n_data, n_traj, n_history) + r_env_params.shape
                                )

                            all_trajectories[i_data, i_traj, i_history] = trajectory
                            all_actions[i_data, i_traj, i_history] = rollout_actions[i]
                            all_env_params[i_data, i_traj, i_history] = r_env_params

                        context_buffer.clear()
                        action_buffer.clear()
                        indices_buffer.clear()
                        env_params_buffer.clear()
                        
                        if log_fn is not None:
                            log_fn(pbar.n, pbar.total)

    # stack results
    return all_trajectories, all_actions, all_env_params


def downsample_state_trajectories(state_trajectories, env_name):
    # downsample data
    """
    trajectories shape:     (n_data, n_traj, n_history, 30, n_state)
    """
    state_traj_length = ENV_CONTEXTS[env_name]["state_traj_length"]
    if state_trajectories.shape[3] % state_traj_length == 0:
        skip_frames = state_trajectories.shape[3] // state_traj_length
        indices = np.arange(0, state_trajectories.shape[3], skip_frames)
        sampled_state_trajectories = np.take(state_trajectories, indices, axis=3)
    else:
        print(state_trajectories.shape, state_traj_length)
        raise NotImplementedError

    return sampled_state_trajectories


def plot_boundary_push_one(ax):
    ax.plot([0.45, 0.45], [0.24, -0.24], "-", c="k", alpha=1)
    ax.plot([-0.45, 0.45], [0.24, 0.24], "-", c="k", alpha=1)
    ax.plot([-0.45, 0.45], [-0.24, -0.24], "-", c="k", alpha=1)
    # axes_ave.axes.set_xlim(-0.5, 0.5)
    # axes_ave.axes.set_ylim(-0.5, 0.5)


# def plot_push_one_traj_average(ax, eval_results):
#     """
#     eval_result = {
#         "context_history":              (n_eval, n_history, n_context)
#         "context_real":                 (n_eval, 1,         n_context)
#         "action_history":               (n_eval, n_history, n_action)
#         "input_sim_traj_histories":     (n_eval, n_history, n_state * n_timestep)
#         "input_real_traj_histories":    (n_eval, n_history, n_state * n_timestep)
#         "input_timestep_histories":     (n_eval, n_history, 1)
#     }
#     """
#     n_eval, n_history = eval_results["context_history"].shape[:2]
#     sim_traj_histories = eval_results["input_sim_traj_histories"].reshape(
#         n_eval, n_history, 10, -1
#     )
#     real_traj_histories = eval_results["input_real_traj_histories"].reshape(
#         n_eval, n_history, 10, -1
#     )
#     diff = sim_traj_histories - real_traj_histories
#     final_diff = diff[:, -1, :]
#     final_diff = np.linalg.norm(final_diff, axis=-1)
#     final_diff = np.mean(final_diff, axis=-1)

#     mean_sim_traj_histories = sim_traj_histories.mean(axis=0)
#     mean_real_traj_histories = real_traj_histories.mean(axis=0)
#     plot_boundary_push_one(ax)
#     i = -1
#     ax.plot(
#         mean_sim_traj_histories[i, :, 0],
#         mean_sim_traj_histories[i, :, 1],
#         "o-",
#         label=f"sim_traj_{np.round(np.mean(final_diff), 4)}",
#     )
#     ax.plot(
#         mean_real_traj_histories[i, :, 0],
#         mean_real_traj_histories[i, :, 1],
#         "^-",
#         label=f"real_traj_{np.round(np.mean(final_diff), 4)}",
#     )
#     ax.legend()
#     ax.set_title("Average trajectories")


# def plot_push_one_traj_min(ax, eval_results):
#     """
#     eval_result = {
#         "context_history":              (n_eval, n_history, n_context)
#         "context_real":                 (n_eval, 1,         n_context)
#         "action_history":               (n_eval, n_history, n_action)
#         "input_sim_traj_histories":     (n_eval, n_history, n_state * n_timestep)
#         "input_real_traj_histories":    (n_eval, n_history, n_state * n_timestep)
#         "input_timestep_histories":     (n_eval, n_history, 1)
#     }
#     """
#     n_eval, n_history = eval_results["context_history"].shape[:2]
#     sim_traj_histories = eval_results["input_sim_traj_histories"].reshape(
#         n_eval, n_history, 10, -1
#     )
#     real_traj_histories = eval_results["input_real_traj_histories"].reshape(
#         n_eval, n_history, 10, -1
#     )
#     diff = sim_traj_histories - real_traj_histories
#     final_diff = diff[:, -1, :]
#     final_diff = np.linalg.norm(final_diff, axis=-1)
#     final_diff = np.mean(final_diff, axis=-1)

#     min_index = np.argmin(final_diff)
#     min_sim_traj_histories = sim_traj_histories[min_index]
#     min_real_traj_histories = real_traj_histories[min_index]

#     plot_boundary_push_one(ax)
#     ax.plot(
#         min_sim_traj_histories[-1, :, 0],
#         min_sim_traj_histories[-1, :, 1],
#         "o-",
#         label=f"sim_{np.round(np.min(final_diff), 4)}",
#     )
#     ax.plot(
#         min_real_traj_histories[-1, :, 0],
#         min_real_traj_histories[-1, :, 1],
#         "o-",
#         label=f"real_{np.round(np.min(final_diff), 4)}",
#     )
#     ax.legend()
#     ax.set_title("Trajectories with minimum error.")


def plot_push_one_traj(ax, eval_results, percentile):
    """
    eval_result = {
        "context_history":              (n_eval, n_history, n_context)
        "context_real":                 (n_eval, 1,         n_context)
        "action_history":               (n_eval, n_history, n_action)
        "input_sim_traj_histories":     (n_eval, n_history, n_state * n_timestep)
        "input_real_traj_histories":    (n_eval, n_history, n_state * n_timestep)
        "input_timestep_histories":     (n_eval, n_history, 1)
    }
    """
    # load trajectories between sim and real
    n_eval, n_history = eval_results["context_history"].shape[:2]
    sim_traj_histories = eval_results["input_sim_traj_histories"].reshape(
        n_eval, n_history, 10, -1
    )
    real_traj_histories = eval_results["input_real_traj_histories"].reshape(
        n_eval, n_history, 10, -1
    )

    # get trajectories difference of the last step
    diff = sim_traj_histories - real_traj_histories
    final_diff = diff[:, -1, :]
    final_diff = np.linalg.norm(final_diff, axis=-1)
    final_diff = np.mean(final_diff, axis=-1)

    # get the trajectories difference at the given percentile
    req_diff = np.percentile(final_diff, percentile)
    req_idx = np.argmin(np.abs(final_diff - req_diff))
    select_sim_traj_histories = sim_traj_histories[req_idx]
    select_real_traj_histories = real_traj_histories[req_idx]
    select_diff = final_diff[req_idx]

    plot_boundary_push_one(ax)
    ax.plot(
        select_sim_traj_histories[-1, :, 0],
        select_sim_traj_histories[-1, :, 1],
        "o-",
        label=f"sim_{np.round(select_diff, 4)}",
    )
    ax.plot(
        select_real_traj_histories[-1, :, 0],
        select_real_traj_histories[-1, :, 1],
        "o-",
        label=f"sim_{np.round(select_diff, 4)}",
    )
    ax.legend()
    ax.set_title(f"Trajectories with error greater than {percentile}% of evaluations.")


def plot_push_bar_traj(ax, eval_results, percentile):
    pass


def plot_traj(env_name, eval_results):
    if env_name == "push_one":
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        plot_push_one_traj(axs[0], eval_results, percentile=25)
        plot_push_one_traj(axs[1], eval_results, percentile=50)
        plot_push_one_traj(axs[2], eval_results, percentile=75)
        return {
            "trajectories/visualize": fig,
        }
    elif env_name == "scoop":

        return {}
    else:
        raise ValueError(f"{env_name} is not supported.")
