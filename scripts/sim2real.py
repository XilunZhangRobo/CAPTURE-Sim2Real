import numpy as np
import torch
import wandb
import os
from copy import deepcopy
import time
import sys
import warnings
from stable_baselines3 import PPO, SAC

from capture.decision_transformer.evaluation.evaluate_episodes import (
    evaluate_batch,
)
from capture.decision_transformer.models.sim2real_transformer import DecisionTransformer
from capture.decision_transformer.models.sim2real_mlp import MLP
from capture.decision_transformer.training.seq_trainer_sim2real import SequenceTrainer

from gym_env.utilities import init_env, plot_traj
from gym_env.utilities import ENV_CONTEXTS, SAC_CONFIG, PPO_CONFIG
from capture.utilities import WarmupCosineSchedule, WarmupInverseSqrtSchedule
from scripts.utilities import (
    get_model_dir,
    get_parser,
    get_task_policy,
    load_data,
    normalize_context,
    to_tensor,
    set_seed,
)


def get_model_dt(args, n_context, n_state, n_action, state_traj_length):
    # TODO: Why is env name required for the transformer model?
    model = DecisionTransformer(
        theta_dim=n_context,
        traj_dim=state_traj_length * n_state,
        action_dim=n_action,
        state_dim=n_state,
        max_length=args.window_size,
        max_ep_len=args.n_history,
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4 * args.embed_dim,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        env_name=args.env_name,
        relative_position=args.relative_position,
        single_context_token=args.single_context_token,
    )
    return model


def get_model_mlp(args, n_context, n_state, n_action, state_traj_length):
    # TODO: Why is env name required for the transformer model?
    model = MLP(
        theta_dim=n_context,
        traj_dim=state_traj_length * n_state,
        action_dim=n_action,
        max_length=args.window_size,
        max_ep_len=args.n_history,
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4 * args.embed_dim,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        env_name=args.env_name,
        relative_position=args.relative_position,
    )
    return model


def get_model(args, n_context, n_state, n_action, state_traj_length):
    model_type = args.model_type
    if model_type == "dt":
        return get_model_dt(args, n_context, n_state, n_action, state_traj_length)
    elif model_type == "mlp":
        return get_model_mlp(args, n_context, n_state, n_action, state_traj_length)
    else:
        raise ValueError(f"Unsupported model {model_type}")


def padding_history(input_data, n_padding=2, history_axis=2):
    # pad zero before the history
    padding_shape = list(input_data.shape)
    padding_shape[history_axis] = n_padding
    padding = np.zeros(padding_shape)
    return np.concatenate([padding, input_data], axis=history_axis)


def sample_data(input_data, length, indices):
    data_indices, traj_indices, history_indices = indices
    sampled_data = []
    # add all the histories together
    for i in range(length):
        sampled_data.append(input_data[data_indices, traj_indices, history_indices + i])
    return np.stack(sampled_data, axis=1)


def generate_training_data(
    args,
    contexts,
    sim_trajectories,
    real_trajectories,
    actions,
    indices,
    debug=False,
):
    # generate timesteps for model
    timesteps = np.arange(0, args.n_history)
    timesteps = np.tile(timesteps, (args.n_data, args.n_traj, 1))
    if debug:
        print("contexts shape:", contexts.shape)
        print()
        print("actions shape:", actions.shape)
        print("real_trajectories shape:", real_trajectories.shape)
        print("sim_trajectories shape:", sim_trajectories.shape)
        print("timesteps shape:", timesteps.shape)
    """
    contexts shape:             (n_context,  n_data, n_history)
    actions shape:              (n_data,     n_traj, n_history, n_action)
    real_trajectories shape:    (n_data,     n_traj, n_history, 30, n_state)
    sim_trajectories shape:     (n_data,     n_traj, n_history, 30, n_state)
    timesteps shape:            (n_data,     n_traj, n_history)
    """

    # attention masks, meaning we can attend to all training data.
    masks = np.ones_like(timesteps)

    # duplicate and permute context axis to mach other training data
    whole_contexts = np.tile(contexts, (args.n_traj, 1, 1, 1))
    whole_contexts = np.transpose(whole_contexts, (2, 0, 3, 1))
    real_contexts = whole_contexts[:, :, -1:, :]
    real_contexts = np.tile(real_contexts, (1, 1, args.n_history, 1))
    """
    whole_contexts shape:           (n_traj, n_context, n_data, n_history)
    whole_contexts desire shape:    (n_data, n_traj,  n_history, n_context)
    real_contexts shape:            (n_data, n_traj,  n_history, n_context)
    """

    if debug:
        print("whole_contexts shape:", whole_contexts.shape)

    # padding history
    if args.input_padding > 0:
        n_padding = args.input_padding
        whole_contexts = padding_history(whole_contexts, n_padding=n_padding)
        real_contexts = padding_history(real_contexts, n_padding=n_padding)
        actions = padding_history(actions, n_padding=n_padding)
        real_trajectories = padding_history(real_trajectories, n_padding=n_padding)
        sim_trajectories = padding_history(sim_trajectories, n_padding=n_padding)
        timesteps = padding_history(timesteps, n_padding=n_padding)
        masks = padding_history(masks, n_padding=n_padding)
        if debug:
            print()
            print("padded whole_contexts shape:", whole_contexts.shape)
            print("padded real_contexts shape:", real_contexts.shape)
            print("padded actions shape:", actions.shape)
            print("padded real_trajectories shape:", real_trajectories.shape)
            print("padded sim_trajectories shape:", sim_trajectories.shape)
            print("padded timesteps shape:", timesteps.shape)
            print("padded masks shape:", masks.shape)

    # since to we are predicting next timestep instead of the current one, we need to take one more timestep for the input (window size + 1)
    length = args.window_size + 1

    # sampled data
    sampled_contexts = sample_data(whole_contexts, length, indices)
    sampled_real_contexts = sample_data(real_contexts, length, indices)
    sampled_actions = sample_data(actions, length, indices)
    sampled_sim_trajectories = sample_data(sim_trajectories, length, indices)
    sampled_real_trajectories = sample_data(real_trajectories, length, indices)
    sampled_timesteps = sample_data(timesteps, length, indices)

    # relative positioning for timesteps
    if args.relative_position:
        sampled_timesteps = sampled_timesteps - sampled_timesteps[:, 0:1]

    sampled_masks = sample_data(masks, length, indices)

    # reshape the sim and real state trajectories
    n_samples = sampled_contexts.shape[0]
    sampled_sim_trajectories = sampled_sim_trajectories.reshape([n_samples, length, -1])
    sampled_real_trajectories = sampled_real_trajectories.reshape(
        [n_samples, length, -1]
    )

    if debug:
        print()
        print("sampled whole_contexts shape:", sampled_contexts.shape)
        print("sampled real_contexts shape:", sampled_real_contexts.shape)
        print("sampled actions shape:", sampled_actions.shape)
        print("sampled real_trajectories shape:", sampled_real_trajectories.shape)
        print("sampled sim_trajectories shape:", sampled_sim_trajectories.shape)
        print("sampled timesteps shape:", sampled_timesteps.shape)
        print("sampled masks shape:", sampled_masks.shape)
        """
        sampled whole_contexts shape:       (n_samples, length, n_context)
        sampled real_contexts shape:        (n_samples, length, n_context)
        sampled actions shape:              (n_samples, length, n_action)
        sampled real_trajectories shape:    (n_samples, length, traj_length * n_state)
        sampled sim_trajectories shape:     (n_samples, length, traj_length * n_state)
        sampled timesteps shape:            (n_samples, length)
        sampled masks shape:                (n_samples, length)
        """
    return (
        sampled_contexts,
        sampled_real_contexts,
        sampled_actions,
        sampled_sim_trajectories,
        sampled_real_trajectories,
        sampled_timesteps,
        sampled_masks,
    )


def delta_next_loss_func(delta_theta_pred, theta_input, theta_target, theta_real):
    theta_pred = delta_theta_pred + theta_input
    loss = torch.mean((theta_pred - theta_target) ** 2)
    return loss


def delta_final_loss_func(delta_theta_pred, theta_input, theta_target, theta_real):
    theta_pred = delta_theta_pred + theta_input
    loss = torch.mean((theta_pred - theta_real) ** 2)
    return loss


def next_loss_func(theta_pred, theta_input, theta_target, theta_real):
    loss = torch.mean((theta_pred - theta_target) ** 2)
    return loss


def final_loss_func(theta_pred, theta_input, theta_target, theta_real):
    loss = torch.mean((theta_pred - theta_real) ** 2)
    return loss


def experiment(args):
    # parameters for training and evaluation set
    env_name = args.env_name
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        raise ValueError("Cuda is not available, can not train on device {device}")

    # parameters for the training process
    batch_size = args.batch_size
    log_to_wandb = args.log_to_wandb

    # set random seed
    set_seed(args.seed + 1)

    # get environment parameter
    env_tmp = init_env(env_name=env_name, n_envs=-1, render=False, dr=args.dr)
    action_shape = env_tmp.action_space.shape
    assert len(action_shape) == 1, "Only support one dimensional action space."
    n_state = ENV_CONTEXTS[env_name]["n_state_traj"]
    n_action = action_shape[0]
    state_traj_length = ENV_CONTEXTS[env_name]["state_traj_length"]

    # get env context range
    context_range_dict = ENV_CONTEXTS[env_name]["context_range"]
    context_range = np.stack(list(context_range_dict.values()))
    n_context = context_range.shape[0]

    # load eval policy
    if args.eval_policy_name == "auto":
        args.eval_policy_name = args.policy_name
        if args.eval_policy_checkpoint is None:
            # used the training checkpoint
            args.eval_policy_checkpoint = args.policy_checkpoint

    # load training and evaluation data
    train_args = deepcopy(args)
    train_args.mode = "train"
    eval_args = deepcopy(args)
    eval_args.mode = "eval"
    eval_args.n_data = args.n_data_eval
    if args.sample_method == "cg":
        train_args.sample_method = "normal"  # the training set for compensation generation is the normal dataset.

    (
        train_contexts,
        train_actions,
        train_real_trajectories,
        train_sim_trajectories,
        eval_contexts,
    ) = load_data(train_args, eval_args)
    """
    eval_contexts               (n_context, n_data_eval, n_history)
    train_contexts              (n_context, n_data, n_history)
    train_actions               (n_data, n_traj, n_history, n_action)
    train_real_trajectories     (n_data, n_traj, n_history, traj_length, n_state)
    train_sim_trajectories      (n_data, n_traj, n_history, traj_length, n_state)
    """

    # only keep the first (sim) and last (real) for expert distillation (ED)
    if args.method == "ed":
        history_indices = [0, -1]
        train_contexts = train_contexts[:, :, history_indices]
        train_actions = train_actions[:, :, history_indices]
        train_real_trajectories = train_real_trajectories[:, :, history_indices]
        train_sim_trajectories = train_sim_trajectories[:, :, history_indices]
        eval_contexts = eval_contexts[:, :, history_indices]
        # modified args parameters for expert distillation (ED)
        args.n_history = len(history_indices)
        args.window_size = 1
        args.input_padding = 0


    # normalize contexts
    train_contexts_normal = normalize_context(train_contexts, context_range)
    eval_contexts_normal = normalize_context(eval_contexts, context_range)

    # since to we are predicting next timestep instead of the current one, we need to take one more timestep for the input (window size + 1)
    length = args.window_size + 1
    max_window_num = args.n_history + args.input_padding - length + 1

    # generate indices for each training iteration and shuffle training samples
    n_total = args.n_data * args.n_traj * max_window_num
    flatten_indices = np.arange(0, n_total, dtype=int)

    # generate training batches
    def get_batch(batch_size, idx=0):
        # shuffle data at beginning of each epoch(step)
        if idx == 0:
            np.random.shuffle(flatten_indices)

        # get the index based on the current batch id and ids
        # this is necessary if we want to iterate through the entire training set without overlapping.
        batch_index_indices = (
            np.arange(idx * batch_size, (idx + 1) * batch_size, dtype=int) % n_total
        )
        batch_indices = flatten_indices[batch_index_indices]

        history_indices = batch_indices % max_window_num
        batch_indices = batch_indices // max_window_num
        traj_indices = batch_indices % args.n_traj
        batch_indices = batch_indices // args.n_traj
        data_indices = batch_indices % args.n_data

        indices = [
            data_indices,
            traj_indices,
            history_indices,
        ]

        (
            batch_contexts,
            batch_real_contexts,
            batch_action,
            batch_sim_traj,
            batch_real_traj,
            batch_timesteps,
            batch_masks,
        ) = generate_training_data(
            args,
            train_contexts_normal.copy(),
            train_sim_trajectories.copy(),
            train_real_trajectories.copy(),
            train_actions.copy(),
            indices=indices,
            debug=False,
        )

        return (
            to_tensor(batch_contexts, device=device),
            to_tensor(batch_real_contexts, device=device),
            to_tensor(batch_action, device=device),
            to_tensor(batch_sim_traj, device=device),
            to_tensor(batch_real_traj, device=device),
            to_tensor(batch_timesteps, device=device, dtype=torch.long),
            to_tensor(batch_masks, device=device, dtype=torch.long),
        )

    # save the default context
    env_tmp = init_env(env_name=env_name, n_envs=-1, render=False, dr=args.dr)
    env_tmp.reset()
    default_context = deepcopy(env_tmp.get_context())
    env_tmp.close()

    # initialize environments for evaluation
    # make sure the number of evaluation environment is smaller than the total evaluation contexts.
    args.n_envs = min(args.n_envs, args.n_data_eval)
    env_eval = init_env(env_name, n_envs=args.n_envs, render=False, dr=args.dr)

    # generate indices for the evaluation contexts
    """
    contexts normal shape:      (n_context,  n_data, n_history)
    whole context shape:        (n_data, n_history, n_context)
    """

    whole_eval_contexts_normal = np.transpose(eval_contexts_normal, (1, 2, 0))
    n_eval_total = whole_eval_contexts_normal.shape[0]
    whole_eval_indices = np.arange(0, whole_eval_contexts_normal.shape[0], dtype=int)

    # evaluation functions
    eval_policy_args = get_task_policy(
        args.eval_policy_name, args.eval_policy_checkpoint
    )

    def get_eval_fn(n_iteration):
        def fn(model):
            eval_batch_size = args.n_envs
            batch_eval_results = []

            # evaluate model
            model.eval()
            with torch.no_grad():
                for i in range(0, n_eval_total, eval_batch_size):
                    print(
                        f"Evaluating batch [{i}~{i + eval_batch_size}]/{n_eval_total}"
                    )
                    indices = (
                        np.arange(i, i + eval_batch_size, dtype=int) % n_eval_total
                    )
                    batch_indices = whole_eval_indices[indices]
                    batch_eval_contexts_normal = whole_eval_contexts_normal[
                        batch_indices
                    ]
                    batch_eval_result = evaluate_batch(
                        env_eval,
                        model,
                        batch_eval_contexts_normal,
                        env_name,
                        default_context,
                        args.eval_policy_name,
                        eval_policy_args,
                        window_size=args.window_size,
                        delta_output=args.loss_fn in ["delta_next", "delta_final"],
                        n_iteration=n_iteration,
                        relative_position=args.relative_position,
                        args=args,
                    )
                    batch_eval_results.append(batch_eval_result)
            model.train()

            # stack results
            eval_result = {}
            for k in batch_eval_results[0]:
                values = [x[k] for x in batch_eval_results]
                eval_result[k] = np.concatenate(values, axis=0)[:n_eval_total]

            eval_log = {}

            # log trajectories differences
            diff_traj_histories = (
                eval_result["input_sim_traj_histories"]
                - eval_result["input_real_traj_histories"]
            )
            """
            diff_traj_histories: (n_eval, n_history, traj_length * n_state)
            """
            diff_traj_histories = diff_traj_histories.reshape(
                n_eval_total, -1, state_traj_length, n_state
            )
            diff_traj_histories = np.mean(
                np.linalg.norm(diff_traj_histories, axis=-1), axis=-1
            )
            diff_traj_histories_mean = np.mean(diff_traj_histories, axis=0)
            diff_traj_histories_std = np.std(diff_traj_histories, axis=0)
            print("Evaluation trajectories diff:\n", diff_traj_histories_mean)
            eval_log["trajectories/diff mean"] = diff_traj_histories_mean[-1]
            eval_log["trajectories/diff std"] = diff_traj_histories_std[-1]

            # log state trajectories differences
            diff_context_histories = (
                eval_result["context_history"] - eval_result["context_real"]
            )
            diff_context_histories = np.abs(diff_context_histories) 
                
            diff_context_histories_mean = np.mean(diff_context_histories, axis=0)
            diff_context_histories_std = np.std(diff_context_histories, axis=0)
            print("Evaluation context diff:\n", diff_context_histories_mean.T)
            for i, context_name in enumerate(context_range_dict):
                eval_log[f"contexts/{context_name}_diff mean"] = (
                    diff_context_histories_mean[-1, i]
                )
                eval_log[f"contexts/{context_name}_diff std"] = (
                    diff_context_histories_std[-1, i]
                )

            # visualize the trajectories changes over history
            if log_to_wandb:
                figs = plot_traj(env_name, eval_result)
                for k, fig in figs.items():
                    eval_log[k] = wandb.Image(fig)

            # visualize the context changes during adaptation.
            return eval_log

        return fn

    # initialize model
    model = get_model(args, n_context, n_state, n_action, state_traj_length)
    model = model.to(device=device)

    # load model parameter from a checkpoint if specified
    save_dir = get_model_dir(args)
    if args.load_checkpoint >= 0:
        assert args.mode == "eval", "Please only load saved model during evaluation."
        checkpoint_path = os.path.join(
            save_dir, f"checkpoint_{args.load_checkpoint}.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"load model: {checkpoint_path}")

    # Setup wandb
    if log_to_wandb:
        group_name = get_model_dir(args)
        exp_prefix = f'{group_name}-{time.strftime("%Y-%m-%d-%H-%M")}'
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=f"sim2real-capture {args.env_name}",
            config=args,
        )
        wandb.watch(model)

    # setup model trainer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.max_iters * args.num_steps_per_iter
    scheduler = WarmupInverseSqrtSchedule(optimizer, args.warmup_steps, total_steps)

    if args.loss_fn == "delta_next":
        # set up the loss function to compute the different to the next timestep context
        loss_fn = delta_next_loss_func
    elif args.loss_fn == "delta_final":
        # set up the loss function to compute the different to the real (final timestep) context
        loss_fn = delta_final_loss_func
    elif args.loss_fn == "next":
        # set up the loss function to compute the different to the real (final timestep) context
        loss_fn = next_loss_func
    elif args.loss_fn == "final":
        # set up the loss function to compute the different to the real (final timestep) context
        loss_fn = final_loss_func
    else:
        raise ValueError(f"Unsupported loss function {args.loss_fn}")

    if args.mode == "eval":
        eval_iteration = 30 # This is for paper plotting putpose, we evlauated 30 steps
    else:
        eval_iteration = 7
    
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=[get_eval_fn(eval_iteration)],
        max_length=args.window_size,
        seed=args.seed,
        device=device,
        args=args,
        theta_dim=n_context,
    )

    # print training/evaluation arguments
    print("Experiment parameters")
    args_keys = sorted(list(vars(args).keys()))
    max_key_length = max(len(k) for k in args_keys)
    var_args = vars(args)
    for k in args_keys:
        print(f"    {k.ljust(max_key_length)} - {var_args[k]}")

    if args.mode == "train":
        for iter in range(1, args.max_iters + 1):
            # train the model
            outputs = trainer.train_iteration(
                num_steps=args.num_steps_per_iter,
                training_len=n_total,
            )

            # save the model
            if iter % args.save_interval == 0:
                trainer.save(save_dir, iter)

            # evaluate the model
            if iter % args.evaluation_interval == 0:
                eval_outputs = trainer.evaluate()
                outputs.update(eval_outputs)

            # log to wandb
            if log_to_wandb:
                wandb.log(outputs, step=iter * args.num_steps_per_iter, commit=True)
    else:
        outputs = trainer.evaluate()
        if log_to_wandb:
            wandb.log(outputs, step=0, commit=True)


def validate_args(args):
    if args.mode == "eval":
        assert (
            args.load_checkpoint > 0
        ), "Need to specify checkpoint iteration in evaluation mode."


def main(args):
    experiment(args)


if __name__ == "__main__":
    parser = get_parser()
    # parameters about training set and evaluation set
    parser.add_argument(
        "--n_data_eval",
        type=int,
        default=100,
        help="The number of samples in the evaluation set.",
    )
    # parameters about the model
    parser.add_argument("--method", type=str, default="ad", choices=["ad", "ed"])
    parser.add_argument("--window_size", type=int, default=4) # hyperparameter used in paper: k = 4
    parser.add_argument("--model_type", type=str, default="dt")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--loss_fn", type=str, default="next")
    parser.add_argument("--single_context_token", action="store_true")
    parser.add_argument("--input_padding", "-ip", type=int, default=3)
    parser.add_argument("--relative_position", "-rp", action="store_true") # relative position for positional embeddings 
    # parameters about the training process
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--num_steps_per_iter", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--evaluation_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--log_to_wandb", "-w", action="store_true")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    # parameters for the evaluation
    parser.add_argument("--n_envs", type=int, default=12)
    parser.add_argument("--load_checkpoint", type=int, default=-1)
    parser.add_argument("--eval_policy_name", type=str, default="auto")
    parser.add_argument("--eval_policy_checkpoint", type=str, default=None)
    
    # main function
    args = parser.parse_args()
    main(args)

