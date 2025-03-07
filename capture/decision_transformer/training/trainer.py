import numpy as np
import torch
import os
import time
from tqdm import tqdm
from scripts.utilities import get_model_dir, get_unwrapped_model


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        scheduler=None,
        eval_fns=None,
        max_length=None,
        seed=0,
        device="cuda",
        args=None,
        theta_dim=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.max_length = max_length
        self.seed = seed
        self.device = device
        self.args = args
        self.theta_dim = theta_dim

        self.start_time = time.time()

    def train_iteration(
        self,
        num_steps,
        training_len,
    ):

        logs = dict()
        train_losses = []
        train_start = time.time()
        self.model.train()
        print("################# Training #################")
        for _ in tqdm(range(num_steps), desc=f"Train iteration", ascii=" >="):
            train_loss = self.train_step(training_len)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs["time/total"] = time.time() - self.start_time
        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)
        logs["training/learning_rate"] = self.scheduler.get_lr()[0]
        return logs

    def evaluate(self):
        print("################# Evaluation #################")
        logs = dict()
        eval_start = time.time()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v

        logs["time/evaluation"] = time.time() - eval_start
        return logs

    def save(self, save_dir, iter):
        print("################# Saving checkpoint #################")
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_{iter}.pt")
        model = get_unwrapped_model(self.model)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def train_step(self):
        return NotImplementedError


# class Trainer_ED:

#     def __init__(
#         self,
#         model,
#         optimizer,
#         batch_size,
#         get_batch,
#         loss_fn,
#         scheduler=None,
#         eval_fns=None,
#         max_length=None,
#         seed=0,
#         device="cuda",
#         variant=None,
#         theta_dim=3,
#     ):
#         self.model = model
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.get_batch = get_batch
#         self.loss_fn = loss_fn
#         self.scheduler = scheduler
#         self.eval_fns = [] if eval_fns is None else eval_fns
#         self.diagnostics = dict()
#         self.max_length = max_length
#         self.seed = seed
#         self.device = device
#         self.variant = variant
#         self.theta_dim = theta_dim

#         self.start_time = time.time()

#     def train_iteration(
#         self,
#         num_steps,
#         iter_num=0,
#         print_logs=False,
#         evaluation_only=False,
#         training_len=1280,
#     ):

#         ## TODO: Define Root Path
#         root_path = "/home/xilun/Sim2Real_DT"

#         logs = dict()
#         # set seed for reproducibility
#         torch.manual_seed(seed=self.seed)
#         if not evaluation_only:
#             train_losses = []

#             train_start = time.time()
#             self.model.train()
#             print("#################Training#################")
#             for _ in tqdm(range(num_steps), ascii=" >="):
#                 train_loss = self.train_step(training_len)
#                 train_losses.append(train_loss)
#                 if self.scheduler is not None:
#                     self.scheduler.step()

#             logs["time/training"] = time.time() - train_start
#             logs["training/train_loss_mean"] = np.mean(train_losses)
#             logs["training/train_loss_std"] = np.std(train_losses)
#             # save training model
#             # create a folder if folder is not exist
#             if not os.path.exists(
#                 root_path
#                 + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_ED_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}'
#             ):
#                 os.makedirs(
#                     root_path
#                     + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_ED_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}'
#                 )

#             torch.save(
#                 self.model.state_dict(),
#                 root_path
#                 + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_ED_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}/Batch_train_{iter_num}.pt',
#             )
#         else:
#             # load model
#             model_path = (
#                 root_path
#                 + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_ED_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}/Batch_train_10.pt'
#             )
#             # The map location needs to be checked, has not varify does it affect the results.
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))

#         eval_start = time.time()

#         # self.model.eval()
#         # self.model.train()

#         for eval_fn in self.eval_fns:
#             outputs = eval_fn(self.model)
#             for k, v in outputs.items():
#                 logs[f"evaluation/{k}"] = v

#         logs["time/total"] = time.time() - self.start_time
#         logs["time/evaluation"] = time.time() - eval_start

#         for k in self.diagnostics:
#             logs[k] = self.diagnostics[k]

#         if print_logs:
#             print("=" * 80)
#             print(f"Iteration {iter_num}")
#             for k, v in logs.items():
#                 print(f"{k}: {v}")

#         return logs

#     def train_step(self):

#         return NotImplementedError


# class Trainer_TuneNet:

#     def __init__(
#         self,
#         model,
#         optimizer,
#         batch_size,
#         get_batch,
#         loss_fn,
#         scheduler=None,
#         eval_fns=None,
#         max_length=None,
#         seed=0,
#         device="cuda",
#         variant=None,
#         theta_dim=3,
#     ):
#         self.model = model
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.get_batch = get_batch
#         self.loss_fn = loss_fn
#         self.scheduler = scheduler
#         self.eval_fns = [] if eval_fns is None else eval_fns
#         self.diagnostics = dict()
#         self.max_length = max_length
#         self.seed = seed
#         self.device = device
#         self.variant = variant
#         self.theta_dim = theta_dim

#         self.start_time = time.time()

#     def train_iteration(
#         self,
#         num_steps,
#         iter_num=0,
#         print_logs=False,
#         evaluation_only=False,
#         training_len=1280,
#     ):

#         ## TODO: Define Root Path
#         root_path = "/home/xilun/Sim2Real_DT"

#         logs = dict()
#         # set seed for reproducibility
#         torch.manual_seed(seed=self.seed)
#         if not evaluation_only:
#             train_losses = []

#             train_start = time.time()
#             self.model.train()
#             print("#################Training#################")
#             for _ in tqdm(range(num_steps), ascii=" >="):
#                 train_loss = self.train_step(training_len)
#                 train_losses.append(train_loss)
#                 if self.scheduler is not None:
#                     self.scheduler.step()

#             logs["time/training"] = time.time() - train_start
#             logs["training/train_loss_mean"] = np.mean(train_losses)
#             logs["training/train_loss_std"] = np.std(train_losses)
#             # save training model
#             # create a folder if folder is not exist
#             if not os.path.exists(
#                 root_path
#                 + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_TuneNet_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}'
#             ):
#                 os.makedirs(
#                     root_path
#                     + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_TuneNet_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}'
#                 )

#             torch.save(
#                 self.model.state_dict(),
#                 root_path
#                 + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_TuneNet_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}/Batch_train_{iter_num}.pt',
#             )
#         else:
#             # load model
#             model_path = (
#                 root_path
#                 + f'/gym/decision_transformer/checkpoints/{self.variant["env"]}_TuneNet_oor{self.variant["out_of_range"]}_interp{self.variant["input_ood"]}/{self.variant["data_file"]}/seed_{self.variant["seed"]}_cd{self.variant["custom_dataset"]}/Batch_train_10.pt'
#             )
#             # The map location needs to be checked, has not varify does it affect the results.
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))

#         eval_start = time.time()

#         # self.model.eval()
#         # self.model.train()

#         for eval_fn in self.eval_fns:
#             outputs = eval_fn(self.model)
#             for k, v in outputs.items():
#                 logs[f"evaluation/{k}"] = v

#         logs["time/total"] = time.time() - self.start_time
#         logs["time/evaluation"] = time.time() - eval_start

#         for k in self.diagnostics:
#             logs[k] = self.diagnostics[k]

#         if print_logs:
#             print("=" * 80)
#             print(f"Iteration {iter_num}")
#             for k, v in logs.items():
#                 print(f"{k}: {v}")

#         return logs

#     def train_step(self):

#         return NotImplementedError
