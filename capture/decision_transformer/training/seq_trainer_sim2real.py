import numpy as np
import torch

from capture.decision_transformer.training.trainer import Trainer

# from capture.decision_transformer.training.trainer import Trainer_ED
# from capture.decision_transformer.training.trainer import Trainer_TuneNet


class SequenceTrainer(Trainer):

    def train_step(self, training_len):
        # all_peds = []
        # all_theta_input = []
        # all_theta_target = []
        # all_theta_real = []

        total_loss = 0
        # print("data", training_len, "batch size", self.batch_size)
        n_batch = max(training_len // self.batch_size, 1)
        for idx in range(n_batch):

            theta, theta_real, act, sim_traj, real_traj, timesteps, attention_mask = (
                self.get_batch(self.batch_size, idx)
            )

            theta_target = torch.clone(theta[:, 1:, :])
            theta_real = torch.clone(theta_real[:, 1:, :])

            theta_input = torch.clone(theta[:, :-1, :])
            act_input = torch.clone(act[:, :-1, :])
            sim_traj_input = torch.clone(sim_traj[:, :-1, :])
            real_traj_input = torch.clone(real_traj[:, :-1, :])
            timesteps_input = torch.clone(timesteps[:, :-1])
            attention_mask_input = torch.clone(attention_mask[:, :-1])

            _, _, preds = self.model.forward(
                theta_input,
                act_input,
                sim_traj_input,
                real_traj_input,
                timesteps_input,
                attention_mask=attention_mask_input,
            )
            # print(
            #     theta_input,
            #     act_input,
            #     sim_traj_input,
            #     real_traj_input,
            #     timesteps_input,
            #     attention_mask_input,
            # )

            # idx = 10
            # print("theta_input", theta_input[idx])
            # print("theta_target", theta_target[idx])
            # print("theta_real", theta_real[idx])
            # print("act_input", act_input[idx])
            # print("sim_traj_input", sim_traj_input[idx])
            # print("real_traj_input", real_traj_input[idx])
            # print("timesteps_input", timesteps_input[idx])
            # print("attention_mask_input", attention_mask_input[idx])
            # quit()

            # print("theta_target", theta_target.shape)
            # print("preds", preds.shape)
            # print("theta_real", theta_real.shape)
            # print("theta_input", theta_input.shape)

            # attention_mask = attention_mask_input.unsqueeze(-1).repeat(1, 1, theta_preds.shape[2])

            # mask = attention_mask > 0
            # theta_preds = torch.where(mask, theta_preds, torch.zeros_like(theta_preds))
            # theta_target = torch.where(mask, theta_target, torch.zeros_like(theta_target))

            # compute the loss
            theta_input = torch.clone(theta[:, :-1, :])
            theta_dim = preds.shape[2]
            preds = preds.reshape(-1, theta_dim)[attention_mask_input.reshape(-1) > 0]
            theta_target = theta_target.reshape(-1, theta_dim)[
                attention_mask_input.reshape(-1) > 0
            ]
            theta_input = theta_input.reshape(-1, theta_dim)[
                attention_mask_input.reshape(-1) > 0
            ]
            theta_real = theta_real.reshape(-1, theta_dim)[
                attention_mask_input.reshape(-1) > 0
            ]

            # with torch.no_grad():

            #     for i in range(preds.shape[1]):
            #         prediction_los[idx, i] = torch.mean(
            #             torch.abs(preds[:, i] + theta_input[:, i] - theta_target[:, i])
            #         )
            #     prediction_l1[idx] = torch.mean(
            #         torch.abs(preds + theta_input - theta_target)
            #     )
            #     prediction_l2[idx] = torch.mean(
            #         torch.norm(preds + theta_input - theta_target, dim=(-1))
            #     )

            loss = self.loss_fn(
                preds,
                theta_input,
                theta_target,
                theta_real,
            )

            # log predictions
            # predictions.append(preds.detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            total_loss += loss.detach().cpu().item()

        # predictions = np.concatenate(predictions, axis=0)
        # self.log["predictions"] = predictions

        # with torch.no_grad():
        #     self.diagnostics["training/prediction_error"] = (
        #         torch.mean(prediction_l2).detach().cpu().item()
        #     )
        #     self.diagnostics["training/prediction_error_l1"] = (
        #         torch.mean(prediction_l1).detach().cpu().item()
        #     )
        #     for i in range(preds.shape[1]):
        #         self.diagnostics[f"training/prediction_error_{i}"] = (
        #             torch.mean(prediction_los[:, i]).detach().cpu().item()
        #         )

        return total_loss / n_batch


# class SequenceTrainer_ED(Trainer_ED):

#     def train_step(self, training_len):
#         prediction_los = torch.zeros(
#             (training_len // self.batch_size, self.theta_dim), device=self.device
#         )
#         prediction_l2 = torch.zeros(
#             (training_len // self.batch_size), device=self.device
#         )
#         prediction_l1 = torch.zeros(
#             (training_len // self.batch_size), device=self.device
#         )
#         total_loss = 0
#         for idx in range(training_len // self.batch_size):

#             theta, act, sim_traj, real_traj, timesteps, attention_mask = self.get_batch(
#                 self.batch_size, idx, max_length=self.max_length
#             )

#             theta_target = torch.clone(theta[:, 1:, :])

#             theta_input = torch.clone(theta[:, :-1, :])
#             act_input = torch.clone(act[:, :-1, :])
#             sim_traj_input = torch.clone(sim_traj[:, :-1, :])
#             real_traj_input = torch.clone(real_traj[:, :-1, :])
#             timesteps_input = torch.clone(timesteps[:, :-1])
#             attention_mask_input = torch.clone(attention_mask[:, :-1])

#             _, _, theta_preds = self.model.forward(
#                 theta_input,
#                 act_input,
#                 sim_traj_input,
#                 real_traj_input,
#                 timesteps_input,
#                 attention_mask=attention_mask_input,
#             )

#             # attention_mask = attention_mask_input.unsqueeze(-1).repeat(1, 1, theta_preds.shape[2])

#             # mask = attention_mask > 0
#             # theta_preds = torch.where(mask, theta_preds, torch.zeros_like(theta_preds))
#             # theta_target = torch.where(mask, theta_target, torch.zeros_like(theta_target))

#             theta_dim = theta_preds.shape[2]
#             theta_preds = theta_preds.reshape(-1, theta_dim)[
#                 attention_mask_input.reshape(-1) > 0
#             ]
#             theta_target = theta_target.reshape(-1, theta_dim)[
#                 attention_mask_input.reshape(-1) > 0
#             ]

#             with torch.no_grad():
#                 for i in range(theta_preds.shape[1]):
#                     prediction_los[idx, i] = torch.mean(
#                         torch.abs(theta_preds[:, i] - theta_target[:, i])
#                     )
#                 prediction_l1[idx] = torch.mean(torch.abs(theta_preds - theta_target))
#                 prediction_l2[idx] = torch.mean(
#                     torch.norm(theta_preds - theta_target, dim=(-1))
#                 )

#             loss = self.loss_fn(
#                 theta_preds,
#                 theta_target,
#             )

#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
#             self.optimizer.step()

#             total_loss += loss.detach().cpu().item()

#         with torch.no_grad():
#             self.diagnostics["training/prediction_error"] = (
#                 torch.mean(prediction_l2).detach().cpu().item()
#             )
#             self.diagnostics["training/prediction_error_l1"] = (
#                 torch.mean(prediction_l1).detach().cpu().item()
#             )
#             for i in range(theta_preds.shape[1]):
#                 self.diagnostics[f"training/prediction_error_{i}"] = (
#                     torch.mean(prediction_los[:, i]).detach().cpu().item()
#                 )

#         return total_loss / (training_len // self.batch_size)


# class SequenceTrainer_TuneNet(Trainer_TuneNet):

#     def train_step(self, training_len):
#         prediction_los = torch.zeros(
#             (training_len // self.batch_size, self.theta_dim), device=self.device
#         )
#         prediction_l2 = torch.zeros(
#             (training_len // self.batch_size), device=self.device
#         )
#         prediction_l1 = torch.zeros(
#             (training_len // self.batch_size), device=self.device
#         )
#         total_loss = 0
#         for idx in range(training_len // self.batch_size):

#             theta, act, sim_traj, real_traj, timesteps, attention_mask = self.get_batch(
#                 self.batch_size, idx, max_length=self.max_length
#             )

#             delta_theta = torch.clone(theta[:, 1:, :] - theta[:, :-1, :])

#             delta_theta = torch.concatenate(
#                 [
#                     torch.zeros(
#                         (delta_theta.shape[0], 1, delta_theta.shape[2]),
#                         device=delta_theta.device,
#                     ),
#                     delta_theta,
#                 ],
#                 dim=1,
#             )

#             target_delta = delta_theta[:, -1]
#             delta_theta_input = delta_theta[:, :-1]

#             theta_input = torch.clone(theta[:, :-1, :])
#             act_input = torch.clone(act[:, :-1, :])
#             sim_traj_input = torch.clone(sim_traj[:, :-1, :])
#             real_traj_input = torch.clone(real_traj[:, :-1, :])
#             timesteps_input = torch.clone(timesteps[:, :-1])
#             attention_mask_input = torch.clone(attention_mask[:, :-1])

#             _, _, preds = self.model.forward(
#                 delta_theta_input,
#                 theta_input,
#                 act_input,
#                 sim_traj_input,
#                 real_traj_input,
#                 timesteps_input,
#                 attention_mask=attention_mask_input,
#             )

#             delta_theta_dim = preds.shape[2]
#             preds = preds.reshape(-1, delta_theta_dim)[
#                 attention_mask_input.reshape(-1) > 0
#             ]
#             target_delta = target_delta.reshape(-1, delta_theta_dim)[
#                 attention_mask_input.reshape(-1) > 0
#             ]

#             with torch.no_grad():
#                 for i in range(preds.shape[1]):
#                     prediction_los[idx, i] = torch.mean(
#                         torch.abs(preds[:, i] - target_delta[:, i])
#                     )
#                 prediction_l1[idx] = torch.mean(torch.abs(preds - target_delta))
#                 prediction_l2[idx] = torch.mean(
#                     torch.norm(preds - target_delta, dim=(-1))
#                 )

#             loss = self.loss_fn(
#                 preds,
#                 target_delta,
#             )

#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
#             self.optimizer.step()

#             total_loss += loss.detach().cpu().item()

#         with torch.no_grad():
#             self.diagnostics["training/prediction_error"] = (
#                 torch.mean(prediction_l2).detach().cpu().item()
#             )
#             self.diagnostics["training/prediction_error_l1"] = (
#                 torch.mean(prediction_l1).detach().cpu().item()
#             )
#             for i in range(preds.shape[1]):
#                 self.diagnostics[f"training/prediction_error_{i}"] = (
#                     torch.mean(prediction_los[:, i]).detach().cpu().item()
#                 )

#         return total_loss / (training_len // self.batch_size)
