import numpy as np
import torch

from capture.decision_transformer.training.trainer import Trainer

# from capture.decision_transformer.training.trainer import Trainer_ED
# from capture.decision_transformer.training.trainer import Trainer_TuneNet


class SequenceTrainer(Trainer):

    def train_step(self, training_len):

        total_loss = 0
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

            loss = self.loss_fn(
                preds,
                theta_input,
                theta_target,
                theta_real,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            total_loss += loss.detach().cpu().item()

        return total_loss / n_batch