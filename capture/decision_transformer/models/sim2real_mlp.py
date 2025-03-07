import numpy as np
import torch
import torch.nn as nn


from capture.decision_transformer.models.model import TrajectoryModel


class MLP(TrajectoryModel):
    """
    This model uses GPT to model (Theta_1, Episode_1, Delta_theta_1, Theta_2, Episode_2, ...)
    """

    def __init__(
        self,
        theta_dim,
        traj_dim,
        action_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,  # normal length should be 2018 (10 iters * (7D+204D+7D)), maybe max 2500
        action_tanh=True,
        **kwargs
    ):
        super().__init__(theta_dim, traj_dim, max_length=max_length)

        self.hidden_size = hidden_size

        self.theta_dim = theta_dim
        self.trajectory_dim = traj_dim
        self.act_dim = action_dim
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        if kwargs["relative_position"]:
            self.embed_timestep = nn.Embedding(max_length, hidden_size)
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_theta = torch.nn.Linear(self.theta_dim, hidden_size)
        self.embed_trajectory = torch.nn.Linear(self.trajectory_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_traj_timestep = nn.Embedding(self.trajectory_dim, hidden_size)
        self.layer1 = nn.Linear(hidden_size * 4, hidden_size)
        self.layer2 = nn.Linear(hidden_size * max_length, theta_dim * max_length)
        self.relu = nn.ReLU()
        self.max_length = max_length

    def forward(
        self, theta, action, sim_traj, real_traj, timesteps, attention_mask=None
    ):

        batch_size, seq_length = theta.shape[0], theta.shape[1]

        # sim_traj = torch.zeros_like(sim_traj)

        sim_traj_embeddings = self.embed_trajectory(sim_traj)
        real_traj_embeddings = self.embed_trajectory(real_traj)
        theta_embeddings = self.embed_theta(theta)
        action_embeddings = self.embed_action(action)
        time_embeddings = self.embed_timestep(timesteps)

        theta_embeddings = theta_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        sim_traj_embeddings = sim_traj_embeddings + time_embeddings
        real_traj_embeddings = real_traj_embeddings + time_embeddings

        x = torch.cat(
            [
                theta_embeddings,
                action_embeddings,
                sim_traj_embeddings,
                real_traj_embeddings,
            ],
            dim=-1,
        )
        x = self.relu(x)
        x = self.layer1(x)
        x = x.view(batch_size, -1)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(batch_size, self.max_length, -1)
        return None, None, x

    def get_action(
        self, theta, action, sim_traj, real_traj, timesteps, batch_size, **kwargs
    ):

        theta = theta.clone().reshape(batch_size, -1, self.theta_dim)
        action = action.clone().reshape(batch_size, -1, self.act_dim)
        sim_traj = sim_traj.clone().reshape(batch_size, -1, self.trajectory_dim)
        real_traj = real_traj.clone().reshape(batch_size, -1, self.trajectory_dim)
        timesteps = timesteps.clone().reshape(batch_size, -1)

        if self.max_length is not None:
            theta = theta[:, -self.max_length :]
            action = action[:, -self.max_length :]
            sim_traj = sim_traj[:, -self.max_length :]
            real_traj = real_traj[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]
            attention_mask = torch.ones_like(timesteps, dtype=torch.long)
            theta = torch.cat(
                [
                    torch.zeros(
                        (
                            theta.shape[0],
                            self.max_length - theta.shape[1],
                            self.theta_dim,
                        ),
                        device=theta.device,
                    ),
                    theta,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            action = torch.cat(
                [
                    torch.zeros(
                        (
                            action.shape[0],
                            self.max_length - action.shape[1],
                            self.act_dim,
                        ),
                        device=action.device,
                    ),
                    action,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            sim_traj = torch.cat(
                [
                    torch.zeros(
                        (
                            sim_traj.shape[0],
                            self.max_length - sim_traj.shape[1],
                            self.trajectory_dim,
                        ),
                        device=sim_traj.device,
                    ),
                    sim_traj,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            real_traj = torch.cat(
                [
                    torch.zeros(
                        (
                            real_traj.shape[0],
                            self.max_length - real_traj.shape[1],
                            self.trajectory_dim,
                        ),
                        device=real_traj.device,
                    ),
                    real_traj,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
            attention_mask = torch.cat(
                [
                    torch.zeros(
                        (
                            attention_mask.shape[0],
                            self.max_length - attention_mask.shape[1],
                        ),
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            ).to(dtype=torch.long)

            # attention_mask[theta == 0] = 0
            # attention_mask = attention_mask[:, :, 0]

        else:
            attention_mask = None

        # assert torch.all(theta[torch.where(attention_mask == 1)] != 0)
        # assert torch.all(theta[torch.where(attention_mask == 0)] == 0)

        assert torch.all(attention_mask[:, -1] == 1), attention_mask

        # print("timesteps", timesteps.shape, timesteps[0], timesteps[-1])
        _, _, theta_preds = self.forward(
            theta,
            action,
            sim_traj,
            real_traj,
            timesteps,
            attention_mask=attention_mask,
            **kwargs
        )
        # input()

        return theta_preds[:, -1], theta_preds
