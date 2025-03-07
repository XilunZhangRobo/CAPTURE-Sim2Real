import numpy as np
import torch
import torch.nn as nn

import transformers

from capture.decision_transformer.models.model import TrajectoryModel
from capture.decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):
    """
    This model uses GPT to model (Theta_1, Episode_1, Delta_theta_1, Theta_2, Episode_2, ...)
    """

    def __init__(
        self,
        theta_dim,
        traj_dim,
        action_dim,
        state_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,  # normal length should be 2018 (10 iters * (7D+204D+7D)), maybe max 2500
        action_tanh=True,
        **kwargs
    ):
        super().__init__(theta_dim, traj_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.theta_dim = theta_dim
        self.trajectory_dim = traj_dim
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.state_traj_length = traj_dim // state_dim
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        
        self.single_context_token = kwargs["single_context_token"]

        if kwargs["relative_position"]:
            self.embed_timestep = nn.Embedding(max_length, hidden_size)
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            
        if self.single_context_token:
            self.embed_single_context = nn.Linear(1, hidden_size)
        else: 
            self.embed_theta = torch.nn.Linear(self.theta_dim, hidden_size)

        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_theta = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.theta_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )
        self.embed_single_traj = nn.Linear(self.state_dim, hidden_size)

    def forward(
        self, theta, action, sim_traj, real_traj, timesteps, attention_mask=None
    ):
        """
        theta           torch.Size([batch_size, window_size, n_context])
        action          torch.Size([batch_size, window_size, n_action])
        sim_traj        torch.Size([batch_size, window_size, trajectory_length * n_state])
        real_traj       torch.Size([batch_size, window_size, trajectory_length * n_state]])
        timesteps       torch.Size([batch_size, window_size])
        attention_mask  torch.Size([batch_size, window_size])
        """

        # theta.shape = [# of diff trajs, # of iterations, 5]
        batch_size, seq_length = theta.shape[0], theta.shape[1]

        # idx = 5
        # print("theta", theta.shape, theta[idx, :, 0])
        # print("action", action.shape, action[idx, :, 0])
        # print("sim_traj", sim_traj.shape, sim_traj[idx, :, 0])
        # print("real_traj", real_traj.shape, real_traj[idx, :, 0])
        # print("timesteps", timesteps.shape, timesteps[idx])
        # print("attention_mask", attention_mask.shape, attention_mask[idx])
        # # print("theta")
        # input("Press any key to continue")

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # sim_traj = torch.zeros_like(sim_traj)

        # sim_traj_embeddings = self.embed_trajectory(sim_traj)
        # real_traj_embeddings = self.embed_trajectory(real_traj)
        # theta_embeddings = self.embed_theta(theta)
        # action_embeddings = self.embed_action(action)
        # time_embeddings = self.embed_timestep(timesteps)

        # theta_embeddings = theta_embeddings + time_embeddings
        # action_embeddings = action_embeddings + time_embeddings
        # sim_traj_embeddings = sim_traj_embeddings + time_embeddings
        # real_traj_embeddings = real_traj_embeddings + time_embeddings

        # # Assuming theta_embeddings is your starting point and is a tensor
        # stacked_inputs = [theta_embeddings]  # Start with theta_embeddings in the list

        # # Add sim_trajectory_embeddings to the list
        # stacked_inputs.append(sim_traj_embeddings)

        # # Add real_trajectory_embeddings to the list
        # stacked_inputs.append(real_traj_embeddings)

        # # Finally, add theta_embeddings to the list
        # stacked_inputs.append(action_embeddings)

        # stacked_inputs_tensor = torch.stack(stacked_inputs)
        # stacked_inputs = stacked_inputs_tensor.permute(1, 2, 0, 3).reshape(
        #     batch_size, 4 * seq_length, -1
        # )
        # stacked_inputs = self.embed_ln(stacked_inputs)

        # stacked_attention_mask = []
        # for i in range(4):
        #     stacked_attention_mask.append(attention_mask)
        # stacked_attention_mask_tensor = torch.stack(stacked_attention_mask)

        # transformer_outputs = self.transformer(
        #     inputs_embeds=stacked_inputs,
        #     attention_mask=stacked_attention_mask_tensor,
        # )
        # x = transformer_outputs["last_hidden_state"]

        # # reshape x so that the second dimension corresponds to the original
        # # theta (0), sim_traj (1), real_traj (2),action (3); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)
        # theta_preds = self.predict_theta(x[:, 3])[:, -seq_length:, :]

        ## original code
        sim_traj = sim_traj.reshape(
            batch_size, seq_length, self.state_traj_length, self.state_dim
        )
        real_traj = real_traj.reshape(
            batch_size, seq_length, self.state_traj_length, self.state_dim
        )
        time_embeddings = self.embed_timestep(timesteps)
        if self.single_context_token:
            context = theta.reshape(batch_size, seq_length, self.theta_dim, 1)
            context_embeddings = [
                self.embed_single_context(context[:, :, i]) for i in range(self.theta_dim)
            ]
            for i in range(len(context_embeddings)):
                context_embeddings[i] = context_embeddings[i] + time_embeddings
        else:
            theta_embeddings = self.embed_theta(theta)
            theta_embeddings = theta_embeddings + time_embeddings

        sim_trajectory_embeddings = [
            self.embed_single_traj(sim_traj[:, :, i]) for i in range(sim_traj.shape[2])
        ]
        # print ("shape", real_traj[:,:,2].shape)
        real_trajectory_embeddings = [
            self.embed_single_traj(real_traj[:, :, i])
            for i in range(real_traj.shape[2])
        ]

        action_embeddings = self.embed_action(action)
        action_embeddings = action_embeddings + time_embeddings

        ## add time_embeddings to each component in the sim_trajectory_embeddings and real_trajectory_embeddings lists
        for i in range(len(sim_trajectory_embeddings)):
            sim_trajectory_embeddings[i] = (
                sim_trajectory_embeddings[i] + time_embeddings
            )
            real_trajectory_embeddings[i] = (
                real_trajectory_embeddings[i] + time_embeddings
            )

        # Assuming theta_embeddings is your starting point and is a tensor
        stacked_inputs = []  # Start with theta_embeddings in the list
        if self.single_context_token:
            for context_emb in context_embeddings:
                stacked_inputs.append(context_emb)
        else:
            stacked_inputs.append(theta_embeddings)

        # Add sim_trajectory_embeddings to the list
        for sim_traj_emb in sim_trajectory_embeddings:
            stacked_inputs.append(sim_traj_emb)

        # Add real_trajectory_embeddings to the list
        for real_traj_emb in real_trajectory_embeddings:
            stacked_inputs.append(real_traj_emb)

        # Finally, add theta_embeddings to the list
        stacked_inputs.append(action_embeddings)

        stacked_inputs_tensor = torch.stack(stacked_inputs)
        # 22: 10 sim + 10 real + 1 action + 1
        context_length = self.theta_dim if self.single_context_token else 1
        n_input = self.state_traj_length * 2 + 1 + context_length
        stacked_inputs = stacked_inputs_tensor.permute(1, 2, 0, 3).reshape(
            batch_size, n_input * seq_length, -1
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Assuming attention_mask is a tensor
        stacked_attention_mask = []
        for i in range(n_input):
            stacked_attention_mask.append(attention_mask)
        stacked_attention_mask_tensor = torch.stack(stacked_attention_mask)
        stacked_attention_mask = stacked_attention_mask_tensor.permute(1, 2, 0).reshape(
            batch_size, n_input * seq_length
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # Here the input and output size are the same
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # theta (0), sim_traj (1-10), real_traj (10-20),action (21),  theta_1; i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, n_input, self.hidden_size).permute(
            0, 2, 1, 3
        )
        # print ("x_shape", x.shape)
        theta_preds = self.predict_theta(x[:, -1])[:, -seq_length:, :]
        # input()
        return None, None, theta_preds

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
