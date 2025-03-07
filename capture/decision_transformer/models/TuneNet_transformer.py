import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):

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
            max_ep_len=4096, # normal length should be 2018 (10 iters * (7D+204D+7D)), maybe max 2500
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
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_theta = torch.nn.Linear(self.theta_dim, hidden_size)
        self.embed_delta_theta = torch.nn.Linear(self.theta_dim, hidden_size)
        self.embed_trajectory = torch.nn.Linear(self.trajectory_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_theta = nn.Sequential(
            *([nn.Linear(hidden_size, self.theta_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        
        self.embed_traj_timestep = nn.Embedding(self.trajectory_dim, hidden_size)
        if kwargs['env_name'] == 'pusher':
            self.embed_single_traj = nn.Linear(2, hidden_size)
        else:
            self.embed_single_traj = nn.Linear(3, hidden_size)

    def forward(self, delta_theta, theta, action, sim_traj, real_traj, timesteps, attention_mask=None):

        # theta.shape = [# of diff trajs, # of iterations, 5]
        batch_size, seq_length = delta_theta.shape[0], delta_theta.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            
        sim_traj = sim_traj.reshape(batch_size, seq_length, 10, -1)
        real_traj = real_traj.reshape(batch_size, seq_length, 10, -1)
        
        sim_trajectory_embeddings = [self.embed_single_traj(sim_traj[:,:,i]) for i in range(sim_traj.shape[2])]
        real_trajectory_embeddings = [self.embed_single_traj(real_traj[:,:,i]) for i in range(real_traj.shape[2])]
        
        delta_theta_embeddings = self.embed_delta_theta(delta_theta)
        theta_embeddings = self.embed_theta(theta)
        action_embeddings = self.embed_action(action)
        time_embeddings = self.embed_timestep(timesteps)

        theta_embeddings = theta_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        delta_theta_embeddings = delta_theta_embeddings + time_embeddings
        
        ## add time_embeddings to each component in the sim_trajectory_embeddings and real_trajectory_embeddings lists 
        for i in range(len(sim_trajectory_embeddings)):
            sim_trajectory_embeddings[i] = sim_trajectory_embeddings[i] + time_embeddings
            real_trajectory_embeddings[i] = real_trajectory_embeddings[i] + time_embeddings
        
        
        # Assuming theta_embeddings is your starting point and is a tensor
        stacked_inputs = [delta_theta_embeddings]  # Start with theta_embeddings in the list
        # stacked_inputs = [action_embeddings]  # Start with action_embeddings in the list
        stacked_inputs.append(theta_embeddings)

        # Add sim_trajectory_embeddings to the list
        for sim_traj_emb in sim_trajectory_embeddings:
            stacked_inputs.append(sim_traj_emb)

        # Add real_trajectory_embeddings to the list
        for real_traj_emb in real_trajectory_embeddings:
            stacked_inputs.append(real_traj_emb)

        # Finally, add theta_embeddings to the list
        stacked_inputs.append(action_embeddings)
        # stacked_inputs.append(theta_embeddings)
        
        stacked_inputs_tensor = torch.stack(stacked_inputs)
        stacked_inputs = stacked_inputs_tensor.permute(1,2,0,3).reshape(batch_size, 23*seq_length, -1)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Assuming attention_mask is a tensor
        stacked_attention_mask = []
        for i in range(23):
            stacked_attention_mask.append(attention_mask)
        stacked_attention_mask_tensor = torch.stack(stacked_attention_mask)
        stacked_attention_mask = stacked_attention_mask_tensor.permute(1,2,0).reshape(batch_size, 23*seq_length)
    
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # Here the input and output size are the same
        x = transformer_outputs['last_hidden_state']
    
        # reshape x so that the second dimension corresponds to the original
        # delta_theta(0), theta (1), sim_traj (2-11), real_traj (11-21),action (22),  theta_1; i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 23, self.hidden_size).permute(0, 2, 1, 3)
        # print ('x', x[45])
        delta_theta_preds = self.predict_theta(x[:,22])[:, -seq_length:, :]


        return None, None, delta_theta_preds

    def get_action(self, delta_theta, theta, action, sim_traj, real_traj, timesteps, batch_size, **kwargs):
        
        theta = theta.clone().reshape(batch_size, -1, self.theta_dim)
        delta_theta = delta_theta.clone().reshape(batch_size, -1, self.theta_dim)
        action = action.clone().reshape(batch_size, -1, self.act_dim)
        sim_traj = sim_traj.clone().reshape(batch_size, -1, self.trajectory_dim)
        real_traj = real_traj.clone().reshape(batch_size, -1, self.trajectory_dim)
        timesteps = timesteps.clone().reshape(batch_size, -1)
        

        if self.max_length is not None:
            delta_theta = delta_theta[:,-self.max_length:]
            theta = theta[:,-self.max_length:]
            action = action[:,-self.max_length:]
            sim_traj = sim_traj[:,-self.max_length:]
            real_traj = real_traj[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            
            delta_theta = torch.cat(
                [torch.zeros((delta_theta.shape[0], self.max_length-delta_theta.shape[1], self.theta_dim), device=delta_theta.device), delta_theta],
                dim=1).to(dtype=torch.float32)
            theta = torch.cat(
                [torch.zeros((theta.shape[0], self.max_length-theta.shape[1], self.theta_dim), device=theta.device), theta],
                dim=1).to(dtype=torch.float32)
            action = torch.cat(
                [torch.zeros((action.shape[0], self.max_length-action.shape[1], self.act_dim), device=action.device), action],
                dim=1).to(dtype=torch.float32)
            sim_traj = torch.cat(
                [torch.zeros((sim_traj.shape[0], self.max_length - sim_traj.shape[1], self.trajectory_dim),
                             device=sim_traj.device), sim_traj],
                dim=1).to(dtype=torch.float32)
            real_traj = torch.cat(
                [torch.zeros((real_traj.shape[0], self.max_length - real_traj.shape[1], self.trajectory_dim),
                             device=real_traj.device), real_traj],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            
            attention_mask = torch.ones_like(delta_theta, dtype=torch.long, device=theta.device)
            attention_mask[delta_theta == 0] = 0
            attention_mask = attention_mask[:,:,0]
            
        else:
            attention_mask = None
            
        assert torch.all(delta_theta[torch.where(attention_mask == 1)] != 0)
        assert torch.all(delta_theta[torch.where(attention_mask == 0)] == 0)
        # print ('attention_mask', attention_mask)
        # print ('delta_theta', delta_theta)
        assert torch.all(attention_mask[:,-1] == 1)
        print ('timesteps', timesteps.shape, timesteps[0], timesteps[-1])
        _, _, delta_theta_preds = self.forward(
            delta_theta, theta, action, sim_traj, real_traj, timesteps, attention_mask=attention_mask, **kwargs)
        

        return delta_theta_preds[:, -1], delta_theta_preds
