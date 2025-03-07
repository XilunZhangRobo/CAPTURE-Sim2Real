import gym
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


class NormalizeActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Store both the high and low arrays in their original forms
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high

        # We normalize action space to a range [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
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


class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]
        

class WarmupInverseSqrtSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupInverseSqrtSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Inverse square root decay
            inverse_sqrt_decay = np.sqrt(self.warmup_steps / (self.last_epoch + 1))
            return [base_lr * inverse_sqrt_decay for base_lr in self.base_lrs]