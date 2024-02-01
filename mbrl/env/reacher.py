# Neelay
import numpy as np
import torch
from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv


class CustomReacherEnv(ReacherEnv):
    # Lengths of reacher arms 1 and 2
    l1 = 0.1
    l2 = 0.1

    def step(self, action):
        ob, _reward, terminated, truncated, info = super().step(action)

        # Estimate fingertip position
        assert len(ob) == 11  # Original size
        theta1 = np.arctan2(ob[2], ob[0])
        theta2 = np.arctan2(ob[3], ob[1])
        fingertipx = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        fingertipy = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)

        targetx = ob[4]
        targety = ob[5]

        dist_squared = (fingertipx - targetx) ** 2 + (fingertipy - targety) ** 2
        reward_pos = np.exp(-dist_squared)

        reward_ctrl = 0.5 * np.exp(-(action**2).sum())

        reward = reward_pos + reward_ctrl

        return ob, reward, terminated, truncated, info

    @staticmethod
    def _preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = np.expand_dims(state, 0)
        # [0., 1., 2., 3., 4., 5.] ->
        # [1., 2., 3.]
        ret = state[..., 1:4]
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def _preprocess_state_torch(state):
        assert isinstance(state, torch.Tensor)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = state.unsqueeze(0)
        # [0., 1., 2., 3., 4., 5.] ->
        # [1., 2., 3.]
        ret = state[..., 1:4]
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    # Symmetry projection
    # Assumes state is as output of observation_edit
    @staticmethod
    def preprocess_fn(state):
        if isinstance(state, np.ndarray):
            return CustomReacherEnv._preprocess_state_np(state)
        if isinstance(state, torch.Tensor):
            return CustomReacherEnv._preprocess_state_torch(state)
        raise ValueError("Invalid state type (must be np.ndarray or torch.Tensor).")

    @staticmethod
    def observation_edit(obs):
        assert len(obs) == 11
        new_obs = np.zeros(6)
        new_obs[0] = np.arctan2(obs[2], obs[0])  # Angle of first arm
        new_obs[1] = np.arctan2(obs[3], obs[1])  # Angle of second arm
        new_obs[2] = obs[6]  # Angular velocity of first arm
        new_obs[3] = obs[7]  # Angular velocity of second arm
        new_obs[4] = obs[4]  # x-coordinate of target
        new_obs[5] = obs[5]  # y-coordinate of target
        return new_obs
