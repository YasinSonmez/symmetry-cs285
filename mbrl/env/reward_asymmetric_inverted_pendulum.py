# Neelay
import numpy as np
import torch
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv


class RewardAsymmetricInvertedPendulumEnv(InvertedPendulumEnv):
    goal_pos = 1.0

    def step(self, action):
        ob, _reward, _terminated, truncated, info = super().step(action)

        x = ob[0]
        theta = ob[1]
        pos_reward = np.exp(-((x - self.goal_pos) ** 2))
        upright_reward = np.exp(-(theta**2))
        reward = pos_reward + upright_reward

        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.79))

        return ob, reward, terminated, truncated, info

    @staticmethod
    def _preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = np.expand_dims(state, 0)
        # [0., 1., 2., 3.] ->
        # [1., 2., 3.]
        ret = state[..., 1:]
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
        # [0., 1., 2., 3.] ->
        # [1., 2., 3.]
        ret = state[..., 1:]
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    # Symmetry projection
    @staticmethod
    def preprocess_fn(state):
        if isinstance(state, np.ndarray):
            return RewardAsymmetricInvertedPendulumEnv._preprocess_state_np(state)
        if isinstance(state, torch.Tensor):
            return RewardAsymmetricInvertedPendulumEnv._preprocess_state_torch(state)
        raise ValueError("Invalid state type (must be np.ndarray or torch.Tensor).")
