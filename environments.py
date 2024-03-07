import gym

# gym 23.1 gives InvertedPendulum v2
# from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
import numpy as np
import torch
from gym.envs.mujoco.inverted_pendulum import (
    InvertedPendulumEnv,
)


class RewardAssymetricInvertedPendulum(InvertedPendulumEnv):
    goal_x = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, _reward, done, info = super().step(action)
        diff = obs[0] - self.goal_x
        b = 2
        c = 1.5
        reward = b / (np.power(c, diff) + np.power(c, -diff))
        # reward = -diff**2

        # notdone = np.isfinite(obs).all() and (np.abs(obs[1]) <= 0.2) # Changed angle threshold to 0.75rad from 0.2rad
        # done = not notdone
        return obs, reward, done, info

    def rho(x, gammax=None):
        assert x.ndim == 2
        return x[:, 1:]

    def phi(alpha, x):
        assert x.ndim == 2
        assert alpha.ndim == 1
        assert alpha.shape[0] == x.shape[0]
        xprime = x.clone()
        xprime[:, 0] += alpha
        return xprime

    def gamma(x):
        return -x[:, 0]

    def psi(alpha, u):
        return u

    def R(alpha):
        return torch.eye(4, device=alpha.device)

    def group_inv(alpha):
        return -alpha

    def submanifold_dim():
        return (3,)


class CarEnv(gym.Env):
    #  Car model from Maidens and Arcak paper.
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(self, render_mode=None):
        self.L = 1

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                10,
                10,
                10 * np.pi,
                10,
                10,
                10 * np.pi,
            ],
            dtype=np.float32,
        )

        act_high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.state = None

    def step(self, action):
        action = action.squeeze()
        v, s, vtilde, stilde = action
        z, y, theta, ztilde, ytilde, thetatilde = self.state

        zprime = z + v * np.cos(theta)
        yprime = y + v * np.sin(theta)
        thetaprime = theta + (1 / self.L) * v * np.sin(s)

        ztildeprime = ztilde + vtilde * np.cos(thetatilde)
        ytildeprime = ytilde + vtilde * np.sin(thetatilde)
        thetatildeprime = thetatilde + (1 / self.L) * vtilde * np.sin(stilde)

        self.state = np.array(
            [zprime, yprime, thetaprime, ztildeprime, ytildeprime, thetatildeprime]
        )

        terminated = self.state not in self.observation_space

        reward = np.exp(-(np.linalg.norm(self.state) ** 2))

        return self.state, reward, terminated, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(
            low=-self.observation_space.high, high=self.observation_space.high
        )
        # return self.state, {}
        return self.state

    def render(self):
        gym.logger.warn(
            "You are calling render method without specifying any render mode. "
            "You can specify the render_mode at initialization, "
            f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
        )
        return

    def close(self):
        return

    @staticmethod
    def _preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            state = np.expand_dims(state, 0)

        x1 = state[..., 0]
        x2 = state[..., 1]
        x3 = state[..., 2]
        x4 = state[..., 3]
        x5 = state[..., 4]
        x6 = state[..., 5]

        x1tilde = (x4 - x1) * np.cos(x3) + (x5 - x2) * np.sin(x3)
        x2tilde = -(x4 - x1) * np.sin(x3) + (x5 - x2) * np.cos(x3)
        x3tilde = x6 - x3

        xtilde = np.stack((x1tilde, x2tilde, x3tilde), axis=-1)
        if d1:
            xtilde = xtilde.squeeze()
        return xtilde

    @staticmethod
    def _preprocess_state_torch(state):
        assert isinstance(state, torch.Tensor)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            state = state.unsqueeze(0)

        x1 = state[..., 0]
        x2 = state[..., 1]
        x3 = state[..., 2]
        x4 = state[..., 3]
        x5 = state[..., 4]
        x6 = state[..., 5]

        x1tilde = (x4 - x1) * torch.cos(x3) + (x5 - x2) * torch.sin(x3)
        x2tilde = -(x4 - x1) * torch.sin(x3) + (x5 - x2) * torch.cos(x3)
        x3tilde = x6 - x3

        xtilde = torch.stack((x1tilde, x2tilde, x3tilde), axis=-1)
        if d1:
            xtilde = xtilde.squeeze()
        return xtilde

    # Symmetry projection
    @staticmethod
    def preprocess_fn(state):
        if isinstance(state, np.ndarray):
            return CarEnv._preprocess_state_np(state)
        if isinstance(state, torch.Tensor):
            return CarEnv._preprocess_state_torch(state)
        raise ValueError("Invalid state type (must be np.ndarray or torch.Tensor).")
