import gym
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv # gym 23.1 gives InvertedPendulum v2
# from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
import numpy as np

class RewardAssymetricInvertedPendulum(InvertedPendulumEnv):
    goal_x = 1.0
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, _reward, done, info = super().step(action)
        diff = obs[0] - self.goal_x
        b = 2
        c = 1.5
        reward = b/(np.power(c, diff) + np.power(c, -diff))
        # reward = -diff**2

        # notdone = np.isfinite(obs).all() and (np.abs(obs[1]) <= 0.2) # Changed angle threshold to 0.75rad from 0.2rad
        # done = not notdone
        return obs, reward, done, info
