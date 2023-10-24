from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
import numpy as np

class RewardAssymetricInvertedPendulum(InvertedPendulumEnv):
    goal_x = 10.0
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, _reward, done, info = super().step(action)
        b = 100
        c = 1.03
        diff = obs[0] - self.goal_x
        reward = b/(np.power(c, diff) + np.power(c, -diff))
        return obs, reward, done, info
