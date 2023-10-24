import d3rlpy
from d3rlpy.algos import COMBO
from sklearn.model_selection import train_test_split
#import gymnasium as gym
import gym
from gym.wrappers import TransformObservation
import numpy as np
import encoders
import os
import json

seed = 1
d3rlpy.seed(seed)
use_gpu = True
# prepare environment
#env = gym.make("InvertedPendulum-v2")
#eval_env = gym.make("InvertedPendulum-v2")
env = gym.make("Reacher-v2")
eval_env = gym.make("Reacher-v2")
env.reset(seed=seed)
eval_env.reset(seed=seed)

def observation_edit1(obs):
    new_obs = np.zeros(8)
    new_obs[0] = np.arctan2(obs[2], obs[0])
    new_obs[1] = np.arctan2(obs[3], obs[1])
    new_obs[2:] = obs[4:-1]
    return new_obs

env1 = TransformObservation(env, observation_edit1)
env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype= np.float64 )
print(env1.reset(seed=seed))

eval_env1 = TransformObservation(eval_env, observation_edit1)
eval_env1.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype= np.float64 )
print(env1.reset(seed=seed))

trained_policy = d3rlpy.algos.SAC()
trained_policy.build_with_env(env1)
trained_policy.load_model('d3rlpy_logs/exp_6_SAC_reacher_20231024131731/model_200000.pt')

scorer = d3rlpy.metrics.scorer.evaluate_on_environment(env1, render=True)
mean_episode_return = scorer(trained_policy)